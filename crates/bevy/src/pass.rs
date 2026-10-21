//! Submission set schedule build pass implementation.
//!
//! This module provides [`RenderSetsPass`], a custom Bevy schedule build pass that
//! transforms submission sets into a sequence of systems with shared command encoding state.
//!
//! By default, this render pass is added to the [`PostUpdate`](bevy::app::PostUpdate)
//! schedule, in the hope to maximize overlap between rendering and other work that the
//! app might have to do in this schedule. As such, you should add all your render systems
//! to the [`PostUpdate`](bevy::app::PostUpdate) schedule.
//!
//! # Implementation
//!
//! When you register a submission set via [`PumiciteApp::add_submission_set`](crate::PumiciteApp::add_submission_set),
//! this build pass:
//!
//! 1. **Maps** the system set to include `prelude` and `submission` systems
//! 2. **Creates** shared state ([`RenderSetSharedState`](crate::system::RenderSetSharedState))
//! 3. **Configures** all systems in the set to use the shared command encoder
//! 4. **Adds dependencies** so `prelude`` runs before all systems in the set, and
//!     `submission` runs after all systems in the set.
//!
//! # Internal Use Only
//!
//! This module is an implementation detail. Users should interact with submission sets
//! through [`PumiciteApp::add_submission_set`](crate::PumiciteApp::add_submission_set) and
//! [`SubmissionState`](crate::SubmissionState). This gives us the ability to fine-tune the
//! scheduling of render systems, , the actual scheduling of the render systems
//!

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use bevy_ecs::{
    change_detection::MaybeLocation,
    component::ComponentId,
    query::FilteredAccess,
    resource::IS_RESOURCE,
    schedule::{
        InternedSystemSet, NodeId, ScheduleBuildPass, ScheduleGraph, SystemKey, SystemSetKey,
    },
    system::{IntoSystem, System, SystemAccess, SystemParamBuilder},
    world::World,
};

use crate::{
    plugin::SubmissionSetConfig,
    queue::{QueueConfig, SharedQueue},
};
use pumicite::Device;

use super::system::{RenderSetSharedState, RenderSetSharedStateConfig};

#[derive(Debug)]
struct RenderSetMetaSystems {
    prelude: SystemKey,
    submission: SystemKey,
    shared_state: ComponentId,
}

/// Schedule build pass that configures submission sets for GPU command encoding.
///
/// This pass is automatically applied to the [`PostUpdate`](bevy::app::PostUpdate) schedule
/// when submission sets are registered.
/// It transforms system sets into sequences that share command encoding state.
///
/// It must run before [`AutoInsertApplyDeferredPass`](bevy_ecs::schedule::passes::AutoInsertApplyDeferredPass),
/// so that the orderings it sees are the user's, not ones created by shared sync points.
///
/// # Internal Implementation
///
/// For each registered submission set, this pass:
/// 1. Inserts a prelude system that begins command recording
/// 2. Configures member systems to share the command encoder
/// 3. Inserts a submission system that ends recording and submits
///
/// For each registered render set, this pass:
/// 1. Ensures the config system (which begins the render pass) runs before other systems
/// 2. Validates the render set belongs to exactly one submission set
/// 3. Runs it as one block: the submission set's other systems that record commands,
///    including other render sets, run entirely before or after it
/// 4. Groups it with the submission set's other systems into alternating render and
///    compute stages, to minimize transitions between them
#[derive(Debug, Default)]
pub(super) struct SubmissionSetsPass {
    /// Maps submission set to its associated queue component ID.
    pub(crate) submission_sets_to_queue:
        HashMap<InternedSystemSet, (ComponentId, SubmissionSetConfig)>,
    /// Maps submission set to its prelude/submission meta-systems and shared state.
    /// These are reused when the schedule is rebuilt.
    submission_sets_to_meta_systems: HashMap<InternedSystemSet, RenderSetMetaSystems>,
    /// Maps render sets to its config system
    pub(crate) render_sets_to_systems: HashMap<InternedSystemSet, SystemKey>,
    /// Maps render sets to its ending system (added during schedule build, and reused
    /// when the schedule is rebuilt)
    render_sets_to_ending_systems: BTreeMap<SystemSetKey, SystemKey>,
    /// Systems in submission sets that have a `SubmissionState` param, and so record into
    /// the set's command encoder (collected during schedule build)
    systems_using_encoder: HashSet<SystemKey>,
}

impl ScheduleBuildPass for SubmissionSetsPass {
    type EdgeOptions = ();

    fn add_dependency(
        &mut self,
        _from: bevy_ecs::schedule::NodeId,
        _to: bevy_ecs::schedule::NodeId,
        _options: Option<&Self::EdgeOptions>,
    ) {
    }

    fn map_set_to_systems(
        &mut self,
        set: bevy_ecs::schedule::SystemSetKey,
        systems: &mut indexmap::IndexSet<SystemKey, bevy_platform::hash::FixedHasher>,
        world: &mut World,
        graph: &mut ScheduleGraph,
    ) {
        let interned_set = graph.system_sets.get(set).unwrap();

        // Handle render sets - add ending system
        if let Some(&config_system) = self.render_sets_to_systems.get(&interned_set) {
            if graph.systems.get(config_system).is_none() {
                // The config system was removed from the schedule, so this is no longer
                // a render set. `add_render_set` registers it again.
                self.render_sets_to_systems.remove(&interned_set);
                self.render_sets_to_ending_systems.remove(&set);
            } else {
                // Reuse the ending system from an earlier build, unless it was removed.
                let ending = match self.render_sets_to_ending_systems.get(&set) {
                    Some(&ending) if graph.systems.get(ending).is_some() => ending,
                    _ => {
                        let ending =
                            add_system(graph, world, super::system::render_set_ending_system);
                        self.render_sets_to_ending_systems.insert(set, ending);
                        ending
                    }
                };
                systems.insert(ending);
            }
        }

        let Some((queue_component_id, config)) = self.submission_sets_to_queue.get(&interned_set)
        else {
            return; // not a submission set.
        };

        // Meta systems left out of an empty set don't run, and are reused if the set
        // gets systems again.
        if systems.is_empty() {
            return; // Skip empty submission sets
        }

        // Reuse the meta systems from an earlier build, so that rebuilding the schedule
        // doesn't create a second shared state for this set.
        let existing = self.submission_sets_to_meta_systems.get(&interned_set);
        if let Some(meta_systems) = existing
            && graph.systems.get(meta_systems.prelude).is_some()
            && graph.systems.get(meta_systems.submission).is_some()
        {
            systems.insert(meta_systems.submission);
            systems.insert(meta_systems.prelude);
            return;
        }

        // The meta systems are new, or were removed from the schedule along with the
        // set's other systems. In the latter case, keep using the existing shared state.
        let shared_state_component_id = match existing {
            Some(meta_systems) => meta_systems.shared_state,
            None => {
                let device = world.resource::<Device>().clone();
                let queue_family_index = unsafe {
                    world
                        .get_resource_by_id(*queue_component_id)
                        .unwrap()
                        .deref::<SharedQueue>()
                        .family_index()
                };
                // create shared state
                let shared_state_component_id = world.register_component_with_descriptor(
                    bevy_ecs::component::ComponentDescriptor::new::<RenderSetSharedState>(),
                );
                bevy_ptr::OwningPtr::make(
                    RenderSetSharedState::new(
                        device,
                        queue_family_index,
                        format!("{interned_set:?}"),
                        config.debug_color,
                    )
                    .unwrap(),
                    |ptr| unsafe {
                        if world
                            .resource_entities()
                            .get(shared_state_component_id)
                            .is_none()
                        {
                            world.spawn(bevy_ecs::resource::IsResource::new(
                                shared_state_component_id,
                            ));
                        }
                        // SAFETY: forwarded from this function's contract.
                        world.insert_resource_by_id(
                            shared_state_component_id,
                            ptr,
                            MaybeLocation::caller(),
                        );
                    },
                );
                shared_state_component_id
            }
        };

        // These are built with their param state already set, so their access
        // is declared by the params themselves.
        let submission = add_system(
            graph,
            world,
            (
                RenderSetSharedStateConfig::new(shared_state_component_id),
                QueueConfig(*queue_component_id),
            )
                .build_system(super::system::submission_system),
        );
        let prelude = add_system(
            graph,
            world,
            (RenderSetSharedStateConfig::new(shared_state_component_id),)
                .build_system(super::system::prelude_system),
        );
        self.submission_sets_to_meta_systems.insert(
            interned_set,
            RenderSetMetaSystems {
                prelude,
                submission,
                shared_state: shared_state_component_id,
            },
        );
        systems.insert(submission);
        systems.insert(prelude);
    }

    fn collapse_set(
        &mut self,
        set: bevy_ecs::schedule::SystemSetKey,
        systems: &indexmap::IndexSet<SystemKey, bevy_platform::hash::FixedHasher>,
        _world: &mut World,
        graph: &mut bevy_ecs::schedule::ScheduleGraph,
        dependency_flattened: &bevy_ecs::schedule::graph::DiGraph<NodeId>,
    ) -> impl Iterator<Item = (bevy_ecs::schedule::NodeId, bevy_ecs::schedule::NodeId)> {
        let mut edges: Vec<(NodeId, NodeId)> = Vec::new();
        let interned_set = graph.system_sets.get(set).unwrap();

        // --- Handle render sets ---
        if let Some(&config_system) = self.render_sets_to_systems.get(&interned_set) {
            let ending_system = self
                .render_sets_to_ending_systems
                .get(&set)
                .copied()
                .unwrap();
            // Config system must run before all other systems in this render set.
            // The config system is responsible for beginning the render pass.
            // Ending system must run after all other systems in this render set.
            // The ending system is responsible for ending the render pass.
            for &system in systems {
                if system != config_system && system != ending_system {
                    edges.push((config_system.into(), system.into()));
                    edges.push((system.into(), ending_system.into()));
                }
            }
            // Ensure config → ending even when there are no user systems
            edges.push((config_system.into(), ending_system.into()));
        }

        // --- Handle submission sets ---
        if self.submission_sets_to_queue.contains_key(&interned_set) {
            if systems.is_empty() {
                return Vec::new().into_iter();
            }
            let meta_systems = self
                .submission_sets_to_meta_systems
                .get(&interned_set)
                .unwrap();

            let mut shared_config = RenderSetSharedStateConfig::new(meta_systems.shared_state);
            for system_node in systems.iter() {
                let system = graph.systems.get_mut(*system_node).unwrap();
                shared_config.used = false;
                system.configurate(&mut shared_config);
                if !shared_config.used {
                    continue; // It doesn't record into the shared command encoder.
                }
                self.systems_using_encoder.insert(*system_node);
                // On the build that adds them, systems added by build passes are initialized
                // again after the passes run, which replaces this access with what their
                // params declare. For every other system, the access added here sticks.
                let mut shared_state_write = FilteredAccess::default();
                shared_state_write.add_write(meta_systems.shared_state);
                shared_state_write.and_with(IS_RESOURCE);
                let access = system.access_mut();
                // Systems already have this access when the schedule is rebuilt.
                if let SystemAccess::Shared(access) = &*access
                    && access.filtered_accesses().contains(&shared_state_write)
                {
                    continue;
                }
                if access.try_add(shared_state_write).is_err() {
                    panic!(
                        "error[B0002]: SubmissionState in system {} conflicts with a previous \
                         system parameter. Consider removing the conflicting access, or using \
                         `Without<IsResource>` to make queries disjoint from resources. See: \
                         https://bevy.org/learn/errors/b0002",
                        system.name()
                    );
                }
            }

            let user_systems = systems
                .iter()
                .copied()
                .filter(|&x| x != meta_systems.prelude && x != meta_systems.submission)
                .collect::<Vec<_>>();

            // prelude → each user system, each user system → submission
            for &system in &user_systems {
                edges.push((meta_systems.prelude.into(), system.into()));
                edges.push((system.into(), meta_systems.submission.into()));
            }

            // Forward parent deps → prelude
            for parent in dependency_flattened
                .neighbors_directed(set.into(), bevy_ecs::schedule::graph::Direction::Incoming)
            {
                edges.push((parent, meta_systems.prelude.into()));
            }
            // Forward submission → child deps
            for child in dependency_flattened
                .neighbors_directed(set.into(), bevy_ecs::schedule::graph::Direction::Outgoing)
            {
                edges.push((meta_systems.submission.into(), child));
            }
        }

        edges.into_iter()
    }

    fn build(
        &mut self,
        _world: &mut bevy_ecs::world::World,
        graph: &mut bevy_ecs::schedule::ScheduleGraph,
        mut dependency_flattened: bevy_ecs::schedule::FlattenedDependencies<'_>,
    ) -> Result<(), bevy_ecs::schedule::ScheduleBuildError> {
        // Collected again on every build.
        let systems_using_encoder = std::mem::take(&mut self.systems_using_encoder);

        if self.render_sets_to_systems.is_empty() {
            return Ok(());
        }

        // Build reverse map: InternedSystemSet → SystemSetKey
        // TODO, in bevy, directly expose the reverse map.
        let set_key_map: HashMap<InternedSystemSet, SystemSetKey> = graph
            .system_sets
            .iter()
            .map(|(key, interned, _)| (interned, key))
            .collect();

        // Forget render sets that were removed from the schedule, along with their
        // config systems. `add_render_set` registers them again.
        self.render_sets_to_systems
            .retain(|render_set, _| set_key_map.contains_key(render_set));
        self.render_sets_to_ending_systems
            .retain(|&render_set_key, _| graph.system_sets.get(render_set_key).is_some());

        let hierarchy = graph.hierarchy().graph();

        // Render sets in each submission set, mapped to their config and ending systems.
        let mut render_sets_by_submission_set: BTreeMap<
            SystemSetKey,
            BTreeMap<SystemSetKey, (SystemKey, SystemKey)>,
        > = BTreeMap::new();

        // --- Validation: each render set must be in exactly one submission set ---
        for (&render_set_interned, &config_system) in &self.render_sets_to_systems {
            let render_set_key = set_key_map
                .get(&render_set_interned)
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "Render set {render_set_interned:?} was not found in the schedule graph."
                    )
                });

            // Walk ancestors in the hierarchy to find parent submission sets
            let mut count = 0usize;
            let mut submission_set_key = None;
            let mut to_visit: Vec<NodeId> = hierarchy
                .neighbors_directed(
                    NodeId::Set(render_set_key),
                    bevy_ecs::schedule::graph::Direction::Incoming,
                )
                .collect();
            let mut visited = HashSet::new();

            while let Some(ancestor) = to_visit.pop() {
                if !visited.insert(ancestor) {
                    continue;
                }
                if let NodeId::Set(ancestor_key) = ancestor {
                    if let Some(ancestor_interned) = graph.system_sets.get(ancestor_key)
                        && self
                            .submission_sets_to_queue
                            .contains_key(&ancestor_interned)
                    {
                        count += 1;
                        submission_set_key = Some(ancestor_key);
                    }
                    to_visit.extend(hierarchy.neighbors_directed(
                        ancestor,
                        bevy_ecs::schedule::graph::Direction::Incoming,
                    ));
                }
            }

            if count == 0 {
                panic!(
                    "Render set {render_set_interned:?} is not inside any submission set. \
                     Place it inside a submission set using `.in_set(your_submission_set)`."
                );
            }
            if count > 1 {
                panic!(
                    "Render set {render_set_interned:?} is inside {count} submission sets. \
                     A render set must belong to exactly one submission set."
                );
            }

            let ending_system = self
                .render_sets_to_ending_systems
                .get(&render_set_key)
                .copied()
                .unwrap();
            render_sets_by_submission_set
                .entry(submission_set_key.unwrap())
                .or_default()
                .insert(render_set_key, (config_system, ending_system));
        }

        // --- Ordering within submission sets ---
        // A submission set's systems share one command encoder, and only render work can
        // be recorded while a render pass is open. So each render set runs as one block:
        // the submission set's other systems, including other render sets, run entirely
        // before or after it.
        //
        // On tile-based GPUs, switching between render and compute is also expensive. So
        // the blocks and the other systems are grouped into alternating render and compute
        // stages using a modified Kahn's algorithm that prefers render work first. Edges
        // between adjacent stages enforce the grouping, and edges between consecutive
        // render sets in a render stage keep them from overlapping.
        for &submission_set_interned in self.submission_sets_to_queue.keys() {
            let Some(&submission_set_key) = set_key_map.get(&submission_set_interned) else {
                continue;
            };
            let Some(meta) = self
                .submission_sets_to_meta_systems
                .get(&submission_set_interned)
            else {
                continue;
            };
            let Some(render_sets) = render_sets_by_submission_set.get(&submission_set_key) else {
                continue; // Without render sets, everything is one compute stage.
            };

            let meta_set: HashSet<SystemKey> =
                [meta.prelude, meta.submission].into_iter().collect();

            // The submission set's systems, each with the render set it's in, if any.
            let mut systems = HashMap::new();
            collect_descendant_systems(
                hierarchy,
                NodeId::Set(submission_set_key),
                &self.render_sets_to_ending_systems,
                None,
                &meta_set,
                &mut systems,
            );
            // Systems outside render sets that don't record commands can run anywhere,
            // so treat them like systems outside the submission set.
            systems.retain(|sys, render_set| {
                render_set.is_some() || systems_using_encoder.contains(sys)
            });

            if render_sets.len() < 2 && !systems.values().any(Option::is_none) {
                continue; // A lone render set is already one block.
            }

            use bevy_ecs::schedule::graph::Direction::{Incoming, Outgoing};

            // A render set can only start once every other system of the submission set
            // that must run before any of its systems has been staged. Its systems all run
            // before its ending system, so those are the ancestors of its ending system.
            let mut blockers: BTreeMap<SystemSetKey, usize> = BTreeMap::new();
            let mut blocked_render_sets: HashMap<SystemKey, Vec<SystemSetKey>> = HashMap::new();
            for (&render_set, &(_, ending_system)) in render_sets {
                let mut count = 0;
                let mut visited = HashSet::new();
                let mut to_visit = vec![ending_system];
                while let Some(sys) = to_visit.pop() {
                    for prev in dependency_flattened.neighbors_directed(sys, Incoming) {
                        if !visited.insert(prev) {
                            continue;
                        }
                        to_visit.push(prev);
                        if let Some(&prev_render_set) = systems.get(&prev)
                            && prev_render_set != Some(render_set)
                        {
                            count += 1;
                            blocked_render_sets
                                .entry(prev)
                                .or_default()
                                .push(render_set);
                        }
                    }
                }
                blockers.insert(render_set, count);
            }

            // The algorithm runs over the whole graph, not just this set's systems, because
            // an ordering between two of them can pass through systems outside the set.
            // Systems outside the set join no stage, and may run inside a render set, as
            // they don't record into this set's command encoder. That includes the sync
            // points that `AutoInsertApplyDeferredPass` adds after this pass runs.
            let mut in_degree: HashMap<SystemKey, usize> = HashMap::new();
            let mut newly_ready = Vec::new();
            for sys in dependency_flattened.nodes() {
                let deg = dependency_flattened
                    .neighbors_directed(sys, Incoming)
                    .count();
                if deg == 0 {
                    newly_ready.push(sys);
                } else {
                    in_degree.insert(sys, deg);
                }
            }

            let mut ready_outside = Vec::new();
            let mut ready_compute = Vec::new();
            // Render sets whose config system is ready.
            let mut ready_render_sets = BTreeSet::new();
            // The render set being staged, and its ready systems.
            let mut open_render_set: Option<SystemSetKey> = None;
            let mut ready_in_open_render_set = Vec::new();
            // Ending system of the last render set staged in the current stage.
            let mut last_ending_system = None;
            let mut current_is_render = false;
            let mut stages: Vec<Vec<SystemKey>> = Vec::new();
            let mut render_set_edges = Vec::new();
            // Systems after the set's last one can't change its stages.
            let mut unstaged = systems.len();

            while unstaged > 0 {
                for sys in newly_ready.drain(..) {
                    match systems.get(&sys) {
                        None => ready_outside.push(sys),
                        Some(None) => ready_compute.push(sys),
                        Some(&Some(render_set)) => {
                            if render_sets[&render_set].0 == sys {
                                ready_render_sets.insert(render_set);
                            } else {
                                // Its config system has been staged, so its render set
                                // is the open one.
                                ready_in_open_render_set.push(sys);
                            }
                        }
                    }
                }

                // Systems outside the set go first, as they may be what holds back
                // a system of the current stage's kind.
                let sys = if let Some(sys) = ready_outside.pop() {
                    sys
                } else if let Some(render_set) = open_render_set {
                    // Finish the open render set before staging anything else.
                    let sys = ready_in_open_render_set
                        .pop()
                        .expect("a render set can't get blocked once it has started");
                    if sys == render_sets[&render_set].1 {
                        open_render_set = None;
                        last_ending_system = Some(sys);
                    }
                    stages.last_mut().unwrap().push(sys);
                    unstaged -= 1;
                    sys
                } else {
                    // Among render sets that can start, take the one with the smallest key.
                    let startable_render_set = ready_render_sets
                        .iter()
                        .copied()
                        .find(|render_set| blockers[render_set] == 0);
                    if startable_render_set.is_none() && ready_compute.is_empty() {
                        let not_started: Vec<InternedSystemSet> = blockers
                            .keys()
                            .map(|&render_set| graph.system_sets.get(render_set).unwrap())
                            .collect();
                        panic!(
                            "Render sets {not_started:?} in submission set \
                             {submission_set_interned:?} can't be scheduled: ordering \
                             constraints or shared systems put other systems of the \
                             submission set between their config and ending systems. Only \
                             render work can be recorded while a render pass is open, so each \
                             render set must run entirely before or after the submission \
                             set's other systems, including other render sets."
                        );
                    }
                    let stay = !stages.is_empty()
                        && if current_is_render {
                            startable_render_set.is_some()
                        } else {
                            !ready_compute.is_empty()
                        };
                    if !stay {
                        current_is_render = if stages.is_empty() {
                            startable_render_set.is_some()
                        } else {
                            !current_is_render
                        };
                        stages.push(Vec::new());
                        last_ending_system = None;
                    }
                    let sys = if current_is_render {
                        let render_set = startable_render_set.unwrap();
                        ready_render_sets.remove(&render_set);
                        blockers.remove(&render_set);
                        open_render_set = Some(render_set);
                        let config_system = render_sets[&render_set].0;
                        if let Some(ending_system) = last_ending_system {
                            render_set_edges.push((ending_system, config_system));
                        }
                        config_system
                    } else {
                        ready_compute.pop().unwrap()
                    };
                    stages.last_mut().unwrap().push(sys);
                    unstaged -= 1;
                    sys
                };

                if let Some(blocked) = blocked_render_sets.get(&sys) {
                    for render_set in blocked {
                        if let Some(count) = blockers.get_mut(render_set) {
                            *count -= 1;
                        }
                    }
                }
                for successor in dependency_flattened.neighbors_directed(sys, Outgoing) {
                    let deg = in_degree.get_mut(&successor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        newly_ready.push(successor);
                    }
                }
            }

            // Add edges between adjacent stages
            for pair in stages.windows(2) {
                for &from in &pair[0] {
                    for &to in &pair[1] {
                        dependency_flattened.add_edge(from, to);
                    }
                }
            }
            // Add edges between consecutive render sets in a render stage
            for (from, to) in render_set_edges {
                dependency_flattened.add_edge(from, to);
            }
        }

        Ok(())
    }
}

/// Recursively walks the hierarchy graph from `node`, collecting descendant systems
/// into `systems` along with the render set they are inside, if any.
fn collect_descendant_systems(
    hierarchy: &bevy_ecs::schedule::graph::DiGraph<NodeId>,
    node: NodeId,
    render_sets_to_ending_systems: &BTreeMap<SystemSetKey, SystemKey>,
    render_set: Option<SystemSetKey>,
    meta_systems: &HashSet<SystemKey>,
    systems: &mut HashMap<SystemKey, Option<SystemSetKey>>,
) {
    for child in hierarchy.neighbors_directed(node, bevy_ecs::schedule::graph::Direction::Outgoing)
    {
        match child {
            NodeId::System(sys) => {
                if meta_systems.contains(&sys) {
                    continue;
                }
                // A system reached both inside and outside a render set is render work.
                let entry = systems.entry(sys).or_default();
                if entry.is_none() {
                    *entry = render_set;
                }
            }
            NodeId::Set(set_key) => {
                let mut child_render_set = render_set;
                if let Some(&ending_system) = render_sets_to_ending_systems.get(&set_key) {
                    systems.insert(ending_system, Some(set_key));
                    child_render_set = Some(set_key);
                }
                collect_descendant_systems(
                    hierarchy,
                    child,
                    render_sets_to_ending_systems,
                    child_render_set,
                    meta_systems,
                    systems,
                );
            }
        }
    }
}

fn add_system<Marker, T: IntoSystem<(), (), Marker>>(
    graph: &mut ScheduleGraph,
    world: &mut World,
    system: T,
) -> SystemKey {
    let mut system: T::System = IntoSystem::into_system(system);
    // Initialize now so that `collapse_set` can call `configurate` on it. The schedule
    // initializes it again after the build passes run, and takes its access from that.
    system.initialize(world);

    let id = graph.systems.insert(Box::new(system), Vec::new());

    // ignore ambiguities with auto sync points
    // They aren't under user control, so no one should know or care.
    graph.ambiguous_with_all.insert(id.into());

    id
}
