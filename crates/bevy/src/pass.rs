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

use std::collections::{BTreeMap, HashMap, HashSet};

use bevy_ecs::{
    change_detection::MaybeLocation,
    component::ComponentId,
    schedule::{
        InternedSystemSet, NodeId, ScheduleBuildPass, ScheduleGraph, SystemKey, SystemSetKey,
    },
    system::{IntoSystem, System},
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
}

/// Schedule build pass that configures submission sets for GPU command encoding.
///
/// This pass is automatically applied to the [`PostUpdate`](bevy::app::PostUpdate) schedule
/// when submission sets are registered.
/// It transforms system sets into sequences that share command encoding state.
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
/// 3. Adds ordering edges so all render-set work completes before non-render-set work
#[derive(Debug, Default)]
pub(super) struct SubmissionSetsPass {
    /// Maps submission set to its associated queue component ID.
    pub(crate) submission_sets_to_queue:
        HashMap<InternedSystemSet, (ComponentId, SubmissionSetConfig)>,
    /// Maps submission set to its prelude/submission meta-systems.
    submission_sets_to_meta_systems: HashMap<SystemSetKey, RenderSetMetaSystems>,
    /// Maps render sets to its config system
    pub(crate) render_sets_to_systems: HashMap<InternedSystemSet, SystemKey>,
    /// Maps render sets to its ending system (added during schedule build)
    render_sets_to_ending_systems: BTreeMap<SystemSetKey, SystemKey>,
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
        systems: &mut Vec<SystemKey>,
        world: &mut World,
        graph: &mut ScheduleGraph,
    ) {
        let interned_set = graph.system_sets.get(set).unwrap();

        // Handle render sets - add ending system
        if self.render_sets_to_systems.contains_key(&interned_set) {
            let ending = add_system(graph, world, super::system::render_set_ending_system);
            self.render_sets_to_ending_systems.insert(set, ending);
            systems.push(ending);
        }

        if !self.submission_sets_to_queue.contains_key(&interned_set) {
            return; // not a submission set.
        };

        if systems.is_empty() {
            return; // Skip empty submission sets
        }

        let submission = add_system(graph, world, super::system::submission_system);
        let prelude = add_system(graph, world, super::system::prelude_system);
        self.submission_sets_to_meta_systems.insert(
            set,
            RenderSetMetaSystems {
                prelude,
                submission,
            },
        );
        systems.push(submission);
        systems.push(prelude);
    }

    fn collapse_set(
        &mut self,
        set: bevy_ecs::schedule::SystemSetKey,
        systems: &[bevy_ecs::schedule::SystemKey],
        world: &mut World,
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
        if let Some((queue_component_id, config)) = self.submission_sets_to_queue.get(&interned_set)
        {
            if systems.is_empty() {
                return Vec::new().into_iter();
            }
            let device = world.resource::<Device>().clone();
            let queue_family_index = unsafe {
                world
                    .get_resource_by_id(*queue_component_id)
                    .unwrap()
                    .deref::<SharedQueue>()
                    .family_index()
            };
            let shared_state_component_id = {
                // create shared state
                let shared_state_component_id = world.register_component_with_descriptor(
                    bevy_ecs::component::ComponentDescriptor::new_resource::<RenderSetSharedState>(
                    ),
                );
                bevy_ptr::OwningPtr::make(
                    RenderSetSharedState::new(
                        device,
                        queue_family_index,
                        format!("{interned_set:?}"),
                        config.debug_color,
                    ),
                    |ptr| unsafe {
                        // SAFETY: component_id was just initialized and corresponds to resource of type R.
                        world.insert_resource_by_id(
                            shared_state_component_id,
                            ptr,
                            MaybeLocation::caller(),
                        );
                    },
                );
                shared_state_component_id
            };
            let meta_systems = self.submission_sets_to_meta_systems.get(&set).unwrap();

            let mut queue_config = QueueConfig(*queue_component_id);
            let mut shared_config = RenderSetSharedStateConfig::new(shared_state_component_id);
            for system_node in systems.iter() {
                let system = graph.systems.get_mut(*system_node).unwrap();
                system.configurate(&mut shared_config);
                system
                    .access
                    .add_unfiltered_resource_write(shared_state_component_id);
            }

            let submission_system = graph.systems.get_mut(meta_systems.submission).unwrap();
            submission_system.configurate(&mut queue_config);
            submission_system
                .access
                .add_unfiltered_resource_write(*queue_component_id);

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
        dependency_flattened: &mut bevy_ecs::schedule::graph::DiGraph<SystemKey>,
    ) -> Result<(), bevy_ecs::schedule::ScheduleBuildError> {
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

        let hierarchy = graph.hierarchy().graph();

        // Collect render set keys for quick lookup
        let render_set_keys: HashSet<SystemSetKey> = self
            .render_sets_to_systems
            .keys()
            .filter_map(|interned| set_key_map.get(interned).copied())
            .collect();

        // --- Validation: each render set must be in exactly one submission set ---
        for (&render_set_interned, _) in &self.render_sets_to_systems {
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
                    if let Some(ancestor_interned) = graph.system_sets.get(ancestor_key) {
                        if self
                            .submission_sets_to_queue
                            .contains_key(&ancestor_interned)
                        {
                            count += 1;
                        }
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
        }

        // --- Optimization: minimize render/compute transitions ---
        // On tile-based GPUs, switching between render and compute is expensive,
        // and the CommandEncoder auto-ends the render pass when non-rendering
        // workload is encoded. We group same-type systems into stages using a
        // modified Kahn's algorithm that prefers render systems first, then add
        // edges between adjacent stages to enforce the grouping.
        for (&submission_set_interned, _) in &self.submission_sets_to_queue {
            let Some(&submission_set_key) = set_key_map.get(&submission_set_interned) else {
                continue;
            };
            let Some(meta) = self
                .submission_sets_to_meta_systems
                .get(&submission_set_key)
            else {
                continue;
            };

            let meta_set: HashSet<SystemKey> =
                [meta.prelude, meta.submission].into_iter().collect();

            let mut render_systems = Vec::new();
            let mut nonrender_systems = Vec::new();

            collect_descendant_systems(
                hierarchy,
                NodeId::Set(submission_set_key),
                &render_set_keys,
                false,
                &meta_set,
                &mut render_systems,
                &mut nonrender_systems,
            );

            if render_systems.is_empty() || nonrender_systems.is_empty() {
                continue;
            }

            let render_set: HashSet<SystemKey> = render_systems.iter().copied().collect();
            let all_systems: HashSet<SystemKey> = render_systems
                .iter()
                .chain(nonrender_systems.iter())
                .copied()
                .collect();

            // Build subgraph: in-degrees and adjacency for just these systems
            let mut in_degree: HashMap<SystemKey, usize> =
                all_systems.iter().map(|&s| (s, 0)).collect();
            let mut successors: HashMap<SystemKey, Vec<SystemKey>> =
                all_systems.iter().map(|&s| (s, Vec::new())).collect();

            for &sys in &all_systems {
                for neighbor in dependency_flattened
                    .neighbors_directed(sys, bevy_ecs::schedule::graph::Direction::Outgoing)
                {
                    if all_systems.contains(&neighbor) {
                        successors.get_mut(&sys).unwrap().push(neighbor);
                        *in_degree.get_mut(&neighbor).unwrap() += 1;
                    }
                }
            }

            // Modified Kahn's algorithm: prefer render systems to minimize transitions
            let mut ready_render = Vec::new();
            let mut ready_nonrender = Vec::new();

            // Start from systems with no incoming edges
            for (&sys, &deg) in &in_degree {
                if deg == 0 {
                    if render_set.contains(&sys) {
                        ready_render.push(sys);
                    } else {
                        ready_nonrender.push(sys);
                    }
                }
            }

            let mut current_is_render = !ready_render.is_empty();
            let mut stages: Vec<Vec<SystemKey>> = Vec::new();
            stages.push(Vec::new());

            while !ready_render.is_empty() || !ready_nonrender.is_empty() {
                let queue = if current_is_render && !ready_render.is_empty() {
                    &mut ready_render
                } else if !current_is_render && !ready_nonrender.is_empty() {
                    &mut ready_nonrender
                } else {
                    current_is_render = !current_is_render;
                    stages.push(Vec::new());
                    if current_is_render {
                        &mut ready_render
                    } else {
                        &mut ready_nonrender
                    }
                };

                let sys = queue.pop().unwrap();
                stages.last_mut().unwrap().push(sys);

                for &successor in &successors[&sys] {
                    let deg = in_degree.get_mut(&successor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        if render_set.contains(&successor) {
                            ready_render.push(successor);
                        } else {
                            ready_nonrender.push(successor);
                        }
                    }
                }
            }

            // Add edges between adjacent stages
            for i in 0..stages.len() - 1 {
                for &from in &stages[i] {
                    for &to in &stages[i + 1] {
                        dependency_flattened.add_edge(from, to);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Recursively walks the hierarchy graph from `node`, collecting descendant systems
/// into either `render_systems` or `nonrender_systems` based on whether they are
/// inside a render set.
fn collect_descendant_systems(
    hierarchy: &bevy_ecs::schedule::graph::DiGraph<NodeId>,
    node: NodeId,
    render_set_keys: &HashSet<SystemSetKey>,
    in_render_set: bool,
    meta_systems: &HashSet<SystemKey>,
    render_systems: &mut Vec<SystemKey>,
    nonrender_systems: &mut Vec<SystemKey>,
) {
    for child in hierarchy.neighbors_directed(node, bevy_ecs::schedule::graph::Direction::Outgoing)
    {
        match child {
            NodeId::System(sys) => {
                if meta_systems.contains(&sys) {
                    continue;
                }
                if in_render_set {
                    render_systems.push(sys);
                } else {
                    nonrender_systems.push(sys);
                }
            }
            NodeId::Set(set_key) => {
                let child_in_render = in_render_set || render_set_keys.contains(&set_key);
                collect_descendant_systems(
                    hierarchy,
                    child,
                    render_set_keys,
                    child_in_render,
                    meta_systems,
                    render_systems,
                    nonrender_systems,
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
    let access = system.initialize(world);

    let id = graph.systems.insert(Box::new(system), Vec::new());

    // ignore ambiguities with auto sync points
    // They aren't under user control, so no one should know or care.
    graph.ambiguous_with_all.insert(id.into());
    graph.systems.get_mut(id).unwrap().access.extend(access);

    id
}
