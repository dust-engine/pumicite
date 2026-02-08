//! Render set schedule build pass implementation.
//!
//! This module provides [`RenderSetsPass`], a custom Bevy schedule build pass that
//! transforms render sets into a sequence of systems with shared command encoding state.
//!
//! By default, this render pass is added to the [`PostUpdate`](bevy::app::PostUpdate)
//! schedule, in the hope to maximize overlap between rendering and other work that the
//! app might have to do in this schedule. As such, you should add all your render systems
//! to the [`PostUpdate`](bevy::app::PostUpdate) schedule.
//!
//! # Implementation
//!
//! When you register a render set via [`PumiciteApp::add_submission_set`](crate::PumiciteApp::add_submission_set),
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
//! This module is an implementation detail. Users should interact with render sets
//! through [`PumiciteApp::add_submission_set`](crate::PumiciteApp::add_submission_set) and
//! [`RenderState`](crate::RenderState). This gives us the ability to fine-tune the
//! scheduling of render systems, , the actual scheduling of the render systems
//!

use std::collections::HashMap;

use bevy_ecs::{
    change_detection::MaybeLocation,
    component::ComponentId,
    schedule::{
        InternedSystemSet, NodeId, ScheduleBuildPass, ScheduleGraph, SystemKey, SystemSetKey,
    },
    system::{IntoSystem, System},
    world::World,
};
use either::Either;

use crate::queue::{QueueConfig, SharedQueue};
use pumicite::Device;

use super::system::{RenderSetSharedState, RenderSetSharedStateConfig};

#[derive(Debug)]
struct RenderSetMetaSystems {
    prelude: SystemKey,
    submission: SystemKey,
}

/// Schedule build pass that configures render sets for GPU command encoding.
///
/// This pass is automatically applied to the [`PostUpdate`](bevy::app::PostUpdate) schedule
/// when render sets are registered.
/// It transforms system sets into sequences that share command encoding state.
///
/// # Internal Implementation
///
/// For each registered render set, this pass:
/// 1. Inserts a prelude system that begins command recording
/// 2. Configures member systems to share the command encoder
/// 3. Inserts a submission system that ends recording and submits
#[derive(Debug, Default)]
pub(super) struct SubmissionSetsPass {
    /// Maps render set to its associated queue component ID.
    pub(crate) submission_sets_to_queue: HashMap<InternedSystemSet, ComponentId>,
    /// Maps render set to its prelude/submission meta-systems.
    render_sets_to_meta_systems: HashMap<SystemSetKey, RenderSetMetaSystems>,
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
        world: &mut World,
        graph: &mut ScheduleGraph,
    ) -> impl Iterator<Item = SystemKey> {
        let interned_set = graph.system_sets.get(set).unwrap();
        if !self.submission_sets_to_queue.contains_key(&interned_set) {
            return Either::Left(std::iter::empty()); // not a system set.
        };

        let submission = add_system(graph, world, super::system::submission_system);
        let prelude = add_system(graph, world, super::system::prelude_system);
        self.render_sets_to_meta_systems.insert(
            set,
            RenderSetMetaSystems {
                prelude,
                submission,
            },
        );
        Either::Right([submission, prelude].into_iter())
    }

    fn collapse_set(
        &mut self,
        set: bevy_ecs::schedule::SystemSetKey,
        systems: &[bevy_ecs::schedule::SystemKey],
        world: &mut World,
        graph: &mut bevy_ecs::schedule::ScheduleGraph,
        dependency_flattened: &bevy_ecs::schedule::graph::DiGraph<NodeId>,
    ) -> impl Iterator<Item = (bevy_ecs::schedule::NodeId, bevy_ecs::schedule::NodeId)> {
        let interned_set = graph.system_sets.get(set).unwrap();
        let Some(&queue_component_id) = self.submission_sets_to_queue.get(&interned_set) else {
            return Either::Left(std::iter::empty()); // not a system set.
        };

        let device = world.resource::<Device>().clone();
        let queue_family_index = unsafe {
            world
                .get_resource_by_id(queue_component_id)
                .unwrap()
                .deref::<SharedQueue>()
                .family_index()
        };
        let shared_state_component_id = {
            // create shared state
            let shared_state_component_id = world.register_component_with_descriptor(
                bevy_ecs::component::ComponentDescriptor::new_resource::<RenderSetSharedState>(),
            );
            bevy_ptr::OwningPtr::make(
                RenderSetSharedState::new(device, queue_family_index, format!("{interned_set:?}")),
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
        let meta_systems = self.render_sets_to_meta_systems.get(&set).unwrap();

        let mut queue_config = QueueConfig(queue_component_id);
        let mut shared_config = RenderSetSharedStateConfig::new(shared_state_component_id);
        for system_node in systems.iter() {
            let system = graph.systems.get_mut(*system_node).unwrap();
            system.configurate(&mut shared_config);
            system
                .access
                .add_unfiltered_resource_write(shared_state_component_id);
            println!("Configured one system named {}", system.name());
        }

        let submission_system = graph.systems.get_mut(meta_systems.submission).unwrap();
        submission_system.configurate(&mut queue_config);
        submission_system
            .access
            .add_unfiltered_resource_write(queue_component_id);

        let user_systems = systems
            .iter()
            .copied()
            .filter(|&x| x != meta_systems.prelude && x != meta_systems.submission)
            .collect::<Vec<_>>();

        Either::Right(
            user_systems
                .into_iter()
                .flat_map(move |system| {
                    [
                        (meta_systems.prelude.into(), system.into()),
                        (system.into(), meta_systems.submission.into()),
                    ]
                })
                .chain(
                    dependency_flattened
                        .neighbors_directed(
                            set.into(),
                            bevy_ecs::schedule::graph::Direction::Incoming,
                        )
                        .map(move |parent| (parent, meta_systems.prelude.into())),
                )
                .chain(
                    dependency_flattened
                        .neighbors_directed(
                            set.into(),
                            bevy_ecs::schedule::graph::Direction::Outgoing,
                        )
                        .map(move |child| (meta_systems.submission.into(), child)),
                ),
        )
    }

    fn build(
        &mut self,
        _world: &mut bevy_ecs::world::World,
        _graph: &mut bevy_ecs::schedule::ScheduleGraph,
        _dependency_flattened: &mut bevy_ecs::schedule::graph::DiGraph<SystemKey>,
    ) -> Result<(), bevy_ecs::schedule::ScheduleBuildError> {
        Ok(())
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
