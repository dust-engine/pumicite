//! Vulkan queue management and system parameters.
//!
//! This module provides the infrastructure for accessing Vulkan queues from Bevy systems.
//! It handles queue family selection, queue aliasing, and provides type-safe queue access
//! through marker types.
//!
//! # Vulkan Queue Families
//!
//! Vulkan devices expose one or more *queue families*, each supporting different operations:
//! - **Graphics** queues can execute rendering commands
//! - **Compute** queues can execute compute shaders
//! - **Transfer** queues can copy data between buffers/images
//!
//! Queues with the **Graphics** or **Compute** capabilities are guaranteed to also have
//! **Transfer** capability.
//!
//! Some families support multiple operation types (e.g., queue family 0 typically
//! supports all of graphics, compute and transfer).
//!
//! # Queue Types
//!
//! This module defines four queue marker types, each optimized for specific workloads:
//!
//! - [`RenderQueue`] - Primary graphics queue (priority 1.0)
//! - [`ComputeQueue`] - Synchronous compute operations (priority 1.0)
//! - [`TransferQueue`] - Data uploads/downloads (priority 0.1)
//! - [`AsyncComputeQueue`] - Background compute work (priority 0.1)
//!
//! # Queue Aliasing
//!
//! Not all GPUs provide dedicated queues for each type. When a dedicated queue isn't
//! available, Pumicite automatically *aliases* a compatible existing queue. For example,
//! if no dedicated transfer queue exists, [`TransferQueue`] might alias [`ComputeQueue`].
//!
//! Aliased queues share the same underlying Vulkan queue, so systems declaring access to
//! [`Queue<T>`] won't actually run in parallel. However, the API remains the same,
//! simplifying code that doesn't need to care about implementation specifics.
//!
//! # Accessing Queues in Systems
//!
//! Use the [`Queue<T>`] system parameter to access a specific queue:
//!
//! ```ignore
//! use bevy_pumicite::Queue;
//! use bevy_pumicite::queue::RenderQueue;
//!
//! fn submit_work(mut queue: Queue<RenderQueue>) {
//!     queue.submit(&mut command_buffer).unwrap();
//! }
//! ```
//!
//! For shared queue access across threads, use [`SharedQueue`] via [`QueueWorldExt::make_shared_queue`].

use std::{
    any::TypeId,
    collections::BTreeMap,
    ffi::CString,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use bevy_ecs::{
    change_detection::Tick,
    component::ComponentId,
    prelude::*,
    ptr::OwningPtr,
    query::FilteredAccess,
    resource::IS_RESOURCE,
    system::{SystemMeta, SystemParam, SystemParamBuilder},
    world::{Mut, unsafe_world_cell::UnsafeWorldCell},
};

use pumicite::{Device, device::DeviceQueueRef};
use pumicite::{ash::vk, debug::DebugObject};

/// Information about a registered queue.
///
/// Provides metadata about a queue's capabilities and configuration.
pub struct QueueInfo {
    /// Vulkan queue capability flags (e.g., `GRAPHICS`, `COMPUTE`, `TRANSFER`).
    pub flags: vk::QueueFlags,
    /// Index of the queue family this queue belongs to.
    pub family_index: u32,
    /// Queue priority (0.0 to 1.0).
    pub priority: f32,
    /// Human-readable name for debugging.
    pub name: &'static str,
}

#[derive(Resource, Default)]
pub(crate) struct QueueConfiguration {
    mapping: BTreeMap<TypeId, ComponentId>,
    queues: BTreeMap<ComponentId, QueueInfo>,
    queues_to_initialize: Vec<(ComponentId, DeviceQueueRef, &'static str)>,
}
impl QueueConfiguration {
    pub(crate) fn component_id_of_queue<Q: 'static>(&self) -> Option<ComponentId> {
        self.mapping.get(&std::any::TypeId::of::<Q>()).cloned()
    }
    pub fn queue_info<Q: 'static>(&self) -> Option<&QueueInfo> {
        let id = self.component_id_of_queue::<Q>()?;
        self.queues.get(&id)
    }
    pub fn register_queue<T: 'static>(
        &mut self,
        component_id: ComponentId,
        queue_ref: DeviceQueueRef,
        priority: f32,
        name: &'static str,
    ) {
        let original = self
            .mapping
            .insert(std::any::TypeId::of::<T>(), component_id);
        if original.is_some() {
            panic!("Queue {} has already been registered!", name);
        }
        self.queues.insert(
            component_id,
            QueueInfo {
                flags: queue_ref.flags(),
                family_index: queue_ref.family_index(),
                priority,
                name,
            },
        );
        self.queues_to_initialize
            .push((component_id, queue_ref, std::any::type_name::<T>()));
    }
    pub fn alias_queue<T: 'static>(
        &mut self,
        required_queue_flags: vk::QueueFlags,
        priority: f32,
    ) -> Option<&QueueInfo> {
        let (&component_id, info) = self
            .queues
            .iter()
            .filter(|(_, info)| info.flags.contains(required_queue_flags))
            .min_by_key(|(_, info)| {
                (
                    info.flags.as_raw().count_ones(),
                    ((info.priority - priority).abs() * 100000.0) as u64,
                )
            })?;
        let original = self
            .mapping
            .insert(std::any::TypeId::of::<T>(), component_id);
        if original.is_some() {
            panic!(
                "Queue type {} has already been registered!",
                std::any::type_name::<T>()
            );
        }
        Some(info)
    }
    pub fn init_queues(world: &mut World) {
        let mut this = world.resource_mut::<Self>();
        let queues_to_initialize = std::mem::take(&mut this.queues_to_initialize);
        let device = world.resource::<Device>().clone();
        for (component_id, queue_ref, name) in queues_to_initialize.into_iter() {
            let name = CString::new(name).unwrap();
            let queue = device.get_queue(queue_ref).with_name(name.as_c_str());
            OwningPtr::make(SharedQueue::new(queue), |ptr| unsafe {
                if world.resource_entities().get(component_id).is_none() {
                    world.spawn(bevy_ecs::resource::IsResource::new(component_id));
                }
                world.insert_resource_by_id(
                    component_id,
                    ptr,
                    bevy_ecs::change_detection::MaybeLocation::caller(),
                );
            });
        }
    }
}

/// A thread-safe wrapper around a Vulkan queue.
///
/// `SharedQueue` provides shared ownership of a queue via either [`Arc<Mutex<Queue>>`],
/// allowing multiple threads or async tasks to submit work to the same queue.
///
/// Created via [`QueueWorldExt::make_shared_queue`].
///
/// # When to Use
///
/// - **In systems**: Prefer [`Queue<T>`] as the bevy scheduler will guarantee exclusive
///   mutable access to the queue objects.
/// - **In async tasks**: Use `SharedQueue` when you need to submit from background tasks
///   or when multiple places need concurrent access.
#[derive(Clone, Resource)]
pub struct SharedQueue {
    inner: Arc<std::sync::Mutex<pumicite::Queue>>,
    family_index: u32,
}
impl SharedQueue {
    fn new(queue: pumicite::Queue) -> Self {
        Self {
            family_index: queue.family_index(),
            inner: Arc::new(std::sync::Mutex::new(queue)),
        }
    }
    pub fn family_index(&self) -> u32 {
        self.family_index
    }
}

impl Deref for SharedQueue {
    type Target = Arc<std::sync::Mutex<pumicite::Queue>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// System parameter for accessing a Vulkan queue.
///
/// `Queue<T>` provides mutable access to a specific queue identified by the marker
/// type `T`. It implements `Deref` and `DerefMut` to [`pumicite::Queue`], allowing
/// direct use of queue methods.
///
/// # Type Parameter
///
/// The type parameter `T` identifies which queue to access:
/// - [`RenderQueue`] - Graphics operations
/// - [`ComputeQueue`] - Compute shader dispatch
/// - [`TransferQueue`] - Data transfers
///
/// # Exclusive Access
///
/// Only one system can access a given queue at a time. If you need shared access
/// (e.g., from async tasks), use [`SharedQueue`] via [`QueueWorldExt::make_shared_queue`].
pub struct Queue<'a, T: 'static> {
    queue: &'a mut pumicite::Queue,
    _guard: Option<std::sync::MutexGuard<'a, pumicite::Queue>>,
    _marker: PhantomData<T>,
}
impl<'a, T: 'static> Deref for Queue<'a, T> {
    type Target = pumicite::Queue;

    fn deref(&self) -> &Self::Target {
        self.queue
    }
}
impl<'a, T: 'static> DerefMut for Queue<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.queue
    }
}

unsafe impl<'a, T: 'static> SystemParam for Queue<'a, T> {
    type State = ComponentId;

    type Item<'world, 'state> = Queue<'world, T>;

    fn init_state(world: &mut World) -> ComponentId {
        let config = world.resource::<QueueConfiguration>();

        config
            .component_id_of_queue::<T>()
            .expect("Please call app.init_device_queue")
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        _system_meta: &SystemMeta,
        world: UnsafeWorldCell<'world>,
        _change_tick: Tick,
    ) -> Result<Self::Item<'world, 'state>, bevy_ecs::system::SystemParamValidationError> {
        let shared_queue: Mut<'world, SharedQueue> =
            unsafe { world.get_resource_mut_by_id(*state).unwrap().with_type() };
        let shared_queue: &'world mut SharedQueue = shared_queue.into_inner();

        if Arc::get_mut(&mut shared_queue.inner).is_some() {
            // TODO: calling Arc::get_mut twice because of https://github.com/rust-lang/rust/issues/54663 and is_unique is not stable.
            Ok(Queue {
                queue: Arc::get_mut(&mut shared_queue.inner)
                    .unwrap()
                    .get_mut()
                    .unwrap(),
                _guard: None,
                _marker: PhantomData,
            })
        } else {
            let mut guard = shared_queue.inner.lock().unwrap();
            let queue: &mut pumicite::Queue = &mut guard;
            Ok(Queue {
                // Forceibly transmute to 'world lifetime. Should be fine, since `Queue` also holds the guard.
                queue: unsafe {
                    std::mem::transmute::<&mut pumicite::Queue, &mut pumicite::Queue>(queue)
                },
                _guard: Some(guard),
                _marker: PhantomData,
            })
        }
    }

    fn init_access(
        state: &Self::State,
        system_meta: &mut SystemMeta,
        system_access: &mut bevy_ecs::system::SystemAccess,
        _world: &mut World,
    ) {
        let mut queue_write = FilteredAccess::default();
        queue_write.add_write(*state);
        queue_write.and_with(IS_RESOURCE);
        if system_access.try_add(queue_write).is_err() {
            panic!(
                "error[B0002]: Queue<{}> in system {} conflicts with a previous system \
                 parameter, such as another `Queue` for the same or an aliased queue. Consider \
                 removing the conflicting access, or using `Without<IsResource>` to make queries \
                 disjoint from resources. See: https://bevy.org/learn/errors/b0002",
                std::any::type_name::<T>(),
                system_meta.name()
            );
        }
    }
}

/// Builds a [`Queue`] param bound to the queue resource with this component ID,
/// regardless of the param's marker type.
pub struct QueueConfig(pub ComponentId);

// SAFETY: `QueueConfig` is only constructed by `SubmissionSetsPass`, with component IDs
// registered in `QueueConfiguration`. Those always refer to a `SharedQueue` resource.
unsafe impl<'w, T: 'static> SystemParamBuilder<Queue<'w, T>> for QueueConfig {
    fn build(self, _world: &mut World) -> ComponentId {
        self.0
    }
}

/// Marker type for the primary graphics/render queue.
///
/// This queue supports `GRAPHICS` operations and typically also `COMPUTE` and `TRANSFER`.
/// Created with priority 1.0 (highest), making it suitable for latency-sensitive work.
pub struct RenderQueue;

/// Marker type for the dedicated transfer queue.
pub struct TransferQueue;

/// Marker type for the dedicated compute queue.
///
/// Created with priority 1.0, suitable for standalone compute work
/// (e.g., GPU-driven culling, physics simulation).
///
/// # Aliasing
///
/// May alias to [`RenderQueue`] if no dedicated compute queue exists.
pub struct ComputeQueue;

/// Marker type for asynchronous/background compute operations.
///
/// Created with priority 0.1, suitable for compute work that can run in the background
/// without blocking the main render path (e.g., BLAS building, texture processing).
///
/// You typically use this queue marker as a [SharedQueue] by callling [`QueueWorldExt::make_shared_queue`]
///
/// # When to Use
///
/// Use this queue for:
/// - Work that spans multiple frames
/// - Operations that shouldn't block rendering
/// - Lower-priority background processing
pub struct AsyncComputeQueue;

/// Extension trait for [`World`] that provides queue access utilities.
pub trait QueueWorldExt {
    /// Creates a [`SharedQueue`] for the specified queue type.
    ///
    /// Use this when you do need thread-safe access a queue from outside the ECS
    /// (e.g., in async tasks or background threads).
    ///
    /// # Implementation
    /// After calling this method, all accesses to the queue will be protected with a
    /// mutex.
    ///
    /// TODO: Implement shared queue with "VK_KHR_internally_synchronized_queues".
    ///
    /// # Panics
    ///
    /// Panics if the queue type `T` was never registered via
    /// [`PumiciteApp::init_device_queue_with_caps`](crate::PumiciteApp::init_device_queue_with_caps).
    fn make_shared_queue<T: 'static>(&self) -> SharedQueue;
}
impl QueueWorldExt for World {
    fn make_shared_queue<T: 'static>(&self) -> SharedQueue {
        let config = self.resource::<QueueConfiguration>();
        let component_id = config
            .component_id_of_queue::<T>()
            .expect("Please call app.init_device_queue");

        let shared_queue: &SharedQueue =
            unsafe { self.get_resource_by_id(component_id).unwrap().deref() };
        shared_queue.clone()
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::{
        prelude::*,
        resource::IsResource,
        system::{ParamBuilder, SystemAccess, SystemParamBuilder},
        world::EntityRef,
    };

    use super::{Queue, QueueConfig};

    // The queues are built with `QueueConfig`, so these tests don't need a device. Only
    // the queues' component IDs matter for access, so any resource stands in for them.
    #[derive(Resource, Default)]
    struct QueueA;
    #[derive(Resource, Default)]
    struct QueueB;
    #[derive(Resource, Default)]
    struct Other;

    #[test]
    fn query_without_resources_then_queue() {
        fn system(_: Query<EntityRef, Without<IsResource>>, _: Queue<()>) {}
        let mut world = World::new();
        let queue = world.init_resource::<QueueA>();
        IntoSystem::into_system((ParamBuilder, QueueConfig(queue)).build_system(system))
            .initialize(&mut world);
    }

    #[test]
    fn queue_then_query_without_resources() {
        fn system(_: Queue<()>, _: Query<EntityRef, Without<IsResource>>) {}
        let mut world = World::new();
        let queue = world.init_resource::<QueueA>();
        IntoSystem::into_system((QueueConfig(queue), ParamBuilder).build_system(system))
            .initialize(&mut world);
    }

    #[test]
    #[should_panic(expected = "error[B0002]: Queue<()>")]
    fn query_entity_ref_then_queue() {
        fn system(_: Query<EntityRef>, _: Queue<()>) {}
        let mut world = World::new();
        let queue = world.init_resource::<QueueA>();
        IntoSystem::into_system((ParamBuilder, QueueConfig(queue)).build_system(system))
            .initialize(&mut world);
    }

    #[test]
    #[should_panic(expected = "error[B0001]")]
    fn queue_then_query_entity_ref() {
        fn system(_: Queue<()>, _: Query<EntityRef>) {}
        let mut world = World::new();
        let queue = world.init_resource::<QueueA>();
        IntoSystem::into_system((QueueConfig(queue), ParamBuilder).build_system(system))
            .initialize(&mut world);
    }

    #[test]
    #[should_panic(expected = "error[B0002]: Queue<()>")]
    fn world_then_queue() {
        fn system(_: &World, _: Queue<()>) {}
        let mut world = World::new();
        let queue = world.init_resource::<QueueA>();
        IntoSystem::into_system((ParamBuilder, QueueConfig(queue)).build_system(system))
            .initialize(&mut world);
    }

    #[test]
    #[should_panic(expected = "&World in system")]
    fn queue_then_world() {
        fn system(_: Queue<()>, _: &World) {}
        let mut world = World::new();
        let queue = world.init_resource::<QueueA>();
        IntoSystem::into_system((QueueConfig(queue), ParamBuilder).build_system(system))
            .initialize(&mut world);
    }

    #[test]
    #[should_panic(expected = "error[B0002]: Queue<()>")]
    fn same_queue_twice() {
        // Also what happens when one queue aliases the other.
        fn system(_: Queue<()>, _: Queue<()>) {}
        let mut world = World::new();
        let queue = world.init_resource::<QueueA>();
        IntoSystem::into_system((QueueConfig(queue), QueueConfig(queue)).build_system(system))
            .initialize(&mut world);
    }

    #[test]
    fn different_queues() {
        fn system(_: Queue<()>, _: Queue<()>) {}
        let mut world = World::new();
        let queue_a = world.init_resource::<QueueA>();
        let queue_b = world.init_resource::<QueueB>();
        let access = IntoSystem::into_system(
            (QueueConfig(queue_a), QueueConfig(queue_b)).build_system(system),
        )
        .initialize(&mut world);
        let SystemAccess::Shared(access) = access else {
            panic!("expected shared access, got {access:?}");
        };
        assert!(access.combined_access().has_write(queue_a));
        assert!(access.combined_access().has_write(queue_b));
    }

    #[test]
    fn res_then_queue() {
        fn system(_: Res<Other>, _: Queue<()>) {}
        let mut world = World::new();
        let queue = world.init_resource::<QueueA>();
        IntoSystem::into_system((ParamBuilder, QueueConfig(queue)).build_system(system))
            .initialize(&mut world);
    }
}
