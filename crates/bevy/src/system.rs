//! Command encoding infrastructure for render systems.
//!
//! This module provides the system parameter [`RenderState`] for recording GPU commands
//! within Bevy systems. It manages command buffer allocation, encoding, and submission
//! as part of the render set system.
//!
//! # Command Encoding Workflow
//!
//! When you add a system to a render set (via [`PumiciteApp::add_render_set`](crate::PumiciteApp::add_render_set)),
//! the following happens automatically:
//!
//! 1. **Prelude**: A command buffer is allocated and recording begins
//! 2. **Your systems**: Use [`RenderState`] to record commands
//! 3. **Submission**: The command buffer is ended and submitted to the queue
//!
//! All systems in the same render set share a single command buffer and execute
//! serially. Commands are recorded in system execution order.
//!
//! # Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy_pumicite::{RenderState, DefaultRenderSet};
//!
//! fn my_render_system(mut ctx: RenderState) {
//!     ctx.record(|encoder| {
//!         // Record transfer, compute, or setup commands
//!     });
//!
//!     ctx.render(|render_pass| {
//!         // Record rendering commands inside an active render pass
//!         render_pass.draw(0..3, 0..1);
//!     });
//! }
//!
//! // Add to a render set
//! app.add_systems(PostUpdate, my_render_system.in_set(DefaultRenderSet));
//! ```
//!
//! # Active Render Pass
//!
//! The [`RenderState::render`] method only works when a render pass is active.
//! If called without an active render pass, the callback is not invoked.
//! Call [`CommandEncoder::begin_rendering`] to begin a render pass.
//!
//! Use [`RenderState::record`] for commands outside render passes.

use bevy_ecs::{
    component::{ComponentId, Tick},
    resource::Resource,
    system::{SystemMeta, SystemParam},
    world::{Mut, World, unsafe_world_cell::UnsafeWorldCell},
};

use pumicite::{
    Device,
    ash::VkResult,
    command::{CommandBuffer, CommandEncoder, CommandPool, RenderPass},
    debug::DebugObject,
    sync::Timeline,
};
use std::{ffi::CString, mem::MaybeUninit};

use super::queue::Queue;

/// Internal state shared by all systems in a render set.
///
/// Each render set has its own `RenderSetSharedState` instance, managing:
/// - Command pool and buffer allocation
/// - Command encoding state
/// - Timeline synchronization
///
/// This resource is created automatically when you call
/// [`PumiciteApp::add_render_set`](crate::PumiciteApp::add_render_set) and should
/// not be accessed directly.
#[derive(Resource)]
pub(crate) struct RenderSetSharedState {
    /// Command pool for allocating command buffers.
    command_pool: CommandPool,
    /// Ring buffer of pending command buffers for triple-buffering.
    pending_command_buffers: RingBuffer<CommandBuffer, 3>,

    /// The command buffer currently being recorded.
    recording_command_buffer: Box<Option<CommandBuffer>>,
    /// Active command encoder. Lifetime was marked as 'static because it contains
    /// a self-reference to `recording_command_buffer`.
    encoder: CommandEncoder<'static>,
    /// Timeline semaphore for GPU synchronization.
    timeline: Timeline,
    /// Current stage index for barrier emission.
    stage_index: u32,
    /// Debug name for this render set.
    name: CString,
}
impl Drop for RenderSetSharedState {
    fn drop(&mut self) {
        // This would only run during application shutdown
        while let Some(mut cb) = self.pending_command_buffers.try_pop() {
            cb.block_until_completion().unwrap();
            self.command_pool.free(cb);
        }
    }
}
impl RenderSetSharedState {
    pub(crate) fn new(device: Device, queue_family_index: u32, name: String) -> VkResult<Self> {
        let cstring = CString::new(name).unwrap();
        Ok(Self {
            command_pool: CommandPool::new_resettable(device.clone(), queue_family_index)?
                .with_name(cstring.as_c_str()),
            pending_command_buffers: RingBuffer::new(),
            recording_command_buffer: Box::new(None),
            encoder: unsafe { CommandEncoder::new() },
            timeline: Timeline::new(device)?.with_name(cstring.as_c_str()),
            stage_index: 0,
            name: cstring,
        })
    }
}
unsafe impl Send for RenderSetSharedState {}
unsafe impl Sync for RenderSetSharedState {}

/// System parameter for recording GPU commands within a render set.
///
/// `RenderState` provides access to the shared command encoder for systems
/// in a render set. Use it to record rendering, compute, and transfer commands
/// that will be submitted together with other systems in the same set.
///
/// # Requirements
///
/// This parameter is only valid for systems added to a render set via
/// [`PumiciteApp::add_render_set`](crate::PumiciteApp::add_render_set).
///
/// # Panics
///
/// Panics if used in a system that isn't part of a render set.
///
/// # Example
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_pumicite::{RenderState, DefaultRenderSet};
///
/// fn my_render_system(mut ctx: RenderState) {
///     ctx.record(|encoder| {
///         // Record transfer, compute, or setup commands
///     });
///
///     ctx.render(|render_pass| {
///         // Record rendering commands inside an active render pass
///         render_pass.draw(0..3, 0..1);
///     });
/// }
///
/// // Add to a render set
/// app.add_systems(PostUpdate, my_render_system.in_set(DefaultRenderSet));
/// ```
pub struct RenderState<'world> {
    state: Mut<'world, RenderSetSharedState>,
}
impl RenderState<'_> {
    /// Encode commands outside an active render pass.
    ///
    /// If the command encoder has an active encoder, end the active encoder
    /// before recording any additional commands.
    pub fn record(&mut self, encode: impl FnOnce(&mut CommandEncoder)) {
        if let Some(render_pass) = self.state.encoder.continue_rendering() {
            render_pass.end();
        }
        encode(&mut self.state.encoder);
    }

    /// Encode commands on an active render pass.
    ///
    /// If the command encoder doesn't already have an active render pass,
    /// `encode` will not be called and nothing will be encoded.
    pub fn render(&mut self, encode: impl FnOnce(RenderPass)) {
        let Some(pass) = self.state.encoder.continue_rendering() else {
            tracing::warn!(
                "`RenderSetSharedStateWrapper::render` called without an active render pass"
            );
            return;
        };
        encode(pass);
    }

    pub fn inside_renderpass(&self) -> bool {
        self.state.encoder.inside_renderpass()
    }
}

unsafe impl SystemParam for RenderState<'_> {
    type State = ComponentId;

    type Item<'world, 'state> = RenderState<'world>;

    fn init_state(_world: &mut World) -> ComponentId {
        ComponentId::new(usize::MAX)
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &SystemMeta,
        world: UnsafeWorldCell<'world>,
        _change_tick: Tick,
    ) -> Self::Item<'world, 'state> {
        unsafe {
            if *state == ComponentId::new(usize::MAX) {
                panic!(
                    "System {} was never added to a RenderSet!",
                    system_meta.name()
                )
            }
            let value = world.get_resource_mut_by_id(*state).unwrap();
            RenderState {
                state: value.with_type(),
            }
        }
    }

    unsafe fn validate_param(
        _state: &mut Self::State,
        _system_meta: &SystemMeta,
        _world: UnsafeWorldCell,
    ) -> Result<(), bevy_ecs::system::SystemParamValidationError> {
        Ok(())
    }

    fn configurate(
        state: &mut Self::State,
        system_meta: &mut SystemMeta,
        config: &mut dyn std::any::Any,
    ) {
        if let Some(config) = config.downcast_ref::<RenderSetSharedStateConfig>() {
            println!("System {} received config", system_meta.name());
            *state = config.id;
        }
    }

    fn init_access(
        _state: &Self::State,
        _system_meta: &mut SystemMeta,
        _component_access_set: &mut bevy_ecs::query::FilteredAccessSet,
        _world: &mut World,
    ) {
    }
}

pub(crate) struct RenderSetSharedStateConfig {
    id: ComponentId,
}
impl RenderSetSharedStateConfig {
    pub(crate) fn new(id: ComponentId) -> Self {
        Self { id }
    }
}

pub(crate) fn prelude_system(mut shared: RenderState) {
    let shared = shared.state.as_mut();
    shared.stage_index = 0;
    assert!(shared.recording_command_buffer.is_none());
    let mut cb =
        if let Some(mut reused_command_buffer) = shared.pending_command_buffers.pop_if_full() {
            reused_command_buffer.block_until_completion().unwrap();
            shared.command_pool.reset(&mut reused_command_buffer);
            reused_command_buffer
        } else {
            shared
                .command_pool
                .alloc()
                .unwrap()
                .with_name(shared.name.as_c_str())
        };

    shared.timeline.schedule(&mut cb);
    shared.command_pool.begin(&mut cb).unwrap();
    shared.recording_command_buffer.replace(cb);
    unsafe {
        shared.encoder.reset();
        shared
            .encoder
            .set_buffer(shared.recording_command_buffer.as_mut().as_mut().unwrap());
    }
}

pub(crate) fn submission_system(
    mut shared: RenderState,
    mut queue: Queue<()>, // to be configured.
) {
    shared.record(|_| {}); // Close any open encoders.
    let shared = shared.state.as_mut();
    unsafe {
        shared.encoder.reset();
    }
    let Some(mut cb) = shared.recording_command_buffer.take() else {
        return;
    };
    shared.command_pool.finish(&mut cb).unwrap();
    queue.submit(&mut cb).unwrap();
    shared.pending_command_buffers.push(cb);
}

pub struct RingBuffer<T, const N: usize> {
    items: [MaybeUninit<T>; N],
    start: u32,
    len: u32,
}

impl<T, const N: usize> Drop for RingBuffer<T, N> {
    fn drop(&mut self) {
        loop {
            if self.try_pop().is_none() {
                return;
            }
        }
    }
}

impl<T, const N: usize> RingBuffer<T, N> {
    pub const fn new() -> Self {
        Self {
            items: [const { MaybeUninit::uninit() }; N],
            start: 0,
            len: 0,
        }
    }
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.len as usize
    }
    pub fn push(&mut self, item: T) {
        if self.is_full() {
            panic!()
        };
        let mut location = self.start + self.len;
        if location >= N as u32 {
            location -= N as u32;
        }
        self.items[location as usize].write(item);
        self.len += 1;
    }
    pub fn pop(&mut self) -> T {
        self.try_pop().unwrap()
    }
    pub fn try_pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let item = unsafe { self.items[self.start as usize].assume_init_read() };
        self.len -= 1;
        self.start += 1;
        if self.start == N as u32 {
            self.start = 0;
        }
        Some(item)
    }
    pub fn pop_if_full(&mut self) -> Option<T> {
        if self.is_full() {
            Some(self.pop())
        } else {
            None
        }
    }
    pub fn is_full(&self) -> bool {
        self.len == N as u32
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[allow(dead_code)]
    pub fn peek(&self) -> &T {
        if self.is_empty() {
            panic!()
        }
        unsafe { self.items[self.start as usize].assume_init_ref() }
    }
    #[allow(dead_code)]
    pub fn peek_mut(&mut self) -> &mut T {
        if self.is_empty() {
            panic!()
        }
        unsafe { self.items[self.start as usize].assume_init_mut() }
    }
}

/*
Let's defer this until AsyncFnOnce's Future can be bounded by Send, Sync and 'static.


pub struct RenderSystem<F: Future, T>
{
    inner: T,
    future: Box<Option<F>>,
    shared_state: ComponentId,
    frames: RingBuffer<F::Output, 3>,

    component_access: Access<ComponentId>,
    archetype_component_access: Access<ArchetypeComponentId>,
    stage_index: u32,
}

impl<
    F: Future + Send + Sync + 'static,
    FN: for<'a, 'b> AsyncFnOnce<(&'a mut CommandEncoder<'b>, ), CallOnceFuture = F> + Send + Sync + 'static,
    T: bevy_ecs::system::System<In = In<Option<F::Output>>, Out = FN>,
> bevy_ecs::system::System for RenderSystem<F, T>
where
    F::Output: Send + Sync,
{
    type In = ();

    type Out = ();

    fn default_system_sets(&self) -> Vec<InternedSystemSet> {
        self.inner.default_system_sets()
    }

    fn name(&self) -> Cow<'static, str> {
        self.inner.name()
    }

    fn component_access(&self) -> &Access<ComponentId> {
        &self.component_access
    }

    fn archetype_component_access(&self) -> &Access<ArchetypeComponentId> {
        &self.archetype_component_access
    }

    fn is_send(&self) -> bool {
        self.inner.is_send()
    }

    fn is_exclusive(&self) -> bool {
        self.inner.is_exclusive()
    }

    fn has_deferred(&self) -> bool {
        self.inner.has_deferred()
    }

    unsafe fn run_unsafe(&mut self, _: Self::In, world: UnsafeWorldCell) -> Self::Out {
        let mut shared_state = unsafe {
            world
                .get_resource_mut_by_id(self.shared_state)
                .unwrap()
                .with_type::<RenderSetSharedState>()
        };
        if self.future.is_none() {
            // This is the first time that this has run this frame.
            let frame = self.frames.pop_if_full();
            unsafe {
                let f = self.inner.run_unsafe(frame, world);
                let future = (f)(&mut shared_state.encoder);
                *self.future = Some(future);
            }
        } else if self.stage_index > shared_state.stage_index {
            shared_state.stage_index = self.stage_index;
            // this is the first system that runs for the current stage. emit barrier.
            shared_state.encoder.emit_barriers();
        }
        let pinned_future = unsafe {
            // Safety: Future is boxed up.
            std::pin::Pin::new_unchecked(self.future.as_mut().as_mut().unwrap())
        };
        match crate::command::gpu_future_poll(pinned_future) {
            std::task::Poll::Ready(result) => {
                self.frames.push(result);
                *self.future = None;
            }
            std::task::Poll::Pending => {
                self.stage_index += 1;
                return;
            }
        }
    }

    fn yielded(&self) -> bool {
        self.future.is_some()
    }

    fn apply_deferred(&mut self, world: &mut World) {
        self.inner.apply_deferred(world);
    }

    fn queue_deferred(&mut self, world: DeferredWorld) {
        self.inner.queue_deferred(world);
    }

    fn initialize(&mut self, world: &mut World) {
        self.inner.initialize(world);

        self.component_access.extend(self.inner.component_access());
    }

    fn update_archetype_component_access(&mut self, world: UnsafeWorldCell) {
        self.inner.update_archetype_component_access(world);

        self.archetype_component_access
            .extend(self.inner.archetype_component_access());
    }

    fn check_change_tick(&mut self, change_tick: Tick) {
        self.inner.check_change_tick(change_tick);
    }

    fn get_last_run(&self) -> Tick {
        self.inner.get_last_run()
    }

    fn set_last_run(&mut self, last_run: Tick) {
        self.inner.set_last_run(last_run);
    }

    fn component_access_set(&self) -> &bevy_ecs::query::FilteredAccessSet<ComponentId> {
        self.inner.component_access_set()
    }

    unsafe fn validate_param_unsafe(
        &mut self,
        world: UnsafeWorldCell,
    ) -> Result<(), bevy_ecs::system::SystemParamValidationError> {
        unsafe { self.inner.validate_param_unsafe(world) }
    }

    fn configurate(&mut self, config: &mut dyn std::any::Any) {
        if let Some(config) = config.downcast_ref::<RenderSetSharedStateConfig>() {
            self.shared_state = config.id;
            self.component_access.add_resource_write(self.shared_state);
            self.archetype_component_access.add_resource_write(config.archetype_component_id);
        } else {
            self.inner.configurate(config);
        }
    }
}

pub trait IntoRenderSystem<Out, Marker> {
    fn into_render_system(self) -> impl System<In = (), Out = ()>;
}



pub struct MarkerA;
impl<Out, Marker, T, F>
    IntoRenderSystem<Out, (Marker, MarkerA)> for T
where
    T: IntoSystem<(), Out, Marker> + Send + Sync + 'static,
    F: Future<Output = ()> + Send + Sync + 'static,
    Out: for<'a, 'b> AsyncFnOnce<(&'a mut CommandEncoder<'b>, ), CallOnceFuture = F, Output = ()> + Send + Sync + 'static,
{
    fn into_render_system(self) -> impl System<In = (), Out = ()> {
        fn empty_system(In(_i): In<Option<()>>) {}
        let pip1 = bevy_ecs::system::PipeSystem::new(
            IntoSystem::into_system(empty_system),
            IntoSystem::into_system(self),
            "".into()
        );
        RenderSystem {
            inner: pip1,
            frames: RingBuffer::new(),
            component_access: Default::default(),
            archetype_component_access: Default::default(),
            future: Box::new(None),
            shared_state: ComponentId::invalid(),
            stage_index: 0,
        }
    }
}

pub struct MarkerB;
impl<Out, F, Marker, T>
    IntoRenderSystem<Out, (Marker, MarkerB)> for T
where
    T: IntoSystem<In<Option<F::Output>>, Out, Marker> + Send + Sync + 'static,
    F: Future + Send + Sync + 'static,
    F::Output: Send + Sync + 'static,
    Out: FnOnce(&mut CommandEncoder) -> F + Send + Sync + 'static,
{
    fn into_render_system(self) -> impl System<In = (), Out = ()> {
        RenderSystem {
            inner: IntoSystem::into_system(self),
            frames: RingBuffer::new(),
            component_access: Default::default(),
            archetype_component_access: Default::default(),
            future: Box::new(None),
            shared_state: ComponentId::invalid(),
            stage_index: 0,
        }
    }
}
*/

#[cfg(test)]
mod tests {
    use super::RingBuffer;

    #[test]
    fn test_len() {
        let mut buf: RingBuffer<usize, 3> = RingBuffer::new();
        buf.push(12);
        assert_eq!(buf.len(), 1);
        assert_eq!(*buf.peek(), 12);

        buf.push(23);
        assert_eq!(buf.len(), 2);
        assert_eq!(*buf.peek(), 12);

        buf.push(34);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.pop(), 12);
        assert_eq!(buf.pop(), 23);
        assert_eq!(buf.pop(), 34);
    }
}
