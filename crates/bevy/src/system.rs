//! Command encoding infrastructure for render systems.
//!
//! This module provides the system parameter [`SubmissionState`] for recording GPU commands
//! within Bevy systems. It manages command buffer allocation, encoding, and submission
//! as part of the submission set system.
//!
//! # Command Encoding Workflow
//!
//! When you add a system to a submission set (via [`PumiciteApp::add_submission_set`](crate::PumiciteApp::add_submission_set)),
//! the following happens automatically:
//!
//! 1. **Prelude**: A command buffer is allocated and recording begins
//! 2. **Your systems**: Use [`SubmissionState`] to record commands
//! 3. **Submission**: The command buffer is ended and submitted to the queue
//!
//! All systems in the same submission set share a single command buffer and execute
//! serially. Commands are recorded in system execution order.
//!
//! # Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy_pumicite::{SubmissionState, DefaultRenderSet};
//!
//! fn my_render_system(mut ctx: SubmissionState) {
//!     ctx.record(|encoder| {
//!         // Record transfer, compute, or setup commands
//!     });
//!
//!     ctx.render(|mut render_pass| {
//!         // Record rendering commands inside an active render pass
//!         render_pass.draw(0..3, 0..1);
//!     });
//! }
//!
//! # let mut app = App::new();
//! // Add to a submission set
//! app.add_systems(PostUpdate, my_render_system.in_set(DefaultRenderSet));
//! ```
//!
//! # Active Render Pass
//!
//! The [`SubmissionState::render`] method only works when a render pass is active.
//! If called without an active render pass, the callback is not invoked.
//! Call [`CommandEncoder::begin_rendering`] to begin a render pass.
//!
//! Use [`SubmissionState::record`] for commands outside render passes.

use bevy_ecs::{
    change_detection::Tick,
    component::ComponentId,
    query::FilteredAccess,
    resource::{IS_RESOURCE, Resource},
    schedule::InternedSystemSet,
    system::{SystemMeta, SystemParam, SystemParamBuilder},
    world::{Mut, World, unsafe_world_cell::UnsafeWorldCell},
};

use glam::Vec4;
use pumicite::{
    Device,
    ash::{VkResult, vk},
    command::{
        CommandBuffer, CommandEncoder, CommandEncoderRenderPassState, CommandPool, RenderPass,
    },
    debug::DebugObject,
    sync::Timeline,
};
use std::{ffi::CString, mem::MaybeUninit};

use super::queue::Queue;

/// Internal state shared by all systems in a submission set.
///
/// Each submission set has its own `RenderSetSharedState` instance, managing:
/// - Command pool and buffer allocation
/// - Command encoding state
/// - Timeline synchronization
///
/// This resource is created automatically when you call
/// [`PumiciteApp::add_submission_set`](crate::PumiciteApp::add_submission_set) and should
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
    /// The submission set this state belongs to.
    set: InternedSystemSet,
    /// Debug name for this submission set.
    name: CString,
    /// Debug color for the debug region of this submission set.
    color: Vec4,
    /// Binary semaphores the next submission waits on and signals, for swapchain images.
    binary_wait: Option<vk::Semaphore>,
    binary_signal: Option<vk::Semaphore>,
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
    pub(crate) fn new(
        device: Device,
        queue_family_index: u32,
        set: InternedSystemSet,
        color: Vec4,
    ) -> VkResult<Self> {
        let cstring = CString::new(format!("{set:?}")).unwrap();
        Ok(Self {
            command_pool: CommandPool::new_resettable(device.clone(), queue_family_index)?
                .with_name(cstring.as_c_str()),
            pending_command_buffers: RingBuffer::new(),
            recording_command_buffer: Box::new(None),
            encoder: unsafe { CommandEncoder::new() },
            timeline: Timeline::new(device)?.with_name(cstring.as_c_str()),
            stage_index: 0,
            set,
            name: cstring,
            color,
            binary_wait: None,
            binary_signal: None,
        })
    }
}
unsafe impl Send for RenderSetSharedState {}
unsafe impl Sync for RenderSetSharedState {}

/// System parameter for recording GPU commands within a submission set.
///
/// `SubmissionState` provides access to the shared command encoder for systems
/// in a submission set. Use it to record rendering, compute, and transfer commands
/// that will be submitted together with other systems in the same set.
///
/// # Requirements
///
/// This parameter is only valid for systems added to a submission set.
/// Submission sets are (SystemSet)[`bevy_ecs::schedule::SystemSet`]s added via
/// [`PumiciteApp::add_submission_set`](crate::PumiciteApp::add_submission_set).
///
/// # Panics
///
/// Panics if used in a system that isn't part of a submission set, or if a system has
/// more than one `SubmissionState` param or other conflicting access.
///
/// # Example
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_pumicite::{SubmissionState, DefaultRenderSet};
///
/// fn my_render_system(mut ctx: SubmissionState) {
///     ctx.record(|encoder| {
///         // Record transfer, compute, or setup commands
///     });
///
///     ctx.render(|mut render_pass| {
///         // Record rendering commands inside an active render pass
///         render_pass.draw(0..3, 0..1);
///     });
/// }
///
/// # let mut app = App::new();
/// // Add to a submission set
/// app.add_systems(PostUpdate, my_render_system.in_set(DefaultRenderSet));
/// ```
pub struct SubmissionState<'world> {
    state: Mut<'world, RenderSetSharedState>,
}
impl SubmissionState<'_> {
    /// Encode commands outside an active render pass.
    ///
    /// If the command encoder has an active render pass, `encode` will not be called
    /// and nothing will be encoded.
    #[track_caller]
    pub fn record(&mut self, encode: impl FnOnce(&mut CommandEncoder)) {
        if let CommandEncoderRenderPassState::InsideRenderPass { start_location, .. } = self.state()
        {
            tracing::warn!(
                "`SubmissionState::record` called at {} when there is an active render pass open. The active render pass was opened at {}",
                std::panic::Location::caller(),
                start_location
            );
            return;
        }
        encode(&mut self.state.encoder);
    }

    /// Encode commands on an active render pass.
    ///
    /// If the command encoder doesn't already have an active render pass,
    /// `encode` will not be called and nothing will be encoded.
    #[track_caller]
    pub fn render(&mut self, encode: impl FnOnce(RenderPass)) {
        let Some(pass) = self.state.encoder.continue_rendering() else {
            tracing::warn!(
                "`SubmissionState::render` called at {} without an active render pass",
                std::panic::Location::caller()
            );
            return;
        };
        encode(pass);
    }

    pub fn state(&self) -> &CommandEncoderRenderPassState {
        self.state.encoder.render_pass_state()
    }

    /// Makes this frame's submission wait on the binary semaphore `wait` and signal the binary
    /// semaphore `signal`.
    ///
    /// # Safety
    /// Same as [`pumicite::Queue::submit_with_binary_semaphores`].
    pub(crate) unsafe fn set_binary_semaphores(
        &mut self,
        wait: vk::Semaphore,
        signal: vk::Semaphore,
    ) {
        self.state.binary_wait = Some(wait);
        self.state.binary_signal = Some(signal);
    }
}

unsafe impl SystemParam for SubmissionState<'_> {
    type State = ComponentId;

    type Item<'world, 'state> = SubmissionState<'world>;

    fn init_state(_world: &mut World) -> ComponentId {
        ComponentId::new(usize::MAX)
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &SystemMeta,
        world: UnsafeWorldCell<'world>,
        _change_tick: Tick,
    ) -> Result<Self::Item<'world, 'state>, bevy_ecs::system::SystemParamValidationError> {
        unsafe {
            if *state == ComponentId::new(usize::MAX) {
                panic!(
                    "System {} was never added to a SubmissionSet!",
                    system_meta.name()
                )
            }
            let value = world.get_resource_mut_by_id(*state).unwrap();
            Ok(SubmissionState {
                state: value.with_type(),
            })
        }
    }

    fn configurate(
        state: &mut Self::State,
        _system_meta: &mut SystemMeta,
        config: &mut dyn std::any::Any,
    ) {
        if let Some(config) = config.downcast_mut::<RenderSetSharedStateConfig>() {
            *state = config.id;
            config.used = true;
        }
    }

    fn init_access(
        state: &Self::State,
        system_meta: &mut SystemMeta,
        system_access: &mut bevy_ecs::system::SystemAccess,
        world: &mut World,
    ) {
        let component_access_set = system_access.require_shared_access::<Self>(system_meta);

        // Two `SubmissionState`s in one system would hand out the same shared state
        // mutably at once. Each declares a read of a marker resource, which doesn't
        // conflict with other systems, so that the second one finds the first.
        let mut marker_access = FilteredAccess::default();
        marker_access.add_read(
            world
                .components_registrator()
                .register_component::<SubmissionStateMarker>(),
        );
        marker_access.and_with(IS_RESOURCE);
        if component_access_set
            .filtered_accesses()
            .contains(&marker_access)
        {
            panic!(
                "System {} has more than one `SubmissionState` param. They would all hand out \
                 the same command encoder mutably at once, so use only one.",
                system_meta.name()
            );
        }
        component_access_set.add(marker_access);

        // User systems are only configured after initialization, so the build pass
        // declares their write instead.
        if *state != ComponentId::new(usize::MAX) {
            let mut shared_state_write = FilteredAccess::default();
            shared_state_write.add_write(*state);
            shared_state_write.and_with(IS_RESOURCE);
            if system_access.try_add(shared_state_write).is_err() {
                panic!(
                    "error[B0002]: SubmissionState in system {} conflicts with a previous \
                     system parameter. Consider removing the conflicting access, or using \
                     `Without<IsResource>` to make queries disjoint from resources. See: \
                     https://bevy.org/learn/errors/b0002",
                    system_meta.name()
                );
            }
        }
    }
}

/// Resource that every [`SubmissionState`] declares a read of, so that a system with
/// more than one can be detected. It is never inserted.
#[derive(Resource)]
struct SubmissionStateMarker;

pub(crate) struct RenderSetSharedStateConfig {
    id: ComponentId,
    /// Set when a [`SubmissionState`] param is configured with this, which tells the
    /// build pass that the system records into the shared command encoder.
    pub(crate) used: bool,
}
impl RenderSetSharedStateConfig {
    pub(crate) fn new(id: ComponentId) -> Self {
        Self { id, used: false }
    }
}

// SAFETY: `RenderSetSharedStateConfig` is only constructed by `SubmissionSetsPass`, with
// component IDs it registered for a `RenderSetSharedState` resource.
unsafe impl<'w> SystemParamBuilder<SubmissionState<'w>> for RenderSetSharedStateConfig {
    fn build(self, _world: &mut World) -> ComponentId {
        self.id
    }
}

pub(crate) fn prelude_system(mut shared: SubmissionState) {
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

pub(crate) fn render_set_ending_system(mut shared: SubmissionState) {
    let shared = shared.state.as_mut();
    // End the render pass if one is active
    if let Some(pass) = shared.encoder.continue_rendering() {
        // End the debug label region started by the config system
        pass.end();
    }
    shared.encoder.end_label();
}

pub(crate) fn submission_system(
    mut shared: SubmissionState,
    mut queue: Queue<()>, // built with `QueueConfig`.
) {
    shared.record(|_| {}); // Close any open encoders.
    let shared = shared.state.as_mut();
    unsafe {
        shared.encoder.reset();
    }
    let binary_wait = shared.binary_wait.take();
    let binary_signal = shared.binary_signal.take();
    let Some(mut cb) = shared.recording_command_buffer.take() else {
        return;
    };
    shared.command_pool.finish(&mut cb).unwrap();
    queue.begin_label(shared.name.as_c_str(), shared.color);
    // Safety: the binary semaphores were set with `SubmissionState::set_binary_semaphores`,
    // whose caller upholds this.
    unsafe { queue.submit_with_binary_semaphores(&mut cb, binary_wait, binary_signal) }.unwrap();
    queue.end_label();
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
