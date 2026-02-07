//! GPU synchronization primitives.
//!
//! This module provides synchronization primitives for coordinating CPU and GPU work,
//! including timeline semaphores and GPU-aware mutexes.
//!
//! # Key Types
//!
//! - [`Semaphore`]: A wrapper around Vulkan timeline or binary semaphores with cached
//!   counter values for efficient polling.
//! - [`SharedSemaphore`]: Reference-counted semaphore for shared ownership.
//! - [`GPUMutex<T>`]: A mutex that tracks GPU access to a resource, enabling safe
//!   CPU access after GPU work completes.
//!
//! # GPU Mutex
//!
//! [`GPUMutex`] protects a resource that may be accessed by both CPU and GPU. It tracks
//! which timeline semaphore value must be reached before the resource is safe to access:

use ::portable_atomic::AtomicU128;
use std::{
    collections::BTreeMap,
    fmt::Debug,
    hash::Hash,
    mem::ManuallyDrop,
    ops::Deref,
    sync::{Arc, atomic::AtomicU64},
};

use ash::{
    VkResult,
    vk::{self, Handle, TaggedStructure},
};

use crate::{Device, HasDevice, utils::AsVkHandle};

/// A mutex for resources with read/write access from multiple queues, or from both
/// the device and the queue.
///
/// `GPUMutex` pairs a resource with a timeline semaphore and tracks which timeline
/// semaphore value must be reached before the resource can be safely accessed by the
/// CPU. This enables:
///
/// - **Cross-queue syncronization**: Once (locked)[`crate::command::CommandEncoder::lock`]
///   on a (CommandEncoder)[`crate::command::CommandEncoder`], the command buffer will
///   automatically keep track of the semaphores to wait and signal upon submission.
/// - **Host syncronization**: The CPU can check [`is_locked`](Self::is_locked) or wait
///   via [`unwrap_block`](Self::unwrap_block) to safely wait for the GPU work to finish
///   before accessing the wrapped resource.
/// - **Deferred cleanup**: If dropped while GPU work is pending, the resource is
///   automatically sent to a cleanup queue and dropped only after the GPU finishes.
pub struct GPUMutex<T> {
    /// (SharedSemaphore::as_ptr(), u64 timestamp)
    ptr: AtomicU128,
    pub(crate) inner: ManuallyDrop<Box<T>>,
}
impl<T> Deref for GPUMutex<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

fn encode_semaphore(ptr: *const Semaphore, value: u64) -> u128 {
    unsafe { std::mem::transmute((ptr, value)) }
}
fn decode_semaphore(encoded: u128) -> (*const Semaphore, u64) {
    unsafe { std::mem::transmute(encoded) }
}

impl<T: Send> GPUMutex<T> {
    /// Creates a new unlocked GPU mutex.
    pub fn new(item: T) -> Self
    where
        T: Sized,
    {
        GPUMutex {
            ptr: AtomicU128::new(0),
            inner: ManuallyDrop::new(Box::new(item)),
        }
    }

    /// Creates a new GPU mutex that is already locked.
    ///
    /// The mutex will be considered locked until the given semaphore reaches the specified value.
    pub fn new_locked(item: Box<T>, semaphore: SharedSemaphore, value: u64) -> Self
    where
        T: Sized,
    {
        let semaphore_ptr = semaphore.into_raw();
        GPUMutex {
            ptr: AtomicU128::new(encode_semaphore(semaphore_ptr, value)),
            inner: ManuallyDrop::new(item),
        }
    }

    /// Returns true if there is currently pending GPU work using the locked resource.
    pub fn is_locked(&self) -> bool {
        let (semaphore_ptr, signal_value) =
            decode_semaphore(self.ptr.load(std::sync::atomic::Ordering::Relaxed));
        if semaphore_ptr.is_null() || signal_value == 0 {
            return false;
        }
        unsafe {
            let semaphore = &*semaphore_ptr; // This should be fine because GPUMutex already retains the ownership of the semaphore
            !semaphore.is_signaled(signal_value)
        }
    }

    /// Obtain a mutable reference to the wrapped item if there is no currently pending GPU work using the locked resource.
    pub fn try_deref_mut(&mut self) -> Option<&mut T> {
        if self.is_locked() {
            None
        } else {
            Some(&mut self.inner)
        }
    }

    /// Method for GPU access.
    /// In the event that the returned semaphore is None, the caller does not need to wait
    /// on any semaphores before using this resource.
    ///
    /// If the returned semaphore is None, the call gets exclusive access to the resource immediately.
    /// The exclusive access ends when the passed in semaphore gets signaled.
    ///
    /// If the returned semaphore is Some, the call gets exclusive access to the resource after the returned
    /// semaphore was signaled. The exclusive access ends when the passed in semaphore was signaled.
    pub(crate) unsafe fn lock_until(
        &self,
        semaphore: &SharedSemaphore,
        signal_value: u64,
    ) -> (Option<SharedSemaphore>, u64) {
        let new_semaphore_ptr = semaphore.as_ptr();
        let old_value = self.ptr.swap(
            encode_semaphore(new_semaphore_ptr, signal_value),
            std::sync::atomic::Ordering::Relaxed,
        );
        let (old_semaphore_ptr, old_signal_value): (*const Semaphore, u64) =
            decode_semaphore(old_value);
        if new_semaphore_ptr == old_semaphore_ptr {
            debug_assert!(signal_value >= old_signal_value);
            (None, old_signal_value)
        } else {
            unsafe {
                // GPUMutex retains ownership over the new semaphore. Clone the new semaphore.
                Arc::increment_strong_count(new_semaphore_ptr);
                if old_semaphore_ptr.is_null() {
                    (None, old_signal_value)
                } else {
                    // The old semaphore have its ownership transferred out.
                    (
                        Some(SharedSemaphore::from_raw(old_semaphore_ptr)),
                        old_signal_value,
                    )
                }
            }
        }
    }

    /// Blocks until GPU work completes, then returns the protected value.
    pub fn unwrap_block(mut self) -> Box<T> {
        unsafe {
            let (semaphore_ptr, wait_value): (*const Semaphore, u64) =
                decode_semaphore(*self.ptr.get_mut());
            if !semaphore_ptr.is_null() {
                // At this point we have exclusive ownership over the timeline semaphore.
                let semaphore = Arc::from_raw(semaphore_ptr);
                // Wait for GPU access to finish
                semaphore.wait_blocked(wait_value, !0).unwrap();
            }
            let inner = ManuallyDrop::take(&mut self.inner);
            std::mem::forget(self);
            inner
        }
    }

    /// Async version of [`unwrap_block`](Self::unwrap_block).
    pub async fn unwrap_block_async(mut self) -> VkResult<Box<T>> {
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };
        let ptr = *self.ptr.get_mut();
        std::mem::forget(self);

        let (semaphore_ptr, wait_value): (*const Semaphore, u64) = decode_semaphore(ptr);
        if semaphore_ptr.is_null() {
            return Ok(inner);
        }
        // At this point we have exclusive ownership over the timeline semaphore.
        let semaphore = unsafe { Arc::from_raw(semaphore_ptr) };
        semaphore.wait_async(wait_value).await?;
        Ok(inner)
    }

    /// Extracts the inner value without waiting.
    ///
    /// # Safety
    ///
    /// Caller must ensure the semaphore has been signaled before using the resource.
    pub unsafe fn into_inner(mut self) -> (Box<T>, Option<SharedSemaphore>, u64) {
        let (semaphore_ptr, wait_value): (*const Semaphore, u64) =
            decode_semaphore(*self.ptr.get_mut());
        unsafe {
            let semaphore = if semaphore_ptr.is_null() {
                None
            } else {
                Some(SharedSemaphore::from_raw(semaphore_ptr))
            };
            let raw_ptr = self.inner.as_mut() as *mut T;
            std::mem::forget(self);
            (Box::from_raw(raw_ptr), semaphore, wait_value)
        }
    }
}

/// A future that resolves when a GPU mutex becomes unlocked.
///
/// Used for async waiting on GPU resources.
pub struct GPUMutexFuture<T> {
    listener: Option<event_listener::EventListener<()>>,
    item: Option<Box<T>>,
}
impl<T> GPUMutexFuture<T> {
    pub fn resolved(item: Box<T>) -> Self {
        Self {
            listener: None,
            item: Some(item),
        }
    }
    pub fn pending(item: Box<T>, listener: event_listener::EventListener<()>) -> Self {
        Self {
            listener: Some(listener),
            item: Some(item),
        }
    }
}

impl<T> Future for GPUMutexFuture<T> {
    type Output = Box<T>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if let Some(listener) = &mut self.as_mut().listener {
            let listener = std::pin::pin!(listener);
            match listener.poll(cx) {
                std::task::Poll::Ready(_) => {
                    self.as_mut().listener = None;
                    std::task::Poll::Ready(self.as_mut().item.take().unwrap())
                }
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        } else {
            std::task::Poll::Ready(
                self.as_mut()
                    .item
                    .take()
                    .expect("Unexpectedly polled GPUMutexFuture after it has been completed"),
            )
        }
    }
}

impl<T> Drop for GPUMutex<T> {
    fn drop(&mut self) {
        unsafe {
            let (semaphore_ptr, wait_value): (*const Semaphore, u64) =
                decode_semaphore(*self.ptr.get_mut());
            if semaphore_ptr.is_null() {
                ManuallyDrop::drop(&mut self.inner);
                return;
            }
            // At this point we have exclusive ownership over the timeline semaphore.
            let semaphore = Arc::from_raw(semaphore_ptr);
            if semaphore.is_signaled(wait_value) {
                ManuallyDrop::drop(&mut self.inner);
                return;
            }
            fn drop_box<T>(ptr_to_drop: *mut ()) {
                let box_to_drop: Box<T> = unsafe { Box::from_raw(ptr_to_drop as *mut T) };
                drop(box_to_drop);
            }

            semaphore
                .clone()
                .device
                .schedule_resource_for_deferred_drop(RetiredGPUMutex {
                    semaphore: SharedSemaphore(semaphore),
                    value: wait_value,
                    resource: Box::<T>::as_mut_ptr(&mut *self.inner) as *mut (),
                    drop: drop_box::<T>,
                });
        }
    }
}

//region TimelineSemaphore

/// A Vulkan semaphore wrapper representing either a timeline semaphore or a binary semaphore.
///
/// # Cached Counter Value
///
/// For timeline semaphores, the counter value is cached in an [`AtomicU64`] to reduce
/// API calls when polling. The cache is updated on every [`value()`](Self::value) call.
///
/// # Binary vs Timeline
///
/// Timeline semaphores are used on most scenarios. They have an associated counter that can be
/// signaled to specific values and waited on.
///
/// Binary semaphores are useful for syncronization with swapchain acquire-present and was used
/// exclusively in that context. We always pair a binary semaphore with a [`vk::Fence`] to enable
/// CPU-side waiting.
///
/// Use [`is_binary()`](Self::is_binary) to check the semaphore type.
pub struct Semaphore {
    device: Device,
    handle: vk::Semaphore,
    value: AtomicU64,

    /// Binary semaphore will always have a non-null Fence.
    fence: vk::Fence,
}
impl Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(if self.is_binary() {
            "BinarySemaphore"
        } else {
            "TimelineSemaphore"
        })
        .field(&self.handle)
        .field(&self.value())
        .finish()
    }
}

impl Semaphore {
    /// Returns `true` if this is a binary semaphore.
    pub fn is_binary(&self) -> bool {
        !self.fence.is_null()
    }

    /// Creates a new timeline semaphore with the given initial value.
    pub fn new(device: Device, initial_value: u64) -> VkResult<Self> {
        let semaphore = unsafe {
            let mut type_info = vk::SemaphoreTypeCreateInfo {
                semaphore_type: vk::SemaphoreType::TIMELINE,
                ..Default::default()
            };
            let info = vk::SemaphoreCreateInfo::default().push(&mut type_info);
            device.create_semaphore(&info, None)
        }?;
        Ok(Self {
            device,
            handle: semaphore,
            value: AtomicU64::new(initial_value),
            fence: vk::Fence::null(),
        })
    }

    /// Creates a new binary semaphore with an associated fence.
    ///
    /// Binary semaphores use a fence for CPU-side waiting since binary semaphores
    /// cannot be waited on directly from the CPU in Vulkan.
    pub fn new_binary(device: Device, signaled: bool) -> VkResult<Self> {
        let semaphore = unsafe {
            let info = vk::SemaphoreCreateInfo::default();
            device.create_semaphore(&info, None)
        }?;
        let fence = unsafe {
            let info = vk::FenceCreateInfo {
                flags: if signaled {
                    vk::FenceCreateFlags::SIGNALED
                } else {
                    vk::FenceCreateFlags::empty()
                },
                ..Default::default()
            };
            device.create_fence(&info, None)?
        };
        Ok(Self {
            device,
            handle: semaphore,
            value: AtomicU64::new(0),
            fence,
        })
    }

    /// Returns the raw fence handle (binary semaphores only).
    pub fn raw_fence(&self) -> vk::Fence {
        self.fence
    }

    /// Returns the current counter value (timeline semaphores only).
    ///
    /// Queries the device and updates the cached value.
    pub fn value(&self) -> u64 {
        assert!(!self.is_binary());
        let new_value = unsafe {
            self.device
                .get_semaphore_counter_value(self.handle)
                .unwrap()
        };
        let old_value = self
            .value
            .fetch_max(new_value, std::sync::atomic::Ordering::Relaxed);
        debug_assert!(old_value <= new_value); // why would we panic here?
        new_value
    }

    /// Returns `true` if the semaphore has reached the specified value.
    ///
    /// For binary semaphores, checks the fence status.
    /// For timeline semaphores, compares against the cached counter.
    pub fn is_signaled(&self, val: u64) -> bool {
        if self.is_binary() {
            unsafe { self.device.get_fence_status(self.fence).unwrap() }
        } else {
            self.value() >= val
        }
    }

    /// Signals the semaphore to the specified value from the CPU (timeline only).
    ///
    /// No-op if the semaphore is already at or past the given value.
    pub fn signal(&self, val: u64) {
        assert!(!self.is_binary());
        let old_value = self.value.load(std::sync::atomic::Ordering::Relaxed);
        if old_value >= val {
            return;
        }
        unsafe {
            self.device
                .signal_semaphore(&vk::SemaphoreSignalInfo {
                    semaphore: self.handle,
                    value: val,
                    ..Default::default()
                })
                .unwrap();
            self.value.store(val, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Blocks until the semaphore reaches the specified value.
    ///
    /// For binary semaphores, waits on the fence and resets it.
    /// For timeline semaphores, returns early if already signaled.
    pub fn wait_blocked(&self, value: u64, timeout: u64) -> VkResult<()> {
        if self.is_binary() {
            unsafe {
                self.device.wait_for_fences(&[self.fence], false, timeout)?;
                self.device.reset_fences(&[self.fence])?;
            }
            Ok(())
        } else {
            if self.value.load(std::sync::atomic::Ordering::Relaxed) >= value {
                return Ok(());
            }
            unsafe {
                self.device.wait_semaphores(
                    &vk::SemaphoreWaitInfo {
                        semaphore_count: 1,
                        p_semaphores: &self.handle,
                        p_values: &value,
                        ..Default::default()
                    },
                    timeout,
                )?;
            }
            self.value
                .fetch_max(value, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
    }

    /// Async version of [`wait_blocked`](Self::wait_blocked).
    pub async fn wait_async(self: &Arc<Self>, value: u64) -> VkResult<()> {
        if self.is_binary() {
            let fence = self.fence;
            let device = self.device.clone();
            blocking::unblock(move || unsafe {
                device.wait_for_fences(&[fence], false, !0)?;
                device.reset_fences(&[fence])?;
                Ok::<(), vk::Result>(())
            })
            .await?;
            Ok(())
        } else {
            if self.is_signaled(value) {
                return Ok(());
            }

            let event = event_listener::Event::new();
            let listener = event.listen();

            unsafe fn drop_event(event: *mut ()) {
                let event: event_listener::Event = unsafe { std::mem::transmute(event) };
                event.notify_relaxed(1);
            }
            self.device
                .schedule_resource_for_deferred_drop(RetiredGPUMutex {
                    semaphore: SharedSemaphore(self.clone()),
                    value,
                    resource: unsafe {
                        std::mem::transmute::<event_listener::Event, *mut ()>(event)
                    },
                    drop: drop_event,
                });
            listener.await;
            Ok(())
        }
    }
}
impl HasDevice for Semaphore {
    fn device(&self) -> &Device {
        &self.device
    }
}
impl AsVkHandle for Semaphore {
    type Handle = vk::Semaphore;
    fn vk_handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.handle, None);
            if !self.fence.is_null() {
                self.device.destroy_fence(self.fence, None);
            }
        }
    }
}

/// A reference-counted semaphore for shared ownership.
#[derive(Clone)]
pub struct SharedSemaphore(Arc<Semaphore>);
impl Hash for SharedSemaphore {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}
impl Ord for SharedSemaphore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Arc::as_ptr(&self.0).cmp(&Arc::as_ptr(&other.0))
    }
}
impl PartialOrd for SharedSemaphore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for SharedSemaphore {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for SharedSemaphore {}
impl Deref for SharedSemaphore {
    type Target = Arc<Semaphore>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl HasDevice for SharedSemaphore {
    fn device(&self) -> &Device {
        &self.0.device
    }
}
impl AsVkHandle for SharedSemaphore {
    type Handle = vk::Semaphore;
    fn vk_handle(&self) -> Self::Handle {
        self.0.handle
    }
}
impl SharedSemaphore {
    pub fn new(device: Device, initial_value: u64) -> VkResult<Self> {
        Ok(SharedSemaphore(Arc::new(Semaphore::new(
            device,
            initial_value,
        )?)))
    }
    pub fn new_binary(device: Device, signaled: bool) -> VkResult<Self> {
        Ok(SharedSemaphore(Arc::new(Semaphore::new_binary(
            device, signaled,
        )?)))
    }
    pub fn as_ptr(&self) -> *const Semaphore {
        Arc::as_ptr(&self.0)
    }
    pub fn into_raw(self) -> *const Semaphore {
        Arc::into_raw(self.0)
    }
    /// # Safety
    /// The provided pointer must have been created by [`into_raw`](Self::into_raw).
    pub unsafe fn from_raw(ptr: *const Semaphore) -> Self {
        unsafe { Self(Arc::from_raw(ptr)) }
    }
}

/// A resource awaiting deferred destruction.
///
/// When a [`GPUMutex`] is dropped while the GPU is still using its resource,
/// the resource is wrapped in this struct and sent to the recycler thread.
/// The resource will be dropped once the semaphore reaches the specified value.
pub(crate) struct RetiredGPUMutex {
    /// The semaphore to wait on before dropping.
    semaphore: SharedSemaphore,
    /// The semaphore value that must be reached.
    value: u64,
    /// Type-erased pointer to the resource.
    resource: *mut (),
    /// Type-erased drop function for the resource.
    drop: unsafe fn(*mut ()),
}
unsafe impl Send for RetiredGPUMutex {}
unsafe impl Sync for RetiredGPUMutex {}
impl Drop for RetiredGPUMutex {
    fn drop(&mut self) {
        unsafe {
            (self.drop)(self.resource);
        }
    }
}

/// A timeline for scheduling command buffers with guaranteed execution ordering.
///
/// A timeline manages the **logical** execution order of command buffers using
/// timeline semaphores. Command buffers scheduled onto the same timeline are
/// guaranteed to execute in the order they were scheduled.
///
/// # Usage
///
/// ```
/// # use pumicite::{Device, sync::Timeline, command::CommandPool};
/// # let (device, queue) = Device::create_system_default().unwrap();
/// // Create a timeline for a specific execution sequence
/// let mut timeline = Timeline::new(device.clone()).unwrap();
/// let mut pool = CommandPool::new(device, queue.family_index()).unwrap();
///
/// // Schedule command buffers in order
/// let mut command_buffer1 = pool.alloc().unwrap();
/// let mut command_buffer2 = pool.alloc().unwrap();
///
/// timeline.schedule(&mut command_buffer1); // timestamp = 1
/// timeline.schedule(&mut command_buffer2); // timestamp = 2
///
/// // cmd2 will wait for cmd1 to complete, regardless of submission order
/// ```
pub struct Timeline {
    /// The timeline semaphore used for ordering and synchronization.
    semaphore: SharedSemaphore,

    /// Timestamp value for the previously scheduled command buffer.
    /// The next scheduled command buffer will wait for the timeline semaphore to be `value`,
    /// and signal the timeline semaphore at `value + 1`.
    value: u64,
}
impl Timeline {
    pub fn new(device: Device) -> VkResult<Self> {
        Ok(Self {
            semaphore: SharedSemaphore::new(device, 0)?,
            value: 0,
        })
    }
    /// Schedule a command buffer for execution on the current timeline.
    ///
    /// Must be called before recording the command buffer.
    pub fn schedule(&mut self, cb: &mut crate::command::CommandBuffer) {
        assert!(
            cb.semaphore.is_none(),
            "CommandBuffer was already scheduled"
        );
        self.value += 1;
        cb.timestamp = self.value;
        cb.semaphore = Some(self.semaphore.clone());
    }
}
impl HasDevice for Timeline {
    fn device(&self) -> &Device {
        self.semaphore.device()
    }
}
impl AsVkHandle for Timeline {
    type Handle = vk::Semaphore;
    fn vk_handle(&self) -> Self::Handle {
        self.semaphore.vk_handle()
    }
}

/// Spawns the background thread for deferred resource cleanup.
///
/// This thread receives [`RetiredGPUMutex`] instances via the channel and waits
/// for their associated semaphores to be signaled before dropping the resources.
///
/// # Algorithm
///
/// 1. **Receive phase**: Collects pending resources from the channel, grouping them
///    by semaphore and timestamp.
/// 2. **Wait phase**: Uses `vkWaitSemaphores` with `ANY` flag to wait for any
///    semaphore to reach its target value.
/// 3. **Cleanup phase**: Drops all resources whose semaphores have been signaled.
///
/// The thread uses object pooling to minimize allocations for the internal data structures.
///
/// # Lifetime
///
/// The thread runs until the channel is disconnected (i.e., the [`Device`] is dropped).
pub(crate) fn spawn_recycler_thread(
    device: Device,
    receiver: crossbeam_channel::Receiver<RetiredGPUMutex>,
) {
    std::thread::Builder::new()
        .name("Pumicite Deferred Drops".to_string())
        .stack_size(512)
        .spawn(move || {
            let mut queues: BTreeMap<SharedSemaphore, BTreeMap<u64, Vec<RetiredGPUMutex>>> =
                BTreeMap::new();

            // Keep a pool of reused Vec containers to reduce allocations.
            let mut container_pool: Vec<Vec<RetiredGPUMutex>> = Vec::new();
            let mut container_pool2: Vec<BTreeMap<u64, Vec<RetiredGPUMutex>>> = Vec::new();

            let mut semaphores: Vec<vk::Semaphore> = Vec::new();
            let mut semaphore_values: Vec<u64> = Vec::new();

            'outer: loop {
                'inner: loop {
                    let item = if queues.is_empty() {
                        match receiver.recv() {
                            Ok(a) => a,
                            Err(_) => break 'outer, // disconnected
                        }
                    } else {
                        match receiver.try_recv() {
                            Ok(a) => a,
                            Err(err) => match err {
                                crossbeam_channel::TryRecvError::Empty => break 'inner,
                                crossbeam_channel::TryRecvError::Disconnected => break 'outer,
                            },
                        }
                    };

                    if item.semaphore.is_signaled(item.value) {
                        drop(item);
                        continue;
                    }
                    let queue = queues
                        .entry(item.semaphore.clone())
                        .or_insert_with(|| container_pool2.pop().unwrap_or_default());
                    let items = queue
                        .entry(item.value)
                        .or_insert_with(|| container_pool.pop().unwrap_or_default());
                    items.push(item);
                }

                // Wait for any semaphore to be signaled
                for (timeline_semaphore, timestamps) in queues.iter_mut() {
                    semaphores.push(timeline_semaphore.vk_handle());
                    semaphore_values.push(*timestamps.first_entry().unwrap().key());
                }
                // There shouldn't be any empty waits. If the wait list is empty we would've blocked at receiver.recv
                debug_assert!(!semaphores.is_empty());
                unsafe {
                    device
                        .wait_semaphores(
                            &vk::SemaphoreWaitInfo {
                                flags: vk::SemaphoreWaitFlags::ANY,
                                ..Default::default()
                            }
                            .semaphores(&semaphores)
                            .values(&semaphore_values),
                            !0,
                        )
                        .unwrap();
                }
                semaphores.clear();
                semaphore_values.clear();

                // Destroy resources for which the semaphore was signaled
                for (timeline_semamphore, timestamps) in queues.iter_mut() {
                    while let Some((&timestamp, _)) = timestamps.first_key_value() {
                        if timeline_semamphore.is_signaled(timestamp) {
                            // destroy all items
                            let (_, mut items) = timestamps.pop_first().unwrap();
                            items.clear();
                            container_pool.push(items);
                        } else {
                            break;
                        }
                    }
                }

                // Maintain the queues container such that empty entries are removed
                queues.retain(|_, v| {
                    if v.is_empty() {
                        container_pool2.push(std::mem::take(v));
                        false
                    } else {
                        true
                    }
                });
            }
            drop(receiver);
        })
        .unwrap();
}
