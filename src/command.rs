//! # Command Encoding
//!
//! This module provides support for resource tracking, synchronization and submission for command encoding.
//!
//! During encoding, a command encoder is responsible for:
//! - Resource Lifetimes. The command encoder must keep the resource alive until execution finishes on the GPU.
//! - synchronization. The command encoder must block the execution for the current submission until all resources
//!   become available.
//! - Resource Tracking. The command encoder must keep track of resource dependencies, insert pipeline barriers
//!   and transition resources into the correct states where needed.
//!
//! ## Key Concepts
//! - Submission: A call to [vkQueueSubmit](https://vkdoc.net/man/vkQueueSubmit2). Requires exclusive ownership over the queue that the submission was made on.
//!   This has a significant performance cost on CPU, so applications should try to reduce the number of submissions.[^nvperf]
//! - Queue: A [vkQueue](https://vkdoc.net/man/VkQueue#man-header). Submissions made on the same queue begin in-order but may finish out-of-order, so we cannot
//!   count on the queue for execution dependency between submissions. It may be more helpful to consider them as "schedulers"
//!   responsible for assigning work to GPU cores. Queues are not thread-safe, so multiple queues (or schedulers) are needed
//!   if you want to submit work from multiple CPU threads without locks.
//! - Timeline Semaphore: A coarse-grained synchronization primitive for establishing execution dependency
//!   between submissions, or between a submission and the host. Each Timeline Semaphore is associated with an incrementing u64 value.
//!   Submissions and the host can wait for some Timeline Semaphores to reach a certain number, and
//!   signal some other Timeline Semaphores by setting their values to be a certain number.
//! - Timelime: Timelines function in much the same way a queue in the traditional sense would do: submissions made on the same
//!   queue are serialized, first in, first out. Each timeline maintains a monotonically increasing counter, and
//!   command buffers are assigned timestamps that determine their execution order.
//! - Pipeline Barrier: A call to [vkCmdPipelineBarrier](https://vkdoc.net/man/vkCmdPipelineBarrier2). Those are synchronization primitive within the same submission,
//!   which specifies that a [Pipeline Stage](https://vkdoc.net/man/VkPipelineStageFlagBits2) may not proceed until another [Pipeline Stage](https://vkdoc.net/man/VkPipelineStageFlagBits2)
//!   has finished. The use of Pipeline Barriers should be minimized and batched as much as possible because they
//!   may cause a GPU pipeline flush.
//! - Resource State: Information such as the [pipeline stage](https://vkdoc.net/man/VkPipelineStageFlagBits2) and [access mode](https://vkdoc.net/man/VkAccessFlagBits2)
//!   that the resource was last accessed with, and the current [layout](https://vkdoc.net/man/VkImageLayout) for textures. The command encoder
//!   must keep track of resource states to insert pipeline barriers automatically.
//! - Resource Transition: Before a resource can be used on the GPU for some purpose, we must prepare the resource by transitioning
//!   it into a state that can be used for such purpose. This may involve compressing / decompressing a texture, cache flush
//!   and invalidate, and execution dependency between pipeline stages. We just need to know the "before state" and "after state"
//!   of all the transitioning resources and send that to the driver with a pipeline barrier.
//!
//! ## Key Components
//!
//! - [`CommandBuffer`]: Represents a [Vulkan command buffer](vk::CommandBuffer).
//! - [`CommandEncoder`]: Encodes segments of a command buffer and batches pipeline barriers.
//! - [`CommandPool`]: Manages allocation, recording, and lifecycle of command buffers
//! - [`Timeline`](crate::sync::Timeline): Provides logical execution ordering for command buffers using timeline semaphores
//! - [`ResourceState`]: Tracking information that can be stored alongside a GPU resource for automatic
//!   resource tracking.
//!
//! ## Example Usage
//!
//! ```
//! # use pumicite::{Device, command::CommandPool, sync::Timeline};
//! # let (device, queue) = Device::create_system_default().unwrap();
//! // Create a command pool and timeline
//! let mut pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
//! let mut timeline = Timeline::new(device).unwrap();
//!
//! // Allocate and schedule a command buffer
//! let mut cmd = pool.alloc().unwrap();
//! timeline.schedule(&mut cmd);
//!
//! // Begin and record commands
//! pool.begin(&mut cmd).unwrap();
//! pool.record(&mut cmd, |encoder| {
//!     // Record your commands here
//! });
//!
//! // Finish recording
//! pool.finish(&mut cmd).unwrap();
//! ```
//!
//! [^nvperf]: [NVIDIA Tips and Tricks: Vulkan Dos and Don’t](https://developer.nvidia.com/blog/vulkan-dos-donts/)

mod compute;
mod render;
mod transfer;

use crate::{
    Device, HasDevice,
    buffer::BufferLike,
    image::ImageLike,
    sync::{GPUMutex, SharedSemaphore},
    tracking::{Access, MemoryBarrier, ResourceState},
    utils::AsVkHandle,
};
use ash::{VkResult, vk};
pub use render::*;
use std::{
    alloc::Layout,
    collections::BTreeMap,
    ops::{Deref, DerefMut, Range},
    pin::Pin,
    sync::{Arc, atomic::AtomicBool},
    task::Poll,
};

/// The `CommandEncoder` is responsible for recording commands into a Vulkan command buffer while
/// automatically managing pipeline barriers, resource transitions, and resource lifetime. It batches
/// barriers for optimal performance and ensures proper synchronization between operations.
///
/// ## Lifetime Parameter
///
/// The lifetime parameter `'a` represents the GPU execution lifetime - the duration for which
/// the command encoder (and its recorded commands) will be alive on the GPU timeline. This
/// enables safe resource retention and lifetime extension.
///
/// Command encoders are only ever provided as an argument to a closure. This ensures that the
/// actual lifetime that the command buffer was executed on the GPU always outlives 'a.
///
/// To obtain a reference with `'a` lifetime, call [`CommandEncoder::retain`]. Alternatively,
/// if you already have a [`GPUMutex`], call [`CommandEncoder::lock`]
///
/// ## Barrier Management
///
/// The encoder automatically batches memory barriers, image layout transitions, and buffer
/// barriers to minimize the number of pipeline barrier commands issued. Barriers are emitted
/// when:
/// - [`emit_barriers()`](Self::emit_barriers) is called explicitly
/// - During async future polling (for GPU futures)
pub struct CommandEncoder<'a> {
    /// Raw pointer to the command buffer being recorded into.
    pub(crate) buffer: *mut CommandBuffer,

    /// Accumulated global memory barriers that will be emitted in the next barrier batch.
    pending_memory_barrier: MemoryBarrier,

    /// Accumulated image memory barriers that will be emitted in the next barrier batch.
    pending_image_barrier: Vec<vk::ImageMemoryBarrier2<'a>>,

    /// Accumulated buffer memory barriers that will be emitted in the next barrier batch.
    pending_buffer_barrier: Vec<vk::BufferMemoryBarrier2<'a>>,

    render_pass_state: RenderPassState,
}
unsafe impl<'a> Send for CommandEncoder<'a> {}
unsafe impl<'a> Sync for CommandEncoder<'a> {}

enum RenderPassState {
    OutsideRenderPass,
    InsideRenderPass { render_area: vk::Rect2D },
}

impl CommandEncoder<'_> {
    /// Creates a new command encoder with no associated command buffer.
    ///
    /// # Safety
    ///
    /// The encoder must have a valid command buffer set via [`set_buffer()`](Self::set_buffer)
    /// before any recording operations are performed.
    pub unsafe fn new() -> Self {
        Self {
            buffer: std::ptr::null_mut(),
            pending_memory_barrier: MemoryBarrier::default(),
            pending_image_barrier: Vec::new(),
            pending_buffer_barrier: Vec::new(),
            render_pass_state: RenderPassState::OutsideRenderPass,
        }
    }

    /// Resets the command encoder to its initial state.
    ///
    /// Use this method to reset a command encoder instead of creating a new one to reuse
    /// the allocated memory.
    ///
    /// # Safety
    ///
    /// The encoder must have a valid command buffer set via [`set_buffer()`](Self::set_buffer)
    /// before any recording operations are performed.
    pub unsafe fn reset(&mut self) {
        self.pending_image_barrier.clear();
        self.pending_buffer_barrier.clear();
        self.pending_memory_barrier = MemoryBarrier::default();
        self.buffer = std::ptr::null_mut();
    }

    /// Returns a reference to the underlying command buffer.
    ///
    /// # Panics
    ///
    /// Panics if no command buffer has been set via [`set_buffer()`](Self::set_buffer).
    pub fn buffer(&self) -> &CommandBuffer {
        assert!(
            !self.buffer.is_null(),
            "CommandEncoder has no associated command buffer"
        );
        unsafe { &*self.buffer }
    }

    /// Sets the command buffer that this encoder will record into.
    ///
    /// # Safety
    ///
    /// The provided buffer pointer must:
    /// - Be valid for the lifetime of the encoder
    /// - Point to a properly initialized CommandBuffer in Recording state
    /// - Not be used by other encoders simultaneously
    pub unsafe fn set_buffer(&mut self, buffer: *mut CommandBuffer) {
        self.buffer = buffer;
    }

    /// Returns a mutable reference to the underlying command buffer.
    ///
    /// # Panics
    ///
    /// Panics if no command buffer has been set via [`set_buffer()`](Self::set_buffer).
    pub fn buffer_mut(&mut self) -> &mut CommandBuffer {
        assert!(
            !self.buffer.is_null(),
            "CommandEncoder has no associated command buffer"
        );
        unsafe { &mut *self.buffer }
    }

    /// Emits all pending barriers as a single pipeline barrier command.
    ///
    /// The pending barriers are accumulated from calls to [use_resource](`Self::use_resource`),
    /// [use_image_resource](`Self::use_image_resource`), [use_buffer_resource](`Self::use_buffer_resource`),
    /// and [memory_barrier](`Self::memory_barrier`)
    ///
    /// This batches all accumulated memory barriers, image barriers, and buffer barriers
    /// into a single `vkCmdPipelineBarrier2` call for optimal performance. After emission,
    /// all pending barriers are cleared.
    ///
    /// This method is typically called before render or dispatch commands to ensure that the resources
    /// used by these commands are transitioned into the correct states.
    pub fn emit_barriers(&mut self) {
        // Only emit barriers if there's actually something to synchronize
        if !self.has_pending_barriers() {
            return;
        }

        let memory_barrier = std::mem::take(&mut self.pending_memory_barrier);
        unsafe {
            self.device().cmd_pipeline_barrier2(
                self.buffer().buffer,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&self.pending_image_barrier)
                    .buffer_memory_barriers(&self.pending_buffer_barrier)
                    .memory_barriers(&[vk::MemoryBarrier2 {
                        src_stage_mask: memory_barrier.src.stage,
                        src_access_mask: memory_barrier.src.access,
                        dst_stage_mask: memory_barrier.dst.stage,
                        dst_access_mask: memory_barrier.dst.access,
                        ..Default::default()
                    }]),
            );
        }
        // Clear all pending barriers after emission
        self.pending_image_barrier.clear();
        self.pending_buffer_barrier.clear();
    }

    pub fn has_pending_barriers(&self) -> bool {
        self.pending_memory_barrier.src.stage != vk::PipelineStageFlags2::empty()
            || !self.pending_image_barrier.is_empty()
            || !self.pending_buffer_barrier.is_empty()
    }
}

/// GPU resource tracking / retention related command encoder methods.
///
/// The lifetime parameter `'a` represents the GPU execution lifetime - the duration for which
/// resources and locks will be retained during GPU execution.
impl<'a> CommandEncoder<'a> {
    /// Retains an object until the command buffer completes execution.
    ///
    /// This extends the lifetime of the object to match the GPU execution lifetime,
    /// ensuring it remains valid throughout command buffer execution. The object
    /// will be automatically released when the command buffer finishes.
    ///
    /// # Parameters
    /// - `arc`: The object to retain
    ///
    /// # Returns
    /// A reference with lifetime `'a` (GPU execution lifetime) to the retained object.
    ///
    /// # Example
    /// ```
    /// # use pumicite::{Device, command::CommandPool, sync::Timeline, Allocator, ash::vk, buffer::Buffer};
    /// # let (device, queue) = Device::create_system_default().unwrap();
    /// # let mut pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
    /// # let mut timeline = Timeline::new(device.clone()).unwrap();
    /// # let mut cmd = pool.alloc().unwrap();
    /// # let allocator = Allocator::new(device).unwrap();
    /// # timeline.schedule(&mut cmd);
    /// # pool.begin(&mut cmd).unwrap();
    /// # use std::sync::Arc;
    /// let data = Arc::new(Buffer::new_private(allocator, 128, 4, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap());
    /// pool.record(&mut cmd, |encoder| {
    ///     let retained = encoder.retain(data.clone());
    ///     // retained is guaranteed to live until GPU execution completes
    /// });
    /// # pool.finish(&mut cmd).unwrap();
    /// ```
    pub fn retain<T: Sized>(&mut self, arc: T) -> &'a T {
        unsafe {
            let ptr = self.buffer_mut().retainer.add(arc);

            // Safety: It's safe to dereference here because we guarantee that the retainer
            // won't be cleared until the command encoder finishes execution on the GPU.
            &*ptr
        }
    }

    /// Acquires a lock on a [`GPUMutex`] and blocks pipeline stages until the mutex is available.
    ///
    /// This method integrates GPU synchronization with the command buffer execution model.
    /// The specified pipeline stages on the current submission will be blocked until the
    /// mutex becomes available,
    /// and the mutex will remain locked until this command buffer completes execution.
    ///
    /// # Parameters
    /// - `res`: The GPU mutex to lock
    /// - `stages`: Pipeline stages that should wait for the mutex to become available
    ///
    /// # Returns
    /// A reference with GPU execution lifetime to the protected resource.
    ///
    /// # Synchronization Behavior
    /// - If the mutex is already available, no additional synchronization is needed
    /// - If the mutex is locked on another timeline, after submission the driver
    ///   will wait for that work to complete before it schedules this command buffer
    ///   for execution.
    /// - The mutex remains locked until this command buffer completes execution
    /// - If the mutex is dropped while locked, cleanup is deferred to avoid blocking
    pub fn lock<T: Send + Sync>(
        &mut self,
        res: &GPUMutex<T>,
        stages: vk::PipelineStageFlags2,
    ) -> &'a T {
        let (wait_semaphore, wait_value) = unsafe {
            // This GPUMutex is locked until self.buffer().semaphore
            // had its value set to self.buffer().timestamp
            res.lock_until(
                self.buffer().semaphore.as_ref().unwrap(),
                self.buffer().timestamp,
            )
        };

        if let Some(wait_semaphore) = wait_semaphore {
            // The command buffer needs to wait for this semaphore before executing.
            let entry = self
                .buffer_mut()
                .wait_semaphores
                .entry(wait_semaphore)
                .or_default();

            // Merge or update the wait condition based on semaphore values
            if entry.0 == wait_value {
                // Same wait value - combine the pipeline stages
                entry.1 |= stages;
            } else if entry.0 < wait_value {
                // Higher wait value - replace with new condition
                entry.0 = wait_value;
                entry.1 = stages;
            }
        } else {
            // wait_semaphore is None when the resource is being used for the first time.
            // We don't have to wait for anything if that's the case.
        }

        unsafe {
            // Safety: The lifetime extension is safe because the GPUMutex ensures
            // the resource remains valid until the command buffer completes execution.
            &*Box::as_ptr(&res.inner)
        }
    }

    /// Adds a global memory barrier between pipeline stages.
    ///
    /// This creates
    /// - An exeuction dependency such that `before.stage` completes before any
    ///   operations in `after.stage` begin
    /// - A memory dependency such that memory regions touched by `before.access` are flushed,
    ///   and memory regions that will be touched by `after.access` are invalidated.
    ///
    /// The barrier will be batched with other barriers and emitted when
    /// [`emit_barriers()`](Self::emit_barriers) is called.
    ///
    /// # Parameters
    /// - `before`: Source access that must complete first
    /// - `before`: Destination access that waits for source completion
    pub fn memory_barrier(&mut self, before: Access, after: Access) {
        self.pending_memory_barrier.src |= before;
        self.pending_memory_barrier.dst |= after;
    }

    /// Creates an image memory barrier.
    ///
    /// - Creates an exeuction dependency such that `before.stage` completes before anyy
    ///   operations in `after.stage` begin
    /// - Creates a memory dependency such that memory regions touched by `before.access` are flushed,
    ///   and memory regions that will be touched by `after.access` are invalidated.
    /// - Transition the image layout from `old_layout` to `new_layout`
    pub fn image_barrier(
        &mut self,
        image: &'a impl ImageLike,
        before: Access,
        after: Access,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_level_range: Range<u32>,
        array_layer_range: Range<u32>,
    ) {
        self.pending_image_barrier.push(vk::ImageMemoryBarrier2 {
            src_stage_mask: before.stage,
            src_access_mask: before.access,
            dst_stage_mask: after.stage,
            dst_access_mask: after.access,
            old_layout,
            new_layout,
            image: image.vk_handle(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: image.aspects(),
                base_array_layer: array_layer_range.start,
                layer_count: array_layer_range.len() as u32,
                base_mip_level: mip_level_range.start,
                level_count: mip_level_range.len() as u32,
            },
            ..Default::default()
        });
    }

    /// Transitions a resource to a new state and adds necessary global barriers.
    ///
    /// # Parameters
    /// - `state`: Mutable reference to the resource's current state
    /// - `access`: The new access pattern for the resource
    pub fn use_resource<T>(&mut self, state: &mut ResourceState, access: Access) {
        let memory_barrier = state.transition(access, false);
        self.pending_memory_barrier |= memory_barrier;
    }

    /// Transitions a buffer resource to a new state with fine-grained barrier tracking.
    ///
    /// This method creates buffer-specific memory barriers for fine-grained synchronization.
    /// However, this does come with increased CPU overhead that may not be worthwhile for most
    /// GPU. Call [use_resource()](`Self::use_resource`) instead of this method unless there's
    /// a measurable performance benefit.
    ///
    /// # Parameters
    /// - `resource`: The buffer resource being accessed. The resource must have GPU execution lifetime.
    ///   Call [retain()](Self::retain) or [lock()](Self::lock) to extend the lifetime of a resource.
    /// - `state`: Mutable reference to the resource's current state
    /// - `access`: The new access pattern for the resource
    pub fn use_buffer_resource<T: BufferLike>(
        &mut self,
        resource: &'a T,
        state: &mut ResourceState,
        access: Access,
    ) {
        let memory_barrier = state.transition(access, false);
        self.pending_buffer_barrier.push(vk::BufferMemoryBarrier2 {
            src_access_mask: memory_barrier.src.access,
            src_stage_mask: memory_barrier.src.stage,
            dst_access_mask: memory_barrier.dst.access,
            dst_stage_mask: memory_barrier.dst.stage,
            buffer: resource.vk_handle(),
            offset: resource.offset(),
            size: resource.size(),
            ..Default::default()
        })
    }

    /// Transitions an image resource to a new state.
    ///
    /// # Parameters
    /// - `resource`: The image resource being accessed. The resource must have GPU execution lifetime.
    ///   Call [retain()](Self::retain) or [lock()](Self::lock) to extend the lifetime of a resource.
    /// - `state`: Mutable reference to the [ResourceState] object tracking the specified mip levels and
    ///   array layers for this image.
    /// - `access`: The new access pattern for the resource
    /// - `layout`: The required image layout for the new access
    /// - `mip_level_range`: Range of mip levels affected by the transition
    /// - `array_layer_range`: Range of array layers affected by the transition
    /// - `discard_content`: If true, previous image content can be discarded (by setting
    ///   [vk::ImageMemoryBarrier2::old_layout] to [vk::ImageLayout::UNDEFINED]).
    ///   
    ///
    /// # Layout Transition Behavior
    /// - If the layout changes, an image memory barrier is created. Otherwise, a global memory barrier is used.
    /// - Discarding content can improve performance by saving unnecessary blits between different image layouts
    ///   and avoiding unneeded cache flush operations.
    pub fn use_image_resource<T: ImageLike>(
        &mut self,
        resource: &'a T,
        state: &mut ResourceState,
        access: Access,
        layout: vk::ImageLayout,
        mip_level_range: Range<u32>,
        array_layer_range: Range<u32>,
        discard_content: bool,
    ) {
        let with_layout_transition = state.layout != layout;
        let memory_barrier = state.transition(access, with_layout_transition);
        if with_layout_transition {
            // Layout transition requires image barrier
            self.pending_image_barrier.push(vk::ImageMemoryBarrier2 {
                src_stage_mask: memory_barrier.src.stage,
                src_access_mask: memory_barrier.src.access,
                dst_stage_mask: memory_barrier.dst.stage,
                dst_access_mask: memory_barrier.dst.access,
                old_layout: if discard_content {
                    // UNDEFINED layout allows the driver to discard previous content
                    vk::ImageLayout::UNDEFINED
                } else {
                    state.layout
                },
                new_layout: layout,
                image: resource.vk_handle(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: resource.aspects(),
                    base_mip_level: mip_level_range.start,
                    level_count: mip_level_range.end - mip_level_range.start,
                    base_array_layer: array_layer_range.start,
                    layer_count: array_layer_range.end - array_layer_range.start,
                },
                ..Default::default()
            });
        } else {
            // No layout change - use global memory barrier for reduced CPU overhead.
            self.pending_memory_barrier |= memory_barrier;
        }

        // Update the resource state with the new layout
        state.layout = layout;
    }
}
impl HasDevice for CommandEncoder<'_> {
    fn device(&self) -> &Device {
        self.buffer().device()
    }
}

/// Internal storage for retained objects during command buffer execution.
///
/// This structure holds raw pointers and their corresponding drop functions,
/// allowing heterogeneous object storage without type erasure overhead.
/// Objects are automatically dropped when the retainer is cleared or dropped.
#[derive(Default)]
struct ArcRetainer {
    meta: Vec<(unsafe fn(*mut ()), *mut u8)>,

    ptr: *mut u8,
    size: usize,
    chunks: Vec<*mut u8>,
    free_chunk: usize,
}
unsafe impl Send for ArcRetainer {}
unsafe impl Sync for ArcRetainer {}

impl ArcRetainer {
    const MAX_ALIGNMENT: usize = 64;
    const CHUNK_SIZE: usize = 4 * 1024; // 4KB page size

    unsafe fn alloc_chunk(&mut self) -> *mut u8 {
        if self.free_chunk < self.chunks.len() {
            let chunk_ptr = self.chunks[self.free_chunk];
            self.free_chunk += 1;
            chunk_ptr
        } else {
            let new_ptr = unsafe {
                std::alloc::alloc(
                    Layout::from_size_align(Self::CHUNK_SIZE, Self::MAX_ALIGNMENT).unwrap(),
                )
            };
            self.chunks.push(new_ptr);
            self.free_chunk += 1;
            new_ptr
        }
    }
    fn reserve_space(&mut self, layout: Layout) -> *mut u8 {
        let padding_needed = self.size.next_multiple_of(layout.align()) - self.size;
        let new_size = self.size + padding_needed + layout.size();
        if self.ptr.is_null() || new_size > Self::CHUNK_SIZE {
            // realloc
            unsafe {
                let new_ptr = self.alloc_chunk();
                self.ptr = new_ptr;
                self.size = 0;
            }
        }
        let ptr = unsafe { self.ptr.add(self.size + padding_needed) };
        self.size = new_size;
        ptr
    }
    /// Adds a retainable object to the storage.
    ///
    /// The object will be kept alive until [`clear()`](Self::clear) is called
    /// or the retainer is dropped.
    pub fn add<T: Sized>(&mut self, item: T) -> *mut T {
        assert!(std::mem::align_of::<T>() <= Self::MAX_ALIGNMENT);
        assert!(std::mem::size_of::<T>() <= Self::CHUNK_SIZE);
        let ptr = self.reserve_space(Layout::new::<T>());
        unsafe {
            std::ptr::copy_nonoverlapping(&item, ptr as *mut T, 1);
            std::mem::forget(item);
            let drop_ptr: unsafe fn(*mut T) = std::ptr::drop_in_place::<T>;
            self.meta.push((
                std::mem::transmute::<unsafe fn(*mut T), unsafe fn(*mut ())>(drop_ptr),
                ptr,
            ));

            ptr as *mut T
        }
    }

    /// Drops all retained objects and clears the storage.
    ///
    /// This calls the drop function for each retained object in the order
    /// they were added, then clears the internal storage.
    pub fn clear(&mut self) {
        unsafe {
            for (drop_fn, ptr) in self.meta.iter().cloned() {
                (drop_fn)(ptr as *mut ());
            }
        }
        self.meta.clear();
        self.size = 0;
        self.free_chunk = 0;
        self.ptr = std::ptr::null_mut();
    }

    pub fn is_unused(&self) -> bool {
        self.chunks.is_empty()
    }
}

impl Drop for ArcRetainer {
    fn drop(&mut self) {
        self.clear();

        for chunk in self.chunks.iter().cloned() {
            unsafe {
                std::alloc::dealloc(
                    chunk,
                    Layout::from_size_align_unchecked(Self::CHUNK_SIZE, Self::MAX_ALIGNMENT),
                );
            }
        }
    }
}

/// Represents the current state of a command buffer in its lifecycle.
///
/// Command buffers transition through these states as they are allocated,
/// recorded, submitted, and completed. The state determines which operations
/// are valid on the command buffer.
///
/// # State Transitions
///
/// ```text
/// Initial -> Recording -> Executable -> Pending -> Invalid -> Freed
///    |                                              |
///    +----------------<<<---------------- (reset) --+
/// ```
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum CommandBufferState {
    /// Newly allocated command buffer, ready for scheduling.
    Initial,

    /// Currently being recorded with commands.
    Recording,

    /// Recording finished, ready for submission to a queue.
    Executable,

    /// Submitted to a queue and currently executing on the GPU.
    Pending,

    /// Execution completed, can be reset or freed.
    Invalid,

    /// Returned to the command pool's free list.
    Freed,
}

/// A single-use Vulkan command buffer with managed synchronization and resource retention.
///
/// # Lifecycle
///
/// 1. **Allocation**: Created by [`CommandPool::alloc`]
/// 2. **Scheduling**: Prepared for recording via [`crate::sync::Timeline::schedule`]. The order of execution of a command buffer in regards to a Timeline
///    depends on the order that [`crate::sync::Timeline::schedule`] was called.
/// 3. **Begin**: Before recording, call [`CommandPool::begin`]
/// 3. **Recording**: Commands recorded via [`CommandPool::record()`]
/// 4. **Execution**: Submitted with [Queue::submit](crate::Queue::submit)
/// 5. **Completion**: The host confirms completion by calling
///    [CommandBuffer::block_until_completion] or [CommandBuffer::try_complete]
/// 6. **Cleanup**: Command buffers cannot be dropped. Either [reset](CommandPool::reset) for reuse or [free](CommandPool::free) it back to the pool.
///
/// # Resource Retention
///
/// The command buffer may retain resources used during
/// recording, ensuring they remain valid throughout GPU execution. These resources
/// are automatically released when execution completes. See [CommandEncoder::retain].
///
/// # Synchronization
///
/// Command buffers syncronize with the host and other submissions using timeline semaphores.
/// Each command buffer is assigned a timestamp and will signal a timeline semaphore to that timestamp
/// when execution finishes.
pub struct CommandBuffer {
    /// Reference to the command pool that allocated this buffer.
    ///
    /// This ensures the command pool remains alive as long as there are
    /// outstanding command buffers, preventing premature cleanup.
    /// The command pool should never be accessed mutably through this reference.
    pool: Arc<CommandPoolInner>,

    /// The underlying Vulkan command buffer handle.
    pub(crate) buffer: vk::CommandBuffer,

    /// Current state in the command buffer lifecycle.
    pub(crate) state: CommandBufferState,

    /// Timeline semaphore that will be signaled upon command buffer completion.
    ///
    /// This is set when the command buffer is scheduled onto a timeline and
    /// cleared when execution completes.
    pub(crate) semaphore: Option<SharedSemaphore>,

    /// Timestamp at which the semaphore will be signaled.
    ///
    /// This corresponds to the command buffer's position in the timeline that it was [scheduled](CommandPool::schedule) on
    pub(crate) timestamp: u64,

    /// Semaphores that this command buffer must wait for before execution.
    ///
    /// Maps semaphore handles to (wait_value, pipeline_stages) pairs to ensure that we uniquely
    /// wait on a semaphore only once.
    /// The command buffer will wait for each semaphore to reach the specified
    /// value before allowing the specified pipeline stages to execute.
    pub(crate) wait_semaphores: BTreeMap<SharedSemaphore, (u64, vk::PipelineStageFlags2)>,

    /// Storage for objects that must remain alive during command buffer execution.
    retainer: ArcRetainer,
}
impl AsVkHandle for CommandBuffer {
    type Handle = vk::CommandBuffer;

    fn vk_handle(&self) -> Self::Handle {
        self.buffer
    }
}
impl HasDevice for CommandBuffer {
    fn device(&self) -> &Device {
        &self.pool.device
    }
}
impl CommandBuffer {
    /// Returns the current state of the command buffer.
    ///
    /// The state indicates what operations are currently valid on this command buffer
    /// and where it is in its lifecycle.
    pub fn state(&self) -> CommandBufferState {
        self.state
    }

    /// Blocks the current thread until the command buffer completes execution on the GPU.
    ///
    /// This method will wait for the command buffer to finish executing.
    /// Once execution completes, the command buffer transitions to the [`Invalid`](CommandBufferState::Invalid)
    /// state and all retained resources are released.
    ///
    /// # Returns
    /// - `Ok(())` if the command buffer completed successfully or was not pending
    /// - `Err(VkResult)` if there was an error waiting for completion
    ///
    /// # Behavior
    /// - Panics if the command buffer hasn't been submitted for execution or was freed
    /// - Multiple calls to this method is OK. Subsequent calls will return immediately.
    ///
    /// # Example
    /// ```
    /// # use pumicite::{Device, command::{CommandPool, CommandBufferState}, sync::Timeline};
    /// # let (device, mut queue) = Device::create_system_default().unwrap();
    /// # let mut pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
    /// # let mut timeline = Timeline::new(device).unwrap();
    /// # let mut cmd = pool.alloc().unwrap();
    /// # timeline.schedule(&mut cmd);
    /// # pool.begin(&mut cmd).unwrap();
    /// # pool.record(&mut cmd, |_encoder| {});
    /// # pool.finish(&mut cmd).unwrap();
    /// # queue.submit(&mut cmd).unwrap();
    /// assert_eq!(cmd.state(), CommandBufferState::Pending);
    ///
    /// // Wait for completion
    /// cmd.block_until_completion().unwrap();
    /// assert_eq!(cmd.state(), CommandBufferState::Invalid);
    /// ```
    pub fn block_until_completion(&mut self) -> VkResult<()> {
        match self.state {
            CommandBufferState::Pending => (),
            CommandBufferState::Invalid => return Ok(()),
            _ => panic!("Command buffer must be recorded and submitted for execution first"),
        }

        // Wait for the timeline semaphore to reach our timestamp
        self.semaphore
            .as_ref()
            .unwrap()
            .wait_blocked(self.timestamp, !0)?;

        // Clean up and transition to invalid state
        self.state = CommandBufferState::Invalid;
        self.semaphore = None;
        self.retainer.clear();
        self.wait_semaphores.clear();
        self.timestamp = 0;
        Ok(())
    }

    pub async fn block_async_until_completion(&mut self) -> VkResult<()> {
        match self.state {
            CommandBufferState::Pending => (),
            CommandBufferState::Invalid => return Ok(()),
            _ => panic!("Command buffer must be recorded and submitted for execution first"),
        }
        // Wait for the timeline semaphore to reach our timestamp
        self.semaphore
            .as_ref()
            .unwrap()
            .wait_async(self.timestamp)
            .await?;

        // Clean up and transition to invalid state
        self.state = CommandBufferState::Invalid;
        self.semaphore = None;
        self.retainer.clear();
        self.wait_semaphores.clear();
        self.timestamp = 0;
        Ok(())
    }

    /// Checks if the command buffer has completed execution without blocking.
    ///
    /// This method polls the timeline semaphore to determine if execution has finished.
    /// If the command buffer has completed, it automatically transitions the command buffer to the
    /// [`Invalid`](CommandBufferState::Invalid) state and cleans up resources.
    ///
    /// # Returns
    /// - `true` if the command buffer has completed execution
    /// - `false` if the command buffer is still executing
    ///
    /// # Example
    /// ```
    /// # use pumicite::{Device, command::{CommandPool, CommandBufferState}, sync::Timeline};
    /// # let (device, mut queue) = Device::create_system_default().unwrap();
    /// # let mut pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
    /// # let mut timeline = Timeline::new(device).unwrap();
    /// # let mut cmd = pool.alloc().unwrap();
    /// # timeline.schedule(&mut cmd);
    /// # pool.begin(&mut cmd).unwrap();
    /// # pool.record(&mut cmd, |_encoder| {});
    /// # pool.finish(&mut cmd).unwrap();
    /// # queue.submit(&mut cmd).unwrap();
    /// assert_eq!(cmd.state(), CommandBufferState::Pending);
    ///
    /// // Poll for completion
    /// while !cmd.try_complete() {
    ///     // Do other work...
    /// }
    /// assert_eq!(cmd.state(), CommandBufferState::Invalid);
    /// ```
    pub fn try_complete(&mut self) -> bool {
        if self.state == CommandBufferState::Invalid {
            return true;
        }
        if self.state != CommandBufferState::Pending {
            return false;
        }

        let completed = self.semaphore.as_ref().unwrap().is_signaled(self.timestamp);
        if completed {
            // Clean up and transition to invalid state
            self.state = CommandBufferState::Invalid;
            self.semaphore = None;
            self.retainer.clear();
            self.wait_semaphores.clear();
            self.timestamp = 0;
        }
        completed
    }
}
impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if self
            .pool
            .command_pool_dropped
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        match self.state {
            CommandBufferState::Pending => {
                self.block_until_completion().unwrap();
                tracing::error!(
                    "Dropping CommandBuffer {:?} without waiting for its completion",
                    self.buffer
                );
            }
            CommandBufferState::Freed => (),
            _ => {
                tracing::warn!(
                    "Dropping CommandBuffer {:?} without returning it to the CommandPool {:?}",
                    self.buffer,
                    self.pool.handle
                );
            }
        }
    }
}

/// A pool for allocating and managing resources used by Vulkan command buffers.
///
/// `CommandPool` provides a high-level interface for command buffer lifecycle management,
/// including allocation, recording, scheduling, and cleanup.
///
/// # Thread Safety
///
/// Command pools are not thread-safe and should be used from a single thread.
/// Command recording is also not thread-safe and requires mutable reference to the pool from
/// which the command buffer was allocated from.
/// However, command buffers can be safely moved between threads once encoding has finished.
pub struct CommandPool {
    /// Shared inner state that can be referenced by outstanding command buffers.
    inner: Arc<CommandPoolInner>,

    /// Pool of reusable retainer storage to avoid repeated allocations.
    /// When command buffers are freed or reset, their retainers are moved here
    /// for reuse by future command buffers.
    retainer_pool: Vec<ArcRetainer>,
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        // Signal to outstanding command buffers that the pool has been dropped.
        // This prevents warning messages when command buffers are dropped
        // after the pool, since users can't free them in that case.
        self.inner
            .command_pool_dropped
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl AsVkHandle for CommandPool {
    type Handle = vk::CommandPool;

    fn vk_handle(&self) -> Self::Handle {
        self.inner.handle
    }
}

impl HasDevice for CommandPool {
    fn device(&self) -> &Device {
        &self.inner.device
    }
}

/// Internal shared state for a command pool.
///
/// This structure is shared between the command pool and all command buffers
/// allocated from it, ensuring the Vulkan command pool remains valid as long
/// as there are outstanding command buffers.
struct CommandPoolInner {
    /// The device that owns this command pool.
    device: Device,

    /// The underlying Vulkan command pool handle.
    handle: vk::CommandPool,

    /// Flag indicating whether the main CommandPool has been dropped.
    ///
    /// This is used to suppress warnings when command buffers are dropped
    /// after their parent pool. In such cases, users cannot free the command
    /// buffers properly, so warnings would be misleading.
    command_pool_dropped: AtomicBool,
}

impl Drop for CommandPoolInner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.handle, None);
        }
    }
}

pub struct CommandEncoderGuard<'a> {
    pool: &'a mut CommandPool,
    /// Buffer is in a box to ensure the validity of the pointer from CommandEncoder.
    /// This allows [`CommandEncoderGuard`] to be Unpin.
    buffer: Option<Box<CommandBuffer>>,
    encoder: CommandEncoder<'a>,
}
impl<'a> CommandEncoderGuard<'a> {
    pub fn finish(mut self) -> VkResult<CommandBuffer> {
        self.encoder.emit_barriers();
        let buffer = self.buffer.take().unwrap();
        Ok(*buffer)
    }
}
impl<'a> Drop for CommandEncoderGuard<'a> {
    fn drop(&mut self) {
        if let Some(cb) = self.buffer.take() {
            self.pool.free(*cb);
        }
    }
}
impl<'a> Deref for CommandEncoderGuard<'a> {
    type Target = CommandEncoder<'a>;

    fn deref(&self) -> &Self::Target {
        &self.encoder
    }
}
impl<'a> DerefMut for CommandEncoderGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.encoder
    }
}

impl CommandPool {
    /// Creates a new command pool for a specific queue family
    pub fn new(device: Device, queue_family_index: u32) -> VkResult<Self> {
        Self::new_with_flags(
            device,
            queue_family_index,
            vk::CommandPoolCreateFlags::TRANSIENT,
        )
    }
    /// Creates a new resettable command pool for a specific queue family.
    ///
    /// Command buffers allocated from this command pool can be individually resetted.
    pub fn new_resettable(device: Device, queue_family_index: u32) -> VkResult<Self> {
        Self::new_with_flags(
            device,
            queue_family_index,
            vk::CommandPoolCreateFlags::TRANSIENT
                | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )
    }
    fn new_with_flags(
        device: Device,
        queue_family_index: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> VkResult<Self> {
        unsafe {
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags,
                    queue_family_index,
                    ..Default::default()
                },
                None,
            )?;
            Ok(Self {
                inner: Arc::new(CommandPoolInner {
                    device,
                    handle: pool,
                    command_pool_dropped: AtomicBool::new(false),
                }),
                retainer_pool: Vec::new(),
            })
        }
    }
    pub fn record_with_guard(&mut self, command_buffer: CommandBuffer) -> CommandEncoderGuard<'_> {
        assert_eq!(
            command_buffer.state,
            CommandBufferState::Recording,
            "Must call CommandPool::begin before recording a command buffer"
        );
        assert!(
            command_buffer.semaphore.is_some(),
            "Must call Timeline::schedule before recording a command buffer"
        );
        assert!(
            Arc::ptr_eq(&command_buffer.pool, &self.inner),
            "Command buffer recorded on the wrong pool!"
        );
        let mut command_buffer = Box::new(command_buffer);

        let encoder = CommandEncoder {
            buffer: &mut *command_buffer,
            pending_memory_barrier: MemoryBarrier::default(),
            pending_image_barrier: Vec::new(),
            pending_buffer_barrier: Vec::new(),
            render_pass_state: RenderPassState::OutsideRenderPass,
        };

        CommandEncoderGuard {
            buffer: Some(command_buffer),
            encoder,
            pool: self,
        }
    }
    /// Record commands using the provided [`CommandEncoder`].
    ///
    /// This method is defined on the CommandPool to ensure that the caller has a mutable reference to the pool.
    pub fn record<T>(
        &mut self,
        command_buffer: &mut CommandBuffer,
        callback: impl FnOnce(&mut CommandEncoder) -> T,
    ) -> T {
        assert_eq!(
            command_buffer.state,
            CommandBufferState::Recording,
            "Must call CommandPool::begin before recording a command buffer"
        );
        assert!(
            command_buffer.semaphore.is_some(),
            "Must call Timeline::schedule before recording a command buffer"
        );
        assert!(
            Arc::ptr_eq(&command_buffer.pool, &self.inner),
            "Command buffer recorded on the wrong pool!"
        );

        let mut encoder = CommandEncoder {
            buffer: command_buffer,
            pending_memory_barrier: MemoryBarrier::default(),
            pending_image_barrier: Vec::new(),
            pending_buffer_barrier: Vec::new(),
            render_pass_state: RenderPassState::OutsideRenderPass,
        };

        (callback)(&mut encoder)
    }
    /// Experimental. Record commands using the provided [`CommandEncoder`] in a Future.
    ///
    /// This method is defined on the CommandPool to ensure that the caller has a mutable reference to the pool.
    pub fn record_future<T: Send>(
        &mut self,
        command_buffer: &mut CommandBuffer,
        future: impl AsyncFnOnce(&mut CommandEncoder) -> T,
    ) -> GPUMutex<T> {
        assert_eq!(
            command_buffer.state,
            CommandBufferState::Recording,
            "Must call CommandPool::begin before recording a command buffer"
        );
        assert!(
            command_buffer.semaphore.is_some(),
            "Must call Timeline::schedule before recording a command buffer"
        );
        assert!(
            Arc::ptr_eq(&command_buffer.pool, &self.inner),
            "Command buffer recorded on the wrong pool!"
        );

        let semaphore = command_buffer.semaphore.as_ref().unwrap().clone();
        let timestamp = command_buffer.timestamp;

        self.record(command_buffer, move |mut encoder| {
            // Safety: casting `encoder` so that we can borrow it later for `encoder.emit_barriers()`
            let future = future(unsafe { &mut *std::ptr::addr_of_mut!(encoder) });
            let mut future = std::pin::pin!(future);

            let result = loop {
                match gpu_future_poll(future.as_mut()) {
                    Poll::Ready(result) => {
                        break result;
                    }
                    Poll::Pending => {
                        encoder.emit_barriers();
                    }
                }
            };
            GPUMutex::new_locked(Box::new(result), semaphore, timestamp)
        })
    }
    /// Allocate a new command buffer from the pool.
    pub fn alloc(&mut self) -> VkResult<CommandBuffer> {
        unsafe {
            let mut command_buffer = vk::CommandBuffer::null();
            (self.inner.device.fp_v1_0().allocate_command_buffers)(
                self.inner.device.handle(),
                &vk::CommandBufferAllocateInfo {
                    command_pool: self.inner.handle,
                    command_buffer_count: 1,
                    level: vk::CommandBufferLevel::PRIMARY,
                    ..Default::default()
                },
                &mut command_buffer,
            )
            .result()?;
            Ok(CommandBuffer {
                pool: self.inner.clone(),
                buffer: command_buffer,
                state: CommandBufferState::Initial,
                semaphore: None,
                timestamp: 0,
                wait_semaphores: BTreeMap::new(),
                retainer: self.retainer_pool.pop().unwrap_or_default(),
            })
        }
    }
    /// Begin recording a command buffer. Must be called before calling [`CommandPool::record`] or [`CommandPool::record_future`].
    pub fn begin(&mut self, cb: &mut CommandBuffer) -> VkResult<()> {
        assert_eq!(
            cb.state,
            CommandBufferState::Initial,
            "This command buffer has already been scheduled!"
        );
        assert!(
            Arc::ptr_eq(&cb.pool, &self.inner),
            "Command buffer beginning on the wrong pool!"
        );
        cb.state = CommandBufferState::Recording;
        unsafe {
            self.device().begin_command_buffer(
                cb.buffer,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )
        }
    }
    /// End recording on a command buffer. Must be called before submitting the command buffer.
    pub fn finish(&mut self, cb: &mut CommandBuffer) -> VkResult<()> {
        assert!(
            Arc::ptr_eq(&cb.pool, &self.inner),
            "Command buffer finished on the wrong pool!"
        );
        unsafe {
            self.device().end_command_buffer(cb.buffer)?;
            cb.state = CommandBufferState::Executable;
            Ok(())
        }
    }
    /// Returning a command buffer to the pool.
    pub fn free(&mut self, mut command_buffer: CommandBuffer) {
        assert!(
            Arc::ptr_eq(&command_buffer.pool, &self.inner),
            "Command buffer returned to the wrong pool!"
        );
        assert_ne!(
            command_buffer.state,
            CommandBufferState::Pending,
            "Command buffer is still being executed!"
        );
        command_buffer.retainer.clear();
        if !command_buffer.retainer.is_unused() {
            self.retainer_pool
                .push(std::mem::take(&mut command_buffer.retainer));
        }
        unsafe {
            self.device()
                .free_command_buffers(self.inner.handle, &[command_buffer.buffer]);
        }
        command_buffer.state = CommandBufferState::Freed;
    }

    /// Resets a command buffer. The command pool must be created with [`CommandPool::new_resettable`]
    pub fn reset(&mut self, command_buffer: &mut CommandBuffer) {
        assert!(
            Arc::ptr_eq(&command_buffer.pool, &self.inner),
            "Command buffer resetting on the wrong pool!"
        );
        assert_ne!(
            command_buffer.state,
            CommandBufferState::Pending,
            "Command buffer is still being executed!"
        );
        command_buffer.retainer.clear();
        command_buffer.wait_semaphores.clear();
        command_buffer.state = CommandBufferState::Initial;
        if !command_buffer.retainer.is_unused() {
            self.retainer_pool
                .push(std::mem::take(&mut command_buffer.retainer));
        }
        unsafe {
            self.device()
                .reset_command_buffer(command_buffer.buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }
    }
}

pub(crate) fn gpu_future_poll<T: Future>(gpu_future: Pin<&mut T>) -> Poll<T::Output> {
    use std::task::{RawWaker, RawWakerVTable, Waker};

    fn null_waker_clone_fn(_ptr: *const ()) -> RawWaker {
        panic!("GPU Futures cannot be executed from regular async executors");
    }
    fn null_waker_fn(_ptr: *const ()) {
        panic!("GPU Futures cannot be executed from regular async executors");
    }

    const NULL_WAKER_VTABLE: &RawWakerVTable = &RawWakerVTable::new(
        null_waker_clone_fn, // clone
        null_waker_fn,       // wake
        null_waker_fn,       // wake_by_ref
        null_waker_fn,       // drop
    );
    const NULL_WAKER: &Waker = unsafe { &Waker::new(std::ptr::null(), NULL_WAKER_VTABLE) };

    // Create a context with the null waker and poll the future
    let mut ctx = std::task::Context::from_waker(NULL_WAKER);
    gpu_future.poll(&mut ctx)
}
