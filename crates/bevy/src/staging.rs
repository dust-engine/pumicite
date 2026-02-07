//! GPU memory staging and transfer utilities.
//!
//! This module provides ring buffer allocators and async transfer infrastructure
//! for efficiently uploading data to the GPU in a Bevy application.
//!
//! # Ring Buffers
//!
//! Three specialized ring buffers are provided for transient data, each optimized for a
//!  different use cases:
//!
//! - [`DeviceLocalRingBuffer`]: For transient device-local buffers, like acceleration structures,
//!   and scratch buffers.
//!
//! - [`UniformRingBuffer`]: For uniform buffers. Uses `DEVICE_LOCAL`, `HOST_VISIBLE` memory
//!   on all platforms.
//!
//! - [`HostVisibleRingBuffer`]: For staging buffers. Always `HOST_VISIBLE`, never `DEVICE_LOCAL`.
//!
//! # Buffer Initialization
//!
//! The [`BufferInitializer`] system parameter provides convenient ways to create preinitialized
//! device-local buffers, automatically handling the staging path when direct writes aren't available.
//!
//! # Async Transfers
//!
//! [`AsyncTransfer`] enables background uploads on a dedicated transfer queue.

use std::{
    alloc::Layout,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use async_lock::Mutex;
use bevy_app::Plugin;
use bevy_ecs::{
    resource::Resource,
    system::{ResMut, SystemParam},
    world::FromWorld,
};

use pumicite::{
    ash::{self, VkResult, vk},
    buffer::{RingBuffer, RingBufferSuballocation},
    command::{CommandEncoderGuard, CommandPool},
    prelude::*,
    sync::Timeline,
};

use crate::{
    PumiciteApp,
    queue::{QueueWorldExt, SharedQueue, TransferQueue},
};

/// Bevy plugin that initializes the staging belt infrastructure.
///
/// This plugin creates the three ring buffer resources ([`DeviceLocalRingBuffer`],
/// [`UniformRingBuffer`], [`HostVisibleRingBuffer`]) and the [`AsyncTransfer`] resource.
///
/// # Configuration
///
/// Chunk sizes can be customized. Larger chunks reduce allocation overhead but
/// may consume more memory upfront.
///
/// ```ignore
/// app.add_plugins(StagingBeltPlugin {
///     device_local_chunk_size: 64 * 1024 * 1024,  // 64MB for device-local
///     uniform_chunk_size: 512 * 1024,              // 512KB for uniforms
///     host_visible_chunk_size: 64 * 1024 * 1024,   // 64MB for staging
/// });
/// ```
pub struct StagingBeltPlugin {
    pub device_local_chunk_size: u32,
    pub uniform_chunk_size: u32,
    pub host_visible_chunk_size: u32,
}
impl Default for StagingBeltPlugin {
    fn default() -> Self {
        Self {
            device_local_chunk_size: 64 * 1024 * 1024,
            uniform_chunk_size: 512 * 1024,
            host_visible_chunk_size: 64 * 1024 * 1024,
        }
    }
}
impl Plugin for StagingBeltPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        // Needed for the Uploader which requests buffer device address automatically.
        app.enable_feature::<vk::PhysicalDeviceBufferDeviceAddressFeatures>(|x| {
            &mut x.buffer_device_address
        })
        .unwrap();
    }
    fn finish(&self, app: &mut bevy_app::App) {
        let device = app.world().resource::<Device>().clone();

        let device_local_ring_buffer =
            DeviceLocalRingBuffer::new(device.clone(), self.device_local_chunk_size).unwrap();
        let host_visible_ring_buffer =
            HostVisibleRingBuffer::new(device.clone(), self.host_visible_chunk_size).unwrap();
        let uniform_ring_buffer =
            UniformRingBuffer::new(device.clone(), self.uniform_chunk_size).unwrap();
        app.world_mut().insert_resource(device_local_ring_buffer);
        app.world_mut().insert_resource(host_visible_ring_buffer);
        app.world_mut().insert_resource(uniform_ring_buffer);
        app.world_mut().init_resource::<AsyncTransfer>();
    }
}

/// Ring buffer for device-local GPU data.
///
/// Supported usages:
/// - [`vk::BufferUsageFlags::STORAGE_BUFFER`]
/// - [`vk::BufferUsageFlags::TRANSFER_DST`]
/// - [`vk::BufferUsageFlags::INDEX_BUFFER`]
/// - [`vk::BufferUsageFlags::VERTEX_BUFFER`]
/// - [`vk::BufferUsageFlags::VERTEX_BUFFER`]
/// - [`vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS`]
/// - [`vk::BufferUsageFlags::UNIFORM_BUFFER`]
#[derive(Resource)]
pub struct DeviceLocalRingBuffer(RingBuffer);

impl Deref for DeviceLocalRingBuffer {
    type Target = RingBuffer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for DeviceLocalRingBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl DeviceLocalRingBuffer {
    pub fn new(device: Device, chunk_size: u32) -> VkResult<Self> {
        let memory_type_index = device
            .physical_device()
            .properties()
            .memory_type_map()
            .private;

        let mut flags = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::UNIFORM_BUFFER;
        if device
            .get_extension::<ash::khr::acceleration_structure::Meta>()
            .is_ok()
        {
            flags |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
            flags |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR;
        }

        if device
            .get_extension::<ash::khr::ray_tracing_pipeline::Meta>()
            .is_ok()
        {
            flags |= vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR;
        }
        // By default, 64MB page size.
        Ok(Self(RingBuffer::new(
            device,
            chunk_size as u64,
            memory_type_index,
            flags,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            "DeviceLocalRingBuffer",
        )))
    }
}

/// Ring buffer for small host-visible uniform buffers.
///
/// Uses host-visible device-local memory when possible, enabling direct
/// CPU writes without staging.
#[derive(Resource)]
pub struct UniformRingBuffer(RingBuffer);

impl Deref for UniformRingBuffer {
    type Target = RingBuffer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for UniformRingBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl UniformRingBuffer {
    pub fn new(device: Device, chunk_size: u32) -> VkResult<Self> {
        let memory_type_index = device
            .physical_device()
            .properties()
            .memory_type_map()
            .uniform;

        if memory_type_index == u32::MAX {
            return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        }

        // uniform memory type is always DEVICE_LOCAL + HOST_VISIBLE
        let flags = vk::BufferUsageFlags::UNIFORM_BUFFER;
        // By default, 512KB page size.
        Ok(Self(RingBuffer::new(
            device,
            chunk_size as u64,
            memory_type_index,
            flags,
            vk::MemoryAllocateFlags::empty(),
            "UniformRingBuffer",
        )))
    }
}
impl UniformRingBuffer {
    /// Creates a uniform buffer with the given data.
    ///
    /// If the memory is host-visible, writes directly. Otherwise, uses
    /// `vkCmdUpdateBuffer` to copy the data inline in the command buffer.
    ///
    /// The buffer is retained by the encoder and remains valid for the
    /// lifetime of the command buffer.
    pub fn create_uniform<'a>(
        &mut self,
        encoder: &mut CommandEncoder<'a>,
        data: &[u8],
    ) -> &'a RingBufferSuballocation {
        let alignment = self
            .0
            .device()
            .physical_device()
            .properties()
            .limits
            .min_uniform_buffer_offset_alignment;
        let mut buffer = self.allocate_buffer(data.len() as u64, alignment);
        if let Some(slice) = buffer.as_slice_mut() {
            slice.copy_from_slice(data);
            encoder.retain(buffer)
        } else {
            let buffer = encoder.retain(buffer);
            encoder.update_buffer(buffer, data);
            buffer
        }
    }
}

/// Ring buffer for host-visible staging buffers.
///
/// Used mostly for staging data before copying to device-local memory.
/// Always placed in system RAM (`HOST_VISIBLE`), never `DEVICE_LOCAL`.
///
/// Default chunk size: 64MB.
#[derive(Resource)]
pub struct HostVisibleRingBuffer(RingBuffer);

impl Deref for HostVisibleRingBuffer {
    type Target = RingBuffer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for HostVisibleRingBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl HostVisibleRingBuffer {
    pub fn new(device: Device, chunk_size: u32) -> VkResult<Self> {
        let memory_type_index = device.physical_device().properties().memory_type_map().host;

        let flags = vk::BufferUsageFlags::TRANSFER_SRC;
        // By default, 64MB page size.
        Ok(Self(RingBuffer::new(
            device,
            chunk_size as u64,
            memory_type_index,
            flags,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            "HostVisibleRingBuffer",
        )))
    }
}

/// System parameter for creating pre-initialized device-local buffers.
///
/// Provides access to both [`HostVisibleRingBuffer`] and [`DeviceLocalRingBuffer`],
/// utomatically selecting the optimal upload path based on memory visibility:
/// - Direct write if device memory is host-visible (ReBar, integrated)
/// - Staging copy otherwise (discrete GPUs)
#[derive(SystemParam)]
pub struct BufferInitializer<'w> {
    /// Staging buffer for indirect uploads (used on discrete GPUs).
    pub host_buffer: ResMut<'w, HostVisibleRingBuffer>,
    /// Target buffer for GPU data.
    pub device_buffer: ResMut<'w, DeviceLocalRingBuffer>,
}
impl BufferInitializer<'_> {
    /// Creates a GPU buffer initialized with data, returning a [`GPUMutex`] for synchronization.
    ///
    /// If device memory is host-visible, writes directly. Otherwise, allocates a staging
    /// buffer, writes to it, and records a copy command.
    pub fn create_preinitialized_buffer(
        &mut self,
        encoder: &mut CommandEncoder,
        layout: Layout,
        writer: impl FnOnce(&mut [u8]),
    ) -> GPUMutex<RingBufferSuballocation> {
        assert!(!encoder.inside_renderpass());
        let mut buffer = self
            .device_buffer
            .allocate_buffer(layout.size() as u64, layout.align() as u64);
        if let Some(slice) = buffer.as_slice_mut() {
            if layout.size() > 0 {
                writer(slice);
            }
            GPUMutex::new(buffer)
        } else {
            let buffer = GPUMutex::new(buffer);
            if layout.size() > 0 {
                let mut host_buffer = self
                    .host_buffer
                    .allocate_buffer(layout.size() as u64, layout.align() as u64);
                writer(host_buffer.as_slice_mut().unwrap());
                let host_buffer = encoder.retain(host_buffer);
                let locked_buffer = encoder.lock(&buffer, vk::PipelineStageFlags2::COPY);
                encoder.copy_buffer(host_buffer, locked_buffer);
            }
            buffer
        }
    }

    /// Creates a GPU buffer initialized with data, retained by the command encoder.
    ///
    /// Similar to [`create_preinitialized_buffer`](Self::create_preinitialized_buffer),
    /// but the buffer is retained by the encoder rather than wrapped in a [`GPUMutex`].
    pub fn create_preinitialized_buffer_retained<'a>(
        &mut self,
        ctx: &mut CommandEncoder<'a>,
        layout: Layout,
        writer: impl FnOnce(&mut [u8]),
    ) -> &'a RingBufferSuballocation {
        assert!(!ctx.inside_renderpass());
        let mut buffer = self
            .device_buffer
            .allocate_buffer(layout.size() as u64, layout.align() as u64);
        if let Some(slice) = buffer.as_slice_mut() {
            if layout.size() > 0 {
                writer(slice);
            }
            ctx.retain(buffer)
        } else {
            let buffer = ctx.retain(buffer);
            if layout.size() > 0 {
                let mut host_buffer = self
                    .host_buffer
                    .allocate_buffer(layout.size() as u64, layout.align() as u64);
                writer(host_buffer.as_slice_mut().unwrap());

                let host_buffer = ctx.retain(host_buffer);
                ctx.copy_buffer(host_buffer, buffer);
            }
            buffer
        }
    }
}

/// Resource for performing async data transfers on a dedicated queue.
///
/// Uses a separate transfer queue (when available) to overlap data uploads with
/// rendering work. Manages its own command pool and timeline for synchronization.
///
/// # Usage
///
/// ```ignore
/// async fn upload_data(transfer: Res<AsyncTransfer>) {
///     let mut batch = transfer.batch().await?;
///     // Record transfer commands...
///     batch.submit().await?;
/// }
/// ```
#[derive(Clone, Resource)]
pub struct AsyncTransfer(Arc<AsyncTransferInner>);
struct AsyncTransferInner {
    queue: SharedQueue,
    command_pool: Mutex<AsyncTransferCommandContext>,
}

impl FromWorld for AsyncTransfer {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        let queue = world.make_shared_queue::<TransferQueue>();
        let device = world.resource::<Device>().clone();
        let pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
        let timeline = Timeline::new(device).unwrap();
        Self(Arc::new(AsyncTransferInner {
            queue,
            command_pool: Mutex::new(AsyncTransferCommandContext { timeline, pool }),
        }))
    }
}

struct AsyncTransferCommandContext {
    pool: CommandPool,
    timeline: Timeline,
}

/// Guard for an active async transfer batch.
///
/// Derefs to [`CommandEncoder`] for recording transfer commands. When finished,
/// call [`submit`](Self::submit) to execute the transfers asynchronously.
///
/// Optionally call [`flush`](Self::flush) to submit partial work and free staging
/// memory during long upload sequences.
pub struct AsyncTransferGuard<'a> {
    inner: &'a Arc<AsyncTransferInner>,
    encoder: CommandEncoderGuard<'a>,
    lock: async_lock::MutexGuard<'a, AsyncTransferCommandContext>,

    pending_task: Option<bevy_tasks::Task<VkResult<CommandBuffer>>>,
}

impl AsyncTransferGuard<'_> {
    /// Submits recorded work so far to free staging memory.
    ///
    /// This is a hint to the implementation - it may submit all work, some work,
    /// or nothing at all depending on whether previous work has completed.
    /// Call this periodically during long upload sequences to avoid exhausting
    /// staging buffer space.
    pub async fn flush(&mut self) -> VkResult<()> {
        if let Some(pending_task) = self.pending_task.as_ref() {
            if pending_task.is_finished() {
                let task = self.pending_task.take().unwrap();
                let cb = task.await?;
                self.lock.pool.free(cb);
            } else {
                // Still some pending work ongoing.
                return Ok(());
            }
        }

        let mut new_cb = self.lock.pool.alloc()?;
        self.lock.timeline.schedule(&mut new_cb);
        self.lock.pool.begin(&mut new_cb)?;
        let new_encoder = unsafe {
            let pool_ptr: *mut CommandPool = &mut self.lock.pool;
            (&mut *pool_ptr).record_with_guard(new_cb)
        };
        let old_encoder = std::mem::replace(&mut self.encoder, new_encoder);

        let mut cb = old_encoder.finish()?;
        self.lock.pool.finish(&mut cb)?;

        let inner = self.inner.clone();
        let task = bevy_tasks::IoTaskPool::get().spawn(async move {
            {
                let mut queue = inner.queue.lock().unwrap();
                queue.submit(&mut cb)?;
                drop(queue);
            }
            cb.block_async_until_completion().await?;
            Ok::<CommandBuffer, vk::Result>(cb)
        });
        self.pending_task = Some(task);
        Ok(())
    }

    /// Submits all remaining work and waits for completion.
    ///
    /// This finalizes the transfer batch, submits the command buffer to the
    /// transfer queue, and asynchronously waits until all transfers complete.
    pub async fn submit(mut self) -> VkResult<()> {
        if let Some(pending_task) = self.pending_task.take() {
            // Finish any remaining async transfers
            let cb = pending_task.await?;
            self.lock.pool.free(cb);
        }
        let mut cb = self.encoder.finish()?;
        self.lock.pool.finish(&mut cb)?;
        drop(self.lock);

        {
            let mut queue = self.inner.queue.lock().unwrap();
            queue.submit(&mut cb)?;
            drop(queue);
        }
        cb.block_async_until_completion().await?;

        let mut command_pool = self.inner.command_pool.lock().await;
        command_pool.pool.free(cb);
        Ok(())
    }
}
impl<'a> Deref for AsyncTransferGuard<'a> {
    type Target = CommandEncoder<'a>;
    fn deref(&self) -> &Self::Target {
        self.encoder.deref()
    }
}
impl<'a> DerefMut for AsyncTransferGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.encoder.deref_mut()
    }
}

impl AsyncTransfer {
    /// Begins a new async transfer batch.
    ///
    /// Returns a guard that can be used to record transfer commands. The guard
    /// derefs to [`CommandEncoder`] for convenient command recording.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut batch = async_transfer.batch().await?;
    /// batch.copy_buffer(src, dst);
    /// batch.submit().await?;
    /// ```
    pub async fn batch(&self) -> VkResult<AsyncTransferGuard<'_>> {
        let mut lock = self.0.command_pool.lock().await;
        let mut new_command_buffer = lock.pool.alloc()?;
        lock.timeline.schedule(&mut new_command_buffer);
        lock.pool.begin(&mut new_command_buffer)?;

        // Safety: The returned struct is self referencing. This is ok.
        let encoder = unsafe {
            let pool_ptr: *mut CommandPool = &mut lock.pool;
            (&mut *pool_ptr).record_with_guard(new_command_buffer)
        };
        Ok(AsyncTransferGuard {
            lock,
            encoder,
            inner: &self.0,
            pending_task: None,
        })
    }
}
