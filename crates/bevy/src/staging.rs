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
    collections::VecDeque,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use async_lock::Mutex;
use bevy_app::{Plugin, Startup};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Res, ResMut, SystemParam},
    world::{FromWorld, World},
};

use pumicite::{
    ash::{self, VkResult, vk},
    buffer::{RingBuffer, RingBufferSuballocation, StagingBufferAllocator},
    command::{CommandEncoderRenderPassState, CommandPool, GPURefMut},
    device::DeviceBuilder,
    prelude::*,
    sync::{Timeline, Timestamp},
};

use crate::{
    CreateDevice,
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
        let device_local_chunk_size = self.device_local_chunk_size;
        let uniform_chunk_size = self.uniform_chunk_size;
        let host_visible_chunk_size = self.host_visible_chunk_size;
        app.add_systems(
            Startup,
            (
                // Needed for the Uploader which requests buffer device address automatically.
                (|mut device_builder: ResMut<DeviceBuilder>| {
                    device_builder
                        .enable_feature::<vk::PhysicalDeviceBufferDeviceAddressFeatures>(|x| {
                            &mut x.buffer_device_address
                        })
                        .unwrap();
                })
                .before(CreateDevice),
                (move |world: &mut World| {
                    let device = world.resource::<Device>().clone();
                    world.insert_resource(
                        DeviceLocalRingBuffer::new(device.clone(), device_local_chunk_size)
                            .unwrap(),
                    );
                    world.insert_resource(
                        HostVisibleRingBuffer::new(device.clone(), host_visible_chunk_size)
                            .unwrap(),
                    );
                    world.insert_resource(
                        UniformRingBuffer::new(device, uniform_chunk_size).unwrap(),
                    );
                })
                .after(CreateDevice),
            ),
        );
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
    /// Creates a **small** uniform buffer with the given data.
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
    ) -> GPURefMut<'a, RingBufferSuballocation> {
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
        debug_assert!(matches!(
            encoder.render_pass_state(),
            CommandEncoderRenderPassState::OutsideRenderPass
        ));
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
    ) -> GPURefMut<'a, RingBufferSuballocation> {
        debug_assert!(matches!(
            ctx.render_pass_state(),
            CommandEncoderRenderPassState::OutsideRenderPass
        ));
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
/// Transfers from every batch are recorded into one shared command buffer.
/// [`async_transfer_submission_system`] submits it once per frame. If the staging data recorded
/// since the last submission exceeds [`submit_threshold`](Self::submit_threshold), the
/// recording batch submits it immediately instead, so large uploads don't wait for the frame.
///
/// # Usage
///
/// ```ignore
/// let mut batch = transfer.batch().await?;
/// batch
///     .update_image(
///         &mut image,
///         async |staging| {
///             // Fill `staging` with the image's contents.
///             Ok::<(), vk::Result>(())
///         },
///         &mut allocator,
///         vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
///     )
///     .await?;
/// // Waits until the upload has completed on the GPU.
/// batch.submit().await?;
/// ```
#[derive(Clone, Resource)]
pub struct AsyncTransfer(Arc<AsyncTransferInner>);
impl FromWorld for AsyncTransfer {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        let queue = world.make_shared_queue::<TransferQueue>();
        let device = world.resource::<Device>().clone();
        let mut command_pool =
            CommandPool::new_resettable(device.clone(), queue.family_index()).unwrap();
        let mut timeline = Timeline::new(device).unwrap();
        let mut command_buffer = command_pool.alloc().unwrap();
        let current_timestamp = timeline.schedule(&mut command_buffer);
        command_pool.begin(&mut command_buffer).unwrap();

        Self(Arc::new(AsyncTransferInner {
            queue,
            submit_threshold: AsyncTransfer::DEFAULT_SUBMIT_THRESHOLD,
            command_pool: Mutex::new(AsyncTransferCommandContext {
                timeline,
                command_pool,
                current_command_buffer: command_buffer,
                current_timestamp,
                pending_bytes: 0,
                in_flight: VecDeque::new(),
                free: Vec::new(),
            }),
        }))
    }
}

struct AsyncTransferInner {
    queue: SharedQueue,
    /// Staging bytes recorded into the current command buffer that trigger an immediate
    /// submission.
    submit_threshold: u64,
    command_pool: Mutex<AsyncTransferCommandContext>,
}
impl Drop for AsyncTransferInner {
    fn drop(&mut self) {
        // This would only run during application shutdown. A scheduled command buffer must be
        // submitted, so submit the current one even if it's empty, then wait for everything.
        // `command_pool` is declared first in the context, so it drops before
        // `current_command_buffer`, which then doesn't need to be freed.
        let ctx = self.command_pool.get_mut();
        let cb = &mut ctx.current_command_buffer;
        ctx.command_pool.finish(cb).unwrap();
        self.queue.lock().unwrap().submit(cb).unwrap();
        cb.block_until_completion().unwrap();
        while let Some(mut cb) = ctx.in_flight.pop_front() {
            cb.block_until_completion().unwrap();
            ctx.command_pool.free(cb);
        }
        for cb in ctx.free.drain(..) {
            ctx.command_pool.free(cb);
        }
    }
}

struct AsyncTransferCommandContext {
    command_pool: CommandPool,
    timeline: Timeline,

    current_command_buffer: CommandBuffer,
    /// Timestamp `current_command_buffer` signals when it completes execution.
    current_timestamp: Timestamp,
    /// Staging bytes recorded into `current_command_buffer`. Zero means it has no work.
    pending_bytes: u64,
    /// Submitted command buffers that may still be executing, oldest first. Each one retains
    /// the staging buffers of the transfers it contains until it's reclaimed.
    in_flight: VecDeque<CommandBuffer>,
    /// Completed command buffers, reset and ready for reuse.
    free: Vec<CommandBuffer>,
}
impl AsyncTransferCommandContext {
    /// Reclaims completed command buffers, then, if anything has been recorded, submits the
    /// current command buffer and starts recording a new one.
    fn submit_current(&mut self, queue: &SharedQueue) -> VkResult<()> {
        // Every command buffer waits for the previous one on the timeline, so they complete in
        // submission order.
        while let Some(oldest) = self.in_flight.front_mut()
            && oldest.try_complete()
        {
            let mut cb = self.in_flight.pop_front().unwrap();
            self.command_pool.reset(&mut cb);
            self.free.push(cb);
        }

        if self.pending_bytes == 0 {
            return Ok(());
        }

        let mut next = match self.free.pop() {
            Some(cb) => cb,
            None => self.command_pool.alloc()?,
        };

        let next_timestamp = self.timeline.schedule(&mut next);
        self.command_pool.begin(&mut next)?;

        let mut cb = std::mem::replace(&mut self.current_command_buffer, next);
        self.current_timestamp = next_timestamp;
        self.pending_bytes = 0;

        self.command_pool.finish(&mut cb)?;
        queue.lock().unwrap().submit(&mut cb)?;
        self.in_flight.push_back(cb);
        Ok(())
    }
}

/// Submits the transfers recorded through [`AsyncTransfer`] since the last submission, and
/// reclaims command buffers whose transfers have completed, releasing their staging memory.
///
/// Runs once per frame. If a batch is recording at the moment, the frame is skipped and the work
/// is picked up next frame, rather than blocking the frame on the recording task.
pub fn async_transfer_submission_system(transfer: Res<AsyncTransfer>) {
    let inner = &*transfer.0;
    let Some(mut ctx) = inner.command_pool.try_lock() else {
        return;
    };
    ctx.submit_current(&inner.queue).unwrap();
}

/// Guard for an active async transfer batch.
///
/// Transfers are recorded into a command buffer shared by all batches, which is submitted once
/// per frame by [`async_transfer_submission_system`], or immediately once enough staging data
/// has accumulated (see [`AsyncTransfer::submit_threshold`]). Call [`submit`](Self::submit) to
/// wait until the transfers recorded through this guard have completed on the GPU.
///
/// If the guard is dropped before [`submit`](Self::submit) completes (for example because
/// the enclosing future was cancelled), its destructor blocks the current thread until those
/// transfers complete. This keeps resources borrowed by [`update_image`](Self::update_image)
/// alive for as long as the GPU uses them.
///
/// # Leaking
///
/// The guard must not be leaked (e.g. with [`std::mem::forget`]) while it has transfers
/// pending. A leaked guard never waits, so the borrows it holds end while the GPU may still
/// be using the borrowed resources, and dropping one of them causes undefined behavior. This
/// is the same limitation the pre-1.0 `thread::scoped` API had; Rust has no way to express
/// types that cannot be leaked.
pub struct AsyncTransferGuard<'a> {
    inner: &'a Arc<AsyncTransferInner>,
    /// Latest timeline point of the shared command buffer recorded into through this guard.
    pending: Option<Timestamp>,
}

impl<'a> AsyncTransferGuard<'a> {
    /// Uploads data to `image` through a staging buffer, then transitions it to `target_layout`.
    ///
    /// `writer` fills the staging buffer with the contents of every mip level, tightly packed.
    ///
    /// The copy runs when the shared command buffer is submitted. `image` stays borrowed until
    /// the guard is consumed by [`submit`](Self::submit) or dropped, both of which wait for the
    /// copy to complete. See [Leaking](Self#leaking) for the one case this doesn't cover.
    pub async fn update_image<A: StagingBufferAllocator, E: From<vk::Result>>(
        &mut self,
        image: &'a mut impl ImageLike,
        writer: impl AsyncFnOnce(&mut [u8]) -> Result<(), E>,
        staging_allocator: &mut A,
        target_layout: vk::ImageLayout,
    ) -> Result<(), E> {
        let format_properties = pumicite_types::format::Format::from(image.format()).properties();
        let bytes_required = format_properties
            .bytes_required_for_texture(image.extent(), image.mip_level_count())
            * image.array_layer_count() as u64;
        let mut staging_buffer = staging_allocator.allocate_staging_buffer(bytes_required)?;
        let staging_slice = staging_buffer
            .as_slice_mut()
            .expect("Staging buffer allocator must return a host-visible buffer!");
        writer(staging_slice).await?;

        let command_ctx = &mut *self.inner.command_pool.lock().await;
        command_ctx
            .command_pool
            .record(&mut command_ctx.current_command_buffer, |encoder| {
                let staging_buffer = encoder.retain(staging_buffer);
                encoder.image_barrier(
                    unsafe { GPURefMut::new_unchecked(image) },
                    Access::NONE,
                    Access::COPY_WRITE,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    0..image.mip_level_count(),
                    0..image.array_layer_count(),
                );
                encoder.emit_barriers();
                let mut buffer_offset = 0;
                let mut mip_size = image.extent();
                let regions: smallvec::SmallVec<[vk::BufferImageCopy; 1]> = (0..image
                    .mip_level_count())
                    .map(|i| {
                        let copy = vk::BufferImageCopy {
                            buffer_offset,
                            image_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: i,
                                base_array_layer: 0,
                                layer_count: image.array_layer_count(),
                            },
                            image_extent: vk::Extent3D {
                                width: mip_size.x,
                                height: mip_size.y,
                                depth: mip_size.z,
                            },
                            ..Default::default()
                        };
                        buffer_offset += format_properties.bytes_required_for_texture(mip_size, 1);
                        mip_size.x = mip_size.x.div_ceil(2);
                        mip_size.y = mip_size.y.div_ceil(2);
                        mip_size.z = mip_size.z.div_ceil(2);
                        copy
                    })
                    .collect();
                encoder.copy_buffer_to_image_with_layout(
                    staging_buffer,
                    unsafe { GPURefMut::new_unchecked(image) },
                    &regions,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                encoder.image_barrier(
                    unsafe { GPURefMut::new_unchecked(image) },
                    Access::COPY_WRITE,
                    Access::NONE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    target_layout,
                    0..image.mip_level_count(),
                    0..image.array_layer_count(),
                );
            });
        self.record_pending(&command_ctx.current_timestamp);
        command_ctx.pending_bytes += bytes_required;
        if command_ctx.pending_bytes >= self.inner.submit_threshold {
            command_ctx.submit_current(&self.inner.queue)?;
        }

        Ok(())
    }

    fn record_pending(&mut self, timestamp: &Timestamp) {
        // Every command buffer comes from the same timeline, so later ones have larger values.
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| pending.value() < timestamp.value())
        {
            self.pending = Some(timestamp.clone());
        }
    }
    /// Waits until the transfers recorded through this guard have completed on the GPU.
    ///
    /// This doesn't submit anything itself: the shared command buffer is submitted by
    /// [`async_transfer_submission_system`], or earlier once
    /// [`submit_threshold`](AsyncTransfer::submit_threshold) is exceeded.
    pub async fn submit(mut self) -> VkResult<()> {
        if let Some(timestamp) = &self.pending {
            // `pending` is only cleared once the wait completes, so if this future is
            // cancelled mid-wait, `Drop` still waits.
            timestamp.wait_async().await?;
        }
        self.pending = None;
        Ok(())
    }
}

impl Drop for AsyncTransferGuard<'_> {
    fn drop(&mut self) {
        if let Some(timestamp) = self.pending.take() {
            // Resources borrowed by this guard must outlive the GPU's use of them. There's no
            // way to report an error from here, and returning early would end those borrows.
            if let Err(err) = timestamp.wait_blocked(u64::MAX) {
                tracing::error!("Failed to wait for pending async transfers: {err:?}");
            }
        }
    }
}

impl AsyncTransfer {
    /// Default for [`submit_threshold`](Self::submit_threshold): 64 MiB.
    pub const DEFAULT_SUBMIT_THRESHOLD: u64 = 64 * 1024 * 1024;

    /// Amount of staging data, in bytes, that can be recorded before the recording batch
    /// submits it immediately instead of waiting for [`async_transfer_submission_system`].
    pub fn submit_threshold(&self) -> u64 {
        self.0.submit_threshold
    }

    /// Begins a new async transfer batch.
    ///
    /// Returns a guard for recording uploads, such as
    /// [`update_image`](AsyncTransferGuard::update_image).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut batch = transfer.batch().await?;
    /// batch
    ///     .update_image(
    ///         &mut image,
    ///         async |staging| {
    ///             // Fill `staging` with the image's contents.
    ///             Ok::<(), vk::Result>(())
    ///         },
    ///         &mut allocator,
    ///         vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    ///     )
    ///     .await?;
    /// // Waits until the upload has completed on the GPU.
    /// batch.submit().await?;
    /// ```
    pub async fn batch(&self) -> VkResult<AsyncTransferGuard<'_>> {
        Ok(AsyncTransferGuard {
            inner: &self.0,
            pending: None,
        })
    }
}
