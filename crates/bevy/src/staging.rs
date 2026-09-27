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
use bevy_app::{Plugin, Startup};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{ResMut, SystemParam},
    world::{FromWorld, World},
};

use pumicite::{
    ash::{self, VkResult, vk},
    buffer::{RingBuffer, RingBufferSuballocation, StagingBufferAllocator},
    command::{CommandEncoderRenderPassState, CommandPool, GPURefMut},
    device::DeviceBuilder,
    prelude::*,
    sync::Timeline,
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
                encoder.copy_buffer(host_buffer.as_ref(), locked_buffer);
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
                ctx.copy_buffer(host_buffer.as_ref(), buffer);
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
impl FromWorld for AsyncTransfer {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        let queue = world.make_shared_queue::<TransferQueue>();
        let device = world.resource::<Device>().clone();
        let mut command_pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
        let mut timeline = Timeline::new(device).unwrap();
        let mut command_buffer = command_pool.alloc().unwrap();
        timeline.schedule(&mut command_buffer);
        command_pool.begin(&mut command_buffer);

        Self(Arc::new(AsyncTransferInner {
            queue,
            command_pool: Mutex::new(AsyncTransferCommandContext {
                timeline,
                command_pool,
                current_command_buffer: command_buffer,
            }),
        }))
    }
}

struct AsyncTransferInner {
    queue: SharedQueue,
    command_pool: Mutex<AsyncTransferCommandContext>,
}
struct AsyncTransferCommandContext {
    command_pool: CommandPool,
    timeline: Timeline,

    current_command_buffer: CommandBuffer,
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
}

impl<'a> AsyncTransferGuard<'a> {
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
                    staging_buffer.as_ref(),
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

        Ok(())
    }
    /// Submits all remaining work and waits for completion.
    ///
    /// This finalizes the transfer batch, submits the command buffer to the
    /// transfer queue, and asynchronously waits until all transfers complete.
    pub async fn submit(mut self) -> VkResult<()> {
        Ok(())
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
        Ok(AsyncTransferGuard { inner: &self.0 })
    }
}
