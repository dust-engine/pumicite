//! Image and image view abstractions.
//!
//! This module provides safe wrappers for Vulkan images and utilities for common image operations.

use ash::{VkResult, vk};
use glam::UVec3;

use crate::buffer::{BufferLike, StagingBufferAllocator};
use crate::command::CommandEncoder;
use crate::prelude::Access;
use crate::{Allocator, HasDevice, utils::AsVkHandle};
use vk_mem::Alloc;

/// Common interface for Vulkan image types.
///
/// This trait abstracts over different image implementations, providing access
/// to fundamental image properties needed for operations like view creation
/// and data transfers.
pub trait ImageLike: AsVkHandle<Handle = vk::Image> + Send + Sync + 'static {
    /// Returns the image aspect flags based on the image format.
    fn aspects(&self) -> vk::ImageAspectFlags;

    /// Returns the number of array layers in the image.
    fn array_layer_count(&self) -> u32;

    /// Returns the number of mip levels in the image.
    fn mip_level_count(&self) -> u32;

    /// Returns the image extent as a 3D vector (width, height, depth).
    fn extent(&self) -> UVec3;

    /// Returns the image format.
    fn format(&self) -> vk::Format;

    /// Returns the image type (1D, 2D, or 3D).
    fn ty(&self) -> vk::ImageType;
}

/// Common interface for Vulkan image view types.
///
/// Image views define how an image is accessed in shaders, including the
/// view type, mip levels, and array layers visible through the view.
pub trait ImageViewLike: AsVkHandle<Handle = vk::ImageView> + Send + Sync + 'static {
    /// Returns the image view type (1D, 2D, 3D, Cube, etc.).
    fn ty(&self) -> vk::ImageViewType;

    /// Returns the number of array layers visible through this view.
    fn array_layer_count(&self) -> u32;

    /// Returns the number of mip levels visible through this view.
    fn mip_level_count(&self) -> u32;
}

/// A regular image fully backed by memory
pub struct Image {
    allocator: Allocator,
    handle: vk::Image,
    allocation: vk_mem::Allocation,
    extent: UVec3,
    array_layer_count: u32,
    mipmap_level_count: u32,
    format: vk::Format,
    ty: vk::ImageType,
}
impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_image(self.handle, &mut self.allocation);
        }
    }
}
impl HasDevice for Image {
    fn device(&self) -> &crate::Device {
        self.allocator.device()
    }
}
impl Image {
    /// Create an image that is accessible exclusively from the GPU.
    ///
    /// The allocated image will have at least the following flags:
    /// - Discrete GPU: [DEVICE_LOCAL](`vk::MemoryPropertyFlags::DEVICE_LOCAL`)
    /// - Resizable BAR: [DEVICE_LOCAL](`vk::MemoryPropertyFlags::DEVICE_LOCAL`)
    /// - AMD APU: [DEVICE_LOCAL](`vk::MemoryPropertyFlags::DEVICE_LOCAL`)
    /// - Intel iGPU:  [DEVICE_LOCAL](`vk::MemoryPropertyFlags::DEVICE_LOCAL`)
    /// - Apple: [DEVICE_LOCAL](`vk::MemoryPropertyFlags::DEVICE_LOCAL`)
    pub fn new_private(allocator: Allocator, info: &vk::ImageCreateInfo) -> VkResult<Self> {
        let memory_type = allocator
            .device()
            .physical_device()
            .properties()
            .memory_type_map()
            .private;

        unsafe {
            let (image, allocation) = allocator.create_image(
                info,
                &vk_mem::AllocationCreateInfo {
                    memory_type_bits: 1 << memory_type,
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    ..Default::default()
                },
            )?;
            Ok(Self {
                extent: UVec3::new(info.extent.width, info.extent.height, info.extent.depth),
                allocator,
                handle: image,
                allocation,
                format: info.format,
                array_layer_count: info.array_layers,
                mipmap_level_count: info.mip_levels,
                ty: info.image_type,
            })
        }
    }
    /// Create a DEVICE_LOCAL image that is **preferably** host-writable.
    ///
    /// On GPUs with resizable BAR and integrated GPUs, the allocated image is host-visible,
    /// and uploads can be done directly. On GPUs without resizable BAR, a staging buffer
    /// is necessary.
    ///
    /// Always use [`ImageExt::update_contents_async`] to update its content.
    ///
    /// Uses the pre-calculated `upload` memory type from [`MemoryTypeMap`](crate::physical_device::MemoryTypeMap).
    ///
    /// The allocated image will have at least the following flags:
    /// - Discrete GPU: [DEVICE_LOCAL](`vk::MemoryPropertyFlags::DEVICE_LOCAL`)
    /// - Resizable BAR: [DEVICE_LOCAL](`vk::MemoryPropertyFlags::DEVICE_LOCAL`) | [HOST_VISIBLE](`vk::MemoryPropertyFlags::HOST_VISIBLE`)
    /// - AMD APU: [HOST_VISIBLE](`vk::MemoryPropertyFlags::HOST_VISIBLE`)
    /// - Intel iGPU:  [DEVICE_LOCAL](`vk::MemoryPropertyFlags::DEVICE_LOCAL`) | [HOST_VISIBLE](`vk::MemoryPropertyFlags::HOST_VISIBLE`)
    /// - Apple: [DEVICE_LOCAL](`vk::MemoryPropertyFlags::DEVICE_LOCAL`) | [HOST_VISIBLE](`vk::MemoryPropertyFlags::HOST_VISIBLE`)
    pub fn new_upload(allocator: Allocator, info: &vk::ImageCreateInfo) -> VkResult<Self> {
        let mut info = *info;
        let memory_type_map = allocator
            .device()
            .physical_device()
            .properties()
            .memory_type_map();

        if !memory_type_map.upload_host_visible {
            info.usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }

        unsafe {
            let (image, allocation) = allocator.create_image(
                &info,
                &vk_mem::AllocationCreateInfo {
                    memory_type_bits: 1 << memory_type_map.upload,
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD,
                    ..Default::default()
                },
            )?;
            Ok(Self {
                extent: UVec3::new(info.extent.width, info.extent.height, info.extent.depth),
                allocator,
                handle: image,
                allocation,
                format: info.format,
                array_layer_count: info.array_layers,
                mipmap_level_count: info.mip_levels,
                ty: info.image_type,
            })
        }
    }

    pub fn allocation_info(&self) -> vk_mem::AllocationInfo2 {
        self.allocator.get_allocation_info2(&self.allocation)
    }
}
impl ImageLike for Image {
    fn extent(&self) -> UVec3 {
        self.extent
    }
    fn format(&self) -> vk::Format {
        self.format
    }

    fn aspects(&self) -> vk::ImageAspectFlags {
        match self.format {
            vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
                vk::ImageAspectFlags::DEPTH
            }
            vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            vk::Format::G8_B8_R8_3PLANE_420_UNORM
            | vk::Format::G8_B8_R8_3PLANE_422_UNORM
            | vk::Format::G8_B8_R8_3PLANE_444_UNORM
            | vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16
            | vk::Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16
            | vk::Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16
            | vk::Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16
            | vk::Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16
            | vk::Format::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16
            | vk::Format::G16_B16_R16_3PLANE_420_UNORM
            | vk::Format::G16_B16_R16_3PLANE_422_UNORM
            | vk::Format::G16_B16_R16_3PLANE_444_UNORM => {
                vk::ImageAspectFlags::PLANE_0
                    | vk::ImageAspectFlags::PLANE_1
                    | vk::ImageAspectFlags::PLANE_2
            }
            vk::Format::G8_B8R8_2PLANE_420_UNORM
            | vk::Format::G8_B8R8_2PLANE_422_UNORM
            | vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
            | vk::Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16
            | vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16
            | vk::Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16
            | vk::Format::G16_B16R16_2PLANE_420_UNORM
            | vk::Format::G16_B16R16_2PLANE_422_UNORM
            | vk::Format::G8_B8R8_2PLANE_444_UNORM
            | vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
            | vk::Format::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16
            | vk::Format::G16_B16R16_2PLANE_444_UNORM => {
                vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1
            }

            vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
            _ => vk::ImageAspectFlags::COLOR,
        }
    }

    fn array_layer_count(&self) -> u32 {
        self.array_layer_count
    }

    fn mip_level_count(&self) -> u32 {
        self.mipmap_level_count
    }

    fn ty(&self) -> vk::ImageType {
        self.ty
    }
}
impl AsVkHandle for Image {
    type Handle = vk::Image;
    fn vk_handle(&self) -> Self::Handle {
        self.handle
    }
}

/// An image bundled with a full image view.
///
/// Wraps an image along with an image view that covers all mip levels and array layers.
/// This is a common pattern for textures that are accessed entirely through a single view.
///
/// The view is automatically destroyed when the `FullImageView` is dropped.
pub struct FullImageView<T: ImageLike + HasDevice> {
    image: T,
    view: vk::ImageView,
    ty: vk::ImageViewType,
}
impl<T: HasDevice + ImageLike> HasDevice for FullImageView<T> {
    fn device(&self) -> &crate::Device {
        self.image.device()
    }
}

impl<T: ImageLike + HasDevice> Drop for FullImageView<T> {
    fn drop(&mut self) {
        unsafe {
            self.image.device().destroy_image_view(self.view, None);
        }
    }
}
impl<T: ImageLike + HasDevice> FullImageView<T> {
    /// Returns a reference to the underlying image.
    pub fn image(&self) -> &T {
        &self.image
    }
}
impl<T: ImageLike + HasDevice> AsVkHandle for FullImageView<T> {
    type Handle = vk::ImageView;
    fn vk_handle(&self) -> Self::Handle {
        self.view
    }
}
impl<T: ImageLike + HasDevice> ImageViewLike for FullImageView<T> {
    fn array_layer_count(&self) -> u32 {
        self.image.array_layer_count()
    }

    fn mip_level_count(&self) -> u32 {
        self.image.mip_level_count()
    }

    fn ty(&self) -> vk::ImageViewType {
        self.ty
    }
}

/// Extension trait providing helper methods for images.
///
/// This trait is automatically implemented for all types implementing [`ImageLike`].
pub trait ImageExt: ImageLike {
    /// Creates a full image view covering all mip levels and array layers.
    fn create_full_view(self) -> VkResult<FullImageView<Self>>
    where
        Self: HasDevice + Sized,
    {
        let view_type = match self.ty() {
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => unreachable!(),
        };
        unsafe {
            let view = self.device().create_image_view(
                &vk::ImageViewCreateInfo {
                    image: self.vk_handle(),
                    view_type,
                    format: self.format(),
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: self.aspects(),
                        base_mip_level: 0,
                        base_array_layer: 0,
                        level_count: self.mip_level_count(),
                        layer_count: self.array_layer_count(),
                    },
                    ..Default::default()
                },
                None,
            )?;
            Ok(FullImageView {
                image: self,
                view,
                ty: view_type,
            })
        }
    }

    /// Uploads data to the image asynchronously via a staging buffer.
    ///
    /// This method handles the complete upload workflow:
    /// 1. Allocates a staging buffer from the provided allocator
    /// 2. Calls the async writer to fill the staging buffer
    /// 3. Transitions the image to `TRANSFER_DST_OPTIMAL`
    /// 4. Copies data for all mip levels
    /// 5. Transitions the image to the target layout
    ///
    /// The staging buffer is retained by the command encoder until the commands complete.
    #[must_use]
    fn update_contents_async<'a, A: StagingBufferAllocator, E>(
        &'a mut self,
        writer: impl AsyncFnOnce(&mut [u8]) -> Result<(), E>,
        encoder: &mut CommandEncoder<'a>,
        staging_allocator: &mut A,
        target_layout: vk::ImageLayout,
    ) -> impl Future<Output = Result<(), E>>
    where
        E: From<vk::Result>,
        Self: Sized,
    {
        async move {
            let format_properties = crate::utils::format::Format::from(self.format()).properties();
            let bytes_required = format_properties
                .bytes_required_for_texture(self.extent(), self.mip_level_count())
                * self.array_layer_count() as u64;
            let mut staging_buffer = staging_allocator.allocate_staging_buffer(bytes_required)?;
            let staging_slice = staging_buffer
                .as_slice_mut()
                .expect("Staging buffer allocator must return a host-visible buffer!");
            writer(staging_slice).await?;

            let staging_buffer = encoder.retain(staging_buffer);
            encoder.image_barrier(
                self,
                Access::NONE,
                Access::COPY_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                0..self.mip_level_count(),
                0..self.array_layer_count(),
            );
            encoder.emit_barriers();
            let mut buffer_offset = 0;
            let mut mip_size = self.extent();
            let regions: smallvec::SmallVec<[vk::BufferImageCopy; 1]> = (0..self.mip_level_count())
                .map(|i| {
                    let copy = vk::BufferImageCopy {
                        buffer_offset,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: i,
                            base_array_layer: 0,
                            layer_count: self.array_layer_count(),
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
                self,
                &regions,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            encoder.image_barrier(
                self,
                Access::COPY_WRITE,
                Access::NONE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                target_layout,
                0..self.mip_level_count(),
                0..self.array_layer_count(),
            );

            Ok(())
        }
    }
}

impl<T> ImageExt for T where T: ImageLike {}
