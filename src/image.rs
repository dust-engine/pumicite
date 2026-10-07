//! Image and image view abstractions.
//!
//! This module provides safe wrappers for Vulkan images and utilities for common image operations.

use std::ops::Deref;

use ash::{VkResult, vk, vk::TaggedStructure};
use glam::UVec3;

use crate::buffer::StagingBufferAllocator;
use crate::command::{GPURef, GPURefMut, NoHostMapping, project_host_metadata};
use crate::prelude::*;
use vk_mem::Alloc;

/// Common interface for Vulkan image types.
///
/// This trait abstracts over different image implementations, providing access
/// to fundamental image properties needed for operations like view creation
/// and data transfers.
///
/// # Safety
///
/// Command recording relies on implementations to name only an image they own:
///
/// - The image returned by `vk_handle()` must be owned by this value alone. No value that
///   isn't part of this one (the way the image inside a [`FullImageView`] is) may expose it
///   while this one is alive. Otherwise a write token for one value would let the GPU write,
///   or change the layout of, an image another value is using.
/// - The image must stay valid until this value is dropped.
/// - The other methods must describe that image accurately: barriers and copies are recorded
///   from them.
pub unsafe trait ImageLike: AsVkHandle<Handle = vk::Image> + Send + Sync + 'static {
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
project_host_metadata! {
    impl[T: ImageLike + ?Sized] T {
        fn aspects(&self) -> vk::ImageAspectFlags;
        fn array_layer_count(&self) -> u32;
        fn mip_level_count(&self) -> u32;
        fn extent(&self) -> UVec3;
        fn format(&self) -> vk::Format;
        fn ty(&self) -> vk::ImageType;
    }
}

/// Common interface for Vulkan image view types.
///
/// Image views define how an image is accessed in shaders, including the
/// view type, mip levels, and array layers visible through the view.
///
/// # Safety
///
/// A write token for a view, such as a render pass attachment, lets the GPU write the
/// image behind it. So:
///
/// - The image must be owned, in the sense of [`ImageLike`], by this value, or by the
///   value this view is part of. It must be impossible to obtain the view other than
///   through that owner, so that a token for the view can only come from a token for the
///   owner. For example, the views inside a [`FullImageView`] aren't `Clone` and have no
///   public constructor.
/// - The view and its image must stay valid until that owner is dropped.
/// - The other methods must describe the view accurately.
pub unsafe trait ImageViewLike: AsVkHandle<Handle = vk::ImageView> + Send + Sync {
    /// Returns the image view type (1D, 2D, 3D, Cube, etc.).
    fn ty(&self) -> vk::ImageViewType;

    /// Returns the array layers of the base image visible through this view.
    fn array_layers(&self) -> std::ops::Range<u32>;

    /// Returns the mip levels of the base image visible through this view.
    fn mip_levels(&self) -> std::ops::Range<u32>;

    /// Returns the image view format.
    fn format(&self) -> vk::Format;
}

/// A GPU-allocated image fully backed by device memory.
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
    pub fn new_device(allocator: Allocator, info: &vk::ImageCreateInfo) -> VkResult<Self> {
        let memory_type = allocator.device().memory_type_map().private_image(info)?;

        unsafe {
            let (image, allocation) = allocator.create_image(
                info,
                &vk_mem::AllocationCreateInfo {
                    memory_type_bits: 1 << memory_type,
                    usage: vk_mem::MemoryUsage::Unknown,
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
    /// Upload its contents through a staging buffer, for example with
    /// `AsyncTransfer::update_image` in `bevy_pumicite`.
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
        let memory_type_map = allocator.device().memory_type_map();

        if !memory_type_map.upload_host_visible {
            info.usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        let memory_type = memory_type_map.upload_image(&info)?;

        unsafe {
            let (image, allocation) = allocator.create_image(
                &info,
                &vk_mem::AllocationCreateInfo {
                    memory_type_bits: 1 << memory_type,
                    usage: vk_mem::MemoryUsage::Unknown,
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

    /// Returns the underlying VMA allocation.
    pub fn allocation_info(&self) -> vk_mem::AllocationInfo2 {
        self.allocator.get_allocation_info2(&self.allocation)
    }
}
project_host_metadata! {
    impl[] Image {
        /// Returns the underlying VMA allocation.
        fn allocation_info(&self) -> vk_mem::AllocationInfo2;
    }
}
// Safety: an `Image` exclusively owns its `VkImage` and allocation, both created by its
// constructors and destroyed on drop. `Image` isn't `Clone`, and the metadata comes from the
// create info.
unsafe impl ImageLike for Image {
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

/// Defines an image-view wrapper newtype: an image `T` bundled with one extra
/// [`ImageViewItem`]. Delegates [`ImageLike`], [`HasDevice`], and [`AsVkHandle`]
/// to the wrapped image, dereferences to it, exposes the view via the named
/// accessor, and destroys the view when dropped.
///
/// Wrappers hold a single view; chain constructors (e.g.
/// `image.create_full_view().create_srgb_view(..)`) to attach more.
macro_rules! image_view_wrapper {
    (
        $(#[$struct_meta:meta])*
        $name:ident {
            $(#[$accessor_meta:meta])*
            $field:ident => $accessor:ident $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        pub struct $name<T: ImageLike + HasDevice> {
            image: T,
            $field: ImageViewItem,
        }
        impl<T: ImageLike + HasDevice> $name<T> {
            $(#[$accessor_meta])*
            pub fn $accessor(&self) -> &ImageViewItem {
                &self.$field
            }
        }
        impl<T: ImageLike + HasDevice> crate::sync::GPUMutex<$name<T>> {
            $(#[$accessor_meta])*
            pub fn $accessor(&self) -> &ImageViewItem {
                &self.inner.$field
            }
        }
        impl<'a, T: ImageLike + HasDevice> GPURef<'a, $name<T>> {
            $(#[$accessor_meta])*
            pub fn $accessor(self) -> GPURef<'a, ImageViewItem> {
                unsafe { GPURef::new_unchecked(&self.unwrap().$field) }
            }
        }
        impl<'a, T: ImageLike + HasDevice> GPURefMut<'a, $name<T>> {
            $(#[$accessor_meta])*
            pub fn $accessor(self) -> GPURefMut<'a, ImageViewItem> {
                unsafe { GPURefMut::new_unchecked(&self.unwrap().$field) }
            }
        }
        impl<'a, T: ImageLike + HasDevice> GPURef<'a, $name<T>> {
            pub fn deref_inner(self) -> GPURef<'a, T> {
                unsafe { GPURef::new_unchecked(&self.unwrap().image) }
            }
        }
        impl<'a, T: ImageLike + HasDevice> GPURefMut<'a, $name<T>> {
            pub fn deref_inner(self) -> GPURefMut<'a, T> {
                unsafe { GPURefMut::new_unchecked(&self.unwrap().image) }
            }
        }
        impl<T: ImageLike + HasDevice> HasDevice for $name<T> {
            fn device(&self) -> &crate::Device {
                self.image.device()
            }
        }
        impl<T: ImageLike + HasDevice> Deref for $name<T> {
            type Target = T;
            fn deref(&self) -> &Self::Target {
                &self.image
            }
        }
        impl<T: ImageLike + HasDevice> AsVkHandle for $name<T> {
            type Handle = vk::Image;
            fn vk_handle(&self) -> Self::Handle {
                self.image.vk_handle()
            }
        }
        // Safety: the wrapper owns `image`, which uniquely owns its image since `T: ImageLike`.
        // It only lends it out through `Deref`, which can't produce a token, and describes it
        // by forwarding to it.
        unsafe impl<T: ImageLike + HasDevice> ImageLike for $name<T> {
            fn aspects(&self) -> vk::ImageAspectFlags {
                self.image.aspects()
            }
            fn array_layer_count(&self) -> u32 {
                self.image.array_layer_count()
            }
            fn mip_level_count(&self) -> u32 {
                self.image.mip_level_count()
            }
            fn extent(&self) -> UVec3 {
                self.image.extent()
            }
            fn format(&self) -> vk::Format {
                self.image.format()
            }
            fn ty(&self) -> vk::ImageType {
                self.image.ty()
            }
        }
        impl<T: ImageLike + HasDevice> Drop for $name<T> {
            fn drop(&mut self) {
                unsafe {
                    self.image
                        .device()
                        .destroy_image_view(self.$field.view, None);
                }
            }
        }
    };
}

image_view_wrapper! {
    /// An image bundled with a full image view.
    ///
    /// Wraps an image along with an image view that covers all mip levels and array layers.
    /// This is a common pattern for textures that are accessed entirely through a single view.
    ///
    /// The view is automatically destroyed when the `FullImageView` is dropped.
    FullImageView {
        /// Returns the full image view covering all mip levels and array layers.
        view => full_view
    }
}

image_view_wrapper! {
    /// An image bundled with a view that samples it through its sRGB-format
    /// counterpart, applying the sRGB transfer function (decode) on read.
    ///
    /// The wrapped image must have a *linear* format that has an sRGB counterpart
    /// (enforced by [`ImageExt::create_srgb_view`]), and be created with
    /// [`vk::ImageCreateFlags::MUTABLE_FORMAT`] and a compatible format list
    /// ([`vk::ImageFormatListCreateInfo`]). To also access the image through its
    /// own linear format, wrap a full view:
    /// `image.create_full_view()?.create_srgb_view(..)`.
    ///
    /// The view is automatically destroyed when the `SrgbImageView` is dropped.
    SrgbImageView {
        /// Returns the sRGB-format view of the image.
        srgb_view => srgb_view
    }
}

image_view_wrapper! {
    /// An image bundled with a view that reinterprets its texels as unsigned integers.
    ///
    /// Wraps an image along with a `UINT`-format view covering all mip levels and array
    /// layers — for example, an `R8G8B8A8_UNORM` image is viewed as `R8G8B8A8_UINT`. This
    /// lets the raw integer contents of each texel be read or written in a shader (e.g. for
    /// bitwise or atomic access) without any format conversion applied.
    ///
    /// The image must have been created with [`vk::ImageCreateFlags::MUTABLE_FORMAT`] and a
    /// compatible format list ([`vk::ImageFormatListCreateInfo`]) that includes the `UINT`
    /// variant.
    ///
    /// The view is automatically destroyed when the `UintImageView` is dropped.
    UintImageView {
        /// Returns the `UINT`-format view covering all mip levels and array layers.
        uint_view => uint_view
    }
}

image_view_wrapper! {
    /// An image bundled with an extra view that remaps components when sampled — for
    /// example broadcasting the alpha channel into every component to expose a value
    /// packed in `.a` (such as roughness stored in a normal G-buffer's alpha) as a
    /// single-channel input.
    ///
    /// The swizzle only affects reads and keeps the image's format, so unlike a
    /// format-reinterpreting view it needs no `MUTABLE_FORMAT`/format-list on the
    /// image. The view is destroyed when the `SwizzledImageView` is dropped.
    SwizzledImageView {
        /// Returns the component-swizzled view covering all mip levels and array layers.
        swizzled_view => swizzled_view
    }
}

/// An image bundled with one single-level view per mip level, for shaders that
/// write a mip chain through storage images (`RWTexture2D` per level: an HZB or
/// any other in-shader reduction). Wrap a full view first when the chain must
/// also be sampled as a whole: `image.create_full_view()?.create_mip_views()?`.
///
/// Every view is destroyed when the `MipImageViews` is dropped.
pub struct MipImageViews<T: ImageLike + HasDevice> {
    image: T,
    views: Vec<ImageViewItem>,
}
impl<T: ImageLike + HasDevice> MipImageViews<T> {
    /// Returns the view covering exactly mip level `level`.
    ///
    /// # Panics
    /// If `level >= mip_level_count()`.
    pub fn mip_view(&self, level: u32) -> &ImageViewItem {
        &self.views[level as usize]
    }
}
impl<T: ImageLike + HasDevice> crate::sync::GPUMutex<MipImageViews<T>> {
    /// Returns the view covering exactly mip level `level`.
    ///
    /// # Panics
    /// If `level >= mip_level_count()`.
    pub fn mip_view(&self, level: u32) -> &ImageViewItem {
        &self.inner.views[level as usize]
    }
}
impl<'a, T: ImageLike + HasDevice> GPURef<'a, MipImageViews<T>> {
    /// Returns the view covering exactly mip level `level`.
    ///
    /// # Panics
    /// If `level >= mip_level_count()`.
    pub fn mip_view(self, level: u32) -> GPURef<'a, ImageViewItem> {
        // The views are owned by the `MipImageViews`, so they live as long as it does and
        // share its access.
        unsafe { GPURef::new_unchecked(&self.unwrap().views[level as usize]) }
    }
}
impl<'a, T: ImageLike + HasDevice> GPURefMut<'a, MipImageViews<T>> {
    /// Returns the view covering exactly mip level `level`.
    ///
    /// # Panics
    /// If `level >= mip_level_count()`.
    pub fn mip_view(self, level: u32) -> GPURefMut<'a, ImageViewItem> {
        // The views are owned by the `MipImageViews`, so they live as long as it does and
        // share its access.
        unsafe { GPURefMut::new_unchecked(&self.unwrap().views[level as usize]) }
    }
}
impl<'a, T: ImageLike + HasDevice> GPURef<'a, MipImageViews<T>> {
    pub fn deref_inner(self) -> GPURef<'a, T> {
        unsafe { GPURef::new_unchecked(&self.unwrap().image) }
    }
}
impl<'a, T: ImageLike + HasDevice> GPURefMut<'a, MipImageViews<T>> {
    pub fn deref_inner(self) -> GPURefMut<'a, T> {
        unsafe { GPURefMut::new_unchecked(&self.unwrap().image) }
    }
}
impl<T: ImageLike + HasDevice> HasDevice for MipImageViews<T> {
    fn device(&self) -> &crate::Device {
        self.image.device()
    }
}
impl<T: ImageLike + HasDevice> Deref for MipImageViews<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.image
    }
}
impl<T: ImageLike + HasDevice> AsVkHandle for MipImageViews<T> {
    type Handle = vk::Image;
    fn vk_handle(&self) -> Self::Handle {
        self.image.vk_handle()
    }
}
// Safety: as for the view wrappers. `MipImageViews` owns `image`, lends it out only through
// `Deref`, and forwards to it.
unsafe impl<T: ImageLike + HasDevice> ImageLike for MipImageViews<T> {
    fn aspects(&self) -> vk::ImageAspectFlags {
        self.image.aspects()
    }
    fn array_layer_count(&self) -> u32 {
        self.image.array_layer_count()
    }
    fn mip_level_count(&self) -> u32 {
        self.image.mip_level_count()
    }
    fn extent(&self) -> UVec3 {
        self.image.extent()
    }
    fn format(&self) -> vk::Format {
        self.image.format()
    }
    fn ty(&self) -> vk::ImageType {
        self.image.ty()
    }
}
impl<T: ImageLike + HasDevice> Drop for MipImageViews<T> {
    fn drop(&mut self) {
        unsafe {
            for view in &self.views {
                self.image.device().destroy_image_view(view.view, None);
            }
        }
    }
}

/// A borrowed handle to a single image view, paired with the metadata needed to
/// use it in descriptor writes and rendering commands.
///
/// Returned by the view accessors on [`FullImageView`], [`SrgbImageView`], and
/// [`UintImageView`]. Implements [`ImageViewLike`] and [`AsVkHandle`].
pub struct ImageViewItem {
    ty: vk::ImageViewType,
    format: vk::Format,
    mip_levels: std::ops::Range<u32>,
    array_layers: std::ops::Range<u32>,
    view: vk::ImageView,
}
impl AsVkHandle for ImageViewItem {
    type Handle = vk::ImageView;
    fn vk_handle(&self) -> Self::Handle {
        self.view
    }
}
// Safety: an `ImageViewItem` holds only a view handle and its metadata.
unsafe impl NoHostMapping for ImageViewItem {}
// Safety: `ImageViewItem`s are only created inside the view wrappers and `MipImageViews`, which
// own the image and destroy the view on drop. They aren't `Clone` and have no public
// constructor, so they're only reachable through their owner.
unsafe impl ImageViewLike for ImageViewItem {
    fn ty(&self) -> vk::ImageViewType {
        self.ty
    }
    fn mip_levels(&self) -> std::ops::Range<u32> {
        self.mip_levels.clone()
    }

    fn array_layers(&self) -> std::ops::Range<u32> {
        self.array_layers.clone()
    }

    fn format(&self) -> vk::Format {
        self.format
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
            vk::ImageType::TYPE_1D if self.array_layer_count() > 1 => {
                vk::ImageViewType::TYPE_1D_ARRAY
            }
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_2D if self.array_layer_count() > 1 => {
                vk::ImageViewType::TYPE_2D_ARRAY
            }
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
                view: ImageViewItem {
                    ty: view_type,
                    format: self.format(),
                    mip_levels: 0..self.mip_level_count(),
                    array_layers: 0..self.array_layer_count(),
                    view,
                },
                image: self,
            })
        }
    }

    /// Creates a [`MipImageViews`]: one view per mip level, each covering that
    /// single level and all array layers, in the image's own format.
    fn create_mip_views(self) -> VkResult<MipImageViews<Self>>
    where
        Self: HasDevice + Sized,
    {
        let view_type = match self.ty() {
            vk::ImageType::TYPE_1D if self.array_layer_count() > 1 => {
                vk::ImageViewType::TYPE_1D_ARRAY
            }
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_2D if self.array_layer_count() > 1 => {
                vk::ImageViewType::TYPE_2D_ARRAY
            }
            vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => unreachable!(),
        };
        let mut views = Vec::with_capacity(self.mip_level_count() as usize);
        for level in 0..self.mip_level_count() {
            let result = unsafe {
                self.device().create_image_view(
                    &vk::ImageViewCreateInfo {
                        image: self.vk_handle(),
                        view_type,
                        format: self.format(),
                        components: vk::ComponentMapping::default(),
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: self.aspects(),
                            base_mip_level: level,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: self.array_layer_count(),
                        },
                        ..Default::default()
                    },
                    None,
                )
            };
            let view = match result {
                Ok(view) => view,
                Err(err) => {
                    // Undo the levels created so far; the image is dropped by the caller.
                    for created in &views {
                        let created: &ImageViewItem = created;
                        unsafe { self.device().destroy_image_view(created.view, None) };
                    }
                    return Err(err);
                }
            };
            views.push(ImageViewItem {
                ty: view_type,
                format: self.format(),
                mip_levels: level..level + 1,
                array_layers: 0..self.array_layer_count(),
                view,
            });
        }
        Ok(MipImageViews { image: self, views })
    }

    /// Creates an [`SrgbImageView`]: a view that samples the image through its
    /// sRGB-format counterpart (sRGB decode on read).
    ///
    /// The image must have a *linear* format with an sRGB counterpart, and be
    /// created with [`vk::ImageCreateFlags::MUTABLE_FORMAT`] and a format list
    /// ([`vk::ImageFormatListCreateInfo`]) including the sRGB variant. To also
    /// access the image through its own linear format, wrap a full view first:
    /// `image.create_full_view()?.create_srgb_view(usage)`.
    ///
    /// # Parameters
    /// - `usage` - Usage flags for the sRGB view (sRGB formats do not support
    ///   [`vk::ImageUsageFlags::STORAGE`]).
    ///
    /// # Errors
    /// Returns [`vk::Result::ERROR_FORMAT_NOT_SUPPORTED`] if the image's format
    /// has no sRGB counterpart.
    fn create_srgb_view(self, usage: vk::ImageUsageFlags) -> VkResult<SrgbImageView<Self>>
    where
        Self: HasDevice + Sized,
    {
        let view_type = match self.ty() {
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => panic!("Unsupported view type: {:?}", self.ty()),
        };
        let format = pumicite_types::format::Format::from(self.format());
        debug_assert!(
            vk::Format::from(format.to_linear_format()) == self.format(),
            "create_srgb_view requires a linear (non-sRGB) image format; got {:?}",
            self.format(),
        );
        let srgb_format: vk::Format = format
            .to_srgb_format()
            .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?
            .into();
        unsafe {
            let view = self.device().create_image_view(
                &vk::ImageViewCreateInfo {
                    image: self.vk_handle(),
                    view_type,
                    format: srgb_format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: self.aspects(),
                        base_mip_level: 0,
                        base_array_layer: 0,
                        level_count: self.mip_level_count(),
                        layer_count: self.array_layer_count(),
                    },
                    ..Default::default()
                }
                .push(&mut vk::ImageViewUsageCreateInfo {
                    usage,
                    ..Default::default()
                }),
                None,
            )?;
            Ok(SrgbImageView {
                srgb_view: ImageViewItem {
                    ty: view_type,
                    format: srgb_format,
                    mip_levels: 0..self.mip_level_count(),
                    array_layers: 0..self.array_layer_count(),
                    view,
                },
                image: self,
            })
        }
    }

    /// Creates a [`UintImageView`] that reinterprets the image's texels using its
    /// `UINT`-format counterpart (e.g. an `R8G8B8A8_UNORM` image is viewed as
    /// `R8G8B8A8_UINT`).
    ///
    /// The image must have been created with [`vk::ImageCreateFlags::MUTABLE_FORMAT`] and
    /// a format list ([`vk::ImageFormatListCreateInfo`]) that includes the `UINT` variant.
    ///
    /// # Parameters
    /// - `uint_usage` - Usage flags for the `UINT` view
    ///
    /// # Errors
    /// Returns [`vk::Result::ERROR_FORMAT_NOT_SUPPORTED`] if the image's format has no
    /// `UINT` counterpart with an identical bit layout.
    fn create_uint_view(self, uint_usage: vk::ImageUsageFlags) -> VkResult<UintImageView<Self>>
    where
        Self: HasDevice + Sized,
    {
        let view_type = match self.ty() {
            vk::ImageType::TYPE_1D if self.array_layer_count() > 1 => {
                vk::ImageViewType::TYPE_1D_ARRAY
            }
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_2D if self.array_layer_count() > 1 => {
                vk::ImageViewType::TYPE_2D_ARRAY
            }
            vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => unreachable!(),
        };
        unsafe {
            let uint_format: vk::Format = pumicite_types::format::Format::from(self.format())
                .to_uint_format()
                .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?
                .into();
            let view = self.device().create_image_view(
                &vk::ImageViewCreateInfo {
                    image: self.vk_handle(),
                    view_type,
                    format: uint_format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: self.aspects(),
                        base_mip_level: 0,
                        base_array_layer: 0,
                        level_count: self.mip_level_count(),
                        layer_count: self.array_layer_count(),
                    },
                    ..Default::default()
                }
                .push(&mut vk::ImageViewUsageCreateInfo {
                    usage: uint_usage,
                    ..Default::default()
                }),
                None,
            )?;
            Ok(UintImageView {
                uint_view: ImageViewItem {
                    ty: view_type,
                    format: uint_format,
                    mip_levels: 0..self.mip_level_count(),
                    array_layers: 0..self.array_layer_count(),
                    view,
                },
                image: self,
            })
        }
    }

    /// Creates a [`SwizzledImageView`] whose view remaps components on read
    /// according to `components` (e.g. `{r,g,b,a} <- A` to broadcast alpha into
    /// every channel).
    ///
    /// The view keeps the image's format — only sampling is remapped — so no
    /// `MUTABLE_FORMAT`/format-list is required on the image. `usage` restricts
    /// the view's usage; it must exclude [`vk::ImageUsageFlags::STORAGE`], which
    /// Vulkan only permits with an identity swizzle.
    fn create_swizzled_view(
        self,
        components: vk::ComponentMapping,
        usage: vk::ImageUsageFlags,
    ) -> VkResult<SwizzledImageView<Self>>
    where
        Self: HasDevice + Sized,
    {
        let view_type = match self.ty() {
            vk::ImageType::TYPE_1D if self.array_layer_count() > 1 => {
                vk::ImageViewType::TYPE_1D_ARRAY
            }
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_2D if self.array_layer_count() > 1 => {
                vk::ImageViewType::TYPE_2D_ARRAY
            }
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
                    components,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: self.aspects(),
                        base_mip_level: 0,
                        base_array_layer: 0,
                        level_count: self.mip_level_count(),
                        layer_count: self.array_layer_count(),
                    },
                    ..Default::default()
                }
                .push(&mut vk::ImageViewUsageCreateInfo {
                    usage,
                    ..Default::default()
                }),
                None,
            )?;
            Ok(SwizzledImageView {
                swizzled_view: ImageViewItem {
                    ty: view_type,
                    format: self.format(),
                    mip_levels: 0..self.mip_level_count(),
                    array_layers: 0..self.array_layer_count(),
                    view,
                },
                image: self,
            })
        }
    }
}

impl<T> ImageExt for T where T: ImageLike {}
