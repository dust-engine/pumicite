use ash::vk::{Handle, TaggedStructure};
use glam::UVec2;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::sync::Arc;

use crate::command::{GPURef, GPURefMut};
use crate::fault::{DeviceError, DeviceResult, ResultWithDevice};
use crate::image::{ImageLike, ImageViewLike};
use crate::physical_device::PhysicalDevice;
use crate::utils::AsVkHandle;
use crate::{Device, Surface};
use crate::{HasDevice, queue::Queue};
use ash::khr::swapchain::Meta as KhrSwapchain;
use ash::{VkResult, vk};

pub struct Swapchain {
    inner: Arc<SwapchainInner>,
    images: SmallVec<[Option<Box<SwapchainImageInner>>; 3]>,
    extra_acquire_semaphore: vk::Semaphore,
    extra_acquire_fence: vk::Fence,
    /// Tracks whether the swapchain is still valid. Once `VK_ERROR_OUT_OF_DATE_KHR` is returned,
    /// calling Vulkan functions on the swapchain is an error, so we mark it invalid.
    is_valid: bool,
    /// Set when a recreate attempt failed after passing the current swapchain as `oldSwapchain`,
    /// which retires it. The next recreate must then pass `VK_NULL_HANDLE` instead.
    is_retired: bool,
    has_maintenance_1: bool,
}
impl Swapchain {
    pub fn extent(&self) -> UVec2 {
        self.inner.extent
    }
}
impl HasDevice for Swapchain {
    fn device(&self) -> &Device {
        &self.inner.device
    }
}
impl Drop for Swapchain {
    fn drop(&mut self) {
        // Drop the images first, so their Drop waits for pending acquires and presents.
        self.images.clear();
        unsafe {
            // The extra fence was reset after its last acquire completed, so it isn't in use.
            self.inner
                .device
                .destroy_semaphore(self.extra_acquire_semaphore, None);
            self.inner
                .device
                .destroy_fence(self.extra_acquire_fence, None);
        }
    }
}

struct SwapchainInner {
    device: Device,
    surface: Surface,
    inner: vk::SwapchainKHR,
    format: vk::Format,

    color_space: vk::ColorSpaceKHR,
    extent: UVec2,
    layer_count: u32,
}
impl SwapchainInner {
    fn create(
        device: Device,
        surface: Surface,
        info: &SwapchainCreateInfo,
        old_swapchain: vk::SwapchainKHR,
    ) -> VkResult<(
        Arc<SwapchainInner>,
        SmallVec<[Option<Box<SwapchainImageInner>>; 3]>,
    )> {
        // Validate that extent has positive dimensions. On some platforms (Windows minimized,
        // Wayland), the window extent can become (0, 0) which is not valid for swapchain creation.
        if info.image_extent.x == 0 || info.image_extent.y == 0 {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }

        let format: pumicite_types::format::Format = info.image_format.into();
        let srgb_format = format.to_srgb_format();
        let linear_format = format.to_linear_format();
        let possible_formats: [vk::Format; 2] = [
            linear_format.into(),
            srgb_format
                .map(|x| x.into())
                .unwrap_or(vk::Format::UNDEFINED),
        ];
        let mut possible_formats = vk::ImageFormatListCreateInfo {
            view_format_count: if srgb_format.is_none() { 1 } else { 2 },
            p_view_formats: possible_formats.as_ptr(),
            ..Default::default()
        };

        let mut create_info = vk::SwapchainCreateInfoKHR {
            flags: info.flags | vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT,
            surface: surface.vk_handle(),
            min_image_count: info.min_image_count,
            image_format: linear_format.into(),
            image_color_space: info.image_color_space,
            image_extent: vk::Extent2D {
                width: info.image_extent.x,
                height: info.image_extent.y,
            },
            image_array_layers: info.image_array_layers,
            image_usage: info.image_usage,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: info.pre_transform,
            composite_alpha: info.composite_alpha,
            present_mode: info.present_mode,
            clipped: info.clipped.into(),
            old_swapchain,
            ..Default::default()
        }
        .push(&mut possible_formats);
        let mut present_scaling = vk::SwapchainPresentScalingCreateInfoKHR {
            scaling_behavior: info.present_scaling,
            present_gravity_x: info.present_gravity[0],
            present_gravity_y: info.present_gravity[1],
            ..Default::default()
        };
        if !info.present_scaling.is_empty() {
            create_info = create_info.push(&mut present_scaling);
        }
        if !info
            .image_usage
            .intersects(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
        {
            // A spec workaround so that a view can always be created, even if the swapchain was only created with TRANSFER_DST.
            create_info.image_usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        let swapchain_loader = device.extension::<KhrSwapchain>();
        let new_swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };

        let images = unsafe { swapchain_loader.get_swapchain_images(new_swapchain)? };

        let inner = SwapchainInner {
            device,
            surface,
            inner: new_swapchain,
            extent: info.image_extent,
            layer_count: info.image_array_layers,
            format: info.image_format,
            color_space: info.image_color_space,
        };
        let inner = Arc::new(inner);

        let images: SmallVec<[_; 3]> = images
            .into_iter()
            .enumerate()
            .map(|(i, image)| {
                let srgb_view = if let Some(srgb_format) = srgb_format {
                    // Exclude STORAGE usage for sRGB views since sRGB formats
                    // don't support VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT.
                    let srgb_usage = info.image_usage & !vk::ImageUsageFlags::STORAGE;
                    unsafe {
                        inner.device.create_image_view(
                            &vk::ImageViewCreateInfo {
                                image,
                                view_type: vk::ImageViewType::TYPE_2D,
                                format: srgb_format.into(),
                                components: vk::ComponentMapping::default(),
                                subresource_range: vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                                ..Default::default()
                            }
                            .push(&mut vk::ImageViewUsageCreateInfo {
                                usage: srgb_usage,
                                ..Default::default()
                            }),
                            None,
                        )?
                    }
                } else {
                    vk::ImageView::null()
                };
                let linear_view = unsafe {
                    inner.device.create_image_view(
                        &vk::ImageViewCreateInfo {
                            image,
                            view_type: vk::ImageViewType::TYPE_2D,
                            format: linear_format.into(),
                            components: vk::ComponentMapping::default(),
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            ..Default::default()
                        },
                        None,
                    )?
                };

                let acquire_semaphore = unsafe {
                    inner
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
                };
                inner
                    .device
                    .set_debug_name(acquire_semaphore, c"Swapchain Acquire Semaphore")
                    .ok();

                let present_semaphore = unsafe {
                    inner
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
                };
                inner
                    .device
                    .set_debug_name(present_semaphore, c"Swapchain Present Semaphore")
                    .ok();
                let present_fence = unsafe {
                    inner.device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )?
                };
                inner
                    .device
                    .set_debug_name(present_fence, c"Present Fence")
                    .ok();
                let acquire_fence = unsafe {
                    inner.device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )?
                };
                inner
                    .device
                    .set_debug_name(acquire_fence, c"Acquire Fence")
                    .ok();
                let img = SwapchainImageInner {
                    image,
                    indice: i as u32,
                    swapchain: inner.clone(),
                    srgb_view: SwapchainImageView(
                        srgb_view,
                        srgb_format
                            .map(|f| f.into())
                            .unwrap_or(vk::Format::UNDEFINED),
                    ),
                    linear_view: SwapchainImageView(linear_view, linear_format.into()),
                    acquire_semaphore,
                    acquire_fence,
                    present_semaphore,
                    present_fence,
                    last_present_queue: vk::Queue::null(),
                };
                Ok(Some(Box::new(img)))
            })
            .collect::<VkResult<Vec<Option<Box<SwapchainImageInner>>>>>()?
            .into();
        Ok((inner, images))
    }
}

impl Drop for SwapchainInner {
    fn drop(&mut self) {
        unsafe {
            self.device
                .extension::<KhrSwapchain>()
                .destroy_swapchain(self.inner, None);
        }
    }
}

pub struct SwapchainCreateInfo {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: UVec2,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub clipped: bool,
    /// How the presentation engine shows an image whose extent doesn't match the surface.
    /// Requires the `swapchainMaintenance1` feature.
    pub present_scaling: vk::PresentScalingFlagsKHR,
    /// When `present_scaling` is enabled, this specifies
    /// where a scaled image is placed on the surface, along x and y.
    pub present_gravity: [vk::PresentGravityFlagsKHR; 2],
}

/// Unsafe APIs for Swapchain
impl Swapchain {
    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html>
    pub fn create(device: Device, surface: Surface, info: &SwapchainCreateInfo) -> VkResult<Self> {
        tracing::info!(
            width = %info.image_extent.x,
            height = %info.image_extent.y,
            format = ?info.image_format,
            color_space = ?info.image_color_space,
            "Creating swapchain"
        );
        let (inner, images) =
            SwapchainInner::create(device.clone(), surface, info, vk::SwapchainKHR::null())?;
        let extra_acquire_semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        device
            .set_debug_name(extra_acquire_semaphore, c"Swapchain Acquire Semaphore")
            .ok();
        let extra_acquire_fence =
            unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        device
            .set_debug_name(extra_acquire_fence, c"Swapchain Acquire Fence")
            .ok();

        let has_maintenance_1 = device
            .feature::<vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT>()
            .is_some_and(|x| x.swapchain_maintenance1 == vk::TRUE);
        Ok(Swapchain {
            images,
            inner,
            extra_acquire_semaphore,
            extra_acquire_fence,
            is_valid: true,
            is_retired: false,
            has_maintenance_1,
        })
    }

    /// Recreates the swapchain. On failure (including a zero-sized extent, e.g. a minimized
    /// window, reported as `VK_ERROR_OUT_OF_DATE_KHR`) the swapchain is marked invalid so that
    /// `acquire` keeps failing with `VK_ERROR_OUT_OF_DATE_KHR` until a later recreate succeeds.
    pub fn recreate(&mut self, info: &SwapchainCreateInfo) -> VkResult<()> {
        if info.image_extent.x == 0 || info.image_extent.y == 0 {
            self.is_valid = false;
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
        tracing::info!(
            width = %info.image_extent.x,
            height = %info.image_extent.y,
            format = ?info.image_format,
            color_space = ?info.image_color_space,
            "Recreating swapchain"
        );

        let old_swapchain = if self.is_retired {
            vk::SwapchainKHR::null()
        } else {
            self.inner.inner
        };
        match SwapchainInner::create(
            self.device().clone(),
            self.inner.surface.clone(),
            info,
            old_swapchain,
        ) {
            Ok((inner, images)) => {
                self.inner = inner;
                self.images = images;
                self.is_valid = true;
                self.is_retired = false;
                Ok(())
            }
            Err(e) => {
                // oldSwapchain is retired even if vkCreateSwapchainKHR fails, and a retired
                // swapchain must not be passed as oldSwapchain again.
                self.is_valid = false;
                self.is_retired = true;
                Err(e)
            }
        }
    }

    /// Returns false once the swapchain has reported `VK_ERROR_OUT_OF_DATE_KHR`
    /// or a recreate failed; it must be recreated before further use.
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    pub fn acquire(&mut self) -> DeviceResult<(Box<SwapchainImageInner>, bool)> {
        // Check if swapchain is still valid. Once invalidated by VK_ERROR_OUT_OF_DATE_KHR,
        // the swapchain must be recreated before further use.
        if !self.is_valid {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR.into());
        }

        unsafe {
            let result = self
                .device()
                .extension::<KhrSwapchain>()
                .acquire_next_image(
                    self.inner.inner,
                    !0,
                    // We don't know which swapchain image we will get, so use the extra acquire semaphore.
                    self.extra_acquire_semaphore,
                    self.extra_acquire_fence,
                );

            let (indice, suboptimal) = match result {
                Ok(val) => val,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.is_valid = false;
                    return Err(vk::Result::ERROR_OUT_OF_DATE_KHR.into());
                }
                Err(e) => return Err(DeviceError::new(e, self.device())),
            };

            let mut image = self.images[indice as usize].take().unwrap();
            // The image takes the semaphore and fence this acquire signals, and its previous
            // pair becomes the extra pair for the next acquire.
            std::mem::swap(
                &mut image.acquire_semaphore,
                &mut self.extra_acquire_semaphore,
            );
            std::mem::swap(&mut image.acquire_fence, &mut self.extra_acquire_fence);
            self.device()
                .wait_for_fences(&[self.extra_acquire_fence], true, !0)
                .with_device(self.device())?;
            self.device().reset_fences(&[self.extra_acquire_fence])?;
            if self.has_maintenance_1 {
                // The previous present of this image must finish before its present semaphore
                // can be signaled again.
                self.device()
                    .wait_for_fences(&[image.present_fence], true, !0)
                    .with_device(self.device())?;
            }

            Ok((image, suboptimal))
        }
    }

    /// The image must have been transitioned into [`vk::ImageLayout::PRESENT_SRC_KHR`] before calling this function,
    /// by a submission that signals its [`present_semaphore`](SwapchainImageInner::present_semaphore).
    pub fn present(
        &mut self,
        mut image: Box<SwapchainImageInner>,
        queue: &mut Queue,
    ) -> DeviceResult<()> {
        // Check if swapchain is still valid. Once invalidated by VK_ERROR_OUT_OF_DATE_KHR,
        // the swapchain must be recreated before further use.
        if !self.is_valid {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR.into());
        }

        unsafe {
            let indice = image.indice;
            assert!(self.images[indice as usize].is_none());
            if !self.has_maintenance_1 {
                // Workaround: when we don't have VK_KHR_swapchain_maintenance1, set
                // last_present_queue to be the last presented queue.
                image.last_present_queue = queue.vk_handle();
            }

            {
                let wait_semaphores = [image.present_semaphore];
                let image_indices = [image.indice];
                let swapchains = [image.swapchain.inner];
                let mut present_info = vk::PresentInfoKHR::default()
                    .wait_semaphores(&wait_semaphores)
                    .image_indices(&image_indices)
                    .swapchains(&swapchains);
                let fences = [image.present_fence];
                let mut fence_info = vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);
                if self.has_maintenance_1 {
                    // If has maintenance 1, signal the fence at the actual queue present step.
                    self.device().reset_fences(&fences)?;
                    present_info = present_info.push(&mut fence_info);
                }
                let present_result = self
                    .device()
                    .extension::<KhrSwapchain>()
                    .queue_present(queue.vk_handle(), &present_info);

                // Always return the image to the array (if swapchain not retired)
                if Arc::ptr_eq(&image.swapchain, &self.inner) {
                    self.images[indice as usize] = Some(image);
                } // otherwise, the swapchain was retired.

                // Handle present errors after returning the image
                match present_result {
                    Ok(false) => Ok(()),
                    Ok(true) => Ok(()), // suboptimal is not an error
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        self.is_valid = false;
                        Err(vk::Result::ERROR_OUT_OF_DATE_KHR.into())
                    }
                    Err(e) => Err(DeviceError::new(e, self.device())),
                }
            }
        }
    }
}

pub struct SwapchainImageInner {
    image: vk::Image,
    linear_view: SwapchainImageView,
    srgb_view: SwapchainImageView,
    indice: u32,
    swapchain: Arc<SwapchainInner>,

    acquire_semaphore: vk::Semaphore,
    acquire_fence: vk::Fence,
    present_semaphore: vk::Semaphore,
    present_fence: vk::Fence,

    /// Workaround for when implementation does not have VK_KHR_swapchain_maintenance1.
    last_present_queue: vk::Queue,
}

impl SwapchainImageInner {
    pub fn color_space(&self) -> vk::ColorSpaceKHR {
        self.swapchain.color_space
    }
    /// The binary semaphore that the acquire of this image signals. The submission that
    /// renders to the image must wait on it.
    pub fn acquire_semaphore(&self) -> vk::Semaphore {
        self.acquire_semaphore
    }
    /// The binary semaphore that [`Swapchain::present`] waits on. The submission that renders
    /// to the image must signal it.
    pub fn present_semaphore(&self) -> vk::Semaphore {
        self.present_semaphore
    }
}
crate::command::project_host_metadata! {
    impl[] SwapchainImageInner {
        /// The swapchain's color space.
        fn color_space(&self) -> vk::ColorSpaceKHR;
    }
}

impl Drop for SwapchainImageInner {
    fn drop(&mut self) {
        unsafe {
            self.swapchain
                .device
                .wait_for_fences(&[self.acquire_fence, self.present_fence], true, !0)
                .with_device(&self.swapchain.device)
                .unwrap();
        }
        if self.last_present_queue != vk::Queue::null() {
            unsafe {
                // Workaround: when VK_KHR_swapchain_maintenance1 is unavailable,
                // we conservatively wait for the queue itself to become idle.
                self.swapchain
                    .device
                    .queue_wait_idle(self.last_present_queue)
                    .with_device(&self.swapchain.device)
                    .unwrap();
            }
        }
        unsafe {
            self.swapchain
                .device
                .destroy_fence(self.acquire_fence, None);
            self.swapchain
                .device
                .destroy_fence(self.present_fence, None);
            self.swapchain
                .device
                .destroy_semaphore(self.acquire_semaphore, None);
            self.swapchain
                .device
                .destroy_semaphore(self.present_semaphore, None);
        }
        if !self.linear_view.0.is_null() {
            unsafe {
                self.swapchain
                    .device
                    .destroy_image_view(self.linear_view.0, None);
            }
        }
        if !self.srgb_view.0.is_null() {
            unsafe {
                self.swapchain
                    .device
                    .destroy_image_view(self.srgb_view.0, None);
            }
        }
    }
}
impl AsVkHandle for SwapchainImageInner {
    type Handle = vk::Image;

    fn vk_handle(&self) -> Self::Handle {
        self.image
    }
}
// Safety: there's one `SwapchainImageInner` per swapchain image. `Swapchain::acquire` moves it
// out of `images` and `Swapchain::present` moves it back, so only one value names each image.
// It isn't `Clone`, and it keeps the swapchain alive through `swapchain`.
unsafe impl ImageLike for SwapchainImageInner {
    fn aspects(&self) -> vk::ImageAspectFlags {
        vk::ImageAspectFlags::COLOR
    }

    fn array_layer_count(&self) -> u32 {
        self.swapchain.layer_count
    }

    fn mip_level_count(&self) -> u32 {
        1
    }

    fn extent(&self) -> glam::UVec3 {
        glam::UVec3::new(self.swapchain.extent.x, self.swapchain.extent.y, 1)
    }

    fn format(&self) -> vk::Format {
        self.swapchain.format
    }

    fn ty(&self) -> vk::ImageType {
        vk::ImageType::TYPE_2D
    }
}

pub struct SwapchainImageView(vk::ImageView, vk::Format);
impl AsVkHandle for SwapchainImageView {
    type Handle = vk::ImageView;
    fn vk_handle(&self) -> Self::Handle {
        self.0
    }
}
// Safety: a `SwapchainImageView` holds only a view handle and its format.
unsafe impl crate::command::NoHostMapping for SwapchainImageView {}
// Safety: `SwapchainImageView`s only exist inside the `SwapchainImageInner` that owns their
// image, which destroys them on drop. Their fields are private and they aren't `Clone`, so
// they're only reachable through it.
unsafe impl ImageViewLike for SwapchainImageView {
    fn ty(&self) -> vk::ImageViewType {
        vk::ImageViewType::TYPE_2D
    }

    fn array_layers(&self) -> std::ops::Range<u32> {
        0..1
    }

    fn mip_levels(&self) -> std::ops::Range<u32> {
        0..1
    }

    fn format(&self) -> vk::Format {
        self.1
    }
}

impl SwapchainImageInner {
    pub fn linear_view(&self) -> &SwapchainImageView {
        debug_assert!(!self.linear_view.0.is_null());
        &self.linear_view
    }
    /// An image view of the swapchain image in sRGB format.
    ///
    /// May be null if the swapchain format doesn't have a corresponding srgb format.
    /// See [to_srgb_format](`pumicite_types::format::Format::to_srgb_format`)
    pub fn srgb_view(&self) -> Option<&SwapchainImageView> {
        if self.srgb_view.0.is_null() {
            None
        } else {
            Some(&self.srgb_view)
        }
    }
}
impl crate::sync::GPUMutex<SwapchainImageInner> {
    pub fn linear_view(&self) -> &SwapchainImageView {
        self.inner.linear_view()
    }
    /// An image view of the swapchain image in sRGB format.
    ///
    /// May be null if the swapchain format doesn't have a corresponding srgb format.
    /// See [to_srgb_format](`pumicite_types::format::Format::to_srgb_format`)
    pub fn srgb_view(&self) -> Option<&SwapchainImageView> {
        self.inner.srgb_view()
    }
}
impl<'a> GPURef<'a, SwapchainImageInner> {
    pub fn linear_view(self) -> GPURef<'a, SwapchainImageView> {
        unsafe { GPURef::new_unchecked(self.get_unchecked().linear_view()) }
    }
    pub fn srgb_view(self) -> Option<GPURef<'a, SwapchainImageView>> {
        unsafe {
            self.get_unchecked()
                .srgb_view()
                .map(|v| GPURef::new_unchecked(v))
        }
    }
}
impl<'a> GPURefMut<'a, SwapchainImageInner> {
    pub fn linear_view(self) -> GPURefMut<'a, SwapchainImageView> {
        unsafe { GPURefMut::new_unchecked(self.get_unchecked().linear_view()) }
    }
    pub fn srgb_view(self) -> Option<GPURefMut<'a, SwapchainImageView>> {
        unsafe {
            self.get_unchecked()
                .srgb_view()
                .map(|v| GPURefMut::new_unchecked(v))
        }
    }
}
/// Specifies the desired swapchain color output mode.
///
/// Each variant corresponds to a distinct rendering pipeline with different format,
/// color space, and post-processing requirements.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SwapchainColorMode {
    /// Standard dynamic range output with hardware sRGB encoding.
    ///
    /// Creates a swapchain with [`vk::ColorSpaceKHR::SRGB_NONLINEAR`] and an 8-bit-per-
    /// channel format. The selected format always has a corresponding sRGB format,
    /// so [`SwapchainImageInner::srgb_view`] is guaranteed to return `Some`, enabling
    /// the implementation to perform sRGB encoding on writes.
    SDR8Bit,

    /// Standard dynamic range output, with potentially 10-bit color depth.
    ///
    /// Creates a swapchain with [`vk::ColorSpaceKHR::SRGB_NONLINEAR`].
    /// Prefers [`vk::Format::A2B10G10R10_UNORM_PACK32`] when available for reduced
    /// banding, falling back to [`vk::Format::B8G8R8A8_UNORM`].
    ///
    /// **Caveat**: 10-bit packed formats have no hardware sRGB counterpart, so
    /// [`SwapchainImageInner::srgb_view`] returns `None` when this format is selected;
    /// the application must perform its own sRGB encoding in shader. If you want
    /// hardware sRGB encoding via [`SwapchainImageInner::srgb_view`], use
    /// [`SwapchainColorMode::SDR8Bit`] instead.
    SDR10Bit,

    /// HDR via scRGB linear float16. Easiest HDR migration path.
    ///
    /// Creates a swapchain with [`vk::Format::R16G16B16A16_SFLOAT`].
    /// Colors are encoded in linear BT.709 (Rec.709) primaries. Values above 1.0
    /// represent HDR luminance (1.0 = 80 nits SDR white). No gamut mapping or OETF
    /// is needed — the compositor handles the conversion to the display’s native space.
    ScRgbLinear,

    /// HDR with explicit color space requiring tone mapping, gamut mapping, and OETF.
    ///
    /// Creates a swapchain with [`vk::Format::A2B10G10R10_UNORM_PACK32`]. The actual
    /// color space selected depends on display capabilities. The application **must** query
    /// [`SwapchainImageInner::color_space`] and configure its post-processing pipeline
    /// (gamut mapping matrix, transfer function) accordingly.
    HDR,
}

pub fn get_surface_preferred_format(
    surface: &Surface,
    physical_device: &PhysicalDevice,
    required_feature_flags: vk::FormatFeatureFlags,
    color_mode: &SwapchainColorMode,
) -> Option<vk::SurfaceFormatKHR> {
    let supported_formats = physical_device.get_surface_formats(surface).unwrap();

    let supported_formats: HashSet<_> = supported_formats
        .iter()
        .filter(|&surface_format| {
            let format_properties = unsafe {
                physical_device
                    .instance()
                    .get_physical_device_format_properties(
                        physical_device.vk_handle(),
                        surface_format.format,
                    )
            };
            format_properties
                .optimal_tiling_features
                .contains(required_feature_flags)
                | format_properties
                    .linear_tiling_features
                    .contains(required_feature_flags)
        })
        .collect();

    fn find(
        candidates: impl IntoIterator<Item = vk::SurfaceFormatKHR>,
        supported_formats: HashSet<&vk::SurfaceFormatKHR>,
    ) -> Option<vk::SurfaceFormatKHR> {
        candidates
            .into_iter()
            .find(|f| supported_formats.contains(f))
    }

    const SDR_10BIT_CANDIDATES: &[vk::SurfaceFormatKHR] = &[
        vk::SurfaceFormatKHR {
            format: vk::Format::A2B10G10R10_UNORM_PACK32,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
        vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
        vk::SurfaceFormatKHR {
            format: vk::Format::R8G8B8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
    ];

    /// 8-bit-only candidates. Each format has a corresponding sRGB format, so
    /// [`SwapchainImageInner::srgb_view`] is guaranteed to be available.
    const SDR_8BIT_CANDIDATES: &[vk::SurfaceFormatKHR] = &[
        vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
        vk::SurfaceFormatKHR {
            format: vk::Format::R8G8B8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
    ];

    const SCRGB_CANDIDATES: &[vk::SurfaceFormatKHR] = &[
        vk::SurfaceFormatKHR {
            format: vk::Format::R16G16B16A16_SFLOAT,
            color_space: vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD,
        },
        vk::SurfaceFormatKHR {
            format: vk::Format::R16G16B16A16_SFLOAT,
            color_space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
        },
    ];

    const DEFAULT_HDR_COLOR_SPACES: &[vk::ColorSpaceKHR] = &[
        vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD,
        vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        vk::ColorSpaceKHR::HDR10_HLG_EXT,
        vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
        vk::ColorSpaceKHR::DISPLAY_P3_LINEAR_EXT,
        vk::ColorSpaceKHR::BT2020_LINEAR_EXT,
        vk::ColorSpaceKHR::BT709_LINEAR_EXT,
        vk::ColorSpaceKHR::PASS_THROUGH_EXT,
    ];

    match color_mode {
        SwapchainColorMode::SDR10Bit => {
            find(SDR_10BIT_CANDIDATES.iter().copied(), supported_formats)
        }
        SwapchainColorMode::SDR8Bit => find(SDR_8BIT_CANDIDATES.iter().copied(), supported_formats),
        SwapchainColorMode::ScRgbLinear => find(
            SCRGB_CANDIDATES.iter().chain(SDR_8BIT_CANDIDATES).copied(),
            supported_formats,
        ),
        SwapchainColorMode::HDR => DEFAULT_HDR_COLOR_SPACES
            .iter()
            .find_map(|&cs| {
                let candidate = vk::SurfaceFormatKHR {
                    format: vk::Format::A2B10G10R10_UNORM_PACK32,
                    color_space: cs,
                };
                supported_formats.contains(&candidate).then_some(candidate)
            })
            .or_else(|| {
                find(
                    SCRGB_CANDIDATES.iter().chain(SDR_10BIT_CANDIDATES).copied(),
                    supported_formats,
                )
            }),
    }
}
