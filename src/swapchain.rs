use ash::vk::{Handle, TaggedStructure};
use glam::UVec2;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::sync::Arc;

use crate::debug::DebugObject;
use crate::image::{ImageLike, ImageViewLike};
use crate::physical_device::PhysicalDevice;
use crate::tracking::ResourceState;
use crate::{Device, Surface, utils::SharingMode};
use crate::{HasDevice, Queue, sync::GPUMutex, utils::format::ColorSpace};
use crate::{sync::SharedSemaphore, utils::AsVkHandle};
use ash::khr::swapchain::Meta as KhrSwapchain;
use ash::{VkResult, vk};

pub struct Swapchain {
    inner: Arc<SwapchainInner>,
    images: SmallVec<[Option<Box<SwapchainImageInner>>; 3]>,
    extra_acquire_semaphore: SharedSemaphore,
    /// Tracks whether the swapchain is still valid. Once `VK_ERROR_OUT_OF_DATE_KHR` is returned,
    /// calling Vulkan functions on the swapchain is an error, so we mark it invalid.
    is_valid: bool,
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

struct SwapchainInner {
    device: Device,
    surface: Surface,
    inner: vk::SwapchainKHR,
    format: vk::Format,

    color_space: ColorSpace,
    extent: UVec2,
    layer_count: u32,
    command_pool: vk::CommandPool,
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

        let format: crate::utils::format::Format = info.image_format.into();
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
        if !info
            .image_usage
            .intersects(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
        {
            // A spec workaround so that a view can always be created, even if the swapchain was only created with TRANSFER_DST.
            create_info.image_usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        match &info.image_sharing_mode {
            SharingMode::Exclusive => (),
            SharingMode::Concurrent {
                queue_family_indices,
            } => {
                create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
                create_info.p_queue_family_indices = queue_family_indices.as_ptr();
            }
        }
        let swapchain_loader = device.extension::<KhrSwapchain>();
        let new_swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };

        let images = unsafe { swapchain_loader.get_swapchain_images(new_swapchain)? };

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                        | vk::CommandPoolCreateFlags::TRANSIENT,
                    queue_family_index: info.queue_family_index,
                    ..Default::default()
                },
                None,
            )?
        };
        let inner = SwapchainInner {
            device,
            surface,
            inner: new_swapchain,
            extent: info.image_extent,
            layer_count: info.image_array_layers,
            format: info.image_format,
            color_space: info.image_color_space.into(),
            command_pool,
        };
        let inner = Arc::new(inner);

        let images: SmallVec<[_; 3]> = images
            .into_iter()
            .enumerate()
            .map(|(i, image)| {
                let srgb_view = if let Some(srgb_format) = srgb_format {
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
                            },
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
                let img = SwapchainImageInner {
                    image,
                    indice: i as u32,
                    swapchain: inner.clone(),
                    srgb_view: SwapchainImageView(srgb_view),
                    linear_view: SwapchainImageView(linear_view),
                    acquire_semaphore: SharedSemaphore::new_binary(inner.device.clone(), true)?
                        .with_name(c"Swapchain Acquire Semaphore"),
                    present_semaphore: SharedSemaphore::new_binary(inner.device.clone(), true)?
                        .with_name(c"Swapchain Present Semaphore"),
                    command_buffer: unsafe {
                        let mut command_buffer = vk::CommandBuffer::null();
                        (inner.device.fp_v1_0().allocate_command_buffers)(
                            inner.device.handle(),
                            &vk::CommandBufferAllocateInfo {
                                command_pool,
                                level: vk::CommandBufferLevel::PRIMARY,
                                command_buffer_count: 1,
                                ..Default::default()
                            },
                            &mut command_buffer,
                        )
                        .result()?;
                        inner
                            .device
                            .set_debug_name(command_buffer, c"Swapchain Img Layout Command Buffer")
                            .ok();
                        command_buffer
                    },
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
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

pub struct SwapchainCreateInfo<'a> {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: UVec2,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub image_sharing_mode: SharingMode<&'a [u32]>,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub clipped: bool,
    /// Queue family index for the command pool used for pre-present layout transitions.
    /// This should match the queue family of the queue used for presentation.
    pub queue_family_index: u32,
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
        Ok(Swapchain {
            images,
            inner,
            extra_acquire_semaphore: SharedSemaphore::new_binary(device, false)?
                .with_name(c"Swapchain Acquire Semaphore"),
            is_valid: true,
        })
    }

    pub fn recreate(&mut self, info: &SwapchainCreateInfo) -> VkResult<()> {
        tracing::info!(
            width = %info.image_extent.x,
            height = %info.image_extent.y,
            format = ?info.image_format,
            color_space = ?info.image_color_space,
            "Recreating swapchain"
        );

        let (inner, images) = SwapchainInner::create(
            self.device().clone(),
            self.inner.surface.clone(),
            info,
            self.inner.inner,
        )?;
        self.inner = inner;
        self.images = images;
        self.is_valid = true;
        Ok(())
    }

    pub fn acquire(&mut self) -> VkResult<(GPUMutex<SwapchainImageInner>, bool)> {
        // Check if swapchain is still valid. Once invalidated by VK_ERROR_OUT_OF_DATE_KHR,
        // the swapchain must be recreated before further use.
        if !self.is_valid {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }

        unsafe {
            let result = self
                .device()
                .extension::<KhrSwapchain>()
                .acquire_next_image(
                    self.inner.inner,
                    !0,
                    self.extra_acquire_semaphore.vk_handle(),
                    self.extra_acquire_semaphore.raw_fence(),
                );

            let (indice, suboptimal) = match result {
                Ok(val) => val,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.is_valid = false;
                    return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
                }
                Err(e) => return Err(e),
            };

            let mut image = self.images[indice as usize].take().unwrap();
            image.acquire_semaphore.wait_blocked(0, !0)?;
            std::mem::swap(
                &mut image.acquire_semaphore,
                &mut self.extra_acquire_semaphore,
            );

            let acquire_semaphore = image.acquire_semaphore.clone();
            image.present_semaphore.wait_blocked(0, !0).unwrap();
            Ok((
                GPUMutex::new_locked(image, acquire_semaphore, 0),
                suboptimal,
            ))
        }
    }
    pub fn present(
        &mut self,
        image: GPUMutex<SwapchainImageInner>,
        image_state: ResourceState,
        queue: &mut Queue,
    ) -> VkResult<()> {
        // Check if swapchain is still valid. Once invalidated by VK_ERROR_OUT_OF_DATE_KHR,
        // the swapchain must be recreated before further use.
        if !self.is_valid {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }

        unsafe {
            let (image, semaphore, wait_value) = image.into_inner();
            let indice = image.indice;
            assert!(self.images[indice as usize].is_none());
            {
                // Record pre-present command buffer
                self.device().reset_command_buffer(
                    image.command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )?;
                self.device().begin_command_buffer(
                    image.command_buffer,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )?;
                self.device().cmd_pipeline_barrier2(
                    image.command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2 {
                            src_access_mask: image_state.write.access,
                            src_stage_mask: image_state.write.stage | image_state.reads,
                            dst_access_mask: vk::AccessFlags2::empty(),
                            dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                            old_layout: image_state.layout,
                            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                            image: image.image,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: vk::REMAINING_MIP_LEVELS,
                                base_array_layer: 0,
                                layer_count: vk::REMAINING_ARRAY_LAYERS,
                            },
                            ..Default::default()
                        },
                    ]),
                );
                self.device()
                    .end_command_buffer(image.command_buffer)
                    .unwrap();
            }
            let has_maintenance_1 = self
                .device()
                .feature::<vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT>()
                .is_some_and(|x| x.swapchain_maintenance1 == vk::TRUE);
            self.device().queue_submit2(
                queue.vk_handle(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo {
                        command_buffer: image.command_buffer,
                        ..Default::default()
                    }])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo {
                        // unwrap the semaphore because there should always be a semaphore here
                        // because we need to at least wait on the swapchain acquire semaphore.
                        semaphore: semaphore.unwrap().vk_handle(),
                        value: wait_value,
                        stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                        ..Default::default()
                    }])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo {
                        semaphore: image.present_semaphore.vk_handle(),
                        stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                        value: 0,
                        ..Default::default()
                    }])],
                if has_maintenance_1 {
                    vk::Fence::null()
                } else {
                    // If without maintenance 1, signal the fence at the end of this submission.
                    image.present_semaphore.raw_fence()
                },
            )?;

            {
                let wait_semaphores = [image.present_semaphore.vk_handle()];
                let image_indices = [image.indice];
                let swapchains = [image.swapchain.inner];
                let mut present_info = vk::PresentInfoKHR::default()
                    .wait_semaphores(&wait_semaphores)
                    .image_indices(&image_indices)
                    .swapchains(&swapchains);
                let fences = [image.present_semaphore.raw_fence()];
                let mut fence_info = vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);
                if has_maintenance_1 {
                    // If has maintenance 1, signal the fence at the actual queue present step.
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
                        Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
                    }
                    Err(e) => Err(e),
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
    /// A command buffer for pre-present commands
    command_buffer: vk::CommandBuffer,

    acquire_semaphore: SharedSemaphore,
    present_semaphore: SharedSemaphore,
}

impl SwapchainImageInner {
    pub fn color_space(&self) -> ColorSpace {
        self.swapchain.color_space.clone()
    }
}

impl Drop for SwapchainImageInner {
    fn drop(&mut self) {
        self.acquire_semaphore.wait_blocked(0, !0).unwrap();
        self.present_semaphore.wait_blocked(0, !0).unwrap();
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
impl ImageLike for SwapchainImageInner {
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

pub struct SwapchainImageView(vk::ImageView);
impl AsVkHandle for SwapchainImageView {
    type Handle = vk::ImageView;
    fn vk_handle(&self) -> Self::Handle {
        self.0
    }
}
impl ImageViewLike for SwapchainImageView {
    fn ty(&self) -> vk::ImageViewType {
        vk::ImageViewType::TYPE_2D
    }

    fn array_layer_count(&self) -> u32 {
        1
    }

    fn mip_level_count(&self) -> u32 {
        1
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
    /// See [to_srgb_format](`crate::utils::format::Format::to_srgb_format`)
    pub fn srgb_view(&self) -> Option<&SwapchainImageView> {
        if self.srgb_view.0.is_null() {
            None
        } else {
            Some(&self.srgb_view)
        }
    }
}
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum HDRMode {
    /// Creates a swapchain with [`vk::Format::B8G8R8A8_SRGB`]. The display engine expects the color stored in the
    /// swapchain to be in sRGB color space ([`vk::ColorSpaceKHR::SRGB_NONLINEAR`]).
    Off,

    /// If possible, creates a swapchain with 10 bit colors ([`vk::Format::A2B10G10R10_UNORM_PACK32`]).
    /// The display engine expects the color stored in the
    /// swapchain to be in sRGB color space ([`vk::ColorSpaceKHR::SRGB_NONLINEAR`]).
    ///
    /// Falls back to a swapchain with 8 bit color. When operating in this mode, the application should apply the
    /// gamut curve in postprocessing.
    Vivid,

    /// This display mode was made to help ease developers that have a Rec709 pipeline into rendering
    /// in HDR without having to significantly change their color pipelines. The colors in the framebuffer
    /// for this mode are expected to be encoded in Rec709, but when displayed, if a color is outside of the
    /// Rec709 RGB cube but inside the display native’s color space, then the color will be correctly displayed
    /// (otherwise it will be clamped to the monitor’s native color space).
    ///
    /// Creates a swapchain with [`vk::Format::R16G16B16A16_SFLOAT`]. Color should be stored into the
    /// swapchain image in scRGB color space ([`vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT`]) **with no gamma curve applied**.
    Progressive,

    /// This display mode was made to provide developers with more sophisticated HDR color pipelines easy integration
    /// of their pipelines with tone and gamut mapping. The colors in the framebuffer for this mode are expected to be
    /// encoded in a supported color space queried by the application.
    ///
    ///
    /// Creates a swapchain with [`vk::Format::A2B10G10R10_UNORM_PACK32`]. Application should query
    /// [`SwapchainImageInner::color_space`] and store colors into the swapchain in this color space.
    On,
}

pub fn get_surface_preferred_format(
    surface: &Surface,
    physical_device: &PhysicalDevice,
    required_feature_flags: vk::FormatFeatureFlags,
    hdr_mode: HDRMode,
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
    let query_list: &'static [vk::SurfaceFormatKHR] = match hdr_mode {
        HDRMode::Off => &[vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }],
        HDRMode::Vivid => &[
            vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ],
        HDRMode::Progressive => &[
            vk::SurfaceFormatKHR {
                format: vk::Format::R16G16B16A16_SFLOAT,
                color_space: vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::R16G16B16A16_SFLOAT,
                color_space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ],
        HDRMode::On => &[
            vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::HDR10_ST2084_EXT,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::HDR10_HLG_EXT,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::PASS_THROUGH_EXT,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::BT2020_LINEAR_EXT,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ],
    };
    for query in query_list.iter() {
        if supported_formats.contains(query) {
            return Some(*query);
        }
    }
    None
}
