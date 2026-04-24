//! Vulkan surface creation for Bevy windows.
//!
//! This module handles the creation of Vulkan surfaces (`VkSurfaceKHR`) for Bevy windows.
//! Surfaces are the bridge between the windowing system and Vulkan rendering.
//!
//! # Platform Support
//!
//! Surface creation is platform-specific. This module automatically enables the
//! appropriate Vulkan surface extension based on the current platform:
//!
//! - **Windows**: `VK_KHR_win32_surface`
//! - **Linux/X11**: `VK_KHR_xlib_surface`
//! - **Linux/Wayland**: `VK_KHR_wayland_surface`
//! - **macOS/iOS**: `VK_EXT_metal_surface`
//! - **Android**: `VK_KHR_android_surface`
//!
//! # Usage
//!
//! [`SurfacePlugin`] must be added **before** [`PumicitePlugin`](crate::PumicitePlugin)
//! because it configures instance extensions.
//!
//! Surfaces are automatically created for windows when they're spawned, and the
//! [`Surface`](pumicite::Surface) and [`SurfaceCapabilities`] components are added to
//! the window entity.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_window::{RawHandleWrapper, Window};
use bevy_winit::DisplayHandleWrapper;
use pumicite::ash::{amd::display_native_hdr, ext, vk};
use pumicite::utils::{AsVkHandle, NextChainMap};
use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};

use pumicite::ash::ext::swapchain_colorspace::Meta as ExtSwapchainColorspace;
use pumicite::ash::khr::get_surface_capabilities2::Meta as KhrSurfaceCapabilities2;
use pumicite::ash::khr::surface::Meta as KhrSurface;
use pumicite::ash::khr::surface_maintenance1::Meta as KhrSurfaceMaintenance1;
use pumicite::{Device, Instance, Surface, ash};

use super::PumiciteApp;

#[derive(Component)]
pub struct SurfaceCapabilities(pub NextChainMap<vk::SurfaceCapabilities2KHR<'static>>);

impl std::fmt::Debug for SurfaceCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let surface_capabilities = &self.0.head.surface_capabilities;

        let mut f = f.debug_struct("SurfaceCapabilities");
        f.field(
            "image_count(range)",
            &(surface_capabilities.min_image_count..=surface_capabilities.max_image_count),
        )
        .field(
            "current_extent",
            &format!(
                "{}x{}",
                surface_capabilities.current_extent.width,
                surface_capabilities.current_extent.height
            ),
        )
        .field(
            "image_extent(range)",
            &format!(
                "{}x{} ..= {}x{}",
                surface_capabilities.min_image_extent.width,
                surface_capabilities.min_image_extent.height,
                surface_capabilities.max_image_extent.width,
                surface_capabilities.max_image_extent.height
            ),
        )
        .field(
            "max_image_array_layers",
            &surface_capabilities.max_image_array_layers,
        )
        .field(
            "supported_transforms",
            &surface_capabilities.supported_transforms,
        )
        .field("current_transform", &surface_capabilities.current_transform)
        .field(
            "supported_composite_alpha",
            &surface_capabilities.supported_composite_alpha,
        )
        .field(
            "supported_usage_flags",
            &surface_capabilities.supported_usage_flags,
        );

        if let Some(caps) = self
            .0
            .get::<vk::DisplayNativeHdrSurfaceCapabilitiesAMD<'static>>()
        {
            f.field(
                "local_dimming_support",
                &(caps.local_dimming_support == vk::TRUE),
            );
        } else {
            f.field(
                "local_dimming_support",
                &"Requires VkDisplayNativeHdrSurfaceCapabilitiesAMD",
            );
        }
        if let Some(caps) = self
            .0
            .get::<vk::SurfaceCapabilitiesFullScreenExclusiveEXT<'static>>()
        {
            f.field(
                "full_screen_exclusive_supported",
                &(caps.full_screen_exclusive_supported == vk::TRUE),
            );
        } else {
            f.field(
                "full_screen_exclusive_supported",
                &"Requires VkSurfaceCapabilitiesFullScreenExclusiveEXT",
            );
        }

        if let Some(metadata) = self.0.get::<vk::HdrMetadataEXT<'static>>() {
            #[derive(Debug)]
            #[allow(unused)]
            struct HdrMetadata {
                display_primary: [WhitePoint; 3],
                white_point: WhitePoint,
                luminance_range: std::ops::Range<f32>,
                max_content_light_level: f32,
                max_frame_average_light_level: f32,
            }
            struct WhitePoint(f32, f32);
            impl std::fmt::Debug for WhitePoint {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_fmt(format_args!("({},{})", self.0, self.1))
                }
            }
            f.field(
                "hdr_metadata",
                &HdrMetadata {
                    display_primary: [
                        WhitePoint(
                            metadata.display_primary_red.x,
                            metadata.display_primary_red.y,
                        ),
                        WhitePoint(
                            metadata.display_primary_green.x,
                            metadata.display_primary_green.y,
                        ),
                        WhitePoint(
                            metadata.display_primary_blue.x,
                            metadata.display_primary_blue.y,
                        ),
                    ],
                    white_point: WhitePoint(metadata.white_point.x, metadata.white_point.y),
                    luminance_range: metadata.min_luminance..metadata.max_luminance,
                    max_content_light_level: metadata.max_content_light_level,
                    max_frame_average_light_level: metadata.max_frame_average_light_level,
                },
            );
        } else {
            f.field("hdr_metadata", &"Requires VkHdrMetadataEXT");
        }

        f.finish()
    }
}

pub(super) fn extract_surfaces(
    mut commands: Commands,
    instance: Res<Instance>,
    device: Res<Device>,
    mut window_created_events: MessageReader<bevy_window::WindowCreated>,
    query: Query<
        (
            &RawHandleWrapper,
            Option<&Surface>,
            Option<&SurfaceCapabilities>,
        ),
        With<Window>,
    >,
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: Option<
        NonSend<bevy_ecs::system::NonSendMarker>,
    >,
) {
    for create_event in window_created_events.read() {
        let (raw_handle, surface, surface_capabilities) = query.get(create_event.window).unwrap();
        let raw_handle = unsafe { raw_handle.get_handle() };
        assert!(surface.is_none());
        assert!(surface_capabilities.is_none());
        let new_surface = Surface::create(instance.clone(), &raw_handle, &raw_handle).unwrap();
        let surface_capabilities =
            get_surface_capabilities(&instance, &device, &new_surface).unwrap();
        tracing::info!("Created new surface: {:#?}", surface_capabilities);
        commands
            .entity(create_event.window)
            .insert(new_surface)
            .insert(surface_capabilities);
    }
}

fn get_surface_capabilities(
    instance: &Instance,
    device: &Device,
    surface: &Surface,
) -> ash::VkResult<SurfaceCapabilities> {
    let mut capabilities = NextChainMap::<vk::SurfaceCapabilities2KHR<'static>>::default();
    let physical_device = device.physical_device();

    let Ok(surface_capabilities2) = instance.get_extension::<KhrSurfaceCapabilities2>() else {
        capabilities.head.surface_capabilities =
            physical_device.get_surface_capabilities(surface)?;
        return Ok(SurfaceCapabilities(capabilities));
    };

    if device.has_extension_named(display_native_hdr::NAME) {
        capabilities.get_mut_or_insert_with::<vk::DisplayNativeHdrSurfaceCapabilitiesAMD<'static>>(
            |_| Default::default(),
        );
    }
    if device.has_extension_named(ext::full_screen_exclusive::NAME) {
        capabilities
            .get_mut_or_insert_with::<vk::SurfaceCapabilitiesFullScreenExclusiveEXT<'static>>(
                |_| Default::default(),
            );
    }
    if device.has_extension_named(ext::hdr_metadata::NAME) {
        capabilities.get_mut_or_insert_with::<vk::HdrMetadataEXT<'static>>(|_| Default::default());
    }
    capabilities.make_chain();

    let surface_info = vk::PhysicalDeviceSurfaceInfo2KHR::default().surface(surface.vk_handle());
    unsafe {
        surface_capabilities2.get_physical_device_surface_capabilities2(
            physical_device.vk_handle(),
            &surface_info,
            &mut capabilities.head,
        )?;
    }

    Ok(SurfaceCapabilities(capabilities))
}

/// Plugin that enables Vulkan surface creation for windows.
///
/// This plugin is an **instance plugin** and must be added **before**
/// [`PumicitePlugin`](crate::PumicitePlugin).
///
/// # What It Does
///
/// 1. Enables the `VK_KHR_surface` instance extension
/// 2. Enables platform-specific surface extensions (e.g., `VK_KHR_win32_surface`)
/// 3. Optionally enables surface capability extensions for HDR/advanced features
/// 4. Registers a system to create surfaces for new windows
///
/// # Components
///
/// When a window is created, the following component is added to its entity:
/// - [`Surface`](pumicite::Surface) - The Vulkan surface handle
/// - [`SurfaceCapabilities`] - The queried capabilities for that surface
#[derive(Default)]
pub struct SurfacePlugin;
impl Plugin for SurfacePlugin {
    fn build(&self, app: &mut App) {
        app.add_instance_extension::<KhrSurface>().unwrap();
        app.add_instance_extension::<ExtSwapchainColorspace>().ok();
        app.add_instance_extension::<KhrSurfaceCapabilities2>().ok();
        app.add_instance_extension::<KhrSurfaceMaintenance1>().ok();

        if let Some(display_handle) = app.world().get_resource::<DisplayHandleWrapper>() {
            match display_handle.display_handle().unwrap().as_raw() {
                #[cfg(target_os = "windows")]
                RawDisplayHandle::Windows(_) => {
                    app.add_instance_extension::<ash::khr::win32_surface::Meta>()
                        .unwrap();
                }
                #[cfg(target_os = "linux")]
                RawDisplayHandle::Xlib(_) => {
                    app.add_instance_extension::<ash::khr::xlib_surface::Meta>()
                        .unwrap();
                }
                #[cfg(target_os = "linux")]
                RawDisplayHandle::Xcb(_) => {
                    app.add_instance_extension::<ash::khr::xcb_surface::Meta>()
                        .unwrap();
                }
                #[cfg(target_os = "linux")]
                RawDisplayHandle::Wayland(_) => {
                    app.add_instance_extension::<ash::khr::wayland_surface::Meta>()
                        .unwrap();
                }
                #[cfg(target_os = "android")]
                RawDisplayHandle::Android(_) => {
                    app.add_instance_extension::<ash::khr::android_surface::Meta>()
                        .unwrap();
                }
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                RawDisplayHandle::UiKit(_) | RawDisplayHandle::AppKit(_) => {
                    app.add_instance_extension::<ash::ext::metal_surface::Meta>()
                        .unwrap();
                }
                _ => tracing::warn!("Your display is not supported."),
            };
        } else {
            panic!("pumicite::SurfacePlugin must be inserted after bevy_winit::WinitPlugin.")
        };

        app.add_systems(PostUpdate, extract_surfaces);
    }
}
