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
//! [`Surface`](pumicite::Surface) component is added to the window entity.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_window::{RawHandleWrapper, Window};
use bevy_winit::DisplayHandleWrapper;
use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};

use pumicite::ash::ext::surface_maintenance1::Meta as ExtSurfaceMaintenance1;
use pumicite::ash::ext::swapchain_colorspace::Meta as ExtSwapchainColorspace;
use pumicite::ash::khr::get_surface_capabilities2::Meta as KhrSurfaceCapabilities2;
use pumicite::ash::khr::surface::Meta as KhrSurface;
use pumicite::{Instance, Surface, ash};

use super::PumiciteApp;

pub(super) fn extract_surfaces(
    mut commands: Commands,
    instance: Res<Instance>,
    mut window_created_events: MessageReader<bevy_window::WindowCreated>,
    query: Query<(&RawHandleWrapper, Option<&Surface>), With<Window>>,
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: Option<
        NonSend<bevy_ecs::system::NonSendMarker>,
    >,
) {
    for create_event in window_created_events.read() {
        let (raw_handle, surface) = query.get(create_event.window).unwrap();
        let raw_handle = unsafe { raw_handle.get_handle() };
        assert!(surface.is_none());
        let new_surface = Surface::create(instance.clone(), &raw_handle, &raw_handle).unwrap();
        commands.entity(create_event.window).insert(new_surface);
    }
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
#[derive(Default)]
pub struct SurfacePlugin;
impl Plugin for SurfacePlugin {
    fn build(&self, app: &mut App) {
        app.add_instance_extension::<KhrSurface>().unwrap();
        app.add_instance_extension::<ExtSwapchainColorspace>().ok();
        app.add_instance_extension::<KhrSurfaceCapabilities2>().ok();
        app.add_instance_extension::<ExtSurfaceMaintenance1>().ok();

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
