use crate::{Instance, physical_device::PhysicalDevice, utils::AsVkHandle};
use ash::{
    VkResult,
    khr::{self, surface::Meta as KhrSurface},
    vk,
};
use raw_window_handle::{DisplayHandle, RawDisplayHandle, RawWindowHandle, WindowHandle};
use std::sync::Arc;

#[derive(Clone)]
pub struct Surface(Arc<SurfaceInner>);
struct SurfaceInner {
    instance: Instance,
    handle: vk::SurfaceKHR,
}
impl Drop for SurfaceInner {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .extension::<KhrSurface>()
                .destroy_surface(self.handle, None);
        }
    }
}
impl Surface {
    pub fn create(
        instance: Instance,
        window_handle: &impl raw_window_handle::HasWindowHandle,
        display_handle: &impl raw_window_handle::HasDisplayHandle,
    ) -> VkResult<Surface> {
        let surface = unsafe {
            create_surface(
                &instance,
                display_handle.display_handle().unwrap(),
                window_handle.window_handle().unwrap(),
            )?
        };
        Ok(Surface(Arc::new(SurfaceInner {
            instance,
            handle: surface,
        })))
    }
}
impl AsVkHandle for Surface {
    type Handle = vk::SurfaceKHR;
    fn vk_handle(&self) -> Self::Handle {
        self.0.handle
    }
}

impl PhysicalDevice {
    pub fn get_surface_capabilities(
        &self,
        surface: &Surface,
    ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            surface
                .0
                .instance
                .extension::<KhrSurface>()
                .get_physical_device_surface_capabilities(self.vk_handle(), surface.0.handle)
        }
    }
    pub fn get_surface_formats(&self, surface: &Surface) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            surface
                .0
                .instance
                .extension::<KhrSurface>()
                .get_physical_device_surface_formats(self.vk_handle(), surface.0.handle)
        }
    }
    pub fn get_surface_present_modes(
        &self,
        surface: &Surface,
    ) -> VkResult<Vec<vk::PresentModeKHR>> {
        unsafe {
            surface
                .0
                .instance
                .extension::<KhrSurface>()
                .get_physical_device_surface_present_modes(self.vk_handle(), surface.0.handle)
        }
    }
    pub fn supports_surface(&self, surface: &Surface, queue_family_index: u32) -> VkResult<bool> {
        unsafe {
            surface
                .0
                .instance
                .extension::<KhrSurface>()
                .get_physical_device_surface_support(
                    self.vk_handle(),
                    queue_family_index,
                    surface.0.handle,
                )
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn create_surface(
    instance: &Instance,
    display_handle: DisplayHandle,
    window_handle: WindowHandle,
) -> VkResult<vk::SurfaceKHR> {
    match (display_handle.as_raw(), window_handle.as_raw()) {
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => instance
            .extension::<khr::win32_surface::Meta>()
            .create_win32_surface(
                &vk::Win32SurfaceCreateInfoKHR {
                    hinstance: window.hinstance.unwrap().get() as ash::vk::HINSTANCE,
                    hwnd: window.hwnd.get() as ash::vk::HWND,
                    ..Default::default()
                },
                None,
            ),

        (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => instance
            .extension::<khr::wayland_surface::Meta>()
            .create_wayland_surface(
                &vk::WaylandSurfaceCreateInfoKHR {
                    display: display.display.as_ptr(),
                    surface: window.surface.as_ptr(),
                    ..Default::default()
                },
                None,
            ),

        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => instance
            .extension::<khr::xlib_surface::Meta>()
            .create_xlib_surface(
                &vk::XlibSurfaceCreateInfoKHR {
                    dpy: display.display.unwrap().as_ptr() as *mut _,
                    window: window.window,
                    ..Default::default()
                },
                None,
            ),

        (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => instance
            .extension::<khr::xcb_surface::Meta>()
            .create_xcb_surface(
                &vk::XcbSurfaceCreateInfoKHR {
                    connection: display.connection.unwrap().as_ptr(),
                    window: window.window.get(),
                    ..Default::default()
                },
                None,
            ),

        (RawDisplayHandle::Android(_), RawWindowHandle::AndroidNdk(window)) => instance
            .extension::<khr::android_surface::Meta>()
            .create_android_surface(
                &vk::AndroidSurfaceCreateInfoKHR {
                    window: window.a_native_window.as_ptr(),
                    ..Default::default()
                },
                None,
            ),

        #[cfg(target_os = "macos")]
        (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(window)) => {
            use raw_window_metal::{Layer, appkit};

            let layer = match appkit::metal_layer_from_handle(window) {
                Layer::Existing(layer) | Layer::Allocated(layer) => layer as *mut _,
            };

            let surface_desc = vk::MetalSurfaceCreateInfoEXT {
                p_layer: &*layer,
                ..Default::default()
            };
            instance
                .extension::<ash::ext::metal_surface::Meta>()
                .create_metal_surface(&surface_desc, None)
        }

        #[cfg(target_os = "ios")]
        (RawDisplayHandle::UiKit(_), RawWindowHandle::UiKit(window)) => {
            use raw_window_metal::{Layer, uikit};

            let layer = match uikit::metal_layer_from_handle(window) {
                Layer::Existing(layer) | Layer::Allocated(layer) => layer as *mut _,
            };

            let surface_desc = vk::MetalSurfaceCreateInfoEXT::builder().layer(&*layer);
            instance
                .extension::<ash::ext::MetalSurface>()
                .create_metal_surface(&surface_desc, None)
        }

        _ => Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
    }
}
