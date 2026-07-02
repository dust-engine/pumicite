use crate::HasDevice;
use crate::utils::AsVkHandle;
use ash::vk;



pub trait AsMTLTexture: AsVkHandle<Handle = vk::Image> + HasDevice {
    /// Returns the underlying Metal texture (`id<MTLTexture>`) backing this image.
    ///
    /// `plane` selects the aspect/plane to export. For single-plane images this is
    /// [`vk::ImageAspectFlags::PLANE_0`]; for multi-planar images pass the desired
    /// `PLANE_0`/`PLANE_1`/`PLANE_2` aspect.
    ///
    /// Requires the [`VK_EXT_metal_objects`](ash::ext::metal_objects) extension.
    fn mtl_texture(
        &self,
        plane: vk::ImageAspectFlags,
    ) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture> {
        let mut texture_info = vk::ExportMetalTextureInfoEXT::default()
            .image(self.vk_handle())
            .plane(plane);
        let mut info = vk::ExportMetalObjectsInfoEXT::default();
        info.p_next = (&mut texture_info as *mut vk::ExportMetalTextureInfoEXT).cast();
        unsafe {
            self.device()
                .extension::<ash::ext::metal_objects::Meta>()
                .export_metal_objects(&mut info);
            &*texture_info
                .mtl_texture
                .cast::<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>()
        }
    }
}

impl<T> AsMTLTexture for T where T: AsVkHandle<Handle = vk::Image> + HasDevice {}


pub trait AsMTLIOSurface: AsVkHandle<Handle = vk::Image> + HasDevice {
    /// Returns the `IOSurface` backing this image, or `null` if the image is not
    /// backed by one.
    ///
    /// Requires the [`VK_EXT_metal_objects`](ash::ext::metal_objects) extension.
    fn io_surface(&self) -> vk::IOSurfaceRef {
        let mut io_surface_info =
            vk::ExportMetalIOSurfaceInfoEXT::default().image(self.vk_handle());
        let mut info = vk::ExportMetalObjectsInfoEXT::default();
        info.p_next = (&mut io_surface_info as *mut vk::ExportMetalIOSurfaceInfoEXT).cast();
        unsafe {
            self.device()
                .extension::<ash::ext::metal_objects::Meta>()
                .export_metal_objects(&mut info);
        }
        io_surface_info.io_surface
    }
}

impl<T> AsMTLIOSurface for T where T: AsVkHandle<Handle = vk::Image> + HasDevice {}


pub trait AsMTLCommandQueue: AsVkHandle<Handle = vk::Queue> + HasDevice {
    /// Returns the underlying Metal command queue (`id<MTLCommandQueue>`) backing
    /// this queue.
    ///
    /// Requires the [`VK_EXT_metal_objects`](ash::ext::metal_objects) extension.
    fn mtl_command_queue(
        &self,
    ) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandQueue> {
        let mut queue_info = vk::ExportMetalCommandQueueInfoEXT::default().queue(self.vk_handle());
        let mut info = vk::ExportMetalObjectsInfoEXT::default();
        info.p_next = (&mut queue_info as *mut vk::ExportMetalCommandQueueInfoEXT).cast();
        unsafe {
            self.device()
                .extension::<ash::ext::metal_objects::Meta>()
                .export_metal_objects(&mut info);
            &*queue_info
                .mtl_command_queue
                .cast::<objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandQueue>>()
        }
    }
}

impl<T> AsMTLCommandQueue for T where T: AsVkHandle<Handle = vk::Queue> + HasDevice {}


pub trait AsMTLSharedEvent: AsVkHandle<Handle = vk::Semaphore> + HasDevice {
    /// Returns the underlying Metal shared event (`id<MTLSharedEvent>`) backing
    /// this semaphore.
    ///
    /// Requires the [`VK_EXT_metal_objects`](ash::ext::metal_objects) extension.
    fn get_mtl_shared_event(
        &self,
    ) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent> {
        let mut event_info =
            vk::ExportMetalSharedEventInfoEXT::default().semaphore(self.vk_handle());
        let mut info = vk::ExportMetalObjectsInfoEXT::default();
        info.p_next = (&mut event_info as *mut vk::ExportMetalSharedEventInfoEXT).cast();
        unsafe {
            self.device()
                .extension::<ash::ext::metal_objects::Meta>()
                .export_metal_objects(&mut info);
            &*event_info
                .mtl_shared_event
                .cast::<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>()
        }
    }
}

impl<T> AsMTLSharedEvent for T where T: AsVkHandle<Handle = vk::Semaphore> + HasDevice {}


pub trait AsMTLDevice: HasDevice {
    /// Returns the underlying Metal device (`id<MTLDevice>`) backing this logical
    /// device.
    ///
    /// Requires the [`VK_EXT_metal_objects`](ash::ext::metal_objects) extension.
    fn mtl_device(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice> {
        let mut device_info = vk::ExportMetalDeviceInfoEXT::default();
        let mut info = vk::ExportMetalObjectsInfoEXT::default();
        info.p_next = (&mut device_info as *mut vk::ExportMetalDeviceInfoEXT).cast();
        unsafe {
            self.device()
                .extension::<ash::ext::metal_objects::Meta>()
                .export_metal_objects(&mut info);
            &*device_info
                .mtl_device
                .cast::<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>()
        }
    }
}

impl<T> AsMTLDevice for T where T: HasDevice {}


pub trait AsMTLCommandBuffer: AsVkHandle<Handle = vk::CommandBuffer> + HasDevice {
    /// Returns the underlying Metal command buffer (`id<MTL4CommandBuffer>`)
    /// backing this command buffer.
    fn mtl_command_buffer(
        &self,
    ) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandBuffer> {
        let mut queue_info = vk::ExportMetalCommandQueueInfoEXT::default();
        queue_info.s_type = vk::StructureType::from_raw(1000311012);
        let mut info = vk::ExportMetalObjectsInfoEXT::default();
        info.p_next = (&mut queue_info as *mut vk::ExportMetalCommandQueueInfoEXT).cast();
        unsafe {
            self.device()
                .extension::<ash::ext::metal_objects::Meta>()
                .export_metal_objects(&mut info);
            &*queue_info
                .mtl_command_queue
                .cast::<objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandBuffer>>()
        }
    }
}

impl<T> AsMTLCommandBuffer for T where T: AsVkHandle<Handle = vk::CommandBuffer> + HasDevice {}
