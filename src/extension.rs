use ash::{Device, vk};

pub trait Extension: ash::vk::ExtensionMeta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device);
}
impl<T> Extension for T
where
    T: ash::vk::ExtensionMeta,
{
    default fn promote_device(_device: &mut ash::Device, _ext: &Self::Device) {}
}
#[derive(Debug)]
pub enum MissingFeatureError {
    Extension(&'static std::ffi::CStr),
    Feature {
        feature_set: &'static str,
        feature: &'static str,
    },
}
impl From<MissingFeatureError> for ash::vk::Result {
    fn from(error: MissingFeatureError) -> Self {
        match error {
            MissingFeatureError::Extension(_) => vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            MissingFeatureError::Feature { .. } => vk::Result::ERROR_FEATURE_NOT_PRESENT,
        }
    }
}

impl Extension for ash::khr::dynamic_rendering::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.cmd_begin_rendering = ext.fp().cmd_begin_rendering_khr;
        device.device_fn_1_3.cmd_end_rendering = ext.fp().cmd_end_rendering_khr;
    }
}

impl Extension for ash::khr::synchronization2::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.cmd_pipeline_barrier2 = ext.fp().cmd_pipeline_barrier2_khr;
        device.device_fn_1_3.cmd_reset_event2 = ext.fp().cmd_reset_event2_khr;
        device.device_fn_1_3.cmd_set_event2 = ext.fp().cmd_set_event2_khr;
        device.device_fn_1_3.cmd_wait_events2 = ext.fp().cmd_wait_events2_khr;
        device.device_fn_1_3.cmd_write_timestamp2 = ext.fp().cmd_write_timestamp2_khr;
        device.device_fn_1_3.queue_submit2 = ext.fp().queue_submit2_khr;
    }
}

impl Extension for ash::khr::maintenance1::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_1.trim_command_pool = ext.fp().trim_command_pool_khr;
    }
}

impl Extension for ash::khr::maintenance3::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_1.get_descriptor_set_layout_support =
            ext.fp().get_descriptor_set_layout_support_khr;
    }
}

impl Extension for ash::khr::maintenance4::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.get_device_buffer_memory_requirements =
            ext.fp().get_device_buffer_memory_requirements_khr;
        device.device_fn_1_3.get_device_image_memory_requirements =
            ext.fp().get_device_image_memory_requirements_khr;
        device
            .device_fn_1_3
            .get_device_image_sparse_memory_requirements =
            ext.fp().get_device_image_sparse_memory_requirements_khr;
    }
}

impl Extension for ash::ext::extended_dynamic_state::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.cmd_set_cull_mode = ext.fp().cmd_set_cull_mode_ext;
        device.device_fn_1_3.cmd_set_front_face = ext.fp().cmd_set_front_face_ext;
        device.device_fn_1_3.cmd_set_primitive_topology = ext.fp().cmd_set_primitive_topology_ext;
        device.device_fn_1_3.cmd_set_viewport_with_count = ext.fp().cmd_set_viewport_with_count_ext;
        device.device_fn_1_3.cmd_set_scissor_with_count = ext.fp().cmd_set_scissor_with_count_ext;
        device.device_fn_1_3.cmd_bind_vertex_buffers2 = ext.fp().cmd_bind_vertex_buffers2_ext;
        device.device_fn_1_3.cmd_set_depth_test_enable = ext.fp().cmd_set_depth_test_enable_ext;
        device.device_fn_1_3.cmd_set_depth_write_enable = ext.fp().cmd_set_depth_write_enable_ext;
        device.device_fn_1_3.cmd_set_depth_compare_op = ext.fp().cmd_set_depth_compare_op_ext;
        device.device_fn_1_3.cmd_set_depth_bounds_test_enable =
            ext.fp().cmd_set_depth_bounds_test_enable_ext;
        device.device_fn_1_3.cmd_set_stencil_test_enable = ext.fp().cmd_set_stencil_test_enable_ext;
        device.device_fn_1_3.cmd_set_stencil_op = ext.fp().cmd_set_stencil_op_ext;
    }
}

#[allow(dead_code)]
struct ExposedDevice {
    pub handle: vk::Device,
    pub device_fn_1_0: ash::DeviceFnV1_0,
    pub device_fn_1_1: ash::DeviceFnV1_1,
    pub device_fn_1_2: ash::DeviceFnV1_2,
    pub device_fn_1_3: ash::DeviceFnV1_3,
}
impl ExposedDevice {
    pub fn new(device: &mut Device) -> &mut Self {
        unsafe { std::mem::transmute(device) }
    }
}
