//! Physical device enumeration and properties.
//!
//! This module provides the [`PhysicalDevice`] type for querying GPU capabilities
//! and selecting a device for logical device creation.
//!
//! # Overview
//!
//! A physical device represents a GPU in the system. Before creating a logical
//! device, you typically:
//!
//! 1. Enumerate available physical devices
//! 2. Query their properties and capabilities
//! 3. Select one based on your application's requirements
//!
//! # Example
//!
//! ```
//! # use std::sync::Arc;
//! # use pumicite::{Instance, ash::vk};
//! # let entry = Arc::new(unsafe { ash::Entry::load() }.unwrap());
//! # let instance = Instance::builder(entry).build().unwrap();
//! // Enumerate all GPUs
//! let physical_devices: Vec<_> = instance.enumerate_physical_devices().unwrap().collect();
//!
//! // Find a discrete GPU (or use any available)
//! let gpu = physical_devices.iter().find(|d| {
//!     d.properties().device_type == vk::PhysicalDeviceType::DISCRETE_GPU
//! }).unwrap_or(&physical_devices[0]);
//!
//! // Check properties
//! println!("Using: {:?}", gpu.properties().device_name());
//! ```
use crate::{
    MissingFeatureError,
    utils::{AsVkHandle, NextChainMap, Version, VkTaggedObject},
};

use super::Instance;
use ash::{
    VkResult, ext, khr, nv,
    vk::{self, ExtensionMeta, PromotionStatus, TaggedStructure},
};
use core::ffi::c_void;
use std::{
    collections::BTreeMap,
    ffi::CStr,
    ops::Deref,
    ptr::NonNull,
    sync::{Arc, RwLock},
};

/// A handle to a physical GPU device.
///
/// Physical devices represent GPUs (or other Vulkan implementations) available
/// on the system. They are enumerated from an [`Instance`] and used to query
/// device capabilities before creating a logical [`Device`](crate::Device).
///
/// This type is reference-counted and cheap to clone.
#[derive(Clone)]
pub struct PhysicalDevice(Arc<PhysicalDeviceInner>);
impl PartialEq for PhysicalDevice {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for PhysicalDevice {}

struct PhysicalDeviceInner {
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    properties: PhysicalDeviceProperties,
}

impl Instance {
    /// Enumerates all physical devices (GPUs) available on the system.
    ///
    /// Returns an iterator over [`PhysicalDevice`] handles that can be used to
    /// query device properties and create logical devices.
    pub fn enumerate_physical_devices<'a>(
        &'a self,
    ) -> VkResult<impl ExactSizeIterator<Item = PhysicalDevice> + 'a> {
        let pdevices = unsafe { self.deref().enumerate_physical_devices().unwrap() };
        Ok(pdevices.into_iter().map(|pdevice| {
            let properties = PhysicalDeviceProperties::new(self.clone(), pdevice);
            PhysicalDevice(Arc::new(PhysicalDeviceInner {
                instance: self.clone(),
                physical_device: pdevice,
                properties,
            }))
        }))
    }
}
impl AsVkHandle for PhysicalDevice {
    type Handle = vk::PhysicalDevice;

    fn vk_handle(&self) -> Self::Handle {
        self.0.physical_device
    }
}
impl PhysicalDevice {
    /// Returns the instance this physical device was enumerated from.
    pub fn instance(&self) -> &Instance {
        &self.0.instance
    }

    /// Queries image format properties for a specific configuration.
    ///
    /// Returns `Ok(None)` if the format is not supported for the given parameters.
    pub fn image_format_properties(
        &self,
        format_info: &vk::PhysicalDeviceImageFormatInfo2,
    ) -> VkResult<Option<vk::ImageFormatProperties2<'_>>> {
        let mut out = vk::ImageFormatProperties2::default();
        unsafe {
            match self
                .0
                .instance
                .get_physical_device_image_format_properties2(
                    self.0.physical_device,
                    format_info,
                    &mut out,
                ) {
                Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(None),
                Ok(_) => Ok(Some(out)),
                Err(_) => panic!(),
            }
        }
    }

    /// Queries format properties for a specific format.
    ///
    /// Returns the capabilities of the format for buffer, linear image, and
    /// optimal tiling image usage.
    pub fn format_properties(&self, format: vk::Format) -> vk::FormatProperties3<'static> {
        let mut format_properties3 = vk::FormatProperties3::default();
        let mut format_properties2 = vk::FormatProperties2::default().push(&mut format_properties3);
        unsafe {
            self.instance().get_physical_device_format_properties2(
                self.0.physical_device,
                format,
                &mut format_properties2,
            );
        }
        format_properties3
    }
    pub(crate) fn get_queue_family_properties(&self) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.0
                .instance
                .get_physical_device_queue_family_properties(self.0.physical_device)
        }
    }

    /// Returns the physical device properties.
    pub fn properties(&self) -> &PhysicalDeviceProperties {
        &self.0.properties
    }
}

/// Properties and capabilities of a physical device.
///
/// This struct caches physical device properties and provides access to
/// device-specific information like memory heaps, API version, and extended
/// properties via the [`get`](Self::get) method.
pub struct PhysicalDeviceProperties {
    instance: Instance,
    pdevice: vk::PhysicalDevice,
    inner: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    memory_type_map: MemoryTypeMap,
    properties: RwLock<BTreeMap<vk::StructureType, Box<VkTaggedObject>>>,
}
unsafe impl Send for PhysicalDeviceProperties {}
unsafe impl Sync for PhysicalDeviceProperties {}
impl PhysicalDeviceProperties {
    fn new(instance: Instance, pdevice: vk::PhysicalDevice) -> Self {
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let pdevice_properties = unsafe { instance.get_physical_device_properties(pdevice) };

        let memory_types =
            &memory_properties.memory_types[0..memory_properties.memory_type_count as usize];
        let memory_heaps =
            &memory_properties.memory_heaps[0..memory_properties.memory_heap_count as usize];
        let memory_type_map =
            MemoryTypeMap::new(memory_types, memory_heaps, pdevice_properties.device_type);

        Self {
            instance,
            pdevice,
            properties: Default::default(),
            memory_properties,
            memory_type_map,
            inner: pdevice_properties,
        }
    }

    /// Gets an extended property structure by type.
    ///
    /// This method lazily queries and caches extended device properties.
    /// Properties are fetched once and cached for subsequent calls.
    pub fn get<
        T: vk::Extends<vk::PhysicalDeviceProperties2<'static>>
            + vk::TaggedStructure<'static>
            + Default
            + 'static,
    >(
        &self,
    ) -> &T {
        let properties = self.properties.read().unwrap();
        if let Some(entry) = properties.get(&T::STRUCTURE_TYPE) {
            let item = entry.deref().downcast_ref::<T>().unwrap();
            let item: NonNull<T> = item.into();
            unsafe {
                // This is ok because entry is boxed and never removed as long as self is still alive.
                return item.as_ref();
            }
        }
        drop(properties);

        let mut wrapper = vk::PhysicalDeviceProperties2::default();
        let mut item = T::default();
        unsafe {
            wrapper.p_next = &mut item as *mut T as *mut c_void;
            self.instance
                .get_physical_device_properties2(self.pdevice, &mut wrapper);
        }
        let item = VkTaggedObject::new(item);
        let item_ptr = item.downcast_ref::<T>().unwrap();
        let item_ptr: NonNull<T> = item_ptr.into();

        let mut properties = self.properties.write().unwrap();
        properties.insert(T::STRUCTURE_TYPE, item);
        drop(properties);

        unsafe {
            // This is ok because entry is boxed and never removed as long as self is still alive.
            item_ptr.as_ref()
        }
    }

    /// Returns the device name as a C string.
    pub fn device_name(&self) -> &CStr {
        self.inner.device_name_as_c_str().unwrap()
    }

    /// Returns the maximum supported API version for this physical device.
    pub fn api_version(&self) -> Version {
        Version(self.inner.api_version)
    }

    /// Returns the driver version.
    pub fn driver_version(&self) -> Version {
        Version(self.inner.driver_version)
    }

    /// Returns the available memory types.
    pub fn memory_types(&self) -> &[vk::MemoryType] {
        &self.memory_properties.memory_types[0..self.memory_properties.memory_type_count as usize]
    }

    /// Returns the available memory heaps.
    pub fn memory_heaps(&self) -> &[vk::MemoryHeap] {
        &self.memory_properties.memory_heaps[0..self.memory_properties.memory_heap_count as usize]
    }

    /// Returns the pre-calculated memory type indices for common allocation strategies.
    pub fn memory_type_map(&self) -> &MemoryTypeMap {
        &self.memory_type_map
    }
}
impl Deref for PhysicalDeviceProperties {
    type Target = vk::PhysicalDeviceProperties;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// A memory type on a physical device.
pub struct MemoryType {
    /// Flags describing the memory type's properties.
    pub property_flags: vk::MemoryPropertyFlags,
    /// The index of the heap this memory type belongs to.
    pub heap_index: u32,
}

/// A memory heap on a physical device.
pub struct MemoryHeap {
    /// The size of the heap in bytes.
    pub size: vk::DeviceSize,
    /// Flags describing the heap's properties.
    pub flags: vk::MemoryHeapFlags,
    /// The current memory budget (if memory budget extension is available).
    pub budget: vk::DeviceSize,
    /// The current memory usage (if memory budget extension is available).
    pub usage: vk::DeviceSize,
}

/// Pre-calculated memory type indices for common buffer allocation strategies.
///
/// These indices are computed once during physical device enumeration based on
/// the available memory types and their properties. This avoids repeated lookups
/// during buffer allocation.
///
/// # Memory Type Selection
///
/// The selection algorithm accounts for the various memory type patterns seen
/// across different GPU architectures:
///
/// - **Intel integrated**: Single heap, all DEVICE_LOCAL + HOST_VISIBLE
/// - **NVIDIA/AMD discrete**: Separate VRAM (DEVICE_LOCAL) and system RAM (HOST_VISIBLE)
/// - **AMD with 256MB BAR**: Additional small DEVICE_LOCAL + HOST_VISIBLE heap
/// - **Resizable BAR (SAM)**: Entire VRAM accessible as DEVICE_LOCAL + HOST_VISIBLE
/// - **AMD APU**: 256MB "virtual" DEVICE_LOCAL heap, rest is HOST_VISIBLE only
#[derive(Debug, Clone, Copy)]
pub struct MemoryTypeMap {
    /// GPU-exclusive memory for render targets, scratch buffers, GPU-generated data.
    ///
    /// Selection: DEVICE_LOCAL required, prefers non-HOST_VISIBLE (pure VRAM is faster
    /// on discrete GPUs when not accessed via BAR).
    pub private: u32,

    /// Staging memory for CPU-to-GPU transfers.
    ///
    /// Selection: HOST_VISIBLE required, prefers HOST_COHERENT, avoids HOST_CACHED
    /// (write-combined memory is fine for sequential writes).
    pub host: u32,

    /// CPU-readable memory for GPU-to-CPU readback.
    ///
    /// Selection: HOST_VISIBLE + HOST_CACHED required (fast CPU reads),
    /// prefers DEVICE_LOCAL (benefits integrated GPUs).
    pub dynamic: u32,

    /// Upload memory for CPU-written, GPU-read data.
    ///
    /// Selection: DEVICE_LOCAL required, prefers HOST_VISIBLE (avoids staging).
    /// Check [`upload_host_visible`](Self::upload_host_visible) to determine
    /// if a staging copy is needed.
    pub upload: u32,

    /// Whether [`upload`](Self::upload) memory is host visible.
    ///
    /// True on discrete GPUs without resizable BAR. These GPUs do not have a large
    /// DEVICE_LOCAL, HOST_VISIBLE pool.
    pub upload_host_visible: bool,

    /// Memory for uniform buffers: guaranteed to be device-local and host-visible.
    ///
    /// May use the 256MB BAR on discrete GPUs without resizable BAR.
    /// Set to `u32::MAX` on GPUs that have no device-local host-visible memory type at all.
    pub uniform: u32,
}

impl MemoryTypeMap {
    /// Computes the memory type map for a physical device.
    ///
    /// # Panics
    ///
    /// Panics if required memory types cannot be found (exotic/unsupported hardware).
    pub(crate) fn new(
        memory_types: &[vk::MemoryType],
        memory_heaps: &[vk::MemoryHeap],
        device_type: vk::PhysicalDeviceType,
    ) -> Self {
        // Helper to check if a memory type has all required flags
        let has_flags = |mt: &vk::MemoryType, required: vk::MemoryPropertyFlags| {
            mt.property_flags.contains(required)
        };

        // Helper to get heap size for a memory type
        let heap_size = |mt: &vk::MemoryType| memory_heaps[mt.heap_index as usize].size;

        // === PRIVATE: DEVICE_LOCAL, prefer non-HOST_VISIBLE ===
        // On discrete GPUs, pure VRAM without BAR access is typically faster.
        let private = memory_types
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, mt)| has_flags(mt, vk::MemoryPropertyFlags::DEVICE_LOCAL))
            // Prefer non-HOST_VISIBLE (pure VRAM), then larger heaps
            .max_by_key(|(_, mt)| {
                let not_host_visible = !mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
                (not_host_visible, heap_size(mt))
            })
            .map(|(i, _)| i as u32)
            .expect("No DEVICE_LOCAL memory type found - unsupported hardware");

        // === HOST: HOST_VISIBLE, prefer HOST_COHERENT, avoid HOST_CACHED ===
        // Write-combined memory is ideal for staging (sequential CPU writes).
        let host = memory_types
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, mt)| has_flags(mt, vk::MemoryPropertyFlags::HOST_VISIBLE))
            // Prefer: HOST_COHERENT, not HOST_CACHED, not DEVICE_LOCAL, larger heap
            .max_by_key(|(_, mt)| {
                let coherent = mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                let not_cached = !mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_CACHED);
                let not_device_local = !mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
                (coherent, not_cached, not_device_local, heap_size(mt))
            })
            .map(|(i, _)| i as u32)
            .expect("No HOST_VISIBLE memory type found - unsupported hardware");

        // === DYNAMIC: HOST_VISIBLE + HOST_CACHED, prefer DEVICE_LOCAL ===
        // Fast CPU reads required; DEVICE_LOCAL benefits integrated GPUs.
        let dynamic = memory_types
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, mt)| {
                has_flags(
                    mt,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
                )
            })
            // Prefer: DEVICE_LOCAL (for integrated GPUs), HOST_COHERENT, larger heap
            .max_by_key(|(_, mt)| {
                let device_local = mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
                let coherent = mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                (device_local, coherent, heap_size(mt))
            })
            .map(|(i, _)| i as u32)
            .expect("No HOST_VISIBLE + HOST_CACHED memory type found - unsupported hardware");

        // === UPLOAD: DEVICE_LOCAL, prefer HOST_VISIBLE ===
        // If HOST_VISIBLE is available, we can write directly without staging.
        let (upload, upload_host_visible) = {
            // First, try to find DEVICE_LOCAL + HOST_VISIBLE
            let device_local_host_visible = memory_types
                .iter()
                .enumerate()
                .rev()
                .filter(|(_, mt)| {
                    has_flags(
                        mt,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL
                            | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    )
                })
                // Prefer larger heaps (avoid 256MB BAR if full VRAM is available via ReBAR)
                .max_by_key(|(_, mt)| heap_size(mt));

            if let Some((idx, mt)) = device_local_host_visible {
                // On AMD APU, the 256MB DEVICE_LOCAL heap is too small for general use.
                // Fall back to HOST_VISIBLE-only memory if the heap is suspiciously small
                // and we're on an integrated GPU.
                let is_small_heap = heap_size(mt) <= 256 * 1024 * 1024; // 256MB threshold
                if is_small_heap {
                    if device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                        // AMD APU pattern: use the HOST_VISIBLE memory instead
                        // (it's system RAM which is actually what the GPU uses)
                        (host, true)
                    } else {
                        // discrete GPU without resizable bar: use private
                        (private, false)
                    }
                } else {
                    (idx as u32, true)
                }
            } else {
                if device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                    (host, false)
                } else {
                    // No DEVICE_LOCAL + HOST_VISIBLE: discrete GPU without ReBAR
                    // Use pure DEVICE_LOCAL and require staging
                    (private, false)
                }
            }
        };

        // === UNIFORM: DEVICE_LOCAL + HOST_VISIBLE, accepts 256MB BAR ===
        // For uniform buffers that need direct CPU writes. Unlike upload, we accept
        // even small heaps (256MB BAR) since uniform data is typically small.
        // Returns u32::MAX if no such memory type exists.
        let uniform = memory_types
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, mt)| {
                has_flags(
                    mt,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )
            })
            // Prefer larger heaps when available
            .max_by_key(|(_, mt)| heap_size(mt))
            .map(|(i, _)| i as u32)
            .unwrap_or(u32::MAX);

        Self {
            private,
            host,
            dynamic,
            upload,
            upload_host_visible,
            uniform,
        }
    }
}

/// Trait for Vulkan device features.
///
/// This trait abstracts over the various feature structures in Vulkan, allowing
/// uniform handling of feature queries and enabling.
///
/// # Safety
///
/// Implementors must correctly specify the associated extension and structure type.
pub unsafe trait Feature {
    /// The device extension that provides this feature.
    const REQUIRED_DEVICE_EXT: &'static CStr;
    /// Whether this feature has been promoted to Vulkan core.
    const PROMOTION_STATUS: PromotionStatus = PromotionStatus::None;
    /// The structure type for this feature, or `None` for base features.
    const STRUCTURE_TYPE: Option<vk::StructureType>;

    fn get_from_chain<'a>(
        chain: &'a NextChainMap<vk::PhysicalDeviceFeatures2<'static>>,
    ) -> Option<&'a Self>;
    fn get_mut_or_insert_from_chain<'a>(
        chain: &'a mut NextChainMap<vk::PhysicalDeviceFeatures2<'static>>,
        insert: impl FnOnce(&mut vk::PhysicalDeviceFeatures2<'static>) -> Self,
    ) -> &'a mut Self;
}

/// Utility for setting up physical device features
pub struct PhysicalDeviceFeatureMap {
    physical_device: PhysicalDevice,
    available_features: NextChainMap<vk::PhysicalDeviceFeatures2<'static>>,
    enabled_features: NextChainMap<vk::PhysicalDeviceFeatures2<'static>>,
}
impl PhysicalDeviceFeatureMap {
    /// Creates a new feature map for the given physical device.
    ///
    /// Queries the device for its supported features.
    pub fn new(physical_device: PhysicalDevice) -> Self {
        let mut this = Self {
            physical_device: physical_device.clone(),
            available_features: NextChainMap::default(),
            enabled_features: NextChainMap::default(),
        };
        unsafe {
            physical_device.instance().get_physical_device_features2(
                physical_device.vk_handle(),
                &mut this.available_features.head,
            );
        }
        this
    }

    /// Checks if a feature is available on the device.
    pub fn available_feature<T: Feature + Default + 'static>(&self) -> Option<&T> {
        <T as Feature>::get_from_chain(&self.available_features)
    }

    /// Checks if a feature has been enabled.
    pub fn enabled_feature<T: Feature + Default + 'static>(&self) -> Option<&T> {
        <T as Feature>::get_from_chain(&self.enabled_features)
    }

    /// Enables a specific feature flag.
    ///
    /// The `selector` closure should return a mutable reference to the specific
    /// `VkBool32` field to enable within the feature structure.
    ///
    /// # Errors
    ///
    /// Returns [`MissingFeatureError`] if the feature is not available.
    pub fn enable_feature<T: Feature + Default + 'static>(
        &mut self,
        mut selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Result<(), MissingFeatureError> {
        let feature = T::get_mut_or_insert_from_chain(&mut self.available_features, |base| {
            let mut feature = T::default();
            base.p_next = &mut feature as *mut T as *mut std::ffi::c_void;
            unsafe {
                self.physical_device
                    .instance()
                    .get_physical_device_features2(self.physical_device.vk_handle(), base);
            }
            base.p_next = std::ptr::null_mut();
            feature
        });
        let feature_available: vk::Bool32 = *selector(feature);
        if feature_available == vk::FALSE {
            // feature unavailable
            return Err(MissingFeatureError::Feature {
                feature: "",
                feature_set: "",
            });
        }

        let enabled_features =
            T::get_mut_or_insert_from_chain(&mut self.enabled_features, |_| T::default());
        let feature_to_enable = selector(enabled_features);
        *feature_to_enable = vk::TRUE;
        Ok(())
    }

    /// Finishes building and returns the enabled features chain.
    ///
    /// The returned chain can be used during device creation.
    pub fn finish(mut self) -> NextChainMap<vk::PhysicalDeviceFeatures2<'static>> {
        self.enabled_features.make_chain();
        self.enabled_features
    }
}

macro_rules! impl_feature_for_ext {
    ($feature:ty, $ext:ty) => {
        unsafe impl Feature for $feature {
            const REQUIRED_DEVICE_EXT: &'static CStr = <$ext>::NAME;
            const PROMOTION_STATUS: PromotionStatus = <$ext>::PROMOTION_STATUS;
            const STRUCTURE_TYPE: Option<vk::StructureType> =
                Some(<$feature as TaggedStructure>::STRUCTURE_TYPE);
            fn get_from_chain<'a>(
                chain: &'a NextChainMap<vk::PhysicalDeviceFeatures2<'static>>,
            ) -> Option<&'a Self> {
                chain.get::<Self>()
            }
            fn get_mut_or_insert_from_chain<'a>(
                chain: &'a mut NextChainMap<vk::PhysicalDeviceFeatures2<'static>>,
                insert: impl FnOnce(&mut vk::PhysicalDeviceFeatures2<'static>) -> Self,
            ) -> &'a mut Self {
                chain.get_mut_or_insert_with::<Self>(insert)
            }
        }
    };
}
unsafe impl Feature for vk::PhysicalDeviceFeatures {
    const REQUIRED_DEVICE_EXT: &'static CStr = c"Vulkan Base";

    const PROMOTION_STATUS: PromotionStatus =
        PromotionStatus::PromotedToCore(vk::make_api_version(0, 1, 0, 0));
    const STRUCTURE_TYPE: Option<vk::StructureType> = None;
    fn get_from_chain<'a>(
        chain: &'a NextChainMap<vk::PhysicalDeviceFeatures2<'static>>,
    ) -> Option<&'a Self> {
        Some(&chain.head.features)
    }

    fn get_mut_or_insert_from_chain<'a>(
        chain: &'a mut NextChainMap<vk::PhysicalDeviceFeatures2<'static>>,
        _insert: impl FnOnce(&mut vk::PhysicalDeviceFeatures2<'static>) -> Self,
    ) -> &'a mut Self {
        &mut chain.head.features
    }
}

impl_feature_for_ext!(
    vk::PhysicalDeviceSynchronization2FeaturesKHR<'static>,
    khr::synchronization2::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceTimelineSemaphoreFeatures<'static>,
    khr::timeline_semaphore::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceDynamicRenderingFeatures<'static>,
    khr::dynamic_rendering::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'static>,
    khr::ray_tracing_pipeline::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT<'static>,
    khr::ray_tracing_pipeline::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
    khr::acceleration_structure::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceBufferDeviceAddressFeatures<'static>,
    khr::buffer_device_address::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceRayTracingMotionBlurFeaturesNV<'static>,
    nv::ray_tracing_motion_blur::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDevice8BitStorageFeatures<'static>,
    khr::_8bit_storage::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDevice16BitStorageFeatures<'static>,
    khr::_16bit_storage::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceShaderFloat16Int8Features<'static>,
    khr::shader_float16_int8::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceScalarBlockLayoutFeatures<'static>,
    khr::shader_float16_int8::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT<'static>,
    ext::extended_dynamic_state::Meta
);

impl_feature_for_ext!(
    vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT<'static>,
    ext::extended_dynamic_state2::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT<'static>,
    ext::extended_dynamic_state3::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'static>,
    ext::swapchain_maintenance1::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceDescriptorIndexingFeatures<'static>,
    ext::descriptor_indexing::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT<'static>,
    ext::mutable_descriptor_type::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceDescriptorPoolOverallocationFeaturesNV<'static>,
    nv::descriptor_pool_overallocation::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceShaderDrawParameterFeatures<'static>,
    khr::shader_draw_parameters::Meta
);

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;

    // Memory property flag helper
    fn flags(list: &[vk::MemoryPropertyFlags]) -> vk::MemoryPropertyFlags {
        let mut result = vk::MemoryPropertyFlags::empty();
        for f in list {
            result |= *f;
        }
        result
    }

    const DL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const DLC_AMD: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD;
    const DLU_AMD: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD;
    const HV: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_VISIBLE;
    const HC: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_COHERENT;
    const HCA: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_CACHED;

    fn mem_type(heap_index: u32, flags: vk::MemoryPropertyFlags) -> vk::MemoryType {
        vk::MemoryType {
            property_flags: flags,
            heap_index,
        }
    }

    fn mem_heap(size: u64, device_local: bool) -> vk::MemoryHeap {
        vk::MemoryHeap {
            size,
            flags: if device_local {
                vk::MemoryHeapFlags::DEVICE_LOCAL
            } else {
                vk::MemoryHeapFlags::empty()
            },
        }
    }

    /// Intel integrated GPU: Single unified memory heap.
    /// All memory is DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT + HOST_CACHED.
    #[test]
    fn test_intel_integrated() {
        // Intel UHD Graphics 630 pattern:
        // Heap 0: ~25GB (system RAM, DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
        let heaps = [mem_heap(25 * GB, true)];
        let types = [mem_type(0, flags(&[DL, HV, HC, HCA]))];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        // All strategies should use type 0 (the only type)
        assert_eq!(map.private, 0, "private should use type 0");
        assert_eq!(map.host, 0, "host should use type 0");
        assert_eq!(map.dynamic, 0, "dynamic should use type 0");
        assert_eq!(map.upload, 0, "upload should use type 0");
        assert!(
            map.upload_host_visible,
            "integrated GPU should get host visible upload buffers"
        );
        assert_eq!(map.uniform, 0, "uniform should use type 0");
    }

    /// Apple Silicon (M1/M2/M3): Unified memory architecture.
    /// Similar to Intel integrated but with MoltenVK translation layer.
    #[test]
    fn test_apple_silicon() {
        // Apple M1 pattern (via MoltenVK):
        // Heap 0: ~16GB unified memory (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
        let heaps = [mem_heap(16 * GB, true)];
        let types = [mem_type(0, flags(&[DL, HV, HC, HCA]))];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private, 0);
        assert_eq!(map.host, 0);
        assert_eq!(map.dynamic, 0);
        assert_eq!(map.upload, 0);
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 0);
    }

    /// NVIDIA discrete GPU without resizable BAR.
    /// Separate VRAM and system RAM heaps, no host-visible VRAM.
    #[test]
    fn test_nvidia_discrete() {
        // NVIDIA RTX 3080 pattern (no ReBAR):
        // Heap 0: 10GB VRAM (DEVICE_LOCAL)
        // Heap 1: 32GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0) - GPU only
        // Type 1: HOST_VISIBLE | HOST_COHERENT (heap 1) - staging
        // Type 2: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1) - readback
        let heaps = [mem_heap(10 * GB, true), mem_heap(32 * GB, false)];
        let types = [
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private, 0, "private should use DEVICE_LOCAL type");
        assert_eq!(
            map.host, 1,
            "host should use HOST_VISIBLE without HOST_CACHED"
        );
        assert_eq!(map.dynamic, 2, "dynamic should use HOST_CACHED type");
        assert_eq!(
            map.upload, 0,
            "upload should use DEVICE_LOCAL (with staging)"
        );
        assert!(
            !map.upload_host_visible,
            "discrete GPU without ReBAR requires staging"
        );
        assert_eq!(
            map.uniform,
            u32::MAX,
            "no DEVICE_LOCAL + HOST_VISIBLE = uniform unavailable"
        );
    }

    /// NVIDIA discrete GPU with resizable BAR (SAM).
    /// Entire VRAM is host-visible.
    #[test]
    fn test_nvidia_rebar() {
        // NVIDIA RTX 3080 with ReBAR enabled:
        // Heap 0: 10GB VRAM (DEVICE_LOCAL)
        // Heap 1: 32GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 0) - ReBAR!
        // Type 2: HOST_VISIBLE | HOST_COHERENT (heap 1)
        // Type 3: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        let heaps = [mem_heap(10 * GB, true), mem_heap(32 * GB, false)];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(
            map.private, 0,
            "private should prefer non-HOST_VISIBLE DEVICE_LOCAL"
        );
        assert_eq!(map.host, 2, "host should use system RAM for staging");
        assert_eq!(map.dynamic, 3, "dynamic should use HOST_CACHED type");
        assert_eq!(
            map.upload, 1,
            "upload should use ReBAR type (DEVICE_LOCAL + HOST_VISIBLE)"
        );
        assert!(map.upload_host_visible, "ReBAR allows direct upload");
        assert_eq!(map.uniform, 1, "uniform should use ReBAR type");
    }

    /// AMD discrete GPU with 256MB BAR (pre-SAM).
    /// Small host-visible window into VRAM.
    #[test]
    fn test_amd_256mb_bar() {
        // AMD RX 6800 XT without SAM:
        // Heap 0: 16GB VRAM (DEVICE_LOCAL)
        // Heap 1: 256MB BAR (DEVICE_LOCAL) - small host-visible window
        // Heap 2: 16GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 1) - 256MB BAR
        // Type 2: HOST_VISIBLE | HOST_COHERENT (heap 2)
        // Type 3: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 2)
        let heaps = [
            mem_heap(16 * GB, true),
            mem_heap(256 * MB, true),
            mem_heap(16 * GB, false),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[HV, HC])),
            mem_type(2, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private, 0, "private should use main VRAM heap");
        assert_eq!(map.host, 2, "host should use system RAM");
        assert_eq!(map.dynamic, 3, "dynamic should use HOST_CACHED");
        // For discrete GPU, 256MB BAR is still usable for uploads
        assert_eq!(
            map.upload, 0,
            "upload should use private, requiring staging"
        );
        assert!(
            !map.upload_host_visible,
            "256MB BAR does not allow direct upload"
        );
        assert_eq!(map.uniform, 1, "uniform should use 256MB BAR");
    }

    /// AMD discrete GPU with SAM (Smart Access Memory) / Resizable BAR.
    /// Full VRAM is host-visible.
    #[test]
    fn test_amd_sam() {
        // AMD RX 6800 XT with SAM enabled:
        // Heap 0: 16GB VRAM (DEVICE_LOCAL)
        // Heap 1: 16GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 0) - full ReBAR
        // Type 2: HOST_VISIBLE | HOST_COHERENT (heap 1)
        // Type 3: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        let heaps = [mem_heap(16 * GB, true), mem_heap(16 * GB, false)];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private, 0, "private should prefer pure DEVICE_LOCAL");
        assert_eq!(map.host, 2, "host should use system RAM");
        assert_eq!(map.dynamic, 3, "dynamic should use HOST_CACHED");
        assert_eq!(map.upload, 1, "upload should use SAM/ReBAR type");
        assert!(map.upload_host_visible, "SAM allows direct upload");
        assert_eq!(map.uniform, 1, "uniform should use SAM/ReBAR type");
    }

    /// AMD APU (e.g., Steam Deck, Ryzen 7000 series APU).
    /// 256MB "virtual" DEVICE_LOCAL heap, rest is HOST_VISIBLE system RAM.
    #[test]
    fn test_amd_apu() {
        // AMD Ryzen 7 6800U (Rembrandt APU) pattern:
        // Heap 0: 256MB "carve-out" (DEVICE_LOCAL) - virtual VRAM
        // Heap 1: 14GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0) - too small for general use!
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 0)
        // Type 2: HOST_VISIBLE | HOST_COHERENT (heap 1) - main memory
        // Type 3: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        let heaps = [
            mem_heap(16 * GB, true),
            mem_heap(8 * GB, false),
            mem_heap(256 * MB, true),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(0, DL),
            mem_type(0, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(1, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(1, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
        ];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        // Private can still use the small DEVICE_LOCAL heap for scratch data
        assert_eq!(map.private, 0, "private should use DEVICE_LOCAL");
        assert_eq!(map.host, 4, "host should use system RAM (non-cached)");
        assert_eq!(map.dynamic, 5, "dynamic should use HOST_CACHED");
        // Upload should fall back to system RAM because 256MB is too small
        assert_eq!(
            map.upload, 4,
            "upload should fall back to HOST_VISIBLE (system RAM)"
        );
        assert!(
            map.upload_host_visible,
            "APU with system RAM doesn't need staging"
        );
        assert_eq!(
            map.uniform, 12,
            "uniform should use 256MB BAR (DL+HV on heap 2)"
        );
    }

    /// Steam Deck (AMD Van Gogh APU).
    /// Similar to AMD APU but with specific memory configuration.
    #[test]
    fn test_steam_deck() {
        // Steam Deck (AMD Van Gogh) pattern:
        // Heap 0: 256MB DEVICE_LOCAL
        // Heap 1: ~14GB system RAM
        // Same type layout as other AMD APUs
        let heaps = [mem_heap(4 * MB, false), mem_heap(8 * GB, true)];
        let types = [
            mem_type(0, flags(&[HV, HC])),
            mem_type(0, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(0, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
            mem_type(1, DL),
            mem_type(1, DL),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
        ];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private, 5);
        assert_eq!(map.host, 0);
        assert_eq!(map.dynamic, 1);
        assert_eq!(map.upload, 7);
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 7, "uniform should use DL+HV type on heap 1");
    }

    /// Older Intel discrete GPU (Arc series).
    /// Discrete with separate heaps but potentially different layout.
    #[test]
    fn test_intel_arc() {
        // Intel Arc A770 pattern:
        // Heap 0: 16GB VRAM (DEVICE_LOCAL)
        // Heap 1: 32GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 0) - ReBAR
        // Type 2: HOST_VISIBLE | HOST_COHERENT (heap 1)
        // Type 3: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        let heaps = [mem_heap(16 * GB, true), mem_heap(32 * GB, false)];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private, 0, "private should use pure DEVICE_LOCAL");
        assert_eq!(map.host, 2, "host should use system RAM");
        assert_eq!(map.dynamic, 3, "dynamic should use HOST_CACHED");
        assert_eq!(map.upload, 1, "upload should use ReBAR");
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 1, "uniform should use ReBAR type");
    }

    /// Qualcomm Adreno (mobile GPU in Android/Windows on ARM).
    /// Unified memory similar to other integrated GPUs.
    #[test]
    fn test_qualcomm_adreno() {
        // Qualcomm Adreno 740 pattern:
        // Single heap, unified memory
        let heaps = [mem_heap(8 * GB, true)];
        let types = [mem_type(0, DL), mem_type(0, flags(&[DL, HV, HC, HCA]))];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        // Should prefer non-HOST_VISIBLE for private if available
        assert_eq!(map.private, 0, "private should use pure DEVICE_LOCAL");
        assert_eq!(map.host, 1, "host should use HOST_VISIBLE type");
        assert_eq!(map.dynamic, 1, "dynamic should use HOST_CACHED type");
        assert_eq!(map.upload, 1, "upload should use HOST_VISIBLE DEVICE_LOCAL");
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 1, "uniform should use DL+HV type");
    }

    /// Edge case: Minimal configuration with only essential memory types.
    #[test]
    fn test_minimal_discrete() {
        // Minimal discrete GPU configuration:
        // Heap 0: VRAM
        // Heap 1: System RAM
        // Only two memory types
        let heaps = [mem_heap(4 * GB, true), mem_heap(8 * GB, false)];
        let types = [mem_type(0, DL), mem_type(1, flags(&[HV, HC, HCA]))];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private, 0);
        assert_eq!(map.host, 1);
        assert_eq!(map.dynamic, 1);
        assert_eq!(map.upload, 0, "upload must use DEVICE_LOCAL");
        assert!(
            !map.upload_host_visible,
            "no HOST_VISIBLE DEVICE_LOCAL = staging required"
        );
        assert_eq!(
            map.uniform,
            u32::MAX,
            "no DEVICE_LOCAL + HOST_VISIBLE = uniform unavailable"
        );
    }

    /// Test that larger heaps are preferred when multiple options exist.
    #[test]
    fn test_heap_size_preference() {
        // Two DEVICE_LOCAL heaps of different sizes
        let heaps = [
            mem_heap(2 * GB, true),   // Smaller VRAM
            mem_heap(8 * GB, true),   // Larger VRAM
            mem_heap(16 * GB, false), // System RAM
        ];
        let types = [
            mem_type(0, DL),
            mem_type(1, DL),
            mem_type(2, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        // Should prefer the larger VRAM heap
        assert_eq!(map.private, 1, "private should use larger VRAM heap");
        assert_eq!(
            map.uniform,
            u32::MAX,
            "no DEVICE_LOCAL + HOST_VISIBLE = uniform unavailable"
        );
    }

    /// Test ReBAR preference: larger heap should be chosen over 256MB BAR.
    #[test]
    fn test_rebar_over_256mb_bar() {
        // System with both 256MB BAR and full ReBAR
        // (unusual but possible during driver transitions)
        let heaps = [
            mem_heap(8 * GB, true),   // Main VRAM
            mem_heap(256 * MB, true), // Old 256MB BAR
            mem_heap(16 * GB, false), // System RAM
        ];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])), // ReBAR on main VRAM
            mem_type(1, flags(&[DL, HV, HC])), // 256MB BAR
            mem_type(2, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMap::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        // Upload should prefer the larger ReBAR heap over the 256MB BAR
        assert_eq!(map.upload, 1, "upload should prefer larger ReBAR heap");
        assert!(map.upload_host_visible);
        assert_eq!(
            map.uniform, 1,
            "uniform should prefer larger ReBAR heap over 256MB BAR"
        );
    }
}
