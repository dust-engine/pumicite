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
    Device, MissingFeatureError,
    utils::{AsVkHandle, NextChainMap, Version, VkTaggedObject},
};

use super::Instance;
use ash::{
    VkResult, ext, khr, nv,
    vk::{self, ExtensionMeta, PromotionStatus, TaggedStructure},
};
use core::ffi::c_void;
use smallvec::{SmallVec, smallvec};
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
    properties: RwLock<BTreeMap<vk::StructureType, Box<VkTaggedObject>>>,
}
unsafe impl Send for PhysicalDeviceProperties {}
unsafe impl Sync for PhysicalDeviceProperties {}
impl PhysicalDeviceProperties {
    fn new(instance: Instance, pdevice: vk::PhysicalDevice) -> Self {
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let pdevice_properties = unsafe { instance.get_physical_device_properties(pdevice) };

        Self {
            instance,
            pdevice,
            properties: Default::default(),
            memory_properties,
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
/// - **Intel integrated**: Single DEVICE_LOCAL heap (system RAM), with and without HOST_VISIBLE
/// - **NVIDIA/AMD discrete**: Separate VRAM (DEVICE_LOCAL) and system RAM (HOST_VISIBLE)
/// - **Discrete without resizable BAR**: Additional 256MB DEVICE_LOCAL + HOST_VISIBLE heap
/// - **Resizable BAR (SAM)**: Entire VRAM accessible as DEVICE_LOCAL + HOST_VISIBLE
/// - **AMD APU (radv)**: 2/3 of the carve-out plus system RAM as one DEVICE_LOCAL +
///   HOST_VISIBLE heap, the rest as HOST_VISIBLE only
/// - **Integrated with a small carve-out**: DEVICE_LOCAL + HOST_VISIBLE heap of 256MB or
///   less, the rest is HOST_VISIBLE only
///
/// Memory types with identical flags can still accept different resources: Intel Xe2+
/// lists a compressed DEVICE_LOCAL type first that only images can use. So `private` and
/// `upload` memory are ranked lists of candidates, and the `private_*` and `upload_*`
/// methods return the first candidate the resource's `memoryTypeBits` allow.
pub struct MemoryTypeMap<'a> {
    device: &'a Device,
    inner: &'a MemoryTypeMapInner,
}

impl Deref for MemoryTypeMap<'_> {
    type Target = MemoryTypeMapInner;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

/// The memory types of a [`MemoryTypeMap`], stored by its device.
#[derive(Debug, Clone)]
pub struct MemoryTypeMapInner {
    /// Every DEVICE_LOCAL type, non-HOST_VISIBLE first, then larger heaps, then index order.
    private: SmallVec<[u32; 2]>,

    /// Staging memory for CPU-to-GPU transfers.
    ///
    /// Selection: HOST_VISIBLE + HOST_COHERENT required, prefers system RAM over
    /// DEVICE_LOCAL (keeps staging out of a 256MB BAR heap), then avoids HOST_CACHED
    /// (write-combined memory is fine for sequential writes).
    pub staging: u32,

    /// CPU-readable memory for GPU-to-CPU readback.
    ///
    /// Selection: HOST_VISIBLE + HOST_COHERENT required, prefers HOST_CACHED (fast CPU
    /// reads), then DEVICE_LOCAL (benefits integrated GPUs).
    pub dynamic: u32,

    /// The best DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT type, or the `private`
    /// candidates on discrete GPUs that need staging.
    upload: SmallVec<[u32; 2]>,

    /// Memory for uniform buffers: guaranteed to be device-local, host-visible and
    /// host-coherent.
    ///
    /// May use the 256MB BAR on discrete GPUs without resizable BAR.
    /// Set to `u32::MAX` on GPUs that have no device-local host-visible memory type at all.
    pub uniform: u32,

    /// Whether the device can use [`dynamic`](Self::dynamic) memory as its own, so data
    /// there needs no staging copy into [`private_buffer`](MemoryTypeMap::private_buffer) memory.
    ///
    /// True on integrated GPUs, whose system RAM is the GPU's memory even when it isn't
    /// marked DEVICE_LOCAL (AMD APUs), and on other GPUs whose `dynamic` memory type is
    /// DEVICE_LOCAL (e.g. GPUs with a cache-coherent link to the host).
    pub dynamic_device_local: bool,

    /// Whether [`upload_buffer`](MemoryTypeMap::upload_buffer) and
    /// [`upload_image`](MemoryTypeMap::upload_image) memory is host visible.
    ///
    /// True on discrete GPUs without resizable BAR. These GPUs do not have a large
    /// DEVICE_LOCAL, HOST_VISIBLE pool.
    pub upload_host_visible: bool,
}

impl<'a> MemoryTypeMap<'a> {
    pub(crate) fn new(device: &'a Device, inner: &'a MemoryTypeMapInner) -> Self {
        Self { device, inner }
    }

    /// GPU-exclusive memory for scratch buffers and GPU-generated data.
    ///
    /// Selection: DEVICE_LOCAL required, prefers non-HOST_VISIBLE (pure VRAM is faster on
    /// discrete GPUs when not accessed via BAR). Fails with `ERROR_FEATURE_NOT_PRESENT` if
    /// a buffer created with `info` accepts no DEVICE_LOCAL memory type.
    pub fn private_buffer(&self, info: &vk::BufferCreateInfo) -> VkResult<u32> {
        first_allowed(&self.private, self.buffer_memory_type_bits(info))
    }

    /// GPU-exclusive memory for render targets and GPU-generated images.
    ///
    /// Selection as in [`private_buffer`](Self::private_buffer). Images that can be
    /// compressed get the compressed memory type on Intel Xe2+.
    pub fn private_image(&self, info: &vk::ImageCreateInfo) -> VkResult<u32> {
        first_allowed(&self.private, self.image_memory_type_bits(info))
    }

    /// Upload memory for CPU-written, GPU-read buffers.
    ///
    /// Selection: DEVICE_LOCAL required, prefers HOST_VISIBLE + HOST_COHERENT (avoids
    /// staging).
    /// Check [`upload_host_visible`](MemoryTypeMapInner::upload_host_visible) to determine
    /// if a staging copy is needed.
    pub fn upload_buffer(&self, info: &vk::BufferCreateInfo) -> VkResult<u32> {
        first_allowed(&self.upload, self.buffer_memory_type_bits(info))
    }

    /// Upload memory for CPU-written, GPU-read images.
    ///
    /// Selection as in [`upload_buffer`](Self::upload_buffer).
    pub fn upload_image(&self, info: &vk::ImageCreateInfo) -> VkResult<u32> {
        first_allowed(&self.upload, self.image_memory_type_bits(info))
    }

    fn buffer_memory_type_bits(&self, info: &vk::BufferCreateInfo) -> u32 {
        let mut requirements = vk::MemoryRequirements2::default();
        unsafe {
            self.device.get_device_buffer_memory_requirements(
                &vk::DeviceBufferMemoryRequirements::default().create_info(info),
                &mut requirements,
            );
        }
        requirements.memory_requirements.memory_type_bits
    }

    fn image_memory_type_bits(&self, info: &vk::ImageCreateInfo) -> u32 {
        let mut requirements = vk::MemoryRequirements2::default();
        unsafe {
            self.device.get_device_image_memory_requirements(
                &vk::DeviceImageMemoryRequirements::default().create_info(info),
                &mut requirements,
            );
        }
        requirements.memory_requirements.memory_type_bits
    }
}

/// The first of `candidates` that `memory_type_bits` allows.
fn first_allowed(candidates: &[u32], memory_type_bits: u32) -> VkResult<u32> {
    candidates
        .iter()
        .copied()
        .find(|&i| memory_type_bits & (1 << i) != 0)
        .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)
}

impl MemoryTypeMapInner {
    /// Computes the memory types for a physical device.
    ///
    /// # Panics
    ///
    /// Panics if required memory types cannot be found (exotic/unsupported hardware).
    pub(crate) fn new(
        memory_types: &[vk::MemoryType],
        memory_heaps: &[vk::MemoryHeap],
        device_type: vk::PhysicalDeviceType,
    ) -> Self {
        // Memory types that general-purpose allocations can't use: PROTECTED memory needs
        // protected submissions, LAZILY_ALLOCATED memory only backs transient attachments, and
        // DEVICE_COHERENT_AMD memory needs a device feature pumicite doesn't enable (VMA
        // excludes it too). Buffers and images are pinned to the selected types, so picking
        // one of these would make every allocation fail.
        let unusable = vk::MemoryPropertyFlags::PROTECTED
            | vk::MemoryPropertyFlags::LAZILY_ALLOCATED
            | vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD;

        // Helper to check if a memory type is usable and has all required flags
        let has_flags = |mt: &vk::MemoryType, required: vk::MemoryPropertyFlags| {
            mt.property_flags.contains(required) && !mt.property_flags.intersects(unusable)
        };

        // Helper to get heap size for a memory type
        let heap_size = |mt: &vk::MemoryType| memory_heaps[mt.heap_index as usize].size;

        // === PRIVATE: DEVICE_LOCAL, prefer non-HOST_VISIBLE ===
        // On discrete GPUs, pure VRAM without BAR access is typically faster.
        let mut private: SmallVec<[u32; 2]> = (0..memory_types.len() as u32)
            .filter(|&i| {
                has_flags(
                    &memory_types[i as usize],
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
            })
            .collect();
        // Prefer non-HOST_VISIBLE (pure VRAM), then larger heaps. The sort is stable, so
        // equally ranked types stay in index order, where drivers put the one they prefer.
        private.sort_by_key(|&i| {
            let mt = &memory_types[i as usize];
            let not_host_visible = !mt
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
            std::cmp::Reverse((not_host_visible, heap_size(mt)))
        });
        assert!(
            !private.is_empty(),
            "No DEVICE_LOCAL memory type found - unsupported hardware"
        );

        // === HOST: HOST_VISIBLE + HOST_COHERENT, prefer system RAM, avoid HOST_CACHED ===
        // Write-combined memory is ideal for staging (sequential CPU writes), but not at
        // the cost of VRAM: Intel discrete GPUs expose only HOST_CACHED system RAM, and
        // without resizable BAR their only uncached type is the 256MB BAR heap.
        let staging = memory_types
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, mt)| has_flags(mt, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT))
            // Prefer: not DEVICE_LOCAL, not HOST_CACHED, larger heap
            .max_by_key(|(_, mt)| {
                let not_cached = !mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_CACHED);
                let not_device_local = !mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
                (not_device_local, not_cached, heap_size(mt))
            })
            .map(|(i, _)| i as u32)
            .expect("No HOST_VISIBLE + HOST_COHERENT memory type found - unsupported hardware");

        // === DYNAMIC: HOST_VISIBLE + HOST_COHERENT, prefer HOST_CACHED, DEVICE_LOCAL ===
        // HOST_CACHED makes CPU reads fast, but some GPUs only cache non-coherent maps
        // (Intel without LLC, Tegra), so uncached memory is the fallback.
        // DEVICE_LOCAL benefits integrated GPUs.
        let dynamic = memory_types
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, mt)| has_flags(mt, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT))
            // Prefer: HOST_CACHED, DEVICE_LOCAL (for integrated GPUs), larger heap
            .max_by_key(|(_, mt)| {
                let cached = mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_CACHED);
                let device_local = mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
                (cached, device_local, heap_size(mt))
            })
            .map(|(i, _)| i as u32)
            .expect("No HOST_VISIBLE + HOST_COHERENT memory type found - unsupported hardware");
        let dynamic_device_local = device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            || has_flags(
                &memory_types[dynamic as usize],
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

        // === UPLOAD: DEVICE_LOCAL, prefer HOST_VISIBLE + HOST_COHERENT ===
        // If HOST_VISIBLE is available, we can write directly without staging.
        let (upload, upload_host_visible) = {
            // First, try to find DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT
            let device_local_host_visible = memory_types
                .iter()
                .enumerate()
                .rev()
                .filter(|(_, mt)| {
                    has_flags(mt, vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
                })
                // Prefer larger heaps (avoid 256MB BAR if full VRAM is available via ReBAR)
                .max_by_key(|(_, mt)| heap_size(mt));

            if let Some((idx, mt)) = device_local_host_visible {
                // An integrated GPU's carve-out can be too small for general use (radv
                // avoids this by reporting 2/3 of system memory as the carve-out heap).
                // Fall back to HOST_VISIBLE-only memory if the heap is suspiciously small
                // and we're on an integrated GPU.
                let is_small_heap = heap_size(mt) <= 256 * 1024 * 1024; // 256MB threshold
                if is_small_heap {
                    if device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                        // Small carve-out: use the HOST_VISIBLE memory instead
                        // (it's system RAM which is actually what the GPU uses)
                        (smallvec![staging], true)
                    } else {
                        // discrete GPU without resizable bar: use private
                        (private.clone(), false)
                    }
                } else {
                    (smallvec![idx as u32], true)
                }
            } else {
                if device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                    (smallvec![staging], false)
                } else {
                    // No DEVICE_LOCAL + HOST_VISIBLE: discrete GPU without ReBAR
                    // Use pure DEVICE_LOCAL and require staging
                    (private.clone(), false)
                }
            }
        };

        // === UNIFORM: DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT, accepts 256MB BAR ===
        // For uniform buffers that need direct CPU writes. Unlike upload, we accept
        // even small heaps (256MB BAR) since uniform data is typically small.
        // Returns u32::MAX if no such memory type exists.
        let uniform = memory_types
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, mt)| has_flags(mt, vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT))
            // Prefer larger heaps when available
            .max_by_key(|(_, mt)| heap_size(mt))
            .map(|(i, _)| i as u32)
            .unwrap_or(u32::MAX);

        Self {
            private,
            staging,
            dynamic,
            dynamic_device_local,
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
    ext::pipeline_library_group_handles::Meta
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
    vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR<'static>,
    khr::swapchain_maintenance1::Meta
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
impl_feature_for_ext!(
    vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
    ext::mesh_shader::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceRobustness2FeaturesKHR<'static>,
    ext::robustness2::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceHostQueryResetFeatures<'static>,
    ext::host_query_reset::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceShaderAtomicInt64Features<'static>,
    khr::shader_atomic_int64::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceMultiviewFeaturesKHR<'static>,
    khr::multiview::Meta
);
impl_feature_for_ext!(
    vk::PhysicalDeviceMaintenance4Features<'static>,
    khr::maintenance4::Meta
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
    const PROT: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::PROTECTED;
    const LAZY: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::LAZILY_ALLOCATED;

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

    // Intel (Mesa anv) layouts below follow anv_physical_device_init_heaps and the
    // i915/xe `*_physical_device_init_memory_types` tables. anv appends every type that
    // isn't PROTECTED or compressed a second time with identical flags, for descriptor
    // buffers only: normal buffers and images can't use those copies, so the lowest-index
    // tie must win. The system RAM heap is 75% of RAM (32GB machines here).

    /// Checks `map` against the memoryTypeBits anv reports (anv_buffer.c, anv_image.c).
    /// `buffer_bits` are the types buffers and images may use: the base types minus PROTECTED
    /// and compressed ones. `compressed_bits` are the types only compressible images may also
    /// use. Returns the types a buffer, a compressible image and an uploaded buffer get.
    fn anv_selection(
        map: &MemoryTypeMapInner,
        buffer_bits: u32,
        compressed_bits: u32,
    ) -> (u32, u32, u32) {
        let allowed = |i: u32| buffer_bits & (1 << i) != 0;
        assert!(
            allowed(map.staging),
            "staging must be a type buffers can use"
        );
        assert!(
            allowed(map.dynamic),
            "dynamic must be a type buffers can use"
        );
        if map.uniform != u32::MAX {
            assert!(
                allowed(map.uniform),
                "uniform must be a type buffers can use"
            );
        }
        (
            first_allowed(&map.private, buffer_bits).unwrap(),
            first_allowed(&map.private, buffer_bits | compressed_bits).unwrap(),
            first_allowed(&map.upload, buffer_bits).unwrap(),
        )
    }

    /// Intel integrated GPU with LLC on i915: Skylake through Raptor Lake (HD/UHD Graphics,
    /// Iris Xe). Meteor Lake and Arrow Lake on the Xe KMD expose the same flags.
    #[test]
    fn test_intel_llc_i915() {
        // Intel UHD Graphics 630 (Coffee Lake):
        // Heap 0: 24GB system RAM (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (write-combined)
        // Type 2: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
        // Types 3-5: descriptor buffer copies of types 0-2
        let heaps = [mem_heap(24 * GB, true)];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[DL, HV, HC, HCA])),
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[DL, HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(
            map.private[..],
            [0, 3, 1, 2, 4, 5],
            "private should use non-HOST_VISIBLE type"
        );
        assert_eq!(map.staging, 1, "staging should use write-combined type");
        assert_eq!(map.dynamic, 2, "dynamic should use HOST_CACHED type");
        assert!(map.dynamic_device_local);
        assert_eq!(map.upload[..], [1]);
        assert!(
            map.upload_host_visible,
            "integrated GPU should get host visible upload buffers"
        );
        assert_eq!(map.uniform, 1);
        assert_eq!(anv_selection(&map, 0b111, 0), (0, 0, 1));
    }

    /// Intel integrated GPU with LLC on the Xe KMD: Tiger Lake through Raptor Lake.
    /// Xe can't select the CPU caching mode at mmap time, so there's no write-combined type.
    #[test]
    fn test_intel_llc_xe() {
        // Heap 0: 24GB system RAM (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
        // Types 2-3: descriptor buffer copies of types 0-1
        let heaps = [mem_heap(24 * GB, true)];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC, HCA])),
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private[..], [0, 2, 1, 3]);
        assert_eq!(map.staging, 1, "staging falls back to the only HOST_VISIBLE type");
        assert_eq!(map.dynamic, 1);
        assert!(map.dynamic_device_local);
        assert_eq!(map.upload[..], [1]);
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 1);
        assert_eq!(anv_selection(&map, 0b11, 0), (0, 0, 1));
    }

    /// Intel integrated GPU without LLC on i915: Meteor Lake and Arrow Lake (Core Ultra),
    /// and, without the PROTECTED type, Apollo Lake and Gemini Lake. The only HOST_CACHED
    /// type is not HOST_COHERENT.
    #[test]
    fn test_intel_non_llc_i915() {
        // Heap 0: 24GB system RAM (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (write-combined)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED (not coherent)
        // Type 2: DEVICE_LOCAL | PROTECTED
        // Types 3-4: descriptor buffer copies of types 0-1
        let heaps = [mem_heap(24 * GB, true)];
        let types = [
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[DL, HV, HCA])),
            mem_type(0, flags(&[DL, PROT])),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[DL, HV, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(
            map.private[..],
            [0, 1, 3, 4],
            "private falls back to a HOST_VISIBLE type"
        );
        assert_eq!(map.staging, 0, "staging should use the coherent type");
        assert_eq!(map.dynamic, 0, "dynamic should use the coherent type");
        assert!(map.dynamic_device_local);
        assert_eq!(map.upload[..], [0]);
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 0);
        assert_eq!(anv_selection(&map, 0b011, 0), (0, 0, 0));
    }

    /// Mapped memory is never flushed or invalidated, so `dynamic` must be HOST_COHERENT
    /// even when the only HOST_CACHED type isn't (Intel without LLC on i915).
    #[test]
    fn test_intel_non_llc_i915_dynamic_coherent() {
        let heaps = [mem_heap(24 * GB, true)];
        let types = [
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[DL, HV, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert!(
            types[map.dynamic as usize].property_flags.contains(HC),
            "dynamic must be HOST_COHERENT"
        );
    }

    /// Intel Xe2+ integrated GPU on the Xe KMD: Lunar Lake and Panther Lake (Core Ultra
    /// 200V and Series 3).
    #[test]
    fn test_intel_xe2_integrated() {
        // Heap 0: 24GB system RAM (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL, compressed (images only)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (write-combined)
        // Type 2: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
        // Type 3: DEVICE_LOCAL | PROTECTED
        // Types 4-5: descriptor buffer copies of types 1-2
        let heaps = [mem_heap(24 * GB, true)];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[DL, HV, HC, HCA])),
            mem_type(0, flags(&[DL, PROT])),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[DL, HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        // anv's memoryTypeBits allow types 1-2 for buffers and images that can't be
        // compressed, so they get type 1; compressible images also allow type 0.
        assert_eq!(map.private[..], [0, 1, 2, 4, 5]);
        assert_eq!(map.staging, 1, "host should use write-combined type");
        assert_eq!(map.dynamic, 2, "dynamic should use HOST_CACHED type");
        assert!(map.dynamic_device_local);
        assert_eq!(map.upload[..], [1]);
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 1);
        assert_eq!(
            anv_selection(&map, 0b0110, 0b0001),
            (1, 0, 1),
            "buffers skip the compressed type, compressible images take it"
        );
    }

    /// Apple Silicon (M1/M2/M3): Unified memory architecture via MoltenVK.
    #[test]
    fn test_apple_silicon() {
        // Apple M1 pattern (via MoltenVK):
        // Heap 0: ~16GB unified memory (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
        let heaps = [mem_heap(16 * GB, true)];
        let types = [mem_type(0, flags(&[DL, HV, HC, HCA]))];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private[..], [0]);
        assert_eq!(map.staging, 0);
        assert_eq!(map.dynamic, 0);
        assert!(map.dynamic_device_local);
        assert_eq!(map.upload[..], [0]);
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

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0], "private should use DEVICE_LOCAL type");
        assert_eq!(
            map.staging, 1,
            "staging should use HOST_VISIBLE without HOST_CACHED"
        );
        assert_eq!(map.dynamic, 2, "dynamic should use HOST_CACHED type");
        assert!(!map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [0],
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

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(
            map.private[..],
            [0, 1],
            "private should prefer non-HOST_VISIBLE DEVICE_LOCAL"
        );
        assert_eq!(map.staging, 2, "staging should use system RAM for staging");
        assert_eq!(map.dynamic, 3, "dynamic should use HOST_CACHED type");
        assert!(!map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [1],
            "upload should use ReBAR type (DEVICE_LOCAL + HOST_VISIBLE)"
        );
        assert!(map.upload_host_visible, "ReBAR allows direct upload");
        assert_eq!(map.uniform, 1, "uniform should use ReBAR type");
    }

    // NVIDIA on Mesa NVK: layouts follow nvk_create_drm_physical_device. NVK exposes at most
    // three types and only one system RAM type, which is HOST_CACHED (the GPU snoops CPU
    // caches across PCIe). VRAM is CPU-mappable only from Maxwell on, through a separate
    // BAR heap unless the BAR covers all of VRAM. The system RAM heap is 75% of RAM (32GB
    // machines here).

    /// NVK on a Maxwell or newer discrete GPU without resizable BAR.
    #[test]
    fn test_nvk_small_bar() {
        // Heap 0: 10GB VRAM (DEVICE_LOCAL)
        // Heap 1: 256MB BAR (DEVICE_LOCAL), the CPU-visible window into VRAM
        // Heap 2: 24GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 1)
        // Type 2: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 2)
        let heaps = [
            mem_heap(10 * GB, true),
            mem_heap(256 * MB, true),
            mem_heap(24 * GB, false),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0, 1], "private prefers VRAM over the BAR");
        assert_eq!(
            map.staging, 2,
            "staging stays out of the BAR even though system RAM is HOST_CACHED"
        );
        assert_eq!(map.dynamic, 2);
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [0, 1], "256MB BAR is too small for upload");
        assert!(!map.upload_host_visible);
        assert_eq!(map.uniform, 1, "uniform should use the BAR");
    }

    /// NVK without resizable BAR, for images with HOST_TRANSFER usage (host image copy,
    /// Turing+). When the BAR is smaller than VRAM, NVK allows such images only the
    /// HOST_VISIBLE types, so even `private` memory lands in the BAR heap.
    #[test]
    fn test_nvk_small_bar_host_image_copy() {
        let heaps = [
            mem_heap(10 * GB, true),
            mem_heap(256 * MB, true),
            mem_heap(24 * GB, false),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        // nvk_get_image_memory_requirements drops type 0
        let host_transfer_bits = 0b110;
        assert_eq!(first_allowed(&map.private, host_transfer_bits), Ok(1));
        assert_eq!(first_allowed(&map.upload, host_transfer_bits), Ok(1));
        // Every other buffer and image allows all types
        assert_eq!(first_allowed(&map.private, 0b111), Ok(0));
        assert_eq!(first_allowed(&map.upload, 0b111), Ok(0));
    }

    /// NVK on a Maxwell or newer discrete GPU whose resized BAR is still smaller than VRAM,
    /// e.g. when the host bridge window can't fit all of it. NVK keeps the separate BAR
    /// heap, but at 8GB it's large enough for upload memory.
    #[test]
    fn test_nvk_partial_bar() {
        // Heap 0: 12GB VRAM (DEVICE_LOCAL)
        // Heap 1: 8GB BAR (DEVICE_LOCAL)
        // Heap 2: 24GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 1)
        // Type 2: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 2)
        let heaps = [
            mem_heap(12 * GB, true),
            mem_heap(8 * GB, true),
            mem_heap(24 * GB, false),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0, 1]);
        assert_eq!(map.staging, 2);
        assert_eq!(map.dynamic, 2);
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [1], "upload should use the large BAR");
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 1);
    }

    /// NVK on a Maxwell or newer discrete GPU whose BAR covers all of VRAM. NVK reports
    /// the same layout when it can't learn the BAR size (kernels without
    /// NOUVEAU_GETPARAM_VRAM_BAR_SIZE), even though only part of VRAM is mappable then.
    #[test]
    fn test_nvk_rebar() {
        // Heap 0: 10GB VRAM (DEVICE_LOCAL)
        // Heap 1: 24GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 0)
        // Type 2: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        let heaps = [mem_heap(10 * GB, true), mem_heap(24 * GB, false)];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0, 1]);
        assert_eq!(map.staging, 2, "staging should use system RAM");
        assert_eq!(map.dynamic, 2);
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [1], "upload should use host-visible VRAM");
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 1);
    }

    /// NVK on Kepler discrete GPUs, where VRAM can't be mapped at all. NVK doesn't
    /// support anything older.
    #[test]
    fn test_nvk_kepler() {
        // Heap 0: 3GB VRAM (DEVICE_LOCAL)
        // Heap 1: 24GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        let heaps = [mem_heap(3 * GB, true), mem_heap(24 * GB, false)];
        let types = [mem_type(0, DL), mem_type(1, flags(&[HV, HC, HCA]))];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0]);
        assert_eq!(map.staging, 1);
        assert_eq!(map.dynamic, 1);
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [0]);
        assert!(!map.upload_host_visible);
        assert_eq!(
            map.uniform,
            u32::MAX,
            "no DEVICE_LOCAL + HOST_VISIBLE = uniform unavailable"
        );
    }

    /// NVK on Tegra: system RAM only, reported as one DEVICE_LOCAL heap. The two types
    /// differ only in cached or coherent CPU maps. NVK only exposes Tegra with
    /// NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1.
    #[test]
    fn test_nvk_tegra() {
        // Heap 0: 24GB system RAM (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED (not coherent)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
        let heaps = [mem_heap(24 * GB, true)];
        let types = [
            mem_type(0, flags(&[DL, HV, HCA])),
            mem_type(0, flags(&[DL, HV, HC])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private[..], [0, 1]);
        assert_eq!(map.staging, 1, "staging should use the coherent type");
        assert_eq!(
            map.dynamic, 1,
            "dynamic gives up HOST_CACHED for HOST_COHERENT"
        );
        assert!(map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [1],
            "upload skips the non-coherent type listed first"
        );
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 1, "uniform must be HOST_COHERENT");
    }

    // AMD (Mesa radv) layouts below follow radv_physical_device_init_mem_types. Heaps are
    // VRAM the CPU can't map, GTT (system RAM), then CPU-mappable VRAM, each only if present.
    // Types are VRAM, GTT write-combined, CPU-mappable VRAM, then GTT cached. Every type but
    // GTT write-combined is followed by a copy with identical flags in the 32-bit address
    // space, for descriptor buffers only: normal buffers and images can't use those copies,
    // so the lowest-index tie must win. TMZ adds PROTECTED copies (GFX10+ and Vega10, when the
    // kernel enables it), and GFX9+ then appends DEVICE_COHERENT_AMD | DEVICE_UNCACHED_AMD
    // copies of the types that are neither 32-bit nor PROTECTED.

    /// Checks `map` against the memoryTypeBits radv reports (radv_buffer.c, radv_device.c).
    /// `buffer_bits` are the types buffers and images may use: every type but the 32-bit
    /// copies. Returns the types a buffer and an uploaded buffer get.
    fn radv_selection(map: &MemoryTypeMapInner, buffer_bits: u32) -> (u32, u32) {
        let allowed = |i: u32| buffer_bits & (1 << i) != 0;
        assert!(
            allowed(map.staging),
            "staging must be a type buffers can use"
        );
        assert!(
            allowed(map.dynamic),
            "dynamic must be a type buffers can use"
        );
        if map.uniform != u32::MAX {
            assert!(
                allowed(map.uniform),
                "uniform must be a type buffers can use"
            );
        }
        (
            first_allowed(&map.private, buffer_bits).unwrap(),
            first_allowed(&map.upload, buffer_bits).unwrap(),
        )
    }

    /// AMD discrete GPU, GFX9+ (Vega, RDNA), without resizable BAR: the CPU-mappable part of
    /// VRAM is split into its own 256MB heap. `radv_hide_rebar_on_dgpu` exposes the same
    /// layout with resizable BAR.
    #[test]
    fn test_amd_256mb_bar() {
        // AMD RX 6800 XT without SAM:
        // Heap 0: 15.75GB VRAM not mappable by the CPU (DEVICE_LOCAL)
        // Heap 1: 16GB GTT
        // Heap 2: 256MB CPU-mappable VRAM (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 2: HOST_VISIBLE | HOST_COHERENT (heap 1) - write-combined
        // Type 3: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 2) - 256MB BAR
        // Type 5: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        // Types 1, 4, 6: 32-bit copies of types 0, 3, 5
        // Types 7-10: DEVICE_COHERENT_AMD copies of types 0, 2, 3, 5
        let heaps = [
            mem_heap(16 * GB - 256 * MB, true),
            mem_heap(16 * GB, false),
            mem_heap(256 * MB, true),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(2, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(
            map.private[..],
            [0, 1, 3, 4],
            "private should use main VRAM heap"
        );
        assert_eq!(
            map.staging, 2,
            "staging should use write-combined system RAM"
        );
        assert_eq!(map.dynamic, 5, "dynamic should use HOST_CACHED");
        assert!(!map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [0, 1, 3, 4],
            "upload should use private, requiring staging"
        );
        assert!(
            !map.upload_host_visible,
            "256MB BAR does not allow direct upload"
        );
        assert_eq!(map.uniform, 3, "uniform should use 256MB BAR");
        assert_eq!(radv_selection(&map, 0b111_1010_1101), (0, 0));
    }

    /// AMD discrete GPU, GFX9+, with SAM (Smart Access Memory) / resizable BAR. All of VRAM
    /// is CPU-mappable, so there's no heap for VRAM the CPU can't map.
    #[test]
    fn test_amd_sam() {
        // AMD RX 6800 XT with SAM enabled:
        // Heap 0: 16GB GTT
        // Heap 1: 16GB VRAM (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL (heap 1)
        // Type 2: HOST_VISIBLE | HOST_COHERENT (heap 0) - write-combined
        // Type 3: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 1) - full ReBAR
        // Type 5: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 0)
        // Types 1, 4, 6: 32-bit copies of types 0, 3, 5
        // Types 7-10: DEVICE_COHERENT_AMD copies of types 0, 2, 3, 5
        let heaps = [mem_heap(16 * GB, false), mem_heap(16 * GB, true)];
        let types = [
            mem_type(1, DL),
            mem_type(1, DL),
            mem_type(0, flags(&[HV, HC])),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[HV, HC, HCA])),
            mem_type(1, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(0, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(0, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(
            map.private[..],
            [0, 1, 3, 4],
            "private should prefer pure DEVICE_LOCAL"
        );
        assert_eq!(map.staging, 2, "staging should use system RAM");
        assert_eq!(map.dynamic, 5, "dynamic should use HOST_CACHED");
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [3], "upload should use SAM/ReBAR type");
        assert!(map.upload_host_visible, "SAM allows direct upload");
        assert_eq!(map.uniform, 3, "uniform should use SAM/ReBAR type");
        assert_eq!(radv_selection(&map, 0b111_1010_1101), (0, 3));
    }

    /// AMD discrete GPU, GFX9+, with a BAR larger than 256MB that still leaves over 10% of
    /// VRAM unmappable (e.g. firmware capping the resizable BAR): radv keeps the three-heap
    /// layout of test_amd_256mb_bar, but the CPU-mappable heap is large enough for uploads.
    /// Illustrative layout, not captured from real hardware.
    #[test]
    fn test_amd_partial_bar() {
        // AMD RX 7900 XTX with a 16GB BAR:
        // Heap 0: 8GB VRAM not mappable by the CPU (DEVICE_LOCAL)
        // Heap 1: 16GB GTT
        // Heap 2: 16GB CPU-mappable VRAM (DEVICE_LOCAL)
        // Types 0-10: as in test_amd_256mb_bar
        let heaps = [
            mem_heap(8 * GB, true),
            mem_heap(16 * GB, false),
            mem_heap(16 * GB, true),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(2, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(
            map.private[..],
            [0, 1, 3, 4],
            "private should prefer VRAM the CPU can't map, even from a smaller heap"
        );
        assert_eq!(map.staging, 2);
        assert_eq!(map.dynamic, 5);
        assert!(!map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [3],
            "a 16GB BAR is large enough for uploads"
        );
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 3);
        assert_eq!(radv_selection(&map, 0b111_1010_1101), (0, 3));
    }

    /// AMD discrete GPU before GFX9 (Southern Islands through Polaris): no
    /// DEVICE_COHERENT_AMD types.
    #[test]
    fn test_amd_gfx8_discrete() {
        // AMD RX 580 (Polaris) without resizable BAR:
        // Heap 0: 7.75GB VRAM not mappable by the CPU (DEVICE_LOCAL)
        // Heap 1: 16GB GTT
        // Heap 2: 256MB CPU-mappable VRAM (DEVICE_LOCAL)
        // Types 0-6: as in test_amd_256mb_bar
        let heaps = [
            mem_heap(8 * GB - 256 * MB, true),
            mem_heap(16 * GB, false),
            mem_heap(256 * MB, true),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(1, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0, 1, 3, 4]);
        assert_eq!(map.staging, 2);
        assert_eq!(map.dynamic, 5);
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [0, 1, 3, 4]);
        assert!(!map.upload_host_visible);
        assert_eq!(map.uniform, 3);
        assert_eq!(radv_selection(&map, 0b010_1101), (0, 0));
    }

    /// AMD discrete GPU, GFX10+ or Vega10, with TMZ enabled by the kernel (`amdgpu.tmz=1`).
    /// Unlike on APUs, the PROTECTED VRAM types are split between the VRAM heap the CPU can't
    /// map and the 256MB BAR heap, so the largest DEVICE_LOCAL heap has PROTECTED types.
    #[test]
    fn test_amd_tmz_discrete() {
        // AMD RX 6800 XT without SAM:
        // Heaps as in test_amd_256mb_bar
        // Types 0-6: as in test_amd_256mb_bar
        // Types 7-8: DEVICE_LOCAL | PROTECTED (heap 0), VRAM the CPU can't map
        // Types 9-10: DEVICE_LOCAL | PROTECTED (heap 2), CPU-mappable VRAM
        // Types 11-12: PROTECTED (heap 1), GTT
        // Types 13-16: DEVICE_COHERENT_AMD copies of types 0, 2, 3, 5
        let heaps = [
            mem_heap(16 * GB - 256 * MB, true),
            mem_heap(16 * GB, false),
            mem_heap(256 * MB, true),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[DL, PROT])),
            mem_type(0, flags(&[DL, PROT])),
            mem_type(2, flags(&[DL, PROT])),
            mem_type(2, flags(&[DL, PROT])),
            mem_type(1, PROT),
            mem_type(1, PROT),
            mem_type(0, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(2, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0, 1, 3, 4]);
        assert_eq!(map.staging, 2);
        assert_eq!(map.dynamic, 5);
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [0, 1, 3, 4]);
        assert!(!map.upload_host_visible);
        assert_eq!(map.uniform, 3);
        assert_eq!(radv_selection(&map, 0b1_1110_1010_1010_1101), (0, 0));
    }

    /// AMD APU, GFX9+ (Raven through Strix, including Steam Deck's Van Gogh). radv adds the
    /// VRAM carve-out and GTT together and reports 2/3 of the total as one CPU-mappable
    /// VRAM heap and the rest as GTT, however small the carve-out is.
    #[test]
    fn test_amd_apu() {
        // AMD Ryzen 7 6800U (Rembrandt), 1GB carve-out and 15GB GTT:
        // Heap 0: 1/3 of 16GB GTT
        // Heap 1: 2/3 of 16GB "VRAM" (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL (heap 1)
        // Type 2: HOST_VISIBLE | HOST_COHERENT (heap 0) - write-combined
        // Type 3: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 1)
        // Type 5: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 0)
        // Types 1, 4, 6: 32-bit copies of types 0, 3, 5
        // Types 7-10: DEVICE_COHERENT_AMD copies of types 0, 2, 3, 5
        let total = GB + 15 * GB;
        let vram = (total * 2 / 3).next_multiple_of(4096);
        let heaps = [mem_heap(total - vram, false), mem_heap(vram, true)];
        let types = [
            mem_type(1, DL),
            mem_type(1, DL),
            mem_type(0, flags(&[HV, HC])),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[HV, HC, HCA])),
            mem_type(1, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(0, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(0, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private[..], [0, 1, 3, 4]);
        assert_eq!(map.staging, 2, "staging should use write-combined GTT");
        assert_eq!(map.dynamic, 5, "dynamic should use HOST_CACHED GTT");
        assert!(map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [3],
            "the CPU-mappable VRAM heap is large enough for uploads"
        );
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 3);
        assert_eq!(radv_selection(&map, 0b111_1010_1101), (0, 3));
    }

    /// AMD APU, GFX10+, with TMZ enabled by the kernel (`amdgpu.tmz=1`): PROTECTED types
    /// come between the 32-bit copies and the DEVICE_COHERENT_AMD types.
    #[test]
    fn test_amd_apu_tmz() {
        // Heaps as in test_amd_apu
        // Types 0-6: as in test_amd_apu
        // Types 7-10: DEVICE_LOCAL | PROTECTED (heap 1), VRAM and CPU-mappable VRAM
        // Types 11-12: PROTECTED (heap 0), GTT
        // Types 13-16: DEVICE_COHERENT_AMD copies of types 0, 2, 3, 5
        let total = GB + 15 * GB;
        let vram = (total * 2 / 3).next_multiple_of(4096);
        let heaps = [mem_heap(total - vram, false), mem_heap(vram, true)];
        let types = [
            mem_type(1, DL),
            mem_type(1, DL),
            mem_type(0, flags(&[HV, HC])),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[HV, HC, HCA])),
            mem_type(1, flags(&[DL, PROT])),
            mem_type(1, flags(&[DL, PROT])),
            mem_type(1, flags(&[DL, PROT])),
            mem_type(1, flags(&[DL, PROT])),
            mem_type(0, PROT),
            mem_type(0, PROT),
            mem_type(1, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(0, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(0, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private[..], [0, 1, 3, 4]);
        assert_eq!(map.staging, 2);
        assert_eq!(map.dynamic, 5);
        assert!(map.dynamic_device_local);
        assert_eq!(map.upload[..], [3]);
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 3);
        assert_eq!(radv_selection(&map, 0b1_1110_1010_1010_1101), (0, 3));
    }

    /// AMD APU before GFX9 (Kaveri, Carrizo, Bristol Ridge): no DEVICE_COHERENT_AMD types.
    #[test]
    fn test_amd_gfx8_apu() {
        // AMD A10-9700 (Bristol Ridge), 512MB carve-out and 8GB GTT:
        // Heaps as in test_amd_apu
        // Types 0-6: as in test_amd_apu
        let total = 512 * MB + 8 * GB;
        let vram = (total * 2 / 3).next_multiple_of(4096);
        let heaps = [mem_heap(total - vram, false), mem_heap(vram, true)];
        let types = [
            mem_type(1, DL),
            mem_type(1, DL),
            mem_type(0, flags(&[HV, HC])),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private[..], [0, 1, 3, 4]);
        assert_eq!(map.staging, 2);
        assert_eq!(map.dynamic, 5);
        assert!(map.dynamic_device_local);
        assert_eq!(map.upload[..], [3]);
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 3);
        assert_eq!(radv_selection(&map, 0b010_1101), (0, 3));
    }

    /// AMD APU with `radv_enable_unified_heap_on_apu`: the carve-out and GTT become one
    /// DEVICE_LOCAL heap with only VRAM types, so there's no HOST_CACHED type.
    #[test]
    fn test_amd_apu_unified_heap() {
        // Heap 0: 16GB "VRAM" (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL
        // Type 2: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
        // Types 1, 3: 32-bit copies of types 0, 2
        // Types 4-5: DEVICE_COHERENT_AMD copies of types 0, 2
        let heaps = [mem_heap(16 * GB, true)];
        let types = [
            mem_type(0, DL),
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, flags(&[DL, DLC_AMD, DLU_AMD])),
            mem_type(0, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private[..], [0, 1, 2, 3]);
        assert_eq!(map.staging, 2);
        assert_eq!(map.dynamic, 2, "dynamic falls back to the uncached type");
        assert!(map.dynamic_device_local);
        assert_eq!(map.upload[..], [2]);
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 2);
        assert_eq!(radv_selection(&map, 0b11_0101), (0, 2));
    }

    /// Integrated GPU whose only DEVICE_LOCAL + HOST_VISIBLE heap is a small carve-out, so
    /// uploads go to system RAM instead. radv doesn't expose this layout (see
    /// test_amd_apu). Illustrative layout, not captured from real hardware.
    #[test]
    fn test_integrated_small_carveout() {
        // Heap 0: 256MB carve-out (DEVICE_LOCAL)
        // Heap 1: 16GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 0)
        // Type 2: HOST_VISIBLE | HOST_COHERENT (heap 1)
        // Type 3: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        let heaps = [mem_heap(256 * MB, true), mem_heap(16 * GB, false)];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(1, flags(&[HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        assert_eq!(map.private[..], [0, 1]);
        assert_eq!(map.staging, 2);
        assert_eq!(map.dynamic, 3);
        assert!(map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [2],
            "upload should fall back to HOST_VISIBLE (system RAM)"
        );
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 1, "uniform should use the carve-out");
    }

    /// Intel Arc A-series (Alchemist) discrete GPU on i915 with resizable BAR. DG1 (Iris Xe
    /// MAX) uses the same layout. There's no uncached system RAM type.
    #[test]
    fn test_intel_arc_rebar() {
        // Intel Arc A770:
        // Heap 0: 16GB VRAM (DEVICE_LOCAL)
        // Heap 1: 24GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        // Type 2: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 0) - ReBAR
        // Types 3-5: descriptor buffer copies of types 0-2
        let heaps = [mem_heap(16 * GB, true), mem_heap(24 * GB, false)];
        let types = [
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[DL, HV, HC])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(
            map.private[..],
            [0, 3, 2, 5],
            "private should use pure DEVICE_LOCAL"
        );
        assert_eq!(
            map.staging, 1,
            "host should use system RAM, even though it's HOST_CACHED"
        );
        assert_eq!(map.dynamic, 1, "dynamic should use HOST_CACHED system RAM");
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [2], "upload should use ReBAR");
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 2, "uniform should use ReBAR type");
        assert_eq!(anv_selection(&map, 0b111, 0), (0, 0, 2));
    }

    /// Intel Arc A-series discrete GPU on i915 without resizable BAR: the CPU-mappable part
    /// of VRAM is split into its own 256MB heap.
    #[test]
    fn test_intel_arc_small_bar() {
        // Heap 0: 15.75GB VRAM not mappable by the CPU (DEVICE_LOCAL)
        // Heap 1: 24GB system RAM
        // Heap 2: 256MB CPU-mappable VRAM (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        // Type 2: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 2)
        // Types 3-5: descriptor buffer copies of types 0-2
        let heaps = [
            mem_heap(16 * GB - 256 * MB, true),
            mem_heap(24 * GB, false),
            mem_heap(256 * MB, true),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(2, flags(&[DL, HV, HC])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0, 3, 2, 5]);
        assert_eq!(
            map.staging, 1,
            "staging should use system RAM, not the 256MB BAR heap uniforms share"
        );
        assert_eq!(map.dynamic, 1, "dynamic should use HOST_CACHED system RAM");
        assert!(!map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [0, 3, 2, 5],
            "upload should use private, requiring staging"
        );
        assert!(!map.upload_host_visible);
        assert_eq!(map.uniform, 2, "uniform should use 256MB BAR");
        assert_eq!(anv_selection(&map, 0b111, 0), (0, 0, 0));
    }

    /// Intel Arc B-series (Battlemage) discrete GPU on the Xe KMD with resizable BAR.
    #[test]
    fn test_intel_battlemage() {
        // Intel Arc B580:
        // Heap 0: 12GB VRAM (DEVICE_LOCAL)
        // Heap 1: 24GB system RAM
        // Type 0: DEVICE_LOCAL (heap 0), compressed (images only)
        // Type 1: DEVICE_LOCAL (heap 0)
        // Type 2: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        // Type 3: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 0) - ReBAR
        // Types 4-6: descriptor buffer copies of types 1-3
        let heaps = [mem_heap(12 * GB, true), mem_heap(24 * GB, false)];
        let types = [
            mem_type(0, DL),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[DL, HV, HC])),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[DL, HV, HC])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        // anv's memoryTypeBits allow types 1-3 for buffers and images that can't be
        // compressed, so they get type 1; compressible images also allow type 0.
        assert_eq!(map.private[..], [0, 1, 4, 3, 6]);
        assert_eq!(
            map.staging, 2,
            "host should use system RAM, even though it's HOST_CACHED"
        );
        assert_eq!(map.dynamic, 2, "dynamic should use HOST_CACHED system RAM");
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [3], "upload should use ReBAR");
        assert!(map.upload_host_visible);
        assert_eq!(map.uniform, 3, "uniform should use ReBAR type");
        assert_eq!(
            anv_selection(&map, 0b1110, 0b0001),
            (1, 0, 3),
            "buffers skip the compressed type, compressible images take it"
        );
    }

    /// Intel Arc B-series discrete GPU on the Xe KMD without resizable BAR: the compressed
    /// type of Xe2 plus the 256MB CPU-mappable heap split out of VRAM.
    #[test]
    fn test_intel_battlemage_small_bar() {
        // Heap 0: 11.75GB VRAM not mappable by the CPU (DEVICE_LOCAL)
        // Heap 1: 24GB system RAM
        // Heap 2: 256MB CPU-mappable VRAM (DEVICE_LOCAL)
        // Type 0: DEVICE_LOCAL (heap 0), compressed (images only)
        // Type 1: DEVICE_LOCAL (heap 0)
        // Type 2: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        // Type 3: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (heap 2)
        // Types 4-6: descriptor buffer copies of types 1-3
        let heaps = [
            mem_heap(12 * GB - 256 * MB, true),
            mem_heap(24 * GB, false),
            mem_heap(256 * MB, true),
        ];
        let types = [
            mem_type(0, DL),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(2, flags(&[DL, HV, HC])),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(2, flags(&[DL, HV, HC])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0, 1, 4, 3, 6]);
        assert_eq!(
            map.staging, 2,
            "staging should use system RAM, not the 256MB BAR heap uniforms share"
        );
        assert_eq!(map.dynamic, 2, "dynamic should use HOST_CACHED system RAM");
        assert!(!map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [0, 1, 4, 3, 6],
            "upload should use private, requiring staging"
        );
        assert!(!map.upload_host_visible);
        assert_eq!(map.uniform, 3, "uniform should use 256MB BAR");
        assert_eq!(
            anv_selection(&map, 0b1110, 0b0001),
            (1, 0, 1),
            "uploaded buffers must skip the compressed type"
        );
    }

    /// Qualcomm Adreno (mobile GPU in Android/Windows on ARM).
    /// Unified memory similar to other integrated GPUs.
    #[test]
    fn test_qualcomm_adreno() {
        // Qualcomm Adreno 740 pattern:
        // Single heap, unified memory
        let heaps = [mem_heap(8 * GB, true)];
        let types = [mem_type(0, DL), mem_type(0, flags(&[DL, HV, HC, HCA]))];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::INTEGRATED_GPU);

        // Should prefer non-HOST_VISIBLE for private if available
        assert_eq!(
            map.private[..],
            [0, 1],
            "private should use pure DEVICE_LOCAL"
        );
        assert_eq!(map.staging, 1, "host should use HOST_VISIBLE type");
        assert_eq!(map.dynamic, 1, "dynamic should use HOST_CACHED type");
        assert!(map.dynamic_device_local);
        assert_eq!(
            map.upload[..],
            [1],
            "upload should use HOST_VISIBLE DEVICE_LOCAL"
        );
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

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [0]);
        assert_eq!(map.staging, 1);
        assert_eq!(map.dynamic, 1);
        assert!(!map.dynamic_device_local);
        assert_eq!(map.upload[..], [0], "upload must use DEVICE_LOCAL");
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

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        // Should prefer the larger VRAM heap
        assert_eq!(
            map.private[..],
            [1, 0],
            "private should use larger VRAM heap"
        );
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

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        // Upload should prefer the larger ReBAR heap over the 256MB BAR
        assert_eq!(
            map.upload[..],
            [1],
            "upload should prefer larger ReBAR heap"
        );
        assert!(map.upload_host_visible);
        assert_eq!(
            map.uniform, 1,
            "uniform should prefer larger ReBAR heap over 256MB BAR"
        );
    }

    /// Discrete GPU with a cache-coherent link to the host (e.g. NVIDIA Grace Hopper).
    /// Illustrative layout, not captured from real hardware.
    #[test]
    fn test_coherent_link_discrete() {
        // Heap 0: VRAM (DEVICE_LOCAL)
        // Heap 1: System RAM
        // Type 0: DEVICE_LOCAL (heap 0)
        // Type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 0)
        // Type 2: HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (heap 1)
        let heaps = [mem_heap(96 * GB, true), mem_heap(480 * GB, false)];
        let types = [
            mem_type(0, DL),
            mem_type(0, flags(&[DL, HV, HC, HCA])),
            mem_type(1, flags(&[HV, HC, HCA])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(
            map.dynamic, 1,
            "dynamic should prefer DEVICE_LOCAL + HOST_CACHED"
        );
        assert!(
            map.dynamic_device_local,
            "device-local cached memory needs no staging copy"
        );
    }

    /// Memory types general-purpose allocations can't use are never selected, even when
    /// they're listed before an equivalent usable type.
    #[test]
    fn test_skips_unusable_types() {
        let heaps = [mem_heap(8 * GB, true), mem_heap(16 * GB, false)];
        let types = [
            mem_type(0, flags(&[DL, LAZY])),
            mem_type(0, flags(&[DL, PROT])),
            mem_type(1, flags(&[HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(1, flags(&[HV, HC, HCA, DLC_AMD, DLU_AMD])),
            mem_type(0, flags(&[DL, HV, HC, DLC_AMD, DLU_AMD])),
            mem_type(0, DL),
            mem_type(1, flags(&[HV, HC])),
            mem_type(1, flags(&[HV, HC, HCA])),
            mem_type(0, flags(&[DL, HV, HC])),
        ];

        let map = MemoryTypeMapInner::new(&types, &heaps, vk::PhysicalDeviceType::DISCRETE_GPU);

        assert_eq!(map.private[..], [5, 8]);
        assert_eq!(map.staging, 6);
        assert_eq!(map.dynamic, 7);
        assert_eq!(map.upload[..], [8]);
        assert_eq!(map.uniform, 8);
    }
}
