//! Logical device creation and management.
//!
//! This module provides the core [`Device`] type and [`DeviceBuilder`] for creating
//! and configuring Vulkan logical devices.
//!
//! # Overview
//!
//! A Vulkan logical device represents a connection to the driver of a physical GPU
//! (or more generally, with any Vulkan implementation) with a specific
//! configuration of extensions, features, and queues. This module provides:
//!
//! - [`Device`]: The main device handle, reference-counted for cheap sharing
//! - [`DeviceBuilder`]: Fluent builder for configuring devices before creation
//! - [`DeviceQueueRef`]: Reference to a queue that will be created with the device
//! - [`HasDevice`]: Trait for types associated with a device
//!
//! # Quick Start
//!
//! For simple use cases, [`Device::create_system_default`] creates a device with
//! sensible defaults:
//!
//! ```
//! # use pumicite::Device;
//! let (device, queue) = Device::create_system_default().unwrap();
//! ```
//!
//! # Custom Configuration
//!
//! For more control, use the builder pattern:
//!
//! ```
//! # use std::sync::Arc;
//! # use pumicite::{Instance, Device, ash::vk};
//! # let entry = Arc::new(unsafe { ash::Entry::load() }.unwrap());
//! # let instance = Instance::builder(entry).build().unwrap();
//! # let physical_device = instance.enumerate_physical_devices().unwrap().next().unwrap();
//! let mut builder = Device::builder(physical_device);
//!
//! // Enable extensions
//! builder.enable_extension::<ash::khr::swapchain::Meta>().ok();
//!
//! // Enable features
//! builder.enable_feature::<vk::PhysicalDeviceBufferDeviceAddressFeatures>(|f| {
//!     &mut f.buffer_device_address
//! }).unwrap();
//!
//! // Enable queues
//! let graphics_queue = builder.enable_queue_with_caps(
//!     vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
//!     1.0,
//! ).unwrap();
//!
//! // Build the device
//! let device = builder.build().unwrap();
//! let queue = device.get_queue(graphics_queue);
//! ```
//!
//! # Bindless Resources
//!
//! Enable bindless rendering for descriptor indexing:
//!
//! ```
//! # use std::sync::Arc;
//! # use pumicite::{Instance, Device, bindless::BindlessConfig};
//! # let entry = Arc::new(unsafe { ash::Entry::load() }.unwrap());
//! # let instance = Instance::builder(entry).build().unwrap();
//! # let pdevice = instance.enumerate_physical_devices().unwrap().next().unwrap();
//! let mut builder = Device::builder(pdevice);
//! builder.enable_bindless(BindlessConfig::default()).unwrap();
//! builder.enable_queue(0, 1.0);
//! let device = builder.build().unwrap();
//!
//! // Access the bindless heap
//! let heap = device.bindless_heap();
//! ```

use crate::{
    Extension, Instance, MissingFeatureError,
    debug::DebugObject,
    physical_device::{Feature, PhysicalDevice, PhysicalDeviceFeatureMap},
    queue::Queue,
    utils::{AsVkHandle, NextChainMap, Version},
};
use ash::vk;
use ash::vk::ExtensionMeta;
use ash::{VkResult, vk::TaggedStructure};

use std::{
    any::Any,
    collections::BTreeMap,
    ffi::{CStr, CString},
    fmt::Debug,
    ops::Deref,
    sync::Arc,
};

/// A trait for types created from a Vulkan device.
///
/// This trait provides a common interface for accessing the Vulkan logical device.
pub trait HasDevice {
    /// Returns a reference to the Vulkan device.
    fn device(&self) -> &Device;

    /// Returns a reference to the Vulkan [`PhysicalDevice`].
    ///
    /// This is a convenience method that delegates to `self.device().physical_device()`.
    fn physical_device(&self) -> &PhysicalDevice {
        self.device().physical_device()
    }

    /// Returns a reference to the Vulkan [`Instance`].
    ///
    /// This is a convenience method that delegates to `self.device().physical_device().instance()`.
    fn instance(&self) -> &Instance {
        self.device().physical_device().instance()
    }
}

/// A Vulkan logical device wrapper.
///
/// This struct represents a Vulkan logical device and provides a high-level
/// interface for device operations. It's reference-counted using [`Arc`] for
/// cheap shared access.
///
/// The device provides access to:
/// - Vulkan extensions and their functionality
/// - Device features
/// - Queue management
/// - Resource creation and management
#[derive(Clone)]
pub struct Device(Arc<DeviceInner>);
impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Device {}
impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Device")
            .field(&self.0.device.handle())
            .finish()
    }
}

struct DeviceInner {
    physical_device: PhysicalDevice,
    device: ash::Device,
    /// Map of enabled extensions and their function loaders
    extensions: BTreeMap<&'static CStr, Option<Box<dyn Any + Send + Sync>>>,
    /// Chain of enabled device features
    features: NextChainMap<vk::PhysicalDeviceFeatures2<'static>>,
    /// Channel for scheduling GPU resources for deferred cleanup
    recycler: crossbeam_channel::Sender<crate::sync::RetiredGPUMutex>,
}
unsafe impl Send for DeviceInner {}
unsafe impl Sync for DeviceInner {}

impl Device {
    /// Returns a reference to the Vulkan [`Instance`].
    pub fn instance(&self) -> &Instance {
        self.0.physical_device.instance()
    }

    /// Returns a reference to the [`PhysicalDevice`]
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.0.physical_device
    }

    /// Gets a reference to an enabled device extension.
    ///
    /// Only applicable to extensions not promoted to Vulkan core.
    /// For extensions promoted to Vulkan core, you may directly call the corresponding
    /// function on [`Device`].
    ///
    /// # Returns
    ///
    /// Returns `Ok(&T::Device)` if the extension is enabled, or `Err(MissingFeatureError)`
    /// if the extension was not enabled during device creation.
    ///
    /// # Panics
    ///
    /// Panics if the extension was enabled but did not provide additional commands.
    /// Use [`has_extension_named`](Self::has_extension_named) to test if an extension
    /// was enabled without additional commands.
    pub fn get_extension<T: ExtensionMeta>(&self) -> Result<&T::Device, MissingFeatureError>
    where
        T::Device: 'static,
    {
        self.0
            .extensions
            .get(T::NAME)
            .map(|item| item.as_ref().expect("Extension did not add any additional commands; use `has_extension_named` to test if the extension was enabled.").downcast_ref::<T::Device>().unwrap())
            .ok_or(MissingFeatureError::Extension(T::NAME))
    }

    /// Checks if a device extension is enabled by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The extension name as a C string
    ///
    /// # Returns
    ///
    /// `true` if the extension is enabled, `false` otherwise.
    pub fn has_extension_named(&self, name: &CStr) -> bool {
        self.0.extensions.contains_key(name)
    }

    /// Gets a reference to an enabled device extension, panicking if not found.
    ///
    /// Only applicable to extensions not promoted to Vulkan core.
    /// For extensions promoted to Vulkan core, you may directly call the corresponding
    /// function on [`Device`].
    ///
    /// # Panics
    ///
    /// Panics if the extension is not enabled. Use [`get_extension`](Self::get_extension)
    /// for a non-panicking version.
    #[track_caller]
    pub fn extension<T: ExtensionMeta>(&self) -> &T::Device
    where
        T::Device: 'static,
    {
        self.get_extension::<T>().unwrap()
    }

    /// Gets a reference to an enabled device feature.
    ///
    /// # Returns
    ///
    /// `Some(&T)` if the feature is enabled, `None` otherwise.
    pub fn feature<T: Feature + Default + 'static>(&self) -> Option<&T> {
        T::get_from_chain(&self.0.features)
    }

    /// Creates a system default device with basic configuration.
    ///
    /// This is a convenience method that creates an [`Instance`], selects the first
    /// available [`PhysicalDevice`], and creates a logical device with a default queue.
    /// The device will have essential extensions enabled and use Vulkan 1.2.
    ///
    /// # Returns
    ///
    /// Returns `Ok((Device, Queue))` on success, where the queue is from family 0
    /// with graphics, compute, and transfer capabilities.
    ///
    /// # Errors
    ///
    /// Returns a Vulkan error if device creation fails or if no compatible device
    /// is found.
    pub fn create_system_default() -> VkResult<(Self, Queue)> {
        let entry = Arc::new(unsafe { ash::Entry::load() }.unwrap());
        let mut instance_builder = Instance::builder(entry);
        instance_builder.info.api_version = Version::V1_2;
        instance_builder
            .enable_extension::<ash::ext::debug_utils::Meta>()
            .ok();
        let instance = instance_builder.build()?;

        let pdevice = instance
            .enumerate_physical_devices()
            .unwrap()
            .next()
            .unwrap();
        let mut builder = Device::builder(pdevice);
        let queue = builder
            // Typically queue family 0 is the "default" queue with GRAPHICS, COMPUTE and TRANSFER capabilities.
            .enable_queue(0, 1.0)
            .ok_or(vk::Result::ERROR_INCOMPATIBLE_DRIVER)?;
        let device = builder.build()?;
        let queue = device.get_queue(queue).with_name(c"System Default Queue");
        Ok((device, queue))
    }

    /// Creates a new device builder for the given physical device.
    ///
    /// # Arguments
    ///
    /// * `pdevice` - The physical device to create a logical device from
    ///
    /// # Returns
    ///
    /// A new [`DeviceBuilder`] instance for configuring the device.
    pub fn builder(pdevice: PhysicalDevice) -> DeviceBuilder {
        DeviceBuilder::new(pdevice)
    }

    /// Gets a queue handle from a queue reference.
    ///
    /// # Arguments
    ///
    /// * `queue` - A reference to the queue obtained during device creation
    ///
    /// # Returns
    ///
    /// A [`Queue`] handle that can be used for command submission.
    ///
    /// # Panics
    ///
    /// Panics if the queue reference doesn't belong to this device.
    pub fn get_queue(&self, queue: DeviceQueueRef) -> Queue {
        assert!(self.physical_device() == &queue.pdevice);
        unsafe {
            let raw_queue = self.0.device.get_device_queue(queue.family, queue.index);
            Queue::from_raw(self.clone(), raw_queue, queue.family_index(), queue.flags)
        }
    }

    /// Schedules a GPU resource for deferred cleanup.
    ///
    /// This is used internally to safely clean up GPU resources that may still
    /// be in use by the GPU when they're dropped on the CPU side.
    pub(crate) fn schedule_resource_for_deferred_drop(&self, item: crate::sync::RetiredGPUMutex) {
        self.0.recycler.send(item).unwrap();
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.0.device
    }
}
impl AsVkHandle for Device {
    type Handle = vk::Device;

    fn vk_handle(&self) -> Self::Handle {
        self.0.device.handle()
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        tracing::info!(device = ?self.device.handle(), "drop device");
        self.extensions.clear();
        // Safety: Host synchronization rule for vkDestroyDevice:
        // - Host access to device must be externally synchronized.
        // - Host access to all VkQueue objects created from device must be externally synchronized
        // We have &mut self and therefore exclusive control on device.
        // VkQueue objects may not exist at this point, because Queue retains an Arc to Device.
        // If there still exist a Queue, the Device wouldn't be dropped.
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

/// A function type for building device extension metadata.
///
/// This function takes a Vulkan instance and device, and returns a boxed
/// extension loader that can be stored and used later. It's used internally
/// during device creation to initialize extension function pointers.
pub(crate) type DeviceMetaBuilder =
    fn(&ash::Instance, &mut ash::Device) -> Box<dyn Any + Send + Sync>;

/// A builder for creating Vulkan logical devices.
///
/// This builder provides a fluent interface for configuring a Vulkan device
/// before creation. It allows you to:
/// - Enable device extensions
/// - Enable device features
/// - Configure queues with specific capabilities
/// - Set queue priorities
pub struct DeviceBuilder {
    /// The physical device to create the logical device from
    pdevice: PhysicalDevice,
    /// Map of device features that can be enabled
    features: PhysicalDeviceFeatureMap,
    /// Available extensions on this physical device
    available_extensions: BTreeMap<CString, Version>,

    /// Queue family properties, indexed by queue family index
    queue_info: Box<[vk::QueueFamilyProperties]>,
    /// Queue creation info, indexed by queue family index
    queue_create_info: Box<[vk::DeviceQueueCreateInfo<'static>]>,

    /// Queue priorities storage (unused but required for lifetime management)
    #[allow(dead_code)]
    queue_priorities: Box<[f32]>,

    /// Extensions that will be enabled on the device
    enabled_extensions: BTreeMap<&'static CStr, Option<DeviceMetaBuilder>>,

    /// Will be set to true if [`Self::enable_bindless`] was ever called.
    pub(crate) bindless_enabled: bool,
}

/// An owned reference to a device queue that will be created.
///
/// This struct is returned by [`DeviceBuilder::enable_queue`] and related methods.
/// It represents a queue that will be available after device creation and can be
/// used with [`Device::get_queue`] to obtain the actual queue handle.
pub struct DeviceQueueRef {
    /// The physical device this queue belongs to
    pdevice: PhysicalDevice,
    /// The queue family index
    family: u32,
    /// The queue index within the family
    index: u32,
    /// The queue's supported operations
    flags: vk::QueueFlags,
}

impl DeviceQueueRef {
    /// Returns the queue family index.
    pub fn family_index(&self) -> u32 {
        self.family
    }

    /// Returns the queue's supported operation flags.
    pub fn flags(&self) -> vk::QueueFlags {
        self.flags
    }
}

unsafe impl Send for DeviceBuilder {}
unsafe impl Sync for DeviceBuilder {}
impl DeviceBuilder {
    /// Creates a new device builder for the given physical device.
    ///
    /// The builder is initialized with default required extensions and features:
    /// - `VK_KHR_synchronization2` extension and feature
    /// - `VK_KHR_timeline_semaphore` extension and feature
    ///
    /// # Arguments
    ///
    /// * `pdevice` - The physical device to create a logical device from
    pub fn new(pdevice: PhysicalDevice) -> Self {
        let extension_names = unsafe {
            pdevice
                .instance()
                .enumerate_device_extension_properties(pdevice.vk_handle())
                .unwrap()
        };
        let extension_names = extension_names
            .into_iter()
            .map(|ext| {
                let str = ext.extension_name_as_c_str().unwrap();
                (str.to_owned(), Version(ext.spec_version))
            })
            .collect::<BTreeMap<CString, Version>>();
        let queue_info = pdevice.get_queue_family_properties().into_boxed_slice();
        let max_queue_count: u32 = queue_info.iter().map(|x| x.queue_count).sum();
        let queue_priorities: Box<[f32]> = vec![0.0; max_queue_count as usize].into_boxed_slice();
        let mut this = Self {
            available_extensions: extension_names,
            enabled_extensions: BTreeMap::new(),
            features: PhysicalDeviceFeatureMap::new(pdevice.clone()),
            pdevice,
            bindless_enabled: false,
            queue_create_info: {
                let mut count: u32 = 0;
                queue_info
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let info = vk::DeviceQueueCreateInfo {
                            queue_family_index: i as u32,
                            queue_count: 0,
                            p_queue_priorities: unsafe {
                                queue_priorities.as_ptr().add(count as usize)
                            },
                            ..Default::default()
                        };
                        count += x.queue_count;
                        info
                    })
                    .collect()
            },
            queue_priorities,
            queue_info,
        };
        // Default / required extensions and features
        this.enable_extension::<ash::khr::synchronization2::Meta>()
            .unwrap();
        this.enable_feature::<vk::PhysicalDeviceSynchronization2Features>(|f| {
            &mut f.synchronization2
        })
        .unwrap();

        this.enable_extension::<ash::khr::timeline_semaphore::Meta>()
            .unwrap();
        this.enable_feature::<vk::PhysicalDeviceTimelineSemaphoreFeatures>(|f| {
            &mut f.timeline_semaphore
        })
        .unwrap();

        
        #[cfg(target_vendor = "apple")]
        {
            // Allow the enumeration of non-conformant implementations
            if this
                .enable_extension::<ash::khr::portability_subset::Meta>()
                .is_ok() {
                    tracing::warn!("Running on a Vulkan portability implementation");
                }
        }
        this
    }
    /// Enables a device extension by type.
    ///
    /// This method enables the specified extension if it's available on the physical device.
    /// If the extension has been promoted to Vulkan core and the instance supports the
    /// required version, the extension is considered automatically enabled.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The extension type implementing [`Extension`]
    ///
    /// # Returns
    ///
    /// `Ok(())` if the extension is available and enabled, or `Err(MissingFeatureError)`
    /// if the extension is not supported by the physical device.
    pub fn enable_extension<T: Extension>(&mut self) -> Result<(), MissingFeatureError>
    where
        T::Device: Send + Sync + 'static,
    {
        if let vk::PromotionStatus::PromotedToCore(promoted_extension) = T::PROMOTION_STATUS {
            let promoted_extension = Version(promoted_extension);
            if self.pdevice.instance().api_version() >= promoted_extension {
                return Ok(());
            }
        }
        if let Some(_v) = self.available_extensions.get(T::NAME) {
            self.enabled_extensions.insert(
                T::NAME,
                Some(|instance, device| {
                    let ext = T::load_device(instance, device);
                    T::promote_device(device, &ext);
                    Box::new(ext)
                }),
            );
            Ok(())
        } else {
            Err(MissingFeatureError::Extension(T::NAME))
        }
    }
    /// Enables a device extension by name.
    ///
    /// This is useful for enabling extensions that don't have a corresponding
    /// Rust type or for extensions that only need to be enabled without
    /// providing additional functionality.
    ///
    /// # Arguments
    ///
    /// * `name` - The extension name as a static C string
    ///
    /// # Returns
    ///
    /// `Ok(())` if the extension is available, or `Err(MissingFeatureError)`
    /// if the extension is not supported.
    pub fn enable_extension_named(
        &mut self,
        name: &'static CStr,
    ) -> Result<(), MissingFeatureError> {
        if let Some(_v) = self.available_extensions.get(name) {
            self.enabled_extensions.insert(name, None);
            Ok(())
        } else {
            Err(MissingFeatureError::Extension(name))
        }
    }

    /// Enables a device feature.
    ///
    /// This method enables the specified feature if the required extension is enabled
    /// or if the feature has been promoted to Vulkan core and the instance supports
    /// the required version.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The feature type implementing [`Feature`]
    ///
    /// # Arguments
    ///
    /// * `selector` - A closure that selects which specific feature flag to enable
    ///
    /// # Returns
    ///
    /// `Ok(())` if the feature is enabled, or `Err(MissingFeatureError)` if the
    /// required extension is not enabled.
    pub fn enable_feature<T: Feature + Default + 'static>(
        &mut self,
        selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Result<(), MissingFeatureError> {
        // Note: the problem here is that the extension wasnt added to enabled_extenion.
        if !self.enabled_extensions.contains_key(T::REQUIRED_DEVICE_EXT) {
            if let vk::PromotionStatus::PromotedToCore(promoted_version) = T::PROMOTION_STATUS {
                if self.pdevice.instance().api_version() < Version(promoted_version) {
                    tracing::warn!(
                        "Feature {:?} requires either Vulkan {} or enabling extension {:?}. Current Vulkan version: {}",
                        std::any::type_name::<T>(),
                        promoted_version,
                        T::REQUIRED_DEVICE_EXT,
                        self.pdevice.instance().api_version()
                    );
                }
            } else {
                tracing::warn!(
                    "Feature {:?} requires enabling extension {:?}",
                    std::any::type_name::<T>(),
                    T::REQUIRED_DEVICE_EXT
                );
            }
        }
        self.features.enable_feature::<T>(selector)
    }
    /// Enables a queue from the specified queue family.
    ///
    /// # Arguments
    ///
    /// * `family_index` - The index of the queue family
    /// * `priority` - The queue priority (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// `Some(DeviceQueueRef)` if a queue is available in the family, or `None`
    /// if all queues in the family are already enabled.
    pub fn enable_queue(&mut self, family_index: u32, priority: f32) -> Option<DeviceQueueRef> {
        let queue_create_info = &mut self.queue_create_info[family_index as usize];
        let queue_family_info = &self.queue_info[family_index as usize];
        if queue_create_info.queue_count >= queue_family_info.queue_count {
            return None;
        }
        let queue_index = queue_create_info.queue_count;
        queue_create_info.queue_count += 1;
        unsafe {
            let p_priority = queue_create_info
                .p_queue_priorities
                .add(queue_index as usize) as *mut _;
            *p_priority = priority;
        }
        Some(DeviceQueueRef {
            pdevice: self.pdevice.clone(),
            family: family_index,
            index: queue_index,
            flags: self.queue_info[family_index as usize].queue_flags,
        })
    }
    /// Enables the least capable queue with the required queue capabilities.
    ///
    /// This method finds the queue family with the fewest capabilities that still
    /// supports the required operations, helping to preserve more capable queues
    /// for operations that need them.
    ///
    /// # Arguments
    ///
    /// * `required_queue_capabilities` - The required queue operation flags
    /// * `priority` - The queue priority (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// `Some(DeviceQueueRef)` if a suitable queue is found and available, or `None`
    /// if no queue family supports the required capabilities or all suitable queues
    /// are already enabled.
    pub fn enable_queue_with_caps(
        &mut self,
        required_queue_capabilities: vk::QueueFlags,
        priority: f32,
    ) -> Option<DeviceQueueRef> {
        let (queue_family_index, _info) = self
            .queue_info
            .iter()
            .zip(&self.queue_create_info)
            .enumerate()
            .filter(|(_, (properties, create_info))| {
                properties.queue_flags.contains(required_queue_capabilities)
                    && create_info.queue_count < properties.queue_count
            })
            .min_by_key(|(_, (properties, _))| properties.queue_flags.as_raw().count_ones())?;
        self.enable_queue(queue_family_index as u32, priority)
    }

    /// Builds the logical device with the current configuration.
    ///
    /// This method creates the Vulkan logical device using all the extensions,
    /// features, and queues that have been enabled. It also starts a background
    /// thread for handling deferred resource cleanup.
    ///
    /// # Returns
    ///
    /// `Ok(Device)` if the device is created successfully, or a Vulkan error
    /// if device creation fails.
    ///
    /// # Errors
    ///
    /// Returns a Vulkan error if:
    /// - The physical device doesn't support the requested features
    /// - The requested extensions are not available
    /// - Device creation fails for any other reason
    pub fn build(self) -> VkResult<Device> {
        let mut features = self.features.finish();

        let extension_names = self
            .enabled_extensions
            .keys()
            .map(|k| k.as_ptr())
            .collect::<Vec<_>>();
        let mut queue_create_info = Vec::from(self.queue_create_info);
        queue_create_info.retain(|x| x.queue_count > 0);
        let create_info = unsafe {
            vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_create_info)
                .enabled_extension_names(&extension_names)
                .extend(&mut features.head)
        };
        let mut device = unsafe {
            self.pdevice
                .instance()
                .create_device(self.pdevice.vk_handle(), &create_info, None)
        }?;
        let extensions: BTreeMap<&'static CStr, Option<Box<dyn Any + Send + Sync>>> = self
            .enabled_extensions
            .into_iter()
            .chain(
                self.pdevice
                    .instance()
                    .device_extensions()
                    .iter()
                    .map(|(&name, a)| (name, Some(*a))),
            )
            .map(|(name, builder)| {
                (
                    name,
                    builder.map(|builder| builder(self.pdevice.instance(), &mut device)),
                )
            })
            .collect();

        let (sender, receiver) = crossbeam_channel::unbounded::<crate::sync::RetiredGPUMutex>();
        let device = Device(Arc::new(DeviceInner {
            physical_device: self.pdevice,
            device,
            extensions,
            features,
            recycler: sender,
        }));

        crate::sync::spawn_recycler_thread(device.clone(), receiver);

        Ok(device)
    }
}
