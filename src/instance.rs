//! Instance creation and management.
//!
//! This module provides the [`Instance`] type and [`InstanceBuilder`] for creating
//! and configuring Vulkan instances.
//!
//! # Overview
//!
//! A Vulkan instance is the connection between your application and the Vulkan
//! loader. It is the first object you create and is used to:
//!
//! - Enumerate physical devices (GPUs)
//! - Enable instance-level extensions and layers
//! - Set application metadata
//!
//! # Example
//!
//! ```
//! # use std::sync::Arc;
//! # use std::borrow::Cow;
//! # use pumicite::{Instance, utils::Version};
//! let entry = Arc::new(unsafe { ash::Entry::load().unwrap() });
//! let mut builder = Instance::builder(entry);
//!
//! // Configure the instance
//! builder.info.api_version = Version::V1_3;
//! builder.info.application_name = Cow::Borrowed(c"My Application");
//!
//! // Enable extensions
//! builder.enable_extension::<ash::ext::debug_utils::Meta>().ok();
//!
//! // Enable validation layers (for debugging)
//! builder.enable_layer(c"VK_LAYER_KHRONOS_validation");
//!
//! let instance = builder.build().unwrap();
//! ```
//!
//! # Extensions and Layers
//!
//! - **Extensions** add new functionality to Vulkan (e.g., debug utils, surface support)
//! - **Layers** intercept Vulkan calls for debugging, validation, or profiling
//!
//! Some instance extensions also provide device-level functionality. These are
//! automatically propagated to devices created from this instance.

use crate::{Extension, MissingFeatureError, device::DeviceMetaBuilder, utils::Version};
use ash::{
    VkResult,
    vk::{self, ExtensionMeta},
};
use std::{
    any::Any,
    borrow::Cow,
    collections::BTreeMap,
    ffi::{CStr, CString, c_char},
    ops::Deref,
    sync::Arc,
};

/// A Vulkan instance wrapper.
///
/// The instance is the connection between your application and the Vulkan loader.
/// It is reference-counted using [`Arc`] for cheap shared access.
///
/// The instance is automatically destroyed when the last reference is dropped.
#[derive(Clone)]
pub struct Instance(Arc<InstanceInner>);
impl PartialEq for Instance {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Instance {}

struct InstanceInner {
    entry: Arc<ash::Entry>,
    instance: ash::Instance,
    extensions: BTreeMap<&'static CStr, Option<Box<dyn Any + Send + Sync>>>,

    /// Some instance extensions have device methods.
    /// All devices created from this instance will have these device extensions.
    device_extensions: BTreeMap<&'static CStr, DeviceMetaBuilder>,
    api_version: Version,
}

/// Function type for building instance extension metadata.
pub type InstanceMetaBuilder = fn(&ash::Entry, &ash::Instance) -> Box<dyn Any + Send + Sync>;

/// Configuration for instance creation.
///
/// This struct contains all the metadata and settings used when creating
/// a Vulkan instance.
pub struct InstanceCreateInfo {
    /// Instance creation flags.
    pub flags: vk::InstanceCreateFlags,
    /// The application name (shown in debugging tools).
    pub application_name: Cow<'static, CStr>,
    /// The application version.
    pub application_version: Version,
    /// The engine name.
    pub engine_name: Cow<'static, CStr>,
    /// The engine version.
    pub engine_version: Version,
    /// The Vulkan API version to use.
    pub api_version: Version,
}

impl Default for InstanceCreateInfo {
    fn default() -> Self {
        Self {
            flags: vk::InstanceCreateFlags::empty(),
            application_name: Cow::Borrowed(c"Unnamed Application"),
            application_version: Default::default(),
            engine_name: Cow::Borrowed(c"Unnamed Engine"),
            engine_version: Default::default(),
            api_version: Version::new(0, 1, 3, 0),
        }
    }
}

impl Instance {
    /// Returns device extensions that will be automatically enabled on devices.
    pub(crate) fn device_extensions(&self) -> &BTreeMap<&'static CStr, DeviceMetaBuilder> {
        &self.0.device_extensions
    }

    /// Creates a new instance builder.
    pub fn builder(entry: Arc<ash::Entry>) -> InstanceBuilder {
        InstanceBuilder::new(entry)
    }

    /// Returns the Vulkan entry point.
    pub fn entry(&self) -> &Arc<ash::Entry> {
        &self.0.entry
    }

    /// Gets a reference to an enabled instance extension.
    ///
    /// # Returns
    ///
    /// Returns `Ok(&T::Instance)` if the extension is enabled, or
    /// `Err(MissingFeatureError)` if not.
    ///
    /// # Panics
    ///
    /// Panics if the extension was enabled but doesn't provide instance methods.
    pub fn get_extension<T: ExtensionMeta>(&self) -> Result<&T::Instance, MissingFeatureError>
    where
        T::Instance: 'static,
    {
        self.0
            .extensions
            .get(&T::NAME)
            .map(|item| {
                item.as_ref()
                    .expect("Instance extension does not have a function table.")
                    .downcast_ref::<T::Instance>()
                    .unwrap()
            })
            .ok_or(MissingFeatureError::Extension(T::NAME))
    }

    /// Gets a reference to an enabled instance extension, panicking if not found.
    ///
    /// # Panics
    ///
    /// Panics if the extension is not enabled.
    pub fn extension<T: ExtensionMeta>(&self) -> &T::Instance
    where
        T::Instance: 'static,
    {
        self.get_extension::<T>().unwrap()
    }
    /// Returns the version of the Vulkan API used when creating the instance.
    pub fn api_version(&self) -> Version {
        self.0.api_version
    }
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.0.instance
    }
}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        tracing::info!(instance = ?self.instance.handle(), "drop instance");
        self.extensions.clear();
        // Safety: Host synchronization rule for vkDestroyInstance:
        // - Host access to instance must be externally synchronized.
        // - Host access to all VkPhysicalDevice objects enumerated from instance must be externally synchronized.
        // We have &mut self and therefore exclusive control on instance.
        // VkPhysicalDevice created from this Instance may not exist at this point,
        // because PhysicalDevice retains an Arc to Instance.
        // If there still exist a copy of PhysicalDevice, the Instance wouldn't be dropped.
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

/// Properties of a Vulkan layer.
///
/// Returned by [`InstanceBuilder::enable_layer`] when a layer is successfully enabled.
#[derive(Clone)]
pub struct LayerProperties {
    /// The Vulkan spec version the layer was written against.
    pub spec_version: Version,
    /// The layer's implementation version.
    pub implementation_version: Version,
    /// A human-readable description of the layer.
    pub description: String,
}

/// A builder for creating Vulkan instances.
///
/// Provides a fluent interface for configuring extensions, layers, and
/// application metadata before instance creation.
pub struct InstanceBuilder {
    entry: Arc<ash::Entry>,
    available_extensions: BTreeMap<CString, Version>,
    enabled_extensions: BTreeMap<&'static CStr, Option<InstanceMetaBuilder>>,
    device_extensions: BTreeMap<&'static CStr, DeviceMetaBuilder>,

    available_layers: BTreeMap<CString, LayerProperties>,
    enabled_layers: Vec<*const c_char>,

    /// Instance creation configuration. Modify this to set application metadata.
    pub info: InstanceCreateInfo,
}
impl Default for InstanceBuilder {
    /// Creates a builder with a default entry point.
    ///
    /// Loads the Vulkan library using the default loader.
    fn default() -> Self {
        let entry = unsafe { ash::Entry::load().unwrap() };
        Self::new(Arc::new(entry))
    }
}
impl InstanceBuilder {
    /// Creates a new instance builder with the given entry point.
    ///
    /// Enumerates available extensions and layers from the Vulkan loader.
    pub fn new(entry: Arc<ash::Entry>) -> Self {
        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .unwrap()
            .into_iter()
            .map(|ext| {
                let str = ext.extension_name_as_c_str().unwrap();
                (str.to_owned(), Version(ext.spec_version))
            })
            .collect::<BTreeMap<CString, Version>>();
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .unwrap()
            .into_iter()
            .map(|layer| {
                let str = layer.layer_name_as_c_str().unwrap();
                (
                    str.to_owned(),
                    LayerProperties {
                        implementation_version: Version(layer.implementation_version),
                        spec_version: Version(layer.spec_version),
                        description: layer
                            .description_as_c_str()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .to_string(),
                    },
                )
            })
            .collect::<BTreeMap<CString, LayerProperties>>();
        #[allow(unused_mut)]
        let mut this = Self {
            entry,
            available_extensions,
            enabled_extensions: BTreeMap::new(),
            device_extensions: BTreeMap::new(),
            available_layers,
            enabled_layers: Vec::new(),
            info: InstanceCreateInfo::default(),
        };
        #[cfg(target_vendor = "apple")]
        {
            // Allow the enumeration of non-conformant implementations
            if this
                .enable_extension::<ash::khr::portability_enumeration::Meta>()
                .is_ok()
            {
                this.info.flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }
        }

        this
    }

    /// Enables an instance extension by type.
    ///
    /// If the extension has been promoted to Vulkan core and the API version
    /// is sufficient, the extension is considered automatically enabled.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the extension is available, `Err(MissingFeatureError)` otherwise.
    pub fn enable_extension<T: Extension>(&mut self) -> Result<(), MissingFeatureError>
    where
        T::Instance: Send + Sync + 'static,
        T::Device: Send + Sync + 'static,
    {
        if let vk::PromotionStatus::PromotedToCore(promoted_extension) = T::PROMOTION_STATUS {
            let promoted_extension = Version(promoted_extension);
            if self.info.api_version >= promoted_extension {
                tracing::info!(
                    "Vulkan extension {:?} enabled unnecessarily; it was promoted to Vulkan {:?} core",
                    T::NAME,
                    promoted_extension
                );
                return Ok(());
            }
        }
        if let Some(_v) = self.available_extensions.get(T::NAME) {
            self.enabled_extensions.insert(
                T::NAME,
                Some(|entry, instance| {
                    let ext = T::load_instance(entry, instance);
                    Box::new(ext)
                }),
            );
            self.device_extensions.insert(T::NAME, |instance, device| {
                let ext = T::load_device(instance, device);
                T::promote_device(device, &ext);
                Box::new(ext)
            });
            Ok(())
        } else {
            Err(MissingFeatureError::Extension(T::NAME))
        }
    }

    /// Enables an instance extension by name.
    ///
    /// Use this for extensions that don't have a Rust type or don't need
    /// function pointer loading.
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

    /// Enables a Vulkan layer.
    ///
    /// Layers intercept Vulkan calls for debugging, validation, or profiling.
    /// Common layers include `VK_LAYER_KHRONOS_validation` for API validation.
    ///
    /// When a layer is enabled, any additional extensions it provides become
    /// available for enabling.
    ///
    /// # Returns
    ///
    /// `Some(LayerProperties)` if the layer is available, `None` otherwise.
    pub fn enable_layer(&mut self, layer: &'static CStr) -> Option<LayerProperties> {
        if let Some(v) = self.available_layers.get(layer) {
            let v = v.clone();
            self.enabled_layers.push(layer.as_ptr());

            let additional_instance_extensions = unsafe {
                self.entry
                    .enumerate_instance_extension_properties(Some(layer))
                    .unwrap()
            };
            self.available_extensions
                .extend(additional_instance_extensions.into_iter().map(|a| {
                    (
                        a.extension_name_as_c_str().unwrap().to_owned(),
                        Version(a.spec_version),
                    )
                }));

            Some(v)
        } else {
            None
        }
    }

    /// Builds the Vulkan instance with the current configuration.
    ///
    /// # Returns
    ///
    /// `Ok(Instance)` on success, or a Vulkan error if instance creation fails.
    pub fn build(self) -> VkResult<Instance> {
        let application_info = vk::ApplicationInfo {
            p_application_name: self.info.application_name.as_ptr(),
            application_version: self.info.application_version.0,
            p_engine_name: self.info.engine_name.as_ptr(),
            engine_version: self.info.engine_version.0,
            api_version: self.info.api_version.0,
            ..Default::default()
        };

        let enabled_extension_names = self
            .enabled_extensions
            .keys()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &application_info,
            enabled_layer_count: self.enabled_layers.len() as u32,
            pp_enabled_layer_names: self.enabled_layers.as_ptr(),
            enabled_extension_count: enabled_extension_names.len() as u32,
            pp_enabled_extension_names: enabled_extension_names.as_ptr(),
            flags: self.info.flags,
            ..Default::default()
        };
        // Safety: No Host synchronization rules for vkCreateInstance.
        let instance = unsafe { self.entry.create_instance(&create_info, None)? };
        let extensions: BTreeMap<&'static CStr, _> = self
            .enabled_extensions
            .into_iter()
            .map(|(name, builder)| {
                let item = builder.map(|builder| builder(&self.entry, &instance));
                (name, item)
            })
            .collect();
        Ok(Instance(Arc::new(InstanceInner {
            entry: self.entry,
            instance,
            extensions,
            api_version: self.info.api_version,
            device_extensions: self.device_extensions,
        })))
    }
}
unsafe impl Send for InstanceBuilder {}
unsafe impl Sync for InstanceBuilder {}
