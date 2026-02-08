use bevy_app::prelude::*;
use bevy_asset::AssetApp;
use bevy_ecs::prelude::*;
use pumicite::{
    ash::{
        khr,
        vk::{self, ExtensionMeta},
    },
    bevy::PipelineCache,
    bevy::PipelineLayout,
    physical_device::PhysicalDevice,
};
use std::ffi::CStr;

use crate::{
    DescriptorHeap, shader::ShaderModule, staging::AsyncTransfer, swapchain::SwapchainSet,
};

use super::pass::SubmissionSetsPass;
use super::queue::QueueConfiguration;
use pumicite::{
    Device, Extension, Instance, MissingFeatureError,
    device::DeviceBuilder,
    instance::{InstanceBuilder, LayerProperties},
    physical_device::Feature,
};

/// The default system set for rendering operations.
///
/// Systems in this set are submitted to the [`RenderQueue`](crate::queue::RenderQueue),
/// which supports [`vk::QueueFlags::GRAPHICS`] and most likely [`vk::QueueFlags::COMPUTE`] on
/// desktop GPUs.
///
/// # Ordering
///
/// - Runs **within** [`SwapchainSet`](crate::swapchain::SwapchainSet) (after acquire and before present)
/// - Runs **after** [`DefaultTransferSet`]
///
/// # Usage
///
/// Add rendering systems to this set to ensure they have access to the current
/// swapchain image and run in the correct order relative to swapchain image acquire and presentation.
///
/// ```no_run
///# use bevy::prelude::*;
///# fn my_render_system(){}
/// use bevy_pumicite::DefaultRenderSet;
/// App::new().add_systems(PostUpdate, my_render_system.in_set(DefaultRenderSet));
/// ```
#[derive(Debug, SystemSet, Hash, PartialEq, Eq, Clone, Copy)]
pub struct DefaultRenderSet;

/// The default system set for compute operations.
///
/// Systems in this set are submitted to the [`ComputeQueue`](crate::queue::ComputeQueue),
/// which supports [`vk::QueueFlags::COMPUTE`].
///
/// # Ordering
///
/// - Runs **after** [`DefaultTransferSet`]
#[derive(Debug, SystemSet, Hash, PartialEq, Eq, Clone, Copy)]
pub struct DefaultComputeSet;

/// The default system set for data transfer operations.
///
/// Systems in this set are submitted to the [`TransferQueue`](crate::queue::TransferQueue),
/// which supports [`vk::QueueFlags::TRANSFER`].
///
/// # Ordering
///
/// - Runs **before** [`DefaultRenderSet`] and [`DefaultComputeSet`]
#[derive(Debug, SystemSet, Hash, PartialEq, Eq, Clone, Copy)]
pub struct DefaultTransferSet;

/// Strategy for selecting a Vulkan physical device.
///
/// When multiple GPUs are available, this enum determines which one to use.
#[derive(Default)]
pub enum PhysicalDeviceSearchStrategy {
    /// Prefer discrete GPUs for maximum performance.
    ///
    /// This is the default strategy.
    #[default]
    Performance,

    /// Prefer integrated GPUs for lower power consumption.
    LowPower,

    /// Select a specific GPU by its index in the enumerated device list.
    ///
    /// Use this when you know exactly which GPU to use. The index corresponds
    /// to the order returned by `vkEnumeratePhysicalDevices`.
    Index(usize),

    /// Custom selection logic via a callback function.
    ///
    /// The callback receives a slice of all available physical devices and
    /// returns the index of the device to use.
    Callback(Box<dyn Fn(&[PhysicalDevice]) -> usize>),
}

/// Core plugin that creates the Vulkan instance and logical device.
///
/// This is the central plugin for Pumicite integration. It handles:
/// - Vulkan instance creation
/// - Physical device selection based on the configured strategy
/// - Logical device creation with requested extensions and features
/// - Default queue setup (render, transfer, compute, async compute)
/// - Asset loader registration for shaders and textures
///
/// # Plugin Ordering
///
/// **Critical**: Plugin ordering matters!
///
/// - **Instance plugins** must be added **before** `PumicitePlugin`. These plugins
///   configure the Vulkan instance (e.g., [`SurfacePlugin`](crate::SurfacePlugin),
///   [`DebugUtilsPlugin`](crate::DebugUtilsPlugin)).
///
/// - **Device plugins** must be added **after** `PumicitePlugin`. These plugins
///   configure the Vulkan device (e.g., RTX plugins, custom extensions).
///
/// # Default Queues
///
/// The plugin automatically creates four queue types:
/// - [`RenderQueue`](crate::queue::RenderQueue) - Graphics operations (priority 1.0)
/// - [`TransferQueue`](crate::queue::TransferQueue) - Data transfers (priority 0.1)
/// - [`ComputeQueue`](crate::queue::ComputeQueue) - Synchronous compute (priority 1.0)
/// - [`AsyncComputeQueue`](crate::queue::AsyncComputeQueue) - Background compute (priority 0.1)
///
/// If dedicated queues aren't available, they alias compatible existing queues.
///
/// # Resources Created
///
/// After `finish`, the following resources are available:
/// - [`Instance`](pumicite::Instance) - Vulkan instance
/// - [`PhysicalDevice`](pumicite::physical_device::PhysicalDevice) - Selected GPU
/// - [`Device`](pumicite::Device) - Logical device
/// - [`Allocator`](pumicite::Allocator) - GPU memory allocator
/// - [`PipelineCache`](pumicite::bevy::PipelineCache) - Pipeline caching
pub struct PumicitePlugin {
    /// Strategy for selecting which physical device (GPU) to use.
    pub physical_device: PhysicalDeviceSearchStrategy,
    /// If bindless was enabled, the default capacity for the resource heap.
    pub resource_heap_size: u32,
    /// If bindless was enabled, the default capacity for the sampler heap.
    pub sampler_heap_size: u32,
}
impl Default for PumicitePlugin {
    fn default() -> Self {
        Self {
            physical_device: Default::default(),
            resource_heap_size: 1024,
            sampler_heap_size: 128,
        }
    }
}
unsafe impl Send for PumicitePlugin {}
unsafe impl Sync for PumicitePlugin {}

impl Plugin for PumicitePlugin {
    fn build(&self, app: &mut App) {
        app.world_mut().init_resource::<InstanceBuilder>();
        let instance = app
            .world_mut()
            .remove_resource::<InstanceBuilder>()
            .unwrap()
            .build()
            .unwrap();
        let physical_devices = instance
            .enumerate_physical_devices()
            .unwrap()
            .collect::<Vec<_>>();
        let physical_device_index = match &self.physical_device {
            PhysicalDeviceSearchStrategy::LowPower | PhysicalDeviceSearchStrategy::Performance => {
                physical_devices
                    .iter()
                    .enumerate()
                    .rev()
                    .map(|(i, x)| {
                        let mut score = 0;

                        #[cfg(target_vendor = "apple")]
                        {
                            if x.properties()
                                .get::<vk::PhysicalDeviceDriverProperties>()
                                .driver_id
                                == vk::DriverId::from_raw(28)
                            {
                                // is KosmicKrisp
                                score -= 1000;
                            }
                        }
                        match &self.physical_device {
                            PhysicalDeviceSearchStrategy::Performance
                                if x.properties().device_type
                                    == vk::PhysicalDeviceType::DISCRETE_GPU =>
                            {
                                score += 100;
                            }
                            PhysicalDeviceSearchStrategy::LowPower
                                if x.properties().device_type
                                    == vk::PhysicalDeviceType::INTEGRATED_GPU =>
                            {
                                score += 100;
                            }
                            _ => {}
                        }
                        (i, score)
                    })
                    .max_by_key(|x| x.1)
                    .map(|x| x.0)
                    .unwrap_or_default()
            }
            PhysicalDeviceSearchStrategy::Index(i) => *i,
            PhysicalDeviceSearchStrategy::Callback(callback) => callback(&physical_devices),
        };
        let physical_device = physical_devices
            .into_iter()
            .nth(physical_device_index)
            .expect("Physical device not found");
        tracing::info!(
            "Using {:?} {:?}",
            physical_device.properties().device_type,
            physical_device.properties().device_name(),
        );
        let driver_properties = physical_device
            .properties()
            .get::<vk::PhysicalDeviceDriverProperties>();
        tracing::info!(
            "Driver {:?} ({:?})",
            driver_properties
                .driver_name_as_c_str()
                .unwrap_or(c"unknown"),
            driver_properties
                .driver_info_as_c_str()
                .unwrap_or(c"unknown"),
        );

        app.insert_resource(instance)
            .insert_resource(physical_device.clone())
            .insert_resource(Device::builder(physical_device));

        app.init_device_queue_with_caps::<super::queue::RenderQueue>(vk::QueueFlags::GRAPHICS, 1.0)
            .unwrap();
        app.init_device_queue_with_caps::<super::queue::TransferQueue>(
            vk::QueueFlags::TRANSFER,
            0.1,
        )
        .unwrap();
        app.init_device_queue_with_caps::<super::queue::ComputeQueue>(vk::QueueFlags::COMPUTE, 1.0)
            .unwrap();
        app.init_device_queue_with_caps::<super::queue::AsyncComputeQueue>(
            vk::QueueFlags::COMPUTE,
            0.1,
        )
        .unwrap();

        // Add build pass
        app.get_schedule_mut(PostUpdate)
            .as_mut()
            .unwrap()
            .add_build_pass(super::pass::SubmissionSetsPass::default());

        app.add_submission_set::<super::queue::RenderQueue>(DefaultRenderSet);
        app.add_submission_set::<super::queue::TransferQueue>(DefaultTransferSet);
        app.add_submission_set::<super::queue::ComputeQueue>(DefaultComputeSet);
        app.configure_sets(
            PostUpdate,
            (
                DefaultRenderSet
                    .in_set(SwapchainSet)
                    .after(DefaultTransferSet),
                DefaultComputeSet.after(DefaultTransferSet),
            ),
        );

        // Optional extensions
        app.add_device_extension::<khr::deferred_host_operations::Meta>()
            .ok();

        app.init_asset::<ShaderModule>()
            .init_asset::<PipelineLayout>()
            .init_asset::<crate::shader::ComputePipeline>()
            .init_asset::<crate::shader::GraphicsPipeline>()
            .init_asset::<crate::loader::TextureAsset>();

        app.add_plugins(super::staging::StagingBeltPlugin::default());

        app.preregister_asset_loader::<crate::shader::ShaderLoader>(&["spv"])
            .preregister_asset_loader::<crate::shader::PipelineLayoutLoader>(&["playout.ron"])
            .preregister_asset_loader::<crate::shader::DescriptorSetLayoutLoader>(&["desc.ron"])
            .preregister_asset_loader::<crate::shader::ComputePipelineLoader>(&[
                "comp.pipeline.ron",
            ])
            .preregister_asset_loader::<crate::shader::GraphicsPipelineLoader>(&[
                "gfx.pipeline.ron",
            ])
            .preregister_asset_loader::<crate::loader::DdsLoader>(&["dds"])
            .preregister_asset_loader::<crate::loader::ImageLoader>(&["jpg", "png"])
            .preregister_asset_loader::<crate::loader::KtxLoader>(&["ktx"]);

        //app.register_type::<bevy::image::Image>()
        //    .init_asset::<bevy::image::Image>()
        //    .register_asset_reflect::<bevy::image::Image>();
    }
    fn finish(&self, app: &mut App) {
        let device_builder = app.world_mut().remove_resource::<DeviceBuilder>().unwrap();
        let bindless_enabled = device_builder.bindless_enabled();
        let device = device_builder.build().unwrap();
        app.world_mut().insert_resource(device.clone());
        QueueConfiguration::init_queues(app.world_mut());

        if bindless_enabled {
            let heap = DescriptorHeap::new(
                device.clone(),
                self.resource_heap_size,
                self.sampler_heap_size,
            )
            .unwrap();
            app.world_mut().insert_resource(heap);
        }

        app.world_mut()
            .insert_resource(pumicite::Allocator::new(device).unwrap());
        app.init_resource::<PipelineCache>();

        app.init_resource::<AsyncTransfer>();

        //app.world_mut()
        //    .init_resource::<crate::task::AsyncTaskPool>();
        //app.world_mut()
        //    .init_resource::<crate::DeferredOperationTaskPool>();
        //app.init_asset_loader::<crate::shader::loader::SpirvLoader>();
    }

    fn cleanup(&self, app: &mut App) {
        // Initialize pipeline loaders in cleanup to ensure that the bindless plugin is available
        app.init_asset_loader::<crate::shader::ShaderLoader>()
            .init_asset_loader::<crate::shader::PipelineLayoutLoader>()
            .init_asset_loader::<crate::shader::DescriptorSetLayoutLoader>()
            .init_asset_loader::<crate::shader::ComputePipelineLoader>()
            .init_asset_loader::<crate::shader::GraphicsPipelineLoader>()
            .init_asset_loader::<crate::loader::KtxLoader>()
            .init_asset_loader::<crate::loader::DdsLoader>()
            .init_asset_loader::<crate::loader::ImageLoader>();
    }
}

/// Extension trait for [`App`] that provides Vulkan configuration methods.
///
/// This trait adds methods to configure the Vulkan instance, device, extensions,
/// features, and queues during plugin setup.
///
/// # Example
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_pumicite::PumiciteApp;
/// use pumicite::ash::vk;
///
/// fn build(app: &mut App) {
///     // Enable a Vulkan feature
///     app.enable_feature::<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>(|f| {
///         &mut f.ray_tracing_pipeline
///     }).unwrap();
///
///     // Add a device extension
///     app.add_device_extension::<pumicite::ash::ext::conditional_rendering::Meta>().unwrap();
/// }
/// ```
pub trait PumiciteApp {
    /// TODO: remove in favor of VK_EXT_descriptor_heap
    fn enable_bindless(&mut self) -> Result<(), MissingFeatureError>;

    /// Adds a Vulkan device extension.
    ///
    /// Must be called in [`Plugin::build`] in a plugin inserted **after** [`PumicitePlugin`].
    ///
    /// # Errors
    ///
    /// Returns [`MissingFeatureError`] if the extension isn't supported.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use bevy_pumicite::PumiciteApp;
    /// use pumicite::ash::vk;
    ///
    /// fn build(app: &mut App) {
    ///     // Add a device extension
    ///     app.add_device_extension::<pumicite::ash::ext::conditional_rendering::Meta>().unwrap();
    /// }
    /// ```
    fn add_device_extension<T: ExtensionMeta>(&mut self) -> Result<(), MissingFeatureError>
    where
        T::Device: Send + Sync + 'static;

    /// Adds a Vulkan instance extension using its type metadata.
    ///
    /// Must be called in [`Plugin::build`] in a plugin inserted **before** [`PumicitePlugin`].
    ///
    /// # Errors
    ///
    /// Returns [`MissingFeatureError`] if the extension isn't supported.
    fn add_instance_extension<T: ExtensionMeta>(&mut self) -> Result<(), MissingFeatureError>
    where
        T::Instance: Send + Sync + 'static,
        T::Device: Send + Sync + 'static;

    /// Adds a Vulkan device extension by name.
    ///
    /// Must be called in [`Plugin::build`] in a plugin inserted **after** [`PumicitePlugin`].
    /// [`PumicitePlugin`].
    ///
    /// # Errors
    ///
    /// Returns [`MissingFeatureError`] if the extension isn't supported.
    fn add_device_extension_named(
        &mut self,
        extension: &'static CStr,
    ) -> Result<(), MissingFeatureError>;

    /// Adds a Vulkan instance extension by name.
    ///
    /// Must be called in [`Plugin::build`] in a plugin inserted **before** [`PumicitePlugin`].
    ///
    /// # Errors
    ///
    /// Returns [`MissingFeatureError`] if the extension isn't supported.
    fn add_instance_extension_named(
        &mut self,
        extension: &'static CStr,
    ) -> Result<(), MissingFeatureError>;

    /// Enables a Vulkan validation layer.
    ///
    /// Must be called in [`Plugin::build`] in a plugin inserted **before** [`PumicitePlugin`].
    ///
    /// # Returns
    ///
    /// Returns `Some(LayerProperties)` if the layer was enabled, `None` if unavailable.
    fn add_instance_layer(&mut self, layer: &'static CStr) -> Option<LayerProperties>;

    /// Enables a Vulkan physical device feature.
    ///
    /// The `selector` closure receives the feature struct and returns a mutable
    /// reference to the specific feature flag to enable.
    ///
    /// Must be called in [`Plugin::build`] in a plugin inserted **after** [`PumicitePlugin`].
    ///
    /// # Errors
    ///
    /// Returns [`MissingFeatureError`] if the feature isn't supported.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use bevy_pumicite::PumiciteApp;
    /// use pumicite::ash::vk;
    ///
    /// fn build(app: &mut App) {
    ///     // Enable a Vulkan feature
    ///     app.enable_feature::<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>(|f| {
    ///         &mut f.ray_tracing_pipeline
    ///     }).unwrap();
    /// }
    /// ```
    fn enable_feature<T: Feature + Default + 'static>(
        &mut self,
        selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Result<(), MissingFeatureError>;

    /// Creates or aliases a device queue with the specified capabilities.
    ///
    /// Attempts to find a queue family with `required_queue_capabilities` and create
    /// a new queue. If no queue slots are available, tries to alias an existing
    /// compatible queue.
    ///
    /// The type parameter `T` serves as a marker to identify this queue in systems
    /// via [`Queue<T>`](crate::Queue).
    ///
    /// # Queue Aliasing
    ///
    /// Vulkan implementations are allowed to expose as few as 1 queue and 1 queue family.
    /// When dedicated queues aren't available, multiple "logical"
    /// queue types may alias the same underlying Vulkan queue. This is fine, but submissions
    /// made to the same queue won't be able to run in parallel. Systems requesting access
    /// to the [`Queue<T>`](crate::Queue) system param will be scheduled in
    /// a way that avoids concurrent access to the underlying queue.
    ///
    /// # Parameters
    ///
    /// - `required_queue_capabilities`: Required queue flags (e.g., `GRAPHICS`, `COMPUTE`)
    /// - `priority`: Queue priority from 0.0 (lowest) to 1.0 (highest)
    ///
    /// # Errors
    ///
    /// Returns an error if no compatible queue exists.
    fn init_device_queue_with_caps<T: 'static>(
        &mut self,
        required_queue_capabilities: vk::QueueFlags,
        priority: f32,
    ) -> Result<(), QueueNotFoundError>;

    /// Registers a system set as a render set bound to a specific queue.
    ///
    /// Systems in a render set share command encoding state and are submitted
    /// to the GPU together in a single `vkQueueSubmit` call. There is a one-to-one
    /// mapping between render sets and `vkQueueSubmit` calls.
    ///
    /// # Command Encoding
    ///
    /// All systems in the render set:
    /// - Share a single command pool
    /// - Execute serially (commands are recorded in system order)
    /// - Use [`RenderState`](crate::RenderState) to record commands
    ///
    /// # Type Parameters
    ///
    /// - `Q`: Queue marker type (e.g., [`RenderQueue`](crate::queue::RenderQueue), [`ComputeQueue`](crate::queue::ComputeQueue) )
    fn add_submission_set<Q: 'static>(&mut self, set: impl SystemSet + Copy) -> &mut Self;
}

fn get_device_builder(app: &mut App) -> Mut<'_, DeviceBuilder> {
    if app.world().get_resource::<Device>().is_some() {
        panic!("Device extensions and queues may not be enabled after the device was created.")
    }
    let Some(device_builder) = app.world_mut().get_resource_mut::<DeviceBuilder>() else {
        panic!(
            "Device extensions and queues may only be added after the instance was created. Add PumicitePlugin before all device plugins."
        )
    };
    device_builder
}
fn get_instance_builder(app: &mut App) -> Mut<'_, InstanceBuilder> {
    if app.world().get_resource::<Instance>().is_some() {
        panic!(
            "Instance extensions may only be added before the instance was created. Add PumicitePlugin after all instance plugins."
        )
    }
    app.world_mut().get_resource_or_init::<InstanceBuilder>()
}

/// Error returned when no compatible queue could be found or aliased.
#[derive(Debug)]
pub struct QueueNotFoundError;

impl PumiciteApp for App {
    fn enable_bindless(&mut self) -> Result<(), MissingFeatureError> {
        let mut device_builder = get_device_builder(self);

        device_builder.enable_bindless()
    }
    fn add_device_extension<T: Extension>(&mut self) -> Result<(), MissingFeatureError>
    where
        T::Device: Send + Sync + 'static,
    {
        let mut device_builder = get_device_builder(self);

        device_builder.enable_extension::<T>()
    }
    fn add_device_extension_named(
        &mut self,
        extension: &'static CStr,
    ) -> Result<(), MissingFeatureError> {
        let mut device_builder = get_device_builder(self);

        device_builder.enable_extension_named(extension)
    }

    /// Enable the least capable queue with the required queue capabilities.
    fn init_device_queue_with_caps<T: 'static>(
        &mut self,
        required_queue_capabilities: vk::QueueFlags,
        priority: f32,
    ) -> Result<(), QueueNotFoundError> {
        let name = std::any::type_name::<T>()
            .split("::")
            .last()
            .unwrap_or("??");
        let mut device_builder = get_device_builder(self);
        if let Some(queue_ref) =
            device_builder.enable_queue_with_caps(required_queue_capabilities, priority)
        {
            // Queue created
            let component_id = self.world_mut().register_component_with_descriptor(
                bevy_ecs::component::ComponentDescriptor::new_resource::<crate::queue::SharedQueue>(
                ),
            );
            tracing::info!(
                "Device queue {} using queue family {}",
                name,
                queue_ref.family_index()
            );
            self.world_mut()
                .get_resource_or_init::<QueueConfiguration>()
                .register_queue::<T>(component_id, queue_ref, priority, name);
        } else {
            let mut queue_config = self
                .world_mut()
                .get_resource_or_init::<QueueConfiguration>();
            // Try to alias existing queue
            let aliased_queue = queue_config
                .alias_queue::<T>(required_queue_capabilities, priority)
                .ok_or(QueueNotFoundError)?;
            tracing::info!(
                "Device queue {} aliasing existing queue {}",
                name,
                aliased_queue.name
            );
        }

        Ok(())
    }

    fn add_instance_extension<T: Extension>(&mut self) -> Result<(), MissingFeatureError>
    where
        T::Instance: Send + Sync + 'static,
        T::Device: Send + Sync + 'static,
    {
        let mut builder = get_instance_builder(self);
        builder.enable_extension::<T>()
    }

    fn add_instance_extension_named(
        &mut self,
        extension: &'static CStr,
    ) -> Result<(), MissingFeatureError> {
        let mut builder = get_instance_builder(self);
        builder.enable_extension_named(extension)
    }
    fn add_instance_layer(&mut self, layer: &'static CStr) -> Option<LayerProperties> {
        let mut builder = get_instance_builder(self);
        builder.enable_layer(layer)
    }
    fn enable_feature<T: Feature + Default + 'static>(
        &mut self,
        selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Result<(), MissingFeatureError> {
        let mut device_builder = get_device_builder(self);
        device_builder.enable_feature::<T>(selector)
    }

    fn add_submission_set<Q: 'static>(&mut self, set: impl SystemSet + Copy) -> &mut Self {
        let queue_config = self.world().resource::<QueueConfiguration>();
        let component_id = queue_config
            .component_id_of_queue::<Q>()
            .expect("Please register this queue first");

        let schedule = self.get_schedule_mut(PostUpdate).unwrap();
        let build_pass = schedule.get_build_pass_mut::<SubmissionSetsPass>().unwrap();
        build_pass
            .submission_sets_to_queue
            .insert(set.intern(), component_id);
        self
    }
}
