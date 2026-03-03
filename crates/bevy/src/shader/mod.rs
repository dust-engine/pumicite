//! Shader and pipeline asset loaders.
//!
//! ** CRITICAL **: This module is due for a major overhaul due to VK_EXT_descriptor_heap.
//!
//! This module provides Bevy asset loaders for Vulkan shaders and pipeline configurations.
//! Pipelines are defined in RON format files and reference compiled SPIR-V shader modules.
//!
//! # Supported Asset Types
//!
//! | Extension | Asset Type | Description |
//! |-----------|------------|-------------|
//! | `.spv` | [`ShaderModule`] | Compiled SPIR-V shader bytecode |
//! | `.comp.pipeline.ron` | [`ComputePipeline`](compute::ComputePipeline) | Compute pipeline configuration |
//! | `.gfx.pipeline.ron` | [`GraphicsPipeline`](graphics::GraphicsPipeline) | Graphics pipeline configuration |
//! | `.rtx.pipeline.ron` | [`RayTracingPipelineLibrary`] | Ray tracing pipeline library |
//! | `.playout.ron` | `PipelineLayout` | Pipeline layout definition |
//!
//! # Pipeline Configuration Format
//!
//! Pipelines are defined in RON (Rusty Object Notation) files. Example compute pipeline:
//!
//! ```ron
//! ComputePipeline(
//!     shader: Shader(
//!         path: "shaders/my_compute.spv",
//!         entry_point: "main",
//!     ),
//!     layout: Bindless(push_constant_size: 128),
//! )
//! ```
//!
//! # Layout Options
//!
//! Pipeline layouts can be specified in three ways:
//!
//! - **Inline**: Define descriptor sets and push constants directly
//! - **Path**: Reference a separate `.playout.ron` file
//! - **Bindless**: Use the global bindless descriptor set layout
//!
//! # Caching
//!
//! Shader modules and pipeline layouts are cached by asset path, avoiding duplicate
//! GPU resources when the same shader is used by multiple pipelines.

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    ffi::CString,
    sync::{Arc, Weak},
};

use bevy_asset::{Asset, AssetLoader, AssetPath, LoadDirectError, LoadedAsset};
use bevy_ecs::world::FromWorld;
use bevy_reflect::TypePath;
use pumicite::{
    Device, HasDevice,
    ash::{self, VkResult, vk},
    pipeline::{Pipeline, PipelineCache, ShaderEntry},
    rtx::{RayTracingPipelineLibraryCreateInfo, SbtLayout},
};
use thiserror::Error;

use crate::DescriptorHeap;

pub mod compute;
pub mod graphics;
pub mod ron_types;

/// Error type for shader loading failures.
#[derive(Error, Debug)]
pub enum ShaderLoadError {
    /// Vulkan returned an error during shader module creation.
    #[error("Vulkan Pipeline creation error")]
    VulkanError(#[from] vk::Result),
    /// Failed to read the shader file.
    #[error("IO error")]
    IoError(#[from] std::io::Error),
}

/// Asset loader for SPIR-V shader modules (`.spv` files).
///
/// Loads compiled SPIR-V bytecode and creates Vulkan shader modules.
/// Modules are cached by path to avoid creating duplicates.
pub struct ShaderLoader {
    device: Device,
    cache: async_lock::Mutex<HashMap<AssetPath<'static>, Weak<pumicite::pipeline::ShaderModule>>>,
}
impl FromWorld for ShaderLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            device: world.resource::<Device>().clone(),
            cache: async_lock::Mutex::new(Default::default()),
        }
    }
}

/// A loaded SPIR-V shader module asset.
///
/// Wraps a Vulkan shader module with reference counting for safe sharing
/// across multiple pipelines.
#[derive(Clone, Asset, TypePath)]
pub struct ShaderModule(Arc<pumicite::pipeline::ShaderModule>);

impl AssetLoader for ShaderLoader {
    type Asset = ShaderModule;
    type Settings = ();
    type Error = ShaderLoadError;

    async fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<ShaderModule, Self::Error> {
        let mut lock = self.cache.lock().await;
        if let Some(cached) = lock.get(load_context.asset_path()).and_then(Weak::upgrade) {
            return Ok(ShaderModule(cached));
        };

        let mut code = Vec::new();
        reader.read_to_end(&mut code).await?;
        let item = pumicite::pipeline::ShaderModule::new(self.device.clone(), &code)?;
        let item = Arc::new(item);
        lock.insert(load_context.asset_path().clone(), Arc::downgrade(&item));
        Ok(ShaderModule(item))
    }

    fn extensions(&self) -> &[&str] {
        &["spv"]
    }
}

/// Error type for pipeline loading failures.
#[derive(Error, Debug)]
pub enum PipelineLoaderError {
    /// Vulkan returned an error during pipeline creation.
    #[error("Vulkan Pipeline creation error")]
    VulkanError(#[from] vk::Result),
    /// Failed to read the pipeline configuration file.
    #[error("Vulkan Pipeline creation error")]
    IoError(#[from] std::io::Error),
    /// Failed to load a dependency (shader, layout, etc.).
    #[error("Asset load error: {0}")]
    LoadError(#[from] LoadDirectError),
    /// The RON configuration file has syntax errors.
    #[error("Pipeline config deserialization error:\n{0}")]
    RonError(#[from] ron::de::SpannedError),
    /// The pipeline configuration is invalid.
    #[error("Misconfigured pipeline: {0}")]
    PipelineError(&'static str),
    /// Tried to use bindless layout but bindless wasn't enabled.
    #[error("Bindless plugin not initialized.")]
    BindlessPluginNeededError,
}

/// Asset loader for ray tracing pipeline libraries (`.rtx.pipeline.ron` files).
///
/// Creates pipeline libraries that can be linked into complete ray tracing pipelines
/// via [`RtxPipelineManager`](crate::rtx::RtxPipelineManager).
pub struct RayTracingPipelineLoader {
    pipeline_cache: Arc<PipelineCache>,
    heap: Option<DescriptorHeap>,
    /// Whether to use real VK_KHR_pipeline_library. False when the extension is
    /// unavailable or on drivers known to have bugs (AMD proprietary).
    use_pipeline_library: bool,
}
impl FromWorld for RayTracingPipelineLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        let device = world.resource::<Device>();
        let has_pipeline_library_extension =
            device.has_extension_named(ash::khr::pipeline_library::NAME);
        let driver_properties = device
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceDriverProperties>();
        let has_buggy_driver = driver_properties.driver_id == vk::DriverId::AMD_PROPRIETARY;
        if !has_pipeline_library_extension {
            tracing::warn!(
                "Ray tracing pipelines are using emulated pipeline libraries because VK_KHR_pipeline_library is missing. Pipeline linking might be slower."
            );
        } else if has_buggy_driver {
            tracing::warn!(
                "Ray tracing pipelines are using emulated pipeline libraries because of known bugs in {:?}. Pipeline linking might be slower.",
                driver_properties.driver_name_as_c_str().unwrap()
            );
        }
        Self {
            pipeline_cache: Arc::new(PipelineCache::null(device.clone())),
            heap: world.get_resource().cloned(),
            use_pipeline_library: has_pipeline_library_extension && !has_buggy_driver,
        }
    }
}

/// A ray tracing pipeline library asset.
///
/// Pipeline libraries are building blocks for ray tracing pipelines. They contain
/// shader groups (raygen, miss, hitgroups, callable) that can be linked together
/// to form complete pipelines.
///
/// # Usage
///
/// Typically used with [`RtxPipelineManager`](crate::rtx::RtxPipelineManager):
///
/// ```ignore
/// let base_lib = asset_server.load("shaders/base.rtx.pipeline.ron");
/// let pipeline = manager.add_pipeline(base_lib);
/// ```
#[derive(Clone, Asset, TypePath)]
pub struct RayTracingPipelineLibrary {
    inner: RayTracingPipelineLibraryImpl,
    /// Maximum ray recursion depth for this library's shaders.
    pub max_ray_recursion_depth: u32,
    /// Maximum ray payload size in bytes.
    pub max_ray_payload_size: u32,
    /// Maximum hit attribute size in bytes.
    pub max_hit_attribute_size: u32,
    /// Whether dynamic stack size is used.
    pub dynamic_stack_size: bool,
    /// Shader binding table layout information.
    pub sbt_layout: SbtLayout,
}

#[derive(Clone)]
enum RayTracingPipelineLibraryImpl {
    /// A real Vulkan pipeline library created with VK_KHR_pipeline_library.
    Library(Arc<Pipeline>),
    /// Emulated pipeline library for drivers that don't support VK_KHR_pipeline_library
    /// (or where it's buggy, e.g. AMD proprietary). Stores the shader data to be inlined
    /// into the final monolithic pipeline at link time.
    Monolithic {
        flags: vk::PipelineCreateFlags,
        layout: Arc<pumicite::pipeline::PipelineLayout>,
        shaders: Vec<ShaderEntry<'static>>,
        groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR<'static>>,
    },
}

impl RayTracingPipelineLibrary {
    /// Returns the inner pipeline, if this was created as a real pipeline library.
    /// Panics if called on an monolithic (emulated) library.
    pub fn pipeline(&self) -> &Arc<Pipeline> {
        match &self.inner {
            RayTracingPipelineLibraryImpl::Library(p) => p,
            RayTracingPipelineLibraryImpl::Monolithic { .. } => {
                panic!("Cannot get pipeline from an inline (emulated) pipeline library")
            }
        }
    }

    /// Returns true if this is an monolithic (emulated) pipeline library.
    pub fn is_monolithic(&self) -> bool {
        matches!(self.inner, RayTracingPipelineLibraryImpl::Monolithic { .. })
    }

    /// Returns the inline data (flags, layout, shaders, groups) for merging into a
    /// monolithic pipeline. Panics if called on a real pipeline library.
    pub fn inline_data(
        &self,
    ) -> (
        vk::PipelineCreateFlags,
        &Arc<pumicite::pipeline::PipelineLayout>,
        &[ShaderEntry<'static>],
        &[vk::RayTracingShaderGroupCreateInfoKHR<'static>],
    ) {
        match &self.inner {
            RayTracingPipelineLibraryImpl::Monolithic {
                flags,
                layout,
                shaders,
                groups,
            } => (*flags, layout, shaders, groups),
            RayTracingPipelineLibraryImpl::Library(_) => {
                panic!("Cannot get inline data from a real pipeline library")
            }
        }
    }
}
impl AssetLoader for RayTracingPipelineLoader {
    type Asset = RayTracingPipelineLibrary;
    type Settings = ();
    type Error = PipelineLoaderError;

    async fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<RayTracingPipelineLibrary, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let pipeline: ron_types::RayTracingPipeline = ron::de::from_bytes(&bytes)?;

        let layout = match &pipeline.layout {
            ron_types::PipelineLayoutRef::Inline(pipeline_layout) => {
                PipelineLayoutLoader::load_inner(
                    pipeline_layout,
                    self.pipeline_cache.device().clone(),
                    self.heap.as_ref(),
                    load_context,
                )
                .await?
                .0
            }
            ron_types::PipelineLayoutRef::Path(path) => {
                load_context
                    .loader()
                    .immediate()
                    .load::<pumicite::bevy::PipelineLayout>(path)
                    .await?
                    .take()
                    .0
            }
            ron_types::PipelineLayoutRef::Bindless => {
                let Some(heap) = self.heap.as_ref() else {
                    return Err(PipelineLoaderError::BindlessPluginNeededError);
                };
                heap.bindless_pipeline_layout().clone()
            }
        };

        let mut flags = vk::PipelineCreateFlags::empty();
        if self.use_pipeline_library {
            flags |= vk::PipelineCreateFlags::LIBRARY_KHR;
        }
        {
            if pipeline.disable_optimization {
                flags |= vk::PipelineCreateFlags::DISABLE_OPTIMIZATION;
            }
            if pipeline.dispatch_base {
                flags |= vk::PipelineCreateFlags::DISPATCH_BASE;
            }
        }

        let mut shaders = Vec::new();
        let mut groups = Vec::new();
        let reused_shaders: BTreeMap<String, u32> = BTreeMap::new();
        let mut sbt_layout = SbtLayout::new(self.pipeline_cache.device());

        // Process stages in canonical order: RayGen, Miss, Callable, HitGroup.
        // SbtHandles assumes this ordering when computing handle offsets.
        for shader in
            pipeline
                .stages
                .iter()
                .filter(|s| matches!(s, ron_types::RayTracingPipelineShaderStage::RayGen { .. }))
                .chain(
                    pipeline.stages.iter().filter(|s| {
                        matches!(s, ron_types::RayTracingPipelineShaderStage::Miss { .. })
                    }),
                )
                .chain(pipeline.stages.iter().filter(|s| {
                    matches!(s, ron_types::RayTracingPipelineShaderStage::Callable { .. })
                }))
                .chain(pipeline.stages.iter().filter(|s| {
                    matches!(s, ron_types::RayTracingPipelineShaderStage::HitGroup { .. })
                }))
        {
            let mut process_shader = async |shader: &ron_types::Shader,
                                            stage: vk::ShaderStageFlags|
                   -> Result<u32, Self::Error> {
                let loaded_shader: LoadedAsset<ShaderModule> =
                    load_context.loader().immediate().load(&shader.path).await?;
                let index = shaders.len() as u32;
                let entry = CString::new(shader.entry_point.clone()).unwrap();
                shaders.push(ShaderEntry {
                    module: loaded_shader.get().0.clone(),
                    entry: Cow::Owned(entry),
                    flags: shader.flags(),
                    stage,
                    specialization_info: Cow::Owned(Default::default()),
                });
                Ok(index)
            };
            match shader {
                ron_types::RayTracingPipelineShaderStage::RayGen { shader, param_size } => {
                    sbt_layout.raygen.param_size = sbt_layout.raygen.param_size.max(*param_size);
                    sbt_layout.raygen.count += 1;
                    groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
                        general_shader: process_shader(shader, vk::ShaderStageFlags::RAYGEN_KHR)
                            .await?,
                        any_hit_shader: vk::SHADER_UNUSED_KHR,
                        closest_hit_shader: vk::SHADER_UNUSED_KHR,
                        intersection_shader: vk::SHADER_UNUSED_KHR,
                        ..Default::default()
                    });
                }
                ron_types::RayTracingPipelineShaderStage::Miss { shader, param_size } => {
                    sbt_layout.miss.param_size = sbt_layout.miss.param_size.max(*param_size);
                    sbt_layout.miss.count += 1;
                    groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
                        general_shader: process_shader(shader, vk::ShaderStageFlags::MISS_KHR)
                            .await?,
                        any_hit_shader: vk::SHADER_UNUSED_KHR,
                        closest_hit_shader: vk::SHADER_UNUSED_KHR,
                        intersection_shader: vk::SHADER_UNUSED_KHR,
                        ..Default::default()
                    });
                }
                ron_types::RayTracingPipelineShaderStage::Callable { shader, param_size } => {
                    sbt_layout.callable.param_size =
                        sbt_layout.callable.param_size.max(*param_size);
                    sbt_layout.callable.count += 1;
                    groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
                        general_shader: process_shader(shader, vk::ShaderStageFlags::CALLABLE_KHR)
                            .await?,
                        any_hit_shader: vk::SHADER_UNUSED_KHR,
                        closest_hit_shader: vk::SHADER_UNUSED_KHR,
                        intersection_shader: vk::SHADER_UNUSED_KHR,
                        ..Default::default()
                    });
                }
                ron_types::RayTracingPipelineShaderStage::HitGroup {
                    ty,
                    intersection,
                    any_hit,
                    closest_hit,
                    param_size,
                } => {
                    sbt_layout.hitgroup.param_size =
                        sbt_layout.hitgroup.param_size.max(*param_size);
                    sbt_layout.hitgroup.count += 1;
                    let mut process_hitgroup_shader =
                        async |shader: &ron_types::HitgroupShader,
                               stage: vk::ShaderStageFlags|
                               -> Result<u32, Self::Error> {
                            let index = match shader {
                                ron_types::HitgroupShader::Singleton(shader) => {
                                    process_shader(shader, stage).await?
                                }
                                ron_types::HitgroupShader::Reused(name) => {
                                    if let Some(&group) = reused_shaders.get(name) {
                                        group
                                    } else {
                                        let shader = pipeline
                                            .shaders
                                            .get(name)
                                            .expect("Referencing a shader that doen't exist");
                                        process_shader(shader, stage).await?
                                    }
                                }
                                _ => vk::SHADER_UNUSED_KHR,
                            };
                            Ok(index)
                        };
                    groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
                        ty: match ty {
                            ron_types::RayTracingPipelineShaderHitGroupType::Triangles => {
                                vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
                            }
                            ron_types::RayTracingPipelineShaderHitGroupType::Aabbs => {
                                vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
                            }
                        },
                        general_shader: vk::SHADER_UNUSED_KHR,
                        any_hit_shader: process_hitgroup_shader(
                            any_hit,
                            vk::ShaderStageFlags::ANY_HIT_KHR,
                        )
                        .await?,
                        closest_hit_shader: process_hitgroup_shader(
                            closest_hit,
                            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                        )
                        .await?,
                        intersection_shader: process_hitgroup_shader(
                            intersection,
                            vk::ShaderStageFlags::INTERSECTION_KHR,
                        )
                        .await?,
                        ..Default::default()
                    });
                }
            }
        }

        let inner = if self.use_pipeline_library {
            let span = tracing::span!(
                tracing::Level::INFO,
                "Creating Ray Tracing Pipeline Library",
                path = load_context.asset_path().to_string()
            )
            .entered();
            let pipeline_obj = self.pipeline_cache.create_ray_tracing_pipeline_library(
                RayTracingPipelineLibraryCreateInfo {
                    flags,
                    layout,
                    max_ray_recursion_depth: pipeline.max_ray_recursion_depth,
                    max_ray_payload_size: pipeline.max_ray_payload_size,
                    max_hit_attribute_size: pipeline.max_hit_attribute_size,
                    dynamic_stack_size: pipeline.dynamic_stack_size,
                    shaders: &shaders,
                    groups: &groups,
                },
            )?;
            span.exit();
            RayTracingPipelineLibraryImpl::Library(Arc::new(pipeline_obj))
        } else {
            let owned_shaders = shaders
                .into_iter()
                .map(|s| ShaderEntry {
                    module: s.module,
                    entry: Cow::Owned(s.entry.into_owned()),
                    flags: s.flags,
                    stage: s.stage,
                    specialization_info: s.specialization_info,
                })
                .collect();
            RayTracingPipelineLibraryImpl::Monolithic {
                flags,
                layout,
                shaders: owned_shaders,
                groups,
            }
        };
        Ok(RayTracingPipelineLibrary {
            inner,
            max_ray_recursion_depth: pipeline.max_ray_recursion_depth,
            max_ray_payload_size: pipeline.max_ray_payload_size,
            max_hit_attribute_size: pipeline.max_hit_attribute_size,
            dynamic_stack_size: pipeline.dynamic_stack_size,
            sbt_layout,
        })
    }

    fn extensions(&self) -> &[&str] {
        &["rtx.pipeline.ron"]
    }
}

/// Asset loader for pipeline layouts (`.playout.ron` files).
///
/// Loads pipeline layout configurations including descriptor set layouts
/// and push constant ranges. Layouts are cached by path.
pub struct PipelineLayoutLoader {
    device: Device,
    cache: async_lock::Mutex<HashMap<AssetPath<'static>, Weak<pumicite::pipeline::PipelineLayout>>>,
    heap: Option<DescriptorHeap>,
}
impl FromWorld for PipelineLayoutLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            device: world.resource::<Device>().clone(),
            cache: async_lock::Mutex::new(HashMap::new()),
            heap: world.get_resource().cloned(),
        }
    }
}

impl PipelineLayoutLoader {
    async fn load_inner(
        layout: &ron_types::PipelineLayout,
        device: Device,
        heap: Option<&DescriptorHeap>,
        ctx: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<pumicite::bevy::PipelineLayout, PipelineLoaderError> {
        let mut flags: vk::PipelineLayoutCreateFlags = vk::PipelineLayoutCreateFlags::empty();
        if layout.independent_sets {
            flags |= vk::PipelineLayoutCreateFlags::INDEPENDENT_SETS_EXT;
        }

        let mut set_layouts: Vec<Arc<pumicite::descriptor::DescriptorSetLayout>> =
            Vec::with_capacity(layout.sets.len());
        for descriptor in layout.sets.iter() {
            let layout = match descriptor {
                ron_types::DescriptorSetLayoutRef::Inline(descriptor) => {
                    DescriptorSetLayoutLoader::load_inner(descriptor, device.clone())?.0
                }
                ron_types::DescriptorSetLayoutRef::Path(path) => {
                    let a: LoadedAsset<pumicite::bevy::DescriptorSetLayout> =
                        ctx.loader().immediate().load(path).await?;
                    a.take().0
                }
                ron_types::DescriptorSetLayoutRef::SamplerHeap => {
                    let Some(heap) = heap else {
                        return Err(PipelineLoaderError::BindlessPluginNeededError);
                    };
                    heap.sampler_heap().descriptor_layout().clone()
                }
                ron_types::DescriptorSetLayoutRef::ResourceHeap => {
                    let Some(heap) = heap else {
                        return Err(PipelineLoaderError::BindlessPluginNeededError);
                    };
                    heap.resource_heap().descriptor_layout().clone()
                }
            };
            set_layouts.push(layout);
        }

        let pipeline_layout = pumicite::pipeline::PipelineLayout::new(
            device,
            set_layouts,
            &layout
                .push_constants
                .iter()
                .map(|(&stage, (range_start, range_end))| vk::PushConstantRange {
                    stage_flags: stage.into(),
                    offset: *range_start,
                    size: *range_end - *range_start,
                })
                .collect::<Vec<_>>(),
            flags,
        )?;

        Ok(pumicite::bevy::PipelineLayout(Arc::new(pipeline_layout)))
    }
}

impl AssetLoader for PipelineLayoutLoader {
    type Asset = pumicite::bevy::PipelineLayout;

    type Settings = ();

    type Error = PipelineLoaderError;

    async fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<pumicite::bevy::PipelineLayout, Self::Error> {
        let mut lock = self.cache.lock().await;
        if let Some(cached) = lock.get(load_context.asset_path()).and_then(Weak::upgrade) {
            return Ok(pumicite::bevy::PipelineLayout(cached));
        }

        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let layout: ron_types::PipelineLayout = ron::de::from_bytes(&bytes)?;

        let layout = Self::load_inner(
            &layout,
            self.device.clone(),
            self.heap.as_ref(),
            load_context,
        )
        .await?;
        lock.insert(
            load_context.asset_path().clone_owned(),
            Arc::downgrade(&layout),
        );
        drop(lock);

        Ok(layout)
    }

    fn extensions(&self) -> &[&str] {
        &["playout.ron"]
    }
}

/// Asset loader for descriptor set layouts.
///
/// Used internally by pipeline loaders. Descriptor set layouts are cached.
pub struct DescriptorSetLayoutLoader {
    device: Device,
    cache: async_lock::Mutex<
        HashMap<AssetPath<'static>, Weak<pumicite::descriptor::DescriptorSetLayout>>,
    >,
}
impl FromWorld for DescriptorSetLayoutLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            device: world.resource::<Device>().clone(),
            cache: async_lock::Mutex::new(HashMap::new()),
        }
    }
}
impl DescriptorSetLayoutLoader {
    fn load_inner(
        descriptor: &ron_types::DescriptorSetLayout,
        device: Device,
    ) -> VkResult<pumicite::bevy::DescriptorSetLayout> {
        let bindings: Vec<_> = descriptor
            .bindings
            .iter()
            .map(|binding| vk::DescriptorSetLayoutBinding {
                binding: binding.binding,
                descriptor_type: binding.ty.into(),
                descriptor_count: binding.count,
                stage_flags: binding
                    .stages
                    .iter()
                    .fold(vk::ShaderStageFlags::empty(), |x, &y| x | y.into()),
                ..Default::default()
            })
            .collect();
        let mut flags = vk::DescriptorSetLayoutCreateFlags::empty();
        if descriptor.update_after_bind_pool {
            flags |= vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
        }
        if descriptor.push_descriptor {
            flags |= vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR;
        }
        pumicite::descriptor::DescriptorSetLayout::new(device.clone(), &bindings, &[], &[], flags)
            .map(Arc::new)
            .map(pumicite::bevy::DescriptorSetLayout)
    }
}
impl AssetLoader for DescriptorSetLayoutLoader {
    type Asset = pumicite::bevy::DescriptorSetLayout;

    type Settings = ();

    type Error = PipelineLoaderError;

    async fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<pumicite::bevy::DescriptorSetLayout, Self::Error> {
        let mut lock = self.cache.lock().await;
        if let Some(cached) = lock.get(load_context.asset_path()).and_then(Weak::upgrade) {
            return Ok(pumicite::bevy::DescriptorSetLayout(cached));
        }

        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let layout: ron_types::DescriptorSetLayout = ron::de::from_bytes(&bytes)?;

        let layout = Self::load_inner(&layout, self.device.clone())?;
        lock.insert(
            load_context.asset_path().clone_owned(),
            Arc::downgrade(&layout),
        );

        drop(lock);
        Ok(layout)
    }
}
