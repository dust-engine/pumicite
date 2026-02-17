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
//! | `.comp.pipeline.ron` | [`ComputePipeline`] | Compute pipeline configuration |
//! | `.gfx.pipeline.ron` | [`GraphicsPipeline`] | Graphics pipeline configuration |
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
    ops::Deref,
    sync::{Arc, Weak},
};

use bevy_asset::{Asset, AssetLoader, AssetPath, LoadDirectError, LoadedAsset};
use bevy_ecs::world::FromWorld;
use bevy_reflect::TypePath;
use pumicite::{
    Device, HasDevice,
    ash::{
        self, VkResult,
        vk::{self, TaggedStructure},
    },
    pipeline::{Pipeline, PipelineCache, ShaderEntry},
    rtx::{RayTracingPipelineLibraryCreateInfo, SbtLayout},
    utils::AsVkHandle,
};
use thiserror::Error;

use crate::DescriptorHeap;

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

/// Asset loader for compute pipelines (`.comp.pipeline.ron` files).
///
/// Loads compute pipeline configurations and creates Vulkan compute pipelines.
pub struct ComputePipelineLoader {
    pipeline_cache: Arc<PipelineCache>,
    heap: Option<DescriptorHeap>,
}
impl FromWorld for ComputePipelineLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            pipeline_cache: world
                .resource::<pumicite::bevy::PipelineCache>()
                .deref()
                .clone(),
            heap: world.get_resource().cloned(),
        }
    }
}

/// A loaded compute pipeline asset.
///
/// Load via asset server with `.comp.pipeline.ron` extension.
#[derive(Clone, Asset, TypePath)]
pub struct ComputePipeline(Arc<Pipeline>);
impl AssetLoader for ComputePipelineLoader {
    type Asset = ComputePipeline;
    type Settings = ();
    type Error = PipelineLoaderError;

    async fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<ComputePipeline, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let pipeline: ron_types::ComputePipeline = ron::de::from_bytes(&bytes)?;

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
        {
            if pipeline.disable_optimization {
                flags |= vk::PipelineCreateFlags::DISABLE_OPTIMIZATION;
            }
            if pipeline.dispatch_base {
                flags |= vk::PipelineCreateFlags::DISPATCH_BASE;
            }
        }

        let shader: LoadedAsset<ShaderModule> = load_context
            .loader()
            .immediate()
            .load(&pipeline.shader.path)
            .await?;
        let shader_flags = pipeline.shader.flags();
        let entry_point: CString = CString::new(pipeline.shader.entry_point)
            .map_err(|_| PipelineLoaderError::PipelineError("Invalid entry name"))?;

        let span = tracing::span!(
            tracing::Level::INFO,
            "Creating Compute Pipeline",
            path = load_context.asset_path().to_string()
        )
        .entered();
        let pipeline = self.pipeline_cache.create_compute_pipeline(
            layout,
            flags,
            &ShaderEntry {
                module: shader.get().0.clone(),
                entry: Cow::Owned(entry_point),
                flags: shader_flags,
                stage: vk::ShaderStageFlags::COMPUTE,
                specialization_info: Cow::Owned(Default::default()),
            },
        )?;
        span.exit();
        Ok(ComputePipeline(Arc::new(pipeline)))
    }

    fn extensions(&self) -> &[&str] {
        &["comp.pipeline.ron"]
    }
}
impl ComputePipeline {
    pub fn into_inner(self) -> Arc<Pipeline> {
        self.0
    }
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

/// A loaded graphics pipeline asset.
///
/// Load via asset server with `.gfx.pipeline.ron` extension.
///
/// Graphics pipelines can have runtime variants via [`GraphicsPipelineVariant`](ron_types::GraphicsPipelineVariant)
/// for specialization constants and format overrides.
#[derive(Clone, Asset, TypePath)]
pub struct GraphicsPipeline(Arc<Pipeline>);
impl GraphicsPipeline {
    /// Unwraps the inner pipeline.
    pub fn into_inner(self) -> Arc<Pipeline> {
        self.0
    }
}
impl Deref for GraphicsPipeline {
    type Target = Arc<Pipeline>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Asset loader for graphics pipelines (`.gfx.pipeline.ron` files).
///
/// Loads graphics pipeline configurations with full support for:
/// - Vertex input, rasterization, depth/stencil state
/// - Dynamic state
/// - Specialization constants
/// - Pipeline variants
pub struct GraphicsPipelineLoader {
    pipeline_cache: Arc<PipelineCache>,
    heap: Option<DescriptorHeap>,
}
impl FromWorld for GraphicsPipelineLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            pipeline_cache: Arc::new(PipelineCache::null(world.resource::<Device>().clone())),
            heap: world.get_resource::<DescriptorHeap>().cloned(),
        }
    }
}
impl AssetLoader for GraphicsPipelineLoader {
    type Asset = GraphicsPipeline;
    type Settings = ron_types::GraphicsPipelineVariant;
    type Error = PipelineLoaderError;

    async fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<GraphicsPipeline, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let mut pipeline: ron_types::GraphicsPipeline = ron::de::from_bytes(&bytes)?;
        settings.apply_on(&mut pipeline);

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

        let mut dynamic_states = Vec::<vk::DynamicState>::new();
        let mut shader_modules = Vec::with_capacity(pipeline.shaders.len());
        for (_, shader) in pipeline.shaders.iter() {
            let module: LoadedAsset<ShaderModule> =
                load_context.loader().immediate().load(&shader.path).await?;
            shader_modules.push(module.take());
        }
        let shader_entry_names = pipeline
            .shaders
            .iter()
            .map(|x| CString::new(&*x.1.entry_point))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| PipelineLoaderError::PipelineError("Invalid entry name"))?;

        // Build specialization info data structures
        let mut specialization_map_entries: Vec<Vec<vk::SpecializationMapEntry>> = Vec::new();
        let mut specialization_data: Vec<Vec<u8>> = Vec::new();

        for (_, shader) in pipeline.shaders.iter() {
            if shader.specialization_constants.is_empty() {
                specialization_map_entries.push(Vec::new());
                specialization_data.push(Vec::new());
            } else {
                let mut entries = Vec::new();
                let mut data = Vec::new();

                for (&constant_id, value) in shader.specialization_constants.iter() {
                    let offset = data.len();

                    let size = value.extend(&mut data);

                    entries.push(vk::SpecializationMapEntry {
                        constant_id,
                        offset: offset as u32,
                        size,
                    });
                }

                specialization_map_entries.push(entries);
                specialization_data.push(data);
            }
        }

        // Build the SpecializationInfo structs with stable pointers
        let specialization_infos: Vec<vk::SpecializationInfo> = specialization_map_entries
            .iter()
            .zip(specialization_data.iter())
            .map(|(entries, data)| vk::SpecializationInfo {
                map_entry_count: entries.len() as u32,
                p_map_entries: entries.as_ptr(),
                data_size: data.len(),
                p_data: data.as_ptr() as *const std::ffi::c_void,
                ..Default::default()
            })
            .collect();

        let stages = pipeline
            .shaders
            .iter()
            .zip(shader_modules.iter())
            .zip(shader_entry_names.iter())
            .enumerate()
            .map(|(i, (((&stage, shader), module), entry_name))| {
                vk::PipelineShaderStageCreateInfo {
                    flags: shader.flags(),
                    stage: stage.into(),
                    module: module.0.vk_handle(),
                    p_specialization_info: if specialization_map_entries[i].is_empty() {
                        std::ptr::null()
                    } else {
                        &specialization_infos[i]
                    },
                    p_name: entry_name.as_ptr(),
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();

        let mut info = vk::GraphicsPipelineCreateInfo {
            layout: layout.vk_handle(),
            ..Default::default()
        }
        .stages(&stages);

        let mut vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();
        let vertex_attribute_descriptions;
        let vertex_binding_descriptions;
        if let Some(bindings) = pipeline.vertex_bindings.unwrap(&mut dynamic_states) {
            vertex_attribute_descriptions = bindings
                .iter()
                .flat_map(|(&binding, desc)| {
                    desc.attributes.iter().map(move |(&location, attribute)| {
                        Some(vk::VertexInputAttributeDescription {
                            location,
                            binding,
                            format: attribute.format.into(),
                            offset: attribute.offset,
                        })
                    })
                })
                .collect::<Option<Vec<_>>>()
                .ok_or(PipelineLoaderError::PipelineError("Invalid format"))?;
            vertex_binding_descriptions = bindings
                .iter()
                .map(|(&binding, desc)| vk::VertexInputBindingDescription {
                    binding,
                    stride: desc
                        .stride
                        .unwrap(&mut dynamic_states)
                        .cloned()
                        .unwrap_or_default(),
                    input_rate: desc.input_rate.into(),
                })
                .collect::<Vec<_>>();
            vertex_input_state = vertex_input_state
                .vertex_attribute_descriptions(&vertex_attribute_descriptions)
                .vertex_binding_descriptions(&vertex_binding_descriptions);
        }
        info = info.vertex_input_state(&vertex_input_state);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: pipeline
                .topology
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            primitive_restart_enable: pipeline
                .primitive_restart_enabled
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            ..Default::default()
        };
        info = info.input_assembly_state(&input_assembly_state);

        let tessellation_state = vk::PipelineTessellationStateCreateInfo {
            patch_control_points: pipeline
                .tessellation_patch_control_points
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default(),
            ..Default::default()
        };
        info = info.tessellation_state(&tessellation_state);

        let mut viewport_state = vk::PipelineViewportStateCreateInfo::default();
        let viewports: Vec<vk::Viewport>;
        let scissors: Vec<vk::Rect2D>;
        match pipeline.viewports {
            ron_types::CountedDynamicState::Dynamic => {
                dynamic_states.push(vk::DynamicState::VIEWPORT_WITH_COUNT);
            }
            ron_types::CountedDynamicState::Count(count) => {
                dynamic_states.push(vk::DynamicState::VIEWPORT);
                viewport_state.viewport_count = count;
            }
            ron_types::CountedDynamicState::Static(items) => {
                viewports = items.iter().map(|view| view.clone().into()).collect();
                viewport_state = viewport_state.viewports(&viewports);
            }
        }
        match pipeline.scissors {
            ron_types::CountedDynamicState::Dynamic => {
                dynamic_states.push(vk::DynamicState::SCISSOR_WITH_COUNT);
            }
            ron_types::CountedDynamicState::Count(count) => {
                dynamic_states.push(vk::DynamicState::SCISSOR);
                viewport_state.scissor_count = count;
            }
            ron_types::CountedDynamicState::Static(items) => {
                scissors = items.iter().map(|view| view.clone().into()).collect();
                viewport_state = viewport_state.scissors(&scissors);
            }
        }
        if viewport_state.scissor_count > 0 || viewport_state.viewport_count > 0 {
            info = info.viewport_state(&viewport_state);
        }

        let mut rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: pipeline
                .depth_clamp_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            rasterizer_discard_enable: pipeline
                .rasterizer_discard_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            polygon_mode: pipeline
                .polygon_mode
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            cull_mode: pipeline
                .cull_mode
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            front_face: pipeline
                .front_face
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            depth_bias_enable: pipeline
                .depth_bias_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            line_width: pipeline
                .line_width
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default(),
            ..Default::default()
        };

        if let Some(depth_bias) = pipeline.depth_bias.unwrap(&mut dynamic_states) {
            rasterization_state.depth_bias_constant_factor = depth_bias.constant_factor;
            rasterization_state.depth_bias_clamp = depth_bias.clamp;
            rasterization_state.depth_bias_slope_factor = depth_bias.slope_factor;
        }

        info = info.rasterization_state(&rasterization_state);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: match pipeline.sample_count.unwrap(&mut dynamic_states) {
                None => vk::SampleCountFlags::empty(),
                Some(n) => match n {
                    1 | 2 | 4 | 8 | 16 | 32 | 64 => vk::SampleCountFlags::from_raw(*n as u32),
                    _ => {
                        return Err(PipelineLoaderError::PipelineError(
                            "Unrecognized: sample_count",
                        ));
                    }
                },
            },
            sample_shading_enable: pipeline.sample_shading.is_some().into(),
            min_sample_shading: pipeline.sample_shading.unwrap_or_default(),
            alpha_to_coverage_enable: pipeline
                .alpha_to_coverage_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            alpha_to_one_enable: pipeline
                .alpha_to_one_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            ..Default::default()
        }
        .sample_mask(
            pipeline
                .sample_mask
                .unwrap(&mut dynamic_states)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        );
        info = info.multisample_state(&multisample_state);

        let depth_test_enable: bool = pipeline
            .depth_test_enable
            .unwrap(&mut dynamic_states)
            .cloned()
            .unwrap_or_default();
        let mut ds_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: depth_test_enable.into(),
            depth_write_enable: pipeline
                .depth_write_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            depth_compare_op: if depth_test_enable {
                compare_op_str(
                    pipeline
                        .depth_compare_op
                        .unwrap(&mut dynamic_states)
                        .map(String::as_str),
                    "Invalid depth compare op",
                )?
            } else {
                Default::default()
            },
            depth_bounds_test_enable: pipeline
                .depth_bounds_test_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            stencil_test_enable: pipeline
                .stencil_test_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            ..Default::default()
        };
        if let Some((min, max)) = pipeline.depth_bounds.unwrap(&mut dynamic_states) {
            ds_state.min_depth_bounds = *min;
            ds_state.max_depth_bounds = *max;
        }
        if let Some(front) = &pipeline.front_stencil
            && let Some(ops) = front.ops.unwrap(&mut dynamic_states)
        {
            ds_state.front.fail_op = ops.fail.into();
            ds_state.front.pass_op = ops.pass.into();
            ds_state.front.depth_fail_op = ops.depth_fail.into();
            ds_state.front.compare_op = compare_op_str(
                Some(ops.compare.as_str()),
                "Invalid comapre op for front stencil",
            )?;
        }
        if let Some(back) = &pipeline.back_stencil
            && let Some(ops) = back.ops.unwrap(&mut dynamic_states)
        {
            ds_state.back.fail_op = ops.fail.into();
            ds_state.back.pass_op = ops.pass.into();
            ds_state.back.depth_fail_op = ops.depth_fail.into();
            ds_state.back.compare_op = compare_op_str(
                Some(ops.compare.as_str()),
                "Invalid compare op for back stencil",
            )?;
        }

        info = info.depth_stencil_state(&ds_state);

        let color_blend_state_attachments = {
            let mut blend_enable_dynamic_count = 0_u32;
            let mut blend_equation_dynamic_count = 0_u32;
            let mut color_write_mask_dynamic_count = 0_u32;
            let attachments = pipeline
                .attachments
                .iter()
                .map(|attachment| {
                    let mut state = vk::PipelineColorBlendAttachmentState::default();
                    match attachment.blend_enable {
                        ron_types::RequiredDynamicState::Dynamic => {
                            blend_enable_dynamic_count += 1;
                        }
                        ron_types::RequiredDynamicState::Static(bool) => {
                            state.blend_enable = bool.into();
                        }
                    }
                    match &attachment.blend_equation {
                        ron_types::OptionalDynamicState::Dynamic => {
                            blend_equation_dynamic_count += 1
                        }
                        ron_types::OptionalDynamicState::Static(equation) => {
                            state.color_blend_op = equation.color.1.into();
                            state.src_color_blend_factor = equation.color.0.into();
                            state.dst_color_blend_factor = equation.color.2.into();

                            state.alpha_blend_op = equation.alpha.1.into();
                            state.src_alpha_blend_factor = equation.alpha.0.into();
                            state.dst_alpha_blend_factor = equation.alpha.2.into();
                        }
                        ron_types::OptionalDynamicState::None => {
                            if matches!(
                                attachment.blend_enable,
                                ron_types::RequiredDynamicState::Static(true)
                            ) {
                                return Err(PipelineLoaderError::PipelineError(
                                    "Blending enabled; blend equation required",
                                ));
                            }
                        }
                    }
                    match &attachment.color_write_mask {
                        ron_types::OptionalDynamicState::None => {
                            state.color_write_mask = vk::ColorComponentFlags::RGBA;
                        }
                        ron_types::OptionalDynamicState::Dynamic => {
                            color_write_mask_dynamic_count += 1
                        }
                        ron_types::OptionalDynamicState::Static(mask) => {
                            let mut flags = vk::ColorComponentFlags::empty();
                            for char in mask.chars() {
                                match char {
                                    'r' => flags |= vk::ColorComponentFlags::R,
                                    'g' => flags |= vk::ColorComponentFlags::G,
                                    'b' => flags |= vk::ColorComponentFlags::B,
                                    'a' => flags |= vk::ColorComponentFlags::A,
                                    _ => {
                                        return Err(PipelineLoaderError::PipelineError(
                                            "unrecognized color write mask",
                                        ));
                                    }
                                }
                            }
                            state.color_write_mask = flags;
                        }
                    }
                    Ok(state)
                })
                .collect::<Result<Vec<_>, PipelineLoaderError>>()?;
            if blend_enable_dynamic_count == pipeline.attachments.len() as u32 {
                dynamic_states.push(vk::DynamicState::COLOR_BLEND_ENABLE_EXT);
            } else if blend_enable_dynamic_count != 0 {
                return Err(PipelineLoaderError::PipelineError(
                    "If some attachment[n].blend_enable was set to Dynamic, it must be set to Dynamic for all attachments",
                ));
            }
            if blend_equation_dynamic_count == pipeline.attachments.len() as u32 {
                dynamic_states.push(vk::DynamicState::COLOR_BLEND_EQUATION_EXT);
            } else if blend_equation_dynamic_count != 0 {
                return Err(PipelineLoaderError::PipelineError(
                    "If some attachment[n].blend_equation was set to Dynamic, it must be set to Dynamic for all attachments",
                ));
            }
            if color_write_mask_dynamic_count == pipeline.attachments.len() as u32 {
                dynamic_states.push(vk::DynamicState::COLOR_WRITE_MASK_EXT);
            } else if color_write_mask_dynamic_count != 0 {
                return Err(PipelineLoaderError::PipelineError(
                    "If some attachment[n].color_write_mask was set to Dynamic, it must be set to Dynamic for all attachments",
                ));
            }

            attachments
        };
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: pipeline
                .blend_logic_op_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            logic_op: if let Some(a) = pipeline.blend_logic_op.unwrap(&mut dynamic_states) {
                match a.as_str() {
                    "s & t" => vk::LogicOp::AND,
                    "s & !t" => vk::LogicOp::AND_REVERSE,
                    "!s & t" => vk::LogicOp::AND_INVERTED,
                    "!(s & t)" => vk::LogicOp::NAND,

                    "s | t" => vk::LogicOp::OR,
                    "s | !t" => vk::LogicOp::OR_REVERSE,
                    "!s | t" => vk::LogicOp::OR_INVERTED,
                    "!(s | t)" => vk::LogicOp::NOR,

                    "s ^ t" => vk::LogicOp::XOR,
                    "!(s ^ t)" => vk::LogicOp::EQUIVALENT,

                    "d" => vk::LogicOp::NO_OP,
                    "!d" => vk::LogicOp::INVERT,
                    "s" => vk::LogicOp::COPY,
                    "!s" => vk::LogicOp::COPY_INVERTED,
                    "0" => vk::LogicOp::CLEAR,
                    "1" => vk::LogicOp::SET,
                    _ => {
                        return Err(PipelineLoaderError::PipelineError(
                            "Unrecognized: blend_logic_op",
                        ));
                    }
                }
            } else {
                vk::LogicOp::CLEAR
            },
            blend_constants: pipeline
                .blend_constants
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default(),
            ..Default::default()
        }
        .attachments(&color_blend_state_attachments);
        info = info.color_blend_state(&color_blend_state);

        let dynamic_states =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        info = info.dynamic_state(&dynamic_states);

        let rendering_state_color_attachments_format: Vec<vk::Format> = pipeline
            .attachments
            .iter()
            .map(|x| x.format.into())
            .collect();
        let mut rendering_state = vk::PipelineRenderingCreateInfo {
            depth_attachment_format: pipeline.depth_format.into(),
            stencil_attachment_format: pipeline.stencil_format.into(),
            ..Default::default()
        }
        .color_attachment_formats(&rendering_state_color_attachments_format);

        info = info.push(&mut rendering_state);

        fn compare_op_str(
            op: Option<&str>,
            err: &'static str,
        ) -> Result<vk::CompareOp, PipelineLoaderError> {
            match op {
                None | Some("false" | "0") => Ok(vk::CompareOp::NEVER),
                Some("<") => Ok(vk::CompareOp::LESS),
                Some("==") => Ok(vk::CompareOp::EQUAL),
                Some("<=") => Ok(vk::CompareOp::LESS_OR_EQUAL),
                Some(">") => Ok(vk::CompareOp::GREATER),
                Some("!=") => Ok(vk::CompareOp::NOT_EQUAL),
                Some(">=") => Ok(vk::CompareOp::GREATER_OR_EQUAL),
                Some("true" | "1") => Ok(vk::CompareOp::ALWAYS),
                _ => Err(PipelineLoaderError::PipelineError(err)),
            }
        }
        let span = tracing::span!(
            tracing::Level::INFO,
            "Creating Graphics Pipeline",
            path = load_context.asset_path().to_string()
        )
        .entered();
        let pipeline = self
            .pipeline_cache
            .create_graphics_pipeline(layout, &info)?;
        span.exit();
        Ok(GraphicsPipeline(Arc::new(pipeline)))
    }

    fn extensions(&self) -> &[&str] {
        &["gfx.pipeline.ron"]
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

pub mod ron_types {
    use pumicite::ash::vk;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    #[derive(Serialize, Deserialize)]
    pub struct ComputePipeline {
        /// The compute shader used for compiling the compute pipeline
        pub shader: Shader,

        /// Path to the pipeline layout
        #[serde(default)]
        pub layout: PipelineLayoutRef,

        /// The created pipeline will not be optimized. Using this flag may reduce the time
        /// taken to create the pipeline.
        #[serde(default)]
        pub disable_optimization: bool,

        /// The compute pipeline can be used with vkCmdDispatchBase with a non-zero base workgroup.
        #[serde(default)]
        pub dispatch_base: bool,
    }

    #[derive(Serialize, Deserialize, Default)]
    #[serde(untagged)]
    pub enum OptionalDynamicState<T, const DYNAMIC_STATE: i32> {
        Dynamic,
        #[default]
        None,
        Static(T),
    }

    impl<T, const DYNAMIC_STATE: i32> OptionalDynamicState<T, DYNAMIC_STATE> {
        pub fn unwrap(&self, dynamic_states: &mut Vec<vk::DynamicState>) -> Option<&T> {
            match self {
                Self::Dynamic => {
                    dynamic_states.push(vk::DynamicState::from_raw(DYNAMIC_STATE));
                    None
                }
                Self::None => None,
                Self::Static(a) => Some(a),
            }
        }
    }
    #[derive(Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum RequiredDynamicState<T, const DYNAMIC_STATE: i32> {
        Dynamic,
        Static(T),
    }
    impl<T: Default, const DYNAMIC_STATE: i32> Default for RequiredDynamicState<T, DYNAMIC_STATE> {
        fn default() -> Self {
            Self::Static(T::default())
        }
    }

    impl<T, const DYNAMIC_STATE: i32> RequiredDynamicState<T, DYNAMIC_STATE> {
        pub fn unwrap(&self, dynamic_states: &mut Vec<vk::DynamicState>) -> Option<&T> {
            match self {
                Self::Dynamic => {
                    dynamic_states.push(vk::DynamicState::from_raw(DYNAMIC_STATE));
                    None
                }
                Self::Static(a) => Some(a),
            }
        }
    }

    #[derive(Serialize, Deserialize)]
    pub enum CountedDynamicState<T> {
        Dynamic,
        Count(u32),
        Static(Vec<T>),
    }

    #[derive(Serialize, Deserialize)]
    pub struct GraphicsPipeline {
        pub shaders: BTreeMap<ShaderStage, Shader>,

        /// Dynamic: Requires VK_DYNAMIC_STATE_VERTEX_INPUT_EXT
        /// May be None when using mesh shading
        #[serde(default)]
        pub vertex_bindings: OptionalDynamicState<
            BTreeMap<u32, VertexInputBinding>,
            { vk::DynamicState::VERTEX_INPUT_EXT.as_raw() },
        >,

        /// Dynamic: VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY
        #[serde(default)]
        pub topology: RequiredDynamicState<
            PrimitiveTopology,
            { vk::DynamicState::PRIMITIVE_TOPOLOGY.as_raw() },
        >,

        /// Dynamic: VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE
        #[serde(default)]
        pub primitive_restart_enabled:
            RequiredDynamicState<bool, { vk::DynamicState::PRIMITIVE_RESTART_ENABLE.as_raw() }>,

        #[serde(default)]
        pub tessellation_patch_control_points:
            OptionalDynamicState<u32, { vk::DynamicState::PATCH_CONTROL_POINTS_EXT.as_raw() }>,

        #[serde(default = "viewport_default")]
        pub viewports: CountedDynamicState<Viewport>,
        #[serde(default = "scissor_default")]
        pub scissors: CountedDynamicState<Rect2D>,

        #[serde(default)]
        pub depth_clamp_enable:
            RequiredDynamicState<bool, { vk::DynamicState::DEPTH_CLAMP_ENABLE_EXT.as_raw() }>,

        #[serde(default)]
        pub rasterizer_discard_enable:
            RequiredDynamicState<bool, { vk::DynamicState::RASTERIZER_DISCARD_ENABLE.as_raw() }>,

        #[serde(default)]
        pub polygon_mode:
            RequiredDynamicState<PolygonMode, { vk::DynamicState::POLYGON_MODE_EXT.as_raw() }>,

        #[serde(default)]
        pub cull_mode: RequiredDynamicState<CullMode, { vk::DynamicState::CULL_MODE.as_raw() }>,

        #[serde(default)]
        pub front_face: RequiredDynamicState<FrontFace, { vk::DynamicState::FRONT_FACE.as_raw() }>,

        #[serde(default)]
        pub depth_bias_enable:
            RequiredDynamicState<bool, { vk::DynamicState::DEPTH_BIAS_ENABLE.as_raw() }>,

        #[serde(default)]
        pub depth_bias: OptionalDynamicState<DepthBias, { vk::DynamicState::DEPTH_BIAS.as_raw() }>,

        #[serde(default = "line_width_default")]
        pub line_width: RequiredDynamicState<f32, { vk::DynamicState::LINE_WIDTH.as_raw() }>,

        #[serde(default = "sample_count_default")]
        pub sample_count:
            RequiredDynamicState<u8, { vk::DynamicState::RASTERIZATION_SAMPLES_EXT.as_raw() }>,

        /// If None, sample shading is turned off.
        /// If Some(x), sample shading is turned on, and x is the minimum fraction of sample shading.
        #[serde(default)]
        pub sample_shading: Option<f32>,

        #[serde(default)]
        pub sample_mask:
            OptionalDynamicState<Vec<u32>, { vk::DynamicState::SAMPLE_MASK_EXT.as_raw() }>,

        #[serde(default)]
        pub alpha_to_coverage_enable:
            RequiredDynamicState<bool, { vk::DynamicState::ALPHA_TO_COVERAGE_ENABLE_EXT.as_raw() }>,
        #[serde(default)]
        pub alpha_to_one_enable:
            RequiredDynamicState<bool, { vk::DynamicState::ALPHA_TO_ONE_ENABLE_EXT.as_raw() }>,
        #[serde(default)]
        pub depth_test_enable:
            RequiredDynamicState<bool, { vk::DynamicState::DEPTH_TEST_ENABLE.as_raw() }>,
        #[serde(default)]
        pub depth_write_enable:
            RequiredDynamicState<bool, { vk::DynamicState::DEPTH_WRITE_ENABLE.as_raw() }>,

        #[serde(default)]
        pub depth_compare_op:
            RequiredDynamicState<String, { vk::DynamicState::DEPTH_COMPARE_OP.as_raw() }>,

        #[serde(default)]
        pub depth_bounds_test_enable:
            RequiredDynamicState<bool, { vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE.as_raw() }>,

        #[serde(default)]
        pub stencil_test_enable:
            RequiredDynamicState<bool, { vk::DynamicState::STENCIL_TEST_ENABLE.as_raw() }>,

        #[serde(default)]
        pub front_stencil: Option<StencilState>,

        #[serde(default)]
        pub back_stencil: Option<StencilState>,

        /// min, max
        #[serde(default)]
        pub depth_bounds:
            OptionalDynamicState<(f32, f32), { vk::DynamicState::DEPTH_BOUNDS.as_raw() }>,

        #[serde(default)]
        pub blend_logic_op_enable:
            RequiredDynamicState<bool, { vk::DynamicState::LOGIC_OP_ENABLE_EXT.as_raw() }>,

        #[serde(default)]
        pub blend_logic_op:
            OptionalDynamicState<String, { vk::DynamicState::LOGIC_OP_EXT.as_raw() }>,

        pub attachments: Vec<Attachment>,

        #[serde(default)]
        pub depth_format: pumicite::utils::format::Format,

        #[serde(default)]
        pub stencil_format: pumicite::utils::format::Format,

        #[serde(default)]
        pub blend_constants:
            RequiredDynamicState<[f32; 4], { vk::DynamicState::BLEND_CONSTANTS.as_raw() }>,

        #[serde(default)]
        pub layout: PipelineLayoutRef,
    }

    /// A "patch" to a [`GraphicsPipeline`] that allows the creation of a runtime variant of an existing
    /// pipeline. Generally, runtime modification of pipeline states should be done with
    /// [dynamic states](vk::DynamicState). However, a small amount of states cannot be modified
    /// with dynamic states. This struct provides a workaround by allowing the user to specify these
    /// values by providing them programmatically at pipeline creation time.
    ///
    /// If a pipeline state is not known statically but you also cannot find it here,
    /// it should probably be modified with [dynamic states](vk::DynamicState).
    /// This helps reducing pipeline variant counts and allows the driver to select the best approach
    /// to modify the attribute.
    ///
    /// If the state you want to modify cannot be modified with dynamic state or that certain
    /// drivers implement that dynamic state suboptimally, re-evalue
    /// your use case. If you're confident that this is something you absolutely need, open a PR
    /// and present your use case so that we can keep those use cases well-documented.
    #[derive(Serialize, Deserialize, Default, Clone)]
    pub struct GraphicsPipelineVariant {
        #[serde(default)]
        pub shaders: BTreeMap<ShaderStage, BTreeMap<u32, SpecializationConstantType>>,

        #[serde(default)]
        pub depth_format: Option<pumicite::utils::format::Format>,

        #[serde(default)]
        pub stencil_format: Option<pumicite::utils::format::Format>,

        #[serde(default)]
        pub color_formats: BTreeMap<u32, pumicite::utils::format::Format>,
    }
    impl GraphicsPipelineVariant {
        pub fn apply_on(&self, pipeline: &mut GraphicsPipeline) {
            for (shader_stage, specialization_constants) in self.shaders.iter() {
                if let Some(shader) = pipeline.shaders.get_mut(shader_stage) {
                    shader
                        .specialization_constants
                        .append(&mut specialization_constants.clone());
                }
            }
            if let Some(depth_format) = self.depth_format {
                pipeline.depth_format = depth_format;
            }
            if let Some(stencil_format) = self.stencil_format {
                pipeline.stencil_format = stencil_format;
            }
            for (index, format) in self.color_formats.iter() {
                pipeline.attachments[*index as usize].format = *format;
            }
        }
    }

    #[derive(Serialize, Deserialize)]
    pub struct Attachment {
        pub blend_enable:
            RequiredDynamicState<bool, { vk::DynamicState::COLOR_BLEND_ENABLE_EXT.as_raw() }>,

        #[serde(default)]
        pub blend_equation: OptionalDynamicState<
            BlendEquation,
            { vk::DynamicState::COLOR_BLEND_EQUATION_EXT.as_raw() },
        >,

        #[serde(default)]
        pub color_write_mask:
            OptionalDynamicState<String, { vk::DynamicState::COLOR_WRITE_MASK_EXT.as_raw() }>,

        pub format: pumicite::utils::format::Format,
    }
    #[derive(Serialize, Deserialize)]
    pub struct BlendEquation {
        pub color: (BlendFactor, BlendOp, BlendFactor),
        pub alpha: (BlendFactor, BlendOp, BlendFactor),
    }

    #[derive(Serialize, Deserialize, Clone, Copy)]
    pub enum BlendFactor {
        Zero = 0,
        One = 1,
        SrcColor = 2,
        OneMinusSrcColor = 3,
        DstColor = 4,
        OneMinusDstColor = 5,
        SrcAlpha = 6,
        OneMinusSrcAlpha = 7,
        DstAlpha = 8,
        OneMinusDstAlpha = 9,
        ConstantColor = 10,
        OneMinusConstantColor = 11,
        ConstantAlpha = 12,
        OneMinusConstantAlpha = 13,
        SrcAlphaSaturate = 14,
        Src1Color = 15,
        OneMinusSrc1Color = 16,
        Src1Alpha = 17,
        OneMiusSrc1Alpha = 18,
    }
    impl From<BlendFactor> for vk::BlendFactor {
        fn from(value: BlendFactor) -> Self {
            vk::BlendFactor::from_raw(value as i32)
        }
    }

    #[derive(Serialize, Deserialize, Clone, Copy)]
    pub enum BlendOp {
        Add = 0,
        Subtract = 1,
        ReverseSubtract = 2,
        Min = 3,
        Max = 4,
    }
    impl From<BlendOp> for vk::BlendOp {
        fn from(value: BlendOp) -> Self {
            vk::BlendOp::from_raw(value as i32)
        }
    }

    #[derive(Serialize, Deserialize, Clone, Copy)]
    pub enum StencilOp {
        Keep = 0,
        Zero = 1,
        Replace = 2,
        IncrementAndClamp = 3,
        DecrementAndClamp = 4,
        Invert = 5,
        IncrementAndWrap = 6,
        DecrementAndWrap = 7,
    }
    impl From<StencilOp> for vk::StencilOp {
        fn from(value: StencilOp) -> Self {
            vk::StencilOp::from_raw(value as i32)
        }
    }

    #[derive(Serialize, Deserialize)]
    pub struct StencilState {
        pub ops: RequiredDynamicState<StencilStateOps, { vk::DynamicState::STENCIL_OP.as_raw() }>,
        pub compare_mask:
            RequiredDynamicState<u32, { vk::DynamicState::STENCIL_COMPARE_MASK.as_raw() }>,
        pub write_mask:
            RequiredDynamicState<u32, { vk::DynamicState::STENCIL_WRITE_MASK.as_raw() }>,
        pub reference: RequiredDynamicState<u32, { vk::DynamicState::STENCIL_REFERENCE.as_raw() }>,
    }
    #[derive(Serialize, Deserialize)]
    pub struct StencilStateOps {
        pub fail: StencilOp,
        pub pass: StencilOp,
        pub depth_fail: StencilOp,
        pub compare: String,
    }

    #[derive(Serialize, Deserialize)]
    pub struct DepthBias {
        pub constant_factor: f32,
        pub clamp: f32,
        pub slope_factor: f32,
    }

    #[derive(Default, Serialize, Deserialize, Clone, Copy)]
    pub enum PolygonMode {
        #[default]
        Fill = 0,
        Line = 1,
        Point = 2,
    }
    impl From<PolygonMode> for vk::PolygonMode {
        fn from(value: PolygonMode) -> Self {
            vk::PolygonMode::from_raw(value as i32)
        }
    }

    #[derive(Default, Serialize, Deserialize, Clone, Copy)]
    pub enum CullMode {
        #[default]
        None,
        /// Back-facing triangles are discarded
        Back,
        /// front-facing triangles are discarded
        Front,
        /// All triangles are discarded
        FrontAndBack,
    }
    impl From<CullMode> for vk::CullModeFlags {
        fn from(value: CullMode) -> Self {
            match value {
                CullMode::None => vk::CullModeFlags::NONE,
                CullMode::Back => vk::CullModeFlags::BACK,
                CullMode::Front => vk::CullModeFlags::FRONT,
                CullMode::FrontAndBack => vk::CullModeFlags::FRONT_AND_BACK,
            }
        }
    }

    #[derive(Default, Serialize, Deserialize, Clone, Copy)]
    pub enum FrontFace {
        #[default]
        CounterClockwise,
        Clockwise,
    }
    impl From<FrontFace> for vk::FrontFace {
        fn from(value: FrontFace) -> Self {
            match value {
                FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
                FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
            }
        }
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct Viewport {
        pub x: f32,
        pub y: f32,
        pub width: f32,
        pub height: f32,
        pub min_depth: f32,
        pub max_depth: f32,
    }

    impl From<Viewport> for vk::Viewport {
        fn from(value: Viewport) -> Self {
            vk::Viewport {
                x: value.x,
                y: value.y,
                width: value.width,
                height: value.height,
                min_depth: value.min_depth,
                max_depth: value.max_depth,
            }
        }
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct Rect2D {
        pub offset: (i32, i32),
        pub extent: (u32, u32),
    }
    impl From<Rect2D> for vk::Rect2D {
        fn from(value: Rect2D) -> Self {
            Self {
                offset: vk::Offset2D {
                    x: value.offset.0,
                    y: value.offset.1,
                },
                extent: vk::Extent2D {
                    width: value.extent.0,
                    height: value.extent.1,
                },
            }
        }
    }

    #[derive(Serialize, Deserialize, Default, Clone, Copy)]
    pub enum PrimitiveTopology {
        PointList = 0,
        LineList = 1,
        LineStrip = 2,
        #[default]
        TriangleList = 3,
        TriangleStrip = 4,
        TriangleFan = 5,
        LineListWithAdjacency = 6,
        LineStripWithAdjacency = 7,
        TriangleListWithAdjacency = 8,
        TriangleStripWithAdjacency = 9,
        PatchList = 10,
    }
    impl From<PrimitiveTopology> for vk::PrimitiveTopology {
        fn from(value: PrimitiveTopology) -> Self {
            vk::PrimitiveTopology::from_raw(value as i32)
        }
    }

    #[derive(Serialize, Deserialize)]
    pub struct VertexInputBinding {
        pub stride:
            RequiredDynamicState<u32, { vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE.as_raw() }>,
        pub input_rate: VertexInputRate,
        pub attributes: BTreeMap<u32, VertexInputAttributes>,
    }
    #[derive(Serialize, Deserialize, Clone)]
    pub struct VertexInputAttributes {
        pub format: pumicite::utils::format::Format,
        pub offset: u32,
    }
    #[derive(Serialize, Deserialize, Clone, Copy)]
    pub enum VertexInputRate {
        Vertex,
        Instance,
    }
    impl From<VertexInputRate> for vk::VertexInputRate {
        fn from(value: VertexInputRate) -> Self {
            match value {
                VertexInputRate::Vertex => vk::VertexInputRate::VERTEX,
                VertexInputRate::Instance => vk::VertexInputRate::INSTANCE,
            }
        }
    }

    #[derive(Serialize, Deserialize)]
    pub struct Shader {
        /// Path to the shader asset
        pub path: String,
        /// The entry point name of the shader for this stage.
        pub entry_point: String,

        /// Specialization constants for this shader.
        #[serde(default)]
        pub specialization_constants: BTreeMap<u32, SpecializationConstantType>,

        /// Specifies that `SubgroupSize` may vary in the shader stage
        #[serde(default)]
        pub allow_varying_subgroup: bool,
        /// Specifies that the subgroup sizes must be launched with all invocations active
        /// in the task, mesh, or compute stage.
        #[serde(default)]
        pub require_full_subgroups: bool,
    }
    impl Shader {
        pub fn flags(&self) -> vk::PipelineShaderStageCreateFlags {
            let mut flags = vk::PipelineShaderStageCreateFlags::empty();
            if self.require_full_subgroups {
                flags |= vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS;
            }
            if self.allow_varying_subgroup {
                flags |= vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE;
            }
            flags
        }
    }

    #[derive(Serialize, Deserialize)]
    pub struct PipelineLayout {
        #[serde(default)]
        pub sets: Vec<DescriptorSetLayoutRef>,

        #[serde(default)]
        pub push_constants: BTreeMap<ShaderStage, (u32, u32)>,

        /// Provided by VK_EXT_graphics_pipeline_library
        /// Specifies that implementations must ensure that the properties and/or absence of a particular descriptor set
        /// do not influence any other properties of the pipeline layout. This allows pipelines libraries linked without
        /// VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT to be created with a subset of the total descriptor sets.
        #[serde(default)]
        pub independent_sets: bool,
    }

    #[derive(Serialize, Deserialize, Default)]
    pub enum PipelineLayoutRef {
        Inline(PipelineLayout),
        Path(String),
        #[default]
        Bindless,
    }

    #[derive(Serialize, Deserialize)]
    pub enum DescriptorSetLayoutRef {
        Inline(DescriptorSetLayout),
        Path(String),
        ResourceHeap,
        SamplerHeap,
    }

    fn binding_count_default() -> u32 {
        1
    }
    fn sample_count_default()
    -> RequiredDynamicState<u8, { vk::DynamicState::RASTERIZATION_SAMPLES_EXT.as_raw() }> {
        RequiredDynamicState::Static(1)
    }
    fn viewport_default() -> CountedDynamicState<Viewport> {
        CountedDynamicState::Count(1)
    }
    fn scissor_default() -> CountedDynamicState<Rect2D> {
        CountedDynamicState::Count(1)
    }
    fn line_width_default() -> RequiredDynamicState<f32, { vk::DynamicState::LINE_WIDTH.as_raw() }>
    {
        RequiredDynamicState::Static(1.0)
    }

    #[derive(Serialize, Deserialize)]
    pub struct Binding {
        /// The type of resource descriptors that are used for this binding.
        pub ty: DescriptorType,
        pub binding: u32,
        /// The number of descriptors contained in the binding, accessed in a shader as an array, except if
        /// [`Binding::ty`] is [`DescriptorType::InlineUniformBlock`] in which case `count` is the size in bytes of the inline uniform block.
        #[serde(default = "binding_count_default")]
        pub count: u32,
        /// Pipeline shader stages that can access a resource for this binding.
        pub stages: Vec<ShaderStage>,

        #[serde(default)]
        pub samplers: (),

        /// Specifies that if descriptors in this binding are updated between when the descriptor set is bound in a command buffer
        /// and when that command buffer is submitted to a queue, then the submission will use the most recently set descriptors for
        /// this binding and the updates do not invalidate the command buffer.
        #[serde(default)]
        pub update_after_bind: bool,

        /// Specifies that descriptors in this binding that are not dynamically used need not contain valid descriptors at the time
        /// the descriptors are consumed. A descriptor is dynamically used if any shader invocation executes an instruction that
        /// performs any memory access using the descriptor. If a descriptor is not dynamically used, any resource referenced by
        /// the descriptor is not considered to be referenced during command execution.
        #[serde(default)]
        pub update_unused_while_pending: bool,

        /// Specifies that descriptors in this binding that are not dynamically used need not contain valid descriptors at the time
        /// the descriptors are consumed. A descriptor is dynamically used if any shader invocation executes an instruction that performs
        /// any memory access using the descriptor. If a descriptor is not dynamically used, any resource referenced by the descriptor is not
        /// considered to be referenced during command execution.
        #[serde(default)]
        pub partially_bound: bool,

        /// Specifies that this is a variable-sized descriptor binding whose size will be specified when a descriptor set is allocated
        /// using this layout. The value of descriptorCount is treated as an upper bound on the size of the binding. This must only be used
        /// for the last binding in the descriptor set layout (i.e. the binding with the largest value of binding).
        #[serde(default)]
        pub variable_descriptor_count: bool,
    }
    #[derive(Serialize, Deserialize)]
    pub struct DescriptorSetLayout {
        /// Specifies that descriptor sets must not be allocated using this layout, and descriptors are instead pushed by vkCmdPushDescriptorSet.
        #[serde(default)]
        pub push_descriptor: bool,

        /// Specifies that descriptor sets using this layout must be allocated from a descriptor pool created with the
        /// VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT bit set.
        #[serde(default)]
        pub update_after_bind_pool: bool,

        /// Specifies that this layout must only be used with descriptor buffers.
        #[serde(default)]
        pub descriptor_buffer: bool,

        /// Descriptor bindings
        pub bindings: Vec<Binding>,
    }

    #[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
    pub enum DescriptorType {
        Sampler,
        CombinedImageSampler,
        SampledImage,
        StorageImage,
        UniformTexelBuffer,
        StorageTexelBuffer,
        UniformBuffer,
        StorageBuffer,
        UniformBufferDynamic,
        StorageBufferDynamic,
        InputAttachment,
        InlineUniformBlock,
        AccelerationStructure,
        Mutable,
    }
    impl From<DescriptorType> for pumicite::ash::vk::DescriptorType {
        fn from(value: DescriptorType) -> Self {
            match value {
                DescriptorType::Sampler => vk::DescriptorType::SAMPLER,
                DescriptorType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                DescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
                DescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
                DescriptorType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                DescriptorType::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                DescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                DescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
                DescriptorType::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                DescriptorType::StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                DescriptorType::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
                DescriptorType::InlineUniformBlock => vk::DescriptorType::INLINE_UNIFORM_BLOCK,
                DescriptorType::AccelerationStructure => {
                    vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
                }
                DescriptorType::Mutable => vk::DescriptorType::MUTABLE_EXT,
            }
        }
    }

    #[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
    pub enum ShaderStage {
        Vertex,
        TessellationControl,
        TessellationEvaluation,
        Geometry,
        Fragment,
        Compute,

        RayGen,
        AnyHit,
        ClosestHit,
        Miss,
        Intersection,
        Callable,

        Task,
        Mesh,
    }
    impl From<ShaderStage> for pumicite::ash::vk::ShaderStageFlags {
        fn from(value: ShaderStage) -> Self {
            match value {
                ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
                ShaderStage::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
                ShaderStage::TessellationEvaluation => {
                    vk::ShaderStageFlags::TESSELLATION_EVALUATION
                }
                ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
                ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
                ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
                ShaderStage::RayGen => vk::ShaderStageFlags::RAYGEN_KHR,
                ShaderStage::AnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
                ShaderStage::ClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                ShaderStage::Miss => vk::ShaderStageFlags::MISS_KHR,
                ShaderStage::Intersection => vk::ShaderStageFlags::INTERSECTION_KHR,
                ShaderStage::Callable => vk::ShaderStageFlags::CALLABLE_KHR,
                ShaderStage::Task => vk::ShaderStageFlags::TASK_EXT,
                ShaderStage::Mesh => vk::ShaderStageFlags::MESH_EXT,
            }
        }
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub enum SpecializationConstantType {
        UInt32(u32),
        UInt16(u16),
        UInt8(u8),
        Int32(i32),
        Int16(i16),
        Int8(i8),
        Bool(bool),
        Float(f32),
    }
    impl SpecializationConstantType {
        pub fn extend(&self, data: &mut Vec<u8>) -> usize {
            match self {
                SpecializationConstantType::UInt32(v) => {
                    data.extend_from_slice(&v.to_ne_bytes());
                    4
                }
                SpecializationConstantType::UInt16(v) => {
                    data.extend_from_slice(&v.to_ne_bytes());
                    2
                }
                SpecializationConstantType::UInt8(v) => {
                    data.push(*v);
                    1
                }
                SpecializationConstantType::Int32(v) => {
                    data.extend_from_slice(&v.to_ne_bytes());
                    4
                }
                SpecializationConstantType::Int16(v) => {
                    data.extend_from_slice(&v.to_ne_bytes());
                    2
                }
                SpecializationConstantType::Int8(v) => {
                    data.push(*v as u8);
                    1
                }
                SpecializationConstantType::Bool(v) => {
                    let vk_bool = if *v { 1u32 } else { 0u32 };
                    data.extend_from_slice(&vk_bool.to_ne_bytes());
                    4
                }
                SpecializationConstantType::Float(v) => {
                    data.extend_from_slice(&v.to_ne_bytes());
                    4
                }
            }
        }
    }

    fn default_max_ray_recursion_depth() -> u32 {
        1
    }
    #[derive(Serialize, Deserialize)]
    pub struct RayTracingPipeline {
        /// The compute shader used for compiling the compute pipeline
        pub stages: Vec<RayTracingPipelineShaderStage>,

        #[serde(default)]
        /// Shaders that could be referenced by hitgroups
        pub shaders: BTreeMap<String, Shader>,

        /// Path to the pipeline layout
        #[serde(default)]
        pub layout: PipelineLayoutRef,

        #[serde(default = "default_max_ray_recursion_depth")]
        pub max_ray_recursion_depth: u32,

        pub max_ray_payload_size: u32,
        pub max_hit_attribute_size: u32,

        #[serde(default)]
        pub dynamic_stack_size: bool,

        /// The created pipeline will not be optimized. Using this flag may reduce the time
        /// taken to create the pipeline.
        #[serde(default)]
        pub disable_optimization: bool,

        /// The compute pipeline can be used with vkCmdDispatchBase with a non-zero base workgroup.
        #[serde(default)]
        pub dispatch_base: bool,
    }

    #[derive(Default, Serialize, Deserialize)]
    pub enum RayTracingPipelineShaderHitGroupType {
        #[default]
        Triangles,
        Aabbs,
    }

    #[derive(Serialize, Deserialize)]
    pub enum RayTracingPipelineShaderStage {
        RayGen {
            shader: Shader,
            #[serde(default)]
            param_size: u32,
        },
        Miss {
            shader: Shader,
            #[serde(default)]
            param_size: u32,
        },
        Callable {
            shader: Shader,
            #[serde(default)]
            param_size: u32,
        },
        HitGroup {
            #[serde(default)]
            ty: RayTracingPipelineShaderHitGroupType,
            #[serde(default)]
            intersection: HitgroupShader,
            #[serde(default)]
            any_hit: HitgroupShader,
            #[serde(default)]
            closest_hit: HitgroupShader,
            #[serde(default)]
            param_size: u32,
        },
    }
    #[derive(Serialize, Deserialize, Default)]
    #[serde(untagged)]
    pub enum HitgroupShader {
        Reused(String),
        Singleton(Shader),
        #[default]
        None,
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn test_deserialize() {
            let ron = "
                ComputePipeline(
                    shader: Shader(
                        path: \"aaa.spv\",
                        entry_point: \"main\",
                    ),
                    layout: \"mylayout.playout.ron\"
                )
            ";
            let _pipeline: ComputePipeline = ron::de::from_str(ron).unwrap();

            let ron = "
                RayTracingPipeline(
                    layout: \"mylayout.playout.ron\",
                    max_ray_payload_size: 0,
                    max_hit_attribute_size: 0,

                    shaders: [
                        RayGen((
                            path: \"aaa.spv\",
                            entry_point: \"main\",
                        )),
                        HitGroup(
                            closest_hit: Shader(
                                path: \"aaa.spv\",
                                entry_point: \"main\",
                            ),
                            any_hit: \"MyShader\",
                        )
                    ]
                )
            ";
            let _pipeline: RayTracingPipeline = ron::de::from_str(ron).unwrap();
        }
    }
}
