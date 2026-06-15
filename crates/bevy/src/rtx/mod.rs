use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    sync::Arc,
};

use bevy_app::{Plugin, PostUpdate, Startup};
use bevy_asset::{
    Asset, AssetApp, AssetEvent, AssetHandleProvider, AssetId, AssetServer, Assets, Handle,
};
use bevy_ecs::prelude::*;
use bevy_reflect::TypePath;
use pumicite::{
    HasDevice,
    ash::vk,
    bevy::PipelineCache,
    device::DeviceBuilder,
    pipeline::Pipeline,
    rtx::{RayTracingPipelineLibraryCreateInfo, SbtLayout, SbtRemapper, ShaderBindingTable},
};

use crate::{CreateDevice, shader::RayTracingPipelineLibrary};
pub mod blas;
pub mod tlas;

pub struct RtxPipelinePlugin;
impl Plugin for RtxPipelinePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(PostUpdate, build_rtx_pipeline_system);
        app.init_asset::<RayTracingPipeline>()
            .init_asset::<RayTracingPipelineLibrary>();
        #[cfg(any(feature = "ron", feature = "postcard"))]
        app.preregister_asset_loader::<crate::shader::RayTracingPipelineLoader>(&[
            #[cfg(feature = "ron")]
            "rtx.pipeline.ron",
            #[cfg(feature = "postcard")]
            "rtx.pipeline.bin",
        ]);
        app.init_resource::<RtxPipelineManager>();

        app.add_systems(
            Startup,
            (
                (|mut device_builder: ResMut<DeviceBuilder>| {
                    device_builder
                        .enable_extension::<pumicite::ash::khr::acceleration_structure::Meta>()
                        .unwrap();
                    device_builder
                        .enable_extension::<pumicite::ash::khr::ray_tracing_pipeline::Meta>()
                        .unwrap();
                    device_builder
                        .enable_extension::<pumicite::ash::khr::ray_tracing_maintenance1::Meta>()
                        .ok();
                    device_builder
                        .enable_extension::<pumicite::ash::khr::pipeline_library::Meta>()
                        .ok();
                    device_builder
                        .enable_feature(
                            |rtx_features: &mut vk::PhysicalDeviceAccelerationStructureFeaturesKHR| {
                                &mut rtx_features.acceleration_structure
                            },
                        )
                        .unwrap();
                    device_builder
                        .enable_feature(
                            |rtx_features: &mut vk::PhysicalDeviceRayTracingPipelineFeaturesKHR| {
                                &mut rtx_features.ray_tracing_pipeline
                            },
                        )
                        .unwrap();
                    device_builder
                        .enable_feature(
                            |rtx_features: &mut vk::PhysicalDeviceHostQueryResetFeatures| {
                                &mut rtx_features.host_query_reset // For ray tracing AS compaction size query
                            },
                        )
                        .unwrap();
                })
                .before(CreateDevice),
                #[cfg(any(feature = "ron", feature = "postcard"))]
                (|world: &mut World| {
                    let asset_server = world
                        .remove_resource::<AssetServer>()
                        .expect("Requires asset server");
                    asset_server
                        .register_loader(crate::shader::RayTracingPipelineLoader::from_world(world));
                    world.insert_resource(asset_server);
                })
                .after(CreateDevice),
            ),
        );
    }
}

#[derive(Asset, TypePath, Clone)]
pub struct RayTracingPipeline {
    inner: Arc<Pipeline>,
    layout: SbtLayout,
    remapper: SbtRemapper,
}
impl Deref for RayTracingPipeline {
    type Target = Arc<Pipeline>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl RayTracingPipeline {
    pub fn create_sbt(&self, reusing_sbt: Option<ShaderBindingTable>) -> ShaderBindingTable {
        ShaderBindingTable::new_with_mapper(
            &self.inner,
            self.layout.clone(),
            &self.remapper,
            reusing_sbt,
        )
    }
}

#[derive(Resource)]
pub struct RtxPipelineManager {
    handle_provider: AssetHandleProvider,
    pipelines: BTreeMap<Handle<RayTracingPipeline>, ManagedRtxPipeline>,
}
impl FromWorld for RtxPipelineManager {
    fn from_world(world: &mut World) -> Self {
        Self {
            pipelines: BTreeMap::new(),
            handle_provider: world
                .resource::<Assets<RayTracingPipeline>>()
                .get_handle_provider(),
        }
    }
}
struct ManagedRtxPipeline {
    /// Basically an id allocator for library ids. Supporting up to 64 libraries for now.
    library_mask: u64,
    libraries: Vec<Option<Handle<RayTracingPipelineLibrary>>>,
    library_dependencies: BTreeSet<AssetId<RayTracingPipelineLibrary>>,
}
impl RtxPipelineManager {
    pub fn add_pipeline(&mut self) -> Handle<RayTracingPipeline> {
        let handle: Handle<RayTracingPipeline> = self.handle_provider.reserve_handle().typed();
        let managed_pipeline = ManagedRtxPipeline {
            libraries: Vec::new(),
            library_dependencies: BTreeSet::new(),
            library_mask: 0,
        };
        self.pipelines.insert(handle.clone(), managed_pipeline);
        handle
    }
    /// Returns the library index
    pub fn add_library_for_pipeline(
        &mut self,
        pipeline: &Handle<RayTracingPipeline>,
        library: Handle<RayTracingPipelineLibrary>,
    ) -> u16 {
        let library_id = library.id();
        let managed_pipeline = self
            .pipelines
            .get_mut(pipeline)
            .expect("Base pipeline was never added to the RtxPipelineManager");
        let index = managed_pipeline.library_mask.trailing_ones() as u16;
        if index >= managed_pipeline.libraries.len() as u16 {
            assert_eq!(index, managed_pipeline.libraries.len() as u16);
            managed_pipeline.libraries.push(Some(library));
        } else {
            assert!(managed_pipeline.libraries[index as usize].is_none());
            managed_pipeline.libraries[index as usize] = Some(library);
        }
        managed_pipeline.library_mask |= 1 << index;
        managed_pipeline.library_dependencies.insert(library_id);
        index
    }
    pub fn remove_library_for_pipeline(
        &mut self,
        pipeline: &Handle<RayTracingPipeline>,
        library_id: u32,
    ) -> Option<Handle<RayTracingPipelineLibrary>> {
        let managed_pipeline = self
            .pipelines
            .get_mut(pipeline)
            .expect("Pipeline was never added to the RtxPipelineManager");
        let Some(library) = managed_pipeline.libraries[library_id as usize].take() else {
            return None;
        };
        managed_pipeline.library_dependencies.remove(&library.id());
        managed_pipeline.library_mask &= !(1 << library_id);
        Some(library)
    }
}

pub fn build_rtx_pipeline_system(
    manager: ResMut<RtxPipelineManager>,
    asset_server: Res<AssetServer>,
    library_assets: Res<Assets<RayTracingPipelineLibrary>>,
    mut rtx_pipeline_update_event: MessageReader<AssetEvent<RayTracingPipelineLibrary>>,
    pipeline_cache: Res<PipelineCache>,
) {
    // Find all pipeline libraries that have updated / removed
    let pipeline_libraries_updated: BTreeSet<AssetId<RayTracingPipelineLibrary>> =
        rtx_pipeline_update_event
            .read()
            .filter_map(|event| match event {
                AssetEvent::Added { id } | AssetEvent::Modified { id } => Some(*id),
                _ => None,
            })
            .collect();
    // For each rtx pipeline, check whether they have some libraries that have updated
    for (handle, pipeline) in manager.pipelines.iter() {
        let id = handle.id();
        if pipeline
            .library_dependencies
            .intersection(&pipeline_libraries_updated)
            .next()
            .is_none()
        {
            continue;
        }
        let Some(libraries) = pipeline
            .libraries
            .iter()
            .filter_map(Option::as_ref) // Filter out empty slots
            .map(|x| library_assets.get(x))
            .collect::<Option<Vec<&RayTracingPipelineLibrary>>>()
        else {
            continue;
        };
        tracing::info!("Scheduling pipeline {} for update", id);
        let mut remapper = SbtRemapper::default();
        for lib in pipeline.libraries.iter() {
            if let Some(lib) = lib {
                let lib = library_assets.get(lib).unwrap(); // If this was None, we would have made an early-out earlier.
                remapper.push_library(
                    lib.sbt_layout.raygen.count as u16
                        + lib.sbt_layout.callable.count as u16
                        + lib.sbt_layout.hitgroup.count as u16
                        + lib.sbt_layout.miss.count as u16,
                );
            } else {
                remapper.push_library(0);
            }
        }
        let mut max_ray_recursion_depth = 0;
        let mut max_ray_payload_size = 0;
        let mut max_hit_attribute_size = 0;
        let mut dynamic_stack_size = false;
        let mut sbt_layout = SbtLayout::new(pipeline_cache.device());
        for library in libraries.iter() {
            max_ray_recursion_depth = max_ray_recursion_depth.max(library.max_ray_recursion_depth);
            max_ray_payload_size = max_ray_payload_size.max(library.max_ray_payload_size);
            max_hit_attribute_size = max_hit_attribute_size.max(library.max_hit_attribute_size);
            dynamic_stack_size |= library.dynamic_stack_size;
            // Every linked library contributes its own raygen/miss/callable/hitgroup
            // groups to the final pipeline, so all four stage counts must be summed
            // here. The total drives how many shader group handles are fetched, and
            // it must match the ranges produced by `SbtRemapper`.
            sbt_layout.raygen.count += library.sbt_layout.raygen.count;
            sbt_layout.raygen.param_size = sbt_layout
                .raygen
                .param_size
                .max(library.sbt_layout.raygen.param_size);
            sbt_layout.miss.count += library.sbt_layout.miss.count;
            sbt_layout.miss.param_size = sbt_layout
                .miss
                .param_size
                .max(library.sbt_layout.miss.param_size);
            sbt_layout.callable.count += library.sbt_layout.callable.count;
            sbt_layout.callable.param_size = sbt_layout
                .callable
                .param_size
                .max(library.sbt_layout.callable.param_size);
            sbt_layout.hitgroup.count += library.sbt_layout.hitgroup.count;
            sbt_layout.hitgroup.param_size = sbt_layout
                .hitgroup
                .param_size
                .max(library.sbt_layout.hitgroup.param_size);
        }
        let pipeline_cache = pipeline_cache.clone();
        let build_monolithic = libraries.iter().all(|lib| lib.is_monolithic());
        let build_linked = libraries.iter().all(|lib| !lib.is_monolithic());
        assert!(build_monolithic != build_linked);

        if build_monolithic {
            // Inline (emulated) path: merge all shaders and groups into one monolithic pipeline.
            let mut merged_shaders = Vec::new();
            let mut merged_groups = Vec::new();
            let mut layout = None;
            let mut flags = vk::PipelineCreateFlags::empty();
            for lib in &libraries {
                let (lib_flags, lib_layout, lib_shaders, lib_groups) = lib.inline_data();
                let shader_offset = merged_shaders.len() as u32;
                flags = lib_flags;
                layout = Some(lib_layout.clone());
                merged_shaders.extend(lib_shaders.iter().cloned());
                // Rebase shader indices in groups by the current shader offset.
                for group in lib_groups {
                    let mut group = *group;
                    if group.general_shader != vk::SHADER_UNUSED_KHR {
                        group.general_shader += shader_offset;
                    }
                    if group.closest_hit_shader != vk::SHADER_UNUSED_KHR {
                        group.closest_hit_shader += shader_offset;
                    }
                    if group.any_hit_shader != vk::SHADER_UNUSED_KHR {
                        group.any_hit_shader += shader_offset;
                    }
                    if group.intersection_shader != vk::SHADER_UNUSED_KHR {
                        group.intersection_shader += shader_offset;
                    }
                    merged_groups.push(group);
                }
            }
            let layout = layout.unwrap();
            // Remove LIBRARY_KHR flag since this is a final monolithic pipeline.
            let flags = flags & !vk::PipelineCreateFlags::LIBRARY_KHR;
            asset_server.update_async::<RayTracingPipeline, vk::Result>(handle, async move {
                let rtx_pipeline = pipeline_cache.create_ray_tracing_pipeline_monolithic(
                    RayTracingPipelineLibraryCreateInfo {
                        flags,
                        layout,
                        max_ray_recursion_depth,
                        max_ray_payload_size,
                        max_hit_attribute_size,
                        dynamic_stack_size,
                        shaders: &merged_shaders,
                        groups: &merged_groups,
                    },
                )?;
                tracing::info!("Pipeline {} updated (inline)", id);
                Ok(RayTracingPipeline {
                    inner: Arc::new(rtx_pipeline),
                    layout: sbt_layout,
                    remapper,
                })
            });
        } else if build_linked {
            // Library path: link pre-compiled pipeline libraries.
            let libraries = libraries
                .iter()
                .map(|lib| lib.pipeline().clone())
                .collect::<Vec<Arc<Pipeline>>>();
            asset_server.update_async::<RayTracingPipeline, vk::Result>(handle, async move {
                let rtx_pipeline = pipeline_cache.create_ray_tracing_pipeline(
                    libraries,
                    max_ray_recursion_depth,
                    max_ray_payload_size,
                    max_hit_attribute_size,
                    dynamic_stack_size,
                )?;
                tracing::info!("Pipeline {} updated", id);
                Ok(RayTracingPipeline {
                    inner: Arc::new(rtx_pipeline),
                    layout: sbt_layout,
                    remapper,
                })
            });
        } else {
            // Has a mix of linked / monolithic libs
            panic!()
        }
    }
}
