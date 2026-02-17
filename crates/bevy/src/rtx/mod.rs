use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    sync::Arc,
};

use bevy_app::{Plugin, PostUpdate};
use bevy_asset::{
    Asset, AssetApp, AssetEvent, AssetHandleProvider, AssetId, AssetServer, Assets, Handle,
};
use bevy_ecs::prelude::*;
use bevy_reflect::TypePath;
use pumicite::{
    ash::vk,
    bevy::PipelineCache,
    pipeline::Pipeline,
    rtx::{RayTracingPipelineLibraryCreateInfo, SbtLayout, ShaderBindingTable},
};

use crate::{PumiciteApp, shader::RayTracingPipelineLibrary};
pub mod blas;
pub mod tlas;

pub struct RtxPipelinePlugin;
impl Plugin for RtxPipelinePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(PostUpdate, build_rtx_pipeline_system);
        app.init_asset::<RayTracingPipeline>()
            .init_asset::<RayTracingPipelineLibrary>()
            .preregister_asset_loader::<crate::shader::RayTracingPipelineLoader>(&[
                "rtx.pipeline.ron",
            ]);

        app.add_device_extension::<pumicite::ash::khr::acceleration_structure::Meta>()
            .unwrap();
        app.add_device_extension::<pumicite::ash::khr::ray_tracing_pipeline::Meta>()
            .unwrap();
        app.add_device_extension::<pumicite::ash::khr::ray_tracing_maintenance1::Meta>()
            .ok();
        app.add_device_extension::<pumicite::ash::khr::pipeline_library::Meta>()
            .ok();

        app.enable_feature(
            |rtx_features: &mut vk::PhysicalDeviceAccelerationStructureFeaturesKHR| {
                &mut rtx_features.acceleration_structure
            },
        )
        .unwrap();
        app.enable_feature(
            |rtx_features: &mut vk::PhysicalDeviceRayTracingPipelineFeaturesKHR| {
                &mut rtx_features.ray_tracing_pipeline
            },
        )
        .unwrap();
    }
    fn cleanup(&self, app: &mut bevy_app::App) {
        app.init_resource::<RtxPipelineManager>()
            .init_asset_loader::<crate::shader::RayTracingPipelineLoader>();
    }
}

#[derive(Asset, TypePath, Clone)]
pub struct RayTracingPipeline {
    inner: Arc<Pipeline>,
    layout: SbtLayout,
    hitgroup_mask: u64,
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
            self.hitgroup_mask,
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
    base_library: Handle<RayTracingPipelineLibrary>,
    /// Basically an id allocator for hitgroup ids. Supporting up to 64 hitgroups for now.
    hitgroup_mask: u64,
    hitgroups: Vec<Option<Handle<RayTracingPipelineLibrary>>>,
    library_dependencies: BTreeSet<AssetId<RayTracingPipelineLibrary>>,
}
impl RtxPipelineManager {
    pub fn add_pipeline(
        &mut self,
        base_library: Handle<RayTracingPipelineLibrary>,
    ) -> Handle<RayTracingPipeline> {
        let handle: Handle<RayTracingPipeline> = self.handle_provider.reserve_handle().typed();
        let base_library_id = base_library.id();
        let mut managed_pipeline = ManagedRtxPipeline {
            base_library,
            hitgroups: Vec::new(),
            library_dependencies: BTreeSet::new(),
            hitgroup_mask: 0,
        };
        managed_pipeline
            .library_dependencies
            .insert(base_library_id);
        self.pipelines.insert(handle.clone(), managed_pipeline);
        handle
    }
    /// Returns the hitgroup index
    pub fn add_hitgroup_for_pipeline(
        &mut self,
        pipeline: &Handle<RayTracingPipeline>,
        hitgroup: Handle<RayTracingPipelineLibrary>,
    ) -> u32 {
        let hitgroup_id = hitgroup.id();
        let managed_pipeline = self
            .pipelines
            .get_mut(pipeline)
            .expect("Pipeline was never added to the RtxPipelineManager");
        let index = managed_pipeline.hitgroup_mask.trailing_ones();
        if index >= managed_pipeline.hitgroups.len() as u32 {
            assert_eq!(index, managed_pipeline.hitgroups.len() as u32);
            managed_pipeline.hitgroups.push(Some(hitgroup));
        } else {
            assert!(managed_pipeline.hitgroups[index as usize].is_none());
            managed_pipeline.hitgroups[index as usize] = Some(hitgroup);
        }
        managed_pipeline.hitgroup_mask |= 1 << index;
        managed_pipeline.library_dependencies.insert(hitgroup_id);
        index
    }
    pub fn remove_hitgroup_for_pipeline(
        &mut self,
        pipeline: &Handle<RayTracingPipeline>,
        hitgroup_id: u32,
    ) -> Option<Handle<RayTracingPipelineLibrary>> {
        let managed_pipeline = self
            .pipelines
            .get_mut(pipeline)
            .expect("Pipeline was never added to the RtxPipelineManager");
        let Some(hitgroup_library) = managed_pipeline.hitgroups[hitgroup_id as usize].take() else {
            return None;
        };
        managed_pipeline
            .library_dependencies
            .remove(&hitgroup_library.id());
        managed_pipeline.hitgroup_mask &= !(1 << hitgroup_id);
        Some(hitgroup_library)
    }
}

pub fn build_rtx_pipeline_system(
    manager: ResMut<RtxPipelineManager>,
    asset_server: Res<AssetServer>,
    libraries: Res<Assets<RayTracingPipelineLibrary>>,
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
        let Some(base_library) = libraries.get(&pipeline.base_library) else {
            continue;
        };
        let Some(hitgroups) = pipeline
            .hitgroups
            .iter()
            .filter_map(Option::as_ref) // Filter out empty slots
            .map(|x| libraries.get(x))
            .collect::<Option<Vec<&RayTracingPipelineLibrary>>>()
        else {
            continue;
        };
        tracing::info!("Scheduling pipeline {} for update", id);
        let hitgroup_mask = pipeline.hitgroup_mask;

        let mut max_ray_recursion_depth = base_library.max_ray_recursion_depth;
        let mut max_ray_payload_size = base_library.max_ray_payload_size;
        let mut max_hit_attribute_size = base_library.max_hit_attribute_size;
        let mut dynamic_stack_size = base_library.dynamic_stack_size;
        let mut sbt_layout = base_library.sbt_layout.clone();
        for hitgroup in hitgroups.iter() {
            max_ray_recursion_depth = max_ray_recursion_depth.max(hitgroup.max_ray_recursion_depth);
            max_ray_payload_size = max_ray_payload_size.max(hitgroup.max_ray_payload_size);
            max_hit_attribute_size = max_hit_attribute_size.max(hitgroup.max_hit_attribute_size);
            dynamic_stack_size |= hitgroup.dynamic_stack_size;
            sbt_layout.hitgroup.count += hitgroup.sbt_layout.hitgroup.count;
            sbt_layout.hitgroup.param_size = sbt_layout
                .hitgroup
                .param_size
                .max(hitgroup.sbt_layout.hitgroup.param_size);
        }
        let pipeline_cache = pipeline_cache.clone();
        let all_libraries: Vec<&RayTracingPipelineLibrary> = std::iter::once(base_library)
            .chain(hitgroups.iter().copied())
            .collect();
        let build_monolithic = all_libraries.iter().all(|lib| lib.is_monolithic());
        let build_linked = all_libraries.iter().all(|lib| !lib.is_monolithic());
        assert!(build_monolithic != build_linked);

        if build_monolithic {
            // Inline (emulated) path: merge all shaders and groups into one monolithic pipeline.
            let mut merged_shaders = Vec::new();
            let mut merged_groups = Vec::new();
            let mut layout = None;
            let mut flags = vk::PipelineCreateFlags::empty();
            for lib in &all_libraries {
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
                    hitgroup_mask,
                })
            });
        } else if build_linked {
            // Library path: link pre-compiled pipeline libraries.
            let libraries = all_libraries
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
                    hitgroup_mask,
                })
            });
        } else {
            // Has a mix of linked / monolithic libs
            panic!()
        }
    }
}
