use std::{collections::BTreeSet, ffi::CString, marker::PhantomData, ops::Deref, sync::Arc};

use bevy_app::{App, Plugin, PostUpdate};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{ArchetypeFilter, QueryFilter, QueryItem, ReadOnlyQueryData, Without},
    resource::Resource,
    system::{
        Commands, Local, Query, Res, ResMut, StaticSystemParam, SystemParam, SystemParamItem,
    },
    world::FromWorld,
};
use pumicite::{
    ash::khr::acceleration_structure::Meta as AccelerationStructureKhr, prelude::*,
    rtx::AccelStruct, sync::Timeline,
};
use smallvec::SmallVec;

use crate::queue::AsyncComputeQueue;
#[derive(Component)]
pub struct BLAS {
    // problem is, this needs to be retained by the TLAS. Really should be Arc<T>.
    // It has multiple states.
    accel_struct: Arc<AccelStruct>,
}
impl Deref for BLAS {
    type Target = Arc<AccelStruct>;
    fn deref(&self) -> &Self::Target {
        &self.accel_struct
    }
}

/// The BLASBuilder look at all entities with QueryData and QueryFilter, and insert the BLAS component.
pub trait BLASBuilder: Resource + FromWorld {
    /// Associated entities to be passed.
    type QueryData: ReadOnlyQueryData;

    /// Note: If the BLAS will never be updated, you may add Without<BLAS> here
    /// to exclude all entities with BLAS already built.
    type QueryFilter: QueryFilter + ArchetypeFilter;
    /// Additional system entities to be passed.
    type Params: SystemParam;

    #[allow(unused_variables)]
    fn build_flags(
        &mut self,
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> vk::BuildAccelerationStructureFlagsKHR {
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
    }

    type BufferType: BufferLike;
    /// The geometries to be built. The implementation shall write directly into the dst buffer.
    /// The iterator returned shall contain offset values into the dst buffer.
    fn geometries<'w, 's, 't, 't2, 'b, 'bb>(
        &mut self,
        params: &mut SystemParamItem<'w, 's, Self::Params>,
        data: QueryItem<'t, 't2, Self::QueryData>,
        recorder: &'bb mut CommandEncoder<'b>,
    ) -> impl Future<Output = SmallVec<[BLASBuildGeometry<'b, Self::BufferType>; 1]>>
    + use<'w, 's, 't, 't2, 'b, 'bb, Self>;
}

pub enum BLASBuildGeometry<'a, A: BufferLike> {
    Triangles {
        vertex_format: vk::Format,
        vertex_data: &'a A,
        vertex_stride: vk::DeviceSize,
        max_vertex: u32,
        index_type: vk::IndexType,
        index_data: &'a A,
        transform_data: Option<vk::TransformMatrixKHR>,
        flags: vk::GeometryFlagsKHR,
        /// Number of triangles to be built, where each triangle is treated as 3 vertices
        primitive_count: u32,
    },
    Aabbs {
        buffer: &'a A,
        stride: vk::DeviceSize,
        flags: vk::GeometryFlagsKHR,
        /// Number of AABBs to be built, where each triangle is treated as 3 vertices
        primitive_count: u32,
    },
}

#[derive(Resource)]
pub struct ASBuildCommandPool {
    pool: pumicite::command::CommandPool,
    timeline: Timeline,
}
impl FromWorld for ASBuildCommandPool {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        let async_compute_family_index = world
            .resource::<crate::queue::QueueConfiguration>()
            .queue_info::<crate::queue::AsyncComputeQueue>()
            .unwrap()
            .family_index;
        let device = world.resource::<Device>().clone();

        let pool = pumicite::command::CommandPool::new(device.clone(), async_compute_family_index)
            .unwrap();
        let timeline = Timeline::new(device).unwrap();
        Self { pool, timeline }
    }
}

fn build_blas_system<T: BLASBuilder>(
    builder: ResMut<T>,
    device: Res<Device>,
    allocator: Res<Allocator>,
    mut cmd_pool: ResMut<ASBuildCommandPool>,
    mut commands: Commands,

    entities: Query<(Entity, T::QueryData), (T::QueryFilter, Without<BLAS>)>,
    params: StaticSystemParam<T::Params>,
    mut queue: crate::Queue<AsyncComputeQueue>,

    mut pending_command_buffer: Local<Option<(CommandBuffer, Vec<(Entity, AccelStruct)>)>>,
) {
    let builder = builder.into_inner();
    let mut pending_accel_structs = Vec::new();
    let mut accel_structs_just_completed = BTreeSet::new();
    if let Some((pending_cb, _)) = pending_command_buffer.as_mut() {
        if pending_cb.try_complete() {
            let (completed_cb, mut built_accel_structs) = pending_command_buffer.take().unwrap();
            cmd_pool.pool.free(completed_cb);
            tracing::info!(
                "BLAS build completed for {} entities",
                built_accel_structs.len()
            );
            for (entity, mut accel_struct) in built_accel_structs.drain(..) {
                let mut target = commands.entity(entity);

                accel_struct.set_name(
                    CString::new(format!("BLAS {:?}", target.id()))
                        .unwrap()
                        .as_c_str(),
                );

                if accel_struct
                    .flags()
                    .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION)
                {
                    target.insert(BLASNeedsCompaction);
                }
                target.insert(BLAS {
                    accel_struct: Arc::new(accel_struct),
                });

                accel_structs_just_completed.insert(entity);
            }
            pending_accel_structs = built_accel_structs; // Reuse memory.
        } else {
            return;
        }
    }
    if entities.is_empty() {
        return;
    }
    if entities
        .iter()
        .filter(|(entity, _)| !accel_structs_just_completed.contains(entity))
        .count()
        == 0
    {
        return;
    }

    let mut geometry_infos = Vec::<vk::AccelerationStructureGeometryKHR>::new();
    let mut geometry_infos_primitive_counts: Vec<u32> = Vec::new();
    let mut build_range_infos = Vec::<vk::AccelerationStructureBuildRangeInfoKHR>::new();
    let mut infos = Vec::<vk::AccelerationStructureBuildGeometryInfoKHR>::new();
    let future = async |recorder: &mut CommandEncoder| {
        let mut params = params.into_inner();
        let geometry_transfer_futures = entities.iter().map(|(_, query)| {
            T::geometries(builder, &mut params, query, unsafe {
                // Unsafely reborrow the recorder.
                // We should be able to get rid of this after coroutine.
                &mut *(recorder as *mut CommandEncoder)
            })
        });
        let geometry_transfers = pumicite::utils::future::zip_many(geometry_transfer_futures).await;
        tracing::info!("Building {} BLAS", geometry_transfers.len());
        recorder.memory_barrier(
            Access::ALL_COMMANDS,
            Access::ACCELERATION_STRUCTURE_BUILD_READ,
        );
        pumicite::utils::future::yield_now().await;

        let mut total_scratch_size: u64 = 0;
        let scratch_offset_alignment: u32 = device
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
            .min_acceleration_structure_scratch_offset_alignment;

        for ((entity, query), geometries) in entities.iter().zip(geometry_transfers.into_iter()) {
            if accel_structs_just_completed.contains(&entity) {
                continue;
            }
            geometry_infos_primitive_counts.clear();
            geometry_infos_primitive_counts.extend(geometries.iter().map(
                |geometry| match geometry {
                    BLASBuildGeometry::Triangles {
                        primitive_count, ..
                    } => primitive_count,
                    BLASBuildGeometry::Aabbs {
                        primitive_count, ..
                    } => primitive_count,
                },
            ));

            build_range_infos.extend(geometries.iter().map(|geometry| match geometry {
                BLASBuildGeometry::Triangles {
                    primitive_count, ..
                } => vk::AccelerationStructureBuildRangeInfoKHR {
                    primitive_count: *primitive_count,
                    ..Default::default()
                },
                BLASBuildGeometry::Aabbs {
                    primitive_count, ..
                } => vk::AccelerationStructureBuildRangeInfoKHR {
                    primitive_count: *primitive_count,
                    ..Default::default()
                },
            }));
            let num_geometries = geometries.len() as u32;
            geometry_infos.extend(geometries.into_iter().map(|geometry| match geometry {
                BLASBuildGeometry::Triangles {
                    vertex_format,
                    vertex_data,
                    vertex_stride,
                    max_vertex,
                    index_type,
                    index_data,
                    transform_data: _,
                    flags,
                    primitive_count: _,
                } => vk::AccelerationStructureGeometryKHR {
                    geometry_type: vk::GeometryTypeKHR::TRIANGLES,
                    geometry: vk::AccelerationStructureGeometryDataKHR {
                        triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                            vertex_format,
                            vertex_data: vk::DeviceOrHostAddressConstKHR {
                                device_address: vertex_data.device_address(),
                            },
                            vertex_stride,
                            max_vertex,
                            index_type,
                            index_data: vk::DeviceOrHostAddressConstKHR {
                                device_address: index_data.device_address(),
                            },
                            ..Default::default()
                        },
                    },
                    flags,
                    ..Default::default()
                },
                BLASBuildGeometry::Aabbs {
                    buffer,
                    stride,
                    flags,
                    primitive_count: _,
                } => vk::AccelerationStructureGeometryKHR {
                    geometry_type: vk::GeometryTypeKHR::AABBS,
                    geometry: vk::AccelerationStructureGeometryDataKHR {
                        aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                            data: vk::DeviceOrHostAddressConstKHR {
                                device_address: buffer.device_address(),
                            },
                            stride,
                            ..Default::default()
                        },
                    },
                    flags,
                    ..Default::default()
                },
            }));

            let mut info = vk::AccelerationStructureBuildGeometryInfoKHR {
                ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                flags: T::build_flags(builder, &mut params, &query),
                mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                geometry_count: num_geometries,
                p_geometries: unsafe {
                    geometry_infos
                        .as_ptr()
                        .add(geometry_infos.len() - num_geometries as usize)
                },
                ..Default::default()
            };

            let build_sizes = unsafe {
                let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
                device
                    .extension::<AccelerationStructureKhr>()
                    .get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &info,
                        &geometry_infos_primitive_counts,
                        &mut size_info,
                    );
                size_info
            };

            total_scratch_size =
                total_scratch_size.next_multiple_of(scratch_offset_alignment as u64);
            info.scratch_data.device_address = total_scratch_size;
            total_scratch_size += build_sizes.build_scratch_size;

            let blas = AccelStruct::new(
                allocator.clone(),
                build_sizes.acceleration_structure_size,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            )
            .unwrap();
            info.dst_acceleration_structure = blas.vk_handle();
            pending_accel_structs.push((entity, blas));

            infos.push(info);
        }
        drop(geometry_infos_primitive_counts);

        let scratch_buffer = Buffer::new_private(
            allocator.clone(),
            total_scratch_size,
            scratch_offset_alignment as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap();
        let scratch_buffer = recorder.retain(Box::new(scratch_buffer));

        // Second pass to patch up infos
        let mut geometry_i: u32 = 0;
        let mut ptrs: Vec<*const vk::AccelerationStructureBuildRangeInfoKHR> =
            Vec::with_capacity(infos.len());
        for info in infos.iter_mut() {
            unsafe {
                // Safety: On the previous pass, we've set info.scratch_data.device_address to be the
                // offset into the scratch buffer
                info.scratch_data.device_address += scratch_buffer.device_address();
            }
            unsafe {
                info.p_geometries = geometry_infos.as_ptr().add(geometry_i as usize);
                ptrs.push(build_range_infos.as_ptr().add(geometry_i as usize));
                geometry_i += info.geometry_count;
            }
        }

        unsafe {
            (device
                .extension::<AccelerationStructureKhr>()
                .fp()
                .cmd_build_acceleration_structures_khr)(
                recorder.buffer().vk_handle(),
                infos.len() as u32,
                infos.as_ptr(),
                ptrs.as_ptr(),
            );
        }
    };

    let cmd_pool = cmd_pool.into_inner();

    let mut cmd_buf = cmd_pool
        .pool
        .alloc()
        .unwrap()
        .with_name(c"BLAS Build Command Buffer");

    cmd_pool.timeline.schedule(&mut cmd_buf);
    cmd_pool.pool.begin(&mut cmd_buf).unwrap();
    cmd_pool.pool.record_future(&mut cmd_buf, future);
    cmd_pool.pool.finish(&mut cmd_buf).unwrap();
    queue.submit(&mut cmd_buf).unwrap();
    tracing::info!(
        "Scheduled BLAS build for {} entities",
        pending_accel_structs.len()
    );

    *pending_command_buffer = Some((cmd_buf, pending_accel_structs));
}

pub struct BLASBuilderPlugin<T: BLASBuilder> {
    _marker: PhantomData<T>,
}
impl<T: BLASBuilder> Default for BLASBuilderPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: BLASBuilder> Plugin for BLASBuilderPlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, build_blas_system::<T>);
    }

    fn finish(&self, app: &mut App) {
        app.init_resource::<ASBuildCommandPool>();
        app.init_resource::<T>();
    }
}

#[derive(Component)]
pub struct BLASNeedsCompaction;

/* Compaction


#[derive(Component)]
pub struct BLASProperties {
    compacted_size: vk::DeviceSize,
}

#[derive(Resource, Default)]
pub(crate) struct BLASCompactionTask {
    /// Array of (Entity, original_size) pairs
    queried_entities: Vec<(Entity, u64)>,
    query_pool: Option<RenderObject<QueryPool>>,
    compacted_entities: Vec<(Entity, AccelStruct)>,
    task: Option<AsyncComputeTask<()>>,
}
pub(crate) fn blas_compaction_system(
    mut commands: Commands,
    mut task_pool: ResMut<AsyncTaskPool>,
    mut pending_query_task: ResMut<BLASCompactionTask>,
    mut existing_blas: Query<&mut BLAS>,
    // mut render_commands: RenderCommands<'c'>,
) {
    if let Some(pending_task) = pending_query_task.task.as_ref() {
        // Has pending task
        if !pending_task.is_finished() {
            return;
        }
        // Pending task finished
        let pending_task = pending_query_task.task.take().unwrap();
        task_pool.wait_blocked(pending_task);

        // Insert all compacted BLAS
        for (entity, accel_struct) in pending_query_task.compacted_entities.drain(..) {
            commands.entity(entity).insert(BLASCompacted);
            let old_blas = std::mem::replace(
                &mut existing_blas.get_mut(entity).unwrap().accel_struct,
                accel_struct,
            );
            //RenderObject::new(old_blas).use_on(&mut render_commands); // Defer dropping the BLAS until after the current frame finishes
        }

        // If did query, get query results
        if let Some(pool) = pending_query_task.query_pool.take() {
            assert!(!pending_query_task.queried_entities.is_empty());
            let mut results = vec![0_u64; pending_query_task.queried_entities.len()];
            pool.get().get_results_u64(0, &mut results).unwrap();
            for ((entity, original_size), compacted_size) in
                pending_query_task.queried_entities.drain(..).zip(results)
            {
                commands
                    .entity(entity)
                    .insert(BLASProperties { compacted_size });
            }
        }
    }
}

pub(crate) fn blas_compaction_system_schedule(
    mut commands: Commands,
    query_candidates: Query<(Entity, &BLAS), Without<BLASProperties>>,
    copy_candidates: Query<(Entity, &BLAS, &BLASProperties), Without<BLASCompacted>>,
    mut task_pool: ResMut<AsyncTaskPool>,
    device: Res<Device>,
    allocator: Res<Allocator>,
    mut pending_query_task: ResMut<BLASCompactionTask>,
) {
    if pending_query_task.task.is_some() {
        return;
    }
    if query_candidates.is_empty() && copy_candidates.is_empty() {
        return;
    }
    let mut task = task_pool.spawn_compute();

    let query_accel_structs = query_candidates
        .iter()
        .map(|x| x.1.accel_struct.raw)
        .collect::<Vec<_>>();
    let query_pool = if !query_accel_structs.is_empty() {
        let pool = QueryPool::new(
            device.clone(),
            vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            query_accel_structs.len() as u32,
        )
        .unwrap();
        let mut pool = RenderObject::new(pool);
        task.reset_query_pool(&mut pool, ..);
        task.write_acceleration_structures_properties(&query_accel_structs, &mut pool, 0);
        Some(pool)
    } else {
        None
    };

    assert!(pending_query_task.compacted_entities.is_empty());

    pending_query_task.compacted_entities = copy_candidates
        .iter()
        .filter_map(|(entity, blas, blas_properties)| {
            if blas.size() * 9 / 10 <= blas_properties.compacted_size {
                tracing::debug!(
                    "Skipped BLAS compaction for {:?}: {} -> {}",
                    entity,
                    blas.size(),
                    blas_properties.compacted_size
                );
                commands.entity(entity).insert(BLASCompacted);
                return None;
            }
            let compacted_blas = AccelStruct::new(
                allocator.clone(),
                blas_properties.compacted_size,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            )
            .unwrap();
            task.copy_acceleration_structure(&vk::CopyAccelerationStructureInfoKHR {
                src: blas.raw,
                dst: compacted_blas.raw,
                mode: vk::CopyAccelerationStructureModeKHR::COMPACT,
                ..Default::default()
            });
            tracing::debug!(
                "Compacted BLAS for {:?}: {} -> {}",
                entity,
                blas.size(),
                compacted_blas.size()
            );
            Some((entity, compacted_blas))
        })
        .collect();

    pending_query_task
        .task
        .replace(task.finish((), vk::PipelineStageFlags2::empty()));
    pending_query_task.query_pool = query_pool;
    pending_query_task.queried_entities = query_candidates
        .iter()
        .map(|(entity, blas)| (entity, blas.size()))
        .collect();
}
*/
