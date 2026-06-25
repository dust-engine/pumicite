use std::{collections::VecDeque, ffi::CString, marker::PhantomData, ops::Deref, sync::Arc};

use bevy_app::{App, Plugin, PostUpdate, Startup};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{QueryFilter, QueryItem, ReadOnlyQueryData, Without},
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{
        Commands, Local, Query, Res, ResMut, StaticSystemParam, SystemParam, SystemParamItem,
    },
    world::{FromWorld, World},
};

use crate::CreateDevice;
use pumicite::{
    ash::khr::acceleration_structure::Meta as AccelerationStructureKhr, prelude::*,
    query::QueryPool, rtx::AccelStruct, sync::Timeline,
};
use smallvec::SmallVec;

use crate::queue::AsyncComputeQueue;
/// Component holding a built bottom-level acceleration structure for an entity.
///
/// Inserted by the [`BLASBuilder`] pipeline once the GPU build command buffer
/// for the entity has signalled completion. The inner [`AccelStruct`] is wrapped
/// in an [`Arc`] so it can be retained by a TLAS while a newer BLAS replaces it.
#[derive(Component)]
pub struct BLAS {
    accel_struct: Arc<AccelStruct>,
}
impl Deref for BLAS {
    type Target = Arc<AccelStruct>;
    fn deref(&self) -> &Self::Target {
        &self.accel_struct
    }
}

/// Defines a source of bottom-level acceleration structure builds.
///
/// Each implementation describes which entities to build BLASes for ([`QueryData`](Self::QueryData) /
/// [`QueryFilter`](Self::QueryFilter)) and how to produce their geometry data ([`geometries`](Self::geometries)).
/// Register a builder by adding [`BLASBuilderPlugin<T>`] to the app. Implicitly, each implementation
/// covers a unique set of entities and there should be no overlap between entities covered by each
/// implementation.
///
/// # Build pipeline
///
/// Each frame, in [`PostUpdate`]:
/// 1. [`drain_built_blas_system`] inspects in-flight builds and, for any that have
///    completed on the GPU, inserts a [`BLAS`] component on the corresponding entity
///    and removes the [`BLASBuildPending`] marker.
/// 2. The per-builder build system queries up to [`batch_size`](Self::batch_size)
///    matching entities that do **not** have [`BLASBuildPending`], records their
///    geometry uploads and `vkCmdBuildAccelerationStructuresKHR` calls into a single
///    command buffer, submits it on the async-compute queue, and tags those
///    entities with [`BLASBuildPending`] so they are not resubmitted while in flight.
///
/// Multiple batches may be in flight concurrently — submission is fire-and-forget,
/// gated only by [`is_ready`](Self::is_ready).
///
/// # Avoiding redundant rebuilds
///
/// The system only filters out entities with an in-flight build. It does **not**
/// filter out entities that already have a [`BLAS`]. If you want each matching
/// entity to be built exactly once, add `Without<BLAS>` to your
/// [`QueryFilter`](Self::QueryFilter); otherwise every batch will resubmit
/// already-built entities.
pub trait BLASBuilder: Resource + FromWorld {
    /// Per-entity data fetched from the ECS and forwarded to [`build_flags`](Self::build_flags)
    /// and [`geometries`](Self::geometries).
    type QueryData: ReadOnlyQueryData;

    /// Filter applied on top of the system's own `Without<BLASBuildPending>` filter.
    ///
    /// Use this to scope which entities this builder is responsible for. If the
    /// BLAS for a matching entity should only ever be built once, include
    /// `Without<BLAS>` here — the system does not exclude already-built entities
    /// on its own.
    type QueryFilter: QueryFilter;

    /// Extra [`SystemParam`]s threaded through to the trait methods (e.g. asset
    /// storages or staging buffers needed to materialise geometry).
    type Params: SystemParam;

    /// Per-entity Vulkan build flags. Called once per entity per submitted batch.
    ///
    /// Defaults to `PREFER_FAST_TRACE | ALLOW_COMPACTION`. Entities built with
    /// `ALLOW_COMPACTION` automatically receive the [`BLASNeedsCompaction`] marker
    /// once their build completes.
    #[allow(unused_variables)]
    fn build_flags(
        &mut self,
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> vk::BuildAccelerationStructureFlagsKHR {
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
    }

    /// Buffer type backing the geometry data returned by [`geometries`](Self::geometries).
    type BufferType: BufferLike;

    /// Records geometry uploads for one entity and returns the resulting
    /// [`BLASBuildGeometry`] descriptors.
    ///
    /// The future is awaited inside the build system's command-recording context,
    /// so implementations may issue transfer commands on `recorder` (e.g. staging
    /// vertex/index data into device-local buffers). The returned descriptors must
    /// reference buffers that will remain valid until the submitted build command
    /// buffer signals completion; use `recorder.retain(...)` to extend their lifetime.
    fn geometries<'w, 's, 't, 't2, 'b, 'bb>(
        &mut self,
        params: &mut SystemParamItem<'w, 's, Self::Params>,
        data: QueryItem<'t, 't2, Self::QueryData>,
        recorder: &'bb mut CommandEncoder<'b>,
    ) -> impl Future<Output = SmallVec<[BLASBuildGeometry<'b, Self::BufferType>; 1]>>
    + use<'w, 's, 't, 't2, 'b, 'bb, Self>;

    /// Gates submission of new batches.
    ///
    /// Returning `false` skips submission for the current frame without affecting
    /// in-flight builds — the drain step still runs. Use this to delay builds
    /// until external prerequisites (e.g. asset loads) are ready, or as a
    /// backpressure hook against the in-flight queue.
    fn is_ready(&self, _params: &mut SystemParamItem<'_, '_, Self::Params>) -> bool {
        true
    }

    /// Maximum number of entities to include in a single submitted command buffer.
    ///
    /// Entities beyond this cap remain in the query and are picked up by
    /// subsequent frames. Larger batches amortise submission overhead and scratch
    /// allocation; smaller batches reduce per-submit GPU latency and peak scratch
    /// memory. Defaults to 32.
    fn batch_size(&self, _params: &mut SystemParamItem<'_, '_, Self::Params>) -> usize {
        32
    }
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

struct QueuedBuild {
    command_buffer: CommandBuffer,
    accel_structs: Vec<(Entity, AccelStruct)>,
    query_pool: Option<QueryPool>,
}

/// Shared command pool, timeline, and in-flight-build queue used by every
/// [`BLASBuilder`] in the app.
///
/// Initialised once on [`Startup`] (via the first registered [`BLASBuilderPlugin`])
/// on the async-compute queue family. `queued_builds` holds command buffers that
/// have been submitted but not yet known to be complete; [`drain_built_blas_system`]
/// is responsible for popping completed entries and freeing their command buffers.
#[derive(Resource)]
pub struct ASBuildCommandPool {
    pool: pumicite::command::CommandPool,
    timeline: Timeline,
    queued_builds: VecDeque<QueuedBuild>,
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
        Self {
            pool,
            timeline,
            queued_builds: VecDeque::new(),
        }
    }
}

/// Look at all queued builds in [`ASBuildCommandPool`], dequeue the ones that are already finished, and insert [`BLAS`] components to their entities.
fn drain_built_blas_system(mut commands: Commands, mut cmd_pool: ResMut<ASBuildCommandPool>) {
    while let Some(build) = cmd_pool
        .queued_builds
        .pop_front_if(|build| build.command_buffer.try_complete())
    {
        cmd_pool.pool.free(build.command_buffer);
        tracing::info!(
            "BLAS build completed for {} entities",
            build.accel_structs.len()
        );
        let compacted_sizes = if let Some(query_pool) = build.query_pool {
            let mut sizes = vec![0; query_pool.len() as usize];
            query_pool
                .get_results::<u64>(0, &mut sizes, vk::QueryResultFlags::TYPE_64)
                .unwrap();
            sizes
        } else {
            Vec::new()
        };
        let mut compacted_i = 0;
        for (entity, mut accel_struct) in build.accel_structs.into_iter() {
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
                target.insert(BLASNeedsCompaction {
                    compacted_size: compacted_sizes[compacted_i],
                });
                compacted_i += 1;
            }
            target.insert(BLAS {
                accel_struct: Arc::new(accel_struct),
            });
            target.remove::<BLASBuildPending>();
        }
    }
}

fn build_blas_system<T: BLASBuilder>(
    mut commands: Commands,
    builder: ResMut<T>,
    device: Res<Device>,
    allocator: Res<Allocator>,
    cmd_pool: ResMut<ASBuildCommandPool>,
    entities: Query<(Entity, T::QueryData), (T::QueryFilter, Without<BLASBuildPending>)>,
    params: StaticSystemParam<T::Params>,
    mut queue: crate::Queue<AsyncComputeQueue>,
) {
    if entities.is_empty() {
        return;
    }
    let mut params = params.into_inner();
    let builder = builder.into_inner();
    if !T::is_ready(builder, &mut params) {
        return;
    }
    let batch_size = T::batch_size(builder, &mut params);
    let mut pending_accel_structs = Vec::new();
    let mut accel_structs_to_query_compaction_sizes = Vec::new();
    let mut query_pool: Option<QueryPool> = None;

    let mut geometry_infos = Vec::<vk::AccelerationStructureGeometryKHR>::new();
    let mut geometry_infos_primitive_counts: Vec<u32> = Vec::new();
    let mut build_range_infos = Vec::<vk::AccelerationStructureBuildRangeInfoKHR>::new();
    let mut infos = Vec::<vk::AccelerationStructureBuildGeometryInfoKHR>::new();
    let future = async |recorder: &mut CommandEncoder| {
        let geometry_transfer_futures = entities.iter().take(batch_size).map(|(_, query)| {
            T::geometries(builder, &mut params, query, unsafe {
                // Unsafely reborrow the recorder.
                // We should be able to get rid of this after coroutine.
                &mut *(recorder as *mut CommandEncoder)
            })
        });
        let geometry_transfers = pumicite::utils::future::zip_many(geometry_transfer_futures).await;
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

        for ((entity, query), geometries) in entities
            .iter()
            .take(batch_size)
            .zip(geometry_transfers.into_iter())
        {
            if geometries.is_empty() {
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
                info.flags,
            )
            .unwrap();
            if info
                .flags
                .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION)
            {
                accel_structs_to_query_compaction_sizes.push(blas.vk_handle());
            }
            info.dst_acceleration_structure = blas.vk_handle();
            pending_accel_structs.push((entity, blas));

            infos.push(info);
        }
        drop(geometry_infos_primitive_counts);
        if infos.is_empty() {
            return;
        }

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
        if !accel_structs_to_query_compaction_sizes.is_empty() {
            let pool = QueryPool::new(
                device.clone(),
                vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                accel_structs_to_query_compaction_sizes.len() as u32,
            )
            .unwrap();
            pool.host_reset(0..pool.len());
            // The build above writes the AS; the property query below reads it.
            // No automatic dependency exists within the same command buffer.
            recorder.memory_barrier(
                Access {
                    stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                    access: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                },
                Access {
                    stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                    access: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                },
            );
            recorder.emit_barriers();
            recorder.write_acceleration_structures_properties(
                &accel_structs_to_query_compaction_sizes,
                &pool,
                0,
            );
            query_pool = Some(pool);
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

    let pending_entities = pending_accel_structs
        .iter()
        .map(|(entity, _)| *entity)
        .collect::<Vec<_>>();
    commands.insert_batch(
        pending_entities
            .into_iter()
            .map(|entity| (entity, BLASBuildPending)),
    );
    cmd_pool.queued_builds.push_back(QueuedBuild {
        command_buffer: cmd_buf,
        accel_structs: pending_accel_structs,
        query_pool,
    });
}

/// Plugin that wires a single [`BLASBuilder`] implementation into the app.
///
/// On [`Startup`] (after [`CreateDevice`]) it initialises the builder resource and
/// the shared [`ASBuildCommandPool`]. On every [`PostUpdate`] it runs
/// `build_blas_system::<T>` after [`drain_built_blas_system`].
///
/// The drain system is registered exactly once across all [`BLASBuilderPlugin<T>`]
/// instances via an internal [`BLASPlugin`] guarded by [`App::is_plugin_added`].
/// Adding `BLASBuilderPlugin<T>` for multiple `T` is supported.
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

struct BLASPlugin;
impl Plugin for BLASPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            (drain_built_blas_system, compact_blas_system).chain(),
        );
    }
}

/// One in-flight compaction submission: a recorded copy command buffer plus the
/// per-entity book-keeping needed to publish the compacted AS once the GPU
/// finishes.
///
/// `entries` holds, for each entity:
/// 1. the freshly-allocated compacted [`AccelStruct`] (owned, becomes the new
///    [`BLAS`] when the copy completes);
/// 2. an [`Arc`] of the source AS, kept alive until the GPU has finished
///    reading it.
struct PendingCompaction {
    command_buffer: CommandBuffer,
    entries: Vec<(Entity, AccelStruct, Arc<AccelStruct>)>,
}

/// Compacts BLASes whose builds reported a smaller post-compaction size.
///
/// On every [`PostUpdate`]:
/// 1. Pops any in-flight compaction command buffers from `pending` whose GPU
///    work has finished, frees their command buffer, and replaces each
///    entity's [`BLAS`] with the compacted copy.
/// 2. For every entity that has [`BLASNeedsCompaction`] (and is not currently
///    being rebuilt), allocates a destination [`AccelStruct`] sized to
///    `compacted_size`, records `vkCmdCopyAccelerationStructureKHR` in
///    `MODE_COMPACT`, removes the [`BLASNeedsCompaction`] marker, and submits
///    on the async-compute queue. The submission is appended to `pending`
///    for a later frame to drain.
///
/// The shared [`ASBuildCommandPool`] timeline serialises this submission after
/// any outstanding builds, so the source AS is guaranteed to be readable. The
/// source [`Arc`] is retained inside `pending` for the GPU lifetime; replacing
/// the entity's [`BLAS`] only happens after `try_complete()` has confirmed the
/// copy is done.
fn compact_blas_system(
    mut commands: Commands,
    cmd_pool: ResMut<ASBuildCommandPool>,
    allocator: Res<Allocator>,
    device: Res<Device>,
    needs_compaction: Query<(Entity, &BLAS, &BLASNeedsCompaction), Without<BLASBuildPending>>,
    mut queue: crate::Queue<AsyncComputeQueue>,
    mut pending: Local<VecDeque<PendingCompaction>>,
) {
    let cmd_pool = cmd_pool.into_inner();

    // 1. Drain completed compactions.
    while let Some(done) = pending.pop_front_if(|p| p.command_buffer.try_complete()) {
        cmd_pool.pool.free(done.command_buffer);
        tracing::info!(
            "BLAS compaction completed for {} entities",
            done.entries.len()
        );
        for (entity, mut compacted, _retain_src) in done.entries.into_iter() {
            compacted.set_name(
                CString::new(format!("BLAS {:?} (compacted)", entity))
                    .unwrap()
                    .as_c_str(),
            );
            commands.entity(entity).insert(BLAS {
                accel_struct: Arc::new(compacted),
            });
        }
    }

    // 2. Submit new compactions.
    if needs_compaction.is_empty() {
        return;
    }
    let mut entries: Vec<(Entity, AccelStruct, Arc<AccelStruct>)> = Vec::new();
    let mut copy_infos: Vec<vk::CopyAccelerationStructureInfoKHR> = Vec::new();
    for (entity, blas, needs) in needs_compaction.iter() {
        let compacted = AccelStruct::new(
            allocator.clone(),
            needs.compacted_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            blas.flags(),
        )
        .unwrap();
        copy_infos.push(vk::CopyAccelerationStructureInfoKHR {
            src: blas.vk_handle(),
            dst: compacted.vk_handle(),
            mode: vk::CopyAccelerationStructureModeKHR::COMPACT,
            ..Default::default()
        });
        entries.push((entity, compacted, blas.accel_struct.clone()));
        commands.entity(entity).remove::<BLASNeedsCompaction>();
    }

    let mut cmd_buf = cmd_pool
        .pool
        .alloc()
        .unwrap()
        .with_name(c"BLAS Compaction Command Buffer");
    cmd_pool.timeline.schedule(&mut cmd_buf);
    cmd_pool.pool.begin(&mut cmd_buf).unwrap();
    cmd_pool.pool.record(&mut cmd_buf, |encoder| {
        for info in &copy_infos {
            unsafe {
                device
                    .extension::<AccelerationStructureKhr>()
                    .cmd_copy_acceleration_structure(encoder.buffer().vk_handle(), info);
            }
        }
    });
    cmd_pool.pool.finish(&mut cmd_buf).unwrap();
    queue.submit(&mut cmd_buf).unwrap();

    let total_before: u64 = entries.iter().map(|(_, _, src)| src.size()).sum();
    let total_after: u64 = entries.iter().map(|(_, dst, _)| dst.size()).sum();
    let saved_pct = if total_before == 0 {
        0.0
    } else {
        (total_before - total_after) as f64 / total_before as f64 * 100.0
    };
    tracing::info!(
        "Scheduled BLAS compaction for {} entities: {} -> {} bytes ({:.1}% saved)",
        entries.len(),
        total_before,
        total_after,
        saved_pct,
    );
    pending.push_back(PendingCompaction {
        command_buffer: cmd_buf,
        entries,
    });
}

impl<T: BLASBuilder> Plugin for BLASBuilderPlugin<T> {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<BLASPlugin>() {
            app.add_plugins(BLASPlugin);
        }
        app.add_systems(
            PostUpdate,
            build_blas_system::<T>.after(drain_built_blas_system),
        );
        app.add_systems(
            Startup,
            (|world: &mut World| {
                world.init_resource::<ASBuildCommandPool>();
                world.init_resource::<T>();
            })
            .after(CreateDevice),
        );
    }
}

/// Marker inserted on entities whose just-built [`BLAS`] was created with
/// `ALLOW_COMPACTION` and is therefore eligible to be replaced by a compacted copy.
///
/// The compaction pass is responsible for removing this marker once it has acted.
#[derive(Component)]
pub struct BLASNeedsCompaction {
    pub compacted_size: vk::DeviceSize,
}

/// Marker inserted by `build_blas_system` while a build for this entity is in
/// flight on the GPU, and removed by [`drain_built_blas_system`] once the build
/// completes (at the same time [`BLAS`] is inserted).
///
/// The build system queries entities with `Without<BLASBuildPending>`, so this
/// marker prevents the same entity from being submitted again before its
/// previous build finishes. Implementors of [`BLASBuilder`] should not insert
/// or remove this component themselves.
#[derive(Component)]
pub struct BLASBuildPending;
