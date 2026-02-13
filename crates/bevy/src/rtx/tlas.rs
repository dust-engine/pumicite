use std::{
    alloc::Layout, fmt::Debug, hash::Hash, marker::PhantomData, mem::MaybeUninit, ops::Deref,
    sync::Arc,
};

use bevy_app::{Plugin, PostUpdate};
use bevy_ecs::reflect::{ReflectComponent, ReflectMapEntities};
use bevy_ecs::{
    component::Component,
    entity::{Entity, MapEntities},
    resource::Resource,
    schedule::{IntoScheduleConfigs, SystemSet},
    system::{Query, Res, ResMut},
};
use bevy_reflect::{Reflect, TypePath};
use bevy_transform::components::GlobalTransform;
use pumicite::{
    Device,
    ash::{self, vk},
    buffer::{BufferLike, RingBufferSuballocation},
    rtx::AccelStruct,
    sync::GPUMutex,
    utils::{AsVkHandle, glam_to_vk_transform},
};

use crate::{
    DefaultComputeSet, DefaultTransferSet, RenderState,
    rtx::blas::BLAS,
    staging::{BufferInitializer, DeviceLocalRingBuffer},
};

#[derive(Component, Reflect, MapEntities)]
#[reflect(opaque, Component, MapEntities)]
pub struct TLASInstance<Marker> {
    /// TLAS builder will grab the BLAS on this entity
    #[entities]
    pub blas: Entity,
    pub instance_custom_index_and_mask: vk::Packed24_8,
    pub instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8,

    pub disabled: bool,
    _marker: PhantomData<Marker>,
}
impl<Marker> Clone for TLASInstance<Marker> {
    fn clone(&self) -> Self {
        Self {
            blas: self.blas,
            instance_custom_index_and_mask: self.instance_custom_index_and_mask,
            instance_shader_binding_table_record_offset_and_flags: self
                .instance_shader_binding_table_record_offset_and_flags,
            disabled: self.disabled,
            _marker: self._marker,
        }
    }
}
impl<Marker> Default for TLASInstance<Marker> {
    fn default() -> Self {
        Self {
            blas: Entity::PLACEHOLDER,
            instance_custom_index_and_mask: vk::Packed24_8::new(0, u8::MAX),
            instance_shader_binding_table_record_offset_and_flags: Default::default(),
            disabled: true,
            _marker: Default::default(),
        }
    }
}
impl<Marker> TLASInstance<Marker> {
    pub fn new(blas: Entity) -> Self {
        Self {
            instance_custom_index_and_mask: vk::Packed24_8::new(0, u8::MAX),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
            blas,
            disabled: true,
            _marker: PhantomData,
        }
    }
    pub fn set_flags(&mut self, flags: vk::GeometryInstanceFlagsKHR) {
        self.instance_shader_binding_table_record_offset_and_flags = vk::Packed24_8::new(
            self.instance_shader_binding_table_record_offset_and_flags
                .low_24(),
            flags.as_raw() as u8,
        );
    }
    pub fn set_sbt_offset(&mut self, sbt_offset: u32) {
        self.instance_shader_binding_table_record_offset_and_flags = vk::Packed24_8::new(
            sbt_offset,
            self.instance_shader_binding_table_record_offset_and_flags
                .high_8(),
        );
    }
    pub fn set_instance_custom_index(&mut self, index: u32) {
        self.instance_custom_index_and_mask =
            vk::Packed24_8::new(index, self.instance_custom_index_and_mask.high_8());
    }
    pub fn set_mask(&mut self, mask: u8) {
        self.instance_custom_index_and_mask =
            vk::Packed24_8::new(self.instance_custom_index_and_mask.low_24(), mask);
    }
}

#[derive(Resource)]
pub struct TLAS<Marker = ()> {
    tlas: Option<GPUMutex<TLASInner>>,
    tlas_input_buffer: Option<(GPUMutex<RingBufferSuballocation>, Vec<Arc<AccelStruct>>)>,
    _marker: PhantomData<Marker>,
}
impl<Marker> TLAS<Marker> {
    pub fn get(&self) -> Option<&GPUMutex<TLASInner>> {
        self.tlas.as_ref()
    }
}

pub struct TLASInner {
    tlas: AccelStruct<RingBufferSuballocation>,
    _referenced_blas: Vec<Arc<AccelStruct>>,
}
impl AsVkHandle for TLASInner {
    type Handle = vk::AccelerationStructureKHR;
    fn vk_handle(&self) -> Self::Handle {
        self.tlas.vk_handle()
    }
}

pub fn tlas_build_input_upload_system<T: Send + Sync + 'static>(
    query: Query<(&GlobalTransform, &TLASInstance<T>)>,
    blas: Query<&BLAS>,
    mut uploader: BufferInitializer,
    mut tlas_resource: ResMut<TLAS<T>>,
    mut ctx: RenderState,
) {
    assert!(tlas_resource.tlas_input_buffer.is_none());
    // There will be some duplicated entires but that's fine.
    let referenced_blas = query
        .iter()
        .filter(|(_, instance)| !instance.disabled)
        .filter_map(|(_, instance)| blas.get(instance.blas).ok().map(|x| x.deref().clone()))
        .collect::<Vec<Arc<AccelStruct>>>();
    let num_instances = referenced_blas.len() as u32;
    if num_instances == 0 {
        return;
    }
    ctx.record(|encoder| {
        let buffer = uploader.create_preinitialized_buffer(
            encoder,
            Layout::new::<vk::AccelerationStructureInstanceKHR>()
                .repeat(num_instances as usize)
                .unwrap()
                .0
                .align_to(16)
                .unwrap(), // The Vulkan spec states: For any element of pInfos[i].pGeometries or pInfos[i].ppGeometries with a geometryType of VK_GEOMETRY_TYPE_INSTANCES_KHR, if geometry.arrayOfPointers is VK_FALSE, geometry.instances.data.deviceAddress must be aligned to 16 bytes (https://vulkan.lunarg.com/doc/view/1.4.321.1/windows/antora/spec/latest/chapters/accelstructures.html#VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03715)
            |data| {
                let slice: &mut [MaybeUninit<vk::AccelerationStructureInstanceKHR>] = unsafe {
                    std::slice::from_raw_parts_mut(
                        data.as_mut_ptr() as *mut MaybeUninit<vk::AccelerationStructureInstanceKHR>,
                        num_instances as usize,
                    )
                };
                for (i, (transform, instance)) in query
                    .iter()
                    .filter(|(_, instance)| blas.contains(instance.blas) && !instance.disabled)
                    .enumerate()
                {
                    let blas = blas.get(instance.blas).unwrap();
                    slice[i].write(vk::AccelerationStructureInstanceKHR {
                        transform: glam_to_vk_transform(transform.affine()),
                        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                            device_handle: blas.device_address(),
                        },
                        instance_custom_index_and_mask: instance.instance_custom_index_and_mask,
                        instance_shader_binding_table_record_offset_and_flags: instance
                            .instance_shader_binding_table_record_offset_and_flags,
                    });
                }
            },
        );
        tlas_resource.tlas_input_buffer = Some((buffer, referenced_blas));
    });
}

pub fn tlas_build_system<T: Send + Sync + 'static>(
    device: Res<Device>,
    mut ctx: RenderState,
    mut device_local_ring_buffer: ResMut<DeviceLocalRingBuffer>,
    mut tlas_resource: ResMut<TLAS<T>>,
) {
    let Some((buffer, referenced_blas)) = tlas_resource.tlas_input_buffer.take() else {
        return;
    };
    let num_instances = referenced_blas.len() as u32;
    let geometry = vk::AccelerationStructureGeometryKHR {
        geometry_type: vk::GeometryTypeKHR::INSTANCES,
        flags: vk::GeometryFlagsKHR::empty(),
        geometry: vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                data: vk::DeviceOrHostAddressConstKHR {
                    device_address: buffer.device_address(),
                },
                ..Default::default()
            },
        },
        ..Default::default()
    };
    let mut info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        geometry_count: 1,
        p_geometries: &geometry,
        ..Default::default()
    };
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        device
            .extension::<ash::khr::acceleration_structure::Meta>()
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &[num_instances],
                &mut sizes,
            );
    }
    let tlas_backing_buffer =
        device_local_ring_buffer.allocate_buffer(sizes.acceleration_structure_size, 256); // offset must be a multiple of 256
    let tlas = AccelStruct::create_on_buffer(
        device.clone(),
        tlas_backing_buffer,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
    )
    .unwrap();
    let scratch_offset_alignment: u64 = device
        .physical_device()
        .properties()
        .get::<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
        .min_acceleration_structure_scratch_offset_alignment
        .into();
    let scratch_buffer = device_local_ring_buffer.allocate_buffer(sizes.build_scratch_size, scratch_offset_alignment);
    info.dst_acceleration_structure = tlas.vk_handle();
    info.scratch_data = vk::DeviceOrHostAddressKHR {
        device_address: scratch_buffer.device_address(),
    };
    let tlas_inner = GPUMutex::new(TLASInner {
        tlas,
        _referenced_blas: referenced_blas,
    });
    ctx.record(|encoder| {
        encoder.lock(
            &buffer,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        );
        encoder.lock(
            &tlas_inner,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        );
        encoder.retain(scratch_buffer);
        encoder.build_accel_struct(
            &info,
            &[vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: num_instances,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            }],
        );
    });
    tlas_resource.tlas = Some(tlas_inner);
}

#[derive(SystemSet, Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct AllTLASBuilderSet;

#[derive(SystemSet, Copy)]
pub struct TLASBuilderSet<T = ()>(PhantomData<T>);
impl<T> Default for TLASBuilderSet<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<T> Clone for TLASBuilderSet<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}
impl<T> PartialEq for TLASBuilderSet<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<T> Eq for TLASBuilderSet<T> {}
impl<T> Debug for TLASBuilderSet<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TLASBuilderSet").finish()
    }
}
impl<T> Hash for TLASBuilderSet<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

pub struct TLASBuilderPlugin<T = ()> {
    _marker: std::marker::PhantomData<T>,
}
impl<T> Default for TLASBuilderPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}
impl<T: Send + Sync + TypePath + 'static> Plugin for TLASBuilderPlugin<T> {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            PostUpdate,
            (
                tlas_build_input_upload_system::<T>
                    .in_set(DefaultTransferSet)
                    .after(TLASBuilderSet::<T>::default()),
                tlas_build_system::<T>.in_set(DefaultComputeSet),
            )
                .chain(),
        );

        app.insert_resource(TLAS {
            tlas: None,
            tlas_input_buffer: None,
            _marker: PhantomData::<T>,
        });

        app.register_type::<TLASInstance<T>>();
    }
}
