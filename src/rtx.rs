//! Ray tracing pipeline and acceleration structure support.
//!
//! This module provides types for hardware-accelerated ray tracing using the
//! `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure` extensions.
//!
//! # Overview
//!
//! The main components to enable ray tracing in Vulkan are:
//! - **Acceleration structures**: Spatial data structures (BLAS/TLAS) for fast ray-scene intersection
//! - **Ray tracing pipelines**: Shader programs for ray generation, intersection, and shading
//! - **Shader binding tables**: Maps shader indices to shader groups
//!
//! # Pipeline Creation
//!
//! Ray tracing pipelines can be created as libraries and linked together:
//!
//! ```ignore
//! // Create a pipeline library with shaders
//! let library = cache.create_ray_tracing_pipeline_library(RayTracingPipelineLibraryCreateInfo {
//!     shaders: &[raygen, miss, closest_hit],
//!     groups: &[raygen_group, miss_group, hit_group],
//!     max_ray_recursion_depth: 1,
//!     ..Default::default()
//! })?;
//!
//! // Link libraries into a final pipeline
//! let pipeline = cache.create_ray_tracing_pipeline(
//!     &[library],
//!     2,  // max ray recursion depth
//!     32, // max ray payload size
//!     8,  // max hit attribute size
//!     false // dynamic stack size
//! )?;
//! ```
//!
//! # Shader Binding Table
//!
//! The [`ShaderBindingTable`] manages shader handles and per-shader data. Each
//! push identifies a shader group by its `(library_id, shader_id)` pair:
//!
//! ```ignore
//! let mut sbt = ShaderBindingTable::new_with_mapper(&pipeline, layout, &remapper, None);
//! sbt.push_raygen(0, 0, |_| {});
//! sbt.push_miss(0, 1, |_| {});
//! sbt.push_hitgroup(0, 2, |data: &mut [u8]| { /* write inline data */ });
//! ```

use std::{alloc::Layout, ops::Deref, sync::Arc};

use crate::{
    Allocator, Device, HasDevice,
    buffer::{Buffer, BufferLike},
    command::CommandEncoder,
    debug::DebugObject,
    utils::AsVkHandle,
};

use crate::pipeline::{Pipeline, PipelineCache, PipelineLayout, ShaderEntry};
use ash::{
    VkResult,
    khr::acceleration_structure::Meta as AccelerationStructureExt,
    vk::{self},
};
use glam::UVec3;

/// Configuration for creating a ray tracing pipeline library.
///
/// Pipeline libraries contain compiled shaders that can be linked together
/// to form a complete ray tracing pipeline.
pub struct RayTracingPipelineLibraryCreateInfo<'a> {
    /// Pipeline creation flags.
    pub flags: vk::PipelineCreateFlags,
    /// The pipeline layout.
    pub layout: Arc<PipelineLayout>,
    /// Maximum ray recursion depth (e.g., for reflections).
    pub max_ray_recursion_depth: u32,
    /// Maximum size of ray payload data passed between shaders.
    pub max_ray_payload_size: u32,
    /// Maximum size of hit attribute data from intersection shaders.
    pub max_hit_attribute_size: u32,
    /// Whether to use dynamic stack size.
    pub dynamic_stack_size: bool,
    /// Shader stages in this library.
    pub shaders: &'a [ShaderEntry<'a>],
    /// Shader groups defining how shaders are combined.
    pub groups: &'a [vk::RayTracingShaderGroupCreateInfoKHR<'static>],
}

impl PipelineCache {
    /// Creates a ray tracing pipeline by linking pipeline libraries.
    ///
    /// At least one pipeline library is required.
    pub fn create_ray_tracing_pipeline<I: IntoIterator>(
        &self,
        libraries: I,
        max_ray_recursion_depth: u32,
        max_ray_payload_size: u32,
        max_hit_attribute_size: u32,
        dynamic_stack_size: bool,
    ) -> VkResult<Pipeline>
    where
        I::Item: Deref<Target = Pipeline>,
    {
        let mut libraries = libraries.into_iter();
        let library_interface = vk::RayTracingPipelineInterfaceCreateInfoKHR {
            max_pipeline_ray_hit_attribute_size: max_hit_attribute_size,
            max_pipeline_ray_payload_size: max_ray_payload_size,
            ..Default::default()
        };
        let dynamic_states = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::RAY_TRACING_PIPELINE_STACK_SIZE_KHR]);
        let mut library_handles = Vec::new();
        let first_library = libraries
            .next()
            .expect("At least one pipeline library required!");
        let layout = first_library.layout().clone();
        library_handles.push(first_library.vk_handle());
        library_handles.extend(libraries.map(|x| x.vk_handle()));
        let library_info = vk::PipelineLibraryCreateInfoKHR::default().libraries(&library_handles);
        let mut vk_create_info = vk::RayTracingPipelineCreateInfoKHR {
            max_pipeline_ray_recursion_depth: max_ray_recursion_depth,
            layout: layout.vk_handle(),
            ..Default::default()
        }
        .library_interface(&library_interface)
        .library_info(&library_info);
        if dynamic_stack_size {
            vk_create_info = vk_create_info.dynamic_state(&dynamic_states);
        }

        let mut pipeline = vk::Pipeline::null();
        unsafe {
            (self
                .device()
                .extension::<ash::khr::ray_tracing_pipeline::Meta>()
                .fp()
                .create_ray_tracing_pipelines_khr)(
                self.device().handle(),
                vk::DeferredOperationKHR::null(),
                self.vk_handle(),
                1,
                &vk_create_info,
                std::ptr::null(),
                &mut pipeline,
            )
            .result()?;
        };

        Ok(Pipeline::from_raw(self.device().clone(), pipeline, layout))
    }

    /// Creates a monolithic ray tracing pipeline directly from shaders and groups,
    /// without using VK_KHR_pipeline_library. This is the fallback path for drivers
    /// that don't support pipeline libraries.
    pub fn create_ray_tracing_pipeline_monolithic(
        &self,
        create_info: RayTracingPipelineLibraryCreateInfo,
    ) -> VkResult<Pipeline> {
        let specialization_infos: Vec<_> = create_info
            .shaders
            .iter()
            .map(|shader| shader.specialization_info.raw_specialization_info())
            .collect();
        let stages: Vec<_> = create_info
            .shaders
            .iter()
            .zip(&specialization_infos)
            .map(
                |(shader, specialization_info)| vk::PipelineShaderStageCreateInfo {
                    flags: shader.flags,
                    stage: shader.stage,
                    module: shader.module.vk_handle(),
                    p_name: shader.entry.as_ptr(),
                    p_specialization_info: specialization_info,
                    ..Default::default()
                },
            )
            .collect();
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::RAY_TRACING_PIPELINE_STACK_SIZE_KHR]);
        let mut vk_create_info = vk::RayTracingPipelineCreateInfoKHR {
            flags: create_info.flags,
            max_pipeline_ray_recursion_depth: create_info.max_ray_recursion_depth,
            layout: create_info.layout.vk_handle(),
            ..Default::default()
        }
        .stages(&stages)
        .groups(create_info.groups);
        if create_info.dynamic_stack_size {
            vk_create_info = vk_create_info.dynamic_state(&dynamic_state);
        }

        let mut pipeline = vk::Pipeline::null();
        unsafe {
            (self
                .device()
                .extension::<ash::khr::ray_tracing_pipeline::Meta>()
                .fp()
                .create_ray_tracing_pipelines_khr)(
                self.device().handle(),
                vk::DeferredOperationKHR::null(),
                self.vk_handle(),
                1,
                &vk_create_info,
                std::ptr::null(),
                &mut pipeline,
            )
            .result()?;
        };
        Ok(Pipeline::from_raw(
            self.device().clone(),
            pipeline,
            create_info.layout,
        ))
    }

    /// Creates a ray tracing pipeline library from shaders and shader groups.
    ///
    /// Pipeline libraries can be linked together using [`create_ray_tracing_pipeline`](Self::create_ray_tracing_pipeline).
    pub fn create_ray_tracing_pipeline_library(
        &self,
        create_info: RayTracingPipelineLibraryCreateInfo,
    ) -> VkResult<Pipeline> {
        let specialization_infos: Vec<_> = create_info
            .shaders
            .iter()
            .map(|shader| shader.specialization_info.raw_specialization_info())
            .collect();
        let stages: Vec<_> = create_info
            .shaders
            .iter()
            .zip(&specialization_infos)
            .map(
                |(shader, specialization_info)| vk::PipelineShaderStageCreateInfo {
                    flags: shader.flags,
                    stage: shader.stage,
                    module: shader.module.vk_handle(),
                    p_name: shader.entry.as_ptr(),
                    p_specialization_info: specialization_info,
                    ..Default::default()
                },
            )
            .collect();
        let library_interface = vk::RayTracingPipelineInterfaceCreateInfoKHR {
            max_pipeline_ray_hit_attribute_size: create_info.max_hit_attribute_size,
            max_pipeline_ray_payload_size: create_info.max_ray_payload_size,
            ..Default::default()
        };
        let mut vk_create_info = vk::RayTracingPipelineCreateInfoKHR {
            flags: create_info.flags,
            max_pipeline_ray_recursion_depth: create_info.max_ray_recursion_depth,
            layout: create_info.layout.vk_handle(),
            ..Default::default()
        }
        .stages(&stages)
        .groups(create_info.groups)
        .library_interface(&library_interface);
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::RAY_TRACING_PIPELINE_STACK_SIZE_KHR]);
        if create_info.dynamic_stack_size {
            vk_create_info = vk_create_info.dynamic_state(&dynamic_state);
        }

        let mut pipeline = vk::Pipeline::null();
        unsafe {
            (self
                .device()
                .extension::<ash::khr::ray_tracing_pipeline::Meta>()
                .fp()
                .create_ray_tracing_pipelines_khr)(
                self.device().handle(),
                vk::DeferredOperationKHR::null(),
                self.vk_handle(),
                1,
                &vk_create_info,
                std::ptr::null(),
                &mut pipeline,
            )
            .result()?;
        };
        Ok(Pipeline::from_raw(
            self.device().clone(),
            pipeline,
            create_info.layout,
        ))
    }
}

/// Raw shader group handles extracted from a ray tracing pipeline.
///
/// These handles are opaque driver-specific data that must be included in the
/// shader binding table for `vkCmdTraceRays` to locate shader code.
#[derive(Clone)]
pub struct SbtHandles {
    data: Box<[u8]>,
    handle_size: u32,
    total_num_groups: u16,
}
impl SbtHandles {
    fn new(pipeline: &Pipeline, total_num_groups: u16) -> VkResult<SbtHandles> {
        let handle_size = pipeline
            .device()
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>()
            .shader_group_handle_size;
        let data = unsafe {
            pipeline
                .device()
                .extension::<ash::khr::ray_tracing_pipeline::Meta>()
                .get_ray_tracing_shader_group_handles(
                    pipeline.vk_handle(),
                    0,
                    total_num_groups as u32,
                    handle_size as usize * total_num_groups as usize,
                )?
        }
        .into_boxed_slice();
        Ok(SbtHandles {
            data,
            handle_size,
            total_num_groups,
        })
    }

    /// Returns the handle bytes for the shader group at `index`.
    pub fn handle(&self, index: u16) -> &[u8] {
        assert!(index < self.total_num_groups);
        let start = self.handle_size * index as u32;
        let end = start + self.handle_size;
        &self.data[start as usize..end as usize]
    }
}

/// Layout for a shader binding table.
///
/// Defines the size and alignment requirements for each shader stage,
/// based on device properties.
#[derive(Clone)]
pub struct SbtLayout {
    /// Size of each shader group handle in bytes.
    pub handle_size: u32,
    /// Base alignment for the SBT buffer.
    pub base_aligment: u32,
    /// Alignment for individual entries within stages.
    pub entry_alignment: u32,
    /// Layout for ray generation shaders.
    pub raygen: SbtLayoutStage,
    /// Layout for miss shaders.
    pub miss: SbtLayoutStage,
    /// Layout for callable shaders.
    pub callable: SbtLayoutStage,
    /// Layout for hit groups.
    pub hitgroup: SbtLayoutStage,
}
impl SbtLayout {
    /// Creates a new SBT layout based on device ray tracing properties.
    pub fn new(device: &Device) -> Self {
        let properties = device
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        Self {
            handle_size: properties.shader_group_handle_size,
            base_aligment: properties.shader_group_base_alignment,
            entry_alignment: properties.shader_group_handle_alignment,
            raygen: SbtLayoutStage::default(),
            miss: SbtLayoutStage::default(),
            callable: SbtLayoutStage::default(),
            hitgroup: SbtLayoutStage::default(),
        }
    }

    /// Returns the memory layout for ray generation shader entries.
    pub fn raygen_layout(&self) -> Layout {
        unsafe {
            Layout::from_size_align_unchecked(
                (self.handle_size + self.raygen.param_size) as usize,
                self.base_aligment as usize, // Use base alignment for raygen because it's directly passed into the trace_rays call
            )
        }
    }

    /// Returns the memory layout for miss shader entries.
    pub fn miss_layout(&self) -> Layout {
        unsafe {
            Layout::from_size_align_unchecked(
                (self.handle_size + self.miss.param_size) as usize,
                self.entry_alignment as usize,
            )
        }
    }

    /// Returns the memory layout for callable shader entries.
    pub fn callable_layout(&self) -> Layout {
        unsafe {
            Layout::from_size_align_unchecked(
                (self.handle_size + self.callable.param_size) as usize,
                self.entry_alignment as usize,
            )
        }
    }

    /// Returns the memory layout for hit group entries.
    pub fn hitgroup_layout(&self) -> Layout {
        unsafe {
            Layout::from_size_align_unchecked(
                (self.handle_size + self.hitgroup.param_size) as usize,
                self.entry_alignment as usize,
            )
        }
    }
}

/// Layout configuration for a single shader stage in the SBT.
#[derive(Default, Clone)]
pub struct SbtLayoutStage {
    /// Size of inline parameter data per entry (beyond the handle).
    pub param_size: u32,
    /// Number of shaders in this stage.
    pub count: u8,
}

/// A shader binding table for ray tracing dispatch.
///
/// The SBT contains shader group handles and optional per-shader inline data.
/// It is uploaded to a GPU buffer and passed to `vkCmdTraceRays`.
///
/// # Building
///
/// Entries must be pushed in groups by type. Once you start pushing a new type,
/// you cannot go back to push more entries of the previous type.
pub struct ShaderBindingTable {
    sbt_remapper: SbtRemapper,
    handles: SbtHandles,
    layout: SbtLayout,

    /// The raw data of the pipeline
    raygen: Vec<u8>,
    miss: Vec<u8>,
    hitgroup: Vec<u8>,
    callable: Vec<u8>,
    raygen_count: u16,
    miss_count: u16,
    callable_count: u16,
    hitgroup_count: u32,
}

/// Maps a `(library_id, shader_id)` pair to a flat index into the pipeline's
/// shader group handle array.
///
/// `shader_id` is the group's index *within its library* (libraries lay their
/// groups out in canonical RayGen → Miss → Callable → HitGroup order). The
/// remapper stores the running total of group counts per library, so a lookup
/// is just that library's base plus the local `shader_id`.
#[derive(Default, Clone)]
pub struct SbtRemapper(Vec<u16>);
impl SbtRemapper {
    pub fn group_index(&self, library_id: u16, shader_id: u16) -> u16 {
        if library_id == 0 {
            return shader_id;
        };
        self.0[library_id as usize - 1] + shader_id
    }
    pub fn push_library(&mut self, num_groups: u16) {
        let existing = self.0.last().cloned().unwrap_or(0);
        self.0.push(existing + num_groups);
    }
}

impl ShaderBindingTable {
    pub fn new_with_mapper(
        pipeline: &Pipeline,
        layout: SbtLayout,
        mapper: &SbtRemapper,
        reusing_old_sbt: Option<Self>,
    ) -> Self {
        let handles = SbtHandles::new(
            pipeline,
            layout.raygen.count as u16
                + layout.miss.count as u16
                + layout.callable.count as u16
                + layout.hitgroup.count as u16,
        )
        .unwrap();

        // Reuse the previous SBT's heap allocations when one is provided, but
        // clear their contents so recording starts from an empty buffer.
        let (mut raygen, mut miss, mut hitgroup, mut callable, mut sbt_remapper) = reusing_old_sbt
            .map(|old| {
                (
                    old.raygen,
                    old.miss,
                    old.hitgroup,
                    old.callable,
                    old.sbt_remapper,
                )
            })
            .unwrap_or_default();
        raygen.clear();
        miss.clear();
        hitgroup.clear();
        callable.clear();
        sbt_remapper.0.clear();
        sbt_remapper.0.extend_from_slice(&mapper.0);

        Self {
            handles,
            raygen,
            miss,
            hitgroup,
            callable,
            layout,
            sbt_remapper,
            raygen_count: 0,
            miss_count: 0,
            callable_count: 0,
            hitgroup_count: 0,
        }
    }
    fn push_impl(
        &mut self,
        buffer: &mut Vec<u8>,
        group_id: u16,
        write_inline_data: impl FnOnce(&mut [u8]),
        entry_layout: Layout,
    ) {
        let reserved_size = entry_layout.pad_to_align().size();
        let old_len = buffer.len();
        buffer.resize(old_len + reserved_size, 0);

        unsafe {
            // Copy handle
            std::ptr::copy_nonoverlapping(
                self.handles.handle(group_id).as_ptr(),
                buffer.as_mut_ptr().add(old_len),
                self.layout.handle_size as usize,
            );
            if (self.layout.handle_size as usize) < entry_layout.size() {
                // Copy inline data
                let slice = &mut std::slice::from_raw_parts_mut(
                    buffer.as_mut_ptr().add(old_len),
                    entry_layout.size(),
                )[self.layout.handle_size as usize..entry_layout.size()];
                write_inline_data(slice);
            }
        }
    }

    /// Adds a hit group entry to the SBT.
    ///
    /// Returns the index of this entry within the hit group section.
    pub fn push_hitgroup(
        &mut self,
        library_id: u16,
        shader_id: u16,
        write_inline_data: impl FnOnce(&mut [u8]),
    ) -> u32 {
        let group_id = self.sbt_remapper.group_index(library_id, shader_id);

        let mut buffer = std::mem::take(&mut self.hitgroup);
        self.push_impl(
            &mut buffer,
            group_id,
            write_inline_data,
            self.layout.hitgroup_layout(),
        );
        self.hitgroup = buffer;
        let index = self.hitgroup_count;
        self.hitgroup_count += 1;
        index
    }

    /// Adds a ray generation shader entry to the SBT.
    ///
    /// Returns the index of this entry within the raygen section.
    pub fn push_raygen(
        &mut self,
        library_id: u16,
        shader_id: u16,
        write_inline_data: impl FnOnce(&mut [u8]),
    ) -> u16 {
        let group_id = self.sbt_remapper.group_index(library_id, shader_id);

        let mut buffer = std::mem::take(&mut self.raygen);
        self.push_impl(
            &mut buffer,
            group_id,
            write_inline_data,
            self.layout.raygen_layout(),
        );
        self.raygen = buffer;
        let index = self.raygen_count;
        self.raygen_count += 1;
        index
    }

    /// Adds a miss shader entry to the SBT.
    ///
    /// Returns the index of this entry within the miss section.
    pub fn push_miss(
        &mut self,
        library_id: u16,
        shader_id: u16,
        write_inline_data: impl FnOnce(&mut [u8]),
    ) -> u16 {
        let group_id = self.sbt_remapper.group_index(library_id, shader_id);

        let mut buffer = std::mem::take(&mut self.miss);
        self.push_impl(
            &mut buffer,
            group_id,
            write_inline_data,
            self.layout.miss_layout(),
        );
        self.miss = buffer;
        let index = self.miss_count;
        self.miss_count += 1;
        index
    }

    /// Adds a callable shader entry to the SBT.
    ///
    /// Returns the index of this entry within the callable section.
    pub fn push_callable(
        &mut self,
        library_id: u16,
        shader_id: u16,
        write_inline_data: impl FnOnce(&mut [u8]),
    ) -> u16 {
        let group_id = self.sbt_remapper.group_index(library_id, shader_id);

        let mut buffer = std::mem::take(&mut self.callable);
        self.push_impl(
            &mut buffer,
            group_id,
            write_inline_data,
            self.layout.callable_layout(),
        );
        self.callable = buffer;
        let index = self.callable_count;
        self.callable_count += 1;
        index
    }

    /// Returns the raw SBT data to be uploaded to a GPU buffer.
    pub fn write_buffer(&self, slice: &mut [u8]) {
        slice[0..self.raygen.len()].copy_from_slice(&self.raygen);
        let slice = &mut slice[self
            .raygen
            .len()
            .next_multiple_of(self.layout.base_aligment as usize)..];

        slice[0..self.miss.len()].copy_from_slice(&self.miss);
        let slice = &mut slice[self
            .miss
            .len()
            .next_multiple_of(self.layout.base_aligment as usize)..];

        slice[0..self.hitgroup.len()].copy_from_slice(&self.hitgroup);
        let slice = &mut slice[self
            .hitgroup
            .len()
            .next_multiple_of(self.layout.base_aligment as usize)..];

        slice[0..self.callable.len()].copy_from_slice(&self.callable);
    }

    /// Returns the memory layout requirements for the SBT buffer.
    pub fn layout(&self) -> Layout {
        let mut layout =
            Layout::from_size_align(self.raygen.len(), self.layout.base_aligment as usize).unwrap();
        layout = layout
            .extend(
                Layout::from_size_align(self.miss.len(), self.layout.base_aligment as usize)
                    .unwrap(),
            )
            .unwrap()
            .0;
        layout = layout
            .extend(
                Layout::from_size_align(self.hitgroup.len(), self.layout.base_aligment as usize)
                    .unwrap(),
            )
            .unwrap()
            .0;
        layout = layout
            .extend(
                Layout::from_size_align(self.callable.len(), self.layout.base_aligment as usize)
                    .unwrap(),
            )
            .unwrap()
            .0;
        layout
    }
}

impl<'a> CommandEncoder<'a> {
    /// Dispatches rays for ray tracing.
    ///
    /// # Parameters
    ///
    /// - `shader_binding_table`: The SBT containing shader handles
    /// - `raygen_shader`: Index of the ray generation shader to invoke
    /// - `buffer`: GPU buffer containing the SBT data
    /// - `size`: Dispatch dimensions (width, height, depth)
    pub fn trace_rays(
        &mut self,
        shader_binding_table: &ShaderBindingTable,
        raygen_shader: u32,
        buffer: &'a impl BufferLike,
        size: UVec3,
    ) {
        let raygen_stride = shader_binding_table
            .layout
            .raygen_layout()
            .pad_to_align()
            .size() as u32;
        let miss_stride = shader_binding_table
            .layout
            .miss_layout()
            .pad_to_align()
            .size() as u32;
        let hitgroup_stride = shader_binding_table
            .layout
            .hitgroup_layout()
            .pad_to_align()
            .size() as u32;
        let callable_stride = shader_binding_table
            .layout
            .callable_layout()
            .pad_to_align()
            .size() as u32;

        // Each stage's section is laid out back-to-back in the buffer, each
        // padded up to the base alignment (matching `write_buffer` and
        // `ShaderBindingTable::layout`).
        let base = buffer.device_address();
        let align = shader_binding_table.layout.base_aligment as usize;
        let raygen_offset = 0u64;
        let miss_offset =
            raygen_offset + shader_binding_table.raygen.len().next_multiple_of(align) as u64;
        let hitgroup_offset =
            miss_offset + shader_binding_table.miss.len().next_multiple_of(align) as u64;
        let callable_offset =
            hitgroup_offset + shader_binding_table.hitgroup.len().next_multiple_of(align) as u64;

        unsafe {
            self.device()
                .extension::<ash::khr::ray_tracing_pipeline::Meta>()
                .cmd_trace_rays(
                    self.buffer().buffer,
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: base
                            + raygen_offset
                            + (raygen_stride * raygen_shader) as u64,
                        stride: raygen_stride as u64,
                        size: raygen_stride as u64,
                    },
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: base + miss_offset,
                        stride: miss_stride as u64,
                        size: shader_binding_table.miss.len() as u64,
                    },
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: base + hitgroup_offset,
                        stride: hitgroup_stride as u64,
                        size: shader_binding_table.hitgroup.len() as u64,
                    },
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: base + callable_offset,
                        stride: callable_stride as u64,
                        size: shader_binding_table.callable.len() as u64,
                    },
                    size.x,
                    size.y,
                    size.z,
                );
        }
    }
}

/// A Vulkan acceleration structure for ray tracing.
///
/// Acceleration structures are spatial data structures that enable fast ray-scene
/// intersection. There are two types:
///
/// - **BLAS (Bottom-Level)**: Contains geometry (triangles or AABBs)
/// - **TLAS (Top-Level)**: Contains instances of BLASes with transforms
///
/// The structure owns its backing buffer and is automatically destroyed on drop.
pub struct AccelStruct<T: BufferLike = Buffer> {
    device: Device,
    buffer: T,
    pub(crate) raw: vk::AccelerationStructureKHR,
    pub(crate) flags: vk::BuildAccelerationStructureFlagsKHR,
    pub(crate) device_address: vk::DeviceAddress,
}
impl<T: BufferLike> Drop for AccelStruct<T> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .extension::<AccelerationStructureExt>()
                .destroy_acceleration_structure(self.raw, None);
        }
    }
}
impl<T: BufferLike> HasDevice for AccelStruct<T> {
    fn device(&self) -> &Device {
        &self.device
    }
}
impl<T: BufferLike> AsVkHandle for AccelStruct<T> {
    fn vk_handle(&self) -> vk::AccelerationStructureKHR {
        self.raw
    }
    type Handle = vk::AccelerationStructureKHR;
}
impl<T: BufferLike> AccelStruct<T> {
    /// Returns the build flags used when this structure was built.
    pub fn flags(&self) -> vk::BuildAccelerationStructureFlagsKHR {
        self.flags
    }

    /// Returns the device address for use in shaders and TLAS instances.
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    /// Returns the size of the backing buffer in bytes.
    pub fn size(&self) -> vk::DeviceSize {
        self.buffer.size()
    }

    /// Creates an acceleration structure on an existing buffer.
    pub fn create_on_buffer(
        device: Device,
        buffer: T,
        ty: vk::AccelerationStructureTypeKHR,
        build_flags: vk::BuildAccelerationStructureFlagsKHR,
    ) -> VkResult<Self> {
        unsafe {
            let raw = device
                .extension::<AccelerationStructureExt>()
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR {
                        ty,
                        size: buffer.size(),
                        offset: buffer.offset(),
                        buffer: buffer.vk_handle(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let device_address = device
                .extension::<AccelerationStructureExt>()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR {
                        acceleration_structure: raw,
                        ..Default::default()
                    },
                );
            Ok(Self {
                device,
                buffer,
                raw,
                flags: build_flags,
                device_address,
            })
        }
    }
}

impl AccelStruct {
    /// Creates a new acceleration structure with a freshly allocated buffer.
    ///
    /// The buffer is allocated with appropriate usage flags for acceleration
    /// structure storage and shader device address access.
    pub fn new(
        allocator: Allocator,
        size: vk::DeviceSize,
        ty: vk::AccelerationStructureTypeKHR,
        build_flags: vk::BuildAccelerationStructureFlagsKHR,
    ) -> VkResult<Self> {
        let mut buffer = Buffer::new_private(
            allocator,
            size,
            1,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;

        let name = if ty == vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL {
            c"BLAS backing buffer"
        } else {
            c"TLAS backing buffer"
        };
        buffer.set_name(name);
        Self::create_on_buffer(buffer.device().clone(), buffer, ty, build_flags)
    }
}
