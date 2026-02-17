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
//! The [`ShaderBindingTable`] manages shader handles and per-shader data:
//!
//! ```ignore
//! let mut sbt = ShaderBindingTable::new(&pipeline, layout);
//! sbt.push_raygen(0, |_| {});
//! sbt.push_miss(0, |_| {});
//! sbt.push_hitgroup(0, |data: &mut [u8]| { /* write inline data */ });
//! ```

use std::{alloc::Layout, fmt::Debug, ops::Deref, sync::Arc};

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
    num_raygen: u8,
    num_miss: u8,
    num_callable: u8,
    num_hitgroup: u8,
}
impl Debug for SbtHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct HexStr<'a>(&'a [u8]);
        impl<'a> Debug for HexStr<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("0x")?;
                for item in self.0 {
                    f.write_fmt(format_args!("{:x?}", item))?;
                }
                Ok(())
            }
        }
        let mut f_struct = f.debug_struct("SbtHandles");
        f_struct.field("handle_size", &self.handle_size);
        f_struct.field(
            "raygen",
            &(0..self.num_raygen)
                .map(|i| HexStr(self.rgen(i as u32)))
                .collect::<Vec<_>>(),
        );
        f_struct.field(
            "miss",
            &(0..self.num_miss)
                .map(|i| HexStr(self.rmiss(i as u32)))
                .collect::<Vec<_>>(),
        );
        f_struct.field(
            "callable",
            &(0..self.num_callable)
                .map(|i| HexStr(self.callable(i as u32)))
                .collect::<Vec<_>>(),
        );
        f_struct.field(
            "hitgroup",
            &(0..self.num_hitgroup)
                .map(|i| HexStr(self.hitgroup(i as u32)))
                .collect::<Vec<_>>(),
        );
        Ok(())
    }
}
impl SbtHandles {
    fn new(
        pipeline: &Pipeline,
        num_raygen: u8,
        num_miss: u8,
        num_callable: u8,
        num_hitgroup: u8,
    ) -> VkResult<SbtHandles> {
        let total_num_groups =
            num_hitgroup as u32 + num_miss as u32 + num_callable as u32 + num_raygen as u32;
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
                    total_num_groups,
                    handle_size as usize * total_num_groups as usize,
                )?
        }
        .into_boxed_slice();
        Ok(SbtHandles {
            data,
            handle_size,
            num_raygen,
            num_miss,
            num_callable,
            num_hitgroup,
        })
    }

    /// Returns the handle bytes for a ray generation shader.
    pub fn rgen(&self, index: u32) -> &[u8] {
        assert!(index < self.num_raygen as u32);
        // Note that
        // VUID-vkGetRayTracingShaderGroupHandlesKHR-dataSize-02420
        // dataSize must be at least VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize × groupCount
        // This implies all handles are tightly packed. No need to call `pad_to_align` here
        let start = self.handle_size * index;
        let end = start + self.handle_size;
        &self.data[start as usize..end as usize]
    }

    /// Returns the handle bytes for a miss shader.
    pub fn rmiss(&self, index: u32) -> &[u8] {
        assert!(index < self.num_miss as u32);
        let start = self.handle_size * (index + self.num_raygen as u32);
        let end = start + self.handle_size;
        &self.data[start as usize..end as usize]
    }

    /// Returns the handle bytes for a callable shader.
    pub fn callable(&self, index: u32) -> &[u8] {
        assert!(index < self.num_callable as u32);
        let start = self.handle_size * (index + self.num_raygen as u32 + self.num_miss as u32);
        let end = start + self.handle_size;
        &self.data[start as usize..end as usize]
    }

    /// Returns the handle bytes for a hit group.
    pub fn hitgroup(&self, index: u32) -> &[u8] {
        assert!(index < self.num_hitgroup as u32);
        let start = self.handle_size
            * (index + self.num_miss as u32 + self.num_callable as u32 + self.num_raygen as u32);
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
    hitgroup_id_mapper: u64,
    handles: SbtHandles,
    layout: SbtLayout,

    /// The raw data of the pipeline
    buffer: Vec<u8>,

    raygen_index: ShaderBindingTableState,
    miss_index: ShaderBindingTableState,
    callable_index: ShaderBindingTableState,
    hitgroup_index: ShaderBindingTableState,
}
enum ShaderBindingTableState {
    NotSet,
    Recording { base_offset: u64, index: u32 },
    Recorded { base_offset: u64, index: u32 },
}
impl ShaderBindingTableState {
    /// It'll save the base_offset for the first time this was called.
    fn increment(&mut self, base_offset: u64) -> u32 {
        match self {
            Self::NotSet => {
                *self = Self::Recording {
                    index: 1,
                    base_offset,
                };
                0
            }
            Self::Recording { index, .. } => {
                let current = *index;
                *index += 1;
                current
            }
            Self::Recorded { .. } => {
                panic!("Must record types of SBT entries consecutively!")
            }
        }
    }
    fn end(&mut self, buffer: &mut Vec<u8>, alignment: u32) {
        if let Self::Recording { base_offset, index } = self {
            buffer.resize(buffer.len().next_multiple_of(alignment as usize), 0);
            *self = Self::Recorded {
                base_offset: *base_offset,
                index: *index,
            };
        }
    }
    fn index(&self) -> u32 {
        match self {
            Self::NotSet => 0,
            Self::Recording { index, .. } | Self::Recorded { index, .. } => *index,
        }
    }
    fn base_offset(&self) -> u64 {
        match self {
            Self::NotSet => 0,
            Self::Recording { base_offset, .. } | Self::Recorded { base_offset, .. } => {
                *base_offset
            }
        }
    }
}

impl ShaderBindingTable {
    /// Creates a new shader binding table for the given pipeline and layout.
    pub fn new(pipeline: &Pipeline, layout: SbtLayout) -> Self {
        Self {
            handles: SbtHandles::new(
                pipeline,
                layout.raygen.count,
                layout.miss.count,
                layout.callable.count,
                layout.hitgroup.count,
            )
            .unwrap(),
            hitgroup_index: ShaderBindingTableState::NotSet,
            callable_index: ShaderBindingTableState::NotSet,
            miss_index: ShaderBindingTableState::NotSet,
            raygen_index: ShaderBindingTableState::NotSet,
            buffer: Vec::new(),
            layout,
            hitgroup_id_mapper: u64::MAX,
        }
    }
    pub fn new_with_mapper(
        pipeline: &Pipeline,
        layout: SbtLayout,
        mapper: u64,
        reusing_old_sbt: Option<Self>,
    ) -> Self {
        let handles = SbtHandles::new(
            pipeline,
            layout.raygen.count,
            layout.miss.count,
            layout.callable.count,
            layout.hitgroup.count,
        )
        .unwrap();
        Self {
            handles: handles,
            hitgroup_index: ShaderBindingTableState::NotSet,
            callable_index: ShaderBindingTableState::NotSet,
            miss_index: ShaderBindingTableState::NotSet,
            raygen_index: ShaderBindingTableState::NotSet,
            buffer: reusing_old_sbt
                .map(|mut x| {
                    x.buffer.clear();
                    x.buffer
                })
                .unwrap_or_default(),
            layout,
            hitgroup_id_mapper: mapper,
        }
    }
    fn push_impl(
        &mut self,
        hitgroup_id: u32,
        write_inline_data: impl FnOnce(&mut [u8]),
        handle: fn(&SbtHandles, u32) -> &[u8],
        entry_layout: Layout,
    ) {
        let reserved_size = entry_layout.pad_to_align().size();
        let old_len = self.buffer.len();
        self.buffer.resize(old_len + reserved_size, 0);

        unsafe {
            // Copy handle
            std::ptr::copy_nonoverlapping(
                (handle)(&self.handles, hitgroup_id).as_ptr(),
                self.buffer.as_mut_ptr().add(old_len),
                self.layout.handle_size as usize,
            );
            if (self.layout.handle_size as usize) < entry_layout.size() {
                // Copy inline data
                let slice = &mut std::slice::from_raw_parts_mut(
                    self.buffer.as_mut_ptr().add(old_len),
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
        hitgroup_id: u32,
        write_inline_data: impl FnOnce(&mut [u8]),
    ) -> u32 {
        // Close any other shader types
        self.raygen_index
            .end(&mut self.buffer, self.layout.base_aligment);
        self.miss_index
            .end(&mut self.buffer, self.layout.base_aligment);
        self.callable_index
            .end(&mut self.buffer, self.layout.base_aligment);
        let hitgroup_id = (self.hitgroup_id_mapper & ((1 << hitgroup_id) - 1)).count_ones();

        let base_offset = self.buffer.len() as u64;
        self.push_impl(
            hitgroup_id,
            write_inline_data,
            SbtHandles::hitgroup,
            self.layout.hitgroup_layout(),
        );
        self.hitgroup_index.increment(base_offset)
    }

    /// Adds a ray generation shader entry to the SBT.
    ///
    /// Returns the index of this entry within the raygen section.
    pub fn push_raygen(
        &mut self,
        shader_id: u32,
        write_inline_data: impl FnOnce(&mut [u8]),
    ) -> u32 {
        // Close any other shader types
        self.hitgroup_index
            .end(&mut self.buffer, self.layout.base_aligment);
        self.miss_index
            .end(&mut self.buffer, self.layout.base_aligment);
        self.callable_index
            .end(&mut self.buffer, self.layout.base_aligment);

        let base_offset = self.buffer.len() as u64;
        self.push_impl(
            shader_id,
            write_inline_data,
            SbtHandles::rgen,
            self.layout.raygen_layout(),
        );
        self.raygen_index.increment(base_offset)
    }

    /// Adds a miss shader entry to the SBT.
    ///
    /// Returns the index of this entry within the miss section.
    pub fn push_miss(&mut self, shader_id: u32, write_inline_data: impl FnOnce(&mut [u8])) -> u32 {
        // Close any other shader types
        self.raygen_index
            .end(&mut self.buffer, self.layout.base_aligment);
        self.hitgroup_index
            .end(&mut self.buffer, self.layout.base_aligment);
        self.callable_index
            .end(&mut self.buffer, self.layout.base_aligment);

        let base_offset = self.buffer.len() as u64;
        self.push_impl(
            shader_id,
            write_inline_data,
            SbtHandles::rmiss,
            self.layout.miss_layout(),
        );
        self.miss_index.increment(base_offset)
    }

    /// Adds a callable shader entry to the SBT.
    ///
    /// Returns the index of this entry within the callable section.
    pub fn push_callable(
        &mut self,
        shader_id: u32,
        write_inline_data: impl FnOnce(&mut [u8]),
    ) -> u32 {
        // Close any other shader types
        self.raygen_index
            .end(&mut self.buffer, self.layout.base_aligment);
        self.miss_index
            .end(&mut self.buffer, self.layout.base_aligment);
        self.hitgroup_index
            .end(&mut self.buffer, self.layout.base_aligment);

        let base_offset = self.buffer.len() as u64;
        self.push_impl(
            shader_id,
            write_inline_data,
            SbtHandles::callable,
            self.layout.callable_layout(),
        );
        self.callable_index.increment(base_offset)
    }

    /// Returns the raw SBT data to be uploaded to a GPU buffer.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Returns the memory layout requirements for the SBT buffer.
    pub fn layout(&self) -> Layout {
        Layout::from_size_align(self.buffer().len(), self.layout.base_aligment as usize).unwrap()
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
        unsafe {
            self.device()
                .extension::<ash::khr::ray_tracing_pipeline::Meta>()
                .cmd_trace_rays(
                    self.buffer().buffer,
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: buffer.device_address()
                            + shader_binding_table.raygen_index.base_offset()
                            + (raygen_stride * raygen_shader) as u64,
                        stride: raygen_stride as u64,
                        size: raygen_stride as u64,
                    },
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: buffer.device_address()
                            + shader_binding_table.miss_index.base_offset(),
                        stride: miss_stride as u64,
                        size: (shader_binding_table.miss_index.index() * miss_stride) as u64,
                    },
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: buffer.device_address()
                            + shader_binding_table.hitgroup_index.base_offset(),
                        stride: hitgroup_stride as u64,
                        size: (shader_binding_table.hitgroup_index.index() * hitgroup_stride)
                            as u64,
                    },
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: buffer.device_address()
                            + shader_binding_table.callable_index.base_offset(),
                        stride: callable_stride as u64,
                        size: (shader_binding_table.callable_index.index() * callable_stride)
                            as u64,
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
                flags: vk::BuildAccelerationStructureFlagsKHR::empty(),
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
        Self::create_on_buffer(buffer.device().clone(), buffer, ty)
    }
}
