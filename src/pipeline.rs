//! Vulkan pipeline management.
//!
//! This module provides safe wrappers for creating and managing Vulkan pipelines,
//! shader modules, pipeline layouts, and pipeline caches.
//!
//! # Key Types
//!
//! - [`Pipeline`]: A compiled graphics or compute pipeline.
//! - [`ShaderModule`]: A compiled SPIR-V shader module.
//! - [`PipelineLayout`]: Defines the interface between shaders and descriptor sets.
//! - [`PipelineCache`]: Caches compiled pipeline data for faster subsequent loads.
//! - [`SpecializationInfo`]: Compile-time constants for shader specialization.
//!
//! # Pipeline Creation
//!
//! Pipelines are created through [`PipelineCache`]:
//!
//! ```ignore
//! // Create a compute pipeline
//! let pipeline = cache.create_compute_pipeline(
//!     layout,
//!     vk::PipelineCreateFlags::empty(),
//!     &shader_entry,
//! )?;
//!
//! // Create a graphics pipeline
//! let pipeline = cache.create_graphics_pipeline(layout, &create_info)?;
//! ```
//!
//! # Specialization Constants
//!
//! Use [`SpecializationInfo`] to provide compile-time constants:
//!
//! ```
//! # use pumicite::pipeline::SpecializationInfo;
//! let mut spec = SpecializationInfo::new();
//! spec.push(0, 16u32);      // constant_id 0 = 16
//! spec.push(1, true);       // constant_id 1 = true (converted to VkBool32)
//! ```

use std::{borrow::Cow, ffi::CStr, mem::MaybeUninit, sync::Arc};

use ash::{
    VkResult,
    vk::{self, Handle},
};

use crate::{Device, HasDevice, descriptor::DescriptorSetLayout, utils::AsVkHandle};

/// A compiled Vulkan pipeline state object (PSO).
///
/// Pipelines encapsulate all the state needed to execute shaders on the GPU.
/// They are created via [`PipelineCache::create_compute_pipeline`] or
/// [`PipelineCache::create_graphics_pipeline`].
pub struct Pipeline {
    device: Device,
    handle: vk::Pipeline,
    layout: Arc<PipelineLayout>,
}
impl HasDevice for Pipeline {
    fn device(&self) -> &Device {
        &self.device
    }
}
impl Pipeline {
    /// Creates a pipeline from raw Vulkan handles.
    pub fn from_raw(device: Device, raw: vk::Pipeline, layout: Arc<PipelineLayout>) -> Self {
        Self {
            device,
            handle: raw,
            layout,
        }
    }

    /// Returns the pipeline layout associated with this pipeline.
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }
}
impl AsVkHandle for Pipeline {
    type Handle = vk::Pipeline;

    fn vk_handle(&self) -> Self::Handle {
        self.handle
    }
}
impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
        }
    }
}

/// Represents SPIR-V shader source.
///
/// Shader modules contain SPIR-V shader source that can be used to create
/// PSOs. The SPIR-V bytecode must be properly aligned (4-byte).
pub struct ShaderModule {
    device: Device,
    handle: vk::ShaderModule,
}
impl ShaderModule {
    /// Creates a shader module from SPIR-V bytecode.
    ///
    /// # Errors
    ///
    /// Returns an error if the code length is not a multiple of 4 bytes.
    pub fn new(device: Device, code: &[u8]) -> VkResult<Self> {
        if !code.len().is_multiple_of(4) {
            return VkResult::Err(vk::Result::ERROR_INVALID_SHADER_NV);
        };
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    p_code: code.as_ptr() as *const u32,
                    code_size: std::mem::size_of_val(code),
                    ..Default::default()
                },
                None,
            )?
        };
        Ok(Self {
            device,
            handle: module,
        })
    }
}
impl AsVkHandle for ShaderModule {
    type Handle = vk::ShaderModule;
    fn vk_handle(&self) -> Self::Handle {
        self.handle
    }
}
impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.handle, None);
        }
    }
}

/// Configuration for a shader stage in a pipeline.
///
/// Specifies the shader module, entry point, stage, and specialization constants
/// for a single pipeline stage.
pub struct ShaderEntry<'a> {
    /// The SPIR-V shader source.
    pub module: Arc<ShaderModule>,
    /// The entry point function name (e.g., "main").
    pub entry: Cow<'a, CStr>,
    /// Stage creation flags.
    pub flags: vk::PipelineShaderStageCreateFlags,
    /// The shader stage (vertex, fragment, compute, etc.).
    pub stage: vk::ShaderStageFlags,
    /// Specialization constants for this stage.
    pub specialization_info: Cow<'a, SpecializationInfo>,
}

/// A cache for compiled pipeline data.
///
/// Pipeline caches store compiled pipeline state to speed up subsequent pipeline
/// creation. The cache data can be saved to disk and reloaded across application
/// runs using [`get_data`](Self::get_data) and [`from_initial_data`](Self::from_initial_data).
///
/// A null cache ([`null`](Self::null)) can be used when caching is not desired.
pub struct PipelineCache {
    device: Device,

    /// May be null.
    handle: vk::PipelineCache,
}
impl Drop for PipelineCache {
    fn drop(&mut self) {
        if self.handle != vk::PipelineCache::null() {
            unsafe {
                self.device.destroy_pipeline_cache(self.handle, None);
            }
        }
    }
}
impl HasDevice for PipelineCache {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl PipelineCache {
    /// Returns a null pipeline cache. The null pipeline cache doesn't perform any caching.
    pub fn null(device: Device) -> Self {
        Self {
            device,
            handle: vk::PipelineCache::null(),
        }
    }
    /// Creates an empty pipeline cache.
    pub fn empty(device: Device) -> VkResult<Self> {
        let cache = unsafe {
            device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo {
                    flags: vk::PipelineCacheCreateFlags::empty(),
                    ..Default::default()
                },
                None,
            )
        }?;
        Ok(Self {
            device,
            handle: cache,
        })
    }
    /// Creates a pipeline cache from some initial data. The data can be obtained from an existing [`PipelineCache`]
    /// using [`PipelineCache::get_data`]
    pub fn from_initial_data(device: Device, initial_data: &[u8]) -> VkResult<Self> {
        let cache = unsafe {
            device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo {
                    flags: vk::PipelineCacheCreateFlags::empty(),
                    ..Default::default()
                }
                .initial_data(initial_data),
                None,
            )
        }?;
        Ok(Self {
            device,
            handle: cache,
        })
    }
    /// Combine the data stores of pipeline caches
    pub fn merge(&mut self, other: &Self) -> VkResult<()> {
        if self.handle.is_null() {
            return VkResult::Ok(());
        }
        assert_eq!(self.device, other.device);
        unsafe {
            self.device
                .merge_pipeline_caches(self.handle, &[other.handle])
        }
    }
    /// Get the data store from a pipeline cache
    pub fn get_data(&self) -> VkResult<Box<[u8]>> {
        if self.handle.is_null() {
            return Ok(Box::new([]));
        }
        unsafe {
            let mut size = 0;
            (self.device.fp_v1_0().get_pipeline_cache_data)(
                self.device.handle(),
                self.handle,
                &mut size,
                std::ptr::null_mut(),
            )
            .result()?;
            let mut dst: Box<[MaybeUninit<u8>]> = Box::new_uninit_slice(size);
            (self.device.fp_v1_0().get_pipeline_cache_data)(
                self.device.handle(),
                self.handle,
                &mut size,
                dst.as_mut_ptr() as *mut _,
            )
            .result()?;
            Ok(dst.assume_init())
        }
    }
}

impl AsVkHandle for PipelineCache {
    type Handle = vk::PipelineCache;
    fn vk_handle(&self) -> Self::Handle {
        self.handle
    }
}

impl PipelineCache {
    /// Creates a compute pipeline.
    ///
    /// The pipeline is compiled using the provided shader entry and layout.
    pub fn create_compute_pipeline(
        &self,
        layout: Arc<PipelineLayout>,
        flags: vk::PipelineCreateFlags,
        shader: &ShaderEntry,
    ) -> VkResult<Pipeline> {
        let specialization_info = shader.specialization_info.raw_specialization_info();
        let create_info = vk::ComputePipelineCreateInfo {
            flags,
            layout: layout.vk_handle(),
            stage: vk::PipelineShaderStageCreateInfo {
                flags: shader.flags,
                stage: shader.stage,
                module: shader.module.vk_handle(),
                p_name: shader.entry.as_ptr(),
                p_specialization_info: &specialization_info,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut pipeline = vk::Pipeline::null();
        unsafe {
            (self.device().fp_v1_0().create_compute_pipelines)(
                self.device().handle(),
                self.vk_handle(),
                1,
                &create_info,
                std::ptr::null(),
                &mut pipeline,
            )
            .result()?;
        };

        Ok(Pipeline::from_raw(self.device().clone(), pipeline, layout))
    }

    /// Creates a graphics pipeline.
    ///
    /// The `create_info.layout` must match the provided `layout` parameter.
    pub fn create_graphics_pipeline(
        &self,
        layout: Arc<PipelineLayout>,
        create_info: &vk::GraphicsPipelineCreateInfo,
    ) -> VkResult<Pipeline> {
        assert_eq!(create_info.layout, layout.vk_handle());
        let mut pipeline = vk::Pipeline::null();
        unsafe {
            (self.device().fp_v1_0().create_graphics_pipelines)(
                self.device().handle(),
                self.vk_handle(),
                1,
                create_info,
                std::ptr::null(),
                &mut pipeline,
            )
            .result()?;
        }
        Ok(Pipeline::from_raw(self.device().clone(), pipeline, layout))
    }
}

/// A pipeline layout defining the shader interface.
///
/// Pipeline layouts specify the descriptor set layouts and push constant ranges
/// that shaders in the pipeline will use.
#[derive(Clone)]
pub struct PipelineLayout {
    device: Device,
    handle: vk::PipelineLayout,
    /// Keep descriptor set layouts alive
    _descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>
}

impl PipelineLayout {
    /// Creates a new pipeline layout.
    ///
    /// # Parameters
    ///
    /// - `set_layouts`: The descriptor set layouts for each set index.
    /// - `push_constant_ranges`: Push constant ranges accessible to shaders.
    /// - `flags`: Layout creation flags.
    pub fn new<'a>(
        device: Device,
        set_layouts:  Vec<Arc<DescriptorSetLayout>>,
        push_constant_ranges: &[vk::PushConstantRange],
        flags: vk::PipelineLayoutCreateFlags,
    ) -> VkResult<Self> {
        let raw_set_layouts: Vec<_> = set_layouts.iter().map(|a| a.vk_handle()).collect();
        let info = vk::PipelineLayoutCreateInfo {
            flags,
            set_layout_count: raw_set_layouts.len() as u32,
            p_set_layouts: raw_set_layouts.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as u32,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
            ..Default::default()
        };

        let layout = unsafe { device.create_pipeline_layout(&info, None)? };
        Ok(Self {
            device,
            handle: layout,
            _descriptor_set_layouts: set_layouts,
        })
    }
}
impl AsVkHandle for PipelineLayout {
    type Handle = vk::PipelineLayout;

    fn vk_handle(&self) -> Self::Handle {
        self.handle
    }
}
impl HasDevice for PipelineLayout {
    fn device(&self) -> &Device {
        &self.device
    }
}
impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.handle, None);
        }
    }
}

/// Specialization constants for shader compilation.
///
/// Specialization constants allow setting compile-time values in SPIR-V shaders.
/// This enables the driver to optimize the shader based on known constant values.
///
/// # Example
///
/// ```
/// use pumicite::pipeline::SpecializationInfo;
///
/// let mut spec = SpecializationInfo::new();
/// spec.push(0, 256u32);     // Set constant_id 0 to 256
/// spec.push(1, false);      // Set constant_id 1 to false
/// ```
///
/// # Boolean Handling
///
/// Rust `bool` values are automatically converted to `VkBool32` (4 bytes) to match
/// the SPIR-V `OpSpecConstantTrue`/`OpSpecConstantFalse` representation.
#[derive(Clone, Default, Debug)]
pub struct SpecializationInfo {
    pub(super) data: Vec<u8>,
    pub(super) entries: Vec<vk::SpecializationMapEntry>,
}
impl SpecializationInfo {
    /// Returns the raw constant data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns the specialization map entries.
    pub fn entries(&self) -> &[vk::SpecializationMapEntry] {
        &self.entries
    }

    /// Creates an empty specialization info.
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// Adds a specialization constant.
    ///
    /// The constant is appended to the data buffer and a map entry is created.
    /// Boolean values are automatically converted to `VkBool32`.
    pub fn push<T: Copy + 'static>(&mut self, constant_id: u32, item: T) {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
            unsafe {
                let value: bool = std::mem::transmute_copy(&item);
                self.push_bool(constant_id, value);
                return;
            }
        }
        let size = std::mem::size_of::<T>();
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset: self.data.len() as u32,
            size,
        });
        self.data.reserve(size);
        unsafe {
            let target_ptr = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(&item as *const T as *const u8, target_ptr, size);
            self.data.set_len(self.data.len() + size);
        }
    }
    fn push_bool(&mut self, constant_id: u32, item: bool) {
        let size = std::mem::size_of::<vk::Bool32>();
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset: self.data.len() as u32,
            size,
        });
        self.data.reserve(size);
        unsafe {
            let item: vk::Bool32 = if item { vk::TRUE } else { vk::FALSE };
            let target_ptr = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(
                &item as *const vk::Bool32 as *const u8,
                target_ptr,
                size,
            );
            self.data.set_len(self.data.len() + size);
        }
    }

    /// Converts to a Vulkan specialization info structure.
    pub fn raw_specialization_info(&self) -> vk::SpecializationInfo<'_> {
        vk::SpecializationInfo::default()
            .map_entries(&self.entries)
            .data(&self.data)
    }
}
