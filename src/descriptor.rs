//! Vulkan descriptor set and descriptor pool management.
//!
//! This module provides safe wrappers around Vulkan descriptors,
//! including descriptor set layouts and descriptor pools.
//!
//! # Key Types
//!
//! - [`DescriptorSetLayout`]: A descriptor set layout that can be either reference-counted
//!   (shared) or device-owned (for internal/cached layouts).
//! - [`DescriptorPool`]: A pool for allocating descriptor sets, with support for
//!   variable-count descriptors.
//!
//! # Example
//!
//! ```
//! # use pumicite::{Device, ash::vk, descriptor::{DescriptorSetLayout, DescriptorPool}};
//! # let (device, queue) = Device::create_system_default().unwrap();
//! // Create a layout with a single storage buffer binding
//! let layout = DescriptorSetLayout::new(
//!     device.clone(),
//!     &[vk::DescriptorSetLayoutBinding {
//!         binding: 0,
//!         descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
//!         descriptor_count: 1,
//!         stage_flags: vk::ShaderStageFlags::COMPUTE,
//!         ..Default::default()
//!     }],
//!     &[],  // no binding flags
//!     &[],  // no mutability
//!     vk::DescriptorSetLayoutCreateFlags::empty(),
//! ).unwrap();
//!
//! // Create a pool and allocate a descriptor set
//! let mut pool = DescriptorPool::new(
//!     device,
//!     &[vk::DescriptorPoolSize {
//!         ty: vk::DescriptorType::STORAGE_BUFFER,
//!         descriptor_count: 1,
//!     }],
//!     1,  // max_sets
//!     vk::DescriptorPoolCreateFlags::empty(),
//! ).unwrap();
//! let descriptor_set = pool.allocate_one(&layout).unwrap();
//! ```

use ash::{
    VkResult,
    vk::{self, TaggedStructure},
};

use crate::{Device, HasDevice, utils::AsVkHandle};

/// A descriptor set layout.
#[derive(Clone)]
pub struct DescriptorSetLayout {
    device: Device,
    handle: vk::DescriptorSetLayout,
}
impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.handle, None);
        }
    }
}
impl DescriptorSetLayout {
    /// Creates a descriptor set layout with N bindings.
    ///
    /// # Parameters
    /// - `binding_infos`: A slice of length `N` describing the bindings to be created in this descriptor set layout.
    /// - `binding_flags`: An empty slice, or a slice of length `N` with flags annotating the bindings.
    /// - `mutability`: An empty slice, or if any of the bindings are mutable, a slice of length N describing the
    ///   descriptor types that the binding may mutate into.
    pub fn new(
        device: Device,
        binding_infos: &[vk::DescriptorSetLayoutBinding],
        binding_flags: &[vk::DescriptorBindingFlags],
        mutability: &[&[vk::DescriptorType]],
        flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> VkResult<Self> {
        assert!(binding_flags.is_empty() || binding_flags.len() == binding_infos.len());
        assert!(mutability.is_empty() || mutability.len() == mutability.len());
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(binding_flags);

        let lists: Vec<_> = mutability
            .iter()
            .map(|x| vk::MutableDescriptorTypeListEXT::default().descriptor_types(x))
            .collect();
        let mut mutability_info =
            vk::MutableDescriptorTypeCreateInfoEXT::default().mutable_descriptor_type_lists(&lists);
        let mut info = vk::DescriptorSetLayoutCreateInfo {
            flags,
            ..Default::default()
        }
        .bindings(binding_infos);
        if !binding_flags.is_empty() {
            info = info.push(&mut flags_info);
        }
        if !mutability.is_empty() {
            info = info.push(&mut mutability_info);
        }
        let raw = unsafe { device.create_descriptor_set_layout(&info, None) }?;

        Ok(Self {
            device,
            handle: raw,
        })
    }
}
impl AsVkHandle for DescriptorSetLayout {
    type Handle = vk::DescriptorSetLayout;

    fn vk_handle(&self) -> Self::Handle {
        self.handle
    }
}

/// A pool for allocating descriptor sets.
///
/// Descriptor pools manage the memory for descriptor sets. Sets allocated from
/// a pool remain valid until explicitly freed or the pool is destroyed.
pub struct DescriptorPool {
    device: Device,
    raw: vk::DescriptorPool,
}
impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.raw, None);
        }
    }
}
impl AsVkHandle for DescriptorPool {
    type Handle = vk::DescriptorPool;
    fn vk_handle(&self) -> Self::Handle {
        self.raw
    }
}
impl HasDevice for DescriptorPool {
    fn device(&self) -> &Device {
        &self.device
    }
}
impl DescriptorPool {
    /// Creates a new descriptor pool.
    ///
    /// # Parameters
    ///
    /// - `pool_sizes`: Specifies the number of descriptors of each type the pool can allocate.
    /// - `max_sets`: Maximum number of descriptor sets that can be allocated from this pool.
    /// - `flags`: Pool creation flags (e.g., `FREE_DESCRIPTOR_SET`, `UPDATE_AFTER_BIND`).
    pub fn new(
        device: Device,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
        flags: vk::DescriptorPoolCreateFlags,
    ) -> VkResult<Self> {
        unsafe {
            let raw = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    flags,
                    max_sets,

                    ..Default::default()
                }
                .pool_sizes(pool_sizes),
                None,
            )?;
            Ok(Self { device, raw })
        }
    }

    /// Allocates a single descriptor set from the pool.
    pub fn allocate_one(&mut self, layout: &DescriptorSetLayout) -> VkResult<vk::DescriptorSet> {
        unsafe {
            let mut descriptor = vk::DescriptorSet::null();
            (self.device.fp_v1_0().allocate_descriptor_sets)(
                self.device.handle(),
                &vk::DescriptorSetAllocateInfo {
                    descriptor_pool: self.raw,
                    descriptor_set_count: 1,
                    p_set_layouts: &layout.vk_handle(),
                    ..Default::default()
                },
                &mut descriptor,
            )
            .result()?;
            Ok(descriptor)
        }
    }

    /// Allocates a descriptor set with a variable descriptor count.
    ///
    /// Use this when the layout has a binding with `VARIABLE_DESCRIPTOR_COUNT` flag.
    /// The `size` parameter specifies the actual number of descriptors for that binding.
    pub fn allocate_one_variably_sized(
        &mut self,
        layout: &DescriptorSetLayout,
        size: u32,
    ) -> VkResult<vk::DescriptorSet> {
        unsafe {
            let mut descriptor = vk::DescriptorSet::null();
            let size = [size];
            let mut sizes = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(&size);
            (self.device.fp_v1_0().allocate_descriptor_sets)(
                self.device.handle(),
                &vk::DescriptorSetAllocateInfo {
                    descriptor_pool: self.raw,
                    descriptor_set_count: 1,
                    p_set_layouts: &layout.vk_handle(),
                    ..Default::default()
                }
                .push(&mut sizes),
                &mut descriptor,
            )
            .result()?;
            Ok(descriptor)
        }
    }

    /// Allocates multiple descriptor sets from the pool.
    pub fn allocate<'a>(
        &mut self,
        layouts: impl IntoIterator<Item = &'a DescriptorSetLayout>,
    ) -> VkResult<Vec<vk::DescriptorSet>> {
        unsafe {
            let handles: Vec<vk::DescriptorSetLayout> =
                layouts.into_iter().map(|x| x.vk_handle()).collect();
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo {
                    descriptor_pool: self.raw,
                    ..Default::default()
                }
                .set_layouts(&handles),
            )
        }
    }

    /// Frees descriptor sets back to the pool.
    ///
    /// The pool must have been created with [`vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET`] flag.
    pub fn free(&mut self, descriptors: &[vk::DescriptorSet]) -> VkResult<()> {
        unsafe { self.device.free_descriptor_sets(self.raw, descriptors) }
    }
}
