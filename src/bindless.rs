//! Bindless resource management.
//!
//! This module implements a bindless (or "descriptor indexing") architecture where
//! resources are stored in large descriptor arrays and accessed by index in shaders,
//! rather than binding individual descriptors per draw call.
//!
//! # Overview
//!
//! In a bindless architecture:
//! - All resources (textures, buffers, samplers) live in a global [`BindlessResourceHeap`]
//! - Shaders access resources via indices passed through push constants or buffers
//! - No descriptor set switching is needed between draw calls
//! - Resources can be added/removed dynamically without rebinding
//!
//! # Heap Structure
//!
//! The bindless heap uses three bindings:
//! - **Binding 0**: Samplers (`VK_DESCRIPTOR_TYPE_SAMPLER`)
//! - **Binding 1**: Combined image samplers (`VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`)
//! - **Binding 2**: Mutable descriptors for images, buffers, and texel buffers
//!
//! # Usage
//!
//! ```
//! # use std::sync::Arc;
//! # use pumicite::{Instance, Device, bindless::BindlessConfig};
//! # let entry = Arc::new(unsafe { ash::Entry::load() }.unwrap());
//! # let instance = Instance::builder(entry).build().unwrap();
//! # let pdevice = instance.enumerate_physical_devices().unwrap().next().unwrap();
//! // Enable bindless on device creation
//! let mut builder = Device::builder(pdevice);
//! builder.enable_bindless(BindlessConfig::default()).unwrap();
//! builder.enable_queue(0, 1.0);
//! let device = builder.build().unwrap();
//!
//! // Get the bindless heap
//! let heap = device.get_bindless_heap().unwrap();
//!
//! // Use heap.add_resource() to add textures/buffers
//! // Pass handles to shaders via push constants
//! ```
//!
//! # Requirements
//!
//! - `VK_EXT_mutable_descriptor_type` or `VK_VALVE_mutable_descriptor_type`
//! - `VK_EXT_descriptor_indexing` (Vulkan 1.2 core)

use std::sync::{Arc, Mutex};

use ash::{
    VkResult,
    vk::{self, TaggedStructure},
};

use crate::{
    HasDevice, MissingFeatureError,
    descriptor::{DescriptorPool, DescriptorSetLayout},
    device::DeviceBuilder,
    prelude::*,
    utils::IdAlloc,
};

/// A bindless resource heap.
///
/// This is a large descriptor pool containing all bindable resources. Resources are
/// added dynamically and accessed in shaders by their index ([`BindlessResourceHandle::id`]).
///
/// # Thread Safety
///
/// The heap uses internal mutexes for ID allocation, so resources can be added from
/// multiple threads concurrently.
///
/// # Descriptor Layout
///
/// - Binding 0: Samplers (separate from images)
/// - Binding 1: Combined image samplers (unused for forward compatibility)
/// - Binding 2: Mutable descriptors (sampled images, storage images, buffers, etc.)
struct ResourceHeapInner {
    pool: DescriptorPool,
    layout: Arc<DescriptorSetLayout>,
    set: vk::DescriptorSet,

    id_alloc: Mutex<IdAlloc>,
}

#[derive(Clone)]
pub struct ResourceHeap(Arc<ResourceHeapInner>);

impl ResourceHeap {
    pub fn new(device: Device, capacity: u32) -> VkResult<Self> {
        let layout = DescriptorSetLayout::new(
            device.clone(),
            &[vk::DescriptorSetLayoutBinding {
                descriptor_type: vk::DescriptorType::MUTABLE_EXT,
                descriptor_count: capacity,
                stage_flags: vk::ShaderStageFlags::ALL,
                ..Default::default()
            }],
            &[vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING],
            &[&[
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
            ]],
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )?;
        let mut pool = DescriptorPool::new(
            device,
            &[vk::DescriptorPoolSize {
                ty: vk::DescriptorType::MUTABLE_EXT,
                descriptor_count: capacity,
            }],
            1,
            vk::DescriptorPoolCreateFlags::empty(),
        )?;

        let set = pool.allocate_one_variably_sized(&layout, capacity)?;

        Ok(Self(Arc::new(ResourceHeapInner {
            pool,
            layout: Arc::new(layout),
            id_alloc: Mutex::new(IdAlloc::default()),
            set,
        })))
    }

    pub fn descriptor_layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.0.layout
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.0.set
    }

    pub fn add_image_view_with_layout(
        &self,
        image_layout: vk::ImageLayout,
        view: &vk::ImageViewCreateInfo,
        access: ImageAccessMode,
    ) -> VkResult<u32> {
        unsafe {
            let image_view = self.0.pool.device().create_image_view(view, None)?;
            let mut guard = self.0.id_alloc.lock().unwrap();
            let handle = guard.alloc_one();
            self.0.pool.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    dst_set: self.0.set,
                    descriptor_count: 1,
                    dst_array_element: handle,
                    descriptor_type: match access {
                        ImageAccessMode::Sampled => vk::DescriptorType::SAMPLED_IMAGE,
                        ImageAccessMode::Storage => vk::DescriptorType::STORAGE_IMAGE,
                    },
                    p_image_info: &vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view,
                        image_layout,
                    },
                    ..Default::default()
                }],
                &[],
            );
            self.0.pool.device().destroy_image_view(image_view, None); // Should be ok???
            Ok(handle)
        }
    }
    pub fn add_image_view(
        &self,
        view: &vk::ImageViewCreateInfo,
        access: ImageAccessMode,
    ) -> VkResult<u32> {
        self.add_image_view_with_layout(vk::ImageLayout::GENERAL, view, access)
    }

    pub fn add_image_with_layout(
        &self,
        image: &impl ImageLike,
        image_layout: vk::ImageLayout,
        access: ImageAccessMode,
    ) -> VkResult<u32> {
        let view_type = match image.ty() {
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => unreachable!(),
        };
        self.add_image_view_with_layout(
            image_layout,
            &vk::ImageViewCreateInfo {
                image: image.vk_handle(),
                view_type,
                format: image.format(),
                components: vk::ComponentMapping::default(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: image.aspects(),
                    base_mip_level: 0,
                    base_array_layer: 0,
                    level_count: image.mip_level_count(),
                    layer_count: image.array_layer_count(),
                },
                ..Default::default()
            }
            .push(&mut vk::ImageViewUsageCreateInfo {
                usage: match access {
                    ImageAccessMode::Sampled => vk::ImageUsageFlags::SAMPLED,
                    ImageAccessMode::Storage => vk::ImageUsageFlags::STORAGE,
                },
                ..Default::default()
            }),
            access,
        )
    }

    pub fn add_image(&self, image: &impl ImageLike, access: ImageAccessMode) -> VkResult<u32> {
        self.add_image_with_layout(image, vk::ImageLayout::GENERAL, access)
    }

    pub fn add_texel_buffer(
        &self,
        buffer: impl BufferLike,
        format: vk::Format,
        access: BufferAccessMode,
    ) -> VkResult<u32> {
        unsafe {
            let buffer_view = self.0.pool.device().create_buffer_view(
                &vk::BufferViewCreateInfo {
                    buffer: buffer.vk_handle(),
                    format,
                    offset: buffer.offset(),
                    range: buffer.size(),
                    ..Default::default()
                },
                None,
            )?;
            let mut guard = self.0.id_alloc.lock().unwrap();
            let handle = guard.alloc_one();
            self.0.pool.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    dst_set: self.0.set,
                    descriptor_count: 1,
                    dst_array_element: handle,
                    descriptor_type: match access {
                        BufferAccessMode::Uniform => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                        BufferAccessMode::Storage => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                    },
                    p_texel_buffer_view: &buffer_view,
                    ..Default::default()
                }],
                &[],
            );
            self.0.pool.device().destroy_buffer_view(buffer_view, None); // Should be ok???
            Ok(handle)
        }
    }

    pub fn add_buffer(&self, buffer: impl BufferLike, access: BufferAccessMode) -> VkResult<u32> {
        unsafe {
            let mut guard = self.0.id_alloc.lock().unwrap();
            let handle = guard.alloc_one();
            self.0.pool.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    dst_set: self.0.set,
                    descriptor_count: 1,
                    dst_array_element: handle,
                    descriptor_type: match access {
                        BufferAccessMode::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                        BufferAccessMode::Storage => vk::DescriptorType::STORAGE_BUFFER,
                    },
                    p_buffer_info: &vk::DescriptorBufferInfo {
                        buffer: buffer.vk_handle(),
                        offset: buffer.offset(),
                        range: buffer.size(),
                    },
                    ..Default::default()
                }],
                &[],
            );
            Ok(handle)
        }
    }

    pub fn remove(&self, handle: u32) {
        let mut guard = self.0.id_alloc.lock().unwrap();
        guard.free(handle, 1);
    }
}

pub enum ImageAccessMode {
    Sampled,
    Storage,
}
pub enum BufferAccessMode {
    Uniform,
    Storage,
}

/// A bindless sampler heap.
///
/// # Thread Safety
///
/// The heap uses internal mutexes for ID allocation, so samplers can be added from
/// multiple threads concurrently.
struct SamplerHeapInner {
    pool: DescriptorPool,
    layout: Arc<DescriptorSetLayout>,
    set: vk::DescriptorSet,

    id_alloc: Mutex<IdAlloc>,
}

#[derive(Clone)]
pub struct SamplerHeap(Arc<SamplerHeapInner>);

/// A sampler handle that automatically removes itself from the heap when dropped.
pub struct SamplerHandle {
    heap: SamplerHeap,
    handle: u32,
}
impl SamplerHandle {
    pub fn new(heap: SamplerHeap, create_info: &vk::SamplerCreateInfo) -> VkResult<Self> {
        let handle = heap.add(create_info)?;
        Ok(Self { heap, handle })
    }
    pub fn id(&self) -> u32 {
        self.handle
    }
}
impl Drop for SamplerHandle {
    fn drop(&mut self) {
        self.heap.remove(self.handle);
    }
}

impl SamplerHeap {
    pub fn new(device: Device, capacity: u32) -> VkResult<Self> {
        let layout = DescriptorSetLayout::new(
            device.clone(),
            &[vk::DescriptorSetLayoutBinding {
                descriptor_type: vk::DescriptorType::SAMPLER,
                descriptor_count: capacity,
                stage_flags: vk::ShaderStageFlags::ALL,
                ..Default::default()
            }],
            &[vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING],
            &[],
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )?;
        let mut pool = DescriptorPool::new(
            device,
            &[vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: capacity,
            }],
            1,
            vk::DescriptorPoolCreateFlags::empty(),
        )?;

        let set = pool.allocate_one_variably_sized(&layout, capacity)?;

        Ok(Self(Arc::new(SamplerHeapInner {
            pool,
            layout: Arc::new(layout),
            id_alloc: Mutex::new(IdAlloc::default()),
            set,
        })))
    }
    pub fn descriptor_layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.0.layout
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.0.set
    }

    pub fn add(&self, create_info: &vk::SamplerCreateInfo) -> VkResult<u32> {
        unsafe {
            let sampler = self.0.pool.device().create_sampler(create_info, None)?;
            let mut guard = self.0.id_alloc.lock().unwrap();
            let handle = guard.alloc_one();
            self.0.pool.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    dst_set: self.0.set,
                    descriptor_count: 1,
                    dst_array_element: handle,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    p_image_info: &vk::DescriptorImageInfo {
                        sampler,
                        image_view: vk::ImageView::null(),
                        image_layout: vk::ImageLayout::UNDEFINED,
                    },
                    ..Default::default()
                }],
                &[],
            );
            self.0.pool.device().destroy_sampler(sampler, None);
            Ok(handle)
        }
    }

    pub fn remove(&self, handle: u32) {
        let mut guard = self.0.id_alloc.lock().unwrap();
        guard.free(handle, 1);
    }
}

impl DeviceBuilder {
    /// Enables the bindless resource system.
    ///
    /// This enables all required extensions and features for bindless rendering:
    /// - `VK_EXT_descriptor_indexing` with required features
    /// - `VK_EXT_mutable_descriptor_type`
    /// - `VK_NV_descriptor_pool_overallocation` (optional, for larger heaps)
    ///
    /// If called multiple times, configurations are merged (maximums are taken).
    ///
    /// # Errors
    ///
    /// Returns an error if required extensions or features are not available.
    pub fn enable_bindless(&mut self) -> Result<(), MissingFeatureError> {
        self.enable_extension::<ash::ext::descriptor_indexing::Meta>()?;
        self.enable_feature::<vk::PhysicalDeviceDescriptorIndexingFeatures>(|x| {
            &mut x.descriptor_binding_update_unused_while_pending
        })?;
        self.enable_feature::<vk::PhysicalDeviceDescriptorIndexingFeatures>(|x| {
            &mut x.descriptor_binding_variable_descriptor_count
        })?;
        self.enable_feature::<vk::PhysicalDeviceDescriptorIndexingFeatures>(|x| {
            &mut x.descriptor_binding_partially_bound
        })?;

        self.enable_extension::<ash::ext::mutable_descriptor_type::Meta>()?;
        self.enable_feature::<vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT>(|x| {
            &mut x.mutable_descriptor_type
        })?;
        self.enable_feature::<vk::PhysicalDeviceDescriptorIndexingFeatures>(|x| {
            &mut x.runtime_descriptor_array
        })?;

        self.enable_extension::<ash::nv::descriptor_pool_overallocation::Meta>()
            .ok();
        self.enable_feature::<vk::PhysicalDeviceDescriptorPoolOverallocationFeaturesNV>(|x| {
            &mut x.descriptor_pool_overallocation
        })
        .ok();

        self.bindless_enabled = true;

        Ok(())
    }

    pub fn bindless_enabled(&self) -> bool {
        self.bindless_enabled
    }
}
