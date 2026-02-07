use bevy_ecs::resource::Resource;
use pumicite::{
    Device, HasDevice,
    ash::{VkResult, vk},
    bindless::{ResourceHeap, SamplerHeap},
    command::CommandEncoder,
    pipeline::PipelineLayout,
    utils::AsVkHandle,
};
use std::sync::Arc;

pub struct DescriptorHeapInner {
    resource_heap: ResourceHeap,
    sampler_heap: SamplerHeap,
    bindless_pipeline_layout: Arc<PipelineLayout>,
}

#[derive(Resource, Clone)]
pub struct DescriptorHeap(Arc<DescriptorHeapInner>);

impl DescriptorHeap {
    pub fn bindless_pipeline_layout(&self) -> &Arc<PipelineLayout> {
        &self.0.bindless_pipeline_layout
    }
    pub fn sampler_heap(&self) -> &SamplerHeap {
        &self.0.sampler_heap
    }
    pub fn resource_heap(&self) -> &ResourceHeap {
        &self.0.resource_heap
    }

    /// Binds the descriptor heap to the command encoder.
    ///
    /// This binds the resource heap to set 0 and the sampler heap to set 1.
    pub fn bind(&self, encoder: &mut CommandEncoder, bind_point: vk::PipelineBindPoint) {
        let descriptor_sets = [
            self.0.resource_heap.descriptor_set(),
            self.0.sampler_heap.descriptor_set(),
        ];
        unsafe {
            encoder.device().cmd_bind_descriptor_sets(
                encoder.buffer().vk_handle(),
                bind_point,
                self.0.bindless_pipeline_layout.vk_handle(),
                0,
                &descriptor_sets,
                &[],
            );
        }
    }

    pub fn new(
        device: Device,
        resource_heap_capacity: u32,
        sampler_heap_capacity: u32,
    ) -> VkResult<Self> {
        let resource_heap = ResourceHeap::new(device.clone(), resource_heap_capacity)?;
        let sampler_heap = SamplerHeap::new(device.clone(), sampler_heap_capacity)?;
        let bindless_pipeline_layout = pumicite::pipeline::PipelineLayout::new(
            device,
            vec![
                resource_heap.descriptor_layout().clone(),
                sampler_heap.descriptor_layout().clone(),
            ],
            &[vk::PushConstantRange {
                offset: 0,
                size: 128, // Minimum guaranteed by vulkan specs
                stage_flags: vk::ShaderStageFlags::ALL,
            }],
            vk::PipelineLayoutCreateFlags::empty(),
        )?;
        Ok(Self(Arc::new(DescriptorHeapInner {
            resource_heap,
            sampler_heap,
            bindless_pipeline_layout: Arc::new(bindless_pipeline_layout),
        })))
    }
}
