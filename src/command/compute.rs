//! Compute shader and pipeline binding commands.
//!
//! This module extends [`CommandEncoder`] with methods for binding pipelines,
//! descriptor sets, push constants, and dispatching compute work.

use ash::vk;
use glam::UVec3;

use crate::{
    HasDevice,
    pipeline::{Pipeline, PipelineLayout},
    utils::AsVkHandle,
};

use super::CommandEncoder;

impl<'a> CommandEncoder<'a> {
    /// Binds a pipeline to the command buffer.
    pub fn bind_pipeline(&mut self, bind_point: vk::PipelineBindPoint, pipeline: &'a Pipeline) {
        unsafe {
            self.device()
                .cmd_bind_pipeline(self.buffer().buffer, bind_point, pipeline.vk_handle());
        }
    }
    /// Binds descriptor sets to the command buffer.
    pub fn bind_descriptor_sets(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.device().cmd_bind_descriptor_sets(
                self.buffer().buffer,
                bind_point,
                layout.vk_handle(),
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }

    /// Pushes a descriptor set directly to the command buffer with driver-managed allocation.
    ///
    /// This is an alternative to `bind_descriptor_sets` that allows updating descriptors
    /// inline without allocating from a descriptor pool. Requires the
    /// `VK_KHR_push_descriptor` extension.
    pub fn push_descriptor_set(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: &PipelineLayout,
        set: u32,
        descriptor_writes: &[vk::WriteDescriptorSet<'_>],
    ) {
        debug_assert!(layout.device() == self.device());
        unsafe {
            self.device()
                .extension::<ash::khr::push_descriptor::Meta>()
                .cmd_push_descriptor_set(
                    self.buffer().buffer,
                    bind_point,
                    layout.vk_handle(),
                    set,
                    descriptor_writes,
                );
        }
    }
    /// Updates push constant data for the pipeline.
    ///
    /// Push constants are a small block of values accessible directly from shaders,
    /// providing a fast path for frequently-changing uniform data without descriptor
    /// set updates.
    pub fn push_constants(
        &mut self,
        layout: &PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        debug_assert!(layout.device() == self.device());
        unsafe {
            self.device().cmd_push_constants(
                self.buffer().buffer,
                layout.vk_handle(),
                stages,
                offset,
                constants,
            );
        }
    }

    /// Dispatches compute work.
    ///
    /// Executes the currently bound compute pipeline with the specified number of
    /// workgroups in each dimension.
    ///
    /// # Parameters
    /// - `size`: The number of workgroups to dispatch in (x, y, z) dimensions
    pub fn dispatch(&mut self, size: UVec3) {
        unsafe {
            self.device()
                .cmd_dispatch(self.buffer().buffer, size.x, size.y, size.z);
        }
    }

    /// Builds or updates an acceleration structure for ray tracing.
    ///
    /// Requires the `VK_KHR_acceleration_structure` extension.
    pub fn build_accel_struct(
        &mut self,
        infos: &vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
        build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ) {
        unsafe {
            (self
                .device()
                .extension::<ash::khr::acceleration_structure::Meta>()
                .fp()
                .cmd_build_acceleration_structures_khr)(
                self.buffer().buffer,
                1,
                infos,
                &build_range_infos.as_ptr(),
            );
        }
    }
}
