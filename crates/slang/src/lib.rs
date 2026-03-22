use std::collections::BTreeMap;

use pumicite_types::{
    Binding, DescriptorSetLayout, DescriptorSetLayoutRef, DescriptorType, PipelineLayout,
    ShaderStage,
};
use shader_slang as slang;

fn slang_binding_type_to_descriptor_type(
    binding_type: slang::BindingType,
) -> Option<DescriptorType> {
    match binding_type {
        slang::BindingType::Sampler => Some(DescriptorType::Sampler),
        slang::BindingType::Texture => Some(DescriptorType::SampledImage),
        slang::BindingType::ConstantBuffer => Some(DescriptorType::UniformBuffer),
        slang::BindingType::TypedBuffer => Some(DescriptorType::UniformTexelBuffer),
        slang::BindingType::RawBuffer => Some(DescriptorType::StorageBuffer),
        slang::BindingType::CombinedTextureSampler => Some(DescriptorType::CombinedImageSampler),
        slang::BindingType::InputRenderTarget => Some(DescriptorType::InputAttachment),
        slang::BindingType::InlineUniformData => Some(DescriptorType::InlineUniformBlock),
        slang::BindingType::RayTracingAccelerationStructure => {
            Some(DescriptorType::AccelerationStructure)
        }
        slang::BindingType::MutableTeture => Some(DescriptorType::StorageImage),
        slang::BindingType::MutableTypedBuffer => Some(DescriptorType::StorageTexelBuffer),
        slang::BindingType::MutableRawBuffer => Some(DescriptorType::StorageBuffer),
        // VaryingInput, VaryingOutput, ExistentialValue, PushConstant, ParameterBlock, etc.
        // are not descriptor types.
        _ => None,
    }
}

fn slang_stage_to_shader_stage(stage: slang::Stage) -> Option<ShaderStage> {
    match stage {
        slang::Stage::Vertex => Some(ShaderStage::Vertex),
        slang::Stage::Hull => Some(ShaderStage::TessellationControl),
        slang::Stage::Domain => Some(ShaderStage::TessellationEvaluation),
        slang::Stage::Geometry => Some(ShaderStage::Geometry),
        slang::Stage::Fragment => Some(ShaderStage::Fragment),
        slang::Stage::Compute => Some(ShaderStage::Compute),
        slang::Stage::RayGeneration => Some(ShaderStage::RayGen),
        slang::Stage::Intersection => Some(ShaderStage::Intersection),
        slang::Stage::AnyHit => Some(ShaderStage::AnyHit),
        slang::Stage::ClosestHit => Some(ShaderStage::ClosestHit),
        slang::Stage::Miss => Some(ShaderStage::Miss),
        slang::Stage::Callable => Some(ShaderStage::Callable),
        slang::Stage::Mesh => Some(ShaderStage::Mesh),
        slang::Stage::Amplification => Some(ShaderStage::Task),
        _ => None,
    }
}

fn collect_stages(reflection: &slang::reflection::Shader) -> Vec<ShaderStage> {
    reflection
        .entry_points()
        .filter_map(|ep| slang_stage_to_shader_stage(ep.stage()))
        .collect()
}

pub fn build_pipeline_layout(reflection: &slang::reflection::Shader) -> PipelineLayout {
    let stages = collect_stages(reflection);

    let global_type_layout = reflection
        .global_params_type_layout()
        .expect("no global params type layout");

    // Build descriptor sets from the global type layout's descriptor set structure.
    let descriptor_set_count = global_type_layout.descriptor_set_count();
    let mut sets: Vec<DescriptorSetLayout> = Vec::new();

    for set_idx in 0..descriptor_set_count {
        let range_count = global_type_layout.descriptor_set_descriptor_range_count(set_idx);
        let mut bindings = Vec::new();

        for range_idx in 0..range_count {
            let binding_type =
                global_type_layout.descriptor_set_descriptor_range_type(set_idx, range_idx);
            let descriptor_count = global_type_layout
                .descriptor_set_descriptor_range_descriptor_count(set_idx, range_idx);
            let index_offset =
                global_type_layout.descriptor_set_descriptor_range_index_offset(set_idx, range_idx);

            let Some(descriptor_type) = slang_binding_type_to_descriptor_type(binding_type) else {
                continue;
            };

            bindings.push(Binding {
                ty: descriptor_type,
                binding: index_offset as u32,
                count: descriptor_count.max(1) as u32,
                stages: stages.clone(),
                samplers: (),
                update_after_bind: false,
                update_unused_while_pending: false,
                partially_bound: false,
                variable_descriptor_count: false,
            });
        }

        if !bindings.is_empty() {
            sets.push(DescriptorSetLayout {
                push_descriptor: false,
                update_after_bind_pool: false,
                descriptor_buffer: false,
                bindings,
            });
        }
    }

    // Also collect descriptor sets from entry-point-specific parameters.
    for entry_point in reflection.entry_points() {
        let Some(type_layout) = entry_point.type_layout() else {
            continue;
        };
        let ep_stage =
            slang_stage_to_shader_stage(entry_point.stage()).unwrap_or(ShaderStage::Compute);
        let ep_set_count = type_layout.descriptor_set_count();

        for set_idx in 0..ep_set_count {
            let range_count = type_layout.descriptor_set_descriptor_range_count(set_idx);
            let mut bindings = Vec::new();

            for range_idx in 0..range_count {
                let binding_type =
                    type_layout.descriptor_set_descriptor_range_type(set_idx, range_idx);
                let descriptor_count = type_layout
                    .descriptor_set_descriptor_range_descriptor_count(set_idx, range_idx);
                let index_offset =
                    type_layout.descriptor_set_descriptor_range_index_offset(set_idx, range_idx);

                let Some(descriptor_type) = slang_binding_type_to_descriptor_type(binding_type)
                else {
                    continue;
                };

                bindings.push(Binding {
                    ty: descriptor_type,
                    binding: index_offset as u32,
                    count: descriptor_count.max(1) as u32,
                    stages: vec![ep_stage],
                    samplers: (),
                    update_after_bind: false,
                    update_unused_while_pending: false,
                    partially_bound: false,
                    variable_descriptor_count: false,
                });
            }

            if !bindings.is_empty() {
                sets.push(DescriptorSetLayout {
                    push_descriptor: false,
                    update_after_bind_pool: false,
                    descriptor_buffer: false,
                    bindings,
                });
            }
        }
    }

    // Build push constants from global uniform data and entry point uniform data.
    let mut push_constants = BTreeMap::new();

    let global_uniform_size = global_type_layout.size(slang::ParameterCategory::Uniform);
    if global_uniform_size > 0 {
        // Global push constants are accessible from all stages.
        for &stage in &stages {
            push_constants.insert(stage, (0, global_uniform_size as u32));
        }
    }

    let mut offset = global_uniform_size as u32;
    for entry_point in reflection.entry_points() {
        let Some(type_layout) = entry_point.type_layout() else {
            continue;
        };
        let ep_uniform_size = type_layout.size(slang::ParameterCategory::Uniform);
        if ep_uniform_size > 0 {
            if let Some(stage) = slang_stage_to_shader_stage(entry_point.stage()) {
                push_constants.insert(stage, (offset, ep_uniform_size as u32));
                offset += ep_uniform_size as u32;
            }
        }
    }

    let descriptor_set_refs = sets
        .into_iter()
        .map(DescriptorSetLayoutRef::Inline)
        .collect();

    PipelineLayout {
        sets: descriptor_set_refs,
        push_constants,
        independent_sets: false,
    }
}
