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

pub fn build_descriptor_set_layout(
    type_layout: &slang::reflection::TypeLayout,
    set_idx: i64,
    stages: &[ShaderStage],
) -> Option<DescriptorSetLayout> {
    let range_count = type_layout.descriptor_set_descriptor_range_count(set_idx);
    let mut bindings = Vec::new();

    for range_idx in 0..range_count {
        let binding_type = type_layout.descriptor_set_descriptor_range_type(set_idx, range_idx);
        let descriptor_count =
            type_layout.descriptor_set_descriptor_range_descriptor_count(set_idx, range_idx);
        let index_offset =
            type_layout.descriptor_set_descriptor_range_index_offset(set_idx, range_idx);

        let Some(descriptor_type) = slang_binding_type_to_descriptor_type(binding_type) else {
            continue;
        };

        bindings.push(Binding {
            ty: descriptor_type,
            binding: index_offset as u32,
            count: descriptor_count.max(1) as u32,
            stages: stages.to_vec(),
            samplers: (),
            update_after_bind: false,
            update_unused_while_pending: false,
            partially_bound: false,
            variable_descriptor_count: false,
        });
    }

    (!bindings.is_empty()).then(|| DescriptorSetLayout {
        push_descriptor: false,
        update_after_bind_pool: false,
        descriptor_buffer: false,
        bindings,
    })
}

pub fn build_pipeline_layout(reflection: &slang::reflection::Shader) -> PipelineLayout {
    let stages = collect_stages(reflection);

    let global_type_layout = reflection
        .global_params_type_layout()
        .expect("no global params type layout");

    // Descriptor sets from global parameters.
    let mut sets: Vec<DescriptorSetLayout> = Vec::new();
    for set_idx in 0..global_type_layout.descriptor_set_count() {
        if let Some(set) = build_descriptor_set_layout(global_type_layout, set_idx, &stages) {
            sets.push(set);
        }
    }

    // Descriptor sets from entry-point-specific parameters.
    for entry_point in reflection.entry_points() {
        let Some(type_layout) = entry_point.type_layout() else {
            continue;
        };
        let ep_stage =
            slang_stage_to_shader_stage(entry_point.stage()).unwrap_or(ShaderStage::Compute);
        let ep_stages = &[ep_stage];

        for set_idx in 0..type_layout.descriptor_set_count() {
            if let Some(set) = build_descriptor_set_layout(type_layout, set_idx, ep_stages) {
                sets.push(set);
            }
        }
    }

    // Push constants from global uniform data and entry point uniform data.
    let mut push_constants = BTreeMap::new();

    let global_uniform_size = global_type_layout.size(slang::ParameterCategory::Uniform);
    if global_uniform_size > 0 {
        for &stage in &stages {
            push_constants.insert(stage, (0, global_uniform_size as u32));
        }
    }

    let mut offset = global_uniform_size as u32;
    for entry_point in reflection.entry_points() {
        // The entry point's type_layout() is a ConstantBuffer wrapper whose
        // size(Uniform) is 0. To find the actual push constant size we iterate
        // the entry point's parameters and sum those in the Uniform category.
        let ep_uniform_size: usize = entry_point
            .parameters()
            .filter(|p| {
                p.type_layout()
                    .is_some_and(|tl| tl.parameter_category() == slang::ParameterCategory::Uniform)
            })
            .map(|p| {
                p.type_layout()
                    .unwrap()
                    .size(slang::ParameterCategory::Uniform)
            })
            .sum();

        if ep_uniform_size > 0 {
            if let Some(stage) = slang_stage_to_shader_stage(entry_point.stage()) {
                push_constants.insert(stage, (offset, ep_uniform_size as u32));
                offset += ep_uniform_size as u32;
            }
        }
    }

    PipelineLayout {
        sets: sets
            .into_iter()
            .map(DescriptorSetLayoutRef::Inline)
            .collect(),
        push_constants,
        independent_sets: false,
    }
}
