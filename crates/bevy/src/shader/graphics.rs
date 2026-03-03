use std::{ffi::CString, ops::Deref, sync::Arc};

use bevy_asset::{Asset, AssetLoader, LoadedAsset};
use bevy_ecs::world::FromWorld;
use bevy_reflect::TypePath;
use pumicite::{
    Device, HasDevice,
    ash::vk::{self, TaggedStructure},
    pipeline::{Pipeline, PipelineCache},
    utils::AsVkHandle,
};

use crate::{
    DescriptorHeap,
    shader::{PipelineLayoutLoader, PipelineLoaderError, ShaderModule, ron_types},
};

/// A loaded graphics pipeline asset.
///
/// Load via asset server with `.gfx.pipeline.ron` extension.
///
/// Graphics pipelines can have runtime variants via [`GraphicsPipelineVariant`](ron_types::GraphicsPipelineVariant)
/// for specialization constants and format overrides.
#[derive(Clone, Asset, TypePath)]
pub struct GraphicsPipeline(Arc<Pipeline>);
impl GraphicsPipeline {
    /// Unwraps the inner pipeline.
    pub fn into_inner(self) -> Arc<Pipeline> {
        self.0
    }
}
impl Deref for GraphicsPipeline {
    type Target = Arc<Pipeline>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Asset loader for graphics pipelines (`.gfx.pipeline.ron` files).
///
/// Loads graphics pipeline configurations with full support for:
/// - Vertex input, rasterization, depth/stencil state
/// - Dynamic state
/// - Specialization constants
/// - Pipeline variants
pub struct GraphicsPipelineLoader {
    pipeline_cache: Arc<PipelineCache>,
    heap: Option<DescriptorHeap>,
}
impl FromWorld for GraphicsPipelineLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            pipeline_cache: Arc::new(PipelineCache::null(world.resource::<Device>().clone())),
            heap: world.get_resource::<DescriptorHeap>().cloned(),
        }
    }
}
impl AssetLoader for GraphicsPipelineLoader {
    type Asset = GraphicsPipeline;
    type Settings = ron_types::GraphicsPipelineVariant;
    type Error = PipelineLoaderError;

    async fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<GraphicsPipeline, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let mut pipeline: ron_types::GraphicsPipeline = ron::de::from_bytes(&bytes)?;
        settings.apply_on(&mut pipeline);

        let layout = match &pipeline.layout {
            ron_types::PipelineLayoutRef::Inline(pipeline_layout) => {
                PipelineLayoutLoader::load_inner(
                    pipeline_layout,
                    self.pipeline_cache.device().clone(),
                    self.heap.as_ref(),
                    load_context,
                )
                .await?
                .0
            }
            ron_types::PipelineLayoutRef::Path(path) => {
                load_context
                    .loader()
                    .immediate()
                    .load::<pumicite::bevy::PipelineLayout>(path)
                    .await?
                    .take()
                    .0
            }
            ron_types::PipelineLayoutRef::Bindless => {
                let Some(heap) = self.heap.as_ref() else {
                    return Err(PipelineLoaderError::BindlessPluginNeededError);
                };
                heap.bindless_pipeline_layout().clone()
            }
        };

        let mut dynamic_states = Vec::<vk::DynamicState>::new();
        let mut shader_modules = Vec::with_capacity(pipeline.shaders.len());
        for (_, shader) in pipeline.shaders.iter() {
            let module: LoadedAsset<ShaderModule> =
                load_context.loader().immediate().load(&shader.path).await?;
            shader_modules.push(module.take());
        }
        let shader_entry_names = pipeline
            .shaders
            .iter()
            .map(|x| CString::new(&*x.1.entry_point))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| PipelineLoaderError::PipelineError("Invalid entry name"))?;

        // Build specialization info data structures
        let mut specialization_map_entries: Vec<Vec<vk::SpecializationMapEntry>> = Vec::new();
        let mut specialization_data: Vec<Vec<u8>> = Vec::new();

        for (_, shader) in pipeline.shaders.iter() {
            if shader.specialization_constants.is_empty() {
                specialization_map_entries.push(Vec::new());
                specialization_data.push(Vec::new());
            } else {
                let mut entries = Vec::new();
                let mut data = Vec::new();

                for (&constant_id, value) in shader.specialization_constants.iter() {
                    let offset = data.len();

                    let size = value.extend(&mut data);

                    entries.push(vk::SpecializationMapEntry {
                        constant_id,
                        offset: offset as u32,
                        size,
                    });
                }

                specialization_map_entries.push(entries);
                specialization_data.push(data);
            }
        }

        // Build the SpecializationInfo structs with stable pointers
        let specialization_infos: Vec<vk::SpecializationInfo> = specialization_map_entries
            .iter()
            .zip(specialization_data.iter())
            .map(|(entries, data)| vk::SpecializationInfo {
                map_entry_count: entries.len() as u32,
                p_map_entries: entries.as_ptr(),
                data_size: data.len(),
                p_data: data.as_ptr() as *const std::ffi::c_void,
                ..Default::default()
            })
            .collect();

        let stages = pipeline
            .shaders
            .iter()
            .zip(shader_modules.iter())
            .zip(shader_entry_names.iter())
            .enumerate()
            .map(|(i, (((&stage, shader), module), entry_name))| {
                vk::PipelineShaderStageCreateInfo {
                    flags: shader.flags(),
                    stage: stage.into(),
                    module: module.0.vk_handle(),
                    p_specialization_info: if specialization_map_entries[i].is_empty() {
                        std::ptr::null()
                    } else {
                        &specialization_infos[i]
                    },
                    p_name: entry_name.as_ptr(),
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();

        let mut info = vk::GraphicsPipelineCreateInfo {
            layout: layout.vk_handle(),
            ..Default::default()
        }
        .stages(&stages);

        let mut vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();
        let vertex_attribute_descriptions;
        let vertex_binding_descriptions;
        if let Some(bindings) = pipeline.vertex_bindings.unwrap(&mut dynamic_states) {
            vertex_attribute_descriptions = bindings
                .iter()
                .flat_map(|(&binding, desc)| {
                    desc.attributes.iter().map(move |(&location, attribute)| {
                        Some(vk::VertexInputAttributeDescription {
                            location,
                            binding,
                            format: attribute.format.into(),
                            offset: attribute.offset,
                        })
                    })
                })
                .collect::<Option<Vec<_>>>()
                .ok_or(PipelineLoaderError::PipelineError("Invalid format"))?;
            vertex_binding_descriptions = bindings
                .iter()
                .map(|(&binding, desc)| vk::VertexInputBindingDescription {
                    binding,
                    stride: desc
                        .stride
                        .unwrap(&mut dynamic_states)
                        .cloned()
                        .unwrap_or_default(),
                    input_rate: desc.input_rate.into(),
                })
                .collect::<Vec<_>>();
            vertex_input_state = vertex_input_state
                .vertex_attribute_descriptions(&vertex_attribute_descriptions)
                .vertex_binding_descriptions(&vertex_binding_descriptions);
        }
        info = info.vertex_input_state(&vertex_input_state);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: pipeline
                .topology
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            primitive_restart_enable: pipeline
                .primitive_restart_enabled
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            ..Default::default()
        };
        info = info.input_assembly_state(&input_assembly_state);

        let tessellation_state = vk::PipelineTessellationStateCreateInfo {
            patch_control_points: pipeline
                .tessellation_patch_control_points
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default(),
            ..Default::default()
        };
        info = info.tessellation_state(&tessellation_state);

        let mut viewport_state = vk::PipelineViewportStateCreateInfo::default();
        let viewports: Vec<vk::Viewport>;
        let scissors: Vec<vk::Rect2D>;
        match pipeline.viewports {
            ron_types::CountedDynamicState::Dynamic => {
                dynamic_states.push(vk::DynamicState::VIEWPORT_WITH_COUNT);
            }
            ron_types::CountedDynamicState::Count(count) => {
                dynamic_states.push(vk::DynamicState::VIEWPORT);
                viewport_state.viewport_count = count;
            }
            ron_types::CountedDynamicState::Static(items) => {
                viewports = items.iter().map(|view| view.clone().into()).collect();
                viewport_state = viewport_state.viewports(&viewports);
            }
        }
        match pipeline.scissors {
            ron_types::CountedDynamicState::Dynamic => {
                dynamic_states.push(vk::DynamicState::SCISSOR_WITH_COUNT);
            }
            ron_types::CountedDynamicState::Count(count) => {
                dynamic_states.push(vk::DynamicState::SCISSOR);
                viewport_state.scissor_count = count;
            }
            ron_types::CountedDynamicState::Static(items) => {
                scissors = items.iter().map(|view| view.clone().into()).collect();
                viewport_state = viewport_state.scissors(&scissors);
            }
        }
        if viewport_state.scissor_count > 0 || viewport_state.viewport_count > 0 {
            info = info.viewport_state(&viewport_state);
        }

        let mut rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: pipeline
                .depth_clamp_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            rasterizer_discard_enable: pipeline
                .rasterizer_discard_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            polygon_mode: pipeline
                .polygon_mode
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            cull_mode: pipeline
                .cull_mode
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            front_face: pipeline
                .front_face
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            depth_bias_enable: pipeline
                .depth_bias_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            line_width: pipeline
                .line_width
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default(),
            ..Default::default()
        };

        if let Some(depth_bias) = pipeline.depth_bias.unwrap(&mut dynamic_states) {
            rasterization_state.depth_bias_constant_factor = depth_bias.constant_factor;
            rasterization_state.depth_bias_clamp = depth_bias.clamp;
            rasterization_state.depth_bias_slope_factor = depth_bias.slope_factor;
        }

        info = info.rasterization_state(&rasterization_state);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: match pipeline.sample_count.unwrap(&mut dynamic_states) {
                None => vk::SampleCountFlags::empty(),
                Some(n) => match n {
                    1 | 2 | 4 | 8 | 16 | 32 | 64 => vk::SampleCountFlags::from_raw(*n as u32),
                    _ => {
                        return Err(PipelineLoaderError::PipelineError(
                            "Unrecognized: sample_count",
                        ));
                    }
                },
            },
            sample_shading_enable: pipeline.sample_shading.is_some().into(),
            min_sample_shading: pipeline.sample_shading.unwrap_or_default(),
            alpha_to_coverage_enable: pipeline
                .alpha_to_coverage_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            alpha_to_one_enable: pipeline
                .alpha_to_one_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            ..Default::default()
        }
        .sample_mask(
            pipeline
                .sample_mask
                .unwrap(&mut dynamic_states)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        );
        info = info.multisample_state(&multisample_state);

        let depth_test_enable: bool = pipeline
            .depth_test_enable
            .unwrap(&mut dynamic_states)
            .cloned()
            .unwrap_or_default();
        let mut ds_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: depth_test_enable.into(),
            depth_write_enable: pipeline
                .depth_write_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            depth_compare_op: if depth_test_enable {
                compare_op_str(
                    pipeline
                        .depth_compare_op
                        .unwrap(&mut dynamic_states)
                        .map(String::as_str),
                    "Invalid depth compare op",
                )?
            } else {
                Default::default()
            },
            depth_bounds_test_enable: pipeline
                .depth_bounds_test_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            stencil_test_enable: pipeline
                .stencil_test_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            ..Default::default()
        };
        if let Some((min, max)) = pipeline.depth_bounds.unwrap(&mut dynamic_states) {
            ds_state.min_depth_bounds = *min;
            ds_state.max_depth_bounds = *max;
        }
        if let Some(front) = &pipeline.front_stencil
            && let Some(ops) = front.ops.unwrap(&mut dynamic_states)
        {
            ds_state.front.fail_op = ops.fail.into();
            ds_state.front.pass_op = ops.pass.into();
            ds_state.front.depth_fail_op = ops.depth_fail.into();
            ds_state.front.compare_op = compare_op_str(
                Some(ops.compare.as_str()),
                "Invalid comapre op for front stencil",
            )?;
        }
        if let Some(back) = &pipeline.back_stencil
            && let Some(ops) = back.ops.unwrap(&mut dynamic_states)
        {
            ds_state.back.fail_op = ops.fail.into();
            ds_state.back.pass_op = ops.pass.into();
            ds_state.back.depth_fail_op = ops.depth_fail.into();
            ds_state.back.compare_op = compare_op_str(
                Some(ops.compare.as_str()),
                "Invalid compare op for back stencil",
            )?;
        }

        info = info.depth_stencil_state(&ds_state);

        let color_blend_state_attachments = {
            let mut blend_enable_dynamic_count = 0_u32;
            let mut blend_equation_dynamic_count = 0_u32;
            let mut color_write_mask_dynamic_count = 0_u32;
            let attachments = pipeline
                .attachments
                .iter()
                .map(|attachment| {
                    let mut state = vk::PipelineColorBlendAttachmentState::default();
                    match attachment.blend_enable {
                        ron_types::RequiredDynamicState::Dynamic => {
                            blend_enable_dynamic_count += 1;
                        }
                        ron_types::RequiredDynamicState::Static(bool) => {
                            state.blend_enable = bool.into();
                        }
                    }
                    match &attachment.blend_equation {
                        ron_types::OptionalDynamicState::Dynamic => {
                            blend_equation_dynamic_count += 1
                        }
                        ron_types::OptionalDynamicState::Static(equation) => {
                            state.color_blend_op = equation.color.1.into();
                            state.src_color_blend_factor = equation.color.0.into();
                            state.dst_color_blend_factor = equation.color.2.into();

                            state.alpha_blend_op = equation.alpha.1.into();
                            state.src_alpha_blend_factor = equation.alpha.0.into();
                            state.dst_alpha_blend_factor = equation.alpha.2.into();
                        }
                        ron_types::OptionalDynamicState::None => {
                            if matches!(
                                attachment.blend_enable,
                                ron_types::RequiredDynamicState::Static(true)
                            ) {
                                return Err(PipelineLoaderError::PipelineError(
                                    "Blending enabled; blend equation required",
                                ));
                            }
                        }
                    }
                    match &attachment.color_write_mask {
                        ron_types::OptionalDynamicState::None => {
                            state.color_write_mask = vk::ColorComponentFlags::RGBA;
                        }
                        ron_types::OptionalDynamicState::Dynamic => {
                            color_write_mask_dynamic_count += 1
                        }
                        ron_types::OptionalDynamicState::Static(mask) => {
                            let mut flags = vk::ColorComponentFlags::empty();
                            for char in mask.chars() {
                                match char {
                                    'r' => flags |= vk::ColorComponentFlags::R,
                                    'g' => flags |= vk::ColorComponentFlags::G,
                                    'b' => flags |= vk::ColorComponentFlags::B,
                                    'a' => flags |= vk::ColorComponentFlags::A,
                                    _ => {
                                        return Err(PipelineLoaderError::PipelineError(
                                            "unrecognized color write mask",
                                        ));
                                    }
                                }
                            }
                            state.color_write_mask = flags;
                        }
                    }
                    Ok(state)
                })
                .collect::<Result<Vec<_>, PipelineLoaderError>>()?;
            if blend_enable_dynamic_count == pipeline.attachments.len() as u32 {
                dynamic_states.push(vk::DynamicState::COLOR_BLEND_ENABLE_EXT);
            } else if blend_enable_dynamic_count != 0 {
                return Err(PipelineLoaderError::PipelineError(
                    "If some attachment[n].blend_enable was set to Dynamic, it must be set to Dynamic for all attachments",
                ));
            }
            if blend_equation_dynamic_count == pipeline.attachments.len() as u32 {
                dynamic_states.push(vk::DynamicState::COLOR_BLEND_EQUATION_EXT);
            } else if blend_equation_dynamic_count != 0 {
                return Err(PipelineLoaderError::PipelineError(
                    "If some attachment[n].blend_equation was set to Dynamic, it must be set to Dynamic for all attachments",
                ));
            }
            if color_write_mask_dynamic_count == pipeline.attachments.len() as u32 {
                dynamic_states.push(vk::DynamicState::COLOR_WRITE_MASK_EXT);
            } else if color_write_mask_dynamic_count != 0 {
                return Err(PipelineLoaderError::PipelineError(
                    "If some attachment[n].color_write_mask was set to Dynamic, it must be set to Dynamic for all attachments",
                ));
            }

            attachments
        };
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: pipeline
                .blend_logic_op_enable
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default()
                .into(),
            logic_op: if let Some(a) = pipeline.blend_logic_op.unwrap(&mut dynamic_states) {
                match a.as_str() {
                    "s & t" => vk::LogicOp::AND,
                    "s & !t" => vk::LogicOp::AND_REVERSE,
                    "!s & t" => vk::LogicOp::AND_INVERTED,
                    "!(s & t)" => vk::LogicOp::NAND,

                    "s | t" => vk::LogicOp::OR,
                    "s | !t" => vk::LogicOp::OR_REVERSE,
                    "!s | t" => vk::LogicOp::OR_INVERTED,
                    "!(s | t)" => vk::LogicOp::NOR,

                    "s ^ t" => vk::LogicOp::XOR,
                    "!(s ^ t)" => vk::LogicOp::EQUIVALENT,

                    "d" => vk::LogicOp::NO_OP,
                    "!d" => vk::LogicOp::INVERT,
                    "s" => vk::LogicOp::COPY,
                    "!s" => vk::LogicOp::COPY_INVERTED,
                    "0" => vk::LogicOp::CLEAR,
                    "1" => vk::LogicOp::SET,
                    _ => {
                        return Err(PipelineLoaderError::PipelineError(
                            "Unrecognized: blend_logic_op",
                        ));
                    }
                }
            } else {
                vk::LogicOp::CLEAR
            },
            blend_constants: pipeline
                .blend_constants
                .unwrap(&mut dynamic_states)
                .cloned()
                .unwrap_or_default(),
            ..Default::default()
        }
        .attachments(&color_blend_state_attachments);
        info = info.color_blend_state(&color_blend_state);

        let dynamic_states =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        info = info.dynamic_state(&dynamic_states);

        let rendering_state_color_attachments_format: Vec<vk::Format> = pipeline
            .attachments
            .iter()
            .map(|x| x.format.into())
            .collect();
        let mut rendering_state = vk::PipelineRenderingCreateInfo {
            depth_attachment_format: pipeline.depth_format.into(),
            stencil_attachment_format: pipeline.stencil_format.into(),
            ..Default::default()
        }
        .color_attachment_formats(&rendering_state_color_attachments_format);

        info = info.push(&mut rendering_state);

        fn compare_op_str(
            op: Option<&str>,
            err: &'static str,
        ) -> Result<vk::CompareOp, PipelineLoaderError> {
            match op {
                None | Some("false" | "0") => Ok(vk::CompareOp::NEVER),
                Some("<") => Ok(vk::CompareOp::LESS),
                Some("==") => Ok(vk::CompareOp::EQUAL),
                Some("<=") => Ok(vk::CompareOp::LESS_OR_EQUAL),
                Some(">") => Ok(vk::CompareOp::GREATER),
                Some("!=") => Ok(vk::CompareOp::NOT_EQUAL),
                Some(">=") => Ok(vk::CompareOp::GREATER_OR_EQUAL),
                Some("true" | "1") => Ok(vk::CompareOp::ALWAYS),
                _ => Err(PipelineLoaderError::PipelineError(err)),
            }
        }
        let span = tracing::span!(
            tracing::Level::INFO,
            "Creating Graphics Pipeline",
            path = load_context.asset_path().to_string()
        )
        .entered();
        let pipeline = self
            .pipeline_cache
            .create_graphics_pipeline(layout, &info)?;
        span.exit();
        Ok(GraphicsPipeline(Arc::new(pipeline)))
    }

    fn extensions(&self) -> &[&str] {
        &["gfx.pipeline.ron"]
    }
}
