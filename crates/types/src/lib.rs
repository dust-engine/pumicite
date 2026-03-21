use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use ash::vk;
pub mod format;


#[derive(Serialize, Deserialize)]
pub struct ComputePipeline {
    /// The compute shader used for compiling the compute pipeline
    pub shader: Shader,

    /// Path to the pipeline layout
    #[serde(default)]
    pub layout: PipelineLayoutRef,

    /// The created pipeline will not be optimized. Using this flag may reduce the time
    /// taken to create the pipeline.
    #[serde(default)]
    pub disable_optimization: bool,

    /// The compute pipeline can be used with vkCmdDispatchBase with a non-zero base workgroup.
    #[serde(default)]
    pub dispatch_base: bool,
}

#[derive(Serialize, Deserialize, Default)]
#[serde(untagged)]
pub enum OptionalDynamicState<T, const DYNAMIC_STATE: i32> {
    Dynamic,
    #[default]
    None,
    Static(T),
}

impl<T, const DYNAMIC_STATE: i32> OptionalDynamicState<T, DYNAMIC_STATE> {
    pub fn unwrap(&self, dynamic_states: &mut Vec<vk::DynamicState>) -> Option<&T> {
        match self {
            Self::Dynamic => {
                dynamic_states.push(vk::DynamicState::from_raw(DYNAMIC_STATE));
                None
            }
            Self::None => None,
            Self::Static(a) => Some(a),
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequiredDynamicState<T, const DYNAMIC_STATE: i32> {
    Dynamic,
    Static(T),
}
impl<T: Default, const DYNAMIC_STATE: i32> Default for RequiredDynamicState<T, DYNAMIC_STATE> {
    fn default() -> Self {
        Self::Static(T::default())
    }
}

impl<T, const DYNAMIC_STATE: i32> RequiredDynamicState<T, DYNAMIC_STATE> {
    pub fn unwrap(&self, dynamic_states: &mut Vec<vk::DynamicState>) -> Option<&T> {
        match self {
            Self::Dynamic => {
                dynamic_states.push(vk::DynamicState::from_raw(DYNAMIC_STATE));
                None
            }
            Self::Static(a) => Some(a),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum CountedDynamicState<T> {
    Dynamic,
    Count(u32),
    Static(Vec<T>),
}

#[derive(Serialize, Deserialize)]
pub struct GraphicsPipeline {
    pub shaders: BTreeMap<ShaderStage, Shader>,

    /// Dynamic: Requires VK_DYNAMIC_STATE_VERTEX_INPUT_EXT
    /// May be None when using mesh shading
    #[serde(default)]
    pub vertex_bindings: OptionalDynamicState<
        BTreeMap<u32, VertexInputBinding>,
        { vk::DynamicState::VERTEX_INPUT_EXT.as_raw() },
    >,

    /// Dynamic: VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY
    #[serde(default)]
    pub topology:
        RequiredDynamicState<PrimitiveTopology, { vk::DynamicState::PRIMITIVE_TOPOLOGY.as_raw() }>,

    /// Dynamic: VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE
    #[serde(default)]
    pub primitive_restart_enabled:
        RequiredDynamicState<bool, { vk::DynamicState::PRIMITIVE_RESTART_ENABLE.as_raw() }>,

    #[serde(default)]
    pub tessellation_patch_control_points:
        OptionalDynamicState<u32, { vk::DynamicState::PATCH_CONTROL_POINTS_EXT.as_raw() }>,

    #[serde(default = "viewport_default")]
    pub viewports: CountedDynamicState<Viewport>,
    #[serde(default = "scissor_default")]
    pub scissors: CountedDynamicState<Rect2D>,

    #[serde(default)]
    pub depth_clamp_enable:
        RequiredDynamicState<bool, { vk::DynamicState::DEPTH_CLAMP_ENABLE_EXT.as_raw() }>,

    #[serde(default)]
    pub rasterizer_discard_enable:
        RequiredDynamicState<bool, { vk::DynamicState::RASTERIZER_DISCARD_ENABLE.as_raw() }>,

    #[serde(default)]
    pub polygon_mode:
        RequiredDynamicState<PolygonMode, { vk::DynamicState::POLYGON_MODE_EXT.as_raw() }>,

    #[serde(default)]
    pub cull_mode: RequiredDynamicState<CullMode, { vk::DynamicState::CULL_MODE.as_raw() }>,

    #[serde(default)]
    pub front_face: RequiredDynamicState<FrontFace, { vk::DynamicState::FRONT_FACE.as_raw() }>,

    #[serde(default)]
    pub depth_bias_enable:
        RequiredDynamicState<bool, { vk::DynamicState::DEPTH_BIAS_ENABLE.as_raw() }>,

    #[serde(default)]
    pub depth_bias: OptionalDynamicState<DepthBias, { vk::DynamicState::DEPTH_BIAS.as_raw() }>,

    #[serde(default = "line_width_default")]
    pub line_width: RequiredDynamicState<f32, { vk::DynamicState::LINE_WIDTH.as_raw() }>,

    #[serde(default = "sample_count_default")]
    pub sample_count:
        RequiredDynamicState<u8, { vk::DynamicState::RASTERIZATION_SAMPLES_EXT.as_raw() }>,

    /// If None, sample shading is turned off.
    /// If Some(x), sample shading is turned on, and x is the minimum fraction of sample shading.
    #[serde(default)]
    pub sample_shading: Option<f32>,

    #[serde(default)]
    pub sample_mask: OptionalDynamicState<Vec<u32>, { vk::DynamicState::SAMPLE_MASK_EXT.as_raw() }>,

    #[serde(default)]
    pub alpha_to_coverage_enable:
        RequiredDynamicState<bool, { vk::DynamicState::ALPHA_TO_COVERAGE_ENABLE_EXT.as_raw() }>,
    #[serde(default)]
    pub alpha_to_one_enable:
        RequiredDynamicState<bool, { vk::DynamicState::ALPHA_TO_ONE_ENABLE_EXT.as_raw() }>,
    #[serde(default)]
    pub depth_test_enable:
        RequiredDynamicState<bool, { vk::DynamicState::DEPTH_TEST_ENABLE.as_raw() }>,
    #[serde(default)]
    pub depth_write_enable:
        RequiredDynamicState<bool, { vk::DynamicState::DEPTH_WRITE_ENABLE.as_raw() }>,

    #[serde(default)]
    pub depth_compare_op:
        RequiredDynamicState<String, { vk::DynamicState::DEPTH_COMPARE_OP.as_raw() }>,

    #[serde(default)]
    pub depth_bounds_test_enable:
        RequiredDynamicState<bool, { vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE.as_raw() }>,

    #[serde(default)]
    pub stencil_test_enable:
        RequiredDynamicState<bool, { vk::DynamicState::STENCIL_TEST_ENABLE.as_raw() }>,

    #[serde(default)]
    pub front_stencil: Option<StencilState>,

    #[serde(default)]
    pub back_stencil: Option<StencilState>,

    /// min, max
    #[serde(default)]
    pub depth_bounds: OptionalDynamicState<(f32, f32), { vk::DynamicState::DEPTH_BOUNDS.as_raw() }>,

    #[serde(default)]
    pub blend_logic_op_enable:
        RequiredDynamicState<bool, { vk::DynamicState::LOGIC_OP_ENABLE_EXT.as_raw() }>,

    #[serde(default)]
    pub blend_logic_op: OptionalDynamicState<String, { vk::DynamicState::LOGIC_OP_EXT.as_raw() }>,

    pub attachments: Vec<Attachment>,

    #[serde(default)]
    pub depth_format: format::Format,

    #[serde(default)]
    pub stencil_format: format::Format,

    #[serde(default)]
    pub blend_constants:
        RequiredDynamicState<[f32; 4], { vk::DynamicState::BLEND_CONSTANTS.as_raw() }>,

    #[serde(default)]
    pub layout: PipelineLayoutRef,
}

/// A "patch" to a [`GraphicsPipeline`] that allows the creation of a runtime variant of an existing
/// pipeline. Generally, runtime modification of pipeline states should be done with
/// [dynamic states](vk::DynamicState). However, a small amount of states cannot be modified
/// with dynamic states. This struct provides a workaround by allowing the user to specify these
/// values by providing them programmatically at pipeline creation time.
///
/// If a pipeline state is not known statically but you also cannot find it here,
/// it should probably be modified with [dynamic states](vk::DynamicState).
/// This helps reducing pipeline variant counts and allows the driver to select the best approach
/// to modify the attribute.
///
/// If the state you want to modify cannot be modified with dynamic state or that certain
/// drivers implement that dynamic state suboptimally, re-evalue
/// your use case. If you're confident that this is something you absolutely need, open a PR
/// and present your use case so that we can keep those use cases well-documented.
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct GraphicsPipelineVariant {
    #[serde(default)]
    pub shaders: BTreeMap<ShaderStage, BTreeMap<u32, SpecializationConstantType>>,

    #[serde(default)]
    pub depth_format: Option<format::Format>,

    #[serde(default)]
    pub stencil_format: Option<format::Format>,

    #[serde(default)]
    pub color_formats: BTreeMap<u32, format::Format>,
}
impl GraphicsPipelineVariant {
    pub fn apply_on(&self, pipeline: &mut GraphicsPipeline) {
        for (shader_stage, specialization_constants) in self.shaders.iter() {
            if let Some(shader) = pipeline.shaders.get_mut(shader_stage) {
                shader
                    .specialization_constants
                    .append(&mut specialization_constants.clone());
            }
        }
        if let Some(depth_format) = self.depth_format {
            pipeline.depth_format = depth_format;
        }
        if let Some(stencil_format) = self.stencil_format {
            pipeline.stencil_format = stencil_format;
        }
        for (index, format) in self.color_formats.iter() {
            pipeline.attachments[*index as usize].format = *format;
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Attachment {
    pub blend_enable:
        RequiredDynamicState<bool, { vk::DynamicState::COLOR_BLEND_ENABLE_EXT.as_raw() }>,

    #[serde(default)]
    pub blend_equation: OptionalDynamicState<
        BlendEquation,
        { vk::DynamicState::COLOR_BLEND_EQUATION_EXT.as_raw() },
    >,

    #[serde(default)]
    pub color_write_mask:
        OptionalDynamicState<String, { vk::DynamicState::COLOR_WRITE_MASK_EXT.as_raw() }>,

    pub format: format::Format,
}
#[derive(Serialize, Deserialize)]
pub struct BlendEquation {
    pub color: (BlendFactor, BlendOp, BlendFactor),
    pub alpha: (BlendFactor, BlendOp, BlendFactor),
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum BlendFactor {
    Zero = 0,
    One = 1,
    SrcColor = 2,
    OneMinusSrcColor = 3,
    DstColor = 4,
    OneMinusDstColor = 5,
    SrcAlpha = 6,
    OneMinusSrcAlpha = 7,
    DstAlpha = 8,
    OneMinusDstAlpha = 9,
    ConstantColor = 10,
    OneMinusConstantColor = 11,
    ConstantAlpha = 12,
    OneMinusConstantAlpha = 13,
    SrcAlphaSaturate = 14,
    Src1Color = 15,
    OneMinusSrc1Color = 16,
    Src1Alpha = 17,
    OneMiusSrc1Alpha = 18,
}
impl From<BlendFactor> for vk::BlendFactor {
    fn from(value: BlendFactor) -> Self {
        vk::BlendFactor::from_raw(value as i32)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum BlendOp {
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Min = 3,
    Max = 4,
}
impl From<BlendOp> for vk::BlendOp {
    fn from(value: BlendOp) -> Self {
        vk::BlendOp::from_raw(value as i32)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum StencilOp {
    Keep = 0,
    Zero = 1,
    Replace = 2,
    IncrementAndClamp = 3,
    DecrementAndClamp = 4,
    Invert = 5,
    IncrementAndWrap = 6,
    DecrementAndWrap = 7,
}
impl From<StencilOp> for vk::StencilOp {
    fn from(value: StencilOp) -> Self {
        vk::StencilOp::from_raw(value as i32)
    }
}

#[derive(Serialize, Deserialize)]
pub struct StencilState {
    pub ops: RequiredDynamicState<StencilStateOps, { vk::DynamicState::STENCIL_OP.as_raw() }>,
    pub compare_mask:
        RequiredDynamicState<u32, { vk::DynamicState::STENCIL_COMPARE_MASK.as_raw() }>,
    pub write_mask: RequiredDynamicState<u32, { vk::DynamicState::STENCIL_WRITE_MASK.as_raw() }>,
    pub reference: RequiredDynamicState<u32, { vk::DynamicState::STENCIL_REFERENCE.as_raw() }>,
}
#[derive(Serialize, Deserialize)]
pub struct StencilStateOps {
    pub fail: StencilOp,
    pub pass: StencilOp,
    pub depth_fail: StencilOp,
    pub compare: String,
}

#[derive(Serialize, Deserialize)]
pub struct DepthBias {
    pub constant_factor: f32,
    pub clamp: f32,
    pub slope_factor: f32,
}

#[derive(Default, Serialize, Deserialize, Clone, Copy)]
pub enum PolygonMode {
    #[default]
    Fill = 0,
    Line = 1,
    Point = 2,
}
impl From<PolygonMode> for vk::PolygonMode {
    fn from(value: PolygonMode) -> Self {
        vk::PolygonMode::from_raw(value as i32)
    }
}

#[derive(Default, Serialize, Deserialize, Clone, Copy)]
pub enum CullMode {
    #[default]
    None,
    /// Back-facing triangles are discarded
    Back,
    /// front-facing triangles are discarded
    Front,
    /// All triangles are discarded
    FrontAndBack,
}
impl From<CullMode> for vk::CullModeFlags {
    fn from(value: CullMode) -> Self {
        match value {
            CullMode::None => vk::CullModeFlags::NONE,
            CullMode::Back => vk::CullModeFlags::BACK,
            CullMode::Front => vk::CullModeFlags::FRONT,
            CullMode::FrontAndBack => vk::CullModeFlags::FRONT_AND_BACK,
        }
    }
}

#[derive(Default, Serialize, Deserialize, Clone, Copy)]
pub enum FrontFace {
    #[default]
    CounterClockwise,
    Clockwise,
}
impl From<FrontFace> for vk::FrontFace {
    fn from(value: FrontFace) -> Self {
        match value {
            FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl From<Viewport> for vk::Viewport {
    fn from(value: Viewport) -> Self {
        vk::Viewport {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
            min_depth: value.min_depth,
            max_depth: value.max_depth,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Rect2D {
    pub offset: (i32, i32),
    pub extent: (u32, u32),
}
impl From<Rect2D> for vk::Rect2D {
    fn from(value: Rect2D) -> Self {
        Self {
            offset: vk::Offset2D {
                x: value.offset.0,
                y: value.offset.1,
            },
            extent: vk::Extent2D {
                width: value.extent.0,
                height: value.extent.1,
            },
        }
    }
}

#[derive(Serialize, Deserialize, Default, Clone, Copy)]
pub enum PrimitiveTopology {
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    #[default]
    TriangleList = 3,
    TriangleStrip = 4,
    TriangleFan = 5,
    LineListWithAdjacency = 6,
    LineStripWithAdjacency = 7,
    TriangleListWithAdjacency = 8,
    TriangleStripWithAdjacency = 9,
    PatchList = 10,
}
impl From<PrimitiveTopology> for vk::PrimitiveTopology {
    fn from(value: PrimitiveTopology) -> Self {
        vk::PrimitiveTopology::from_raw(value as i32)
    }
}

#[derive(Serialize, Deserialize)]
pub struct VertexInputBinding {
    pub stride:
        RequiredDynamicState<u32, { vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE.as_raw() }>,
    pub input_rate: VertexInputRate,
    pub attributes: BTreeMap<u32, VertexInputAttributes>,
}
#[derive(Serialize, Deserialize, Clone)]
pub struct VertexInputAttributes {
    pub format: format::Format,
    pub offset: u32,
}
#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum VertexInputRate {
    Vertex,
    Instance,
}
impl From<VertexInputRate> for vk::VertexInputRate {
    fn from(value: VertexInputRate) -> Self {
        match value {
            VertexInputRate::Vertex => vk::VertexInputRate::VERTEX,
            VertexInputRate::Instance => vk::VertexInputRate::INSTANCE,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Shader {
    /// Path to the shader asset
    pub path: String,
    /// The entry point name of the shader for this stage.
    pub entry_point: String,

    /// Specialization constants for this shader.
    #[serde(default)]
    pub specialization_constants: BTreeMap<u32, SpecializationConstantType>,

    /// Specifies that `SubgroupSize` may vary in the shader stage
    #[serde(default)]
    pub allow_varying_subgroup: bool,
    /// Specifies that the subgroup sizes must be launched with all invocations active
    /// in the task, mesh, or compute stage.
    #[serde(default)]
    pub require_full_subgroups: bool,
}
impl Shader {
    pub fn flags(&self) -> vk::PipelineShaderStageCreateFlags {
        let mut flags = vk::PipelineShaderStageCreateFlags::empty();
        if self.require_full_subgroups {
            flags |= vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS;
        }
        if self.allow_varying_subgroup {
            flags |= vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE;
        }
        flags
    }
}

#[derive(Serialize, Deserialize)]
pub struct PipelineLayout {
    #[serde(default)]
    pub sets: Vec<DescriptorSetLayoutRef>,

    #[serde(default)]
    pub push_constants: BTreeMap<ShaderStage, (u32, u32)>,

    /// Provided by VK_EXT_graphics_pipeline_library
    /// Specifies that implementations must ensure that the properties and/or absence of a particular descriptor set
    /// do not influence any other properties of the pipeline layout. This allows pipelines libraries linked without
    /// VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT to be created with a subset of the total descriptor sets.
    #[serde(default)]
    pub independent_sets: bool,
}

#[derive(Serialize, Deserialize, Default)]
pub enum PipelineLayoutRef {
    Inline(PipelineLayout),
    Path(String),
    #[default]
    Bindless,
}

#[derive(Serialize, Deserialize)]
pub enum DescriptorSetLayoutRef {
    Inline(DescriptorSetLayout),
    Path(String),
    ResourceHeap,
    SamplerHeap,
}

fn binding_count_default() -> u32 {
    1
}
fn sample_count_default()
-> RequiredDynamicState<u8, { vk::DynamicState::RASTERIZATION_SAMPLES_EXT.as_raw() }> {
    RequiredDynamicState::Static(1)
}
fn viewport_default() -> CountedDynamicState<Viewport> {
    CountedDynamicState::Count(1)
}
fn scissor_default() -> CountedDynamicState<Rect2D> {
    CountedDynamicState::Count(1)
}
fn line_width_default() -> RequiredDynamicState<f32, { vk::DynamicState::LINE_WIDTH.as_raw() }> {
    RequiredDynamicState::Static(1.0)
}

#[derive(Serialize, Deserialize)]
pub struct Binding {
    /// The type of resource descriptors that are used for this binding.
    pub ty: DescriptorType,
    pub binding: u32,
    /// The number of descriptors contained in the binding, accessed in a shader as an array, except if
    /// [`Binding::ty`] is [`DescriptorType::InlineUniformBlock`] in which case `count` is the size in bytes of the inline uniform block.
    #[serde(default = "binding_count_default")]
    pub count: u32,
    /// Pipeline shader stages that can access a resource for this binding.
    pub stages: Vec<ShaderStage>,

    #[serde(default)]
    pub samplers: (),

    /// Specifies that if descriptors in this binding are updated between when the descriptor set is bound in a command buffer
    /// and when that command buffer is submitted to a queue, then the submission will use the most recently set descriptors for
    /// this binding and the updates do not invalidate the command buffer.
    #[serde(default)]
    pub update_after_bind: bool,

    /// Specifies that descriptors in this binding that are not dynamically used need not contain valid descriptors at the time
    /// the descriptors are consumed. A descriptor is dynamically used if any shader invocation executes an instruction that
    /// performs any memory access using the descriptor. If a descriptor is not dynamically used, any resource referenced by
    /// the descriptor is not considered to be referenced during command execution.
    #[serde(default)]
    pub update_unused_while_pending: bool,

    /// Specifies that descriptors in this binding that are not dynamically used need not contain valid descriptors at the time
    /// the descriptors are consumed. A descriptor is dynamically used if any shader invocation executes an instruction that performs
    /// any memory access using the descriptor. If a descriptor is not dynamically used, any resource referenced by the descriptor is not
    /// considered to be referenced during command execution.
    #[serde(default)]
    pub partially_bound: bool,

    /// Specifies that this is a variable-sized descriptor binding whose size will be specified when a descriptor set is allocated
    /// using this layout. The value of descriptorCount is treated as an upper bound on the size of the binding. This must only be used
    /// for the last binding in the descriptor set layout (i.e. the binding with the largest value of binding).
    #[serde(default)]
    pub variable_descriptor_count: bool,
}
#[derive(Serialize, Deserialize)]
pub struct DescriptorSetLayout {
    /// Specifies that descriptor sets must not be allocated using this layout, and descriptors are instead pushed by vkCmdPushDescriptorSet.
    #[serde(default)]
    pub push_descriptor: bool,

    /// Specifies that descriptor sets using this layout must be allocated from a descriptor pool created with the
    /// VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT bit set.
    #[serde(default)]
    pub update_after_bind_pool: bool,

    /// Specifies that this layout must only be used with descriptor buffers.
    #[serde(default)]
    pub descriptor_buffer: bool,

    /// Descriptor bindings
    pub bindings: Vec<Binding>,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum DescriptorType {
    Sampler,
    CombinedImageSampler,
    SampledImage,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    UniformBufferDynamic,
    StorageBufferDynamic,
    InputAttachment,
    InlineUniformBlock,
    AccelerationStructure,
    Mutable,
}
impl From<DescriptorType> for ash::vk::DescriptorType {
    fn from(value: DescriptorType) -> Self {
        match value {
            DescriptorType::Sampler => vk::DescriptorType::SAMPLER,
            DescriptorType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            DescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            DescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            DescriptorType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            DescriptorType::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            DescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            DescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            DescriptorType::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            DescriptorType::StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            DescriptorType::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
            DescriptorType::InlineUniformBlock => vk::DescriptorType::INLINE_UNIFORM_BLOCK,
            DescriptorType::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            DescriptorType::Mutable => vk::DescriptorType::MUTABLE_EXT,
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum ShaderStage {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Fragment,
    Compute,

    RayGen,
    AnyHit,
    ClosestHit,
    Miss,
    Intersection,
    Callable,

    Task,
    Mesh,
}
impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
        match value {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            ShaderStage::TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            ShaderStage::RayGen => vk::ShaderStageFlags::RAYGEN_KHR,
            ShaderStage::AnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
            ShaderStage::ClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ShaderStage::Miss => vk::ShaderStageFlags::MISS_KHR,
            ShaderStage::Intersection => vk::ShaderStageFlags::INTERSECTION_KHR,
            ShaderStage::Callable => vk::ShaderStageFlags::CALLABLE_KHR,
            ShaderStage::Task => vk::ShaderStageFlags::TASK_EXT,
            ShaderStage::Mesh => vk::ShaderStageFlags::MESH_EXT,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum SpecializationConstantType {
    UInt32(u32),
    UInt16(u16),
    UInt8(u8),
    Int32(i32),
    Int16(i16),
    Int8(i8),
    Bool(bool),
    Float(f32),
}
impl SpecializationConstantType {
    pub fn extend(&self, data: &mut Vec<u8>) -> usize {
        match self {
            SpecializationConstantType::UInt32(v) => {
                data.extend_from_slice(&v.to_ne_bytes());
                4
            }
            SpecializationConstantType::UInt16(v) => {
                data.extend_from_slice(&v.to_ne_bytes());
                2
            }
            SpecializationConstantType::UInt8(v) => {
                data.push(*v);
                1
            }
            SpecializationConstantType::Int32(v) => {
                data.extend_from_slice(&v.to_ne_bytes());
                4
            }
            SpecializationConstantType::Int16(v) => {
                data.extend_from_slice(&v.to_ne_bytes());
                2
            }
            SpecializationConstantType::Int8(v) => {
                data.push(*v as u8);
                1
            }
            SpecializationConstantType::Bool(v) => {
                let vk_bool = if *v { 1u32 } else { 0u32 };
                data.extend_from_slice(&vk_bool.to_ne_bytes());
                4
            }
            SpecializationConstantType::Float(v) => {
                data.extend_from_slice(&v.to_ne_bytes());
                4
            }
        }
    }
}

fn default_max_ray_recursion_depth() -> u32 {
    1
}
#[derive(Serialize, Deserialize)]
pub struct RayTracingPipeline {
    /// The compute shader used for compiling the compute pipeline
    pub stages: Vec<RayTracingPipelineShaderStage>,

    #[serde(default)]
    /// Shaders that could be referenced by hitgroups
    pub shaders: BTreeMap<String, Shader>,

    /// Path to the pipeline layout
    #[serde(default)]
    pub layout: PipelineLayoutRef,

    #[serde(default = "default_max_ray_recursion_depth")]
    pub max_ray_recursion_depth: u32,

    pub max_ray_payload_size: u32,
    pub max_hit_attribute_size: u32,

    #[serde(default)]
    pub dynamic_stack_size: bool,

    /// The created pipeline will not be optimized. Using this flag may reduce the time
    /// taken to create the pipeline.
    #[serde(default)]
    pub disable_optimization: bool,

    /// The compute pipeline can be used with vkCmdDispatchBase with a non-zero base workgroup.
    #[serde(default)]
    pub dispatch_base: bool,
}

#[derive(Default, Serialize, Deserialize)]
pub enum RayTracingPipelineShaderHitGroupType {
    #[default]
    Triangles,
    Aabbs,
}

#[derive(Serialize, Deserialize)]
pub enum RayTracingPipelineShaderStage {
    RayGen {
        shader: Shader,
        #[serde(default)]
        param_size: u32,
    },
    Miss {
        shader: Shader,
        #[serde(default)]
        param_size: u32,
    },
    Callable {
        shader: Shader,
        #[serde(default)]
        param_size: u32,
    },
    HitGroup {
        #[serde(default)]
        ty: RayTracingPipelineShaderHitGroupType,
        #[serde(default)]
        intersection: HitgroupShader,
        #[serde(default)]
        any_hit: HitgroupShader,
        #[serde(default)]
        closest_hit: HitgroupShader,
        #[serde(default)]
        param_size: u32,
    },
}
#[derive(Serialize, Deserialize, Default)]
#[serde(untagged)]
pub enum HitgroupShader {
    Reused(String),
    Singleton(Shader),
    #[default]
    None,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_deserialize() {
        let ron = "
            ComputePipeline(
                shader: Shader(
                    path: \"aaa.spv\",
                    entry_point: \"main\",
                ),
                layout: \"mylayout.playout.ron\"
            )
        ";
        let _pipeline: ComputePipeline = ron::de::from_str(ron).unwrap();

        let ron = "
            RayTracingPipeline(
                layout: \"mylayout.playout.ron\",
                max_ray_payload_size: 0,
                max_hit_attribute_size: 0,

                shaders: [
                    RayGen((
                        path: \"aaa.spv\",
                        entry_point: \"main\",
                    )),
                    HitGroup(
                        closest_hit: Shader(
                            path: \"aaa.spv\",
                            entry_point: \"main\",
                        ),
                        any_hit: \"MyShader\",
                    )
                ]
            )
        ";
        let _pipeline: RayTracingPipeline = ron::de::from_str(ron).unwrap();
    }
}
