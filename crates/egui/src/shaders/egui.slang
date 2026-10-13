// `TextureId::User` textures: the id is a bindless handle into the descriptor
// heap's resource heap (set 0), sampled with a sampler from its sampler heap
// (set 1).
[vk::binding(0, 0)]
__DynamicResource<__DynamicResourceKind.General> resourceHandles[];

[vk::binding(0, 1)]
__DynamicResource<__DynamicResourceKind.Sampler> samplerHandles[];

export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
{
    __target_switch
    {
    case spirv:
        if (T.kind == DescriptorKind.Sampler)
            return samplerHandles[((uint2)handle).x].asOpaqueDescriptor<T>();
        else
            return resourceHandles[((uint2)handle).x].asOpaqueDescriptor<T>();
    default:
        return defaultGetDescriptorFromHandle(handle);
    }
}

// The managed texture of the mesh, push-descriptor bound.
[vk::binding(0, 2)]
Sampler2D managedTexture;

// Each stage reads its own range: the vertex stage `u_transform`, the fragment
// stage the rest.
struct PushConstants
{
    float2 u_transform;
    // All ones to sample `managedTexture`. Otherwise, the low 32 bits are a
    // resource-heap handle, and the high 32 bits the f16 pair (min, max) that
    // remaps the texel's color to 0-1, or 0 to leave it as is.
    Texture2D.Handle texture;
    SamplerState.Handle sampler;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

// Vertex  Shader
struct VertexStageOutput
{
    float4 sv_position : SV_Position;
    float4 color : COLOR;
    float2 uv : UV;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    float2 position,
    float2 uv,
    float4 color,
) {
    VertexStageOutput output;

    output.sv_position = float4(2.0 * position / pushConstants.u_transform - 1.0, 0.0, 1.0);

    // Do vertex color interpolation in gamma space.
    // https://github.com/emilk/egui/pull/2071
    output.color = color;
    output.uv = uv;

    return output;
}

float3 SRGBToLinear(float3 color) {
    return float3(
        color.x <= 0.04045 ? color.x / 12.92 : pow((color.x + 0.055) / 1.055, 2.4),
        color.y <= 0.04045 ? color.y / 12.92 : pow((color.y + 0.055) / 1.055, 2.4),
        color.z <= 0.04045 ? color.z / 12.92 : pow((color.z + 0.055) / 1.055, 2.4)
    );
}

float3 LinearToSRGB(float3 color) {
    return float3(
        color.x <= 0.0031308 ? color.x * 12.92 : 1.055 * pow(color.x, 1.0 / 2.4) - 0.055,
        color.y <= 0.0031308 ? color.y * 12.92 : 1.055 * pow(color.y, 1.0 / 2.4) - 0.055,
        color.z <= 0.0031308 ? color.z * 12.92 : 1.055 * pow(color.z, 1.0 / 2.4) - 0.055
    );
}


[[vk::constant_id(0)]]
const bool CONVERT_TO_LINEAR = true;

[shader("fragment")]
float4 fragmentMain(
    float4 color: COLOR,
    float2 uv: UV,
) : SV_Target
{
    float4 texel;
    uint2 texture = (uint2)pushConstants.texture;
    if (all(texture == uint2(0xFFFFFFFF))) {
        // This blending is done in gamma space and already has alpha premultiplied.
        texel = managedTexture.Sample(uv);
    } else {
        float4 linear = pushConstants.texture->Sample(*pushConstants.sampler, uv);
        if (texture.y != 0) {
            float2 range = unpackHalf2x16ToHalf(texture.y);
            linear.rgb = range.y > range.x
                ? (linear.rgb - range.x) / (range.y - range.x)
                : step(range.x, linear.rgb);
        }
        // Managed textures hold sRGB-encoded bytes, while user textures sample
        // as linear values, as in egui-wgpu. Encode them so that the tint
        // applies in gamma space, as it does for managed textures.
        texel = float4(LinearToSRGB(saturate(linear.rgb)), linear.a);
    }

    float4 result = color * texel;

    if (CONVERT_TO_LINEAR) {

        result.xyz = SRGBToLinear(result.xyz);
    }


    return result;
}
