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
    uniform float2 u_transform,
) {
    VertexStageOutput output;

    output.sv_position = float4(2.0 * position / u_transform - 1.0, 0.0, 1.0);

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


[[vk::constant_id(0)]]
const bool CONVERT_TO_LINEAR = true;

[shader("fragment")]
float4 fragmentMain(
    float4 color: COLOR,
    float2 uv: UV,
    Sampler2D sampler,
) : SV_Target
{
    float4 texture_color_linear = sampler.Sample(uv); // This blending is done in gamma space and already has alpha premultiplied.

    float4 result = color * sampler.Sample(uv);

    if (CONVERT_TO_LINEAR) {

        result.xyz = SRGBToLinear(result.xyz);
    }


    return result;
}
