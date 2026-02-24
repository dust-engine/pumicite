
// Output of the mesh shader, and input to the fragment shader.
struct CoarseVertex
{
    float3 color;
};

// Output of the fragment shader
struct Fragment
{
    float4 color;
};

// Mesh Shader

struct MeshStageOutput
{
    CoarseVertex    coarseVertex    : CoarseVertex;
    float4          sv_position     : SV_Position;
};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshMain(
    out indices uint3 triangles[1], 
    out vertices MeshStageOutput vertices[3], 
    uint3 DispatchThreadID: SV_DispatchThreadID
) {
    SetMeshOutputCounts(3, 1);

    vertices[0].sv_position = float4(0, -0.5, 0, 1);
    vertices[1].sv_position = float4(0.5, 0.5, 0, 1);
    vertices[2].sv_position = float4(-0.5, 0.5, 0, 1);

    vertices[0].coarseVertex = CoarseVertex(float3(1, 0, 0));
    vertices[1].coarseVertex = CoarseVertex(float3(0, 1, 0));
    vertices[2].coarseVertex = CoarseVertex(float3(0, 0, 1));
    
    triangles[0] = uint3(0, 1, 2);
}

// Fragment Shader

[shader("fragment")]
Fragment fragmentMain(
    CoarseVertex coarseVertex : CoarseVertex
) : SV_Target {
    float3 color = coarseVertex.color;

    Fragment output;
    output.color = float4(color, 1.0);
    return output;
}
