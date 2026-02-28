// Mesh Shader Culling

// Number of task shader invocations per task shader workgroup
static const uint numTaskInvocationsX = 2;
static const uint numTaskInvocationsY = 2;

static const uint numMeshInvocationsX = 2;
static const uint numMeshInvocationsY = 2;

struct SharedData {
    float2 position;
    float2 offsets[numTaskInvocationsX * numTaskInvocationsY];
    float2 size;
}

groupshared SharedData payload;

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

struct PushConstants {
    float2 cull_center;
    float cull_radius;
}
[vk::push_constant]
PushConstants pc;

float square(float x) {
    return x * x;
}

float square(float2 p) {
    return p.x * p.x + p.y * p.y;
}

// Task Shader
//
// Use numTaskInvocationsX * numTaskInvocationsY task shader invocations per task shader workgroup,
// resulting in N * N * numTaskInvocationsX * numTaskInvocationsY invocations

[shader("amplification")]
[numthreads(numTaskInvocationsX, numTaskInvocationsY, 1)]
void taskMain(
    uint3 DispatchThreadID: SV_DispatchThreadID,
    uint GroupIndex: SV_GroupIndex
) {
    // calculate size and offset of sub-rect per task shader invocation
    float2 size = 1.0 / float2(WorkgroupCount().x * WorkgroupSize().x, WorkgroupCount().y * WorkgroupSize().y);
    float2 offset = 0.5 * float2(DispatchThreadID.x * size.x, DispatchThreadID.y * size.y);

    // determine the four corners of the sub-rect and check if it's completely out of the culling circle
    // (ignoring the case that a sub-rect could completely include the culling circle...)

    float2 position = pc.cull_center;
    float2 p0 = position + offset;
    float2 p1 = p0 + float2(size.x, 0);
    float2 p2 = p0 + size;
    float2 p3 = p0 + float2(0, size.y);
    float squaredRadius = square(pc.cull_radius);
    bool isValid = (square(p0) < squaredRadius)
                && (square(p1) < squaredRadius)
                && (square(p2) < squaredRadius)
                && (square(p3) < squaredRadius);

    if (isValid) {
        // get the next free index into the offsets array
        uint index = WavePrefixCountBits(isValid);
        payload.offsets[index] = offset;
    }
    if (GroupIndex == 0) {
        // for just one task shader invocation we can emit mesh shaders
        payload.position = position;
        payload.size = size;
        // get the actual number of task shader invocations that are determined to display something
        uint validCount = WaveActiveCountBits(isValid);
        DispatchMesh(validCount, 1, 1, payload);
    }
}

struct MeshStageOutput
{
    CoarseVertex    coarseVertex    : CoarseVertex;
    float4          sv_position     : SV_Position;
};

// Mesh Shader
//
// We'll emit 4 vertices on 2 primitives per mesh shader invocation
[shader("mesh")]
[numthreads(numMeshInvocationsX, numMeshInvocationsY, 1)]
[outputtopology("triangle")]
void meshMain(
    out indices uint3 triangles[2 * numMeshInvocationsX * numMeshInvocationsY], 
    out vertices MeshStageOutput vertices[4 * numMeshInvocationsX * numMeshInvocationsY], 
    uint3 DispatchThreadID: SV_DispatchThreadID,
    uint3 GroupThreadID: SV_GroupThreadID,
    uint GroupID: SV_GroupID,
    uint GroupIndex: SV_GroupIndex
) {
    if (GroupIndex == 0) {
        // set the number of vertices and primitives to put out just once for the complete workgroup
        SetMeshOutputCounts(4 * numMeshInvocationsX * numMeshInvocationsY, 2 * numMeshInvocationsX * numMeshInvocationsY);
    }
    
    // determine 4 vertices by combining some global position and offset, as well as some local offset and size
    float2 globalOffset = payload.offsets[GroupID];
    float2 localSize = payload.size / float2(WorkgroupSize().x, WorkgroupSize().y);
    float2 localOffset = float2(GroupThreadID.x, GroupThreadID.y) * localSize;
    float2 v0 = payload.position + globalOffset + localOffset;
    float2 v1 = v0 + float2(localSize.x, 0);
    float2 v2 = v0 + localSize;
    float2 v3 = v0 + float2(0, localSize.y);

    uint vertexBaseIndex = 4 * GroupIndex;
    vertices[vertexBaseIndex + 0].sv_position = float4(2.0 * v0, 0, 1);
    vertices[vertexBaseIndex + 1].sv_position = float4(2.0 * v1, 0, 1);
    vertices[vertexBaseIndex + 2].sv_position = float4(2.0 * v2, 0, 1);
    vertices[vertexBaseIndex + 3].sv_position = float4(2.0 * v3, 0, 1);

    vertices[vertexBaseIndex + 0].coarseVertex = CoarseVertex(float3(1, 0 ,0));
    vertices[vertexBaseIndex + 1].coarseVertex = CoarseVertex(float3(0, 1, 0));
    vertices[vertexBaseIndex + 2].coarseVertex = CoarseVertex(float3(0, 0, 1));
    vertices[vertexBaseIndex + 3].coarseVertex = CoarseVertex(float3(1, 1, 0));

    uint triangleBaseIndex = 2 * GroupIndex;
    triangles[triangleBaseIndex + 0] = uint3(vertexBaseIndex + 0, vertexBaseIndex + 1, vertexBaseIndex + 2);
    triangles[triangleBaseIndex + 1] = uint3(vertexBaseIndex + 2, vertexBaseIndex + 3, vertexBaseIndex + 0);
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
