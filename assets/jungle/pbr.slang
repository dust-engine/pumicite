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

struct ModelVertex {
    float3 position;
    float3 normal;
    float3 color;
    float2 texcoords;
}

struct ModelData {
    float3* positions;
    float3* normals;
    float3 *colors;
    float2 *texcoords;
    ModelVertex vertexAt(uint vertexID) {
        ModelVertex vertex;
        if (positions != nullptr) {
            vertex.position = positions[vertexID];
        }
        if (normals != nullptr) {
            vertex.normal = normals[vertexID];
        }
        if (colors != nullptr) {
            vertex.color = colors[vertexID];
        }
        if (texcoords != nullptr) {
            vertex.texcoords = texcoords[vertexID];
        }
        return vertex;
    }
}
struct InstanceData {
    float4x4 instance_transforms[];
}

struct CameraUniforms {
    float4x4 view;
    float4x4 projection;
}
struct GltfMaterialData {
    float4 baseColorFactor;
    uint32_t baseColorTextureIndex;
    uint32_t baseColorSamplerIndex;
    uint32_t _padding[2];
}

struct PushConstants {
    // Vertex shader
    ModelData *p_model_data;
    InstanceData *p_instance_data;
    CameraUniforms *p_camera;
    GltfMaterialData *p_material;
}
[vk::push_constant]
PushConstants pc;

// Output of the vertex shader, and input to the fragment shader.
struct CoarseVertex
{
    float3 worldPos;
    float3 normal;
    float3 color;
    float2 texcoords;
};

// Output of the fragment shader
struct Fragment
{
    float4 color;
};

// Vertex  Shader

struct VertexStageOutput
{
    CoarseVertex    coarseVertex    : CoarseVertex;
    float4          sv_position     : SV_Position; // Clip-space position
};

[shader("vertex")]
VertexStageOutput vertexMain(
    uint vertexID : SV_VertexID,
    uint instanceID : SV_InstanceID,
) {
    VertexStageOutput output;

    // Pull vertex data from buffers using vertex ID
    ModelVertex vertex = pc.p_model_data.vertexAt(vertexID);
    float4x4 model = pc.p_instance_data.instance_transforms[instanceID];

    // Transform position to world space
    float4 worldPos = mul(model, float4(vertex.position, 1.0));
    
    // Transform normal to world space (assuming uniform scaling, otherwise use normal matrix)
    float3 worldNormal = normalize(mul((float3x3)model, vertex.normal));

    // Transform to clip space
    float4 viewPos = mul(pc.p_camera.view, worldPos);
    float4 clipPos = mul(pc.p_camera.projection, viewPos);

    output.coarseVertex.worldPos = worldPos.xyz;
    output.coarseVertex.normal = worldNormal;
    if (pc.p_model_data.colors) {
        output.coarseVertex.color = vertex.color;
    } else {
        output.coarseVertex.color = float3(1.0, 1.0, 1.0);
    }
    output.coarseVertex.texcoords = vertex.texcoords;
    output.sv_position = clipPos;

    return output;
}

// Fragment Shader
/*
[shader("fragment")]
Fragment fragmentMain(
    CoarseVertex coarseVertex : CoarseVertex,
    uniform float3 viewPos,
    uniform float3 lightPos,
    uniform float3 lightColor,
    uniform float3 ambientColor,
    uniform float shininess) : SV_Target
{
    float3 worldPos = coarseVertex.worldPos;
    float3 N = normalize(coarseVertex.normal);
    float3 albedo = coarseVertex.color;

    // Light direction
    float3 L = normalize(lightPos - worldPos);
    
    // View direction
    float3 V = normalize(viewPos - worldPos);
    
    // Half vector for Blinn-Phong
    float3 H = normalize(L + V);

    // Ambient component
    float3 ambient = ambientColor * albedo;

    // Diffuse component
    float NdotL = max(dot(N, L), 0.0);
    float3 diffuse = albedo * lightColor * NdotL;

    // Specular component (Blinn-Phong)
    float NdotH = max(dot(N, H), 0.0);
    float specular = pow(NdotH, shininess);
    float3 specularColor = lightColor * specular;

    // Combine all components
    float3 finalColor = ambient + diffuse + specularColor;

    Fragment output;
    output.color = float4(finalColor, 1.0);
    return output;
}
*/


[shader("fragment")]
Fragment fragmentMain(
    float3 fragCoords: SV_Position,
    CoarseVertex coarseVertex,
) : SV_Target {

    float4 baseColor = pc.p_material.baseColorFactor;
    if (pc.p_material.baseColorTextureIndex != ~0) {
        Texture2D.Handle baseColorTexture = (Texture2D.Handle)pc.p_material.baseColorTextureIndex;
        SamplerState.Handle baseColorSampler = (SamplerState.Handle)pc.p_material.baseColorSamplerIndex;
        baseColor *= baseColorTexture.Sample(baseColorSampler, coarseVertex.texcoords);
    } 

    Fragment output;
    output.color = float4(baseColor.xyz, 1.0);
    return output;
}
