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



[numthreads(8, 8, 1)]
void main(
    uint2 dispatchThreadID: SV_DispatchThreadID,
    uniform uint32_t image_handle,
    uniform float iTime,
) {
    RWTexture2D.Handle image = (RWTexture2D.Handle)image_handle;
    uint2 iResolution;
    image.GetDimensions(iResolution.x, iResolution.y);

    float4 out;
    mainImage(out, float2(dispatchThreadID), iResolution, iTime);
    image->Store(iResolution - dispatchThreadID, out);
}

// From https://www.shadertoy.com/view/tXjXDy
void mainImage(out float4 o, float2 u, float2 iResolution, float iTime) {
    o = float4(1.0);
    float s = .002, i = 0.0, n = 0.0;
    float3 r = float3(iResolution, 0.0);
    float3 p = float3(0.0);
    u = (u - r.xy / 2.) / r.y - .3;
    o *= i;
    for (; i++ < 32. && s > .001; o += float4(5, 2, 1, 0) / length(u - .1)) {
        p += float3(u * s, s);
        s = 1. + p.y;
        n = .01;
        for (; n < 1.; n += n) {
            s += abs(dot(sin(p.z + iTime + p / n), float3(1.0))) * n * .1;
        }
    }
    o = tanh(o / 5e2);
}
