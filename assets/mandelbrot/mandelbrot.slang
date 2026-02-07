struct MandelbrotState {
    float2 center;    // The center of the view (usually set by user input)
    float scale;      // Scale for zooming 
    uint max_iter;    // Maximum number of iterations to check
};

[[vk::binding(0, 0)]] ConstantBuffer<MandelbrotState> state;
[format("rgba8")]
[[vk::binding(1, 0)]] RWTexture2D<float4> outputImage; 

[numthreads(8, 8, 1)] 
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 coord = int2(dispatchThreadID.xy);

    uint width, height;
    outputImage.GetDimensions(width, height); 

    // Scale pixel coordinates to Mandelbrot set coordinates 
    float2 c = float2(
        state.center.x + (float(coord.x) - float(width) / 2.0f) * 0.2446 * state.scale,  
        state.center.y + (float(coord.y) - float(height) / 2.0f) * 0.224 * state.scale   
    );

    float2 z = float2(0.0, 0.0); 
    uint iterations = 0;

    float bailout_radius_squared = 256.0f;

    for (uint i = 0; i < state.max_iter; i++) {
        float x_new = z.x * z.x - z.y * z.y + c.x;  // Real part of z^2 + c
        float y_new = 2.0 * z.x * z.y + c.y;        // Imaginary part of z^2 + c
        z = float2(x_new, y_new);                    // Update z

        if (dot(z, z) > bailout_radius_squared) {
            break;
        }

        iterations++;
    }

    float3 color = float3(0.0, 0.0, 0.0); 

    // If the pixel escaped, calculate the color using a smooth coloring method
    if (iterations < state.max_iter) {
        float log_zn = log(dot(z, z)) / 2.0;  
        float nu = log(log_zn / log(2.0)) / log(2.0);  

        float smooth_iterations = float(iterations) + 1.0 - nu;

        float t = smooth_iterations / float(state.max_iter);

        float r = 0.5 + 0.5 * sin(6.28 * t + 0.0); 
        float g = 0.5 + 0.5 * sin(6.28 * t + 3.0); 
        float b = 0.5 + 0.5 * sin(6.28 * t + 2.0); 

        color = float3(r, g, b);
    }

    outputImage[coord] = float4(color, 1.0);
}
