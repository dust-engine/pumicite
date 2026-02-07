struct NewLine {
    float2 point1;
    float2 point2;
    uint exists; // 1 for true, 0 for false
};

[[vk::push_constant]]
cbuffer MyPushConstants {
    uint clear_pass; // 1 for true, 0 for false
    uint uOffset_cascade_pass;
    uint2 texture_size;
    uint3 draw_color;
    uint last_image;
    uint base_ray_count;
};

static float4 BASE_COLOR = float4(0.0, 0.0, 0.0, 0.0);
static float LINE_RADIUS = 5.0; 
static float TAU = 6.28318530718;
static int RAY_COUNT = 16;
static int MAX_STEPS = 32;

[[vk::binding(0, 0)]] StructuredBuffer<NewLine> new_line;
[[vk::binding(1, 0)]] RWTexture2D<float4> paint_image;
[[vk::binding(2, 0)]] RWTexture2D<float4> distance_image1;
[[vk::binding(3, 0)]] RWTexture2D<float4> distance_image2;
[[vk::binding(4, 0)]] RWTexture2D<float4> output_image;
[[vk::binding(5, 0)]] RWTexture2D<float4> cascade_image1;
[[vk::binding(6, 0)]] RWTexture2D<float4> cascade_image2;



float sdfLineSquared(float2 p, float2 from, float2 to) {
    float2 toStart = p - from;
    float2 line = to - from;
    float lineLengthSquared = dot(line, line);
    float t = clamp(dot(toStart, line) / lineLengthSquared, 0.0, 1.0);
    float2 closestVector = toStart - line * t;
    return dot(closestVector, closestVector);
}

[numthreads(8, 8, 1)]
void make_lines(uint3 dispatchThreadID : SV_DispatchThreadID) {
    NewLine line = new_line[0];
    int2 coord = int2(dispatchThreadID.xy);

    // clear_pass image if necessary
    if (clear_pass == 1) {
        paint_image[coord] = BASE_COLOR;
    }

    if (line.exists == 1) {
        // Convert normalized [0–1] line points into pixel space
        float2 p1 = line.point1 * float2(texture_size);
        float2 p2 = line.point2 * float2(texture_size);
        float2 p = float2(coord);

        if (any(output_image[uint2(p1)] != float4(0.0, 0.0, 0.0, 0.0))) {
            return;
        }

        // Compute distance in pixel space
        float distSquared = sdfLineSquared(p, p1, p2);

        // Set radius in pixel units
        if (distSquared <= LINE_RADIUS * LINE_RADIUS) {
            paint_image[coord] = float4(float3(draw_color) / 255.0, 1.0);
        }


    }
}

[numthreads(8, 8, 1)]
void distance(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 coord = int2(dispatchThreadID.xy);

    // if first pass
    if (clear_pass == 0) {
        // set the lines to be thier pixel coords
        float alpha = paint_image[coord].a;
        distance_image2[coord] = float4(coord * alpha, 0, 1);
        return;
    }
    // if last pass
    if (last_image == 1) {
        float2 nearestSeed;

        if (clear_pass % 2 == 0) {
            nearestSeed = distance_image2[coord].xy;
        } else {
            nearestSeed = distance_image1[coord].xy;
        }

        // Clamp by the size of our texture
        float distanceVal = clamp(distance(coord, nearestSeed), 0.0, max(texture_size.x, texture_size.y));

        // Normalize and visualize the distance
        if (clear_pass % 2 == 0) {
            distance_image1[coord] = float4(float3(distanceVal), 1.0);
        } else {
            distance_image2[coord] = float4(float3(distanceVal), 1.0);
        }
    }

    float4 nearestSeed = float4(-50000.0);
    float nearestDist = 9999999.9;

    for (int y = -1; y <= 1; y += 1) {
        for (int x = -1; x <= 1; x += 1) {
            int2 sampleUV = coord + (int2(x, y) * uOffset_cascade_pass);

            if (sampleUV.x < 0 || sampleUV.x > texture_size.x - 1 || sampleUV.y < 0 || sampleUV.y > texture_size.y - 1) continue;

            float4 sampleValue;
            if (clear_pass % 2 == 0) {
                sampleValue = distance_image1[sampleUV];
            } else {
                sampleValue = distance_image2[sampleUV];
            }

            float2 sampleSeed = sampleValue.xy;

            if (sampleSeed.x != 0.0 || sampleSeed.y != 0.0) {
                float2 diff = sampleSeed - float2(coord);
                float dist = dot(diff, diff);
                if (dist < nearestDist) {
                    nearestDist = dist;
                    nearestSeed = sampleValue;
                }
            }
        }
    }
    if (clear_pass % 2 == 0) {
        distance_image2[coord] = nearestSeed;
    } else {
        distance_image1[coord] = nearestSeed;
    }
}

//
// RADIANCE CASCADE LIGHTING PASS
//

bool outOfBounds(float2 uv) {
    return uv.x < 0.0 || uv.y < 0.0 ||
           uv.x >= float(texture_size.x) ||
           uv.y >= float(texture_size.y);
}
float random(float2 seed)
{
    return frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453123);
}
float4 sampleBilinear(RWTexture2D<float4> tex, float2 coord, float2 gridSpacing)
{
    int2 base = int2(floor(coord));
    float2 f = coord - float2(base);

    float2 gridBase = floor(coord / gridSpacing) * gridSpacing;

    float4 sum = 0;
    int count = 0;

    int2 texSize = int2(texture_size);

    // Loop over 2x2 neighbors
    [unroll]
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int2 sampleCoord = base + int2(dx, dy);

            // Skip out-of-bounds
            if (sampleCoord.x < 0 || sampleCoord.y < 0 || 
                sampleCoord.x >= texSize.x || sampleCoord.y >= texSize.y)
                continue;

            // Skip if outside the same grid cell
            if (sampleCoord.x >= gridBase.x + gridSpacing.x ||
                sampleCoord.y >= gridBase.y + gridSpacing.y)
                continue;

            sum += tex[uint2(sampleCoord)];
            count++;
        }
    }

    return (count > 0) ? sum / float(count) : float4(0,0,0,0);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint cascadeIndex = uOffset_cascade_pass;
    int2 coord = int2(dispatchThreadID.xy);
    uint isFirstPass = last_image;
    if (any(output_image[coord] != float4(0.0, 0.0, 0.0, 0.0)) && cascadeIndex == 0) {
        return;
    }
    // Direct emission (the drawn line pixels)
    float4 base_color = paint_image[coord];
    if (base_color.a > 0.01 && cascadeIndex == 0) {
        output_image[coord] = base_color;
        return;
    }
    if (isFirstPass == 1) {
        cascade_image1[coord] = float4(0.0);
        cascade_image2[coord] = float4(0.0);
    }
    float rayCount = pow(base_ray_count, cascadeIndex + 1.0);

    // A handy term we use in other calculations
    float sqrtBase = sqrt(float(base_ray_count));
    // The width / space between probes
    float spacing = pow(sqrtBase, cascadeIndex);
    // Calculate the number of probes per x/y dimension
    float2 size = floor(float2(texture_size) / spacing);
    // Calculate which probe we're processing this pass
    float2 probeRelativePosition = fmod(float2(coord), size);
    // Calculate which group of rays we're processing this pass
    float2 rayPos = floor(float2(coord) / size);
    // Calculate the index of the set of rays we're processing
    float baseIndex = float(base_ray_count) * (rayPos.x + (spacing * rayPos.y));
    // Calculate the size of our angle step
    float angleStepSize = TAU / float(rayCount);
    // Find the center of the probe we're processing
    float2 probeCenter = (probeRelativePosition + 0.5) * spacing;
    uint2 effectiveCoord = uint2(round(probeCenter));

    float modifierHack = base_ray_count < 16.0 ? 2.0 : sqrtBase;
    float intervalStart = cascadeIndex == 0 ? 0.0 : (pow(base_ray_count, cascadeIndex - 1.0));
    float intervalLength = pow(base_ray_count, cascadeIndex);
    intervalStart *= modifierHack;
    intervalLength *= modifierHack;

    float oneOverRayCount = 1.0 / float(base_ray_count);

    float4 radiance = float4(0.0, 0.0, 0.0, 1.0);

    RWTexture2D<float4> distanceTexture = (clear_pass % 2 == 0) ? distance_image1 : distance_image2;
    RWTexture2D<float4> readCascade = cascadeIndex % 2 == 0 ? cascade_image1 : cascade_image2;
    RWTexture2D<float4> writeCascade = cascadeIndex % 2 == 0 ? cascade_image2 : cascade_image1;

    for (int i = 0; i < base_ray_count; i++) {
        float index = baseIndex + float(i);
        float angleStep = index + 0.5;
        float angle = angleStepSize * angleStep;
        float2 rayDirection = float2(cos(angle), -sin(angle));

        // Start at the probe center and raymarch outward
        float2 sampleCoord = effectiveCoord + (rayDirection * intervalStart);
        float traveled = 0.0;
        float4 radDelta = float4(0.0);

        for (int step = 1; step < MAX_STEPS; step++) {
            float dist = distanceTexture[uint2(sampleCoord)].r;
            sampleCoord += rayDirection * dist;

            if (outOfBounds(sampleCoord)) break;

            if (dist < 1.0) {
                float4 colorSample = paint_image[uint2(sampleCoord)];
                colorSample.rgb = pow(colorSample.rgb, 2.2);
                radDelta += float4(colorSample.rgb, 1.0);
                break;
            }
            traveled += dist;
            if (traveled >= intervalLength) break;
        }
        if (isFirstPass != 1 && radDelta.a == 0.0) {
            // The spacing between probes
            // The spacing between probes
            float upperSpacing = pow(sqrtBase, cascadeIndex + 1.0);
            // Grid of probes
            float2 upperSize = floor(texture_size / upperSpacing);
            // Position of _this_ probe
            float2 upperPosition = float2(
                fmod(index, upperSpacing), floor(index / upperSpacing)
            ) * upperSize;
            float2 offset = (probeRelativePosition + 0.5) / sqrtBase;
            float2 clamped = clamp(offset, float2(0.5), upperSize - 0.5);
            float2 upperCoord = (upperPosition + clamped);


            radDelta += sampleBilinear(readCascade, upperCoord, upperSize);
        }

        radiance += radDelta;
    }

    radiance.rgb *= oneOverRayCount;
    if (cascadeIndex != 0) {
        writeCascade[coord] = radiance;
        return;
    }

    // Final pass: write the computed radiance to the output image
    radiance = float4(pow(radiance.rgb, 1.0 / 2.2), 1.0);
    output_image[coord] = saturate(radiance);
}
