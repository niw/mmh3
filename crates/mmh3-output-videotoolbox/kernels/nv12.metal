#include <metal_stdlib>
using namespace metal;
float3 rgb(device const float *pixels, uint pos, uint stride) {
    return float3(pixels[pos], pixels[pos + stride], pixels[pos + stride * 2]);
}

float luma(float3 value) {
    return 0.2126f * value.r + (1.0f - 0.2126f - 0.0722f) * value.g + 0.0722f * value.b;
}

float quantize(float value) { return clamp(floor(value + 0.5f), 0.0f, 255.0f) / 255.0f; }

kernel void rgb_to_nv12(device const float *pixels [[buffer(0)]], constant uint *p [[buffer(1)]],
                        texture2d<float, access::write> y [[texture(0)]],
                        texture2d<float, access::write> uv [[texture(1)]],
                        uint2 i [[thread_position_in_grid]]) {
    uint w = p[0], h = p[1], frames = p[2], frame = p[3];
    if (i.x >= w / 2 || i.y >= h / 2)
        return;
    uint stride = frames * w * h, offset = frame * w * h;
    float3 sum = float3(0);

    for (uint dy = 0; dy < 2; ++dy) {
        for (uint dx = 0; dx < 2; ++dx) {
            uint2 coord = i * 2 + uint2(dx, dy);
            float3 value = rgb(pixels, offset + coord.y * w + coord.x, stride);
            y.write(float4(quantize(16.0f + 219.0f * luma(value)), 0, 0, 1), coord);
            sum += value;
        }
    }

    float3 average = sum / 4.0f;
    float yy = luma(average);
    float cb = (average.b - yy) / (2.0f * (1.0f - 0.0722f));
    float cr = (average.r - yy) / (2.0f * (1.0f - 0.2126f));
    uv.write(float4(quantize(128.0f + 224.0f * cb), quantize(128.0f + 224.0f * cr), 0, 1), i);
}
