#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// The activation side of INT8 ConvRot layers, shared by the DiT and VAE kernels: rotating
// 256-column groups by the normalized regular Hadamard matrix and quantizing rows to INT8.

namespace {

constexpr int CONVROT_GROUP = 256;

// Groups of 256 columns a warp rotates at most, so rows up to 32 warps × 4 × 256 = 32,768 columns
// fit in registers.
constexpr int MAX_GROUPS_PER_WARP = 4;

// Rotates a 256 group by the normalized regular Hadamard matrix: four radix-4 stages with strides
// 1, 4, 16 and 64 of the symmetric 4 × 4 kernel y0 = x0 + x1 + x2 - x3, y1 = x0 + x1 - x2 + x3, y2
// = x0 - x1 + x2 + x3, y3 = -x0 + x1 + x2 + x3, then a factor of 1/16. Lane l holds elements 8l to
// 8l + 7. Of the index bits, bits 0-1 (stride 1) are in the register index, bits 2-3 (stride 4) are
// register bit 2 and lane bit 0, bits 4-5 (stride 16) lane bits 1-2 and bits 6-7 (stride 64) lane
// bits 3-4.
//
// NOTE: Every output keeps the order of additions written above, so the result is the same in every
// bit whichever lane computes it. The rewritten forms below only use a + b == b + a and a + (-b) ==
// a - b, which hold exactly.
__device__ __forceinline__ void rotate_group(float (&values)[8], int lane) {
#pragma unroll
    for (int base = 0; base < 8; base += 4) {
        const float x0 = values[base], x1 = values[base + 1], x2 = values[base + 2],
                    x3 = values[base + 3];
        values[base] = x0 + x1 + x2 - x3;
        values[base + 1] = x0 + x1 - x2 + x3;
        values[base + 2] = x0 - x1 + x2 + x3;
        values[base + 3] = -x0 + x1 + x2 + x3;
    }
    // Stride 4: even lanes hold x0 and x1 and compute y0 and y1, odd lanes hold x2 and x3 and
    // compute y2 and y3.
    const bool odd_lane = lane & 1;
#pragma unroll
    for (int low = 0; low < 4; low++) {
        const float own_0 = values[low], own_1 = values[low + 4];
        const float other_0 = __shfl_xor_sync(0xffffffff, own_0, 1);
        const float other_1 = __shfl_xor_sync(0xffffffff, own_1, 1);
        const float first_0 = odd_lane ? other_0 - other_1 : own_0 + own_1;
        const float first_1 = odd_lane ? other_1 - other_0 : own_0 + own_1;
        values[low] = first_0 + (odd_lane ? own_0 : other_0) + (odd_lane ? own_1 : -other_1);
        values[low + 4] = first_1 + (odd_lane ? own_0 : -other_0) + (odd_lane ? own_1 : other_1);
    }
    // Strides 16 and 64: the lane with digit j finds x_(j ^ k) at xor distance k << shift. With v_k
    // read from there, y0 = (v0 + v1 + v2) - v3, y1 = (v0 + v1 - v3) + v2, y2 = (v2 - v3 + v0) + v1
    // and y3 = (v2 - v3 + v1) + v0.
#pragma unroll
    for (int shift = 1; shift <= 3; shift += 2) {
        const int digit = (lane >> shift) & 3;
        const bool high = digit & 2;
        const bool odd = digit & 1;
#pragma unroll
        for (int index = 0; index < 8; index++) {
            const float v0 = values[index];
            const float v1 = __shfl_xor_sync(0xffffffff, v0, 1 << shift);
            const float v2 = __shfl_xor_sync(0xffffffff, v0, 2 << shift);
            const float v3 = __shfl_xor_sync(0xffffffff, v0, 3 << shift);
            const float first = high ? v2 - v3 : v0 + v1;
            const float second = high ? (odd ? v1 : v0) : (odd ? -v3 : v2);
            const float third = high ? (odd ? v0 : v1) : (odd ? v2 : -v3);
            values[index] = first + second + third;
        }
    }
#pragma unroll
    for (int index = 0; index < 8; index++) {
        values[index] *= 1.0f / 16.0f;
    }
}

__device__ __forceinline__ float2 to_float2(__nv_bfloat162 pair) {
    return __bfloat1622float2(pair);
}

__device__ __forceinline__ float2 to_float2(__half2 pair) { return __half22float2(pair); }

// Rotates each 256 group of a row of BF16 or FP16 values, then quantizes the row to INT8 with scale
// = max |x| / 127, rounding half to even. The block's warps share the row, which may lie in global
// or shared memory: warp w rotates groups w, w + warps and so on.
template <int GROUPS_PER_WARP, typename Pair>
__device__ __forceinline__ void rotate_quantize_row(const Pair *row, int8_t *output_row,
                                                    float *scale_output, int columns) {
    __shared__ float warp_maxima[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    const int groups = columns / CONVROT_GROUP;

    float values[GROUPS_PER_WARP][8];
    float maximum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < GROUPS_PER_WARP; slot++) {
        const int group = warp + slot * warps;
        if (group < groups) {
            const uint4 packed =
                *reinterpret_cast<const uint4 *>(row + (group * CONVROT_GROUP + lane * 8) / 2);
            const Pair *pairs = reinterpret_cast<const Pair *>(&packed);
#pragma unroll
            for (int pair = 0; pair < 4; pair++) {
                const float2 value = to_float2(pairs[pair]);
                values[slot][pair * 2] = value.x;
                values[slot][pair * 2 + 1] = value.y;
            }
            rotate_group(values[slot], lane);
#pragma unroll
            for (int index = 0; index < 8; index++) {
                maximum = fmaxf(maximum, fabsf(values[slot][index]));
            }
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    if (lane == 0) {
        warp_maxima[warp] = maximum;
    }
    __syncthreads();
    maximum = 0.0f;
    for (int index = 0; index < warps; index++) {
        maximum = fmaxf(maximum, warp_maxima[index]);
    }
    const float scale = fmaxf(maximum / 127.0f, 1e-30f);
    if (threadIdx.x == 0) {
        *scale_output = scale;
    }
#pragma unroll
    for (int slot = 0; slot < GROUPS_PER_WARP; slot++) {
        const int group = warp + slot * warps;
        if (group < groups) {
            uint32_t words[2] = {0, 0};
#pragma unroll
            for (int index = 0; index < 8; index++) {
                const int quantized = static_cast<int>(
                    fminf(fmaxf(rintf(values[slot][index] / scale), -128.0f), 127.0f));
                words[index / 4] |= (static_cast<uint32_t>(quantized) & 0xFF) << (index % 4 * 8);
            }
            *reinterpret_cast<uint2 *>(output_row + group * CONVROT_GROUP + lane * 8) =
                make_uint2(words[0], words[1]);
        }
    }
}

} // namespace
