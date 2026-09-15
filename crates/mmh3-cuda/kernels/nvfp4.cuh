#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "convrot.cuh"

// NVFP4 operands for cuBLASLt's block-scaled FP4 GEMM: FP4 E2M1 values, two per byte with the first
// in the low nibble, one UE4M3 scale per 16 values along K and one FP32 scale per tensor, so that a
// value is q · block scale · tensor scale. The tensor scale maps a reference magnitude to the
// largest block scale. Block scales use cuBLASLt's layout for VEC16_UE4M3: tiles of 128 rows by 4
// blocks, 512 bytes each and in row-tile-major order, where the scale of row r and block c of a
// tile sits at (r % 32) · 16 + (r / 32) · 4 + c. Rows are padded to a multiple of 128 there.
//
// Activations are rotated like ConvRot activations first, so they meet the rotated weights of the
// INT8 ConvRot checkpoints. A warp quantizes a 256-column group, eight values per lane, and each
// pair of lanes shares a block.
//
// A row may have more columns than the layer has inputs. The columns after the inputs hold a
// low-rank adapter, its down projection in the activations and its up projection in the weight,
// unrotated, or zeros.

namespace {

constexpr float NVFP4_VALUE_MAX = 6.0f;
constexpr float NVFP4_SCALE_MAX = 448.0f;

__device__ __forceinline__ size_t nvfp4_scale_offset(int row, int block, int blocks) {
    const size_t tile = static_cast<size_t>(row / 128) * (blocks / 4) + block / 4;
    const int tile_row = row % 128;
    return tile * 512 + (tile_row % 32) * 16 + (tile_row / 32) * 4 + block % 4;
}

__device__ __forceinline__ float nvfp4_tensor_scale(float reference) {
    return reference > 0.0f ? reference / (NVFP4_VALUE_MAX * NVFP4_SCALE_MAX) : 1.0f;
}

// The block scale of a block whose largest magnitude is `maximum`.
__device__ __forceinline__ __nv_fp8_storage_t nvfp4_block_scale(float maximum,
                                                                float inverse_tensor_scale) {
    return __nv_cvt_float_to_fp8(maximum / NVFP4_VALUE_MAX * inverse_tensor_scale, __NV_SATFINITE,
                                 __NV_E4M3);
}

// Packs eight values of a block with block scale `scale` into four bytes.
__device__ __forceinline__ uint32_t nvfp4_pack(const float *values, __nv_fp8_storage_t scale,
                                               float inverse_tensor_scale) {
    const float block_scale = __half2float(__half(__nv_cvt_fp8_to_halfraw(scale, __NV_E4M3)));
    const float factor = block_scale > 0.0f ? inverse_tensor_scale / block_scale : 0.0f;
    uint32_t packed = 0;
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        const __nv_fp4x2_storage_t two = __nv_cvt_float2_to_fp4x2(
            make_float2(values[2 * pair] * factor, values[2 * pair + 1] * factor), __NV_E2M1,
            cudaRoundNearest);
        packed |= static_cast<uint32_t>(two) << (8 * pair);
    }
    return packed;
}

// The row that mmh3_interleave_swiglu_rows moves row `row` of `rows` to.
__device__ __forceinline__ int nvfp4_interleaved_row(int row, int rows) {
    const int half = rows / 2;
    const int feature = row < half ? row : row - half;
    return feature / 4 * 8 + feature % 4 + (row < half ? 0 : 4);
}

// Quantizes the eight values this lane holds of 256-column group `group` of row `row` of a matrix
// with `columns` columns, the block shared with the neighbouring lane, and stores them with their
// block scale.
__device__ __forceinline__ void nvfp4_store_group(const float (&values)[8], int lane, int row,
                                                  int group, int columns,
                                                  float inverse_tensor_scale,
                                                  uint8_t *__restrict__ quantized,
                                                  uint8_t *__restrict__ scales) {
    float maximum = 0.0f;
    #pragma unroll
    for (int index = 0; index < 8; index++) {
        maximum = fmaxf(maximum, fabsf(values[index]));
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    const __nv_fp8_storage_t scale = nvfp4_block_scale(maximum, inverse_tensor_scale);
    const uint32_t packed = nvfp4_pack(values, scale, inverse_tensor_scale);
    *reinterpret_cast<uint32_t *>(quantized + static_cast<size_t>(row) * columns / 2 +
                                  group * CONVROT_GROUP / 2 + lane * 4) = packed;
    if (lane % 2 == 0) {
        scales[nvfp4_scale_offset(row, group * (CONVROT_GROUP / 16) + lane / 2, columns / 16)] =
            scale;
    }
}

// Folds a thread's largest magnitude into `maximum` for the whole block, with an atomic max on the
// float bits, which order like the values for non-negative floats. Every thread of the block must
// call it.
__device__ __forceinline__ void nvfp4_fold_maximum(float local, unsigned *maximum,
                                                   float *warp_maxima) {
    for (int offset = 16; offset > 0; offset /= 2) {
        local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, offset));
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        warp_maxima[warp] = local;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float row_maximum = 0.0f;
        for (int index = 0; index < static_cast<int>(blockDim.x / 32); index++) {
            row_maximum = fmaxf(row_maximum, warp_maxima[index]);
        }
        atomicMax(maximum, __float_as_uint(row_maximum));
    }
}

} // namespace
