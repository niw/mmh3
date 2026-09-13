#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Elementwise and row-wise kernels of the DiT blocks. The residual stream is FP32 and the inputs of the linear
// layers are BF16. Modulation tables are FP32 `[rows, chunks, hidden]`, and every token picks its row.

namespace {

constexpr int ROW_THREADS = 256;
constexpr int CONVROT_GROUP = 256;
constexpr int HEAD_DIM = 128;

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float total = 0.0f;
    for (int index = 0; index < static_cast<int>(blockDim.x / 32); index++) {
        total += scratch[index];
    }
    __syncthreads();
    return total;
}

__device__ __forceinline__ void store(float* output, float value) {
    *output = value;
}

__device__ __forceinline__ void store(__nv_bfloat16* output, float value) {
    *output = __float2bfloat16_rn(value);
}

// output = rms_norm(input) · weight · (1 + scale) + shift, where shift and scale come from the token's modulation
// row. Without a modulation table it is a plain RMSNorm. Works in place when input and output alias.
template <typename Output>
__global__ void __launch_bounds__(ROW_THREADS)
    rms_norm_modulate_kernel(const float* input, const __nv_bfloat16* __restrict__ weight,
                             const float* __restrict__ modulation, const int32_t* __restrict__ rows, int chunks,
                             int shift_chunk, int scale_chunk, Output* output, int hidden, float epsilon) {
    __shared__ float scratch[ROW_THREADS / 32];
    const int token = blockIdx.x;
    const float* row = input + static_cast<size_t>(token) * hidden;
    float squares = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        squares += row[index] * row[index];
    }
    const float inverse = rsqrtf(block_sum(squares, scratch) / hidden + epsilon);
    const float* shift = nullptr;
    const float* scale = nullptr;
    if (modulation != nullptr) {
        const float* vectors = modulation + static_cast<size_t>(rows[token]) * chunks * hidden;
        shift = vectors + static_cast<size_t>(shift_chunk) * hidden;
        scale = vectors + static_cast<size_t>(scale_chunk) * hidden;
    }
    Output* output_row = output + static_cast<size_t>(token) * hidden;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        float value = row[index] * inverse * __bfloat162float(weight[index]);
        if (modulation != nullptr) {
            value = value * (1.0f + scale[index]) + shift[index];
        }
        store(output_row + index, value);
    }
}

__global__ void gated_residual_add_kernel(float* __restrict__ residual, const __nv_bfloat16* __restrict__ delta,
                                          const float* __restrict__ modulation, const int32_t* __restrict__ rows,
                                          int chunks, int gate_chunk, size_t count, int hidden) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        float value = __bfloat162float(delta[index]);
        if (modulation != nullptr) {
            const size_t token = index / hidden;
            const int column = static_cast<int>(index % hidden);
            value *= modulation[(static_cast<size_t>(rows[token]) * chunks + gate_chunk) * hidden + column];
        }
        residual[index] += value;
    }
}

// Per-head RMSNorm of q and k inside the fused [tokens, 3, heads, 128] qkv buffer, then the split-half rotary
// embedding on the first 2 × pairs dimensions when angles are given. One warp handles one head of q or k.
// Per-head RMSNorm of q and k in rows of [q (query_heads × 128) | k (key_heads × 128) | v (key_heads × 128)], then
// the split-half rotary embedding on the first 2 × pairs dimensions when angles are given.
__global__ void qk_norm_rope_kernel(__nv_bfloat16* qkv, const __nv_bfloat16* __restrict__ query_weight,
                                    const __nv_bfloat16* __restrict__ key_weight, const float* __restrict__ angles,
                                    int pairs, int tokens, int query_heads, int key_heads, float epsilon) {
    __shared__ float values[ROW_THREADS / 32][HEAD_DIM];
    const int warp_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int heads_per_token = query_heads + key_heads;
    const int64_t item = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp_in_block;
    if (item >= static_cast<int64_t>(tokens) * heads_per_token) {
        return;
    }
    const int head = static_cast<int>(item % heads_per_token);
    const int64_t token = item / heads_per_token;
    const bool is_key = head >= query_heads;
    __nv_bfloat16* vector = qkv + token * (query_heads + 2 * key_heads) * HEAD_DIM + head * HEAD_DIM;
    const __nv_bfloat16* weight = is_key ? key_weight : query_weight;
    float* shared = values[warp_in_block];

    float squares = 0.0f;
    for (int index = lane; index < HEAD_DIM; index += 32) {
        float value = __bfloat162float(vector[index]);
        shared[index] = value;
        squares += value * value;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        squares += __shfl_xor_sync(0xffffffff, squares, offset);
    }
    const float inverse = rsqrtf(squares / HEAD_DIM + epsilon);
    for (int index = lane; index < HEAD_DIM; index += 32) {
        shared[index] *= inverse * __bfloat162float(weight[index]);
    }
    __syncwarp();
    if (angles != nullptr) {
        const float* token_angles = angles + token * pairs;
        for (int pair = lane; pair < pairs; pair += 32) {
            float sine, cosine;
            sincosf(token_angles[pair], &sine, &cosine);
            const float first = shared[pair];
            const float second = shared[pair + pairs];
            shared[pair] = first * cosine - second * sine;
            shared[pair + pairs] = first * sine + second * cosine;
        }
        __syncwarp();
    }
    for (int index = lane; index < HEAD_DIM; index += 32) {
        vector[index] = __float2bfloat16_rn(shared[index]);
    }
}

__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output,
                              size_t count, int ffn) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        const size_t token = index / ffn;
        const int column = static_cast<int>(index % ffn);
        const float gate = __bfloat162float(input[token * 2 * ffn + column]);
        const float up = __bfloat162float(input[token * 2 * ffn + ffn + column]);
        output[index] = __float2bfloat16_rn(gate / (1.0f + __expf(-gate)) * up);
    }
}

// Groups of 256 columns a warp rotates at most, so rows up to 32 warps × 4 × 256 = 32,768 columns fit in registers.
constexpr int MAX_GROUPS_PER_WARP = 4;

// Rotates a 256 group by the normalized regular Hadamard matrix: four radix-4 stages with strides 1, 4, 16 and 64 of
// the symmetric 4 × 4 kernel y0 = x0 + x1 + x2 - x3, y1 = x0 + x1 - x2 + x3, y2 = x0 - x1 + x2 + x3,
// y3 = -x0 + x1 + x2 + x3, then a factor of 1/16. Lane l holds elements 8l to 8l + 7. Of the index bits, bits 0-1
// (stride 1) are in the register index, bits 2-3 (stride 4) are register bit 2 and lane bit 0, bits 4-5 (stride 16)
// lane bits 1-2 and bits 6-7 (stride 64) lane bits 3-4.
//
// NOTE: Every output keeps the order of additions written above, so the result is the same in every bit whichever
// lane computes it. The rewritten forms below only use a + b == b + a and a + (-b) == a - b, which hold exactly.
__device__ __forceinline__ void rotate_group(float (&values)[8], int lane) {
#pragma unroll
    for (int base = 0; base < 8; base += 4) {
        const float x0 = values[base], x1 = values[base + 1], x2 = values[base + 2], x3 = values[base + 3];
        values[base] = x0 + x1 + x2 - x3;
        values[base + 1] = x0 + x1 - x2 + x3;
        values[base + 2] = x0 - x1 + x2 + x3;
        values[base + 3] = -x0 + x1 + x2 + x3;
    }
    // Stride 4: even lanes hold x0 and x1 and compute y0 and y1, odd lanes hold x2 and x3 and compute y2 and y3.
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
    // Strides 16 and 64: the lane with digit j finds x_(j ^ k) at xor distance k << shift. With v_k read from there,
    // y0 = (v0 + v1 + v2) - v3, y1 = (v0 + v1 - v3) + v2, y2 = (v2 - v3 + v0) + v1 and y3 = (v2 - v3 + v1) + v0.
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

// Rotates each 256 group of a row, then quantizes the row to INT8 with scale = max |x| / 127, rounding half to even.
// One block handles one row, and warp w rotates groups w, w + warps and so on.
template <int GROUPS_PER_WARP>
__global__ void rotate_quantize_kernel(const __nv_bfloat16* __restrict__ input, int8_t* __restrict__ output,
                                       float* __restrict__ scales, int columns) {
    __shared__ float warp_maxima[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    const int groups = columns / CONVROT_GROUP;
    const size_t row_offset = static_cast<size_t>(blockIdx.x) * columns;

    float values[GROUPS_PER_WARP][8];
    float maximum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < GROUPS_PER_WARP; slot++) {
        const int group = warp + slot * warps;
        if (group < groups) {
            const uint4 packed = *reinterpret_cast<const uint4*>(input + row_offset + group * CONVROT_GROUP + lane * 8);
            const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
            for (int pair = 0; pair < 4; pair++) {
                const float2 value = __bfloat1622float2(pairs[pair]);
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
        scales[blockIdx.x] = scale;
    }
#pragma unroll
    for (int slot = 0; slot < GROUPS_PER_WARP; slot++) {
        const int group = warp + slot * warps;
        if (group < groups) {
            uint32_t words[2] = {0, 0};
#pragma unroll
            for (int index = 0; index < 8; index++) {
                const int quantized = static_cast<int>(fminf(fmaxf(rintf(values[slot][index] / scale), -128.0f), 127.0f));
                words[index / 4] |= (static_cast<uint32_t>(quantized) & 0xFF) << (index % 4 * 8);
            }
            *reinterpret_cast<uint2*>(output + row_offset + group * CONVROT_GROUP + lane * 8) =
                make_uint2(words[0], words[1]);
        }
    }
}

// output[step, o] = Σ_r time_embedding[step, r] · weight[o, r] + bias[o], with FP16 weights.
__global__ void modulation_kernel(const float* __restrict__ time_embedding, const __half* __restrict__ weight,
                                  const __half* __restrict__ bias, float* __restrict__ output, int steps, int rank,
                                  int outputs) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= outputs) {
        return;
    }
    for (int step = 0; step < steps; step++) {
        float value = __half2float(bias[column]);
        for (int index = 0; index < rank; index++) {
            value += time_embedding[step * rank + index] * __half2float(weight[static_cast<size_t>(column) * rank + index]);
        }
        output[static_cast<size_t>(step) * outputs + column] = value;
    }
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ input, float* __restrict__ output, size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        output[index] = __bfloat162float(input[index]);
    }
}

unsigned grid_for(size_t count) {
    size_t blocks = (count + ROW_THREADS - 1) / ROW_THREADS;
    return static_cast<unsigned>(blocks < 8192 ? blocks : 8192);
}

}  // namespace

extern "C" int mmh3_rms_norm_modulate(const float* input, const __nv_bfloat16* weight, const float* modulation,
                                      const int32_t* rows, int chunks, int shift_chunk, int scale_chunk, void* output,
                                      int output_is_f32, int tokens, int hidden, float epsilon, cudaStream_t stream) {
    if (output_is_f32) {
        rms_norm_modulate_kernel<float><<<tokens, ROW_THREADS, 0, stream>>>(
            input, weight, modulation, rows, chunks, shift_chunk, scale_chunk, static_cast<float*>(output), hidden,
            epsilon);
    } else {
        rms_norm_modulate_kernel<__nv_bfloat16><<<tokens, ROW_THREADS, 0, stream>>>(
            input, weight, modulation, rows, chunks, shift_chunk, scale_chunk, static_cast<__nv_bfloat16*>(output),
            hidden, epsilon);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_gated_residual_add(float* residual, const __nv_bfloat16* delta, const float* modulation,
                                       const int32_t* rows, int chunks, int gate_chunk, int tokens, int hidden,
                                       cudaStream_t stream) {
    const size_t count = static_cast<size_t>(tokens) * hidden;
    gated_residual_add_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(residual, delta, modulation, rows, chunks,
                                                                           gate_chunk, count, hidden);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_qk_norm_rope(__nv_bfloat16* qkv, const __nv_bfloat16* query_weight,
                                 const __nv_bfloat16* key_weight, const float* angles, int pairs, int tokens,
                                 int query_heads, int key_heads, float epsilon, cudaStream_t stream) {
    const int64_t items = static_cast<int64_t>(tokens) * (query_heads + key_heads);
    const int warps_per_block = ROW_THREADS / 32;
    const unsigned blocks = static_cast<unsigned>((items + warps_per_block - 1) / warps_per_block);
    qk_norm_rope_kernel<<<blocks, ROW_THREADS, 0, stream>>>(qkv, query_weight, key_weight, angles, pairs, tokens,
                                                            query_heads, key_heads, epsilon);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_swiglu(const __nv_bfloat16* input, __nv_bfloat16* output, int tokens, int ffn, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(tokens) * ffn;
    swiglu_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(input, output, count, ffn);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_rotate_quantize(const __nv_bfloat16* input, int8_t* output, float* scales, int tokens, int columns,
                                    cudaStream_t stream) {
    const int groups = columns / CONVROT_GROUP;
    const int groups_per_warp = (groups + 31) / 32;
    if (columns % CONVROT_GROUP != 0 || groups == 0 || groups_per_warp > MAX_GROUPS_PER_WARP) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int threads = (groups + groups_per_warp - 1) / groups_per_warp * 32;
    switch (groups_per_warp) {
        case 1:
            rotate_quantize_kernel<1><<<tokens, threads, 0, stream>>>(input, output, scales, columns);
            break;
        case 2:
            rotate_quantize_kernel<2><<<tokens, threads, 0, stream>>>(input, output, scales, columns);
            break;
        case 3:
            rotate_quantize_kernel<3><<<tokens, threads, 0, stream>>>(input, output, scales, columns);
            break;
        default:
            rotate_quantize_kernel<4><<<tokens, threads, 0, stream>>>(input, output, scales, columns);
            break;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_modulation(const float* time_embedding, const __half* weight, const __half* bias, float* output,
                               int steps, int rank, int outputs, cudaStream_t stream) {
    modulation_kernel<<<(outputs + ROW_THREADS - 1) / ROW_THREADS, ROW_THREADS, 0, stream>>>(
        time_embedding, weight, bias, output, steps, rank, outputs);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_bf16_to_f32(const __nv_bfloat16* input, float* output, size_t count, cudaStream_t stream) {
    bf16_to_f32_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(input, output, count);
    return static_cast<int>(cudaGetLastError());
}
