#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "attention_workspace.cuh"
#include "convrot.cuh"

// Elementwise and row-wise kernels of the DiT blocks. The residual stream is FP32 and the inputs of
// the linear layers are BF16. Modulation tables are FP32 `[rows, chunks, hidden]`, and every token
// picks its row.

namespace {

constexpr int ROW_THREADS = 256;
constexpr int HEAD_DIM = 128;

__device__ __forceinline__ float block_sum(float value, float *scratch) {
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

__device__ __forceinline__ void store(float *output, float value) { *output = value; }

__device__ __forceinline__ void store(__nv_bfloat16 *output, float value) {
    *output = __float2bfloat16_rn(value);
}

// output = rms_norm(input) · weight · (1 + scale) + shift, where shift and scale come from the
// token's modulation row. Without a modulation table it is a plain RMSNorm. Works in place when
// input and output alias.
template <typename Output>
__global__ void __launch_bounds__(ROW_THREADS)
    rms_norm_modulate_kernel(const float *input, const __nv_bfloat16 *__restrict__ weight,
                             const float *__restrict__ modulation, const int32_t *__restrict__ rows,
                             int chunks, int shift_chunk, int scale_chunk, Output *output,
                             int hidden, float epsilon) {
    __shared__ float scratch[ROW_THREADS / 32];
    const int token = blockIdx.x;
    const float *row = input + static_cast<size_t>(token) * hidden;
    float squares = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        squares += row[index] * row[index];
    }
    const float inverse = rsqrtf(block_sum(squares, scratch) / hidden + epsilon);
    const float *shift = nullptr;
    const float *scale = nullptr;
    if (modulation != nullptr) {
        const float *vectors = modulation + static_cast<size_t>(rows[token]) * chunks * hidden;
        shift = vectors + static_cast<size_t>(shift_chunk) * hidden;
        scale = vectors + static_cast<size_t>(scale_chunk) * hidden;
    }
    Output *output_row = output + static_cast<size_t>(token) * hidden;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        float value = row[index] * inverse * __bfloat162float(weight[index]);
        if (modulation != nullptr) {
            value = value * (1.0f + scale[index]) + shift[index];
        }
        store(output_row + index, value);
    }
}

__global__ void gated_residual_add_kernel(float *__restrict__ residual,
                                          const __nv_bfloat16 *__restrict__ delta,
                                          const float *__restrict__ modulation,
                                          const int32_t *__restrict__ rows, int chunks,
                                          int gate_chunk, size_t count, int hidden) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        float value = __bfloat162float(delta[index]);
        if (modulation != nullptr) {
            const size_t token = index / hidden;
            const int column = static_cast<int>(index % hidden);
            value *= modulation[(static_cast<size_t>(rows[token]) * chunks + gate_chunk) * hidden +
                                column];
        }
        residual[index] += value;
    }
}

// RMSNorm of one head of 128 values from `source` with its weight, then the split-half rotary
// embedding of the first 2 × pairs dimensions when angles are given. One warp works on the head,
// which it leaves in FP32 in `values`.
//
// NOTE: the _rn intrinsics spell out the multiply-adds that the compiler forms from the plain
// expressions, so that every kernel normalizing heads rounds the same way.
__device__ __forceinline__ void normalize_rotate_head(const __nv_bfloat16 *source, float *values,
                                                      const __nv_bfloat16 *__restrict__ weight,
                                                      const float *__restrict__ token_angles,
                                                      int pairs, float epsilon) {
    const int lane = threadIdx.x % 32;
    float squares = 0.0f;
    for (int index = lane; index < HEAD_DIM; index += 32) {
        const float value = __bfloat162float(source[index]);
        values[index] = value;
        squares = __fmaf_rn(value, value, squares);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        squares += __shfl_xor_sync(0xffffffff, squares, offset);
    }
    const float inverse = rsqrtf(__fmaf_rn(squares, 1.0f / HEAD_DIM, epsilon));
    for (int index = lane; index < HEAD_DIM; index += 32) {
        values[index] =
            __fmul_rn(values[index], __fmul_rn(inverse, __bfloat162float(weight[index])));
    }
    __syncwarp();
    if (token_angles != nullptr) {
        for (int pair = lane; pair < pairs; pair += 32) {
            float sine, cosine;
            sincosf(token_angles[pair], &sine, &cosine);
            const float first = values[pair];
            const float second = values[pair + pairs];
            values[pair] = __fmaf_rn(cosine, first, -__fmul_rn(sine, second));
            values[pair + pairs] = __fmaf_rn(sine, first, __fmul_rn(cosine, second));
        }
        __syncwarp();
    }
}

// Per-head RMSNorm of q and k in rows of [q (query_heads × 128) | k (key_heads × 128) | v
// (key_heads × 128)], then the split-half rotary embedding on the first 2 × pairs dimensions when
// angles are given. One warp handles one head of q or k.
__global__ void qk_norm_rope_kernel(__nv_bfloat16 *qkv,
                                    const __nv_bfloat16 *__restrict__ query_weight,
                                    const __nv_bfloat16 *__restrict__ key_weight,
                                    const float *__restrict__ angles, int pairs, int tokens,
                                    int query_heads, int key_heads, float epsilon) {
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
    __nv_bfloat16 *vector =
        qkv + token * (query_heads + 2 * key_heads) * HEAD_DIM + head * HEAD_DIM;
    float *shared = values[warp_in_block];
    normalize_rotate_head(vector, shared, is_key ? key_weight : query_weight,
                          angles != nullptr ? angles + token * pairs : nullptr, pairs, epsilon);
    for (int index = lane; index < HEAD_DIM; index += 32) {
        vector[index] = __float2bfloat16_rn(shared[index]);
    }
}

constexpr int INPUT_BLOCK = 64;
constexpr int INPUT_THREADS = 2 * HEAD_DIM;
constexpr int INPUT_WARPS = INPUT_THREADS / 32;

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    return value;
}

// qk_norm_rope_kernel for one 64-token block and one head of the fused qkv rows of `heads` heads,
// together with what the attention needs from that block:
//
// - STATS: Sol-Attn's block statistics, the mean query, the mean key and the summed value, as
// block_stats_kernel in
//   sparse_attention.cu computes them.
// - QUANTIZE: INT8 q and k with their block scales, and the block maximum of |v|, as quantize_qk
// and value_max in
//   quantized_attention.cu compute them.
//
// Normalized q goes back to qkv in BF16 when the attention or the Sol-Attn row offsets read it, and
// normalized k when the attention reads BF16 keys. q and k take turns in one tile, and the rows of
// k load while q is normalized. The register limit fits four CTAs on an SM, so that the loads of
// some overlap the arithmetic of others.
template <bool STATS, bool QUANTIZE>
__global__ void __launch_bounds__(INPUT_THREADS, 4)
    attention_inputs_kernel(__nv_bfloat16 *__restrict__ qkv,
                            const __nv_bfloat16 *__restrict__ query_weight,
                            const __nv_bfloat16 *__restrict__ key_weight,
                            const float *__restrict__ angles, int pairs, int tokens, int heads,
                            float epsilon, Mmh3SparseWorkspace sparse,
                            Mmh3QuantizedWorkspace quantized) {
    // A row of 128 BF16 values is 16 chunks of 16 bytes.
    constexpr int CHUNKS_PER_THREAD = INPUT_BLOCK * 16 / INPUT_THREADS;
    __shared__ __align__(16) __nv_bfloat16 tile[INPUT_BLOCK][HEAD_DIM];
    __shared__ float head_values[INPUT_WARPS][HEAD_DIM];
    __shared__ float maxima[2][HEAD_DIM / 32];
    const int block = blockIdx.x;
    const int head = blockIdx.y;
    const int blocks = gridDim.x;
    const int rows = min(INPUT_BLOCK, tokens - block * INPUT_BLOCK);
    const int inner = heads * HEAD_DIM;
    const int64_t row_stride = 3 * static_cast<int64_t>(inner);
    const int64_t first_token = static_cast<int64_t>(block) * INPUT_BLOCK;
    __nv_bfloat16 *block_rows = qkv + first_token * row_stride + head * HEAD_DIM;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const size_t statistic = (static_cast<size_t>(head) * blocks + block) * HEAD_DIM;

    uint4 chunks[CHUNKS_PER_THREAD];
    auto load = [&](int part) {
        #pragma unroll
        for (int index = 0; index < CHUNKS_PER_THREAD; index++) {
            const int chunk = threadIdx.x + index * INPUT_THREADS;
            if (chunk / 16 < rows) {
                chunks[index] = *reinterpret_cast<const uint4 *>(
                    block_rows + (chunk / 16) * row_stride + part * inner + (chunk % 16) * 8);
            }
        }
    };
    load(0);
    for (int part = 0; part < 2; part++) {
        #pragma unroll
        for (int index = 0; index < CHUNKS_PER_THREAD; index++) {
            const int chunk = threadIdx.x + index * INPUT_THREADS;
            if (chunk / 16 < rows) {
                *reinterpret_cast<uint4 *>(&tile[chunk / 16][(chunk % 16) * 8]) = chunks[index];
            }
        }
        __syncthreads();
        if (part == 0) {
            load(1);
        }
        for (int row = warp; row < rows; row += INPUT_WARPS) {
            normalize_rotate_head(
                tile[row], head_values[warp], part ? key_weight : query_weight,
                angles != nullptr ? angles + (first_token + row) * pairs : nullptr, pairs, epsilon);
            for (int index = lane; index < HEAD_DIM; index += 32) {
                tile[row][index] = __float2bfloat16_rn(head_values[warp][index]);
            }
            __syncwarp();
        }
        __syncthreads();

        // Sums and maxima over the rows of the block, one thread per dimension, and for v during
        // the turn of q.
        if (threadIdx.x < HEAD_DIM) {
            const int dimension = threadIdx.x;
            float sum = 0.0f, maximum = 0.0f;
            for (int row = 0; row < rows; row++) {
                const float value = __bfloat162float(tile[row][dimension]);
                sum += value;
                maximum = fmaxf(maximum, fabsf(value));
            }
            if constexpr (STATS) {
                (part ? sparse.block_keys : sparse.centroids)[statistic + dimension] = sum / rows;
            }
            maximum = warp_max(maximum);
            if (lane == 0) {
                maxima[0][warp] = maximum;
            }
        } else if (part == 0) {
            const int dimension = threadIdx.x - HEAD_DIM;
            const __nv_bfloat16 *value = block_rows + 2 * inner + dimension;
            float sum = 0.0f, maximum = 0.0f;
            #pragma unroll 8
            for (int row = 0; row < rows; row++) {
                const float element = __bfloat162float(value[row * row_stride]);
                sum += element;
                maximum = fmaxf(maximum, fabsf(element));
            }
            if constexpr (STATS) {
                sparse.value_sums[statistic + dimension] = sum;
            }
            maximum = warp_max(maximum);
            if (lane == 0) {
                maxima[1][warp - HEAD_DIM / 32] = maximum;
            }
        }

        if constexpr (QUANTIZE) {
            __syncthreads();
            float maximum = maxima[0][0];
            for (int index = 1; index < HEAD_DIM / 32; index++) {
                maximum = fmaxf(maximum, maxima[0][index]);
            }
            const float scale = fmaxf(maximum / 127.0f, 1e-30f);
            if (threadIdx.x == 0) {
                (part ? quantized.key_scales : quantized.query_scales)[head * blocks + block] =
                    scale;
                if (part == 0) {
                    float value_max = maxima[1][0];
                    for (int index = 1; index < HEAD_DIM / 32; index++) {
                        value_max = fmaxf(value_max, maxima[1][index]);
                    }
                    quantized.value_maxima[head * blocks + block] = value_max;
                }
            }
            // A row of 128 INT8 values is 8 chunks of 16 bytes, each from 16 BF16 values.
            uint8_t *destination =
                (part ? quantized.key : quantized.query) + (first_token * heads + head) * HEAD_DIM;
            for (int chunk = threadIdx.x; chunk < rows * 8; chunk += INPUT_THREADS) {
                const uint4 *source =
                    reinterpret_cast<const uint4 *>(&tile[chunk / 8][(chunk % 8) * 16]);
                const uint4 halves[2] = {source[0], source[1]};
                const __nv_bfloat16 *values = reinterpret_cast<const __nv_bfloat16 *>(halves);
                uint32_t words[4];
                #pragma unroll
                for (int word = 0; word < 4; word++) {
                    uint32_t packed = 0;
                    #pragma unroll
                    for (int byte = 0; byte < 4; byte++) {
                        const int8_t value = static_cast<int8_t>(
                            __float2int_rn(__bfloat162float(values[word * 4 + byte]) / scale));
                        packed |= static_cast<uint32_t>(static_cast<uint8_t>(value)) << (8 * byte);
                    }
                    words[word] = packed;
                }
                *reinterpret_cast<uint4 *>(
                    destination + static_cast<int64_t>(chunk / 8) * heads * HEAD_DIM +
                    (chunk % 8) * 16) = make_uint4(words[0], words[1], words[2], words[3]);
            }
        }

        if (part == 0 ? STATS || !QUANTIZE : !QUANTIZE) {
            for (int chunk = threadIdx.x; chunk < rows * 16; chunk += INPUT_THREADS) {
                *reinterpret_cast<uint4 *>(block_rows + (chunk / 16) * row_stride + part * inner +
                                           (chunk % 16) * 8) =
                    *reinterpret_cast<const uint4 *>(&tile[chunk / 16][(chunk % 16) * 8]);
            }
        }
        __syncthreads();
    }
}

__global__ void swiglu_kernel(const __nv_bfloat16 *__restrict__ input,
                              __nv_bfloat16 *__restrict__ output, size_t count, int ffn) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const size_t token = index / ffn;
        const int column = static_cast<int>(index % ffn);
        const float gate = __bfloat162float(input[token * 2 * ffn + column]);
        const float up = __bfloat162float(input[token * 2 * ffn + ffn + column]);
        output[index] = __float2bfloat16_rn(gate / (1.0f + __expf(-gate)) * up);
    }
}

// One block per row.
template <int GROUPS_PER_WARP, typename Pair>
__global__ void rotate_quantize_kernel(const Pair *__restrict__ input, int8_t *__restrict__ output,
                                       float *__restrict__ scales, int columns) {
    const size_t row_offset = static_cast<size_t>(blockIdx.x) * columns;
    rotate_quantize_row<GROUPS_PER_WARP>(input + row_offset / 2, output + row_offset,
                                         scales + blockIdx.x, columns);
}

// For each token: residual += gate ⊙ delta when a delta is given, with the gate from
// `gate_modulation`, then rms_norm(residual) ⊙ weight ⊙ (1 + scale) + shift in BF16 with shift and
// scale from `modulation`, kept in `normalized` when it is not null, and its ConvRot rotation
// quantized to INT8 with the row's scale. Both modulation tables share the token's row.
//
// NOTE: The result matches gated_residual_add_kernel, rms_norm_modulate_kernel and
// rotate_quantize_kernel run one after the other in every bit. The compiler fuses the residual
// update, the sum of squares and the modulation of those kernels into multiply-adds, which the _rn
// intrinsics spell out here, and the normalized row reaches the rotation through shared memory in
// BF16.
template <int GROUPS_PER_WARP>
__global__ void __launch_bounds__(ROW_THREADS)
    add_norm_quantize_kernel(float *__restrict__ residual, const __nv_bfloat16 *__restrict__ delta,
                             const float *__restrict__ gate_modulation,
                             const float *__restrict__ modulation, const int32_t *__restrict__ rows,
                             int chunks, int gate_chunk, int shift_chunk, int scale_chunk,
                             const __nv_bfloat16 *__restrict__ weight,
                             __nv_bfloat16 *__restrict__ normalized, int8_t *__restrict__ quantized,
                             float *__restrict__ scales, int hidden, float epsilon) {
    extern __shared__ __align__(16) __nv_bfloat16 normalized_row[];
    __shared__ float scratch[ROW_THREADS / 32];
    const int64_t token = blockIdx.x;
    float *row = residual + token * hidden;
    const float *vectors = modulation != nullptr
                               ? modulation + static_cast<size_t>(rows[token]) * chunks * hidden
                               : nullptr;
    const float *gate_vectors =
        gate_modulation != nullptr
            ? gate_modulation + static_cast<size_t>(rows[token]) * chunks * hidden
            : nullptr;

    float squares = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        float value = row[index];
        if (delta != nullptr) {
            const float change = __bfloat162float(delta[token * hidden + index]);
            value = gate_vectors != nullptr
                        ? __fmaf_rn(change,
                                    gate_vectors[static_cast<size_t>(gate_chunk) * hidden + index],
                                    value)
                        : __fadd_rn(value, change);
            row[index] = value;
        }
        squares = __fmaf_rn(value, value, squares);
    }
    const float inverse = rsqrtf(block_sum(squares, scratch) / hidden + epsilon);
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        float value = __fmul_rn(__fmul_rn(row[index], inverse), __bfloat162float(weight[index]));
        if (vectors != nullptr) {
            value = __fmaf_rn(
                value, __fadd_rn(1.0f, vectors[static_cast<size_t>(scale_chunk) * hidden + index]),
                vectors[static_cast<size_t>(shift_chunk) * hidden + index]);
        }
        const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
        normalized_row[index] = rounded;
        if (normalized != nullptr) {
            normalized[token * hidden + index] = rounded;
        }
    }
    __syncthreads();
    rotate_quantize_row<GROUPS_PER_WARP>(reinterpret_cast<const __nv_bfloat162 *>(normalized_row),
                                         quantized + token * hidden, scales + token, hidden);
}

// output[step, o] = Σ_r time_embedding[step, r] · weight[o, r] + bias[o], with FP16 weights.
__global__ void modulation_kernel(const float *__restrict__ time_embedding,
                                  const __half *__restrict__ weight,
                                  const __half *__restrict__ bias, float *__restrict__ output,
                                  int steps, int rank, int outputs) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= outputs) {
        return;
    }
    for (int step = 0; step < steps; step++) {
        float value = __half2float(bias[column]);
        for (int index = 0; index < rank; index++) {
            value += time_embedding[step * rank + index] *
                     __half2float(weight[static_cast<size_t>(column) * rank + index]);
        }
        output[static_cast<size_t>(step) * outputs + column] = value;
    }
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16 *__restrict__ input,
                                   float *__restrict__ output, size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        output[index] = __bfloat162float(input[index]);
    }
}

unsigned grid_for(size_t count) {
    size_t blocks = (count + ROW_THREADS - 1) / ROW_THREADS;
    return static_cast<unsigned>(blocks < 8192 ? blocks : 8192);
}

} // namespace

extern "C" int mmh3_rms_norm_modulate(const float *input, const __nv_bfloat16 *weight,
                                      const float *modulation, const int32_t *rows, int chunks,
                                      int shift_chunk, int scale_chunk, void *output,
                                      int output_is_f32, int tokens, int hidden, float epsilon,
                                      cudaStream_t stream) {
    if (output_is_f32) {
        rms_norm_modulate_kernel<float><<<tokens, ROW_THREADS, 0, stream>>>(
            input, weight, modulation, rows, chunks, shift_chunk, scale_chunk,
            static_cast<float *>(output), hidden, epsilon);
    } else {
        rms_norm_modulate_kernel<__nv_bfloat16><<<tokens, ROW_THREADS, 0, stream>>>(
            input, weight, modulation, rows, chunks, shift_chunk, scale_chunk,
            static_cast<__nv_bfloat16 *>(output), hidden, epsilon);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_gated_residual_add(float *residual, const __nv_bfloat16 *delta,
                                       const float *modulation, const int32_t *rows, int chunks,
                                       int gate_chunk, int tokens, int hidden,
                                       cudaStream_t stream) {
    const size_t count = static_cast<size_t>(tokens) * hidden;
    gated_residual_add_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(
        residual, delta, modulation, rows, chunks, gate_chunk, count, hidden);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_qk_norm_rope(__nv_bfloat16 *qkv, const __nv_bfloat16 *query_weight,
                                 const __nv_bfloat16 *key_weight, const float *angles, int pairs,
                                 int tokens, int query_heads, int key_heads, float epsilon,
                                 cudaStream_t stream) {
    const int64_t items = static_cast<int64_t>(tokens) * (query_heads + key_heads);
    const int warps_per_block = ROW_THREADS / 32;
    const unsigned blocks = static_cast<unsigned>((items + warps_per_block - 1) / warps_per_block);
    qk_norm_rope_kernel<<<blocks, ROW_THREADS, 0, stream>>>(
        qkv, query_weight, key_weight, angles, pairs, tokens, query_heads, key_heads, epsilon);
    return static_cast<int>(cudaGetLastError());
}

// mmh3_qk_norm_rope over rows of heads × (q, k, v) fused with the attention's per-block work: the
// Sol-Attn block statistics when `sparse` is given and the INT8 q and k with the |v| block maxima
// when `quantized` is given. The attention that follows skips those steps.
extern "C" int mmh3_attention_inputs(__nv_bfloat16 *qkv, const __nv_bfloat16 *query_weight,
                                     const __nv_bfloat16 *key_weight, const float *angles,
                                     int pairs, int tokens, int heads, float epsilon,
                                     const Mmh3SparseWorkspace *sparse,
                                     const Mmh3QuantizedWorkspace *quantized, cudaStream_t stream) {
    if (tokens <= 0 || heads <= 0 || (sparse == nullptr && quantized == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const dim3 grid((tokens + INPUT_BLOCK - 1) / INPUT_BLOCK, heads);
    const Mmh3SparseWorkspace sparse_workspace =
        sparse != nullptr ? *sparse : Mmh3SparseWorkspace{};
    const Mmh3QuantizedWorkspace quantized_workspace =
        quantized != nullptr ? *quantized : Mmh3QuantizedWorkspace{};
    auto launch = [&](auto kernel) {
        kernel<<<grid, INPUT_THREADS, 0, stream>>>(qkv, query_weight, key_weight, angles, pairs,
                                                   tokens, heads, epsilon, sparse_workspace,
                                                   quantized_workspace);
    };
    if (sparse != nullptr && quantized != nullptr) {
        launch(attention_inputs_kernel<true, true>);
    } else if (sparse != nullptr) {
        launch(attention_inputs_kernel<true, false>);
    } else {
        launch(attention_inputs_kernel<false, true>);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_swiglu(const __nv_bfloat16 *input, __nv_bfloat16 *output, int tokens, int ffn,
                           cudaStream_t stream) {
    const size_t count = static_cast<size_t>(tokens) * ffn;
    swiglu_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(input, output, count, ffn);
    return static_cast<int>(cudaGetLastError());
}

namespace {

template <typename Pair>
int launch_rotate_quantize(const Pair *input, int8_t *output, float *scales, int tokens,
                           int columns, cudaStream_t stream) {
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

} // namespace

extern "C" int mmh3_rotate_quantize(const void *input, int input_is_f16, int8_t *output,
                                    float *scales, int tokens, int columns, cudaStream_t stream) {
    if (input_is_f16) {
        return launch_rotate_quantize(static_cast<const __half2 *>(input), output, scales, tokens,
                                      columns, stream);
    }
    return launch_rotate_quantize(static_cast<const __nv_bfloat162 *>(input), output, scales,
                                  tokens, columns, stream);
}

extern "C" int mmh3_modulation(const float *time_embedding, const __half *weight,
                               const __half *bias, float *output, int steps, int rank, int outputs,
                               cudaStream_t stream) {
    modulation_kernel<<<(outputs + ROW_THREADS - 1) / ROW_THREADS, ROW_THREADS, 0, stream>>>(
        time_embedding, weight, bias, output, steps, rank, outputs);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_bf16_to_f32(const __nv_bfloat16 *input, float *output, size_t count,
                                cudaStream_t stream) {
    bf16_to_f32_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(input, output, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_add_norm_quantize(float *residual, const __nv_bfloat16 *delta,
                                      const float *gate_modulation, const float *modulation,
                                      const int32_t *rows, int chunks, int gate_chunk,
                                      int shift_chunk, int scale_chunk, const __nv_bfloat16 *weight,
                                      __nv_bfloat16 *normalized, int8_t *quantized, float *scales,
                                      int tokens, int hidden, float epsilon, cudaStream_t stream) {
    const int groups = hidden / CONVROT_GROUP;
    const int groups_per_warp = (groups + ROW_THREADS / 32 - 1) / (ROW_THREADS / 32);
    if (hidden % CONVROT_GROUP != 0 || groups == 0 || groups_per_warp > MAX_GROUPS_PER_WARP) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t shared_bytes = static_cast<size_t>(hidden) * sizeof(__nv_bfloat16);
    auto launch = [&](auto kernel) {
        kernel<<<tokens, ROW_THREADS, shared_bytes, stream>>>(
            residual, delta, gate_modulation, modulation, rows, chunks, gate_chunk, shift_chunk,
            scale_chunk, weight, normalized, quantized, scales, hidden, epsilon);
    };
    switch (groups_per_warp) {
    case 1:
        launch(add_norm_quantize_kernel<1>);
        break;
    case 2:
        launch(add_norm_quantize_kernel<2>);
        break;
    case 3:
        launch(add_norm_quantize_kernel<3>);
        break;
    default:
        launch(add_norm_quantize_kernel<4>);
        break;
    }
    return static_cast<int>(cudaGetLastError());
}
