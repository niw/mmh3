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

__device__ __forceinline__ float block_max(float value, float* scratch) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float maximum = 0.0f;
    for (int index = 0; index < static_cast<int>(blockDim.x / 32); index++) {
        maximum = fmaxf(maximum, scratch[index]);
    }
    __syncthreads();
    return maximum;
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

// Columns rotated at a time. Wider rows are rotated twice, once for the scale and once for the values.
constexpr int ROTATE_SEGMENT = 16384;

__device__ void rotate_segment(float* row, const __nv_bfloat16* source, int count) {
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        row[index] = __bfloat162float(source[index]);
    }
    __syncthreads();
    const int butterflies = count / 4;
    for (int stride = 1; stride < CONVROT_GROUP; stride *= 4) {
        for (int butterfly = threadIdx.x; butterfly < butterflies; butterfly += blockDim.x) {
            const int first = (butterfly / stride) * (stride * 4) + butterfly % stride;
            const float x0 = row[first];
            const float x1 = row[first + stride];
            const float x2 = row[first + 2 * stride];
            const float x3 = row[first + 3 * stride];
            row[first] = x0 + x1 + x2 - x3;
            row[first + stride] = x0 + x1 - x2 + x3;
            row[first + 2 * stride] = x0 - x1 + x2 + x3;
            row[first + 3 * stride] = -x0 + x1 + x2 + x3;
        }
        __syncthreads();
    }
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        row[index] *= 1.0f / 16.0f;
    }
    __syncthreads();
}

// Rotates each 256-wide group of a row by the normalized regular Hadamard matrix (four radix-4 stages of the
// symmetric 4 × 4 kernel), then quantizes the row to INT8 with scale = max |x| / 127, rounding half to even.
__global__ void __launch_bounds__(ROW_THREADS)
    rotate_quantize_kernel(const __nv_bfloat16* __restrict__ input, int8_t* __restrict__ output,
                           float* __restrict__ scales, int columns) {
    extern __shared__ float row[];
    __shared__ float scratch[ROW_THREADS / 32];
    const int64_t token = blockIdx.x;
    const __nv_bfloat16* source = input + token * columns;
    const int segment = min(columns, ROTATE_SEGMENT);
    float maximum = 0.0f;
    for (int start = 0; start < columns; start += segment) {
        const int count = min(segment, columns - start);
        rotate_segment(row, source + start, count);
        for (int index = threadIdx.x; index < count; index += blockDim.x) {
            maximum = fmaxf(maximum, fabsf(row[index]));
        }
        __syncthreads();
    }
    const float scale = fmaxf(block_max(maximum, scratch) / 127.0f, 1e-30f);
    if (threadIdx.x == 0) {
        scales[token] = scale;
    }
    int8_t* destination = output + token * columns;
    for (int start = 0; start < columns; start += segment) {
        const int count = min(segment, columns - start);
        if (columns > segment) {
            rotate_segment(row, source + start, count);
        }
        for (int index = threadIdx.x; index < count; index += blockDim.x) {
            destination[start + index] = static_cast<int8_t>(fminf(fmaxf(rintf(row[index] / scale), -128.0f), 127.0f));
        }
        __syncthreads();
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
    if (columns % CONVROT_GROUP != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t shared_bytes = static_cast<size_t>(columns < ROTATE_SEGMENT ? columns : ROTATE_SEGMENT) * sizeof(float);
    static size_t configured_bytes = 0;
    if (shared_bytes > configured_bytes) {
        cudaError_t status = cudaFuncSetAttribute(rotate_quantize_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(shared_bytes));
        if (status != cudaSuccess) {
            return static_cast<int>(status);
        }
        configured_bytes = shared_bytes;
    }
    rotate_quantize_kernel<<<tokens, ROW_THREADS, shared_bytes, stream>>>(input, output, scales, columns);
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
