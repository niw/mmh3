#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Kernels of the Qwen3-VL vision tower, which runs in FP32 on the patches of a picture: layer
// norms, GELU, the 2D rotary embedding and the softmax of its attention. The GEMMs run through
// cuBLASLt.

namespace {

constexpr int THREADS = 256;

unsigned grid_for(size_t count) {
    size_t blocks = (count + THREADS - 1) / THREADS;
    return static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
}

__device__ __forceinline__ float block_sum(float value, float *scratch) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __syncthreads();
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float total = 0.0f;
    for (int index = 0; index < static_cast<int>(blockDim.x / 32); index++) {
        total += scratch[index];
    }
    return total;
}

__device__ __forceinline__ float block_max(float value, float *scratch) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __syncthreads();
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    float total = scratch[0];
    for (int index = 1; index < static_cast<int>(blockDim.x / 32); index++) {
        total = fmaxf(total, scratch[index]);
    }
    return total;
}

// One block per row: output = (input − mean) / sqrt(variance + epsilon) · weight + bias.
__global__ void __launch_bounds__(THREADS)
    layer_norm_kernel(const float *__restrict__ input, const float *__restrict__ weight,
                      const float *__restrict__ bias, float epsilon, int width,
                      float *__restrict__ output) {
    __shared__ float scratch[THREADS / 32];
    const float *row = input + static_cast<size_t>(blockIdx.x) * width;
    float sum = 0.0f;
    for (int index = threadIdx.x; index < width; index += THREADS) {
        sum += row[index];
    }
    const float mean = block_sum(sum, scratch) / width;
    float squares = 0.0f;
    for (int index = threadIdx.x; index < width; index += THREADS) {
        const float centered = row[index] - mean;
        squares = fmaf(centered, centered, squares);
    }
    const float inverse = rsqrtf(block_sum(squares, scratch) / width + epsilon);
    float *destination = output + static_cast<size_t>(blockIdx.x) * width;
    for (int index = threadIdx.x; index < width; index += THREADS) {
        destination[index] = fmaf((row[index] - mean) * inverse, weight[index], bias[index]);
    }
}

// GELU in place, with the tanh approximation or the exact error function.
template <bool TANH> __global__ void gelu_kernel(float *values, size_t count) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float x = values[index];
        if constexpr (TANH) {
            const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            values[index] = 0.5f * x * (1.0f + tanhf(inner));
        } else {
            values[index] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
        }
    }
}

// Splits qkv [tokens, 3, heads, dim] into query and key [heads, tokens, dim], rotated by
// angles[token, dim / 2] with dimension i paired with i + dim / 2, and the transposed values
// [heads, dim, tokens].
__global__ void split_rotate_kernel(const float *__restrict__ qkv, const float *__restrict__ angles,
                                    int tokens, int heads, int dim, float *__restrict__ query,
                                    float *__restrict__ key, float *__restrict__ values) {
    const int half = dim / 2;
    const size_t count = static_cast<size_t>(tokens) * heads * dim;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int d = static_cast<int>(index % dim);
        const int head = static_cast<int>((index / dim) % heads);
        const int token = static_cast<int>(index / (static_cast<size_t>(dim) * heads));
        const size_t row = static_cast<size_t>(token) * 3 * heads * dim;
        const int pair = d < half ? d : d - half;
        const float angle = angles[static_cast<size_t>(token) * half + pair];
        float cosine, sine;
        sincosf(angle, &sine, &cosine);
        const size_t destination = (static_cast<size_t>(head) * tokens + token) * dim + d;
        for (int part = 0; part < 2; part++) {
            const float *source = qkv + row + static_cast<size_t>(part) * heads * dim +
                                  static_cast<size_t>(head) * dim;
            const float rotated = d < half ? source[d] * cosine - source[d + half] * sine
                                           : source[d] * cosine + source[d - half] * sine;
            (part == 0 ? query : key)[destination] = rotated;
        }
        values[(static_cast<size_t>(head) * dim + d) * tokens + token] =
            qkv[row + 2 * static_cast<size_t>(heads) * dim + static_cast<size_t>(head) * dim + d];
    }
}

// One block per row of scores: softmax(scale · row) in place.
__global__ void __launch_bounds__(THREADS) softmax_kernel(float *scores, int columns, float scale) {
    __shared__ float scratch[THREADS / 32];
    float *row = scores + static_cast<size_t>(blockIdx.x) * columns;
    float maximum = -INFINITY;
    for (int index = threadIdx.x; index < columns; index += THREADS) {
        maximum = fmaxf(maximum, row[index] * scale);
    }
    maximum = block_max(maximum, scratch);
    float sum = 0.0f;
    for (int index = threadIdx.x; index < columns; index += THREADS) {
        const float value = __expf(row[index] * scale - maximum);
        row[index] = value;
        sum += value;
    }
    const float inverse = 1.0f / block_sum(sum, scratch);
    for (int index = threadIdx.x; index < columns; index += THREADS) {
        row[index] *= inverse;
    }
}

__global__ void add_kernel(float *values, const float *__restrict__ other, size_t count) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        values[index] += other[index];
    }
}

// [heads, tokens, dim] to [tokens, heads · dim].
__global__ void merge_heads_kernel(const float *__restrict__ input, int tokens, int heads, int dim,
                                   float *__restrict__ output) {
    const size_t count = static_cast<size_t>(tokens) * heads * dim;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int d = static_cast<int>(index % dim);
        const int head = static_cast<int>((index / dim) % heads);
        const size_t token = index / (static_cast<size_t>(dim) * heads);
        output[index] = input[(static_cast<size_t>(head) * tokens + token) * dim + d];
    }
}

} // namespace

extern "C" int mmh3_vision_layer_norm(const float *input, const float *weight, const float *bias,
                                      float epsilon, int rows, int width, float *output,
                                      cudaStream_t stream) {
    layer_norm_kernel<<<rows, THREADS, 0, stream>>>(input, weight, bias, epsilon, width, output);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vision_gelu(float *values, size_t count, int tanh_approximation,
                                cudaStream_t stream) {
    if (tanh_approximation) {
        gelu_kernel<true><<<grid_for(count), THREADS, 0, stream>>>(values, count);
    } else {
        gelu_kernel<false><<<grid_for(count), THREADS, 0, stream>>>(values, count);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vision_split_rotate(const float *qkv, const float *angles, int tokens,
                                        int heads, int dim, float *query, float *key, float *values,
                                        cudaStream_t stream) {
    split_rotate_kernel<<<grid_for(static_cast<size_t>(tokens) * heads * dim), THREADS, 0,
                          stream>>>(qkv, angles, tokens, heads, dim, query, key, values);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vision_softmax(float *scores, int rows, int columns, float scale,
                                   cudaStream_t stream) {
    softmax_kernel<<<rows, THREADS, 0, stream>>>(scores, columns, scale);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vision_merge_heads(const float *input, int tokens, int heads, int dim,
                                       float *output, cudaStream_t stream) {
    merge_heads_kernel<<<grid_for(static_cast<size_t>(tokens) * heads * dim), THREADS, 0, stream>>>(
        input, tokens, heads, dim, output);
    return static_cast<int>(cudaGetLastError());
}

// values += other over `count` FP32 values.
extern "C" int mmh3_vision_add(float *values, const float *other, size_t count,
                               cudaStream_t stream) {
    add_kernel<<<grid_for(count), THREADS, 0, stream>>>(values, other, count);
    return static_cast<int>(cudaGetLastError());
}
