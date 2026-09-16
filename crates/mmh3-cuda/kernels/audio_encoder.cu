#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Kernels of the audio VAE's DAC encoder and of the attention head that projects its features onto
// the latent channels. Signals are FP32 and time-major, [length, channels], one stereo channel at a
// time, and the convolutions become GEMMs over the im2col rows of audio_vae.cu.

namespace {

constexpr int THREADS = 256;
constexpr float SNAKE_EPSILON = 1e-9f;

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

// Snake with one parameter per channel, which the encoder uses for both α and β:
// x + sin²(αx) / (α + 1e-9).
__global__ void snake_kernel(const float *__restrict__ input, float *__restrict__ output,
                             const float *__restrict__ alpha, size_t count, int channels) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const float parameter = alpha[index % channels];
        const float value = input[index];
        const float sine = sinf(parameter * value);
        output[index] = fmaf(sine * sine, 1.0f / (parameter + SNAKE_EPSILON), value);
    }
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

// Splits qkv [rows, 3, heads, dim] into query and key [heads, rows, dim] and the transposed values
// [heads, dim, rows], which the GEMM of the attention weights reads as its weight.
__global__ void split_qkv_kernel(const float *__restrict__ qkv, int rows, int heads, int dim,
                                 float *__restrict__ query, float *__restrict__ key,
                                 float *__restrict__ values) {
    const size_t count = static_cast<size_t>(rows) * heads * dim;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const int channel = static_cast<int>(index % dim);
        const int head = static_cast<int>((index / dim) % heads);
        const int row = static_cast<int>(index / (static_cast<size_t>(dim) * heads));
        const size_t source =
            static_cast<size_t>(row) * 3 * heads * dim + static_cast<size_t>(head) * dim + channel;
        const size_t plane = static_cast<size_t>(heads) * dim;
        query[(static_cast<size_t>(head) * rows + row) * dim + channel] = qkv[source];
        key[(static_cast<size_t>(head) * rows + row) * dim + channel] = qkv[source + plane];
        values[(static_cast<size_t>(head) * dim + channel) * rows + row] = qkv[source + 2 * plane];
    }
}

// One block per query row: softmax(scale · row) in place over the keys up to the query, zero after
// it.
__global__ void __launch_bounds__(THREADS)
    causal_softmax_kernel(float *scores, int columns, float scale) {
    __shared__ float scratch[THREADS / 32];
    float *row = scores + static_cast<size_t>(blockIdx.x) * columns;
    const int keys = static_cast<int>(blockIdx.x) + 1;
    float maximum = -INFINITY;
    for (int index = threadIdx.x; index < keys; index += THREADS) {
        maximum = fmaxf(maximum, row[index] * scale);
    }
    maximum = block_max(maximum, scratch);
    float sum = 0.0f;
    for (int index = threadIdx.x; index < keys; index += THREADS) {
        const float value = __expf(row[index] * scale - maximum);
        row[index] = value;
        sum += value;
    }
    const float total = block_sum(sum, scratch);
    for (int index = threadIdx.x; index < columns; index += THREADS) {
        row[index] = index < keys ? row[index] / total : 0.0f;
    }
}

// The mean of the heads, then average pooling of the head dimension down to `outputs` groups, the
// adaptive pooling of the reference.
__global__ void pool_heads_kernel(const float *__restrict__ attention, int rows, int heads, int dim,
                                  int outputs, float *__restrict__ pooled) {
    const size_t count = static_cast<size_t>(rows) * outputs;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const int output = static_cast<int>(index % outputs);
        const int row = static_cast<int>(index / outputs);
        const int start = output * dim / outputs;
        const int end = (output * dim + dim + outputs - 1) / outputs;
        float sum = 0.0f;
        for (int head = 0; head < heads; head++) {
            const float *values = attention + (static_cast<size_t>(head) * rows + row) * dim;
            for (int channel = start; channel < end; channel++) {
                sum += values[channel];
            }
        }
        pooled[index] = sum / (static_cast<float>(heads) * (end - start));
    }
}

// GeGLU: the gate through GELU with the tanh approximation, times the value.
__global__ void geglu_kernel(const float *__restrict__ gate, const float *__restrict__ value,
                             float *__restrict__ output, size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const float x = gate[index];
        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        output[index] = 0.5f * x * (1.0f + tanhf(inner)) * value[index];
    }
}

} // namespace

extern "C" int mmh3_audio_encoder_snake(const float *input, float *output, const float *alpha,
                                        int length, int channels, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(length) * channels;
    snake_kernel<<<grid_for(count), THREADS, 0, stream>>>(input, output, alpha, count, channels);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_audio_encoder_layer_norm(const float *input, const float *weight,
                                             const float *bias, float epsilon, int rows, int width,
                                             float *output, cudaStream_t stream) {
    layer_norm_kernel<<<rows, THREADS, 0, stream>>>(input, weight, bias, epsilon, width, output);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_audio_encoder_split_qkv(const float *qkv, int rows, int heads, int dim,
                                            float *query, float *key, float *values,
                                            cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * heads * dim;
    split_qkv_kernel<<<grid_for(count), THREADS, 0, stream>>>(qkv, rows, heads, dim, query, key,
                                                              values);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_audio_encoder_causal_softmax(float *scores, int rows, int columns, float scale,
                                                 cudaStream_t stream) {
    causal_softmax_kernel<<<rows, THREADS, 0, stream>>>(scores, columns, scale);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_audio_encoder_pool_heads(const float *attention, int rows, int heads, int dim,
                                             int outputs, float *pooled, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * outputs;
    pool_heads_kernel<<<grid_for(count), THREADS, 0, stream>>>(attention, rows, heads, dim, outputs,
                                                               pooled);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_audio_encoder_geglu(const float *gate, const float *value, float *output,
                                        size_t count, cudaStream_t stream) {
    geglu_kernel<<<grid_for(count), THREADS, 0, stream>>>(gate, value, output, count);
    return static_cast<int>(cudaGetLastError());
}
