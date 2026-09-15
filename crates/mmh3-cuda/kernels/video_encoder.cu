#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Kernels of the video VAE's encoder, a causal 3D CNN. For a single frame every temporal kernel
// reduces to its last tap, so the encoder runs 2D convolutions on FP16 activations stored
// pixel-major, [height, width, channels], as GEMMs over im2col rows.

namespace {

constexpr int THREADS = 256;
constexpr int GROUPS = 32;
// Pixels one block of group_norm_partials_kernel reduces.
constexpr int PARTIAL_PIXELS = 256;

unsigned grid_for(size_t count) {
    size_t blocks = (count + THREADS - 1) / THREADS;
    return static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
}

// Index `index` of an axis of `length` values mirrored at its ends without repeating them.
__device__ __forceinline__ int reflect(int index, int length) {
    if (index < 0) {
        return -index;
    }
    return index >= length ? 2 * length - 2 - index : index;
}

// One tile of the canvas [height, width, 3] in [0, 1], normalized with the ImageNet statistics,
// into [tile_height, tile_width, 3].
__global__ void tile_input_kernel(const float *__restrict__ canvas, int width, int top, int left,
                                  int tile_height, int tile_width, __half *__restrict__ output) {
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float deviation[3] = {0.229f, 0.224f, 0.225f};
    const size_t count = static_cast<size_t>(tile_height) * tile_width * 3;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % 3);
        const size_t pixel = index / 3;
        const int y = static_cast<int>(pixel / tile_width) + top;
        const int x = static_cast<int>(pixel % tile_width) + left;
        const float value = canvas[(static_cast<size_t>(y) * width + x) * 3 + channel];
        output[index] = __float2half_rn((value - mean[channel]) / deviation[channel]);
    }
}

// Rows of the 3 × 3 taps of each output pixel: output[(y, x), (ky, kx, c)] = input[reflect(y ·
// stride + ky − pad), reflect(x · stride + kx − pad), c], with the columns padded with zeros to
// `columns`. Eight channels per thread when `channels` is a multiple of 8.
template <bool VECTOR>
__global__ void im2col_kernel(const __half *__restrict__ input, int height, int width, int channels,
                              int stride, int pad, int output_height, int output_width, int columns,
                              __half *__restrict__ output) {
    const int step = VECTOR ? 8 : 1;
    const int chunks = columns / step;
    const size_t count = static_cast<size_t>(output_height) * output_width * chunks;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % chunks) * step;
        const size_t pixel = index / chunks;
        const int y = static_cast<int>(pixel / output_width);
        const int x = static_cast<int>(pixel % output_width);
        __half *destination = output + pixel * columns + column;
        if (column >= 9 * channels) {
            for (int offset = 0; offset < step; offset++) {
                destination[offset] = __float2half(0.0f);
            }
            continue;
        }
        const int tap = column / channels;
        const int channel = column % channels;
        const int source_y = reflect(y * stride + tap / 3 - pad, height);
        const int source_x = reflect(x * stride + tap % 3 - pad, width);
        const __half *source =
            input + (static_cast<size_t>(source_y) * width + source_x) * channels + channel;
        if constexpr (VECTOR) {
            *reinterpret_cast<uint4 *>(destination) = *reinterpret_cast<const uint4 *>(source);
        } else {
            *destination = *source;
        }
    }
}

// Sums and sums of squares of each group over PARTIAL_PIXELS pixels per block, into
// partials[block, group, 2].
__global__ void __launch_bounds__(THREADS)
    group_norm_partials_kernel(const __half *__restrict__ input, int pixels, int channels,
                               float *__restrict__ partials) {
    __shared__ float sums[GROUPS * 2];
    for (int index = threadIdx.x; index < GROUPS * 2; index += THREADS) {
        sums[index] = 0.0f;
    }
    __syncthreads();
    const int group_size = channels / GROUPS;
    const size_t first = static_cast<size_t>(blockIdx.x) * PARTIAL_PIXELS;
    const size_t last = min(first + PARTIAL_PIXELS, static_cast<size_t>(pixels));
    // Threads stride over channels, so each keeps the sums of fixed groups.
    for (int channel = threadIdx.x; channel < channels; channel += THREADS) {
        float sum = 0.0f, squares = 0.0f;
        for (size_t pixel = first; pixel < last; pixel++) {
            const float value = __half2float(input[pixel * channels + channel]);
            sum += value;
            squares = fmaf(value, value, squares);
        }
        const int group = channel / group_size;
        atomicAdd(&sums[group * 2], sum);
        atomicAdd(&sums[group * 2 + 1], squares);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < GROUPS * 2; index += THREADS) {
        partials[static_cast<size_t>(blockIdx.x) * GROUPS * 2 + index] = sums[index];
    }
}

// Mean and reciprocal standard deviation of each group from the partials, into statistics[group,
// 2].
__global__ void group_norm_statistics_kernel(const float *__restrict__ partials, int blocks,
                                             double count, float epsilon,
                                             float *__restrict__ statistics) {
    const int group = threadIdx.x;
    if (group >= GROUPS) {
        return;
    }
    double sum = 0.0, squares = 0.0;
    for (int block = 0; block < blocks; block++) {
        sum += partials[(static_cast<size_t>(block) * GROUPS + group) * 2];
        squares += partials[(static_cast<size_t>(block) * GROUPS + group) * 2 + 1];
    }
    const double mean = sum / count;
    const double variance = fmax(squares / count - mean * mean, 0.0);
    statistics[group * 2] = static_cast<float>(mean);
    statistics[group * 2 + 1] = static_cast<float>(1.0 / sqrt(variance + epsilon));
}

// SiLU of the group-normalized input with a per-channel affine.
__global__ void group_norm_silu_kernel(const __half *__restrict__ input, int pixels, int channels,
                                       const float *__restrict__ statistics,
                                       const __half *__restrict__ weight,
                                       const __half *__restrict__ bias,
                                       __half *__restrict__ output) {
    const int group_size = channels / GROUPS;
    const size_t count = static_cast<size_t>(pixels) * channels;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const int group = channel / group_size;
        const float normalized =
            (__half2float(input[index]) - statistics[group * 2]) * statistics[group * 2 + 1];
        const float value =
            fmaf(normalized, __half2float(weight[channel]), __half2float(bias[channel]));
        output[index] = __float2half_rn(value / (1.0f + __expf(-value)));
    }
}

} // namespace

// Normalizes the tile at (top, left) of a canvas [height, width, 3] in [0, 1] into FP16.
extern "C" int mmh3_video_encoder_tile_input(const float *canvas, int width, int top, int left,
                                             int tile_height, int tile_width, __half *output,
                                             cudaStream_t stream) {
    tile_input_kernel<<<grid_for(static_cast<size_t>(tile_height) * tile_width * 3), THREADS, 0,
                        stream>>>(canvas, width, top, left, tile_height, tile_width, output);
    return static_cast<int>(cudaGetLastError());
}

// im2col rows of a 3 × 3 convolution with reflect padding `pad` before each axis and as much
// after as the output needs, see im2col_kernel.
extern "C" int mmh3_video_encoder_im2col(const __half *input, int height, int width, int channels,
                                         int stride, int pad, int output_height, int output_width,
                                         int columns, __half *output, cudaStream_t stream) {
    if (columns < 9 * channels || height < 2 || width < 2) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const bool vector = channels % 8 == 0 && columns % 8 == 0;
    const size_t count =
        static_cast<size_t>(output_height) * output_width * (vector ? columns / 8 : columns);
    if (vector) {
        im2col_kernel<true><<<grid_for(count), THREADS, 0, stream>>>(input, height, width, channels,
                                                                     stride, pad, output_height,
                                                                     output_width, columns, output);
    } else {
        im2col_kernel<false>
            <<<grid_for(count), THREADS, 0, stream>>>(input, height, width, channels, stride, pad,
                                                      output_height, output_width, columns, output);
    }
    return static_cast<int>(cudaGetLastError());
}

// SiLU(GroupNorm(input)) with 32 groups over [pixels, channels]. `partials` holds 64 floats per
// 256 pixels and `statistics` 64 floats.
extern "C" int mmh3_video_encoder_group_norm_silu(const __half *input, int pixels, int channels,
                                                  const __half *weight, const __half *bias,
                                                  float epsilon, float *partials, float *statistics,
                                                  __half *output, cudaStream_t stream) {
    if (channels % GROUPS != 0 || pixels <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int blocks = (pixels + PARTIAL_PIXELS - 1) / PARTIAL_PIXELS;
    group_norm_partials_kernel<<<blocks, THREADS, 0, stream>>>(input, pixels, channels, partials);
    group_norm_statistics_kernel<<<1, GROUPS, 0, stream>>>(
        partials, blocks, static_cast<double>(pixels) * (channels / GROUPS), epsilon, statistics);
    group_norm_silu_kernel<<<grid_for(static_cast<size_t>(pixels) * channels), THREADS, 0,
                             stream>>>(input, pixels, channels, statistics, weight, bias, output);
    return static_cast<int>(cudaGetLastError());
}
