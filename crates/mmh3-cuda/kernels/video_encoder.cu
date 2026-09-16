#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Kernels of the video VAE's encoder, a causal 3D CNN. Activations are FP16 and pixel-major,
// [frames, height, width, channels], and the convolutions run as GEMMs over im2col rows. A
// convolution keeps `taps` of its temporal kernel: three for a clip, whose front padding is two
// zero frames, or one for a single frame, where every earlier tap reads a zero frame anyway.

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

// One tile of the canvas [frames, height, width, 3] in [0, 1], normalized with the ImageNet
// statistics, into [frames, tile_height, tile_width, 3]. `frame_offset` picks the clip's first
// frame, and a frame past the canvas repeats its last one, as the reference pads a short clip.
__global__ void tile_input_kernel(const float *__restrict__ canvas, int canvas_frames, int height,
                                  int width, int frame_offset, int top, int left, int frames,
                                  int tile_height, int tile_width, __half *__restrict__ output) {
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float deviation[3] = {0.229f, 0.224f, 0.225f};
    const size_t count = static_cast<size_t>(frames) * tile_height * tile_width * 3;
    const size_t plane = static_cast<size_t>(height) * width * 3;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % 3);
        const size_t pixel = (index / 3) % (static_cast<size_t>(tile_height) * tile_width);
        const int frame =
            static_cast<int>(index / (3 * static_cast<size_t>(tile_height) * tile_width));
        const int source_frame = min(frame_offset + frame, canvas_frames - 1);
        const int y = static_cast<int>(pixel / tile_width) + top;
        const int x = static_cast<int>(pixel % tile_width) + left;
        const float value =
            canvas[source_frame * plane + (static_cast<size_t>(y) * width + x) * 3 + channel];
        output[index] = __float2half_rn((value - mean[channel]) / deviation[channel]);
    }
}

// Rows of the taps of each output pixel, for the `rows` output rows that start at `row_offset`:
// output[r, (kt, ky, kx, c)] = input[t · time_stride + kt − (taps − 1), reflect(y · stride + ky −
// pad), reflect(x · stride + kx − pad), c], zero where the temporal tap falls before the clip, and
// the columns padded with zeros to `columns`. A row is frame t = row / (output_height ·
// output_width) and pixel (y, x) within the frame. Eight channels per thread when `channels` is a
// multiple of 8.
template <bool VECTOR>
__global__ void im2col_kernel(const __half *__restrict__ input, int height, int width, int channels,
                              int stride, int pad, int taps, int time_stride, int output_height,
                              int output_width, int columns, size_t row_offset, size_t rows,
                              __half *__restrict__ output) {
    const int step = VECTOR ? 8 : 1;
    const int chunks = columns / step;
    const size_t count = rows * chunks;
    const size_t frame_pixels = static_cast<size_t>(output_height) * output_width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % chunks) * step;
        const size_t row = index / chunks;
        const size_t source_row = row_offset + row;
        const int frame = static_cast<int>(source_row / frame_pixels);
        const size_t pixel = source_row % frame_pixels;
        const int y = static_cast<int>(pixel / output_width);
        const int x = static_cast<int>(pixel % output_width);
        __half *destination = output + row * columns + column;
        const int tap = column / channels;
        const int source_frame = frame * time_stride + tap / 9 - (taps - 1);
        if (column >= taps * 9 * channels || source_frame < 0) {
            for (int offset = 0; offset < step; offset++) {
                destination[offset] = __float2half(0.0f);
            }
            continue;
        }
        const int spatial = tap % 9;
        const int channel = column % channels;
        const int source_y = reflect(y * stride + spatial / 3 - pad, height);
        const int source_x = reflect(x * stride + spatial % 3 - pad, width);
        const __half *source = input +
                               (static_cast<size_t>(source_frame) * height * width +
                                static_cast<size_t>(source_y) * width + source_x) *
                                   channels +
                               channel;
        if constexpr (VECTOR) {
            *reinterpret_cast<uint4 *>(destination) = *reinterpret_cast<const uint4 *>(source);
        } else {
            *destination = *source;
        }
    }
}

// Sums and sums of squares of each group over PARTIAL_PIXELS pixels per block, into
// partials[frame, block, group, 2]. The statistics of the reference are per frame, so a frame is
// the grid's second axis and never mixes with another.
__global__ void __launch_bounds__(THREADS)
    group_norm_partials_kernel(const __half *__restrict__ input, int pixels, int channels,
                               float *__restrict__ partials) {
    __shared__ float sums[GROUPS * 2];
    for (int index = threadIdx.x; index < GROUPS * 2; index += THREADS) {
        sums[index] = 0.0f;
    }
    __syncthreads();
    const int group_size = channels / GROUPS;
    input += static_cast<size_t>(blockIdx.y) * pixels * channels;
    partials += static_cast<size_t>(blockIdx.y) * gridDim.x * GROUPS * 2;
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

// Mean and reciprocal standard deviation of each group of each frame from the partials, into
// statistics[frame, group, 2].
__global__ void group_norm_statistics_kernel(const float *__restrict__ partials, int blocks,
                                             double count, float epsilon,
                                             float *__restrict__ statistics) {
    const int group = threadIdx.x;
    if (group >= GROUPS) {
        return;
    }
    partials += static_cast<size_t>(blockIdx.x) * blocks * GROUPS * 2;
    statistics += static_cast<size_t>(blockIdx.x) * GROUPS * 2;
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

// SiLU of the group-normalized input with a per-channel affine, each frame with its own
// statistics.
__global__ void group_norm_silu_kernel(const __half *__restrict__ input, int pixels, int channels,
                                       const float *__restrict__ statistics,
                                       const __half *__restrict__ weight,
                                       const __half *__restrict__ bias,
                                       __half *__restrict__ output) {
    const int group_size = channels / GROUPS;
    const size_t frame_values = static_cast<size_t>(pixels) * channels;
    input += static_cast<size_t>(blockIdx.y) * frame_values;
    output += static_cast<size_t>(blockIdx.y) * frame_values;
    statistics += static_cast<size_t>(blockIdx.y) * GROUPS * 2;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < frame_values; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
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

// Normalizes `frames` frames of the tile at (top, left) of a canvas [canvas_frames, height, width,
// 3] in [0, 1] into FP16, starting at frame `frame_offset` and repeating the last frame of the
// canvas past its end.
extern "C" int mmh3_video_encoder_tile_input(const float *canvas, int canvas_frames, int height,
                                             int width, int frame_offset, int top, int left,
                                             int frames, int tile_height, int tile_width,
                                             __half *output, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(frames) * tile_height * tile_width * 3;
    tile_input_kernel<<<grid_for(count), THREADS, 0, stream>>>(canvas, canvas_frames, height, width,
                                                               frame_offset, top, left, frames,
                                                               tile_height, tile_width, output);
    return static_cast<int>(cudaGetLastError());
}

// im2col rows of a 3 × 3 × `taps` convolution with reflect padding `pad` before each spatial axis
// and as much after as the output needs, and zeros before the clip, see im2col_kernel.
extern "C" int mmh3_video_encoder_im2col(const __half *input, int height, int width, int channels,
                                         int stride, int pad, int taps, int time_stride,
                                         int output_height, int output_width, int columns,
                                         size_t row_offset, size_t rows, __half *output,
                                         cudaStream_t stream) {
    if (columns < taps * 9 * channels || height < 2 || width < 2) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const bool vector = channels % 8 == 0 && columns % 8 == 0;
    const size_t count = rows * (vector ? columns / 8 : columns);
    if (vector) {
        im2col_kernel<true><<<grid_for(count), THREADS, 0, stream>>>(
            input, height, width, channels, stride, pad, taps, time_stride, output_height,
            output_width, columns, row_offset, rows, output);
    } else {
        im2col_kernel<false><<<grid_for(count), THREADS, 0, stream>>>(
            input, height, width, channels, stride, pad, taps, time_stride, output_height,
            output_width, columns, row_offset, rows, output);
    }
    return static_cast<int>(cudaGetLastError());
}

// SiLU(GroupNorm(input)) with 32 groups over [frames, pixels, channels], the statistics of each
// frame its own. `partials` holds 64 floats per frame per 256 pixels and `statistics` 64 per frame.
extern "C" int mmh3_video_encoder_group_norm_silu(const __half *input, int frames, int pixels,
                                                  int channels, const __half *weight,
                                                  const __half *bias, float epsilon,
                                                  float *partials, float *statistics,
                                                  __half *output, cudaStream_t stream) {
    if (channels % GROUPS != 0 || pixels <= 0 || frames <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int blocks = (pixels + PARTIAL_PIXELS - 1) / PARTIAL_PIXELS;
    group_norm_partials_kernel<<<dim3(blocks, frames), THREADS, 0, stream>>>(input, pixels,
                                                                             channels, partials);
    group_norm_statistics_kernel<<<frames, GROUPS, 0, stream>>>(
        partials, blocks, static_cast<double>(pixels) * (channels / GROUPS), epsilon, statistics);
    group_norm_silu_kernel<<<dim3(grid_for(static_cast<size_t>(pixels) * channels), frames),
                             THREADS, 0, stream>>>(input, pixels, channels, statistics, weight,
                                                   bias, output);
    return static_cast<int>(cudaGetLastError());
}
