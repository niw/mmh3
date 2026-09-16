#include "tensor_core.cuh"
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

// A 3 x 3 x taps convolution at stride one as one pass over the input: a block keeps a patch of
// output pixels in registers and stages the input its taps read, so the twenty-seven columns of
// every output pixel never reach memory. Pixel rows in shared memory hold a chunk of channels
// padded to 80 bytes, which spreads the eight addresses of every ldmatrix phase over eight bank
// groups. One temporal tap of one chunk of channels is a slab, and the next slab is staged a slice
// at a time along with the weights of the taps, so its arrival overlaps the products of this one.
// A block covers as many output pixels as it can, since the weight it reads costs two bytes per
// pixel however many it keeps.
constexpr int CONV_TILE_ROWS = 8;
constexpr int CONV_TILE_PIXELS = 16;
constexpr int CONV_BLOCK_N = 128;
constexpr int CONV_CHUNK = 32;
/// Rows one warp keeps, which share every weight fragment it reads from shared memory.
constexpr int CONV_WARP_ROWS = 1;
constexpr int CONV_WARPS = CONV_TILE_ROWS / CONV_WARP_ROWS * (CONV_TILE_PIXELS / 16);
constexpr int CONV_THREADS = CONV_WARPS * 32;
constexpr int CONV_ROW_BYTES = CONV_CHUNK * 2;
constexpr int CONV_ROW_CHUNKS = CONV_CHUNK / 8;
constexpr int CONV_PATCH_PIXELS = CONV_TILE_PIXELS + 2;
constexpr int CONV_PATCH_ROWS = CONV_TILE_ROWS + 2;
/// Spatial taps of one temporal tap, each of the nine a separate product.
constexpr int CONV_COMBINATIONS = 9;

// Exchanging two of the four 16-byte chunks of every other pair of rows spreads the eight
// addresses of an ldmatrix phase over eight bank groups, which no padding of 16-byte rows does.
__device__ __forceinline__ int conv_offset(int row, int chunk) {
    return row * CONV_ROW_BYTES + ((chunk ^ ((row >> 1) & 3)) << 4);
}

constexpr int CONV_SLAB_LINES = CONV_PATCH_ROWS * CONV_PATCH_PIXELS;
constexpr int CONV_SLAB_BYTES = CONV_SLAB_LINES * CONV_ROW_BYTES;
constexpr int CONV_WEIGHT_BYTES = CONV_BLOCK_N * CONV_ROW_BYTES;
/// Blocks that share a multiprocessor, and the slab stages that leaves room for. Two blocks of
/// eight warps hide the products' waits better than one block of sixteen.
constexpr int CONV_BLOCKS_PER_SM = 2;
constexpr int CONV_SLAB_STAGES = 2;
constexpr int CONV_SHARED_BYTES = CONV_SLAB_STAGES * CONV_SLAB_BYTES + 2 * CONV_WEIGHT_BYTES;

template <int TAPS, bool NORMALIZE>
__global__ void __launch_bounds__(CONV_THREADS, CONV_BLOCKS_PER_SM)
    conv3d_kernel(const __half *__restrict__ input, int frames, int height, int width, int channels,
                  const __half *__restrict__ weight, int columns, const __half *__restrict__ bias,
                  int outputs, int accumulate, const float *__restrict__ statistics,
                  const __half *__restrict__ norm_weight, const __half *__restrict__ norm_bias,
                  __half *__restrict__ output) {
    extern __shared__ __align__(128) uint8_t shared_memory[];
    const uint32_t slabs_base = shared_address(shared_memory);
    const uint32_t weights = slabs_base + CONV_SLAB_STAGES * CONV_SLAB_BYTES;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int matrix = lane / 8;
    const int matrix_row = lane % 8;
    const int warp_row = warp * CONV_WARP_ROWS;

    const int first_pixel = blockIdx.x * CONV_TILE_PIXELS;
    const int first_row = blockIdx.y * CONV_TILE_ROWS;
    const int frame = blockIdx.z % frames;
    const int first_output = (blockIdx.z / frames) * CONV_BLOCK_N;
    const size_t pixels = static_cast<size_t>(height) * width;
    const int chunks = channels / CONV_CHUNK;
    const int slabs = chunks * TAPS;

    auto slab_stage = [&](int slab) {
        return slabs_base + (slab % CONV_SLAB_STAGES) * CONV_SLAB_BYTES;
    };
    auto weight_stage = [&](int product) { return weights + (product % 2) * CONV_WEIGHT_BYTES; };
    // Decodes copy `index` of slab `slab`: the pixel of a spatial row, and a chunk of its channels.
    // Rows and pixels past the input have nothing to mirror and only feed accumulators the epilogue
    // drops, and a temporal tap before the clip stays zero, as its columns are.
    auto slab_copy = [&](int slab, int index, int &line, int &sub, int &source_frame,
                         const __half *&source) {
        sub = index % CONV_ROW_CHUNKS;
        line = index / CONV_ROW_CHUNKS;
        const int pixel = line % CONV_PATCH_PIXELS;
        const int ky = line / CONV_PATCH_PIXELS;
        source_frame = frame + slab % TAPS - (TAPS - 1);
        const int source_row = first_row + ky - 1;
        const int source_pixel = first_pixel + pixel - 1;
        const bool valid = source_frame >= 0 && source_row <= height && source_pixel <= width;
        source = input +
                 (valid ? (static_cast<size_t>(source_frame) * pixels +
                           reflect(source_row, height) * width + reflect(source_pixel, width)) *
                              channels
                        : 0) +
                 slab / TAPS * CONV_CHUNK + sub * 8;
        return valid;
    };
    auto load_slab_slice = [&](int slab, int slice) {
        const uint32_t stage = slab_stage(slab);
        for (int index = slice * CONV_THREADS + threadIdx.x;
             index < CONV_SLAB_LINES * CONV_ROW_CHUNKS; index += CONV_COMBINATIONS * CONV_THREADS) {
            int line = 0, sub = 0, source_frame = 0;
            const __half *source = nullptr;
            const bool valid = slab_copy(slab, index, line, sub, source_frame, source);
            copy_async_16(stage + conv_offset(line, sub), source, valid);
        }
    };
    // SiLU of the group norm of the slab, in place, so that pass never writes to memory. The
    // groups divide the channels by a power of two, and eight channels share their affine, so the
    // loop keeps to shifts and two vector loads.
    const int group_shift = __ffs(channels / GROUPS) - 1;
    auto normalize_slab = [&](int slab) {
        for (int index = threadIdx.x; index < CONV_SLAB_LINES * CONV_ROW_CHUNKS;
             index += CONV_THREADS) {
            int line = 0, sub = 0, source_frame = 0;
            const __half *source = nullptr;
            if (!slab_copy(slab, index, line, sub, source_frame, source)) {
                continue;
            }
            const int channel = slab / TAPS * CONV_CHUNK + sub * 8;
            const float *frame_statistics = statistics + source_frame * GROUPS * 2;
            const uint4 scale = *reinterpret_cast<const uint4 *>(norm_weight + channel);
            const uint4 shift = *reinterpret_cast<const uint4 *>(norm_bias + channel);
            const __half *scales = reinterpret_cast<const __half *>(&scale);
            const __half *shifts = reinterpret_cast<const __half *>(&shift);
            __half *elements = reinterpret_cast<__half *>(
                shared_memory + (slab % CONV_SLAB_STAGES) * CONV_SLAB_BYTES +
                conv_offset(line, sub));
            #pragma unroll
            for (int element = 0; element < 8; element++) {
                const int group = (channel + element) >> group_shift;
                const float normalized =
                    (__half2float(elements[element]) - frame_statistics[group * 2]) *
                    frame_statistics[group * 2 + 1];
                const float value =
                    fmaf(normalized, __half2float(scales[element]), __half2float(shifts[element]));
                elements[element] = __float2half_rn(__fdividef(value, 1.0f + __expf(-value)));
            }
        }
    };
    // The weight rows of one spatial and temporal tap, for a chunk of channels.
    auto load_weight = [&](int product) {
        const uint32_t stage = weight_stage(product);
        const int slab = product / CONV_COMBINATIONS;
        const int tap = slab % TAPS * CONV_COMBINATIONS + product % CONV_COMBINATIONS;
        for (int index = threadIdx.x; index < CONV_BLOCK_N * CONV_ROW_CHUNKS;
             index += CONV_THREADS) {
            const int sub = index % CONV_ROW_CHUNKS;
            const int line = index / CONV_ROW_CHUNKS;
            const int out = first_output + line;
            const bool valid = out < outputs;
            const __half *source = weight + (valid ? static_cast<size_t>(out) * columns : 0) +
                                   tap * channels + slab / TAPS * CONV_CHUNK + sub * 8;
            copy_async_16(stage + conv_offset(line, sub), source, valid);
        }
    };

    // Two accumulator groups per sixteen outputs a weight stage holds, for each row.
    float accumulators[CONV_WARP_ROWS][CONV_BLOCK_N / 8][4] = {};
    for (int slice = 0; slice < CONV_COMBINATIONS; slice++) {
        load_slab_slice(0, slice);
    }
    load_weight(0);
    copy_async_commit();
    if constexpr (CONV_SLAB_STAGES == 1) {
        copy_async_wait<0>();
        __syncthreads();
    }
    const int products = slabs * CONV_COMBINATIONS;
    for (int product = 0; product < products; product++) {
        const int combination = product % CONV_COMBINATIONS;
        const int slab = product / CONV_COMBINATIONS;
        if (combination == 0) {
            // One stage holds the slab this block is about to read, so the next one waits.
            if constexpr (CONV_SLAB_STAGES == 1) {
                if (slab > 0) {
                    __syncthreads();
                    for (int slice = 0; slice < CONV_COMBINATIONS; slice++) {
                        load_slab_slice(slab, slice);
                    }
                    copy_async_commit();
                }
                copy_async_wait<0>();
                __syncthreads();
            } else if constexpr (NORMALIZE) {
                copy_async_wait<0>();
                __syncthreads();
            }
            if constexpr (NORMALIZE) {
                normalize_slab(slab);
            }
        }
        if (CONV_SLAB_STAGES == 2 && slab + 1 < slabs) {
            load_slab_slice(slab + 1, combination);
        }
        if (product + 1 < products) {
            load_weight(product + 1);
        }
        copy_async_commit();
        copy_async_wait<1>();
        __syncthreads();

        const uint32_t stage = weight_stage(product);
        const int patch_line = (warp_row + combination / 3) * CONV_PATCH_PIXELS;
        const int pixel = (matrix % 2) * 8 + matrix_row + combination % 3;
        #pragma unroll
        for (int step = 0; step < CONV_CHUNK / 16; step++) {
            uint32_t rows[CONV_WARP_ROWS][4];
            #pragma unroll
            for (int row = 0; row < CONV_WARP_ROWS; row++) {
                load_matrix_x4(rows[row],
                               slab_stage(slab) +
                                   conv_offset(patch_line + row * CONV_PATCH_PIXELS + pixel,
                                               step * 2 + matrix / 2));
            }
            #pragma unroll
            for (int pair = 0; pair < CONV_BLOCK_N / 16; pair++) {
                uint32_t lines[4];
                const int line = pair * 16 + (matrix / 2) * 8 + matrix_row;
                load_matrix_x4(lines, stage + conv_offset(line, step * 2 + matrix % 2));
                #pragma unroll
                for (int row = 0; row < CONV_WARP_ROWS; row++) {
                    Numeric<__half>::mma(accumulators[row][pair * 2], rows[row], lines[0],
                                         lines[1]);
                    Numeric<__half>::mma(accumulators[row][pair * 2 + 1], rows[row], lines[2],
                                         lines[3]);
                }
            }
        }
        __syncthreads();
    }

    // Thread `lane` holds pixels lane / 4 and lane / 4 + 8 of each pair, at two adjacent outputs,
    // which writes four bytes at a time. The staging buffers are free by now, so the tile passes
    // through them in the order the output holds it and leaves in whole vectors.
    __half *tile = reinterpret_cast<__half *>(shared_memory);
    const int stride = CONV_WARP_ROWS * CONV_TILE_PIXELS * CONV_BLOCK_N;
    #pragma unroll
    for (int line = 0; line < CONV_WARP_ROWS; line++) {
        #pragma unroll
        for (int pair = 0; pair < CONV_BLOCK_N / 8; pair++) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const int pixel = lane / 4 + half * 8;
                const int out = (pair / 2) * 16 + (pair % 2) * 8 + (lane % 4) * 2;
                #pragma unroll
                for (int index = 0; index < 2; index++) {
                    const int position =
                        warp * stride + (line * CONV_TILE_PIXELS + pixel) * CONV_BLOCK_N + out;
                    tile[position + index] = __float2half_rn(
                        accumulators[line][pair][half * 2 + index] +
                        __half2float(bias[min(first_output + out + index, outputs - 1)]));
                }
            }
        }
    }
    __syncthreads();
    // Eight halves per thread of the rows this block wrote, in the output's own order.
    for (int index = threadIdx.x * 8; index < CONV_WARPS * stride; index += CONV_THREADS * 8) {
        const int out = index % CONV_BLOCK_N;
        const int pixel = index / CONV_BLOCK_N % CONV_TILE_PIXELS;
        const int row = index / (CONV_BLOCK_N * CONV_TILE_PIXELS);
        if (first_row + row >= height || first_pixel + pixel >= width ||
            first_output + out >= outputs) {
            continue;
        }
        const size_t position = ((static_cast<size_t>(frame) * pixels + (first_row + row) * width +
                                  first_pixel + pixel) *
                                     outputs +
                                 first_output + out);
        uint4 values = *reinterpret_cast<const uint4 *>(tile + index);
        if (accumulate != 0) {
            const uint4 previous = *reinterpret_cast<const uint4 *>(output + position);
            __half *sums = reinterpret_cast<__half *>(&values);
            const __half *held = reinterpret_cast<const __half *>(&previous);
            #pragma unroll
            for (int element = 0; element < 8; element++) {
                sums[element] =
                    __float2half_rn(__half2float(sums[element]) + __half2float(held[element]));
            }
        }
        *reinterpret_cast<uint4 *>(output + position) = values;
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

// A 3 x 3 x `taps` convolution at stride one, with reflect padding in space and zeros before the
// clip: output[t, y, x, o] = bias[o] + sum over the taps and channels, added onto the output when
// `accumulate` is set. The weight holds `columns` per output, the taps ordered (kt, ky, kx,
// channel). `channels` must be a multiple of 32 and `taps` one or three.
extern "C" int mmh3_video_encoder_conv3d(const __half *input, int frames, int height, int width,
                                         int channels, int taps, const __half *weight, int columns,
                                         const __half *bias, int outputs, int accumulate,
                                         const float *statistics, const __half *norm_weight,
                                         const __half *norm_bias, __half *output,
                                         cudaStream_t stream) {
    if (channels % CONV_CHUNK != 0 || columns < taps * 9 * channels || (taps != 1 && taps != 3) ||
        height < 2 || width < 2 || frames <= 0 || outputs <= 0 ||
        (statistics != nullptr && channels % GROUPS != 0)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int output_blocks = (outputs + CONV_BLOCK_N - 1) / CONV_BLOCK_N;
    const dim3 grid((width + CONV_TILE_PIXELS - 1) / CONV_TILE_PIXELS,
                    (height + CONV_TILE_ROWS - 1) / CONV_TILE_ROWS, frames * output_blocks);
    auto launch = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             CONV_SHARED_BYTES);
        kernel<<<grid, CONV_THREADS, CONV_SHARED_BYTES, stream>>>(
            input, frames, height, width, channels, weight, columns, bias, outputs, accumulate,
            statistics, norm_weight, norm_bias, output);
    };
    if (statistics != nullptr) {
        if (taps == 3) {
            launch(conv3d_kernel<3, true>);
        } else {
            launch(conv3d_kernel<1, true>);
        }
    } else if (taps == 3) {
        launch(conv3d_kernel<3, false>);
    } else {
        launch(conv3d_kernel<1, false>);
    }
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

// Mean and reciprocal standard deviation of each group of each frame, into statistics[frame, group,
// 2], for a convolution that normalizes its own input.
extern "C" int mmh3_video_encoder_group_norm_statistics(const __half *input, int frames, int pixels,
                                                        int channels, float epsilon,
                                                        float *partials, float *statistics,
                                                        cudaStream_t stream) {
    if (channels % GROUPS != 0 || pixels <= 0 || frames <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int blocks = (pixels + PARTIAL_PIXELS - 1) / PARTIAL_PIXELS;
    group_norm_partials_kernel<<<dim3(blocks, frames), THREADS, 0, stream>>>(input, pixels,
                                                                             channels, partials);
    group_norm_statistics_kernel<<<frames, GROUPS, 0, stream>>>(
        partials, blocks, static_cast<double>(pixels) * (channels / GROUPS), epsilon, statistics);
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
