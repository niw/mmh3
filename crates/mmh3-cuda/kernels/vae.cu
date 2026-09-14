#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "convrot.cuh"

// Kernels of the video VAE's ViT decoder. The residual stream is FP32, linear layers run in FP16
// like the reference, and the decoded pixels are blended and finalized in FP32.

namespace {

constexpr int ROW_THREADS = 256;
constexpr int VAE_HEAD_DIM = 64;
constexpr int PATCH = 16;
constexpr int PATCH_FRAMES = 4;

__device__ __forceinline__ float block_sum(float value, float *scratch) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = value;
    }
    __syncthreads();
    float total = 0.0f;
    for (int index = 0; index < static_cast<int>(blockDim.x / 32); index++) {
        total += scratch[index];
    }
    __syncthreads();
    return total;
}

// RMSNorm with an affine weight, or LayerNorm with weight and bias when `bias` is set.
__global__ void __launch_bounds__(ROW_THREADS)
    norm_kernel(const float *__restrict__ input, const __half *__restrict__ weight,
                const __half *__restrict__ bias, __half *__restrict__ output, int width,
                float epsilon) {
    __shared__ float scratch[ROW_THREADS / 32];
    const float *row = input + static_cast<size_t>(blockIdx.x) * width;
    float mean = 0.0f;
    if (bias != nullptr) {
        float sum = 0.0f;
        for (int index = threadIdx.x; index < width; index += blockDim.x) {
            sum += row[index];
        }
        mean = block_sum(sum, scratch) / width;
    }
    float squares = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float centered = row[index] - mean;
        squares += centered * centered;
    }
    const float inverse = rsqrtf(block_sum(squares, scratch) / width + epsilon);
    __half *output_row = output + static_cast<size_t>(blockIdx.x) * width;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        float value = (row[index] - mean) * inverse * __half2float(weight[index]);
        if (bias != nullptr) {
            value += __half2float(bias[index]);
        }
        output_row[index] = __float2half_rn(value);
    }
}

// Per-head RMSNorm without weight of q and k in the [tokens, heads, 3, 64] qkv buffer, then the
// split-half rotary embedding on the first 2 × pairs dimensions. Angles are per token within a
// tile.
__global__ void qk_norm_rope_kernel(__half *qkv, const float *__restrict__ angles, int pairs,
                                    int tile_tokens, int tokens, int heads, float epsilon) {
    __shared__ float values[ROW_THREADS / 32][VAE_HEAD_DIM];
    const int warp_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t item = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp_in_block;
    if (item >= static_cast<int64_t>(tokens) * heads * 2) {
        return;
    }
    const int is_key = static_cast<int>(item % 2);
    const int head = static_cast<int>((item / 2) % heads);
    const int64_t token = item / 2 / heads;
    __half *vector = qkv + (token * heads + head) * 3 * VAE_HEAD_DIM + is_key * VAE_HEAD_DIM;
    float *shared = values[warp_in_block];

    float squares = 0.0f;
    for (int index = lane; index < VAE_HEAD_DIM; index += 32) {
        const float value = __half2float(vector[index]);
        shared[index] = value;
        squares += value * value;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        squares += __shfl_xor_sync(0xffffffff, squares, offset);
    }
    const float inverse = rsqrtf(squares / VAE_HEAD_DIM + epsilon);
    for (int index = lane; index < VAE_HEAD_DIM; index += 32) {
        shared[index] *= inverse;
    }
    __syncwarp();
    const float *token_angles = angles + (token % tile_tokens) * pairs;
    for (int pair = lane; pair < pairs; pair += 32) {
        float sine, cosine;
        sincosf(token_angles[pair], &sine, &cosine);
        const float first = shared[pair];
        const float second = shared[pair + pairs];
        shared[pair] = first * cosine - second * sine;
        shared[pair + pairs] = first * sine + second * cosine;
    }
    __syncwarp();
    for (int index = lane; index < VAE_HEAD_DIM; index += 32) {
        vector[index] = __float2half_rn(shared[index]);
    }
}

// For each token: residual += scale ⊙ delta when a delta is given, then the unbiased norm_kernel of
// the residual in FP16, rotated and quantized to INT8 with the row's scale.
//
// NOTE: The result matches residual_add_scaled_kernel, norm_kernel without a bias and
// rotate_quantize_kernel run one after the other in every bit. The compiler fuses the residual
// update and the sum of squares of those kernels into multiply-adds, which the _rn intrinsics spell
// out here, and the normalized row reaches the rotation through shared memory in FP16.
template <int GROUPS_PER_WARP>
__global__ void __launch_bounds__(ROW_THREADS)
    add_norm_quantize_kernel(float *__restrict__ residual, const __half *__restrict__ delta,
                             const __half *__restrict__ scale, const __half *__restrict__ weight,
                             int8_t *__restrict__ quantized, float *__restrict__ scales, int width,
                             float epsilon) {
    extern __shared__ __align__(16) __half normalized_row[];
    __shared__ float scratch[ROW_THREADS / 32];
    const int64_t token = blockIdx.x;
    float *row = residual + token * width;
    float squares = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        float value = row[index];
        if (delta != nullptr) {
            value = __fmaf_rn(__half2float(delta[token * width + index]),
                              __half2float(scale[index]), value);
            row[index] = value;
        }
        squares = __fmaf_rn(value, value, squares);
    }
    const float inverse = rsqrtf(block_sum(squares, scratch) / width + epsilon);
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        normalized_row[index] =
            __float2half_rn(__fmul_rn(__fmul_rn(row[index], inverse), __half2float(weight[index])));
    }
    __syncthreads();
    rotate_quantize_row<GROUPS_PER_WARP>(reinterpret_cast<const __half2 *>(normalized_row),
                                         quantized + token * width, scales + token, width);
}

__global__ void residual_add_scaled_kernel(float *__restrict__ residual,
                                           const __half *__restrict__ delta,
                                           const __half *__restrict__ scale, size_t count,
                                           int width) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        residual[index] += __half2float(delta[index]) * __half2float(scale[index % width]);
    }
}

__global__ void swiglu_kernel(const __half *__restrict__ input, __half *__restrict__ output,
                              size_t count, int width) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const size_t token = index / width;
        const int column = static_cast<int>(index % width);
        const float gate = __half2float(input[token * 2 * width + column]);
        const float up = __half2float(input[token * 2 * width + width + column]);
        output[index] = __float2half_rn(gate / (1.0f + __expf(-gate)) * up);
    }
}

// Widens the embedded tokens to FP32 and writes the register tokens and the zero token after each
// tile's patches.
__global__ void embed_suffix_kernel(const __half *__restrict__ embedded,
                                    const __half *__restrict__ registers,
                                    float *__restrict__ residual, int tile_tokens, int patches,
                                    int registers_count, size_t count, int width) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const int column = static_cast<int>(index % width);
        const int position = static_cast<int>((index / width) % tile_tokens);
        float value;
        if (position < patches) {
            value = __half2float(embedded[index]);
        } else if (position < patches + registers_count) {
            value = __half2float(registers[(position - patches) * width + column]);
        } else {
            value = 0.0f;
        }
        residual[index] = value;
    }
}

struct TileGrid {
    const int *starts_y;
    const int *starts_x;
    const int *overlaps_y;
    const int *overlaps_x;
    int tiles_y;
    int tiles_x;
    int tile_height;
    int tile_width;
    int latent_frames;
    int tile_tokens;
};

// Raw decoder output of one tile at (channel, frame, row, column) of its pixels.
__device__ __forceinline__ float tile_pixel(const __half *projected, const TileGrid &grid, int tile,
                                            int channel, int frame, int row, int column) {
    const int latent_height = grid.tile_height / PATCH;
    const int latent_width = grid.tile_width / PATCH;
    const int token =
        ((frame / PATCH_FRAMES) * latent_height + row / PATCH) * latent_width + column / PATCH;
    const int feature =
        ((channel * PATCH_FRAMES + frame % PATCH_FRAMES) * PATCH + row % PATCH) * PATCH +
        column % PATCH;
    const int features = 3 * PATCH_FRAMES * PATCH * PATCH;
    return __half2float(
        projected[(static_cast<size_t>(tile) * grid.tile_tokens + token) * features + feature]);
}

// Assembles one chunk's canvas [3, frames, height, width] from the decoded tiles. Each pixel
// belongs to the last tile starting at or before it and blends linearly with the raw tile above,
// then with the raw tile to the left, across their overlaps, the same order as the reference.
__global__ void unpatchify_blend_kernel(const __half *__restrict__ projected, TileGrid grid,
                                        float *__restrict__ canvas, int height, int width) {
    const int frames = grid.latent_frames * PATCH_FRAMES;
    const size_t count = static_cast<size_t>(3) * frames * height * width;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const int x = static_cast<int>(index % width);
        const int y = static_cast<int>((index / width) % height);
        const int frame =
            static_cast<int>((index / (static_cast<size_t>(width) * height)) % frames);
        const int channel =
            static_cast<int>(index / (static_cast<size_t>(width) * height * frames));
        int tile_row = 0;
        while (tile_row + 1 < grid.tiles_y && grid.starts_y[tile_row + 1] <= y) {
            tile_row++;
        }
        int tile_column = 0;
        while (tile_column + 1 < grid.tiles_x && grid.starts_x[tile_column + 1] <= x) {
            tile_column++;
        }
        const int row = y - grid.starts_y[tile_row];
        const int column = x - grid.starts_x[tile_column];
        const int tile = tile_row * grid.tiles_x + tile_column;
        float value = tile_pixel(projected, grid, tile, channel, frame, row, column);
        if (tile_row > 0 && row < grid.overlaps_y[tile_row - 1]) {
            const int extent = grid.overlaps_y[tile_row - 1];
            const float above = tile_pixel(projected, grid, tile - grid.tiles_x, channel, frame,
                                           grid.tile_height - extent + row, column);
            const float weight = static_cast<float>(row) / extent;
            value = above * (1.0f - weight) + value * weight;
        }
        if (tile_column > 0 && column < grid.overlaps_x[tile_column - 1]) {
            const int extent = grid.overlaps_x[tile_column - 1];
            const float left = tile_pixel(projected, grid, tile - 1, channel, frame, row,
                                          grid.tile_width - extent + column);
            const float weight = static_cast<float>(column) / extent;
            value = left * (1.0f - weight) + value * weight;
        }
        canvas[index] = value;
    }
}

// Writes `count` canvas frames starting at `first` into the output at `position`, blending the
// first `blend_frames` with the saved overlap, then maps them to pixels in [0, 1].
__global__ void write_frames_kernel(const float *__restrict__ canvas, int canvas_frames, int first,
                                    int count, const float *__restrict__ overlap, int blend_frames,
                                    float *__restrict__ output, int output_frames, int position,
                                    int plane, float mean_red, float mean_green, float mean_blue,
                                    float deviation_red, float deviation_green,
                                    float deviation_blue) {
    const size_t total = static_cast<size_t>(3) * count * plane;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        const int pixel = static_cast<int>(index % plane);
        const int frame = static_cast<int>((index / plane) % count);
        const int channel = static_cast<int>(index / (static_cast<size_t>(plane) * count));
        if (position + frame >= output_frames) {
            continue;
        }
        float value =
            canvas[(static_cast<size_t>(channel) * canvas_frames + first + frame) * plane + pixel];
        if (frame < blend_frames) {
            const float weight = static_cast<float>(frame) / blend_frames;
            const float previous =
                overlap[(static_cast<size_t>(channel) * blend_frames + frame) * plane + pixel];
            value = previous * (1.0f - weight) + value * weight;
        }
        const float mean = channel == 0 ? mean_red : (channel == 1 ? mean_green : mean_blue);
        const float deviation =
            channel == 0 ? deviation_red : (channel == 1 ? deviation_green : deviation_blue);
        value = fminf(fmaxf(value * deviation + mean, 0.0f), 1.0f);
        output[(static_cast<size_t>(channel) * output_frames + position + frame) * plane + pixel] =
            value;
    }
}

// Saves `count` canvas frames starting at `first` as the overlap for the next chunk.
__global__ void save_overlap_kernel(const float *__restrict__ canvas, int canvas_frames, int first,
                                    int count, float *__restrict__ overlap, int plane) {
    const size_t total = static_cast<size_t>(3) * count * plane;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        const int pixel = static_cast<int>(index % plane);
        const int frame = static_cast<int>((index / plane) % count);
        const int channel = static_cast<int>(index / (static_cast<size_t>(plane) * count));
        overlap[(static_cast<size_t>(channel) * count + frame) * plane + pixel] =
            canvas[(static_cast<size_t>(channel) * canvas_frames + first + frame) * plane + pixel];
    }
}

unsigned grid_for(size_t count) {
    size_t blocks = (count + ROW_THREADS - 1) / ROW_THREADS;
    return static_cast<unsigned>(blocks < 16384 ? blocks : 16384);
}

// BT.709 luma weights of red and blue.
constexpr float LUMA_RED = 0.2126f;
constexpr float LUMA_BLUE = 0.0722f;

__device__ __forceinline__ uint8_t limited_range(float value, float offset, float scale) {
    return static_cast<uint8_t>(
        fminf(fmaxf(roundf(__fadd_rn(offset, __fmul_rn(scale, value))), 0.0f), 255.0f));
}

__device__ __forceinline__ float luma(float red, float green, float blue, float luma_green) {
    return __fadd_rn(__fadd_rn(__fmul_rn(LUMA_RED, red), __fmul_rn(luma_green, green)),
                     __fmul_rn(LUMA_BLUE, blue));
}

__device__ __forceinline__ float block_average(const float *values, int top_left, int width) {
    return __fdiv_rn(__fadd_rn(__fadd_rn(__fadd_rn(values[top_left], values[top_left + 1]),
                                         values[top_left + width]),
                               values[top_left + width + 1]),
                     4.0f);
}

// Converts pixels [3, frames, height, width] in [0, 1] into 4:2:0 BT.709 limited-range YUV, per
// frame the Y plane, then Cb, then Cr. Each thread converts one 2 × 2 block. The _rn intrinsics
// keep the rounding of every operation of mmh3_core::media::Yuv420::from_pixels, which the compiler
// would otherwise fuse into multiply-adds.
__global__ void yuv420_kernel(const float *__restrict__ pixels, uint8_t *__restrict__ output,
                              int frames, int height, int width) {
    const int half_height = height / 2;
    const int half_width = width / 2;
    const size_t plane = static_cast<size_t>(height) * width;
    const size_t blocks = static_cast<size_t>(frames) * half_height * half_width;
    const float luma_green = __fsub_rn(__fsub_rn(1.0f, LUMA_RED), LUMA_BLUE);
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < blocks;
         index += stride) {
        const int x = static_cast<int>(index % half_width);
        const int y = static_cast<int>(index / half_width % half_height);
        const size_t frame = index / (static_cast<size_t>(half_width) * half_height);
        const float *red = pixels + frame * plane;
        const float *green = pixels + (frames + frame) * plane;
        const float *blue = pixels + (2 * static_cast<size_t>(frames) + frame) * plane;
        uint8_t *frame_output = output + frame * plane * 3 / 2;
        const int top_left = 2 * y * width + 2 * x;
        for (const int corner : {top_left, top_left + 1, top_left + width, top_left + width + 1}) {
            frame_output[corner] = limited_range(
                luma(red[corner], green[corner], blue[corner], luma_green), 16.0f, 219.0f);
        }
        const float red_average = block_average(red, top_left, width);
        const float blue_average = block_average(blue, top_left, width);
        const float luma_average =
            luma(red_average, block_average(green, top_left, width), blue_average, luma_green);
        const size_t chroma = static_cast<size_t>(y) * half_width + x;
        frame_output[plane + chroma] =
            limited_range(__fdiv_rn(__fsub_rn(blue_average, luma_average),
                                    __fmul_rn(2.0f, __fsub_rn(1.0f, LUMA_BLUE))),
                          128.0f, 224.0f);
        frame_output[plane + plane / 4 + chroma] =
            limited_range(__fdiv_rn(__fsub_rn(red_average, luma_average),
                                    __fmul_rn(2.0f, __fsub_rn(1.0f, LUMA_RED))),
                          128.0f, 224.0f);
    }
}

__global__ void nv12_frame_kernel(const float *__restrict__ pixels, uint8_t *__restrict__ output,
                                  int frames, int height, int width, int frame, size_t pitch) {
    const int half_width = width / 2;
    const size_t plane = static_cast<size_t>(height) * width;
    const size_t blocks = plane / 4;
    const float *red = pixels + frame * plane;
    const float *green = red + frames * plane;
    const float *blue = green + frames * plane;
    const float luma_green = __fsub_rn(__fsub_rn(1.0f, LUMA_RED), LUMA_BLUE);
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < blocks;
         index += stride) {
        const int x = 2 * static_cast<int>(index % half_width);
        const int y = 2 * static_cast<int>(index / half_width);
        const int top_left = y * width + x;
        for (int row = 0; row < 2; ++row) {
            for (int column = 0; column < 2; ++column) {
                const int corner = top_left + row * width + column;
                output[(y + row) * pitch + x + column] = limited_range(
                    luma(red[corner], green[corner], blue[corner], luma_green), 16.0f, 219.0f);
            }
        }
        const float red_average = block_average(red, top_left, width);
        const float blue_average = block_average(blue, top_left, width);
        const float luma_average =
            luma(red_average, block_average(green, top_left, width), blue_average, luma_green);
        const size_t chroma = (height + y / 2) * pitch + x;
        output[chroma] = limited_range(__fdiv_rn(__fsub_rn(blue_average, luma_average),
                                                 __fmul_rn(2.0f, __fsub_rn(1.0f, LUMA_BLUE))),
                                       128.0f, 224.0f);
        output[chroma + 1] = limited_range(__fdiv_rn(__fsub_rn(red_average, luma_average),
                                                     __fmul_rn(2.0f, __fsub_rn(1.0f, LUMA_RED))),
                                           128.0f, 224.0f);
    }
}

} // namespace

extern "C" int mmh3_vae_norm(const float *input, const __half *weight, const __half *bias,
                             __half *output, int tokens, int width, float epsilon,
                             cudaStream_t stream) {
    norm_kernel<<<tokens, ROW_THREADS, 0, stream>>>(input, weight, bias, output, width, epsilon);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vae_qk_norm_rope(__half *qkv, const float *angles, int pairs, int tile_tokens,
                                     int tokens, int heads, float epsilon, cudaStream_t stream) {
    const int64_t items = static_cast<int64_t>(tokens) * heads * 2;
    const int warps_per_block = ROW_THREADS / 32;
    const unsigned blocks = static_cast<unsigned>((items + warps_per_block - 1) / warps_per_block);
    qk_norm_rope_kernel<<<blocks, ROW_THREADS, 0, stream>>>(qkv, angles, pairs, tile_tokens, tokens,
                                                            heads, epsilon);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vae_residual_add_scaled(float *residual, const __half *delta,
                                            const __half *scale, int tokens, int width,
                                            cudaStream_t stream) {
    const size_t count = static_cast<size_t>(tokens) * width;
    residual_add_scaled_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(residual, delta, scale,
                                                                            count, width);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vae_swiglu(const __half *input, __half *output, int tokens, int width,
                               cudaStream_t stream) {
    const size_t count = static_cast<size_t>(tokens) * width;
    swiglu_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(input, output, count, width);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vae_embed_suffix(const __half *embedded, const __half *registers,
                                     float *residual, int tiles, int tile_tokens, int patches,
                                     int registers_count, int width, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(tiles) * tile_tokens * width;
    embed_suffix_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(
        embedded, registers, residual, tile_tokens, patches, registers_count, count, width);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vae_unpatchify_blend(const __half *projected, const int *starts_y,
                                         const int *starts_x, const int *overlaps_y,
                                         const int *overlaps_x, int tiles_y, int tiles_x,
                                         int tile_height, int tile_width, int latent_frames,
                                         int tile_tokens, float *canvas, int height, int width,
                                         cudaStream_t stream) {
    TileGrid grid = {starts_y, starts_x,    overlaps_y, overlaps_x,    tiles_y,
                     tiles_x,  tile_height, tile_width, latent_frames, tile_tokens};
    const size_t count = static_cast<size_t>(3) * latent_frames * PATCH_FRAMES * height * width;
    unpatchify_blend_kernel<<<grid_for(count), ROW_THREADS, 0, stream>>>(projected, grid, canvas,
                                                                         height, width);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vae_write_frames(const float *canvas, int canvas_frames, int first, int count,
                                     const float *overlap, int blend_frames, float *output,
                                     int output_frames, int position, int plane, const float *mean,
                                     const float *deviation, cudaStream_t stream) {
    const size_t total = static_cast<size_t>(3) * count * plane;
    write_frames_kernel<<<grid_for(total), ROW_THREADS, 0, stream>>>(
        canvas, canvas_frames, first, count, overlap, blend_frames, output, output_frames, position,
        plane, mean[0], mean[1], mean[2], deviation[0], deviation[1], deviation[2]);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vae_save_overlap(const float *canvas, int canvas_frames, int first, int count,
                                     float *overlap, int plane, cudaStream_t stream) {
    const size_t total = static_cast<size_t>(3) * count * plane;
    save_overlap_kernel<<<grid_for(total), ROW_THREADS, 0, stream>>>(canvas, canvas_frames, first,
                                                                     count, overlap, plane);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_yuv420(const float *pixels, uint8_t *output, int frames, int height, int width,
                           cudaStream_t stream) {
    if (height % 2 != 0 || width % 2 != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t blocks = static_cast<size_t>(frames) * (height / 2) * (width / 2);
    yuv420_kernel<<<grid_for(blocks), ROW_THREADS, 0, stream>>>(pixels, output, frames, height,
                                                                width);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_vae_add_norm_quantize(float *residual, const __half *delta, const __half *scale,
                                          const __half *weight, int8_t *quantized, float *scales,
                                          int tokens, int width, float epsilon,
                                          cudaStream_t stream) {
    const int groups = width / CONVROT_GROUP;
    const int groups_per_warp = (groups + ROW_THREADS / 32 - 1) / (ROW_THREADS / 32);
    if (width % CONVROT_GROUP != 0 || groups == 0 || groups_per_warp > MAX_GROUPS_PER_WARP) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t shared_bytes = static_cast<size_t>(width) * sizeof(__half);
    auto launch = [&](auto kernel) {
        kernel<<<tokens, ROW_THREADS, shared_bytes, stream>>>(residual, delta, scale, weight,
                                                              quantized, scales, width, epsilon);
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

extern "C" int mmh3_nv12_frame(const float *pixels, uint8_t *output, int frames, int height,
                               int width, int frame, size_t pitch, cudaStream_t stream) {
    const size_t blocks = static_cast<size_t>(height) * width / 4;
    nv12_frame_kernel<<<grid_for(blocks), ROW_THREADS, 0, stream>>>(pixels, output, frames, height,
                                                                    width, frame, pitch);
    return static_cast<int>(cudaGetLastError());
}
