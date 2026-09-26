#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "device.cuh"
#include "tensor_core.cuh"
#include "tma.cuh"

// Scaled INT8 GEMM for the DiT, text encoder and video VAE linear layers:
//   output[m, n] = round(sum_k activations[m, k] * weights[n, k] * activation_scales[m] *
//   weight_scales[n]
//                        + adapter_scale * sum_r adapter_down[m, r] * adapter_up[n, r] + bias[n])
// rounded to BF16 or FP16, where the low-rank adapter and the bias are optional. Activations and
// weights are row-major with K contiguous. N must be a multiple of the block width and K a multiple
// of 128, which every H3 linear layer satisfies. Only M may be ragged. For SwiGLU layers whose
// weight rows went through interleave_swiglu_rows, the kernel can instead write silu(gate) · up,
// [m, n / 2].
//
// NOTE: A persistent grid of one CTA per SM walks the output tiles. TMA copies 128-byte K slices of
// both operands into a two-stage ring with the 128B swizzle, and eight warps multiply their parts
// of the tile with mma.sync. There is no producer warp because a ninth warp would limit every warp
// to 168 registers. Instead, the warp that releases a stage last issues the copies that refill it.
// An adapter adds stages of 64 ranks of its BF16 operands after the K blocks of each tile,
// multiplied with BF16 MMAs into the scaled FP32 result before the single rounding.
//
// Devices without TMA (sm_89) fill the same swizzled stages with cp.async instead: every thread
// copies its share of the next stage while the warps multiply the current one, and the whole CTA
// synchronizes around each stage.

namespace {

constexpr int BLOCK_K = 128;
constexpr int STAGES = 2;
constexpr int GROUP_M = 8;
constexpr int WARPS = 8;

__device__ __forceinline__ void mma_s8(int32_t (&accumulator)[4], const uint32_t (&a)[4],
                                       uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, "
                 "%7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0]), "+r"(accumulator[1]), "+r"(accumulator[2]),
                   "+r"(accumulator[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int BLOCK_M, int BLOCK_N, int WARPS_M> struct Int8GemmConfig {
    static constexpr int block_m = BLOCK_M;
    static constexpr int block_n = BLOCK_N;
    static constexpr int warps_n = WARPS / WARPS_M;
    static constexpr int warp_m = BLOCK_M / WARPS_M;
    static constexpr int warp_n = BLOCK_N / warps_n;
    static constexpr int m_tiles = warp_m / 16;
    static constexpr int n_tiles = warp_n / 8;
    static constexpr int a_bytes = BLOCK_M * BLOCK_K;
    static constexpr int stage_bytes = (BLOCK_M + BLOCK_N) * BLOCK_K;
    // The stages, alignment slack for the swizzle, and a barrier and release count per stage.
    static constexpr int shared_bytes = STAGES * stage_bytes + 1024 + STAGES * 16;
    static_assert(BLOCK_M <= 256 && BLOCK_N <= 256, "one TMA box per operand and stage");
    static_assert(warp_m % 16 == 0 && warp_n % 32 == 0,
                  "warp tiles must hold whole groups of four MMA tiles");
};

__device__ __forceinline__ void mma(int32_t (&accumulator)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1) {
    mma_s8(accumulator, a, b0, b1);
}

__device__ __forceinline__ void mma(float (&accumulator)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                 "%6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
                   "+f"(accumulator[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Operands described to TMA. The adapter maps cover the low-rank adapter's down activations [m,
// rank] and up weights [n, rank] in BF16 and repeat the INT8 maps when there is no adapter.
struct TensorMaps {
    CUtensorMap activations;
    CUtensorMap weights;
    CUtensorMap adapter_down;
    CUtensorMap adapter_up;
};

// The same operands for the cp.async path, with the adapter's rank and the stride of its down rows.
struct Operands {
    const int8_t *activations;
    const int8_t *weights;
    const __nv_bfloat16 *adapter_down;
    const __nv_bfloat16 *adapter_up;
    int rank;
    int down_stride;
};

struct TileGrid {
    int tiles_m;
    int tiles_n;
    int tiles;
    int k_blocks;
    // K blocks followed by adapter blocks of 64 ranks, one stage each.
    int blocks_per_tile;

    // Tiles go down GROUP_M rows of tiles before moving right, so the CTAs in flight share rows of
    // activations and columns of weights in L2.
    __device__ void coordinates(int tile, int &tile_m, int &tile_n) const {
        const int tiles_per_group = GROUP_M * tiles_n;
        const int first_tile_m = tile / tiles_per_group * GROUP_M;
        const int group_height = min(tiles_m - first_tile_m, GROUP_M);
        tile_m = first_tile_m + tile % tiles_per_group % group_height;
        tile_n = tile % tiles_per_group / group_height;
    }
};

// Fills a stage with this CTA's `sequence`-th block over all of its tiles: a K block of the INT8
// operands, or a block of 64 ranks of the adapter operands, which take the same 128 bytes per row.
template <typename Config>
__device__ void fill_stage(const TileGrid &grid, int sequence, uint32_t stage, uint32_t barrier,
                           const TensorMaps &maps) {
    const int tile = blockIdx.x + sequence / grid.blocks_per_tile * gridDim.x;
    if (tile >= grid.tiles) {
        return;
    }
    const int block = sequence % grid.blocks_per_tile;
    int tile_m, tile_n;
    grid.coordinates(tile, tile_m, tile_n);
    barrier_expect_bytes(barrier, Config::stage_bytes);
    if (block < grid.k_blocks) {
        copy_tile(stage, &maps.activations, barrier, block * BLOCK_K, tile_m * Config::block_m);
        copy_tile(stage + Config::a_bytes, &maps.weights, barrier, block * BLOCK_K,
                  tile_n * Config::block_n);
    } else {
        const int rank = (block - grid.k_blocks) * BLOCK_K / 2;
        copy_tile(stage, &maps.adapter_down, barrier, rank, tile_m * Config::block_m);
        copy_tile(stage + Config::a_bytes, &maps.adapter_up, barrier, rank,
                  tile_n * Config::block_n);
    }
}

// fill_stage with cp.async from every thread of the CTA, into the same swizzled layout TMA writes.
// Activation rows past `m` are zero-filled, as TMA fills them.
template <typename Config>
__device__ void copy_stage(const TileGrid &grid, int sequence, uint32_t stage,
                           const Operands &operands, int m, int k) {
    const int tile = blockIdx.x + sequence / grid.blocks_per_tile * gridDim.x;
    if (tile >= grid.tiles) {
        return;
    }
    const int block = sequence % grid.blocks_per_tile;
    int tile_m, tile_n;
    grid.coordinates(tile, tile_m, tile_n);
    const uint8_t *a_base;
    const uint8_t *b_base;
    int64_t a_stride, b_stride;
    if (block < grid.k_blocks) {
        a_base = reinterpret_cast<const uint8_t *>(operands.activations) + block * BLOCK_K;
        b_base = reinterpret_cast<const uint8_t *>(operands.weights) + block * BLOCK_K;
        a_stride = k;
        b_stride = k;
    } else {
        const int rank = (block - grid.k_blocks) * BLOCK_K / 2;
        a_base = reinterpret_cast<const uint8_t *>(operands.adapter_down + rank);
        b_base = reinterpret_cast<const uint8_t *>(operands.adapter_up + rank);
        a_stride = static_cast<int64_t>(operands.down_stride) * 2;
        b_stride = static_cast<int64_t>(operands.rank) * 2;
    }
    constexpr int CHUNKS_PER_ROW = BLOCK_K / 16;
    constexpr int CHUNKS = (Config::block_m + Config::block_n) * CHUNKS_PER_ROW;
    for (int index = threadIdx.x; index < CHUNKS; index += WARPS * 32) {
        const int row = index / CHUNKS_PER_ROW;
        const int chunk = index % CHUNKS_PER_ROW;
        if (row < Config::block_m) {
            const int source_row = tile_m * Config::block_m + row;
            const bool valid = source_row < m;
            copy_async_16(stage + swizzled_offset(row, chunk),
                          a_base + (valid ? source_row * a_stride : 0) + chunk * 16, valid);
        } else {
            const int b_row = row - Config::block_m;
            copy_async_16(stage + Config::a_bytes + swizzled_offset(b_row, chunk),
                          b_base + (tile_n * Config::block_n + b_row) * b_stride + chunk * 16,
                          true);
        }
    }
}

// Multiplies one stage into the warp's accumulators: INT8 m16n8k32 MMAs for int32 accumulators,
// BF16 m16n8k16 MMAs for float ones. Both read 32 bytes of K per step, so the fragments load the
// same way.
template <typename Config, typename Accumulator>
__device__ __forceinline__ void
multiply_stage(Accumulator (&accumulators)[Config::m_tiles][Config::n_tiles][4], uint32_t stage_a,
               int warp_row, int warp_column, int lane) {
    const uint32_t stage_b = stage_a + Config::a_bytes;
    const int matrix = lane / 8;
    const int matrix_row = lane % 8;
    #pragma unroll
    for (int k_step = 0; k_step < BLOCK_K / 32; k_step++) {
        uint32_t a_fragments[Config::m_tiles][4];
        uint32_t b_fragments[Config::n_tiles][2];
        #pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
            const int row = warp_row + m_tile * 16 + (matrix % 2) * 8 + matrix_row;
            load_matrix_x4(a_fragments[m_tile],
                           stage_a + swizzled_offset(row, k_step * 2 + matrix / 2));
        }
        #pragma unroll
        for (int n_pair = 0; n_pair < Config::n_tiles / 2; n_pair++) {
            // NOTE: Within each group of 32 weight rows, MMA column 2t + e of n-tile j reads row
            // 8t + 2((j + t) % 4) + e. Lane t of every quad then holds eight adjacent output
            // columns for a 16-byte store, and the eight rows of each ldmatrix phase still differ
            // modulo 8.
            const int n_tile = n_pair * 2 + matrix / 2;
            const int row_quad_lane = matrix_row / 2;
            const int row = warp_column + n_tile / 4 * 32 + row_quad_lane * 8 +
                            ((n_tile + row_quad_lane) & 3) * 2 + matrix_row % 2;
            uint32_t registers[4];
            load_matrix_x4(registers, stage_b + swizzled_offset(row, k_step * 2 + matrix % 2));
            b_fragments[n_pair * 2][0] = registers[0];
            b_fragments[n_pair * 2][1] = registers[1];
            b_fragments[n_pair * 2 + 1][0] = registers[2];
            b_fragments[n_pair * 2 + 1][1] = registers[3];
        }
        #pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
            #pragma unroll
            for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
                mma(accumulators[m_tile][n_tile], a_fragments[m_tile], b_fragments[n_tile][0],
                    b_fragments[n_tile][1]);
            }
        }
    }
}

__device__ __forceinline__ uint32_t pack(__nv_bfloat16 *, float low, float high) {
    __nv_bfloat162 value = __floats2bfloat162_rn(low, high);
    return *reinterpret_cast<uint32_t *>(&value);
}

__device__ __forceinline__ uint32_t pack(__half *, float low, float high) {
    __half2 value = __floats2half2_rn(low, high);
    return *reinterpret_cast<uint32_t *>(&value);
}

__device__ __forceinline__ float2 unpack(__nv_bfloat16 *, uint32_t word) {
    return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&word));
}

__device__ __forceinline__ float2 unpack(__half *, uint32_t word) {
    return __half22float2(*reinterpret_cast<__half2 *>(&word));
}

// silu(gate) · up of two features, rounded like a separate SwiGLU pass over the rounded products.
template <typename Output>
__device__ __forceinline__ uint32_t swiglu_pair(Output *output, uint32_t gate_word,
                                                uint32_t up_word) {
    const float2 gate = unpack(output, gate_word);
    const float2 up = unpack(output, up_word);
    return pack(output, gate.x / (1.0f + __expf(-gate.x)) * up.x,
                gate.y / (1.0f + __expf(-gate.y)) * up.y);
}

// Ring position of a warp, shared by the INT8 and adapter blocks.
struct Ring {
    int stage = 0;
    uint32_t phase = 0;
    int sequence = 0;

    __device__ void advance() {
        sequence++;
        if (++stage == STAGES) {
            stage = 0;
            phase ^= 1;
        }
    }
};

template <typename Config, typename Output, bool SWIGLU, bool TMA>
__global__ void __launch_bounds__(WARPS * 32, 1)
    int8_gemm_kernel(const __grid_constant__ TensorMaps maps, const Operands operands,
                     const float *__restrict__ activation_scales,
                     const float *__restrict__ weight_scales, const float *__restrict__ bias,
                     Output *__restrict__ output, int m, int n, int k, int output_stride,
                     int adapter_blocks, float adapter_scale) {
    extern __shared__ __align__(1024) uint8_t shared_memory[];
    const uint32_t shared_base = (shared_address(shared_memory) + 1023) & ~1023u;
    const uint32_t barriers = shared_base + STAGES * Config::stage_bytes;
    int *release_counts = reinterpret_cast<int *>(
        shared_memory + (barriers - shared_address(shared_memory)) + STAGES * 8);

    TileGrid grid;
    grid.tiles_m = (m + Config::block_m - 1) / Config::block_m;
    grid.tiles_n = n / Config::block_n;
    grid.tiles = grid.tiles_m * grid.tiles_n;
    grid.k_blocks = k / BLOCK_K;
    grid.blocks_per_tile = grid.k_blocks + adapter_blocks;

    if constexpr (TMA) {
        if (threadIdx.x == 0) {
            for (int stage = 0; stage < STAGES; stage++) {
                barrier_init(barriers + stage * 8, 1);
                release_counts[stage] = 0;
            }
            barrier_init_fence();
            for (int stage = 0; stage < STAGES; stage++) {
                fill_stage<Config>(grid, stage, shared_base + stage * Config::stage_bytes,
                                   barriers + stage * 8, maps);
            }
        }
        __syncthreads();
    } else {
        // Every stage but the last starts filling. Each consume then fills the one after them.
        for (int stage = 0; stage < STAGES - 1; stage++) {
            copy_stage<Config>(grid, stage, shared_base + stage * Config::stage_bytes, operands, m,
                               k);
            copy_async_commit();
        }
    }

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warp_row = warp / Config::warps_n * Config::warp_m;
    const int warp_column = warp % Config::warps_n * Config::warp_n;
    // Position of the lane in the MMA accumulator layout.
    const int group_id = lane / 4;
    const int quad_lane = lane % 4;

    // Waits for the ring's current stage, multiplies it and releases it. With TMA, the warp that
    // releases it last refills it. Without, the CTA starts filling the stage that follows the
    // ones in flight before it waits for the current one.
    Ring ring;
    auto consume = [&](auto &accumulators) {
        const uint32_t stage_a = shared_base + ring.stage * Config::stage_bytes;
        if constexpr (TMA) {
            const uint32_t barrier = barriers + ring.stage * 8;
            barrier_wait(barrier, ring.phase);
            multiply_stage<Config>(accumulators, stage_a, warp_row, warp_column, lane);
            __syncwarp();
            if (lane == 0) {
                __threadfence_block();
                if (atomicAdd(release_counts + ring.stage, 1) == WARPS - 1) {
                    release_counts[ring.stage] = 0;
                    async_proxy_fence();
                    fill_stage<Config>(grid, ring.sequence + STAGES, stage_a, barrier, maps);
                }
            }
        } else {
            const int next_stage = (ring.stage + STAGES - 1) % STAGES;
            copy_stage<Config>(grid, ring.sequence + STAGES - 1,
                               shared_base + next_stage * Config::stage_bytes, operands, m, k);
            copy_async_commit();
            copy_async_wait<STAGES - 1>();
            __syncthreads();
            multiply_stage<Config>(accumulators, stage_a, warp_row, warp_column, lane);
            // The next consume refills this stage.
            __syncthreads();
        }
        ring.advance();
    };

    for (int tile = blockIdx.x; tile < grid.tiles; tile += gridDim.x) {
        int tile_m, tile_n;
        grid.coordinates(tile, tile_m, tile_n);
        int32_t accumulators[Config::m_tiles][Config::n_tiles][4];
        #pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
            #pragma unroll
            for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
                #pragma unroll
                for (int index = 0; index < 4; index++) {
                    accumulators[m_tile][n_tile][index] = 0;
                }
            }
        }
        for (int k_block = 0; k_block < grid.k_blocks; k_block++) {
            consume(accumulators);
        }

        // Scale the INT8 products. Slot j of each quad of n-tiles holds output columns 2((j + t) %
        // 4) and the one after among the lane's eight.
        float values[Config::m_tiles][Config::n_tiles][4];
        float row_scales[Config::m_tiles][2];
        #pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const int row =
                    tile_m * Config::block_m + warp_row + m_tile * 16 + half * 8 + group_id;
                row_scales[m_tile][half] = row < m ? activation_scales[row] : 0.0f;
            }
        }
        #pragma unroll
        for (int quad = 0; quad < Config::n_tiles / 4; quad++) {
            const int column = tile_n * Config::block_n + warp_column + quad * 32 + quad_lane * 8;
            #pragma unroll
            for (int slot = 0; slot < 4; slot++) {
                const float2 column_scale = *reinterpret_cast<const float2 *>(
                    weight_scales + column + ((slot + quad_lane) & 3) * 2);
                #pragma unroll
                for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        const int32_t *products = accumulators[m_tile][quad * 4 + slot] + half * 2;
                        float *scaled = values[m_tile][quad * 4 + slot] + half * 2;
                        scaled[0] = static_cast<float>(products[0]) * row_scales[m_tile][half] *
                                    column_scale.x;
                        scaled[1] = static_cast<float>(products[1]) * row_scales[m_tile][half] *
                                    column_scale.y;
                    }
                }
            }
        }

        // Add adapter_scale · down · upᵀ in FP32, accumulating the products at the scale of the
        // adapter.
        if (adapter_blocks > 0) {
            const float inverse_scale = 1.0f / adapter_scale;
            #pragma unroll
            for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
                #pragma unroll
                for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
                    #pragma unroll
                    for (int index = 0; index < 4; index++) {
                        values[m_tile][n_tile][index] *= inverse_scale;
                    }
                }
            }
            for (int block = 0; block < adapter_blocks; block++) {
                consume(values);
            }
            #pragma unroll
            for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
                #pragma unroll
                for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
                    #pragma unroll
                    for (int index = 0; index < 4; index++) {
                        values[m_tile][n_tile][index] *= adapter_scale;
                    }
                }
            }
        }

        if (bias != nullptr) {
            #pragma unroll
            for (int quad = 0; quad < Config::n_tiles / 4; quad++) {
                const int column =
                    tile_n * Config::block_n + warp_column + quad * 32 + quad_lane * 8;
                #pragma unroll
                for (int slot = 0; slot < 4; slot++) {
                    const float2 column_bias = *reinterpret_cast<const float2 *>(
                        bias + column + ((slot + quad_lane) & 3) * 2);
                    #pragma unroll
                    for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
                        #pragma unroll
                        for (int half = 0; half < 2; half++) {
                            values[m_tile][quad * 4 + slot][half * 2] += column_bias.x;
                            values[m_tile][quad * 4 + slot][half * 2 + 1] += column_bias.y;
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int quad = 0; quad < Config::n_tiles / 4; quad++) {
            const int column = tile_n * Config::block_n + warp_column + quad * 32 + quad_lane * 8;
            #pragma unroll
            for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    uint32_t words[4];
                    #pragma unroll
                    for (int slot = 0; slot < 4; slot++) {
                        const float *scaled = values[m_tile][quad * 4 + slot] + half * 2;
                        words[slot] = pack(output, scaled[0], scaled[1]);
                    }
                    // Rotate the slots right by t to put the columns in order.
                    if (quad_lane & 1) {
                        const uint32_t last = words[3];
                        words[3] = words[2];
                        words[2] = words[1];
                        words[1] = words[0];
                        words[0] = last;
                    }
                    if (quad_lane & 2) {
                        uint32_t swapped = words[0];
                        words[0] = words[2];
                        words[2] = swapped;
                        swapped = words[1];
                        words[1] = words[3];
                        words[3] = swapped;
                    }
                    const int row =
                        tile_m * Config::block_m + warp_row + m_tile * 16 + half * 8 + group_id;
                    if (row < m) {
                        if constexpr (SWIGLU) {
                            // The lane's eight columns hold the gates of features column / 2 to
                            // column / 2 + 3, then their up projections.
                            *reinterpret_cast<uint2 *>(
                                output + static_cast<size_t>(row) * output_stride + column / 2) =
                                make_uint2(swiglu_pair(output, words[0], words[2]),
                                           swiglu_pair(output, words[1], words[3]));
                        } else {
                            *reinterpret_cast<uint4 *>(
                                output + static_cast<size_t>(row) * output_stride + column) =
                                make_uint4(words[0], words[1], words[2], words[3]);
                        }
                    }
                }
            }
        }
    }
    if constexpr (!TMA) {
        copy_async_wait<0>();
    }
}

using TallTile = Int8GemmConfig<256, 128, 4>;
using WideTile = Int8GemmConfig<128, 256, 4>;

// Describes `rows` rows of `columns` elements to TMA in boxes of `box_rows` rows by 128 bytes,
// zero-filling rows past the end.
// A [rows, columns] matrix whose rows lie row_stride elements apart.
bool encode_tensor_map(CUtensorMap *map, const void *base, CUtensorMapDataType type,
                       int element_bytes, int rows, int columns, int box_rows, int row_stride) {
    PFN_cuTensorMapEncodeTiled_v12000 encoder = tensor_map_encoder();
    if (encoder == nullptr) {
        return false;
    }
    const cuuint64_t dimensions[2] = {static_cast<cuuint64_t>(columns),
                                      static_cast<cuuint64_t>(rows)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_stride) * element_bytes};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(BLOCK_K / element_bytes),
                               static_cast<cuuint32_t>(box_rows)};
    const cuuint32_t element_strides[2] = {1, 1};
    return encoder(map, type, 2, const_cast<void *>(base), dimensions, strides, box,
                   element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// A low-rank adapter to add to the product, or none when `down` is null. The rows of `down` lie
// `down_stride` elements apart.
struct Adapter {
    const __nv_bfloat16 *down;
    const __nv_bfloat16 *up;
    int rank;
    int down_stride;
    float scale;
};

template <typename Config, typename Output, bool SWIGLU>
int launch(const int8_t *activations, const int8_t *weights, const float *activation_scales,
           const float *weight_scales, const float *bias, Output *output, int m, int n, int k,
           int output_stride, Adapter adapter, cudaStream_t stream) {
    const bool has_adapter =
        adapter.down != nullptr && adapter.up != nullptr && adapter.scale != 0.0f;
    const int adapter_blocks = has_adapter ? adapter.rank * 2 / BLOCK_K : 0;
    if (n % Config::block_n != 0 || k % BLOCK_K != 0 || m <= 0 ||
        (has_adapter && (adapter.rank <= 0 || adapter.rank * 2 % BLOCK_K != 0 ||
                         adapter.down_stride < adapter.rank || adapter.down_stride % 8 != 0))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const Operands operands{activations, weights,      adapter.down,
                            adapter.up,  adapter.rank, adapter.down_stride};
    const bool tma = mmh3_tma_available();
    TensorMaps maps = {};
    if (tma) {
        if (!encode_tensor_map(&maps.activations, activations, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, m,
                               k, Config::block_m, k) ||
            !encode_tensor_map(&maps.weights, weights, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, n, k,
                               Config::block_n, k)) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        if (has_adapter) {
            if (!encode_tensor_map(&maps.adapter_down, adapter.down,
                                   CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, m, adapter.rank,
                                   Config::block_m, adapter.down_stride) ||
                !encode_tensor_map(&maps.adapter_up, adapter.up, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                   2, n, adapter.rank, Config::block_n, adapter.rank)) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
        } else {
            maps.adapter_down = maps.activations;
            maps.adapter_up = maps.weights;
        }
    } else if (reinterpret_cast<uintptr_t>(activations) % 16 != 0 ||
               reinterpret_cast<uintptr_t>(weights) % 16 != 0 ||
               (has_adapter && (reinterpret_cast<uintptr_t>(adapter.down) % 16 != 0 ||
                                reinterpret_cast<uintptr_t>(adapter.up) % 16 != 0))) {
        // cp.async copies 16 bytes from 16-byte aligned addresses, as TMA needs its bases.
        return static_cast<int>(cudaErrorInvalidValue);
    }
    auto kernel = tma ? int8_gemm_kernel<Config, Output, SWIGLU, true>
                      : int8_gemm_kernel<Config, Output, SWIGLU, false>;
    static Mmh3SharedMemory shared_memory[2];
    const cudaError_t configured =
        mmh3_configure_shared_memory(shared_memory[tma], kernel, Config::shared_bytes);
    if (configured != cudaSuccess) {
        return static_cast<int>(configured);
    }
    const int tiles = (m + Config::block_m - 1) / Config::block_m * (n / Config::block_n);
    const int processors = mmh3_multiprocessor_count();
    if (processors == 0) {
        return static_cast<int>(cudaErrorInvalidDevice);
    }
    const int blocks = tiles < processors ? tiles : processors;
    kernel<<<blocks, WARPS * 32, Config::shared_bytes, stream>>>(
        maps, operands, activation_scales, weight_scales, bias, output, m, n, k, output_stride,
        adapter_blocks, adapter.scale);
    return static_cast<int>(cudaGetLastError());
}

template <typename Output, bool SWIGLU>
int launch_config(int config, const int8_t *activations, const int8_t *weights,
                  const float *activation_scales, const float *weight_scales, const float *bias,
                  Output *output, int m, int n, int k, int output_stride, Adapter adapter,
                  cudaStream_t stream) {
    switch (config) {
    case 0:
        return launch<TallTile, Output, SWIGLU>(activations, weights, activation_scales,
                                                weight_scales, bias, output, m, n, k, output_stride,
                                                adapter, stream);
    case 1:
        return launch<WideTile, Output, SWIGLU>(activations, weights, activation_scales,
                                                weight_scales, bias, output, m, n, k, output_stride,
                                                adapter, stream);
    default:
        return static_cast<int>(cudaErrorInvalidValue);
    }
}

template <typename Output>
int launch_output(int config, const int8_t *activations, const int8_t *weights,
                  const float *activation_scales, const float *weight_scales, const float *bias,
                  Output *output, bool swiglu, int m, int n, int k, int output_stride,
                  Adapter adapter, cudaStream_t stream) {
    // Zero means the rows sit side by side, which is every caller but a shared-out step writing one
    // rank's heads into a row that holds every rank's.
    if (swiglu) {
        return launch_config<Output, true>(
            config, activations, weights, activation_scales, weight_scales, bias, output, m, n, k,
            output_stride > 0 ? output_stride : n / 2, adapter, stream);
    }
    return launch_config<Output, false>(config, activations, weights, activation_scales,
                                        weight_scales, bias, output, m, n, k,
                                        output_stride > 0 ? output_stride : n, adapter, stream);
}

// Row r of the interleaved matrix is row 4g + p of the gates for p < 4 and row 4g + p - 4 of the up
// projections otherwise, with g = r / 8 and p = r % 8.
__global__ void interleave_swiglu_rows_kernel(const uint32_t *__restrict__ source,
                                              uint32_t *__restrict__ destination, int rows,
                                              int row_words) {
    const int row = blockIdx.x;
    const int group = row / 8;
    const int position = row % 8;
    const int source_row =
        position < 4 ? group * 4 + position : rows / 2 + group * 4 + position - 4;
    for (int index = threadIdx.x; index < row_words; index += blockDim.x) {
        destination[static_cast<size_t>(row) * row_words + index] =
            source[static_cast<size_t>(source_row) * row_words + index];
    }
}

} // namespace

extern "C" int mmh3_int8_gemm_config_count() { return 2; }

// Config 0 uses 256 × 128 tiles and needs N to be a multiple of 128. Config 1 uses 128 × 256 tiles,
// which waste less work on short inputs, and needs N to be a multiple of 256.
extern "C" int mmh3_int8_gemm_bf16(int config, const int8_t *activations, const int8_t *weights,
                                   const float *activation_scales, const float *weight_scales,
                                   __nv_bfloat16 *output, int m, int n, int k,
                                   cudaStream_t stream) {
    return launch_config<__nv_bfloat16, false>(config, activations, weights, activation_scales,
                                               weight_scales, nullptr, output, m, n, k, n,
                                               Adapter{nullptr, nullptr, 0, 0, 0.0f}, stream);
}

// The same product with an optional FP32 bias [n], BF16 or FP16 output, and an optional adapter
// adapter_scale · adapter_down · adapter_upᵀ with adapter_down [m, rank], its rows
// adapter_down_stride elements apart (a multiple of 8), and adapter_up [n, rank] in BF16 and the
// rank a multiple of 64. The result is rounded once. With `swiglu`, the output is silu(gate) · up,
// [m, n / 2], for weights, weight scales, bias and adapter_up whose rows went through
// mmh3_interleave_swiglu_rows.
extern "C" int mmh3_int8_gemm(int config, const int8_t *activations, const int8_t *weights,
                              const float *activation_scales, const float *weight_scales,
                              const float *bias, void *output, int output_is_f16, int swiglu, int m,
                              int n, int k, int output_stride, const __nv_bfloat16 *adapter_down,
                              const __nv_bfloat16 *adapter_up, int rank, int adapter_down_stride,
                              float adapter_scale, cudaStream_t stream) {
    const Adapter adapter{adapter_down, adapter_up, rank, adapter_down_stride, adapter_scale};
    if (output_is_f16) {
        return launch_output(config, activations, weights, activation_scales, weight_scales, bias,
                             static_cast<__half *>(output), swiglu != 0, m, n, k, output_stride,
                             adapter, stream);
    }
    return launch_output(config, activations, weights, activation_scales, weight_scales, bias,
                         static_cast<__nv_bfloat16 *>(output), swiglu != 0, m, n, k, output_stride,
                         adapter, stream);
}

// Reorders the rows of a [rows, row_bytes] matrix whose first half are SwiGLU gates and second half
// the matching up projections, so that each group of eight rows holds the gates of four features
// followed by their up projections.
extern "C" int mmh3_interleave_swiglu_rows(const void *source, void *destination, int rows,
                                           int row_bytes, cudaStream_t stream) {
    if (rows % 8 != 0 || row_bytes % 4 != 0 || rows <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    interleave_swiglu_rows_kernel<<<rows, 128, 0, stream>>>(static_cast<const uint32_t *>(source),
                                                            static_cast<uint32_t *>(destination),
                                                            rows, row_bytes / 4);
    return static_cast<int>(cudaGetLastError());
}
