#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "device.cuh"
#include "tensor_core.cuh"
#include "tma.cuh"
#include "wgmma.cuh"

// Dense attention in the FlashAttention-2 style for head dimensions 64 and 128, BF16 or FP16,
// bidirectional or causal, with grouped key and value heads. The strided layout covers the DiT's
// fused [q | k | v] projection, the VAE decoder's per-head [q, k, v] layout and the text encoder's
// [q | k | v] with fewer key and value heads alike.
//
// On Hopper (sm_90) warpgroups multiply the same tiles with wgmma, see attention_hopper_kernel.
//
// NOTE: the tiles arrive through TMA, which describes each of them as a box of a tensor map and
// leaves the addresses, the bounds and the swizzle to the copy engine. Devices without TMA (sm_89)
// and layouts TMA cannot describe, such as strides that are not multiples of 16 bytes, fall back to
// per-thread cp.async copies.

namespace {

enum Operand { QUERY = 0, KEY = 1, VALUE = 2, OUTPUT = 3 };

constexpr int BLOCK_M = 128;
constexpr int WARPS = BLOCK_M / 16;
constexpr int THREADS = WARPS * 32;
// The query tile borrows one K/V stage, so it must fit there.
static_assert(BLOCK_M * Tiles<64>::row_bytes <= Tiles<64>::stage_bytes,
              "the query tile must fit in one stage");
static_assert(BLOCK_M * Tiles<128>::row_bytes <= Tiles<128>::stage_bytes,
              "the query tile must fit in one stage");

// Elements of the 128-byte box that TMA fills from one row of the head dimension.
template <typename Element> constexpr int BOX_COLUMNS = 128 / sizeof(Element);

// Where row `row`, 16-byte chunk `chunk` of a tile of `ROWS` rows sits. TMA fills one box per
// 128-byte column slab, each slab swizzled on its own, while cp.async writes whole rows.
template <int HEAD_DIM, int ROWS, bool TMA>
__device__ __forceinline__ int tile_offset(int row, int chunk) {
    if constexpr (TMA) {
        return (chunk >> 3) * (ROWS * 128) + swizzled_offset(row, chunk);
    } else {
        return Tiles<HEAD_DIM>::offset(row, chunk);
    }
}

// What the VAE decoder's query takes between its projection and the attention: an RMS norm over the
// head dimension and a RoPE from prepared angles, [tokens, pairs]. Null angles leave the tile as it
// arrives, which is what the DiT and the text encoder want, since they normalize their own.
//
// NOTE: only the query, which belongs to one block. Every block reads all of the keys, so a key
// normalized here would be normalized once per block of queries, and the separate pass over the
// keys costs less than that.
struct AttentionRope {
    const float *angles;
    int pairs;
    float epsilon;
};

// Normalizes and rotates `rows` rows of a tile in place, one of WARP_COUNT warps per row, `warp`
// being this one, in the order and the precision of the separate pass it replaces. Rows past the
// last token stay as the copy left them.
template <typename Element, int HEAD_DIM, int ROWS, bool TMA, int WARP_COUNT>
__device__ void normalize_rotate(uint32_t tile, int first_token, int rows, int tokens,
                                 const AttentionRope &rope, float *scratch, int warp) {
    using N = Numeric<Element>;
    const int lane = threadIdx.x % 32;
    for (int row = warp; row < rows; row += WARP_COUNT) {
        if (first_token + row >= tokens) {
            continue;
        }
        float *values = scratch + warp * HEAD_DIM;
        uint32_t addresses[HEAD_DIM / 32];
        float squares = 0.0f;
        #pragma unroll
        for (int slot = 0; slot < HEAD_DIM / 32; slot++) {
            const int index = slot * 32 + lane;
            addresses[slot] =
                tile + tile_offset<HEAD_DIM, ROWS, TMA>(row, index / 8) + (index % 8) * 2;
            const float value = N::to_float(load_shared_16(addresses[slot]));
            values[index] = value;
            squares += value * value;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            squares += __shfl_xor_sync(0xffffffff, squares, offset);
        }
        const float inverse = rsqrtf(squares / HEAD_DIM + rope.epsilon);
        #pragma unroll
        for (int slot = 0; slot < HEAD_DIM / 32; slot++) {
            values[slot * 32 + lane] *= inverse;
        }
        __syncwarp();
        const float *angles = rope.angles + static_cast<int64_t>(first_token + row) * rope.pairs;
        for (int pair = lane; pair < rope.pairs; pair += 32) {
            float sine, cosine;
            sincosf(angles[pair], &sine, &cosine);
            const float first = values[pair];
            const float second = values[pair + rope.pairs];
            values[pair] = first * cosine - second * sine;
            values[pair + rope.pairs] = first * sine + second * cosine;
        }
        __syncwarp();
        #pragma unroll
        for (int slot = 0; slot < HEAD_DIM / 32; slot++) {
            store_shared_16(addresses[slot], N::from_float(values[slot * 32 + lane]));
        }
    }
}

// The query, key and value tensors as TMA sees them: [head dimension, tokens, heads, batch].
struct AttentionMaps {
    CUtensorMap query;
    CUtensorMap key;
    CUtensorMap value;
};

// Head dimension 64 asks for two blocks per SM: its stages are half the size, so two fit in shared
// memory and one block's MMAs cover the other's copies.
template <typename Element, int HEAD_DIM, bool TMA, bool ROPE>
__global__ void __launch_bounds__(THREADS, HEAD_DIM == 64 ? 2 : 1)
    attention_kernel(const Element *__restrict__ query, const Element *__restrict__ key,
                     const Element *__restrict__ value, Element *__restrict__ output, int tokens,
                     Mmh3AttentionLayout layout, float scale_log2,
                     const __grid_constant__ AttentionMaps maps, AttentionRope rope) {
    __shared__ float rope_scratch[ROPE ? WARPS * HEAD_DIM : 1];
    using T = Tiles<HEAD_DIM>;
    using N = Numeric<Element>;
    constexpr int SLABS = HEAD_DIM * sizeof(Element) / 128;
    constexpr int COLUMNS = BOX_COLUMNS<Element>;
    extern __shared__ __align__(128) uint8_t shared_memory[];
    const uint32_t shared_base = (shared_address(shared_memory) + 127) & ~127u;
    // Two stages of key and value tiles, then one barrier per stage and one for the query tile.
    const uint32_t barriers = shared_base + 2 * T::stage_bytes;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int first_query = blockIdx.x * BLOCK_M;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int matrix = lane / 8;
    const int matrix_row = lane % 8;

    const int key_value_head = head / layout.heads_per_key_value;
    auto head_base = [&](const Element *base, Operand operand) {
        const int operand_head = operand == KEY || operand == VALUE ? key_value_head : head;
        return base + batch * layout.batch_stride[operand] +
               operand_head * layout.head_stride[operand];
    };
    const Element *query_head = head_base(query, QUERY);
    const Element *key_head = head_base(key, KEY);
    const Element *value_head = head_base(value, VALUE);

    auto stage_address = [&](int stage) { return shared_base + stage * T::stage_bytes; };
    auto load_key_value_block = [&](int block, int stage) {
        if constexpr (TMA) {
            const uint32_t barrier = barriers + stage * 8;
            barrier_expect_bytes(barrier, 2 * T::tile_bytes);
            constexpr int slab_bytes = BLOCK_N * 128;
            #pragma unroll
            for (int slab = 0; slab < SLABS; slab++) {
                copy_tile(stage_address(stage) + slab * slab_bytes, &maps.key, barrier,
                          slab * COLUMNS, block * BLOCK_N, key_value_head, batch);
                copy_tile(stage_address(stage) + T::tile_bytes + slab * slab_bytes, &maps.value,
                          barrier, slab * COLUMNS, block * BLOCK_N, key_value_head, batch);
            }
        } else {
            load_rows<HEAD_DIM, THREADS>(stage_address(stage), key_head, layout.token_stride[KEY],
                                         block * BLOCK_N, BLOCK_N, tokens);
            load_rows<HEAD_DIM, THREADS>(stage_address(stage) + T::tile_bytes, value_head,
                                         layout.token_stride[VALUE], block * BLOCK_N, BLOCK_N,
                                         tokens);
        }
    };

    int blocks = (tokens + BLOCK_N - 1) / BLOCK_N;
    if (layout.causal) {
        blocks = min(blocks, (first_query + BLOCK_M + BLOCK_N - 1) / BLOCK_N);
    }

    // The query tile borrows stage 1 until it has been copied into registers.
    if constexpr (TMA) {
        if (threadIdx.x == 0) {
            barrier_init(barriers, 1);
            barrier_init(barriers + 8, 1);
            barrier_init(barriers + 16, 1);
            barrier_init_fence();
            barrier_expect_bytes(barriers + 16, BLOCK_M * HEAD_DIM * sizeof(Element));
            #pragma unroll
            for (int slab = 0; slab < SLABS; slab++) {
                copy_tile(stage_address(1) + slab * BLOCK_M * 128, &maps.query, barriers + 16,
                          slab * COLUMNS, first_query, head, batch);
            }
            load_key_value_block(0, 0);
        }
        __syncthreads();
        barrier_wait(barriers + 16, 0);
    } else {
        load_rows<HEAD_DIM, THREADS>(stage_address(1), query_head, layout.token_stride[QUERY],
                                     first_query, BLOCK_M, tokens);
        copy_async_commit();
        load_key_value_block(0, 0);
        copy_async_commit();
        copy_async_wait<1>();
        __syncthreads();
    }

    if constexpr (ROPE) {
        normalize_rotate<Element, HEAD_DIM, BLOCK_M, TMA, WARPS>(
            stage_address(1), first_query, BLOCK_M, tokens, rope, rope_scratch, warp);
        __syncthreads();
    }

    uint32_t query_fragments[HEAD_DIM / 16][4];
    #pragma unroll
    for (int k_step = 0; k_step < HEAD_DIM / 16; k_step++) {
        int row = warp * 16 + (matrix % 2) * 8 + matrix_row;
        int chunk = k_step * 2 + matrix / 2;
        load_matrix_x4(query_fragments[k_step],
                       stage_address(1) + tile_offset<HEAD_DIM, BLOCK_M, TMA>(row, chunk));
    }
    __syncthreads();

    float output_accumulators[HEAD_DIM / 8][4] = {};
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {0.0f, 0.0f};

    for (int block = 0; block < blocks; block++) {
        if constexpr (TMA) {
            barrier_wait(barriers + (block % 2) * 8, (block / 2) & 1);
            if (threadIdx.x == 0 && block + 1 < blocks) {
                async_proxy_fence();
                load_key_value_block(block + 1, (block + 1) % 2);
            }
        } else {
            if (block + 1 < blocks) {
                load_key_value_block(block + 1, (block + 1) % 2);
            }
            copy_async_commit();
            copy_async_wait<1>();
            __syncthreads();
        }

        const uint32_t key_tile = stage_address(block % 2);
        const uint32_t value_tile = key_tile + T::tile_bytes;

        float scores[BLOCK_N / 8][4] = {};
        #pragma unroll
        for (int k_step = 0; k_step < HEAD_DIM / 16; k_step++) {
            #pragma unroll
            for (int key_pair = 0; key_pair < BLOCK_N / 16; key_pair++) {
                int row = key_pair * 16 + (matrix / 2) * 8 + matrix_row;
                int chunk = k_step * 2 + matrix % 2;
                uint32_t registers[4];
                load_matrix_x4(registers,
                               key_tile + tile_offset<HEAD_DIM, BLOCK_N, TMA>(row, chunk));
                N::mma(scores[key_pair * 2], query_fragments[k_step], registers[0], registers[1]);
                N::mma(scores[key_pair * 2 + 1], query_fragments[k_step], registers[2],
                       registers[3]);
            }
        }

        // NOTE: every query row sees key 0 in the first block, so a causal row never starts from an
        // all-masked block.
        const bool masked_block = (block + 1) * BLOCK_N > tokens ||
                                  (layout.causal && (block + 1) * BLOCK_N - 1 > first_query);
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                scores[key_tile_index][element] *= scale_log2;
            }
        }
        // NOTE: the masking is a branch of its own, which keeps the compiler from turning it into
        // selects that every block would pay for.
        if (masked_block) {
            #pragma unroll
            for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
                #pragma unroll
                for (int element = 0; element < 4; element++) {
                    const int key_index =
                        block * BLOCK_N + key_tile_index * 8 + (lane % 4) * 2 + (element % 2);
                    const int query_index = first_query + warp * 16 + (element / 2) * 8 + lane / 4;
                    if (key_index >= tokens || (layout.causal && key_index > query_index)) {
                        scores[key_tile_index][element] = -FLT_MAX;
                    }
                }
            }
        }
        float block_max[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                block_max[element / 2] =
                    fmaxf(block_max[element / 2], scores[key_tile_index][element]);
            }
        }
        float correction[2];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            block_max[half] =
                fmaxf(block_max[half], __shfl_xor_sync(0xffffffff, block_max[half], 1));
            block_max[half] =
                fmaxf(block_max[half], __shfl_xor_sync(0xffffffff, block_max[half], 2));
            float new_max = fmaxf(row_max[half], block_max[half]);
            correction[half] = exp2f(row_max[half] - new_max);
            row_max[half] = new_max;
            row_sum[half] *= correction[half];
        }
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                float probability = exp2f(scores[key_tile_index][element] - row_max[element / 2]);
                scores[key_tile_index][element] = probability;
                row_sum[element / 2] += probability;
            }
        }
        #pragma unroll
        for (int dimension_tile = 0; dimension_tile < HEAD_DIM / 8; dimension_tile++) {
            output_accumulators[dimension_tile][0] *= correction[0];
            output_accumulators[dimension_tile][1] *= correction[0];
            output_accumulators[dimension_tile][2] *= correction[1];
            output_accumulators[dimension_tile][3] *= correction[1];
        }

        #pragma unroll
        for (int key_step = 0; key_step < BLOCK_N / 16; key_step++) {
            uint32_t probability_fragment[4] = {
                N::pack(scores[key_step * 2][0], scores[key_step * 2][1]),
                N::pack(scores[key_step * 2][2], scores[key_step * 2][3]),
                N::pack(scores[key_step * 2 + 1][0], scores[key_step * 2 + 1][1]),
                N::pack(scores[key_step * 2 + 1][2], scores[key_step * 2 + 1][3]),
            };
            #pragma unroll
            for (int dimension_pair = 0; dimension_pair < HEAD_DIM / 16; dimension_pair++) {
                int row = key_step * 16 + (matrix % 2) * 8 + matrix_row;
                int chunk = dimension_pair * 2 + matrix / 2;
                uint32_t registers[4];
                load_matrix_x4_transposed(
                    registers, value_tile + tile_offset<HEAD_DIM, BLOCK_N, TMA>(row, chunk));
                N::mma(output_accumulators[dimension_pair * 2], probability_fragment, registers[0],
                       registers[1]);
                N::mma(output_accumulators[dimension_pair * 2 + 1], probability_fragment,
                       registers[2], registers[3]);
            }
        }
        __syncthreads();
    }
    if constexpr (!TMA) {
        copy_async_wait<0>();
    }

    #pragma unroll
    for (int half = 0; half < 2; half++) {
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 2);
    }
    const int group_id = lane / 4;
    Element *output_head =
        output + batch * layout.batch_stride[OUTPUT] + head * layout.head_stride[OUTPUT];
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int token = first_query + warp * 16 + half * 8 + group_id;
        if (token >= tokens) {
            continue;
        }
        const float inverse_sum = 1.0f / row_sum[half];
        Element *output_row =
            output_head + static_cast<int64_t>(token) * layout.token_stride[OUTPUT];
        #pragma unroll
        for (int dimension_tile = 0; dimension_tile < HEAD_DIM / 8; dimension_tile++) {
            const int dimension = dimension_tile * 8 + (lane % 4) * 2;
            *reinterpret_cast<uint32_t *>(output_row + dimension) =
                N::pack(output_accumulators[dimension_tile][half * 2] * inverse_sum,
                        output_accumulators[dimension_tile][half * 2 + 1] * inverse_sum);
        }
    }
}

// The Hopper kernel: warpgroup 0 copies the query tile once and the key and value tiles into a
// ring with TMA, and warpgroups 1 and 2 each own 64 of the block's queries, multiplying Q · Kᵀ and
// P · V with wgmma. The softmax, the masking and the output are the mma.sync kernel's, since the
// accumulators lie in the same layout.
constexpr int HOPPER_STAGES = 4;
constexpr int HOPPER_THREADS = 384;
constexpr int HOPPER_CONSUMER_THREADS = 256;
constexpr int HOPPER_PRODUCER_REGISTERS = 40;
constexpr int HOPPER_CONSUMER_REGISTERS = 232;

template <int HEAD_DIM> struct HopperTiles {
    static constexpr int query_bytes = BLOCK_M * HEAD_DIM * 2;
    static constexpr int stage_bytes = Tiles<HEAD_DIM>::stage_bytes;
    // The query tile, the stages, alignment slack, and the query barrier and a full and an empty
    // barrier per stage.
    static constexpr int shared_bytes =
        query_bytes + HOPPER_STAGES * stage_bytes + 1024 + (1 + 2 * HOPPER_STAGES) * 8;
};

template <typename Element> struct HopperForms;

template <> struct HopperForms<__nv_bfloat16> {
    __device__ static void scores(uint32_t (&d)[BLOCK_N / 2], uint64_t a, uint64_t b,
                                  int accumulate) {
        wgmma_bf16_m64n64k16(d, a, b, accumulate);
    }
    template <int HEAD_DIM>
    __device__ static void values(uint32_t (&d)[HEAD_DIM / 2], const uint32_t (&a)[4], uint64_t b) {
        if constexpr (HEAD_DIM == 128) {
            wgmma_bf16_m64n128k16_registers_transposed(d, a, b, 1);
        } else {
            wgmma_bf16_m64n64k16_registers_transposed(d, a, b, 1);
        }
    }
};

template <> struct HopperForms<__half> {
    __device__ static void scores(uint32_t (&d)[BLOCK_N / 2], uint64_t a, uint64_t b,
                                  int accumulate) {
        wgmma_f16_m64n64k16(d, a, b, accumulate);
    }
    template <int HEAD_DIM>
    __device__ static void values(uint32_t (&d)[HEAD_DIM / 2], const uint32_t (&a)[4], uint64_t b) {
        if constexpr (HEAD_DIM == 128) {
            wgmma_f16_m64n128k16_registers_transposed(d, a, b, 1);
        } else {
            wgmma_f16_m64n64k16_registers_transposed(d, a, b, 1);
        }
    }
};

// Waits for the threads of the multiplying warpgroups alone.
__device__ __forceinline__ void consumers_synchronize() {
    asm volatile("bar.sync 1, %0;\n" ::"n"(HOPPER_CONSUMER_THREADS) : "memory");
}

template <typename Element, int HEAD_DIM, bool ROPE>
__global__ void __launch_bounds__(HOPPER_THREADS, 1)
    attention_hopper_kernel(Element *__restrict__ output, int tokens, Mmh3AttentionLayout layout,
                            float scale_log2, const __grid_constant__ AttentionMaps maps,
                            AttentionRope rope) {
    __shared__ float rope_scratch[ROPE ? WARPS * HEAD_DIM : 1];
    using T = Tiles<HEAD_DIM>;
    using H = HopperTiles<HEAD_DIM>;
    using N = Numeric<Element>;
    using Forms = HopperForms<Element>;
    constexpr int SLABS = HEAD_DIM * sizeof(Element) / 128;
    constexpr int COLUMNS = BOX_COLUMNS<Element>;
    extern __shared__ __align__(128) uint8_t shared_memory[];
    const uint32_t shared_base = (shared_address(shared_memory) + 1023) & ~1023u;
    const uint32_t query_tile = shared_base;
    const uint32_t stages = shared_base + H::query_bytes;
    const uint32_t query_barrier = stages + HOPPER_STAGES * H::stage_bytes;
    const uint32_t full_barriers = query_barrier + 8;
    const uint32_t empty_barriers = full_barriers + HOPPER_STAGES * 8;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int first_query = blockIdx.x * BLOCK_M;
    const int key_value_head = head / layout.heads_per_key_value;

    int blocks = (tokens + BLOCK_N - 1) / BLOCK_N;
    if (layout.causal) {
        blocks = min(blocks, (first_query + BLOCK_M + BLOCK_N - 1) / BLOCK_N);
    }

    if (threadIdx.x == 0) {
        barrier_init(query_barrier, 1);
        for (int stage = 0; stage < HOPPER_STAGES; stage++) {
            barrier_init(full_barriers + stage * 8, 1);
            barrier_init(empty_barriers + stage * 8, HOPPER_CONSUMER_THREADS);
        }
        barrier_init_fence();
    }
    __syncthreads();

    const int warpgroup = threadIdx.x / 128;
    if (warpgroup == 0) {
        registers_decrease<HOPPER_PRODUCER_REGISTERS>();
        if (threadIdx.x == 0) {
            barrier_expect_bytes(query_barrier, BLOCK_M * HEAD_DIM * sizeof(Element));
            #pragma unroll
            for (int slab = 0; slab < SLABS; slab++) {
                copy_tile(query_tile + slab * BLOCK_M * 128, &maps.query, query_barrier,
                          slab * COLUMNS, first_query, head, batch);
            }
            for (int block = 0; block < blocks; block++) {
                const int stage = block % HOPPER_STAGES;
                // A stage's first fill waits for nothing, and every later one for both
                // multiplying warpgroups to have let go of it.
                if (block >= HOPPER_STAGES) {
                    barrier_wait(empty_barriers + stage * 8, (block / HOPPER_STAGES - 1) & 1);
                }
                const uint32_t barrier = full_barriers + stage * 8;
                const uint32_t key_tile = stages + stage * H::stage_bytes;
                barrier_expect_bytes(barrier, H::stage_bytes);
                #pragma unroll
                for (int slab = 0; slab < SLABS; slab++) {
                    copy_tile(key_tile + slab * BLOCK_N * 128, &maps.key, barrier, slab * COLUMNS,
                              block * BLOCK_N, key_value_head, batch);
                    copy_tile(key_tile + T::tile_bytes + slab * BLOCK_N * 128, &maps.value, barrier,
                              slab * COLUMNS, block * BLOCK_N, key_value_head, batch);
                }
            }
        }
        return;
    }

    registers_increase<HOPPER_CONSUMER_REGISTERS>();
    const int consumer = warpgroup - 1;
    // Warps 0 to 7 of the two multiplying warpgroups own 16 queries each, as the mma.sync kernel's
    // eight warps do.
    const int warp = threadIdx.x / 32 - 4;
    const int lane = threadIdx.x % 32;

    barrier_wait(query_barrier, 0);
    if constexpr (ROPE) {
        normalize_rotate<Element, HEAD_DIM, BLOCK_M, true, WARPS>(query_tile, first_query, BLOCK_M,
                                                                  tokens, rope, rope_scratch, warp);
        // The rotated queries reach wgmma through the async proxy.
        shared_to_async_proxy_fence();
        consumers_synchronize();
    }
    const uint32_t query_rows = query_tile + consumer * 64 * 128;

    uint32_t output_words[HEAD_DIM / 2];
    #pragma unroll
    for (int index = 0; index < HEAD_DIM / 2; index++) {
        output_words[index] = 0;
    }
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {0.0f, 0.0f};

    for (int block = 0; block < blocks; block++) {
        const int stage = block % HOPPER_STAGES;
        barrier_wait(full_barriers + stage * 8, (block / HOPPER_STAGES) & 1);
        const uint32_t key_tile = stages + stage * H::stage_bytes;
        const uint32_t value_tile = key_tile + T::tile_bytes;

        uint32_t score_words[BLOCK_N / 2];
        wgmma_fence();
        #pragma unroll
        for (int k_step = 0; k_step < HEAD_DIM / 16; k_step++) {
            // 32 bytes of the head dimension per step, four to a 128-byte slab.
            const int slab = k_step / 4;
            const int offset = (k_step % 4) * 32;
            Forms::scores(score_words, wgmma_descriptor(query_rows + slab * BLOCK_M * 128 + offset),
                          wgmma_descriptor(key_tile + slab * BLOCK_N * 128 + offset), k_step);
        }
        wgmma_commit();
        wgmma_wait<0>();

        float scores[BLOCK_N / 8][4];
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                scores[key_tile_index][element] =
                    __uint_as_float(score_words[key_tile_index * 4 + element]) * scale_log2;
            }
        }
        const bool masked_block = (block + 1) * BLOCK_N > tokens ||
                                  (layout.causal && (block + 1) * BLOCK_N - 1 > first_query);
        if (masked_block) {
            #pragma unroll
            for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
                #pragma unroll
                for (int element = 0; element < 4; element++) {
                    const int key_index =
                        block * BLOCK_N + key_tile_index * 8 + (lane % 4) * 2 + (element % 2);
                    const int query_index = first_query + warp * 16 + (element / 2) * 8 + lane / 4;
                    if (key_index >= tokens || (layout.causal && key_index > query_index)) {
                        scores[key_tile_index][element] = -FLT_MAX;
                    }
                }
            }
        }
        float block_max[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                block_max[element / 2] =
                    fmaxf(block_max[element / 2], scores[key_tile_index][element]);
            }
        }
        float correction[2];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            block_max[half] =
                fmaxf(block_max[half], __shfl_xor_sync(0xffffffff, block_max[half], 1));
            block_max[half] =
                fmaxf(block_max[half], __shfl_xor_sync(0xffffffff, block_max[half], 2));
            float new_max = fmaxf(row_max[half], block_max[half]);
            correction[half] = exp2f(row_max[half] - new_max);
            row_max[half] = new_max;
            row_sum[half] *= correction[half];
        }
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                float probability = exp2f(scores[key_tile_index][element] - row_max[element / 2]);
                scores[key_tile_index][element] = probability;
                row_sum[element / 2] += probability;
            }
        }
        #pragma unroll
        for (int index = 0; index < HEAD_DIM / 2; index++) {
            output_words[index] =
                __float_as_uint(__uint_as_float(output_words[index]) * correction[(index / 2) % 2]);
        }

        wgmma_fence();
        #pragma unroll
        for (int key_step = 0; key_step < BLOCK_N / 16; key_step++) {
            const uint32_t probability_fragment[4] = {
                N::pack(scores[key_step * 2][0], scores[key_step * 2][1]),
                N::pack(scores[key_step * 2][2], scores[key_step * 2][3]),
                N::pack(scores[key_step * 2 + 1][0], scores[key_step * 2 + 1][1]),
                N::pack(scores[key_step * 2 + 1][2], scores[key_step * 2 + 1][3]),
            };
            // 16 keys per step, 2048 bytes of the value slabs.
            Forms::template values<HEAD_DIM>(
                output_words, probability_fragment,
                wgmma_descriptor_mn_major(value_tile + key_step * 16 * 128, BLOCK_N * 128));
        }
        wgmma_commit();
        wgmma_wait<0>();
        barrier_arrive(empty_barriers + stage * 8);
    }

    #pragma unroll
    for (int half = 0; half < 2; half++) {
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 2);
    }
    const int group_id = lane / 4;
    Element *output_head =
        output + batch * layout.batch_stride[OUTPUT] + head * layout.head_stride[OUTPUT];
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int token = first_query + warp * 16 + half * 8 + group_id;
        if (token >= tokens) {
            continue;
        }
        const float inverse_sum = 1.0f / row_sum[half];
        Element *output_row =
            output_head + static_cast<int64_t>(token) * layout.token_stride[OUTPUT];
        #pragma unroll
        for (int dimension_tile = 0; dimension_tile < HEAD_DIM / 8; dimension_tile++) {
            const int dimension = dimension_tile * 8 + (lane % 4) * 2;
            *reinterpret_cast<uint32_t *>(output_row + dimension) = N::pack(
                __uint_as_float(output_words[dimension_tile * 4 + half * 2]) * inverse_sum,
                __uint_as_float(output_words[dimension_tile * 4 + half * 2 + 1]) * inverse_sum);
        }
    }
}

// Describes one operand as [head dimension, tokens, heads, batch] in boxes of `box_rows` tokens by
// 128 bytes of the head dimension, zero-filling boxes that reach past the last token. A dimension
// of one never uses its stride, but the encoder rejects a zero one.
bool encode_attention_map(CUtensorMap *map, const void *base, CUtensorMapDataType type,
                          int element_bytes, int head_dim, int tokens, int heads, int batch,
                          int box_rows, int64_t token_stride, int64_t head_stride,
                          int64_t batch_stride) {
    PFN_cuTensorMapEncodeTiled_v12000 encoder = tensor_map_encoder();
    if (encoder == nullptr || reinterpret_cast<uintptr_t>(base) % 16 != 0) {
        return false;
    }
    const cuuint64_t dimensions[4] = {
        static_cast<cuuint64_t>(head_dim), static_cast<cuuint64_t>(tokens),
        static_cast<cuuint64_t>(heads), static_cast<cuuint64_t>(batch)};
    cuuint64_t strides[3] = {static_cast<cuuint64_t>(token_stride) * element_bytes,
                             static_cast<cuuint64_t>(head_stride) * element_bytes,
                             static_cast<cuuint64_t>(batch_stride) * element_bytes};
    if (strides[1] == 0) {
        strides[1] = strides[0] * tokens;
    }
    if (strides[2] == 0) {
        strides[2] = strides[1] * heads;
    }
    for (int dimension = 0; dimension < 3; dimension++) {
        if (strides[dimension] % 16 != 0) {
            return false;
        }
    }
    const cuuint32_t box[4] = {static_cast<cuuint32_t>(128 / element_bytes),
                               static_cast<cuuint32_t>(box_rows), 1, 1};
    const cuuint32_t element_strides[4] = {1, 1, 1, 1};
    return encoder(map, type, 4, const_cast<void *>(base), dimensions, strides, box,
                   element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <typename Element> CUtensorMapDataType map_type();
template <> CUtensorMapDataType map_type<__nv_bfloat16>() {
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}
template <> CUtensorMapDataType map_type<__half>() { return CU_TENSOR_MAP_DATA_TYPE_FLOAT16; }

template <typename Element, int HEAD_DIM, bool TMA, bool ROPE>
int launch_kernel(const Element *query, const Element *key, const Element *value, Element *output,
                  int tokens, int heads, int batch, const Mmh3AttentionLayout &layout,
                  const AttentionMaps &maps, const AttentionRope &rope, float scale_log2,
                  cudaStream_t stream) {
    dim3 grid((tokens + BLOCK_M - 1) / BLOCK_M, heads, batch);
    if constexpr (TMA) {
        if (mmh3_wgmma_available()) {
            auto kernel = attention_hopper_kernel<Element, HEAD_DIM, ROPE>;
            constexpr int shared_bytes = HopperTiles<HEAD_DIM>::shared_bytes;
            static Mmh3SharedMemory shared_memory;
            const cudaError_t configured =
                mmh3_configure_shared_memory(shared_memory, kernel, shared_bytes);
            if (configured != cudaSuccess) {
                return static_cast<int>(configured);
            }
            kernel<<<grid, HOPPER_THREADS, shared_bytes, stream>>>(output, tokens, layout,
                                                                   scale_log2, maps, rope);
            return static_cast<int>(cudaGetLastError());
        }
    }
    auto kernel = attention_kernel<Element, HEAD_DIM, TMA, ROPE>;
    // The TMA path adds its barriers and the alignment its boxes need.
    constexpr int shared_bytes = Tiles<HEAD_DIM>::shared_bytes + (TMA ? 256 : 0);
    static Mmh3SharedMemory shared_memory;
    const cudaError_t configured =
        mmh3_configure_shared_memory(shared_memory, kernel, shared_bytes);
    if (configured != cudaSuccess) {
        return static_cast<int>(configured);
    }
    kernel<<<grid, THREADS, shared_bytes, stream>>>(query, key, value, output, tokens, layout,
                                                    scale_log2, maps, rope);
    return static_cast<int>(cudaGetLastError());
}

template <typename Element, int HEAD_DIM, bool TMA>
int launch_roped(const Element *query, const Element *key, const Element *value, Element *output,
                 int tokens, int heads, int batch, const Mmh3AttentionLayout &layout,
                 const AttentionMaps &maps, const AttentionRope &rope, float scale_log2,
                 cudaStream_t stream) {
    if (rope.angles != nullptr) {
        return launch_kernel<Element, HEAD_DIM, TMA, true>(query, key, value, output, tokens, heads,
                                                           batch, layout, maps, rope, scale_log2,
                                                           stream);
    }
    return launch_kernel<Element, HEAD_DIM, TMA, false>(
        query, key, value, output, tokens, heads, batch, layout, maps, rope, scale_log2, stream);
}

template <typename Element, int HEAD_DIM>
int launch(const void *query, const void *key, const void *value, void *output, int tokens,
           int heads, int batch, const Mmh3AttentionLayout &layout, const AttentionRope &rope,
           float scale, cudaStream_t stream) {
    Mmh3AttentionLayout normalized = layout;
    normalized.heads_per_key_value =
        layout.heads_per_key_value > 0 ? layout.heads_per_key_value : 1;
    const Element *query_base = static_cast<const Element *>(query);
    const Element *key_base = static_cast<const Element *>(key);
    const Element *value_base = static_cast<const Element *>(value);
    Element *output_base = static_cast<Element *>(output);
    const float scale_log2 = scale * 1.4426950408889634f;
    const int key_value_heads = heads / normalized.heads_per_key_value;

    AttentionMaps maps = {};
    const CUtensorMapDataType type = map_type<Element>();
    const bool mapped =
        mmh3_tma_available() &&
        encode_attention_map(&maps.query, query_base, type, sizeof(Element), HEAD_DIM, tokens,
                             heads, batch, BLOCK_M, normalized.token_stride[QUERY],
                             normalized.head_stride[QUERY], normalized.batch_stride[QUERY]) &&
        encode_attention_map(&maps.key, key_base, type, sizeof(Element), HEAD_DIM, tokens,
                             key_value_heads, batch, BLOCK_N, normalized.token_stride[KEY],
                             normalized.head_stride[KEY], normalized.batch_stride[KEY]) &&
        encode_attention_map(&maps.value, value_base, type, sizeof(Element), HEAD_DIM, tokens,
                             key_value_heads, batch, BLOCK_N, normalized.token_stride[VALUE],
                             normalized.head_stride[VALUE], normalized.batch_stride[VALUE]);
    if (mapped) {
        return launch_roped<Element, HEAD_DIM, true>(query_base, key_base, value_base, output_base,
                                                     tokens, heads, batch, normalized, maps, rope,
                                                     scale_log2, stream);
    }
    return launch_roped<Element, HEAD_DIM, false>(query_base, key_base, value_base, output_base,
                                                  tokens, heads, batch, normalized, maps, rope,
                                                  scale_log2, stream);
}

} // namespace

// element_type 0 is BF16 and 1 is FP16. head_dim is 64 or 128.
// With `angles`, q and k are normalized over the head dimension and rotated by `pairs` angle pairs
// per token as they reach shared memory, which is what the video VAE's decoder needs.
extern "C" int mmh3_attention_roped(int element_type, int head_dim, const void *query,
                                    const void *key, const void *value, void *output, int tokens,
                                    int heads, int batch, const Mmh3AttentionLayout *layout,
                                    float scale, const float *angles, int pairs, float epsilon,
                                    cudaStream_t stream) {
    if (tokens <= 0 || heads <= 0 || batch <= 0 ||
        (layout->heads_per_key_value > 0 && heads % layout->heads_per_key_value != 0)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const AttentionRope rope = {angles, pairs, epsilon};
    if (element_type == 0 && head_dim == 128) {
        return launch<__nv_bfloat16, 128>(query, key, value, output, tokens, heads, batch, *layout,
                                          rope, scale, stream);
    }
    if (element_type == 1 && head_dim == 64) {
        return launch<__half, 64>(query, key, value, output, tokens, heads, batch, *layout, rope,
                                  scale, stream);
    }
    if (element_type == 0 && head_dim == 64) {
        return launch<__nv_bfloat16, 64>(query, key, value, output, tokens, heads, batch, *layout,
                                         rope, scale, stream);
    }
    if (element_type == 1 && head_dim == 128) {
        return launch<__half, 128>(query, key, value, output, tokens, heads, batch, *layout, rope,
                                   scale, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

extern "C" int mmh3_attention(int element_type, int head_dim, const void *query, const void *key,
                              const void *value, void *output, int tokens, int heads, int batch,
                              const Mmh3AttentionLayout *layout, float scale, cudaStream_t stream) {
    return mmh3_attention_roped(element_type, head_dim, query, key, value, output, tokens, heads,
                                batch, layout, scale, nullptr, 0, 0.0f, stream);
}

// The DiT's layout: q, k and v with their own token strides and heads of 128 packed contiguously.
extern "C" int mmh3_attention_bf16(const __nv_bfloat16 *query, const __nv_bfloat16 *key,
                                   const __nv_bfloat16 *value, __nv_bfloat16 *output, int tokens,
                                   int heads, int64_t query_stride, int64_t key_stride,
                                   int64_t value_stride, int64_t output_stride, float scale,
                                   cudaStream_t stream) {
    Mmh3AttentionLayout layout = {{query_stride, key_stride, value_stride, output_stride},
                                  {128, 128, 128, 128},
                                  {0, 0, 0, 0},
                                  1,
                                  0};
    return mmh3_attention(0, 128, query, key, value, output, tokens, heads, 1, &layout, scale,
                          stream);
}
