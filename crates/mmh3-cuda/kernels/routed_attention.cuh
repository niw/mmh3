#pragma once

#include <cfloat>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "attention_workspace.cuh"
#include "device.cuh"
#include "tensor_core.cuh"
#include "tma.cuh"
#include "wgmma.cuh"

// The Hopper kernel of BF16 attention over routed key blocks, which Sol-Attn and VSA share: one
// warpgroup per block of up to 64 queries, whose thread 0 copies the query tile and the key and
// value tiles of the routed blocks into a ring of stages with TMA, and which multiplies Q · Kᵀ and
// P · V with wgmma. The scores, the softmax and the output of each are those of its mma.sync
// kernel, since the accumulators lie in the same layout.

namespace {

constexpr int ROUTED_HEAD = 128;
constexpr int ROUTED_BLOCK = 64;
constexpr int ROUTED_THREADS = 128;
constexpr int ROUTED_STAGES = 3;
// A tile of 64 rows of 128 BF16 values, as two slabs of 128 bytes per row.
constexpr int ROUTED_SLAB_BYTES = ROUTED_BLOCK * 128;
constexpr int ROUTED_TILE_BYTES = 2 * ROUTED_SLAB_BYTES;
// The query tile, a key and a value tile per stage, alignment slack, and the query barrier and a
// barrier per stage. Two CTAs share an SM.
constexpr int ROUTED_SHARED_BYTES =
    ROUTED_TILE_BYTES * (1 + 2 * ROUTED_STAGES) + 1024 + (1 + ROUTED_STAGES) * 8;

// The query, key and value tensors as TMA sees them: [128 values, tokens, heads].
struct RoutedMaps {
    CUtensorMap query;
    CUtensorMap key;
    CUtensorMap value;
};

// Which key blocks each query block attends to, and what finishes its output: Sol-Attn's blocks of
// 64 tokens, their row offsets and pooled tails, or VSA's tiles and gated coarse branch.
struct RoutedAttention {
    // VSA's tiles, or null for Sol-Attn's blocks of 64 tokens.
    const int32_t *tile_starts;
    const int32_t *tile_lengths;
    const uint16_t *routes;
    const int32_t *route_counts;
    // Query blocks, and the length of every row of routes.
    int blocks;
    const float *row_offsets;
    const float *tail_max;
    const float *tail_sum;
    const float *tail_values;
    const float *coarse;
    const __nv_bfloat16 *gate;
    int64_t gate_stride;
};

enum RoutedOperand { ROUTED_QUERY = 0, ROUTED_KEY = 1, ROUTED_VALUE = 2, ROUTED_OUTPUT = 3 };

template <bool VSA>
__global__ void __launch_bounds__(ROUTED_THREADS, 2)
    routed_attention_hopper_kernel(__nv_bfloat16 *__restrict__ output, int tokens,
                                   Mmh3AttentionLayout layout, RoutedAttention routed,
                                   float scale_log2, const __grid_constant__ RoutedMaps maps) {
    using N = Numeric<__nv_bfloat16>;
    extern __shared__ __align__(128) uint8_t shared_memory[];
    const uint32_t shared_base = (shared_address(shared_memory) + 1023) & ~1023u;
    const uint32_t query_tile = shared_base;
    const uint32_t stages = shared_base + ROUTED_TILE_BYTES;
    const uint32_t query_barrier = stages + ROUTED_STAGES * 2 * ROUTED_TILE_BYTES;
    const uint32_t full_barriers = query_barrier + 8;
    const int query_block = blockIdx.x;
    const int head = blockIdx.y;
    const int first_query = VSA ? routed.tile_starts[query_block] : query_block * ROUTED_BLOCK;
    const int query_rows =
        VSA ? routed.tile_lengths[query_block] : min(ROUTED_BLOCK, tokens - first_query);
    const size_t row = static_cast<size_t>(head) * routed.blocks + query_block;
    const uint16_t *route = routed.routes + row * routed.blocks;
    const int count = routed.route_counts[row];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    auto first_key_of = [&](int entry) {
        const int block = route[entry];
        return VSA ? routed.tile_starts[block] : block * ROUTED_BLOCK;
    };
    // Copies the key and value tiles of the `entry`-th routed block.
    auto issue = [&](int entry) {
        const int stage = entry % ROUTED_STAGES;
        const int first_key = first_key_of(entry);
        const uint32_t barrier = full_barriers + stage * 8;
        const uint32_t key_tile = stages + stage * 2 * ROUTED_TILE_BYTES;
        barrier_expect_bytes(barrier, 2 * ROUTED_TILE_BYTES);
        #pragma unroll
        for (int slab = 0; slab < 2; slab++) {
            copy_tile(key_tile + slab * ROUTED_SLAB_BYTES, &maps.key, barrier, slab * 64, first_key,
                      head);
            copy_tile(key_tile + ROUTED_TILE_BYTES + slab * ROUTED_SLAB_BYTES, &maps.value, barrier,
                      slab * 64, first_key, head);
        }
    };
    if (threadIdx.x == 0) {
        barrier_init(query_barrier, 1);
        for (int stage = 0; stage < ROUTED_STAGES; stage++) {
            barrier_init(full_barriers + stage * 8, 1);
        }
        barrier_init_fence();
        barrier_expect_bytes(query_barrier, ROUTED_TILE_BYTES);
        #pragma unroll
        for (int slab = 0; slab < 2; slab++) {
            copy_tile(query_tile + slab * ROUTED_SLAB_BYTES, &maps.query, query_barrier, slab * 64,
                      first_query, head);
        }
        for (int entry = 0; entry < min(count, ROUTED_STAGES); entry++) {
            issue(entry);
        }
    }
    __syncthreads();
    barrier_wait(query_barrier, 0);

    const int group_id = lane / 4;
    float offsets[2] = {0.0f, 0.0f};
    if constexpr (!VSA) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int token = first_query + warp * 16 + half * 8 + group_id;
            offsets[half] = token < tokens
                                ? routed.row_offsets[static_cast<size_t>(head) * tokens + token]
                                : 0.0f;
        }
    }
    uint32_t output_words[ROUTED_HEAD / 2];
    #pragma unroll
    for (int index = 0; index < ROUTED_HEAD / 2; index++) {
        output_words[index] = 0;
    }
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {0.0f, 0.0f};

    for (int entry = 0; entry < count; entry++) {
        const int stage = entry % ROUTED_STAGES;
        barrier_wait(full_barriers + stage * 8, (entry / ROUTED_STAGES) & 1);
        const uint32_t key_tile = stages + stage * 2 * ROUTED_TILE_BYTES;
        const uint32_t value_tile = key_tile + ROUTED_TILE_BYTES;

        uint32_t score_words[ROUTED_BLOCK / 2];
        wgmma_fence();
        #pragma unroll
        for (int k_step = 0; k_step < ROUTED_HEAD / 16; k_step++) {
            // 32 bytes of the head per step, four to a slab.
            const int offset = (k_step / 4) * ROUTED_SLAB_BYTES + (k_step % 4) * 32;
            wgmma_bf16_m64n64k16(score_words, wgmma_descriptor(query_tile + offset),
                                 wgmma_descriptor(key_tile + offset), k_step);
        }
        wgmma_commit();
        wgmma_wait<0>();

        // Keys past the block's tokens, or past the VSA tile, are masked.
        const int key_rows = VSA ? routed.tile_lengths[route[entry]] : tokens - first_key_of(entry);
        const bool partial_block = key_rows < ROUTED_BLOCK;
        float scores[ROUTED_BLOCK / 8][4];
        float block_max[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < ROUTED_BLOCK / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                float score =
                    __uint_as_float(score_words[key_tile_index * 4 + element]) * scale_log2 -
                    offsets[element / 2];
                if (partial_block &&
                    key_tile_index * 8 + (lane % 4) * 2 + (element % 2) >= key_rows) {
                    score = -FLT_MAX;
                }
                scores[key_tile_index][element] = score;
                block_max[element / 2] = fmaxf(block_max[element / 2], score);
            }
        }
        float correction[2];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            block_max[half] =
                fmaxf(block_max[half], __shfl_xor_sync(0xffffffff, block_max[half], 1));
            block_max[half] =
                fmaxf(block_max[half], __shfl_xor_sync(0xffffffff, block_max[half], 2));
            const float new_max = fmaxf(row_max[half], block_max[half]);
            correction[half] = exp2f(row_max[half] - new_max);
            row_max[half] = new_max;
            row_sum[half] *= correction[half];
        }
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < ROUTED_BLOCK / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                const float probability =
                    exp2f(scores[key_tile_index][element] - row_max[element / 2]);
                scores[key_tile_index][element] = probability;
                row_sum[element / 2] += probability;
            }
        }
        #pragma unroll
        for (int index = 0; index < ROUTED_HEAD / 2; index++) {
            output_words[index] =
                __float_as_uint(__uint_as_float(output_words[index]) * correction[(index / 2) % 2]);
        }

        wgmma_fence();
        #pragma unroll
        for (int key_step = 0; key_step < ROUTED_BLOCK / 16; key_step++) {
            const uint32_t probability_fragment[4] = {
                N::pack(scores[key_step * 2][0], scores[key_step * 2][1]),
                N::pack(scores[key_step * 2][2], scores[key_step * 2][3]),
                N::pack(scores[key_step * 2 + 1][0], scores[key_step * 2 + 1][1]),
                N::pack(scores[key_step * 2 + 1][2], scores[key_step * 2 + 1][3]),
            };
            // 16 keys per step, 2048 bytes of the value slabs.
            wgmma_bf16_m64n128k16_registers_transposed(
                output_words, probability_fragment,
                wgmma_descriptor_mn_major(value_tile + key_step * 16 * 128, ROUTED_SLAB_BYTES), 1);
        }
        wgmma_commit();
        wgmma_wait<0>();
        // Every warp is done with the stage before thread 0 refills it.
        __syncthreads();
        if (threadIdx.x == 0 && entry + ROUTED_STAGES < count) {
            issue(entry + ROUTED_STAGES);
        }
    }

    auto accumulator = [&](int dimension_tile, int index) {
        return __uint_as_float(output_words[dimension_tile * 4 + index]);
    };
    __nv_bfloat16 *output_head = output + head * layout.head_stride[ROUTED_OUTPUT];
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 2);
        const int tile_row = warp * 16 + half * 8 + group_id;
        if (tile_row >= query_rows) {
            continue;
        }
        const int64_t token = first_query + tile_row;
        __nv_bfloat16 *output_row = output_head + token * layout.token_stride[ROUTED_OUTPUT];
        if constexpr (VSA) {
            const float inverse_sum = 1.0f / row_sum[half];
            const float *coarse =
                routed.gate != nullptr ? routed.coarse + row * ROUTED_HEAD : nullptr;
            const __nv_bfloat16 *gate_row =
                routed.gate != nullptr
                    ? routed.gate + token * routed.gate_stride + head * ROUTED_HEAD
                    : nullptr;
            #pragma unroll
            for (int dimension_tile = 0; dimension_tile < ROUTED_HEAD / 8; dimension_tile++) {
                const int dimension = dimension_tile * 8 + (lane % 4) * 2;
                float first = accumulator(dimension_tile, half * 2) * inverse_sum;
                float second = accumulator(dimension_tile, half * 2 + 1) * inverse_sum;
                if (gate_row != nullptr) {
                    const __nv_bfloat162 gates =
                        *reinterpret_cast<const __nv_bfloat162 *>(gate_row + dimension);
                    first = fmaf(__low2float(gates), coarse[dimension], first);
                    second = fmaf(__high2float(gates), coarse[dimension + 1], second);
                }
                *reinterpret_cast<uint32_t *>(output_row + dimension) = N::pack(first, second);
            }
        } else {
            // Merge the routed blocks with the pooled tail of the query block.
            const float tail_max = routed.tail_max[row];
            const float tail_sum = routed.tail_sum[row];
            const float *tail_values = routed.tail_values + row * ROUTED_HEAD;
            const float merged_max = fmaxf(row_max[half], tail_max);
            const float routed_weight = exp2f(row_max[half] - merged_max);
            const float tail_weight = exp2f(tail_max - merged_max);
            const float inverse_sum =
                1.0f / (row_sum[half] * routed_weight + tail_sum * tail_weight);
            #pragma unroll
            for (int dimension_tile = 0; dimension_tile < ROUTED_HEAD / 8; dimension_tile++) {
                const int dimension = dimension_tile * 8 + (lane % 4) * 2;
                const float first = (accumulator(dimension_tile, half * 2) * routed_weight +
                                     tail_values[dimension] * tail_weight) *
                                    inverse_sum;
                const float second = (accumulator(dimension_tile, half * 2 + 1) * routed_weight +
                                      tail_values[dimension + 1] * tail_weight) *
                                     inverse_sum;
                *reinterpret_cast<uint32_t *>(output_row + dimension) = N::pack(first, second);
            }
        }
    }
}

// Describes one BF16 operand as [128 values, tokens, heads] in boxes of 64 values by 64 tokens.
inline bool encode_routed_map(CUtensorMap *map, const void *base, int tokens, int heads,
                              int64_t token_stride, int64_t head_stride) {
    PFN_cuTensorMapEncodeTiled_v12000 encoder = tensor_map_encoder();
    if (encoder == nullptr || reinterpret_cast<uintptr_t>(base) % 16 != 0) {
        return false;
    }
    const cuuint64_t dimensions[3] = {ROUTED_HEAD, static_cast<cuuint64_t>(tokens),
                                      static_cast<cuuint64_t>(heads)};
    cuuint64_t strides[2] = {static_cast<cuuint64_t>(token_stride) * 2,
                             static_cast<cuuint64_t>(head_stride) * 2};
    // A dimension of one never uses its stride, but the encoder rejects a zero one.
    if (strides[1] == 0) {
        strides[1] = strides[0] * tokens;
    }
    if (strides[0] % 16 != 0 || strides[1] % 16 != 0) {
        return false;
    }
    const cuuint32_t box[3] = {64, ROUTED_BLOCK, 1};
    const cuuint32_t element_strides[3] = {1, 1, 1};
    return encoder(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void *>(base), dimensions,
                   strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// Runs the Hopper kernel over `blocks` query blocks and `heads` heads. Answers -1 without launching
// where the device has no wgmma or TMA cannot describe the layout, so that the caller runs its
// mma.sync kernel instead.
template <bool VSA>
int launch_routed_attention_hopper(const __nv_bfloat16 *query, const __nv_bfloat16 *key,
                                   const __nv_bfloat16 *value, __nv_bfloat16 *output, int tokens,
                                   int heads, const Mmh3AttentionLayout &layout,
                                   const RoutedAttention &routed, float scale_log2,
                                   cudaStream_t stream) {
    if (!mmh3_wgmma_available()) {
        return -1;
    }
    RoutedMaps maps;
    if (!encode_routed_map(&maps.query, query, tokens, heads, layout.token_stride[ROUTED_QUERY],
                           layout.head_stride[ROUTED_QUERY]) ||
        !encode_routed_map(&maps.key, key, tokens, heads, layout.token_stride[ROUTED_KEY],
                           layout.head_stride[ROUTED_KEY]) ||
        !encode_routed_map(&maps.value, value, tokens, heads, layout.token_stride[ROUTED_VALUE],
                           layout.head_stride[ROUTED_VALUE])) {
        return -1;
    }
    auto kernel = routed_attention_hopper_kernel<VSA>;
    static Mmh3SharedMemory shared_memory;
    const cudaError_t configured =
        mmh3_configure_shared_memory(shared_memory, kernel, ROUTED_SHARED_BYTES);
    if (configured != cudaSuccess) {
        return static_cast<int>(configured);
    }
    kernel<<<dim3(routed.blocks, heads), ROUTED_THREADS, ROUTED_SHARED_BYTES, stream>>>(
        output, tokens, layout, routed, scale_log2, maps);
    return static_cast<int>(cudaGetLastError());
}

} // namespace
