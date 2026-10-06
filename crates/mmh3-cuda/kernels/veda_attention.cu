#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "device.cuh"
#include "tensor_core.cuh"

// Veda over BF16 heads of 128 (see mmh3-core's dit/veda.rs for the algorithm) for a group of a
// layer's heads that share a tile shape. The rows stay in the packed order: every kernel reads a
// tile's rows through the shape's order table, and the attention writes its output back in place.
// A tile of 128 is cut into two tiles of at most 64 for the attention.
//
// 1. pool: per head and video tile, the mean, the maximum and the minimum of the queries and of
//    the keys.
// 2. project: the predictor's view of a tile, pooled · P + mean, with P the FP8 projection of the
//    layer and head times its scale, rounded to BF16 as the predictor runs it.
// 3. select: per head and query tile of 128, the scores against every video tile, and the tiles of
//    64 it attends, which the two halves of the tile share.
// 4. attention: FlashAttention over the selected tiles of each query tile of 64.

// The tables of one tile shape: the packed row of each position in tile order, each tile's first
// position and rows, and each video query tile's counts for the reference and generated blocks.
struct Mmh3VedaShape {
    const int32_t *order;
    const int32_t *starts;
    const int32_t *lengths;
    const int32_t *allowed;
    int32_t tiles;
    int32_t video;
    int32_t reference;
};

// Scratch of one group: pooled [members, video, 384] and projected [members, video, 128] queries
// and keys, the routes [members, tiles, 2 × tiles] of tiles of 64 and their counts [members,
// tiles], and the video tiles kept so far.
struct Mmh3VedaWorkspace {
    float *pooled_query;
    float *pooled_key;
    float *projected_query;
    float *projected_key;
    uint16_t *routes;
    int32_t *counts;
    unsigned long long *kept;
};

namespace {

enum Operand { QUERY = 0, KEY = 1, VALUE = 2, OUTPUT = 3 };

constexpr int HEAD = 128;
constexpr int POOLED = 3 * HEAD;
constexpr int BLOCK = 64;
constexpr int WARPS = BLOCK / 16;
constexpr int THREADS = WARPS * 32;
// Video tiles one CTA of project_kernel projects, sharing the reads of P.
constexpr int PROJECT_TILES = 16;
using T = Tiles<HEAD>;
using N = Numeric<__nv_bfloat16>;
static_assert(BLOCK == BLOCK_N, "a tile of 64 fills one shared memory tile");
static_assert(THREADS == HEAD, "the predictor's kernels give each thread one dimension");
constexpr int ATTENTION_SHARED_BYTES = 2 * T::tile_bytes;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ int block_sum(int value, int *scratch) {
    value = __reduce_add_sync(0xffffffff, value);
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = value;
    }
    __syncthreads();
    int result = 0;
    for (int warp = 0; warp < THREADS / 32; warp++) {
        result += scratch[warp];
    }
    return result;
}

// Maps a float to an unsigned integer with the same order.
__device__ __forceinline__ uint32_t ordered_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

// Rows of tile `tile` of 64 in the tiles of 128.
__device__ __forceinline__ int small_rows(const int32_t *lengths, int tile) {
    const int length = lengths[tile / 2];
    return tile % 2 == 0 ? min(length, BLOCK) : max(length - BLOCK, 0);
}

// An empty tile pools to zeros.
__global__ void __launch_bounds__(THREADS)
    pool_kernel(const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ key,
                Mmh3AttentionLayout layout, Mmh3VedaShape shape, const int32_t *heads,
                Mmh3VedaWorkspace workspace) {
    const int tile = blockIdx.x;
    const int member = blockIdx.y;
    const int dimension = threadIdx.x;
    const int head = heads[member];
    const int start = shape.starts[tile];
    const int rows = shape.lengths[tile];
    const __nv_bfloat16 *query_head = query + head * layout.head_stride[QUERY] + dimension;
    const __nv_bfloat16 *key_head = key + head * layout.head_stride[KEY] + dimension;
    float query_sum = 0.0f, query_max = -FLT_MAX, query_min = FLT_MAX;
    float key_sum = 0.0f, key_max = -FLT_MAX, key_min = FLT_MAX;
    for (int row = 0; row < rows; row++) {
        const int64_t token = shape.order[start + row];
        const float query_value = __bfloat162float(query_head[token * layout.token_stride[QUERY]]);
        const float key_value = __bfloat162float(key_head[token * layout.token_stride[KEY]]);
        query_sum += query_value;
        query_max = fmaxf(query_max, query_value);
        query_min = fminf(query_min, query_value);
        key_sum += key_value;
        key_max = fmaxf(key_max, key_value);
        key_min = fminf(key_min, key_value);
    }
    const size_t index = (static_cast<size_t>(member) * shape.video + tile) * POOLED + dimension;
    const bool empty = rows == 0;
    workspace.pooled_query[index] = empty ? 0.0f : query_sum / rows;
    workspace.pooled_query[index + HEAD] = empty ? 0.0f : query_max;
    workspace.pooled_query[index + 2 * HEAD] = empty ? 0.0f : query_min;
    workspace.pooled_key[index] = empty ? 0.0f : key_sum / rows;
    workspace.pooled_key[index + HEAD] = empty ? 0.0f : key_max;
    workspace.pooled_key[index + 2 * HEAD] = empty ? 0.0f : key_min;
}

// `weights` holds the layer's P, [heads, 384, 128] FP8 E4M3 values, and `scales` one per head.
__global__ void __launch_bounds__(THREADS)
    project_kernel(const float *__restrict__ pooled, const uint8_t *__restrict__ weights,
                   const float *__restrict__ scales, const int32_t *heads, int video,
                   float *__restrict__ projected) {
    __shared__ float tiles[PROJECT_TILES][POOLED];
    const int first = blockIdx.x * PROJECT_TILES;
    const int member = blockIdx.y;
    const int dimension = threadIdx.x;
    const int head = heads[member];
    for (int index = threadIdx.x; index < PROJECT_TILES * POOLED; index += THREADS) {
        const int tile = first + index / POOLED;
        tiles[index / POOLED][index % POOLED] =
            tile < video
                ? pooled[(static_cast<size_t>(member) * video + tile) * POOLED + index % POOLED]
                : 0.0f;
    }
    __syncthreads();
    const uint8_t *column = weights + static_cast<size_t>(head) * POOLED * HEAD + dimension;
    const float scale = scales[head];
    float sums[PROJECT_TILES] = {};
    for (int row = 0; row < POOLED; row++) {
        __nv_fp8_e4m3 bits;
        bits.__x = column[row * HEAD];
        const float weight =
            __bfloat162float(__float2bfloat16_rn(static_cast<float>(bits) * scale));
        #pragma unroll
        for (int tile = 0; tile < PROJECT_TILES; tile++) {
            sums[tile] = fmaf(tiles[tile][row], weight, sums[tile]);
        }
    }
    #pragma unroll
    for (int tile = 0; tile < PROJECT_TILES; tile++) {
        if (first + tile < video) {
            projected[(static_cast<size_t>(member) * video + first + tile) * HEAD + dimension] =
                sums[tile] + tiles[tile][dimension];
        }
    }
}

// A CTA a query tile of 128 of one head. A video query tile keeps, of each column block, the
// allowed number of tiles of the highest scores, its own tile first and ties going to the lower
// tile, and never an empty tile; then every global tile. A global query tile keeps every tile.
__global__ void __launch_bounds__(THREADS)
    select_kernel(Mmh3VedaShape shape, Mmh3VedaWorkspace workspace, float scale) {
    extern __shared__ float scores[];
    __shared__ float pooled[HEAD];
    __shared__ int scratch[THREADS / 32];
    const int tiles = shape.tiles;
    const int video = shape.video;
    const int query = blockIdx.x;
    const int member = blockIdx.y;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const bool global = query >= video;
    // A video tile's flag sits after the scores.
    bool *chosen = reinterpret_cast<bool *>(scores + video);

    if (!global) {
        pooled[threadIdx.x] =
            workspace.projected_query[(static_cast<size_t>(member) * video + query) * HEAD +
                                      threadIdx.x];
        for (int tile = threadIdx.x; tile < video; tile += THREADS) {
            chosen[tile] = false;
        }
        __syncthreads();
        const float *keys = workspace.projected_key + static_cast<size_t>(member) * video * HEAD;
        for (int tile = warp; tile < video; tile += WARPS) {
            float partial = 0.0f;
            #pragma unroll
            for (int part = 0; part < HEAD / 32; part++) {
                partial = fmaf(pooled[lane + part * 32],
                               keys[static_cast<size_t>(tile) * HEAD + lane + part * 32], partial);
            }
            partial = warp_sum(partial);
            if (lane == 0) {
                scores[tile] = partial * scale;
            }
        }
        __syncthreads();
        for (int block = shape.reference == 0 ? 1 : 0; block < 2; block++) {
            const int start = block == 0 ? 0 : shape.reference;
            const int end = block == 0 ? shape.reference : video;
            auto candidate = [&](int tile) { return tile == query || shape.lengths[tile] > 0; };
            auto bits_of = [&](int tile) {
                return ordered_bits(tile == query ? INFINITY : scores[tile]);
            };
            int candidates = 0;
            for (int tile = start + threadIdx.x; tile < end; tile += THREADS) {
                candidates += candidate(tile);
            }
            candidates = block_sum(candidates, scratch);
            const int count = min(shape.allowed[query * 2 + block], candidates);
            // The count-th largest candidate score, one bit at a time from the top: `threshold`
            // gathers its bits, and `ties` ends as how many candidates equal to it are kept.
            uint32_t threshold = 0;
            int ties = count;
            for (int bit = 31; bit >= 0 && count > 0; bit--) {
                const uint32_t high = bit == 31 ? 0u : ~0u << (bit + 1);
                int ones = 0;
                for (int tile = start + threadIdx.x; tile < end; tile += THREADS) {
                    if (candidate(tile)) {
                        const uint32_t bits = bits_of(tile);
                        ones += ((bits & high) == threshold) && ((bits >> bit) & 1u);
                    }
                }
                ones = block_sum(ones, scratch);
                if (ones >= ties) {
                    threshold |= 1u << bit;
                } else {
                    ties -= ones;
                }
            }
            if (warp == 0 && count > 0) {
                int seen = 0;
                for (int first = start; first < end; first += 32) {
                    const int tile = first + lane;
                    const bool live = tile < end && candidate(tile);
                    const uint32_t bits = live ? bits_of(tile) : 0;
                    const bool tie = live && bits == threshold;
                    const unsigned tie_mask = __ballot_sync(0xffffffff, tie);
                    const int rank = seen + __popc(tie_mask & ((1u << lane) - 1));
                    seen += __popc(tie_mask);
                    if (live && (bits > threshold || (tie && rank < ties))) {
                        chosen[tile] = true;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (warp != 0) {
        return;
    }
    const int small = 2 * tiles;
    const size_t row = static_cast<size_t>(member) * tiles + query;
    uint16_t *route = workspace.routes + row * small;
    int count = 0;
    int kept = 0;
    for (int first = 0; first < small; first += 32) {
        const int tile = first + lane;
        const int large = tile / 2;
        const bool flag = tile < small && small_rows(shape.lengths, tile) > 0 &&
                          (global || large >= video || chosen[large]);
        const unsigned selected = __ballot_sync(0xffffffff, flag);
        if (flag) {
            route[count + __popc(selected & ((1u << lane) - 1))] = static_cast<uint16_t>(tile);
        }
        count += __popc(selected);
        if (!global) {
            kept += __popc(__ballot_sync(0xffffffff, tile < small && tile % 2 == 0 &&
                                                         large < video && chosen[large]));
        }
    }
    if (lane == 0) {
        workspace.counts[row] = count;
        if (!global) {
            atomicAdd(workspace.kept, static_cast<unsigned long long>(kept));
        }
    }
}

// `rows` rows from position `first` of the order table, zero-filled past `count`.
__device__ __forceinline__ void load_gathered(uint32_t destination, const __nv_bfloat16 *head_base,
                                              int64_t stride, const int32_t *order, int first,
                                              int count) {
    for (int index = threadIdx.x; index < BLOCK * T::chunks_per_row; index += THREADS) {
        const int row = index / T::chunks_per_row;
        const int chunk = index % T::chunks_per_row;
        const bool valid = row < count;
        const int64_t token = valid ? order[first + row] : 0;
        copy_async_16(destination + T::offset(row, chunk), head_base + token * stride + chunk * 8,
                      valid);
    }
}

__global__ void __launch_bounds__(THREADS, 3)
    attention_kernel(const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ key,
                     const __nv_bfloat16 *__restrict__ value, __nv_bfloat16 *__restrict__ output,
                     Mmh3AttentionLayout layout, Mmh3VedaShape shape, const int32_t *heads,
                     Mmh3VedaWorkspace workspace, float scale_log2) {
    extern __shared__ __align__(128) uint8_t shared_memory[];
    const uint32_t shared_base = shared_address(shared_memory);
    const int query_tile = blockIdx.x;
    const int member = blockIdx.y;
    const int head = heads[member];
    const int tiles = shape.tiles;
    const int first_query = shape.starts[query_tile / 2] + (query_tile % 2) * BLOCK;
    const int query_rows = small_rows(shape.lengths, query_tile);
    if (query_rows == 0) {
        return;
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int matrix = lane / 8;
    const int matrix_row = lane % 8;
    const size_t row = static_cast<size_t>(member) * tiles + query_tile / 2;
    const uint16_t *route = workspace.routes + row * 2 * tiles;
    const int count = workspace.counts[row];

    const __nv_bfloat16 *query_head = query + head * layout.head_stride[QUERY];
    const __nv_bfloat16 *key_head = key + head * layout.head_stride[KEY];
    const __nv_bfloat16 *value_head = value + head * layout.head_stride[VALUE];
    // NOTE: keys and values have one buffer each. The next key tile loads while the warps apply
    // softmax and multiply the values, and the next value tile while they score the next keys.
    const uint32_t key_tile = shared_base;
    const uint32_t value_tile = shared_base + T::tile_bytes;
    auto load_tile = [&](uint32_t destination, const __nv_bfloat16 *head_base, int64_t stride,
                         int entry) {
        const int tile = route[entry];
        load_gathered(destination, head_base, stride, shape.order,
                      shape.starts[tile / 2] + (tile % 2) * BLOCK, small_rows(shape.lengths, tile));
    };

    // The query tile borrows the value buffer until it has been copied into registers.
    load_gathered(value_tile, query_head, layout.token_stride[QUERY], shape.order, first_query,
                  query_rows);
    if (count > 0) {
        load_tile(key_tile, key_head, layout.token_stride[KEY], 0);
    }
    copy_async_commit();
    copy_async_wait<0>();
    __syncthreads();

    uint32_t query_fragments[HEAD / 16][4];
    #pragma unroll
    for (int k_step = 0; k_step < HEAD / 16; k_step++) {
        const int tile_row = warp * 16 + (matrix % 2) * 8 + matrix_row;
        load_matrix_x4(query_fragments[k_step],
                       value_tile + T::offset(tile_row, k_step * 2 + matrix / 2));
    }
    __syncthreads();
    if (count > 0) {
        load_tile(value_tile, value_head, layout.token_stride[VALUE], 0);
    }
    copy_async_commit();

    const int group_id = lane / 4;
    float output_accumulators[HEAD / 8][4] = {};
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {0.0f, 0.0f};

    for (int entry = 0; entry < count; entry++) {
        float scores[BLOCK / 8][4] = {};
        #pragma unroll
        for (int k_step = 0; k_step < HEAD / 16; k_step++) {
            #pragma unroll
            for (int key_pair = 0; key_pair < BLOCK / 16; key_pair++) {
                const int tile_row = key_pair * 16 + (matrix / 2) * 8 + matrix_row;
                uint32_t registers[4];
                load_matrix_x4(registers, key_tile + T::offset(tile_row, k_step * 2 + matrix % 2));
                N::mma(scores[key_pair * 2], query_fragments[k_step], registers[0], registers[1]);
                N::mma(scores[key_pair * 2 + 1], query_fragments[k_step], registers[2],
                       registers[3]);
            }
        }

        // The value tile of this entry has arrived, and no warp reads the key buffer any more.
        copy_async_wait<0>();
        __syncthreads();
        if (entry + 1 < count) {
            load_tile(key_tile, key_head, layout.token_stride[KEY], entry + 1);
        }
        copy_async_commit();

        const int key_rows = small_rows(shape.lengths, route[entry]);
        float block_max[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                float score = scores[key_tile_index][element] * scale_log2;
                if (key_tile_index * 8 + (lane % 4) * 2 + (element % 2) >= key_rows) {
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
        for (int key_tile_index = 0; key_tile_index < BLOCK / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                const float probability =
                    exp2f(scores[key_tile_index][element] - row_max[element / 2]);
                scores[key_tile_index][element] = probability;
                row_sum[element / 2] += probability;
            }
        }
        #pragma unroll
        for (int dimension_tile = 0; dimension_tile < HEAD / 8; dimension_tile++) {
            output_accumulators[dimension_tile][0] *= correction[0];
            output_accumulators[dimension_tile][1] *= correction[0];
            output_accumulators[dimension_tile][2] *= correction[1];
            output_accumulators[dimension_tile][3] *= correction[1];
        }
        #pragma unroll
        for (int key_step = 0; key_step < BLOCK / 16; key_step++) {
            const uint32_t probability_fragment[4] = {
                N::pack(scores[key_step * 2][0], scores[key_step * 2][1]),
                N::pack(scores[key_step * 2][2], scores[key_step * 2][3]),
                N::pack(scores[key_step * 2 + 1][0], scores[key_step * 2 + 1][1]),
                N::pack(scores[key_step * 2 + 1][2], scores[key_step * 2 + 1][3]),
            };
            #pragma unroll
            for (int dimension_pair = 0; dimension_pair < HEAD / 16; dimension_pair++) {
                const int tile_row = key_step * 16 + (matrix % 2) * 8 + matrix_row;
                uint32_t registers[4];
                load_matrix_x4_transposed(
                    registers, value_tile + T::offset(tile_row, dimension_pair * 2 + matrix / 2));
                N::mma(output_accumulators[dimension_pair * 2], probability_fragment, registers[0],
                       registers[1]);
                N::mma(output_accumulators[dimension_pair * 2 + 1], probability_fragment,
                       registers[2], registers[3]);
            }
        }
        // The key tile of the next entry has arrived, and no warp reads the value buffer any more.
        copy_async_wait<0>();
        __syncthreads();
        if (entry + 1 < count) {
            load_tile(value_tile, value_head, layout.token_stride[VALUE], entry + 1);
        }
        copy_async_commit();
    }
    copy_async_wait<0>();

    __nv_bfloat16 *output_head = output + head * layout.head_stride[OUTPUT];
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 2);
        const int tile_row = warp * 16 + half * 8 + group_id;
        if (tile_row >= query_rows) {
            continue;
        }
        const int64_t token = shape.order[first_query + tile_row];
        const float inverse_sum = 1.0f / row_sum[half];
        __nv_bfloat16 *output_row = output_head + token * layout.token_stride[OUTPUT];
        #pragma unroll
        for (int dimension_tile = 0; dimension_tile < HEAD / 8; dimension_tile++) {
            const int dimension = dimension_tile * 8 + (lane % 4) * 2;
            *reinterpret_cast<uint32_t *>(output_row + dimension) =
                N::pack(output_accumulators[dimension_tile][half * 2] * inverse_sum,
                        output_accumulators[dimension_tile][half * 2 + 1] * inverse_sum);
        }
    }
}

} // namespace

// Veda over the `members` heads of `heads` (device memory) that share `shape`, for one sequence of
// BF16 heads of 128 in the packed order. The layout's batch strides are ignored. `weights` and
// `scales` are the layer's query and key projections, [heads, 384, 128] FP8 E4M3 values and one
// FP32 scale per head each.
extern "C" int mmh3_veda_attention(const void *query, const void *key, const void *value,
                                   void *output, const Mmh3AttentionLayout *layout,
                                   const Mmh3VedaShape *shape, const int32_t *heads, int members,
                                   const uint8_t *query_weights, const float *query_scales,
                                   const uint8_t *key_weights, const float *key_scales, float scale,
                                   const Mmh3VedaWorkspace *workspace, cudaStream_t stream) {
    if (members <= 0 || members > 65535 || shape->tiles <= 0 || 2 * shape->tiles > 65535 ||
        shape->video < 0 || shape->video > shape->tiles || shape->reference < 0 ||
        shape->reference > shape->video) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto *q = static_cast<const __nv_bfloat16 *>(query);
    const auto *k = static_cast<const __nv_bfloat16 *>(key);
    const auto *v = static_cast<const __nv_bfloat16 *>(value);
    if (shape->video > 0) {
        pool_kernel<<<dim3(shape->video, members), THREADS, 0, stream>>>(q, k, *layout, *shape,
                                                                         heads, *workspace);
        const dim3 project_grid((shape->video + PROJECT_TILES - 1) / PROJECT_TILES, members);
        project_kernel<<<project_grid, THREADS, 0, stream>>>(workspace->pooled_query, query_weights,
                                                             query_scales, heads, shape->video,
                                                             workspace->projected_query);
        project_kernel<<<project_grid, THREADS, 0, stream>>>(workspace->pooled_key, key_weights,
                                                             key_scales, heads, shape->video,
                                                             workspace->projected_key);
    }
    const size_t select_shared = static_cast<size_t>(shape->video) * (sizeof(float) + 1);
    static Mmh3SharedMemoryLimit limit;
    if (select_shared > 32 * 1024) {
        const cudaError_t status = mmh3_raise_shared_memory(limit, select_kernel, select_shared);
        if (status != cudaSuccess) {
            return static_cast<int>(status);
        }
    }
    select_kernel<<<dim3(shape->tiles, members), THREADS, select_shared, stream>>>(
        *shape, *workspace, scale);
    attention_kernel<<<dim3(2 * shape->tiles, members), THREADS, ATTENTION_SHARED_BYTES, stream>>>(
        q, k, v, static_cast<__nv_bfloat16 *>(output), *layout, *shape, heads, *workspace,
        scale * 1.4426950408889634f);
    return static_cast<int>(cudaGetLastError());
}
