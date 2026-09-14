#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mutex>

#include "attention_workspace.cuh"
#include "tensor_core.cuh"

// VSA over BF16 heads of 128 (see mmh3-core's dit/vsa.rs for the algorithm). The sequence is
// already in tile order, and every tile is a run of at most 64 consecutive tokens.
//
// 1. pool: per head and tile, the mean query and key and the summed value, unless
//    mmh3_attention_inputs has computed them.
// 2. select: per head and query tile, the pooled scores against every tile, the tiles it attends
//    token by token, and the coarse output softmax(scores) · mean values.
// 3. attention: FlashAttention over the selected tiles of each query tile, plus gate · coarse.

namespace {

enum Operand { QUERY = 0, KEY = 1, VALUE = 2, OUTPUT = 3 };

constexpr int HEAD = 128;
constexpr int BLOCK = 64;
constexpr int WARPS = BLOCK / 16;
constexpr int THREADS = WARPS * 32;
using T = Tiles<HEAD>;
using N = Numeric<__nv_bfloat16>;
static_assert(BLOCK == BLOCK_N, "a tile fills one shared memory tile");
static_assert(THREADS == HEAD, "the pooling and selection kernels give each thread one dimension");
constexpr int ATTENTION_SHARED_BYTES = 2 * T::tile_bytes;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    return value;
}

template <bool MAXIMUM> __device__ float block_reduce(float value, float *scratch) {
    value = MAXIMUM ? warp_max(value) : warp_sum(value);
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = value;
    }
    __syncthreads();
    float result = scratch[0];
    for (int warp = 1; warp < THREADS / 32; warp++) {
        result = MAXIMUM ? fmaxf(result, scratch[warp]) : result + scratch[warp];
    }
    __syncthreads();
    return result;
}

// Maps a float to an unsigned integer with the same order.
__device__ __forceinline__ uint32_t ordered_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

__global__ void __launch_bounds__(THREADS)
    pool_kernel(const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ key,
                const __nv_bfloat16 *__restrict__ value, Mmh3AttentionLayout layout, int tiles,
                Mmh3VsaWorkspace workspace) {
    const int tile = blockIdx.x;
    const int head = blockIdx.y;
    const int dimension = threadIdx.x;
    const int start = workspace.tile_starts[tile];
    const int rows = workspace.tile_lengths[tile];
    float query_sum = 0.0f, key_sum = 0.0f, value_sum = 0.0f;
    for (int row = 0; row < rows; row++) {
        const int64_t token = start + row;
        query_sum += __bfloat162float(query[token * layout.token_stride[QUERY] +
                                            head * layout.head_stride[QUERY] + dimension]);
        key_sum += __bfloat162float(
            key[token * layout.token_stride[KEY] + head * layout.head_stride[KEY] + dimension]);
        value_sum += __bfloat162float(value[token * layout.token_stride[VALUE] +
                                            head * layout.head_stride[VALUE] + dimension]);
    }
    const size_t index = (static_cast<size_t>(head) * tiles + tile) * HEAD + dimension;
    workspace.pooled_query[index] = query_sum / rows;
    workspace.pooled_key[index] = key_sum / rows;
    workspace.pooled_value[index] = value_sum;
}

// Selects the tiles of QUERIES consecutive query tiles of one head, which share every pooled key
// and value the CTA loads. `kept` video tiles are kept for video query tiles, by the value of
// their scores with ties going to the lower index.
template <int QUERIES>
__global__ void __launch_bounds__(THREADS)
    select_kernel(Mmh3VsaWorkspace workspace, int tiles, int prefix_tiles, int kept, float scale,
                  int coarse) {
    extern __shared__ float scores[];
    constexpr int SCORE_BATCH = 32 / QUERIES;
    constexpr int VALUE_BATCH = 32;
    __shared__ float queries_pooled[QUERIES][HEAD];
    __shared__ float maxima[QUERIES];
    __shared__ float sums[QUERIES];
    __shared__ float scratch[THREADS / 32];
    const int first_query = blockIdx.x * QUERIES;
    const int queries = min(QUERIES, tiles - first_query);
    const int head = blockIdx.y;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const size_t first_row = static_cast<size_t>(head) * tiles + first_query;

    for (int query = 0; query < QUERIES; query++) {
        queries_pooled[query][threadIdx.x] =
            query < queries ? workspace.pooled_query[(first_row + query) * HEAD + threadIdx.x]
                            : 0.0f;
    }
    __syncthreads();

    // A warp scores 32 (query, key tile) pairs at once and reduces them together over one xor
    // butterfly, and reads several key tiles at once so that their loads overlap.
    const float *keys = workspace.pooled_key + static_cast<size_t>(head) * tiles * HEAD;
    for (int first = warp * SCORE_BATCH; first < tiles; first += WARPS * SCORE_BATCH) {
        float loaded[SCORE_BATCH][HEAD / 32];
        #pragma unroll
        for (int batch = 0; batch < SCORE_BATCH; batch++) {
            #pragma unroll
            for (int part = 0; part < HEAD / 32; part++) {
                loaded[batch][part] =
                    first + batch < tiles
                        ? keys[static_cast<size_t>(first + batch) * HEAD + lane + part * 32]
                        : 0.0f;
            }
        }
        float partials[32];
        #pragma unroll
        for (int query = 0; query < QUERIES; query++) {
            #pragma unroll
            for (int batch = 0; batch < SCORE_BATCH; batch++) {
                float partial = 0.0f;
                #pragma unroll
                for (int part = 0; part < HEAD / 32; part++) {
                    partial = __fmaf_rn(queries_pooled[query][lane + part * 32],
                                        loaded[batch][part], partial);
                }
                partials[query * SCORE_BATCH + batch] = partial;
            }
        }
        // Each step adds the partner lane's half of the pairs, so lane l ends up with the sum of
        // pair l.
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            const bool upper = lane & offset;
            #pragma unroll
            for (int index = 0; index < offset; index++) {
                const float kept_part = upper ? partials[index + offset] : partials[index];
                const float sent = upper ? partials[index] : partials[index + offset];
                partials[index] = kept_part + __shfl_xor_sync(0xffffffff, sent, offset);
            }
        }
        const int query = lane / SCORE_BATCH;
        const int tile = first + lane % SCORE_BATCH;
        if (query < queries && tile < tiles) {
            scores[query * tiles + tile] = partials[0] * scale;
        }
    }
    __syncthreads();

    // Each warp selects the tiles of its own query tiles.
    const int video_tiles = tiles - prefix_tiles;
    for (int query = warp; query < queries; query += THREADS / 32) {
        const int query_tile = first_query + query;
        const float *query_scores = scores + query * tiles;
        const bool dense = query_tile < prefix_tiles || kept >= video_tiles;
        // The kept-th largest video score, one bit at a time from the top: `threshold` gathers its
        // bits, and `ties` ends as how many video scores equal to it are kept.
        uint32_t threshold = 0;
        int ties = kept;
        if (!dense) {
            for (int bit = 31; bit >= 0; bit--) {
                const uint32_t high = bit == 31 ? 0u : ~0u << (bit + 1);
                int ones = 0;
                for (int tile = prefix_tiles + lane; tile < tiles; tile += 32) {
                    const uint32_t bits = ordered_bits(query_scores[tile]);
                    ones += ((bits & high) == threshold) && ((bits >> bit) & 1u);
                }
                ones = __reduce_add_sync(0xffffffff, ones);
                if (ones >= ties) {
                    threshold |= 1u << bit;
                } else {
                    ties -= ones;
                }
            }
        }
        uint16_t *route = workspace.routes + (first_row + query) * tiles;
        int count = 0;
        int seen_ties = 0;
        for (int start = 0; start < tiles; start += 32) {
            const int tile = start + lane;
            const bool candidate = !dense && tile >= prefix_tiles && tile < tiles;
            const uint32_t bits = candidate ? ordered_bits(query_scores[tile]) : 0;
            const bool tie = candidate && bits == threshold;
            const unsigned tie_mask = __ballot_sync(0xffffffff, tie);
            const int tie_rank = seen_ties + __popc(tie_mask & ((1u << lane) - 1));
            seen_ties += __popc(tie_mask);
            const bool flag =
                candidate ? bits > threshold || (tie && tie_rank < ties) : tile < tiles;
            const unsigned selected = __ballot_sync(0xffffffff, flag);
            if (flag) {
                route[count + __popc(selected & ((1u << lane) - 1))] = static_cast<uint16_t>(tile);
            }
            count += __popc(selected);
        }
        if (lane == 0) {
            workspace.route_counts[first_row + query] = count;
        }
    }
    __syncthreads();

    if (!coarse) {
        return;
    }
    for (int query = 0; query < queries; query++) {
        float *query_scores = scores + query * tiles;
        float maximum = -FLT_MAX;
        for (int tile = threadIdx.x; tile < tiles; tile += THREADS) {
            maximum = fmaxf(maximum, query_scores[tile]);
        }
        maximum = block_reduce<true>(maximum, scratch);
        // The weights divide by the tile lengths, since the pooled values are sums.
        float sum = 0.0f;
        for (int tile = threadIdx.x; tile < tiles; tile += THREADS) {
            const float weight = expf(query_scores[tile] - maximum);
            query_scores[tile] = weight / static_cast<float>(workspace.tile_lengths[tile]);
            sum += weight;
        }
        sum = block_reduce<false>(sum, scratch);
        if (threadIdx.x == 0) {
            maxima[query] = maximum;
            sums[query] = sum;
        }
    }
    __syncthreads();
    const float *values =
        workspace.pooled_value + static_cast<size_t>(head) * tiles * HEAD + threadIdx.x;
    float outputs[QUERIES] = {};
    for (int first = 0; first < tiles; first += VALUE_BATCH) {
        float loaded[VALUE_BATCH];
        #pragma unroll
        for (int batch = 0; batch < VALUE_BATCH; batch++) {
            loaded[batch] =
                first + batch < tiles ? values[static_cast<size_t>(first + batch) * HEAD] : 0.0f;
        }
        #pragma unroll
        for (int batch = 0; batch < VALUE_BATCH; batch++) {
            #pragma unroll
            for (int query = 0; query < QUERIES; query++) {
                if (first + batch < tiles) {
                    outputs[query] =
                        fmaf(scores[query * tiles + first + batch], loaded[batch], outputs[query]);
                }
            }
        }
    }
    #pragma unroll
    for (int query = 0; query < QUERIES; query++) {
        if (query < queries) {
            workspace.coarse[(first_row + query) * HEAD + threadIdx.x] =
                outputs[query] / sums[query];
        }
    }
}

template <int QUERIES>
int launch_select(const Mmh3VsaWorkspace &workspace, int tiles, int heads, int prefix_tiles,
                  int kept, float scale, int coarse, cudaStream_t stream) {
    constexpr size_t MAX_SHARED = 64 * 1024;
    const size_t shared = static_cast<size_t>(QUERIES) * tiles * sizeof(float);
    if constexpr (QUERIES > 1) {
        if (shared > MAX_SHARED) {
            return launch_select<QUERIES / 2>(workspace, tiles, heads, prefix_tiles, kept, scale,
                                              coarse, stream);
        }
    }
    // The limit only grows, so a launch never meets a smaller limit that another thread set.
    static std::mutex mutex;
    static size_t configured_shared = 0;
    static const size_t static_shared = [] {
        cudaFuncAttributes attributes{};
        cudaFuncGetAttributes(&attributes, select_kernel<QUERIES>);
        return attributes.sharedSizeBytes;
    }();
    if (static_shared + shared > 48 * 1024) {
        std::lock_guard<std::mutex> lock(mutex);
        if (shared > configured_shared) {
            cudaError_t status = cudaFuncSetAttribute(select_kernel<QUERIES>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                      static_cast<int>(shared));
            if (status != cudaSuccess) {
                return static_cast<int>(status);
            }
            configured_shared = shared;
        }
    }
    select_kernel<QUERIES>
        <<<dim3((tiles + QUERIES - 1) / QUERIES, heads), THREADS, shared, stream>>>(
            workspace, tiles, prefix_tiles, kept, scale, coarse);
    return static_cast<int>(cudaGetLastError());
}

__global__ void __launch_bounds__(THREADS, 3)
    attention_kernel(const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ key,
                     const __nv_bfloat16 *__restrict__ value,
                     const __nv_bfloat16 *__restrict__ gate, int64_t gate_stride,
                     __nv_bfloat16 *__restrict__ output, Mmh3AttentionLayout layout, int tiles,
                     Mmh3VsaWorkspace workspace, float scale_log2) {
    extern __shared__ __align__(128) uint8_t shared_memory[];
    const uint32_t shared_base = shared_address(shared_memory);
    const int query_tile = blockIdx.x;
    const int head = blockIdx.y;
    const int first_query = workspace.tile_starts[query_tile];
    const int query_rows = workspace.tile_lengths[query_tile];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int matrix = lane / 8;
    const int matrix_row = lane % 8;
    const size_t row = static_cast<size_t>(head) * tiles + query_tile;
    const uint16_t *route = workspace.routes + row * tiles;
    const int count = workspace.route_counts[row];

    const __nv_bfloat16 *query_head = query + head * layout.head_stride[QUERY];
    const __nv_bfloat16 *key_head = key + head * layout.head_stride[KEY];
    const __nv_bfloat16 *value_head = value + head * layout.head_stride[VALUE];
    // NOTE: keys and values have one buffer each. The next key tile loads while the warps apply
    // softmax and multiply the values, and the next value tile while they score the next keys.
    const uint32_t key_tile = shared_base;
    const uint32_t value_tile = shared_base + T::tile_bytes;
    // Rows past a tile's length are zero-filled and masked.
    auto load_tile = [&](uint32_t destination, const __nv_bfloat16 *head_base, int64_t stride,
                         int entry) {
        const int tile = route[entry];
        const int start = workspace.tile_starts[tile];
        load_rows<HEAD, THREADS>(destination, head_base, stride, start, BLOCK,
                                 start + workspace.tile_lengths[tile]);
    };

    // The query tile borrows the value buffer until it has been copied into registers.
    load_rows<HEAD, THREADS>(value_tile, query_head, layout.token_stride[QUERY], first_query, BLOCK,
                             first_query + query_rows);
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

        const int key_rows = workspace.tile_lengths[route[entry]];
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

    const float *coarse = gate != nullptr ? workspace.coarse + row * HEAD : nullptr;
    __nv_bfloat16 *output_head = output + head * layout.head_stride[OUTPUT];
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 2);
        const int tile_row = warp * 16 + half * 8 + group_id;
        if (tile_row >= query_rows) {
            continue;
        }
        const int64_t token = first_query + tile_row;
        const float inverse_sum = 1.0f / row_sum[half];
        __nv_bfloat16 *output_row = output_head + token * layout.token_stride[OUTPUT];
        const __nv_bfloat16 *gate_row =
            gate != nullptr ? gate + token * gate_stride + head * HEAD : nullptr;
        #pragma unroll
        for (int dimension_tile = 0; dimension_tile < HEAD / 8; dimension_tile++) {
            const int dimension = dimension_tile * 8 + (lane % 4) * 2;
            float first = output_accumulators[dimension_tile][half * 2] * inverse_sum;
            float second = output_accumulators[dimension_tile][half * 2 + 1] * inverse_sum;
            if (gate_row != nullptr) {
                const __nv_bfloat162 gates =
                    *reinterpret_cast<const __nv_bfloat162 *>(gate_row + dimension);
                first = fmaf(__low2float(gates), coarse[dimension], first);
                second = fmaf(__high2float(gates), coarse[dimension + 1], second);
            }
            *reinterpret_cast<uint32_t *>(output_row + dimension) = N::pack(first, second);
        }
    }
}

} // namespace

// VSA over one sequence of BF16 heads of 128 in tile order. The layout's batch strides are ignored.
// With `inputs_ready`, mmh3_attention_inputs has left the pooled tiles in the workspace. A
// non-null `quantized` workspace runs the token attention with INT8 QK and FP8 PV, which needs the
// prepared inputs.
// Video query tiles keep `kept` of the tiles after the `prefix_tiles` text and audio tiles. A
// non-null `gate` (BF16 rows of heads × 128 values, `gate_stride` apart) adds the coarse branch.
extern "C" int mmh3_vsa_attention(const void *query, const void *key, const void *value,
                                  const void *gate, int64_t gate_stride, void *output, int tokens,
                                  int heads, const Mmh3AttentionLayout *layout, float scale,
                                  int tiles, int prefix_tiles, int kept,
                                  const Mmh3VsaWorkspace *workspace,
                                  const Mmh3QuantizedWorkspace *quantized, int inputs_ready,
                                  cudaStream_t stream) {
    if (tokens <= 0 || heads <= 0 || tiles <= 0 || tiles > 65536 || prefix_tiles < 0 ||
        prefix_tiles > tiles || kept < 1 || (quantized != nullptr && !inputs_ready)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto *q = static_cast<const __nv_bfloat16 *>(query);
    const auto *k = static_cast<const __nv_bfloat16 *>(key);
    const auto *v = static_cast<const __nv_bfloat16 *>(value);
    if (!inputs_ready) {
        pool_kernel<<<dim3(tiles, heads), THREADS, 0, stream>>>(q, k, v, *layout, tiles,
                                                                *workspace);
    }
    const int selected = launch_select<4>(*workspace, tiles, heads, prefix_tiles, kept, scale,
                                          gate != nullptr, stream);
    if (selected != 0) {
        return selected;
    }
    if (quantized != nullptr) {
        return mmh3_vsa_attention_quantized(value, gate, gate_stride, output, tokens, heads, layout,
                                            scale, tiles, workspace, quantized, stream);
    }
    attention_kernel<<<dim3(tiles, heads), THREADS, ATTENTION_SHARED_BYTES, stream>>>(
        q, k, v, static_cast<const __nv_bfloat16 *>(gate), gate_stride,
        static_cast<__nv_bfloat16 *>(output), *layout, tiles, *workspace,
        scale * 1.4426950408889634f);
    return static_cast<int>(cudaGetLastError());
}
