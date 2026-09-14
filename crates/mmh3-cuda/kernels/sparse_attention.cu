#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "attention_workspace.cuh"
#include "tensor_core.cuh"

// Sol-Attn block-sparse attention for BF16 heads of 128 (see mmh3-core's dit/sparse.rs for the
// algorithm).
//
// 1. block_stats: per head and 64-token block, the mean query (centroid), mean key and summed
// value.
// 2. center_keys: per head, the mean and variance of the block keys, which are then centered.
// 3. route: per head and query block, the pooled scores against every key block, the routed key
// blocks and the
//    pooled tail of the rest as an online-softmax state (max, weighted sum, weighted values).
// 4. row_offsets: per token and head, q · key mean, which centers the keys of the token-level
// scores.
// 5. sparse_attention: FlashAttention over the routed key blocks of each query block, merged with
// its tail.

namespace {

enum Operand { QUERY = 0, KEY = 1, VALUE = 2, OUTPUT = 3 };

constexpr int HEAD = 128;
constexpr int BLOCK = 64;
constexpr int WARPS = BLOCK / 16;
constexpr int THREADS = WARPS * 32;
using T = Tiles<HEAD>;
using N = Numeric<__nv_bfloat16>;
static_assert(BLOCK * T::row_bytes <= T::tile_bytes, "the query tile must fit in the value buffer");
// A key and a value buffer, small enough for two CTAs per SM.
constexpr int ATTENTION_SHARED_BYTES = 2 * T::tile_bytes;
static_assert(THREADS == HEAD, "the routing kernel gives each thread one dimension");

__device__ __forceinline__ int block_rows(int block, int tokens) {
    return min(BLOCK, tokens - block * BLOCK);
}

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

// Sum or max over the block of THREADS threads.
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

__global__ void __launch_bounds__(THREADS)
    block_stats_kernel(const __nv_bfloat16 *__restrict__ query,
                       const __nv_bfloat16 *__restrict__ key,
                       const __nv_bfloat16 *__restrict__ value, int tokens,
                       Mmh3AttentionLayout layout, int blocks, float *__restrict__ centroids,
                       float *__restrict__ block_keys, float *__restrict__ value_sums) {
    const int block = blockIdx.x;
    const int head = blockIdx.y;
    const int dimension = threadIdx.x;
    const int rows = block_rows(block, tokens);
    float query_sum = 0.0f, key_sum = 0.0f, value_sum = 0.0f;
    for (int row = 0; row < rows; row++) {
        const int64_t token = static_cast<int64_t>(block) * BLOCK + row;
        query_sum += __bfloat162float(query[token * layout.token_stride[QUERY] +
                                            head * layout.head_stride[QUERY] + dimension]);
        key_sum += __bfloat162float(
            key[token * layout.token_stride[KEY] + head * layout.head_stride[KEY] + dimension]);
        value_sum += __bfloat162float(value[token * layout.token_stride[VALUE] +
                                            head * layout.head_stride[VALUE] + dimension]);
    }
    const size_t index = (static_cast<size_t>(head) * blocks + block) * HEAD + dimension;
    centroids[index] = query_sum / rows;
    block_keys[index] = key_sum / rows;
    value_sums[index] = value_sum;
}

__global__ void __launch_bounds__(HEAD)
    center_keys_kernel(float *__restrict__ block_keys, int blocks, float *__restrict__ key_mean,
                       float *__restrict__ key_variance) {
    const int head = blockIdx.x;
    const int dimension = threadIdx.x;
    float *keys = block_keys + static_cast<size_t>(head) * blocks * HEAD + dimension;
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
        sum += keys[static_cast<size_t>(block) * HEAD];
    }
    const float mean = sum / blocks;
    float squares = 0.0f;
    for (int block = 0; block < blocks; block++) {
        const float centered = keys[static_cast<size_t>(block) * HEAD] - mean;
        keys[static_cast<size_t>(block) * HEAD] = centered;
        squares += centered * centered;
    }
    key_mean[head * HEAD + dimension] = mean;
    key_variance[head * HEAD + dimension] = squares / blocks;
}

// Routes QUERIES consecutive query blocks of one head, which share every block key and value sum
// the CTA loads.
//
// NOTE: the _rn intrinsics spell out the multiply-adds that the compiler forms from the plain
// expressions. A warp scores 32 (query, key block) pairs at once and reduces them together over the
// same xor butterfly as warp_sum, and the loops read several key blocks at once so that their loads
// overlap. Every sum keeps its order, so the results do not depend on QUERIES.
template <int QUERIES>
__global__ void __launch_bounds__(THREADS)
    route_kernel(Mmh3SparseWorkspace workspace, int tokens, int blocks, float tau, float log2_scale,
                 int sink_key_start, int sink_key_end, int sink_query_start, int sink_query_end) {
    constexpr int ROUTE_BATCH = 32 / QUERIES;
    constexpr int TAIL_BATCH = 32;
    extern __shared__ float weights[];
    uint8_t *routed = reinterpret_cast<uint8_t *>(weights + QUERIES * blocks);
    __shared__ float centroids[QUERIES][HEAD];
    __shared__ float thresholds[QUERIES];
    __shared__ float tail_maxima[QUERIES];
    __shared__ float tail_sums[QUERIES];
    __shared__ float scratch[THREADS / 32];
    const int first_query = blockIdx.x * QUERIES;
    const int queries = min(QUERIES, blocks - first_query);
    const int head = blockIdx.y;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const size_t first_row = static_cast<size_t>(head) * blocks + first_query;

    for (int query = 0; query < queries; query++) {
        const float centroid = workspace.centroids[(first_row + query) * HEAD + threadIdx.x];
        centroids[query][threadIdx.x] = centroid;
        const float spread =
            block_reduce<false>(__fmul_rn(__fmul_rn(centroid, centroid),
                                          workspace.key_variance[head * HEAD + threadIdx.x]),
                                scratch);
        if (threadIdx.x == 0) {
            thresholds[query] =
                __fmul_rn(tau, sqrtf(__fmaf_rn(__fmul_rn(spread, log2_scale), log2_scale, 1e-6f)));
        }
    }
    __syncthreads();

    const float *keys = workspace.block_keys + static_cast<size_t>(head) * blocks * HEAD;
    for (int first = warp * ROUTE_BATCH; first < blocks; first += THREADS / 32 * ROUTE_BATCH) {
        float loaded[ROUTE_BATCH][HEAD / 32];
#pragma unroll
        for (int batch = 0; batch < ROUTE_BATCH; batch++) {
#pragma unroll
            for (int part = 0; part < HEAD / 32; part++) {
                loaded[batch][part] =
                    first + batch < blocks
                        ? keys[static_cast<size_t>(first + batch) * HEAD + lane + part * 32]
                        : 0.0f;
            }
        }
        float partials[32];
#pragma unroll
        for (int query = 0; query < QUERIES; query++) {
#pragma unroll
            for (int batch = 0; batch < ROUTE_BATCH; batch++) {
                float partial = 0.0f;
#pragma unroll
                for (int part = 0; part < HEAD / 32; part++) {
                    partial =
                        __fmaf_rn(centroids[query][lane + part * 32], loaded[batch][part], partial);
                }
                partials[query * ROUTE_BATCH + batch] = partial;
            }
        }
        // Each step adds the partner lane's half of the pairs, so lane l ends up with the sum of
        // pair l.
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            const bool upper = lane & offset;
#pragma unroll
            for (int index = 0; index < offset; index++) {
                const float kept = upper ? partials[index + offset] : partials[index];
                const float sent = upper ? partials[index] : partials[index + offset];
                partials[index] = kept + __shfl_xor_sync(0xffffffff, sent, offset);
            }
        }
        const int query = lane / ROUTE_BATCH;
        const int key_block = first + lane % ROUTE_BATCH;
        if (query < queries && key_block < blocks) {
            const int query_block = first_query + query;
            const float score = partials[0] * log2_scale;
            weights[query * blocks + key_block] = score;
            routed[query * blocks + key_block] =
                score > thresholds[query] || abs(query_block - key_block) <= 1 ||
                (key_block >= sink_key_start && key_block < sink_key_end) ||
                (query_block >= sink_query_start && query_block < sink_query_end);
        }
    }
    __syncthreads();

    for (int query = 0; query < queries; query++) {
        float *query_weights = weights + query * blocks;
        const uint8_t *query_routed = routed + query * blocks;
        float maximum = -FLT_MAX;
        for (int key_block = threadIdx.x; key_block < blocks; key_block += THREADS) {
            if (!query_routed[key_block]) {
                maximum = fmaxf(maximum, query_weights[key_block]);
            }
        }
        maximum = block_reduce<true>(maximum, scratch);
        float sum = 0.0f;
        for (int key_block = threadIdx.x; key_block < blocks; key_block += THREADS) {
            const float weight =
                query_routed[key_block] ? 0.0f : exp2f(query_weights[key_block] - maximum);
            query_weights[key_block] = weight;
            sum = __fmaf_rn(weight, static_cast<float>(block_rows(key_block, tokens)), sum);
        }
        sum = block_reduce<false>(sum, scratch);
        if (threadIdx.x == 0) {
            tail_maxima[query] = maximum;
            tail_sums[query] = sum;
        }
    }
    __syncthreads();

    const float *values =
        workspace.value_sums + static_cast<size_t>(head) * blocks * HEAD + threadIdx.x;
    float tails[QUERIES] = {};
    for (int first = 0; first < blocks; first += TAIL_BATCH) {
        float loaded[TAIL_BATCH];
#pragma unroll
        for (int batch = 0; batch < TAIL_BATCH; batch++) {
            loaded[batch] =
                first + batch < blocks ? values[static_cast<size_t>(first + batch) * HEAD] : 0.0f;
        }
#pragma unroll
        for (int batch = 0; batch < TAIL_BATCH; batch++) {
            const int key_block = first + batch;
#pragma unroll
            for (int query = 0; query < QUERIES; query++) {
                if (query < queries && key_block < blocks && !routed[query * blocks + key_block]) {
                    tails[query] =
                        __fmaf_rn(weights[query * blocks + key_block], loaded[batch], tails[query]);
                }
            }
        }
    }
#pragma unroll
    for (int query = 0; query < QUERIES; query++) {
        if (query < queries) {
            workspace.tail_values[(first_row + query) * HEAD + threadIdx.x] = tails[query];
        }
    }

    for (int query = warp; query < queries; query += THREADS / 32) {
        const size_t row = first_row + query;
        uint16_t *route = workspace.routes + row * blocks;
        int count = 0;
        for (int start = 0; start < blocks; start += 32) {
            const int key_block = start + lane;
            const bool flag = key_block < blocks && routed[query * blocks + key_block];
            const unsigned mask = __ballot_sync(0xffffffff, flag);
            if (flag) {
                route[count + __popc(mask & ((1u << lane) - 1))] = static_cast<uint16_t>(key_block);
            }
            count += __popc(mask);
        }
        if (lane == 0) {
            workspace.route_counts[row] = count;
            workspace.tail_max[row] = tail_maxima[query];
            workspace.tail_sum[row] = tail_sums[query];
        }
    }
}

// Launches route_kernel with as many query blocks per CTA as their weights and flags fit in shared
// memory.
template <int QUERIES>
int launch_route(const Mmh3SparseWorkspace &workspace, int tokens, int blocks, int heads, float tau,
                 float log2_scale, int sink_key_start, int sink_key_end, int sink_query_start,
                 int sink_query_end, cudaStream_t stream) {
    constexpr size_t MAX_SHARED = 90 * 1024;
    const size_t shared = static_cast<size_t>(QUERIES) * blocks * (sizeof(float) + 1);
    if constexpr (QUERIES > 1) {
        if (shared > MAX_SHARED) {
            return launch_route<QUERIES / 2>(workspace, tokens, blocks, heads, tau, log2_scale,
                                             sink_key_start, sink_key_end, sink_query_start,
                                             sink_query_end, stream);
        }
    }
    static size_t configured_shared = 0;
    if (shared > 48 * 1024 && shared > configured_shared) {
        cudaError_t status =
            cudaFuncSetAttribute(route_kernel<QUERIES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(shared));
        if (status != cudaSuccess) {
            return static_cast<int>(status);
        }
        configured_shared = shared;
    }
    route_kernel<QUERIES>
        <<<dim3((blocks + QUERIES - 1) / QUERIES, heads), THREADS, shared, stream>>>(
            workspace, tokens, blocks, tau, log2_scale, sink_key_start, sink_key_end,
            sink_query_start, sink_query_end);
    return static_cast<int>(cudaGetLastError());
}

__global__ void row_offsets_kernel(const __nv_bfloat16 *__restrict__ query, int tokens, int heads,
                                   Mmh3AttentionLayout layout, const float *__restrict__ key_mean,
                                   float log2_scale, float *__restrict__ row_offsets) {
    const int lane = threadIdx.x % 32;
    const int64_t item = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (item >= static_cast<int64_t>(tokens) * heads) {
        return;
    }
    const int head = static_cast<int>(item % heads);
    const int64_t token = item / heads;
    const __nv_bfloat16 *row =
        query + token * layout.token_stride[QUERY] + head * layout.head_stride[QUERY];
    float partial = 0.0f;
    for (int dimension = lane; dimension < HEAD; dimension += 32) {
        partial += __bfloat162float(row[dimension]) * key_mean[head * HEAD + dimension];
    }
    partial = warp_sum(partial);
    if (lane == 0) {
        row_offsets[static_cast<size_t>(head) * tokens + token] = partial * log2_scale;
    }
}

__global__ void __launch_bounds__(THREADS, 2) sparse_attention_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ key,
    const __nv_bfloat16 *__restrict__ value, __nv_bfloat16 *__restrict__ output, int tokens,
    Mmh3AttentionLayout layout, int blocks, Mmh3SparseWorkspace workspace, float scale_log2) {
    extern __shared__ __align__(128) uint8_t shared_memory[];
    const uint32_t shared_base = shared_address(shared_memory);
    const int query_block = blockIdx.x;
    const int head = blockIdx.y;
    const int first_query = query_block * BLOCK;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int matrix = lane / 8;
    const int matrix_row = lane % 8;
    const size_t row = static_cast<size_t>(head) * blocks + query_block;
    const uint16_t *route = workspace.routes + row * blocks;
    const int count = workspace.route_counts[row];

    const __nv_bfloat16 *query_head = query + head * layout.head_stride[QUERY];
    const __nv_bfloat16 *key_head = key + head * layout.head_stride[KEY];
    const __nv_bfloat16 *value_head = value + head * layout.head_stride[VALUE];
    // NOTE: keys and values have one buffer each. The next key tile loads while the warps apply
    // softmax and multiply the values, and the next value tile while they score the next keys.
    const uint32_t key_tile = shared_base;
    const uint32_t value_tile = shared_base + T::tile_bytes;
    auto load_keys = [&](int entry) {
        load_rows<HEAD, THREADS>(key_tile, key_head, layout.token_stride[KEY], route[entry] * BLOCK,
                                 BLOCK, tokens);
    };
    auto load_values = [&](int entry) {
        load_rows<HEAD, THREADS>(value_tile, value_head, layout.token_stride[VALUE],
                                 route[entry] * BLOCK, BLOCK, tokens);
    };

    // The query tile borrows the value buffer until it has been copied into registers.
    load_rows<HEAD, THREADS>(value_tile, query_head, layout.token_stride[QUERY], first_query, BLOCK,
                             tokens);
    if (count > 0) {
        load_keys(0);
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
        load_values(0);
    }
    copy_async_commit();

    const int group_id = lane / 4;
    float offsets[2];
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const int token = first_query + warp * 16 + half * 8 + group_id;
        offsets[half] = token < tokens
                            ? workspace.row_offsets[static_cast<size_t>(head) * tokens + token]
                            : 0.0f;
    }
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
            load_keys(entry + 1);
        }
        copy_async_commit();

        const int first_key = route[entry] * BLOCK;
        const bool partial_block = first_key + BLOCK > tokens;
        float block_max[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK / 8; key_tile_index++) {
#pragma unroll
            for (int element = 0; element < 4; element++) {
                float score = scores[key_tile_index][element] * scale_log2 - offsets[element / 2];
                if (partial_block &&
                    first_key + key_tile_index * 8 + (lane % 4) * 2 + (element % 2) >= tokens) {
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
            load_values(entry + 1);
        }
        copy_async_commit();
    }
    copy_async_wait<0>();

    // Merge the routed blocks with the pooled tail of the query block.
    const float tail_max = workspace.tail_max[row];
    const float tail_sum = workspace.tail_sum[row];
    const float *tail_values = workspace.tail_values + row * HEAD;
    __nv_bfloat16 *output_head = output + head * layout.head_stride[OUTPUT];
#pragma unroll
    for (int half = 0; half < 2; half++) {
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 2);
        const int token = first_query + warp * 16 + half * 8 + group_id;
        if (token >= tokens) {
            continue;
        }
        const float merged_max = fmaxf(row_max[half], tail_max);
        const float routed_weight = exp2f(row_max[half] - merged_max);
        const float tail_weight = exp2f(tail_max - merged_max);
        const float inverse_sum = 1.0f / (row_sum[half] * routed_weight + tail_sum * tail_weight);
        __nv_bfloat16 *output_row =
            output_head + static_cast<int64_t>(token) * layout.token_stride[OUTPUT];
#pragma unroll
        for (int dimension_tile = 0; dimension_tile < HEAD / 8; dimension_tile++) {
            const int dimension = dimension_tile * 8 + (lane % 4) * 2;
            const float first = (output_accumulators[dimension_tile][half * 2] * routed_weight +
                                 tail_values[dimension] * tail_weight) *
                                inverse_sum;
            const float second =
                (output_accumulators[dimension_tile][half * 2 + 1] * routed_weight +
                 tail_values[dimension + 1] * tail_weight) *
                inverse_sum;
            *reinterpret_cast<uint32_t *>(output_row + dimension) = N::pack(first, second);
        }
    }
}

} // namespace

// Sol-Attn over one sequence of BF16 heads of 128. The layout's batch strides are ignored. With
// `inputs_ready`, the workspaces already hold the block statistics and, for quantized attention,
// the INT8 inputs of mmh3_attention_inputs.
extern "C" int mmh3_sparse_attention(const void *query, const void *key, const void *value,
                                     void *output, int tokens, int heads,
                                     const Mmh3AttentionLayout *layout, float scale, float tau,
                                     int sink_key_start, int sink_key_end, int sink_query_start,
                                     int sink_query_end, const Mmh3SparseWorkspace *workspace,
                                     const Mmh3QuantizedWorkspace *quantized, int inputs_ready,
                                     cudaStream_t stream) {
    if (tokens <= 0 || heads <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int blocks = (tokens + BLOCK - 1) / BLOCK;
    const float log2_scale = scale * 1.4426950408889634f;
    const auto *q = static_cast<const __nv_bfloat16 *>(query);
    const auto *k = static_cast<const __nv_bfloat16 *>(key);
    const auto *v = static_cast<const __nv_bfloat16 *>(value);

    if (!inputs_ready) {
        block_stats_kernel<<<dim3(blocks, heads), THREADS, 0, stream>>>(
            q, k, v, tokens, *layout, blocks, workspace->centroids, workspace->block_keys,
            workspace->value_sums);
    }
    center_keys_kernel<<<heads, HEAD, 0, stream>>>(workspace->block_keys, blocks,
                                                   workspace->key_mean, workspace->key_variance);
    const int routed =
        launch_route<4>(*workspace, tokens, blocks, heads, tau, log2_scale, sink_key_start,
                        sink_key_end, sink_query_start, sink_query_end, stream);
    if (routed != 0) {
        return routed;
    }
    const int64_t items = static_cast<int64_t>(tokens) * heads;
    row_offsets_kernel<<<static_cast<unsigned>((items + 7) / 8), 256, 0, stream>>>(
        q, tokens, heads, *layout, workspace->key_mean, log2_scale, workspace->row_offsets);
    if (quantized != nullptr) {
        return mmh3_attention_quantized(query, key, value, output, tokens, heads, layout, scale,
                                        workspace, quantized, inputs_ready, stream);
    }
    sparse_attention_kernel<<<dim3(blocks, heads), THREADS, ATTENTION_SHARED_BYTES, stream>>>(
        q, k, v, static_cast<__nv_bfloat16 *>(output), tokens, *layout, blocks, *workspace,
        log2_scale);
    return static_cast<int>(cudaGetLastError());
}
