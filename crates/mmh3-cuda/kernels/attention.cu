#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tensor_core.cuh"

// Dense attention in the FlashAttention-2 style for head dimensions 64 and 128, BF16 or FP16,
// bidirectional or causal, with grouped key and value heads. The strided layout covers the DiT's
// fused [q | k | v] projection, the VAE decoder's per-head [q, k, v] layout and the text encoder's
// [q | k | v] with fewer key and value heads alike.

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

template <typename Element, int HEAD_DIM>
__global__ void __launch_bounds__(THREADS, 1)
    attention_kernel(const Element *__restrict__ query, const Element *__restrict__ key,
                     const Element *__restrict__ value, Element *__restrict__ output, int tokens,
                     Mmh3AttentionLayout layout, float scale_log2) {
    using T = Tiles<HEAD_DIM>;
    using N = Numeric<Element>;
    extern __shared__ __align__(128) uint8_t shared_memory[];
    const uint32_t shared_base = shared_address(shared_memory);
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
        load_rows<HEAD_DIM, THREADS>(stage_address(stage), key_head, layout.token_stride[KEY],
                                     block * BLOCK_N, BLOCK_N, tokens);
        load_rows<HEAD_DIM, THREADS>(stage_address(stage) + T::tile_bytes, value_head,
                                     layout.token_stride[VALUE], block * BLOCK_N, BLOCK_N, tokens);
    };

    // The query tile borrows stage 1 until it has been copied into registers.
    load_rows<HEAD_DIM, THREADS>(stage_address(1), query_head, layout.token_stride[QUERY],
                                 first_query, BLOCK_M, tokens);
    copy_async_commit();
    load_key_value_block(0, 0);
    copy_async_commit();
    copy_async_wait<1>();
    __syncthreads();

    uint32_t query_fragments[HEAD_DIM / 16][4];
    #pragma unroll
    for (int k_step = 0; k_step < HEAD_DIM / 16; k_step++) {
        int row = warp * 16 + (matrix % 2) * 8 + matrix_row;
        int chunk = k_step * 2 + matrix / 2;
        load_matrix_x4(query_fragments[k_step], stage_address(1) + T::offset(row, chunk));
    }
    __syncthreads();

    float output_accumulators[HEAD_DIM / 8][4] = {};
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {0.0f, 0.0f};
    int blocks = (tokens + BLOCK_N - 1) / BLOCK_N;
    if (layout.causal) {
        blocks = min(blocks, (first_query + BLOCK_M + BLOCK_N - 1) / BLOCK_N);
    }

    for (int block = 0; block < blocks; block++) {
        if (block + 1 < blocks) {
            load_key_value_block(block + 1, (block + 1) % 2);
        }
        copy_async_commit();
        copy_async_wait<1>();
        __syncthreads();

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
                load_matrix_x4(registers, key_tile + T::offset(row, chunk));
                N::mma(scores[key_pair * 2], query_fragments[k_step], registers[0], registers[1]);
                N::mma(scores[key_pair * 2 + 1], query_fragments[k_step], registers[2],
                       registers[3]);
            }
        }

        // NOTE: every query row sees key 0 in the first block, so a causal row never starts from an
        // all-masked block.
        const bool masked_block = (block + 1) * BLOCK_N > tokens ||
                                  (layout.causal && (block + 1) * BLOCK_N - 1 > first_query);
        float block_max[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                float score = scores[key_tile_index][element] * scale_log2;
                if (masked_block) {
                    const int key_index =
                        block * BLOCK_N + key_tile_index * 8 + (lane % 4) * 2 + (element % 2);
                    const int query_index = first_query + warp * 16 + (element / 2) * 8 + lane / 4;
                    if (key_index >= tokens || (layout.causal && key_index > query_index)) {
                        score = -FLT_MAX;
                    }
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
                load_matrix_x4_transposed(registers, value_tile + T::offset(row, chunk));
                N::mma(output_accumulators[dimension_pair * 2], probability_fragment, registers[0],
                       registers[1]);
                N::mma(output_accumulators[dimension_pair * 2 + 1], probability_fragment,
                       registers[2], registers[3]);
            }
        }
        __syncthreads();
    }
    copy_async_wait<0>();

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

template <typename Element, int HEAD_DIM>
int launch(const void *query, const void *key, const void *value, void *output, int tokens,
           int heads, int batch, const Mmh3AttentionLayout &layout, float scale,
           cudaStream_t stream) {
    auto kernel = attention_kernel<Element, HEAD_DIM>;
    static const cudaError_t configured = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Tiles<HEAD_DIM>::shared_bytes);
    if (configured != cudaSuccess) {
        return static_cast<int>(configured);
    }
    Mmh3AttentionLayout normalized = layout;
    normalized.heads_per_key_value =
        layout.heads_per_key_value > 0 ? layout.heads_per_key_value : 1;
    dim3 grid((tokens + BLOCK_M - 1) / BLOCK_M, heads, batch);
    kernel<<<grid, THREADS, Tiles<HEAD_DIM>::shared_bytes, stream>>>(
        static_cast<const Element *>(query), static_cast<const Element *>(key),
        static_cast<const Element *>(value), static_cast<Element *>(output), tokens, normalized,
        scale * 1.4426950408889634f);
    return static_cast<int>(cudaGetLastError());
}

} // namespace

// element_type 0 is BF16 and 1 is FP16. head_dim is 64 or 128.
extern "C" int mmh3_attention(int element_type, int head_dim, const void *query, const void *key,
                              const void *value, void *output, int tokens, int heads, int batch,
                              const Mmh3AttentionLayout *layout, float scale, cudaStream_t stream) {
    if (tokens <= 0 || heads <= 0 || batch <= 0 ||
        (layout->heads_per_key_value > 0 && heads % layout->heads_per_key_value != 0)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (element_type == 0 && head_dim == 128) {
        return launch<__nv_bfloat16, 128>(query, key, value, output, tokens, heads, batch, *layout,
                                          scale, stream);
    }
    if (element_type == 1 && head_dim == 64) {
        return launch<__half, 64>(query, key, value, output, tokens, heads, batch, *layout, scale,
                                  stream);
    }
    if (element_type == 0 && head_dim == 64) {
        return launch<__nv_bfloat16, 64>(query, key, value, output, tokens, heads, batch, *layout,
                                         scale, stream);
    }
    if (element_type == 1 && head_dim == 128) {
        return launch<__half, 128>(query, key, value, output, tokens, heads, batch, *layout, scale,
                                   stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
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
