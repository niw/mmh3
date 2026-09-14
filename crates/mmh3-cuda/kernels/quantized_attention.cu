#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "attention_workspace.cuh"
#include "tensor_core.cuh"

// Quantized DiT attention: Q/K use symmetric INT8 scales per head and 64-token block.
// V uses FP8 E4M3 with one scale per head, and is transposed for the PV MMA. Probabilities
// are scaled by 448 before FP8 conversion. Softmax, the pooled sparse tail, and accumulation
// stay in FP32. Both paths read BF16 inputs and write BF16 output.

namespace {

enum Operand { QUERY = 0, KEY = 1, VALUE = 2, OUTPUT = 3 };

constexpr int HEAD_DIM = 128;
constexpr int BLOCK_M = 64;
constexpr int WARPS = BLOCK_M / 16;
constexpr int THREADS = WARPS * 32;
constexpr int SHARED_BYTES = 32768;
using Element = __nv_bfloat16;
using QKAccumulator = int32_t;

// A stage holds an 8 KiB key tile and an 8 KiB transposed value tile of 64-byte rows. Two stages
// take 32 KiB, so that three CTAs share an SM.
struct QuantizedTiles {
    static constexpr int tile_bytes = 8192;
    static constexpr int stage_bytes = 16384;
};

// Swizzling the 16-byte chunk of a 64-byte value row by (row / 2) % 4 spreads every ldmatrix phase
// over eight distinct bank groups.
__device__ __forceinline__ int value_offset(int row, int chunk) {
    return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
}
static_assert(BLOCK_M * HEAD_DIM <= QuantizedTiles::stage_bytes,
              "the query tile borrows one stage");

__device__ __forceinline__ void mma_qk(QKAccumulator (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                       uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, "
                 "%7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_pv(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                       uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                 "%6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// exp2f without its handling of results below 2^-126, which it flushes to zero. Probabilities and
// corrections that small vanish in FP8 and against the row sums, which reach at least one.
__device__ __forceinline__ float exp2_flushed(float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ void load_qk8(uint32_t destination, const uint8_t *source, int64_t stride, int first,
                         int count, int tokens) {
    using Q = Tiles<64>;
    for (int index = threadIdx.x; index < count * 8; index += THREADS) {
        const int row = index / 8;
        const int chunk = index % 8;
        const bool valid = first + row < tokens;
        copy_async_16(
            destination + Q::offset(row, chunk),
            source + (valid ? static_cast<int64_t>(first + row) * stride : 0) + chunk * 16, valid);
    }
}

// First reduce V within each block, then reduce the small block-maxima array per head.
__global__ void value_max(const __nv_bfloat16 *value, int tokens, Mmh3AttentionLayout layout,
                          float *partial) {
    __shared__ float scratch[8];
    const int block = blockIdx.x;
    const int head = blockIdx.y;
    float maximum = 0.0f;
    for (int index = threadIdx.x; index < 64 * HEAD_DIM; index += 256) {
        const int token = block * 64 + index / HEAD_DIM;
        const int dimension = index % HEAD_DIM;
        if (token < tokens) {
            maximum =
                fmaxf(maximum, fabsf(__bfloat162float(
                                   value[static_cast<int64_t>(token) * layout.token_stride[VALUE] +
                                         head * layout.head_stride[VALUE] + dimension])));
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = maximum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int warp = 0; warp < 8; warp++) {
            maximum = fmaxf(maximum, scratch[warp]);
        }
        partial[head * ((tokens + 63) / 64) + block] = maximum;
    }
}

__global__ void value_scales(const float *partial, float *scales, int blocks) {
    float maximum = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        maximum = fmaxf(maximum, partial[blockIdx.x * blocks + block]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    if (threadIdx.x == 0) {
        scales[blockIdx.x] = fmaxf(maximum / 448.0f, 1e-30f);
    }
}

// Quantize and transpose V in 32-by-32 tiles, padding the last key block with zeroes.
__global__ void quantize_value(const __nv_bfloat16 *value, uint8_t *output, const float *scales,
                               int tokens, int padded, Mmh3AttentionLayout layout) {
    __shared__ uint8_t tile[32][33];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int head = blockIdx.z;
    const float scale = scales[head];
    for (int part = 0; part < 32; part += 8) {
        const int token = blockIdx.x * 32 + y + part;
        const int dimension = blockIdx.y * 32 + x;
        const float v =
            token < tokens
                ? __bfloat162float(value[static_cast<int64_t>(token) * layout.token_stride[VALUE] +
                                         head * layout.head_stride[VALUE] + dimension]) /
                      scale
                : 0.0f;
        tile[y + part][x] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
    __syncthreads();
    for (int part = 0; part < 32; part += 8) {
        const int token = blockIdx.x * 32 + x;
        const int dimension = blockIdx.y * 32 + y + part;
        output[(static_cast<int64_t>(head) * HEAD_DIM + dimension) * padded + token] =
            tile[x][y + part];
    }
}

__global__ void quantize_qk(const __nv_bfloat16 *query, const __nv_bfloat16 *key, uint8_t *query8,
                            uint8_t *key8, float *query_scales, float *key_scales, int tokens,
                            int heads, Mmh3AttentionLayout layout) {
    __shared__ float scratch[8];
    const int block = blockIdx.x;
    const int head = blockIdx.y;
    const int part = blockIdx.z;
    const int blocks = (tokens + 63) / 64;
    const __nv_bfloat16 *source = part ? key : query;
    uint8_t *destination = part ? key8 : query8;
    float *scales = part ? key_scales : query_scales;
    float maximum = 0.0f;
    for (int index = threadIdx.x; index < 64 * HEAD_DIM; index += 256) {
        const int token = block * 64 + index / HEAD_DIM;
        const int dimension = index % HEAD_DIM;
        if (token < tokens) {
            maximum =
                fmaxf(maximum, fabsf(__bfloat162float(
                                   source[static_cast<int64_t>(token) * layout.token_stride[part] +
                                          head * layout.head_stride[part] + dimension])));
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = maximum;
    }
    __syncthreads();
    maximum = scratch[0];
    for (int warp = 1; warp < 8; warp++) {
        maximum = fmaxf(maximum, scratch[warp]);
    }
    const float scale = fmaxf(maximum / 127.0f, 1e-30f);
    if (threadIdx.x == 0) {
        scales[head * blocks + block] = scale;
    }
    for (int index = threadIdx.x; index < 64 * HEAD_DIM; index += 256) {
        const int token = block * 64 + index / HEAD_DIM;
        const int dimension = index % HEAD_DIM;
        if (token < tokens) {
            const float value =
                __bfloat162float(source[static_cast<int64_t>(token) * layout.token_stride[part] +
                                        head * layout.head_stride[part] + dimension]) /
                scale;
            destination[(static_cast<int64_t>(token) * heads + head) * HEAD_DIM + dimension] =
                static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(value)));
        }
    }
}

template <bool SPARSE>
__global__ void __launch_bounds__(THREADS, 3)
    attention_kernel(const uint8_t *__restrict__ query, const uint8_t *__restrict__ key,
                     const uint8_t *__restrict__ value, Element *__restrict__ output, int tokens,
                     Mmh3AttentionLayout layout, float scale_log2, const float *q_scales,
                     const float *k_scales, const float *value_scale,
                     Mmh3SparseWorkspace workspace) {
    using T = QuantizedTiles;
    using Q = Tiles<64>;
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
    const int route_blocks = (tokens + 63) / 64;
    const size_t route_row = static_cast<size_t>(head) * route_blocks + blockIdx.x;

    const int key_value_head = head / layout.heads_per_key_value;
    auto head_base = [&](const auto *base, Operand operand) {
        const int operand_head = operand == KEY || operand == VALUE ? key_value_head : head;
        return base + batch * layout.batch_stride[operand] +
               operand_head * layout.head_stride[operand];
    };
    const auto *query_head = head_base(query, QUERY);
    const auto *key_head = head_base(key, KEY);
    const int padded_tokens = ((tokens + 63) / 64) * 64;
    const uint8_t *value_head = value + static_cast<int64_t>(head) * HEAD_DIM * padded_tokens;

    auto stage_address = [&](int stage) { return shared_base + stage * T::stage_bytes; };
    // Every thread copies the same 16-byte chunks of each key and value tile, so their offsets are
    // computed once.
    constexpr int CHUNKS = BLOCK_N * 8 / THREADS;
    uint32_t key_shared[CHUNKS], value_shared[CHUNKS];
    int64_t key_offsets[CHUNKS];
    int key_rows[CHUNKS], value_offsets[CHUNKS];
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS; chunk++) {
        const int index = threadIdx.x + chunk * THREADS;
        key_rows[chunk] = index / 8;
        key_shared[chunk] = Q::offset(index / 8, index % 8);
        key_offsets[chunk] =
            static_cast<int64_t>(index / 8) * layout.token_stride[KEY] + (index % 8) * 16;
        value_shared[chunk] = T::tile_bytes + value_offset(index / 4, index % 4);
        value_offsets[chunk] = (index / 4) * padded_tokens + (index % 4) * 16;
    }
    auto load_key_value_block = [&](int entry, int stage) {
        const int block = SPARSE ? workspace.routes[route_row * route_blocks + entry] : entry;
        const uint8_t *key_block =
            key_head + static_cast<int64_t>(block) * BLOCK_N * layout.token_stride[KEY];
        const uint8_t *value_block = value_head + block * BLOCK_N;
        const uint32_t base = stage_address(stage);
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS; chunk++) {
            const bool valid = block * BLOCK_N + key_rows[chunk] < tokens;
            copy_async_16(base + key_shared[chunk],
                          valid ? key_block + key_offsets[chunk] : key_head, valid);
        }
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS; chunk++) {
            copy_async_16(base + value_shared[chunk], value_block + value_offsets[chunk], true);
        }
    };

    // The query tile borrows stage 1 until it has been copied into registers.
    load_qk8(stage_address(1), query_head, layout.token_stride[QUERY], first_query, BLOCK_M,
             tokens);
    copy_async_commit();
    load_key_value_block(0, 0);
    copy_async_commit();
    copy_async_wait<1>();
    __syncthreads();

    uint32_t query_fragments[HEAD_DIM / 32][4];
#pragma unroll
    for (int k_step = 0; k_step < HEAD_DIM / 32; k_step++) {
        int row = warp * 16 + (matrix % 2) * 8 + matrix_row;
        int chunk = k_step * 2 + matrix / 2;
        load_matrix_x4(query_fragments[k_step], stage_address(1) + Q::offset(row, chunk));
    }
    __syncthreads();

    float output_accumulators[HEAD_DIM / 8][4] = {};
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {0.0f, 0.0f};
    const int blocks = SPARSE ? workspace.route_counts[route_row] : route_blocks;
    const uint16_t *route = SPARSE ? workspace.routes + route_row * route_blocks : nullptr;
    float offsets[2];
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const int token = first_query + warp * 16 + half * 8 + lane / 4;
        offsets[half] = SPARSE && token < tokens
                            ? workspace.row_offsets[static_cast<size_t>(head) * tokens + token]
                            : 0.0f;
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

        QKAccumulator raw_scores[BLOCK_N / 8][4] = {};
        float scores[BLOCK_N / 8][4];
#pragma unroll
        for (int k_step = 0; k_step < HEAD_DIM / 32; k_step++) {
#pragma unroll
            for (int key_pair = 0; key_pair < BLOCK_N / 16; key_pair++) {
                int row = key_pair * 16 + (matrix / 2) * 8 + matrix_row;
                int chunk = k_step * 2 + matrix % 2;
                uint32_t registers[4];
                load_matrix_x4(registers, key_tile + Q::offset(row, chunk));
                mma_qk(raw_scores[key_pair * 2], query_fragments[k_step], registers[0],
                       registers[1]);
                mma_qk(raw_scores[key_pair * 2 + 1], query_fragments[k_step], registers[2],
                       registers[3]);
            }
        }

        const int key_block = SPARSE ? route[block] : block;
        const bool masked_block = (key_block + 1) * BLOCK_N > tokens;
        float block_max[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
#pragma unroll
            for (int element = 0; element < 4; element++) {
                float score = static_cast<float>(raw_scores[key_tile_index][element]) *
                                  (scale_log2 *
                                   q_scales[head * route_blocks + (first_query + warp * 16) / 64] *
                                   k_scales[head * route_blocks + key_block]) -
                              offsets[element / 2];
                if (masked_block) {
                    const int key_index =
                        key_block * BLOCK_N + key_tile_index * 8 + (lane % 4) * 2 + (element % 2);
                    if (key_index >= tokens) {
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
            correction[half] = exp2_flushed(row_max[half] - new_max);
            row_max[half] = new_max;
            row_sum[half] *= correction[half];
        }
#pragma unroll
        for (int key_tile_index = 0; key_tile_index < BLOCK_N / 8; key_tile_index++) {
#pragma unroll
            for (int element = 0; element < 4; element++) {
                float probability =
                    exp2_flushed(scores[key_tile_index][element] - row_max[element / 2]);
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

        // Repack two adjacent FP8 pairs with word shuffles into MMA's four A registers.
#pragma unroll
        for (int key_step = 0; key_step < BLOCK_N / 32; key_step++) {
            uint32_t probability_fragment[4];
#pragma unroll
            for (int key_half = 0; key_half < 2; key_half++) {
                const int tile_base = key_step * 4 + key_half * 2;
                const int source_lane = (lane & ~3) + 2 * (lane % 2);
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    uint32_t pairs = static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(
                        make_float2(scores[tile_base][half * 2] * 448.0f,
                                    scores[tile_base][half * 2 + 1] * 448.0f),
                        __NV_SATFINITE, __NV_E4M3));
                    pairs |= static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(
                                 make_float2(scores[tile_base + 1][half * 2] * 448.0f,
                                             scores[tile_base + 1][half * 2 + 1] * 448.0f),
                                 __NV_SATFINITE, __NV_E4M3))
                             << 16;
                    const uint32_t first =
                        __shfl_sync(0xffffffff, pairs, source_lane) >> ((lane & 2) * 8);
                    const uint32_t second =
                        __shfl_sync(0xffffffff, pairs, source_lane + 1) >> ((lane & 2) * 8);
                    probability_fragment[key_half * 2 + half] = (first & 0xffffu) | (second << 16);
                }
            }
#pragma unroll
            for (int dimension_pair = 0; dimension_pair < HEAD_DIM / 16; dimension_pair++) {
                const int row = dimension_pair * 16 + (matrix / 2) * 8 + matrix_row;
                const int chunk = key_step * 2 + matrix % 2;
                uint32_t registers[4];
                load_matrix_x4(registers, value_tile + value_offset(row, chunk));
                mma_pv(output_accumulators[dimension_pair * 2], probability_fragment, registers[0],
                       registers[1]);
                mma_pv(output_accumulators[dimension_pair * 2 + 1], probability_fragment,
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
        Element *output_row =
            output_head + static_cast<int64_t>(token) * layout.token_stride[OUTPUT];
        const float value_factor = value_scale[head] / 448.0f;
        if constexpr (SPARSE) {
            const float tail_max = workspace.tail_max[route_row];
            const float tail_sum = workspace.tail_sum[route_row];
            const float *tail_values = workspace.tail_values + route_row * HEAD_DIM;
            const float merged_max = fmaxf(row_max[half], tail_max);
            const float routed_weight = exp2f(row_max[half] - merged_max);
            const float tail_weight = exp2f(tail_max - merged_max);
            const float inverse_sum =
                1.0f / (row_sum[half] * routed_weight + tail_sum * tail_weight);
#pragma unroll
            for (int dimension_tile = 0; dimension_tile < HEAD_DIM / 8; dimension_tile++) {
                const int dimension = dimension_tile * 8 + (lane % 4) * 2;
                const float first =
                    (output_accumulators[dimension_tile][half * 2] * value_factor * routed_weight +
                     tail_values[dimension] * tail_weight) *
                    inverse_sum;
                const float second = (output_accumulators[dimension_tile][half * 2 + 1] *
                                          value_factor * routed_weight +
                                      tail_values[dimension + 1] * tail_weight) *
                                     inverse_sum;
                *reinterpret_cast<uint32_t *>(output_row + dimension) = N::pack(first, second);
            }
        } else {
            const float inverse_sum = value_factor / row_sum[half];
#pragma unroll
            for (int dimension_tile = 0; dimension_tile < HEAD_DIM / 8; dimension_tile++) {
                const int dimension = dimension_tile * 8 + (lane % 4) * 2;
                *reinterpret_cast<uint32_t *>(output_row + dimension) =
                    N::pack(output_accumulators[dimension_tile][half * 2] * inverse_sum,
                            output_accumulators[dimension_tile][half * 2 + 1] * inverse_sum);
            }
        }
    }
}

// The launch path makes no allocations or host downloads. All scratch belongs to the caller.
template <bool SPARSE>
int launch_attention(const Mmh3QuantizedWorkspace &workspace, __nv_bfloat16 *output, int tokens,
                     int heads, Mmh3AttentionLayout layout, float scale, Mmh3SparseWorkspace sparse,
                     cudaStream_t stream) {
    static bool configured = false;
    if (!configured) {
        cudaError_t status = cudaFuncSetAttribute(
            attention_kernel<SPARSE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES);
        if (status != cudaSuccess) {
            return static_cast<int>(status);
        }
        configured = true;
    }
    layout.token_stride[0] = layout.token_stride[1] = static_cast<int64_t>(heads) * HEAD_DIM;
    layout.head_stride[0] = layout.head_stride[1] = HEAD_DIM;
    layout.heads_per_key_value = 1;
    attention_kernel<SPARSE><<<dim3((tokens + 63) / 64, heads), THREADS, SHARED_BYTES, stream>>>(
        workspace.query, workspace.key, workspace.value, output, tokens, layout,
        scale * 1.4426950408889634f, workspace.query_scales, workspace.key_scales,
        workspace.value_scales, sparse);
    return static_cast<int>(cudaGetLastError());
}

} // namespace

extern "C" int mmh3_attention_quantized(const void *query, const void *key, const void *value,
                                        void *output, int tokens, int heads,
                                        const Mmh3AttentionLayout *layout, float scale,
                                        const Mmh3SparseWorkspace *sparse,
                                        const Mmh3QuantizedWorkspace *workspace, int inputs_ready,
                                        cudaStream_t stream) {
    if (tokens <= 0 || heads <= 0 || layout->causal || layout->heads_per_key_value > 1) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int blocks = (tokens + 63) / 64;
    const int padded = blocks * 64;
    const auto *q = static_cast<const __nv_bfloat16 *>(query);
    const auto *k = static_cast<const __nv_bfloat16 *>(key);
    const auto *v = static_cast<const __nv_bfloat16 *>(value);
    auto *result = static_cast<__nv_bfloat16 *>(output);
    if (!inputs_ready) {
        value_max<<<dim3(blocks, heads), 256, 0, stream>>>(v, tokens, *layout,
                                                           workspace->value_maxima);
    }
    value_scales<<<heads, 32, 0, stream>>>(workspace->value_maxima, workspace->value_scales,
                                           blocks);
    quantize_value<<<dim3(padded / 32, 4, heads), dim3(32, 8), 0, stream>>>(
        v, workspace->value, workspace->value_scales, tokens, padded, *layout);
    if (!inputs_ready) {
        quantize_qk<<<dim3(blocks, heads, 2), 256, 0, stream>>>(
            q, k, workspace->query, workspace->key, workspace->query_scales, workspace->key_scales,
            tokens, heads, *layout);
    }
    if (sparse != nullptr) {
        return launch_attention<true>(*workspace, result, tokens, heads, *layout, scale, *sparse,
                                      stream);
    }
    return launch_attention<false>(*workspace, result, tokens, heads, *layout, scale, {}, stream);
}
