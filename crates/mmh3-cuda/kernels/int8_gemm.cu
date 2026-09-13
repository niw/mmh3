#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Scaled INT8 GEMM for the DiT linear layers:
//   output[m, n] = bf16(sum_k activations[m, k] * weights[n, k] * activation_scales[m] * weight_scales[n])
// Activations and weights are row-major with K contiguous. N must be a multiple of the block width and
// K a multiple of 64, which every H3 linear layer satisfies. Only M may be ragged.

namespace {

constexpr int BLOCK_K = 64;
constexpr int CHUNKS_PER_ROW = BLOCK_K / 16;
constexpr int GROUP_M = 8;

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void copy_async_16(uint32_t destination, const void* source, bool valid) {
    int source_bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(destination), "l"(source), "r"(source_bytes));
}

__device__ __forceinline__ void copy_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int PENDING>
__device__ __forceinline__ void copy_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING));
}

__device__ __forceinline__ void load_matrix_x4(uint32_t (&registers)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]), "=r"(registers[3])
                 : "r"(address));
}

__device__ __forceinline__ void mma_s8(int32_t (&accumulator)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(accumulator[0]), "+r"(accumulator[1]), "+r"(accumulator[2]), "+r"(accumulator[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Rows are 64 bytes (four 16-byte chunks). XOR-swizzling the chunk by (row / 2) makes every ldmatrix
// phase touch eight distinct 16-byte bank groups.
__device__ __forceinline__ int swizzled_offset(int row, int chunk) {
    return row * BLOCK_K + ((chunk ^ ((row >> 1) & 3)) << 4);
}

template <int BLOCK_M, int BLOCK_N, int WARPS_M, int WARPS_N, int STAGES>
struct Int8GemmConfig {
    static constexpr int block_m = BLOCK_M;
    static constexpr int block_n = BLOCK_N;
    static constexpr int threads = WARPS_M * WARPS_N * 32;
    static constexpr int warp_m = BLOCK_M / WARPS_M;
    static constexpr int warp_n = BLOCK_N / WARPS_N;
    static constexpr int m_tiles = warp_m / 16;
    static constexpr int n_tiles = warp_n / 8;
    static constexpr int stages = STAGES;
    static constexpr int stage_bytes = (BLOCK_M + BLOCK_N) * BLOCK_K;
    static constexpr int shared_bytes = STAGES * stage_bytes;
    static constexpr int warps_n = WARPS_N;
    static_assert(warp_m % 16 == 0 && warp_n % 16 == 0, "warp tiles must hold whole MMA tiles");
};

template <typename Config>
__global__ void __launch_bounds__(Config::threads, 1)
    int8_gemm_kernel(const int8_t* __restrict__ activations, const int8_t* __restrict__ weights,
                     const float* __restrict__ activation_scales, const float* __restrict__ weight_scales,
                     __nv_bfloat16* __restrict__ output, int m, int n, int k) {
    extern __shared__ __align__(128) uint8_t shared_memory[];

    const int tiles_m = (m + Config::block_m - 1) / Config::block_m;
    const int tiles_n = n / Config::block_n;
    const int tiles_per_group = GROUP_M * tiles_n;
    const int group = blockIdx.x / tiles_per_group;
    const int first_tile_m = group * GROUP_M;
    const int group_height = min(tiles_m - first_tile_m, GROUP_M);
    const int tile_m = first_tile_m + (blockIdx.x % tiles_per_group) % group_height;
    const int tile_n = (blockIdx.x % tiles_per_group) / group_height;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warp_row = (warp / Config::warps_n) * Config::warp_m;
    const int warp_column = (warp % Config::warps_n) * Config::warp_n;

    const uint32_t shared_base = shared_address(shared_memory);
    const int8_t* activation_block = activations + static_cast<size_t>(tile_m) * Config::block_m * k;
    const int8_t* weight_block = weights + static_cast<size_t>(tile_n) * Config::block_n * k;
    const int valid_rows = min(Config::block_m, m - tile_m * Config::block_m);

    auto load_stage = [&](int stage, int k_tile) {
        const uint32_t stage_a = shared_base + stage * Config::stage_bytes;
        const uint32_t stage_b = stage_a + Config::block_m * BLOCK_K;
        const int k_offset = k_tile * BLOCK_K;
        for (int index = threadIdx.x; index < Config::block_m * CHUNKS_PER_ROW; index += Config::threads) {
            int row = index / CHUNKS_PER_ROW;
            int chunk = index % CHUNKS_PER_ROW;
            bool valid = row < valid_rows;
            const int8_t* source = activation_block + static_cast<size_t>(valid ? row : 0) * k + k_offset + chunk * 16;
            copy_async_16(stage_a + swizzled_offset(row, chunk), source, valid);
        }
        for (int index = threadIdx.x; index < Config::block_n * CHUNKS_PER_ROW; index += Config::threads) {
            int row = index / CHUNKS_PER_ROW;
            int chunk = index % CHUNKS_PER_ROW;
            const int8_t* source = weight_block + static_cast<size_t>(row) * k + k_offset + chunk * 16;
            copy_async_16(stage_b + swizzled_offset(row, chunk), source, true);
        }
    };

    int32_t accumulators[Config::m_tiles][Config::n_tiles][4] = {};
    const int k_tiles = k / BLOCK_K;

    for (int stage = 0; stage < Config::stages - 1; stage++) {
        if (stage < k_tiles) {
            load_stage(stage, stage);
        }
        copy_async_commit();
    }

    // Lane-dependent parts of the ldmatrix row addresses.
    const int matrix = lane / 8;
    const int matrix_row = lane % 8;

    for (int k_tile = 0; k_tile < k_tiles; k_tile++) {
        copy_async_wait<Config::stages - 2>();
        __syncthreads();

        const int next_tile = k_tile + Config::stages - 1;
        if (next_tile < k_tiles) {
            load_stage(next_tile % Config::stages, next_tile);
        }
        copy_async_commit();

        const uint32_t stage_a = shared_base + (k_tile % Config::stages) * Config::stage_bytes;
        const uint32_t stage_b = stage_a + Config::block_m * BLOCK_K;

#pragma unroll
        for (int k_step = 0; k_step < BLOCK_K / 32; k_step++) {
            uint32_t a_fragments[Config::m_tiles][4];
            uint32_t b_fragments[Config::n_tiles][2];
#pragma unroll
            for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
                int row = warp_row + m_tile * 16 + (matrix % 2) * 8 + matrix_row;
                int chunk = k_step * 2 + matrix / 2;
                load_matrix_x4(a_fragments[m_tile], stage_a + swizzled_offset(row, chunk));
            }
#pragma unroll
            for (int n_pair = 0; n_pair < Config::n_tiles / 2; n_pair++) {
                int row = warp_column + n_pair * 16 + (matrix / 2) * 8 + matrix_row;
                int chunk = k_step * 2 + matrix % 2;
                uint32_t registers[4];
                load_matrix_x4(registers, stage_b + swizzled_offset(row, chunk));
                b_fragments[n_pair * 2][0] = registers[0];
                b_fragments[n_pair * 2][1] = registers[1];
                b_fragments[n_pair * 2 + 1][0] = registers[2];
                b_fragments[n_pair * 2 + 1][1] = registers[3];
            }
#pragma unroll
            for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
#pragma unroll
                for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
                    mma_s8(accumulators[m_tile][n_tile], a_fragments[m_tile], b_fragments[n_tile][0],
                           b_fragments[n_tile][1]);
                }
            }
        }
    }
    copy_async_wait<0>();

    const int group_id = lane / 4;
    const int column_in_tile = (lane % 4) * 2;
#pragma unroll
    for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
        const int column = tile_n * Config::block_n + warp_column + n_tile * 8 + column_in_tile;
        const float weight_scale_0 = weight_scales[column];
        const float weight_scale_1 = weight_scales[column + 1];
#pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const int row = tile_m * Config::block_m + warp_row + m_tile * 16 + half * 8 + group_id;
                if (row < m) {
                    const float activation_scale = activation_scales[row];
                    __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(accumulators[m_tile][n_tile][half * 2]) * activation_scale * weight_scale_0,
                        static_cast<float>(accumulators[m_tile][n_tile][half * 2 + 1]) * activation_scale *
                            weight_scale_1);
                    *reinterpret_cast<__nv_bfloat162*>(output + static_cast<size_t>(row) * n + column) = value;
                }
            }
        }
    }
}

using LargeTile = Int8GemmConfig<128, 256, 2, 4, 3>;
using MediumTile = Int8GemmConfig<128, 128, 2, 2, 4>;

template <typename Config>
int launch(const int8_t* activations, const int8_t* weights, const float* activation_scales,
           const float* weight_scales, __nv_bfloat16* output, int m, int n, int k, cudaStream_t stream) {
    if (n % Config::block_n != 0 || k % BLOCK_K != 0 || m <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    static bool configured = false;
    if (!configured) {
        cudaError_t status = cudaFuncSetAttribute(int8_gemm_kernel<Config>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  Config::shared_bytes);
        if (status != cudaSuccess) {
            return static_cast<int>(status);
        }
        configured = true;
    }
    const int tiles = ((m + Config::block_m - 1) / Config::block_m) * (n / Config::block_n);
    int8_gemm_kernel<Config><<<tiles, Config::threads, Config::shared_bytes, stream>>>(
        activations, weights, activation_scales, weight_scales, output, m, n, k);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int mmh3_int8_gemm_config_count() {
    return 2;
}

extern "C" int mmh3_int8_gemm_bf16(int config, const int8_t* activations, const int8_t* weights,
                                   const float* activation_scales, const float* weight_scales, __nv_bfloat16* output,
                                   int m, int n, int k, cudaStream_t stream) {
    switch (config) {
        case 0:
            return launch<LargeTile>(activations, weights, activation_scales, weight_scales, output, m, n, k, stream);
        case 1:
            return launch<MediumTile>(activations, weights, activation_scales, weight_scales, output, m, n, k, stream);
        default:
            return static_cast<int>(cudaErrorInvalidValue);
    }
}
