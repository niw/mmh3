#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Scaled INT8 GEMM for the DiT and text encoder linear layers:
//   output[m, n] = bf16(sum_k activations[m, k] * weights[n, k] * activation_scales[m] * weight_scales[n])
// Activations and weights are row-major with K contiguous. N must be a multiple of the block width and K a multiple
// of 128, which every H3 linear layer satisfies. Only M may be ragged.
//
// NOTE: A persistent grid of one CTA per SM walks the output tiles. TMA copies 128-byte K slices of both operands
// into a two-stage ring with the 128B swizzle, and eight warps multiply their parts of the tile with mma.sync. There
// is no producer warp because a ninth warp would limit every warp to 168 registers. Instead, the warp that releases
// a stage last issues the copies that refill it.

namespace {

constexpr int BLOCK_K = 128;
constexpr int STAGES = 2;
constexpr int GROUP_M = 8;
constexpr int WARPS = 8;

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void barrier_init(uint32_t barrier, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier), "r"(count));
}

__device__ __forceinline__ void barrier_expect_bytes(uint32_t barrier, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier), "r"(bytes) : "memory");
}

__device__ __forceinline__ void barrier_wait(uint32_t barrier, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra WAIT;\n"
        "}\n" ::"r"(barrier),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ void copy_tile(uint32_t destination, const CUtensorMap* map, uint32_t barrier, int column,
                                          int row) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::
            "r"(destination),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(column), "r"(row), "r"(barrier)
        : "memory");
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

// The TMA 128B swizzle stores 16-byte chunk c of a 128-byte row r at chunk c ^ (r % 8), so every ldmatrix phase
// touches eight distinct bank groups.
__device__ __forceinline__ uint32_t swizzled_offset(int row, int chunk) {
    return row * BLOCK_K + ((chunk ^ (row & 7)) << 4);
}

template <int BLOCK_M, int BLOCK_N, int WARPS_M>
struct Int8GemmConfig {
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
    static_assert(warp_m % 16 == 0 && warp_n % 32 == 0, "warp tiles must hold whole groups of four MMA tiles");
};

struct TileGrid {
    int tiles_m;
    int tiles_n;
    int tiles;
    int k_blocks;

    // Tiles go down GROUP_M rows of tiles before moving right, so the CTAs in flight share rows of activations and
    // columns of weights in L2.
    __device__ void coordinates(int tile, int& tile_m, int& tile_n) const {
        const int tiles_per_group = GROUP_M * tiles_n;
        const int first_tile_m = tile / tiles_per_group * GROUP_M;
        const int group_height = min(tiles_m - first_tile_m, GROUP_M);
        tile_m = first_tile_m + tile % tiles_per_group % group_height;
        tile_n = tile % tiles_per_group / group_height;
    }
};

// Fills a stage with the K block of this CTA's `sequence`-th K block over all of its tiles.
template <typename Config>
__device__ void fill_stage(const TileGrid& grid, int sequence, uint32_t stage, uint32_t barrier,
                           const CUtensorMap* activation_map, const CUtensorMap* weight_map) {
    const int tile = blockIdx.x + sequence / grid.k_blocks * gridDim.x;
    if (tile >= grid.tiles) {
        return;
    }
    const int k_offset = sequence % grid.k_blocks * BLOCK_K;
    int tile_m, tile_n;
    grid.coordinates(tile, tile_m, tile_n);
    barrier_expect_bytes(barrier, Config::stage_bytes);
    copy_tile(stage, activation_map, barrier, k_offset, tile_m * Config::block_m);
    copy_tile(stage + Config::a_bytes, weight_map, barrier, k_offset, tile_n * Config::block_n);
}

template <typename Config>
__global__ void __launch_bounds__(WARPS * 32, 1)
    int8_gemm_kernel(const __grid_constant__ CUtensorMap activation_map, const __grid_constant__ CUtensorMap weight_map,
                     const float* __restrict__ activation_scales, const float* __restrict__ weight_scales,
                     __nv_bfloat16* __restrict__ output, int m, int n, int k) {
    extern __shared__ __align__(1024) uint8_t shared_memory[];
    const uint32_t shared_base = (shared_address(shared_memory) + 1023) & ~1023u;
    const uint32_t barriers = shared_base + STAGES * Config::stage_bytes;
    int* release_counts = reinterpret_cast<int*>(shared_memory + (barriers - shared_address(shared_memory)) + STAGES * 8);

    TileGrid grid;
    grid.tiles_m = (m + Config::block_m - 1) / Config::block_m;
    grid.tiles_n = n / Config::block_n;
    grid.tiles = grid.tiles_m * grid.tiles_n;
    grid.k_blocks = k / BLOCK_K;

    if (threadIdx.x == 0) {
        for (int stage = 0; stage < STAGES; stage++) {
            barrier_init(barriers + stage * 8, 1);
            release_counts[stage] = 0;
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        for (int stage = 0; stage < STAGES; stage++) {
            fill_stage<Config>(grid, stage, shared_base + stage * Config::stage_bytes, barriers + stage * 8,
                               &activation_map, &weight_map);
        }
    }
    __syncthreads();

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warp_row = warp / Config::warps_n * Config::warp_m;
    const int warp_column = warp % Config::warps_n * Config::warp_n;
    // Lane-dependent parts of the ldmatrix row addresses.
    const int matrix = lane / 8;
    const int matrix_row = lane % 8;
    // Position of the lane in the MMA accumulator layout.
    const int group_id = lane / 4;
    const int quad_lane = lane % 4;

    int stage = 0;
    uint32_t phase = 0;
    int sequence = 0;
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

        for (int k_block = 0; k_block < grid.k_blocks; k_block++, sequence++) {
            const uint32_t barrier = barriers + stage * 8;
            barrier_wait(barrier, phase);
            const uint32_t stage_a = shared_base + stage * Config::stage_bytes;
            const uint32_t stage_b = stage_a + Config::a_bytes;
#pragma unroll
            for (int k_step = 0; k_step < BLOCK_K / 32; k_step++) {
                uint32_t a_fragments[Config::m_tiles][4];
                uint32_t b_fragments[Config::n_tiles][2];
#pragma unroll
                for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
                    const int row = warp_row + m_tile * 16 + (matrix % 2) * 8 + matrix_row;
                    load_matrix_x4(a_fragments[m_tile], stage_a + swizzled_offset(row, k_step * 2 + matrix / 2));
                }
#pragma unroll
                for (int n_pair = 0; n_pair < Config::n_tiles / 2; n_pair++) {
                    // NOTE: Within each group of 32 weight rows, MMA column 2t + e of n-tile j reads row
                    // 8t + 2((j + t) % 4) + e. Lane t of every quad then holds eight adjacent output columns for a
                    // 16-byte store, and the eight rows of each ldmatrix phase still differ modulo 8.
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
                        mma_s8(accumulators[m_tile][n_tile], a_fragments[m_tile], b_fragments[n_tile][0],
                               b_fragments[n_tile][1]);
                    }
                }
            }
            __syncwarp();
            if (lane == 0) {
                __threadfence_block();
                if (atomicAdd(release_counts + stage, 1) == WARPS - 1) {
                    release_counts[stage] = 0;
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    fill_stage<Config>(grid, sequence + STAGES, stage_a, barrier, &activation_map, &weight_map);
                }
            }
            if (++stage == STAGES) {
                stage = 0;
                phase ^= 1;
            }
        }

        float row_scales[Config::m_tiles][2];
#pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const int row = tile_m * Config::block_m + warp_row + m_tile * 16 + half * 8 + group_id;
                row_scales[m_tile][half] = row < m ? activation_scales[row] : 0.0f;
            }
        }
#pragma unroll
        for (int quad = 0; quad < Config::n_tiles / 4; quad++) {
            const int column = tile_n * Config::block_n + warp_column + quad * 32 + quad_lane * 8;
            // Slot j of the quad holds output columns column + 2((j + t) % 4) and the one after.
            float2 column_scales[4];
#pragma unroll
            for (int slot = 0; slot < 4; slot++) {
                column_scales[slot] =
                    *reinterpret_cast<const float2*>(weight_scales + column + ((slot + quad_lane) & 3) * 2);
            }
#pragma unroll
            for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    uint32_t words[4];
#pragma unroll
                    for (int slot = 0; slot < 4; slot++) {
                        const int32_t* values = accumulators[m_tile][quad * 4 + slot] + half * 2;
                        const float row_scale = row_scales[m_tile][half];
                        __nv_bfloat162 value =
                            __floats2bfloat162_rn(static_cast<float>(values[0]) * row_scale * column_scales[slot].x,
                                                  static_cast<float>(values[1]) * row_scale * column_scales[slot].y);
                        words[slot] = *reinterpret_cast<uint32_t*>(&value);
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
                    const int row = tile_m * Config::block_m + warp_row + m_tile * 16 + half * 8 + group_id;
                    if (row < m) {
                        *reinterpret_cast<uint4*>(output + static_cast<size_t>(row) * n + column) =
                            make_uint4(words[0], words[1], words[2], words[3]);
                    }
                }
            }
        }
    }
}

using TallTile = Int8GemmConfig<256, 128, 4>;
using WideTile = Int8GemmConfig<128, 256, 4>;

PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
    static PFN_cuTensorMapEncodeTiled_v12000 encoder = nullptr;
    if (encoder == nullptr) {
        void* function = nullptr;
        cudaDriverEntryPointQueryResult result;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &function, 12000, cudaEnableDefault, &result) ==
                cudaSuccess &&
            result == cudaDriverEntryPointSuccess) {
            encoder = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function);
        }
    }
    return encoder;
}

// Describes `rows` rows of `k` bytes to TMA in boxes of `box_rows` rows by one K block, zero-filling rows past the end.
bool encode_tensor_map(CUtensorMap* map, const int8_t* base, int rows, int k, int box_rows) {
    PFN_cuTensorMapEncodeTiled_v12000 encoder = tensor_map_encoder();
    if (encoder == nullptr) {
        return false;
    }
    const cuuint64_t dimensions[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(rows)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(k)};
    const cuuint32_t box[2] = {BLOCK_K, static_cast<cuuint32_t>(box_rows)};
    const cuuint32_t element_strides[2] = {1, 1};
    return encoder(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<int8_t*>(base), dimensions, strides, box,
                   element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

int multiprocessor_count() {
    static int count = 0;
    if (count == 0) {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            count = 0;
        }
    }
    return count;
}

template <typename Config>
int launch(const int8_t* activations, const int8_t* weights, const float* activation_scales,
           const float* weight_scales, __nv_bfloat16* output, int m, int n, int k, cudaStream_t stream) {
    if (n % Config::block_n != 0 || k % BLOCK_K != 0 || m <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    CUtensorMap activation_map;
    CUtensorMap weight_map;
    if (!encode_tensor_map(&activation_map, activations, m, k, Config::block_m) ||
        !encode_tensor_map(&weight_map, weights, n, k, Config::block_n)) {
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
    const int tiles = (m + Config::block_m - 1) / Config::block_m * (n / Config::block_n);
    const int processors = multiprocessor_count();
    if (processors == 0) {
        return static_cast<int>(cudaErrorInvalidDevice);
    }
    int8_gemm_kernel<Config><<<tiles < processors ? tiles : processors, WARPS * 32, Config::shared_bytes, stream>>>(
        activation_map, weight_map, activation_scales, weight_scales, output, m, n, k);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int mmh3_int8_gemm_config_count() {
    return 2;
}

// Config 0 uses 256 × 128 tiles and needs N to be a multiple of 128. Config 1 uses 128 × 256 tiles, which waste
// less work on short inputs, and needs N to be a multiple of 256.
extern "C" int mmh3_int8_gemm_bf16(int config, const int8_t* activations, const int8_t* weights,
                                   const float* activation_scales, const float* weight_scales, __nv_bfloat16* output,
                                   int m, int n, int k, cudaStream_t stream) {
    switch (config) {
        case 0:
            return launch<TallTile>(activations, weights, activation_scales, weight_scales, output, m, n, k, stream);
        case 1:
            return launch<WideTile>(activations, weights, activation_scales, weight_scales, output, m, n, k, stream);
        default:
            return static_cast<int>(cudaErrorInvalidValue);
    }
}
