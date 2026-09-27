#include <cstdint>
#include <cuda_runtime.h>

#include "device.cuh"
#include "tensor_core.cuh"
#include "tma.cuh"
#include "wgmma.cuh"

// One warpgroup multiplies a 64-row tile of A by an N-row tile of B over 128 bytes of K, both
// copied into shared memory in the TMA 128B swizzle, with the wgmma forms the Hopper kernels build
// on, and writes D row-major. It checks the descriptors, the swizzle and the accumulator layout on
// an H100 apart from any pipeline, the first thing to rule out when a Hopper kernel is wrong. A
// second check takes A from registers and B MN-major, as attention multiplies P by V, and a third
// FP8 A from registers and B in rows of 64 bytes, as the quantized attention does.

namespace {

constexpr int ROW_BYTES = 128;
constexpr int A_ROWS = 64;

template <int N, bool BF16>
__global__ void __launch_bounds__(128)
    wgmma_check_kernel(const uint8_t *__restrict__ a, const uint8_t *__restrict__ b,
                       uint32_t *__restrict__ d) {
    extern __shared__ __align__(1024) uint8_t shared_memory[];
    const uint32_t base = (shared_address(shared_memory) + 1023) & ~1023u;
    uint8_t *const aligned = shared_memory + (base - shared_address(shared_memory));
    constexpr int CHUNKS_PER_ROW = ROW_BYTES / 16;
    for (int index = threadIdx.x; index < (A_ROWS + N) * CHUNKS_PER_ROW; index += 128) {
        const int row = index / CHUNKS_PER_ROW;
        const int chunk = index % CHUNKS_PER_ROW;
        const bool in_a = row < A_ROWS;
        const uint8_t *source = in_a ? a + row * ROW_BYTES : b + (row - A_ROWS) * ROW_BYTES;
        const uint32_t offset = in_a ? swizzled_offset(row, chunk)
                                     : A_ROWS * ROW_BYTES + swizzled_offset(row - A_ROWS, chunk);
        *reinterpret_cast<uint4 *>(aligned + offset) =
            *reinterpret_cast<const uint4 *>(source + chunk * 16);
    }
    shared_to_async_proxy_fence();
    __syncthreads();

    uint32_t accumulators[N / 2];
    #pragma unroll
    for (int index = 0; index < N / 2; index++) {
        accumulators[index] = 0;
    }
    wgmma_fence();
    // Both forms read 32 bytes of K per instruction.
    #pragma unroll
    for (int k_step = 0; k_step < ROW_BYTES / 32; k_step++) {
        const uint64_t a_descriptor = wgmma_descriptor(base + k_step * 32);
        const uint64_t b_descriptor = wgmma_descriptor(base + A_ROWS * ROW_BYTES + k_step * 32);
        if constexpr (BF16 && N == 256) {
            wgmma_bf16_m64n256k16(accumulators, a_descriptor, b_descriptor, k_step);
        } else if constexpr (BF16) {
            wgmma_bf16_m64n128k16(accumulators, a_descriptor, b_descriptor, k_step);
        } else if constexpr (N == 256) {
            wgmma_s8_m64n256k32(accumulators, a_descriptor, b_descriptor, k_step);
        } else {
            wgmma_s8_m64n128k32(accumulators, a_descriptor, b_descriptor, k_step);
        }
    }
    wgmma_commit();
    wgmma_wait<0>();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    #pragma unroll
    for (int index = 0; index < N / 2; index++) {
        const int row = warp * 16 + lane / 4 + 8 * ((index / 2) % 2);
        const int column = 8 * (index / 4) + 2 * (lane % 4) + index % 2;
        d[row * N + column] = accumulators[index];
    }
}

// D [64, N] = A [64, 64] · V [64, N], A 16-bit from registers and V's rows of N values in 128-byte
// slabs, the way attention's value tiles arrive.
template <int N, bool F16>
__global__ void __launch_bounds__(128)
    wgmma_transposed_check_kernel(const uint16_t *__restrict__ a, const uint8_t *__restrict__ v,
                                  uint32_t *__restrict__ d) {
    extern __shared__ __align__(1024) uint8_t shared_memory[];
    const uint32_t base = (shared_address(shared_memory) + 1023) & ~1023u;
    uint8_t *const aligned = shared_memory + (base - shared_address(shared_memory));
    constexpr int KEYS = 64;
    constexpr int SLABS = N * 2 / ROW_BYTES;
    constexpr int SLAB_BYTES = KEYS * ROW_BYTES;
    constexpr int CHUNKS_PER_ROW = ROW_BYTES / 16;
    for (int index = threadIdx.x; index < KEYS * SLABS * CHUNKS_PER_ROW; index += 128) {
        const int key = index / (SLABS * CHUNKS_PER_ROW);
        const int slab = index / CHUNKS_PER_ROW % SLABS;
        const int chunk = index % CHUNKS_PER_ROW;
        *reinterpret_cast<uint4 *>(aligned + slab * SLAB_BYTES + swizzled_offset(key, chunk)) =
            *reinterpret_cast<const uint4 *>(v + key * N * 2 + slab * ROW_BYTES + chunk * 16);
    }
    shared_to_async_proxy_fence();
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    uint32_t accumulators[N / 2];
    #pragma unroll
    for (int index = 0; index < N / 2; index++) {
        accumulators[index] = 0;
    }
    wgmma_fence();
    #pragma unroll
    for (int k_step = 0; k_step < KEYS / 16; k_step++) {
        // The mma.sync m16n8k16 A layout of the warp's 16 rows.
        uint32_t fragment[4];
        #pragma unroll
        for (int index = 0; index < 4; index++) {
            const int row = warp * 16 + lane / 4 + 8 * (index % 2);
            const int column = k_step * 16 + 2 * (lane % 4) + 8 * (index / 2);
            fragment[index] = *reinterpret_cast<const uint32_t *>(a + row * KEYS + column);
        }
        const uint64_t b = wgmma_descriptor_mn_major(base + k_step * 16 * ROW_BYTES, SLAB_BYTES);
        if constexpr (F16 && N == 128) {
            wgmma_f16_m64n128k16_registers_transposed(accumulators, fragment, b, k_step);
        } else if constexpr (F16) {
            wgmma_f16_m64n64k16_registers_transposed(accumulators, fragment, b, k_step);
        } else if constexpr (N == 128) {
            wgmma_bf16_m64n128k16_registers_transposed(accumulators, fragment, b, k_step);
        } else {
            wgmma_bf16_m64n64k16_registers_transposed(accumulators, fragment, b, k_step);
        }
    }
    wgmma_commit();
    wgmma_wait<0>();

    #pragma unroll
    for (int index = 0; index < N / 2; index++) {
        const int row = warp * 16 + lane / 4 + 8 * ((index / 2) % 2);
        const int column = 8 * (index / 4) + 2 * (lane % 4) + index % 2;
        d[row * N + column] = accumulators[index];
    }
}

template <int N, bool F16>
int launch_transposed(const void *a, const void *v, void *d, cudaStream_t stream) {
    auto kernel = wgmma_transposed_check_kernel<N, F16>;
    constexpr int shared_bytes = 64 * N * 2 + 1024;
    static Mmh3SharedMemory shared_memory;
    const cudaError_t configured =
        mmh3_configure_shared_memory(shared_memory, kernel, shared_bytes);
    if (configured != cudaSuccess) {
        return static_cast<int>(configured);
    }
    kernel<<<1, 128, shared_bytes, stream>>>(static_cast<const uint16_t *>(a),
                                             static_cast<const uint8_t *>(v),
                                             static_cast<uint32_t *>(d));
    return static_cast<int>(cudaGetLastError());
}

// D [64, 128] = A [64, 64] · B [128, 64]ᵀ in FP8 E4M3 into FP32, A from registers and B's rows of
// 64 bytes in the TMA 64B swizzle.
__global__ void __launch_bounds__(128)
    wgmma_fp8_check_kernel(const uint8_t *__restrict__ a, const uint8_t *__restrict__ b,
                           uint32_t *__restrict__ d) {
    extern __shared__ __align__(1024) uint8_t shared_memory[];
    const uint32_t base = (shared_address(shared_memory) + 1023) & ~1023u;
    uint8_t *const aligned = shared_memory + (base - shared_address(shared_memory));
    constexpr int K = 64;
    constexpr int N = 128;
    for (int index = threadIdx.x; index < N * 4; index += 128) {
        const int row = index / 4;
        const int chunk = index % 4;
        *reinterpret_cast<uint4 *>(aligned + row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4)) =
            *reinterpret_cast<const uint4 *>(b + row * K + chunk * 16);
    }
    shared_to_async_proxy_fence();
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    uint32_t accumulators[N / 2];
    #pragma unroll
    for (int index = 0; index < N / 2; index++) {
        accumulators[index] = 0;
    }
    wgmma_fence();
    #pragma unroll
    for (int k_step = 0; k_step < K / 32; k_step++) {
        // The mma.sync m16n8k32 A layout of the warp's 16 rows.
        uint32_t fragment[4];
        #pragma unroll
        for (int index = 0; index < 4; index++) {
            const int row = warp * 16 + lane / 4 + 8 * (index % 2);
            const int column = k_step * 32 + 4 * (lane % 4) + 16 * (index / 2);
            fragment[index] = *reinterpret_cast<const uint32_t *>(a + row * K + column);
        }
        wgmma_e4m3_m64n128k32_registers(accumulators, fragment,
                                        wgmma_descriptor_64b(base) + k_step * 2, k_step);
    }
    wgmma_commit();
    wgmma_wait<0>();

    #pragma unroll
    for (int index = 0; index < N / 2; index++) {
        const int row = warp * 16 + lane / 4 + 8 * ((index / 2) % 2);
        const int column = 8 * (index / 4) + 2 * (lane % 4) + index % 2;
        d[row * N + column] = accumulators[index];
    }
}

template <int N, bool BF16> int launch(const void *a, const void *b, void *d, cudaStream_t stream) {
    auto kernel = wgmma_check_kernel<N, BF16>;
    constexpr int shared_bytes = (A_ROWS + N) * ROW_BYTES + 1024;
    static Mmh3SharedMemory shared_memory;
    const cudaError_t configured =
        mmh3_configure_shared_memory(shared_memory, kernel, shared_bytes);
    if (configured != cudaSuccess) {
        return static_cast<int>(configured);
    }
    kernel<<<1, 128, shared_bytes, stream>>>(static_cast<const uint8_t *>(a),
                                             static_cast<const uint8_t *>(b),
                                             static_cast<uint32_t *>(d));
    return static_cast<int>(cudaGetLastError());
}

} // namespace

// D [64, n] = A [64, K] · B [n, K]ᵀ for n 128 or 256, with K 128 INT8 values into INT32 or, with
// `bf16`, 64 BF16 values into FP32, through one warpgroup's wgmma. Answers cudaErrorNotSupported
// on a device without wgmma.
extern "C" int mmh3_wgmma_check(int bf16, int n, const void *a, const void *b, void *d,
                                cudaStream_t stream) {
    if (!mmh3_wgmma_available()) {
        return static_cast<int>(cudaErrorNotSupported);
    }
    if (n == 128) {
        return bf16 ? launch<128, true>(a, b, d, stream) : launch<128, false>(a, b, d, stream);
    }
    if (n == 256) {
        return bf16 ? launch<256, true>(a, b, d, stream) : launch<256, false>(a, b, d, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

// D [64, n] = A [64, 64] · V [64, n] in FP32 for n 64 or 128, A and V row-major BF16 or, with
// `f16`, FP16, through one warpgroup's wgmma with A in registers and V MN-major. Answers
// cudaErrorNotSupported on a device without wgmma.
extern "C" int mmh3_wgmma_check_transposed(int f16, int n, const void *a, const void *v, void *d,
                                           cudaStream_t stream) {
    if (!mmh3_wgmma_available()) {
        return static_cast<int>(cudaErrorNotSupported);
    }
    if (n == 64) {
        return f16 ? launch_transposed<64, true>(a, v, d, stream)
                   : launch_transposed<64, false>(a, v, d, stream);
    }
    if (n == 128) {
        return f16 ? launch_transposed<128, true>(a, v, d, stream)
                   : launch_transposed<128, false>(a, v, d, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

// D [64, 128] = A [64, 64] · B [128, 64]ᵀ in FP32 for row-major FP8 E4M3 A and B, through one
// warpgroup's wgmma with A in registers and B in rows of 64 bytes. Answers cudaErrorNotSupported on
// a device without wgmma.
extern "C" int mmh3_wgmma_check_fp8(const void *a, const void *b, void *d, cudaStream_t stream) {
    if (!mmh3_wgmma_available()) {
        return static_cast<int>(cudaErrorNotSupported);
    }
    constexpr int shared_bytes = 128 * 64 + 1024;
    static Mmh3SharedMemory shared_memory;
    const cudaError_t configured =
        mmh3_configure_shared_memory(shared_memory, wgmma_fp8_check_kernel, shared_bytes);
    if (configured != cudaSuccess) {
        return static_cast<int>(configured);
    }
    wgmma_fp8_check_kernel<<<1, 128, shared_bytes, stream>>>(static_cast<const uint8_t *>(a),
                                                             static_cast<const uint8_t *>(b),
                                                             static_cast<uint32_t *>(d));
    return static_cast<int>(cudaGetLastError());
}
