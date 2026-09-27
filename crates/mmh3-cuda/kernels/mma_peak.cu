#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "device.cuh"
#include "tensor_core.cuh"
#include "wgmma.cuh"

// Loops that measure tensor core throughput per instruction form: register-only mma.sync, and on
// Hopper wgmma over operands that stay in shared memory.

namespace {

enum MmaKind : int {
    MMA_KIND_BF16_F32 = 0,
    MMA_KIND_F16_F16 = 1,
    MMA_KIND_S8_S32 = 2,
    MMA_KIND_E4M3_F32 = 3,
    MMA_KIND_E4M3_F16 = 4,
    MMA_KIND_WGMMA_S8_S32 = 5,
    MMA_KIND_WGMMA_BF16_F32 = 6,
};

constexpr int CHAINS = 8;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCKS_PER_MULTIPROCESSOR = 4;

template <int KIND> struct MmaForm;

template <> struct MmaForm<MMA_KIND_BF16_F32> {
    static constexpr int operations = 2 * 16 * 8 * 16;
    using Accumulator = float[4];
    __device__ static void run(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                     "%5, %6, %7}, {%8, %9}, "
                     "{%0, %1, %2, %3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

template <> struct MmaForm<MMA_KIND_F16_F16> {
    static constexpr int operations = 2 * 16 * 8 * 16;
    using Accumulator = uint32_t[2];
    __device__ static void run(uint32_t (&c)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, "
                     "%5}, {%6, %7}, {%0, %1};\n"
                     : "+r"(c[0]), "+r"(c[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

template <> struct MmaForm<MMA_KIND_S8_S32> {
    static constexpr int operations = 2 * 16 * 8 * 32;
    using Accumulator = int32_t[4];
    __device__ static void run(int32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, "
                     "%6, %7}, {%8, %9}, "
                     "{%0, %1, %2, %3};\n"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

template <> struct MmaForm<MMA_KIND_E4M3_F32> {
    static constexpr int operations = 2 * 16 * 8 * 32;
    using Accumulator = float[4];
    __device__ static void run(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, "
                     "%5, %6, %7}, {%8, %9}, "
                     "{%0, %1, %2, %3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

template <> struct MmaForm<MMA_KIND_E4M3_F16> {
    static constexpr int operations = 2 * 16 * 8 * 32;
    using Accumulator = uint32_t[2];
    __device__ static void run(uint32_t (&c)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, "
                     "%5}, {%6, %7}, {%0, %1};\n"
                     : "+r"(c[0]), "+r"(c[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

template <int KIND>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    mma_peak_kernel(int iterations, uint32_t *sink) {
    using Form = MmaForm<KIND>;
    // Small positive operands in every format keep the loop free of special values.
    const uint32_t operand = KIND == MMA_KIND_S8_S32
                                 ? 0x01010101u
                                 : (KIND >= MMA_KIND_E4M3_F32 ? 0x20202020u : 0x3C003C00u);
    const uint32_t a[4] = {operand, operand, operand, operand};
    const uint32_t b[2] = {operand, operand};
    typename Form::Accumulator accumulators[CHAINS] = {};
    for (int iteration = 0; iteration < iterations; iteration++) {
        #pragma unroll
        for (int chain = 0; chain < CHAINS; chain++) {
            Form::run(accumulators[chain], a, b);
        }
    }
    uint32_t checksum = 0;
    for (int chain = 0; chain < CHAINS; chain++) {
        for (int element = 0; element < static_cast<int>(sizeof(accumulators[chain]) / 4);
             element++) {
            checksum ^= reinterpret_cast<const uint32_t *>(&accumulators[chain])[element];
        }
    }
    if (checksum == 0x9E3779B9u) {
        sink[threadIdx.x] = checksum;
    }
}

// Two warpgroups per SM, each with one m64n256 accumulator, which is all the registers allow.
constexpr int WGMMA_THREADS = 256;
// A 64-row tile of A and a 256-row tile of B, 128 bytes per row, and the alignment slack.
constexpr int WGMMA_SHARED_BYTES = (64 + 256) * 128 + 1024;
// One iteration is four wgmma over 128 bytes of K, a few hundred times the work of an iteration of
// the mma.sync loops, so the wgmma loops run this many times fewer.
constexpr int WGMMA_ITERATION_SHARE = 64;

template <bool BF16>
__global__ void __launch_bounds__(WGMMA_THREADS) wgmma_peak_kernel(int iterations, uint32_t *sink) {
    extern __shared__ __align__(1024) uint8_t shared_memory[];
    const uint32_t base = (shared_address(shared_memory) + 1023) & ~1023u;
    // Zeros keep the products free of special values.
    for (int index = threadIdx.x; index < WGMMA_SHARED_BYTES / 16; index += WGMMA_THREADS) {
        reinterpret_cast<uint4 *>(shared_memory)[index] = make_uint4(0, 0, 0, 0);
    }
    shared_to_async_proxy_fence();
    __syncthreads();
    uint32_t accumulators[128];
    #pragma unroll
    for (int index = 0; index < 128; index++) {
        accumulators[index] = 0;
    }
    const uint64_t a = wgmma_descriptor(base);
    const uint64_t b = wgmma_descriptor(base + 64 * 128);
    wgmma_fence();
    for (int iteration = 0; iteration < iterations; iteration++) {
        #pragma unroll
        for (int k_step = 0; k_step < 4; k_step++) {
            // A descriptor counts 16-byte units, and each step reads 32 bytes of K.
            if constexpr (BF16) {
                wgmma_bf16_m64n256k16(accumulators, a + k_step * 2, b + k_step * 2, 1);
            } else {
                wgmma_s8_m64n256k32(accumulators, a + k_step * 2, b + k_step * 2, 1);
            }
        }
        wgmma_commit();
        wgmma_wait<1>();
    }
    wgmma_wait<0>();
    uint32_t checksum = 0;
    #pragma unroll
    for (int index = 0; index < 128; index++) {
        checksum ^= accumulators[index];
    }
    if (checksum == 0x9E3779B9u) {
        sink[threadIdx.x] = checksum;
    }
}

template <bool BF16> cudaError_t launch_wgmma_peak(int blocks, int iterations, uint32_t *sink) {
    auto kernel = wgmma_peak_kernel<BF16>;
    static Mmh3SharedMemory shared_memory;
    const cudaError_t configured =
        mmh3_configure_shared_memory(shared_memory, kernel, WGMMA_SHARED_BYTES);
    if (configured != cudaSuccess) {
        return configured;
    }
    kernel<<<blocks, WGMMA_THREADS, WGMMA_SHARED_BYTES>>>(iterations, sink);
    return cudaGetLastError();
}

template <int KIND> cudaError_t launch_peak(int blocks, int iterations, uint32_t *sink) {
    mma_peak_kernel<KIND><<<blocks, THREADS_PER_BLOCK>>>(iterations, sink);
    return cudaGetLastError();
}

// Runs `iterations` of the form and returns the operations the launch does in `operations`.
cudaError_t launch_kind(int kind, int multiprocessors, int iterations, uint32_t *sink,
                        double *operations) {
    const int blocks = multiprocessors * BLOCKS_PER_MULTIPROCESSOR;
    const double warps = static_cast<double>(blocks) * THREADS_PER_BLOCK / 32;
    auto mma_sync = [&](auto form, cudaError_t (*launch)(int, int, uint32_t *)) {
        *operations = warps * iterations * CHAINS * decltype(form)::operations;
        return launch(blocks, iterations, sink);
    };
    switch (kind) {
    case MMA_KIND_BF16_F32:
        return mma_sync(MmaForm<MMA_KIND_BF16_F32>{}, launch_peak<MMA_KIND_BF16_F32>);
    case MMA_KIND_F16_F16:
        return mma_sync(MmaForm<MMA_KIND_F16_F16>{}, launch_peak<MMA_KIND_F16_F16>);
    case MMA_KIND_S8_S32:
        return mma_sync(MmaForm<MMA_KIND_S8_S32>{}, launch_peak<MMA_KIND_S8_S32>);
    case MMA_KIND_E4M3_F32:
        return mma_sync(MmaForm<MMA_KIND_E4M3_F32>{}, launch_peak<MMA_KIND_E4M3_F32>);
    case MMA_KIND_E4M3_F16:
        return mma_sync(MmaForm<MMA_KIND_E4M3_F16>{}, launch_peak<MMA_KIND_E4M3_F16>);
    case MMA_KIND_WGMMA_S8_S32:
    case MMA_KIND_WGMMA_BF16_F32: {
        if (!mmh3_wgmma_available()) {
            return cudaErrorNotSupported;
        }
        const bool bf16 = kind == MMA_KIND_WGMMA_BF16_F32;
        const int wgmma_iterations = iterations / WGMMA_ITERATION_SHARE + 1;
        const double warpgroups = static_cast<double>(multiprocessors) * WGMMA_THREADS / 128;
        // Four wgmma of 64 × 256 over 16 BF16 or 32 INT8 values of K per iteration.
        *operations = warpgroups * wgmma_iterations * 4 * 2.0 * 64 * 256 * (bf16 ? 16 : 32);
        return bf16 ? launch_wgmma_peak<true>(multiprocessors, wgmma_iterations, sink)
                    : launch_wgmma_peak<false>(multiprocessors, wgmma_iterations, sink);
    }
    default:
        return cudaErrorInvalidValue;
    }
}

} // namespace

extern "C" int mmh3_bench_mma_peak(int kind, int iterations, float *tera_operations_per_second,
                                   char *message, size_t message_size) {
    // Multiprocessors of the card this thread measures, which on a machine of two kinds need not
    // be the first's.
    const int multiprocessors = mmh3_multiprocessor_count();
    uint32_t *sink = nullptr;
    cudaError_t status = cudaMalloc(&sink, THREADS_PER_BLOCK * sizeof(uint32_t));
    double operations = 0.0;
    cudaEvent_t start = nullptr, stop = nullptr;
    if (status == cudaSuccess) {
        status = launch_kind(kind, multiprocessors, iterations / 8 + 1, sink, &operations);
    }
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    float elapsed = 0.0f;
    if (status == cudaSuccess) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        status = launch_kind(kind, multiprocessors, iterations, sink, &operations);
        cudaEventRecord(stop);
        if (status == cudaSuccess) {
            status = cudaEventSynchronize(stop);
        }
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    cudaFree(sink);
    if (status != cudaSuccess) {
        if (message != nullptr && message_size > 0) {
            std::snprintf(message, message_size, "mma peak: %s", cudaGetErrorString(status));
        }
        return static_cast<int>(status);
    }
    *tera_operations_per_second = static_cast<float>(operations / (elapsed * 1e-3) / 1e12);
    return 0;
}
