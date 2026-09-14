#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// Register-only mma.sync loops that measure tensor core throughput per instruction form.

namespace {

enum MmaKind : int {
    MMA_KIND_BF16_F32 = 0,
    MMA_KIND_F16_F16 = 1,
    MMA_KIND_S8_S32 = 2,
    MMA_KIND_E4M3_F32 = 3,
    MMA_KIND_E4M3_F16 = 4,
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

template <int KIND> cudaError_t launch_peak(int blocks, int iterations, uint32_t *sink) {
    mma_peak_kernel<KIND><<<blocks, THREADS_PER_BLOCK>>>(iterations, sink);
    return cudaGetLastError();
}

cudaError_t launch_kind(int kind, int blocks, int iterations, uint32_t *sink,
                        int *operations_per_mma) {
    switch (kind) {
    case MMA_KIND_BF16_F32:
        *operations_per_mma = MmaForm<MMA_KIND_BF16_F32>::operations;
        return launch_peak<MMA_KIND_BF16_F32>(blocks, iterations, sink);
    case MMA_KIND_F16_F16:
        *operations_per_mma = MmaForm<MMA_KIND_F16_F16>::operations;
        return launch_peak<MMA_KIND_F16_F16>(blocks, iterations, sink);
    case MMA_KIND_S8_S32:
        *operations_per_mma = MmaForm<MMA_KIND_S8_S32>::operations;
        return launch_peak<MMA_KIND_S8_S32>(blocks, iterations, sink);
    case MMA_KIND_E4M3_F32:
        *operations_per_mma = MmaForm<MMA_KIND_E4M3_F32>::operations;
        return launch_peak<MMA_KIND_E4M3_F32>(blocks, iterations, sink);
    case MMA_KIND_E4M3_F16:
        *operations_per_mma = MmaForm<MMA_KIND_E4M3_F16>::operations;
        return launch_peak<MMA_KIND_E4M3_F16>(blocks, iterations, sink);
    default:
        return cudaErrorInvalidValue;
    }
}

} // namespace

extern "C" int mmh3_bench_mma_peak(int kind, int iterations, float *tera_operations_per_second,
                                   char *message, size_t message_size) {
    int multiprocessors = 0;
    cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = multiprocessors * BLOCKS_PER_MULTIPROCESSOR;
    uint32_t *sink = nullptr;
    cudaError_t status = cudaMalloc(&sink, THREADS_PER_BLOCK * sizeof(uint32_t));
    int operations_per_mma = 0;
    cudaEvent_t start = nullptr, stop = nullptr;
    if (status == cudaSuccess) {
        status = launch_kind(kind, blocks, iterations / 8 + 1, sink, &operations_per_mma);
    }
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    float elapsed = 0.0f;
    if (status == cudaSuccess) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        status = launch_kind(kind, blocks, iterations, sink, &operations_per_mma);
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
    const double warps = static_cast<double>(blocks) * THREADS_PER_BLOCK / 32;
    const double operations = warps * iterations * CHAINS * operations_per_mma;
    *tera_operations_per_second = static_cast<float>(operations / (elapsed * 1e-3) / 1e12);
    return 0;
}
