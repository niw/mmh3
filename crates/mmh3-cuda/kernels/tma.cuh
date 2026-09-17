#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

// Tensor Memory Accelerator copies: one thread issues a whole tile against a tensor map that the
// host encoded, and an mbarrier counts the bytes it delivers. Addresses, bounds and the swizzle are
// the engine's work, so the issuing warp keeps its registers and issue slots.

namespace {

__device__ __forceinline__ void barrier_init(uint32_t barrier, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier), "r"(count));
}

__device__ __forceinline__ void barrier_expect_bytes(uint32_t barrier, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier),
                 "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ bool barrier_try_wait(uint32_t barrier, uint32_t parity) {
    uint32_t done;
    asm volatile("{\n"
                 ".reg .pred arrived;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 arrived, [%1], %2;\n"
                 "selp.b32 %0, 1, 0, arrived;\n"
                 "}\n"
                 : "=r"(done)
                 : "r"(barrier), "r"(parity)
                 : "memory");
    return done != 0;
}

__device__ __forceinline__ void barrier_wait(uint32_t barrier, uint32_t parity) {
    while (!barrier_try_wait(barrier, parity)) {
    }
}

// Orders this CTA's reads of a stage before the copies that refill it.
__device__ __forceinline__ void async_proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void copy_tile(uint32_t destination, const CUtensorMap *map,
                                          uint32_t barrier, int column, int row) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(destination),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(column), "r"(row), "r"(barrier)
                 : "memory");
}

__device__ __forceinline__ void copy_tile(uint32_t destination, const CUtensorMap *map,
                                          uint32_t barrier, int column, int row, int head,
                                          int batch) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(destination),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(column), "r"(row), "r"(head), "r"(batch),
                 "r"(barrier)
                 : "memory");
}

// The TMA 128B swizzle stores 16-byte chunk c of a 128-byte row r at chunk c ^ (r % 8), so every
// ldmatrix phase touches eight distinct bank groups.
__device__ __forceinline__ uint32_t swizzled_offset(int row, int chunk) {
    return row * 128 + (((chunk & 7) ^ (row & 7)) << 4);
}

PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
    static const PFN_cuTensorMapEncodeTiled_v12000 encoder = [] {
        void *function = nullptr;
        cudaDriverEntryPointQueryResult result;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &function, 12000,
                                             cudaEnableDefault, &result) != cudaSuccess ||
            result != cudaDriverEntryPointSuccess) {
            return static_cast<PFN_cuTensorMapEncodeTiled_v12000>(nullptr);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function);
    }();
    return encoder;
}

} // namespace
