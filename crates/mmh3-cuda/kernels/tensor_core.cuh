#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Tensor core building blocks shared by the attention kernels: cp.async copies, ldmatrix loads, the
// m16n8k16 MMA with FP32 accumulation, and 64-row tiles of K or V swizzled in shared memory.

// Element (batch, token, head, dimension) of the query, key, value and output tensors, in that
// order, lives at base + batch * batch_stride + token * token_stride + head * head_stride +
// dimension.
struct Mmh3AttentionLayout {
    int64_t token_stride[4];
    int64_t head_stride[4];
    int64_t batch_stride[4];
    // Query heads that share one key and value head, where zero means one.
    int32_t heads_per_key_value;
    // Nonzero masks the keys after each query's own position.
    int32_t causal;
};

namespace {

constexpr int BLOCK_N = 64;

__device__ __forceinline__ uint32_t shared_address(const void *pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void copy_async_16(uint32_t destination, const void *source,
                                              bool valid) {
    int source_bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(destination), "l"(source),
                 "r"(source_bytes));
}

__device__ __forceinline__ void copy_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int PENDING> __device__ __forceinline__ void copy_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING));
}

__device__ __forceinline__ void load_matrix_x4(uint32_t (&registers)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]), "=r"(registers[3])
                 : "r"(address));
}

__device__ __forceinline__ void load_matrix_x4_transposed(uint32_t (&registers)[4],
                                                          uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]), "=r"(registers[3])
                 : "r"(address));
}

template <typename Element> struct Numeric;

template <> struct Numeric<__nv_bfloat16> {
    __device__ static void mma(float (&accumulator)[4], const uint32_t (&a)[4], uint32_t b0,
                               uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                     "%5, %6, %7}, {%8, %9}, "
                     "{%0, %1, %2, %3};\n"
                     : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
                       "+f"(accumulator[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ static uint32_t pack(float low, float high) {
        __nv_bfloat162 value = __floats2bfloat162_rn(low, high);
        return *reinterpret_cast<uint32_t *>(&value);
    }
    __device__ static float to_float(uint16_t bits) {
        return __bfloat162float(__ushort_as_bfloat16(bits));
    }
    __device__ static uint16_t from_float(float value) {
        return __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
};

template <> struct Numeric<__half> {
    __device__ static void mma(float (&accumulator)[4], const uint32_t (&a)[4], uint32_t b0,
                               uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, "
                     "%6, %7}, {%8, %9}, "
                     "{%0, %1, %2, %3};\n"
                     : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
                       "+f"(accumulator[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ static uint32_t pack(float low, float high) {
        __half2 value = __floats2half2_rn(low, high);
        return *reinterpret_cast<uint32_t *>(&value);
    }
    __device__ static float to_float(uint16_t bits) { return __half2float(__ushort_as_half(bits)); }
    __device__ static uint16_t from_float(float value) {
        return __half_as_ushort(__float2half_rn(value));
    }
};

__device__ __forceinline__ uint16_t load_shared_16(uint32_t address) {
    uint16_t value;
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(value) : "r"(address));
    return value;
}

__device__ __forceinline__ void store_shared_16(uint32_t address, uint16_t value) {
    asm volatile("st.shared.b16 [%0], %1;\n" ::"r"(address), "h"(value) : "memory");
}

template <int HEAD_DIM> struct Tiles {
    static constexpr int row_bytes = HEAD_DIM * 2;
    static constexpr int chunks_per_row = row_bytes / 16;
    static constexpr int tile_bytes = BLOCK_N * row_bytes;
    static constexpr int stage_bytes = 2 * tile_bytes;
    static constexpr int shared_bytes = 2 * stage_bytes;
    static_assert(chunks_per_row >= 8, "the swizzle needs at least eight chunks per row");

    // XOR-swizzling the chunk by (row % 8) makes every ldmatrix phase, plain or transposed, touch
    // eight distinct 16-byte bank groups.
    __device__ static int offset(int row, int chunk) {
        return row * row_bytes + ((chunk ^ (row & 7)) << 4);
    }
};

template <int HEAD_DIM, int THREAD_COUNT, typename Element>
__device__ __forceinline__ void load_rows(uint32_t destination, const Element *base, int64_t stride,
                                          int first_row, int row_count, int valid_rows) {
    using T = Tiles<HEAD_DIM>;
    for (int index = threadIdx.x; index < row_count * T::chunks_per_row; index += THREAD_COUNT) {
        int row = index / T::chunks_per_row;
        int chunk = index % T::chunks_per_row;
        bool valid = first_row + row < valid_rows;
        const Element *source =
            base + (valid ? static_cast<int64_t>(first_row + row) * stride : 0) + chunk * 8;
        copy_async_16(destination + T::offset(row, chunk), source, valid);
    }
}

} // namespace
