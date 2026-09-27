#pragma once

#include <cstdint>

// Hopper's warpgroup MMA (wgmma): the four warps of a warpgroup issue one asynchronous product of a
// 64-row tile of A and an N-row tile of B, read from shared memory through descriptors or, for A,
// from registers in the mma.sync A layout, into accumulators spread over the warpgroup's
// registers. Thread t of warp w holds, in register i, row
// 16 w + (t / 4) + 8 ((i / 2) % 2) and column 8 (i / 4) + 2 (t % 4) + (i % 2), the mma.sync C
// layout repeated over the columns.
//
// NOTE: only the sm_90a machine code has these instructions. Every other architecture still
// compiles the kernels that use them, since one template holds both paths, and the host launches
// them only where mmh3_wgmma_available() says so, so there they become traps.

#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MMH3_WGMMA_TRAP 1
#endif

namespace {

// A shared memory descriptor for a K-major operand in the TMA 128B swizzle: rows of 128 bytes in
// groups of eight, 1024 bytes apart. `address` must lie in a 1024-byte aligned group of rows, and a
// K step inside the rows moves it by its bytes.
__device__ __forceinline__ uint64_t wgmma_descriptor(uint32_t address) {
    constexpr uint64_t leading_byte_offset = 1;
    constexpr uint64_t stride_byte_offset = 1024 >> 4;
    constexpr uint64_t swizzle_128b = 1;
    return static_cast<uint64_t>((address & 0x3FFFF) >> 4) | leading_byte_offset << 16 |
           stride_byte_offset << 32 | swizzle_128b << 62;
}

// The same for rows of 64 bytes in the TMA 64B swizzle, eight of them 512 bytes apart.
__device__ __forceinline__ uint64_t wgmma_descriptor_64b(uint32_t address) {
    constexpr uint64_t leading_byte_offset = 1;
    constexpr uint64_t stride_byte_offset = 512 >> 4;
    constexpr uint64_t swizzle_64b = 2;
    return static_cast<uint64_t>((address & 0x3FFFF) >> 4) | leading_byte_offset << 16 |
           stride_byte_offset << 32 | swizzle_64b << 62;
}

// A descriptor for an MN-major B in the same swizzle, which the transposed forms read: the N values
// of a row lie in 128-byte slabs `slab_bytes` apart, and K runs down the rows, eight of them 1024
// bytes apart. A K step of 16 rows moves `address` by 2048 bytes.
__device__ __forceinline__ uint64_t wgmma_descriptor_mn_major(uint32_t address,
                                                              uint32_t slab_bytes) {
    constexpr uint64_t stride_byte_offset = 1024 >> 4;
    constexpr uint64_t swizzle_128b = 1;
    return static_cast<uint64_t>((address & 0x3FFFF) >> 4) |
           static_cast<uint64_t>(slab_bytes >> 4) << 16 | stride_byte_offset << 32 |
           swizzle_128b << 62;
}

// Makes the accumulator registers written by other instructions visible to the next wgmma.
__device__ __forceinline__ void wgmma_fence() {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void wgmma_commit() {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}

// Waits until at most PENDING committed groups are still running.
template <int PENDING> __device__ __forceinline__ void wgmma_wait() {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(PENDING) : "memory");
#endif
}

// Orders this thread's ordinary shared memory writes before the async proxy, which wgmma and TMA
// read through.
__device__ __forceinline__ void shared_to_async_proxy_fence() {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

// Hands registers from the warpgroups that copy to the ones that multiply.
template <int REGISTERS> __device__ __forceinline__ void registers_decrease() {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(REGISTERS));
#endif
}

template <int REGISTERS> __device__ __forceinline__ void registers_increase() {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(REGISTERS));
#endif
}

// D (+)= A · Bᵀ with both K-major in shared memory, over K = 32 bytes: 32 INT8 values into INT32,
// or 16 BF16 or FP16 values into FP32. `accumulate` zero overwrites D.

__device__ __forceinline__ void wgmma_s8_m64n256k32(uint32_t (&d)[128], uint64_t a, uint64_t b,
                                                    int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile(
        "{\n"
        ".reg .pred accumulate;\n"
        "setp.ne.b32 accumulate, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, "
        "%37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
        "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, "
        "%73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, "
        "%91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, "
        "%107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, "
        "%122, %123, %124, %125, %126, %127}, "
        "%128, %129, accumulate;\n"
        "}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
          "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]),
          "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]),
          "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]),
          "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]),
          "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]),
          "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]),
          "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
          "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]),
          "+r"(d[63]), "+r"(d[64]), "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), "+r"(d[68]), "+r"(d[69]),
          "+r"(d[70]), "+r"(d[71]), "+r"(d[72]), "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), "+r"(d[76]),
          "+r"(d[77]), "+r"(d[78]), "+r"(d[79]), "+r"(d[80]), "+r"(d[81]), "+r"(d[82]), "+r"(d[83]),
          "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]), "+r"(d[88]), "+r"(d[89]), "+r"(d[90]),
          "+r"(d[91]), "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]), "+r"(d[96]), "+r"(d[97]),
          "+r"(d[98]), "+r"(d[99]), "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]),
          "+r"(d[104]), "+r"(d[105]), "+r"(d[106]), "+r"(d[107]), "+r"(d[108]), "+r"(d[109]),
          "+r"(d[110]), "+r"(d[111]), "+r"(d[112]), "+r"(d[113]), "+r"(d[114]), "+r"(d[115]),
          "+r"(d[116]), "+r"(d[117]), "+r"(d[118]), "+r"(d[119]), "+r"(d[120]), "+r"(d[121]),
          "+r"(d[122]), "+r"(d[123]), "+r"(d[124]), "+r"(d[125]), "+r"(d[126]), "+r"(d[127])
        : "l"(a), "l"(b), "r"(accumulate));
#endif
}

__device__ __forceinline__ void wgmma_s8_m64n128k32(uint32_t (&d)[64], uint64_t a, uint64_t b,
                                                    int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %66, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "
                 "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "
                 "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
                 "%64, %65, accumulate;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]),
                   "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]),
                   "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
                   "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]),
                   "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]),
                   "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
                 : "l"(a), "l"(b), "r"(accumulate));
#endif
}

__device__ __forceinline__ void wgmma_bf16_m64n256k16(uint32_t (&d)[128], uint64_t a, uint64_t b,
                                                      int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile(
        "{\n"
        ".reg .pred accumulate;\n"
        "setp.ne.b32 accumulate, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, "
        "%37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
        "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, "
        "%73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, "
        "%91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, "
        "%107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, "
        "%122, %123, %124, %125, %126, %127}, "
        "%128, %129, accumulate, 1, 1, 0, 0;\n"
        "}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
          "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]),
          "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]),
          "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]),
          "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]),
          "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]),
          "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]),
          "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
          "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]),
          "+r"(d[63]), "+r"(d[64]), "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), "+r"(d[68]), "+r"(d[69]),
          "+r"(d[70]), "+r"(d[71]), "+r"(d[72]), "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), "+r"(d[76]),
          "+r"(d[77]), "+r"(d[78]), "+r"(d[79]), "+r"(d[80]), "+r"(d[81]), "+r"(d[82]), "+r"(d[83]),
          "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]), "+r"(d[88]), "+r"(d[89]), "+r"(d[90]),
          "+r"(d[91]), "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]), "+r"(d[96]), "+r"(d[97]),
          "+r"(d[98]), "+r"(d[99]), "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]),
          "+r"(d[104]), "+r"(d[105]), "+r"(d[106]), "+r"(d[107]), "+r"(d[108]), "+r"(d[109]),
          "+r"(d[110]), "+r"(d[111]), "+r"(d[112]), "+r"(d[113]), "+r"(d[114]), "+r"(d[115]),
          "+r"(d[116]), "+r"(d[117]), "+r"(d[118]), "+r"(d[119]), "+r"(d[120]), "+r"(d[121]),
          "+r"(d[122]), "+r"(d[123]), "+r"(d[124]), "+r"(d[125]), "+r"(d[126]), "+r"(d[127])
        : "l"(a), "l"(b), "r"(accumulate));
#endif
}

__device__ __forceinline__ void wgmma_bf16_m64n128k16(uint32_t (&d)[64], uint64_t a, uint64_t b,
                                                      int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %66, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "
                 "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "
                 "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
                 "%64, %65, accumulate, 1, 1, 0, 0;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]),
                   "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]),
                   "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
                   "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]),
                   "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]),
                   "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
                 : "l"(a), "l"(b), "r"(accumulate));
#endif
}

__device__ __forceinline__ void wgmma_bf16_m64n64k16(uint32_t (&d)[32], uint64_t a, uint64_t b,
                                                     int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "%32, %33, accumulate, 1, 1, 0, 0;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31])
                 : "l"(a), "l"(b), "r"(accumulate));
#endif
}

__device__ __forceinline__ void wgmma_f16_m64n64k16(uint32_t (&d)[32], uint64_t a, uint64_t b,
                                                    int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "%32, %33, accumulate, 1, 1, 0, 0;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31])
                 : "l"(a), "l"(b), "r"(accumulate));
#endif
}

__device__ __forceinline__ void wgmma_s8_m64n64k32(uint32_t (&d)[32], uint64_t a, uint64_t b,
                                                   int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "%32, %33, accumulate;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31])
                 : "l"(a), "l"(b), "r"(accumulate));
#endif
}

// D (+)= A · B over K = 16 values, with A from this thread's registers in the mma.sync m16n8k16 A
// layout of its warp's 16 rows, and B MN-major in shared memory (wgmma_descriptor_mn_major).

__device__ __forceinline__ void wgmma_bf16_m64n128k16_registers_transposed(uint32_t (&d)[64],
                                                                           const uint32_t (&a)[4],
                                                                           uint64_t b,
                                                                           int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %69, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "
                 "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "
                 "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
                 "{%64, %65, %66, %67}, %68, accumulate, 1, 1, 1;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]),
                   "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]),
                   "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
                   "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]),
                   "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]),
                   "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(accumulate));
#endif
}

__device__ __forceinline__ void wgmma_bf16_m64n64k16_registers_transposed(uint32_t (&d)[32],
                                                                          const uint32_t (&a)[4],
                                                                          uint64_t b,
                                                                          int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %37, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "{%32, %33, %34, %35}, %36, accumulate, 1, 1, 1;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(accumulate));
#endif
}

__device__ __forceinline__ void wgmma_f16_m64n128k16_registers_transposed(uint32_t (&d)[64],
                                                                          const uint32_t (&a)[4],
                                                                          uint64_t b,
                                                                          int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %69, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "
                 "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "
                 "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
                 "{%64, %65, %66, %67}, %68, accumulate, 1, 1, 1;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]),
                   "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]),
                   "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
                   "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]),
                   "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]),
                   "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(accumulate));
#endif
}

__device__ __forceinline__ void wgmma_f16_m64n64k16_registers_transposed(uint32_t (&d)[32],
                                                                         const uint32_t (&a)[4],
                                                                         uint64_t b,
                                                                         int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %37, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "{%32, %33, %34, %35}, %36, accumulate, 1, 1, 1;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(accumulate));
#endif
}

// D (+)= A · Bᵀ over K = 32 FP8 E4M3 values into FP32, with A from this thread's registers in the
// mma.sync m16n8k32 A layout and B K-major in shared memory.

__device__ __forceinline__ void wgmma_e4m3_m64n128k32_registers(uint32_t (&d)[64],
                                                                const uint32_t (&a)[4], uint64_t b,
                                                                int accumulate) {
#ifdef MMH3_WGMMA_TRAP
    __trap();
#else
    asm volatile("{\n"
                 ".reg .pred accumulate;\n"
                 "setp.ne.b32 accumulate, %69, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "
                 "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "
                 "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
                 "{%64, %65, %66, %67}, %68, accumulate, 1, 1;\n"
                 "}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                   "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
                   "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                   "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]),
                   "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]),
                   "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
                   "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]),
                   "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]),
                   "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(accumulate));
#endif
}

} // namespace
