// Loaded lazily on macOS 26+, independently of the macOS 15 FP32 kernels.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;
using namespace mpp::tensor_ops;

// Four SIMD groups compute a 64×64 output tile. Checkpoint weights stay INT8;
// row scales restore the activation range after FP16 conversion / INT8 quantization.
template <typename Input, typename Accumulator>
void packed_product(device Input *a, device int8_t *b, device float *c, device const float *scales,
                    constant uint *p, uint g) {
    const int M = p[0], N = p[1], K = p[2];
    const uint tiles = (uint(N) + 63) / 64;
    const int col = (g % tiles) * 64, row = (g / tiles) * 64;
    tensor<device Input, dextents<int, 2>, tensor_inline> A(a, dextents<int, 2>(K, M),
                                                            array<int, 2>{1, K});
    tensor<device int8_t, dextents<int, 2>, tensor_inline> B(b, dextents<int, 2>(K, N),
                                                             array<int, 2>{1, K});
    auto at = A.slice(0, row);
    auto bt = B.slice(0, col);
    constexpr auto desc = matmul2d_descriptor(64, 64, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto acc =
        op.template get_destination_cooperative_tensor<decltype(at), decltype(bt), Accumulator>();
    op.run(at, bt, acc);
#pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto xy = acc.get_multidimensional_index(i);
            const int r = row + xy[1], n = col + xy[0];
            if (r < M && n < N)
                c[r * N + n] = float(acc[i]) * scales[r];
        }
    }
}

kernel void mpp_fp16(device half *a [[buffer(0)]], device int8_t *b [[buffer(1)]],
                     device float *c [[buffer(2)]], device const float *scales [[buffer(3)]],
                     constant uint *p [[buffer(4)]], uint g [[threadgroup_position_in_grid]]) {
    packed_product<half, float>(a, b, c, scales, p, g);
}

kernel void mpp_int8(device int8_t *a [[buffer(0)]], device int8_t *b [[buffer(1)]],
                     device float *c [[buffer(2)]], device const float *scales [[buffer(3)]],
                     constant uint *p [[buffer(4)]], uint g [[threadgroup_position_in_grid]]) {
    packed_product<int8_t, int32_t>(a, b, c, scales, p, g);
}
