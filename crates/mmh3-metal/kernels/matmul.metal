// Loaded lazily on macOS 26+, independently of the macOS 15 FP32 kernels.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;
using namespace mpp::tensor_ops;

// What a product does to its result besides the rows' scales, as bits of its flags: scale each
// output by the weight's own scale, add a bias, add what the output already holds, and add a
// LoRA's up projection of the down projection's result, `mid`.
constant constexpr uint SCALE_OUTPUTS = 1, ADD_BIAS = 2, ACCUMULATE = 4, ADD_LORA = 8;

// A threadgroup computes a 128×64 output tile. Checkpoint weights stay INT8; row scales restore
// the activation range after FP16 conversion / INT8 quantization. INT8 runs on four SIMD groups
// and FP16 on eight: those are the shapes that keep the matrix units busy without spilling.
//
// A LoRA's up projection is a product of its own over the tile, with the rank for depth. Written
// out whole it would be an FP32 output the size of the layer's, written once and read back once,
// so the tile's share goes to threadgroup memory and the product's writes add it. It goes in FP16:
// in FP32 it would fill the 32 KB a threadgroup has and slow the product, and its rounding is a
// small fraction of the INT8 activations' own.
template <typename Input, typename Accumulator, int GROUPS>
void packed_product(device Input *a, device int8_t *b, device float *c, device const float *scales,
                    device const float *outputs, device const float *bias, device float *mid,
                    device half *up, constant uint *p, uint g, threadgroup half *lora) {
    constexpr int TM = 128, TN = 64;
    const int M = p[0], N = p[1], K = p[2], rank = p[4];
    const uint flags = p[3];
    const uint tiles = (uint(N) + TN - 1) / TN;
    const int col = (g % tiles) * TN, row = (g / tiles) * TM;
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);

    if (flags & ADD_LORA) {
        tensor<device float, dextents<int, 2>, tensor_inline> Mid(mid, dextents<int, 2>(rank, M),
                                                                  array<int, 2>{1, rank});
        tensor<device half, dextents<int, 2>, tensor_inline> Up(up, dextents<int, 2>(rank, N),
                                                                array<int, 2>{1, rank});
        auto mt = Mid.slice(0, row);
        auto ut = Up.slice(0, col);
        matmul2d<desc, execution_simdgroups<GROUPS>> lora_op;
        auto added =
            lora_op
                .template get_destination_cooperative_tensor<decltype(mt), decltype(ut), float>();
        lora_op.run(mt, ut, added);
#pragma unroll
        for (uint i = 0; i < added.get_capacity(); ++i) {
            if (added.is_valid_element(i)) {
                auto xy = added.get_multidimensional_index(i);
                lora[xy[1] * TN + xy[0]] = half(added[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    tensor<device Input, dextents<int, 2>, tensor_inline> A(a, dextents<int, 2>(K, M),
                                                            array<int, 2>{1, K});
    tensor<device int8_t, dextents<int, 2>, tensor_inline> B(b, dextents<int, 2>(K, N),
                                                             array<int, 2>{1, K});
    auto at = A.slice(0, row);
    auto bt = B.slice(0, col);
    matmul2d<desc, execution_simdgroups<GROUPS>> op;
    auto acc =
        op.template get_destination_cooperative_tensor<decltype(at), decltype(bt), Accumulator>();
    op.run(at, bt, acc);
#pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto xy = acc.get_multidimensional_index(i);
            const int r = row + xy[1], n = col + xy[0];
            if (r < M && n < N) {
                float value = float(acc[i]) * scales[r];
                if (flags & SCALE_OUTPUTS)
                    value *= outputs[n];
                if (flags & ADD_BIAS)
                    value += bias[n];
                if (flags & ACCUMULATE)
                    value += c[r * N + n];
                if (flags & ADD_LORA)
                    value += float(lora[xy[1] * TN + xy[0]]);
                c[r * N + n] = value;
            }
        }
    }
}

#define PACKED_PRODUCT(NAME, INPUT, ACCUMULATOR, GROUPS)                                           \
    kernel void NAME(device INPUT *a [[buffer(0)]], device int8_t *b [[buffer(1)]],                \
                     device float *c [[buffer(2)]], device const float *scales [[buffer(3)]],      \
                     device const float *outputs [[buffer(4)]],                                    \
                     device const float *bias [[buffer(5)]], device float *mid [[buffer(6)]],      \
                     device half *up [[buffer(7)]], constant uint *p [[buffer(8)]],                \
                     uint g [[threadgroup_position_in_grid]]) {                                    \
        threadgroup half lora[128 * 64];                                                           \
        packed_product<INPUT, ACCUMULATOR, GROUPS>(a, b, c, scales, outputs, bias, mid, up, p, g,  \
                                                   lora);                                          \
    }
PACKED_PRODUCT(mpp_fp16, half, float, 8)
PACKED_PRODUCT(mpp_int8, int8_t, int32_t, 4)
#undef PACKED_PRODUCT

// C = A · Bᵀ for FP32 activations and FP16 weights, the two products of a LoRA. A 64×64 tile a
// threadgroup: a LoRA's down projection has few outputs, and larger tiles would leave most of
// the GPU without one.
kernel void mpp_lora(device float *a [[buffer(0)]], device half *b [[buffer(1)]],
                     device float *c [[buffer(2)]], constant uint *p [[buffer(3)]],
                     uint g [[threadgroup_position_in_grid]]) {
    constexpr int TM = 64, TN = 64;
    const int M = p[0], N = p[1], K = p[2];
    const uint tiles = (uint(N) + TN - 1) / TN;
    const int col = (g % tiles) * TN, row = (g / tiles) * TM;
    tensor<device float, dextents<int, 2>, tensor_inline> A(a, dextents<int, 2>(K, M),
                                                            array<int, 2>{1, K});
    tensor<device half, dextents<int, 2>, tensor_inline> B(b, dextents<int, 2>(K, N),
                                                           array<int, 2>{1, K});
    auto at = A.slice(0, row);
    auto bt = B.slice(0, col);
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(at), decltype(bt), float>();
    op.run(at, bt, acc);
#pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto xy = acc.get_multidimensional_index(i);
            const int r = row + xy[1], n = col + xy[0];
            if (r < M && n < N)
                c[r * N + n] = acc[i];
        }
    }
}

// Flash attention on the matrix units for heads of D, 64 or 128. A threadgroup of eight SIMD
// groups owns 128 queries of one head and walks the keys 64 at a time. Q·Kᵀ and P·V are tensor
// products in FP16 with FP32 accumulation, which the eight groups run together: that is the shape
// that keeps the matrix units busy, where a group running its own rows leaves them mostly idle.
//
// The online softmax runs between the products, two threads a query row. The scores go to
// threadgroup memory as FP32, and the FP16 probabilities are written back over them, so the
// 32 KB a threadgroup has holds both. A row's probabilities fill the first half of its 256 bytes,
// and its correction sits in the last four.
//
// Rescaling the output for a new maximum costs as much as the softmax, so a row moves its
// reference only when a score passes it by more than RAISE. Probabilities then reach e^RAISE,
// which FP16 holds, and a tile where no row moved skips the rescale. Whether any row of a SIMD
// group moved goes in the float before the correction of the group's first row.
constant constexpr int ATTENTION_QUERIES = 128, ATTENTION_KEYS = 64;
constant constexpr float RAISE = 8;

template <int D>
void flash_attention(device half *q, device half *k, device half *v, device float *o,
                     constant uint *p, uint g, uint tid, uint simd, uint lane,
                     threadgroup float *scores) {
    constexpr int QUERIES = ATTENTION_QUERIES, KEYS = ATTENTION_KEYS;
    constexpr int SPAN = KEYS / 2, LAST = KEYS - 1;
    const int count = p[0], heads = p[1], kv_heads = p[2], causal = p[4], rows = p[5];
    const int head = g % heads, first = (g / heads) * QUERIES;
    const int kv_head = head / (heads / kv_heads);
    const int limit = causal ? min(count, first + QUERIES) : count;
    const float scale = rsqrt(float(D));
    threadgroup half *probabilities = (threadgroup half *)scores;

    using Tile = tensor<device half, dextents<int, 2>, tensor_inline>;
    using Shared = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
    Tile queries(q + (first * heads + head) * D, dextents<int, 2>(D, min(QUERIES, rows - first)),
                 array<int, 2>{1, heads * D});

    constexpr auto score_desc =
        matmul2d_descriptor(QUERIES, KEYS, dynamic_length_v<int>, false, true, false);
    matmul2d<score_desc, execution_simdgroups<8>> score_op;
    constexpr auto value_desc =
        matmul2d_descriptor(QUERIES, D, dynamic_length_v<int>, false, false, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<value_desc, execution_simdgroups<8>> value_op;

    auto out = value_op.template get_destination_cooperative_tensor<Shared, Tile, float>();
#pragma unroll
    for (uint i = 0; i < out.get_capacity(); ++i)
        if (out.is_valid_element(i))
            out[i] = 0;

    const int row = tid / 2, column = (tid % 2) * SPAN;
    const bool live = first + row < rows;
    // The reference is -∞ until the row has seen a key.
    float reference = -INFINITY, total = 0;
    for (int base = 0; base < limit; base += KEYS) {
        const int length = min(KEYS, count - base);
        Tile keys(k + (base * kv_heads + kv_head) * D, dextents<int, 2>(D, length),
                  array<int, 2>{1, kv_heads * D});
        auto s = score_op.template get_destination_cooperative_tensor<Tile, Tile, float>();
        score_op.run(queries, keys, s);
#pragma unroll
        for (uint i = 0; i < s.get_capacity(); ++i) {
            if (s.is_valid_element(i)) {
                auto xy = s.get_multidimensional_index(i);
                scores[xy[1] * KEYS + xy[0]] = s[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float x[SPAN], local = -INFINITY;
#pragma unroll
        for (int c = 0; c < SPAN; ++c) {
            const int key = base + column + c;
            const bool seen = live && key < count && !(causal && key > first + row);
            x[c] = seen ? scores[row * KEYS + column + c] * scale : -INFINITY;
            local = max(local, x[c]);
        }
        local = max(local, simd_shuffle_xor(local, 1));
        float correction = 1;
        if (reference == -INFINITY) {
            reference = local;
        } else if (local > reference + RAISE) {
            correction = exp(reference - local);
            reference = local;
        }
        // A row with no key seen yet takes nothing from this tile.
        const float offset = reference == -INFINITY ? 0.0f : reference;
        float sum = 0;
#pragma unroll
        for (int c = 0; c < SPAN; ++c) {
            x[c] = exp(x[c] - offset);
            sum += x[c];
        }
        sum += simd_shuffle_xor(sum, 1);
        total = total * correction + sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
        for (int c = 0; c < SPAN; ++c)
            probabilities[row * 2 * KEYS + column + c] = half(x[c]);
        if (column == 0)
            scores[row * KEYS + LAST] = correction;
        const bool moved = simd_any(correction != 1);
        if (lane == 0)
            scores[simd * KEYS + LAST - 1] = moved;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool rescale = false;
        for (int group = 0; group < 8; ++group)
            rescale |= scores[group * KEYS + LAST - 1] != 0;

        if (rescale) {
#pragma unroll
            for (uint i = 0; i < out.get_capacity(); ++i)
                if (out.is_valid_element(i))
                    out[i] *= scores[out.get_multidimensional_index(i)[1] * KEYS + LAST];
        }
        Shared weights(probabilities, dextents<int, 2>(length, QUERIES),
                       array<int, 2>{1, 2 * KEYS});
        Tile values(v + (base * kv_heads + kv_head) * D, dextents<int, 2>(D, length),
                    array<int, 2>{1, kv_heads * D});
        value_op.run(weights, values, out);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (column == 0)
        scores[row * KEYS + LAST] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint i = 0; i < out.get_capacity(); ++i) {
        if (out.is_valid_element(i)) {
            auto xy = out.get_multidimensional_index(i);
            const int r = first + xy[1];
            if (r < rows && xy[0] < D)
                o[(r * heads + head) * D + xy[0]] = out[i] / scores[xy[1] * KEYS + LAST];
        }
    }
}

#define ATTENTION(D)                                                                               \
    kernel void mpp_attention_##D(                                                                 \
        device half *q [[buffer(0)]], device half *k [[buffer(1)]], device half *v [[buffer(2)]],  \
        device float *o [[buffer(3)]], constant uint *p [[buffer(4)]],                             \
        uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],         \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {   \
        threadgroup float scores[ATTENTION_QUERIES * ATTENTION_KEYS];                              \
        flash_attention<D>(q, k, v, o, p, g, tid, simd, lane, scores);                             \
    }
ATTENTION(64)
ATTENTION(128)
#undef ATTENTION
