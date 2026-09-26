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
    // Threadgroups go down bands of p[5] row tiles, so the ones running together share weights in
    // the cache.
    const uint tiles = (uint(N) + TN - 1) / TN, row_tiles = (uint(M) + TM - 1) / TM;
    const uint BAND = max(p[5], 1u);
    const uint band = g / (BAND * tiles), in_band = g % (BAND * tiles);
    const uint band_rows = min(BAND, row_tiles - band * BAND);
    const int col = (in_band / band_rows) * TN, row = (band * BAND + in_band % band_rows) * TM;
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

// Flash attention on the matrix units, after MLX's attention for the Neural Accelerators: each SIMD
// group owns 16 query rows and keeps their scores, softmax and output in its registers, as 16 × 16
// fragments of the matrix units' cooperative tensors, so no threadgroup memory or barrier is
// needed. A fragment's lane holds rows r and r + 8 and four of the columns from c, the same in
// both. Products take FP16 queries, keys and values with FP32 accumulation, and the scores stay
// FP32 into the product with the values. The scores are in log2 units.
constant constexpr int FRAGMENT = 16, ATTENTION_QUERIES = 64, ATTENTION_KEYS = 32;

using fragment_half = vec<half, 8>;
using fragment_float = vec<float, 8>;

// Where the cooperative tensors put their elements depends on the GPU, so the runtime sets
// FRAGMENT_PAIRS for one without the Neural Accelerators. There a lane holds columns c, c + 1,
// c + 8 and c + 9 of a fragment rather than c to c + 3, a 16 × 32 tensor goes through its two
// fragments' first rows before their second, and the right operand of a product with b transposed
// holds its columns of b by the lane's rows. Either way, the lanes that hold a row differ in bits 0
// and 3.
#ifndef FRAGMENT_PAIRS
#define FRAGMENT_PAIRS 0
#endif

// The column c and first row r of a lane's elements in a fragment.
short2 fragment_coord(ushort lane) {
    const short row = (lane & 16) >> 2 | ((lane >> 1) & 3);
    return FRAGMENT_PAIRS ? short2{short((lane & 1) * 2 + ((lane >> 3) & 1) * 4), row}
                          : short2{short((lane & 1) * 4 + ((lane >> 3) & 1) * 8), row};
}

// How far the j-th of a lane's four columns in a fragment is from its first, c.
constexpr short fragment_column(short j) { return FRAGMENT_PAIRS ? (j & 1) + (j >> 1) * 8 : j; }

// A fragment of `src`, rows `stride` apart, or with RIGHT the right operand of a product with b
// transposed that `src` holds, b's columns as its rows. With CHECK, only its first `rows` rows are
// read and the rest are zero; a whole block skips the checks, which slow the attention down
// markedly. A lane's columns are offsets from one address, so that the consecutive ones are read
// together.
template <bool CHECK, bool RIGHT = false, typename T = half>
vec<T, 8> load_fragment(device const T *src, int stride, int rows, short2 at) {
    vec<T, 8> out;
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 2; ++i) {
        const int r = at.y + i * 8;
        _Pragma("clang loop unroll(full)") for (short j = 0; j < 4; ++j) {
            const short column = fragment_column(j);
            if constexpr (RIGHT && FRAGMENT_PAIRS) {
                out[i * 4 + j] =
                    !CHECK || at.x + column < rows ? src[(at.x + column) * stride + r] : T(0);
            } else {
                out[i * 4 + j] = !CHECK || r < rows ? src[r * stride + at.x + column] : T(0);
            }
        }
    }
    return out;
}

// Which fragment of a 16 × 32 tensor, and which of its elements, is the tensor's element i.
constexpr short fragment_of(short i) { return FRAGMENT_PAIRS ? (i >> 2) & 1 : i >> 3; }
constexpr short element_of(short i) { return FRAGMENT_PAIRS ? (i >> 3) << 2 | (i & 3) : i & 7; }

// c (16 × 32, two fragments) += a (16 × 16) · b (16 × 32, two fragments), with b transposed when
// TRANSPOSE: then its fragments are its rows 0 to 15 and 16 to 31 as stored, loaded with RIGHT.
template <bool TRANSPOSE, typename A, typename B = half, typename C = float>
void multiply_fragments(thread vec<C, 8> &c0, thread vec<C, 8> &c1, thread const vec<A, 8> &a,
                        thread const vec<B, 8> &b0, thread const vec<B, 8> &b1) {
    constexpr auto desc = matmul2d_descriptor(FRAGMENT, 2 * FRAGMENT, FRAGMENT, false, TRANSPOSE,
                                              true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
    auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
    auto out =
        op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(left)>,
                                                       remove_addrspace_t<decltype(right)>, C>();
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 8; ++i) left[i] = a[i];
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) {
        const short e = element_of(i);
        right[i] = fragment_of(i) ? b1[e] : b0[e];
        out[i] = fragment_of(i) ? c1[e] : c0[e];
    }
    op.run(left, right, out);
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) {
        if (fragment_of(i))
            c1[element_of(i)] = out[i];
        else
            c0[element_of(i)] = out[i];
    }
}

// The scores of a SIMD group's 16 query rows against a block of up to 32 keys, by FP16 products:
// `queries` are the group's rows, `rows` of them real, and `keys` the block's, `length` of them
// real.
template <int D> struct HalfScores {
    device const half *queries, *keys;
    int q_stride, kv_stride;

    template <bool CHECK>
    void score(thread fragment_float (&s)[2], int rows, int length, short2 at) const {
        _Pragma("clang loop unroll(full)") for (short part = 0; part < D / FRAGMENT; ++part) {
            const fragment_half a =
                load_fragment<CHECK>(queries + part * FRAGMENT, q_stride, rows, at);
            const fragment_half b0 =
                load_fragment<CHECK, true>(keys + part * FRAGMENT, kv_stride, length, at);
            const fragment_half b1 = load_fragment<CHECK, true>(
                keys + FRAGMENT * kv_stride + part * FRAGMENT, kv_stride, length - FRAGMENT, at);
            multiply_fragments<true>(s[0], s[1], a, b0, b1);
        }
    }
};

// HalfScores by INT8 products with INT32 accumulation, each score then taking its query's and
// key's scales, which are `scale_stride` apart row by row.
template <int D> struct Int8Scores {
    device const int8_t *queries, *keys;
    device const float *query_scales, *key_scales;
    int q_stride, kv_stride, scale_stride;

    template <bool CHECK>
    void score(thread fragment_float (&s)[2], int rows, int length, short2 at) const {
        vec<int, 8> products[2] = {0, 0};
        _Pragma("clang loop unroll(full)") for (short part = 0; part < D / FRAGMENT; ++part) {
            const vec<int8_t, 8> a =
                load_fragment<CHECK>(queries + part * FRAGMENT, q_stride, rows, at);
            const vec<int8_t, 8> b0 =
                load_fragment<CHECK, true>(keys + part * FRAGMENT, kv_stride, length, at);
            const vec<int8_t, 8> b1 = load_fragment<CHECK, true>(
                keys + FRAGMENT * kv_stride + part * FRAGMENT, kv_stride, length - FRAGMENT, at);
            multiply_fragments<true, int8_t, int8_t, int>(products[0], products[1], a, b0, b1);
        }
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 2; ++i) {
            const int r = at.y + i * 8;
            const float query = !CHECK || r < rows ? query_scales[r * scale_stride] : 0.0f;
            _Pragma("clang loop unroll(full)") for (short f = 0; f < 2; ++f) {
                _Pragma("clang loop unroll(full)") for (short j = 0; j < 4; ++j) {
                    const int key = f * FRAGMENT + at.x + fragment_column(j);
                    const float scale =
                        !CHECK || key < length ? query * key_scales[key * scale_stride] : 0.0f;
                    s[f][i * 4 + j] = float(products[f][i * 4 + j]) * scale;
                }
            }
        }
    }
};

// The running softmax of a SIMD group's 16 query rows over the keys seen so far: its output before
// the division, and each of the lane's two rows' reference and total.
template <int D> struct AttentionState {
    fragment_float out[D / FRAGMENT];
    float reference[2] = {-INFINITY, -INFINITY}, total[2] = {0, 0};

    AttentionState() {
        _Pragma("clang loop unroll(full)") for (short f = 0; f < D / FRAGMENT; ++f) out[f] = 0;
    }

    // Takes in a block of up to 32 keys, which `scores` scores against the group's rows, `rows`
    // of them real. `values` point at the block's first, rows `kv_stride` apart, and `length` of
    // them are real. `seen` says whether a (row, key) pair takes part, and `offset` is taken off
    // each of the lane's rows.
    template <bool CHECK, typename Scores, typename Seen>
    void attend(thread const Scores &scores, int rows, device const half *values, int kv_stride,
                int length, float scale_log2, thread const float *offset, Seen seen, short2 at) {
        constexpr int PARTS = D / FRAGMENT;
        // The SIMD groups take each block together, so that its keys and values are read into
        // the cache once for all four.
        threadgroup_barrier(mem_flags::mem_none);
        fragment_float s[2] = {0, 0};
        scores.template score<CHECK>(s, rows, length, at);

        float maximum[2];
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 2; ++i) {
            maximum[i] = reference[i];
            _Pragma("clang loop unroll(full)") for (short f = 0; f < 2; ++f) {
                _Pragma("clang loop unroll(full)") for (short j = 0; j < 4; ++j) {
                    const int key = f * FRAGMENT + at.x + fragment_column(j);
                    const float x = key < length && seen(at.y + i * 8, key)
                                        ? s[f][i * 4 + j] * scale_log2 - offset[i]
                                        : -INFINITY;
                    s[f][i * 4 + j] = x;
                    maximum[i] = max(maximum[i], x);
                }
            }
            maximum[i] = max(maximum[i], simd_shuffle_xor(maximum[i], 1));
            maximum[i] = max(maximum[i], simd_shuffle_xor(maximum[i], 8));
        }
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 2; ++i) {
            // A row with no key seen yet takes nothing from this block.
            const float shift = maximum[i] == -INFINITY ? 0.0f : maximum[i];
            const float correction = reference[i] == -INFINITY ? 0.0f : exp2(reference[i] - shift);
            float sum = 0;
            _Pragma("clang loop unroll(full)") for (short f = 0; f < 2; ++f) {
                _Pragma("clang loop unroll(full)") for (short j = 0; j < 4; ++j) {
                    const float x = exp2(s[f][i * 4 + j] - shift);
                    s[f][i * 4 + j] = x;
                    sum += x;
                }
            }
            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 8);
            total[i] = total[i] * correction + sum;
            reference[i] = maximum[i];
            _Pragma("clang loop unroll(full)") for (short f = 0; f < PARTS; ++f)
                _Pragma("clang loop unroll(full)") for (short j = 0; j < 4; ++j)
                    out[f][i * 4 + j] *= correction;
        }

        _Pragma("clang loop unroll(full)")

            for (short part = 0; part < PARTS; part += 2) {
            _Pragma("clang loop unroll(full)") for (short half_block = 0; half_block < 2;
                                                    ++half_block) {
                device const half *rows_of = values + half_block * FRAGMENT * kv_stride;
                const int left = length - half_block * FRAGMENT;
                const fragment_half b0 =
                    load_fragment<CHECK>(rows_of + part * FRAGMENT, kv_stride, left, at);
                const fragment_half b1 =
                    load_fragment<CHECK>(rows_of + (part + 1) * FRAGMENT, kv_stride, left, at);
                multiply_fragments<false>(out[part], out[part + 1], s[half_block], b0, b1);
            }
        }
    }
};

// Dense attention for heads of D, grouped heads and a causal mask included: a threadgroup of four
// SIMD groups takes 64 queries of one head.
template <int D, bool INT8>
void flash_attention(device half *q, device half *k, device half *v, device float *o,
                     device const int8_t *q8, device const int8_t *k8, device const float *scales,
                     constant uint *p, uint g, uint simd, uint lane) {
    const int count = p[0], heads = p[1], kv_heads = p[2], causal = p[4], rows = p[5];
    // Threadgroups take a head's query blocks one after another, so the ones running together
    // read the same keys and values, which then stay in the cache. Taking the heads of one block
    // together instead is far slower.
    const int blocks = (rows + ATTENTION_QUERIES - 1) / ATTENTION_QUERIES;
    const int head = g / blocks, kv_head = head / (heads / kv_heads), block = g % blocks;
    const int first = block * ATTENTION_QUERIES + simd * FRAGMENT;
    // A group past the last query still takes every block, since the four take them together.
    const int mine = rows - first;
    const short2 at = fragment_coord(lane);
    const float scale_log2 = rsqrt(float(D)) * M_LOG2E_F, offset[2] = {0, 0};
    const int q_stride = heads * D, kv_stride = kv_heads * D;
    device const half *queries = q + (first * heads + head) * D;
    const int limit = causal ? min(count, (block + 1) * ATTENTION_QUERIES) : count;
    AttentionState<D> state;
    for (int base = 0; base < limit; base += ATTENTION_KEYS) {
        device const half *keys = k + (base * kv_heads + kv_head) * D;
        device const half *values = v + (base * kv_heads + kv_head) * D;
        const int length = min(ATTENTION_KEYS, count - base);
        auto seen = [&](int row, int key) { return !causal || base + key <= first + row; };
        auto attend = [&](thread const auto &scores) {
            if (length == ATTENTION_KEYS && mine >= FRAGMENT)
                state.template attend<false>(scores, mine, values, kv_stride, length, scale_log2,
                                             offset, seen, at);
            else
                state.template attend<true>(scores, mine, values, kv_stride, length, scale_log2,
                                            offset, seen, at);
        };
        if constexpr (INT8)
            attend(Int8Scores<D>{
                q8 + (first * heads + head) * D, k8 + (base * kv_heads + kv_head) * D,
                scales + first * heads + head, scales + (rows + base) * kv_heads + kv_head,
                q_stride, kv_stride, heads});
        else
            attend(HalfScores<D>{queries, keys, q_stride, kv_stride});
    }
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 2; ++i) {
        const int r = at.y + i * 8;
        if (r >= mine)
            continue;
        _Pragma("clang loop unroll(full)") for (short f = 0; f < D / FRAGMENT; ++f)
            _Pragma("clang loop unroll(full)") for (short j = 0; j < 4; ++j)
                o[((first + r) * heads + head) * D + f * FRAGMENT + at.x + fragment_column(j)] =
                    state.out[f][i * 4 + j] / state.total[i];
    }
}

#define ATTENTION(D)                                                                               \
    [[kernel, max_total_threads_per_threadgroup(128)]] void mpp_attention_##D(                     \
        device half *q [[buffer(0)]], device half *k [[buffer(1)]], device half *v [[buffer(2)]],  \
        device float *o [[buffer(3)]], constant uint *p [[buffer(4)]],                             \
        uint g [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],     \
        uint lane [[thread_index_in_simdgroup]]) {                                                 \
        flash_attention<D, false>(q, k, v, o, nullptr, nullptr, nullptr, p, g, simd, lane);        \
    }
ATTENTION(64)
ATTENTION(128)
#undef ATTENTION

// mpp_attention_128 with the scores of INT8 queries and keys, one scale a token and head.
[[kernel, max_total_threads_per_threadgroup(128)]] void mpp_attention_int8_128(
    device half *q [[buffer(0)]], device half *k [[buffer(1)]], device half *v [[buffer(2)]],
    device float *o [[buffer(3)]], device const int8_t *q8 [[buffer(4)]],
    device const int8_t *k8 [[buffer(5)]], device const float *scales [[buffer(6)]],
    constant uint *p [[buffer(7)]], uint g [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    flash_attention<128, true>(q, k, v, o, q8, k8, scales, p, g, simd, lane);
}

// Block-sparse attention on the matrix units over heads of 128, the last step of Sol-Attn and VSA
// (see ops.metal): flash_attention over the tiles one query tile attends token by token, a
// threadgroup of four SIMD groups a query tile of at most 64 tokens. Sol-Attn centers the keys by
// their mean, which a row takes off its scores, and merges the pooled tail at the end. Gated VSA
// adds the gate times the coarse output.
//
// Parameters: tiles, heads, the method (0 for Sol-Attn, 1 for VSA), whether VSA is gated, and the
// scale in log2 units.
template <bool INT8>
void sparse_attention(device half *q, device half *k, device half *v, device float *o,
                      device const uint *starts, device const uint *lengths,
                      device const ushort *routes, device const uint *counts,
                      device const float *tails, device const float *key_mean,
                      device const float *gate, device const int8_t *q8, device const int8_t *k8,
                      device const float *scales, constant uint *p, uint g, uint simd, uint lane) {
    constexpr int D = 128, TAIL = D + 2;
    const int tiles = p[0], heads = p[1], gated = p[3];
    const bool sol = p[2] == 0;
    const float scale_log2 = as_type<float>(p[4]);
    const int tile = g % tiles, head = g / tiles, index = g;
    const int first = starts[tile] + simd * FRAGMENT;
    const int mine = int(lengths[tile]) - int(simd) * FRAGMENT;
    const short2 at = fragment_coord(lane);
    const int stride = heads * D;
    device const half *queries = q + (first * heads + head) * D;

    float offset[2] = {0, 0};
    for (short i = 0; i < 2 && sol; ++i) {
        const int r = at.y + i * 8;
        if (r >= mine)
            continue;
        for (int d = 0; d < D; ++d)
            offset[i] += float(queries[r * stride + d]) * key_mean[head * D + d];
        offset[i] *= scale_log2;
    }

    AttentionState<D> state;
    device const ushort *route = routes + index * tiles;
    for (int entry = 0; entry < int(counts[index]); ++entry) {
        const int base = starts[route[entry]], length = lengths[route[entry]];
        for (int block = 0; block < length; block += ATTENTION_KEYS) {
            const int at_key = ((base + block) * heads + head) * D;
            const int keys = min(ATTENTION_KEYS, length - block);
            auto seen = [](int, int) { return true; };
            auto attend = [&](thread const auto &scores) {
                if (keys == ATTENTION_KEYS && mine >= FRAGMENT)
                    state.template attend<false>(scores, mine, v + at_key, stride, keys, scale_log2,
                                                 offset, seen, at);
                else
                    state.template attend<true>(scores, mine, v + at_key, stride, keys, scale_log2,
                                                offset, seen, at);
            };
            if constexpr (INT8)
                attend(Int8Scores<D>{
                    q8 + (first * heads + head) * D, k8 + at_key, scales + first * heads + head,
                    scales + (int(p[5]) + base + block) * heads + head, stride, stride, heads});
            else
                attend(HalfScores<D>{queries, k + at_key, stride, stride});
        }
    }

    // Each row's weights for the routed keys and the pooled tail, worked out once. The loops
    // are unrolled so the output fragments stay in registers: indexing them at run time here puts
    // them in memory for the whole kernel, which made it far slower.
    device const float *tail = tails + index * TAIL;
    float routed[2], pooled[2];
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 2; ++i) {
        if (sol) {
            const float merged = max(state.reference[i], tail[D]);
            const float inverse = 1.0f / (state.total[i] * exp2(state.reference[i] - merged) +
                                          tail[D + 1] * exp2(tail[D] - merged));
            routed[i] = exp2(state.reference[i] - merged) * inverse;
            pooled[i] = exp2(tail[D] - merged) * inverse;
        } else {
            routed[i] = 1.0f / state.total[i];
            pooled[i] = gated ? 1.0f : 0.0f;
        }
    }
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 2; ++i) {
        const int r = at.y + i * 8;
        if (r >= mine)
            continue;
        _Pragma("clang loop unroll(full)") for (short f = 0; f < D / FRAGMENT; ++f) {
            _Pragma("clang loop unroll(full)") for (short j = 0; j < 4; ++j) {
                const int d = f * FRAGMENT + at.x + fragment_column(j),
                          out = ((first + r) * heads + head) * D + d;
                const float extra = sol ? tail[d] : gated ? gate[out] * tail[d] : 0.0f;
                o[out] = state.out[f][i * 4 + j] * routed[i] + extra * pooled[i];
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void mpp_sparse_attention_128(
    device half *q [[buffer(0)]], device half *k [[buffer(1)]], device half *v [[buffer(2)]],
    device float *o [[buffer(3)]], device const uint *starts [[buffer(4)]],
    device const uint *lengths [[buffer(5)]], device const ushort *routes [[buffer(6)]],
    device const uint *counts [[buffer(7)]], device const float *tails [[buffer(8)]],
    device const float *key_mean [[buffer(9)]], device const float *gate [[buffer(10)]],
    constant uint *p [[buffer(11)]], uint g [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    sparse_attention<false>(q, k, v, o, starts, lengths, routes, counts, tails, key_mean, gate,
                            nullptr, nullptr, nullptr, p, g, simd, lane);
}

// mpp_sparse_attention_128 with the scores of INT8 queries and keys, one scale a token and head.
// Parameters: those of mpp_sparse_attention_128, and the tokens.
[[kernel, max_total_threads_per_threadgroup(128)]] void mpp_sparse_attention_int8_128(
    device half *q [[buffer(0)]], device half *k [[buffer(1)]], device half *v [[buffer(2)]],
    device float *o [[buffer(3)]], device const uint *starts [[buffer(4)]],
    device const uint *lengths [[buffer(5)]], device const ushort *routes [[buffer(6)]],
    device const uint *counts [[buffer(7)]], device const float *tails [[buffer(8)]],
    device const float *key_mean [[buffer(9)]], device const float *gate [[buffer(10)]],
    device const int8_t *q8 [[buffer(11)]], device const int8_t *k8 [[buffer(12)]],
    device const float *scales [[buffer(13)]], constant uint *p [[buffer(14)]],
    uint g [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    sparse_attention<true>(q, k, v, o, starts, lengths, routes, counts, tails, key_mean, gate, q8,
                           k8, scales, p, g, simd, lane);
}
