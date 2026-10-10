#include <metal_stdlib>
using namespace metal;
float load_value(device const uchar *x, uint i, uint type) {
    if (type == 0)
        return ((device const float *)x)[i];
    if (type == 1)
        return float(((device const half *)x)[i]);
    if (type == 2)
        return as_type<float>(uint(((device const ushort *)x)[i]) << 16);
    return float(((device const char *)x)[i]);
}

kernel void fill_zero(device float *y [[buffer(0)]], constant uint *p [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = 0;
}

kernel void convert_float(device const uchar *x [[buffer(0)]], device float *y [[buffer(1)]],
                          constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = load_value(x, i + p[2], p[1]);
}

kernel void binary_op(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                      device float *y [[buffer(2)]], constant uint *p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = p[2] == 0 ? a[i] + b[i % p[1]] : a[i] * b[i % p[1]];
}

kernel void modulation(device const float *x [[buffer(0)]], device const float *m [[buffer(1)]],
                       device const uint *rows [[buffer(2)]],
                       device const float *delta [[buffer(3)]], device float *out [[buffer(4)]],
                       constant uint *p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint width = p[1], base = rows[i / width] * p[2] + i % width;
    float a = m[base + p[3] * width], b = m[base + p[4] * width];
    out[i] = p[5] ? x[i] + delta[i] * a : x[i] * (1 + b) + a;
}

// x + delta × scale, with `scale` a row of p[1] values: a residual stream taking a scaled block.
kernel void add_scaled(device const float *x [[buffer(0)]], device const float *delta [[buffer(1)]],
                       device const float *scale [[buffer(2)]], device float *y [[buffer(3)]],
                       constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = x[i] + delta[i] * scale[i % p[1]];
}

// erfc(x) for x ≥ 0 by the Chebyshev fit of Numerical Recipes, within 1.2e-7 of it relatively.
// Metal has no error function, and GELU needs one close to FP32's own.
float erfc_positive(float x) {
    const float t = 1 / (1 + 0.5f * x);
    const float series =
        -1.26551223f +
        t * (1.00002368f +
             t * (0.37409196f +
                  t * (0.09678418f +
                       t * (-0.18628806f +
                            t * (0.27886807f + t * (-1.13520398f +
                                                    t * (1.48851587f + t * (-0.82215223f +
                                                                            t * 0.17087277f))))))));
    return t * exp(-x * x + series);
}

// GELU with the exact error function: x · Φ(x). Each side takes the complement of the tail, so
// neither loses precision where Φ(x) is near 0 or 1.
float gelu_erf(float v) {
    const float tail = 0.5f * erfc_positive(abs(v) * 0.70710678118654752f);
    return v < 0 ? v * tail : v - v * tail;
}

// GELU with the tanh approximation. Metal's tanh is NaN for large arguments, so they are clamped
// where it is ±1 in FP32 already.
float gelu_tanh(float v) {
    const float inner = 0.79788456080286536f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1 + tanh(clamp(inner, -10.0f, 10.0f)));
}

// p[1] picks the operation: 0 scales by p[2], 1 is SiLU, 2 clamps to ±p[2], 3 is GELU with the
// tanh approximation and 4 GELU with the error function.
kernel void unary_op(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                     constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    float v = x[i], s = as_type<float>(p[2]);
    y[i] = p[1] == 0   ? v * s
           : p[1] == 1 ? v / (1 + exp(-v))
           : p[1] == 2 ? clamp(v, -s, s)
           : p[1] == 3 ? gelu_tanh(v)
           : p[1] == 4 ? gelu_erf(v)
                       : v;
}

kernel void slice_rows(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                       constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = x[(i / p[2] + p[3]) * p[1] + i % p[2] + p[4]];
}

kernel void copy_rows(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                      constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[(i / p[1] + p[3]) * p[2] + i % p[1] + p[4]] = x[i];
}

kernel void embedding(device const uchar *x [[buffer(0)]], device const uint *indices [[buffer(1)]],
                      device float *y [[buffer(2)]], constant uint *p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = load_value(x, indices[i / p[1]] * p[1] + i % p[1], p[2]);
}

// The values a thread keeps of a row it normalizes: rows of up to 256 × ROW_VALUES values, which
// both models' are, are read once rather than once a pass.
constant constexpr uint ROW_VALUES = 24;

// The sum over a threadgroup of 256 threads, which every thread gets.
float row_sum(float v, threadgroup float *scratch, uint tid) {
    v = simd_sum(v);
    if (tid % 32 == 0)
        scratch[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint i = 0; i < 8; ++i)
        sum += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sum;
}

// One row of normalize, and with MODULATE of normalize_modulate: `x` and `y` point at the row, and
// `table` at its row of the modulation table.
template <bool MODULATE>
void normalize_row(device const float *x, device const float *w, device float *y, uint width,
                   float epsilon, bool center, device const float *table, uint shift, uint scale,
                   uint tid, threadgroup float *scratch) {
    const bool kept = width <= 256 * ROW_VALUES;
    float v[ROW_VALUES], sum = 0;
    if (kept) {
        _Pragma("clang loop unroll(full)") for (uint i = 0; i < ROW_VALUES; ++i) {
            const uint c = tid + i * 256;
            v[i] = c < width ? x[c] : 0;
            sum += v[i];
        }
    } else {
        for (uint c = tid; c < width; c += 256)
            sum += x[c];
    }
    const float mean = center ? row_sum(sum, scratch, tid) / width : 0;

    float squares = 0;
    if (kept) {
        _Pragma("clang loop unroll(full)") for (uint i = 0; i < ROW_VALUES; ++i) {
            const float d = tid + i * 256 < width ? v[i] - mean : 0;
            squares += d * d;
        }
    } else {
        for (uint c = tid; c < width; c += 256)
            squares += (x[c] - mean) * (x[c] - mean);
    }
    const float inv = rsqrt(row_sum(squares, scratch, tid) / width + epsilon);

    auto write = [&](uint c, float value) {
        float normalized = (value - mean) * inv * w[c];
        if (MODULATE)
            normalized = normalized * (1 + table[scale * width + c]) + table[shift * width + c];
        y[c] = normalized;
    };
    if (kept) {
        _Pragma("clang loop unroll(full)") for (uint i = 0; i < ROW_VALUES;
                                                ++i) if (tid + i * 256 < width)
            write(tid + i * 256, v[i]);
    } else {
        for (uint c = tid; c < width; c += 256)
            write(c, x[c]);
    }
}

kernel void normalize(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                      device float *y [[buffer(2)]], constant uint *p [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    const uint width = p[0];
    normalize_row<false>(x + row * width, w, y + row * width, width, as_type<float>(p[1]), p[2], x,
                         0, 0, lane, scratch);
}

// normalize without centering, followed by modulation's x × (1 + scale) + shift, in one pass over
// a row. The row's scale and shift are chunks p[4] and p[3] of its row of the modulation table.
kernel void normalize_modulate(device const float *x [[buffer(0)]],
                               device const float *w [[buffer(1)]],
                               device const float *m [[buffer(2)]],
                               device const uint *rows [[buffer(3)]], device float *y [[buffer(4)]],
                               constant uint *p [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    const uint width = p[0];
    normalize_row<true>(x + row * width, w, y + row * width, width, as_type<float>(p[1]), false,
                        m + rows[row] * p[2], p[3], p[4], lane, scratch);
}

// The maximum over a threadgroup of 256 threads, which every thread gets.
float row_max(float v, threadgroup float *scratch, uint tid) {
    v = simd_max(v);
    if (tid % 32 == 0)
        scratch[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = scratch[0];
    for (uint i = 1; i < 8; ++i)
        maximum = max(maximum, scratch[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return maximum;
}

// softmax(p[1] · row) of rows of p[0] values, a threadgroup a row: the scores of an attention
// whose products run as matrix products.
kernel void softmax_rows(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                         constant uint *p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    const uint width = p[0];
    const float scale = as_type<float>(p[1]);
    device const float *source = x + size_t(row) * width;
    device float *destination = y + size_t(row) * width;
    float maximum = -INFINITY;
    for (uint c = lane; c < width; c += 256)
        maximum = max(maximum, source[c] * scale);
    maximum = row_max(maximum, scratch, lane);
    float sum = 0;
    for (uint c = lane; c < width; c += 256)
        sum += exp(source[c] * scale - maximum);
    const float inverse = 1 / row_sum(sum, scratch, lane);
    for (uint c = lane; c < width; c += 256)
        destination[c] = exp(source[c] * scale - maximum) * inverse;
}

// [p[1] rows, p[2] columns] to [columns, rows].
kernel void transpose(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                      constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = x[(i % p[1]) * p[2] + i / p[1]];
}

kernel void rotary(device const float *x [[buffer(0)]], device const float *angles [[buffer(1)]],
                   device float *y [[buffer(2)]], constant uint *p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint c = i % p[2], pairs = p[3];
    if (c >= 2 * pairs) {
        y[i] = x[i];
        return;
    }

    uint pair = c % pairs;
    float a = angles[(i / p[1]) * pairs + pair];
    uint start = i - c;
    float first = x[start + pair], second = x[start + pair + pairs];
    y[i] = c < pairs ? first * cos(a) - second * sin(a) : first * sin(a) + second * cos(a);
}

kernel void hadamard(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                     constant uint *p [[buffer(2)]], uint block [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    values[lane] = x[block * 256 + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ConvRot uses the regular radix-4 Hadamard, not the Sylvester radix-2 transform.

    for (uint s = 1; s < 256; s *= 4) {
        uint digit = (lane / s) % 4, base = lane - digit * s;
        float a = values[base], b = values[base + s], c = values[base + 2 * s],
              d = values[base + 3 * s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[lane] = digit == 0   ? a + b + c - d
                       : digit == 1 ? a + b - c + d
                       : digit == 2 ? a - b + c + d
                                    : -a + b + c + d;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    y[block * 256 + lane] = values[lane] * (1.0f / 16.0f);
}

// Eight queries share a K/V tile. Each SIMD group owns one query, keeping softmax
// and output accumulators in registers; only tile loads need threadgroup barriers.
template <uint D>
void tiled_attention(device const float *q, device const float *k, device const float *v,
                     device float *o, constant uint *p, uint group, uint tid, uint lane, uint simd,
                     threadgroup float *keys, threadgroup float *values) {
    constexpr uint TILE = 16, QUERIES = 8, LANES = 32;
    uint heads = p[1], kv_heads = p[2], dim = p[3], head = group % heads;
    uint first = (group / heads) * QUERIES, row = first + simd;
    uint kh = head / (heads / kv_heads);
    uint count = p[4] ? min(p[0], first + QUERIES) : p[0];
    float query[D / LANES], acc[D / LANES];

    for (uint c = 0; c < D / LANES; ++c) {
        uint col = lane + c * LANES;
        query[c] = row < p[5] && col < dim ? q[(row * heads + head) * dim + col] : 0;
        acc[c] = 0;
    }

    float maximum = -INFINITY, total = 0;
    for (uint base = 0; base < count; base += TILE) {
        uint length = min(TILE, count - base);
        for (uint i = tid; i < length * D; i += 256) {
            uint t = i / D, col = i % D;
            uint offset = ((base + t) * kv_heads + kh) * dim + col;
            keys[i] = col < dim ? k[offset] : 0;
            values[i] = col < dim ? v[offset] : 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < p[5]) {
            for (uint t = 0; t < length; ++t) {
                if (p[4] && base + t > row)
                    break;
                float dot = 0;
                for (uint c = 0; c < D / LANES; ++c)
                    dot += query[c] * keys[t * D + lane + c * LANES];
                float score = simd_sum(dot) * rsqrt(float(dim));
                float next = max(maximum, score), correction = exp(maximum - next);
                float probability = exp(score - next);
                for (uint c = 0; c < D / LANES; ++c)
                    acc[c] = acc[c] * correction + probability * values[t * D + lane + c * LANES];
                total = total * correction + probability;
                maximum = next;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < p[5]) {
        for (uint c = 0; c < D / LANES; ++c) {
            uint col = lane + c * LANES;
            if (col < dim)
                o[(row * heads + head) * dim + col] = acc[c] / total;
        }
    }
}
#define ATTENTION(D)                                                                               \
    kernel void attention_##D(                                                                     \
        device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],                  \
        device const float *v [[buffer(2)]], device float *o [[buffer(3)]],                        \
        constant uint *p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],               \
        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],         \
        uint simd [[simdgroup_index_in_threadgroup]]) {                                            \
        threadgroup float keys[16 * D], values[16 * D];                                            \
        tiled_attention<D>(q, k, v, o, p, group, tid, lane, simd, keys, values);                   \
    }
ATTENTION(32)
ATTENTION(64)
ATTENTION(128)
ATTENTION(256)
#undef ATTENTION
// im2col rows of sequences of p[1] samples, p[2] channels each, for a kernel of p[3] taps p[4]
// apart that steps p[5] samples a row over p[6] samples of zero padding, giving p[7] rows a
// sequence. The rows start at output row p[8], so that a long signal's go a slice at a time.
kernel void audio_columns(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                          constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint channels = p[2], kernel_size = p[3], row = i / (channels * kernel_size) + p[8],
         tap = i / channels % kernel_size;
    int source = int(row % p[7] * p[5] + tap * p[4]) - int(p[6]);
    y[i] = source >= 0 && source < int(p[1])
               ? x[((row / p[7]) * p[1] + source) * channels + i % channels]
               : 0;
}

kernel void audio_reorder_weight(device const uchar *x [[buffer(0)]], device float *y [[buffer(1)]],
                                 constant uint *p [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint inputs = p[1], outputs = p[2], kernel_size = p[3], input = i % inputs;
    uint tap = p[4] ? i / (inputs * outputs) : i / inputs % kernel_size;
    uint output = p[4] ? i / inputs % outputs : i / (inputs * kernel_size);
    uint source = p[4] ? (input * outputs + output) * kernel_size + tap
                       : (output * inputs + input) * kernel_size + tap;
    y[i] = load_value(x, source, p[5]);
}

kernel void audio_transpose(device const float *x [[buffer(0)]],
                            device const float *bias [[buffer(1)]], device float *y [[buffer(2)]],
                            constant uint *p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint channels = p[3], c = i % channels, t = i / channels % p[2],
         sequence = i / (channels * p[2]);
    int padding = int(p[4] - p[5]) / 2;
    float sum = bias[c];

    for (int tap = (int(t) + padding) % int(p[5]); tap < int(p[4]); tap += p[5]) {
        int shifted = int(t) + padding - tap;
        if (shifted < 0)
            break;
        uint source = uint(shifted) / p[5];
        if (source < p[1])
            sum += x[((sequence * p[1] + source) * p[4] + tap) * channels + c];
    }

    y[i] = sum;
}

kernel void audio_snake(device const float *x [[buffer(0)]],
                        device const float *parameters [[buffer(1)]],
                        device const float *up [[buffer(2)]],
                        device const float *down [[buffer(3)]], device float *y [[buffer(4)]],
                        constant uint *p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    int length = p[1], channels = p[2], channel = i % channels, time = i / channels % length,
        sequence = i / (channels * length);
    float alpha = parameters[channel], inv_beta = parameters[channels + channel], sum = 0;
    for (int tap = 0; tap < 12; ++tap) {
        int position = clamp(2 * time + tap - 5, 0, 2 * length - 1), shifted = position + 15;
        float value = 0;
        for (int j = shifted % 2; j < 12; j += 2) {
            int source = clamp((shifted - j) / 2 - 5, 0, length - 1);
            value += up[j] * x[(sequence * length + source) * channels + channel];
        }

        value *= 2;
        float sine = sin(alpha * value);
        sum += down[tap] * (value + sine * sine * inv_beta);
    }

    y[i] = sum;
}

// Snake with one parameter per channel of p[1], which the audio encoder uses as both α and β:
// x + sin²(αx) / (α + 1e-9).
kernel void audio_encoder_snake(device const float *x [[buffer(0)]],
                                device const float *alpha [[buffer(1)]],
                                device float *y [[buffer(2)]], constant uint *p [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    float a = alpha[i % p[1]], v = x[i], sine = sin(a * v);
    y[i] = v + sine * sine / (a + 1e-9f);
}

// The mean of the p[1] heads of p[2] values in a row of attention output, then average pooling
// of the head's values down to p[3] groups, bounded as adaptive pooling bounds them.
kernel void audio_pool_heads(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                             constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint heads = p[1], dim = p[2], outputs = p[3], output = i % outputs, row = i / outputs;
    uint start = output * dim / outputs, end = (output * dim + dim + outputs - 1) / outputs;
    float sum = 0;
    for (uint head = 0; head < heads; ++head)
        for (uint c = start; c < end; ++c)
            sum += x[(row * heads + head) * dim + c];
    y[i] = sum / float(heads * (end - start));
}

// GeGLU: the gate through GELU with the tanh approximation, times the value.
kernel void audio_geglu(device const float *gate [[buffer(0)]],
                        device const float *value [[buffer(1)]], device float *y [[buffer(2)]],
                        constant uint *p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    float x = gate[i], inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    y[i] = 0.5f * x * (1 + tanh(inner)) * value[i];
}

kernel void video_unpatch(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                          constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint w = p[3], h = p[2], f = p[1];
    uint col = i % w, row = i / w % h, t = i / (w * h) % f, c = i / (w * h * f);
    uint token = ((t / 4) * (h / 16) + row / 16) * (w / 16) + col / 16;
    uint feature = c * 1024 + t % 4 * 256 + row % 16 * 16 + col % 16;
    y[i] = x[token * 3072 + feature];
}

// Write only the region owned by this tile. Neighbors are the original, unblended
// tiles, preserving the reference's vertical-then-horizontal blend at intersections.
kernel void video_blend_tile(device const float *tile [[buffer(0)]],
                             device const float *above [[buffer(1)]],
                             device const float *left [[buffer(2)]],
                             device float *out [[buffer(3)]], constant uint *p [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint w = p[1], h = p[2], th = p[3], tw = p[4], kh = p[7], kw = p[8];
    uint x = i % kw, y = i / kw % kh, ct = i / (kw * kh);
    float value = tile[(ct * th + y) * tw + x];

    if (y < p[9]) {
        float b = float(y) / float(p[9]);
        value = above[(ct * th + th - p[9] + y) * tw + x] * (1 - b) + value * b;
    }

    if (x < p[10]) {
        float b = float(x) / float(p[10]);
        value = left[(ct * th + y) * tw + tw - p[10] + x] * (1 - b) + value * b;
    }

    out[(ct * h + p[5] + y) * w + p[6] + x] = value;
}

kernel void video_write_frames(device const float *canvas [[buffer(0)]],
                               device const float *tail [[buffer(1)]],
                               device float *out [[buffer(2)]], constant uint *p [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint plane = p[1], f = i / plane % p[5], c = i / (plane * p[5]), pixel = i % plane;
    float value = canvas[(c * p[2] + p[4] + f) * plane + pixel];
    if (p[6] && f < 5) {
        float b = float(f) / 5;
        value = tail[(c * 5 + f) * plane + pixel] * (1 - b) + value * b;
    }

    const float scale[3] = {0.229, 0.224, 0.225}, mean[3] = {0.485, 0.456, 0.406};
    out[(c * p[3] + p[7] + f) * plane + pixel] = clamp(value * scale[c] + mean[c], 0.0f, 1.0f);
}

// One group per activation row. Normalizing before conversion avoids FP16 overflow
// and gives INT8 a symmetric per-token scale. All-zero rows use unit scale.
kernel void pack_linear_input(device const float *x [[buffer(0)]],
                              device uchar *packed [[buffer(1)]],
                              device float *scales [[buffer(2)]], constant uint *p [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float maxima[8];
    const uint cols = p[0];
    float maximum = 0;
    for (uint j = tid; j < cols; j += 256)
        maximum = max(maximum, abs(x[row * cols + j]));
    maximum = simd_max(maximum);
    if (lane == 0)
        maxima[sg] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = simd_max(lane < 8 ? maxima[lane] : 0.0f);
    const float scale = maximum == 0 ? 1.0f : maximum;
    if (tid == 0)
        scales[row] = p[1] ? scale / 127.0f : scale;
    for (uint j = tid; j < cols; j += 256) {
        const uint i = row * cols + j;
        const float normalized = x[i] / scale;
        if (p[1])
            ((device char *)packed)[i] = char(rint(clamp(normalized * 127.0f, -127.0f, 127.0f)));
        else
            ((device half *)packed)[i] = half(normalized);
    }
}

// SiLU of the first half of a row times its second half, the MLP's gate: rows of 2 × p[1] values
// in, of p[1] out.
kernel void swiglu(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                   constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    const uint width = p[1], at = i / width * 2 * width + i % width;
    const float gate = x[at];
    y[i] = gate / (1 + exp(-gate)) * x[at + width];
}

// swiglu of rows whose outputs alternate 16 gates and their 16 up values, as a SwiGLU layer's
// weights do once `interleave_swiglu` has put them in that order.
kernel void swiglu_interleaved(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                               constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    const uint width = p[1], c = i % width, at = i / width * 2 * width + c / 16 * 32 + c % 16;
    const float gate = x[at];
    y[i] = gate / (1 + exp(-gate)) * x[at + 16];
}

// The rows of a SwiGLU layer's gate half and up half, p[0] rows of p[1] bytes, reordered to
// alternate 16 gate rows and their 16 up rows. One thread a byte.
kernel void interleave_swiglu_rows(device const uchar *x [[buffer(0)]],
                                   device uchar *y [[buffer(1)]], constant uint *p [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    const uint rows = p[0], row_bytes = p[1];
    if (i >= rows * row_bytes)
        return;
    const uint row = i / row_bytes, pair = row / 32, k = row % 32;
    const uint source = (k < 16 ? 0 : rows / 2) + pair * 16 + k % 16;
    y[i] = x[source * row_bytes + i % row_bytes];
}

// A block's attention inputs from its qkv projection, rows of [3][heads][p[4]], or of
// [heads][3][p[4]] with p[5] = 1, for heads of up to 256: the queries and keys RMS-normalized per
// head by their weights, and with p[1] pairs of angles a token rotated in the first 2 × p[1]
// dimensions of each head. All three are written as FP16 for the matrix units, or as FP32. With
// p[6], the queries and keys are also written as INT8 with one scale a token and head, the
// queries' p[7] tokens of scales first and then the keys'. Eight SIMD groups a token, a head at a
// time.
kernel void
attention_inputs(device const float *qkv [[buffer(0)]], device const float *q_norm [[buffer(1)]],
                 device const float *k_norm [[buffer(2)]], device const float *angles [[buffer(3)]],
                 device uchar *q [[buffer(4)]], device uchar *k [[buffer(5)]],
                 device uchar *v [[buffer(6)]], device char *q8 [[buffer(7)]],
                 device char *k8 [[buffer(8)]], device float *scales [[buffer(9)]],
                 constant uint *p [[buffer(10)]], uint token [[threadgroup_position_in_grid]],
                 uint simd [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint MAX_DIM = 256, PARTS = MAX_DIM / 32;
    threadgroup float normalized[8][MAX_DIM];
    const uint heads = p[0], pairs = p[1], half_out = p[2], dim = p[4], inner = heads * dim;
    const float epsilon = as_type<float>(p[3]);
    for (uint h = simd; h < 3 * heads; h += 8) {
        const uint tensor = h / heads, head = h % heads;
        device const float *row = qkv + token * 3 * inner +
                                  (p[5] ? (head * 3 + tensor) * dim : tensor * inner + head * dim);
        float x[PARTS], squares = 0;
        for (uint c = 0; c < PARTS; ++c) {
            const uint d = lane + c * 32;
            x[c] = d < dim ? row[d] : 0;
            squares += x[c] * x[c];
        }
        if (tensor < 2) {
            const float inverse = rsqrt(simd_sum(squares) / dim + epsilon);
            device const float *weight = tensor == 0 ? q_norm : k_norm;
            for (uint c = 0; c < PARTS; ++c)
                if (lane + c * 32 < dim)
                    x[c] *= inverse * weight[lane + c * 32];
        }
        if (tensor < 2 && pairs) {
            for (uint c = 0; c < PARTS; ++c)
                normalized[simd][lane + c * 32] = x[c];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < PARTS; ++c) {
                const uint d = lane + c * 32, pair = d % pairs;
                if (d >= 2 * pairs)
                    continue;
                const float a = angles[token * pairs + pair];
                const float first = normalized[simd][pair], second = normalized[simd][pair + pairs];
                x[c] =
                    d < pairs ? first * cos(a) - second * sin(a) : first * sin(a) + second * cos(a);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (p[6] && tensor < 2) {
            float maximum = 0;
            for (uint c = 0; c < PARTS; ++c)
                maximum = max(maximum, abs(x[c]));
            const float scale = max(simd_max(maximum), 1e-30f) / 127.0f;
            if (lane == 0)
                scales[(tensor * p[7] + token) * heads + head] = scale;
            device char *out8 = tensor == 0 ? q8 : k8;
            for (uint c = 0; c < PARTS; ++c) {
                const uint d = lane + c * 32;
                if (d < dim)
                    out8[(token * heads + head) * dim + d] = char(rint(x[c] / scale));
            }
        }
        device uchar *out = tensor == 0 ? q : tensor == 1 ? k : v;
        for (uint c = 0; c < PARTS; ++c) {
            const uint d = lane + c * 32, i = (token * heads + head) * dim + d;
            if (d >= dim)
                continue;
            if (half_out)
                ((device half *)out)[i] = half(x[c]);
            else
                ((device float *)out)[i] = x[c];
        }
    }
}

// The radix-4 butterfly of the Hadamard kernel: the output of digit `digit` from the four inputs
// that differ only in that digit.
float radix4(float a, float b, float c, float d, uint digit) {
    return digit == 0   ? a + b + c - d
           : digit == 1 ? a + b - c + d
           : digit == 2 ? a - b + c + d
                        : -a + b + c + d;
}

// ConvRot's rotation of one 256-value block in a SIMD group's registers, eight values a lane at
// index lane + 32 j. The four digits of an index are its bit pairs: the first two are the lane's
// low bits, the third the lane's top bit with j's lowest, and the last j's top bits. The stages and
// their sums are the hadamard kernel's, in its order, so the results are the same. A lane holds one
// of the four values of each butterfly and takes the other three from the lanes whose digit differs
// from its own, rather than shuffling all four in: the shuffles, not memory, bounded the kernels.
void hadamard_registers(thread float *x, uint lane) {
    for (uint s = 1; s <= 4; s *= 4) {
        const uint digit = lane / s % 4;
        for (uint j = 0; j < 8; ++j) {
            // The lanes that differ in this digit, by how far their digit is from this lane's.
            const float own = x[j], one = simd_shuffle_xor(own, s),
                        two = simd_shuffle_xor(own, 2 * s), three = simd_shuffle_xor(own, 3 * s);
            const float a = digit == 0 ? own : digit == 1 ? one : digit == 2 ? two : three;
            const float b = digit == 1 ? own : digit == 0 ? one : digit == 3 ? two : three;
            const float c = digit == 2 ? own : digit == 3 ? one : digit == 0 ? two : three;
            const float d = digit == 3 ? own : digit == 2 ? one : digit == 1 ? two : three;
            x[j] = radix4(a, b, c, d, digit);
        }
    }
    const uint bit = lane / 16;
    for (uint j = 0; j < 8; j += 2) {
        const float first = simd_shuffle_xor(x[j], 16), second = simd_shuffle_xor(x[j + 1], 16);
        const float a = bit ? first : x[j], b = bit ? x[j] : first;
        const float c = bit ? second : x[j + 1], d = bit ? x[j + 1] : second;
        x[j] = radix4(a, b, c, d, bit);
        x[j + 1] = radix4(a, b, c, d, 2 + bit);
    }
    for (uint j = 0; j < 2; ++j) {
        const float a = x[j], b = x[j + 2], c = x[j + 4], d = x[j + 6];
        for (uint digit = 0; digit < 4; ++digit)
            x[j + 2 * digit] = radix4(a, b, c, d, digit);
    }
    for (uint j = 0; j < 8; ++j)
        x[j] *= 1.0f / 16.0f;
}

// A row's value at `c`, or with `gated` SwiGLU of its gate at `c` and its up half at `cols` + `c`,
// as the swiglu kernel computes it.
float gated_value(device const float *row, uint c, uint cols, uint gated) {
    if (!gated)
        return row[c];
    const float gate = row[c];
    return gate / (1 + exp(-gate)) * row[cols + c];
}

// hadamard and pack_linear_input in one pass over a row: the first pass rotates each block to find
// the row's maximum, and the second rotates it again to pack it rather than keeping the row. With
// p[2] = 1 the rows are an MLP's gate and up halves, 2 × p[0] values, and SwiGLU of them is what
// is rotated and packed, so the gated rows are never written out.
// The 256-value blocks of a row a SIMD group of rotate_pack_linear_input keeps in registers: rows
// of up to 14,336 values, which every DiT and VAE product reads, are read once rather than twice.
constant constexpr uint ROTATE_BLOCKS = 7;

// A value of block `k` packed as rotate_pack_linear_input packs it, at `i`.
void pack_value(device uchar *packed, uint i, float value, float scale, bool int8) {
    const float normalized = value / scale;
    if (int8)
        ((device char *)packed)[i] = char(rint(clamp(normalized * 127.0f, -127.0f, 127.0f)));
    else
        ((device half *)packed)[i] = half(normalized);
}

kernel void rotate_pack_linear_input(device const float *x [[buffer(0)]],
                                     device uchar *packed [[buffer(1)]],
                                     device float *scales [[buffer(2)]],
                                     constant uint *p [[buffer(3)]],
                                     uint row [[threadgroup_position_in_grid]],
                                     uint simd [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxima[8];
    const uint cols = p[0], blocks = cols / 256, gated = p[2];
    device const float *in = x + row * cols * (gated ? 2 : 1);
    const bool kept = blocks <= 8 * ROTATE_BLOCKS;
    float v[ROTATE_BLOCKS][8], maximum = 0;
    if (kept) {
        _Pragma("clang loop unroll(full)") for (uint k = 0; k < ROTATE_BLOCKS; ++k) {
            const uint block = simd + k * 8;
            if (block >= blocks)
                continue;
            _Pragma("clang loop unroll(full)") for (uint j = 0; j < 8; ++j) v[k][j] =
                gated_value(in, block * 256 + j * 32 + lane, cols, gated);
            hadamard_registers(v[k], lane);
            _Pragma("clang loop unroll(full)") for (uint j = 0; j < 8; ++j) maximum =
                max(maximum, abs(v[k][j]));
        }
    } else {
        for (uint block = simd; block < blocks; block += 8) {
            float w[8];
            for (uint j = 0; j < 8; ++j)
                w[j] = gated_value(in, block * 256 + j * 32 + lane, cols, gated);
            hadamard_registers(w, lane);
            for (uint j = 0; j < 8; ++j)
                maximum = max(maximum, abs(w[j]));
        }
    }
    maximum = simd_max(maximum);
    if (lane == 0)
        maxima[simd] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = simd_max(lane < 8 ? maxima[lane] : 0.0f);
    const float scale = maximum == 0 ? 1.0f : maximum;
    if (simd == 0 && lane == 0)
        scales[row] = p[1] ? scale / 127.0f : scale;
    if (kept) {
        _Pragma("clang loop unroll(full)") for (uint k = 0; k < ROTATE_BLOCKS; ++k) {
            const uint block = simd + k * 8;
            if (block >= blocks)
                continue;
            _Pragma("clang loop unroll(full)") for (uint j = 0; j < 8; ++j)
                pack_value(packed, row * cols + block * 256 + j * 32 + lane, v[k][j], scale, p[1]);
        }
    } else {
        for (uint block = simd; block < blocks; block += 8) {
            float w[8];
            for (uint j = 0; j < 8; ++j)
                w[j] = gated_value(in, block * 256 + j * 32 + lane, cols, gated);
            hadamard_registers(w, lane);
            for (uint j = 0; j < 8; ++j)
                pack_value(packed, row * cols + block * 256 + j * 32 + lane, w[j], scale, p[1]);
        }
    }
}

// The 256-value blocks of a row a SIMD group of add_norm_pack keeps: rows of up to 6,144 values.
constant constexpr uint PACK_BLOCKS = 3;

// A residual stream's next block input in one pass over a row, as add_gated, normalize_modulate and
// rotate_pack_linear_input would make it in three: with p[5], the row takes `delta` times its gate,
// chunk p[6] of its row of `gates`, and the new row is written to `hidden` with p[10]; it is then
// RMS-normalized by `w`, modulated by chunks p[4] (scale) and p[3] (shift) of its row of `m`,
// written to `normed` with p[9] for a LoRA's down projection, and rotated and packed as
// rotate_pack_linear_input packs it, INT8 with p[8] and FP16 otherwise. The row stays in the SIMD
// groups' registers, a group's blocks laid out as the Hadamard rotation wants them.
kernel void
add_norm_pack(device const float *x [[buffer(0)]], device const float *delta [[buffer(1)]],
              device const float *gates [[buffer(2)]], device const float *m [[buffer(3)]],
              device const uint *rows [[buffer(4)]], device const float *w [[buffer(5)]],
              device float *hidden [[buffer(6)]], device float *normed [[buffer(7)]],
              device uchar *packed [[buffer(8)]], device float *scales [[buffer(9)]],
              constant uint *p [[buffer(10)]], uint row [[threadgroup_position_in_grid]],
              uint tid [[thread_index_in_threadgroup]],
              uint simd [[simdgroup_index_in_threadgroup]],
              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[8];
    const uint width = p[0], blocks = width / 256;
    device const float *table = m + rows[row] * p[2];
    device const float *gate = gates + rows[row] * p[7] + p[6] * width;
    float v[PACK_BLOCKS][8], squares = 0;
    _Pragma("clang loop unroll(full)") for (uint k = 0; k < PACK_BLOCKS; ++k) {
        const uint block = simd + k * 8;
        if (block >= blocks)
            continue;
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < 8; ++j) {
            const uint c = block * 256 + j * 32 + lane, i = row * width + c;
            float value = x[i];
            if (p[5]) {
                value = value + delta[i] * gate[c];
                if (p[10])
                    hidden[i] = value;
            }
            v[k][j] = value;
            squares += value * value;
        }
    }
    const float inv = rsqrt(row_sum(squares, scratch, tid) / width + as_type<float>(p[1]));

    float maximum = 0;
    _Pragma("clang loop unroll(full)") for (uint k = 0; k < PACK_BLOCKS; ++k) {
        const uint block = simd + k * 8;
        if (block >= blocks)
            continue;
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < 8; ++j) {
            const uint c = block * 256 + j * 32 + lane;
            const float value = v[k][j] * inv * w[c];
            v[k][j] = value * (1 + table[p[4] * width + c]) + table[p[3] * width + c];
            if (p[9])
                normed[row * width + c] = v[k][j];
        }
        hadamard_registers(v[k], lane);
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < 8; ++j) maximum =
            max(maximum, abs(v[k][j]));
    }
    maximum = simd_max(maximum);
    if (lane == 0)
        scratch[simd] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = simd_max(lane < 8 ? scratch[lane] : 0.0f);
    const float scale = maximum == 0 ? 1.0f : maximum;
    if (tid == 0)
        scales[row] = p[8] ? scale / 127.0f : scale;
    _Pragma("clang loop unroll(full)") for (uint k = 0; k < PACK_BLOCKS; ++k) {
        const uint block = simd + k * 8;
        if (block >= blocks)
            continue;
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < 8; ++j) {
            const uint i = row * width + block * 256 + j * 32 + lane;
            const float normalized = v[k][j] / scale;
            if (p[8])
                ((device char *)packed)[i] =
                    char(rint(clamp(normalized * 127.0f, -127.0f, 127.0f)));
            else
                ((device half *)packed)[i] = half(normalized);
        }
    }
}

// The FP16 copy of an attention input that the tensor-product attention reads.
kernel void float_to_half(device const float *x [[buffer(0)]], device half *y [[buffer(1)]],
                          constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = half(x[i]);
}

// float_to_half of heads of p[1] values, each padded with zeros to p[2] values and multiplied by
// the float in p[3], for the attention on the matrix units, which takes only some head widths.
// p[0] is the output's length.
kernel void pad_heads_to_half(device const float *x [[buffer(0)]], device half *y [[buffer(1)]],
                              constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    const uint dim = p[1], padded = p[2], column = i % padded, head = i / padded;
    y[i] = column < dim ? half(x[head * dim + column] * as_type<float>(p[3])) : half(0);
}

// The first p[1] of every p[2] values: the heads of a padded attention's output at their width.
kernel void unpad_heads(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                        constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = x[i / p[1] * p[2] + i % p[1]];
}

// MPS comparison path: INT8 is exactly representable as FP16.
kernel void int8_to_half(device const char *x [[buffer(0)]], device half *y [[buffer(1)]],
                         constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = half(x[p[1] + i]);
}

// The MPS product's result, finished the way the MPP kernels finish theirs: the rows' scales,
// and by the flags the outputs' scales, a bias and a result to add.
kernel void finish_linear(device float *x [[buffer(0)]], device const float *scales [[buffer(1)]],
                          device const float *outputs [[buffer(2)]],
                          device const float *bias [[buffer(3)]],
                          device const float *addend [[buffer(4)]], constant uint *p [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    const uint n = i % p[1];
    float value = x[i] * scales[i / p[1]];
    if (p[2] & 1)
        value *= outputs[n];
    if (p[2] & 2)
        value += bias[n];
    if (p[2] & 4)
        value += addend[i];
    x[i] = value;
}

// Gathers the heads one rank owns out of [tokens][tensors][heads][dim] into
// [tokens][tensors][span][dim], the shape the attention already reads.
kernel void shard_pack(device const float *source [[buffer(0)]], device float *packed [[buffer(1)]],
                       constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint dim = p[1], span = p[2], tensors = p[3], heads = p[4];
    uint lane = i % dim, rest = i / dim;
    uint head = rest % span + p[5], above = rest / span;
    uint tensor = above % tensors, token = above / tensors;
    packed[i] = source[((token * tensors + tensor) * heads + head) * dim + lane];
}

// Scatters one rank's share of an attention output, [tokens][span][dim], into the rows this rank
// carries onward, [tokens][heads][dim], at head p[4]. The heads it does not own are left alone.
kernel void shard_unpack(device const float *part [[buffer(0)]], device float *output [[buffer(1)]],
                         constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint dim = p[1], span = p[2], heads = p[3];
    uint lane = i % dim, rest = i / dim;
    uint head = rest % span + p[4], token = rest / span;
    output[(token * heads + head) * dim + lane] = part[i];
}

// The exchange carries a block's attention tensors in bf16, which Metal has no type for, so they
// convert on the way out and back. Rounds to nearest, ties to even, and leaves a NaN a NaN, which
// is what mmh3_core::numeric does on the host.
kernel void to_bf16(device const float *x [[buffer(0)]], device ushort *y [[buffer(1)]],
                    constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint bits = as_type<uint>(x[i]);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u)
        y[p[1] + i] = ushort((bits >> 16) | 0x40u);
    else
        y[p[1] + i] = ushort((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

kernel void from_bf16(device const ushort *x [[buffer(0)]], device float *y [[buffer(1)]],
                      constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    y[i] = as_type<float>(uint(x[p[1] + i]) << 16);
}

// An input another rank rotated and quantized, back in FP32. Only the road that runs products in
// FP32 needs it; the packed roads take the INT8 values as they are.
kernel void dequantize_rows(device const char *q [[buffer(0)]],
                            device const float *scales [[buffer(1)]], device float *y [[buffer(2)]],
                            constant uint *p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    y[i] = float(q[i]) * scales[i / p[1]];
}

// Moves a run of bytes into a region at a byte offset, which is how a rank's own rows reach the
// memory its peers read.
kernel void copy_bytes(device const char *x [[buffer(0)]], device char *y [[buffer(1)]],
                       constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[p[2] + i] = x[p[1] + i];
}

// Block-sparse attention over heads of 128: Sol-Attn and VSA (see mmh3-core's dit/sparse.rs and
// dit/vsa.rs). The sequence is cut into tiles of at most 64 consecutive tokens, whose starts and
// lengths are tables of their own: 64-token blocks for Sol-Attn, and for VSA its tiles of the
// sequence in the order it runs it.
//
// 1. sparse_pool: per head and tile, the mean query and key and the summed value.
// 2. sparse_center_keys (Sol-Attn): per head, the mean and variance of the tile keys, which are
//    then centered.
// 3. sol_route or vsa_select: per head and query tile, the pooled scores against every tile and
//    the tiles it attends token by token, in ascending order. Sol-Attn keeps the pooled tail of
//    the other tiles as a softmax state: its maximum, its weighted sum and its weighted values.
//    Gated VSA keeps its coarse output, softmax(scores) · mean values.
// 4. attention_sparse_128, or mpp_sparse_attention_128 on the matrix units: attention over those
//    tiles, merged with the tail, or plus the gate times the coarse output.
//
// A query tile's row in the tables is head × tiles + tile, and its tail row holds 128 values and,
// for Sol-Attn, the tail's maximum and sum.
constant constexpr uint SPARSE_HEAD = 128, SPARSE_TAIL = SPARSE_HEAD + 2;
constant constexpr uint SPARSE_MAX_TILES = 4096;

// The sum or the maximum over a threadgroup of eight SIMD groups, which every thread gets.
template <bool MAXIMUM>
float group_reduce(float value, threadgroup float *scratch, uint simd, uint lane) {
    value = MAXIMUM ? simd_max(value) : simd_sum(value);
    if (lane == 0)
        scratch[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = scratch[0];
    for (uint i = 1; i < 8; ++i)
        result = MAXIMUM ? max(result, scratch[i]) : result + scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// Two heads a threadgroup, a thread a dimension. The inputs are FP32, or FP16 with p[2] = 1.
kernel void sparse_pool(device const uchar *q [[buffer(0)]], device const uchar *k [[buffer(1)]],
                        device const uchar *v [[buffer(2)]],
                        device const uint *starts [[buffer(3)]],
                        device const uint *lengths [[buffer(4)]],
                        device float *pooled_q [[buffer(5)]], device float *pooled_k [[buffer(6)]],
                        device float *pooled_v [[buffer(7)]], constant uint *p [[buffer(8)]],
                        uint g [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    const uint tiles = p[0], heads = p[1];
    const uint tile = g % tiles, head = g / tiles * 2 + tid / SPARSE_HEAD;
    const uint d = tid % SPARSE_HEAD;
    if (head >= heads)
        return;
    const uint start = starts[tile], rows = lengths[tile];
    float queries = 0, keys = 0, values = 0;
    for (uint r = 0; r < rows; ++r) {
        const uint i = ((start + r) * heads + head) * SPARSE_HEAD + d;
        queries += load_value(q, i, p[2]);
        keys += load_value(k, i, p[2]);
        values += load_value(v, i, p[2]);
    }
    const uint o = (head * tiles + tile) * SPARSE_HEAD + d;
    pooled_q[o] = queries / rows;
    pooled_k[o] = keys / rows;
    pooled_v[o] = values;
}

kernel void sparse_center_keys(device float *pooled_k [[buffer(0)]],
                               device float *key_mean [[buffer(1)]],
                               device float *key_variance [[buffer(2)]],
                               constant uint *p [[buffer(3)]],
                               uint g [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    const uint tiles = p[0], heads = p[1];
    const uint head = g * 2 + tid / SPARSE_HEAD, d = tid % SPARSE_HEAD;
    if (head >= heads)
        return;
    device float *keys = pooled_k + head * tiles * SPARSE_HEAD + d;
    float sum = 0;
    for (uint t = 0; t < tiles; ++t)
        sum += keys[t * SPARSE_HEAD];
    const float mean = sum / tiles;
    float squares = 0;
    for (uint t = 0; t < tiles; ++t) {
        const float centered = keys[t * SPARSE_HEAD] - mean;
        keys[t * SPARSE_HEAD] = centered;
        squares += centered * centered;
    }
    key_mean[head * SPARSE_HEAD + d] = mean;
    key_variance[head * SPARSE_HEAD + d] = squares / tiles;
}

// scores[t] = (query · pooled[t]) × scale for every tile, a SIMD group a tile at a time.
void pooled_scores(threadgroup const float *query, device const float *pooled, uint tiles,
                   float scale, threadgroup float *scores, uint simd, uint lane) {
    for (uint t = simd; t < tiles; t += 8) {
        device const float *row = pooled + t * SPARSE_HEAD;
        float partial = 0;
        for (uint c = 0; c < SPARSE_HEAD; c += 32)
            partial += query[lane + c] * row[lane + c];
        const float score = simd_sum(partial) * scale;
        if (lane == 0)
            scores[t] = score;
    }
}

// Σ weights[t] · values[t] for this thread's dimension, the two halves of the threadgroup taking
// every other tile. The first half answers.
float weighted_values(threadgroup const float *weights, device const float *values, uint tiles,
                      threadgroup float *halves, uint tid) {
    const uint d = tid % SPARSE_HEAD;
    float sum = 0;
    for (uint t = tid / SPARSE_HEAD; t < tiles; t += 2)
        sum += weights[t] * values[t * SPARSE_HEAD + d];
    if (tid >= SPARSE_HEAD)
        halves[d] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sum + halves[d];
}

// A threadgroup a query tile of one head. The scores are in log2 units.
kernel void
sol_route(device const float *pooled_q [[buffer(0)]], device const float *pooled_k [[buffer(1)]],
          device const float *pooled_v [[buffer(2)]],
          device const float *key_variance [[buffer(3)]], device const uint *lengths [[buffer(4)]],
          device ushort *routes [[buffer(5)]], device uint *counts [[buffer(6)]],
          device float *tails [[buffer(7)]], device atomic_uint *routed [[buffer(8)]],
          constant uint *p [[buffer(9)]], uint g [[threadgroup_position_in_grid]],
          uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[SPARSE_MAX_TILES];
    threadgroup bool exact[SPARSE_MAX_TILES];
    threadgroup float centroid[SPARSE_HEAD], halves[SPARSE_HEAD], scratch[8];
    const uint tiles = p[0];
    const float tau = as_type<float>(p[2]), log2_scale = as_type<float>(p[3]);
    const uint query = g % tiles, head = g / tiles, row = g;
    const bool dense_query = query >= p[6] && query < p[7];

    float spread = 0;
    if (tid < SPARSE_HEAD) {
        const float c = pooled_q[row * SPARSE_HEAD + tid];
        centroid[tid] = c;
        spread = c * c * key_variance[head * SPARSE_HEAD + tid];
    }
    spread = group_reduce<false>(spread, scratch, simd, lane);
    const float threshold = tau * sqrt(spread * log2_scale * log2_scale + 1e-6f);
    pooled_scores(centroid, pooled_k + head * tiles * SPARSE_HEAD, tiles, log2_scale, scores, simd,
                  lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maximum = -FLT_MAX;
    for (uint t = tid; t < tiles; t += 256) {
        const bool flag = dense_query || scores[t] > threshold ||
                          max(query, t) - min(query, t) <= 1 || (t >= p[4] && t < p[5]);
        exact[t] = flag;
        if (!flag)
            maximum = max(maximum, scores[t]);
    }
    maximum = group_reduce<true>(maximum, scratch, simd, lane);
    float sum = 0;
    for (uint t = tid; t < tiles; t += 256) {
        const float weight = exact[t] ? 0.0f : exp2(scores[t] - maximum);
        scores[t] = weight;
        sum += weight * lengths[t];
    }
    sum = group_reduce<false>(sum, scratch, simd, lane);

    const float tail =
        weighted_values(scores, pooled_v + head * tiles * SPARSE_HEAD, tiles, halves, tid);
    device float *out = tails + row * SPARSE_TAIL;
    if (tid < SPARSE_HEAD)
        out[tid] = tail;
    if (tid == 0) {
        out[SPARSE_HEAD] = maximum;
        out[SPARSE_HEAD + 1] = sum;
    }

    if (simd == 0) {
        device ushort *route = routes + row * tiles;
        uint count = 0;
        for (uint first = 0; first < tiles; first += 32) {
            const uint t = first + lane;
            const uint flag = t < tiles && exact[t];
            const uint before = simd_prefix_exclusive_sum(flag);
            if (flag)
                route[count + before] = ushort(t);
            count += simd_sum(flag);
        }
        if (lane == 0) {
            counts[row] = count;
            atomic_fetch_add_explicit(routed, count, memory_order_relaxed);
        }
    }
}

// A float's bits as an unsigned integer of the same order.
uint ordered_bits(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

// A threadgroup a query tile of one head. A video query tile keeps the `kept` video tiles of the
// highest scores, ties going to the lower tile, besides every tile before the video.
kernel void
vsa_select(device const float *pooled_q [[buffer(0)]], device const float *pooled_k [[buffer(1)]],
           device const float *pooled_v [[buffer(2)]], device const uint *lengths [[buffer(3)]],
           device ushort *routes [[buffer(4)]], device uint *counts [[buffer(5)]],
           device float *coarse [[buffer(6)]], constant uint *p [[buffer(7)]],
           uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
           uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[SPARSE_MAX_TILES];
    threadgroup float pooled[SPARSE_HEAD], halves[SPARSE_HEAD], scratch[8];
    const uint tiles = p[0], prefix = p[2], kept = p[3], gated = p[5];
    const float scale = as_type<float>(p[4]);
    const uint query = g % tiles, head = g / tiles, row = g;

    if (tid < SPARSE_HEAD)
        pooled[tid] = pooled_q[row * SPARSE_HEAD + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pooled_scores(pooled, pooled_k + head * tiles * SPARSE_HEAD, tiles, scale, scores, simd, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The kept-th largest video score, a bit at a time from the top: `threshold` gathers its bits,
    // and `ties` ends as how many video scores equal to it are kept.
    const bool dense = query < prefix || kept >= tiles - prefix;
    uint threshold = 0, ties = kept;
    for (int bit = 31; bit >= 0 && !dense; --bit) {
        const uint high = bit == 31 ? 0u : ~0u << (bit + 1);
        float ones = 0;
        for (uint t = prefix + tid; t < tiles; t += 256) {
            const uint bits = ordered_bits(scores[t]);
            ones += (bits & high) == threshold && (bits >> bit) & 1u;
        }
        const uint total = uint(group_reduce<false>(ones, scratch, simd, lane));
        if (total >= ties)
            threshold |= 1u << bit;
        else
            ties -= total;
    }

    if (simd == 0) {
        device ushort *route = routes + row * tiles;
        uint count = 0, seen = 0;
        for (uint first = 0; first < tiles; first += 32) {
            const uint t = first + lane;
            const bool candidate = !dense && t >= prefix && t < tiles;
            const uint bits = candidate ? ordered_bits(scores[t]) : 0;
            const uint tie = candidate && bits == threshold;
            const uint rank = seen + simd_prefix_exclusive_sum(tie);
            seen += simd_sum(tie);
            const uint flag = candidate ? bits > threshold || (tie && rank < ties) : t < tiles;
            const uint before = simd_prefix_exclusive_sum(flag);
            if (flag)
                route[count + before] = ushort(t);
            count += simd_sum(flag);
        }
        if (lane == 0)
            counts[row] = count;
    }
    if (!gated)
        return;

    // The weights divide by the tile lengths, since the pooled values are sums.
    float maximum = -FLT_MAX;
    for (uint t = tid; t < tiles; t += 256)
        maximum = max(maximum, scores[t]);
    maximum = group_reduce<true>(maximum, scratch, simd, lane);
    float sum = 0;
    for (uint t = tid; t < tiles; t += 256) {
        const float weight = exp(scores[t] - maximum);
        scores[t] = weight / lengths[t];
        sum += weight;
    }
    sum = group_reduce<false>(sum, scratch, simd, lane);
    const float value =
        weighted_values(scores, pooled_v + head * tiles * SPARSE_HEAD, tiles, halves, tid);
    if (tid < SPARSE_HEAD)
        coarse[row * SPARSE_TAIL + tid] = value / sum;
}

// The attention of step 4 in FP32. Eight queries of one tile share each key tile, a SIMD group a
// query, as tiled_attention does. Parameters: tiles, heads, the method (0 for Sol-Attn, 1 for
// VSA), whether VSA is gated, and the scale in log2 units.
kernel void attention_sparse_128(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *o [[buffer(3)]],
    device const uint *starts [[buffer(4)]], device const uint *lengths [[buffer(5)]],
    device const ushort *routes [[buffer(6)]], device const uint *counts [[buffer(7)]],
    device const float *tails [[buffer(8)]], device const float *key_mean [[buffer(9)]],
    device const float *gate [[buffer(10)]], constant uint *p [[buffer(11)]],
    uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TILE = 16, D = SPARSE_HEAD, PARTS = D / 32;
    threadgroup float keys[TILE * D], values[TILE * D];
    const uint tiles = p[0], heads = p[1], sol = p[2] == 0, gated = p[3];
    const float scale_log2 = as_type<float>(p[4]);
    const uint head = g / (tiles * 8), tile = g / 8 % tiles, row = head * tiles + tile;
    const uint r = g % 8 * 8 + simd, token = starts[tile] + r;
    const bool live = r < lengths[tile];

    float query[PARTS], acc[PARTS], offset = 0;
    for (uint c = 0; c < PARTS; ++c) {
        query[c] = live ? q[(token * heads + head) * D + lane + c * 32] : 0;
        acc[c] = 0;
        if (sol)
            offset += query[c] * key_mean[head * D + lane + c * 32];
    }
    offset = simd_sum(offset) * scale_log2;

    float maximum = -INFINITY, total = 0;
    device const ushort *route = routes + row * tiles;
    for (uint entry = 0; entry < counts[row]; ++entry) {
        const uint first = starts[route[entry]], rows = lengths[route[entry]];
        for (uint base = 0; base < rows; base += TILE) {
            const uint length = min(TILE, rows - base);
            for (uint i = tid; i < length * D; i += 256) {
                const uint at = ((first + base + i / D) * heads + head) * D + i % D;
                keys[i] = k[at];
                values[i] = v[at];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (live) {
                for (uint t = 0; t < length; ++t) {
                    float dot = 0;
                    for (uint c = 0; c < PARTS; ++c)
                        dot += query[c] * keys[t * D + lane + c * 32];
                    const float score = simd_sum(dot) * scale_log2 - offset;
                    const float next = max(maximum, score), correction = exp2(maximum - next);
                    const float probability = exp2(score - next);
                    for (uint c = 0; c < PARTS; ++c)
                        acc[c] = acc[c] * correction + probability * values[t * D + lane + c * 32];
                    total = total * correction + probability;
                    maximum = next;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (!live)
        return;

    device const float *tail = tails + row * SPARSE_TAIL;
    for (uint c = 0; c < PARTS; ++c) {
        const uint d = lane + c * 32, i = (token * heads + head) * D + d;
        if (sol) {
            const float merged = max(maximum, tail[D]);
            const float routed = exp2(maximum - merged), pooled = exp2(tail[D] - merged);
            o[i] = (acc[c] * routed + tail[d] * pooled) / (total * routed + tail[D + 1] * pooled);
        } else {
            o[i] = acc[c] / total + (gated ? gate[i] * tail[d] : 0.0f);
        }
    }
}

// Veda (see mmh3-core's dit/veda.rs) attends with each group of a layer's heads that share a tile
// shape over a copy of their rows in the shape's tile order, where every tile is consecutive and
// cut into two tiles of at most 64 for the attention of steps 4 above.
//
// 1. veda_gather: a group's rows in tile order, and after the attention veda_scatter its output
//    back in the packed order.
// 2. veda_pool: per head and video tile, the mean, the maximum and the minimum of the queries and
//    of the keys.
// 3. veda_project: the predictor's view of a tile, pooled · P + mean, with P the FP8 projection of
//    the layer and head times its scale, rounded to BF16 as the predictor runs it.
// 4. veda_select: per head and query tile of 128, the scores against every video tile, and the
//    tiles of 64 it attends as routes for attention_sparse_128 or mpp_sparse_attention_128, the
//    same for the two halves of the tile.
constant constexpr uint VEDA_POOLED = 3 * SPARSE_HEAD;
// Video tiles one threadgroup of veda_project projects, sharing the reads of P.
constant constexpr uint VEDA_PROJECT_TILES = 8;

// Words of a (row, head) of the input, from row p[4] + order[row] and head group[member], into
// row p[5] + row and member `member` of the output.
// p: rows, heads, members, words, input row offset, output row offset.
kernel void veda_gather(device const uint *x [[buffer(0)]], device uint *y [[buffer(1)]],
                        device const uint *order [[buffer(2)]],
                        device const uint *group [[buffer(3)]], constant uint *p [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    const uint rows = p[0], heads = p[1], members = p[2], words = p[3];
    if (i >= rows * members * words)
        return;
    const uint word = i % words, member = i / words % members, row = i / (words * members);
    y[((p[5] + row) * members + member) * words + word] =
        x[((p[4] + order[row]) * heads + group[member]) * words + word];
}

// The inverse of veda_gather for the attention's FP32 output of heads of 128.
// p: rows, heads, members.
kernel void veda_scatter(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                         device const uint *order [[buffer(2)]],
                         device const uint *group [[buffer(3)]], constant uint *p [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
    const uint rows = p[0], heads = p[1], members = p[2];
    if (i >= rows * members * SPARSE_HEAD)
        return;
    const uint d = i % SPARSE_HEAD, member = i / SPARSE_HEAD % members;
    const uint row = i / (SPARSE_HEAD * members);
    y[(order[row] * heads + group[member]) * SPARSE_HEAD + d] = x[i];
}

// Two heads a threadgroup, a thread a dimension. An empty tile pools to zeros.
// p: video tiles, members, whether the inputs are FP16.
kernel void veda_pool(device const uchar *q [[buffer(0)]], device const uchar *k [[buffer(1)]],
                      device const uint *starts [[buffer(2)]],
                      device const uint *lengths [[buffer(3)]],
                      device float *pooled_q [[buffer(4)]], device float *pooled_k [[buffer(5)]],
                      constant uint *p [[buffer(6)]], uint g [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    const uint tiles = p[0], heads = p[1];
    const uint tile = g % tiles, head = g / tiles * 2 + tid / SPARSE_HEAD;
    const uint d = tid % SPARSE_HEAD;
    if (head >= heads)
        return;
    const uint start = starts[tile], rows = lengths[tile];
    float query_sum = 0, query_max = -INFINITY, query_min = INFINITY;
    float key_sum = 0, key_max = -INFINITY, key_min = INFINITY;
    for (uint r = 0; r < rows; ++r) {
        const uint i = ((start + r) * heads + head) * SPARSE_HEAD + d;
        const float query = load_value(q, i, p[2]), key = load_value(k, i, p[2]);
        query_sum += query;
        query_max = max(query_max, query);
        query_min = min(query_min, query);
        key_sum += key;
        key_max = max(key_max, key);
        key_min = min(key_min, key);
    }
    const uint o = (head * tiles + tile) * VEDA_POOLED + d;
    const bool empty = rows == 0;
    pooled_q[o] = empty ? 0 : query_sum / rows;
    pooled_q[o + SPARSE_HEAD] = empty ? 0 : query_max;
    pooled_q[o + 2 * SPARSE_HEAD] = empty ? 0 : query_min;
    pooled_k[o] = empty ? 0 : key_sum / rows;
    pooled_k[o + SPARSE_HEAD] = empty ? 0 : key_max;
    pooled_k[o + 2 * SPARSE_HEAD] = empty ? 0 : key_min;
}

float fp8_e4m3(uchar bits) {
    const int exponent = (bits >> 3) & 0xF;
    const float mantissa = float(bits & 7) * 0.125f;
    const float value = exponent == 0 ? mantissa * 0.015625f : ldexp(1.0f + mantissa, exponent - 7);
    return (bits & 0x80) ? -value : value;
}

// Rounds to nearest BF16, ties to even.
float round_bf16(float value) {
    const uint bits = as_type<uint>(value);
    return as_type<float>((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000u);
}

// A threadgroup VEDA_PROJECT_TILES tiles of one head, a thread a dimension of half of them.
// `weights` holds every layer's P, [layers, heads, 384, 128], and `scales` their [layers, heads].
// p: video tiles, members, layer, heads.
kernel void veda_project(device const float *pooled [[buffer(0)]],
                         device const uchar *weights [[buffer(1)]],
                         device const float *scales [[buffer(2)]],
                         device const uint *group [[buffer(3)]], device float *out [[buffer(4)]],
                         constant uint *p [[buffer(5)]], uint g [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint HALF = VEDA_PROJECT_TILES / 2;
    threadgroup float tiles[VEDA_PROJECT_TILES * VEDA_POOLED];
    const uint count = p[0], blocks = (count + VEDA_PROJECT_TILES - 1) / VEDA_PROJECT_TILES;
    const uint member = g / blocks, first = g % blocks * VEDA_PROJECT_TILES;
    const uint head = p[2] * p[3] + group[member];
    for (uint i = tid; i < VEDA_PROJECT_TILES * VEDA_POOLED; i += 256) {
        const uint tile = first + i / VEDA_POOLED;
        tiles[i] =
            tile < count ? pooled[(member * count + tile) * VEDA_POOLED + i % VEDA_POOLED] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d = tid % SPARSE_HEAD, half_index = tid / SPARSE_HEAD;
    threadgroup const float *mine = tiles + half_index * HALF * VEDA_POOLED;
    device const uchar *column = weights + head * VEDA_POOLED * SPARSE_HEAD + d;
    const float scale = scales[head];
    float sums[HALF] = {};
    for (uint row = 0; row < VEDA_POOLED; ++row) {
        const float weight = round_bf16(fp8_e4m3(column[row * SPARSE_HEAD]) * scale);
        for (uint t = 0; t < HALF; ++t)
            sums[t] += mine[t * VEDA_POOLED + row] * weight;
    }
    for (uint t = 0; t < HALF; ++t) {
        const uint tile = first + half_index * HALF + t;
        if (tile < count)
            out[(member * count + tile) * SPARSE_HEAD + d] = sums[t] + mine[t * VEDA_POOLED + d];
    }
}

// Rows of tile `tile` of 64 in Veda's tiles of 128.
uint veda_rows(device const uint *lengths, uint tile) {
    const uint length = lengths[tile / 2];
    return tile % 2 == 0 ? min(length, 64u) : (length > 64 ? length - 64 : 0);
}

// A threadgroup a query tile of 128 of one head. A video query tile keeps, of each column block,
// the allowed number of tiles of the highest scores, its own tile first and ties going to the
// lower tile, and never an empty tile; then every global tile. A global query tile keeps all.
// `allowed` holds each video query tile's count for the reference block and the generated block,
// and `kept` gathers how many video tiles the query tiles kept.
// p: tiles, video tiles, reference tiles, scale.
kernel void
veda_select(device const float *queries [[buffer(0)]], device const float *keys [[buffer(1)]],
            device const uint *lengths [[buffer(2)]], device const uint *allowed [[buffer(3)]],
            device ushort *routes [[buffer(4)]], device uint *counts [[buffer(5)]],
            device atomic_uint *kept [[buffer(6)]], constant uint *p [[buffer(7)]],
            uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
            uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[SPARSE_MAX_TILES];
    threadgroup bool chosen[SPARSE_MAX_TILES];
    threadgroup float pooled[SPARSE_HEAD], scratch[8];
    const uint tiles = p[0], video = p[1], reference = p[2];
    const float scale = as_type<float>(p[3]);
    const uint query = g % tiles, head = g / tiles, small = 2 * tiles;
    const uint row = head * small + 2 * query;
    const bool global = query >= video;

    if (!global) {
        if (tid < SPARSE_HEAD)
            pooled[tid] = queries[(head * video + query) * SPARSE_HEAD + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pooled_scores(pooled, keys + head * video * SPARSE_HEAD, video, scale, scores, simd, lane);
        for (uint t = tid; t < video; t += 256)
            chosen[t] = false;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint block = reference == 0 ? 1 : 0; block < 2; ++block) {
            const uint start = block == 0 ? 0 : reference, end = block == 0 ? reference : video;
            // The kept-th largest candidate score, a bit at a time from the top, as vsa_select
            // finds it.
            float candidates = 0;
            for (uint t = start + tid; t < end; t += 256)
                candidates += t == query || lengths[t] > 0;
            const uint total = uint(group_reduce<false>(candidates, scratch, simd, lane));
            const uint count = min(allowed[query * 2 + block], total);
            uint threshold = 0, ties = count;
            for (int bit = 31; bit >= 0 && count > 0; --bit) {
                const uint high = bit == 31 ? 0u : ~0u << (bit + 1);
                float ones = 0;
                for (uint t = start + tid; t < end; t += 256) {
                    if (t != query && lengths[t] == 0)
                        continue;
                    const uint bits = ordered_bits(t == query ? INFINITY : scores[t]);
                    ones += (bits & high) == threshold && (bits >> bit) & 1u;
                }
                const uint above = uint(group_reduce<false>(ones, scratch, simd, lane));
                if (above >= ties)
                    threshold |= 1u << bit;
                else
                    ties -= above;
            }
            if (simd == 0 && count > 0) {
                uint seen = 0;
                for (uint first = start; first < end; first += 32) {
                    const uint t = first + lane;
                    const bool candidate = t < end && (t == query || lengths[t] > 0);
                    const uint bits =
                        candidate ? ordered_bits(t == query ? INFINITY : scores[t]) : 0;
                    const uint tie = candidate && bits == threshold;
                    const uint rank = seen + simd_prefix_exclusive_sum(tie);
                    seen += simd_sum(tie);
                    if (candidate && (bits > threshold || (tie && rank < ties)))
                        chosen[t] = true;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (simd == 0) {
        device ushort *route = routes + row * small, *twin = route + small;
        uint count = 0, video_count = 0;
        for (uint first = 0; first < small; first += 32) {
            const uint t = first + lane, tile = t / 2;
            const bool live = t < small && veda_rows(lengths, t) > 0;
            const uint flag = live && (global || tile >= video || chosen[tile]);
            const uint before = simd_prefix_exclusive_sum(flag);
            if (flag) {
                route[count + before] = ushort(t);
                twin[count + before] = ushort(t);
            }
            count += simd_sum(flag);
            if (!global)
                video_count +=
                    simd_sum(uint(t < small && t % 2 == 0 && tile < video && chosen[tile]));
        }
        if (lane == 0) {
            counts[row] = count;
            counts[row + 1] = count;
            if (!global)
                atomic_fetch_add_explicit(kept, video_count, memory_order_relaxed);
        }
    }
}

// Kernels of the video VAE's encoder, whose activations are FP16 [sequences, frames, height,
// width, channels]: a sequence is one tile of one clip, and the group norms' statistics are those
// of one frame of one sequence.
constant constexpr uint VIDEO_GROUPS = 32;
// Pixels one threadgroup of video_encoder_norm_partials sums.
constant constexpr uint VIDEO_NORM_PIXELS = 1024;
// Channels each of its threads sums at most, which holds the encoder's 1024.
constant constexpr uint VIDEO_NORM_SLOTS = 4;

// The tiles of the canvas [frames, height, width, 3] in [0, 1], normalized with the ImageNet
// statistics, into [sequences, frames, tile height, tile width, p[7]] with the channels past the
// third zero, so the first convolution reads whole fragments. `tiles` holds each sequence's first
// frame, top and left, and a frame past the canvas repeats its last one, as the reference pads a
// short clip.
//
// p: values, canvas frames, height, width, frames, tile height, tile width, channels.
kernel void video_encoder_tile_input(device const float *canvas [[buffer(0)]],
                                     device const uint *tiles [[buffer(1)]],
                                     device half *y [[buffer(2)]], constant uint *p [[buffer(3)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    const uint channels = p[7], tile_pixels = p[5] * p[6];
    const uint channel = i % channels, pixel = i / channels % tile_pixels;
    const uint frame = i / (channels * tile_pixels) % p[4];
    const uint sequence = i / (channels * tile_pixels * p[4]);
    if (channel >= 3) {
        y[i] = 0;
        return;
    }
    const float mean[3] = {0.485, 0.456, 0.406}, deviation[3] = {0.229, 0.224, 0.225};
    device const uint *tile = tiles + sequence * 3;
    const uint source_frame = min(tile[0] + frame, p[1] - 1);
    const uint row = pixel / p[6] + tile[1], column = pixel % p[6] + tile[2];
    const float value = canvas[((size_t(source_frame) * p[2] + row) * p[3] + column) * 3 + channel];
    y[i] = half((value - mean[channel]) / deviation[channel]);
}

// Sums and sums of squares of each group over VIDEO_NORM_PIXELS pixels of one frame, into
// partials[frame, chunk, group, 2]. The values are taken from the group's first value in the frame,
// so that a mean far from zero does not swallow the variance. A thread's channels land in their own
// slots and a group adds them up in a fixed order, so two encodes of one input agree.
//
// p: pixels, channels, chunks of a frame.
kernel void video_encoder_norm_partials(device const half *x [[buffer(0)]],
                                        device float *partials [[buffer(1)]],
                                        constant uint *p [[buffer(2)]],
                                        uint g [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256 * VIDEO_NORM_SLOTS * 2];
    const uint pixels = p[0], channels = p[1], chunks = p[2];
    const uint group_size = channels / VIDEO_GROUPS;
    x += size_t(g / chunks) * pixels * channels;
    // Threads go across a pixel's channels, so the reads of one pixel are contiguous.
    const uint lanes = min(channels, 256u), lane = tid % lanes, step = 256 / lanes;
    const uint first = g % chunks * VIDEO_NORM_PIXELS;
    const uint last = min(first + VIDEO_NORM_PIXELS, pixels);
    for (uint slot = 0; slot < VIDEO_NORM_SLOTS; ++slot) {
        const uint channel = lane + slot * 256;
        float sum = 0, squares = 0;
        if (channel < channels) {
            const float shift = float(x[channel / group_size * group_size]);
            for (uint pixel = first + tid / lanes; pixel < last; pixel += step) {
                const float value = float(x[size_t(pixel) * channels + channel]) - shift;
                sum += value;
                squares = fma(value, value, squares);
            }
        }
        sums[(tid * VIDEO_NORM_SLOTS + slot) * 2] = sum;
        sums[(tid * VIDEO_NORM_SLOTS + slot) * 2 + 1] = squares;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= VIDEO_GROUPS)
        return;
    float sum = 0, squares = 0;
    for (uint other = 0; other < 256; ++other) {
        for (uint slot = 0; slot < VIDEO_NORM_SLOTS; ++slot) {
            const uint channel = other % lanes + slot * 256;
            if (channel < channels && channel / group_size == tid) {
                sum += sums[(other * VIDEO_NORM_SLOTS + slot) * 2];
                squares += sums[(other * VIDEO_NORM_SLOTS + slot) * 2 + 1];
            }
        }
    }
    partials[(size_t(g) * VIDEO_GROUPS + tid) * 2] = sum;
    partials[(size_t(g) * VIDEO_GROUPS + tid) * 2 + 1] = squares;
}

// The mean and reciprocal standard deviation of each group of each frame from its partials, into
// statistics[frame, group, 2].
//
// p: frames × groups, pixels, channels, chunks of a frame, epsilon.
kernel void video_encoder_norm_statistics(device const half *x [[buffer(0)]],
                                          device const float *partials [[buffer(1)]],
                                          device float *statistics [[buffer(2)]],
                                          constant uint *p [[buffer(3)]],
                                          uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    const uint frame = i / VIDEO_GROUPS, group = i % VIDEO_GROUPS;
    const uint pixels = p[1], channels = p[2], chunks = p[3], group_size = channels / VIDEO_GROUPS;
    float sum = 0, squares = 0;
    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const size_t at = ((size_t(frame) * chunks + chunk) * VIDEO_GROUPS + group) * 2;
        sum += partials[at];
        squares += partials[at + 1];
    }
    const float count = float(pixels * group_size);
    const float mean = sum / count, variance = max(squares / count - mean * mean, 0.0f);
    const float shift = float(x[size_t(frame) * pixels * channels + group * group_size]);
    statistics[i * 2] = shift + mean;
    statistics[i * 2 + 1] = rsqrt(variance + as_type<float>(p[4]));
}

// SiLU of the group-normalized input with a per-channel affine, each frame with its own
// statistics.
//
// p: values, pixels of a frame, channels.
kernel void video_encoder_norm_silu(device const half *x [[buffer(0)]],
                                    device const float *statistics [[buffer(1)]],
                                    device const float *w [[buffer(2)]],
                                    device const float *b [[buffer(3)]],
                                    device half *y [[buffer(4)]], constant uint *p [[buffer(5)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    const uint channels = p[2], channel = i % channels;
    const uint group = i / (channels * p[1]) * VIDEO_GROUPS + channel / (channels / VIDEO_GROUPS);
    const float value =
        (float(x[i]) - statistics[group * 2]) * statistics[group * 2 + 1] * w[channel] + b[channel];
    y[i] = half(value / (1 + exp(-value)));
}

// A depthwise convolution of the latent upscaler along time, FP16 [frames, pixels, channels]:
// y[t, p, c] = b[c] + sum over k of w[c, k] · x[t + k − kernel / 2, p, c], with zeros past either
// end of the clip.
//
// p: values, pixels of a frame, channels, frames, kernel.
kernel void latent_upscaler_temporal(device const half *x [[buffer(0)]],
                                     device const float *w [[buffer(1)]],
                                     device const float *b [[buffer(2)]],
                                     device half *y [[buffer(3)]], constant uint *p [[buffer(4)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    const uint channels = p[2], frame_values = p[1] * channels, kernel_taps = p[4];
    const uint channel = i % channels;
    const int frame = int(i / frame_values), frames = int(p[3]);
    float sum = b[channel];
    for (uint tap = 0; tap < kernel_taps; ++tap) {
        const int source = frame + int(tap) - int(kernel_taps / 2);
        if (source >= 0 && source < frames)
            sum = fma(w[channel * kernel_taps + tap],
                      float(x[size_t(source) * frame_values + i % frame_values]), sum);
    }
    y[i] = half(sum);
}

// A bilinear resize of every frame of FP16 [frames, height, width, channels], with the pixel
// centers aligned as align_corners=False aligns them.
//
// p: values, height, width, channels, output height, output width.
kernel void latent_upscaler_resize(device const half *x [[buffer(0)]], device half *y [[buffer(1)]],
                                   constant uint *p [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    const uint height = p[1], width = p[2], channels = p[3], output_height = p[4],
               output_width = p[5];
    const uint channel = i % channels, pixel = i / channels;
    const uint column = pixel % output_width, row = pixel / output_width % output_height;
    const uint frame = pixel / (output_width * output_height);
    const float sy = max((float(row) + 0.5f) * float(height) / float(output_height) - 0.5f, 0.0f);
    const float sx = max((float(column) + 0.5f) * float(width) / float(output_width) - 0.5f, 0.0f);
    const uint top = min(uint(sy), height - 1), bottom = min(top + 1, height - 1);
    const uint left = min(uint(sx), width - 1), right = min(left + 1, width - 1);
    const float down = sy - float(top), across = sx - float(left);
    device const half *plane = x + size_t(frame) * height * width * channels + channel;
    const float upper = float(plane[(top * width + left) * channels]) * (1 - across) +
                        float(plane[(top * width + right) * channels]) * across;
    const float lower = float(plane[(bottom * width + left) * channels]) * (1 - across) +
                        float(plane[(bottom * width + right) * channels]) * across;
    y[i] = half(upper * (1 - down) + lower * down);
}
