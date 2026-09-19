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

kernel void unary_op(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                     constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    float v = x[i], s = as_type<float>(p[2]);
    y[i] = p[1] == 0 ? v * s : p[1] == 1 ? v / (1 + exp(-v)) : p[1] == 2 ? clamp(v, -s, s) : v;
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

float reduce_sum(float v, threadgroup float *scratch, uint lane) {
    scratch[lane] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (lane < s)
            scratch[lane] += scratch[lane + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

kernel void normalize(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                      device float *y [[buffer(2)]], constant uint *p [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[256];
    uint width = p[0];
    float sum = 0, squares = 0;
    for (uint c = lane; c < width; c += 256)
        sum += x[row * width + c];
    float mean = p[2] ? reduce_sum(sum, scratch, lane) / width : 0;

    for (uint c = lane; c < width; c += 256) {
        float v = x[row * width + c] - mean;
        squares += v * v;
    }

    float inv = rsqrt(reduce_sum(squares, scratch, lane) / width + as_type<float>(p[1]));
    for (uint c = lane; c < width; c += 256)
        y[row * width + c] = (x[row * width + c] - mean) * inv * w[c];
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
kernel void audio_columns(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                          constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0])
        return;
    uint channels = p[2], kernel_size = p[3], row = i / (channels * kernel_size),
         tap = i / channels % kernel_size;
    int source = int(row % p[1]) + int(tap * p[4]) - int(p[4] * (kernel_size - 1) / 2);
    y[i] = source >= 0 && source < int(p[1])
               ? x[((row / p[1]) * p[1] + source) * channels + i % channels]
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

// MPS comparison path: INT8 is exactly representable as FP16.
kernel void int8_to_half(device const char *x [[buffer(0)]], device half *y [[buffer(1)]],
                         constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        y[i] = half(x[p[1] + i]);
}

kernel void scale_linear_rows(device float *x [[buffer(0)]],
                              device const float *scales [[buffer(1)]],
                              constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0])
        x[i] *= scales[i / p[1]];
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
