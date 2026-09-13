#pragma once

#include <cstdint>
#include <cuda_runtime.h>

struct Mmh3AttentionLayout;

struct Mmh3SparseWorkspace {
    float* centroids;       // [heads, blocks, 128]
    float* block_keys;      // [heads, blocks, 128], centered by center_keys
    float* value_sums;      // [heads, blocks, 128]
    float* key_mean;        // [heads, 128]
    float* key_variance;    // [heads, 128]
    float* row_offsets;     // [heads, tokens]
    uint16_t* routes;       // [heads, blocks, blocks], the first route_counts entries of each row are used
    int32_t* route_counts;  // [heads, blocks]
    float* tail_max;        // [heads, blocks]
    float* tail_sum;        // [heads, blocks]
    float* tail_values;     // [heads, blocks, 128]
};

// One sequence of quantized Q/K, transposed V, and their scales. Owned by the Rust workspace.
struct Mmh3QuantizedWorkspace {
    uint8_t* query;          // [tokens, heads, 128], INT8
    uint8_t* key;            // [tokens, heads, 128], INT8
    uint8_t* value;          // [heads, 128, tokens rounded up to 64], FP8 E4M3
    float* query_scales;     // [heads, ceil(tokens / 64)]
    float* key_scales;       // [heads, ceil(tokens / 64)]
    float* value_scales;     // [heads]
    float* value_maxima;     // [heads, ceil(tokens / 64)]
};

// INT8 QK and FP8 PV, FP32 softmax/accumulation, BF16 output. A null sparse workspace means dense attention.
extern "C" int mmh3_attention_quantized(const void* query, const void* key, const void* value, void* output,
                                         int tokens, int heads, const Mmh3AttentionLayout* layout, float scale,
                                         const Mmh3SparseWorkspace* sparse, const Mmh3QuantizedWorkspace* workspace,
                                         cudaStream_t stream);
