#include "nvfp4.cuh"

// Quantization of activations and weights for the NVFP4 linear layers, see nvfp4.cuh.

namespace {

constexpr int THREADS = 256;

// Loads the eight values lane `lane` holds of 256-column group `group` of a row of `k` inputs and
// rotates the group. With SWIGLU the input row holds `k` gates followed by as many up projections,
// and the values are silu(gate) · up rounded to BF16 like mmh3_swiglu.
template <bool SWIGLU>
__device__ __forceinline__ void load_rotated(const __nv_bfloat16 *__restrict__ input, int row,
                                             int group, int lane, int k, float (&values)[8]) {
    const size_t row_offset = static_cast<size_t>(row) * k * (SWIGLU ? 2 : 1);
    const int column = group * CONVROT_GROUP + lane * 8;
    const uint4 packed = *reinterpret_cast<const uint4 *>(input + row_offset + column);
    const __nv_bfloat162 *pairs = reinterpret_cast<const __nv_bfloat162 *>(&packed);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        const float2 value = __bfloat1622float2(pairs[pair]);
        values[pair * 2] = value.x;
        values[pair * 2 + 1] = value.y;
    }
    if constexpr (SWIGLU) {
        const uint4 packed_up = *reinterpret_cast<const uint4 *>(input + row_offset + k + column);
        const __nv_bfloat162 *up_pairs = reinterpret_cast<const __nv_bfloat162 *>(&packed_up);
        #pragma unroll
        for (int pair = 0; pair < 4; pair++) {
            const float2 up = __bfloat1622float2(up_pairs[pair]);
            const float first = values[pair * 2], second = values[pair * 2 + 1];
            values[pair * 2] =
                __bfloat162float(__float2bfloat16_rn(first / (1.0f + __expf(-first)) * up.x));
            values[pair * 2 + 1] =
                __bfloat162float(__float2bfloat16_rn(second / (1.0f + __expf(-second)) * up.y));
        }
    }
    rotate_group(values, lane);
}

// One block per row: the largest rotated magnitude of the rows, folded into `maximum`.
template <bool SWIGLU>
__global__ void __launch_bounds__(THREADS)
    maximum_kernel(const __nv_bfloat16 *__restrict__ input, unsigned *maximum, int k) {
    __shared__ float warp_maxima[THREADS / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    float local = 0.0f;
    for (int group = warp; group < k / CONVROT_GROUP; group += THREADS / 32) {
        float values[8];
        load_rotated<SWIGLU>(input, blockIdx.x, group, lane, k, values);
        #pragma unroll
        for (int index = 0; index < 8; index++) {
            local = fmaxf(local, fabsf(values[index]));
        }
    }
    nvfp4_fold_maximum(local, maximum, warp_maxima);
}

// One block per row: rotates and quantizes the `k` inputs of the row into a matrix with `columns`
// columns with the tensor scale of margin · reference, folds the rows' largest magnitude into
// `observed` when it is not null, and writes the tensor scale to `tensor_scale`.
template <bool SWIGLU>
__global__ void __launch_bounds__(THREADS)
    quantize_kernel(const __nv_bfloat16 *__restrict__ input, uint8_t *__restrict__ values,
                    uint8_t *__restrict__ scales, float *tensor_scale, const unsigned *reference,
                    float margin, unsigned *observed, int k, int columns) {
    __shared__ float warp_maxima[THREADS / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row = blockIdx.x;
    const float scale = nvfp4_tensor_scale(__uint_as_float(*reference) * margin);
    const float inverse_tensor_scale = 1.0f / scale;
    float local = 0.0f;
    for (int group = warp; group < k / CONVROT_GROUP; group += THREADS / 32) {
        float group_values[8];
        load_rotated<SWIGLU>(input, row, group, lane, k, group_values);
        #pragma unroll
        for (int index = 0; index < 8; index++) {
            local = fmaxf(local, fabsf(group_values[index]));
        }
        nvfp4_store_group(group_values, lane, row, group, columns, inverse_tensor_scale, values,
                          scales);
    }
    if (observed != nullptr) {
        nvfp4_fold_maximum(local, observed, warp_maxima);
    }
    if (row == 0 && threadIdx.x == 0) {
        *tensor_scale = scale;
    }
}

// alpha_beta = {activation tensor scale · weight tensor scale, 0}.
__global__ void alpha_kernel(const float *tensor_scale, float weight_scale, float *alpha_beta) {
    alpha_beta[0] = *tensor_scale * weight_scale;
    alpha_beta[1] = 0.0f;
}

// One block per output row of an INT8 weight [n, k] with per-row scales, requantized to NVFP4 with
// the given tensor scale into a matrix with `columns` columns. With `deinterleave`, output row r
// comes from the row that mmh3_interleave_swiglu_rows moved it to.
__global__ void __launch_bounds__(THREADS)
    quantize_weights_kernel(const int8_t *__restrict__ weights,
                            const float *__restrict__ row_scales, uint8_t *__restrict__ values,
                            uint8_t *__restrict__ scales, float tensor_scale, int n, int k,
                            int columns, int deinterleave) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row = blockIdx.x;
    const int source_row = deinterleave ? nvfp4_interleaved_row(row, n) : row;
    const float row_scale = row_scales[source_row];
    const int8_t *source = weights + static_cast<size_t>(source_row) * k;
    for (int group = warp; group < k / CONVROT_GROUP; group += THREADS / 32) {
        const uint2 packed_weights =
            *reinterpret_cast<const uint2 *>(source + group * CONVROT_GROUP + lane * 8);
        const int8_t *bytes = reinterpret_cast<const int8_t *>(&packed_weights);
        float group_values[8];
        #pragma unroll
        for (int index = 0; index < 8; index++) {
            group_values[index] = static_cast<float>(bytes[index]) * row_scale;
        }
        nvfp4_store_group(group_values, lane, row, group, columns, 1.0f / tensor_scale, values,
                          scales);
    }
}

// One thread per block of 16 values: row r of BF16 rows [m, count] times `multiplier`, unrotated,
// into columns [offset, offset + count) of an NVFP4 matrix with `columns` columns. The tensor
// scale is at `tensor_scale`, or `fixed_tensor_scale` when that is null. With `deinterleave`, row r
// comes from the row that mmh3_interleave_swiglu_rows moved it to.
__global__ void __launch_bounds__(THREADS)
    quantize_columns_kernel(const __nv_bfloat16 *__restrict__ input, float multiplier,
                            uint8_t *__restrict__ values, uint8_t *__restrict__ scales,
                            const float *tensor_scale, float fixed_tensor_scale, int deinterleave,
                            int m, int count, int offset, int columns) {
    const int blocks = count / 16;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(m) * blocks) {
        return;
    }
    const int row = static_cast<int>(index / blocks);
    const int block = static_cast<int>(index % blocks);
    const int source_row = deinterleave ? nvfp4_interleaved_row(row, m) : row;
    const uint4 *source = reinterpret_cast<const uint4 *>(
        input + static_cast<size_t>(source_row) * count + block * 16);
    float block_values[16];
    float maximum = 0.0f;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const uint4 packed = source[half];
        const __nv_bfloat162 *pairs = reinterpret_cast<const __nv_bfloat162 *>(&packed);
        #pragma unroll
        for (int pair = 0; pair < 4; pair++) {
            const float2 value = __bfloat1622float2(pairs[pair]);
            block_values[half * 8 + pair * 2] = value.x * multiplier;
            block_values[half * 8 + pair * 2 + 1] = value.y * multiplier;
        }
    }
    #pragma unroll
    for (int value = 0; value < 16; value++) {
        maximum = fmaxf(maximum, fabsf(block_values[value]));
    }
    const float inverse_tensor_scale =
        1.0f / (tensor_scale != nullptr ? *tensor_scale : fixed_tensor_scale);
    const __nv_fp8_storage_t scale = nvfp4_block_scale(maximum, inverse_tensor_scale);
    const uint2 packed = make_uint2(nvfp4_pack(block_values, scale, inverse_tensor_scale),
                                    nvfp4_pack(block_values + 8, scale, inverse_tensor_scale));
    *reinterpret_cast<uint2 *>(values + static_cast<size_t>(row) * columns / 2 + offset / 2 +
                               block * 8) = packed;
    scales[nvfp4_scale_offset(row, offset / 16 + block, columns / 16)] = scale;
}

// One block per row of INT8 rows [rows, k] with per-row scales: the largest row norm, folded into
// `maximum`.
__global__ void __launch_bounds__(THREADS)
    row_norm_kernel(const int8_t *__restrict__ weights, const float *__restrict__ row_scales,
                    unsigned *maximum, int k) {
    __shared__ float warp_sums[THREADS / 32];
    const int8_t *row = weights + static_cast<size_t>(blockIdx.x) * k;
    float squares = 0.0f;
    for (int column = threadIdx.x; column < k; column += THREADS) {
        const float value = static_cast<float>(row[column]);
        squares = fmaf(value, value, squares);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        squares += __shfl_xor_sync(0xffffffff, squares, offset);
    }
    if (threadIdx.x % 32 == 0) {
        warp_sums[threadIdx.x / 32] = squares;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int warp = 0; warp < THREADS / 32; warp++) {
            sum += warp_sums[warp];
        }
        atomicMax(maximum, __float_as_uint(row_scales[blockIdx.x] * sqrtf(sum)));
    }
}

// The largest magnitude of `count` BF16 values, folded into `maximum`.
__global__ void __launch_bounds__(THREADS)
    maximum_bf16_kernel(const __nv_bfloat16 *__restrict__ input, size_t count, unsigned *maximum) {
    __shared__ float warp_maxima[THREADS / 32];
    float local = 0.0f;
    for (size_t index = static_cast<size_t>(blockIdx.x) * THREADS + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * THREADS) {
        local = fmaxf(local, fabsf(__bfloat162float(input[index])));
    }
    nvfp4_fold_maximum(local, maximum, warp_maxima);
}

} // namespace

// Quantizes BF16 activations [m, k], or with `swiglu` the SwiGLU of gates and up projections
// [m, 2k], to NVFP4 after the ConvRot rotation into the first k of `columns` columns, and writes
// the tensor scale to `tensor_scale`. With `exact`, the tensor scale comes from the rows' own
// largest magnitude, found in a first pass and left in `reference`. Otherwise it comes from
// margin · reference, a magnitude seen before, and the rows' largest magnitude goes to `observed`.
// k must be a multiple of 256 and columns of 64.
extern "C" int mmh3_nvfp4_quantize(const __nv_bfloat16 *input, int swiglu, uint8_t *values,
                                   uint8_t *scales, float *tensor_scale, unsigned *reference,
                                   float margin, unsigned *observed, int exact, int m, int k,
                                   int columns, cudaStream_t stream) {
    if (k % CONVROT_GROUP != 0 || columns < k || columns % 64 != 0 || m <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    unsigned *cleared = exact ? reference : observed;
    cudaError_t status = cudaMemsetAsync(cleared, 0, sizeof(unsigned), stream);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    auto launch = [&](auto maximum, auto quantize) {
        if (exact) {
            maximum<<<m, THREADS, 0, stream>>>(input, reference, k);
            quantize<<<m, THREADS, 0, stream>>>(input, values, scales, tensor_scale, reference,
                                                1.0f, nullptr, k, columns);
        } else {
            quantize<<<m, THREADS, 0, stream>>>(input, values, scales, tensor_scale, reference,
                                                margin, observed, k, columns);
        }
    };
    if (swiglu) {
        launch(maximum_kernel<true>, quantize_kernel<true>);
    } else {
        launch(maximum_kernel<false>, quantize_kernel<false>);
    }
    return static_cast<int>(cudaGetLastError());
}

// Writes alpha and beta = 0 for mmh3_cublaslt_nvfp4 from the activations' tensor scale.
extern "C" int mmh3_nvfp4_alpha(const float *tensor_scale, float weight_scale, float *alpha_beta,
                                cudaStream_t stream) {
    alpha_kernel<<<1, 1, 0, stream>>>(tensor_scale, weight_scale, alpha_beta);
    return static_cast<int>(cudaGetLastError());
}

// Requantizes an INT8 ConvRot weight [n, k] with per-row scales to NVFP4 with `tensor_scale` into
// the first k of `columns` columns.
extern "C" int mmh3_nvfp4_quantize_weights(const int8_t *weights, const float *row_scales,
                                           uint8_t *values, uint8_t *scales, float tensor_scale,
                                           int n, int k, int columns, int deinterleave,
                                           cudaStream_t stream) {
    if (k % CONVROT_GROUP != 0 || columns < k || columns % 64 != 0 || n <= 0 ||
        (deinterleave && n % 8 != 0)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    quantize_weights_kernel<<<n, THREADS, 0, stream>>>(weights, row_scales, values, scales,
                                                       tensor_scale, n, k, columns, deinterleave);
    return static_cast<int>(cudaGetLastError());
}

// Quantizes BF16 rows [m, count] times `multiplier` without rotation into columns
// [offset, offset + count) of an NVFP4 matrix with `columns` columns, see quantize_columns_kernel.
// count and offset must be multiples of 64.
extern "C" int mmh3_nvfp4_quantize_columns(const __nv_bfloat16 *input, float multiplier,
                                           uint8_t *values, uint8_t *scales,
                                           const float *tensor_scale, float fixed_tensor_scale,
                                           int deinterleave, int m, int count, int offset,
                                           int columns, cudaStream_t stream) {
    if (m <= 0 || count <= 0 || count % 64 != 0 || offset % 64 != 0 || offset + count > columns ||
        columns % 64 != 0 || (deinterleave && m % 8 != 0)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int64_t threads = static_cast<int64_t>(m) * (count / 16);
    quantize_columns_kernel<<<static_cast<unsigned>((threads + THREADS - 1) / THREADS), THREADS, 0,
                              stream>>>(input, multiplier, values, scales, tensor_scale,
                                        fixed_tensor_scale, deinterleave, m, count, offset,
                                        columns);
    return static_cast<int>(cudaGetLastError());
}

// Writes the largest row norm of the INT8 rows `down` [rank, k] with per-row scales to maxima[0]
// and the largest magnitude of the `up_count` BF16 values `up` to maxima[1], both as float bits.
extern "C" int mmh3_nvfp4_adapter_ranges(const int8_t *down, const float *down_scales, int rank,
                                         int k, const __nv_bfloat16 *up, size_t up_count,
                                         unsigned *maxima, cudaStream_t stream) {
    if (rank <= 0 || k <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const cudaError_t status = cudaMemsetAsync(maxima, 0, 2 * sizeof(unsigned), stream);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    row_norm_kernel<<<rank, THREADS, 0, stream>>>(down, down_scales, maxima, k);
    maximum_bf16_kernel<<<256, THREADS, 0, stream>>>(up, up_count, maxima + 1);
    return static_cast<int>(cudaGetLastError());
}
