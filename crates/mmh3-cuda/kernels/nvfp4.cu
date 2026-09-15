#include "nvfp4.cuh"

// Quantization of activations and weights for the NVFP4 linear layers, see nvfp4.cuh.

namespace {

constexpr int THREADS = 256;

// Loads the eight values lane `lane` holds of 256-column group `group` of a row of `columns`
// inputs and rotates the group. With SWIGLU the input row holds `columns` gates followed by as many
// up projections, and the values are silu(gate) · up rounded to BF16 like mmh3_swiglu.
template <bool SWIGLU>
__device__ __forceinline__ void load_rotated(const __nv_bfloat16 *__restrict__ input, int row,
                                             int group, int lane, int columns, float (&values)[8]) {
    const size_t row_offset = static_cast<size_t>(row) * columns * (SWIGLU ? 2 : 1);
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
        const uint4 packed_up =
            *reinterpret_cast<const uint4 *>(input + row_offset + columns + column);
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
    maximum_kernel(const __nv_bfloat16 *__restrict__ input, unsigned *maximum, int columns) {
    __shared__ float warp_maxima[THREADS / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    float local = 0.0f;
    for (int group = warp; group < columns / CONVROT_GROUP; group += THREADS / 32) {
        float values[8];
        load_rotated<SWIGLU>(input, blockIdx.x, group, lane, columns, values);
        #pragma unroll
        for (int index = 0; index < 8; index++) {
            local = fmaxf(local, fabsf(values[index]));
        }
    }
    nvfp4_fold_maximum(local, maximum, warp_maxima);
}

// One block per row: rotates and quantizes with the tensor scale of margin · reference, folds the
// rows' largest magnitude into `observed` when it is not null, and writes the tensor scale to
// `tensor_scale`.
template <bool SWIGLU>
__global__ void __launch_bounds__(THREADS)
    quantize_kernel(const __nv_bfloat16 *__restrict__ input, uint8_t *__restrict__ values,
                    uint8_t *__restrict__ scales, float *tensor_scale, const unsigned *reference,
                    float margin, unsigned *observed, int columns) {
    __shared__ float warp_maxima[THREADS / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row = blockIdx.x;
    const float scale = nvfp4_tensor_scale(__uint_as_float(*reference) * margin);
    const float inverse_tensor_scale = 1.0f / scale;
    float local = 0.0f;
    for (int group = warp; group < columns / CONVROT_GROUP; group += THREADS / 32) {
        float group_values[8];
        load_rotated<SWIGLU>(input, row, group, lane, columns, group_values);
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
// the given tensor scale. With `deinterleave`, output row r comes from the row that
// mmh3_interleave_swiglu_rows moved it to.
__global__ void __launch_bounds__(THREADS)
    quantize_weights_kernel(const int8_t *__restrict__ weights,
                            const float *__restrict__ row_scales, uint8_t *__restrict__ values,
                            uint8_t *__restrict__ scales, float tensor_scale, int n, int k,
                            int deinterleave) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row = blockIdx.x;
    int source_row = row;
    if (deinterleave) {
        const int half = n / 2;
        const int feature = row < half ? row : row - half;
        source_row = feature / 4 * 8 + feature % 4 + (row < half ? 0 : 4);
    }
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
        nvfp4_store_group(group_values, lane, row, group, k, 1.0f / tensor_scale, values, scales);
    }
}

} // namespace

// Quantizes BF16 activations [m, k], or with `swiglu` the SwiGLU of gates and up projections
// [m, 2k], to NVFP4 after the ConvRot rotation, and writes the tensor scale to `tensor_scale`.
// With `exact`, the tensor scale comes from the rows' own largest magnitude, found in a first pass
// and left in `reference`. Otherwise it comes from margin · reference, a magnitude seen before, and
// the rows' largest magnitude goes to `observed`. k must be a multiple of 256.
extern "C" int mmh3_nvfp4_quantize(const __nv_bfloat16 *input, int swiglu, uint8_t *values,
                                   uint8_t *scales, float *tensor_scale, unsigned *reference,
                                   float margin, unsigned *observed, int exact, int m, int k,
                                   cudaStream_t stream) {
    if (k % CONVROT_GROUP != 0 || m <= 0) {
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
                                                1.0f, nullptr, k);
        } else {
            quantize<<<m, THREADS, 0, stream>>>(input, values, scales, tensor_scale, reference,
                                                margin, observed, k);
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

// Requantizes an INT8 ConvRot weight [n, k] with per-row scales to NVFP4 with `tensor_scale`.
extern "C" int mmh3_nvfp4_quantize_weights(const int8_t *weights, const float *row_scales,
                                           uint8_t *values, uint8_t *scales, float tensor_scale,
                                           int n, int k, int deinterleave, cudaStream_t stream) {
    if (k % CONVROT_GROUP != 0 || n <= 0 || (deinterleave && n % 8 != 0)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    quantize_weights_kernel<<<n, THREADS, 0, stream>>>(weights, row_scales, values, scales,
                                                       tensor_scale, n, k, deinterleave);
    return static_cast<int>(cudaGetLastError());
}
