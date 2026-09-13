#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Kernels of the audio VAE's BigVGAN decoder. Signals are FP32 and time-major, [sequences, length, channels], so
// convolutions become GEMMs over im2col rows.

namespace {

constexpr int THREADS = 256;
// Anti-aliased activations upsample by 2 and downsample by 2 with 12-tap filters.
constexpr int RESAMPLE_TAPS = 12;
constexpr int UPSAMPLE_PAD = RESAMPLE_TAPS / 2 - 1;
constexpr int UPSAMPLE_CROP = UPSAMPLE_PAD * 2 + (RESAMPLE_TAPS - 2) / 2;
constexpr int DOWNSAMPLE_PAD_LEFT = RESAMPLE_TAPS / 2 - 1;

unsigned grid_for(size_t count) {
    size_t blocks = (count + THREADS - 1) / THREADS;
    return static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
}

// columns[(s, t), j, c] = input[s, t + j · dilation − padding, c], zero outside the sequence.
__global__ void im2col_kernel(const float* __restrict__ input, float* __restrict__ columns, int sequences, int length,
                              int channels, int kernel, int dilation, int padding) {
    const size_t count = static_cast<size_t>(sequences) * length * kernel * channels;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        const int channel = static_cast<int>(index % channels);
        const int tap = static_cast<int>((index / channels) % kernel);
        const size_t row = index / (static_cast<size_t>(channels) * kernel);
        const int time = static_cast<int>(row % length);
        const size_t sequence = row / length;
        const int source = time + tap * dilation - padding;
        columns[index] = source >= 0 && source < length
                             ? input[(sequence * length + source) * channels + channel]
                             : 0.0f;
    }
}

// Transposed convolution from its per-tap products: products[(s, i), j, c] = Σ input[s, i, :] · weight[:, c, j].
// output[s, t, c] = bias[c] + Σ products[s, i, j, c] over the taps j with i · stride = t + padding − j.
__global__ void conv_transpose_gather_kernel(const float* __restrict__ products, const float* __restrict__ bias,
                                             float* __restrict__ output, int sequences, int input_length,
                                             int output_length, int channels, int kernel, int stride_size, int padding) {
    const size_t count = static_cast<size_t>(sequences) * output_length * channels;
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += grid_stride) {
        const int channel = static_cast<int>(index % channels);
        const int time = static_cast<int>((index / channels) % output_length);
        const size_t sequence = index / (static_cast<size_t>(channels) * output_length);
        float sum = bias != nullptr ? bias[channel] : 0.0f;
        for (int tap = (time + padding) % stride_size; tap < kernel; tap += stride_size) {
            const int shifted = time + padding - tap;
            if (shifted < 0) {
                break;
            }
            const int source = shifted / stride_size;
            if (source < input_length) {
                sum += products[((sequence * input_length + source) * kernel + tap) * channels + channel];
            }
        }
        output[index] = sum;
    }
}

// Upsampled signal at position `position` of 2 × length: replicate-padded transposed convolution with the upsample
// filter, scaled by 2 and cropped.
__device__ __forceinline__ float upsampled(const float* signal, int length, int channels, int channel, int position,
                                           const float* filter) {
    const int shifted = position + UPSAMPLE_CROP;
    float sum = 0.0f;
    for (int tap = shifted % 2; tap < RESAMPLE_TAPS; tap += 2) {
        int source = (shifted - tap) / 2 - UPSAMPLE_PAD;
        source = source < 0 ? 0 : (source >= length ? length - 1 : source);
        sum += filter[tap] * signal[static_cast<size_t>(source) * channels + channel];
    }
    return 2.0f * sum;
}

// SnakeBeta between a 2× upsample and a 2× downsample: x + sin²(αx) / β on the upsampled signal, then the
// replicate-padded low-pass filter with stride 2. `parameters` holds α and 1 / (β + 1e-9) per channel.
__global__ void snake_beta_kernel(const float* __restrict__ input, float* __restrict__ output,
                                  const float* __restrict__ parameters, const float* __restrict__ up_filter,
                                  const float* __restrict__ down_filter, int sequences, int length, int channels) {
    __shared__ float filters[2 * RESAMPLE_TAPS];
    if (threadIdx.x < 2 * RESAMPLE_TAPS) {
        filters[threadIdx.x] = threadIdx.x < RESAMPLE_TAPS ? up_filter[threadIdx.x] : down_filter[threadIdx.x - RESAMPLE_TAPS];
    }
    __syncthreads();
    const size_t count = static_cast<size_t>(sequences) * length * channels;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        const int channel = static_cast<int>(index % channels);
        const int time = static_cast<int>((index / channels) % length);
        const size_t sequence = index / (static_cast<size_t>(channels) * length);
        const float* signal = input + sequence * length * channels;
        const float alpha = parameters[channel];
        const float inverse_beta = parameters[channels + channel];
        float sum = 0.0f;
        for (int tap = 0; tap < RESAMPLE_TAPS; tap++) {
            int position = 2 * time + tap - DOWNSAMPLE_PAD_LEFT;
            position = position < 0 ? 0 : (position >= 2 * length ? 2 * length - 1 : position);
            const float value = upsampled(signal, length, channels, channel, position, filters);
            const float sine = sinf(alpha * value);
            sum += filters[RESAMPLE_TAPS + tap] * (value + sine * sine * inverse_beta);
        }
        output[index] = sum;
    }
}

// output = (first + second) / divisor, in place when output aliases an input.
__global__ void add_kernel(float* output, const float* first, const float* second, size_t count, float divisor) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        const float sum = first[index] + second[index];
        output[index] = divisor == 1.0f ? sum : sum / divisor;
    }
}

__global__ void clamp_kernel(float* values, size_t count, float limit) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        values[index] = fminf(fmaxf(values[index], -limit), limit);
    }
}

}  // namespace

extern "C" int mmh3_audio_im2col(const float* input, float* columns, int sequences, int length, int channels,
                                 int kernel, int dilation, int padding, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(sequences) * length * kernel * channels;
    im2col_kernel<<<grid_for(count), THREADS, 0, stream>>>(input, columns, sequences, length, channels, kernel,
                                                            dilation, padding);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_audio_conv_transpose_gather(const float* products, const float* bias, float* output, int sequences,
                                                int input_length, int output_length, int channels, int kernel,
                                                int stride, int padding, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(sequences) * output_length * channels;
    conv_transpose_gather_kernel<<<grid_for(count), THREADS, 0, stream>>>(
        products, bias, output, sequences, input_length, output_length, channels, kernel, stride, padding);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_audio_snake_beta(const float* input, float* output, const float* parameters, const float* up_filter,
                                     const float* down_filter, int sequences, int length, int channels,
                                     cudaStream_t stream) {
    const size_t count = static_cast<size_t>(sequences) * length * channels;
    snake_beta_kernel<<<grid_for(count), THREADS, 0, stream>>>(input, output, parameters, up_filter, down_filter,
                                                                sequences, length, channels);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_audio_add(float* output, const float* first, const float* second, size_t count, float divisor,
                              cudaStream_t stream) {
    add_kernel<<<grid_for(count), THREADS, 0, stream>>>(output, first, second, count, divisor);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mmh3_audio_clamp(float* values, size_t count, float limit, cudaStream_t stream) {
    clamp_kernel<<<grid_for(count), THREADS, 0, stream>>>(values, count, limit);
    return static_cast<int>(cudaGetLastError());
}
