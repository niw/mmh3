#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Kernels of the latent upscaler that the video encoder's do not cover. Activations are FP16 and
// pixel-major, [frames, height, width, channels].

namespace {

constexpr int THREADS = 256;

unsigned grid_for(size_t count) {
    size_t blocks = (count + THREADS - 1) / THREADS;
    return static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
}

// output[t, p, c] = bias[c] + sum over k of weight[c, k] · input[t + k − kernel / 2, p, c], with
// zeros past either end of the clip.
__global__ void temporal_kernel(const __half *__restrict__ input, int frames, int pixels,
                                int channels, const __half *__restrict__ weight, int kernel,
                                const __half *__restrict__ bias, __half *__restrict__ output) {
    const size_t frame_values = static_cast<size_t>(pixels) * channels;
    const size_t count = static_cast<size_t>(frames) * frame_values;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const int frame = static_cast<int>(index / frame_values);
        float sum = __half2float(bias[channel]);
        for (int tap = 0; tap < kernel; tap++) {
            const int source = frame + tap - kernel / 2;
            if (source < 0 || source >= frames) {
                continue;
            }
            sum = fmaf(__half2float(weight[channel * kernel + tap]),
                       __half2float(input[index + (static_cast<ptrdiff_t>(source) - frame) *
                                                      static_cast<ptrdiff_t>(frame_values)]),
                       sum);
        }
        output[index] = __float2half_rn(sum);
    }
}

// A source coordinate of output coordinate `index` with the pixel centers aligned, as
// align_corners=False aligns them: the lower neighbour, the upper one and the upper one's weight.
__device__ __forceinline__ void taps(int index, int size, int source_size, int &low, int &high,
                                     float &weight) {
    const float position =
        fmaxf((index + 0.5f) * static_cast<float>(source_size) / size - 0.5f, 0.0f);
    low = min(static_cast<int>(position), source_size - 1);
    high = min(low + 1, source_size - 1);
    weight = position - low;
}

__global__ void resize_kernel(const __half *__restrict__ input, int frames, int height, int width,
                              int channels, int output_height, int output_width,
                              __half *__restrict__ output) {
    const size_t count =
        static_cast<size_t>(frames) * output_height * output_width * channels;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const size_t pixel = index / channels;
        const int x = static_cast<int>(pixel % output_width);
        const int y = static_cast<int>(pixel / output_width % output_height);
        const int frame = static_cast<int>(pixel / (static_cast<size_t>(output_width) * output_height));
        int top = 0, bottom = 0, left = 0, right = 0;
        float down = 0.0f, across = 0.0f;
        taps(y, output_height, height, top, bottom, down);
        taps(x, output_width, width, left, right, across);
        const __half *plane = input + static_cast<size_t>(frame) * height * width * channels;
        auto at = [&](int row, int column) {
            return __half2float(plane[(static_cast<size_t>(row) * width + column) * channels + channel]);
        };
        const float upper = at(top, left) * (1.0f - across) + at(top, right) * across;
        const float lower = at(bottom, left) * (1.0f - across) + at(bottom, right) * across;
        output[index] = __float2half_rn(upper * (1.0f - down) + lower * down);
    }
}

} // namespace

// A depthwise convolution along time with `kernel` taps per channel, weight [channels, kernel], and
// zeros past either end of the clip.
extern "C" int mmh3_latent_upscaler_temporal(const __half *input, int frames, int pixels,
                                             int channels, const __half *weight, int kernel,
                                             const __half *bias, __half *output,
                                             cudaStream_t stream) {
    if (frames <= 0 || pixels <= 0 || channels <= 0 || kernel <= 0 || kernel % 2 == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t count = static_cast<size_t>(frames) * pixels * channels;
    temporal_kernel<<<grid_for(count), THREADS, 0, stream>>>(input, frames, pixels, channels,
                                                             weight, kernel, bias, output);
    return static_cast<int>(cudaGetLastError());
}

// Bilinear resize of every frame to `output_height × output_width`, which is the trilinear resize
// of the reference when the frames keep their number.
extern "C" int mmh3_latent_upscaler_resize(const __half *input, int frames, int height, int width,
                                           int channels, int output_height, int output_width,
                                           __half *output, cudaStream_t stream) {
    if (frames <= 0 || height <= 0 || width <= 0 || channels <= 0 || output_height <= 0 ||
        output_width <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t count = static_cast<size_t>(frames) * output_height * output_width * channels;
    resize_kernel<<<grid_for(count), THREADS, 0, stream>>>(input, frames, height, width, channels,
                                                           output_height, output_width, output);
    return static_cast<int>(cudaGetLastError());
}
