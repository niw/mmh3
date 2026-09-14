#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Kernels of the Qwen3 text encoder that the DiT's kernels do not cover.

namespace {

constexpr int THREADS = 256;

// output[token, :] = FP32 of table[ids[token], :].
__global__ void embedding_kernel(const int32_t *__restrict__ ids,
                                 const __nv_bfloat16 *__restrict__ table,
                                 float *__restrict__ output, int hidden) {
    const __nv_bfloat16 *row = table + static_cast<size_t>(ids[blockIdx.x]) * hidden;
    float *output_row = output + static_cast<size_t>(blockIdx.x) * hidden;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        output_row[index] = __bfloat162float(row[index]);
    }
}

} // namespace

extern "C" int mmh3_embedding_bf16(const int32_t *ids, const __nv_bfloat16 *table, float *output,
                                   int tokens, int hidden, cudaStream_t stream) {
    if (tokens <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    embedding_kernel<<<tokens, THREADS, 0, stream>>>(ids, table, output, hidden);
    return static_cast<int>(cudaGetLastError());
}
