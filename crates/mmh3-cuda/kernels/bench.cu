#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" int mmh3_attention_bf16(const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
                                   __nv_bfloat16* output, int tokens, int heads, int64_t query_stride,
                                   int64_t key_stride, int64_t value_stride, int64_t output_stride, float scale,
                                   cudaStream_t stream);
extern "C" int mmh3_int8_gemm_config_count();
extern "C" int mmh3_int8_gemm_bf16(int config, const int8_t* activations, const int8_t* weights,
                                   const float* activation_scales, const float* weight_scales, __nv_bfloat16* output,
                                   int m, int n, int k, cudaStream_t stream);

namespace {

enum GemmKind : int {
    GEMM_KIND_BF16 = 0,
    GEMM_KIND_FP8_E4M3 = 1,
    GEMM_KIND_INT8 = 2,
    GEMM_KIND_NVFP4 = 3,
};

constexpr size_t WORKSPACE_BYTES = 64ull << 20;
constexpr int REQUESTED_ALGORITHMS = 8;
constexpr int WARMUP_RUNS = 2;

struct DeviceMemory {
    void* pointer = nullptr;
    ~DeviceMemory() {
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
    }
};

struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ~EventPair() {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
    }
};

struct LtObjects {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t layout_a = nullptr;
    cublasLtMatrixLayout_t layout_b = nullptr;
    cublasLtMatrixLayout_t layout_d = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    ~LtObjects() {
        if (preference != nullptr) {
            cublasLtMatmulPreferenceDestroy(preference);
        }
        if (layout_d != nullptr) {
            cublasLtMatrixLayoutDestroy(layout_d);
        }
        if (layout_b != nullptr) {
            cublasLtMatrixLayoutDestroy(layout_b);
        }
        if (layout_a != nullptr) {
            cublasLtMatrixLayoutDestroy(layout_a);
        }
        if (operation != nullptr) {
            cublasLtMatmulDescDestroy(operation);
        }
        if (handle != nullptr) {
            cublasLtDestroy(handle);
        }
    }
};

void write_message(char* message, size_t message_size, const char* text) {
    if (message != nullptr && message_size > 0) {
        std::snprintf(message, message_size, "%s", text);
    }
}

int cuda_failure(cudaError_t status, const char* what, char* message, size_t message_size) {
    char text[256];
    std::snprintf(text, sizeof(text), "%s: %s", what, cudaGetErrorString(status));
    write_message(message, message_size, text);
    return static_cast<int>(status);
}

int lt_failure(cublasStatus_t status, const char* what, char* message, size_t message_size) {
    char text[256];
    std::snprintf(text, sizeof(text), "%s: %s", what, cublasLtGetStatusString(status));
    write_message(message, message_size, text);
    return 1000 + static_cast<int>(status);
}

__global__ void fill_pattern_kernel(uint8_t* data, size_t bytes, uint32_t seed) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < bytes; index += stride) {
        uint32_t hash = static_cast<uint32_t>(index) * 2654435761u ^ seed;
        hash ^= hash >> 15;
        data[index] = static_cast<uint8_t>(hash & 0x3F);
    }
}

cudaError_t fill_pattern(void* data, size_t bytes, uint32_t seed) {
    fill_pattern_kernel<<<1024, 256>>>(static_cast<uint8_t*>(data), bytes, seed);
    return cudaGetLastError();
}

size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

__global__ void copy_kernel(const uint4* source, uint4* destination, size_t count) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        destination[index] = source[index];
    }
}

}  // namespace

// Times y[m, n] = x[m, k] * w[n, k]^T through cuBLASLt and reports the fastest heuristic algorithm.
extern "C" int mmh3_bench_cublaslt_gemm(int kind, int64_t m, int64_t n, int64_t k, int iterations,
                                        float* best_milliseconds, int* algorithm_count, char* message,
                                        size_t message_size) {
    *best_milliseconds = 0.0f;
    *algorithm_count = 0;

    cudaDataType_t input_type;
    cudaDataType_t output_type = CUDA_R_16BF;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    size_t output_element_bytes = 2;
    switch (kind) {
        case GEMM_KIND_BF16:
            input_type = CUDA_R_16BF;
            break;
        case GEMM_KIND_FP8_E4M3:
            input_type = CUDA_R_8F_E4M3;
            break;
        case GEMM_KIND_INT8:
            input_type = CUDA_R_8I;
            output_type = CUDA_R_32I;
            compute_type = CUBLAS_COMPUTE_32I;
            scale_type = CUDA_R_32I;
            output_element_bytes = 4;
            break;
        case GEMM_KIND_NVFP4:
            input_type = CUDA_R_4F_E2M1;
            break;
        default:
            write_message(message, message_size, "unknown GEMM kind");
            return -1;
    }

    size_t weight_bytes;
    size_t activation_bytes;
    if (kind == GEMM_KIND_NVFP4) {
        weight_bytes = static_cast<size_t>(n) * k / 2;
        activation_bytes = static_cast<size_t>(m) * k / 2;
    } else {
        size_t input_element_bytes = kind == GEMM_KIND_BF16 ? 2 : 1;
        weight_bytes = static_cast<size_t>(n) * k * input_element_bytes;
        activation_bytes = static_cast<size_t>(m) * k * input_element_bytes;
    }
    size_t output_bytes = static_cast<size_t>(m) * n * output_element_bytes;

    DeviceMemory weight, activation, output, workspace, weight_scales, activation_scales;
    cudaError_t cuda_status;
    if ((cuda_status = cudaMalloc(&weight.pointer, weight_bytes)) != cudaSuccess ||
        (cuda_status = cudaMalloc(&activation.pointer, activation_bytes)) != cudaSuccess ||
        (cuda_status = cudaMalloc(&output.pointer, output_bytes)) != cudaSuccess ||
        (cuda_status = cudaMalloc(&workspace.pointer, WORKSPACE_BYTES)) != cudaSuccess) {
        return cuda_failure(cuda_status, "cudaMalloc", message, message_size);
    }
    if ((cuda_status = fill_pattern(weight.pointer, weight_bytes, 1)) != cudaSuccess ||
        (cuda_status = fill_pattern(activation.pointer, activation_bytes, 2)) != cudaSuccess) {
        return cuda_failure(cuda_status, "fill", message, message_size);
    }

    LtObjects lt;
    cublasStatus_t status;
    if ((status = cublasLtCreate(&lt.handle)) != CUBLAS_STATUS_SUCCESS) {
        return lt_failure(status, "cublasLtCreate", message, message_size);
    }
    if ((status = cublasLtMatmulDescCreate(&lt.operation, compute_type, scale_type)) != CUBLAS_STATUS_SUCCESS) {
        return lt_failure(status, "cublasLtMatmulDescCreate", message, message_size);
    }
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasOperation_t no_transpose = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(lt.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
    cublasLtMatmulDescSetAttribute(lt.operation, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose, sizeof(no_transpose));

    if (kind == GEMM_KIND_NVFP4) {
        // NOTE: block scales use cuBLASLt's swizzled layout, which needs 128-row by 4-column padding.
        // Their values do not affect timing, so the buffers are only sized and filled.
        size_t weight_scale_bytes = round_up(n, 128) * round_up(k / 16, 4);
        size_t activation_scale_bytes = round_up(m, 128) * round_up(k / 16, 4);
        if ((cuda_status = cudaMalloc(&weight_scales.pointer, weight_scale_bytes)) != cudaSuccess ||
            (cuda_status = cudaMalloc(&activation_scales.pointer, activation_scale_bytes)) != cudaSuccess) {
            return cuda_failure(cuda_status, "cudaMalloc scales", message, message_size);
        }
        if ((cuda_status = cudaMemset(weight_scales.pointer, 0x38, weight_scale_bytes)) != cudaSuccess ||
            (cuda_status = cudaMemset(activation_scales.pointer, 0x38, activation_scale_bytes)) != cudaSuccess) {
            return cuda_failure(cuda_status, "cudaMemset scales", message, message_size);
        }
        cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulDescSetAttribute(lt.operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
        cublasLtMatmulDescSetAttribute(lt.operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
        cublasLtMatmulDescSetAttribute(lt.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scales.pointer,
                                       sizeof(weight_scales.pointer));
        cublasLtMatmulDescSetAttribute(lt.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &activation_scales.pointer,
                                       sizeof(activation_scales.pointer));
    }

    // Column-major view: D[n, m] = W^T[k, n]^T * X^T[k, m], which is row-major y[m, n].
    if ((status = cublasLtMatrixLayoutCreate(&lt.layout_a, input_type, k, n, k)) != CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatrixLayoutCreate(&lt.layout_b, input_type, k, m, k)) != CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatrixLayoutCreate(&lt.layout_d, output_type, n, m, n)) != CUBLAS_STATUS_SUCCESS) {
        return lt_failure(status, "cublasLtMatrixLayoutCreate", message, message_size);
    }
    if ((status = cublasLtMatmulPreferenceCreate(&lt.preference)) != CUBLAS_STATUS_SUCCESS) {
        return lt_failure(status, "cublasLtMatmulPreferenceCreate", message, message_size);
    }
    size_t workspace_bytes = WORKSPACE_BYTES;
    cublasLtMatmulPreferenceSetAttribute(lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
                                         sizeof(workspace_bytes));

    cublasLtMatmulHeuristicResult_t results[REQUESTED_ALGORITHMS];
    int returned = 0;
    status = cublasLtMatmulAlgoGetHeuristic(lt.handle, lt.operation, lt.layout_a, lt.layout_b, lt.layout_d, lt.layout_d,
                                            lt.preference, REQUESTED_ALGORITHMS, results, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        return lt_failure(status == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_NOT_SUPPORTED : status,
                          "no cuBLASLt algorithm for this configuration", message, message_size);
    }

    float alpha_float = 1.0f, beta_float = 0.0f;
    int32_t alpha_int = 1, beta_int = 0;
    const void* alpha = kind == GEMM_KIND_INT8 ? static_cast<const void*>(&alpha_int) : &alpha_float;
    const void* beta = kind == GEMM_KIND_INT8 ? static_cast<const void*>(&beta_int) : &beta_float;

    EventPair events;
    cudaEventCreate(&events.start);
    cudaEventCreate(&events.stop);
    float best = 0.0f;
    int timed = 0;
    for (int index = 0; index < returned; index++) {
        const cublasLtMatmulAlgo_t* algorithm = &results[index].algo;
        bool usable = true;
        for (int run = 0; run < WARMUP_RUNS && usable; run++) {
            usable = cublasLtMatmul(lt.handle, lt.operation, alpha, weight.pointer, lt.layout_a, activation.pointer,
                                    lt.layout_b, beta, output.pointer, lt.layout_d, output.pointer, lt.layout_d,
                                    algorithm, workspace.pointer, WORKSPACE_BYTES, nullptr) == CUBLAS_STATUS_SUCCESS;
        }
        if (!usable || cudaDeviceSynchronize() != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        cudaEventRecord(events.start);
        for (int run = 0; run < iterations; run++) {
            cublasLtMatmul(lt.handle, lt.operation, alpha, weight.pointer, lt.layout_a, activation.pointer, lt.layout_b,
                           beta, output.pointer, lt.layout_d, output.pointer, lt.layout_d, algorithm,
                           workspace.pointer, WORKSPACE_BYTES, nullptr);
        }
        cudaEventRecord(events.stop);
        if ((cuda_status = cudaEventSynchronize(events.stop)) != cudaSuccess) {
            return cuda_failure(cuda_status, "cublasLtMatmul", message, message_size);
        }
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, events.start, events.stop);
        float average = elapsed / iterations;
        if (timed == 0 || average < best) {
            best = average;
        }
        timed++;
    }
    if (timed == 0) {
        write_message(message, message_size, "every cuBLASLt algorithm failed to run");
        return -1;
    }
    *best_milliseconds = best;
    *algorithm_count = timed;
    return 0;
}

// Measures device memory copy throughput with a vectorized kernel and with cudaMemcpy.
extern "C" int mmh3_bench_memory_copy(size_t bytes, int iterations, float* kernel_gigabytes_per_second,
                                      float* memcpy_gigabytes_per_second, char* message, size_t message_size) {
    bytes = bytes / sizeof(uint4) * sizeof(uint4);
    DeviceMemory source, destination;
    cudaError_t status;
    if ((status = cudaMalloc(&source.pointer, bytes)) != cudaSuccess ||
        (status = cudaMalloc(&destination.pointer, bytes)) != cudaSuccess) {
        return cuda_failure(status, "cudaMalloc", message, message_size);
    }
    if ((status = fill_pattern(source.pointer, bytes, 3)) != cudaSuccess) {
        return cuda_failure(status, "fill", message, message_size);
    }
    int multiprocessors = 0;
    cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0);
    size_t count = bytes / sizeof(uint4);
    unsigned blocks = static_cast<unsigned>(multiprocessors * 16);

    EventPair events;
    cudaEventCreate(&events.start);
    cudaEventCreate(&events.stop);
    float elapsed = 0.0f;

    copy_kernel<<<blocks, 256>>>(static_cast<const uint4*>(source.pointer), static_cast<uint4*>(destination.pointer), count);
    cudaEventRecord(events.start);
    for (int run = 0; run < iterations; run++) {
        copy_kernel<<<blocks, 256>>>(static_cast<const uint4*>(source.pointer), static_cast<uint4*>(destination.pointer),
                                     count);
    }
    cudaEventRecord(events.stop);
    if ((status = cudaEventSynchronize(events.stop)) != cudaSuccess) {
        return cuda_failure(status, "copy kernel", message, message_size);
    }
    cudaEventElapsedTime(&elapsed, events.start, events.stop);
    *kernel_gigabytes_per_second = static_cast<float>(2.0 * bytes * iterations / (elapsed * 1e-3) / 1e9);

    cudaMemcpy(destination.pointer, source.pointer, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(events.start);
    for (int run = 0; run < iterations; run++) {
        cudaMemcpyAsync(destination.pointer, source.pointer, bytes, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(events.stop);
    if ((status = cudaEventSynchronize(events.stop)) != cudaSuccess) {
        return cuda_failure(status, "cudaMemcpy", message, message_size);
    }
    cudaEventElapsedTime(&elapsed, events.start, events.stop);
    *memcpy_gigabytes_per_second = static_cast<float>(2.0 * bytes * iterations / (elapsed * 1e-3) / 1e9);
    return 0;
}

// Times the mmh3 INT8 GEMM kernel over every tile configuration and reports the fastest one.
extern "C" int mmh3_bench_int8_gemm(int64_t m, int64_t n, int64_t k, int iterations, float* best_milliseconds,
                                    int* configs_timed, int* best_config, char* message, size_t message_size) {
    *best_milliseconds = 0.0f;
    *configs_timed = 0;
    *best_config = -1;
    DeviceMemory activations, weights, activation_scales, weight_scales, output;
    cudaError_t status;
    if ((status = cudaMalloc(&activations.pointer, m * k)) != cudaSuccess ||
        (status = cudaMalloc(&weights.pointer, n * k)) != cudaSuccess ||
        (status = cudaMalloc(&activation_scales.pointer, m * sizeof(float))) != cudaSuccess ||
        (status = cudaMalloc(&weight_scales.pointer, n * sizeof(float))) != cudaSuccess ||
        (status = cudaMalloc(&output.pointer, m * n * sizeof(__nv_bfloat16))) != cudaSuccess) {
        return cuda_failure(status, "cudaMalloc", message, message_size);
    }
    if ((status = fill_pattern(activations.pointer, m * k, 4)) != cudaSuccess ||
        (status = fill_pattern(weights.pointer, n * k, 5)) != cudaSuccess ||
        (status = fill_pattern(activation_scales.pointer, m * sizeof(float), 6)) != cudaSuccess ||
        (status = fill_pattern(weight_scales.pointer, n * sizeof(float), 7)) != cudaSuccess) {
        return cuda_failure(status, "fill", message, message_size);
    }

    EventPair events;
    cudaEventCreate(&events.start);
    cudaEventCreate(&events.stop);
    auto run = [&](int config) {
        return mmh3_int8_gemm_bf16(config, static_cast<const int8_t*>(activations.pointer),
                                   static_cast<const int8_t*>(weights.pointer),
                                   static_cast<const float*>(activation_scales.pointer),
                                   static_cast<const float*>(weight_scales.pointer),
                                   static_cast<__nv_bfloat16*>(output.pointer), static_cast<int>(m),
                                   static_cast<int>(n), static_cast<int>(k), nullptr);
    };
    for (int config = 0; config < mmh3_int8_gemm_config_count(); config++) {
        bool usable = true;
        for (int warmup = 0; warmup < WARMUP_RUNS && usable; warmup++) {
            usable = run(config) == 0;
        }
        if (!usable || cudaDeviceSynchronize() != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        cudaEventRecord(events.start);
        for (int index = 0; index < iterations; index++) {
            run(config);
        }
        cudaEventRecord(events.stop);
        if ((status = cudaEventSynchronize(events.stop)) != cudaSuccess) {
            return cuda_failure(status, "int8 GEMM", message, message_size);
        }
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, events.start, events.stop);
        float average = elapsed / iterations;
        if (*configs_timed == 0 || average < *best_milliseconds) {
            *best_milliseconds = average;
            *best_config = config;
        }
        (*configs_timed)++;
    }
    if (*configs_timed == 0) {
        write_message(message, message_size, "every INT8 GEMM configuration failed to run");
        return -1;
    }
    return 0;
}

// Times dense attention over a fused [tokens, 3, heads, 128] qkv buffer.
extern "C" int mmh3_bench_attention(int tokens, int heads, int iterations, float* milliseconds, char* message,
                                    size_t message_size) {
    const int64_t inner = static_cast<int64_t>(heads) * 128;
    DeviceMemory qkv, output;
    cudaError_t status;
    if ((status = cudaMalloc(&qkv.pointer, tokens * 3 * inner * sizeof(__nv_bfloat16))) != cudaSuccess ||
        (status = cudaMalloc(&output.pointer, tokens * inner * sizeof(__nv_bfloat16))) != cudaSuccess) {
        return cuda_failure(status, "cudaMalloc", message, message_size);
    }
    if ((status = fill_pattern(qkv.pointer, tokens * 3 * inner * sizeof(__nv_bfloat16), 8)) != cudaSuccess) {
        return cuda_failure(status, "fill", message, message_size);
    }
    const __nv_bfloat16* base = static_cast<const __nv_bfloat16*>(qkv.pointer);
    auto run = [&]() {
        return mmh3_attention_bf16(base, base + inner, base + 2 * inner, static_cast<__nv_bfloat16*>(output.pointer),
                                   tokens, heads, 3 * inner, 3 * inner, 3 * inner, inner, 0.08838834764831845f,
                                   nullptr);
    };
    int code = run();
    if (code != 0 || (status = cudaDeviceSynchronize()) != cudaSuccess) {
        return cuda_failure(code != 0 ? static_cast<cudaError_t>(code) : status, "attention", message, message_size);
    }
    EventPair events;
    cudaEventCreate(&events.start);
    cudaEventCreate(&events.stop);
    cudaEventRecord(events.start);
    for (int index = 0; index < iterations; index++) {
        run();
    }
    cudaEventRecord(events.stop);
    if ((status = cudaEventSynchronize(events.stop)) != cudaSuccess) {
        return cuda_failure(status, "attention", message, message_size);
    }
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, events.start, events.stop);
    *milliseconds = elapsed / iterations;
    return 0;
}
