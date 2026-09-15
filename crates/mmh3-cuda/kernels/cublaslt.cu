#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <tuple>

// y[m, n] = x[m, k] · w[n, k]ᵀ + bias[n] through cuBLASLt, for the linear layers that stay in BF16
// or FP32.

namespace {

constexpr size_t WORKSPACE_BYTES = 32ull << 20;

int status_code(cublasStatus_t status) {
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + static_cast<int>(status);
}

// A handle and workspace for the calls of one host thread, or the status that made creating them
// fail.
// NOTE: an algorithm may run several kernels that pass partial results through the workspace. Calls
// from two threads interleave their kernels even on the legacy default stream, so a shared
// workspace lets one call read the partial results of another. The calls of one thread share its
// workspace, so they must use one stream.
struct ThreadState {
    cublasLtHandle_t handle = nullptr;
    void *workspace = nullptr;
    int status = 0;

    ThreadState() {
        status = status_code(cublasLtCreate(&handle));
        if (status != 0) {
            handle = nullptr;
            return;
        }
        status = static_cast<int>(cudaMalloc(&workspace, WORKSPACE_BYTES));
    }

    // cudaFree waits for the kernels that still use the workspace.
    ~ThreadState() {
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        if (handle != nullptr) {
            cublasLtDestroy(handle);
        }
    }

    ThreadState(const ThreadState &) = delete;
    ThreadState &operator=(const ThreadState &) = delete;
};

struct Descriptors {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t weight = nullptr;
    cublasLtMatrixLayout_t input = nullptr;
    cublasLtMatrixLayout_t output = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    ~Descriptors() {
        if (preference != nullptr) {
            cublasLtMatmulPreferenceDestroy(preference);
        }
        if (output != nullptr) {
            cublasLtMatrixLayoutDestroy(output);
        }
        if (input != nullptr) {
            cublasLtMatrixLayoutDestroy(input);
        }
        if (weight != nullptr) {
            cublasLtMatrixLayoutDestroy(weight);
        }
        if (operation != nullptr) {
            cublasLtMatmulDescDestroy(operation);
        }
    }
};

// Bumped whenever the descriptors of mmh3_cublaslt_nvfp4 change, so that algorithms chosen for the
// old ones are not taken over.
constexpr int NVFP4_ALGORITHM_FORMAT = 1;

using Shape = std::tuple<int64_t, int64_t, int64_t>;

// The algorithm chosen for each NVFP4 GEMM shape (m, n, k), shared by the threads of the process.
std::mutex nvfp4_algorithms_mutex;
std::map<Shape, cublasLtMatmulAlgo_t> nvfp4_algorithms;

} // namespace

// An algorithm chosen for an NVFP4 GEMM shape, see mmh3_cublaslt_nvfp4_algorithms.
struct Mmh3Nvfp4Algorithm {
    int64_t m;
    int64_t n;
    int64_t k;
    uint64_t data[8];
};

extern "C" const char *mmh3_cublaslt_status_string(int code) {
    return cublasLtGetStatusString(static_cast<cublasStatus_t>(code - 1000));
}

// output = alpha · input · weightᵀ + beta · output + bias.
// kind 0: BF16 input, weight, bias and output. kind 1: FP32 everywhere. kind 2: FP16 everywhere.
extern "C" int mmh3_cublaslt_matmul(int kind, const void *input, const void *weight,
                                    const void *bias, void *output, int64_t m, int64_t n, int64_t k,
                                    float alpha, float beta, cudaStream_t stream) {
    thread_local const ThreadState state;
    if (state.status != 0) {
        return state.status;
    }
    const cudaDataType_t data_type =
        kind == 0 ? CUDA_R_16BF : (kind == 1 ? CUDA_R_32F : CUDA_R_16F);
    Descriptors descriptors;
    cublasStatus_t status;
    if ((status = cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F,
                                           CUDA_R_32F)) != CUBLAS_STATUS_SUCCESS) {
        return status_code(status);
    }
    const cublasOperation_t transpose = CUBLAS_OP_T;
    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                   sizeof(transpose));
    cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &no_transpose, sizeof(no_transpose));
    if (bias != nullptr) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                       &epilogue, sizeof(epilogue));
        cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bias, sizeof(bias));
    }
    // Column-major view: D[n, m] = W^T[k, n]^T · X^T[k, m], which is row-major y[m, n].
    if ((status = cublasLtMatrixLayoutCreate(&descriptors.weight, data_type, k, n, k)) !=
            CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatrixLayoutCreate(&descriptors.input, data_type, k, m, k)) !=
            CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatrixLayoutCreate(&descriptors.output, data_type, n, m, n)) !=
            CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatmulPreferenceCreate(&descriptors.preference)) !=
            CUBLAS_STATUS_SUCCESS) {
        return status_code(status);
    }
    size_t workspace_bytes = WORKSPACE_BYTES;
    cublasLtMatmulPreferenceSetAttribute(descriptors.preference,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
                                         sizeof(workspace_bytes));
    cublasLtMatmulHeuristicResult_t result;
    int returned = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        state.handle, descriptors.operation, descriptors.weight, descriptors.input,
        descriptors.output, descriptors.output, descriptors.preference, 1, &result, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        return status_code(status == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_NOT_SUPPORTED : status);
    }
    status =
        cublasLtMatmul(state.handle, descriptors.operation, &alpha, weight, descriptors.weight,
                       input, descriptors.input, &beta, output, descriptors.output, output,
                       descriptors.output, &result.algo, state.workspace, WORKSPACE_BYTES, stream);
    return status_code(status);
}

extern "C" int mmh3_cublaslt_linear(int kind, const void *input, const void *weight,
                                    const void *bias, void *output, int64_t m, int64_t n, int64_t k,
                                    cudaStream_t stream) {
    return mmh3_cublaslt_matmul(kind, input, weight, bias, output, m, n, k, 1.0f, 0.0f, stream);
}

// output[m, n] = alpha · activations[m, k] · weightsᵀ in BF16 for NVFP4 operands laid out as in
// nvfp4.cu, with alpha and beta = 0 read from the device at `alpha_beta`.
extern "C" int mmh3_cublaslt_nvfp4(const void *weights, const void *weight_scales,
                                   const void *activations, const void *activation_scales,
                                   const float *alpha_beta, void *output, int64_t m, int64_t n,
                                   int64_t k, cudaStream_t stream) {
    thread_local const ThreadState state;
    if (state.status != 0) {
        return state.status;
    }
    Descriptors descriptors;
    cublasStatus_t status;
    if ((status = cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F,
                                           CUDA_R_32F)) != CUBLAS_STATUS_SUCCESS) {
        return status_code(status);
    }
    const cublasOperation_t transpose = CUBLAS_OP_T;
    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    const cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                   sizeof(transpose));
    cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &no_transpose, sizeof(no_transpose));
    cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                   &pointer_mode, sizeof(pointer_mode));
    cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                   &scale_mode, sizeof(scale_mode));
    cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                   &scale_mode, sizeof(scale_mode));
    cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                   &weight_scales, sizeof(weight_scales));
    cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                   &activation_scales, sizeof(activation_scales));
    // Column-major view: D[n, m] = W^T[k, n]^T · X^T[k, m], which is row-major y[m, n].
    if ((status = cublasLtMatrixLayoutCreate(&descriptors.weight, CUDA_R_4F_E2M1, k, n, k)) !=
            CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatrixLayoutCreate(&descriptors.input, CUDA_R_4F_E2M1, k, m, k)) !=
            CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatrixLayoutCreate(&descriptors.output, CUDA_R_16BF, n, m, n)) !=
            CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatmulPreferenceCreate(&descriptors.preference)) !=
            CUBLAS_STATUS_SUCCESS) {
        return status_code(status);
    }
    size_t workspace_bytes = WORKSPACE_BYTES;
    cublasLtMatmulPreferenceSetAttribute(descriptors.preference,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
                                         sizeof(workspace_bytes));
    // NOTE: the first heuristic result for the MLP down projection (K = 14,336) is a stream-K
    // kernel that takes 57 ms at 768p, against 16 ms for the fastest candidate. The first call of
    // each shape times the candidates on its own operands and keeps the fastest. One run each is
    // enough: a second run takes within 1% of the first, which includes loading the kernel.
    const Shape key = std::make_tuple(m, n, k);
    auto run = [&](const cublasLtMatmulAlgo_t *algorithm) {
        return cublasLtMatmul(state.handle, descriptors.operation, alpha_beta, weights,
                              descriptors.weight, activations, descriptors.input, alpha_beta + 1,
                              output, descriptors.output, output, descriptors.output, algorithm,
                              state.workspace, WORKSPACE_BYTES, stream);
    };
    const std::lock_guard<std::mutex> lock(nvfp4_algorithms_mutex);
    const auto found = nvfp4_algorithms.find(key);
    if (found != nvfp4_algorithms.end()) {
        status = run(&found->second);
        if (status == CUBLAS_STATUS_SUCCESS) {
            return 0;
        }
        // An algorithm taken over from an earlier process may not run here.
        nvfp4_algorithms.erase(found);
    }
    constexpr int CANDIDATES = 8;
    cublasLtMatmulHeuristicResult_t results[CANDIDATES];
    int returned = 0;
    status =
        cublasLtMatmulAlgoGetHeuristic(state.handle, descriptors.operation, descriptors.weight,
                                       descriptors.input, descriptors.output, descriptors.output,
                                       descriptors.preference, CANDIDATES, results, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        return status_code(status == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_NOT_SUPPORTED : status);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float best = 0.0f;
    int best_index = -1;
    for (int index = 0; index < returned; index++) {
        cudaEventRecord(start, stream);
        const cublasStatus_t timed = run(&results[index].algo);
        cudaEventRecord(stop, stream);
        float elapsed = 0.0f;
        if (timed != CUBLAS_STATUS_SUCCESS || cudaEventSynchronize(stop) != cudaSuccess ||
            cudaEventElapsedTime(&elapsed, start, stop) != cudaSuccess) {
            continue;
        }
        if (best_index < 0 || elapsed < best) {
            best = elapsed;
            best_index = index;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (best_index < 0) {
        return status_code(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    nvfp4_algorithms[key] = results[best_index].algo;
    return status_code(run(&results[best_index].algo));
}

// Copies up to `capacity` of the algorithms chosen for NVFP4 GEMM shapes to `algorithms` and
// returns how many there are.
extern "C" int mmh3_cublaslt_nvfp4_algorithms(Mmh3Nvfp4Algorithm *algorithms, int capacity) {
    const std::lock_guard<std::mutex> lock(nvfp4_algorithms_mutex);
    int index = 0;
    for (const auto &[shape, algorithm] : nvfp4_algorithms) {
        if (index < capacity) {
            algorithms[index].m = std::get<0>(shape);
            algorithms[index].n = std::get<1>(shape);
            algorithms[index].k = std::get<2>(shape);
            for (int word = 0; word < 8; word++) {
                algorithms[index].data[word] = algorithm.data[word];
            }
        }
        index++;
    }
    return index;
}

// Takes over algorithms for NVFP4 GEMM shapes that have none chosen yet.
extern "C" void mmh3_cublaslt_nvfp4_adopt(const Mmh3Nvfp4Algorithm *algorithms, int count) {
    const std::lock_guard<std::mutex> lock(nvfp4_algorithms_mutex);
    for (int index = 0; index < count; index++) {
        cublasLtMatmulAlgo_t algorithm;
        for (int word = 0; word < 8; word++) {
            algorithm.data[word] = algorithms[index].data[word];
        }
        nvfp4_algorithms.emplace(
            std::make_tuple(algorithms[index].m, algorithms[index].n, algorithms[index].k),
            algorithm);
    }
}

// Writes a line naming what the chosen NVFP4 algorithms depend on, the cuBLASLt version, the GPU
// and the descriptors, to `key`, which holds `capacity` bytes.
extern "C" int mmh3_cublaslt_nvfp4_algorithm_key(char *key, int capacity) {
    int device = 0;
    cudaDeviceProp properties;
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess) {
        status = cudaGetDeviceProperties(&properties, device);
    }
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    const int written = std::snprintf(
        key, static_cast<size_t>(capacity), "cuBLASLt %zu, %s, sm_%d%d, %d SMs, format %d",
        cublasLtGetVersion(), properties.name, properties.major, properties.minor,
        properties.multiProcessorCount, NVFP4_ALGORITHM_FORMAT);
    return written < 0 || written >= capacity ? static_cast<int>(cudaErrorInvalidValue) : 0;
}
