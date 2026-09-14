#include <cstddef>
#include <cstdint>
#include <cublasLt.h>
#include <cuda_runtime.h>

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

} // namespace

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
