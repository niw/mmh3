#include <atomic>
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

// A handle and workspace for the calls of one host thread.
// NOTE: an algorithm may run several kernels that pass partial results through the workspace. Calls
// from two threads interleave their kernels even on the legacy default stream, so a shared
// workspace lets one call read the partial results of another. The calls of one thread share its
// workspace, so they must use one stream.
struct ThreadState {
    cublasLtHandle_t handle = nullptr;
    void *workspace = nullptr;

    ThreadState() = default;

    // Makes what is still missing, or says why it could not. A device that had no memory left may
    // have some after its caller lets go of a model, so a failure is not kept for the next call,
    // and the failed allocation is cleared from the last error, where the next kernel launch
    // would read it as its own.
    int prepare() {
        if (handle == nullptr) {
            const int status = status_code(cublasLtCreate(&handle));
            if (status != 0) {
                handle = nullptr;
                return status;
            }
        }
        if (workspace == nullptr) {
            const cudaError_t status = cudaMalloc(&workspace, WORKSPACE_BYTES);
            if (status != cudaSuccess) {
                workspace = nullptr;
                cudaGetLastError();
                return static_cast<int>(status);
            }
        }
        return 0;
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

// Bumped whenever the descriptors of mmh3_cublaslt_nvfp4 or the rule that picks between its
// candidates change, so that algorithms chosen under the old ones are not taken over.
constexpr int NVFP4_ALGORITHM_FORMAT = 2;
// The same for the ordinary GEMM shapes.
constexpr int MATMUL_ALGORITHM_FORMAT = 2;

using Shape = std::tuple<int64_t, int64_t, int64_t>;

// NOTE: the tables of chosen algorithms below and their locks are made once and never destroyed. A
// worker's session thread saves them when its session ends, which can be while the process is
// already exiting, and a table the exit handlers destroyed under it would crash the process after
// its work is done.

// The algorithm chosen for each NVFP4 GEMM shape (m, n, k), shared by the threads of the process.
std::mutex &nvfp4_algorithms_mutex = *new std::mutex;
std::map<Shape, cublasLtMatmulAlgo_t> &nvfp4_algorithms =
    *new std::map<Shape, cublasLtMatmulAlgo_t>;

// The same for the plain GEMMs, keyed by kind, shape and whether a bias is added.
using MatmulKey = std::tuple<int, int64_t, int64_t, int64_t, bool>;
std::mutex &matmul_algorithms_mutex = *new std::mutex;
std::map<MatmulKey, cublasLtMatmulAlgo_t> &matmul_algorithms =
    *new std::map<MatmulKey, cublasLtMatmulAlgo_t>;

// How many heuristic candidates a new shape times against each other. cuBLASLt offers only a few
// for the convolution shapes of the VAE encoder, so the list is as long as it will fill.
constexpr int MATMUL_CANDIDATES = 32;
constexpr int NVFP4_CANDIDATES = 8;

// NOTE: two algorithms round differently, and cuBLASLt picks one by the shape, M included. A timed
// choice also moves with the noise of the measurement. So without consistency the same rows come
// out differently when a step is split across ranks, in two processes, or on two machines. With
// it, a GEMM runs the one algorithm cuBLASLt's heuristic names for its operands at CONSISTENT_ROWS
// rows, whatever its own M is, without a split of K, and nothing is timed. Every row then goes
// through the same kernel in the same order.
constexpr int64_t CONSISTENT_ROWS = 4096;

// Consistency for every thread that has not been told otherwise, and for this thread. -1 follows
// the process.
std::atomic<bool> consistent_by_default{false};
thread_local int consistent_here = -1;

bool consistent() {
    return consistent_here >= 0 ? consistent_here != 0
                                : consistent_by_default.load(std::memory_order_relaxed);
}

// The algorithm consistency runs for each kind, n, k and bias of the plain GEMMs, and for each n
// and k of the NVFP4 ones. It is derived rather than measured, so it is not kept between runs.
using ConsistentKey = std::tuple<int, int64_t, int64_t, bool>;
std::mutex &consistent_mutex = *new std::mutex;
std::map<ConsistentKey, cublasLtMatmulAlgo_t> &consistent_matmul =
    *new std::map<ConsistentKey, cublasLtMatmulAlgo_t>;
std::map<std::tuple<int64_t, int64_t>, cublasLtMatmulAlgo_t> &consistent_nvfp4 =
    *new std::map<std::tuple<int64_t, int64_t>, cublasLtMatmulAlgo_t>;

// Asks the heuristic for the algorithm of `operation` over `weight` at CONSISTENT_ROWS rows.
cublasStatus_t consistent_algorithm(cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
                                    cublasLtMatrixLayout_t weight, cudaDataType_t input_type,
                                    cudaDataType_t output_type, int64_t n, int64_t k,
                                    cublasLtMatmulPreference_t preference,
                                    cublasLtMatmulAlgo_t *algorithm) {
    cublasLtMatrixLayout_t input = nullptr;
    cublasLtMatrixLayout_t output = nullptr;
    cublasStatus_t status = cublasLtMatrixLayoutCreate(&input, input_type, k, CONSISTENT_ROWS, k);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&output, output_type, n, CONSISTENT_ROWS, n);
    }
    cublasLtMatmulHeuristicResult_t result;
    int returned = 0;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulAlgoGetHeuristic(handle, operation, weight, input, output, output,
                                                preference, 1, &result, &returned);
    }
    if (status == CUBLAS_STATUS_SUCCESS && returned == 0) {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        *algorithm = result.algo;
    }
    if (output != nullptr) {
        cublasLtMatrixLayoutDestroy(output);
    }
    if (input != nullptr) {
        cublasLtMatrixLayoutDestroy(input);
    }
    return status;
}

// Runs `run` with the algorithm consistency takes for `table[key]`, choosing it on the first call.
// A call whose own shape that algorithm cannot run takes the heuristic's first answer for its own
// shape, which still does not move from run to run.
template <typename Key, typename Choose, typename Run>
cublasStatus_t run_consistently(std::map<Key, cublasLtMatmulAlgo_t> &table, const Key &key,
                                cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
                                cublasLtMatrixLayout_t weight, cublasLtMatrixLayout_t input,
                                cublasLtMatrixLayout_t output,
                                cublasLtMatmulPreference_t preference, Choose choose, Run run) {
    const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                         &reduction, sizeof(reduction));
    cublasLtMatmulAlgo_t algorithm;
    {
        const std::lock_guard<std::mutex> lock(consistent_mutex);
        const auto found = table.find(key);
        if (found != table.end()) {
            algorithm = found->second;
        } else {
            const cublasStatus_t status = choose(&algorithm);
            if (status != CUBLAS_STATUS_SUCCESS) {
                return status;
            }
            table.emplace(key, algorithm);
        }
    }
    cublasLtMatmulHeuristicResult_t checked;
    if (cublasLtMatmulAlgoCheck(handle, operation, weight, input, output, output, &algorithm,
                                &checked) != CUBLAS_STATUS_SUCCESS ||
        checked.workspaceSize > WORKSPACE_BYTES) {
        int returned = 0;
        const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            handle, operation, weight, input, output, output, preference, 1, &checked, &returned);
        if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
            return status == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_NOT_SUPPORTED : status;
        }
        algorithm = checked.algo;
    }
    return run(&algorithm);
}

} // namespace

// An algorithm chosen for an NVFP4 GEMM shape, see mmh3_cublaslt_nvfp4_algorithms.
struct Mmh3Nvfp4Algorithm {
    int64_t m;
    int64_t n;
    int64_t k;
    uint64_t data[8];
};

// An algorithm chosen for an ordinary GEMM shape, see mmh3_cublaslt_matmul_algorithms. The fields
// before the words are the key of the choice: the kind of the operands, whether the call adds a
// bias, and the shape.
struct Mmh3MatmulAlgorithm {
    int64_t kind;
    int64_t bias;
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
    thread_local ThreadState state;
    if (const int prepared = state.prepare(); prepared != 0) {
        return prepared;
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
    auto run = [&](const cublasLtMatmulAlgo_t *algorithm) {
        return cublasLtMatmul(state.handle, descriptors.operation, &alpha, weight,
                              descriptors.weight, input, descriptors.input, &beta, output,
                              descriptors.output, output, descriptors.output, algorithm,
                              state.workspace, WORKSPACE_BYTES, stream);
    };
    if (consistent()) {
        return status_code(run_consistently(
            consistent_matmul, std::make_tuple(kind, n, k, bias != nullptr), state.handle,
            descriptors.operation, descriptors.weight, descriptors.input, descriptors.output,
            descriptors.preference,
            [&](cublasLtMatmulAlgo_t *algorithm) {
                return consistent_algorithm(state.handle, descriptors.operation, descriptors.weight,
                                            data_type, data_type, n, k, descriptors.preference,
                                            algorithm);
            },
            run));
    }
    // NOTE: the first heuristic result is often a small-tile kernel that runs at a fraction of the
    // bandwidth the shape allows, so the first call of each shape times the candidates on its own
    // operands and keeps the fastest. A call that accumulates cannot be repeated, so it takes the
    // algorithm of the same shape without one, or the heuristic's first result.
    const MatmulKey key = std::make_tuple(kind, m, n, k, bias != nullptr);
    const std::lock_guard<std::mutex> lock(matmul_algorithms_mutex);
    const auto found = matmul_algorithms.find(key);
    if (found != matmul_algorithms.end()) {
        status = run(&found->second);
        if (status == CUBLAS_STATUS_SUCCESS) {
            return 0;
        }
        matmul_algorithms.erase(found);
    }
    cublasLtMatmulHeuristicResult_t results[MATMUL_CANDIDATES];
    int returned = 0;
    status = cublasLtMatmulAlgoGetHeuristic(state.handle, descriptors.operation, descriptors.weight,
                                            descriptors.input, descriptors.output,
                                            descriptors.output, descriptors.preference,
                                            MATMUL_CANDIDATES, results, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        return status_code(status == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_NOT_SUPPORTED : status);
    }
    if (beta != 0.0f) {
        return status_code(run(&results[0].algo));
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float best = 0.0f;
    int chosen = -1;
    for (int index = 0; index < returned; index++) {
        // The first run of a candidate loads its kernel, which costs more than the call itself.
        if (run(&results[index].algo) != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        cudaEventRecord(start, stream);
        const cublasStatus_t timed = run(&results[index].algo);
        cudaEventRecord(stop, stream);
        float elapsed = 0.0f;
        if (timed != CUBLAS_STATUS_SUCCESS || cudaEventSynchronize(stop) != cudaSuccess ||
            cudaEventElapsedTime(&elapsed, start, stop) != cudaSuccess) {
            continue;
        }
        if (chosen < 0 || elapsed < best) {
            best = elapsed;
            chosen = index;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (chosen < 0) {
        return status_code(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    matmul_algorithms[key] = results[chosen].algo;
    return status_code(run(&results[chosen].algo));
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
    thread_local ThreadState state;
    if (const int prepared = state.prepare(); prepared != 0) {
        return prepared;
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
    if (consistent()) {
        return status_code(run_consistently(
            consistent_nvfp4, std::make_tuple(n, k), state.handle, descriptors.operation,
            descriptors.weight, descriptors.input, descriptors.output, descriptors.preference,
            [&](cublasLtMatmulAlgo_t *algorithm) {
                return consistent_algorithm(state.handle, descriptors.operation, descriptors.weight,
                                            CUDA_R_4F_E2M1, CUDA_R_16BF, n, k,
                                            descriptors.preference, algorithm);
            },
            run));
    }
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
    cublasLtMatmulHeuristicResult_t results[NVFP4_CANDIDATES];
    int returned = 0;
    status = cublasLtMatmulAlgoGetHeuristic(state.handle, descriptors.operation, descriptors.weight,
                                            descriptors.input, descriptors.output,
                                            descriptors.output, descriptors.preference,
                                            NVFP4_CANDIDATES, results, &returned);
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

// Writes a line naming what a table of chosen algorithms depends on, the cuBLASLt version, the
// GPU, the descriptors and the rule that picked between the candidates, to `key`, which holds
// `capacity` bytes.
static int algorithm_key(char *key, int capacity, const char *table, int format) {
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
        key, static_cast<size_t>(capacity), "cuBLASLt %zu, %s, sm_%d%d, %d SMs, %s format %d",
        cublasLtGetVersion(), properties.name, properties.major, properties.minor,
        properties.multiProcessorCount, table, format);
    return written < 0 || written >= capacity ? static_cast<int>(cudaErrorInvalidValue) : 0;
}

extern "C" int mmh3_cublaslt_nvfp4_algorithm_key(char *key, int capacity) {
    return algorithm_key(key, capacity, "NVFP4", NVFP4_ALGORITHM_FORMAT);
}

extern "C" int mmh3_cublaslt_matmul_algorithm_key(char *key, int capacity) {
    return algorithm_key(key, capacity, "matmul", MATMUL_ALGORITHM_FORMAT);
}

// Copies up to `capacity` of the algorithms chosen for ordinary GEMM shapes to `algorithms` and
// returns how many there are.
extern "C" int mmh3_cublaslt_matmul_algorithms(Mmh3MatmulAlgorithm *algorithms, int capacity) {
    const std::lock_guard<std::mutex> lock(matmul_algorithms_mutex);
    int index = 0;
    for (const auto &[key, algorithm] : matmul_algorithms) {
        if (index < capacity) {
            algorithms[index].kind = std::get<0>(key);
            algorithms[index].bias = std::get<4>(key) ? 1 : 0;
            algorithms[index].m = std::get<1>(key);
            algorithms[index].n = std::get<2>(key);
            algorithms[index].k = std::get<3>(key);
            for (int word = 0; word < 8; word++) {
                algorithms[index].data[word] = algorithm.data[word];
            }
        }
        index++;
    }
    return index;
}

// Makes every thread that has not been told otherwise run consistently, or not.
extern "C" void mmh3_cublaslt_consistent_by_default(int consistent) {
    consistent_by_default.store(consistent != 0, std::memory_order_relaxed);
}

// Makes this thread run consistently or not whatever the process does, or follow the process
// again for -1.
extern "C" void mmh3_cublaslt_consistent_on_this_thread(int consistent) {
    consistent_here = consistent < 0 ? -1 : (consistent != 0 ? 1 : 0);
}

// Takes over algorithms for ordinary GEMM shapes that have none chosen yet.
extern "C" void mmh3_cublaslt_matmul_adopt(const Mmh3MatmulAlgorithm *algorithms, int count) {
    const std::lock_guard<std::mutex> lock(matmul_algorithms_mutex);
    for (int index = 0; index < count; index++) {
        cublasLtMatmulAlgo_t algorithm;
        for (int word = 0; word < 8; word++) {
            algorithm.data[word] = algorithms[index].data[word];
        }
        matmul_algorithms.emplace(std::make_tuple(static_cast<int>(algorithms[index].kind),
                                                  algorithms[index].m, algorithms[index].n,
                                                  algorithms[index].k, algorithms[index].bias != 0),
                                  algorithm);
    }
}
