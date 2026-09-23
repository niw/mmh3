#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

struct Mmh3DeviceInfo {
    char name[256];
    int32_t compute_major;
    int32_t compute_minor;
    int32_t multiprocessor_count;
    int32_t shared_memory_per_block_optin;
    int32_t shared_memory_per_multiprocessor;
    int32_t l2_cache_bytes;
    int32_t integrated;
    uint64_t total_global_bytes;
};

extern "C" int mmh3_cuda_device_count(int *count) {
    return static_cast<int>(cudaGetDeviceCount(count));
}

extern "C" int mmh3_cuda_device_info(int device, Mmh3DeviceInfo *info) {
    cudaDeviceProp properties;
    cudaError_t status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    std::memset(info, 0, sizeof(*info));
    std::strncpy(info->name, properties.name, sizeof(info->name) - 1);
    info->compute_major = properties.major;
    info->compute_minor = properties.minor;
    info->multiprocessor_count = properties.multiProcessorCount;
    info->shared_memory_per_block_optin = static_cast<int32_t>(properties.sharedMemPerBlockOptin);
    info->shared_memory_per_multiprocessor =
        static_cast<int32_t>(properties.sharedMemPerMultiprocessor);
    info->l2_cache_bytes = properties.l2CacheSize;
    info->integrated = properties.integrated;
    info->total_global_bytes = properties.totalGlobalMem;
    return 0;
}

extern "C" int mmh3_cuda_memory_info(size_t *free_bytes, size_t *total_bytes) {
    return static_cast<int>(cudaMemGetInfo(free_bytes, total_bytes));
}

extern "C" int mmh3_cuda_reads_host_memory(int *reads) {
    int device = 0;
    const cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    return static_cast<int>(
        cudaDeviceGetAttribute(reads, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
}

extern "C" int mmh3_cuda_can_access_peer(int device, int peer, int *can) {
    return static_cast<int>(cudaDeviceCanAccessPeer(can, device, peer));
}

// Lets the current device read `peer`'s memory. A pair that has it already is not an error, since
// every run over the same two cards asks again.
extern "C" int mmh3_cuda_enable_peer_access(int peer) {
    const cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return 0;
    }
    return static_cast<int>(status);
}

// Copies between any two addresses the process holds, on one card, between two cards or through
// the host, in order with the kernels on the current device's default stream.
extern "C" int mmh3_cuda_copy_across(void *destination, const void *source, size_t bytes) {
    return static_cast<int>(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDefault));
}

extern "C" const char *mmh3_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

// A failed allocation also stays behind as the last error, where the next kernel launch would read
// it as its own. The caller has the status already and may free something and try again, so it is
// cleared here.
extern "C" int mmh3_cuda_malloc(void **pointer, size_t bytes) {
    const cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess) {
        cudaGetLastError();
    }
    return static_cast<int>(status);
}

extern "C" int mmh3_cuda_free(void *pointer) { return static_cast<int>(cudaFree(pointer)); }

// Frees an allocation on the device it was made on, which is not always the one this thread is
// computing on. The current device belongs to the thread, so it is put back before returning.
extern "C" int mmh3_cuda_free_on(int device, void *pointer) {
    int current = 0;
    cudaError_t status = cudaGetDevice(&current);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    if (current == device) {
        return static_cast<int>(cudaFree(pointer));
    }
    if ((status = cudaSetDevice(device)) != cudaSuccess) {
        return static_cast<int>(status);
    }
    const cudaError_t freed = cudaFree(pointer);
    const cudaError_t restored = cudaSetDevice(current);
    return static_cast<int>(freed != cudaSuccess ? freed : restored);
}

extern "C" int mmh3_cuda_copy_to_device(void *destination, const void *source, size_t bytes) {
    return static_cast<int>(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
}

extern "C" int mmh3_cuda_copy_to_host(void *destination, const void *source, size_t bytes) {
    return static_cast<int>(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
}

extern "C" int mmh3_cuda_copy_device(void *destination, const void *source, size_t bytes) {
    return static_cast<int>(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice));
}

// Clears a failed allocation from the last error, as `mmh3_cuda_malloc` does.
extern "C" int mmh3_cuda_host_alloc(void **pointer, size_t bytes) {
    const cudaError_t status = cudaHostAlloc(pointer, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        cudaGetLastError();
    }
    return static_cast<int>(status);
}

extern "C" int mmh3_cuda_host_free(void *pointer) {
    return static_cast<int>(cudaFreeHost(pointer));
}

extern "C" int mmh3_cuda_stream_create(cudaStream_t *stream) {
    return static_cast<int>(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
}

extern "C" int mmh3_cuda_stream_destroy(cudaStream_t stream) {
    return static_cast<int>(cudaStreamDestroy(stream));
}

extern "C" int mmh3_cuda_stream_synchronize(cudaStream_t stream) {
    return static_cast<int>(cudaStreamSynchronize(stream));
}

extern "C" int mmh3_cuda_event_create(cudaEvent_t *event) {
    return static_cast<int>(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
}

extern "C" int mmh3_cuda_event_destroy(cudaEvent_t event) {
    return static_cast<int>(cudaEventDestroy(event));
}

extern "C" int mmh3_cuda_event_record(cudaEvent_t event, cudaStream_t stream) {
    return static_cast<int>(cudaEventRecord(event, stream));
}

extern "C" int mmh3_cuda_event_synchronize(cudaEvent_t event) {
    return static_cast<int>(cudaEventSynchronize(event));
}

// Holds later work on `stream` back until the work before the last record of `event` is done. The
// null stream is the legacy default stream, which every kernel here is launched on.
extern "C" int mmh3_cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
    return static_cast<int>(cudaStreamWaitEvent(stream, event, 0));
}

extern "C" int mmh3_cuda_get_device(int *device) { return static_cast<int>(cudaGetDevice(device)); }

extern "C" int mmh3_cuda_set_device(int device) { return static_cast<int>(cudaSetDevice(device)); }

extern "C" int mmh3_cuda_copy_to_device_async(void *destination, const void *source, size_t bytes,
                                              cudaStream_t stream) {
    return static_cast<int>(
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
}

extern "C" int mmh3_cuda_memset(void *pointer, int value, size_t bytes) {
    return static_cast<int>(cudaMemset(pointer, value, bytes));
}

extern "C" int mmh3_cuda_synchronize() { return static_cast<int>(cudaDeviceSynchronize()); }

__global__ void fill_f32_kernel(float *data, float value, size_t count) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        data[index] = value;
    }
}

extern "C" int mmh3_cuda_fill_f32(float *data, float value, size_t count) {
    if (count == 0) {
        return 0;
    }
    const unsigned threads_per_block = 256;
    size_t block_count = (count + threads_per_block - 1) / threads_per_block;
    if (block_count > 4096) {
        block_count = 4096;
    }
    fill_f32_kernel<<<static_cast<unsigned>(block_count), threads_per_block>>>(data, value, count);
    return static_cast<int>(cudaGetLastError());
}
