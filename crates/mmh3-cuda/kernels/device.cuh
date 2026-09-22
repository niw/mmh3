#pragma once

#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <mutex>

// What a launch has to know about the device it is going for. The opt-in shared memory a kernel
// may ask for and the number of multiprocessors a persistent grid covers both belong to a device
// rather than to the process, so both are worked out once per device: a process computing on two
// cards configures each, where an answer cached from the first would fail the launch on the second
// or size its grid for the wrong card.

// The most devices one process computes on. It matches `MAX_DEVICES` on the Rust side, which
// indexes its allocation counters by device the same way.
constexpr int MMH3_MAX_DEVICES = 16;

// The device this thread computes on, or -1 where there is none.
inline int mmh3_current_device() {
    int device = 0;
    return cudaGetDevice(&device) == cudaSuccess ? device : -1;
}

// What one kernel's shared memory has cost so far, a slot per device holding the status of
// `cudaFuncSetAttribute` plus one so that zero means untried.
//
// NOTE: a launcher keeps one of these per kernel, as a function-local static of the template that
// launches it. It cannot live inside `mmh3_configure_shared_memory` instead: two kernels of the
// same signature are one type there, and the second would take the first's answer and launch
// without the shared memory it asked for.
struct Mmh3SharedMemory {
    std::atomic<int> configured[MMH3_MAX_DEVICES];
};

// Gives `kernel` the shared memory it asks for, once per device.
//
// NOTE: two threads reaching an unconfigured device set the same attribute to the same value, and
// the driver takes it as many times as it is given, so the slot needs no lock.
template <typename Kernel>
inline cudaError_t mmh3_configure_shared_memory(Mmh3SharedMemory &state, Kernel kernel,
                                                int shared_bytes) {
    const int device = mmh3_current_device();
    if (device < 0 || device >= MMH3_MAX_DEVICES) {
        return cudaErrorInvalidDevice;
    }
    if (const int kept = state.configured[device].load(std::memory_order_relaxed); kept != 0) {
        return static_cast<cudaError_t>(kept - 1);
    }
    const cudaError_t status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    state.configured[device].store(static_cast<int>(status) + 1, std::memory_order_relaxed);
    return status;
}

// The largest shared memory one kernel has been given so far on each device, for a kernel whose
// launches ask for more or less by their shape. The limit only grows, so a launch never meets a
// smaller one another launch set, and the lock keeps two launches from setting it out of order.
struct Mmh3SharedMemoryLimit {
    std::mutex mutex;
    size_t configured[MMH3_MAX_DEVICES] = {};
};

// Raises `kernel`'s shared memory on this thread's device to `shared_bytes` where it is lower.
template <typename Kernel>
inline cudaError_t mmh3_raise_shared_memory(Mmh3SharedMemoryLimit &state, Kernel kernel,
                                            size_t shared_bytes) {
    const int device = mmh3_current_device();
    if (device < 0 || device >= MMH3_MAX_DEVICES) {
        return cudaErrorInvalidDevice;
    }
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (shared_bytes <= state.configured[device]) {
        return cudaSuccess;
    }
    const cudaError_t status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    if (status == cudaSuccess) {
        state.configured[device] = shared_bytes;
    }
    return status;
}

// Multiprocessors on the device this thread computes on, or zero where it cannot be asked.
inline int mmh3_multiprocessor_count() {
    static std::atomic<int> counts[MMH3_MAX_DEVICES];
    const int device = mmh3_current_device();
    if (device < 0 || device >= MMH3_MAX_DEVICES) {
        return 0;
    }
    if (const int kept = counts[device].load(std::memory_order_relaxed); kept != 0) {
        return kept;
    }
    int processors = 0;
    if (cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, device) !=
        cudaSuccess) {
        return 0;
    }
    counts[device].store(processors, std::memory_order_relaxed);
    return processors;
}
