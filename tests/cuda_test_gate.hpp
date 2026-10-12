#pragma once

#include <stdexcept>
#include <utility>

namespace nbody::test_support {

enum class CudaRuntimeTestDecision {
    execute,
    skip_no_device,
};

// Mandatory CUDA tests may skip only after successful enumeration reports
// exactly zero devices. Enumeration failures deliberately propagate to the
// test executable's failure handler instead of being converted into a skip.
template <typename DeviceEnumerator>
CudaRuntimeTestDecision strict_cuda_runtime_test_decision(
    DeviceEnumerator&& enumerate_devices) {
    const int device_count = std::forward<DeviceEnumerator>(
        enumerate_devices)();
    if (device_count < 0) {
        throw std::runtime_error(
            "CUDA device enumeration returned a negative device count");
    }
    return device_count == 0
        ? CudaRuntimeTestDecision::skip_no_device
        : CudaRuntimeTestDecision::execute;
}

}  // namespace nbody::test_support
