#pragma once

#include "nbody/barnes_hut.hpp"
#include "nbody/cuda_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nbody {

inline constexpr std::uint32_t cuda_barnes_hut_maximum_stack_capacity = 1024;

// Test-only host fault injection. The injected failure occurs after the owned
// stream is quiescent but before the first deallocation during a live resize.
// It exercises terminal ownership without simulating a CUDA device failure.
enum class CudaBarnesHutHostFailureInjection {
    none,
    live_release_before_first_deallocation,
};

// Stage 3A is a standalone force evaluator. It does not integrate state and is
// not wired into the headless runner, viewer, or counterfactual engine.
struct CudaBarnesHutConfiguration {
    int device_index{0};
    BarnesHutConfiguration tree{};
    std::size_t device_memory_budget_bytes{512U * 1024U * 1024U};
    std::size_t host_staging_memory_budget_bytes{512U * 1024U * 1024U};
    std::uint32_t traversal_stack_capacity{512};
    CudaBarnesHutHostFailureInjection host_failure_injection{
        CudaBarnesHutHostFailureInjection::none};
};

enum class CudaBarnesHutFailureCode : std::uint32_t {
    none = 0,
    invalid_tree = 1,
    singular_geometry = 2,
    nonrepresentable_geometry = 3,
    nonrepresentable_acceleration = 4,
    traversal_stack_exhausted = 5,
    traversal_work_exhausted = 6,
    runtime_failure = 7,
};

struct CudaBarnesHutFailure {
    CudaBarnesHutFailureCode code{CudaBarnesHutFailureCode::none};
    std::uint32_t target_index{barnes_hut_no_packed_index};
    std::uint32_t node_index{barnes_hut_no_packed_index};
};

struct CudaBarnesHutStatistics {
    std::size_t body_count{0};
    std::size_t node_count{0};
    std::size_t device_storage_bytes{0};
    std::size_t host_staging_bytes{0};
    std::size_t traversal_stack_bytes_per_thread{0};
    std::size_t traversal_work{0};
    BarnesHutStatistics host_tree;
};

class CudaBarnesHutForceBackend final : public ForceBackend {
public:
    explicit CudaBarnesHutForceBackend(CudaBarnesHutConfiguration configuration = {});
    ~CudaBarnesHutForceBackend() override;
    CudaBarnesHutForceBackend(CudaBarnesHutForceBackend&&) noexcept;
    CudaBarnesHutForceBackend& operator=(CudaBarnesHutForceBackend&&) noexcept;
    CudaBarnesHutForceBackend(const CudaBarnesHutForceBackend&) = delete;
    CudaBarnesHutForceBackend& operator=(const CudaBarnesHutForceBackend&) = delete;

    const char* name() const noexcept override { return "cuda-barnes-hut"; }
    std::vector<Vec3> compute_accelerations(
        const SimulationState& state,
        double gravitational_constant_value = gravitational_constant) const override;

    const CudaBarnesHutConfiguration& configuration() const noexcept {
        return configuration_;
    }
    CudaBarnesHutStatistics last_statistics() const noexcept;
    CudaBarnesHutFailure last_failure() const noexcept;
    CudaDeviceInfo device_info() const;

private:
    class Impl;
    CudaBarnesHutConfiguration configuration_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nbody
