#pragma once

#include "nbody/barnes_hut.hpp"
#include "nbody/cuda_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

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

enum class CudaBarnesHutIntegrationPhase : std::uint32_t {
    none = 0,
    input_validation = 1,
    first_force_evaluation = 2,
    position_update = 3,
    second_force_evaluation = 4,
    integration_post_force_validation = 5,
    velocity_update = 6,
    state_commit = 7,
};

struct CudaBarnesHutIntegrationFailure {
    CudaBarnesHutIntegrationPhase phase{CudaBarnesHutIntegrationPhase::none};
    CudaBarnesHutFailure force_failure{};
    bool executor_terminal{false};
};

struct CudaBarnesHutIntegrationConfiguration {
    CudaBarnesHutConfiguration force{};
    // Bounds the transactional host candidate, two returned acceleration
    // arrays, and dynamic body-name storage. Allocator metadata is excluded.
    std::size_t host_integration_memory_budget_bytes{256U * 1024U * 1024U};
#ifdef NBODY_CUDA_BARNES_HUT_ENABLE_TEST_HOOKS
    enum class TestResourceFailurePoint : std::uint32_t {
        none = 0,
        after_first_force = 1,
        after_second_force = 2,
    };
    // Compiled only into the CUDA integration-test library. Production
    // targets have neither this field nor the associated failure branches.
    TestResourceFailurePoint test_resource_failure_point{
        TestResourceFailurePoint::none};
    std::size_t test_resource_failure_advance{1};
#endif
};

struct CudaBarnesHutIntegrationStatistics {
    CudaBarnesHutStatistics first_force;
    CudaBarnesHutStatistics second_force;
    std::size_t step_traversal_work{0};
    std::size_t step_traversal_work_limit{0};
    std::size_t host_integration_storage_bytes{0};
};

inline std::size_t cuda_barnes_hut_step_traversal_work_limit(
    const CudaBarnesHutConfiguration& configuration) {
    const std::size_t per_evaluation = configuration.tree.maximum_traversal_work;
    if (per_evaluation > std::numeric_limits<std::size_t>::max() / 2U) {
        throw std::length_error(
            "CUDA Barnes-Hut per-step traversal-work limit is not representable");
    }
    return per_evaluation * 2U;
}

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
    bool is_terminal() const noexcept;
    CudaDeviceInfo device_info() const;

private:
    class Impl;
    CudaBarnesHutConfiguration configuration_;
    std::unique_ptr<Impl> impl_;
};

// Transactional one-step Velocity Verlet using two complete Stage 3A force
// evaluations. The caller state changes only after the second tree/force call,
// drift/kick validation, and resource checks all succeed.
class CudaBarnesHutVelocityVerlet {
public:
    explicit CudaBarnesHutVelocityVerlet(
        CudaBarnesHutIntegrationConfiguration configuration = {});
    ~CudaBarnesHutVelocityVerlet();
    CudaBarnesHutVelocityVerlet(CudaBarnesHutVelocityVerlet&&) noexcept;
    CudaBarnesHutVelocityVerlet& operator=(CudaBarnesHutVelocityVerlet&&) noexcept;
    CudaBarnesHutVelocityVerlet(const CudaBarnesHutVelocityVerlet&) = delete;
    CudaBarnesHutVelocityVerlet& operator=(const CudaBarnesHutVelocityVerlet&) = delete;

    void advance(
        SimulationState& state,
        double dt_s,
        double gravitational_constant_value = gravitational_constant);

    const CudaBarnesHutIntegrationConfiguration& configuration() const noexcept {
        return configuration_;
    }
    CudaBarnesHutIntegrationStatistics last_statistics() const noexcept;
    CudaBarnesHutIntegrationFailure last_failure() const noexcept;
    bool is_terminal() const noexcept;
    CudaDeviceInfo device_info() const;

private:
    class Impl;
    CudaBarnesHutIntegrationConfiguration configuration_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nbody
