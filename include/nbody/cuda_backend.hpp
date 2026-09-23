#pragma once

#include "nbody/simulation.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace nbody {

struct CudaDeviceInfo {
    int device_index{0};
    std::string name;
    int compute_capability_major{0};
    int compute_capability_minor{0};
    std::size_t total_global_memory_bytes{0};
    int multiprocessor_count{0};
    int cuda_driver_version{0};
    int cuda_runtime_version{0};
};

struct CudaRunTimings {
    double allocation_ms{0.0};
    double host_staging_ms{0.0};
    double host_to_device_ms{0.0};
    double device_compute_ms{0.0};
    double device_to_host_ms{0.0};
    double transfer_inclusive_ms{0.0};
    double backend_wall_ms{0.0};
};

struct CudaIntegrationResult {
    SimulationState final_state;
    CudaRunTimings timings;
};

int cuda_device_count();
CudaDeviceInfo cuda_device_info(int device_index = 0);

class CudaForceBackend final : public ForceBackend {
public:
    explicit CudaForceBackend(int device_index = 0);
    ~CudaForceBackend() override;
    CudaForceBackend(CudaForceBackend&&) noexcept;
    CudaForceBackend& operator=(CudaForceBackend&&) noexcept;
    CudaForceBackend(const CudaForceBackend&) = delete;
    CudaForceBackend& operator=(const CudaForceBackend&) = delete;

    const char* name() const noexcept override;
    std::vector<Vec3> compute_accelerations(
        const SimulationState& state,
        double gravitational_constant_value) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class CudaVelocityVerlet {
public:
    explicit CudaVelocityVerlet(std::size_t body_count, int device_index = 0);
    ~CudaVelocityVerlet();
    CudaVelocityVerlet(CudaVelocityVerlet&&) noexcept;
    CudaVelocityVerlet& operator=(CudaVelocityVerlet&&) noexcept;
    CudaVelocityVerlet(const CudaVelocityVerlet&) = delete;
    CudaVelocityVerlet& operator=(const CudaVelocityVerlet&) = delete;

    CudaIntegrationResult run(
        const SimulationState& initial_state,
        std::size_t steps,
        double dt_s,
        double gravitational_constant_value = gravitational_constant);
    CudaDeviceInfo device_info() const;
    double allocation_ms() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nbody
