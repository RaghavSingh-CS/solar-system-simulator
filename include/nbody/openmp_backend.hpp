#pragma once

#include "nbody/simulation.hpp"

namespace nbody {

class OpenMpForceBackend final : public ForceBackend {
public:
    explicit OpenMpForceBackend(int thread_count);

    const char* name() const noexcept override;
    int thread_count() const noexcept;
    std::vector<Vec3> compute_accelerations(
        const SimulationState& state,
        double gravitational_constant_value) const override;

private:
    int thread_count_;
};

int openmp_processor_count() noexcept;
int openmp_runtime_max_threads() noexcept;

}  // namespace nbody
