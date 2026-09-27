#include "nbody/openmp_backend.hpp"

#include "nbody/direct_force.hpp"

#include <omp.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace nbody {

OpenMpForceBackend::OpenMpForceBackend(int thread_count) : thread_count_(thread_count) {
    if (thread_count_ <= 0) {
        throw std::invalid_argument("OpenMP thread count must be positive");
    }
}

const char* OpenMpForceBackend::name() const noexcept { return "openmp"; }

int OpenMpForceBackend::thread_count() const noexcept { return thread_count_; }

std::vector<Vec3> OpenMpForceBackend::compute_accelerations(
    const SimulationState& state,
    double gravitational_constant_value) const {
    validate_direct_force_inputs(state, gravitational_constant_value);

    std::vector<Vec3> accelerations(state.bodies.size());
    int invalid_position = 0;
    const auto body_count = static_cast<std::int64_t>(state.bodies.size());

#pragma omp parallel for schedule(static) num_threads(thread_count_) reduction(| : invalid_position)
    for (std::int64_t raw_i = 0; raw_i < body_count; ++raw_i) {
        const auto i = static_cast<std::size_t>(raw_i);
        Vec3 acceleration;
        if (!try_compute_direct_acceleration(
                state, i, gravitational_constant_value, acceleration)) {
            invalid_position = 1;
        }
        accelerations[i] = acceleration;
    }

    if (invalid_position != 0) {
        throw std::domain_error("coincident or invalid body positions");
    }
    return accelerations;
}

int openmp_processor_count() noexcept { return omp_get_num_procs(); }

int openmp_runtime_max_threads() noexcept { return omp_get_max_threads(); }

int openmp_effective_thread_count(int requested_thread_count) {
    if (requested_thread_count <= 0) {
        throw std::invalid_argument("OpenMP thread count must be positive");
    }
    int effective_thread_count = 0;
#pragma omp parallel num_threads(requested_thread_count)
    {
#pragma omp single
        effective_thread_count = omp_get_num_threads();
    }
    return effective_thread_count;
}

int openmp_specification_date() noexcept { return _OPENMP; }

}  // namespace nbody
