#include "nbody/openmp_backend.hpp"

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
    if (!(gravitational_constant_value > 0.0) || !std::isfinite(gravitational_constant_value)) {
        throw std::invalid_argument("gravitational constant must be finite and positive");
    }
    for (const auto& body : state.bodies) {
        if (!(body.mass_kg > 0.0) || !std::isfinite(body.mass_kg)) {
            throw std::invalid_argument("body mass must be finite and positive");
        }
    }

    std::vector<Vec3> accelerations(state.bodies.size());
    int invalid_position = 0;
    const auto body_count = static_cast<std::int64_t>(state.bodies.size());

#pragma omp parallel for schedule(static) num_threads(thread_count_) reduction(| : invalid_position)
    for (std::int64_t raw_i = 0; raw_i < body_count; ++raw_i) {
        const auto i = static_cast<std::size_t>(raw_i);
        Vec3 acceleration;
        for (std::size_t j = 0; j < state.bodies.size(); ++j) {
            if (i == j) {
                continue;
            }
            const Vec3 displacement = state.bodies[j].position_m - state.bodies[i].position_m;
            const double distance_squared = norm_squared(displacement);
            if (!(distance_squared > 0.0) || !std::isfinite(distance_squared)) {
                invalid_position = 1;
                continue;
            }
            const double inverse_distance = 1.0 / std::sqrt(distance_squared);
            const double inverse_distance_cubed =
                inverse_distance * inverse_distance * inverse_distance;
            acceleration += displacement *
                (gravitational_constant_value * state.bodies[j].mass_kg * inverse_distance_cubed);
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

}  // namespace nbody
