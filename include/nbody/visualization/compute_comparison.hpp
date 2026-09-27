#pragma once

#include "nbody/simulation.hpp"

#include <cstddef>

namespace nbody::visualization {

struct LiveComputeComparison {
    std::size_t body_count{0};
    std::size_t integration_steps{0};
    int openmp_threads{1};
    double sequential_ms{0.0};
    double openmp_ms{0.0};
    double sequential_to_openmp_ratio{0.0};
    double maximum_absolute_position_m{0.0};
    double maximum_scaled_position{0.0};
    double maximum_absolute_velocity_m_s{0.0};
    double maximum_scaled_velocity{0.0};
    bool finite{false};
    bool correctness_passed{false};
};

std::size_t live_comparison_step_count(std::size_t body_count);

bool live_comparison_matches(
    const LiveComputeComparison& comparison,
    std::size_t body_count,
    int openmp_threads) noexcept;

LiveComputeComparison run_live_compute_comparison(
    const SimulationState& initial_state,
    double dt_s,
    int openmp_threads);

}  // namespace nbody::visualization
