#include "nbody/visualization/compute_comparison.hpp"

#ifdef NBODY_VIS_HAS_OPENMP
#include "nbody/openmp_backend.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace nbody::visualization {

namespace {

struct TimedState {
    SimulationState state;
    double elapsed_ms{0.0};
};

TimedState integrate_timed(
    const SimulationState& initial_state,
    const ForceBackend& backend,
    std::size_t steps,
    double dt_s) {
    SimulationState state = initial_state;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t step = 0; step < steps; ++step) {
        velocity_verlet_step(state, dt_s, backend);
    }
    const auto stop = std::chrono::steady_clock::now();
    return {
        std::move(state),
        std::chrono::duration<double, std::milli>(stop - start).count(),
    };
}

bool finite_state(const SimulationState& state) {
    for (const auto& body : state.bodies) {
        if (!std::isfinite(body.mass_kg) ||
            !std::isfinite(body.position_m.x) || !std::isfinite(body.position_m.y) ||
            !std::isfinite(body.position_m.z) || !std::isfinite(body.velocity_m_s.x) ||
            !std::isfinite(body.velocity_m_s.y) || !std::isfinite(body.velocity_m_s.z)) {
            return false;
        }
    }
    return std::isfinite(state.time_s);
}

}  // namespace

std::size_t live_comparison_step_count(std::size_t body_count) {
    if (body_count < 2) {
        throw std::invalid_argument("live comparison requires at least two bodies");
    }
    constexpr std::size_t target_pair_visits = 8U * 1024U * 1024U;
    const std::size_t per_step = body_count * body_count;
    return std::clamp(target_pair_visits / per_step, std::size_t{8}, std::size_t{256});
}

bool live_comparison_matches(
    const LiveComputeComparison& comparison,
    std::size_t body_count,
    int openmp_threads) noexcept {
    return comparison.body_count == body_count &&
        comparison.openmp_threads == openmp_threads;
}

LiveComputeComparison run_live_compute_comparison(
    const SimulationState& initial_state,
    double dt_s,
    int openmp_threads) {
    if (initial_state.bodies.size() < 2 || !(dt_s > 0.0) || !std::isfinite(dt_s) ||
        openmp_threads <= 0) {
        throw std::invalid_argument("live comparison input is invalid");
    }
#ifndef NBODY_VIS_HAS_OPENMP
    (void)initial_state;
    (void)dt_s;
    (void)openmp_threads;
    throw std::runtime_error("live CPU comparison requires an OpenMP visualization build");
#else
    const OpenMpForceBackend openmp(openmp_threads);
    const auto& sequential = sequential_force_backend();
    const std::size_t steps = live_comparison_step_count(initial_state.bodies.size());

    // One identical untimed step initializes both paths outside the measured batch.
    (void)integrate_timed(initial_state, sequential, 1, dt_s);
    (void)integrate_timed(initial_state, openmp, 1, dt_s);

    const auto sequential_result = integrate_timed(initial_state, sequential, steps, dt_s);
    const auto openmp_result = integrate_timed(initial_state, openmp, steps, dt_s);

    LiveComputeComparison result;
    result.body_count = initial_state.bodies.size();
    result.integration_steps = steps;
    result.openmp_threads = openmp_threads;
    result.sequential_ms = sequential_result.elapsed_ms;
    result.openmp_ms = openmp_result.elapsed_ms;
    result.sequential_to_openmp_ratio =
        result.openmp_ms > 0.0 ? result.sequential_ms / result.openmp_ms : 0.0;
    result.finite = finite_state(sequential_result.state) && finite_state(openmp_result.state);

    for (std::size_t index = 0; index < result.body_count; ++index) {
        const auto& reference = sequential_result.state.bodies[index];
        const auto& candidate = openmp_result.state.bodies[index];
        const double position_difference = norm(candidate.position_m - reference.position_m);
        const double velocity_difference = norm(candidate.velocity_m_s - reference.velocity_m_s);
        result.maximum_absolute_position_m =
            std::max(result.maximum_absolute_position_m, position_difference);
        result.maximum_scaled_position = std::max(
            result.maximum_scaled_position,
            position_difference / std::max(norm(reference.position_m), 1.0));
        result.maximum_absolute_velocity_m_s =
            std::max(result.maximum_absolute_velocity_m_s, velocity_difference);
        result.maximum_scaled_velocity = std::max(
            result.maximum_scaled_velocity,
            velocity_difference / std::max(norm(reference.velocity_m_s), 1.0));
    }
    const bool position_matches = result.maximum_absolute_position_m <= 1.0e-6 ||
        result.maximum_scaled_position <= 1.0e-13;
    const bool velocity_matches = result.maximum_absolute_velocity_m_s <= 1.0e-9 ||
        result.maximum_scaled_velocity <= 1.0e-13;
    result.correctness_passed = result.finite && position_matches && velocity_matches;
    return result;
#endif
}

}  // namespace nbody::visualization
