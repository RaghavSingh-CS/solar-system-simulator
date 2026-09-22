#include "nbody/diagnostics.hpp"
#include "nbody/openmp_backend.hpp"
#include "nbody/scenarios.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct StateDifference {
    double maximum_absolute_position_m{0.0};
    double maximum_scaled_position{0.0};
    double maximum_absolute_velocity_m_s{0.0};
    double maximum_scaled_velocity{0.0};
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double normalized_difference(double difference, double reference_norm, double absolute_floor) {
    return difference / std::max(reference_norm, absolute_floor);
}

double relative_scalar_drift(double value, double initial_value, double absolute_floor) {
    return std::abs(value - initial_value) /
        std::max({std::abs(value), std::abs(initial_value), absolute_floor});
}

StateDifference compare_states(
    const nbody::SimulationState& reference,
    const nbody::SimulationState& candidate) {
    require(reference.bodies.size() == candidate.bodies.size(), "body counts differ");
    StateDifference difference;
    for (std::size_t i = 0; i < reference.bodies.size(); ++i) {
        const double position_difference =
            nbody::norm(candidate.bodies[i].position_m - reference.bodies[i].position_m);
        const double velocity_difference =
            nbody::norm(candidate.bodies[i].velocity_m_s - reference.bodies[i].velocity_m_s);
        difference.maximum_absolute_position_m =
            std::max(difference.maximum_absolute_position_m, position_difference);
        difference.maximum_scaled_position = std::max(
            difference.maximum_scaled_position,
            normalized_difference(position_difference, nbody::norm(reference.bodies[i].position_m), 1.0));
        difference.maximum_absolute_velocity_m_s =
            std::max(difference.maximum_absolute_velocity_m_s, velocity_difference);
        difference.maximum_scaled_velocity = std::max(
            difference.maximum_scaled_velocity,
            normalized_difference(velocity_difference, nbody::norm(reference.bodies[i].velocity_m_s), 1.0));
    }
    return difference;
}

nbody::SimulationState integrate(
    nbody::SimulationState state,
    const nbody::ForceBackend& backend,
    std::size_t steps,
    double dt_s) {
    for (std::size_t step = 0; step < steps; ++step) {
        nbody::velocity_verlet_step(state, dt_s, backend);
    }
    return state;
}

void validate_scenario(
    const std::string& scenario_name,
    const nbody::SimulationState& initial_state,
    std::size_t steps,
    double dt_s,
    int threads) {
    const nbody::OpenMpForceBackend openmp_backend(threads);
    const auto initial_diagnostics = nbody::compute_diagnostics(initial_state);
    const auto sequential_state =
        integrate(initial_state, nbody::sequential_force_backend(), steps, dt_s);
    const auto openmp_state = integrate(initial_state, openmp_backend, steps, dt_s);
    const auto repeated_openmp_state = integrate(initial_state, openmp_backend, steps, dt_s);

    const StateDifference cross_backend = compare_states(sequential_state, openmp_state);
    const StateDifference repeatability = compare_states(openmp_state, repeated_openmp_state);
    const auto sequential_diagnostics = nbody::compute_diagnostics(sequential_state);
    const auto openmp_diagnostics = nbody::compute_diagnostics(openmp_state);

    const double sequential_energy_drift = relative_scalar_drift(
        sequential_diagnostics.total_energy_j(), initial_diagnostics.total_energy_j(), 1.0);
    const double openmp_energy_drift = relative_scalar_drift(
        openmp_diagnostics.total_energy_j(), initial_diagnostics.total_energy_j(), 1.0);
    const double sequential_angular_drift = normalized_difference(
        nbody::norm(sequential_diagnostics.angular_momentum_kg_m2_s -
                    initial_diagnostics.angular_momentum_kg_m2_s),
        nbody::norm(initial_diagnostics.angular_momentum_kg_m2_s), 1.0);
    const double openmp_angular_drift = normalized_difference(
        nbody::norm(openmp_diagnostics.angular_momentum_kg_m2_s -
                    initial_diagnostics.angular_momentum_kg_m2_s),
        nbody::norm(initial_diagnostics.angular_momentum_kg_m2_s), 1.0);
    const double momentum_difference = nbody::norm(
        openmp_diagnostics.linear_momentum_kg_m_s -
        sequential_diagnostics.linear_momentum_kg_m_s);
    const double center_of_mass_difference = nbody::norm(
        openmp_diagnostics.center_of_mass_m - sequential_diagnostics.center_of_mass_m);

    double total_mass = 0.0;
    double momentum_scale = 0.0;
    double position_scale = 1.0;
    for (const auto& body : initial_state.bodies) {
        total_mass += body.mass_kg;
        momentum_scale += body.mass_kg * nbody::norm(body.velocity_m_s);
        position_scale = std::max(position_scale, nbody::norm(body.position_m));
    }
    const nbody::Vec3 expected_center_of_mass =
        initial_diagnostics.center_of_mass_m +
        initial_diagnostics.linear_momentum_kg_m_s /
            total_mass * (static_cast<double>(steps) * dt_s);
    const double sequential_momentum_drift = normalized_difference(
        nbody::norm(sequential_diagnostics.linear_momentum_kg_m_s -
                    initial_diagnostics.linear_momentum_kg_m_s),
        momentum_scale, 1.0);
    const double openmp_momentum_drift = normalized_difference(
        nbody::norm(openmp_diagnostics.linear_momentum_kg_m_s -
                    initial_diagnostics.linear_momentum_kg_m_s),
        momentum_scale, 1.0);
    const double sequential_center_of_mass_drift = normalized_difference(
        nbody::norm(sequential_diagnostics.center_of_mass_m - expected_center_of_mass),
        position_scale, 1.0);
    const double openmp_center_of_mass_drift = normalized_difference(
        nbody::norm(openmp_diagnostics.center_of_mass_m - expected_center_of_mass),
        position_scale, 1.0);
    const double cross_backend_momentum_scaled =
        normalized_difference(momentum_difference, momentum_scale, 1.0);
    const double cross_backend_center_of_mass_scaled =
        normalized_difference(center_of_mass_difference, position_scale, 1.0);

    std::cout << std::setprecision(17)
              << "validation scenario=" << scenario_name
              << " threads=" << threads
              << " max_abs_position_m=" << cross_backend.maximum_absolute_position_m
              << " max_scaled_position=" << cross_backend.maximum_scaled_position
              << " max_abs_velocity_m_s=" << cross_backend.maximum_absolute_velocity_m_s
              << " max_scaled_velocity=" << cross_backend.maximum_scaled_velocity
              << " sequential_energy_drift=" << sequential_energy_drift
              << " openmp_energy_drift=" << openmp_energy_drift
              << " sequential_angular_drift=" << sequential_angular_drift
              << " openmp_angular_drift=" << openmp_angular_drift
              << " momentum_difference_kg_m_s=" << momentum_difference
              << " momentum_difference_scaled=" << cross_backend_momentum_scaled
              << " center_of_mass_difference_m=" << center_of_mass_difference
              << " center_of_mass_difference_scaled=" << cross_backend_center_of_mass_scaled
              << " sequential_momentum_drift=" << sequential_momentum_drift
              << " openmp_momentum_drift=" << openmp_momentum_drift
              << " sequential_center_of_mass_drift=" << sequential_center_of_mass_drift
              << " openmp_center_of_mass_drift=" << openmp_center_of_mass_drift
              << " repeat_position_m=" << repeatability.maximum_absolute_position_m
              << " repeat_velocity_m_s=" << repeatability.maximum_absolute_velocity_m_s
              << '\n';

    constexpr double position_absolute_tolerance_m = 1.0e-6;
    constexpr double position_relative_tolerance = 1.0e-13;
    constexpr double velocity_absolute_tolerance_m_s = 1.0e-9;
    constexpr double velocity_relative_tolerance = 1.0e-13;
    require(cross_backend.maximum_absolute_position_m <= position_absolute_tolerance_m ||
                cross_backend.maximum_scaled_position <= position_relative_tolerance,
            "OpenMP position result exceeded mixed tolerance");
    require(cross_backend.maximum_absolute_velocity_m_s <= velocity_absolute_tolerance_m_s ||
                cross_backend.maximum_scaled_velocity <= velocity_relative_tolerance,
            "OpenMP velocity result exceeded mixed tolerance");
    require(momentum_difference <= 1.0e6 || cross_backend_momentum_scaled <= 1.0e-13,
            "cross-backend momentum difference is too large");
    require(center_of_mass_difference <= 1.0e-6 ||
                cross_backend_center_of_mass_scaled <= 1.0e-13,
            "cross-backend center-of-mass difference is too large");
    require(repeatability.maximum_absolute_position_m <= position_absolute_tolerance_m ||
                repeatability.maximum_scaled_position <= position_relative_tolerance,
            "OpenMP position results are not repeatable");
    require(repeatability.maximum_absolute_velocity_m_s <= velocity_absolute_tolerance_m_s ||
                repeatability.maximum_scaled_velocity <= velocity_relative_tolerance,
            "OpenMP velocity results are not repeatable");
    require(openmp_energy_drift <= 5.0e-6, "OpenMP energy drift exceeded guard threshold");
    require(openmp_angular_drift <= 5.0e-10,
            "OpenMP angular-momentum drift exceeded guard threshold");
    require(sequential_energy_drift <= 5.0e-6,
            "sequential energy drift exceeded guard threshold");
    require(sequential_angular_drift <= 5.0e-10,
            "sequential angular-momentum drift exceeded guard threshold");
    require(sequential_momentum_drift <= 1.0e-12 && openmp_momentum_drift <= 1.0e-12,
            "linear-momentum drift exceeded guard threshold");
    require(sequential_center_of_mass_drift <= 1.0e-12 &&
                openmp_center_of_mass_drift <= 1.0e-12,
            "center-of-mass drift exceeded guard threshold");
}

void test_invalid_inputs() {
    bool rejected_thread_count = false;
    try {
        (void)nbody::OpenMpForceBackend(0);
    } catch (const std::invalid_argument&) {
        rejected_thread_count = true;
    }
    require(rejected_thread_count, "zero OpenMP thread count was not rejected");

    const nbody::OpenMpForceBackend backend(2);
    auto state = nbody::make_seeded_synthetic_system(4, 17);
    bool rejected_timestep = false;
    try {
        nbody::velocity_verlet_step(state, 0.0, backend);
    } catch (const std::invalid_argument&) {
        rejected_timestep = true;
    }
    require(rejected_timestep, "OpenMP integration accepted a zero timestep");

    state.bodies[1].position_m = state.bodies[0].position_m;
    bool rejected_collision = false;
    try {
        (void)backend.compute_accelerations(state, nbody::gravitational_constant);
    } catch (const std::domain_error&) {
        rejected_collision = true;
    }
    require(rejected_collision, "OpenMP backend accepted coincident bodies");

    bool rejected_gravity = false;
    try {
        (void)backend.compute_accelerations(state, -nbody::gravitational_constant);
    } catch (const std::invalid_argument&) {
        rejected_gravity = true;
    }
    require(rejected_gravity, "OpenMP backend accepted a negative gravitational constant");
}

}  // namespace

int main() {
    try {
        const auto binary = nbody::make_analytic_circular_two_body();
        const auto seeded = nbody::make_seeded_synthetic_system(32, 0x5eedULL);
        for (const int threads : std::array<int, 3>{1, 2, 4}) {
            validate_scenario(
                "analytic_two_body", binary.initial_state, 4000,
                binary.orbital_period_s / 2000.0, threads);
            validate_scenario("seeded_32_body", seeded, 250, 10.0, threads);
        }
        test_invalid_inputs();
        std::cout << "all OpenMP validation tests passed; runtime_processors="
                  << nbody::openmp_processor_count()
                  << " runtime_max_threads=" << nbody::openmp_runtime_max_threads() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
