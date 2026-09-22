#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
double relative_change(double value, double reference) {
    return std::abs((value - reference) / reference);
}

void test_pair_force_balance() {
    const auto scenario = nbody::make_analytic_circular_two_body();
    const auto accelerations = nbody::compute_accelerations(scenario.initial_state);
    const auto& first = scenario.initial_state.bodies[0];
    const auto& second = scenario.initial_state.bodies[1];
    const nbody::Vec3 force_sum =
        accelerations[0] * first.mass_kg + accelerations[1] * second.mass_kg;
    const double force_scale = nbody::norm(accelerations[0] * first.mass_kg);
    require(nbody::norm(force_sum) <= force_scale * 1.0e-14,
            "pair forces do not balance");
}

void test_five_orbits() {
    const auto scenario = nbody::make_analytic_circular_two_body();
    auto state = scenario.initial_state;
    const auto initial = nbody::compute_diagnostics(state);
    const double initial_energy = initial.total_energy_j();
    const double initial_angular_momentum = nbody::norm(initial.angular_momentum_kg_m2_s);

    constexpr std::size_t steps_per_orbit = 4000;
    constexpr std::size_t orbit_count = 5;
    const double dt_s = scenario.orbital_period_s / static_cast<double>(steps_per_orbit);
    double maximum_relative_energy_drift = 0.0;

    for (std::size_t step = 0; step < steps_per_orbit * orbit_count; ++step) {
        nbody::velocity_verlet_step(state, dt_s);
        if (step % 100 == 0) {
            const auto current = nbody::compute_diagnostics(state);
            maximum_relative_energy_drift = std::max(
                maximum_relative_energy_drift,
                relative_change(current.total_energy_j(), initial_energy));
        }
    }

    const auto final = nbody::compute_diagnostics(state);
    const double final_separation =
        nbody::norm(state.bodies[1].position_m - state.bodies[0].position_m);
    const double relative_separation_error =
        relative_change(final_separation, scenario.separation_m);
    const double relative_angular_momentum_drift =
        relative_change(nbody::norm(final.angular_momentum_kg_m2_s), initial_angular_momentum);
    const double normalized_center_of_mass_drift =
        nbody::norm(final.center_of_mass_m - initial.center_of_mass_m) / scenario.separation_m;

    require(maximum_relative_energy_drift <= 5.0e-6,
            "relative energy drift exceeded guard threshold");
    require(relative_separation_error <= 5.0e-5,
            "relative separation error exceeded guard threshold");
    require(relative_angular_momentum_drift <= 5.0e-10,
            "angular momentum drift exceeded guard threshold");
    require(normalized_center_of_mass_drift <= 1.0e-9,
            "center-of-mass drift exceeded guard threshold");
}

void test_invalid_inputs() {
    auto scenario = nbody::make_analytic_circular_two_body();
    bool rejected_timestep = false;
    try {
        nbody::velocity_verlet_step(scenario.initial_state, 0.0);
    } catch (const std::invalid_argument&) {
        rejected_timestep = true;
    }
    require(rejected_timestep, "zero timestep was not rejected");

    scenario.initial_state.bodies[1].position_m = scenario.initial_state.bodies[0].position_m;
    bool rejected_collision = false;
    try {
        (void)nbody::compute_accelerations(scenario.initial_state);
    } catch (const std::domain_error&) {
        rejected_collision = true;
    }
    require(rejected_collision, "coincident bodies were not rejected");
}

}  // namespace

int main() {
    try {
        test_pair_force_balance();
        test_five_orbits();
        test_invalid_inputs();
        std::cout << "all two-body tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
