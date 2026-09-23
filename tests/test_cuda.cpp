#include "nbody/cuda_backend.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double scaled_difference(double difference, double reference, double floor) {
    return difference / std::max(std::abs(reference), floor);
}

void compare_accelerations(const nbody::SimulationState& state, nbody::CudaForceBackend& cuda) {
    const auto reference = nbody::compute_accelerations(state);
    const auto candidate = cuda.compute_accelerations(state, nbody::gravitational_constant);
    require(reference.size() == candidate.size(), "CUDA acceleration count differs");
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double difference = nbody::norm(candidate[i] - reference[i]);
        require(difference <= 1.0e-18 ||
                    scaled_difference(difference, nbody::norm(reference[i]), 1.0e-18) <= 1.0e-12,
                "CUDA acceleration exceeded predeclared mixed tolerance");
    }
}

nbody::SimulationState integrate_cpu(
    nbody::SimulationState state,
    std::size_t steps,
    double dt_s) {
    for (std::size_t step = 0; step < steps; ++step) {
        nbody::velocity_verlet_step(state, dt_s);
    }
    return state;
}

void compare_states(
    const nbody::SimulationState& reference,
    const nbody::SimulationState& candidate,
    const std::string& label) {
    require(reference.bodies.size() == candidate.bodies.size(), label + " body count differs");
    for (std::size_t i = 0; i < reference.bodies.size(); ++i) {
        const double position_difference =
            nbody::norm(candidate.bodies[i].position_m - reference.bodies[i].position_m);
        const double velocity_difference =
            nbody::norm(candidate.bodies[i].velocity_m_s - reference.bodies[i].velocity_m_s);
        require(position_difference <= 1.0e-3 ||
                    scaled_difference(
                        position_difference, nbody::norm(reference.bodies[i].position_m), 1.0) <=
                        2.0e-12,
                label + " position exceeded predeclared mixed tolerance");
        require(velocity_difference <= 1.0e-8 ||
                    scaled_difference(
                        velocity_difference, nbody::norm(reference.bodies[i].velocity_m_s), 1.0) <=
                        2.0e-12,
                label + " velocity exceeded predeclared mixed tolerance");
    }
}

void validate_integration(
    const nbody::SimulationState& initial,
    std::size_t steps,
    double dt_s,
    const std::string& label) {
    const auto cpu = integrate_cpu(initial, steps, dt_s);
    nbody::CudaVelocityVerlet cuda(initial.bodies.size());
    const auto first = cuda.run(initial, steps, dt_s).final_state;
    const auto second = cuda.run(initial, steps, dt_s).final_state;
    compare_states(cpu, first, label + " CPU/CUDA");
    compare_states(first, second, label + " repeatability");

    const auto initial_diagnostics = nbody::compute_diagnostics(initial);
    const auto cpu_diagnostics = nbody::compute_diagnostics(cpu);
    const auto cuda_diagnostics = nbody::compute_diagnostics(first);
    const double energy_difference = std::abs(
        cuda_diagnostics.total_energy_j() - cpu_diagnostics.total_energy_j());
    require(scaled_difference(energy_difference, cpu_diagnostics.total_energy_j(), 1.0) <= 2.0e-12,
            label + " energy differs from CPU reference");
    const double angular_difference = nbody::norm(
        cuda_diagnostics.angular_momentum_kg_m2_s -
        cpu_diagnostics.angular_momentum_kg_m2_s);
    require(scaled_difference(
                angular_difference,
                nbody::norm(cpu_diagnostics.angular_momentum_kg_m2_s), 1.0) <= 2.0e-12,
            label + " angular momentum differs from CPU reference");
    const double energy_drift = std::abs(
        cuda_diagnostics.total_energy_j() - initial_diagnostics.total_energy_j()) /
        std::max(std::abs(initial_diagnostics.total_energy_j()), 1.0);
    require(energy_drift <= 5.0e-6, label + " CUDA energy drift exceeded guard threshold");
    const double angular_drift = nbody::norm(
        cuda_diagnostics.angular_momentum_kg_m2_s -
        initial_diagnostics.angular_momentum_kg_m2_s) /
        std::max(nbody::norm(initial_diagnostics.angular_momentum_kg_m2_s), 1.0);
    require(angular_drift <= 5.0e-10,
            label + " CUDA angular-momentum drift exceeded guard threshold");

    double total_mass = 0.0;
    double momentum_scale = 0.0;
    double position_scale = 1.0;
    for (const auto& body : initial.bodies) {
        total_mass += body.mass_kg;
        momentum_scale += body.mass_kg * nbody::norm(body.velocity_m_s);
        position_scale = std::max(position_scale, nbody::norm(body.position_m));
    }
    const auto expected_center_of_mass =
        initial_diagnostics.center_of_mass_m +
        initial_diagnostics.linear_momentum_kg_m_s /
            total_mass * (static_cast<double>(steps) * dt_s);
    const double momentum_drift = nbody::norm(
        cuda_diagnostics.linear_momentum_kg_m_s -
        initial_diagnostics.linear_momentum_kg_m_s) /
        std::max(momentum_scale, 1.0);
    const double center_of_mass_drift = nbody::norm(
        cuda_diagnostics.center_of_mass_m - expected_center_of_mass) /
        position_scale;
    require(momentum_drift <= 1.0e-12,
            label + " CUDA linear-momentum drift exceeded guard threshold");
    require(center_of_mass_drift <= 1.0e-12,
            label + " CUDA center-of-mass drift exceeded guard threshold");
}

void test_invalid_input(nbody::CudaForceBackend& cuda) {
    auto state = nbody::make_seeded_synthetic_system(4, 17);
    state.bodies[1].position_m = state.bodies[0].position_m;
    bool rejected_collision = false;
    try {
        (void)cuda.compute_accelerations(state, nbody::gravitational_constant);
    } catch (const std::domain_error&) {
        rejected_collision = true;
    }
    require(rejected_collision, "CUDA backend accepted coincident bodies");

    state = nbody::make_seeded_synthetic_system(4, 17);
    state.bodies[0].position_m.x = std::numeric_limits<double>::quiet_NaN();
    bool rejected_nan = false;
    try {
        (void)cuda.compute_accelerations(state, nbody::gravitational_constant);
    } catch (const std::invalid_argument&) {
        rejected_nan = true;
    }
    require(rejected_nan, "CUDA backend accepted NaN input");
}

}  // namespace

int main() {
    try {
        require(nbody::cuda_device_count() > 0, "no CUDA device available");
        nbody::CudaForceBackend cuda;
        const auto binary = nbody::make_analytic_circular_two_body();
        compare_accelerations(binary.initial_state, cuda);
        for (const std::size_t bodies : std::array<std::size_t, 3>{8, 32, 128}) {
            compare_accelerations(nbody::make_seeded_synthetic_system(bodies, 0x5eedULL), cuda);
        }
        validate_integration(
            binary.initial_state, 4000, binary.orbital_period_s / 2000.0,
            "analytic two-body");
        validate_integration(
            nbody::make_seeded_synthetic_system(32, 0x5eedULL), 250, 10.0,
            "seeded 32-body");
        test_invalid_input(cuda);
        const auto device = nbody::cuda_device_info();
        std::cout << "all CUDA validation tests passed; device=" << device.name
                  << " compute_capability=" << device.compute_capability_major << '.'
                  << device.compute_capability_minor << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
