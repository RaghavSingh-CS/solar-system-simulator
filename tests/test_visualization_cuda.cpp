#include "nbody/cuda_backend.hpp"
#include "nbody/scenarios.hpp"
#include "nbody/visualization/controller.hpp"
#include "cuda_test_gate.hpp"

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

bool same_state(const nbody::SimulationState& lhs, const nbody::SimulationState& rhs) {
    if (lhs.time_s != rhs.time_s || lhs.bodies.size() != rhs.bodies.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.bodies.size(); ++index) {
        const auto& left = lhs.bodies[index];
        const auto& right = rhs.bodies[index];
        if (left.name != right.name || left.mass_kg != right.mass_kg ||
            left.position_m.x != right.position_m.x ||
            left.position_m.y != right.position_m.y ||
            left.position_m.z != right.position_m.z ||
            left.velocity_m_s.x != right.velocity_m_s.x ||
            left.velocity_m_s.y != right.velocity_m_s.y ||
            left.velocity_m_s.z != right.velocity_m_s.z) {
            return false;
        }
    }
    return true;
}

double scaled_difference(double difference, double reference, double floor) {
    return difference / std::max(std::abs(reference), floor);
}

void compare_states(
    const nbody::SimulationState& reference,
    const nbody::SimulationState& candidate,
    const std::string& label) {
    require(reference.bodies.size() == candidate.bodies.size(), label + " body count differs");
    require(reference.time_s == candidate.time_s, label + " simulated time differs");
    for (std::size_t index = 0; index < reference.bodies.size(); ++index) {
        const double position_difference = nbody::norm(
            candidate.bodies[index].position_m - reference.bodies[index].position_m);
        const double velocity_difference = nbody::norm(
            candidate.bodies[index].velocity_m_s - reference.bodies[index].velocity_m_s);
        require(position_difference <= 1.0e-3 ||
                    scaled_difference(
                        position_difference,
                        nbody::norm(reference.bodies[index].position_m),
                        1.0) <= 2.0e-12,
                label + " position exceeded mixed tolerance");
        require(velocity_difference <= 1.0e-8 ||
                    scaled_difference(
                        velocity_difference,
                        nbody::norm(reference.bodies[index].velocity_m_s),
                        1.0) <= 2.0e-12,
                label + " velocity exceeded mixed tolerance");
    }
}

nbody::SimulationState integrate_sequential(
    nbody::SimulationState state,
    std::size_t steps,
    double dt_s) {
    for (std::size_t step = 0; step < steps; ++step) {
        nbody::velocity_verlet_step(state, dt_s);
    }
    return state;
}

void test_cuda_controller() {
    constexpr double dt_s = 10.0;
    const auto initial = nbody::make_seeded_synthetic_system(8, 24301);
    nbody::visualization::SimulationController controller(
        initial,
        {dt_s, 3, nbody::visualization::BackendKind::sequential, 1, 0});

    controller.advance_steps(2);
    const auto switch_state = controller.state();
    controller.select_backend(nbody::visualization::BackendKind::cuda, 1, 0);
    require(controller.backend_kind() == nbody::visualization::BackendKind::cuda &&
                std::string(controller.backend_name()) == "cuda" &&
                same_state(controller.state(), switch_state),
            "switching to CUDA reset or changed the state");

    const auto reference = integrate_sequential(switch_state, 3, dt_s);
    controller.advance_frame();
    compare_states(reference, controller.state(), "batched CUDA frame");
    require(controller.last_execution() &&
                controller.last_execution()->backend == nbody::visualization::BackendKind::cuda &&
                controller.last_execution()->solver == nbody::visualization::SolverKind::direct &&
                controller.last_execution()->steps == 3U,
            "committed CUDA execution is not identified as CUDA Direct");

    controller.set_paused(true);
    const auto paused_state = controller.state();
    controller.advance_frame();
    require(same_state(paused_state, controller.state()), "paused CUDA controller advanced");

    controller.reset();
    require(same_state(initial, controller.state()), "CUDA reset did not restore initial state");
    controller.set_paused(false);
    controller.advance_steps(7);
    compare_states(
        integrate_sequential(initial, 7, dt_s), controller.state(), "seven-step CUDA batch");

    const auto before_cpu_switch = controller.state();
    controller.select_backend(nbody::visualization::BackendKind::sequential);
    require(same_state(before_cpu_switch, controller.state()),
            "switching from CUDA to Sequential changed state");

    if (nbody::visualization::SimulationController::openmp_available()) {
        const auto openmp_reference = integrate_sequential(controller.state(), 1U, dt_s);
        controller.select_backend(nbody::visualization::BackendKind::openmp, 2);
        controller.advance_steps(1U);
        compare_states(openmp_reference, controller.state(), "OpenMP Direct in CUDA-enabled viewer");
        require(controller.last_execution() &&
                    controller.last_execution()->backend == nbody::visualization::BackendKind::openmp,
                "CUDA-enabled viewer reported an OpenMP request without executing OpenMP");
        controller.select_solver(nbody::visualization::SolverKind::barnes_hut);
        controller.advance_steps(1U);
        require(controller.barnes_hut_statistics() &&
                    controller.last_execution() &&
                    controller.barnes_hut_statistics()->worker_threads == 2 &&
                    controller.last_execution()->solver == nbody::visualization::SolverKind::barnes_hut,
                "OpenMP Barnes-Hut is inactive in the combined CUDA build");
        controller.select_solver(nbody::visualization::SolverKind::direct);
    }

    const auto replacement = nbody::make_seeded_synthetic_system(16, 91);
    controller.select_backend(nbody::visualization::BackendKind::cuda, 1, 0);
    controller.replace_scenario(replacement, dt_s, 2);
    require(!controller.paused() &&
                controller.backend_kind() == nbody::visualization::BackendKind::cuda &&
                same_state(replacement, controller.state()),
            "running CUDA scenario replacement changed run state, backend, or state");
    controller.advance_frame();
    compare_states(
        integrate_sequential(replacement, 2, dt_s),
        controller.state(),
        "replacement CUDA frame");

    controller.set_paused(true);
    const auto paused_replacement = nbody::make_seeded_synthetic_system(12, 92);
    controller.replace_scenario(paused_replacement, dt_s, 4);
    require(controller.paused() &&
                controller.backend_kind() == nbody::visualization::BackendKind::cuda &&
                same_state(paused_replacement, controller.state()),
            "paused CUDA scenario replacement changed pause, backend, or initial state");
    const auto paused_replacement_state = controller.state();
    controller.advance_frame();
    require(same_state(paused_replacement_state, controller.state()),
            "paused CUDA replacement advanced through advance_frame");
    controller.advance_steps(2);
    compare_states(
        integrate_sequential(paused_replacement, 2, dt_s),
        controller.state(),
        "paused replacement explicit CUDA steps");
}

void test_transactional_failures() {
    const auto initial = nbody::make_seeded_synthetic_system(8, 73);
    nbody::visualization::SimulationController controller(
        initial,
        {10.0, 1, nbody::visualization::BackendKind::sequential, 1, 0});
    const auto state_before_switch = controller.state();
    bool switch_rejected = false;
    try {
        controller.select_backend(nbody::visualization::BackendKind::cuda, 1, 1000000);
    } catch (const std::exception&) {
        switch_rejected = true;
    }
    require(switch_rejected &&
                controller.backend_kind() == nbody::visualization::BackendKind::sequential &&
                same_state(state_before_switch, controller.state()),
            "failed CUDA switch changed backend or state");

    auto coincident = nbody::make_seeded_synthetic_system(8, 74);
    coincident.bodies[1].position_m = coincident.bodies[0].position_m;
    nbody::visualization::SimulationController failing_controller(
        coincident,
        {10.0, 1, nbody::visualization::BackendKind::cuda, 1, 0});
    const auto state_before_advance = failing_controller.state();
    bool advance_rejected = false;
    try {
        failing_controller.advance_frame();
    } catch (const std::domain_error&) {
        advance_rejected = true;
    }
    require(advance_rejected && same_state(state_before_advance, failing_controller.state()),
            "failed CUDA advancement changed host state");
}

}  // namespace

int main() {
    try {
        const auto device_decision =
            nbody::test_support::strict_cuda_runtime_test_decision(
                [] { return nbody::cuda_device_count(); });
        if (device_decision ==
            nbody::test_support::CudaRuntimeTestDecision::skip_no_device) {
            std::cout << "SKIP: CUDA visualization support compiled, but no CUDA device is available\n";
            return 77;
        }
        test_cuda_controller();
        test_transactional_failures();
        std::cout << "all CUDA visualization integration tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
