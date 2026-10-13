#include "nbody/astronomy/solar_system_60.hpp"
#include "nbody/barnes_hut.hpp"
#include "nbody/cuda_backend.hpp"
#include "nbody/cuda_barnes_hut.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"
#include "nbody/solar_system.hpp"

#include "cuda_test_gate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef NBODY_CUDA_BARNES_HUT_ENABLE_TEST_HOOKS
#error "CUDA Barnes-Hut integration tests require the isolated test-hook library"
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

template <typename Exception, typename Function>
std::string require_throws_with_message(
    Function&& function,
    const std::string& message) {
    try {
        function();
    } catch (const Exception& error) {
        return error.what();
    }
    throw std::runtime_error(message);
}

bool same_double_bits(double left, double right) {
    std::uint64_t left_bits = 0;
    std::uint64_t right_bits = 0;
    static_assert(sizeof(left_bits) == sizeof(left), "binary64 is required");
    std::memcpy(&left_bits, &left, sizeof(left));
    std::memcpy(&right_bits, &right, sizeof(right));
    return left_bits == right_bits;
}

bool exactly_equal(
    const nbody::SimulationState& left,
    const nbody::SimulationState& right) {
    if (!same_double_bits(left.time_s, right.time_s) ||
        left.bodies.size() != right.bodies.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.bodies.size(); ++index) {
        const auto& a = left.bodies[index];
        const auto& b = right.bodies[index];
        if (a.name != b.name || !same_double_bits(a.mass_kg, b.mass_kg) ||
            !same_double_bits(a.position_m.x, b.position_m.x) ||
            !same_double_bits(a.position_m.y, b.position_m.y) ||
            !same_double_bits(a.position_m.z, b.position_m.z) ||
            !same_double_bits(a.velocity_m_s.x, b.velocity_m_s.x) ||
            !same_double_bits(a.velocity_m_s.y, b.velocity_m_s.y) ||
            !same_double_bits(a.velocity_m_s.z, b.velocity_m_s.z)) {
            return false;
        }
    }
    return true;
}

bool same_barnes_hut_statistics(
    const nbody::BarnesHutStatistics& left,
    const nbody::BarnesHutStatistics& right) {
    return left.body_count == right.body_count &&
        left.node_count == right.node_count &&
        left.leaf_count == right.leaf_count &&
        left.maximum_depth_reached == right.maximum_depth_reached &&
        left.visited_nodes == right.visited_nodes &&
        left.approximated_nodes == right.approximated_nodes &&
        left.direct_body_interactions == right.direct_body_interactions &&
        left.working_memory_bytes == right.working_memory_bytes &&
        left.worker_threads == right.worker_threads &&
        same_double_bits(left.tree_build_seconds, right.tree_build_seconds) &&
        same_double_bits(
            left.force_evaluation_seconds,
            right.force_evaluation_seconds);
}

bool same_force_statistics(
    const nbody::CudaBarnesHutStatistics& left,
    const nbody::CudaBarnesHutStatistics& right) {
    return left.body_count == right.body_count &&
        left.node_count == right.node_count &&
        left.device_storage_bytes == right.device_storage_bytes &&
        left.host_staging_bytes == right.host_staging_bytes &&
        left.traversal_stack_bytes_per_thread ==
            right.traversal_stack_bytes_per_thread &&
        left.traversal_work == right.traversal_work &&
        same_barnes_hut_statistics(left.host_tree, right.host_tree);
}

bool same_integration_statistics(
    const nbody::CudaBarnesHutIntegrationStatistics& left,
    const nbody::CudaBarnesHutIntegrationStatistics& right) {
    return same_force_statistics(left.first_force, right.first_force) &&
        same_force_statistics(left.second_force, right.second_force) &&
        left.step_traversal_work == right.step_traversal_work &&
        left.step_traversal_work_limit == right.step_traversal_work_limit &&
        left.host_integration_storage_bytes ==
            right.host_integration_storage_bytes;
}

double stable_difference(const nbody::Vec3& left, const nbody::Vec3& right) {
    return std::hypot(
        left.x - right.x,
        left.y - right.y,
        left.z - right.z);
}

void require_mixed_close(
    const nbody::Vec3& reference,
    const nbody::Vec3& candidate,
    double absolute_tolerance,
    double relative_tolerance,
    const std::string& label) {
    const double difference = stable_difference(reference, candidate);
    const double scale = std::max(nbody::norm(reference), 1.0e-30);
    require(std::isfinite(difference) &&
                (difference <= absolute_tolerance ||
                 difference / scale <= relative_tolerance),
            label + " exceeded mixed tolerance");
}

void require_scalar_mixed_close(
    double reference,
    double candidate,
    double absolute_tolerance,
    double relative_tolerance,
    const std::string& label) {
    const double difference = std::abs(reference - candidate);
    const double scale = std::max(std::abs(reference), 1.0e-30);
    require(std::isfinite(reference) && std::isfinite(candidate) &&
                std::isfinite(difference) &&
                (difference <= absolute_tolerance ||
                 difference / scale <= relative_tolerance),
            label + " exceeded mixed tolerance");
}

struct TrajectoryErrorMetrics {
    double maximum_position_m{0.0};
    double rms_position_m{0.0};
    double maximum_velocity_m_s{0.0};
    double rms_velocity_m_s{0.0};
};

TrajectoryErrorMetrics trajectory_error_metrics(
    const nbody::SimulationState& approximation,
    const nbody::SimulationState& direct,
    const std::string& label) {
    require(approximation.time_s == direct.time_s &&
                approximation.bodies.size() == direct.bodies.size(),
            label + " does not use matched times/body counts");
    long double position_square_sum = 0.0L;
    long double velocity_square_sum = 0.0L;
    TrajectoryErrorMetrics metrics;
    for (std::size_t index = 0; index < approximation.bodies.size(); ++index) {
        const auto& candidate = approximation.bodies[index];
        const auto& reference = direct.bodies[index];
        require(candidate.name == reference.name &&
                    candidate.mass_kg == reference.mass_kg,
                label + " changed body identity/order");
        const double position_error =
            stable_difference(candidate.position_m, reference.position_m);
        const double velocity_error =
            stable_difference(candidate.velocity_m_s, reference.velocity_m_s);
        require(std::isfinite(position_error) && std::isfinite(velocity_error),
                label + " contains a non-finite trajectory error");
        metrics.maximum_position_m =
            std::max(metrics.maximum_position_m, position_error);
        metrics.maximum_velocity_m_s =
            std::max(metrics.maximum_velocity_m_s, velocity_error);
        position_square_sum +=
            static_cast<long double>(position_error) * position_error;
        velocity_square_sum +=
            static_cast<long double>(velocity_error) * velocity_error;
    }
    if (!approximation.bodies.empty()) {
        const long double count =
            static_cast<long double>(approximation.bodies.size());
        metrics.rms_position_m =
            std::sqrt(static_cast<double>(position_square_sum / count));
        metrics.rms_velocity_m_s =
            std::sqrt(static_cast<double>(velocity_square_sum / count));
    }
    require(std::isfinite(metrics.rms_position_m) &&
                std::isfinite(metrics.rms_velocity_m_s),
            label + " RMS trajectory error is non-finite");
    return metrics;
}

void require_state_close(
    const nbody::SimulationState& reference,
    const nbody::SimulationState& candidate,
    const std::string& label) {
    require(reference.time_s == candidate.time_s,
            label + " changed timestep semantics");
    require(reference.bodies.size() == candidate.bodies.size(),
            label + " body count differs");
    for (std::size_t index = 0; index < reference.bodies.size(); ++index) {
        require(reference.bodies[index].name == candidate.bodies[index].name &&
                    reference.bodies[index].mass_kg == candidate.bodies[index].mass_kg,
                label + " changed body identity or mass");
        require_mixed_close(
            reference.bodies[index].position_m,
            candidate.bodies[index].position_m,
            1.0e-4,
            2.0e-11,
            label + " position");
        require_mixed_close(
            reference.bodies[index].velocity_m_s,
            candidate.bodies[index].velocity_m_s,
            1.0e-8,
            2.0e-11,
            label + " velocity");
    }
}

nbody::CudaBarnesHutIntegrationConfiguration configuration(double theta = 0.25) {
    nbody::CudaBarnesHutIntegrationConfiguration value;
    value.force.tree.theta = theta;
    value.force.tree.leaf_capacity = 4;
    value.force.tree.maximum_nodes = 100'000;
    value.force.tree.maximum_traversal_work = 10'000'000;
    value.force.tree.memory_budget_bytes = 64U * 1024U * 1024U;
    value.force.device_memory_budget_bytes = 64U * 1024U * 1024U;
    value.force.host_staging_memory_budget_bytes = 64U * 1024U * 1024U;
    value.host_integration_memory_budget_bytes = 64U * 1024U * 1024U;
    return value;
}

void advance_cpu(
    nbody::SimulationState& state,
    std::size_t steps,
    double dt_s,
    const nbody::BarnesHutConfiguration& configuration,
    double gravitational_constant_value = nbody::gravitational_constant) {
    nbody::SequentialBarnesHutForceBackend backend(configuration);
    for (std::size_t step = 0; step < steps; ++step) {
        nbody::velocity_verlet_step(
            state, dt_s, backend, gravitational_constant_value);
    }
}

nbody::SimulationState astronomical_subset() {
    const auto catalog = nbody::astronomy::make_solar_system_60_catalog();
    nbody::SimulationState state;
    const std::size_t count = std::min<std::size_t>(16, catalog.records().size());
    state.bodies.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto& record = catalog.records()[index];
        state.bodies.push_back({
            record.stable_id(),
            record.mass_kg(),
            record.position_m(),
            record.velocity_m_s(),
        });
    }
    return state;
}

nbody::SimulationState planar_state(std::size_t count) {
    auto state = nbody::make_seeded_synthetic_system(count, 0x504c414e4152ULL);
    for (auto& body : state.bodies) {
        body.position_m.z = 0.0;
        body.velocity_m_s.z = 0.0;
    }
    return state;
}

nbody::SimulationState clustered_state(std::size_t count) {
    auto state = nbody::make_seeded_synthetic_system(count, 0x434c5553544552ULL);
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        const double center = index < count / 2 ? -2.0e9 : 2.0e9;
        state.bodies[index].position_m.x =
            center + static_cast<double>(index % 11) * 1.0e5;
        state.bodies[index].position_m.y =
            static_cast<double>((index * 7) % 13) * 2.0e5;
        state.bodies[index].position_m.z =
            static_cast<double>((index * 5) % 17) * 3.0e5;
    }
    return state;
}

void compare_fixture(
    nbody::SimulationState initial,
    std::size_t steps,
    double dt_s,
    double theta,
    const std::string& label) {
    const auto config = configuration(theta);
    auto reference = initial;
    advance_cpu(reference, steps, dt_s, config.force.tree);

    nbody::CudaBarnesHutVelocityVerlet cuda(config);
    for (std::size_t step = 0; step < steps; ++step) {
        cuda.advance(initial, dt_s);
        const auto stats = cuda.last_statistics();
        require(stats.first_force.body_count == initial.bodies.size() &&
                    stats.second_force.body_count == initial.bodies.size(),
                label + " omitted a force evaluation");
        require(stats.step_traversal_work ==
                    stats.first_force.traversal_work +
                        stats.second_force.traversal_work &&
                    stats.step_traversal_work <= stats.step_traversal_work_limit,
                label + " violated the combined traversal-work contract");
        require(stats.host_integration_storage_bytes <=
                    config.host_integration_memory_budget_bytes,
                label + " exceeded host integration admission");
    }
    require_state_close(reference, initial, label);
}

void test_zero_one_two_body_semantics() {
    nbody::CudaBarnesHutVelocityVerlet integrator(configuration());

    nbody::SimulationState empty;
    empty.time_s = 7.0;
    integrator.advance(empty, 3.0, 1.0);
    require(empty.time_s == 10.0 && empty.bodies.empty(),
            "empty integration changed state incorrectly");
    require(integrator.last_statistics().step_traversal_work == 0,
            "empty integration reported force work");

    nbody::SimulationState single;
    single.bodies.push_back({"single", 3.0, {1.0, -2.0, 4.0}, {2.0, 3.0, -1.0}});
    integrator.advance(single, 2.0, 1.0);
    require(single.time_s == 2.0 &&
                single.bodies[0].position_m.x == 5.0 &&
                single.bodies[0].position_m.y == 4.0 &&
                single.bodies[0].position_m.z == 2.0 &&
                single.bodies[0].velocity_m_s.x == 2.0 &&
                single.bodies[0].velocity_m_s.y == 3.0 &&
                single.bodies[0].velocity_m_s.z == -1.0,
            "single-body inertial motion is incorrect");

    compare_fixture(
        nbody::make_analytic_circular_two_body().initial_state,
        4,
        10.0,
        0.25,
        "analytic two body");
}

void test_scientific_fixtures_and_multiple_steps() {
    compare_fixture(
        nbody::make_approximate_solar_system().initial_state,
        3,
        60.0,
        0.25,
        "legacy Solar fixture");
    compare_fixture(astronomical_subset(), 2, 1.0, 0.25, "astronomical subset");
    compare_fixture(
        nbody::make_seeded_synthetic_system(96, 0x554e49464f524dULL),
        3,
        5.0,
        0.25,
        "uniform fixture");
    compare_fixture(clustered_state(96), 2, 1.0, 0.25, "clustered fixture");
    compare_fixture(planar_state(96), 2, 1.0, 0.25, "planar fixture");
    compare_fixture(clustered_state(64), 2, 1.0, 1.0e-6, "small theta");
    compare_fixture(clustered_state(64), 2, 1.0, 2.0, "large theta");
}

void test_conservation_and_direct_characterization() {
    auto candidate = nbody::make_approximate_solar_system().initial_state;
    auto reference = candidate;
    auto direct = candidate;
    const auto initial_diagnostics = nbody::compute_diagnostics(candidate);
    const auto config = configuration(0.25);
    nbody::CudaBarnesHutVelocityVerlet cuda(config);
    nbody::SequentialBarnesHutForceBackend cpu(config.force.tree);
    for (std::size_t step = 0; step < 8; ++step) {
        cuda.advance(candidate, 60.0);
        nbody::velocity_verlet_step(reference, 60.0, cpu);
        nbody::velocity_verlet_step(direct, 60.0);
    }
    require_state_close(reference, candidate, "multi-step Solar comparison");
    const auto cuda_diagnostics = nbody::compute_diagnostics(candidate);
    const auto cpu_diagnostics = nbody::compute_diagnostics(reference);
    const double cuda_energy_drift = std::abs(
        (cuda_diagnostics.total_energy_j() - initial_diagnostics.total_energy_j()) /
        initial_diagnostics.total_energy_j());
    const double cpu_energy_drift = std::abs(
        (cpu_diagnostics.total_energy_j() - initial_diagnostics.total_energy_j()) /
        initial_diagnostics.total_energy_j());
    require(std::isfinite(cuda_energy_drift) && cuda_energy_drift < 1.0e-8 &&
                std::isfinite(cpu_energy_drift) && cpu_energy_drift < 1.0e-8,
            "CUDA Barnes-Hut conservation characterization exceeded guard");
    require_mixed_close(
        cpu_diagnostics.linear_momentum_kg_m_s,
        cuda_diagnostics.linear_momentum_kg_m_s,
        1.0e10,
        5.0e-11,
        "CPU/CUDA final linear momentum");
    require_mixed_close(
        cpu_diagnostics.angular_momentum_kg_m2_s,
        cuda_diagnostics.angular_momentum_kg_m2_s,
        1.0e20,
        5.0e-11,
        "CPU/CUDA final angular momentum");

    const TrajectoryErrorMetrics cpu_direct =
        trajectory_error_metrics(reference, direct, "Sequential BH/direct");
    const TrajectoryErrorMetrics cuda_direct =
        trajectory_error_metrics(candidate, direct, "CUDA BH/direct");

    // Fixed before NVIDIA execution. Absolute floors avoid cancellation near a
    // zero approximation error; relative bounds compare the independently
    // produced Sequential-BH/direct and CUDA-BH/direct characterizations.
    constexpr double position_absolute_tolerance_m = 1.0e-3;
    constexpr double velocity_absolute_tolerance_m_s = 1.0e-9;
    constexpr double characterization_relative_tolerance = 5.0e-8;
    require_scalar_mixed_close(
        cpu_direct.maximum_position_m,
        cuda_direct.maximum_position_m,
        position_absolute_tolerance_m,
        characterization_relative_tolerance,
        "maximum BH/direct position separation");
    require_scalar_mixed_close(
        cpu_direct.rms_position_m,
        cuda_direct.rms_position_m,
        position_absolute_tolerance_m,
        characterization_relative_tolerance,
        "RMS BH/direct position separation");
    require_scalar_mixed_close(
        cpu_direct.maximum_velocity_m_s,
        cuda_direct.maximum_velocity_m_s,
        velocity_absolute_tolerance_m_s,
        characterization_relative_tolerance,
        "maximum BH/direct velocity separation");
    require_scalar_mixed_close(
        cpu_direct.rms_velocity_m_s,
        cuda_direct.rms_velocity_m_s,
        velocity_absolute_tolerance_m_s,
        characterization_relative_tolerance,
        "RMS BH/direct velocity separation");

    for (std::size_t index = 0; index < direct.bodies.size(); ++index) {
        require_mixed_close(
            reference.bodies[index].position_m - direct.bodies[index].position_m,
            candidate.bodies[index].position_m - direct.bodies[index].position_m,
            position_absolute_tolerance_m,
            characterization_relative_tolerance,
            "per-body BH/direct position-separation vector");
        require_mixed_close(
            reference.bodies[index].velocity_m_s -
                direct.bodies[index].velocity_m_s,
            candidate.bodies[index].velocity_m_s -
                direct.bodies[index].velocity_m_s,
            velocity_absolute_tolerance_m_s,
            characterization_relative_tolerance,
            "per-body BH/direct velocity-separation vector");
    }
}

void test_transactional_input_and_first_force_failures() {
    nbody::CudaBarnesHutVelocityVerlet integrator(configuration());
    auto state = nbody::make_analytic_circular_two_body().initial_state;
    integrator.advance(state, 1.0);
    const auto committed_state = state;
    const auto successful_before = integrator.last_statistics();
    require_throws<std::invalid_argument>(
        [&] { integrator.advance(state, 0.0); },
        "zero timestep was accepted");
    require(exactly_equal(state, committed_state),
            "invalid timestep changed caller state");
    require(integrator.last_failure().phase ==
                nbody::CudaBarnesHutIntegrationPhase::input_validation &&
                !integrator.last_failure().executor_terminal,
            "invalid timestep reported the wrong failure/reuse state");
    require(integrator.last_statistics().step_traversal_work ==
                successful_before.step_traversal_work &&
                same_integration_statistics(
                    integrator.last_statistics(), successful_before),
            "invalid timestep changed last-success statistics");

    nbody::SimulationState singular;
    singular.bodies = {
        {"a", 1.0, {0.0, 0.0, 0.0}, {}},
        {"b", 1.0, {0.0, 0.0, 0.0}, {}},
    };
    const auto singular_original = singular;
    require_throws<std::domain_error>(
        [&] { integrator.advance(singular, 1.0, 1.0); },
        "first-force singularity was accepted");
    require(exactly_equal(singular, singular_original),
            "first-force failure changed caller state");
    const auto failure = integrator.last_failure();
    require(failure.phase ==
                nbody::CudaBarnesHutIntegrationPhase::first_force_evaluation &&
                failure.force_failure.code ==
                    nbody::CudaBarnesHutFailureCode::singular_geometry &&
                !failure.executor_terminal,
            "first-force failure metadata is incorrect");

    require_throws<std::invalid_argument>(
        [&] { integrator.advance(singular, 0.0, 1.0); },
        "invalid timestep after a force failure was accepted");
    require(integrator.last_failure().phase ==
                nbody::CudaBarnesHutIntegrationPhase::input_validation &&
                integrator.last_failure().force_failure.code ==
                    nbody::CudaBarnesHutFailureCode::none,
            "input validation reused stale force-failure metadata");

    integrator.advance(state, 1.0);
    require(!integrator.is_terminal() &&
                integrator.last_failure().phase ==
                    nbody::CudaBarnesHutIntegrationPhase::none,
            "nonterminal first-force failure prevented valid reuse");
}

void test_transactional_second_force_failure() {
    nbody::SimulationState collision;
    collision.bodies = {
        {"left", 1.0, {-1.0, 0.0, 0.0}, {0.875, 0.0, 0.0}},
        {"right", 1.0, {1.0, 0.0, 0.0}, {-0.875, 0.0, 0.0}},
    };
    const auto original = collision;
    nbody::CudaBarnesHutVelocityVerlet integrator(configuration());
    require_throws<std::domain_error>(
        [&] { integrator.advance(collision, 1.0, 1.0); },
        "second-force singularity was accepted");
    require(exactly_equal(collision, original),
            "second-force failure exposed drifted state");
    const auto failure = integrator.last_failure();
    require(failure.phase ==
                nbody::CudaBarnesHutIntegrationPhase::second_force_evaluation &&
                failure.force_failure.code ==
                    nbody::CudaBarnesHutFailureCode::singular_geometry &&
                !failure.executor_terminal,
            "second-force failure metadata is incorrect");

    auto valid = nbody::make_analytic_circular_two_body().initial_state;
    integrator.advance(valid, 1.0);
    require(!integrator.is_terminal(),
            "nonterminal second-force failure prevented valid reuse");
}

void test_transactional_drift_and_kick_failures() {
    nbody::CudaBarnesHutVelocityVerlet integrator(configuration());
    auto successful = nbody::make_analytic_circular_two_body().initial_state;
    integrator.advance(successful, 1.0);
    const auto before_drift_failure = integrator.last_statistics();

    nbody::SimulationState drift_overflow;
    drift_overflow.bodies = {{
        "finite-drift-overflow",
        1.0,
        {},
        {std::numeric_limits<double>::max(), 0.0, 0.0},
    }};
    const auto drift_original = drift_overflow;
    require_throws<std::domain_error>(
        [&] { integrator.advance(drift_overflow, 2.0, 1.0); },
        "finite-input drift overflow was accepted");
    require(exactly_equal(drift_overflow, drift_original),
            "drift failure changed caller state bits");
    require(same_integration_statistics(
                integrator.last_statistics(), before_drift_failure),
            "drift failure changed last-success statistics");
    require(integrator.last_failure().phase ==
                nbody::CudaBarnesHutIntegrationPhase::position_update &&
                integrator.last_failure().force_failure.code ==
                    nbody::CudaBarnesHutFailureCode::none &&
                !integrator.last_failure().executor_terminal &&
                !integrator.is_terminal(),
            "drift failure phase/reuse metadata is incorrect");

    auto reuse_after_drift = nbody::make_analytic_circular_two_body().initial_state;
    integrator.advance(reuse_after_drift, 1.0);
    require(integrator.last_failure().phase ==
                nbody::CudaBarnesHutIntegrationPhase::none,
            "healthy executor did not recover after drift failure");
    const auto before_kick_failure = integrator.last_statistics();

    // Both force calls return finite acceleration. dt^2 underflows to zero, so
    // drift is exactly unchanged; adding the two approximately 1e308 target
    // accelerations overflows only at the real average-kick expression.
    nbody::SimulationState kick_overflow;
    kick_overflow.bodies = {
        {"target", 1.0, {-0.5, 0.0, 0.0}, {}},
        {"source", 1.0e308, {0.5, 0.0, 0.0}, {}},
    };
    const auto kick_original = kick_overflow;
    require_throws<std::domain_error>(
        [&] { integrator.advance(kick_overflow, 1.0e-200, 1.0); },
        "finite-input velocity-kick overflow was accepted");
    require(exactly_equal(kick_overflow, kick_original),
            "velocity-kick failure changed caller state bits");
    require(same_integration_statistics(
                integrator.last_statistics(), before_kick_failure),
            "velocity-kick failure changed last-success statistics");
    require(integrator.last_failure().phase ==
                nbody::CudaBarnesHutIntegrationPhase::velocity_update &&
                integrator.last_failure().force_failure.code ==
                    nbody::CudaBarnesHutFailureCode::none &&
                !integrator.last_failure().executor_terminal &&
                !integrator.is_terminal(),
            "velocity-kick failure phase/reuse metadata is incorrect");

    auto reuse_after_kick = nbody::make_analytic_circular_two_body().initial_state;
    integrator.advance(reuse_after_kick, 1.0);
    require(integrator.last_failure().phase ==
                nbody::CudaBarnesHutIntegrationPhase::none,
            "healthy executor did not recover after velocity-kick failure");
}

void test_nonrepresentable_force_geometry_rollback() {
    nbody::CudaBarnesHutVelocityVerlet integrator(configuration());
    auto successful = nbody::make_analytic_circular_two_body().initial_state;
    integrator.advance(successful, 1.0);
    const auto successful_before = integrator.last_statistics();

    nbody::SimulationState nonrepresentable_geometry;
    nonrepresentable_geometry.bodies = {
        {"origin", 1.0, {}, {}},
        {"diagonal", 1.0, {1.4e308, 1.4e308, 0.0}, {}},
    };
    const auto original = nonrepresentable_geometry;
    require_throws<std::domain_error>(
        [&] { integrator.advance(nonrepresentable_geometry, 1.0, 1.0); },
        "nonrepresentable force geometry was accepted");
    require(exactly_equal(nonrepresentable_geometry, original),
            "nonrepresentable force geometry changed caller state bits");
    require(same_integration_statistics(
                integrator.last_statistics(), successful_before),
            "nonrepresentable force geometry changed last-success statistics");
    require(integrator.last_failure().phase ==
                nbody::CudaBarnesHutIntegrationPhase::first_force_evaluation &&
                integrator.last_failure().force_failure.code ==
                    nbody::CudaBarnesHutFailureCode::nonrepresentable_geometry &&
                !integrator.last_failure().executor_terminal &&
                !integrator.is_terminal(),
            "nonrepresentable force failure metadata/reuse state is incorrect");

    auto reusable = nbody::make_analytic_circular_two_body().initial_state;
    integrator.advance(reusable, 1.0);
    require(integrator.last_failure().phase ==
                nbody::CudaBarnesHutIntegrationPhase::none,
            "nonterminal geometry failure prevented safe reuse");
}

void test_post_force_resource_failure_classification() {
    const auto exercise = [](
                              nbody::CudaBarnesHutIntegrationConfiguration::
                                  TestResourceFailurePoint point,
                              const std::string& diagnostic_fragment) {
        auto config = configuration();
        config.test_resource_failure_point = point;
        config.test_resource_failure_advance = 2;
        nbody::CudaBarnesHutVelocityVerlet integrator(config);

        auto committed = nbody::make_analytic_circular_two_body().initial_state;
        integrator.advance(committed, 1.0);
        const auto successful_before = integrator.last_statistics();
        auto candidate = nbody::make_analytic_circular_two_body().initial_state;
        const auto original = candidate;
        const std::string diagnostic = require_throws_with_message<std::length_error>(
            [&] { integrator.advance(candidate, 1.0); },
            "post-force resource test hook did not fail");
        require(diagnostic.find(diagnostic_fragment) != std::string::npos,
                "post-force resource failure used the wrong diagnostic");
        require(exactly_equal(candidate, original),
                "post-force resource failure changed caller state bits");
        require(same_integration_statistics(
                    integrator.last_statistics(), successful_before),
                "post-force resource failure changed last-success statistics");
        require(integrator.last_failure().phase ==
                    nbody::CudaBarnesHutIntegrationPhase::
                        integration_post_force_validation &&
                    integrator.last_failure().force_failure.code ==
                        nbody::CudaBarnesHutFailureCode::none &&
                    !integrator.last_failure().executor_terminal &&
                    !integrator.is_terminal(),
                "post-force host accounting failure was misclassified");

        integrator.advance(candidate, 1.0);
        require(integrator.last_failure().phase ==
                    nbody::CudaBarnesHutIntegrationPhase::none,
                "post-force host accounting failure prevented safe reuse");
    };

    exercise(
        nbody::CudaBarnesHutIntegrationConfiguration::
            TestResourceFailurePoint::after_first_force,
        "after first force evaluation");
    exercise(
        nbody::CudaBarnesHutIntegrationConfiguration::
            TestResourceFailurePoint::after_second_force,
        "after second force evaluation");
}

void test_resource_failures_and_terminal_reuse() {
    auto memory_config = configuration();
    memory_config.host_integration_memory_budget_bytes = 1;
    nbody::CudaBarnesHutVelocityVerlet memory_limited(memory_config);
    auto state = nbody::make_seeded_synthetic_system(8, 7);
    const auto original = state;
    require_throws<std::length_error>(
        [&] { memory_limited.advance(state, 1.0); },
        "host integration memory exhaustion was accepted");
    require(exactly_equal(state, original) && !memory_limited.is_terminal(),
            "host admission failure changed state or terminalized resources");

    auto device_memory_config = configuration();
    device_memory_config.force.device_memory_budget_bytes = 1;
    nbody::CudaBarnesHutVelocityVerlet device_memory_limited(
        device_memory_config);
    state = nbody::make_seeded_synthetic_system(8, 8);
    const auto device_memory_original = state;
    require_throws<std::length_error>(
        [&] { device_memory_limited.advance(state, 1.0); },
        "device memory exhaustion was accepted");
    require(exactly_equal(state, device_memory_original) &&
                !device_memory_limited.is_terminal(),
            "device memory admission changed state or terminalized resources");

    auto staging_config = configuration();
    staging_config.force.host_staging_memory_budget_bytes = 1;
    nbody::CudaBarnesHutVelocityVerlet staging_limited(staging_config);
    state = nbody::make_seeded_synthetic_system(8, 10);
    const auto staging_original = state;
    require_throws<std::length_error>(
        [&] { staging_limited.advance(state, 1.0); },
        "force host-staging memory exhaustion was accepted");
    require(exactly_equal(state, staging_original) &&
                !staging_limited.is_terminal(),
            "force staging admission changed state or terminalized resources");

    auto work_config = configuration();
    work_config.force.tree.maximum_traversal_work = 1;
    nbody::CudaBarnesHutVelocityVerlet work_limited(work_config);
    state = nbody::make_seeded_synthetic_system(16, 9);
    const auto work_original = state;
    require_throws<std::length_error>(
        [&] { work_limited.advance(state, 1.0); },
        "force traversal-work exhaustion was accepted");
    require(exactly_equal(state, work_original) && !work_limited.is_terminal() &&
                work_limited.last_failure().phase ==
                    nbody::CudaBarnesHutIntegrationPhase::first_force_evaluation,
            "work exhaustion violated rollback or reuse policy");

    auto stack_config = configuration();
    stack_config.force.tree.leaf_capacity = 1;
    stack_config.force.tree.theta = 1.0e-6;
    stack_config.force.traversal_stack_capacity = 1;
    nbody::CudaBarnesHutVelocityVerlet stack_limited(stack_config);
    state = nbody::make_seeded_synthetic_system(32, 11);
    const auto stack_original = state;
    require_throws<std::length_error>(
        [&] { stack_limited.advance(state, 1.0); },
        "force traversal-stack exhaustion was accepted");
    require(exactly_equal(state, stack_original),
            "stack exhaustion changed caller state");

    auto terminal_config = configuration();
    terminal_config.force.host_failure_injection =
        nbody::CudaBarnesHutHostFailureInjection::
            live_release_before_first_deallocation;
    nbody::CudaBarnesHutVelocityVerlet terminal(terminal_config);
    auto small = nbody::make_seeded_synthetic_system(8, 13);
    terminal.advance(small, 1.0);
    const auto committed = terminal.last_statistics();
    auto large = nbody::make_seeded_synthetic_system(64, 15);
    const auto large_original = large;
    require_throws<std::runtime_error>(
        [&] { terminal.advance(large, 1.0); },
        "injected live-release failure was accepted");
    require(exactly_equal(large, large_original) && terminal.is_terminal() &&
                terminal.last_failure().executor_terminal &&
                terminal.last_statistics().step_traversal_work ==
                    committed.step_traversal_work,
            "terminal release failure violated rollback/statistics ownership");
    require_throws<std::runtime_error>(
        [&] { terminal.advance(large, 1.0); },
        "terminal CUDA Barnes-Hut integrator was reused");
}

void test_repeated_lifetime() {
    for (int repetition = 0; repetition < 8; ++repetition) {
        auto state = nbody::make_seeded_synthetic_system(24, 17 + repetition);
        nbody::CudaBarnesHutVelocityVerlet integrator(configuration());
        integrator.advance(state, 1.0);
        integrator.advance(state, 1.0);
    }
}

}  // namespace

int main() {
    try {
        const auto decision = nbody::test_support::strict_cuda_runtime_test_decision(
            [] { return nbody::cuda_device_count(); });
        if (decision == nbody::test_support::CudaRuntimeTestDecision::skip_no_device) {
            std::cout << "SKIP: CUDA support compiled, but no CUDA device is available\n";
            return 77;
        }
        test_zero_one_two_body_semantics();
        test_scientific_fixtures_and_multiple_steps();
        test_conservation_and_direct_characterization();
        test_transactional_input_and_first_force_failures();
        test_transactional_second_force_failure();
        test_transactional_drift_and_kick_failures();
        test_nonrepresentable_force_geometry_rollback();
        test_post_force_resource_failure_classification();
        test_resource_failures_and_terminal_reuse();
        test_repeated_lifetime();
        const auto device = nbody::cuda_device_info();
        std::cout << "CUDA Barnes-Hut integration validation passed on "
                  << device.name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA Barnes-Hut integration test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
