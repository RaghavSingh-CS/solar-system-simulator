#include "nbody/cuda_backend.hpp"
#include "nbody/experiments/counterfactual.hpp"
#include "cuda_test_gate.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using nbody::SimulationState;
using nbody::experiments::CounterfactualExperimentConfiguration;
using nbody::experiments::CounterfactualExperimentRunner;
using nbody::experiments::ExperimentBackend;
using nbody::experiments::ExperimentIntegrator;
using nbody::experiments::ExperimentSnapshot;
using nbody::experiments::ExperimentStatus;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool same_state(const SimulationState& left, const SimulationState& right) {
    if (left.time_s != right.time_s || left.bodies.size() != right.bodies.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.bodies.size(); ++index) {
        const auto& a = left.bodies[index];
        const auto& b = right.bodies[index];
        if (a.name != b.name || a.mass_kg != b.mass_kg ||
            a.position_m.x != b.position_m.x ||
            a.position_m.y != b.position_m.y ||
            a.position_m.z != b.position_m.z ||
            a.velocity_m_s.x != b.velocity_m_s.x ||
            a.velocity_m_s.y != b.velocity_m_s.y ||
            a.velocity_m_s.z != b.velocity_m_s.z) {
            return false;
        }
    }
    return true;
}

bool same_vector(const nbody::Vec3& left, const nbody::Vec3& right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool same_diagnostics(
    const nbody::experiments::BranchConservationDiagnostics& left,
    const nbody::experiments::BranchConservationDiagnostics& right) {
    return left.initial.kinetic_energy_j == right.initial.kinetic_energy_j &&
        left.initial.potential_energy_j == right.initial.potential_energy_j &&
        same_vector(left.initial.linear_momentum_kg_m_s,
            right.initial.linear_momentum_kg_m_s) &&
        same_vector(left.initial.angular_momentum_kg_m2_s,
            right.initial.angular_momentum_kg_m2_s) &&
        same_vector(left.initial.center_of_mass_m, right.initial.center_of_mass_m) &&
        left.current.kinetic_energy_j == right.current.kinetic_energy_j &&
        left.current.potential_energy_j == right.current.potential_energy_j &&
        same_vector(left.current.linear_momentum_kg_m_s,
            right.current.linear_momentum_kg_m_s) &&
        same_vector(left.current.angular_momentum_kg_m2_s,
            right.current.angular_momentum_kg_m2_s) &&
        same_vector(left.current.center_of_mass_m, right.current.center_of_mass_m) &&
        left.absolute_energy_drift_j == right.absolute_energy_drift_j &&
        left.relative_energy_drift == right.relative_energy_drift &&
        left.linear_momentum_drift_kg_m_s == right.linear_momentum_drift_kg_m_s &&
        left.angular_momentum_drift_kg_m2_s ==
            right.angular_momentum_drift_kg_m2_s;
}

bool same_committed_samples(
    const std::vector<nbody::experiments::ExperimentSample>& left,
    const std::vector<nbody::experiments::ExperimentSample>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t sample = 0; sample < left.size(); ++sample) {
        if (left[sample].elapsed_time_s != right[sample].elapsed_time_s ||
            left[sample].bodies.size() != right[sample].bodies.size() ||
            left[sample].baseline_reference_positions_m.size() !=
                right[sample].baseline_reference_positions_m.size()) {
            return false;
        }
        for (std::size_t body = 0; body < left[sample].bodies.size(); ++body) {
            const auto& a = left[sample].bodies[body];
            const auto& b = right[sample].bodies[body];
            if (!same_vector(a.baseline_position_m, b.baseline_position_m) ||
                !same_vector(a.counterfactual_position_m,
                    b.counterfactual_position_m) ||
                !same_vector(a.baseline_velocity_m_s, b.baseline_velocity_m_s) ||
                !same_vector(a.counterfactual_velocity_m_s,
                    b.counterfactual_velocity_m_s) ||
                a.absolute_separation_m != b.absolute_separation_m ||
                a.parent_relative_separation_m !=
                    b.parent_relative_separation_m ||
                a.normalized_deviation != b.normalized_deviation ||
                a.velocity_difference_m_s != b.velocity_difference_m_s) {
                return false;
            }
        }
        for (std::size_t reference = 0;
             reference < left[sample].baseline_reference_positions_m.size();
             ++reference) {
            if (!same_vector(
                    left[sample].baseline_reference_positions_m[reference],
                    right[sample].baseline_reference_positions_m[reference])) {
                return false;
            }
        }
    }
    return true;
}

bool same_committed_summaries(
    const std::vector<nbody::experiments::BodyDeviationSummary>& left,
    const std::vector<nbody::experiments::BodyDeviationSummary>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto& a = left[index];
        const auto& b = right[index];
        if (a.stable_id != b.stable_id ||
            a.fixed_normalization_scale_m != b.fixed_normalization_scale_m ||
            a.final_absolute_separation_m != b.final_absolute_separation_m ||
            a.maximum_absolute_separation_m != b.maximum_absolute_separation_m ||
            a.maximum_absolute_separation_time_s !=
                b.maximum_absolute_separation_time_s ||
            a.final_normalized_deviation != b.final_normalized_deviation ||
            a.final_velocity_difference_m_s != b.final_velocity_difference_m_s ||
            a.final_parent_relative_separation_m !=
                b.final_parent_relative_separation_m ||
            a.maximum_parent_relative_separation_m !=
                b.maximum_parent_relative_separation_m ||
            a.maximum_parent_relative_separation_time_s !=
                b.maximum_parent_relative_separation_time_s ||
            a.parent_relative_available != b.parent_relative_available ||
            a.maximum_normalized_deviation != b.maximum_normalized_deviation ||
            a.maximum_velocity_difference_m_s !=
                b.maximum_velocity_difference_m_s ||
            a.first_threshold_crossing_time_s !=
                b.first_threshold_crossing_time_s ||
            a.persistent_threshold_time_s != b.persistent_threshold_time_s ||
            a.significant != b.significant) {
            return false;
        }
    }
    return true;
}

bool same_committed_issues(
    const std::vector<nbody::experiments::ExperimentIssue>& left,
    const std::vector<nbody::experiments::ExperimentIssue>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto& a = left[index];
        const auto& b = right[index];
        if (a.kind != b.kind || a.numerical_invalidity != b.numerical_invalidity ||
            a.message != b.message || a.time_s != b.time_s ||
            a.first_body_stable_id != b.first_body_stable_id ||
            a.second_body_stable_id != b.second_body_stable_id ||
            a.branch != b.branch) {
            return false;
        }
    }
    return true;
}

ExperimentSnapshot snapshot(
    double time_s = 0.0,
    bool timestep_convergence_established = true) {
    SimulationState state;
    state.time_s = time_s;
    state.bodies = {
        {"Central", 1.0e20, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"Removed", 1.0e15, {1.0e7, 0.0, 0.0}, {0.0, 1.0, 0.0}},
        {"Tracer", 1.0e10, {2.0e7, 0.0, 0.0}, {0.0, 2.0, 0.0}},
    };
    return ExperimentSnapshot({
        "cuda-counterfactual-fixture",
        "deterministic CUDA counterfactual test fixture",
        "common inertial frame",
        "central",
        std::move(state),
        {
            {"central", std::nullopt, 1.0, false},
            {"removed", std::nullopt, 1.0, true},
            {"tracer", std::optional<std::string>{"central"}, 1.0, false},
        },
        1.0,
        ExperimentIntegrator::velocity_verlet,
        timestep_convergence_established,
        8.0,
    });
}

CounterfactualExperimentConfiguration configuration(ExperimentBackend backend) {
    CounterfactualExperimentConfiguration value;
    value.removed_body_stable_id = "removed";
    value.backend = backend;
    value.cuda_device = 0;
    value.limits.observation_horizon_s = 4.0;
    value.limits.sample_cadence_s = 1.0;
    value.limits.integration_steps_per_batch = 2U;
    value.limits.maximum_monitored_bodies = 2U;
    value.limits.maximum_stored_samples = 16U;
    value.limits.memory_budget_bytes = 8U * 1024U * 1024U;
    value.limits.workload_budget_interactions = 1000000ULL;
    value.significance.absolute_separation_threshold_m = 1.0e-12;
    value.significance.normalized_deviation_threshold = 0.0;
    value.significance.normalization_denominator_floor_m = 1.0;
    value.guards.maximum_relative_energy_drift = 1.0;
    value.monitored_body_stable_ids = {"central", "tracer"};
    return value;
}

double scaled_difference(double difference, double reference, double floor) {
    return difference / std::max(std::abs(reference), floor);
}

void compare_states(
    const SimulationState& reference,
    const SimulationState& candidate,
    const std::string& label) {
    require(reference.time_s == candidate.time_s, label + " simulated time differs");
    require(reference.bodies.size() == candidate.bodies.size(),
        label + " body count differs");
    for (std::size_t index = 0; index < reference.bodies.size(); ++index) {
        const double position_difference = nbody::norm(
            candidate.bodies[index].position_m - reference.bodies[index].position_m);
        const double velocity_difference = nbody::norm(
            candidate.bodies[index].velocity_m_s -
            reference.bodies[index].velocity_m_s);
        require(position_difference <= 1.0e-3 ||
                scaled_difference(position_difference,
                    nbody::norm(reference.bodies[index].position_m), 1.0) <= 2.0e-12,
            label + " position exceeded mixed tolerance");
        require(velocity_difference <= 1.0e-8 ||
                scaled_difference(velocity_difference,
                    nbody::norm(reference.bodies[index].velocity_m_s), 1.0) <= 2.0e-12,
            label + " velocity exceeded mixed tolerance");
    }
}

void test_cuda_pair_and_numerical_agreement() {
    auto sequential_configuration = configuration(ExperimentBackend::sequential);
    CounterfactualExperimentRunner sequential(snapshot(), sequential_configuration);
    sequential.run_to_completion();

    auto cuda_configuration = configuration(ExperimentBackend::cuda);
    CounterfactualExperimentRunner cuda(snapshot(), cuda_configuration);
    require(cuda.cuda_resources_valid(),
        "CUDA runner did not retain two initialized device executors");
    const auto& initial = cuda.result();
    require(initial.baseline_state.bodies.size() == 3U &&
            initial.counterfactual_state.bodies.size() == 2U &&
            initial.survivor_mapping.size() == 2U,
        "CUDA branch sizing or survivor mapping is incorrect");
    for (const auto& mapping : initial.survivor_mapping) {
        const auto& baseline = initial.baseline_state.bodies[mapping.baseline_index];
        const auto& counterfactual =
            initial.counterfactual_state.bodies[mapping.counterfactual_index];
        require(baseline.name == counterfactual.name &&
                baseline.mass_kg == counterfactual.mass_kg &&
                baseline.position_m.x == counterfactual.position_m.x &&
                baseline.position_m.y == counterfactual.position_m.y &&
                baseline.position_m.z == counterfactual.position_m.z &&
                baseline.velocity_m_s.x == counterfactual.velocity_m_s.x &&
                baseline.velocity_m_s.y == counterfactual.velocity_m_s.y &&
                baseline.velocity_m_s.z == counterfactual.velocity_m_s.z,
            "CUDA survivor initial state differs between paired branches");
    }
    require(cuda.advance_batch() && cuda.result().completed_steps == 2U &&
            cuda.result().samples.size() == 3U &&
            cuda.result().status == ExperimentStatus::running,
        "CUDA batch did not publish one complete synchronized paired boundary");
    cuda.run_to_completion();
    require(cuda.result().status == ExperimentStatus::completed &&
            cuda.result().completed_steps == 4U &&
            cuda.cuda_resources_valid(),
        "CUDA paired experiment did not complete with valid resources");
    compare_states(sequential.result().baseline_state,
        cuda.result().baseline_state, "CUDA baseline");
    compare_states(sequential.result().counterfactual_state,
        cuda.result().counterfactual_state, "CUDA counterfactual");
    require(cuda.result().monitored_body_stable_ids ==
            sequential.result().monitored_body_stable_ids &&
            cuda.result().summaries.size() == sequential.result().summaries.size(),
        "CUDA result metrics lost stable-ID correspondence");
    for (std::size_t index = 0; index < cuda.result().summaries.size(); ++index) {
        const auto& reference = sequential.result().summaries[index];
        const auto& candidate = cuda.result().summaries[index];
        require(reference.stable_id == candidate.stable_id &&
                reference.fixed_normalization_scale_m ==
                    candidate.fixed_normalization_scale_m &&
                std::abs(reference.final_absolute_separation_m -
                    candidate.final_absolute_separation_m) <= 1.0e-3 &&
                std::abs(reference.final_velocity_difference_m_s -
                    candidate.final_velocity_difference_m_s) <= 1.0e-8 &&
                reference.first_threshold_crossing_time_s ==
                    candidate.first_threshold_crossing_time_s &&
                reference.persistent_threshold_time_s ==
                    candidate.persistent_threshold_time_s,
            "CUDA scientific metrics exceeded mixed tolerances or changed sampling semantics");
    }
}

void test_cuda_cancellation_and_repeated_lifetime() {
    for (int repetition = 0; repetition < 2; ++repetition) {
        CounterfactualExperimentRunner runner(
            snapshot(), configuration(ExperimentBackend::cuda));
        require(runner.advance_batch(), "CUDA repeated runner did not advance");
        const auto committed_baseline = runner.result().baseline_state;
        const auto committed_counterfactual = runner.result().counterfactual_state;
        const auto committed_steps = runner.result().completed_steps;
        runner.request_cancel();
        require(!runner.advance_batch() &&
                runner.result().status == ExperimentStatus::cancelled &&
                runner.result().completed_steps == committed_steps &&
                same_state(runner.result().baseline_state, committed_baseline) &&
                same_state(runner.result().counterfactual_state,
                    committed_counterfactual),
            "CUDA cancellation changed a committed paired boundary");
    }
}

void test_recoverable_host_failure_after_enqueue() {
    CounterfactualExperimentRunner runner(
        snapshot(), configuration(ExperimentBackend::cuda));
    require(runner.advance_batch() && runner.result().completed_steps == 2U,
        "CUDA host-failure fixture did not establish a committed paired batch");
    const auto committed = runner.result();

    runner.inject_cuda_host_failure_after_enqueue_for_testing(
        nbody::experiments::ExperimentBranch::counterfactual);
    require(!runner.advance_batch() &&
            runner.result().status == ExperimentStatus::failed &&
            runner.result().terminal_failure.has_value() &&
            runner.result().terminal_failure->message_text().find(
                "injected host-side failure after CUDA work enqueue") !=
                std::string_view::npos &&
            !runner.cuda_resources_valid(),
        "injected post-enqueue host failure was not terminal or did not invalidate resources");
    require(runner.result().completed_steps == committed.completed_steps &&
            same_state(runner.result().baseline_state, committed.baseline_state) &&
            same_state(runner.result().counterfactual_state,
                committed.counterfactual_state) &&
            same_committed_samples(runner.result().samples, committed.samples) &&
            same_committed_summaries(runner.result().summaries,
                committed.summaries) &&
            same_committed_issues(runner.result().issues, committed.issues) &&
            runner.result().dropped_issue_count == committed.dropped_issue_count &&
            same_diagnostics(runner.result().baseline_conservation,
                committed.baseline_conservation) &&
            same_diagnostics(runner.result().counterfactual_conservation,
                committed.counterfactual_conservation),
        "injected post-enqueue host failure changed committed paired results");

    CounterfactualExperimentRunner fresh(
        snapshot(), configuration(ExperimentBackend::cuda));
    fresh.run_to_completion();
    CounterfactualExperimentRunner sequential(
        snapshot(), configuration(ExperimentBackend::sequential));
    sequential.run_to_completion();
    require(fresh.result().status == ExperimentStatus::completed &&
            fresh.result().completed_steps == 4U &&
            fresh.cuda_resources_valid(),
        "fresh CUDA executors failed after recoverable injected host failure cleanup");
    compare_states(sequential.result().baseline_state,
        fresh.result().baseline_state,
        "fresh CUDA baseline after injected host failure");
    compare_states(sequential.result().counterfactual_state,
        fresh.result().counterfactual_state,
        "fresh CUDA counterfactual after injected host failure");
}

void test_cuda_failure_rollback_and_invalidation() {
    auto failure_configuration = configuration(ExperimentBackend::cuda);
    failure_configuration.limits.observation_horizon_s = 2.0;
    CounterfactualExperimentRunner runner(
        snapshot(9007199254740991.0, false), failure_configuration);
    const auto committed_baseline = runner.result().baseline_state;
    const auto committed_counterfactual = runner.result().counterfactual_state;
    const auto committed_samples = runner.result().samples.size();
    const auto committed_issues = runner.result().issues;
    const auto committed_dropped = runner.result().dropped_issue_count;
    const auto committed_summaries = runner.result().summaries;
    const auto committed_baseline_conservation =
        runner.result().baseline_conservation;
    const auto committed_counterfactual_conservation =
        runner.result().counterfactual_conservation;
    require(!runner.advance_batch() &&
            runner.result().status == ExperimentStatus::failed &&
            runner.result().terminal_failure.has_value() &&
            runner.result().completed_steps == 0U &&
            same_state(runner.result().baseline_state, committed_baseline) &&
            same_state(runner.result().counterfactual_state,
                committed_counterfactual) &&
            runner.result().samples.size() == committed_samples &&
            runner.result().issues.size() == committed_issues.size() &&
            !runner.result().issues.empty() &&
            runner.result().issues.front().message ==
                committed_issues.front().message &&
            runner.result().dropped_issue_count == committed_dropped &&
            runner.result().summaries.size() == committed_summaries.size() &&
            same_diagnostics(runner.result().baseline_conservation,
                committed_baseline_conservation) &&
            same_diagnostics(runner.result().counterfactual_conservation,
                committed_counterfactual_conservation) &&
            !runner.cuda_resources_valid(),
        "failed CUDA batch published partial state or retained speculative resources");
    require(!runner.advance_batch() && runner.result().completed_steps == 0U,
        "failed CUDA runner resumed from invalidated device state");
}

void test_unavailable_cuda_device_rejected() {
    auto value = configuration(ExperimentBackend::cuda);
    value.cuda_device = std::numeric_limits<int>::max();
    bool rejected = false;
    try {
        CounterfactualExperimentRunner runner(snapshot(), value);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected,
        "unavailable CUDA experiment device was not rejected explicitly");
}

}  // namespace

int main() {
    try {
        if (!CounterfactualExperimentRunner::cuda_compiled()) {
            std::cerr << "test failure: CUDA counterfactual target lacks CUDA support\n";
            return 1;
        }
        const auto device_decision =
            nbody::test_support::strict_cuda_runtime_test_decision(
                [] { return nbody::cuda_device_count(); });
        if (device_decision ==
            nbody::test_support::CudaRuntimeTestDecision::skip_no_device) {
            std::cout << "SKIP: CUDA counterfactual support compiled, but no CUDA device is available\n";
            return 77;
        }
        test_unavailable_cuda_device_rejected();
        test_cuda_pair_and_numerical_agreement();
        test_cuda_cancellation_and_repeated_lifetime();
        test_recoverable_host_failure_after_enqueue();
        test_cuda_failure_rollback_and_invalidation();
        std::cout << "CUDA counterfactual experiment tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA counterfactual experiment test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
