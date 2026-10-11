#include "nbody/astronomy/solar_system_60.hpp"
#include "nbody/experiments/counterfactual.hpp"
#include "nbody/simulation.hpp"
#include "nbody/solar_system.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using nbody::Body;
using nbody::SimulationState;
using nbody::experiments::CounterfactualExperimentConfiguration;
using nbody::experiments::CounterfactualExperimentRunner;
using nbody::experiments::ExperimentBackend;
using nbody::experiments::ExperimentBranch;
using nbody::experiments::ExperimentBodyIdentity;
using nbody::experiments::ExperimentIssueKind;
using nbody::experiments::ExperimentSnapshot;
using nbody::experiments::ExperimentSnapshotDefinition;
using nbody::experiments::ExperimentStatus;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
    bool threw = false;
    try {
        callable();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

bool same_body(const Body& left, const Body& right) {
    return left.name == right.name && left.mass_kg == right.mass_kg &&
        left.position_m.x == right.position_m.x &&
        left.position_m.y == right.position_m.y &&
        left.position_m.z == right.position_m.z &&
        left.velocity_m_s.x == right.velocity_m_s.x &&
        left.velocity_m_s.y == right.velocity_m_s.y &&
        left.velocity_m_s.z == right.velocity_m_s.z;
}

bool same_state(const SimulationState& left, const SimulationState& right) {
    if (left.time_s != right.time_s || left.bodies.size() != right.bodies.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.bodies.size(); ++index) {
        if (!same_body(left.bodies[index], right.bodies[index])) {
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

bool same_issue(
    const nbody::experiments::ExperimentIssue& left,
    const nbody::experiments::ExperimentIssue& right) {
    return left.kind == right.kind &&
        left.numerical_invalidity == right.numerical_invalidity &&
        left.message == right.message && left.time_s == right.time_s &&
        left.first_body_stable_id == right.first_body_stable_id &&
        left.second_body_stable_id == right.second_body_stable_id &&
        left.branch == right.branch;
}

bool same_samples(
    const std::vector<nbody::experiments::ExperimentSample>& left,
    const std::vector<nbody::experiments::ExperimentSample>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t sample = 0; sample < left.size(); ++sample) {
        if (left[sample].elapsed_time_s != right[sample].elapsed_time_s ||
            left[sample].bodies.size() != right[sample].bodies.size()) {
            return false;
        }
        for (std::size_t body = 0; body < left[sample].bodies.size(); ++body) {
            const auto& a = left[sample].bodies[body];
            const auto& b = right[sample].bodies[body];
            if (!same_vector(a.baseline_position_m, b.baseline_position_m) ||
                !same_vector(a.counterfactual_position_m, b.counterfactual_position_m) ||
                !same_vector(a.baseline_velocity_m_s, b.baseline_velocity_m_s) ||
                !same_vector(a.counterfactual_velocity_m_s, b.counterfactual_velocity_m_s) ||
                a.absolute_separation_m != b.absolute_separation_m ||
                a.parent_relative_separation_m != b.parent_relative_separation_m ||
                a.normalized_deviation != b.normalized_deviation ||
                a.velocity_difference_m_s != b.velocity_difference_m_s) {
                return false;
            }
        }
    }
    return true;
}

bool same_summaries(
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
            a.maximum_absolute_separation_time_s != b.maximum_absolute_separation_time_s ||
            a.final_parent_relative_separation_m != b.final_parent_relative_separation_m ||
            a.maximum_parent_relative_separation_m != b.maximum_parent_relative_separation_m ||
            a.maximum_parent_relative_separation_time_s !=
                b.maximum_parent_relative_separation_time_s ||
            a.final_normalized_deviation != b.final_normalized_deviation ||
            a.maximum_normalized_deviation != b.maximum_normalized_deviation ||
            a.final_velocity_difference_m_s != b.final_velocity_difference_m_s ||
            a.maximum_velocity_difference_m_s != b.maximum_velocity_difference_m_s ||
            a.first_threshold_crossing_time_s != b.first_threshold_crossing_time_s ||
            a.persistent_threshold_time_s != b.persistent_threshold_time_s ||
            a.significant != b.significant ||
            a.parent_relative_available != b.parent_relative_available) {
            return false;
        }
    }
    return true;
}

ExperimentSnapshot controlled_snapshot(bool convergence_established = true) {
    SimulationState state;
    state.time_s = 125.0;
    state.bodies = {
        {"Central", 1.0e20, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"Removed", 1.0e18, {1.0e7, 0.0, 0.0}, {0.0, 400.0, 0.0}},
        {"Tracer", 1.0e10, {2.0e7, 0.0, 0.0}, {0.0, 300.0, 0.0}},
        {"Outer", 1.0e12, {0.0, 4.0e7, 0.0}, {-100.0, 0.0, 0.0}},
    };
    return ExperimentSnapshot({
        "controlled-removal-fixture",
        "analytic deterministic test fixture captured at t=125 s",
        "fixed inertial Cartesian test frame with common origin",
        "central",
        state,
        {
            {"central", std::nullopt, 10.0, false},
            {"removed", std::optional<std::string>{"central"}, 2.0, true},
            {"tracer", std::optional<std::string>{"central"}, 1.0, false},
            {"outer", std::optional<std::string>{"central"}, 1.0, false},
        },
        1.0,
        nbody::experiments::ExperimentIntegrator::velocity_verlet,
        convergence_established,
        100.0,
    });
}

CounterfactualExperimentConfiguration controlled_configuration() {
    CounterfactualExperimentConfiguration configuration;
    configuration.removed_body_stable_id = "removed";
    configuration.limits.observation_horizon_s = 4.0;
    configuration.limits.sample_cadence_s = 1.0;
    configuration.limits.integration_steps_per_batch = 2;
    configuration.limits.maximum_monitored_bodies = 3;
    configuration.limits.maximum_stored_samples = 8;
    configuration.limits.memory_budget_bytes = 1024U * 1024U;
    configuration.limits.workload_budget_interactions = 100000U;
    configuration.significance.absolute_separation_threshold_m = 1.0e-8;
    configuration.significance.persistence_samples = 2;
    configuration.significance.normalization_denominator_floor_m = 1.0;
    configuration.guards.maximum_relative_energy_drift = 1.0;
    configuration.monitored_body_stable_ids = {"tracer", "central"};
    return configuration;
}

ExperimentSnapshot motion_snapshot(
    const nbody::Vec3& first_position,
    const nbody::Vec3& first_velocity,
    const nbody::Vec3& second_position,
    const nbody::Vec3& second_velocity,
    double first_radius = 0.5,
    double second_radius = 0.5,
    double source_time_s = 0.0,
    double timestep_s = 1.0,
    double maximum_horizon_s = 10.0,
    const nbody::Vec3& removed_position = {1.0e6, 2.0e6, 3.0e6}) {
    SimulationState state;
    state.time_s = source_time_s;
    state.bodies = {
        {"First", 1.0e-30, first_position, first_velocity},
        {"Second", 1.0e-30, second_position, second_velocity},
        {"Removed", 1.0e-30, removed_position, {0.0, 0.0, 0.0}},
    };
    return ExperimentSnapshot({
        "linear-motion-fixture",
        "deterministic swept-proximity test fixture",
        "fixed inertial Cartesian test frame with common origin",
        "first",
        std::move(state),
        {
            {"first", std::nullopt, first_radius, false},
            {"second", std::nullopt, second_radius, false},
            {"removed", std::nullopt, 0.1, true},
        },
        timestep_s,
        nbody::experiments::ExperimentIntegrator::velocity_verlet,
        true,
        maximum_horizon_s,
    });
}

CounterfactualExperimentConfiguration motion_configuration(double timestep_s = 1.0) {
    CounterfactualExperimentConfiguration configuration;
    configuration.removed_body_stable_id = "removed";
    configuration.limits.observation_horizon_s = timestep_s;
    configuration.limits.sample_cadence_s = timestep_s;
    configuration.limits.integration_steps_per_batch = 1;
    configuration.limits.maximum_monitored_bodies = 2;
    configuration.limits.maximum_stored_samples = 2;
    configuration.limits.memory_budget_bytes = 1024U * 1024U;
    configuration.limits.workload_budget_interactions = 1000U;
    configuration.significance.absolute_separation_threshold_m = 1.0e30;
    configuration.significance.persistence_samples = 1;
    configuration.significance.normalization_denominator_floor_m = 1.0;
    configuration.guards.maximum_relative_energy_drift = 1.0e100;
    configuration.monitored_body_stable_ids = {"first", "second"};
    return configuration;
}

ExperimentSnapshot retimed_controlled_snapshot(
    double source_time_s,
    double timestep_s,
    double maximum_horizon_s) {
    auto base = controlled_snapshot();
    auto state = base.source_state();
    state.time_s = source_time_s;
    return ExperimentSnapshot({
        "retimed-controlled-fixture", "test fixture", "common inertial frame",
        "central", std::move(state), base.identities(), timestep_s,
        nbody::experiments::ExperimentIntegrator::velocity_verlet, true,
        maximum_horizon_s,
    });
}

std::size_t summary_index(
    const nbody::experiments::CounterfactualExperimentResult& result,
    const std::string& stable_id) {
    const auto iterator = std::find_if(
        result.summaries.begin(), result.summaries.end(), [&](const auto& value) {
            return value.stable_id == stable_id;
        });
    require(iterator != result.summaries.end(), "missing deviation summary: " + stable_id);
    return static_cast<std::size_t>(iterator - result.summaries.begin());
}

bool has_issue(
    const nbody::experiments::CounterfactualExperimentResult& result,
    ExperimentIssueKind kind) {
    return std::any_of(result.issues.begin(), result.issues.end(), [&](const auto& issue) {
        return issue.kind == kind;
    });
}

std::size_t issue_count(
    const nbody::experiments::CounterfactualExperimentResult& result,
    ExperimentIssueKind kind,
    const std::string& first,
    const std::string& second,
    std::optional<ExperimentBranch> branch = std::nullopt) {
    return static_cast<std::size_t>(std::count_if(
        result.issues.begin(), result.issues.end(), [&](const auto& issue) {
            return issue.kind == kind && issue.first_body_stable_id == first &&
                issue.second_body_stable_id == second &&
                (!branch || issue.branch == branch);
        }));
}

void test_snapshot_and_mapping() {
    auto source = controlled_snapshot();
    const auto captured = source.source_state();
    auto configuration = controlled_configuration();
    CounterfactualExperimentRunner runner(source, configuration);

    require(runner.snapshot().scenario_id() == "controlled-removal-fixture",
        "snapshot lost its scenario identity");
    require(runner.snapshot().source_state().time_s == 125.0,
        "snapshot lost its source simulation timestamp");
    require(runner.snapshot().integrator() ==
            nbody::experiments::ExperimentIntegrator::velocity_verlet,
        "snapshot lost its explicit integrator configuration");
    require(runner.result().baseline_state.bodies.size() == 4U &&
        runner.result().counterfactual_state.bodies.size() == 3U,
        "one-body removal did not compact exactly one branch");
    require(runner.result().survivor_mapping.size() == 3U,
        "survivor mapping has the wrong size");
    for (const auto& mapping : runner.result().survivor_mapping) {
        require(same_body(
            runner.result().baseline_state.bodies[mapping.baseline_index],
            runner.result().counterfactual_state.bodies[mapping.counterfactual_index]),
            "a survivor did not start from an identical state");
    }
    const auto tracer = std::find_if(
        runner.result().survivor_mapping.begin(), runner.result().survivor_mapping.end(),
        [](const auto& value) { return value.stable_id == "tracer"; });
    require(tracer != runner.result().survivor_mapping.end() &&
        tracer->baseline_index == 2U && tracer->counterfactual_index == 1U,
        "stable-ID matching failed after counterfactual vector compaction");
    require(runner.result().samples.size() == 1U,
        "runner did not retain exactly one initial sample");
    for (const auto& body : runner.result().samples.front().bodies) {
        require(body.absolute_separation_m == 0.0 &&
            body.velocity_difference_m_s == 0.0,
            "identical initial branches did not produce a zero-deviation fixture");
    }

    // Both runner and snapshot own copies, rather than aliases to live state.
    require(same_state(captured, runner.snapshot().source_state()),
        "snapshot state changed through an external value");
}

void test_genuine_removal_metrics_and_frame() {
    auto analytic_configuration = controlled_configuration();
    analytic_configuration.limits.observation_horizon_s = 1.0;
    analytic_configuration.limits.integration_steps_per_batch = 1;
    CounterfactualExperimentRunner analytic(
        controlled_snapshot(), analytic_configuration);
    require(analytic.advance_batch(), "analytic one-step fixture did not advance");
    const auto analytic_index = summary_index(analytic.result(), "tracer");
    const double expected_position_separation_m = 0.5 * nbody::gravitational_constant *
        1.0e18 / (1.0e7 * 1.0e7);
    const double measured_position_separation_m =
        analytic.result().summaries[analytic_index].final_absolute_separation_m;
    require(std::abs(measured_position_separation_m - expected_position_separation_m) <=
            expected_position_separation_m * 0.05,
        "one-step removal separation disagrees with the analytic removed-body acceleration term");

    auto configuration = controlled_configuration();
    CounterfactualExperimentRunner runner(controlled_snapshot(), configuration);
    const auto common_origin_before = runner.result().baseline_state.bodies[0].position_m;
    require(runner.advance_batch(), "controlled paired batch did not advance");
    const auto& result = runner.result();
    require(result.baseline_state.time_s == 127.0 &&
        result.counterfactual_state.time_s == 127.0,
        "paired branches did not remain synchronized");

    const auto index = summary_index(result, "tracer");
    const auto& tracer = result.summaries[index];
    require(tracer.final_absolute_separation_m > 0.0,
        "physical removal did not change the surviving tracer trajectory");
    require(tracer.final_velocity_difference_m_s > 0.0 &&
        tracer.maximum_normalized_deviation > 0.0,
        "velocity or normalized removal metrics were not populated");
    require(tracer.fixed_normalization_scale_m == 2.0e7 &&
        std::abs(tracer.final_normalized_deviation -
            tracer.final_absolute_separation_m / 2.0e7) <= 1.0e-20,
        "normalized deviation did not retain its fixed initial reference scale");
    require(tracer.parent_relative_available &&
        tracer.final_parent_relative_separation_m.has_value(),
        "defined parent-relative metrics were not produced");
    require(tracer.first_threshold_crossing_time_s == 1.0 &&
        tracer.persistent_threshold_time_s == 2.0 && tracer.significant,
        "threshold persistence timing is incorrect");
    require(result.samples.size() == 3U,
        "bounded cadence did not retain initial plus two step samples");

    // The original inertial coordinates are advanced directly; neither branch
    // is independently translated to its center of mass.
    require(common_origin_before.x == 0.0 &&
        (result.baseline_state.bodies[0].position_m.x != 0.0 ||
         result.counterfactual_state.bodies[0].position_m.x != 0.0),
        "branches appear to have been recentered after integration");
    require(std::isfinite(result.baseline_conservation.relative_energy_drift) &&
        std::isfinite(result.counterfactual_conservation.relative_energy_drift),
        "per-branch conservation diagnostics are not finite");
    require(!result.escape_assessment_supported,
        "unsupported escape classification was presented as available");

    auto endpoint_configuration = controlled_configuration();
    endpoint_configuration.limits.observation_horizon_s = 3.0;
    endpoint_configuration.limits.sample_cadence_s = 2.0;
    endpoint_configuration.limits.integration_steps_per_batch = 3;
    CounterfactualExperimentRunner endpoint(
        controlled_snapshot(), endpoint_configuration);
    endpoint.run_to_completion();
    require(endpoint.result().samples.size() == 3U &&
        endpoint.result().samples.back().elapsed_time_s == 3.0,
        "exact horizon endpoint was not retained when cadence did not divide the horizon");
}

void test_pause_cancel_and_atomic_rollback() {
    auto configuration = controlled_configuration();
    CounterfactualExperimentRunner paused(controlled_snapshot(), configuration);
    paused.set_paused(true);
    const auto paused_baseline = paused.result().baseline_state;
    require(!paused.advance_batch() && paused.result().status == ExperimentStatus::paused &&
        same_state(paused_baseline, paused.result().baseline_state),
        "paused runner published advancement");
    paused.set_paused(false);
    require(paused.advance_batch(), "resumed runner did not advance a batch");
    const auto boundary_state = paused.result().baseline_state;
    const auto boundary_steps = paused.result().completed_steps;
    paused.request_cancel();
    require(!paused.advance_batch() &&
        paused.result().status == ExperimentStatus::cancelled &&
        paused.result().completed_steps == boundary_steps &&
        same_state(boundary_state, paused.result().baseline_state),
        "cancellation was not honored at a completed pair boundary");

    auto rollback_configuration = controlled_configuration();
    rollback_configuration.guards.maximum_relative_energy_drift = 1.0e-30;
    CounterfactualExperimentRunner rollback(controlled_snapshot(), rollback_configuration);
    const auto baseline_before = rollback.result().baseline_state;
    const auto counter_before = rollback.result().counterfactual_state;
    require(!rollback.advance_batch() && rollback.result().status == ExperimentStatus::failed,
        "forced scientific guard failure did not fail the batch");
    require(same_state(baseline_before, rollback.result().baseline_state) &&
        same_state(counter_before, rollback.result().counterfactual_state) &&
        rollback.result().completed_steps == 0U && rollback.result().samples.size() == 1U,
        "failed paired batch was partially published");
    require(rollback.result().issues.empty() &&
        rollback.result().terminal_failure.has_value() &&
        rollback.result().terminal_failure->kind ==
            ExperimentIssueKind::excessive_conservation_drift &&
        !rollback.result().terminal_failure->message_text().empty(),
        "rollback did not report failure outside committed warning history");
}

void test_resource_and_policy_guards() {
    auto memory = controlled_configuration();
    memory.limits.memory_budget_bytes = 1U;
    require_throws([&] {
        CounterfactualExperimentRunner runner(controlled_snapshot(), memory);
    }, "memory-budget overflow was accepted");

    auto workload = controlled_configuration();
    workload.limits.workload_budget_interactions = 1U;
    require_throws([&] {
        CounterfactualExperimentRunner runner(controlled_snapshot(), workload);
    }, "workload-budget overflow was accepted");

    auto horizon = controlled_configuration();
    horizon.limits.observation_horizon_s = 101.0;
    require_throws([&] {
        CounterfactualExperimentRunner runner(controlled_snapshot(), horizon);
    }, "unsupported experiment horizon was accepted");

    auto sun = controlled_configuration();
    sun.removed_body_stable_id = "central";
    require_throws([&] {
        CounterfactualExperimentRunner runner(controlled_snapshot(), sun);
    }, "Sun removal was accepted");

    auto uncurated = controlled_configuration();
    uncurated.removed_body_stable_id = "tracer";
    require_throws([&] {
        CounterfactualExperimentRunner runner(controlled_snapshot(), uncurated);
    }, "uncurated removal was accepted");

    auto provisional = controlled_configuration();
    CounterfactualExperimentRunner provisional_runner(
        controlled_snapshot(false), provisional);
    require(has_issue(provisional_runner.result(),
        ExperimentIssueKind::insufficient_timestep_convergence),
        "provisional timestep was not reported");

    require_throws([] {
        auto definition = ExperimentSnapshotDefinition{
            "invalid", "test", "frame", "a",
            {0.0, {
                {"a", 1.0, {0.0, 0.0, 0.0}, {}},
                {"b", 1.0, {0.0, 0.0, 0.0}, {}},
            }},
            {{"a", std::nullopt, 1.0, false}, {"b", std::nullopt, 1.0, true}},
            1.0, nbody::experiments::ExperimentIntegrator::velocity_verlet, true, 2.0,
        };
        ExperimentSnapshot snapshot(std::move(definition));
    }, "coincident source state was accepted");

    require_throws([] {
        auto snapshot = controlled_snapshot();
        auto definition = ExperimentSnapshotDefinition{
            "invalid", "test", "frame", "central", snapshot.source_state(),
            {
                {"central", std::nullopt, 1.0, false},
                {"duplicate", std::nullopt, 1.0, true},
                {"duplicate", std::nullopt, 1.0, false},
                {"outer", std::nullopt, 1.0, false},
            },
            1.0, nbody::experiments::ExperimentIntegrator::velocity_verlet, true, 2.0,
        };
        ExperimentSnapshot invalid(std::move(definition));
    }, "duplicate stable-ID mapping was accepted");

    require_throws([] {
        auto snapshot = controlled_snapshot();
        auto definition = ExperimentSnapshotDefinition{
            "invalid", "test", "frame", "central", snapshot.source_state(),
            {
                {"central", std::nullopt, 1.0, false},
                {"removed", std::nullopt, -1.0, true},
                {"tracer", std::nullopt, 1.0, false},
                {"outer", std::nullopt, 1.0, false},
            },
            1.0, nbody::experiments::ExperimentIntegrator::velocity_verlet, true, 2.0,
        };
        ExperimentSnapshot invalid(std::move(definition));
    }, "invalid physical radius was accepted");

    require_throws([] {
        auto snapshot = controlled_snapshot();
        auto state = snapshot.source_state();
        state.bodies[2].velocity_m_s.x = std::numeric_limits<double>::quiet_NaN();
        ExperimentSnapshot invalid({
            "invalid", "test", "frame", "central", state, snapshot.identities(),
            1.0, nbody::experiments::ExperimentIntegrator::velocity_verlet, true, 2.0,
        });
    }, "non-finite source state was accepted");
}

void test_close_approach_and_parent_removal() {
    auto base = controlled_snapshot();
    auto definition = ExperimentSnapshotDefinition{
        "close-approach", "test fixture", "common frame", "central",
        base.source_state(), base.identities(), 1.0,
        nbody::experiments::ExperimentIntegrator::velocity_verlet, true, 10.0,
    };
    definition.identities[0].physical_radius_m = 4.0e6;
    definition.identities[1].physical_radius_m = 1.0e6;
    auto config = controlled_configuration();
    config.limits.observation_horizon_s = 1.0;
    config.limits.integration_steps_per_batch = 1;
    CounterfactualExperimentRunner close(ExperimentSnapshot(std::move(definition)), config);
    close.run_to_completion();
    require(close.result().status == ExperimentStatus::completed &&
        has_issue(close.result(), ExperimentIssueKind::dangerous_close_approach),
        "dangerous close approach was not reported separately from significance");

    auto overlap_base = controlled_snapshot();
    auto overlap_definition = ExperimentSnapshotDefinition{
        "radius-overlap", "test fixture", "common frame", "central",
        overlap_base.source_state(), overlap_base.identities(), 1.0,
        nbody::experiments::ExperimentIntegrator::velocity_verlet, true, 10.0,
    };
    overlap_definition.identities[0].physical_radius_m = 6.0e6;
    overlap_definition.identities[1].physical_radius_m = 6.0e6;
    CounterfactualExperimentRunner overlap(
        ExperimentSnapshot(std::move(overlap_definition)), config);
    require(has_issue(overlap.result(), ExperimentIssueKind::physical_radius_overlap),
        "sourced-radius overlap was not reported");

    const auto catalog = nbody::astronomy::make_solar_system_60_catalog();
    SimulationState state;
    for (const auto& record : catalog.records()) {
        state.bodies.push_back({
            record.display_name(), record.mass_kg(), record.position_m(), record.velocity_m_s(),
        });
    }
    auto catalog_snapshot = nbody::experiments::capture_astronomical_catalog_snapshot(state);
    CounterfactualExperimentConfiguration catalog_config;
    catalog_config.removed_body_stable_id = "sol:earth";
    catalog_config.limits.observation_horizon_s = 60.0;
    catalog_config.limits.sample_cadence_s = 60.0;
    catalog_config.limits.integration_steps_per_batch = 1;
    catalog_config.limits.maximum_monitored_bodies = 1;
    catalog_config.limits.maximum_stored_samples = 2;
    catalog_config.limits.memory_budget_bytes = 1024U * 1024U;
    catalog_config.limits.workload_budget_interactions = 20000;
    catalog_config.significance.absolute_separation_threshold_m = 1.0;
    catalog_config.guards.maximum_relative_energy_drift = 1.0;
    catalog_config.monitored_body_stable_ids = {"sol:moon"};
    CounterfactualExperimentRunner parent_removed(
        std::move(catalog_snapshot), catalog_config);
    parent_removed.run_to_completion();
    require(parent_removed.result().status == ExperimentStatus::completed,
        "catalog parent-removal fixture did not complete");
    require(!parent_removed.result().summaries[0].parent_relative_available &&
        !parent_removed.result().summaries[0].final_parent_relative_separation_m &&
        !parent_removed.result().samples.back().bodies[0].parent_relative_separation_m,
        "parent-relative metrics were invented after parent removal");
}

void test_swept_proximity_guards() {
    auto exchange = motion_configuration();
    CounterfactualExperimentRunner exchange_runner(
        motion_snapshot({-10.0, 0.0, 0.0}, {20.0, 0.0, 0.0},
            {10.0, 0.0, 0.0}, {-20.0, 0.0, 0.0}),
        exchange);
    exchange_runner.run_to_completion();
    require(exchange_runner.result().status == ExperimentStatus::completed &&
        has_issue(exchange_runner.result(), ExperimentIssueKind::physical_radius_overlap),
        "swept guard missed bodies that exchanged positions between safe endpoints");
    require(issue_count(exchange_runner.result(),
            ExperimentIssueKind::physical_radius_overlap, "first", "second",
            ExperimentBranch::baseline) == 1U &&
        issue_count(exchange_runner.result(),
            ExperimentIssueKind::physical_radius_overlap, "first", "second",
            ExperimentBranch::counterfactual) == 1U,
        "branch-aware warning deduplication lost a paired swept event");

    CounterfactualExperimentRunner zero_relative(
        motion_snapshot({0.0, 0.0, 0.0}, {3.0, 0.0, 0.0},
            {10.0, 0.0, 0.0}, {3.0, 0.0, 0.0}),
        motion_configuration());
    zero_relative.run_to_completion();
    require(!has_issue(zero_relative.result(),
            ExperimentIssueKind::dangerous_close_approach) &&
        !has_issue(zero_relative.result(), ExperimentIssueKind::physical_radius_overlap),
        "zero-relative-displacement swept test produced a false proximity warning");

    CounterfactualExperimentRunner parallel(
        motion_snapshot({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {0.0, 10.0, 0.0}, {2.0, 0.0, 0.0}),
        motion_configuration());
    parallel.run_to_completion();
    require(!has_issue(parallel.result(), ExperimentIssueKind::physical_radius_overlap),
        "parallel swept motion produced a false overlap warning");

    CounterfactualExperimentRunner tangent(
        motion_snapshot({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
            {-10.0, 1.0, 0.0}, {20.0, 0.0, 0.0}),
        motion_configuration());
    tangent.run_to_completion();
    require(has_issue(tangent.result(), ExperimentIssueKind::physical_radius_overlap),
        "tangential swept-radius contact was not reported");

    CounterfactualExperimentRunner receding(
        motion_snapshot({0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
            {4.0, 0.0, 0.0}, {1.0, 0.0, 0.0}),
        motion_configuration());
    receding.run_to_completion();
    require(!has_issue(receding.result(),
            ExperimentIssueKind::dangerous_close_approach) &&
        !has_issue(receding.result(), ExperimentIssueKind::physical_radius_overlap),
        "receding swept motion produced a false proximity warning");

    CounterfactualExperimentRunner initial_overlap(
        motion_snapshot({0.0, 0.0, 0.0}, {}, {0.5, 0.0, 0.0}, {}),
        motion_configuration());
    require(issue_count(initial_overlap.result(),
            ExperimentIssueKind::physical_radius_overlap, "first", "second",
            ExperimentBranch::baseline) == 1U &&
        issue_count(initial_overlap.result(),
            ExperimentIssueKind::physical_radius_overlap, "first", "second",
            ExperimentBranch::counterfactual) == 1U,
        "initial overlap was not retained independently for both branches");

    CounterfactualExperimentRunner baseline_only(
        motion_snapshot({0.0, 0.0, 0.0}, {}, {10.0, 0.0, 0.0}, {},
            0.5, 0.5, 0.0, 1.0, 10.0, {0.5, 0.0, 0.0}),
        motion_configuration());
    require(issue_count(baseline_only.result(),
            ExperimentIssueKind::physical_radius_overlap, "first", "removed",
            ExperimentBranch::baseline) == 1U &&
        issue_count(baseline_only.result(),
            ExperimentIssueKind::physical_radius_overlap, "first", "removed",
            ExperimentBranch::counterfactual) == 0U,
        "branch-specific proximity outcome was attributed to both branches");

    require_throws([] {
        motion_snapshot({-std::numeric_limits<double>::max(), 0.0, 0.0}, {},
            {std::numeric_limits<double>::max(), 0.0, 0.0}, {});
    }, "non-finite relative-distance input passed snapshot admission");
}

void test_range_stable_swept_geometry() {
    const auto swept = [](const nbody::Vec3& left_before,
                           const nbody::Vec3& right_before,
                           const nbody::Vec3& left_after,
                           const nbody::Vec3& right_after) {
        return nbody::experiments::linear_swept_minimum_distance(
            left_before, right_before, left_after, right_after);
    };

    require(swept({}, {-1.0e154, 0.0, 0.0}, {}, {1.0e154, 0.0, 0.0}) == 0.0,
        "binary64-scale endpoint-safe crossing did not reach zero separation");
    require(swept({}, {-1.6e308, 0.0, 0.0}, {}, {1.6e308, 0.0, 0.0}) == 0.0,
        "representable near-limit endpoints were rejected because their delta overflowed");

    const double parallel = swept(
        {-5.0e153, -2.0e153, 0.0}, {5.0e153, 3.0e153, 0.0},
        {-4.0e153, -1.0e153, 0.0}, {6.0e153, 4.0e153, 0.0});
    require(std::isfinite(parallel) &&
        std::abs(parallel - std::hypot(1.0e154, 5.0e153)) <= parallel * 1.0e-15,
        "large finite parallel motion lost its constant relative distance");

    require(swept({}, {1.0e154, 0.0, 0.0}, {}, {2.0e154, 0.0, 0.0}) == 1.0e154,
        "large receding trajectory was not clamped to its initial endpoint");
    require(swept({}, {1.0e154, 0.0, 0.0}, {}, {5.0e153, 0.0, 0.0}) == 5.0e153,
        "approaching non-crossing trajectory was not clamped to its final endpoint");

    require(swept({1.0, 2.0, 3.0}, {4.0, 6.0, 3.0},
            {11.0, 12.0, 13.0}, {14.0, 16.0, 13.0}) == 5.0,
        "zero-relative-displacement degenerate path changed distance");

    const double grazing = swept(
        {}, {-1.0e154, 3.0, 0.0}, {}, {1.0e154, 3.0, 0.0});
    require(std::abs(grazing - 3.0) <= 1.0e-12,
        "large finite tangential trajectory lost its grazing distance");

    const double near_overflow = swept(
        {-8.0e307, 0.0, 0.0}, {8.0e307, 0.0, 0.0},
        {-8.0e307, 0.0, 0.0}, {8.0e307, 0.0, 0.0});
    require(std::isfinite(near_overflow) &&
        std::abs(near_overflow / 1.6e308 - 1.0) <= 1.0e-15,
        "near-overflow endpoint subtraction was not handled representably");

    require(std::abs(swept({}, {-10.0, 1.0, 0.0}, {}, {10.0, 1.0, 0.0}) - 1.0) <=
            1.0e-12,
        "ordinary-scale swept regression changed");

    require_throws([&] {
        (void)swept(
            {-std::numeric_limits<double>::max(), 0.0, 0.0},
            {std::numeric_limits<double>::max(), 0.0, 0.0},
            {}, {1.0, 0.0, 0.0});
    }, "genuinely non-representable relative geometry was accepted");
}

void test_checked_step_workload_and_memory_arithmetic() {
    auto ordinary = controlled_configuration();
    CounterfactualExperimentRunner ordinary_runner(controlled_snapshot(), ordinary);
    require(ordinary_runner.result().status == ExperimentStatus::ready,
        "ordinary integral step count was rejected");

    for (const double invalid : {
             0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity(), 1.5, 2.0 - 1.0e-6,
             2.0 + 1.0e-6}) {
        auto configuration = controlled_configuration();
        configuration.limits.observation_horizon_s = invalid;
        require_throws([&] {
            CounterfactualExperimentRunner runner(controlled_snapshot(), configuration);
        }, "invalid or non-integral experiment duration was accepted");
    }
    for (const double invalid_timestep : {
             0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity()}) {
        require_throws([&] {
            retimed_controlled_snapshot(0.0, invalid_timestep, 2.0);
        }, "invalid fixed timestep was accepted");
    }

    const double maximum_steps = static_cast<double>(
        nbody::experiments::maximum_experiment_step_count);
    auto boundary = motion_configuration();
    boundary.limits.observation_horizon_s = maximum_steps;
    boundary.limits.sample_cadence_s = maximum_steps;
    boundary.limits.workload_budget_interactions =
        std::numeric_limits<std::uint64_t>::max();
    CounterfactualExperimentRunner boundary_runner(
        motion_snapshot({0.0, 0.0, 0.0}, {}, {10.0, 0.0, 0.0}, {},
            0.5, 0.5, 0.0, 1.0, maximum_steps),
        boundary);
    require(boundary_runner.result().status == ExperimentStatus::ready,
        "maximum operational integral step count was rejected");

    auto above = boundary;
    above.limits.observation_horizon_s = maximum_steps + 1.0;
    above.limits.sample_cadence_s = maximum_steps + 1.0;
    require_throws([&] {
        CounterfactualExperimentRunner runner(
            motion_snapshot({0.0, 0.0, 0.0}, {}, {10.0, 0.0, 0.0}, {},
                0.5, 0.5, 0.0, 1.0, maximum_steps + 1.0),
            above);
    }, "step count above the operational limit was accepted");

    auto enormous = boundary;
    enormous.limits.observation_horizon_s = static_cast<double>(
        nbody::experiments::maximum_exact_integer_step_count);
    enormous.limits.sample_cadence_s = enormous.limits.observation_horizon_s;
    require_throws([&] {
        CounterfactualExperimentRunner runner(
            motion_snapshot({0.0, 0.0, 0.0}, {}, {10.0, 0.0, 0.0}, {},
                0.5, 0.5, 0.0, 1.0,
                enormous.limits.observation_horizon_s), enormous);
    }, "representable but policy-prohibitive step count was accepted");

    auto near_size_max = boundary;
    near_size_max.limits.observation_horizon_s = static_cast<double>(
        std::numeric_limits<std::size_t>::max());
    near_size_max.limits.sample_cadence_s = near_size_max.limits.observation_horizon_s;
    require_throws([&] {
        CounterfactualExperimentRunner runner(
            motion_snapshot({0.0, 0.0, 0.0}, {}, {10.0, 0.0, 0.0}, {},
                0.5, 0.5, 0.0, 1.0,
                near_size_max.limits.observation_horizon_s), near_size_max);
    }, "near-size_t-max step request was accepted");

    auto huge_finite = boundary;
    huge_finite.limits.observation_horizon_s =
        std::numeric_limits<double>::max();
    huge_finite.limits.sample_cadence_s = huge_finite.limits.observation_horizon_s;
    require_throws([&] {
        CounterfactualExperimentRunner runner(
            motion_snapshot({0.0, 0.0, 0.0}, {}, {10.0, 0.0, 0.0}, {},
                0.5, 0.5, 0.0, 1.0,
                huge_finite.limits.observation_horizon_s), huge_finite);
    }, "maximum finite step request was accepted");

    require_throws([] {
        (void)nbody::experiments::estimate_counterfactual_direct_force_workload(
            std::numeric_limits<std::size_t>::max(),
            std::numeric_limits<std::size_t>::max(),
            std::numeric_limits<std::size_t>::max());
    }, "workload arithmetic overflow was accepted");

    auto memory_overflow = controlled_configuration();
    memory_overflow.limits.maximum_issue_records =
        std::numeric_limits<std::size_t>::max();
    memory_overflow.limits.memory_budget_bytes =
        std::numeric_limits<std::size_t>::max();
    require_throws([&] {
        CounterfactualExperimentRunner runner(controlled_snapshot(), memory_overflow);
    }, "memory-accounting overflow was accepted");
}

void test_time_progress_and_transactional_failure() {
    require_throws([] {
        retimed_controlled_snapshot(1.0e20, 1.0, 2.0);
    }, "snapshot whose absolute timestamp cannot advance was accepted");

    auto fractional = controlled_configuration();
    fractional.limits.observation_horizon_s = 0.3;
    fractional.limits.sample_cadence_s = 0.1;
    fractional.limits.integration_steps_per_batch = 1;
    CounterfactualExperimentRunner fractional_runner(
        retimed_controlled_snapshot(125.0, 0.1, 1.0), fractional);
    fractional_runner.run_to_completion();
    require(fractional_runner.result().status == ExperimentStatus::completed &&
        fractional_runner.result().completed_steps == 3U &&
        fractional_runner.result().samples.back().elapsed_time_s == 3.0 * 0.1 &&
        fractional_runner.result().baseline_state.time_s ==
            fractional_runner.result().counterfactual_state.time_s,
        "non-integral timestep did not retain step-derived elapsed time and paired synchronization");

    auto stagnation = controlled_configuration();
    stagnation.limits.observation_horizon_s = 2.0;
    stagnation.limits.sample_cadence_s = 1.0;
    stagnation.limits.integration_steps_per_batch = 1;
    CounterfactualExperimentRunner stagnation_runner(
        retimed_controlled_snapshot(9007199254740991.0, 1.0, 2.0), stagnation);
    require(stagnation_runner.advance_batch(),
        "first exactly advancing large-timestamp batch failed");
    const auto committed_baseline = stagnation_runner.result().baseline_state;
    const auto committed_counterfactual = stagnation_runner.result().counterfactual_state;
    const auto committed_samples = stagnation_runner.result().samples;
    require(!stagnation_runner.advance_batch() &&
        stagnation_runner.result().status == ExperimentStatus::failed &&
        stagnation_runner.result().completed_steps == 1U &&
        same_state(committed_baseline, stagnation_runner.result().baseline_state) &&
        same_state(committed_counterfactual,
            stagnation_runner.result().counterfactual_state) &&
        stagnation_runner.result().samples.size() == committed_samples.size() &&
        stagnation_runner.result().terminal_failure.has_value() &&
        stagnation_runner.result().terminal_failure->kind ==
            ExperimentIssueKind::invalid_simulated_time,
        "post-commit time stagnation did not roll back the failing batch");

    SimulationState collision_state;
    collision_state.bodies = {
        {"First", 1.0e10, {-1.0, 0.0, 0.0}, {}},
        {"Second", 1.0e10, {1.0, 0.0, 0.0}, {}},
        {"Removed", 1.0e20, {-1.0, 10.0, 0.0}, {}},
    };
    SimulationState counter_start;
    counter_start.bodies = {collision_state.bodies[0], collision_state.bodies[1]};
    const auto accelerations = nbody::compute_accelerations(counter_start);
    for (std::size_t index = 0; index < 2U; ++index) {
        collision_state.bodies[index].velocity_m_s =
            collision_state.bodies[index].position_m * -1.0 -
            accelerations[index] * 0.5;
        counter_start.bodies[index].velocity_m_s =
            collision_state.bodies[index].velocity_m_s;
    }
    auto baseline_probe = collision_state;
    nbody::velocity_verlet_step(baseline_probe, 1.0);
    require(baseline_probe.time_s == 1.0,
        "second-branch failure fixture unexpectedly failed in the baseline branch");
    require_throws([&] {
        nbody::velocity_verlet_step(counter_start, 1.0);
    }, "second-branch failure fixture did not fail in the counterfactual branch");
    ExperimentSnapshot collision_snapshot({
        "second-branch-failure", "test fixture", "common inertial frame", "first",
        collision_state,
        {
            {"first", std::nullopt, 0.01, false},
            {"second", std::nullopt, 0.01, false},
            {"removed", std::nullopt, 0.01, true},
        },
        1.0, nbody::experiments::ExperimentIntegrator::velocity_verlet, true, 1.0,
    });
    auto collision_configuration = motion_configuration();
    CounterfactualExperimentRunner second_branch_failure(
        std::move(collision_snapshot), collision_configuration);
    const auto initial_baseline = second_branch_failure.result().baseline_state;
    const auto initial_counter = second_branch_failure.result().counterfactual_state;
    require(!second_branch_failure.advance_batch() &&
        second_branch_failure.result().status == ExperimentStatus::failed &&
        second_branch_failure.result().completed_steps == 0U &&
        same_state(initial_baseline, second_branch_failure.result().baseline_state) &&
        same_state(initial_counter,
            second_branch_failure.result().counterfactual_state) &&
        second_branch_failure.result().samples.size() == 1U &&
        second_branch_failure.result().terminal_failure.has_value() &&
        second_branch_failure.result().terminal_failure->kind ==
            ExperimentIssueKind::integration_failure,
        "counterfactual-branch integration failure partially published the baseline branch");
}

void test_failed_batch_preserves_committed_diagnostics() {
    auto base = controlled_snapshot();
    auto state = base.source_state();
    state.time_s = 9007199254740991.0;
    auto identities = base.identities();
    identities[0].physical_radius_m = 6.0e6;
    identities[1].physical_radius_m = 6.0e6;
    ExperimentSnapshot snapshot({
        "committed-warning-rollback", "test fixture", "common inertial frame",
        "central", std::move(state), std::move(identities), 1.0,
        nbody::experiments::ExperimentIntegrator::velocity_verlet, true, 2.0,
    });
    auto configuration = controlled_configuration();
    configuration.limits.observation_horizon_s = 2.0;
    configuration.limits.sample_cadence_s = 1.0;
    configuration.limits.integration_steps_per_batch = 1;
    configuration.limits.maximum_issue_records = 1U;
    configuration.limits.maximum_issue_message_bytes = 1024U;
    CounterfactualExperimentRunner runner(std::move(snapshot), configuration);
    require(runner.result().issues.size() == 1U,
        "rollback fixture did not fill its committed warning buffer");
    require(runner.advance_batch() && runner.result().completed_steps == 1U,
        "rollback fixture did not publish its first valid batch");

    const auto committed = runner.result();
    require(!runner.advance_batch() && runner.result().status == ExperimentStatus::failed,
        "large-timestamp second batch did not fail transactionally");
    const auto& failed = runner.result();
    require(same_state(committed.baseline_state, failed.baseline_state) &&
        same_state(committed.counterfactual_state, failed.counterfactual_state) &&
        committed.completed_steps == failed.completed_steps &&
        committed.dropped_issue_count == failed.dropped_issue_count &&
        committed.issues.size() == failed.issues.size() &&
        same_issue(committed.issues.front(), failed.issues.front()) &&
        same_samples(committed.samples, failed.samples) &&
        same_summaries(committed.summaries, failed.summaries) &&
        same_diagnostics(committed.baseline_conservation,
            failed.baseline_conservation) &&
        same_diagnostics(committed.counterfactual_conservation,
            failed.counterfactual_conservation),
        "failed batch mutated a previously committed experiment result");
    require(failed.terminal_failure.has_value() &&
        failed.terminal_failure->kind == ExperimentIssueKind::invalid_simulated_time &&
        !failed.terminal_failure->message_text().empty(),
        "full warning buffer suppressed independent terminal-failure reporting");

    const auto terminal_kind = failed.terminal_failure->kind;
    const auto terminal_time = failed.terminal_failure->time_s;
    const std::string terminal_message(failed.terminal_failure->message_text());
    const auto repeated_issue_count = runner.result().issues.size();
    const auto repeated_dropped_count = runner.result().dropped_issue_count;
    require(runner.result().terminal_failure->kind == terminal_kind &&
        runner.result().terminal_failure->time_s == terminal_time &&
        runner.result().terminal_failure->message_text() ==
            std::string_view(terminal_message) &&
        same_state(committed.baseline_state, runner.result().baseline_state) &&
        same_state(committed.counterfactual_state,
            runner.result().counterfactual_state) &&
        committed.completed_steps == runner.result().completed_steps &&
        same_samples(committed.samples, runner.result().samples) &&
        same_summaries(committed.summaries, runner.result().summaries) &&
        same_diagnostics(committed.baseline_conservation,
            runner.result().baseline_conservation) &&
        same_diagnostics(committed.counterfactual_conservation,
            runner.result().counterfactual_conservation) &&
        runner.result().issues.size() == repeated_issue_count &&
        runner.result().dropped_issue_count == repeated_dropped_count,
        "repeated failed-result access mutated terminal or committed diagnostics");

    auto empty_configuration = controlled_configuration();
    empty_configuration.guards.maximum_relative_energy_drift = 1.0e-30;
    CounterfactualExperimentRunner empty_failure(
        controlled_snapshot(), empty_configuration);
    require(empty_failure.result().issues.empty() &&
        !empty_failure.advance_batch() &&
        empty_failure.result().issues.empty() &&
        empty_failure.result().dropped_issue_count == 0U &&
        empty_failure.result().terminal_failure.has_value() &&
        empty_failure.result().terminal_failure->kind ==
            ExperimentIssueKind::excessive_conservation_drift,
        "empty warning buffer did not retain independent failure metadata");
}

void test_bounded_issues_and_ownership_contract() {
    static_assert(!std::is_copy_constructible_v<CounterfactualExperimentRunner>);
    static_assert(!std::is_move_constructible_v<CounterfactualExperimentRunner>);

    auto limited = motion_configuration();
    limited.limits.maximum_issue_records = 1U;
    limited.limits.maximum_issue_message_bytes = 1024U;
    CounterfactualExperimentRunner limited_runner(
        motion_snapshot({0.0, 0.0, 0.0}, {}, {0.5, 0.0, 0.0}, {}), limited);
    require(limited_runner.result().issues.size() == 1U &&
        limited_runner.result().dropped_issue_count >= 1U,
        "bounded issue storage did not report a dropped branch warning");

    auto message_limited = motion_configuration();
    message_limited.limits.maximum_issue_message_bytes = 1U;
    CounterfactualExperimentRunner message_limited_runner(
        motion_snapshot({0.0, 0.0, 0.0}, {}, {0.5, 0.0, 0.0}, {}),
        message_limited);
    require(message_limited_runner.result().issues.empty() &&
        message_limited_runner.result().dropped_issue_count >= 2U,
        "issue-message byte cap did not bound retained diagnostics");
}

void test_legacy_and_catalog_compatibility() {
    const auto legacy_before = nbody::make_approximate_solar_system();
    const auto legacy_copy = legacy_before.initial_state;
    auto legacy_snapshot = nbody::experiments::capture_legacy_solar_system_snapshot(
        legacy_before.initial_state);
    auto independently_mutable_live_state = legacy_before.initial_state;
    auto isolated_snapshot = nbody::experiments::capture_legacy_solar_system_snapshot(
        independently_mutable_live_state);
    const double isolated_x = isolated_snapshot.source_state().bodies[0].position_m.x;
    independently_mutable_live_state.bodies[0].position_m.x += 12345.0;
    require(legacy_snapshot.identities().size() == 9U &&
        same_state(legacy_copy, legacy_before.initial_state) &&
        isolated_snapshot.source_state().bodies[0].position_m.x == isolated_x,
        "legacy snapshot capture changed the baseline or retained a live-state alias");
    CounterfactualExperimentConfiguration legacy_config;
    legacy_config.removed_body_stable_id = "jupiter";
    legacy_config.limits.observation_horizon_s = nbody::solar_system_recommended_dt_s;
    legacy_config.limits.sample_cadence_s = nbody::solar_system_recommended_dt_s;
    legacy_config.limits.integration_steps_per_batch = 1;
    legacy_config.limits.maximum_monitored_bodies = 8;
    legacy_config.limits.maximum_stored_samples = 2;
    legacy_config.limits.memory_budget_bytes = 1024U * 1024U;
    legacy_config.limits.workload_budget_interactions = 1000;
    legacy_config.significance.absolute_separation_threshold_m = 1.0;
    legacy_config.guards.maximum_relative_energy_drift = 1.0;
    CounterfactualExperimentRunner legacy_runner(std::move(legacy_snapshot), legacy_config);
    legacy_runner.run_to_completion();
    require(legacy_runner.result().status == ExperimentStatus::completed &&
        legacy_runner.result().baseline_state.bodies.size() == 9U &&
        legacy_runner.result().counterfactual_state.bodies.size() == 8U,
        "legacy nine-body experiment compatibility failed");

    const auto catalog = nbody::astronomy::make_solar_system_60_catalog();
    SimulationState catalog_state;
    for (const auto& record : catalog.records()) {
        catalog_state.bodies.push_back({
            record.display_name(), record.mass_kg(), record.position_m(), record.velocity_m_s(),
        });
    }
    const auto catalog_copy = catalog_state;
    auto catalog_snapshot = nbody::experiments::capture_astronomical_catalog_snapshot(
        catalog_state);
    require(catalog_snapshot.identities().size() == 60U &&
        catalog_snapshot.fixed_timestep_s() == 60.0 &&
        !catalog_snapshot.timestep_convergence_established() &&
        same_state(catalog_copy, catalog_state),
        "60-body catalog snapshot compatibility or provisional policy failed");
    CounterfactualExperimentConfiguration catalog_config;
    catalog_config.removed_body_stable_id = "sol:jupiter";
    catalog_config.limits.observation_horizon_s = 60.0;
    catalog_config.limits.sample_cadence_s = 60.0;
    catalog_config.limits.integration_steps_per_batch = 1;
    catalog_config.limits.maximum_monitored_bodies = 2;
    catalog_config.limits.maximum_stored_samples = 2;
    catalog_config.limits.memory_budget_bytes = 1024U * 1024U;
    catalog_config.limits.workload_budget_interactions = 20000;
    catalog_config.significance.absolute_separation_threshold_m = 1.0;
    catalog_config.guards.maximum_relative_energy_drift = 1.0;
    catalog_config.monitored_body_stable_ids = {"sol:earth", "sol:moon"};
    CounterfactualExperimentRunner catalog_runner(std::move(catalog_snapshot), catalog_config);
    catalog_runner.run_to_completion();
    require(catalog_runner.result().status == ExperimentStatus::completed &&
        catalog_runner.result().baseline_state.bodies.size() == 60U &&
        catalog_runner.result().counterfactual_state.bodies.size() == 59U &&
        has_issue(catalog_runner.result(),
            ExperimentIssueKind::insufficient_timestep_convergence),
        "full catalog experiment compatibility failed");
}

void test_sequential_repeatability() {
    const auto configuration = controlled_configuration();
    CounterfactualExperimentRunner first(controlled_snapshot(), configuration);
    CounterfactualExperimentRunner second(controlled_snapshot(), configuration);
    first.run_to_completion();
    second.run_to_completion();
    require(first.result().status == ExperimentStatus::completed &&
        second.result().status == ExperimentStatus::completed &&
        same_state(first.result().baseline_state, second.result().baseline_state) &&
        same_state(first.result().counterfactual_state,
            second.result().counterfactual_state),
        "supported Sequential experiment was not repeatable");
    require(first.result().samples.size() == second.result().samples.size(),
        "repeatable experiments retained different sample counts");
    for (std::size_t sample = 0; sample < first.result().samples.size(); ++sample) {
        require(first.result().samples[sample].elapsed_time_s ==
                second.result().samples[sample].elapsed_time_s,
            "repeatable experiments produced different sample times");
        for (std::size_t body = 0;
             body < first.result().samples[sample].bodies.size(); ++body) {
            require(first.result().samples[sample].bodies[body].absolute_separation_m ==
                    second.result().samples[sample].bodies[body].absolute_separation_m,
                "repeatable experiments produced different trajectory metrics");
        }
    }
}

#ifdef NBODY_TEST_HAS_OPENMP
void test_openmp_agreement() {
    auto sequential_configuration = controlled_configuration();
    sequential_configuration.backend = ExperimentBackend::sequential;
    CounterfactualExperimentRunner sequential(
        controlled_snapshot(), sequential_configuration);
    sequential.run_to_completion();

    auto openmp_configuration = controlled_configuration();
    openmp_configuration.backend = ExperimentBackend::openmp;
    openmp_configuration.openmp_threads = 2;
    CounterfactualExperimentRunner openmp(controlled_snapshot(), openmp_configuration);
    openmp.run_to_completion();
    require(CounterfactualExperimentRunner::openmp_supported(),
        "OpenMP build did not advertise experiment support");
    require(openmp.result().status == ExperimentStatus::completed &&
        sequential.result().status == ExperimentStatus::completed,
        "Sequential/OpenMP comparison did not complete");
    for (std::size_t index = 0; index < sequential.result().baseline_state.bodies.size(); ++index) {
        const auto delta = nbody::norm(
            sequential.result().baseline_state.bodies[index].position_m -
            openmp.result().baseline_state.bodies[index].position_m);
        require(delta <= 1.0e-6,
            "OpenMP baseline position disagrees with Sequential");
    }
    for (std::size_t index = 0;
         index < sequential.result().counterfactual_state.bodies.size(); ++index) {
        const auto delta = nbody::norm(
            sequential.result().counterfactual_state.bodies[index].position_m -
            openmp.result().counterfactual_state.bodies[index].position_m);
        require(delta <= 1.0e-6,
            "OpenMP counterfactual position disagrees with Sequential");
    }
}
#else
void test_openmp_gating() {
    require(!CounterfactualExperimentRunner::openmp_supported(),
        "CPU-only build incorrectly advertises OpenMP experiment support");
    auto configuration = controlled_configuration();
    configuration.backend = ExperimentBackend::openmp;
    require_throws([&] {
        CounterfactualExperimentRunner runner(controlled_snapshot(), configuration);
    }, "CPU-only experiment silently fell back from OpenMP");
}
#endif

}  // namespace

int main() {
    try {
        test_snapshot_and_mapping();
        test_genuine_removal_metrics_and_frame();
        test_pause_cancel_and_atomic_rollback();
        test_resource_and_policy_guards();
        test_close_approach_and_parent_removal();
        test_swept_proximity_guards();
        test_range_stable_swept_geometry();
        test_checked_step_workload_and_memory_arithmetic();
        test_time_progress_and_transactional_failure();
        test_failed_batch_preserves_committed_diagnostics();
        test_bounded_issues_and_ownership_contract();
        test_legacy_and_catalog_compatibility();
        test_sequential_repeatability();
#ifdef NBODY_TEST_HAS_OPENMP
        test_openmp_agreement();
#else
        test_openmp_gating();
#endif
        std::cout << "Counterfactual experiment tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Counterfactual experiment tests failed: " << error.what() << '\n';
        return 1;
    }
}
