#include "nbody/visualization/controller.hpp"
#include "nbody/visualization/counterfactual_worker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#ifdef NBODY_EXPERIMENT_ENABLE_TEST_HOOKS
#error "Production-linked worker tests must not expose experiment test hooks"
#endif

namespace {

using nbody::experiments::CounterfactualExperimentConfiguration;
using nbody::experiments::ExperimentBackend;
using nbody::experiments::ExperimentBodyIdentity;
using nbody::experiments::ExperimentFailurePhase;
using nbody::experiments::ExperimentIntegrator;
using nbody::experiments::ExperimentSolver;
using nbody::experiments::ExperimentSnapshot;
using nbody::experiments::ExperimentStatus;
using nbody::visualization::CounterfactualExperimentWorker;
using nbody::visualization::CounterfactualPathBranch;
using nbody::visualization::CounterfactualWorkerPublication;
using nbody::visualization::CounterfactualWorkerState;

template <typename Runner, typename = void>
struct exposes_cpu_host_failure_hook : std::false_type {};

template <typename Runner>
struct exposes_cpu_host_failure_hook<Runner, std::void_t<decltype(
    &Runner::inject_cpu_host_failure_before_advance_for_testing)>>
    : std::true_type {};

template <typename Runner, typename = void>
struct exposes_cpu_force_failure_hook : std::false_type {};

template <typename Runner>
struct exposes_cpu_force_failure_hook<Runner, std::void_t<decltype(
    &Runner::inject_cpu_force_failure_after_evaluation_for_testing)>>
    : std::true_type {};

template <typename Runner, typename = void>
struct exposes_cuda_host_failure_hook : std::false_type {};

template <typename Runner>
struct exposes_cuda_host_failure_hook<Runner, std::void_t<decltype(
    &Runner::inject_cuda_host_failure_after_enqueue_for_testing)>>
    : std::true_type {};

static_assert(!exposes_cpu_host_failure_hook<
        nbody::experiments::CounterfactualExperimentRunner>::value,
    "production counterfactual API exposes the host-failure test hook");
static_assert(!exposes_cpu_force_failure_hook<
        nbody::experiments::CounterfactualExperimentRunner>::value,
    "production counterfactual API exposes the force-failure test hook");
static_assert(!exposes_cuda_host_failure_hook<
        nbody::experiments::CounterfactualExperimentRunner>::value,
    "production counterfactual API exposes the CUDA host-failure test hook");

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool same_state(const nbody::SimulationState& left, const nbody::SimulationState& right) {
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

bool same_render_position(const glm::vec3& left, const glm::vec3& right) {
    return std::abs(left.x - right.x) < 1.0e-6F &&
        std::abs(left.y - right.y) < 1.0e-6F &&
        std::abs(left.z - right.z) < 1.0e-6F;
}

ExperimentSnapshot worker_snapshot(
    double time_s = 0.0,
    double maximum_horizon_s = 100000.0,
    bool convergence_established = true) {
    nbody::SimulationState state;
    state.time_s = time_s;
    state.bodies = {
        {"Central", 1.0e20, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"Removed", 1.0e15, {1.0e7, 0.0, 0.0}, {0.0, 1.0, 0.0}},
        {"Tracer", 1.0e10, {2.0e7, 0.0, 0.0}, {0.0, 2.0, 0.0}},
    };
    return ExperimentSnapshot({
        "worker-fixture", "test fixture", "common inertial frame", "central",
        std::move(state),
        {
            {"central", std::nullopt, 1.0, false},
            {"removed", std::nullopt, 1.0, true},
            {"tracer", std::optional<std::string>{"central"}, 1.0, false},
        },
        1.0, ExperimentIntegrator::velocity_verlet,
        convergence_established, maximum_horizon_s,
    });
}

CounterfactualExperimentConfiguration worker_configuration(
    double horizon_s = 10.0) {
    CounterfactualExperimentConfiguration configuration;
    configuration.removed_body_stable_id = "removed";
    configuration.backend = ExperimentBackend::sequential;
    configuration.limits.observation_horizon_s = horizon_s;
    configuration.limits.sample_cadence_s = 1.0;
    configuration.limits.integration_steps_per_batch = 1U;
    configuration.limits.maximum_monitored_bodies = 2U;
    configuration.limits.maximum_stored_samples = 100001U;
    configuration.limits.memory_budget_bytes = 128U * 1024U * 1024U;
    configuration.limits.workload_budget_interactions = 100000000ULL;
    configuration.significance.absolute_separation_threshold_m = 1.0e-12;
    configuration.significance.normalized_deviation_threshold = 0.0;
    configuration.significance.normalization_denominator_floor_m = 1.0;
    configuration.guards.maximum_relative_energy_drift = 1.0;
    configuration.monitored_body_stable_ids = {"central", "tracer"};
    return configuration;
}

void test_snapshot_handoff_and_eligibility() {
    const auto legacy = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    auto live = legacy.initial_state;
    auto snapshot = nbody::visualization::capture_counterfactual_snapshot(legacy, live);
    const auto captured = snapshot.source_state();
    live.bodies[0].position_m.x += 12345.0;
    require(same_state(captured, snapshot.source_state()),
        "snapshot handoff retained a live-state alias");

    const auto candidates =
        nbody::visualization::counterfactual_removal_candidates(legacy);
    require(candidates.size() == 8U &&
        std::find(candidates.begin(), candidates.end(), "sun") == candidates.end(),
        "legacy removal eligibility admitted the Sun or omitted a planet");

    const auto synthetic = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::synthetic, 32U, 24301U);
    require(nbody::visualization::counterfactual_removal_candidates(synthetic).empty(),
        "synthetic bodies were admitted for curated astronomical removal");
}

void test_worker_completion_progress_and_restart() {
    CounterfactualExperimentWorker worker;
    const auto first_generation = worker.start(worker_snapshot(), worker_configuration());
    require(worker.wait_for_progress(first_generation, 1U),
        "worker did not publish paired-batch progress");
    worker.wait_until_idle();
    const auto first = worker.publication();
    require(first.generation == first_generation &&
        first.state == CounterfactualWorkerState::completed && first.result &&
        first.result->status == ExperimentStatus::completed &&
        first.completed_steps == first.total_steps &&
        first.result->completed_steps == first.completed_steps,
        "worker completion publication is incomplete or inconsistent");
    require(first.identities && first.identities->size() == 3U &&
        (*first.identities)[0].stable_id == "central" &&
        (*first.identities)[1].stable_id == "removed" &&
        (*first.identities)[2].stable_id == "tracer" &&
        first.result->monitored_body_stable_ids ==
            std::vector<std::string>({"central", "tracer"}),
        "worker publication lost stable-ID identity mapping");

    const auto second_generation = worker.start(
        worker_snapshot(), worker_configuration(2.0));
    require(second_generation > first_generation,
        "repeated worker start did not create a new generation");
    worker.wait_until_idle();
    const auto second = worker.publication();
    require(second.generation == second_generation &&
        second.state == CounterfactualWorkerState::completed &&
        second.result && second.result->completed_steps == 2U,
        "repeated worker start reused stale results");
}

void test_barnes_hut_worker_metadata_is_immutable() {
    CounterfactualExperimentWorker worker;
    auto configuration = worker_configuration(4.0);
    configuration.solver = ExperimentSolver::barnes_hut;
    configuration.barnes_hut.theta = 0.25;
    configuration.barnes_hut.leaf_capacity = 1U;
    const auto generation = worker.start(worker_snapshot(), configuration);

    // These are next-run edits to the caller's copy. The accepted job and its
    // immutable descriptor must continue with the captured Barnes--Hut policy.
    configuration.solver = ExperimentSolver::direct;
    configuration.barnes_hut.theta = 0.8;
    worker.wait_until_idle();
    const auto publication = worker.publication();
    require(publication.generation == generation && publication.run &&
            publication.result &&
            publication.state == CounterfactualWorkerState::completed &&
            publication.run->solver == ExperimentSolver::barnes_hut &&
            publication.run->barnes_hut.theta == 0.25 &&
            publication.run->barnes_hut.leaf_capacity == 1U &&
            publication.result->solver == ExperimentSolver::barnes_hut &&
            publication.result->barnes_hut_configuration &&
            publication.result->barnes_hut_configuration->theta == 0.25 &&
            publication.result->baseline_barnes_hut_statistics &&
            publication.result->counterfactual_barnes_hut_statistics &&
            publication.result->baseline_barnes_hut_statistics->body_count == 3U &&
            publication.result->counterfactual_barnes_hut_statistics->body_count == 2U,
        "worker publication was relabeled by post-launch solver/theta edits");
}

void test_barnes_hut_failure_context_and_persistent_notice() {
    auto failing_configuration = worker_configuration(2.0);
    failing_configuration.solver = ExperimentSolver::barnes_hut;
    failing_configuration.barnes_hut.leaf_capacity = 1U;
    failing_configuration.barnes_hut.maximum_traversal_work = 1U;
    CounterfactualExperimentWorker failing_worker;
    const auto failing_generation = failing_worker.start(
        worker_snapshot(), failing_configuration,
        {17U, nbody::visualization::CounterfactualSnapshotSource::current_state});
    failing_worker.wait_until_idle();
    const auto failed = failing_worker.publication();
    require(failed.generation == failing_generation && failed.result &&
            failed.state == CounterfactualWorkerState::failed &&
            failed.result->terminal_failure &&
            failed.result->terminal_failure->branch ==
                nbody::experiments::ExperimentBranch::baseline &&
            failed.result->terminal_failure->phase ==
                ExperimentFailurePhase::old_position_force &&
            failed.result->terminal_failure->message_text().find(
                "traversal work budget exceeded") != std::string_view::npos,
        "worker publication lost structured Barnes-Hut failure context");

    auto saturated_configuration = worker_configuration(2.0);
    saturated_configuration.solver = ExperimentSolver::barnes_hut;
    saturated_configuration.barnes_hut.theta = 0.25;
    saturated_configuration.barnes_hut.leaf_capacity = 1U;
    saturated_configuration.limits.maximum_issue_records = 1U;
    CounterfactualExperimentWorker worker;
    nbody::visualization::CounterfactualPublicationGate gate(31U);
    const auto generation = worker.start(
        worker_snapshot(0.0, 100000.0, false), saturated_configuration,
        {31U, nbody::visualization::CounterfactualSnapshotSource::current_state});
    gate.accept_job(generation, 31U);
    saturated_configuration.solver = ExperimentSolver::direct;
    saturated_configuration.barnes_hut.theta = 0.8;
    worker.wait_until_idle();
    const auto saturated = worker.publication();
    const bool retained_approximation_issue = saturated.result &&
        std::any_of(saturated.result->issues.begin(), saturated.result->issues.end(),
            [](const auto& issue) {
                return issue.kind == nbody::experiments::ExperimentIssueKind::
                    approximate_force_solver;
            });
    const auto notice =
        nbody::visualization::counterfactual_barnes_hut_approximation_notice(
            saturated, gate, true);
    require(saturated.state == CounterfactualWorkerState::completed &&
            saturated.result && saturated.run &&
            saturated.result->dropped_issue_count == 1U &&
            !retained_approximation_issue && !notice.empty() &&
            notice.find("matched Direct") != std::string_view::npos &&
            saturated.run->solver == ExperimentSolver::barnes_hut &&
            saturated.run->barnes_hut.theta == 0.25,
        "executed Barnes-Hut notice depended on bounded warnings or live controls");
    require(nbody::visualization::counterfactual_barnes_hut_approximation_notice(
                saturated, gate, false).empty(),
        "composition mismatch did not suppress the executed-solver notice");

    gate.replace_scenario();
    require(nbody::visualization::counterfactual_barnes_hut_approximation_notice(
                saturated, gate, true).empty(),
        "scenario replacement did not invalidate the prior Barnes-Hut notice");

    auto direct_configuration = worker_configuration(1.0);
    const auto direct_generation = worker.start(
        worker_snapshot(), direct_configuration,
        {gate.scenario_generation(),
         nbody::visualization::CounterfactualSnapshotSource::initial_state});
    gate.accept_job(direct_generation, gate.scenario_generation());
    worker.wait_until_idle();
    const auto direct = worker.publication();
    require(direct.state == CounterfactualWorkerState::completed &&
            direct.run && direct.run->solver == ExperimentSolver::direct &&
            nbody::visualization::counterfactual_barnes_hut_approximation_notice(
                direct, gate, true).empty(),
        "Direct execution incorrectly displayed a Barnes-Hut approximation notice");

    gate.invalidate_job();
    require(nbody::visualization::counterfactual_barnes_hut_approximation_notice(
                direct, gate, true).empty(),
        "reset-style job invalidation left an executed-solver notice visible");
}

std::shared_ptr<const nbody::visualization::CounterfactualRunDescriptor>
run_identity(std::uint64_t scenario_generation, std::uint64_t worker_generation) {
    auto run = std::make_shared<nbody::visualization::CounterfactualRunDescriptor>();
    run->scenario_generation = scenario_generation;
    run->worker_generation = worker_generation;
    return run;
}

void test_application_publication_generation_gate() {
    nbody::visualization::CounterfactualPublicationGate gate;
    CounterfactualWorkerPublication first;
    first.generation = 11U;
    first.state = CounterfactualWorkerState::running;
    first.run = run_identity(1U, 11U);
    first.result = std::make_shared<
        const nbody::experiments::CounterfactualExperimentResult>();
    first.worker_error = "stale diagnostic";
    gate.accept_job(11U, 1U);
    require(gate.accepts(first), "accepted job publication was rejected");

    gate.replace_scenario();
    require(gate.scenario_generation() == 2U && !gate.accepts(first),
        "same-ID scenario replacement did not invalidate stale progress");
    first.state = CounterfactualWorkerState::completed;
    require(!gate.accepts(first),
        "worker completion after scenario replacement revived stale output");
    gate.replace_scenario();
    require(gate.scenario_generation() == 3U && !gate.accepts(first),
        "switching away and back made an old job current again");

    CounterfactualWorkerPublication second;
    second.generation = 12U;
    second.state = CounterfactualWorkerState::pending;
    second.run = run_identity(3U, 12U);
    gate.accept_job(12U, 3U);
    require(gate.accepts(second) && !gate.accepts(first),
        "repeated start did not replace the accepted worker generation");
    second.state = CounterfactualWorkerState::cancelled;
    require(gate.accepts(second), "cancelled current job lost its identity");

    CounterfactualWorkerPublication third;
    third.generation = 13U;
    third.state = CounterfactualWorkerState::running;
    third.run = run_identity(3U, 13U);
    gate.accept_job(13U, 3U);
    require(gate.accepts(third) && !gate.accepts(second),
        "new job after cancellation accepted stale terminal output");

    bool overflow_rejected = false;
    try {
        (void)nbody::visualization::checked_next_counterfactual_generation(
            std::numeric_limits<std::uint64_t>::max());
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    require(overflow_rejected,
        "counterfactual generation rollover was not rejected");
    nbody::visualization::CounterfactualPublicationGate terminal_gate(
        std::numeric_limits<std::uint64_t>::max());
    overflow_rejected = false;
    try {
        terminal_gate.replace_scenario();
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    require(overflow_rejected && terminal_gate.scenario_generation() ==
            std::numeric_limits<std::uint64_t>::max(),
        "scenario-generation overflow was not rejected transactionally");
}

void test_immutable_executed_run_descriptor() {
    auto configuration = worker_configuration(100000.0);
    configuration.limits.sample_cadence_s = 100000.0;
    configuration.significance.absolute_separation_threshold_m = 123.0;
    configuration.significance.normalized_deviation_threshold = 4.5e-6;
    configuration.significance.persistence_samples = 7U;
    configuration.cuda_device = 7;
    CounterfactualExperimentWorker worker;
    const auto generation = worker.start(
        worker_snapshot(42.0, 100000.0), configuration,
        {17U, nbody::visualization::CounterfactualSnapshotSource::initial_state});

    configuration.backend = ExperimentBackend::openmp;
    configuration.openmp_threads = 99;
    configuration.cuda_device = 99;
    configuration.removed_body_stable_id = "tracer";
    configuration.limits.observation_horizon_s = 1.0;
    configuration.limits.sample_cadence_s = 1.0;
    configuration.monitored_body_stable_ids = {"tracer"};
    configuration.significance.absolute_separation_threshold_m = 999.0;
    configuration.significance.normalized_deviation_threshold = 0.5;
    configuration.significance.persistence_samples = 99U;

    require(worker.wait_for_progress(generation, 1U),
        "descriptor fixture did not publish progress");
    const auto running = worker.publication();
    require(running.run && running.run->scenario_generation == 17U &&
        running.run->worker_generation == generation &&
        running.run->snapshot_source ==
            nbody::visualization::CounterfactualSnapshotSource::initial_state &&
        running.run->snapshot_time_s == 42.0 &&
        running.run->source_state_provenance == "test fixture" &&
        running.run->coordinate_frame == "common inertial frame" &&
        running.run->scenario_id == "worker-fixture" &&
        running.run->removed_body_stable_id == "removed" &&
        running.run->backend == ExperimentBackend::sequential &&
        running.run->openmp_threads == 1 &&
        running.run->cuda_device == 7 &&
        running.run->physical_timestep_s == 1.0 &&
        running.run->observation_horizon_s == 100000.0 &&
        running.run->sample_cadence_s == 100000.0 &&
        running.run->monitored_body_stable_ids ==
            std::vector<std::string>({"central", "tracer"}) &&
        running.run->absolute_separation_threshold_m == 123.0 &&
        running.run->normalized_deviation_threshold == 4.5e-6 &&
        running.run->persistence_samples == 7U &&
        running.run->accepted_step_count == 100000U &&
        running.run->estimated_direct_force_interactions > 0U &&
        running.run->estimated_retained_bytes > 0U,
        "running publication did not retain the submitted immutable descriptor");
    worker.request_cancel();
    worker.wait_until_idle();
    const auto terminal = worker.publication();
    require(terminal.run == running.run &&
        terminal.run->observation_horizon_s == 100000.0,
        "cancelled publication did not preserve its submitted metadata");

    auto completed_configuration = worker_configuration(2.0);
    CounterfactualExperimentWorker completed_worker;
    (void)completed_worker.start(worker_snapshot(), completed_configuration,
        {23U, nbody::visualization::CounterfactualSnapshotSource::current_state});
    completed_worker.wait_until_idle();
    const auto completed = completed_worker.publication();
    require(completed.state == CounterfactualWorkerState::completed && completed.run,
        "completed descriptor fixture did not finish");
    completed_configuration.removed_body_stable_id = "tracer";
    completed_configuration.limits.observation_horizon_s = 999.0;
    completed_configuration.significance.absolute_separation_threshold_m = 777.0;
    const auto inspected_again = completed_worker.publication();
    require(inspected_again.run == completed.run &&
        inspected_again.run->removed_body_stable_id == "removed" &&
        inspected_again.run->observation_horizon_s == 2.0 &&
        inspected_again.run->absolute_separation_threshold_m == 1.0e-12,
        "post-completion editor changes altered executed metadata");
}

void test_cuda_worker_metadata_and_rejection() {
    auto configuration = worker_configuration(2.0);
    configuration.backend = ExperimentBackend::cuda;
    configuration.cuda_device = std::numeric_limits<int>::max();
    CounterfactualExperimentWorker worker;
    const auto generation = worker.start(
        worker_snapshot(), configuration,
        {23U, nbody::visualization::CounterfactualSnapshotSource::current_state});
    worker.wait_until_idle();
    const auto publication = worker.publication();
    require(publication.generation == generation && publication.run &&
            publication.run->backend == ExperimentBackend::cuda &&
            publication.run->cuda_device == std::numeric_limits<int>::max() &&
            publication.state == CounterfactualWorkerState::failed &&
            !publication.result && !publication.worker_error.empty(),
        "unavailable CUDA worker request fell back or lost immutable metadata");
}

void test_cancellation_and_active_destruction() {
    CounterfactualExperimentWorker worker;
    auto long_configuration = worker_configuration(100000.0);
    long_configuration.solver = ExperimentSolver::barnes_hut;
    long_configuration.barnes_hut.theta = 0.25;
    long_configuration.barnes_hut.leaf_capacity = 1U;
    long_configuration.limits.sample_cadence_s = 100000.0;
    const auto generation = worker.start(
        worker_snapshot(0.0, 100000.0), long_configuration);
    require(worker.wait_for_progress(generation, 1U),
        "long worker fixture did not reach a synchronized boundary");
    worker.request_cancel();
    worker.wait_until_idle();
    const auto cancelled = worker.publication();
    require(cancelled.state == CounterfactualWorkerState::cancelled &&
        cancelled.result && cancelled.result->status == ExperimentStatus::cancelled &&
        cancelled.completed_steps > 0U &&
        cancelled.completed_steps < cancelled.total_steps,
        "cooperative cancellation did not stop at a paired boundary");

    {
        CounterfactualExperimentWorker active;
        (void)active.start(worker_snapshot(0.0, 100000.0), long_configuration);
    }
}

void test_failure_preserves_committed_result() {
    CounterfactualExperimentWorker worker;
    auto configuration = worker_configuration(2.0);
    configuration.limits.sample_cadence_s = 1.0;
    const auto generation = worker.start(
        worker_snapshot(9007199254740991.0, 2.0), configuration);
    worker.wait_until_idle();
    const auto publication = worker.publication();
    require(publication.generation == generation &&
        publication.state == CounterfactualWorkerState::failed &&
        publication.result && publication.result->status == ExperimentStatus::failed &&
        publication.result->completed_steps == 1U &&
        publication.result->samples.size() == 2U &&
        publication.result->terminal_failure.has_value() &&
        publication.run && publication.run->snapshot_time_s == 9007199254740991.0 &&
        publication.run->removed_body_stable_id == "removed" &&
        publication.run->observation_horizon_s == 2.0,
        "worker lost the last committed paired result after failure");
}

void test_scenario_switch_does_not_alias_live_controller() {
    const auto legacy = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    nbody::visualization::SimulationController controller(
        legacy.initial_state, {legacy.dt_s, 1U,
            nbody::visualization::BackendKind::sequential, 1, 0});
    const auto controller_before = controller.state();
    auto snapshot = nbody::visualization::capture_counterfactual_snapshot(
        legacy, controller.state());
    auto configuration = worker_configuration(legacy.dt_s);
    configuration.removed_body_stable_id = "jupiter";
    configuration.limits.sample_cadence_s = legacy.dt_s;
    configuration.limits.maximum_monitored_bodies = 8U;
    configuration.limits.maximum_stored_samples = 2U;
    configuration.limits.workload_budget_interactions = 100000U;
    configuration.monitored_body_stable_ids = {"earth"};

    CounterfactualExperimentWorker worker;
    (void)worker.start(std::move(snapshot), configuration);
    const auto catalog = nbody::visualization::make_astronomical_catalog_scenario();
    controller.replace_scenario(catalog.initial_state, catalog.dt_s, 1U);
    worker.request_cancel();
    worker.wait_until_idle();
    require(controller.state().bodies.size() == 60U &&
        same_state(controller.initial_state(), catalog.initial_state) &&
        !same_state(controller_before, controller.state()),
        "scenario switching and worker execution shared mutable controller state");
}

void test_worker_does_not_mutate_live_controller() {
    const auto legacy = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    nbody::visualization::SimulationController controller(
        legacy.initial_state, {legacy.dt_s, 1U,
            nbody::visualization::BackendKind::sequential, 1, 0});
    controller.set_paused(true);
    const auto before = controller.state();
    auto snapshot = nbody::visualization::capture_counterfactual_snapshot(
        legacy, controller.state());
    auto configuration = worker_configuration(legacy.dt_s);
    configuration.removed_body_stable_id = "jupiter";
    configuration.limits.sample_cadence_s = legacy.dt_s;
    configuration.limits.maximum_monitored_bodies = 8U;
    configuration.limits.maximum_stored_samples = 2U;
    configuration.limits.workload_budget_interactions = 100000U;
    configuration.monitored_body_stable_ids = {"earth"};

    CounterfactualExperimentWorker worker;
    (void)worker.start(std::move(snapshot), configuration);
    worker.wait_until_idle();
    require(worker.publication().state == CounterfactualWorkerState::completed &&
        same_state(before, controller.state()),
        "worker experiment mutated the live simulation controller");
}

void test_live_solver_switch_and_reset_do_not_retarget_worker() {
    const auto legacy = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    nbody::visualization::SimulationController controller(
        legacy.initial_state,
        {legacy.dt_s, 1U, nbody::visualization::BackendKind::sequential, 1, 0});

    auto configuration = worker_configuration(100000.0);
    configuration.limits.sample_cadence_s = 100000.0;
    CounterfactualExperimentWorker worker;
    nbody::visualization::CounterfactualPublicationGate gate(41U);
    const auto generation = worker.start(
        worker_snapshot(0.0, 100000.0), configuration,
        {41U, nbody::visualization::CounterfactualSnapshotSource::current_state});
    gate.accept_job(generation, 41U);
    require(worker.wait_for_progress(generation, 1U),
        "solver-switch fixture did not reach a synchronized worker boundary");
    const auto before_switch = worker.publication();
    require(gate.accepts(before_switch) && before_switch.run &&
        before_switch.run->backend == ExperimentBackend::sequential &&
        before_switch.run->estimated_direct_force_interactions > 0U,
        "worker publication omitted its immutable direct-force execution metadata");

    auto application_publication = before_switch;
    nbody::visualization::CounterfactualPresentationCache presentation;
    nbody::experiments::CounterfactualExperimentResult cached_result;
    presentation.generation = generation;
    presentation.result = &cached_result;
    presentation.result_rows.push_back({});
    presentation.warning_groups.push_back({});
    presentation.error = "cached presentation error";
    presentation.frame_error = "cached frame error";

    nbody::visualization::invalidate_counterfactual_application_publication(
        worker, gate, application_publication, presentation);
    require(worker.cancellation_requested() &&
            !gate.accepted_worker_generation().has_value() &&
            !gate.accepts(before_switch) &&
            application_publication.generation == 0U &&
            !application_publication.result && presentation.empty(),
        "application reset lifecycle did not cancel, invalidate, or clear publication caches");

    nbody::BarnesHutConfiguration barnes_hut;
    barnes_hut.theta = 0.25;
    controller.select_solver(
        nbody::visualization::SolverKind::barnes_hut, barnes_hut);
    controller.reset();
    require(controller.solver_kind() ==
                nbody::visualization::SolverKind::barnes_hut &&
            same_state(controller.state(), legacy.initial_state),
        "live solver selection/reset did not preserve the scenario initial state");

    worker.request_cancel();
    worker.wait_until_idle();
    const auto terminal = worker.publication();
    require(!gate.accepts(terminal) && terminal.run == before_switch.run &&
            terminal.run->backend == ExperimentBackend::sequential &&
            terminal.run->estimated_direct_force_interactions ==
                before_switch.run->estimated_direct_force_interactions,
        "late terminal completion passed the reset gate or changed immutable metadata");
    if (gate.accepts(terminal)) {
        application_publication = terminal;
    }
    require(application_publication.generation == 0U && presentation.empty(),
        "rejected late terminal completion repopulated application presentation state");

    auto fresh_configuration = worker_configuration(2.0);
    const auto fresh_generation = worker.start(
        worker_snapshot(), fresh_configuration,
        {41U, nbody::visualization::CounterfactualSnapshotSource::current_state});
    gate.accept_job(fresh_generation, 41U);
    worker.wait_until_idle();
    const auto fresh = worker.publication();
    if (gate.accepts(fresh)) {
        application_publication = fresh;
    }
    require(fresh_generation != generation && gate.accepts(fresh) &&
            fresh.state == CounterfactualWorkerState::completed && fresh.result &&
            application_publication.generation == fresh_generation,
        "fresh post-reset job could not publish through the invalidated gate");
}

void test_resource_estimate_and_normalized_threshold() {
    auto snapshot = worker_snapshot();
    auto configuration = worker_configuration(10.0);
    configuration.limits.sample_cadence_s = 2.0;
    const auto estimate = nbody::visualization::estimate_counterfactual_resources(
        snapshot, configuration);
    require(estimate.integration_steps == 10U && estimate.retained_samples == 6U &&
        estimate.monitored_bodies == 2U &&
        estimate.direct_force_interactions == 160U &&
        estimate.conservative_retained_bytes > 0U,
        "worker resource estimate does not match engine batching semantics");

    configuration.significance.normalized_deviation_threshold = 1.0e100;
    CounterfactualExperimentWorker worker;
    (void)worker.start(std::move(snapshot), configuration);
    worker.wait_until_idle();
    const auto publication = worker.publication();
    require(publication.result &&
        std::none_of(publication.result->summaries.begin(),
            publication.result->summaries.end(),
            [](const auto& summary) { return summary.significant; }),
        "normalized deviation threshold was ignored");

    auto rejected_configuration = worker_configuration(10.0);
    rejected_configuration.limits.maximum_stored_samples = 1U;
    CounterfactualExperimentWorker rejected_worker;
    (void)rejected_worker.start(worker_snapshot(), rejected_configuration);
    rejected_worker.wait_until_idle();
    const auto rejected = rejected_worker.publication();
    require(rejected.state == CounterfactualWorkerState::failed &&
        !rejected.worker_error.empty() && !rejected.result,
        "worker did not surface engine history-admission failure");
}

void test_shared_render_space_paths() {
    auto result = std::make_shared<nbody::experiments::CounterfactualExperimentResult>();
    result->monitored_body_stable_ids = {"central", "tracer"};
    nbody::experiments::ExperimentSample first;
    first.bodies.resize(2U);
    first.bodies[0].baseline_position_m = {100.0, 0.0, 0.0};
    first.bodies[0].counterfactual_position_m = {101.0, 0.0, 0.0};
    first.bodies[1].baseline_position_m = {105.0, 0.0, 0.0};
    first.bodies[1].counterfactual_position_m = {106.0, 0.0, 0.0};
    auto second = first;
    second.bodies[0].baseline_position_m = {110.0, 0.0, 0.0};
    second.bodies[0].counterfactual_position_m = {112.0, 0.0, 0.0};
    second.bodies[1].baseline_position_m = {117.0, 0.0, 0.0};
    second.bodies[1].counterfactual_position_m = {120.0, 0.0, 0.0};
    result->samples = {first, second};

    CounterfactualWorkerPublication publication;
    publication.state = CounterfactualWorkerState::completed;
    publication.result = result;
    nbody::SimulationState reference;
    reference.bodies = {
        {"A", 1.0, {0.0, 0.0, 0.0}, {}},
        {"B", 1.0, {10.0, 0.0, 0.0}, {}},
    };
    nbody::visualization::RenderSpaceTransform transform(reference, {}, 10.0F);
    const auto reconstructed = nbody::visualization::build_counterfactual_render_paths(
        publication, transform, {"tracer"}, std::optional<std::string>{"central"});
    const auto& paths = reconstructed.paths;
    require(paths.size() == 2U &&
        reconstructed.frame_unavailable_reason.empty() &&
        paths[0].branch == CounterfactualPathBranch::baseline &&
        paths[1].branch == CounterfactualPathBranch::counterfactual &&
        paths[0].points.size() == 2U && paths[1].points.size() == 2U &&
        std::abs(paths[0].points[0].x - 5.0F) < 1.0e-6F &&
        std::abs(paths[0].points[1].x - 7.0F) < 1.0e-6F &&
        std::abs(paths[1].points[0].x - 6.0F) < 1.0e-6F &&
        std::abs(paths[1].points[1].x - 10.0F) < 1.0e-6F,
        "counterfactual paths did not share the baseline parent frame");
}

void test_hidden_reference_origin_and_frame_failures() {
    auto configuration = worker_configuration(2.0);
    configuration.monitored_body_stable_ids = {"tracer"};
    configuration.retained_reference_body_stable_ids = {"central"};
    const auto estimate = nbody::visualization::estimate_counterfactual_resources(
        worker_snapshot(), configuration);
    auto without_reference = configuration;
    without_reference.retained_reference_body_stable_ids.clear();
    const auto estimate_without = nbody::visualization::estimate_counterfactual_resources(
        worker_snapshot(), without_reference);
    require(estimate.retained_reference_bodies == 1U &&
        estimate.conservative_retained_bytes >
            estimate_without.conservative_retained_bytes,
        "hidden origin samples were not included in resource admission");

    CounterfactualExperimentWorker worker;
    (void)worker.start(worker_snapshot(), configuration);
    worker.wait_until_idle();
    auto publication = worker.publication();
    require(publication.result &&
        publication.result->monitored_body_stable_ids ==
            std::vector<std::string>({"tracer"}) &&
        publication.result->reference_body_stable_ids ==
            std::vector<std::string>({"central"}) &&
        publication.result->summaries.size() == 1U &&
        publication.result->summaries[0].stable_id == "tracer",
        "hidden origin incorrectly acquired monitored trajectory metrics");

    nbody::SimulationState reference;
    reference.bodies = {
        {"A", 1.0, {0.0, 0.0, 0.0}, {}},
        {"B", 1.0, {10.0, 0.0, 0.0}, {}},
    };
    nbody::visualization::RenderSpaceTransform transform(reference, {}, 10.0F);
    const auto reconstructed = nbody::visualization::build_counterfactual_render_paths(
        publication, transform, {"tracer"}, std::optional<std::string>{"central"});
    require(reconstructed.frame_unavailable_reason.empty() &&
        reconstructed.paths.size() == 2U,
        "unmonitored surviving origin was not retained for rendering");
    const auto local_status = nbody::visualization::counterfactual_frame_status(
        publication, std::optional<std::string>{"central"});
    const auto inertial_status = nbody::visualization::counterfactual_frame_status(
        publication);
    require(local_status.available &&
        local_status.coordinate_frame_label.find("central") != std::string::npos &&
        local_status.explanation.find("never independently recentered") !=
            std::string::npos &&
        inertial_status.available &&
        inertial_status.coordinate_frame_label.find("inertial") != std::string::npos,
        "available comparison frames are not labeled with their shared-frame contract");
    for (std::size_t sample_index = 0;
         sample_index < publication.result->samples.size(); ++sample_index) {
        const auto& sample = publication.result->samples[sample_index];
        const auto expected_baseline = transform.to_render_offset(
            sample.bodies[0].baseline_position_m -
            sample.baseline_reference_positions_m[0]);
        const auto expected_counterfactual = transform.to_render_offset(
            sample.bodies[0].counterfactual_position_m -
            sample.baseline_reference_positions_m[0]);
        require(same_render_position(
                    reconstructed.paths[0].points[sample_index], expected_baseline) &&
            same_render_position(
                    reconstructed.paths[1].points[sample_index], expected_counterfactual),
            "local overlay used a current or independently recentered origin");
    }

    const auto changed_origin = nbody::visualization::build_counterfactual_render_paths(
        publication, transform, {"tracer"}, std::optional<std::string>{"other"});
    require(changed_origin.paths.empty() &&
        changed_origin.frame_unavailable_reason.find("no retained historical") !=
            std::string::npos &&
        changed_origin.frame_unavailable_reason.find("SSB/inertial") !=
            std::string::npos,
        "switching to an unretained local origin silently changed frames");

    auto removed_result = std::make_shared<
        nbody::experiments::CounterfactualExperimentResult>(*publication.result);
    removed_result->removed_body_stable_id = "central";
    publication.result = removed_result;
    const auto removed_origin = nbody::visualization::build_counterfactual_render_paths(
        publication, transform, {"tracer"}, std::optional<std::string>{"central"});
    require(removed_origin.paths.empty() &&
        removed_origin.frame_unavailable_reason.find("removed") != std::string::npos &&
        removed_origin.frame_unavailable_reason.find("surviving common origin") !=
            std::string::npos,
        "removed local origin did not suppress its overlay explicitly");

    removed_result->removed_body_stable_id = "removed";
    removed_result->samples[0].baseline_reference_positions_m.clear();
    const auto incomplete_origin = nbody::visualization::build_counterfactual_render_paths(
        publication, transform, {"tracer"}, std::optional<std::string>{"central"});
    require(incomplete_origin.paths.empty() &&
        incomplete_origin.frame_unavailable_reason.find("incomplete") !=
            std::string::npos &&
        incomplete_origin.frame_unavailable_reason.find("SSB/inertial") !=
            std::string::npos,
        "missing origin samples were not suppressed explicitly");
}

void test_overlay_visibility_policy() {
    const auto legacy = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    std::vector<bool> profile(legacy.stable_ids.size(), false);
    std::vector<bool> individual(legacy.stable_ids.size(), true);
    auto classes = nbody::visualization::BodyClassVisibility{{
        true, true, true, true, true, true, true}};
    const auto earth = *nbody::visualization::find_body_index(legacy, "earth");
    profile[earth] = true;
    auto visible = nbody::visualization::counterfactual_visible_overlay_ids(
        legacy, profile, individual, classes,
        {"earth", "jupiter", "earth"}, 12U);
    require(visible == std::vector<std::string>({"earth"}),
        "profile visibility did not filter or deduplicate overlay IDs");
    individual[earth] = false;
    visible = nbody::visualization::counterfactual_visible_overlay_ids(
        legacy, profile, individual, classes, {"earth"}, 12U);
    require(visible.empty(),
        "individual visibility did not suppress the overlay");
    individual[earth] = true;
    classes[nbody::visualization::body_class_visibility_index(
        nbody::astronomy::AstronomicalBodyClass::planet)] = false;
    visible = nbody::visualization::counterfactual_visible_overlay_ids(
        legacy, profile, individual, classes, {"earth"}, 12U);
    require(visible.empty(), "class visibility did not suppress the overlay");
}

void test_result_presentation_rows_sorting_and_missing_metrics() {
    nbody::experiments::CounterfactualExperimentResult result;
    nbody::experiments::BodyDeviationSummary tracer;
    tracer.stable_id = "tracer";
    tracer.final_absolute_separation_m = 3.0;
    tracer.maximum_absolute_separation_m = 8.0;
    tracer.fixed_normalization_scale_m = 20.0;
    tracer.final_normalized_deviation = 0.15;
    tracer.final_velocity_difference_m_s = 0.25;
    tracer.final_parent_relative_separation_m = 2.0;
    tracer.first_threshold_crossing_time_s = 4.0;
    tracer.persistent_threshold_time_s = 6.0;
    tracer.significant = true;
    nbody::experiments::BodyDeviationSummary central;
    central.stable_id = "central";
    central.final_absolute_separation_m = 1.0;
    central.maximum_absolute_separation_m = 9.0;
    central.fixed_normalization_scale_m = 10.0;
    central.final_normalized_deviation = 0.1;
    central.final_velocity_difference_m_s = 0.5;
    result.summaries = {tracer, central};
    const auto original = result.summaries;

    auto rows = nbody::visualization::build_counterfactual_result_rows(
        result, {{"central", "Central Body"}, {"tracer", "Tracer Body"}});
    require(rows.size() == 2U && rows[0].stable_id == "central" &&
        rows[0].display_name == "Central Body" &&
        rows[1].stable_id == "tracer" && rows[1].significant &&
        rows[1].first_threshold_crossing_time_s == 4.0 &&
        rows[1].persistent_threshold_time_s == 6.0,
        "stable-ID result rows lost names or threshold/persistence semantics");
    nbody::visualization::sort_counterfactual_result_rows(
        rows,
        nbody::visualization::CounterfactualResultSortKey::maximum_absolute_separation,
        false);
    require(rows[0].stable_id == "central" && rows[1].stable_id == "tracer",
        "descending numerical result sorting is incorrect");
    nbody::visualization::sort_counterfactual_result_rows(
        rows,
        nbody::visualization::CounterfactualResultSortKey::parent_relative_separation,
        false);
    require(rows[0].stable_id == "tracer" && rows[1].stable_id == "central",
        "missing optional metrics did not remain last during sorting");
    require(nbody::visualization::counterfactual_optional_measurement(
                std::nullopt, "s") == "N/A" &&
        nbody::visualization::counterfactual_optional_measurement(
                std::optional<double>{0.0}, "s").find(" s") != std::string::npos,
        "missing or zero-valued metric presentation is ambiguous");
    require(result.summaries.size() == original.size() &&
        result.summaries[0].stable_id == original[0].stable_id &&
        result.summaries[0].maximum_absolute_separation_m ==
            original[0].maximum_absolute_separation_m,
        "presentation row construction mutated the committed result");

    bool unknown_id_rejected = false;
    try {
        (void)nbody::visualization::build_counterfactual_result_rows(
            result, {{"central", "Central Body"}});
    } catch (const std::logic_error&) {
        unknown_id_rejected = true;
    }
    require(unknown_id_rejected,
        "presentation silently accepted a result stable ID absent from the scenario");
}

void test_warning_grouping_and_control_layout() {
    nbody::experiments::CounterfactualExperimentResult result;
    const auto add_issue = [&](
        nbody::experiments::ExperimentIssueKind kind,
        std::optional<nbody::experiments::ExperimentBranch> branch,
        const char* message) {
        nbody::experiments::ExperimentIssue issue;
        issue.kind = kind;
        issue.branch = branch;
        issue.message = message;
        result.issues.push_back(std::move(issue));
    };
    add_issue(nbody::experiments::ExperimentIssueKind::dangerous_close_approach,
        nbody::experiments::ExperimentBranch::baseline, "first");
    add_issue(nbody::experiments::ExperimentIssueKind::dangerous_close_approach,
        nbody::experiments::ExperimentBranch::baseline, "second");
    add_issue(nbody::experiments::ExperimentIssueKind::dangerous_close_approach,
        nbody::experiments::ExperimentBranch::counterfactual, "third");
    add_issue(nbody::experiments::ExperimentIssueKind::excessive_conservation_drift,
        std::nullopt, "fourth");
    result.dropped_issue_count = 7U;
    const auto messages_before = std::vector<std::string>{
        result.issues[0].message, result.issues[1].message,
        result.issues[2].message, result.issues[3].message};
    const auto groups = nbody::visualization::group_counterfactual_warnings(result);
    require(groups.size() == 3U &&
        groups[0].retained_issue_indices == std::vector<std::size_t>({0U, 1U}) &&
        groups[1].retained_issue_indices == std::vector<std::size_t>({2U}) &&
        groups[2].retained_issue_indices == std::vector<std::size_t>({3U}) &&
        result.dropped_issue_count == 7U &&
        result.issues[0].message == messages_before[0] &&
        std::string(nbody::visualization::counterfactual_issue_branch_name(
            groups[1].branch)) == "Counterfactual",
        "warning grouping lost branch/type identity, dropped counts, or retained text");

    const auto narrow =
        nbody::visualization::counterfactual_control_layout(280.0F);
    const auto wide =
        nbody::visualization::counterfactual_control_layout(520.0F);
    require(narrow.stacked && narrow.value_width == 280.0F &&
        !wide.stacked && wide.label_width >= 140.0F &&
        wide.value_width > 0.0F &&
        wide.label_width + wide.value_width <= 520.0F,
        "responsive experiment-control layout can clip its label/value columns");
}

void test_openmp_worker() {
    auto configuration = worker_configuration(2.0);
    configuration.backend = ExperimentBackend::openmp;
    configuration.solver = ExperimentSolver::barnes_hut;
    configuration.barnes_hut.theta = 0.25;
    configuration.barnes_hut.leaf_capacity = 1U;
    configuration.openmp_threads = 2;
    if (!nbody::experiments::CounterfactualExperimentRunner::openmp_supported()) {
        bool rejected = false;
        try {
            CounterfactualExperimentWorker worker;
            (void)worker.start(worker_snapshot(), configuration);
            worker.wait_until_idle();
            rejected = worker.publication().state == CounterfactualWorkerState::failed;
        } catch (...) {
            rejected = true;
        }
        require(rejected, "CPU-only worker silently accepted OpenMP");
        return;
    }
    CounterfactualExperimentWorker worker;
    (void)worker.start(worker_snapshot(), configuration);
    worker.wait_until_idle();
    const auto publication = worker.publication();
    require(publication.state == CounterfactualWorkerState::completed &&
            publication.run && publication.result &&
            publication.run->backend == ExperimentBackend::openmp &&
            publication.run->solver == ExperimentSolver::barnes_hut &&
            publication.run->barnes_hut.theta == 0.25 &&
            publication.result->baseline_barnes_hut_statistics &&
            publication.result->counterfactual_barnes_hut_statistics &&
            publication.result->baseline_barnes_hut_statistics->worker_threads == 2 &&
            publication.result->counterfactual_barnes_hut_statistics->worker_threads == 2,
        "OpenMP worker experiment did not complete");
}

}  // namespace

int main() {
    try {
        static_assert(!std::is_copy_constructible_v<CounterfactualExperimentWorker>);
        static_assert(!std::is_move_constructible_v<CounterfactualExperimentWorker>);
        static_assert(std::is_const_v<std::remove_reference_t<
            decltype(*std::declval<CounterfactualWorkerPublication>().result)>>);
        test_snapshot_handoff_and_eligibility();
        test_worker_completion_progress_and_restart();
        test_barnes_hut_worker_metadata_is_immutable();
        test_barnes_hut_failure_context_and_persistent_notice();
        test_application_publication_generation_gate();
        test_immutable_executed_run_descriptor();
        test_cuda_worker_metadata_and_rejection();
        test_cancellation_and_active_destruction();
        test_failure_preserves_committed_result();
        test_scenario_switch_does_not_alias_live_controller();
        test_worker_does_not_mutate_live_controller();
        test_live_solver_switch_and_reset_do_not_retarget_worker();
        test_resource_estimate_and_normalized_threshold();
        test_shared_render_space_paths();
        test_hidden_reference_origin_and_frame_failures();
        test_overlay_visibility_policy();
        test_result_presentation_rows_sorting_and_missing_metrics();
        test_warning_grouping_and_control_layout();
        test_openmp_worker();
        std::cout << "Counterfactual worker tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Counterfactual worker tests failed: " << error.what() << '\n';
        return 1;
    }
}
