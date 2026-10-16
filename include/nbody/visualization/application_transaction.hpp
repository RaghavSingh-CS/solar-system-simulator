#pragma once

#include "nbody/visualization/controller.hpp"
#include "nbody/visualization/counterfactual_worker.hpp"
#include "nbody/visualization/scenario.hpp"
#include "nbody/visualization/workload.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace nbody::visualization {

struct SimulationTimingSample {
    double batch_milliseconds{0.0};
    double simulation_steps_per_second{0.0};
};

// Timing belongs to one effective execution configuration. A caller captures
// generation() before starting work and may publish only while that generation
// is still current. Generation zero is a permanently exhausted sentinel; this
// avoids wrapping to an old token after UINT64_MAX invalidations.
class SimulationTimingTracker {
public:
    std::uint64_t generation() const noexcept { return generation_; }
    const std::optional<SimulationTimingSample>& sample() const noexcept {
        return sample_;
    }
    bool publish(
        std::uint64_t generation,
        double batch_milliseconds,
        double simulation_steps_per_second) noexcept;
    void invalidate() noexcept;

private:
    std::uint64_t generation_{1U};
    std::optional<SimulationTimingSample> sample_;
};

bool same_barnes_hut_configuration(
    const BarnesHutConfiguration& left,
    const BarnesHutConfiguration& right) noexcept;

bool backend_selection_changes_execution(
    const SimulationController& controller,
    BackendKind backend,
    int openmp_threads,
    int cuda_device) noexcept;

bool solver_selection_changes_execution(
    const SimulationController& controller,
    SolverKind solver,
    const BarnesHutConfiguration& barnes_hut) noexcept;

bool steps_per_frame_changes_execution(
    const SimulationController& controller,
    std::size_t steps_per_frame) noexcept;

void select_application_backend(
    SimulationController& controller,
    SimulationTimingTracker& timing,
    BackendKind backend,
    int openmp_threads = 1,
    int cuda_device = 0);

void select_application_solver(
    SimulationController& controller,
    SimulationTimingTracker& timing,
    SolverKind solver,
    const BarnesHutConfiguration& barnes_hut);

void set_application_steps_per_frame(
    SimulationController& controller,
    SimulationTimingTracker& timing,
    std::size_t steps_per_frame);

struct PreparedApplicationScenarioTransaction {
    PreparedScenarioReplacement visual;
    InteractiveWorkloadPolicy workload;
    std::uint64_t next_scenario_generation{0U};
};

using ScenarioPreparationCompletion =
    std::function<void(const PreparedScenarioReplacement&)>;

// Builds and validates every replacement resource before the active
// application is mutated. completion is the production extension point used
// to prepare GUI-owned analysis resources and a deterministic failure-injection
// point for headless transaction tests.
PreparedApplicationScenarioTransaction prepare_application_scenario_transaction(
    ScenarioKind kind,
    std::size_t synthetic_body_count,
    std::uint64_t synthetic_seed,
    LargeNDistribution synthetic_distribution,
    const std::vector<std::string>& included_stable_ids,
    const SimulationController& controller,
    const CounterfactualPublicationGate& gate,
    const ScenarioPreparationCompletion& completion = {});

struct ApplicationScenarioTransactionTargets {
    SimulationController& controller;
    ScenarioDefinition& scenario;
    RenderSpaceTransform& render_space;
    std::vector<bool>& profile_visibility;
    std::vector<bool>& individual_visibility;
    CounterfactualPublicationGate& experiment_gate;
    CounterfactualExperimentWorker& experiment_worker;
    CounterfactualWorkerPublication& experiment_publication;
    CounterfactualPresentationCache& presentation;
    SimulationTimingTracker& timing;
};

// The only production commit path for scene identity, controller state,
// visibility, worker generation, publication/cache state, and simulation
// timing. Preparation failures never reach this function. Once the controller
// replacement succeeds, the remaining operations are non-throwing under the
// application's single UI-thread ownership contract.
void commit_application_scenario_transaction(
    PreparedApplicationScenarioTransaction&& prepared,
    ApplicationScenarioTransactionTargets targets);

}  // namespace nbody::visualization
