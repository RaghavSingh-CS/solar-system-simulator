#pragma once

#include "nbody/diagnostics.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nbody::experiments {

enum class ExperimentBackend {
    sequential,
    openmp,
};

enum class ExperimentIntegrator {
    velocity_verlet,
};

enum class ExperimentStatus {
    ready,
    running,
    paused,
    completed,
    cancelled,
    failed,
};

enum class ExperimentIssueKind {
    insufficient_timestep_convergence,
    dangerous_close_approach,
    physical_radius_overlap,
    excessive_conservation_drift,
    invalid_simulated_time,
    non_finite_state,
    coincident_bodies,
    integration_failure,
};

enum class ExperimentBranch {
    baseline,
    counterfactual,
};

struct ExperimentBodyIdentity {
    std::string stable_id;
    std::optional<std::string> parent_stable_id;
    double physical_radius_m{0.0};
    bool curated_removal_candidate{false};
};

struct ExperimentSnapshotDefinition {
    std::string scenario_id;
    std::string source_state_provenance;
    std::string coordinate_frame;
    std::string central_star_stable_id;
    SimulationState source_state;
    std::vector<ExperimentBodyIdentity> identities;
    double fixed_timestep_s{0.0};
    ExperimentIntegrator integrator{ExperimentIntegrator::velocity_verlet};
    bool timestep_convergence_established{false};
    double maximum_supported_horizon_s{0.0};
};

class ExperimentSnapshot {
public:
    explicit ExperimentSnapshot(ExperimentSnapshotDefinition definition);
    ExperimentSnapshot(const ExperimentSnapshot&) = default;
    ExperimentSnapshot(ExperimentSnapshot&&) noexcept = default;
    ExperimentSnapshot& operator=(const ExperimentSnapshot&) = delete;
    ExperimentSnapshot& operator=(ExperimentSnapshot&&) = delete;

    const std::string& scenario_id() const noexcept;
    const std::string& source_state_provenance() const noexcept;
    const std::string& coordinate_frame() const noexcept;
    const std::string& central_star_stable_id() const noexcept;
    const SimulationState& source_state() const noexcept;
    const std::vector<ExperimentBodyIdentity>& identities() const noexcept;
    double fixed_timestep_s() const noexcept;
    ExperimentIntegrator integrator() const noexcept;
    bool timestep_convergence_established() const noexcept;
    double maximum_supported_horizon_s() const noexcept;

private:
    std::string scenario_id_;
    std::string source_state_provenance_;
    std::string coordinate_frame_;
    std::string central_star_stable_id_;
    SimulationState source_state_;
    std::vector<ExperimentBodyIdentity> identities_;
    double fixed_timestep_s_{0.0};
    ExperimentIntegrator integrator_{ExperimentIntegrator::velocity_verlet};
    bool timestep_convergence_established_{false};
    double maximum_supported_horizon_s_{0.0};
};

// Capture helpers copy the supplied live state. They never retain a reference
// to or mutate viewer/controller state.
ExperimentSnapshot capture_legacy_solar_system_snapshot(
    const SimulationState& source_state);

ExperimentSnapshot capture_astronomical_catalog_snapshot(
    const SimulationState& source_state,
    const std::vector<std::string>& included_stable_ids = {});

struct SurvivorMapping {
    std::string stable_id;
    std::size_t baseline_index{0};
    std::size_t counterfactual_index{0};
    std::optional<std::size_t> baseline_parent_index;
    std::optional<std::size_t> counterfactual_parent_index;
};

struct ExperimentLimits {
    double observation_horizon_s{0.0};
    double sample_cadence_s{0.0};
    std::size_t integration_steps_per_batch{1};
    std::size_t maximum_monitored_bodies{64};
    std::size_t maximum_reference_bodies{4};
    std::size_t maximum_stored_samples{4096};
    std::size_t memory_budget_bytes{64U * 1024U * 1024U};
    std::uint64_t workload_budget_interactions{500000000ULL};
    std::size_t maximum_issue_records{256};
    std::size_t maximum_issue_message_bytes{64U * 1024U};
};

inline constexpr std::uint64_t maximum_exact_integer_step_count =
    1ULL << 53U;
inline constexpr std::uint64_t maximum_experiment_step_count =
    100000000ULL;

// Checked direct-force estimate used by admission policy and exposed for
// operator tooling. Throws if any intermediate cannot be represented.
std::uint64_t estimate_counterfactual_direct_force_workload(
    std::size_t baseline_body_count,
    std::size_t counterfactual_body_count,
    std::size_t step_count);

// Minimum separation of the linearly interpolated relative path over [0, 1].
// This is a screening primitive, not a curved-trajectory guarantee.
double linear_swept_minimum_distance(
    const Vec3& left_before,
    const Vec3& right_before,
    const Vec3& left_after,
    const Vec3& right_after);

struct SignificancePolicy {
    double absolute_separation_threshold_m{1.0};
    double normalized_deviation_threshold{0.0};
    std::size_t persistence_samples{1};
    double normalization_denominator_floor_m{1.0};
};

struct ScientificGuardPolicy {
    double dangerous_close_approach_radius_multiple{3.0};
    double maximum_relative_energy_drift{1.0e-6};
};

struct CounterfactualExperimentConfiguration {
    std::string removed_body_stable_id;
    ExperimentBackend backend{ExperimentBackend::sequential};
    int openmp_threads{1};
    ExperimentLimits limits;
    SignificancePolicy significance;
    ScientificGuardPolicy guards;
    std::vector<std::string> monitored_body_stable_ids;
    // Baseline-only positions retained for render-frame reconstruction. These
    // IDs do not acquire deviation summaries or threshold metrics.
    std::vector<std::string> retained_reference_body_stable_ids;
};

struct ExperimentIssue {
    ExperimentIssueKind kind{ExperimentIssueKind::integration_failure};
    bool numerical_invalidity{false};
    std::string message;
    double time_s{0.0};
    std::optional<std::string> first_body_stable_id;
    std::optional<std::string> second_body_stable_id;
    std::optional<ExperimentBranch> branch;
};

inline constexpr std::size_t maximum_terminal_failure_message_bytes = 512U;

struct ExperimentTerminalFailure {
    ExperimentIssueKind kind{ExperimentIssueKind::integration_failure};
    double time_s{0.0};
    std::array<char, maximum_terminal_failure_message_bytes + 1U> message{};
    std::size_t message_length{0U};
    bool message_truncated{false};

    [[nodiscard]] std::string_view message_text() const noexcept {
        return {message.data(), message_length};
    }
};

struct BranchConservationDiagnostics {
    Diagnostics initial;
    Diagnostics current;
    double absolute_energy_drift_j{0.0};
    double relative_energy_drift{0.0};
    double linear_momentum_drift_kg_m_s{0.0};
    double angular_momentum_drift_kg_m2_s{0.0};
};

struct BodyTrajectorySample {
    Vec3 baseline_position_m;
    Vec3 counterfactual_position_m;
    Vec3 baseline_velocity_m_s;
    Vec3 counterfactual_velocity_m_s;
    double absolute_separation_m{0.0};
    std::optional<double> parent_relative_separation_m;
    double normalized_deviation{0.0};
    double velocity_difference_m_s{0.0};
};

struct ExperimentSample {
    double elapsed_time_s{0.0};
    std::vector<BodyTrajectorySample> bodies;
    std::vector<Vec3> baseline_reference_positions_m;
};

struct BodyDeviationSummary {
    std::string stable_id;
    double fixed_normalization_scale_m{0.0};
    double final_absolute_separation_m{0.0};
    double maximum_absolute_separation_m{0.0};
    double maximum_absolute_separation_time_s{0.0};
    std::optional<double> final_parent_relative_separation_m;
    std::optional<double> maximum_parent_relative_separation_m;
    std::optional<double> maximum_parent_relative_separation_time_s;
    double final_normalized_deviation{0.0};
    double maximum_normalized_deviation{0.0};
    double final_velocity_difference_m_s{0.0};
    double maximum_velocity_difference_m_s{0.0};
    std::optional<double> first_threshold_crossing_time_s;
    std::optional<double> persistent_threshold_time_s;
    bool significant{false};
    bool parent_relative_available{false};
};

struct CounterfactualExperimentResult {
    ExperimentStatus status{ExperimentStatus::ready};
    std::string removed_body_stable_id;
    SimulationState baseline_state;
    SimulationState counterfactual_state;
    std::vector<SurvivorMapping> survivor_mapping;
    std::vector<std::string> monitored_body_stable_ids;
    std::vector<std::string> reference_body_stable_ids;
    std::vector<ExperimentSample> samples;
    std::vector<BodyDeviationSummary> summaries;
    BranchConservationDiagnostics baseline_conservation;
    BranchConservationDiagnostics counterfactual_conservation;
    std::vector<ExperimentIssue> issues;
    std::size_t dropped_issue_count{0};
    // Failure metadata is not part of the committed warning/history buffer.
    std::optional<ExperimentTerminalFailure> terminal_failure;
    std::size_t completed_steps{0};
    bool escape_assessment_supported{false};
};

class CounterfactualExperimentRunner {
public:
    CounterfactualExperimentRunner(
        ExperimentSnapshot snapshot,
        CounterfactualExperimentConfiguration configuration);

    CounterfactualExperimentRunner(const CounterfactualExperimentRunner&) = delete;
    CounterfactualExperimentRunner& operator=(
        const CounterfactualExperimentRunner&) = delete;
    CounterfactualExperimentRunner(CounterfactualExperimentRunner&&) = delete;
    CounterfactualExperimentRunner& operator=(
        CounterfactualExperimentRunner&&) = delete;

    static bool openmp_supported() noexcept;

    const ExperimentSnapshot& snapshot() const noexcept;
    const CounterfactualExperimentConfiguration& configuration() const noexcept;
    // Runners are single-thread-owned. No method, including result(), pause,
    // or cancellation control, may be called concurrently with another.
    const CounterfactualExperimentResult& result() const noexcept;

    // Publishes either both completed branches and their metrics, or neither.
    bool advance_batch();
    void run_to_completion();
    void set_paused(bool paused) noexcept;
    bool paused() const noexcept;
    void request_cancel() noexcept;

private:
    ExperimentSnapshot snapshot_;
    CounterfactualExperimentConfiguration configuration_;
    CounterfactualExperimentResult result_;
    std::vector<std::size_t> monitored_mapping_indices_;
    std::vector<std::size_t> reference_mapping_indices_;
    std::vector<std::size_t> consecutive_threshold_samples_;
    bool cancellation_requested_{false};
    bool pause_requested_{false};
};

const char* experiment_backend_name(ExperimentBackend backend) noexcept;
const char* experiment_status_name(ExperimentStatus status) noexcept;

}  // namespace nbody::experiments
