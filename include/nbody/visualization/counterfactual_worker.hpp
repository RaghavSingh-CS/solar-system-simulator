#pragma once

#include "nbody/experiments/counterfactual.hpp"
#include "nbody/visualization/render_space.hpp"
#include "nbody/visualization/presentation.hpp"
#include "nbody/visualization/scenario.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace nbody::visualization {

enum class CounterfactualWorkerState {
    idle,
    pending,
    running,
    completed,
    cancelled,
    failed,
};

struct CounterfactualResourceEstimate {
    std::size_t integration_steps{0};
    std::size_t retained_samples{0};
    std::size_t monitored_bodies{0};
    std::size_t retained_reference_bodies{0};
    std::uint64_t direct_force_interactions{0};
    std::size_t conservative_retained_bytes{0};
};

enum class CounterfactualSnapshotSource {
    current_state,
    initial_state,
};

struct CounterfactualJobContext {
    std::uint64_t scenario_generation{0};
    CounterfactualSnapshotSource snapshot_source{
        CounterfactualSnapshotSource::current_state};
};

struct CounterfactualRunDescriptor {
    std::uint64_t scenario_generation{0};
    std::uint64_t worker_generation{0};
    CounterfactualSnapshotSource snapshot_source{
        CounterfactualSnapshotSource::current_state};
    double snapshot_time_s{0.0};
    std::string source_state_provenance;
    std::string coordinate_frame;
    std::string scenario_id;
    std::string removed_body_stable_id;
    experiments::ExperimentBackend backend{
        experiments::ExperimentBackend::sequential};
    int openmp_threads{1};
    int cuda_device{0};
    double physical_timestep_s{0.0};
    double observation_horizon_s{0.0};
    std::size_t accepted_step_count{0};
    double sample_cadence_s{0.0};
    std::vector<std::string> monitored_body_stable_ids;
    std::vector<std::string> retained_reference_body_stable_ids;
    double absolute_separation_threshold_m{0.0};
    double normalized_deviation_threshold{0.0};
    std::size_t persistence_samples{0};
    std::uint64_t estimated_direct_force_interactions{0};
    std::size_t estimated_retained_bytes{0};
};

struct CounterfactualWorkerPublication {
    std::uint64_t generation{0};
    CounterfactualWorkerState state{CounterfactualWorkerState::idle};
    std::size_t completed_steps{0};
    std::size_t total_steps{0};
    std::string scenario_id;
    std::string removed_body_stable_id;
    std::shared_ptr<const CounterfactualRunDescriptor> run;
    std::shared_ptr<const std::vector<experiments::ExperimentBodyIdentity>> identities;
    std::shared_ptr<const experiments::CounterfactualExperimentResult> result;
    std::string worker_error;

    bool terminal() const noexcept;
};

enum class CounterfactualPathBranch {
    baseline,
    counterfactual,
};

struct CounterfactualRenderPath {
    std::string stable_id;
    CounterfactualPathBranch branch{CounterfactualPathBranch::baseline};
    std::vector<glm::vec3> points;
};

struct CounterfactualRenderPathResult {
    std::vector<CounterfactualRenderPath> paths;
    std::string frame_unavailable_reason;
};

struct CounterfactualControlLayout {
    bool stacked{false};
    float label_width{0.0F};
    float value_width{0.0F};
};

struct CounterfactualBodyLabel {
    std::string stable_id;
    std::string display_name;
};

struct CounterfactualResultRow {
    std::string stable_id;
    std::string display_name;
    double final_absolute_separation_m{0.0};
    double maximum_absolute_separation_m{0.0};
    double fixed_normalization_scale_m{0.0};
    double final_normalized_deviation{0.0};
    double final_velocity_difference_m_s{0.0};
    std::optional<double> final_parent_relative_separation_m;
    std::optional<double> first_threshold_crossing_time_s;
    std::optional<double> persistent_threshold_time_s;
    bool significant{false};
};

enum class CounterfactualResultSortKey {
    display_name,
    stable_id,
    final_absolute_separation,
    maximum_absolute_separation,
    normalization_scale,
    normalized_deviation,
    velocity_difference,
    parent_relative_separation,
    first_threshold_crossing,
    persistence_time,
    significance,
};

struct CounterfactualWarningGroup {
    experiments::ExperimentIssueKind kind{
        experiments::ExperimentIssueKind::integration_failure};
    std::optional<experiments::ExperimentBranch> branch;
    std::vector<std::size_t> retained_issue_indices;
};

struct CounterfactualFrameStatus {
    bool available{false};
    std::string coordinate_frame_label;
    std::string explanation;
};

CounterfactualControlLayout counterfactual_control_layout(float available_width);

std::vector<CounterfactualResultRow> build_counterfactual_result_rows(
    const experiments::CounterfactualExperimentResult& result,
    const std::vector<CounterfactualBodyLabel>& labels);

void sort_counterfactual_result_rows(
    std::vector<CounterfactualResultRow>& rows,
    CounterfactualResultSortKey key,
    bool ascending);

std::vector<CounterfactualWarningGroup> group_counterfactual_warnings(
    const experiments::CounterfactualExperimentResult& result);

CounterfactualFrameStatus counterfactual_frame_status(
    const CounterfactualWorkerPublication& publication,
    const std::optional<std::string>& common_baseline_origin_stable_id = std::nullopt);

const char* counterfactual_issue_kind_name(
    experiments::ExperimentIssueKind kind) noexcept;
const char* counterfactual_issue_branch_name(
    const std::optional<experiments::ExperimentBranch>& branch) noexcept;
std::string counterfactual_optional_measurement(
    const std::optional<double>& value,
    const char* unit);

std::uint64_t checked_next_counterfactual_generation(std::uint64_t current);

class CounterfactualPublicationGate {
public:
    explicit CounterfactualPublicationGate(
        std::uint64_t initial_scenario_generation = 1U);

    std::uint64_t scenario_generation() const noexcept;
    std::optional<std::uint64_t> accepted_worker_generation() const noexcept;
    void replace_scenario();
    void accept_job(
        std::uint64_t worker_generation,
        std::uint64_t submitted_scenario_generation);
    void invalidate_job() noexcept;
    bool accepts(const CounterfactualWorkerPublication& publication) const noexcept;

private:
    std::uint64_t scenario_generation_{1U};
    std::optional<std::uint64_t> accepted_worker_generation_;
};

experiments::ExperimentSnapshot capture_counterfactual_snapshot(
    const ScenarioDefinition& scenario,
    const SimulationState& state);

std::vector<std::string> counterfactual_removal_candidates(
    const ScenarioDefinition& scenario);

CounterfactualResourceEstimate estimate_counterfactual_resources(
    const experiments::ExperimentSnapshot& snapshot,
    const experiments::CounterfactualExperimentConfiguration& configuration);

CounterfactualRenderPathResult build_counterfactual_render_paths(
    const CounterfactualWorkerPublication& publication,
    const RenderSpaceTransform& transform,
    const std::vector<std::string>& displayed_stable_ids,
    const std::optional<std::string>& common_baseline_origin_stable_id = std::nullopt,
    std::size_t maximum_displayed_bodies = 12U);

std::vector<std::string> counterfactual_visible_overlay_ids(
    const ScenarioDefinition& scenario,
    const std::vector<bool>& profile_visibility,
    const std::vector<bool>& individual_visibility,
    const BodyClassVisibility& class_visibility,
    const std::vector<std::string>& requested_stable_ids,
    std::size_t maximum_displayed_bodies = 12U);

class CounterfactualExperimentWorker {
public:
    CounterfactualExperimentWorker();
    ~CounterfactualExperimentWorker();
    CounterfactualExperimentWorker(const CounterfactualExperimentWorker&) = delete;
    CounterfactualExperimentWorker& operator=(const CounterfactualExperimentWorker&) = delete;
    CounterfactualExperimentWorker(CounterfactualExperimentWorker&&) = delete;
    CounterfactualExperimentWorker& operator=(CounterfactualExperimentWorker&&) = delete;

    std::uint64_t start(
        experiments::ExperimentSnapshot snapshot,
        experiments::CounterfactualExperimentConfiguration configuration,
        CounterfactualJobContext context = {1U,
            CounterfactualSnapshotSource::current_state});
    void request_cancel() noexcept;
    CounterfactualWorkerPublication publication() const;
    bool busy() const;

    // Primarily useful for deterministic tests and orderly shutdown. The GUI
    // does not wait during normal frame processing.
    void wait_until_idle();
    bool wait_for_progress(std::uint64_t generation, std::size_t completed_steps);

private:
    struct Job;

    void worker_loop() noexcept;
    void publish_progress(
        const Job& job,
        const std::shared_ptr<experiments::CounterfactualExperimentRunner>& runner,
        std::size_t total_steps,
        CounterfactualWorkerState state,
        bool include_result);
    void publish_worker_failure(
        const Job& job,
        const std::shared_ptr<experiments::CounterfactualExperimentRunner>& runner,
        std::size_t total_steps,
        const char* message) noexcept;

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable idle_;
    std::unique_ptr<Job> pending_job_;
    CounterfactualWorkerPublication publication_;
    std::thread thread_;
    std::atomic<bool> cancellation_requested_{false};
    bool stopping_{false};
    bool active_{false};
    std::uint64_t next_generation_{1};
};

const char* counterfactual_worker_state_name(CounterfactualWorkerState state) noexcept;
const char* counterfactual_snapshot_source_name(
    CounterfactualSnapshotSource source) noexcept;

}  // namespace nbody::visualization
