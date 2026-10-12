#include "nbody/visualization/counterfactual_worker.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace nbody::visualization {
namespace {

constexpr std::size_t maximum_worker_error_bytes = 512U;
constexpr std::size_t conservative_bytes_per_body_sample = 256U;
constexpr std::size_t conservative_sample_overhead_bytes = 64U;

static_assert(std::is_nothrow_move_assignable_v<CounterfactualWorkerPublication>,
    "worker publication must be atomically movable under its lock");

std::size_t checked_add(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::invalid_argument("counterfactual resource estimate overflows size_t");
    }
    return left + right;
}

std::size_t checked_multiply(std::size_t left, std::size_t right) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::invalid_argument("counterfactual resource estimate overflows size_t");
    }
    return left * right;
}

std::size_t integral_steps(double duration_s, double timestep_s, const char* label) {
    if (!(duration_s > 0.0) || !std::isfinite(duration_s) ||
        !(timestep_s > 0.0) || !std::isfinite(timestep_s)) {
        throw std::invalid_argument(std::string(label) + " must be finite and positive");
    }
    const double raw = duration_s / timestep_s;
    if (!std::isfinite(raw) || raw < 1.0 ||
        raw > static_cast<double>(experiments::maximum_experiment_step_count)) {
        throw std::invalid_argument(std::string(label) + " exceeds the experiment step limit");
    }
    const double rounded = std::round(raw);
    const double tolerance = std::max(
        1.0e-12,
        std::min(1.0e-7, 16.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, std::abs(raw))));
    if (std::abs(raw - rounded) > tolerance) {
        throw std::invalid_argument(std::string(label) +
            " must be an integral multiple of the fixed timestep");
    }
    return static_cast<std::size_t>(rounded);
}

std::string bounded_error(const char* message) {
    const char* source = message == nullptr ? "unknown worker failure" : message;
    std::size_t length = 0U;
    while (length < maximum_worker_error_bytes && source[length] != '\0') {
        ++length;
    }
    return std::string(source, length);
}

CounterfactualWorkerState terminal_state(experiments::ExperimentStatus status) {
    switch (status) {
        case experiments::ExperimentStatus::completed:
            return CounterfactualWorkerState::completed;
        case experiments::ExperimentStatus::cancelled:
            return CounterfactualWorkerState::cancelled;
        case experiments::ExperimentStatus::failed:
            return CounterfactualWorkerState::failed;
        default:
            return CounterfactualWorkerState::running;
    }
}

}  // namespace

struct CounterfactualExperimentWorker::Job {
    Job(
        std::uint64_t generation_value,
        experiments::ExperimentSnapshot snapshot_value,
        experiments::CounterfactualExperimentConfiguration configuration_value,
        CounterfactualJobContext context,
        const CounterfactualResourceEstimate& estimate)
        : generation(generation_value),
          snapshot(std::move(snapshot_value)),
          configuration(std::move(configuration_value)),
          scenario_id(snapshot.scenario_id()),
          identities(std::make_shared<const std::vector<
              experiments::ExperimentBodyIdentity>>(snapshot.identities())) {
        CounterfactualRunDescriptor value;
        value.scenario_generation = context.scenario_generation;
        value.worker_generation = generation;
        value.snapshot_source = context.snapshot_source;
        value.snapshot_time_s = snapshot.source_state().time_s;
        value.source_state_provenance = snapshot.source_state_provenance();
        value.coordinate_frame = snapshot.coordinate_frame();
        value.scenario_id = snapshot.scenario_id();
        value.removed_body_stable_id = configuration.removed_body_stable_id;
        value.backend = configuration.backend;
        value.openmp_threads = configuration.openmp_threads;
        value.cuda_device = configuration.cuda_device;
        value.physical_timestep_s = snapshot.fixed_timestep_s();
        value.observation_horizon_s = configuration.limits.observation_horizon_s;
        value.accepted_step_count = estimate.integration_steps;
        value.sample_cadence_s = configuration.limits.sample_cadence_s;
        value.monitored_body_stable_ids = configuration.monitored_body_stable_ids;
        if (value.monitored_body_stable_ids.empty()) {
            for (const auto& identity : snapshot.identities()) {
                if (identity.stable_id != configuration.removed_body_stable_id) {
                    value.monitored_body_stable_ids.push_back(identity.stable_id);
                }
            }
        }
        value.retained_reference_body_stable_ids =
            configuration.retained_reference_body_stable_ids;
        value.absolute_separation_threshold_m =
            configuration.significance.absolute_separation_threshold_m;
        value.normalized_deviation_threshold =
            configuration.significance.normalized_deviation_threshold;
        value.persistence_samples = configuration.significance.persistence_samples;
        value.estimated_direct_force_interactions =
            estimate.direct_force_interactions;
        value.estimated_retained_bytes = estimate.conservative_retained_bytes;
        run = std::make_shared<const CounterfactualRunDescriptor>(std::move(value));
    }

    std::uint64_t generation;
    experiments::ExperimentSnapshot snapshot;
    experiments::CounterfactualExperimentConfiguration configuration;
    std::string scenario_id;
    std::shared_ptr<const std::vector<experiments::ExperimentBodyIdentity>> identities;
    std::shared_ptr<const CounterfactualRunDescriptor> run;
};

std::uint64_t checked_next_counterfactual_generation(std::uint64_t current) {
    if (current == 0U || current == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("counterfactual generation cannot be advanced");
    }
    return current + 1U;
}

CounterfactualPublicationGate::CounterfactualPublicationGate(
    std::uint64_t initial_scenario_generation)
    : scenario_generation_(initial_scenario_generation) {
    if (scenario_generation_ == 0U) {
        throw std::invalid_argument("scenario generation must be nonzero");
    }
}

std::uint64_t CounterfactualPublicationGate::scenario_generation() const noexcept {
    return scenario_generation_;
}

std::optional<std::uint64_t>
CounterfactualPublicationGate::accepted_worker_generation() const noexcept {
    return accepted_worker_generation_;
}

void CounterfactualPublicationGate::replace_scenario() {
    scenario_generation_ = checked_next_counterfactual_generation(
        scenario_generation_);
    accepted_worker_generation_.reset();
}

void CounterfactualPublicationGate::accept_job(
    std::uint64_t worker_generation,
    std::uint64_t submitted_scenario_generation) {
    if (worker_generation == 0U ||
        submitted_scenario_generation != scenario_generation_) {
        throw std::invalid_argument(
            "counterfactual job identity does not match the active scenario generation");
    }
    accepted_worker_generation_ = worker_generation;
}

void CounterfactualPublicationGate::invalidate_job() noexcept {
    accepted_worker_generation_.reset();
}

bool CounterfactualPublicationGate::accepts(
    const CounterfactualWorkerPublication& publication) const noexcept {
    return accepted_worker_generation_ && publication.run &&
        publication.generation == *accepted_worker_generation_ &&
        publication.run->worker_generation == publication.generation &&
        publication.run->scenario_generation == scenario_generation_;
}

bool CounterfactualWorkerPublication::terminal() const noexcept {
    return state == CounterfactualWorkerState::completed ||
        state == CounterfactualWorkerState::cancelled ||
        state == CounterfactualWorkerState::failed;
}

experiments::ExperimentSnapshot capture_counterfactual_snapshot(
    const ScenarioDefinition& scenario,
    const SimulationState& state) {
    if (scenario.kind == ScenarioKind::solar_system) {
        return experiments::capture_legacy_solar_system_snapshot(state);
    }
    if (scenario.kind == ScenarioKind::astronomical_catalog) {
        return experiments::capture_astronomical_catalog_snapshot(state, scenario.stable_ids);
    }
    throw std::invalid_argument(
        "counterfactual experiments require an astronomical scenario");
}

std::vector<std::string> counterfactual_removal_candidates(
    const ScenarioDefinition& scenario) {
    std::vector<std::string> result;
    for (const auto& metadata : scenario.body_metadata) {
        if (metadata.counterfactual_removal_eligible) {
            result.push_back(metadata.stable_id);
        }
    }
    return result;
}

CounterfactualResourceEstimate estimate_counterfactual_resources(
    const experiments::ExperimentSnapshot& snapshot,
    const experiments::CounterfactualExperimentConfiguration& configuration) {
    CounterfactualResourceEstimate result;
    result.integration_steps = integral_steps(
        configuration.limits.observation_horizon_s, snapshot.fixed_timestep_s(),
        "observation horizon");
    const std::size_t sample_steps = integral_steps(
        configuration.limits.sample_cadence_s, snapshot.fixed_timestep_s(),
        "sample cadence");
    result.retained_samples = result.integration_steps / sample_steps;
    if (result.integration_steps % sample_steps != 0U) {
        result.retained_samples = checked_add(result.retained_samples, 1U);
    }
    result.retained_samples = checked_add(result.retained_samples, 1U);
    result.monitored_bodies = configuration.monitored_body_stable_ids.empty()
        ? snapshot.source_state().bodies.size() - 1U
        : configuration.monitored_body_stable_ids.size();
    result.retained_reference_bodies =
        configuration.retained_reference_body_stable_ids.size();
    result.direct_force_interactions =
        experiments::estimate_counterfactual_direct_force_workload(
            snapshot.source_state().bodies.size(),
            snapshot.source_state().bodies.size() - 1U,
            result.integration_steps);
    const std::size_t retained_bodies = checked_add(
        result.monitored_bodies, result.retained_reference_bodies);
    const std::size_t per_sample = checked_add(
        checked_multiply(retained_bodies, conservative_bytes_per_body_sample),
        conservative_sample_overhead_bytes);
    result.conservative_retained_bytes = checked_multiply(
        checked_multiply(result.retained_samples, per_sample), 4U);
    return result;
}

CounterfactualControlLayout counterfactual_control_layout(float available_width) {
    if (!(available_width > 0.0F) || !std::isfinite(available_width)) {
        throw std::invalid_argument("counterfactual control width must be finite and positive");
    }
    constexpr float stacked_threshold = 430.0F;
    constexpr float column_gap = 12.0F;
    if (available_width < stacked_threshold) {
        return {true, available_width, available_width};
    }
    const float label_width = std::clamp(available_width * 0.44F, 140.0F, 210.0F);
    return {false, label_width,
        std::max(1.0F, available_width - label_width - column_gap)};
}

std::vector<CounterfactualResultRow> build_counterfactual_result_rows(
    const experiments::CounterfactualExperimentResult& result,
    const std::vector<CounterfactualBodyLabel>& labels) {
    std::unordered_map<std::string, std::string> names;
    names.reserve(labels.size());
    for (const auto& label : labels) {
        if (label.stable_id.empty() || label.display_name.empty() ||
            !names.emplace(label.stable_id, label.display_name).second) {
            throw std::invalid_argument(
                "counterfactual presentation labels must be unique and non-empty");
        }
    }
    std::unordered_set<std::string> seen;
    std::vector<CounterfactualResultRow> rows;
    rows.reserve(result.summaries.size());
    for (const auto& summary : result.summaries) {
        const auto name = names.find(summary.stable_id);
        if (name == names.end()) {
            throw std::logic_error(
                "counterfactual summary stable ID is absent from presentation labels");
        }
        if (!seen.insert(summary.stable_id).second) {
            throw std::logic_error(
                "counterfactual summaries contain a duplicate stable ID");
        }
        rows.push_back({
            summary.stable_id,
            name->second,
            summary.final_absolute_separation_m,
            summary.maximum_absolute_separation_m,
            summary.fixed_normalization_scale_m,
            summary.final_normalized_deviation,
            summary.final_velocity_difference_m_s,
            summary.final_parent_relative_separation_m,
            summary.first_threshold_crossing_time_s,
            summary.persistent_threshold_time_s,
            summary.significant,
        });
    }
    sort_counterfactual_result_rows(
        rows, CounterfactualResultSortKey::display_name, true);
    return rows;
}

namespace {

template <typename Value>
int compare_values(const Value& left, const Value& right) {
    if (left < right) {
        return -1;
    }
    if (right < left) {
        return 1;
    }
    return 0;
}

int compare_optional_values(
    const std::optional<double>& left,
    const std::optional<double>& right) {
    if (left.has_value() != right.has_value()) {
        return left ? -1 : 1;
    }
    return left ? compare_values(*left, *right) : 0;
}

bool optional_sort_key(CounterfactualResultSortKey key) {
    return key == CounterfactualResultSortKey::parent_relative_separation ||
        key == CounterfactualResultSortKey::first_threshold_crossing ||
        key == CounterfactualResultSortKey::persistence_time;
}

int compare_result_rows(
    const CounterfactualResultRow& left,
    const CounterfactualResultRow& right,
    CounterfactualResultSortKey key) {
    switch (key) {
        case CounterfactualResultSortKey::display_name:
            return compare_values(left.display_name, right.display_name);
        case CounterfactualResultSortKey::stable_id:
            return compare_values(left.stable_id, right.stable_id);
        case CounterfactualResultSortKey::final_absolute_separation:
            return compare_values(
                left.final_absolute_separation_m, right.final_absolute_separation_m);
        case CounterfactualResultSortKey::maximum_absolute_separation:
            return compare_values(
                left.maximum_absolute_separation_m, right.maximum_absolute_separation_m);
        case CounterfactualResultSortKey::normalization_scale:
            return compare_values(
                left.fixed_normalization_scale_m, right.fixed_normalization_scale_m);
        case CounterfactualResultSortKey::normalized_deviation:
            return compare_values(
                left.final_normalized_deviation, right.final_normalized_deviation);
        case CounterfactualResultSortKey::velocity_difference:
            return compare_values(
                left.final_velocity_difference_m_s, right.final_velocity_difference_m_s);
        case CounterfactualResultSortKey::parent_relative_separation:
            return compare_optional_values(
                left.final_parent_relative_separation_m,
                right.final_parent_relative_separation_m);
        case CounterfactualResultSortKey::first_threshold_crossing:
            return compare_optional_values(
                left.first_threshold_crossing_time_s,
                right.first_threshold_crossing_time_s);
        case CounterfactualResultSortKey::persistence_time:
            return compare_optional_values(
                left.persistent_threshold_time_s,
                right.persistent_threshold_time_s);
        case CounterfactualResultSortKey::significance:
            return compare_values(left.significant, right.significant);
    }
    return 0;
}

}  // namespace

void sort_counterfactual_result_rows(
    std::vector<CounterfactualResultRow>& rows,
    CounterfactualResultSortKey key,
    bool ascending) {
    std::stable_sort(rows.begin(), rows.end(), [&](const auto& left, const auto& right) {
        int comparison = compare_result_rows(left, right, key);
        if (comparison != 0) {
            if (optional_sort_key(key) &&
                ((key == CounterfactualResultSortKey::parent_relative_separation &&
                     left.final_parent_relative_separation_m.has_value() !=
                         right.final_parent_relative_separation_m.has_value()) ||
                 (key == CounterfactualResultSortKey::first_threshold_crossing &&
                     left.first_threshold_crossing_time_s.has_value() !=
                         right.first_threshold_crossing_time_s.has_value()) ||
                 (key == CounterfactualResultSortKey::persistence_time &&
                     left.persistent_threshold_time_s.has_value() !=
                         right.persistent_threshold_time_s.has_value()))) {
                return comparison < 0;
            }
            return ascending ? comparison < 0 : comparison > 0;
        }
        return left.stable_id < right.stable_id;
    });
}

std::vector<CounterfactualWarningGroup> group_counterfactual_warnings(
    const experiments::CounterfactualExperimentResult& result) {
    std::vector<CounterfactualWarningGroup> groups;
    groups.reserve(result.issues.size());
    for (std::size_t index = 0; index < result.issues.size(); ++index) {
        const auto& issue = result.issues[index];
        const auto found = std::find_if(groups.begin(), groups.end(), [&](const auto& group) {
            return group.kind == issue.kind && group.branch == issue.branch;
        });
        if (found == groups.end()) {
            groups.push_back({issue.kind, issue.branch, {index}});
        } else {
            found->retained_issue_indices.push_back(index);
        }
    }
    return groups;
}

CounterfactualFrameStatus counterfactual_frame_status(
    const CounterfactualWorkerPublication& publication,
    const std::optional<std::string>& common_baseline_origin_stable_id) {
    if (!publication.result) {
        return {false, "Unavailable",
            "No committed paired result is available for trajectory comparison."};
    }
    const auto& result = *publication.result;
    if (!common_baseline_origin_stable_id) {
        if (result.samples.size() < 2U) {
            return {false, "SSB-centered inertial frame",
                "Trajectory comparison requires at least two matched-time retained samples."};
        }
        return {true, "SSB-centered inertial frame",
            "Both branches use the same stored SSB-centered inertial coordinates."};
    }
    const auto& origin = *common_baseline_origin_stable_id;
    const std::string label = "Common baseline-relative origin: " + origin;
    if (origin == result.removed_body_stable_id) {
        return {false, label,
            "The requested local origin was removed. Switch to the SSB/inertial frame or select a surviving common origin."};
    }
    const auto monitored = std::find(
        result.monitored_body_stable_ids.begin(),
        result.monitored_body_stable_ids.end(), origin);
    const auto reference = std::find(
        result.reference_body_stable_ids.begin(),
        result.reference_body_stable_ids.end(), origin);
    if (monitored == result.monitored_body_stable_ids.end() &&
        reference == result.reference_body_stable_ids.end()) {
        return {false, label,
            "The requested local origin has no retained historical reference samples. Switch to the SSB/inertial frame or rerun with that surviving origin retained."};
    }
    if (result.samples.size() < 2U) {
        return {false, label,
            "Trajectory comparison requires at least two matched-time retained samples."};
    }
    const std::size_t index = monitored != result.monitored_body_stable_ids.end()
        ? static_cast<std::size_t>(std::distance(
              result.monitored_body_stable_ids.begin(), monitored))
        : static_cast<std::size_t>(std::distance(
              result.reference_body_stable_ids.begin(), reference));
    const bool monitored_origin = monitored != result.monitored_body_stable_ids.end();
    const bool incomplete = std::any_of(result.samples.begin(), result.samples.end(),
        [&](const auto& sample) {
            return monitored_origin ? index >= sample.bodies.size()
                                    : index >= sample.baseline_reference_positions_m.size();
        });
    if (incomplete) {
        return {false, label,
            "Matched-time samples for the requested local origin are incomplete. Switch to the SSB/inertial frame or choose a retained surviving origin."};
    }
    return {true, label,
        "Both branches subtract the same matched-time baseline origin; the counterfactual branch is never independently recentered."};
}

const char* counterfactual_issue_kind_name(
    experiments::ExperimentIssueKind kind) noexcept {
    switch (kind) {
        case experiments::ExperimentIssueKind::insufficient_timestep_convergence:
            return "Timestep convergence not established";
        case experiments::ExperimentIssueKind::dangerous_close_approach:
            return "Dangerous close approach";
        case experiments::ExperimentIssueKind::physical_radius_overlap:
            return "Physical-radius overlap";
        case experiments::ExperimentIssueKind::excessive_conservation_drift:
            return "Conservation drift exceeded guard";
        case experiments::ExperimentIssueKind::invalid_simulated_time:
            return "Invalid simulated time";
        case experiments::ExperimentIssueKind::non_finite_state:
            return "Non-finite state";
        case experiments::ExperimentIssueKind::coincident_bodies:
            return "Coincident bodies";
        case experiments::ExperimentIssueKind::integration_failure:
            return "Integration failure";
    }
    return "Unknown issue";
}

const char* counterfactual_issue_branch_name(
    const std::optional<experiments::ExperimentBranch>& branch) noexcept {
    if (!branch) {
        return "Paired/common";
    }
    return *branch == experiments::ExperimentBranch::baseline
        ? "Baseline" : "Counterfactual";
}

std::string counterfactual_optional_measurement(
    const std::optional<double>& value,
    const char* unit) {
    if (!value) {
        return "N/A";
    }
    std::ostringstream text;
    text << std::scientific << std::setprecision(6) << *value;
    if (unit != nullptr && unit[0] != '\0') {
        text << ' ' << unit;
    }
    return text.str();
}

CounterfactualRenderPathResult build_counterfactual_render_paths(
    const CounterfactualWorkerPublication& publication,
    const RenderSpaceTransform& transform,
    const std::vector<std::string>& displayed_stable_ids,
    const std::optional<std::string>& common_baseline_origin_stable_id,
    std::size_t maximum_displayed_bodies) {
    CounterfactualRenderPathResult output;
    if (maximum_displayed_bodies == 0U) {
        return output;
    }
    const auto frame = counterfactual_frame_status(
        publication, common_baseline_origin_stable_id);
    if (!frame.available) {
        output.frame_unavailable_reason = frame.explanation;
        return output;
    }
    const auto& result = *publication.result;
    std::unordered_map<std::string, std::size_t> monitored;
    for (std::size_t index = 0; index < result.monitored_body_stable_ids.size(); ++index) {
        monitored.emplace(result.monitored_body_stable_ids[index], index);
    }
    std::optional<std::size_t> origin_index;
    std::optional<std::size_t> reference_origin_index;
    if (common_baseline_origin_stable_id) {
        const auto found = monitored.find(*common_baseline_origin_stable_id);
        if (found != monitored.end()) {
            origin_index = found->second;
        } else {
            const auto reference = std::find(
                result.reference_body_stable_ids.begin(),
                result.reference_body_stable_ids.end(),
                *common_baseline_origin_stable_id);
            if (reference != result.reference_body_stable_ids.end()) {
                reference_origin_index = static_cast<std::size_t>(std::distance(
                    result.reference_body_stable_ids.begin(), reference));
            }
        }
    }

    std::unordered_set<std::string> selected;
    for (const auto& id : displayed_stable_ids) {
        if (selected.size() >= maximum_displayed_bodies) {
            break;
        }
        if (monitored.count(id) != 0U) {
            selected.insert(id);
        }
    }
    for (const auto& id : displayed_stable_ids) {
        if (selected.count(id) == 0U) {
            continue;
        }
        const std::size_t body_index = monitored.at(id);
        CounterfactualRenderPath baseline{id, CounterfactualPathBranch::baseline, {}};
        CounterfactualRenderPath counterfactual{
            id, CounterfactualPathBranch::counterfactual, {}};
        baseline.points.reserve(result.samples.size());
        counterfactual.points.reserve(result.samples.size());
        for (const auto& sample : result.samples) {
            if (body_index >= sample.bodies.size()) {
                throw std::logic_error("counterfactual sample mapping is inconsistent");
            }
            const auto& body = sample.bodies[body_index];
            if (origin_index || reference_origin_index) {
                const Vec3 common_origin = origin_index
                    ? sample.bodies[*origin_index].baseline_position_m
                    : sample.baseline_reference_positions_m[*reference_origin_index];
                baseline.points.push_back(transform.to_render_offset(
                    body.baseline_position_m - common_origin));
                counterfactual.points.push_back(transform.to_render_offset(
                    body.counterfactual_position_m - common_origin));
            } else {
                baseline.points.push_back(transform.to_render_position(
                    body.baseline_position_m));
                counterfactual.points.push_back(transform.to_render_position(
                    body.counterfactual_position_m));
            }
        }
        output.paths.push_back(std::move(baseline));
        output.paths.push_back(std::move(counterfactual));
    }
    return output;
}

std::vector<std::string> counterfactual_visible_overlay_ids(
    const ScenarioDefinition& scenario,
    const std::vector<bool>& profile_visibility,
    const std::vector<bool>& individual_visibility,
    const BodyClassVisibility& class_visibility,
    const std::vector<std::string>& requested_stable_ids,
    std::size_t maximum_displayed_bodies) {
    std::vector<std::string> result;
    result.reserve(std::min(maximum_displayed_bodies, requested_stable_ids.size()));
    std::unordered_set<std::string> seen;
    for (const auto& id : requested_stable_ids) {
        if (result.size() >= maximum_displayed_bodies || !seen.insert(id).second) {
            continue;
        }
        const auto index = find_body_index(scenario, id);
        if (index && effective_body_visibility(
                scenario, *index, profile_visibility, individual_visibility,
                class_visibility)) {
            result.push_back(id);
        }
    }
    return result;
}

CounterfactualExperimentWorker::CounterfactualExperimentWorker()
    : thread_(&CounterfactualExperimentWorker::worker_loop, this) {}

CounterfactualExperimentWorker::~CounterfactualExperimentWorker() {
    cancellation_requested_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
}

std::uint64_t CounterfactualExperimentWorker::start(
    experiments::ExperimentSnapshot snapshot,
    experiments::CounterfactualExperimentConfiguration configuration,
    CounterfactualJobContext context) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
        throw std::logic_error("counterfactual worker is shutting down");
    }
    if (active_ || pending_job_) {
        throw std::logic_error("a counterfactual experiment is already active");
    }
    if (context.scenario_generation == 0U) {
        throw std::invalid_argument("scenario generation must be nonzero");
    }
    const std::uint64_t generation = next_generation_;
    const std::uint64_t following_generation =
        checked_next_counterfactual_generation(next_generation_);
    const auto estimate = estimate_counterfactual_resources(snapshot, configuration);
    auto next_job = std::make_unique<Job>(
        generation, std::move(snapshot), std::move(configuration), context, estimate);
    CounterfactualWorkerPublication next_publication;
    next_publication.generation = generation;
    next_publication.state = CounterfactualWorkerState::pending;
    next_publication.scenario_id = next_job->scenario_id;
    next_publication.removed_body_stable_id =
        next_job->configuration.removed_body_stable_id;
    next_publication.identities = next_job->identities;
    next_publication.run = next_job->run;
    pending_job_ = std::move(next_job);
    publication_ = std::move(next_publication);
    cancellation_requested_.store(false, std::memory_order_release);
    active_ = true;
    next_generation_ = following_generation;
    wake_.notify_one();
    return generation;
}

void CounterfactualExperimentWorker::request_cancel() noexcept {
    cancellation_requested_.store(true, std::memory_order_release);
}

CounterfactualWorkerPublication CounterfactualExperimentWorker::publication() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return publication_;
}

bool CounterfactualExperimentWorker::busy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
}

void CounterfactualExperimentWorker::wait_until_idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [&] { return !active_ && !pending_job_; });
}

bool CounterfactualExperimentWorker::wait_for_progress(
    std::uint64_t generation,
    std::size_t completed_steps) {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [&] {
        return publication_.generation != generation || publication_.terminal() ||
            publication_.completed_steps >= completed_steps;
    });
    return publication_.generation == generation &&
        publication_.completed_steps >= completed_steps;
}

void CounterfactualExperimentWorker::publish_progress(
    const Job& job,
    const std::shared_ptr<experiments::CounterfactualExperimentRunner>& runner,
    std::size_t total_steps,
    CounterfactualWorkerState state,
    bool include_result) {
    CounterfactualWorkerPublication next;
    next.generation = job.generation;
    next.state = state;
    next.completed_steps = runner->result().completed_steps;
    next.total_steps = total_steps;
    next.scenario_id = job.scenario_id;
    next.removed_body_stable_id = job.configuration.removed_body_stable_id;
    next.identities = job.identities;
    next.run = job.run;
    if (include_result) {
        next.result = std::shared_ptr<const experiments::CounterfactualExperimentResult>(
            runner, &runner->result());
    }
    std::lock_guard<std::mutex> lock(mutex_);
    publication_ = std::move(next);
    idle_.notify_all();
}

void CounterfactualExperimentWorker::publish_worker_failure(
    const Job& job,
    const std::shared_ptr<experiments::CounterfactualExperimentRunner>& runner,
    std::size_t total_steps,
    const char* message) noexcept {
    try {
        CounterfactualWorkerPublication next;
        next.generation = job.generation;
        next.state = CounterfactualWorkerState::failed;
        next.total_steps = total_steps;
        next.scenario_id = job.scenario_id;
        next.removed_body_stable_id = job.configuration.removed_body_stable_id;
        next.identities = job.identities;
        next.run = job.run;
        if (runner) {
            next.completed_steps = runner->result().completed_steps;
            next.result = std::shared_ptr<
                const experiments::CounterfactualExperimentResult>(
                    runner, &runner->result());
        }
        next.worker_error = bounded_error(message);
        std::lock_guard<std::mutex> lock(mutex_);
        publication_ = std::move(next);
        idle_.notify_all();
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (publication_.generation == job.generation) {
            publication_.state = CounterfactualWorkerState::failed;
        }
    }
}

void CounterfactualExperimentWorker::worker_loop() noexcept {
    for (;;) {
        std::unique_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stopping_ || pending_job_; });
            if (stopping_ && !pending_job_) {
                return;
            }
            job = std::move(pending_job_);
        }
        std::shared_ptr<experiments::CounterfactualExperimentRunner> runner;
        std::size_t total_steps = job->run->accepted_step_count;
        try {
            runner = std::make_shared<experiments::CounterfactualExperimentRunner>(
                std::move(job->snapshot), job->configuration);
            publish_progress(*job, runner, total_steps,
                CounterfactualWorkerState::running, false);
            for (;;) {
                if (cancellation_requested_.load(std::memory_order_acquire)) {
                    runner->request_cancel();
                }
                const bool advanced = runner->advance_batch();
                const CounterfactualWorkerState state = terminal_state(
                    runner->result().status);
                const bool terminal = state != CounterfactualWorkerState::running;
                publish_progress(*job, runner, total_steps, state, terminal);
                if (terminal || !advanced) {
                    break;
                }
            }
        } catch (const std::exception& error) {
            publish_worker_failure(*job, runner, total_steps, error.what());
        } catch (...) {
            publish_worker_failure(
                *job, runner, total_steps, "non-standard worker exception");
        }
        bool should_stop = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_ = false;
            should_stop = stopping_;
        }
        idle_.notify_all();
        if (should_stop) {
            return;
        }
    }
}

const char* counterfactual_worker_state_name(
    CounterfactualWorkerState state) noexcept {
    switch (state) {
        case CounterfactualWorkerState::idle: return "Idle";
        case CounterfactualWorkerState::pending: return "Pending";
        case CounterfactualWorkerState::running: return "Running";
        case CounterfactualWorkerState::completed: return "Completed";
        case CounterfactualWorkerState::cancelled: return "Cancelled";
        case CounterfactualWorkerState::failed: return "Failed";
    }
    return "Unknown";
}

const char* counterfactual_snapshot_source_name(
    CounterfactualSnapshotSource source) noexcept {
    switch (source) {
        case CounterfactualSnapshotSource::current_state: return "Current simulation state";
        case CounterfactualSnapshotSource::initial_state: return "Initial scenario state";
    }
    return "Unknown";
}

}  // namespace nbody::visualization
