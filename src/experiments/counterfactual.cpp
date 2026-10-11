#include "nbody/experiments/counterfactual.hpp"

#include "nbody/astronomy/solar_system_60.hpp"
#include "nbody/solar_system.hpp"
#ifdef NBODY_EXPERIMENT_HAS_OPENMP
#include "nbody/openmp_backend.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace nbody::experiments {
namespace {

static_assert(std::is_nothrow_move_assignable_v<CounterfactualExperimentResult>,
    "transactional result publication must not throw");
static_assert(std::is_nothrow_move_assignable_v<std::vector<std::size_t>>,
    "transactional threshold-state publication must not throw");
static_assert(std::is_nothrow_copy_assignable_v<
        std::optional<ExperimentTerminalFailure>>,
    "terminal failure reporting must not throw");

constexpr double catalog_provisional_timestep_s = 60.0;
constexpr double catalog_initial_horizon_limit_s = 86400.0;
constexpr double legacy_initial_horizon_limit_s = 365.25 * 86400.0;
constexpr std::size_t conservative_bytes_per_body_sample = 256;
constexpr std::size_t conservative_allocation_overhead_bytes = 64;

class ScientificFailure final : public std::runtime_error {
public:
    ScientificFailure(ExperimentIssueKind issue_kind, const std::string& message)
        : std::runtime_error(message), issue_kind_(issue_kind) {}

    ExperimentIssueKind issue_kind() const noexcept { return issue_kind_; }

private:
    ExperimentIssueKind issue_kind_;
};

bool finite_vector(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool same_body(const Body& left, const Body& right) {
    return left.name == right.name && left.mass_kg == right.mass_kg &&
        left.position_m.x == right.position_m.x && left.position_m.y == right.position_m.y &&
        left.position_m.z == right.position_m.z &&
        left.velocity_m_s.x == right.velocity_m_s.x &&
        left.velocity_m_s.y == right.velocity_m_s.y &&
        left.velocity_m_s.z == right.velocity_m_s.z;
}

bool checked_subtract(double minuend, double subtrahend, double& result) noexcept {
    if (!std::isfinite(minuend) || !std::isfinite(subtrahend)) {
        return false;
    }
    const double maximum = std::numeric_limits<double>::max();
    if ((subtrahend > 0.0 && minuend < -maximum + subtrahend) ||
        (subtrahend < 0.0 && minuend > maximum + subtrahend)) {
        return false;
    }
    result = minuend - subtrahend;
    return std::isfinite(result);
}

bool checked_relative_vector(
    const Vec3& left,
    const Vec3& right,
    Vec3& right_minus_left) noexcept {
    return checked_subtract(right.x, left.x, right_minus_left.x) &&
        checked_subtract(right.y, left.y, right_minus_left.y) &&
        checked_subtract(right.z, left.z, right_minus_left.z);
}

bool stable_vector_norm(const Vec3& value, double& result) noexcept {
    if (!finite_vector(value)) {
        return false;
    }
    result = std::hypot(value.x, value.y, value.z);
    return std::isfinite(result);
}

bool stable_relative_distance(
    const Vec3& left,
    const Vec3& right,
    double& result) noexcept {
    Vec3 relative;
    return checked_relative_vector(left, right, relative) &&
        stable_vector_norm(relative, result);
}

double checked_combined_radius(
    const ExperimentBodyIdentity& left,
    const ExperimentBodyIdentity& right) {
    const double maximum = std::numeric_limits<double>::max();
    if (left.physical_radius_m > maximum - right.physical_radius_m) {
        throw ScientificFailure(
            ExperimentIssueKind::non_finite_state,
            "combined physical radius is not representable");
    }
    return left.physical_radius_m + right.physical_radius_m;
}

double checked_close_threshold(
    double combined_radius,
    double radius_multiple) {
    if (combined_radius > std::numeric_limits<double>::max() / radius_multiple) {
        throw ScientificFailure(
            ExperimentIssueKind::non_finite_state,
            "close-approach threshold is not representable");
    }
    return combined_radius * radius_multiple;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::invalid_argument(std::string(label) + " overflows size_t");
    }
    return left + right;
}

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* label) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::invalid_argument(std::string(label) + " overflows size_t");
    }
    return left * right;
}

std::uint64_t checked_add_u64(
    std::uint64_t left, std::uint64_t right, const char* label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::invalid_argument(std::string(label) + " overflows uint64_t");
    }
    return left + right;
}

std::uint64_t checked_multiply_u64(
    std::uint64_t left, std::uint64_t right, const char* label) {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::invalid_argument(std::string(label) + " overflows uint64_t");
    }
    return left * right;
}

std::size_t checked_integral_steps(double duration_s, double timestep_s, const char* label) {
    if (!(duration_s > 0.0) || !std::isfinite(duration_s) ||
        !(timestep_s > 0.0) || !std::isfinite(timestep_s)) {
        throw std::invalid_argument(
            std::string(label) + " and fixed timestep must be finite and positive");
    }

    const long double raw_steps = static_cast<long double>(duration_s) /
        static_cast<long double>(timestep_s);
    if (!std::isfinite(raw_steps) || raw_steps < 1.0L ||
        raw_steps > static_cast<long double>(maximum_exact_integer_step_count) ||
        raw_steps > static_cast<long double>(maximum_experiment_step_count) ||
        raw_steps > static_cast<long double>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string(label) +
            " exceeds the exact or operational experiment step-count limit");
    }

    const long double rounded_steps = std::round(raw_steps);
    const long double scaled_roundoff = 16.0L *
        static_cast<long double>(std::numeric_limits<double>::epsilon()) *
        std::max(1.0L, std::abs(raw_steps));
    const long double tolerance = std::max(
        1.0e-12L, std::min(1.0e-7L, scaled_roundoff));
    if (std::abs(raw_steps - rounded_steps) > tolerance) {
        throw std::invalid_argument(std::string(label) +
            " must be an integral multiple of the fixed timestep");
    }
    return static_cast<std::size_t>(rounded_steps);
}

double checked_elapsed_time(std::size_t completed_steps, double timestep_s) {
    const long double elapsed = static_cast<long double>(completed_steps) *
        static_cast<long double>(timestep_s);
    if (!std::isfinite(elapsed) || elapsed < 0.0L ||
        elapsed > static_cast<long double>(std::numeric_limits<double>::max())) {
        throw ScientificFailure(
            ExperimentIssueKind::invalid_simulated_time,
            "experiment elapsed-time calculation became non-finite");
    }
    const double result = static_cast<double>(elapsed);
    if (!std::isfinite(result)) {
        throw ScientificFailure(
            ExperimentIssueKind::invalid_simulated_time,
            "experiment elapsed-time conversion became non-finite");
    }
    return result;
}

std::uint64_t checked_directed_pairs(std::size_t body_count) {
    if (body_count > std::numeric_limits<std::uint64_t>::max()) {
        throw std::invalid_argument("body count does not fit the workload counter");
    }
    const auto count = static_cast<std::uint64_t>(body_count);
    return count < 2U ? 0U : checked_multiply_u64(
        count, count - 1U, "directed pair count");
}

std::uint64_t checked_workload(
    std::size_t baseline_count,
    std::size_t counterfactual_count,
    std::size_t step_count) {
    const std::uint64_t pairs = checked_add_u64(
        checked_directed_pairs(baseline_count),
        checked_directed_pairs(counterfactual_count),
        "paired directed pair count");
    const std::uint64_t per_step = checked_multiply_u64(
        2U, pairs, "paired Velocity Verlet force evaluations");
    if (step_count > std::numeric_limits<std::uint64_t>::max()) {
        throw std::invalid_argument("step count does not fit the workload counter");
    }
    return checked_multiply_u64(
        per_step, static_cast<std::uint64_t>(step_count),
        "experiment workload estimate");
}

void validate_state(const SimulationState& state, bool reject_coincident) {
    if (!std::isfinite(state.time_s)) {
        throw std::invalid_argument("source simulation timestamp must be finite");
    }
    if (state.bodies.size() < 2U) {
        throw std::invalid_argument("an experiment snapshot requires at least two bodies");
    }
    for (const auto& body : state.bodies) {
        if (body.name.empty() || !(body.mass_kg > 0.0) || !std::isfinite(body.mass_kg) ||
            !finite_vector(body.position_m) || !finite_vector(body.velocity_m_s)) {
            throw std::invalid_argument("source state contains invalid physical parameters");
        }
    }
    if (reject_coincident) {
        for (std::size_t left = 0; left < state.bodies.size(); ++left) {
            for (std::size_t right = left + 1U; right < state.bodies.size(); ++right) {
                double distance = 0.0;
                if (!stable_relative_distance(
                        state.bodies[left].position_m,
                        state.bodies[right].position_m,
                        distance)) {
                    throw std::invalid_argument(
                        "source state contains non-representable relative geometry");
                }
                if (!(distance > 0.0)) {
                    throw std::invalid_argument("source state contains coincident bodies");
                }
            }
        }
    }
}

std::unordered_map<std::string, std::size_t> identity_index(
    const std::vector<ExperimentBodyIdentity>& identities) {
    std::unordered_map<std::string, std::size_t> result;
    for (std::size_t index = 0; index < identities.size(); ++index) {
        if (!result.emplace(identities[index].stable_id, index).second) {
            throw std::invalid_argument("experiment snapshot stable IDs must be unique");
        }
    }
    return result;
}

BranchConservationDiagnostics conservation(
    const SimulationState& state,
    const Diagnostics& initial) {
    BranchConservationDiagnostics result;
    result.initial = initial;
    result.current = compute_diagnostics(state);
    if (!std::isfinite(result.current.kinetic_energy_j) ||
        !std::isfinite(result.current.potential_energy_j) ||
        !finite_vector(result.current.linear_momentum_kg_m_s) ||
        !finite_vector(result.current.angular_momentum_kg_m2_s) ||
        !finite_vector(result.current.center_of_mass_m)) {
        throw ScientificFailure(
            ExperimentIssueKind::non_finite_state,
            "branch diagnostics became non-finite");
    }
    result.absolute_energy_drift_j =
        result.current.total_energy_j() - initial.total_energy_j();
    result.relative_energy_drift = std::abs(result.absolute_energy_drift_j) /
        std::max(std::abs(initial.total_energy_j()), 1.0);
    result.linear_momentum_drift_kg_m_s =
        norm(result.current.linear_momentum_kg_m_s - initial.linear_momentum_kg_m_s);
    result.angular_momentum_drift_kg_m2_s =
        norm(result.current.angular_momentum_kg_m2_s - initial.angular_momentum_kg_m2_s);
    return result;
}

void increment_dropped_issue_count(CounterfactualExperimentResult& result) noexcept {
    if (result.dropped_issue_count != std::numeric_limits<std::size_t>::max()) {
        ++result.dropped_issue_count;
    }
}

std::size_t stored_issue_message_bytes(const CounterfactualExperimentResult& result) {
    std::size_t total = 0U;
    for (const auto& issue : result.issues) {
        total = checked_add(total, issue.message.size(), "stored issue message bytes");
    }
    return total;
}

void add_issue_once(
    CounterfactualExperimentResult& result,
    const ExperimentLimits& limits,
    ExperimentIssue issue) {
    const auto duplicate = std::find_if(
        result.issues.begin(), result.issues.end(), [&](const auto& existing) {
        return existing.kind == issue.kind &&
            existing.first_body_stable_id == issue.first_body_stable_id &&
            existing.second_body_stable_id == issue.second_body_stable_id &&
            existing.branch == issue.branch;
    });
    if (duplicate != result.issues.end()) {
        return;
    }

    auto fits = [&]() {
        return result.issues.size() < limits.maximum_issue_records &&
            issue.message.size() <= limits.maximum_issue_message_bytes &&
            stored_issue_message_bytes(result) <=
                limits.maximum_issue_message_bytes - issue.message.size();
    };
    if (fits()) {
        result.issues.push_back(std::move(issue));
    } else {
        increment_dropped_issue_count(result);
    }
}

ExperimentTerminalFailure make_terminal_failure(
    ExperimentIssueKind kind,
    double time_s,
    const char* detail) noexcept {
    ExperimentTerminalFailure failure;
    failure.kind = kind;
    failure.time_s = time_s;
    constexpr std::string_view prefix = "Paired batch was rolled back: ";
    auto append = [&](std::string_view text) {
        const std::size_t remaining = maximum_terminal_failure_message_bytes -
            failure.message_length;
        const std::size_t copied = std::min(remaining, text.size());
        std::copy_n(text.data(), copied,
            failure.message.data() + failure.message_length);
        failure.message_length += copied;
        failure.message_truncated = failure.message_truncated || copied != text.size();
    };
    append(prefix);
    append(detail == nullptr ? std::string_view{"unknown failure"} :
        std::string_view{detail});
    failure.message[failure.message_length] = '\0';
    return failure;
}

const char* branch_name(ExperimentBranch branch) noexcept {
    return branch == ExperimentBranch::baseline ? "baseline" : "counterfactual";
}

void inspect_branch_endpoint(
    const SimulationState& state,
    const std::vector<ExperimentBodyIdentity>& identities,
    const ScientificGuardPolicy& policy,
    ExperimentBranch branch,
    CounterfactualExperimentResult& result,
    const ExperimentLimits& limits) {
    const std::string label = branch_name(branch);
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        const auto& body = state.bodies[index];
        if (!(body.mass_kg > 0.0) || !std::isfinite(body.mass_kg) ||
            !finite_vector(body.position_m) || !finite_vector(body.velocity_m_s)) {
            throw ScientificFailure(
                ExperimentIssueKind::non_finite_state,
                label + " branch produced a non-finite state");
        }
    }
    for (std::size_t left = 0; left < state.bodies.size(); ++left) {
        for (std::size_t right = left + 1U; right < state.bodies.size(); ++right) {
            double distance = 0.0;
            if (!stable_relative_distance(
                    state.bodies[left].position_m,
                    state.bodies[right].position_m,
                    distance)) {
                throw ScientificFailure(
                    ExperimentIssueKind::non_finite_state,
                    label + " branch contains non-representable relative geometry");
            }
            if (!(distance > 0.0)) {
                throw ScientificFailure(
                    ExperimentIssueKind::coincident_bodies,
                    label + " branch contains coincident bodies");
            }
            const double combined_radius = checked_combined_radius(
                identities[left], identities[right]);
            const double close_threshold = checked_close_threshold(
                combined_radius, policy.dangerous_close_approach_radius_multiple);
            if (distance <= combined_radius) {
                add_issue_once(result, limits, {
                    ExperimentIssueKind::physical_radius_overlap,
                    false,
                    label + " branch entered sourced-radius overlap; point-mass results are not physically resolved",
                    state.time_s,
                    identities[left].stable_id,
                    identities[right].stable_id,
                    branch,
                });
            } else if (distance <= close_threshold) {
                add_issue_once(result, limits, {
                    ExperimentIssueKind::dangerous_close_approach,
                    false,
                    label + " branch entered the configured close-approach warning envelope",
                    state.time_s,
                    identities[left].stable_id,
                    identities[right].stable_id,
                    branch,
                });
            }
        }
    }
}

std::vector<Vec3> body_positions(const SimulationState& state) {
    std::vector<Vec3> result;
    result.reserve(state.bodies.size());
    for (const auto& body : state.bodies) {
        result.push_back(body.position_m);
    }
    return result;
}

double checked_linear_swept_minimum_distance(
    const Vec3& left_before,
    const Vec3& right_before,
    const Vec3& left_after,
    const Vec3& right_after) {
    if (!finite_vector(left_before) || !finite_vector(right_before) ||
        !finite_vector(left_after) || !finite_vector(right_after)) {
        throw ScientificFailure(
            ExperimentIssueKind::non_finite_state,
            "swept proximity input became non-finite");
    }
    Vec3 relative_before;
    Vec3 relative_after;
    if (!checked_relative_vector(left_before, right_before, relative_before) ||
        !checked_relative_vector(left_after, right_after, relative_after)) {
        throw ScientificFailure(
            ExperimentIssueKind::non_finite_state,
            "swept relative geometry is not representable");
    }

    const double common_scale = std::max({
        std::abs(relative_before.x),
        std::abs(relative_before.y),
        std::abs(relative_before.z),
        std::abs(relative_after.x),
        std::abs(relative_after.y),
        std::abs(relative_after.z),
    });
    double fraction = 0.0;
    if (common_scale > 0.0) {
        const Vec3 before_scaled{
            relative_before.x / common_scale,
            relative_before.y / common_scale,
            relative_before.z / common_scale,
        };
        const Vec3 after_scaled{
            relative_after.x / common_scale,
            relative_after.y / common_scale,
            relative_after.z / common_scale,
        };
        const Vec3 delta_scaled{
            after_scaled.x - before_scaled.x,
            after_scaled.y - before_scaled.y,
            after_scaled.z - before_scaled.z,
        };
        const double delta_scaled_squared =
            delta_scaled.x * delta_scaled.x +
            delta_scaled.y * delta_scaled.y +
            delta_scaled.z * delta_scaled.z;
        if (!std::isfinite(delta_scaled_squared)) {
            throw ScientificFailure(
                ExperimentIssueKind::non_finite_state,
                "scaled swept proximity geometry is invalid");
        }
        if (delta_scaled_squared > 0.0) {
            const double scaled_projection =
                before_scaled.x * delta_scaled.x +
                before_scaled.y * delta_scaled.y +
                before_scaled.z * delta_scaled.z;
            if (!std::isfinite(scaled_projection)) {
                throw ScientificFailure(
                    ExperimentIssueKind::non_finite_state,
                    "scaled swept proximity projection is invalid");
            }
            if (scaled_projection < 0.0) {
                fraction = std::min(1.0, -scaled_projection /
                    delta_scaled_squared);
            }
        }
    }
    fraction = std::clamp(fraction, 0.0, 1.0);

    const double before_weight = 1.0 - fraction;
    Vec3 closest{
        std::fma(fraction, relative_after.x,
            before_weight * relative_before.x),
        std::fma(fraction, relative_after.y,
            before_weight * relative_before.y),
        std::fma(fraction, relative_after.z,
            before_weight * relative_before.z),
    };
    double distance = 0.0;
    if (!stable_vector_norm(closest, distance)) {
        throw ScientificFailure(
            ExperimentIssueKind::non_finite_state,
            "swept proximity distance is not representable");
    }
    return distance;
}

void inspect_swept_branch(
    const std::vector<Vec3>& positions_before,
    const SimulationState& state_after,
    const std::vector<ExperimentBodyIdentity>& identities,
    const ScientificGuardPolicy& policy,
    ExperimentBranch branch,
    CounterfactualExperimentResult& result,
    const ExperimentLimits& limits) {
    if (positions_before.size() != state_after.bodies.size() ||
        identities.size() != state_after.bodies.size()) {
        throw std::logic_error("swept proximity body metadata is inconsistent");
    }
    const std::string label = branch_name(branch);
    for (std::size_t left = 0; left < state_after.bodies.size(); ++left) {
        for (std::size_t right = left + 1U; right < state_after.bodies.size(); ++right) {
            const double distance = linear_swept_minimum_distance(
                positions_before[left], positions_before[right],
                state_after.bodies[left].position_m,
                state_after.bodies[right].position_m);
            const double combined_radius = checked_combined_radius(
                identities[left], identities[right]);
            const double close_threshold = checked_close_threshold(
                combined_radius, policy.dangerous_close_approach_radius_multiple);
            if (distance <= combined_radius) {
                add_issue_once(result, limits, {
                    ExperimentIssueKind::physical_radius_overlap,
                    false,
                    label + " branch crossed the sourced-radius envelope under a linear swept-step proximity test; point-mass results are not physically resolved",
                    state_after.time_s,
                    identities[left].stable_id,
                    identities[right].stable_id,
                    branch,
                });
            } else if (distance <= close_threshold) {
                add_issue_once(result, limits, {
                    ExperimentIssueKind::dangerous_close_approach,
                    false,
                    label + " branch crossed the configured close-approach envelope under a linear swept-step proximity test",
                    state_after.time_s,
                    identities[left].stable_id,
                    identities[right].stable_id,
                    branch,
                });
            }
        }
    }
}

std::vector<ExperimentBodyIdentity> compact_identities(
    const std::vector<ExperimentBodyIdentity>& identities,
    std::size_t removed_index) {
    std::vector<ExperimentBodyIdentity> result;
    result.reserve(identities.size() - 1U);
    for (std::size_t index = 0; index < identities.size(); ++index) {
        if (index != removed_index) {
            result.push_back(identities[index]);
        }
    }
    return result;
}

void append_sample(
    CounterfactualExperimentResult& result,
    const ExperimentSnapshot& snapshot,
    const CounterfactualExperimentConfiguration& configuration,
    const std::vector<std::size_t>& monitored_mapping_indices,
    const std::vector<std::size_t>& reference_mapping_indices,
    std::vector<std::size_t>& consecutive_threshold_samples) {
    ExperimentSample sample;
    sample.elapsed_time_s = checked_elapsed_time(
        result.completed_steps, snapshot.fixed_timestep_s());
    sample.bodies.reserve(monitored_mapping_indices.size());
    sample.baseline_reference_positions_m.reserve(reference_mapping_indices.size());

    for (std::size_t monitored = 0; monitored < monitored_mapping_indices.size(); ++monitored) {
        const auto& mapping = result.survivor_mapping[monitored_mapping_indices[monitored]];
        const auto& baseline = result.baseline_state.bodies[mapping.baseline_index];
        const auto& counterfactual =
            result.counterfactual_state.bodies[mapping.counterfactual_index];
        BodyTrajectorySample body_sample;
        body_sample.baseline_position_m = baseline.position_m;
        body_sample.counterfactual_position_m = counterfactual.position_m;
        body_sample.baseline_velocity_m_s = baseline.velocity_m_s;
        body_sample.counterfactual_velocity_m_s = counterfactual.velocity_m_s;
        body_sample.absolute_separation_m = norm(
            baseline.position_m - counterfactual.position_m);
        body_sample.velocity_difference_m_s = norm(
            baseline.velocity_m_s - counterfactual.velocity_m_s);

        auto& summary = result.summaries[monitored];
        body_sample.normalized_deviation = body_sample.absolute_separation_m /
            summary.fixed_normalization_scale_m;
        if (mapping.baseline_parent_index && mapping.counterfactual_parent_index) {
            const Vec3 baseline_relative = baseline.position_m -
                result.baseline_state.bodies[*mapping.baseline_parent_index].position_m;
            const Vec3 counterfactual_relative = counterfactual.position_m -
                result.counterfactual_state.bodies[*mapping.counterfactual_parent_index].position_m;
            body_sample.parent_relative_separation_m = norm(
                baseline_relative - counterfactual_relative);
        }

        summary.final_absolute_separation_m = body_sample.absolute_separation_m;
        summary.final_normalized_deviation = body_sample.normalized_deviation;
        summary.final_velocity_difference_m_s = body_sample.velocity_difference_m_s;
        if (body_sample.absolute_separation_m > summary.maximum_absolute_separation_m) {
            summary.maximum_absolute_separation_m = body_sample.absolute_separation_m;
            summary.maximum_absolute_separation_time_s = sample.elapsed_time_s;
        }
        summary.maximum_normalized_deviation = std::max(
            summary.maximum_normalized_deviation, body_sample.normalized_deviation);
        summary.maximum_velocity_difference_m_s = std::max(
            summary.maximum_velocity_difference_m_s, body_sample.velocity_difference_m_s);
        if (body_sample.parent_relative_separation_m) {
            summary.final_parent_relative_separation_m =
                body_sample.parent_relative_separation_m;
            if (!summary.maximum_parent_relative_separation_m ||
                *body_sample.parent_relative_separation_m >
                    *summary.maximum_parent_relative_separation_m) {
                summary.maximum_parent_relative_separation_m =
                    body_sample.parent_relative_separation_m;
                summary.maximum_parent_relative_separation_time_s = sample.elapsed_time_s;
            }
        }

        if (body_sample.absolute_separation_m >=
                configuration.significance.absolute_separation_threshold_m &&
            body_sample.normalized_deviation >=
                configuration.significance.normalized_deviation_threshold) {
            if (!summary.first_threshold_crossing_time_s) {
                summary.first_threshold_crossing_time_s = sample.elapsed_time_s;
            }
            ++consecutive_threshold_samples[monitored];
            if (!summary.significant && consecutive_threshold_samples[monitored] >=
                    configuration.significance.persistence_samples) {
                summary.significant = true;
                summary.persistent_threshold_time_s = sample.elapsed_time_s;
            }
        } else {
            consecutive_threshold_samples[monitored] = 0U;
        }
        sample.bodies.push_back(body_sample);
    }
    for (const std::size_t mapping_index : reference_mapping_indices) {
        const auto& mapping = result.survivor_mapping[mapping_index];
        sample.baseline_reference_positions_m.push_back(
            result.baseline_state.bodies[mapping.baseline_index].position_m);
    }
    result.samples.push_back(std::move(sample));
}

}  // namespace

std::uint64_t estimate_counterfactual_direct_force_workload(
    std::size_t baseline_body_count,
    std::size_t counterfactual_body_count,
    std::size_t step_count) {
    return checked_workload(
        baseline_body_count, counterfactual_body_count, step_count);
}

double linear_swept_minimum_distance(
    const Vec3& left_before,
    const Vec3& right_before,
    const Vec3& left_after,
    const Vec3& right_after) {
    return checked_linear_swept_minimum_distance(
        left_before, right_before, left_after, right_after);
}

ExperimentSnapshot::ExperimentSnapshot(ExperimentSnapshotDefinition definition)
    : scenario_id_(std::move(definition.scenario_id)),
      source_state_provenance_(std::move(definition.source_state_provenance)),
      coordinate_frame_(std::move(definition.coordinate_frame)),
      central_star_stable_id_(std::move(definition.central_star_stable_id)),
      source_state_(std::move(definition.source_state)),
      identities_(std::move(definition.identities)),
      fixed_timestep_s_(definition.fixed_timestep_s),
      integrator_(definition.integrator),
      timestep_convergence_established_(definition.timestep_convergence_established),
      maximum_supported_horizon_s_(definition.maximum_supported_horizon_s) {
    if (scenario_id_.empty() || source_state_provenance_.empty() ||
        coordinate_frame_.empty() || central_star_stable_id_.empty()) {
        throw std::invalid_argument("experiment snapshot identity and provenance fields are required");
    }
    if (!(fixed_timestep_s_ > 0.0) || !std::isfinite(fixed_timestep_s_) ||
        !(maximum_supported_horizon_s_ > 0.0) ||
        !std::isfinite(maximum_supported_horizon_s_)) {
        throw std::invalid_argument("experiment timestep and horizon limit must be finite and positive");
    }
    validate_state(source_state_, true);
    const double first_advanced_time = source_state_.time_s + fixed_timestep_s_;
    if (!std::isfinite(first_advanced_time) ||
        !(first_advanced_time > source_state_.time_s)) {
        throw std::invalid_argument(
            "source absolute timestamp cannot advance reliably by the fixed timestep");
    }
    if (identities_.size() != source_state_.bodies.size()) {
        throw std::invalid_argument("experiment identity count does not match source body count");
    }
    const auto index = identity_index(identities_);
    for (const auto& identity : identities_) {
        if (identity.stable_id.empty() || !(identity.physical_radius_m > 0.0) ||
            !std::isfinite(identity.physical_radius_m)) {
            throw std::invalid_argument("experiment body identity has invalid physical metadata");
        }
        if (identity.parent_stable_id && *identity.parent_stable_id == identity.stable_id) {
            throw std::invalid_argument("experiment body cannot be its own parent");
        }
        if (identity.parent_stable_id) {
            const auto parent = index.find(*identity.parent_stable_id);
            const auto child = index.find(identity.stable_id);
            if (parent != index.end() && parent->second >= child->second) {
                throw std::invalid_argument(
                    "present experiment parents must precede their children");
            }
        }
    }
}

const std::string& ExperimentSnapshot::scenario_id() const noexcept { return scenario_id_; }
const std::string& ExperimentSnapshot::source_state_provenance() const noexcept {
    return source_state_provenance_;
}
const std::string& ExperimentSnapshot::coordinate_frame() const noexcept {
    return coordinate_frame_;
}
const std::string& ExperimentSnapshot::central_star_stable_id() const noexcept {
    return central_star_stable_id_;
}
const SimulationState& ExperimentSnapshot::source_state() const noexcept { return source_state_; }
const std::vector<ExperimentBodyIdentity>& ExperimentSnapshot::identities() const noexcept {
    return identities_;
}
double ExperimentSnapshot::fixed_timestep_s() const noexcept { return fixed_timestep_s_; }
ExperimentIntegrator ExperimentSnapshot::integrator() const noexcept { return integrator_; }
bool ExperimentSnapshot::timestep_convergence_established() const noexcept {
    return timestep_convergence_established_;
}
double ExperimentSnapshot::maximum_supported_horizon_s() const noexcept {
    return maximum_supported_horizon_s_;
}

ExperimentSnapshot capture_legacy_solar_system_snapshot(const SimulationState& source_state) {
    const auto reference = make_approximate_solar_system();
    if (source_state.bodies.size() != reference.initial_state.bodies.size()) {
        throw std::invalid_argument("legacy Solar snapshot body count is invalid");
    }
    std::vector<ExperimentBodyIdentity> identities;
    identities.reserve(reference.metadata.size());
    for (std::size_t index = 0; index < reference.metadata.size(); ++index) {
        if (source_state.bodies[index].name != reference.initial_state.bodies[index].name ||
            source_state.bodies[index].mass_kg != reference.initial_state.bodies[index].mass_kg) {
            throw std::invalid_argument("legacy Solar snapshot body ordering or mass changed");
        }
        identities.push_back({
            reference.metadata[index].stable_id,
            index == 0U ? std::optional<std::string>{} :
                std::optional<std::string>{reference.metadata[0].stable_id},
            reference.metadata[index].mean_radius_m,
            index != 0U,
        });
    }
    return ExperimentSnapshot({
        "legacy-approximate-nine-body",
        std::string(solar_system_position_source) + "; captured simulation time " +
            std::to_string(source_state.time_s) + " s",
        solar_system_frame,
        reference.metadata.front().stable_id,
        source_state,
        std::move(identities),
        solar_system_recommended_dt_s,
        ExperimentIntegrator::velocity_verlet,
        true,
        legacy_initial_horizon_limit_s,
    });
}

ExperimentSnapshot capture_astronomical_catalog_snapshot(
    const SimulationState& source_state,
    const std::vector<std::string>& included_stable_ids) {
    const auto catalog = astronomy::make_solar_system_60_catalog();
    std::unordered_set<std::string> requested;
    if (included_stable_ids.empty()) {
        for (const auto& record : catalog.records()) {
            requested.insert(record.stable_id());
        }
    } else {
        for (const auto& id : included_stable_ids) {
            if (!requested.insert(id).second || catalog.find(id) == nullptr) {
                throw std::invalid_argument("catalog snapshot includes an unknown or duplicate stable ID");
            }
        }
    }

    std::vector<ExperimentBodyIdentity> identities;
    identities.reserve(requested.size());
    std::size_t source_index = 0U;
    for (const auto& record : catalog.records()) {
        if (requested.count(record.stable_id()) == 0U) {
            continue;
        }
        if (source_index >= source_state.bodies.size() ||
            source_state.bodies[source_index].name != record.display_name() ||
            source_state.bodies[source_index].mass_kg != record.mass_kg()) {
            throw std::invalid_argument("catalog snapshot body ordering or mass does not match the catalog");
        }
        const bool curated = record.body_class() == astronomy::AstronomicalBodyClass::planet ||
            record.body_class() == astronomy::AstronomicalBodyClass::dwarf_planet;
        identities.push_back({
            record.stable_id(), record.parent_stable_id(), record.mean_radius_m(), curated,
        });
        ++source_index;
    }
    if (source_index != source_state.bodies.size()) {
        throw std::invalid_argument("catalog snapshot body count does not match selected stable IDs");
    }
    return ExperimentSnapshot({
        astronomy::solar_system_60_dataset_id,
        std::string("immutable JPL snapshot ") + astronomy::solar_system_60_dataset_version +
            "; captured simulation time " + std::to_string(source_state.time_s) + " s",
        "SSB-centered J2000 ICRF geometric Cartesian SI; source epoch JD 2451545.0 TDB",
        "sol:sun",
        source_state,
        std::move(identities),
        catalog_provisional_timestep_s,
        ExperimentIntegrator::velocity_verlet,
        false,
        catalog_initial_horizon_limit_s,
    });
}

CounterfactualExperimentRunner::CounterfactualExperimentRunner(
    ExperimentSnapshot snapshot,
    CounterfactualExperimentConfiguration configuration)
    : snapshot_(std::move(snapshot)), configuration_(std::move(configuration)) {
    if (configuration_.removed_body_stable_id.empty()) {
        throw std::invalid_argument("exactly one removed body stable ID is required");
    }
    if (configuration_.backend == ExperimentBackend::openmp) {
        if (!openmp_supported()) {
            throw std::runtime_error("OpenMP experiment backend was requested but is not compiled");
        }
        if (configuration_.openmp_threads < 1) {
            throw std::invalid_argument("OpenMP experiment thread count must be positive");
        }
    }
    if (configuration_.limits.integration_steps_per_batch == 0U ||
        configuration_.limits.maximum_monitored_bodies == 0U ||
        configuration_.limits.maximum_reference_bodies == 0U ||
        configuration_.limits.maximum_stored_samples == 0U ||
        configuration_.limits.memory_budget_bytes == 0U ||
        configuration_.limits.workload_budget_interactions == 0U ||
        configuration_.limits.maximum_issue_records == 0U ||
        configuration_.limits.maximum_issue_message_bytes == 0U) {
        throw std::invalid_argument("experiment resource limits must be nonzero");
    }
    if (!(configuration_.significance.absolute_separation_threshold_m > 0.0) ||
        !std::isfinite(configuration_.significance.absolute_separation_threshold_m) ||
        configuration_.significance.normalized_deviation_threshold < 0.0 ||
        !std::isfinite(configuration_.significance.normalized_deviation_threshold) ||
        configuration_.significance.persistence_samples == 0U ||
        !(configuration_.significance.normalization_denominator_floor_m > 0.0) ||
        !std::isfinite(configuration_.significance.normalization_denominator_floor_m)) {
        throw std::invalid_argument("experiment significance policy is invalid");
    }
    if (!(configuration_.guards.dangerous_close_approach_radius_multiple >= 1.0) ||
        !std::isfinite(configuration_.guards.dangerous_close_approach_radius_multiple) ||
        !(configuration_.guards.maximum_relative_energy_drift > 0.0) ||
        !std::isfinite(configuration_.guards.maximum_relative_energy_drift)) {
        throw std::invalid_argument("experiment scientific guard policy is invalid");
    }

    const std::size_t total_steps = checked_integral_steps(
        configuration_.limits.observation_horizon_s, snapshot_.fixed_timestep_s(),
        "observation horizon");
    const std::size_t sample_steps = checked_integral_steps(
        configuration_.limits.sample_cadence_s, snapshot_.fixed_timestep_s(),
        "sample cadence");
    if (configuration_.limits.observation_horizon_s > snapshot_.maximum_supported_horizon_s()) {
        throw std::invalid_argument("observation horizon exceeds the snapshot's supported policy limit");
    }
    std::size_t stored_samples = total_steps / sample_steps;
    if (total_steps % sample_steps != 0U) {
        stored_samples = checked_add(
            stored_samples, 1U, "experiment final retained sample count");
    }
    stored_samples = checked_add(
        stored_samples, 1U, "experiment retained sample count including t0");
    if (stored_samples > configuration_.limits.maximum_stored_samples) {
        throw std::invalid_argument("experiment exceeds the maximum stored-sample limit");
    }

    const auto indices = identity_index(snapshot_.identities());
    const auto removed = indices.find(configuration_.removed_body_stable_id);
    if (removed == indices.end()) {
        throw std::invalid_argument("removed body stable ID is absent from the snapshot");
    }
    if (configuration_.removed_body_stable_id == snapshot_.central_star_stable_id()) {
        throw std::invalid_argument("Sun removal is not supported in this increment");
    }
    if (!snapshot_.identities()[removed->second].curated_removal_candidate) {
        throw std::invalid_argument("removed body is not in the curated one-body removal policy");
    }
    if (snapshot_.source_state().bodies.size() <= 2U) {
        throw std::invalid_argument("removal must leave at least two gravitating bodies");
    }

    result_.removed_body_stable_id = configuration_.removed_body_stable_id;
    result_.baseline_state = snapshot_.source_state();
    result_.counterfactual_state.time_s = snapshot_.source_state().time_s;
    result_.counterfactual_state.bodies.reserve(snapshot_.source_state().bodies.size() - 1U);
    std::unordered_map<std::string, std::size_t> counter_indices;
    for (std::size_t index = 0; index < snapshot_.source_state().bodies.size(); ++index) {
        if (index == removed->second) {
            continue;
        }
        counter_indices.emplace(
            snapshot_.identities()[index].stable_id,
            result_.counterfactual_state.bodies.size());
        result_.counterfactual_state.bodies.push_back(snapshot_.source_state().bodies[index]);
    }

    for (std::size_t baseline_index = 0;
         baseline_index < snapshot_.identities().size(); ++baseline_index) {
        if (baseline_index == removed->second) {
            continue;
        }
        const auto& identity = snapshot_.identities()[baseline_index];
        SurvivorMapping mapping;
        mapping.stable_id = identity.stable_id;
        mapping.baseline_index = baseline_index;
        mapping.counterfactual_index = counter_indices.at(identity.stable_id);
        if (!same_body(snapshot_.source_state().bodies[baseline_index],
                       result_.counterfactual_state.bodies[mapping.counterfactual_index])) {
            throw std::logic_error("counterfactual survivor initial state is not identical");
        }
        if (identity.parent_stable_id) {
            const auto baseline_parent = indices.find(*identity.parent_stable_id);
            const auto counter_parent = counter_indices.find(*identity.parent_stable_id);
            if (baseline_parent != indices.end() && counter_parent != counter_indices.end()) {
                mapping.baseline_parent_index = baseline_parent->second;
                mapping.counterfactual_parent_index = counter_parent->second;
            }
        }
        result_.survivor_mapping.push_back(std::move(mapping));
    }

    std::unordered_set<std::string> monitored_ids;
    if (configuration_.monitored_body_stable_ids.empty()) {
        for (const auto& mapping : result_.survivor_mapping) {
            monitored_ids.insert(mapping.stable_id);
        }
    } else {
        for (const auto& id : configuration_.monitored_body_stable_ids) {
            if (!monitored_ids.insert(id).second) {
                throw std::invalid_argument("monitored body stable IDs must be unique");
            }
            if (id == configuration_.removed_body_stable_id) {
                throw std::invalid_argument("the removed body cannot be monitored as a survivor");
            }
        }
    }
    for (std::size_t mapping_index = 0;
         mapping_index < result_.survivor_mapping.size(); ++mapping_index) {
        if (monitored_ids.count(result_.survivor_mapping[mapping_index].stable_id) != 0U) {
            monitored_mapping_indices_.push_back(mapping_index);
            result_.monitored_body_stable_ids.push_back(
                result_.survivor_mapping[mapping_index].stable_id);
        }
    }
    if (monitored_mapping_indices_.size() != monitored_ids.size()) {
        throw std::invalid_argument("a monitored stable ID is absent from the survivor mapping");
    }
    if (monitored_mapping_indices_.size() > configuration_.limits.maximum_monitored_bodies) {
        throw std::invalid_argument("experiment exceeds the monitored-body limit");
    }
    std::unordered_set<std::string> reference_ids;
    for (const auto& id : configuration_.retained_reference_body_stable_ids) {
        if (!reference_ids.insert(id).second) {
            throw std::invalid_argument("reference body stable IDs must be unique");
        }
        if (id == configuration_.removed_body_stable_id) {
            throw std::invalid_argument("the removed body cannot be a retained reference");
        }
        if (monitored_ids.count(id) != 0U) {
            throw std::invalid_argument(
                "reference bodies must be distinct from monitored metric bodies");
        }
    }
    for (std::size_t mapping_index = 0;
         mapping_index < result_.survivor_mapping.size(); ++mapping_index) {
        if (reference_ids.count(result_.survivor_mapping[mapping_index].stable_id) != 0U) {
            reference_mapping_indices_.push_back(mapping_index);
            result_.reference_body_stable_ids.push_back(
                result_.survivor_mapping[mapping_index].stable_id);
        }
    }
    if (reference_mapping_indices_.size() != reference_ids.size()) {
        throw std::invalid_argument("a reference stable ID is absent from the survivor mapping");
    }
    if (reference_mapping_indices_.size() >
        configuration_.limits.maximum_reference_bodies) {
        throw std::invalid_argument("experiment exceeds the reference-body limit");
    }
    const std::size_t per_sample_body_bytes = checked_multiply(
        monitored_mapping_indices_.size(), conservative_bytes_per_body_sample,
        "retained per-body sample memory estimate");
    const std::size_t per_sample_reference_bytes = checked_multiply(
        reference_mapping_indices_.size(),
        sizeof(Vec3) + conservative_allocation_overhead_bytes,
        "retained reference sample memory estimate");
    const std::size_t per_sample_bytes = checked_add(
        checked_add(
            checked_add(sizeof(ExperimentSample), per_sample_body_bytes,
                "retained sample memory estimate"),
            per_sample_reference_bytes, "retained sample memory estimate"),
        conservative_allocation_overhead_bytes,
        "retained sample allocation estimate");
    const std::size_t history_bytes = checked_multiply(
        stored_samples, per_sample_bytes, "retained history memory estimate");
    std::size_t estimated_bytes = checked_multiply(
        4U, history_bytes,
        "transactional history and vector-capacity memory estimate");
    estimated_bytes = checked_add(estimated_bytes, checked_multiply(
        monitored_mapping_indices_.size(),
        sizeof(BodyDeviationSummary) + sizeof(std::size_t) +
            conservative_allocation_overhead_bytes,
        "summary memory estimate"), "experiment memory estimate");
    estimated_bytes = checked_add(estimated_bytes, checked_multiply(
        snapshot_.source_state().bodies.size(),
        5U * (sizeof(Body) + conservative_allocation_overhead_bytes),
        "transactional state memory estimate"), "experiment memory estimate");
    estimated_bytes = checked_add(estimated_bytes, checked_multiply(
        snapshot_.identities().size(),
        2U * (sizeof(ExperimentBodyIdentity) + conservative_allocation_overhead_bytes),
        "identity memory estimate"), "experiment memory estimate");
    std::size_t maximum_identity_bytes = 0U;
    for (const auto& identity : snapshot_.identities()) {
        maximum_identity_bytes = std::max(
            maximum_identity_bytes, identity.stable_id.size());
        if (identity.parent_stable_id) {
            maximum_identity_bytes = std::max(
                maximum_identity_bytes, identity.parent_stable_id->size());
        }
    }
    const std::size_t per_issue_bytes = checked_add(
        sizeof(ExperimentIssue) + 2U * conservative_allocation_overhead_bytes,
        checked_multiply(2U, checked_add(maximum_identity_bytes, 1U,
            "issue identity string estimate"), "issue identity string estimate"),
        "issue record memory estimate");
    estimated_bytes = checked_add(estimated_bytes, checked_multiply(
        checked_multiply(2U, configuration_.limits.maximum_issue_records,
            "transactional issue record count estimate"),
        per_issue_bytes,
        "issue record memory estimate"), "experiment memory estimate");
    estimated_bytes = checked_add(
        estimated_bytes, checked_multiply(
            2U, configuration_.limits.maximum_issue_message_bytes,
            "transactional issue message memory estimate"),
        "issue message memory estimate");
    std::size_t string_bytes = checked_add(
        checked_add(snapshot_.scenario_id().size(),
            snapshot_.source_state_provenance().size(), "snapshot string memory estimate"),
        checked_add(snapshot_.coordinate_frame().size(),
            snapshot_.central_star_stable_id().size(), "snapshot string memory estimate"),
        "snapshot string memory estimate");
    string_bytes = checked_add(string_bytes,
        configuration_.removed_body_stable_id.size(), "configuration string memory estimate");
    for (const auto& identity : snapshot_.identities()) {
        string_bytes = checked_add(string_bytes, identity.stable_id.size(),
            "identity string memory estimate");
        if (identity.parent_stable_id) {
            string_bytes = checked_add(string_bytes, identity.parent_stable_id->size(),
                "identity parent string memory estimate");
        }
    }
    for (const auto& body : snapshot_.source_state().bodies) {
        string_bytes = checked_add(string_bytes, body.name.size(),
            "body-name string memory estimate");
    }
    estimated_bytes = checked_add(estimated_bytes,
        checked_multiply(8U, string_bytes, "copied string memory estimate"),
        "experiment memory estimate");
    estimated_bytes = checked_add(estimated_bytes, checked_multiply(
        snapshot_.source_state().bodies.size(), 256U,
        "mapping and container memory estimate"), "experiment memory estimate");
    estimated_bytes = checked_add(estimated_bytes,
        4U * sizeof(BranchConservationDiagnostics) +
            2U * sizeof(CounterfactualExperimentResult) +
            conservative_allocation_overhead_bytes,
        "diagnostic and result memory estimate");
    if (estimated_bytes > configuration_.limits.memory_budget_bytes) {
        throw std::invalid_argument("experiment exceeds the configured history memory budget");
    }
    const std::uint64_t workload = estimate_counterfactual_direct_force_workload(
        result_.baseline_state.bodies.size(), result_.counterfactual_state.bodies.size(),
        total_steps);
    if (workload > configuration_.limits.workload_budget_interactions) {
        throw std::invalid_argument("experiment exceeds the configured direct-force workload budget");
    }

    result_.baseline_conservation.initial = compute_diagnostics(result_.baseline_state);
    result_.counterfactual_conservation.initial =
        compute_diagnostics(result_.counterfactual_state);
    result_.baseline_conservation = conservation(
        result_.baseline_state, result_.baseline_conservation.initial);
    result_.counterfactual_conservation = conservation(
        result_.counterfactual_state, result_.counterfactual_conservation.initial);
    const auto compact = compact_identities(snapshot_.identities(), removed->second);
    inspect_branch_endpoint(result_.baseline_state, snapshot_.identities(),
        configuration_.guards, ExperimentBranch::baseline, result_, configuration_.limits);
    inspect_branch_endpoint(result_.counterfactual_state, compact,
        configuration_.guards, ExperimentBranch::counterfactual, result_,
        configuration_.limits);
    if (!snapshot_.timestep_convergence_established()) {
        add_issue_once(result_, configuration_.limits, {
            ExperimentIssueKind::insufficient_timestep_convergence,
            false,
            "The fixed timestep lacks sufficient convergence evidence for general long-horizon conclusions",
            snapshot_.source_state().time_s,
            std::nullopt,
            std::nullopt,
            std::nullopt,
        });
    }

    result_.summaries.reserve(monitored_mapping_indices_.size());
    for (const std::size_t mapping_index : monitored_mapping_indices_) {
        const auto& mapping = result_.survivor_mapping[mapping_index];
        double scale = norm(result_.baseline_state.bodies[mapping.baseline_index].position_m);
        bool parent_available = false;
        if (mapping.baseline_parent_index && mapping.counterfactual_parent_index) {
            scale = norm(
                result_.baseline_state.bodies[mapping.baseline_index].position_m -
                result_.baseline_state.bodies[*mapping.baseline_parent_index].position_m);
            parent_available = true;
        }
        scale = std::max(scale,
            configuration_.significance.normalization_denominator_floor_m);
        BodyDeviationSummary summary;
        summary.stable_id = mapping.stable_id;
        summary.fixed_normalization_scale_m = scale;
        summary.parent_relative_available = parent_available;
        result_.summaries.push_back(std::move(summary));
    }
    consecutive_threshold_samples_.assign(monitored_mapping_indices_.size(), 0U);
    append_sample(result_, snapshot_, configuration_, monitored_mapping_indices_,
        reference_mapping_indices_, consecutive_threshold_samples_);
}

bool CounterfactualExperimentRunner::openmp_supported() noexcept {
#ifdef NBODY_EXPERIMENT_HAS_OPENMP
    return true;
#else
    return false;
#endif
}

const ExperimentSnapshot& CounterfactualExperimentRunner::snapshot() const noexcept {
    return snapshot_;
}
const CounterfactualExperimentConfiguration&
CounterfactualExperimentRunner::configuration() const noexcept {
    return configuration_;
}
const CounterfactualExperimentResult& CounterfactualExperimentRunner::result() const noexcept {
    return result_;
}

bool CounterfactualExperimentRunner::advance_batch() {
    if (result_.status == ExperimentStatus::completed ||
        result_.status == ExperimentStatus::cancelled ||
        result_.status == ExperimentStatus::failed) {
        return false;
    }
    if (cancellation_requested_) {
        result_.status = ExperimentStatus::cancelled;
        return false;
    }
    if (pause_requested_) {
        result_.status = ExperimentStatus::paused;
        return false;
    }

    const std::size_t total_steps = checked_integral_steps(
        configuration_.limits.observation_horizon_s, snapshot_.fixed_timestep_s(),
        "observation horizon");
    const std::size_t sample_steps = checked_integral_steps(
        configuration_.limits.sample_cadence_s, snapshot_.fixed_timestep_s(),
        "sample cadence");
    if (result_.completed_steps > total_steps) {
        throw std::logic_error("committed experiment step count exceeds its limit");
    }
    const std::size_t remaining_steps = total_steps - result_.completed_steps;
    if (remaining_steps == 0U) {
        result_.status = ExperimentStatus::completed;
        return false;
    }
    const std::size_t steps = std::min(
        configuration_.limits.integration_steps_per_batch,
        remaining_steps);

    try {
        auto next = result_;
        auto next_threshold_counts = consecutive_threshold_samples_;
        const auto compact = compact_identities(
            snapshot_.identities(),
            identity_index(snapshot_.identities()).at(
                configuration_.removed_body_stable_id));
        std::unique_ptr<ForceBackend> owned_backend;
        const ForceBackend* backend = &sequential_force_backend();
#ifdef NBODY_EXPERIMENT_HAS_OPENMP
        if (configuration_.backend == ExperimentBackend::openmp) {
            owned_backend = std::make_unique<OpenMpForceBackend>(configuration_.openmp_threads);
            backend = owned_backend.get();
        }
#endif
        for (std::size_t step = 0; step < steps; ++step) {
            const auto baseline_positions_before = body_positions(next.baseline_state);
            const auto counterfactual_positions_before = body_positions(
                next.counterfactual_state);
            const double baseline_time_before = next.baseline_state.time_s;
            const double counterfactual_time_before = next.counterfactual_state.time_s;
            velocity_verlet_step(next.baseline_state, snapshot_.fixed_timestep_s(), *backend);
            if (!std::isfinite(next.baseline_state.time_s) ||
                !(next.baseline_state.time_s > baseline_time_before)) {
                throw ScientificFailure(
                    ExperimentIssueKind::invalid_simulated_time,
                    "baseline branch simulated time failed to advance");
            }
            velocity_verlet_step(next.counterfactual_state, snapshot_.fixed_timestep_s(), *backend);
            if (!std::isfinite(next.counterfactual_state.time_s) ||
                !(next.counterfactual_state.time_s > counterfactual_time_before)) {
                throw ScientificFailure(
                    ExperimentIssueKind::invalid_simulated_time,
                    "counterfactual branch simulated time failed to advance");
            }
            if (next.baseline_state.time_s != next.counterfactual_state.time_s) {
                throw ScientificFailure(
                    ExperimentIssueKind::invalid_simulated_time,
                    "paired branch simulated times diverged");
            }
            next.completed_steps = checked_add(
                next.completed_steps, 1U, "completed experiment step count");
            (void)checked_elapsed_time(
                next.completed_steps, snapshot_.fixed_timestep_s());
            inspect_branch_endpoint(next.baseline_state, snapshot_.identities(),
                configuration_.guards, ExperimentBranch::baseline, next,
                configuration_.limits);
            inspect_branch_endpoint(next.counterfactual_state, compact,
                configuration_.guards, ExperimentBranch::counterfactual, next,
                configuration_.limits);
            inspect_swept_branch(baseline_positions_before, next.baseline_state,
                snapshot_.identities(), configuration_.guards,
                ExperimentBranch::baseline, next, configuration_.limits);
            inspect_swept_branch(counterfactual_positions_before,
                next.counterfactual_state, compact, configuration_.guards,
                ExperimentBranch::counterfactual, next, configuration_.limits);
            if (next.completed_steps % sample_steps == 0U ||
                next.completed_steps == total_steps) {
                append_sample(next, snapshot_, configuration_, monitored_mapping_indices_,
                    reference_mapping_indices_, next_threshold_counts);
            }
        }
        next.baseline_conservation = conservation(
            next.baseline_state, result_.baseline_conservation.initial);
        next.counterfactual_conservation = conservation(
            next.counterfactual_state, result_.counterfactual_conservation.initial);
        if (next.baseline_conservation.relative_energy_drift >
                configuration_.guards.maximum_relative_energy_drift ||
            next.counterfactual_conservation.relative_energy_drift >
                configuration_.guards.maximum_relative_energy_drift) {
            add_issue_once(next, configuration_.limits, {
                ExperimentIssueKind::excessive_conservation_drift,
                true,
                "A branch exceeded the configured relative energy-drift guard",
                next.baseline_state.time_s,
                std::nullopt,
                std::nullopt,
                std::nullopt,
            });
            throw ScientificFailure(
                ExperimentIssueKind::excessive_conservation_drift,
                "relative energy-drift guard exceeded");
        }
        next.status = next.completed_steps == total_steps ?
            ExperimentStatus::completed : ExperimentStatus::running;
        result_ = std::move(next);
        consecutive_threshold_samples_ = std::move(next_threshold_counts);
        if (cancellation_requested_ && result_.status != ExperimentStatus::completed) {
            result_.status = ExperimentStatus::cancelled;
        } else if (pause_requested_ &&
                   result_.status != ExperimentStatus::completed) {
            result_.status = ExperimentStatus::paused;
        }
        return true;
    } catch (const ScientificFailure& error) {
        const auto terminal = make_terminal_failure(
            error.issue_kind(), result_.baseline_state.time_s, error.what());
        result_.terminal_failure = terminal;
        result_.status = ExperimentStatus::failed;
        return false;
    } catch (const std::exception& error) {
        const auto terminal = make_terminal_failure(
            ExperimentIssueKind::integration_failure,
            result_.baseline_state.time_s, error.what());
        result_.terminal_failure = terminal;
        result_.status = ExperimentStatus::failed;
        return false;
    }
}

void CounterfactualExperimentRunner::run_to_completion() {
    while (result_.status != ExperimentStatus::completed &&
           result_.status != ExperimentStatus::cancelled &&
           result_.status != ExperimentStatus::failed) {
        if (!advance_batch() && result_.status != ExperimentStatus::running) {
            break;
        }
    }
}

void CounterfactualExperimentRunner::set_paused(bool paused) noexcept {
    pause_requested_ = paused;
}

bool CounterfactualExperimentRunner::paused() const noexcept {
    return pause_requested_;
}
void CounterfactualExperimentRunner::request_cancel() noexcept {
    cancellation_requested_ = true;
}

const char* experiment_backend_name(ExperimentBackend backend) noexcept {
    return backend == ExperimentBackend::openmp ? "OpenMP" : "Sequential";
}

const char* experiment_status_name(ExperimentStatus status) noexcept {
    switch (status) {
        case ExperimentStatus::ready: return "Ready";
        case ExperimentStatus::running: return "Running";
        case ExperimentStatus::paused: return "Paused";
        case ExperimentStatus::completed: return "Completed";
        case ExperimentStatus::cancelled: return "Cancelled";
        case ExperimentStatus::failed: return "Failed";
    }
    return "Unknown";
}

}  // namespace nbody::experiments
