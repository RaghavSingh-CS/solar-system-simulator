#pragma once

#include "nbody/barnes_hut.hpp"
#include "nbody/diagnostics.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nbody {

enum class LargeNDistribution { uniform, clustered, planar };
enum class LargeNBackend { sequential, openmp, cuda };
enum class LargeNSolver { direct, barnes_hut };

std::string_view large_n_name(LargeNDistribution value) noexcept;
std::string_view large_n_name(LargeNBackend value) noexcept;
std::string_view large_n_name(LargeNSolver value) noexcept;

struct LargeNGenerationConfiguration {
    std::size_t body_count{0};
    std::optional<std::uint64_t> seed;
    LargeNDistribution distribution{LargeNDistribution::uniform};
    double position_scale_m{1.0e9};
    double velocity_scale_m_s{100.0};
    double minimum_mass_kg{1.0e20};
    double maximum_mass_kg{1.0e21};
};

SimulationState make_large_n_synthetic_system(
    const LargeNGenerationConfiguration& configuration);

struct LargeNResourcePolicy {
    std::size_t maximum_body_count{100'000};
    std::size_t memory_budget_bytes{512U * 1024U * 1024U};
    std::uint64_t maximum_direct_interactions{500'000'000ULL};
    std::size_t maximum_steps{1'000};
    std::size_t maximum_warmups{10};
    std::size_t maximum_repetitions{100};
    double wall_time_budget_seconds{30.0};
    std::size_t maximum_direct_reference_bodies{4'096};
};

struct LargeNExperimentConfiguration {
    LargeNGenerationConfiguration generation;
    LargeNBackend backend{LargeNBackend::sequential};
    LargeNSolver solver{LargeNSolver::direct};
    int requested_threads{1};
    double timestep_s{10.0};
    std::size_t steps{1};
    std::size_t warmups{1};
    std::size_t repetitions{2};
    bool request_direct_reference{true};
    BarnesHutConfiguration barnes_hut;
    LargeNResourcePolicy resources;
};

struct LargeNPreflight {
    bool admitted{false};
    std::string reason;
    std::size_t estimated_memory_bytes{0};
    std::size_t generation_memory_bytes{0};
    std::size_t direct_reference_memory_bytes{0};
    std::size_t warmup_memory_bytes{0};
    std::size_t retained_repetition_memory_bytes{0};
    std::size_t characterization_memory_bytes{0};
    std::size_t serialization_memory_bytes{0};
    std::uint64_t estimated_direct_interactions{0};
    int effective_threads{0};
    bool direct_reference_available{false};
    std::string direct_reference_reason;
};

LargeNPreflight preflight_large_n_experiment(
    const LargeNExperimentConfiguration& configuration);

struct LargeNAccelerationErrorSummary {
    bool available{false};
    std::string unavailable_reason;
    double maximum_absolute_m_s2{0.0};
    double rms_absolute_m_s2{0.0};
    double p50_relative{0.0};
    double p95_relative{0.0};
    double p99_relative{0.0};
    double maximum_relative{0.0};
    double relative_denominator_floor_m_s2{0.0};
    std::size_t cancellation_sensitive_count{0};
};

LargeNAccelerationErrorSummary summarize_large_n_acceleration_errors(
    const std::vector<Vec3>& reference,
    const std::vector<Vec3>& candidate);

struct LargeNTrajectoryAccuracy {
    bool available{false};
    std::string unavailable_reason;
    double maximum_position_separation_m{0.0};
    double rms_position_separation_m{0.0};
    double maximum_velocity_separation_m_s{0.0};
    double rms_velocity_separation_m_s{0.0};
    double candidate_relative_energy_drift{0.0};
    double direct_relative_energy_drift{0.0};
    double candidate_relative_angular_momentum_drift{0.0};
    double direct_relative_angular_momentum_drift{0.0};
};

struct LargeNRepetitionMeasurement {
    std::size_t repetition{0};
    int observed_worker_threads{0};
    std::string worker_count_provenance;
    double complete_force_seconds{0.0};
    double complete_timestep_seconds{0.0};
    double tree_build_seconds{0.0};
    double force_traversal_seconds{0.0};
    std::size_t solver_working_memory_bytes{0};
    std::size_t node_count{0};
    std::size_t visited_nodes{0};
    std::size_t approximated_nodes{0};
    std::size_t direct_interactions{0};
};

struct LargeNExperimentResult {
    LargeNExperimentConfiguration configuration;
    LargeNPreflight preflight;
    std::string status{"not_run"};
    std::string message;
    double generation_seconds{0.0};
    double initial_direct_force_seconds{0.0};
    double direct_trajectory_seconds{0.0};
    double diagnostics_seconds{0.0};
    double total_experiment_seconds{0.0};
    LargeNAccelerationErrorSummary acceleration_error;
    LargeNTrajectoryAccuracy trajectory_accuracy;
    std::vector<LargeNRepetitionMeasurement> measurements;
};

// Test/embedding seam for deterministic boundary-budget validation. Production
// callers omit this argument and use the steady-clock elapsed time. An override
// may only increase the elapsed value observed at a named completed boundary.
struct LargeNExecutionControl {
    std::function<double(std::string_view, double)> boundary_elapsed_override;
};

LargeNExperimentResult run_large_n_experiment(
    const LargeNExperimentConfiguration& configuration,
    const LargeNExecutionControl& control = {});

std::string large_n_csv_header();
std::vector<std::string> serialize_large_n_csv_rows(
    const LargeNExperimentResult& result);

bool large_n_openmp_compiled() noexcept;

}  // namespace nbody
