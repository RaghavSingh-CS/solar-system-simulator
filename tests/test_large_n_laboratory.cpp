#include "nbody/large_n_laboratory.hpp"
#include "nbody/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint64_t bits(double value) {
    std::uint64_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::uint64_t checksum(const nbody::SimulationState& state) {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto append = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
    };
    for (const auto& body : state.bodies) {
        append(body.name.data(), body.name.size());
        const std::uint64_t values[]{
            bits(body.mass_kg), bits(body.position_m.x), bits(body.position_m.y),
            bits(body.position_m.z), bits(body.velocity_m_s.x),
            bits(body.velocity_m_s.y), bits(body.velocity_m_s.z),
        };
        append(values, sizeof(values));
    }
    return hash;
}

nbody::LargeNGenerationConfiguration generation(
    nbody::LargeNDistribution distribution,
    std::size_t count = 256,
    std::uint64_t seed = 0x4c3154455354ULL) {
    nbody::LargeNGenerationConfiguration result;
    result.body_count = count;
    result.seed = seed;
    result.distribution = distribution;
    return result;
}

void test_deterministic_generation_and_identities() {
    for (const auto distribution : {
             nbody::LargeNDistribution::uniform,
             nbody::LargeNDistribution::clustered,
             nbody::LargeNDistribution::planar}) {
        const auto first = nbody::make_large_n_synthetic_system(generation(distribution));
        const auto second = nbody::make_large_n_synthetic_system(generation(distribution));
        require(checksum(first) == checksum(second),
                "same large-N configuration was not bitwise reproducible");
        require(first.bodies.size() == 256, "large-N generator returned wrong body count");
        std::set<std::string> identities;
        std::set<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> positions;
        for (std::size_t index = 0; index < first.bodies.size(); ++index) {
            const auto& body = first.bodies[index];
            require(body.name == "large_n_" + std::string(8 - std::to_string(index).size(), '0') +
                        std::to_string(index),
                    "large-N stable identity/order changed");
            require(identities.insert(body.name).second, "large-N body identity repeated");
            require(positions.emplace(bits(body.position_m.x), bits(body.position_m.y),
                                      bits(body.position_m.z)).second,
                    "normal large-N fixture contained coincident positions");
            require(body.mass_kg > 0.0 && std::isfinite(body.mass_kg) &&
                        std::isfinite(body.position_m.x) &&
                        std::isfinite(body.position_m.y) &&
                        std::isfinite(body.position_m.z) &&
                        std::isfinite(body.velocity_m_s.x) &&
                        std::isfinite(body.velocity_m_s.y) &&
                        std::isfinite(body.velocity_m_s.z),
                    "large-N generator returned non-finite body data");
        }
        require(checksum(first) != checksum(nbody::make_large_n_synthetic_system(
                    generation(distribution, 256, 0x4c3154455355ULL))),
                "different large-N seeds produced the same state");
    }

    const auto regression = nbody::make_large_n_synthetic_system(
        generation(nbody::LargeNDistribution::uniform, 16, 0x123456789abcdef0ULL));
    constexpr std::uint64_t expected_checksum = 16770294616094591480ULL;
    require(checksum(regression) == expected_checksum,
            "large-N generator regression checksum changed: actual=" +
                std::to_string(checksum(regression)));
}

double axis_rms(const nbody::SimulationState& state, char axis) {
    long double sum = 0.0L;
    for (const auto& body : state.bodies) {
        const double value = axis == 'x' ? body.position_m.x :
            (axis == 'y' ? body.position_m.y : body.position_m.z);
        sum += static_cast<long double>(value) * value;
    }
    return std::sqrt(static_cast<double>(sum / state.bodies.size()));
}

void test_distribution_properties() {
    const auto uniform = nbody::make_large_n_synthetic_system(
        generation(nbody::LargeNDistribution::uniform, 2048));
    const auto planar = nbody::make_large_n_synthetic_system(
        generation(nbody::LargeNDistribution::planar, 2048));
    const auto clustered = nbody::make_large_n_synthetic_system(
        generation(nbody::LargeNDistribution::clustered, 2048));
    require(axis_rms(uniform, 'z') > 0.5 * axis_rms(uniform, 'x'),
            "uniform fixture collapsed toward a plane");
    require(axis_rms(planar, 'z') < 0.02 * axis_rms(planar, 'x') &&
                axis_rms(planar, 'z') < 0.02 * axis_rms(planar, 'y'),
            "planar fixture thickness is not bounded relative to in-plane spread");

    nbody::Vec3 group_means[4]{};
    std::size_t group_counts[4]{};
    for (std::size_t index = 0; index < clustered.bodies.size(); ++index) {
        const std::size_t group = index % 4U;
        group_means[group] += clustered.bodies[index].position_m;
        ++group_counts[group];
    }
    for (std::size_t group = 0; group < 4; ++group) {
        group_means[group] = group_means[group] /
            static_cast<double>(group_counts[group]);
    }
    double minimum_center_distance = std::numeric_limits<double>::infinity();
    for (std::size_t lhs = 0; lhs < 4; ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < 4; ++rhs) {
            minimum_center_distance = std::min(
                minimum_center_distance, nbody::norm(group_means[lhs] - group_means[rhs]));
        }
    }
    require(minimum_center_distance > 4.0e8,
            "clustered fixture did not retain separated cluster centers");
}

void test_invalid_generation_and_degenerate_fixture() {
    bool rejected = false;
    auto invalid = generation(nbody::LargeNDistribution::uniform);
    invalid.seed.reset();
    try {
        (void)nbody::make_large_n_synthetic_system(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "missing explicit large-N seed was accepted");
    invalid = generation(nbody::LargeNDistribution::uniform, 1);
    rejected = false;
    try {
        (void)nbody::make_large_n_synthetic_system(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "single-body large-N fixture was accepted");
    invalid = generation(nbody::LargeNDistribution::uniform);
    invalid.position_scale_m = std::numeric_limits<double>::infinity();
    rejected = false;
    try {
        (void)nbody::make_large_n_synthetic_system(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "non-finite large-N scale was accepted");

    nbody::SimulationState coincident;
    coincident.bodies = {
        {"degenerate_a", 1.0, {}, {}}, {"degenerate_b", 2.0, {}, {}},
    };
    rejected = false;
    try {
        (void)nbody::compute_accelerations(coincident, 1.0);
    } catch (const std::domain_error&) {
        rejected = true;
    }
    require(rejected, "explicit degenerate direct-force fixture was not rejected");
    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend;
        (void)backend.compute_accelerations(coincident, 1.0);
    } catch (const std::domain_error&) {
        rejected = true;
    }
    require(rejected, "explicit degenerate Barnes-Hut fixture was not rejected");
}

nbody::LargeNExperimentConfiguration experiment(
    std::size_t count,
    nbody::LargeNSolver solver = nbody::LargeNSolver::barnes_hut) {
    nbody::LargeNExperimentConfiguration result;
    result.generation = generation(nbody::LargeNDistribution::uniform, count);
    result.solver = solver;
    result.repetitions = 1;
    result.warmups = 0;
    result.steps = 1;
    return result;
}

void test_preflight_and_overflow_boundaries() {
    for (const std::size_t count :
         {256U, 1024U, 4096U, 8192U, 16384U, 32768U, 50000U, 100000U}) {
        auto configuration = experiment(count);
        configuration.request_direct_reference = false;
        const auto preflight = nbody::preflight_large_n_experiment(configuration);
        require(preflight.admitted,
                "supported requested Barnes-Hut size failed conservative admission: " +
                    std::to_string(count) + " " + preflight.reason);
    }
    auto direct = experiment(100000, nbody::LargeNSolver::direct);
    direct.request_direct_reference = false;
    require(!nbody::preflight_large_n_experiment(direct).admitted,
            "100K direct force bypassed the direct-work guard");

    auto no_reference = experiment(8192);
    const auto reference_preflight = nbody::preflight_large_n_experiment(no_reference);
    require(reference_preflight.admitted && !reference_preflight.direct_reference_available &&
                reference_preflight.direct_reference_reason ==
                    "body count exceeds direct-reference limit",
            "missing direct reference was not reported explicitly");

    auto memory = experiment(4096);
    memory.request_direct_reference = false;
    memory.resources.memory_budget_bytes = 1;
    require(!nbody::preflight_large_n_experiment(memory).admitted,
            "one-byte laboratory memory budget was bypassed");

    auto corrected_peak = experiment(4096);
    const std::size_t old_per_body = 2U * sizeof(nbody::Body) +
        4U * sizeof(nbody::Vec3) + 64U;
    const std::size_t old_estimate = corrected_peak.generation.body_count * old_per_body +
        corrected_peak.barnes_hut.memory_budget_bytes;
    corrected_peak.resources.memory_budget_bytes = old_estimate + 1U;
    const auto corrected_peak_preflight =
        nbody::preflight_large_n_experiment(corrected_peak);
    require(!corrected_peak_preflight.admitted &&
                corrected_peak_preflight.retained_repetition_memory_bytes > old_estimate,
            "corrected retained-repetition high water did not reject the old estimate");

    auto phase_accounting = experiment(256);
    const auto phase_preflight = nbody::preflight_large_n_experiment(phase_accounting);
    require(phase_preflight.admitted &&
                phase_preflight.estimated_memory_bytes == std::max({
                    phase_preflight.generation_memory_bytes,
                    phase_preflight.direct_reference_memory_bytes,
                    phase_preflight.warmup_memory_bytes,
                    phase_preflight.retained_repetition_memory_bytes,
                    phase_preflight.characterization_memory_bytes,
                    phase_preflight.serialization_memory_bytes}) &&
                phase_preflight.generation_memory_bytes > 0 &&
                phase_preflight.serialization_memory_bytes > 0,
            "phase-specific memory high-water accounting is inconsistent");

    auto overflow = experiment(2);
    overflow.generation.body_count = std::numeric_limits<std::size_t>::max();
    overflow.resources.maximum_body_count = std::numeric_limits<std::size_t>::max();
    require(!nbody::preflight_large_n_experiment(overflow).admitted,
            "overflowing body count was admitted");

    auto cuda = experiment(256);
    cuda.backend = nbody::LargeNBackend::cuda;
    require(!nbody::preflight_large_n_experiment(cuda).admitted,
            "metadata-only CUDA execution was admitted in L1");

    auto openmp = experiment(256);
    openmp.backend = nbody::LargeNBackend::openmp;
    openmp.requested_threads = 2;
    const auto openmp_preflight = nbody::preflight_large_n_experiment(openmp);
    require(openmp_preflight.admitted == nbody::large_n_openmp_compiled(),
            "OpenMP compile gate disagreed with laboratory preflight");
    if (openmp_preflight.admitted) {
        require(openmp_preflight.effective_threads == 2,
                "OpenMP effective thread metadata did not match admission");
    }

    auto invalid_barnes_hut = experiment(256);
    invalid_barnes_hut.barnes_hut.theta = 0.0;
    require(!nbody::preflight_large_n_experiment(invalid_barnes_hut).admitted,
            "invalid Barnes-Hut policy bypassed laboratory preflight");
}

void test_accuracy_statistics_and_execution() {
    const std::vector<nbody::Vec3> reference{{0.0, 0.0, 0.0}, {3.0, 4.0, 0.0}};
    const std::vector<nbody::Vec3> candidate{{1.0e-20, 0.0, 0.0}, {3.0, 5.0, 0.0}};
    const auto errors = nbody::summarize_large_n_acceleration_errors(reference, candidate);
    require(errors.available && errors.maximum_absolute_m_s2 == 1.0 &&
                errors.cancellation_sensitive_count == 1 &&
                errors.relative_denominator_floor_m_s2 > 0.0 &&
                errors.p50_relative <= errors.p95_relative &&
                errors.p95_relative <= errors.p99_relative &&
                errors.p99_relative <= errors.maximum_relative,
            "cancellation-aware acceleration summary is inconsistent");

    bool rejected = false;
    try {
        (void)nbody::summarize_large_n_acceleration_errors(
            {{0.0, 0.0, 0.0}}, {{1.0e308, 0.0, 0.0}});
    } catch (const std::domain_error&) {
        rejected = true;
    }
    require(rejected, "overflowing relative acceleration error was published");

    auto direct_configuration = experiment(32, nbody::LargeNSolver::direct);
    const auto direct = nbody::run_large_n_experiment(direct_configuration);
    require(direct.status == "completed" && direct.measurements.size() == 1,
            "bounded direct laboratory run did not complete");
    require(direct.acceleration_error.available &&
                direct.acceleration_error.maximum_absolute_m_s2 == 0.0 &&
                direct.trajectory_accuracy.available &&
                direct.trajectory_accuracy.maximum_position_separation_m == 0.0,
            "direct self-reference was not exact");
    require(direct.measurements[0].complete_force_seconds >= 0.0 &&
                direct.measurements[0].complete_timestep_seconds >= 0.0 &&
                direct.generation_seconds >= 0.0 && direct.total_experiment_seconds >= 0.0,
            "laboratory timing fields were invalid");

    auto approximate_configuration = experiment(64);
    approximate_configuration.barnes_hut.theta = 0.5;
    const auto approximate = nbody::run_large_n_experiment(approximate_configuration);
    require(approximate.status == "completed" &&
                approximate.acceleration_error.available &&
                approximate.trajectory_accuracy.available &&
                approximate.measurements[0].node_count > 0 &&
                approximate.measurements[0].solver_working_memory_bytes > 0,
            "bounded Barnes-Hut laboratory run did not publish solver metrics");

    auto unavailable_configuration = experiment(8192);
    unavailable_configuration.resources.wall_time_budget_seconds = 1.0;
    unavailable_configuration.warmups = 0;
    unavailable_configuration.repetitions = 1;
    const auto preflight = nbody::preflight_large_n_experiment(unavailable_configuration);
    require(preflight.admitted && !preflight.direct_reference_available,
            "large-N direct-reference omission did not survive execution preflight");

    const auto rows = nbody::serialize_large_n_csv_rows(approximate);
    require(rows.size() == approximate.measurements.size() &&
                nbody::large_n_csv_header().find("tree_build_s") != std::string::npos &&
                rows[0].find("completed") == 0,
            "laboratory evidence serialization omitted required fields");

    auto failure_configuration = experiment(32);
    failure_configuration.barnes_hut.maximum_traversal_work = 1;
    const auto failed = nbody::run_large_n_experiment(failure_configuration);
    require(failed.status == "failed" && failed.measurements.empty() &&
                !failed.acceleration_error.available &&
                !failed.trajectory_accuracy.available,
            "failed laboratory run published partial scientific results");

    auto extreme = experiment(2, nbody::LargeNSolver::direct);
    extreme.generation.position_scale_m = 1.0e100;
    extreme.generation.velocity_scale_m_s = 0.0;
    extreme.generation.minimum_mass_kg = 1.0e200;
    extreme.generation.maximum_mass_kg = 1.0e200;
    const auto extreme_result = nbody::run_large_n_experiment(extreme);
    require(extreme_result.status == "failed" && extreme_result.measurements.empty() &&
                extreme_result.message.find("diagnostics") != std::string::npos,
            "finite inputs producing nonrepresentable diagnostics were published");

    auto final_budget = experiment(32, nbody::LargeNSolver::direct);
    nbody::LargeNExecutionControl control;
    control.boundary_elapsed_override = [&final_budget](
        std::string_view boundary, double observed) {
        return boundary == "final scientific characterization"
            ? final_budget.resources.wall_time_budget_seconds + 1.0
            : observed;
    };
    const auto budget_result = nbody::run_large_n_experiment(final_budget, control);
    require(budget_result.status == "failed" && budget_result.measurements.empty() &&
                budget_result.message.find("final scientific characterization") !=
                    std::string::npos,
            "final-characterization wall-budget overrun was not transactional");
}

}  // namespace

int main() {
    try {
        test_deterministic_generation_and_identities();
        test_distribution_properties();
        test_invalid_generation_and_degenerate_fixture();
        test_preflight_and_overflow_boundaries();
        test_accuracy_statistics_and_execution();
        std::cout << "Large-N laboratory tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Large-N laboratory test failure: " << error.what() << '\n';
        return 1;
    }
}
