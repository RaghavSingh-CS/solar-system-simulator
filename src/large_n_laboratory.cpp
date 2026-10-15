#include "nbody/large_n_laboratory.hpp"

#include "nbody/openmp_backend.hpp"
#include "nbody/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace nbody {
namespace {

using Clock = std::chrono::steady_clock;

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* label) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::length_error(std::string(label) + " size overflow");
    }
    return lhs + rhs;
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::length_error(std::string(label) + " size overflow");
    }
    return lhs * rhs;
}

std::uint64_t checked_u64_multiply(
    std::uint64_t lhs, std::uint64_t rhs, const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::length_error(std::string(label) + " work overflow");
    }
    return lhs * rhs;
}

std::uint64_t checked_u64_add(
    std::uint64_t lhs, std::uint64_t rhs, const char* label) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::length_error(std::string(label) + " work overflow");
    }
    return lhs + rhs;
}

bool finite(const Vec3& value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double stable_norm(const Vec3& value) noexcept {
    return std::hypot(value.x, value.y, value.z);
}

double elapsed(const Clock::time_point& begin, const Clock::time_point& end) {
    return std::chrono::duration<double>(end - begin).count();
}

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() noexcept {
        std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    double unit() noexcept {
        return static_cast<double>(next() >> 11U) * 0x1.0p-53;
    }

    double symmetric() noexcept { return 2.0 * unit() - 1.0; }

private:
    std::uint64_t state_;
};

std::uint64_t bits(double value) noexcept {
    std::uint64_t result = 0;
    static_assert(sizeof(result) == sizeof(value), "binary64 storage is required");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

struct PositionKey {
    std::uint64_t x;
    std::uint64_t y;
    std::uint64_t z;

    bool operator==(const PositionKey& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct PositionHash {
    std::size_t operator()(const PositionKey& value) const noexcept {
        std::uint64_t hash = value.x ^ (value.y + 0x9e3779b97f4a7c15ULL +
            (value.x << 6U) + (value.x >> 2U));
        hash ^= value.z + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        return static_cast<std::size_t>(hash);
    }
};

constexpr std::size_t conservative_vector_overhead_bytes = 64;
constexpr std::size_t conservative_name_allocation_bytes = 32;
constexpr std::size_t conservative_hash_entry_overhead_bytes = 128;
constexpr std::size_t conservative_serialized_row_bytes = 4096;

std::size_t vector_storage(
    std::size_t count, std::size_t element_size, const char* label) {
    return checked_add(
        checked_multiply(count, element_size, label),
        conservative_vector_overhead_bytes,
        label);
}

std::size_t state_storage(std::size_t count) {
    return checked_add(
        vector_storage(count, sizeof(Body), "body vector"),
        checked_multiply(
            count, conservative_name_allocation_bytes, "body-name allocation reserve"),
        "state storage");
}

std::size_t repeated_storage(
    std::size_t count, std::size_t one, const char* label) {
    return checked_multiply(count, one, label);
}

struct PhaseMemoryEstimates {
    std::size_t generation{0};
    std::size_t direct_reference{0};
    std::size_t warmup{0};
    std::size_t retained_repetition{0};
    std::size_t characterization{0};
    std::size_t serialization{0};

    std::size_t high_water() const noexcept {
        return std::max({generation, direct_reference, warmup,
                         retained_repetition, characterization, serialization});
    }
};

PhaseMemoryEstimates estimate_phase_memory(
    const LargeNExperimentConfiguration& configuration) {
    const std::size_t count = configuration.generation.body_count;
    const std::size_t state = state_storage(count);
    const std::size_t vec3 = vector_storage(count, sizeof(Vec3), "Vec3 vector");
    const std::size_t doubles = vector_storage(count, sizeof(double), "double vector");
    const std::size_t measurements = checked_add(
        vector_storage(configuration.repetitions,
                       sizeof(LargeNRepetitionMeasurement),
                       "measurement vector"),
        checked_multiply(configuration.repetitions, 64U,
                         "measurement string reserve"),
        "measurement storage");
    const std::size_t solver = configuration.solver == LargeNSolver::barnes_hut
        ? configuration.barnes_hut.memory_budget_bytes
        : 0;
    const bool reference = configuration.request_direct_reference &&
        count <= configuration.resources.maximum_direct_reference_bodies;

    PhaseMemoryEstimates result;
    const std::size_t uniqueness_entry = checked_add(
        sizeof(PositionKey), conservative_hash_entry_overhead_bytes,
        "generator uniqueness entry");
    result.generation = checked_add(
        state,
        checked_add(
            repeated_storage(count, uniqueness_entry, "generator uniqueness storage"),
            conservative_vector_overhead_bytes + 512,
            "generator container storage"),
        "generation high water");

    result.direct_reference = checked_add(
        repeated_storage(reference ? 2U : 1U, state, "reference states"),
        checked_add(
            repeated_storage(reference ? 3U : 0U, vec3, "reference force arrays"),
            measurements,
            "reference auxiliary storage"),
        "direct-reference high water");

    result.warmup = checked_add(
        repeated_storage(reference ? 3U : 2U, state, "warmup states"),
        checked_add(
            repeated_storage(reference ? 4U : 3U, vec3, "warmup force arrays"),
            checked_add(solver, measurements, "warmup solver/result storage"),
            "warmup auxiliary storage"),
        "warmup high water");

    result.retained_repetition = checked_add(
        repeated_storage(reference ? 4U : 3U, state, "retained states"),
        checked_add(
            repeated_storage(reference ? 5U : 4U, vec3, "retained force arrays"),
            checked_add(solver, measurements, "retained solver/result storage"),
            "retained auxiliary storage"),
        "retained-repetition high water");

    result.characterization = checked_add(
        repeated_storage(reference ? 3U : 1U, state, "characterization states"),
        checked_add(
            repeated_storage(reference ? 2U : 0U, vec3, "characterization vectors"),
            checked_add(reference ? doubles : 0U, measurements,
                        "characterization scratch/results"),
            "characterization auxiliary storage"),
        "characterization high water");

    result.serialization = checked_add(
        measurements,
        checked_add(
            checked_multiply(configuration.repetitions,
                             conservative_serialized_row_bytes,
                             "serialized row reserve"),
            8192U,
            "serialization fixed reserve"),
        "serialization high water");
    return result;
}

PositionKey position_key(const Vec3& value) noexcept {
    return {bits(value.x), bits(value.y), bits(value.z)};
}

void validate_generation_configuration(const LargeNGenerationConfiguration& configuration) {
    if (configuration.body_count < 2) {
        throw std::invalid_argument("large-N generation requires at least two bodies");
    }
    if (!configuration.seed.has_value()) {
        throw std::invalid_argument("large-N generation requires an explicit seed");
    }
    if (!(configuration.position_scale_m > 0.0) ||
        !std::isfinite(configuration.position_scale_m)) {
        throw std::invalid_argument("position scale must be finite and positive");
    }
    if (!(configuration.velocity_scale_m_s >= 0.0) ||
        !std::isfinite(configuration.velocity_scale_m_s)) {
        throw std::invalid_argument("velocity scale must be finite and nonnegative");
    }
    if (!(configuration.minimum_mass_kg > 0.0) ||
        !std::isfinite(configuration.minimum_mass_kg) ||
        !(configuration.maximum_mass_kg >= configuration.minimum_mass_kg) ||
        !std::isfinite(configuration.maximum_mass_kg)) {
        throw std::invalid_argument("mass bounds must be finite, positive, and ordered");
    }
}

Vec3 generated_position(
    SplitMix64& random,
    LargeNDistribution distribution,
    std::size_t index,
    double scale) {
    if (distribution == LargeNDistribution::uniform) {
        return {scale * random.symmetric(), scale * random.symmetric(),
                scale * random.symmetric()};
    }
    if (distribution == LargeNDistribution::clustered) {
        constexpr double centers[4][3] = {
            {-0.55, -0.25, 0.15}, {0.48, -0.38, -0.20},
            {-0.20, 0.55, -0.35}, {0.42, 0.40, 0.38},
        };
        const auto& center = centers[index % 4U];
        constexpr double radius = 0.08;
        return {
            scale * (center[0] + radius * random.symmetric()),
            scale * (center[1] + radius * random.symmetric()),
            scale * (center[2] + radius * random.symmetric()),
        };
    }
    constexpr double thickness = 0.01;
    return {scale * random.symmetric(), scale * random.symmetric(),
            scale * thickness * random.symmetric()};
}

void ensure_finite_state(const SimulationState& state) {
    if (!std::isfinite(state.time_s)) {
        throw std::domain_error("large-N state time is non-finite");
    }
    std::unordered_set<PositionKey, PositionHash> positions;
    positions.reserve(state.bodies.size());
    for (const auto& body : state.bodies) {
        if (body.name.empty() || !(body.mass_kg > 0.0) ||
            !std::isfinite(body.mass_kg) || !finite(body.position_m) ||
            !finite(body.velocity_m_s)) {
            throw std::domain_error("large-N state contains invalid body data");
        }
        if (!positions.insert(position_key(body.position_m)).second) {
            throw std::domain_error("large-N generator produced coincident bodies");
        }
    }
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())));
    return values[std::max<std::size_t>(1, rank) - 1];
}

double relative_change(double initial, double final) {
    if (!std::isfinite(initial) || !std::isfinite(final)) {
        throw std::domain_error("scalar diagnostic is non-finite");
    }
    const double difference = final - initial;
    const double denominator = std::max(std::abs(initial), 1.0);
    const double result = std::abs(difference) / denominator;
    if (!std::isfinite(difference) || !std::isfinite(denominator) ||
        !(denominator > 0.0) || !std::isfinite(result)) {
        throw std::domain_error("scalar diagnostic drift is nonrepresentable");
    }
    return result;
}

double relative_change(const Vec3& initial, const Vec3& final) {
    if (!finite(initial) || !finite(final)) {
        throw std::domain_error("vector diagnostic is non-finite");
    }
    const double difference = stable_norm(final - initial);
    const double denominator = std::max(stable_norm(initial), 1.0);
    const double result = difference / denominator;
    if (!std::isfinite(difference) || !std::isfinite(denominator) ||
        !(denominator > 0.0) || !std::isfinite(result)) {
        throw std::domain_error("vector diagnostic drift is nonrepresentable");
    }
    return result;
}

void validate_diagnostics(const Diagnostics& diagnostics, const char* label) {
    if (!std::isfinite(diagnostics.kinetic_energy_j) ||
        !std::isfinite(diagnostics.potential_energy_j) ||
        !std::isfinite(diagnostics.total_energy_j()) ||
        !finite(diagnostics.linear_momentum_kg_m_s) ||
        !finite(diagnostics.angular_momentum_kg_m2_s) ||
        !finite(diagnostics.center_of_mass_m)) {
        throw std::domain_error(std::string(label) + " diagnostics are nonrepresentable");
    }
}

std::pair<double, double> separation_summary(
    const SimulationState& reference,
    const SimulationState& candidate,
    bool positions) {
    if (reference.bodies.size() != candidate.bodies.size() || reference.bodies.empty()) {
        throw std::invalid_argument("trajectory states must have equal nonzero body counts");
    }
    double maximum = 0.0;
    long double scaled_sum = 0.0L;
    long double scale = 0.0L;
    for (std::size_t index = 0; index < reference.bodies.size(); ++index) {
        const Vec3 difference = positions
            ? candidate.bodies[index].position_m - reference.bodies[index].position_m
            : candidate.bodies[index].velocity_m_s - reference.bodies[index].velocity_m_s;
        const double magnitude = stable_norm(difference);
        if (!std::isfinite(magnitude)) {
            throw std::domain_error("trajectory difference is non-finite");
        }
        maximum = std::max(maximum, magnitude);
        if (magnitude > static_cast<double>(scale)) {
            const long double ratio = scale == 0.0L ? 0.0L : scale / magnitude;
            scaled_sum = scaled_sum * ratio * ratio + 1.0L;
            scale = magnitude;
        } else if (scale != 0.0L) {
            const long double ratio = magnitude / scale;
            scaled_sum += ratio * ratio;
        }
    }
    const double rms = scale == 0.0L ? 0.0 : static_cast<double>(
        scale * std::sqrt(scaled_sum / reference.bodies.size()));
    return {maximum, rms};
}

struct BackendSelection {
    std::unique_ptr<ForceBackend> owned;
    const ForceBackend* backend{nullptr};
    SequentialBarnesHutForceBackend* sequential_bh{nullptr};
    OpenMpBarnesHutForceBackend* openmp_bh{nullptr};
};

BackendSelection make_backend(const LargeNExperimentConfiguration& configuration) {
    BackendSelection selection;
    if (configuration.backend == LargeNBackend::sequential &&
        configuration.solver == LargeNSolver::direct) {
        selection.backend = &sequential_force_backend();
        return selection;
    }
    if (configuration.backend == LargeNBackend::sequential) {
        auto backend = std::make_unique<SequentialBarnesHutForceBackend>(
            configuration.barnes_hut);
        selection.sequential_bh = backend.get();
        selection.backend = backend.get();
        selection.owned = std::move(backend);
        return selection;
    }
#ifdef NBODY_LARGE_N_HAS_OPENMP
    if (configuration.backend == LargeNBackend::openmp &&
        configuration.solver == LargeNSolver::direct) {
        auto backend = std::make_unique<OpenMpForceBackend>(configuration.requested_threads);
        selection.backend = backend.get();
        selection.owned = std::move(backend);
        return selection;
    }
    if (configuration.backend == LargeNBackend::openmp) {
        auto backend = std::make_unique<OpenMpBarnesHutForceBackend>(
            configuration.requested_threads, configuration.barnes_hut);
        selection.openmp_bh = backend.get();
        selection.backend = backend.get();
        selection.owned = std::move(backend);
        return selection;
    }
#endif
    throw std::runtime_error("requested large-N backend is unavailable");
}

BarnesHutStatistics backend_statistics(const BackendSelection& selection) {
    if (selection.sequential_bh != nullptr) {
        return selection.sequential_bh->last_statistics();
    }
    if (selection.openmp_bh != nullptr) {
        return selection.openmp_bh->last_statistics();
    }
    return {};
}

void check_wall_budget(
    const Clock::time_point& begin,
    double budget_seconds,
    const char* boundary,
    const LargeNExecutionControl& control) {
    double observed = elapsed(begin, Clock::now());
    if (control.boundary_elapsed_override) {
        const double overridden = control.boundary_elapsed_override(boundary, observed);
        if (!std::isfinite(overridden) || overridden < observed) {
            throw std::logic_error(
                "large-N boundary elapsed override must be finite and nondecreasing");
        }
        observed = overridden;
    }
    if (observed > budget_seconds) {
        throw std::runtime_error(
            std::string("large-N wall-time budget exceeded after ") + boundary);
    }
}

std::string csv_escape(std::string_view value) {
    if (value.find_first_of(",\"\r\n") == std::string_view::npos) {
        return std::string(value);
    }
    std::string result{"\""};
    for (const char character : value) {
        if (character == '\"') {
            result += "\"\"";
        } else {
            result += character;
        }
    }
    result += '\"';
    return result;
}

}  // namespace

std::string_view large_n_name(LargeNDistribution value) noexcept {
    switch (value) {
        case LargeNDistribution::uniform: return "uniform";
        case LargeNDistribution::clustered: return "clustered";
        case LargeNDistribution::planar: return "planar";
    }
    return "unknown";
}

std::string_view large_n_name(LargeNBackend value) noexcept {
    switch (value) {
        case LargeNBackend::sequential: return "sequential";
        case LargeNBackend::openmp: return "openmp";
        case LargeNBackend::cuda: return "cuda";
    }
    return "unknown";
}

std::string_view large_n_name(LargeNSolver value) noexcept {
    switch (value) {
        case LargeNSolver::direct: return "direct";
        case LargeNSolver::barnes_hut: return "barnes-hut";
    }
    return "unknown";
}

SimulationState make_large_n_synthetic_system(
    const LargeNGenerationConfiguration& configuration) {
    validate_generation_configuration(configuration);
    (void)checked_multiply(configuration.body_count, sizeof(Body), "body storage");
    SplitMix64 random(*configuration.seed);
    SimulationState state;
    state.bodies.reserve(configuration.body_count);
    const double mass_range = configuration.maximum_mass_kg - configuration.minimum_mass_kg;
    std::unordered_set<PositionKey, PositionHash> positions;
    positions.reserve(configuration.body_count);
    for (std::size_t index = 0; index < configuration.body_count; ++index) {
        Vec3 position;
        bool unique = false;
        for (int attempt = 0; attempt < 16 && !unique; ++attempt) {
            position = generated_position(
                random, configuration.distribution, index,
                configuration.position_scale_m);
            unique = positions.insert(position_key(position)).second;
        }
        if (!unique) {
            throw std::domain_error("could not generate a unique finite body position");
        }
        Body body;
        std::ostringstream identity;
        identity << "large_n_" << std::setw(8) << std::setfill('0') << index;
        body.name = identity.str();
        body.mass_kg = configuration.minimum_mass_kg + mass_range * random.unit();
        body.position_m = position;
        body.velocity_m_s = {
            configuration.velocity_scale_m_s * random.symmetric(),
            configuration.velocity_scale_m_s * random.symmetric(),
            configuration.velocity_scale_m_s * random.symmetric(),
        };
        state.bodies.push_back(std::move(body));
    }

    double total_mass = 0.0;
    Vec3 weighted_position;
    Vec3 momentum;
    for (const auto& body : state.bodies) {
        total_mass += body.mass_kg;
        weighted_position += body.position_m * body.mass_kg;
        momentum += body.velocity_m_s * body.mass_kg;
    }
    if (!(total_mass > 0.0) || !std::isfinite(total_mass) ||
        !finite(weighted_position) || !finite(momentum)) {
        throw std::domain_error("large-N center-of-mass reduction is nonrepresentable");
    }
    const Vec3 center = weighted_position / total_mass;
    const Vec3 center_velocity = momentum / total_mass;
    for (auto& body : state.bodies) {
        body.position_m -= center;
        body.velocity_m_s -= center_velocity;
    }
    ensure_finite_state(state);
    return state;
}

bool large_n_openmp_compiled() noexcept {
#ifdef NBODY_LARGE_N_HAS_OPENMP
    return true;
#else
    return false;
#endif
}

LargeNPreflight preflight_large_n_experiment(
    const LargeNExperimentConfiguration& configuration) {
    LargeNPreflight result;
    try {
        validate_generation_configuration(configuration.generation);
        const auto& policy = configuration.resources;
        if (configuration.generation.body_count > policy.maximum_body_count) {
            result.reason = "body count exceeds the configured laboratory limit";
            return result;
        }
        if (configuration.steps == 0 || configuration.steps > policy.maximum_steps ||
            configuration.warmups > policy.maximum_warmups ||
            configuration.repetitions == 0 ||
            configuration.repetitions > policy.maximum_repetitions) {
            result.reason = "step/warmup/repetition count exceeds the configured limit";
            return result;
        }
        if (!(configuration.timestep_s > 0.0) ||
            !std::isfinite(configuration.timestep_s)) {
            result.reason = "timestep must be finite and positive";
            return result;
        }
        if (!(policy.wall_time_budget_seconds > 0.0) ||
            !std::isfinite(policy.wall_time_budget_seconds)) {
            result.reason = "wall-time budget must be finite and positive";
            return result;
        }
        if (configuration.backend == LargeNBackend::cuda) {
            result.reason = "CUDA execution is metadata-only and unavailable in L1";
            return result;
        }
        if (configuration.solver == LargeNSolver::barnes_hut) {
            (void)SequentialBarnesHutForceBackend(configuration.barnes_hut);
        }
        if (configuration.backend == LargeNBackend::openmp) {
            if (!large_n_openmp_compiled()) {
                result.reason = "OpenMP backend is not compiled";
                return result;
            }
#ifdef NBODY_LARGE_N_HAS_OPENMP
            result.effective_threads = openmp_effective_thread_count(
                configuration.requested_threads);
            if (result.effective_threads != configuration.requested_threads) {
                result.reason = "OpenMP runtime did not provide the requested thread count";
                return result;
            }
#endif
        } else {
            if (configuration.requested_threads != 1) {
                result.reason = "Sequential execution requires exactly one thread";
                return result;
            }
            result.effective_threads = 1;
        }

        const PhaseMemoryEstimates memory = estimate_phase_memory(configuration);
        result.generation_memory_bytes = memory.generation;
        result.direct_reference_memory_bytes = memory.direct_reference;
        result.warmup_memory_bytes = memory.warmup;
        result.retained_repetition_memory_bytes = memory.retained_repetition;
        result.characterization_memory_bytes = memory.characterization;
        result.serialization_memory_bytes = memory.serialization;
        result.estimated_memory_bytes = memory.high_water();
        if (result.estimated_memory_bytes > policy.memory_budget_bytes) {
            result.reason = "conservative memory estimate exceeds the configured budget";
            return result;
        }

        const std::uint64_t count = configuration.generation.body_count;
        const std::uint64_t pair_work = checked_u64_multiply(
            count, count - 1U, "direct interaction");
        const std::uint64_t calls_per_execution = checked_u64_add(
            1U, checked_u64_multiply(2U, configuration.steps, "force call"),
            "force call");
        const std::uint64_t executions = checked_u64_add(
            configuration.warmups, configuration.repetitions, "execution count");
        if (configuration.solver == LargeNSolver::direct) {
            result.estimated_direct_interactions = checked_u64_multiply(
                pair_work,
                checked_u64_multiply(calls_per_execution, executions, "direct execution"),
                "direct interaction");
        }

        result.direct_reference_available = configuration.request_direct_reference &&
            configuration.generation.body_count <= policy.maximum_direct_reference_bodies;
        if (!configuration.request_direct_reference) {
            result.direct_reference_reason = "direct reference disabled by configuration";
        } else if (!result.direct_reference_available) {
            result.direct_reference_reason = "body count exceeds direct-reference limit";
        }
        if (result.direct_reference_available) {
            const std::uint64_t reference_calls = checked_u64_add(
                1U, checked_u64_multiply(2U, configuration.steps, "reference force call"),
                "reference force call");
            result.estimated_direct_interactions = checked_u64_add(
                result.estimated_direct_interactions,
                checked_u64_multiply(pair_work, reference_calls, "reference interaction"),
                "total direct interaction");
            const std::uint64_t diagnostic_pairs = checked_u64_multiply(
                pair_work / 2U, 3U, "diagnostic interaction");
            result.estimated_direct_interactions = checked_u64_add(
                result.estimated_direct_interactions,
                diagnostic_pairs,
                "total diagnostic interaction");
        }
        if (result.estimated_direct_interactions > policy.maximum_direct_interactions) {
            result.reason = "direct/reference interaction estimate exceeds the configured budget";
            return result;
        }
        result.admitted = true;
        result.reason = "admitted";
        return result;
    } catch (const std::exception& error) {
        result.reason = error.what();
        return result;
    }
}

LargeNAccelerationErrorSummary summarize_large_n_acceleration_errors(
    const std::vector<Vec3>& reference,
    const std::vector<Vec3>& candidate) {
    if (reference.empty() || reference.size() != candidate.size()) {
        throw std::invalid_argument("acceleration arrays must have equal nonzero sizes");
    }
    LargeNAccelerationErrorSummary result;
    result.available = true;
    double maximum_reference = 0.0;
    for (const auto& value : reference) {
        const double magnitude = stable_norm(value);
        if (!std::isfinite(magnitude)) {
            throw std::domain_error("reference acceleration is non-finite");
        }
        maximum_reference = std::max(maximum_reference, magnitude);
    }
    result.relative_denominator_floor_m_s2 = std::max(
        maximum_reference * 1.0e-12, std::numeric_limits<double>::min());
    std::vector<double> relative;
    relative.reserve(reference.size());
    long double scale = 0.0L;
    long double scaled_sum = 0.0L;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double reference_magnitude = stable_norm(reference[index]);
        const double absolute = stable_norm(candidate[index] - reference[index]);
        if (!std::isfinite(absolute)) {
            throw std::domain_error("candidate acceleration difference is non-finite");
        }
        result.maximum_absolute_m_s2 = std::max(result.maximum_absolute_m_s2, absolute);
        if (absolute > static_cast<double>(scale)) {
            const long double ratio = scale == 0.0L ? 0.0L : scale / absolute;
            scaled_sum = scaled_sum * ratio * ratio + 1.0L;
            scale = absolute;
        } else if (scale != 0.0L) {
            const long double ratio = absolute / scale;
            scaled_sum += ratio * ratio;
        }
        if (reference_magnitude <= result.relative_denominator_floor_m_s2) {
            ++result.cancellation_sensitive_count;
        }
        const double relative_error = absolute / std::max(
            reference_magnitude, result.relative_denominator_floor_m_s2);
        if (!std::isfinite(relative_error)) {
            throw std::domain_error("relative acceleration error is nonrepresentable");
        }
        result.maximum_relative = std::max(result.maximum_relative, relative_error);
        relative.push_back(relative_error);
    }
    result.rms_absolute_m_s2 = scale == 0.0L ? 0.0 : static_cast<double>(
        scale * std::sqrt(scaled_sum / reference.size()));
    result.p50_relative = percentile(relative, 0.50);
    result.p95_relative = percentile(relative, 0.95);
    result.p99_relative = percentile(relative, 0.99);
    return result;
}

LargeNExperimentResult run_large_n_experiment(
    const LargeNExperimentConfiguration& configuration,
    const LargeNExecutionControl& control) {
    LargeNExperimentResult result;
    result.configuration = configuration;
    result.preflight = preflight_large_n_experiment(configuration);
    if (!result.preflight.admitted) {
        result.status = "skipped";
        result.message = result.preflight.reason;
        return result;
    }

    const auto experiment_begin = Clock::now();
    try {
        const auto generation_begin = Clock::now();
        const SimulationState initial = make_large_n_synthetic_system(configuration.generation);
        result.generation_seconds = elapsed(generation_begin, Clock::now());
        check_wall_budget(experiment_begin, configuration.resources.wall_time_budget_seconds,
                          "input generation", control);

        BackendSelection selected = make_backend(configuration);
        std::vector<Vec3> direct_reference;
        SimulationState direct_trajectory;
        Diagnostics initial_diagnostics;
        Diagnostics direct_final_diagnostics;
        if (result.preflight.direct_reference_available) {
            const auto initial_diagnostics_begin = Clock::now();
            initial_diagnostics = compute_diagnostics(initial);
            validate_diagnostics(initial_diagnostics, "initial");
            result.diagnostics_seconds += elapsed(initial_diagnostics_begin, Clock::now());
            const auto direct_begin = Clock::now();
            direct_reference = compute_accelerations(initial);
            result.initial_direct_force_seconds = elapsed(direct_begin, Clock::now());
            direct_trajectory = initial;
            const auto direct_trajectory_begin = Clock::now();
            for (std::size_t step = 0; step < configuration.steps; ++step) {
                velocity_verlet_step(direct_trajectory, configuration.timestep_s);
            }
            result.direct_trajectory_seconds = elapsed(
                direct_trajectory_begin, Clock::now());
            const auto direct_diagnostics_begin = Clock::now();
            direct_final_diagnostics = compute_diagnostics(direct_trajectory);
            validate_diagnostics(direct_final_diagnostics, "Direct final");
            result.diagnostics_seconds += elapsed(direct_diagnostics_begin, Clock::now());
        } else {
            result.acceleration_error.unavailable_reason =
                result.preflight.direct_reference_reason;
            result.trajectory_accuracy.unavailable_reason =
                result.preflight.direct_reference_reason;
        }
        check_wall_budget(experiment_begin, configuration.resources.wall_time_budget_seconds,
                          "direct reference", control);

        for (std::size_t warmup = 0; warmup < configuration.warmups; ++warmup) {
            (void)selected.backend->compute_accelerations(initial, gravitational_constant);
            auto warmed = initial;
            for (std::size_t step = 0; step < configuration.steps; ++step) {
                velocity_verlet_step(
                    warmed, configuration.timestep_s, *selected.backend);
            }
            ensure_finite_state(warmed);
            check_wall_budget(experiment_begin,
                              configuration.resources.wall_time_budget_seconds,
                              "warmup", control);
        }

        SimulationState first_trajectory;
        std::vector<Vec3> first_acceleration;
        result.measurements.reserve(configuration.repetitions);
        for (std::size_t repetition = 0; repetition < configuration.repetitions;
             ++repetition) {
            const auto force_begin = Clock::now();
            auto acceleration = selected.backend->compute_accelerations(
                initial, gravitational_constant);
            const double force_seconds = elapsed(force_begin, Clock::now());
            for (const auto& value : acceleration) {
                if (!finite(value)) {
                    throw std::domain_error("large-N force result is non-finite");
                }
            }
            const BarnesHutStatistics statistics = backend_statistics(selected);
            if (configuration.backend == LargeNBackend::openmp &&
                configuration.solver == LargeNSolver::barnes_hut &&
                statistics.worker_threads != result.preflight.effective_threads) {
                throw std::runtime_error(
                    "OpenMP Barnes-Hut worker count changed after admission");
            }
            auto advanced = initial;
            const auto step_begin = Clock::now();
            for (std::size_t step = 0; step < configuration.steps; ++step) {
                velocity_verlet_step(
                    advanced, configuration.timestep_s, *selected.backend);
            }
            const double step_seconds = elapsed(step_begin, Clock::now());
            ensure_finite_state(advanced);
            if (repetition == 0) {
                first_acceleration = std::move(acceleration);
                first_trajectory = std::move(advanced);
            }
            result.measurements.push_back({
                repetition,
                configuration.backend == LargeNBackend::sequential
                    ? 1
                    : (configuration.solver == LargeNSolver::barnes_hut
                           ? statistics.worker_threads
                           : 0),
                configuration.backend == LargeNBackend::sequential
                    ? "serial"
                    : (configuration.solver == LargeNSolver::barnes_hut
                           ? "observed_in_timed_region"
                           : "runtime_probe_only"),
                force_seconds,
                step_seconds,
                statistics.tree_build_seconds,
                statistics.force_evaluation_seconds,
                statistics.working_memory_bytes,
                statistics.node_count,
                statistics.visited_nodes,
                statistics.approximated_nodes,
                statistics.direct_body_interactions,
            });
            check_wall_budget(experiment_begin,
                              configuration.resources.wall_time_budget_seconds,
                              "retained repetition", control);
        }

        if (result.preflight.direct_reference_available) {
            result.acceleration_error = summarize_large_n_acceleration_errors(
                direct_reference, first_acceleration);
            result.trajectory_accuracy.available = true;
            const auto position = separation_summary(direct_trajectory, first_trajectory, true);
            const auto velocity = separation_summary(direct_trajectory, first_trajectory, false);
            result.trajectory_accuracy.maximum_position_separation_m = position.first;
            result.trajectory_accuracy.rms_position_separation_m = position.second;
            result.trajectory_accuracy.maximum_velocity_separation_m_s = velocity.first;
            result.trajectory_accuracy.rms_velocity_separation_m_s = velocity.second;
            const auto candidate_diagnostics_begin = Clock::now();
            const Diagnostics candidate_final = compute_diagnostics(first_trajectory);
            validate_diagnostics(candidate_final, "candidate final");
            result.diagnostics_seconds += elapsed(candidate_diagnostics_begin, Clock::now());
            result.trajectory_accuracy.candidate_relative_energy_drift = relative_change(
                initial_diagnostics.total_energy_j(), candidate_final.total_energy_j());
            result.trajectory_accuracy.direct_relative_energy_drift = relative_change(
                initial_diagnostics.total_energy_j(), direct_final_diagnostics.total_energy_j());
            result.trajectory_accuracy.candidate_relative_angular_momentum_drift =
                relative_change(initial_diagnostics.angular_momentum_kg_m2_s,
                                candidate_final.angular_momentum_kg_m2_s);
            result.trajectory_accuracy.direct_relative_angular_momentum_drift =
                relative_change(initial_diagnostics.angular_momentum_kg_m2_s,
                                direct_final_diagnostics.angular_momentum_kg_m2_s);
        }
        check_wall_budget(experiment_begin,
                          configuration.resources.wall_time_budget_seconds,
                          "final scientific characterization", control);
        result.total_experiment_seconds = elapsed(experiment_begin, Clock::now());
        const auto finite_nonnegative = [](double value) {
            return std::isfinite(value) && value >= 0.0;
        };
        if (!finite_nonnegative(result.generation_seconds) ||
            !finite_nonnegative(result.initial_direct_force_seconds) ||
            !finite_nonnegative(result.direct_trajectory_seconds) ||
            !finite_nonnegative(result.diagnostics_seconds) ||
            !finite_nonnegative(result.total_experiment_seconds)) {
            throw std::domain_error("large-N published timing is nonrepresentable");
        }
        for (const auto& measurement : result.measurements) {
            if (!finite_nonnegative(measurement.complete_force_seconds) ||
                !finite_nonnegative(measurement.complete_timestep_seconds) ||
                !finite_nonnegative(measurement.tree_build_seconds) ||
                !finite_nonnegative(measurement.force_traversal_seconds)) {
                throw std::domain_error("large-N measurement timing is nonrepresentable");
            }
        }
        if (result.acceleration_error.available) {
            const auto& value = result.acceleration_error;
            if (!finite_nonnegative(value.maximum_absolute_m_s2) ||
                !finite_nonnegative(value.rms_absolute_m_s2) ||
                !finite_nonnegative(value.p50_relative) ||
                !finite_nonnegative(value.p95_relative) ||
                !finite_nonnegative(value.p99_relative) ||
                !finite_nonnegative(value.maximum_relative) ||
                !(value.relative_denominator_floor_m_s2 > 0.0) ||
                !std::isfinite(value.relative_denominator_floor_m_s2)) {
                throw std::domain_error("large-N force accuracy is nonrepresentable");
            }
        }
        if (result.trajectory_accuracy.available) {
            const auto& value = result.trajectory_accuracy;
            if (!finite_nonnegative(value.maximum_position_separation_m) ||
                !finite_nonnegative(value.rms_position_separation_m) ||
                !finite_nonnegative(value.maximum_velocity_separation_m_s) ||
                !finite_nonnegative(value.rms_velocity_separation_m_s) ||
                !finite_nonnegative(value.candidate_relative_energy_drift) ||
                !finite_nonnegative(value.direct_relative_energy_drift) ||
                !finite_nonnegative(value.candidate_relative_angular_momentum_drift) ||
                !finite_nonnegative(value.direct_relative_angular_momentum_drift)) {
                throw std::domain_error("large-N trajectory accuracy is nonrepresentable");
            }
        }
        result.status = "completed";
        result.message = "completed";
        return result;
    } catch (const std::exception& error) {
        result.measurements.clear();
        result.acceleration_error = {};
        result.acceleration_error.unavailable_reason =
            "experiment failed before transactional scientific publication";
        result.trajectory_accuracy = {};
        result.trajectory_accuracy.unavailable_reason =
            "experiment failed before transactional scientific publication";
        result.status = "failed";
        result.message = error.what();
        result.total_experiment_seconds = elapsed(experiment_begin, Clock::now());
        return result;
    }
}

std::string large_n_csv_header() {
    return "status,message,distribution,bodies,seed,backend,solver,theta,requested_threads,"
           "effective_threads,observed_worker_threads,worker_count_provenance,"
           "timestep_s,steps,warmups,repetitions,repetition,generation_s,"
           "initial_direct_force_s,direct_trajectory_s,diagnostics_s,"
           "complete_force_s,complete_timestep_s,total_experiment_s,tree_build_s,"
           "force_traversal_s,solver_memory_bytes,"
           "node_count,visited_nodes,approximated_nodes,direct_interactions,"
           "estimated_memory_bytes,generation_memory_bytes,direct_reference_memory_bytes,"
           "warmup_memory_bytes,retained_repetition_memory_bytes,"
           "characterization_memory_bytes,serialization_memory_bytes,"
           "estimated_direct_interactions,direct_reference_available,"
           "direct_reference_reason,max_abs_acceleration_error_m_s2,"
           "rms_abs_acceleration_error_m_s2,p50_relative_error,p95_relative_error,"
           "p99_relative_error,max_relative_error,relative_floor_m_s2,"
           "cancellation_sensitive_count,max_position_separation_m,"
           "rms_position_separation_m,max_velocity_separation_m_s,"
           "rms_velocity_separation_m_s,candidate_relative_energy_drift,"
           "direct_relative_energy_drift,candidate_relative_angular_momentum_drift,"
           "direct_relative_angular_momentum_drift,wall_time_budget_s,memory_budget_bytes,"
           "maximum_direct_interactions";
}

std::vector<std::string> serialize_large_n_csv_rows(
    const LargeNExperimentResult& result) {
    std::vector<LargeNRepetitionMeasurement> rows = result.measurements;
    if (rows.empty()) {
        rows.push_back({});
    }
    std::vector<std::string> serialized;
    serialized.reserve(rows.size());
    for (const auto& measurement : rows) {
        std::ostringstream output;
        output << std::setprecision(17)
               << csv_escape(result.status) << ',' << csv_escape(result.message) << ','
               << large_n_name(result.configuration.generation.distribution) << ','
               << result.configuration.generation.body_count << ','
               << result.configuration.generation.seed.value_or(0) << ','
               << large_n_name(result.configuration.backend) << ','
               << large_n_name(result.configuration.solver) << ','
               << result.configuration.barnes_hut.theta << ','
               << result.configuration.requested_threads << ','
               << result.preflight.effective_threads << ','
               << measurement.observed_worker_threads << ','
               << csv_escape(measurement.worker_count_provenance) << ','
               << result.configuration.timestep_s << ',' << result.configuration.steps << ','
               << result.configuration.warmups << ',' << result.configuration.repetitions << ','
               << measurement.repetition << ',' << result.generation_seconds << ','
               << result.initial_direct_force_seconds << ','
               << result.direct_trajectory_seconds << ','
               << result.diagnostics_seconds << ','
               << measurement.complete_force_seconds << ','
               << measurement.complete_timestep_seconds << ','
               << result.total_experiment_seconds << ','
               << measurement.tree_build_seconds << ','
               << measurement.force_traversal_seconds << ','
               << measurement.solver_working_memory_bytes << ','
               << measurement.node_count << ',' << measurement.visited_nodes << ','
               << measurement.approximated_nodes << ','
               << measurement.direct_interactions << ','
               << result.preflight.estimated_memory_bytes << ','
               << result.preflight.generation_memory_bytes << ','
               << result.preflight.direct_reference_memory_bytes << ','
               << result.preflight.warmup_memory_bytes << ','
               << result.preflight.retained_repetition_memory_bytes << ','
               << result.preflight.characterization_memory_bytes << ','
               << result.preflight.serialization_memory_bytes << ','
               << result.preflight.estimated_direct_interactions << ','
               << (result.preflight.direct_reference_available ? 1 : 0) << ','
               << csv_escape(result.preflight.direct_reference_reason) << ','
               << result.acceleration_error.maximum_absolute_m_s2 << ','
               << result.acceleration_error.rms_absolute_m_s2 << ','
               << result.acceleration_error.p50_relative << ','
               << result.acceleration_error.p95_relative << ','
               << result.acceleration_error.p99_relative << ','
               << result.acceleration_error.maximum_relative << ','
               << result.acceleration_error.relative_denominator_floor_m_s2 << ','
               << result.acceleration_error.cancellation_sensitive_count << ','
               << result.trajectory_accuracy.maximum_position_separation_m << ','
               << result.trajectory_accuracy.rms_position_separation_m << ','
               << result.trajectory_accuracy.maximum_velocity_separation_m_s << ','
               << result.trajectory_accuracy.rms_velocity_separation_m_s << ','
               << result.trajectory_accuracy.candidate_relative_energy_drift << ','
               << result.trajectory_accuracy.direct_relative_energy_drift << ','
               << result.trajectory_accuracy.candidate_relative_angular_momentum_drift << ','
               << result.trajectory_accuracy.direct_relative_angular_momentum_drift << ','
               << result.configuration.resources.wall_time_budget_seconds << ','
               << result.configuration.resources.memory_budget_bytes << ','
               << result.configuration.resources.maximum_direct_interactions;
        serialized.push_back(output.str());
    }
    return serialized;
}

}  // namespace nbody
