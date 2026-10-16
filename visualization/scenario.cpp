#include "nbody/visualization/application_transaction.hpp"

#include "nbody/astronomy/solar_system_60.hpp"
#include "nbody/scenarios.hpp"
#include "nbody/solar_system.hpp"
#include "nbody/visualization/workload.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace nbody::visualization {

namespace {

constexpr std::array<glm::vec3, 9> solar_colors{{
    {1.00F, 0.72F, 0.22F},  // Sun
    {0.55F, 0.54F, 0.52F},  // Mercury
    {0.88F, 0.73F, 0.46F},  // Venus
    {0.22F, 0.48F, 0.95F},  // Earth
    {0.86F, 0.31F, 0.16F},  // Mars
    {0.78F, 0.56F, 0.35F},  // Jupiter
    {0.88F, 0.76F, 0.48F},  // Saturn
    {0.35F, 0.82F, 0.88F},  // Uranus
    {0.18F, 0.32F, 0.92F},  // Neptune
}};

constexpr std::array<BodyMaterial, 9> solar_materials{{
    BodyMaterial::sun,
    BodyMaterial::mercury,
    BodyMaterial::venus,
    BodyMaterial::earth,
    BodyMaterial::mars,
    BodyMaterial::jupiter,
    BodyMaterial::saturn,
    BodyMaterial::uranus,
    BodyMaterial::neptune,
}};

constexpr std::array<float, 9> solar_visual_scales{{
    1.35F, 0.92F, 1.0F, 1.08F, 1.0F, 1.25F, 1.18F, 1.05F, 1.05F,
}};

constexpr std::array<glm::vec3, 8> synthetic_colors{{
    {0.95F, 0.75F, 0.30F}, {0.35F, 0.70F, 1.00F},
    {0.75F, 0.40F, 0.95F}, {0.35F, 0.90F, 0.62F},
    {0.95F, 0.42F, 0.42F}, {0.92F, 0.88F, 0.48F},
    {0.45F, 0.82F, 0.90F}, {0.92F, 0.58F, 0.82F},
}};

glm::vec3 catalog_color(const astronomy::AstronomicalBodyRecord& record) {
    const auto& id = record.stable_id();
    if (id == "sol:sun") return solar_colors[0];
    if (id == "sol:mercury") return solar_colors[1];
    if (id == "sol:venus") return solar_colors[2];
    if (id == "sol:earth") return solar_colors[3];
    if (id == "sol:mars") return solar_colors[4];
    if (id == "sol:jupiter") return solar_colors[5];
    if (id == "sol:saturn") return solar_colors[6];
    if (id == "sol:uranus") return solar_colors[7];
    if (id == "sol:neptune") return solar_colors[8];
    switch (record.body_class()) {
        case astronomy::AstronomicalBodyClass::star: return {1.00F, 0.72F, 0.22F};
        case astronomy::AstronomicalBodyClass::planet: return {0.52F, 0.72F, 0.96F};
        case astronomy::AstronomicalBodyClass::dwarf_planet: return {0.70F, 0.62F, 0.84F};
        case astronomy::AstronomicalBodyClass::natural_satellite: return {0.72F, 0.74F, 0.78F};
        case astronomy::AstronomicalBodyClass::asteroid: return {0.72F, 0.52F, 0.34F};
        case astronomy::AstronomicalBodyClass::comet: return {0.58F, 0.90F, 0.88F};
        case astronomy::AstronomicalBodyClass::trans_neptunian_object:
            return {0.66F, 0.58F, 0.82F};
    }
    return {0.75F, 0.75F, 0.75F};
}

BodyMaterial catalog_material(const std::string& id) {
    if (id == "sol:sun") return BodyMaterial::sun;
    if (id == "sol:mercury") return BodyMaterial::mercury;
    if (id == "sol:venus") return BodyMaterial::venus;
    if (id == "sol:earth") return BodyMaterial::earth;
    if (id == "sol:mars") return BodyMaterial::mars;
    if (id == "sol:jupiter") return BodyMaterial::jupiter;
    if (id == "sol:saturn") return BodyMaterial::saturn;
    if (id == "sol:uranus") return BodyMaterial::uranus;
    if (id == "sol:neptune") return BodyMaterial::neptune;
    return BodyMaterial::generic;
}

bool is_future_removal_eligible(astronomy::AstronomicalBodyClass body_class) {
    return body_class == astronomy::AstronomicalBodyClass::planet ||
        body_class == astronomy::AstronomicalBodyClass::dwarf_planet;
}

void append_legacy_metadata(ScenarioDefinition& result) {
    result.body_metadata.reserve(result.stable_ids.size());
    for (std::size_t index = 0; index < result.stable_ids.size(); ++index) {
        result.body_metadata.push_back({
            result.stable_ids[index],
            index == 0 ? std::optional<std::string>{} : std::optional<std::string>{"sun"},
            index == 0 ? astronomy::AstronomicalBodyClass::star
                       : astronomy::AstronomicalBodyClass::planet,
            index != 0,
        });
    }
}

// This is the presentation adapter for the committed Large-N generator. It
// deliberately preserves that generator's SplitMix64 sequence, distribution
// geometry, SI scales, mass order, identity format, and barycentric reduction.
// Keeping the adapter here avoids changing CMakeLists.txt, whose exact bytes are
// part of the immutable L1/L2 benchmark source custody.
class ViewerSplitMix64 {
public:
    explicit ViewerSplitMix64(std::uint64_t seed) : state_(seed) {}

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

std::uint64_t binary64_bits(double value) noexcept {
    std::uint64_t result = 0;
    static_assert(sizeof(result) == sizeof(value), "binary64 storage is required");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

Vec3 viewer_generated_position(
    ViewerSplitMix64& random,
    LargeNDistribution distribution,
    std::size_t index) {
    constexpr double scale = 1.0e9;
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
    if (distribution == LargeNDistribution::planar) {
        constexpr double thickness = 0.01;
        return {scale * random.symmetric(), scale * random.symmetric(),
                scale * thickness * random.symmetric()};
    }
    throw std::invalid_argument("unknown synthetic distribution");
}

SimulationState make_viewer_large_n_synthetic_system(
    std::size_t body_count,
    std::uint64_t seed,
    LargeNDistribution distribution) {
    ViewerSplitMix64 random(seed);
    SimulationState state;
    state.bodies.reserve(body_count);
    std::set<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> positions;
    for (std::size_t index = 0; index < body_count; ++index) {
        Vec3 position;
        bool unique = false;
        for (int attempt = 0; attempt < 16 && !unique; ++attempt) {
            position = viewer_generated_position(random, distribution, index);
            unique = positions.emplace(
                binary64_bits(position.x), binary64_bits(position.y),
                binary64_bits(position.z)).second;
        }
        if (!unique) {
            throw std::domain_error(
                "Large-N viewer generator could not produce a unique body position");
        }
        Body body;
        std::ostringstream identity;
        identity << "large_n_" << std::setw(8) << std::setfill('0') << index;
        body.name = identity.str();
        body.mass_kg = 1.0e20 + 9.0e20 * random.unit();
        body.position_m = position;
        body.velocity_m_s = {
            100.0 * random.symmetric(),
            100.0 * random.symmetric(),
            100.0 * random.symmetric(),
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
        !std::isfinite(weighted_position.x) || !std::isfinite(weighted_position.y) ||
        !std::isfinite(weighted_position.z) || !std::isfinite(momentum.x) ||
        !std::isfinite(momentum.y) || !std::isfinite(momentum.z)) {
        throw std::domain_error("Large-N viewer barycentric reduction is nonrepresentable");
    }
    const Vec3 center = weighted_position / total_mass;
    const Vec3 center_velocity = momentum / total_mass;
    for (auto& body : state.bodies) {
        body.position_m -= center;
        body.velocity_m_s -= center_velocity;
    }
    return state;
}

}  // namespace

const char* astronomical_body_class_name(
    astronomy::AstronomicalBodyClass body_class) noexcept {
    switch (body_class) {
        case astronomy::AstronomicalBodyClass::star: return "Star";
        case astronomy::AstronomicalBodyClass::planet: return "Planet";
        case astronomy::AstronomicalBodyClass::dwarf_planet: return "Dwarf planet";
        case astronomy::AstronomicalBodyClass::natural_satellite: return "Natural satellite";
        case astronomy::AstronomicalBodyClass::asteroid: return "Asteroid";
        case astronomy::AstronomicalBodyClass::comet: return "Comet";
        case astronomy::AstronomicalBodyClass::trans_neptunian_object:
            return "Trans-Neptunian object";
    }
    return "Unknown";
}

const char* synthetic_distribution_name(LargeNDistribution distribution) noexcept {
    switch (distribution) {
        case LargeNDistribution::uniform: return "uniform";
        case LargeNDistribution::clustered: return "clustered";
        case LargeNDistribution::planar: return "planar";
    }
    return "unknown";
}

bool valid_synthetic_distribution(LargeNDistribution distribution) noexcept {
    return distribution == LargeNDistribution::uniform ||
        distribution == LargeNDistribution::clustered ||
        distribution == LargeNDistribution::planar;
}

void validate_visual_scenario(const ScenarioDefinition& scenario) {
    const std::size_t body_count = scenario.initial_state.bodies.size();
    if (body_count == 0U || !std::isfinite(scenario.initial_state.time_s) ||
        !(scenario.dt_s > 0.0) || !std::isfinite(scenario.dt_s) ||
        scenario.speed_presets.empty() ||
        scenario.default_speed_index >= scenario.speed_presets.size() ||
        !(scenario.trail_sample_interval_s > 0.0) ||
        !std::isfinite(scenario.trail_sample_interval_s) ||
        scenario.stable_ids.size() != body_count ||
        scenario.body_metadata.size() != body_count ||
        scenario.styles.size() != body_count) {
        throw std::invalid_argument("visual scenario structure is invalid");
    }
    std::unordered_set<std::string> stable_ids;
    stable_ids.reserve(body_count);
    const auto finite_vector = [](const Vec3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) &&
            std::isfinite(value.z);
    };
    for (std::size_t index = 0; index < body_count; ++index) {
        const auto& body = scenario.initial_state.bodies[index];
        if (body.name.empty() || !(body.mass_kg > 0.0) ||
            !std::isfinite(body.mass_kg) || !finite_vector(body.position_m) ||
            !finite_vector(body.velocity_m_s) || scenario.stable_ids[index].empty() ||
            scenario.body_metadata[index].stable_id != scenario.stable_ids[index] ||
            !stable_ids.insert(scenario.stable_ids[index]).second) {
            throw std::invalid_argument("visual scenario body identity or state is invalid");
        }
    }
    if (scenario.kind == ScenarioKind::synthetic) {
        if (!is_supported_compute_body_count(body_count) ||
            !scenario.synthetic_distribution.has_value() ||
            !valid_synthetic_distribution(*scenario.synthetic_distribution) ||
            !scenario.synthetic_seed.has_value() || !render_resource_admitted(body_count)) {
            throw std::invalid_argument("synthetic visual scenario admission is invalid");
        }
    }
}

std::optional<std::size_t> find_body_index(
    const ScenarioDefinition& scenario,
    const std::string& stable_id) noexcept {
    const auto iterator = std::find(
        scenario.stable_ids.begin(), scenario.stable_ids.end(), stable_id);
    if (iterator == scenario.stable_ids.end()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(std::distance(scenario.stable_ids.begin(), iterator));
}

ScenarioDefinition make_astronomical_catalog_scenario(
    const std::vector<std::string>& included_stable_ids) {
    const auto catalog = astronomy::make_solar_system_60_catalog();
    std::set<std::string> requested;
    if (included_stable_ids.empty()) {
        for (const auto& record : catalog.records()) {
            requested.insert(record.stable_id());
        }
    } else {
        for (const auto& id : included_stable_ids) {
            if (!requested.insert(id).second) {
                throw std::invalid_argument("catalog subset contains duplicate stable ID: " + id);
            }
            if (catalog.find(id) == nullptr) {
                throw std::invalid_argument("catalog subset contains unknown stable ID: " + id);
            }
        }
    }
    if (requested.size() < 2U) {
        throw std::invalid_argument("catalog scenario requires at least two solver bodies");
    }

    ScenarioDefinition result;
    result.kind = ScenarioKind::astronomical_catalog;
    result.name = "Solar System Sandbox (Horizons 60)";
    result.provenance =
        "jpl-ssd-j2000-60-v1: geometric SSB-centered ICRF states at JD 2451545.0 TDB";
    result.dt_s = astronomical_catalog_initial_dt_s;
    result.speed_presets = {
        {"1 simulated minute per frame", 1},
        {"10 simulated minutes per frame", 10},
        {"30 simulated minutes per frame", 30},
        {"2 simulated hours per frame", 120},
    };
    result.default_speed_index = 1;
    result.trail_sample_interval_s = 1800.0;
    result.default_radius_multiplier = 1.0F;
    result.label_policy = LabelPolicy::all_bodies;
    result.timestep_validation_complete = false;
    result.scientific_limitations =
        "The 60 s fixed timestep is a convergence-informed provisional policy, not a validated long-term "
        "accuracy guarantee. Point masses omit relativity, oblateness, tides, non-gravitational "
        "comet forces, and ephemeris-fit perturbations; reference-ephemeris disagreement remains "
        "to be measured.";

    for (const auto& record : catalog.records()) {
        if (requested.count(record.stable_id()) == 0U) {
            continue;
        }
        result.initial_state.bodies.push_back({
            record.display_name(), record.mass_kg(), record.position_m(), record.velocity_m_s()});
        result.stable_ids.push_back(record.stable_id());
        result.body_metadata.push_back({
            record.stable_id(), record.parent_stable_id(), record.body_class(),
            is_future_removal_eligible(record.body_class()),
        });
        result.styles.push_back({
            catalog_color(record), record.mean_radius_m(),
            record.body_class() == astronomy::AstronomicalBodyClass::star,
            catalog_material(record.stable_id()), 1.0F,
        });
    }

    if (requested.count("sol:sun") == 0U) {
        result.composition_warnings.push_back(
            "Sun excluded: the dominant Solar System gravitational source is absent.");
    }
    constexpr std::array<const char*, 8> major_sources{{
        "sol:mercury", "sol:venus", "sol:earth", "sol:mars", "sol:jupiter",
        "sol:saturn", "sol:uranus", "sol:neptune"}};
    for (const auto* id : major_sources) {
        if (requested.count(id) == 0U) {
            result.composition_warnings.push_back(
                std::string("Major gravitational source excluded: ") + id);
        }
    }
    for (const auto& metadata : result.body_metadata) {
        if (metadata.parent_stable_id &&
            requested.count(*metadata.parent_stable_id) == 0U) {
            result.composition_warnings.push_back(
                metadata.stable_id + " included without catalog parent " +
                *metadata.parent_stable_id);
        }
    }
    return result;
}

ScenarioDefinition make_visual_scenario(
    ScenarioKind kind,
    std::size_t synthetic_body_count,
    std::uint64_t synthetic_seed,
    LargeNDistribution synthetic_distribution) {
    ScenarioDefinition result;
    result.kind = kind;

    if (kind == ScenarioKind::solar_system) {
        auto solar = make_approximate_solar_system();
        result.name = "Solar System (approximate J2000)";
        result.provenance =
            "JPL approximate J2000 elements; lower accuracy, not a Horizons ephemeris";
        result.initial_state = std::move(solar.initial_state);
        result.dt_s = solar_system_recommended_dt_s;
        result.speed_presets = {
            {"6 simulated hours per frame", 1},
            {"1 simulated day per frame", 4},
            {"4 simulated days per frame", 16},
            {"16 simulated days per frame", 64},
        };
        result.default_speed_index = 1;
        result.trail_sample_interval_s = 86400.0;
        result.default_radius_multiplier = 80.0F;
        result.reference_paths_m = make_approximate_solar_reference_orbits();
        result.label_policy = LabelPolicy::all_bodies;
        for (std::size_t index = 0; index < solar.metadata.size(); ++index) {
            result.stable_ids.push_back(solar.metadata[index].stable_id);
            result.styles.push_back({
                solar_colors[index],
                solar.metadata[index].mean_radius_m,
                index == 0,
                solar_materials[index],
                solar_visual_scales[index],
            });
        }
        append_legacy_metadata(result);
        return result;
    }

    if (kind == ScenarioKind::astronomical_catalog) {
        return make_astronomical_catalog_scenario();
    }

    if (kind != ScenarioKind::synthetic) {
        throw std::invalid_argument("unknown visualization scenario kind");
    }

    if (!is_supported_compute_body_count(synthetic_body_count)) {
        throw std::invalid_argument(
            "synthetic viewer body count must be one of 256, 1024, 4096, or 8192");
    }
    if (!valid_synthetic_distribution(synthetic_distribution)) {
        throw std::invalid_argument(
            "synthetic distribution must be uniform, clustered, or planar");
    }
    result.name = "Synthetic N-body Sandbox";
    result.provenance = "deterministic Large-N generator; distribution=" +
        std::string(synthetic_distribution_name(synthetic_distribution)) + "; seed=" +
        std::to_string(synthetic_seed) + "; not an astronomical ephemeris";
    result.initial_state = make_viewer_large_n_synthetic_system(
        synthetic_body_count, synthetic_seed, synthetic_distribution);
    result.synthetic_distribution = synthetic_distribution;
    result.synthetic_seed = synthetic_seed;
    result.dt_s = 10.0;
    result.timestep_validation_complete = false;
    result.scientific_limitations =
        "Synthetic deterministic point masses are a compute/visualization fixture, not an "
        "astronomical model. The 10-second timestep is a bounded demonstration setting, "
        "not a general accuracy guarantee.";
    const std::size_t balanced_steps = synthetic_body_count > 1024 ? 1 :
        synthetic_body_count <= 64 ? 32 :
        synthetic_body_count <= 128 ? 16 :
        synthetic_body_count <= 256 ? 8 :
        synthetic_body_count <= 512 ? 4 : 2;
    result.speed_presets = {
        {"Light compute batch", std::max<std::size_t>(1, balanced_steps / 4)},
        {"Balanced compute batch", balanced_steps},
        {"Heavy compute batch", synthetic_body_count > 1024 ? 1 : balanced_steps * 4},
        {"Maximum compute batch", synthetic_body_count > 1024 ? 1 : balanced_steps * 8},
    };
    result.default_speed_index = 1;
    result.trail_sample_interval_s = 1000.0;
    result.default_radius_multiplier = 1.0F;
    result.label_policy = LabelPolicy::selected_only;
    for (std::size_t index = 0; index < result.initial_state.bodies.size(); ++index) {
        result.stable_ids.push_back("synthetic:" + result.initial_state.bodies[index].name);
        result.styles.push_back({
            synthetic_colors[index % synthetic_colors.size()],
            0.0,
            false,
            BodyMaterial::generic,
            1.0F,
        });
        result.body_metadata.push_back({
            result.stable_ids.back(), std::nullopt,
            astronomy::AstronomicalBodyClass::asteroid, false,
        });
    }
    return result;
}

PreparedScenarioReplacement prepare_visual_scenario_replacement(
    ScenarioKind kind,
    std::size_t synthetic_body_count,
    std::uint64_t synthetic_seed,
    LargeNDistribution synthetic_distribution,
    const std::vector<std::string>& included_stable_ids) {
    ScenarioDefinition candidate = kind == ScenarioKind::astronomical_catalog
        ? make_astronomical_catalog_scenario(included_stable_ids)
        : make_visual_scenario(
            kind, synthetic_body_count, synthetic_seed, synthetic_distribution);
    validate_visual_scenario(candidate);
    const std::size_t speed_index = candidate.default_speed_index;
    std::size_t selected_body = 0U;
    if (const auto earth = find_body_index(
            candidate, candidate.kind == ScenarioKind::astronomical_catalog
                ? "sol:earth" : "earth")) {
        selected_body = *earth;
    }
    RenderSpaceTransform render_space(
        candidate.initial_state, candidate.styles, 9.0F,
        candidate.default_radius_multiplier);
    std::vector<bool> profile_visibility(candidate.initial_state.bodies.size(), true);
    std::vector<bool> individual_visibility(candidate.initial_state.bodies.size(), true);
    return {
        std::move(candidate),
        std::move(render_space),
        std::move(profile_visibility),
        std::move(individual_visibility),
        speed_index,
        selected_body,
    };
}

bool SimulationTimingTracker::publish(
    std::uint64_t generation,
    double batch_milliseconds,
    double simulation_steps_per_second) noexcept {
    if (generation_ == 0U || generation != generation_ ||
        !std::isfinite(batch_milliseconds) || batch_milliseconds < 0.0 ||
        !std::isfinite(simulation_steps_per_second) ||
        simulation_steps_per_second < 0.0) {
        return false;
    }
    sample_ = SimulationTimingSample{
        batch_milliseconds, simulation_steps_per_second};
    return true;
}

void SimulationTimingTracker::invalidate() noexcept {
    sample_.reset();
    if (generation_ == 0U) {
        return;
    }
    if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
        generation_ = 0U;
        return;
    }
    ++generation_;
}

bool same_barnes_hut_configuration(
    const BarnesHutConfiguration& left,
    const BarnesHutConfiguration& right) noexcept {
    return left.theta == right.theta &&
        left.leaf_capacity == right.leaf_capacity &&
        left.maximum_depth == right.maximum_depth &&
        left.minimum_cell_size_m == right.minimum_cell_size_m &&
        left.maximum_nodes == right.maximum_nodes &&
        left.maximum_traversal_work == right.maximum_traversal_work &&
        left.memory_budget_bytes == right.memory_budget_bytes;
}

bool backend_selection_changes_execution(
    const SimulationController& controller,
    BackendKind backend,
    int openmp_threads,
    int cuda_device) noexcept {
    return backend != controller.backend_kind() ||
        (backend == BackendKind::openmp &&
         openmp_threads != controller.openmp_threads()) ||
        (backend == BackendKind::cuda &&
         cuda_device != controller.cuda_device());
}

bool solver_selection_changes_execution(
    const SimulationController& controller,
    SolverKind solver,
    const BarnesHutConfiguration& barnes_hut) noexcept {
    return solver != controller.solver_kind() ||
        (solver == SolverKind::barnes_hut &&
         !same_barnes_hut_configuration(
             barnes_hut, controller.barnes_hut_configuration()));
}

bool steps_per_frame_changes_execution(
    const SimulationController& controller,
    std::size_t steps_per_frame) noexcept {
    return steps_per_frame != controller.steps_per_frame();
}

void select_application_backend(
    SimulationController& controller,
    SimulationTimingTracker& timing,
    BackendKind backend,
    int openmp_threads,
    int cuda_device) {
    const bool changed = backend_selection_changes_execution(
        controller, backend, openmp_threads, cuda_device);
    controller.select_backend(backend, openmp_threads, cuda_device);
    if (changed) {
        timing.invalidate();
    }
}

void select_application_solver(
    SimulationController& controller,
    SimulationTimingTracker& timing,
    SolverKind solver,
    const BarnesHutConfiguration& barnes_hut) {
    const bool changed = solver_selection_changes_execution(
        controller, solver, barnes_hut);
    controller.select_solver(solver, barnes_hut);
    if (changed) {
        timing.invalidate();
    }
}

void set_application_steps_per_frame(
    SimulationController& controller,
    SimulationTimingTracker& timing,
    std::size_t steps_per_frame) {
    const bool changed = steps_per_frame_changes_execution(
        controller, steps_per_frame);
    controller.set_steps_per_frame(steps_per_frame);
    if (changed) {
        timing.invalidate();
    }
}

namespace {

WorkloadBackend transaction_workload_backend(BackendKind backend) noexcept {
    switch (backend) {
        case BackendKind::sequential: return WorkloadBackend::sequential;
        case BackendKind::openmp: return WorkloadBackend::openmp;
        case BackendKind::cuda: return WorkloadBackend::cuda;
    }
    return WorkloadBackend::sequential;
}

WorkloadSolver transaction_workload_solver(SolverKind solver) noexcept {
    return solver == SolverKind::barnes_hut
        ? WorkloadSolver::barnes_hut : WorkloadSolver::direct;
}

}  // namespace

PreparedApplicationScenarioTransaction prepare_application_scenario_transaction(
    ScenarioKind kind,
    std::size_t synthetic_body_count,
    std::uint64_t synthetic_seed,
    LargeNDistribution synthetic_distribution,
    const std::vector<std::string>& included_stable_ids,
    const SimulationController& controller,
    const CounterfactualPublicationGate& gate,
    const ScenarioPreparationCompletion& completion) {
    const std::uint64_t next_generation =
        checked_next_counterfactual_generation(gate.scenario_generation());
    auto visual = prepare_visual_scenario_replacement(
        kind, synthetic_body_count, synthetic_seed, synthetic_distribution,
        included_stable_ids);
    const auto workload = visual.scenario.kind == ScenarioKind::synthetic
        ? interactive_workload_policy(
            visual.scenario.initial_state.bodies.size(),
            transaction_workload_backend(controller.backend_kind()),
            transaction_workload_solver(controller.solver_kind()))
        : InteractiveWorkloadPolicy{};
    if (visual.scenario.kind == ScenarioKind::synthetic &&
        !workload.continuous_advancement && !workload.manual_single_step) {
        throw std::invalid_argument(workload.explanation);
    }
    if (completion) {
        completion(visual);
    }
    return {std::move(visual), workload, next_generation};
}

void commit_application_scenario_transaction(
    PreparedApplicationScenarioTransaction&& prepared,
    ApplicationScenarioTransactionTargets targets) {
    const std::size_t speed_index = prepared.visual.speed_index;
    targets.controller.replace_scenario(
        prepared.visual.scenario.initial_state,
        prepared.visual.scenario.dt_s,
        prepared.visual.scenario.speed_presets[speed_index].steps_per_frame);

    // checked_next_counterfactual_generation() already succeeded during
    // preparation. Under single UI-thread ownership, the generation cannot
    // change between prepare and commit except through this transaction.
    targets.experiment_worker.request_cancel();
    targets.experiment_gate.replace_scenario();
    if (targets.experiment_gate.scenario_generation() !=
        prepared.next_scenario_generation) {
        std::terminate();
    }

    static_assert(std::is_nothrow_swappable_v<ScenarioDefinition>);
    static_assert(std::is_nothrow_swappable_v<RenderSpaceTransform>);
    static_assert(std::is_nothrow_move_assignable_v<
        CounterfactualWorkerPublication>);
    using std::swap;
    swap(targets.scenario, prepared.visual.scenario);
    swap(targets.render_space, prepared.visual.render_space);
    targets.profile_visibility.swap(prepared.visual.profile_visibility);
    targets.individual_visibility.swap(prepared.visual.individual_visibility);
    targets.experiment_publication = {};
    targets.presentation.clear();
    targets.timing.invalidate();
}

}  // namespace nbody::visualization
