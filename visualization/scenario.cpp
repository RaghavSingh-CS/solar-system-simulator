#include "nbody/visualization/scenario.hpp"

#include "nbody/astronomy/solar_system_60.hpp"
#include "nbody/scenarios.hpp"
#include "nbody/solar_system.hpp"
#include "nbody/visualization/workload.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>
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
    std::uint64_t synthetic_seed) {
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

    result.name = "Compute Experiment";
    result.provenance = "deterministic seed fixture; not the Solar System";
    result.initial_state = make_seeded_synthetic_system(synthetic_body_count, synthetic_seed);
    result.dt_s = 10.0;
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
        result.stable_ids.push_back("synthetic_" + std::to_string(index));
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

}  // namespace nbody::visualization
