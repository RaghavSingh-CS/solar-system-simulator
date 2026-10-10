#pragma once

#include "nbody/astronomy/catalog.hpp"
#include "nbody/simulation.hpp"
#include "nbody/visualization/render_space.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nbody::visualization {

enum class ScenarioKind {
    solar_system,
    astronomical_catalog,
    synthetic,
};

enum class LabelPolicy {
    all_bodies,
    selected_only,
};

struct SpeedPreset {
    const char* name;
    std::size_t steps_per_frame;
};

struct ScenarioBodyMetadata {
    std::string stable_id;
    std::optional<std::string> parent_stable_id;
    astronomy::AstronomicalBodyClass body_class{astronomy::AstronomicalBodyClass::asteroid};
    bool counterfactual_removal_eligible{false};
};

struct ScenarioDefinition {
    ScenarioKind kind{ScenarioKind::solar_system};
    std::string name;
    std::string provenance;
    SimulationState initial_state;
    std::vector<std::string> stable_ids;
    std::vector<ScenarioBodyMetadata> body_metadata;
    std::vector<RenderBodyStyle> styles;
    std::vector<std::vector<Vec3>> reference_paths_m;
    double dt_s{1.0};
    std::vector<SpeedPreset> speed_presets;
    std::size_t default_speed_index{0};
    double trail_sample_interval_s{1.0};
    float default_radius_multiplier{1.0F};
    LabelPolicy label_policy{LabelPolicy::all_bodies};
    bool timestep_validation_complete{true};
    std::string scientific_limitations;
    std::vector<std::string> composition_warnings;
};

inline constexpr double astronomical_catalog_initial_dt_s = 60.0;

ScenarioDefinition make_visual_scenario(
    ScenarioKind kind,
    std::size_t synthetic_body_count = 32,
    std::uint64_t synthetic_seed = 24301);

ScenarioDefinition make_astronomical_catalog_scenario(
    const std::vector<std::string>& included_stable_ids = {});

std::optional<std::size_t> find_body_index(
    const ScenarioDefinition& scenario,
    const std::string& stable_id) noexcept;

const char* astronomical_body_class_name(
    astronomy::AstronomicalBodyClass body_class) noexcept;

}  // namespace nbody::visualization
