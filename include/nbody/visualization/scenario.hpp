#pragma once

#include "nbody/astronomy/catalog.hpp"
#include "nbody/large_n_laboratory.hpp"
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
    std::optional<LargeNDistribution> synthetic_distribution;
    std::optional<std::uint64_t> synthetic_seed;
};

struct PreparedScenarioReplacement {
    ScenarioDefinition scenario;
    RenderSpaceTransform render_space;
    std::vector<bool> profile_visibility;
    std::vector<bool> individual_visibility;
    std::size_t speed_index{0};
    std::size_t selected_body{0};
};

inline constexpr double astronomical_catalog_initial_dt_s = 60.0;

ScenarioDefinition make_visual_scenario(
    ScenarioKind kind,
    std::size_t synthetic_body_count = 256,
    std::uint64_t synthetic_seed = 24301,
    LargeNDistribution synthetic_distribution = LargeNDistribution::uniform);

ScenarioDefinition make_astronomical_catalog_scenario(
    const std::vector<std::string>& included_stable_ids = {});

PreparedScenarioReplacement prepare_visual_scenario_replacement(
    ScenarioKind kind,
    std::size_t synthetic_body_count,
    std::uint64_t synthetic_seed,
    LargeNDistribution synthetic_distribution,
    const std::vector<std::string>& included_stable_ids = {});

void validate_visual_scenario(const ScenarioDefinition& scenario);

std::optional<std::size_t> find_body_index(
    const ScenarioDefinition& scenario,
    const std::string& stable_id) noexcept;

const char* astronomical_body_class_name(
    astronomy::AstronomicalBodyClass body_class) noexcept;

const char* synthetic_distribution_name(LargeNDistribution distribution) noexcept;
bool valid_synthetic_distribution(LargeNDistribution distribution) noexcept;

}  // namespace nbody::visualization
