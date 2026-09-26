#pragma once

#include "nbody/simulation.hpp"
#include "nbody/visualization/render_space.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nbody::visualization {

enum class ScenarioKind {
    solar_system,
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

struct ScenarioDefinition {
    ScenarioKind kind{ScenarioKind::solar_system};
    std::string name;
    std::string provenance;
    SimulationState initial_state;
    std::vector<std::string> stable_ids;
    std::vector<RenderBodyStyle> styles;
    std::vector<std::vector<Vec3>> reference_paths_m;
    double dt_s{1.0};
    std::vector<SpeedPreset> speed_presets;
    std::size_t default_speed_index{0};
    double trail_sample_interval_s{1.0};
    float default_radius_multiplier{1.0F};
    LabelPolicy label_policy{LabelPolicy::all_bodies};
};

ScenarioDefinition make_visual_scenario(
    ScenarioKind kind,
    std::size_t synthetic_body_count = 32,
    std::uint64_t synthetic_seed = 24301);

}  // namespace nbody::visualization
