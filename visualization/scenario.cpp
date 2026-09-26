#include "nbody/visualization/scenario.hpp"

#include "nbody/scenarios.hpp"
#include "nbody/solar_system.hpp"

#include <array>
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

constexpr std::array<glm::vec3, 8> synthetic_colors{{
    {0.95F, 0.75F, 0.30F}, {0.35F, 0.70F, 1.00F},
    {0.75F, 0.40F, 0.95F}, {0.35F, 0.90F, 0.62F},
    {0.95F, 0.42F, 0.42F}, {0.92F, 0.88F, 0.48F},
    {0.45F, 0.82F, 0.90F}, {0.92F, 0.58F, 0.82F},
}};

}  // namespace

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
            {"Slow - 6 h/frame", 1},
            {"Normal - 1 day/frame", 4},
            {"Fast - 4 days/frame", 16},
            {"Very Fast - 16 days/frame", 64},
        };
        result.default_speed_index = 1;
        result.trail_sample_interval_s = 86400.0;
        result.default_radius_multiplier = 80.0F;
        for (std::size_t index = 0; index < solar.metadata.size(); ++index) {
            result.stable_ids.push_back(solar.metadata[index].stable_id);
            result.styles.push_back({
                solar_colors[index],
                solar.metadata[index].mean_radius_m,
                index == 0,
            });
        }
        return result;
    }

    result.name = "Synthetic N-body fixture";
    result.provenance = "deterministic seed fixture; not the Solar System";
    result.initial_state = make_seeded_synthetic_system(synthetic_body_count, synthetic_seed);
    result.dt_s = 10.0;
    result.speed_presets = {
        {"Slow - 8 steps/frame", 8},
        {"Normal - 32 steps/frame", 32},
        {"Fast - 128 steps/frame", 128},
        {"Very Fast - 256 steps/frame", 256},
    };
    result.default_speed_index = 1;
    result.trail_sample_interval_s = 1000.0;
    result.default_radius_multiplier = 1.0F;
    for (std::size_t index = 0; index < result.initial_state.bodies.size(); ++index) {
        result.stable_ids.push_back("synthetic_" + std::to_string(index));
        result.styles.push_back({synthetic_colors[index % synthetic_colors.size()], 0.0, false});
    }
    return result;
}

}  // namespace nbody::visualization
