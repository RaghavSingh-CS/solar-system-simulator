#pragma once

#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/render_space.hpp"
#include "nbody/visualization/scenario.hpp"

#include <glm/vec2.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace nbody::visualization {

inline constexpr double energy_drift_bound = 5.0e-6;
inline constexpr double angular_momentum_drift_bound = 5.0e-10;

struct NumericalHealth {
    bool energy_within_bound{true};
    bool angular_momentum_within_bound{true};

    bool all_within_bounds() const noexcept {
        return energy_within_bound && angular_momentum_within_bound;
    }
};

NumericalHealth classify_numerical_health(
    double energy_relative,
    double angular_momentum_relative) noexcept;

enum class CameraPreset {
    overview,
    inner_system,
    earth,
    jupiter,
    saturn,
};

struct CameraFraming {
    float yaw_degrees{35.0F};
    float pitch_degrees{28.0F};
    float distance{22.0F};
    glm::vec3 target{0.0F};
    std::optional<std::size_t> tracked_body;
};

CameraFraming camera_framing(
    CameraPreset preset,
    ScenarioKind scenario,
    const std::vector<RenderBody>& bodies);

void apply_camera_framing(OrbitCamera& camera, const CameraFraming& framing);

glm::vec3 amplify_from_barycenter(
    const glm::vec3& render_position,
    const glm::vec3& barycenter,
    float amplification);

struct PanelWidthPolicy {
    float minimum{380.0F};
    float maximum{680.0F};
    float width{438.0F};
};

PanelWidthPolicy panel_width_policy(float display_width, float requested_width);

std::string format_measurement(double value, const char* unit, int significant_digits = 6);

enum class HelpTopic {
    scenario,
    pause,
    reset,
    simulation_speed,
    sequential_backend,
    openmp_backend,
    openmp_threads,
    workload_size,
    compare_cpu_backends,
    body_selector,
    focus_selected,
    camera_presets,
    history_trails,
    reference_orbits,
    labels,
    barycenter,
    sun_motion_amplification,
    body_size,
    energy_conservation,
    angular_momentum_conservation,
    numerical_status,
    trail_history,
    advanced_details,
    manual_large_step,
    count,
};

const char* help_text(HelpTopic topic) noexcept;

inline constexpr float sun_minimum_apparent_radius_pixels = 20.0F;
inline constexpr float sun_maximum_apparent_radius_pixels = 36.0F;
inline constexpr float body_minimum_apparent_radius_pixels = 3.0F;
inline constexpr float body_maximum_apparent_radius_pixels = 13.0F;

float apparent_radius_pixels(
    float world_radius,
    float distance_to_camera,
    float viewport_height_pixels);

void apply_screen_space_body_sizing(
    std::vector<RenderBody>& bodies,
    const OrbitCamera& camera,
    float viewport_height_pixels,
    bool solar_system);

struct LabelCandidate {
    std::size_t body_index{0};
    glm::vec2 anchor{0.0F};
    glm::vec2 size{0.0F};
    int priority{0};
    bool selected{false};
};

struct LabelPlacement {
    std::size_t body_index{0};
    glm::vec2 anchor{0.0F};
    glm::vec2 position{0.0F};
    glm::vec2 size{0.0F};
    bool selected{false};
    bool leader_line{false};
};

std::vector<LabelPlacement> declutter_labels(
    std::vector<LabelCandidate> candidates,
    glm::vec2 viewport_size,
    float left_inset = 0.0F,
    float margin = 6.0F,
    const std::vector<LabelPlacement>& previous = {},
    const std::vector<LabelPlacement>& reserved = {});

std::vector<LabelPlacement> stable_inner_system_callouts(
    const std::vector<LabelCandidate>& candidates,
    glm::vec2 viewport_size,
    float left_inset,
    float margin = 8.0F);

}  // namespace nbody::visualization
