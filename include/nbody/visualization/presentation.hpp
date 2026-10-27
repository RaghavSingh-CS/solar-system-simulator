#pragma once

#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/render_space.hpp"
#include "nbody/visualization/scenario.hpp"

#include <glm/vec2.hpp>

#include <array>
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
    earth_moon,
    jupiter_system,
    saturn_system,
    uranus_system,
    neptune_triton,
    pluto_charon,
    asteroid_belt,
};

struct CameraFraming {
    float yaw_degrees{35.0F};
    float pitch_degrees{28.0F};
    float distance{22.0F};
    glm::vec3 target{0.0F};
    std::optional<std::size_t> tracked_body;
    std::optional<std::size_t> local_origin_body;
    std::vector<std::size_t> visible_bodies;
    bool parent_relative_trails{false};
};

CameraFraming camera_framing(
    CameraPreset preset,
    const ScenarioDefinition& scenario,
    const std::vector<RenderBody>& bodies);

void apply_camera_framing(OrbitCamera& camera, const CameraFraming& framing);

// Presentation only: instantaneous two-body ellipses from catalog r/v and
// parent mass. Empty for missing parents, unbound or degenerate conics.
// These guides are neither integrated histories nor ephemeris predictions.
std::vector<std::vector<Vec3>> catalog_osculating_orbit_guides(
    const ScenarioDefinition& scenario);

// Bound synchronous GPU work on the GL/event thread; never alter physical dt.
std::size_t responsive_viewer_batch_steps(bool cuda, std::size_t requested_steps);

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
    cuda_backend,
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
    std::optional<std::size_t> sun_index);

inline constexpr std::size_t astronomical_body_class_count = 7;
using BodyClassVisibility = std::array<bool, astronomical_body_class_count>;

std::size_t body_class_visibility_index(
    astronomy::AstronomicalBodyClass body_class) noexcept;

bool effective_body_visibility(
    const ScenarioDefinition& scenario,
    std::size_t body_index,
    const std::vector<bool>& profile_membership,
    const std::vector<bool>& individual_visibility,
    const BodyClassVisibility& class_visibility);

void apply_effective_body_visibility(
    const ScenarioDefinition& scenario,
    const std::vector<bool>& profile_membership,
    const std::vector<bool>& individual_visibility,
    const BodyClassVisibility& class_visibility,
    std::vector<RenderBody>& bodies);

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

int astronomical_label_priority(
    astronomy::AstronomicalBodyClass body_class,
    bool selected,
    std::size_t deterministic_order) noexcept;

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
