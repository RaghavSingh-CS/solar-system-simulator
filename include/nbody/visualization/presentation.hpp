#pragma once

#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/render_space.hpp"
#include "nbody/visualization/scenario.hpp"

#include <glm/vec2.hpp>

#include <cstddef>
#include <optional>
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
    float margin = 6.0F);

}  // namespace nbody::visualization
