#include "nbody/visualization/presentation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nbody::visualization {

NumericalHealth classify_numerical_health(
    double energy_relative,
    double angular_momentum_relative) noexcept {
    return {
        std::isfinite(energy_relative) && energy_relative <= energy_drift_bound,
        std::isfinite(angular_momentum_relative) &&
            angular_momentum_relative <= angular_momentum_drift_bound,
    };
}

CameraFraming camera_framing(
    CameraPreset preset,
    ScenarioKind scenario,
    const std::vector<RenderBody>& bodies) {
    if (bodies.empty()) {
        throw std::invalid_argument("camera framing requires at least one render body");
    }
    const auto body_target = [&](std::size_t index) {
        if (index >= bodies.size()) {
            throw std::invalid_argument("camera preset body is unavailable in this scenario");
        }
        return bodies[index].position;
    };
    if (scenario != ScenarioKind::solar_system && preset != CameraPreset::overview) {
        throw std::invalid_argument("planet camera presets require the Solar System scenario");
    }
    switch (preset) {
        case CameraPreset::overview:
            return {35.0F, 28.0F, 22.0F, glm::vec3(0.0F), std::nullopt};
        case CameraPreset::inner_system:
            return {-30.0F, 58.0F, 2.65F, body_target(0), 0U};
        case CameraPreset::earth:
            return {38.0F, 22.0F, 2.15F, body_target(3), 3U};
        case CameraPreset::jupiter:
            return {35.0F, 20.0F, 2.35F, body_target(5), 5U};
        case CameraPreset::saturn:
            return {30.0F, 25.0F, 2.55F, body_target(6), 6U};
    }
    throw std::invalid_argument("unknown camera preset");
}

void apply_camera_framing(OrbitCamera& camera, const CameraFraming& framing) {
    camera.set_pose(
        framing.yaw_degrees,
        framing.pitch_degrees,
        framing.distance,
        framing.target);
}

namespace {

bool overlaps(const LabelPlacement& left, const LabelPlacement& right, float margin) {
    return left.position.x < right.position.x + right.size.x + margin &&
        left.position.x + left.size.x + margin > right.position.x &&
        left.position.y < right.position.y + right.size.y + margin &&
        left.position.y + left.size.y + margin > right.position.y;
}

}  // namespace

std::vector<LabelPlacement> declutter_labels(
    std::vector<LabelCandidate> candidates,
    glm::vec2 viewport_size,
    float left_inset,
    float margin) {
    if (!(viewport_size.x > 0.0F) || !(viewport_size.y > 0.0F) ||
        left_inset < 0.0F || margin < 0.0F) {
        throw std::invalid_argument("label viewport and margins must be valid");
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.priority > right.priority;
    });
    constexpr std::array<glm::vec2, 10> offsets{{
        {12.0F, -18.0F}, {12.0F, 8.0F}, {-12.0F, -18.0F}, {-12.0F, 8.0F},
        {20.0F, -34.0F}, {20.0F, 24.0F}, {-20.0F, -34.0F}, {-20.0F, 24.0F},
        {34.0F, -10.0F}, {-34.0F, -10.0F},
    }};
    std::vector<LabelPlacement> placed;
    placed.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        for (const auto& offset : offsets) {
            glm::vec2 position = candidate.anchor + offset;
            if (offset.x < 0.0F) {
                position.x -= candidate.size.x;
            }
            position.x = std::clamp(
                position.x,
                left_inset + margin,
                std::max(left_inset + margin, viewport_size.x - candidate.size.x - margin));
            position.y = std::clamp(
                position.y,
                margin,
                std::max(margin, viewport_size.y - candidate.size.y - margin));
            LabelPlacement proposal{
                candidate.body_index,
                candidate.anchor,
                position,
                candidate.size,
                candidate.selected,
                std::abs(offset.x) > 15.0F || std::abs(offset.y) > 20.0F,
            };
            const bool collision = std::any_of(placed.begin(), placed.end(), [&](const auto& item) {
                return overlaps(proposal, item, margin);
            });
            if (!collision) {
                placed.push_back(proposal);
                break;
            }
        }
    }
    return placed;
}

}  // namespace nbody::visualization
