#include "nbody/visualization/camera.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nbody::visualization {

OrbitCamera::OrbitCamera(float yaw_degrees, float pitch_degrees, float distance)
    : initial_yaw_degrees_(yaw_degrees),
      initial_pitch_degrees_(pitch_degrees),
      initial_distance_(distance),
      yaw_degrees_(yaw_degrees),
      pitch_degrees_(pitch_degrees),
      distance_(distance) {
    if (!(distance > 0.0F) || !std::isfinite(distance)) {
        throw std::invalid_argument("camera distance must be finite and positive");
    }
}

void OrbitCamera::set_aspect_ratio(float aspect_ratio) {
    if (!(aspect_ratio > 0.0F) || !std::isfinite(aspect_ratio)) {
        throw std::invalid_argument("camera aspect ratio must be finite and positive");
    }
    aspect_ratio_ = aspect_ratio;
}

void OrbitCamera::orbit(float delta_x, float delta_y) {
    if (!std::isfinite(delta_x) || !std::isfinite(delta_y)) return;
    yaw_degrees_ = std::remainder(yaw_degrees_ + std::remainder(delta_x, 1800.0F) * 0.2F, 360.0F);
    pitch_degrees_ = std::remainder(pitch_degrees_ - std::remainder(delta_y, 1800.0F) * 0.2F, 360.0F);
}

void OrbitCamera::pan(float delta_x, float delta_y) {
    if (!std::isfinite(delta_x) || !std::isfinite(delta_y)) return;
    const float yaw = glm::radians(yaw_degrees_);
    const float pitch = glm::radians(pitch_degrees_);
    const glm::vec3 right(-std::sin(yaw), 0.0F, std::cos(yaw));
    const glm::vec3 up(-std::sin(pitch) * std::cos(yaw), std::cos(pitch),
                       -std::sin(pitch) * std::sin(yaw));
    target_ += (-right * std::clamp(delta_x, -2000.0F, 2000.0F) +
               up * std::clamp(delta_y, -2000.0F, 2000.0F)) * (distance_ * 0.0015F);
}

void OrbitCamera::dolly(float amount) {
    if (!std::isfinite(amount)) return;
    distance_ *= std::exp(-std::clamp(amount, -12.0F, 12.0F) * 0.08F);
    distance_ = std::clamp(distance_, 0.15F, 160.0F);
}

void OrbitCamera::set_distance(float distance) {
    if (!(distance > 0.0F) || !std::isfinite(distance)) {
        throw std::invalid_argument("camera distance must be finite and positive");
    }
    distance_ = std::clamp(distance, 0.15F, 160.0F);
}

void OrbitCamera::set_pose(
    float yaw_degrees,
    float pitch_degrees,
    float distance,
    const glm::vec3& target) {
    if (!std::isfinite(yaw_degrees) || !std::isfinite(pitch_degrees) ||
        !(distance > 0.0F) || !std::isfinite(distance) ||
        !std::isfinite(target.x) || !std::isfinite(target.y) || !std::isfinite(target.z)) {
        throw std::invalid_argument("camera pose must be finite with positive distance");
    }
    yaw_degrees_ = std::remainder(yaw_degrees, 360.0F);
    pitch_degrees_ = std::remainder(pitch_degrees, 360.0F);
    distance_ = std::clamp(distance, 0.15F, 160.0F);
    target_ = target;
}

void OrbitCamera::approach_target(const glm::vec3& target, float fraction) {
    if (!(fraction > 0.0F) || fraction > 1.0F || !std::isfinite(fraction) ||
        !std::isfinite(target.x) || !std::isfinite(target.y) || !std::isfinite(target.z)) {
        throw std::invalid_argument("camera target and smoothing fraction must be finite");
    }
    target_ += (target - target_) * fraction;
}

void OrbitCamera::reset() {
    yaw_degrees_ = initial_yaw_degrees_;
    pitch_degrees_ = initial_pitch_degrees_;
    distance_ = initial_distance_;
    target_ = glm::vec3(0.0F);
}

float OrbitCamera::aspect_ratio() const noexcept { return aspect_ratio_; }
float OrbitCamera::distance() const noexcept { return distance_; }
float OrbitCamera::yaw_degrees() const noexcept { return yaw_degrees_; }
float OrbitCamera::pitch_degrees() const noexcept { return pitch_degrees_; }
const glm::vec3& OrbitCamera::target() const noexcept { return target_; }

glm::vec3 OrbitCamera::position() const {
    const float yaw = glm::radians(yaw_degrees_);
    const float pitch = glm::radians(pitch_degrees_);
    const float horizontal = distance_ * std::cos(pitch);
    return target_ + glm::vec3(
        horizontal * std::cos(yaw),
        distance_ * std::sin(pitch),
        horizontal * std::sin(yaw));
}

glm::mat4 OrbitCamera::view_matrix() const {
    const float yaw = glm::radians(yaw_degrees_);
    const float pitch = glm::radians(pitch_degrees_);
    const glm::vec3 up(-std::sin(pitch) * std::cos(yaw), std::cos(pitch),
                       -std::sin(pitch) * std::sin(yaw));
    return glm::lookAt(position(), target_, up);
}

glm::mat4 OrbitCamera::projection_matrix() const {
    return glm::perspective(glm::radians(45.0F), aspect_ratio_, 0.002F, 512.0F);
}

}  // namespace nbody::visualization
