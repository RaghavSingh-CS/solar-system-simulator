#include "nbody/visualization/render_space.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace nbody::visualization {

namespace {

bool finite_vector(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

}  // namespace

RenderSpaceTransform::RenderSpaceTransform(
    const SimulationState& reference_state,
    std::vector<RenderBodyStyle> styles,
    float target_extent,
    float radius_multiplier,
    float minimum_render_radius)
    : radius_multiplier_(radius_multiplier),
      minimum_render_radius_(minimum_render_radius),
      styles_(std::move(styles)) {
    if (reference_state.bodies.empty()) {
        throw std::invalid_argument("render-space reference state is empty");
    }
    if (!(target_extent > 0.0F) || !std::isfinite(target_extent) ||
        !(radius_multiplier > 0.0F) || !std::isfinite(radius_multiplier) ||
        !(minimum_render_radius > 0.0F) || !std::isfinite(minimum_render_radius)) {
        throw std::invalid_argument("render-space scales must be finite and positive");
    }
    if (!styles_.empty() && styles_.size() != reference_state.bodies.size()) {
        throw std::invalid_argument("render style count must match body count");
    }
    for (const auto& style : styles_) {
        if (!(style.visual_scale > 0.0F) || !std::isfinite(style.visual_scale)) {
            throw std::invalid_argument("render body visual scale must be finite and positive");
        }
    }
    double maximum_mass = 0.0;
    for (const auto& body : reference_state.bodies) {
        maximum_mass = std::max(maximum_mass, body.mass_kg);
    }
    if (!(maximum_mass > 0.0)) {
        throw std::invalid_argument("render-space reference state has no usable scale");
    }
    maximum_mass_kg_ = static_cast<float>(maximum_mass);
    if (styles_.empty()) {
        styles_.resize(reference_state.bodies.size());
    }
    std::vector<std::size_t> all_indices(reference_state.bodies.size());
    for (std::size_t index = 0; index < all_indices.size(); ++index) {
        all_indices[index] = index;
    }
    configure_view(reference_state, {}, all_indices, target_extent);
}

void RenderSpaceTransform::update_snapshot(
    const SimulationState& state,
    std::vector<RenderBody>& snapshot) const {
    snapshot.clear();
    if (snapshot.capacity() < state.bodies.size()) {
        snapshot.reserve(state.bodies.size());
    }
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        const auto& body = state.bodies[index];
        const float mass_ratio = std::max(0.0F, static_cast<float>(body.mass_kg) / maximum_mass_kg_);
        const auto& style = styles_[index];
        const float physical_radius =
            static_cast<float>(style.physical_radius_m) * position_scale_ * radius_multiplier_;
        const float synthetic_radius =
            radius_multiplier_ * (0.08F + 0.20F * std::cbrt(mass_ratio));
        float radius = style.physical_radius_m > 0.0
            ? std::max(minimum_render_radius_, physical_radius) * style.visual_scale
            : synthetic_radius;
        snapshot.push_back({
            to_render_position(body.position_m),
            radius,
            style.color,
            style.emissive,
            style.material,
            true,
        });
    }
}

glm::vec3 RenderSpaceTransform::to_render_position(const Vec3& position_m) const {
    return to_render_offset(position_m - origin_m_);
}

glm::vec3 RenderSpaceTransform::to_render_offset(const Vec3& offset_m) const {
    return {
        static_cast<float>(offset_m.x * static_cast<double>(position_scale_)),
        static_cast<float>(offset_m.y * static_cast<double>(position_scale_)),
        static_cast<float>(offset_m.z * static_cast<double>(position_scale_)),
    };
}

void RenderSpaceTransform::configure_view(
    const SimulationState& state,
    const Vec3& origin_m,
    const std::vector<std::size_t>& framing_body_indices,
    float target_extent) {
    if (state.bodies.size() != styles_.size() || framing_body_indices.empty() ||
        !(target_extent > 0.0F) || !std::isfinite(target_extent) ||
        !finite_vector(origin_m)) {
        throw std::invalid_argument("render-space view configuration is invalid");
    }
    double maximum_distance = 0.0;
    for (const auto index : framing_body_indices) {
        if (index >= state.bodies.size()) {
            throw std::out_of_range("render-space framing body index is invalid");
        }
        maximum_distance = std::max(
            maximum_distance, norm(state.bodies[index].position_m - origin_m));
    }
    if (!(maximum_distance > 0.0) || !std::isfinite(maximum_distance)) {
        throw std::invalid_argument("render-space view has no usable spatial extent");
    }
    origin_m_ = origin_m;
    position_scale_ = static_cast<float>(static_cast<double>(target_extent) / maximum_distance);
}

void RenderSpaceTransform::set_origin_m(const Vec3& origin_m) {
    if (!finite_vector(origin_m)) {
        throw std::invalid_argument("render-space origin must be finite");
    }
    origin_m_ = origin_m;
}

void RenderSpaceTransform::set_radius_multiplier(float multiplier) {
    if (!(multiplier > 0.0F) || !std::isfinite(multiplier) || multiplier > 1000.0F) {
        throw std::invalid_argument("visual radius multiplier must be in (0, 1000]");
    }
    radius_multiplier_ = multiplier;
}

float RenderSpaceTransform::position_scale() const noexcept { return position_scale_; }
const Vec3& RenderSpaceTransform::origin_m() const noexcept { return origin_m_; }
float RenderSpaceTransform::radius_multiplier() const noexcept { return radius_multiplier_; }
float RenderSpaceTransform::minimum_render_radius() const noexcept {
    return minimum_render_radius_;
}

}  // namespace nbody::visualization
