#include "nbody/visualization/render_space.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nbody::visualization {

namespace {

constexpr std::array<glm::vec3, 8> palette{{
    {0.95F, 0.75F, 0.30F},
    {0.35F, 0.70F, 1.00F},
    {0.75F, 0.40F, 0.95F},
    {0.35F, 0.90F, 0.62F},
    {0.95F, 0.42F, 0.42F},
    {0.92F, 0.88F, 0.48F},
    {0.45F, 0.82F, 0.90F},
    {0.92F, 0.58F, 0.82F},
}};

}  // namespace

RenderSpaceTransform::RenderSpaceTransform(
    const SimulationState& reference_state,
    float target_extent,
    float radius_exaggeration)
    : radius_exaggeration_(radius_exaggeration) {
    if (reference_state.bodies.empty()) {
        throw std::invalid_argument("render-space reference state is empty");
    }
    if (!(target_extent > 0.0F) || !std::isfinite(target_extent) ||
        !(radius_exaggeration > 0.0F) || !std::isfinite(radius_exaggeration)) {
        throw std::invalid_argument("render-space scales must be finite and positive");
    }
    double maximum_distance = 0.0;
    double maximum_mass = 0.0;
    for (const auto& body : reference_state.bodies) {
        maximum_distance = std::max(maximum_distance, norm(body.position_m));
        maximum_mass = std::max(maximum_mass, body.mass_kg);
    }
    if (!(maximum_distance > 0.0) || !(maximum_mass > 0.0)) {
        throw std::invalid_argument("render-space reference state has no usable scale");
    }
    position_scale_ = target_extent / static_cast<float>(maximum_distance);
    maximum_mass_kg_ = static_cast<float>(maximum_mass);
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
        const float radius =
            radius_exaggeration_ * (0.08F + 0.20F * std::cbrt(mass_ratio));
        snapshot.push_back({
            {
                static_cast<float>(body.position_m.x) * position_scale_,
                static_cast<float>(body.position_m.y) * position_scale_,
                static_cast<float>(body.position_m.z) * position_scale_,
            },
            radius,
            palette[index % palette.size()],
        });
    }
}

float RenderSpaceTransform::position_scale() const noexcept { return position_scale_; }
float RenderSpaceTransform::radius_exaggeration() const noexcept {
    return radius_exaggeration_;
}

}  // namespace nbody::visualization
