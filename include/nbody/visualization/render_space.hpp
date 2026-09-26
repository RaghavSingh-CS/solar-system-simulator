#pragma once

#include "nbody/simulation.hpp"

#include <glm/vec3.hpp>

#include <cstddef>
#include <vector>

namespace nbody::visualization {

struct RenderBody {
    glm::vec3 position;
    float radius{0.1F};
    glm::vec3 color{1.0F};
    bool emissive{false};
};

struct RenderBodyStyle {
    glm::vec3 color{1.0F};
    double physical_radius_m{0.0};
    bool emissive{false};
};

class RenderSpaceTransform {
public:
    explicit RenderSpaceTransform(
        const SimulationState& reference_state,
        std::vector<RenderBodyStyle> styles = {},
        float target_extent = 8.0F,
        float radius_multiplier = 1.0F,
        float minimum_render_radius = 0.035F);

    void update_snapshot(
        const SimulationState& state,
        std::vector<RenderBody>& snapshot) const;
    glm::vec3 to_render_position(const Vec3& position_m) const;
    void set_radius_multiplier(float multiplier);
    float position_scale() const noexcept;
    float radius_multiplier() const noexcept;
    float minimum_render_radius() const noexcept;

private:
    float position_scale_{1.0F};
    float maximum_mass_kg_{1.0F};
    float radius_multiplier_{1.0F};
    float minimum_render_radius_{0.035F};
    std::vector<RenderBodyStyle> styles_;
};

}  // namespace nbody::visualization
