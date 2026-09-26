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
};

class RenderSpaceTransform {
public:
    explicit RenderSpaceTransform(
        const SimulationState& reference_state,
        float target_extent = 8.0F,
        float radius_exaggeration = 1.0F);

    void update_snapshot(
        const SimulationState& state,
        std::vector<RenderBody>& snapshot) const;
    float position_scale() const noexcept;
    float radius_exaggeration() const noexcept;

private:
    float position_scale_{1.0F};
    float maximum_mass_kg_{1.0F};
    float radius_exaggeration_{1.0F};
};

}  // namespace nbody::visualization
