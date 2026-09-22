#pragma once

#include "nbody/vector3.hpp"

#include <string>

namespace nbody {

struct Body {
    std::string name;
    double mass_kg{0.0};
    Vec3 position_m;
    Vec3 velocity_m_s;
};

}  // namespace nbody
