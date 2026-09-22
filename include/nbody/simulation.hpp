#pragma once

#include "nbody/body.hpp"

#include <vector>

namespace nbody {

// 2022 CODATA recommended value. Units: m^3 kg^-1 s^-2.
inline constexpr double gravitational_constant = 6.67430e-11;

struct SimulationState {
    double time_s{0.0};
    std::vector<Body> bodies;
};

std::vector<Vec3> compute_accelerations(
    const SimulationState& state,
    double gravitational_constant_value = gravitational_constant);

void velocity_verlet_step(
    SimulationState& state,
    double dt_s,
    double gravitational_constant_value = gravitational_constant);

}  // namespace nbody
