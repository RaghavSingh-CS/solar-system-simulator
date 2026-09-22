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

class ForceBackend {
public:
    virtual ~ForceBackend() = default;
    virtual const char* name() const noexcept = 0;
    virtual std::vector<Vec3> compute_accelerations(
        const SimulationState& state,
        double gravitational_constant_value) const = 0;
};

const ForceBackend& sequential_force_backend();

std::vector<Vec3> compute_accelerations(
    const SimulationState& state,
    double gravitational_constant_value = gravitational_constant);

void velocity_verlet_step(
    SimulationState& state,
    double dt_s,
    double gravitational_constant_value = gravitational_constant);

void velocity_verlet_step(
    SimulationState& state,
    double dt_s,
    const ForceBackend& force_backend,
    double gravitational_constant_value = gravitational_constant);

}  // namespace nbody
