#pragma once

#include "nbody/simulation.hpp"

namespace nbody {

struct Diagnostics {
    double kinetic_energy_j{0.0};
    double potential_energy_j{0.0};
    Vec3 linear_momentum_kg_m_s;
    Vec3 angular_momentum_kg_m2_s;
    Vec3 center_of_mass_m;

    double total_energy_j() const { return kinetic_energy_j + potential_energy_j; }
};

Diagnostics compute_diagnostics(
    const SimulationState& state,
    double gravitational_constant_value = gravitational_constant);

}  // namespace nbody
