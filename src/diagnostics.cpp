#include "nbody/diagnostics.hpp"

#include <cmath>
#include <stdexcept>

namespace nbody {

Diagnostics compute_diagnostics(
    const SimulationState& state,
    double gravitational_constant_value) {
    Diagnostics result;
    double total_mass = 0.0;

    for (const auto& body : state.bodies) {
        if (!(body.mass_kg > 0.0) || !std::isfinite(body.mass_kg)) {
            throw std::invalid_argument("body mass must be finite and positive");
        }
        total_mass += body.mass_kg;
        result.kinetic_energy_j += 0.5 * body.mass_kg * norm_squared(body.velocity_m_s);
        result.linear_momentum_kg_m_s += body.velocity_m_s * body.mass_kg;
        result.angular_momentum_kg_m2_s +=
            cross(body.position_m, body.velocity_m_s * body.mass_kg);
        result.center_of_mass_m += body.position_m * body.mass_kg;
    }

    if (total_mass > 0.0) {
        result.center_of_mass_m = result.center_of_mass_m / total_mass;
    }

    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        for (std::size_t j = i + 1; j < state.bodies.size(); ++j) {
            const double distance = norm(state.bodies[j].position_m - state.bodies[i].position_m);
            if (!(distance > 0.0) || !std::isfinite(distance)) {
                throw std::domain_error("coincident or invalid body positions");
            }
            result.potential_energy_j -= gravitational_constant_value *
                state.bodies[i].mass_kg * state.bodies[j].mass_kg / distance;
        }
    }

    return result;
}

}  // namespace nbody
