#include "nbody/scenarios.hpp"

#include <cmath>

namespace nbody {

TwoBodyScenario make_analytic_circular_two_body() {
    // Deliberately generic values: this is an analytic mechanics fixture, not ephemeris data.
    constexpr double primary_mass_kg = 1.0e30;
    constexpr double secondary_mass_kg = 1.0e24;
    constexpr double separation_m = 1.0e11;
    constexpr double two_pi = 6.283185307179586476925286766559;

    const double total_mass = primary_mass_kg + secondary_mass_kg;
    const double angular_speed =
        std::sqrt(gravitational_constant * total_mass / (separation_m * separation_m * separation_m));
    const double primary_radius = separation_m * secondary_mass_kg / total_mass;
    const double secondary_radius = separation_m * primary_mass_kg / total_mass;

    SimulationState state;
    state.bodies = {
        {"primary", primary_mass_kg, {-primary_radius, 0.0, 0.0},
         {0.0, -angular_speed * primary_radius, 0.0}},
        {"secondary", secondary_mass_kg, {secondary_radius, 0.0, 0.0},
         {0.0, angular_speed * secondary_radius, 0.0}},
    };

    return {state, separation_m, two_pi / angular_speed};
}

}  // namespace nbody
