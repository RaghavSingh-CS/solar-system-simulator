#include "nbody/scenarios.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

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

SimulationState make_seeded_synthetic_system(std::size_t body_count, std::uint64_t seed) {
    if (body_count < 2) {
        throw std::invalid_argument("synthetic scenario requires at least two bodies");
    }

    constexpr double position_scale_m = 1.0e9;
    constexpr double velocity_scale_m_s = 100.0;
    constexpr double minimum_mass_kg = 1.0e20;
    constexpr double mass_range_kg = 9.0e20;

    std::mt19937_64 generator(seed);
    const auto unit_interval = [&generator]() {
        return static_cast<double>(generator() >> 11) * 0x1.0p-53;
    };
    const auto symmetric = [&unit_interval]() { return 2.0 * unit_interval() - 1.0; };

    SimulationState state;
    state.bodies.reserve(body_count);
    for (std::size_t i = 0; i < body_count; ++i) {
        Body body;
        body.name = "synthetic_" + std::to_string(i);
        body.mass_kg = minimum_mass_kg + mass_range_kg * unit_interval();
        body.position_m = {
            position_scale_m * symmetric(),
            position_scale_m * symmetric(),
            position_scale_m * symmetric(),
        };
        body.velocity_m_s = {
            velocity_scale_m_s * symmetric(),
            velocity_scale_m_s * symmetric(),
            velocity_scale_m_s * symmetric(),
        };
        state.bodies.push_back(std::move(body));
    }

    double total_mass = 0.0;
    Vec3 weighted_position;
    Vec3 total_momentum;
    for (const auto& body : state.bodies) {
        total_mass += body.mass_kg;
        weighted_position += body.position_m * body.mass_kg;
        total_momentum += body.velocity_m_s * body.mass_kg;
    }
    const Vec3 center_of_mass = weighted_position / total_mass;
    const Vec3 center_of_mass_velocity = total_momentum / total_mass;
    for (auto& body : state.bodies) {
        body.position_m -= center_of_mass;
        body.velocity_m_s -= center_of_mass_velocity;
    }

    return state;
}

}  // namespace nbody
