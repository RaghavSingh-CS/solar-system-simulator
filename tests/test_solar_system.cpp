#include "nbody/diagnostics.hpp"
#include "nbody/solar_system.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool finite_vector(const nbody::Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void test_solar_system_definition() {
    const auto first = nbody::make_approximate_solar_system();
    const auto second = nbody::make_approximate_solar_system();
    constexpr std::array<const char*, 9> expected{{
        "Sun", "Mercury", "Venus", "Earth", "Mars",
        "Jupiter", "Saturn", "Uranus", "Neptune",
    }};

    require(first.initial_state.bodies.size() == expected.size(),
            "Solar-System scenario must contain Sun plus eight planets");
    require(first.metadata.size() == expected.size(), "Solar metadata count differs");
    require(first.initial_state.time_s == 0.0, "Solar scenario must start at elapsed time zero");

    double total_mass = 0.0;
    nbody::Vec3 total_momentum;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const auto& body = first.initial_state.bodies[index];
        const auto& repeated = second.initial_state.bodies[index];
        require(body.name == expected[index], "unexpected Solar-System body order/name");
        require(body.name == repeated.name && body.mass_kg == repeated.mass_kg &&
                    body.position_m.x == repeated.position_m.x &&
                    body.position_m.y == repeated.position_m.y &&
                    body.position_m.z == repeated.position_m.z &&
                    body.velocity_m_s.x == repeated.velocity_m_s.x &&
                    body.velocity_m_s.y == repeated.velocity_m_s.y &&
                    body.velocity_m_s.z == repeated.velocity_m_s.z,
                "Solar-System reconstruction is not deterministic");
        require(body.mass_kg > 0.0 && std::isfinite(body.mass_kg) &&
                    finite_vector(body.position_m) && finite_vector(body.velocity_m_s),
                "Solar-System body contains invalid physical state");
        require(first.metadata[index].mean_radius_m > 0.0 &&
                    std::isfinite(first.metadata[index].mean_radius_m),
                "Solar-System physical radius is invalid");
        total_mass += body.mass_kg;
        total_momentum += body.velocity_m_s * body.mass_kg;
    }

    const auto diagnostics = nbody::compute_diagnostics(first.initial_state);
    require(nbody::norm(diagnostics.center_of_mass_m) < 1.0e-3,
            "Solar-System center of mass is not barycentric within tolerance");
    require(nbody::norm(total_momentum / total_mass) < 1.0e-9,
            "Solar-System center-of-mass velocity is not near zero");

    const double mercury_sun_distance = nbody::norm(
        first.initial_state.bodies[1].position_m - first.initial_state.bodies[0].position_m);
    require(mercury_sun_distance > 4.0e10 && mercury_sun_distance < 7.5e10,
            "Mercury J2000 distance is outside a defensible orbital range");

    auto evolved = first.initial_state;
    const auto initial_diagnostics = nbody::compute_diagnostics(evolved);
    for (int step = 0; step < 352; ++step) {
        nbody::velocity_verlet_step(evolved, nbody::solar_system_recommended_dt_s);
    }
    const auto final_diagnostics = nbody::compute_diagnostics(evolved);
    const double energy_drift = std::abs(
        final_diagnostics.total_energy_j() - initial_diagnostics.total_energy_j()) /
        std::max({std::abs(final_diagnostics.total_energy_j()),
                  std::abs(initial_diagnostics.total_energy_j()), 1.0});
    const double angular_drift = nbody::norm(
        final_diagnostics.angular_momentum_kg_m2_s -
        initial_diagnostics.angular_momentum_kg_m2_s) /
        std::max(nbody::norm(initial_diagnostics.angular_momentum_kg_m2_s), 1.0);
    require(energy_drift < 1.0e-4,
            "recommended six-hour timestep exceeded the Stage 6B one-Mercury-orbit energy guard");
    require(angular_drift < 1.0e-10,
            "recommended six-hour timestep exceeded the angular-momentum guard");
    std::cout << "one-Mercury-orbit energy drift=" << energy_drift
              << " angular drift=" << angular_drift << '\n';
}

}  // namespace

int main() {
    try {
        test_solar_system_definition();
        std::cout << "all approximate Solar-System tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Solar-System test failure: " << error.what() << '\n';
        return 1;
    }
}
