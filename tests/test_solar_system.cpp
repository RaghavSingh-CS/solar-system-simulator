#include "nbody/diagnostics.hpp"
#include "nbody/solar_system.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

#ifndef NBODY_BUILD_CONFIGURATION
#define NBODY_BUILD_CONFIGURATION "unknown"
#endif
#ifndef NBODY_STAGE5_SOURCE_SHA
#define NBODY_STAGE5_SOURCE_SHA "uncontrolled"
#endif
#ifndef NBODY_STAGE5_PROTOCOL_VERSION
#define NBODY_STAGE5_PROTOCOL_VERSION "none"
#endif

constexpr double astronomical_unit_m = 149597870700.0;

struct InnerElement {
    const char* name;
    std::size_t body_index;
    double semi_major_axis_au;
    double eccentricity;
    double inclination_deg;
    double longitude_node_deg;
    double longitude_perihelion_deg;
    double mean_longitude_deg;
};

constexpr std::array<InnerElement, 4> inner_elements{{
    {"Mercury", 1, 0.38709927, 0.20563593, 7.00497902,
     48.33076593, 77.45779628, 252.25032350},
    {"Venus", 2, 0.72333566, 0.00677672, 3.39467605,
     76.67984255, 131.60246718, 181.97909950},
    {"Earth", 3, 1.00000261, 0.01671123, -0.00001531,
     0.0, 102.93768193, 100.46457166},
    {"Mars", 4, 1.52371034, 0.09339410, 1.84969142,
     49.55953891, -23.94362959, -4.55343205},
}};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool finite_vector(const nbody::Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double osculating_eccentricity(
    const nbody::Body& sun,
    const nbody::Body& planet) {
    const nbody::Vec3 position = planet.position_m - sun.position_m;
    const nbody::Vec3 velocity = planet.velocity_m_s - sun.velocity_m_s;
    const double mu = nbody::gravitational_constant * (sun.mass_kg + planet.mass_kg);
    const nbody::Vec3 angular_momentum = nbody::cross(position, velocity);
    const nbody::Vec3 eccentricity_vector =
        nbody::cross(velocity, angular_momentum) / mu - position / nbody::norm(position);
    return nbody::norm(eccentricity_vector);
}

void test_inner_orbital_shapes() {
    const auto solar = nbody::make_approximate_solar_system();
    const auto& initial = solar.initial_state;
    constexpr double initial_e_tolerance = 5.0e-4;
    std::array<double, 4> initial_state_eccentricity{};
    for (std::size_t planet = 0; planet < inner_elements.size(); ++planet) {
        const auto& element = inner_elements[planet];
        require(element.eccentricity > 0.0,
                std::string(element.name) + " source eccentricity was discarded");
        const double state_e = osculating_eccentricity(
            initial.bodies[0], initial.bodies[element.body_index]);
        initial_state_eccentricity[planet] = state_e;
        require(std::abs(state_e - element.eccentricity) <= initial_e_tolerance,
                std::string(element.name) +
                    " Cartesian initialization does not retain source eccentricity");
    }

    struct Extrema {
        double minimum{std::numeric_limits<double>::infinity()};
        double maximum{0.0};
    };
    std::array<Extrema, 4> integrated{};
    auto evolved = initial;
    constexpr std::size_t steps = 2800;  // 700 days; longer than one Mars orbit.
    for (std::size_t step = 0; step <= steps; ++step) {
        for (std::size_t planet = 0; planet < inner_elements.size(); ++planet) {
            const std::size_t index = inner_elements[planet].body_index;
            const double radius = nbody::norm(
                evolved.bodies[index].position_m - evolved.bodies[0].position_m);
            integrated[planet].minimum = std::min(integrated[planet].minimum, radius);
            integrated[planet].maximum = std::max(integrated[planet].maximum, radius);
        }
        if (step < steps) {
            nbody::velocity_verlet_step(evolved, nbody::solar_system_recommended_dt_s);
        }
    }

    constexpr double integrated_e_tolerance = 5.0e-4;
    const auto guides = nbody::make_approximate_solar_reference_orbits(4097);
    const nbody::Vec3 guide_sun = initial.bodies[0].position_m;
    std::cout << std::setprecision(15);
    for (std::size_t planet = 0; planet < inner_elements.size(); ++planet) {
        const auto& element = inner_elements[planet];
        const auto& extrema = integrated[planet];
        const double estimated_e =
            (extrema.maximum - extrema.minimum) / (extrema.maximum + extrema.minimum);
        require(extrema.maximum / extrema.minimum > 1.0 + 0.5 * element.eccentricity,
                std::string(element.name) + " integrated orbit is implausibly circular");
        require(std::abs(estimated_e - element.eccentricity) <= integrated_e_tolerance,
                std::string(element.name) +
                    " integrated radial extrema are inconsistent with initialization");

        Extrema guide;
        for (const auto& point : guides[element.body_index]) {
            const double radius = nbody::norm(point - guide_sun);
            guide.minimum = std::min(guide.minimum, radius);
            guide.maximum = std::max(guide.maximum, radius);
        }
        const double expected_minimum =
            element.semi_major_axis_au * (1.0 - element.eccentricity) * astronomical_unit_m;
        const double expected_maximum =
            element.semi_major_axis_au * (1.0 + element.eccentricity) * astronomical_unit_m;
        require(std::abs(guide.minimum - expected_minimum) / expected_minimum < 1.0e-5 &&
                    std::abs(guide.maximum - expected_maximum) / expected_maximum < 1.0e-5,
                std::string(element.name) +
                    " reference guide does not preserve element-model eccentricity");

        std::cout << "orbit_shape," << element.name
                  << ",integrated_min_m=" << extrema.minimum
                  << ",integrated_max_m=" << extrema.maximum
                  << ",integrated_min_au=" << extrema.minimum / astronomical_unit_m
                  << ",integrated_max_au=" << extrema.maximum / astronomical_unit_m
                  << ",ratio=" << extrema.maximum / extrema.minimum
                  << ",e_est=" << estimated_e
                  << ",e_source=" << element.eccentricity
                  << ",e_initial_state=" << initial_state_eccentricity[planet]
                  << ",guide_min_m=" << guide.minimum
                  << ",guide_max_m=" << guide.maximum << '\n';
    }
}

void test_mercury_sun_clearance() {
    const auto solar = nbody::make_approximate_solar_system();
    auto state = solar.initial_state;
    constexpr std::size_t ten_year_steps =
        static_cast<std::size_t>(10.0 * 365.25 * 86400.0 /
                                 nbody::solar_system_recommended_dt_s);
    double minimum_separation_m = std::numeric_limits<double>::infinity();
    for (std::size_t step = 0; step <= ten_year_steps; ++step) {
        minimum_separation_m = std::min(
            minimum_separation_m,
            nbody::norm(state.bodies[1].position_m - state.bodies[0].position_m));
        if (step < ten_year_steps) {
            nbody::velocity_verlet_step(state, nbody::solar_system_recommended_dt_s);
        }
    }
    const double physical_contact_distance_m =
        solar.metadata[0].mean_radius_m + solar.metadata[1].mean_radius_m;
    require(minimum_separation_m > physical_contact_distance_m,
            "Mercury physically intersects the Sun during the ten-year validation interval");
    std::cout << "ten-year Mercury minimum Sun separation=" << minimum_separation_m
              << " m (" << minimum_separation_m / 149597870700.0
              << " AU), physical contact distance=" << physical_contact_distance_m
              << " m\n";
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

    const auto reference_paths = nbody::make_approximate_solar_reference_orbits(65);
    require(reference_paths.size() == expected.size() && reference_paths[0].empty(),
            "Solar reference paths do not preserve the body index contract");
    for (std::size_t index = 1; index < reference_paths.size(); ++index) {
        require(reference_paths[index].size() == 65,
                "Solar reference orbit has the wrong sample count");
        for (const auto& point : reference_paths[index]) {
            require(finite_vector(point), "Solar reference orbit contains a non-finite point");
        }
    }

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

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--build-info") {
            std::cout << "{\"component\":\"nbody_solar_system_tests\",\"source_commit\":\""
                      << NBODY_STAGE5_SOURCE_SHA << "\",\"protocol_version\":\""
                      << NBODY_STAGE5_PROTOCOL_VERSION << "\",\"build_configuration\":\""
                      << NBODY_BUILD_CONFIGURATION << "\"}\n";
            return 0;
        }
        test_inner_orbital_shapes();
        test_solar_system_definition();
        test_mercury_sun_clearance();
        std::cout << "all approximate Solar-System tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Solar-System test failure: " << error.what() << '\n';
        return 1;
    }
}
