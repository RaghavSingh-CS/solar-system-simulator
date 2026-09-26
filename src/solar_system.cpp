#include "nbody/solar_system.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace nbody {

namespace {

constexpr double astronomical_unit_m = 149597870700.0;
constexpr double seconds_per_julian_century = 36525.0 * 86400.0;
constexpr double pi = 3.1415926535897932384626433832795;
constexpr double degrees_to_radians = pi / 180.0;

struct ElementPair {
    double value;
    double rate_per_century;
};

struct PlanetDefinition {
    const char* stable_id;
    const char* name;
    double mass_kg;
    double mean_radius_m;
    ElementPair semi_major_axis_au;
    ElementPair eccentricity;
    ElementPair inclination_deg;
    ElementPair mean_longitude_deg;
    ElementPair longitude_perihelion_deg;
    ElementPair longitude_node_deg;
};

// JPL SSD Approximate Positions of the Planets, Table 1 (1800-2050).
// Earth uses the table's Earth-Moon barycenter orbital elements; the Moon is
// intentionally outside this nine-body approximation.
constexpr std::array<PlanetDefinition, 8> planets{{
    {"mercury", "Mercury", 0.33010e24, 2439.7e3,
     {0.38709927, 0.00000037}, {0.20563593, 0.00001906},
     {7.00497902, -0.00594749}, {252.25032350, 149472.67411175},
     {77.45779628, 0.16047689}, {48.33076593, -0.12534081}},
    {"venus", "Venus", 4.8673e24, 6051.8e3,
     {0.72333566, 0.00000390}, {0.00677672, -0.00004107},
     {3.39467605, -0.00078890}, {181.97909950, 58517.81538729},
     {131.60246718, 0.00268329}, {76.67984255, -0.27769418}},
    {"earth", "Earth", 5.9722e24, 6371.0e3,
     {1.00000261, 0.00000562}, {0.01671123, -0.00004392},
     {-0.00001531, -0.01294668}, {100.46457166, 35999.37244981},
     {102.93768193, 0.32327364}, {0.0, 0.0}},
    {"mars", "Mars", 0.64169e24, 3389.5e3,
     {1.52371034, 0.00001847}, {0.09339410, 0.00007882},
     {1.84969142, -0.00813131}, {-4.55343205, 19140.30268499},
     {-23.94362959, 0.44441088}, {49.55953891, -0.29257343}},
    {"jupiter", "Jupiter", 1898.13e24, 69911.0e3,
     {5.20288700, -0.00011607}, {0.04838624, -0.00013253},
     {1.30439695, -0.00183714}, {34.39644051, 3034.74612775},
     {14.72847983, 0.21252668}, {100.47390909, 0.20469106}},
    {"saturn", "Saturn", 568.32e24, 58232.0e3,
     {9.53667594, -0.00125060}, {0.05386179, -0.00050991},
     {2.48599187, 0.00193609}, {49.95424423, 1222.49362201},
     {92.59887831, -0.41897216}, {113.66242448, -0.28867794}},
    {"uranus", "Uranus", 86.811e24, 25362.0e3,
     {19.18916464, -0.00196176}, {0.04725744, -0.00004397},
     {0.77263783, -0.00242939}, {313.23810451, 428.48202785},
     {170.95427630, 0.40805281}, {74.01692503, 0.04240589}},
    {"neptune", "Neptune", 102.409e24, 24622.0e3,
     {30.06992276, 0.00026291}, {0.00859048, 0.00005105},
     {1.77004347, 0.00035372}, {-55.12002969, 218.45945325},
     {44.96476227, -0.32241464}, {131.78422574, -0.00508664}},
}};

double at_century(const ElementPair& element, double centuries) {
    return element.value + element.rate_per_century * centuries;
}

double normalize_radians(double angle) {
    return std::remainder(angle, 2.0 * pi);
}

double eccentric_anomaly(double mean_anomaly, double eccentricity) {
    double estimate = mean_anomaly + eccentricity * std::sin(mean_anomaly);
    for (int iteration = 0; iteration < 20; ++iteration) {
        const double residual = estimate - eccentricity * std::sin(estimate) - mean_anomaly;
        const double correction = residual / (1.0 - eccentricity * std::cos(estimate));
        estimate -= correction;
        if (std::abs(correction) <= 1.0e-14) {
            return estimate;
        }
    }
    throw std::runtime_error("JPL approximate-element Kepler solve did not converge");
}

Vec3 heliocentric_position(const PlanetDefinition& planet, double centuries) {
    const double a = at_century(planet.semi_major_axis_au, centuries) * astronomical_unit_m;
    const double e = at_century(planet.eccentricity, centuries);
    const double inclination =
        at_century(planet.inclination_deg, centuries) * degrees_to_radians;
    const double longitude_perihelion =
        at_century(planet.longitude_perihelion_deg, centuries) * degrees_to_radians;
    const double longitude_node =
        at_century(planet.longitude_node_deg, centuries) * degrees_to_radians;
    const double mean_anomaly = normalize_radians(
        (at_century(planet.mean_longitude_deg, centuries) -
         at_century(planet.longitude_perihelion_deg, centuries)) * degrees_to_radians);
    const double argument_perihelion = longitude_perihelion - longitude_node;
    const double anomaly = eccentric_anomaly(mean_anomaly, e);

    const double orbital_x = a * (std::cos(anomaly) - e);
    const double orbital_y = a * std::sqrt(1.0 - e * e) * std::sin(anomaly);
    const double cos_w = std::cos(argument_perihelion);
    const double sin_w = std::sin(argument_perihelion);
    const double cos_node = std::cos(longitude_node);
    const double sin_node = std::sin(longitude_node);
    const double cos_i = std::cos(inclination);
    const double sin_i = std::sin(inclination);

    return {
        (cos_w * cos_node - sin_w * sin_node * cos_i) * orbital_x +
            (-sin_w * cos_node - cos_w * sin_node * cos_i) * orbital_y,
        (cos_w * sin_node + sin_w * cos_node * cos_i) * orbital_x +
            (-sin_w * sin_node + cos_w * cos_node * cos_i) * orbital_y,
        sin_w * sin_i * orbital_x + cos_w * sin_i * orbital_y,
    };
}

Vec3 heliocentric_velocity(const PlanetDefinition& planet) {
    // Centered differentiation of the same JPL fitted-element position formula
    // keeps position and velocity construction internally consistent.
    constexpr double half_interval_s = 3600.0;
    const double offset_centuries = half_interval_s / seconds_per_julian_century;
    return (heliocentric_position(planet, offset_centuries) -
            heliocentric_position(planet, -offset_centuries)) /
        (2.0 * half_interval_s);
}

void make_barycentric(SimulationState& state) {
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
}

}  // namespace

SolarSystemScenario make_approximate_solar_system() {
    SolarSystemScenario scenario;
    scenario.initial_state.bodies.reserve(planets.size() + 1);
    scenario.metadata.reserve(planets.size() + 1);

    scenario.initial_state.bodies.push_back({"Sun", 1988400.0e24, {}, {}});
    scenario.metadata.push_back({"sun", 695700.0e3});

    for (const auto& planet : planets) {
        scenario.initial_state.bodies.push_back({
            planet.name,
            planet.mass_kg,
            heliocentric_position(planet, 0.0),
            heliocentric_velocity(planet),
        });
        scenario.metadata.push_back({planet.stable_id, planet.mean_radius_m});
    }

    make_barycentric(scenario.initial_state);
    return scenario;
}

}  // namespace nbody
