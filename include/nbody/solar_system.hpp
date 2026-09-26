#pragma once

#include "nbody/simulation.hpp"

#include <string>
#include <vector>

namespace nbody {

struct PhysicalBodyMetadata {
    std::string stable_id;
    double mean_radius_m{0.0};
};

struct SolarSystemScenario {
    SimulationState initial_state;
    std::vector<PhysicalBodyMetadata> metadata;
};

inline constexpr double solar_system_recommended_dt_s = 21600.0;
inline constexpr const char* solar_system_epoch =
    "J2000.0 (JD 2451545.0 TDB; 2000-01-01 12:00 TDB)";
inline constexpr const char* solar_system_frame =
    "barycentric Cartesian axes aligned with the mean ecliptic and equinox of J2000";
inline constexpr const char* solar_system_position_source =
    "JPL Solar System Dynamics, Approximate Positions of the Planets, Table 1";
inline constexpr const char* solar_system_physical_source =
    "NASA NSSDCA planetary fact sheets (mass and volumetric mean radius)";

SolarSystemScenario make_approximate_solar_system();

}  // namespace nbody
