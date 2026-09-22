#pragma once

#include "nbody/simulation.hpp"

#include <cstddef>
#include <cstdint>

namespace nbody {

struct TwoBodyScenario {
    SimulationState initial_state;
    double separation_m{0.0};
    double orbital_period_s{0.0};
};

TwoBodyScenario make_analytic_circular_two_body();

SimulationState make_seeded_synthetic_system(
    std::size_t body_count,
    std::uint64_t seed = 0x4e424f4459ULL);

}  // namespace nbody
