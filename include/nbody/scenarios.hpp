#pragma once

#include "nbody/simulation.hpp"

namespace nbody {

struct TwoBodyScenario {
    SimulationState initial_state;
    double separation_m{0.0};
    double orbital_period_s{0.0};
};

TwoBodyScenario make_analytic_circular_two_body();

}  // namespace nbody
