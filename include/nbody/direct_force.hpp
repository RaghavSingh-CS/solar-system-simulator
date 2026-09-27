#pragma once

#include "nbody/simulation.hpp"

#include <cstddef>

namespace nbody {

// Shared binary64 direct-force target kernel used by the CPU backends. Input
// validation is separated so parallel callers can validate once before their
// worker regions.
void validate_direct_force_inputs(
    const SimulationState& state,
    double gravitational_constant_value);

bool try_compute_direct_acceleration(
    const SimulationState& state,
    std::size_t target_index,
    double gravitational_constant_value,
    Vec3& acceleration) noexcept;

}  // namespace nbody
