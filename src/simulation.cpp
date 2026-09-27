#include "nbody/simulation.hpp"

#include "nbody/direct_force.hpp"

#include <cmath>
#include <stdexcept>

namespace nbody {

void validate_direct_force_inputs(
    const SimulationState& state,
    double gravitational_constant_value) {
    if (!(gravitational_constant_value > 0.0) || !std::isfinite(gravitational_constant_value)) {
        throw std::invalid_argument("gravitational constant must be finite and positive");
    }
    for (const auto& body : state.bodies) {
        if (!(body.mass_kg > 0.0) || !std::isfinite(body.mass_kg)) {
            throw std::invalid_argument("body mass must be finite and positive");
        }
    }
}

bool try_compute_direct_acceleration(
    const SimulationState& state,
    std::size_t target_index,
    double gravitational_constant_value,
    Vec3& acceleration) noexcept {
    if (target_index >= state.bodies.size()) {
        return false;
    }
    acceleration = {};
    for (std::size_t source_index = 0; source_index < state.bodies.size(); ++source_index) {
        if (target_index == source_index) {
            continue;
        }
        const Vec3 displacement =
            state.bodies[source_index].position_m - state.bodies[target_index].position_m;
        const double distance_squared = norm_squared(displacement);
        if (!(distance_squared > 0.0) || !std::isfinite(distance_squared)) {
            return false;
        }
        const double inverse_distance = 1.0 / std::sqrt(distance_squared);
        const double inverse_distance_cubed =
            inverse_distance * inverse_distance * inverse_distance;
        acceleration += displacement *
            (gravitational_constant_value * state.bodies[source_index].mass_kg *
             inverse_distance_cubed);
    }
    return true;
}

namespace {

class SequentialForceBackend final : public ForceBackend {
public:
    const char* name() const noexcept override { return "sequential"; }

    std::vector<Vec3> compute_accelerations(
        const SimulationState& state,
        double gravitational_constant_value) const override {
        return nbody::compute_accelerations(state, gravitational_constant_value);
    }
};

}  // namespace

const ForceBackend& sequential_force_backend() {
    static const SequentialForceBackend backend;
    return backend;
}

std::vector<Vec3> compute_accelerations(
    const SimulationState& state,
    double gravitational_constant_value) {
    validate_direct_force_inputs(state, gravitational_constant_value);
    std::vector<Vec3> accelerations(state.bodies.size());
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        if (!try_compute_direct_acceleration(
                state, i, gravitational_constant_value, accelerations[i])) {
            throw std::domain_error("coincident or invalid body positions");
        }
    }
    return accelerations;
}

void velocity_verlet_step(
    SimulationState& state,
    double dt_s,
    double gravitational_constant_value) {
    velocity_verlet_step(state, dt_s, sequential_force_backend(), gravitational_constant_value);
}

void velocity_verlet_step(
    SimulationState& state,
    double dt_s,
    const ForceBackend& force_backend,
    double gravitational_constant_value) {
    if (!(dt_s > 0.0) || !std::isfinite(dt_s)) {
        throw std::invalid_argument("time step must be finite and positive");
    }

    const auto old_accelerations =
        force_backend.compute_accelerations(state, gravitational_constant_value);
    const double half_dt_squared = 0.5 * dt_s * dt_s;

    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        state.bodies[i].position_m +=
            state.bodies[i].velocity_m_s * dt_s + old_accelerations[i] * half_dt_squared;
    }

    const auto new_accelerations =
        force_backend.compute_accelerations(state, gravitational_constant_value);
    const double half_dt = 0.5 * dt_s;
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        state.bodies[i].velocity_m_s +=
            (old_accelerations[i] + new_accelerations[i]) * half_dt;
    }
    state.time_s += dt_s;
}

}  // namespace nbody
