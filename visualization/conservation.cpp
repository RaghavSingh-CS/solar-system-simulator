#include "nbody/visualization/conservation.hpp"

#include <algorithm>
#include <cmath>

namespace nbody::visualization {

ConservationTracker::ConservationTracker(const SimulationState& reference_state) {
    reset(reference_state);
}

void ConservationTracker::reset(const SimulationState& reference_state) {
    const auto diagnostics = compute_diagnostics(reference_state);
    reference_energy_j_ = diagnostics.total_energy_j();
    reference_angular_momentum_kg_m2_s_ = diagnostics.angular_momentum_kg_m2_s;
    drift_ = {};
}

const ConservationDrift& ConservationTracker::update(const SimulationState& state) {
    const auto diagnostics = compute_diagnostics(state);
    const double current_energy = diagnostics.total_energy_j();
    drift_.energy_relative = std::abs(current_energy - reference_energy_j_) /
        std::max({std::abs(current_energy), std::abs(reference_energy_j_), 1.0});
    drift_.angular_momentum_relative =
        norm(diagnostics.angular_momentum_kg_m2_s - reference_angular_momentum_kg_m2_s_) /
        std::max(norm(reference_angular_momentum_kg_m2_s_), 1.0);
    drift_.maximum_energy_relative =
        std::max(drift_.maximum_energy_relative, drift_.energy_relative);
    drift_.maximum_angular_momentum_relative = std::max(
        drift_.maximum_angular_momentum_relative,
        drift_.angular_momentum_relative);
    return drift_;
}

const ConservationDrift& ConservationTracker::drift() const noexcept { return drift_; }

}  // namespace nbody::visualization
