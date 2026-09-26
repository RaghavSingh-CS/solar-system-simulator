#pragma once

#include "nbody/diagnostics.hpp"

namespace nbody::visualization {

struct ConservationDrift {
    double energy_relative{0.0};
    double angular_momentum_relative{0.0};
    double maximum_energy_relative{0.0};
    double maximum_angular_momentum_relative{0.0};
};

class ConservationTracker {
public:
    explicit ConservationTracker(const SimulationState& reference_state);

    void reset(const SimulationState& reference_state);
    const ConservationDrift& update(const SimulationState& state);
    const ConservationDrift& drift() const noexcept;

private:
    double reference_energy_j_{0.0};
    Vec3 reference_angular_momentum_kg_m2_s_;
    ConservationDrift drift_;
};

}  // namespace nbody::visualization
