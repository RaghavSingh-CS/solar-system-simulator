#include "nbody/visualization/trails.hpp"

#include <cmath>
#include <stdexcept>

namespace nbody::visualization {

TrailHistory::TrailHistory(
    const SimulationState& state,
    std::size_t capacity,
    double sample_interval_s)
    : capacity_(capacity), sample_interval_s_(sample_interval_s) {
    if (capacity_ < 2 || !(sample_interval_s_ > 0.0) || !std::isfinite(sample_interval_s_)) {
        throw std::invalid_argument("trail capacity must be at least two and cadence positive");
    }
    reset(state);
}

void TrailHistory::reset(const SimulationState& state) {
    positions_.assign(state.bodies.size(), {});
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        positions_[index].push_back(state.bodies[index].position_m);
    }
    last_sample_time_s_ = state.time_s;
}

void TrailHistory::sample(const SimulationState& state) {
    if (state.bodies.size() != positions_.size()) {
        throw std::invalid_argument("trail state body count changed without reset");
    }
    if (state.time_s < last_sample_time_s_) {
        throw std::invalid_argument("trail state time moved backwards without reset");
    }
    if (state.time_s - last_sample_time_s_ < sample_interval_s_) {
        return;
    }
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        auto& trail = positions_[index];
        trail.push_back(state.bodies[index].position_m);
        while (trail.size() > capacity_) {
            trail.pop_front();
        }
    }
    last_sample_time_s_ = state.time_s;
}

void TrailHistory::set_capacity(std::size_t capacity) {
    if (capacity < 2) {
        throw std::invalid_argument("trail capacity must be at least two");
    }
    capacity_ = capacity;
    for (auto& trail : positions_) {
        while (trail.size() > capacity_) {
            trail.pop_front();
        }
    }
}

std::size_t TrailHistory::capacity() const noexcept { return capacity_; }
double TrailHistory::sample_interval_s() const noexcept { return sample_interval_s_; }
const std::vector<std::deque<Vec3>>& TrailHistory::positions() const noexcept {
    return positions_;
}

}  // namespace nbody::visualization
