#pragma once

#include "nbody/simulation.hpp"

#include <cstddef>
#include <deque>
#include <vector>

namespace nbody::visualization {

class TrailHistory {
public:
    TrailHistory(const SimulationState& state, std::size_t capacity, double sample_interval_s);

    void reset(const SimulationState& state);
    void sample(const SimulationState& state);
    void set_capacity(std::size_t capacity);

    std::size_t capacity() const noexcept;
    double sample_interval_s() const noexcept;
    const std::vector<std::deque<Vec3>>& positions() const noexcept;

private:
    std::size_t capacity_;
    double sample_interval_s_;
    double last_sample_time_s_{0.0};
    std::vector<std::deque<Vec3>> positions_;
};

std::vector<Vec3> parent_relative_trail(
    const std::deque<Vec3>& body_positions,
    const std::deque<Vec3>& parent_positions);

}  // namespace nbody::visualization
