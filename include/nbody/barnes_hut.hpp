#pragma once

#include "nbody/simulation.hpp"

#include <cstddef>
#include <vector>

namespace nbody {

// Sequential three-dimensional Barnes--Hut policy. Lengths are SI metres.
// The solver uses the same unsoftened Newtonian point-mass law as the direct
// backends; coincident positions therefore remain invalid.
struct BarnesHutConfiguration {
    double theta{0.25};
    std::size_t leaf_capacity{8};
    std::size_t maximum_depth{64};
    double minimum_cell_size_m{1.0e-6};
    std::size_t maximum_nodes{1'000'000};
    std::size_t maximum_traversal_work{1'000'000'000};
    std::size_t memory_budget_bytes{256U * 1024U * 1024U};
};

struct BarnesHutStatistics {
    std::size_t body_count{0};
    std::size_t node_count{0};
    std::size_t leaf_count{0};
    std::size_t maximum_depth_reached{0};
    std::size_t visited_nodes{0};
    std::size_t approximated_nodes{0};
    std::size_t direct_body_interactions{0};
    std::size_t working_memory_bytes{0};
    double tree_build_seconds{0.0};
    double force_evaluation_seconds{0.0};
};

// Returns the fixed index-array and acceleration-output requirement used before
// any octree nodes are admitted. Throws std::length_error on overflow.
std::size_t barnes_hut_fixed_memory_bytes(std::size_t body_count);

class SequentialBarnesHutForceBackend final : public ForceBackend {
public:
    explicit SequentialBarnesHutForceBackend(BarnesHutConfiguration configuration = {});

    const char* name() const noexcept override { return "sequential-barnes-hut"; }
    std::vector<Vec3> compute_accelerations(
        const SimulationState& state,
        double gravitational_constant_value = gravitational_constant) const override;

    const BarnesHutConfiguration& configuration() const noexcept { return configuration_; }
    // Statistics describe the most recent successful call. Like the mutable
    // integration state that consumes it, one backend instance is single-owner;
    // concurrent callers must use separate instances.
    BarnesHutStatistics last_statistics() const noexcept { return last_statistics_; }

private:
    BarnesHutConfiguration configuration_;
    mutable BarnesHutStatistics last_statistics_;
};

}  // namespace nbody
