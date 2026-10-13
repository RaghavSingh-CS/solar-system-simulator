#pragma once

#include "nbody/simulation.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
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
    int worker_threads{0};
    double tree_build_seconds{0.0};
    double force_evaluation_seconds{0.0};
};

// Returns the size-based lower bound for the three index arrays and
// acceleration output used before allocation. Actual vector capacities are
// charged separately after allocation. Throws std::length_error on overflow.
std::size_t barnes_hut_fixed_memory_bytes(std::size_t body_count);

// Pointer-free, fixed-width representation shared with CUDA Barnes--Hut.
// Node zero is the root for non-empty trees. Cell center plus side length
// defines the spatial bounds. begin/end index the deterministic permutation;
// the inverse rank array lets a traversal determine whether a node contains
// its target without relying on geometric roundoff.
inline constexpr std::uint32_t barnes_hut_no_packed_index =
    std::numeric_limits<std::uint32_t>::max();

struct BarnesHutPackedNode {
    double center_x_m{0.0};
    double center_y_m{0.0};
    double center_z_m{0.0};
    double center_of_mass_x_m{0.0};
    double center_of_mass_y_m{0.0};
    double center_of_mass_z_m{0.0};
    double side_length_m{0.0};
    double mass_kg{0.0};
    std::uint32_t begin{0};
    std::uint32_t end{0};
    std::uint32_t children[8]{
        barnes_hut_no_packed_index,
        barnes_hut_no_packed_index,
        barnes_hut_no_packed_index,
        barnes_hut_no_packed_index,
        barnes_hut_no_packed_index,
        barnes_hut_no_packed_index,
        barnes_hut_no_packed_index,
        barnes_hut_no_packed_index,
    };
    std::uint32_t leaf{1};
};

struct BarnesHutPackedTree {
    std::vector<BarnesHutPackedNode> nodes;
    std::vector<std::uint32_t> permutation;
    std::vector<std::uint32_t> inverse_rank;
    BarnesHutStatistics statistics;
    std::size_t storage_bytes{0};
};

// Size-only lower bound for the pointer-free node/permutation/rank snapshot.
// Actual vector capacities are checked by build_barnes_hut_packed_tree().
// Throws std::length_error if the calculation is not representable.
std::size_t barnes_hut_packed_memory_bytes(
    std::size_t body_count,
    std::size_t node_count);

// Reuses the validated deterministic CPU builder and converts its completed
// immutable tree into the GPU-readable representation. The packed budget is
// separate from BarnesHutConfiguration::memory_budget_bytes, which continues
// to bound transient CPU tree construction. No force evaluation occurs here.
BarnesHutPackedTree build_barnes_hut_packed_tree(
    const SimulationState& state,
    const BarnesHutConfiguration& configuration = {},
    std::size_t packed_memory_budget_bytes = 256U * 1024U * 1024U);

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

// OpenMP shares the exact Sequential Barnes--Hut tree construction and
// per-target traversal. Only independent target evaluation is parallel. A
// CPU-only build retains this type so callers receive an explicit unavailable
// diagnostic rather than a silent Sequential fallback.
class OpenMpBarnesHutForceBackend final : public ForceBackend {
public:
    OpenMpBarnesHutForceBackend(
        int thread_count,
        BarnesHutConfiguration configuration = {});

    const char* name() const noexcept override { return "openmp-barnes-hut"; }
    std::vector<Vec3> compute_accelerations(
        const SimulationState& state,
        double gravitational_constant_value = gravitational_constant) const override;

    int thread_count() const noexcept { return thread_count_; }
    const BarnesHutConfiguration& configuration() const noexcept { return configuration_; }
    // One backend instance is single-owner because successful calls publish
    // mutable statistics. Concurrent callers must use separate instances.
    BarnesHutStatistics last_statistics() const noexcept { return last_statistics_; }
    // Separate attempt diagnostic: unlike last_statistics(), this is updated
    // after a failed traversal so callers/tests can verify the hard work cap.
    // It never represents a partially committed force result.
    std::size_t last_attempt_traversal_work() const noexcept {
        return last_attempt_traversal_work_;
    }

private:
    int thread_count_{1};
    BarnesHutConfiguration configuration_;
    mutable BarnesHutStatistics last_statistics_;
    mutable std::size_t last_attempt_traversal_work_{0};
};

bool openmp_barnes_hut_compiled() noexcept;

}  // namespace nbody
