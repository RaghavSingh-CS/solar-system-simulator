#include "nbody/barnes_hut.hpp"
#include "nbody/cuda_barnes_hut.hpp"
#include "nbody/scenarios.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

bool same_node(
    const nbody::BarnesHutPackedNode& lhs,
    const nbody::BarnesHutPackedNode& rhs) {
    if (lhs.center_x_m != rhs.center_x_m ||
        lhs.center_y_m != rhs.center_y_m ||
        lhs.center_z_m != rhs.center_z_m ||
        lhs.center_of_mass_x_m != rhs.center_of_mass_x_m ||
        lhs.center_of_mass_y_m != rhs.center_of_mass_y_m ||
        lhs.center_of_mass_z_m != rhs.center_of_mass_z_m ||
        lhs.side_length_m != rhs.side_length_m ||
        lhs.mass_kg != rhs.mass_kg ||
        lhs.begin != rhs.begin || lhs.end != rhs.end ||
        lhs.leaf != rhs.leaf) {
        return false;
    }
    for (std::size_t child = 0; child < 8; ++child) {
        if (lhs.children[child] != rhs.children[child]) {
            return false;
        }
    }
    return true;
}

void validate_tree(
    const nbody::SimulationState& state,
    const nbody::BarnesHutConfiguration& configuration = {}) {
    const auto first = nbody::build_barnes_hut_packed_tree(state, configuration);
    const auto second = nbody::build_barnes_hut_packed_tree(state, configuration);
    require(first.nodes.size() == second.nodes.size(), "packed node count changed");
    require(first.permutation == second.permutation, "packed permutation changed");
    require(first.inverse_rank == second.inverse_rank, "packed inverse rank changed");
    for (std::size_t index = 0; index < first.nodes.size(); ++index) {
        require(same_node(first.nodes[index], second.nodes[index]), "packed node changed");
    }

    require(!first.nodes.empty(), "non-empty state produced no packed root");
    require(first.nodes.front().begin == 0, "root begin is not zero");
    require(first.nodes.front().end == state.bodies.size(), "root range is incomplete");
    require(first.permutation.size() == state.bodies.size(), "permutation size differs");
    require(first.inverse_rank.size() == state.bodies.size(), "rank size differs");
    require(first.storage_bytes >= nbody::barnes_hut_packed_memory_bytes(
                state.bodies.size(), first.nodes.size()),
            "actual packed capacity is below its size lower bound");

    std::vector<bool> seen(state.bodies.size(), false);
    for (std::size_t rank = 0; rank < first.permutation.size(); ++rank) {
        const std::uint32_t body = first.permutation[rank];
        require(body < state.bodies.size(), "permutation body is out of range");
        require(!seen[body], "permutation contains a duplicate body");
        seen[body] = true;
        require(first.inverse_rank[body] == rank, "inverse rank does not invert permutation");
    }

    for (std::size_t node_index = 0; node_index < first.nodes.size(); ++node_index) {
        const auto& node = first.nodes[node_index];
        require(node.begin < node.end, "packed node has an empty range");
        require(node.end <= state.bodies.size(), "packed node range exceeds body count");
        require(node.mass_kg > 0.0 && std::isfinite(node.mass_kg),
                "packed node mass is invalid");
        require(node.side_length_m >= 0.0 && std::isfinite(node.side_length_m),
                "packed node side length is invalid");
        const double half_side = node.side_length_m * 0.5;
        const double bound_tolerance =
            std::max(1.0, node.side_length_m) *
            std::numeric_limits<double>::epsilon() * 8.0;
        for (std::uint32_t offset = node.begin; offset < node.end; ++offset) {
            const auto& position = state.bodies[first.permutation[offset]].position_m;
            require(std::abs(position.x - node.center_x_m) <=
                        half_side + bound_tolerance &&
                    std::abs(position.y - node.center_y_m) <=
                        half_side + bound_tolerance &&
                    std::abs(position.z - node.center_z_m) <=
                        half_side + bound_tolerance,
                    "packed node spatial bounds do not contain their body range");
        }
        std::uint32_t next_child_begin = node.begin;
        bool has_child = false;
        for (std::uint32_t child : node.children) {
            if (child == nbody::barnes_hut_no_packed_index) {
                continue;
            }
            has_child = true;
            require(node.leaf == 0, "leaf contains a child index");
            require(child < first.nodes.size(), "child index is out of range");
            require(child > node_index, "child does not follow parent in deterministic order");
            const auto& child_node = first.nodes[child];
            require(child_node.begin >= node.begin && child_node.end <= node.end,
                    "child range is outside parent range");
            require(child_node.begin == next_child_begin,
                    "child ranges do not form a contiguous deterministic partition");
            next_child_begin = child_node.end;
        }
        require(node.leaf != 0 || (has_child && next_child_begin == node.end),
                "internal-node child ranges do not cover the parent range");

        double node_mass = 0.0;
        long double weighted_node_x = 0.0L;
        long double weighted_node_y = 0.0L;
        long double weighted_node_z = 0.0L;
        double node_position_scale = 1.0;
        for (std::uint32_t offset = node.begin; offset < node.end; ++offset) {
            const auto& body = state.bodies[first.permutation[offset]];
            node_mass += body.mass_kg;
            weighted_node_x +=
                static_cast<long double>(body.mass_kg) * body.position_m.x;
            weighted_node_y +=
                static_cast<long double>(body.mass_kg) * body.position_m.y;
            weighted_node_z +=
                static_cast<long double>(body.mass_kg) * body.position_m.z;
            node_position_scale = std::max({
                node_position_scale,
                std::abs(body.position_m.x),
                std::abs(body.position_m.y),
                std::abs(body.position_m.z),
            });
        }
        require(std::abs(node.mass_kg - node_mass) <=
                    std::numeric_limits<double>::epsilon() * node_mass * 16.0,
                "packed node mass differs from its body range");
        const auto require_node_center = [&](double packed_value, long double weighted) {
            const double expected = static_cast<double>(weighted / node_mass);
            require(std::isfinite(packed_value) &&
                        std::abs(packed_value - expected) <=
                            node_position_scale * 1.0e-12,
                    "packed node center of mass differs from its body range");
        };
        require_node_center(node.center_of_mass_x_m, weighted_node_x);
        require_node_center(node.center_of_mass_y_m, weighted_node_y);
        require_node_center(node.center_of_mass_z_m, weighted_node_z);
    }

    double total_mass = 0.0;
    long double weighted_x = 0.0L;
    long double weighted_y = 0.0L;
    long double weighted_z = 0.0L;
    double position_scale = 1.0;
    for (const auto& body : state.bodies) {
        total_mass += body.mass_kg;
        weighted_x += static_cast<long double>(body.mass_kg) * body.position_m.x;
        weighted_y += static_cast<long double>(body.mass_kg) * body.position_m.y;
        weighted_z += static_cast<long double>(body.mass_kg) * body.position_m.z;
        position_scale = std::max({
            position_scale,
            std::abs(body.position_m.x),
            std::abs(body.position_m.y),
            std::abs(body.position_m.z),
        });
    }
    require(std::abs(first.nodes.front().mass_kg - total_mass) <=
                std::numeric_limits<double>::epsilon() * total_mass * 8.0,
            "packed root mass differs from source bodies");
    const auto require_center = [&](double packed_value, long double weighted) {
        const double expected = static_cast<double>(weighted / total_mass);
        const double tolerance = position_scale * 1.0e-12;
        require(std::abs(packed_value - expected) <= tolerance,
                "packed root center of mass differs from source bodies");
    };
    require_center(first.nodes.front().center_of_mass_x_m, weighted_x);
    require_center(first.nodes.front().center_of_mass_y_m, weighted_y);
    require_center(first.nodes.front().center_of_mass_z_m, weighted_z);
}

void test_empty_and_single_body() {
    const nbody::SimulationState empty;
    const auto empty_tree = nbody::build_barnes_hut_packed_tree(empty);
    require(empty_tree.nodes.empty(), "empty state produced nodes");
    require(empty_tree.storage_bytes == 0, "empty packed tree charged storage");

    nbody::SimulationState single;
    single.bodies.push_back({"single", 3.0, {1.0, -2.0, 4.0}, {}});
    const auto tree = nbody::build_barnes_hut_packed_tree(single);
    require(tree.nodes.size() == 1, "single body did not produce one node");
    require(tree.nodes.front().leaf == 1, "single-body node is not a leaf");
    require(tree.permutation.front() == 0 && tree.inverse_rank.front() == 0,
            "single-body mapping changed");
}

void test_two_body_and_degenerate_fixtures() {
    nbody::SimulationState two_body;
    two_body.bodies = {
        {"left", 2.0, {-4.0, 1.0, 0.0}, {}},
        {"right", 3.0, {6.0, -2.0, 5.0}, {}},
    };
    nbody::BarnesHutConfiguration split;
    split.leaf_capacity = 1;
    validate_tree(two_body, split);

    nbody::SimulationState coincident;
    coincident.bodies = {
        {"coincident-a", 2.0, {7.0, -3.0, 11.0}, {}},
        {"coincident-b", 5.0, {7.0, -3.0, 11.0}, {}},
    };
    split.minimum_cell_size_m = 0.0;
    split.maximum_depth = 8;
    validate_tree(coincident, split);

    nbody::SimulationState near_degenerate;
    near_degenerate.bodies = {
        {"near-a", 2.0, {0.0, 0.0, 0.0}, {}},
        {"near-b", 5.0, {std::numeric_limits<double>::denorm_min(), 0.0, 0.0}, {}},
    };
    validate_tree(near_degenerate, split);
}

void test_budget_and_input_failures() {
    require_throws<std::length_error>(
        [] {
            (void)nbody::barnes_hut_packed_memory_bytes(
                std::numeric_limits<std::size_t>::max(),
                std::numeric_limits<std::size_t>::max());
        },
        "packed memory size overflow was accepted");
    const auto state = nbody::make_seeded_synthetic_system(64, 0x4355444154524545ULL);
    const auto tree = nbody::build_barnes_hut_packed_tree(state);
    require_throws<std::length_error>(
        [&] {
            (void)nbody::build_barnes_hut_packed_tree(
                state, {}, tree.storage_bytes - 1);
        },
        "packed budget below actual storage was accepted");
    require_throws<std::invalid_argument>(
        [&] { (void)nbody::build_barnes_hut_packed_tree(state, {}, 0); },
        "zero packed memory budget was accepted");

    nbody::BarnesHutConfiguration node_limited;
    node_limited.maximum_nodes = 1;
    node_limited.leaf_capacity = 1;
    require_throws<std::length_error>(
        [&] { (void)nbody::build_barnes_hut_packed_tree(state, node_limited); },
        "CPU tree node budget was bypassed by packing");

    auto nonfinite = state;
    nonfinite.bodies.front().position_m.x =
        std::numeric_limits<double>::infinity();
    require_throws<std::invalid_argument>(
        [&] { (void)nbody::build_barnes_hut_packed_tree(nonfinite); },
        "non-finite position was packed");
}

void test_cuda_header_contract() {
    static_assert(std::is_standard_layout<nbody::BarnesHutPackedNode>::value,
                  "packed node must have standard layout");
    static_assert(std::is_trivially_copyable<nbody::BarnesHutPackedNode>::value,
                  "packed node must be trivially copyable");
    static_assert(std::is_standard_layout<nbody::CudaBarnesHutFailure>::value,
                  "failure report must have standard layout");

    const nbody::CudaBarnesHutConfiguration configuration;
    require(configuration.device_index == 0, "unexpected default CUDA device");
    require(configuration.tree.theta == 0.25, "CUDA tree theta differs from CPU default");
    require(configuration.traversal_stack_capacity >= 449,
            "default CUDA stack cannot hold the depth-64 worst-case DFS frontier");
    require(configuration.host_failure_injection ==
                nbody::CudaBarnesHutHostFailureInjection::none,
            "CUDA host failure injection must be disabled by default");
    require(configuration.device_memory_budget_bytes > 0 &&
                configuration.host_staging_memory_budget_bytes > 0,
            "CUDA memory budgets must be explicit and positive");
}

}  // namespace

int main() {
    try {
        test_cuda_header_contract();
        test_empty_and_single_body();
        test_two_body_and_degenerate_fixtures();
        validate_tree(nbody::make_seeded_synthetic_system(257, 0x5041434b45445452ULL));
        test_budget_and_input_failures();
        std::cout << "Packed Barnes-Hut tree tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Packed Barnes-Hut tree test failure: " << error.what() << '\n';
        return 1;
    }
}
