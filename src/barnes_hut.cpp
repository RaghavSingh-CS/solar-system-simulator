#include "nbody/barnes_hut.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nbody {
namespace {

constexpr std::size_t no_node = std::numeric_limits<std::size_t>::max();

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* context) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        throw std::length_error(std::string(context) + " size overflow");
    }
    return lhs + rhs;
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* context) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::length_error(std::string(context) + " size overflow");
    }
    return lhs * rhs;
}

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double finite_span(double minimum, double maximum) {
    if (minimum < 0.0 && maximum > 0.0) {
        const double negative_magnitude = -minimum;
        if (negative_magnitude > std::numeric_limits<double>::max() - maximum) {
            throw std::domain_error("Barnes-Hut coordinate span is not representable");
        }
        return negative_magnitude + maximum;
    }
    const double span = maximum - minimum;
    if (!std::isfinite(span) || span < 0.0) {
        throw std::domain_error("Barnes-Hut coordinate span is not representable");
    }
    return span;
}

double midpoint(double minimum, double maximum) {
    const double result = minimum * 0.5 + maximum * 0.5;
    if (!std::isfinite(result)) {
        throw std::domain_error("Barnes-Hut cell center is not representable");
    }
    return result;
}

double bounded_shift(double value, double magnitude, bool positive) {
    const double maximum = std::numeric_limits<double>::max();
    if (positive) {
        return value > maximum - magnitude ? maximum : value + magnitude;
    }
    return value < -maximum + magnitude ? -maximum : value - magnitude;
}

double stable_norm(const Vec3& value) {
    return std::hypot(value.x, value.y, value.z);
}

struct CompensatedVec3 {
    Vec3 sum;
    Vec3 correction;

    void add(const Vec3& value) {
        const auto add_component = [](double input, double& accumulated, double& compensation) {
            const double adjusted = input - compensation;
            const double next = accumulated + adjusted;
            compensation = (next - accumulated) - adjusted;
            accumulated = next;
        };
        add_component(value.x, sum.x, correction.x);
        add_component(value.y, sum.y, correction.y);
        add_component(value.z, sum.z, correction.z);
    }
};

struct OctreeNode {
    Vec3 center_m;
    Vec3 center_of_mass_m;
    double side_length_m{0.0};
    double mass_kg{0.0};
    std::size_t begin{0};
    std::size_t end{0};
    std::array<std::size_t, 8> children{
        no_node, no_node, no_node, no_node, no_node, no_node, no_node, no_node};
    bool leaf{true};
};

struct Tree {
    const SimulationState& state;
    const BarnesHutConfiguration& configuration;
    std::vector<OctreeNode> nodes;
    std::vector<std::size_t> permutation;
    std::vector<std::size_t> scratch;
    std::vector<std::size_t> rank;
    std::size_t effective_maximum_nodes{0};
    BarnesHutStatistics statistics;

    Tree(const SimulationState& input_state, const BarnesHutConfiguration& input_configuration)
        : state(input_state), configuration(input_configuration) {
        const std::size_t count = state.bodies.size();
        const std::size_t fixed_bytes = barnes_hut_fixed_memory_bytes(count);
        if (fixed_bytes > configuration.memory_budget_bytes) {
            throw std::length_error("Barnes-Hut fixed working storage exceeds memory budget");
        }
        const std::size_t node_bytes = configuration.memory_budget_bytes - fixed_bytes;
        effective_maximum_nodes = std::min(
            configuration.maximum_nodes,
            node_bytes / sizeof(OctreeNode));
        if (count != 0 && effective_maximum_nodes == 0) {
            throw std::length_error("Barnes-Hut memory budget cannot hold the root node");
        }

        permutation.resize(count);
        scratch.resize(count);
        rank.resize(count);
        for (std::size_t index = 0; index < count; ++index) {
            permutation[index] = index;
        }

        const std::size_t estimated_nodes = count == 0
            ? 0
            : std::min(effective_maximum_nodes, checked_add(count, count, "Barnes-Hut reserve"));
        reserve_nodes(estimated_nodes);
    }

    void reserve_nodes(std::size_t requested) {
        if (requested > effective_maximum_nodes) {
            throw std::length_error("Barnes-Hut node budget exceeded");
        }
        nodes.reserve(requested);
    }

    std::size_t append_node(OctreeNode node) {
        if (nodes.size() >= effective_maximum_nodes) {
            throw std::length_error("Barnes-Hut node or memory budget exceeded");
        }
        if (nodes.size() == nodes.capacity()) {
            const std::size_t doubled = nodes.capacity() == 0
                ? 1
                : checked_add(nodes.capacity(), nodes.capacity(), "Barnes-Hut node capacity");
            reserve_nodes(std::min(effective_maximum_nodes, doubled));
        }
        nodes.push_back(std::move(node));
        return nodes.size() - 1;
    }

    std::pair<double, Vec3> mass_and_center(std::size_t begin, std::size_t end) const {
        double total_mass = 0.0;
        double mass_correction = 0.0;
        Vec3 center;
        for (std::size_t offset = begin; offset < end; ++offset) {
            const Body& body = state.bodies[permutation[offset]];
            const double adjusted_mass = body.mass_kg - mass_correction;
            const double next_mass = total_mass + adjusted_mass;
            if (!std::isfinite(next_mass)) {
                throw std::domain_error("Barnes-Hut aggregate mass is not representable");
            }
            mass_correction = (next_mass - total_mass) - adjusted_mass;
            const double ratio = body.mass_kg / next_mass;
            const Vec3 displacement = body.position_m - center;
            if (!finite(displacement)) {
                throw std::domain_error("Barnes-Hut center-of-mass displacement is not representable");
            }
            center += displacement * ratio;
            if (!finite(center)) {
                throw std::domain_error("Barnes-Hut center of mass is not representable");
            }
            total_mass = next_mass;
        }
        return {total_mass, center};
    }

    static std::size_t octant(const Vec3& position, const Vec3& center) {
        return (position.x >= center.x ? 1U : 0U) |
            (position.y >= center.y ? 2U : 0U) |
            (position.z >= center.z ? 4U : 0U);
    }

    std::size_t build_node(
        std::size_t begin,
        std::size_t end,
        const Vec3& center,
        double side_length,
        std::size_t depth) {
        const auto aggregate = mass_and_center(begin, end);
        OctreeNode node;
        node.center_m = center;
        node.center_of_mass_m = aggregate.second;
        node.side_length_m = side_length;
        node.mass_kg = aggregate.first;
        node.begin = begin;
        node.end = end;
        const std::size_t node_index = append_node(node);
        statistics.maximum_depth_reached = std::max(statistics.maximum_depth_reached, depth);

        const std::size_t body_count = end - begin;
        const double child_side = side_length * 0.5;
        if (body_count <= configuration.leaf_capacity ||
            depth >= configuration.maximum_depth ||
            side_length <= configuration.minimum_cell_size_m ||
            !(child_side > 0.0) || !std::isfinite(child_side)) {
            ++statistics.leaf_count;
            return node_index;
        }

        std::array<std::size_t, 8> counts{};
        for (std::size_t offset = begin; offset < end; ++offset) {
            ++counts[octant(state.bodies[permutation[offset]].position_m, center)];
        }
        std::array<std::size_t, 8> starts{};
        starts[0] = begin;
        for (std::size_t index = 1; index < starts.size(); ++index) {
            starts[index] = starts[index - 1] + counts[index - 1];
        }
        auto write_positions = starts;
        for (std::size_t offset = begin; offset < end; ++offset) {
            const std::size_t body_index = permutation[offset];
            scratch[write_positions[octant(state.bodies[body_index].position_m, center)]++] =
                body_index;
        }
        std::copy(scratch.begin() + static_cast<std::ptrdiff_t>(begin),
                  scratch.begin() + static_cast<std::ptrdiff_t>(end),
                  permutation.begin() + static_cast<std::ptrdiff_t>(begin));

        nodes[node_index].leaf = false;
        const double offset = side_length * 0.25;
        for (std::size_t child = 0; child < counts.size(); ++child) {
            if (counts[child] == 0) {
                continue;
            }
            const Vec3 child_center{
                bounded_shift(center.x, offset, (child & 1U) != 0U),
                bounded_shift(center.y, offset, (child & 2U) != 0U),
                bounded_shift(center.z, offset, (child & 4U) != 0U),
            };
            if (!finite(child_center)) {
                throw std::domain_error("Barnes-Hut child cell is not representable");
            }
            const std::size_t child_begin = starts[child];
            const std::size_t child_index = build_node(
                child_begin,
                child_begin + counts[child],
                child_center,
                child_side,
                depth + 1);
            nodes[node_index].children[child] = child_index;
        }
        return node_index;
    }

    void build() {
        if (state.bodies.empty()) {
            return;
        }

        Vec3 minimum = state.bodies.front().position_m;
        Vec3 maximum = minimum;
        for (const Body& body : state.bodies) {
            minimum.x = std::min(minimum.x, body.position_m.x);
            minimum.y = std::min(minimum.y, body.position_m.y);
            minimum.z = std::min(minimum.z, body.position_m.z);
            maximum.x = std::max(maximum.x, body.position_m.x);
            maximum.y = std::max(maximum.y, body.position_m.y);
            maximum.z = std::max(maximum.z, body.position_m.z);
        }
        const Vec3 center{
            midpoint(minimum.x, maximum.x),
            midpoint(minimum.y, maximum.y),
            midpoint(minimum.z, maximum.z),
        };
        const double side_length = std::max({
            finite_span(minimum.x, maximum.x),
            finite_span(minimum.y, maximum.y),
            finite_span(minimum.z, maximum.z),
        });
        build_node(0, state.bodies.size(), center, side_length, 0);
        for (std::size_t offset = 0; offset < permutation.size(); ++offset) {
            rank[permutation[offset]] = offset;
        }
        statistics.node_count = nodes.size();
        statistics.working_memory_bytes = checked_add(
            barnes_hut_fixed_memory_bytes(state.bodies.size()),
            checked_multiply(nodes.capacity(), sizeof(OctreeNode), "Barnes-Hut node memory"),
            "Barnes-Hut working memory");
    }
};

Vec3 acceleration_contribution(
    const Vec3& displacement,
    double source_mass,
    double gravitational_constant_value) {
    const double distance = stable_norm(displacement);
    if (!(distance > 0.0) || !std::isfinite(distance)) {
        throw std::domain_error("coincident or nonrepresentable Barnes-Hut geometry");
    }

    const double ordinary_magnitude =
        (gravitational_constant_value * source_mass) / distance / distance;
    if (ordinary_magnitude > 0.0 && std::isfinite(ordinary_magnitude)) {
        const Vec3 ordinary = (displacement / distance) * ordinary_magnitude;
        const bool lost_nonzero_component =
            (displacement.x != 0.0 && ordinary.x == 0.0) ||
            (displacement.y != 0.0 && ordinary.y == 0.0) ||
            (displacement.z != 0.0 && ordinary.z == 0.0);
        if (finite(ordinary) && !lost_nonzero_component) {
            return ordinary;
        }
    }

    const auto component = [&](double displacement_component) {
        if (displacement_component == 0.0) {
            return 0.0;
        }

        int gravitational_exponent = 0;
        int mass_exponent = 0;
        int displacement_exponent = 0;
        int distance_exponent = 0;
        const double gravitational_fraction =
            std::frexp(gravitational_constant_value, &gravitational_exponent);
        const double mass_fraction = std::frexp(source_mass, &mass_exponent);
        const double displacement_fraction =
            std::frexp(std::abs(displacement_component), &displacement_exponent);
        const double distance_fraction = std::frexp(distance, &distance_exponent);

        double fraction = gravitational_fraction * mass_fraction * displacement_fraction /
            (distance_fraction * distance_fraction * distance_fraction);
        int fraction_exponent = 0;
        fraction = std::frexp(fraction, &fraction_exponent);
        const long long combined_exponent =
            static_cast<long long>(gravitational_exponent) + mass_exponent +
            displacement_exponent - 3LL * distance_exponent + fraction_exponent;
        if (combined_exponent > std::numeric_limits<double>::max_exponent) {
            throw std::domain_error("Barnes-Hut acceleration is not representable");
        }
        if (combined_exponent < std::numeric_limits<double>::min_exponent -
                std::numeric_limits<double>::digits) {
            return std::copysign(0.0, displacement_component);
        }
        const double magnitude = std::ldexp(fraction, static_cast<int>(combined_exponent));
        if (!std::isfinite(magnitude)) {
            throw std::domain_error("Barnes-Hut acceleration is not representable");
        }
        return std::copysign(magnitude, displacement_component);
    };

    const Vec3 contribution{
        component(displacement.x),
        component(displacement.y),
        component(displacement.z),
    };
    if (!finite(contribution)) {
        throw std::domain_error("Barnes-Hut acceleration is not representable");
    }
    return contribution;
}

void validate_configuration(const BarnesHutConfiguration& configuration) {
    if (!(configuration.theta > 0.0) || configuration.theta > 2.0 ||
        !std::isfinite(configuration.theta)) {
        throw std::invalid_argument("Barnes-Hut theta must be finite and in (0, 2]");
    }
    if (configuration.leaf_capacity == 0) {
        throw std::invalid_argument("Barnes-Hut leaf capacity must be positive");
    }
    if (configuration.maximum_depth == 0 || configuration.maximum_depth > 128) {
        throw std::invalid_argument("Barnes-Hut maximum depth must be in [1, 128]");
    }
    if (configuration.minimum_cell_size_m < 0.0 ||
        !std::isfinite(configuration.minimum_cell_size_m)) {
        throw std::invalid_argument("Barnes-Hut minimum cell size must be finite and nonnegative");
    }
    if (configuration.maximum_nodes == 0 ||
        configuration.maximum_traversal_work == 0 ||
        configuration.memory_budget_bytes == 0) {
        throw std::invalid_argument(
            "Barnes-Hut node, traversal, and memory budgets must be positive");
    }
}

void validate_state(const SimulationState& state, double gravitational_constant_value) {
    if (!(gravitational_constant_value > 0.0) || !std::isfinite(gravitational_constant_value)) {
        throw std::invalid_argument("gravitational constant must be finite and positive");
    }
    for (const Body& body : state.bodies) {
        if (!(body.mass_kg > 0.0) || !std::isfinite(body.mass_kg)) {
            throw std::invalid_argument("body mass must be finite and positive");
        }
        if (!finite(body.position_m)) {
            throw std::invalid_argument("body position must be finite");
        }
    }
}

void accumulate_node(
    const Tree& tree,
    std::size_t node_index,
    std::size_t target_index,
    double gravitational_constant_value,
    CompensatedVec3& acceleration,
    BarnesHutStatistics& statistics) {
    const auto consume_work = [&]() {
        const std::size_t used = checked_add(
            statistics.visited_nodes,
            statistics.direct_body_interactions,
            "Barnes-Hut traversal work");
        if (used >= tree.configuration.maximum_traversal_work) {
            throw std::length_error("Barnes-Hut traversal work budget exceeded");
        }
    };
    consume_work();
    const OctreeNode& node = tree.nodes[node_index];
    ++statistics.visited_nodes;
    const std::size_t target_rank = tree.rank[target_index];
    const bool contains_target = target_rank >= node.begin && target_rank < node.end;

    if (node.leaf) {
        for (std::size_t offset = node.begin; offset < node.end; ++offset) {
            const std::size_t source_index = tree.permutation[offset];
            if (source_index == target_index) {
                continue;
            }
            const Vec3 displacement =
                tree.state.bodies[source_index].position_m -
                tree.state.bodies[target_index].position_m;
            acceleration.add(acceleration_contribution(
                displacement,
                tree.state.bodies[source_index].mass_kg,
                gravitational_constant_value));
            consume_work();
            ++statistics.direct_body_interactions;
        }
        return;
    }

    const Vec3 displacement =
        node.center_of_mass_m - tree.state.bodies[target_index].position_m;
    const double distance = stable_norm(displacement);
    if (!contains_target && distance > 0.0 && std::isfinite(distance) &&
        node.side_length_m / distance < tree.configuration.theta) {
        acceleration.add(acceleration_contribution(
            displacement, node.mass_kg, gravitational_constant_value));
        ++statistics.approximated_nodes;
        return;
    }

    for (std::size_t child : node.children) {
        if (child != no_node) {
            accumulate_node(
                tree,
                child,
                target_index,
                gravitational_constant_value,
                acceleration,
                statistics);
        }
    }
}

}  // namespace

std::size_t barnes_hut_fixed_memory_bytes(std::size_t body_count) {
    const std::size_t index_bytes = checked_multiply(
        checked_multiply(body_count, 3, "Barnes-Hut index arrays"),
        sizeof(std::size_t),
        "Barnes-Hut index arrays");
    const std::size_t output_bytes =
        checked_multiply(body_count, sizeof(Vec3), "Barnes-Hut output array");
    return checked_add(index_bytes, output_bytes, "Barnes-Hut fixed storage");
}

SequentialBarnesHutForceBackend::SequentialBarnesHutForceBackend(
    BarnesHutConfiguration configuration)
    : configuration_(configuration) {
    validate_configuration(configuration_);
}

std::vector<Vec3> SequentialBarnesHutForceBackend::compute_accelerations(
    const SimulationState& state,
    double gravitational_constant_value) const {
    validate_state(state, gravitational_constant_value);

    BarnesHutStatistics candidate_statistics;
    candidate_statistics.body_count = state.bodies.size();
    if (state.bodies.empty()) {
        last_statistics_ = candidate_statistics;
        return {};
    }

    const auto build_start = std::chrono::steady_clock::now();
    Tree tree(state, configuration_);
    tree.build();
    const auto build_end = std::chrono::steady_clock::now();
    candidate_statistics = tree.statistics;
    candidate_statistics.body_count = state.bodies.size();
    candidate_statistics.tree_build_seconds =
        std::chrono::duration<double>(build_end - build_start).count();

    const auto force_start = std::chrono::steady_clock::now();
    std::vector<Vec3> accelerations(state.bodies.size());
    for (std::size_t target = 0; target < state.bodies.size(); ++target) {
        CompensatedVec3 acceleration;
        accumulate_node(
            tree,
            0,
            target,
            gravitational_constant_value,
            acceleration,
            candidate_statistics);
        if (!finite(acceleration.sum)) {
            throw std::domain_error("Barnes-Hut accumulated acceleration is not representable");
        }
        accelerations[target] = acceleration.sum;
    }
    const auto force_end = std::chrono::steady_clock::now();
    candidate_statistics.force_evaluation_seconds =
        std::chrono::duration<double>(force_end - force_start).count();
    last_statistics_ = candidate_statistics;
    return accelerations;
}

}  // namespace nbody
