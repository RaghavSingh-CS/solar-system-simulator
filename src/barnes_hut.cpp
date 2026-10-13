#include "nbody/barnes_hut.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace nbody {
namespace {

constexpr std::size_t no_node = std::numeric_limits<std::size_t>::max();
constexpr int maximum_openmp_barnes_hut_threads = 256;

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

struct TraversalStatistics {
    // Counts every successful work-budget reservation, including work whose
    // numerical evaluation subsequently throws. This is intentionally
    // separate from the completed-work statistics published on success.
    std::size_t charged_work{0};
    std::size_t visited_nodes{0};
    std::size_t approximated_nodes{0};
    std::size_t direct_body_interactions{0};
};

template <typename TraversalBudget>
void reserve_traversal_work(
    TraversalBudget& budget,
    TraversalStatistics& statistics) {
    budget.consume();
    statistics.charged_work = checked_add(
        statistics.charged_work, 1, "Barnes-Hut charged traversal work");
}

class SequentialTraversalBudget {
public:
    explicit SequentialTraversalBudget(std::size_t limit) : limit_(limit) {}

    void consume() {
        if (consumed_ >= limit_) {
            throw std::length_error("Barnes-Hut traversal work budget exceeded");
        }
        ++consumed_;
    }

    std::size_t consumed() const noexcept { return consumed_; }

private:
    std::size_t limit_{0};
    std::size_t consumed_{0};
};

#ifdef NBODY_HAS_OPENMP
struct alignas(64) ThreadTraversalStatistics {
    TraversalStatistics values;
};

class ParallelTraversalBudget {
public:
    ParallelTraversalBudget(std::size_t limit, bool requires_runtime_tracking)
        : limit_(limit), requires_runtime_tracking_(requires_runtime_tracking) {}

    void consume() {
        if (!requires_runtime_tracking_) {
            return;
        }
        std::size_t current = consumed_.load(std::memory_order_relaxed);
        while (true) {
            if (exhausted_.load(std::memory_order_relaxed) || current >= limit_) {
                exhausted_.store(true, std::memory_order_relaxed);
                throw std::length_error("Barnes-Hut traversal work budget exceeded");
            }
            if (consumed_.compare_exchange_weak(
                    current,
                    current + 1,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                return;
            }
        }
    }

    bool exhausted() const noexcept {
        if (!requires_runtime_tracking_) {
            return false;
        }
        return exhausted_.load(std::memory_order_relaxed);
    }

    std::size_t consumed() const noexcept {
        return consumed_.load(std::memory_order_relaxed);
    }

    bool requires_runtime_tracking() const noexcept {
        return requires_runtime_tracking_;
    }

private:
    std::size_t limit_{0};
    bool requires_runtime_tracking_{false};
    std::atomic<std::size_t> consumed_{0};
    std::atomic<bool> exhausted_{false};
};
#endif

std::size_t actual_vector_bytes(
    std::size_t capacity,
    std::size_t element_size,
    const char* context) {
    return checked_multiply(capacity, element_size, context);
}

std::size_t fixed_storage_lower_bound(
    std::size_t body_count,
    std::size_t requested_additional_bytes) {
    return checked_add(
        barnes_hut_fixed_memory_bytes(body_count),
        requested_additional_bytes,
        "Barnes-Hut fixed storage lower bound");
}

void validate_fixed_storage_lower_bound(
    std::size_t body_count,
    std::size_t requested_additional_bytes,
    std::size_t memory_budget_bytes) {
    if (fixed_storage_lower_bound(body_count, requested_additional_bytes) >
        memory_budget_bytes) {
        throw std::length_error("Barnes-Hut fixed working storage exceeds memory budget");
    }
}

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
    std::size_t accounted_fixed_bytes{0};
    BarnesHutStatistics statistics;

    Tree(
        const SimulationState& input_state,
        const BarnesHutConfiguration& input_configuration,
        std::size_t output_capacity_bytes,
        std::size_t additional_capacity_bytes = 0)
        : state(input_state), configuration(input_configuration) {
        const std::size_t count = state.bodies.size();
        permutation.resize(count);
        scratch.resize(count);
        rank.resize(count);
        accounted_fixed_bytes = checked_add(
            checked_add(
                actual_vector_bytes(
                    permutation.capacity(),
                    sizeof(std::size_t),
                    "Barnes-Hut permutation capacity"),
                actual_vector_bytes(
                    scratch.capacity(),
                    sizeof(std::size_t),
                    "Barnes-Hut scratch capacity"),
                "Barnes-Hut index capacity"),
            actual_vector_bytes(
                rank.capacity(), sizeof(std::size_t), "Barnes-Hut rank capacity"),
            "Barnes-Hut index capacity");
        accounted_fixed_bytes = checked_add(
            accounted_fixed_bytes, output_capacity_bytes, "Barnes-Hut output capacity");
        accounted_fixed_bytes = checked_add(
            accounted_fixed_bytes,
            additional_capacity_bytes,
            "Barnes-Hut additional capacity");
        if (accounted_fixed_bytes > configuration.memory_budget_bytes) {
            throw std::length_error(
                "Barnes-Hut actual fixed vector capacity exceeds memory budget");
        }
        const std::size_t node_bytes =
            configuration.memory_budget_bytes - accounted_fixed_bytes;
        effective_maximum_nodes = std::min(
            configuration.maximum_nodes,
            node_bytes / sizeof(OctreeNode));
        if (count != 0 && effective_maximum_nodes == 0) {
            throw std::length_error("Barnes-Hut memory budget cannot hold the root node");
        }
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
        const std::size_t admitted_bytes = checked_add(
            accounted_fixed_bytes,
            actual_vector_bytes(
                nodes.capacity(), sizeof(OctreeNode), "Barnes-Hut node capacity"),
            "Barnes-Hut working memory");
        if (nodes.capacity() > effective_maximum_nodes ||
            admitted_bytes > configuration.memory_budget_bytes) {
            throw std::length_error(
                "Barnes-Hut actual node capacity exceeds memory budget");
        }
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
            accounted_fixed_bytes,
            checked_multiply(nodes.capacity(), sizeof(OctreeNode), "Barnes-Hut node memory"),
            "Barnes-Hut working memory");
        if (statistics.working_memory_bytes > configuration.memory_budget_bytes) {
            throw std::length_error("Barnes-Hut reserved storage exceeds memory budget");
        }
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

template <typename TraversalBudget>
void accumulate_node(
    const Tree& tree,
    std::size_t node_index,
    std::size_t target_index,
    double gravitational_constant_value,
    CompensatedVec3& acceleration,
    TraversalStatistics& statistics,
    TraversalBudget& budget) {
    reserve_traversal_work(budget, statistics);
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
            reserve_traversal_work(budget, statistics);
            const Vec3 displacement =
                tree.state.bodies[source_index].position_m -
                tree.state.bodies[target_index].position_m;
            acceleration.add(acceleration_contribution(
                displacement,
                tree.state.bodies[source_index].mass_kg,
                gravitational_constant_value));
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
                statistics,
                budget);
        }
    }
}

void merge_traversal_statistics(
    BarnesHutStatistics& destination,
    const TraversalStatistics& source) {
    destination.visited_nodes = checked_add(
        destination.visited_nodes, source.visited_nodes, "Barnes-Hut visited-node count");
    destination.approximated_nodes = checked_add(
        destination.approximated_nodes,
        source.approximated_nodes,
        "Barnes-Hut approximated-node count");
    destination.direct_body_interactions = checked_add(
        destination.direct_body_interactions,
        source.direct_body_interactions,
        "Barnes-Hut direct-interaction count");
}

#ifdef NBODY_HAS_OPENMP
bool parallel_traversal_requires_runtime_budget(
    std::size_t body_count,
    std::size_t node_count,
    std::size_t limit) {
    try {
        const std::size_t per_target = checked_add(
            node_count,
            body_count == 0 ? 0 : body_count - 1,
            "Barnes-Hut parallel traversal bound");
        const std::size_t maximum_work = checked_multiply(
            body_count, per_target, "Barnes-Hut parallel traversal bound");
        return maximum_work > limit;
    } catch (const std::length_error&) {
        return true;
    }
}
#endif

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

std::size_t barnes_hut_packed_memory_bytes(
    std::size_t body_count,
    std::size_t node_count) {
    const std::size_t body_index_bytes = checked_multiply(
        checked_multiply(body_count, 2, "Barnes-Hut packed body-index arrays"),
        sizeof(std::uint32_t),
        "Barnes-Hut packed body-index arrays");
    const std::size_t node_bytes = checked_multiply(
        node_count,
        sizeof(BarnesHutPackedNode),
        "Barnes-Hut packed nodes");
    return checked_add(body_index_bytes, node_bytes, "Barnes-Hut packed storage");
}

BarnesHutPackedTree build_barnes_hut_packed_tree(
    const SimulationState& state,
    const BarnesHutConfiguration& configuration,
    std::size_t packed_memory_budget_bytes) {
    validate_configuration(configuration);
    validate_state(state, gravitational_constant);
    if (packed_memory_budget_bytes == 0) {
        throw std::invalid_argument("Barnes-Hut packed memory budget must be positive");
    }
    if (state.bodies.size() >=
        static_cast<std::size_t>(barnes_hut_no_packed_index)) {
        throw std::length_error("Barnes-Hut body count exceeds packed index range");
    }

    BarnesHutPackedTree packed;
    packed.statistics.body_count = state.bodies.size();
    if (state.bodies.empty()) {
        return packed;
    }

    validate_fixed_storage_lower_bound(
        state.bodies.size(), 0, configuration.memory_budget_bytes);
    Tree tree(state, configuration, 0);
    tree.build();
    if (tree.nodes.size() >=
        static_cast<std::size_t>(barnes_hut_no_packed_index)) {
        throw std::length_error("Barnes-Hut node count exceeds packed index range");
    }

    const std::size_t lower_bound = barnes_hut_packed_memory_bytes(
        state.bodies.size(), tree.nodes.size());
    if (lower_bound > packed_memory_budget_bytes) {
        throw std::length_error("Barnes-Hut packed tree exceeds memory budget");
    }

    packed.nodes.resize(tree.nodes.size());
    packed.permutation.resize(tree.permutation.size());
    packed.inverse_rank.resize(tree.rank.size());
    packed.storage_bytes = checked_add(
        checked_add(
            actual_vector_bytes(
                packed.nodes.capacity(),
                sizeof(BarnesHutPackedNode),
                "Barnes-Hut packed node capacity"),
            actual_vector_bytes(
                packed.permutation.capacity(),
                sizeof(std::uint32_t),
                "Barnes-Hut packed permutation capacity"),
            "Barnes-Hut packed capacity"),
        actual_vector_bytes(
            packed.inverse_rank.capacity(),
            sizeof(std::uint32_t),
            "Barnes-Hut packed inverse-rank capacity"),
        "Barnes-Hut packed capacity");
    if (packed.storage_bytes > packed_memory_budget_bytes) {
        throw std::length_error(
            "Barnes-Hut actual packed vector capacity exceeds memory budget");
    }

    for (std::size_t index = 0; index < tree.nodes.size(); ++index) {
        const OctreeNode& source = tree.nodes[index];
        BarnesHutPackedNode& destination = packed.nodes[index];
        destination.center_x_m = source.center_m.x;
        destination.center_y_m = source.center_m.y;
        destination.center_z_m = source.center_m.z;
        destination.center_of_mass_x_m = source.center_of_mass_m.x;
        destination.center_of_mass_y_m = source.center_of_mass_m.y;
        destination.center_of_mass_z_m = source.center_of_mass_m.z;
        destination.side_length_m = source.side_length_m;
        destination.mass_kg = source.mass_kg;
        destination.begin = static_cast<std::uint32_t>(source.begin);
        destination.end = static_cast<std::uint32_t>(source.end);
        destination.leaf = source.leaf ? 1U : 0U;
        for (std::size_t child = 0; child < source.children.size(); ++child) {
            destination.children[child] = source.children[child] == no_node
                ? barnes_hut_no_packed_index
                : static_cast<std::uint32_t>(source.children[child]);
        }
    }
    for (std::size_t index = 0; index < tree.permutation.size(); ++index) {
        packed.permutation[index] =
            static_cast<std::uint32_t>(tree.permutation[index]);
        packed.inverse_rank[index] =
            static_cast<std::uint32_t>(tree.rank[index]);
    }

    packed.statistics = tree.statistics;
    packed.statistics.body_count = state.bodies.size();
    packed.statistics.worker_threads = 1;
    return packed;
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

    validate_fixed_storage_lower_bound(
        state.bodies.size(), 0, configuration_.memory_budget_bytes);
    std::vector<Vec3> accelerations(state.bodies.size());
    const std::size_t output_capacity_bytes = actual_vector_bytes(
        accelerations.capacity(), sizeof(Vec3), "Barnes-Hut output capacity");
    const auto build_start = std::chrono::steady_clock::now();
    Tree tree(state, configuration_, output_capacity_bytes);
    tree.build();
    const auto build_end = std::chrono::steady_clock::now();
    candidate_statistics = tree.statistics;
    candidate_statistics.body_count = state.bodies.size();
    candidate_statistics.worker_threads = 1;
    candidate_statistics.tree_build_seconds =
        std::chrono::duration<double>(build_end - build_start).count();

    const auto force_start = std::chrono::steady_clock::now();
    TraversalStatistics traversal_statistics;
    SequentialTraversalBudget budget(configuration_.maximum_traversal_work);
    for (std::size_t target = 0; target < state.bodies.size(); ++target) {
        CompensatedVec3 acceleration;
        accumulate_node(
            tree,
            0,
            target,
            gravitational_constant_value,
            acceleration,
            traversal_statistics,
            budget);
        if (!finite(acceleration.sum)) {
            throw std::domain_error("Barnes-Hut accumulated acceleration is not representable");
        }
        accelerations[target] = acceleration.sum;
    }
    merge_traversal_statistics(candidate_statistics, traversal_statistics);
    const auto force_end = std::chrono::steady_clock::now();
    candidate_statistics.force_evaluation_seconds =
        std::chrono::duration<double>(force_end - force_start).count();
    last_statistics_ = candidate_statistics;
    return accelerations;
}

OpenMpBarnesHutForceBackend::OpenMpBarnesHutForceBackend(
    int thread_count,
    BarnesHutConfiguration configuration)
    : thread_count_(thread_count), configuration_(configuration) {
    validate_configuration(configuration_);
    if (thread_count_ <= 0 || thread_count_ > maximum_openmp_barnes_hut_threads) {
        throw std::invalid_argument("OpenMP Barnes-Hut thread count must be in [1, 256]");
    }
#ifdef NBODY_HAS_OPENMP
    if (thread_count_ > omp_get_thread_limit()) {
        throw std::invalid_argument("OpenMP Barnes-Hut thread count exceeds runtime limit");
    }
#endif
}

bool openmp_barnes_hut_compiled() noexcept {
#ifdef NBODY_HAS_OPENMP
    return true;
#else
    return false;
#endif
}

std::vector<Vec3> OpenMpBarnesHutForceBackend::compute_accelerations(
    const SimulationState& state,
    double gravitational_constant_value) const {
#ifndef NBODY_HAS_OPENMP
    (void)state;
    (void)gravitational_constant_value;
    throw std::runtime_error(
        "OpenMP Barnes-Hut is unavailable in this build; configure with NBODY_ENABLE_OPENMP=ON");
#else
    validate_state(state, gravitational_constant_value);
    if (omp_in_parallel() != 0) {
        throw std::runtime_error(
            "OpenMP Barnes-Hut cannot be called from an active OpenMP parallel region");
    }

    BarnesHutStatistics candidate_statistics;
    candidate_statistics.body_count = state.bodies.size();
    if (state.bodies.empty()) {
        last_statistics_ = candidate_statistics;
        last_attempt_traversal_work_ = 0;
        return {};
    }
    if (state.bodies.size() > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())) {
        throw std::length_error("OpenMP Barnes-Hut body count exceeds loop index range");
    }

    const std::size_t requested_thread_storage_bytes = checked_multiply(
        static_cast<std::size_t>(thread_count_),
        sizeof(ThreadTraversalStatistics),
        "OpenMP Barnes-Hut thread statistics");
    validate_fixed_storage_lower_bound(
        state.bodies.size(),
        requested_thread_storage_bytes,
        configuration_.memory_budget_bytes);
    std::vector<Vec3> accelerations(state.bodies.size());
    std::vector<ThreadTraversalStatistics> thread_statistics(
        static_cast<std::size_t>(thread_count_));
    const std::size_t output_capacity_bytes = actual_vector_bytes(
        accelerations.capacity(), sizeof(Vec3), "Barnes-Hut output capacity");
    const std::size_t thread_storage_bytes = checked_multiply(
        thread_statistics.capacity(),
        sizeof(ThreadTraversalStatistics),
        "OpenMP Barnes-Hut thread statistics");
    const auto build_start = std::chrono::steady_clock::now();
    Tree tree(
        state, configuration_, output_capacity_bytes, thread_storage_bytes);
    tree.build();
    const auto build_end = std::chrono::steady_clock::now();
    candidate_statistics = tree.statistics;
    candidate_statistics.body_count = state.bodies.size();
    candidate_statistics.tree_build_seconds =
        std::chrono::duration<double>(build_end - build_start).count();

    ParallelTraversalBudget budget(
        configuration_.maximum_traversal_work,
        parallel_traversal_requires_runtime_budget(
            state.bodies.size(), tree.nodes.size(), configuration_.maximum_traversal_work));
    std::exception_ptr first_failure;
    std::size_t first_failure_target = std::numeric_limits<std::size_t>::max();
    int effective_threads = 0;
    const std::int64_t body_count = static_cast<std::int64_t>(state.bodies.size());

    const auto force_start = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(thread_count_)
    {
        const int thread_index = omp_get_thread_num();
#pragma omp single
        effective_threads = omp_get_num_threads();

        TraversalStatistics& local_statistics =
            thread_statistics[static_cast<std::size_t>(thread_index)].values;
#ifdef NBODY_BARNES_HUT_TEST_RUNTIME_SCHEDULE
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(static)
#endif
        for (std::int64_t raw_target = 0; raw_target < body_count; ++raw_target) {
            if (budget.exhausted()) {
                continue;
            }
            const std::size_t target = static_cast<std::size_t>(raw_target);
            try {
                CompensatedVec3 acceleration;
                accumulate_node(
                    tree,
                    0,
                    target,
                    gravitational_constant_value,
                    acceleration,
                    local_statistics,
                    budget);
                if (!finite(acceleration.sum)) {
                    throw std::domain_error(
                        "Barnes-Hut accumulated acceleration is not representable");
                }
                accelerations[target] = acceleration.sum;
            } catch (...) {
                if (budget.exhausted()) {
                    continue;
                }
                const std::exception_ptr candidate_failure = std::current_exception();
#pragma omp critical(nbody_barnes_hut_failure)
                {
                    if (target < first_failure_target) {
                        first_failure_target = target;
                        first_failure = candidate_failure;
                    }
                }
            }
        }
    }
    const auto force_end = std::chrono::steady_clock::now();

    std::size_t observed_work = budget.consumed();
    if (!budget.requires_runtime_tracking()) {
        observed_work = 0;
        for (int thread = 0; thread < effective_threads; ++thread) {
            const auto& values =
                thread_statistics[static_cast<std::size_t>(thread)].values;
            observed_work = checked_add(
                observed_work, values.charged_work, "OpenMP Barnes-Hut attempt work");
        }
    }
    last_attempt_traversal_work_ = observed_work;

    if (effective_threads != thread_count_) {
        throw std::runtime_error(
            "OpenMP runtime did not honor the configured Barnes-Hut thread count");
    }
    if (budget.exhausted()) {
        throw std::length_error("Barnes-Hut traversal work budget exceeded");
    }
    if (first_failure) {
        std::rethrow_exception(first_failure);
    }
    for (int thread = 0; thread < effective_threads; ++thread) {
        merge_traversal_statistics(
            candidate_statistics,
            thread_statistics[static_cast<std::size_t>(thread)].values);
    }
    candidate_statistics.worker_threads = effective_threads;
    candidate_statistics.force_evaluation_seconds =
        std::chrono::duration<double>(force_end - force_start).count();
    last_statistics_ = candidate_statistics;
    return accelerations;
#endif
}

#ifdef NBODY_BARNES_HUT_TEST_RUNTIME_SCHEDULE
// Deliberately absent from production builds. The schedule-variation test
// calls this symbol to prove that it linked the schedule(runtime) variant of
// this exact implementation rather than the production schedule(static)
// object or a duplicated traversal.
bool openmp_barnes_hut_runtime_schedule_test_hook_enabled() noexcept {
    return true;
}
#endif

}  // namespace nbody
