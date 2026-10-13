#include "nbody/barnes_hut.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool finite(const nbody::Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

std::string format_double(double value) {
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

double vector_difference(const nbody::Vec3& lhs, const nbody::Vec3& rhs) {
    return std::hypot(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

bool exactly_equal(
    const std::vector<nbody::Vec3>& lhs,
    const std::vector<nbody::Vec3>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index].x != rhs[index].x ||
            lhs[index].y != rhs[index].y ||
            lhs[index].z != rhs[index].z) {
            return false;
        }
    }
    return true;
}

struct ErrorSummary {
    double maximum_absolute{0.0};
    double rms_absolute{0.0};
    double maximum_relative{0.0};
    double percentile_95_relative{0.0};
};

ErrorSummary error_summary(
    const std::vector<nbody::Vec3>& reference,
    const std::vector<nbody::Vec3>& candidate) {
    require(reference.size() == candidate.size(), "error summary size mismatch");
    std::vector<double> relative_errors;
    relative_errors.reserve(reference.size());
    long double squared_sum = 0.0L;
    ErrorSummary result;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double absolute = vector_difference(reference[index], candidate[index]);
        const double scale = std::max(std::hypot(
            reference[index].x, reference[index].y, reference[index].z), 1.0e-30);
        const double relative = absolute / scale;
        require(std::isfinite(absolute) && std::isfinite(relative),
                "non-finite force error metric");
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        result.maximum_relative = std::max(result.maximum_relative, relative);
        squared_sum += static_cast<long double>(absolute) * absolute;
        relative_errors.push_back(relative);
    }
    if (!reference.empty()) {
        result.rms_absolute = std::sqrt(
            static_cast<double>(squared_sum / static_cast<long double>(reference.size())));
        std::sort(relative_errors.begin(), relative_errors.end());
        const std::size_t percentile_index =
            (95 * relative_errors.size() + 99) / 100 - 1;
        result.percentile_95_relative = relative_errors[percentile_index];
    }
    return result;
}

nbody::BarnesHutConfiguration test_configuration(double theta = 0.5) {
    nbody::BarnesHutConfiguration configuration;
    configuration.theta = theta;
    configuration.leaf_capacity = 4;
    configuration.maximum_depth = 64;
    configuration.minimum_cell_size_m = 1.0e-12;
    configuration.maximum_nodes = 100'000;
    configuration.memory_budget_bytes = 64U * 1024U * 1024U;
    return configuration;
}

nbody::SimulationState make_planar_state(std::size_t count) {
    nbody::SimulationState state;
    state.bodies.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const double angle = 6.283185307179586 * static_cast<double>(index) /
            static_cast<double>(count);
        const double radius = 2.0e8 + static_cast<double>(index % 7) * 1.0e7;
        state.bodies.push_back({
            "planar_" + std::to_string(index),
            1.0e20 + static_cast<double>(index) * 1.0e18,
            {radius * std::cos(angle), radius * std::sin(angle), 0.0},
            {},
        });
    }
    return state;
}

void test_empty_single_and_two_body() {
    nbody::SequentialBarnesHutForceBackend backend(test_configuration());
    const nbody::SimulationState empty;
    require(backend.compute_accelerations(empty).empty(), "empty state was not supported");
    require(backend.last_statistics().node_count == 0, "empty tree created a node");

    nbody::SimulationState single;
    single.bodies.push_back({"single", 2.0, {3.0, -4.0, 5.0}, {}});
    const auto single_acceleration = backend.compute_accelerations(single, 1.0);
    require(single_acceleration.size() == 1 &&
            single_acceleration[0].x == 0.0 &&
            single_acceleration[0].y == 0.0 &&
            single_acceleration[0].z == 0.0,
            "single body has nonzero self acceleration");

    const auto scenario = nbody::make_analytic_circular_two_body();
    const auto direct = nbody::compute_accelerations(scenario.initial_state);
    const auto approximate = backend.compute_accelerations(scenario.initial_state);
    const auto errors = error_summary(direct, approximate);
    require(errors.maximum_relative <= 5.0e-15, "two-body result is not direct-force equivalent");
    require(backend.last_statistics().direct_body_interactions == 2,
            "two-body leaf did not exclude self force correctly");
}

void test_direct_agreement_and_opening_limit() {
    const auto clustered = nbody::make_seeded_synthetic_system(192, 0x42484f4354ULL);
    const auto direct = nbody::compute_accelerations(clustered);

    nbody::SequentialBarnesHutForceBackend tight(test_configuration(0.25));
    nbody::SequentialBarnesHutForceBackend nominal(test_configuration(0.5));
    nbody::SequentialBarnesHutForceBackend loose(test_configuration(0.8));
    const auto tight_error = error_summary(direct, tight.compute_accelerations(clustered));
    const auto nominal_error = error_summary(direct, nominal.compute_accelerations(clustered));
    const auto loose_error = error_summary(direct, loose.compute_accelerations(clustered));
    require(loose.last_statistics().approximated_nodes > 0,
            "opening criterion never accepted an aggregate node");

    require(tight_error.maximum_relative < 0.015 &&
            tight_error.percentile_95_relative < 0.004,
            "theta=0.25 direct-force agreement exceeded guard");
    require(nominal_error.maximum_relative < 0.08 &&
            nominal_error.percentile_95_relative < 0.025,
            "theta=0.5 direct-force agreement exceeded guard");
    require(loose_error.maximum_relative < 0.30 &&
            loose_error.percentile_95_relative < 0.12,
            "theta=0.8 direct-force agreement exceeded guard");
    require(tight_error.rms_absolute <= nominal_error.rms_absolute * 1.1,
            "tighter theta unexpectedly worsened RMS force error materially");
    require(nominal_error.rms_absolute <= loose_error.rms_absolute * 1.1,
            "nominal theta unexpectedly worsened RMS force error materially");

    auto near_direct_configuration = test_configuration(1.0e-12);
    nbody::SequentialBarnesHutForceBackend near_direct(near_direct_configuration);
    const auto near_direct_error =
        error_summary(direct, near_direct.compute_accelerations(clustered));
    require(near_direct_error.maximum_relative < 2.0e-14,
            "theta approaching zero did not approach direct force");
    require(near_direct.last_statistics().direct_body_interactions ==
            clustered.bodies.size() * (clustered.bodies.size() - 1),
            "near-zero theta still accepted an aggregate node");

    nbody::SimulationState monopole;
    monopole.bodies = {
        {"source_a", 2.0, {0.0, 0.0, 0.0}, {}},
        {"source_b", 6.0, {4.0, 0.0, 0.0}, {}},
        {"target", 1.0, {1.0e9, 0.0, 0.0}, {}},
    };
    auto monopole_configuration = test_configuration(2.0);
    monopole_configuration.leaf_capacity = 1;
    nbody::SequentialBarnesHutForceBackend monopole_backend(monopole_configuration);
    const auto monopole_acceleration = monopole_backend.compute_accelerations(monopole, 1.0);
    const double target_distance = 1.0e9 - 3.0;
    const double expected_target_acceleration = -8.0 / (target_distance * target_distance);
    require(std::abs(
        (monopole_acceleration[2].x - expected_target_acceleration) /
        expected_target_acceleration) < 2.0e-15,
        "node mass or center-of-mass monopole was incorrect");
    require(monopole_backend.last_statistics().approximated_nodes > 0,
            "monopole center-of-mass fixture did not exercise aggregation");
}

void test_geometries_translation_scaling_and_order() {
    const auto planar = make_planar_state(64);
    const auto planar_direct = nbody::compute_accelerations(planar);
    nbody::SequentialBarnesHutForceBackend backend(test_configuration(0.4));
    const auto planar_bh = backend.compute_accelerations(planar);
    require(error_summary(planar_direct, planar_bh).maximum_relative < 0.04,
            "planar force error exceeded guard");

    nbody::SimulationState symmetric;
    symmetric.bodies.push_back({"center", 3.0, {0.0, 0.0, 0.0}, {}});
    for (int x : {-1, 1}) {
        for (int y : {-1, 1}) {
            for (int z : {-1, 1}) {
                symmetric.bodies.push_back({
                    "corner_" + std::to_string(x) + std::to_string(y) + std::to_string(z),
                    1.0,
                    {static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)},
                    {},
                });
            }
        }
    }
    const auto symmetric_result = backend.compute_accelerations(symmetric, 1.0);
    require(std::hypot(
        symmetric_result[0].x,
        symmetric_result[0].y,
        symmetric_result[0].z) < 1.0e-15,
        "symmetric distribution produced spurious center acceleration");

    nbody::SimulationState collinear;
    for (std::size_t index = 0; index < 96; ++index) {
        collinear.bodies.push_back({
            "line_" + std::to_string(index),
            1.0e18 + static_cast<double>(index) * 1.0e16,
            {static_cast<double>(index) * 1.0e6, 0.0, 0.0},
            {},
        });
    }
    const auto collinear_direct = nbody::compute_accelerations(collinear);
    const auto collinear_error = error_summary(
        collinear_direct, backend.compute_accelerations(collinear));
    double collinear_scale = 0.0;
    for (const auto& acceleration : collinear_direct) {
        collinear_scale = std::max(
            collinear_scale, std::hypot(acceleration.x, acceleration.y, acceleration.z));
    }
    require(collinear_error.percentile_95_relative < 0.08 &&
            collinear_error.maximum_absolute < collinear_scale * 0.06,
            "collinear force error exceeded robust guards: p95=" +
                std::to_string(collinear_error.percentile_95_relative) +
                " max_scaled=" +
                std::to_string(collinear_error.maximum_absolute / collinear_scale));

    auto translated = planar;
    for (auto& body : translated.bodies) {
        body.position_m += nbody::Vec3{1.0e12, -2.0e12, 3.0e12};
    }
    const auto translated_acceleration = backend.compute_accelerations(translated);
    require(error_summary(
        nbody::compute_accelerations(translated),
        translated_acceleration).maximum_relative < 0.04,
        "translated Barnes-Hut force error exceeded direct-force guard");

    auto scaled = planar;
    for (auto& body : scaled.bodies) {
        body.position_m *= 10.0;
        body.mass_kg *= 100.0;
    }
    const auto scaled_acceleration = backend.compute_accelerations(scaled);
    require(error_summary(
        nbody::compute_accelerations(scaled),
        scaled_acceleration).maximum_relative < 0.04,
        "scaled Barnes-Hut force error exceeded direct-force guard");

    const auto first = backend.compute_accelerations(planar);
    const auto first_stats = backend.last_statistics();
    const auto second = backend.compute_accelerations(planar);
    const auto second_stats = backend.last_statistics();
    require(exactly_equal(first, second), "repeated tree build was not bitwise deterministic");
    require(first_stats.node_count == second_stats.node_count &&
            first_stats.leaf_count == second_stats.leaf_count &&
            first_stats.maximum_depth_reached == second_stats.maximum_depth_reached &&
            first_stats.visited_nodes == second_stats.visited_nodes &&
            first_stats.approximated_nodes == second_stats.approximated_nodes &&
            first_stats.direct_body_interactions == second_stats.direct_body_interactions,
            "deterministic tree traversal counters changed");

    auto reversed = planar;
    std::reverse(reversed.bodies.begin(), reversed.bodies.end());
    const auto reversed_result = backend.compute_accelerations(reversed);
    std::map<std::string, nbody::Vec3> by_name;
    for (std::size_t index = 0; index < reversed.bodies.size(); ++index) {
        by_name.emplace(reversed.bodies[index].name, reversed_result[index]);
    }
    std::vector<nbody::Vec3> remapped;
    for (const auto& body : planar.bodies) {
        remapped.push_back(by_name.at(body.name));
    }
    require(error_summary(first, remapped).maximum_relative < 2.0e-12,
            "body input ordering materially changed Barnes-Hut result");
}

void test_extreme_scales_and_degenerate_paths() {
    auto configuration = test_configuration(0.5);
    configuration.leaf_capacity = 1;
    nbody::SequentialBarnesHutForceBackend backend(configuration);

    nbody::SimulationState large;
    large.bodies = {
        {"large_a", 1.0e200, {1.0e150, 0.0, 0.0}, {}},
        {"large_b", 2.0e200, {1.0e150 + 1.0e140, 0.0, 0.0}, {}},
    };
    const auto large_acceleration = backend.compute_accelerations(large, 1.0);
    require(finite(large_acceleration[0]) && finite(large_acceleration[1]) &&
            large_acceleration[0].x > 0.0 && large_acceleration[1].x < 0.0,
            "large representable geometry did not produce finite directed acceleration");

    nbody::SimulationState small;
    small.bodies = {
        {"small_a", 1.0e-250, {-1.0e-100, 0.0, 0.0}, {}},
        {"small_b", 2.0e-250, {1.0e-100, 0.0, 0.0}, {}},
    };
    const auto small_acceleration = backend.compute_accelerations(small, 1.0);
    require(finite(small_acceleration[0]) && finite(small_acceleration[1]) &&
            small_acceleration[0].x > 0.0 && small_acceleration[1].x < 0.0,
            "small representable geometry did not produce finite directed acceleration");

    nbody::SimulationState scaled_extreme;
    scaled_extreme.bodies = {
        {"unit_mass", 1.0, {0.0, 0.0, 0.0}, {}},
        {"tiny_mass", 1.0e-300, {1.0e-300, 0.0, 0.0}, {}},
    };
    const auto scaled_extreme_acceleration =
        backend.compute_accelerations(scaled_extreme, 1.0e-300);
    require(std::abs(scaled_extreme_acceleration[0].x - 1.0) < 2.0e-15 &&
            std::abs(scaled_extreme_acceleration[1].x + 1.0e300) < 2.0e285,
            "scaled binary64 force evaluation lost a representable extreme result");

    nbody::SimulationState nearly_coincident;
    nearly_coincident.bodies = {
        {"near_a", 1.0, {0.0, 0.0, 0.0}, {}},
        {"near_b", 1.0, {1.0e-9, 0.0, 0.0}, {}},
    };
    require(finite(backend.compute_accelerations(nearly_coincident, 1.0)[0]),
            "near-coincident finite geometry was rejected");

    nbody::SimulationState unbalanced;
    for (std::size_t index = 0; index < 180; ++index) {
        unbalanced.bodies.push_back({
            "deep_" + std::to_string(index),
            1.0,
            {std::ldexp(1.0, -static_cast<int>(index)), 0.0, 0.0},
            {},
        });
    }
    auto bounded_configuration = test_configuration(0.5);
    bounded_configuration.leaf_capacity = 1;
    bounded_configuration.maximum_depth = 32;
    bounded_configuration.minimum_cell_size_m = 0.0;
    nbody::SequentialBarnesHutForceBackend bounded(bounded_configuration);
    const auto bounded_result = bounded.compute_accelerations(unbalanced, 1.0e-200);
    require(std::all_of(bounded_result.begin(), bounded_result.end(), finite),
            "unbalanced bounded tree produced non-finite output");
    require(bounded.last_statistics().maximum_depth_reached <= 32,
            "tree exceeded configured maximum depth");

    const double maximum = std::numeric_limits<double>::max();
    nbody::SimulationState near_maximum;
    near_maximum.bodies = {
        {"near_max_a", 1.0, {maximum, maximum, maximum}, {}},
        {"near_max_b", 1.0, {
            std::nextafter(maximum, 0.0),
            std::nextafter(maximum, 0.0),
            std::nextafter(maximum, 0.0)}, {}},
    };
    const auto near_maximum_result =
        backend.compute_accelerations(near_maximum, 1.0e-300);
    require(std::all_of(near_maximum_result.begin(), near_maximum_result.end(), finite),
            "same-sign near-maximum bounds overflowed midpoint construction");
}

void test_rejections_and_resource_budgets() {
    bool rejected = false;
    try {
        auto bad = test_configuration();
        bad.theta = 0.0;
        (void)nbody::SequentialBarnesHutForceBackend(bad);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid theta was accepted");

    nbody::SimulationState duplicate;
    duplicate.bodies = {
        {"a", 1.0, {1.0, 2.0, 3.0}, {}},
        {"b", 2.0, {1.0, 2.0, 3.0}, {}},
    };
    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend(test_configuration());
        (void)backend.compute_accelerations(duplicate, 1.0);
    } catch (const std::domain_error&) {
        rejected = true;
    }
    require(rejected, "coincident bodies were not rejected");

    nbody::SimulationState nonrepresentable;
    nonrepresentable.bodies = {
        {"negative", 1.0, {-std::numeric_limits<double>::max(), 0.0, 0.0}, {}},
        {"positive", 1.0, {std::numeric_limits<double>::max(), 0.0, 0.0}, {}},
    };
    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend(test_configuration());
        (void)backend.compute_accelerations(nonrepresentable, 1.0);
    } catch (const std::domain_error&) {
        rejected = true;
    }
    require(rejected, "nonrepresentable coordinate span was not rejected");

    nbody::SimulationState nonrepresentable_acceleration;
    nonrepresentable_acceleration.bodies = {
        {"target", 1.0, {0.0, 0.0, 0.0}, {}},
        {"source", std::numeric_limits<double>::max(), {1.0, 0.0, 0.0}, {}},
    };
    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend(test_configuration());
        (void)backend.compute_accelerations(
            nonrepresentable_acceleration,
            std::numeric_limits<double>::max());
    } catch (const std::domain_error&) {
        rejected = true;
    }
    require(rejected, "nonrepresentable acceleration was not rejected");

    auto invalid = duplicate;
    invalid.bodies.resize(1);
    invalid.bodies[0].position_m.x = std::numeric_limits<double>::infinity();
    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend(test_configuration());
        (void)backend.compute_accelerations(invalid, 1.0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "non-finite singleton position was accepted");

    const auto state = nbody::make_seeded_synthetic_system(64, 99);
    auto node_limited = test_configuration();
    node_limited.leaf_capacity = 1;
    node_limited.maximum_nodes = 1;
    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend(node_limited);
        (void)backend.compute_accelerations(state);
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "node budget exhaustion was not rejected");

    auto traversal_limited = test_configuration();
    traversal_limited.maximum_traversal_work = 1;
    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend(traversal_limited);
        (void)backend.compute_accelerations(state);
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "traversal work budget exhaustion was not rejected");

    auto cell_limited = test_configuration();
    cell_limited.leaf_capacity = 1;
    cell_limited.minimum_cell_size_m = 1.0e20;
    nbody::SequentialBarnesHutForceBackend cell_limited_backend(cell_limited);
    (void)cell_limited_backend.compute_accelerations(state);
    require(cell_limited_backend.last_statistics().node_count == 1 &&
            cell_limited_backend.last_statistics().direct_body_interactions ==
                state.bodies.size() * (state.bodies.size() - 1),
            "minimum-cell termination did not create one direct leaf");

    auto memory_limited = test_configuration();
    memory_limited.memory_budget_bytes = 1;
    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend(memory_limited);
        (void)backend.compute_accelerations(state);
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "memory budget exhaustion was not rejected");

    rejected = false;
    try {
        (void)nbody::barnes_hut_fixed_memory_bytes(
            std::numeric_limits<std::size_t>::max());
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "checked memory arithmetic did not reject overflow");

    invalid = state;
    invalid.bodies[0].mass_kg = 0.0;
    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend(test_configuration());
        (void)backend.compute_accelerations(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "nonpositive mass was accepted");

    rejected = false;
    try {
        nbody::SequentialBarnesHutForceBackend backend(test_configuration());
        (void)backend.compute_accelerations(state, -1.0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "negative gravitational constant was accepted");
}

void test_cpu_only_openmp_rejection() {
    if (nbody::openmp_barnes_hut_compiled()) {
        return;
    }
    nbody::OpenMpBarnesHutForceBackend backend(1, test_configuration());
    bool rejected = false;
    try {
        (void)backend.compute_accelerations(nbody::make_seeded_synthetic_system(2, 1));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "CPU-only Barnes-Hut build silently accepted OpenMP execution");
}

double maximum_position_difference(
    const nbody::SimulationState& first,
    const nbody::SimulationState& second) {
    require(first.bodies.size() == second.bodies.size(), "trajectory size mismatch");
    double maximum = 0.0;
    for (std::size_t index = 0; index < first.bodies.size(); ++index) {
        maximum = std::max(maximum, vector_difference(
            first.bodies[index].position_m, second.bodies[index].position_m));
    }
    return maximum;
}

struct TrajectoryCharacterization {
    double tight_separation_m{0.0};
    double loose_separation_m{0.0};
    double tight_energy_drift{0.0};
    double loose_energy_drift{0.0};
    double tight_angular_drift{0.0};
    double loose_angular_drift{0.0};
};

TrajectoryCharacterization characterize_trajectory(double dt_s, std::size_t steps) {
    const auto initial = nbody::make_seeded_synthetic_system(48, 0x5155414c495459ULL);
    auto direct_state = initial;
    auto tight_state = initial;
    auto loose_state = initial;
    nbody::SequentialBarnesHutForceBackend tight(test_configuration(0.25));
    nbody::SequentialBarnesHutForceBackend loose(test_configuration(0.8));
    const auto tight_initial = nbody::compute_diagnostics(tight_state);
    const auto loose_initial = nbody::compute_diagnostics(loose_state);

    for (std::size_t step = 0; step < steps; ++step) {
        nbody::velocity_verlet_step(direct_state, dt_s);
        nbody::velocity_verlet_step(tight_state, dt_s, tight);
        nbody::velocity_verlet_step(loose_state, dt_s, loose);
    }

    require(direct_state.time_s == tight_state.time_s && tight_state.time_s == loose_state.time_s,
            "solver choice changed timestep semantics");
    TrajectoryCharacterization result;
    result.tight_separation_m = maximum_position_difference(direct_state, tight_state);
    result.loose_separation_m = maximum_position_difference(direct_state, loose_state);
    require(result.tight_separation_m < 1.0e-3 && result.loose_separation_m < 0.5,
            "short trajectory separation exceeded characterization guard: tight_m=" +
                format_double(result.tight_separation_m) + " loose_m=" +
                format_double(result.loose_separation_m));
    require(result.tight_separation_m <= result.loose_separation_m * 1.2,
            "tighter theta materially worsened trajectory agreement");

    const auto tight_final = nbody::compute_diagnostics(tight_state);
    const auto loose_final = nbody::compute_diagnostics(loose_state);
    result.tight_energy_drift = std::abs(
        (tight_final.total_energy_j() - tight_initial.total_energy_j()) /
        tight_initial.total_energy_j());
    result.loose_energy_drift = std::abs(
        (loose_final.total_energy_j() - loose_initial.total_energy_j()) /
        loose_initial.total_energy_j());
    result.tight_angular_drift = vector_difference(
        tight_final.angular_momentum_kg_m2_s,
        tight_initial.angular_momentum_kg_m2_s) /
        std::max(nbody::norm(tight_initial.angular_momentum_kg_m2_s), 1.0);
    result.loose_angular_drift = vector_difference(
        loose_final.angular_momentum_kg_m2_s,
        loose_initial.angular_momentum_kg_m2_s) /
        std::max(nbody::norm(loose_initial.angular_momentum_kg_m2_s), 1.0);
    require(result.tight_energy_drift < 2.0e-10 && result.loose_energy_drift < 1.0e-7,
            "Barnes-Hut short-run energy drift exceeded guard: tight=" +
                format_double(result.tight_energy_drift) + " loose=" +
                format_double(result.loose_energy_drift));
    require(result.tight_angular_drift < 2.0e-9 && result.loose_angular_drift < 5.0e-6,
            "Barnes-Hut short-run angular drift exceeded guard: tight=" +
                format_double(result.tight_angular_drift) + " loose=" +
                format_double(result.loose_angular_drift));
    return result;
}

void test_trajectory_and_conservation_characterization() {
    const auto coarse = characterize_trajectory(25.0, 120);
    const auto fine = characterize_trajectory(12.5, 240);
    require(std::isfinite(coarse.tight_energy_drift) &&
            std::isfinite(fine.tight_energy_drift),
            "multiple-timestep characterization was not finite");
    std::cout << std::setprecision(17)
              << "trajectory_characterization"
              << " coarse_tight_position_m=" << coarse.tight_separation_m
              << " coarse_loose_position_m=" << coarse.loose_separation_m
              << " coarse_tight_energy_drift=" << coarse.tight_energy_drift
              << " coarse_loose_energy_drift=" << coarse.loose_energy_drift
              << " coarse_tight_angular_drift=" << coarse.tight_angular_drift
              << " coarse_loose_angular_drift=" << coarse.loose_angular_drift
              << " fine_tight_position_m=" << fine.tight_separation_m
              << " fine_loose_position_m=" << fine.loose_separation_m
              << " fine_tight_energy_drift=" << fine.tight_energy_drift
              << " fine_loose_energy_drift=" << fine.loose_energy_drift
              << " fine_tight_angular_drift=" << fine.tight_angular_drift
              << " fine_loose_angular_drift=" << fine.loose_angular_drift
              << '\n';
}

}  // namespace

int main() {
    try {
        test_empty_single_and_two_body();
        test_direct_agreement_and_opening_limit();
        test_geometries_translation_scaling_and_order();
        test_extreme_scales_and_degenerate_paths();
        test_rejections_and_resource_budgets();
        test_cpu_only_openmp_rejection();
        test_trajectory_and_conservation_characterization();
        std::cout << "all Barnes-Hut tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Barnes-Hut test failure: " << error.what() << '\n';
        return 1;
    }
}
