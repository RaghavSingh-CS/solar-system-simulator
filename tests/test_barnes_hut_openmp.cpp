#include "nbody/barnes_hut.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/openmp_backend.hpp"
#include "nbody/scenarios.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#ifdef NBODY_BARNES_HUT_EXPECT_RUNTIME_SCHEDULE
namespace nbody {
bool openmp_barnes_hut_runtime_schedule_test_hook_enabled() noexcept;
}
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint64_t double_representation(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "tests require binary64-sized double storage");
    static_assert(std::numeric_limits<double>::is_iec559,
                  "tests require IEC 559 / IEEE-754 double semantics");
    std::uint64_t representation = 0;
    std::memcpy(&representation, &value, sizeof(representation));
    return representation;
}

bool bitwise_equal(double lhs, double rhs) {
    return double_representation(lhs) == double_representation(rhs);
}

bool bitwise_equal(const nbody::Vec3& lhs, const nbody::Vec3& rhs) {
    return bitwise_equal(lhs.x, rhs.x) && bitwise_equal(lhs.y, rhs.y) &&
        bitwise_equal(lhs.z, rhs.z);
}

bool bitwise_equal(
    const std::vector<nbody::Vec3>& lhs,
    const std::vector<nbody::Vec3>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (!bitwise_equal(lhs[index], rhs[index])) {
            return false;
        }
    }
    return true;
}

bool bitwise_equal(
    const nbody::SimulationState& lhs,
    const nbody::SimulationState& rhs) {
    if (!bitwise_equal(lhs.time_s, rhs.time_s) ||
        lhs.bodies.size() != rhs.bodies.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.bodies.size(); ++index) {
        const auto& a = lhs.bodies[index];
        const auto& b = rhs.bodies[index];
        if (a.name != b.name || !bitwise_equal(a.mass_kg, b.mass_kg) ||
            !bitwise_equal(a.position_m, b.position_m) ||
            !bitwise_equal(a.velocity_m_s, b.velocity_m_s)) {
            return false;
        }
    }
    return true;
}

bool bitwise_equal(
    const nbody::BarnesHutStatistics& lhs,
    const nbody::BarnesHutStatistics& rhs) {
    return lhs.body_count == rhs.body_count &&
        lhs.node_count == rhs.node_count &&
        lhs.leaf_count == rhs.leaf_count &&
        lhs.maximum_depth_reached == rhs.maximum_depth_reached &&
        lhs.visited_nodes == rhs.visited_nodes &&
        lhs.approximated_nodes == rhs.approximated_nodes &&
        lhs.direct_body_interactions == rhs.direct_body_interactions &&
        lhs.working_memory_bytes == rhs.working_memory_bytes &&
        lhs.worker_threads == rhs.worker_threads &&
        bitwise_equal(lhs.tree_build_seconds, rhs.tree_build_seconds) &&
        bitwise_equal(lhs.force_evaluation_seconds, rhs.force_evaluation_seconds);
}

nbody::BarnesHutConfiguration configuration(double theta = 0.5) {
    nbody::BarnesHutConfiguration result;
    result.theta = theta;
    result.leaf_capacity = 4;
    result.maximum_depth = 64;
    result.minimum_cell_size_m = 1.0e-12;
    result.maximum_nodes = 100'000;
    result.maximum_traversal_work = 100'000'000;
    result.memory_budget_bytes = 64U * 1024U * 1024U;
    return result;
}

std::vector<int> supported_thread_counts() {
    std::vector<int> result;
    for (const int requested : std::array<int, 4>{1, 2, 4, 8}) {
        if (requested <= nbody::openmp_runtime_max_threads() &&
            requested <= nbody::openmp_processor_count() &&
            nbody::openmp_effective_thread_count(requested) == requested) {
            result.push_back(requested);
        } else {
            std::cout << "skipping unsupported OpenMP thread count " << requested << '\n';
        }
    }
    require(!result.empty() && result.front() == 1,
            "OpenMP runtime could not provide one worker thread");
    return result;
}

std::vector<omp_sched_t> exercised_schedules() {
#ifdef NBODY_BARNES_HUT_EXPECT_RUNTIME_SCHEDULE
    require(nbody::openmp_barnes_hut_runtime_schedule_test_hook_enabled(),
            "schedule test did not link the schedule(runtime) implementation variant");
    return {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
#else
    return {omp_sched_static};
#endif
}

const char* schedule_name(omp_sched_t schedule) {
    switch (schedule) {
        case omp_sched_static:
            return "static";
        case omp_sched_dynamic:
            return "dynamic";
        case omp_sched_guided:
            return "guided";
        default:
            return "unknown";
    }
}

void select_and_verify_schedule(omp_sched_t schedule) {
    omp_set_schedule(schedule, 1);
    omp_sched_t actual{};
    int chunk = 0;
    omp_get_schedule(&actual, &chunk);
    require(actual == schedule,
            std::string("OpenMP runtime did not activate requested ") +
                schedule_name(schedule) + " schedule");
    require(chunk == 1,
            std::string("OpenMP runtime changed requested chunk for ") +
                schedule_name(schedule) + " schedule");
}

nbody::SimulationState make_clustered(std::size_t count) {
    auto state = nbody::make_seeded_synthetic_system(count, 0x4f4d504248ULL);
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        state.bodies[index].position_m *= 0.04;
        state.bodies[index].position_m.x += index % 2 == 0 ? -3.0e8 : 3.0e8;
    }
    return state;
}

nbody::SimulationState make_planar(std::size_t count) {
    auto state = nbody::make_seeded_synthetic_system(count, 0x504c414e4152ULL);
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        const double angle = 6.283185307179586 * static_cast<double>(index) /
            static_cast<double>(count);
        const double radius = 2.0e8 + static_cast<double>(index % 17) * 3.0e6;
        state.bodies[index].position_m = {
            radius * std::cos(angle), radius * std::sin(angle), 0.0};
    }
    return state;
}

void compare_backends(
    const nbody::SimulationState& state,
    double theta,
    const std::vector<int>& thread_counts) {
    const auto policy = configuration(theta);
    nbody::SequentialBarnesHutForceBackend sequential(policy);
    const auto reference = sequential.compute_accelerations(state);
    const auto reference_statistics = sequential.last_statistics();

    for (const int threads : thread_counts) {
        nbody::OpenMpBarnesHutForceBackend parallel(threads, policy);
        const auto candidate = parallel.compute_accelerations(state);
        const auto first_statistics = parallel.last_statistics();
        const auto repeated = parallel.compute_accelerations(state);
        const auto repeated_statistics = parallel.last_statistics();

        require(bitwise_equal(reference, candidate),
                "OpenMP force result differed from Sequential Barnes-Hut");
        require(bitwise_equal(candidate, repeated),
                "OpenMP force result was not bitwise repeatable");
        require(reference_statistics.node_count == first_statistics.node_count &&
                reference_statistics.leaf_count == first_statistics.leaf_count &&
                reference_statistics.maximum_depth_reached ==
                    first_statistics.maximum_depth_reached &&
                reference_statistics.visited_nodes == first_statistics.visited_nodes &&
                reference_statistics.approximated_nodes ==
                    first_statistics.approximated_nodes &&
                reference_statistics.direct_body_interactions ==
                    first_statistics.direct_body_interactions,
                "OpenMP structural/work statistics differed from Sequential Barnes-Hut");
        require(first_statistics.worker_threads == threads,
                "OpenMP runtime did not honor the configured Barnes-Hut team size");
        require(parallel.last_attempt_traversal_work() ==
                    first_statistics.visited_nodes +
                        first_statistics.direct_body_interactions,
                "successful traversal attempt work did not equal completed reservations");
        require(first_statistics.node_count == repeated_statistics.node_count &&
                first_statistics.visited_nodes == repeated_statistics.visited_nodes &&
                first_statistics.approximated_nodes ==
                    repeated_statistics.approximated_nodes &&
                first_statistics.direct_body_interactions ==
                    repeated_statistics.direct_body_interactions,
                "OpenMP repeated traversal statistics changed");
    }
}

void test_representation_comparator_negative_control() {
    const double positive_zero = 0.0;
    const double negative_zero = -0.0;
    require(positive_zero == negative_zero,
            "signed-zero control did not satisfy ordinary floating-point equality");
    require(!bitwise_equal(positive_zero, negative_zero),
            "representation comparator failed to distinguish signed zero");
    require(!bitwise_equal(
                nbody::Vec3{positive_zero, 1.0, 2.0},
                nbody::Vec3{negative_zero, 1.0, 2.0}),
            "Vec3 representation comparator failed signed-zero negative control");
}

void test_force_equivalence(const std::vector<int>& thread_counts) {
    const std::array<nbody::SimulationState, 3> states{
        nbody::make_seeded_synthetic_system(192, 0x554e49464f524dULL),
        make_clustered(192),
        make_planar(192),
    };
    for (const auto& state : states) {
        for (const double theta : std::array<double, 3>{0.25, 0.5, 0.8}) {
            compare_backends(state, theta, thread_counts);
        }
    }
}

void test_boundaries_mass_extremes_and_self_force(const std::vector<int>& thread_counts) {
    nbody::SimulationState boundary;
    boundary.bodies = {
        {"center", 1.0e25, {0.0, 0.0, 0.0}, {}},
        {"x_positive", 1.0e5, {1.0, 0.0, 0.0}, {}},
        {"x_negative", 1.0e20, {-1.0, 0.0, 0.0}, {}},
        {"y_positive", 1.0e10, {0.0, 1.0, 0.0}, {}},
        {"y_negative", 1.0e30, {0.0, -1.0, 0.0}, {}},
        {"z_positive", 1.0e15, {0.0, 0.0, 1.0}, {}},
        {"z_negative", 1.0e28, {0.0, 0.0, -1.0}, {}},
        {"corner", 1.0, {1.0, 1.0, 1.0}, {}},
    };
    compare_backends(boundary, 0.5, thread_counts);

    nbody::SimulationState single;
    single.bodies.push_back({"single", 1.0, {4.0, -2.0, 9.0}, {}});
    for (const int threads : thread_counts) {
        nbody::OpenMpBarnesHutForceBackend backend(threads, configuration());
        require(backend.compute_accelerations({}).empty(), "empty state was not supported");
        const auto acceleration = backend.compute_accelerations(single, 1.0);
        require(acceleration.size() == 1 && bitwise_equal(acceleration[0], {}),
                "single body acquired self acceleration");
        require(backend.last_statistics().direct_body_interactions == 0,
                "single body recorded a self interaction");
    }
}

void test_extreme_and_degenerate_inputs(const std::vector<int>& thread_counts) {
    nbody::SimulationState extreme;
    extreme.bodies = {
        {"large_a", 1.0e200, {1.0e150, 0.0, 0.0}, {}},
        {"large_b", 2.0e200, {1.0e150 + 1.0e140, 0.0, 0.0}, {}},
        {"small", 1.0e-200, {1.0e150 - 1.0e140, 1.0e140, 0.0}, {}},
    };
    compare_backends(extreme, 0.5, thread_counts);

    nbody::SimulationState near;
    near.bodies = {
        {"near_a", 1.0, {0.0, 0.0, 0.0}, {}},
        {"near_b", 2.0, {1.0e-9, 0.0, 0.0}, {}},
    };
    compare_backends(near, 0.5, thread_counts);

    nbody::SimulationState coincident = near;
    coincident.bodies[1].position_m = coincident.bodies[0].position_m;
    nbody::SimulationState nonrepresentable;
    nonrepresentable.bodies = {
        {"target", 1.0, {0.0, 0.0, 0.0}, {}},
        {"source", 1.0e308, {1.0e-200, 0.0, 0.0}, {}},
    };
    const auto successful = nbody::make_seeded_synthetic_system(12, 0x415454454d5054ULL);
    for (const int threads : thread_counts) {
        for (int repetition = 0; repetition < 3; ++repetition) {
            nbody::OpenMpBarnesHutForceBackend backend(threads, configuration());
            (void)backend.compute_accelerations(successful);
            const auto committed = backend.last_statistics();
            const std::size_t successful_work = backend.last_attempt_traversal_work();
            require(successful_work == committed.visited_nodes +
                        committed.direct_body_interactions,
                    "successful call underreported charged traversal work");

            bool rejected = false;
            try {
                (void)backend.compute_accelerations(coincident, 1.0);
            } catch (const std::domain_error&) {
                rejected = true;
            }
            require(rejected, "OpenMP Barnes-Hut accepted coincident bodies");
            require(backend.last_attempt_traversal_work() == 4,
                    "coincident-body failure did not report all reserved work");
            require(bitwise_equal(committed, backend.last_statistics()),
                    "coincident-body failure mutated committed statistics");

            rejected = false;
            try {
                (void)backend.compute_accelerations(nonrepresentable, 1.0);
            } catch (const std::domain_error&) {
                rejected = true;
            }
            require(rejected, "OpenMP Barnes-Hut accepted nonrepresentable force");
            require(backend.last_attempt_traversal_work() == 4,
                    "nonrepresentable-force failure did not report all reserved work");
            require(bitwise_equal(committed, backend.last_statistics()),
                    "nonrepresentable-force failure mutated committed statistics");

            require(backend.compute_accelerations({}).empty(),
                    "empty call after nonempty/failure sequence was not supported");
            require(backend.last_attempt_traversal_work() == 0,
                    "empty successful call did not reset attempt work");
            require(backend.last_statistics().body_count == 0,
                    "empty successful call did not publish empty statistics");
        }
    }
}

void test_limits_and_transactional_statistics(const std::vector<int>& thread_counts) {
    const auto small = nbody::make_seeded_synthetic_system(2, 7);
    const auto larger = nbody::make_seeded_synthetic_system(96, 8);

    auto limited_policy = configuration();
    limited_policy.maximum_traversal_work = 4;
    omp_sched_t original_schedule{};
    int original_chunk = 0;
    omp_get_schedule(&original_schedule, &original_chunk);
    for (const omp_sched_t schedule : exercised_schedules()) {
        select_and_verify_schedule(schedule);
        compare_backends(make_clustered(96), 0.5, thread_counts);
        for (const int threads : thread_counts) {
            nbody::OpenMpBarnesHutForceBackend limited(threads, limited_policy);
            (void)limited.compute_accelerations(small);
            const auto committed = limited.last_statistics();
            require(limited.last_attempt_traversal_work() == 4,
                    "successful strict-budget traversal reported the wrong work count");
            bool rejected = false;
            std::string parallel_failure;
            try {
                (void)limited.compute_accelerations(larger);
            } catch (const std::length_error& error) {
                rejected = true;
                parallel_failure = error.what();
            }
            require(rejected, "OpenMP traversal budget was not enforced");
            require(parallel_failure == "Barnes-Hut traversal work budget exceeded",
                    "OpenMP traversal failure was not normalized deterministically");
            require(limited.last_attempt_traversal_work() ==
                        limited_policy.maximum_traversal_work,
                    "parallel traversal consumed beyond or below the hard work cap");
            require(bitwise_equal(committed, limited.last_statistics()),
                    "failed OpenMP traversal mutated committed statistics");
        }
    }
    omp_set_schedule(original_schedule, original_chunk);

    bool rejected = false;
    auto node_policy = configuration();
    node_policy.leaf_capacity = 1;
    node_policy.maximum_nodes = 1;
    rejected = false;
    try {
        nbody::OpenMpBarnesHutForceBackend backend(thread_counts.back(), node_policy);
        (void)backend.compute_accelerations(larger);
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "OpenMP node budget was not enforced");

    auto memory_policy = configuration();
    memory_policy.memory_budget_bytes = 1;
    rejected = false;
    try {
        nbody::OpenMpBarnesHutForceBackend backend(thread_counts.back(), memory_policy);
        (void)backend.compute_accelerations(larger);
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "OpenMP memory budget was not enforced");

    auto depth_policy = configuration(1.0e-12);
    depth_policy.leaf_capacity = 1;
    depth_policy.maximum_depth = 1;
    nbody::SequentialBarnesHutForceBackend sequential(depth_policy);
    nbody::OpenMpBarnesHutForceBackend parallel(thread_counts.back(), depth_policy);
    require(bitwise_equal(
        sequential.compute_accelerations(larger),
        parallel.compute_accelerations(larger)),
        "maximum-depth termination changed OpenMP forces");
    require(parallel.last_statistics().maximum_depth_reached <= 1,
            "OpenMP tree exceeded maximum depth");

    rejected = false;
    try {
        (void)nbody::OpenMpBarnesHutForceBackend(0, configuration());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "zero OpenMP Barnes-Hut thread count was accepted");
    rejected = false;
    try {
        (void)nbody::OpenMpBarnesHutForceBackend(257, configuration());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "excessive OpenMP Barnes-Hut thread count was accepted");
}

void test_actual_capacity_memory_admission(const std::vector<int>& thread_counts) {
    const int threads = thread_counts.back();
    nbody::SimulationState single;
    single.bodies.push_back({"single", 1.0, {0.0, 0.0, 0.0}, {}});

    auto probe_policy = configuration();
    nbody::OpenMpBarnesHutForceBackend probe(threads, probe_policy);
    (void)probe.compute_accelerations(single, 1.0);
    std::size_t low = 1;
    std::size_t high = probe.last_statistics().working_memory_bytes;
    while (low < high) {
        const std::size_t middle = low + (high - low) / 2;
        auto candidate_policy = probe_policy;
        candidate_policy.memory_budget_bytes = middle;
        bool admitted = true;
        try {
            nbody::OpenMpBarnesHutForceBackend candidate(threads, candidate_policy);
            (void)candidate.compute_accelerations(single, 1.0);
        } catch (const std::length_error&) {
            admitted = false;
        }
        if (admitted) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    const std::size_t minimum_admitted_budget = low;
    auto boundary_policy = probe_policy;
    boundary_policy.memory_budget_bytes = minimum_admitted_budget;
    nbody::OpenMpBarnesHutForceBackend boundary(threads, boundary_policy);
    (void)boundary.compute_accelerations(single, 1.0);
    const auto committed = boundary.last_statistics();
    require(committed.working_memory_bytes <= minimum_admitted_budget,
            "reported actual capacities exceeded the admitted memory budget");
    require(committed.working_memory_bytes >=
                nbody::barnes_hut_fixed_memory_bytes(single.bodies.size()),
            "actual-capacity accounting fell below the documented lower bound");

    auto below_policy = boundary_policy;
    --below_policy.memory_budget_bytes;
    bool rejected = false;
    try {
        nbody::OpenMpBarnesHutForceBackend below(threads, below_policy);
        (void)below.compute_accelerations(single, 1.0);
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "memory budget immediately below admission boundary was accepted");

    rejected = false;
    try {
        (void)boundary.compute_accelerations(
            nbody::make_seeded_synthetic_system(32, 0x4d454d4f5259ULL));
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "larger state bypassed the admitted actual-capacity boundary");
    require(bitwise_equal(committed, boundary.last_statistics()),
            "memory admission failure mutated committed statistics");
}

void test_nested_openmp_rejection(const std::vector<int>& thread_counts) {
    require(thread_counts.back() >= 2,
            "nested OpenMP rejection test requires a two-thread outer team");
    const int original_dynamic = omp_get_dynamic();
    const int original_max_active_levels = omp_get_max_active_levels();
    omp_set_dynamic(0);
    for (const int active_levels : std::array<int, 2>{1, 2}) {
        omp_set_max_active_levels(active_levels);
        nbody::OpenMpBarnesHutForceBackend backend(2, configuration());
        const auto state = nbody::make_seeded_synthetic_system(32, 0x4e4553544544ULL);
        (void)backend.compute_accelerations(state);
        const auto committed = backend.last_statistics();
        const auto committed_attempt_work = backend.last_attempt_traversal_work();
        bool rejected = false;
        std::string message;
#pragma omp parallel num_threads(2) shared(rejected, message)
        {
#pragma omp single
            {
                try {
                    (void)backend.compute_accelerations(state);
                } catch (const std::runtime_error& error) {
                    rejected = true;
                    message = error.what();
                }
            }
        }
        require(rejected, "nested OpenMP Barnes-Hut call was not rejected");
        require(message ==
                    "OpenMP Barnes-Hut cannot be called from an active OpenMP parallel region",
                "nested OpenMP Barnes-Hut rejection used the wrong diagnostic");
        require(bitwise_equal(committed, backend.last_statistics()),
                "nested-call rejection mutated committed statistics");
        require(committed_attempt_work == backend.last_attempt_traversal_work(),
                "nested-call rejection mutated attempt diagnostics");
    }
    omp_set_max_active_levels(original_max_active_levels);
    omp_set_dynamic(original_dynamic);
}

void test_velocity_verlet_and_conservation(const std::vector<int>& thread_counts) {
    const auto initial = nbody::make_seeded_synthetic_system(96, 0x5452414a454354ULL);
    auto sequential_state = initial;
    const auto policy = configuration(0.5);
    nbody::SequentialBarnesHutForceBackend sequential(policy);
    nbody::OpenMpBarnesHutForceBackend parallel(thread_counts.back(), policy);
    auto parallel_state = initial;
    for (std::size_t step = 0; step < 40; ++step) {
        nbody::velocity_verlet_step(sequential_state, 5.0, sequential);
        nbody::velocity_verlet_step(parallel_state, 5.0, parallel);
    }
    require(bitwise_equal(sequential_state, parallel_state),
            "OpenMP Barnes-Hut changed Velocity Verlet trajectory semantics");
    const auto sequential_diagnostics = nbody::compute_diagnostics(sequential_state);
    const auto parallel_diagnostics = nbody::compute_diagnostics(parallel_state);
    require(bitwise_equal(
                sequential_diagnostics.kinetic_energy_j,
                parallel_diagnostics.kinetic_energy_j) &&
            bitwise_equal(
                sequential_diagnostics.potential_energy_j,
                parallel_diagnostics.potential_energy_j) &&
            bitwise_equal(
                sequential_diagnostics.total_energy_j(),
                parallel_diagnostics.total_energy_j()) &&
            bitwise_equal(
                sequential_diagnostics.linear_momentum_kg_m_s,
                parallel_diagnostics.linear_momentum_kg_m_s) &&
            bitwise_equal(
                sequential_diagnostics.angular_momentum_kg_m2_s,
                parallel_diagnostics.angular_momentum_kg_m2_s) &&
            bitwise_equal(
                sequential_diagnostics.center_of_mass_m,
                parallel_diagnostics.center_of_mass_m),
            "OpenMP Barnes-Hut conservation diagnostics differed from Sequential reference");
}

}  // namespace

int main() {
    try {
        require(nbody::openmp_barnes_hut_compiled(),
                "OpenMP Barnes-Hut test was built without OpenMP support");
        const auto thread_counts = supported_thread_counts();
        test_representation_comparator_negative_control();
        test_force_equivalence(thread_counts);
        test_boundaries_mass_extremes_and_self_force(thread_counts);
        test_extreme_and_degenerate_inputs(thread_counts);
        test_limits_and_transactional_statistics(thread_counts);
        test_actual_capacity_memory_admission(thread_counts);
        test_nested_openmp_rejection(thread_counts);
        test_velocity_verlet_and_conservation(thread_counts);
        std::cout << "OpenMP Barnes-Hut tests passed with";
        for (const int threads : thread_counts) {
            std::cout << ' ' << threads;
        }
#ifdef NBODY_BARNES_HUT_EXPECT_RUNTIME_SCHEDULE
        std::cout << " threads; actual runtime schedules: static dynamic guided\n";
#else
        std::cout << " threads; production schedule: static\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OpenMP Barnes-Hut test failure: " << error.what() << '\n';
        return 1;
    }
}
