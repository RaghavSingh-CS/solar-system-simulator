#include "nbody/barnes_hut.hpp"
#include "nbody/cuda_backend.hpp"
#include "nbody/cuda_barnes_hut.hpp"
#include "nbody/scenarios.hpp"
#include "cuda_test_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
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

struct ErrorMetrics {
    double maximum_absolute{0.0};
    double rms_absolute{0.0};
    double maximum_relative{0.0};
    double percentile_95_relative{0.0};
};

ErrorMetrics error_metrics(
    const std::vector<nbody::Vec3>& reference,
    const std::vector<nbody::Vec3>& candidate) {
    require(reference.size() == candidate.size(), "force vector sizes differ");
    std::vector<double> relative;
    relative.reserve(reference.size());
    long double squared_sum = 0.0L;
    ErrorMetrics result;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double absolute = nbody::norm(candidate[index] - reference[index]);
        require(std::isfinite(absolute), "force difference is not finite");
        const double scale = std::max(nbody::norm(reference[index]), 1.0e-30);
        const double scaled = absolute / scale;
        require(std::isfinite(scaled), "relative force difference is not finite");
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        result.maximum_relative = std::max(result.maximum_relative, scaled);
        squared_sum += static_cast<long double>(absolute) * absolute;
        relative.push_back(scaled);
    }
    if (!relative.empty()) {
        result.rms_absolute = std::sqrt(
            static_cast<double>(squared_sum / relative.size()));
        std::sort(relative.begin(), relative.end());
        const std::size_t percentile_index =
            (95 * relative.size() + 99) / 100 - 1;
        result.percentile_95_relative = relative[percentile_index];
    }
    return result;
}

void require_finite(const std::vector<nbody::Vec3>& values, const std::string& label) {
    for (const auto& value : values) {
        require(std::isfinite(value.x) && std::isfinite(value.y) &&
                    std::isfinite(value.z),
                label + " contains a non-finite acceleration");
    }
}

void require_failure_metadata(
    const nbody::CudaBarnesHutFailure& failure,
    const nbody::SimulationState& state,
    std::size_t node_count,
    nbody::CudaBarnesHutFailureCode expected,
    const std::string& label) {
    require(failure.code == expected, label + " failure code differs");
    require(failure.target_index < state.bodies.size(),
            label + " target index is outside the fixture");
    require(failure.node_index < node_count,
            label + " node index is outside the packed tree");
}

template <typename Exception>
nbody::CudaBarnesHutFailure require_device_failure_without_publication(
    nbody::CudaBarnesHutForceBackend& backend,
    const nbody::SimulationState& state,
    const std::string& label) {
    const std::vector<nbody::Vec3> sentinel{{1.0, 2.0, 3.0}, {-4.0, 5.0, -6.0}};
    std::vector<nbody::Vec3> published = sentinel;
    bool threw = false;
    try {
        published = backend.compute_accelerations(state);
    } catch (const Exception&) {
        threw = true;
    }
    require(threw, label + " did not throw the expected exception");
    require(published.size() == sentinel.size(), label + " published partial output");
    for (std::size_t index = 0; index < sentinel.size(); ++index) {
        require(published[index].x == sentinel[index].x &&
                    published[index].y == sentinel[index].y &&
                    published[index].z == sentinel[index].z,
                label + " mutated output before failure publication");
    }
    return backend.last_failure();
}

nbody::SimulationState clustered_state() {
    auto state = nbody::make_seeded_synthetic_system(96, 0x434c5553544552ULL);
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        const double center = index < state.bodies.size() / 2 ? -2.0e9 : 2.0e9;
        state.bodies[index].position_m.x =
            center + static_cast<double>(index % 11) * 1.0e5;
        state.bodies[index].position_m.y =
            static_cast<double>((index * 7) % 13) * 2.0e5;
        state.bodies[index].position_m.z =
            static_cast<double>((index * 5) % 17) * 3.0e5;
    }
    return state;
}

nbody::SimulationState planar_state() {
    auto state = nbody::make_seeded_synthetic_system(96, 0x504c414e4152ULL);
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        state.bodies[index].position_m.z = 0.0;
    }
    return state;
}

nbody::SimulationState boundary_state() {
    nbody::SimulationState state;
    const std::array<nbody::Vec3, 10> positions{{
        {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0},
        {-1.0, 1.0, -1.0}, {1.0, 1.0, -1.0},
        {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
        {-1.0, 1.0, 1.0}, {1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.5},
    }};
    for (std::size_t index = 0; index < positions.size(); ++index) {
        state.bodies.push_back({
            "boundary-" + std::to_string(index),
            1.0e20 + static_cast<double>(index) * 1.0e18,
            positions[index] * 1.0e7,
            {},
        });
    }
    return state;
}

void compare_with_cpu(
    const nbody::SimulationState& state,
    nbody::BarnesHutConfiguration tree_configuration,
    const std::string& label) {
    nbody::SequentialBarnesHutForceBackend cpu(tree_configuration);
    nbody::CudaBarnesHutConfiguration cuda_configuration;
    cuda_configuration.tree = tree_configuration;
    nbody::CudaBarnesHutForceBackend cuda(cuda_configuration);
    const auto reference = cpu.compute_accelerations(state);
    const auto candidate = cuda.compute_accelerations(state);
    require_finite(reference, label + " CPU");
    require_finite(candidate, label + " CUDA");
    const ErrorMetrics cpu_cuda = error_metrics(reference, candidate);
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double absolute = nbody::norm(candidate[index] - reference[index]);
        const double relative = absolute /
            std::max(nbody::norm(reference[index]), 1.0e-30);
        require(absolute <= 1.0e-18 || relative <= 5.0e-12,
                label + " exceeded the predeclared CPU/CUDA mixed tolerance");
    }
    require(std::isfinite(cpu_cuda.maximum_absolute) &&
                std::isfinite(cpu_cuda.rms_absolute) &&
                std::isfinite(cpu_cuda.maximum_relative) &&
                std::isfinite(cpu_cuda.percentile_95_relative),
            label + " CPU/CUDA error characterization is not finite");

    const auto direct = nbody::compute_accelerations(state);
    const ErrorMetrics approximation = error_metrics(direct, candidate);
    require(std::isfinite(approximation.maximum_absolute) &&
                std::isfinite(approximation.rms_absolute) &&
                std::isfinite(approximation.maximum_relative) &&
                std::isfinite(approximation.percentile_95_relative),
            label + " direct-reference error characterization is not finite");

    const auto statistics = cuda.last_statistics();
    require(statistics.body_count == state.bodies.size(), label + " body count differs");
    require(statistics.node_count > 0, label + " node count is empty");
    require(statistics.traversal_work > 0 &&
                statistics.traversal_work <= tree_configuration.maximum_traversal_work,
            label + " traversal work is outside its hard bound");
    require(cuda.last_failure().code == nbody::CudaBarnesHutFailureCode::none,
            label + " retained a failure after success");
}

void test_single_body() {
    nbody::SimulationState state;
    state.bodies.push_back({"single", 1.0e20, {4.0, -3.0, 2.0}, {}});
    nbody::CudaBarnesHutForceBackend cuda;
    const auto acceleration = cuda.compute_accelerations(state);
    require(acceleration.size() == 1, "single-body output size differs");
    require(acceleration[0].x == 0.0 && acceleration[0].y == 0.0 &&
                acceleration[0].z == 0.0,
            "single body has self acceleration");
}

void test_reference_fixtures() {
    nbody::BarnesHutConfiguration configuration;
    compare_with_cpu(
        nbody::make_analytic_circular_two_body().initial_state,
        configuration,
        "two body");
    compare_with_cpu(
        nbody::make_seeded_synthetic_system(128, 0x554e49464f524dULL),
        configuration,
        "uniform");
    compare_with_cpu(clustered_state(), configuration, "clustered");
    compare_with_cpu(planar_state(), configuration, "planar");
    compare_with_cpu(boundary_state(), configuration, "boundary");

    auto unequal = boundary_state();
    unequal.bodies.front().mass_kg = 1.0e30;
    unequal.bodies.back().mass_kg = 1.0e10;
    compare_with_cpu(unequal, configuration, "unequal masses");

    auto near = nbody::make_analytic_circular_two_body().initial_state;
    near.bodies[0].position_m = {0.0, 0.0, 0.0};
    near.bodies[1].position_m = {1.0e-6, 0.0, 0.0};
    near.bodies[0].mass_kg = 1.0;
    near.bodies[1].mass_kg = 2.0;
    compare_with_cpu(near, configuration, "near coincident");

    nbody::SimulationState extreme;
    extreme.bodies = {
        {"negative", 1.0e100, {-1.0e150, 0.0, 0.0}, {}},
        {"positive", 2.0e100, {1.0e150, 0.0, 0.0}, {}},
        {"offset", 3.0e100, {0.0, 1.0e150, 0.0}, {}},
    };
    compare_with_cpu(extreme, configuration, "extreme binary64");

    configuration.theta = 1.0e-6;
    compare_with_cpu(clustered_state(), configuration, "small theta");
    configuration.theta = 2.0;
    compare_with_cpu(clustered_state(), configuration, "large theta");
    configuration.theta = 0.25;
    configuration.maximum_depth = 1;
    configuration.leaf_capacity = 1;
    compare_with_cpu(boundary_state(), configuration, "maximum depth termination");
}

void test_explicit_failures() {
    auto coincident = boundary_state();
    coincident.bodies[1].position_m = coincident.bodies[0].position_m;
    const auto coincident_tree = nbody::build_barnes_hut_packed_tree(coincident);
    nbody::CudaBarnesHutForceBackend singular;
    const auto singular_failure =
        require_device_failure_without_publication<std::domain_error>(
            singular, coincident, "coincident bodies");
    require_failure_metadata(
        singular_failure,
        coincident,
        coincident_tree.nodes.size(),
        nbody::CudaBarnesHutFailureCode::singular_geometry,
        "coincident bodies");

    nbody::BarnesHutConfiguration node_configuration;
    node_configuration.maximum_nodes = 1;
    node_configuration.leaf_capacity = 1;
    nbody::CudaBarnesHutConfiguration cuda_node_configuration;
    cuda_node_configuration.tree = node_configuration;
    nbody::CudaBarnesHutForceBackend node_limited(cuda_node_configuration);
    require_throws<std::length_error>(
        [&] { (void)node_limited.compute_accelerations(boundary_state()); },
        "node exhaustion was accepted");

    nbody::CudaBarnesHutConfiguration device_memory_configuration;
    device_memory_configuration.device_memory_budget_bytes = 1;
    nbody::CudaBarnesHutForceBackend device_memory_limited(
        device_memory_configuration);
    require_throws<std::length_error>(
        [&] { (void)device_memory_limited.compute_accelerations(boundary_state()); },
        "device memory exhaustion was accepted");

    nbody::CudaBarnesHutConfiguration host_memory_configuration;
    host_memory_configuration.host_staging_memory_budget_bytes = 1;
    nbody::CudaBarnesHutForceBackend host_memory_limited(host_memory_configuration);
    require_throws<std::length_error>(
        [&] { (void)host_memory_limited.compute_accelerations(boundary_state()); },
        "host staging exhaustion was accepted");

    nbody::CudaBarnesHutConfiguration stack_configuration;
    stack_configuration.tree.leaf_capacity = 1;
    stack_configuration.tree.theta = 1.0e-6;
    stack_configuration.traversal_stack_capacity = 1;
    const auto stack_state = boundary_state();
    const auto stack_tree = nbody::build_barnes_hut_packed_tree(
        stack_state, stack_configuration.tree);
    nbody::CudaBarnesHutForceBackend stack_limited(stack_configuration);
    const auto stack_failure =
        require_device_failure_without_publication<std::length_error>(
            stack_limited, stack_state, "traversal stack exhaustion");
    require_failure_metadata(
        stack_failure,
        stack_state,
        stack_tree.nodes.size(),
        nbody::CudaBarnesHutFailureCode::traversal_stack_exhausted,
        "traversal stack exhaustion");

    nbody::CudaBarnesHutConfiguration work_configuration;
    work_configuration.tree.maximum_traversal_work = 1;
    const auto work_state = boundary_state();
    const auto work_tree = nbody::build_barnes_hut_packed_tree(
        work_state, work_configuration.tree);
    nbody::CudaBarnesHutForceBackend work_limited(work_configuration);
    const auto work_failure =
        require_device_failure_without_publication<std::length_error>(
            work_limited, work_state, "traversal work exhaustion");
    require_failure_metadata(
        work_failure,
        work_state,
        work_tree.nodes.size(),
        nbody::CudaBarnesHutFailureCode::traversal_work_exhausted,
        "traversal work exhaustion");
}

void test_failure_publication_contention() {
    nbody::SimulationState coincident;
    for (std::size_t index = 0; index < 256; ++index) {
        coincident.bodies.push_back({
            "contended-" + std::to_string(index),
            1.0e20 + static_cast<double>(index),
            {7.0, -3.0, 11.0},
            {},
        });
    }
    const auto tree = nbody::build_barnes_hut_packed_tree(coincident);
    nbody::CudaBarnesHutForceBackend backend;
    for (int repetition = 0; repetition < 16; ++repetition) {
        const auto failure =
            require_device_failure_without_publication<std::domain_error>(
                backend, coincident, "contended singular failure");
        require_failure_metadata(
            failure,
            coincident,
            tree.nodes.size(),
            nbody::CudaBarnesHutFailureCode::singular_geometry,
            "contended singular failure");
    }

    const auto work_state = nbody::make_seeded_synthetic_system(
        512, 0x434f4e54454e5449ULL);
    nbody::CudaBarnesHutConfiguration work_configuration;
    work_configuration.tree.maximum_traversal_work = 32;
    const auto work_tree = nbody::build_barnes_hut_packed_tree(
        work_state, work_configuration.tree);
    nbody::CudaBarnesHutForceBackend work_backend(work_configuration);
    for (int repetition = 0; repetition < 16; ++repetition) {
        const auto failure =
            require_device_failure_without_publication<std::length_error>(
                work_backend, work_state, "contended work failure");
        require_failure_metadata(
            failure,
            work_state,
            work_tree.nodes.size(),
            nbody::CudaBarnesHutFailureCode::traversal_work_exhausted,
            "contended work failure");
    }
}

void test_live_release_failure_is_terminal() {
    nbody::CudaBarnesHutConfiguration configuration;
    configuration.host_failure_injection =
        nbody::CudaBarnesHutHostFailureInjection::
            live_release_before_first_deallocation;
    nbody::CudaBarnesHutForceBackend backend(configuration);
    const auto small = nbody::make_seeded_synthetic_system(8, 0x534d414c4cULL);
    const auto large = nbody::make_seeded_synthetic_system(64, 0x4c41524745ULL);
    (void)backend.compute_accelerations(small);
    const auto committed_statistics = backend.last_statistics();

    require_throws<std::runtime_error>(
        [&] { (void)backend.compute_accelerations(large); },
        "injected live-release failure was accepted");
    require(backend.last_failure().code ==
                nbody::CudaBarnesHutFailureCode::runtime_failure,
            "live-release failure did not terminalize the executor");
    require(backend.last_statistics().body_count == committed_statistics.body_count &&
                backend.last_statistics().node_count == committed_statistics.node_count &&
                backend.last_statistics().device_storage_bytes ==
                    committed_statistics.device_storage_bytes &&
                backend.last_statistics().host_staging_bytes ==
                    committed_statistics.host_staging_bytes,
            "live-release failure changed last-success statistics");
    require_throws<std::runtime_error>(
        [&] { (void)backend.compute_accelerations(small); },
        "terminal executor was reused after live-release failure");
    require(backend.last_failure().code ==
                nbody::CudaBarnesHutFailureCode::runtime_failure,
            "terminal retry changed the retained runtime failure");
}

void test_repeated_lifetime() {
    const auto state = nbody::make_seeded_synthetic_system(32, 0x4c49464554494d45ULL);
    for (int iteration = 0; iteration < 8; ++iteration) {
        nbody::CudaBarnesHutForceBackend cuda;
        (void)cuda.compute_accelerations(state);
        (void)cuda.compute_accelerations(state);
    }
}

}  // namespace

int main() {
    try {
        const auto decision = nbody::test_support::strict_cuda_runtime_test_decision(
            [] { return nbody::cuda_device_count(); });
        if (decision == nbody::test_support::CudaRuntimeTestDecision::skip_no_device) {
            std::cout << "SKIP: CUDA support compiled, but no CUDA device is available\n";
            return 77;
        }
        test_single_body();
        test_reference_fixtures();
        test_explicit_failures();
        test_failure_publication_contention();
        test_live_release_failure_is_terminal();
        test_repeated_lifetime();
        const auto device = nbody::cuda_device_info();
        std::cout << "CUDA Barnes-Hut validation passed on " << device.name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA Barnes-Hut test failure: " << error.what() << '\n';
        return 1;
    }
}
