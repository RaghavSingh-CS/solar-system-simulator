#include "nbody/barnes_hut.hpp"
#include "nbody/openmp_backend.hpp"
#include "nbody/scenarios.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

#ifndef NBODY_BUILD_CONFIGURATION
#define NBODY_BUILD_CONFIGURATION "unknown"
#endif
#ifndef NBODY_CXX_COMPILER_ID
#define NBODY_CXX_COMPILER_ID "unknown"
#endif
#ifndef NBODY_CXX_COMPILER_VERSION
#define NBODY_CXX_COMPILER_VERSION "unknown"
#endif
#ifndef NBODY_OPENMP_SPECIFICATION
#define NBODY_OPENMP_SPECIFICATION "unknown"
#endif

struct Options {
    std::string output_path;
    std::size_t repetitions{2};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " --output PATH [--repetitions N]\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--output") {
            options.output_path = value;
        } else if (argument == "--repetitions") {
            options.repetitions = std::stoull(value);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.output_path.empty() || options.repetitions == 0) {
        throw std::invalid_argument("output path and positive repetitions are required");
    }
    return options;
}

nbody::SimulationState make_distribution(
    const std::string& distribution,
    std::size_t body_count,
    std::uint64_t seed) {
    auto state = nbody::make_seeded_synthetic_system(body_count, seed);
    if (distribution == "clustered") {
        for (std::size_t index = 0; index < state.bodies.size(); ++index) {
            const double cluster = index % 2 == 0 ? -4.0e8 : 4.0e8;
            state.bodies[index].position_m *= 0.06;
            state.bodies[index].position_m.x += cluster;
        }
    } else if (distribution == "planar") {
        for (std::size_t index = 0; index < state.bodies.size(); ++index) {
            const double angle = 6.283185307179586 * static_cast<double>(index) /
                static_cast<double>(state.bodies.size());
            const double radius = 2.0e8 + static_cast<double>(index % 31) * 2.0e6;
            state.bodies[index].position_m = {
                radius * std::cos(angle), radius * std::sin(angle), 0.0};
        }
    }
    return state;
}

double elapsed_seconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

std::uint64_t double_representation(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "benchmark requires binary64-sized double storage");
    static_assert(std::numeric_limits<double>::is_iec559,
                  "benchmark requires IEC 559 / IEEE-754 double semantics");
    std::uint64_t representation = 0;
    std::memcpy(&representation, &value, sizeof(representation));
    return representation;
}

bool bitwise_equal(double lhs, double rhs) {
    return double_representation(lhs) == double_representation(rhs);
}

bool bitwise_equal(
    const std::vector<nbody::Vec3>& lhs,
    const std::vector<nbody::Vec3>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (!bitwise_equal(lhs[index].x, rhs[index].x) ||
            !bitwise_equal(lhs[index].y, rhs[index].y) ||
            !bitwise_equal(lhs[index].z, rhs[index].z)) {
            return false;
        }
    }
    return true;
}

std::vector<int> supported_thread_counts() {
    std::vector<int> supported;
    for (const int requested : std::array<int, 4>{1, 2, 4, 8}) {
        if (requested <= nbody::openmp_runtime_max_threads() &&
            requested <= nbody::openmp_processor_count() &&
            nbody::openmp_effective_thread_count(requested) == requested) {
            supported.push_back(requested);
        }
    }
    if (supported.empty() || supported.front() != 1) {
        throw std::runtime_error("OpenMP runtime cannot provide even one worker thread");
    }
    return supported;
}

template <typename Function>
std::pair<double, std::vector<nbody::Vec3>> timed_force(Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    auto result = function();
    const auto end = std::chrono::steady_clock::now();
    return {elapsed_seconds(start, end), std::move(result)};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<int> thread_counts = supported_thread_counts();
        std::ofstream output(options.output_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("could not open output file: " + options.output_path);
        }

        output << std::setprecision(17)
               << "# study=openmp_barnes_hut_stage2_exploratory\n"
               << "# stage5_evidence=false\n"
               << "# build_configuration=" << NBODY_BUILD_CONFIGURATION << "\n"
               << "# cxx_compiler_id=" << NBODY_CXX_COMPILER_ID << "\n"
               << "# cxx_compiler_version=" << NBODY_CXX_COMPILER_VERSION << "\n"
               << "# openmp_specification=" << NBODY_OPENMP_SPECIFICATION << "\n"
               << "# openmp_runtime_specification=" << nbody::openmp_specification_date() << "\n"
               << "# openmp_processor_count=" << nbody::openmp_processor_count() << "\n"
               << "# openmp_runtime_max_threads=" << nbody::openmp_runtime_max_threads() << "\n"
               << "# warmups_per_configuration=1\n"
               << "# repetitions=" << options.repetitions << "\n"
               << "# dt_s=10\n"
               << "# timing_scope=single_force_call_and_single_velocity_verlet_step\n"
               << "distribution,bodies,theta,threads,repetition,"
                  "sequential_direct_force_s,openmp_direct_force_s,"
                  "sequential_bh_tree_build_s,sequential_bh_traversal_s,"
                  "sequential_bh_force_wall_s,openmp_bh_tree_build_s,"
                  "openmp_bh_traversal_s,openmp_bh_force_wall_s,"
                  "openmp_bh_step_s,node_count,leaf_count,max_depth,"
                  "visited_nodes,approximated_nodes,direct_interactions,"
                  "working_memory_bytes,effective_threads\n";

        constexpr std::array<std::size_t, 3> sizes{256, 1024, 4096};
        constexpr std::array<double, 3> thetas{0.25, 0.5, 0.8};
        const std::array<std::string, 3> distributions{"uniform", "clustered", "planar"};
        for (const std::string& distribution : distributions) {
            for (const std::size_t body_count : sizes) {
                const auto state = make_distribution(
                    distribution,
                    body_count,
                    0x4f4d5042485332ULL + body_count);
                for (const double theta : thetas) {
                    nbody::BarnesHutConfiguration configuration;
                    configuration.theta = theta;
                    configuration.leaf_capacity = 8;
                    nbody::SequentialBarnesHutForceBackend sequential_bh(configuration);
                    std::vector<std::unique_ptr<nbody::OpenMpForceBackend>> direct_backends;
                    std::vector<std::unique_ptr<nbody::OpenMpBarnesHutForceBackend>> bh_backends;
                    direct_backends.reserve(thread_counts.size());
                    bh_backends.reserve(thread_counts.size());
                    for (const int threads : thread_counts) {
                        direct_backends.push_back(
                            std::make_unique<nbody::OpenMpForceBackend>(threads));
                        bh_backends.push_back(
                            std::make_unique<nbody::OpenMpBarnesHutForceBackend>(
                                threads, configuration));
                    }

                    (void)nbody::compute_accelerations(state);
                    const auto warm_reference = sequential_bh.compute_accelerations(state);
                    for (std::size_t index = 0; index < thread_counts.size(); ++index) {
                        (void)direct_backends[index]->compute_accelerations(
                            state, nbody::gravitational_constant);
                        const auto warm_candidate =
                            bh_backends[index]->compute_accelerations(state);
                        if (!bitwise_equal(warm_reference, warm_candidate)) {
                            throw std::runtime_error(
                                "OpenMP Barnes-Hut warmup differed from Sequential reference");
                        }
                    }

                    for (std::size_t repetition = 0; repetition < options.repetitions;
                         ++repetition) {
                        const auto sequential_direct = timed_force([&]() {
                            return nbody::compute_accelerations(state);
                        });
                        const auto sequential_bh_force = timed_force([&]() {
                            return sequential_bh.compute_accelerations(state);
                        });
                        const auto sequential_statistics = sequential_bh.last_statistics();

                        for (std::size_t index = 0; index < thread_counts.size(); ++index) {
                            const auto openmp_direct = timed_force([&]() {
                                return direct_backends[index]->compute_accelerations(
                                    state, nbody::gravitational_constant);
                            });
                            const auto openmp_bh_force = timed_force([&]() {
                                return bh_backends[index]->compute_accelerations(state);
                            });
                            const auto parallel_statistics =
                                bh_backends[index]->last_statistics();
                            if (!bitwise_equal(
                                    sequential_bh_force.second, openmp_bh_force.second)) {
                                throw std::runtime_error(
                                    "OpenMP Barnes-Hut measurement differed from Sequential reference");
                            }

                            auto advanced = state;
                            const auto step_start = std::chrono::steady_clock::now();
                            nbody::velocity_verlet_step(
                                advanced, 10.0, *bh_backends[index]);
                            const auto step_end = std::chrono::steady_clock::now();

                            output << distribution << ',' << body_count << ',' << theta << ','
                                   << thread_counts[index] << ',' << repetition << ','
                                   << sequential_direct.first << ',' << openmp_direct.first << ','
                                   << sequential_statistics.tree_build_seconds << ','
                                   << sequential_statistics.force_evaluation_seconds << ','
                                   << sequential_bh_force.first << ','
                                   << parallel_statistics.tree_build_seconds << ','
                                   << parallel_statistics.force_evaluation_seconds << ','
                                   << openmp_bh_force.first << ','
                                   << elapsed_seconds(step_start, step_end) << ','
                                   << parallel_statistics.node_count << ','
                                   << parallel_statistics.leaf_count << ','
                                   << parallel_statistics.maximum_depth_reached << ','
                                   << parallel_statistics.visited_nodes << ','
                                   << parallel_statistics.approximated_nodes << ','
                                   << parallel_statistics.direct_body_interactions << ','
                                   << parallel_statistics.working_memory_bytes << ','
                                   << parallel_statistics.worker_threads << '\n';
                        }
                    }
                }
            }
        }
        std::cout << "wrote " << options.output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
