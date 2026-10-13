#include "nbody/barnes_hut.hpp"
#include "nbody/scenarios.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
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

struct Options {
    std::string output_path;
    std::size_t repetitions{3};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            std::cout << "Usage: " << argv[0] << " --output PATH [--repetitions N]\n";
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

struct Errors {
    double maximum_absolute{0.0};
    double rms_absolute{0.0};
    double maximum_relative{0.0};
    double percentile_95_relative{0.0};
};

Errors summarize_errors(
    const std::vector<nbody::Vec3>& reference,
    const std::vector<nbody::Vec3>& candidate) {
    std::vector<double> relative;
    relative.reserve(reference.size());
    long double squares = 0.0L;
    Errors result;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const nbody::Vec3 difference = candidate[index] - reference[index];
        const double absolute = std::hypot(difference.x, difference.y, difference.z);
        const double reference_magnitude = std::max(
            std::hypot(reference[index].x, reference[index].y, reference[index].z),
            1.0e-30);
        const double scaled = absolute / reference_magnitude;
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        result.maximum_relative = std::max(result.maximum_relative, scaled);
        squares += static_cast<long double>(absolute) * absolute;
        relative.push_back(scaled);
    }
    result.rms_absolute = std::sqrt(
        static_cast<double>(squares / static_cast<long double>(reference.size())));
    std::sort(relative.begin(), relative.end());
    result.percentile_95_relative = relative[(95 * relative.size() + 99) / 100 - 1];
    return result;
}

double elapsed_seconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::ofstream output(options.output_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("could not open output file: " + options.output_path);
        }
        output << std::setprecision(17)
               << "# study=sequential_barnes_hut_stage1_exploratory\n"
               << "# stage5_evidence=false\n"
               << "# build_configuration=" << NBODY_BUILD_CONFIGURATION << "\n"
               << "# cxx_compiler_id=" << NBODY_CXX_COMPILER_ID << "\n"
               << "# cxx_compiler_version=" << NBODY_CXX_COMPILER_VERSION << "\n"
               << "# warmups_per_configuration=1\n"
               << "# timing_scope=single_force_and_single_velocity_verlet_step\n"
               << "distribution,bodies,theta,repetition,direct_force_s,tree_build_s,"
                  "tree_force_s,barnes_hut_force_wall_s,barnes_hut_step_s,node_count,"
                  "leaf_count,max_depth,visited_nodes,approximated_nodes,direct_interactions,"
                  "working_memory_bytes,max_absolute_error_m_s2,rms_absolute_error_m_s2,"
                  "max_relative_error,p95_relative_error\n";

        constexpr std::size_t sizes[] = {128, 512, 2048, 4096};
        constexpr double thetas[] = {0.25, 0.5, 0.8};
        const std::vector<std::string> distributions{"uniform", "clustered", "planar"};
        for (const std::string& distribution : distributions) {
            for (std::size_t body_count : sizes) {
                const auto state = make_distribution(
                    distribution, body_count, 0x42485f5354414745ULL + body_count);
                for (double theta : thetas) {
                    nbody::BarnesHutConfiguration configuration;
                    configuration.theta = theta;
                    configuration.leaf_capacity = 8;
                    nbody::SequentialBarnesHutForceBackend barnes_hut(configuration);
                    (void)nbody::compute_accelerations(state);
                    (void)barnes_hut.compute_accelerations(state);
                    for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
                        const auto direct_start = std::chrono::steady_clock::now();
                        const auto direct = nbody::compute_accelerations(state);
                        const auto direct_end = std::chrono::steady_clock::now();

                        const auto force_start = std::chrono::steady_clock::now();
                        const auto approximate = barnes_hut.compute_accelerations(state);
                        const auto force_end = std::chrono::steady_clock::now();
                        const auto statistics = barnes_hut.last_statistics();
                        const auto errors = summarize_errors(direct, approximate);

                        auto advanced = state;
                        const auto step_start = std::chrono::steady_clock::now();
                        nbody::velocity_verlet_step(advanced, 10.0, barnes_hut);
                        const auto step_end = std::chrono::steady_clock::now();

                        output << distribution << ',' << body_count << ',' << theta << ','
                               << repetition << ','
                               << elapsed_seconds(direct_start, direct_end) << ','
                               << statistics.tree_build_seconds << ','
                               << statistics.force_evaluation_seconds << ','
                               << elapsed_seconds(force_start, force_end) << ','
                               << elapsed_seconds(step_start, step_end) << ','
                               << statistics.node_count << ',' << statistics.leaf_count << ','
                               << statistics.maximum_depth_reached << ','
                               << statistics.visited_nodes << ','
                               << statistics.approximated_nodes << ','
                               << statistics.direct_body_interactions << ','
                               << statistics.working_memory_bytes << ','
                               << errors.maximum_absolute << ',' << errors.rms_absolute << ','
                               << errors.maximum_relative << ',' << errors.percentile_95_relative
                               << '\n';
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
