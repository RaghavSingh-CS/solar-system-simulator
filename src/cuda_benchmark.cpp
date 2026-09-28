#include "nbody/cuda_backend.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#ifndef NBODY_BUILD_CONFIGURATION
#define NBODY_BUILD_CONFIGURATION "unknown"
#endif
#ifndef NBODY_STAGE5_SOURCE_SHA
#define NBODY_STAGE5_SOURCE_SHA "uncontrolled"
#endif
#ifndef NBODY_STAGE5_PROTOCOL_VERSION
#define NBODY_STAGE5_PROTOCOL_VERSION "none"
#endif
#ifndef NBODY_STAGE5_CXX_COMPILER
#define NBODY_STAGE5_CXX_COMPILER "unknown"
#endif
#ifndef NBODY_STAGE5_CXX_COMPILER_ID
#define NBODY_STAGE5_CXX_COMPILER_ID "unknown"
#endif
#ifndef NBODY_STAGE5_CXX_COMPILER_VERSION
#define NBODY_STAGE5_CXX_COMPILER_VERSION "unknown"
#endif
#ifndef NBODY_STAGE5_GENERATOR
#define NBODY_STAGE5_GENERATOR "unknown"
#endif
#ifndef NBODY_STAGE5_CUDA_COMPILER
#define NBODY_STAGE5_CUDA_COMPILER "unknown"
#endif
#ifndef NBODY_STAGE5_CUDA_COMPILER_ID
#define NBODY_STAGE5_CUDA_COMPILER_ID "unknown"
#endif
#ifndef NBODY_STAGE5_CUDA_COMPILER_VERSION
#define NBODY_STAGE5_CUDA_COMPILER_VERSION "unknown"
#endif
#ifndef NBODY_STAGE5_CUDA_HOST_COMPILER
#define NBODY_STAGE5_CUDA_HOST_COMPILER "unknown"
#endif
#ifndef NBODY_STAGE5_CUDA_ARCHITECTURES
#define NBODY_STAGE5_CUDA_ARCHITECTURES "unknown"
#endif

namespace {

struct Options {
    std::size_t bodies{256};
    std::size_t steps{100};
    double dt_s{10.0};
    std::size_t warmups{1};
    std::size_t repetitions{1};
    std::uint64_t seed{24301};
    int device{0};
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--bodies N] [--steps N] [--dt SECONDS] [--warmups N]"
                 " [--repetitions N] [--seed N] [--device N]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++i];
        if (argument == "--bodies") {
            options.bodies = std::stoull(value);
        } else if (argument == "--steps") {
            options.steps = std::stoull(value);
        } else if (argument == "--dt") {
            options.dt_s = std::stod(value);
        } else if (argument == "--warmups") {
            options.warmups = std::stoull(value);
        } else if (argument == "--repetitions") {
            options.repetitions = std::stoull(value);
        } else if (argument == "--seed") {
            options.seed = std::stoull(value);
        } else if (argument == "--device") {
            options.device = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.bodies < 2 || options.steps == 0 || options.repetitions == 0 ||
        !(options.dt_s > 0.0) || options.device < 0) {
        throw std::invalid_argument(
            "bodies >= 2, steps/repetitions > 0, dt > 0, and device >= 0 are required");
    }
    return options;
}

std::string csv_quote(const std::string& value) {
    std::string escaped{"\""};
    for (const char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

double relative_scalar_difference(double value, double reference, double floor) {
    return std::abs(value - reference) /
        std::max({std::abs(value), std::abs(reference), floor});
}

double state_checksum(const nbody::SimulationState& state) {
    double checksum = state.time_s;
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        const double weight = static_cast<double>(i + 1);
        checksum += weight *
            (state.bodies[i].position_m.x + 3.0 * state.bodies[i].position_m.y +
             5.0 * state.bodies[i].position_m.z + state.bodies[i].velocity_m_s.x +
             7.0 * state.bodies[i].velocity_m_s.y + 11.0 * state.bodies[i].velocity_m_s.z);
    }
    return checksum;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--build-info") {
            std::cout << "{\"component\":\"nbody_cuda_benchmark\",\"source_commit\":\""
                      << NBODY_STAGE5_SOURCE_SHA << "\",\"protocol_version\":\""
                      << NBODY_STAGE5_PROTOCOL_VERSION << "\",\"build_configuration\":\""
                      << NBODY_BUILD_CONFIGURATION << "\",\"cxx_compiler\":\""
                      << NBODY_STAGE5_CXX_COMPILER << "\",\"compiler_id\":\""
                      << NBODY_STAGE5_CXX_COMPILER_ID << "\",\"compiler_version\":\""
                      << NBODY_STAGE5_CXX_COMPILER_VERSION << "\",\"generator\":\""
                      << NBODY_STAGE5_GENERATOR << "\",\"cuda_compiler\":\""
                      << NBODY_STAGE5_CUDA_COMPILER << "\",\"cuda_compiler_id\":\""
                      << NBODY_STAGE5_CUDA_COMPILER_ID << "\",\"cuda_compiler_version\":\""
                      << NBODY_STAGE5_CUDA_COMPILER_VERSION << "\",\"cuda_host_compiler\":\""
                      << NBODY_STAGE5_CUDA_HOST_COMPILER << "\",\"cuda_architectures\":\""
                      << NBODY_STAGE5_CUDA_ARCHITECTURES << "\"}\n";
            return 0;
        }
        const Options options = parse_options(argc, argv);
        const auto initial_state = nbody::make_seeded_synthetic_system(options.bodies, options.seed);
        const auto initial_diagnostics = nbody::compute_diagnostics(initial_state);
        nbody::CudaVelocityVerlet integrator(options.bodies, options.device);
        const nbody::CudaDeviceInfo device = integrator.device_info();

        for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
            (void)integrator.run(initial_state, options.steps, options.dt_s);
        }

        std::cout << "backend,bodies,steps,dt_s,warmups,repetition,seed,device_index,"
                     "device_compute_ms,transfer_inclusive_ms,host_to_device_ms,"
                     "device_to_host_ms,host_staging_ms,backend_wall_ms,allocation_ms,"
                     "initial_energy_j,final_energy_j,relative_energy_drift,"
                     "relative_angular_momentum_drift,final_momentum_norm_kg_m_s,"
                     "final_center_of_mass_norm_m,checksum,correctness_status,"
                     "gpu_name,compute_capability,total_global_memory_bytes,"
                     "multiprocessor_count,cuda_driver_version,cuda_runtime_version,"
                     "build_configuration,host_threads\n";

        bool all_valid = true;
        for (std::size_t repetition = 1; repetition <= options.repetitions; ++repetition) {
            const auto result = integrator.run(initial_state, options.steps, options.dt_s);
            const auto diagnostics = nbody::compute_diagnostics(result.final_state);
            const double energy_drift = relative_scalar_difference(
                diagnostics.total_energy_j(), initial_diagnostics.total_energy_j(), 1.0);
            const double angular_drift =
                nbody::norm(diagnostics.angular_momentum_kg_m2_s -
                            initial_diagnostics.angular_momentum_kg_m2_s) /
                std::max(nbody::norm(initial_diagnostics.angular_momentum_kg_m2_s), 1.0);
            const bool valid = std::isfinite(energy_drift) && std::isfinite(angular_drift) &&
                energy_drift <= 5.0e-6 && angular_drift <= 5.0e-10;
            all_valid = all_valid && valid;

            std::cout << std::setprecision(17)
                      << "cuda," << options.bodies << ',' << options.steps << ','
                      << options.dt_s << ',' << options.warmups << ',' << repetition << ','
                      << options.seed << ',' << options.device << ','
                      << result.timings.device_compute_ms << ','
                      << result.timings.transfer_inclusive_ms << ','
                      << result.timings.host_to_device_ms << ','
                      << result.timings.device_to_host_ms << ','
                      << result.timings.host_staging_ms << ','
                      << result.timings.backend_wall_ms << ','
                      << result.timings.allocation_ms << ','
                      << initial_diagnostics.total_energy_j() << ','
                      << diagnostics.total_energy_j() << ',' << energy_drift << ','
                      << angular_drift << ','
                      << nbody::norm(diagnostics.linear_momentum_kg_m_s) << ','
                      << nbody::norm(diagnostics.center_of_mass_m) << ','
                      << state_checksum(result.final_state) << ','
                      << (valid ? "pass" : "fail") << ',' << csv_quote(device.name) << ','
                      << device.compute_capability_major << '.'
                      << device.compute_capability_minor << ','
                      << device.total_global_memory_bytes << ','
                      << device.multiprocessor_count << ',' << device.cuda_driver_version << ','
                      << device.cuda_runtime_version << ','
                      << csv_quote(NBODY_BUILD_CONFIGURATION) << ','
                      << std::thread::hardware_concurrency() << '\n';
        }
        return all_valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
