#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"

#ifdef NBODY_HAS_OPENMP
#include "nbody/openmp_backend.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef NBODY_BUILD_CONFIGURATION
#define NBODY_BUILD_CONFIGURATION "unknown"
#endif

namespace {

struct Options {
    std::string backend{"sequential"};
    std::size_t bodies{256};
    std::size_t steps{100};
    double dt_s{10.0};
    int threads{1};
    std::size_t warmups{1};
    std::size_t repetitions{5};
    std::uint64_t seed{0x4e424f4459ULL};
};

struct RunResult {
    double elapsed_s{0.0};
    double selector_steady_state_s{0.0};
    nbody::SimulationState final_state;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " [--backend sequential|openmp] [--bodies N] [--steps N] [--dt SECONDS]\n"
        << "       [--threads N] [--warmups N] [--repetitions N] [--seed N]\n";
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
        if (argument == "--backend") {
            options.backend = value;
        } else if (argument == "--bodies") {
            options.bodies = std::stoull(value);
        } else if (argument == "--steps") {
            options.steps = std::stoull(value);
        } else if (argument == "--dt") {
            options.dt_s = std::stod(value);
        } else if (argument == "--threads") {
            options.threads = std::stoi(value);
        } else if (argument == "--warmups") {
            options.warmups = std::stoull(value);
        } else if (argument == "--repetitions") {
            options.repetitions = std::stoull(value);
        } else if (argument == "--seed") {
            options.seed = std::stoull(value);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.bodies < 2 || options.steps == 0 || options.repetitions == 0 ||
        !(options.dt_s > 0.0) || options.threads <= 0) {
        throw std::invalid_argument(
            "bodies >= 2, steps/repetitions/threads > 0, and dt > 0 are required");
    }
    return options;
}

RunResult run_once(
    const nbody::SimulationState& initial_state,
    const nbody::ForceBackend& backend,
    std::size_t steps,
    double dt_s) {
    const auto selector_start = std::chrono::steady_clock::now();
    nbody::SimulationState state = initial_state;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t step = 0; step < steps; ++step) {
        nbody::velocity_verlet_step(state, dt_s, backend);
    }
    const auto stop = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double>(stop - start).count(),
        std::chrono::duration<double>(stop - selector_start).count(),
        std::move(state),
    };
}

double quantile(const std::vector<double>& sorted_values, double probability) {
    const double index = probability * static_cast<double>(sorted_values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = static_cast<std::size_t>(std::ceil(index));
    const double fraction = index - static_cast<double>(lower);
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction;
}

double relative_change(double value, double reference, double absolute_floor) {
    return std::abs(value - reference) /
        std::max({std::abs(value), std::abs(reference), absolute_floor});
}

std::string environment_value(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : value;
}

std::string csv_quote(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

std::string compiler_description() {
#if defined(__GNUC__)
    return "GNU " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
        std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
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
        const Options options = parse_options(argc, argv);
        const nbody::ForceBackend* backend = &nbody::sequential_force_backend();
        std::unique_ptr<nbody::ForceBackend> owned_backend;

        if (options.backend == "openmp") {
#ifdef NBODY_HAS_OPENMP
            owned_backend = std::make_unique<nbody::OpenMpForceBackend>(options.threads);
            backend = owned_backend.get();
#else
            throw std::runtime_error(
                "OpenMP backend is unavailable in this build; configure with -DNBODY_ENABLE_OPENMP=ON");
#endif
        } else if (options.backend != "sequential") {
            throw std::invalid_argument("backend must be sequential or openmp");
        }

        const auto initial_state =
            nbody::make_seeded_synthetic_system(options.bodies, options.seed);
        const auto initial_diagnostics = nbody::compute_diagnostics(initial_state);

        for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
            (void)run_once(initial_state, *backend, options.steps, options.dt_s);
        }

        std::vector<double> timings;
        std::vector<double> selector_timings;
        timings.reserve(options.repetitions);
        selector_timings.reserve(options.repetitions);
        nbody::SimulationState final_state;
        for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
            RunResult result = run_once(initial_state, *backend, options.steps, options.dt_s);
            timings.push_back(result.elapsed_s);
            selector_timings.push_back(result.selector_steady_state_s);
            final_state = std::move(result.final_state);
        }
        std::sort(timings.begin(), timings.end());
        std::sort(selector_timings.begin(), selector_timings.end());

        const auto final_diagnostics = nbody::compute_diagnostics(final_state);
        const double energy_drift = relative_change(
            final_diagnostics.total_energy_j(), initial_diagnostics.total_energy_j(), 1.0);
        const double angular_momentum_drift =
            nbody::norm(final_diagnostics.angular_momentum_kg_m2_s -
                        initial_diagnostics.angular_momentum_kg_m2_s) /
            std::max(nbody::norm(initial_diagnostics.angular_momentum_kg_m2_s), 1.0);

        int effective_threads = 1;
        int openmp_specification_date = 0;
#ifdef NBODY_HAS_OPENMP
        if (options.backend == "openmp") {
            effective_threads = nbody::openmp_effective_thread_count(options.threads);
            openmp_specification_date = nbody::openmp_specification_date();
        }
#endif

        std::cout << std::setprecision(17)
                  << "backend,requested_threads,effective_threads,bodies,steps,dt_s,warmups,repetitions,median_s,"
                     "iqr_s,min_s,max_s,initial_energy_j,final_energy_j,relative_energy_drift,"
                     "relative_angular_momentum_drift,final_momentum_norm_kg_m_s,"
                     "final_center_of_mass_norm_m,checksum,compiler,build_configuration,"
                     "hardware_threads,host,processor,openmp_specification_date,omp_dynamic,"
                     "omp_proc_bind,omp_places,gomp_cpu_affinity,selector_steady_state_s\n"
                  << backend->name() << ','
                  << (options.backend == "openmp" ? options.threads : 1) << ','
                  << effective_threads << ','
                  << options.bodies << ',' << options.steps << ',' << options.dt_s << ','
                  << options.warmups << ',' << options.repetitions << ','
                  << quantile(timings, 0.5) << ','
                  << (quantile(timings, 0.75) - quantile(timings, 0.25)) << ','
                  << timings.front() << ',' << timings.back() << ','
                  << initial_diagnostics.total_energy_j() << ','
                  << final_diagnostics.total_energy_j() << ',' << energy_drift << ','
                  << angular_momentum_drift << ','
                  << nbody::norm(final_diagnostics.linear_momentum_kg_m_s) << ','
                  << nbody::norm(final_diagnostics.center_of_mass_m) << ','
                  << state_checksum(final_state) << ',' << csv_quote(compiler_description()) << ','
                  << csv_quote(NBODY_BUILD_CONFIGURATION) << ','
                  << std::thread::hardware_concurrency() << ','
                  << csv_quote(environment_value("COMPUTERNAME", "unknown")) << ','
                  << csv_quote(environment_value("PROCESSOR_IDENTIFIER", "unknown")) << ','
                  << openmp_specification_date << ','
                  << csv_quote(environment_value("OMP_DYNAMIC", "unset")) << ','
                  << csv_quote(environment_value("OMP_PROC_BIND", "unset")) << ','
                  << csv_quote(environment_value("OMP_PLACES", "unset")) << ','
                  << csv_quote(environment_value("GOMP_CPU_AFFINITY", "unset")) << ','
                  << quantile(selector_timings, 0.5) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
