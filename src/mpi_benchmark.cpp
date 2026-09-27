#include "nbody/diagnostics.hpp"
#include "nbody/mpi_backend.hpp"
#include "nbody/scenarios.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NBODY_BUILD_CONFIGURATION
#define NBODY_BUILD_CONFIGURATION "unknown"
#endif

namespace {

struct Options {
    std::size_t bodies{256};
    std::size_t steps{10};
    double dt_s{10.0};
    int threads{1};
    std::size_t warmups{1};
    std::size_t repetitions{3};
    std::uint64_t seed{24301};
};

struct RunResult {
    double elapsed_s{0.0};
    double local_compute_s{0.0};
    double communication_s{0.0};
    nbody::SimulationState state;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: " << argv[0]
                << " [--bodies N] [--steps N] [--dt SECONDS] [--threads N]"
                   " [--warmups N] [--repetitions N] [--seed N]\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--bodies") {
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
    if (options.bodies < 2 || options.steps == 0 || options.threads <= 0 ||
        options.repetitions == 0 || !(options.dt_s > 0.0) || !std::isfinite(options.dt_s)) {
        throw std::invalid_argument("positive bodies, steps, threads, repetitions, and dt are required");
    }
    return options;
}

double quantile(const std::vector<double>& sorted, double probability) {
    const double position = probability * static_cast<double>(sorted.size() - 1U);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

double relative_change(double value, double reference) {
    return std::abs(value - reference) /
        std::max({std::abs(value), std::abs(reference), 1.0});
}

double checksum(const nbody::SimulationState& state) {
    double value = state.time_s;
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        const auto& body = state.bodies[index];
        const double weight = static_cast<double>(index + 1U);
        value += weight * (body.position_m.x + 3.0 * body.position_m.y +
            5.0 * body.position_m.z + body.velocity_m_s.x +
            7.0 * body.velocity_m_s.y + 11.0 * body.velocity_m_s.z);
    }
    return value;
}

bool finite_state(const nbody::SimulationState& state) {
    for (const auto& body : state.bodies) {
        if (!std::isfinite(body.position_m.x) || !std::isfinite(body.position_m.y) ||
            !std::isfinite(body.position_m.z) || !std::isfinite(body.velocity_m_s.x) ||
            !std::isfinite(body.velocity_m_s.y) || !std::isfinite(body.velocity_m_s.z)) {
            return false;
        }
    }
    return std::isfinite(state.time_s);
}

RunResult run_once(
    const nbody::SimulationState& initial,
    nbody::MpiForceBackend& backend,
    const Options& options) {
    nbody::SimulationState state = initial;
    backend.reset_timing();
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (std::size_t step = 0; step < options.steps; ++step) {
        nbody::velocity_verlet_step(state, options.dt_s, backend);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_elapsed = MPI_Wtime() - start;
    const auto local_timing = backend.timing();
    double maximum_elapsed = 0.0;
    double maximum_compute = 0.0;
    double maximum_communication = 0.0;
    MPI_Allreduce(&local_elapsed, &maximum_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_timing.local_compute_s, &maximum_compute, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_timing.communication_s, &maximum_communication, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    return {maximum_elapsed, maximum_compute, maximum_communication, std::move(state)};
}

std::string mpi_library_version() {
    char value[MPI_MAX_LIBRARY_VERSION_STRING]{};
    int length = 0;
    MPI_Get_library_version(value, &length);
    std::string result(value, static_cast<std::size_t>(length));
    std::replace(result.begin(), result.end(), ',', ';');
    std::replace(result.begin(), result.end(), '\n', ' ');
    std::replace(result.begin(), result.end(), '\r', ' ');
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level) != MPI_SUCCESS) {
        std::cerr << "error: MPI_Init_thread failed\n";
        return 1;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        const Options options = parse_options(argc, argv);
        if (provided_thread_level < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI runtime did not provide MPI_THREAD_FUNNELED");
        }
        nbody::MpiForceBackend backend(MPI_COMM_WORLD, options.threads);
        const auto initial = nbody::make_seeded_synthetic_system(options.bodies, options.seed);
        const auto initial_diagnostics = nbody::compute_diagnostics(initial);
        for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
            (void)run_once(initial, backend, options);
        }

        std::vector<double> elapsed;
        std::vector<double> computation;
        std::vector<double> communication;
        nbody::SimulationState final_state;
        for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
            auto result = run_once(initial, backend, options);
            elapsed.push_back(result.elapsed_s);
            computation.push_back(result.local_compute_s);
            communication.push_back(result.communication_s);
            final_state = std::move(result.state);
        }
        std::sort(elapsed.begin(), elapsed.end());
        std::sort(computation.begin(), computation.end());
        std::sort(communication.begin(), communication.end());

        const auto final_diagnostics = nbody::compute_diagnostics(final_state);
        const double energy_drift = relative_change(
            final_diagnostics.total_energy_j(), initial_diagnostics.total_energy_j());
        const double angular_drift = nbody::norm(
            final_diagnostics.angular_momentum_kg_m2_s -
            initial_diagnostics.angular_momentum_kg_m2_s) /
            std::max(nbody::norm(initial_diagnostics.angular_momentum_kg_m2_s), 1.0);
        const bool valid = finite_state(final_state) && energy_drift <= 5.0e-6 &&
            angular_drift <= 5.0e-10;

        if (rank == 0) {
            std::cout << std::setprecision(17)
                << "backend,ranks,threads_per_rank,bodies,steps,dt_s,warmups,repetitions,"
                   "median_s,median_per_step_s,median_local_compute_s,"
                   "median_communication_s,relative_energy_drift,"
                   "relative_angular_momentum_drift,correctness_status,checksum,"
                   "mpi_thread_level,mpi_library,compiler,build_configuration\n"
                << backend.name() << ',' << backend.rank_count() << ',' << options.threads << ','
                << options.bodies << ',' << options.steps << ',' << options.dt_s << ','
                << options.warmups << ',' << options.repetitions << ','
                << quantile(elapsed, 0.5) << ','
                << quantile(elapsed, 0.5) / static_cast<double>(options.steps) << ','
                << quantile(computation, 0.5) << ',' << quantile(communication, 0.5) << ','
                << energy_drift << ',' << angular_drift << ',' << (valid ? "pass" : "fail")
                << ',' << checksum(final_state) << ',' << provided_thread_level << ",\""
                << mpi_library_version() << "\",\"GNU " << __GNUC__ << '.' << __GNUC_MINOR__
                << '.' << __GNUC_PATCHLEVEL__ << "\",\"" << NBODY_BUILD_CONFIGURATION
                << "\"\n";
        }
        MPI_Finalize();
        return valid ? 0 : 2;
    } catch (const std::exception& error) {
        if (rank == 0) {
            std::cerr << "error: " << error.what() << '\n';
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
