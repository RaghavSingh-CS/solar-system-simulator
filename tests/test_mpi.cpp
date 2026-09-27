#include "nbody/diagnostics.hpp"
#include "nbody/mpi_backend.hpp"
#include "nbody/scenarios.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double scaled_difference(const nbody::Vec3& left, const nbody::Vec3& right) {
    return nbody::norm(left - right) / std::max(nbody::norm(right), 1.0);
}

void require_close(
    const nbody::SimulationState& distributed,
    const nbody::SimulationState& reference) {
    require(distributed.bodies.size() == reference.bodies.size(), "MPI body count differs");
    for (std::size_t index = 0; index < reference.bodies.size(); ++index) {
        const auto& actual = distributed.bodies[index];
        const auto& expected = reference.bodies[index];
        const double position_absolute = nbody::norm(actual.position_m - expected.position_m);
        const double velocity_absolute = nbody::norm(actual.velocity_m_s - expected.velocity_m_s);
        require(position_absolute <= 1.0e-6 ||
                    scaled_difference(actual.position_m, expected.position_m) <= 1.0e-13,
                "MPI position differs from Sequential");
        require(velocity_absolute <= 1.0e-9 ||
                    scaled_difference(actual.velocity_m_s, expected.velocity_m_s) <= 1.0e-13,
                "MPI velocity differs from Sequential");
    }
}

void run_validation(int openmp_threads) {
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    constexpr std::size_t body_count = 11;
    std::size_t covered = 0;
    std::size_t expected_begin = 0;
    for (int owner = 0; owner < rank_count; ++owner) {
        const auto partition = nbody::contiguous_mpi_partition(body_count, owner, rank_count);
        require(partition.begin == expected_begin && partition.end >= partition.begin,
                "MPI partition is not contiguous");
        expected_begin = partition.end;
        covered += partition.size();
    }
    require(covered == body_count && expected_begin == body_count,
            "MPI partition does not cover a non-divisible workload exactly");

    const auto initial = nbody::make_seeded_synthetic_system(body_count, 24301);
    auto reference = initial;
    auto distributed = initial;
    auto repeated = initial;
    nbody::MpiForceBackend backend(MPI_COMM_WORLD, openmp_threads);
    for (int step = 0; step < 4; ++step) {
        nbody::velocity_verlet_step(reference, 10.0);
        nbody::velocity_verlet_step(distributed, 10.0, backend);
    }
    backend.reset_timing();
    for (int step = 0; step < 4; ++step) {
        nbody::velocity_verlet_step(repeated, 10.0, backend);
    }
    require_close(distributed, reference);
    require_close(repeated, reference);

    const auto initial_diagnostics = nbody::compute_diagnostics(initial);
    const auto final_diagnostics = nbody::compute_diagnostics(distributed);
    const double energy_drift = std::abs(
        final_diagnostics.total_energy_j() - initial_diagnostics.total_energy_j()) /
        std::max({std::abs(final_diagnostics.total_energy_j()),
                  std::abs(initial_diagnostics.total_energy_j()), 1.0});
    const double angular_drift = nbody::norm(
        final_diagnostics.angular_momentum_kg_m2_s -
        initial_diagnostics.angular_momentum_kg_m2_s) /
        std::max(nbody::norm(initial_diagnostics.angular_momentum_kg_m2_s), 1.0);
    require(std::isfinite(energy_drift) && std::isfinite(angular_drift),
            "MPI diagnostics are non-finite");
    require(energy_drift <= 5.0e-6 && angular_drift <= 5.0e-10,
            "MPI conservation exceeded established gates");
    if (rank == 0) {
        std::cout << "MPI validation passed: ranks=" << rank_count
                  << " threads_per_rank=" << openmp_threads
                  << " N=" << body_count << " energy_drift=" << energy_drift
                  << " angular_drift=" << angular_drift << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI_THREAD_FUNNELED unavailable");
        }
        int threads = 1;
        if (argc == 3 && std::string(argv[1]) == "--threads") {
            threads = std::stoi(argv[2]);
        }
        run_validation(threads);
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        if (rank == 0) {
            std::cerr << "MPI validation failure: " << error.what() << '\n';
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
