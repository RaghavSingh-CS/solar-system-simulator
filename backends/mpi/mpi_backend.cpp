#include "nbody/mpi_backend.hpp"

#include "nbody/direct_force.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nbody {

namespace {

void check_mpi(int status, const char* operation) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(
        std::string(operation) + " failed: " + std::string(message, static_cast<std::size_t>(length)));
}

double elapsed_seconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& stop) {
    return std::chrono::duration<double>(stop - start).count();
}

}  // namespace

MpiPartition contiguous_mpi_partition(
    std::size_t body_count,
    int rank,
    int rank_count) {
    if (rank_count <= 0 || rank < 0 || rank >= rank_count) {
        throw std::invalid_argument("MPI rank and rank count are invalid");
    }
    const std::size_t ranks = static_cast<std::size_t>(rank_count);
    const std::size_t rank_index = static_cast<std::size_t>(rank);
    const std::size_t base = body_count / ranks;
    const std::size_t remainder = body_count % ranks;
    const std::size_t begin = rank_index * base + std::min(rank_index, remainder);
    const std::size_t count = base + (rank_index < remainder ? 1U : 0U);
    return {begin, begin + count};
}

MpiForceBackend::MpiForceBackend(MPI_Comm communicator, int openmp_threads)
    : openmp_threads_(openmp_threads) {
    int initialized = 0;
    check_mpi(MPI_Initialized(&initialized), "MPI_Initialized");
    if (initialized == 0) {
        throw std::runtime_error("MPI must be initialized before constructing MpiForceBackend");
    }
    if (communicator == MPI_COMM_NULL || openmp_threads_ <= 0) {
        throw std::invalid_argument("MPI communicator and OpenMP thread count must be valid");
    }
#ifndef NBODY_MPI_HAS_OPENMP
    if (openmp_threads_ != 1) {
        throw std::runtime_error(
            "hybrid MPI+OpenMP requires configuring with NBODY_ENABLE_OPENMP=ON");
    }
#endif
    check_mpi(MPI_Comm_dup(communicator, &communicator_), "MPI_Comm_dup");
    check_mpi(MPI_Comm_set_errhandler(communicator_, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
    check_mpi(MPI_Comm_rank(communicator_, &rank_), "MPI_Comm_rank");
    check_mpi(MPI_Comm_size(communicator_, &rank_count_), "MPI_Comm_size");
}

MpiForceBackend::~MpiForceBackend() {
    int finalized = 0;
    if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0 &&
        communicator_ != MPI_COMM_NULL) {
        MPI_Comm_free(&communicator_);
    }
}

const char* MpiForceBackend::name() const noexcept {
    return openmp_threads_ > 1 ? "mpi+openmp" : "mpi";
}

std::vector<Vec3> MpiForceBackend::compute_accelerations(
    const SimulationState& state,
    double gravitational_constant_value) const {
    validate_direct_force_inputs(state, gravitational_constant_value);
    if (state.bodies.size() > static_cast<std::size_t>(INT_MAX / 3)) {
        throw std::length_error("body count exceeds MPI_Allgatherv count range");
    }

    const MpiPartition owned = contiguous_mpi_partition(
        state.bodies.size(), rank_, rank_count_);
    std::vector<double> local(owned.size() * 3U);
    int invalid_local = 0;
    const auto compute_start = std::chrono::steady_clock::now();
#ifdef NBODY_MPI_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(openmp_threads_) reduction(| : invalid_local)
#endif
    for (long long offset = 0; offset < static_cast<long long>(owned.size()); ++offset) {
        Vec3 acceleration;
        const std::size_t target = owned.begin + static_cast<std::size_t>(offset);
        if (!try_compute_direct_acceleration(
                state, target, gravitational_constant_value, acceleration)) {
            invalid_local = 1;
        }
        const std::size_t packed = static_cast<std::size_t>(offset) * 3U;
        local[packed] = acceleration.x;
        local[packed + 1U] = acceleration.y;
        local[packed + 2U] = acceleration.z;
    }
    const auto compute_stop = std::chrono::steady_clock::now();

    const auto communication_start = std::chrono::steady_clock::now();
    int invalid_global = 0;
    check_mpi(
        MPI_Allreduce(&invalid_local, &invalid_global, 1, MPI_INT, MPI_MAX, communicator_),
        "MPI_Allreduce invalid-position status");
    if (invalid_global != 0) {
        throw std::domain_error("coincident or invalid body positions");
    }

    std::vector<int> counts(static_cast<std::size_t>(rank_count_));
    std::vector<int> displacements(static_cast<std::size_t>(rank_count_));
    for (int rank = 0; rank < rank_count_; ++rank) {
        const MpiPartition partition = contiguous_mpi_partition(
            state.bodies.size(), rank, rank_count_);
        counts[static_cast<std::size_t>(rank)] = static_cast<int>(partition.size() * 3U);
        displacements[static_cast<std::size_t>(rank)] = static_cast<int>(partition.begin * 3U);
    }
    std::vector<double> packed_accelerations(state.bodies.size() * 3U);
    check_mpi(
        MPI_Allgatherv(
            local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
            packed_accelerations.data(), counts.data(), displacements.data(), MPI_DOUBLE,
            communicator_),
        "MPI_Allgatherv accelerations");
    const auto communication_stop = std::chrono::steady_clock::now();

    std::vector<Vec3> accelerations(state.bodies.size());
    for (std::size_t index = 0; index < accelerations.size(); ++index) {
        accelerations[index] = {
            packed_accelerations[index * 3U],
            packed_accelerations[index * 3U + 1U],
            packed_accelerations[index * 3U + 2U],
        };
    }
    timing_.local_compute_s += elapsed_seconds(compute_start, compute_stop);
    timing_.communication_s += elapsed_seconds(communication_start, communication_stop);
    ++timing_.force_evaluations;
    return accelerations;
}

int MpiForceBackend::rank() const noexcept { return rank_; }
int MpiForceBackend::rank_count() const noexcept { return rank_count_; }
int MpiForceBackend::openmp_threads() const noexcept { return openmp_threads_; }
MpiTiming MpiForceBackend::timing() const noexcept { return timing_; }
void MpiForceBackend::reset_timing() const noexcept { timing_ = {}; }

}  // namespace nbody
