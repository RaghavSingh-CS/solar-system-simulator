#pragma once

#include "nbody/simulation.hpp"

#include <mpi.h>

#include <cstddef>

namespace nbody {

struct MpiPartition {
    std::size_t begin{0};
    std::size_t end{0};

    std::size_t size() const noexcept { return end - begin; }
};

MpiPartition contiguous_mpi_partition(
    std::size_t body_count,
    int rank,
    int rank_count);

struct MpiTiming {
    double local_compute_s{0.0};
    double communication_s{0.0};
    std::size_t force_evaluations{0};
};

class MpiForceBackend final : public ForceBackend {
public:
    explicit MpiForceBackend(MPI_Comm communicator = MPI_COMM_WORLD, int openmp_threads = 1);
    ~MpiForceBackend() override;
    MpiForceBackend(const MpiForceBackend&) = delete;
    MpiForceBackend& operator=(const MpiForceBackend&) = delete;

    const char* name() const noexcept override;
    std::vector<Vec3> compute_accelerations(
        const SimulationState& state,
        double gravitational_constant_value) const override;

    int rank() const noexcept;
    int rank_count() const noexcept;
    int openmp_threads() const noexcept;
    MpiTiming timing() const noexcept;
    void reset_timing() const noexcept;

private:
    MPI_Comm communicator_{MPI_COMM_NULL};
    int rank_{0};
    int rank_count_{1};
    int openmp_threads_{1};
    mutable MpiTiming timing_;
};

}  // namespace nbody
