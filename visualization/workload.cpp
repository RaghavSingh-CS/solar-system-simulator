#include "nbody/visualization/workload.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace nbody::visualization {

bool is_supported_compute_body_count(std::size_t body_count) noexcept {
    return std::find(compute_body_counts.begin(), compute_body_counts.end(), body_count) !=
        compute_body_counts.end();
}

bool render_resource_admitted(std::size_t body_count) noexcept {
    return body_count > 0U && body_count <= maximum_rendered_synthetic_bodies;
}

std::size_t checked_render_upload_bytes(
    std::size_t element_count,
    std::size_t element_size) {
    if (element_size != 0U &&
        element_count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("render upload byte count overflow");
    }
    return element_count * element_size;
}

bool render_upload_allows_draw(unsigned int gl_error_code) noexcept {
    return gl_error_code == 0U;
}

InteractiveWorkloadPolicy interactive_workload_policy(
    std::size_t body_count,
    WorkloadBackend backend,
    WorkloadSolver solver) noexcept {
    if (!is_supported_compute_body_count(body_count)) {
        return {false, false, false, false, false, true, 1,
            "Unsupported synthetic body count; choose 256, 1024, 4096, or 8192."};
    }
    if (backend == WorkloadBackend::cuda && solver == WorkloadSolver::barnes_hut) {
        return {false, false, false, false, false, true, 1,
            "CUDA Barnes-Hut viewer execution is unavailable pending NVIDIA validation and explicit integration."};
    }
    if (body_count <= 1024) {
        return {true, true, true, true, true, false, 32,
            "Continuous execution is admitted for this bounded synthetic scene."};
    }
    if (body_count == 4096) {
        const bool admitted = solver == WorkloadSolver::barnes_hut &&
            backend != WorkloadBackend::cuda;
        return {false, admitted, false, false, false, true, 1,
            admitted
                ? "N=4096 is paused; one manual Sequential/OpenMP Barnes-Hut step is admitted."
                : "N=4096 requires Sequential or OpenMP Barnes-Hut for a bounded manual step; Direct and CUDA execution are rejected."};
    }
    const bool admitted = body_count == 8192 &&
        solver == WorkloadSolver::barnes_hut && backend == WorkloadBackend::openmp;
    return {false, admitted, false, false, false, true, 1,
        admitted
            ? "N=8192 is paused; one manual OpenMP Barnes-Hut step is admitted."
            : "N=8192 requires OpenMP Barnes-Hut for a bounded manual step; Direct, Sequential Barnes-Hut, and CUDA execution are rejected."};
}

}  // namespace nbody::visualization
