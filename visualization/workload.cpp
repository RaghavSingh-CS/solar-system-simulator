#include "nbody/visualization/workload.hpp"

#include <algorithm>

namespace nbody::visualization {

bool is_supported_compute_body_count(std::size_t body_count) noexcept {
    return std::find(compute_body_counts.begin(), compute_body_counts.end(), body_count) !=
        compute_body_counts.end();
}

InteractiveWorkloadPolicy interactive_workload_policy(
    std::size_t body_count,
    bool openmp_backend) noexcept {
    if (body_count <= 1024) {
        return {true, true, true, true, true, false, 32};
    }
    if (body_count <= 2048) {
        return {false, true, false, false, false, true, 1};
    }
    if (body_count <= 8192) {
        return {false, openmp_backend, false, false, false, true, 1};
    }
    return {false, false, false, false, false, true, 1};
}

}  // namespace nbody::visualization
