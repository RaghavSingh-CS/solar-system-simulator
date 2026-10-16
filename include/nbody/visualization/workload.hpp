#pragma once

#include <array>
#include <cstddef>

namespace nbody::visualization {

inline constexpr std::array<std::size_t, 4> compute_body_counts{{
    256, 1024, 4096, 8192,
}};

inline constexpr std::size_t maximum_rendered_synthetic_bodies = 8192;

struct InteractiveWorkloadPolicy {
    bool continuous_advancement{true};
    bool manual_single_step{true};
    bool history_trails{true};
    bool live_cpu_comparison{true};
    bool live_conservation{true};
    bool warning_required{false};
    std::size_t maximum_steps_per_frame{32};
    const char* explanation{""};
};

enum class WorkloadBackend {
    sequential,
    openmp,
    cuda,
};

enum class WorkloadSolver {
    direct,
    barnes_hut,
};

bool is_supported_compute_body_count(std::size_t body_count) noexcept;

bool render_resource_admitted(std::size_t body_count) noexcept;

std::size_t checked_render_upload_bytes(
    std::size_t element_count,
    std::size_t element_size);

bool render_upload_allows_draw(unsigned int gl_error_code) noexcept;

InteractiveWorkloadPolicy interactive_workload_policy(
    std::size_t body_count,
    WorkloadBackend backend,
    WorkloadSolver solver) noexcept;

}  // namespace nbody::visualization
