#pragma once

#include <array>
#include <cstddef>

namespace nbody::visualization {

inline constexpr std::array<std::size_t, 13> compute_body_counts{{
    32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 100000,
}};

struct InteractiveWorkloadPolicy {
    bool continuous_advancement{true};
    bool manual_single_step{true};
    bool history_trails{true};
    bool live_cpu_comparison{true};
    bool live_conservation{true};
    bool warning_required{false};
    std::size_t maximum_steps_per_frame{32};
};

enum class WorkloadBackend {
    sequential,
    openmp,
    cuda,
};

bool is_supported_compute_body_count(std::size_t body_count) noexcept;

InteractiveWorkloadPolicy interactive_workload_policy(
    std::size_t body_count,
    WorkloadBackend backend) noexcept;

}  // namespace nbody::visualization
