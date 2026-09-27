#pragma once

#include "nbody/simulation.hpp"

#include <cstddef>
#include <memory>

namespace nbody::visualization {

enum class BackendKind {
    sequential,
    openmp,
};

struct ControllerConfig {
    double dt_s{10.0};
    std::size_t steps_per_frame{256};
    BackendKind backend{BackendKind::sequential};
    int openmp_threads{1};
};

class SimulationController {
public:
    SimulationController(SimulationState initial_state, ControllerConfig config);
    ~SimulationController();
    SimulationController(SimulationController&&) noexcept;
    SimulationController& operator=(SimulationController&&) noexcept;
    SimulationController(const SimulationController&) = delete;
    SimulationController& operator=(const SimulationController&) = delete;

    const SimulationState& state() const noexcept;
    const SimulationState& initial_state() const noexcept;
    double dt_s() const noexcept;
    std::size_t steps_per_frame() const noexcept;
    bool paused() const noexcept;
    BackendKind backend_kind() const noexcept;
    const char* backend_name() const noexcept;
    int openmp_threads() const noexcept;

    void set_paused(bool paused) noexcept;
    void toggle_paused() noexcept;
    void set_steps_per_frame(std::size_t steps);
    void select_backend(BackendKind backend, int openmp_threads = 1);
    void replace_scenario(
        SimulationState initial_state,
        double dt_s,
        std::size_t steps_per_frame);
    void advance_frame();
    void advance_steps(std::size_t steps);
    void reset();

    static bool openmp_available() noexcept;

private:
    void validate_config() const;
    void configure_backend(BackendKind backend, int openmp_threads);

    SimulationState initial_state_;
    SimulationState state_;
    ControllerConfig config_;
    bool paused_{false};
    const ForceBackend* backend_{nullptr};
    std::unique_ptr<ForceBackend> owned_backend_;
};

}  // namespace nbody::visualization
