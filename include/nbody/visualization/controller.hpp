#pragma once

#include "nbody/barnes_hut.hpp"
#include "nbody/simulation.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace nbody {
namespace visualization {

enum class BackendKind {
    sequential,
    openmp,
    cuda,
};

enum class SolverKind {
    direct,
    barnes_hut,
};

struct ControllerConfig {
    double dt_s{10.0};
    std::size_t steps_per_frame{256};
    BackendKind backend{BackendKind::sequential};
    int openmp_threads{1};
    int cuda_device{0};
    SolverKind solver{SolverKind::direct};
    BarnesHutConfiguration barnes_hut{};
};

// Published only after an advancement commits. Selection alone is not execution.
struct ControllerExecution {
    BackendKind backend;
    SolverKind solver;
    std::size_t steps;
    double end_time_s;
    int openmp_threads;
    int cuda_device;
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
    SolverKind solver_kind() const noexcept;
    const char* solver_name() const noexcept;
    int openmp_threads() const noexcept;
    int cuda_device() const noexcept;
    const BarnesHutConfiguration& barnes_hut_configuration() const noexcept;
    std::optional<BarnesHutStatistics> barnes_hut_statistics() const noexcept;
    const std::string& last_error() const noexcept;
    const std::optional<ControllerExecution>& last_execution() const noexcept;

    void set_paused(bool paused) noexcept;
    void toggle_paused() noexcept;
    void set_steps_per_frame(std::size_t steps);
    void select_backend(
        BackendKind backend,
        int openmp_threads = 1,
        int cuda_device = 0);
    void select_solver(
        SolverKind solver,
        std::optional<BarnesHutConfiguration> barnes_hut = std::nullopt);
    void replace_scenario(
        SimulationState initial_state,
        double dt_s,
        std::size_t steps_per_frame);
    void advance_frame();
    void advance_steps(std::size_t steps);
    void reset();

    static bool openmp_available() noexcept;
    static bool cuda_compiled() noexcept;
    // UI/convenience probe only: false also covers enumeration errors because
    // this API is noexcept. Mandatory GPU tests use strict cuda_device_count()
    // enumeration so runtime/driver errors fail rather than skip.
    static bool cuda_available(int device_index = 0) noexcept;

private:
    class CudaExecutor;

    void validate_config() const;
    void configure_execution(
        BackendKind backend,
        SolverKind solver,
        int openmp_threads,
        int cuda_device,
        const BarnesHutConfiguration& barnes_hut);

    SimulationState initial_state_;
    SimulationState state_;
    ControllerConfig config_;
    bool paused_{false};
    const ForceBackend* backend_{nullptr};
    std::unique_ptr<ForceBackend> owned_backend_;
    std::unique_ptr<CudaExecutor> cuda_integrator_;
    std::optional<BarnesHutStatistics> last_barnes_hut_statistics_;
    std::string last_error_;
    std::optional<ControllerExecution> last_execution_;
};

}  // namespace visualization
}  // namespace nbody
