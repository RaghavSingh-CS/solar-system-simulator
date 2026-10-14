#include "nbody/visualization/controller.hpp"

#ifdef NBODY_VIS_HAS_OPENMP
#include "nbody/openmp_backend.hpp"
#endif
#ifdef NBODY_VIS_HAS_CUDA
#include "nbody/cuda_backend.hpp"
#endif

#include <cmath>
#include <stdexcept>
#include <utility>

namespace nbody::visualization {

class SimulationController::CudaExecutor {
public:
#ifdef NBODY_VIS_HAS_CUDA
    CudaExecutor(std::size_t body_count, int device_index)
        : integrator(body_count, device_index) {}

    CudaVelocityVerlet integrator;
#endif
};

namespace {

enum class AdvancementPhase {
    candidate_copy,
    validation,
    old_force,
    position_drift,
    new_force,
    velocity_kick,
    cuda_batch,
    commit,
};

const char* advancement_phase_name(AdvancementPhase phase) noexcept {
    switch (phase) {
    case AdvancementPhase::candidate_copy: return "candidate-copy";
    case AdvancementPhase::validation: return "validation";
    case AdvancementPhase::old_force: return "old-force";
    case AdvancementPhase::position_drift: return "position-drift";
    case AdvancementPhase::new_force: return "new-force";
    case AdvancementPhase::velocity_kick: return "velocity-kick";
    case AdvancementPhase::cuda_batch: return "cuda-batch";
    case AdvancementPhase::commit: return "commit";
    }
    return "unknown";
}

class PhaseTrackingForceBackend final : public ForceBackend {
public:
    PhaseTrackingForceBackend(
        const ForceBackend& delegate,
        AdvancementPhase& phase) noexcept
        : delegate_(delegate), phase_(phase) {}

    const char* name() const noexcept override { return delegate_.name(); }

    std::vector<Vec3> compute_accelerations(
        const SimulationState& state,
        double gravitational_constant_value) const override {
        if (force_calls_ == 0U) {
            phase_ = AdvancementPhase::old_force;
        } else if (force_calls_ == 1U) {
            phase_ = AdvancementPhase::new_force;
        } else {
            phase_ = AdvancementPhase::validation;
        }
        auto accelerations = delegate_.compute_accelerations(
            state, gravitational_constant_value);
        ++force_calls_;
        phase_ = force_calls_ == 1U
            ? AdvancementPhase::position_drift
            : AdvancementPhase::velocity_kick;
        return accelerations;
    }

private:
    const ForceBackend& delegate_;
    AdvancementPhase& phase_;
    mutable std::size_t force_calls_{0};
};

void validate_state(const SimulationState& state) {
    if (state.bodies.size() < 2) {
        throw std::invalid_argument("visualization scenario requires at least two bodies");
    }
    for (const auto& body : state.bodies) {
        const auto finite_vector = [](const Vec3& value) {
            return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
        };
        if (body.name.empty() || !(body.mass_kg > 0.0) || !std::isfinite(body.mass_kg) ||
            !finite_vector(body.position_m) || !finite_vector(body.velocity_m_s)) {
            throw std::invalid_argument("visualization scenario contains an invalid body");
        }
    }
}

}  // namespace

SimulationController::SimulationController(
    SimulationState initial_state,
    ControllerConfig config)
    : initial_state_(std::move(initial_state)), state_(initial_state_), config_(config) {
    validate_state(initial_state_);
    validate_config();
    configure_execution(
        config_.backend,
        config_.solver,
        config_.openmp_threads,
        config_.cuda_device,
        config_.barnes_hut);
}

SimulationController::~SimulationController() = default;
SimulationController::SimulationController(SimulationController&&) noexcept = default;
SimulationController& SimulationController::operator=(SimulationController&&) noexcept = default;

const SimulationState& SimulationController::state() const noexcept { return state_; }
const SimulationState& SimulationController::initial_state() const noexcept { return initial_state_; }
double SimulationController::dt_s() const noexcept { return config_.dt_s; }
std::size_t SimulationController::steps_per_frame() const noexcept {
    return config_.steps_per_frame;
}
bool SimulationController::paused() const noexcept { return paused_; }
BackendKind SimulationController::backend_kind() const noexcept { return config_.backend; }
const char* SimulationController::backend_name() const noexcept {
    switch (config_.backend) {
    case BackendKind::sequential:
        return "sequential";
    case BackendKind::openmp:
        return "openmp";
    case BackendKind::cuda:
        return "cuda";
    }
    return "unknown";
}
SolverKind SimulationController::solver_kind() const noexcept { return config_.solver; }
const char* SimulationController::solver_name() const noexcept {
    switch (config_.solver) {
    case SolverKind::direct:
        return "direct";
    case SolverKind::barnes_hut:
        return "barnes-hut";
    }
    return "unknown";
}
int SimulationController::openmp_threads() const noexcept { return config_.openmp_threads; }
int SimulationController::cuda_device() const noexcept { return config_.cuda_device; }
const BarnesHutConfiguration&
SimulationController::barnes_hut_configuration() const noexcept {
    return config_.barnes_hut;
}
std::optional<BarnesHutStatistics>
SimulationController::barnes_hut_statistics() const noexcept {
    return last_barnes_hut_statistics_;
}
const std::string& SimulationController::last_error() const noexcept {
    return last_error_;
}

void SimulationController::set_paused(bool paused_value) noexcept { paused_ = paused_value; }
void SimulationController::toggle_paused() noexcept { paused_ = !paused_; }

void SimulationController::set_steps_per_frame(std::size_t steps) {
    if (steps == 0) {
        throw std::invalid_argument("steps per frame must be positive; use pause to stop advancement");
    }
    config_.steps_per_frame = steps;
}

void SimulationController::select_backend(
    BackendKind backend,
    int openmp_threads,
    int cuda_device) {
    configure_execution(
        backend,
        config_.solver,
        openmp_threads,
        cuda_device,
        config_.barnes_hut);
}

void SimulationController::select_solver(
    SolverKind solver,
    std::optional<BarnesHutConfiguration> barnes_hut) {
    configure_execution(
        config_.backend,
        solver,
        config_.openmp_threads,
        config_.cuda_device,
        barnes_hut.value_or(config_.barnes_hut));
}

void SimulationController::replace_scenario(
    SimulationState initial_state,
    double dt_s,
    std::size_t steps_per_frame) {
    validate_state(initial_state);
    if (!(dt_s > 0.0) || !std::isfinite(dt_s) || steps_per_frame == 0) {
        throw std::invalid_argument("replacement scenario requires positive finite dt and steps");
    }

#ifdef NBODY_VIS_HAS_CUDA
    std::unique_ptr<CudaExecutor> replacement_cuda;
    if (config_.backend == BackendKind::cuda &&
        initial_state.bodies.size() != state_.bodies.size()) {
        replacement_cuda = std::make_unique<CudaExecutor>(
            initial_state.bodies.size(), config_.cuda_device);
    }
#endif

    initial_state_ = std::move(initial_state);
    state_ = initial_state_;
    config_.dt_s = dt_s;
    config_.steps_per_frame = steps_per_frame;
    last_barnes_hut_statistics_.reset();
    last_error_.clear();
#ifdef NBODY_VIS_HAS_CUDA
    if (replacement_cuda) {
        cuda_integrator_ = std::move(replacement_cuda);
    }
#endif
}

void SimulationController::advance_frame() {
    if (paused_) {
        return;
    }
    advance_steps(config_.steps_per_frame);
}

void SimulationController::advance_steps(std::size_t steps) {
    if (steps == 0) {
        throw std::invalid_argument("manual advancement requires at least one step");
    }
    AdvancementPhase phase = AdvancementPhase::validation;
    std::size_t active_step = 0U;
    try {
        if (config_.backend == BackendKind::cuda) {
#ifdef NBODY_VIS_HAS_CUDA
            phase = AdvancementPhase::cuda_batch;
            if (!cuda_integrator_) {
                throw std::logic_error("CUDA backend is active without an integration executor");
            }
            auto result = cuda_integrator_->integrator.run(state_, steps, config_.dt_s);
            state_ = std::move(result.final_state);
            last_error_.clear();
            return;
#else
            throw std::logic_error("CUDA backend is active in a non-CUDA visualization build");
#endif
        }
        phase = AdvancementPhase::candidate_copy;
        SimulationState candidate = state_;
        for (std::size_t step = 0; step < steps; ++step) {
            active_step = step + 1U;
            phase = AdvancementPhase::validation;
            PhaseTrackingForceBackend tracked_backend(*backend_, phase);
            velocity_verlet_step(candidate, config_.dt_s, tracked_backend);
        }
        phase = AdvancementPhase::commit;
        state_ = std::move(candidate);
        if (config_.solver == SolverKind::barnes_hut) {
            if (const auto* sequential =
                    dynamic_cast<const SequentialBarnesHutForceBackend*>(backend_)) {
                last_barnes_hut_statistics_ = sequential->last_statistics();
            } else if (const auto* openmp =
                           dynamic_cast<const OpenMpBarnesHutForceBackend*>(backend_)) {
                last_barnes_hut_statistics_ = openmp->last_statistics();
            }
        } else {
            last_barnes_hut_statistics_.reset();
        }
        last_error_.clear();
    } catch (const std::exception& error) {
        paused_ = true;
        last_error_ = "backend=" + std::string(backend_name()) +
            " solver=" + solver_name() +
            " step=" + std::to_string(active_step) + "/" +
            std::to_string(steps) +
            " phase=" + advancement_phase_name(phase) + ": " + error.what();
        throw;
    } catch (...) {
        paused_ = true;
        last_error_ = "backend=" + std::string(backend_name()) +
            " solver=" + solver_name() +
            " step=" + std::to_string(active_step) + "/" +
            std::to_string(steps) +
            " phase=" + advancement_phase_name(phase) +
            ": non-standard simulation advancement failure";
        throw;
    }
}

void SimulationController::reset() {
    state_ = initial_state_;
    last_barnes_hut_statistics_.reset();
    last_error_.clear();
}

bool SimulationController::openmp_available() noexcept {
#ifdef NBODY_VIS_HAS_OPENMP
    return true;
#else
    return false;
#endif
}

bool SimulationController::cuda_compiled() noexcept {
#ifdef NBODY_VIS_HAS_CUDA
    return true;
#else
    return false;
#endif
}

bool SimulationController::cuda_available(int device_index) noexcept {
#ifdef NBODY_VIS_HAS_CUDA
    if (device_index < 0) {
        return false;
    }
    try {
        return cuda_device_count() > device_index;
    } catch (...) {
        return false;
    }
#else
    (void)device_index;
    return false;
#endif
}

void SimulationController::validate_config() const {
    if (!(config_.dt_s > 0.0) || !std::isfinite(config_.dt_s)) {
        throw std::invalid_argument("visualization timestep must be finite and positive");
    }
    if (config_.steps_per_frame == 0) {
        throw std::invalid_argument("initial steps per frame must be positive");
    }
    if (config_.openmp_threads <= 0) {
        throw std::invalid_argument("OpenMP thread count must be positive");
    }
    if (config_.cuda_device < 0) {
        throw std::invalid_argument("CUDA device index must be non-negative");
    }
    if (!(config_.barnes_hut.theta > 0.0) ||
        config_.barnes_hut.theta > 2.0 ||
        !std::isfinite(config_.barnes_hut.theta)) {
        throw std::invalid_argument(
            "Barnes-Hut theta must be finite and in (0, 2]");
    }
}

void SimulationController::configure_execution(
    BackendKind backend,
    SolverKind solver,
    int openmp_threads,
    int cuda_device,
    const BarnesHutConfiguration& barnes_hut) {
    if (solver == SolverKind::barnes_hut &&
        (!(barnes_hut.theta > 0.0) || barnes_hut.theta > 2.0 ||
         !std::isfinite(barnes_hut.theta))) {
        throw std::invalid_argument(
            "Barnes-Hut theta must be finite and in (0, 2]");
    }
    if (backend == BackendKind::cuda && solver == SolverKind::barnes_hut) {
        throw std::runtime_error(
            "CUDA Barnes-Hut is unavailable in the viewer pending NVIDIA validation");
    }

    if (backend == BackendKind::sequential) {
        std::unique_ptr<ForceBackend> replacement;
        const ForceBackend* replacement_backend = &sequential_force_backend();
        if (solver == SolverKind::barnes_hut) {
            replacement =
                std::make_unique<SequentialBarnesHutForceBackend>(barnes_hut);
            replacement_backend = replacement.get();
        }
        owned_backend_ = std::move(replacement);
        cuda_integrator_.reset();
        backend_ = replacement_backend;
        config_.backend = backend;
        config_.solver = solver;
        config_.barnes_hut = barnes_hut;
        config_.openmp_threads = 1;
        config_.cuda_device = 0;
        last_barnes_hut_statistics_.reset();
        last_error_.clear();
        return;
    }
    if (backend == BackendKind::openmp) {
        if (openmp_threads <= 0) {
            throw std::invalid_argument("OpenMP thread count must be positive");
        }
#ifdef NBODY_VIS_HAS_OPENMP
        std::unique_ptr<ForceBackend> replacement;
        if (solver == SolverKind::direct) {
            replacement = std::make_unique<OpenMpForceBackend>(openmp_threads);
        } else {
            replacement = std::make_unique<OpenMpBarnesHutForceBackend>(
                openmp_threads, barnes_hut);
        }
        owned_backend_ = std::move(replacement);
        cuda_integrator_.reset();
        backend_ = owned_backend_.get();
        config_.backend = backend;
        config_.solver = solver;
        config_.barnes_hut = barnes_hut;
        config_.openmp_threads = openmp_threads;
        config_.cuda_device = 0;
        last_barnes_hut_statistics_.reset();
        last_error_.clear();
        return;
#else
        throw std::runtime_error(
            "OpenMP is unavailable in this visualization build; configure with "
            "-DNBODY_ENABLE_OPENMP=ON");
#endif
    }

    if (cuda_device < 0) {
        throw std::invalid_argument("CUDA device index must be non-negative");
    }
#ifdef NBODY_VIS_HAS_CUDA
    auto replacement = std::make_unique<CudaExecutor>(state_.bodies.size(), cuda_device);
    cuda_integrator_ = std::move(replacement);
    owned_backend_.reset();
    backend_ = nullptr;
    config_.backend = backend;
    config_.solver = SolverKind::direct;
    config_.barnes_hut = barnes_hut;
    config_.openmp_threads = 1;
    config_.cuda_device = cuda_device;
    last_barnes_hut_statistics_.reset();
    last_error_.clear();
#else
    (void)cuda_device;
    throw std::runtime_error(
        "CUDA is unavailable in this visualization build; configure with "
        "-DNBODY_ENABLE_CUDA=ON");
#endif
}

}  // namespace nbody::visualization
