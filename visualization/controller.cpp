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
    configure_backend(config_.backend, config_.openmp_threads, config_.cuda_device);
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
int SimulationController::openmp_threads() const noexcept { return config_.openmp_threads; }
int SimulationController::cuda_device() const noexcept { return config_.cuda_device; }

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
    configure_backend(backend, openmp_threads, cuda_device);
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
    paused_ = false;
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
    if (config_.backend == BackendKind::cuda) {
#ifdef NBODY_VIS_HAS_CUDA
        if (!cuda_integrator_) {
            throw std::logic_error("CUDA backend is active without an integration executor");
        }
        auto result = cuda_integrator_->integrator.run(state_, steps, config_.dt_s);
        state_ = std::move(result.final_state);
        return;
#else
        throw std::logic_error("CUDA backend is active in a non-CUDA visualization build");
#endif
    }
    for (std::size_t step = 0; step < steps; ++step) {
        velocity_verlet_step(state_, config_.dt_s, *backend_);
    }
}

void SimulationController::reset() { state_ = initial_state_; }

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
}

void SimulationController::configure_backend(
    BackendKind backend,
    int openmp_threads,
    int cuda_device) {
    if (backend == BackendKind::sequential) {
        owned_backend_.reset();
        cuda_integrator_.reset();
        backend_ = &sequential_force_backend();
        config_.backend = backend;
        config_.openmp_threads = 1;
        config_.cuda_device = 0;
        return;
    }
    if (backend == BackendKind::openmp) {
        if (openmp_threads <= 0) {
            throw std::invalid_argument("OpenMP thread count must be positive");
        }
#ifdef NBODY_VIS_HAS_OPENMP
        auto replacement = std::make_unique<OpenMpForceBackend>(openmp_threads);
        owned_backend_ = std::move(replacement);
        cuda_integrator_.reset();
        backend_ = owned_backend_.get();
        config_.backend = backend;
        config_.openmp_threads = openmp_threads;
        config_.cuda_device = 0;
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
    config_.openmp_threads = 1;
    config_.cuda_device = cuda_device;
#else
    (void)cuda_device;
    throw std::runtime_error(
        "CUDA is unavailable in this visualization build; configure with "
        "-DNBODY_ENABLE_CUDA=ON");
#endif
}

}  // namespace nbody::visualization
