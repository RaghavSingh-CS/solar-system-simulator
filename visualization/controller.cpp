#include "nbody/visualization/controller.hpp"

#ifdef NBODY_VIS_HAS_OPENMP
#include "nbody/openmp_backend.hpp"
#endif

#include <cmath>
#include <stdexcept>
#include <utility>

namespace nbody::visualization {

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
    configure_backend(config_.backend, config_.openmp_threads);
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
    return config_.backend == BackendKind::sequential ? "sequential" : "openmp";
}
int SimulationController::openmp_threads() const noexcept { return config_.openmp_threads; }

void SimulationController::set_paused(bool paused_value) noexcept { paused_ = paused_value; }
void SimulationController::toggle_paused() noexcept { paused_ = !paused_; }

void SimulationController::set_steps_per_frame(std::size_t steps) {
    if (steps == 0) {
        throw std::invalid_argument("steps per frame must be positive; use pause to stop advancement");
    }
    config_.steps_per_frame = steps;
}

void SimulationController::select_backend(BackendKind backend, int openmp_threads) {
    configure_backend(backend, openmp_threads);
}

void SimulationController::replace_scenario(
    SimulationState initial_state,
    double dt_s,
    std::size_t steps_per_frame) {
    validate_state(initial_state);
    if (!(dt_s > 0.0) || !std::isfinite(dt_s) || steps_per_frame == 0) {
        throw std::invalid_argument("replacement scenario requires positive finite dt and steps");
    }
    initial_state_ = std::move(initial_state);
    state_ = initial_state_;
    config_.dt_s = dt_s;
    config_.steps_per_frame = steps_per_frame;
    paused_ = false;
}

void SimulationController::advance_frame() {
    if (paused_) {
        return;
    }
    for (std::size_t step = 0; step < config_.steps_per_frame; ++step) {
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
}

void SimulationController::configure_backend(BackendKind backend, int openmp_threads) {
    if (backend == BackendKind::sequential) {
        owned_backend_.reset();
        backend_ = &sequential_force_backend();
        config_.backend = backend;
        config_.openmp_threads = 1;
        return;
    }
    if (openmp_threads <= 0) {
        throw std::invalid_argument("OpenMP thread count must be positive");
    }
#ifdef NBODY_VIS_HAS_OPENMP
    owned_backend_ = std::make_unique<OpenMpForceBackend>(openmp_threads);
    backend_ = owned_backend_.get();
    config_.backend = backend;
    config_.openmp_threads = openmp_threads;
#else
    (void)openmp_threads;
    throw std::runtime_error(
        "OpenMP is unavailable in this visualization build; configure with "
        "-DNBODY_ENABLE_OPENMP=ON");
#endif
}

}  // namespace nbody::visualization
