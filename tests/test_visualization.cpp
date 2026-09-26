#include "nbody/scenarios.hpp"
#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/controller.hpp"
#include "nbody/visualization/render_space.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool same_state(const nbody::SimulationState& left, const nbody::SimulationState& right) {
    if (left.time_s != right.time_s || left.bodies.size() != right.bodies.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.bodies.size(); ++index) {
        const auto& a = left.bodies[index];
        const auto& b = right.bodies[index];
        if (a.name != b.name || a.mass_kg != b.mass_kg ||
            a.position_m.x != b.position_m.x || a.position_m.y != b.position_m.y ||
            a.position_m.z != b.position_m.z || a.velocity_m_s.x != b.velocity_m_s.x ||
            a.velocity_m_s.y != b.velocity_m_s.y || a.velocity_m_s.z != b.velocity_m_s.z) {
            return false;
        }
    }
    return true;
}

void test_controller() {
    const auto initial = nbody::make_seeded_synthetic_system(8, 24301);
    nbody::visualization::SimulationController controller(
        initial,
        {10.0, 3, nbody::visualization::BackendKind::sequential, 1});
    require(controller.backend_kind() == nbody::visualization::BackendKind::sequential,
            "controller did not select sequential backend");
    controller.advance_frame();
    require(controller.state().time_s == 30.0, "controller advanced the wrong physical duration");
    controller.set_paused(true);
    const auto paused_state = controller.state();
    controller.advance_frame();
    require(same_state(paused_state, controller.state()), "paused controller advanced state");
    controller.toggle_paused();
    controller.advance_frame();
    require(controller.state().time_s == 60.0, "resumed controller did not advance");
    controller.reset();
    require(same_state(initial, controller.state()), "reset did not reproduce the initial state");

    if (nbody::visualization::SimulationController::openmp_available()) {
        controller.select_backend(nbody::visualization::BackendKind::openmp, 2);
        require(controller.backend_kind() == nbody::visualization::BackendKind::openmp &&
                    controller.openmp_threads() == 2,
                "controller did not select requested OpenMP backend");
        controller.advance_frame();
        require(controller.state().time_s == 30.0, "OpenMP controller advancement failed");
    } else {
        bool rejected = false;
        try {
            controller.select_backend(nbody::visualization::BackendKind::openmp, 2);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "OpenMP selection succeeded in a build without OpenMP");
    }
}

void test_render_space_is_read_only() {
    const auto state = nbody::make_seeded_synthetic_system(12, 91);
    const auto original = state;
    nbody::visualization::RenderSpaceTransform transform(state, 8.0F, 1.5F);
    std::vector<nbody::visualization::RenderBody> snapshot;
    transform.update_snapshot(state, snapshot);
    require(snapshot.size() == state.bodies.size(), "render snapshot body count differs");
    require(transform.position_scale() > 0.0F, "render position scale is invalid");
    for (const auto& body : snapshot) {
        require(std::isfinite(body.position.x) && std::isfinite(body.position.y) &&
                    std::isfinite(body.position.z) && body.radius > 0.0F,
                "render snapshot contains invalid geometry");
    }
    require(same_state(original, state), "render-space conversion modified physics state");
}

void test_camera_math() {
    nbody::visualization::OrbitCamera camera;
    camera.set_aspect_ratio(16.0F / 9.0F);
    const auto original_position = camera.position();
    camera.orbit(30.0F, -10.0F);
    camera.dolly(2.0F);
    require(camera.distance() >= 2.0F && camera.distance() <= 120.0F,
            "camera dolly escaped its safe range");
    require(camera.position() != original_position, "camera controls did not change position");
    const auto view = camera.view_matrix();
    const auto projection = camera.projection_matrix();
    for (int index = 0; index < 16; ++index) {
        require(std::isfinite(glm::value_ptr(view)[index]) &&
                    std::isfinite(glm::value_ptr(projection)[index]),
                "camera produced a non-finite matrix");
    }
    camera.reset();
    require(camera.position() == original_position, "camera reset did not restore its framing");
}

}  // namespace

int main() {
    try {
        test_controller();
        test_render_space_is_read_only();
        test_camera_math();
        std::cout << "all visualization model tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visualization model test failure: " << error.what() << '\n';
        return 1;
    }
}
