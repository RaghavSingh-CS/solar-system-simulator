#include "nbody/scenarios.hpp"
#include "nbody/solar_system.hpp"
#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/conservation.hpp"
#include "nbody/visualization/compute_comparison.hpp"
#include "nbody/visualization/controller.hpp"
#include "nbody/visualization/presentation.hpp"
#include "nbody/visualization/render_space.hpp"
#include "nbody/visualization/scenario.hpp"
#include "nbody/visualization/trails.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/geometric.hpp>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
    nbody::visualization::RenderSpaceTransform transform(state, {}, 8.0F, 1.5F);
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

void test_scenario_switch_and_diagnostics() {
    const auto solar = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    const auto synthetic = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::synthetic, 10, 24301);
    require(solar.initial_state.bodies.size() == 9 && solar.styles.size() == 9,
            "visual Solar-System definition is incomplete");
    require(solar.dt_s == nbody::solar_system_recommended_dt_s,
            "visual Solar-System timestep differs from the declared value");
    require(solar.reference_paths_m.size() == 9 &&
                solar.label_policy == nbody::visualization::LabelPolicy::all_bodies &&
                solar.styles[6].material == nbody::visualization::BodyMaterial::saturn,
            "visual Solar-System presentation metadata is incomplete");
    require(synthetic.label_policy == nbody::visualization::LabelPolicy::selected_only &&
                synthetic.name == "Compute Experiment",
            "synthetic mode is not intentionally framed as a low-clutter compute experiment");

    nbody::visualization::RenderSpaceTransform solar_transform(
        solar.initial_state, solar.styles, 9.0F, solar.default_radius_multiplier);
    std::vector<nbody::visualization::RenderBody> solar_snapshot;
    solar_transform.update_snapshot(solar.initial_state, solar_snapshot);
    require(std::abs(solar_snapshot[0].radius - 0.052F) < 1.0e-6F,
            "Solar render-only Sun cap was not applied");
    require(glm::distance(solar_snapshot[0].position, solar_snapshot[1].position) >
                solar_snapshot[0].radius + solar_snapshot[1].radius,
            "rendered Sun overlaps Mercury in the initial Solar composition");
    constexpr double validated_ten_year_mercury_minimum_m = 4.60084e10;
    require(validated_ten_year_mercury_minimum_m * solar_transform.position_scale() >
                solar_snapshot[0].radius + solar_snapshot[1].radius,
            "render-only Sun cap still covers Mercury at the validated physical minimum");

    nbody::visualization::SimulationController controller(
        solar.initial_state,
        {solar.dt_s, 1, nbody::visualization::BackendKind::sequential, 1});
    controller.replace_scenario(synthetic.initial_state, synthetic.dt_s, 3);
    require(controller.state().bodies.size() == 10 && controller.state().time_s == 0.0 &&
                controller.dt_s() == synthetic.dt_s && controller.steps_per_frame() == 3,
            "scenario replacement did not reset controller state/configuration");
    controller.advance_frame();
    controller.reset();
    require(same_state(synthetic.initial_state, controller.state()),
            "reset after scenario replacement did not restore the replacement state");

    nbody::visualization::ConservationTracker tracker(solar.initial_state);
    const auto initial_drift = tracker.update(solar.initial_state);
    require(initial_drift.energy_relative == 0.0 &&
                initial_drift.angular_momentum_relative == 0.0,
            "conservation tracker reference drift is not zero");
    auto evolved = solar.initial_state;
    nbody::velocity_verlet_step(evolved, solar.dt_s);
    const auto evolved_drift = tracker.update(evolved);
    require(std::isfinite(evolved_drift.energy_relative) &&
                std::isfinite(evolved_drift.angular_momentum_relative) &&
                evolved_drift.maximum_energy_relative >= evolved_drift.energy_relative,
            "conservation tracker produced invalid established drift metrics");
    tracker.reset(evolved);
    require(tracker.drift().energy_relative == 0.0 &&
                tracker.drift().maximum_angular_momentum_relative == 0.0,
            "conservation reset did not clear current/maximum drift");
}

void test_trail_boundedness() {
    auto state = nbody::make_seeded_synthetic_system(4, 7);
    nbody::visualization::TrailHistory trails(state, 3, 10.0);
    for (int step = 0; step < 8; ++step) {
        nbody::velocity_verlet_step(state, 10.0);
        trails.sample(state);
    }
    require(trails.positions().size() == 4, "trail body count differs");
    for (const auto& trail : trails.positions()) {
        require(trail.size() == 3, "trail exceeded or failed to fill bounded capacity");
    }
    trails.set_capacity(2);
    for (const auto& trail : trails.positions()) {
        require(trail.size() == 2, "trail capacity reduction did not trim old samples");
    }
    trails.reset(state);
    for (const auto& trail : trails.positions()) {
        require(trail.size() == 1, "trail reset did not clear and seed history");
    }
}

void test_actual_sun_trail_and_barycentric_transform() {
    const auto solar = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    auto state = solar.initial_state;
    nbody::visualization::TrailHistory trails(state, 16, 86400.0);
    for (int step = 0; step < 8; ++step) {
        nbody::velocity_verlet_step(state, solar.dt_s);
        trails.sample(state);
    }
    require(trails.positions()[0].size() == 3,
            "actual Sun trajectory was not sampled at the declared cadence");
    const auto& sampled_sun = trails.positions()[0].back();
    require(sampled_sun.x == state.bodies[0].position_m.x &&
                sampled_sun.y == state.bodies[0].position_m.y &&
                sampled_sun.z == state.bodies[0].position_m.z,
            "Sun trail does not contain the integrated Sun position");

    nbody::visualization::RenderSpaceTransform transform(
        solar.initial_state, solar.styles, 9.0F, solar.default_radius_multiplier);
    const glm::vec3 barycenter = transform.to_render_position({});
    require(barycenter == glm::vec3(0.0F), "barycenter marker transform is not at the origin");
    const glm::vec3 rendered_sun = transform.to_render_position(sampled_sun);
    const glm::vec3 amplified = nbody::visualization::amplify_from_barycenter(
        rendered_sun, barycenter, 10.0F);
    require(glm::length(amplified - barycenter) > glm::length(rendered_sun - barycenter),
            "Sun-motion amplification did not affect only displacement from barycenter");
}

void test_camera_math() {
    nbody::visualization::OrbitCamera camera;
    camera.set_aspect_ratio(16.0F / 9.0F);
    const auto original_position = camera.position();
    camera.orbit(30.0F, -10.0F);
    camera.dolly(2.0F);
    require(camera.distance() >= 2.0F && camera.distance() <= 120.0F,
            "camera dolly escaped its safe range");
    camera.set_distance(4.5F);
    require(camera.distance() == 4.5F, "camera focus framing distance was not applied");
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
    camera.approach_target({4.0F, -2.0F, 1.0F}, 0.25F);
    require(camera.target() == glm::vec3(1.0F, -0.5F, 0.25F),
            "camera focus target did not move smoothly by the requested fraction");
    camera.reset();
    require(camera.target() == glm::vec3(0.0F),
            "camera overview reset did not restore the origin target");
}

void test_presentation_helpers() {
    const auto within = nbody::visualization::classify_numerical_health(5.0e-6, 5.0e-10);
    require(within.all_within_bounds(), "numerical-health bounds rejected exact limits");
    const auto exceeded = nbody::visualization::classify_numerical_health(
        5.0001e-6, std::numeric_limits<double>::infinity());
    require(!exceeded.energy_within_bound && !exceeded.angular_momentum_within_bound,
            "numerical-health warning mapping accepted invalid drift");

    const auto solar = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    nbody::visualization::RenderSpaceTransform transform(
        solar.initial_state, solar.styles, 9.0F, solar.default_radius_multiplier);
    std::vector<nbody::visualization::RenderBody> bodies;
    transform.update_snapshot(solar.initial_state, bodies);
    const auto earth = nbody::visualization::camera_framing(
        nbody::visualization::CameraPreset::earth,
        nbody::visualization::ScenarioKind::solar_system,
        bodies);
    require(earth.tracked_body == 3U && earth.target == bodies[3].position &&
                earth.distance >= 2.0F,
            "Earth camera preset is not deterministic or body-relative");
    nbody::visualization::OrbitCamera camera;
    nbody::visualization::apply_camera_framing(camera, earth);
    require(camera.target() == bodies[3].position && camera.distance() == earth.distance,
            "camera preset did not apply its exact framing");

    std::vector<nbody::visualization::LabelCandidate> candidates{
        {0, {500.0F, 300.0F}, {70.0F, 24.0F}, 900, false},
        {1, {502.0F, 301.0F}, {70.0F, 24.0F}, 100, false},
        {3, {501.0F, 300.0F}, {70.0F, 24.0F}, 1000, true},
    };
    const auto placements = nbody::visualization::declutter_labels(
        candidates, {1280.0F, 720.0F}, 420.0F);
    require(!placements.empty() && placements.front().body_index == 3 &&
                placements.front().selected,
            "selected label did not receive highest decluttering priority");
    for (std::size_t left = 0; left < placements.size(); ++left) {
        for (std::size_t right = left + 1; right < placements.size(); ++right) {
            const auto& a = placements[left];
            const auto& b = placements[right];
            const bool overlap = a.position.x < b.position.x + b.size.x &&
                a.position.x + a.size.x > b.position.x &&
                a.position.y < b.position.y + b.size.y &&
                a.position.y + a.size.y > b.position.y;
            require(!overlap, "decluttered labels still overlap");
        }
    }

    auto moved_candidates = candidates;
    for (auto& candidate : moved_candidates) {
        candidate.anchor += glm::vec2(0.25F, -0.20F);
    }
    const auto stable = nbody::visualization::declutter_labels(
        moved_candidates, {1280.0F, 720.0F}, 420.0F, 6.0F, placements);
    require(stable.size() == placements.size(),
            "tiny projected motion changed the accepted label set");
    for (std::size_t index = 0; index < stable.size(); ++index) {
        require(glm::distance(
                    stable[index].position - stable[index].anchor,
                    placements[index].position - placements[index].anchor) < 1.0e-4F,
                "label hysteresis did not retain the prior accepted anchor");
    }
}

void test_compute_experiment_scaling_and_comparison() {
    constexpr std::array<std::size_t, 6> supported{{32, 64, 128, 256, 512, 1024}};
    for (const std::size_t count : supported) {
        const auto first = nbody::visualization::make_visual_scenario(
            nbody::visualization::ScenarioKind::synthetic, count, 24301);
        const auto second = nbody::visualization::make_visual_scenario(
            nbody::visualization::ScenarioKind::synthetic, count, 24301);
        require(first.initial_state.bodies.size() == count &&
                    same_state(first.initial_state, second.initial_state),
                "supported Compute Experiment workload is not deterministic");
        nbody::visualization::RenderSpaceTransform transform(
            first.initial_state, first.styles, 9.0F, first.default_radius_multiplier);
        std::vector<nbody::visualization::RenderBody> snapshot;
        transform.update_snapshot(first.initial_state, snapshot);
        require(snapshot.size() == count,
                "large-N Compute Experiment render snapshot is incomplete");
    }

    const auto initial = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::synthetic, 512, 24301);
    const auto visible_state = initial.initial_state;
    if (nbody::visualization::SimulationController::openmp_available()) {
        const auto comparison = nbody::visualization::run_live_compute_comparison(
            initial.initial_state, initial.dt_s, 4);
        require(comparison.body_count == 512 && comparison.integration_steps > 0 &&
                    comparison.sequential_ms > 0.0 && comparison.openmp_ms > 0.0,
                "live CPU comparison did not execute the declared fixed workload");
        require(comparison.correctness_passed,
                "live Sequential/OpenMP comparison failed established mixed tolerances");
        require(nbody::visualization::live_comparison_matches(comparison, 512, 4) &&
                    !nbody::visualization::live_comparison_matches(comparison, 256, 4) &&
                    !nbody::visualization::live_comparison_matches(comparison, 512, 2),
                "live comparison result did not invalidate for N/thread changes");
        require(same_state(visible_state, initial.initial_state),
                "live CPU comparison mutated its visible input state");
        std::cout << "live diagnostic (not benchmark evidence): N=512 steps="
                  << comparison.integration_steps
                  << " sequential_ms=" << comparison.sequential_ms
                  << " openmp4_ms=" << comparison.openmp_ms
                  << " ratio=" << comparison.sequential_to_openmp_ratio << '\n';
    }
}

}  // namespace

int main() {
    try {
        test_controller();
        test_render_space_is_read_only();
        test_scenario_switch_and_diagnostics();
        test_trail_boundedness();
        test_actual_sun_trail_and_barycentric_transform();
        test_camera_math();
        test_presentation_helpers();
        test_compute_experiment_scaling_and_comparison();
        std::cout << "all visualization model tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visualization model test failure: " << error.what() << '\n';
        return 1;
    }
}
