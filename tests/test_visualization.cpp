#include "nbody/astronomy/solar_system_60.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/scenarios.hpp"
#include "nbody/solar_system.hpp"
#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/application_transaction.hpp"
#include "nbody/visualization/conservation.hpp"
#include "nbody/visualization/compute_comparison.hpp"
#include "nbody/visualization/controller.hpp"
#include "nbody/visualization/counterfactual_worker.hpp"
#include "nbody/visualization/presentation.hpp"
#include "nbody/visualization/render_space.hpp"
#include "nbody/visualization/scenario.hpp"
#include "nbody/visualization/trails.hpp"
#include "nbody/visualization/workload.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
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

std::uint64_t state_checksum(const nbody::SimulationState& state) {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto append = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
    };
    const auto bits = [](double value) {
        std::uint64_t result = 0;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    };
    for (const auto& body : state.bodies) {
        append(body.name.data(), body.name.size());
        const std::uint64_t values[]{
            bits(body.mass_kg), bits(body.position_m.x), bits(body.position_m.y),
            bits(body.position_m.z), bits(body.velocity_m_s.x),
            bits(body.velocity_m_s.y), bits(body.velocity_m_s.z),
        };
        append(values, sizeof(values));
    }
    return hash;
}

bool same_diagnostics(
    const nbody::Diagnostics& left,
    const nbody::Diagnostics& right) {
    return left.kinetic_energy_j == right.kinetic_energy_j &&
        left.potential_energy_j == right.potential_energy_j &&
        left.linear_momentum_kg_m_s.x == right.linear_momentum_kg_m_s.x &&
        left.linear_momentum_kg_m_s.y == right.linear_momentum_kg_m_s.y &&
        left.linear_momentum_kg_m_s.z == right.linear_momentum_kg_m_s.z &&
        left.angular_momentum_kg_m2_s.x == right.angular_momentum_kg_m2_s.x &&
        left.angular_momentum_kg_m2_s.y == right.angular_momentum_kg_m2_s.y &&
        left.angular_momentum_kg_m2_s.z == right.angular_momentum_kg_m2_s.z &&
        left.center_of_mass_m.x == right.center_of_mass_m.x &&
        left.center_of_mass_m.y == right.center_of_mass_m.y &&
        left.center_of_mass_m.z == right.center_of_mass_m.z;
}

bool same_barnes_hut_work_statistics(
    const nbody::BarnesHutStatistics& left,
    const nbody::BarnesHutStatistics& right) {
    return left.body_count == right.body_count &&
        left.node_count == right.node_count &&
        left.leaf_count == right.leaf_count &&
        left.maximum_depth_reached == right.maximum_depth_reached &&
        left.visited_nodes == right.visited_nodes &&
        left.approximated_nodes == right.approximated_nodes &&
        left.direct_body_interactions == right.direct_body_interactions &&
        left.working_memory_bytes == right.working_memory_bytes;
}

void require_barnes_hut_boundary(
    const nbody::SimulationState& reference,
    const nbody::BarnesHutStatistics& reference_statistics,
    const nbody::visualization::SimulationController& controller,
    nbody::visualization::BackendKind expected_backend,
    int expected_threads,
    double expected_theta,
    std::size_t step,
    double dt_s,
    const std::string& backend_label) {
    const std::string context = backend_label +
        " Barnes-Hut boundary step " + std::to_string(step);
    require(controller.state().time_s == static_cast<double>(step) * dt_s,
        context + " has the wrong timestamp");
    require(controller.state().bodies.size() == reference.bodies.size(),
        context + " has the wrong body count");
    require(same_state(reference, controller.state()),
        context + " differs in timestamp, body ordering/name, mass, position, or velocity");
    require(same_diagnostics(
                nbody::compute_diagnostics(reference),
                nbody::compute_diagnostics(controller.state())),
        context + " differs in conservation diagnostics");
    require(controller.backend_kind() == expected_backend &&
                controller.solver_kind() ==
                    nbody::visualization::SolverKind::barnes_hut &&
                controller.barnes_hut_configuration().theta == expected_theta &&
                (expected_backend != nbody::visualization::BackendKind::openmp ||
                 controller.openmp_threads() == expected_threads),
        context + " reports the wrong effective backend, solver, theta, or thread count");
    const auto statistics = controller.barnes_hut_statistics();
    require(statistics.has_value(), context + " did not publish Barnes-Hut statistics");
    require(statistics->worker_threads == expected_threads &&
                same_barnes_hut_work_statistics(reference_statistics, *statistics),
        context + " differs in deterministic force/tree statistics");
}

std::size_t required_index(
    const nbody::visualization::ScenarioDefinition& scenario,
    const std::string& stable_id) {
    const auto index = nbody::visualization::find_body_index(scenario, stable_id);
    require(index.has_value(), "missing required stable ID: " + stable_id);
    return *index;
}

double maximum_parent_relative_difference(
    const nbody::visualization::ScenarioDefinition& scenario,
    const nbody::SimulationState& left,
    const nbody::SimulationState& right,
    double& maximum_absolute_m,
    std::string* worst_stable_id = nullptr) {
    maximum_absolute_m = 0.0;
    double maximum_normalized = 0.0;
    for (std::size_t index = 0; index < scenario.body_metadata.size(); ++index) {
        const auto& parent_id = scenario.body_metadata[index].parent_stable_id;
        if (!parent_id) {
            continue;
        }
        const std::size_t parent = required_index(scenario, *parent_id);
        const nbody::Vec3 left_relative =
            left.bodies[index].position_m - left.bodies[parent].position_m;
        const nbody::Vec3 right_relative =
            right.bodies[index].position_m - right.bodies[parent].position_m;
        const double difference = nbody::norm(left_relative - right_relative);
        const double scale = std::max(
            {nbody::norm(left_relative), nbody::norm(right_relative), 1.0});
        if (difference / scale > maximum_normalized) {
            maximum_normalized = difference / scale;
            if (worst_stable_id != nullptr) {
                *worst_stable_id = scenario.stable_ids[index];
            }
        }
        maximum_absolute_m = std::max(maximum_absolute_m, difference);
    }
    return maximum_normalized;
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
    controller.advance_steps(2);
    require(controller.state().time_s == 20.0,
            "manual fixed-step advancement did not ignore paused-frame batching safely");
    controller.reset();

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

    if (!nbody::visualization::SimulationController::cuda_available()) {
        const auto state_before_selection = controller.state();
        const auto backend_before_selection = controller.backend_kind();
        bool rejected = false;
        try {
            controller.select_backend(nbody::visualization::BackendKind::cuda);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "CUDA selection succeeded without an available CUDA device");
        require(controller.backend_kind() == backend_before_selection &&
                    same_state(controller.state(), state_before_selection),
                "failed CUDA selection changed the active backend or state");
    }
}

void test_controller_solver_selection_and_transactionality() {
    using nbody::visualization::BackendKind;
    using nbody::visualization::ControllerConfig;
    using nbody::visualization::SimulationController;
    using nbody::visualization::SolverKind;

    const auto initial = nbody::make_seeded_synthetic_system(12, 77123);
    ControllerConfig direct_config{10.0, 2, BackendKind::sequential, 1};
    SimulationController direct(initial, direct_config);
    require(direct.solver_kind() == SolverKind::direct &&
                std::string(direct.solver_name()) == "direct",
            "controller default did not preserve direct-force behavior");

    auto direct_reference = initial;
    nbody::velocity_verlet_step(
        direct_reference, direct_config.dt_s, nbody::sequential_force_backend());
    direct.advance_steps(1);
    require(same_state(direct_reference, direct.state()),
            "Sequential/direct controller differs from the headless reference step");

    nbody::BarnesHutConfiguration barnes_hut;
    barnes_hut.theta = 0.25;
    SimulationController sequential_barnes_hut(
        initial,
        {10.0, 2, BackendKind::sequential, 1, 0, SolverKind::barnes_hut, barnes_hut});
    auto sequential_barnes_hut_reference = initial;
    nbody::SequentialBarnesHutForceBackend sequential_reference_backend(barnes_hut);
    nbody::velocity_verlet_step(
        sequential_barnes_hut_reference, 10.0, sequential_reference_backend);
    sequential_barnes_hut.advance_steps(1);
    require(same_state(sequential_barnes_hut_reference, sequential_barnes_hut.state()),
            "Sequential/Barnes-Hut controller differs from its headless reference step");
    require(sequential_barnes_hut.barnes_hut_statistics().has_value() &&
                sequential_barnes_hut.barnes_hut_statistics()->body_count ==
                    initial.bodies.size(),
            "Barnes-Hut effective diagnostics were not published");

    constexpr std::size_t correspondence_steps = 7U;
    constexpr double correspondence_dt_s = 10.0;
    auto multistep_reference = initial;
    nbody::SequentialBarnesHutForceBackend multistep_reference_backend(barnes_hut);
    SimulationController multistep_sequential(
        initial,
        {correspondence_dt_s, 1, BackendKind::sequential, 1, 0,
         SolverKind::barnes_hut, barnes_hut});
    for (std::size_t step = 1; step <= correspondence_steps; ++step) {
        nbody::velocity_verlet_step(
            multistep_reference, correspondence_dt_s, multistep_reference_backend);
        multistep_sequential.advance_frame();
        require_barnes_hut_boundary(
            multistep_reference,
            multistep_reference_backend.last_statistics(),
            multistep_sequential,
            BackendKind::sequential,
            1,
            barnes_hut.theta,
            step,
            correspondence_dt_s,
            "Sequential");

        if (step == 1U) {
            auto perturbed_boundary = multistep_reference;
            perturbed_boundary.bodies.front().position_m.x = std::nextafter(
                perturbed_boundary.bodies.front().position_m.x,
                std::numeric_limits<double>::infinity());
            require(!same_state(multistep_reference, perturbed_boundary),
                "per-step correspondence negative control did not detect an isolated "
                "intermediate position perturbation");
        }
    }
    require(same_state(multistep_reference, multistep_sequential.state()) &&
                multistep_sequential.state().time_s == 70.0 &&
                same_diagnostics(
                    nbody::compute_diagnostics(multistep_reference),
                    nbody::compute_diagnostics(multistep_sequential.state())) &&
                multistep_sequential.barnes_hut_statistics().has_value() &&
                multistep_sequential.barnes_hut_statistics()->body_count ==
                    initial.bodies.size(),
            "multi-step Sequential/Barnes-Hut viewer advancement differs from headless execution");

    const auto state_before_switch = sequential_barnes_hut.state();
    sequential_barnes_hut.select_solver(SolverKind::direct, barnes_hut);
    require(sequential_barnes_hut.solver_kind() == SolverKind::direct &&
                same_state(state_before_switch, sequential_barnes_hut.state()),
            "solver switching reset or advanced the authoritative state");
    sequential_barnes_hut.select_solver(SolverKind::barnes_hut, barnes_hut);
    require(sequential_barnes_hut.solver_kind() == SolverKind::barnes_hut &&
                same_state(state_before_switch, sequential_barnes_hut.state()),
            "switching back to Barnes-Hut changed the authoritative state");
    auto applied_configuration = barnes_hut;
    applied_configuration.theta = 0.5;
    sequential_barnes_hut.select_solver(
        SolverKind::barnes_hut, applied_configuration);
    require(sequential_barnes_hut.barnes_hut_configuration().theta == 0.5 &&
                same_state(state_before_switch, sequential_barnes_hut.state()),
            "successful theta application did not update only the effective solver configuration");
    sequential_barnes_hut.select_solver(SolverKind::barnes_hut, barnes_hut);
    sequential_barnes_hut.advance_steps(1);
    require(sequential_barnes_hut.state().time_s == state_before_switch.time_s + 10.0,
            "Barnes-Hut did not continue from the solver-switch boundary");

    if (SimulationController::openmp_available()) {
        SimulationController openmp_direct(
            initial, {10.0, 1, BackendKind::openmp, 2, 0, SolverKind::direct, barnes_hut});
        openmp_direct.advance_steps(1);
        require(same_state(direct_reference, openmp_direct.state()),
                "OpenMP/direct controller differs from the Sequential reference");

        SimulationController openmp_barnes_hut(
            initial,
            {10.0, 1, BackendKind::openmp, 2, 0, SolverKind::barnes_hut, barnes_hut});
        openmp_barnes_hut.advance_steps(1);
        require(same_state(
                    sequential_barnes_hut_reference, openmp_barnes_hut.state()),
                "OpenMP/Barnes-Hut controller differs from the Sequential Barnes-Hut reference");
        require(openmp_barnes_hut.barnes_hut_statistics().has_value() &&
                    openmp_barnes_hut.barnes_hut_statistics()->worker_threads == 2,
                "OpenMP/Barnes-Hut effective diagnostics omit the requested thread count");

        SimulationController multistep_openmp(
            initial,
            {correspondence_dt_s, 1, BackendKind::openmp, 2, 0,
             SolverKind::barnes_hut, barnes_hut});
        auto multistep_openmp_reference = initial;
#ifdef _OPENMP
        nbody::OpenMpBarnesHutForceBackend multistep_openmp_reference_backend(
            2, barnes_hut);
#else
        nbody::SequentialBarnesHutForceBackend multistep_openmp_reference_backend(
            barnes_hut);
#endif
        for (std::size_t step = 1; step <= correspondence_steps; ++step) {
            nbody::velocity_verlet_step(
                multistep_openmp_reference,
                correspondence_dt_s,
                multistep_openmp_reference_backend);
            multistep_openmp.advance_frame();
            require_barnes_hut_boundary(
                multistep_openmp_reference,
                multistep_openmp_reference_backend.last_statistics(),
                multistep_openmp,
                BackendKind::openmp,
                2,
                barnes_hut.theta,
                step,
                correspondence_dt_s,
                "OpenMP");
        }
        require(same_state(multistep_openmp_reference, multistep_openmp.state()) &&
                    multistep_openmp.state().time_s == 70.0 &&
                    same_diagnostics(
                        nbody::compute_diagnostics(multistep_openmp_reference),
                        nbody::compute_diagnostics(multistep_openmp.state())) &&
                    multistep_openmp.barnes_hut_statistics().has_value() &&
                    multistep_openmp.barnes_hut_statistics()->worker_threads == 2,
                "multi-step OpenMP/Barnes-Hut viewer advancement differs from headless execution");
    } else {
        bool rejected = false;
        try {
            SimulationController unavailable(
                initial,
                {10.0, 1, BackendKind::openmp, 2, 0, SolverKind::barnes_hut, barnes_hut});
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected,
                "OpenMP/Barnes-Hut was not rejected in the CPU-only build");
    }

    for (const double invalid_theta : {
             0.0, -0.25, 2.0001, std::numeric_limits<double>::quiet_NaN()}) {
        auto invalid = barnes_hut;
        invalid.theta = invalid_theta;
        const auto state_before_rejection = sequential_barnes_hut.state();
        const auto solver_before_rejection = sequential_barnes_hut.solver_kind();
        bool rejected = false;
        try {
            sequential_barnes_hut.select_solver(SolverKind::barnes_hut, invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected &&
                    sequential_barnes_hut.solver_kind() == solver_before_rejection &&
                    same_state(state_before_rejection, sequential_barnes_hut.state()) &&
                    sequential_barnes_hut.barnes_hut_configuration().theta == barnes_hut.theta,
                "invalid theta changed effective solver configuration or state");
    }

    const auto state_before_cuda_rejection = sequential_barnes_hut.state();
    bool cuda_barnes_hut_rejected = false;
    try {
        sequential_barnes_hut.select_backend(BackendKind::cuda, 1, 0);
    } catch (const std::runtime_error& error) {
        cuda_barnes_hut_rejected =
            std::string(error.what()).find("pending NVIDIA validation") != std::string::npos;
    }
    require(cuda_barnes_hut_rejected &&
                sequential_barnes_hut.backend_kind() == BackendKind::sequential &&
                sequential_barnes_hut.solver_kind() == SolverKind::barnes_hut &&
                same_state(state_before_cuda_rejection, sequential_barnes_hut.state()),
            "unsupported CUDA/Barnes-Hut selection fell back or changed state");

    nbody::SimulationState collision;
    const double collision_mass = 4.0 / nbody::gravitational_constant;
    collision.bodies = {
        {"left", collision_mass, {-1.0, 0.0, 0.0}, {0.5, 0.0, 0.0}},
        {"right", collision_mass, {1.0, 0.0, 0.0}, {-0.5, 0.0, 0.0}},
    };
    auto collision_config = barnes_hut;
    collision_config.leaf_capacity = 1;
    SimulationController failing(
        collision,
        {1.0, 1, BackendKind::sequential, 1, 0,
         SolverKind::barnes_hut, collision_config});
    bool advancement_failed = false;
    try {
        failing.advance_steps(1);
    } catch (const std::exception&) {
        advancement_failed = true;
    }
    require(advancement_failed && failing.paused() &&
                same_state(collision, failing.state()) && !failing.last_error().empty() &&
                !failing.barnes_hut_statistics().has_value() &&
                failing.last_error().find("backend=sequential") != std::string::npos &&
                failing.last_error().find("solver=barnes-hut") != std::string::npos &&
                failing.last_error().find("step=1/1") != std::string::npos &&
                failing.last_error().find("phase=new-force") != std::string::npos,
            "failed Barnes-Hut batch published partial state/statistics or omitted its diagnostic");
    failing.reset();
    require(same_state(collision, failing.state()) && failing.last_error().empty(),
            "reset did not restore the initial state and clear the failure diagnostic");

    const auto legacy = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    const auto catalog = nbody::visualization::make_astronomical_catalog_scenario();
    SimulationController scenario_controller(
        legacy.initial_state,
        {legacy.dt_s, 1, BackendKind::sequential, 1, 0,
         SolverKind::barnes_hut, barnes_hut});
    scenario_controller.set_paused(true);
    scenario_controller.advance_steps(2);
    const auto state_before_invalid_replacement = scenario_controller.state();
    const auto initial_before_invalid_replacement = scenario_controller.initial_state();
    const double dt_before_invalid_replacement = scenario_controller.dt_s();
    const std::size_t steps_before_invalid_replacement =
        scenario_controller.steps_per_frame();
    auto invalid_replacement = catalog.initial_state;
    invalid_replacement.bodies.front().mass_kg =
        std::numeric_limits<double>::quiet_NaN();
    bool invalid_replacement_rejected = false;
    try {
        scenario_controller.replace_scenario(
            std::move(invalid_replacement), catalog.dt_s, 1);
    } catch (const std::invalid_argument&) {
        invalid_replacement_rejected = true;
    }
    require(invalid_replacement_rejected && scenario_controller.paused() &&
                same_state(state_before_invalid_replacement, scenario_controller.state()) &&
                same_state(initial_before_invalid_replacement,
                    scenario_controller.initial_state()) &&
                scenario_controller.dt_s() == dt_before_invalid_replacement &&
                scenario_controller.steps_per_frame() == steps_before_invalid_replacement,
            "failed controller scenario replacement changed state, time, or metadata");
    scenario_controller.replace_scenario(catalog.initial_state, catalog.dt_s, 1);
    require(scenario_controller.paused() &&
                scenario_controller.solver_kind() == SolverKind::barnes_hut &&
                same_state(catalog.initial_state, scenario_controller.state()) &&
                scenario_controller.state().bodies.size() == 60U,
            "catalog scenario replacement changed solver, pause state, or initial data");
    scenario_controller.reset();
    require(same_state(catalog.initial_state, scenario_controller.state()),
            "Barnes-Hut reset did not preserve the 60-body catalog initial state");
    scenario_controller.replace_scenario(legacy.initial_state, legacy.dt_s, 1);
    require(same_state(legacy.initial_state, scenario_controller.state()) &&
                same_state(
                    legacy.initial_state,
                    nbody::make_approximate_solar_system().initial_state),
            "legacy Solar scenario changed during Barnes-Hut scenario switching");
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

void test_solar_screen_space_sizing() {
    constexpr float viewport_width = 1920.0F - 446.0F;
    constexpr float viewport_height = 1080.0F;
    const auto solar = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    nbody::visualization::RenderSpaceTransform transform(
        solar.initial_state, solar.styles, 9.0F, solar.default_radius_multiplier);

    const auto evaluate = [&](const nbody::SimulationState& state,
                              const nbody::visualization::OrbitCamera& camera) {
        const auto state_before = state;
        std::vector<nbody::visualization::RenderBody> snapshot;
        transform.update_snapshot(state, snapshot);
        const auto original_positions = snapshot;
        nbody::visualization::apply_screen_space_body_sizing(
            snapshot, camera, viewport_height, std::size_t{0});
        require(same_state(state_before, state),
                "presentation sizing mutated the physical simulation state");
        for (std::size_t index = 0; index < snapshot.size(); ++index) {
            require(snapshot[index].position == original_positions[index].position,
                    "screen-space sizing changed a physical body center");
        }
        return nbody::visualization::apparent_radius_pixels(
            snapshot[0].radius,
            glm::distance(snapshot[0].position, camera.position()), viewport_height);
    };

    std::vector<nbody::visualization::RenderBody> reference_snapshot;
    transform.update_snapshot(solar.initial_state, reference_snapshot);
    nbody::visualization::OrbitCamera overview_camera;
    nbody::visualization::apply_camera_framing(
        overview_camera,
        nbody::visualization::camera_framing(
            nbody::visualization::CameraPreset::overview,
            solar,
            reference_snapshot));
    overview_camera.set_aspect_ratio(viewport_width / viewport_height);
    const float overview_sun = evaluate(solar.initial_state, overview_camera);
    const auto inner_request = nbody::visualization::camera_framing(
        nbody::visualization::CameraPreset::inner_system, solar, reference_snapshot);
    transform.configure_view(
        solar.initial_state,
        solar.initial_state.bodies[*inner_request.local_origin_body].position_m,
        inner_request.visible_bodies,
        8.0F);
    transform.update_snapshot(solar.initial_state, reference_snapshot);
    nbody::visualization::OrbitCamera inner_camera;
    nbody::visualization::apply_camera_framing(
        inner_camera,
        nbody::visualization::camera_framing(
            nbody::visualization::CameraPreset::inner_system,
            solar,
            reference_snapshot));
    inner_camera.set_aspect_ratio(viewport_width / viewport_height);
    const float inner_sun = evaluate(solar.initial_state, inner_camera);
    require(overview_sun >= nbody::visualization::sun_minimum_apparent_radius_pixels - 0.01F &&
                overview_sun <= nbody::visualization::sun_maximum_apparent_radius_pixels + 0.01F &&
                inner_sun >= nbody::visualization::sun_minimum_apparent_radius_pixels - 0.01F &&
                inner_sun <= nbody::visualization::sun_maximum_apparent_radius_pixels + 0.01F,
            "Solar presets violated the Sun readability bounds");
    require(inner_sun > overview_sun,
            "Inner System zoom did not increase the Sun's stable apparent size");

    nbody::visualization::OrbitCamera orbit_test_camera(35.0F, 28.0F, 7.0F);
    orbit_test_camera.set_aspect_ratio(viewport_width / viewport_height);
    auto evolved = solar.initial_state;
    const float invariant_sun = evaluate(evolved, orbit_test_camera);
    for (int step = 0; step < 352; ++step) {
        nbody::velocity_verlet_step(evolved, solar.dt_s);
        const float current_sun = evaluate(evolved, orbit_test_camera);
        require(std::abs(current_sun - invariant_sun) <= 1.0e-4F,
                "fixed-camera Sun apparent radius changed with orbital phase");
    }

    nbody::visualization::OrbitCamera slightly_farther(35.0F, 28.0F, 7.01F);
    slightly_farther.set_aspect_ratio(viewport_width / viewport_height);
    const float nearby_zoom_sun = evaluate(solar.initial_state, slightly_farther);
    require(nearby_zoom_sun <= invariant_sun + 1.0e-4F &&
                std::abs(nearby_zoom_sun - invariant_sun) < 0.1F,
            "small Sun zoom response is discontinuous or has the wrong direction");
    nbody::visualization::OrbitCamera farther_camera(35.0F, 28.0F, 10.0F);
    farther_camera.set_aspect_ratio(viewport_width / viewport_height);
    const float farther_sun = evaluate(solar.initial_state, farther_camera);
    require(farther_sun < invariant_sun,
            "meaningful zoom-out did not reduce the Sun's apparent radius");

    std::vector<nbody::visualization::RenderBody> bounded_snapshot;
    transform.update_snapshot(solar.initial_state, bounded_snapshot);
    nbody::visualization::apply_screen_space_body_sizing(
        bounded_snapshot, overview_camera, viewport_height, std::size_t{0});
    for (std::size_t index = 1; index < bounded_snapshot.size(); ++index) {
        const float pixels = nbody::visualization::apparent_radius_pixels(
            bounded_snapshot[index].radius,
            glm::distance(bounded_snapshot[index].position, overview_camera.position()),
            viewport_height);
        require(pixels >= nbody::visualization::body_minimum_apparent_radius_pixels - 0.01F &&
                    pixels <= nbody::visualization::body_maximum_apparent_radius_pixels + 0.01F,
                "non-Sun apparent-radius bounds changed unexpectedly");
    }

    std::cout << "Stable Solar sizing at 1920x1080: overview Sun=" << overview_sun
              << "px; inner Sun=" << inner_sun
              << "px; full-orbit fixed-camera Sun=" << invariant_sun << "px\n";
}

void test_scenario_switch_and_diagnostics() {
    const auto solar = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    const auto synthetic = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::synthetic, 256, 24301);
    require(solar.initial_state.bodies.size() == 9 && solar.styles.size() == 9,
            "visual Solar-System definition is incomplete");
    require(solar.dt_s == nbody::solar_system_recommended_dt_s,
            "visual Solar-System timestep differs from the declared value");
    require(solar.reference_paths_m.size() == 9 &&
                solar.label_policy == nbody::visualization::LabelPolicy::all_bodies &&
                solar.styles[6].material == nbody::visualization::BodyMaterial::saturn,
            "visual Solar-System presentation metadata is incomplete");
    require(synthetic.label_policy == nbody::visualization::LabelPolicy::selected_only &&
                synthetic.name == "Synthetic N-body Sandbox",
            "synthetic mode is not intentionally framed as a low-clutter compute experiment");

    nbody::visualization::SimulationController controller(
        solar.initial_state,
        {solar.dt_s, 1, nbody::visualization::BackendKind::sequential, 1});
    controller.replace_scenario(synthetic.initial_state, synthetic.dt_s, 3);
    require(controller.state().bodies.size() == 256 && controller.state().time_s == 0.0 &&
                controller.dt_s() == synthetic.dt_s && controller.steps_per_frame() == 3,
            "scenario replacement did not reset controller state/configuration");
    controller.advance_frame();
    controller.reset();
    require(same_state(synthetic.initial_state, controller.state()),
            "reset after scenario replacement did not restore the replacement state");

    const auto clustered = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::synthetic, 256, 91,
        nbody::LargeNDistribution::clustered);
    controller.set_paused(true);
    controller.replace_scenario(clustered.initial_state, clustered.dt_s, 1);
    const auto paused_state = controller.state();
    controller.advance_frame();
    require(controller.paused() && same_state(paused_state, controller.state()) &&
                controller.state().time_s == 0.0,
            "synthetic scenario replacement did not preserve pause or reset time");
    controller.advance_steps(1);
    require(controller.state().time_s == clustered.dt_s,
            "manual synthetic advancement while paused used incorrect timestep");
    controller.reset();
    require(controller.paused() && same_state(clustered.initial_state, controller.state()),
            "synthetic reset did not restore the selected distribution and pause state");

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

void test_astronomical_catalog_composition() {
    const auto legacy = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    const auto legacy_baseline = nbody::make_approximate_solar_system();
    require(same_state(legacy.initial_state, legacy_baseline.initial_state),
            "catalog integration changed the legacy nine-body initial state");

    const auto catalog = nbody::astronomy::make_solar_system_60_catalog();
    const auto full = nbody::visualization::make_astronomical_catalog_scenario();
    require(full.initial_state.bodies.size() == 60U && full.stable_ids.size() == 60U &&
                full.body_metadata.size() == 60U && full.styles.size() == 60U,
            "default astronomical scenario does not contain exactly 60 complete bodies");
    require(full.dt_s == nbody::visualization::astronomical_catalog_initial_dt_s &&
                !full.timestep_validation_complete && full.reference_paths_m.empty(),
            "catalog timestep status or reference-guide policy is misleading");
    require(full.composition_warnings.empty(),
            "complete catalog scenario unexpectedly reports missing-source warnings");
    for (std::size_t index = 0; index < catalog.records().size(); ++index) {
        const auto& record = catalog.records()[index];
        const auto& body = full.initial_state.bodies[index];
        require(full.stable_ids[index] == record.stable_id() &&
                    full.body_metadata[index].stable_id == record.stable_id() &&
                    full.body_metadata[index].parent_stable_id == record.parent_stable_id() &&
                    body.name == record.display_name() && body.mass_kg == record.mass_kg() &&
                    body.position_m.x == record.position_m().x &&
                    body.position_m.y == record.position_m().y &&
                    body.position_m.z == record.position_m().z &&
                    body.velocity_m_s.x == record.velocity_m_s().x &&
                    body.velocity_m_s.y == record.velocity_m_s().y &&
                    body.velocity_m_s.z == record.velocity_m_s().z,
                "catalog scenario changed record identity, ordering, or initial state");
    }
    nbody::visualization::RenderSpaceTransform catalog_transform(
        full.initial_state, full.styles, 8.0F, full.default_radius_multiplier);
    std::vector<nbody::visualization::RenderBody> catalog_snapshot;
    catalog_transform.update_snapshot(full.initial_state, catalog_snapshot);
    const std::array<std::pair<nbody::visualization::CameraPreset, const char*>, 7> views{{
        {nbody::visualization::CameraPreset::earth_moon, "sol:earth"},
        {nbody::visualization::CameraPreset::jupiter_system, "sol:jupiter"},
        {nbody::visualization::CameraPreset::saturn_system, "sol:saturn"},
        {nbody::visualization::CameraPreset::uranus_system, "sol:uranus"},
        {nbody::visualization::CameraPreset::neptune_triton, "sol:neptune"},
        {nbody::visualization::CameraPreset::pluto_charon, "sol:pluto"},
        {nbody::visualization::CameraPreset::asteroid_belt, "sol:sun"},
    }};
    for (const auto& [preset, stable_id] : views) {
        const auto framing = nbody::visualization::camera_framing(
            preset, full, catalog_snapshot);
        require(framing.tracked_body == required_index(full, stable_id) &&
                    framing.local_origin_body == framing.tracked_body &&
                    framing.visible_bodies.size() >= 2U,
                "stable-ID subsystem view resolved the wrong body or incomplete composition");
    }

    const std::vector<std::string> selected{
        "sol:sun", "sol:earth", "sol:moon", "sol:jupiter", "sol:io"};
    const auto subset = nbody::visualization::make_astronomical_catalog_scenario(selected);
    const auto repeated = nbody::visualization::make_astronomical_catalog_scenario(selected);
    require(subset.stable_ids == repeated.stable_ids &&
                same_state(subset.initial_state, repeated.initial_state),
            "catalog subset composition is not deterministic");
    const std::vector<std::string> expected_order{
        "sol:sun", "sol:earth", "sol:jupiter", "sol:moon", "sol:io"};
    require(subset.stable_ids == expected_order,
            "catalog subset did not preserve parent-before-child catalog ordering");
    require(!subset.composition_warnings.empty(),
            "major gravitational-source exclusions were not disclosed");

    const auto parentless = nbody::visualization::make_astronomical_catalog_scenario(
        {"sol:sun", "sol:moon"});
    require(std::any_of(
                parentless.composition_warnings.begin(), parentless.composition_warnings.end(),
                [](const std::string& warning) {
                    return warning.find("sol:moon included without catalog parent sol:earth") !=
                        std::string::npos;
                }),
            "including a satellite without its parent did not produce an explicit warning");

    bool duplicate_rejected = false;
    try {
        static_cast<void>(nbody::visualization::make_astronomical_catalog_scenario(
            {"sol:sun", "sol:sun"}));
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    bool unknown_rejected = false;
    try {
        static_cast<void>(nbody::visualization::make_astronomical_catalog_scenario(
            {"sol:sun", "sol:not-a-body"}));
    } catch (const std::invalid_argument&) {
        unknown_rejected = true;
    }
    require(duplicate_rejected && unknown_rejected,
            "invalid stable-ID catalog selections were admitted");
}

void test_local_render_precision_and_visibility() {
    nbody::SimulationState state;
    state.bodies = {
        {"Parent", 1.0e24, {1.0e16, -2.0e16, 3.0e16}, {}},
        {"Child", 1.0e20, {1.0e16 + 4096.0, -2.0e16 + 2048.0, 3.0e16 - 1024.0}, {}},
    };
    const auto original = state;
    nbody::visualization::RenderSpaceTransform transform(state);
    transform.configure_view(state, state.bodies[0].position_m, {0U, 1U}, 8.0F);
    std::vector<nbody::visualization::RenderBody> snapshot;
    transform.update_snapshot(state, snapshot);
    require(snapshot[0].position == glm::vec3(0.0F),
            "local render origin did not map exactly to zero");
    const glm::vec3 expected = transform.to_render_offset(
        state.bodies[1].position_m - state.bodies[0].position_m);
    require(snapshot[1].position == expected && glm::length(expected) > 0.0F,
            "double-precision subtraction lost local child separation");
    snapshot[1].visible = false;
    require(!snapshot[1].visible && same_state(state, original),
            "render visibility altered gravitational state");

    nbody::SimulationState translated = state;
    const nbody::Vec3 translation{8.0e15, 5.0e15, -9.0e15};
    for (auto& body : translated.bodies) {
        body.position_m += translation;
    }
    transform.configure_view(
        translated, translated.bodies[0].position_m, {0U, 1U}, 8.0F);
    std::vector<nbody::visualization::RenderBody> translated_snapshot;
    transform.update_snapshot(translated, translated_snapshot);
    require(glm::distance(translated_snapshot[1].position, expected) < 1.0e-6F,
            "local render geometry changed under a common SSB translation");
}

void test_composed_visibility_policy() {
    const auto scenario = nbody::visualization::make_astronomical_catalog_scenario();
    const auto state_before = scenario.initial_state;
    const std::size_t earth = required_index(scenario, "sol:earth");
    const std::size_t moon = required_index(scenario, "sol:moon");
    const std::size_t jupiter = required_index(scenario, "sol:jupiter");
    const std::size_t io = required_index(scenario, "sol:io");
    std::vector<bool> profile(scenario.initial_state.bodies.size(), false);
    std::vector<bool> individual(scenario.initial_state.bodies.size(), true);
    nbody::visualization::BodyClassVisibility classes;
    classes.fill(true);

    nbody::visualization::RenderSpaceTransform transform(
        scenario.initial_state, scenario.styles);
    std::vector<nbody::visualization::RenderBody> snapshot;
    transform.update_snapshot(scenario.initial_state, snapshot);

    profile[earth] = true;
    profile[moon] = true;
    nbody::visualization::apply_effective_body_visibility(
        scenario, profile, individual, classes, snapshot);
    require(snapshot[earth].visible && snapshot[moon].visible &&
                !snapshot[jupiter].visible && !snapshot[io].visible,
            "Earth-Moon profile did not restrict effective visibility");

    const std::size_t satellite_class = nbody::visualization::body_class_visibility_index(
        nbody::astronomy::AstronomicalBodyClass::natural_satellite);
    classes[satellite_class] = false;
    nbody::visualization::apply_effective_body_visibility(
        scenario, profile, individual, classes, snapshot);
    require(snapshot[earth].visible && !snapshot[moon].visible &&
                !snapshot[io].visible,
            "satellite class filter did not compose with subsystem membership");
    classes[satellite_class] = true;
    nbody::visualization::apply_effective_body_visibility(
        scenario, profile, individual, classes, snapshot);
    require(snapshot[moon].visible && !snapshot[io].visible,
            "re-enabling a class revealed an out-of-profile body");

    individual[moon] = false;
    nbody::visualization::apply_effective_body_visibility(
        scenario, profile, individual, classes, snapshot);
    require(snapshot[earth].visible && !snapshot[moon].visible,
            "individual visibility did not remain independent within the active profile");
    individual[moon] = true;

    std::fill(profile.begin(), profile.end(), false);
    profile[jupiter] = true;
    profile[io] = true;
    nbody::visualization::apply_effective_body_visibility(
        scenario, profile, individual, classes, snapshot);
    require(!snapshot[earth].visible && !snapshot[moon].visible &&
                snapshot[jupiter].visible && snapshot[io].visible,
            "switching subsystem profiles retained stale profile membership");

    std::fill(profile.begin(), profile.end(), true);
    nbody::visualization::apply_effective_body_visibility(
        scenario, profile, individual, classes, snapshot);
    require(snapshot[earth].visible && snapshot[moon].visible &&
                snapshot[jupiter].visible && snapshot[io].visible,
            "overview profile did not restore bodies allowed by user filters");
    require(same_state(state_before, scenario.initial_state) &&
                scenario.initial_state.bodies.size() == 60U,
            "visual filtering changed solver state or catalog body count");
}

void test_parent_relative_trails() {
    std::deque<nbody::Vec3> parent{{100.0, 50.0, -20.0}, {105.0, 48.0, -19.0}};
    std::deque<nbody::Vec3> child{{110.0, 47.0, -18.0}, {116.0, 44.0, -16.0}};
    const auto relative = nbody::visualization::parent_relative_trail(child, parent);
    require(relative.size() == 2U && relative[0].x == 10.0 && relative[0].y == -3.0 &&
                relative[1].x == 11.0 && relative[1].y == -4.0,
            "parent-relative trails did not subtract matching historical parent samples");
    bool mismatch_rejected = false;
    try {
        parent.pop_back();
        static_cast<void>(nbody::visualization::parent_relative_trail(child, parent));
    } catch (const std::invalid_argument&) {
        mismatch_rejected = true;
    }
    require(mismatch_rejected, "mismatched parent trail history was silently accepted");
}

void test_catalog_backend_equivalence_and_timestep_convergence() {
    const auto scenario = nbody::visualization::make_astronomical_catalog_scenario();
    nbody::visualization::SimulationController sequential(
        scenario.initial_state,
        {scenario.dt_s, 1, nbody::visualization::BackendKind::sequential, 1});
    sequential.set_paused(true);
    const auto paused = sequential.state();
    sequential.advance_frame();
    require(same_state(paused, sequential.state()),
            "paused catalog scenario advanced unexpectedly");
    sequential.advance_steps(1);
    require(sequential.state().time_s == scenario.dt_s,
            "manual catalog step did not advance exactly one fixed timestep");
    sequential.reset();
    require(same_state(scenario.initial_state, sequential.state()),
            "catalog reset did not restore the exact sourced initial state");
    sequential.advance_steps(24);
    require(sequential.state().time_s == 24.0 * scenario.dt_s,
            "catalog sequential batching advanced the wrong simulated duration");
    if (nbody::visualization::SimulationController::openmp_available()) {
        nbody::visualization::SimulationController openmp(
            scenario.initial_state,
            {scenario.dt_s, 1, nbody::visualization::BackendKind::openmp, 4});
        openmp.advance_steps(24);
        double maximum_position_m = 0.0;
        const double maximum_scaled = maximum_parent_relative_difference(
            scenario, sequential.state(), openmp.state(), maximum_position_m);
        double maximum_velocity_m_s = 0.0;
        double maximum_position_scale = 1.0;
        double maximum_velocity_scale = 1.0;
        double maximum_absolute_state_position_m = 0.0;
        for (std::size_t index = 0; index < scenario.initial_state.bodies.size(); ++index) {
            maximum_absolute_state_position_m = std::max(
                maximum_absolute_state_position_m,
                nbody::norm(sequential.state().bodies[index].position_m -
                            openmp.state().bodies[index].position_m));
            maximum_velocity_m_s = std::max(
                maximum_velocity_m_s,
                nbody::norm(sequential.state().bodies[index].velocity_m_s -
                            openmp.state().bodies[index].velocity_m_s));
            maximum_position_scale = std::max(
                maximum_position_scale,
                nbody::norm(sequential.state().bodies[index].position_m));
            maximum_velocity_scale = std::max(
                maximum_velocity_scale,
                nbody::norm(sequential.state().bodies[index].velocity_m_s));
        }
        require(maximum_position_m <= 1.0e-5 || maximum_scaled <= 1.0e-13,
                "catalog OpenMP state exceeded the established mixed position tolerance");
        require(maximum_absolute_state_position_m <= 1.0e-6 ||
                    maximum_absolute_state_position_m / maximum_position_scale <= 1.0e-13,
                "catalog OpenMP absolute state position exceeded mixed tolerance");
        require(maximum_velocity_m_s <= 1.0e-9 ||
                    maximum_velocity_m_s / maximum_velocity_scale <= 1.0e-13,
                "catalog OpenMP velocity exceeded mixed tolerance");
    }

    const auto subset = nbody::visualization::make_astronomical_catalog_scenario(
        {"sol:sun", "sol:earth", "sol:moon"});
    sequential.replace_scenario(subset.initial_state, subset.dt_s, 10);
    require(sequential.paused() && sequential.state().bodies.size() == 3U &&
                sequential.state().time_s == 0.0 && sequential.steps_per_frame() == 10U,
            "paused catalog subset replacement changed pause state or controller state");
    const auto paused_subset = sequential.state();
    sequential.advance_frame();
    require(same_state(paused_subset, sequential.state()),
            "paused controller advanced after catalog subset replacement");
    sequential.advance_steps(3);
    require(sequential.state().time_s == 3.0 * subset.dt_s,
            "explicit manual advancement while paused used incorrect step semantics");
    sequential.reset();
    require(sequential.paused() && same_state(subset.initial_state, sequential.state()),
            "reset after catalog subset replacement lost pause or physical composition");

    sequential.set_paused(false);
    sequential.replace_scenario(scenario.initial_state, scenario.dt_s, 1);
    require(!sequential.paused() && same_state(scenario.initial_state, sequential.state()) &&
                sequential.state().time_s == 0.0,
            "running scenario replacement changed run state or sourced initial state");

    constexpr double horizon_s = 86400.0;
    double minimum_clearance_ratio = std::numeric_limits<double>::infinity();
    std::string minimum_clearance_pair;
    const auto update_clearance = [&](const nbody::SimulationState& state) {
        for (std::size_t left = 0; left < state.bodies.size(); ++left) {
            for (std::size_t right = left + 1; right < state.bodies.size(); ++right) {
                const double combined_radius = scenario.styles[left].physical_radius_m +
                    scenario.styles[right].physical_radius_m;
                const double ratio = nbody::norm(
                    state.bodies[left].position_m - state.bodies[right].position_m) /
                    combined_radius;
                if (ratio < minimum_clearance_ratio) {
                    minimum_clearance_ratio = ratio;
                    minimum_clearance_pair = scenario.stable_ids[left] + "/" +
                        scenario.stable_ids[right];
                }
            }
        }
    };
    auto integrate = [&](double dt_s, bool track_clearance = false) {
        auto state = scenario.initial_state;
        const std::size_t steps = static_cast<std::size_t>(horizon_s / dt_s);
        if (track_clearance) {
            update_clearance(state);
        }
        for (std::size_t step = 0; step < steps; ++step) {
            nbody::velocity_verlet_step(state, dt_s);
            if (track_clearance) {
                update_clearance(state);
            }
        }
        return state;
    };
    const auto coarse = integrate(300.0);
    const auto fine = integrate(150.0);
    const auto finest = integrate(75.0);
    const auto policy = integrate(60.0, true);
    const auto policy_reference = integrate(30.0);
    double coarse_absolute_m = 0.0;
    double fine_absolute_m = 0.0;
    std::string coarse_worst_body;
    std::string fine_worst_body;
    const double coarse_scaled = maximum_parent_relative_difference(
        scenario, coarse, fine, coarse_absolute_m, &coarse_worst_body);
    const double fine_scaled = maximum_parent_relative_difference(
        scenario, fine, finest, fine_absolute_m, &fine_worst_body);
    double policy_absolute_m = 0.0;
    std::string policy_worst_body;
    const double policy_scaled = maximum_parent_relative_difference(
        scenario, policy, policy_reference, policy_absolute_m, &policy_worst_body);
    nbody::visualization::ConservationTracker coarse_conservation(scenario.initial_state);
    nbody::visualization::ConservationTracker finest_conservation(scenario.initial_state);
    const auto coarse_drift = coarse_conservation.update(coarse);
    const auto finest_drift = finest_conservation.update(finest);
    std::cout << std::setprecision(10)
              << "catalog timestep assessment horizon_s=" << horizon_s
              << " coarse_300_vs_150_max_parent_relative=" << coarse_scaled
              << " coarse_worst_body=" << coarse_worst_body
              << " coarse_max_absolute_m=" << coarse_absolute_m
              << " fine_150_vs_75_max_parent_relative=" << fine_scaled
              << " fine_worst_body=" << fine_worst_body
              << " fine_max_absolute_m=" << fine_absolute_m
              << " policy_60_vs_30_max_parent_relative=" << policy_scaled
              << " policy_worst_body=" << policy_worst_body
              << " policy_max_absolute_m=" << policy_absolute_m
              << " minimum_center_distance_over_combined_radii=" << minimum_clearance_ratio
              << " minimum_clearance_pair=" << minimum_clearance_pair
              << " energy_drift_300=" << coarse_drift.energy_relative
              << " energy_drift_75=" << finest_drift.energy_relative << '\n';
    require(std::isfinite(coarse_scaled) && std::isfinite(fine_scaled) &&
                fine_scaled < coarse_scaled && fine_absolute_m < coarse_absolute_m,
            "300/150/75 s catalog integrations did not exhibit timestep convergence");
    require(policy_scaled < 2.0e-3 &&
                minimum_clearance_ratio > 1.0 &&
                std::isfinite(coarse_drift.energy_relative) &&
                std::isfinite(finest_drift.energy_relative),
            "one-day 60 s policy convergence or conservation result is not numerically usable");
    require(!scenario.timestep_validation_complete,
            "bounded convergence evidence was incorrectly promoted to ephemeris validation");
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

void test_planet_trail_geometry_fidelity() {
    const auto solar = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    auto state = solar.initial_state;
    nbody::visualization::TrailHistory trails(state, 128, 86400.0);
    for (int step = 0; step < 352; ++step) {
        nbody::velocity_verlet_step(state, solar.dt_s);
        trails.sample(state);
    }
    const auto& sun = trails.positions()[0];
    const auto& mercury = trails.positions()[1];
    require(sun.size() == mercury.size() && mercury.size() >= 88,
            "Mercury trail did not retain one orbit at daily resolution");

    nbody::visualization::RenderSpaceTransform transform(
        solar.initial_state, solar.styles, 9.0F, solar.default_radius_multiplier);
    double physical_minimum = std::numeric_limits<double>::infinity();
    double physical_maximum = 0.0;
    float rendered_minimum = std::numeric_limits<float>::infinity();
    float rendered_maximum = 0.0F;
    for (std::size_t sample = 0; sample < mercury.size(); ++sample) {
        const double physical_radius = nbody::norm(mercury[sample] - sun[sample]);
        const float rendered_radius = glm::length(
            transform.to_render_position(mercury[sample]) -
            transform.to_render_position(sun[sample]));
        const float expected_rendered =
            static_cast<float>(physical_radius) * transform.position_scale();
        require(std::abs(rendered_radius - expected_rendered) <=
                    std::max(1.0e-6F, expected_rendered * 2.0e-5F),
                "trail render preparation changed heliocentric radius");
        physical_minimum = std::min(physical_minimum, physical_radius);
        physical_maximum = std::max(physical_maximum, physical_radius);
        rendered_minimum = std::min(rendered_minimum, rendered_radius);
        rendered_maximum = std::max(rendered_maximum, rendered_radius);
    }
    require(physical_maximum / physical_minimum > 1.45 &&
                std::abs(rendered_maximum / rendered_minimum -
                         static_cast<float>(physical_maximum / physical_minimum)) < 1.0e-4F,
            "trail storage/render conversion circularized Mercury's radial variation");
}

void test_inner_system_guide_projection() {
    constexpr float scene_width = 1920.0F - 446.0F;
    constexpr float viewport_height = 1080.0F;
    const auto solar = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    nbody::visualization::RenderSpaceTransform transform(
        solar.initial_state, solar.styles, 9.0F, solar.default_radius_multiplier);
    std::vector<nbody::visualization::RenderBody> snapshot;
    transform.update_snapshot(solar.initial_state, snapshot);
    const auto inner_request = nbody::visualization::camera_framing(
        nbody::visualization::CameraPreset::inner_system, solar, snapshot);
    transform.configure_view(
        solar.initial_state,
        solar.initial_state.bodies[*inner_request.local_origin_body].position_m,
        inner_request.visible_bodies,
        8.0F);
    transform.update_snapshot(solar.initial_state, snapshot);
    nbody::visualization::OrbitCamera camera;
    nbody::visualization::apply_camera_framing(
        camera,
        nbody::visualization::camera_framing(
            nbody::visualization::CameraPreset::inner_system,
            solar,
            snapshot));
    camera.set_aspect_ratio(scene_width / viewport_height);

    const auto project = [&](const glm::vec3& point) {
        const glm::vec4 clip = camera.projection_matrix() * camera.view_matrix() *
            glm::vec4(point, 1.0F);
        require(clip.w > 0.0F, "Inner System guide point projected behind camera");
        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        return glm::vec2(
            (ndc.x * 0.5F + 0.5F) * scene_width,
            (-ndc.y * 0.5F + 0.5F) * viewport_height);
    };

    std::vector<glm::vec2> projected;
    projected.reserve(solar.reference_paths_m[1].size());
    glm::vec2 mean{0.0F};
    for (const auto& point : solar.reference_paths_m[1]) {
        const glm::vec2 screen = project(transform.to_render_position(point));
        projected.push_back(screen);
        mean += screen;
    }
    mean /= static_cast<float>(projected.size());
    double covariance_xx = 0.0;
    double covariance_xy = 0.0;
    double covariance_yy = 0.0;
    for (const auto& point : projected) {
        const glm::vec2 delta = point - mean;
        covariance_xx += static_cast<double>(delta.x) * delta.x;
        covariance_xy += static_cast<double>(delta.x) * delta.y;
        covariance_yy += static_cast<double>(delta.y) * delta.y;
    }
    covariance_xx /= projected.size();
    covariance_xy /= projected.size();
    covariance_yy /= projected.size();
    const double trace = covariance_xx + covariance_yy;
    const double discriminant = std::sqrt(
        std::max(0.0, (covariance_xx - covariance_yy) *
                          (covariance_xx - covariance_yy) +
                          4.0 * covariance_xy * covariance_xy));
    const double major_variance = 0.5 * (trace + discriminant);
    const double minor_variance = 0.5 * (trace - discriminant);
    const double apparent_axis_ratio = std::sqrt(minor_variance / major_variance);
    const double apparent_major_radius = std::sqrt(2.0 * major_variance);
    const double apparent_minor_radius = std::sqrt(2.0 * minor_variance);
    const glm::vec2 sun_screen = project(snapshot[0].position);
    const double focus_offset_pixels = glm::distance(sun_screen, mean);
    require(apparent_axis_ratio > 0.0 && apparent_axis_ratio <= 1.0 &&
                focus_offset_pixels > 1.0,
            "Inner System projection lost Mercury guide shape/focus offset");
    std::cout << "Inner System Mercury projection: covariance_axis_ratio="
              << apparent_axis_ratio << " Sun_to_guide_centroid_px="
              << focus_offset_pixels << " apparent_major_radius_px="
              << apparent_major_radius << " apparent_minor_radius_px="
              << apparent_minor_radius << '\n';
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

    const auto narrow_panel = nbody::visualization::panel_width_policy(800.0F, 900.0F);
    const auto wide_panel = nbody::visualization::panel_width_policy(1920.0F, 520.0F);
    require(narrow_panel.width == narrow_panel.maximum && narrow_panel.minimum == 380.0F &&
                wide_panel.width == 520.0F && wide_panel.maximum == 680.0F,
            "responsive panel width policy escaped its declared constraints");
    require(nbody::visualization::format_measurement(5.9722e24, "kg", 6).find("kg") !=
                std::string::npos,
            "measurement formatting omitted its complete unit");
    for (std::size_t topic = 0;
         topic < static_cast<std::size_t>(nbody::visualization::HelpTopic::count); ++topic) {
        require(std::string(nbody::visualization::help_text(
                    static_cast<nbody::visualization::HelpTopic>(topic))).size() > 20U,
                "tooltip registry contains an incomplete entry");
    }

    const auto solar = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    nbody::visualization::RenderSpaceTransform transform(
        solar.initial_state, solar.styles, 9.0F, solar.default_radius_multiplier);
    std::vector<nbody::visualization::RenderBody> bodies;
    transform.update_snapshot(solar.initial_state, bodies);
    const auto earth = nbody::visualization::camera_framing(
        nbody::visualization::CameraPreset::earth_moon,
        solar,
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
    require(nbody::visualization::astronomical_label_priority(
                nbody::astronomy::AstronomicalBodyClass::planet, false, 8U) >
            nbody::visualization::astronomical_label_priority(
                nbody::astronomy::AstronomicalBodyClass::natural_satellite,
                false, 1U) &&
        nbody::visualization::astronomical_label_priority(
                nbody::astronomy::AstronomicalBodyClass::asteroid, true, 59U) ==
            1000,
        "full-system labels do not prioritize major or selected bodies");
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

    std::vector<nbody::visualization::LabelCandidate> inner_candidates;
    for (std::size_t index = 0; index < 5; ++index) {
        inner_candidates.push_back({
            index,
            {620.0F + static_cast<float>(index), 350.0F},
            {72.0F, 24.0F},
            index == 2 ? 1000 : 900 - static_cast<int>(index),
            index == 2,
        });
    }
    const auto callouts = nbody::visualization::stable_inner_system_callouts(
        inner_candidates, {1280.0F, 720.0F}, 420.0F);
    require(callouts.size() == 5U && callouts[2].selected,
            "stable inner callouts omitted a body or selected-body priority");
    auto moved_inner = inner_candidates;
    for (auto& candidate : moved_inner) {
        candidate.anchor += glm::vec2(0.5F, -0.25F);
    }
    const auto moved_callouts = nbody::visualization::stable_inner_system_callouts(
        moved_inner, {1280.0F, 720.0F}, 420.0F);
    for (std::size_t index = 0; index < callouts.size(); ++index) {
        require(callouts[index].body_index == moved_callouts[index].body_index &&
                    callouts[index].position == moved_callouts[index].position &&
                    moved_callouts[index].anchor == moved_inner[index].anchor,
                "inner callout slot or body association changed under small motion");
        require(callouts[index].position.x >= 428.0F && callouts[index].position.y >= 8.0F &&
                    callouts[index].position.x + callouts[index].size.x <= 1272.0F &&
                    callouts[index].position.y + callouts[index].size.y <= 712.0F,
                "inner callout escaped viewport containment");
        if (index > 0) {
            require(callouts[index - 1].position.y + callouts[index - 1].size.y <=
                        callouts[index].position.y,
                    "stable inner callouts overlap");
        }
    }
}

void test_compute_experiment_scaling_and_comparison() {
    // Captured independently from the committed nbody_large_n target using its
    // public generator and the exact field-wise checksum below.
    constexpr std::array<std::array<std::uint64_t, 3>, 4> expected_checksums{{
        {{7825184865471401019ULL, 6917549025945125182ULL, 13520912006326907165ULL}},
        {{4564111068250750988ULL, 4745212845247103561ULL, 1790368443155452445ULL}},
        {{8111121325248395463ULL, 14334474822168884208ULL, 6706342535779300325ULL}},
        {{13542330320252189013ULL, 1315021589615441400ULL, 2429501922212237394ULL}},
    }};
    std::array<std::array<std::uint64_t, 3>, 4> observed_checksums{};
    std::size_t count_index = 0U;
    for (const std::size_t count : nbody::visualization::compute_body_counts) {
        std::size_t distribution_index = 0U;
        for (const auto distribution : {
                 nbody::LargeNDistribution::uniform,
                 nbody::LargeNDistribution::clustered,
                 nbody::LargeNDistribution::planar}) {
            const auto first = nbody::visualization::make_visual_scenario(
                nbody::visualization::ScenarioKind::synthetic, count, 24301, distribution);
            const auto second = nbody::visualization::make_visual_scenario(
                nbody::visualization::ScenarioKind::synthetic, count, 24301, distribution);
            require(first.initial_state.bodies.size() == count &&
                        same_state(first.initial_state, second.initial_state),
                    "supported synthetic scene is not deterministic");
            require(first.stable_ids.size() == count &&
                        first.synthetic_distribution == distribution &&
                        first.synthetic_seed == 24301U,
                    "synthetic scene identity metadata is incomplete");
            for (std::size_t index = 0; index < count; ++index) {
                require(first.stable_ids[index] == second.stable_ids[index] &&
                            first.stable_ids[index] ==
                                "synthetic:" + first.initial_state.bodies[index].name,
                        "synthetic stable identity changed with regeneration");
            }
            observed_checksums[count_index][distribution_index] =
                state_checksum(first.initial_state);
            require(observed_checksums[count_index][distribution_index] ==
                        expected_checksums[count_index][distribution_index],
                    "viewer scene differs from the independently captured committed Large-N generator state");
            nbody::visualization::RenderSpaceTransform transform(
                first.initial_state, first.styles, 9.0F, first.default_radius_multiplier);
            std::vector<nbody::visualization::RenderBody> snapshot;
            transform.update_snapshot(first.initial_state, snapshot);
            require(snapshot.size() == count &&
                        nbody::visualization::render_resource_admitted(snapshot.size()),
                    "bounded synthetic render snapshot is incomplete or unadmitted");
            ++distribution_index;
        }
        ++count_index;
    }

    const auto uniform = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::synthetic, 256, 24301,
        nbody::LargeNDistribution::uniform);
    const auto clustered = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::synthetic, 256, 24301,
        nbody::LargeNDistribution::clustered);
    const auto planar = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::synthetic, 256, 24301,
        nbody::LargeNDistribution::planar);
    require(!same_state(uniform.initial_state, clustered.initial_state) &&
                !same_state(uniform.initial_state, planar.initial_state) &&
                uniform.stable_ids == clustered.stable_ids &&
                uniform.stable_ids == planar.stable_ids,
            "synthetic distributions are not distinct or changed stable identities");
    require(state_checksum(uniform.initial_state) == 7825184865471401019ULL &&
                state_checksum(clustered.initial_state) == 6917549025945125182ULL &&
                state_checksum(planar.initial_state) == 13520912006326907165ULL,
            "viewer adapter diverged from the committed Large-N generator contract");

    bool rejected_invalid_distribution = false;
    try {
        (void)nbody::visualization::make_visual_scenario(
            nbody::visualization::ScenarioKind::synthetic, 256, 24301,
            static_cast<nbody::LargeNDistribution>(99));
    } catch (const std::invalid_argument&) {
        rejected_invalid_distribution = true;
    }
    require(rejected_invalid_distribution,
        "invalid synthetic distribution was not rejected explicitly");

    bool rejected_high_count = false;
    try {
        (void)nbody::visualization::make_visual_scenario(
            nbody::visualization::ScenarioKind::synthetic, 16384, 24301);
    } catch (const std::invalid_argument&) {
        rejected_high_count = true;
    }
    require(rejected_high_count &&
                !nbody::visualization::render_resource_admitted(8193) &&
                nbody::visualization::render_resource_admitted(8192),
            "synthetic scene or renderer admission exceeded the G1 bound");

    const std::array backends{
        nbody::visualization::WorkloadBackend::sequential,
        nbody::visualization::WorkloadBackend::openmp,
        nbody::visualization::WorkloadBackend::cuda,
    };
    const std::array solvers{
        nbody::visualization::WorkloadSolver::direct,
        nbody::visualization::WorkloadSolver::barnes_hut,
    };
    for (const std::size_t count : nbody::visualization::compute_body_counts) {
        for (const auto backend : backends) {
            for (const auto solver : solvers) {
                const auto policy = nbody::visualization::interactive_workload_policy(
                    count, backend, solver);
                const bool cuda_barnes_hut =
                    backend == nbody::visualization::WorkloadBackend::cuda &&
                    solver == nbody::visualization::WorkloadSolver::barnes_hut;
                const bool expected_continuous = count <= 1024U && !cuda_barnes_hut;
                const bool expected_manual = expected_continuous ||
                    (count == 4096U &&
                     backend != nbody::visualization::WorkloadBackend::cuda &&
                     solver == nbody::visualization::WorkloadSolver::barnes_hut) ||
                    (count == 8192U &&
                     backend == nbody::visualization::WorkloadBackend::openmp &&
                     solver == nbody::visualization::WorkloadSolver::barnes_hut);
                require(policy.continuous_advancement == expected_continuous &&
                            policy.manual_single_step == expected_manual &&
                            (expected_manual || policy.explanation[0] != '\0'),
                        "backend/solver/count workload admission matrix is incorrect");
            }
        }
    }
    const auto prohibited = nbody::visualization::interactive_workload_policy(
        16384, nbody::visualization::WorkloadBackend::cuda,
        nbody::visualization::WorkloadSolver::direct);
    require(!prohibited.continuous_advancement && !prohibited.manual_single_step &&
                prohibited.explanation[0] != '\0',
            "unsupported workload did not fail with an explicit reason");

    require(nbody::visualization::checked_render_upload_bytes(8192U, 32U) ==
                262144U,
            "checked render upload byte count is incorrect");
    bool rejected_upload_overflow = false;
    try {
        (void)nbody::visualization::checked_render_upload_bytes(
            std::numeric_limits<std::size_t>::max(), 2U);
    } catch (const std::overflow_error&) {
        rejected_upload_overflow = true;
    }
    require(rejected_upload_overflow,
        "render upload byte-count overflow was not rejected");
    require(nbody::visualization::render_upload_allows_draw(0U) &&
                !nbody::visualization::render_upload_allows_draw(0x0505U),
            "failed OpenGL upload was not prevented from reaching the draw path");

    const auto prior = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::solar_system);
    nbody::visualization::CounterfactualPublicationGate gate(7U);
    gate.accept_job(3U, 7U);
    const auto prior_generation = gate.scenario_generation();
    const auto prior_worker = gate.accepted_worker_generation();
    try {
        (void)nbody::visualization::prepare_visual_scenario_replacement(
            nbody::visualization::ScenarioKind::synthetic, 256U, 24301U,
            static_cast<nbody::LargeNDistribution>(99));
        require(false, "invalid replacement preparation unexpectedly succeeded");
    } catch (const std::invalid_argument&) {
    }
    require(prior.initial_state.bodies.size() == 9U &&
                gate.scenario_generation() == prior_generation &&
                gate.accepted_worker_generation() == prior_worker,
            "failed replacement preparation mutated active scene or worker generation");

    auto prepared = nbody::visualization::prepare_visual_scenario_replacement(
        nbody::visualization::ScenarioKind::synthetic, 1024U, 24301U,
        nbody::LargeNDistribution::planar);
    require(prepared.scenario.initial_state.bodies.size() == 1024U &&
                prepared.profile_visibility.size() == 1024U &&
                prepared.individual_visibility.size() == 1024U,
            "valid replacement preparation is incomplete");
    gate.replace_scenario();
    require(gate.scenario_generation() == prior_generation + 1U &&
                !gate.accepted_worker_generation().has_value(),
            "successful prepared replacement did not invalidate stale worker identity");

    if (nbody::visualization::SimulationController::openmp_available()) {
        const auto large = nbody::visualization::make_visual_scenario(
            nbody::visualization::ScenarioKind::synthetic, 8192, 24301,
            nbody::LargeNDistribution::clustered);
        nbody::visualization::SimulationController large_controller(
            large.initial_state,
            {large.dt_s, 1, nbody::visualization::BackendKind::openmp, 4, 0,
             nbody::visualization::SolverKind::barnes_hut, {}});
        large_controller.set_paused(true);
        const auto started = std::chrono::steady_clock::now();
        large_controller.advance_steps(1);
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        require(large_controller.paused() &&
                    large_controller.state().time_s == large.dt_s &&
                    large_controller.state().bodies.size() == 8192U &&
                    large_controller.barnes_hut_statistics().has_value(),
                "admitted 8192-body OpenMP Barnes-Hut manual step was not complete");
        large_controller.reset();
        require(large_controller.paused() &&
                    same_state(large.initial_state, large_controller.state()),
                "8192-body manual-step reset did not restore the selected scene");
        std::cout << "live diagnostic (not benchmark evidence): N=8192 OpenMP-4 "
                  << "Barnes-Hut manual_step_ms=" << elapsed_ms << '\n';
    }

    const auto initial = nbody::visualization::make_visual_scenario(
        nbody::visualization::ScenarioKind::synthetic, 256, 24301);
    const auto visible_state = initial.initial_state;
    if (nbody::visualization::SimulationController::openmp_available()) {
        const auto comparison = nbody::visualization::run_live_compute_comparison(
            initial.initial_state, initial.dt_s, 4);
        require(comparison.body_count == 256 && comparison.integration_steps > 0 &&
                    comparison.sequential_ms > 0.0 && comparison.openmp_ms > 0.0,
                "live CPU comparison did not execute the declared fixed workload");
        require(comparison.correctness_passed,
                "live Sequential/OpenMP comparison failed established mixed tolerances");
        require(nbody::visualization::live_comparison_matches(comparison, 256, 4) &&
                    !nbody::visualization::live_comparison_matches(comparison, 1024, 4) &&
                    !nbody::visualization::live_comparison_matches(comparison, 256, 2),
                "live comparison result did not invalidate for N/thread changes");
        require(same_state(visible_state, initial.initial_state),
                "live CPU comparison mutated its visible input state");
        std::cout << "live diagnostic (not benchmark evidence): N=256 steps="
                  << comparison.integration_steps
                  << " sequential_ms=" << comparison.sequential_ms
                  << " openmp4_ms=" << comparison.openmp_ms
                  << " ratio=" << comparison.sequential_to_openmp_ratio << '\n';
    }
}

void test_application_transaction_and_timing_provenance() {
    using namespace nbody::visualization;

    const auto solar = make_visual_scenario(ScenarioKind::solar_system);
    SimulationController controller(
        solar.initial_state,
        {solar.dt_s, 1U, BackendKind::sequential, 1, 0,
         SolverKind::direct, {}});
    controller.set_paused(true);
    controller.advance_steps(1U);
    ScenarioDefinition active = solar;
    RenderSpaceTransform render_space(
        active.initial_state, active.styles, 9.0F,
        active.default_radius_multiplier);
    render_space.set_origin_m({11.0, 22.0, 33.0});
    std::vector<bool> profile(active.initial_state.bodies.size(), true);
    std::vector<bool> individual(active.initial_state.bodies.size(), true);
    individual[2] = false;
    CounterfactualPublicationGate gate(7U);
    gate.accept_job(3U, 7U);
    CounterfactualExperimentWorker worker;
    CounterfactualWorkerPublication publication;
    publication.generation = 3U;
    publication.state = CounterfactualWorkerState::running;
    publication.scenario_id = active.name;
    auto run = std::make_shared<CounterfactualRunDescriptor>();
    run->scenario_generation = 7U;
    run->worker_generation = 3U;
    run->scenario_id = active.name;
    publication.run = run;
    auto accepted_result =
        std::make_shared<nbody::experiments::CounterfactualExperimentResult>();
    accepted_result->issues.push_back({});
    publication.result = accepted_result;
    require(gate.accepts(publication),
        "transaction fixture did not begin with an accepted publication");
    CounterfactualPresentationCache presentation;
    presentation.generation = 3U;
    presentation.result = accepted_result.get();
    presentation.result_rows.push_back({});
    presentation.error = "retained presentation marker";
    presentation.frame_error = "retained frame marker";
    presentation.warning_groups.push_back({});
    SimulationTimingTracker timing;
    const auto timing_generation = timing.generation();
    require(timing.publish(timing_generation, 12.5, 80.0),
        "transaction fixture timing publication failed");

    const nbody::SimulationState committed_state = controller.state();
    const std::string committed_scenario_name = active.name;
    const auto committed_stable_ids = active.stable_ids;
    const auto committed_body_metadata = active.body_metadata;
    const double committed_dt = controller.dt_s();
    const std::size_t committed_steps_per_frame = controller.steps_per_frame();
    const nbody::Vec3 committed_origin = render_space.origin_m();
    const float committed_position_scale = render_space.position_scale();
    const float committed_radius_multiplier = render_space.radius_multiplier();
    const auto committed_profile = profile;
    const auto committed_individual = individual;
    const auto committed_publication = publication;
    const std::uint64_t committed_gate_generation = gate.scenario_generation();
    const auto committed_worker_generation = gate.accepted_worker_generation();

    const auto require_failed_preparation_preserved_state =
        [&](const char* failure_kind) {
            bool metadata_unchanged =
                active.body_metadata.size() == committed_body_metadata.size();
            for (std::size_t index = 0;
                 metadata_unchanged && index < active.body_metadata.size(); ++index) {
                const auto& observed = active.body_metadata[index];
                const auto& expected = committed_body_metadata[index];
                metadata_unchanged = observed.stable_id == expected.stable_id &&
                    observed.parent_stable_id == expected.parent_stable_id &&
                    observed.body_class == expected.body_class &&
                    observed.counterfactual_removal_eligible ==
                        expected.counterfactual_removal_eligible;
            }
            require(same_state(controller.state(), committed_state) &&
                        controller.dt_s() == committed_dt &&
                        controller.steps_per_frame() == committed_steps_per_frame &&
                        controller.backend_kind() == BackendKind::sequential &&
                        controller.solver_kind() == SolverKind::direct &&
                        active.name == committed_scenario_name &&
                        active.stable_ids == committed_stable_ids &&
                        metadata_unchanged &&
                        render_space.origin_m().x == committed_origin.x &&
                        render_space.origin_m().y == committed_origin.y &&
                        render_space.origin_m().z == committed_origin.z &&
                        render_space.position_scale() == committed_position_scale &&
                        render_space.radius_multiplier() ==
                            committed_radius_multiplier &&
                        profile == committed_profile &&
                        individual == committed_individual &&
                        gate.scenario_generation() == committed_gate_generation &&
                        gate.accepted_worker_generation() == committed_worker_generation &&
                        publication.generation == committed_publication.generation &&
                        publication.run == committed_publication.run &&
                        publication.result == committed_publication.result &&
                        presentation.generation == 3U &&
                        presentation.result == accepted_result.get() &&
                        presentation.result_rows.size() == 1U &&
                        presentation.error == "retained presentation marker" &&
                        presentation.frame_error == "retained frame marker" &&
                        presentation.warning_groups.size() == 1U &&
                        timing.generation() == timing_generation &&
                        timing.sample() &&
                        timing.sample()->batch_milliseconds == 12.5,
                    std::string(failure_kind) +
                        " mutated committed application transaction state");
        };

    bool rejected_invalid_distribution = false;
    try {
        (void)prepare_application_scenario_transaction(
            ScenarioKind::synthetic, 256U, 24301U,
            static_cast<nbody::LargeNDistribution>(99), {}, controller, gate);
    } catch (const std::invalid_argument&) {
        rejected_invalid_distribution = true;
    }
    require(rejected_invalid_distribution,
        "production transaction accepted an invalid distribution");
    require_failed_preparation_preserved_state("invalid distribution");

    bool rejected_unsupported_count = false;
    try {
        (void)prepare_application_scenario_transaction(
            ScenarioKind::synthetic, 16384U, 24301U,
            nbody::LargeNDistribution::uniform, {}, controller, gate);
    } catch (const std::invalid_argument&) {
        rejected_unsupported_count = true;
    }
    require(rejected_unsupported_count,
        "production transaction accepted an unsupported body count");
    require_failed_preparation_preserved_state("unsupported count");

    bool rejected_workload = false;
    try {
        (void)prepare_application_scenario_transaction(
            ScenarioKind::synthetic, 4096U, 24301U,
            nbody::LargeNDistribution::uniform, {}, controller, gate);
    } catch (const std::invalid_argument&) {
        rejected_workload = true;
    }
    require(rejected_workload,
        "production transaction accepted a prohibited Direct workload");
    require_failed_preparation_preserved_state("workload admission failure");

    bool injected_allocation_failure = false;
    try {
        (void)prepare_application_scenario_transaction(
            ScenarioKind::synthetic, 256U, 24301U,
            nbody::LargeNDistribution::uniform, {}, controller, gate,
            [](const PreparedScenarioReplacement&) {
                throw std::bad_alloc();
            });
    } catch (const std::bad_alloc&) {
        injected_allocation_failure = true;
    }
    require(injected_allocation_failure,
        "production preparation failure injection did not propagate");
    require_failed_preparation_preserved_state("allocation failure");

    auto prepared = prepare_application_scenario_transaction(
        ScenarioKind::synthetic, 1024U, 24301U,
        nbody::LargeNDistribution::planar, {}, controller, gate);
    const auto expected_initial_state = prepared.visual.scenario.initial_state;
    const std::string expected_name = prepared.visual.scenario.name;
    const double expected_dt = prepared.visual.scenario.dt_s;
    const std::size_t expected_steps = prepared.visual.scenario.speed_presets[
        prepared.visual.speed_index].steps_per_frame;
    commit_application_scenario_transaction(
        std::move(prepared), {
            controller,
            active,
            render_space,
            profile,
            individual,
            gate,
            worker,
            publication,
            presentation,
            timing,
        });
    require(active.kind == ScenarioKind::synthetic && active.name == expected_name &&
                active.initial_state.bodies.size() == 1024U &&
                same_state(controller.state(), expected_initial_state) &&
                controller.state().time_s == 0.0 && controller.dt_s() == expected_dt &&
                controller.steps_per_frame() == expected_steps &&
                controller.paused() &&
                controller.backend_kind() == BackendKind::sequential &&
                controller.solver_kind() == SolverKind::direct &&
                profile.size() == 1024U && individual.size() == 1024U &&
                gate.scenario_generation() == committed_gate_generation + 1U &&
                !gate.accepted_worker_generation() &&
                publication.generation == 0U && !publication.run &&
                presentation.empty() && !timing.sample() &&
                !gate.accepts(committed_publication),
            "successful production transaction did not atomically reset scene, "
            "controller, timing, generation, publication, and overlays");

    SimulationTimingTracker scoped_timing;
    const std::uint64_t direct_generation = scoped_timing.generation();
    require(scoped_timing.publish(direct_generation, 2.0, 500.0),
        "initial configuration-scoped timing was not published");
    require(!backend_selection_changes_execution(
                controller, BackendKind::sequential, 99, 17) &&
            !solver_selection_changes_execution(
                controller, SolverKind::direct,
                controller.barnes_hut_configuration()) &&
            !steps_per_frame_changes_execution(
                controller, controller.steps_per_frame()),
        "identical effective backend/solver reselection was treated as a change");
    select_application_backend(
        controller, scoped_timing, BackendKind::sequential, 99, 17);
    select_application_solver(
        controller, scoped_timing, SolverKind::direct,
        controller.barnes_hut_configuration());
    set_application_steps_per_frame(
        controller, scoped_timing, controller.steps_per_frame());
    require(scoped_timing.sample() &&
                scoped_timing.sample()->batch_milliseconds == 2.0,
        "identical effective reselection invalidated timing");

    auto barnes_hut = controller.barnes_hut_configuration();
    require(solver_selection_changes_execution(
                controller, SolverKind::barnes_hut, barnes_hut),
        "successful solver change was not identified");
    select_application_solver(
        controller, scoped_timing, SolverKind::barnes_hut, barnes_hut);
    require(!scoped_timing.sample(),
        "successful solver change retained stale timing");
    const std::uint64_t barnes_hut_generation = scoped_timing.generation();
    require(scoped_timing.publish(barnes_hut_generation, 3.0, 333.0),
        "Barnes-Hut timing was not published");

    auto changed_theta = barnes_hut;
    changed_theta.theta = 0.5;
    require(solver_selection_changes_execution(
                controller, SolverKind::barnes_hut, changed_theta),
        "theta change was not identified as an execution change");
    select_application_solver(
        controller, scoped_timing, SolverKind::barnes_hut, changed_theta);
    require(!scoped_timing.sample() &&
                !scoped_timing.publish(barnes_hut_generation, 99.0, 99.0),
        "stale asynchronous timing republished after theta invalidation");

    const std::uint64_t rejected_generation = scoped_timing.generation();
    require(scoped_timing.publish(rejected_generation, 4.0, 250.0),
        "pre-rejection timing was not published");
    auto invalid_barnes_hut = changed_theta;
    invalid_barnes_hut.theta = 0.0;
    bool rejected_solver_change = false;
    try {
        select_application_solver(
            controller, scoped_timing, SolverKind::barnes_hut,
            invalid_barnes_hut);
    } catch (const std::invalid_argument&) {
        rejected_solver_change = true;
    }
    require(rejected_solver_change && scoped_timing.sample() &&
                scoped_timing.generation() == rejected_generation &&
                scoped_timing.sample()->batch_milliseconds == 4.0,
        "rejected solver change invalidated valid timing");

    bool rejected_backend_change = false;
    try {
        select_application_backend(
            controller, scoped_timing, BackendKind::openmp, 0, 0);
    } catch (const std::invalid_argument&) {
        rejected_backend_change = true;
    }
    require(rejected_backend_change && scoped_timing.sample() &&
                scoped_timing.generation() == rejected_generation,
        "rejected backend change invalidated valid timing");

    if (SimulationController::openmp_available()) {
        require(backend_selection_changes_execution(
                    controller, BackendKind::openmp, 2, 0),
            "successful OpenMP backend change was not identified");
        select_application_backend(
            controller, scoped_timing, BackendKind::openmp, 2, 0);
        require(!scoped_timing.sample(),
            "successful backend change retained stale timing");
        const std::uint64_t openmp_generation = scoped_timing.generation();
        require(scoped_timing.publish(openmp_generation, 5.0, 200.0),
            "OpenMP timing was not published");
        require(!backend_selection_changes_execution(
                    controller, BackendKind::openmp, 2, 99),
            "identical effective OpenMP reselection was treated as a change");
        select_application_backend(
            controller, scoped_timing, BackendKind::openmp, 2, 99);
        require(scoped_timing.sample() &&
                    scoped_timing.sample()->batch_milliseconds == 5.0,
            "identical OpenMP reselection invalidated timing");
        require(backend_selection_changes_execution(
                    controller, BackendKind::openmp, 3, 0),
            "OpenMP thread change was not identified");
        select_application_backend(
            controller, scoped_timing, BackendKind::openmp, 3, 0);
        require(!scoped_timing.sample(),
            "successful OpenMP thread change retained stale timing");
    }

    const std::size_t old_steps = controller.steps_per_frame();
    require(steps_per_frame_changes_execution(controller, old_steps + 1U),
        "steps-per-frame change was not identified");
    set_application_steps_per_frame(
        controller, scoped_timing, old_steps + 1U);
    require(!scoped_timing.sample(),
        "successful steps-per-frame change retained stale timing");
}

}  // namespace

int main() {
    try {
        test_controller();
        test_controller_solver_selection_and_transactionality();
        test_render_space_is_read_only();
        test_solar_screen_space_sizing();
        test_scenario_switch_and_diagnostics();
        test_astronomical_catalog_composition();
        test_local_render_precision_and_visibility();
        test_composed_visibility_policy();
        test_parent_relative_trails();
        test_catalog_backend_equivalence_and_timestep_convergence();
        test_trail_boundedness();
        test_actual_sun_trail_and_barycentric_transform();
        test_planet_trail_geometry_fidelity();
        test_inner_system_guide_projection();
        test_camera_math();
        test_presentation_helpers();
        test_compute_experiment_scaling_and_comparison();
        test_application_transaction_and_timing_provenance();
        std::cout << "all visualization model tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visualization model test failure: " << error.what() << '\n';
        return 1;
    }
}
