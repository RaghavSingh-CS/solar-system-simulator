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
#include "renderer.hpp"

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/vec4.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using nbody::visualization::BackendKind;
using nbody::visualization::ScenarioKind;
using nbody::visualization::SolverKind;
using nbody::LargeNDistribution;

struct Options {
    ScenarioKind scenario{ScenarioKind::solar_system};
    std::string backend{"sequential"};
    std::string solver{"direct"};
    double barnes_hut_theta{0.25};
    std::size_t bodies{256};
    std::uint64_t seed{24301};
    LargeNDistribution distribution{LargeNDistribution::uniform};
    int openmp_threads{4};
    int cuda_device{0};
    int width{1440};
    int height{900};
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "  --scenario solar|catalog|synthetic Default: solar\n"
              << "  --backend sequential|openmp|cuda  CUDA is accepted only in CUDA-enabled viewers\n"
              << "  --solver direct|barnes-hut       Default: direct\n"
              << "  --theta X                         Barnes-Hut opening angle (0, 2], default 0.25\n"
              << "  --threads N                        OpenMP threads (default 4)\n"
              << "  --device N                         CUDA device index (default 0)\n"
              << "  --bodies N                         Synthetic bodies: 256|1024|4096|8192\n"
              << "  --seed N                           Synthetic seed (default 24301)\n"
              << "  --distribution uniform|clustered|planar  Synthetic distribution\n"
              << "  --width PIXELS --height PIXELS     Initial window size\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--scenario") {
            if (value == "solar") {
                options.scenario = ScenarioKind::solar_system;
            } else if (value == "catalog" || value == "horizons") {
                options.scenario = ScenarioKind::astronomical_catalog;
            } else if (value == "synthetic") {
                options.scenario = ScenarioKind::synthetic;
            } else {
                throw std::invalid_argument("scenario must be solar, catalog, or synthetic");
            }
        } else if (argument == "--backend") {
            options.backend = value;
        } else if (argument == "--solver") {
            options.solver = value;
        } else if (argument == "--theta") {
            options.barnes_hut_theta = std::stod(value);
        } else if (argument == "--threads") {
            options.openmp_threads = std::stoi(value);
        } else if (argument == "--device") {
            options.cuda_device = std::stoi(value);
        } else if (argument == "--bodies") {
            options.bodies = std::stoull(value);
        } else if (argument == "--seed") {
            options.seed = std::stoull(value);
        } else if (argument == "--distribution") {
            if (value == "uniform") {
                options.distribution = LargeNDistribution::uniform;
            } else if (value == "clustered") {
                options.distribution = LargeNDistribution::clustered;
            } else if (value == "planar") {
                options.distribution = LargeNDistribution::planar;
            } else {
                throw std::invalid_argument(
                    "distribution must be uniform, clustered, or planar");
            }
        } else if (argument == "--width") {
            options.width = std::stoi(value);
        } else if (argument == "--height") {
            options.height = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.backend != "sequential" && options.backend != "openmp" &&
        options.backend != "cuda") {
        throw std::invalid_argument("backend must be sequential, openmp, or cuda");
    }
    if (options.solver != "direct" && options.solver != "barnes-hut") {
        throw std::invalid_argument("solver must be direct or barnes-hut");
    }
    if (!(options.barnes_hut_theta > 0.0) ||
        options.barnes_hut_theta > 2.0 ||
        !std::isfinite(options.barnes_hut_theta)) {
        throw std::invalid_argument("Barnes-Hut theta must be finite and in (0, 2]");
    }
    if (options.bodies < 2 || options.openmp_threads <= 0 ||
        options.cuda_device < 0 || options.width <= 0 || options.height <= 0) {
        throw std::invalid_argument(
            "bodies >= 2, positive threads, non-negative device, and positive window dimensions are required");
    }
    if (options.scenario == ScenarioKind::synthetic &&
        !nbody::visualization::is_supported_compute_body_count(options.bodies)) {
        throw std::invalid_argument(
            "synthetic body count must be one of 256, 1024, 4096, or 8192");
    }
    return options;
}

void glfw_error_callback(int code, const char* description) {
    std::cerr << "GLFW error " << code << ": "
              << (description == nullptr ? "no description" : description) << '\n';
}

class GlfwRuntime {
public:
    GlfwRuntime() {
        glfwSetErrorCallback(glfw_error_callback);
        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error(
                "GLFW initialization failed; verify desktop access and graphics drivers");
        }
    }
    ~GlfwRuntime() { glfwTerminate(); }
    GlfwRuntime(const GlfwRuntime&) = delete;
    GlfwRuntime& operator=(const GlfwRuntime&) = delete;
};

struct WindowDeleter {
    void operator()(GLFWwindow* window) const noexcept {
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
    }
};

class ImGuiRuntime {
public:
    explicit ImGuiRuntime(GLFWwindow* window) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        float x_scale = 1.0F;
        float y_scale = 1.0F;
        glfwGetWindowContentScale(window, &x_scale, &y_scale);
        const float ui_scale = std::clamp((x_scale + y_scale) * 0.5F, 1.0F, 1.35F);
        io.FontGlobalScale = ui_scale;
        std::vector<std::filesystem::path> font_candidates;
        if (const char* windows_directory = std::getenv("WINDIR")) {
            font_candidates.emplace_back(
                std::filesystem::path(windows_directory) / "Fonts" / "segoeui.ttf");
            font_candidates.emplace_back(
                std::filesystem::path(windows_directory) / "Fonts" / "arial.ttf");
        }
        font_candidates.emplace_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
        font_candidates.emplace_back("/System/Library/Fonts/SFNS.ttf");
        bool font_loaded = false;
        for (const auto& path : font_candidates) {
            std::error_code error;
            if (std::filesystem::exists(path, error) && !error &&
                io.Fonts->AddFontFromFileTTF(path.string().c_str(), 17.0F) != nullptr) {
                font_loaded = true;
                break;
            }
        }
        if (!font_loaded) {
            io.Fonts->AddFontDefault();
        }
        ImGui::StyleColorsDark();
        auto& style = ImGui::GetStyle();
        style.ScaleAllSizes(ui_scale);
        style.WindowRounding = 10.0F;
        style.ChildRounding = 7.0F;
        style.FrameRounding = 6.0F;
        style.PopupRounding = 7.0F;
        style.GrabRounding = 6.0F;
        style.WindowPadding = {16.0F, 14.0F};
        style.FramePadding = {10.0F, 7.0F};
        style.ItemSpacing = {9.0F, 8.0F};
        style.ItemInnerSpacing = {7.0F, 5.0F};
        style.IndentSpacing = 18.0F;
        style.ScrollbarSize = 13.0F;
        style.WindowBorderSize = 1.0F;
        style.FrameBorderSize = 0.0F;
        auto* colors = style.Colors;
        colors[ImGuiCol_WindowBg] = {0.025F, 0.035F, 0.060F, 0.96F};
        colors[ImGuiCol_Border] = {0.18F, 0.27F, 0.38F, 0.78F};
        colors[ImGuiCol_FrameBg] = {0.075F, 0.105F, 0.145F, 0.96F};
        colors[ImGuiCol_FrameBgHovered] = {0.11F, 0.17F, 0.23F, 1.0F};
        colors[ImGuiCol_FrameBgActive] = {0.13F, 0.22F, 0.29F, 1.0F};
        colors[ImGuiCol_Button] = {0.10F, 0.20F, 0.27F, 1.0F};
        colors[ImGuiCol_ButtonHovered] = {0.16F, 0.34F, 0.43F, 1.0F};
        colors[ImGuiCol_ButtonActive] = {0.18F, 0.43F, 0.50F, 1.0F};
        colors[ImGuiCol_Header] = {0.10F, 0.24F, 0.31F, 1.0F};
        colors[ImGuiCol_HeaderHovered] = {0.15F, 0.34F, 0.42F, 1.0F};
        colors[ImGuiCol_HeaderActive] = {0.19F, 0.42F, 0.49F, 1.0F};
        colors[ImGuiCol_CheckMark] = {0.35F, 0.82F, 0.78F, 1.0F};
        colors[ImGuiCol_SliderGrab] = {0.31F, 0.68F, 0.70F, 1.0F};
        colors[ImGuiCol_SliderGrabActive] = {0.46F, 0.88F, 0.82F, 1.0F};
        colors[ImGuiCol_Text] = {0.88F, 0.92F, 0.96F, 1.0F};
        colors[ImGuiCol_TextDisabled] = {0.43F, 0.51F, 0.60F, 1.0F};
        if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
            ImGui::DestroyContext();
            throw std::runtime_error("Dear ImGui GLFW backend initialization failed");
        }
        if (!ImGui_ImplOpenGL3_Init("#version 330 core")) {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            throw std::runtime_error("Dear ImGui OpenGL3 backend initialization failed");
        }
        initialized_ = true;
    }
    ~ImGuiRuntime() {
        if (initialized_) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
    }
    ImGuiRuntime(const ImGuiRuntime&) = delete;
    ImGuiRuntime& operator=(const ImGuiRuntime&) = delete;

private:
    bool initialized_{false};
};

struct ApplicationModel {
    struct ExperimentUiState {
        bool capture_current_state{true};
        int backend_index{0};
        int solver_index{0};
        double barnes_hut_theta{0.25};
        double observation_horizon_s{86400.0};
        double sample_cadence_s{3600.0};
        double absolute_threshold_m{1000.0};
        double normalized_threshold{1.0e-6};
        int persistence_samples{2};
        std::string removed_body_stable_id;
        std::vector<bool> monitored;
        bool show_overlay{true};
        bool show_significant{true};
        int comparison_view{0}; // 0 live, 1 paired baseline, 2 body-removed endpoint
        std::string error;
        nbody::visualization::CounterfactualPresentationCache presentation;
    };

    struct PreparedApplicationScenario {
        nbody::visualization::PreparedApplicationScenarioTransaction core;
        std::unique_ptr<nbody::visualization::ConservationTracker> conservation;
        std::unique_ptr<nbody::visualization::TrailHistory> trails;
        std::vector<bool> catalog_inclusion;
        ExperimentUiState experiment_ui;
        bool show_reference_orbits{false};
        bool show_trails{false};
    };

    nbody::visualization::ScenarioDefinition scenario;
    nbody::visualization::SimulationController controller;
    nbody::visualization::RenderSpaceTransform render_space;
    std::unique_ptr<nbody::visualization::ConservationTracker> conservation;
    std::unique_ptr<nbody::visualization::TrailHistory> trails;
    nbody::visualization::CounterfactualPublicationGate experiment_gate;
    nbody::visualization::CounterfactualExperimentWorker experiment_worker;
    nbody::visualization::CounterfactualWorkerPublication experiment_publication;
    ExperimentUiState experiment_ui;
    std::size_t speed_index{0};
    std::size_t selected_body{0};
    std::optional<std::size_t> focused_body;
    std::optional<std::size_t> local_origin_body;
    std::optional<std::size_t> trail_reference_body;
    std::vector<bool> profile_visibility;
    std::vector<bool> individual_visibility;
    std::vector<std::string> catalog_roster_ids;
    std::vector<std::string> catalog_roster_names;
    std::vector<bool> catalog_inclusion_draft;
    nbody::visualization::BodyClassVisibility class_visibility{{
        true, true, true, true, true, true, true}};
    std::vector<nbody::visualization::LabelPlacement> label_layout;
    std::optional<nbody::visualization::LiveComputeComparison> live_comparison;
    std::string live_comparison_error;
    std::string view_error;
    std::size_t synthetic_body_count{256};
    std::uint64_t synthetic_seed{24301};
    LargeNDistribution synthetic_distribution{LargeNDistribution::uniform};
    std::size_t synthetic_body_count_draft{256};
    std::uint64_t synthetic_seed_draft{24301};
    LargeNDistribution synthetic_distribution_draft{LargeNDistribution::uniform};
    int preferred_cuda_device{0};
    bool cuda_device_available{false};
    double barnes_hut_theta_draft{0.25};
    std::string solver_error;
    std::string simulation_error;
    nbody::visualization::SimulationTimingTracker simulation_timing;
    bool show_trails{true};
    bool show_labels{true};
    bool show_reference_orbits{true};
    bool show_barycentric_motion{true};
    float sun_motion_amplification{10.0F};
    float panel_width{438.0F};
    int panel_page{0}; // Demo / Comparison / Advanced
    int previous_panel_page{-1};

    static nbody::visualization::ControllerConfig controller_config(
        const Options& options,
        const nbody::visualization::ScenarioDefinition& scenario,
        BackendKind backend) {
        nbody::BarnesHutConfiguration barnes_hut;
        barnes_hut.theta = options.barnes_hut_theta;
        return {
            scenario.dt_s,
            scenario.speed_presets[scenario.default_speed_index].steps_per_frame,
            backend,
            options.openmp_threads,
            options.cuda_device,
            options.solver == "barnes-hut"
                ? SolverKind::barnes_hut : SolverKind::direct,
            barnes_hut,
        };
    }

    ApplicationModel(const Options& options, BackendKind backend)
        : scenario(nbody::visualization::make_visual_scenario(
              options.scenario, options.bodies, options.seed, options.distribution)),
          controller(scenario.initial_state, controller_config(options, scenario, backend)),
          render_space(
              scenario.initial_state,
              scenario.styles,
              9.0F,
              scenario.default_radius_multiplier),
          speed_index(scenario.default_speed_index),
          selected_body(0U),
          synthetic_body_count(options.bodies),
          synthetic_seed(options.seed),
          synthetic_distribution(options.distribution),
          synthetic_body_count_draft(options.bodies),
          synthetic_seed_draft(options.seed),
          synthetic_distribution_draft(options.distribution),
          preferred_cuda_device(options.cuda_device),
          cuda_device_available(
              nbody::visualization::SimulationController::cuda_available(options.cuda_device)) {
        barnes_hut_theta_draft = options.barnes_hut_theta;
        if (const auto earth = nbody::visualization::find_body_index(
                scenario, scenario.kind == ScenarioKind::astronomical_catalog
                    ? "sol:earth" : "earth")) {
            selected_body = *earth;
        }
        profile_visibility.assign(scenario.initial_state.bodies.size(), true);
        individual_visibility.assign(scenario.initial_state.bodies.size(), true);
        const auto full_catalog = nbody::visualization::make_astronomical_catalog_scenario();
        catalog_roster_ids = full_catalog.stable_ids;
        catalog_inclusion_draft.assign(catalog_roster_ids.size(), true);
        for (const auto& body : full_catalog.initial_state.bodies) {
            catalog_roster_names.push_back(body.name);
        }
        if (scenario.kind == ScenarioKind::synthetic) {
            require_admitted_workload(workload_policy());
        }
        rebuild_optional_analysis();
        apply_workload_safety();
        reset_experiment_ui();
    }

    nbody::visualization::InteractiveWorkloadPolicy workload_policy() const noexcept {
        if (scenario.kind != ScenarioKind::synthetic) {
            return {};
        }
        nbody::visualization::WorkloadBackend backend =
            nbody::visualization::WorkloadBackend::sequential;
        if (controller.backend_kind() == BackendKind::openmp) {
            backend = nbody::visualization::WorkloadBackend::openmp;
        } else if (controller.backend_kind() == BackendKind::cuda) {
            backend = nbody::visualization::WorkloadBackend::cuda;
        }
        const auto solver = controller.solver_kind() == SolverKind::barnes_hut
            ? nbody::visualization::WorkloadSolver::barnes_hut
            : nbody::visualization::WorkloadSolver::direct;
        return nbody::visualization::interactive_workload_policy(
            synthetic_body_count, backend, solver);
    }

    nbody::visualization::InteractiveWorkloadPolicy workload_policy_for(
        BackendKind backend_kind,
        SolverKind solver_kind,
        std::size_t body_count) const noexcept {
        nbody::visualization::WorkloadBackend backend =
            nbody::visualization::WorkloadBackend::sequential;
        if (backend_kind == BackendKind::openmp) {
            backend = nbody::visualization::WorkloadBackend::openmp;
        } else if (backend_kind == BackendKind::cuda) {
            backend = nbody::visualization::WorkloadBackend::cuda;
        }
        const auto solver = solver_kind == SolverKind::barnes_hut
            ? nbody::visualization::WorkloadSolver::barnes_hut
            : nbody::visualization::WorkloadSolver::direct;
        return nbody::visualization::interactive_workload_policy(
            body_count, backend, solver);
    }

    static void require_admitted_workload(
        const nbody::visualization::InteractiveWorkloadPolicy& policy) {
        if (!policy.continuous_advancement && !policy.manual_single_step) {
            throw std::invalid_argument(policy.explanation);
        }
    }

    void select_backend_admitted(
        BackendKind backend,
        int openmp_threads = 1,
        int cuda_device = 0) {
        if (scenario.kind == ScenarioKind::synthetic) {
            require_admitted_workload(workload_policy_for(
                backend, controller.solver_kind(), synthetic_body_count));
        }
        nbody::visualization::select_application_backend(
            controller, simulation_timing, backend, openmp_threads, cuda_device);
        apply_workload_safety();
    }

    void select_solver_admitted(
        SolverKind solver,
        const nbody::BarnesHutConfiguration& barnes_hut) {
        if (scenario.kind == ScenarioKind::synthetic) {
            require_admitted_workload(workload_policy_for(
                controller.backend_kind(), solver, synthetic_body_count));
        }
        nbody::visualization::select_application_solver(
            controller, simulation_timing, solver, barnes_hut);
        apply_workload_safety();
    }

    void set_steps_per_frame(std::size_t steps) {
        nbody::visualization::set_application_steps_per_frame(
            controller, simulation_timing, steps);
    }

    static ExperimentUiState make_experiment_ui(
        const nbody::visualization::ScenarioDefinition& next_scenario) {
        ExperimentUiState result;
        result.observation_horizon_s =
            next_scenario.kind == ScenarioKind::astronomical_catalog
                ? 86400.0 : 30.0 * 86400.0;
        result.sample_cadence_s =
            next_scenario.kind == ScenarioKind::astronomical_catalog
                ? 3600.0 : 86400.0;
        result.monitored.assign(next_scenario.stable_ids.size(), true);
        const auto candidates =
            nbody::visualization::counterfactual_removal_candidates(next_scenario);
        if (!candidates.empty()) {
            result.removed_body_stable_id = candidates.front();
        }
        return result;
    }

    PreparedApplicationScenario prepare_scenario(
        ScenarioKind kind,
        std::size_t bodies,
        std::uint64_t seed,
        LargeNDistribution distribution,
        const std::vector<std::string>& catalog_subset) const {
        std::unique_ptr<nbody::visualization::ConservationTracker> next_conservation;
        std::unique_ptr<nbody::visualization::TrailHistory> next_trails;
        std::vector<bool> next_catalog_inclusion = catalog_inclusion_draft;
        ExperimentUiState next_experiment_ui;
        bool next_show_trails = false;
        bool next_show_reference_orbits = false;
        auto core = nbody::visualization::prepare_application_scenario_transaction(
            kind, bodies, seed, distribution, catalog_subset, controller,
            experiment_gate,
            [&](const nbody::visualization::PreparedScenarioReplacement& visual) {
                const bool full_analysis =
                    visual.scenario.kind != ScenarioKind::synthetic ||
                    visual.scenario.initial_state.bodies.size() <= 1024U;
                if (full_analysis) {
                    next_conservation =
                        std::make_unique<nbody::visualization::ConservationTracker>(
                            visual.scenario.initial_state);
                    next_trails = std::make_unique<nbody::visualization::TrailHistory>(
                        visual.scenario.initial_state, 768,
                        visual.scenario.trail_sample_interval_s);
                }
                if (visual.scenario.kind == ScenarioKind::astronomical_catalog) {
                    for (std::size_t index = 0;
                         index < catalog_roster_ids.size(); ++index) {
                        next_catalog_inclusion[index] =
                            nbody::visualization::find_body_index(
                                visual.scenario, catalog_roster_ids[index]).has_value();
                    }
                }
                next_experiment_ui = make_experiment_ui(visual.scenario);
                next_show_trails =
                    visual.scenario.kind != ScenarioKind::synthetic ||
                    visual.scenario.initial_state.bodies.size() <= 1024U;
                next_show_reference_orbits =
                    visual.scenario.kind != ScenarioKind::synthetic;
            }
        );
        return {
            std::move(core),
            std::move(next_conservation),
            std::move(next_trails),
            std::move(next_catalog_inclusion),
            std::move(next_experiment_ui),
            next_show_reference_orbits,
            next_show_trails,
        };
    }

    void rebuild_optional_analysis() {
        const bool full_analysis = scenario.kind != ScenarioKind::synthetic ||
            synthetic_body_count <= 1024;
        if (full_analysis) {
            conservation = std::make_unique<nbody::visualization::ConservationTracker>(
                controller.state());
            trails = std::make_unique<nbody::visualization::TrailHistory>(
                controller.state(), 768, scenario.trail_sample_interval_s);
        } else {
            conservation.reset();
            trails.reset();
            show_trails = false;
        }
    }

    void apply_workload_safety() {
        if (scenario.kind == ScenarioKind::synthetic &&
            !workload_policy().continuous_advancement) {
            controller.set_paused(true);
        }
    }

    void switch_scenario(
        ScenarioKind kind,
        std::size_t bodies,
        std::uint64_t seed,
        LargeNDistribution distribution,
        const std::vector<std::string>& catalog_subset = {}) {
        auto prepared = prepare_scenario(kind, bodies, seed, distribution, catalog_subset);
        const std::size_t next_speed = prepared.core.visual.speed_index;
        const std::size_t next_selected_body = prepared.core.visual.selected_body;
        static_assert(std::is_nothrow_move_assignable_v<ExperimentUiState>);
        nbody::visualization::commit_application_scenario_transaction(
            std::move(prepared.core), {
                controller,
                scenario,
                render_space,
                profile_visibility,
                individual_visibility,
                experiment_gate,
                experiment_worker,
                experiment_publication,
                experiment_ui.presentation,
                simulation_timing,
            });
        conservation.swap(prepared.conservation);
        trails.swap(prepared.trails);
        catalog_inclusion_draft.swap(prepared.catalog_inclusion);
        experiment_ui = std::move(prepared.experiment_ui);
        experiment_ui.backend_index = static_cast<int>(controller.backend_kind());
        experiment_ui.solver_index = static_cast<int>(controller.solver_kind());
        speed_index = next_speed;
        selected_body = next_selected_body;
        class_visibility.fill(true);
        if (scenario.kind == ScenarioKind::synthetic) {
            synthetic_body_count = scenario.initial_state.bodies.size();
            synthetic_seed = *scenario.synthetic_seed;
            synthetic_distribution = *scenario.synthetic_distribution;
            synthetic_body_count_draft = synthetic_body_count;
            synthetic_seed_draft = synthetic_seed;
            synthetic_distribution_draft = synthetic_distribution;
        }
        show_labels = true;
        show_reference_orbits = prepared.show_reference_orbits;
        show_trails = prepared.show_trails;
        apply_workload_safety();
        label_layout.clear();
        live_comparison.reset();
        live_comparison_error.clear();
        simulation_error.clear();
        view_error.clear();
        focused_body.reset();
        local_origin_body.reset();
        trail_reference_body.reset();
    }

    bool astronomical() const noexcept {
        return scenario.kind != ScenarioKind::synthetic;
    }

    void reset_experiment_ui() {
        experiment_ui = make_experiment_ui(scenario);
        experiment_ui.backend_index = static_cast<int>(controller.backend_kind());
        experiment_ui.solver_index = static_cast<int>(controller.solver_kind());
    }

    void refresh_experiment_publication() {
        auto next = experiment_worker.publication();
        if (experiment_gate.accepts(next)) {
            experiment_publication = std::move(next);
        }
        if (experiment_publication.state == nbody::visualization::CounterfactualWorkerState::failed) {
            experiment_ui.comparison_view = 0;
        }
        const auto* result = experiment_publication.result.get();
        if (experiment_ui.presentation.generation ==
                experiment_publication.generation &&
            experiment_ui.presentation.result == result &&
            experiment_gate.accepts(experiment_publication)) {
            return;
        }
        experiment_ui.presentation.generation = experiment_publication.generation;
        experiment_ui.presentation.result = result;
        experiment_ui.presentation.result_rows.clear();
        experiment_ui.presentation.warning_groups.clear();
        experiment_ui.presentation.error.clear();
        if (result == nullptr ||
            !experiment_gate.accepts(experiment_publication)) {
            return;
        }
        try {
            std::vector<nbody::visualization::CounterfactualBodyLabel> labels;
            labels.reserve(scenario.stable_ids.size());
            for (std::size_t index = 0; index < scenario.stable_ids.size(); ++index) {
                labels.push_back({
                    scenario.stable_ids[index],
                    controller.state().bodies[index].name,
                });
            }
            experiment_ui.presentation.result_rows =
                nbody::visualization::build_counterfactual_result_rows(
                    *result, labels);
            experiment_ui.presentation.warning_groups =
                nbody::visualization::group_counterfactual_warnings(*result);
        } catch (const std::exception& error) {
            experiment_ui.presentation.error = error.what();
        }
    }

    void apply_visibility(std::vector<nbody::visualization::RenderBody>& snapshot) const {
        nbody::visualization::apply_effective_body_visibility(
            scenario, profile_visibility, individual_visibility, class_visibility, snapshot);
    }

    bool body_visible(std::size_t index) const {
        return nbody::visualization::effective_body_visibility(
            scenario, index, profile_visibility, individual_visibility, class_visibility);
    }

    void reset() {
        nbody::visualization::invalidate_counterfactual_application_publication(
            experiment_worker,
            experiment_gate,
            experiment_publication,
            experiment_ui.presentation);
        experiment_ui.error.clear();
        experiment_ui.comparison_view = 0;
        controller.reset();
        simulation_error.clear();
        simulation_timing.invalidate();
        if (conservation) {
            conservation->reset(controller.state());
        }
        if (trails) {
            trails->reset(controller.state());
        }
        apply_workload_safety();
    }
};

void refresh_render_origin(ApplicationModel& app) {
    if (app.local_origin_body && *app.local_origin_body < app.controller.state().bodies.size()) {
        if (app.experiment_ui.comparison_view != 0 &&
            app.experiment_gate.accepts(app.experiment_publication) &&
            app.experiment_publication.state == nbody::visualization::CounterfactualWorkerState::completed &&
            app.experiment_publication.result &&
            app.scenario.stable_ids[*app.local_origin_body] != app.experiment_publication.removed_body_stable_id) {
            app.render_space.set_origin_m(app.experiment_publication.result->baseline_state
                .bodies[*app.local_origin_body].position_m);
            return;
        }
        app.render_space.set_origin_m(
            app.controller.state().bodies[*app.local_origin_body].position_m);
    } else {
        app.render_space.set_origin_m({});
    }
}

void apply_view_preset(
    ApplicationModel& app,
    nbody::visualization::OrbitCamera& camera,
    nbody::visualization::CameraPreset preset,
    std::vector<nbody::visualization::RenderBody>& snapshot) {
    app.render_space.update_snapshot(app.controller.state(), snapshot);
    const auto requested = nbody::visualization::camera_framing(
        preset, app.scenario, snapshot);
    std::vector<bool> next_profile_visibility(
        snapshot.size(), requested.visible_bodies.empty());
    if (!requested.visible_bodies.empty()) {
        for (const auto index : requested.visible_bodies) {
            next_profile_visibility[index] = true;
        }
    }
    const nbody::Vec3 origin = requested.local_origin_body
        ? app.controller.state().bodies[*requested.local_origin_body].position_m
        : nbody::Vec3{};
    std::vector<std::size_t> framing_indices = requested.visible_bodies;
    if (framing_indices.empty()) {
        framing_indices.resize(snapshot.size());
        for (std::size_t index = 0; index < framing_indices.size(); ++index) {
            framing_indices[index] = index;
        }
    }
    auto next_transform = app.render_space;
    next_transform.configure_view(
        app.controller.state(), origin, framing_indices, 8.0F);
    std::vector<nbody::visualization::RenderBody> next_snapshot;
    next_transform.update_snapshot(app.controller.state(), next_snapshot);
    nbody::visualization::apply_effective_body_visibility(
        app.scenario, next_profile_visibility, app.individual_visibility,
        app.class_visibility, next_snapshot);
    const auto framing = nbody::visualization::camera_framing(
        preset, app.scenario, next_snapshot);
    app.render_space = std::move(next_transform);
    snapshot = std::move(next_snapshot);
    app.profile_visibility = std::move(next_profile_visibility);
    app.local_origin_body = requested.local_origin_body;
    app.trail_reference_body = requested.parent_relative_trails
        ? requested.local_origin_body : std::nullopt;
    nbody::visualization::apply_camera_framing(camera, framing);
    app.focused_body = framing.tracked_body;
    if (framing.tracked_body) {
        app.selected_body = *framing.tracked_body;
    }
}

void focus_selected_body(
    ApplicationModel& app,
    nbody::visualization::OrbitCamera& camera,
    std::vector<nbody::visualization::RenderBody>& snapshot) {
    if (app.selected_body >= app.controller.state().bodies.size()) {
        return;
    }
    std::vector<std::size_t> framing{app.selected_body};
    const auto& metadata = app.scenario.body_metadata[app.selected_body];
    if (metadata.parent_stable_id) {
        if (const auto parent = nbody::visualization::find_body_index(
                app.scenario, *metadata.parent_stable_id)) {
            framing.push_back(*parent);
        }
    }
    if (framing.size() == 1U && app.controller.state().bodies.size() > 1U) {
        framing.push_back(app.selected_body == 0U ? 1U : 0U);
    }
    const auto origin = app.controller.state().bodies[app.selected_body].position_m;
    app.render_space.configure_view(app.controller.state(), origin, framing, 5.5F);
    app.local_origin_body = app.selected_body;
    app.trail_reference_body = app.selected_body;
    app.focused_body = app.selected_body;
    app.profile_visibility.assign(app.controller.state().bodies.size(), false);
    for (const auto index : framing) {
        app.profile_visibility[index] = true;
    }
    app.render_space.update_snapshot(app.controller.state(), snapshot);
    app.apply_visibility(snapshot);
    camera.set_pose(camera.yaw_degrees(), camera.pitch_degrees(), 12.0F, glm::vec3(0.0F));
}

struct InteractionState {
    nbody::visualization::OrbitCamera* camera{nullptr};
    ApplicationModel* application{nullptr};
    int preferred_openmp_threads{1};
    bool dragging{false};
    bool panning{false};
    double last_x{0.0};
    double last_y{0.0};
    double drag_distance{0.0};
    bool pending_selection{false};
    double selection_x{0.0};
    double selection_y{0.0};
    float scene_left{0.0F};
};

InteractionState* interaction(GLFWwindow* window) noexcept {
    return static_cast<InteractionState*>(glfwGetWindowUserPointer(window));
}

bool wants_keyboard() {
    return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard;
}

bool wants_mouse() {
    return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse;
}

void log_backend(const nbody::visualization::SimulationController& controller) {
    std::cout << "active_backend=" << controller.backend_name()
              << " active_solver=" << controller.solver_name();
    if (controller.backend_kind() == BackendKind::openmp) {
        std::cout << " threads=" << controller.openmp_threads();
    } else if (controller.backend_kind() == BackendKind::cuda) {
        std::cout << " device=" << controller.cuda_device();
    }
    if (controller.solver_kind() == SolverKind::barnes_hut) {
        std::cout << " theta=" << controller.barnes_hut_configuration().theta;
    }
    std::cout << '\n';
}

void key_callback(GLFWwindow* window, int key, int, int action, int) {
    if ((action != GLFW_PRESS && action != GLFW_REPEAT) || wants_keyboard()) {
        return;
    }
    auto* input = interaction(window);
    if (input == nullptr) {
        return;
    }
    try {
        auto& app = *input->application;
        const bool initial_press = action == GLFW_PRESS;
        if (key == GLFW_KEY_ESCAPE && initial_press) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (key == GLFW_KEY_SPACE && initial_press) {
            if (app.scenario.kind == ScenarioKind::synthetic &&
                !app.workload_policy().continuous_advancement) {
                app.controller.set_paused(true);
                app.solver_error =
                    "Continuous advancement is not admitted for this synthetic workload; "
                    "use an enabled bounded manual step.";
            } else {
                app.experiment_ui.comparison_view = 0;
                app.controller.toggle_paused();
            }
        } else if (key == GLFW_KEY_R && initial_press) {
            app.reset();
        } else if (key == GLFW_KEY_C && initial_press) {
            app.focused_body.reset();
            app.local_origin_body.reset();
            app.trail_reference_body.reset();
            app.profile_visibility.assign(app.controller.state().bodies.size(), true);
            std::vector<std::size_t> indices(app.controller.state().bodies.size());
            for (std::size_t index = 0; index < indices.size(); ++index) {
                indices[index] = index;
            }
            app.render_space.configure_view(app.controller.state(), {}, indices, 8.0F);
            input->camera->reset();
        } else if (key == GLFW_KEY_1 && initial_press) {
            app.select_backend_admitted(BackendKind::sequential);
            log_backend(app.controller);
        } else if (key == GLFW_KEY_2 && initial_press) {
            app.select_backend_admitted(
                BackendKind::openmp, input->preferred_openmp_threads);
            log_backend(app.controller);
        } else if (key == GLFW_KEY_3 && initial_press) {
            app.select_backend_admitted(
                BackendKind::cuda, 1, app.preferred_cuda_device);
            log_backend(app.controller);
        } else if (key == GLFW_KEY_A) {
            input->camera->orbit(-8.0F, 0.0F);
        } else if (key == GLFW_KEY_D) {
            input->camera->orbit(8.0F, 0.0F);
        } else if (key == GLFW_KEY_UP) {
            input->camera->orbit(0.0F, -8.0F);
        } else if (key == GLFW_KEY_DOWN) {
            input->camera->orbit(0.0F, 8.0F);
        } else if (key == GLFW_KEY_W) {
            input->camera->dolly(1.0F);
        } else if (key == GLFW_KEY_S) {
            input->camera->dolly(-1.0F);
        }
    } catch (const std::exception& error) {
        if (input != nullptr && input->application != nullptr) {
            input->application->solver_error = error.what();
        }
        std::cerr << "input error: " << error.what() << '\n';
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT && button != GLFW_MOUSE_BUTTON_RIGHT) {
        return;
    }
    auto* input = interaction(window);
    if (input == nullptr) {
        return;
    }
    if (wants_mouse()) {
        // A viewport drag can end over ImGui; never leave orbit/pan latched.
        if (action == GLFW_RELEASE) input->dragging = false;
        return;
    }
    if (action == GLFW_PRESS) {
        input->dragging = true;
        input->panning = button == GLFW_MOUSE_BUTTON_RIGHT;
        input->drag_distance = 0.0;
        glfwGetCursorPos(window, &input->last_x, &input->last_y);
    } else if (action == GLFW_RELEASE && input->dragging) {
        input->dragging = false;
        if (!input->panning && input->drag_distance < 5.0) {
            input->pending_selection = true;
            glfwGetCursorPos(window, &input->selection_x, &input->selection_y);
        }
    }
}

void cursor_callback(GLFWwindow* window, double x, double y) {
    auto* input = interaction(window);
    if (input == nullptr || !input->dragging || wants_mouse()) {
        return;
    }
    const double delta_x = x - input->last_x;
    const double delta_y = y - input->last_y;
    input->drag_distance += std::hypot(delta_x, delta_y);
    if (input->panning) {
        input->camera->pan(static_cast<float>(delta_x), static_cast<float>(delta_y));
        input->application->focused_body.reset();
    } else {
        input->camera->orbit(static_cast<float>(delta_x), static_cast<float>(delta_y));
    }
    input->last_x = x;
    input->last_y = y;
}

void scroll_callback(GLFWwindow* window, double, double y_offset) {
    auto* input = interaction(window);
    if (input != nullptr && !wants_mouse()) {
        input->camera->dolly(static_cast<float>(y_offset));
    }
}

std::optional<glm::vec2> project_to_screen(
    const glm::vec3& position,
    const nbody::visualization::OrbitCamera& camera,
    const ImVec2& display_size,
    float scene_left = 0.0F) {
    const glm::vec4 clip = camera.projection_matrix() * camera.view_matrix() *
        glm::vec4(position, 1.0F);
    if (clip.w <= 0.0F) {
        return std::nullopt;
    }
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.z < -1.0F || ndc.z > 1.0F) {
        return std::nullopt;
    }
    const float scene_width = std::max(1.0F, display_size.x - scene_left);
    return glm::vec2(
        scene_left + (ndc.x * 0.5F + 0.5F) * scene_width,
        (-ndc.y * 0.5F + 0.5F) * display_size.y);
}

void select_nearest_body(
    ApplicationModel& app,
    const std::vector<nbody::visualization::RenderBody>& bodies,
    const nbody::visualization::OrbitCamera& camera,
    double mouse_x,
    double mouse_y,
    float scene_left) {
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    float best_distance = 22.0F;
    std::optional<std::size_t> best;
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        if (!bodies[index].visible) {
            continue;
        }
        const auto screen = project_to_screen(bodies[index].position, camera, display, scene_left);
        if (!screen) {
            continue;
        }
        const float distance = std::hypot(
            screen->x - static_cast<float>(mouse_x),
            screen->y - static_cast<float>(mouse_y));
        if (distance < best_distance) {
            best_distance = distance;
            best = index;
        }
    }
    if (best) {
        app.selected_body = *best;
    }
}

void update_render_trails(
    ApplicationModel& app,
    std::vector<nbody::visualization::RenderTrail>& output) {
    if (!app.show_trails || !app.trails) {
        output.clear();
    } else {
        const auto& physical = app.trails->positions();
        output.resize(physical.size());
        for (std::size_t index = 0; index < physical.size(); ++index) {
            auto& trail = output[index];
            trail.color = app.scenario.styles[index].color;
            trail.alpha = 0.86F;
            trail.fade_with_age = true;
            trail.line_style = nbody::visualization::RenderLineStyle::solid;
            trail.points.clear();
            if (!app.body_visible(index)) {
                continue;
            }
            const auto sun = nbody::visualization::find_body_index(app.scenario, "sun");
            if (sun && index == *sun && app.scenario.kind == ScenarioKind::solar_system &&
                app.show_barycentric_motion) {
                continue;
            }
            if (app.trail_reference_body && *app.trail_reference_body < physical.size()) {
                if (index == *app.trail_reference_body) {
                    continue;
                }
                const auto relative = nbody::visualization::parent_relative_trail(
                    physical[index], physical[*app.trail_reference_body]);
                trail.points.reserve(relative.size());
                for (const auto& offset : relative) {
                    trail.points.push_back(app.render_space.to_render_offset(offset));
                }
            } else {
                trail.points.reserve(physical[index].size());
                for (const auto& point : physical[index]) {
                    trail.points.push_back(app.render_space.to_render_position(point));
                }
            }
        }
    }
    if (app.show_barycentric_motion &&
        app.scenario.kind == ScenarioKind::solar_system && app.trails &&
        !app.trail_reference_body) {
        const auto& sun_history = app.trails->positions().front();
        nbody::visualization::RenderTrail sun_motion;
        sun_motion.color = {1.0F, 0.66F, 0.22F};
        sun_motion.alpha = 0.92F;
        sun_motion.fade_with_age = true;
        sun_motion.points.reserve(sun_history.size());
        const glm::vec3 barycenter = app.render_space.to_render_position({});
        for (const auto& point : sun_history) {
            sun_motion.points.push_back(nbody::visualization::amplify_from_barycenter(
                app.render_space.to_render_position(point),
                barycenter,
                app.sun_motion_amplification));
        }
        output.push_back(std::move(sun_motion));
    }
    if (app.show_reference_orbits && !app.scenario.reference_paths_m.empty()) {
        const std::size_t first_reference = output.size();
        output.resize(first_reference + app.scenario.reference_paths_m.size());
        for (std::size_t index = 0; index < app.scenario.reference_paths_m.size(); ++index) {
            auto& guide = output[first_reference + index];
            guide.color = app.scenario.styles[index].color *
                (app.scenario.kind == ScenarioKind::astronomical_catalog ? 0.82F : 0.48F);
            guide.alpha = app.scenario.kind == ScenarioKind::astronomical_catalog ? 0.55F : 0.24F;
            guide.fade_with_age = false;
            guide.line_style = nbody::visualization::RenderLineStyle::solid;
            guide.points.clear();
            if (!app.body_visible(index)) {
                continue;
            }
            if (app.scenario.kind == ScenarioKind::astronomical_catalog) {
                const auto& parent_id = app.scenario.body_metadata[index].parent_stable_id;
                // In the full/inner view show heliocentric guides; in a local
                // view show only guides about that view's actual parent.
                if (!parent_id || (app.trail_reference_body
                        ? *parent_id != app.scenario.stable_ids[*app.trail_reference_body]
                        : *parent_id != "sol:sun")) continue;
            }
            guide.points.reserve(app.scenario.reference_paths_m[index].size());
            for (const auto& point : app.scenario.reference_paths_m[index]) {
                if (app.trail_reference_body) {
                    const auto& reference_origin = app.scenario.initial_state
                        .bodies[*app.trail_reference_body].position_m;
                    guide.points.push_back(
                        app.render_space.to_render_offset(point - reference_origin));
                } else {
                    guide.points.push_back(app.render_space.to_render_position(point));
                }
            }
        }
    }
}

bool publication_matches_scenario(const ApplicationModel& app) {
    const auto& publication = app.experiment_publication;
    if (!app.experiment_gate.accepts(publication) || !publication.identities ||
        publication.identities->size() != app.scenario.stable_ids.size()) {
        return false;
    }
    for (std::size_t index = 0; index < app.scenario.stable_ids.size(); ++index) {
        if ((*publication.identities)[index].stable_id != app.scenario.stable_ids[index]) {
            return false;
        }
    }
    return true;
}

void append_counterfactual_trails(
    ApplicationModel& app,
    std::vector<nbody::visualization::RenderTrail>& output) {
    app.experiment_ui.presentation.frame_error.clear();
    if (!app.experiment_ui.show_overlay || !app.experiment_publication.result ||
        app.experiment_publication.state == nbody::visualization::CounterfactualWorkerState::failed ||
        !publication_matches_scenario(app)) {
        return;
    }
    const auto& result = *app.experiment_publication.result;
    std::vector<std::string> displayed;
    if (app.selected_body < app.scenario.stable_ids.size()) {
        displayed.push_back(app.scenario.stable_ids[app.selected_body]);
    }
    if (app.experiment_ui.show_significant) {
        for (const auto& summary : result.summaries) {
            if (summary.significant &&
                std::find(displayed.begin(), displayed.end(), summary.stable_id) ==
                    displayed.end()) {
                displayed.push_back(summary.stable_id);
            }
        }
    }
    // A visible comparison must not depend on significance at global scale.
    for (std::size_t index = 0; index < app.scenario.stable_ids.size(); ++index) {
        if (app.body_visible(index) &&
            std::find(displayed.begin(), displayed.end(), app.scenario.stable_ids[index]) == displayed.end()) {
            displayed.push_back(app.scenario.stable_ids[index]);
        }
    }
    std::optional<std::string> common_origin;
    if (app.trail_reference_body &&
        *app.trail_reference_body < app.scenario.stable_ids.size()) {
        common_origin = app.scenario.stable_ids[*app.trail_reference_body];
    }
    displayed = nbody::visualization::counterfactual_visible_overlay_ids(
        app.scenario, app.profile_visibility, app.individual_visibility,
        app.class_visibility, displayed, 12U);
    const auto reconstructed =
        nbody::visualization::build_counterfactual_render_paths(
            app.experiment_publication, app.render_space, displayed,
            common_origin, 12U);
    app.experiment_ui.presentation.frame_error =
        reconstructed.frame_unavailable_reason;
    for (const auto& path : reconstructed.paths) {
        const auto index = nbody::visualization::find_body_index(
            app.scenario, path.stable_id);
        if (!index) {
            continue;
        }
        nbody::visualization::RenderTrail trail;
        trail.points = path.points;
        const glm::vec3 body_color = app.scenario.styles[*index].color;
        if (path.branch ==
            nbody::visualization::CounterfactualPathBranch::baseline) {
            trail.color = body_color;
            trail.alpha = 0.95F;
            trail.line_style = nbody::visualization::RenderLineStyle::solid;
        } else {
            trail.color = glm::vec3(1.0F, 0.28F, 0.72F) * 0.72F +
                body_color * 0.28F;
            trail.alpha = 0.92F;
            trail.line_style = nbody::visualization::RenderLineStyle::dashed;
        }
        trail.fade_with_age = false;
        output.push_back(std::move(trail));
    }
}

std::string human_time(double seconds) {
    std::ostringstream text;
    const double days = seconds / 86400.0;
    if (days < 2.0) {
        text << std::fixed << std::setprecision(1) << seconds / 3600.0 << " hours";
    } else if (days < 730.5) {
        text << std::fixed << std::setprecision(2) << days << " days";
    } else {
        text << std::fixed << std::setprecision(3) << days / 365.25 << " years";
    }
    return text.str();
}

std::string human_bytes(std::size_t bytes) {
    std::ostringstream text;
    const double mebibytes = static_cast<double>(bytes) / (1024.0 * 1024.0);
    text << std::fixed << std::setprecision(mebibytes < 10.0 ? 2 : 1)
         << mebibytes << " MiB";
    return text.str();
}

std::string join_stable_ids(const std::vector<std::string>& ids) {
    if (ids.empty()) {
        return "None";
    }
    std::ostringstream text;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index != 0U) {
            text << ", ";
        }
        text << ids[index];
    }
    return text.str();
}

nbody::experiments::CounterfactualExperimentConfiguration
make_experiment_configuration(
    const ApplicationModel& app,
    int preferred_openmp_threads) {
    nbody::experiments::CounterfactualExperimentConfiguration configuration;
    configuration.removed_body_stable_id =
        app.experiment_ui.removed_body_stable_id;
    switch (app.experiment_ui.backend_index) {
        case 0:
            configuration.backend =
                nbody::experiments::ExperimentBackend::sequential;
            break;
        case 1:
            configuration.backend = nbody::experiments::ExperimentBackend::openmp;
            break;
        case 2:
            configuration.backend = nbody::experiments::ExperimentBackend::cuda;
            break;
        default:
            throw std::invalid_argument("unknown counterfactual experiment backend");
    }
    configuration.openmp_threads = preferred_openmp_threads;
    configuration.cuda_device = app.preferred_cuda_device;
    switch (app.experiment_ui.solver_index) {
        case 0:
            configuration.solver = nbody::experiments::ExperimentSolver::direct;
            break;
        case 1:
            configuration.solver =
                nbody::experiments::ExperimentSolver::barnes_hut;
            configuration.barnes_hut.theta =
                app.experiment_ui.barnes_hut_theta;
            break;
        default:
            throw std::invalid_argument("unknown counterfactual experiment solver");
    }
    configuration.limits.observation_horizon_s =
        app.experiment_ui.observation_horizon_s;
    configuration.limits.sample_cadence_s = app.experiment_ui.sample_cadence_s;
    configuration.limits.integration_steps_per_batch = 16U;
    configuration.limits.maximum_monitored_bodies = 64U;
    configuration.limits.maximum_reference_bodies = 4U;
    configuration.limits.maximum_stored_samples = 4096U;
    configuration.limits.memory_budget_bytes = 64U * 1024U * 1024U;
    configuration.limits.workload_budget_interactions = 500000000ULL;
    configuration.limits.maximum_issue_records = 256U;
    configuration.limits.maximum_issue_message_bytes = 64U * 1024U;
    configuration.significance.absolute_separation_threshold_m =
        app.experiment_ui.absolute_threshold_m;
    configuration.significance.normalized_deviation_threshold =
        app.experiment_ui.normalized_threshold;
    configuration.significance.persistence_samples = static_cast<std::size_t>(
        std::max(1, app.experiment_ui.persistence_samples));
    configuration.significance.normalization_denominator_floor_m = 1.0;
    configuration.guards.dangerous_close_approach_radius_multiple = 3.0;
    configuration.guards.maximum_relative_energy_drift = 1.0e-6;
    for (std::size_t index = 0;
         index < app.scenario.stable_ids.size() &&
         index < app.experiment_ui.monitored.size(); ++index) {
        if (app.experiment_ui.monitored[index] &&
            app.scenario.stable_ids[index] !=
                configuration.removed_body_stable_id) {
            configuration.monitored_body_stable_ids.push_back(
                app.scenario.stable_ids[index]);
        }
    }
    if (app.trail_reference_body &&
        *app.trail_reference_body < app.scenario.stable_ids.size()) {
        const auto& reference_id = app.scenario.stable_ids[*app.trail_reference_body];
        if (reference_id != configuration.removed_body_stable_id &&
            std::find(configuration.monitored_body_stable_ids.begin(),
                configuration.monitored_body_stable_ids.end(), reference_id) ==
                configuration.monitored_body_stable_ids.end()) {
            configuration.retained_reference_body_stable_ids.push_back(reference_id);
        }
    }
    return configuration;
}

nbody::experiments::ExperimentSnapshot make_ui_experiment_snapshot(
    const ApplicationModel& app) {
    const auto& state = app.experiment_ui.capture_current_state
        ? app.controller.state() : app.controller.initial_state();
    return nbody::visualization::capture_counterfactual_snapshot(
        app.scenario, state);
}

struct UiActions {
    std::optional<ScenarioKind> scenario;
    bool apply_synthetic_configuration{false};
    std::optional<std::vector<std::string>> catalog_subset;
    std::optional<nbody::visualization::CameraPreset> camera_preset;
    bool reset{false};
    bool advance_one_step{false};
    bool frame_focus{false};
    float panel_right{0.0F};
};

void section_heading(const char* label) {
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.43F, 0.86F, 0.82F, 1.0F));
    ImGui::SetWindowFontScale(1.07F);
    ImGui::TextUnformatted(label);
    ImGui::SetWindowFontScale(1.0F);
    ImGui::PopStyleColor();
    ImGui::Separator();
}

void help_for_last_item(nbody::visualization::HelpTopic topic) {
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
        return;
    }
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(std::min(420.0F, ImGui::GetIO().DisplaySize.x * 0.42F));
    ImGui::TextUnformatted(nbody::visualization::help_text(topic));
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void property_row(const char* label, const std::string& value) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", label);
    ImGui::PopStyleColor();
    ImGui::TableSetColumnIndex(1);
    ImGui::TextWrapped("%s", value.c_str());
}

bool begin_property_table(const char* id) {
    const float available_width = ImGui::GetContentRegionAvail().x;
    if (!ImGui::BeginTable(
            id, 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        return false;
    }
    const float label_width = std::clamp(
        available_width * 0.40F, 110.0F, 170.0F);
    ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, label_width);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
    return true;
}

void tooltip_for_last_item(const char* text) {
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
        return;
    }
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(std::min(420.0F, ImGui::GetIO().DisplaySize.x * 0.42F));
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void draw_counterfactual_legend() {
    const ImVec2 start = ImGui::GetCursorScreenPos();
    auto* draw_list = ImGui::GetWindowDrawList();
    constexpr float line_width = 34.0F;
    constexpr float row_height = 18.0F;
    const float middle = start.y + row_height * 0.5F;
    draw_list->AddLine(
        {start.x, middle}, {start.x + line_width, middle},
        IM_COL32(122, 209, 255, 255), 2.0F);
    ImGui::SetCursorScreenPos({start.x + line_width + 8.0F, start.y});
    ImGui::TextUnformatted("Baseline (solid)");
    const ImVec2 second = ImGui::GetCursorScreenPos();
    for (int segment = 0; segment < 4; ++segment) {
        const float x = second.x + static_cast<float>(segment) * 9.0F;
        draw_list->AddLine(
            {x, second.y + row_height * 0.5F},
            {x + 5.0F, second.y + row_height * 0.5F},
            IM_COL32(255, 97, 194, 255), 2.0F);
    }
    ImGui::SetCursorScreenPos({second.x + line_width + 8.0F, second.y});
    ImGui::TextUnformatted("Counterfactual (dashed)");
}

nbody::visualization::CounterfactualResultSortKey counterfactual_sort_key(
    ImGuiID column_id) {
    using Key = nbody::visualization::CounterfactualResultSortKey;
    switch (column_id) {
        case 1U: return Key::display_name;
        case 2U: return Key::final_absolute_separation;
        case 3U: return Key::maximum_absolute_separation;
        case 4U: return Key::normalization_scale;
        case 5U: return Key::normalized_deviation;
        case 6U: return Key::velocity_difference;
        case 7U: return Key::parent_relative_separation;
        case 8U: return Key::first_threshold_crossing;
        case 9U: return Key::persistence_time;
        case 10U: return Key::significance;
        default: return Key::stable_id;
    }
}

void draw_counterfactual_result_table(
    std::vector<nbody::visualization::CounterfactualResultRow>& rows) {
    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders |
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
        ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable |
        ImGuiTableFlags_Sortable | ImGuiTableFlags_ScrollX |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("counterfactual_body_results", 10, flags,
            {0.0F, std::min(330.0F, ImGui::GetIO().DisplaySize.y * 0.40F)})) {
        return;
    }
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("Body / stable ID",
        ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
        170.0F, 1U);
    ImGui::TableSetupColumn("Final separation (m)", 0, 130.0F, 2U);
    ImGui::TableSetupColumn("Maximum separation (m)", 0, 145.0F, 3U);
    ImGui::TableSetupColumn("Normalization scale (m)", 0, 145.0F, 4U);
    ImGui::TableSetupColumn("Normalized deviation", 0, 130.0F, 5U);
    ImGui::TableSetupColumn("Velocity difference (m/s)", 0, 150.0F, 6U);
    ImGui::TableSetupColumn("Parent-relative (m)", 0, 130.0F, 7U);
    ImGui::TableSetupColumn("First crossing (s)", 0, 120.0F, 8U);
    ImGui::TableSetupColumn("Persistence reached (s)", 0, 140.0F, 9U);
    ImGui::TableSetupColumn("Significant", 0, 90.0F, 10U);
    ImGui::TableHeadersRow();
    if (auto* specs = ImGui::TableGetSortSpecs();
        specs != nullptr && specs->SpecsDirty && specs->SpecsCount > 0) {
        const auto& spec = specs->Specs[0];
        nbody::visualization::sort_counterfactual_result_rows(
            rows, counterfactual_sort_key(spec.ColumnUserID),
            spec.SortDirection != ImGuiSortDirection_Descending);
        specs->SpecsDirty = false;
    }
    for (const auto& row : rows) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(row.display_name.c_str());
        ImGui::TextDisabled("%s", row.stable_id.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.6e", row.final_absolute_separation_m);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.6e", row.maximum_absolute_separation_m);
        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.6e", row.fixed_normalization_scale_m);
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%.6e", row.final_normalized_deviation);
        ImGui::TableSetColumnIndex(5);
        ImGui::Text("%.6e", row.final_velocity_difference_m_s);
        ImGui::TableSetColumnIndex(6);
        if (row.final_parent_relative_separation_m) {
            ImGui::Text("%.6e", *row.final_parent_relative_separation_m);
        } else {
            ImGui::TextDisabled("N/A");
        }
        ImGui::TableSetColumnIndex(7);
        if (row.first_threshold_crossing_time_s) {
            ImGui::Text("%.6e", *row.first_threshold_crossing_time_s);
        } else {
            ImGui::TextDisabled("N/A");
        }
        ImGui::TableSetColumnIndex(8);
        if (row.persistent_threshold_time_s) {
            ImGui::Text("%.6e", *row.persistent_threshold_time_s);
        } else {
            ImGui::TextDisabled("N/A");
        }
        ImGui::TableSetColumnIndex(9);
        if (row.significant) {
            ImGui::TextColored(ImVec4(1.0F, 0.73F, 0.25F, 1.0F), "YES");
        } else {
            ImGui::TextDisabled("No");
        }
    }
    ImGui::EndTable();
}

void draw_counterfactual_conservation(
    const char* branch,
    const nbody::experiments::BranchConservationDiagnostics& diagnostics) {
    ImGui::SeparatorText(branch);
    if (begin_property_table(branch)) {
        property_row("Relative energy drift",
            nbody::visualization::format_measurement(
                diagnostics.relative_energy_drift, "dimensionless", 6));
        property_row("Absolute energy drift",
            nbody::visualization::format_measurement(
                diagnostics.absolute_energy_drift_j, "J", 6));
        property_row("Absolute linear-momentum drift",
            nbody::visualization::format_measurement(
                diagnostics.linear_momentum_drift_kg_m_s, "kg m/s", 6));
        property_row("Absolute angular-momentum drift",
            nbody::visualization::format_measurement(
                diagnostics.angular_momentum_drift_kg_m2_s, "kg m^2/s", 6));
        ImGui::EndTable();
    }
}

void draw_counterfactual_warning_groups(
    const nbody::experiments::CounterfactualExperimentResult& result,
    const std::vector<nbody::visualization::CounterfactualWarningGroup>& groups) {
    ImGui::Text("Retained warnings: %zu in %zu branch/type groups",
        result.issues.size(), groups.size());
    if (result.dropped_issue_count > 0U) {
        ImGui::TextColored(ImVec4(0.96F, 0.67F, 0.28F, 1.0F),
            "%zu additional warnings were dropped by the configured storage bound.",
            result.dropped_issue_count);
    }
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto& group = groups[group_index];
        ImGui::PushID(static_cast<int>(group_index));
        const bool open = ImGui::TreeNodeEx("warning_group",
            ImGuiTreeNodeFlags_SpanAvailWidth,
            "%s · %s — %zu retained",
            nbody::visualization::counterfactual_issue_branch_name(group.branch),
            nbody::visualization::counterfactual_issue_kind_name(group.kind),
            group.retained_issue_indices.size());
        if (open) {
            for (const auto issue_index : group.retained_issue_indices) {
                if (issue_index >= result.issues.size()) {
                    ImGui::TextColored(ImVec4(1.0F, 0.32F, 0.28F, 1.0F),
                        "Invalid retained warning index");
                    continue;
                }
                const auto& issue = result.issues[issue_index];
                ImGui::PushID(static_cast<int>(issue_index));
                ImGui::BulletText("t=%.6e s%s", issue.time_s,
                    issue.numerical_invalidity ? " · numerical invalidity" : "");
                if (issue.first_body_stable_id || issue.second_body_stable_id) {
                    ImGui::TextDisabled("Bodies: %s%s%s",
                        issue.first_body_stable_id
                            ? issue.first_body_stable_id->c_str() : "N/A",
                        issue.second_body_stable_id ? " / " : "",
                        issue.second_body_stable_id
                            ? issue.second_body_stable_id->c_str() : "");
                }
                ImGui::TextWrapped("%s", issue.message.c_str());
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
}

void start_counterfactual(ApplicationModel& app, int threads) {
    try {
        auto snapshot = make_ui_experiment_snapshot(app);
        auto configuration = make_experiment_configuration(app, threads);
        const auto scenario_generation = app.experiment_gate.scenario_generation();
        const auto generation = app.experiment_worker.start(
            std::move(snapshot), std::move(configuration),
            {scenario_generation, app.experiment_ui.capture_current_state
                ? nbody::visualization::CounterfactualSnapshotSource::current_state
                : nbody::visualization::CounterfactualSnapshotSource::initial_state});
        app.experiment_gate.accept_job(generation, scenario_generation);
        app.controller.set_paused(true);
        app.experiment_ui.comparison_view = 2;
        app.experiment_ui.error.clear();
        app.refresh_experiment_publication();
    } catch (const std::exception& error) {
        app.experiment_ui.error = error.what();
    }
}

void draw_demo_controls(ApplicationModel& app, UiActions& actions, int threads) {
    int scene = app.scenario.kind == ScenarioKind::solar_system ? 0 :
        app.scenario.kind == ScenarioKind::astronomical_catalog ? 1 : 2;
    ImGui::TextUnformatted("Scene");
    ImGui::SameLine(68.0F);
    ImGui::SetNextItemWidth(-1.0F);
    if (ImGui::Combo("##demo_scene", &scene,
            "Legacy Solar (9)\0Horizons Solar (60)\0Synthetic laboratory\0")) {
        actions.scenario = scene == 0 ? ScenarioKind::solar_system :
            scene == 1 ? ScenarioKind::astronomical_catalog : ScenarioKind::synthetic;
    }
    ImGui::TextUnformatted("Backend");
    ImGui::SameLine(68.0F);
    for (int index = 0; index < 3; ++index) {
        const auto backend = static_cast<BackendKind>(index);
        const bool available = index == 0 || (index == 1
            ? nbody::visualization::SimulationController::openmp_available()
            : app.cuda_device_available && app.controller.solver_kind() == SolverKind::direct);
        if (index) ImGui::SameLine();
        ImGui::BeginDisabled(!available);
        if (ImGui::RadioButton(index == 0 ? "Sequential" : index == 1 ? "OpenMP" : "CUDA",
                              app.controller.backend_kind() == backend)) {
            try { app.select_backend_admitted(backend, threads, app.preferred_cuda_device); app.solver_error.clear(); }
            catch (const std::exception& error) { app.solver_error = error.what(); }
        }
        ImGui::EndDisabled();
    }
    if (!nbody::visualization::SimulationController::openmp_available()) {
        ImGui::TextWrapped("OpenMP not compiled: rebuild the CUDA preset with NBODY_ENABLE_OPENMP=ON.");
    }
    ImGui::TextUnformatted("Solver");
    ImGui::SameLine(68.0F);
    for (int index = 0; index < 2; ++index) {
        if (index) ImGui::SameLine();
        const auto solver = static_cast<SolverKind>(index);
        ImGui::BeginDisabled(index == 1 && app.controller.backend_kind() == BackendKind::cuda);
        if (ImGui::RadioButton(index == 0 ? "Direct" : "Barnes-Hut",
                              app.controller.solver_kind() == solver)) {
            try { app.select_solver_admitted(solver, app.controller.barnes_hut_configuration()); app.solver_error.clear(); }
            catch (const std::exception& error) { app.solver_error = error.what(); }
        }
        ImGui::EndDisabled();
    }
    if (app.controller.solver_kind() == SolverKind::barnes_hut) {
        ImGui::Text("Theta %.2f", app.controller.barnes_hut_configuration().theta);
        if (const auto statistics = app.controller.barnes_hut_statistics()) {
            ImGui::Text("Executed tree: %zu nodes, %d workers", statistics->node_count, statistics->worker_threads);
        } else { ImGui::TextDisabled("Advance once to build the tree."); }
    } else if (app.controller.backend_kind() == BackendKind::cuda) {
        ImGui::TextWrapped("CUDA Direct; select OpenMP to use Barnes-Hut. GPU batches capped at 4 steps for responsiveness.");
    }
    const auto policy = app.workload_policy();
    ImGui::BeginDisabled(!policy.continuous_advancement);
    if (ImGui::Button(app.controller.paused() ? "Start / Resume" : "Pause")) {
        app.experiment_ui.comparison_view = 0;
        app.controller.toggle_paused();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!policy.manual_single_step);
    if (ImGui::Button("Step")) {
        app.experiment_ui.comparison_view = 0;
        actions.advance_one_step = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Reset")) actions.reset = true;
    if (!policy.continuous_advancement) ImGui::TextWrapped("%s", policy.explanation);
    std::vector<const char*> speeds;
    for (const auto& preset : app.scenario.speed_presets) speeds.push_back(preset.name);
    int speed = static_cast<int>(app.speed_index);
    ImGui::TextUnformatted("Speed");
    ImGui::SameLine(68.0F);
    ImGui::SetNextItemWidth(-1.0F);
    if (ImGui::Combo("##demo_speed", &speed, speeds.data(), static_cast<int>(speeds.size()))) {
        app.speed_index = static_cast<std::size_t>(speed);
        app.set_steps_per_frame(app.scenario.speed_presets[app.speed_index].steps_per_frame);
    }
    static int view = 0;
    if (ImGui::Button("Reframe")) actions.camera_preset = static_cast<nbody::visualization::CameraPreset>(view);
    ImGui::SameLine(82.0F);
    ImGui::SetNextItemWidth(-1.0F);
    if (ImGui::Combo("##demo_view", &view,
            "Full System\0Inner System\0Earth-Moon\0Jupiter\0Saturn\0Uranus\0Neptune-Triton\0Pluto-Charon\0Asteroid Belt\0")) {
        actions.camera_preset = static_cast<nbody::visualization::CameraPreset>(view);
    }
    tooltip_for_last_item("Left drag: full orbit. Right drag: pan. Wheel: proportional zoom. Reframe restores a preset.");
    if (app.astronomical()) {
        std::vector<const char*> names;
        for (const auto& body : app.controller.state().bodies) names.push_back(body.name.c_str());
        int selected = static_cast<int>(app.selected_body);
        ImGui::TextUnformatted("Body");
        ImGui::SameLine(68.0F);
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::Combo("##demo_body", &selected, names.data(), static_cast<int>(names.size()))) app.selected_body = static_cast<std::size_t>(selected);
        if (ImGui::Button("Focus body")) actions.frame_focus = true;
        ImGui::SameLine();
        bool visible = app.individual_visibility[app.selected_body];
        if (ImGui::Checkbox("Visible", &visible)) app.individual_visibility[app.selected_body] = visible;
        const auto candidates = nbody::visualization::counterfactual_removal_candidates(app.scenario);
        const auto& id = app.scenario.stable_ids[app.selected_body];
        ImGui::SameLine();
        ImGui::BeginDisabled(std::find(candidates.begin(), candidates.end(), id) == candidates.end());
        if (ImGui::Button("Remove / compare")) {
            app.experiment_ui.removed_body_stable_id = id;
            app.experiment_ui.backend_index = static_cast<int>(app.controller.backend_kind());
            app.experiment_ui.solver_index = static_cast<int>(app.controller.solver_kind());
            app.panel_page = 1;
        }
        ImGui::EndDisabled();
    }
    ImGui::Checkbox("History trails", &app.show_trails);
    ImGui::SameLine();
    ImGui::Checkbox("Orbit guides", &app.show_reference_orbits);
    if (app.scenario.kind == ScenarioKind::astronomical_catalog) {
        ImGui::TextWrapped("Guides = initial osculating ellipses. History = completed integration.");
    }
    if (!app.solver_error.empty()) ImGui::TextWrapped("%s", app.solver_error.c_str());
    if (!app.view_error.empty()) ImGui::TextWrapped("%s", app.view_error.c_str());
}

UiActions draw_control_panel(
    ApplicationModel& app,
    int& preferred_openmp_threads) {
    UiActions actions;
    app.refresh_experiment_publication();
    const auto width_policy = nbody::visualization::panel_width_policy(
        ImGui::GetIO().DisplaySize.x, app.panel_width);
    ImGui::SetNextWindowPos({16.0F, 16.0F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({width_policy.width, 920.0F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        {width_policy.minimum, 480.0F},
        {width_policy.maximum, std::max(480.0F, ImGui::GetIO().DisplaySize.y - 24.0F)});
    ImGui::Begin("Scientific Visualization", nullptr, ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    app.panel_width = ImGui::GetWindowSize().x;
    ImGui::SetScrollY(0.0F);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.96F, 0.79F, 0.35F, 1.0F));
    ImGui::SetWindowFontScale(1.20F);
    ImGui::TextUnformatted("N-body Dynamics");
    ImGui::SetWindowFontScale(1.0F);
    ImGui::PopStyleColor();

    const auto& body = app.controller.state().bodies[app.selected_body];
    const auto& executed = app.controller.last_execution();
    ImGui::TextWrapped("Configured LIVE solver: %s / %s | %zu bodies | %s",
        app.controller.backend_name(), app.controller.solver_name(),
        app.controller.state().bodies.size(), app.controller.paused() ? "PAUSED" : "RUNNING");
    if (executed) {
        ImGui::TextColored(ImVec4(0.40F, 0.88F, 0.62F, 1.0F),
            "ACTIVE EXECUTION: %s / %s",
            executed->backend == BackendKind::cuda ? "CUDA" : executed->backend == BackendKind::openmp ? "OpenMP" : "Sequential",
            executed->solver == SolverKind::direct ? "Direct" : "Barnes-Hut");
        if (app.panel_page == 2) ImGui::Text("Last commit: %zu steps, t=%.0f s", executed->steps, executed->end_time_s);
    } else { ImGui::TextDisabled("ACTIVE EXECUTION: no step committed since selection/reset"); }
    ImGui::Text("Display %.1f FPS", ImGui::GetIO().Framerate);
    if (const auto& timing = app.simulation_timing.sample()) {
        ImGui::SameLine(); ImGui::Text("| physics %.1f steps/s (%.2f ms)",
            timing->simulation_steps_per_second, timing->batch_milliseconds);
    }
    if (!app.simulation_error.empty()) ImGui::TextWrapped("Paused: %s", app.simulation_error.c_str());
    if (app.experiment_gate.accepts(app.experiment_publication)) {
        const auto& publication = app.experiment_publication;
        ImGui::TextWrapped("Paired experiment: %s / %s — %s",
            publication.run ? nbody::experiments::experiment_backend_name(publication.run->backend) : "unknown",
            publication.run ? nbody::experiments::experiment_solver_name(publication.run->solver) : "unknown",
            nbody::visualization::counterfactual_worker_state_name(publication.state));
        if (publication.state == nbody::visualization::CounterfactualWorkerState::completed && publication.result) {
            ImGui::RadioButton("Baseline", &app.experiment_ui.comparison_view, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Body removed", &app.experiment_ui.comparison_view, 2);
            ImGui::TextWrapped("Paired t = %.0f s. Solid = baseline; dashed pink = counterfactual. Separation is not amplified.",
                publication.result->baseline_state.time_s);
            if (app.experiment_ui.comparison_view != 0 && app.local_origin_body &&
                app.scenario.stable_ids[*app.local_origin_body] == publication.removed_body_stable_id) {
                ImGui::TextWrapped("Comparison unavailable in the removed body's frame. Select Full System or a surviving parent view.");
            }
            if (ImGui::Button("Return to live baseline")) app.experiment_ui.comparison_view = 0;
        }
        if (!publication.worker_error.empty()) ImGui::TextWrapped("%s", publication.worker_error.c_str());
    }
    const bool showing_comparison = app.experiment_ui.comparison_view != 0 &&
        publication_matches_scenario(app) &&
        app.experiment_publication.state == nbody::visualization::CounterfactualWorkerState::completed &&
        app.experiment_publication.result &&
        (!app.local_origin_body || app.scenario.stable_ids[*app.local_origin_body] !=
            app.experiment_publication.removed_body_stable_id);
    ImGui::TextWrapped("DISPLAY: %s", showing_comparison
        ? (app.experiment_ui.comparison_view == 1 ? "PAIRED BASELINE endpoint" : "BODY-REMOVED endpoint")
        : "LIVE BASELINE (comparison may be pending/unavailable)");
    ImGui::RadioButton("Demo", &app.panel_page, 0); ImGui::SameLine();
    ImGui::RadioButton("Comparison", &app.panel_page, 1); ImGui::SameLine();
    ImGui::RadioButton("Advanced", &app.panel_page, 2);
    ImGui::Separator();
    ImGui::BeginChild("control_page", {0.0F, 0.0F}, false);
    if (app.previous_panel_page != app.panel_page) {
        ImGui::SetScrollY(0.0F);
        app.previous_panel_page = app.panel_page;
    }
    if (app.panel_page == 0) draw_demo_controls(app, actions, preferred_openmp_threads);
    if (app.panel_page == 2) {

    section_heading("Simulation");
    int scenario_index = app.scenario.kind == ScenarioKind::solar_system ? 0 :
        app.scenario.kind == ScenarioKind::astronomical_catalog ? 1 : 2;
    const char* scenarios[] = {
        "Legacy Solar System (9)", "Horizons Solar System Sandbox (60)",
        "Compute Experiment"};
    ImGui::SetNextItemWidth(-1.0F);
    if (ImGui::Combo("Scenario", &scenario_index, scenarios, 3)) {
        actions.scenario = scenario_index == 0 ? ScenarioKind::solar_system :
            scenario_index == 1 ? ScenarioKind::astronomical_catalog : ScenarioKind::synthetic;
    }
    help_for_last_item(nbody::visualization::HelpTopic::scenario);
    if (app.scenario.kind == ScenarioKind::synthetic) {
        ImGui::TextWrapped(
            "Deterministic synthetic scene: N=%zu, distribution=%s, seed=%llu",
            app.synthetic_body_count,
            nbody::visualization::synthetic_distribution_name(
                app.synthetic_distribution),
            static_cast<unsigned long long>(app.synthetic_seed));
    }
    if (app.scenario.kind == ScenarioKind::astronomical_catalog &&
        ImGui::CollapsingHeader("Physical solver composition")) {
        std::size_t included_count = static_cast<std::size_t>(std::count(
            app.catalog_inclusion_draft.begin(), app.catalog_inclusion_draft.end(), true));
        ImGui::TextWrapped(
            "%zu of 60 catalog bodies selected for gravitational integration. "
            "This is distinct from rendering visibility.", included_count);
        if (ImGui::Button("Select all")) {
            std::fill(app.catalog_inclusion_draft.begin(),
                      app.catalog_inclusion_draft.end(), true);
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear all")) {
            std::fill(app.catalog_inclusion_draft.begin(),
                      app.catalog_inclusion_draft.end(), false);
        }
        ImGui::BeginChild("catalog_composition", {0.0F, 210.0F}, true);
        for (std::size_t index = 0; index < app.catalog_roster_ids.size(); ++index) {
            const std::string label = app.catalog_roster_names[index] + "##include_" +
                app.catalog_roster_ids[index];
            bool included = app.catalog_inclusion_draft[index];
            if (ImGui::Checkbox(label.c_str(), &included)) {
                app.catalog_inclusion_draft[index] = included;
            }
        }
        ImGui::EndChild();
        included_count = static_cast<std::size_t>(std::count(
            app.catalog_inclusion_draft.begin(), app.catalog_inclusion_draft.end(), true));
        ImGui::BeginDisabled(included_count < 2U);
        if (ImGui::Button("Apply physical subset")) {
            std::vector<std::string> selected_ids;
            for (std::size_t index = 0; index < app.catalog_roster_ids.size(); ++index) {
                if (app.catalog_inclusion_draft[index]) {
                    selected_ids.push_back(app.catalog_roster_ids[index]);
                }
            }
            actions.catalog_subset = std::move(selected_ids);
        }
        ImGui::EndDisabled();
        for (const auto& warning : app.scenario.composition_warnings) {
            ImGui::TextColored(ImVec4(0.96F, 0.67F, 0.28F, 1.0F), "%s", warning.c_str());
        }
    }
    const auto workload_policy = app.workload_policy();
    const bool continuous_controls = app.scenario.kind != ScenarioKind::synthetic ||
        workload_policy.continuous_advancement;
    ImGui::BeginDisabled(!continuous_controls);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16F, 0.38F, 0.42F, 1.0F));
    if (ImGui::Button(app.controller.paused() ? "Resume" : "Pause", {118.0F, 0.0F})) {
        app.experiment_ui.comparison_view = 0;
        app.controller.toggle_paused();
    }
    ImGui::PopStyleColor();
    help_for_last_item(nbody::visualization::HelpTopic::pause);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Reset", {92.0F, 0.0F})) {
        actions.reset = true;
    }
    help_for_last_item(nbody::visualization::HelpTopic::reset);
    ImGui::SameLine();
    ImGui::TextColored(
        app.controller.paused() ? ImVec4(0.92F, 0.72F, 0.35F, 1.0F)
                                : ImVec4(0.40F, 0.88F, 0.62F, 1.0F),
        "%s", app.controller.paused() ? "Paused" : "Running");
    if (begin_property_table("simulation_summary")) {
        property_row("Active scenario", app.scenario.name);
        property_row("Bodies", std::to_string(app.controller.state().bodies.size()));
        property_row("Simulated time", human_time(app.controller.state().time_s));
        property_row("Fixed timestep",
            nbody::visualization::format_measurement(app.controller.dt_s(), "s", 3));
        const auto& simulation_timing = app.simulation_timing.sample();
        property_row("Live simulation timing",
            simulation_timing
                ? nbody::visualization::format_measurement(
                      simulation_timing->batch_milliseconds, "ms/batch", 3) + "; " +
                      nbody::visualization::format_measurement(
                          simulation_timing->simulation_steps_per_second,
                          "steps/s", 3)
                : "Not measured for the active execution configuration");
        ImGui::EndTable();
    }

    std::vector<const char*> speed_names;
    for (const auto& preset : app.scenario.speed_presets) {
        speed_names.push_back(preset.name);
    }
    int speed_index = static_cast<int>(app.speed_index);
    ImGui::TextUnformatted("Simulation speed");
    ImGui::SetNextItemWidth(-1.0F);
    if (ImGui::Combo("##SimulationSpeed", &speed_index, speed_names.data(),
                     static_cast<int>(speed_names.size()))) {
        app.speed_index = static_cast<std::size_t>(speed_index);
        app.set_steps_per_frame(
            app.scenario.speed_presets[app.speed_index].steps_per_frame);
    }
    help_for_last_item(nbody::visualization::HelpTopic::simulation_speed);
    ImGui::TextWrapped(
        "This changes fixed-step batching per frame, not the numerical integration timestep.");

    section_heading("Compute Backend");
    if (ImGui::RadioButton("Sequential", app.controller.backend_kind() == BackendKind::sequential)) {
        try {
            app.select_backend_admitted(BackendKind::sequential);
            app.solver_error.clear();
            app.live_comparison.reset();
        } catch (const std::exception& error) {
            app.solver_error = error.what();
        }
    }
    help_for_last_item(nbody::visualization::HelpTopic::sequential_backend);
    const bool openmp_available = nbody::visualization::SimulationController::openmp_available();
    ImGui::BeginDisabled(!openmp_available);
    if (ImGui::RadioButton("OpenMP", app.controller.backend_kind() == BackendKind::openmp)) {
        try {
            app.select_backend_admitted(
                BackendKind::openmp, preferred_openmp_threads);
            app.solver_error.clear();
            app.live_comparison.reset();
        } catch (const std::exception& error) {
            app.solver_error = error.what();
        }
    }
    help_for_last_item(nbody::visualization::HelpTopic::openmp_backend);
    ImGui::EndDisabled();
    const bool cuda_barnes_hut_blocked =
        app.controller.solver_kind() == SolverKind::barnes_hut;
    ImGui::BeginDisabled(!app.cuda_device_available || cuda_barnes_hut_blocked);
    if (ImGui::RadioButton("CUDA", app.controller.backend_kind() == BackendKind::cuda)) {
        try {
            app.select_backend_admitted(
                BackendKind::cuda, 1, app.preferred_cuda_device);
            app.solver_error.clear();
            app.live_comparison.reset();
        } catch (const std::exception& error) {
            app.solver_error = error.what();
        }
    }
    help_for_last_item(nbody::visualization::HelpTopic::cuda_backend);
    ImGui::EndDisabled();
    if (!app.cuda_device_available) {
        ImGui::TextDisabled(
            nbody::visualization::SimulationController::cuda_compiled()
                ? "CUDA device unavailable"
                : "CUDA support not compiled");
    } else if (cuda_barnes_hut_blocked) {
        ImGui::TextDisabled("CUDA Barnes-Hut viewer execution is pending NVIDIA validation");
    }
    if (app.controller.backend_kind() == BackendKind::openmp ||
        app.scenario.kind == ScenarioKind::synthetic) {
        const int hardware_threads =
            static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
        const int maximum_threads = std::min(64, hardware_threads);
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::SliderInt(
                "OpenMP threads", &preferred_openmp_threads, 1, maximum_threads,
                "%d threads")) {
            if (app.controller.backend_kind() == BackendKind::openmp) {
                try {
                    app.select_backend_admitted(
                        BackendKind::openmp, preferred_openmp_threads);
                    app.solver_error.clear();
                } catch (const std::exception& error) {
                    app.solver_error = error.what();
                }
            }
            app.live_comparison.reset();
            app.live_comparison_error.clear();
        }
        help_for_last_item(nbody::visualization::HelpTopic::openmp_threads);
    }

    section_heading("Force Solver");
    if (ImGui::RadioButton(
            "Direct O(N^2)", app.controller.solver_kind() == SolverKind::direct)) {
        try {
            app.select_solver_admitted(
                SolverKind::direct, app.controller.barnes_hut_configuration());
            app.solver_error.clear();
            app.live_comparison.reset();
            app.live_comparison_error.clear();
        } catch (const std::exception& error) {
            app.solver_error = error.what();
        }
    }
    ImGui::BeginDisabled(app.controller.backend_kind() == BackendKind::cuda);
    if (ImGui::RadioButton(
            "Barnes-Hut", app.controller.solver_kind() == SolverKind::barnes_hut)) {
        try {
            auto configuration = app.controller.barnes_hut_configuration();
            configuration.theta = app.barnes_hut_theta_draft;
            app.select_solver_admitted(SolverKind::barnes_hut, configuration);
            app.solver_error.clear();
            app.live_comparison.reset();
            app.live_comparison_error.clear();
        } catch (const std::exception& error) {
            app.solver_error = error.what();
        }
    }
    ImGui::EndDisabled();
    if (app.controller.backend_kind() == BackendKind::cuda) {
        ImGui::TextDisabled(
            "Barnes-Hut is unavailable with CUDA in the viewer until RTX validation completes.");
    }

    ImGui::TextUnformatted("Barnes-Hut opening angle theta");
    ImGui::SetNextItemWidth(-1.0F);
    ImGui::InputDouble(
        "##BarnesHutTheta", &app.barnes_hut_theta_draft, 0.01, 0.05, "%.4f");
    const bool valid_theta = std::isfinite(app.barnes_hut_theta_draft) &&
        app.barnes_hut_theta_draft > 0.0 && app.barnes_hut_theta_draft <= 2.0;
    const bool theta_changed = valid_theta &&
        app.barnes_hut_theta_draft !=
            app.controller.barnes_hut_configuration().theta;
    ImGui::BeginDisabled(
        app.controller.solver_kind() != SolverKind::barnes_hut || !theta_changed);
    if (ImGui::Button("Apply theta")) {
        try {
            auto configuration = app.controller.barnes_hut_configuration();
            configuration.theta = app.barnes_hut_theta_draft;
            app.select_solver_admitted(SolverKind::barnes_hut, configuration);
            app.solver_error.clear();
            app.live_comparison.reset();
            app.live_comparison_error.clear();
        } catch (const std::exception& error) {
            app.solver_error = error.what();
        }
    }
    ImGui::EndDisabled();
    if (!valid_theta) {
        ImGui::TextColored(
            ImVec4(0.96F, 0.48F, 0.38F, 1.0F),
            "Theta must be finite and in (0, 2]. The effective solver is unchanged.");
    }
    if (begin_property_table("effective_solver")) {
        property_row("Effective execution",
            std::string(app.controller.backend_name()) + " / " +
                app.controller.solver_name());
        property_row("Effective theta",
            app.controller.solver_kind() == SolverKind::barnes_hut
                ? nbody::visualization::format_measurement(
                    app.controller.barnes_hut_configuration().theta, "", 4)
                : "Not used by direct force");
        if (app.controller.solver_kind() == SolverKind::barnes_hut) {
            const auto& limits = app.controller.barnes_hut_configuration();
            property_row("Tree bounds",
                std::to_string(limits.maximum_nodes) + " nodes; " +
                    std::to_string(limits.maximum_traversal_work) + " work units");
            property_row("Memory bound",
                std::to_string(limits.memory_budget_bytes / (1024U * 1024U)) + " MiB");
        }
        ImGui::EndTable();
    }
    if (const auto statistics = app.controller.barnes_hut_statistics()) {
        ImGui::TextDisabled(
            "Last force: %zu nodes, %zu visited, %zu bytes, %.3f ms tree + %.3f ms force",
            statistics->node_count,
            statistics->visited_nodes,
            statistics->working_memory_bytes,
            statistics->tree_build_seconds * 1000.0,
            statistics->force_evaluation_seconds * 1000.0);
    }
    if (!app.solver_error.empty()) {
        ImGui::TextColored(
            ImVec4(0.96F, 0.48F, 0.38F, 1.0F), "%s", app.solver_error.c_str());
    }
    if (!app.simulation_error.empty() || !app.controller.last_error().empty()) {
        const std::string& error = app.simulation_error.empty()
            ? app.controller.last_error() : app.simulation_error;
        ImGui::TextColored(
            ImVec4(0.96F, 0.38F, 0.32F, 1.0F), "Simulation paused: %s", error.c_str());
    }
    if (app.astronomical()) {
        if (app.controller.solver_kind() == SolverKind::direct) {
            ImGui::TextWrapped(
                "This astronomical scenario uses direct O(N^2) Velocity Verlet. Backend speedup is not "
                "research evidence at this small body count.");
        } else {
            ImGui::TextWrapped(
                "Barnes-Hut uses the same Velocity Verlet timestep and Newtonian force law with a "
                "monopole approximation. Theta 0.25 is provisional and distribution-dependent.");
        }
    } else {
        const std::string current_count = std::to_string(app.synthetic_body_count_draft);
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::BeginCombo("Workload size", current_count.c_str())) {
            for (const std::size_t count : nbody::visualization::compute_body_counts) {
                const std::string label = std::to_string(count);
                if (ImGui::Selectable(
                        label.c_str(), app.synthetic_body_count_draft == count)) {
                    app.synthetic_body_count_draft = count;
                }
            }
            ImGui::EndCombo();
        }
        help_for_last_item(nbody::visualization::HelpTopic::workload_size);
        constexpr const char* distribution_names[] = {
            "Uniform volume", "Four clusters", "Thin planar disk"};
        int distribution_index = static_cast<int>(app.synthetic_distribution_draft);
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::Combo(
                "Distribution", &distribution_index, distribution_names, 3)) {
            app.synthetic_distribution_draft =
                static_cast<LargeNDistribution>(distribution_index);
        }
        unsigned long long seed =
            static_cast<unsigned long long>(app.synthetic_seed_draft);
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::InputScalar("Deterministic seed", ImGuiDataType_U64, &seed)) {
            app.synthetic_seed_draft = static_cast<std::uint64_t>(seed);
        }
        if (ImGui::Button("Apply synthetic scene", {220.0F, 0.0F})) {
            actions.apply_synthetic_configuration = true;
        }
        ImGui::TextDisabled(
            "Applying replaces the physical scene and invalidates prior diagnostics and overlays.");
        if (workload_policy.warning_required) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.96F, 0.72F, 0.32F, 1.0F));
            ImGui::TextWrapped("%s", workload_policy.explanation);
            if (!workload_policy.manual_single_step) {
                ImGui::TextWrapped(
                    "This size is rendering/inspection only here; use the bounded pilot runner for computation.");
            }
            ImGui::PopStyleColor();
            ImGui::BeginDisabled(!workload_policy.manual_single_step);
            if (ImGui::Button("Advance one fixed step")) {
                actions.advance_one_step = true;
            }
            help_for_last_item(nbody::visualization::HelpTopic::manual_large_step);
            ImGui::EndDisabled();
        }
        ImGui::BeginDisabled(!openmp_available || !workload_policy.live_cpu_comparison);
        if (ImGui::Button("Compare CPU backends", {220.0F, 0.0F})) {
            try {
                app.live_comparison = nbody::visualization::run_live_compute_comparison(
                    app.controller.initial_state(), app.controller.dt_s(), preferred_openmp_threads);
                app.live_comparison_error.clear();
            } catch (const std::exception& error) {
                app.live_comparison.reset();
                app.live_comparison_error = error.what();
            }
        }
        help_for_last_item(nbody::visualization::HelpTopic::compare_cpu_backends);
        ImGui::EndDisabled();
        ImGui::TextColored(
            ImVec4(0.92F, 0.67F, 0.31F, 1.0F),
            "LIVE DIAGNOSTIC - NOT BENCHMARK EVIDENCE");
        if (app.live_comparison && !nbody::visualization::live_comparison_matches(
                *app.live_comparison, app.synthetic_body_count, preferred_openmp_threads)) {
            app.live_comparison.reset();
        }
        if (app.live_comparison) {
            const auto& comparison = *app.live_comparison;
            if (begin_property_table("comparison_result")) {
                property_row("Workload", "N=" + std::to_string(comparison.body_count) +
                    ", " + std::to_string(comparison.integration_steps) + " fixed steps");
                property_row("Sequential", nbody::visualization::format_measurement(
                    comparison.sequential_ms, "ms", 3));
                property_row("OpenMP", nbody::visualization::format_measurement(
                    comparison.openmp_ms, "ms", 3));
                ImGui::EndTable();
            }
            if (comparison.correctness_passed) {
                ImGui::Text("Observed Seq/OpenMP ratio %.3fx",
                            comparison.sequential_to_openmp_ratio);
                if (comparison.sequential_to_openmp_ratio < 1.0) {
                    ImGui::TextDisabled("No speedup observed at this workload.");
                }
                ImGui::TextColored(
                    ImVec4(0.38F, 0.90F, 0.61F, 1.0F),
                    "Correctness pass");
            } else {
                ImGui::TextColored(
                    ImVec4(1.0F, 0.34F, 0.28F, 1.0F),
                    "Comparison invalid: correctness failure");
            }
        } else if (!app.live_comparison_error.empty()) {
            ImGui::TextColored(
                ImVec4(1.0F, 0.34F, 0.28F, 1.0F), "%s",
                app.live_comparison_error.c_str());
        }
    }

    section_heading("Body Inspector");
    if (app.astronomical()) {
        std::vector<const char*> body_names;
        for (const auto& candidate : app.controller.state().bodies) {
            body_names.push_back(candidate.name.c_str());
        }
        int selected = static_cast<int>(app.selected_body);
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::Combo("Selected body", &selected, body_names.data(),
                         static_cast<int>(body_names.size()))) {
            app.selected_body = static_cast<std::size_t>(selected);
        }
    } else {
        unsigned long long selected = static_cast<unsigned long long>(app.selected_body);
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::InputScalar("Selected body index", ImGuiDataType_U64, &selected)) {
            app.selected_body = std::min<std::size_t>(
                static_cast<std::size_t>(selected), app.controller.state().bodies.size() - 1U);
        }
    }
    help_for_last_item(nbody::visualization::HelpTopic::body_selector);
    const auto& style = app.scenario.styles[app.selected_body];
    const auto sun_index = nbody::visualization::find_body_index(
        app.scenario,
        app.scenario.kind == ScenarioKind::astronomical_catalog ? "sol:sun" : "sun");
    const std::size_t distance_reference = sun_index.value_or(0U);
    const char* distance_label = app.astronomical() && sun_index
        ? "Distance from Sun" : "Distance from solver body 0";
    if (begin_property_table("body_properties")) {
        property_row("Name", body.name);
        property_row("Stable ID", app.scenario.stable_ids[app.selected_body]);
        if (app.astronomical()) {
            const auto& metadata = app.scenario.body_metadata[app.selected_body];
            property_row("Class", nbody::visualization::astronomical_body_class_name(
                metadata.body_class));
            property_row("Catalog parent", metadata.parent_stable_id.value_or("None"));
            property_row("Future removal eligibility",
                metadata.counterfactual_removal_eligible
                    ? "Curated major-body candidate" : "Not in initial curated list");
        }
        property_row(distance_label, nbody::visualization::format_measurement(
            nbody::norm(body.position_m -
                app.controller.state().bodies[distance_reference].position_m) /
                149597870700.0, "AU", 6));
        property_row("Speed", nbody::visualization::format_measurement(
            nbody::norm(body.velocity_m_s) / 1000.0, "km/s", 6));
        property_row("Mass", nbody::visualization::format_measurement(body.mass_kg, "kg", 6));
        property_row("Mean radius", style.physical_radius_m > 0.0
            ? nbody::visualization::format_measurement(style.physical_radius_m / 1000.0, "km", 6)
            : "Not defined for this synthetic workload");
        ImGui::EndTable();
    }
    if (app.astronomical()) {
        bool visible = app.individual_visibility[app.selected_body];
        if (ImGui::Checkbox("Visible (rendering only)", &visible)) {
            app.individual_visibility[app.selected_body] = visible;
        }
        ImGui::TextDisabled(
            "Effective visibility = active view + class filter + individual setting; forces are unchanged.");
        if (ImGui::CollapsingHeader("Body-class visibility filters")) {
            constexpr std::array<nbody::astronomy::AstronomicalBodyClass, 7> classes{{
                nbody::astronomy::AstronomicalBodyClass::star,
                nbody::astronomy::AstronomicalBodyClass::planet,
                nbody::astronomy::AstronomicalBodyClass::dwarf_planet,
                nbody::astronomy::AstronomicalBodyClass::natural_satellite,
                nbody::astronomy::AstronomicalBodyClass::asteroid,
                nbody::astronomy::AstronomicalBodyClass::comet,
                nbody::astronomy::AstronomicalBodyClass::trans_neptunian_object,
            }};
            for (std::size_t class_index = 0; class_index < classes.size(); ++class_index) {
                bool enabled = app.class_visibility[class_index];
                if (ImGui::Checkbox(
                        nbody::visualization::astronomical_body_class_name(classes[class_index]),
                        &enabled)) {
                    app.class_visibility[class_index] = enabled;
                }
            }
        }
    }
    if (ImGui::Button("Focus selected", {154.0F, 0.0F})) {
        app.focused_body = app.selected_body;
        actions.frame_focus = true;
    }
    help_for_last_item(nbody::visualization::HelpTopic::focus_selected);

    section_heading("Numerical Accuracy");
    if (!app.scenario.timestep_validation_complete) {
        ImGui::TextColored(
            ImVec4(0.96F, 0.67F, 0.28F, 1.0F),
            "Provisional 60 s timestep — bounded convergence tested locally, long-horizon "
            "ephemeris accuracy not established.");
    }
    const auto health_color = [](bool okay) {
        return okay ? ImVec4(0.38F, 0.90F, 0.61F, 1.0F)
                    : ImVec4(1.0F, 0.34F, 0.28F, 1.0F);
    };
    if (app.conservation) {
        const auto& drift = app.conservation->drift();
        const auto health = nbody::visualization::classify_numerical_health(
            drift.energy_relative, drift.angular_momentum_relative);
        ImGui::TextWrapped("Energy conservation: %.3e (bound %.1e)",
                           drift.energy_relative, nbody::visualization::energy_drift_bound);
        help_for_last_item(nbody::visualization::HelpTopic::energy_conservation);
        ImGui::TextColored(health_color(health.energy_within_bound), "%s",
                           health.energy_within_bound ? "Within bound" : "Bound exceeded");
        ImGui::TextWrapped("Angular-momentum conservation: %.3e (bound %.1e)",
                           drift.angular_momentum_relative,
                           nbody::visualization::angular_momentum_drift_bound);
        help_for_last_item(nbody::visualization::HelpTopic::angular_momentum_conservation);
        ImGui::TextColored(health_color(health.angular_momentum_within_bound), "%s",
                           health.angular_momentum_within_bound ? "Within bound" : "Bound exceeded");
        ImGui::TextColored(health_color(health.all_within_bounds()), "%s",
                           health.all_within_bounds()
                               ? "All established bounds satisfied"
                               : "Numerical bound warning");
        help_for_last_item(nbody::visualization::HelpTopic::numerical_status);
    } else {
        ImGui::TextWrapped(
            "Live O(N^2) conservation diagnostics are disabled for this display-only large workload. "
            "The bounded benchmark runner retains correctness checks.");
    }

    }
    if (app.panel_page == 1) {
    section_heading("Counterfactual Experiment");
    ImGui::TextWrapped(
        "Hypothetical gravitational counterfactual: one curated body is removed from a copied "
        "state. Visibility controls do not affect this physical removal.");
    if (!app.astronomical()) {
        ImGui::TextDisabled(
            "Counterfactual experiments are available only for astronomical scenarios.");
    } else {
        auto& experiment = app.experiment_ui;
        ImGui::TextDisabled("Settings below configure the NEXT run.");
        const auto control_layout =
            nbody::visualization::counterfactual_control_layout(
                ImGui::GetContentRegionAvail().x);
        const auto candidates =
            nbody::visualization::counterfactual_removal_candidates(app.scenario);
        ImGui::TextUnformatted("Start from");
        if (ImGui::RadioButton(
                "Current simulation snapshot", experiment.capture_current_state)) {
            experiment.capture_current_state = true;
        }
        tooltip_for_last_item(
            "Copies the current simulated state and its current timestamp. The live viewer is never mutated by the experiment.");
        if (!control_layout.stacked) {
            ImGui::SameLine();
        }
        if (ImGui::RadioButton(
                "Initial scenario state (t = 0)", !experiment.capture_current_state)) {
            experiment.capture_current_state = false;
        }
        tooltip_for_last_item(
            "Copies the scenario's defined initial state at simulation time zero, independent of the live viewer's current time.");

        int removal_index = 0;
        std::vector<const char*> removal_names;
        removal_names.reserve(candidates.size());
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto body_index = nbody::visualization::find_body_index(
                app.scenario, candidates[index]);
            removal_names.push_back(body_index
                ? app.controller.state().bodies[*body_index].name.c_str()
                : candidates[index].c_str());
            if (candidates[index] == experiment.removed_body_stable_id) {
                removal_index = static_cast<int>(index);
            }
        }
        if (!removal_names.empty()) {
            ImGui::TextUnformatted("Hypothetically remove one gravitating body");
            ImGui::SetNextItemWidth(-1.0F);
            if (ImGui::Combo("##counterfactual_remove_body", &removal_index,
                    removal_names.data(), static_cast<int>(removal_names.size()))) {
                experiment.removed_body_stable_id =
                    candidates[static_cast<std::size_t>(removal_index)];
            }
            tooltip_for_last_item(
                "The selected body is physically absent from the counterfactual solver branch; this is not a visibility toggle.");
        }
        ImGui::TextDisabled("Sun and unapproved catalog bodies are not eligible.");
        ImGui::BeginDisabled(app.experiment_worker.busy() || candidates.empty());
        if (ImGui::Button("Run selected removal / show result")) start_counterfactual(app, preferred_openmp_threads);
        ImGui::EndDisabled();
        ImGui::TextWrapped("Launch pauses the live scene. Completed paired endpoints are displayed at the same time; the live solver is preserved.");
        if (!experiment.error.empty()) ImGui::TextWrapped("%s", experiment.error.c_str());

        ImGui::TextUnformatted("Experiment backend");
        ImGui::RadioButton("Sequential CPU", &experiment.backend_index, 0);
        if (!control_layout.stacked) {
            ImGui::SameLine();
        }
        const bool experiment_openmp_available =
            nbody::experiments::CounterfactualExperimentRunner::openmp_supported();
        ImGui::BeginDisabled(!experiment_openmp_available);
        ImGui::RadioButton("OpenMP CPU", &experiment.backend_index, 1);
        ImGui::EndDisabled();
        if (!experiment_openmp_available) {
            ImGui::TextDisabled(
                "OpenMP is unavailable in this build. Select a supported backend explicitly.");
        }
        if (nbody::experiments::CounterfactualExperimentRunner::cuda_compiled()) {
            if (!control_layout.stacked) {
                ImGui::SameLine();
            }
            const bool experiment_cuda_available = app.cuda_device_available;
            ImGui::BeginDisabled(!experiment_cuda_available);
            ImGui::RadioButton("CUDA GPU", &experiment.backend_index, 2);
            ImGui::EndDisabled();
            if (!experiment_cuda_available) {
                ImGui::TextDisabled(
                    "CUDA support is compiled, but the selected NVIDIA device is unavailable; CUDA experiments cannot start.");
            }
        }

        ImGui::TextUnformatted("Experiment force solver");
        ImGui::RadioButton("Direct O(N^2)", &experiment.solver_index, 0);
        if (!control_layout.stacked) {
            ImGui::SameLine();
        }
        ImGui::BeginDisabled(experiment.backend_index == 2);
        ImGui::RadioButton("Barnes-Hut (approximate)",
            &experiment.solver_index, 1);
        ImGui::EndDisabled();
        if (experiment.backend_index == 2) ImGui::TextWrapped("CUDA counterfactuals support Direct only. Select OpenMP or Sequential explicitly for Barnes-Hut.");
        if (experiment.solver_index == 1) {
            ImGui::TextWrapped(
                "Barnes-Hut approximates distant groups. Measured separation can contain both the removal response and solver approximation error; compare against Direct before scientific interpretation.");
        }

        const bool control_table = !control_layout.stacked && ImGui::BeginTable(
            "counterfactual_inputs", 2,
            ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings);
        if (control_table) {
            ImGui::TableSetupColumn("Parameter", ImGuiTableColumnFlags_WidthFixed,
                control_layout.label_width);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        }
        const auto begin_input = [&](const char* label) {
            if (control_table) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextWrapped("%s", label);
                ImGui::TableSetColumnIndex(1);
            } else {
                ImGui::TextWrapped("%s", label);
            }
            ImGui::SetNextItemWidth(-1.0F);
        };
        begin_input("Observation horizon (s)");
        ImGui::InputDouble("##counterfactual_horizon",
            &experiment.observation_horizon_s, 0.0, 0.0, "%.0f");
        tooltip_for_last_item(
            "Total simulated duration. The catalog's one-day ceiling is an operational policy, not an accuracy guarantee.");
        begin_input("Retained-sample cadence (s)");
        ImGui::InputDouble("##counterfactual_cadence",
            &experiment.sample_cadence_s, 0.0, 0.0, "%.0f");
        tooltip_for_last_item(
            "Time between committed comparison samples. No intermediate trajectory points are fabricated.");
        begin_input("Absolute separation threshold (m)");
        ImGui::InputDouble("##counterfactual_absolute_threshold",
            &experiment.absolute_threshold_m, 0.0, 0.0, "%.6g");
        tooltip_for_last_item(
            "A survivor must exceed this absolute baseline/counterfactual separation and the normalized threshold.");
        begin_input("Normalized-deviation threshold");
        ImGui::InputDouble("##counterfactual_normalized_threshold",
            &experiment.normalized_threshold, 0.0, 0.0, "%.6g");
        tooltip_for_last_item(
            "Dimensionless separation divided by the body's fixed launch-time characteristic scale.");
        begin_input("Persistence (retained samples)");
        ImGui::InputInt("##counterfactual_persistence", &experiment.persistence_samples);
        tooltip_for_last_item(
            "Both thresholds must remain satisfied for this many consecutive retained samples.");
        if (experiment.solver_index == 1) {
            begin_input("Barnes-Hut opening angle theta");
            ImGui::InputDouble("##counterfactual_barnes_hut_theta",
                &experiment.barnes_hut_theta, 0.0, 0.0, "%.6g");
            tooltip_for_last_item(
                "Accuracy/performance policy captured immutably for both paired branches. Smaller theta generally evaluates more of the tree.");
        }
        if (control_table) {
            ImGui::EndTable();
        }
        experiment.persistence_samples = std::max(1, experiment.persistence_samples);
        ImGui::TextDisabled(
            "Significant = both thresholds met for the configured persistence; large separation alone is not escape.");

        if (ImGui::CollapsingHeader("Monitored survivor bodies")) {
            ImGui::BeginChild("experiment_monitored", {0.0F, 170.0F}, true);
            for (std::size_t index = 0; index < app.scenario.stable_ids.size(); ++index) {
                const bool removed = app.scenario.stable_ids[index] ==
                    experiment.removed_body_stable_id;
                ImGui::BeginDisabled(removed);
                bool monitored = index < experiment.monitored.size() &&
                    experiment.monitored[index] && !removed;
                const std::string label = app.controller.state().bodies[index].name +
                    "##experiment_monitor_" + app.scenario.stable_ids[index];
                if (ImGui::Checkbox(label.c_str(), &monitored) &&
                    index < experiment.monitored.size()) {
                    experiment.monitored[index] = monitored;
                }
                ImGui::EndDisabled();
            }
            ImGui::EndChild();
        }

        std::optional<nbody::visualization::CounterfactualResourceEstimate> estimate;
        std::string estimate_error;
        try {
            auto snapshot = make_ui_experiment_snapshot(app);
            const auto configuration = make_experiment_configuration(
                app, preferred_openmp_threads);
            if (configuration.monitored_body_stable_ids.empty()) {
                estimate_error = "Select at least one surviving monitored body.";
            } else if (!(configuration.significance.absolute_separation_threshold_m > 0.0) ||
                !std::isfinite(configuration.significance.absolute_separation_threshold_m)) {
                estimate_error = "Absolute threshold must be finite and positive.";
            } else if (configuration.significance.normalized_deviation_threshold < 0.0 ||
                !std::isfinite(configuration.significance.normalized_deviation_threshold)) {
                estimate_error = "Normalized threshold must be finite and non-negative.";
            } else if (configuration.solver ==
                    nbody::experiments::ExperimentSolver::barnes_hut &&
                (!(configuration.barnes_hut.theta > 0.0) ||
                    configuration.barnes_hut.theta > 2.0 ||
                    !std::isfinite(configuration.barnes_hut.theta))) {
                estimate_error =
                    "Barnes-Hut theta must be finite and in (0, 2].";
            } else if (configuration.backend ==
                    nbody::experiments::ExperimentBackend::cuda &&
                configuration.solver ==
                    nbody::experiments::ExperimentSolver::barnes_hut) {
                estimate_error =
                    "CUDA Barnes-Hut counterfactual execution is not implemented; choose Direct or a CPU backend.";
            } else if (configuration.backend ==
                    nbody::experiments::ExperimentBackend::cuda &&
                !app.cuda_device_available) {
                estimate_error =
                    "The selected CUDA device is unavailable for counterfactual execution.";
            } else {
                estimate = nbody::visualization::estimate_counterfactual_resources(
                    snapshot, configuration);
            }
            if (estimate_error.empty() && configuration.limits.observation_horizon_s >
                snapshot.maximum_supported_horizon_s()) {
                estimate_error = "Observation horizon exceeds this snapshot's supported policy limit.";
            } else if (estimate && estimate->retained_samples >
                configuration.limits.maximum_stored_samples) {
                estimate_error = "Sampling request exceeds the 4096-sample history limit.";
            } else if (estimate && estimate->direct_force_interactions >
                configuration.limits.workload_budget_interactions) {
                estimate_error = "Direct-equivalent admission estimate exceeds the worker workload budget.";
            } else if (estimate && estimate->conservative_retained_bytes >
                configuration.limits.memory_budget_bytes) {
                estimate_error = "Conservative retained-history estimate exceeds 64 MiB.";
            }
        } catch (const std::exception& error) {
            estimate_error = error.what();
        }
        if (estimate) {
            ImGui::TextWrapped(
                "Estimate: %zu steps, %zu retained samples, %zu monitored bodies, "
                "%zu hidden frame references, "
                "%llu direct-equivalent admission interactions, %s conservative retained storage.",
                estimate->integration_steps, estimate->retained_samples,
                estimate->monitored_bodies,
                estimate->retained_reference_bodies,
                static_cast<unsigned long long>(estimate->direct_force_interactions),
                human_bytes(estimate->conservative_retained_bytes).c_str());
        }
        if (!estimate_error.empty()) {
            ImGui::TextColored(ImVec4(0.96F, 0.47F, 0.31F, 1.0F), "%s",
                estimate_error.c_str());
        }

        const bool busy = app.experiment_worker.busy();
        const bool start_allowed = !busy && !candidates.empty() &&
            estimate_error.empty();
        ImGui::BeginDisabled(!start_allowed);
        if (ImGui::Button("Start counterfactual", {178.0F, 0.0F})) {
            start_counterfactual(app, preferred_openmp_threads);
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!busy);
        if (ImGui::Button("Cancel")) {
            app.experiment_worker.request_cancel();
        }
        ImGui::EndDisabled();
        if (!start_allowed) {
            if (busy) {
                ImGui::TextDisabled(
                    "Start is disabled while the current paired run is active; cancel or wait for a synchronized boundary.");
            } else if (candidates.empty()) {
                ImGui::TextDisabled(
                    "Start is disabled because this scenario has no approved removal candidate.");
            } else if (!estimate_error.empty()) {
                ImGui::TextDisabled(
                    "Start is disabled until the configuration error above is corrected.");
            }
        }
        if (!experiment.error.empty()) {
            ImGui::TextColored(ImVec4(0.96F, 0.47F, 0.31F, 1.0F), "%s",
                experiment.error.c_str());
        }

        const auto& publication = app.experiment_publication;
        const auto raw_worker_publication = app.experiment_worker.publication();
        const bool stale_worker_publication = raw_worker_publication.run &&
            !app.experiment_gate.accepts(raw_worker_publication);
        const bool publication_is_current = publication_matches_scenario(app);
        const bool presentation_eligible = !stale_worker_publication &&
            (!publication.result || publication_is_current);
        const auto barnes_hut_notice =
            nbody::visualization::counterfactual_barnes_hut_approximation_notice(
                publication, app.experiment_gate, publication_is_current);
        if (stale_worker_publication) {
            ImGui::TextColored(ImVec4(0.96F, 0.67F, 0.28F, 1.0F),
                "Status: Stale result suppressed");
            ImGui::TextWrapped(
                "A prior-scenario publication is ineligible for progress, metrics, warnings, and overlays. Start a new run for the active scenario.");
        }
        if (!stale_worker_publication && publication.total_steps > 0U) {
            const float progress = static_cast<float>(publication.completed_steps) /
                static_cast<float>(publication.total_steps);
            ImGui::ProgressBar(std::clamp(progress, 0.0F, 1.0F), {-1.0F, 0.0F});
            ImGui::TextDisabled("%zu / %zu paired steps published",
                publication.completed_steps, publication.total_steps);
        }
        if (presentation_eligible && !publication.worker_error.empty()) {
            ImGui::TextColored(ImVec4(1.0F, 0.32F, 0.28F, 1.0F),
                "Worker failure: %s", publication.worker_error.c_str());
        }
        if (!barnes_hut_notice.empty()) {
            ImGui::TextColored(ImVec4(0.96F, 0.67F, 0.28F, 1.0F),
                "Executed-solver notice: %.*s",
                static_cast<int>(barnes_hut_notice.size()),
                barnes_hut_notice.data());
        }
        if (publication.result && !publication_is_current) {
            ImGui::TextColored(ImVec4(0.96F, 0.67F, 0.28F, 1.0F),
                "Completed result belongs to a different physical composition; all result presentation is suppressed.");
        }
        if (presentation_eligible && publication.result &&
            publication.result->terminal_failure) {
            const auto& failure = *publication.result->terminal_failure;
            ImGui::TextColored(ImVec4(1.0F, 0.32F, 0.28F, 1.0F),
                "Terminal failure [%s; %s; %s] at t=%.6e s: %.*s%s",
                nbody::visualization::counterfactual_issue_kind_name(
                    failure.kind),
                nbody::visualization::counterfactual_issue_branch_name(
                    failure.branch),
                nbody::experiments::experiment_failure_phase_name(
                    failure.phase),
                failure.time_s,
                static_cast<int>(failure.message_text().size()),
                failure.message_text().data(),
                failure.message_truncated ? " [message truncated]" : "");
        }
        if (!experiment.presentation.error.empty()) {
            ImGui::TextColored(ImVec4(1.0F, 0.32F, 0.28F, 1.0F),
                "Result presentation error: %s",
                experiment.presentation.error.c_str());
        }

        ImGui::SeparatorText("Results");
        std::optional<std::string> requested_origin;
        if (app.trail_reference_body &&
            *app.trail_reference_body < app.scenario.stable_ids.size()) {
            requested_origin = app.scenario.stable_ids[*app.trail_reference_body];
        }
        const auto frame_status = presentation_eligible
            ? nbody::visualization::counterfactual_frame_status(
                publication, requested_origin)
            : nbody::visualization::CounterfactualFrameStatus{
                false, "Unavailable", "Stale or mismatched results are suppressed."};
        if (ImGui::CollapsingHeader(
                "Summary", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (begin_property_table("counterfactual_summary")) {
                property_row("Experiment status",
                    stale_worker_publication ? "Stale (suppressed)" :
                    nbody::visualization::counterfactual_worker_state_name(
                        publication.state));
                if (presentation_eligible && publication.run) {
                    const auto& run = *publication.run;
                    const auto removed_index = nbody::visualization::find_body_index(
                        app.scenario, run.removed_body_stable_id);
                    property_row("Removed body", removed_index
                        ? app.controller.state().bodies[*removed_index].name +
                            " (" + run.removed_body_stable_id + ")"
                        : run.removed_body_stable_id);
                    std::string backend =
                        nbody::experiments::experiment_backend_name(run.backend);
                    if (run.backend ==
                        nbody::experiments::ExperimentBackend::openmp) {
                        backend += " (" + std::to_string(run.openmp_threads) +
                            " threads)";
                    } else if (run.backend ==
                        nbody::experiments::ExperimentBackend::cuda) {
                        backend += " (device " + std::to_string(run.cuda_device) + ")";
                    }
                    property_row("Executed backend", backend);
                    std::string solver =
                        nbody::experiments::experiment_solver_name(run.solver);
                    if (run.solver ==
                        nbody::experiments::ExperimentSolver::barnes_hut) {
                        solver += " (theta=" +
                            nbody::visualization::format_measurement(
                                run.barnes_hut.theta, "", 6) + ")";
                    }
                    property_row("Executed solver", solver);
                    property_row("Observation horizon",
                        human_time(run.observation_horizon_s));
                    property_row("Completed paired steps",
                        std::to_string(publication.completed_steps) + " / " +
                        std::to_string(publication.total_steps));
                    property_row("Monitored bodies",
                        std::to_string(run.monitored_body_stable_ids.size()));
                    const auto significant_count = publication.result
                        ? static_cast<std::size_t>(std::count_if(
                            publication.result->summaries.begin(),
                            publication.result->summaries.end(),
                            [](const auto& row) { return row.significant; }))
                        : 0U;
                    property_row("Significant bodies",
                        publication.result ? std::to_string(significant_count) : "N/A");
                    property_row("Active comparison frame",
                        frame_status.coordinate_frame_label);
                    property_row("Overlay availability",
                        frame_status.available
                            ? (experiment.show_overlay
                                ? "Frame available; enabled (visibility filters apply)"
                                : "Frame available; display disabled")
                            : "Unavailable");
                    if (publication.result) {
                        property_row("Scientific warnings",
                            std::to_string(publication.result->issues.size()) +
                            " retained; " +
                            std::to_string(publication.result->dropped_issue_count) +
                            " dropped by bounds");
                    }
                } else {
                    property_row("Executed run", "No experiment submitted");
                }
                ImGui::EndTable();
            }
            if (publication.result && publication_is_current) {
                ImGui::Checkbox(
                    "Show trajectory comparison", &experiment.show_overlay);
                ImGui::SameLine();
                ImGui::Checkbox(
                    "Include significant bodies", &experiment.show_significant);
                draw_counterfactual_legend();
                if (frame_status.available) {
                    ImGui::TextDisabled("%s", frame_status.explanation.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.96F, 0.67F, 0.28F, 1.0F),
                        "%s", frame_status.explanation.c_str());
                }
            }
        }
        if (publication.result && publication_is_current) {
            const auto& result = *publication.result;
            if (ImGui::CollapsingHeader(
                    "Per-body comparison", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextDisabled(
                    "N/A means the committed result does not define that metric. Click a column heading to sort.");
                draw_counterfactual_result_table(
                    experiment.presentation.result_rows);
            }
            if (ImGui::CollapsingHeader("Detailed diagnostics")) {
                ImGui::TextWrapped(
                    "Conservation drift is evaluated independently against each branch's own initial state. Cross-branch energy differences are not integration error.");
                draw_counterfactual_conservation(
                    "Baseline branch", result.baseline_conservation);
                draw_counterfactual_conservation(
                    "Counterfactual branch", result.counterfactual_conservation);
                if (result.solver ==
                        nbody::experiments::ExperimentSolver::barnes_hut &&
                    result.baseline_barnes_hut_statistics &&
                    result.counterfactual_barnes_hut_statistics) {
                    const auto& baseline =
                        *result.baseline_barnes_hut_statistics;
                    const auto& counterfactual =
                        *result.counterfactual_barnes_hut_statistics;
                    ImGui::SeparatorText("Last committed Barnes-Hut force evaluations");
                    ImGui::TextWrapped(
                        "Baseline: %zu bodies, %zu nodes, %zu visited nodes, %zu approximations, %zu direct interactions.",
                        baseline.body_count, baseline.node_count,
                        baseline.visited_nodes, baseline.approximated_nodes,
                        baseline.direct_body_interactions);
                    ImGui::TextWrapped(
                        "Counterfactual: %zu bodies, %zu nodes, %zu visited nodes, %zu approximations, %zu direct interactions.",
                        counterfactual.body_count, counterfactual.node_count,
                        counterfactual.visited_nodes,
                        counterfactual.approximated_nodes,
                        counterfactual.direct_body_interactions);
                }
                ImGui::SeparatorText("Scientific warnings");
                if (result.issues.empty() && result.dropped_issue_count == 0U) {
                    ImGui::TextDisabled("No retained scientific warnings.");
                } else {
                    draw_counterfactual_warning_groups(
                        result, experiment.presentation.warning_groups);
                }
            }
        }
        if (presentation_eligible && publication.run && ImGui::CollapsingHeader(
                "Executed configuration & provenance")) {
            const auto& run = *publication.run;
            ImGui::TextDisabled(
                "Immutable values captured at launch; edits above configure only the next run.");
            if (begin_property_table("counterfactual_run_metadata")) {
                property_row("Run generations",
                    "scenario " + std::to_string(run.scenario_generation) +
                    ", worker " + std::to_string(run.worker_generation));
                property_row("Snapshot",
                    std::string(nbody::visualization::counterfactual_snapshot_source_name(
                        run.snapshot_source)) + " at t=" +
                    nbody::visualization::format_measurement(
                        run.snapshot_time_s, "s", 10));
                property_row("Scenario / dataset", run.scenario_id);
                property_row("Removed stable ID", run.removed_body_stable_id);
                property_row("Snapshot provenance", run.source_state_provenance);
                property_row("Coordinate contract", run.coordinate_frame);
                std::string executed_backend =
                    nbody::experiments::experiment_backend_name(run.backend);
                if (run.backend == nbody::experiments::ExperimentBackend::openmp) {
                    executed_backend += " (" + std::to_string(run.openmp_threads) +
                        " threads)";
                } else if (run.backend ==
                    nbody::experiments::ExperimentBackend::cuda) {
                    executed_backend += " (device " +
                        std::to_string(run.cuda_device) + ")";
                }
                property_row("Executed backend", executed_backend);
                property_row("Executed solver",
                    nbody::experiments::experiment_solver_name(run.solver));
                if (run.solver ==
                    nbody::experiments::ExperimentSolver::barnes_hut) {
                    property_row("Barnes-Hut theta",
                        nbody::visualization::format_measurement(
                            run.barnes_hut.theta, "dimensionless", 8));
                    property_row("Barnes-Hut limits",
                        "nodes " + std::to_string(run.barnes_hut.maximum_nodes) +
                        ", traversal work " +
                        std::to_string(run.barnes_hut.maximum_traversal_work) +
                        ", working memory " +
                        human_bytes(run.barnes_hut.memory_budget_bytes));
                }
                property_row("Physical timestep",
                    nbody::visualization::format_measurement(
                        run.physical_timestep_s, "s", 10));
                property_row("Horizon / steps",
                    nbody::visualization::format_measurement(
                        run.observation_horizon_s, "s", 10) + " / " +
                    std::to_string(run.accepted_step_count));
                property_row("Sample cadence",
                    nbody::visualization::format_measurement(
                        run.sample_cadence_s, "s", 10));
                property_row("Absolute threshold",
                    nbody::visualization::format_measurement(
                        run.absolute_separation_threshold_m, "m", 6));
                property_row("Normalized threshold",
                    nbody::visualization::format_measurement(
                        run.normalized_deviation_threshold, "dimensionless", 6));
                property_row("Persistence rule",
                    "both thresholds for " +
                    std::to_string(run.persistence_samples) +
                    " consecutive retained samples");
                property_row("Monitored / frame references",
                    std::to_string(run.monitored_body_stable_ids.size()) + " / " +
                    std::to_string(run.retained_reference_body_stable_ids.size()));
                property_row("Monitored stable IDs",
                    join_stable_ids(run.monitored_body_stable_ids));
                property_row("Retained frame-reference IDs",
                    join_stable_ids(run.retained_reference_body_stable_ids));
                property_row("Direct-equivalent admission interactions",
                    std::to_string(run.estimated_direct_force_interactions));
                property_row("Estimated retained storage",
                    human_bytes(run.estimated_retained_bytes));
                ImGui::EndTable();
            }
        }
        ImGui::TextWrapped(
            "This experiment is hypothetical. Large separation is not escape or ejection. "
            "The catalog timestep is provisional; close encounters remain point-mass and "
            "linear swept-step screened, not collision-resolved.");
    }

    }
    if (app.panel_page == 2) {
    section_heading("Visualization & View");
    if (!app.view_error.empty()) {
        ImGui::TextColored(ImVec4(0.96F, 0.67F, 0.28F, 1.0F), "%s", app.view_error.c_str());
    }
    if (ImGui::Button("Overview")) {
        actions.camera_preset = nbody::visualization::CameraPreset::overview;
    }
    help_for_last_item(nbody::visualization::HelpTopic::camera_presets);
    if (app.astronomical()) {
        ImGui::SameLine();
        if (ImGui::Button("Inner System")) {
            actions.camera_preset = nbody::visualization::CameraPreset::inner_system;
        }
        help_for_last_item(nbody::visualization::HelpTopic::camera_presets);
        ImGui::SameLine();
        if (ImGui::Button(
                app.scenario.kind == ScenarioKind::astronomical_catalog
                    ? "Earth-Moon" : "Earth")) {
            actions.camera_preset = nbody::visualization::CameraPreset::earth_moon;
        }
        help_for_last_item(nbody::visualization::HelpTopic::camera_presets);
        if (ImGui::Button("Jupiter")) {
            actions.camera_preset = nbody::visualization::CameraPreset::jupiter_system;
        }
        help_for_last_item(nbody::visualization::HelpTopic::camera_presets);
        ImGui::SameLine();
        if (ImGui::Button("Saturn")) {
            actions.camera_preset = nbody::visualization::CameraPreset::saturn_system;
        }
        help_for_last_item(nbody::visualization::HelpTopic::camera_presets);
        if (app.scenario.kind == ScenarioKind::astronomical_catalog) {
            if (ImGui::Button("Uranus")) {
                actions.camera_preset = nbody::visualization::CameraPreset::uranus_system;
            }
            ImGui::SameLine();
            if (ImGui::Button("Neptune-Triton")) {
                actions.camera_preset = nbody::visualization::CameraPreset::neptune_triton;
            }
            if (ImGui::Button("Pluto-Charon")) {
                actions.camera_preset = nbody::visualization::CameraPreset::pluto_charon;
            }
            ImGui::SameLine();
            if (ImGui::Button("Asteroid Belt")) {
                actions.camera_preset = nbody::visualization::CameraPreset::asteroid_belt;
            }
        }
    }
    ImGui::BeginDisabled(!app.trails);
    ImGui::Checkbox("Simulation history trails", &app.show_trails);
    help_for_last_item(nbody::visualization::HelpTopic::history_trails);
    ImGui::EndDisabled();
    if (app.astronomical()) {
        ImGui::Checkbox("SSB origin marker", &app.show_barycentric_motion);
        help_for_last_item(nbody::visualization::HelpTopic::barycenter);
    }
    if (app.scenario.kind == ScenarioKind::solar_system) {
        ImGui::Checkbox("Reference orbit guides", &app.show_reference_orbits);
        help_for_last_item(nbody::visualization::HelpTopic::reference_orbits);
        if (app.show_barycentric_motion) {
            ImGui::SetNextItemWidth(-1.0F);
            ImGui::SliderFloat(
                "Sun-motion presentation amplification", &app.sun_motion_amplification,
                1.0F, 100.0F, "%.0fx", ImGuiSliderFlags_Logarithmic);
            help_for_last_item(nbody::visualization::HelpTopic::sun_motion_amplification);
            ImGui::TextDisabled(
                "The trail uses integrated Sun positions; amplification affects only display.");
        }
    }
    if (app.scenario.kind == ScenarioKind::astronomical_catalog) {
        ImGui::Checkbox("Initial osculating orbit guides", &app.show_reference_orbits);
        ImGui::TextWrapped("Two-body presentation guides from the initial catalog state; not ephemeris predictions or integrated histories.");
    }
    ImGui::Checkbox("Body labels", &app.show_labels);
    help_for_last_item(nbody::visualization::HelpTopic::labels);
    float radius_multiplier = app.render_space.radius_multiplier();
    ImGui::SetNextItemWidth(-1.0F);
    if (ImGui::SliderFloat("Body-size emphasis", &radius_multiplier, 1.0F, 500.0F, "%.1fx",
                           ImGuiSliderFlags_Logarithmic)) {
        app.render_space.set_radius_multiplier(radius_multiplier);
    }
    help_for_last_item(nbody::visualization::HelpTopic::body_size);
    ImGui::TextWrapped(
        "Body sizes are camera-aware and presentation-only. Centers and orbital geometry remain physical.");

    if (ImGui::CollapsingHeader("Advanced / Technical Details")) {
        help_for_last_item(nbody::visualization::HelpTopic::advanced_details);
        ImGui::TextWrapped("%s", app.scenario.provenance.c_str());
        ImGui::TextWrapped("Execution: %s / %s point-mass Velocity Verlet",
                           app.controller.backend_name(), app.controller.solver_name());
        if (!app.scenario.scientific_limitations.empty()) {
            ImGui::TextWrapped("Scientific limitations: %s",
                               app.scenario.scientific_limitations.c_str());
        }
        ImGui::TextWrapped("Position: (%.6e, %.6e, %.6e) m",
                           body.position_m.x, body.position_m.y, body.position_m.z);
        ImGui::TextWrapped("Velocity: (%.6e, %.6e, %.6e) m/s",
                           body.velocity_m_s.x, body.velocity_m_s.y, body.velocity_m_s.z);
        if (app.conservation) {
            const auto& drift = app.conservation->drift();
            ImGui::TextWrapped("Maximum energy drift: %.6e", drift.maximum_energy_relative);
            ImGui::TextWrapped("Maximum angular-momentum drift: %.6e",
                               drift.maximum_angular_momentum_relative);
        }
        ImGui::TextWrapped("Fixed integration timestep: %.6e s", app.controller.dt_s());
        ImGui::TextWrapped("Physics steps per frame: %zu", app.controller.steps_per_frame());
        if (app.trails) {
            int trail_capacity = static_cast<int>(app.trails->capacity());
            ImGui::SetNextItemWidth(-1.0F);
            if (ImGui::SliderInt("Trail history points", &trail_capacity, 64, 2048)) {
                app.trails->set_capacity(static_cast<std::size_t>(trail_capacity));
            }
            help_for_last_item(nbody::visualization::HelpTopic::trail_history);
        }
        if (app.scenario.kind == ScenarioKind::synthetic) {
            ImGui::TextWrapped("Deterministic scene: N=%zu, %s, seed=%llu",
                               app.synthetic_body_count,
                               nbody::visualization::synthetic_distribution_name(
                                   app.synthetic_distribution),
                               static_cast<unsigned long long>(app.synthetic_seed));
        }
        const float fps = ImGui::GetIO().Framerate;
        ImGui::TextWrapped("Display: %.1f FPS, %.2f ms/frame",
                           fps, fps > 0.0F ? 1000.0F / fps : 0.0F);
        ImGui::TextColored(ImVec4(0.92F, 0.67F, 0.31F, 1.0F),
                           "Display diagnostic — not benchmark evidence");
    } else {
        help_for_last_item(nbody::visualization::HelpTopic::advanced_details);
    }

    }
    ImGui::EndChild();
    actions.panel_right = ImGui::GetWindowPos().x + ImGui::GetWindowSize().x;
    ImGui::End();
    return actions;
}

bool apply_counterfactual_display(
    ApplicationModel& app,
    std::vector<nbody::visualization::RenderBody>& snapshot,
    std::vector<nbody::visualization::RenderTrail>& render_trails) {
    if (app.experiment_ui.comparison_view == 0) return false;
    std::optional<std::string> origin;
    if (app.local_origin_body) origin = app.scenario.stable_ids[*app.local_origin_body];
    std::vector<nbody::visualization::RenderBody> endpoints;
    if (!nbody::visualization::build_counterfactual_endpoint_snapshot(
            app.experiment_publication, app.experiment_gate, app.scenario,
            app.render_space, app.experiment_ui.comparison_view == 2, origin, endpoints)) {
        return false;
    }
    for (std::size_t index = 0; index < endpoints.size(); ++index) {
        endpoints[index].visible = endpoints[index].visible && app.body_visible(index);
        if (app.experiment_ui.comparison_view == 2 && !endpoints[index].emissive) {
            endpoints[index].color = endpoints[index].color * 0.45F + glm::vec3(0.55F, 0.12F, 0.38F);
        }
    }
    snapshot = std::move(endpoints);
    // The result has its own matched-time history. Never overlay live history
    // or initial conic guides on a future paired endpoint as if it were live.
    render_trails.clear();
    append_counterfactual_trails(app, render_trails);
    return true;
}

void draw_labels(
    ApplicationModel& app,
    const std::vector<nbody::visualization::RenderBody>& bodies,
    const nbody::visualization::OrbitCamera& camera,
    float panel_right) {
    if (!app.show_labels) {
        app.label_layout.clear();
        return;
    }
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    auto* draw_list = ImGui::GetForegroundDrawList();
    std::vector<nbody::visualization::LabelCandidate> candidates;
    const auto append_candidate = [&](std::size_t index) {
        if (!bodies[index].visible) {
            return;
        }
        const auto screen = project_to_screen(bodies[index].position, camera, display, panel_right);
        if (!screen || screen->x < panel_right || screen->x > display.x ||
            screen->y < 0.0F || screen->y > display.y) {
            return;
        }
        const char* label = app.controller.state().bodies[index].name.c_str();
        const ImVec2 size = ImGui::CalcTextSize(label);
        const bool selected = index == app.selected_body;
        const int priority = app.astronomical()
            ? nbody::visualization::astronomical_label_priority(
                app.scenario.body_metadata[index].body_class, selected, index)
            : (selected ? 1000 : 500 - static_cast<int>(index));
        candidates.push_back({
            index,
            {screen->x, screen->y},
            {size.x + 10.0F, size.y + 6.0F},
            priority,
            selected,
        });
    };
    if (app.scenario.label_policy == nbody::visualization::LabelPolicy::selected_only) {
        if (app.selected_body < bodies.size()) {
            append_candidate(app.selected_body);
        }
    } else {
        for (std::size_t index = 0; index < bodies.size(); ++index) {
            append_candidate(index);
        }
    }
    if (app.scenario.kind == ScenarioKind::solar_system) {
        const auto inner = nbody::visualization::stable_inner_system_callouts(
            candidates, {display.x, display.y}, panel_right + 8.0F, 8.0F);
        std::vector<nbody::visualization::LabelCandidate> outer;
        std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(outer),
                     [](const auto& candidate) { return candidate.body_index > 4U; });
        app.label_layout = nbody::visualization::declutter_labels(
            std::move(outer), {display.x, display.y}, panel_right + 8.0F, 7.0F,
            app.label_layout, inner);
    } else {
        app.label_layout = nbody::visualization::declutter_labels(
            std::move(candidates), {display.x, display.y}, panel_right + 8.0F, 7.0F,
            app.label_layout);
    }
    for (const auto& placement : app.label_layout) {
        const char* label = app.controller.state().bodies[placement.body_index].name.c_str();
        const ImVec2 position{placement.position.x, placement.position.y};
        if (placement.leader_line) {
            draw_list->AddLine(
                {placement.anchor.x, placement.anchor.y},
                {position.x + 4.0F, position.y + placement.size.y * 0.5F},
                IM_COL32(150, 170, 190, 120), 1.0F);
        }
        draw_list->AddRectFilled(
            position,
            {position.x + placement.size.x, position.y + placement.size.y},
            placement.selected ? IM_COL32(44, 52, 62, 225) : IM_COL32(10, 16, 28, 185),
            4.0F);
        const ImU32 color = placement.selected
            ? IM_COL32(255, 230, 110, 255)
            : IM_COL32(225, 232, 245, 240);
        draw_list->AddText({position.x + 5.0F, position.y + 3.0F}, color, label);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        BackendKind backend = BackendKind::sequential;
        if (options.backend == "openmp") {
            backend = BackendKind::openmp;
        } else if (options.backend == "cuda") {
            if (options.solver == "barnes-hut") {
                throw std::runtime_error(
                    "CUDA Barnes-Hut viewer execution is unavailable pending NVIDIA validation; "
                    "choose --solver direct or a CPU backend.");
            }
            if (!nbody::visualization::SimulationController::cuda_compiled()) {
                throw std::runtime_error(
                    "CUDA visualization is unavailable in this build; configure with "
                    "-DNBODY_ENABLE_CUDA=ON or choose sequential/openmp.");
            }
            if (!nbody::visualization::SimulationController::cuda_available(
                    options.cuda_device)) {
                throw std::runtime_error(
                    "requested CUDA device is unavailable; choose an available --device index "
                    "or a CPU backend.");
            }
            backend = BackendKind::cuda;
        }
        ApplicationModel app(options, backend);
        nbody::visualization::OrbitCamera camera(35.0F, 28.0F, 22.0F);

        GlfwRuntime glfw_runtime;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);
        std::unique_ptr<GLFWwindow, WindowDeleter> window(glfwCreateWindow(
            options.width, options.height, "N-body Solar System Simulator", nullptr, nullptr));
        if (!window) {
            throw std::runtime_error(
                "OpenGL window/context creation failed; verify OpenGL 3.3 support and desktop access");
        }
        glfwMakeContextCurrent(window.get());
        glfwSwapInterval(1);
        const int loaded_version = gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress));
        if (loaded_version == 0) {
            throw std::runtime_error("OpenGL loader initialization failed after context creation");
        }

        InteractionState input{&camera, &app, options.openmp_threads};
        glfwSetWindowUserPointer(window.get(), &input);
        glfwSetKeyCallback(window.get(), key_callback);
        glfwSetMouseButtonCallback(window.get(), mouse_button_callback);
        glfwSetCursorPosCallback(window.get(), cursor_callback);
        glfwSetScrollCallback(window.get(), scroll_callback);
        ImGuiRuntime imgui(window.get());
        nbody::visualization::BodyRenderer renderer;

        const auto gl_string = [](unsigned int name) {
            const auto* value = glGetString(name);
            return value == nullptr ? "unavailable" : reinterpret_cast<const char*>(value);
        };
        std::cout << "scenario=" << app.scenario.name << '\n'
                  << "provenance=" << app.scenario.provenance << '\n'
                  << "fixed_dt_s=" << app.controller.dt_s() << '\n'
                  << "OpenGL vendor=" << gl_string(GL_VENDOR)
                  << " renderer=" << gl_string(GL_RENDERER)
                  << " version=" << gl_string(GL_VERSION) << '\n'
                  << "controls: ImGui panel; Space pause/resume, R reset, C overview, "
                     "left-click select, left-drag orbit, wheel zoom, Esc exit\n";
        log_backend(app.controller);

        std::vector<nbody::visualization::RenderBody> snapshot;
        std::vector<nbody::visualization::RenderTrail> render_trails;
        snapshot.reserve(app.controller.state().bodies.size());

        while (glfwWindowShouldClose(window.get()) == GLFW_FALSE) {
            glfwPollEvents();
            int framebuffer_width = 0;
            int framebuffer_height = 0;
            glfwGetFramebufferSize(window.get(), &framebuffer_width, &framebuffer_height);
            if (framebuffer_width <= 0 || framebuffer_height <= 0) {
                glfwWaitEventsTimeout(0.05);
                continue;
            }

            try {
                if (!app.controller.paused()) {
                    const auto policy = app.workload_policy();
                    if (app.scenario.kind == ScenarioKind::synthetic &&
                        !policy.continuous_advancement) {
                        app.controller.set_paused(true);
                        app.solver_error = policy.explanation;
                        throw std::runtime_error(policy.explanation);
                    }
                    const auto started = std::chrono::steady_clock::now();
                    const std::size_t steps = nbody::visualization::responsive_viewer_batch_steps(
                        app.controller.backend_kind() == BackendKind::cuda,
                        app.controller.steps_per_frame());
                    const std::uint64_t timing_generation =
                        app.simulation_timing.generation();
                    app.controller.advance_steps(steps);
                    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                    app.simulation_timing.publish(
                        timing_generation,
                        elapsed_ms,
                        elapsed_ms > 0.0
                            ? static_cast<double>(steps) * 1000.0 / elapsed_ms
                            : 0.0);
                }
                app.simulation_error.clear();
            } catch (const std::exception& error) {
                app.simulation_error = app.controller.last_error().empty()
                    ? error.what() : app.controller.last_error();
                std::cerr << "simulation advancement failed; state was preserved and paused: "
                          << app.simulation_error << '\n';
            }
            if (app.trails) {
                app.trails->sample(app.controller.state());
            }
            if (app.conservation) {
                app.conservation->update(app.controller.state());
            }
            refresh_render_origin(app);
            app.render_space.update_snapshot(app.controller.state(), snapshot);
            app.apply_visibility(snapshot);
            if (app.focused_body && *app.focused_body < snapshot.size()) {
                camera.approach_target(snapshot[*app.focused_body].position,
                    1.0F - std::exp(-9.0F * std::clamp(ImGui::GetIO().DeltaTime, 0.0F, 0.1F)));
            }
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            const UiActions actions = draw_control_panel(
                app, input.preferred_openmp_threads);
            bool state_replaced = false;
            if (actions.reset) {
                app.reset();
                state_replaced = true;
            }
            if (actions.advance_one_step) {
                try {
                    const auto policy = app.workload_policy();
                    if (app.scenario.kind == ScenarioKind::synthetic &&
                        !policy.manual_single_step) {
                        app.solver_error = policy.explanation;
                        throw std::runtime_error(policy.explanation);
                    }
                    const auto started = std::chrono::steady_clock::now();
                    const std::uint64_t timing_generation =
                        app.simulation_timing.generation();
                    app.controller.advance_steps(1);
                    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                    app.simulation_timing.publish(
                        timing_generation,
                        elapsed_ms,
                        elapsed_ms > 0.0 ? 1000.0 / elapsed_ms : 0.0);
                    app.simulation_error.clear();
                    if (app.trails) {
                        app.trails->sample(app.controller.state());
                    }
                    if (app.conservation) {
                        app.conservation->update(app.controller.state());
                    }
                    state_replaced = true;
                } catch (const std::exception& error) {
                    app.simulation_error = app.controller.last_error().empty()
                        ? error.what() : app.controller.last_error();
                    std::cerr << "manual advancement failed; state was preserved and paused: "
                              << app.simulation_error << '\n';
                }
            }
            try {
                if (actions.apply_synthetic_configuration) {
                    app.switch_scenario(
                        ScenarioKind::synthetic,
                        app.synthetic_body_count_draft,
                        app.synthetic_seed_draft,
                        app.synthetic_distribution_draft);
                    camera.reset();
                    state_replaced = true;
                } else if (actions.catalog_subset) {
                    app.switch_scenario(
                        ScenarioKind::astronomical_catalog,
                        app.synthetic_body_count,
                        app.synthetic_seed,
                        app.synthetic_distribution,
                        *actions.catalog_subset);
                    camera.reset();
                    state_replaced = true;
                } else if (actions.scenario) {
                    app.switch_scenario(
                        *actions.scenario,
                        app.synthetic_body_count_draft,
                        app.synthetic_seed_draft,
                        app.synthetic_distribution_draft);
                    camera.reset();
                    state_replaced = true;
                }
            } catch (const std::exception& error) {
                app.view_error = error.what();
                std::cerr << "scenario replacement rejected; active state was preserved: "
                          << app.view_error << '\n';
            }
            if (state_replaced) {
                refresh_render_origin(app);
                app.render_space.update_snapshot(app.controller.state(), snapshot);
                app.apply_visibility(snapshot);
            }
            if (actions.camera_preset) {
                try {
                    apply_view_preset(app, camera, *actions.camera_preset, snapshot);
                    app.view_error.clear();
                } catch (const std::invalid_argument& error) {
                    app.view_error = std::string("View unavailable: ") + error.what();
                    std::cerr << app.view_error << '\n';
                }
            }
            if (actions.frame_focus && app.selected_body < snapshot.size()) {
                focus_selected_body(app, camera, snapshot);
            }

            input.scene_left = actions.panel_right + 8.0F;
            const ImVec2 display = ImGui::GetIO().DisplaySize;
            const float logical_scene_width = std::max(1.0F, display.x - input.scene_left);
            camera.set_aspect_ratio(logical_scene_width / std::max(display.y, 1.0F));
            refresh_render_origin(app);
            app.render_space.update_snapshot(app.controller.state(), snapshot);
            app.apply_visibility(snapshot);
            if (!apply_counterfactual_display(app, snapshot, render_trails)) {
                update_render_trails(app, render_trails);
                append_counterfactual_trails(app, render_trails);
            }
            nbody::visualization::apply_screen_space_body_sizing(
                snapshot, camera, display.y,
                nbody::visualization::find_body_index(
                    app.scenario,
                    app.scenario.kind == ScenarioKind::astronomical_catalog
                        ? "sol:sun" : "sun"));

            if (input.pending_selection && !ImGui::GetIO().WantCaptureMouse) {
                select_nearest_body(
                    app, snapshot, camera, input.selection_x, input.selection_y, input.scene_left);
            }
            input.pending_selection = false;
            draw_labels(app, snapshot, camera, actions.panel_right);
            ImGui::Render();
            std::optional<nbody::visualization::RenderMarker> barycenter_marker;
            if (app.show_barycentric_motion &&
                app.astronomical()) {
                barycenter_marker = nbody::visualization::RenderMarker{
                    app.render_space.to_render_position({}),
                    {0.96F, 0.78F, 0.30F},
                    0.065F,
                };
            }
            const int scene_viewport_x = static_cast<int>(std::lround(
                input.scene_left / std::max(display.x, 1.0F) * framebuffer_width));
            renderer.render(
                snapshot, render_trails, camera, framebuffer_width, framebuffer_height,
                scene_viewport_x, framebuffer_width - scene_viewport_x,
                app.selected_body, barycenter_marker);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window.get());
        }
        std::cout << "live_display_fps_not_benchmark=" << ImGui::GetIO().Framerate
                  << " live_simulation_batch_ms_not_benchmark="
                  << (app.simulation_timing.sample()
                          ? app.simulation_timing.sample()->batch_milliseconds : 0.0)
                  << " live_simulation_steps_per_second_not_benchmark="
                  << (app.simulation_timing.sample()
                          ? app.simulation_timing.sample()->simulation_steps_per_second
                          : 0.0)
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visualization error: " << error.what() << '\n';
        return 1;
    }
}
