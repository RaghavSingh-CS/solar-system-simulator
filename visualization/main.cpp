#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/conservation.hpp"
#include "nbody/visualization/compute_comparison.hpp"
#include "nbody/visualization/controller.hpp"
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
#include <utility>
#include <vector>

namespace {

using nbody::visualization::BackendKind;
using nbody::visualization::ScenarioKind;

struct Options {
    ScenarioKind scenario{ScenarioKind::solar_system};
    std::string backend{"sequential"};
    std::size_t bodies{32};
    std::uint64_t seed{24301};
    int openmp_threads{4};
    int cuda_device{0};
    int width{1440};
    int height{900};
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "  --scenario solar|catalog|synthetic Default: solar\n"
              << "  --backend sequential|openmp|cuda  CUDA is accepted only in CUDA-enabled viewers\n"
              << "  --threads N                        OpenMP threads (default 4)\n"
              << "  --device N                         CUDA device index (default 0)\n"
              << "  --bodies N                         Synthetic body count (default 32)\n"
              << "  --seed N                           Synthetic seed (default 24301)\n"
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
        } else if (argument == "--threads") {
            options.openmp_threads = std::stoi(value);
        } else if (argument == "--device") {
            options.cuda_device = std::stoi(value);
        } else if (argument == "--bodies") {
            options.bodies = std::stoull(value);
        } else if (argument == "--seed") {
            options.seed = std::stoull(value);
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
    if (options.bodies < 2 || options.openmp_threads <= 0 ||
        options.cuda_device < 0 || options.width <= 0 || options.height <= 0) {
        throw std::invalid_argument(
            "bodies >= 2, positive threads, non-negative device, and positive window dimensions are required");
    }
    if (options.scenario == ScenarioKind::synthetic &&
        !nbody::visualization::is_supported_compute_body_count(options.bodies)) {
        throw std::invalid_argument(
            "synthetic body count must be one of 32,64,128,256,512,1024,2048,4096,"
            "8192,16384,32768,65536,100000");
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
    nbody::visualization::ScenarioDefinition scenario;
    nbody::visualization::SimulationController controller;
    nbody::visualization::RenderSpaceTransform render_space;
    std::unique_ptr<nbody::visualization::ConservationTracker> conservation;
    std::unique_ptr<nbody::visualization::TrailHistory> trails;
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
    std::size_t synthetic_body_count{32};
    std::uint64_t synthetic_seed{24301};
    int preferred_cuda_device{0};
    bool cuda_device_available{false};
    bool show_trails{true};
    bool show_labels{true};
    bool show_reference_orbits{true};
    bool show_barycentric_motion{true};
    float sun_motion_amplification{10.0F};
    float panel_width{438.0F};

    ApplicationModel(const Options& options, BackendKind backend)
        : scenario(nbody::visualization::make_visual_scenario(
              options.scenario, options.bodies, options.seed)),
          controller(
              scenario.initial_state,
              {scenario.dt_s,
               scenario.speed_presets[scenario.default_speed_index].steps_per_frame,
               backend,
               options.openmp_threads,
               options.cuda_device}),
          render_space(
              scenario.initial_state,
              scenario.styles,
              9.0F,
              scenario.default_radius_multiplier),
          speed_index(scenario.default_speed_index),
          selected_body(0U),
          synthetic_body_count(options.bodies),
          synthetic_seed(options.seed),
          preferred_cuda_device(options.cuda_device),
          cuda_device_available(
              nbody::visualization::SimulationController::cuda_available(options.cuda_device)) {
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
        rebuild_optional_analysis();
        apply_workload_safety();
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
        return nbody::visualization::interactive_workload_policy(
            synthetic_body_count, backend);
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
        const std::vector<std::string>& catalog_subset = {}) {
        auto next = kind == ScenarioKind::astronomical_catalog
            ? nbody::visualization::make_astronomical_catalog_scenario(catalog_subset)
            : nbody::visualization::make_visual_scenario(kind, bodies, seed);
        const std::size_t next_speed = next.default_speed_index;
        controller.replace_scenario(
            next.initial_state, next.dt_s, next.speed_presets[next_speed].steps_per_frame);
        scenario = std::move(next);
        render_space = nbody::visualization::RenderSpaceTransform(
            scenario.initial_state, scenario.styles, 9.0F, scenario.default_radius_multiplier);
        speed_index = next_speed;
        selected_body = 0U;
        if (const auto earth = nbody::visualization::find_body_index(
                scenario, scenario.kind == ScenarioKind::astronomical_catalog
                    ? "sol:earth" : "earth")) {
            selected_body = *earth;
        }
        profile_visibility.assign(scenario.initial_state.bodies.size(), true);
        individual_visibility.assign(scenario.initial_state.bodies.size(), true);
        class_visibility.fill(true);
        if (scenario.kind == ScenarioKind::synthetic) {
            synthetic_body_count = bodies;
            synthetic_seed = seed;
        } else if (scenario.kind == ScenarioKind::astronomical_catalog) {
            for (std::size_t index = 0; index < catalog_roster_ids.size(); ++index) {
                catalog_inclusion_draft[index] =
                    nbody::visualization::find_body_index(
                        scenario, catalog_roster_ids[index]).has_value();
            }
        }
        show_labels = true;
        show_reference_orbits = scenario.kind == ScenarioKind::solar_system;
        show_trails = scenario.kind != ScenarioKind::synthetic || bodies <= 1024;
        rebuild_optional_analysis();
        apply_workload_safety();
        label_layout.clear();
        live_comparison.reset();
        live_comparison_error.clear();
        view_error.clear();
        focused_body.reset();
        local_origin_body.reset();
        trail_reference_body.reset();
    }

    bool astronomical() const noexcept {
        return scenario.kind != ScenarioKind::synthetic;
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
        controller.reset();
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
    std::cout << "active_backend=" << controller.backend_name();
    if (controller.backend_kind() == BackendKind::openmp) {
        std::cout << " threads=" << controller.openmp_threads();
    } else if (controller.backend_kind() == BackendKind::cuda) {
        std::cout << " device=" << controller.cuda_device();
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
            app.controller.toggle_paused();
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
            app.controller.select_backend(BackendKind::sequential);
            log_backend(app.controller);
        } else if (key == GLFW_KEY_2 && initial_press) {
            app.controller.select_backend(BackendKind::openmp, input->preferred_openmp_threads);
            log_backend(app.controller);
        } else if (key == GLFW_KEY_3 && initial_press) {
            app.controller.select_backend(BackendKind::cuda, 1, app.preferred_cuda_device);
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
        std::cerr << "input error: " << error.what() << '\n';
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }
    auto* input = interaction(window);
    if (input == nullptr || wants_mouse()) {
        return;
    }
    if (action == GLFW_PRESS) {
        input->dragging = true;
        input->drag_distance = 0.0;
        glfwGetCursorPos(window, &input->last_x, &input->last_y);
    } else if (action == GLFW_RELEASE && input->dragging) {
        input->dragging = false;
        if (input->drag_distance < 5.0) {
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
    input->camera->orbit(static_cast<float>(delta_x), static_cast<float>(delta_y));
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
            trail.color = app.scenario.styles[index].color * 0.76F;
            trail.alpha = 0.68F;
            trail.fade_with_age = true;
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
            guide.color = app.scenario.styles[index].color * 0.48F;
            guide.alpha = 0.24F;
            guide.fade_with_age = false;
            guide.points.clear();
            if (!app.body_visible(index)) {
                continue;
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

struct UiActions {
    std::optional<ScenarioKind> scenario;
    std::optional<std::size_t> synthetic_body_count;
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
    ImGui::TextDisabled("%s", label);
    ImGui::TableSetColumnIndex(1);
    ImGui::TextWrapped("%s", value.c_str());
}

bool begin_property_table(const char* id) {
    if (!ImGui::BeginTable(
            id, 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        return false;
    }
    ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 150.0F);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
    return true;
}

UiActions draw_control_panel(
    ApplicationModel& app,
    int& preferred_openmp_threads) {
    UiActions actions;
    const auto width_policy = nbody::visualization::panel_width_policy(
        ImGui::GetIO().DisplaySize.x, app.panel_width);
    ImGui::SetNextWindowPos({16.0F, 16.0F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({width_policy.width, 920.0F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        {width_policy.minimum, 480.0F},
        {width_policy.maximum, std::max(480.0F, ImGui::GetIO().DisplaySize.y - 24.0F)});
    ImGui::Begin("Scientific Visualization", nullptr, ImGuiWindowFlags_NoCollapse);
    app.panel_width = ImGui::GetWindowSize().x;

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.96F, 0.79F, 0.35F, 1.0F));
    ImGui::SetWindowFontScale(1.20F);
    ImGui::TextUnformatted("N-body Dynamics");
    ImGui::SetWindowFontScale(1.0F);
    ImGui::PopStyleColor();
    ImGui::TextWrapped("Scientific simulation and parallel-computing explorer");
    ImGui::TextDisabled("Drag the right edge to resize this panel");

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
            "Deterministic direct N-body workload: N=%zu, seed=%llu",
            app.synthetic_body_count,
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
        property_row("Simulated time", human_time(app.controller.state().time_s));
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
        app.controller.set_steps_per_frame(
            app.scenario.speed_presets[app.speed_index].steps_per_frame);
    }
    help_for_last_item(nbody::visualization::HelpTopic::simulation_speed);
    ImGui::TextWrapped(
        "This changes fixed-step batching per frame, not the numerical integration timestep.");

    section_heading("Compute Backend");
    if (ImGui::RadioButton("Sequential", app.controller.backend_kind() == BackendKind::sequential)) {
        app.controller.select_backend(BackendKind::sequential);
        app.live_comparison.reset();
        app.apply_workload_safety();
    }
    help_for_last_item(nbody::visualization::HelpTopic::sequential_backend);
    const bool openmp_available = nbody::visualization::SimulationController::openmp_available();
    ImGui::BeginDisabled(!openmp_available);
    if (ImGui::RadioButton("OpenMP", app.controller.backend_kind() == BackendKind::openmp)) {
        app.controller.select_backend(BackendKind::openmp, preferred_openmp_threads);
        app.live_comparison.reset();
        app.apply_workload_safety();
    }
    help_for_last_item(nbody::visualization::HelpTopic::openmp_backend);
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!app.cuda_device_available);
    if (ImGui::RadioButton("CUDA", app.controller.backend_kind() == BackendKind::cuda)) {
        app.controller.select_backend(BackendKind::cuda, 1, app.preferred_cuda_device);
        app.live_comparison.reset();
        app.apply_workload_safety();
    }
    help_for_last_item(nbody::visualization::HelpTopic::cuda_backend);
    ImGui::EndDisabled();
    if (!app.cuda_device_available) {
        ImGui::TextDisabled(
            nbody::visualization::SimulationController::cuda_compiled()
                ? "CUDA device unavailable"
                : "CUDA support not compiled");
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
                app.controller.select_backend(BackendKind::openmp, preferred_openmp_threads);
                app.apply_workload_safety();
            }
            app.live_comparison.reset();
            app.live_comparison_error.clear();
        }
        help_for_last_item(nbody::visualization::HelpTopic::openmp_threads);
    }
    if (app.astronomical()) {
        ImGui::TextWrapped(
            "This astronomical scenario uses direct O(N^2) Velocity Verlet. Backend speedup is not "
            "research evidence at this small body count.");
    } else {
        const std::string current_count = std::to_string(app.synthetic_body_count);
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::BeginCombo("Workload size", current_count.c_str())) {
            for (const std::size_t count : nbody::visualization::compute_body_counts) {
                const std::string label = std::to_string(count);
                if (ImGui::Selectable(label.c_str(), app.synthetic_body_count == count)) {
                    actions.synthetic_body_count = count;
                }
            }
            ImGui::EndCombo();
        }
        help_for_last_item(nbody::visualization::HelpTopic::workload_size);
        if (workload_policy.warning_required) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.96F, 0.72F, 0.32F, 1.0F));
            ImGui::TextWrapped(
                "Direct O(N^2) workload: continuous advancement is paused to keep the viewer responsive.");
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
    const auto& body = app.controller.state().bodies[app.selected_body];
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
        ImGui::TextWrapped("Solver: direct O(N^2) point-mass Velocity Verlet (%s)",
                           app.controller.backend_name());
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
            ImGui::TextWrapped("Deterministic workload: N=%zu, seed=%llu",
                               app.synthetic_body_count,
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

    actions.panel_right = ImGui::GetWindowPos().x + ImGui::GetWindowSize().x;
    ImGui::End();
    return actions;
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
        const bool star = app.scenario.body_metadata[index].body_class ==
            nbody::astronomy::AstronomicalBodyClass::star;
        const int priority = selected ? 1000 : (star ? 900 : 700 - static_cast<int>(index));
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

            app.controller.advance_frame();
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
                camera.approach_target(snapshot[*app.focused_body].position, 0.14F);
            }
            update_render_trails(app, render_trails);

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            if (input.pending_selection && !ImGui::GetIO().WantCaptureMouse) {
                select_nearest_body(
                    app, snapshot, camera, input.selection_x, input.selection_y, input.scene_left);
            }
            input.pending_selection = false;

            const UiActions actions = draw_control_panel(
                app, input.preferred_openmp_threads);
            bool state_replaced = false;
            if (actions.reset) {
                app.reset();
                state_replaced = true;
            }
            if (actions.advance_one_step) {
                app.controller.advance_steps(1);
                if (app.trails) {
                    app.trails->sample(app.controller.state());
                }
                if (app.conservation) {
                    app.conservation->update(app.controller.state());
                }
                state_replaced = true;
            }
            if (actions.synthetic_body_count) {
                app.switch_scenario(
                    ScenarioKind::synthetic, *actions.synthetic_body_count, app.synthetic_seed);
                camera.reset();
                state_replaced = true;
            } else if (actions.catalog_subset) {
                app.switch_scenario(
                    ScenarioKind::astronomical_catalog,
                    app.synthetic_body_count,
                    app.synthetic_seed,
                    *actions.catalog_subset);
                camera.reset();
                state_replaced = true;
            } else if (actions.scenario) {
                app.switch_scenario(
                    *actions.scenario, app.synthetic_body_count, app.synthetic_seed);
                camera.reset();
                state_replaced = true;
            }
            if (state_replaced) {
                refresh_render_origin(app);
                app.render_space.update_snapshot(app.controller.state(), snapshot);
                app.apply_visibility(snapshot);
                update_render_trails(app, render_trails);
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
            nbody::visualization::apply_screen_space_body_sizing(
                snapshot, camera, display.y,
                nbody::visualization::find_body_index(
                    app.scenario,
                    app.scenario.kind == ScenarioKind::astronomical_catalog
                        ? "sol:sun" : "sun"));

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
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visualization error: " << error.what() << '\n';
        return 1;
    }
}
