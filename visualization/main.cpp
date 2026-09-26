#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/conservation.hpp"
#include "nbody/visualization/controller.hpp"
#include "nbody/visualization/presentation.hpp"
#include "nbody/visualization/render_space.hpp"
#include "nbody/visualization/scenario.hpp"
#include "nbody/visualization/trails.hpp"
#include "renderer.hpp"

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/vec4.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
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
    int width{1440};
    int height{900};
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "  --scenario solar|synthetic        Default: solar\n"
              << "  --backend sequential|openmp|cuda  CUDA reports unavailable in Stage 6C\n"
              << "  --threads N                        OpenMP threads (default 4)\n"
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
            } else if (value == "synthetic") {
                options.scenario = ScenarioKind::synthetic;
            } else {
                throw std::invalid_argument("scenario must be solar or synthetic");
            }
        } else if (argument == "--backend") {
            options.backend = value;
        } else if (argument == "--threads") {
            options.openmp_threads = std::stoi(value);
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
    if (options.backend == "cuda") {
        throw std::runtime_error(
            "CUDA visualization is unavailable in Stage 6C; choose sequential or openmp. "
            "Auto awaits Stage 5 calibration and is not operational.");
    }
    if (options.bodies < 2 || options.openmp_threads <= 0 ||
        options.width <= 0 || options.height <= 0) {
        throw std::invalid_argument(
            "bodies >= 2, positive threads, and positive window dimensions are required");
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
        io.FontGlobalScale = 1.08F * ui_scale;
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
    nbody::visualization::ConservationTracker conservation;
    nbody::visualization::TrailHistory trails;
    std::size_t speed_index{0};
    std::size_t selected_body{0};
    std::optional<std::size_t> focused_body;
    bool show_trails{true};
    bool show_labels{true};
    bool show_reference_orbits{true};

    ApplicationModel(const Options& options, BackendKind backend)
        : scenario(nbody::visualization::make_visual_scenario(
              options.scenario, options.bodies, options.seed)),
          controller(
              scenario.initial_state,
              {scenario.dt_s,
               scenario.speed_presets[scenario.default_speed_index].steps_per_frame,
               backend,
               options.openmp_threads}),
          render_space(
              scenario.initial_state,
              scenario.styles,
              9.0F,
              scenario.default_radius_multiplier),
          conservation(scenario.initial_state),
          trails(scenario.initial_state, 768, scenario.trail_sample_interval_s),
          speed_index(scenario.default_speed_index),
          selected_body(scenario.kind == ScenarioKind::solar_system ? 3U : 0U) {}

    void switch_scenario(ScenarioKind kind, std::size_t bodies, std::uint64_t seed) {
        auto next = nbody::visualization::make_visual_scenario(kind, bodies, seed);
        const std::size_t next_speed = next.default_speed_index;
        controller.replace_scenario(
            next.initial_state, next.dt_s, next.speed_presets[next_speed].steps_per_frame);
        scenario = std::move(next);
        render_space = nbody::visualization::RenderSpaceTransform(
            scenario.initial_state, scenario.styles, 9.0F, scenario.default_radius_multiplier);
        conservation.reset(controller.state());
        trails = nbody::visualization::TrailHistory(
            controller.state(), 768, scenario.trail_sample_interval_s);
        speed_index = next_speed;
        selected_body = scenario.kind == ScenarioKind::solar_system ? 3U : 0U;
        show_labels = true;
        show_reference_orbits = scenario.kind == ScenarioKind::solar_system;
        focused_body.reset();
    }

    void reset() {
        controller.reset();
        conservation.reset(controller.state());
        trails.reset(controller.state());
    }
};

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
            input->camera->reset();
        } else if (key == GLFW_KEY_1 && initial_press) {
            app.controller.select_backend(BackendKind::sequential);
            log_backend(app.controller);
        } else if (key == GLFW_KEY_2 && initial_press) {
            app.controller.select_backend(BackendKind::openmp, input->preferred_openmp_threads);
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
    const ImVec2& display_size) {
    const glm::vec4 clip = camera.projection_matrix() * camera.view_matrix() *
        glm::vec4(position, 1.0F);
    if (clip.w <= 0.0F) {
        return std::nullopt;
    }
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.z < -1.0F || ndc.z > 1.0F) {
        return std::nullopt;
    }
    return glm::vec2(
        (ndc.x * 0.5F + 0.5F) * display_size.x,
        (-ndc.y * 0.5F + 0.5F) * display_size.y);
}

void select_nearest_body(
    ApplicationModel& app,
    const std::vector<nbody::visualization::RenderBody>& bodies,
    const nbody::visualization::OrbitCamera& camera,
    double mouse_x,
    double mouse_y) {
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    float best_distance = 22.0F;
    std::optional<std::size_t> best;
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        const auto screen = project_to_screen(bodies[index].position, camera, display);
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
    const ApplicationModel& app,
    std::vector<nbody::visualization::RenderTrail>& output) {
    if (!app.show_trails) {
        output.clear();
    } else {
        const auto& physical = app.trails.positions();
        output.resize(physical.size());
        for (std::size_t index = 0; index < physical.size(); ++index) {
            auto& trail = output[index];
            trail.color = app.scenario.styles[index].color * 0.76F;
            trail.alpha = 0.68F;
            trail.fade_with_age = true;
            trail.points.clear();
            trail.points.reserve(physical[index].size());
            for (const auto& point : physical[index]) {
                trail.points.push_back(app.render_space.to_render_position(point));
            }
        }
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
            guide.points.reserve(app.scenario.reference_paths_m[index].size());
            for (const auto& point : app.scenario.reference_paths_m[index]) {
                guide.points.push_back(app.render_space.to_render_position(point));
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
    std::optional<nbody::visualization::CameraPreset> camera_preset;
    bool reset{false};
    bool frame_focus{false};
    float panel_right{0.0F};
};

void section_heading(const char* label) {
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.43F, 0.86F, 0.82F, 1.0F));
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::Separator();
}

UiActions draw_control_panel(
    ApplicationModel& app,
    int& preferred_openmp_threads,
    std::size_t synthetic_bodies,
    std::uint64_t synthetic_seed) {
    UiActions actions;
    ImGui::SetNextWindowPos({16.0F, 16.0F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({438.0F, 980.0F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints({410.0F, 600.0F}, {480.0F, ImGui::GetIO().DisplaySize.y - 24.0F});
    ImGui::Begin("Scientific Visualization", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.96F, 0.79F, 0.35F, 1.0F));
    ImGui::SetWindowFontScale(1.16F);
    ImGui::TextUnformatted("SOLAR SYSTEM SIMULATOR");
    ImGui::SetWindowFontScale(1.0F);
    ImGui::PopStyleColor();
    ImGui::TextDisabled("N-body dynamics  /  parallel compute  /  live integrity");

    section_heading("SIMULATION");
    int scenario_index = app.scenario.kind == ScenarioKind::solar_system ? 0 : 1;
    const char* scenarios[] = {"Solar System", "Compute Experiment"};
    if (ImGui::Combo("Scenario", &scenario_index, scenarios, 2)) {
        actions.scenario = scenario_index == 0 ? ScenarioKind::solar_system : ScenarioKind::synthetic;
    }
    if (app.scenario.kind == ScenarioKind::synthetic) {
        ImGui::TextDisabled(
            "Deterministic synthetic N-body workload  /  N=%zu  seed=%llu",
            synthetic_bodies, static_cast<unsigned long long>(synthetic_seed));
    }
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16F, 0.38F, 0.42F, 1.0F));
    if (ImGui::Button(app.controller.paused() ? "PLAY" : "PAUSE", {112.0F, 0.0F})) {
        app.controller.toggle_paused();
    }
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (ImGui::Button("RESET", {92.0F, 0.0F})) {
        actions.reset = true;
    }
    ImGui::SameLine();
    ImGui::TextColored(
        app.controller.paused() ? ImVec4(0.92F, 0.72F, 0.35F, 1.0F)
                                : ImVec4(0.40F, 0.88F, 0.62F, 1.0F),
        "%s", app.controller.paused() ? "PAUSED" : "RUNNING");
    ImGui::Text("Elapsed simulated time     %s", human_time(app.controller.state().time_s).c_str());

    std::vector<const char*> speed_names;
    for (const auto& preset : app.scenario.speed_presets) {
        speed_names.push_back(preset.name);
    }
    int speed_index = static_cast<int>(app.speed_index);
    ImGui::SetNextItemWidth(-1.0F);
    if (ImGui::Combo("##SimulationSpeed", &speed_index, speed_names.data(),
                     static_cast<int>(speed_names.size()))) {
        app.speed_index = static_cast<std::size_t>(speed_index);
        app.controller.set_steps_per_frame(
            app.scenario.speed_presets[app.speed_index].steps_per_frame);
    }
    ImGui::TextDisabled("Fixed-step simulation batch; actual wall-time rate follows render FPS.");

    section_heading("COMPUTE");
    ImGui::Text("ACTIVE  %s", app.controller.backend_kind() == BackendKind::openmp
        ? "OPENMP" : "SEQUENTIAL");
    if (ImGui::RadioButton("Sequential", app.controller.backend_kind() == BackendKind::sequential)) {
        app.controller.select_backend(BackendKind::sequential);
    }
    const bool openmp_available = nbody::visualization::SimulationController::openmp_available();
    ImGui::BeginDisabled(!openmp_available);
    if (ImGui::RadioButton("OpenMP", app.controller.backend_kind() == BackendKind::openmp)) {
        app.controller.select_backend(BackendKind::openmp, preferred_openmp_threads);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("CUDA");
    ImGui::SameLine();
    ImGui::TextDisabled("Auto");
    if (app.controller.backend_kind() == BackendKind::openmp) {
        const int hardware_threads =
            static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
        const int maximum_threads = std::min(64, hardware_threads);
        if (ImGui::SliderInt("OpenMP threads", &preferred_openmp_threads, 1, maximum_threads)) {
            app.controller.select_backend(BackendKind::openmp, preferred_openmp_threads);
        }
    }
    ImGui::TextDisabled("CUDA not integrated  /  Auto awaits Stage 5 calibration");

    section_heading(app.scenario.kind == ScenarioKind::solar_system
        ? "SELECTED BODY" : "SELECTED WORKLOAD BODY");
    std::vector<const char*> body_names;
    for (const auto& body : app.controller.state().bodies) {
        body_names.push_back(body.name.c_str());
    }
    int selected = static_cast<int>(app.selected_body);
    if (ImGui::Combo("Body", &selected, body_names.data(), static_cast<int>(body_names.size()))) {
        app.selected_body = static_cast<std::size_t>(selected);
    }
    const auto& body = app.controller.state().bodies[app.selected_body];
    const auto& style = app.scenario.styles[app.selected_body];
    ImGui::Text("Name                       %s", body.name.c_str());
    const char* distance_label = app.scenario.kind == ScenarioKind::solar_system
        ? "Distance from Sun" : "Distance from body 0";
    ImGui::Text("%-26s %.6f AU", distance_label,
                nbody::norm(body.position_m - app.controller.state().bodies[0].position_m) /
                    149597870700.0);
    ImGui::Text("Speed                      %.3f km/s",
                nbody::norm(body.velocity_m_s) / 1000.0);
    ImGui::Text("Mass                       %.6e kg", body.mass_kg);
    if (style.physical_radius_m > 0.0) {
        ImGui::Text("Mean radius                %.1f km", style.physical_radius_m / 1000.0);
    } else {
        ImGui::TextDisabled("Physical radius not defined for synthetic workload");
    }
    if (ImGui::Button("FOCUS SELECTED", {154.0F, 0.0F})) {
        app.focused_body = app.selected_body;
        actions.frame_focus = true;
    }

    section_heading("NUMERICAL HEALTH");
    const auto& drift = app.conservation.drift();
    const auto health = nbody::visualization::classify_numerical_health(
        drift.energy_relative, drift.angular_momentum_relative);
    const auto health_color = [](bool okay) {
        return okay ? ImVec4(0.38F, 0.90F, 0.61F, 1.0F)
                    : ImVec4(1.0F, 0.34F, 0.28F, 1.0F);
    };
    ImGui::Text("Energy drift       %.3e / %.1e", drift.energy_relative,
                nbody::visualization::energy_drift_bound);
    ImGui::SameLine();
    ImGui::TextColored(health_color(health.energy_within_bound), "%s",
        health.energy_within_bound ? "PASS" : "EXCEEDED");
    ImGui::Text("Angular drift      %.3e / %.1e", drift.angular_momentum_relative,
                nbody::visualization::angular_momentum_drift_bound);
    ImGui::SameLine();
    ImGui::TextColored(health_color(health.angular_momentum_within_bound), "%s",
        health.angular_momentum_within_bound ? "PASS" : "EXCEEDED");
    ImGui::TextColored(health_color(health.all_within_bounds()), "%s",
        health.all_within_bounds() ? "WITHIN ESTABLISHED BOUNDS" : "NUMERICAL BOUND WARNING");

    section_heading("VIEW");
    if (ImGui::Button("OVERVIEW")) {
        actions.camera_preset = nbody::visualization::CameraPreset::overview;
    }
    if (app.scenario.kind == ScenarioKind::solar_system) {
        ImGui::SameLine();
        if (ImGui::Button("INNER SYSTEM")) {
            actions.camera_preset = nbody::visualization::CameraPreset::inner_system;
        }
        ImGui::SameLine();
        if (ImGui::Button("EARTH")) {
            actions.camera_preset = nbody::visualization::CameraPreset::earth;
        }
        if (ImGui::Button("JUPITER")) {
            actions.camera_preset = nbody::visualization::CameraPreset::jupiter;
        }
        ImGui::SameLine();
        if (ImGui::Button("SATURN")) {
            actions.camera_preset = nbody::visualization::CameraPreset::saturn;
        }
    }
    ImGui::Checkbox("Simulation history trails", &app.show_trails);
    if (app.scenario.kind == ScenarioKind::solar_system) {
        ImGui::Checkbox("Reference orbit guides", &app.show_reference_orbits);
    }
    ImGui::Checkbox("Body labels", &app.show_labels);
    float radius_multiplier = app.render_space.radius_multiplier();
    if (ImGui::SliderFloat("Visual radius x", &radius_multiplier, 1.0F, 500.0F, "%.1f",
                           ImGuiSliderFlags_Logarithmic)) {
        app.render_space.set_radius_multiplier(radius_multiplier);
    }
    ImGui::TextWrapped(
        "Positions remain linear and physical. Body sizes and orbit guides are presentation-only.");

    section_heading("VIEWER PERFORMANCE");
    const float fps = ImGui::GetIO().Framerate;
    ImGui::Text("%.1f FPS  /  %.2f ms frame", fps, fps > 0.0F ? 1000.0F / fps : 0.0F);
    ImGui::TextColored(ImVec4(0.92F, 0.67F, 0.31F, 1.0F),
                       "DISPLAY ONLY - NOT BENCHMARK EVIDENCE");

    if (ImGui::CollapsingHeader("Advanced / Technical Details")) {
        ImGui::TextWrapped("%s", app.scenario.provenance.c_str());
        ImGui::Text("Position (m)  (%.3e, %.3e, %.3e)",
                    body.position_m.x, body.position_m.y, body.position_m.z);
        ImGui::Text("Velocity (m/s) (%.3e, %.3e, %.3e)",
                    body.velocity_m_s.x, body.velocity_m_s.y, body.velocity_m_s.z);
        ImGui::Text("Maximum energy drift        %.6e", drift.maximum_energy_relative);
        ImGui::Text("Maximum angular drift       %.6e", drift.maximum_angular_momentum_relative);
        ImGui::Text("Fixed physical dt           %.2f hours", app.controller.dt_s() / 3600.0);
        ImGui::Text("Physics steps per render    %zu", app.controller.steps_per_frame());
        ImGui::Text("Minimum render radius       %.3f", app.render_space.minimum_render_radius());
        int trail_capacity = static_cast<int>(app.trails.capacity());
        if (ImGui::SliderInt("Trail history points", &trail_capacity, 64, 2048)) {
            app.trails.set_capacity(static_cast<std::size_t>(trail_capacity));
        }
        if (app.scenario.kind == ScenarioKind::synthetic) {
            ImGui::Text("Deterministic workload N=%zu seed=%llu", synthetic_bodies,
                        static_cast<unsigned long long>(synthetic_seed));
        }
    }

    actions.panel_right = ImGui::GetWindowPos().x + ImGui::GetWindowSize().x;
    ImGui::End();
    return actions;
}

void draw_labels(
    const ApplicationModel& app,
    const std::vector<nbody::visualization::RenderBody>& bodies,
    const nbody::visualization::OrbitCamera& camera,
    float panel_right) {
    if (!app.show_labels) {
        return;
    }
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    auto* draw_list = ImGui::GetForegroundDrawList();
    std::vector<nbody::visualization::LabelCandidate> candidates;
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        if (app.scenario.label_policy == nbody::visualization::LabelPolicy::selected_only &&
            index != app.selected_body) {
            continue;
        }
        const auto screen = project_to_screen(bodies[index].position, camera, display);
        if (!screen || screen->x < 0.0F || screen->x > display.x ||
            screen->y < 0.0F || screen->y > display.y) {
            continue;
        }
        const char* label = app.controller.state().bodies[index].name.c_str();
        const ImVec2 size = ImGui::CalcTextSize(label);
        const bool selected = index == app.selected_body;
        const int priority = selected ? 1000 : (index == 0 ? 900 : 700 - static_cast<int>(index));
        candidates.push_back({
            index,
            {screen->x, screen->y},
            {size.x + 10.0F, size.y + 6.0F},
            priority,
            selected,
        });
    }
    const auto placements = nbody::visualization::declutter_labels(
        std::move(candidates), {display.x, display.y}, panel_right + 8.0F, 5.0F);
    for (const auto& placement : placements) {
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
        const auto backend = options.backend == "openmp" ? BackendKind::openmp : BackendKind::sequential;
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

            camera.set_aspect_ratio(
                static_cast<float>(framebuffer_width) / static_cast<float>(framebuffer_height));
            app.controller.advance_frame();
            app.trails.sample(app.controller.state());
            app.conservation.update(app.controller.state());
            app.render_space.update_snapshot(app.controller.state(), snapshot);
            if (app.focused_body && *app.focused_body < snapshot.size()) {
                camera.approach_target(snapshot[*app.focused_body].position, 0.14F);
            }
            update_render_trails(app, render_trails);

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            if (input.pending_selection && !ImGui::GetIO().WantCaptureMouse) {
                select_nearest_body(app, snapshot, camera, input.selection_x, input.selection_y);
            }
            input.pending_selection = false;

            const UiActions actions = draw_control_panel(
                app, input.preferred_openmp_threads, options.bodies, options.seed);
            bool state_replaced = false;
            if (actions.reset) {
                app.reset();
                state_replaced = true;
            }
            if (actions.scenario) {
                app.switch_scenario(*actions.scenario, options.bodies, options.seed);
                camera.reset();
                state_replaced = true;
            }
            if (state_replaced) {
                app.render_space.update_snapshot(app.controller.state(), snapshot);
                update_render_trails(app, render_trails);
            }
            if (actions.camera_preset) {
                const auto framing = nbody::visualization::camera_framing(
                    *actions.camera_preset, app.scenario.kind, snapshot);
                nbody::visualization::apply_camera_framing(camera, framing);
                app.focused_body = framing.tracked_body;
                if (framing.tracked_body) {
                    app.selected_body = *framing.tracked_body;
                }
            }
            if (actions.frame_focus && app.selected_body < snapshot.size()) {
                camera.set_pose(
                    camera.yaw_degrees(), camera.pitch_degrees(), 2.35F,
                    snapshot[app.selected_body].position);
            }

            draw_labels(app, snapshot, camera, actions.panel_right);
            ImGui::Render();
            renderer.render(
                snapshot, render_trails, camera, framebuffer_width, framebuffer_height,
                app.selected_body);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window.get());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visualization error: " << error.what() << '\n';
        return 1;
    }
}
