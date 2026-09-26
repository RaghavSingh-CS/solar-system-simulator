#include "nbody/scenarios.hpp"
#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/controller.hpp"
#include "nbody/visualization/render_space.hpp"
#include "renderer.hpp"

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string scenario{"synthetic32"};
    std::string backend{"sequential"};
    std::size_t bodies{32};
    std::uint64_t seed{24301};
    double dt_s{10.0};
    std::size_t steps_per_frame{256};
    int openmp_threads{4};
    int width{1280};
    int height{720};
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --scenario synthetic32|analytic   Deterministic fixture (default synthetic32)\n"
        << "  --backend sequential|openmp|cuda  Manual backend request; CUDA is unavailable in Stage 6A\n"
        << "  --threads N                        OpenMP threads (default 4)\n"
        << "  --bodies N                         Synthetic body count (default 32)\n"
        << "  --seed N                           Synthetic seed (default 24301)\n"
        << "  --dt SECONDS                       Fixed physical timestep (default 10)\n"
        << "  --steps-per-frame N                Fixed steps per rendered frame (default 256)\n"
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
            options.scenario = value;
        } else if (argument == "--backend") {
            options.backend = value;
        } else if (argument == "--threads") {
            options.openmp_threads = std::stoi(value);
        } else if (argument == "--bodies") {
            options.bodies = std::stoull(value);
        } else if (argument == "--seed") {
            options.seed = std::stoull(value);
        } else if (argument == "--dt") {
            options.dt_s = std::stod(value);
        } else if (argument == "--steps-per-frame") {
            options.steps_per_frame = std::stoull(value);
        } else if (argument == "--width") {
            options.width = std::stoi(value);
        } else if (argument == "--height") {
            options.height = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.scenario != "synthetic32" && options.scenario != "analytic") {
        throw std::invalid_argument("scenario must be synthetic32 or analytic");
    }
    if (options.backend != "sequential" && options.backend != "openmp" &&
        options.backend != "cuda") {
        throw std::invalid_argument("backend must be sequential, openmp, or cuda");
    }
    if (options.backend == "cuda") {
        throw std::runtime_error(
            "CUDA visualization is unavailable in Stage 6A; choose sequential or openmp. "
            "No automatic backend selector is implemented.");
    }
    if (options.bodies < 2 || options.steps_per_frame == 0 || options.openmp_threads <= 0 ||
        !(options.dt_s > 0.0) || options.width <= 0 || options.height <= 0) {
        throw std::invalid_argument(
            "bodies >= 2, positive dt/steps/threads, and positive window dimensions are required");
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

struct InteractionState {
    nbody::visualization::OrbitCamera* camera{nullptr};
    nbody::visualization::SimulationController* controller{nullptr};
    int preferred_openmp_threads{1};
    bool dragging{false};
    double last_x{0.0};
    double last_y{0.0};
};

InteractionState& interaction(GLFWwindow* window) {
    auto* state = static_cast<InteractionState*>(glfwGetWindowUserPointer(window));
    if (state == nullptr) {
        throw std::runtime_error("window interaction state is unavailable");
    }
    return *state;
}

void log_backend(const nbody::visualization::SimulationController& controller) {
    std::cout << "active_backend=" << controller.backend_name();
    if (controller.backend_kind() == nbody::visualization::BackendKind::openmp) {
        std::cout << " threads=" << controller.openmp_threads();
    }
    std::cout << '\n';
}

void key_callback(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) {
        return;
    }
    try {
        auto& state = interaction(window);
        const bool initial_press = action == GLFW_PRESS;
        if (key == GLFW_KEY_ESCAPE && initial_press) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (key == GLFW_KEY_SPACE && initial_press) {
            state.controller->toggle_paused();
            std::cout << (state.controller->paused() ? "simulation_paused\n" : "simulation_resumed\n");
        } else if (key == GLFW_KEY_R && initial_press) {
            state.controller->reset();
            std::cout << "scenario_reset\n";
        } else if (key == GLFW_KEY_C && initial_press) {
            state.camera->reset();
        } else if (key == GLFW_KEY_1 && initial_press) {
            state.controller->select_backend(nbody::visualization::BackendKind::sequential);
            log_backend(*state.controller);
        } else if (key == GLFW_KEY_2 && initial_press) {
            state.controller->select_backend(
                nbody::visualization::BackendKind::openmp,
                state.preferred_openmp_threads);
            log_backend(*state.controller);
        } else if (key == GLFW_KEY_A) {
            state.camera->orbit(-8.0F, 0.0F);
        } else if (key == GLFW_KEY_D) {
            state.camera->orbit(8.0F, 0.0F);
        } else if (key == GLFW_KEY_UP) {
            state.camera->orbit(0.0F, -8.0F);
        } else if (key == GLFW_KEY_DOWN) {
            state.camera->orbit(0.0F, 8.0F);
        } else if (key == GLFW_KEY_W) {
            state.camera->dolly(1.0F);
        } else if (key == GLFW_KEY_S) {
            state.camera->dolly(-1.0F);
        }
    } catch (const std::exception& error) {
        std::cerr << "input error: " << error.what() << '\n';
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }
    auto& state = interaction(window);
    state.dragging = action == GLFW_PRESS;
    if (state.dragging) {
        glfwGetCursorPos(window, &state.last_x, &state.last_y);
    }
}

void cursor_callback(GLFWwindow* window, double x, double y) {
    auto& state = interaction(window);
    if (state.dragging) {
        state.camera->orbit(
            static_cast<float>(x - state.last_x),
            static_cast<float>(y - state.last_y));
    }
    state.last_x = x;
    state.last_y = y;
}

void scroll_callback(GLFWwindow* window, double, double y_offset) {
    interaction(window).camera->dolly(static_cast<float>(y_offset));
}

std::string window_title(
    const Options& options,
    const nbody::visualization::SimulationController& controller) {
    std::ostringstream title;
    title << "N-body Visualizer | "
          << (options.scenario == "synthetic32" ? "synthetic fixture (not Solar System)" :
                                                  "analytic two-body fixture")
          << " | " << controller.backend_name();
    if (controller.backend_kind() == nbody::visualization::BackendKind::openmp) {
        title << '-' << controller.openmp_threads();
    }
    title << " | " << (controller.paused() ? "PAUSED" : "RUNNING")
          << " | t=" << std::fixed << std::setprecision(0) << controller.state().time_s << " s"
          << " | " << controller.steps_per_frame() << " steps/frame";
    return title.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        nbody::SimulationState initial_state;
        if (options.scenario == "analytic") {
            initial_state = nbody::make_analytic_circular_two_body().initial_state;
        } else {
            initial_state = nbody::make_seeded_synthetic_system(options.bodies, options.seed);
        }

        const auto backend = options.backend == "openmp"
            ? nbody::visualization::BackendKind::openmp
            : nbody::visualization::BackendKind::sequential;
        nbody::visualization::SimulationController controller(
            std::move(initial_state),
            {options.dt_s, options.steps_per_frame, backend, options.openmp_threads});
        nbody::visualization::RenderSpaceTransform render_space(controller.initial_state());
        nbody::visualization::OrbitCamera camera;

        GlfwRuntime glfw_runtime;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);
        std::unique_ptr<GLFWwindow, WindowDeleter> window(glfwCreateWindow(
            options.width, options.height, "N-body Visualizer", nullptr, nullptr));
        if (!window) {
            throw std::runtime_error(
                "OpenGL window/context creation failed; verify OpenGL 3.3 support and desktop access");
        }
        glfwMakeContextCurrent(window.get());
        glfwSwapInterval(1);
        const int loaded_version = gladLoadGL(
            reinterpret_cast<GLADloadfunc>(glfwGetProcAddress));
        if (loaded_version == 0) {
            throw std::runtime_error("OpenGL loader initialization failed after context creation");
        }

        const auto gl_string = [](unsigned int name) {
            const auto* value = glGetString(name);
            return value == nullptr ? "unavailable" : reinterpret_cast<const char*>(value);
        };
        std::cout << "scenario=" << options.scenario
                  << " provenance=deterministic_fixture_not_solar_system\n"
                  << "fixed_dt_s=" << controller.dt_s()
                  << " steps_per_rendered_frame=" << controller.steps_per_frame() << '\n'
                  << "render_position_scale=" << render_space.position_scale()
                  << " render_radius_exaggeration=" << render_space.radius_exaggeration() << '\n'
                  << "OpenGL vendor=" << gl_string(GL_VENDOR)
                  << " renderer=" << gl_string(GL_RENDERER)
                  << " version=" << gl_string(GL_VERSION) << '\n'
                  << "controls: Space pause/resume, R reset, 1 sequential, 2 OpenMP, "
                     "left-drag/A-D/arrows orbit, wheel/W-S zoom, C reset camera, Esc exit\n";
        log_backend(controller);

        InteractionState input{&camera, &controller, options.openmp_threads};
        glfwSetWindowUserPointer(window.get(), &input);
        glfwSetKeyCallback(window.get(), key_callback);
        glfwSetMouseButtonCallback(window.get(), mouse_button_callback);
        glfwSetCursorPosCallback(window.get(), cursor_callback);
        glfwSetScrollCallback(window.get(), scroll_callback);

        nbody::visualization::BodyRenderer renderer;
        std::vector<nbody::visualization::RenderBody> snapshot;
        snapshot.reserve(controller.state().bodies.size());
        double last_title_update = -1.0;
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
            controller.advance_frame();
            render_space.update_snapshot(controller.state(), snapshot);
            renderer.render(snapshot, camera, framebuffer_width, framebuffer_height);
            glfwSwapBuffers(window.get());

            const double now = glfwGetTime();
            if (now - last_title_update >= 0.25) {
                glfwSetWindowTitle(window.get(), window_title(options, controller).c_str());
                last_title_update = now;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visualization error: " << error.what() << '\n';
        return 1;
    }
}
