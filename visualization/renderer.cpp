#include "renderer.hpp"
#include "nbody/visualization/workload.hpp"

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat3x3.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nbody::visualization {

namespace {

constexpr const char* vertex_shader_source = R"glsl(
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 world_normal;
out vec3 world_position;
out vec3 object_normal;

void main() {
    world_position = vec3(model * vec4(in_position, 1.0));
    world_normal = mat3(transpose(inverse(model))) * in_normal;
    object_normal = in_normal;
    gl_Position = projection * view * vec4(world_position, 1.0);
}
)glsl";

constexpr const char* fragment_shader_source = R"glsl(
#version 330 core
in vec3 world_normal;
in vec3 world_position;
in vec3 object_normal;

uniform vec3 body_color;
uniform vec3 sun_position;
uniform int emissive;
uniform int material_kind;

out vec4 fragment_color;

void main() {
    if (emissive != 0) {
        float granulation = 0.90 + 0.10 * sin(object_normal.x * 19.0 +
            sin(object_normal.y * 23.0) + object_normal.z * 17.0);
        fragment_color = vec4(body_color * granulation, 1.0);
        return;
    }
    vec3 normal = normalize(world_normal);
    vec3 light_direction = normalize(sun_position - world_position);
    float diffuse = max(dot(normal, light_direction), 0.0);
    vec3 identity = body_color;
    if (material_kind == 4) {
        float continents = sin(object_normal.x * 11.0 + object_normal.z * 5.0) +
            sin(object_normal.z * 17.0 - object_normal.y * 7.0);
        if (continents > 0.75) identity = mix(body_color, vec3(0.24, 0.55, 0.27), 0.72);
        float cloud = smoothstep(0.80, 0.97, sin(object_normal.x * 31.0 + object_normal.z * 13.0));
        identity = mix(identity, vec3(0.92), cloud * 0.35);
    } else if (material_kind == 6) {
        float bands = 0.76 + 0.24 * sin(object_normal.y * 42.0 + sin(object_normal.x * 5.0));
        identity *= bands;
    } else if (material_kind == 7) {
        float bands = 0.86 + 0.14 * sin(object_normal.y * 48.0);
        identity *= bands;
    } else if (material_kind == 2) {
        identity *= 0.82 + 0.18 * sin(object_normal.x * 16.0 + object_normal.z * 9.0);
    } else if (material_kind == 3) {
        identity *= 0.92 + 0.08 * sin(object_normal.y * 20.0);
    }
    vec3 color = identity * (0.20 + 0.80 * diffuse);
    fragment_color = vec4(color, 1.0);
}
)glsl";

constexpr const char* line_vertex_shader_source = R"glsl(
#version 330 core
layout(location = 0) in vec3 in_position;
uniform mat4 view;
uniform mat4 projection;
uniform int vertex_count;
uniform int fade_with_age;
out float age_alpha;
out float path_fraction;
void main() {
    gl_Position = projection * view * vec4(in_position, 1.0);
    float fraction = vertex_count > 1 ? float(gl_VertexID) / float(vertex_count - 1) : 1.0;
    age_alpha = fade_with_age != 0 ? mix(0.08, 1.0, fraction) : 1.0;
    path_fraction = fraction;
}
)glsl";

constexpr const char* line_fragment_shader_source = R"glsl(
#version 330 core
uniform vec3 trail_color;
uniform float base_alpha;
uniform int dashed;
in float age_alpha;
in float path_fraction;
out vec4 fragment_color;
void main() {
    if (dashed != 0 && fract(path_fraction * 24.0) < 0.38) {
        discard;
    }
    fragment_color = vec4(trail_color, base_alpha * age_alpha);
}
)glsl";

constexpr const char* ring_vertex_shader_source = R"glsl(
#version 330 core
layout(location = 0) in vec3 in_position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out float ring_radius;
void main() {
    ring_radius = length(in_position.xz);
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
)glsl";

constexpr const char* ring_fragment_shader_source = R"glsl(
#version 330 core
in float ring_radius;
out vec4 fragment_color;
void main() {
    float stripe = 0.68 + 0.20 * sin(ring_radius * 65.0);
    float edge = smoothstep(1.05, 1.16, ring_radius) * (1.0 - smoothstep(2.20, 2.34, ring_radius));
    fragment_color = vec4(vec3(0.78, 0.68, 0.48) * stripe, 0.58 * edge);
}
)glsl";

constexpr const char* star_vertex_shader_source = R"glsl(
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_brightness;
uniform mat4 view;
uniform mat4 projection;
out float brightness;
void main() {
    gl_Position = projection * view * vec4(in_position, 1.0);
    gl_PointSize = 1.0 + 1.5 * in_brightness;
    brightness = in_brightness;
}
)glsl";

constexpr const char* star_fragment_shader_source = R"glsl(
#version 330 core
in float brightness;
out vec4 fragment_color;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float soft = 1.0 - smoothstep(0.15, 1.0, dot(p, p));
    fragment_color = vec4(vec3(0.70, 0.78, 0.92) * (0.45 + 0.55 * brightness), soft * 0.72);
}
)glsl";

constexpr const char* point_body_vertex_shader_source = R"glsl(
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in float in_size;
uniform mat4 view;
uniform mat4 projection;
out vec3 point_color;
void main() {
    gl_Position = projection * view * vec4(in_position, 1.0);
    gl_PointSize = in_size;
    point_color = in_color;
}
)glsl";

constexpr const char* point_body_fragment_shader_source = R"glsl(
#version 330 core
in vec3 point_color;
out vec4 fragment_color;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float distance_squared = dot(p, p);
    if (distance_squared > 1.0) discard;
    float light = 0.72 + 0.28 * (1.0 - distance_squared);
    fragment_color = vec4(point_color * light, 0.96);
}
)glsl";

struct Vertex {
    float position[3];
    float normal[3];
};

struct StarVertex {
    glm::vec3 position;
    float brightness;
};

struct PointBodyVertex {
    glm::vec3 position;
    glm::vec3 color;
    float size_pixels;
};

unsigned int compile_shader(unsigned int type, const char* source, const char* label) {
    const unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    int compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE) {
        return shader;
    }
    int length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(length > 0 ? length : 1), '\0');
    glGetShaderInfoLog(shader, length, nullptr, log.data());
    glDeleteShader(shader);
    throw std::runtime_error(std::string(label) + " shader compilation failed: " + log);
}

unsigned int create_program() {
    const unsigned int vertex = compile_shader(GL_VERTEX_SHADER, vertex_shader_source, "vertex");
    unsigned int fragment = 0;
    try {
        fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source, "fragment");
    } catch (...) {
        glDeleteShader(vertex);
        throw;
    }
    const unsigned int program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    int linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_TRUE) {
        return program;
    }
    int length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(length > 0 ? length : 1), '\0');
    glGetProgramInfoLog(program, length, nullptr, log.data());
    glDeleteProgram(program);
    throw std::runtime_error("shader program link failed: " + log);
}

unsigned int create_line_program() {
    const unsigned int vertex =
        compile_shader(GL_VERTEX_SHADER, line_vertex_shader_source, "trail vertex");
    unsigned int fragment = 0;
    try {
        fragment = compile_shader(
            GL_FRAGMENT_SHADER, line_fragment_shader_source, "trail fragment");
    } catch (...) {
        glDeleteShader(vertex);
        throw;
    }
    const unsigned int program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    int linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_TRUE) {
        return program;
    }
    int length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(length > 0 ? length : 1), '\0');
    glGetProgramInfoLog(program, length, nullptr, log.data());
    glDeleteProgram(program);
    throw std::runtime_error("trail shader program link failed: " + log);
}

unsigned int create_aux_program(
    const char* vertex_source,
    const char* fragment_source,
    const char* label) {
    const unsigned int vertex = compile_shader(GL_VERTEX_SHADER, vertex_source, label);
    unsigned int fragment = 0;
    try {
        fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source, label);
    } catch (...) {
        glDeleteShader(vertex);
        throw;
    }
    const unsigned int program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    int linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_TRUE) {
        return program;
    }
    int length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(length > 0 ? length : 1), '\0');
    glGetProgramInfoLog(program, length, nullptr, log.data());
    glDeleteProgram(program);
    throw std::runtime_error(std::string(label) + " program link failed: " + log);
}

void make_sphere(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices) {
    constexpr int stacks = 16;
    constexpr int sectors = 24;
    constexpr float pi = 3.14159265358979323846F;
    vertices.reserve(static_cast<std::size_t>((stacks + 1) * (sectors + 1)));
    for (int stack = 0; stack <= stacks; ++stack) {
        const float phi = pi * static_cast<float>(stack) / static_cast<float>(stacks);
        const float y = std::cos(phi);
        const float ring = std::sin(phi);
        for (int sector = 0; sector <= sectors; ++sector) {
            const float theta = 2.0F * pi * static_cast<float>(sector) /
                static_cast<float>(sectors);
            const float x = ring * std::cos(theta);
            const float z = ring * std::sin(theta);
            vertices.push_back({{x, y, z}, {x, y, z}});
        }
    }
    for (int stack = 0; stack < stacks; ++stack) {
        const unsigned int row = static_cast<unsigned int>(sectors + 1);
        const unsigned int first = static_cast<unsigned int>(stack) * row;
        const unsigned int next = first + row;
        for (int sector = 0; sector < sectors; ++sector) {
            const unsigned int offset = static_cast<unsigned int>(sector);
            if (stack != 0) {
                indices.insert(indices.end(), {first + offset, next + offset, first + offset + 1});
            }
            if (stack != stacks - 1) {
                indices.insert(indices.end(), {first + offset + 1, next + offset, next + offset + 1});
            }
        }
    }
}

void make_ring(std::vector<glm::vec3>& vertices) {
    constexpr int segments = 128;
    constexpr float pi = 3.14159265358979323846F;
    vertices.reserve(static_cast<std::size_t>((segments + 1) * 2));
    for (int segment = 0; segment <= segments; ++segment) {
        const float angle = 2.0F * pi * static_cast<float>(segment) /
            static_cast<float>(segments);
        const float x = std::cos(angle);
        const float z = std::sin(angle);
        vertices.push_back({1.15F * x, 0.0F, 1.15F * z});
        vertices.push_back({2.30F * x, 0.0F, 2.30F * z});
    }
}

std::vector<StarVertex> make_stars() {
    constexpr std::size_t count = 720;
    constexpr float pi = 3.14159265358979323846F;
    std::vector<StarVertex> stars;
    stars.reserve(count);
    std::uint32_t state = 0x6c8e9cf5U;
    auto unit = [&state]() {
        state = state * 1664525U + 1013904223U;
        return static_cast<float>(state >> 8) / static_cast<float>(0x00ffffffU);
    };
    for (std::size_t index = 0; index < count; ++index) {
        const float z = 2.0F * unit() - 1.0F;
        const float angle = 2.0F * pi * unit();
        const float ring = std::sqrt(std::max(0.0F, 1.0F - z * z));
        const float radius = 82.0F + 8.0F * unit();
        stars.push_back({
            radius * glm::vec3(ring * std::cos(angle), z, ring * std::sin(angle)),
            0.25F + 0.75F * unit(),
        });
    }
    return stars;
}

int uniform_location(unsigned int program, const char* name) {
    const int location = glGetUniformLocation(program, name);
    if (location < 0) {
        throw std::runtime_error(std::string("required shader uniform is unavailable: ") + name);
    }
    return location;
}

const char* gl_error_name(GLenum error) noexcept {
    switch (error) {
        case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
#ifdef GL_CONTEXT_LOST
        case GL_CONTEXT_LOST: return "GL_CONTEXT_LOST";
#endif
        default: return "unknown OpenGL error";
    }
}

[[noreturn]] void throw_gl_error(const char* operation, GLenum error) {
    std::ostringstream message;
    message << operation << " failed with " << gl_error_name(error)
            << " (0x" << std::hex << static_cast<unsigned int>(error) << ')';
    throw std::runtime_error(message.str());
}

void checked_buffer_data(
    GLenum target,
    std::size_t element_count,
    std::size_t element_size,
    const void* data,
    GLenum usage,
    const char* operation) {
    const std::size_t bytes = checked_render_upload_bytes(element_count, element_size);
    if (bytes > static_cast<std::size_t>(std::numeric_limits<GLsizeiptr>::max())) {
        throw std::overflow_error(std::string(operation) +
            " exceeds the OpenGL buffer-size range");
    }
    const GLenum prior_error = glGetError();
    if (prior_error != GL_NO_ERROR) {
        throw_gl_error("OpenGL state before buffer upload", prior_error);
    }
    glBufferData(
        target, static_cast<GLsizeiptr>(bytes), data, usage);
    const GLenum upload_error = glGetError();
    if (!render_upload_allows_draw(static_cast<unsigned int>(upload_error))) {
        // Do not issue a draw after a failed allocation/upload. OpenGL retains
        // implementation-defined prior storage; this frame is explicitly invalid.
        throw_gl_error(operation, upload_error);
    }
}

}  // namespace

BodyRenderer::BodyRenderer() : program_(create_program()) {
    try {
        line_program_ = create_line_program();
        ring_program_ = create_aux_program(
            ring_vertex_shader_source, ring_fragment_shader_source, "Saturn ring");
        star_program_ = create_aux_program(
            star_vertex_shader_source, star_fragment_shader_source, "starfield");
        point_program_ = create_aux_program(
            point_body_vertex_shader_source, point_body_fragment_shader_source,
            "compute point body");
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        make_sphere(vertices, indices);
        index_count_ = static_cast<int>(indices.size());

        glGenVertexArrays(1, &vertex_array_);
        glGenBuffers(1, &vertex_buffer_);
        glGenBuffers(1, &index_buffer_);
        glBindVertexArray(vertex_array_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
        checked_buffer_data(
            GL_ARRAY_BUFFER, vertices.size(), sizeof(Vertex), vertices.data(),
            GL_STATIC_DRAW, "sphere vertex-buffer upload");
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
        checked_buffer_data(
            GL_ELEMENT_ARRAY_BUFFER, indices.size(), sizeof(unsigned int),
            indices.data(), GL_STATIC_DRAW, "sphere index-buffer upload");
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
            reinterpret_cast<const void*>(offsetof(Vertex, position)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
            reinterpret_cast<const void*>(offsetof(Vertex, normal)));
        glBindVertexArray(0);

        model_location_ = uniform_location(program_, "model");
        view_location_ = uniform_location(program_, "view");
        projection_location_ = uniform_location(program_, "projection");
        color_location_ = uniform_location(program_, "body_color");
        sun_position_location_ = uniform_location(program_, "sun_position");
        emissive_location_ = uniform_location(program_, "emissive");
        material_location_ = uniform_location(program_, "material_kind");

        glGenVertexArrays(1, &line_vertex_array_);
        glGenBuffers(1, &line_vertex_buffer_);
        glBindVertexArray(line_vertex_array_);
        glBindBuffer(GL_ARRAY_BUFFER, line_vertex_buffer_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
        glBindVertexArray(0);
        line_view_location_ = uniform_location(line_program_, "view");
        line_projection_location_ = uniform_location(line_program_, "projection");
        line_color_location_ = uniform_location(line_program_, "trail_color");
        line_alpha_location_ = uniform_location(line_program_, "base_alpha");
        line_fade_location_ = uniform_location(line_program_, "fade_with_age");
        line_count_location_ = uniform_location(line_program_, "vertex_count");
        line_style_location_ = uniform_location(line_program_, "dashed");

        std::vector<glm::vec3> ring_vertices;
        make_ring(ring_vertices);
        ring_vertex_count_ = static_cast<int>(ring_vertices.size());
        glGenVertexArrays(1, &ring_vertex_array_);
        glGenBuffers(1, &ring_vertex_buffer_);
        glBindVertexArray(ring_vertex_array_);
        glBindBuffer(GL_ARRAY_BUFFER, ring_vertex_buffer_);
        checked_buffer_data(
            GL_ARRAY_BUFFER, ring_vertices.size(), sizeof(glm::vec3),
            ring_vertices.data(), GL_STATIC_DRAW, "ring vertex-buffer upload");
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
        glBindVertexArray(0);
        ring_model_location_ = uniform_location(ring_program_, "model");
        ring_view_location_ = uniform_location(ring_program_, "view");
        ring_projection_location_ = uniform_location(ring_program_, "projection");

        const auto stars = make_stars();
        star_count_ = static_cast<int>(stars.size());
        glGenVertexArrays(1, &star_vertex_array_);
        glGenBuffers(1, &star_vertex_buffer_);
        glBindVertexArray(star_vertex_array_);
        glBindBuffer(GL_ARRAY_BUFFER, star_vertex_buffer_);
        checked_buffer_data(
            GL_ARRAY_BUFFER, stars.size(), sizeof(StarVertex), stars.data(),
            GL_STATIC_DRAW, "starfield vertex-buffer upload");
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, sizeof(StarVertex),
            reinterpret_cast<const void*>(offsetof(StarVertex, position)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(
            1, 1, GL_FLOAT, GL_FALSE, sizeof(StarVertex),
            reinterpret_cast<const void*>(offsetof(StarVertex, brightness)));
        glBindVertexArray(0);
        star_view_location_ = uniform_location(star_program_, "view");
        star_projection_location_ = uniform_location(star_program_, "projection");

        glGenVertexArrays(1, &point_vertex_array_);
        glGenBuffers(1, &point_vertex_buffer_);
        glBindVertexArray(point_vertex_array_);
        glBindBuffer(GL_ARRAY_BUFFER, point_vertex_buffer_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, sizeof(PointBodyVertex),
            reinterpret_cast<const void*>(offsetof(PointBodyVertex, position)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, sizeof(PointBodyVertex),
            reinterpret_cast<const void*>(offsetof(PointBodyVertex, color)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(
            2, 1, GL_FLOAT, GL_FALSE, sizeof(PointBodyVertex),
            reinterpret_cast<const void*>(offsetof(PointBodyVertex, size_pixels)));
        glBindVertexArray(0);
        point_view_location_ = uniform_location(point_program_, "view");
        point_projection_location_ = uniform_location(point_program_, "projection");

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glCullFace(GL_BACK);
        glEnable(GL_PROGRAM_POINT_SIZE);
    } catch (...) {
        release();
        throw;
    }
}

BodyRenderer::~BodyRenderer() { release(); }

void BodyRenderer::release() noexcept {
    if (point_vertex_buffer_ != 0) {
        glDeleteBuffers(1, &point_vertex_buffer_);
        point_vertex_buffer_ = 0;
    }
    if (point_vertex_array_ != 0) {
        glDeleteVertexArrays(1, &point_vertex_array_);
        point_vertex_array_ = 0;
    }
    if (point_program_ != 0) {
        glDeleteProgram(point_program_);
        point_program_ = 0;
    }
    if (star_vertex_buffer_ != 0) {
        glDeleteBuffers(1, &star_vertex_buffer_);
        star_vertex_buffer_ = 0;
    }
    if (star_vertex_array_ != 0) {
        glDeleteVertexArrays(1, &star_vertex_array_);
        star_vertex_array_ = 0;
    }
    if (star_program_ != 0) {
        glDeleteProgram(star_program_);
        star_program_ = 0;
    }
    if (ring_vertex_buffer_ != 0) {
        glDeleteBuffers(1, &ring_vertex_buffer_);
        ring_vertex_buffer_ = 0;
    }
    if (ring_vertex_array_ != 0) {
        glDeleteVertexArrays(1, &ring_vertex_array_);
        ring_vertex_array_ = 0;
    }
    if (ring_program_ != 0) {
        glDeleteProgram(ring_program_);
        ring_program_ = 0;
    }
    if (line_vertex_buffer_ != 0) {
        glDeleteBuffers(1, &line_vertex_buffer_);
        line_vertex_buffer_ = 0;
    }
    if (line_vertex_array_ != 0) {
        glDeleteVertexArrays(1, &line_vertex_array_);
        line_vertex_array_ = 0;
    }
    if (line_program_ != 0) {
        glDeleteProgram(line_program_);
        line_program_ = 0;
    }
    if (index_buffer_ != 0) {
        glDeleteBuffers(1, &index_buffer_);
        index_buffer_ = 0;
    }
    if (vertex_buffer_ != 0) {
        glDeleteBuffers(1, &vertex_buffer_);
        vertex_buffer_ = 0;
    }
    if (vertex_array_ != 0) {
        glDeleteVertexArrays(1, &vertex_array_);
        vertex_array_ = 0;
    }
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
}

void BodyRenderer::render(
    const std::vector<RenderBody>& bodies,
    const std::vector<RenderTrail>& trails,
    const OrbitCamera& camera,
    int framebuffer_width,
    int framebuffer_height,
    int scene_viewport_x,
    int scene_viewport_width,
    std::optional<std::size_t> selected_body,
    std::optional<RenderMarker> barycenter_marker) const {
    if (!render_resource_admitted(bodies.size())) {
        throw std::length_error(
            "renderer body count exceeds the bounded 8192-body viewer admission");
    }
    if (framebuffer_width <= 0 || framebuffer_height <= 0) {
        return;
    }
    scene_viewport_x = std::clamp(scene_viewport_x, 0, framebuffer_width - 1);
    scene_viewport_width = std::clamp(
        scene_viewport_width, 1, framebuffer_width - scene_viewport_x);
    glViewport(scene_viewport_x, 0, scene_viewport_width, framebuffer_height);
    glClearColor(0.006F, 0.009F, 0.022F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    const glm::mat4 view = camera.view_matrix();
    const glm::mat4 projection = camera.projection_matrix();

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glUseProgram(star_program_);
    const glm::mat4 star_view{glm::mat3(view)};
    glUniformMatrix4fv(star_view_location_, 1, GL_FALSE, glm::value_ptr(star_view));
    glUniformMatrix4fv(star_projection_location_, 1, GL_FALSE, glm::value_ptr(projection));
    glBindVertexArray(star_vertex_array_);
    glDrawArrays(GL_POINTS, 0, star_count_);
    glEnable(GL_DEPTH_TEST);

    glUseProgram(line_program_);
    glUniformMatrix4fv(line_view_location_, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(line_projection_location_, 1, GL_FALSE, glm::value_ptr(projection));
    glBindVertexArray(line_vertex_array_);
    glLineWidth(1.25F);
    for (const auto& trail : trails) {
        if (trail.points.size() < 2) {
            continue;
        }
        glBindBuffer(GL_ARRAY_BUFFER, line_vertex_buffer_);
        checked_buffer_data(
            GL_ARRAY_BUFFER, trail.points.size(), sizeof(glm::vec3),
            trail.points.data(), GL_DYNAMIC_DRAW, "trail vertex-buffer upload");
        glUniform3fv(line_color_location_, 1, glm::value_ptr(trail.color));
        glUniform1f(line_alpha_location_, trail.alpha);
        glUniform1i(line_fade_location_, trail.fade_with_age ? 1 : 0);
        glUniform1i(line_count_location_, static_cast<int>(trail.points.size()));
        glUniform1i(line_style_location_,
            trail.line_style == RenderLineStyle::dashed ? 1 : 0);
        glDrawArrays(GL_LINE_STRIP, 0, static_cast<int>(trail.points.size()));
    }

    if (selected_body && *selected_body < bodies.size() && bodies[*selected_body].visible) {
        constexpr int marker_segments = 64;
        constexpr float pi = 3.14159265358979323846F;
        const auto& body = bodies[*selected_body];
        const glm::mat4 inverse_view = glm::inverse(view);
        const glm::vec3 right = glm::normalize(glm::vec3(inverse_view[0]));
        const glm::vec3 up = glm::normalize(glm::vec3(inverse_view[1]));
        std::vector<glm::vec3> marker;
        marker.reserve(marker_segments + 1);
        const float radius = std::max(body.radius * 1.65F, 0.075F);
        for (int segment = 0; segment <= marker_segments; ++segment) {
            const float angle = 2.0F * pi * static_cast<float>(segment) /
                static_cast<float>(marker_segments);
            marker.push_back(body.position + radius *
                (std::cos(angle) * right + std::sin(angle) * up));
        }
        glBindBuffer(GL_ARRAY_BUFFER, line_vertex_buffer_);
        checked_buffer_data(
            GL_ARRAY_BUFFER, marker.size(), sizeof(glm::vec3), marker.data(),
            GL_DYNAMIC_DRAW, "selection marker vertex-buffer upload");
        const glm::vec3 marker_color(1.0F, 0.80F, 0.28F);
        glUniform3fv(line_color_location_, 1, glm::value_ptr(marker_color));
        glUniform1f(line_alpha_location_, 0.92F);
        glUniform1i(line_fade_location_, 0);
        glUniform1i(line_count_location_, static_cast<int>(marker.size()));
        glUniform1i(line_style_location_, 0);
        glDrawArrays(GL_LINE_STRIP, 0, static_cast<int>(marker.size()));
    }
    if (barycenter_marker) {
        const auto& marker = *barycenter_marker;
        const std::vector<glm::vec3> diamond{
            marker.position + glm::vec3(marker.radius, 0.0F, 0.0F),
            marker.position + glm::vec3(0.0F, marker.radius, 0.0F),
            marker.position - glm::vec3(marker.radius, 0.0F, 0.0F),
            marker.position - glm::vec3(0.0F, marker.radius, 0.0F),
        };
        glBindBuffer(GL_ARRAY_BUFFER, line_vertex_buffer_);
        checked_buffer_data(
            GL_ARRAY_BUFFER, diamond.size(), sizeof(glm::vec3), diamond.data(),
            GL_DYNAMIC_DRAW, "barycenter marker vertex-buffer upload");
        glUniform3fv(line_color_location_, 1, glm::value_ptr(marker.color));
        glUniform1f(line_alpha_location_, 0.90F);
        glUniform1i(line_fade_location_, 0);
        glUniform1i(line_count_location_, static_cast<int>(diamond.size()));
        glUniform1i(line_style_location_, 0);
        glDrawArrays(GL_LINE_LOOP, 0, static_cast<int>(diamond.size()));
    }
    glDepthMask(GL_TRUE);

    const bool use_point_bodies = bodies.size() > 64 &&
        std::all_of(bodies.begin(), bodies.end(), [](const auto& body) {
            return body.material == BodyMaterial::generic;
    });
    if (use_point_bodies) {
        static thread_local std::vector<PointBodyVertex> points;
        points.clear();
        if (points.capacity() < bodies.size()) {
            points.reserve(bodies.size());
        }
        for (const auto& body : bodies) {
            if (!body.visible) {
                continue;
            }
            points.push_back({
                body.position,
                body.color,
                std::clamp(4.0F + body.radius * 24.0F, 4.0F, 11.0F),
            });
        }
        glUseProgram(point_program_);
        glUniformMatrix4fv(point_view_location_, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(point_projection_location_, 1, GL_FALSE, glm::value_ptr(projection));
        glBindVertexArray(point_vertex_array_);
        glBindBuffer(GL_ARRAY_BUFFER, point_vertex_buffer_);
        checked_buffer_data(
            GL_ARRAY_BUFFER, points.size(), sizeof(PointBodyVertex), points.data(),
            GL_DYNAMIC_DRAW, "synthetic point vertex-buffer upload");
        glDrawArrays(GL_POINTS, 0, static_cast<int>(points.size()));
    } else {
        glUseProgram(program_);
        glUniformMatrix4fv(view_location_, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projection_location_, 1, GL_FALSE, glm::value_ptr(projection));
        glm::vec3 sun_position(0.0F);
        for (const auto& body : bodies) {
            if (body.emissive) {
                sun_position = body.position;
                break;
            }
        }
        glUniform3fv(sun_position_location_, 1, glm::value_ptr(sun_position));
        glBindVertexArray(vertex_array_);
        for (const auto& body : bodies) {
            if (!body.visible) {
                continue;
            }
            glm::mat4 model(1.0F);
            model = glm::translate(model, body.position);
            model = glm::scale(model, glm::vec3(body.radius));
            glUniformMatrix4fv(model_location_, 1, GL_FALSE, glm::value_ptr(model));
            glUniform3fv(color_location_, 1, glm::value_ptr(body.color));
            glUniform1i(emissive_location_, body.emissive ? 1 : 0);
            glUniform1i(material_location_, static_cast<int>(body.material));
            glDrawElements(GL_TRIANGLES, index_count_, GL_UNSIGNED_INT, nullptr);
        }
    }

    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glUseProgram(ring_program_);
    glUniformMatrix4fv(ring_view_location_, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(ring_projection_location_, 1, GL_FALSE, glm::value_ptr(projection));
    glBindVertexArray(ring_vertex_array_);
    for (const auto& body : bodies) {
        if (!body.visible || body.material != BodyMaterial::saturn) {
            continue;
        }
        glm::mat4 model(1.0F);
        model = glm::translate(model, body.position);
        model = glm::rotate(model, glm::radians(26.7F), glm::vec3(0.0F, 0.0F, 1.0F));
        model = glm::scale(model, glm::vec3(body.radius));
        glUniformMatrix4fv(ring_model_location_, 1, GL_FALSE, glm::value_ptr(model));
        glDrawArrays(GL_TRIANGLE_STRIP, 0, ring_vertex_count_);
    }
    glEnable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
}

}  // namespace nbody::visualization
