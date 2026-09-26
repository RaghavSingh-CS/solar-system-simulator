#include "renderer.hpp"

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <cstddef>
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

void main() {
    world_normal = mat3(transpose(inverse(model))) * in_normal;
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
)glsl";

constexpr const char* fragment_shader_source = R"glsl(
#version 330 core
in vec3 world_normal;

uniform vec3 body_color;

out vec4 fragment_color;

void main() {
    vec3 normal = normalize(world_normal);
    vec3 light_direction = normalize(vec3(0.45, 0.75, 0.35));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float rim = pow(1.0 - max(normal.z, 0.0), 2.0) * 0.12;
    vec3 color = body_color * (0.22 + 0.78 * diffuse) + rim * body_color;
    fragment_color = vec4(color, 1.0);
}
)glsl";

struct Vertex {
    float position[3];
    float normal[3];
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

int uniform_location(unsigned int program, const char* name) {
    const int location = glGetUniformLocation(program, name);
    if (location < 0) {
        throw std::runtime_error(std::string("required shader uniform is unavailable: ") + name);
    }
    return location;
}

}  // namespace

BodyRenderer::BodyRenderer() : program_(create_program()) {
    try {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        make_sphere(vertices, indices);
        index_count_ = static_cast<int>(indices.size());

        glGenVertexArrays(1, &vertex_array_);
        glGenBuffers(1, &vertex_buffer_);
        glGenBuffers(1, &index_buffer_);
        glBindVertexArray(vertex_array_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
        glBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)),
            vertices.data(),
            GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned int)),
            indices.data(),
            GL_STATIC_DRAW);
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

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
    } catch (...) {
        release();
        throw;
    }
}

BodyRenderer::~BodyRenderer() { release(); }

void BodyRenderer::release() noexcept {
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
    const OrbitCamera& camera,
    int framebuffer_width,
    int framebuffer_height) const {
    if (framebuffer_width <= 0 || framebuffer_height <= 0) {
        return;
    }
    glViewport(0, 0, framebuffer_width, framebuffer_height);
    glClearColor(0.006F, 0.009F, 0.022F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(program_);
    const glm::mat4 view = camera.view_matrix();
    const glm::mat4 projection = camera.projection_matrix();
    glUniformMatrix4fv(view_location_, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projection_location_, 1, GL_FALSE, glm::value_ptr(projection));
    glBindVertexArray(vertex_array_);
    for (const auto& body : bodies) {
        glm::mat4 model(1.0F);
        model = glm::translate(model, body.position);
        model = glm::scale(model, glm::vec3(body.radius));
        glUniformMatrix4fv(model_location_, 1, GL_FALSE, glm::value_ptr(model));
        glUniform3fv(color_location_, 1, glm::value_ptr(body.color));
        glDrawElements(GL_TRIANGLES, index_count_, GL_UNSIGNED_INT, nullptr);
    }
    glBindVertexArray(0);
}

}  // namespace nbody::visualization
