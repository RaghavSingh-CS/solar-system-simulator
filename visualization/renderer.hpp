#pragma once

#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/render_space.hpp"

#include <glad/gl.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace nbody::visualization {

struct RenderTrail {
    std::vector<glm::vec3> points;
    glm::vec3 color{1.0F};
    float alpha{0.58F};
    bool fade_with_age{true};
};

class BodyRenderer {
public:
    BodyRenderer();
    ~BodyRenderer();
    BodyRenderer(const BodyRenderer&) = delete;
    BodyRenderer& operator=(const BodyRenderer&) = delete;
    BodyRenderer(BodyRenderer&&) = delete;
    BodyRenderer& operator=(BodyRenderer&&) = delete;

    void render(
        const std::vector<RenderBody>& bodies,
        const std::vector<RenderTrail>& trails,
        const OrbitCamera& camera,
        int framebuffer_width,
        int framebuffer_height,
        std::optional<std::size_t> selected_body = std::nullopt) const;

private:
    void release() noexcept;

    unsigned int program_{0};
    unsigned int vertex_array_{0};
    unsigned int vertex_buffer_{0};
    unsigned int index_buffer_{0};
    unsigned int line_program_{0};
    unsigned int line_vertex_array_{0};
    unsigned int line_vertex_buffer_{0};
    unsigned int ring_program_{0};
    unsigned int ring_vertex_array_{0};
    unsigned int ring_vertex_buffer_{0};
    unsigned int star_program_{0};
    unsigned int star_vertex_array_{0};
    unsigned int star_vertex_buffer_{0};
    int index_count_{0};
    int ring_vertex_count_{0};
    int star_count_{0};
    int model_location_{-1};
    int view_location_{-1};
    int projection_location_{-1};
    int color_location_{-1};
    int sun_position_location_{-1};
    int emissive_location_{-1};
    int material_location_{-1};
    int line_view_location_{-1};
    int line_projection_location_{-1};
    int line_color_location_{-1};
    int line_alpha_location_{-1};
    int line_fade_location_{-1};
    int line_count_location_{-1};
    int ring_model_location_{-1};
    int ring_view_location_{-1};
    int ring_projection_location_{-1};
    int star_view_location_{-1};
    int star_projection_location_{-1};
};

}  // namespace nbody::visualization
