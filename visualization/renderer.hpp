#pragma once

#include "nbody/visualization/camera.hpp"
#include "nbody/visualization/render_space.hpp"

#include <glad/gl.h>

#include <vector>

namespace nbody::visualization {

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
        const OrbitCamera& camera,
        int framebuffer_width,
        int framebuffer_height) const;

private:
    void release() noexcept;

    unsigned int program_{0};
    unsigned int vertex_array_{0};
    unsigned int vertex_buffer_{0};
    unsigned int index_buffer_{0};
    int index_count_{0};
    int model_location_{-1};
    int view_location_{-1};
    int projection_location_{-1};
    int color_location_{-1};
};

}  // namespace nbody::visualization
