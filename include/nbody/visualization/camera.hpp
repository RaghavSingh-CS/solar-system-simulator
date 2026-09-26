#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace nbody::visualization {

class OrbitCamera {
public:
    OrbitCamera(float yaw_degrees = 35.0F, float pitch_degrees = 25.0F, float distance = 20.0F);

    void set_aspect_ratio(float aspect_ratio);
    void orbit(float delta_x, float delta_y);
    void dolly(float amount);
    void set_distance(float distance);
    void set_pose(float yaw_degrees, float pitch_degrees, float distance, const glm::vec3& target);
    void approach_target(const glm::vec3& target, float fraction);
    void reset();

    float aspect_ratio() const noexcept;
    float distance() const noexcept;
    float yaw_degrees() const noexcept;
    float pitch_degrees() const noexcept;
    const glm::vec3& target() const noexcept;
    glm::vec3 position() const;
    glm::mat4 view_matrix() const;
    glm::mat4 projection_matrix() const;

private:
    float initial_yaw_degrees_;
    float initial_pitch_degrees_;
    float initial_distance_;
    float yaw_degrees_;
    float pitch_degrees_;
    float distance_;
    float aspect_ratio_{16.0F / 9.0F};
    glm::vec3 target_{0.0F};
};

}  // namespace nbody::visualization
