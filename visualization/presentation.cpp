#include "nbody/visualization/presentation.hpp"

#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nbody::visualization {

NumericalHealth classify_numerical_health(
    double energy_relative,
    double angular_momentum_relative) noexcept {
    return {
        std::isfinite(energy_relative) && energy_relative <= energy_drift_bound,
        std::isfinite(angular_momentum_relative) &&
            angular_momentum_relative <= angular_momentum_drift_bound,
    };
}

CameraFraming camera_framing(
    CameraPreset preset,
    ScenarioKind scenario,
    const std::vector<RenderBody>& bodies) {
    if (bodies.empty()) {
        throw std::invalid_argument("camera framing requires at least one render body");
    }
    const auto body_target = [&](std::size_t index) {
        if (index >= bodies.size()) {
            throw std::invalid_argument("camera preset body is unavailable in this scenario");
        }
        return bodies[index].position;
    };
    if (scenario != ScenarioKind::solar_system && preset != CameraPreset::overview) {
        throw std::invalid_argument("planet camera presets require the Solar System scenario");
    }
    switch (preset) {
        case CameraPreset::overview:
            return {35.0F, 28.0F, 22.0F, glm::vec3(0.0F), std::nullopt};
        case CameraPreset::inner_system:
            return {-30.0F, 58.0F, 2.65F, body_target(0), 0U};
        case CameraPreset::earth:
            return {38.0F, 22.0F, 2.15F, body_target(3), 3U};
        case CameraPreset::jupiter:
            return {35.0F, 20.0F, 2.35F, body_target(5), 5U};
        case CameraPreset::saturn:
            return {30.0F, 25.0F, 2.55F, body_target(6), 6U};
    }
    throw std::invalid_argument("unknown camera preset");
}

void apply_camera_framing(OrbitCamera& camera, const CameraFraming& framing) {
    camera.set_pose(
        framing.yaw_degrees,
        framing.pitch_degrees,
        framing.distance,
        framing.target);
}

glm::vec3 amplify_from_barycenter(
    const glm::vec3& render_position,
    const glm::vec3& barycenter,
    float amplification) {
    if (!(amplification >= 1.0F) || !std::isfinite(amplification) ||
        !std::isfinite(render_position.x) || !std::isfinite(render_position.y) ||
        !std::isfinite(render_position.z) || !std::isfinite(barycenter.x) ||
        !std::isfinite(barycenter.y) || !std::isfinite(barycenter.z)) {
        throw std::invalid_argument("barycentric visual amplification must be finite and at least one");
    }
    return barycenter + (render_position - barycenter) * amplification;
}

PanelWidthPolicy panel_width_policy(float display_width, float requested_width) {
    if (!(display_width > 0.0F) || !std::isfinite(display_width) ||
        !std::isfinite(requested_width)) {
        throw std::invalid_argument("panel width inputs must be finite and positive");
    }
    PanelWidthPolicy policy;
    policy.maximum = std::max(policy.minimum, std::min(680.0F, display_width * 0.55F));
    policy.width = std::clamp(requested_width, policy.minimum, policy.maximum);
    return policy;
}

std::string format_measurement(double value, const char* unit, int significant_digits) {
    if (!std::isfinite(value) || unit == nullptr || significant_digits < 1 ||
        significant_digits > 12) {
        throw std::invalid_argument("measurement formatting input is invalid");
    }
    std::ostringstream text;
    const double magnitude = std::abs(value);
    if (magnitude == 0.0 || (magnitude >= 1.0e-3 && magnitude < 1.0e6)) {
        const int decimal_places = magnitude == 0.0
            ? significant_digits - 1
            : std::max(0, significant_digits - 1 -
                static_cast<int>(std::floor(std::log10(magnitude))));
        text << std::fixed << std::setprecision(std::min(decimal_places, 9)) << value;
    } else {
        text << std::scientific << std::setprecision(significant_digits - 1) << value;
    }
    text << ' ' << unit;
    return text.str();
}

const char* help_text(HelpTopic topic) noexcept {
    constexpr std::array<const char*, static_cast<std::size_t>(HelpTopic::count)> help{{
        "Choose the physical Solar System or a deterministic synthetic scaling workload.",
        "Pause physical simulation while leaving camera and inspection controls active.",
        "Restore the current scenario to its deterministic initial state.",
        "Controls simulated time advanced per rendered frame; the integration timestep is unchanged.",
        "Compute gravitational forces using one CPU thread.",
        "Compute gravitational forces in parallel using shared-memory CPU threads.",
        "Number of CPU threads requested by the OpenMP force backend.",
        "Choose a deterministic direct N-body workload. Very large sizes are display-only in the viewer.",
        "Time isolated Sequential and OpenMP copies and validate their final states without advancing the view.",
        "Choose the body whose physical state and selection marker are shown.",
        "Center the camera on the selected body's current simulated position.",
        "Apply a deterministic camera composition without changing body positions.",
        "Show bounded paths sampled from integrated body positions.",
        "Show static approximate orbital context; these guides do not drive the simulation.",
        "Show stable body callouts. Synthetic workloads label only the selected body.",
        "Mark the barycentric origin and show the Sun's actual integrated motion trail.",
        "Magnify only the rendered Sun displacement; 1x is the actual integrated displacement.",
        "Adjust presentation-only body emphasis. Centers and orbital distances remain unchanged.",
        "Relative drift in total system energy; lower is better and is compared with the predefined bound.",
        "Relative drift in the angular-momentum vector; lower is better and is compared with the predefined bound.",
        "Summarizes whether both predefined conservation bounds are currently satisfied.",
        "Set the maximum number of stored samples per enabled history trail.",
        "Reveal provenance, coordinates, fixed timestep, and other technical details.",
        "Advance exactly one fixed integration step while the large workload remains paused.",
    }};
    const auto index = static_cast<std::size_t>(topic);
    return index < help.size() ? help[index] : "";
}

float apparent_radius_pixels(
    float world_radius,
    float distance_to_camera,
    float viewport_height_pixels) {
    if (!(world_radius > 0.0F) || !(distance_to_camera > 0.0F) ||
        !(viewport_height_pixels > 0.0F) || !std::isfinite(world_radius) ||
        !std::isfinite(distance_to_camera) || !std::isfinite(viewport_height_pixels)) {
        throw std::invalid_argument("screen-space radius inputs must be finite and positive");
    }
    constexpr float half_vertical_fov_radians = 0.39269908169872414F;
    const float world_height = 2.0F * distance_to_camera * std::tan(half_vertical_fov_radians);
    return world_radius * viewport_height_pixels / world_height;
}

void apply_screen_space_body_sizing(
    std::vector<RenderBody>& bodies,
    const OrbitCamera& camera,
    float viewport_height_pixels,
    bool solar_system,
    std::optional<float> sun_mercury_screen_separation_pixels) {
    if (!(viewport_height_pixels > 0.0F) || !std::isfinite(viewport_height_pixels)) {
        throw std::invalid_argument("viewport height must be finite and positive");
    }
    const glm::vec3 camera_position = camera.position();
    std::vector<float> pixels(bodies.size());
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        const float distance = std::max(glm::length(bodies[index].position - camera_position), 0.001F);
        const float current = apparent_radius_pixels(bodies[index].radius, distance, viewport_height_pixels);
        const float minimum = solar_system && index == 0
            ? sun_minimum_apparent_radius_pixels
            : body_minimum_apparent_radius_pixels;
        const float maximum = solar_system && index == 0
            ? sun_maximum_apparent_radius_pixels
            : body_maximum_apparent_radius_pixels;
        pixels[index] = std::clamp(current, minimum, maximum);
    }
    if (solar_system && bodies.size() > 1 && sun_mercury_screen_separation_pixels) {
        const float available = std::max(
            0.5F,
            *sun_mercury_screen_separation_pixels - sun_mercury_clearance_pixels);
        if (pixels[0] + pixels[1] > available) {
            // Preserve the readable Sun target for as long as the real projected
            // separation permits by continuously reducing Mercury first. Once
            // Mercury reaches a sub-pixel marker, clearance becomes the Sun's
            // upper bound. Min/clamp operations keep zoom response continuous.
            pixels[1] = std::min(pixels[1], std::max(0.25F, available - pixels[0]));
            pixels[0] = std::min(pixels[0], std::max(0.25F, available - pixels[1]));
        }
    }
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        const float distance = std::max(glm::length(bodies[index].position - camera_position), 0.001F);
        constexpr float half_vertical_fov_radians = 0.39269908169872414F;
        const float world_per_pixel =
            2.0F * distance * std::tan(half_vertical_fov_radians) / viewport_height_pixels;
        bodies[index].radius = pixels[index] * world_per_pixel;
    }
}

namespace {

bool overlaps(const LabelPlacement& left, const LabelPlacement& right, float margin) {
    return left.position.x < right.position.x + right.size.x + margin &&
        left.position.x + left.size.x + margin > right.position.x &&
        left.position.y < right.position.y + right.size.y + margin &&
        left.position.y + left.size.y + margin > right.position.y;
}

}  // namespace

std::vector<LabelPlacement> declutter_labels(
    std::vector<LabelCandidate> candidates,
    glm::vec2 viewport_size,
    float left_inset,
    float margin,
    const std::vector<LabelPlacement>& previous,
    const std::vector<LabelPlacement>& reserved) {
    if (!(viewport_size.x > 0.0F) || !(viewport_size.y > 0.0F) ||
        left_inset < 0.0F || margin < 0.0F) {
        throw std::invalid_argument("label viewport and margins must be valid");
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.priority > right.priority;
    });
    constexpr std::array<glm::vec2, 10> offsets{{
        {12.0F, -18.0F}, {12.0F, 8.0F}, {-12.0F, -18.0F}, {-12.0F, 8.0F},
        {20.0F, -34.0F}, {20.0F, 24.0F}, {-20.0F, -34.0F}, {-20.0F, 24.0F},
        {34.0F, -10.0F}, {-34.0F, -10.0F},
    }};
    std::vector<LabelPlacement> placed = reserved;
    placed.reserve(reserved.size() + candidates.size());
    for (const auto& candidate : candidates) {
        const auto try_position = [&](glm::vec2 position, bool leader_line) {
            position.x = std::clamp(
                position.x,
                left_inset + margin,
                std::max(left_inset + margin, viewport_size.x - candidate.size.x - margin));
            position.y = std::clamp(
                position.y,
                margin,
                std::max(margin, viewport_size.y - candidate.size.y - margin));
            LabelPlacement proposal{
                candidate.body_index,
                candidate.anchor,
                position,
                candidate.size,
                candidate.selected,
                leader_line,
            };
            const bool collision = std::any_of(placed.begin(), placed.end(), [&](const auto& item) {
                return overlaps(proposal, item, margin);
            });
            if (!collision) {
                placed.push_back(proposal);
            }
            return !collision;
        };
        const auto prior = std::find_if(previous.begin(), previous.end(), [&](const auto& item) {
            return item.body_index == candidate.body_index;
        });
        if (prior != previous.end() &&
            try_position(candidate.anchor + prior->position - prior->anchor, prior->leader_line)) {
            continue;
        }
        for (const auto& offset : offsets) {
            glm::vec2 position = candidate.anchor + offset;
            if (offset.x < 0.0F) {
                position.x -= candidate.size.x;
            }
            if (try_position(
                    position,
                    std::abs(offset.x) > 15.0F || std::abs(offset.y) > 20.0F)) {
                break;
            }
        }
    }
    return placed;
}

std::vector<LabelPlacement> stable_inner_system_callouts(
    const std::vector<LabelCandidate>& candidates,
    glm::vec2 viewport_size,
    float left_inset,
    float margin) {
    if (!(viewport_size.x > 0.0F) || !(viewport_size.y > 0.0F) ||
        left_inset < 0.0F || margin < 0.0F) {
        throw std::invalid_argument("inner callout viewport and margins must be valid");
    }
    std::vector<LabelCandidate> inner;
    for (const auto& candidate : candidates) {
        if (candidate.body_index <= 4U) {
            inner.push_back(candidate);
        }
    }
    std::sort(inner.begin(), inner.end(), [](const auto& left, const auto& right) {
        return left.body_index < right.body_index;
    });
    float maximum_width = 0.0F;
    float maximum_height = 0.0F;
    for (const auto& candidate : inner) {
        maximum_width = std::max(maximum_width, candidate.size.x);
        maximum_height = std::max(maximum_height, candidate.size.y);
    }
    const float slot_height = maximum_height + margin;
    const float total_height = inner.empty() ? 0.0F :
        maximum_height + slot_height * static_cast<float>(inner.size() - 1U);
    const float start_y = std::clamp(
        viewport_size.y * 0.18F,
        margin,
        std::max(margin, viewport_size.y - total_height - margin));
    const float x = std::clamp(
        viewport_size.x - maximum_width - 22.0F,
        left_inset + margin,
        std::max(left_inset + margin, viewport_size.x - maximum_width - margin));
    std::vector<LabelPlacement> result;
    result.reserve(inner.size());
    for (std::size_t slot = 0; slot < inner.size(); ++slot) {
        const auto& candidate = inner[slot];
        result.push_back({
            candidate.body_index,
            candidate.anchor,
            {x, start_y + static_cast<float>(slot) * slot_height},
            candidate.size,
            candidate.selected,
            true,
        });
    }
    return result;
}

}  // namespace nbody::visualization
