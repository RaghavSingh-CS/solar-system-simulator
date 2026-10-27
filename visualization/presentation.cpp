#include "nbody/visualization/presentation.hpp"

#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nbody::visualization {

std::size_t responsive_viewer_batch_steps(bool cuda, std::size_t requested_steps) {
    if (requested_steps == 0U) throw std::invalid_argument("viewer batch must be positive");
    return cuda ? std::min<std::size_t>(requested_steps, 4U) : requested_steps;
}

std::vector<std::vector<Vec3>> catalog_osculating_orbit_guides(
    const ScenarioDefinition& scenario) {
    std::vector<std::vector<Vec3>> paths(scenario.stable_ids.size());
    if (scenario.kind != ScenarioKind::astronomical_catalog) return paths;
    constexpr std::size_t segments = 192U;
    for (std::size_t index = 0; index < paths.size(); ++index) {
        const auto& parent_id = scenario.body_metadata[index].parent_stable_id;
        if (!parent_id) continue;
        const auto parent = find_body_index(scenario, *parent_id);
        if (!parent) continue;
        const auto& body = scenario.initial_state.bodies[index];
        const auto& primary = scenario.initial_state.bodies[*parent];
        const auto r = body.position_m - primary.position_m;
        const auto v = body.velocity_m_s - primary.velocity_m_s;
        const double radius = norm(r);
        const double mu = gravitational_constant * (body.mass_kg + primary.mass_kg);
        const auto h = cross(r, v);
        const double h_norm = norm(h);
        if (!(radius > 0.0) || !(mu > 0.0) || !(h_norm > 0.0)) continue;
        const auto eccentricity_vector = cross(v, h) / mu - r / radius;
        const double eccentricity = norm(eccentricity_vector);
        const double p = h_norm * (h_norm / mu);
        if (!std::isfinite(p) || !(p > 0.0) || !std::isfinite(eccentricity) ||
            eccentricity >= 0.999) continue;
        const auto x_axis = eccentricity > 1.0e-10
            ? eccentricity_vector / eccentricity : r / radius;
        const auto y_axis = cross(h / h_norm, x_axis);
        auto& path = paths[index];
        path.reserve(segments + 1U);
        for (std::size_t sample = 0; sample <= segments; ++sample) {
            const double angle = 6.2831853071795864769 *
                static_cast<double>(sample) / static_cast<double>(segments);
            const double distance = p / (1.0 + eccentricity * std::cos(angle));
            path.push_back(primary.position_m +
                (x_axis * std::cos(angle) + y_axis * std::sin(angle)) * distance);
        }
    }
    return paths;
}

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
    const ScenarioDefinition& scenario,
    const std::vector<RenderBody>& bodies) {
    if (bodies.empty()) {
        throw std::invalid_argument("camera framing requires at least one render body");
    }
    const bool catalog = scenario.kind == ScenarioKind::astronomical_catalog;
    const auto id = [&](const char* legacy_id, const char* catalog_id) {
        return catalog ? std::string(catalog_id) : std::string(legacy_id);
    };
    const auto index_of = [&](const std::string& stable_id) {
        const auto index = find_body_index(scenario, stable_id);
        if (!index || *index >= bodies.size()) {
            throw std::invalid_argument(
                "camera preset body is unavailable in this scenario: " + stable_id);
        }
        return *index;
    };
    const auto available = [&](std::initializer_list<std::string> stable_ids) {
        std::vector<std::size_t> result;
        for (const auto& stable_id : stable_ids) {
            if (const auto index = find_body_index(scenario, stable_id)) {
                result.push_back(*index);
            }
        }
        return result;
    };
    const auto local = [&](const std::string& center_id,
                           std::vector<std::size_t> visible,
                           float yaw, float pitch, float distance,
                           bool parent_relative = true) {
        const auto center = index_of(center_id);
        return CameraFraming{
            yaw, pitch, distance, bodies[center].position, center, center,
            std::move(visible), parent_relative,
        };
    };
    if (scenario.kind == ScenarioKind::synthetic && preset != CameraPreset::overview) {
        throw std::invalid_argument("planet camera presets require the Solar System scenario");
    }
    switch (preset) {
        case CameraPreset::overview:
            return {35.0F, 28.0F, 22.0F, glm::vec3(0.0F), std::nullopt,
                    std::nullopt, {}, false};
        case CameraPreset::inner_system:
            return local(id("sun", "sol:sun"), available({
                id("sun", "sol:sun"), id("mercury", "sol:mercury"),
                id("venus", "sol:venus"), id("earth", "sol:earth"),
                id("mars", "sol:mars")}), -30.0F, 58.0F, 20.0F);
        case CameraPreset::earth_moon:
            return local(id("earth", "sol:earth"), available({
                id("earth", "sol:earth"), id("sun", "sol:moon")}),
                38.0F, 22.0F, 18.0F);
        case CameraPreset::jupiter_system:
            return local(id("jupiter", "sol:jupiter"), available({
                id("jupiter", "sol:jupiter"), id("sun", "sol:io"), "sol:europa",
                "sol:ganymede", "sol:callisto"}), 35.0F, 20.0F, 18.0F);
        case CameraPreset::saturn_system:
            return local(id("saturn", "sol:saturn"), available({
                id("saturn", "sol:saturn"), id("sun", "sol:titan"), "sol:rhea", "sol:iapetus",
                "sol:dione", "sol:tethys", "sol:enceladus", "sol:mimas", "sol:hyperion",
                "sol:phoebe", "sol:janus", "sol:epimetheus", "sol:helene", "sol:atlas",
                "sol:prometheus", "sol:pandora"}), 30.0F, 25.0F, 18.0F);
        case CameraPreset::uranus_system:
            return local(id("uranus", "sol:uranus"), available({
                id("uranus", "sol:uranus"), "sol:titania", "sol:oberon", "sol:ariel",
                "sol:umbriel", "sol:miranda"}), 32.0F, 24.0F, 18.0F);
        case CameraPreset::neptune_triton:
            return local(id("neptune", "sol:neptune"), available({
                id("neptune", "sol:neptune"), "sol:triton", "sol:proteus", "sol:thalassa"}),
                34.0F, 23.0F, 18.0F);
        case CameraPreset::pluto_charon:
            return local("sol:pluto", available({"sol:pluto", "sol:charon", "sol:nix", "sol:hydra"}),
                38.0F, 20.0F, 18.0F);
        case CameraPreset::asteroid_belt:
            return local(id("sun", "sol:sun"), available({
                id("sun", "sol:sun"), "sol:ceres", "sol:vesta", "sol:pallas", "sol:hygiea",
                "sol:juno", "sol:psyche", "sol:davida", "sol:interamnia", "sol:eunomia"}),
                -26.0F, 62.0F, 20.0F);
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
        "Choose the unchanged nine-body baseline, the sourced 60-body catalog sandbox, or a deterministic synthetic scaling workload.",
        "Pause physical simulation while leaving camera and inspection controls active.",
        "Restore the current scenario to its deterministic initial state.",
        "Controls simulated time advanced per rendered frame; the integration timestep is unchanged.",
        "Compute gravitational forces using one CPU thread.",
        "Compute gravitational forces in parallel using shared-memory CPU threads.",
        "Advance each fixed-step frame batch with the CUDA Velocity Verlet executor. No CPU fallback occurs.",
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
    std::optional<std::size_t> sun_index) {
    if (!(viewport_height_pixels > 0.0F) || !std::isfinite(viewport_height_pixels)) {
        throw std::invalid_argument("viewport height must be finite and positive");
    }
    const glm::vec3 camera_position = camera.position();
    std::vector<float> pixels(bodies.size());
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        const float distance = std::max(glm::length(bodies[index].position - camera_position), 0.001F);
        // The Sun's target apparent size is a presentation invariant. Use the
        // camera's explicit zoom distance rather than any orbiting body position,
        // then convert that target back at the Sun's actual depth below.
        const bool sun = sun_index && index == *sun_index;
        const float sizing_distance = sun
            ? camera.distance()
            : distance;
        const float current = apparent_radius_pixels(
            bodies[index].radius, sizing_distance, viewport_height_pixels);
        const float minimum = sun
            ? sun_minimum_apparent_radius_pixels
            : body_minimum_apparent_radius_pixels;
        const float maximum = sun
            ? sun_maximum_apparent_radius_pixels
            : body_maximum_apparent_radius_pixels;
        pixels[index] = std::clamp(current, minimum, maximum);
    }
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        const float distance = std::max(glm::length(bodies[index].position - camera_position), 0.001F);
        constexpr float half_vertical_fov_radians = 0.39269908169872414F;
        const float world_per_pixel =
            2.0F * distance * std::tan(half_vertical_fov_radians) / viewport_height_pixels;
        bodies[index].radius = pixels[index] * world_per_pixel;
    }
}

std::size_t body_class_visibility_index(
    astronomy::AstronomicalBodyClass body_class) noexcept {
    switch (body_class) {
        case astronomy::AstronomicalBodyClass::star: return 0;
        case astronomy::AstronomicalBodyClass::planet: return 1;
        case astronomy::AstronomicalBodyClass::dwarf_planet: return 2;
        case astronomy::AstronomicalBodyClass::natural_satellite: return 3;
        case astronomy::AstronomicalBodyClass::asteroid: return 4;
        case astronomy::AstronomicalBodyClass::comet: return 5;
        case astronomy::AstronomicalBodyClass::trans_neptunian_object: return 6;
    }
    return astronomical_body_class_count;
}

namespace {

void validate_visibility_policy(
    const ScenarioDefinition& scenario,
    const std::vector<bool>& profile_membership,
    const std::vector<bool>& individual_visibility) {
    const std::size_t body_count = scenario.initial_state.bodies.size();
    if (scenario.body_metadata.size() != body_count ||
        profile_membership.size() != body_count ||
        individual_visibility.size() != body_count) {
        throw std::invalid_argument(
            "visibility policy must contain one profile, individual, and metadata entry per body");
    }
}

}  // namespace

bool effective_body_visibility(
    const ScenarioDefinition& scenario,
    std::size_t body_index,
    const std::vector<bool>& profile_membership,
    const std::vector<bool>& individual_visibility,
    const BodyClassVisibility& class_visibility) {
    validate_visibility_policy(scenario, profile_membership, individual_visibility);
    if (body_index >= scenario.initial_state.bodies.size()) {
        throw std::out_of_range("visibility body index is invalid");
    }
    const std::size_t class_index = body_class_visibility_index(
        scenario.body_metadata[body_index].body_class);
    if (class_index >= class_visibility.size()) {
        throw std::invalid_argument("unknown astronomical body class");
    }
    return profile_membership[body_index] && individual_visibility[body_index] &&
        class_visibility[class_index];
}

void apply_effective_body_visibility(
    const ScenarioDefinition& scenario,
    const std::vector<bool>& profile_membership,
    const std::vector<bool>& individual_visibility,
    const BodyClassVisibility& class_visibility,
    std::vector<RenderBody>& bodies) {
    validate_visibility_policy(scenario, profile_membership, individual_visibility);
    if (bodies.size() != scenario.initial_state.bodies.size()) {
        throw std::invalid_argument("render snapshot body count differs from visibility policy");
    }
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        const std::size_t class_index = body_class_visibility_index(
            scenario.body_metadata[index].body_class);
        if (class_index >= class_visibility.size()) {
            throw std::invalid_argument("unknown astronomical body class");
        }
        bodies[index].visible = profile_membership[index] && individual_visibility[index] &&
            class_visibility[class_index];
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

int astronomical_label_priority(
    astronomy::AstronomicalBodyClass body_class,
    bool selected,
    std::size_t deterministic_order) noexcept {
    if (selected) {
        return 1000;
    }
    int class_priority = 500;
    switch (body_class) {
        case astronomy::AstronomicalBodyClass::star:
            class_priority = 950;
            break;
        case astronomy::AstronomicalBodyClass::planet:
            class_priority = 900;
            break;
        case astronomy::AstronomicalBodyClass::dwarf_planet:
            class_priority = 820;
            break;
        case astronomy::AstronomicalBodyClass::natural_satellite:
            class_priority = 740;
            break;
        case astronomy::AstronomicalBodyClass::asteroid:
            class_priority = 640;
            break;
        case astronomy::AstronomicalBodyClass::comet:
            class_priority = 620;
            break;
        case astronomy::AstronomicalBodyClass::trans_neptunian_object:
            class_priority = 600;
            break;
    }
    return class_priority - static_cast<int>(std::min<std::size_t>(
        deterministic_order, 99U));
}

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
