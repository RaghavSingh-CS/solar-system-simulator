#include "nbody/cuda_barnes_hut.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace nbody {
namespace {

constexpr int threads_per_block = 256;

struct DeviceVec3 {
    double x;
    double y;
    double z;
};

struct DeviceFailure {
    unsigned int claimed;
    unsigned int published;
    std::uint32_t code;
    std::uint32_t target_index;
    std::uint32_t node_index;
};

[[noreturn]] void throw_cuda_error(cudaError_t error, const char* operation) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorName(error) << " ("
            << cudaGetErrorString(error) << ')';
    throw std::runtime_error(message.str());
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw_cuda_error(error, operation);
    }
}

void check_launch(const char* kernel_name) {
    check_cuda(cudaGetLastError(), kernel_name);
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* context) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        throw std::length_error(std::string(context) + " size overflow");
    }
    return lhs + rhs;
}

std::size_t checked_multiply(
    std::size_t lhs,
    std::size_t rhs,
    const char* context) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::length_error(std::string(context) + " size overflow");
    }
    return lhs * rhs;
}

std::size_t vector_bytes(
    std::size_t count,
    std::size_t element_size,
    const char* context) {
    return checked_multiply(count, element_size, context);
}

struct StorageRequirements {
    std::size_t device_bytes{0};
    std::size_t pinned_host_bytes{0};
};

StorageRequirements storage_requirements(
    std::size_t body_count,
    std::size_t node_count) {
    StorageRequirements result;
    const auto charge = [&](std::size_t count, std::size_t size, const char* context) {
        const std::size_t bytes = vector_bytes(count, size, context);
        result.device_bytes = checked_add(result.device_bytes, bytes, context);
        result.pinned_host_bytes = checked_add(result.pinned_host_bytes, bytes, context);
    };
    charge(body_count, sizeof(double), "CUDA Barnes-Hut masses");
    charge(body_count, sizeof(DeviceVec3), "CUDA Barnes-Hut positions");
    charge(body_count, sizeof(DeviceVec3), "CUDA Barnes-Hut accelerations");
    charge(body_count, sizeof(std::uint32_t), "CUDA Barnes-Hut permutation");
    charge(body_count, sizeof(std::uint32_t), "CUDA Barnes-Hut inverse rank");
    charge(node_count, sizeof(BarnesHutPackedNode), "CUDA Barnes-Hut nodes");
    charge(1, sizeof(DeviceFailure), "CUDA Barnes-Hut failure report");
    charge(1, sizeof(unsigned long long), "CUDA Barnes-Hut work counter");
    return result;
}

void validate_configuration(const CudaBarnesHutConfiguration& configuration) {
    if (configuration.device_index < 0) {
        throw std::invalid_argument("CUDA Barnes-Hut device index must be nonnegative");
    }
    if (!(configuration.tree.theta > 0.0) ||
        configuration.tree.theta > 2.0 ||
        !std::isfinite(configuration.tree.theta)) {
        throw std::invalid_argument("CUDA Barnes-Hut theta must be finite and in (0, 2]");
    }
    if (configuration.tree.leaf_capacity == 0 ||
        configuration.tree.maximum_depth == 0 ||
        configuration.tree.maximum_depth > 128 ||
        configuration.tree.maximum_nodes == 0 ||
        configuration.tree.maximum_traversal_work == 0 ||
        configuration.tree.memory_budget_bytes == 0) {
        throw std::invalid_argument("CUDA Barnes-Hut tree limits must be positive and bounded");
    }
    if (configuration.tree.minimum_cell_size_m < 0.0 ||
        !std::isfinite(configuration.tree.minimum_cell_size_m)) {
        throw std::invalid_argument(
            "CUDA Barnes-Hut minimum cell size must be finite and nonnegative");
    }
    if (configuration.device_memory_budget_bytes == 0 ||
        configuration.host_staging_memory_budget_bytes == 0) {
        throw std::invalid_argument("CUDA Barnes-Hut memory budgets must be positive");
    }
    if (configuration.traversal_stack_capacity == 0 ||
        configuration.traversal_stack_capacity >
            cuda_barnes_hut_maximum_stack_capacity) {
        throw std::invalid_argument("CUDA Barnes-Hut traversal stack capacity is invalid");
    }
    if (configuration.host_failure_injection !=
            CudaBarnesHutHostFailureInjection::none &&
        configuration.host_failure_injection !=
            CudaBarnesHutHostFailureInjection::
                live_release_before_first_deallocation) {
        throw std::invalid_argument(
            "CUDA Barnes-Hut host failure injection is invalid");
    }
    if (configuration.tree.maximum_traversal_work >
        static_cast<std::size_t>(std::numeric_limits<unsigned long long>::max())) {
        throw std::length_error("CUDA Barnes-Hut traversal budget exceeds device counter range");
    }
}

void validate_force_inputs(
    const SimulationState& state,
    double gravitational_constant_value) {
    if (!(gravitational_constant_value > 0.0) ||
        !std::isfinite(gravitational_constant_value)) {
        throw std::invalid_argument("gravitational constant must be finite and positive");
    }
    if (state.bodies.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("body count exceeds CUDA kernel launch range");
    }
    for (const Body& body : state.bodies) {
        if (!(body.mass_kg > 0.0) || !std::isfinite(body.mass_kg)) {
            throw std::invalid_argument("body mass must be finite and positive");
        }
        if (!std::isfinite(body.position_m.x) ||
            !std::isfinite(body.position_m.y) ||
            !std::isfinite(body.position_m.z)) {
            throw std::invalid_argument("body position must be finite");
        }
    }
}

int block_count(std::size_t body_count) {
    return static_cast<int>(
        (body_count + static_cast<std::size_t>(threads_per_block) - 1U) /
        static_cast<std::size_t>(threads_per_block));
}

__device__ void record_failure(
    DeviceFailure* failure,
    CudaBarnesHutFailureCode code,
    std::uint32_t target_index,
    std::uint32_t node_index) {
    // Claim and publication are separate so no thread can observe a published
    // code before its metadata is complete. Both synchronization words are
    // accessed atomically by every device thread.
    if (atomicCAS(&failure->claimed, 0U, 1U) == 0U) {
        failure->code = static_cast<std::uint32_t>(code);
        failure->target_index = target_index;
        failure->node_index = node_index;
        __threadfence();
        atomicExch(&failure->published, 1U);
    }
}

__device__ bool failure_published(DeviceFailure* failure) {
    return atomicAdd(&failure->published, 0U) != 0U;
}

__device__ bool reserve_work(
    unsigned long long* work,
    unsigned long long limit,
    DeviceFailure* failure,
    std::uint32_t target_index,
    std::uint32_t node_index) {
    unsigned long long current = atomicAdd(work, 0ULL);
    while (current < limit) {
        const unsigned long long observed = atomicCAS(work, current, current + 1ULL);
        if (observed == current) {
            return true;
        }
        current = observed;
    }
    record_failure(
        failure,
        CudaBarnesHutFailureCode::traversal_work_exhausted,
        target_index,
        node_index);
    return false;
}

__device__ bool finite_vec(const DeviceVec3& value) {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__device__ double stable_norm(const DeviceVec3& value) {
    const double scale = fmax(fabs(value.x), fmax(fabs(value.y), fabs(value.z)));
    if (scale == 0.0) {
        return 0.0;
    }
    if (!isfinite(scale)) {
        return scale;
    }
    const double x = value.x / scale;
    const double y = value.y / scale;
    const double z = value.z / scale;
    return scale * sqrt(x * x + y * y + z * z);
}

__device__ bool scaled_acceleration_component(
    double displacement_component,
    double distance,
    double source_mass,
    double gravitational_constant_value,
    double* result) {
    if (displacement_component == 0.0) {
        *result = 0.0;
        return true;
    }
    int gravitational_exponent = 0;
    int mass_exponent = 0;
    int displacement_exponent = 0;
    int distance_exponent = 0;
    const double gravitational_fraction =
        frexp(gravitational_constant_value, &gravitational_exponent);
    const double mass_fraction = frexp(source_mass, &mass_exponent);
    const double displacement_fraction =
        frexp(fabs(displacement_component), &displacement_exponent);
    const double distance_fraction = frexp(distance, &distance_exponent);
    double fraction = gravitational_fraction * mass_fraction * displacement_fraction /
        (distance_fraction * distance_fraction * distance_fraction);
    int fraction_exponent = 0;
    fraction = frexp(fraction, &fraction_exponent);
    const long long exponent =
        static_cast<long long>(gravitational_exponent) + mass_exponent +
        displacement_exponent - 3LL * distance_exponent + fraction_exponent;
    if (exponent > DBL_MAX_EXP) {
        return false;
    }
    if (exponent < DBL_MIN_EXP - DBL_MANT_DIG) {
        *result = copysign(0.0, displacement_component);
        return true;
    }
    const double magnitude = ldexp(fraction, static_cast<int>(exponent));
    if (!isfinite(magnitude)) {
        return false;
    }
    *result = copysign(magnitude, displacement_component);
    return true;
}

__device__ bool acceleration_contribution(
    const DeviceVec3& displacement,
    double source_mass,
    double gravitational_constant_value,
    DeviceVec3* contribution,
    CudaBarnesHutFailureCode* failure_code) {
    if (!finite_vec(displacement)) {
        *failure_code = CudaBarnesHutFailureCode::nonrepresentable_geometry;
        return false;
    }
    const double distance = stable_norm(displacement);
    if (distance == 0.0) {
        *failure_code = CudaBarnesHutFailureCode::singular_geometry;
        return false;
    }
    if (!isfinite(distance)) {
        *failure_code = CudaBarnesHutFailureCode::nonrepresentable_geometry;
        return false;
    }

    const double ordinary_magnitude =
        (gravitational_constant_value * source_mass) / distance / distance;
    if (ordinary_magnitude > 0.0 && isfinite(ordinary_magnitude)) {
        const double scale = ordinary_magnitude / distance;
        const DeviceVec3 ordinary{
            displacement.x * scale,
            displacement.y * scale,
            displacement.z * scale,
        };
        const bool lost_component =
            (displacement.x != 0.0 && ordinary.x == 0.0) ||
            (displacement.y != 0.0 && ordinary.y == 0.0) ||
            (displacement.z != 0.0 && ordinary.z == 0.0);
        if (finite_vec(ordinary) && !lost_component) {
            *contribution = ordinary;
            return true;
        }
    }

    DeviceVec3 scaled{};
    if (!scaled_acceleration_component(
            displacement.x,
            distance,
            source_mass,
            gravitational_constant_value,
            &scaled.x) ||
        !scaled_acceleration_component(
            displacement.y,
            distance,
            source_mass,
            gravitational_constant_value,
            &scaled.y) ||
        !scaled_acceleration_component(
            displacement.z,
            distance,
            source_mass,
            gravitational_constant_value,
            &scaled.z) ||
        !finite_vec(scaled)) {
        *failure_code = CudaBarnesHutFailureCode::nonrepresentable_acceleration;
        return false;
    }
    *contribution = scaled;
    return true;
}

__device__ void compensated_add_component(
    double input,
    double* accumulated,
    double* compensation) {
    const double adjusted = input - *compensation;
    const double next = *accumulated + adjusted;
    *compensation = (next - *accumulated) - adjusted;
    *accumulated = next;
}

__device__ void compensated_add(
    const DeviceVec3& value,
    DeviceVec3* sum,
    DeviceVec3* correction) {
    compensated_add_component(value.x, &sum->x, &correction->x);
    compensated_add_component(value.y, &sum->y, &correction->y);
    compensated_add_component(value.z, &sum->z, &correction->z);
}

__global__ void barnes_hut_force_kernel(
    const double* masses,
    const DeviceVec3* positions,
    const BarnesHutPackedNode* nodes,
    std::uint32_t node_count,
    const std::uint32_t* permutation,
    const std::uint32_t* inverse_rank,
    std::uint32_t body_count,
    double theta,
    double gravitational_constant_value,
    std::uint32_t stack_capacity,
    unsigned long long traversal_limit,
    DeviceVec3* accelerations,
    unsigned long long* traversal_work,
    DeviceFailure* failure) {
    const std::uint32_t target =
        static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (target >= body_count || failure_published(failure)) {
        return;
    }
    if (node_count == 0U || inverse_rank[target] >= body_count) {
        record_failure(
            failure, CudaBarnesHutFailureCode::invalid_tree, target,
            barnes_hut_no_packed_index);
        return;
    }

    std::uint32_t stack[cuda_barnes_hut_maximum_stack_capacity];
    std::uint32_t stack_size = 1U;
    stack[0] = 0U;
    DeviceVec3 acceleration{};
    DeviceVec3 correction{};

    while (stack_size != 0U) {
        const std::uint32_t node_index = stack[--stack_size];
        if (failure_published(failure)) {
            return;
        }
        if (node_index >= node_count ||
            !reserve_work(
                traversal_work,
                traversal_limit,
                failure,
                target,
                node_index)) {
            if (node_index >= node_count) {
                record_failure(
                    failure,
                    CudaBarnesHutFailureCode::invalid_tree,
                    target,
                    node_index);
            }
            return;
        }

        const BarnesHutPackedNode node = nodes[node_index];
        if (node.begin >= node.end || node.end > body_count ||
            !(node.mass_kg > 0.0) || !isfinite(node.mass_kg) ||
            !(node.side_length_m >= 0.0) || !isfinite(node.side_length_m)) {
            record_failure(
                failure, CudaBarnesHutFailureCode::invalid_tree, target, node_index);
            return;
        }
        const std::uint32_t target_rank = inverse_rank[target];
        const bool contains_target =
            target_rank >= node.begin && target_rank < node.end;

        if (node.leaf != 0U) {
            for (std::uint32_t offset = node.begin; offset < node.end; ++offset) {
                const std::uint32_t source = permutation[offset];
                if (source >= body_count) {
                    record_failure(
                        failure,
                        CudaBarnesHutFailureCode::invalid_tree,
                        target,
                        node_index);
                    return;
                }
                if (source == target) {
                    continue;
                }
                if (!reserve_work(
                        traversal_work,
                        traversal_limit,
                        failure,
                        target,
                        node_index)) {
                    return;
                }
                const DeviceVec3 displacement{
                    positions[source].x - positions[target].x,
                    positions[source].y - positions[target].y,
                    positions[source].z - positions[target].z,
                };
                DeviceVec3 contribution{};
                CudaBarnesHutFailureCode code = CudaBarnesHutFailureCode::none;
                if (!acceleration_contribution(
                        displacement,
                        masses[source],
                        gravitational_constant_value,
                        &contribution,
                        &code)) {
                    record_failure(failure, code, target, node_index);
                    return;
                }
                compensated_add(contribution, &acceleration, &correction);
            }
            continue;
        }

        const DeviceVec3 displacement{
            node.center_of_mass_x_m - positions[target].x,
            node.center_of_mass_y_m - positions[target].y,
            node.center_of_mass_z_m - positions[target].z,
        };
        const double distance = stable_norm(displacement);
        if (!contains_target && distance > 0.0 && isfinite(distance) &&
            node.side_length_m / distance < theta) {
            DeviceVec3 contribution{};
            CudaBarnesHutFailureCode code = CudaBarnesHutFailureCode::none;
            if (!acceleration_contribution(
                    displacement,
                    node.mass_kg,
                    gravitational_constant_value,
                    &contribution,
                    &code)) {
                record_failure(failure, code, target, node_index);
                return;
            }
            compensated_add(contribution, &acceleration, &correction);
            continue;
        }

        bool pushed_child = false;
        for (int child_slot = 7; child_slot >= 0; --child_slot) {
            const std::uint32_t child = node.children[child_slot];
            if (child == barnes_hut_no_packed_index) {
                continue;
            }
            if (child >= node_count) {
                record_failure(
                    failure,
                    CudaBarnesHutFailureCode::invalid_tree,
                    target,
                    node_index);
                return;
            }
            if (stack_size >= stack_capacity ||
                stack_size >= cuda_barnes_hut_maximum_stack_capacity) {
                record_failure(
                    failure,
                    CudaBarnesHutFailureCode::traversal_stack_exhausted,
                    target,
                    node_index);
                return;
            }
            stack[stack_size++] = child;
            pushed_child = true;
        }
        if (!pushed_child) {
            record_failure(
                failure,
                CudaBarnesHutFailureCode::invalid_tree,
                target,
                node_index);
            return;
        }
    }

    if (!finite_vec(acceleration)) {
        record_failure(
            failure,
            CudaBarnesHutFailureCode::nonrepresentable_acceleration,
            target,
            barnes_hut_no_packed_index);
        return;
    }
    accelerations[target] = acceleration;
}

const char* failure_message(CudaBarnesHutFailureCode code) {
    switch (code) {
    case CudaBarnesHutFailureCode::none:
        return "no CUDA Barnes-Hut failure";
    case CudaBarnesHutFailureCode::invalid_tree:
        return "CUDA Barnes-Hut packed tree is invalid";
    case CudaBarnesHutFailureCode::singular_geometry:
        return "CUDA Barnes-Hut encountered coincident geometry";
    case CudaBarnesHutFailureCode::nonrepresentable_geometry:
        return "CUDA Barnes-Hut geometry is not representable";
    case CudaBarnesHutFailureCode::nonrepresentable_acceleration:
        return "CUDA Barnes-Hut acceleration is not representable";
    case CudaBarnesHutFailureCode::traversal_stack_exhausted:
        return "CUDA Barnes-Hut traversal stack capacity exceeded";
    case CudaBarnesHutFailureCode::traversal_work_exhausted:
        return "CUDA Barnes-Hut traversal work budget exceeded";
    case CudaBarnesHutFailureCode::runtime_failure:
        return "CUDA Barnes-Hut runtime failure";
    }
    return "unknown CUDA Barnes-Hut failure";
}

}  // namespace

class CudaBarnesHutForceBackend::Impl {
public:
    explicit Impl(const CudaBarnesHutConfiguration& configuration)
        : device_index_(configuration.device_index),
          host_failure_injection_(configuration.host_failure_injection) {
        device_info_ = cuda_device_info(device_index_);
        check_cuda(cudaSetDevice(device_index_), "cudaSetDevice CUDA Barnes-Hut");
        check_cuda(cudaFree(nullptr), "CUDA Barnes-Hut context initialization");
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate CUDA Barnes-Hut");
    }

    ~Impl() { release(); }

    void mark_runtime_failure() noexcept {
        terminal_ = true;
        last_failure_.code = CudaBarnesHutFailureCode::runtime_failure;
        last_failure_.target_index = barnes_hut_no_packed_index;
        last_failure_.node_index = barnes_hut_no_packed_index;
    }

    void require_usable() const {
        if (terminal_) {
            throw std::runtime_error(
                "CUDA Barnes-Hut executor is terminal after a runtime failure; restart the process before further GPU work");
        }
    }

    bool has_buffers() const noexcept {
        return device_work_ != nullptr || device_failure_ != nullptr ||
            device_accelerations_ != nullptr || device_inverse_rank_ != nullptr ||
            device_permutation_ != nullptr || device_nodes_ != nullptr ||
            device_positions_ != nullptr || device_masses_ != nullptr ||
            host_work_ != nullptr || host_failure_ != nullptr ||
            host_accelerations_ != nullptr || host_inverse_rank_ != nullptr ||
            host_permutation_ != nullptr || host_nodes_ != nullptr ||
            host_positions_ != nullptr || host_masses_ != nullptr;
    }

    void reset_buffer_accounting() noexcept {
        body_capacity_ = 0;
        node_capacity_ = 0;
        allocated_device_bytes_ = 0;
        allocated_pinned_host_bytes_ = 0;
    }

    void release_buffers_noexcept() noexcept {
        // Destruction cannot report errors. Quiescence was proved by release()
        // before this path; any failed free therefore leaves that allocation
        // retained by the CUDA runtime until context/process teardown.
        (void)cudaFree(device_work_);
        (void)cudaFree(device_failure_);
        (void)cudaFree(device_accelerations_);
        (void)cudaFree(device_inverse_rank_);
        (void)cudaFree(device_permutation_);
        (void)cudaFree(device_nodes_);
        (void)cudaFree(device_positions_);
        (void)cudaFree(device_masses_);
        (void)cudaFreeHost(host_work_);
        (void)cudaFreeHost(host_failure_);
        (void)cudaFreeHost(host_accelerations_);
        (void)cudaFreeHost(host_inverse_rank_);
        (void)cudaFreeHost(host_permutation_);
        (void)cudaFreeHost(host_nodes_);
        (void)cudaFreeHost(host_positions_);
        (void)cudaFreeHost(host_masses_);
        clear_buffer_pointers();
    }

    void release_buffers_checked(bool permit_injection) {
        if (permit_injection && has_buffers() &&
            host_failure_injection_ ==
                CudaBarnesHutHostFailureInjection::
                    live_release_before_first_deallocation) {
            mark_runtime_failure();
            throw std::runtime_error(
                "injected CUDA Barnes-Hut live-release failure before deallocation");
        }

        const auto release_device = [&](auto*& pointer, const char* operation) {
            if (pointer == nullptr) {
                return;
            }
            const cudaError_t result = cudaFree(pointer);
            if (result != cudaSuccess) {
                mark_runtime_failure();
                throw_cuda_error(result, operation);
            }
            pointer = nullptr;
        };
        const auto release_host = [&](auto*& pointer, const char* operation) {
            if (pointer == nullptr) {
                return;
            }
            const cudaError_t result = cudaFreeHost(pointer);
            if (result != cudaSuccess) {
                mark_runtime_failure();
                throw_cuda_error(result, operation);
            }
            pointer = nullptr;
        };

        release_device(device_work_, "cudaFree CUDA Barnes-Hut work");
        release_device(device_failure_, "cudaFree CUDA Barnes-Hut failure");
        release_device(
            device_accelerations_, "cudaFree CUDA Barnes-Hut accelerations");
        release_device(
            device_inverse_rank_, "cudaFree CUDA Barnes-Hut inverse rank");
        release_device(
            device_permutation_, "cudaFree CUDA Barnes-Hut permutation");
        release_device(device_nodes_, "cudaFree CUDA Barnes-Hut nodes");
        release_device(device_positions_, "cudaFree CUDA Barnes-Hut positions");
        release_device(device_masses_, "cudaFree CUDA Barnes-Hut masses");
        release_host(host_work_, "cudaFreeHost CUDA Barnes-Hut work");
        release_host(host_failure_, "cudaFreeHost CUDA Barnes-Hut failure");
        release_host(
            host_accelerations_, "cudaFreeHost CUDA Barnes-Hut accelerations");
        release_host(
            host_inverse_rank_, "cudaFreeHost CUDA Barnes-Hut inverse rank");
        release_host(
            host_permutation_, "cudaFreeHost CUDA Barnes-Hut permutation");
        release_host(host_nodes_, "cudaFreeHost CUDA Barnes-Hut nodes");
        release_host(host_positions_, "cudaFreeHost CUDA Barnes-Hut positions");
        release_host(host_masses_, "cudaFreeHost CUDA Barnes-Hut masses");
        reset_buffer_accounting();
    }

    void clear_buffer_pointers() noexcept {
        device_work_ = nullptr;
        device_failure_ = nullptr;
        device_accelerations_ = nullptr;
        device_inverse_rank_ = nullptr;
        device_permutation_ = nullptr;
        device_nodes_ = nullptr;
        device_positions_ = nullptr;
        device_masses_ = nullptr;
        host_work_ = nullptr;
        host_failure_ = nullptr;
        host_accelerations_ = nullptr;
        host_inverse_rank_ = nullptr;
        host_permutation_ = nullptr;
        host_nodes_ = nullptr;
        host_positions_ = nullptr;
        host_masses_ = nullptr;
        reset_buffer_accounting();
    }

    void quarantine() noexcept {
        // A failed device selection or stream synchronization does not prove
        // that queued work has stopped. Retain every stream-referenced device
        // and pinned-host allocation until context/process teardown rather
        // than risk freeing storage still in use. No cudaDeviceReset is issued.
        clear_buffer_pointers();
        stream_ = nullptr;
    }

    void release() noexcept {
        if (stream_ == nullptr) {
            clear_buffer_pointers();
            return;
        }
        const cudaError_t selected = cudaSetDevice(device_index_);
        if (selected != cudaSuccess) {
            quarantine();
            return;
        }
        const cudaError_t synchronized = cudaStreamSynchronize(stream_);
        (void)cudaStreamDestroy(stream_);
        stream_ = nullptr;
        if (synchronized != cudaSuccess) {
            quarantine();
            return;
        }
        release_buffers_noexcept();
    }

    void ensure_capacity(std::size_t body_count, std::size_t node_count) {
        require_usable();
        if (body_count <= body_capacity_ && node_count <= node_capacity_) {
            return;
        }
        const cudaError_t selected = cudaSetDevice(device_index_);
        if (selected != cudaSuccess) {
            mark_runtime_failure();
            throw_cuda_error(selected, "cudaSetDevice CUDA Barnes-Hut capacity");
        }
        const cudaError_t synchronized = cudaStreamSynchronize(stream_);
        if (synchronized != cudaSuccess) {
            mark_runtime_failure();
            throw_cuda_error(synchronized, "cudaStreamSynchronize CUDA Barnes-Hut capacity");
        }
        release_buffers_checked(true);
        const StorageRequirements planned = storage_requirements(body_count, node_count);
        // Charge the complete candidate before the first allocation. If a
        // later allocation and its checked cleanup both fail, the terminal
        // executor retains conservative accounting for every surviving handle.
        body_capacity_ = body_count;
        node_capacity_ = node_count;
        allocated_device_bytes_ = planned.device_bytes;
        allocated_pinned_host_bytes_ = planned.pinned_host_bytes;
        try {
            check_cuda(cudaMalloc(&device_masses_, body_count * sizeof(double)),
                       "cudaMalloc CUDA Barnes-Hut masses");
            check_cuda(cudaMalloc(&device_positions_, body_count * sizeof(DeviceVec3)),
                       "cudaMalloc CUDA Barnes-Hut positions");
            check_cuda(cudaMalloc(&device_nodes_, node_count * sizeof(BarnesHutPackedNode)),
                       "cudaMalloc CUDA Barnes-Hut nodes");
            check_cuda(cudaMalloc(&device_permutation_, body_count * sizeof(std::uint32_t)),
                       "cudaMalloc CUDA Barnes-Hut permutation");
            check_cuda(cudaMalloc(&device_inverse_rank_, body_count * sizeof(std::uint32_t)),
                       "cudaMalloc CUDA Barnes-Hut inverse rank");
            check_cuda(cudaMalloc(&device_accelerations_, body_count * sizeof(DeviceVec3)),
                       "cudaMalloc CUDA Barnes-Hut accelerations");
            check_cuda(cudaMalloc(&device_failure_, sizeof(DeviceFailure)),
                       "cudaMalloc CUDA Barnes-Hut failure");
            check_cuda(cudaMalloc(&device_work_, sizeof(unsigned long long)),
                       "cudaMalloc CUDA Barnes-Hut work");
            check_cuda(cudaMallocHost(&host_masses_, body_count * sizeof(double)),
                       "cudaMallocHost CUDA Barnes-Hut masses");
            check_cuda(cudaMallocHost(&host_positions_, body_count * sizeof(DeviceVec3)),
                       "cudaMallocHost CUDA Barnes-Hut positions");
            check_cuda(cudaMallocHost(&host_nodes_, node_count * sizeof(BarnesHutPackedNode)),
                       "cudaMallocHost CUDA Barnes-Hut nodes");
            check_cuda(cudaMallocHost(&host_permutation_, body_count * sizeof(std::uint32_t)),
                       "cudaMallocHost CUDA Barnes-Hut permutation");
            check_cuda(cudaMallocHost(&host_inverse_rank_, body_count * sizeof(std::uint32_t)),
                       "cudaMallocHost CUDA Barnes-Hut inverse rank");
            check_cuda(cudaMallocHost(&host_accelerations_, body_count * sizeof(DeviceVec3)),
                       "cudaMallocHost CUDA Barnes-Hut accelerations");
            check_cuda(cudaMallocHost(&host_failure_, sizeof(DeviceFailure)),
                       "cudaMallocHost CUDA Barnes-Hut failure");
            check_cuda(cudaMallocHost(&host_work_, sizeof(unsigned long long)),
                       "cudaMallocHost CUDA Barnes-Hut work");
        } catch (...) {
            const std::exception_ptr allocation_failure = std::current_exception();
            try {
                release_buffers_checked(false);
            } catch (...) {
                mark_runtime_failure();
                throw;
            }
            mark_runtime_failure();
            std::rethrow_exception(allocation_failure);
        }
    }

    int device_index_{0};
    CudaBarnesHutHostFailureInjection host_failure_injection_{
        CudaBarnesHutHostFailureInjection::none};
    CudaDeviceInfo device_info_;
    cudaStream_t stream_{nullptr};
    bool terminal_{false};
    std::size_t body_capacity_{0};
    std::size_t node_capacity_{0};
    std::size_t allocated_device_bytes_{0};
    std::size_t allocated_pinned_host_bytes_{0};
    double* device_masses_{nullptr};
    DeviceVec3* device_positions_{nullptr};
    BarnesHutPackedNode* device_nodes_{nullptr};
    std::uint32_t* device_permutation_{nullptr};
    std::uint32_t* device_inverse_rank_{nullptr};
    DeviceVec3* device_accelerations_{nullptr};
    DeviceFailure* device_failure_{nullptr};
    unsigned long long* device_work_{nullptr};
    double* host_masses_{nullptr};
    DeviceVec3* host_positions_{nullptr};
    BarnesHutPackedNode* host_nodes_{nullptr};
    std::uint32_t* host_permutation_{nullptr};
    std::uint32_t* host_inverse_rank_{nullptr};
    DeviceVec3* host_accelerations_{nullptr};
    DeviceFailure* host_failure_{nullptr};
    unsigned long long* host_work_{nullptr};
    CudaBarnesHutStatistics last_statistics_;
    CudaBarnesHutFailure last_failure_;
};

CudaBarnesHutForceBackend::CudaBarnesHutForceBackend(
    CudaBarnesHutConfiguration configuration)
    : configuration_(std::move(configuration)) {
    validate_configuration(configuration_);
    impl_ = std::make_unique<Impl>(configuration_);
}

CudaBarnesHutForceBackend::~CudaBarnesHutForceBackend() = default;
CudaBarnesHutForceBackend::CudaBarnesHutForceBackend(
    CudaBarnesHutForceBackend&&) noexcept = default;
CudaBarnesHutForceBackend& CudaBarnesHutForceBackend::operator=(
    CudaBarnesHutForceBackend&&) noexcept = default;

std::vector<Vec3> CudaBarnesHutForceBackend::compute_accelerations(
    const SimulationState& state,
    double gravitational_constant_value) const {
    impl_->require_usable();
    impl_->last_failure_ = {};
    validate_force_inputs(state, gravitational_constant_value);
    if (state.bodies.empty()) {
        impl_->last_failure_ = {};
        impl_->last_statistics_ = {};
        return {};
    }

    const auto build_start = std::chrono::steady_clock::now();
    BarnesHutPackedTree tree = build_barnes_hut_packed_tree(
        state,
        configuration_.tree,
        configuration_.host_staging_memory_budget_bytes);
    const auto build_end = std::chrono::steady_clock::now();
    if (tree.nodes.empty()) {
        throw std::logic_error("CUDA Barnes-Hut CPU builder produced no root");
    }
    const StorageRequirements storage =
        storage_requirements(state.bodies.size(), tree.nodes.size());
    if (storage.device_bytes > configuration_.device_memory_budget_bytes) {
        throw std::length_error("CUDA Barnes-Hut device storage exceeds memory budget");
    }
    const std::size_t total_host_bytes = checked_add(
        tree.storage_bytes,
        storage.pinned_host_bytes,
        "CUDA Barnes-Hut host staging storage");
    if (total_host_bytes > configuration_.host_staging_memory_budget_bytes) {
        throw std::length_error("CUDA Barnes-Hut host staging exceeds memory budget");
    }

    impl_->ensure_capacity(state.bodies.size(), tree.nodes.size());
    const std::size_t actual_host_bytes = checked_add(
        tree.storage_bytes,
        impl_->allocated_pinned_host_bytes_,
        "CUDA Barnes-Hut actual host staging storage");
    if (impl_->allocated_device_bytes_ > configuration_.device_memory_budget_bytes ||
        actual_host_bytes > configuration_.host_staging_memory_budget_bytes) {
        throw std::logic_error("CUDA Barnes-Hut retained capacity exceeds memory budget");
    }
    for (std::size_t index = 0; index < state.bodies.size(); ++index) {
        impl_->host_masses_[index] = state.bodies[index].mass_kg;
        impl_->host_positions_[index] = {
            state.bodies[index].position_m.x,
            state.bodies[index].position_m.y,
            state.bodies[index].position_m.z,
        };
    }
    std::copy(tree.nodes.begin(), tree.nodes.end(), impl_->host_nodes_);
    std::copy(tree.permutation.begin(), tree.permutation.end(), impl_->host_permutation_);
    std::copy(tree.inverse_rank.begin(), tree.inverse_rank.end(), impl_->host_inverse_rank_);
    *impl_->host_failure_ = {
        0U,
        0U,
        static_cast<std::uint32_t>(CudaBarnesHutFailureCode::none),
        barnes_hut_no_packed_index,
        barnes_hut_no_packed_index,
    };
    *impl_->host_work_ = 0ULL;

    const cudaError_t selected = cudaSetDevice(configuration_.device_index);
    if (selected != cudaSuccess) {
        impl_->mark_runtime_failure();
        throw_cuda_error(selected, "cudaSetDevice CUDA Barnes-Hut force");
    }
    try {
        check_cuda(cudaMemcpyAsync(
                       impl_->device_masses_,
                       impl_->host_masses_,
                       state.bodies.size() * sizeof(double),
                       cudaMemcpyHostToDevice,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut masses");
        check_cuda(cudaMemcpyAsync(
                       impl_->device_positions_,
                       impl_->host_positions_,
                       state.bodies.size() * sizeof(DeviceVec3),
                       cudaMemcpyHostToDevice,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut positions");
        check_cuda(cudaMemcpyAsync(
                       impl_->device_nodes_,
                       impl_->host_nodes_,
                       tree.nodes.size() * sizeof(BarnesHutPackedNode),
                       cudaMemcpyHostToDevice,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut nodes");
        check_cuda(cudaMemcpyAsync(
                       impl_->device_permutation_,
                       impl_->host_permutation_,
                       state.bodies.size() * sizeof(std::uint32_t),
                       cudaMemcpyHostToDevice,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut permutation");
        check_cuda(cudaMemcpyAsync(
                       impl_->device_inverse_rank_,
                       impl_->host_inverse_rank_,
                       state.bodies.size() * sizeof(std::uint32_t),
                       cudaMemcpyHostToDevice,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut inverse rank");
        check_cuda(cudaMemcpyAsync(
                       impl_->device_failure_,
                       impl_->host_failure_,
                       sizeof(DeviceFailure),
                       cudaMemcpyHostToDevice,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut failure reset");
        check_cuda(cudaMemcpyAsync(
                       impl_->device_work_,
                       impl_->host_work_,
                       sizeof(unsigned long long),
                       cudaMemcpyHostToDevice,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut work reset");

        barnes_hut_force_kernel<<<
            block_count(state.bodies.size()),
            threads_per_block,
            0,
            impl_->stream_>>>(
            impl_->device_masses_,
            impl_->device_positions_,
            impl_->device_nodes_,
            static_cast<std::uint32_t>(tree.nodes.size()),
            impl_->device_permutation_,
            impl_->device_inverse_rank_,
            static_cast<std::uint32_t>(state.bodies.size()),
            configuration_.tree.theta,
            gravitational_constant_value,
            configuration_.traversal_stack_capacity,
            static_cast<unsigned long long>(
                configuration_.tree.maximum_traversal_work),
            impl_->device_accelerations_,
            impl_->device_work_,
            impl_->device_failure_);
        check_launch("barnes_hut_force_kernel launch");

        check_cuda(cudaMemcpyAsync(
                       impl_->host_accelerations_,
                       impl_->device_accelerations_,
                       state.bodies.size() * sizeof(DeviceVec3),
                       cudaMemcpyDeviceToHost,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut accelerations");
        check_cuda(cudaMemcpyAsync(
                       impl_->host_failure_,
                       impl_->device_failure_,
                       sizeof(DeviceFailure),
                       cudaMemcpyDeviceToHost,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut failure report");
        check_cuda(cudaMemcpyAsync(
                       impl_->host_work_,
                       impl_->device_work_,
                       sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost,
                       impl_->stream_),
                   "cudaMemcpyAsync CUDA Barnes-Hut work report");
        check_cuda(
            cudaStreamSynchronize(impl_->stream_),
            "cudaStreamSynchronize CUDA Barnes-Hut force");
    } catch (...) {
        impl_->mark_runtime_failure();
        throw;
    }

    if (impl_->host_failure_->claimed != impl_->host_failure_->published ||
        impl_->host_failure_->claimed > 1U) {
        impl_->mark_runtime_failure();
        throw std::runtime_error(
            "CUDA Barnes-Hut failure metadata publication is incomplete");
    }
    if (impl_->host_failure_->published == 0U) {
        impl_->last_failure_ = {};
    } else {
        const auto code = static_cast<CudaBarnesHutFailureCode>(
            impl_->host_failure_->code);
        if (code == CudaBarnesHutFailureCode::none ||
            code > CudaBarnesHutFailureCode::traversal_work_exhausted) {
            impl_->mark_runtime_failure();
            throw std::runtime_error(
                "CUDA Barnes-Hut failure metadata contains an invalid code");
        }
        impl_->last_failure_.code = code;
        impl_->last_failure_.target_index = impl_->host_failure_->target_index;
        impl_->last_failure_.node_index = impl_->host_failure_->node_index;
    }
    if (impl_->last_failure_.code != CudaBarnesHutFailureCode::none) {
        const std::string message = failure_message(impl_->last_failure_.code);
        if (impl_->last_failure_.code ==
                CudaBarnesHutFailureCode::traversal_stack_exhausted ||
            impl_->last_failure_.code ==
                CudaBarnesHutFailureCode::traversal_work_exhausted) {
            throw std::length_error(message);
        }
        throw std::domain_error(message);
    }
    if (*impl_->host_work_ > configuration_.tree.maximum_traversal_work) {
        throw std::logic_error("CUDA Barnes-Hut traversal counter exceeded its hard limit");
    }

    std::vector<Vec3> result(state.bodies.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
        const DeviceVec3 value = impl_->host_accelerations_[index];
        if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
            !std::isfinite(value.z)) {
            impl_->last_failure_ = {
                CudaBarnesHutFailureCode::nonrepresentable_acceleration,
                static_cast<std::uint32_t>(index),
                barnes_hut_no_packed_index,
            };
            throw std::domain_error("CUDA Barnes-Hut returned a non-finite acceleration");
        }
        result[index] = {value.x, value.y, value.z};
    }

    CudaBarnesHutStatistics candidate;
    candidate.body_count = state.bodies.size();
    candidate.node_count = tree.nodes.size();
    candidate.device_storage_bytes = impl_->allocated_device_bytes_;
    candidate.host_staging_bytes = actual_host_bytes;
    candidate.traversal_stack_bytes_per_thread =
        cuda_barnes_hut_maximum_stack_capacity * sizeof(std::uint32_t);
    candidate.traversal_work = static_cast<std::size_t>(*impl_->host_work_);
    candidate.host_tree = tree.statistics;
    candidate.host_tree.tree_build_seconds =
        std::chrono::duration<double>(build_end - build_start).count();
    impl_->last_statistics_ = candidate;
    impl_->last_failure_ = {};
    return result;
}

CudaBarnesHutStatistics CudaBarnesHutForceBackend::last_statistics() const noexcept {
    return impl_->last_statistics_;
}

CudaBarnesHutFailure CudaBarnesHutForceBackend::last_failure() const noexcept {
    return impl_->last_failure_;
}

CudaDeviceInfo CudaBarnesHutForceBackend::device_info() const {
    return impl_->device_info_;
}

}  // namespace nbody
