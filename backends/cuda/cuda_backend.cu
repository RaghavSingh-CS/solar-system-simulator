#include "nbody/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nbody {
namespace {

struct DeviceVec3 {
    double x;
    double y;
    double z;
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

void validate_state(const SimulationState& state) {
    for (const auto& body : state.bodies) {
        if (!(body.mass_kg > 0.0) || !std::isfinite(body.mass_kg)) {
            throw std::invalid_argument("body mass must be finite and positive");
        }
        const auto finite_vector = [](const Vec3& value) {
            return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
        };
        if (!finite_vector(body.position_m) || !finite_vector(body.velocity_m_s)) {
            throw std::invalid_argument("body position and velocity must be finite");
        }
    }
}

void validate_parameters(double gravitational_constant_value) {
    if (!(gravitational_constant_value > 0.0) || !std::isfinite(gravitational_constant_value)) {
        throw std::invalid_argument("gravitational constant must be finite and positive");
    }
}

__global__ void compute_accelerations_kernel(
    const double* masses,
    const DeviceVec3* positions,
    DeviceVec3* accelerations,
    int body_count,
    double gravitational_constant_value,
    int* invalid_position) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= body_count) {
        return;
    }

    DeviceVec3 acceleration{0.0, 0.0, 0.0};
    for (int j = 0; j < body_count; ++j) {
        if (i == j) {
            continue;
        }
        const double dx = positions[j].x - positions[i].x;
        const double dy = positions[j].y - positions[i].y;
        const double dz = positions[j].z - positions[i].z;
        const double distance_squared = dx * dx + dy * dy + dz * dz;
        if (!(distance_squared > 0.0) || !isfinite(distance_squared)) {
            atomicExch(invalid_position, 1);
            continue;
        }
        const double inverse_distance = 1.0 / sqrt(distance_squared);
        const double inverse_distance_cubed =
            inverse_distance * inverse_distance * inverse_distance;
        const double scale =
            gravitational_constant_value * masses[j] * inverse_distance_cubed;
        acceleration.x += dx * scale;
        acceleration.y += dy * scale;
        acceleration.z += dz * scale;
    }
    accelerations[i] = acceleration;
}

__global__ void update_positions_kernel(
    DeviceVec3* positions,
    const DeviceVec3* velocities,
    const DeviceVec3* accelerations,
    int body_count,
    double dt_s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= body_count) {
        return;
    }
    const double half_dt_squared = 0.5 * dt_s * dt_s;
    positions[i].x += velocities[i].x * dt_s + accelerations[i].x * half_dt_squared;
    positions[i].y += velocities[i].y * dt_s + accelerations[i].y * half_dt_squared;
    positions[i].z += velocities[i].z * dt_s + accelerations[i].z * half_dt_squared;
}

__global__ void update_velocities_kernel(
    DeviceVec3* velocities,
    const DeviceVec3* old_accelerations,
    const DeviceVec3* new_accelerations,
    int body_count,
    double dt_s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= body_count) {
        return;
    }
    const double half_dt = 0.5 * dt_s;
    velocities[i].x += (old_accelerations[i].x + new_accelerations[i].x) * half_dt;
    velocities[i].y += (old_accelerations[i].y + new_accelerations[i].y) * half_dt;
    velocities[i].z += (old_accelerations[i].z + new_accelerations[i].z) * half_dt;
}

int block_count(std::size_t body_count) {
    constexpr int threads_per_block = 256;
    return static_cast<int>((body_count + threads_per_block - 1) / threads_per_block);
}

constexpr int threads_per_block = 256;

void launch_accelerations(
    const double* masses,
    const DeviceVec3* positions,
    DeviceVec3* accelerations,
    std::size_t body_count,
    double gravitational_constant_value,
    int* invalid_position,
    cudaStream_t stream) {
    compute_accelerations_kernel<<<block_count(body_count), threads_per_block, 0, stream>>>(
        masses, positions, accelerations, static_cast<int>(body_count),
        gravitational_constant_value, invalid_position);
    check_launch("compute_accelerations_kernel launch");
}

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    return static_cast<double>(milliseconds);
}

}  // namespace

int cuda_device_count() {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

CudaDeviceInfo cuda_device_info(int device_index) {
    const int count = cuda_device_count();
    if (device_index < 0 || device_index >= count) {
        throw std::out_of_range("CUDA device index is unavailable");
    }
    check_cuda(cudaSetDevice(device_index), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_index), "cudaGetDeviceProperties");
    CudaDeviceInfo result;
    result.device_index = device_index;
    result.name = properties.name;
    result.compute_capability_major = properties.major;
    result.compute_capability_minor = properties.minor;
    result.total_global_memory_bytes = properties.totalGlobalMem;
    result.multiprocessor_count = properties.multiProcessorCount;
    check_cuda(cudaDriverGetVersion(&result.cuda_driver_version), "cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&result.cuda_runtime_version), "cudaRuntimeGetVersion");
    return result;
}

class CudaForceBackend::Impl {
public:
    explicit Impl(int device_index) : device_index_(device_index) {
        (void)cuda_device_info(device_index_);
    }

    ~Impl() { release(); }

    void release() noexcept {
        (void)cudaSetDevice(device_index_);
        cudaFree(invalid_position_);
        cudaFree(accelerations_);
        cudaFree(positions_);
        cudaFree(masses_);
        invalid_position_ = nullptr;
        accelerations_ = nullptr;
        positions_ = nullptr;
        masses_ = nullptr;
        capacity_ = 0;
    }

    void ensure_capacity(std::size_t body_count) {
        if (body_count <= capacity_) {
            return;
        }
        check_cuda(cudaSetDevice(device_index_), "cudaSetDevice");
        release();
        try {
            check_cuda(cudaMalloc(&masses_, body_count * sizeof(double)), "cudaMalloc masses");
            check_cuda(cudaMalloc(&positions_, body_count * sizeof(DeviceVec3)), "cudaMalloc positions");
            check_cuda(
                cudaMalloc(&accelerations_, body_count * sizeof(DeviceVec3)),
                "cudaMalloc accelerations");
            check_cuda(cudaMalloc(&invalid_position_, sizeof(int)), "cudaMalloc invalid flag");
            capacity_ = body_count;
        } catch (...) {
            release();
            throw;
        }
    }

    int device_index_{0};
    std::size_t capacity_{0};
    double* masses_{nullptr};
    DeviceVec3* positions_{nullptr};
    DeviceVec3* accelerations_{nullptr};
    int* invalid_position_{nullptr};
};

CudaForceBackend::CudaForceBackend(int device_index) : impl_(std::make_unique<Impl>(device_index)) {}
CudaForceBackend::~CudaForceBackend() = default;
CudaForceBackend::CudaForceBackend(CudaForceBackend&&) noexcept = default;
CudaForceBackend& CudaForceBackend::operator=(CudaForceBackend&&) noexcept = default;

const char* CudaForceBackend::name() const noexcept { return "cuda"; }

std::vector<Vec3> CudaForceBackend::compute_accelerations(
    const SimulationState& state,
    double gravitational_constant_value) const {
    validate_parameters(gravitational_constant_value);
    validate_state(state);
    if (state.bodies.empty()) {
        return {};
    }
    if (state.bodies.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("body count exceeds CUDA kernel index range");
    }
    impl_->ensure_capacity(state.bodies.size());
    check_cuda(cudaSetDevice(impl_->device_index_), "cudaSetDevice");

    std::vector<double> masses(state.bodies.size());
    std::vector<DeviceVec3> positions(state.bodies.size());
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        masses[i] = state.bodies[i].mass_kg;
        positions[i] = {
            state.bodies[i].position_m.x,
            state.bodies[i].position_m.y,
            state.bodies[i].position_m.z,
        };
    }
    check_cuda(
        cudaMemcpy(impl_->masses_, masses.data(), masses.size() * sizeof(double), cudaMemcpyHostToDevice),
        "cudaMemcpy masses H2D");
    check_cuda(
        cudaMemcpy(impl_->positions_, positions.data(), positions.size() * sizeof(DeviceVec3), cudaMemcpyHostToDevice),
        "cudaMemcpy positions H2D");
    check_cuda(cudaMemset(impl_->invalid_position_, 0, sizeof(int)), "cudaMemset invalid flag");
    launch_accelerations(
        impl_->masses_, impl_->positions_, impl_->accelerations_, state.bodies.size(),
        gravitational_constant_value, impl_->invalid_position_, nullptr);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize force backend");

    std::vector<DeviceVec3> device_accelerations(state.bodies.size());
    int invalid_position = 0;
    check_cuda(
        cudaMemcpy(device_accelerations.data(), impl_->accelerations_,
                   device_accelerations.size() * sizeof(DeviceVec3), cudaMemcpyDeviceToHost),
        "cudaMemcpy accelerations D2H");
    check_cuda(
        cudaMemcpy(&invalid_position, impl_->invalid_position_, sizeof(int), cudaMemcpyDeviceToHost),
        "cudaMemcpy invalid flag D2H");
    if (invalid_position != 0) {
        throw std::domain_error("coincident or invalid body positions");
    }

    std::vector<Vec3> result(state.bodies.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = {
            device_accelerations[i].x,
            device_accelerations[i].y,
            device_accelerations[i].z,
        };
    }
    return result;
}

class CudaVelocityVerlet::Impl {
public:
    Impl(std::size_t body_count, int device_index)
        : body_count_(body_count), device_index_(device_index) {
        if (body_count_ < 2) {
            throw std::invalid_argument("CUDA integrator requires at least two bodies");
        }
        if (body_count_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error("body count exceeds CUDA kernel index range");
        }
        const auto start = std::chrono::steady_clock::now();
        device_info_ = cuda_device_info(device_index_);
        try {
            check_cuda(cudaFree(nullptr), "CUDA context initialization");
            check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
            check_cuda(cudaEventCreate(&total_start_), "cudaEventCreate total_start");
            check_cuda(cudaEventCreate(&after_h2d_), "cudaEventCreate after_h2d");
            check_cuda(cudaEventCreate(&after_compute_), "cudaEventCreate after_compute");
            check_cuda(cudaEventCreate(&after_d2h_), "cudaEventCreate after_d2h");
            check_cuda(cudaMalloc(&masses_, body_count_ * sizeof(double)), "cudaMalloc masses");
            check_cuda(cudaMalloc(&positions_, body_count_ * sizeof(DeviceVec3)), "cudaMalloc positions");
            check_cuda(cudaMalloc(&velocities_, body_count_ * sizeof(DeviceVec3)), "cudaMalloc velocities");
            check_cuda(cudaMalloc(&old_accelerations_, body_count_ * sizeof(DeviceVec3)), "cudaMalloc old accelerations");
            check_cuda(cudaMalloc(&new_accelerations_, body_count_ * sizeof(DeviceVec3)), "cudaMalloc new accelerations");
            check_cuda(cudaMalloc(&invalid_position_, sizeof(int)), "cudaMalloc invalid flag");
            check_cuda(cudaMallocHost(&host_masses_, body_count_ * sizeof(double)), "cudaMallocHost masses");
            check_cuda(cudaMallocHost(&host_positions_, body_count_ * sizeof(DeviceVec3)), "cudaMallocHost positions");
            check_cuda(cudaMallocHost(&host_velocities_, body_count_ * sizeof(DeviceVec3)), "cudaMallocHost velocities");
            check_cuda(cudaMallocHost(&host_invalid_position_, sizeof(int)), "cudaMallocHost invalid flag");
        } catch (...) {
            release();
            throw;
        }
        allocation_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }

    ~Impl() { release(); }

    void release() noexcept {
        (void)cudaSetDevice(device_index_);
        cudaFreeHost(host_invalid_position_);
        cudaFreeHost(host_velocities_);
        cudaFreeHost(host_positions_);
        cudaFreeHost(host_masses_);
        cudaFree(invalid_position_);
        cudaFree(new_accelerations_);
        cudaFree(old_accelerations_);
        cudaFree(velocities_);
        cudaFree(positions_);
        cudaFree(masses_);
        cudaEventDestroy(after_d2h_);
        cudaEventDestroy(after_compute_);
        cudaEventDestroy(after_h2d_);
        cudaEventDestroy(total_start_);
        cudaStreamDestroy(stream_);
        host_invalid_position_ = nullptr;
        host_velocities_ = nullptr;
        host_positions_ = nullptr;
        host_masses_ = nullptr;
        invalid_position_ = nullptr;
        new_accelerations_ = nullptr;
        old_accelerations_ = nullptr;
        velocities_ = nullptr;
        positions_ = nullptr;
        masses_ = nullptr;
        after_d2h_ = nullptr;
        after_compute_ = nullptr;
        after_h2d_ = nullptr;
        total_start_ = nullptr;
        stream_ = nullptr;
    }

    std::size_t body_count_{0};
    int device_index_{0};
    CudaDeviceInfo device_info_;
    double allocation_ms_{0.0};
    cudaStream_t stream_{nullptr};
    cudaEvent_t total_start_{nullptr};
    cudaEvent_t after_h2d_{nullptr};
    cudaEvent_t after_compute_{nullptr};
    cudaEvent_t after_d2h_{nullptr};
    double* masses_{nullptr};
    DeviceVec3* positions_{nullptr};
    DeviceVec3* velocities_{nullptr};
    DeviceVec3* old_accelerations_{nullptr};
    DeviceVec3* new_accelerations_{nullptr};
    int* invalid_position_{nullptr};
    double* host_masses_{nullptr};
    DeviceVec3* host_positions_{nullptr};
    DeviceVec3* host_velocities_{nullptr};
    int* host_invalid_position_{nullptr};
};

CudaVelocityVerlet::CudaVelocityVerlet(std::size_t body_count, int device_index)
    : impl_(std::make_unique<Impl>(body_count, device_index)) {}
CudaVelocityVerlet::~CudaVelocityVerlet() = default;
CudaVelocityVerlet::CudaVelocityVerlet(CudaVelocityVerlet&&) noexcept = default;
CudaVelocityVerlet& CudaVelocityVerlet::operator=(CudaVelocityVerlet&&) noexcept = default;

CudaIntegrationResult CudaVelocityVerlet::run(
    const SimulationState& initial_state,
    std::size_t steps,
    double dt_s,
    double gravitational_constant_value) {
    validate_parameters(gravitational_constant_value);
    validate_state(initial_state);
    if (initial_state.bodies.size() != impl_->body_count_) {
        throw std::invalid_argument("initial state body count does not match CUDA allocation");
    }
    if (steps == 0 || !(dt_s > 0.0) || !std::isfinite(dt_s)) {
        throw std::invalid_argument("steps and finite positive time step are required");
    }
    check_cuda(cudaSetDevice(impl_->device_index_), "cudaSetDevice");

    const auto backend_wall_start = std::chrono::steady_clock::now();
    const auto staging_in_start = backend_wall_start;
    for (std::size_t i = 0; i < impl_->body_count_; ++i) {
        impl_->host_masses_[i] = initial_state.bodies[i].mass_kg;
        impl_->host_positions_[i] = {
            initial_state.bodies[i].position_m.x,
            initial_state.bodies[i].position_m.y,
            initial_state.bodies[i].position_m.z,
        };
        impl_->host_velocities_[i] = {
            initial_state.bodies[i].velocity_m_s.x,
            initial_state.bodies[i].velocity_m_s.y,
            initial_state.bodies[i].velocity_m_s.z,
        };
    }
    const auto staging_in_stop = std::chrono::steady_clock::now();

    check_cuda(cudaEventRecord(impl_->total_start_, impl_->stream_), "cudaEventRecord total_start");
    check_cuda(
        cudaMemcpyAsync(impl_->masses_, impl_->host_masses_, impl_->body_count_ * sizeof(double),
                        cudaMemcpyHostToDevice, impl_->stream_),
        "cudaMemcpyAsync masses H2D");
    check_cuda(
        cudaMemcpyAsync(impl_->positions_, impl_->host_positions_, impl_->body_count_ * sizeof(DeviceVec3),
                        cudaMemcpyHostToDevice, impl_->stream_),
        "cudaMemcpyAsync positions H2D");
    check_cuda(
        cudaMemcpyAsync(impl_->velocities_, impl_->host_velocities_, impl_->body_count_ * sizeof(DeviceVec3),
                        cudaMemcpyHostToDevice, impl_->stream_),
        "cudaMemcpyAsync velocities H2D");
    check_cuda(cudaEventRecord(impl_->after_h2d_, impl_->stream_), "cudaEventRecord after_h2d");
    check_cuda(
        cudaMemsetAsync(impl_->invalid_position_, 0, sizeof(int), impl_->stream_),
        "cudaMemsetAsync invalid flag");

    for (std::size_t step = 0; step < steps; ++step) {
        launch_accelerations(
            impl_->masses_, impl_->positions_, impl_->old_accelerations_, impl_->body_count_,
            gravitational_constant_value, impl_->invalid_position_, impl_->stream_);
        update_positions_kernel<<<block_count(impl_->body_count_), threads_per_block, 0, impl_->stream_>>>(
            impl_->positions_, impl_->velocities_, impl_->old_accelerations_,
            static_cast<int>(impl_->body_count_), dt_s);
        check_launch("update_positions_kernel launch");
        launch_accelerations(
            impl_->masses_, impl_->positions_, impl_->new_accelerations_, impl_->body_count_,
            gravitational_constant_value, impl_->invalid_position_, impl_->stream_);
        update_velocities_kernel<<<block_count(impl_->body_count_), threads_per_block, 0, impl_->stream_>>>(
            impl_->velocities_, impl_->old_accelerations_, impl_->new_accelerations_,
            static_cast<int>(impl_->body_count_), dt_s);
        check_launch("update_velocities_kernel launch");
    }
    check_cuda(cudaEventRecord(impl_->after_compute_, impl_->stream_), "cudaEventRecord after_compute");

    *impl_->host_invalid_position_ = 0;
    check_cuda(
        cudaMemcpyAsync(impl_->host_positions_, impl_->positions_, impl_->body_count_ * sizeof(DeviceVec3),
                        cudaMemcpyDeviceToHost, impl_->stream_),
        "cudaMemcpyAsync positions D2H");
    check_cuda(
        cudaMemcpyAsync(impl_->host_velocities_, impl_->velocities_, impl_->body_count_ * sizeof(DeviceVec3),
                        cudaMemcpyDeviceToHost, impl_->stream_),
        "cudaMemcpyAsync velocities D2H");
    check_cuda(
        cudaMemcpyAsync(impl_->host_invalid_position_, impl_->invalid_position_, sizeof(int),
                        cudaMemcpyDeviceToHost, impl_->stream_),
        "cudaMemcpyAsync invalid flag D2H");
    check_cuda(cudaEventRecord(impl_->after_d2h_, impl_->stream_), "cudaEventRecord after_d2h");
    check_cuda(cudaEventSynchronize(impl_->after_d2h_), "cudaEventSynchronize after_d2h");
    check_cuda(cudaStreamSynchronize(impl_->stream_), "cudaStreamSynchronize integration");
    if (*impl_->host_invalid_position_ != 0) {
        throw std::domain_error("coincident or invalid body positions during CUDA integration");
    }

    SimulationState final_state = initial_state;
    final_state.time_s += static_cast<double>(steps) * dt_s;
    const auto staging_out_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < impl_->body_count_; ++i) {
        final_state.bodies[i].position_m = {
            impl_->host_positions_[i].x,
            impl_->host_positions_[i].y,
            impl_->host_positions_[i].z,
        };
        final_state.bodies[i].velocity_m_s = {
            impl_->host_velocities_[i].x,
            impl_->host_velocities_[i].y,
            impl_->host_velocities_[i].z,
        };
        const auto& position = final_state.bodies[i].position_m;
        const auto& velocity = final_state.bodies[i].velocity_m_s;
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
            !std::isfinite(velocity.y) || !std::isfinite(velocity.z)) {
            throw std::runtime_error("CUDA integration produced non-finite state");
        }
    }
    const auto staging_out_stop = std::chrono::steady_clock::now();

    CudaRunTimings timings;
    timings.allocation_ms = impl_->allocation_ms_;
    timings.host_staging_ms = std::chrono::duration<double, std::milli>(
        (staging_in_stop - staging_in_start) + (staging_out_stop - staging_out_start)).count();
    timings.host_to_device_ms = elapsed_ms(impl_->total_start_, impl_->after_h2d_);
    timings.device_compute_ms = elapsed_ms(impl_->after_h2d_, impl_->after_compute_);
    timings.device_to_host_ms = elapsed_ms(impl_->after_compute_, impl_->after_d2h_);
    timings.transfer_inclusive_ms = elapsed_ms(impl_->total_start_, impl_->after_d2h_);
    timings.backend_wall_ms = std::chrono::duration<double, std::milli>(
        staging_out_stop - backend_wall_start).count();
    return {std::move(final_state), timings};
}

CudaDeviceInfo CudaVelocityVerlet::device_info() const { return impl_->device_info_; }
double CudaVelocityVerlet::allocation_ms() const noexcept { return impl_->allocation_ms_; }

}  // namespace nbody
