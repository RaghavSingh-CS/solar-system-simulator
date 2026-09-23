# Architecture

## Scientific model

- Newtonian point masses in an inertial barycentric Cartesian frame.
- SI base units: kilograms, metres, seconds. Output records units explicitly in its header.
- Pair acceleration: `a_i = G * sum(j != i) m_j (r_j-r_i) / |r_j-r_i|^3`.
- The sequential reference uses no gravitational softening; coincident bodies are rejected. Synthetic large-N backends may later expose softening as an explicit experiment parameter, never silently.
- `G = 6.67430e-11 m^3 kg^-1 s^-2`, the 2022 CODATA recommended value (unchanged numerically from 2018), sourced from [NIST](https://physics.nist.gov/constants).

## Integrator decision

Velocity Verlet (kick-drift-kick form) is the Phase 1 reference: it is deterministic, time-reversible, second-order, symplectic for separable Newtonian Hamiltonians, and needs one new acceleration evaluation per step after initialization. Its bounded long-run energy behavior is preferable to forward Euler for a teaching/reference implementation.

This does **not** establish suitability for Gyr Solar System prediction. Before making long-horizon planetary claims, compare against a Wisdom–Holman-class integrator and document close-encounter limitations.

## Components

```text
scenario/data -> SimulationState -> IForceBackend -> VelocityVerlet
                        |                  |
                        |                  +-- sequential (reference)
                        |                  +-- OpenMP (implemented)
                        |                  +-- CUDA (validated on Tesla T4)
                        |                  +-- MPI (optional)
                        +-> diagnostics -> CSV/headless benchmark
                        +-> immutable snapshots -> OpenGL viewer
```

- `include/nbody`: dependency-free domain types and public APIs.
- `src`: sequential backend, integrator, scenarios, diagnostics, CSV runner.
- `backends/openmp`: optional outer-loop-parallel OpenMP backend.
- `backends/cuda`: optional direct-force CUDA kernels and device-resident integrator.
- `backends/mpi`: optional stretch implementation.
- `visualization`: independent viewer; it may link the simulation library but simulation code never depends on OpenGL.
- `tests`: physics invariants and regression tests.

## Shared representation and determinism

`Body` stores a stable name, mass, position, and velocity in double precision. `ForceBackend` owns the acceleration contract; the integrator accepts any backend, while the original overload still selects the sequential reference. The reference backend iterates `i` then `j` in ascending index order. Backends share input/output contracts but parallel implementations need only meet declared numerical tolerances, not bitwise identity.

## Backend interface plan

The interface is `ForceBackend::compute_accelerations(state, G)`. `sequential_force_backend()` wraps the original reference function. `OpenMpForceBackend(thread_count)` uses `schedule(static)` over destination bodies; each loop iteration owns its local accumulator and one output element, so there are no concurrent accumulator writes. Every destination body still visits source bodies in ascending index order. Invalid masses are rejected before the parallel region; coincident/invalid positions are flagged within the region and reported afterward.

`CudaForceBackend` implements the same interface with one CUDA thread per destination body and ascending source traversal. `CudaVelocityVerlet` adds a reusable device-resident execution path: old acceleration, position update, new acceleration, and velocity update remain on the GPU for every step. This avoids presenting a force-only kernel as complete integration timing while retaining the shared force contract for correctness tests.

OpenMP is optional at configure time. `NBODY_ENABLE_OPENMP=OFF` builds the reference, headless runner, and sequential benchmark without finding or linking OpenMP. When explicitly enabled, CMake requires `OpenMP::OpenMP_CXX` and emits a targeted configuration error if unavailable.

## Deterministic synthetic scenario

`make_seeded_synthetic_system` uses `std::mt19937_64` raw output with an explicit 53-bit integer-to-double mapping, avoiding implementation-defined standard-library distributions. It generates generic SI-valued bodies, then removes center-of-mass position and velocity. It is a repeatable scaling/validation input, not a physical Solar System model.

## Visualization dependency decision

Use GLFW plus core OpenGL, with GLAD (or another generated loader) vendored or fetched only in a pinned optional build. GLFW is small, cross-platform, and CMake-friendly on Windows. Visualization is `OFF` by default until dependencies are pinned; the headless library/tests have no graphics dependency. ImGui is not required for the MVP.

## CUDA build split

- Default CMake project declares only C++ and succeeds without a CUDA toolkit.
- `NBODY_ENABLE_CUDA=ON` uses `check_language(CUDA)` and `enable_language(CUDA)` only when requested, and fails with a targeted message if no toolkit is usable.
- CUDA uses `double` throughout, direct O(N²) interactions, reusable allocations, explicit error checking, and no mixed precision or approximate algorithm.
- Colab must validate compilation and correctness against the sequential reference; its performance numbers are reported only for that Colab environment.
- Final crossover calibration is rerun end-to-end on the RTX 4050. No local CPU/Colab GPU timing mixture is permitted.

## Data provenance

The realistic scenario will use JPL Horizons Cartesian vector tables. The committed source record must include query, epoch, TDB/TT choice, reference plane/frame, center, units, Horizons version, retrieval time, and raw response. Until that exists, only the labeled analytic binary is shipped.
