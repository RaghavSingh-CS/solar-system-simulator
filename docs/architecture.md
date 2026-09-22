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
                        |                  +-- OpenMP (planned)
                        |                  +-- CUDA (planned)
                        |                  +-- MPI (optional)
                        +-> diagnostics -> CSV/headless benchmark
                        +-> immutable snapshots -> OpenGL viewer
```

- `include/nbody`: dependency-free domain types and public APIs.
- `src`: sequential backend, integrator, scenarios, diagnostics, CSV runner.
- `backends/openmp`: Phase 2 backend.
- `backends/cuda`: Phase 3 `.cu` kernels and Colab validation material.
- `backends/mpi`: optional stretch implementation.
- `visualization`: independent viewer; it may link the simulation library but simulation code never depends on OpenGL.
- `tests`: physics invariants and regression tests.

## Shared representation and determinism

`Body` stores a stable name, mass, position, and velocity in double precision. The reference backend iterates `i` then `j` in ascending index order. Backends share input/output contracts but parallel reductions need only meet declared numerical tolerances, not bitwise identity.

## Backend interface plan

The first stage keeps acceleration calculation as a replaceable function. Phase 2 will formalize `IForceBackend::compute_accelerations(state, parameters)` without changing integrator/scenario/diagnostic APIs. Backend selection will sit above this interface and will never alter physics parameters to make a backend pass.

## Visualization dependency decision

Use GLFW plus core OpenGL, with GLAD (or another generated loader) vendored or fetched only in a pinned optional build. GLFW is small, cross-platform, and CMake-friendly on Windows. Visualization is `OFF` by default until dependencies are pinned; the headless library/tests have no graphics dependency. ImGui is not required for the MVP.

## CUDA build split

- Default CMake project declares only C++ and succeeds without a CUDA toolkit.
- A later `NBODY_ENABLE_CUDA=ON` path will use `check_language(CUDA)` and fail with a targeted message only when explicitly requested.
- Colab validates compilation and correctness against exported sequential fixtures; its performance numbers are reported only for that Colab GPU.
- Final crossover calibration is rerun end-to-end on the RTX 4050. No local CPU/Colab GPU timing mixture is permitted.

## Data provenance

The realistic scenario will use JPL Horizons Cartesian vector tables. The committed source record must include query, epoch, TDB/TT choice, reference plane/frame, center, units, Horizons version, retrieval time, and raw response. Until that exists, only the labeled analytic binary is shipped.
