# Architecture

## Scientific model

- Newtonian point masses in an inertial barycentric Cartesian frame.
- SI base units: kilograms, metres, seconds. Output records units explicitly in its header.
- Pair acceleration: `a_i = G * sum(j != i) m_j (r_j-r_i) / |r_j-r_i|^3`.
- The direct-force implementations and Stage 1 sequential Barnes--Hut solver
  use no gravitational softening; coincident bodies are rejected. Barnes--Hut
  is opt-in in the headless runner and does not change the viewer workload
  gates.
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
                        |                  +-- MPI / MPI+OpenMP (implemented, optional)
                        |                  +-- sequential Barnes--Hut (implemented, opt-in)
                        +-> diagnostics -> CSV/headless benchmark
                        +-> immutable counterfactual snapshot
                              -> paired Sequential/OpenMP/CUDA runner
                              -> bounded published metrics
                        +-> immutable snapshots -> OpenGL viewer
```

- `include/nbody`: dependency-free domain types and public APIs.
- `src`: sequential backend, integrator, scenarios, diagnostics, CSV runner.
- `backends/openmp`: optional outer-loop-parallel OpenMP backend.
- `backends/cuda`: optional direct-force CUDA kernels and device-resident integrator.
- `backends/mpi`: optional replicated-state MPI implementation with an OpenMP local-work mode.
- `visualization`: optional application controller, camera, render-space conversion, and OpenGL renderer; it links the simulation library but simulation code never depends on OpenGL.
- `tests`: physics invariants and regression tests.

The headless `nbody_experiments` library owns counterfactual snapshots and a
paired Sequential/OpenMP/optional-CUDA runner. It copies source state, removes exactly one
curated non-Sun body from only the counterfactual branch, and reuses the shared
Velocity Verlet/force APIs. Completed pairs are published transactionally with
bounded trajectory samples; it has no OpenGL dependency and no live-viewer
state alias. The optional viewer transfers exclusive runner ownership to one
persistent worker thread. The render thread submits a value-owned snapshot,
requests cancellation through an atomic flag, and reads mutex-protected
immutable publications only at paired batch boundaries. Progress publications
contain lightweight counters; the bounded full result is published once at a
terminal boundary through immutable shared ownership. Window destruction joins
the worker, and scenario changes
request cancellation without mutating either paired branch. At the application
boundary, independent monotonic scenario and worker generations gate every
publication; stable IDs remain a secondary composition check. Each publication
shares a value-owned immutable descriptor of the executed snapshot,
configuration, and admission estimates, so editable controls are not result
metadata. Local comparison views retain bounded, baseline-only matched-time
origin samples separately from monitored metrics and use that one origin for
both paths; unavailable origins explicitly suppress rather than reframe the
overlay. Per-step proximity
screening is a branch-specific linear swept-segment test and is not a curved-
trajectory minimum-distance proof. See
[counterfactual_experiments.md](counterfactual_experiments.md) for metrics,
resource policies, scientific guards, viewer ownership, and CUDA validation work.

## Shared representation and determinism

`Body` stores a stable name, mass, position, and velocity in double precision. `ForceBackend` owns the acceleration contract; the integrator accepts any backend, while the original overload still selects the sequential reference. The reference backend iterates `i` then `j` in ascending index order. Backends share input/output contracts but parallel implementations need only meet declared numerical tolerances, not bitwise identity.

## Backend interface plan

The interface is `ForceBackend::compute_accelerations(state, G)`. `sequential_force_backend()` wraps the original reference function. `OpenMpForceBackend(thread_count)` uses `schedule(static)` over destination bodies; each loop iteration owns its local accumulator and one output element, so there are no concurrent accumulator writes. Every destination body still visits source bodies in ascending index order. Invalid masses are rejected before the parallel region; coincident/invalid positions are flagged within the region and reported afterward.

`CudaForceBackend` implements the same interface with one CUDA thread per destination body and ascending source traversal. `CudaVelocityVerlet` adds a reusable device-resident execution path: old acceleration, position update, new acceleration, and velocity update remain on the GPU for every step. This avoids presenting a force-only kernel as complete integration timing while retaining the shared force contract for correctness tests.

`MpiForceBackend` also implements the shared interface. Every rank holds the same complete state; a deterministic contiguous partition assigns destination bodies, with any remainder assigned to lower ranks. Each rank computes only its local acceleration slice, then `MPI_Allgatherv` reconstructs the complete acceleration vector on every rank. An `MPI_Allreduce` propagates invalid-input status before that exchange. Consequently both force evaluations in Velocity Verlet see globally consistent positions and every rank applies identical drift/kick updates. Memory is O(N) per rank, local force work is O(N^2/P), and each force evaluation communicates O(N) acceleration data. The optional hybrid mode applies `schedule(static)` OpenMP parallelism only inside the local target loop; MPI calls occur outside parallel regions under `MPI_THREAD_FUNNELED`. A future one-rank-per-GPU design is only a conceptual extension, not an implemented or measured path.

`SequentialBarnesHutForceBackend` is a separate optional force solver using the
same acceleration contract and unchanged Velocity Verlet integrator. Its
index-based 3D octree uses stable octant partitioning, contiguous body ranges,
monopole centers of mass, explicit target-range membership, and the standard
`s / d < theta` opening criterion. Nodes containing the target are always
opened. Checked node/memory budgets plus leaf occupancy, maximum depth, and
minimum-cell termination prevent unbounded subdivision. Backend and solver are
separate headless choices: Stage 1 accepts only `sequential + barnes-hut` and
rejects unsupported combinations without fallback. Direct remains default;
OpenMP/CUDA Barnes--Hut and viewer integration remain future stages. Full
policy and evidence are in [barnes_hut.md](barnes_hut.md).

OpenMP is optional at configure time. `NBODY_ENABLE_OPENMP=OFF` builds the reference, headless runner, and sequential benchmark without finding or linking OpenMP. When explicitly enabled, CMake requires `OpenMP::OpenMP_CXX` and emits a targeted configuration error if unavailable.

## Deterministic synthetic scenario

`make_seeded_synthetic_system` uses `std::mt19937_64` raw output with an explicit 53-bit integer-to-double mapping, avoiding implementation-defined standard-library distributions. It generates generic SI-valued bodies, then removes center-of-mass position and velocity. It is a repeatable scaling/validation input, not a physical Solar System model.

## Visualization architecture and dependencies

`NBODY_BUILD_VISUALIZATION=OFF` remains the default. When enabled, CMake fetches exact revisions of GLFW 3.4, GLM 1.0.1, and Dear ImGui 1.92.9b. A reproducibly generated GLAD 2.0.6 OpenGL 3.3 loader is committed under `third_party/glad`, avoiding a build-time Python/Jinja2 dependency and runtime loader DLL. Shaders are embedded.

The dependency direction is one-way:

```text
SimulationState + ForceBackend
          -> SimulationController (fixed physical dt, scenario/reset/manual backend)
          +-> diagnostics + bounded physical-space trails
          +-> documented mean-element reference guides (presentation only)
          -> read-only RenderBody snapshot (linear position scale/material/radius only)
          -> BodyRenderer (Solar meshes; batched large-N points; line overlays)
          -> ImGui controls/labels (read state; issue controller/view commands)
```

`SimulationController` calls the existing CPU `velocity_verlet_step` for Sequential/OpenMP and the existing batched `CudaVelocityVerlet` executor for CUDA; no gravitational calculation exists in renderer code. CUDA allocations are reused while the body count is unchanged, and each requested frame batch completes its device integration and returns an authoritative host state before diagnostics or rendering. There is no CUDA/OpenGL interoperability. The application processes events, advances a named integer count of fixed simulation steps, samples physical state, prepares a snapshot, renders, and swaps buffers. Render frame duration is not passed to the integrator. Trails remain in physical coordinates until rendering, and display radii/materials/rings never feed back into physics. Static orbit guides are sampled from the same documented JPL mean-element model but never drive body motion. Camera presets, label decluttering, health mapping, scenario switching, diagnostics, trails, and render transformation are independently headless-testable. A later renderer can replace per-body sphere draws with points or instancing without changing physics.

The headless counterfactual runner owns two private branches and publishes only
complete paired batches. Sequential/OpenMP use the existing force backend;
optional CUDA owns two separate `CudaVelocityVerlet` objects with persistent
full/compacted allocations. Each CUDA step returns both synchronized host
states before either enters the candidate result. CUDA calls are serialized,
not placed in simultaneous streams. Failed CUDA or post-integration scientific
validation invalidates both CUDA executors so the runner cannot resume from a
speculative branch. Destruction synchronizes each owning stream before freeing
stream-referenced storage. A failed synchronization does not prove quiescence,
so storage is conservatively quarantined until context/process teardown rather
than forcing a device reset shared with the live viewer. Every non-quiescent
executor can retain another complete device/pinned allocation set, so repeated
fatal failures can accumulate memory until teardown. GPU experimentation must
stop after a genuine unrecoverable CUDA error; record diagnostics and restart
the application process rather than assuming recovery. Failed-batch metadata uses
a separate fixed-capacity terminal record, leaving the last committed states,
metrics, histories, conservation values, warnings, and warning-drop count
unchanged. Its between-step close-approach screen uses common-scale binary64
relative endpoints and stable norms, rejecting nonrepresentable geometry; the
straight swept segment remains only an approximation to curved gravitational
motion. The runner remains single-owner and is transferred exclusively to the
viewer worker described above; the render thread never calls runner methods.

The viewer uses the existing bounded physical trail store for the Sun's actual
barycentric trajectory and applies optional amplification only to its rendered
displacement from the origin. Stage 6D replaces the fixed Sun-radius cap with
camera-aware screen-space sizing: apparent radii are smoothly bounded and the
projected Sun/Mercury pair is reduced as needed to preserve visible center
separation. Body centers and physical state never move. Fixed screen slots and
leader lines provide stable Sun-through-Mars callouts while outer labels retain
dynamic decluttering.

Compute Experiment uses sphere draws through N=64 and one reusable dynamic
point-sprite draw above N=64. Supported sizes extend through 100,000, but the
direct O(N^2) solver is deliberately bounded: continuous advancement and live
conservation/trails stop above 1024, manual one-step execution is limited to
OpenMP through 8192 (and Sequential/OpenMP at 2048), and larger cases are
display/inspection-only. Synthetic labeling is selected-body-only and no
per-body UI list is built. Its live CPU comparison is a separate model operation
over two identical state copies; it cannot feed Stage 5 calibration or Auto
selection.

The ImGui panel owns 380--680 pixels, capped at 55% of the current display. The
camera aspect, projection, picking, and labels use the complementary scene
viewport, so 3D content is never composed beneath the panel. A system sans-serif
font is used when available, with Dear ImGui's built-in font as a portable
fallback; no runtime font download is required.

## CUDA build split

- Default CMake project declares only C++ and succeeds without a CUDA toolkit.
- `NBODY_ENABLE_CUDA=ON` uses `check_language(CUDA)` and `enable_language(CUDA)` only when requested, and fails with a targeted message if no toolkit is usable.
- CUDA uses `double` throughout, direct O(N²) interactions, reusable allocations, explicit error checking, and no mixed precision or approximate algorithm.
- CUDA-enabled builds link the headless counterfactual runner to the same
  `nbody_cuda` wrapper; CPU-only builds neither link CUDA nor advertise the
  backend. Device-dependent counterfactual tests explicitly skip with code 77
  when no usable NVIDIA device exists.
- Colab must validate compilation and correctness against the sequential reference; its performance numbers are reported only for that Colab environment.
- Final crossover calibration is rerun end-to-end on the RTX 4050. No local CPU/Colab GPU timing mixture is permitted.

## Data provenance

Stage 6B's approximate Solar demonstration uses JPL's published 1800--2050 mean orbital elements at J2000.0, derives velocities from the same model, and uses NASA factsheet masses/radii. It is explicitly not an ephemeris. A future ephemeris-grade scenario will use JPL Horizons Cartesian vector tables; its committed source record must include query, epoch, TDB/TT choice, reference plane/frame, center, units, Horizons version, retrieval time, and raw response.
