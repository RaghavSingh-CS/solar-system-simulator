# Architecture

## Scientific model

- Newtonian point masses in an inertial barycentric Cartesian frame.
- SI base units: kilograms, metres, seconds. Output records units explicitly in its header.
- Pair acceleration: `a_i = G * sum(j != i) m_j (r_j-r_i) / |r_j-r_i|^3`.
- The direct-force implementations and Stage 1 sequential Barnes--Hut solver
  use no gravitational softening; coincident bodies are rejected. Barnes--Hut
  is opt-in in the headless runner and viewer and does not weaken the viewer
  workload gates.
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
                        |                  +-- Sequential/OpenMP Barnes--Hut (implemented, opt-in)
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

`SequentialBarnesHutForceBackend` and `OpenMpBarnesHutForceBackend` are
optional force solvers using the same acceleration contract and unchanged
Velocity Verlet integrator. Their shared
index-based 3D octree uses stable octant partitioning, contiguous body ranges,
monopole centers of mass, explicit target-range membership, and the standard
`s / d < theta` opening criterion. Nodes containing the target are always
opened. Checked node/memory budgets plus leaf occupancy, maximum depth, and
minimum-cell termination prevent unbounded subdivision. Backend and solver are
separate headless choices. Stage 2 builds one tree per force call, then the
OpenMP backend reads it concurrently while statically partitioning independent
target traversals. Each target preserves Sequential child/leaf order and owns
its compensated accumulator and output. Per-thread padded integer counters are
merged after the region; exceptions are captured in-region and reported only
outside it, with failed outputs/statistics discarded. Direct remains default.
CUDA Barnes--Hut state advancement is implemented as the separately selected
Stage 3B path. Stages 4A and 4B connect the reviewed Sequential/OpenMP solver
to the viewer and paired counterfactual engine; CUDA/Barnes--Hut remains
unavailable in both graphical paths. Full policy and evidence are in
[barnes_hut.md](barnes_hut.md).

CUDA Barnes--Hut Stage 3A is isolated as a force-only source path. The shared
CPU builder exports a fixed-width, pointer-free packed tree after construction;
the CUDA library transfers that immutable tree and evaluates one target per
logical GPU work item. Nodes retain deterministic child order, leaf permutation
ranges, and inverse-rank containment so the device can exclude self force and
must open any node containing the target. A bounded local traversal stack,
strict global work reservation, device failure record, separate device/host
budgets, and terminal quarantine on unproven stream quiescence define the
resource contract. Failure metadata is written by one atomic claimant and
becomes visible only through a fenced atomic publication word; all concurrent
device readers use that publication word atomically. Live capacity replacement
checks every deallocation and terminalizes without losing retained handles or
accounting if a release fails. No CUDA tree construction, Verlet integration,
viewer, counterfactual, or automatic-selection connection is part of Stage 3A.
Implementation `5e1864e` passed independent adversarial source review; CUDA
compilation and runtime behavior remain NVIDIA-unverified.

Stage 3B adds `CudaBarnesHutVelocityVerlet`, a transactional host orchestrator
around that exact Stage 3A evaluator. It value-copies the input state, performs
old-state force evaluation and drift, then performs a second force evaluation
which necessarily rebuilds the deterministic tree from the drifted positions,
and finally applies the average-acceleration kick. A noexcept move publishes
the candidate only after finite-state, identity/mass, work, and memory checks.
First- and second-force failure phases remain distinguishable; last-success
statistics are not overwritten on failure. Nonterminal scientific/resource
errors permit retry, while terminal Stage 3A runtime/ownership errors prohibit
reuse. Per-evaluation traversal bounds remain unchanged and the per-step bound
is their checked two-call sum. Integration scratch has its own host budget in
addition to Stage 3A device and pinned/tree-staging budgets.
Post-force output consistency, host-capacity, and combined-work checks publish
a separate integration post-force-validation phase rather than misclassifying
a successful force evaluation.
Only a separately compiled CUDA test-library variant contains deterministic
hooks for those otherwise allocator/invariant-dependent defensive checks; the
production API and binary contain no integration test hook.
Implementation `2cc2cb0` passed independent source-level review; CUDA
translation-unit compilation and runtime behavior remain NVIDIA-unverified.

OpenMP is optional at configure time. `NBODY_ENABLE_OPENMP=OFF` builds the
reference, headless runner, and Sequential Barnes--Hut benchmark without
finding or linking OpenMP; attempts to select either OpenMP direct or OpenMP
Barnes--Hut fail explicitly. When enabled, CMake requires
`OpenMP::OpenMP_CXX`, builds the direct and Barnes--Hut backends plus their
focused tests/benchmark, and emits a targeted configuration error if OpenMP is
unavailable.

## Deterministic synthetic scenario

`make_seeded_synthetic_system` uses `std::mt19937_64` raw output with an explicit 53-bit integer-to-double mapping, avoiding implementation-defined standard-library distributions. It generates generic SI-valued bodies, then removes center-of-mass position and velocity. It is a repeatable scaling/validation input, not a physical Solar System model.

## Large-N Laboratory L1

The separate `nbody_large_n` library owns deterministic laboratory generation,
checked preflight, measurement records, numerical comparison, and canonical CSV
serialization. `nbody_large_n_lab` is a headless single-configuration frontend;
the Python pilot orchestrator composes a bounded matrix and enforces a hard
per-process timeout. Neither target is linked into the viewer or frozen Stage 5
benchmark.

The generator uses a fixed SplitMix64 stream, explicit seed, stable ordered
body names, distribution-specific geometry, duplicate-position detection, and
finite center-of-mass recentering. Backend and solver remain orthogonal. The
laboratory constructs only existing Direct or Barnes--Hut backends and calls the
unchanged `velocity_verlet_step`; CUDA is represented only as a rejected L1
configuration.

Admission computes checked phase-specific high-water values for generation,
Direct reference, warm-up, retained repetitions, final characterization, and
serialization, including simultaneous states/arrays/containers and the full
overlapping Barnes--Hut memory budget. This is conservative policy accounting,
not an allocator-exact RSS bound. Direct/reference force calls and pairwise
diagnostic work are separately bounded with checked integer arithmetic. The
C++ boundary timer stops further batches after a budget violation, including a
final post-characterization gate, while the orchestrator supplies the hard
timeout for a force call that cannot be interrupted safely. Every published
scientific/timing value must be finite; failed runs clear measurements instead
of publishing partial rows.

The Python runner constructs and verifies a complete immutable run in a unique
scratch directory. An OS-backed interprocess lock serializes migration and
publication. The runner installs the run under `runs/<run-id>/`, then atomically
replaces a small root pointer whose current/history descriptors bind the run
manifests and every artifact hash. It never swaps or deletes the authoritative
evidence directory. A post-install/pre-pointer crash is a detectable orphan,
not partial authority, and requires explicit recovery. The read-only verifier
enforces the exact matrix, nonnegative scientific/resource semantics, immutable
history descriptors, and exact summary reconstruction. Evidence custody is
independent of Stage 1/2 and Stage 5. Timed-region worker counts are available
for Barnes--Hut; protected OpenMP Direct is labeled configured/runtime-probed
rather than independently observed in-region.

Legacy directory-swap recovery is a separate administrative operation, never a
measurement side effect. It verifies exactly one backup, copies it into a
content-addressed immutable run, and atomically appends that descriptor to
history without changing an existing current descriptor. The original backup
remains independently recoverable. These guarantees cover cooperating-process
serialization, process termination, and atomic visibility of a successful
Windows pointer replacement. They do not certify persistence across sudden
power loss: flushing the temporary pointer does not provide a portable Python
guarantee that the containing directory entry or storage-device cache reached
durable media.

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
complete paired batches. Sequential/OpenMP select direct or Barnes--Hut
explicitly. Barnes--Hut owns distinct full/removed backend instances, so both
Velocity Verlet force evaluations build from the correct branch state and no
tree/statistics state is shared. Optional CUDA owns two separate
`CudaVelocityVerlet` objects with persistent
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

## Viewer Barnes--Hut execution model (Stage 4A)

The viewer controller owns one authoritative host `SimulationState` and treats
execution backend (`Sequential`, `OpenMP`, `CUDA`) and force solver (`Direct`,
`Barnes--Hut`) as independent configuration dimensions. The supported viewer
matrix is Sequential/direct, Sequential/Barnes--Hut, OpenMP/direct,
OpenMP/Barnes--Hut, and CUDA/direct. CUDA/Barnes--Hut fails explicitly pending
NVIDIA validation; no direct-force or CPU fallback is permitted.

CPU Barnes--Hut calls reuse the reviewed `ForceBackend` implementations and
the unchanged Velocity Verlet function. A requested frame/manual batch advances
a private state copy and publishes it only after every step succeeds. A force,
tree, work-budget, memory-budget, or finite-state failure pauses the controller,
retains the last committed state/time, and exposes a bounded error string.
Backend/solver replacement constructs replacement resources before changing
the effective configuration, so switching never resets time or leaves stale
accelerations. Reset and scenario replacement retain the selected solver while
restoring the scenario-defined initial state.

The main viewer loop is synchronous and UI-thread-owned; Stage 4A does not add
a simulation thread. The separate counterfactual worker continues to own an
immutable experiment snapshot and its existing scenario/worker generation
gates. Stage 4B captures the experiment's own direct/Barnes--Hut choice and
full theta/resource policy; viewer or next-run edits neither alias nor retarget
an in-flight experiment. Reset requests cooperative cancellation,
invalidates the accepted worker generation, and clears publication-derived
progress, result, warning, frame, and overlay caches before restoring the live
state; delayed progress or terminal completion therefore cannot republish.
Sequential/OpenMP Barnes--Hut counterfactual execution is supported;
CUDA/Barnes--Hut remains unavailable and GPU-unverified.

Stage 4A implementation `c9e9500` passed independent source-level review and
local CPU/OpenMP validation on 2026-10-14. CUDA/Barnes--Hut viewer execution
remains disabled and GPU-unverified pending NVIDIA validation.

## Counterfactual Barnes--Hut execution model (Stage 4B)

Implementation `2ec090d` passed the final focused independent source-level
commit gate and canonical CPU/OpenMP validation on 2026-10-14. CUDA compilation,
NVIDIA execution, and human visual approval are not claimed.

`CounterfactualExperimentConfiguration` adds a solver dimension without
changing its direct default. The runner constructs replacement resources
before execution: direct Sequential uses the stateless reference backend,
direct OpenMP owns two configured backends, and each CPU Barnes--Hut choice
owns one backend per branch. Both branches share only immutable configuration,
never an octree. They use the same gravitational constant, fixed timestep,
Velocity Verlet function, theta/policy, batch boundary, and scientific guards.

The candidate result is still a value copy of the last committed pair. A full-
branch or removed-branch exception leaves both committed states, histories,
metrics, conservation values, warnings, counters, and prior branch statistics
unchanged; terminal failure metadata remains separate. Successful steps copy
the two backends' most recent final-position force statistics into the candidate
before atomic batch publication. Stable-ID compaction and matched-time baseline
reference positions are unchanged.

A non-owning force-backend proxy surrounds each CPU branch only while the
unchanged Velocity Verlet function runs. It records whether the active real
backend call is the old-position or new-position force evaluation, without
duplicating the integrator equations. Terminal records carry typed optional
branch and phase metadata; candidate preparation, solver-statistic collection,
post-step scientific validation, history construction, conservation/summary
evaluation, and commit are labeled only at their actual call sites. Operations
whose branch or phase cannot be established retain an explicit unspecified
value rather than inferring context from exception text.

Fault injection is not part of that production proxy contract. The normal
`nbody_experiments` target compiles without test-hook declarations, state, or
branches. Only `nbody_experiments_test_hooks` defines the guarded test ABI, and
only the focused counterfactual engine tests link that variant. The viewer,
visualization model, and worker tests link the production target, preventing a
runtime configuration from enabling deterministic test failures.

The asynchronous worker copies solver and Barnes--Hut settings into its
immutable run descriptor before scheduling. Publications, stale-generation
gates, cancellation, overlays, and result tables continue to consume that
descriptor rather than mutable GUI controls. The UI reports backend, solver,
theta, hard tree limits, and per-branch last-force statistics. It also states
that Barnes--Hut-versus-direct discrepancy and physical removal response are
not separable within a single approximate run. This notice is computed from the
generation-gated immutable run descriptor rather than the bounded warning
buffer, and therefore survives warning saturation and live control edits while
remaining suppressible on reset or scenario replacement. CUDA/Barnes--Hut is rejected;
no CPU or direct-force fallback is present.

## Data provenance

Stage 6B's approximate Solar demonstration uses JPL's published 1800--2050 mean orbital elements at J2000.0, derives velocities from the same model, and uses NASA factsheet masses/radii. It is explicitly not an ephemeris. A future ephemeris-grade scenario will use JPL Horizons Cartesian vector tables; its committed source record must include query, epoch, TDB/TT choice, reference plane/frame, center, units, Horizons version, retrieval time, and raw response.
