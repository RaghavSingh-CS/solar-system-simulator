# Workload-Aware Parallel N-Body Simulation

A research-oriented C++17 project for comparing sequential, OpenMP, CUDA, and MPI direct N-body execution under explicit numerical-accuracy constraints. The repository contains deterministic sequential and OpenMP backends, an optional CUDA backend validated on an NVIDIA Tesla T4, and an optional replicated-state MPI backend with a clean MPI+OpenMP hybrid mode.

The literature review found that CPU/GPU crossover and adaptive heterogeneous mapping are established topics. Accordingly, this project does **not** claim a new scheduling algorithm. Its proposed study asks whether a small, calibrated selector can find the fastest backend on one target machine while rejecting configurations that violate predeclared conservation bounds. See [novelty assessment](docs/research/novelty_assessment.md).

## Current capabilities

- Newtonian direct O(N^2) force evaluation in double precision plus an
  accuracy-first 3D Barnes--Hut monopole solver with Sequential and OpenMP
  execution.
- Shared force-backend interface with sequential and OpenMP implementations.
- SI units and barycentric Cartesian coordinates.
- Fixed-step Velocity Verlet integration.
- Reproducible analytic circular two-body scenario and a nine-body approximate Solar System demonstration initialized from published JPL mean orbital elements.
- Deterministic seeded multi-body scenarios for validation and CPU scaling tests.
- Energy, linear momentum, angular momentum, and center-of-mass diagnostics.
- Headless CSV output and dependency-free tests.
- Computation-only benchmark plus a controlled, block-randomized CPU experiment runner with raw and summary CSV output.
- Optional double-precision direct CUDA backend, GPU-resident Velocity Verlet executor, and validated dual-regime CUDA benchmark.
- Reproducible sequential conservation-study harness with retained timestep-duration evidence.
- Prepared Stage 5 v2.0 phase-separated runner, deterministic policy/freezer, independent evidence verifier, and source/build/test provenance chain; no Stage 5 timings collected yet.
- Optional replicated-state MPI backend with deterministic target partitioning, global acceleration exchange, and MPI+OpenMP local force computation.
- Optional presentation-ready GLFW/OpenGL 3.3 + ImGui viewer with Solar/Compute Experiment modes, deterministic camera presets, stable inner-system callouts, camera-aware body sizing, reference/history/Sun-motion trails, barycenter marker, procedural planet identity, batched large-N points through 100,000 bodies, live numerical-health bounds, and an isolated CPU-backend comparison.

Stage 4 real-GPU validation and the controlled 154-sample CUDA matrix are complete; evidence is under `data/benchmarks/stage4_cuda/`. No same-machine CPU/OpenMP/CUDA crossover has been measured, so no automatic selector is implemented. Stage 6D completes the MPI, large-N infrastructure, and final presentation feature scope; native-window appearance and interaction still require the final user visual pass. Ephemeris-grade JPL Horizons initial conditions remain outside this stage.

An independent 30-configuration sequential conservation characterization is
documented in `docs/conservation_study.md`, with evidence under
`data/experiments/conservation/`. It is not a backend benchmark or Stage 5.

Stage 5 v2.0 pre-RTX remediation is locally validated. The exact candidate SHA
is delivered by an external post-commit deployment packet. The RTX 4050 pilot,
calibration, policy freeze, held-out acquisition, and evaluation remain
unexecuted; follow `docs/stage5_rtx4050_runbook.md` and stop the first NVIDIA
rehearsal after pilot verification and manifest-bound approval.

## Build and test

All ordinary local builds use the tracked presets and the ignored `build/`
tree. The CPU preset is the default on this AMD Ryzen laptop and has no NVIDIA
dependency:

```powershell
cd C:\Projects\PCAP\solar-system-simulator
.\scripts\test-local.ps1 -Configuration CPU
```

Build and test the validated GCC/libgomp configuration explicitly with:

```powershell
.\scripts\test-local.ps1 -Configuration OpenMP
```

These commands configure, build, and test `build/cpu-release/` and
`build/openmp-release/`. For direct use, the equivalent commands are
`cmake --preset <name>`, `cmake --build --preset <name>`, and
`ctest --preset <name>`. The repository-local script prefers the verified
portable CMake under `.tools/` when present and otherwise uses `PATH`.

### Optional interactive viewer

The canonical presets build the viewer. After the CPU build, launch it with:

```powershell
.\scripts\run-visualizer.ps1
```

OpenMP is never selected implicitly:

```powershell
.\scripts\run-visualizer.ps1 -Backend OpenMP
```

Forward application options as a string array, for example:

```powershell
.\scripts\run-visualizer.ps1 -Backend OpenMP -ApplicationArguments '--threads','4','--scenario','catalog'
```

The viewer defaults to the existing direct-force path. Sequential and OpenMP
can select the reviewed CPU Barnes--Hut evaluator independently, either from
the ImGui `Force Solver` section or at launch:

```powershell
.\scripts\run-visualizer.ps1 -Backend OpenMP -ApplicationArguments '--threads','4','--solver','barnes-hut','--theta','0.25'
```

Theta edits are drafts until explicitly applied. The panel reports the
effective backend, solver, theta, tree/traversal/memory bounds, and last
successful force statistics. CUDA/direct remains available as before;
CUDA/Barnes--Hut is rejected rather than falling back until NVIDIA validation.
Stage 4A viewer integration is committed as `c9e9500` and passed independent
local CPU/OpenMP source-level review; CUDA/Barnes--Hut remains GPU-unverified.
Stage 4B counterfactual integration is committed as `2ec090d` and passed its
focused independent source-level gate plus canonical CPU/OpenMP validation.
CUDA/Barnes--Hut counterfactual execution remains unavailable and
GPU-unverified.

On a machine with a compatible CUDA toolkit and NVIDIA device, run
`.\scripts\test-local.ps1 -Configuration CUDA` and then
`.\scripts\run-visualizer.ps1 -Backend CUDA -ApplicationArguments '--device','0'`.
CUDA remains optional, is never selected automatically, and is never silently
replaced by a CPU backend.

The first visualization configure requires network access to fetch exact pinned revisions of GLFW 3.4, GLM 1.0.1, and Dear ImGui 1.92.9b. The generated GLAD 2.0.6 OpenGL 3.3 loader is committed under `third_party/glad`, so Python, Jinja2, copied DLLs, and runtime asset discovery are not required. Shaders are embedded in the executable.

The default remains the unchanged nine-body Solar System demonstration at
J2000.0. Planet positions come from JPL's published 1800--2050 approximate
mean-element model; centered finite differences of that same model provide
initial velocities, then the state is shifted to its center of mass. This is a
deterministic educational approximation, not a Horizons ephemeris. Its physical
timestep remains fixed at six hours.

`--scenario catalog` selects the separate `jpl-ssd-j2000-60-v1` sandbox: 60
individually identified, gravitationally active bodies initialized from the
committed geometric SSB-centered ICRF snapshot at JD 2451545.0 TDB. Its initial
fixed-step policy is a provisional 60 seconds. A one-day 300/150/75-second
study rejected 300 seconds for fast satellites (Metis was the worst case); 60
versus 30 seconds reduced the worst parent-relative final-position difference
to about `1.07e-3`. This
bounded convergence result is not long-horizon ephemeris validation. A
deterministic seeded `Compute Experiment` remains available with
`--scenario synthetic`.

Controls:

- The themed ImGui panel separates primary simulation/compute/health controls from collapsible technical details. Numerical health compares live drift against the unchanged `5e-6` energy and `5e-10` angular-momentum bounds.
- Backend and force solver are independent controls. Direct remains the
  default; Sequential/Barnes--Hut and OpenMP/Barnes--Hut use the same bounded
  CPU solver as the headless runner. The accuracy-first `theta=0.25` default is
  provisional and distribution-dependent, not a universal accuracy promise.
- Catalog composition checkboxes select gravitational inclusion; per-body and
  body-class visibility controls affect rendering only. Excluded major sources
  or parents produce explicit warnings.
- `Overview`, `Inner System`, `Earth-Moon`, the four giant-planet subsystems,
  `Pluto-Charon`, and `Asteroid Belt` use stable IDs and local double-precision
  render origins without changing SSB physics coordinates.
- Local subsystem trails subtract the matching historical parent position at
  each sample; labels, picking, guides, rings, and bodies use the same view
  transform.
- The `Counterfactual Experiment` panel captures either the current or initial
  astronomical state, removes one explicitly eligible non-Sun body from an
  independent paired run, and publishes bounded baseline/counterfactual
  trajectories from a dedicated worker. Visibility never changes solver
  inclusion. Scenario and worker generations suppress stale publications, and
  every publication retains the immutable executed configuration rather than
  later editor values. Direct remains the experiment default; Sequential and
  OpenMP can explicitly use Barnes--Hut with one captured theta/resource policy
  applied to two independent state-specific trees. The overlay uses a solid
  baseline and dashed altered
  path in one common frame; unmonitored local origins use separately retained
  matched-time baseline references, while unavailable origins suppress the
  overlay explicitly. Responsive controls keep labels separate from numeric
  fields at narrow panel widths. Results are organized into a run summary, a
  sortable stable-ID per-body table, grouped expandable scientific warnings,
  independent per-branch conservation diagnostics, and collapsible immutable
  provenance. Missing metrics are shown as `N/A`; terminal failures remain
  prominent. Sequential and OpenMP are available in their corresponding CPU
  builds. Barnes--Hut results warn that removal response and approximation
  error are not separated by one run; a matched Direct reference is required
  for interpretation. That notice is derived from the immutable executed job,
  so bounded diagnostic saturation or later control edits cannot hide or
  relabel it. Terminal failures report a typed branch and execution phase when
  the runner can establish them without guessing. The production force proxy
  retains that attribution but contains no fault-injection API or state;
  deterministic injection is compiled only into a separate library used by
  the focused engine tests and is never linked into the viewer or worker.
  CUDA appears only in a CUDA-enabled build with an
  available device;
  it owns separate persistent baseline/counterfactual allocations and never
  silently falls back to CPU. CUDA/Barnes--Hut counterfactual execution remains
  unavailable. Separation is not labeled as escape.
- Approximate reference orbit guides provide immediate context; fading history trails remain simulation-derived and planet positions always come from N-body integration.
- `Barycenter & real Sun motion` marks the barycentric origin and traces integrated Sun positions; an explicit presentation-only amplification control defaults to `10x`.
- Compute Experiment offers deterministic `N=32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,100000`; workloads above 64 use one batched point-sprite draw. Continuous simulation and live O(N^2) diagnostics stop above 1024, bounded manual stepping is available only through 8192, and larger workloads are explicitly display/inspection-only in the viewer.
- `Compare CPU Backends` times isolated identical Sequential/OpenMP state copies, checks established mixed correctness tolerances, and is explicitly a live diagnostic—not benchmark evidence.
- `Space`: pause/resume; `R`: exact scenario reset; `Esc`: exit.
- `1`: Sequential; `2`: OpenMP using `--threads N` when OpenMP was built; `3`:
  CUDA using `--device N` when CUDA was built and the device is available.
- Click a body to select it; focus selected, focus Sun/body 0, or return to overview from the panel.
- Left-drag, `A`/`D`, or arrow keys: orbit; mouse wheel or `W`/`S`: dolly; `C`: overview camera.

The panel is resizable from 380 px to the smaller of 680 px or 55% of the window, and the 3D scene viewport always occupies the remaining width. Hover help documents every primary control. CUDA is exposed only in CUDA-enabled builds with an available device; MPI remains a separate benchmark executable. Auto remains absent until Stage 5 produces a valid calibration. Viewer FPS/frame time is display-only and is not controlled benchmark evidence. See [the visualization notes](docs/visualization.md) for provenance, architecture, options, and limitations.
Automated layout/model tests do not replace manual native-window inspection;
the revised narrow-panel controls, results table, warning expansion, legends,
and full-system label density still require appearance validation.

Run (Visual Studio multi-config path shown):

```powershell
./build/cpu-release/nbody_headless.exe --steps 5000 --sample-every 50 --output orbit.csv
```

Use `--help` for options. CSV comment lines document scenario, units, integrator, gravitational constant, and time step; data columns are:

```text
time_s,body_index,name,mass_kg,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s
```

Direct force remains the headless default. Barnes--Hut is selected independently
from the execution backend and supports Sequential, explicitly configured
OpenMP, and optional CUDA execution:

```powershell
.\build\cpu-release\nbody_headless.exe --backend sequential --solver barnes-hut `
  --theta 0.25 --steps 5000 --sample-every 50 --output orbit-bh.csv

.\build\openmp-release\nbody_headless.exe --backend openmp --solver barnes-hut `
  --threads 8 --theta 0.25 --steps 5000 --sample-every 50 --output orbit-omp-bh.csv

# Requires a CUDA-enabled build and NVIDIA validation.
.\build\cuda-release\nbody_headless.exe --backend cuda --solver barnes-hut `
  --cuda-device 0 --theta 0.25 --steps 5000 --sample-every 50 --output orbit-cuda-bh.csv
```

The default accuracy-first policy uses `theta=0.25`, leaf capacity 8, maximum
depth 64, minimum cell size `1e-6 m`, one million nodes, a 256 MiB working
budget, and one billion visited-node/direct-interaction work units. Unsupported
backend/solver combinations fail explicitly. See
[Barnes--Hut Stages 1--3B](docs/barnes_hut.md) for shared-tree architecture,
acceptance semantics, numerical limits, exact Sequential/OpenMP agreement
tests, transactional CUDA integration design, and exploratory scaling evidence.
CUDA Barnes--Hut Stage 3B is source-reviewed but remains GPU-unverified and is
not enabled in either graphical path. The viewer and counterfactual engine use
only the CPU-reviewed Sequential/OpenMP Barnes--Hut backends; none of these
paths alter Stage 5.

Run one reproducible benchmark configuration:

```powershell
./build/openmp-release/nbody_benchmark.exe --backend openmp --threads 4 --bodies 512 --steps 50 --dt 10 --warmups 1 --repetitions 5 --seed 24301
```

The benchmark writes one CSV header and row to standard output. Ad hoc output is disposable; only controlled datasets with protocol and environment provenance belong under `data/benchmarks/`.

### Large-N Laboratory L1

Phase L1 is committed and independently source-reviewed for the bounded CPU
laboratory and retained exploratory pilot described below.

`nbody_large_n_lab` is a separate resource-bounded headless workflow for
uniform, clustered, and approximately planar synthetic workloads. It supports
Sequential/OpenMP as the backend dimension and Direct/Barnes--Hut as the solver
dimension, requires an explicit seed, applies phase-specific memory admission
and finite-result publication gates, records generation/initial-Direct-force/
Direct-trajectory/diagnostic/tree/traversal/complete-step timings separately,
and fails rather than falling back. Each completed evidence set is installed
under an immutable run ID; an interprocess-locked, atomically replaced root
pointer selects the authoritative run while retaining prior runs as history.
CUDA is metadata-only in L1 and is not reported as executed.

The initial Ryzen 7 5700U pilot stopped at 4096 bodies: 144 exploratory rows
cover `N=256,1024,4096`, three distributions, theta `0.25/0.5/0.8`, and two
retained repetitions. It is neither frozen Stage 5 evidence nor a claim about
50K/100K, graphical FPS, or RTX 4050 performance. See
[the Large-N Laboratory methodology](docs/large_n_laboratory.md) and the
[custodied publication pointer](data/benchmarks/large_n_l1_pilot/publication.json).

Run the controlled Stage 3 CPU matrix after configuring the OpenMP Release build:

```powershell
python -B scripts/run_cpu_benchmarks.py --physical-cores 8
```

The runner executes one untimed warm-up before each measured sample, counterbalances configuration order with a recorded deterministic seed, validates checksums and effective thread counts, and writes raw timings, summary statistics, environment metadata, and affinity-probe evidence under `data/benchmarks/stage3_cpu/`. See the benchmark protocol before interpreting results on another machine.

CUDA remains opt-in and does not affect either CPU build. On an NVIDIA-equipped host:

```powershell
.\scripts\test-local.ps1 -Configuration CUDA
.\scripts\run-visualizer.ps1 -Backend CUDA -ApplicationArguments '--device','0'
```

Use `docs/colab_cuda_workflow.md` for the controlled pilot and full benchmark. Do not compare Colab CUDA timings directly with the Stage 3 Windows CPU dataset.

### Optional MPI and large-N pilot

MPI remains outside the three ordinary local presets. Frozen MPI and large-N
evidence retains its recorded historical configuration. Any new approved MPI
diagnostic must use a specifically justified directory under
`build/scratch/`, never a new root-level `build-*` tree. Microsoft MPI and an
MPI-capable compiler wrapper are required on Windows; see
`docs/mpi_and_large_n.md` before constructing such a build.

`nbody_mpi_benchmark` reports the maximum rank wall time plus local compute and collective-communication components. The replicated-state design uses O(N) memory per rank, O(N^2/P) local force work, and two O(N) acceleration exchanges per Velocity Verlet step. See [MPI and large-N notes](docs/mpi_and_large_n.md) before interpreting timings or running the budgeted pilot orchestrator.

Verify the committed CUDA evidence independently:

```bash
python3 -B scripts/verify_cuda_evidence.py data/benchmarks/stage4_cuda/pilot
python3 -B scripts/verify_cuda_evidence.py data/benchmarks/stage4_cuda/run
```

## Project map

- `docs/research`: source-by-source review, novelty verdict, and research questions.
- `docs/architecture.md`: scientific and software design.
- `docs/validation_plan.md`: correctness criteria.
- `docs/benchmark_protocol.md`: same-machine experimental method.
- `docs/conservation_study.md`: independent timestep-duration conservation characterization.
- `docs/stage5_protocol.md`: versioned Stage 5 protocol index and retained v1.0 history.
- `docs/stage5_v2_protocol.md`: authoritative phase custody, policy, provenance, metrics, and gates.
- `docs/stage5_rtx4050_runbook.md`: Windows RTX 4050 pilot-gated execution procedure.
- `docs/visualization.md`: final viewer provenance, presentation policy, controls, and validation boundary.
- `docs/repository_layout.md`: canonical build policy, historical-build inventory, and safe cleanup manifest.
- `docs/mpi_and_large_n.md`: MPI/hybrid architecture, workload safety policy, and bounded pilot protocol.
- `docs/barnes_hut.md`: sequential tree/acceptance policy, numerical validation, resource bounds, and exploratory timings.
- `docs/MASTER_LEDGER.md`: status, decisions, evidence, limitations, and next actions.
- `scripts/run_cpu_benchmarks.py`: controlled CPU experiment and statistical-summary runner.
- `scripts/prepare_stage5_build.py`: fresh source-bound build, tool/topology capture, CTest, and manifest.
- `scripts/run_stage5.py`: Stage 5 v2 preflight, dry-run, pilot, calibration, and held-out dispatcher.
- `scripts/build_stage5_policy.py`: calibration-only deterministic policy and freeze builder.
- `scripts/evaluate_stage5_policy.py`: held-out evaluation without refitting.
- `scripts/verify_stage5_evidence.py`: independent Stage 5 evidence auditor.
- `backends`: implemented optional OpenMP, CUDA, and MPI backends.
- `visualization`: optional desktop application/controller, camera, and OpenGL renderer.

## Scientific boundary

The viewer's “Solar System” scenario is narrowly labeled as an approximate demonstration. It uses JPL mean orbital elements valid for 1800--2050 at epoch J2000.0, not observed Cartesian ephemerides; Earth uses the Earth--Moon barycenter orbit while the Moon is omitted. NASA factsheets provide body masses and mean radii. Ephemeris-grade claims remain reserved for a future committed Horizons extraction recording query, epoch, frame, center, time scale, units, raw response, and provenance.
