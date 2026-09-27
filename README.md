# Workload-Aware Parallel N-Body Simulation

A research-oriented C++17 project for comparing sequential, OpenMP, CUDA, and MPI direct N-body execution under explicit numerical-accuracy constraints. The repository contains deterministic sequential and OpenMP backends, an optional CUDA backend validated on an NVIDIA Tesla T4, and an optional replicated-state MPI backend with a clean MPI+OpenMP hybrid mode.

The literature review found that CPU/GPU crossover and adaptive heterogeneous mapping are established topics. Accordingly, this project does **not** claim a new scheduling algorithm. Its proposed study asks whether a small, calibrated selector can find the fastest backend on one target machine while rejecting configurations that violate predeclared conservation bounds. See [novelty assessment](docs/research/novelty_assessment.md).

## Current capabilities

- Newtonian O(N^2) force evaluation in double precision.
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
- Prepared Stage 5 same-machine runner and independent evidence verifier; no Stage 5 timings collected yet.
- Optional replicated-state MPI backend with deterministic target partitioning, global acceleration exchange, and MPI+OpenMP local force computation.
- Optional presentation-ready GLFW/OpenGL 3.3 + ImGui viewer with Solar/Compute Experiment modes, deterministic camera presets, stable inner-system callouts, camera-aware body sizing, reference/history/Sun-motion trails, barycenter marker, procedural planet identity, batched large-N points through 100,000 bodies, live numerical-health bounds, and an isolated CPU-backend comparison.

Stage 4 real-GPU validation and the controlled 154-sample CUDA matrix are complete; evidence is under `data/benchmarks/stage4_cuda/`. No same-machine CPU/OpenMP/CUDA crossover has been measured, so no automatic selector is implemented. Stage 6D completes the MPI, large-N infrastructure, and final presentation feature scope; native-window appearance and interaction still require the final user visual pass. Ephemeris-grade JPL Horizons initial conditions remain outside this stage.

An independent 30-configuration sequential conservation characterization is
documented in `docs/conservation_study.md`, with evidence under
`data/experiments/conservation/`. It is not a backend benchmark or Stage 5.

Stage 5A preparation is complete at the candidate frozen revision recorded in
`docs/MASTER_LEDGER.md`. The RTX 4050 pilot and full same-machine experiment
remain unexecuted; follow `docs/stage5_rtx4050_runbook.md` and stop for review
after the pilot.

## Build and test

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The default build remains CPU-only and does not require OpenMP. Enable the validated GCC/libgomp backend explicitly:

```powershell
cmake -S . -B build-openmp -G Ninja -DCMAKE_BUILD_TYPE=Release -DNBODY_ENABLE_OPENMP=ON
cmake --build build-openmp
ctest --test-dir build-openmp --output-on-failure
```

### Optional interactive viewer

The viewer is off by default, so headless builds do not fetch or require graphics dependencies. Configure a Release viewer with OpenMP support:

```powershell
cmake -S . -B build-visual -G Ninja -DCMAKE_BUILD_TYPE=Release -DNBODY_BUILD_VISUALIZATION=ON -DNBODY_ENABLE_OPENMP=ON -DNBODY_ENABLE_CUDA=OFF
cmake --build build-visual
ctest --test-dir build-visual --output-on-failure
./build-visual/nbody_visualizer.exe --backend sequential
```

The first visualization configure requires network access to fetch exact pinned revisions of GLFW 3.4, GLM 1.0.1, and Dear ImGui 1.92.9b. The generated GLAD 2.0.6 OpenGL 3.3 loader is committed under `third_party/glad`, so Python, Jinja2, copied DLLs, and runtime asset discovery are not required. Shaders are embedded in the executable.

The default is a nine-body Solar System demonstration at J2000.0. Planet positions come from JPL's published 1800--2050 approximate mean-element model; centered finite differences of that same model provide initial velocities, then the state is shifted to its center of mass. This is a deterministic educational approximation, not a Horizons ephemeris. The physical timestep remains fixed at six hours and named UI speed presets change only the number of physics steps per rendered frame. A deterministic seeded `Compute Experiment` remains available with `--scenario synthetic`.

Controls:

- The themed ImGui panel separates primary simulation/compute/health controls from collapsible technical details. Numerical health compares live drift against the unchanged `5e-6` energy and `5e-10` angular-momentum bounds.
- `Overview`, `Inner System`, `Earth`, `Jupiter`, and `Saturn` provide deterministic demo framing without distorting physical positions.
- Approximate reference orbit guides provide immediate context; fading history trails remain simulation-derived and planet positions always come from N-body integration.
- `Barycenter & real Sun motion` marks the barycentric origin and traces integrated Sun positions; an explicit presentation-only amplification control defaults to `10x`.
- Compute Experiment offers deterministic `N=32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,100000`; workloads above 64 use one batched point-sprite draw. Continuous simulation and live O(N^2) diagnostics stop above 1024, bounded manual stepping is available only through 8192, and larger workloads are explicitly display/inspection-only in the viewer.
- `Compare CPU Backends` times isolated identical Sequential/OpenMP state copies, checks established mixed correctness tolerances, and is explicitly a live diagnostic—not benchmark evidence.
- `Space`: pause/resume; `R`: exact scenario reset; `Esc`: exit.
- `1`: Sequential; `2`: OpenMP using `--threads N` when OpenMP was built.
- Click a body to select it; focus selected, focus Sun/body 0, or return to overview from the panel.
- Left-drag, `A`/`D`, or arrow keys: orbit; mouse wheel or `W`/`S`: dolly; `C`: overview camera.

The panel is resizable from 380 px to the smaller of 680 px or 55% of the window, and the 3D scene viewport always occupies the remaining width. Hover help documents every primary control. CUDA and MPI are intentionally not exposed in the viewer; CUDA requests are rejected with an actionable error, while MPI is exercised by its separate benchmark executable. Auto remains absent until Stage 5 produces a valid calibration. Viewer FPS/frame time is display-only and is not controlled benchmark evidence. See [the visualization notes](docs/visualization.md) for provenance, architecture, options, and limitations.

Run (Visual Studio multi-config path shown):

```powershell
./build/Release/nbody_headless.exe --steps 5000 --sample-every 50 --output orbit.csv
```

Use `--help` for options. CSV comment lines document scenario, units, integrator, gravitational constant, and time step; data columns are:

```text
time_s,body_index,name,mass_kg,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s
```

Run one reproducible benchmark configuration:

```powershell
./build-openmp/nbody_benchmark.exe --backend openmp --threads 4 --bodies 512 --steps 50 --dt 10 --warmups 1 --repetitions 5 --seed 24301
```

The benchmark writes one CSV header and row to standard output. Ad hoc output is disposable; only controlled datasets with protocol and environment provenance belong under `data/benchmarks/`.

Run the controlled Stage 3 CPU matrix after configuring the OpenMP Release build:

```powershell
python -B scripts/run_cpu_benchmarks.py --physical-cores 8
```

The runner executes one untimed warm-up before each measured sample, counterbalances configuration order with a recorded deterministic seed, validates checksums and effective thread counts, and writes raw timings, summary statistics, environment metadata, and affinity-probe evidence under `data/benchmarks/stage3_cpu/`. See the benchmark protocol before interpreting results on another machine.

CUDA remains opt-in and does not affect either CPU build. On an NVIDIA-equipped host:

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DNBODY_ENABLE_CUDA=ON
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

Use `docs/colab_cuda_workflow.md` for the controlled pilot and full benchmark. Do not compare Colab CUDA timings directly with the Stage 3 Windows CPU dataset.

### Optional MPI and large-N pilot

Build MPI separately; Microsoft MPI and an MPI-capable compiler wrapper are required on Windows:

```powershell
cmake -S . -B build-mpi -G Ninja -DCMAKE_BUILD_TYPE=Release -DNBODY_ENABLE_MPI=ON -DNBODY_ENABLE_OPENMP=ON
cmake --build build-mpi
ctest --test-dir build-mpi --output-on-failure
mpiexec -n 4 ./build-mpi/nbody_mpi_benchmark.exe --bodies 512 --steps 50 --dt 10 --threads 1 --warmups 1 --repetitions 5 --seed 24301
```

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
- `docs/stage5_protocol.md`: frozen same-machine matrix, split, metrics, and gates.
- `docs/stage5_rtx4050_runbook.md`: Windows RTX 4050 pilot-gated execution procedure.
- `docs/visualization.md`: final viewer provenance, presentation policy, controls, and validation boundary.
- `docs/mpi_and_large_n.md`: MPI/hybrid architecture, workload safety policy, and bounded pilot protocol.
- `docs/MASTER_LEDGER.md`: status, decisions, evidence, limitations, and next actions.
- `scripts/run_cpu_benchmarks.py`: controlled CPU experiment and statistical-summary runner.
- `scripts/run_stage5.py`: Stage 5 preflight, dry-run, pilot, and full orchestrator.
- `scripts/verify_stage5_evidence.py`: independent Stage 5 evidence auditor.
- `backends`: implemented optional OpenMP, CUDA, and MPI backends.
- `visualization`: optional desktop application/controller, camera, and OpenGL renderer.

## Scientific boundary

The viewer's “Solar System” scenario is narrowly labeled as an approximate demonstration. It uses JPL mean orbital elements valid for 1800--2050 at epoch J2000.0, not observed Cartesian ephemerides; Earth uses the Earth--Moon barycenter orbit while the Moon is omitted. NASA factsheets provide body masses and mean radii. Ephemeris-grade claims remain reserved for a future committed Horizons extraction recording query, epoch, frame, center, time scale, units, raw response, and provenance.
