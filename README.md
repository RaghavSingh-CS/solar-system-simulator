# Workload-Aware Parallel N-Body Simulation

A research-oriented C++17 project for comparing sequential, OpenMP, and CUDA direct N-body backends under explicit numerical-accuracy constraints. The repository contains deterministic sequential and OpenMP backends plus an optional CUDA backend awaiting real-GPU validation.

The literature review found that CPU/GPU crossover and adaptive heterogeneous mapping are established topics. Accordingly, this project does **not** claim a new scheduling algorithm. Its proposed study asks whether a small, calibrated selector can find the fastest backend on one target machine while rejecting configurations that violate predeclared conservation bounds. See [novelty assessment](docs/research/novelty_assessment.md).

## Current capabilities

- Newtonian O(N^2) force evaluation in double precision.
- Shared force-backend interface with sequential and OpenMP implementations.
- SI units and barycentric Cartesian coordinates.
- Fixed-step Velocity Verlet integration.
- Reproducible analytic circular two-body scenario (not Solar System ephemeris data).
- Deterministic seeded multi-body scenarios for validation and CPU scaling tests.
- Energy, linear momentum, angular momentum, and center-of-mass diagnostics.
- Headless CSV output and dependency-free tests.
- Computation-only benchmark plus a controlled, block-randomized CPU experiment runner with raw and summary CSV output.
- Optional double-precision direct CUDA backend, GPU-resident Velocity Verlet executor, and dual-regime CUDA benchmark preparation.

The CUDA code and Colab workflow are prepared but have not been compiled or executed on NVIDIA hardware. MPI, realistic JPL Horizons initial conditions, and GLFW/OpenGL visualization are not implemented or validated.

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

## Project map

- `docs/research`: source-by-source review, novelty verdict, and research questions.
- `docs/architecture.md`: scientific and software design.
- `docs/validation_plan.md`: correctness criteria.
- `docs/benchmark_protocol.md`: same-machine experimental method.
- `docs/MASTER_LEDGER.md`: status, decisions, evidence, limitations, and next actions.
- `scripts/run_cpu_benchmarks.py`: controlled CPU experiment and statistical-summary runner.
- `backends`: implemented optional OpenMP/CUDA backends and the future MPI placeholder.
- `visualization`: future independent desktop viewer.

## Scientific boundary

The bundled binary uses generic analytic parameters. A scenario will be called “Solar System” only after a JPL Horizons state vector is committed with its epoch, frame, center, time scale, units, raw query, and provenance. JPL’s [Horizons documentation](https://ssd.jpl.nasa.gov/horizons/manual.html) is the intended source.
