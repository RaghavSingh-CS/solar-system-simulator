# Workload-Aware Parallel N-Body Simulation

A research-oriented C++17 project for comparing sequential, OpenMP, and CUDA direct N-body backends under explicit numerical-accuracy constraints. The current foundation contains only the deterministic sequential reference and an analytic two-body scenario.

The literature review found that CPU/GPU crossover and adaptive heterogeneous mapping are established topics. Accordingly, this project does **not** claim a new scheduling algorithm. Its proposed study asks whether a small, calibrated selector can find the fastest backend on one target machine while rejecting configurations that violate predeclared conservation bounds. See [novelty assessment](docs/research/novelty_assessment.md).

## Current capabilities

- Newtonian O(N^2) force evaluation in double precision.
- SI units and barycentric Cartesian coordinates.
- Fixed-step Velocity Verlet integration.
- Reproducible analytic circular two-body scenario (not Solar System ephemeris data).
- Energy, linear momentum, angular momentum, and center-of-mass diagnostics.
- Headless CSV output and dependency-free tests.

OpenMP, CUDA, MPI, realistic JPL Horizons initial conditions, and GLFW/OpenGL visualization are planned but are not implemented or validated yet.

## Build and test

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Run (Visual Studio multi-config path shown):

```powershell
./build/Release/nbody_headless.exe --steps 5000 --sample-every 50 --output orbit.csv
```

Use `--help` for options. CSV comment lines document scenario, units, integrator, gravitational constant, and time step; data columns are:

```text
time_s,body_index,name,mass_kg,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s
```

## Project map

- `docs/research`: source-by-source review, novelty verdict, and research questions.
- `docs/architecture.md`: scientific and software design.
- `docs/validation_plan.md`: correctness criteria.
- `docs/benchmark_protocol.md`: same-machine experimental method.
- `docs/MASTER_LEDGER.md`: status, decisions, evidence, limitations, and next actions.
- `backends`: future OpenMP/CUDA/MPI implementations.
- `visualization`: future independent desktop viewer.

## Scientific boundary

The bundled binary uses generic analytic parameters. A scenario will be called “Solar System” only after a JPL Horizons state vector is committed with its epoch, frame, center, time scale, units, raw query, and provenance. JPL’s [Horizons documentation](https://ssd.jpl.nasa.gov/horizons/manual.html) is the intended source.
