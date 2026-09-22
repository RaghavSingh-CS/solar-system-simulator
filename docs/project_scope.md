# Project scope

## Objective

Build a deterministic Newtonian N-body simulator with sequential, OpenMP, and CUDA backends, a native OpenGL viewer, and reproducible performance/accuracy experiments. The research framing is a comparative study of an accuracy-constrained backend selector on one target machine.

## Minimum viable project

- Shared C++17 body/state, SI units, barycentric Cartesian coordinates, and a fixed-step symplectic reference integrator.
- Deterministic O(N^2) sequential backend with analytic two-body validation.
- Functionally equivalent OpenMP and CUDA force backends.
- Headless runner producing documented CSV plus machine-readable run metadata.
- Synthetic Plummer-like or seeded random workloads for scaling, clearly separate from Solar System scenarios.
- One verified JPL Horizons Solar System snapshot with epoch, frame, center, time scale, units, query settings, and raw-source provenance committed as small text data.
- Native GLFW/OpenGL visualization that consumes simulation snapshots but is excluded from benchmark timings.
- Same-machine benchmark matrix and conservation/error checks.
- A static calibrated selector evaluated against fixed baselines and an offline oracle.

## Stretch goals

- Optional MPI j-parallel force reduction, never required for default configure/build/test.
- Online recalibration, uncertainty-aware decisions, mixed precision, and energy measurement.
- A higher-fidelity Wisdom–Holman integrator for long-horizon planetary studies.
- Portable visualization packages and richer camera/orbit controls.

## Out of scope for the month

- General-purpose heterogeneous scheduling runtime.
- Production ephemeris or relativistic model.
- Collision handling, close-encounter regularization, adaptive individual timesteps, Barnes–Hut/FMM, or millions of bodies.
- Claims based on comparing Colab GPU times with local CPU times.
- Publication guarantees.

## Staged delivery

1. **Foundation (this stage):** literature, scope, architecture, deterministic sequential reference, analytic two-body tests, CSV runner.
2. **CPU parallelism:** OpenMP implementation, equivalence tests, benchmark harness.
3. **GPU:** CUDA implementation and Colab compile/correctness workflow; later repeat on RTX 4050.
4. **Visualization and verified Solar System scenario:** OpenGL viewer plus a versioned JPL Horizons snapshot.
5. **Study:** preregister thresholds, collect repeated same-machine data, fit/freeze selector, evaluate held-out workloads, write report.

## Acceptance boundaries

The current implementation is not called a “realistic Solar System simulation” until a verified ephemeris dataset is added and the intended time horizon/model limitations are documented. The analytic binary is a test fixture, not ephemeris data.
