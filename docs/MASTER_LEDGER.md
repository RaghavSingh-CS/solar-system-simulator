# Master ledger

Last updated: 2026-09-22

## Stage status

| Stage | Status | Evidence |
|---|---|---|
| Focused research and novelty assessment | Complete | Eight-source review and explicit comparative-study verdict |
| Scope and architecture | Complete | Scope, units, integrator, dependency and CUDA split decisions documented |
| Sequential reference | Complete | Direct GCC build, physics tests, and CSV smoke run passed |
| OpenMP backend | Complete | GCC/libgomp Release build; 1/2/4-thread validation passed |
| CUDA backend | Not started | Phase 3; requires Colab/RTX access for validation |
| MPI backend | Optional/not started | Stretch goal only |
| OpenGL visualization | Not started | Optional dependency plan only |
| Final experiment | Not started | Requires all candidate backends on same RTX 4050 machine |

## Decisions and rationale

- Research framing is comparative, not algorithmically novel; prior work already establishes crossover and adaptive scheduling.
- Primary RQ uses an accuracy gate plus same-machine calibrated selection.
- SI units and barycentric inertial Cartesian coordinates prevent hidden conversion conventions.
- Velocity Verlet is the minimal symplectic reference; long-horizon planetary claims require later integrator evaluation.
- CPU-only CMake is the default; CUDA, graphics, and MPI are opt-in.
- GLFW/core OpenGL is the intended viewer stack, isolated from simulation and benchmarks.
- Real Solar System state is pending a fully recorded JPL Horizons extraction; no ephemeris values are invented.
- `ForceBackend` is the shared acceleration contract; the sequential overload remains the reference/default.
- OpenMP parallelizes destination bodies with static scheduling and thread-private accumulation.
- OpenMP is opt-in so builds without an OpenMP runtime remain functional.

## Assumptions and unresolved questions

- Intended RTX 4050 exact model, driver, power limit, CPU, and memory configuration are unknown.
- Faculty acceptance criteria for “realistic” horizon and tolerated position error need confirmation before final study.
- Precision modes for CUDA and exact conservation thresholds will be preregistered after pilot validation, not chosen from favorable results.
- OpenMP is validated with GCC 15.1.0/libgomp; availability and numerical behavior under the final MSVC toolchain remain untested.

## Research evidence

See `docs/research/literature_review.md`. Key constraint: Sapporo2 already measured N-body CPU/GPU crossover and precision effects; Qilin and StarPU already perform learned heterogeneous mapping.

## Validation results

Environment: MSYS2 `c++ 15.1.0`; CMake was not on `PATH`.

Direct test build (completed with no warnings):

```powershell
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I C:\Projects\PCAP\solar-system-simulator\include C:\Projects\PCAP\solar-system-simulator\src\simulation.cpp C:\Projects\PCAP\solar-system-simulator\src\scenarios.cpp C:\Projects\PCAP\solar-system-simulator\src\diagnostics.cpp C:\Projects\PCAP\solar-system-simulator\tests\test_two_body.cpp -o C:\Projects\PCAP\solar-system-simulator\nbody_tests.exe
```

Test execution result: `all two-body tests passed`. The suite covers pair-force balance, five circular-orbit periods (20,000 steps), conservation/separation guards, and invalid timestep/coincident-position rejection.

Headless build used the same flags and core sources with `src/main.cpp`. Smoke command:

```powershell
.\nbody_headless.exe --steps 5000 --sample-every 50 --output orbit.csv
```

Result: exit 0, 210 CSV lines, simulated time `24320740.25 s`, reported final relative energy drift `6.045915326e-15`.

CMake, OpenMP, CUDA, MPI, and visualization were **not** built or tested in this environment.

The preceding paragraph is the Stage 1 record. Stage 2 established the following:

- Official portable CMake 4.4.3; Ninja 1.13.2 generator; GCC 15.1.0 with `_OPENMP=201511`/libgomp; Release configuration.
- CPU-only configure/build succeeded; original CTest `1/1` passed; requesting OpenMP produced the expected unavailable-backend diagnostic.
- OpenMP configure found `OpenMP::OpenMP_CXX`; the build completed without reported compiler warnings; CTest `2/2` passed.
- Analytic two-body and seeded 32-body validation passed with 1, 2, and 4 threads. All cross-backend and repeat-run position/velocity differences were `0` in this environment.
- Analytic energy/angular drift: `1.5114788315343793e-15` / `3.5096052562406509e-16`.
- Seeded energy/angular drift: `1.2963732290918661e-15` / `7.6765086087405374e-16`.
- Preliminary benchmark medians show OpenMP-4 slower at 64 bodies (`0.47x` sequential/OMP-4 ratio), then faster at 256 (`2.43x`) and 512 bodies (`2.92x`). These measurements are exploratory, not publication-quality.

CUDA, MPI, and visualization remain unbuilt and untested.

Exact Stage 2 configure/build/test commands (PowerShell, executed from the repository root):

```powershell
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' -S . -B build-sequential -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/c++.exe -DCMAKE_MAKE_PROGRAM=C:/msys64/ucrt64/bin/ninja.exe -DNBODY_ENABLE_OPENMP=OFF
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' --build build-sequential --config Release
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\ctest.exe' --test-dir build-sequential -C Release --output-on-failure
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' -S . -B build-openmp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/c++.exe -DCMAKE_MAKE_PROGRAM=C:/msys64/ucrt64/bin/ninja.exe -DNBODY_ENABLE_OPENMP=ON
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' --build build-openmp --config Release
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\ctest.exe' --test-dir build-openmp -C Release --output-on-failure
```

## Known limitations

- No verified planetary initial-condition dataset yet.
- No close-encounter regularization, softening, relativistic effects, or adaptive timesteps.
- CUDA/MPI backends and the viewer are architectural placeholders only.
- The pacman CMake executable is unusable in the partially updated MSYS2 environment; the verified portable CMake is project-local and ignored.
- Preliminary benchmark data lack randomized run order, affinity control, thermal monitoring, and sufficient repetitions.

## Next actions

1. Re-run the OpenMP matrix on the intended demonstration CPU with randomized order, affinity control, and more repetitions.
2. Acquire and provenance-lock one JPL Horizons state before implementing the realistic scenario.
3. Create the Colab CUDA correctness workflow, then validate on the RTX 4050 before crossover claims.
4. Confirm OpenMP behavior under the final Windows compiler/toolchain if it differs from GCC/libgomp.

## Commit references

- `2d16155` — research, novelty assessment, scope, architecture, validation plan, and benchmark protocol.
- `c241ca7` — deterministic sequential implementation, tests, headless runner, and project scaffolding.
