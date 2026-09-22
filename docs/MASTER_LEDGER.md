# Master ledger

Last updated: 2026-09-22

## Stage status

| Stage | Status | Evidence |
|---|---|---|
| Focused research and novelty assessment | Complete | Eight-source review and explicit comparative-study verdict |
| Scope and architecture | Complete | Scope, units, integrator, dependency and CUDA split decisions documented |
| Sequential reference | Complete | Direct GCC build, physics tests, and CSV smoke run passed |
| OpenMP backend | Not started | Phase 2 |
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

## Assumptions and unresolved questions

- Intended RTX 4050 exact model, driver, power limit, CPU, and memory configuration are unknown.
- Faculty acceptance criteria for “realistic” horizon and tolerated position error need confirmation before final study.
- Precision modes for CUDA and exact conservation thresholds will be preregistered after pilot validation, not chosen from favorable results.
- Whether OpenMP is available under the final MSVC toolchain remains to be tested.

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

## Known limitations

- No verified planetary initial-condition dataset yet.
- No close-encounter regularization, softening, relativistic effects, or adaptive timesteps.
- Optional backends and viewer are architectural placeholders only.
- Local CMake is unavailable at the time of foundation creation.

## Next actions

1. Install/use CMake on the Windows development machine and reproduce the default build.
2. Add OpenMP behind the shared force-backend contract with equivalence tests.
3. Create the Colab CUDA correctness workflow, then validate on the RTX 4050 before crossover claims.
4. Acquire and provenance-lock one JPL Horizons state before implementing the realistic scenario.

## Commit references

- `2d16155` — research, novelty assessment, scope, architecture, validation plan, and benchmark protocol.
- Sequential foundation commit: pending below in Git history at the time this ledger entry was written.
