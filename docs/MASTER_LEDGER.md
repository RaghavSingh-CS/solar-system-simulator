# Master ledger

Last updated: 2026-09-23

## Stage status

| Stage | Status | Evidence |
|---|---|---|
| Focused research and novelty assessment | Complete | Eight-source review and explicit comparative-study verdict |
| Scope and architecture | Complete | Scope, units, integrator, dependency and CUDA split decisions documented |
| Sequential reference | Complete | Direct GCC build, physics tests, and CSV smoke run passed |
| OpenMP backend | Complete | GCC/libgomp Release build; 1/2/4-thread validation passed |
| Controlled CPU benchmarking | Complete | 660 raw samples; randomized 64–1024-body sequential/OpenMP matrix |
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
- Stage 3 records crossover evidence only; it does not encode a backend-selection threshold.
- The controlled CPU matrix uses block-randomized order, one same-process warm-up per sample, and raw repetition retention.
- Windows/libgomp affinity is explicitly classified as uncontrolled after the runtime rejected the requested placement policy.

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

## Stage 3: controlled CPU benchmarking

### Objective and protocol

Stage 3 establishes measurement evidence for later sequential/OpenMP calibration without implementing a selector. The authoritative dataset was generated from clean harness commit `a4c9eb1` and is stored in `data/benchmarks/stage3_cpu/`:

- `raw_timings.csv`: 660 measured repetition rows;
- `summary.csv`: 60 configuration summaries;
- `environment.json`: machine, Git, matrix, timer, and telemetry metadata;
- `affinity_probe.txt`: unedited libgomp placement evidence.

The matrix contains body counts 64, 96, 128, 160, 192, 256, 384, 512, 768, and 1024. Each count uses sequential plus OpenMP 1, 2, 4, 8, and 16 threads. Every sample integrates 100 Velocity Verlet steps at `dt=10 s` from synthetic seed `24301`. One untimed warm-up runs immediately before each measured sample in the same process. Eleven complete repetition blocks are independently shuffled with the recorded xorshift64*/Fisher-Yates order seed `20260923`, so each block contains every configuration exactly once.

The timer covers integration only, including both force evaluations per step. It excludes process startup, scenario generation, initial-state copying, warm-up integration, diagnostics, checksum computation, CSV formatting, and runner I/O. The final state is consumed by diagnostics and a checksum. Summary statistics are median (primary), mean, sample standard deviation, min/max, IQR, coefficient of variation, deterministic 10,000-resample bootstrap 95% interval for the median, speedup, and speedup/thread efficiency.

The size grid retains all Stage 2 sizes, samples 96–256 densely around the preliminary crossover, and extends through 1024 to expose scaling saturation. One hundred fixed steps keep the work definition identical across body counts; eleven repetitions provide an observed median while remaining short enough for a controlled laptop run.

### Machine and execution controls

- CPU: AMD Ryzen 7 5700U with Radeon Graphics; 8 physical cores, 16 logical processors.
- OS: Windows 11 Home Single Language, version `10.0.26200`, build 26200, 64-bit.
- Build: Release; GCC 15.1.0; Ninja 1.13.2; portable CMake 4.4.3; libgomp with `_OPENMP=201511` (OpenMP 4.5).
- All 660 rows report the requested effective thread count, `OMP_DYNAMIC=FALSE`, and the same compiler/build metadata.
- The code uses `schedule(static)`. A separate probe requested `OMP_PROC_BIND=CLOSE` and `OMP_PLACES=CORES`; libgomp reported `Affinity not supported on this configuration`, cleared the places list, and displayed all four probe workers eligible on CPUs 0–15. Measurement therefore used `OMP_PROC_BIND=FALSE` with `OMP_PLACES` and `GOMP_CPU_AFFINITY` unset. Affinity is **not** claimed as controlled.
- WMI recorded a one-time `CurrentClockSpeed`/`MaxClockSpeed` snapshot of 1801 MHz. This is not continuous boost-clock telemetry and is not used as a control. The ACPI thermal-zone query returned `Not supported`; CPU temperature telemetry is unavailable. No monitoring software was installed.

### Summary results

The following table selects the lowest median at each body count. Times are milliseconds; efficiency is speedup divided by requested OpenMP threads.

| Bodies | Best backend | Threads | Median ms | Bootstrap median 95% interval ms | Speedup | Efficiency | CV |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | Sequential | 1 | 4.989 | 4.624–5.104 | 1.00x | — | 9.8% |
| 96 | Sequential | 1 | 10.699 | 9.428–11.478 | 1.00x | — | 12.0% |
| 128 | OpenMP | 4 | 16.346 | 15.281–18.456 | 1.13x | 28.2% | 10.2% |
| 160 | OpenMP | 4 | 19.054 | 17.932–20.974 | 1.65x | 41.3% | 10.8% |
| 192 | OpenMP | 4 | 24.615 | 23.460–26.406 | 1.55x | 38.7% | 12.7% |
| 256 | OpenMP | 4 | 33.460 | 31.642–35.634 | 2.09x | 52.3% | 8.4% |
| 384 | OpenMP | 16 | 56.418 | 53.555–60.297 | 3.02x | 18.9% | 7.8% |
| 512 | OpenMP | 8 | 88.423 | 74.098–93.221 | 3.14x | 39.2% | 12.3% |
| 768 | OpenMP | 16 | 178.359 | 164.117–181.267 | 3.64x | 22.8% | 7.9% |
| 1024 | OpenMP | 16 | 296.887 | 271.483–317.064 | 3.84x | 24.0% | 8.6% |

Observed median crossover by requested OpenMP thread count:

| Threads | First median win | Conservative interpretation |
|---:|---:|---|
| 1 | 128 | Irregular at 128–192; median advantage is continuous only from 256 onward. |
| 2 | 128 | Median wins from 128 onward; 128 intervals overlap sequential. |
| 4 | 128 | Best median at 128; separation is clear by 160. |
| 8 | 160 | Median wins from 160 onward. |
| 16 | 192 | Small-N team overhead delays crossover. |

The best observed backend changes from sequential through 96 bodies to OpenMP-4 at 128. However, the 128-body OpenMP-4 interval (15.281–18.456 ms) overlaps the sequential interval (16.687–19.605 ms); the supported crossover region is therefore approximately 128–160 bodies, not a single calibrated threshold. At 160 bodies the OpenMP-4 interval (17.932–20.974 ms) is separated from sequential (26.174–35.110 ms).

Small workloads show overhead domination: at 64 bodies OpenMP speedup falls from `0.89x` with one thread to `0.19x` with 16. Scaling improves with N but saturates: at 1024 bodies speedup/efficiency are `1.79x/89.6%` (2 threads), `3.01x/75.2%` (4), `3.68x/46.0%` (8), and `3.84x/24.0%` (16). Sixteen threads are slower than eight at 512 bodies and improve over eight by only about 2.6% at 768 and 4.4% at 1024. These are measured facts; affinity, scheduler behavior, memory effects, boost behavior, and thermals are possible influences but were not measured well enough to assign causation.

Across all 60 configurations, coefficient of variation ranges from 4.36% to 24.73%; the highest is OpenMP-4 at 512 bodies. Best-per-size configurations range from 7.8% to 12.7% CV. The 11 samples and bootstrap intervals quantify observed variation but do not justify broad population-level confidence claims.

### Correctness and integrity

- CPU-only CTest: `1/1` passed.
- OpenMP CTest: `2/2` passed, including analytic/seeded cross-backend validation and repeatability.
- Raw dataset: 60 configurations × 11 samples; no missing/duplicate configuration counts.
- Effective thread count matched requested count in all 660 rows.
- Every backend/thread result at a given body count produced the same checksum; no numerical discrepancy was averaged away.
- Release and GCC 15.1.0 metadata are consistent across all rows.
- SHA-256: raw `9884A2CA13D0364B400D22E3B5EC0ED7F4F4F1F785C73E94B8565E823565B286`; summary `FE36B9531A1FED99E4E4790F10BDF8730DA4D4428FD9AAC5FF465A6A677A9A40`.

### Exact reproduction commands

From the repository root, after restoring the ignored portable CMake path documented above:

```powershell
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' -S . -B build-openmp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/c++.exe -DCMAKE_MAKE_PROGRAM=C:/msys64/ucrt64/bin/ninja.exe -DNBODY_ENABLE_OPENMP=ON
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' --build build-openmp --config Release
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\ctest.exe' --test-dir build-openmp -C Release --output-on-failure
python -B scripts\run_cpu_benchmarks.py --executable build-openmp\nbody_benchmark.exe --output-dir data\benchmarks\stage3_cpu --bodies 64,96,128,160,192,256,384,512,768,1024 --threads 1,2,4,8,16 --steps 100 --dt 10 --warmups 1 --repetitions 11 --benchmark-seed 24301 --order-seed 20260923 --bootstrap-resamples 10000 --physical-cores 8 --experiment-id stage3-cpu-2026-09-23
```

### Decisions, unresolved questions, and Stage 4 implication

- No automatic selector or threshold is implemented. The 128–160 observation is evidence for later calibration, not policy.
- The result is specific to this code, Release build, Windows scheduler, GCC/libgomp runtime, CPU, and workload definition.
- The lack of verified affinity and continuous clock/temperature data remains a material limitation.
- The 16-thread SMT results show limited marginal benefit beyond eight threads at large N, but causation is unresolved.
- Stage 4 must rerun or preserve an equivalent controlled CPU matrix on the **same machine and session class** used for CUDA measurements. Cross-machine CPU/GPU ratios are invalid.
- Recommended Stage 4 scope: add an opt-in CUDA force backend and deterministic sequential/OpenMP/CUDA correctness tests; measure CUDA kernel-only and end-to-end transfer-inclusive timings over the same seeded size grid with explicit device synchronization and full GPU metadata. Do not add OpenGL, heterogeneous scheduling, or the final backend-selection policy in Stage 4.

## Known limitations

- No verified planetary initial-condition dataset yet.
- No close-encounter regularization, softening, relativistic effects, or adaptive timesteps.
- CUDA/MPI backends and the viewer are architectural placeholders only.
- The pacman CMake executable is unusable in the partially updated MSYS2 environment; the verified portable CMake is project-local and ignored.
- Stage 3 CPU affinity could not be enforced by this Windows/libgomp configuration; thermal and continuous clock telemetry were unavailable.
- Eleven samples per configuration characterize this run but remain too few for strong distributional or cross-machine claims.

## Next actions

1. Begin only the Stage 4 CUDA foundation/correctness and matched benchmark work described above.
2. Re-run this CPU protocol on the eventual RTX 4050 host if it is not this exact machine.
3. Acquire and provenance-lock one JPL Horizons state before implementing the realistic scenario.
4. Confirm OpenMP behavior under the final Windows compiler/toolchain if it differs from GCC/libgomp.

## Commit references

- `2d16155` — research, novelty assessment, scope, architecture, validation plan, and benchmark protocol.
- `c241ca7` — deterministic sequential implementation, tests, headless runner, and project scaffolding.
- `7f6e2bc` — Stage 1 validation evidence and exact foundation commit references.
- `1485d54` — Stage 2 OpenMP backend, validation suite, benchmark harness, and documentation.
- `a4c9eb1` — frozen Stage 3 controlled benchmark harness and protocol used for measurement.
- `94dbf19` — Stage 3 raw measurements, statistical summaries, environment evidence, and analysis.
