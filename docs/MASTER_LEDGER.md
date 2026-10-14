# Master ledger

Last updated: 2026-10-13

## Stage status

| Stage | Status | Evidence |
|---|---|---|
| Focused research and novelty assessment | Complete | Eight-source review and explicit comparative-study verdict |
| Scope and architecture | Complete | Scope, units, integrator, dependency and CUDA split decisions documented |
| Sequential reference | Complete | Direct GCC build, physics tests, and CSV smoke run passed |
| OpenMP backend | Complete | GCC/libgomp Release build; 1/2/4-thread validation passed |
| Controlled CPU benchmarking | Complete | 660 raw samples; randomized 64–1024-body sequential/OpenMP matrix |
| CUDA backend | Complete | Tesla T4 build/tests plus verified 15-row pilot and 154-row controlled matrix |
| Independent conservation characterization | Complete | 30 sequential timestep-duration configurations; all passed frozen bounds |
| Stage 5 v1.0 preparation | Historical / superseded | Unexecuted revision `4916b3c`; retained without reinterpretation |
| Stage 5 v2.0 pre-RTX remediation | **LOCALLY VALIDATED / NVIDIA REHEARSAL PENDING** | Phase custody, deterministic policy freeze, build provenance, topology derivation, adversarial tests, and external packet generator |
| Stage 5B RTX 4050 execution | Not started | First rehearsal stops after pilot approval; final collection requires separate authorization |
| MPI and hybrid backend | Complete | Replicated-state 1/2/4-rank correctness plus 2-rank x 2-thread hybrid validation |
| Stage 6A OpenGL application foundation | **IMPLEMENTED / USER VISUAL VALIDATION PENDING** | Optional viewer builds/tests; Sequential and OpenMP real-context smoke launches passed |
| Stage 6B Solar System application | Complete | Approximate J2000 scenario and scientific UI passed build/runtime validation and user visual inspection |
| Stage 6C final presentation pass | **IMPLEMENTED / USER VISUAL VALIDATION PENDING** | Themed UI, camera presets, orbit context, decluttered labels, planet identity/rings/starfield, and headless presentation tests |
| Stage 6C.1 corrective pass | **IMPLEMENTED / USER VISUAL VALIDATION PENDING** | Mercury render clearance, stable labels, barycentric Sun motion, scalable variable-N compute mode, and isolated live CPU comparison |
| Stage 6D final feature stage | **IMPLEMENTED / USER VISUAL VALIDATION PENDING** | Camera-aware Solar sizing, fixed inner callouts, polished/resizable/help-complete UI, MPI/hybrid, safe 100,000-body display, and bounded pilot infrastructure |
| Stage 6D.1 Sun readability correction | Superseded | Mercury-dependent clearance override caused visible Sun pulsation |
| Stage 6D.2 stable Sun sizing | **IMPLEMENTED / USER VISUAL VALIDATION PENDING** | Camera-dependent 20--36 px Sun radius invariant to orbital phase |
| Stage 6D.3 inner-orbit shape verification | Complete | Elements, 700-day integrated extrema, reference guides, trails, projection, and Sun invariant verified; no defect or visual change |
| CUDA visualizer integration | **IMPLEMENTED LOCALLY / NVIDIA VALIDATION PENDING** | Host-authoritative batched CUDA controller path; CPU-only 5/5 and OpenMP 6/6 CTests passed; CUDA compilation/runtime unverified on this AMD-only machine |
| Astronomical catalog and 60-body dataset | **IMPLEMENTED LOCALLY / REVIEW PENDING** | Immutable catalog foundation plus a reproducible JPL-sourced 60-body J2000 snapshot, offline deterministic generation, and CUDA-disabled CPU/OpenMP regression validation |
| 60-body scenario composition and visualization | **IMPLEMENTED LOCALLY / REVIEW PENDING** | Full/subset catalog solver composition, independent visibility, stable-ID subsystem views, local-origin rendering, parent-relative trails, and fresh CPU/OpenMP Release validation |
| CUDA paired counterfactual execution | **IMPLEMENTED LOCALLY / NVIDIA VALIDATION PENDING** | Separate persistent full/compacted CUDA executors, transactional paired host publication, explicit device gating, CPU-only rejection, and CUDA-dependent skip-registered tests |
| Barnes--Hut large-N solver roadmap | **STAGE 4A VIEWER INTEGRATION COMMITTED / SOURCE-REVIEWED** | Sequential/OpenMP viewer selection, transactional CPU frame batches, theta/resource diagnostics, and explicit CUDA/Barnes--Hut unavailability; CUDA compilation/runtime and controlled large-N evaluation remain mandatory |

## Decisions and rationale

- Research framing is comparative, not algorithmically novel; prior work already establishes crossover and adaptive scheduling.
- Primary RQ uses an accuracy gate plus same-machine calibrated selection.
- SI units and barycentric inertial Cartesian coordinates prevent hidden conversion conventions.
- Velocity Verlet is the minimal symplectic reference; long-horizon planetary claims require later integrator evaluation.
- CPU-only CMake is the default; CUDA, graphics, and MPI are implemented but opt-in.
- GLFW/core OpenGL is the intended viewer stack, isolated from simulation and benchmarks.
- Stage 6B's Solar scenario is an explicitly approximate JPL mean-element demonstration; ephemeris-grade state remains pending a fully recorded Horizons extraction.
- `ForceBackend` is the shared acceleration contract; the sequential overload remains the reference/default.
- OpenMP parallelizes destination bodies with static scheduling and thread-private accumulation.
- OpenMP is opt-in so builds without an OpenMP runtime remain functional.
- Stage 3 records crossover evidence only; it does not encode a backend-selection threshold.
- The controlled CPU matrix uses block-randomized order, one same-process warm-up per sample, and raw repetition retention.
- Windows/libgomp affinity is explicitly classified as uncontrolled after the runtime rejected the requested placement policy.
- Stage 4 real-GPU evidence is environment-scoped to a Colab Tesla T4; the local Windows machine still cannot compile or execute CUDA.
- Colab CUDA results remain a separate environment and cannot be combined with Stage 3 Windows CPU timings to infer a workstation crossover.
- Stage 5 v2 selector calibration uses only same-machine calibration-phase `selector_steady_state` measurements for the fixed preregistered workload.
- Stage 5 v2 calibration and held-out acquisition are separate; a canonically hashed calibration-only policy must be frozen before held-out timings are exposed.
- Stage 6C remains optional and one-way dependent on the simulation core; presentation geometry cannot affect physics, display timing is never benchmark evidence, and no Auto backend selector exists.
- Stage 6D MPI uses replicated global state, deterministic contiguous force-target partitions, and full acceleration exchange. Hybrid OpenMP is confined to local target work under `MPI_THREAD_FUNNELED`.
- Stage 6D large-N display support does not change the exact O(N^2) algorithm: continuous viewer computation ends above N=1024, manual direct-force steps end above N=8192, and larger workloads are display-only.
- Barnes--Hut is the required approximate-force path for large-N work, not an
  optional enhancement. Stages 1 and 2 implement and validate the Sequential
  and OpenMP variants; Stages 3A/3B implement source-reviewed CUDA force and
  transactional integration but remain GPU-unverified. Stage 4A connects only
  the validated CPU variants to the viewer. Controlled large-N performance,
  CUDA viewer authorization, and RTX validation remain mandatory.
- CUDA viewer execution reuses the existing `CudaVelocityVerlet` allocation and
  advances one complete fixed-step frame batch per call. Host
  `SimulationState` remains authoritative at completed advancement boundaries;
  no CUDA/OpenGL interoperability, force-law change, or automatic selection is
  introduced.
- CUDA counterfactual execution reuses that same wrapper without modifying its
  kernels. Separate baseline/counterfactual objects own persistent allocations;
  both synchronized host results must exist before a candidate pair can be
  committed. Failure invalidates both CUDA executors and never retries on CPU.
- The extended catalog scenario is separate from the byte-for-byte unchanged
  legacy nine-body approximation. Physical inclusion rebuilds solver state;
  visibility is render-only; future curated removal eligibility is metadata
  only and no counterfactual controller is present.
- The 60-body direct-force scenario starts with a provisional 60-second fixed
  timestep. Bounded convergence rejected 300 seconds for fast satellites, but
  neither conservation nor same-initial-state timestep convergence is evidence
  of long-horizon agreement with JPL ephemerides.

## Assumptions and unresolved questions

- Intended RTX 4050 exact model, driver, power limit, CPU, and memory configuration are unknown.
- Faculty acceptance criteria for “realistic” horizon and tolerated position error need confirmation before final study.
- Stage 4 fixed CUDA at double precision; all predeclared comparison/conservation thresholds passed on the Tesla T4 without revision.
- OpenMP is validated locally with GCC 15.1.0/libgomp; the feature-based contract, not a version macro, must still pass under the final ordinary MSVC `/openmp` toolchain.
- The exact RTX 4050 host CPU topology is unknown; native preflight must discover it and programmatically derive the one list bound through pilot, calibration, policy, and held-out evidence.

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

## Stage 4: CUDA backend and controlled GPU benchmarking

### Status and objective

Stage 4 is **complete**. Preparation commit `ae6117b` was compiled and executed on a real NVIDIA Tesla T4. The accepted pilot, controlled benchmark, environment capture, CTest logs, and independent raw-data verification are retained under `data/benchmarks/stage4_cuda/`.

The objective is to validate the same direct O(N²) physical model in double precision and measure GPU scaling without implementing visualization, heterogeneous execution, automatic selection, or a final CPU/GPU crossover policy.

### Prepared implementation

- `NBODY_ENABLE_CUDA` defaults to `OFF`. CMake invokes `check_language(CUDA)`, `enable_language(CUDA)`, and `find_package(CUDAToolkit)` only when explicitly enabled. A requested but unavailable compiler produces a targeted error; CPU-only configuration is unaffected.
- `CudaForceBackend` implements `ForceBackend` for sequential-reference comparisons. One CUDA thread owns each destination body and visits source bodies in ascending index order.
- `CudaVelocityVerlet` owns reusable device and pinned-host buffers and performs old acceleration, position update, new acceleration, and velocity update kernels on one stream for every step.
- CUDA API calls, allocations, copies, event operations, kernel launches, event synchronization, and stream synchronization are checked. Kernels flag coincident/non-finite separations; host validation rejects invalid mass/state inputs and non-finite outputs.
- The implementation uses `double` for state, force, constants, and accumulation. No mixed precision, fast approximate algorithm, softening, tree method, or CPU/GPU decomposition is introduced.

### Numerical validation

Before observing GPU output, tolerances were fixed at acceleration absolute `1e-18 m/s²` or scaled `1e-12`; position absolute `1e-3 m` or scaled `2e-12`; velocity absolute `1e-8 m/s` or scaled `2e-12`; final energy/angular CPU-CUDA scaled difference `2e-12`; energy drift `5e-6`; angular drift `5e-10`; and normalized momentum/center-of-mass drift `1e-12`.

Real-GPU CTest passed `2/2`. `nbody_cuda_validation` covered analytic two-body and seeded 8/32/128-body force cases, analytic and seeded CPU/CUDA integration comparison, repeated CUDA runs, conservation diagnostics, collision rejection, and NaN rejection. No threshold changed. The log establishes that cross-backend differences were within tolerance but does not print their individual values; bitwise identity is not claimed. All 154 benchmark rows passed, and checksums are identical across repetitions within each N. Maximum benchmark relative energy drift was `1.1779626463387057e-13`; maximum relative angular-momentum drift was `3.740862421005083e-15`.

### Environment, provenance, and protocol

- Environment ID: `colab-Tesla_T4-20260923T173009Z`; Linux 6.6.122+; two logical host processors.
- GPU: Tesla T4, compute capability 7.5, 15,360 MiB, 40 multiprocessors.
- Driver 580.82.07. `nvidia-smi` displayed CUDA 13.0; `nvcc` independently reported toolkit/compiler 13.3 / V13.3.73. These are retained as distinct facts.
- Release build: CMake 3.31.10 and GCC/G++ 13.3.0.
- The private repository could not be cloned anonymously. The Windows host produced `solar-system-simulator.zip` with `git archive --format=zip ae6117b`; SHA-256 is `00fc9f3922ceb09647668a877a6193e10c7758338690fe945f16810e8e1b00fd`. A fresh export reproduced it byte-for-byte.
- Git metadata was absent by design after archive extraction. `STAGE4_COMMIT=ae6117b` explicitly declared the source snapshot, so null `git.head/branch/status` fields are preserved. This is archive-declared provenance, weaker than automated checkout verification but not unknown source code.
- Pilot archive SHA-256: `ba84c4c3438135e8450e4aeb6cec6d6743a5502edf4d82a06e2130bcab7c24a2`; full archive SHA-256: `7d3ed161f8bedbc70ec8896f087381bcfe324edcefeab7224660aa3804731326`.

The pilot used N=64/256/1024/2048/4096, 10 steps, one warm-up, and three repetitions (15 rows). The accepted full run used N=64/96/128/160/192/256/384/512/768/1024/1536/2048/3072/4096, 100 steps, `dt=10 s`, seed 24301, one warm-up, and 11 seeded block-randomized repetitions (154 rows).

### Timing definitions and results

- Device-compute timing uses CUDA events after H2D completion through the invalid-flag reset and all four kernels for every requested Velocity Verlet step.
- Transfer-inclusive timing uses CUDA events from H2D start through compute, D2H positions/velocities/status, and explicit completion.
- One-time CUDA context, stream, events, device allocations, and pinned-host allocations are reported separately as setup. Host structure staging, H2D, D2H, and full backend wall time are also separate columns.
- Scenario generation, diagnostics, CSV formatting, process startup, and allocation are excluded from both primary regimes.

Independent verification recomputed every supplied summary statistic, matched the deterministic shuffled schedule, found no missing/duplicate rows, confirmed positive finite timings and all-pass status, and retained every observation. Full per-N results are tabulated in `data/benchmarks/stage4_cuda/README.md` and machine-readable `verification.json` files.

- Median device-compute time rises from 34.903 ms at N=64 to 2140.987 ms at N=4096. Full-run CV is 0.132–0.545%, with no defensible exclusion.
- Median transfer-inclusive overhead is 0.021–0.074 ms and 0.0034–0.0939%. It is negligible for this GPU-resident 100-step definition; allocation and host staging are separate, so this is not a general transfer-cost claim.
- Pilot and full medians normalized per step agree within 1.1% at all five shared sizes.
- The direct algorithm performs O(N²) interactions, but this grid does not display quadratic elapsed-time growth: a 64× N increase produces 61.34× compute time (observed global exponent 0.990); 1024→2048 and 2048→4096 ratios are 1.885× and 1.942×. Hardware utilization and fixed launch costs can influence this curve, but causation was not directly measured.

### Reproducibility defects and repairs

- The uploaded shell script had CRLF endings and initially failed in Colab. `.gitattributes` now enforces LF for `*.sh`; the script contains no CR bytes.
- Git-only clean/HEAD checks rejected a legitimate archive tree. The helper now distinguishes `git-verified` and `archive-declared` modes, requires explicit commit plus archive SHA-256 in archive mode, and the runner records structured source provenance.
- `-G Ninja` was assumed although Ninja was absent. The helper now preflights `nvidia-smi`, `nvcc`, `cmake`, `ctest`, `ninja`, and `python3` and prints the explicit Colab `ninja-build` installation command without installing packages itself.
- The accepted run used equivalent temporary repairs: CRLF removal, omission of Git-only checks, and manual `ninja-build` installation. The permanent changes affect orchestration/provenance only; CUDA code, tests, workload, timing semantics, and thresholds remain exactly those in evidenced commit `ae6117b`, so a GPU rerun is not required to accept the existing scientific evidence.

### Local completion validation

- CUDA-disabled CPU Release configure/build succeeded; CTest `1/1` passed.
- CUDA-disabled OpenMP Release configure/build succeeded; CTest `2/2` passed.
- Host C++ syntax checks passed for the CUDA benchmark and CUDA test translation units.
- Python runner/verifier compile, help, and evidence checks passed; Bash syntax and LF checks passed.
- `NBODY_ENABLE_CUDA=ON` still fails locally with the intended targeted diagnostic because `nvcc` and NVIDIA hardware remain unavailable. Real CUDA build/test evidence comes from the retained T4 run, not this machine.

### Scientific boundary and next requirement

No CPU control was measured on the same Colab VM. Stage 3 Windows CPU times must not be divided by Stage 4 Colab CUDA times; no CPU/GPU speedup, crossover, or selector calibration is supported. Stage 4 establishes real-NVIDIA buildability, CUDA correctness within frozen tolerances, repeatability, conservation, stable T4-only scaling, and negligible transfer overhead under the declared GPU-resident timing scope. It does not answer the final research question.

The next methodological stage is a controlled **same-machine** sequential/OpenMP/CUDA experiment on the intended NVIDIA-capable workstation, with one frozen build/protocol/environment and accuracy gate. Only then may calibration/held-out selector work begin. Stage 5 is not started here.

## Independent sequential conservation characterization

This completed study is an independent numerical-method experiment after Stage
4; it is not Stage 5 and provides no backend crossover or selector evidence.
The research question was how the existing fixed-step Velocity Verlet method's
energy and angular-momentum drifts change with timestep and simulated duration.

The clean harness commit `6d9a162` ran only the deterministic sequential
backend in binary64. Analytic circular two-body and seeded 32-body scenarios
each crossed timestep multipliers `0.25, 0.5, 1, 2, 4` with duration
multipliers `1, 10, 100`. Diagnostics were computed initially and after every
step. Relative energy drift uses `abs(E-E0)/max(abs(E),abs(E0),1 J)`; relative
angular drift uses the vector difference
`norm(L-L0)/max(norm(L0),1 kg m^2/s)`. Existing bounds remained unchanged at
`5e-6` and `5e-10`.

All 30 rows passed. Worst energy drift was `3.8964650316226864e-10` in the
analytic two-body case at timestep multiplier 4 and duration multiplier 100;
worst angular drift was `1.215493287078012e-13` in the analytic case at
timestep multiplier 0.25 and duration multiplier 100. These are respectively
about 12,832 and 4,114 times below the frozen bounds. An exact repeated seeded
configuration matched in every scientific field.

The analytic coarse-timestep energy maxima remain nearly constant across the
tested durations while final errors are commonly much smaller, consistent with
bounded oscillation over this window. Its maximum energy drift grows about
16-fold per timestep doubling from multipliers 1 through 4 in this specific
circular setup. The longest seeded runs show about fourfold growth per
timestep doubling from multipliers 1 through 4 and increasing drift with
duration. Neither pattern is claimed as a formal order or proof of secular
behavior. Angular drift is non-monotonic and near a likely roundoff floor, but
the cause was not isolated.

The complete protocol, table, interpretation, and limits are in
`docs/conservation_study.md`; raw and derived evidence is under
`data/experiments/conservation/`. Runtime is incidental and cannot be used for
performance claims. The large observed accuracy margins mean the existing
bounds remain useful hard validity gates but are not discriminating within
these scenarios. Broader physical cases and horizons are required before any
threshold-policy change.

## Stage 5A: same-machine experiment preparation

### Objective and status

Stage 5 asks whether a calibrated policy can select the fastest valid
sequential, OpenMP, or CUDA backend on one workstation. Same-machine evidence
is necessary because Stage 3 measured Windows CPU/OpenMP while Stage 4 measured
CUDA on a Colab Tesla T4; ratios across those machines are invalid.

Stage 5A is **PREPARED, NOT COMPLETE**. It contains no RTX 4050 timing result,
crossover, policy boundary, speedup, or selector. Stage 5B must execute the
pilot and controlled full matrix on the intended Windows RTX 4050 machine.
The candidate frozen source revision is:

```text
4916b3c43298fa3f3bd15cb4af1698b5074abed3
```

The follow-up provenance commit records that preparation revision without
claiming a commit can contain its own hash.

### Predeclared protocol

Protocol `stage5-v1.0` uses the existing deterministic synthetic scenario,
seed 24301, binary64, direct Newtonian force model, Velocity Verlet, `dt=10 s`,
and 100 steps for the full run. It retains 11 measured blocks and one untimed
same-process warm-up immediately before every measured invocation.

- Calibration N: 64, 128, 192, 256, 512, 1024, 2048, 4096.
- Held-out/interstitial N: 96, 160, 384, 768, 1536, 3072.
- Pilot-only N: 64, 512, 4096 with 10 steps and three blocks.
- Every block contains sequential, OpenMP-1, each other explicitly approved
  OpenMP team size, and CUDA.
- Each complete block is shuffled by xorshift64*/Fisher-Yates with default
  order seed 20260926.

The held-out values cannot inform later policy thresholds. Any pilot-driven
matrix refinement requires an explicit protocol revision and a new frozen
source commit before the full run; pilot rows are never pooled into the final
dataset.

OpenMP counts are chosen only after topology capture. For total physical cores
P and total logical processors L across all CPU packages, the explicit list
contains 1, powers of two below P, P, and L, deduplicated and capped at L.
Requested and effective teams must match, and the identical list is required
for pilot and full.

### Timing, accuracy, and fairness

Legacy timing meanings remain unchanged: CPU `integration_compute`, CUDA
`device_compute`, and CUDA `transfer_inclusive`. A new primary
`selector_steady_state` metric measures CPU initial-state copy plus integration
against CUDA's existing backend wall interval: host packing, H2D, device
integration, D2H, synchronization, output materialization, and validation.
Reusable CUDA context/allocation setup remains separately reported. Scenario
generation, process startup, warm-up, diagnostics, checksum, serialization,
and runner I/O are excluded for all backends.

The CPU benchmark extension only widens an additional timer around the
existing state copy; it does not change the original timer or numerical work.
No force, integrator, CUDA, OpenMP, precision, or workload implementation was
optimized or altered.

Every configuration must pass existing cross-backend CTests and the unchanged
final-state conservation gates: relative energy drift `<=5e-6` and normalized
angular-momentum-vector drift `<=5e-10`. Failed or non-finite rows are
inadmissible and never averaged. Repeat checksums must match within each
configuration; sequential/OpenMP checksums must also agree at each N.

### Prepared tooling and pilot gate

`scripts/run_stage5.py` supports `--preflight`, `--dry-run pilot|full`,
`--pilot`, and `--full`, plus explicit output, seed, order seed, OpenMP list,
device identity, executables, CTest log, environment ID, and source commit. It
captures Git, Windows, CPU/topology, RAM, GPU, driver, CUDA, compiler, CMake,
Ninja, flags, OpenMP environment, optional NVIDIA telemetry, and executable
hashes. Raw evidence is tidy long-form; summaries include median, mean, sample
standard deviation, CV, min/max, IQR, and deterministic bootstrap median 95%
intervals.

`scripts/verify_stage5_evidence.py` independently reconstructs the randomized
schedule, checks the frozen split and workload, exact repetitions, row/metric
uniqueness, finite positive timings, CUDA timing ordering, correctness and
conservation, requested/effective teams, repeatability, source/environment
provenance, CTest success, summary recomputation, and manifest hashes.

Full mode requires both a passing pilot verification JSON and a human approval
JSON matching source revision, environment, protocol, and OpenMP list. The
runner never automatically continues from pilot to full. The complete Windows
procedure is in `docs/stage5_rtx4050_runbook.md`.

### Local validation

- GCC 15.1.0 CPU-only Release configure/build succeeded; CTest `1/1` passed.
- GCC 15.1.0/libgomp Release configure/build succeeded; CTest `2/2` passed.
- Sequential and OpenMP smoke rows exposed both unchanged integration timing
  and the new, wider selector timing; physics checksums matched.
- Three Python unit/synthetic-evidence tests passed, including full independent
  verifier execution on a generated pilot dataset.
- Pilot and full dry-run schedules were generated deterministically without
  invoking benchmark binaries.
- CUDA-enabled configuration failed locally with the intended targeted
  `no usable CUDA compiler` diagnostic. No CUDA build or execution is claimed;
  real validation remains mandatory on the RTX 4050.

No Stage 5 performance matrix was run on the development machine.

## Stage 6A: interactive OpenGL application foundation

### Objective and architecture

Stage 6A implements the minimal optional 3D desktop layer while leaving the
scientific and benchmark paths headless by default. The dependency flow is:

```text
existing SimulationState + ForceBackend + Velocity Verlet
    -> SimulationController
    -> read-only render snapshot
    -> OpenGL renderer
```

The controller owns the exact reset state, paused/running state, explicit
physical timestep, fixed steps-per-rendered-frame count, and manual backend.
It calls the existing integrator and Sequential/OpenMP `ForceBackend`; renderer
code contains no force or integration implementation. The loop polls input,
advances fixed steps, prepares a render-only snapshot, draws, and presents.
Wall-clock frame delta is never substituted for physical `dt`.

### Dependencies and implemented features

- `NBODY_BUILD_VISUALIZATION` is `OFF` by default. Headless configuration does
  not fetch or compile GLFW, GLM, GLAD, or OpenGL application code.
- Visualization configuration FetchContent-fetches exact GLFW 3.4 commit
  `a74efa0d5628b74adc0426af4c5710e287fa7c2c` and GLM 1.0.1 commit
  `0af55ccecd98d4e5a8d1fad7de25ba429d60e863`.
- A reproducibly generated GLAD 2.0.6 OpenGL 3.3 loader from commit
  `658f48e72aee3c6582e80b05ac0f8787a64fe6bb` is committed in-tree, avoiding
  a build-time Python/Jinja2 requirement and runtime DLL copying.
- The real OpenGL 3.3 renderer has embedded shaders, a single reusable smooth
  sphere mesh, perspective projection, depth testing, back-face culling,
  simple lighting/colors, dark background, and framebuffer resize handling.
- The orbit camera supports mouse/keyboard rotation, dolly, reset, pitch and
  distance clamps, and aspect-correct projection.
- Pause/resume, exact-state reset, exit, Sequential (`1`), and OpenMP (`2`)
  controls are implemented. The requested OpenMP thread count is explicit.
  The title/log identifies backend, thread count, run state, simulated time,
  and steps per frame.
- The default seed-24301 32-body scenario is explicitly labeled a deterministic
  synthetic fixture, not the Solar System. The existing analytic two-body
  fixture is also selectable.
- Physics remains double-precision SI state. Position scaling, mass-derived
  display radius, and palette are render-only and never modify physical state.
- CUDA selection fails before window creation with an actionable Stage 6A
  unavailable diagnostic. No Auto selector or silent fallback exists.

### Validation on the Windows development machine

- Fresh GCC 15.1.0 Release headless CPU configure/build succeeded; CTest `1/1`
  passed.
- Fresh GCC 15.1.0/libgomp Release headless OpenMP configure/build succeeded;
  CTest `2/2` passed.
- Release visualization+OpenMP configure/build succeeded. CTest `3/3` passed,
  adding controller/backend, render-space immutability, and camera tests.
- Sequential and OpenMP-4 viewer launches each created a real AMD Radeon OpenGL
  `3.3.0 Core Profile Context 24.9.1.240813`, initialized shaders/renderer, and
  remained in the application loop until deliberately interrupted.
- `--backend cuda` exited `1` with the expected explicit unavailable message;
  it did not crash or silently select another backend.

The automation environment could not reliably inspect or drive the native
window. Therefore body appearance/motion, pause/resume/reset keys, camera,
resize behavior, and live `1`/`2` switching require one user visual pass. Stage
6A status is **IMPLEMENTED / USER VISUAL VALIDATION PENDING**, not falsely
claimed fully visually validated. See `docs/visualization.md`.

The frozen Stage 5 experimental revision remains unchanged at
`4916b3c43298fa3f3bd15cb4af1698b5074abed3`; no Stage 3, Stage 4, conservation,
or Stage 5 protocol/evidence artifact was modified for Stage 6A.

## Stage 6B: Solar System application

### Scenario and scientific boundary

- The default viewer scenario now contains Sun, Mercury, Venus, Earth, Mars,
  Jupiter, Saturn, Uranus, and Neptune in stable order at J2000.0, Julian date
  2451545.0 TDB.
- Planet positions implement JPL Solar System Dynamics Table 1 mean elements
  and its published Kepler/ecliptic-coordinate procedure, valid for 1800--2050.
  Initial velocities are centered derivatives of that same model at plus/minus
  one hour. NASA NSSDCA factsheets supply masses and volumetric mean radii.
- Earth deliberately uses the Earth--Moon-barycenter orbit with Earth mass and
  no Moon. The heliocentric construction is shifted to center-of-mass position
  and velocity before integration.
- This is an approximate educational demonstration, not Horizons or an
  observational ephemeris. Long-horizon Solar System accuracy is not claimed.

### Application and separation

- Dear ImGui 1.92.9b is pinned to exact commit
  `f1cc2ae15e53a861a874c3034aae6798fde194ab` and integrated through its official
  GLFW/OpenGL3 backends. Headless builds remain independent of UI code.
- The dashboard exposes scenario, pause/reset, named speed, Sequential/OpenMP
  and thread count, selected-body data, focus controls, conservation drift,
  labels, bounded trails, and render-only radius scaling. CUDA and Auto remain
  visibly unavailable without silent fallback.
- The Solar timestep is fixed at six hours; speed presets use 1/4/16/64 fixed
  steps per rendered frame. Wall-clock frame time never enters integration.
- Trails are sampled in physical coordinates every simulated day, retained in
  bounded storage, and transformed only for rendering. Body radii and minimum
  visible size are also render-only. Scenario switch/reset reconstruct state,
  conservation baselines, trails, selection, and render transform together.
- Click selection, labels, moving-body focus, Sun focus, and overview navigation
  are implemented. ImGui input capture gates camera and shortcut handling.
- Current and maximum drift use the established diagnostics definitions:
  `abs(E-E0)/max(abs(E),abs(E0),1)` and
  `norm(L-L0)/max(norm(L0),1)`. FPS/frame time is labeled non-benchmark data.

### Validation on the Windows development machine

- Fresh GCC 15.1 Release CPU-only build: CTest `2/2` passed.
- Fresh GCC 15.1/libgomp Release OpenMP build: CTest `3/3` passed.
- Fresh Release visualization+OpenMP build: CTest `4/4` passed.
- The Solar regression constructs all nine finite bodies, checks center-of-mass
  removal and provenance-sensitive ranges, then integrates 352 six-hour steps
  (about one Mercury orbit). Observed relative energy drift was
  `6.46017e-09`; angular-momentum-vector drift was `1.61093e-16`, inside the
  committed `1e-4` and `1e-10` guards.
- Solar/Sequential, Solar/OpenMP-4, and Synthetic/Sequential each created a
  real AMD OpenGL 3.3 core context and remained in the application loop until
  deliberately interrupted. CUDA exited `1` with its intended diagnostic.

Native-window readability, label overlap, trails, selection/focus feel, resize,
and live backend switching still require a user pass. Stage 6B status is
**IMPLEMENTED / USER VISUAL VALIDATION PENDING**. No Stage 5 protocol or evidence
file was modified; its frozen source remains
`4916b3c43298fa3f3bd15cb4af1698b5074abed3`.

The user subsequently completed the Stage 6B visual pass. Its findings (inner
system compression, label collisions, default ImGui appearance, weak planet
identity, sparse initial context, synthetic clutter, and demo camera effort)
became the bounded Stage 6C requirements below.

## Stage 6C: final visual, UX, and presentation pass

### Implemented presentation system

- The primary ImGui panel now uses DPI-aware default-font scaling, deliberate
  spacing/rounding, restrained teal/gold colors, clear hierarchy, and a wider
  1080p-oriented layout. Raw vectors, maxima, fixed-step batching, trail
  capacity, seed, and provenance moved under collapsible technical details.
- Backend state explicitly presents Sequential/OpenMP as selectable, CUDA as
  not integrated, and Auto as awaiting Stage 5 calibration. OpenMP thread count
  appears only when OpenMP is active. No selector or fabricated CUDA path exists.
- Numerical health shows live energy drift against `5e-6` and angular-momentum-
  vector drift against `5e-10`, with PASS/WARNING mapping. Maximum values remain
  available in advanced details; bounds and diagnostic definitions are unchanged.
- Overview, Inner System, Earth, Jupiter, and Saturn camera presets provide
  deterministic linear-scale compositions. Body tracking and manual orbit/dolly
  coexist; `C` remains the obvious overview reset.
- Static orbit guides are sampled from the same JPL mean-element model and are
  clearly separate from fading, bounded, simulation-derived history trails.
  Actual positions always come from Velocity Verlet N-body integration.
- Screen-space label placement prioritizes selected body and Sun, tries bounded
  offsets, rejects overlaps, uses leaders for displaced labels, and suppresses
  off-camera bodies. Compute Experiment uses selected-only labels.
- Lightweight procedural materials give each major body recognizable identity.
  Saturn has a tilted translucent striped ring mesh; the Sun remains warm and
  emissive. A fixed-seed 720-point star sphere and a gold selected-body marker
  improve orientation without runtime assets or post-processing.
- Synthetic mode is named `Compute Experiment` and identifies its deterministic
  N/seed workload without implying astronomical or benchmark meaning. Viewer
  FPS remains explicitly labeled NOT BENCHMARK EVIDENCE.

All positions remain linearly scaled; the existing render-only body-radius
exaggeration and small per-body identity factors are documented and isolated.
Rings, materials, guides, stars, labels, and selection markers never enter the
simulation state or backend timings.

### Stage 6C validation

- Fresh Release CPU-only, OpenMP, and visualization/OpenMP configurations
  passed CTest `2/2`, `3/3`, and `4/4`, respectively.
- Headless coverage now includes finite reference-guide samples, exact camera
  preset targeting, selected-label priority/collision rejection, synthetic
  label policy, health mapping at/exceeding the frozen bounds, and render-only
  material metadata.
- Solar/Sequential, Solar/OpenMP-4, and Compute Experiment/Sequential created
  real AMD OpenGL 3.3 contexts with all Stage 6C shaders/meshes initialized and
  remained in the event loop until deliberate interruption. CUDA exited with
  the intended Stage 6C unavailable diagnostic.

No reliable automated pixel-level inspection was used, so final appearance and
interaction remain **IMPLEMENTED / USER VISUAL VALIDATION PENDING**. Stage 3/4,
conservation, and Stage 5 protocol/evidence files remain untouched; the frozen
Stage 5 source is still `4916b3c43298fa3f3bd15cb4af1698b5074abed3`.

## Stage 6C.1: corrective visual and compute-demonstration pass

- Ten years of the actual integrated Solar scenario produced minimum Sun--
  Mercury separation `4.60084e10 m` (`0.307547 AU`) versus physical contact
  distance `6.9814e8 m`. There is no physical intersection. A fixed render-only
  Sun-radius cap of `0.052` removes the exaggerated visual overlap at the
  validated perihelion without changing physics.
- Label layout retains a previous accepted body-relative screen offset while
  its real text rectangle remains valid, then tries alternate anchors with a
  seven-pixel gap. Selected body and Sun priorities, leaders, and low-priority
  suppression remain. Representative dense layouts and tiny-motion hysteresis
  are headless-tested.
- Solar View now defaults `Barycenter & real Sun motion` on. A gold origin
  marker is distinct from physical bodies; the warm fading Sun path consists of
  actual integrated `TrailHistory` samples. Explicit visual displacement
  amplification defaults to `10x` and can be returned to `1x`; it never changes
  state or stored history. No fake helix or external translation was added.
- Solar Compute explains that N=9 is unsuitable for meaningful speedup claims.
  Compute Experiment supports deterministic N=`32,64,128,256,512,1024`, resets
  dependent state on N change, uses selected-only labels, and switches from
  spheres to one point-sprite draw above N=64.
- The explicit live CPU comparison warms both paths, times the same isolated
  initial state/dt/fixed step count, leaves the visible controller untouched,
  and accepts results only after established mixed position/velocity tolerance
  checks. Its raw milliseconds and honest ratio are labeled NOT BENCHMARK
  EVIDENCE and invalidate on N/thread changes. CUDA and Auto remain absent.
- A development-machine N=512/OpenMP-4 diagnostic example observed 32 steps,
  Sequential `100.13 ms`, OpenMP `33.4279 ms`, ratio `2.99541x`, correctness
  PASS. This single live observation is not controlled evidence and establishes
  no selector threshold.

Fresh Stage 6C.1 validation and runtime smoke results are recorded in
`docs/visualization.md`. Subjective label/marker/trail and large-N UI appearance
remain **IMPLEMENTED / USER VISUAL VALIDATION PENDING**. No protected evidence
or Stage 5 protocol/preparation file changed.

## Stage 6D: final feature stage

Stage 6D completes the requested feature scope without changing the frozen
Stage 5 source/evidence revision
`4916b3c43298fa3f3bd15cb4af1698b5074abed3`.

- A shared direct-force kernel now serves Sequential, OpenMP, and MPI while
  preserving ascending source order, double precision, collision rejection,
  and existing numerical contracts.
- Optional MPI uses replicated global state, deterministic contiguous target
  partitions with lower-rank remainder ownership, local direct-force work, and
  `MPI_Allgatherv` to rebuild complete accelerations. All ranks perform the same
  Velocity Verlet updates. Hybrid mode uses OpenMP only in the local target loop
  and calls MPI under `MPI_THREAD_FUNNELED` outside parallel regions.
- MPI validation covers 1/2/4 ranks, non-divisible N=11, repeatability,
  Sequential agreement, finite values, frozen conservation gates, and 2 ranks x
  2 threads. The benchmark reports max-rank wall time and separates local
  compute from communication.
- `scripts/run_large_n_pilot.py` defines prospective protocol
  `stage6d-large-n-v1.0`: fixed geometric sizes, fixed work, four backend
  families, prospective per-configuration time budgets, complete timeout/error
  retention, raw logs, correctness/conservation/checksum, and environment plus
  topology provenance. It is infrastructure, not committed performance evidence.
- Compute Experiment now supports structured N through 100,000. Continuous
  advancement, trails, live conservation, and comparison are limited to 1024;
  explicit one-step direct-force work is bounded through 8192; larger workloads
  are display-only. The batched point path, selected-only inspection, and O(1)
  selector avoid per-body UI/trail structures.
- Solar display radii are camera-aware and screen-bounded. A projected
  Sun/Mercury clearance rule changes radii only, never centers. Sun through Mars
  use fixed screen slots and leaders; outer labels remain dynamically
  decluttered. The selected state is retained.
- The resizable panel is clamped to 380--680 px and 55% of display width. The
  camera and projection use only the complementary scene viewport. System-font
  fallback, wrapped property tables, complete units, professional section names,
  and tested hover help cover the primary interface. CUDA/Auto development diary
  text is not displayed and MPI remains a separate executable.

### Stage 6D local diagnostics and validation boundary

The Microsoft MPI 10.1.12498.52 N=512 diagnostic used 10 steps, dt=10 s, one
warm-up, three repetitions, seed 24301. Median elapsed seconds were MPI 1x1
`0.0489955`, MPI 2x1 `0.0236225`, MPI 4x1 `0.0229480`, and hybrid 2x2
`0.0184575`; all shared checksum `-382567846011.50037`, energy drift
`1.557002540516534e-14`, and angular drift `1.4733993293434963e-15`. These are
**SINGLE-MACHINE LOCAL DIAGNOSTICS -- NOT CONTROLLED BENCHMARK EVIDENCE**.
Same-session Sequential `0.0397994 s` and OpenMP-4 `0.0145921 s` observations
likewise do not establish a crossover or multi-machine scaling claim.

Fresh GCC 15.1 Release CTest matrices passed CPU-only `3/3`, OpenMP `4/4`,
visualization/OpenMP `5/5`, and MPI+OpenMP `8/8`. The MPI matrix includes ranks
1/2/4 and hybrid ranks 2 x threads 2. Builds used portable CMake 4.4.3, Ninja
1.13.2, GCC/libgomp OpenMP specification date 201511 (4.5), and Microsoft MPI
10.1.12498.52. Solar Sequential/OpenMP-4 and synthetic N=32/256/512/1024/2048/
100000 created real AMD Radeon OpenGL 3.3 contexts and remained in their event
loops until deliberate interruption. CUDA viewer selection exited `1` with the
intended explicit unavailable diagnostic. Native-app automation is unavailable
on this host, so visual composition, resize, hover, and interaction remain
**USER VISUAL VALIDATION PENDING**.

## Stage 6D.1: final Sun apparent-size correction

The preferred presentation-only Sun radius is now 20--36 pixels (40--72 pixel
diameter). Non-Sun bodies retain the existing 3--13 pixel policy. When the real
projected Sun--Mercury center separation cannot contain both desired discs plus
the existing one-pixel visual gap, Mercury shrinks continuously first to a
0.25-pixel sub-pixel marker. The Sun receives the remaining radius; if that is
below 20 pixels, clearance wins. Only render radii change.

The targeted 1920x1080 regression measured Overview Sun `4.79131 px`, Mercury
`0.25 px`, and center separation `6.04131 px`; Overview is therefore
clearance-limited and the Sun occupies the maximum permissible radius. Inner
System measured Sun `23.2458 px`, Mercury `0.25 px`, and separation `24.4958
px`. Tests also cover preferred lower/upper bounds, forced clearance override,
unchanged centers, and continuity under a 0.01 camera-distance change. The
focused visualization model test passed `1/1`; no physics, MPI, large-N,
benchmarking, CUDA, Auto, Stage 5, UI, label, typography, or control code changed.

## Stage 6D.2: remove orbital-phase-dependent Sun breathing

Human validation identified that Stage 6D.1's continuous clearance rule still
made the displayed Sun radius visibly pulsate as Mercury changed projected
separation. Continuity alone was insufficient. Stage 6D.2 removes the
Sun--Mercury separation from both the sizing API and render loop.

The stable target is
`clamp(r_render * H / (2 * D_camera * tan(22.5 degrees)), 20, 36)` pixels,
where `r_render` contains only the explicit presentation radius setting, `H` is
viewport height, and `D_camera` is the camera zoom distance. The fixed 45-degree
projection supplies the half-angle. The target is converted to a world glyph
radius using the Sun's actual camera depth, ensuring the rendered result equals
the target without allowing position to select it. Thus only camera zoom/preset,
viewport/projection, and explicit presentation sizing can influence Sun size.

The replacement regression advances 352 six-hour steps (one Mercury orbit) at
fixed camera and viewport. The unclamped Sun radius remained `27.9539 px` within
`1e-4 px`; Overview was `20 px` and Inner System `36 px`. It also checks zoom
direction and local smoothness, bounds, unchanged physical centers, and no state
mutation. At distant views, exaggerated Sun/Mercury glyphs may overlap. Existing
depth rendering and stable callouts preserve association while physical centers
and orbits remain exact. The focused visualization test passed `1/1`.

## Stage 6D.3: inner-planet orbital-shape verification

Verification found no physical or rendering defect and made no camera or
presentation change. The initializer preserves the nonzero JPL J2000
eccentricities through Kepler solution, 3D orientation, centered velocity
differentiation, and the common barycentric shift. Initial Cartesian osculating
eccentricities for Mercury/Venus/Earth/Mars were `0.205628851`, `0.006788612`,
`0.016723353`, and `0.093428916` versus source values `0.20563593`,
`0.00677672`, `0.01671123`, and `0.09339410`.

A 700-day nine-body run measured extrema-derived eccentricities `0.205552626`,
`0.006836837`, `0.016738451`, and `0.093499109`; full values and distances are
recorded in `docs/visualization.md`. Mercury ranged from `4.600927371e10` to
`6.981784139e10 m`, while the existing ten-year regression independently found
`4.600842350e10 m`. The 4097-sample reference guides recover their source
eccentricities within `1e-6` absolute. Daily trail samples preserve Mercury's
radial variation through the common linear render transform and direct
`GL_LINE_STRIP` upload. No guide drives physics and Sun-motion amplification is
confined to the separate Sun trail.

The intrinsic Keplerian axis ratios are close to unity—Mercury `0.978629`,
Venus `0.999977`, Earth `0.999860`, Mars `0.995629`—so eccentricity is primarily
visible as focus offset and perihelion/aphelion variation, not obvious oval
flattening. The current Inner System projection produced Mercury covariance
axis ratio `0.291186`, major/minor radii `55.1233/16.0512 px`, and a `7.00421 px`
Sun-to-centroid offset. Its geometry is not circularized; subtle low-alpha guides
and partial fading trails can obscure the focus relationship. No eccentricity
exaggeration or ephemeris-grade claim was added. Stage 6D.2 still reports stable
Sun radii of Overview `20 px`, Inner System `36 px`, and fixed-camera orbit test
`27.9539 px`.

Focused validation rebuilt and passed `nbody_solar_system_tests.exe` in both the
CPU and visualization configurations and passed `nbody_visualization_tests.exe`.
The real launch
`nbody_visualizer.exe --scenario solar --backend openmp --threads 4 --width 1920 --height 1080`
created an AMD Radeon OpenGL 3.3 Core context and selected OpenMP with four
threads. This is a launch/context smoke result, not subjective visual approval.

## Stage 5 v2.0 pre-RTX protocol remediation

The immutable adversarial audit at `docs/audits/pre_rtx_final_audit.md` records
the pre-remediation NO-GO. Stage 5 v2.0 prospectively supersedes the unexecuted
v1.0 preparation without rewriting it and without changing physics, the
scientific workload, numerical gates, timing regions, checksums, or historical
Stage 3/4/conservation evidence.

The repaired OpenMP contract tests the actual subset used—enabled compilation,
real requested/effective teams, explicit `num_threads`, `schedule(static)`,
completion, numerical equivalence, and deterministic repetition—rather than
requiring `_OPENMP >= 201511`. Historical GCC/libgomp OpenMP 4.5 reporting
remains factual; ordinary MSVC `/openmp` is acceptable only if the semantic
suite passes.

Evidence custody is now pilot -> calibration-only acquisition -> deterministic
policy freeze -> held-out-only acquisition -> evaluation without refitting.
The policy compares median `selector_steady_state` using piecewise-linear
log-log interpolation. Ties resolve by Sequential, ascending OpenMP thread
count, then CUDA. Canonical policy content excludes timestamps from its hash;
the separate freeze binds policy, calibration manifest, source, build,
environment, and derived topology.

Controlled preparation requires an absent build directory and clean exact
source, embeds source/protocol/toolchain identity in benchmark/test binaries,
validates Release Ninja/MSVC/CUDA `sm_89`, hashes executables and Stage 5
scripts, runs CTest itself, and binds its retained log. Windows CIM supplies P
physical cores and L logical processors; code derives sorted unique `1`, powers
of two below P, P, and L capped at L, with no guessed fallback. Calibration
re-verifies the actual pilot and requires human approval bound to its manifest.

The same-machine selector claim is deliberately N-only for the fixed seeded
binary64, direct-force, Velocity-Verlet, dt=10 s, 100-step workload. Broader
scenario, step-count, precision, accuracy-budget, seed, and cold-start inputs
remain future work. The detailed finding trace is in
`docs/stage5_v2_remediation_trace.md`.

Local validation uses CPU/OpenMP and synthetic fixture custody tests only;
fixtures are not benchmark evidence. No NVIDIA pilot, calibration, held-out
run, policy result, or Stage 5B collection is claimed. The exact frozen source
SHA is supplied only by the external post-commit deployment packet, avoiding
impossible commit self-reference. The first NVIDIA rehearsal must stop after
pilot verification and manifest-bound approval.

The final local matrix passed 29/29 Stage 5 v2 adversarial tests, 4/4 exact-SHA
CPU-only Release CTests, and 5/5 exact-SHA OpenMP Release CTests. A fresh clone
ran `prepare_stage5_build.py` end to end with native CIM topology (P=8, L=16,
derived threads `1,2,4,8,16`), embedded source/protocol identity, binary
hashing, and a passing controlled build manifest. Two earlier clean-room
attempts conservatively retained their artifacts after exposing CTest output
format assumptions; focused regressions cover the corrected combined-stream
and current-summary handling. The expected CUDA-on/no-nvcc configure stopped
with the explicit unavailable diagnostic. Native CUDA/MSVC/RTX execution was
not possible on this AMD-only host.

## CUDA visualizer integration

### Implementation record

Development was performed on `feature/cuda-visualizer` from baseline
`a9b03cd89e585ca6dad154c3b3abdfeb3625578b`. At the start of this ledger audit,
the implementation and its documentation were present as uncommitted working-
tree changes; no push had occurred.

The reviewed implementation/build/test set was then committed locally as
`45be2d33695c774199a462be120e5cd8800624bc`; documentation and ledger changes
were deliberately reserved for the second focused commit. The branch remained
`feature/cuda-visualizer` and was not pushed during this workflow.

`SimulationController` still owns the authoritative host `SimulationState`.
Sequential and OpenMP retain their existing CPU Velocity Verlet paths. CUDA
owns one RAII `CudaVelocityVerlet` executor for the active body count and calls
`run(state, steps, dt)` once per requested frame or manual-step batch. The
executor's device, event, stream, and pinned-host allocations are reused while
the backend and body count remain unchanged. Each completed call synchronizes,
materializes a host result, and only then replaces controller state.

Backend switching is transactional: OpenMP or CUDA replacement resources are
constructed before the active backend fields are changed. Switching never
regenerates or resets the current state or simulated time. A failed CUDA call
leaves the prior host state intact. Scenario replacement constructs a new CUDA
executor before committing a different-sized state; same-sized replacement
reuses the existing allocation. CUDA is offered through `--backend cuda
--device N`, the UI, and shortcut `3` only when support is compiled and that
device is available. There is no silent CPU fallback.

CMake links `nbody_cuda` to the visualization model only under
`NBODY_ENABLE_CUDA=ON`; CUDA-disabled builds do not require NVCC, CUDA runtime
libraries, or NVIDIA hardware. The interactive workload policy now uses an
explicit Sequential/OpenMP/CUDA classification. CUDA continuous advancement
ends above N=1024, bounded manual single-step execution ends above N=8192, and
larger cases remain display-only. This is an interactive safety rule, not
performance evidence or an Auto-selection policy.

Relevant changes cover `CMakeLists.txt`, the visualization controller,
application/backend UI, presentation help, workload policy, focused
visualization/CUDA tests, current README/architecture/visualization guidance,
and this ledger. Physics sources, CUDA kernels/host wrappers, scenarios,
diagnostics, trails, renderer physics separation, Stage 5 tooling, benchmark
programs, and evidence files are unchanged.

### Bug and risk record

Resolved integration hazards are: the formerly unconditional CUDA CLI
rejection; per-force-evaluation transfer/synchronization that would have
resulted from inserting `CudaForceBackend` into the CPU integrator; state loss
during backend switching; stale size-dependent allocations after scenario
replacement; mutation of host state after a failed CUDA batch; accidental
Sequential/OpenMP workload-policy inheritance; and silent fallback while
reporting CUDA.

Open risks remain:

- NVCC compilation, CUDA linkage, device execution, and CUDA/OpenGL process
  coexistence have not been exercised on this machine.
- CUDA execution synchronizes on the viewer thread. A long batch can delay
  window event processing even though its state boundary is correct.
- CPU time advances by repeated additions while CUDA records
  `initial_time + steps * dt`; non-exact binary timesteps may differ in the
  final bits although physical timestep semantics are unchanged.
- CPU/CUDA state agreement is tolerance-based, not bitwise; compiler
  contraction and device math can produce expected small differences.
- Device availability is captured for UI capability at application startup.
  Later device/driver loss raises an error and preserves host state; it does
  not fall back automatically.
- CUDA resource recreation and failure paths are covered by GPU-dependent
  tests but remain pending real-device execution.

### Validation record

On the AMD-only Windows development machine, the final Release validation was:

- Sequential-only viewer (`NBODY_ENABLE_OPENMP=OFF`,
  `NBODY_ENABLE_CUDA=OFF`): configure/build passed and CTest passed `5/5`.
- OpenMP viewer (`NBODY_ENABLE_OPENMP=ON`, `NBODY_ENABLE_CUDA=OFF`):
  configure/build passed and CTest passed `6/6`.
- Focused `nbody_visualization_tests` passed independently in both builds after
  the final test review.
- A CUDA request against the CPU-only executable exited `1` with the explicit
  not-compiled diagnostic before OpenGL initialization; no NVIDIA dependency
  was needed by either CPU build.
- The first all-CTest invocation encountered only a sandbox denial in Python's
  default temporary directory. Re-running with `TEMP`/`TMP` inside the writable
  build tree produced the passing totals above without source changes.
- `nvcc` was unavailable. CUDA-enabled CMake configuration stopped with the
  intended `no usable CUDA compiler was found` diagnostic. Host-compiler syntax
  checks passed for the CUDA-conditional controller code and CUDA test source,
  but these checks are not CUDA compilation or runtime evidence.

`nbody_cuda_validation` and the new `nbody_visualization_cuda` are registered
with CTest skip return code `77`. When CUDA support is compiled but no device is
available, they report an explicit skip instead of a pass. On this machine the
CUDA configure gate occurs before those targets can be built, so neither GPU
test was executed or counted as passing. CUDA viewer compilation and runtime
validation are therefore **UNVERIFIED** here.

The GPU visualization test is prepared to check batched time advancement,
pause/manual-step/reset behavior, Sequential-to-CUDA and CUDA-to-Sequential
state preservation, different-size scenario replacement, transactional switch
failure, failed-advance host-state preservation, and Sequential/CUDA agreement
under the established mixed position/velocity tolerances.

### Frozen Stage 5 boundary

This integration does not alter the frozen Stage 5 v2 scientific workload,
force law, binary64 precision, Velocity Verlet equations, `dt=10 s`, 100-step
measurement case, numerical gates, timing definitions, policy algorithm,
evidence custody, scripts, benchmarks, or retained evidence. It does not add
Auto selection and viewer FPS/frame time remain non-evidence. The existing
CUDA validation executable only adds explicit no-device skip semantics; its
device-present assertions and `--build-info` path are unchanged.

### Outstanding RTX 4050 validation

Before CUDA viewer integration can be called hardware-validated on the target
RTX 4050, a separate non-Stage-5 measurement session must:

1. Configure and build the visualization with the intended VS2022/MSVC host
   toolchain, CUDA 12.8.x-class toolkit, `NBODY_ENABLE_CUDA=ON`, and Ada
   `sm_89` support.
2. Confirm `nbody_cuda_validation` and `nbody_visualization_cuda` execute and
   pass on device 0; a skipped result is not acceptance.
3. Run the complete CPU/OpenMP/CUDA CTest matrix without changing tolerances.
4. Launch a real OpenGL Solar view with `--backend cuda --device 0`, verify
   advancement, pause/manual step/reset, scenario size changes, and live
   Sequential/OpenMP/CUDA switching without state reset or fallback.
5. Exercise a failed CUDA selection/advance and confirm the displayed host
   state remains valid and unchanged.
6. Observe UI responsiveness at the existing bounded frame batches and record
   any synchronous-stall limitation without interpreting FPS as benchmark
   evidence.

This feature validation must remain separate from Stage 5 pilot/calibration/
held-out evidence collection and must not consume, tune, or reinterpret the
frozen selector protocol.

## Astronomical catalog foundation

### Implementation record

The first gravitational-sandbox increment adds only a dependency-isolated
`nbody_astronomy` catalog library and focused tests. It does not contain a real
astronomical body record, retrieve Horizons data, construct a new simulation
scenario, or change the frozen legacy nine-body state.

Catalog ingestion definitions are mutable staging values. Admission validates,
sorts, converts, and stores them as records exposing only const access. Each
record requires a stable ID, display name, body class, at least one external
identifier, finite geometric state vector, finite positive sourced GM, finite
positive sourced mean radius, three complete provenance records, and explicit
model limitations. A provenance record binds its source name, URI, record ID,
retrieval time, and 64-hex-digit SHA-256.

The shared coordinate contract is deliberately narrow: one named finite epoch,
TDB, ICRF, Solar System barycenter origin, geometric vectors, metres, and metres
per second. Every admitted record repeats and must exactly match that contract,
preventing mixed epochs, origins, frames, aberration modes, or units. Explicit
contiguous order values are sorted independently of ingestion order. Stable
IDs, display names, and authority/value identifier pairs are unique; parents
must exist and precede their children.

Mass is never accepted as an independent catalog input. It is derived as
`mass_kg = GM_m3_s2 / gravitational_constant`, using the existing project
constant after validating both operands and the result. The primary Solar
System policy additionally requires at least 60 admitted bodies, exactly one
parentless star, a parent for every non-star, and Solar-compatible parent
classes. This is an admission gate, not evidence that any candidate roster has
passed it.

No 60-body candidate roster is present in the repository or the locally
available task attachments. Its identifier, state-vector, GM, radius,
provenance, substitution, and model-limitation audit is therefore **PENDING**.
No roster item or substitute has been inferred or invented in this increment.

### Bug and risk record

The first focused GCC build exposed that `std::vector::emplace_back` could not
invoke the immutable record's private constructor through the allocator. The
catalog now constructs the record inside its friend and moves the completed
value into the vector. The public immutability and validation boundary are
unchanged.

Open scientific and implementation risks are:

- the approved >=60-body roster and any defensible substitutions remain
  unavailable for audit;
- Horizons solution identities, common-epoch vectors, GM/radius source
  snapshots, retrieval times, and checksums do not yet exist;
- exact equality of the repeated coordinate contract is intentional for a
  generated common-epoch dataset but requires deterministic canonicalization
  by the future importer;
- a positive mean radius is currently mandatory, so a real object without a
  defensible radius will be rejected rather than admitted as a decorative or
  massless placeholder;
- the catalog does not yet create `SimulationState`, drive rendering, expose
  GUI selection, or support removal experiments.

### Validation record

Focused Release catalog execution passed and covers GM-to-mass conversion,
invalid GM/G rejection, immutable const record access, ingestion-order
independence, stable-ID lookup, retained parent identity, shared-coordinate
rejection, unsupported-vector-mode rejection, provenance/hash/model-limitation
rejection, duplicate stable/external identity rejection, unknown and late
parent rejection, non-finite state rejection, positive GM/radius admission,
and the exact 60-record lower boundary. The 60-record case is explicitly a
synthetic unit-test fixture, not astronomical data.

Final CUDA-disabled Release regression results on the AMD Windows development
machine were:

- Fresh headless/default CPU-only configuration with visualization, OpenMP,
  CUDA, and MPI disabled: configure/build passed and CTest passed `5/5`.
- Sequential viewer configuration: CTest passed `6/6`.
- OpenMP viewer configuration: CTest passed `7/7`, including the existing
  OpenMP validation.
- No CUDA compiler, device test, MPI test, benchmark, Horizons request, or
  real-data validation was run for this increment.

### Frozen boundaries and deferred work

No existing physics, Solar scenario, CUDA, OpenMP, MPI, visualization,
synthetic workload, benchmark, Stage 5 protocol, Stage 5 tool, or retained
evidence file changed. `nbody_astronomy` links to `nbody_core`, while existing
Stage 5 executables do not link to the new library.

Curated one-body removal, paired long-horizon impact measurement, expert-mode
Sun-removal architecture, scenario composition, GUI controls, CUDA experiment
execution, and simulation-state composition from catalog selections remain
unimplemented. The mandatory large-N roadmap is Sequential Barnes--Hut, OpenMP
Barnes--Hut, CUDA Barnes--Hut, cross-solver numerical verification, and
controlled large-N performance evaluation; none of those stages has begun.
Existing 50K--100K synthetic selections remain display-only at those sizes. A
1920x1080 RTX 4050 target of 30--45 FPS is a future measurable goal, not an
achieved or predicted result.

## Astronomical 60-body dataset v1

### Dataset and source-custody record

The next catalog-only increment creates `jpl-ssd-j2000-60-v1`, version `1.0.0`,
with exactly 60 real, individually identified, gravitating bodies in explicit
parent-before-child order. The roster SHA-256 is
`2df8e58be8924f7a49b44ca761db8ffb0e0a1204689844044624c907d65273cd`.
The source snapshot manifest SHA-256 is
`82b271f3fe7470172693c9ab4810a664153ab59f94f293498000b19504749229`.
Generation produced normalized catalog SHA-256
`3813b0c4e31b0e58f60fb33f7a23485afe853f32601a49c14459f850f2d759d0`
and typed C++ data SHA-256
`b60199f05e1d29359b37ef9c25416f9c80f0770c18fb1e5098060495b4c37482`.

The immutable raw archive contains 60 exact Horizons batch responses, 13 exact
JPL Small-Body Database API responses, and three exact JPL parameter sources.
Its manifest retains every request parameter, target identifier, source URL,
UTC retrieval time, media type, byte count, and response SHA-256. The three
shared-source hashes are:

- JPL Horizons GM kernel:
  `169cfed3b0927e73929d0a1b5c931f9afb5167a83b921064127ffc54a673df0c`;
- JPL planetary physical parameters:
  `e49a487947295c389ba6e5833dd8fb04b63e826d39f70c123881882096e5e68b`;
- JPL satellite physical parameters:
  `d48a005e6869b199796877492c894abb4c0fd49fe4819d27dfeb280f483aa440`.

Each state was admitted only after matching the roster's unique Horizons/SPK
identity and expected solution, exact `JD 2451545.0 TDB` epoch, Solar System
barycenter origin, J2000 ICRF/equatorial frame, geometric correction mode, and
`KM-S` output units. Generation converts positions, velocities, radii, and GM
to SI and derives mass from the existing project gravitational constant.
Body-specific planet GM values are used rather than planet-system GM values so
that independently represented satellite masses are not counted twice.

The admitted roster is the 51 approved original candidates plus nine approved
replacements: Epimetheus, Helene, Atlas, Prometheus, Pandora, Nix, Hydra,
Thalassa, and Eunomia. Eris, Haumea, Quaoar, Orcus, and Gonggong remain explicit
in the deferred expansion queue, together with the other preliminary candidates
whose current source package was not admitted. No failed or missing datum was
silently replaced during acquisition or generation, and no unresolved source
identity discrepancy was found for the frozen roster.

### Validation record

The offline custody test verifies all 76 archived source checksums, response and
solution identities, request parameters, roster identity, SI conversions, and
byte-for-byte deterministic reconstruction of both generated artifacts. The
C++ dataset test verifies exactly 60 unique admitted bodies, deterministic
ordering, parent relationships, the common coordinate contract, finite state,
positive GM/radius, GM-derived mass, body-only Jupiter GM, and representative
Sun--Earth, Earth--Moon, Jupiter--Io, Saturn--Titan, and Pluto--Charon relative
distance/speed envelopes.

Fresh CUDA-disabled Release validation on the AMD Windows development machine:

- CPU-only configure/build passed; CTest passed `7/7`.
- OpenMP configure/build passed; CTest passed `8/8`, including the existing
  OpenMP validation.
- The focused catalog/dataset/snapshot/legacy-Solar tests passed `4/4` in the
  OpenMP build.
- The first sandboxed full CPU invocation exposed only denied Windows temporary
  directory creation in the unchanged Stage 5 tooling test. The same unchanged
  suite rerun outside that restricted sandbox passed `7/7`; no Stage 5 repair
  or source modification was made.
- No CUDA compilation, NVIDIA execution, MPI test, benchmark, or long-duration
  ephemeris comparison was performed. CUDA viewer validation on the RTX 4050
  remains pending.

### Scientific limitations and frozen boundaries

Catalog admission proves identity, provenance custody, coordinate consistency,
and deterministic conversion; it does not prove long-duration orbital accuracy.
Source epochs and identifiers are exact, but several small-body and small-moon
GM/radius estimates carry material uncertainty (including the documented large
Proteus GM uncertainty), and covariance is not propagated. The simulation does
not reproduce the complete JPL dynamical model and does not include relativity,
extended-body gravity, tides, nongravitational comet acceleration, adaptive
timesteps, or close-encounter regularization.

The legacy approximate nine-body scenario and its regression remain unchanged.
No GUI selector, body-removal/counterfactual controller, Barnes--Hut solver,
CUDA kernel, force law, integrator, benchmark, Stage 5 script, Stage 5 protocol,
or experimental evidence changed. Frozen Stage 5 remains unchanged. Sequential,
OpenMP, and CUDA Barnes--Hut plus numerical verification and large-N performance
evaluation remain mandatory and unimplemented.

## 60-body scenario composition and visualization

### Implementation record

This uncommitted increment on `feature/cuda-visualizer` begins at
`fb44b3537f3a3b7d7718ffc8e92f4db1022ccc5d`. It adds a second, strictly separate
astronomical scenario backed by `jpl-ssd-j2000-60-v1`; the legacy approximate
nine-body constructor, values, timestep, and reference paths remain unchanged.
The default extended composition admits all 60 catalog records. Explicit
stable-ID subsets are validated against the immutable catalog and emitted in
deterministic catalog order, retaining parent relationships and complete
catalog states. Unknown/duplicate IDs and fewer than two solver bodies are
rejected. Excluded planets, the Sun, and included children whose parents are
absent produce visible composition warnings.

Solver inclusion, rendering visibility, and future curated counterfactual
eligibility are distinct data. Applying a subset replaces the physical solver
state; hiding an individual body or class changes only a `RenderBody` visibility
flag. Effective visibility is the intersection of active subsystem membership,
the user's class filters, and per-body preferences; these masks remain separate
so changing a filter cannot reveal an out-of-profile body. Scenario and subset
replacement preserve the controller's prior paused/running state while loading
the replacement at time zero. The future eligibility flag currently identifies
planets/dwarf planets; it does not expose removal controls or alter forces.

Overview, Inner System, Earth-Moon, Jupiter/Galilean, Saturnian, Uranian,
Neptune-Triton, Pluto-Charon, and asteroid-belt profiles resolve membership and
focus from stable IDs. Local profiles subtract their moving origin in binary64
before float conversion and freeze the view scale until another profile/focus
operation. Bodies, trails, reference guides, labels, rings, picking, and
markers use that transform. Local history trails subtract the matching parent
history sample rather than a current parent position. Absolute simulation state
remains SSB-centered SI. Physical radii remain metadata; screen-space radius
exaggeration is explicitly presentation-only.

The UI exposes legacy/catalog/synthetic scenario selection, the 60-record
physical composition checklist, independent selected-body and class visibility,
stable-ID focus and subsystem profiles, backend/solver identity, dataset
provenance, fixed timestep, composition warnings, and scientific limitations.
Sequential/OpenMP behavior remains available without CUDA. Existing optional
CUDA selection and host-authoritative batched execution are reused unchanged;
no kernel, force law, integrator, automatic backend selector, or CUDA/OpenGL
interop was added.

### Scientific timestep and risk record

The initially proposed 300-second step was rejected. Over one simulated day,
the worst final parent-relative difference was `2.661430167e-2` between 300 and
150 seconds (`3448404.503 m`) and `6.676367043e-3` between 150 and 75 seconds
(`863729.1314 m`). The approximately fourfold refinement is consistent with
second-order convergence, but the 300-second fast-satellite phase error is not
acceptable as the starting viewer policy. The implemented provisional policy
is 60 seconds; its 60-versus-30-second one-day maximum parent-relative
difference was `1.069240913e-3` (`138269.1228 m`). The 300-second global
relative energy drift was `5.050393981e-11` and the 75-second drift was
`3.137197214e-12`, demonstrating why global conservation alone cannot screen
local moon-orbit accuracy. Metis was the worst normalized body at all three
comparisons. The one-day 60-second run's closest center separation was the
Jupiter--Metis pair at `1.831596045` combined sourced radii: no contact occurred
in that bounded check, but it is not a general close-encounter guarantee.

The 60-second step is **provisional**, not a claim of scientific validation.
Later-Horizons reference disagreement was not measured because this dataset
contains only the common initial epoch. Longer horizons, body-resolved error
budgets, oblateness-sensitive moons, small-body solution uncertainty,
close-encounter monitoring, and comparison with independent ephemeris states
remain open. The model is still Newtonian point-mass Velocity Verlet without
relativity, tides, multipoles, non-gravitational comet forces, adaptive steps,
or singularity regularization.

### Validation record

Fresh GCC 15.1.0 Release configurations on the AMD Windows development machine
completed as follows:

- CPU-only viewer (`NBODY_ENABLE_OPENMP=OFF`, `NBODY_ENABLE_CUDA=OFF`):
  configure/build passed; CTest passed `8/8`.
- OpenMP viewer (`NBODY_ENABLE_OPENMP=ON`, `NBODY_ENABLE_CUDA=OFF`):
  configure/build passed; CTest passed `9/9`, including OpenMP validation and
  60-body Sequential/OpenMP mixed-tolerance comparison.
- Focused visualization tests verify the unchanged legacy state, exactly 60
  default bodies, deterministic stable-ID subset order, parent/warning/error
  behavior, byte-identical catalog state transfer, stable-ID views, visibility
  independence and composed profile/class/individual filtering, translation-
  invariant local rendering, matching-history parent-relative trails, pause-
  preserving scenario/subset replacement, explicit paused manual advancement,
  running-state preservation, reset/time semantics, CPU/OpenMP agreement, and
  300/150/75 plus 60/30-second convergence.
- A CUDA request to the CPU-only executable exited `1` before OpenGL startup
  with the explicit not-compiled diagnostic. GPU tests remain registered in a
  CUDA build with skip return code `77`; a skip is not a CUDA pass.
- CUDA compilation/runtime and the required RTX 4050 viewer checks remain
  **UNVERIFIED** on this machine.

No Stage 5 source, protocol, policy script, benchmark, or retained evidence
changed. Frozen Stage 5 remains unchanged. Counterfactual experiments are still
pending. Sequential/OpenMP/CUDA Barnes--Hut, direct-force numerical comparison,
and controlled large-N evaluation remain mandatory and unimplemented.

## CPU counterfactual gravitational experiment engine

### Implementation record

This uncommitted increment on `feature/cuda-visualizer` begins at
`b131c86ad9d6d41724649c3893dbb268b36fb8f5`. It adds a headless
`nbody_experiments` library without changing the viewer controller or live
simulation state. Immutable snapshots copy stable identity, parent/radius
metadata, scenario/catalog identity, source provenance and simulation time,
the original coordinate-frame contract, and a shared fixed Velocity Verlet
timestep. Capture adapters validate the unchanged legacy nine-body state and
deterministic full/subset `jpl-ssd-j2000-60-v1` state rather than guessing from
body names or indices.

`CounterfactualExperimentRunner` owns independent baseline and one-body-removed
branches. Every survivor begins byte-for-byte equal and is mapped by stable ID
after counterfactual vector compaction. Initial removal policy admits only
snapshot-marked planets and dwarf planets; Sun, uncurated, massless, and
multiple-body removals remain unsupported. Both branches retain the captured
inertial/SSB origin and call the existing direct-force and Velocity Verlet APIs.
Sequential and optional OpenMP execution are explicit; an unavailable OpenMP
request fails instead of falling back. Batches publish paired state, time,
metrics, and bounded samples transactionally, with pause/cancel at publication
boundaries and rollback after either branch fails.

Stored metrics include baseline/counterfactual trajectory points, absolute and
parent-relative separation, fixed-scale normalized deviation with a positive
floor, velocity difference, final/max values and times, first threshold
crossing, and configurable persistence. A removed/absent parent makes
parent-relative metrics explicitly unavailable. Energy, momentum, and angular
momentum diagnostics are computed per branch against that branch's own initial
values; cross-branch energy difference is not treated as numerical error.
Escape classification remains explicitly unsupported.

Resource gates bound horizon, cadence, batch size, monitored bodies, retained
samples, issue records/message bytes, checked conservative memory admission,
and direct-force interactions. The memory value covers major histories,
transactional copies, state/identity/string/map/metric storage, and bounded
diagnostics but is an admission estimate rather than an allocator-exact upper
bound. Durations use checked integral-step conversion, the binary64 exact-
integer ceiling, and a stricter 100,000,000-step operational limit; sample,
memory, issue, and workload arithmetic is checked before iteration/allocation.
State and step guards cover non-finite values, invalid or duplicate
mappings/physical metadata, coincidence, branch-specific close approach and
sourced-radius overlap, non-advancing or mismatched paired time, conservation
drift, provisional timestep evidence, and unsupported horizons. Each accepted
step screens the linear relative segment between pre/post positions. This
detects endpoint-safe crossings but does not prove minimum distance along the
curved gravitational path. Scientific/model warnings remain distinct from
numerical invalidity and threshold significance.

Experiment elapsed time is derived from completed integer steps, not subtraction
of large absolute timestamps. Failed time progression or either branch's
integration failure rolls back state, metrics, histories, and progress to the
last paired publication. Warning deduplication includes branch identity, and
bounded storage exposes a dropped-issue count. Terminal failures use a separate
fixed 512-byte message record with truncation status; reporting a failed batch
cannot evict a committed warning or change its dropped-warning count. Swept
distance screening uses commonly scaled binary64 relative endpoints, a clamped
projection, fused interpolation, and stable norms. It does not rely on an
extended-range `long double`, and nonrepresentable relative geometry is an
explicit numerical failure rather than a silent no-encounter result. The runner
is explicitly single-thread-owned: advancement, result access, pause, and
cancellation APIs must not be called concurrently. Future GUI work requires
worker ownership plus immutable result publication or explicit synchronization.

### Validation and scientific boundary

Fresh GCC 15.1.0 Release configurations
(`build-counterfactual-p1-cpu` and
`build-counterfactual-p1-openmp`) with the complete optional viewer model
completed on the AMD Windows development machine:

- CPU-only (`NBODY_ENABLE_OPENMP=OFF`, `NBODY_ENABLE_CUDA=OFF`): build passed;
  CTest passed `9/9` after directing Python temporary files to the writable
  fresh-build directory.
- OpenMP (`NBODY_ENABLE_OPENMP=ON`, `NBODY_ENABLE_CUDA=OFF`): build passed;
  CTest passed `10/10`, including Sequential/OpenMP paired-run agreement.
- Focused adversarial coverage verifies exact initial survivors, physical force
  removal against a one-step analytic acceleration term, compacted stable-ID
  mapping, no branch recentering, initial zero deviation, paired time, threshold
  persistence, pause/cancel boundaries, atomic guard rollback, memory/workload
  limits, conservation diagnostics, non-finite/coincident/close-approach
  handling, deterministic Sequential repeatability, OpenMP gating/agreement,
  source-state isolation, legacy preservation, and full 60-body compatibility.
- Targeted remediation adds endpoint-safe swept-crossing, zero-relative,
  parallel, tangential, initial-overlap, receding, branch-divergent and
  non-finite proximity cases; branch-aware warning retention; integral-step
  boundary/NaN/infinity/representability and checked workload/memory overflow
  cases; non-integral timesteps; multi-batch step-derived elapsed time; large-
  timestamp stagnation after a committed batch; second-branch failure rollback;
  issue record/message caps with dropped-count reporting; full-warning-buffer
  rollback with exact preservation of committed histories/metrics/conservation;
  independent terminal reporting with an empty or full warning buffer; repeated
  failed-result inspection; and binary64-range swept geometry through finite
  near-limit endpoint crossings, parallel/receding/degenerate/grazing paths,
  ordinary astronomical scales, and explicit rejection of unrepresentable
  relative coordinates.

These tests verify the CPU engine contract, not the physical plausibility of an
instantaneous body removal or long-horizon Solar System prediction. The catalog
adapter retains the explicitly **provisional** 60-second timestep and an
initial one-day policy ceiling; neither the prior one-day refinement check nor
this engine establishes general counterfactual convergence. Point masses still
omit collision physics, close-encounter regularization, oblateness, relativity,
tides, and non-gravitational comet forces. Large separation is not escape.

CUDA paired integration remains pending. CUDA compilation/runtime was not
attempted and NVIDIA validation is not claimed.
Sequential/OpenMP/CUDA Barnes--Hut, numerical verification, and controlled
large-N performance evaluation remain mandatory and unimplemented. No physics
equation, integrator, CUDA kernel, astronomical source record, Stage 5 source,
protocol, policy, script, benchmark, or retained evidence changed. Frozen
Stage 5 remains unchanged.

## Counterfactual viewer and asynchronous CPU worker

### Implementation record

This increment was committed on `feature/cuda-visualizer` as implementation
commit `e7632e5` and documentation commit `dade679`, starting from
`163f41a071ee2931bdb795b3df9d892cb3cfd6f3`. The optional viewer now provides a
dedicated counterfactual panel without changing the live simulation controller.
It captures a value-owned current or initial astronomical snapshot, selects one
curated non-Sun removal by stable ID, configures Sequential or OpenMP execution,
horizon, cadence, absolute and normalized thresholds, persistence, and a
bounded monitored-body set, then displays checked work/memory estimates before
launch.

A persistent joinable worker exclusively owns each runner. The render thread
submits immutable job data, makes an atomic cancellation request, and consumes
mutex-protected immutable publications. Lightweight progress is published only
after synchronized paired batches; a bounded full result is exposed once at a
terminal boundary. Exceptions are contained in separate bounded worker-failure
metadata. Scenario switching requests cancellation and advances an
application-owned scenario generation on every replacement, including a
same-ID replacement. A publication is current only when both its immutable
scenario and worker generations match the accepted job; exact stable-ID order
is an additional composition check. Overflow is rejected before replacement,
and window shutdown cancels and joins the worker. The live controller is never
advanced or mutated by the experiment.

Pending, progress, cancellation, failure, and completion publications share a
bounded immutable executed-run descriptor: snapshot source/time/provenance and
coordinate contract, scenario/dataset identity, removal, backend/thread count,
timestep, horizon/steps, cadence, monitored/reference IDs, thresholds,
persistence, and admission estimates. The editable panel is labeled for the
next run and cannot rewrite active or historical metadata.

The completed overlay uses only stored samples. It draws a solid baseline and
dashed counterfactual path in the same inertial or common baseline-parent frame,
with double-precision subtraction before float conversion. An unmonitored
surviving local origin is retained as a bounded baseline-only reference at each
matching sample time, separately from monitored metrics and included in memory
admission. Removed, missing, unretained, or incomplete origins explicitly
suppress the local overlay; neither branch silently changes frames. Visible
paths are bounded and must pass profile, class, and individual visibility. The
results panel
reports sampled absolute, normalized, velocity, and available parent-relative
deviation; first/persistent threshold crossings; per-branch energy and momentum
drift; warnings, dropped warnings, and terminal failures. It makes no escape
classification.

The significance policy now requires both the configured absolute and
fixed-scale normalized thresholds. This is a metric-policy correction only;
the Newtonian force law, Velocity Verlet implementation, physical timestep,
CUDA kernels, astronomical records, and legacy nine-body initialization are
unchanged.

### Validation and risk record

Focused worker tests cover value-owned capture, eligibility, stable identities,
immutable publications, progress and completion, cancellation, restart/new
generation, same-ID replacement, away/back replacement, late stale completion,
generation overflow, active destruction/join, engine and worker failure
containment, immutable metadata despite during-run and post-completion editor
changes, scenario switching without live-controller mutation,
estimate/admission bounds, normalized threshold behavior, hidden matched-time
origin retention, unavailable-origin suppression, common-frame numerical
transforms, overlay visibility masks, and CPU-only OpenMP rejection or built
OpenMP execution. Fresh Release suite totals were recorded from independent
Ninja/GCC 15.1.0 configurations with the full optional viewer enabled:
CPU-only `build-counterfactual-remediation-cpu` passed `10/10` CTest tests, and
OpenMP `build-counterfactual-remediation-openmp` passed `11/11`. The focused
worker, counterfactual-engine, and visualization executables also passed
directly in both configurations. CUDA was disabled in both configurations; no
NVIDIA runtime result is claimed.

The UI is not a proof of physical accuracy. Instantaneous removal remains a
hypothetical intervention; large separation is not escape. The catalog timestep
remains explicitly provisional, the one-day engine ceiling is operational, and
point masses omit collision response, close-encounter regularization,
oblateness, relativity, tides, and non-gravitational comet forces. Memory
admission is conservative rather than allocator-exact. CUDA paired execution
and RTX 4050 validation remain pending. Sequential/OpenMP/CUDA Barnes--Hut is
still mandatory and unimplemented. Frozen Stage 5 source, protocols, policy,
scripts, benchmarks, and evidence remain unchanged.

### Counterfactual results-presentation usability pass

This uncommitted pass on `feature/cuda-visualizer` starts from
`1c8c23236a5d88277f8a704dc10619f79173fa5c` and changes presentation only.
Experiment controls now use responsive label/value geometry with explicit
units, distinct current-snapshot versus initial-state choices, clear curated
physical-removal and Sequential/OpenMP controls, concise definitions for
horizon/cadence/thresholds/persistence, and reasons for disabled launch states.
The experiment configuration and engine semantics are unchanged.

Committed immutable results are mapped by stable ID into a cached presentation
model. The panel provides an open summary and a sortable per-body table covering
final/maximum separation, the fixed normalization scale, normalized deviation,
velocity difference, optional parent-relative separation, first threshold
crossing, persistence time, and significance. Missing optional values display
as `N/A`. Baseline and counterfactual conservation are labeled independently:
energy drift is relative, while linear- and angular-momentum drift are absolute
with SI units. Cross-branch energy difference is explicitly not presented as
integration error.

Retained warnings are grouped once per immutable result by branch and issue
type; expandable entries preserve every bounded underlying warning, and the
dropped-warning count remains explicit. Terminal engine/worker failures remain
outside collapsed diagnostics. Overlay feedback names the active SSB/inertial
or shared baseline-relative frame and explains removed, missing, unretained, or
incomplete origins without fallback or independent recentering. The line legend
draws baseline solid and counterfactual dashed styles. Full-system label
decluttering now gives deterministic class-aware priority to selected and major
bodies while leaving focused subsystem behavior and physical visibility
unchanged.

Focused deterministic coverage includes stable-ID row mapping, numeric and
optional-value sorting, `N/A` and zero-time semantics, threshold/persistence
retention, warning grouping and dropped counts, immutable run metadata, frame
availability explanations, common-frame overlay alignment, visibility masks,
narrow/wide control geometry, label priority, and absence of live-controller or
committed-result mutation. Manual native-window appearance verification remains
required after automated validation.

Canonical Release validation passed on 2026-10-12: CPU CTest `12/12` and
OpenMP CTest `13/13`. Focused visualization-model and counterfactual-worker
tests passed `2/2` in each configuration. Both complete suites also passed the
repository-layout checker and pinned dependency-cache verifier. CUDA paired
execution and NVIDIA runtime validation were not performed or claimed.

### CUDA paired counterfactual execution increment

This uncommitted increment on `feature/cuda-visualizer` starts from
`65d429032dc3efaa6f6357a8a1d89d20a4902582`. `ExperimentBackend` now includes
an explicit CUDA choice and device index. A CUDA-enabled
`CounterfactualExperimentRunner` constructs two existing
`CudaVelocityVerlet` executors after all host admission checks: one sized to the
complete baseline and one to the one-body-compacted counterfactual. The
executors have independent streams, pinned staging, and persistent device
allocations; calls remain serialized. No CUDA kernel, force law, integration
equation, timestep, precision, sampling cadence, or coordinate-frame rule was
changed.

Each physical step starts from the transactional candidate host pair. Both CUDA
executors must finish and download finite states before either state is applied
to that candidate. Existing per-step time, endpoint, swept-proximity, sample,
threshold, and conservation logic then runs unchanged on host. Only a complete
validated batch replaces the committed result. If either CUDA call or any later
scientific check fails, RAII destroys both CUDA executors and the runner
becomes terminal: committed states, metrics, histories, warnings, dropped-
warning count, conservation values, and completed steps remain unchanged while
the existing bounded terminal-failure record remains visible. There is no CPU
retry or fallback.

CUDA teardown preserves every stream-referenced event, device allocation, and
pinned staging buffer until an explicit owning-stream synchronization succeeds.
After success, stream/events/buffers are released in safe order. A failed device
selection or synchronization is not treated as proof of quiescence: the stream
is retired where possible and the bounded per-executor resources are
quarantined until CUDA context/process teardown. No device-wide reset is used,
because the live viewer and other executors may share the primary context. A
genuine runtime/device failure therefore makes the runner terminal and does not
carry a fresh-executor recovery guarantee. Every non-quiescent executor can
retain another full device/pinned allocation set; repeated fatal errors may
therefore accumulate unreleased storage until teardown. The safe operational
response is to record diagnostics, stop further GPU experimentation, and restart
the application process.

The asynchronous worker remains the sole runner owner and performs no OpenGL
work. Immutable run descriptors now retain the selected CUDA device. The GUI
shows CUDA for experiments only when CUDA was compiled; it disables the choice
when the selected device is unavailable and reports explicit rejection rather
than relabeling CPU work. Scenario/worker generation gating, cancellation at a
completed paired boundary, stale suppression, overlay frames, and bounded
publication ownership are unchanged.

Focused coverage added CPU-only CUDA rejection, negative device gating,
immutable device metadata, worker failure without fallback, exact initial
survivor equality and compacted branch size, paired batch boundaries,
cancellation, repeated executor lifetimes, failed-batch rollback/resource
invalidation, a recoverable host-side failure injected after real CUDA work and
readback are enqueued, fresh-executor execution after that safe teardown,
stable-ID metric mapping, and mixed-tolerance Sequential/CUDA agreement. The
injection does not simulate or claim recovery from a runtime/device failure.
The device-dependent tests use strict throwing device enumeration. Skip return
code `77` is emitted only when enumeration succeeds with a zero device count;
runtime/driver enumeration errors fail, and a positive count requires genuine
CUDA execution. Skipped execution is not a pass. A host-only decision test
covers the zero/positive/exception paths without pretending to execute CUDA.

Local AMD-only validation on 2026-10-13 passed canonical CPU Release CTest
`12/12` and OpenMP Release CTest `13/13`, including Stage 5/tooling regressions,
catalog/dataset reconstruction, counterfactual engine, visualization model, and
worker tests. Layout and pinned dependency verification passed. `nvcc` was not
on PATH; canonical `cuda-release` configuration stopped with the intended
`NBODY_ENABLE_CUDA=ON, but no usable CUDA compiler was found` error. Therefore
CUDA compilation, linkage, the new skip behavior, device cleanup, numerical
agreement, runtime correctness, and performance are **UNVERIFIED** locally.

The RTX 4050 gate must configure the portable `cuda-release` preset with
`-DCMAKE_CUDA_ARCHITECTURES=89`, verify that exact cache value plus compatible
nvcc/MSVC/driver selection, record toolkit/driver/device identity, build every
CUDA target, run the full CTest matrix, and confirm no CUDA test returns skip
77. It must repeat start/cancel/failure cycles, compare
both final branches and retained metrics with Sequential under established
mixed tolerances, construct fresh executors after the recoverable injected
host-side post-enqueue failure, inspect GUI backend/device metadata, and run
the CUDA viewer backend concurrently with complete/cancel/restart cycles of a
CUDA counterfactual worker to verify shared-process coexistence without reset,
fallback, corruption, or deadlock. No GPU performance or scientific-accuracy
claim is authorized by this implementation. The
provisional catalog 60-second timestep, one-day operational ceiling, point-mass
and linear swept-screening limitations, and unsupported escape classification
remain unchanged. Frozen Stage 5 source/protocols/policy/evidence and the JPL
dataset are untouched. Sequential/OpenMP/CUDA Barnes--Hut remains mandatory and
unimplemented.

## Canonical repository layout and build custody

### Organization record

The ordinary development workflow now has one ignored build root with three
tracked presets: `build/cpu-release/`, `build/openmp-release/`, and optional
`build/cuda-release/`. Temporary approved configurations belong only below
`build/scratch/<purpose>/`. `AGENTS.md` makes this policy binding for automated
agents and developers; `scripts/check_repository_layout.py` provides a
read-only enforcement check, and ordinary CTest runs it when Python is
available. The checker rejects new root `build-*` trees, tracked compiler
binaries, root generator directories, and untracked generated validation
artifacts outside approved locations. It never deletes or moves data.

`scripts/test-local.ps1` runs the layout check, configures, builds, and tests a
requested preset. It prefers the verified ignored portable CMake distribution
under `.tools/` and otherwise uses `PATH`. Python temporary files are directed
to the selected build tree. `scripts/run-visualizer.ps1` defaults explicitly to
the Sequential backend in `build/cpu-release/`, supports explicit OpenMP and
CUDA selections, forwards application arguments as an array, and refuses to
substitute another backend when the requested executable is unavailable.
Optional compiler, Ninja, and offline FetchContent paths remain local through
the ignored `CMakeUserPresets.json` pattern demonstrated by
`CMakeUserPresets.json.example`; tracked presets contain no machine-specific
absolute paths.

The inventory found 54 ignored historical root `build-*` directories. They are
grandfathered by exact name in `tools/repository_layout_baseline.json` only
until a separately approved cleanup. No historical directory was deleted or
moved. Two directories contain potentially unique untracked evidence and must
be retained and investigated: `build-stage3-pilot` and
`build-stage6d-pilot-smoke`. `build-stage6c-visual` must also remain until
canonical dependency acquisition no longer relies on its pinned GLFW, GLM,
and ImGui source trees. The ignored `.tools/` tree remains the offline
toolchain fallback. The grandfathered root source archive
`solar-system-simulator.zip` is hash-bound in the baseline and requires a
historical-value decision before deletion. The exact sizes, references,
classifications, and safe cleanup sequence are recorded in
`docs/repository_layout.md`.

The development presets explicitly mark Stage 5 source/protocol identity as
uncontrolled/non-evidence. They do not replace or reinterpret any frozen
Stage 5 command, cleanroom, protocol, policy, script, benchmark, or retained
evidence. Scientific output must not be placed in the disposable canonical
build tree.

### Canonical validation

Fresh GCC 15.1.0/Ninja Release validation used only the canonical paths. Local
one-time compiler/Ninja and pinned FetchContent source overrides were supplied
at configure time because the machine's toolchain is not globally configured
and network acquisition was not required:

- CPU-only `build/cpu-release/`: configure and build passed; CTest passed
  `11/11`, including the layout checker, legacy/catalog/dataset coverage, the
  counterfactual engine, and asynchronous viewer worker.
- OpenMP `build/openmp-release/`: configure and build passed; CTest passed
  `12/12`, including the same coverage plus OpenMP-specific agreement.
- Both builds produced the documented `nbody_visualizer.exe`. Launcher dry
  inspection selected the CPU executable with `--backend sequential` and the
  OpenMP executable with `--backend openmp`; `--help` launches returned zero
  and exposed the `catalog` scenario and counterfactual-capable viewer options.
- Requesting CUDA without `build/cuda-release/nbody_visualizer.exe` produced an
  explicit error and no fallback. CUDA compilation/runtime and RTX 4050
  validation remain pending.
- The first canonical CPU test attempt exposed an unwritable host temporary
  directory for a Python Stage 5 tooling test. The general wrapper was corrected
  to use `build/<preset>/tmp`; the final complete CPU and OpenMP suites passed.
  No Stage 5 implementation or evidence changed.
- The native interactive GUI was not exercised in this non-interactive
  closeout. Manual usability validation remains pending via
  `.\scripts\run-visualizer.ps1` (or `-Backend OpenMP`).

This organization work was committed locally as `fce3f20` and was not pushed.
Historical cleanup, CUDA paired execution, RTX 4050 validation, and mandatory
Barnes--Hut work were not part of that commit.

### Historical-build dependency and cleanup audit

An independent audit then re-scanned every one of the 54 grandfathered root
`build-*` directories for tracked path references, unique CSV/JSON/log/text
outputs, CMake/test-only products, embedded dependency sources, and canonical
cache imports. No directory was deleted, moved, or archived, and the active
54-name baseline remains unchanged.

The audit classifies 48 directories `SAFE_TO_DELETE` after separate approval,
representing 1,362,737,959 bytes (1,299.61 MiB). Two evidence-only directories
are `ARCHIVE_FIRST`: `build-stage3-pilot` and
`build-stage6d-pilot-smoke`; exact file-level SHA-256 values and proposed
external archive/recovery instructions are recorded in
`tools/historical_build_cleanup_plan.json`. Four build trees remain
`RETAIN_PROVENANCE` because retained protocols/evidence refer to their exact
binary paths: `build-openmp`, `build-sequential`, `build-stage6d-mpi`, and
`build-stage6d-openmp`. No directory remains `RETAIN_DEPENDENCY` after the
dependency migration, and none remains `NEEDS_INVESTIGATION` within this
bounded audit.

Clean source checkouts for the exact pinned GLFW 3.4, GLM 1.0.1, and Dear ImGui
1.92.9b revisions were copied without altering the originals from
`build-stage6c-visual/_deps` to the persistent ignored
`.tools/dependencies/` cache. `tools/dependency_cache_manifest.json` records
repository URLs, requested tags, resolved commits, and Git tree identities;
`scripts/verify_dependency_cache.py` rejects missing, partial, dirty, or
mismatched caches. The local validation wrapper now uses this cache only after
verification, so canonical builds no longer import from a historical build
tree. Pinned FetchContent remains the fallback when the cache is entirely
absent; no dependency version was downloaded or changed during migration.

Post-migration canonical Release validation passed from the persistent cache:
CPU-only CTest `12/12` and OpenMP CTest `13/13`. Both include the repository
layout check, dependency-cache verification, Stage 5 tooling, legacy Solar,
JPL catalog/dataset/snapshot, counterfactual engine/worker, and visualization
model; OpenMP also includes backend agreement. Both canonical CMake caches now
name only `.tools/dependencies/` for GLFW, GLM, and ImGui and contain no
`build-stage6c-visual` reference. Launcher inspection still maps CPU to
`--backend sequential` and OpenMP to `--backend openmp` without substitution.

This follow-up audit and dependency migration were committed locally as
`205770b` and were not pushed. Frozen Stage 5, JPL data custody, force laws,
integrators, and CUDA kernels remained unchanged.

### Approved historical-build cleanup

On 2026-10-12, the separately approved cleanup consumed the exact 48-entry
`SAFE_TO_DELETE` list in `tools/historical_build_cleanup_plan.json`. Before any
removal, the gate verified branch `feature/cuda-visualizer`, exact HEAD
`205770b7a434c06d88200314b4565e5f327ce5b8`, clean Git state, all 54 expected
root historical build directories, literal immediate-child target paths,
absence of reparse points, exact audited sizes, absence of new scientific
evidence, canonical dependency independence, the pinned dependency cache, and
the recorded recursive manifests of all six protected directories.

All 48 approved directories were then removed individually by literal path.
Their exact names are the `SAFE_TO_DELETE` entries retained in the cleanup
plan; the operation reclaimed exactly 1,362,737,959 bytes (1,299.61 MiB).
`tools/repository_layout_baseline.json` now contains exactly six historical
root build names: `build-stage3-pilot`, `build-stage6d-pilot-smoke`,
`build-openmp`, `build-sequential`, `build-stage6d-mpi`, and
`build-stage6d-openmp`. The first two remain `ARCHIVE_FIRST`; the latter four
remain `RETAIN_PROVENANCE`. All six still reproduce their recorded ordinal-path
recursive SHA-256 manifests. No protected directory was deleted, moved,
archived, or modified.

Post-cleanup validation passed:

- repository-layout checker with six grandfathered directories;
- a controlled negative probe proving a new root `build-*` directory is
  rejected, followed by probe removal and a clean positive rerun;
- pinned dependency-cache verification for GLFW 3.4, GLM 1.0.1, and Dear ImGui
  1.92.9b;
- canonical CPU Release CTest 12/12 and OpenMP Release CTest 13/13;
- CPU launcher discovery selecting `build/cpu-release/nbody_visualizer.exe`
  with `--backend sequential`, and OpenMP discovery selecting
  `build/openmp-release/nbody_visualizer.exe` with `--backend openmp`;
- real CPU and OpenMP `--help` launches, both returning success.

Canonical `build/`, `.tools/`, `data/`, `third_party/`, external evidence and
cleanrooms, `solar-system-simulator.zip`, frozen Stage 5, and the JPL dataset
remain preserved. No physics, integrator, CUDA-kernel, benchmark, protocol, or
scientific-data source file changed in this cleanup milestone.

## Sequential Barnes--Hut Stage 1

### Implementation record

Stage 1 was developed on `feature/cuda-visualizer` from
`b3b83fb967910f5476baec9796cc4fe3c58dc98f`. The solver was committed as
`eb6dac75012f47d60a87417f6ccecf6df7c4576a` and its exploratory evidence as
`a035b3ed6b60f11ce3377358f47e6b6dc2ba9231`. It implements the required
Sequential Barnes--Hut reference without changing the direct Sequential,
OpenMP, CUDA, MPI, or Velocity Verlet implementations. `nbody_barnes_hut` is a
separate library implementing the existing `ForceBackend` acceleration
contract. At that Stage 1 checkpoint, the headless runner exposed backend and
solver as distinct choices; direct remained default, only
`sequential + barnes-hut` was supported, and unsupported combinations or
Barnes--Hut options supplied to direct force failed without fallback.

The deterministic 3D octree stores nodes in a contiguous vector and child
indices rather than pointers. Stable octant partitioning produces contiguous
body ranges and an inverse-rank map; this lets traversal prove whether a node
contains the target after permutation and prevents aggregate self force. Nodes
store compensated mass sums and online weighted monopole centers without
forming overflow-prone mass-position products. The standard strict
`side / distance < theta` criterion accepts only nodes that do not contain the
target; opened leaves evaluate non-self members directly under the unchanged
unsoftened Newtonian law.

Termination is bounded by preferred leaf occupancy 8, maximum depth 64,
minimum cell size `1e-6 m`, one million nodes, one billion traversal work
units, and 256 MiB of accounted index/output/node-arena storage. Checked
`size_t` arithmetic rejects size overflow. Exact coincidences reach a bounded
termination leaf and are rejected as singular; near coincidences remain direct.
Stable norms, bounded child-center shifts, compensated mass summation, and a
binary64 exponent-scaled component fallback handle representable extreme
coordinates/forces without assuming that MSVC `long double` has extra range.
Non-finite or genuinely nonrepresentable inputs/results fail explicitly.

### Numerical and performance status

The accuracy-first provisional default is `theta=0.25`; it is not a universal
optimum. Focused regression covers empty/single/two-body, asymmetric, symmetric,
clustered, planar, collinear, translated, scaled, near-coincident, exact-
coincident, near-maximum, very small/large, mass-ratio, nonrepresentable, deeply
unbalanced, deterministic-order, self-exclusion, monopole mass/center, and all
resource-boundary cases. Near-zero theta opens every non-self pair to a direct
leaf. Absolute, RMS, maximum-relative, and 95th-percentile force errors are
retained together so cancellation does not turn a near-zero reference force
into a misleading standalone maximum-relative claim.

The 48-body 3000-second synthetic trajectory characterization uses identical
Velocity Verlet settings at both 25-second and 12.5-second timesteps. At 25
seconds, observed direct/Barnes--Hut maximum position separation was exactly
zero at represented precision for `theta=0.25` and `0.21823208696368748 m` for
`theta=0.8`. Relative energy drift was `1.9898432656444739e-15` and
`3.4913597995655239e-8`; normalized angular-momentum drift was
`7.7249125581288963e-16` and `2.10959820353425e-6`. The 12.5-second results were
consistent (`0`/`0.21823184885520824 m`, `9.1838919952821884e-16`/
`3.4913596465006577e-8`, and `2.1029529340356154e-15`/
`2.1095982030977879e-6`). These are bounded synthetic regression observations,
not long-horizon accuracy or ephemeris claims.

The retained 108-row exploratory Release matrix covers uniform, clustered, and
planar distributions at `N=128,512,2048,4096`, theta `0.25,0.5,0.8`, one
warm-up, and three repetitions. It separately records direct-force wall time,
tree-build time, tree-force time, Barnes--Hut force-call wall time, complete
two-force Velocity Verlet step time, tree/work/memory counters, and four error
metrics. Raw CSV SHA-256 is
`226730b66c96a75822520a962199404c3bb27a1d5b76b9f3d2fd74890d621846`.
At 4096 bodies, median direct-force time divided by the full Barnes--Hut
force-call wall time (tree build plus traversal) was
`0.23x/0.31x/1.27x` for uniform/clustered/planar at theta `0.25`,
`0.76x/0.78x/2.84x` at theta `0.5`, and `2.15x/1.22x/5.67x` at theta `0.8`.
At theta `0.25`, corresponding 95th-percentile relative acceleration errors
were `6.07e-4`, `2.81e-4`, and `3.84e-3`. Distribution and theta sensitivity
preclude a universal crossover; nothing is extrapolated to 50K--100K.

Final local Release validation of this uncommitted candidate passed all
`16/16` canonical CPU CTests and all `17/17` canonical OpenMP CTests. The
OpenMP configuration exercises the unchanged OpenMP direct-force backend; it
does **not** implement or validate an OpenMP Barnes--Hut backend. The focused
astronomical catalog/dataset/snapshot integrity subset passed `3/3`, the
repository-layout and pinned-dependency-cache checks passed, and explicit
headless smoke runs completed for Sequential direct and Sequential
Barnes--Hut. Unsupported `openmp + barnes-hut` and Barnes--Hut-only options
supplied to the direct solver both failed with diagnostics as required.

This matrix is explicitly exploratory, single-machine, three-repetition,
uncontrolled-host evidence measured immediately before commit from the source
captured byte-identically by `eb6dac75012f47d60a87417f6ccecf6df7c4576a`.
It is not frozen Stage 5 evidence, does not alter any Stage 5 policy, and cannot
justify an automatic selector. Independent source-level adversarial review
passed on 2026-10-13 without a P1 blocker. OpenMP Barnes--Hut Stage 2 is now
implemented and source-reviewed as recorded below. CUDA Barnes--Hut,
counterfactual/viewer integration, controlled large-N evaluation, and RTX 4050
compilation/runtime/performance validation remain pending.

## OpenMP Barnes--Hut Stage 2

### Implementation record

Stage 2 was developed on `feature/cuda-visualizer` from
`1f276505e5766d718e2979f4998831189f6865c8`, passed its final independent
source-custody gate on 2026-10-13, and is committed as `80d83f3` (implementation,
tests, harness, and verification tooling) plus `c572be1` (exploratory evidence
and checkout custody). Successive reviews identified and remediated
non-representational equality tests, an uncharged failure replay, missing
nested-region rejection, size-only vector accounting, failed-attempt work
undercount, ineffective schedule variation, incomplete effective-build
provenance, incomplete diagnostics comparison, and Windows checkout
line-ending instability. The final gate independently verified all 216 rows,
all source hashes, and an actual Git-materialized `core.autocrlf=true` checkout.
It adds `OpenMpBarnesHutForceBackend` without
cloning the approximation algorithm: every force call uses the Stage 1
Sequential builder to construct one deterministic index-based octree, then a
static OpenMP target loop reads the immutable tree/state. Each worker has its
own compensated target accumulator, recursion stack, output elements, and
cache-line-padded counters. Tree construction remains intentionally
Sequential; there is no topology mutation, nested region, shared floating-
point reduction, or unsupported fallback.

Thread count is explicit and bounded to `1..256`; the selected runtime must
honor it, and calls from an active OpenMP region are rejected before solver
allocation. A size-based lower-bound preflight precedes allocation, then actual
capacities for output, all three index arrays, nodes, and padded per-thread
counters are charged before force traversal. A checked worst-case traversal
bound avoids atomic overhead when the configured work limit is provably
sufficient; the strict path uses a compare/exchange reservation that cannot
overshoot. Exhaustion stops new targets, performs no fresh-budget replay, and
produces a normalized deterministic budget error. C++ exceptions are caught
within the parallel loop. Failed calls publish neither partial forces nor
candidate statistics. Contention-free per-thread charged-work counters advance
immediately after every successful node/direct reservation, independently of
whether the later numerical calculation throws. Failed fast-path attempt
counts are therefore complete, strict-path counts remain the atomic reserved
total, successful work matches published node-plus-direct work, and empty
successful calls reset the attempt diagnostic.
Successful per-thread integer statistics are merged after the region and
identify the effective worker count. Backend instances remain single-owner
because last-success statistics are mutable. Traversal has no heap scratch;
the native per-worker recursion stack is bounded by maximum tree depth, but
native stack bytes, OpenMP runtime allocations, and allocator metadata are not
included in the reported working-memory estimate. High requested team counts
can oversubscribe processors and increase those excluded costs.

The headless contract now supports all four CPU combinations: Sequential or
OpenMP execution with direct or Barnes--Hut force. Direct remains default.
CPU-only builds retain the public OpenMP Barnes--Hut type but reject execution
explicitly. Barnes--Hut policy options have identical meanings for Sequential
and OpenMP. The viewer and counterfactual engine are intentionally unchanged.

### Numerical, regression, and performance status

Focused testing on GCC 15.1.0/libgomp required byte-representation-equal
Sequential/OpenMP
Barnes--Hut accelerations for theta `0.25`, `0.5`, and `0.8` over uniform,
clustered, planar, boundary, unequal-mass, near-coincident, and extreme-binary64
fixtures at effective thread counts 1, 2, 4, and 8. Repeated forces,
Velocity Verlet states, conservation diagnostics, tree structure, and work
counters were exact. Tests also cover empty/single-body self exclusion,
coincident and nonrepresentable-force failures after reservation,
maximum-depth termination, node/memory/traversal budget
failures, transactional statistics, hard work-cap instrumentation under
actual static/dynamic/guided schedules, actual-capacity memory boundaries, nested-region
rejection with nesting disabled/enabled, invalid thread counts, direct CLI
preservation, and CPU-only rejection. A signed-zero negative control proves
the IEEE-754 comparator differs from ordinary `==`; all five stored diagnostics
fields plus derived total energy are compared fieldwise by representation.
Production remains `schedule(static)`. The schedule matrix uses a separately
named test-only build of the same source with `schedule(runtime)` and a positive
link-time identity hook, so no runtime ICV is mistaken for production schedule
variation. Timing fields are not correctness assertions.

The retained 216-row exploratory Release matrix is under
`data/benchmarks/barnes_hut_openmp_stage2/`; raw CSV SHA-256 is
`11fb34ae4d3b4f87687cf177f1668139d7f5bad57ec944f78b316048f2f96760`.
The exact measured sources are bound by
`source_manifest.sha256` (SHA-256
`853de3784315ded6c90a046144d882ae5806ad43df553fdb04683276f295b255`).
That manifest also binds the raw CSV and deterministic effective-build record,
`build_provenance.json` (SHA-256
`3b4f33cc83ff6f25b2954d1722d259c5e2fa0c3211c165e8eaf8c9ee19cab2c2`).
The latter records GNU 15.1.0's executable/version, CMake 4.4.3, Ninja 1.13.2,
Release/OpenMP flags and cache entries, OpenMP runtime libraries, selected
actual compiler commands, and the benchmark linker command. These captured
inputs identify the local build but do not guarantee repeatable machine-level
timing.
It covers uniform/clustered/planar distributions, `N=256,1024,4096`, theta
`0.25,0.5,0.8`, 1/2/4/8 effective threads, one warm-up, and two retained
repetitions. At 4096 bodies, complete-force speedup versus Sequential
Barnes--Hut ranged from `3.25x` to `3.98x` at four threads and `4.28x` to
`7.51x` at eight. For the accuracy-first theta `0.25`, eight-thread speedups
were `7.02x` uniform, `7.51x` clustered, and `7.33x` planar. Versus
eight-thread direct force at theta `0.25`, Barnes--Hut was `0.26x` uniform,
`0.35x` clustered, and `1.24x` planar. The sequential tree-build time is
included in every complete-force wall time.

These two-repetition uncontrolled-host observations are exploratory, not a
linear-scaling, universal-crossover, graphical-FPS, or 50K--100K claim. They
do not alter Stage 1 error evidence or frozen Stage 5 methodology/evidence.
Final canonical Release validation passed `16/16` CPU CTests and `20/20`
OpenMP CTests. This includes focused Barnes--Hut, the genuine test-only
runtime-schedule matrix, both headless solver paths,
legacy Solar, 60-body catalog/dataset/snapshot, direct-force, Stage 5 tooling,
visualization/counterfactual, layout, and dependency-cache checks.
CUDA Barnes--Hut, GUI/counterfactual solver integration, controlled large-N
evaluation, and RTX 4050 validation remain pending. CUDA compilation/runtime
was not attempted or claimed in this AMD-only Stage 2 increment.

## CUDA Barnes--Hut Stage 3A

### Source architecture and scope

Committed implementation `5e1864e` begins from
`94967e03632f098f81575599dc2eaf3b54889c4c`. It minimally exposes the existing
deterministic CPU builder as `build_barnes_hut_packed_tree()`, producing
fixed-width, standard-layout nodes with cell bounds, monopole data, eight child
indices, deterministic leaf ranges, original body-index permutation, and
inverse rank. The packer has a separate checked budget and preserves the Stage
1/2 construction algorithm rather than cloning it. CPU tests cover layout,
ordering, relationships, identity mapping, finite aggregates, determinism, and
budget/input failures.

`CudaBarnesHutForceBackend` is a separate force-only executor. It is not wired
to Velocity Verlet, headless selection, the viewer, counterfactual execution,
or automatic backend choice. The genuine CUDA kernel assigns one target per
logical work item, uses double precision and unchanged zero-softening Newtonian
gravity, applies the same strict monopole opening rule, always opens a node
containing the target, excludes self indices in leaves, and pushes children in
reverse order to retain CPU depth-first accumulation order. A compiled
1024-entry per-work-item stack and a configurable logical limit make overflow
explicit. A 64-bit compare/exchange counter enforces the global traversal-work
ceiling without overshoot. Device failures distinguish malformed trees,
singular/nonrepresentable geometry, nonrepresentable acceleration, stack
exhaustion, and work exhaustion; no CPU/direct fallback or partial output is
published.

The first independent adversarial source review identified two P1 failure-path
defects: ordinary device reads raced with atomic failure publication, and live
capacity replacement ignored CUDA deallocation errors. The committed
implementation uses an atomic claim plus fenced atomic publication record. All
concurrent device readers inspect only the atomic publication word, and the
host accepts metadata only after synchronization and completeness checks. Live
resize checks every device/pinned deallocation; failure preserves remaining
handles and accounting, terminalizes the executor, and prevents replacement
allocation. Partial allocation failure is likewise terminal and checked, while
destruction remains noexcept and quarantine remains mandatory when quiescence
cannot be established. Independent adversarial source review passed; NVIDIA
compilation and execution remain pending.

Persistent device arrays and pinned host staging arrays are reusable and
separately budgeted with checked arithmetic. Host accounting includes the
temporary packed snapshot. Persistent-device accounting excludes CUDA
allocator/context metadata and compiler-managed backing for the bounded local
stack, which is reported separately as 4096 bytes per work item. One owned
stream orders all transfers, launch, and readback. Device-selection or
synchronization failure makes the executor terminal; if quiescence cannot be
proved, all referenced storage is quarantined until process/context teardown.
No `cudaDeviceReset()` is issued because the primary context may be shared.

CUDA-only tests are registered for single/two-body, uniform, clustered,
planar, unequal-mass, boundary, coincident/near-coincident, extreme binary64,
theta/depth, node/host/device-memory, stack/work, CPU-reference, and repeated
lifetime cases. CPU packed-tree coverage now explicitly includes two-body,
coincident, and near-degenerate snapshots with deterministic mappings,
contiguous child ranges, and per-node aggregate checks. CUDA tests repeat
contended singular/work failures, validate permissible target/node metadata,
prove caller output is not partially published, and inject a host-side live
release failure to verify terminal ownership and preserved last-success
statistics. Strict enumeration permits skip code 77 only after successful
zero-device enumeration. CUDA compilation, kernel execution, numerical
agreement, cleanup behavior on NVIDIA, and GPU performance remain unverified
on this AMD-only/no-nvcc host. The CPU-built tree and per-force host-to-device
transfer may dominate at large N; no 50K/100K throughput or FPS claim is made.
Canonical Release validation passed `17/17` CPU CTests and `21/21` OpenMP
CTests, including the new packed-tree/API contract test and all existing
Barnes--Hut, legacy Solar, astronomical data, Stage 5, visualization, and
counterfactual regressions. An isolated CUDA-enabled configure under
`build/scratch` failed at compiler discovery with the expected "no usable CUDA
compiler" diagnostic and was removed. No CUDA target compiled and no GPU test
ran or skipped. This is a source-reviewed, GPU-unverified milestone.

## CUDA Barnes--Hut Stage 3B

Implementation `2cc2cb0` starts from
`8bacb069af70e1f68a850a38cec3d28ef9acbf9c` and adds
`CudaBarnesHutVelocityVerlet`. It preserves the existing Velocity Verlet
mathematics but applies it transactionally to a value-owned candidate: Stage 3A
evaluates old-state acceleration, candidate positions drift, a second Stage 3A
call rebuilds the deterministic tree at the new positions and evaluates new
acceleration, and the average-acceleration kick completes before a noexcept
move publishes the state. Names, ordering, masses, units, timestep, force law,
theta rule, and zero-softening semantics remain unchanged. No viewer or
counterfactual connection and no fallback were added.

Invalid input fails before force work. Failures retain caller state and the
previous successful integration statistics while recording whether validation,
first force, drift, second force, integration post-force validation,
kick, or commit was active. Configured or scientific Stage 3A errors remain
reusable when the evaluator is nonterminal; runtime, synchronization, or
checked-release terminal state prevents reuse. The Stage 3A quiescence, checked
live-release, conservative quarantine, and no `cudaDeviceReset()` policies
remain authoritative.

The Stage 3A traversal ceiling remains strict per evaluation. The Stage 3B
per-step ceiling is its checked two-evaluation sum, and successful statistics
retain both counters and the total. A separate host-integration budget accounts
for candidate body/name capacity and both acceleration arrays, while Stage 3A
continues to admit device, pinned host, and packed-tree staging. Runtime/context
metadata and native/compiler-managed stacks remain explicit exclusions.

The headless runner now parses all six Sequential/OpenMP/CUDA crossed with
direct/Barnes--Hut combinations. Availability is explicit: CPU-only builds
reject both CUDA solvers, OpenMP remains build-gated, and CUDA builds link the
existing direct integrator or new Barnes--Hut integrator according to the
selected solver. CUDA-dependent tests specify zero/one/two-body, legacy Solar,
JPL subset, distribution/theta/multistep, conservation/reference, invalid input,
first/second-force rollback, naturally reached drift/kick rollback,
nonrepresentable force geometry, explicitly classified post-force resource
checks, stack/work/memory, terminal reuse, and lifetime coverage. Direct-force
characterization compares independent Sequential-Barnes--Hut/direct and CUDA-
Barnes--Hut/direct trajectory metrics under fixed mixed tolerances. Post-force
injections exist only in a separately compiled test library; production has no
integration test hook. Local Release validation passed `18/18` CPU and `22/22`
OpenMP CTests; host syntax, layout, dependency, protected-scope, and whitespace
checks passed. No CUDA translation unit compiled and no GPU test ran or skipped
on this AMD-only/no-nvcc host. Independent adversarial source review passed;
Stage 3B remains GPU-unverified.

## Barnes--Hut Stage 4A viewer integration

Implementation `c9e95007f289ab72ab71d1f715a485da42a4a807` on
`feature/cuda-visualizer` starts from
`6bbd6e7c83529039b201b8af54f4d70743939e12`. The visualization controller now
models backend and force solver independently. Direct remains the default;
Sequential/direct, Sequential/Barnes--Hut, OpenMP/direct, OpenMP/Barnes--Hut,
and the existing CUDA/direct path are explicit. CUDA/Barnes--Hut is rejected
with a pending-NVIDIA-validation diagnostic and never falls back.

The viewer links the reviewed CPU Barnes--Hut library and reuses the unchanged
Velocity Verlet implementation. CPU frame/manual batches advance a value-owned
candidate and publish only after every requested step succeeds. A failure
pauses the controller, retains the last committed state/time, and exposes the
diagnostic. Solver/backend changes construct replacement resources before
publishing effective configuration and preserve current state, time, pause
state, body order, timestep, scenario data, conservation tracking, and trails.

The ImGui panel provides Direct/Barnes--Hut selection, an explicitly applied
theta draft (default/effective `0.25`), effective backend/solver metadata,
bounded tree/work/memory policy, last-success statistics, and force failure
text. The existing large-synthetic workload gates remain active in addition to
the solver's hard resource limits. No graphical crossover, FPS, or interactive
50K--100K physics claim is made.

The live simulation remains synchronous on the UI thread. The separate
counterfactual worker and engine remain direct-force-only; immutable snapshots,
scenario/worker generations, cancellation, and stale-publication suppression
remain authoritative. Reset now requests cancellation, invalidates the accepted
job before restoring state, and clears publication-derived progress, results,
warnings, frame errors, labels, and overlays. Saved progress and delayed
terminal completion are rejected while a later fresh job remains admissible.
Solver and theta edits cannot retarget an in-flight experiment.
CUDA kernels, CUDA Barnes--Hut Stage 3A/3B, direct-force equations, Velocity
Verlet mathematics, JPL data, legacy Solar initialization, benchmark evidence,
and frozen Stage 5 artifacts were not modified.

Canonical Release validation on the AMD Windows host passed CPU `18/18` and
OpenMP `22/22` CTests. Focused controller coverage checks all four CPU solver
combinations, direct default behavior, exact matching to the corresponding
headless backend over seven steps, theta rejection/application,
state/time-preserving switches, scenario/reset preservation, effective
statistics, explicit CUDA/Barnes--Hut rejection, and rollback after a
second-force singularity with backend, solver, step, and `new-force` context.
Worker coverage exercises real scheduled progress, the production invalidation
helper, late terminal rejection, complete cache clearing, and a fresh accepted
job. Sequential and two-thread OpenMP Barnes--Hut headless seven-step smokes
produced byte-identical CSV data rows. A real 1920x1080 catalog/OpenMP/
Barnes--Hut OpenGL launch remained responsive for the ten-second smoke window;
this establishes launch/event-loop behavior, not subjective visual approval.
The final independent source-level commit gate passed on 2026-10-14. NVIDIA
CUDA/Barnes--Hut viewer validation remains pending; no GPU result is claimed.

## Known limitations

- A sourced common-epoch 60-body JPL initial-condition snapshot now exists,
  while the default Stage 6B nine-body baseline remains the bounded mean-
  element approximation. Neither scenario reproduces JPL's complete dynamical
  model or establishes long-horizon ephemeris agreement.
- No close-encounter regularization, softening, relativistic effects, or adaptive timesteps.
- Sequential and OpenMP Barnes--Hut use the same monopoles and target-local
  acceptance, so neither preserves exact pairwise force symmetry. Theta `0.25`
  is provisional and distribution-dependent; Stage 3B has only uncompiled CUDA
  force/integration source using CPU-built trees, with no authorized CUDA viewer
  path or validated 50K--100K interactive force workload.
- CUDA physics is T4-validated and the viewer has an opt-in batched CUDA path,
  but viewer CUDA compilation/runtime and a same-machine CPU/OpenMP/CUDA matrix
  remain unverified until an NVIDIA/CUDA host is available. MPI is implemented
  and locally validated, but no controlled multi-machine evidence exists.
- Stage 6D native-window appearance and interaction need final user visual validation; approximate O(N^2) algorithms, distributed state, one-rank-per-GPU execution, instanced high-detail meshes, high-resolution textures, advanced atmosphere/post-processing, RTX presentation capture, and real ephemerides are deferred.
- Stage 5 v2 tooling is locally CPU/OpenMP-validated, but native VS2022/MSVC, CUDA 12.8.x-class `sm_89`, RTX 4050, and driver integration remain unexecuted until the NVIDIA pilot rehearsal.
- The pacman CMake executable is unusable in the partially updated MSYS2 environment; the verified portable CMake is project-local and ignored.
- Stage 3 CPU affinity could not be enforced by this Windows/libgomp configuration; thermal and continuous clock telemetry were unavailable.
- Eleven samples per configuration characterize this run but remain too few for strong distributional or cross-machine claims.

## Next actions

1. Transfer and verify the external Stage 5 v2 deployment packet, then check out its exact frozen SHA and follow `docs/stage5_rtx4050_runbook.md`.
2. Run fresh preparation, preflight, and the pilot; independently verify and record manifest-bound human approval, then stop the first NVIDIA rehearsal.
3. Only in a separately authorized final session, acquire calibration, freeze the deterministic policy, acquire held-out evidence, and evaluate without refitting.
4. Run the final Stage 6D native-window visual checklist in `docs/visualization.md`; the feature freeze closes after this gate.
5. Do not tune the frozen policy, optimize backends, or pool historical cross-machine evidence during Stage 5B collection.
6. Compile and execute the strict source-reviewed Stage 3A/3B numerical and
   resource suites on NVIDIA hardware before authorizing higher-layer
   integration.

## Commit references

- `2d16155` — research, novelty assessment, scope, architecture, validation plan, and benchmark protocol.
- `c241ca7` — deterministic sequential implementation, tests, headless runner, and project scaffolding.
- `7f6e2bc` — Stage 1 validation evidence and exact foundation commit references.
- `1485d54` — Stage 2 OpenMP backend, validation suite, benchmark harness, and documentation.
- `a4c9eb1` — frozen Stage 3 controlled benchmark harness and protocol used for measurement.
- `94dbf19` — Stage 3 raw measurements, statistical summaries, environment evidence, and analysis.
- `bc03018` — Stage 3 commit provenance and synchronized completion point.
- `ae6117b` — Stage 4 optional CUDA backend and exact source snapshot used for real-GPU validation.
- `ca1df10` — pre-execution workflow bookkeeping and pilot-review gate.
- `1353b55` — verified Stage 4 pilot/full evidence, byte-preservation rules, and independent audit tooling.
- `3481272` — permanent Colab line-ending, dependency-preflight, and dual-mode provenance repairs.
- `194bf04` — Stage 4 completion analysis, scientific boundary, and research-ledger update.
- `6d9a162` — standalone sequential conservation-study harness and fixed 30-configuration protocol.
- `4916b3c` — Stage 5A frozen protocol, runner, verifier, environment capture, and RTX 4050 runbook; exact RTX source revision `4916b3c43298fa3f3bd15cb4af1698b5074abed3`.
- `cfac22b` — optional Stage 6A OpenGL application, reusable render model, interaction controller, and headless visualization tests.
- `7be7104` — Stage 6B approximate Solar System scenario, ImGui application, trails, selection/focus, conservation diagnostics, and validation tests.
- `50a0473` — Stage 6C final presentation system: themed UI, deterministic camera framing, orbit guides, label decluttering, procedural bodies/rings/starfield, and headless presentation tests.
- `96c36a6` — Stage 6C.1 render-clearance, stable-label, real Sun-motion, scalable Compute Experiment, live CPU comparison, and regression-test implementation.
- `b432abe` — Stage 6D MPI/hybrid backend, large-N safety/protocol infrastructure, camera-aware Solar sizing, stable inner callouts, and final UI implementation.
- `45be2d3` — opt-in batched CUDA visualizer integration, transactional
  switching/resource management, explicit workload policy, and focused CPU/GPU
  test coverage.
- `e7632e5` — asynchronous counterfactual experiment worker, immutable
  publication/provenance model, generation gating, common-frame overlays, and
  focused GUI/worker tests.
- `dade679` — counterfactual GUI architecture, validation boundary, scientific
  limitations, and ledger documentation.
- `eb6dac7` — deterministic bounded Sequential Barnes--Hut solver, headless
  selection contract, and focused regression suite.
- `a035b3e` — exploratory Barnes--Hut benchmark harness plus hash-custodied
  accuracy/performance evidence.
- `80d83f3` — OpenMP Barnes--Hut backend, bounded validation suite, exploratory
  benchmark harness, and provenance verification tooling.
- `c572be1` — 216-row exploratory OpenMP Barnes--Hut scaling evidence plus
  source-manifest and Windows checkout line-ending custody.
- `2cc2cb0` — source-reviewed transactional CUDA Barnes--Hut Velocity Verlet,
  headless selection, failure-phase reporting, and focused CPU/GPU tests.
- `c9e9500` — independently source-reviewed Stage 4A Sequential/OpenMP
  Barnes--Hut viewer selection, transactional advancement, and lifecycle tests.
