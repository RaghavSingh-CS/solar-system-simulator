# Master ledger

Last updated: 2026-09-26

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
| Stage 5A same-machine preparation | **PREPARED** | Frozen v1.0 protocol, runner, verifier, environment capture, and RTX 4050 runbook |
| Stage 5B RTX 4050 execution | Not started | Requires pilot, inspection/approval, then controlled full matrix |
| MPI backend | Optional/not started | Stretch goal only |
| Stage 6A OpenGL application foundation | **IMPLEMENTED / USER VISUAL VALIDATION PENDING** | Optional viewer builds/tests; Sequential and OpenMP real-context smoke launches passed |

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
- Stage 4 real-GPU evidence is environment-scoped to a Colab Tesla T4; the local Windows machine still cannot compile or execute CUDA.
- Colab CUDA results remain a separate environment and cannot be combined with Stage 3 Windows CPU timings to infer a workstation crossover.
- Stage 5 selector calibration uses only same-machine `selector_steady_state` measurements from the frozen RTX 4050 full run.
- Stage 5 calibration and held-out N values are fixed before data collection; held-out results cannot influence policy boundaries.
- Stage 6A remains optional and one-way dependent on the simulation core; display timing is never benchmark evidence and no Auto backend selector exists.

## Assumptions and unresolved questions

- Intended RTX 4050 exact model, driver, power limit, CPU, and memory configuration are unknown.
- Faculty acceptance criteria for “realistic” horizon and tolerated position error need confirmation before final study.
- Stage 4 fixed CUDA at double precision; all predeclared comparison/conservation thresholds passed on the Tesla T4 without revision.
- OpenMP is validated with GCC 15.1.0/libgomp; availability and numerical behavior under the final MSVC toolchain remain untested.
- The exact RTX 4050 host CPU topology is unknown; Stage 5 freezes a topology-based thread-selection rule and requires one explicit list to remain unchanged from pilot through full execution.

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

## Known limitations

- No verified planetary initial-condition dataset yet.
- No close-encounter regularization, softening, relativistic effects, or adaptive timesteps.
- CUDA is T4-validated, but no same-machine CPU/OpenMP/CUDA matrix exists; MPI remains a placeholder and viewer CUDA integration is deferred.
- Stage 6A native-window appearance and interaction need user visual validation; large-N instancing/points, trails, selection, polished UI, and real ephemerides are deferred.
- Stage 5A tooling is locally CPU/OpenMP-validated, but its Windows MSVC/CUDA integration remains unexecuted until the RTX 4050 pilot.
- The pacman CMake executable is unusable in the partially updated MSYS2 environment; the verified portable CMake is project-local and ignored.
- Stage 3 CPU affinity could not be enforced by this Windows/libgomp configuration; thermal and continuous clock telemetry were unavailable.
- Eleven samples per configuration characterize this run but remain too few for strong distributional or cross-machine claims.

## Next actions

1. On the RTX 4050 host, check out the exact frozen Stage 5A preparation revision and follow `docs/stage5_rtx4050_runbook.md`.
2. Run preflight, validation, and the pilot; stop for independent verification and human inspection.
3. Only after explicit pilot approval, collect and verify the frozen full calibration/held-out matrix.
4. Run the concise Stage 6A native-window visual checklist; keep any follow-up presentation polish separate from the frozen Stage 5B collection.
5. Do not derive the selector or optimize backends during Stage 5B collection.

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
