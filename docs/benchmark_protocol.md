# Benchmark protocol

## Principle

Crossover is a property of a particular workload, implementation, build, and machine. Only backends timed on the same machine in the same experimental block may be compared.

## Workload families

- **Physical:** verified Solar System snapshot, small N, used for correctness and visualization—not scalability claims.
- **Synthetic:** deterministic seeded distributions over increasing N, used for scalability. These are never described as a Solar System.

## Measurement

- Release build; record compiler and flags.
- One untimed warm-up per configuration, then at least 10 randomized-order repetitions (increase until confidence intervals stabilize).
- Primary backend timing uses steady-clock wall time around integration only. Initialization/state copying, diagnostics, CSV, and rendering are excluded and must be reported separately. A later CUDA study must additionally report end-to-end timing including required transfers.
- Record median, interquartile range, arithmetic mean, standard deviation, and bootstrap 95% confidence interval.
- Pin OpenMP thread counts in the matrix and record affinity/environment.
- Synchronize CUDA before stopping the timer; retain transfer costs unless a separately labeled kernel-only experiment is reported.
- Check thermal/power state where possible and preserve raw observations.

## Matrix

Vary at least N, steps, dt, backend, thread count, and precision. Use a pilot grid to identify transition regions, then sample densely around them. Use identical input states and output cadence across backends.

## Selector evaluation

Split workload configurations into calibration and held-out sets before fitting. Baselines: sequential-only, OpenMP-only, CUDA-only, size-only threshold, and offline fastest-valid oracle. Report validity rate, backend-choice accuracy, selector overhead, regret distribution, and failure cases. Never train and evaluate on the same timing records.

## Accuracy gate

Validity is evaluated before performance ranking. A fast run that violates a declared bound is not a candidate. Thresholds, horizon, and diagnostic formulae are frozen before final measurements.

## Stage 2 harness

`nbody_benchmark` accepts backend, body count, steps, timestep, thread count, warm-up count, repetitions, and seed. Scenario construction and state copying occur before the timed interval. The interval contains only Velocity Verlet steps, including both force evaluations. Diagnostics and CSV formatting occur afterward. The final state feeds diagnostics and a printed checksum so the workload remains observable.

Each output row records median, IQR, minimum and maximum wall time; physics diagnostics; checksum; compiler/build configuration; hardware thread count; host; and processor identifier. The same seed recreates identical initial state across separate backend invocations.

### Preliminary exploratory measurements

Environment: GCC 15.1.0, Release, Windows/MSYS2 UCRT64, 16 reported hardware threads, processor identifier `AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD`. Each row used one warm-up and five repetitions with `dt=10 s` and seed `24301`.

| Bodies | Steps | Sequential median / IQR (ms) | OMP 1 median / IQR (ms) | OMP 2 median / IQR (ms) | OMP 4 median / IQR (ms) | Seq / OMP 4 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 200 | 10.728 / 0.146 | 9.294 / 1.173 | 19.355 / 1.952 | 22.725 / 3.104 | 0.47x |
| 256 | 100 | 82.281 / 6.359 | 72.469 / 11.427 | 52.398 / 3.915 | 33.797 / 5.234 | 2.43x |
| 512 | 50 | 152.591 / 14.290 | 129.461 / 4.452 | 84.643 / 2.383 | 52.262 / 3.214 | 2.92x |

The 64-body case shows parallel-region overhead dominating useful work. These are preliminary engineering measurements: only five repetitions, fixed execution order, no affinity/thermal control, and a sandboxed host. They are not publication-quality data and do not establish a durable crossover threshold.

## Stage 3 controlled CPU protocol

The Stage 3 runner is `scripts/run_cpu_benchmarks.py`. Its frozen default matrix is:

- body counts: 64, 96, 128, 160, 192, 256, 384, 512, 768, and 1024;
- backends: sequential and OpenMP with 1, 2, 4, 8, and 16 requested threads, filtered against detected logical concurrency;
- 100 Velocity Verlet steps per sample with `dt=10 s` and synthetic-state seed `24301`;
- one untimed warm-up immediately before each measured sample, in the same process;
- 11 measured samples per configuration.

The body-count grid retains the Stage 2 cases, samples the preliminary transition region densely from 96 through 256 bodies, and extends through 1024 bodies to expose larger-workload scaling without making the run unnecessarily long. Eleven samples provide an observed median while remaining explicit about limited statistical confidence.

Each repetition block contains every configuration exactly once. A specified xorshift64* generator and Fisher-Yates shuffle randomize each block using order seed `20260923`; the raw file records block, repetition, and global execution order. This distributes configurations over the experiment rather than always timing sequential first.

The runner writes:

- `raw_timings.csv`: every measured repetition and its order, seeds, workload, diagnostics, checksum, effective thread count, build/toolchain, and OpenMP environment;
- `summary.csv`: median, mean, sample standard deviation, min/max, IQR, coefficient of variation, deterministic bootstrap 95% interval for the median, speedup, and parallel efficiency;
- `environment.json`: complete matrix, timer scope, exclusions, host metadata, Git state, and telemetry availability;
- `affinity_probe.txt`: the unedited libgomp environment/affinity probe.

The timer remains inside `nbody_benchmark` and covers only Velocity Verlet integration, including both force evaluations per step. Process startup, seeded scenario generation, initial-state copying, warm-up integration, diagnostics, checksum calculation, CSV formatting, and runner I/O are outside the timed interval. The final state remains observable through diagnostics and a checksum; the runner rejects checksum divergence or a runtime team smaller than requested.

On the validated Windows/libgomp host, `OMP_DYNAMIC=FALSE` is set for measurement. An explicit `OMP_PROC_BIND=CLOSE`, `OMP_PLACES=CORES`, and affinity-display probe reported that affinity is unsupported and showed all workers eligible on logical processors 0–15. Measurements therefore set `OMP_PROC_BIND=FALSE`, leave `OMP_PLACES` and `GOMP_CPU_AFFINITY` unset, and do **not** claim affinity control. The implementation itself uses an explicit `schedule(static)` clause.

Exact default reproduction command from the repository root:

```powershell
python -B scripts/run_cpu_benchmarks.py `
  --executable build-openmp/nbody_benchmark.exe `
  --output-dir data/benchmarks/stage3_cpu `
  --bodies 64,96,128,160,192,256,384,512,768,1024 `
  --threads 1,2,4,8,16 `
  --steps 100 --dt 10 --warmups 1 --repetitions 11 `
  --benchmark-seed 24301 --order-seed 20260923 `
  --bootstrap-resamples 10000 --physical-cores 8 `
  --experiment-id stage3-cpu-2026-09-23
```
