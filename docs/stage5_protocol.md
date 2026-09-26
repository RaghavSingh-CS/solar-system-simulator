# Stage 5 same-machine protocol (version 1.0)

## Status and research boundary

This protocol is frozen during Stage 5A, before RTX 4050 measurements exist.
Stage 5A prepares tooling only. Stage 5B will collect same-machine evidence on
one Windows workstation and must not begin until its pilot has been inspected.

The research question is whether a calibrated policy can select the fastest
valid sequential, OpenMP, or CUDA configuration on one workstation. Stage 5B
collects the calibration and held-out evidence; it does not itself fit or
implement the selector. Stage 3 Windows CPU and Stage 4 Colab T4 timings are
not inputs to Stage 5 comparisons.

Protocol identifier: `stage5-v1.0`.

## Frozen workload and split

Every measured configuration uses the deterministic synthetic scenario with
seed `24301`, binary64 state and arithmetic, the direct Newtonian O(N^2) force
law, fixed-step Velocity Verlet, `dt=10 s`, and no output sampling during the
timed interval.

The full run uses 100 integration steps and 11 measured repetition blocks.
The full N grid and its predeclared selector role are:

| Role | N values |
|---|---|
| Calibration | 64, 128, 192, 256, 512, 1024, 2048, 4096 |
| Held-out/interstitial evaluation | 96, 160, 384, 768, 1536, 3072 |

The held-out values must not be used to derive policy boundaries. They are
revealed only when evaluating a policy fixed from calibration data. The split
interleaves sizes and retains the established Stage 3/4 grid. A future policy
may be a simple piecewise threshold over N and the available OpenMP team; no
machine-learning method is predeclared.

The pilot uses N=`64,512,4096`, 10 steps, and three measured blocks. It checks
the smallest, middle, and largest intended workloads, estimates runtime, and
may expose a need for a separately reviewed protocol revision. Pilot rows are
feasibility evidence only and cannot be pooled into calibration or held-out
data. Any refinement after the pilot requires an explicit protocol revision
and a new frozen source commit before the full run; the full v1.0 grid is never
silently changed.

## Backends and OpenMP thread selection

Each complete block contains:

- the sequential backend;
- OpenMP-1 as a distinct configuration;
- every other explicitly supplied OpenMP thread count; and
- CUDA device 0, after device identity is verified as containing `RTX 4050`.

The RTX host topology is unknown during Stage 5A, so v1.0 freezes a selection
rule rather than inventing a thread list. After preflight records total
physical cores P and total logical processors L across all CPU packages,
choose the sorted unique set containing 1, powers of two below P, P, and L;
omit any value above L. Record the explicit
list in the pilot approval and reuse it unchanged for the full run. The runner
rejects a requested count above L and any OpenMP team whose effective size
differs from the request. Sequential and OpenMP-1 remain separate even if one
is consistently faster.

The measurement environment sets `OMP_DYNAMIC=FALSE` and
`OMP_PROC_BIND=FALSE`; `OMP_PLACES` and `GOMP_CPU_AFFINITY` are unset. This
uses Windows scheduler placement and does not claim affinity control. The
environment capture preserves the values actually used.

## Warm-up and randomized order

Each measured invocation receives one untimed, same-workload warm-up in the
same process immediately before its measurement. This initializes the
sequential code path, OpenMP runtime/team, or CUDA context and reusable
executor as applicable. Process startup and warm-up are not timed.

Each repetition is a complete block containing every backend/thread/N
configuration exactly once. A specified xorshift64* generator and Fisher-Yates
shuffle randomize each block. The full default order seed is `20260926`; raw
rows retain block and global order. The independent verifier reconstructs the
entire schedule.

## Timing metrics and fairness

The runner preserves existing metrics and normalizes them to milliseconds in
tidy long-form rows:

| Metric | Backend | Exact boundary | Use |
|---|---|---|---|
| `integration_compute` | Sequential/OpenMP | Steady-clock interval containing 100 Velocity Verlet steps, including both force evaluations; initial state copy excluded | Retains Stage 3 semantics; CPU implementation analysis |
| `device_compute` | CUDA | CUDA events after H2D completion through invalid-flag reset and all four kernels per step | Retains Stage 4 kernel-path semantics |
| `transfer_inclusive` | CUDA | CUDA events from H2D start through kernels, D2H positions/velocities/status, and completion | Retains Stage 4 transfer metric |
| `selector_steady_state` | Sequential/OpenMP | Steady-clock interval from initial-state copy through completed integration | Primary comparable selector cost |
| `selector_steady_state` | CUDA | Existing `backend_wall_ms`: input packing, H2D, device compute, D2H, synchronization, output materialization, and validation | Primary comparable selector cost |

The primary selector metric deliberately represents one warmed invocation
that returns a usable host `SimulationState`. CUDA context, stream, events,
device buffers, and pinned buffers are reusable setup and are excluded from
that primary metric; `allocation_ms` is retained in raw run metadata and the
environment documentation. This is a steady-state end-to-end backend metric,
not cold-start application time. Scenario generation, process startup,
warm-up, diagnostics, checksums, CSV formatting, runner I/O, and statistical
summaries are excluded for every backend.

No scientific implementation, force ordering, precision, timestep, or number
of steps is changed for timing fairness. The new CPU selector metric only
widens the timer to include the already-existing state copy; the legacy CPU
integration timer remains unchanged.

## Accuracy and admissibility

Each measured final state is observable through diagnostics and a state
checksum. A row is valid only if all numerical fields are finite and:

```text
abs(E-E0) / max(abs(E), abs(E0), 1 joule) <= 5e-6
norm(L-L0) / max(norm(L0), 1 kg m^2/s) <= 5e-10
```

The bounds are frozen and are not tuned from Stage 5 results. Existing CTest
cross-backend validation must pass on the RTX host before the pilot. A failed
configuration is inadmissible regardless of speed and invalidates its
performance summary pending investigation; failed rows are never averaged
away. Checksums must repeat within each backend/thread/N configuration.

## Raw evidence and statistics

`raw_timings.csv` has one row per timing metric per measured invocation and
includes environment ID, source commit, protocol version, provenance mode,
pilot/full role, backend, requested/effective threads, N, repetition block,
order, steps, timestep, metric, elapsed milliseconds, correctness status,
checksum, conservation drift, bounds, and available CUDA/build identity.

`summary.csv` is deterministically derived from raw rows and reports sample
count, median, mean, sample standard deviation, coefficient of variation,
minimum, maximum, IQR, and a deterministic 10,000-resample percentile
bootstrap 95% interval for the median. `environment.json` records Git, OS,
CPU/topology, RAM, GPU, NVIDIA/CUDA, build/tool versions and flags, OpenMP
environment, protocol, seeds, telemetry availability, and executable hashes.
`manifest.json` hashes raw, summary, environment, and validation-log artifacts.

Only full-run measurements from this one Stage 5 environment may be compared
to produce CPU/OpenMP/CUDA speedups or later selector calibration. Pilot data,
Stage 3 data, and Stage 4 data remain separate.

## Completion and pilot gate

Stage 5B proceeds in two explicit commands and directories: pilot, independent
verification, human inspection/approval, then full. The runner never chains a
pilot into a full run. A full run is prohibited until device identity,
toolchain, tests, row/schema checks, conservation, repeatability, timing
relationships, and estimated duration from the pilot have been reviewed.
