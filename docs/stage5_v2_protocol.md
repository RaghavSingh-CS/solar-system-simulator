# Stage 5 same-machine protocol (version 2.0)

## Status, versioning, and research boundary

Protocol identifier: `stage5-v2.0`.

Version 2.0 prospectively replaces the unexecuted `stage5-v1.0` preparation at
revision `4916b3c43298fa3f3bd15cb4af1698b5074abed3`. Version 1.0 remains an
immutable historical artifact. Version 2.0 changes evidence custody, build
provenance, pilot approval binding, and topology validation; it does not change
the scientific workload, numerical gates, timing regions, or checksum semantics.

The experiment asks whether one calibrated policy can map problem size `N` to
the fastest valid configuration among Sequential, topology-derived OpenMP
teams, and CUDA on one Windows/RTX 4050 workstation for the fixed workload in
this document. It does not demonstrate a selector over arbitrary step counts,
precisions, accuracy budgets, scenarios, seeds, or cold-start conditions.

Stage 3 Windows CPU and Stage 4 Colab T4 timings are not inputs. Pilot data are
feasibility evidence only. No Stage 5 timing exists until execution on the
controlled NVIDIA host.

## Frozen scientific workload

Every controlled acquisition uses:

- deterministic seeded synthetic scenario, seed `24301`;
- binary64 state and arithmetic;
- direct Newtonian O(N^2) force evaluation;
- fixed-step Velocity Verlet;
- `dt=10 s`;
- one untimed same-workload warm-up immediately before each measured invocation;
- `selector_steady_state` as the policy metric;
- relative energy-drift gate `5e-6`;
- relative angular-momentum-vector drift gate `5e-10`;
- order seed `20260926`.

The controlled calibration and held-out phases each use 100 integration steps
and 11 complete randomized repetition blocks. The pilot uses 10 steps and three
complete randomized blocks.

| Role | N values |
|---|---|
| Pilot only | 64, 512, 4096 |
| Calibration | 64, 128, 192, 256, 512, 1024, 2048, 4096 |
| Held-out evaluation | 96, 160, 384, 768, 1536, 3072 |

## Required phase separation and custody

The workflow is strictly sequential and uses separate absent output
directories. No command chains phases automatically.

### Phase A — pilot

The pilot schedules only `64,512,4096`. It validates source/build/environment
identity, OpenMP teams, CUDA device identity, checksums, gates, timing
relationships, stability, and projected duration. Pilot results never train or
modify the policy and cannot be pooled with calibration or held-out evidence.

The pilot directory is independently verified. Human approval names the exact
SHA-256 of that pilot's `manifest.json`, plus the matching protocol, source,
build-manifest, environment, and OpenMP topology identities. Approval for one
pilot cannot unlock another build or environment.

### Phase B — calibration acquisition

Calibration schedules only
`64,128,192,256,512,1024,2048,4096`, with deterministic randomized ordering
within each of 11 complete blocks. No held-out N may be scheduled, printed,
measured, written, or summarized by this phase.

Calibration requires the actual approved pilot directory. The runner invokes
the independent verifier over that directory and checks the resulting manifest
hash against human approval before starting. Output is an immutable,
calibration-specific evidence directory. Verification rejects any held-out N,
missing or duplicate row, incorrect order, nonfinite timing, failed gate,
checksum instability, timing relationship error, or identity mismatch.

### Phase C — deterministic policy freeze

The policy builder accepts only a verified `stage5-v2.0` calibration directory.
For every correctness-eligible configuration it computes the median
`selector_steady_state` time at every calibration N. Thread count is part of an
OpenMP configuration.

For a query inside the calibrated range, each configuration's runtime is
predicted by piecewise-linear interpolation in `(log(N), log(median_time))`
between adjacent calibration N values. The configuration with the smallest
prediction is selected. Exact ties use this canonical ordering:

1. `sequential:1`;
2. `openmp:T` ordered by ascending integer thread count;
3. `cuda:0`.

The ordering is a deterministic tie-break only, not a preference learned from
held-out results. The builder does not smooth, refit, remove crossovers,
hand-adjust thresholds, or extrapolate. All held-out N values lie inside the
calibration range.

The machine-readable `policy.json` records protocol and algorithm versions,
source SHA, build-manifest hash, calibration-manifest hash, environment
identity/hash, OpenMP list, metric, calibration medians, canonical ordering,
interpolation definition, and the selected configuration for every held-out
query N. A generation timestamp is metadata only. `policy_sha256` is computed
from canonical policy content that excludes timestamp and the hash field itself.

`policy_freeze.json` binds the policy hash, calibration-manifest hash,
build-manifest hash, environment hash, source SHA, and protocol. Any change to
calibration data, policy content, build, source, or environment breaks the
chain. There is no human threshold-editing path.

### Phase D — held-out acquisition

Held-out acquisition schedules only `96,160,384,768,1536,3072`, with
deterministic randomized ordering within each of 11 complete blocks. It is
unavailable unless:

- calibration verification passes;
- `policy.json` and `policy_freeze.json` exist and their hashes match;
- source, build-manifest, environment, protocol, and topology identities match
  calibration;
- calibration files still match their manifest.

Once this immutable policy chain is verified, held-out output may be visible.
The held-out runner cannot regenerate or replace the policy; any replacement
changes its hash and invalidates evaluation.

### Phase E — held-out evaluation

Evaluation reads the frozen policy and independently verified held-out evidence.
For each held-out N it reports the policy-selected configuration without
refitting, its actual held-out median, the empirical fastest eligible
configuration and median, multiplicative regret
`selected_median / oracle_median`, and whether the selected configuration was
empirically fastest.

Evaluation never rebuilds the policy. Six held-out points do not support a
universal best-backend or universal crossover claim.

## OpenMP feature and topology contract

The project does not require OpenMP 4.5 language features. Historical GCC
evidence reporting `_OPENMP=201511` remains accurate, but Stage 5 validates the
feature subset it actually uses: OpenMP compilation, a real parallel team,
requested/effective `num_threads`, `schedule(static)` target execution,
Sequential/OpenMP numerical agreement, deterministic repetition, and protocol
thread constraints. Ordinary MSVC `/openmp` is acceptable only when those
semantic tests pass. `/openmp:llvm` is not selected merely to increase a
version macro.

On controlled Windows, native processor information supplies total physical
cores `P` and logical processors `L` across all packages. The exact sorted,
unique thread list is `1`, every power of two strictly below `P`, `P`, and `L`,
with values above `L` omitted. Failure to obtain reliable `P` and `L` is a hard
preflight failure; no value is guessed. The inputs, derivation, and final list
are recorded and must match pilot, calibration, policy, and held-out artifacts.

The measurement environment sets `OMP_DYNAMIC=FALSE` and
`OMP_PROC_BIND=FALSE`; `OMP_PLACES`, `GOMP_CPU_AFFINITY`, `OMP_DISPLAY_ENV`,
and `OMP_DISPLAY_AFFINITY` are unset. Windows scheduler placement is used and
affinity control is not claimed.

## Build and test provenance

Controlled preparation requires an absent build directory and a clean checkout
whose HEAD equals the deployment packet's source SHA. The build manifest binds:

- protocol and exact source SHA;
- source directory and `CMAKE_HOME_DIRECTORY`;
- generator and `CMAKE_BUILD_TYPE=Release`;
- C++ and CUDA compiler absolute paths, IDs, versions, and CUDA host compiler;
- CUDA architecture `89`;
- OpenMP flags/runtime identity where available;
- relevant `NBODY_*` options;
- `--build-info` output embedded in benchmark/test binaries;
- SHA-256 of CPU/CUDA benchmark and Stage 5-relevant test executables;
- controlled CTest command, output/log hash, return code, and pass status;
- toolchain/topology capture and its environment hash.

Pilot, calibration, policy, and held-out manifests transitively bind the source,
build-manifest hash, executable hashes, CTest result, environment hash, and
OpenMP topology. A detached CTest text file or copied stale executable is not
accepted.

The preferred deployment family is Windows x64, Visual Studio 2022/MSVC,
ordinary feature-tested MSVC OpenMP, Ninja, CMake, Python 3.12-class, Git, and
CUDA Toolkit 12.8.x with native Ada `sm_89` support. This is a candidate family,
not a claim that the target laptop's exact driver, VS patch, or Toolkit tuple has
already passed. The exact observed tuple is captured during NVIDIA preflight.

## Timing definitions

Timing regions are unchanged from v1.0:

| Metric | Backend | Boundary | Use |
|---|---|---|---|
| `integration_compute` | Sequential/OpenMP | steady-clock integration only; initial copy excluded | implementation analysis |
| `device_compute` | CUDA | CUDA events after H2D through all per-step kernels | kernel-path analysis |
| `transfer_inclusive` | CUDA | CUDA events from H2D through compute, D2H, and completion | transfer-inclusive analysis |
| `selector_steady_state` | Sequential/OpenMP | initial-state copy through completed integration | primary policy metric |
| `selector_steady_state` | CUDA | packing, H2D, compute, D2H, synchronization, output materialization, validation | primary policy metric |

Reusable CUDA context/allocation setup remains separate. The primary metric is
warmed steady-state cost returning a usable host state, not cold-start
application time. Scenario generation, process startup, warm-up, diagnostics,
checksum, formatting, runner I/O, and summaries are excluded consistently.

## Accuracy, checksums, and evidence completeness

A measured invocation is valid only when all required values are finite and:

```text
abs(E-E0) / max(abs(E), abs(E0), 1 joule) <= 5e-6
norm(L-L0) / max(norm(L0), 1 kg m^2/s) <= 5e-10
```

CUDA additionally reports passing backend correctness. Requested/effective
OpenMP teams must match. Checksums repeat within every backend/thread/N
configuration, and Sequential/OpenMP checksums agree at each N. Failed rows are
retained but cannot be summarized as eligible evidence. Partial output and a
failure record remain available; they never satisfy a complete manifest and are
never resumed or overwritten.

Each completed acquisition contains `raw_timings.csv`, `summary.csv`,
`environment.json`, and `manifest.json`. Their schema records phase, role,
source/build/environment identities, schedule, metrics, gates, checksums,
runtime/compiler identity, and hashes. Verification deterministically
reconstructs schedules and statistics and rejects extra, missing, mixed-role,
or modified content.

## Deployment and first-rehearsal boundary

Tracked files deliberately contain no self-referential final commit SHA. After
the candidate v2.0 commit exists, `scripts/create_stage5_deployment_packet.py`
generates an external, hashed operator packet for that exact SHA. The generated
packet—not an in-tree placeholder—is authoritative for checkout and commands.

The first Windows/NVIDIA rehearsal stops after: fresh clone, exact SHA and clean
status, tool/topology capture, absent build directory, Release configure/build,
cache/compiler verification, executable hashes, controlled CTest/build
manifest, Stage 5 preflight, pilot, independent pilot verification, and
manifest-bound human approval. It must not proceed to calibration during that
first rehearsal.
