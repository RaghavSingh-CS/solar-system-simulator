# Large-N Laboratory

## Scope

Phase L1 is a headless, reproducible laboratory for synthetic gravitational
workloads. It reuses the reviewed Direct and Barnes--Hut force backends and the
unchanged Velocity Verlet integrator. It does not add a gravitational
algorithm, alter the viewer, or modify frozen Stage 5. Supported requested body
counts include 256, 1024, 4096, 8192, 16384, 32768, 50000, and 100000, but a
request is not evidence that a workload is feasible.

The bounded CPU implementation, publication tooling, and retained exploratory
pilot are committed and have passed independent source/custody review.

The executable is `nbody_large_n_lab`. Backend and solver are independent:
Sequential/Direct, OpenMP/Direct, Sequential/Barnes--Hut, and
OpenMP/Barnes--Hut are executable in the appropriate build. CUDA is a metadata
value only in L1 and fails admission explicitly. There is no solver fallback.

## Deterministic workloads

`LargeNGenerationConfiguration` requires an explicit 64-bit seed and generates
stable ordered names `large_n_00000000`, `large_n_00000001`, and so on. A fixed
SplitMix64 stream and explicit 53-bit integer-to-binary64 mapping avoid
standard-library distribution variability. Reproduction is promised on the
declared toolchain; floating-point center-of-mass reduction is not claimed to
be bitwise portable across arbitrary compilers or architectures.

Three generic SI-valued distributions are available:

- `uniform`: positions in a three-dimensional cube;
- `clustered`: four deterministic separated centers with bounded jitter;
- `planar`: full in-plane extent with one-percent vertical thickness.

Mass and position/velocity scales are configurable. Generation validates all
values, checks allocation arithmetic, recenters position and velocity, rejects
duplicate final positions, and fails on a nonrepresentable reduction. These
fixtures are not astronomical models. Coincident-body states exist only as
explicit adversarial test fixtures and retain the solvers' unsoftened
point-mass rejection semantics.

## Admission and execution safety

Preflight checks body, step, warm-up, repetition, memory, and direct-work
limits with overflow-safe arithmetic. The default bounds are 100,000 requested
bodies, 512 MiB conservative laboratory memory, 500,000,000 estimated direct
interactions, 1,000 steps, 10 warm-ups, and 100 retained repetitions. Memory is
estimated at six explicit lifetime boundaries: generation, Direct-reference
construction, warm-up, retained repetition, final characterization, and result
serialization. The high-water admission value is the maximum of those phases.
It accounts for every simultaneously retained `SimulationState`, position/
velocity/acceleration and Velocity Verlet buffer, Direct and candidate
trajectory, first retained force result, relative-error/percentile scratch,
measurement/row containers, conservative body-name and uniqueness-hash
storage, vector overhead/capacity reserve, and the full configured
Barnes--Hut working-memory budget where it overlaps. Checked addition and
multiplication are used throughout.

This is a conservative policy estimate, not an allocator-exact process-RSS
guarantee. C++ allocator metadata, runtime and OpenMP state, executable/shared-
library mappings, stacks, and operating-system bookkeeping are outside the
accounted byte total. Barnes--Hut retains its independent solver-internal
working-memory gate.

Direct work charges measured force calls, every force call in each Velocity
Verlet step, complete Direct reference trajectories, and three pairwise
conservation evaluations. A default 100K Direct request is therefore rejected.
Barnes--Hut retains its separate node, depth, traversal-work, minimum-cell, and
working-memory limits. OpenMP admission records and requires the actual runtime
team size. CUDA execution is rejected in L1.

The C++ runner checks elapsed time at completed scientific boundaries,
including one final boundary after trajectory comparisons and conservation
diagnostics. These checks do not preempt a solver or diagnostic already in
progress. The pilot orchestrator additionally applies a hard subprocess
timeout, including launch/orchestration overhead, so a noninterruptible call
still has an external execution boundary. Failed experiments publish no
repetition or scientific result fields. Before `completed`, every required
diagnostic, force/trajectory error, drift, timing, and derived division is
checked for a finite representable result. Explicitly unavailable Direct
references use status/reason fields, never an accidental NaN.

## Timing methodology

Input generation is timed separately. `initial_direct_force_s` is exactly one
initial Sequential Direct acceleration call; `direct_trajectory_s` is the
matched Direct Velocity Verlet trajectory; and `diagnostics_s` contains three
conservation evaluations. Each retained repetition reports one complete force
call and the configured complete Velocity Verlet step batch. For Barnes--Hut,
the reviewed backend also reports tree construction and target traversal,
accounted working memory, nodes, visits, approximations, direct leaf
interactions, and effective worker count. Total experiment duration begins at
generation and ends after final scientific characterization; it excludes CSV
serialization and process-launch/evidence-orchestration overhead.

Tree construction plus traversal are components of a Barnes--Hut force call;
they must not be added to the already complete force wall time. A complete
Velocity Verlet step contains two force evaluations. None of these fields is
graphical FPS, frame time, or renderer performance.

## Numerical characterization

Where admitted, Sequential Direct is the force and short-trajectory reference.
The laboratory records maximum and RMS absolute acceleration error, p50/p95/p99
and maximum relative error, an absolute denominator floor, and the number of
cancellation-sensitive reference accelerations. It also compares matched-step
position and velocity separation and reports each candidate and Direct
reference's energy and angular-momentum drift separately.

Theta `0.25`, `0.5`, and `0.8` are measurement points, not universal error
bounds. Where Direct is disabled or exceeds the reference body limit, the
record explicitly says the reference is unavailable; a second Barnes--Hut run
is never relabeled exact. Large-N performance-only runs still require finite
state and bounded internal-work checks.

## L1 pilot and custody

The bounded Ryzen 7 5700U pilot stopped at 4096 bodies. It contains 144 rows
covering three distributions, Sequential/OpenMP-4, Direct/Barnes--Hut, three
theta values, one warm-up, and two retained repetitions. The evidence and its
limitations are documented under
`data/benchmarks/large_n_l1_pilot/`. Two repetitions, uncontrolled affinity and
thermals, and a single machine make the crossover observations exploratory.
They are not RTX 4050 predictions and do not establish 50K/100K feasibility.

Each run is assembled in a unique `build/scratch` staging directory. Raw data,
summary, metadata, build provenance, source manifest, and README are completed,
strictly verified, and bound by a per-run `run_manifest.json`. Publication is
serialized by an OS-backed interprocess lock. A complete run directory is then
installed exactly once under `runs/<run-id>/`, and the root
`publication.json` pointer is replaced atomically. The pointer names one
authoritative run and retains immutable descriptors for prior runs.

Timeout, malformed output, provenance failure, or verification failure before
installation leaves the previous pointer untouched. A crash after run
installation but before pointer replacement leaves a detectable unreferenced
run; later publication fails closed until an operator explicitly recovers that
run. Incomplete install directories, missing/corrupt authoritative runs,
stranded flat artifacts, and legacy directory-swap backups are also rejected.
The previous flat publication was migrated byte-for-byte into a named legacy
history run before the corrected pilot was published.

A legacy directory-swap backup is recovered only by explicit administration:

```powershell
python -B scripts/run_large_n_l1_pilot.py --output data/benchmarks/large_n_l1_pilot --recover-legacy-backup
```

Recovery requires exactly one `.large_n_l1_pilot.previous-*` directory with a
complete legacy publication manifest and matching artifact hashes. The same
OS-backed lock serializes recovery with publishers. The candidate is copied to
temporary installation storage, checked through the strict verifier, assigned
a content-derived `legacy-backup-<publication-sha256>` identity, installed as
an immutable run, and added to history by atomic pointer replacement. If no
current run exists, the recovered run becomes current but remains explicitly
historical in identity and provenance; it is never presented as a new
measurement. If a current run exists, its descriptor and bytes remain
unchanged. The source backup is not deleted. Repeated recovery returns
`already_adopted`; zero backups returns `no_recovery_needed`. Multiple,
malformed, corrupt, ambiguous, incompletely installed, or unrelated orphaned
candidates fail closed without replacing current authority.

The lock is released by the operating system on process exit, and successful
same-filesystem `os.replace` gives atomic pointer visibility under the validated
Windows process-crash model. A crash before pointer replacement leaves either
the original backup or both it and a complete unreferenced copy. The temporary
pointer file is flushed and passed to `fsync`, but this does not guarantee that
directory-entry updates or device caches survive sudden power loss. No
power-loss durability certification has been performed.

`raw.csv` is byte-custodied as non-text. JSON, Markdown, manifest, and all
manifest-listed source inputs use LF checkout rules. The read-only verifier
requires the exact 144 rows/72 groups and fixed matrix, validates budgets,
workers, repetitions and finite, semantically nonnegative fields, reconstructs every timing/scientific/
resource summary value, checks every source/provenance/publication hash, and
checks the canonical executable hash when that executable is present. OpenMP
Barnes--Hut reports four workers observed inside the timed region. OpenMP
Direct records four requested/effective workers but `observed_worker_threads=0`
and `runtime_probe_only`, because the protected timed Direct region has no safe
worker observation hook.

## L2 controlled-scaling pilot

Phase L2 reuses the L1 executable, checked admission, finite publication gate,
locking, atomic pointer, and immutable run layout. The machine-readable
protocol and rationale are in `tools/large_n_l2_protocol.json` and
`docs/large_n_l2_protocol.md`. Candidate sizes are 8,192, 16,384, 32,768, and
50,000, with 100,000 optional, but the first task executed only 8,192.

The authoritative run
`run-20261015T040233.498852Z-9878e74b0e13` contains 60 rows: 54 completed
Barnes--Hut repetitions and six explicit Direct resource skips. Every
Barnes--Hut group used a complete all-body Sequential Direct force and matched
one-step trajectory reference. No sampled reference exists. OpenMP groups
record four workers observed in the timed region.

Across the three distributions, the largest measured errors at each theta
were:

| Theta | Max absolute acceleration error (m/s^2) | Max RMS absolute error (m/s^2) | Max P95 relative error | Max relative error |
|---:|---:|---:|---:|---:|
| 0.25 | 3.0168081821199618e-6 | 9.539488371983964e-7 | 0.003394105263629308 | 0.0315266838299639 |
| 0.5 | 4.472887708798449e-5 | 1.611858328840151e-5 | 0.019780346613629218 | 0.21783472826199934 |
| 0.8 | 2.553484338127513e-4 | 7.390546265847096e-5 | 0.06739768993217483 | 0.730731603862977 |

All pre-registered theta-specific gates passed. The largest candidate relative
energy drift was `8.539377768340509e-10`; the largest relative angular-momentum
drift was `8.978002562373171e-6`. These are one-step synthetic-fixture results,
not long-duration or universal accuracy guarantees.

Median complete force times ranged from 0.0382162 to 1.7239281 seconds and
median complete timesteps from 0.0759511 to 3.5933919 seconds (0.278289 to
13.166366 steps/s). Tree construction was 0.0012593--0.0014887 seconds; force
traversal was 0.0366581--1.7224039 seconds. OpenMP/Sequential complete-force
ratios for matched groups ranged from 3.5331 to 4.0867. Direct solver groups did
not complete, so the pilot establishes no Direct/Barnes--Hut crossover. The
complete Direct reference force component ranged from 0.6591396 to 0.7235039
seconds but is not a retained Direct solver configuration.

Maximum conservative admitted memory was 273,351,848 bytes (50.92% of the
512 MiB budget), maximum charged work was 301,953,024 interactions, maximum
median total experiment time was 24.3093273 seconds (81.03% of budget), and
maximum force range-over-median was 26.156%. The wall-time result exceeds the
pre-registered 80% continuation limit, so the protocol does not recommend
advancing automatically. It does not show that a complete Direct reference is
admissible at 16,384. No larger size was executed.

The current pointer retains both earlier exact publications as immutable
history. The current run binds 26 exact source/test files and passes protocol-
derived row, command, admission, source, and artifact reconstruction. It is
committed and independently source-reviewed; it is not frozen Stage 5 evidence.

## L3 performance-only checkpoint

Phase L3 adds a separate performance-only protocol for 16,384, 32,768, and
50,000 bodies. It disables complete and sampled Direct references and O(N^2)
conservation diagnostics. The L3 publication layer validates the laboratory's
legacy no-reference zero placeholders, then leaves all published scientific
error and drift fields empty and carries explicit `unavailable_performance_only` accuracy
and conservation status. Finite Barnes--Hut forces and post-step states remain
mandatory.

The first bounded 16,384 attempt began with the predeclared
Sequential/uniform/theta-0.25 configuration. That subprocess exceeded the
70-second hard timeout before producing a completed record. The runner stopped
immediately, cleaned its scratch staging directory, and did not execute the
remaining five configurations. No L3 evidence pointer, immutable run, timing,
throughput, or scientific result was published. Limits were not changed and no
32,768 or 50,000 work was attempted.

Before the retained attempt, one disposable single-repetition Sequential,
uniform, theta-0.5 calibration with no warm-up exercised the performance-only
path. It was neither retained nor treated as controlled evidence. The 60-second
internal and 70-second external limits were fixed before the retained pilot;
the timeout did not trigger a further increase.

The tooling and protocol are committed and independently closeout-reviewed;
there is deliberately no L3 evidence commit. A future retry requires a
separately reviewed protocol decision; this failed attempt itself does not
authorize a reduced matrix, fewer repetitions, a larger timeout, or automatic
progression. It establishes no throughput, numerical-accuracy, memory-headroom,
graphical-FPS, GPU, or 50,000-body feasibility result.

## Future gates

Before any full-scale experiment, independently review L1 admission, timing,
statistics, custody, and pilot interpretation. A later controlled phase must
pre-register larger workloads and stopping rules. CUDA Barnes--Hut requires
compilation and real execution on the RTX 4050; GPU compute throughput and
viewer rendering FPS remain separate experiments.
