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

## Future gates

Before any full-scale experiment, independently review L1 admission, timing,
statistics, custody, and pilot interpretation. A later controlled phase must
pre-register larger workloads and stopping rules. CUDA Barnes--Hut requires
compilation and real execution on the RTX 4050; GPU compute throughput and
viewer rendering FPS remain separate experiments.
