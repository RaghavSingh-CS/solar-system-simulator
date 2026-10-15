# Validation plan

## Foundation tests

- Analytic circular two-body state is barycentric and has near-zero total momentum.
- Pair accelerations satisfy momentum balance within floating-point tolerance.
- After repeated Velocity Verlet steps, separation and center of mass remain within declared tolerance.
- Relative total-energy and angular-momentum drift remain bounded over the test horizon.
- CSV runner emits deterministic row counts and documented columns (manual smoke check in this stage).

The analytic binary uses deliberately generic masses and separation. It tests mechanics; it is not presented as Sun/Earth ephemeris data.

## Cross-backend validation

For each `(scenario, N, dt, steps, precision)`:

1. Generate one seeded input and preserve it.
2. Run the sequential double-precision reference.
3. Compare OpenMP/CUDA final states using normalized RMS position and velocity error.
4. Independently compute total energy, linear momentum, angular momentum, and center of mass.
5. Mark the backend/configuration invalid if any preregistered threshold fails.

Stage 2 uses OpenMP thread counts 1, 2, and 4 on both the analytic binary and a fixed-seed 32-body scenario. Mixed comparison tolerances handle near-zero quantities:

- Position: absolute `<= 1e-6 m` **or** scaled relative `<= 1e-13` (denominator floor `1 m`).
- Velocity: absolute `<= 1e-9 m/s` **or** scaled relative `<= 1e-13` (denominator floor `1 m/s`).
- Cross-backend momentum: absolute `<= 1e6 kg m/s` **or** normalized by total scalar momentum `<= 1e-13`.
- Cross-backend center of mass: absolute `<= 1e-6 m` **or** normalized by initial position scale `<= 1e-13`.
- Per-backend normalized momentum and center-of-mass drift: `<= 1e-12`.
- Relative energy drift `<= 5e-6`; normalized angular-momentum drift `<= 5e-10`.

The absolute momentum bound appears large only because masses are astronomical; the normalized alternative is the meaningful general criterion. Tests also require repeatability under the same mixed position/velocity tolerances and reject zero thread counts, zero timesteps, negative `G`, and coincident positions.

### Stage 2 observed results

With GCC 15.1.0/libgomp in Release mode, every sequential-vs-OpenMP and repeated-OpenMP position/velocity difference was exactly zero for 1, 2, and 4 threads. This exact equality is an observation explained by per-destination ownership and identical inner accumulation order, not a portability requirement.

- Analytic binary over two periods: energy drift `1.5114788315343793e-15`, angular-momentum drift `3.5096052562406509e-16`, normalized momentum drift `8.4290933028685295e-16`, normalized center-of-mass drift `1.9951356883117746e-20`.
- Seeded 32-body system over 250 steps: energy drift `1.2963732290918661e-15`, angular-momentum drift `7.6765086087405374e-16`, normalized momentum drift `1.5187263362149679e-16`, normalized center-of-mass drift `8.1671414833805418e-17`.

Initial thresholds for the analytic two-body CI test are implementation guards, not scientific claims: maximum relative energy drift `<= 5e-6`, relative separation error `<= 5e-5`, normalized angular-momentum drift `<= 5e-10`, and center-of-mass drift `<= 1e-9` of initial separation over five periods. Scenario-specific research thresholds must be fixed before collecting benchmark outcomes.

## Real Solar System validation

- Obtain an auditable JPL Horizons barycentric state at one declared epoch.
- Validate parser/unit/frame behavior independently.
- Compare short-horizon output at later Horizons epochs with position/velocity errors per body.
- Record omitted physics (relativity, oblateness, minor bodies) and do not attribute model error to numerical integration.
- Use conservation diagnostics as internal checks, not as proof of ephemeris accuracy.

## Barnes--Hut Stage 1--2 validation

The direct solver remains the numerical reference. Barnes--Hut tests retain
absolute, RMS, maximum-relative, and 95th-percentile relative acceleration
errors because relative error alone is unstable for near-zero net forces.
Coverage includes empty and singleton states, analytic two-body equivalence,
seeded asymmetric clusters, planar/collinear distributions, translation and
consistent mass/length scaling, extreme representable binary64 coordinates,
near-coincident and exact-coincident inputs, and maximum-depth/node/traversal/
memory termination. Repeated builds must reproduce forces and structural traversal
counters exactly.

The accuracy-first baseline is `theta=0.25`; `theta=0.5` and `0.8` are retained
as exploratory tradeoff points, not globally valid accuracy settings. A
near-zero theta test requires all non-self pairs to reach direct leaves and
agree with direct force within floating-point accumulation tolerance. A short
48-body trajectory compares direct and approximate results under identical
Velocity Verlet timesteps and characterizes energy drift. These bounded
synthetic checks do not establish long-horizon orbital or ephemeris accuracy.

Resource tests require checked size arithmetic, explicit failures at node and
memory limits, bounded depth for highly unbalanced trees, and rejection of
non-finite/nonrepresentable geometry. A headless CTest exercises the supported
`sequential + barnes-hut` composition. OpenMP builds also exercise
`openmp + barnes-hut` and retain a separate `openmp + direct` CLI check;
CPU-only builds require OpenMP selection to fail so fallback cannot be silent.

The Stage 2 suite compares the shared Sequential and OpenMP tree/traversal at
theta `0.25`, `0.5`, and `0.8` over uniform, clustered, and planar inputs. It
requests 1, 2, 4, and 8 workers where supported and requires exact force,
Velocity Verlet state, conservation diagnostic, and structural/work-counter
agreement. It also covers boundary ties, unequal masses, extreme binary64
coordinates, near/exact coincidence, depth/node/memory/traversal limits,
repeated determinism, transactional last-success statistics, actual-capacity
memory admission, and nested-region rejection with nesting both disabled and
enabled. Strict-budget failures are repeated at 1/2/4/8 threads under actual
static, dynamic, and guided schedules in a test-only `schedule(runtime)`
compilation of the same implementation; production remains explicit static.
A compare/exchange reservation proves reported attempt work never exceeds the
configured cap. Separate charged-work counters cover reservations made before
coincident/nonrepresentable-force failures, and empty success resets the
attempt diagnostic. No replay receives a fresh budget, and no partial output
or statistics may be published. Representation comparisons cover every
diagnostics field individually (kinetic, potential, and total energy; linear
and angular momentum; and center of mass) and include a signed-zero negative
control. No sleep or timing assertion is part of correctness validation.

Exploratory scaling records tree build, parallel traversal, complete force
call, complete Velocity Verlet step, and direct-force comparators separately.
It is not a CI performance gate, Stage 5 evidence, an automatic-selector input,
or evidence of graphical FPS or 50K--100K feasibility.

### CUDA Barnes--Hut Stage 3A gates

Implementation `5e1864e` passed the independent source-level commit gate. This
does not satisfy CUDA compilation or NVIDIA execution gates.

CPU-only CI validates the pointer-free packed-tree representation independently
of CUDA: standard-layout/trivial-copyability, deterministic node and body order,
parent-before-child ranges, inverse-rank identity, finite monopoles/bounds,
empty, single-body, explicit two-body, coincident, and near-degenerate cases,
input rejection, contiguous child-range coverage, and separate packing-memory
admission. CPU and OpenMP full suites must remain independent of CUDA tooling.

The CUDA-only `nbody_barnes_hut_cuda` executable is registered only in CUDA
builds and uses the strict runtime gate shared by existing GPU tests. Exit 77 is
valid only when successful enumeration reports zero devices. Enumeration,
allocation, transfer, launch, kernel, synchronization, or readback errors are
failures. No host mock may satisfy the CUDA execution gate.

Required NVIDIA fixtures cover single/two-body, uniform, clustered, planar,
unequal-mass, octant-boundary, coincident, near-coincident, extreme binary64,
small/large theta, maximum-depth termination, node/host/device-memory limits,
traversal-stack and traversal-work exhaustion, and repeated executor lifetime.
Every successful output must be finite. Sequential Barnes--Hut is the force
equivalence reference under predeclared mixed absolute/relative tolerance;
direct force independently characterizes absolute, RMS, maximum-relative, and
95th-percentile approximation error. CUDA compilation or a skipped test is not
GPU numerical validation.

Device-failure validation repeats highly contended singular and work-budget
failures without assuming which thread wins. Each completed record must have an
allowed code and target/node indices inside the exact fixture/tree, while the
throwing API must leave caller-owned output unchanged. A deterministic
host-side live-release injection must preserve last-success statistics,
terminalize the executor before replacement allocation, and reject reuse.

RTX validation must also audit device architecture/toolchain provenance,
stream synchronization and terminal-failure quarantine, repeated allocation
cleanup, and actual persistent/local-memory resource usage. Performance remains
unclaimed until tree-build, transfer, traversal, complete force, and later
complete-step timings are measured separately under a controlled protocol.

### CUDA Barnes--Hut Stage 3B gates

The NVIDIA-only `nbody_barnes_hut_cuda_integration` test must execute genuine
Stage 3A force kernels twice per step. It covers empty, inertial singleton,
analytic two-body, unchanged legacy Solar, a real JPL-derived subset, uniform,
clustered, planar, small/large-theta and multistep fixtures. Sequential
Barnes--Hut is the same-solver reference under fixed mixed tolerances; direct
force is a separate approximation-error characterization. Conservation checks
remain bounded regression guards, not accuracy guarantees.

Transactional tests must preserve the exact caller state and last-success
statistics after invalid timestep, first-force failure, deliberately arranged
second-force collision, naturally reached drift/kick non-finiteness,
nonrepresentable geometry, stack/work/memory failure, and terminal live-release
injection. They verify bitwise caller rollback, complete last-success statistics,
phase-specific failure metadata, nonterminal retry, and terminal reuse
rejection. A separately compiled test-library variant injects only at the
otherwise allocator/invariant-dependent post-force resource boundaries and
must prove the dedicated post-force-validation phase without exposing hooks in
production. Per-step reported work must equal both force-call counters and
remain under the checked two-evaluation limit. CPU-only CTest must reject both
`cuda + direct` and `cuda + barnes-hut` headless selections.

Direct-force characterization must use independently advanced direct and
Sequential Barnes--Hut references from the same initial state, timestep, and
horizon. CUDA-Barnes--Hut/direct maximum and RMS position/velocity separation,
plus per-body separation vectors, must match the Sequential-Barnes--Hut/direct
characterization under fixed mixed absolute/relative tolerances. Merely finite
separation is not an acceptance test.

CUDA configuration or skip 77 is not Stage 3B validation. RTX 4050 acceptance
requires nvcc/MSVC compilation for architecture 89, actual execution without
unexpected skips, numerical and conservation results, both-force rollback
cases, repeated lifetime/cleanup, resource accounting, and separately reported
tree-build, transfer, force, and complete-step timing.

Implementation `2cc2cb0` passed the source-level commit gate. This records no
CUDA compilation or execution result; every NVIDIA gate above remains open.

## Reproducibility

Every result row must carry commit, build type, compiler/options, backend, hardware, scenario hash, seed, precision, N, dt, steps, repetition, and warm-up policy. Failed runs remain in raw data with a reason.

## Stage 4 CUDA criteria (frozen before GPU execution)

CUDA uses binary64 (`double`) for masses, positions, velocities, accelerations, constants, and accumulation. CPU/CUDA bit identity is not required because host and CUDA compilers may contract multiply-adds differently even when each destination body preserves ascending source order.

CUDA validation covers the analytic two-body system, seeded 8/32/128-body force evaluations, seeded 32-body integration, repeated CUDA integration, conservation diagnostics, collision rejection, NaN rejection, CUDA API failures, launch failures, and synchronization failures.

Predeclared CPU/CUDA mixed tolerances are:

- acceleration: absolute `1e-18 m/s²` or scaled `1e-12`;
- final position: absolute `1e-3 m` or scaled `2e-12`;
- final velocity: absolute `1e-8 m/s` or scaled `2e-12`;
- CPU/CUDA final energy and angular momentum: scaled `2e-12`;
- CUDA energy drift guard: `5e-6`;
- benchmark angular-momentum drift guard: `5e-10`;
- normalized linear-momentum and center-of-mass drift: `1e-12`.

These thresholds are deliberately looser than the exact CPU/OpenMP equality observed in Stage 2, yet far below the integrator-level drift guards. They must not be changed merely to accept a failing GPU result; any revision requires preserved failure evidence and a numerical rationale.

### Stage 4 observed results

On the Tesla T4, Release CTest passed `2/2`: the foundation test and `nbody_cuda_validation`. The CUDA suite exercised analytic and seeded force comparison, analytic and seeded integration against the CPU reference, repeat execution, energy/angular/linear-momentum/center-of-mass guards, coincident-position rejection, and NaN-input rejection. All predeclared thresholds above remained unchanged.

The CTest executable reports pass/fail rather than individual CPU/CUDA errors, so the retained log establishes that the errors were within tolerance but does not support a bitwise-identity or exact-error claim. In the controlled benchmark, all 154 rows passed; repeat checksums match exactly within each body count. Maximum recorded relative energy drift was `1.1779626463387057e-13`, and maximum relative angular-momentum drift was `3.740862421005083e-15`, below the `5e-6` and `5e-10` benchmark guards.

## Barnes--Hut viewer Stage 4A validation

The visualization-model suite exercises the real controller selection path for
the four CPU combinations, retains direct as the default, and compares the
controller with an independently advanced matching headless backend after each
of seven consecutive steps. Each boundary checks exact state/time/body identity,
conservation diagnostics, effective configuration, and deterministic tree/force
statistics under the established CPU contract. It
also covers invalid/non-finite theta,
state/time preservation across solver changes, scenario replacement and reset,
effective statistics, explicit CUDA/Barnes--Hut rejection, and a constructed
second-force singularity that must pause without publishing partial state.

Existing counterfactual-worker tests remain the concurrency boundary: they
cover immutable snapshots, scenario replacement during a run, cancellation,
and non-aliasing of the live controller. The reset test starts a real worker,
accepts its real generation and progress through the production gate, invokes
the production reset invalidation helper, verifies every publication-derived
cache is empty, rejects both saved progress and late terminal completion, and
admits a fresh post-reset job.
Stage 4A introduces no additional simulation worker and does not claim
Barnes--Hut counterfactual execution.

Final local acceptance passed canonical CPU `18/18` and OpenMP `22/22` Release
CTest runs, focused visualization and worker tests, viewer builds,
repository/dependency checks, and an independent source-level gate on
2026-10-14. The prior real OpenGL launch is not a substitute for subjective
visual inspection. CUDA/Barnes--Hut build and runtime validation remain an
RTX/NVIDIA gate.

## Barnes--Hut counterfactual Stage 4B validation

Implementation `2ec090d` passed the final focused independent source-level gate
and canonical Release CPU `18/18` and OpenMP `22/22` CTest suites on
2026-10-14. This is a local CPU/OpenMP result; CUDA compilation/runtime and
human visual approval are not claimed.

CPU acceptance must retain bit-for-bit default Direct behavior for the
deterministic fixtures and execute genuine paired Sequential/Barnes--Hut and
OpenMP/Barnes--Hut advancement. Tests inspect every retained time boundary,
stable-ID compaction, matched timestamps, full/removed body counts, immutable
theta/resource metadata, and separate per-branch force/tree statistics.
Sequential and OpenMP Barnes--Hut must agree exactly under the established
shared-tree/per-target contract; nondeterministic timing fields are excluded.

An accuracy-first and a looser theta are each compared against a matched Direct
reference under declared bounded fixture tolerances. Passing those fixtures is
not a universal error guarantee. On the eight-step legacy fixture (six-hour
steps, Jupiter removed), the current Release run measured maximum final-position
differences across both branches of `0.351136 m` at theta `0.25` and `52.3632 m`
at theta `0.8`; the bounded regression gates are `10 m` and `1000 m`,
respectively. Resource tests cover invalid node/memory policies, traversal-work
exhaustion, nonrepresentable geometry, and rejection
when removal would leave fewer than two gravitating bodies. Deterministic
host-side injection after a successful real force evaluation covers the full
baseline/counterfactual by old-position/new-position matrix under Sequential
and OpenMP. Every case begins after a committed paired batch and compares all
states, samples, summaries, conservation values, warnings, counters, and prior
branch statistics exactly. Injection is available only in the separately
compiled `nbody_experiments_test_hooks` target. Compile-time assertions verify
the test executable sees the matching hook-enabled class layout and the
production-linked worker sees no hook macro or callable CPU hook API; link and
symbol inspection verify the viewer/worker use only `nbody_experiments`.
A natural traversal-budget failure verifies typed
context through the worker publication. Worker tests execute Barnes--Hut through
the real publication path, retain captured solver/theta after caller edits,
cancel at paired boundaries, exercise reset/stale-generation rejection, and run
the OpenMP Barnes--Hut path where compiled. A saturated warning buffer also
proves that the production approximation-notice decision uses immutable
accepted-job metadata and that Direct, stale, reset, and mismatched-composition
publications do not display it.

CUDA/Barnes--Hut counterfactual selection must fail explicitly. No local CUDA
compile/runtime result is inferred from CPU acceptance, and no skipped or
mocked test satisfies the future NVIDIA gate.

## Large-N Laboratory L1 validation

Generator tests require exact same-toolchain reproduction, a fixed regression
checksum, stable unique ordered names, finite positive masses, finite state,
noncoincident normal fixtures, distinct seeds, and quantitative uniform/
clustered/planar geometry checks. Missing seed, invalid counts/scales, and
explicit coincident adversarial states must fail.

Preflight tests cover all supported requested sizes through 100K for bounded
Barnes--Hut admission, rejection of unbounded 100K Direct work, missing Direct
reference metadata above 4096, invalid Barnes--Hut policy, unavailable CUDA and
CPU-only OpenMP, actual OpenMP team metadata, one-byte memory rejection, and
integer overflow. They also assert every phase estimate, the maximum-phase
high-water relationship, and a budget deliberately placed between the old
underestimate and corrected retained-repetition peak. Admission of a requested
size is not execution evidence or a process-RSS guarantee.

Numerical tests cover cancellation-aware maximum/RMS/percentile error
statistics, denominator floors, relative-division overflow, finite inputs that
produce nonrepresentable diagnostics, exact Direct self-reference, bounded
Barnes--Hut execution, solver statistics, timing-field nonnegativity, canonical
CSV fields, and transactional clearing after a natural solver failure. A test
execution seam deterministically injects an over-budget value only at the final
scientific boundary, proving late work cannot be published without using
wall-clock sleeps.

Python negative controls exercise the production runner/verifier. They cover
raw tampering; negative timing, error, drift, and resource fields even after
coherent rehashing; zero denominator floors and force durations; wrong matrix
policy; altered summaries; missing/corrupt authoritative runs; incomplete and
unreferenced run directories; stranded legacy backups; and simulated timeout,
malformed output, and provenance failure. Spawned-process tests prove two
publishers serialize and a contender cannot steal a live Windows byte lock.
Checkpoint injection covers pre-install, post-install, and atomic-pointer-
replacement failures, plus explicit orphan recovery and byte-exact flat-layout
migration.

Additional recovery controls corrupt or omit pointer fields, mismatch pointer
run IDs and manifest hashes, recover a real legacy backup with no primary,
append recovered history without changing an existing current run, reject
multiple and corrupt backups, resume a post-install interrupted adoption,
prove repeated recovery is idempotent, and prove ordinary pilot invocation
refuses an unadopted backup before starting measurement.

The controlled pilot gate is limited to 256/1024/4096, three distributions,
Sequential/OpenMP-4, Direct/Barnes--Hut, theta `0.25/0.5/0.8`, one warm-up,
two retained repetitions, one step, and explicit memory/direct-work/wall-time
bounds. Evidence verification must reconstruct all 72 summary medians from 144
raw rows and confirm Release OpenMP build provenance, actual thread metadata,
finite and semantically nonnegative scientific fields, the immutable run
manifest and atomic current/history pointer, exact source/build/
binary custody, and non-Stage-5 labeling. This pilot is exploratory;
no accuracy threshold, stable crossover, 50K/100K feasibility, GPU result, or
graphical FPS claim follows from it.
