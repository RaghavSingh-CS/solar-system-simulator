# Barnes--Hut solver (Sequential Stage 1, OpenMP Stage 2, CUDA Stages 3A--3B)

## Scope and selection contract

Stage 1 added the deterministic Sequential Barnes--Hut reference. Stage 2
reuses that exact tree and traversal for an OpenMP target-parallel backend,
without changing the direct Sequential, OpenMP, CUDA, or MPI implementations.
Direct force remains the default. The headless runner treats execution backend
and force solver as separate choices:

```powershell
.\build\cpu-release\nbody_headless.exe --backend sequential --solver barnes-hut `
  --theta 0.25 --steps 5000 --sample-every 50 --output orbit-bh.csv

.\build\openmp-release\nbody_headless.exe --backend openmp --solver barnes-hut `
  --threads 8 --theta 0.25 --steps 5000 --sample-every 50 --output orbit-omp-bh.csv

.\build\cuda-release\nbody_headless.exe --backend cuda --solver barnes-hut `
  --cuda-device 0 --theta 0.25 --steps 5000 --sample-every 50 --output orbit-cuda-bh.csv
```

`sequential + barnes-hut`, `openmp + barnes-hut`, and optional
`cuda + barnes-hut` are implemented. CPU-only builds reject unavailable OpenMP
or CUDA choices explicitly. No unsupported selection falls back to direct force
or Sequential while reporting another backend.

## OpenMP shared-tree architecture

Each OpenMP force call validates the state, allocates bounded per-thread
statistics, and constructs one tree using the Sequential builder. The completed
tree, body state, permutation, and rank table are then read-only. A
`schedule(static)` loop assigns target indices; each iteration owns its
acceleration accumulator and output slot and traverses children/leaves in the
same order as Sequential. Traversal recursion uses each worker's own bounded
call stack. There is no concurrent topology mutation, shared floating-point
accumulation or simultaneous tree build. Calls made from an already active
OpenMP parallel region are rejected before solver allocation/tree construction;
the backend never silently serializes or creates a nested team.

Requested thread count is explicit (`1..256`) and must be honored by the
runtime. Per-thread counters occupy cache-line-aligned slots and are merged in
ascending worker order after the region. The implementation uses either a
checked proof that the worst-case work bound fits the configured traversal
budget or a shared atomic budget on the slow admission path. Exceptions are
caught inside the region. The strict-budget path uses a compare/exchange
reservation that never increments beyond the configured limit; exhaustion
stops new targets promptly and produces one normalized deterministic error.
It performs no diagnostic replay or other uncharged traversal. Other captured
failures retain the lowest observed target index where feasible. No failed
call publishes output or candidate statistics. A separate attempt-work value
exists only to audit the hard cap and is not a partially committed result.
Successful statistics are transactional and identify the effective worker
count.

The backend object is single-owner because it publishes mutable last-success
statistics. Independent concurrent simulations must use distinct backend
instances. A checked size-based lower-bound preflight precedes allocation;
after allocation, admission charges the actual capacities of the output,
permutation, scratch, inverse-rank, node, and padded per-thread-counter
vectors before force traversal. This is not an allocator-exact resident-set
guarantee: allocator metadata, OpenMP runtime storage, and native worker stacks
are excluded. Traversal has no heap scratch buffer; its per-worker recursion
stack is bounded by the validated maximum depth. High thread counts can still
oversubscribe CPUs and consume substantial unaccounted native/runtime stack
storage, so they require an explicitly provisioned environment.

## Tree and force policy

The solver builds a three-dimensional cubic octree for every force evaluation.
Nodes live in a contiguous vector and refer to children by index, so vector
growth cannot invalidate tree links. A stable octant partition retains a
deterministic body permutation; an inverse-rank table makes target membership
explicit after body indices shift within the tree. Octant boundary ties use
the positive half in each axis, and children are always constructed/traversed
in octant order `0..7`.

Each node stores a compensated total-mass sum and an online weighted monopole
center of mass that avoids forming overflow-prone `mass * position` products.
Leaves store a range
in the shared body-index permutation rather than owning pointers. Subdivision
ends at leaf occupancy, maximum depth, minimum cell size, or loss of a
representable smaller cell. A termination leaf may therefore contain more
than the preferred occupancy and evaluates those bodies directly. This is the
explicit protection against indefinitely subdividing coincident or extremely
close coordinates; exact coincidences are still rejected because the existing
unsoftened point-mass law is singular.

For a target at distance `d` from a node center of mass and node side length
`s`, a node is accepted when `s / d < theta`. A node whose permutation range
contains the target is never accepted as an aggregate, so self force is never
included. Opened leaves exclude the target index explicitly. Only the monopole
term is implemented; quadrupoles and higher moments are out of scope.

The force law remains

`a_i = G sum(j != i) m_j (r_j-r_i) / |r_j-r_i|^3`

in binary64 with no softening. Stable three-component norms avoid overflowing
ordinary squared-distance intermediates. The ordinary physical range uses the
direct algebraic form; an exponent-scaled component calculation handles
otherwise representable extreme results without assuming extended `long
double` range. Non-finite inputs, unrepresentable
coordinate spans, aggregate masses, accelerations, and exact coincident
positions fail explicitly instead of returning a silent approximation.

## Defaults and resource policy

| Parameter | Default | Role |
|---|---:|---|
| `theta` | `0.25` | Accuracy-first opening threshold, valid range `(0, 2]` |
| leaf capacity | `8` | Preferred direct bucket occupancy |
| maximum depth | `64` | Hard recursion/degeneracy bound; accepted range `1..128` |
| minimum cell size | `1e-6 m` | Scale floor before a direct termination leaf |
| maximum nodes | `1,000,000` | Hard node-count budget |
| maximum traversal work | `1,000,000,000` | Hard visited-node plus direct-interaction budget |
| memory budget | `256 MiB` | Index/output, node-arena, and OpenMP counter-array bound |

All size calculations use checked `size_t` arithmetic. The node arena grows
only to the lesser of the node and memory budgets. Build and traversal recursion
cannot exceed the validated maximum depth. Reported working bytes cover actual
C++ vector capacities for all solver-owned arrays and reserved nodes. The
public fixed-memory helper is only the checked size-based preflight lower
bound. Neither value claims allocator metadata, native OpenMP stacks, runtime
allocations, or process resident memory.

One backend instance is single-owner because it publishes mutable last-call
statistics. Independent callers use independent instances; the solver does not
share tree storage or state across calls.

Expected balanced-tree cost is `O(N log N)` force work and `O(N + nodes)`
storage. The policy makes no worst-case `O(N log N)` promise: stringent theta,
pathological geometry, or large termination buckets can approach direct
`O(N^2)` work. The solver reports node/leaf/depth, visited/accepted/direct
interaction counts, build time, force time, and accounted working memory after
each successful evaluation.

## Numerical validation and theta baseline

Focused tests cover empty, singleton, analytic two-body, asymmetric seeded,
planar, collinear, clustered, translated, scaled, extremely large/small, near-
coincident, coincident, and highly unbalanced inputs. They verify deterministic
tree/traversal results, resource failures, self exclusion, theta approaching
direct force, absolute/RMS/maximum/95th-percentile force error, and short-run
trajectory/conservation behavior through the unchanged Velocity Verlet API.

The default `theta=0.25` is the Stage 1 accuracy-first baseline. In the retained
4096-body exploratory matrix its 95th-percentile relative acceleration error
was `2.81e-4`--`3.84e-3` across the three tested distributions. `theta=0.5`
is a balanced exploratory setting, and `theta=0.8` is performance-oriented;
neither is a universal scientific accuracy setting. Near-zero reference forces
make per-body relative error ill-conditioned, so absolute, RMS, percentile,
and maximum errors are retained together. Relative values use
`max(|a_direct|, 1e-30 m/s^2)` as their denominator; that fixed reporting floor
prevents division by zero but is not a scientific significance threshold.

The short deterministic 48-body characterization covers the same 3000-second
horizon at both 25-second and 12.5-second timesteps. In the 25-second run,
against direct integration, the observed maximum position separation was below
`1e-3 m` for `theta=0.25` and about `0.218 m` for `theta=0.8`; relative energy
drift was about `1.99e-15` and `3.49e-8`, respectively. The tests also bound
angular-momentum drift at both theta/timestep combinations. These are bounded
regression observations for a synthetic fixture, not general accuracy or
long-horizon orbital guarantees. Approximate one-sided tree forces need not
obey exact pairwise momentum symmetry.

## Exploratory performance boundary

Raw timings and errors are retained under
`data/benchmarks/barnes_hut_stage1/`. On this Ryzen 7 5700U Release build, the
accuracy-first `theta=0.25` did not beat direct force by 4096 bodies for the
uniform or clustered inputs; the planar input crossed earlier. At `theta=0.5`,
uniform and clustered 4096-body cases were still slower than direct (`0.76x`
and `0.78x` direct/Barnes--Hut speed ratios), while the planar case was faster.
This distribution and theta sensitivity is why no
automatic selection threshold is encoded.

The measurements use three repetitions and uncontrolled host conditions. They
are sufficient to guide the next experiment design, not to establish a
production crossover or modify frozen Stage 5 policy/evidence.

The separate Stage 2 matrix under
`data/benchmarks/barnes_hut_openmp_stage2/` contains 216 measurements at
`N=256,1024,4096`, theta `0.25,0.5,0.8`, and 1/2/4/8 effective threads for the
same three distribution classes. At 4096 bodies, four-thread complete-force
speedup over Sequential Barnes--Hut ranged from `3.25x` to `3.98x`, and
eight-thread speedup ranged from `4.28x` to `7.51x`. At theta `0.25`, the
eight-thread values were `7.02x` uniform, `7.51x` clustered, and `7.33x`
planar. Against eight-thread direct force, however, theta `0.25` remained
slower on uniform (`0.26x`) and clustered (`0.35x`) inputs and faster only on
the planar input (`1.24x`). Tree build remains sequential and is included in
complete-force wall time.

The Stage 2 raw CSV SHA-256 is
`11fb34ae4d3b4f87687cf177f1668139d7f5bad57ec944f78b316048f2f96760`.
The deterministic `source_manifest.sha256` (SHA-256
`853de3784315ded6c90a046144d882ae5806ad43df553fdb04683276f295b255`)
binds the raw matrix to the exact measured solver/header, benchmark harness,
reference/scenario sources, CMake wiring, preset, raw data, and deterministic
build-provenance record used for measurement. The source implementation and
verification tooling are committed as `80d83f3`; the evidence and checkout
custody rules are committed as `c572be1`. `build_provenance.json` (SHA-256
`3b4f33cc83ff6f25b2954d1722d259c5e2fa0c3211c165e8eaf8c9ee19cab2c2`)
records the actual compiler and version, CMake/Ninja versions, Release/OpenMP
flags and cache values, runtime libraries, selected compiler commands, and
benchmark linker command. This provenance identifies the build; it does not
guarantee repeatable machine-level performance.
Only two repetitions were retained under uncontrolled desktop conditions, so
these measurements do not establish linear scaling, a production crossover,
or 50K--100K feasibility. They are not graphical FPS and are not Stage 5
evidence.

Stage 2 focused tests require exact IEEE-754 representation equality for force,
trajectory, and diagnostic
agreement between Sequential and OpenMP Barnes--Hut across 1, 2, 4, and 8
threads where the runtime can supply them. Uniform, clustered, planar,
boundary, mass-ratio, near-coincident, extreme-binary64, maximum-depth, and
resource-failure fixtures are included. The comparator uses `memcpy` into a
fixed-width integer and a signed-zero negative control proves that it is not
ordinary floating equality. Repeated results and structural/work counters are
deterministic; timing fields are intentionally excluded from cross-call
equivalence assertions.
CPU-only builds exercise both the public API and headless rejection path.
Final canonical Release validation passed `16/16` CPU CTests and `20/20`
OpenMP CTests. The latter includes the focused OpenMP Barnes--Hut suite and
the separate test-only runtime-schedule variant, plus both OpenMP headless
solver paths. Production remains explicitly `schedule(static)`; the test-only
variant compiles the same implementation with `schedule(runtime)` and proves
actual static, dynamic, and guided scheduling. The full suites also cover the legacy Solar
scenario, 60-body catalog/dataset/snapshot integrity, existing direct-force
backends, Stage 5 tooling, visualization/counterfactual regressions, repository
layout, and pinned dependency cache. CUDA compilation/runtime was not attempted
or claimed on this AMD-only machine. The final independent source-custody gate
passed on 2026-10-13, including an actual Git-materialized
`core.autocrlf=true` checkout with matching source and evidence bytes.

## CUDA Stage 3A source-reviewed architecture

Stage 3A adds a standalone `CudaBarnesHutForceBackend` source implementation;
it does not connect Barnes--Hut to Velocity Verlet, the headless solver
selector, viewer, or counterfactual engine. The existing CPU builder remains
the only tree constructor. `build_barnes_hut_packed_tree()` converts its
completed deterministic tree to fixed-width, pointer-free nodes containing
cell center/side bounds, monopole mass and center, eight child indices, leaf
permutation ranges, and an inverse-rank map. Body ordering remains the caller's
original index ordering, so higher-layer stable IDs retain their existing
mapping. Packing has a separate checked host-memory budget and rejects any body
or node count outside its 32-bit device-index representation.

The CUDA kernel consumes that immutable snapshot and assigns one target body to
each logical work item. It uses double precision, the same gravitational
constant supplied by the caller, zero softening, compensated accumulation, the
strict `side / distance < theta` opening rule, and no aggregate approximation
for nodes whose permutation range contains the target. Leaf evaluation excludes
the target's original body index. Children are pushed in reverse slot order so
the per-target depth-first accumulation order matches the CPU traversal. CPU
and GPU compilers can nevertheless differ through contraction/FMA and other
platform details; future validation uses fixed mixed tolerances and reports
absolute, RMS, maximum-relative, and percentile errors rather than requiring
bit identity.

Each work item has a compiled 1024-entry (`4096` byte) local index stack plus a
configurable logical limit. Stack overflow is an explicit device failure. A
global 64-bit compare/exchange reservation enforces the traversal-work budget
without overshoot. Malformed packed nodes, singular/overflowed geometry,
nonrepresentable acceleration, stack exhaustion, and work exhaustion are
reported through a bounded device record. Host code synchronizes before reading
that record and never publishes partial force output. There is no CPU or direct
fallback.

The record uses separate atomic claim and publication words. The winning
thread writes code and target/node metadata, applies a device fence, and only
then atomically publishes completion. Concurrent threads perform atomic loads
of the publication word and never read partially initialized metadata. The
host rejects incomplete or invalid records after stream synchronization.

Persistent device storage and pinned host staging have separate checked
budgets. Accounting includes masses, positions, packed nodes, permutation,
inverse rank, accelerations, error record, and work counter; host accounting
also includes the temporary packed snapshot. It excludes CUDA allocator/context
metadata and compiler-managed backing for the bounded per-thread local stack.
All asynchronous operations use one owned stream. If device selection or
synchronization cannot prove quiescence, the executor becomes terminal and its
stream-referenced allocations are quarantined until process/context teardown;
no automatic device reset is issued.

Live capacity replacement is stricter than destructor cleanup. After the
stream is quiescent, each `cudaFree`/`cudaFreeHost` result is checked. A failed
release leaves that and all not-yet-released handles plus the prior accounting
intact, marks the executor terminal, and prevents replacement allocation.
Partial allocation failure is also terminal and uses the checked cleanup path.
The destructor remains noexcept and either releases after proven quiescence or
conservatively retains storage until context/process teardown. A deterministic
host-only fault-injection mode exercises the live-release terminal contract;
it is not evidence of CUDA device-failure recovery.

This CPU-built-tree design minimizes scientific duplication but transfers a new
tree every force call. Its build and transfer overhead may dominate, especially
for 50K+ bodies. No CUDA compilation, kernel execution, numerical-equivalence,
throughput, crossover, or FPS result exists from the AMD-only development host.
Implementation `5e1864e` passed independent adversarial source review. CUDA
translation-unit compilation and kernel execution were unavailable on the
AMD-only/no-nvcc host and remain NVIDIA validation gates.

## CUDA Stage 3B transactional integration

Stage 3B introduces `CudaBarnesHutVelocityVerlet` without changing either the
CPU builder or the Stage 3A approximation. For one timestep it validates the
complete input and timestep, copies the caller-owned state, obtains the old
acceleration from Stage 3A, applies the existing `r += v*dt + 0.5*a*dt^2`
drift, invokes Stage 3A again so a new tree is constructed from those new
positions, and applies `v += 0.5*(a_old+a_new)*dt`. Body order, names, masses,
and timestep meaning are unchanged. The candidate is moved into the caller only
after both force calls and all finite-state/resource checks succeed.

Failure records distinguish input validation, first force, drift, second force,
integration post-force validation, kick, and commit. Last-success
statistics are transactional. Invalid inputs and nonterminal tree,
nonrepresentable geometry, singularity, stack, work, or admission failures
allow reuse; a terminal Stage 3A CUDA runtime, synchronization, or release
failure makes the integrator permanently unusable. The same owned stream/buffer
lifetime rules, checked live release, conservative uncertain-quiescence
quarantine, and no device-wide reset policy remain in force.

`maximum_traversal_work` remains a hard per-force-evaluation ceiling. A complete
step admits exactly two such evaluations, so its derived hard ceiling is the
checked value `2 * maximum_traversal_work`; successful statistics report each
call and their checked sum. The separate integration-host budget includes the
candidate body vector, dynamic body-name capacity, and both returned
acceleration-vector capacities. Stage 3A separately admits persistent device,
pinned-host, and temporary packed-tree staging. Allocator metadata, native host
stacks, CUDA context storage, and compiler-managed per-thread local-stack
backing are not claimed.

The CUDA-only integration suite specifies zero/one/two-body, legacy Solar,
JPL-derived subset, uniform/clustered/planar, theta, multistep, conservation,
Sequential Barnes--Hut, matched Sequential-Barnes--Hut/direct versus CUDA-
Barnes--Hut/direct trajectory characterization, invalid timestep, naturally
reached drift and kick overflow, nonrepresentable force geometry, first- and
second-force singularities, stack/work/memory exhaustion, rollback, terminal
reuse rejection, and repeated lifetime cases. Post-force defensive resource
checks use a separately compiled test-only library variant; the production
library does not contain those hooks. Strict device enumeration cannot convert
a CUDA error into skip 77. This suite has not compiled or run on the AMD-only/
no-nvcc host. Implementation `2cc2cb0` passed independent source-level review;
Stage 3B remains GPU-unverified.

## Remaining stages

- Compile and adversarially validate the Stage 3A force and Stage 3B
  transactional integration paths on NVIDIA hardware before higher-layer use.
- Evaluate GPU-resident/parallel tree construction separately; do not present
  the Stage 3A CPU build plus transfer as a fully GPU-resident solver.
- Run controlled broader large-N performance and accuracy evaluation before
  changing viewer workload gates or promising interactive 50K--100K physics.
- Evaluate higher multipoles only as a separately justified scientific stage.

The JPL catalog, legacy nine-body initial state, counterfactual engine,
direct-force kernels, Velocity Verlet equations, and frozen Stage 5 research
artifacts are outside Stages 1--3B and unchanged.

## Stage 4A viewer integration

The optional OpenGL/ImGui viewer now exposes the existing solver as an
independent selection for Sequential and OpenMP execution. Direct remains the
default. Barnes--Hut starts from the accuracy-first provisional `theta=0.25`
configuration and retains the reviewed defaults: leaf capacity 8, depth 64,
minimum cell `1e-6 m`, one million nodes, one billion traversal work units,
and 256 MiB of solver working memory. The panel distinguishes draft theta from
the effective applied value and reports the current backend/solver, resource
bounds, and most recent successful tree/force statistics.

Viewer integration does not change the tree, opening criterion, force law,
Velocity Verlet equations, body order, timestep, or scenario state. CPU frame
batches are value-transactional, so a failed later force evaluation cannot
publish a partially advanced state. Failure diagnostics retain the original
message and identify the effective backend, solver, batch step, and reliable
source-level phase (`old-force`, `position-drift`, `new-force`,
`velocity-kick`, or commit; CUDA remains one `cuda-batch` phase). Existing
synthetic workload gates remain
in force in addition to Barnes--Hut's own hard resource limits; Stage 4A does
not claim a new graphical crossover or interactive 50K--100K capability.

CUDA/direct is unchanged. Although the source-reviewed Stage 3B headless
executor exists, CUDA/Barnes--Hut is intentionally unavailable in the viewer
until compilation and execution on NVIDIA hardware complete. The separate
counterfactual runner remains direct-force-only in this stage.

Stage 4A implementation `c9e9500` passed independent source-level review and
canonical local CPU/OpenMP validation on 2026-10-14. This status does not imply
CUDA compilation, NVIDIA execution, subjective visual approval, or a large-N
performance result.
