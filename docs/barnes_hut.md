# Sequential Barnes--Hut solver (Stage 1)

## Scope and selection contract

Stage 1 adds a deterministic sequential Barnes--Hut force solver without
changing the direct Sequential, OpenMP, CUDA, or MPI implementations. Direct
force remains the default. The headless runner treats execution backend and
force solver as separate choices:

```powershell
.\build\cpu-release\nbody_headless.exe --backend sequential --solver barnes-hut `
  --theta 0.25 --steps 5000 --sample-every 50 --output orbit-bh.csv
```

Only `sequential + barnes-hut` is implemented. Any other backend/solver pairing
is rejected; there is no direct-force fallback while reporting Barnes--Hut.
OpenMP and CUDA Barnes--Hut remain mandatory later stages and are not implied by
the presence of their direct-force backends.

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
| memory budget | `256 MiB` | Index arrays, acceleration output, and node-arena bound |

All size calculations use checked `size_t` arithmetic. The node arena grows
only to the lesser of the node and memory budgets. Build and traversal recursion
cannot exceed the validated maximum depth. Reported working bytes cover the
fixed arrays/output and reserved node capacity; they are a conservative
admission/accounting measure, not a claim about allocator metadata or process
resident memory.

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

Final local Release regression for this candidate passed `16/16` CPU CTests
and `17/17` OpenMP-configuration CTests. The latter retains the existing
OpenMP direct-force tests but still uses only the Sequential Barnes--Hut
implementation. Catalog/dataset/snapshot integrity passed `3/3`; repository
layout, dependency-cache, direct headless, Barnes--Hut headless, and explicit
unsupported-combination checks also passed. Independent source-level
adversarial review passed on 2026-10-13 before commit; it found no blocking
tree, force, integration, evidence-custody, or build-isolation defect.

## Remaining stages

- Parallelize the frozen sequential tree/acceptance semantics with OpenMP and
  validate numerical agreement and deterministic policy boundaries.
- Design a CUDA-resident tree/build/traversal implementation; do not relabel a
  CPU tree plus GPU direct kernel as CUDA Barnes--Hut.
- Run controlled broader large-N performance and accuracy evaluation before
  changing viewer workload gates or promising interactive 50K--100K physics.
- Evaluate higher multipoles only as a separately justified scientific stage.

The JPL catalog, legacy nine-body initial state, visualization, counterfactual
engine, direct-force kernels, Velocity Verlet equations, and frozen Stage 5
research artifacts are outside this implementation and unchanged.
