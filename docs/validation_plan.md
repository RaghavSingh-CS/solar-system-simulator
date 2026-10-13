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

## Sequential Barnes--Hut Stage 1 validation

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
`sequential + barnes-hut` composition; another requires an unsupported backend
pairing to fail so fallback cannot be silent.

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
