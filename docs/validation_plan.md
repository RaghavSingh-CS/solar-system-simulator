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

Initial thresholds for the analytic two-body CI test are implementation guards, not scientific claims: maximum relative energy drift `<= 5e-6`, relative separation error `<= 5e-5`, normalized angular-momentum drift `<= 5e-10`, and center-of-mass drift `<= 1e-9` of initial separation over five periods. Scenario-specific research thresholds must be fixed before collecting benchmark outcomes.

## Real Solar System validation

- Obtain an auditable JPL Horizons barycentric state at one declared epoch.
- Validate parser/unit/frame behavior independently.
- Compare short-horizon output at later Horizons epochs with position/velocity errors per body.
- Record omitted physics (relativity, oblateness, minor bodies) and do not attribute model error to numerical integration.
- Use conservation diagnostics as internal checks, not as proof of ephemeris accuracy.

## Reproducibility

Every result row must carry commit, build type, compiler/options, backend, hardware, scenario hash, seed, precision, N, dt, steps, repetition, and warm-up policy. Failed runs remain in raw data with a reason.
