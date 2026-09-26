# Sequential conservation characterization

## Research question

How do the existing fixed-step Velocity Verlet implementation's energy and
angular-momentum conservation errors vary with timestep and simulated duration
for the repository's two deterministic reference scenarios?

This is a bounded numerical-method experiment, separate from the parallel
backend studies. It does not modify the integrator or force law, compare
backends, benchmark performance, implement a selector, or constitute Stage 5.

## Method

The study ran the existing sequential backend in binary64 at source commit
`6d9a162`. It crossed five timestep multipliers (`0.25, 0.5, 1, 2, 4`) with
three duration multipliers (`1, 10, 100`) for each of two scenarios, for 30
configurations total:

| Scenario | Base timestep | Base duration | Meaning |
|---|---:|---:|---|
| Analytic circular two-body | 6080.185062019976 s | 24320740.248079903 s | 4000 steps per orbit; one-orbit base duration |
| Seeded 32-body, seed 24301 | 10 s | 2560 s | Existing deterministic validation scenario; 256-step base duration |

The base durations are exactly divisible by every tested timestep. The harness
records the requested duration separately from the floating-point accumulated
state time. The largest absolute accumulation discrepancy was about `0.0755 s`
over a requested duration as long as `2.4320740248e9 s`; this is recorded as
bookkeeping error, not physical drift.

Diagnostics were sampled at the initial state and after every completed step.
For total energy `E` and angular-momentum vector `L`, the recorded relative
drifts are:

```text
energy = abs(E - E0) / max(abs(E), abs(E0), 1 joule)
angular momentum = norm(L - L0) / max(norm(L0), 1 kg m^2/s)
```

Both the maximum over the trajectory and the final value are retained. The
vector definition detects directional as well as magnitude changes. The
existing bounds remained frozen at `5e-6` for energy and `5e-10` for angular
momentum; no result was used to tune them.

## Results

All 30 configurations passed both frozen bounds. The table reports the largest
trajectory drift over the three tested durations at each timestep.

| Scenario | dt multiplier | dt (s) | Maximum energy drift | Maximum angular drift |
|---|---:|---:|---:|---:|
| Analytic two-body | 0.25 | 1520.0463 | 2.475e-13 | 1.215e-13 |
| Analytic two-body | 0.5 | 3040.0925 | 2.606e-13 | 8.540e-14 |
| Analytic two-body | 1 | 6080.1851 | 1.541e-12 | 8.833e-14 |
| Analytic two-body | 2 | 12160.3701 | 2.439e-11 | 2.117e-14 |
| Analytic two-body | 4 | 24320.7402 | 3.896e-10 | 1.743e-14 |
| Seeded 32-body | 0.25 | 2.5 | 4.753e-15 | 1.932e-14 |
| Seeded 32-body | 0.5 | 5 | 2.031e-14 | 1.313e-14 |
| Seeded 32-body | 1 | 10 | 5.834e-14 | 1.092e-14 |
| Seeded 32-body | 2 | 20 | 2.301e-13 | 1.330e-14 |
| Seeded 32-body | 4 | 40 | 9.276e-13 | 7.218e-15 |

The worst observed energy drift was `3.896e-10`, about 12,832 times below its
bound. The worst angular-momentum drift was `1.215e-13`, about 4,114 times below
its bound.

For the analytic orbit, maximum energy drift at the coarser timesteps is nearly
unchanged across 1, 10, and 100 orbits while final drift is often much smaller
than the maximum. That pattern is consistent with bounded oscillatory energy
error over this window. From multipliers 1 to 2 to 4, the maximum grows by
approximately 16 times per timestep doubling in this particular circular
setup. This is empirical behavior, not a formal convergence-order estimate.
At the finest timesteps, longer runs execute many more operations and the
observed floor grows with duration.

For the seeded system at the 100-duration horizon, maximum energy drift grows
from `5.834e-14` to `2.301e-13` to `9.276e-13` for timestep multipliers 1, 2,
and 4, approximately fourfold per doubling. At multiplier 4, it also grows
from `7.346e-15` through `7.044e-14` to `9.276e-13` as duration changes from 1
through 10 to 100. The data establish duration-dependent accumulation in this
window but are insufficient to label it secular or infer a theoretical order.

Angular drift remains at roughly `1e-13` or below and is non-monotonic in
timestep. The largest analytic value occurs in the finest, longest run, which
also has the most steps. The evidence is compatible with a roundoff-dominated
floor, but the experiment does not isolate roundoff as a cause.

## Reproducibility and evidence

The runner verified row count and uniqueness, finite numerical fields, exact
requested durations, pass statuses, and a repeated seeded configuration. For
that repeat, all scientific fields were exactly equal; only experiment ID,
execution order, and incidental wall time were excluded.

Evidence is retained under `data/experiments/conservation/`:

- `raw_results.csv`: all 30 raw rows;
- `summary.json`: per-scenario and per-timestep maxima;
- `environment.json`: protocol, source state, platform/tool versions, and
  repeatability result;
- `README.md`: provenance and file hashes.

Reproduction from a configured Release sequential build:

```powershell
python -B scripts/run_conservation_study.py --executable build-sequential/nbody_conservation_probe.exe --output-dir data/experiments/conservation --experiment-id conservation-study-2026-09-26 --cmake-executable .tools/cmake-4.4.3-windows-x86_64/bin/cmake.exe --compiler-executable C:/msys64/ucrt64/bin/c++.exe
```

## Limits and implications

The analytic case is an ideal two-body orbit, while the seeded case is a
synthetic validation workload rather than a Solar System ephemeris. The study
uses only two deterministic initial conditions, fixed steps, double precision,
and the current direct Newtonian model. It does not test close encounters,
realistic planetary horizons, alternative integrators, or backend-dependent
rounding. Runtime was captured only to audit execution and must not be used as
performance or crossover evidence.

Within this tested domain, the frozen conservation bounds are validity guards
with very large margins rather than discriminating accuracy thresholds. A
future same-machine backend-selector study should retain them as hard gates,
but should not interpret passing them as evidence that a configuration is
equally accurate or scientifically adequate for broader scenarios. Broader
physical cases and horizons are needed before changing or tightening policy.
