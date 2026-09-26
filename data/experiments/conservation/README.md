# Sequential conservation-study evidence

This directory contains the accepted evidence for the bounded numerical-method
experiment described in `docs/conservation_study.md`. It is independent of the
backend performance stages and is not Stage 5.

## Protocol and provenance

- Source commit used for measurement: `6d9a162da387ad58de37d44eaf9928bb9160549f`.
- Experiment ID: `conservation-study-2026-09-26`.
- Backend: deterministic sequential implementation only.
- Integrator and precision: the existing fixed-step Velocity Verlet path in
  binary64; neither was changed for this experiment.
- Scenarios: analytic circular two-body and deterministic seeded 32-body
  (`seed=24301`).
- Matrix: timestep multipliers `0.25, 0.5, 1, 2, 4` crossed with duration
  multipliers `1, 10, 100`, producing 30 rows.
- Diagnostics were evaluated initially and after every completed step. Runtime
  is retained only as incidental experiment metadata and is not performance
  evidence.
- Frozen pass bounds were unchanged: relative energy drift `5e-6` and relative
  angular-momentum-vector drift `5e-10`.

`raw_results.csv` preserves all initial/final values, maximum/final drifts,
angular-momentum vectors, requested and accumulated durations, statuses,
incidental wall time, and final-state checksums. `summary.json` contains only
derived per-scenario/per-timestep maxima. `environment.json` records the exact
protocol, tool versions, clean source state, and the exact repeat check.

All 30 configurations passed. The seeded 32-body `dt=10 s`, duration-10
configuration was repeated, and all scientific fields matched exactly; only
runner order and wall-time metadata were excluded from that comparison.

## Integrity

SHA-256 values for the retained files:

| File | SHA-256 |
|---|---|
| `raw_results.csv` | `1D51481186F8BFEADFFD6384FDB00804294EAE3261CC98D26767FBE90BA58EDE` |
| `summary.json` | `189EC74BE03A31D331703208CF5FAC13E79FD509943D1133A9F2D08D21E86391` |
| `environment.json` | `1AD49E26BE586CC91C75CD50BB6590F1A68331F214CC868BDF2AC137F450DA37` |
