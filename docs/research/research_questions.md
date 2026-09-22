# Research questions

## Primary question

On a single heterogeneous workstation, can a lightweight calibrated policy select the lowest end-to-end runtime among sequential C++, OpenMP, and CUDA direct N-body backends while satisfying predeclared energy- and angular-momentum-conservation bounds?

## Secondary questions

1. How do workload size `(N)`, simulated step count, floating-point precision, and transfer amortization affect the measured crossover among the three backends?
2. How close is the policy to an offline oracle on held-out configurations, and how often does it choose an invalid configuration?
3. Does an accuracy gate change decisions relative to a size-only threshold?
4. How stable are calibration and decisions across repeated runs, compiler settings, and cold/warm GPU conditions on the same machine?
5. How closely do all backends agree with the deterministic sequential reference over a fixed validation horizon?

## Operational definitions

- **End-to-end runtime:** initialization, required host/device transfers, integration, and final diagnostics; file output and interactive rendering are excluded and timed separately.
- **Valid configuration:** passes scenario-specific relative energy drift, angular-momentum drift, center-of-mass drift, and cross-backend state-error thresholds declared before benchmark collection.
- **Workload:** at minimum `(scenario, N, steps, dt, precision, backend, thread count)`.
- **Oracle:** fastest valid measured backend/configuration for that exact workload on the same machine.
- **Selector regret:** `(selected runtime - oracle runtime) / oracle runtime`.

## Experimental falsifiers

- H1 fails if one backend wins all valid tested workloads within uncertainty.
- H2 fails if held-out oracle agreement/regret misses its preregistered targets.
- H3 fails if the accuracy gate never changes a decision.
- H4 cannot be tested until both environments execute the same benchmark matrix; cross-machine raw times must never be pooled.
