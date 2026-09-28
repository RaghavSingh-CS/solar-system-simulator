# Research questions

## Primary question

On one fully specified heterogeneous workstation, can a calibration-only
policy map problem size N to the lowest `selector_steady_state` median among
Sequential, topology-derived OpenMP configurations, and CUDA for the fixed
Stage 5 workload while all results satisfy predeclared energy- and
angular-momentum-conservation bounds?

## Secondary questions

1. How does N affect the measured crossover for the one preregistered binary64, 100-step, seeded synthetic workload?
2. How close is the frozen policy to an offline oracle on the six held-out N values?
3. Do any configurations fail the frozen accuracy gates before performance ranking?
4. How stable are calibration medians and decisions across repeated randomized blocks on the same machine?
5. How closely do all backends agree with the deterministic sequential reference over a fixed validation horizon?

## Operational definitions

- **Stage 5 selector runtime:** one warmed invocation returning a materialized host state. CPU includes state copy plus integration; CUDA includes packing, transfers, kernels, synchronization, materialization, and validation. Process/scenario construction, reusable CUDA allocation/context setup, warm-up, diagnostics, file output, and rendering are excluded.
- **Valid configuration:** passes scenario-specific relative energy drift, angular-momentum drift, center-of-mass drift, and cross-backend state-error thresholds declared before benchmark collection.
- **Stage 5 v2 workload:** scenario, steps, dt, precision, seed, force law, and integrator are fixed; N varies, and backend/thread count identifies a configuration.
- **Oracle:** fastest valid measured backend/configuration for that exact workload on the same machine.
- **Selector multiplicative regret:** `selected median / oracle median`.

Step count, precision, accuracy budget, scenario, seed, compiler portability,
and cold-start behavior remain proposed future selector dimensions. Stage 5
v2 does not claim evidence for them.

## Experimental falsifiers

- H1 fails if one backend wins all valid tested workloads within uncertainty.
- H2 is evaluated descriptively by empirical-fastest agreement and multiplicative regret at the six held-out N values; no post-hoc refit is allowed.
- H3 fails if no configuration is excluded by the frozen accuracy gates.
- H4 remains future work; cross-machine raw times must never be pooled.
