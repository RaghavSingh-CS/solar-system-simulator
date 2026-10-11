# Counterfactual gravitational experiments

## Implemented CPU engine

The counterfactual engine is a headless model component. It has no graphics
dependency and does not retain references to `SimulationController` state. An
`ExperimentSnapshot` copies one physical `SimulationState`, its stable IDs,
parent relationships, sourced radii, scenario/catalog identity, coordinate
frame description, provenance, capture timestamp, fixed timestep, integrator,
and the current scientific horizon policy. Its public API has no mutators and
assignment is disabled.

`capture_legacy_solar_system_snapshot` recognizes the unchanged approximate
nine-body ordering and masses. `capture_astronomical_catalog_snapshot`
recognizes a full or deterministic subset of `jpl-ssd-j2000-60-v1`. Both reject
state/order/mass mismatches instead of guessing an identity. The catalog
adapter retains the original SSB-centered J2000 ICRF geometric coordinates;
the runner never independently recenters either branch.

`CounterfactualExperimentRunner` creates two private branches:

- baseline: every body in the captured physical selection;
- counterfactual: the same ordered state with exactly one curated body erased.

Every survivor is checked for exact initial state equality and mapped by stable
ID after vector compaction. Initial support is restricted to one planet or
dwarf-planet candidate marked by the snapshot adapter. Sun removal, multiple
removal, and massless stand-ins are rejected. Removal is physical: the erased
body is absent from both direct-force evaluations in the counterfactual branch.

Sequential and optional OpenMP modes call the existing `ForceBackend` and
`velocity_verlet_step` APIs. Both branches use the same fixed timestep and
Velocity Verlet configuration. OpenMP requests are rejected in CPU-only builds;
there is no silent backend fallback. Each bounded batch advances private copies,
checks both branches, and publishes states, metrics, histories, and completed
step count only after the pair succeeds. A failed batch leaves the last
published pair, conservation diagnostics, warnings, dropped-warning count, and
threshold histories unchanged. Terminal failure metadata is reported through a
separate fixed-capacity record (512 message bytes plus truncation status), so a
failure remains visible even when the committed warning buffer is full and
failure reporting itself does not allocate or evict prior warnings. Pause and
cancellation are honored at these valid
publication boundaries. A runner is deliberately single-thread-owned: callers
must not invoke advancement, `result()`, pause, or cancellation APIs
concurrently. The viewer transfers runner ownership to one persistent worker
and never calls it from the render thread. GUI cancellation is an atomic
request observed at paired batch boundaries. Mutex-protected publications are
immutable to the GUI: running updates contain bounded progress and shared
identity metadata, while one bounded full result is published at completion,
cancellation, or engine failure. Worker exceptions are contained as bounded
terminal worker metadata; window shutdown joins the thread.

## Viewer experiment interface

The graphical panel is available only for the legacy and catalog astronomical
scenarios. It captures a value-owned snapshot from either the scenario's
initial state or the current synchronized controller state. It exposes one
curated non-Sun removal, Sequential/OpenMP selection, horizon, sampling cadence,
absolute and fixed-scale normalized thresholds, persistence, and monitored
stable IDs. Preflight displays the same checked step/sample/direct-work and
conservative retained-memory estimates used for admission. Invalid or excessive
requests cannot start.

The viewer remains responsive while the worker advances. Scenario replacement
requests cancellation and increments an application-owned scenario generation,
including for a replacement having the same stable-ID composition. Progress,
terminal results, and overlays are accepted only when both that scenario
generation and the accepted worker generation match; the full stable-ID list
is an additional composition check. Generation overflow is rejected before a
scenario or job is replaced. Thus late cancellation or completion cannot make
old work current after switching away and back. The panel distinguishes the
normal worker states and labels an invalidated prior-scenario publication as
stale/suppressed without presenting its progress, diagnostics, or overlay as
current.

Every pending, running, cancelled, failed, or completed publication carries one
shared immutable run descriptor. It records both generations, current/initial
snapshot choice, captured absolute time, provenance and coordinate contract,
scenario/dataset identity, removed stable ID, backend/thread request, physical
timestep, horizon and accepted step count, cadence, monitored and hidden-frame
IDs, both thresholds, persistence, and checked work/memory estimates. Mutable
controls are explicitly settings for the next run; historical labels are read
only from this descriptor.

Overlays use only retained completed samples: the baseline path is solid and
the altered path dashed. Both are transformed in the same inertial or common
baseline-parent frame, so branches are never independently recentered. The UI
reports trajectory/velocity/normalized deviations, parent-relative values when
defined, threshold sampling semantics, per-branch conservation, warnings,
dropped-warning counts, and terminal failures. It does not infer escape.

When a local view origin survives but is not monitored, its baseline position
is retained separately at every trajectory sample time. These hidden reference
samples receive no deviation summary or significance metric, are capped and
included in memory admission, and are used as one common historical origin for
both paths. Changing to an origin not retained by that run, removing the
origin, or encountering incomplete reference samples suppresses the overlay
with an explicit frame-unavailable message; there is no fallback to the live
camera origin or another coordinate frame. Overlay membership additionally
requires active-profile, class-filter, and individual-body visibility, none of
which changes gravitational inclusion.

## Metrics and interpretation

Only configured survivor bodies are sampled. The initial state, cadence
samples, and exact horizon endpoint are retained; intermediate physics steps
are not. Every retained body sample contains baseline and counterfactual
positions and velocities, absolute position separation, velocity difference,
and normalized separation. The normalization denominator is fixed at capture:
the initial parent-relative distance when a surviving parent exists, otherwise
the initial common-frame distance, with a configurable positive floor. It does
not shrink or grow with the later result.

Parent-relative separation subtracts the matching branch's parent at the same
sample time. It is explicitly unavailable if the parent was removed or absent
from the physical selection. Summaries retain final and maximum deviations,
maximum times, first threshold crossing, and the time at which the configured
number of consecutive above-threshold samples establishes persistence.
Significance is therefore a policy result based on sampled trajectory
separation, not an instantaneous gravitational influence score.

Experiment elapsed time is derived from the checked integer completed-step
count multiplied by the fixed timestep. It is distinct from each branch's
absolute source timestamp. Branch timestamps must advance monotonically and
remain exactly paired at every accepted step; timestamp stagnation or
non-finite time rolls back the batch.

Each branch receives its own initial/current energy, linear momentum, and
angular momentum diagnostics. Relative energy drift is measured within that
branch. Baseline and counterfactual total energies are intentionally not
subtracted and presented as integration error because they describe different
physical systems. Escape assessment remains explicitly unsupported: this
increment does not implement the required energetic, outward-motion, distance,
and dwell tests.

## Bounds and scientific guards

Construction rejects non-finite or non-positive physical inputs, duplicate or
missing stable-ID mappings, coincident initial positions, invalid cadence,
unsupported horizons, non-integral fixed-step durations, non-curated removal,
and resource-limit violations. The following limits are explicit:

- observation horizon and sample cadence;
- integration steps per publication batch;
- monitored-body count and retained-sample count;
- a checked, conservative memory admission estimate covering retained and
  transactional histories, copied states and identities, maps, summaries,
  strings, and bounded issue storage. This is an admission estimate, not an
  allocator-exact upper bound;
- issue-record and aggregate issue-message byte caps, with an explicit dropped
  issue counter;
- direct-force interaction-work estimate.

Durations and cadences are admitted only when their ratio to the fixed timestep
is integral within a tightly bounded absolute tolerance. Counts are limited to
exact binary64 integer representation (at most 2^53) and the stricter
100,000,000-step operational ceiling. Sample, batch, memory, issue, and direct-
force workload arithmetic uses checked integer operations before allocation or
iteration.

Every integrated step checks finite state, monotonic and exactly paired branch
times, coincident centers, sourced-radius overlap, and a configurable close-
approach envelope. Proximity uses the minimum separation of the relative
straight-line segment joining the pair's pre-step and post-step positions, with
the minimizing fraction clamped to `[0,1]`, independently for each branch.
Relative endpoints and the projection are range-checked and scaled in binary64
before products are formed; no extra `long double` exponent range is assumed.
Finite but unrepresentable relative coordinates or distances invalidate and
roll back the batch instead of being treated as no encounter. This calculation
therefore detects an endpoint-safe crossing, but remains a linearized swept-
step warning: it is not a guarantee of the true minimum distance along the
curved gravitational trajectory. Smaller timesteps or a close-encounter solver
remain necessary. Radius overlap and close approach are scientific model
warnings, not manufactured trajectory significance. Non-finite/coincident
integration, invalid time progression, and excess per-branch conservation
drift invalidate and roll back the batch.
Insufficient timestep convergence is recorded separately from both numerical
invalidity and dynamical significance.

The catalog adapter deliberately retains the provisional 60-second timestep
and a conservative initial one-day experiment-horizon policy. The one-day
policy is a safety bound for this first engine, not a statement that one day is
accurate for every body or removal. Existing convergence work did not establish
general counterfactual accuracy. The point-mass model still omits oblateness,
relativity, tides, non-gravitational comet forces, collision physics, adaptive
steps, and close-encounter regularization. A large separation is not by itself
an ejection, instability, or physically realizable removal event.

## Deferred boundaries

The asynchronous CPU viewer interface is implemented; there is no CUDA paired
executor. CUDA support requires
separate paired-resource ownership, transactional synchronization, numerical
agreement tests, and RTX 4050 validation; it must not silently report CPU work
as CUDA. Barnes--Hut remains mandatory future work for Sequential, OpenMP, and
CUDA with direct-force verification and controlled large-N evaluation.

Frozen Stage 5 benchmark code, protocols, policy, scripts, and evidence are not
dependencies of this engine and remain unchanged.
