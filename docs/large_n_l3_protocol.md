# Large-N Laboratory Phase L3 protocol

## Status and boundary

Phase L3 is a separate exploratory CPU performance-only study. It does not
override the failed L2 full-characterization continuation gate, authorize a
full Direct reference above 8,192 bodies, or supply scientific accuracy or
conservation evidence. The machine-readable protocol is
`tools/large_n_l3_protocol.json`.

The tooling is committed and independently closeout-reviewed. The first
retained configuration timed out, so the checkpoint is closed without an
authoritative L3 evidence publication or evidence commit.

Candidate checkpoints are 16,384, 32,768, and 50,000 bodies. The first retained
checkpoint executes **only 16,384 bodies**. This task cannot automatically
execute 32,768 or 50,000, and 100,000 is outside the protocol.

## Full study design and first checkpoint

The eventual controlled design covers uniform, clustered, and planar fixtures;
Sequential and OpenMP-4 Barnes--Hut; and theta 0.25, 0.5, and 0.8 with explicit
seeds. The bounded first checkpoint uses three matched backend pairs:

| Distribution | Theta | Backends |
|---|---:|---|
| uniform | 0.25 | Sequential, OpenMP-4 |
| clustered | 0.5 | Sequential, OpenMP-4 |
| planar | 0.8 | Sequential, OpenMP-4 |

The seed is `0x4c33504552460000 + distribution_index*0x100000 + N`, where
distribution indices follow uniform, clustered, planar. Each configuration has
one untimed warm-up, three retained repetitions, and one 10-second Velocity
Verlet step. Generation, complete force, tree construction, traversal, complete
timestep, and total experiment time remain separate. Steps per second is
simulation throughput, not rendering FPS.

Execution classes are distinct. A disposable one-repetition Sequential,
uniform, theta-0.5 calibration (no warm-up) exercised the no-reference path and
was not published or used as a scaling result. The resource limits and the
six-group matrix below were then fixed before the retained exploratory
checkpoint began. That checkpoint is still a pilot; a final controlled L3
measurement set has not been run.

## Resource and stopping policy

The checkpoint retains the reviewed 512 MiB laboratory memory limit,
500,000,000 Direct-interaction limit, 256 MiB Barnes--Hut working-memory limit,
1,000,000-node limit, and 1,000,000,000-visit traversal-work limit. Each
configuration has a 60-second internal wall limit and a 70-second external
subprocess timeout. The external allowance includes launch and orchestration
overhead; it is not additional compute budget.

Any resource rejection, timeout, execution error, or non-finite acceleration,
position, velocity, timing, or derived throughput stops the candidate without
publishing it. Limits are not raised to obtain a result. The task stops after
16,384 regardless of outcome.

A later 32,768 proposal is eligible only if all six groups complete, maximum
median total-time use is at most 80% (48 seconds) of the internal wall budget, maximum
admitted memory is at most 90% of its budget, and maximum within-group force
range-over-median is at most 30%. Eligibility is not authorization: separate
admission and independent review remain mandatory.

## Missing scientific characterization

Every L3 command selects Barnes--Hut and sets `--direct-reference false` with a
zero Direct-reference body limit. No Direct solver fallback, complete Direct
reference, sampled Direct reference, Direct trajectory, or O(N^2) conservation
diagnostic is executed. Acceleration errors, trajectory errors, and per-branch
energy/angular-momentum drift are serialized as empty fields and accompanied
by explicit `unavailable_performance_only` status and reason fields. Numeric
zero is never used as a fabricated scientific result.

Finite Barnes--Hut accelerations and the complete post-step state remain
mandatory. Tree statistics, memory admission, requested/effective/observed
workers, and worker provenance remain measured and verified.

## Evidence custody

L3 uses a separate `data/benchmarks/large_n_l3_pilot/` publication with the
reviewed OS-backed lock, immutable versioned runs, atomic pointer replacement,
strict source/build/artifact manifests, and fail-closed orphan detection. It
does not modify L1 or L2 evidence. Raw CSV is byte-custodied as non-text;
the new manifest-listed L3 sources use narrowly scoped directory-local LF
checkout rules. These rules leave the root `.gitattributes` bytes bound by L1
and L2 unchanged. A fresh Git-materialized checkout with `core.autocrlf=true`
must reproduce every candidate source byte before this tooling is committed.

The checkpoint is exploratory single-machine evidence. Three repetitions are
enough for a bounded variability signal, not a universal scaling conclusion.
Simulation throughput is not graphical FPS, and no CUDA, RTX 4050, GUI, frozen
Stage 5, 32,768, 50,000, or 100,000 execution is included.

## First attempt result

The retained checkpoint started with Sequential, uniform, theta 0.25 at
16,384 bodies. The process exceeded the predeclared 70-second subprocess
timeout before it returned a completed CSV result. The orchestrator stopped
immediately and did not launch the other five configurations. Scratch staging
was removed and no L3 evidence directory, immutable run, or publication pointer
was created. No timing or throughput number is inferred from the timeout.

This is a failed checkpoint, not evidence that Barnes--Hut is numerically
incorrect and not authority to change limits or execute a larger size. Any
retry or revised representative matrix requires a separate reviewed decision.
Future CUDA scaling is outside L3 and requires a separate hardware-specific
protocol and NVIDIA validation environment.
