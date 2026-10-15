# Large-N Laboratory Phase L2 protocol

## Status and purpose

This document pre-registers the controlled Phase L2 scaling study before the
first retained measurement. Phase L2 extends the reviewed Phase L1 laboratory;
it does not alter a force law, Velocity Verlet, the viewer, CUDA, the JPL
catalog, counterfactual physics, frozen Stage 5, or any earlier evidence.

The candidate sizes are 8,192, 16,384, 32,768, and 50,000 bodies. A 100,000
body run is an optional stretch objective subject to separate admission and
approval. The first retained pilot is **only 8,192 bodies**. It stops after that
size regardless of outcome. No larger size may be launched automatically.

Calibration runs are disposable preflight or schema checks and are not
performance evidence. The 8,192-body run is an exploratory retained pilot used
to decide whether a separately reviewed 16,384-body performance-only preflight
is appropriate. A final controlled scaling measurement requires a later,
separately pre-registered protocol after pilot review.

The machine-readable authority for this pilot is
`tools/large_n_l2_protocol.json`.

## Pilot matrix

The pilot crosses these factors:

- body count: 8,192 only;
- distributions: uniform, clustered, and planar;
- backends: Sequential with one worker and OpenMP with four requested workers;
- solvers: Direct and Barnes--Hut;
- Barnes--Hut theta: 0.25, 0.5, and 0.8;
- explicit distribution seed:
  `0x4c3250494c4f5400 + distribution_index*0x100000 + N`;
- timestep: 10 seconds; trajectory length: one Velocity Verlet step;
- one warm-up and three retained repetitions.

This gives 24 configuration groups. Direct configurations are intentionally
submitted to the same admission machinery and are expected to produce six
explicit resource-skip rows at 8,192. The 18 Barnes--Hut groups request a
complete all-body Sequential Direct force and matched one-step trajectory
reference; each completed group contributes three retained rows. No solver or
backend fallback is permitted.

## Resource admission and stopping rules

The reviewed Phase L1 machinery remains authoritative:

- 512 MiB conservative laboratory memory budget;
- 500,000,000 estimated Direct interactions;
- at most 8,192 bodies in a complete Direct reference for this pilot;
- 30 seconds per configuration inside the laboratory;
- 35 seconds hard subprocess timeout including orchestration;
- checked allocation and work arithmetic;
- finite-value publication gates;
- immutable versioned runs, OS-backed publication locking, and an atomic
  authoritative pointer.

The memory estimate is conservative policy accounting, not an allocator-exact
RSS guarantee. A rejected Direct configuration stays an explicit skip. Limits
are not raised to obtain a completed result. A timeout, non-finite value,
execution failure, or malformed result prevents publication. An accuracy or
resource failure prevents any automatic larger-N run. This task always stops
after 8,192.

Sampled Direct reference measurements are neither implemented nor used here.
Any future sampled reference must declare deterministic sampled indices,
sample size, and separately named metrics; it may never be described as an
exact full-system Direct reference.

## Scientific gates

Sequential Direct remains the reference where admitted. The retained fields
include maximum and RMS absolute acceleration error, p95 and maximum relative
acceleration error, the denominator floor and cancellation-sensitive count,
matched one-step position and velocity separation, per-branch energy and
angular-momentum drift, tree statistics, interaction counts, and finite-state
status.

The pre-registered exploratory gates are:

| Theta | P95 relative acceleration error | Maximum relative error | Candidate relative energy drift | Candidate relative angular-momentum drift |
|---:|---:|---:|---:|---:|
| 0.25 | 0.01 | 0.25 | 1e-4 | 1e-4 |
| 0.5 | 0.05 | 1.0 | 1e-4 | 1e-4 |
| 0.8 | 0.15 | 5.0 | 1e-4 | 1e-4 |

These gates are protocol decisions for these deterministic synthetic fixtures,
not universal Barnes--Hut guarantees. One short step does not establish
long-duration accuracy, and complete Direct agreement does not validate a
synthetic distribution as an astronomical model.

## Timing and continuation decision

Generation, tree construction, traversal, one complete force evaluation, one
complete Velocity Verlet timestep, and total experiment time remain distinct.
Steps per second is `configured_steps / complete_timestep_s`; it is simulation
throughput, not rendering FPS. Requested, effective, and observed OpenMP worker
counts retain the Phase L1 provenance semantics. Three repetitions permit a
range-over-median variability warning at 20 percent, but are insufficient for
a universal crossover claim.

The evidence may recommend a **performance-only 16,384 preflight** only if all
18 Barnes--Hut groups complete, all six Direct groups are resource-skipped, all
theta-specific gates pass, the maximum total-time fraction is at most 0.8, and
the maximum admitted-memory fraction is at most 0.9. Such a recommendation is
not authorization to execute 16,384 and does not imply that a complete Direct
reference is admissible there.

## Custody

The L2 evidence lives separately under
`data/benchmarks/large_n_l2_pilot/`. Its source manifest binds the exact
protocol, runner, verifier, provenance capture, tests, build configuration,
laboratory sources, and unchanged physics sources used for measurement. Raw CSV
bytes are non-text in Git; JSON, Markdown, manifests, and manifest-listed source
files have explicit LF checkout rules. The strict verifier reconstructs the
matrix, scientific gates, summaries, hashes, provenance, and authoritative
publication chain.

The committed L1 verifier resolves L1 source custody from its evidence commit,
so later L2 source development cannot reinterpret or invalidate the historical
L1 run. L1 raw data and manifests are not modified.

## First pilot result

The committed, independently source-reviewed authoritative publication is
`run-20261015T040233.498852Z-9878e74b0e13`. It contains exactly 60 rows: 54
completed Barnes--Hut rows and six explicit Direct resource skips. All 18
Barnes--Hut configuration groups completed with complete 8,192-body Direct
references and all theta-specific accuracy gates passed.

Maximum admitted memory was 273,351,848 bytes (0.5091574937 of budget), maximum
median total experiment time was 24.3093273 seconds (0.81031091 of budget), and
the largest force range-over-median was 0.2615628326. The wall-time fraction is
above the pre-registered 0.8 continuation limit, so the protocol says not to
advance automatically. No larger workload was run and no full Direct reference
is promised at 16,384.

Authoritative hashes:

- raw CSV: `c9189b14003f9c4a82ffbbd186b0f5846a494cfef08d77ec689306df86b76329`;
- summary: `ac3940a66cc196544474ebf8c5ddb174280c3cf537d97edfe8c0eb3f66b98e4f`;
- protocol: `529b7c6ce6cfab66309b02795531fadccdcb815b92e6da84faf8650d055b063e`;
- build provenance: `022a090c45c0c9593a3b2b0bc18cebfdc2918d6ffb3a423e712eb6d19bf3d96a`;
- source manifest: `bce6f3b44156982e46da34e3b0d1ba71ea645a0f9197cf3fdd846aaf5dc7ae11`.

Local validation passed 25/25 focused L2 tooling cases, 23/23 inherited L1
publication/custody cases, 21/21 CPU Release CTests, and 25/25 OpenMP Release
CTests. The final focused independent source/custody review passed.
