# Sequential Barnes--Hut Stage 1 exploratory measurements

`raw.csv` preserves 108 measurements taken from the pre-commit Stage 1
candidate based on `b3b83fb967910f5476baec9796cc4fe3c58dc98f`. The
byte-identical solver source was subsequently committed as
`eb6dac75012f47d60a87417f6ccecf6df7c4576a`; the benchmark harness is archived
with this evidence. The raw CSV SHA-256 is
`226730b66c96a75822520a962199404c3bb27a1d5b76b9f3d2fd74890d621846`.
Git treats the raw CSV as binary so checkout line-ending conversion cannot
change those bytes. `metadata.json` records the command, compiler, host,
configuration matrix, and limitations.

These are bounded exploratory measurements, not frozen Stage 5 evidence and
not a cross-machine performance claim. Three repetitions do not establish a
stable production crossover. The useful observations are qualitative:

- At `theta=0.25`, the 4096-body median full Barnes--Hut force call (tree
  build plus traversal) remained slower than
  direct force for uniform (`0.23x`) and clustered (`0.31x`) distributions,
  while the planar case was faster (`1.27x`).
- At `theta=0.5`, the full Barnes--Hut force call remained slower for uniform
  (`0.76x`) and clustered (`0.78x`) inputs and was faster for planar
  (`2.84x`).
- At `theta=0.8`, the full Barnes--Hut force call was faster for all three inputs
  (`1.22x`--`5.67x`), but the force-error increase makes this a performance
  exploration setting rather than the accuracy-first default.
- At 4096 bodies and `theta=0.25`, 95th-percentile relative acceleration error
  was `6.07e-4` (uniform), `2.81e-4` (clustered), and `3.84e-3` (planar).
  The planar maximum relative metric is dominated by near-zero reference
  accelerations, so it must not be interpreted without the absolute and RMS
  columns retained in the raw file.

Tree construction was much smaller than traversal time in these cases, but a
complete Velocity Verlet step builds and traverses twice. Future controlled
crossover work needs more repetitions, randomized ordering, controlled host
conditions, broader distributions, and a preregistered error gate.
