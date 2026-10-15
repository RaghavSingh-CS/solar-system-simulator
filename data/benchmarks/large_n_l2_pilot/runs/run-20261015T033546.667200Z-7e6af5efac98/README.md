# Large-N Laboratory Phase L2 — 8192-body pilot

This immutable run is exploratory controlled-scaling evidence, not frozen Stage 5
evidence and not a final crossover study. It contains the pre-registered N=8192
matrix only. Direct configurations that exceed the unchanged 500,000,000
interaction budget are retained as explicit skips; no fallback occurred.

Barnes--Hut configurations use complete all-body Direct references. No sampled
reference was produced. The theta-specific accuracy gate is
`passed`. The recommendation
for a separately reviewed 16384 performance-only preflight is
`True`.

`complete_force_s`, `complete_timestep_s`, and `steps_per_second` measure
simulation compute only. They are not rendering FPS. Three retained repetitions
support variability reporting but do not establish a universal crossover or
accuracy guarantee. CUDA, graphical integration, 16384, 32768, 50000, and
100000 execution are outside this run.
