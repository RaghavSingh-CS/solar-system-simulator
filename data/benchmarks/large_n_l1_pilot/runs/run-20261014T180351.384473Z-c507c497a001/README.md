# Large-N Laboratory L1 bounded CPU pilot

Exploratory, non-Stage-5 CPU evidence from the uncommitted Phase L1 candidate
based on `91fc9d1a1a5ff26b5d9b96df287ba4600afdb21b`. Run ID: `run-20261014T180351.384473Z-c507c497a001`.
Exactly 144 completed rows cover 72 configurations at N=256/1024/4096,
three synthetic distributions, Sequential/OpenMP-4, Direct/Barnes--Hut, and
theta 0.25/0.5/0.8 for Barnes--Hut. Each used one warm-up, two retained
repetitions, one 10-second Velocity Verlet step, 512 MiB admission,
500,000,000 Direct interactions, a 30-second in-process boundary budget, and a
35-second subprocess timeout.

OpenMP Direct workers are runtime-probed but not observed inside the timed force
region; Barnes--Hut records actual timed-region workers. Results are simulation
throughput, not graphical FPS. No CUDA, 50K, or 100K execution is represented.
Two repetitions and uncontrolled system conditions do not establish a stable
crossover.

At 4096 bodies, configured/probed OpenMP-4 Direct speedup was
`3.36x`--`3.90x`; timed-region-observed
OpenMP-4 Barnes--Hut speedup was `3.63x`--`4.62x`.
Tree construction occupied `0.08%`--`3.61%`
of complete Barnes--Hut force time. P95 relative acceleration-error ranges were
`2.28e-04`--`2.82e-03` at theta 0.25,
`5.83e-03`--`1.79e-02` at theta 0.5, and
`2.77e-02`--`6.73e-02` at theta 0.8. Maximum retained
force range/median was `57.5%`.

Custody hashes:
- raw.csv: `af97a41cfc25ace2369be029e24d113651a11d2bd94e2828c8560d278c90dcf0`
- summary.json: `b16c5387cd2dc18027d0fca471aba9e58795718eceb82d3a51e6bb8d47c155e2`
- metadata.json: `15950426e53a97b30a5558108adedf22e4e3bc403fec4f84c4f064d94fadaf82`
- build_provenance.json: `b989b5bd3a6c83c6fb8e75a9f7550b436906a9c28f05b2cfe269c550c4049ce8`
- source_manifest.sha256: `74a5dec83b6457b68e43ce03f84b107eba3e499af44857946c73e65191d4dade`

The enclosing `run_manifest.json` binds these immutable artifacts. The root
`publication.json` atomically selects the authoritative run and identifies all
historical published runs.
