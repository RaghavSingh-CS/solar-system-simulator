# Large-N Laboratory L1 bounded CPU pilot

Exploratory, non-Stage-5 CPU evidence from the uncommitted Phase L1 candidate
based on `91fc9d1a1a5ff26b5d9b96df287ba4600afdb21b`. Run ID: `run-20261014T175936.127643Z-ef334579b9a1`.
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
`3.48x`--`4.00x`; timed-region-observed
OpenMP-4 Barnes--Hut speedup was `3.39x`--`3.93x`.
Tree construction occupied `0.08%`--`3.65%`
of complete Barnes--Hut force time. P95 relative acceleration-error ranges were
`2.28e-04`--`2.82e-03` at theta 0.25,
`5.83e-03`--`1.79e-02` at theta 0.5, and
`2.77e-02`--`6.73e-02` at theta 0.8. Maximum retained
force range/median was `40.1%`.

Custody hashes:
- raw.csv: `4fc30cde77ac8c16d137da0bfc2a75a006319dd91bc8a34af6931fe9350e5303`
- summary.json: `a577e6ebd85842e5f1142a53113f919458e5055bab6c9f8f18d5b3ab05aaafc7`
- metadata.json: `2fc67a50f7ba8a65994d3c5021f027cb9ce1141548dcca33a92a9b99b8a003f1`
- build_provenance.json: `7afeb15e03d92f2fc45c337003147f5dc1580f06f17e4a407fe2a837b31b5344`
- source_manifest.sha256: `32a575868d4f56ab9952ba2b59f2ae73453badb2289e5c35ea112cb249150a3a`

The enclosing `run_manifest.json` binds these immutable artifacts. The root
`publication.json` atomically selects the authoritative run and identifies all
historical published runs.
