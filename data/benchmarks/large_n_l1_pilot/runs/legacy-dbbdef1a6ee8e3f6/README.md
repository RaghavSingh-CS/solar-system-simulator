# Large-N Laboratory L1 bounded CPU pilot

Exploratory, non-Stage-5 CPU evidence from the uncommitted Phase L1 candidate
based on `91fc9d1a1a5ff26b5d9b96df287ba4600afdb21b`. Exactly 144 completed rows cover 72
configurations at N=256/1024/4096, three synthetic distributions,
Sequential/OpenMP-4, Direct/Barnes--Hut, and theta 0.25/0.5/0.8 for
Barnes--Hut. Each used one warm-up, two retained repetitions, one 10-second
Velocity Verlet step, 512 MiB admission, 500,000,000 Direct interactions, a
30-second in-process boundary budget, and a 35-second subprocess timeout.

OpenMP Direct workers are runtime-probed but not observed inside the timed force
region; Barnes--Hut records actual timed-region workers. Results are simulation
throughput, not graphical FPS. No CUDA, 50K, or 100K execution is represented.
Two repetitions and uncontrolled system conditions do not establish a stable
crossover.

At 4096 bodies, configured/probed OpenMP-4 Direct speedup was
`3.76x`--`3.94x`; timed-region-observed
OpenMP-4 Barnes--Hut speedup was `3.02x`--`3.91x`.
Tree construction occupied `0.09%`--`4.25%`
of complete Barnes--Hut force time. P95 relative acceleration-error ranges were
`2.28e-04`--`2.82e-03` at theta 0.25,
`5.83e-03`--`1.79e-02` at theta 0.5, and
`2.77e-02`--`6.73e-02` at theta 0.8. Maximum retained
force range/median was `52.9%`.

Custody hashes:
- raw.csv: `dbbdef1a6ee8e3f6f64ad995fba34b7e293f4310dadf00d84d73e62b34506da9`
- summary.json: `fb6f5623aad4f7cfbbc9c1b714f4b8ffb37837e17571d3b6005aec3f0a859732`
- metadata.json: `44abeb864b0b48407a84a13e58ba533cf62456af06010780189831878b34a9d6`
- build_provenance.json: `1a8cf1b36046993c9daa73bc62e52301b8aa5642a703dfae4c5741f7205d3cc1`
- source_manifest.sha256: `c78ef4eaf65f95cdfe1416934876b0263b137842b00649e64f79cb148d57fb33`

Run `python -B scripts/verify_large_n_l1_evidence.py` for strict read-only
matrix, summary, provenance, executable, and source-custody verification.
