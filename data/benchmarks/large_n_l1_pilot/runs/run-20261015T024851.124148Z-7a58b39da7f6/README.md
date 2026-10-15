# Large-N Laboratory L1 bounded CPU pilot

Exploratory, non-Stage-5 CPU evidence from the uncommitted Phase L1 candidate
based on `91fc9d1a1a5ff26b5d9b96df287ba4600afdb21b`. Run ID: `run-20261015T024851.124148Z-7a58b39da7f6`.
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
`3.18x`--`3.85x`; timed-region-observed
OpenMP-4 Barnes--Hut speedup was `2.94x`--`3.74x`.
Tree construction occupied `0.08%`--`3.31%`
of complete Barnes--Hut force time. P95 relative acceleration-error ranges were
`2.28e-04`--`2.82e-03` at theta 0.25,
`5.83e-03`--`1.79e-02` at theta 0.5, and
`2.77e-02`--`6.73e-02` at theta 0.8. Maximum retained
force range/median was `53.1%`.

Custody hashes:
- raw.csv: `2e13f307eb5f8f500db667f02c5106c766690eb9d313900f83449a3a46a7b9a8`
- summary.json: `51be7ed226eb51922988118a1bc0008e09d35436c1d12ce485e8cdb18aeed6f3`
- metadata.json: `f29181c6b59389bfb2f89a40c337bb26f3f90534223ad78cf4f06e2718d7b8eb`
- build_provenance.json: `188b76dc83ff21e8dc6c31be40b90c30b936f2b752b6acf514858c26b1a9c2ac`
- source_manifest.sha256: `47f987c664fcf1d9f52cd0e14983cd70691fcd3fe665f297770f0011822b77aa`

The enclosing `run_manifest.json` binds these immutable artifacts. The root
`publication.json` atomically selects the authoritative run and identifies all
historical published runs.

Legacy directory-swap backups are never adopted during ordinary measurement.
An operator may recover exactly one fully verified backup with
`python -B scripts/run_large_n_l1_pilot.py --recover-legacy-backup`; the source
backup is retained, and an existing current run remains current.

The OS lock and same-filesystem `os.replace` provide process-crash consistency
and atomic pointer visibility on the validated Windows filesystem. The pointer
temporary file is flushed and passed to `fsync`, but Python does not establish
directory-entry or storage-device flush durability here. Sudden-power-loss
durability is not certified.
