# Large-N Laboratory L1 bounded CPU pilot

Exploratory, non-Stage-5 CPU evidence from the uncommitted Phase L1 candidate
based on `91fc9d1a1a5ff26b5d9b96df287ba4600afdb21b`. Run ID: `run-20261015T024403.329605Z-6b8ddc1da78e`.
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
`3.78x`--`3.90x`; timed-region-observed
OpenMP-4 Barnes--Hut speedup was `2.96x`--`3.95x`.
Tree construction occupied `0.09%`--`4.43%`
of complete Barnes--Hut force time. P95 relative acceleration-error ranges were
`2.28e-04`--`2.82e-03` at theta 0.25,
`5.83e-03`--`1.79e-02` at theta 0.5, and
`2.77e-02`--`6.73e-02` at theta 0.8. Maximum retained
force range/median was `45.3%`.

Custody hashes:
- raw.csv: `6593ab2a9eb39fe6a0a614570a5afb148cb4a7e7f4fb83f745558c29458b56a4`
- summary.json: `0a8521857d0af52461430629dc52957a0ab4e9fecb188be1fb017004ce42a7c8`
- metadata.json: `80fee3b0687040b70fbbc04ddafd517fe34adfd3f42d3bab9bbab28e8bcdb746`
- build_provenance.json: `5e82a031afcdac6e2c86d20dfeb08ac075c9d3a067becbe617a363e5bb12edbb`
- source_manifest.sha256: `e74f495fe9deb54ae447cc95c97d67c44f6a78b7c34154632c9d19a0611c7890`

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
