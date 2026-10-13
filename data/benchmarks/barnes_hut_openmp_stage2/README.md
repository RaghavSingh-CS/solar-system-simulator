# OpenMP Barnes--Hut Stage 2 exploratory measurements

`raw.csv` contains 216 bounded measurements from the uncommitted Stage 2
candidate based on `1f276505e5766d718e2979f4998831189f6865c8`. Its SHA-256 is
`11fb34ae4d3b4f87687cf177f1668139d7f5bad57ec944f78b316048f2f96760`.
Git treats it as binary so checkout line-ending conversion cannot alter the
measured bytes. `metadata.json` records the host, compiler, OpenMP runtime,
matrix, command, source identity, and limitations. `source_manifest.sha256`
binds the measurements to the exact uncommitted solver, public contracts,
scenario/reference implementation, benchmark harness, CMake wiring, preset,
raw CSV, and deterministic `build_provenance.json`. The provenance record
(SHA-256
`3b4f33cc83ff6f25b2954d1722d259c5e2fa0c3211c165e8eaf8c9ee19cab2c2`)
captures the actual GNU 15.1.0 executable/version, CMake 4.4.3, Ninja 1.13.2,
Release flags, OpenMP flags/cache/runtime libraries, selected compiler commands,
and benchmark link command. Recording these inputs does not guarantee
repeatable timing on another machine.

The harness covers uniform, clustered, and planar deterministic distributions;
`N=256,1024,4096`; theta `0.25,0.5,0.8`; and 1, 2, 4, and 8 effective threads.
There is one warm-up and two retained repetitions. Tree build, traversal,
complete force call, and complete two-force Velocity Verlet step are timed
separately. Direct-force comparisons use both Sequential and OpenMP backends.
Every retained OpenMP Barnes--Hut force result was first required to equal its
Sequential Barnes--Hut result bit for bit.

Verify hashes, the complete/unique matrix, effective thread counts, provenance
content, and recorded summaries offline with:

```powershell
python -B scripts/verify_barnes_hut_openmp_stage2.py
```

At 4096 bodies, median OpenMP Barnes--Hut complete-force speedup over
Sequential Barnes--Hut ranged from `3.25x` to `3.98x` with four threads and
from `4.28x` to `7.51x` with eight threads across the tested distributions and
theta values. The accuracy-first theta `0.25` eight-thread speedups were
`7.02x` uniform, `7.51x` clustered, and `7.33x` planar. Relative to OpenMP
direct force at eight threads, theta `0.25` Barnes--Hut was slower for uniform
(`0.26x`) and clustered (`0.35x`) but faster for planar (`1.24x`). The outcome
therefore remains strongly distribution- and theta-dependent.

These are exploratory single-machine timings with only two repetitions and
uncontrolled host scheduling, clocks, affinity, and thermals. They do not
establish linear scaling, a universal crossover, graphical FPS, or 50K--100K
feasibility. They are separate from and do not modify frozen Stage 5 evidence.
