# Stage 4 CUDA benchmark evidence

This directory preserves the accepted real-NVIDIA Stage 4 evidence. `pilot/` contains 15 rows from the 10-step feasibility run; `run/` contains the controlled 154-row experiment. Each directory retains the unedited `raw_timings.csv`, supplied `summary.csv`, `environment.json`, and `cuda_ctest.txt`, plus an independently generated `verification.json`.

## Provenance

- Source contents: exact `git archive --format=zip ae6117b` export.
- Source archive SHA-256: `00fc9f3922ceb09647668a877a6193e10c7758338690fe945f16810e8e1b00fd`. A fresh local export of `ae6117b` reproduced this hash byte-for-byte.
- Git metadata was intentionally absent in Colab because `git archive` excludes `.git`; `STAGE4_COMMIT=ae6117b` supplied the source identity. The null `git` fields in the captured JSON are therefore retained, not rewritten.
- Environment ID: `colab-Tesla_T4-20260923T173009Z`.
- Pilot archive SHA-256: `ba84c4c3438135e8450e4aeb6cec6d6743a5502edf4d82a06e2130bcab7c24a2`.
- Full archive SHA-256: `7d3ed161f8bedbc70ec8896f087381bcfe324edcefeab7224660aa3804731326`.
- GPU: Tesla T4, compute capability 7.5, 15,360 MiB, 40 multiprocessors.
- Driver: 580.82.07. `nvidia-smi` displayed CUDA 13.0; the installed compiler/toolkit independently reported CUDA 13.3 / nvcc 13.3.73.
- Release build: CMake 3.31.10, GCC/G++ 13.3.0, Linux 6.6.122+, two logical host processors.

The uploaded shell script required temporary CRLF removal, removal of Git-only checks for the archive tree, and installation of `ninja-build`. These were orchestration repairs only; no CUDA source, test, workload, timing definition, or numerical threshold changed before measurement.

## Independent integrity verification

Run:

```bash
python3 -B scripts/verify_cuda_evidence.py data/benchmarks/stage4_cuda/pilot
python3 -B scripts/verify_cuda_evidence.py data/benchmarks/stage4_cuda/run
```

The verifier recomputes the supplied summaries from raw rows and checks row counts, the xorshift64*/Fisher-Yates schedule, missing/duplicate block-body pairs, positive finite timings, transfer-inclusive ordering, correctness status, repeat checksums, metadata consistency, and CTest success. Both datasets pass. Extracted-file SHA-256 values are recorded in each `verification.json`.

## Controlled full-run results

Every row used 100 Velocity Verlet steps, `dt=10 s`, one same-process warm-up, seed 24301, and one of 11 deterministic block-randomized repetitions. No row was excluded. Times are milliseconds; CV and overhead are percentages. Median is primary.

| N | Samples | Compute median | Compute mean | SD | CV | Compute min–max | Transfer-inclusive median | Transfer overhead | Overhead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 11 | 34.903 | 34.919 | 0.190 | 0.545% | 34.682–35.178 | 34.936 | 0.033 | 0.0939% |
| 96 | 11 | 47.641 | 47.607 | 0.250 | 0.525% | 47.289–47.962 | 47.662 | 0.021 | 0.0442% |
| 128 | 11 | 66.578 | 66.763 | 0.320 | 0.479% | 66.462–67.303 | 66.611 | 0.032 | 0.0487% |
| 160 | 11 | 94.524 | 94.693 | 0.369 | 0.390% | 94.389–95.469 | 94.581 | 0.057 | 0.0600% |
| 192 | 11 | 123.443 | 123.541 | 0.358 | 0.290% | 123.272–124.456 | 123.474 | 0.031 | 0.0251% |
| 256 | 11 | 196.193 | 196.414 | 0.543 | 0.276% | 196.054–197.777 | 196.226 | 0.034 | 0.0172% |
| 384 | 11 | 260.865 | 261.116 | 0.673 | 0.258% | 260.730–263.079 | 260.902 | 0.037 | 0.0142% |
| 512 | 11 | 326.349 | 326.409 | 0.967 | 0.296% | 325.577–329.070 | 326.386 | 0.038 | 0.0115% |
| 768 | 11 | 455.304 | 455.833 | 1.123 | 0.246% | 454.935–458.853 | 455.353 | 0.049 | 0.0107% |
| 1024 | 11 | 584.931 | 585.205 | 0.770 | 0.132% | 584.366–586.335 | 584.985 | 0.055 | 0.0093% |
| 1536 | 11 | 843.639 | 844.155 | 1.747 | 0.207% | 842.459–848.560 | 843.697 | 0.059 | 0.0069% |
| 2048 | 11 | 1102.595 | 1103.639 | 2.286 | 0.207% | 1101.537–1109.479 | 1102.650 | 0.055 | 0.0050% |
| 3072 | 11 | 1620.436 | 1620.990 | 3.043 | 0.188% | 1618.100–1628.546 | 1620.504 | 0.068 | 0.0042% |
| 4096 | 11 | 2140.987 | 2141.082 | 3.309 | 0.155% | 2137.275–2148.851 | 2141.060 | 0.074 | 0.0034% |

Full-run CV is 0.132–0.545%, so the retained samples are stable. Transfer-inclusive overhead is 0.021–0.074 ms and at most 0.094% under this implementation's definition; allocation and host staging remain separate. Pilot/full compute medians normalized per step agree within 1.1% at all five shared sizes.

The direct kernel still performs O(N²) interactions, but elapsed time over this measured grid does **not** show quadratic growth: 64→4096 bodies increases N by 64× and median compute time by 61.34× (global observed exponent 0.990); 1024→2048 and 2048→4096 increase time by 1.885× and 1.942×. Changing device utilization and fixed launch costs can affect the observed curve, so the dataset does not establish an asymptotic timing regime.

All 154 benchmark rows report `pass`; repeated checksums match exactly within each N. The maximum recorded relative energy drift is `1.1779626463387057e-13`, and maximum recorded relative angular-momentum drift is `3.740862421005083e-15`, far inside the frozen `5e-6` and `5e-10` benchmark guards. The CTest logs establish that CPU/CUDA force and state comparisons, repeated CUDA execution, conservation guards, collision rejection, and NaN rejection passed their predeclared tolerances; they do not print the individual cross-backend errors, so no bitwise-identity claim is made.

No Colab CPU control was collected. These CUDA times must not be divided by the Stage 3 Windows CPU times or used to infer a CPU/GPU crossover or selector threshold.
