# Stage 5 v2.0 remediation trace

This trace maps the preserved findings in
`docs/audits/pre_rtx_final_audit.md` to prospective remediation. The audit is
unchanged historical evidence. Local fixture rows are tests, never benchmark
evidence, and no Stage 5B collection has run.

| Finding | Remediation | Principal tests/evidence |
|---|---|---|
| B-01 | Replaced the `_OPENMP >= 201511` gate with a real-team feature probe covering explicit `num_threads`, requested/effective size, `schedule(static)`, completion, CPU equivalence, and deterministic repetition. Ordinary MSVC `/openmp` is permitted when semantics pass. | `nbody_openmp_validation`; embedded compiler/OpenMP build info |
| B-02 | Split acquisition into pilot, calibration, policy freeze, held-out, and evaluation. Calibration tooling cannot schedule held-out N. | phase-local deterministic schedules; mixed/missing/duplicate row rejection |
| H-01 | Requires absent build tree, clean exact source, validated Release/Ninja cache, embedded source/protocol/toolchain identity, binary/script hashes, controlled CTest, environment hash, and transitive evidence binding. | wrong/dirty source, stale home, Debug, wrong arch/hash, CTest tamper/pass, environment/topology negatives |
| H-02 | Keeps the tracked runbook SHA-neutral and generates a deterministic external deployment packet after the final commit. | exact-SHA packet generation, file hashes, forbidden-placeholder test |
| M-01 | Calibration independently re-verifies the actual pilot directory; approval binds its manifest SHA plus source/build/environment/topology identities. | pilot tamper and alternate-manifest approval rejection |
| M-02 | Native CIM capture supplies physical cores P and logical processors L; code derives sorted unique `{1, powers-of-two below P, P, L}` capped at L and fails instead of guessing. | derivation and forged-list tests |
| M-05 | Narrows the measured selector to N only for the fixed Stage 5 workload and primary `selector_steady_state` metric. | protocol and research wording |
| M-06 | Targets native VS2022/MSVC + Ninja + CUDA 12.8.x-class `sm_89`; captures absolute tools/versions and rejects MSYS/MinGW selection on controlled CUDA preparation. | build preparation preflight and external runbook |
| M-07 | Adds adversarial synthetic-custody tests without presenting fixtures as measurements. | `tests/test_stage5_v2_tools.py` |

Deferred exactly as audited: M-03 large-N correctness orchestration, M-04 MPI
timeout process-tree cleanup, M-08 historical Stage 3 runner, L-01 CUDA viewer
help, L-02 parser cosmetics, L-03 hostile numerical input API, and L-04
visualization dependency vendoring.
