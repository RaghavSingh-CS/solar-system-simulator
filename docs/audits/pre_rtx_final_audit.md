# Final pre-RTX adversarial audit

Audit date: 2026-09-27  
Audited feature-frozen `main`: `37a50685fccc881c22fadd95a779004f001a700d`  
Audited frozen Stage 5 revision: `4916b3c43298fa3f3bd15cb4af1698b5074abed3`  
Disposition: **NO-GO for the RTX Stage 5 session until B-01 and B-02 are remediated, re-frozen, and rehearsed.**  
This document is an audit artifact, not a fix. No production source, test, CMake, protocol, or historical evidence file was changed.

## 1. Executive summary

The current main revision is internally consistent and clean-room buildable for CPU-only, GCC/libgomp OpenMP, Microsoft MPI plus OpenMP, and visualization plus OpenMP on the audit host. All fresh CTest matrices passed. A clean CUDA configuration failed locally for the expected reason: this AMD host has neither an NVIDIA device nor `nvcc`. Static inspection found no defect in the CUDA force equation, binary64 storage, kernel indexing, two-force Velocity Verlet sequence, event synchronization, transfer semantics, or device-resident integration path. Preserved Stage 3, Stage 4, and conservation evidence reproduced its recorded hashes, row counts, schedules, summary derivations, checksums, and gates. No historical evidence appears invalid or silently modified.

The frozen Stage 5 procedure is nevertheless not currently executable as designed on its prescribed Visual Studio 2022 toolchain. Its OpenMP test requires `_OPENMP >= 201511` (OpenMP 4.5), while ordinary MSVC `/openmp` advertises OpenMP 2.0. The runbook neither selects nor validates an alternative runtime. This is a likely hard CTest stop before the pilot. Separately, a full Stage 5 run randomizes calibration and held-out configurations together, prints held-out timings live, and writes them into the same immediately readable raw and summary files. That contradicts the protocol's statement that held-out results are revealed only after a policy is fixed and creates a direct leakage route capable of invalidating selector evaluation.

Three high risks remain: Stage 5 provenance does not bind the executables and CTest log to the checked-out source; the exact frozen revision's own runbook and approval template still contain an unresolved source placeholder; and CUDA/Auto viewer integration does not exist, so it cannot responsibly be developed and validated during the same one-hour visit. The controlled Stage 5 evidence must take priority over presentation work.

Finding totals:

| Severity | Count |
|---|---:|
| BLOCKER | 2 |
| HIGH | 3 |
| MEDIUM | 8 |
| LOW | 4 |
| INFORMATIONAL | 8 |
| **Total** | **25** |

## 2. Audit scope

The audit covered the current tree and history; the exact frozen Stage 5 tree; physics, integration, diagnostics, and scenarios; Sequential, OpenMP, CUDA, MPI, and hybrid code; all CMake options; Stage 3/4/conservation evidence; all benchmark and evidence scripts; visualization/large-N policy; research and operational documentation; repository hygiene; and a clean-clone build/test exercise.

Executed work deliberately excluded Stage 5B measurements, the Stage 6D large-N pilot, recalibration, CUDA/Auto implementation, and any source or protocol modification. Live GUI work was limited to bounded launch/event-loop smoke checks. Temporary clean-room clones and build outputs were outside the repository.

Evidence standards used here were: an executed clean-room test; an independently recomputed artifact property; direct source/history inspection; or an explicitly identified external prerequisite. Absence of an NVIDIA device on the audit host is treated as a validation boundary, not as a repository failure.

## 3. Immutable revisions/evidence

### Forensic baseline

- Branch: `main`.
- Local HEAD: `37a50685fccc881c22fadd95a779004f001a700d`.
- `origin/main`: `37a50685fccc881c22fadd95a779004f001a700d`.
- Initial tracked status: clean; `main...origin/main` with no divergence.
- Frozen Stage 5 object exists exactly at `4916b3c43298fa3f3bd15cb4af1698b5074abed3` and is an ancestor of main.
- Relevant evidence-source revisions exist: Stage 3 `a4c9eb1176c0ff2efede6ee110ecad75e5bc423f`, Stage 4 `ae6117b`, and conservation `6d9a162da387ad58de37d44eaf9928bb9160549f`.
- No tags, no `.gitmodules`, no configured submodules, no Git LFS objects.
- 108 tracked files total 1,275,220 bytes. The largest tracked files are Stage 3 raw timings (266,303 bytes), committed GLAD header (117,065), Stage 4 full raw timings (68,191), and committed GLAD source (61,400). These are justified evidence or vendored source, not accidental binaries.
- The local checkout occupied about 1.118 GB because it retained `.tools`, 26 ignored build directories, and an ignored 100,328-byte ZIP. These are not part of a clean clone. `git status --ignored` emitted access warnings for four ignored Python dependency directories; this does not affect tracked content.
- `.gitignore` covers build trees, `.tools`, IDE files, executable/object products, and archives; controlled CSV under `data/**` remains trackable. `.gitattributes` fixes shell line endings and marks retained Stage 4 artifacts appropriately.

### Evidence immutability checks

- Stage 3's four current SHA-256 values match `data/benchmarks/stage3_cpu/README.md`. Independent parsing found 660 rows, 60 configurations, 11 complete randomized blocks, no missing/duplicate configuration, exact seeded order, stable within-configuration checksums, cross-CPU-backend checksum equality at each N, exact summaries, and all frozen gates passing.
- Stage 4 pilot and full verifiers passed from current retained files: 15 and 154 rows respectively, correct schedule, exact retained hashes, Release metadata, passing CUDA correctness, stable checksums, exact summary derivation, and frozen conservation gates.
- Conservation's three SHA-256 values match `data/experiments/conservation/README.md`. Independent parsing found 30 unique configurations, requested-duration equivalence across timestep multipliers, `samples = steps + 1`, exact repeatability for the declared repeated case, and all gates passing. The maximum observed time-accumulation discrepancy was approximately 0.075 seconds at the longest duration; requested durations were still equal and the discrepancy is represented in the evidence.
- The current Stage 3 and conservation runners match the recorded evidence-source versions for the relevant logic. Later work did not silently rewrite historical evidence.

## 4. Environment used for audit

- OS: Windows NT `10.0.26200`, x86-64.
- Audit CPU environment string: AMD64 Family 23 Model 104, 16 logical processors. Live CIM queries were access-denied, so physical-core/RAM details were not asserted from this audit session. Historical Stage 3 metadata independently describes a Ryzen 7 5700U, 8 cores/16 threads; that is evidence provenance, not a fresh system query.
- Python 3.12.11.
- Portable Kitware CMake 4.4.3, Ninja 1.13.2, GCC 15.1.0/libgomp.
- Microsoft MPI 10.1.12498.52.
- No `nvcc` and no `nvidia-smi`; the host uses AMD graphics.
- The PATH-selected MSYS CMake/CTest installation was unreliable on this machine. Clean-room tests therefore used the repository-local ignored portable CMake explicitly. This is an audit-host issue and a warning against relying on the first PATH match on the RTX laptop.

## 5. BLOCKERS

### B-01 — Prescribed MSVC build is expected to fail the frozen OpenMP test

- **Severity / subsystem:** BLOCKER — Windows Stage 5 build and validation.
- **Files/lines:** `docs/stage5_rtx4050_runbook.md:19-20,43-48,88-110`; `tests/test_openmp.cpp:238-239`; `docs/MASTER_LEDGER.md:56`.
- **Evidence:** The runbook requires an x64 Visual Studio 2022 environment and enables OpenMP. The test hard-requires specification date `201511` (OpenMP 4.5). Microsoft documents ordinary MSVC `/openmp` as OpenMP 2.0. CMake's `OpenMP_RUNTIME_MSVC` selector, which can request `experimental` or `llvm`, only exists from CMake 3.30, while the runbook permits CMake 3.20 and sets no runtime. The ledger itself says only GCC/libgomp was validated and MSVC behavior is untested. See [Microsoft `/openmp` documentation](https://learn.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170) and [CMake 3.30 FindOpenMP](https://cmake.org/cmake/help/v3.30/module/FindOpenMP.html).
- **Consequence:** The mandatory CTest phase is likely to fail before the RTX pilot even though the implementation uses only old, widely supported constructs (`parallel for`, `schedule(static)`, `num_threads`, and reduction). This can consume the visit and prevents accepted evidence.
- **Reproduction:** Configure frozen Stage 5 with the exact runbook under ordinary VS2022/MSVC/Ninja, build, and run `ctest`; inspect `openmp_specification_date`. This could not be executed on the AMD/GCC audit host, but the compiler/test incompatibility follows directly from the documented macro contract.
- **Recommended fix:** Before the visit, choose and validate one supported Windows CUDA-host/OpenMP toolchain. Either (a) revise the test requirement to the actual OpenMP feature level used, or (b) require CMake >=3.30 and explicitly configure/test a compatible MSVC OpenMP runtime, including its runtime DLL deployment. Do not patch during measurement.
- **Historical evidence impact:** None if only future Stage 5 preparation is revised; Stage 3 GCC and Stage 4 CUDA evidence remain immutable.
- **Frozen Stage 5 affected:** Yes, directly.
- **Retesting:** Full CPU/OpenMP/CUDA Release CTest and pilot on the exact newly frozen revision.
- **Before RTX visit:** Mandatory.

### B-02 — Held-out Stage 5 results are exposed before policy freeze

- **Severity / subsystem:** BLOCKER — experimental design and selector validity.
- **Files/lines:** `docs/stage5_protocol.md:24-35`; `scripts/run_stage5.py:343-377,706-769`; `docs/benchmark_protocol.md:28`.
- **Evidence:** The protocol says held-out N values are revealed only when evaluating a policy fixed from calibration data. Full mode instead combines all calibration and held-out configurations in one randomized schedule, prints every configuration's N/backend/selector timing live, writes every row to one flushed partial file, and produces one raw file and one summary containing both roles. No policy artifact, source hash, timestamp, or calibration-only freeze is required before held-out visibility.
- **Consequence:** A human or later analysis can see held-out performance before selecting thresholds, intentionally or accidentally tune to it, and still produce formally passing evidence. H2 and any claim of held-out evaluation would then be scientifically indefensible.
- **Reproduction:** `python -B scripts/run_stage5.py --dry-run --full ...` produces a mixed schedule; source lines 718-753 show immediate row flush and console disclosure. The dry run contained 98 configurations per block, 11 blocks, 1,078 process invocations, and 2,310 tidy timing rows.
- **Recommended fix:** Split collection/custody into phases or cryptographically seal held-out output. Produce a calibration-only artifact; derive and freeze the selector implementation/configuration plus hash; then unseal/run held-out evaluation with a manifest binding the policy hash. A randomized acquisition schedule may remain mixed only if held-out results are inaccessible to the policy author until after a trusted freeze; the current local file/console design does not provide that property.
- **Historical evidence impact:** None; this applies to uncollected Stage 5.
- **Frozen Stage 5 affected:** Yes, directly.
- **Retesting:** Runner/verifier negative tests, dry-run schedule/custody audit, a new pilot, and independent review.
- **Before RTX visit:** Mandatory.

## 6. HIGH findings

### H-01 — Stage 5 provenance does not prove binaries and CTest came from the frozen source

- **Severity / subsystem:** HIGH — evidence provenance and stale-cache defense.
- **Files/lines:** `scripts/run_stage5.py:482-520,580-608`; `scripts/verify_stage5_evidence.py:218-226`.
- **Evidence:** Preflight validates current Git HEAD/status, executable existence and `--help`, and two success substrings in an arbitrary CTest log. It records executable hashes and selected CMake cache fields only after measurement, but never binds those binaries/logs to this source tree, confirms `CMAKE_HOME_DIRECTORY`, verifies build timestamps/identities, or hashes the tested test executables. A copied build directory and stale successful log can therefore coexist with the correct checkout and create apparently `git-verified` evidence.
- **Consequence:** Accepted rows can be attributed to the frozen source when the executable was built from another revision or configuration. The exact runbook's clean clone and immediate build reduce the operational likelihood but the evidence verifier cannot establish it later.
- **Reproduction:** Copy a compatible old `nbody_benchmark.exe`, `nbody_cuda_benchmark.exe`, and successful CTest text into a clean frozen checkout. The shown preflight checks do not establish their source provenance.
- **Recommended fix:** Require a fresh absent build directory; record and validate `CMAKE_HOME_DIRECTORY`, generator, compiler, `CMAKE_BUILD_TYPE=Release`, CUDA architectures, configure timestamp, source SHA embedded at compile time, and hashes for benchmark/test executables. Generate the CTest log within the runner or bind it to a build manifest. Reject cache source mismatch before the pilot.
- **Historical evidence impact:** None if applied prospectively.
- **Frozen Stage 5 affected:** Yes.
- **Retesting:** New clean Windows CPU/OpenMP/CUDA build, CTest, pilot, and independent verifier tests.
- **Before RTX visit:** Mandatory.

### H-02 — The exact frozen Stage 5 tree is not self-executing

- **Severity / subsystem:** HIGH — runbook usability and source identity.
- **Files/lines:** frozen `4916b3c:docs/stage5_rtx4050_runbook.md`; frozen `4916b3c:docs/stage5_pilot_approval.template.json`.
- **Evidence:** At the exact frozen revision, the authoritative source and checkout command contain literal `STAGE5_FROZEN_SOURCE_COMMIT`, and the approval template contains the same placeholder. Current main later repairs the runbook, but checking out the frozen tree replaces those instructions. Following the frozen document literally cannot check out or approve the intended SHA.
- **Consequence:** The operator can lose time, enter the wrong source SHA, or improvise edits during the measurement session. The current-main runbook works only if it is deliberately retained outside the checkout and followed continuously.
- **Reproduction:** `git show 4916b3c:docs/stage5_rtx4050_runbook.md` and the corresponding template show the unresolved placeholders.
- **Recommended fix:** Publish a single immutable deployment packet outside the measured tree with exact SHA, commands, blank evidence root, and approval template; hash that packet. A new frozen Stage 5 revision should be self-contained.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** Yes.
- **Retesting:** Command-by-command rehearsal from a new clone.
- **Before RTX visit:** Mandatory.

### H-03 — Final CUDA/Auto viewer validation has no implementation to validate

- **Severity / subsystem:** HIGH — one-hour visit scope and presentation plan.
- **Files/lines:** `README.md:79`; `visualization/main.cpp:58,105-107`; `docs/MASTER_LEDGER.md:557-559,628,674-675,785`.
- **Evidence:** The viewer never links/constructs a CUDA backend path and always rejects `--backend cuda`; Auto is explicitly absent pending Stage 5 calibration. No selector artifact format, viewer integration branch, or frozen post-Stage-5 validation procedure exists.
- **Consequence:** “Final CUDA/Auto integration validation” cannot be accomplished as a validation-only tail to the one-hour session. It would be new feature development under time pressure, risking Stage 5 evidence and producing an unreviewed demo.
- **Reproduction:** A fresh visualization build rejected `--backend cuda` with exit 1; source has an unconditional throw.
- **Recommended fix:** Separate the visit goals. Collect and copy Stage 5 evidence first. Develop CUDA/Auto viewer integration later on a distinct revision using already verified data, then schedule a separate validation or remote execution. Preserve a Sequential/OpenMP viewer fallback for the visit.
- **Historical evidence impact:** None; do not rewrite past evidence.
- **Frozen Stage 5 affected:** No; it affects post-Stage-5 scope.
- **Retesting:** CUDA viewer correctness, Auto policy loading/failure behavior, CPU fallbacks, UI state, and real RTX smoke after implementation.
- **Before RTX visit:** The scope decision and fallback must be made before the visit; implementation is not a prerequisite for collecting Stage 5.

## 7. MEDIUM findings

### M-01 — Full-run gate trusts detached JSON rather than the pilot evidence directory

- **Severity / subsystem:** MEDIUM — Stage 5 pilot approval.
- **Files/lines:** `scripts/run_stage5.py:523-558`; `docs/stage5_pilot_approval.template.json`.
- **Evidence:** The gate checks a few fields in a verification JSON and hashes that JSON only after accepting it. It does not re-verify or bind an existing pilot directory/manifest; the approval contains no pilot manifest hash.
- **Consequence:** A hand-edited or mismatched passing report can unlock full collection.
- **Reproduction:** Supply a JSON with the required status/mode/protocol/source/environment/thread fields; no pilot files are opened by this function.
- **Recommended fix:** Require the pilot directory, rerun the independent verifier, and place its manifest hash in the human approval.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** Yes.
- **Retesting:** Full-gate negative tests and a new pilot approval rehearsal.
- **Before RTX visit:** Mandatory.

### M-02 — The topology-based OpenMP list is a human-only rule

- **Severity / subsystem:** MEDIUM — Stage 5 OpenMP matrix.
- **Files/lines:** `docs/stage5_protocol.md:56-60`; `scripts/run_stage5.py:472-477`; `scripts/verify_stage5_evidence.py:244-246`.
- **Evidence:** Code checks sorted/unique/include-1/not-above-logical, but not the prescribed `1 + powers of two below P + P + L` derived from captured topology.
- **Consequence:** A wrong yet syntactically valid list can pass and alter the comparison matrix.
- **Reproduction:** On any 8-core/16-thread description, pass `--omp-threads 1,3,16`; the code-level generic constraints accept it even though it violates the protocol rule.
- **Recommended fix:** Derive or validate the list against captured physical/logical topology and require the same derived list in pilot/full approval.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** Yes.
- **Retesting:** Thread-list negative tests and a new pilot.
- **Before RTX visit:** Mandatory.

### M-03 — Large-N CPU/OpenMP completion defaults to “correct”

- **Severity / subsystem:** MEDIUM — planned large-N evidence.
- **Files/lines:** `scripts/run_large_n_pilot.py:250-269,291-311`.
- **Evidence:** CPU/OpenMP benchmark output lacks `correctness_status`; the orchestrator maps any completed process to `pass` and does not independently require finite drift or enforce `5e-6`/`5e-10`.
- **Consequence:** A numerically invalid completed configuration can be recorded as correct.
- **Reproduction:** Feed `result_row` a completed parsed CPU row without `correctness_status`; line 266 produces `pass`. Line 307 likewise treats the missing field as passing.
- **Recommended fix:** Require and validate explicit finite checksum/drifts/status for every backend and recompute gates in the orchestrator.
- **Historical evidence impact:** None; no controlled large-N pilot exists.
- **Frozen Stage 5 affected:** No.
- **Retesting:** Large-N runner unit/integration tests with nonfinite and over-bound rows.
- **Before RTX visit:** Not for Stage 5; mandatory before any large-N evidence run.

### M-04 — MPI timeout may leave rank processes behind

- **Severity / subsystem:** MEDIUM — MPI large-N orchestration.
- **Files/lines:** `scripts/run_large_n_pilot.py:291-320`.
- **Evidence:** `subprocess.run(..., timeout=...)` controls the direct `mpiexec` process, but there is no Windows Job Object or explicit process-tree teardown.
- **Consequence:** MPI ranks can survive a timeout, consume CPU, and contaminate subsequent configurations.
- **Reproduction:** Source inspection establishes the missing tree teardown; a destructive forced-timeout pilot was deliberately not run during this audit.
- **Recommended fix:** Launch each configuration in a kill-on-close job/process group and verify no ranks remain before continuing.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** No.
- **Retesting:** A bounded timeout integration test that verifies all child ranks exit.
- **Before RTX visit:** Not for Stage 5; mandatory before any large-N pilot.

### M-05 — Declared policy dimensions exceed the frozen Stage 5 matrix

- **Severity / subsystem:** MEDIUM — research claim scope.
- **Files/lines:** `docs/research/novelty_assessment.md:11`; `docs/research/research_questions.md:9,19`; `docs/stage5_protocol.md:24-35`.
- **Evidence:** The proposed selector accepts `(N, step count, precision mode, accuracy budget)`, and research questions discuss step count/precision/cold-warm conditions, while Stage 5 fixes 100 steps, binary64, dt=10, one scenario/seed, and one pair of gates.
- **Consequence:** Stage 5 can support an N/thread/backend policy for that workload, not the broader function.
- **Reproduction:** Compare the research input tuple and secondary questions with the fixed Stage 5 matrix; only N/backend/thread vary.
- **Recommended fix:** Narrow claims to the measured domain or preregister additional independent experiments after Stage 5.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** No if claims narrow; yes if the experiment is expanded.
- **Retesting:** Documentation/claim review, or a separately preregistered experiment if broader claims are retained.
- **Before RTX visit:** The claim boundary should be decided before collection; extra experiments should not be added during the visit.

### M-06 — RTX toolchain prerequisites are not pinned or pre-proven

- **Severity / subsystem:** MEDIUM — RTX deployment prerequisites.
- **Files/lines:** `docs/stage5_rtx4050_runbook.md:43-48,84-99`.
- **Evidence:** The runbook says CMake 3.20+, VS2022, and a “current” compatible Toolkit/driver but does not pin a tested tuple, confirm Windows SDK/components, validate PATH selection, or prescribe a prebuilt smoke artifact. NVIDIA identifies RTX 4050 as compute capability 8.9, and Ada-native `sm_89` support begins with CUDA 11.8. Current NVIDIA documentation shows VS2022 support for current toolkits, but compatibility remains version-specific. See [NVIDIA GPU capability table](https://developer.nvidia.com/cuda/gpus), [Ada compatibility guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/), and [CUDA Windows installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).
- **Consequence:** Tool discovery or compiler compatibility can consume the one-hour visit even after B-01 is fixed.
- **Reproduction:** The AMD audit host can only confirm that CUDA enablement fails clearly without `nvcc`; the unpinned Windows/NVIDIA tuple remains an explicit unverified external dependency.
- **Recommended fix:** Select and rehearse one exact driver/Toolkit/VS/CMake/Ninja/Python/Git tuple on an equivalent Windows/NVIDIA host; cache the full CUDA installer and record checksums.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** Operationally yes.
- **Retesting:** Clean Windows/NVIDIA configure, build, full CTest, and pilot.
- **Before RTX visit:** Mandatory.

### M-07 — Frozen Stage 5 negative-path test coverage is too narrow

- **Severity / subsystem:** MEDIUM — Stage 5 orchestration tests.
- **Files/lines:** `tests/test_stage5_tools.py` (three tests).
- **Evidence:** Tests cover split disjointness, deterministic schedule, and a synthetic passing pilot. They do not cover tampered approval, full gate, stale source/build, non-Release metadata, partial files, duplicate/missing rows, output collision, wrong topology, held-out custody, or failure retention.
- **Consequence:** High-risk orchestration can regress undetected.
- **Reproduction:** `python -B tests/test_stage5_tools.py` reports three passing tests; test enumeration confirms the missing negative cases.
- **Recommended fix:** Add adversarial unit/integration tests for each gate after the protocol design is corrected.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** Yes.
- **Retesting:** Revised unit suite, dry runs, and new pilot.
- **Before RTX visit:** Mandatory.

### M-08 — Stage 3 runner did not enforce physics gates itself

- **Severity / subsystem:** MEDIUM — historical CPU collection methodology.
- **Files/lines:** `scripts/run_cpu_benchmarks.py:299-307`; retained Stage 3 evidence.
- **Evidence:** The historical runner checked requested/effective threads and checksum equality but did not reject non-finite or over-bound energy/angular drift at collection.
- **Consequence:** A future run using that harness could retain numerically invalid rows without failing. The retained 660 rows independently pass both gates, so current evidence is not invalid.
- **Reproduction:** Inspect the cited validation block and independently evaluate the two drift columns in all 660 rows.
- **Recommended fix:** For future CPU studies, enforce finite values and frozen bounds in the collector and verifier. Do not modify or rerun Stage 3 merely to change this.
- **Historical evidence impact:** None; all retained rows pass.
- **Frozen Stage 5 affected:** No; its runner enforces both gates.
- **Retesting:** Future runner negative tests only.
- **Before RTX visit:** No.

## 8. LOW findings

### L-01 — Viewer help implies a nonexistent CUDA-enabled viewer variant

- **Severity / subsystem:** LOW — viewer CLI/documentation.
- **Files/lines:** `visualization/main.cpp:58,105-107`.
- **Evidence:** Help says CUDA is accepted “only in CUDA-enabled viewers,” but every build rejects it.
- **Consequence:** This can waste operator time and imply a capability that does not exist.
- **Reproduction:** Fresh `nbody_visualizer --help` prints the text; `--backend cuda` exits 1.
- **Recommended fix:** After the audit, make help say CUDA viewer integration is unavailable until it truly exists.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** No.
- **Retesting:** CLI help/unavailable-path test.
- **Before RTX visit:** Optional; the fallback runbook should avoid the option.

### L-02 — Malformed and huge CLI input diagnostics are generic

- **Severity / subsystem:** LOW — CLI robustness.
- **Files/lines:** visualization and benchmark parsers.
- **Evidence:** `--bodies abc` reports `stoull`; an extreme benchmark body count reports `vector::reserve`; an unknown option without a value reports “missing value.” All cases exited nonzero and did not crash.
- **Consequence:** Operator diagnostics are not actionable; accepted protocol inputs are unaffected.
- **Reproduction:** The adversarial commands and exit statuses are recorded in section 17.
- **Recommended fix:** Validate conversion/range/known option before allocation and emit option-specific errors.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** No for prescribed inputs.
- **Retesting:** Parser boundary/overflow tests.
- **Before RTX visit:** Optional.

### L-03 — CPU preconditions do not reject every non-finite state component up front

- **Severity / subsystem:** LOW — numerical API robustness.
- **Files/lines:** sequential input validation, integrator, and diagnostics paths.
- **Evidence:** Invalid masses/G and force separations are checked, but a NaN velocity can propagate through integration into diagnostics rather than being rejected at the API boundary. Extremely large finite dt can overflow `dt²` without a post-step finite assertion.
- **Consequence:** Hostile external callers can receive NaN/Inf state instead of an immediate exception. Frozen protocols use finite controlled inputs and dt=10.
- **Reproduction:** The result follows from the input-validation and update paths; no hostile-state production run was needed to establish the missing checks.
- **Recommended fix:** Add full finite-state pre/postcondition tests for public integration entry points after RTX work.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** Not with prescribed inputs.
- **Retesting:** Numerical nonfinite/overflow unit tests.
- **Before RTX visit:** No.

### L-04 — Visualization dependency acquisition is network-dependent

- **Severity / subsystem:** LOW — visualization deployment.
- **Files/lines:** `CMakeLists.txt` FetchContent declarations; `README.md:53-62`.
- **Evidence:** A fresh visualization configure fetched pinned GLFW 3.4, GLM 1.0.1, and ImGui 1.92.9b and took about 108 seconds.
- **Consequence:** Firewall/offline failure will block a clean visualization build. Stage 5 does not enable visualization.
- **Reproduction:** The clean-room visualization configure required network access before its successful 5/5 test run.
- **Recommended fix:** Pre-populate a known-good dependency cache or prepare a verified source bundle before any RTX visualization work; keep Stage 5 build headless.
- **Historical evidence impact:** None.
- **Frozen Stage 5 affected:** No.
- **Retesting:** Offline visual configure/build is desirable.
- **Before RTX visit:** Not for Stage 5; recommended if any visualization activity remains planned.

## 9. Informational observations

### I-01 — Historical evidence passed independent integrity review

No Stage 3, Stage 4, or conservation artifact appears invalid, silently changed, or incorrectly summarized. This is not a claim that those environment-specific timings transfer to the RTX laptop.

### I-02 — No invalid cross-machine crossover claim was found

The repository consistently forbids dividing Stage 3 Windows CPU timings by Stage 4 Colab T4 timings. Viewer timing is labeled diagnostic, local MPI timings are labeled non-controlled, 100,000 bodies are labeled display-only, Auto is absent, and CUDA is not claimed as integrated in the final viewer.

### I-03 — Physics/render separation is real

Render positions/radii, guides, callouts, and amplified Sun trails are downstream of immutable physical snapshots. No renderer path writes simulation state. Reference guides do not drive bodies.

### I-04 — MPI complexity needs per-rank interpretation

Each rank stores O(N), computes approximately O(N²/P) target work, and receives O(N) accelerations per force evaluation. `MPI_Allgatherv` implementation and aggregate fabric traffic can exceed O(N) across the whole system; current documentation is acceptable when “O(N) communication” is read per rank, not as total network traffic independent of P.

### I-05 — CUDA setup timing is intentionally not cold-start application timing

The primary `selector_steady_state` metric excludes reusable context/allocation setup; allocation and staging are recorded separately. This is explicitly documented. Any future claim must retain “steady-state” and must not relabel it cold-start end-to-end time.

### I-06 — Repository secrets scan was clean

No token, credential, password, authenticated private URL, generated binary, crash dump, or core file was found in the tracked tree or searched history. Normal Git author metadata exposes `Raghav Singh <anilinesupremacy@gmail.com>`. Stage 3 provenance contains hostname `ACRYLIC_MARINE` and an absolute project path. These are privacy choices, not secrets or scientific defects.

### I-07 — No submodule/LFS dependency exists

There is no `.gitmodules`, `git lfs ls-files` is empty, and GLAD is deliberately vendored. `git submodule status` failed on this host because Git's internal shell could not find `basename`/`sed`; absence of `.gitmodules` establishes that no configured submodule is required.

### I-08 — Some native UI behavior remains observational

Headless tests exercise reset, scenario/backend/thread changes, camera presets, selection lifetime, trails, labels, health mapping, and presentation invariants. Fresh Solar/OpenMP and 100,000-body display-only processes stayed in the OpenGL event loop for five seconds until deliberate termination. Pixel-level composition, resize dragging, hover behavior, repeated close/reopen, and subjective appearance were not automated in this audit; prior human Stage 6D validation is not replaced by these smoke checks.

## 10. Physics/numerics assessment

### Equations, units, and state

The direct target equation is correctly implemented as

`a_i = G * sum(j != i) m_j (r_j - r_i) / |r_j-r_i|^3`

with `G = 6.67430e-11 m^3 kg^-1 s^-2`. Mass is kg, position m, velocity m/s, and time/dt seconds. Each backend excludes `i == j`; Sequential/OpenMP/CUDA traverse sources in ascending index order. Accumulation and state are binary64. The shared target kernel used by Sequential/OpenMP/MPI preserves the same target/source semantics.

### Velocity Verlet and diagnostics

The implementation computes old acceleration, drifts position by `v*dt + 0.5*a*dt²`, computes new acceleration at the updated positions, kicks velocity by `0.5*(a_old+a_new)*dt`, and increments simulation time once. Backends return acceleration arrays; state ownership remains with the integrator. Diagnostics are outside benchmark timed regions unless explicitly described.

Total kinetic energy is `0.5*m*|v|²`. Potential energy visits each unordered pair once (`j=i+1`) and uses `-G*m_i*m_j/r`. Angular momentum is the vector sum `Σ r × mv`, not a scalar proxy. The benchmark energy drift uses a symmetric magnitude scale with a floor of one; angular drift uses vector difference over initial angular magnitude with a floor of one. Current benchmark, Stage 5 runner/verifier, conservation runner, tests, and documentation agree on the frozen limits `5e-6` and `5e-10`.

### Determinism and falsification checks

The seeded synthetic generator consumes `mt19937_64` raw values through an explicit 53-bit mapping and removes center-of-mass position/velocity. The Solar initializer solves the documented JPL mean-element approximation, derives velocities by centered finite difference, then applies one barycentric shift. It is not a Horizons ephemeris. Reference guides sample the same mean-element model but remain presentation-only.

Attempts to falsify conservation included direct current tests, Stage 3 row-by-row gate/checksum checks, Stage 4 verifier replay, conservation summary recomputation, and MPI equivalence at 1/2/4 ranks plus 2x2 hybrid. They did not reveal backend-self-comparison, stale-state reuse, missing second force evaluation, or relaxed gates. The conservation study exercises up to 100 duration multipliers and reports maximum rather than final-only drift. The remaining low-level finite/overflow gap is L-03.

## 11. Backend assessment

### Sequential

Reference order is deterministic: destinations and sources ascend, each destination owns a local accumulator, and output storage is distinct. No aliasing or lifetime defect was found.

### OpenMP

The backend contains a real `parallel for schedule(static) num_threads(...)`. Each iteration owns its accumulator and output element; invalid input is reduced to a flag. There is no shared floating-point reduction, so a destination's source addition order matches Sequential and was bitwise-equal on this GCC/libgomp host. Requested/effective teams 1/2/4 passed. `OMP_DYNAMIC=FALSE` is set for controlled collection, and affinity is explicitly not claimed. Oversubscription beyond detected logical processors is rejected by Stage 5, but topology-rule enforcement is M-02. MSVC version compatibility is B-01.

### CUDA (static/host-side only here)

- One CUDA thread owns one destination, with integer body-count range checked before launch, index bounds checked, and self excluded.
- `DeviceVec3`, masses, positions, velocities, acceleration, and operations are binary64.
- Launch geometry uses 256 threads per block and ceiling division.
- Each Velocity Verlet step executes old acceleration, position update, new acceleration, and velocity update; state stays resident across steps.
- H2D/D2H, invalid-flag handling, kernel launch errors, CUDA events, event synchronization, stream synchronization, allocations, and frees are checked.
- Transfer-inclusive timing spans H2D through compute and D2H completion. Device-compute timing starts after H2D. Reusable allocation/context and host staging are reported separately.
- No current-main CUDA build or runtime result is claimed on the AMD audit host. Retained real-GPU evidence is the T4 run only.
- RTX 4050 is officially listed as compute capability 8.9; the runbook's `CMAKE_CUDA_ARCHITECTURES=89` is appropriate with a toolkit capable of native Ada code. Toolkit/host compiler compatibility remains an external preflight requirement.

### MPI and hybrid

Every rank holds the full state. Deterministic contiguous destination partitions assign remainder targets to lower ranks. Counts/displacements match target slices, invalidity is globally reduced, and `MPI_Allgatherv` rebuilds the full acceleration vector before each shared-state update. Both Verlet force evaluations use this backend. MPI calls occur outside the hybrid OpenMP region after `MPI_Init_thread(...MPI_THREAD_FUNNELED...)`. One-rank, nondivisible N=11, 2/4-rank, repeatability, and 2-rank x 2-thread tests passed. No deadlock or root/non-root asymmetry was observed. MPI timing uses maximum-rank wall time and separately reports local compute and collective communication. No repository statement elevates local diagnostics to multi-node evidence; MPI+CUDA remains future work.

## 12. Stage 5 frozen-protocol assessment

### Internally consistent elements

- Protocol version `stage5-v1.0`.
- Calibration N: `64,128,192,256,512,1024,2048,4096`.
- Held-out N: `96,160,384,768,1536,3072`.
- Pilot N: `64,512,4096` only.
- Pilot: 3 blocks, 10 steps. Full: 11 blocks, 100 steps.
- Fixed dt 10 s, seed 24301, order seed 20260926, one warmup before every measured process invocation.
- Each repetition is a complete configuration block shuffled by deterministic xorshift64*/Fisher-Yates. Independent dry-run reproduction was exact.
- CPU `integration_compute` excludes initial-state copy, while CPU `selector_steady_state` includes the copy plus integration. CUDA device/transfer/selector metrics retain their documented boundaries. Output values use round-trip-safe textual precision.
- The verifier detects missing/duplicate/order errors, nonpositive/nonfinite timings, timing relationship violations, failed correctness, checksum instability, source/environment/protocol mismatches, and non-Release rows. Output directories must be absent/empty, and partial failures remain `.partial.csv` plus `failure.json`; they cannot satisfy the manifest-based independent verifier.

### Not currently defensible/executable

The protocol is **not currently executable as designed** because B-01 likely blocks the mandatory prescribed build/test and B-02 breaks its held-out reveal rule. H-01/H-02 further weaken provenance and operator reliability. There is no acceptable route to “just run it and document the caveat”: that would consume the scarce host session and either produce no evidence or produce methodologically compromised selector data.

### Policy-generation readiness

Even after valid collection, no policy generator or frozen policy artifact exists. Stage 5B should collect defensible evidence only. Policy derivation must use calibration rows exclusively, record exact code/configuration/hash, and run held-out evaluation only after freeze. Claims must be scoped as in M-05.

## 13. Historical evidence integrity

### Stage 3 CPU

All 660 rows and 60 summaries were independently checked. Eleven blocks each contain the expected 60 configurations; no row/configuration is missing or duplicated. Orders reproduce from the declared seed. Requested/effective teams agree. Checksums agree across Sequential and OpenMP for each N. Every row passes current frozen conservation gates. All reported medians, means, sample deviations, min/max, IQR, CV, and deterministic bootstrap intervals reproduce. The documented 128-160 transition is appropriately called a region because intervals overlap at 128; no universal threshold is claimed.

### Stage 4 T4

The retained 15-row pilot and 154-row full set pass the independent verifier. Environment/source archive provenance, schedule, Release metadata, CUDA timing ordering, summaries, checksums, and gates match. The evidence supports T4-specific real-CUDA build/correctness/scaling under the declared metric definitions only. It does not support an RTX crossover or a cross-machine CPU/GPU ratio.

### Conservation

Thirty configurations cross two scenarios, five dt multipliers, and three equal requested-duration multipliers. Diagnostic sample count is initial plus every step. All configurations pass. Maximum recorded energy/angular vector drift is `3.8964650316226864e-10` / `1.215493287078012e-13` for analytic two-body and `9.275550454152302e-13` / `1.931667677713518e-14` for seeded-32, all well within frozen gates.

### Integrity verdict

No historical evidence appears invalid. The audit found methodology limitations (M-08) and scope boundaries, not a reason to rewrite, discard, or rerun accepted Stage 3/4/conservation evidence.

## 14. Build/deployment assessment

### Clean-room results

From an isolated clone of current main:

| Configuration | Result |
|---|---|
| CPU-only Release | configure/build passed; CTest 3/3 |
| OpenMP Release (GCC/libgomp) | configure/build passed; CTest 4/4 |
| MPI + OpenMP Release (Microsoft MPI) | configure/build passed; CTest 8/8, including ranks 1/2/4 and 2x2 hybrid |
| Visualization + OpenMP Release | dependency fetch/configure/build passed; CTest 5/5 |
| CUDA enabled locally | expected clear configure failure: no usable CUDA compiler |
| Frozen Stage 5 CPU/OpenMP Release | configure/build passed; CTest 2/2; Stage 5 Python tests 3/3 |

CMake requires 3.20 and C++17. CPU defaults are dependency-light. CUDA, OpenMP, MPI, and visualization are opt-in. CUDA language activation and CUDAToolkit discovery occur only when requested. Microsoft MPI was discoverable on this host; deployment still needs its runtime on the target. GLAD is committed; GLFW/GLM/ImGui are pinned but fetched. Shaders are embedded and no runtime font/assets directory is required. Ninja is assumed by the documented workflows. Single-config Release semantics are correct; passing `--config Release` to Ninja is harmless. Paths containing spaces were not exercised, but command construction uses argument arrays or PowerShell quoting in the audited critical scripts.

Absolute path matches under `C:\Projects`, `C:\Users`, and `C:\msys64` are retained evidence provenance or documented local commands. They are not compiled into core runtime behavior. The hardcoded example evidence root `C:\stage5-evidence` is an operator-editable runbook choice.

## 15. RTX 4050 readiness assessment

### Verdict

**Not ready for a clean-room RTX rehearsal or the one-hour measurement visit.** General clean-room readiness for current CPU/OpenMP/MPI/visualization code is good, but RTX Stage 5 readiness is blocked by B-01/B-02 and weakened by H-01/H-02.

### Time-critical pre-flight checklist

Complete this before travel, in order:

1. Resolve B-01 and B-02; create one new immutable Stage 5 source/protocol revision. Do not change physics, workloads, seeds, gates, or timing semantics unless separately justified.
2. Select an exact supported Windows x64 tuple: Windows build, NVIDIA driver, CUDA Toolkit with native `sm_89` support, VS2022/MSVC toolset and Windows SDK, CMake >= the revised minimum, Ninja, Git, and Python. Record versions.
3. On an equivalent NVIDIA Windows machine, perform a fresh absent-directory configure/build/CTest/pilot using that exact tuple and exact new SHA. Confirm requested/effective OpenMP teams and runtime DLL availability.
4. Pre-download the full CUDA installer, matching driver installer if needed, VS Build Tools layout/components, portable CMake/Ninja/Python if policy permits, repository bundle/clone, and pinned visualization dependency cache. Verify hashes. NVIDIA's full installer is intended for offline use; recent Toolkit versions may require a separate driver download.
5. Prepare a signed/hashed deployment packet: exact SHA, clone URL or bundle hash, exact commands, topology rule, blank evidence-root name, corrected approval template, abandonment criteria, and copy-out destinations.
6. Confirm `nvidia-smi`, `nvcc --version`, `cl`, Windows SDK, CMake, Ninja, Python, and Git from the same VS x64 shell; ensure PATH selects the intended tools, not MSYS/MinGW.
7. Confirm AC power, high-performance/vendor performance mode, discrete GPU access for the process, adequate free disk/RAM, sleep disabled, and nonessential background workloads paused. Record laptop model, CPU topology, RAM, GPU name, VRAM, driver, power state, and display/Optimus arrangement. Do not assert constant clocks or thermals if not measured.
8. Run a tiny NVIDIA CUDA sample or the project's CUDA validation outside the measurement window. Reboot if an installer requires it; never discover that during collection.
9. Ensure evidence will be written outside the repository and immediately copied to two destinations after pilot/full. Test write permissions and ZIP/hash commands.
10. Treat CUDA/Auto viewer work as a separate, optional later activity. Carry a verified Sequential/OpenMP viewer binary/demo fallback.

## 16. Large-N assessment

The viewer's 100,000-body claim is rendering/display scalability only. Continuous integration, trails, live O(N²) diagnostics, and live comparison stop above 1,024. Manual direct-force stepping stops above 8,192. Larger choices are display-only; they use a batched point-sprite buffer, selected-only inspection, and O(1) selector state rather than per-body labels/trails. A bounded 100,000-body launch stayed in its event loop until deliberately stopped; the large-N benchmark pilot was not run.

Allocation/index review found body counts constrained to explicit UI choices through 100,000, while CUDA separately rejects counts beyond `int`. No UI path was found that invokes O(N²) force work at 100,000. Documentation does not extrapolate unmeasured performance. Planned timeout rows are retained as `infeasible_within_budget`, and error logs/raw configurations are not silently dropped. M-03 and M-04 must be fixed before that protocol produces evidence.

## 17. Visualization robustness assessment

Executed adversarial CLI results:

| Case | Result |
|---|---|
| Invalid backend | exit 1, explicit allowed values |
| CUDA when unavailable | exit 1, explicit unavailable diagnostic |
| Synthetic bodies 0 | exit 1 |
| Width 0 | exit 1 |
| Benchmark bodies 0 or 1 | exit 1 |
| dt 0 or -1 | exit 1 |
| OpenMP threads 0 | exit 1 |
| malformed/overflow bodies and unknown option | exit 1; diagnostics have L-02 quality issues |
| Solar/OpenMP-4 1920x1080 | remained in event loop for bounded smoke interval |
| Synthetic 100,000 display-only | remained in event loop for bounded smoke interval |

Headless tests cover repeated reset/state equality, scenario changes, backend/thread selection, camera presets, selected-body validity, trails, labels, health gates, viewport/radius behavior, and Solar geometry. The controller clears/invalidate dependent state when scenario, N, or comparison inputs change. Fixed physical dt remains independent of frame time; speed controls alter only integer steps per frame. Solar N=9 is explicitly described as unsuitable for speedup claims. The comparison panel labels results `LIVE DIAGNOSTIC - NOT BENCHMARK EVIDENCE`. The 100,000-body policy is described as display/inspection-only. No stale pointer or renderer-to-physics feedback path was found.

## 18. Documentation/claim matrix

| Claim | Source | Supporting evidence | Status |
|---|---|---|---|
| Sequential/OpenMP implement the same direct binary64 force law | README, architecture | source trace; current cross-backend tests | supported |
| CUDA backend is real-GPU validated on T4 | README, ledger | retained Stage 4 CTest/pilot/full plus verifier | supported, environment-qualified |
| No same-machine CPU/CUDA crossover exists yet | README, protocol, ledger | no RTX evidence; cross-machine warning search | supported |
| Stage 3 transition is approximately N=128-160 on its host | benchmark protocol, ledger | 660-row summaries and overlapping intervals | supported, host-qualified |
| Stage 5 held-out values cannot influence policy | Stage 5 protocol, ledger | contradicted by runner visibility/custody | **contradicted (B-02)** |
| Frozen Stage 5 runbook is executable on VS2022 | runbook | MSVC OpenMP macro mismatch; placeholder in frozen tree | **unsupported / likely contradicted (B-01, H-02)** |
| Proposed selector maps `(N, steps, precision, budget)` | novelty/research questions | Stage 5 varies only N/backend/thread | **overbroad / misleading (M-05)** |
| 100,000 bodies are supported by the viewer | README, visualization | explicit display-only controls and smoke launch | supported only as display scalability |
| Viewer timing is benchmark evidence | explicitly denied in README/UI/docs | labels and isolated diagnostic path | supported denial |
| CUDA is integrated into final viewer | explicitly denied in README/docs; CLI help ambiguous | unconditional runtime rejection | not claimed; help is misleading (L-01) |
| Auto is calibrated/implemented | explicitly denied | no selector/viewer Auto implementation | not claimed |
| MPI scales across a cluster | explicitly not claimed | local correctness and diagnostic timings only | not claimed |
| Solar scenario is physically exact/ephemeris-grade | explicitly denied | JPL mean elements and known EMB/Moon limits | not claimed; approximation correctly qualified |
| Solar/reference orbits retain eccentricity | visualization/ledger | 700-day extrema, osculating tests, guide sampling | supported within stated approximate model |
| Render exaggeration cannot change physics | architecture/visualization | one-way data flow and regression tests | supported |
| Historical evidence is deterministic/immutable | evidence READMEs/ledger | hashes, schedule and summary replay | supported for retained artifacts |

Searches found no claim that Stage 4 proves an RTX crossover, local MPI proves multi-node scaling, viewer diagnostics are controlled evidence, 100,000 bodies are directly integrated, Auto already exists, or final viewer CUDA is integrated. “Validated” is generally accompanied by the relevant host/configuration. “Deterministic” is correctly scoped to scenario generation/order/reference behavior, not universal bitwise cross-device identity.

## 19. Repository hygiene/security

No secret-like token or credential was found in the tracked tree or searched history. The public remote contains no embedded credentials. No executable/object/build product, IDE state, dump, or crash file is tracked. Git LFS and submodules are unnecessary. Vendored GLAD files and evidence CSV/JSON/logs are deliberate.

Search terms such as TODO/FIXME/HACK/XXX/temporary/placeholder/mock/fake/illustrative/hardcoded were classified by context. Most matches are legitimate protocol labels, test fixtures, negative claims, or historical orchestration notes. The actionable placeholder is H-02. Absolute paths are mostly reproducibility metadata and documentation. The normal author email, Stage 3 hostname, and historical absolute path are the only noteworthy personal/machine identifiers.

The local ignored 1.118 GB workspace is not clean-room content. Do not copy this working directory or any build cache to the RTX laptop; start from a verified clone/bundle and absent build directory.

## 20. One-hour RTX risk register

| Risk | Trigger / impact | Preventive action | Abort/fallback |
|---|---|---|---|
| MSVC OpenMP test failure | B-01; no accepted pilot | fix and rehearse before visit | do not patch frozen code; preserve logs; abort Stage 5 |
| Held-out leakage | B-02; invalid selector evaluation | sealed two-phase workflow and policy hash | do not inspect/use held-out data; abort full if custody breaks |
| Wrong/stale binary/cache | copied build or log; false provenance | absent build dir, source-embedded SHA, manifest binding | discard output and rebuild before any measurement |
| Missing/incompatible CUDA/VS/SDK | configure/compiler failure | exact preinstalled/rehearsed tuple and offline installers | spend at most 10 min confirming known repair; otherwise abort measurements |
| Wrong PATH/generator | MSYS/MinGW tool selected | x64 VS shell; print absolute tool paths/versions | restart clean shell, never improvise compiler mixing |
| Driver/device/Optimus/power | device unavailable or unstable timings | AC power, performance mode, `nvidia-smi` precheck | record failure; no CPU/CUDA conclusions |
| Thermal/background load | noisy or biased medians | cool machine, pause workloads, record telemetry/availability | repeat pilot only if preregistered gate says unstable; never cherry-pick rows |
| Network/firewall | clone/FetchContent/install stalls | local verified bundle, full installers, dependency cache | skip visualization; Stage 5 headless has priority |
| Pilot too slow/fails correctness | insufficient hour or invalid path | rehearsed runtime estimate, pilot approval gate | stop; preserve pilot/failure, do not run full |
| Evidence loss | laptop session ends | external evidence root, immediate dual copy and SHA-256 | never delete originals; no claims without manifest |
| CUDA/Auto viewer scope creep | consumes remaining time | declare optional/separate | use Sequential/OpenMP demo; no live coding |

## 21. Ordered remediation plan

1. Resolve B-01 by choosing the exact Windows OpenMP/CUDA-host toolchain and aligning the test with used features or an explicitly supported runtime.
2. Resolve B-02 with enforceable calibration/held-out custody and a frozen policy hash before held-out reveal.
3. Resolve H-01 by cryptographically/source-manifest binding build, binaries, tests, configuration, and evidence.
4. Resolve H-02 by creating one self-contained immutable deployment packet/new frozen revision with no source placeholders.
5. Add M-01, M-02, and M-07 adversarial gate tests; then perform a full clean-room Windows/NVIDIA rehearsal through pilot verification and human approval, stopping before full.
6. Pin/pre-download/rehearse the exact driver/Toolkit/VS/CMake/Ninja/Python/Git tuple (M-06).
7. Narrow planned research claims to the measured domain (M-05).
8. Freeze the new revision and report; do not mutate it at the RTX visit.
9. After Stage 5 evidence is safely copied and independently verified, separately plan CUDA/Auto viewer implementation (H-03).
10. Before any future large-N pilot, repair M-03/M-04. Leave Stage 3 evidence untouched; apply M-08 only to future runners.
11. Address low parser/validation/help/dependency issues only after critical evidence work.

## 22. Exact recommended clean-room rehearsal after remediation

1. Create a brand-new directory on a Windows NVIDIA machine; clone or unpack the verified repository bundle. Confirm HEAD equals the new frozen SHA and `git status --short` is empty.
2. Open the prescribed x64 VS developer shell. Print absolute paths and versions for Git, Python, CMake, Ninja, `cl`, `nvcc`, `ctest`, and `nvidia-smi`; record GPU query, compute capability, driver, CPU physical/logical topology, RAM, and power state.
3. Derive the OpenMP thread list automatically from topology and compare it to the deployment packet.
4. Confirm the build directory and evidence root are absent. Configure Release with the exact revised flags, explicit compiler/toolkit if needed, `NBODY_ENABLE_OPENMP=ON`, `NBODY_ENABLE_CUDA=ON`, and architecture 89.
5. Build. Verify the CMake cache source directory, generator, Release mode, compiler/toolkit versions, CUDA architecture, and OpenMP runtime. Hash benchmark and test executables.
6. Run complete CTest and generate its manifest through the revised runner, not by accepting a detached text file.
7. Run preflight and inspect every captured field. Confirm executable/source/test manifest binding.
8. Run only the pilot. Verify independently from the pilot directory, validate all gates/timing relationships/checksums/rows, review estimated full duration, and create approval containing the pilot manifest hash.
9. Exercise the calibration-only output and policy-freeze mechanism with synthetic/dummy evidence; prove held-out rows cannot be viewed before the policy artifact hash exists. Do not collect real full timings in the rehearsal if the experimental plan reserves the target session.
10. Test failure cases: tampered source SHA, stale build source, Debug cache, wrong architecture, wrong GPU, changed thread list, altered pilot report, duplicate/missing row, partial run, existing output, and attempted held-out reveal.
11. Copy the rehearsal bundle twice, verify ZIP/file hashes on the destination, and time the entire sequence. Accept the visit only if the pilot path plus projected full run fits comfortably inside 60 minutes with contingency.

## 23. Exact recommended RTX session sequence

Assuming all go/no-go criteria are already met:

- **00:00-00:05:** Connect AC power, enable verified performance mode, stop nonessential workloads, confirm free space, launch the prescribed x64 VS shell.
- **00:05-00:10:** Verify device/tool versions, exact SHA, clean tree, topology-derived thread list, absent build/output directories, and local deployment-packet hashes. If any differs from the rehearsal, stop and assess; do not measure.
- **00:10-00:20:** Configure/build the frozen headless CPU/OpenMP/CUDA targets in the absent build directory. Inspect cache/source/compiler/architecture/runtime fields. Abort rather than patch if build fails.
- **00:20-00:25:** Run full CTest and source-bound preflight. Copy the log/preflight to the first backup immediately.
- **00:25-00:32:** Run the pilot, independent verification, and human review. Check GPU identity, row completeness, checksums, gates, requested/effective teams, timing ordering, stability, and projected full duration. Human approval occurs here and binds the pilot manifest hash.
- **00:32-00:50:** If and only if approved, run the frozen full acquisition/custody workflow. Do not inspect held-out results. No browser, install, code edit, presentation build, or unrelated benchmark during collection.
- **00:50-00:55:** Independently verify the full artifact, freeze the calibration-derived policy artifact if that is part of the corrected preregistered workflow, then perform held-out evaluation/unseal exactly as specified. If there is insufficient time, preserve evidence and stop; do not rush interpretation.
- **00:55-01:00:** ZIP/hash the full evidence root and copy it to two independent destinations. Record final hashes and clean source status.
- **Optional only after evidence is safe:** launch the already-built fallback Sequential/OpenMP viewer. Do not attempt CUDA/Auto implementation. If a previously implemented/reviewed CUDA/Auto branch later exists, validate it in a separate scheduled session.

Abandon a failing CUDA/toolchain approach after the single rehearsed correction path or 10 minutes, whichever comes first. A CPU/OpenMP viewer demo is the fallback; it must never be presented as Stage 5 or CUDA/Auto evidence.

## 24. Explicit list of things NOT to change

Do not change the Newtonian force law, `G`, SI units, binary64 precision, Velocity Verlet sequence, dt, step counts, seeds, N grids, randomization seed/algorithm, repetitions, warmup definition, timer boundaries, frozen energy/angular gates, checksum semantics, Stage 3/4/conservation artifacts, historical claims, Solar initialization, MPI architecture, visualization physics separation, or the distinction between controlled evidence and diagnostics.

Do not repair evidence by deleting failed/partial rows, rerunning selected slow/noisy configurations, changing thresholds after seeing output, pooling Stage 3 with Stage 4, using pilot rows as full rows, tuning with held-out results, or editing a frozen checkout during measurement. Do not begin the large-N pilot, Stage 5B, CUDA/Auto integration, broad performance optimization, or visual polish as part of the audit/remediation itself.

Necessary remediation should be narrowly confined to Windows toolchain compatibility, evidence provenance/custody, runbook self-containment, and adversarial orchestration tests, followed by a new explicit freeze and pilot.

## 25. Final go/no-go criteria

### GO for clean-room RTX rehearsal only when all are true

- B-01 and B-02 are closed in a new reviewed/frozen revision.
- H-01 and H-02 are closed; M-01/M-02/M-07 negative tests pass.
- Exact driver/Toolkit/VS/SDK/CMake/Ninja/Python/Git tuple is selected and available offline.
- A fresh NVIDIA Windows build passes all Sequential/OpenMP/CUDA tests with requested/effective teams and `sm_89` confirmed.
- Build/test/binary/evidence manifests bind to the exact frozen source.
- Pilot evidence verifies and approval binds its manifest.
- Held-out output cannot be observed before a hashed policy freeze.

### GO for the one-hour RTX measurement visit only when all are true

- The full clean-room rehearsal above has passed without ad hoc edits.
- Measured pilot time plus projected full acquisition, verification, and dual copy fits within 60 minutes with at least 10 minutes contingency.
- Tool installers/dependencies/repository packet are pre-downloaded and hashed; no network is required.
- The operator has the exact command packet, abort thresholds, external evidence root, and two tested backup destinations.
- Stage 5 evidence is explicitly prioritized over visualization; CUDA/Auto development is excluded.

### Current decision

**NO-GO.** Current main is suitable for CPU/OpenMP/MPI/visualization clean-room validation, and historical evidence remains intact. The exact frozen Stage 5 experiment is not yet executable and defensible on the intended RTX workflow because B-01 and B-02 remain open.
