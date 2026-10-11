# Repository layout and build custody

## Canonical layout

All ordinary builds use `CMakePresets.json` and remain below the ignored
`build/` root:

| Preset | Build directory | Purpose |
|---|---|---|
| `cpu-release` | `build/cpu-release/` | Sequential CPU viewer and tests; no NVIDIA dependency |
| `openmp-release` | `build/openmp-release/` | Sequential/OpenMP viewer and tests |
| `cuda-release` | `build/cuda-release/` | Optional CUDA viewer and tests; toolkit/device dependent |

Temporary approved configurations belong under `build/scratch/<purpose>/`.
Scientific results and provenance never belong in `build/`; controlled evidence
belongs under the applicable versioned `data/` protocol path. Stage 5 reruns
must continue to use their exact frozen protocol, external cleanroom, and
evidence-root instructions rather than these development presets.

Use:

```powershell
.\scripts\test-local.ps1 -Configuration CPU
.\scripts\test-local.ps1 -Configuration OpenMP
.\scripts\run-visualizer.ps1
.\scripts\run-visualizer.ps1 -Backend OpenMP
```

Use `-PrintCommand` to inspect the exact executable/backend selection without
opening a window.

`CMakeUserPresets.json.example` demonstrates optional local compiler, Ninja,
and offline FetchContent source overrides through environment variables. Copy
it to ignored `CMakeUserPresets.json` only when local overrides are required;
never commit machine-specific paths, credentials, or secrets.

The optional persistent offline dependency cache lives under
`.tools/dependencies/`. Its tracked provenance is
`tools/dependency_cache_manifest.json`; `scripts/verify_dependency_cache.py`
requires clean Git checkouts at the exact GLFW 3.4, GLM 1.0.1, and Dear ImGui
1.92.9b revisions pinned by `CMakeLists.txt`. `scripts/test-local.ps1` uses the
cache only when every entry is present and verified. If no cache is installed,
normal pinned FetchContent acquisition remains available; a partial or modified
cache fails explicitly. No historical build tree is a dependency of the
canonical workflow after this migration.

The ignored cache is local state and is not transported by Git. A different
machine must receive all three clean cache directories from a trusted verified
copy before attempting an offline configure, then run
`python -B scripts/verify_dependency_cache.py --require-present`. Otherwise its
first configure requires network access to acquire the same pinned revisions.

`scripts/check_repository_layout.py` is read-only. It rejects new root
`build-*` directories, tracked build binaries, unexpected root generator
directories, and untracked generated validation artifacts outside approved
locations. CTest runs it whenever Python is available. The checker permits only
the named legacy items in `tools/repository_layout_baseline.json`; absence is
allowed so approved cleanup can proceed incrementally.

## Immediate root inventory

Tracked project roots are `backends/`, `data/`, `docs/`, `include/`, `scripts/`,
`src/`, `tests/`, `third_party/`, `tools/`, and `visualization/`, plus the CMake,
Git, README, preset, and developer-policy files. `data/` contains versioned
scientific inputs, raw source custody, and retained experiment evidence and is
not disposable. `third_party/` contains the committed generated GLAD loader.

Ignored `.tools/` contains the verified portable CMake archive/install, Python
CMake runtime, GLAD-generation dependencies, and the verified persistent
visualization dependency cache. It is a local toolchain/cache, not scientific
evidence, but deleting it would remove the current offline build fallback. The
ignored 0.10 MiB
`solar-system-simulator.zip` is a grandfathered early source archive with
SHA-256 `00fc9f3922ceb09647668a877a6193e10c7758338690fe945f16810e8e1b00fd`;
its historical value must be investigated before deletion.

## Historical root build inventory

Sizes are approximate byte sums observed on 2026-10-12. “Regenerable” means
the build products themselves can be reconstructed; it does not authorize
deletion of unique outputs stored inside the directory. Historical references
in the ledger or scientific protocols remain valid records and must not be
rewritten to pretend those runs used the new presets.

The independent final audit classifications and exact byte counts are recorded
in `tools/historical_build_cleanup_plan.json`. Its 48 `SAFE_TO_DELETE` entries
total 1,362,737,959 bytes (1,299.61 MiB); this does not authorize deletion.
The classifications below use that final audit, which supersedes the earlier
provisional “delete candidate” language.

| Directory | MiB | Regenerable | Tracked reference | Unique evidence found | Recommendation |
|---|---:|---|---|---|---|
| `build-catalog-custody-cpu` | 8.55 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-catalog-custody-openmp` | 8.96 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-catalog-dataset-cpu` | 8.54 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-catalog-dataset-openmp` | 8.95 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-catalog-foundation-cpu` | 7.56 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-catalog-scenario-cpu` | 20.47 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-catalog-scenario-openmp` | 20.86 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-conservation-validation` | 1.80 | Yes | Historical study docs use a different named build command | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-commitgate-cpu` | 22.48 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-commitgate-openmp` | 21.63 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-cpu` | 7.64 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-full-cpu` | 21.71 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-full-openmp` | 7.14 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-gui-cpu` | 22.76 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-gui-openmp` | 22.95 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-openmp` | 7.84 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-p1-cpu` | 21.17 | Yes | Ledger validation record | No | **SAFE_TO_DELETE** after approval; retain ledger |
| `build-counterfactual-p1-openmp` | 21.63 | Yes | Ledger validation record | No | **SAFE_TO_DELETE** after approval; retain ledger |
| `build-counterfactual-remediation-cpu` | 22.72 | Yes | Ledger validation record | No | **SAFE_TO_DELETE** after approval; retain ledger |
| `build-counterfactual-remediation-openmp` | 23.18 | Yes | Ledger validation record | No | **SAFE_TO_DELETE** after approval; retain ledger |
| `build-counterfactual-safety-cpu` | 21.18 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-counterfactual-safety-openmp` | 21.63 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-cuda-local-probe` | 0.52 | Yes | No | No; failed/unavailable probe only | **SAFE_TO_DELETE** after approval |
| `build-cuda-visualizer-cpu` | 21.30 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-cuda-visualizer-cuda-probe` | 0.57 | Yes | No | No; failed/unavailable probe only | **SAFE_TO_DELETE** after approval |
| `build-cuda-visualizer-sequential` | 18.51 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-openmp` | 1.90 | Yes | Stage 3 docs/data/scripts | No unique result file, but the exact binary path is part of retained Stage 3 provenance | **RETAIN_PROVENANCE** |
| `build-remediation-cpu` | 20.46 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-remediation-openmp` | 20.91 | Yes | No | No | **SAFE_TO_DELETE** after approval |
| `build-sequential` | 1.78 | Yes | Conservation docs/script | No unique result file, but the exact binary path is part of retained conservation-study provenance | **RETAIN_PROVENANCE** |
| `build-stage3-pilot` | 0.04 | **No, as a whole** | Layout checker and ledger custody record | **Yes:** `affinity_probe.txt`, `environment.json`, `raw_timings.csv`, `summary.csv`; hashes differ from tracked Stage 3 evidence | **ARCHIVE_FIRST** |
| `build-stage5a-cuda-probe` | 0.52 | Yes | Historical ledger context | No; failed/unavailable probe only | **SAFE_TO_DELETE** after approval; preserve ledger |
| `build-stage5a-openmp` | 2.21 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval; preserve ledger |
| `build-stage5a-sequential` | 1.80 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval; preserve ledger |
| `build-stage5v2-cuda-unavailable` | 0.52 | Yes | Historical ledger context | No; failed/unavailable probe only | **SAFE_TO_DELETE** after approval; preserve ledger |
| `build-stage5v2-dev` | 2.81 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval; never alter frozen external Stage 5 custody |
| `build-stage6a-cpu` | 1.80 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval |
| `build-stage6a-openmp` | 2.21 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval |
| `build-stage6a-vis` | 132.04 | Yes | No | No; dependency versions are independently available and current GLAD is vendored | **SAFE_TO_DELETE** after approval |
| `build-stage6a-visual2` | 139.42 | Yes | No | No; dependency versions are independently available and current GLAD is vendored | **SAFE_TO_DELETE** after approval |
| `build-stage6b-cpu` | 2.00 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval |
| `build-stage6b-openmp` | 2.41 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval |
| `build-stage6b-visual` | 269.77 | Yes | Historical visualization record | No; exact current dependency revisions are verified in `.tools/dependencies/` | **SAFE_TO_DELETE** after approval |
| `build-stage6c-cpu` | 2.07 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval |
| `build-stage6c-openmp` | 2.48 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval |
| `build-stage6c-visual` | 270.00 | Yes | Historical visualization record | No scientific evidence; its clean pinned sources were copied and independently verified under `.tools/dependencies/` | **SAFE_TO_DELETE** after approval |
| `build-stage6c1-cpu` | 2.07 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval |
| `build-stage6c1-openmp` | 2.48 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval |
| `build-stage6c1-visual` | 13.20 | Yes | Historical visualization commands | No | **SAFE_TO_DELETE** after approval; retain documentation |
| `build-stage6d-cpu` | 2.40 | Yes | Historical ledger context | No | **SAFE_TO_DELETE** after approval |
| `build-stage6d-mpi` | 3.77 | Yes | MPI documentation and pilot defaults | No unique result file, but the exact binary path supports unique smoke evidence | **RETAIN_PROVENANCE** |
| `build-stage6d-openmp` | 2.79 | Yes | MPI/large-N documentation and pilot defaults | No unique result file, but the exact binary path supports unique smoke evidence | **RETAIN_PROVENANCE** |
| `build-stage6d-pilot-smoke` | 0.01 | **No, as a whole** | Layout checker and ledger custody record | **Yes:** environment, raw results, and four backend logs | **ARCHIVE_FIRST** |
| `build-stage6d-visual` | 13.78 | Yes | Historical visualization commands | No | **SAFE_TO_DELETE** after approval; retain documentation |

## Evidence-sensitive cleanup procedure

No directory in the table may be removed from this audit alone. A separately
approved cleanup must:

1. Re-resolve each exact candidate path and confirm it is an immediate child of
   this repository; never use a broad `build-*` deletion glob.
2. Re-scan for CSV, JSON, logs, manifests, environment records, benchmark
   output, and files not produced by CMake.
3. Preserve and review `build-stage3-pilot` and
   `build-stage6d-pilot-smoke`; archive their exact contents to the proposed
   external evidence root with the recorded file hashes before deletion is
   considered.
4. Verify the persistent dependency cache and confirm canonical CMake caches no
   longer reference `build-stage6c-visual` before approving its deletion.
5. Preserve `.tools/`, versioned `data/`, external Stage 5 deployment/evidence
   roots, sibling cleanrooms, and every frozen historical protocol.
6. Remove only explicitly approved paths, then remove those names from
   `tools/repository_layout_baseline.json` in the same reviewed change.
7. Run the layout checker and canonical CPU/OpenMP suites afterward.

The active baseline intentionally remains unchanged until cleanup actually
occurs. `tools/historical_build_cleanup_plan.json` supplies the exact deletion
list, planned baseline removals, archival destinations, checksums, and recovery
instructions. `build-openmp`, `build-sequential`, `build-stage6d-mpi`, and
`build-stage6d-openmp` remain in place as provenance; they are not part of the
safe deletion total.

The checker never deletes or moves data.
