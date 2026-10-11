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

Ignored `.tools/` is about 272.4 MiB and contains the verified portable CMake
archive/install, Python CMake runtime, and GLAD-generation dependencies. It is
a local toolchain/cache, not scientific evidence, but deleting it would remove
the current offline CMake fallback. The ignored 0.10 MiB
`solar-system-simulator.zip` is a grandfathered early source archive with
SHA-256 `00fc9f3922ceb09647668a877a6193e10c7758338690fe945f16810e8e1b00fd`;
its historical value must be investigated before deletion.

## Historical root build inventory

Sizes are approximate byte sums observed on 2026-10-12. “Regenerable” means
the build products themselves can be reconstructed; it does not authorize
deletion of unique outputs stored inside the directory. Historical references
in the ledger or scientific protocols remain valid records and must not be
rewritten to pretend those runs used the new presets.

| Directory | MiB | Regenerable | Tracked reference | Unique evidence found | Recommendation |
|---|---:|---|---|---|---|
| `build-catalog-custody-cpu` | 8.55 | Yes | No | No | Delete candidate after approval |
| `build-catalog-custody-openmp` | 8.96 | Yes | No | No | Delete candidate after approval |
| `build-catalog-dataset-cpu` | 8.54 | Yes | No | No | Delete candidate after approval |
| `build-catalog-dataset-openmp` | 8.95 | Yes | No | No | Delete candidate after approval |
| `build-catalog-foundation-cpu` | 7.56 | Yes | No | No | Delete candidate after approval |
| `build-catalog-scenario-cpu` | 20.47 | Yes | No | No | Delete candidate after approval |
| `build-catalog-scenario-openmp` | 20.86 | Yes | No | No | Delete candidate after approval |
| `build-conservation-validation` | 1.80 | Yes | Historical study docs use a different named build command | No | Delete candidate after approval |
| `build-counterfactual-commitgate-cpu` | 22.48 | Yes | No | No | Delete candidate after approval |
| `build-counterfactual-commitgate-openmp` | 21.63 | Yes | No | No | Delete candidate after approval |
| `build-counterfactual-cpu` | 7.64 | Yes | No | No | Delete candidate after approval |
| `build-counterfactual-full-cpu` | 21.71 | Yes | No | No | Delete candidate after approval |
| `build-counterfactual-full-openmp` | 7.14 | Yes | No | No | Delete candidate after approval |
| `build-counterfactual-gui-cpu` | 22.76 | Yes | No | No | Delete candidate after approval |
| `build-counterfactual-gui-openmp` | 22.95 | Yes | No | No | Delete candidate after approval |
| `build-counterfactual-openmp` | 7.84 | Yes | No | No | Delete candidate after approval |
| `build-counterfactual-p1-cpu` | 21.17 | Yes | Ledger validation record | No | Delete candidate after approval; retain ledger |
| `build-counterfactual-p1-openmp` | 21.63 | Yes | Ledger validation record | No | Delete candidate after approval; retain ledger |
| `build-counterfactual-remediation-cpu` | 22.72 | Yes | Ledger validation record | No | Delete candidate after approval; retain ledger |
| `build-counterfactual-remediation-openmp` | 23.18 | Yes | Ledger validation record | No | Delete candidate after approval; retain ledger |
| `build-counterfactual-safety-cpu` | 21.18 | Yes | No | No | Delete candidate after approval |
| `build-counterfactual-safety-openmp` | 21.63 | Yes | No | No | Delete candidate after approval |
| `build-cuda-local-probe` | 0.52 | Yes | No | No; failed/unavailable probe only | Delete candidate after approval |
| `build-cuda-visualizer-cpu` | 21.30 | Yes | No | No | Delete candidate after approval |
| `build-cuda-visualizer-cuda-probe` | 0.57 | Yes | No | No; failed/unavailable probe only | Delete candidate after approval |
| `build-cuda-visualizer-sequential` | 18.51 | Yes | No | No | Delete candidate after approval |
| `build-openmp` | 1.90 | Yes | Stage 3 docs/data/scripts | No unique file found; tracked evidence is under `data/benchmarks/stage3_cpu/` | Retain until historical command audit, then delete candidate |
| `build-remediation-cpu` | 20.46 | Yes | No | No | Delete candidate after approval |
| `build-remediation-openmp` | 20.91 | Yes | No | No | Delete candidate after approval |
| `build-sequential` | 1.78 | Yes | Conservation docs/script | No unique file found; tracked evidence is under `data/experiments/conservation/` | Retain until historical command audit, then delete candidate |
| `build-stage3-pilot` | 0.04 | **No, as a whole** | Layout checker and ledger custody record | **Yes:** `affinity_probe.txt`, `environment.json`, `raw_timings.csv`, `summary.csv`; hashes differ from tracked Stage 3 evidence | **Retain and investigate/archive before deletion** |
| `build-stage5a-cuda-probe` | 0.52 | Yes | Historical ledger context | No; failed/unavailable probe only | Delete candidate after approval; preserve ledger |
| `build-stage5a-openmp` | 2.21 | Yes | Historical ledger context | No | Delete candidate after approval; preserve ledger |
| `build-stage5a-sequential` | 1.80 | Yes | Historical ledger context | No | Delete candidate after approval; preserve ledger |
| `build-stage5v2-cuda-unavailable` | 0.52 | Yes | Historical ledger context | No; failed/unavailable probe only | Delete candidate after approval; preserve ledger |
| `build-stage5v2-dev` | 2.81 | Yes | Historical ledger context | No | Delete candidate after approval; never alter frozen external Stage 5 custody |
| `build-stage6a-cpu` | 1.80 | Yes | Historical ledger context | No | Delete candidate after approval |
| `build-stage6a-openmp` | 2.21 | Yes | Historical ledger context | No | Delete candidate after approval |
| `build-stage6a-vis` | 132.04 | Yes | No | No; includes fetched dependency/build cache | Delete candidate after dependency-cache review |
| `build-stage6a-visual2` | 139.42 | Yes | No | No; includes fetched dependency/build cache | Delete candidate after dependency-cache review |
| `build-stage6b-cpu` | 2.00 | Yes | Historical ledger context | No | Delete candidate after approval |
| `build-stage6b-openmp` | 2.41 | Yes | Historical ledger context | No | Delete candidate after approval |
| `build-stage6b-visual` | 269.77 | Yes | Historical visualization record | No; includes fetched dependency/build cache | Delete candidate after dependency-cache review |
| `build-stage6c-cpu` | 2.07 | Yes | Historical ledger context | No | Delete candidate after approval |
| `build-stage6c-openmp` | 2.48 | Yes | Historical ledger context | No | Delete candidate after approval |
| `build-stage6c-visual` | 270.00 | Yes | Historical visualization record | No scientific evidence; currently supplies reusable pinned GLFW/GLM/ImGui source trees | **Retain until canonical dependency acquisition is independent** |
| `build-stage6c1-cpu` | 2.07 | Yes | Historical ledger context | No | Delete candidate after approval |
| `build-stage6c1-openmp` | 2.48 | Yes | Historical ledger context | No | Delete candidate after approval |
| `build-stage6c1-visual` | 13.20 | Yes | Historical visualization commands | No | Delete candidate after approval; retain documentation |
| `build-stage6d-cpu` | 2.40 | Yes | Historical ledger context | No | Delete candidate after approval |
| `build-stage6d-mpi` | 3.77 | Yes | MPI documentation and pilot defaults | No unique file found | Retain until historical MPI command audit, then delete candidate |
| `build-stage6d-openmp` | 2.79 | Yes | MPI/large-N documentation and pilot defaults | No unique file found | Retain until historical pilot command audit, then delete candidate |
| `build-stage6d-pilot-smoke` | 0.01 | **No, as a whole** | Layout checker and ledger custody record | **Yes:** environment, raw results, and four backend logs | **Retain and investigate/archive before deletion** |
| `build-stage6d-visual` | 13.78 | Yes | Historical visualization commands | No | Delete candidate after approval; retain documentation |

## Evidence-sensitive cleanup procedure

No directory in the table may be removed from this audit alone. A separately
approved cleanup must:

1. Re-resolve each exact candidate path and confirm it is an immediate child of
   this repository; never use a broad `build-*` deletion glob.
2. Re-scan for CSV, JSON, logs, manifests, environment records, benchmark
   output, and files not produced by CMake.
3. Preserve and review `build-stage3-pilot` and
   `build-stage6d-pilot-smoke`; archive accepted evidence with hashes and
   provenance before deletion is considered.
4. Keep `build-stage6c-visual` until pinned graphics dependencies can be fetched
   or supplied from a separately approved cache without relying on that tree.
5. Preserve `.tools/`, versioned `data/`, external Stage 5 deployment/evidence
   roots, sibling cleanrooms, and every frozen historical protocol.
6. Remove only explicitly approved paths, then remove those names from
   `tools/repository_layout_baseline.json` in the same reviewed change.
7. Run the layout checker and canonical CPU/OpenMP suites afterward.

The checker never deletes or moves data.
