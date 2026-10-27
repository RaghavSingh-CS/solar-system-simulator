# Five-minute RTX 4050 academic demonstration

For the October 29 GUI correction, first use the
[candidate build and visual-check instructions](../docs/presentation_demo.md).
This correction remains uncommitted until RTX verification; the clean-source
handoff preflight cannot certify a dirty candidate checkout. The updated Demo
page exposes live controls and the Comparison page displays baseline versus
body-removed paired endpoints. Select OpenMP explicitly before Barnes--Hut.
CUDA Direct synchronous batches are capped at four fixed steps. Inspect actual
RTX frame pacing before making any performance statement.

Prepare before the presentation from an x64 VS2022 Native Tools PowerShell:

```powershell
$run = Join-Path $PWD "nvidia_handoff/results/rtx4050-demo-$(Get-Date -Format yyyyMMdd-HHmmss)"
.\nvidia_handoff\scripts\run_demo.ps1 -RunDirectory $run -Device 0 -Scenario solar
```

The launcher must complete preflight, the CUDA Release build, CUDA Direct
tests, and CUDA Barnes--Hut force/integration tests without skips before it
opens the viewer. If compilation or any test fails, do not demonstrate CUDA as
working: close the workflow, retain the ZIP/logs, and report the failure.

## Live presentation

1. **0:00--0:45 — provenance.** Show `preflight.json`: RTX 4050 identity,
   compute capability 8.9, driver, nvcc, MSVC, CMake, and exact Git SHA.
2. **0:45--1:30 — correctness gate.** Show passed status for
   `03_cuda_direct` and `04_cuda_barnes_hut`. Explain that skip 77 is failure
   on this target, so these records require genuine GPU execution.
3. **1:30--4:15 — viewer.** Use the legacy Solar scene with backend **CUDA**
   and solver **Direct**. Demonstrate pause, single-step, resume, reset, camera
   movement, and the active backend/solver metadata. Do not select CUDA
   Barnes--Hut in the viewer; that integration is intentionally unavailable.
4. **4:15--4:45 — performance display.** Show only the live viewer's displayed
   rendering FPS and simulation steps/second. Describe them as a single live
   observation, not a controlled benchmark, crossover, or 50K capability.
5. **4:45--5:00 — limitations and custody.** State that CUDA validation applies
   only to the executed tests and host. CUDA Barnes--Hut viewer execution,
   CUDA Barnes--Hut counterfactuals, automatic backend selection, Stage 5, and
   RTX large-N claims remain unavailable. Close the viewer so the launcher can
   create the bounded result ZIP.

The solid/dashed counterfactual presentation may be discussed, but the current
CUDA counterfactual path is Direct only. A successful process launch alone is
not human visual validation; record any visual observations in
`demo_observations.txt` inside the marked run directory before collection.
