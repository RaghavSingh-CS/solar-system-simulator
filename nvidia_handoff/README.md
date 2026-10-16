# RTX 4050 NVIDIA handoff

This directory is the Windows operator package for compiling and validating
the existing CUDA implementation on an NVIDIA RTX 4050 Laptop GPU. The CUDA
source is independently source-reviewed, but it has **not** been compiled with
nvcc or executed on NVIDIA hardware in the committed development record. A
successful build is not a successful GPU validation.

## Five-minute demonstration path

Run these from an **x64 Native Tools PowerShell for Visual Studio 2022**. The
demo launcher stops at the first failure and always attempts bounded result
collection. Run build and correctness preparation before the presentation;
the live five minutes are described in [`DEMO.md`](DEMO.md).

```powershell
git clone --branch feature/cuda-visualizer https://github.com/RaghavSingh-CS/solar-system-simulator.git
Set-Location solar-system-simulator
$run = Join-Path $PWD "nvidia_handoff/results/rtx4050-$(Get-Date -Format yyyyMMdd-HHmmss)"
.\nvidia_handoff\scripts\run_demo.ps1 -RunDirectory $run -Device 0 -Scenario solar
```

This path performs RTX/toolchain preflight, builds CUDA Release only, runs the
CUDA Direct and CUDA Barnes--Hut correctness suites with zero skips permitted,
launches the staged CUDA Direct Solar viewer, and creates `$run.zip` after the
viewer closes. A full engineering run remains available through `run_all.ps1`.
Never append stages after an archive exists; use a new run directory.

## Required software

- Windows 10/11 x64 and an RTX 4050 Laptop GPU reporting compute capability
  8.9 through `nvidia-smi`.
- A compatible NVIDIA driver and CUDA Toolkit with native `compute_89` support
  (CUDA 11.8 or newer is required; the exact installed tuple is recorded, not
  assumed compatible).
- Visual Studio 2022 C++ desktop workload, Windows SDK, and `cl.exe` available
  from the x64 Native Tools shell.
- CMake 3.23 or newer, Ninja, Git, and Python 3.
- Internet access for the pinned GLFW, GLM, and ImGui FetchContent sources, or
  the repository's separately provisioned pinned dependency cache.

The preflight rejects missing tools, the wrong GPU/device, a compute capability
other than 8.9, nvcc without `compute_89`, dirty source, or CMake/Ninja resolved
from MSYS/MinGW. It does not alter global environment variables or require
administrator rights.

## What the launchers do

| Script | Purpose |
|---|---|
| `00_preflight.ps1` | Records Git, GPU, driver, VRAM, nvcc, compiler, CMake, Ninja, and device compatibility. |
| `01_build.ps1` | Builds canonical CPU, OpenMP, and CUDA Release presets; CUDA uses `CMAKE_CUDA_ARCHITECTURES=89` and OpenMP enabled. Verifies `CMakeCache.txt`, stages binaries, and hashes them. |
| `02_cpu_baseline.ps1` | Runs complete canonical CPU and OpenMP CTest baselines. |
| `03_cuda_direct.ps1` | Runs genuine CUDA Direct, visualization-controller, and paired-counterfactual tests. Any skip is failure. |
| `04_cuda_barnes_hut.ps1` | Runs CUDA Barnes-Hut force and transactional Velocity Verlet tests. |
| `05_cuda_numerics.ps1` | Repeats the combined numerical-agreement, rollback, repeatability, and metadata gate. |
| `06_cuda_resources.ps1` | Executes the test binaries containing resource limits, injected host failures, terminal ownership, cleanup, and repeated-lifetime coverage. |
| `07_stage5_pilot.ps1` | Always fails closed: automated Stage 5 execution is disabled pending separate reconciliation of the authoritative packet and v2 runner commands. |
| `08_viewer.ps1` | Launches the viewer with CUDA Direct for manual inspection. |
| `09_collect_results.ps1` | Transactionally creates a bounded, allowlisted ZIP from one marked run under `results/`; failed collection leaves the run unchanged. |
| `run_all.ps1` | Runs stages 00--06 in order, optionally launches the viewer, and collects results; it never invokes Stage 5. |
| `run_demo.ps1` | Short path: preflight, CUDA-only build, CUDA correctness, CUDA Direct viewer, and result collection. |

Each script locates the repository from its own path, quotes path arguments,
records commands and exit codes, and refuses to overwrite an existing stage
record. Run directories must be marked direct children of
`nvidia_handoff/results`; arbitrary external directories and reparse points are
rejected. Stages are independently runnable once their prerequisites pass.

## Build outputs and executable inventory

Ordinary builds use only the canonical `build/cpu-release`,
`build/openmp-release`, and `build/cuda-release` directories. CUDA is built on
the NVIDIA machine; binaries must not be transferred from another host.

The expected target inventory is checked into `MANIFEST.json`. After a build,
the actual executables are copied to:

```text
nvidia_handoff/bin/<run-id>/<configuration>/
```

The generated `executables.json` records each original target and build path,
SHA-256, size, Release configuration, architecture, and validation status.
Adjacent build-produced DLLs are copied without flattening configuration
directories. Every staged binary is re-hashed against its manifest and matching
build output before execution or collection. CUDA and MSVC system runtimes
remain provided by the validated host toolchain.

## Running stages individually

```powershell
.\nvidia_handoff\scripts\00_preflight.ps1 -RunDirectory $run -Device 0
.\nvidia_handoff\scripts\01_build.ps1 -RunDirectory $run
.\nvidia_handoff\scripts\02_cpu_baseline.ps1 -RunDirectory $run
.\nvidia_handoff\scripts\03_cuda_direct.ps1 -RunDirectory $run
.\nvidia_handoff\scripts\04_cuda_barnes_hut.ps1 -RunDirectory $run
.\nvidia_handoff\scripts\05_cuda_numerics.ps1 -RunDirectory $run
.\nvidia_handoff\scripts\06_cuda_resources.ps1 -RunDirectory $run
```

CUDA CTests use skip return code 77 only when successful enumeration finds zero
devices. In this RTX handoff, **any skipped or not-run GPU test is failure**.
Enumeration, driver, kernel, allocation, synchronization, or cleanup errors are
also failures; they are never converted into a pass or CPU fallback.

## Frozen Stage 5 boundary

Automated Stage 5 execution is disabled in this handoff because the
authoritative deployment packet and Stage 5 v2 runner commands are currently
inconsistent. `07_stage5_pilot.ps1` always fails with that explanation, and
neither master launcher calls it. Use the separate
[`docs/stage5_rtx4050_runbook.md`](../docs/stage5_rtx4050_runbook.md) only after
explicit review. No frozen Stage 5 script, protocol, threshold, or evidence is
changed by this package.

## Viewer inspection

The implemented viewer CUDA path is **Direct only**:

```powershell
.\nvidia_handoff\scripts\08_viewer.ps1 -RunDirectory $run -Scenario solar -Device 0
```

Manually inspect the legacy Solar and catalog scenes, pause/reset/step,
backend metadata, camera behavior, diagnostics, and a CUDA Direct
counterfactual start/cancel/restart. Record observations in a new text file
under the run directory before collecting it. Do not select or claim CUDA
Barnes-Hut in the viewer: that integration is intentionally disabled.

## Failures and result return

On failure, stop GPU experimentation and retain the marked run directory. The
master launchers already attempt collection; it can also be requested once:

```powershell
.\nvidia_handoff\scripts\09_collect_results.ps1 -RunDirectory $run
```

Send back the resulting ZIP. Collection accepts only small handoff records,
logs, statuses, diagnostics, numerical text/CSV/JSON, and executable hashes.
It rejects external directories, reparse points, unrecognized files, more than
512 files, any file over 32 MiB, or more than 128 MiB total. Build trees,
dependency caches, credentials, and unrelated files are never included.

After a genuine CUDA synchronization, device-selection, or fatal runtime
failure, record diagnostics and restart the application/process before further
GPU work. The implementation deliberately performs no automatic device-wide
reset, and recovery from a fatal device error is not guaranteed.

## Explicit limitations

- CUDA compilation and runtime correctness remain unverified until this package
  succeeds on the RTX 4050 with zero unexpected skips.
- CUDA Barnes-Hut is implemented only for headless force and Velocity Verlet
  validation. It is not enabled in the viewer or counterfactual engine.
- The viewer's CUDA counterfactual path is Direct only.
- No automatic backend selection exists.
- This package runs no 50K/100K or uncontrolled stress workload.
- A process-launch smoke is not human visual validation.
- Automated Stage 5 execution is disabled; it requires a separate reviewed workflow.
