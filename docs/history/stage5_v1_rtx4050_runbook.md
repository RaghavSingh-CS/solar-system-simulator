# Historical Stage 5 v1.0 RTX 4050 runbook

This document is preserved for historical review only. Stage 5 v1.0 was
superseded and never executed. None of the commands below are current operator
instructions; use `docs/stage5_rtx4050_runbook.md` and its generated external
Stage 5 v2.0 deployment packet instead.

This runbook executes the frozen `stage5-v1.0` protocol prepared in Stage 5A.
It deliberately stops after the pilot. Do not run the full experiment until
the pilot has passed independent verification and a human has completed the
approval record.

Authoritative frozen source revision:

```text
4916b3c43298fa3f3bd15cb4af1698b5074abed3
```

The revision above is the Stage 5A preparation commit, not necessarily the
tip of `origin/main`. Use it exactly for both pilot and full measurements.

## A. Clone and check out the frozen source

Open an **x64 Native Tools Command Prompt for Visual Studio 2022**, start
PowerShell from it, and run:

```powershell
git clone https://github.com/RaghavSingh-CS/solar-system-simulator.git
Set-Location solar-system-simulator
git fetch origin
$stage5FrozenCommit = '4916b3c43298fa3f3bd15cb4af1698b5074abed3'
git checkout --detach $stage5FrozenCommit
git rev-parse HEAD
git status --short
```

The printed SHA must equal the frozen revision and status must be empty. Do not
develop, patch, pull, or commit on this checkout during measurement.

Choose an evidence directory outside the repository so Git remains clean:

```powershell
$stage5EvidenceRoot = 'C:\stage5-evidence'
$stage5EnvironmentId = 'rtx4050-win-REPLACE_TIMESTAMP'
New-Item -ItemType Directory -Path $stage5EvidenceRoot -Force | Out-Null
```

## B. Verify the toolchain

Install software manually before this procedure; the repository does not
install dependencies. Required components are Git, Python 3.10 or newer,
CMake 3.20 or newer, Ninja, Visual Studio 2022 C++ build tools, a current
NVIDIA driver, and a CUDA Toolkit compatible with that compiler and driver.

```powershell
git --version
python --version
cmake --version
ninja --version
cl
nvcc --version
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader
```

The selected device must report an RTX 4050. Record whether it is a laptop or
desktop variant; do not shorten the reported name. Resolve missing commands,
driver/toolkit incompatibility, or a different GPU before continuing.

## C. Inspect CPU topology and freeze OpenMP counts

```powershell
Get-CimInstance Win32_Processor |
  Select-Object Name,SocketDesignation,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```

Let P be the sum of physical cores and L the sum of logical processors across
the reported CPU packages. Use the
sorted unique set containing 1, powers of two below P, P, and L, omitting
values above L. For example, P=10 and L=16 gives `1,2,4,8,10,16`. Assign the
chosen list once and do not change it between pilot and full:

```powershell
$stage5OmpThreads = '1,2,4,8,REPLACE_PHYSICAL,REPLACE_LOGICAL'
```

## D. Configure and build Release binaries

RTX 4050 GPUs use compute capability 8.9. The device query and later benchmark
metadata independently confirm the runtime device.

```powershell
cmake -S . -B build-stage5 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON `
  -DNBODY_ENABLE_OPENMP=ON `
  -DNBODY_ENABLE_CUDA=ON
cmake --build build-stage5 --config Release
```

Do not substitute a Debug build. If configuration or compilation fails, stop
and diagnose it outside a measurement run; do not patch the frozen checkout.

## E. Run validation and retain its log

```powershell
$stage5CTestLog = Join-Path $stage5EvidenceRoot 'stage5-ctest.txt'
ctest --test-dir build-stage5 -C Release --output-on-failure 2>&1 |
  Tee-Object -FilePath $stage5CTestLog
```

The log must show all foundation, OpenMP, and CUDA tests passing. The runner
will reject a log without `100% tests passed` and `0 tests failed`.

## F. Run preflight

```powershell
python -B scripts/run_stage5.py --preflight `
  --output (Join-Path $stage5EvidenceRoot 'preflight') `
  --omp-threads $stage5OmpThreads `
  --source-commit $stage5FrozenCommit `
  --environment-id $stage5EnvironmentId `
  --ctest-log $stage5CTestLog
```

Preflight fails on dirty/wrong Git state, missing executables, failed tests,
unavailable NVIDIA tooling, RTX 4050 device mismatch, or an impossible thread
request. Inspect `preflight/preflight.json`, including OS/CPU/RAM, exact GPU,
driver, CUDA Toolkit, compiler, CMake/Ninja, Release flags, and executable
hashes.

## G. Run the pilot only

```powershell
python -B scripts/run_stage5.py --pilot `
  --output (Join-Path $stage5EvidenceRoot 'pilot') `
  --omp-threads $stage5OmpThreads `
  --source-commit $stage5FrozenCommit `
  --environment-id $stage5EnvironmentId `
  --experiment-id 'stage5-pilot-rtx4050' `
  --ctest-log $stage5CTestLog `
  --seed 24301 --order-seed 20260926 --device 0 --expected-gpu 'RTX 4050'
```

This runs N=`64,512,4096`, 10 steps, and three complete randomized blocks.
Pilot output is not calibration evidence.

## H. Verify the pilot independently

```powershell
$stage5PilotVerification = Join-Path $stage5EvidenceRoot 'pilot-verification.json'
python -B scripts/verify_stage5_evidence.py `
  (Join-Path $stage5EvidenceRoot 'pilot') `
  --report $stage5PilotVerification
```

The verifier must report `verification_status: pass`.

## I. Mandatory stop and pilot inspection

**STOP HERE. Do not run the full command yet.** Review:

- exact RTX 4050 identity, VRAM, driver, compute capability, and SM count;
- clean frozen source and Release/OpenMP/CUDA build metadata;
- all CTests passing;
- requested/effective OpenMP counts matching;
- complete seeded schedule and exact row counts;
- positive finite metrics and CUDA transfer-inclusive >= device-compute;
- all correctness/conservation gates passing;
- repeat checksums and CPU cross-backend checksums matching;
- CV/outliers, GPU temperature/power context, throttling indications, and the
  estimated duration of the 100-step, 11-block full matrix.

If the pilot indicates a matrix, implementation, toolchain, or protocol defect,
do not improvise on the RTX machine. Return to Stage 5A, revise deliberately,
freeze a new source commit, and repeat the pilot.

## J. Record human pilot approval

Only after a satisfactory review, copy and edit the approval template outside
the repository:

```powershell
$stage5PilotApproval = Join-Path $stage5EvidenceRoot 'pilot-approval.json'
Copy-Item docs/stage5_pilot_approval.template.json $stage5PilotApproval
notepad $stage5PilotApproval
```

Replace every placeholder, preserve the exact source/environment/thread list,
and describe the inspection outcome. The full runner rejects missing or
inconsistent approval fields.

## K. Run the full experiment after approval

```powershell
python -B scripts/run_stage5.py --full `
  --output (Join-Path $stage5EvidenceRoot 'full') `
  --omp-threads $stage5OmpThreads `
  --source-commit $stage5FrozenCommit `
  --environment-id $stage5EnvironmentId `
  --experiment-id 'stage5-full-rtx4050' `
  --ctest-log $stage5CTestLog `
  --pilot-verification $stage5PilotVerification `
  --pilot-approval $stage5PilotApproval `
  --seed 24301 --order-seed 20260926 --device 0 --expected-gpu 'RTX 4050'
```

The full run uses the frozen 14-size grid, 100 steps, and 11 blocks. Do not
reuse partial output after failure; retain it for diagnosis and restart into a
new empty directory only after resolving the cause.

## L. Verify and package full evidence

```powershell
$stage5FullVerification = Join-Path $stage5EvidenceRoot 'full-verification.json'
python -B scripts/verify_stage5_evidence.py `
  (Join-Path $stage5EvidenceRoot 'full') `
  --report $stage5FullVerification
Get-FileHash -Algorithm SHA256 `
  (Join-Path $stage5EvidenceRoot 'full\manifest.json'), `
  $stage5FullVerification
Compress-Archive -Path `
  (Join-Path $stage5EvidenceRoot 'preflight'), `
  (Join-Path $stage5EvidenceRoot 'pilot'), `
  $stage5PilotVerification, $stage5PilotApproval, `
  (Join-Path $stage5EvidenceRoot 'full'), `
  $stage5FullVerification, $stage5CTestLog `
  -DestinationPath (Join-Path $stage5EvidenceRoot 'stage5-rtx4050-evidence.zip')
Get-FileHash -Algorithm SHA256 (Join-Path $stage5EvidenceRoot 'stage5-rtx4050-evidence.zip')
```

Do not derive or implement the selector during collection. Preserve the
verified raw package for later Stage 5 analysis and repository integration.
