# Stage 5 v2.0 Windows RTX 4050 runbook

Stage 5 v1.0 was superseded and never executed. Its obsolete operator material
is retained only in
[`docs/history/stage5_v1_rtx4050_runbook.md`](history/stage5_v1_rtx4050_runbook.md).
Do not use it for an RTX session.

This tracked document defines the prospective v2.0 clean-room procedure. The
authoritative operator commands are the external deployment packet generated
*after* the final remediation commit:

```powershell
python -B scripts/create_stage5_deployment_packet.py `
  --source-sha <FINAL_40_CHARACTER_SHA> `
  --output C:\stage5-v2-deployment-packet
```

The generated packet contains that exact, non-self-referential source SHA,
file hashes, an unmistakably uninitialized approval template, abort criteria,
and the full v2.0 commands. Never infer the source SHA or OpenMP thread list
from this tracked template. The thread list must come from automatic topology
discovery in the controlled build manifest.

The v2.0 workflow is:

```text
fresh exact-SHA clone
 -> native Windows/MSVC/CUDA preflight and topology discovery
 -> absent Release build directory
 -> source-embedded build + controlled CTest + build manifest
 -> Stage 5 preflight
 -> pilot-only acquisition
 -> independent pilot verification
 -> manifest-bound human approval
 -> STOP for the first NVIDIA rehearsal
```

For a separately authorized final collection session only, continue with
calibration-only acquisition, independent verification, deterministic policy
generation and freeze, held-out-only acquisition, independent verification,
and evaluation without refitting. Use
[`docs/stage5_v2_protocol.md`](stage5_v2_protocol.md) for the complete custody
rules.

Preferred candidate family: Windows x64, VS2022/MSVC, ordinary `/openmp`
validated by actual project semantics, Ninja, CMake, Python 3.12-class, Git,
and CUDA Toolkit 12.8.x with `sm_89`. The exact driver, VS patch, Toolkit, GPU,
CPU topology, RAM, and absolute tool paths are captured at preflight; they are
not claimed validated before the target-host rehearsal. PATH resolution to
MSYS/MinGW CMake or Ninja is an abort condition.

Every build/evidence output directory must be absent. Failures retain logs and
partial artifacts and restart only in a new directory. Do not patch the frozen
checkout, reuse a CMake cache, accept a detached CTest text file, supply a
hand-written OpenMP list, or run calibration during the first rehearsal.
