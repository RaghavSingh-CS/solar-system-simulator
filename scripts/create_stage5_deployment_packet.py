#!/usr/bin/env python3
"""Generate an external, deterministic Stage 5 v2.0 deployment packet."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from stage5_common import PROTOCOL_VERSION, Stage5Error, require, require_absent_directory, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--clone-url",
        default="https://github.com/RaghavSingh-CS/solar-system-simulator.git",
    )
    return parser.parse_args()


def runbook(source_sha: str, clone_url: str) -> str:
    return f"""# Stage 5 v2.0 exact deployment packet

Protocol: `{PROTOCOL_VERSION}`
Frozen source: `{source_sha}`

Open an x64 Native Tools Command Prompt for Visual Studio 2022, then start
PowerShell. The preferred candidate family is CUDA 12.8.x + VS2022 with
ordinary feature-tested MSVC OpenMP, Ninja, CMake, Python 3.12-class, and Git.
The exact target tuple is not prevalidated and must be captured below.

```powershell
$ErrorActionPreference = 'Stop'
$stage5Sha = '{source_sha}'
$stage5EvidenceRoot = 'C:\\stage5-v2-evidence'
$stage5EnvironmentId = 'rtx4050-win-SET-UTC-TIMESTAMP'
$stage5PacketRoot = (Resolve-Path (Read-Host 'Absolute path to this deployment packet')).Path
git clone '{clone_url}' solar-system-simulator-stage5-v2
Set-Location solar-system-simulator-stage5-v2
git checkout --detach $stage5Sha
if ((git rev-parse HEAD).Trim() -ne $stage5Sha) {{ throw 'source SHA mismatch' }}
if (git status --porcelain) {{ throw 'checkout is not clean' }}

$stage5Build = Join-Path (Get-Location) 'build-stage5-v2'
$stage5BuildEvidence = Join-Path $stage5EvidenceRoot 'build'
if (Test-Path -LiteralPath $stage5Build) {{ throw 'build directory must be absent' }}
if (Test-Path -LiteralPath $stage5BuildEvidence) {{ throw 'build evidence directory must be absent' }}

python -B scripts/prepare_stage5_build.py `
  --source-dir . --source-commit $stage5Sha `
  --build-dir $stage5Build --output $stage5BuildEvidence `
  --environment-id $stage5EnvironmentId --enable-cuda `
  --expected-gpu 'RTX 4050' --device 0

$stage5BuildManifest = Join-Path $stage5BuildEvidence 'build_manifest.json'
python -B scripts/run_stage5.py --preflight `
  --output (Join-Path $stage5EvidenceRoot 'preflight') `
  --build-manifest $stage5BuildManifest `
  --source-commit $stage5Sha --environment-id $stage5EnvironmentId

python -B scripts/run_stage5.py --pilot `
  --output (Join-Path $stage5EvidenceRoot 'pilot') `
  --build-manifest $stage5BuildManifest `
  --source-commit $stage5Sha --environment-id $stage5EnvironmentId `
  --experiment-id 'stage5-v2-pilot-rtx4050' `
  --seed 24301 --order-seed 20260926 --device 0

python -B scripts/verify_stage5_evidence.py `
  (Join-Path $stage5EvidenceRoot 'pilot') `
  --report (Join-Path $stage5EvidenceRoot 'pilot-verification.json')

Copy-Item (Join-Path $stage5PacketRoot 'pilot_approval.template.json') `
  (Join-Path $stage5EvidenceRoot 'pilot-approval.json')
notepad (Join-Path $stage5EvidenceRoot 'pilot-approval.json')
```

**FIRST NVIDIA REHEARSAL STOP:** stop after independent pilot verification and
manifest-bound human approval. Do not run calibration during the first
rehearsal.

After a separately authorized final collection session, the remaining exact
phase commands are:

```powershell
python -B scripts/run_stage5.py --calibration `
  --output (Join-Path $stage5EvidenceRoot 'calibration') `
  --build-manifest $stage5BuildManifest `
  --source-commit $stage5Sha --environment-id $stage5EnvironmentId `
  --experiment-id 'stage5-v2-calibration-rtx4050' `
  --pilot-dir (Join-Path $stage5EvidenceRoot 'pilot') `
  --pilot-approval (Join-Path $stage5EvidenceRoot 'pilot-approval.json') `
  --seed 24301 --order-seed 20260926 --device 0

python -B scripts/verify_stage5_evidence.py `
  (Join-Path $stage5EvidenceRoot 'calibration') `
  --report (Join-Path $stage5EvidenceRoot 'calibration-verification.json')

python -B scripts/build_stage5_policy.py `
  (Join-Path $stage5EvidenceRoot 'calibration') `
  --output (Join-Path $stage5EvidenceRoot 'policy')

python -B scripts/run_stage5.py --held-out `
  --output (Join-Path $stage5EvidenceRoot 'held-out') `
  --build-manifest $stage5BuildManifest `
  --source-commit $stage5Sha --environment-id $stage5EnvironmentId `
  --experiment-id 'stage5-v2-held-out-rtx4050' `
  --calibration-dir (Join-Path $stage5EvidenceRoot 'calibration') `
  --policy (Join-Path $stage5EvidenceRoot 'policy\\policy.json') `
  --policy-freeze (Join-Path $stage5EvidenceRoot 'policy\\policy_freeze.json') `
  --seed 24301 --order-seed 20260926 --device 0

python -B scripts/verify_stage5_evidence.py `
  (Join-Path $stage5EvidenceRoot 'held-out') `
  --report (Join-Path $stage5EvidenceRoot 'held-out-verification.json')

python -B scripts/evaluate_stage5_policy.py `
  (Join-Path $stage5EvidenceRoot 'held-out') `
  --policy (Join-Path $stage5EvidenceRoot 'policy\\policy.json') `
  --policy-freeze (Join-Path $stage5EvidenceRoot 'policy\\policy_freeze.json') `
  --output (Join-Path $stage5EvidenceRoot 'held-out-evaluation.json')
```
"""


ABORT_CRITERIA = """# Abort criteria

Stop without patching the frozen checkout if any of these occurs:

- source SHA or clean status differs;
- tool or topology capture fails;
- PATH selects MSYS/MinGW instead of the intended VS/NVIDIA toolchain;
- build/evidence directory already exists;
- CMake source, Release mode, MSVC host compiler, CUDA compiler, or `sm_89` differs;
- any binary build identity/hash differs;
- CTest or Stage 5 preflight fails;
- requested/effective OpenMP teams differ;
- pilot row, checksum, gate, timing relationship, or device identity fails;
- projected final collection no longer fits the authorized session;
- calibration, policy freeze, held-out, or environment identity chain differs.

Retain all logs and partial/failure artifacts. Never delete evidence to make a
rerun pass. The first NVIDIA rehearsal always stops after pilot approval.
"""


def main() -> int:
    args = parse_args()
    require(re.fullmatch(r"[0-9a-fA-F]{40}", args.source_sha) is not None, "source SHA must be 40 hexadecimal characters")
    output = require_absent_directory(args.output, "deployment packet directory")
    output.mkdir(parents=True)
    files = {
        "SOURCE_SHA.txt": args.source_sha.lower() + "\n",
        "RUNBOOK.md": runbook(args.source_sha.lower(), args.clone_url),
        "ABORT_CRITERIA.md": ABORT_CRITERIA,
        "pilot_approval.template.json": json.dumps(
            {
                "approved": False,
                "protocol_version": PROTOCOL_VERSION,
                "source_commit": args.source_sha.lower(),
                "build_manifest_sha256": "REPLACE_FROM_BUILD_MANIFEST_FILE_HASH",
                "environment_id": "REPLACE_FROM_BUILD_MANIFEST",
                "environment_sha256": "REPLACE_FROM_BUILD_MANIFEST",
                "pilot_manifest_sha256": "REPLACE_FROM_VERIFIED_PILOT",
                "openmp_threads": "REPLACE_FROM_DERIVED_TOPOLOGY_LIST",
                "reviewer": "REPLACE_WITH_REVIEWER",
                "reviewed_utc": "REPLACE_WITH_ISO_8601_UTC_TIMESTAMP",
                "notes": "REPLACE_WITH_REVIEW_OUTCOME",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    }
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8", newline="\n")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": args.source_sha.lower(),
        "clone_url": args.clone_url,
        "deterministic_packet": True,
        "files": {
            name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
            for name in sorted(files)
        },
    }
    write_json(output / "packet_manifest.json", manifest)
    require("STAGE5_FROZEN_SOURCE_COMMIT" not in (output / "RUNBOOK.md").read_text(encoding="utf-8"), "forbidden source placeholder survived")
    print(f"deployment packet created: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Stage5Error, ValueError, FileNotFoundError) as error:
        print(f"packet generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
