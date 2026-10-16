#!/usr/bin/env python3
"""Host-only fail-closed tests for the RTX 4050 handoff package."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "nvidia_handoff"
SCRIPTS = HANDOFF / "scripts"
RESULTS = HANDOFF / "results"
BIN = HANDOFF / "bin"
SCRATCH = ROOT / "build" / "scratch" / f"nvidia-handoff-tests-{os.getpid()}"
PREFIX = f"host-test-{os.getpid()}"
HEAD = "fea40efa425de19671c5e23bc92af6ac342b777c"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(
    *arguments: str,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(arguments), cwd=ROOT, text=True, capture_output=True, check=False, env=env
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def powershell() -> str:
    value = shutil.which("pwsh") or shutil.which("powershell")
    require(value is not None, "PowerShell is required for handoff launcher tests")
    return str(value)


def run_script(path: Path, *arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return run(
        powershell(), "-NoProfile", "-NonInteractive", "-File", str(path),
        *arguments, env=env,
    )


def run_name(name: str) -> str:
    return f"{PREFIX}-{name}"


def run_path(name: str) -> Path:
    return RESULTS / run_name(name)


def make_marked_run(name: str) -> Path:
    path = run_path(name)
    path.mkdir(parents=False)
    (path / "logs").mkdir()
    (path / "stages").mkdir()
    (path / ".handoff-run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "solar-system-simulator-nvidia-handoff-run",
                "run_id": path.name,
                "created_utc": "2026-10-16T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return path


def write_cmd(path: Path, lines: list[str]) -> None:
    path.write_text("@echo off\r\n" + "\r\n".join(lines) + "\r\n", encoding="ascii")


def fixture_environment(
    name: str,
    *,
    include_nvcc: bool = True,
    include_smi: bool = True,
    architecture: str = "compute_89",
    include_visual_studio_environment: bool = True,
) -> dict[str, str]:
    fixture = SCRATCH / f"tools-{name}"
    fixture.mkdir(parents=True)
    write_cmd(
        fixture / "git.cmd",
        [
            'if "%1"=="--version" echo git version fixture',
            f'if "%3"=="rev-parse" if "%4"=="HEAD" echo {HEAD}',
            'if "%3"=="rev-parse" if "%4"=="--abbrev-ref" echo feature/cuda-visualizer',
            "exit /b 0",
        ],
    )
    for tool in ("python", "cmake", "ctest", "ninja", "cl", "dumpbin"):
        write_cmd(fixture / f"{tool}.cmd", [f"echo {tool} fixture", "exit /b 0"])
    if include_nvcc:
        write_cmd(
            fixture / "nvcc.cmd",
            [
                f'if "%1"=="--list-gpu-arch" echo {architecture}',
                'if not "%1"=="--list-gpu-arch" echo nvcc fixture',
                "exit /b 0",
            ],
        )
    if include_smi:
        write_cmd(
            fixture / "nvidia-smi.cmd",
            [
                'if not "%1"=="" echo 0, NVIDIA GeForce RTX 4050 Laptop GPU, 6144, 555.42, 8.9',
                'if "%1"=="" echo NVIDIA-SMI fixture',
                "exit /b 0",
            ],
        )
    environment = os.environ.copy()
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    environment["PATH"] = str(fixture) + os.pathsep + str(system32)
    if include_visual_studio_environment:
        environment["VCToolsInstallDir"] = r"C:\fixture\VC\Tools"
        environment["WindowsSdkDir"] = r"C:\fixture\WindowsSDK"
    else:
        environment.pop("VCToolsInstallDir", None)
        environment.pop("WindowsSdkDir", None)
    return environment


def read_status(path: Path, stage: str) -> dict[str, object]:
    return json.loads((path / "stages" / f"{stage}.json").read_text(encoding="utf-8-sig"))


def file_snapshot(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def make_staged_fixture(name: str, target: str = "fixture_target") -> tuple[Path, Path, Path]:
    run_directory = make_marked_run(name)
    source = ROOT / "build" / "cuda-release" / f"{PREFIX}-{name}-{target}.exe"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"deterministic staged executable fixture\n")
    destination = BIN / run_directory.name / "cuda-release" / f"{target}.exe"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source_commit": HEAD,
        "executables": [
            {
                "cmake_target": target,
                "build_configuration": "cuda-release",
                "source_build_location": str(source),
                "staged_location": str(destination),
                "sha256": digest,
            }
        ],
    }
    (run_directory / "executables.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_directory, source, destination


def test_inventory_and_scripts() -> None:
    manifest = json.loads((HANDOFF / "MANIFEST.json").read_text(encoding="utf-8"))
    config = json.loads((HANDOFF / "config" / "handoff.json").read_text(encoding="utf-8"))
    require(manifest["cuda_architecture"] == "89", "manifest architecture differs")
    require(config["required_compute_capability"] == "8.9", "compute capability differs")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    targets = set(re.findall(r"add_executable\s*\(\s*([A-Za-z0-9_]+)", cmake))
    declared = [item["target"] for item in manifest["targets"]]
    missing = sorted(set(declared) - targets)
    require(not missing, f"manifest invents CMake executable targets: {missing}")
    pairs = {(item["configuration"], item["target"]) for item in manifest["targets"]}
    require(len(declared) == len(pairs), "manifest has duplicate configuration/target entries")
    require("CUDA Barnes-Hut in the OpenGL viewer" in manifest["unsupported"],
            "unsupported viewer boundary is missing")
    expected = {"common.ps1", "run_all.ps1", "run_demo.ps1"} | {
        f"{index:02d}_{stage}.ps1"
        for index, stage in enumerate(
            ("preflight", "build", "cpu_baseline", "cuda_direct", "cuda_barnes_hut",
             "cuda_numerics", "cuda_resources", "stage5_pilot", "viewer", "collect_results")
        )
    }
    actual = {path.name for path in SCRIPTS.glob("*.ps1")}
    require(expected <= actual, f"missing launcher scripts: {sorted(expected - actual)}")


def test_powershell_syntax() -> None:
    ps = powershell()
    for path in sorted(SCRIPTS.glob("*.ps1")):
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile('{path}',"
            "[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
        )
        run(ps, "-NoProfile", "-NonInteractive", "-Command", command, check=True)


def test_ignore_rules() -> None:
    probes = (
        "nvidia_handoff/bin/example/run.exe",
        "nvidia_handoff/results/example/log.txt",
    )
    completed = run("git", "check-ignore", *probes)
    require(completed.returncode == 0, "generated handoff files are not Git-ignored")
    require(set(completed.stdout.splitlines()) == set(probes), "unexpected ignore result")


def test_preflight_negative_controls() -> None:
    cases = (
        ("missing-nvcc", fixture_environment("missing-nvcc", include_nvcc=False), "nvcc"),
        ("missing-smi", fixture_environment("missing-smi", include_smi=False), "nvidia-smi"),
        ("wrong-architecture", fixture_environment("wrong-architecture", architecture="compute_86"), "compute_89"),
        ("missing-msvc-environment", fixture_environment(
            "missing-msvc-environment", include_visual_studio_environment=False
        ), "Visual C++"),
    )
    for name, environment, expected_text in cases:
        directory = run_path(name)
        completed = run_script(
            SCRIPTS / "00_preflight.ps1", "-RunDirectory", str(directory), env=environment
        )
        require(completed.returncode != 0, f"preflight unexpectedly passed: {name}")
        require(read_status(directory, "00_preflight")["status"] == "failed",
                f"preflight failure status missing: {name}")
        require(expected_text.lower() in (completed.stdout + completed.stderr).lower(),
                f"preflight diagnostic omitted {expected_text}: {name}")


def test_staged_executable_identity() -> None:
    directory, source, staged = make_staged_fixture("staged-identity")
    common = SCRIPTS / "common.ps1"
    command = (
        f". '{common}'; "
        f"Get-HandoffStagedExecutable '{directory}' 'cuda-release' 'fixture_target' | Out-Null"
    )
    run(powershell(), "-NoProfile", "-NonInteractive", "-Command", command, check=True)
    staged.write_bytes(b"tampered\n")
    completed = run(powershell(), "-NoProfile", "-NonInteractive", "-Command", command)
    require(completed.returncode != 0, "tampered staged executable was accepted")
    require("SHA-256 mismatch" in completed.stderr, "hash failure diagnostic missing")
    staged.unlink()
    completed = run(powershell(), "-NoProfile", "-NonInteractive", "-Command", command)
    require(completed.returncode != 0, "missing staged executable was accepted")
    require("missing" in completed.stderr.lower(), "missing executable diagnostic absent")
    source.unlink(missing_ok=True)


def test_cuda_test_outcomes() -> None:
    common = SCRIPTS / "common.ps1"
    for name, lines, diagnostic in (
        ("ctest-nonzero", ["echo CUDA test failure", "exit /b 9"], "exit code 9"),
        ("ctest-skipped", ["echo 1/1 Test #1: fixture ***Skipped", "exit /b 0"], "skipped"),
    ):
        directory, source, _ = make_staged_fixture(name)
        fixture = SCRATCH / f"ctest-{name}"
        fixture.mkdir(parents=True)
        write_cmd(fixture / "ctest.cmd", lines)
        environment = os.environ.copy()
        environment["PATH"] = str(fixture) + os.pathsep + environment["PATH"]
        command = (
            f". '{common}'; Invoke-CtestNoSkips '{directory}' '{name}' '^fixture$' "
            "@('fixture_target') | Out-Null"
        )
        completed = run(
            powershell(), "-NoProfile", "-NonInteractive", "-Command", command,
            env=environment,
        )
        require(completed.returncode != 0, f"CUDA CTest outcome was incorrectly accepted: {name}")
        require(diagnostic.lower() in (completed.stdout + completed.stderr).lower(),
                f"CUDA CTest diagnostic missing: {name}")
        source.unlink(missing_ok=True)


def test_repeat_stage_is_immutable() -> None:
    directory = make_marked_run("repeat-stage")
    (directory / "stages" / "01_build.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    log = directory / "logs" / "03_cuda_direct.log"
    status = directory / "stages" / "03_cuda_direct.json"
    log.write_bytes(b"immutable successful log\r\n")
    status.write_bytes(b'{"status":"passed","exit_code":0}\r\n')
    before_log = log.read_bytes()
    before_status = status.read_bytes()
    completed = run_script(SCRIPTS / "03_cuda_direct.ps1", "-RunDirectory", str(directory))
    require(completed.returncode != 0, "repeated CUDA stage unexpectedly passed")
    require(log.read_bytes() == before_log, "repeated stage mutated its old log")
    require(status.read_bytes() == before_status, "repeated stage mutated its old status")


def test_collection_scope_limits_and_spaces() -> None:
    outside = SCRATCH / "outside-run"
    completed = run_script(SCRIPTS / "09_collect_results.ps1", "-RunDirectory", str(outside))
    require(completed.returncode != 0, "collector accepted an outside directory")
    require(not outside.exists(), "collector created an outside directory")

    for name, relative, writer in (
        ("unrecognized", "secret.key", lambda path: path.write_text("credential", encoding="utf-8")),
        ("oversized", "logs/oversized.log", lambda path: path.open("wb").truncate(32 * 1024 * 1024 + 1)),
    ):
        directory = make_marked_run(name)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        writer(target)
        before = file_snapshot(directory)
        completed = run_script(SCRIPTS / "09_collect_results.ps1", "-RunDirectory", str(directory))
        require(completed.returncode != 0, f"collector accepted {name} content")
        require(file_snapshot(directory) == before, f"failed collection mutated prior results: {name}")
        require(not (RESULTS / f"{directory.name}.zip").exists(), f"failed collection published ZIP: {name}")

    directory = make_marked_run("path with spaces")
    (directory / "logs" / "fixture log.txt").write_text(
        "command=fixture with spaces\ncommand_exit_code=0\n", encoding="utf-8"
    )
    completed = run_script(SCRIPTS / "09_collect_results.ps1", "-RunDirectory", str(directory))
    require(completed.returncode == 0, f"space-safe collection failed: {completed.stderr}")
    archive = RESULTS / f"{directory.name}.zip"
    require(archive.is_file(), "space-safe result ZIP was not created")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        require(any(name.endswith("collection_manifest.json") for name in names),
                "collection manifest missing from ZIP")
        require(any(name.endswith("logs/fixture log.txt") for name in names),
                "space-containing log missing from ZIP")
        require(any(name.endswith("stages/09_collect_results.json") for name in names),
                "collection success status missing from ZIP")


def test_stage5_is_disabled() -> None:
    directory = make_marked_run("stage5-disabled")
    completed = run_script(SCRIPTS / "07_stage5_pilot.ps1", "-RunDirectory", str(directory))
    require(completed.returncode != 0, "automated Stage 5 unexpectedly ran")
    expected = "Automated Stage 5 execution is disabled"
    require(expected in completed.stdout + completed.stderr, "Stage 5 refusal is unclear")
    require(read_status(directory, "07_stage5_pilot")["status"] == "failed",
            "Stage 5 refusal was not recorded")
    master = (SCRIPTS / "run_all.ps1").read_text(encoding="utf-8")
    require("07_stage5_pilot.ps1" not in master, "master runner can still invoke Stage 5")


def test_master_stops_and_collects_after_preflight_failure() -> None:
    directory = run_path("master-failure")
    completed = run_script(SCRIPTS / "run_all.ps1", "-RunDirectory", str(directory))
    require(completed.returncode != 0, "master unexpectedly passed without NVIDIA tooling")
    require(read_status(directory, "00_preflight")["status"] == "failed",
            "master did not retain preflight failure")
    require(not (directory / "stages" / "01_build.json").exists(),
            "master ran downstream build after preflight failure")
    archive = RESULTS / f"{directory.name}.zip"
    require(archive.is_file(), "master did not package partial failure results")
    with zipfile.ZipFile(archive) as bundle:
        stage_records = [name for name in bundle.namelist() if name.endswith("stages/09_collect_results.json")]
        require(len(stage_records) == 1, "collection status missing from partial-failure ZIP")
        status = json.loads(bundle.read(stage_records[0]).decode("utf-8-sig"))
        require(status["status"] == "passed", "partial-failure collection was not successful")


def cleanup() -> None:
    if SCRATCH.exists():
        resolved = SCRATCH.resolve()
        scratch_root = (ROOT / "build" / "scratch").resolve()
        require(scratch_root in resolved.parents, "refusing to clean outside build/scratch")
        shutil.rmtree(resolved, ignore_errors=True)
    for path in RESULTS.iterdir():
        if path.name.startswith(PREFIX):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    for path in BIN.iterdir():
        if path.name.startswith(PREFIX) and path.is_dir():
            shutil.rmtree(path)
    cuda_build = ROOT / "build" / "cuda-release"
    if cuda_build.exists():
        for path in cuda_build.glob(f"{PREFIX}-*.exe"):
            path.unlink()


def main() -> int:
    cleanup()
    SCRATCH.mkdir(parents=True)
    try:
        test_inventory_and_scripts()
        test_powershell_syntax()
        test_ignore_rules()
        test_preflight_negative_controls()
        test_staged_executable_identity()
        test_cuda_test_outcomes()
        test_repeat_stage_is_immutable()
        test_collection_scope_limits_and_spaces()
        test_stage5_is_disabled()
        test_master_stops_and_collects_after_preflight_failure()
    finally:
        cleanup()
    print("NVIDIA handoff host-only tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
