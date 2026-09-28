#!/usr/bin/env python3
"""Create a fresh source-bound Stage 5 v2.0 build and test manifest."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage5_common import (
    CUDA_ARCHITECTURE,
    PROTOCOL_VERSION,
    Stage5Error,
    canonical_sha256,
    command_record,
    git_value,
    parse_cmake_cache,
    require,
    require_absent_directory,
    run_command,
    sha256_file,
    windows_topology,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--enable-cuda", action="store_true")
    parser.add_argument("--expected-gpu", default="RTX 4050")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--topology-fixture",
        metavar="PHYSICAL,LOGICAL",
        help="test/local-only topology fixture; marks the manifest fixture_mode=true",
    )
    return parser.parse_args()


def executable_path(build_dir: Path, name: str) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    candidates = (build_dir / f"{name}{suffix}", build_dir / "Release" / f"{name}{suffix}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise Stage5Error(f"required executable missing after build: {name}")


def parse_fixture(value: str) -> dict[str, Any]:
    try:
        physical_text, logical_text = value.split(",", 1)
        physical = int(physical_text)
        logical = int(logical_text)
    except (ValueError, TypeError) as error:
        raise Stage5Error("--topology-fixture must be PHYSICAL,LOGICAL") from error
    from stage5_common import derive_openmp_threads

    threads = derive_openmp_threads(physical, logical)
    return {
        "processors": [],
        "physical_cores": physical,
        "logical_processors": logical,
        "total_physical_memory_bytes": None,
        "os": {"fixture": True},
        "derivation": "sorted unique {1, powers of two below P, P, L}, capped at L",
        "openmp_threads": list(threads),
        "fixture": True,
    }


def tool_path(name: str) -> str:
    resolved = shutil.which(name)
    return str(Path(resolved).resolve()) if resolved else ""


def tool_record(name: str, version_args: list[str]) -> dict[str, Any]:
    path = tool_path(name)
    record = command_record([path or name, *version_args])
    record["absolute_path"] = path
    return record


def assert_non_msys_tool(path: str, label: str) -> None:
    lowered = path.casefold().replace("/", "\\")
    require(path != "", f"{label} is unavailable on PATH")
    require("\\msys" not in lowered and "\\mingw" not in lowered, f"{label} resolves to MSYS/MinGW: {path}")


def build_info(path: Path, source_commit: str) -> dict[str, Any]:
    completed = run_command([str(path), "--build-info"])
    require(completed.returncode == 0, f"--build-info failed for {path}: {completed.stderr}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise Stage5Error(f"invalid --build-info JSON from {path}") from error
    require(value.get("source_commit") == source_commit, f"binary source identity mismatch: {path}")
    require(value.get("protocol_version") == PROTOCOL_VERSION, f"binary protocol mismatch: {path}")
    require(str(value.get("build_configuration", "")).casefold() == "release", f"non-Release binary: {path}")
    return value


def ctest_succeeded(completed: Any) -> bool:
    output = completed.stdout + completed.stderr
    return (
        completed.returncode == 0
        and "100% tests passed" in output
        and "0 tests failed" in output
    )


def main() -> int:
    args = parse_args()
    source = args.source_dir.resolve()
    build = require_absent_directory(args.build_dir, "controlled build directory")
    output = require_absent_directory(args.output, "build-evidence directory")
    require(len(args.source_commit) == 40, "source commit must be a full 40-character SHA")
    require(git_value(source, "rev-parse", "HEAD") == args.source_commit, "source HEAD mismatch")
    require(git_value(source, "status", "--porcelain=v1") == "", "source checkout must be clean")
    require(args.device >= 0, "CUDA device must be non-negative")

    topology = parse_fixture(args.topology_fixture) if args.topology_fixture else windows_topology()
    if args.enable_cuda:
        require(not topology.get("fixture"), "fixture topology cannot authorize a CUDA controlled build")

    tools = {
        "git": tool_record("git", ["--version"]),
        "python": {
            "absolute_path": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
        },
        "cmake": tool_record(args.cmake, ["--version"]),
        "ninja": tool_record("ninja", ["--version"]),
        "ctest": tool_record(args.ctest, ["--version"]),
        "cl": tool_record("cl", []),
        "nvcc": tool_record("nvcc", ["--version"]),
        "nvidia_smi": tool_record("nvidia-smi", []),
    }
    if args.enable_cuda:
        for label in ("cmake", "ninja"):
            assert_non_msys_tool(str(tools[label]["absolute_path"]), label)
        require(tools["cl"]["absolute_path"], "cl.exe is unavailable; use an x64 VS2022 developer shell")
        require(tools["nvcc"]["return_code"] == 0, "nvcc preflight failed")
        require(tools["nvidia_smi"]["return_code"] == 0, "nvidia-smi preflight failed")
        gpu_query = command_record([
            tools["nvidia_smi"]["absolute_path"],
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ])
        require(gpu_query["return_code"] == 0, "NVIDIA GPU query failed")
        selected = next(
            (
                line
                for line in str(gpu_query["stdout"]).splitlines()
                if line.split(",", 1)[0].strip() == str(args.device)
            ),
            "",
        )
        require(args.expected_gpu.casefold() in selected.casefold(), f"unexpected CUDA device: {selected}")
        require(selected.rstrip().endswith("8.9"), f"selected GPU does not report compute capability 8.9: {selected}")
        tools["nvidia_smi_query"] = gpu_query

    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    configure_command = [
        tools["cmake"]["absolute_path"] or args.cmake,
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        args.generator,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DNBODY_ENABLE_OPENMP=ON",
        f"-DNBODY_ENABLE_CUDA={'ON' if args.enable_cuda else 'OFF'}",
        "-DNBODY_ENABLE_MPI=OFF",
        "-DNBODY_BUILD_VISUALIZATION=OFF",
        f"-DNBODY_STAGE5_SOURCE_SHA={args.source_commit}",
        f"-DNBODY_STAGE5_PROTOCOL_VERSION={PROTOCOL_VERSION}",
    ]
    if args.enable_cuda:
        configure_command.append(f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHITECTURE}")
    configure = run_command(configure_command)
    (output / "configure.txt").write_text(configure.stdout + configure.stderr, encoding="utf-8")
    require(configure.returncode == 0, "controlled CMake configure failed; retained configure.txt")

    build_command = [tools["cmake"]["absolute_path"] or args.cmake, "--build", str(build), "--config", "Release"]
    built = run_command(build_command)
    (output / "build.txt").write_text(built.stdout + built.stderr, encoding="utf-8")
    require(built.returncode == 0, "controlled build failed; retained build.txt")

    cache_path = build / "CMakeCache.txt"
    require(cache_path.is_file(), "CMakeCache.txt missing")
    cache = parse_cmake_cache(cache_path)
    require(Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() == source, "CMAKE_HOME_DIRECTORY mismatch")
    require(cache.get("CMAKE_BUILD_TYPE") == "Release", "CMAKE_BUILD_TYPE must be Release")
    require(cache.get("CMAKE_GENERATOR") == args.generator, "CMake generator mismatch")
    require(cache.get("NBODY_ENABLE_OPENMP") == "ON", "OpenMP build option missing")
    require(cache.get("NBODY_ENABLE_CUDA") == ("ON" if args.enable_cuda else "OFF"), "CUDA build option mismatch")
    require(cache.get("NBODY_STAGE5_SOURCE_SHA") == args.source_commit, "cache source SHA mismatch")
    require(cache.get("NBODY_STAGE5_PROTOCOL_VERSION") == PROTOCOL_VERSION, "cache protocol mismatch")
    if args.enable_cuda:
        require(cache.get("CMAKE_CUDA_ARCHITECTURES") == CUDA_ARCHITECTURE, "CUDA architecture must be 89")

    required_names = [
        "nbody_benchmark",
        "nbody_tests",
        "nbody_solar_system_tests",
        "nbody_openmp_tests",
    ]
    if args.enable_cuda:
        required_names.extend(["nbody_cuda_benchmark", "nbody_cuda_tests"])
    binaries: dict[str, Any] = {}
    for name in required_names:
        path = executable_path(build, name)
        binaries[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "build_info": build_info(path, args.source_commit),
        }
    compiler_identity = binaries["nbody_benchmark"]["build_info"]
    cuda_compiler_identity: dict[str, Any] = {}
    if args.enable_cuda:
        require(compiler_identity.get("compiler_id") == "MSVC", "controlled CUDA build must use MSVC C++")
        cxx_path = str(compiler_identity.get("cxx_compiler", ""))
        require(cxx_path.casefold().endswith("cl.exe"), f"controlled C++ compiler is not cl.exe: {cxx_path}")
        require(compiler_identity.get("generator") == args.generator, "embedded generator mismatch")
        cuda_compiler_identity = binaries["nbody_cuda_benchmark"]["build_info"]
        require(Path(str(cuda_compiler_identity.get("cuda_compiler", ""))).is_absolute(), "CUDA compiler path is not absolute")
        require(str(cuda_compiler_identity.get("cuda_compiler_id", "")) == "NVIDIA", "CUDA compiler ID mismatch")
        require(str(cuda_compiler_identity.get("cuda_compiler_version", "")), "CUDA compiler version missing")
        require(Path(str(cuda_compiler_identity.get("cuda_host_compiler", ""))).is_absolute(), "CUDA host compiler path is not absolute")
        require(str(cuda_compiler_identity.get("cuda_architectures", "")) == CUDA_ARCHITECTURE, "embedded CUDA architecture must be 89")

    ctest_command = [tools["ctest"]["absolute_path"] or args.ctest, "--test-dir", str(build), "-C", "Release", "--output-on-failure"]
    ctest = run_command(ctest_command)
    ctest_path = output / "ctest.txt"
    ctest_output = ctest.stdout + ctest.stderr
    ctest_path.write_text(ctest_output, encoding="utf-8")
    ctest_passed = ctest_succeeded(ctest)

    selected_cache_keys = (
        "CMAKE_HOME_DIRECTORY",
        "CMAKE_GENERATOR",
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_COMPILER_ID",
        "CMAKE_CXX_COMPILER_VERSION",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_COMPILER_ID",
        "CMAKE_CUDA_COMPILER_VERSION",
        "CMAKE_CUDA_HOST_COMPILER",
        "CMAKE_CUDA_ARCHITECTURES",
        "OpenMP_CXX_FLAGS",
        "OpenMP_CXX_LIB_NAMES",
        "NBODY_ENABLE_OPENMP",
        "NBODY_ENABLE_CUDA",
        "NBODY_ENABLE_MPI",
        "NBODY_BUILD_VISUALIZATION",
        "NBODY_STAGE5_SOURCE_SHA",
        "NBODY_STAGE5_PROTOCOL_VERSION",
    )
    environment = {
        "environment_id": args.environment_id,
        "fixture_mode": bool(topology.get("fixture")),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "tools": tools,
        "topology": topology,
        "power_state_notes": {
            "machine_readable": False,
            "note": "AC power, vendor performance mode, thermals, and background load require operator confirmation.",
        },
    }
    environment_hash = canonical_sha256(environment)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": args.source_commit,
        "source_directory": str(source),
        "source_clean": True,
        "environment_id": args.environment_id,
        "environment_sha256": environment_hash,
        "environment": environment,
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_mode": bool(topology.get("fixture")),
        "configure_command": configure_command,
        "build_command": build_command,
        "ctest": {
            "command": ctest_command,
            "return_code": ctest.returncode,
            "passed": ctest_passed,
            "log_sha256": sha256_file(ctest_path),
            "log_bytes": ctest_path.stat().st_size,
        },
        "cmake_cache": {key: cache.get(key, "") for key in selected_cache_keys},
        "compiler_identity": compiler_identity,
        "cuda_compiler_identity": cuda_compiler_identity,
        "openmp_topology": {
            "physical_cores": topology["physical_cores"],
            "logical_processors": topology["logical_processors"],
            "derivation": topology["derivation"],
            "threads": topology["openmp_threads"],
        },
        "binaries": binaries,
        "tool_scripts": {
            name: sha256_file(source / "scripts" / name)
            for name in (
                "stage5_common.py",
                "run_stage5_v2.py",
                "verify_stage5_v2.py",
                "build_stage5_policy.py",
                "evaluate_stage5_policy.py",
            )
            if (source / "scripts" / name).is_file()
        },
        "files": {
            name: {
                "sha256": sha256_file(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in ("configure.txt", "build.txt", "ctest.txt")
        },
    }
    write_json(output / "build_manifest.json", manifest)
    require(ctest_passed, "controlled CTest failed; build manifest and logs retained")
    print(f"Stage 5 build preparation passed: {output / 'build_manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Stage5Error, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
