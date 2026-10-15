#!/usr/bin/env python3
"""Capture exact source/build custody for the Large-N L2 pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import run_large_n_l1_pilot as publication_core


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ".gitattributes",
    "CMakeLists.txt",
    "CMakePresets.json",
    "tools/large_n_l2_protocol.json",
    "include/nbody/body.hpp",
    "include/nbody/vector3.hpp",
    "include/nbody/direct_force.hpp",
    "include/nbody/simulation.hpp",
    "include/nbody/diagnostics.hpp",
    "include/nbody/openmp_backend.hpp",
    "include/nbody/barnes_hut.hpp",
    "include/nbody/large_n_laboratory.hpp",
    "src/simulation.cpp",
    "src/diagnostics.cpp",
    "backends/openmp/openmp_backend.cpp",
    "src/barnes_hut.cpp",
    "src/large_n_laboratory.cpp",
    "src/large_n_laboratory_main.cpp",
    "scripts/run_large_n_l1_pilot.py",
    "scripts/verify_large_n_l1_evidence.py",
    "scripts/run_large_n_l2_pilot.py",
    "scripts/capture_large_n_l2_provenance.py",
    "scripts/verify_large_n_l2_evidence.py",
    "tests/test_large_n_laboratory.cpp",
    "tests/test_large_n_l1_tools.py",
    "tests/test_large_n_l2_tools.py",
)


def cache_values(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        key_with_type, value = line.split("=", 1)
        key = key_with_type.split(":", 1)[0]
        if key in names:
            values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build" / "openmp-release")
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    build_dir = args.build_dir.resolve()
    binary = build_dir / "nbody_large_n_lab.exe"
    for path in (evidence / "raw.csv", evidence / "metadata.json", evidence / "protocol.json", binary, build_dir / "CMakeCache.txt"):
        if not path.is_file():
            raise RuntimeError(f"required L2 custody input missing: {path}")
    for relative in SOURCES:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"L2 source custody input missing: {relative}")

    manifest_path = evidence / "source_manifest.sha256"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for relative in SOURCES:
            stream.write(f"{publication_core.sha256(ROOT / relative)} *{relative}\n")

    cache = cache_values(
        build_dir / "CMakeCache.txt",
        (
            "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_GENERATOR",
            "CMAKE_MAKE_PROGRAM", "NBODY_ENABLE_OPENMP", "NBODY_ENABLE_CUDA",
            "NBODY_ENABLE_MPI", "OpenMP_CXX_FLAGS", "OpenMP_CXX_LIB_NAMES",
        ),
    )
    make_program = cache.get("CMAKE_MAKE_PROGRAM")
    compiler = cache.get("CMAKE_CXX_COMPILER")
    if not make_program or not Path(make_program).is_file():
        raise RuntimeError("CMake cache lacks a usable Ninja executable")
    if not compiler or not Path(compiler).is_file():
        raise RuntimeError("CMake cache lacks a usable C++ compiler")
    commands = subprocess.check_output(
        [make_program, "-C", str(build_dir), "-t", "commands", "nbody_large_n_lab"],
        cwd=ROOT, text=True, errors="replace",
    ).splitlines()
    compiler_version = subprocess.check_output(
        [compiler, "--version"], cwd=ROOT, text=True, errors="replace"
    ).splitlines()
    provenance = {
        "schema": "large-n-l2-build-provenance-v1",
        "build_directory": str(build_dir.relative_to(ROOT)).replace("\\", "/"),
        "cmake_cache": cache,
        "compiler_version_output": compiler_version,
        "effective_commands": commands,
        "executable": {
            "path": str(binary.relative_to(ROOT)).replace("\\", "/"),
            "sha256": publication_core.sha256(binary),
        },
        "source_manifest": "source_manifest.sha256",
        "source_manifest_sha256": publication_core.sha256(manifest_path),
        "protocol_sha256": publication_core.sha256(evidence / "protocol.json"),
    }
    provenance_path = evidence / "build_provenance.json"
    publication_core.write_json(provenance_path, provenance)
    metadata_path = evidence / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["outputs"].update(
        {
            "build_provenance": "build_provenance.json",
            "build_provenance_sha256": publication_core.sha256(provenance_path),
            "source_manifest": "source_manifest.sha256",
            "source_manifest_sha256": publication_core.sha256(manifest_path),
        }
    )
    publication_core.write_json(metadata_path, metadata)
    print(f"captured {len(SOURCES)} L2 source files and build provenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
