#!/usr/bin/env python3
"""Capture source/build custody for the uncommitted Large-N L1 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ".gitattributes",
    "CMakeLists.txt",
    "CMakePresets.json",
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
    "tests/test_large_n_laboratory.cpp",
    "tests/test_large_n_l1_tools.py",
    "scripts/run_large_n_l1_pilot.py",
    "scripts/capture_large_n_l1_provenance.py",
    "scripts/verify_large_n_l1_evidence.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "large_n_l1_pilot",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=ROOT / "build" / "openmp-release"
    )
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    build_dir = args.build_dir.resolve()
    raw = evidence / "raw.csv"
    metadata_path = evidence / "metadata.json"
    binary = build_dir / "nbody_large_n_lab.exe"
    for path in (raw, metadata_path, binary, build_dir / "CMakeCache.txt"):
        if not path.is_file():
            raise RuntimeError(f"required custody input missing: {path}")
    for relative in SOURCES:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"source custody input missing: {relative}")

    manifest_path = evidence / "source_manifest.sha256"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for relative in SOURCES:
            stream.write(f"{sha256(ROOT / relative)} *{relative}\n")

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
        raise RuntimeError("CMakeCache does not identify a usable Ninja executable")
    if not compiler or not Path(compiler).is_file():
        raise RuntimeError("CMakeCache does not identify a usable C++ compiler")
    commands = subprocess.check_output(
        [make_program, "-C", str(build_dir), "-t", "commands", "nbody_large_n_lab"],
        cwd=ROOT, text=True, errors="replace",
    ).splitlines()
    compiler_version = subprocess.check_output(
        [compiler, "--version"], cwd=ROOT, text=True, errors="replace"
    ).splitlines()
    provenance = {
        "build_directory": str(build_dir.relative_to(ROOT)).replace("\\", "/"),
        "cmake_cache": cache,
        "compiler_version_output": compiler_version,
        "effective_commands": commands,
        "executable": {
            "path": str(binary.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(binary),
        },
        "source_manifest": "source_manifest.sha256",
        "source_manifest_sha256": sha256(manifest_path),
    }
    provenance_path = evidence / "build_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["outputs"].update({
        "raw_csv_sha256": sha256(raw),
        "build_provenance": "build_provenance.json",
        "build_provenance_sha256": sha256(provenance_path),
        "source_manifest": "source_manifest.sha256",
        "source_manifest_sha256": sha256(manifest_path),
    })
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"captured {len(SOURCES)} source files and build provenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
