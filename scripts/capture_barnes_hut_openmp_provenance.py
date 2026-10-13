#!/usr/bin/env python3
"""Capture deterministic effective build provenance for the Stage 2 benchmark."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REQUIRED_COMPILE_SOURCES = (
    "src/barnes_hut.cpp",
    "src/barnes_hut_openmp_benchmark.cpp",
    "backends/openmp/openmp_backend.cpp",
    "src/scenarios.cpp",
    "src/simulation.cpp",
)

RELEVANT_CACHE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CXX_COMPILER_AR",
    "CMAKE_CXX_COMPILER_RANLIB",
    "CMAKE_CXX_FLAGS",
    "CMAKE_CXX_FLAGS_RELEASE",
    "CMAKE_EXE_LINKER_FLAGS",
    "CMAKE_EXE_LINKER_FLAGS_RELEASE",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_INSTANCE",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_MAKE_PROGRAM",
    "NBODY_ENABLE_CUDA",
    "NBODY_ENABLE_MPI",
    "NBODY_ENABLE_OPENMP",
    "OpenMP_COMPILE_RESULT_CXX_fopenmp",
    "OpenMP_CXX_FLAGS",
    "OpenMP_CXX_LIB_NAMES",
    "OpenMP_CXX_SPEC_DATE",
    "OpenMP_SPECTEST_CXX_",
    "OpenMP_gomp_LIBRARY",
    "OpenMP_mingwthrd_LIBRARY",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_cache(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        result[key] = value
    return result


def run_text(arguments: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def compiler_identity(build: Path) -> dict[str, str]:
    candidates = sorted((build / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    require(len(candidates) == 1, "expected one CMakeCXXCompiler.cmake")
    text = candidates[0].read_text(encoding="utf-8", errors="strict")

    def setting(name: str) -> str:
        match = re.search(rf'^set\({re.escape(name)} "([^"]*)"\)$', text, re.MULTILINE)
        require(match is not None, f"missing {name} in CMake compiler record")
        return match.group(1)

    return {
        "id": setting("CMAKE_CXX_COMPILER_ID"),
        "version": setting("CMAKE_CXX_COMPILER_VERSION"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = args.repository.resolve()
    build = args.build.resolve()
    output = args.output.resolve()
    cache_path = build / "CMakeCache.txt"
    commands_path = build / "compile_commands.json"
    require(cache_path.is_file(), f"missing CMake cache: {cache_path}")
    require(commands_path.is_file(), f"missing compile commands: {commands_path}")

    cache = parse_cache(cache_path)
    missing_cache = [key for key in RELEVANT_CACHE_KEYS if key not in cache]
    require(not missing_cache, f"missing required CMake cache keys: {missing_cache}")
    require(cache["NBODY_ENABLE_OPENMP"] == "ON", "build is not OpenMP-enabled")
    require(cache["NBODY_ENABLE_CUDA"] == "OFF", "build unexpectedly enables CUDA")

    compiler = Path(cache["CMAKE_CXX_COMPILER"])
    ninja = Path(cache["CMAKE_MAKE_PROGRAM"])
    cmake = Path(cache.get("CMAKE_COMMAND", "cmake"))
    require(compiler.is_file(), f"compiler does not exist: {compiler}")
    require(ninja.is_file(), f"Ninja does not exist: {ninja}")

    compile_database = json.loads(commands_path.read_text(encoding="utf-8"))
    selected_commands: dict[str, str] = {}
    for suffix in REQUIRED_COMPILE_SOURCES:
        matches = [
            entry
            for entry in compile_database
            if Path(entry["file"]).as_posix().endswith(suffix)
            and "nbody_barnes_hut_openmp_schedule_test_impl" not in entry["command"]
        ]
        require(matches, f"compile command missing for {suffix}")
        if suffix == "src/barnes_hut.cpp":
            matches = [
                entry for entry in matches
                if "CMakeFiles\\nbody_barnes_hut.dir" in entry["command"]
                or "CMakeFiles/nbody_barnes_hut.dir" in entry["command"]
            ]
        require(len(matches) == 1, f"compile command is ambiguous for {suffix}")
        selected_commands[suffix] = matches[0]["command"]

    ninja_commands = run_text(
        [str(ninja), "-C", str(build), "-t", "commands", "nbody_barnes_hut_openmp_benchmark"]
    ).splitlines()
    link_matches = [
        line for line in ninja_commands
        if "nbody_barnes_hut_openmp_benchmark.exe" in line and " -o " in line
    ]
    require(len(link_matches) == 1, "benchmark linker command was not uniquely identified")

    replacements = (
        (str(build), "${BUILD_DIR}"),
        (build.as_posix(), "${BUILD_DIR}"),
        (str(repository), "${REPOSITORY_ROOT}"),
        (repository.as_posix(), "${REPOSITORY_ROOT}"),
    )

    def sanitize(value: str) -> str:
        result = value
        for original, replacement in replacements:
            result = re.sub(re.escape(original), replacement, result, flags=re.IGNORECASE)
        # Do not globally rewrite backslashes: compile_commands.json uses them
        # both in Windows paths and to escape quoted preprocessor definitions.
        # Only the two explicit absolute roots above are normalized.
        return result

    compiler_record = compiler_identity(build)
    record = {
        "schema_version": 1,
        "scope": "OpenMP Barnes-Hut Stage 2 exploratory benchmark build",
        "sanitization": {
            "repository_root": "${REPOSITORY_ROOT}",
            "build_directory": "${BUILD_DIR}",
            "note": "Only repository/build absolute paths are replaced; toolchain and runtime paths remain recorded.",
        },
        "toolchain": {
            "compiler_executable": sanitize(str(compiler)),
            "compiler_id": compiler_record["id"],
            "compiler_version": compiler_record["version"],
            "compiler_version_output": run_text([str(compiler), "--version"]),
            "cmake_executable": sanitize(str(cmake)),
            "cmake_version_output": run_text([str(cmake), "--version"]).splitlines()[0],
            "generator": cache["CMAKE_GENERATOR"],
            "ninja_executable": sanitize(str(ninja)),
            "ninja_version": run_text([str(ninja), "--version"]),
        },
        "cmake_cache": {
            key: sanitize(cache[key]) for key in RELEVANT_CACHE_KEYS
        },
        "effective_commands": {
            "compile": {
                source: sanitize(command)
                for source, command in sorted(selected_commands.items())
            },
            "benchmark_link": sanitize(link_matches[0]),
        },
        "limitations": [
            "This records one effective local build; it does not guarantee bitwise-reproducible timings on another machine.",
            "Host scheduling, affinity, clocks, thermals, and background load remain uncontrolled.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
