#!/usr/bin/env python3
"""Shared deterministic primitives for Stage 5 v2.0 tooling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_VERSION = "stage5-v2.0"
POLICY_ALGORITHM_VERSION = "stage5-loglog-policy-v1"
CALIBRATION_BODIES = (64, 128, 192, 256, 512, 1024, 2048, 4096)
HELD_OUT_BODIES = (96, 160, 384, 768, 1536, 3072)
PILOT_BODIES = (64, 512, 4096)
MODE_BODIES = {
    "pilot": PILOT_BODIES,
    "calibration": CALIBRATION_BODIES,
    "held_out": HELD_OUT_BODIES,
}
MODE_ROLE = {
    "pilot": "pilot_only",
    "calibration": "calibration",
    "held_out": "held_out",
}
MODE_STEPS = {"pilot": 10, "calibration": 100, "held_out": 100}
MODE_REPETITIONS = {"pilot": 3, "calibration": 11, "held_out": 11}
ENERGY_BOUND = 5.0e-6
ANGULAR_BOUND = 5.0e-10
DEFAULT_ORDER_SEED = 20260926
DEFAULT_BENCHMARK_SEED = 24301
CUDA_ARCHITECTURE = "89"
PRIMARY_METRIC = "selector_steady_state"


class Stage5Error(RuntimeError):
    """A controlled Stage 5 validation failure."""


class XorShift64Star:
    def __init__(self, seed: int) -> None:
        self.state = seed & ((1 << 64) - 1) or 0x9E3779B97F4A7C15

    def next_u64(self) -> int:
        value = self.state
        value ^= value >> 12
        value ^= (value << 25) & ((1 << 64) - 1)
        value ^= value >> 27
        self.state = value
        return (value * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)

    def randbelow(self, upper_bound: int) -> int:
        if upper_bound <= 0:
            raise ValueError("upper bound must be positive")
        limit = (1 << 64) - ((1 << 64) % upper_bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper_bound

    def shuffle(self, values: list[dict[str, Any]]) -> None:
        for index in range(len(values) - 1, 0, -1):
            replacement = self.randbelow(index + 1)
            values[index], values[replacement] = values[replacement], values[index]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage5Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def run_command(
    command: list[str], environment: dict[str, str] | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(command, 127, "", str(error))


def command_record(command: list[str]) -> dict[str, Any]:
    completed = run_command(command)
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_value(source_dir: Path, *arguments: str) -> str | None:
    safe = f"safe.directory={source_dir.resolve().as_posix()}"
    completed = run_command(["git", "-C", str(source_dir.resolve()), "-c", safe, *arguments])
    return completed.stdout.strip() if completed.returncode == 0 else None


def require_absent_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise Stage5Error(f"{label} must be absent: {resolved}")
    return resolved


def derive_openmp_threads(physical_cores: int, logical_processors: int) -> tuple[int, ...]:
    require(physical_cores > 0, "physical core count must be positive")
    require(logical_processors > 0, "logical processor count must be positive")
    require(
        physical_cores <= logical_processors,
        "physical core count cannot exceed logical processor count",
    )
    values = {1, physical_cores, logical_processors}
    power = 2
    while power < physical_cores:
        values.add(power)
        power *= 2
    return tuple(sorted(value for value in values if value <= logical_processors))


def validate_openmp_threads(
    threads: Iterable[int], physical_cores: int, logical_processors: int
) -> tuple[int, ...]:
    actual = tuple(threads)
    expected = derive_openmp_threads(physical_cores, logical_processors)
    require(actual == expected, f"OpenMP topology list mismatch: expected {expected}, got {actual}")
    return actual


def windows_topology() -> dict[str, Any]:
    script = (
        "$ErrorActionPreference='Stop';"
        "$cpus=@(Get-CimInstance Win32_Processor | Select-Object Name,SocketDesignation,"
        "NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed);"
        "if($cpus.Count -eq 0){throw 'no Win32_Processor records'};"
        "$system=Get-CimInstance Win32_ComputerSystem;"
        "$os=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture;"
        "$p=($cpus|Measure-Object NumberOfCores -Sum).Sum;"
        "$l=($cpus|Measure-Object NumberOfLogicalProcessors -Sum).Sum;"
        "[pscustomobject]@{processors=$cpus;physical_cores=[int]$p;logical_processors=[int]$l;"
        "total_physical_memory_bytes=[int64]$system.TotalPhysicalMemory;os=$os}|"
        "ConvertTo-Json -Compress -Depth 6"
    )
    completed = run_command(["powershell", "-NoProfile", "-Command", script])
    require(completed.returncode == 0, f"reliable Windows topology capture failed: {completed.stderr}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise Stage5Error("Windows topology capture returned invalid JSON") from error
    physical = int(value.get("physical_cores", 0))
    logical = int(value.get("logical_processors", 0))
    threads = derive_openmp_threads(physical, logical)
    value["derivation"] = "sorted unique {1, powers of two below P, P, L}, capped at L"
    value["openmp_threads"] = list(threads)
    return value


def configuration_id(backend: str, threads: int) -> str:
    return f"{backend}:{threads}"


def canonical_configuration_order(threads: Iterable[int]) -> tuple[str, ...]:
    values = ["sequential:1"]
    values.extend(f"openmp:{thread}" for thread in sorted(threads))
    values.append("cuda:0")
    return tuple(values)


def configurations(mode: str, threads: tuple[int, ...]) -> list[dict[str, Any]]:
    require(mode in MODE_BODIES, f"unsupported acquisition mode: {mode}")
    result: list[dict[str, Any]] = []
    for bodies in MODE_BODIES[mode]:
        result.append(
            {"backend": "sequential", "threads": 1, "bodies": bodies, "role": MODE_ROLE[mode]}
        )
        result.extend(
            {"backend": "openmp", "threads": thread, "bodies": bodies, "role": MODE_ROLE[mode]}
            for thread in threads
        )
        result.append({"backend": "cuda", "threads": 0, "bodies": bodies, "role": MODE_ROLE[mode]})
    return result


def ordered_runs(
    mode: str, threads: tuple[int, ...], repetitions: int, seed: int
) -> list[dict[str, Any]]:
    generator = XorShift64Star(seed)
    base = configurations(mode, threads)
    runs: list[dict[str, Any]] = []
    for block in range(1, repetitions + 1):
        values = [dict(configuration, block_id=block, repetition=block) for configuration in base]
        generator.shuffle(values)
        runs.extend(values)
    return runs


def finite_float(value: str | float | int, label: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite {label}: {value}")
    return parsed


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def group_seed(order_seed: int, key: tuple[str, str, int, int, str]) -> int:
    role, backend, threads, bodies, metric = key
    text = f"{role}|{backend}|{threads}|{bodies}|{metric}"
    mixed = order_seed ^ (bodies << 32) ^ (threads << 16)
    for index, character in enumerate(text, start=1):
        mixed ^= index * ord(character)
        mixed = ((mixed << 7) | (mixed >> 57)) & ((1 << 64) - 1)
    return mixed


def bootstrap_interval(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    generator = XorShift64Star(seed)
    medians: list[float] = []
    for _ in range(resamples):
        sample = [values[generator.randbelow(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    return quantile(medians, 0.025), quantile(medians, 0.975)


def measurement_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["OMP_PROC_BIND"] = "FALSE"
    for name in ("OMP_PLACES", "GOMP_CPU_AFFINITY", "OMP_DISPLAY_ENV", "OMP_DISPLAY_AFFINITY"):
        environment.pop(name, None)
    return environment


def parse_cmake_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key, _ = key_type.split(":", 1)
        cache[key] = value
    return cache


def manifest_hash(evidence_dir: Path) -> str:
    return sha256_file(evidence_dir.resolve() / "manifest.json")
