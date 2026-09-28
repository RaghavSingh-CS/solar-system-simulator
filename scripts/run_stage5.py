#!/usr/bin/env python3
"""Prepare or run the frozen Stage 5 same-machine benchmark protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "stage5-v1.0"
CALIBRATION_BODIES = (64, 128, 192, 256, 512, 1024, 2048, 4096)
HELD_OUT_BODIES = (96, 160, 384, 768, 1536, 3072)
FULL_BODIES = tuple(sorted((*CALIBRATION_BODIES, *HELD_OUT_BODIES)))
PILOT_BODIES = (64, 512, 4096)
ENERGY_BOUND = 5.0e-6
ANGULAR_BOUND = 5.0e-10
DEFAULT_ORDER_SEED = 20260926
DEFAULT_BENCHMARK_SEED = 24301
RAW_FIELDS = (
    "environment_id", "source_commit", "protocol_version", "provenance_mode",
    "experiment_id", "mode", "dataset_role", "run_id", "order_seed", "block_id",
    "order_index", "repetition", "backend", "requested_threads", "effective_threads",
    "bodies", "steps", "dt_s", "benchmark_seed", "warmups", "timing_metric",
    "elapsed_ms", "correctness_status", "checksum", "relative_energy_drift",
    "relative_angular_momentum_drift", "energy_drift_bound", "angular_momentum_drift_bound",
    "build_configuration", "openmp_specification_date", "setup_allocation_ms", "host_staging_ms", "host_to_device_ms",
    "device_to_host_ms", "device_index", "gpu_name", "compute_capability",
    "total_global_memory_bytes", "multiprocessor_count", "cuda_driver_version",
    "cuda_runtime_version",
)
SUMMARY_FIELDS = (
    "mode", "dataset_role", "backend", "requested_threads", "bodies", "timing_metric",
    "samples", "median_ms", "mean_ms", "stdev_ms", "coefficient_of_variation",
    "min_ms", "max_ms", "iqr_ms", "median_ci95_low_ms", "median_ci95_high_ms",
)


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


def integer_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique comma-separated positive integers")
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--dry-run", choices=("pilot", "full"), metavar="MODE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omp-threads", type=integer_list, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument("--cpu-executable", type=Path, default=Path("build-stage5/nbody_benchmark.exe"))
    parser.add_argument("--cuda-executable", type=Path, default=Path("build-stage5/nbody_cuda_benchmark.exe"))
    parser.add_argument("--build-dir", type=Path, default=Path("build-stage5"))
    parser.add_argument("--ctest-log", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--environment-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--expected-gpu", default="RTX 4050")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--pilot-verification", type=Path)
    parser.add_argument("--pilot-approval", type=Path)
    return parser.parse_args()


def run_command(command: list[str], environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
    except OSError as error:
        return subprocess.CompletedProcess(command, 127, "", str(error))


def command_record(command: list[str]) -> dict[str, Any]:
    completed = run_command(command)
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_value(*arguments: str) -> str | None:
    safe = f"safe.directory={Path.cwd().resolve().as_posix()}"
    completed = run_command(["git", "-c", safe, *arguments])
    return completed.stdout.strip() if completed.returncode == 0 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def measurement_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["OMP_PROC_BIND"] = "FALSE"
    for name in ("OMP_PLACES", "GOMP_CPU_AFFINITY", "OMP_DISPLAY_ENV", "OMP_DISPLAY_AFFINITY"):
        environment.pop(name, None)
    return environment


def dataset_role(bodies: int, mode: str) -> str:
    if mode == "pilot":
        return "pilot_only"
    return "calibration" if bodies in CALIBRATION_BODIES else "held_out"


def configurations(bodies: tuple[int, ...], threads: tuple[int, ...], mode: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for body_count in bodies:
        role = dataset_role(body_count, mode)
        result.append({"backend": "sequential", "threads": 1, "bodies": body_count, "role": role})
        result.extend(
            {"backend": "openmp", "threads": thread_count, "bodies": body_count, "role": role}
            for thread_count in threads
        )
        result.append({"backend": "cuda", "threads": 0, "bodies": body_count, "role": role})
    return result


def ordered_runs(configs: list[dict[str, Any]], repetitions: int, seed: int) -> list[dict[str, Any]]:
    generator = XorShift64Star(seed)
    runs: list[dict[str, Any]] = []
    for block in range(1, repetitions + 1):
        values = [dict(config, block_id=block, repetition=block) for config in configs]
        generator.shuffle(values)
        runs.extend(values)
    return runs


def parse_csv_row(completed: subprocess.CompletedProcess[str], command: list[str]) -> dict[str, str]:
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    if completed.stderr.strip():
        raise RuntimeError(f"benchmark wrote unexpected stderr: {' '.join(command)}\n{completed.stderr}")
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"expected one benchmark row, received {len(rows)}: {' '.join(command)}")
    return rows[0]


def finite_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {field}: {row[field]}")
    return value


def benchmark_command(args: argparse.Namespace, config: dict[str, Any], steps: int) -> list[str]:
    if config["backend"] == "cuda":
        return [
            str(args.cuda_executable.resolve()), "--bodies", str(config["bodies"]),
            "--steps", str(steps), "--dt", "10", "--warmups", "1", "--repetitions", "1",
            "--seed", str(args.seed), "--device", str(args.device),
        ]
    return [
        str(args.cpu_executable.resolve()), "--backend", str(config["backend"]),
        "--bodies", str(config["bodies"]), "--steps", str(steps), "--dt", "10",
        "--threads", str(config["threads"] or 1), "--warmups", "1", "--repetitions", "1",
        "--seed", str(args.seed),
    ]


def normalized_rows(
    args: argparse.Namespace,
    config: dict[str, Any],
    benchmark: dict[str, str],
    mode: str,
    steps: int,
    block_id: int,
    order_index: int,
) -> list[dict[str, str]]:
    backend = str(config["backend"])
    energy = finite_float(benchmark, "relative_energy_drift")
    angular = finite_float(benchmark, "relative_angular_momentum_drift")
    valid = energy <= ENERGY_BOUND and angular <= ANGULAR_BOUND
    if backend == "cuda":
        valid = valid and benchmark["correctness_status"] == "pass"
        requested_threads = "0"
        effective_threads = "0"
        metrics = (
            ("device_compute", finite_float(benchmark, "device_compute_ms")),
            ("transfer_inclusive", finite_float(benchmark, "transfer_inclusive_ms")),
            ("selector_steady_state", finite_float(benchmark, "backend_wall_ms")),
        )
        if metrics[1][1] < metrics[0][1]:
            raise RuntimeError("CUDA transfer-inclusive timing is less than device-compute timing")
        gpu_name = benchmark["gpu_name"]
        if args.expected_gpu.casefold() not in gpu_name.casefold():
            raise RuntimeError(f"CUDA device mismatch: expected {args.expected_gpu!r}, got {gpu_name!r}")
    else:
        requested_threads = benchmark["requested_threads"]
        effective_threads = benchmark["effective_threads"]
        if requested_threads != effective_threads:
            raise RuntimeError(
                f"OpenMP team mismatch: requested {requested_threads}, effective {effective_threads}"
            )
        metrics = (
            ("integration_compute", finite_float(benchmark, "median_s") * 1000.0),
            ("selector_steady_state", finite_float(benchmark, "selector_steady_state_s") * 1000.0),
        )
        if metrics[1][1] < metrics[0][1]:
            raise RuntimeError("CPU selector timing is less than integration-only timing")
        gpu_name = ""
    if not valid:
        raise RuntimeError(
            f"correctness gate failed for {backend} N={config['bodies']}: energy={energy} angular={angular}"
        )

    run_id = f"b{block_id:02d}-o{order_index:04d}"
    common = {
        "environment_id": args.environment_id,
        "source_commit": args.source_commit,
        "protocol_version": PROTOCOL_VERSION,
        "provenance_mode": "git-verified",
        "experiment_id": args.experiment_id,
        "mode": mode,
        "dataset_role": str(config["role"]),
        "run_id": run_id,
        "order_seed": str(args.order_seed),
        "block_id": str(block_id),
        "order_index": str(order_index),
        "repetition": str(block_id),
        "backend": backend,
        "requested_threads": requested_threads,
        "effective_threads": effective_threads,
        "bodies": str(config["bodies"]),
        "steps": str(steps),
        "dt_s": "10",
        "benchmark_seed": str(args.seed),
        "warmups": "1",
        "correctness_status": "pass",
        "checksum": benchmark["checksum"],
        "relative_energy_drift": f"{energy:.17g}",
        "relative_angular_momentum_drift": f"{angular:.17g}",
        "energy_drift_bound": f"{ENERGY_BOUND:.17g}",
        "angular_momentum_drift_bound": f"{ANGULAR_BOUND:.17g}",
        "build_configuration": benchmark["build_configuration"],
        "openmp_specification_date": benchmark.get("openmp_specification_date", ""),
        "setup_allocation_ms": benchmark.get("allocation_ms", ""),
        "host_staging_ms": benchmark.get("host_staging_ms", ""),
        "host_to_device_ms": benchmark.get("host_to_device_ms", ""),
        "device_to_host_ms": benchmark.get("device_to_host_ms", ""),
        "device_index": benchmark.get("device_index", ""),
        "gpu_name": gpu_name,
        "compute_capability": benchmark.get("compute_capability", ""),
        "total_global_memory_bytes": benchmark.get("total_global_memory_bytes", ""),
        "multiprocessor_count": benchmark.get("multiprocessor_count", ""),
        "cuda_driver_version": benchmark.get("cuda_driver_version", ""),
        "cuda_runtime_version": benchmark.get("cuda_runtime_version", ""),
    }
    rows: list[dict[str, str]] = []
    for metric, elapsed in metrics:
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise RuntimeError(f"invalid {metric} timing: {elapsed}")
        rows.append({**common, "timing_metric": metric, "elapsed_ms": f"{elapsed:.17g}"})
    return rows


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


def summary_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset_role"], row["backend"], int(row["requested_threads"]),
            int(row["bodies"]), row["timing_metric"],
        )
        grouped[key].append(float(row["elapsed_ms"]))
    result: list[dict[str, str]] = []
    for key in sorted(grouped):
        role, backend, threads, bodies, metric = key
        values = grouped[key]
        mean = statistics.mean(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        low, high = bootstrap_interval(
            values, args.bootstrap_resamples, group_seed(args.order_seed, key)
        )
        result.append({
            "mode": "full" if args.full else "pilot",
            "dataset_role": role,
            "backend": backend,
            "requested_threads": str(threads),
            "bodies": str(bodies),
            "timing_metric": metric,
            "samples": str(len(values)),
            "median_ms": f"{statistics.median(values):.17g}",
            "mean_ms": f"{mean:.17g}",
            "stdev_ms": f"{deviation:.17g}",
            "coefficient_of_variation": f"{deviation / mean:.17g}",
            "min_ms": f"{min(values):.17g}",
            "max_ms": f"{max(values):.17g}",
            "iqr_ms": f"{quantile(values, 0.75) - quantile(values, 0.25):.17g}",
            "median_ci95_low_ms": f"{low:.17g}",
            "median_ci95_high_ms": f"{high:.17g}",
        })
    return result


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_collected_rows(
    rows: list[dict[str, str]], schedule: list[dict[str, Any]]
) -> None:
    expected_count = sum(3 if item["backend"] == "cuda" else 2 for item in schedule)
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} tidy rows, received {len(rows)}")
    by_order: dict[int, list[dict[str, str]]] = defaultdict(list)
    checksums: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    cpu_checksums: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        by_order[int(row["order_index"])].append(row)
    if set(by_order) != set(range(1, len(schedule) + 1)):
        raise RuntimeError("collected order indices do not match the generated schedule")
    for order_index, config in enumerate(schedule, start=1):
        selected = by_order[order_index]
        expected_metrics = (
            {"device_compute", "transfer_inclusive", "selector_steady_state"}
            if config["backend"] == "cuda"
            else {"integration_compute", "selector_steady_state"}
        )
        if {row["timing_metric"] for row in selected} != expected_metrics:
            raise RuntimeError(f"metric set mismatch at order {order_index}")
        first = selected[0]
        key = (first["backend"], int(first["requested_threads"]), int(first["bodies"]))
        checksums[key].add(first["checksum"])
        if first["backend"] != "cuda":
            cpu_checksums[int(first["bodies"])].add(first["checksum"])
    if any(len(values) != 1 for values in checksums.values()):
        raise RuntimeError("repeat checksum mismatch")
    if any(len(values) != 1 for values in cpu_checksums.values()):
        raise RuntimeError("sequential/OpenMP checksum mismatch")


def windows_snapshot() -> dict[str, Any]:
    if platform.system() != "Windows":
        return {"available": False, "reason": "not Windows"}
    script = (
        "$ErrorActionPreference='Stop';"
        "$cpu=Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,SocketDesignation,"
        "NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed;"
        "$os=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize;"
        "$memory=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;"
        "[pscustomobject]@{cpu=$cpu;os=$os;total_physical_memory_bytes=$memory}|ConvertTo-Json -Compress -Depth 5"
    )
    completed = run_command(["powershell", "-NoProfile", "-Command", script])
    if completed.returncode != 0:
        return {"available": False, "reason": completed.stderr.strip()}
    return {"available": True, "snapshot": json.loads(completed.stdout)}


def thermal_snapshot() -> dict[str, Any]:
    query = command_record([
        "nvidia-smi",
        "--query-gpu=timestamp,name,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,pstate",
        "--format=csv,noheader,nounits",
    ])
    return {
        "gpu_sample_available": query["return_code"] == 0,
        "gpu_sample": query,
        "cpu_temperature": {"available": False, "reason": "no portable privileged CPU sensor required"},
    }


def cmake_cache(build_dir: Path) -> dict[str, str]:
    path = build_dir.resolve() / "CMakeCache.txt"
    if not path.is_file():
        return {}
    wanted = {
        "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_VERSION",
        "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_COMPILER_VERSION", "CMAKE_CUDA_FLAGS", "CMAKE_CUDA_FLAGS_RELEASE",
        "CMAKE_CUDA_ARCHITECTURES", "CMAKE_GENERATOR", "NBODY_ENABLE_OPENMP", "NBODY_ENABLE_CUDA",
    }
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line:
            continue
        left, value = line.split("=", 1)
        name = left.split(":", 1)[0]
        if name in wanted:
            result[name] = value
    return result


def validate_common_arguments(args: argparse.Namespace) -> None:
    if tuple(sorted(args.omp_threads)) != args.omp_threads or 1 not in args.omp_threads:
        raise ValueError("--omp-threads must be sorted, unique, and include 1")
    logical = os.cpu_count() or 1
    if max(args.omp_threads) > logical and not args.dry_run:
        raise ValueError(f"requested OpenMP count exceeds detected logical processors ({logical})")
    if args.device < 0 or args.bootstrap_resamples <= 0:
        raise ValueError("device must be non-negative and bootstrap resamples positive")


def validate_git_and_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if platform.system() != "Windows":
        raise RuntimeError("Stage 5 measurement and preflight require Windows")
    if not args.source_commit or not args.environment_id:
        raise ValueError("--source-commit and --environment-id are required")
    head = git_value("rev-parse", "HEAD")
    branch = git_value("branch", "--show-current")
    status = git_value("status", "--porcelain=v1")
    if head != args.source_commit:
        raise RuntimeError(f"source commit mismatch: expected {args.source_commit}, got {head}")
    if status:
        raise RuntimeError("Git working tree must be clean before creating Stage 5 output")
    for path in (args.cpu_executable.resolve(), args.cuda_executable.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
        probe = run_command([str(path), "--help"], measurement_environment())
        if probe.returncode != 0:
            raise RuntimeError(f"executable help probe failed: {path}\n{probe.stderr}")
    if args.ctest_log is None or not args.ctest_log.resolve().is_file():
        raise FileNotFoundError("--ctest-log must identify the successful Stage 5 CTest log")
    ctest_text = args.ctest_log.resolve().read_text(encoding="utf-8", errors="replace")
    if "100% tests passed" not in ctest_text or "0 tests failed" not in ctest_text:
        raise RuntimeError("CTest log does not establish full success")
    gpu_query = command_record([
        "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader",
    ])
    if gpu_query["return_code"] != 0:
        raise RuntimeError(f"nvidia-smi query failed: {gpu_query['stderr']}")
    lines = [line for line in str(gpu_query["stdout"]).splitlines() if line.strip()]
    selected = next(
        (line for line in lines if line.split(",", 1)[0].strip() == str(args.device)),
        None,
    )
    if selected is None or args.expected_gpu.casefold() not in selected.casefold():
        raise RuntimeError(
            f"CUDA device {args.device} does not match expected {args.expected_gpu!r}: {lines}"
        )
    return {"head": head, "branch": branch, "status_before_output": status, "gpu_query": gpu_query}


def validate_full_gate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.full:
        return {}
    if args.pilot_verification is None or args.pilot_approval is None:
        raise ValueError("--full requires --pilot-verification and --pilot-approval")
    verification = json.loads(args.pilot_verification.resolve().read_text(encoding="utf-8"))
    approval = json.loads(args.pilot_approval.resolve().read_text(encoding="utf-8"))
    required = {
        "verification_status": "pass",
        "mode": "pilot",
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": args.source_commit,
        "environment_id": args.environment_id,
    }
    for key, expected in required.items():
        if verification.get(key) != expected:
            raise RuntimeError(f"pilot verification {key} mismatch: {verification.get(key)!r}")
    if verification.get("openmp_threads") != list(args.omp_threads):
        raise RuntimeError("full OpenMP list differs from verified pilot")
    if approval.get("approved") is not True:
        raise RuntimeError("pilot approval must contain approved=true")
    for key in ("protocol_version", "source_commit", "environment_id"):
        if approval.get(key) != required[key]:
            raise RuntimeError(f"pilot approval {key} mismatch")
    if approval.get("openmp_threads") != list(args.omp_threads):
        raise RuntimeError("pilot approval OpenMP list differs from full request")
    for field in ("reviewer", "reviewed_utc", "notes"):
        if not str(approval.get(field, "")).strip():
            raise RuntimeError(f"pilot approval requires non-empty {field}")
    return {
        "verification": str(args.pilot_verification.resolve()),
        "verification_sha256": sha256(args.pilot_verification.resolve()),
        "approval_path": str(args.pilot_approval.resolve()),
        "approval_sha256": sha256(args.pilot_approval.resolve()),
        "approval": approval,
    }


def environment_metadata(
    args: argparse.Namespace,
    mode: str,
    git: dict[str, Any],
    started: datetime,
    completed: datetime,
    rows: list[dict[str, str]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    cuda_row = next((row for row in rows if row["backend"] == "cuda"), None)
    openmp_row = next((row for row in rows if row["backend"] == "openmp"), None)
    return {
        "experiment_id": args.experiment_id,
        "environment_id": args.environment_id,
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "duration_seconds": (completed - started).total_seconds(),
        "source_provenance": {
            "mode": "git-verified",
            "source_commit": args.source_commit,
            "git_head": git["head"],
            "branch": git["branch"],
            "status_before_output": git["status_before_output"],
        },
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "version": platform.version(), "machine": platform.machine(),
            "python": platform.python_version(), "logical_processors": os.cpu_count(),
        },
        "windows": windows_snapshot(),
        "tools": {
            "git": command_record(["git", "--version"]),
            "cmake": command_record(["cmake", "--version"]),
            "ninja": command_record(["ninja", "--version"]),
            "compiler": command_record(["cl"]),
            "nvcc": command_record(["nvcc", "--version"]),
            "nvidia_smi": command_record(["nvidia-smi"]),
            "nvidia_smi_query": git["gpu_query"],
        },
        "build": {
            "cmake_cache": cmake_cache(args.build_dir),
            "cpu_executable": str(args.cpu_executable.resolve()),
            "cpu_executable_sha256": sha256(args.cpu_executable.resolve()),
            "cuda_executable": str(args.cuda_executable.resolve()),
            "cuda_executable_sha256": sha256(args.cuda_executable.resolve()),
        },
        "openmp_environment": {
            "requested_threads": list(args.omp_threads),
            "openmp_specification_date": (
                int(openmp_row["openmp_specification_date"])
                if openmp_row and openmp_row["openmp_specification_date"] else None
            ),
            "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "FALSE",
            "OMP_PLACES": "unset", "GOMP_CPU_AFFINITY": "unset",
            "placement_claim": "Windows scheduler placement; affinity is not claimed as controlled",
        },
        "observed_cuda_device": None if cuda_row is None else {
            "device_index": int(cuda_row["device_index"]),
            "name": cuda_row["gpu_name"],
            "compute_capability": cuda_row["compute_capability"],
            "total_global_memory_bytes": int(cuda_row["total_global_memory_bytes"]),
            "multiprocessor_count": int(cuda_row["multiprocessor_count"]),
            "cuda_driver_version": int(cuda_row["cuda_driver_version"]),
            "cuda_runtime_version": int(cuda_row["cuda_runtime_version"]),
        },
        "telemetry": thermal_snapshot(),
        "protocol": {
            "calibration_bodies": list(CALIBRATION_BODIES),
            "held_out_bodies": list(HELD_OUT_BODIES),
            "measured_bodies": list(PILOT_BODIES if mode == "pilot" else FULL_BODIES),
            "steps": 10 if mode == "pilot" else 100,
            "dt_s": 10.0,
            "warmups_before_each_measurement": 1,
            "repetitions": 3 if mode == "pilot" else 11,
            "benchmark_seed": args.seed,
            "order_seed": args.order_seed,
            "ordering": "xorshift64* Fisher-Yates shuffle of every complete repetition block",
            "energy_drift_bound": ENERGY_BOUND,
            "angular_momentum_drift_bound": ANGULAR_BOUND,
            "bootstrap_resamples": args.bootstrap_resamples,
            "raw_rows": len(rows),
        },
        "full_run_pilot_gate": gate,
    }


def make_manifest(output: Path, mode: str, source_commit: str) -> dict[str, Any]:
    names = ("raw_timings.csv", "summary.csv", "environment.json", "ctest.txt")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "source_commit": source_commit,
        "files": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size} for name in names},
    }


def dry_run(args: argparse.Namespace) -> int:
    mode = str(args.dry_run)
    bodies = FULL_BODIES if mode == "full" else PILOT_BODIES
    repetitions = 11 if mode == "full" else 3
    configs = configurations(bodies, args.omp_threads, mode)
    schedule = ordered_runs(configs, repetitions, args.order_seed)
    args.output.resolve().mkdir(parents=True, exist_ok=False)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "calibration_bodies": list(CALIBRATION_BODIES),
        "held_out_bodies": list(HELD_OUT_BODIES),
        "openmp_threads": list(args.omp_threads),
        "configuration_count_per_block": len(configs),
        "blocks": repetitions,
        "underlying_runs": len(schedule),
        "expected_tidy_rows": sum(3 if item["backend"] == "cuda" else 2 for item in schedule),
        "schedule": schedule,
    }
    path = args.output.resolve() / "schedule.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote deterministic schedule to {path}")
    return 0


def main() -> int:
    args = parse_arguments()
    validate_common_arguments(args)
    if args.dry_run:
        return dry_run(args)

    git = validate_git_and_inputs(args)
    gate = validate_full_gate(args)
    mode = "full" if args.full else "pilot" if args.pilot else "preflight"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.preflight:
        payload = environment_metadata(
            args, mode, git, datetime.now(timezone.utc), datetime.now(timezone.utc), [], gate
        )
        path = output / "preflight.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Stage 5 preflight passed; wrote {path}")
        return 0

    if not args.experiment_id:
        raise ValueError("--experiment-id is required for pilot/full")
    bodies = PILOT_BODIES if args.pilot else FULL_BODIES
    steps = 10 if args.pilot else 100
    repetitions = 3 if args.pilot else 11
    configs = configurations(bodies, args.omp_threads, mode)
    schedule = ordered_runs(configs, repetitions, args.order_seed)
    environment = measurement_environment()
    partial = output / "raw_timings.partial.csv"
    raw_path = output / "raw_timings.csv"
    failure_path = output / "failure.json"
    rows: list[dict[str, str]] = []
    started = datetime.now(timezone.utc)
    try:
        with partial.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for order_index, config in enumerate(schedule, start=1):
                command = benchmark_command(args, config, steps)
                completed = run_command(command, environment)
                try:
                    benchmark = parse_csv_row(completed, command)
                    normalized = normalized_rows(
                        args, config, benchmark, mode, steps,
                        int(config["block_id"]), order_index,
                    )
                except Exception as error:
                    failure_path.write_text(json.dumps({
                        "protocol_version": PROTOCOL_VERSION,
                        "mode": mode,
                        "block_id": config["block_id"],
                        "order_index": order_index,
                        "configuration": config,
                        "command": command,
                        "return_code": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "error": str(error),
                    }, indent=2) + "\n", encoding="utf-8")
                    raise
                writer.writerows(normalized)
                stream.flush()
                rows.extend(normalized)
                selector = next(row for row in normalized if row["timing_metric"] == "selector_steady_state")
                print(
                    f"[{order_index:04d}/{len(schedule)}] block={config['block_id']} "
                    f"N={config['bodies']} {config['backend']} t={config['threads']} "
                    f"selector={selector['elapsed_ms']}ms",
                    flush=True,
                )
        validate_collected_rows(rows, schedule)
        partial.replace(raw_path)
    except Exception:
        print(f"partial evidence retained at {partial}; failure detail at {failure_path}", file=sys.stderr)
        raise

    summaries = summary_rows(rows, args)
    write_csv(output / "summary.csv", SUMMARY_FIELDS, summaries)
    shutil.copyfile(args.ctest_log.resolve(), output / "ctest.txt")
    completed_at = datetime.now(timezone.utc)
    metadata = environment_metadata(args, mode, git, started, completed_at, rows, gate)
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = make_manifest(output, mode, str(args.source_commit))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} tidy timing rows to {raw_path}")
    print("run scripts/verify_stage5_evidence.py before interpreting any result")
    return 0


if __name__ == "__main__":
    # v1.0's mixed pilot/full entry point is retired. Keep its implementation
    # above only as historical source context; all operator calls dispatch to
    # the phase-separated v2.0 implementation.
    from stage5_common import Stage5Error
    from run_stage5_v2 import main as stage5_v2_main

    try:
        raise SystemExit(stage5_v2_main())
    except (Stage5Error, OSError, ValueError) as exc:
        print(f"stage5-v2.0 error: {exc}", file=sys.stderr)
        raise SystemExit(2)
