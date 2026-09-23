#!/usr/bin/env python3
"""Run the controlled Stage 3 sequential/OpenMP benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BODIES = (64, 96, 128, 160, 192, 256, 384, 512, 768, 1024)
DEFAULT_THREADS = (1, 2, 4, 8, 16)
RAW_PREFIX_FIELDS = (
    "experiment_id",
    "order_seed",
    "benchmark_seed",
    "block_id",
    "order_index",
    "repetition",
)


class XorShift64Star:
    """Small explicitly specified PRNG used only for reproducible ordering."""

    def __init__(self, seed: int) -> None:
        self.state = seed & ((1 << 64) - 1)
        if self.state == 0:
            self.state = 0x9E3779B97F4A7C15

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

    def shuffle(self, values: list[dict[str, object]]) -> None:
        for index in range(len(values) - 1, 0, -1):
            replacement = self.randbelow(index + 1)
            values[index], values[replacement] = values[replacement], values[index]


def parse_integer_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-openmp/nbody_benchmark.exe"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/stage3_cpu"))
    parser.add_argument("--bodies", type=parse_integer_list, default=DEFAULT_BODIES)
    parser.add_argument("--threads", type=parse_integer_list, default=DEFAULT_THREADS)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--benchmark-seed", type=int, default=24301)
    parser.add_argument("--order-seed", type=int, default=20260923)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--physical-cores", type=int)
    parser.add_argument("--experiment-id", default="stage3-cpu-2026-09-23")
    return parser.parse_args()


def run_command(command: list[str], *, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, env=environment)


def measurement_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["OMP_PROC_BIND"] = "FALSE"
    for name in ("OMP_PLACES", "GOMP_CPU_AFFINITY", "OMP_DISPLAY_ENV", "OMP_DISPLAY_AFFINITY"):
        environment.pop(name, None)
    return environment


def benchmark_command(executable: Path, configuration: dict[str, object], arguments: argparse.Namespace) -> list[str]:
    return [
        str(executable),
        "--backend", str(configuration["backend"]),
        "--bodies", str(configuration["bodies"]),
        "--steps", str(arguments.steps),
        "--dt", str(arguments.dt),
        "--threads", str(configuration["threads"]),
        "--warmups", str(arguments.warmups),
        "--repetitions", "1",
        "--seed", str(arguments.benchmark_seed),
    ]


def parse_benchmark_row(completed: subprocess.CompletedProcess[str], command: list[str]) -> dict[str, str]:
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"expected one benchmark row, received {len(rows)}")
    if completed.stderr.strip():
        raise RuntimeError(f"benchmark wrote unexpected stderr:\n{completed.stderr}")
    return rows[0]


def configurations(arguments: argparse.Namespace) -> list[dict[str, object]]:
    logical_processors = os.cpu_count() or 1
    threads = tuple(value for value in arguments.threads if value <= logical_processors)
    if not threads:
        raise RuntimeError("no requested OpenMP thread count fits the detected logical processor count")
    result: list[dict[str, object]] = []
    for body_count in arguments.bodies:
        result.append({"backend": "sequential", "threads": 1, "bodies": body_count})
        result.extend(
            {"backend": "openmp", "threads": thread_count, "bodies": body_count}
            for thread_count in threads
        )
    return result


def ordered_runs(configs: list[dict[str, object]], repetitions: int, seed: int) -> list[dict[str, object]]:
    generator = XorShift64Star(seed)
    runs: list[dict[str, object]] = []
    for block_id in range(1, repetitions + 1):
        block = [dict(configuration, block_id=block_id, repetition=block_id) for configuration in configs]
        generator.shuffle(block)
        runs.extend(block)
    return runs


def quantile(sorted_values: list[float], probability: float) -> float:
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_median_interval(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    generator = XorShift64Star(seed)
    medians: list[float] = []
    for _ in range(resamples):
        sample = [values[generator.randbelow(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    medians.sort()
    return quantile(medians, 0.025), quantile(medians, 0.975)


def write_summary(raw_rows: list[dict[str, str]], path: Path, arguments: argparse.Namespace) -> None:
    grouped: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in raw_rows:
        key = (int(row["bodies"]), row["backend"], int(row["requested_threads"]))
        grouped[key].append(float(row["median_s"]))

    sequential_medians = {
        bodies: statistics.median(values)
        for (bodies, backend, _), values in grouped.items()
        if backend == "sequential"
    }
    fieldnames = [
        "bodies", "backend", "threads", "samples", "median_s", "mean_s", "stdev_s",
        "min_s", "max_s", "iqr_s", "median_ci95_low_s", "median_ci95_high_s",
        "coefficient_of_variation", "speedup_vs_sequential", "parallel_efficiency",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for ordinal, key in enumerate(sorted(grouped)):
            bodies, backend, threads = key
            values = sorted(grouped[key])
            median = statistics.median(values)
            mean = statistics.mean(values)
            standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
            low, high = bootstrap_median_interval(
                values,
                arguments.bootstrap_resamples,
                arguments.order_seed ^ (bodies << 16) ^ (threads << 4) ^ ordinal,
            )
            speedup = sequential_medians[bodies] / median
            writer.writerow({
                "bodies": bodies,
                "backend": backend,
                "threads": threads,
                "samples": len(values),
                "median_s": f"{median:.17g}",
                "mean_s": f"{mean:.17g}",
                "stdev_s": f"{standard_deviation:.17g}",
                "min_s": f"{values[0]:.17g}",
                "max_s": f"{values[-1]:.17g}",
                "iqr_s": f"{quantile(values, 0.75) - quantile(values, 0.25):.17g}",
                "median_ci95_low_s": f"{low:.17g}",
                "median_ci95_high_s": f"{high:.17g}",
                "coefficient_of_variation": f"{standard_deviation / mean:.17g}",
                "speedup_vs_sequential": f"{speedup:.17g}",
                "parallel_efficiency": "" if backend == "sequential" else f"{speedup / threads:.17g}",
            })


def collect_git_metadata() -> dict[str, object]:
    result: dict[str, object] = {}
    safe_directory = f"safe.directory={Path.cwd().resolve().as_posix()}"
    for key, command in {
        "head": ["git", "-c", safe_directory, "rev-parse", "HEAD"],
        "branch": ["git", "-c", safe_directory, "branch", "--show-current"],
        "status": ["git", "-c", safe_directory, "status", "--porcelain=v1"],
    }.items():
        completed = run_command(command)
        result[key] = completed.stdout.strip() if completed.returncode == 0 else None
    return result


def collect_windows_metadata() -> dict[str, object]:
    if platform.system() != "Windows":
        return {"available": False, "reason": "not Windows"}
    script = (
        "$ErrorActionPreference='Stop';"
        "$cpu=Get-CimInstance Win32_Processor | Select-Object -First 1 Name,Manufacturer,"
        "NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,CurrentClockSpeed;"
        "$os=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture;"
        "[pscustomobject]@{cpu=$cpu;os=$os}|ConvertTo-Json -Compress -Depth 4"
    )
    completed = run_command(["powershell", "-NoProfile", "-Command", script])
    if completed.returncode != 0:
        return {"available": False, "reason": completed.stderr.strip()}
    return {"available": True, "snapshot": json.loads(completed.stdout)}


def probe_thermal_telemetry() -> dict[str, object]:
    if platform.system() != "Windows":
        return {"available": False, "reason": "not Windows"}
    script = (
        "$ErrorActionPreference='Stop';"
        "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature | "
        "Select-Object InstanceName,CurrentTemperature | ConvertTo-Json -Compress"
    )
    completed = run_command(["powershell", "-NoProfile", "-Command", script])
    if completed.returncode != 0:
        return {"available": False, "reason": completed.stderr.strip()}
    return {
        "available": False,
        "reason": "ACPI thermal-zone values are not accepted as CPU package telemetry",
        "raw_probe": completed.stdout.strip(),
    }


def run_affinity_probe(executable: Path, output_path: Path, thread_count: int) -> dict[str, object]:
    environment = measurement_environment()
    environment.update({
        "OMP_PROC_BIND": "CLOSE",
        "OMP_PLACES": "CORES",
        "OMP_DISPLAY_ENV": "VERBOSE",
        "OMP_DISPLAY_AFFINITY": "TRUE",
    })
    command = [
        str(executable), "--backend", "openmp", "--bodies", "64", "--steps", "2",
        "--dt", "10", "--threads", str(thread_count), "--warmups", "0",
        "--repetitions", "1", "--seed", "24301",
    ]
    completed = run_command(command, environment=environment)
    output_path.write_text(
        "COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout +
        "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    unsupported = "Affinity not supported on this configuration" in completed.stderr
    return {
        "command": command,
        "return_code": completed.returncode,
        "requested": {"OMP_PROC_BIND": "CLOSE", "OMP_PLACES": "CORES"},
        "supported": completed.returncode == 0 and not unsupported,
        "result": "unsupported by this libgomp/Windows configuration" if unsupported else "inspect probe",
        "evidence": output_path.name,
    }


def validate_raw_rows(rows: list[dict[str, str]]) -> None:
    checksums: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        checksums[int(row["bodies"])].add(row["checksum"])
        if row["effective_threads"] != row["requested_threads"]:
            raise RuntimeError(f"runtime did not honor requested threads: {row}")
    mismatches = {bodies: values for bodies, values in checksums.items() if len(values) != 1}
    if mismatches:
        raise RuntimeError(f"cross-backend checksum mismatch: {mismatches}")


def main() -> int:
    arguments = parse_arguments()
    if arguments.steps <= 0 or arguments.repetitions < 2 or arguments.warmups <= 0:
        raise ValueError("steps and warmups must be positive; repetitions must be at least two")
    executable = arguments.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)

    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = output_directory / "raw_timings.csv"
    partial_path = output_directory / "raw_timings.partial.csv"
    summary_path = output_directory / "summary.csv"
    metadata_path = output_directory / "environment.json"
    probe_path = output_directory / "affinity_probe.txt"

    configs = configurations(arguments)
    runs = ordered_runs(configs, arguments.repetitions, arguments.order_seed)
    affinity = run_affinity_probe(executable, probe_path, min(4, os.cpu_count() or 1))
    started = datetime.now(timezone.utc)
    environment = measurement_environment()
    raw_rows: list[dict[str, str]] = []
    benchmark_fields: list[str] | None = None

    try:
        with partial_path.open("w", newline="", encoding="utf-8") as output:
            writer: csv.DictWriter[str] | None = None
            for order_index, configuration in enumerate(runs, start=1):
                command = benchmark_command(executable, configuration, arguments)
                benchmark_row = parse_benchmark_row(run_command(command, environment=environment), command)
                if benchmark_fields is None:
                    benchmark_fields = list(benchmark_row)
                    writer = csv.DictWriter(output, fieldnames=[*RAW_PREFIX_FIELDS, *benchmark_fields])
                    writer.writeheader()
                row = {
                    "experiment_id": arguments.experiment_id,
                    "order_seed": str(arguments.order_seed),
                    "benchmark_seed": str(arguments.benchmark_seed),
                    "block_id": str(configuration["block_id"]),
                    "order_index": str(order_index),
                    "repetition": str(configuration["repetition"]),
                    **benchmark_row,
                }
                assert writer is not None
                writer.writerow(row)
                output.flush()
                raw_rows.append(row)
                print(
                    f"[{order_index:03d}/{len(runs)}] N={row['bodies']} "
                    f"{row['backend']} t={row['requested_threads']} elapsed={row['median_s']}s",
                    flush=True,
                )
        validate_raw_rows(raw_rows)
        partial_path.replace(raw_path)
        write_summary(raw_rows, summary_path, arguments)
    except Exception:
        print(f"partial results retained at {partial_path}", file=sys.stderr)
        raise

    completed_at = datetime.now(timezone.utc)
    metadata = {
        "experiment_id": arguments.experiment_id,
        "started_utc": started.isoformat(),
        "completed_utc": completed_at.isoformat(),
        "duration_seconds": (completed_at - started).total_seconds(),
        "git": collect_git_metadata(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_processors": os.cpu_count(),
            "physical_cores_supplied": arguments.physical_cores,
        },
        "windows_cim_snapshot": collect_windows_metadata(),
        "thermal_telemetry": probe_thermal_telemetry(),
        "affinity": affinity,
        "measurement_environment": {
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
            "OMP_PLACES": "unset",
            "GOMP_CPU_AFFINITY": "unset",
            "note": "OS scheduler placement; affinity is not claimed as controlled",
        },
        "protocol": {
            "bodies": list(arguments.bodies),
            "openmp_threads": sorted({int(config["threads"]) for config in configs if config["backend"] == "openmp"}),
            "steps": arguments.steps,
            "dt_seconds": arguments.dt,
            "warmups_immediately_before_each_measurement": arguments.warmups,
            "measured_repetitions_per_configuration": arguments.repetitions,
            "benchmark_seed": arguments.benchmark_seed,
            "order_seed": arguments.order_seed,
            "ordering": "one seeded Fisher-Yates shuffle per complete repetition block using xorshift64*",
            "timer_scope": "Velocity Verlet integration only, including both force evaluations per step",
            "excluded_from_timer": [
                "process startup", "scenario generation", "state copy", "warm-up integration",
                "diagnostics", "CSV formatting", "runner I/O",
            ],
            "bootstrap_resamples": arguments.bootstrap_resamples,
        },
        "outputs": {
            "raw": raw_path.name,
            "summary": summary_path.name,
            "affinity_probe": probe_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(raw_rows)} measurements to {raw_path}")
    print(f"wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
