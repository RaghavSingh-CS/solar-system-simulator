#!/usr/bin/env python3
"""Run the prospectively bounded Stage 6D direct-N-body scalability pilot."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "stage6d-large-n-v1.0"
DEFAULT_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                 16384, 32768, 65536, 100000)
ENERGY_BOUND = 5.0e-6
ANGULAR_BOUND = 5.0e-10
RAW_FIELDS = (
    "protocol_version", "run_id", "backend", "ranks", "threads_per_rank",
    "bodies", "steps", "dt_s", "warmups", "repetitions", "runtime_budget_s",
    "status", "elapsed_ms", "per_step_ms", "local_compute_ms",
    "communication_ms", "correctness_status", "relative_energy_drift",
    "relative_angular_momentum_drift", "checksum", "return_code", "log_file",
)


def positive_int_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not parsed or any(item <= 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("expected unique comma-separated positive integers")
    return parsed


def backend_list(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    allowed = {"sequential", "openmp", "mpi", "mpi+openmp"}
    if not parsed or len(set(parsed)) != len(parsed) or any(item not in allowed for item in parsed):
        raise argparse.ArgumentTypeError(
            "backends must be unique values from sequential,openmp,mpi,mpi+openmp"
        )
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sizes", type=positive_int_list, default=DEFAULT_SIZES)
    parser.add_argument("--backends", type=backend_list,
                        default=("sequential", "openmp", "mpi"))
    parser.add_argument("--runtime-budget-s", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=24301)
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument("--mpi-ranks", type=int, default=2)
    parser.add_argument("--cpu-executable", type=Path,
                        default=Path("build-stage6d-openmp/nbody_benchmark.exe"))
    parser.add_argument("--mpi-executable", type=Path,
                        default=Path("build-stage6d-mpi/nbody_mpi_benchmark.exe"))
    parser.add_argument("--mpiexec", default="mpiexec")
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    if args.runtime_budget_s <= 0 or args.steps <= 0 or args.dt <= 0 or \
            args.warmups < 0 or args.repetitions <= 0 or args.omp_threads <= 0 or \
            args.mpi_ranks <= 0:
        parser.error("runtime budget, steps, dt, repetitions, threads, and ranks must be positive")
    if "mpi+openmp" in args.backends and os.cpu_count() and \
            args.mpi_ranks * args.omp_threads > int(os.cpu_count()):
        parser.error("MPI ranks x OpenMP threads exceeds detected logical CPU concurrency")
    return args


def configurations(args: argparse.Namespace) -> list[dict[str, int | str]]:
    result: list[dict[str, int | str]] = []
    for bodies in args.sizes:
        for backend in args.backends:
            result.append({
                "backend": backend,
                "bodies": bodies,
                "ranks": args.mpi_ranks if backend.startswith("mpi") else 1,
                "threads": args.omp_threads if backend in ("openmp", "mpi+openmp") else 1,
            })
    return result


def command_for(args: argparse.Namespace, config: dict[str, int | str]) -> list[str]:
    common = [
        "--bodies", str(config["bodies"]), "--steps", str(args.steps),
        "--dt", str(args.dt), "--warmups", str(args.warmups),
        "--repetitions", str(args.repetitions), "--seed", str(args.seed),
    ]
    backend = str(config["backend"])
    if backend.startswith("mpi"):
        return [
            args.mpiexec, "-n", str(config["ranks"]),
            str(args.mpi_executable.resolve()), *common,
            "--threads", str(config["threads"]),
        ]
    return [
        str(args.cpu_executable.resolve()), "--backend", backend,
        *common, "--threads", str(config["threads"]),
    ]


def parse_single_csv(stdout: str) -> dict[str, str]:
    rows = list(csv.DictReader(io.StringIO(stdout)))
    if len(rows) != 1:
        raise ValueError(f"expected one CSV result row, received {len(rows)}")
    return rows[0]


def total_memory_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
        except (AttributeError, OSError):
            return None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def physical_core_count() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]
        return psutil.cpu_count(logical=False)
    except (ImportError, AttributeError):
        return None


def command_record(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
        return {"command": command, "return_code": completed.returncode,
                "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def git_record() -> dict[str, Any]:
    def value(*arguments: str) -> str | None:
        completed = subprocess.run(["git", *arguments], text=True, capture_output=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None
    return {
        "source_revision": value("rev-parse", "HEAD"),
        "branch": value("branch", "--show-current"),
        "status": value("status", "--porcelain"),
    }


def environment_record(args: argparse.Namespace, schedule: list[dict[str, int | str]]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "Stage 6D bounded large-N pilot; never Stage 3/4/5 evidence",
        "same_machine_claim_rule": "Only rows from this environment may form direct backend ratios.",
        "runtime_budget_rule": {
            "seconds_per_configuration_process": args.runtime_budget_s,
            "fixed_before_execution": True,
            "timeout_classification": "infeasible_within_budget",
        },
        "workload": {
            "algorithm": "direct Newtonian O(N^2)", "precision": "binary64",
            "integrator": "Velocity Verlet", "sizes": list(args.sizes),
            "steps": args.steps, "dt_s": args.dt, "warmups": args.warmups,
            "repetitions": args.repetitions, "seed": args.seed,
        },
        "topology": {
            "backends": list(args.backends), "openmp_threads": args.omp_threads,
            "mpi_ranks": args.mpi_ranks,
            "schedule": schedule,
        },
        "machine": {
            "identifier": platform.node(), "os": platform.platform(),
            "processor": platform.processor(), "architecture": platform.machine(),
            "physical_cores": physical_core_count(), "logical_cores": os.cpu_count(),
            "ram_bytes": total_memory_bytes(),
        },
        "software": {
            "python": sys.version, "mpi": command_record([args.mpiexec, "-help"]),
            "compiler": command_record(["c++", "--version"]),
            "cmake": command_record(["cmake", "--version"]),
            "ninja": command_record(["ninja", "--version"]),
            "openmp_environment": {
                name: os.environ.get(name, "unset")
                for name in ("OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY")
            },
            "gpu": command_record(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
                if shutil.which("nvidia-smi") else {"available": False},
            "cuda_toolkit": command_record(["nvcc", "--version"])
                if shutil.which("nvcc") else {"available": False},
            "executables": {
                "cpu": str(args.cpu_executable.resolve()),
                "mpi": str(args.mpi_executable.resolve()),
            },
        },
        "source": git_record(),
    }


def result_row(
    args: argparse.Namespace,
    config: dict[str, int | str],
    index: int,
    status: str,
    return_code: int | str,
    parsed: dict[str, str] | None,
    log_file: str,
) -> dict[str, str]:
    parsed = parsed or {}
    mpi = str(config["backend"]).startswith("mpi")
    elapsed_s = parsed.get("median_s", "")
    per_step_s = parsed.get("median_per_step_s", "")
    if elapsed_s and not per_step_s:
        per_step_s = str(float(elapsed_s) / float(args.steps))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": f"c{index:04d}",
        "backend": str(config["backend"]),
        "ranks": str(config["ranks"]),
        "threads_per_rank": str(config["threads"]),
        "bodies": str(config["bodies"]),
        "steps": str(args.steps), "dt_s": str(args.dt),
        "warmups": str(args.warmups), "repetitions": str(args.repetitions),
        "runtime_budget_s": str(args.runtime_budget_s), "status": status,
        "elapsed_ms": str(float(elapsed_s) * 1000.0) if elapsed_s else "",
        "per_step_ms": str(float(per_step_s) * 1000.0) if per_step_s else "",
        "local_compute_ms": str(float(parsed.get("median_local_compute_s", "0")) * 1000.0)
            if mpi and parsed.get("median_local_compute_s") else "",
        "communication_ms": str(float(parsed.get("median_communication_s", "0")) * 1000.0)
            if mpi and parsed.get("median_communication_s") else "",
        "correctness_status": parsed.get("correctness_status", "pass" if status == "completed" else ""),
        "relative_energy_drift": parsed.get("relative_energy_drift", ""),
        "relative_angular_momentum_drift": parsed.get("relative_angular_momentum_drift", ""),
        "checksum": parsed.get("checksum", ""),
        "return_code": str(return_code), "log_file": log_file,
    }


def main() -> int:
    args = parse_arguments()
    schedule = configurations(args)
    if args.dry_run:
        print(json.dumps({"protocol_version": PROTOCOL_VERSION, "schedule": schedule}, indent=2))
        return 0

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    environment = environment_record(args, schedule)
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, str]] = []
    for index, config in enumerate(schedule, start=1):
        command = command_for(args, config)
        log_name = f"c{index:04d}_{config['backend']}_n{config['bodies']}.txt"
        try:
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False,
                timeout=args.runtime_budget_s,
            )
            log_text = (
                "command: " + json.dumps(command) + "\n"
                "stdout:\n" + completed.stdout + "\nstderr:\n" + completed.stderr
            )
            (logs / log_name).write_text(log_text, encoding="utf-8")
            parsed = parse_single_csv(completed.stdout) if completed.returncode == 0 else None
            status = "completed" if completed.returncode == 0 else "error"
            if parsed and parsed.get("correctness_status", "pass") != "pass":
                status = "correctness_failure"
            rows.append(result_row(
                args, config, index, status, completed.returncode, parsed,
                f"logs/{log_name}"))
        except subprocess.TimeoutExpired as error:
            text = "command: " + json.dumps(command) + "\nstatus: timed out\n"
            if error.stdout:
                text += "stdout:\n" + str(error.stdout)
            if error.stderr:
                text += "\nstderr:\n" + str(error.stderr)
            (logs / log_name).write_text(text, encoding="utf-8")
            rows.append(result_row(
                args, config, index, "infeasible_within_budget", "timeout", None,
                f"logs/{log_name}"))
        except (OSError, ValueError) as error:
            (logs / log_name).write_text(
                "command: " + json.dumps(command) + "\nerror: " + str(error) + "\n",
                encoding="utf-8")
            rows.append(result_row(
                args, config, index, "error", "launch_or_parse_error", None,
                f"logs/{log_name}"))

        with (output / "raw_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    completed = sum(row["status"] == "completed" for row in rows)
    print(f"Stage 6D large-N pilot complete: {completed}/{len(rows)} configurations completed; "
          f"all statuses retained in {output / 'raw_results.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
