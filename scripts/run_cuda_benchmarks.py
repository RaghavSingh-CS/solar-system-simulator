#!/usr/bin/env python3
"""Run and summarize the controlled Stage 4 CUDA benchmark matrix."""

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


DEFAULT_BODIES = (64, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
RAW_PREFIX = (
    "experiment_id", "environment_id", "source_commit", "provenance_mode",
    "source_archive_sha256", "order_seed", "block_id", "order_index",
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
        limit = (1 << 64) - ((1 << 64) % upper_bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper_bound

    def shuffle(self, values: list[int]) -> None:
        for index in range(len(values) - 1, 0, -1):
            replacement = self.randbelow(index + 1)
            values[index], values[replacement] = values[replacement], values[index]


def integer_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 2 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated body counts >= 2")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-cuda/nbody_cuda_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/stage4_cuda/run"))
    parser.add_argument("--bodies", type=integer_list, default=DEFAULT_BODIES)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--benchmark-seed", type=int, default=24301)
    parser.add_argument("--order-seed", type=int, default=20260923)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--experiment-id", default="stage4-cuda")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--provenance-mode", required=True,
        choices=("git-verified", "archive-declared"),
    )
    parser.add_argument("--source-archive-sha256")
    return parser.parse_args()


def command_output(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(["git", *arguments], text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def benchmark_command(executable: Path, bodies: int, args: argparse.Namespace) -> list[str]:
    return [
        str(executable), "--bodies", str(bodies), "--steps", str(args.steps),
        "--dt", str(args.dt), "--warmups", str(args.warmups), "--repetitions", "1",
        "--seed", str(args.benchmark_seed), "--device", str(args.device),
    ]


def parse_row(completed: subprocess.CompletedProcess[str], command: list[str]) -> dict[str, str]:
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one CUDA benchmark row from {' '.join(command)}; stderr={completed.stderr}"
        )
    return rows[0]


def quantile(sorted_values: list[float], probability: float) -> float:
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def write_summary(rows: list[dict[str, str]], path: Path) -> None:
    regimes = {
        "device_compute": "device_compute_ms",
        "transfer_inclusive": "transfer_inclusive_ms",
    }
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        for regime, field in regimes.items():
            grouped[(int(row["bodies"]), regime)].append(float(row[field]))

    medians = {key: statistics.median(values) for key, values in grouped.items()}
    fields = [
        "bodies", "timing_regime", "samples", "median_ms", "mean_ms", "stdev_ms",
        "min_ms", "max_ms", "iqr_ms", "coefficient_of_variation",
        "transfer_to_compute_ratio", "transfer_penalty_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            bodies, regime = key
            values = sorted(grouped[key])
            mean = statistics.mean(values)
            compute = medians[(bodies, "device_compute")]
            inclusive = medians[(bodies, "transfer_inclusive")]
            ratio = inclusive / compute
            writer.writerow({
                "bodies": bodies,
                "timing_regime": regime,
                "samples": len(values),
                "median_ms": f"{statistics.median(values):.17g}",
                "mean_ms": f"{mean:.17g}",
                "stdev_ms": f"{statistics.stdev(values):.17g}" if len(values) > 1 else "0",
                "min_ms": f"{values[0]:.17g}",
                "max_ms": f"{values[-1]:.17g}",
                "iqr_ms": f"{quantile(values, 0.75) - quantile(values, 0.25):.17g}",
                "coefficient_of_variation": f"{statistics.stdev(values) / mean:.17g}" if len(values) > 1 else "0",
                "transfer_to_compute_ratio": f"{ratio:.17g}",
                "transfer_penalty_percent": f"{(ratio - 1.0) * 100.0:.17g}",
            })


def validate_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    expected = len(args.bodies) * args.repetitions
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} raw rows, received {len(rows)}")
    for bodies in args.bodies:
        selected = [row for row in rows if int(row["bodies"]) == bodies]
        if len(selected) != args.repetitions:
            raise RuntimeError(f"body count {bodies} has {len(selected)} samples")
        checksums = {row["checksum"] for row in selected}
        if len(checksums) != 1:
            raise RuntimeError(f"CUDA repeatability checksum mismatch for body count {bodies}")


def main() -> int:
    args = arguments()
    if args.steps <= 0 or args.warmups <= 0 or args.repetitions < 2 or args.device < 0:
        raise ValueError("positive steps/warmups, repetitions >= 2, and device >= 0 are required")
    executable = args.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "raw_timings.partial.csv"
    raw_path = output_dir / "raw_timings.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "environment.json"
    failure_path = output_dir / "failure.json"

    generator = XorShift64Star(args.order_seed)
    schedule: list[tuple[int, int]] = []
    for block in range(1, args.repetitions + 1):
        body_counts = list(args.bodies)
        generator.shuffle(body_counts)
        schedule.extend((block, bodies) for bodies in body_counts)

    started = datetime.now(timezone.utc)
    rows: list[dict[str, str]] = []
    try:
        with partial.open("w", newline="", encoding="utf-8") as output:
            writer = None
            for order_index, (block, bodies) in enumerate(schedule, start=1):
                command = benchmark_command(executable, bodies, args)
                completed = subprocess.run(command, text=True, capture_output=True, check=False)
                try:
                    row = parse_row(completed, command)
                except Exception:
                    failure_path.write_text(json.dumps({
                        "block_id": block,
                        "order_index": order_index,
                        "bodies": bodies,
                        "command": command,
                        "return_code": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }, indent=2) + "\n", encoding="utf-8")
                    raise
                enriched = {
                    "experiment_id": args.experiment_id,
                    "environment_id": args.environment_id,
                    "source_commit": args.source_commit,
                    "provenance_mode": args.provenance_mode,
                    "source_archive_sha256": args.source_archive_sha256 or "",
                    "order_seed": str(args.order_seed),
                    "block_id": str(block),
                    "order_index": str(order_index),
                    **row,
                }
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=[*RAW_PREFIX, *row.keys()])
                    writer.writeheader()
                writer.writerow(enriched)
                output.flush()
                rows.append(enriched)
                if (completed.returncode != 0 or row["correctness_status"] != "pass" or
                        completed.stderr.strip()):
                    failure_path.write_text(json.dumps({
                        "block_id": block,
                        "order_index": order_index,
                        "bodies": bodies,
                        "command": command,
                        "return_code": completed.returncode,
                        "correctness_status": row["correctness_status"],
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }, indent=2) + "\n", encoding="utf-8")
                    raise RuntimeError(
                        f"invalid CUDA result retained at {failure_path}; "
                        f"status={row['correctness_status']} return_code={completed.returncode}"
                    )
                print(
                    f"[{order_index:03d}/{len(schedule)}] N={bodies} "
                    f"compute={row['device_compute_ms']}ms inclusive={row['transfer_inclusive_ms']}ms",
                    flush=True,
                )
        validate_rows(rows, args)
        partial.replace(raw_path)
        write_summary(rows, summary_path)
    except Exception:
        print(f"partial results retained at {partial}", file=sys.stderr)
        raise

    completed_at = datetime.now(timezone.utc)
    metadata = {
        "experiment_id": args.experiment_id,
        "environment_id": args.environment_id,
        "started_utc": started.isoformat(),
        "completed_utc": completed_at.isoformat(),
        "git": {
            "head": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "status": git_value("status", "--porcelain=v1"),
        },
        "source_provenance": {
            "mode": args.provenance_mode,
            "source_commit": args.source_commit,
            "source_archive_sha256": args.source_archive_sha256,
            "git_metadata_expected": args.provenance_mode == "git-verified",
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_processors": os.cpu_count(),
        },
        "tools": {
            "nvidia_smi": command_output(["nvidia-smi"]),
            "nvidia_smi_query": command_output([
                "nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ]),
            "nvcc": command_output(["nvcc", "--version"]),
            "cmake": command_output(["cmake", "--version"]),
            "compiler": command_output(["c++", "--version"]),
            "uname": command_output(["uname", "-a"]),
        },
        "protocol": {
            "bodies": list(args.bodies),
            "steps": args.steps,
            "dt_seconds": args.dt,
            "warmups_before_each_measurement": args.warmups,
            "measured_repetitions": args.repetitions,
            "benchmark_seed": args.benchmark_seed,
            "order_seed": args.order_seed,
            "device": args.device,
            "ordering": "one seeded Fisher-Yates shuffle per complete block using xorshift64*",
            "device_compute": "CUDA events around all four kernels per Velocity Verlet step",
            "transfer_inclusive": "CUDA events from H2D start through kernels, D2H, and completion",
            "allocation": "one-time context, stream, event, device, and pinned-host setup; reported separately",
            "host_staging": "structure conversion before H2D and after D2H; reported separately",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} CUDA measurements to {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
