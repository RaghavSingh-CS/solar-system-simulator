#!/usr/bin/env python3
"""Verify and independently summarize one Stage 4 CUDA evidence directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(sorted_values: list[float], probability: float) -> float:
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def statistics_for(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    mean = statistics.mean(ordered)
    return {
        "samples": len(ordered),
        "median_ms": statistics.median(ordered),
        "mean_ms": mean,
        "stdev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "iqr_ms": quantile(ordered, 0.75) - quantile(ordered, 0.25),
        "coefficient_of_variation":
            statistics.stdev(ordered) / mean if len(ordered) > 1 else 0.0,
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_float(row: dict[str, str], field: str, *, positive: bool = False) -> float:
    value = float(row[field])
    require(math.isfinite(value), f"non-finite {field}")
    require(not positive or value > 0.0, f"non-positive {field}")
    return value


def expected_schedule(bodies: list[int], repetitions: int, seed: int) -> list[tuple[int, int]]:
    generator = XorShift64Star(seed)
    schedule: list[tuple[int, int]] = []
    for block in range(1, repetitions + 1):
        block_bodies = list(bodies)
        generator.shuffle(block_bodies)
        schedule.extend((block, body_count) for body_count in block_bodies)
    return schedule


def verify_summary(
    supplied_rows: list[dict[str, str]],
    computed: dict[tuple[int, str], dict[str, float | int]],
) -> None:
    require(len(supplied_rows) == len(computed), "summary row count differs from recomputation")
    seen: set[tuple[int, str]] = set()
    numeric_fields = (
        "median_ms", "mean_ms", "stdev_ms", "min_ms", "max_ms", "iqr_ms",
        "coefficient_of_variation", "transfer_to_compute_ratio", "transfer_penalty_percent",
    )
    for row in supplied_rows:
        key = (int(row["bodies"]), row["timing_regime"])
        require(key in computed, f"unexpected summary key {key}")
        require(key not in seen, f"duplicate summary key {key}")
        seen.add(key)
        expected = computed[key]
        require(int(row["samples"]) == expected["samples"], f"sample count differs for {key}")
        for field in numeric_fields:
            require(
                math.isclose(float(row[field]), float(expected[field]), rel_tol=1.0e-12, abs_tol=1.0e-12),
                f"summary field {field} differs for {key}",
            )


def main() -> int:
    args = arguments()
    evidence_label = args.evidence_dir.as_posix()
    evidence_dir = args.evidence_dir.resolve()
    raw_path = evidence_dir / "raw_timings.csv"
    summary_path = evidence_dir / "summary.csv"
    environment_path = evidence_dir / "environment.json"
    ctest_path = evidence_dir / "cuda_ctest.txt"
    for path in (raw_path, summary_path, environment_path, ctest_path):
        require(path.is_file(), f"missing evidence file: {path}")

    with raw_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    with summary_path.open(newline="", encoding="utf-8") as source:
        supplied_summary = list(csv.DictReader(source))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    protocol = environment["protocol"]
    bodies = [int(value) for value in protocol["bodies"]]
    repetitions = int(protocol["measured_repetitions"])
    expected_rows = len(bodies) * repetitions
    require(len(rows) == expected_rows, f"expected {expected_rows} raw rows, found {len(rows)}")

    observed_schedule: list[tuple[int, int]] = []
    unique_block_body: set[tuple[int, int]] = set()
    order_indices: list[int] = []
    timings: dict[tuple[int, str], list[float]] = defaultdict(list)
    checksums: dict[int, set[str]] = defaultdict(set)
    energy_drifts: dict[int, list[float]] = defaultdict(list)
    angular_drifts: dict[int, list[float]] = defaultdict(list)
    constant_fields = {
        "environment_id": environment["environment_id"],
        "experiment_id": environment["experiment_id"],
        "order_seed": str(protocol["order_seed"]),
        "steps": str(protocol["steps"]),
        "warmups": str(protocol["warmups_before_each_measurement"]),
        "seed": str(protocol["benchmark_seed"]),
        "device_index": str(protocol["device"]),
    }
    gpu_fields = (
        "gpu_name", "compute_capability", "total_global_memory_bytes", "multiprocessor_count",
        "cuda_driver_version", "cuda_runtime_version", "build_configuration", "host_threads",
    )
    first_gpu = {field: rows[0][field] for field in gpu_fields}

    for row in rows:
        for field, expected in constant_fields.items():
            require(row[field] == expected, f"inconsistent {field}")
        require(row["backend"] == "cuda", "non-CUDA backend row")
        require(row["correctness_status"] == "pass", "failed correctness row")
        require(row["repetition"] == "1", "unexpected executable-internal repetition count")
        require({field: row[field] for field in gpu_fields} == first_gpu, "GPU/build metadata changed")
        require(math.isclose(float(row["dt_s"]), float(protocol["dt_seconds"])), "inconsistent dt")

        body_count = int(row["bodies"])
        block = int(row["block_id"])
        order_index = int(row["order_index"])
        key = (block, body_count)
        require(body_count in bodies, f"unexpected body count {body_count}")
        require(key not in unique_block_body, f"duplicate block/body row {key}")
        unique_block_body.add(key)
        order_indices.append(order_index)
        observed_schedule.append(key)

        compute = finite_float(row, "device_compute_ms", positive=True)
        inclusive = finite_float(row, "transfer_inclusive_ms", positive=True)
        require(inclusive >= compute, "transfer-inclusive timing is less than compute timing")
        for field in (
            "host_to_device_ms", "device_to_host_ms", "host_staging_ms", "backend_wall_ms",
            "allocation_ms",
        ):
            finite_float(row, field, positive=True)
        timings[(body_count, "device_compute")].append(compute)
        timings[(body_count, "transfer_inclusive")].append(inclusive)
        checksums[body_count].add(row["checksum"])
        energy_drift = finite_float(row, "relative_energy_drift")
        angular_drift = finite_float(row, "relative_angular_momentum_drift")
        require(energy_drift >= 0.0, "negative relative energy drift")
        require(angular_drift >= 0.0, "negative relative angular-momentum drift")
        energy_drifts[body_count].append(energy_drift)
        angular_drifts[body_count].append(angular_drift)

    require(order_indices == list(range(1, expected_rows + 1)), "order_index is not contiguous")
    require(
        observed_schedule == expected_schedule(bodies, repetitions, int(protocol["order_seed"])),
        "recorded order does not match the seeded block shuffle",
    )
    require(all(len(values) == 1 for values in checksums.values()), "repeat checksum mismatch")
    ctest_text = ctest_path.read_text(encoding="utf-8")
    require("100% tests passed, 0 tests failed" in ctest_text, "CTest log does not show full success")

    computed_summary: dict[tuple[int, str], dict[str, float | int]] = {}
    by_body: list[dict[str, float | int | str]] = []
    previous_body: int | None = None
    previous_compute: float | None = None
    for body_count in sorted(bodies):
        compute_stats = statistics_for(timings[(body_count, "device_compute")])
        inclusive_stats = statistics_for(timings[(body_count, "transfer_inclusive")])
        ratio = float(inclusive_stats["median_ms"]) / float(compute_stats["median_ms"])
        for regime, stats in (
            ("device_compute", compute_stats), ("transfer_inclusive", inclusive_stats),
        ):
            computed_summary[(body_count, regime)] = {
                **stats,
                "transfer_to_compute_ratio": ratio,
                "transfer_penalty_percent": (ratio - 1.0) * 100.0,
            }
        scaling_exponent: float | None = None
        if previous_body is not None and previous_compute is not None:
            scaling_exponent = math.log(float(compute_stats["median_ms"]) / previous_compute) / math.log(
                body_count / previous_body
            )
        by_body.append({
            "bodies": body_count,
            "samples": compute_stats["samples"],
            "compute_median_ms": compute_stats["median_ms"],
            "compute_mean_ms": compute_stats["mean_ms"],
            "compute_stdev_ms": compute_stats["stdev_ms"],
            "compute_cv_percent": float(compute_stats["coefficient_of_variation"]) * 100.0,
            "compute_min_ms": compute_stats["min_ms"],
            "compute_max_ms": compute_stats["max_ms"],
            "inclusive_median_ms": inclusive_stats["median_ms"],
            "transfer_overhead_ms": float(inclusive_stats["median_ms"]) - float(compute_stats["median_ms"]),
            "transfer_overhead_percent": (ratio - 1.0) * 100.0,
            "local_scaling_exponent": scaling_exponent if scaling_exponent is not None else "",
            "max_relative_energy_drift": max(energy_drifts[body_count]),
            "max_relative_angular_momentum_drift": max(angular_drifts[body_count]),
            "checksum": next(iter(checksums[body_count])),
        })
        previous_body = body_count
        previous_compute = float(compute_stats["median_ms"])

    verify_summary(supplied_summary, computed_summary)
    report = {
        "verification_status": "pass",
        "evidence_directory": evidence_label,
        "files": {
            "raw_timings_sha256": sha256(raw_path),
            "summary_sha256": sha256(summary_path),
            "environment_sha256": sha256(environment_path),
            "cuda_ctest_sha256": sha256(ctest_path),
        },
        "experiment_id": environment["experiment_id"],
        "environment_id": environment["environment_id"],
        "rows": len(rows),
        "body_counts": sorted(bodies),
        "repetitions_per_body": repetitions,
        "schedule_matches_seeded_shuffle": True,
        "summary_matches_raw": True,
        "all_correctness_statuses_pass": True,
        "repeat_checksums_match": True,
        "gpu_and_build_fields": first_gpu,
        "maximum_relative_energy_drift": max(max(values) for values in energy_drifts.values()),
        "maximum_relative_angular_momentum_drift": max(max(values) for values in angular_drifts.values()),
        "results_by_body": by_body,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
