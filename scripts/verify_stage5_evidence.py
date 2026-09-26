#!/usr/bin/env python3
"""Independently verify one Stage 5 pilot or full evidence directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "stage5-v1.0"
CALIBRATION_BODIES = (64, 128, 192, 256, 512, 1024, 2048, 4096)
HELD_OUT_BODIES = (96, 160, 384, 768, 1536, 3072)
FULL_BODIES = tuple(sorted((*CALIBRATION_BODIES, *HELD_OUT_BODIES)))
PILOT_BODIES = (64, 512, 4096)
ENERGY_BOUND = 5.0e-6
ANGULAR_BOUND = 5.0e-10
SUMMARY_NUMERIC_FIELDS = (
    "median_ms", "mean_ms", "stdev_ms", "coefficient_of_variation",
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
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def finite(value: str, label: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite {label}: {value}")
    return parsed


def role_for(bodies: int, mode: str) -> str:
    if mode == "pilot":
        return "pilot_only"
    return "calibration" if bodies in CALIBRATION_BODIES else "held_out"


def base_configurations(bodies: tuple[int, ...], threads: tuple[int, ...], mode: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for body_count in bodies:
        role = role_for(body_count, mode)
        configs.append({"backend": "sequential", "threads": 1, "bodies": body_count, "role": role})
        for thread_count in threads:
            configs.append({"backend": "openmp", "threads": thread_count, "bodies": body_count, "role": role})
        configs.append({"backend": "cuda", "threads": 0, "bodies": body_count, "role": role})
    return configs


def expected_schedule(
    bodies: tuple[int, ...], threads: tuple[int, ...], mode: str, repetitions: int, seed: int
) -> list[dict[str, Any]]:
    generator = XorShift64Star(seed)
    schedule: list[dict[str, Any]] = []
    configs = base_configurations(bodies, threads, mode)
    for block in range(1, repetitions + 1):
        values = [dict(config, block_id=block) for config in configs]
        generator.shuffle(values)
        schedule.extend(values)
    return schedule


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


def recompute_summary(
    rows: list[dict[str, str]], order_seed: int, resamples: int, mode: str
) -> dict[tuple[str, str, int, int, str], dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset_role"], row["backend"], int(row["requested_threads"]),
            int(row["bodies"]), row["timing_metric"],
        )
        grouped[key].append(float(row["elapsed_ms"]))
    result: dict[tuple[str, str, int, int, str], dict[str, float | int | str]] = {}
    for key, values in grouped.items():
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        mean = statistics.mean(values)
        low, high = bootstrap_interval(values, resamples, group_seed(order_seed, key))
        result[key] = {
            "mode": mode,
            "samples": len(values),
            "median_ms": statistics.median(values),
            "mean_ms": mean,
            "stdev_ms": deviation,
            "coefficient_of_variation": deviation / mean,
            "min_ms": min(values),
            "max_ms": max(values),
            "iqr_ms": quantile(values, 0.75) - quantile(values, 0.25),
            "median_ci95_low_ms": low,
            "median_ci95_high_ms": high,
        }
    return result


def verify_summary(
    rows: list[dict[str, str]], supplied: list[dict[str, str]], environment: dict[str, Any]
) -> None:
    protocol = environment["protocol"]
    expected = recompute_summary(
        rows, int(protocol["order_seed"]), int(protocol["bootstrap_resamples"]), environment["mode"]
    )
    require(len(supplied) == len(expected), "summary group count mismatch")
    seen: set[tuple[str, str, int, int, str]] = set()
    for row in supplied:
        key = (
            row["dataset_role"], row["backend"], int(row["requested_threads"]),
            int(row["bodies"]), row["timing_metric"],
        )
        require(key in expected and key not in seen, f"unexpected or duplicate summary key: {key}")
        seen.add(key)
        reference = expected[key]
        require(row["mode"] == reference["mode"], f"summary mode mismatch for {key}")
        require(int(row["samples"]) == reference["samples"], f"summary sample mismatch for {key}")
        for field in SUMMARY_NUMERIC_FIELDS:
            actual = finite(row[field], f"summary {field}")
            target = float(reference[field])
            require(math.isclose(actual, target, rel_tol=2e-15, abs_tol=1e-15), f"summary {field} mismatch for {key}")


def verify_manifest(evidence: Path, manifest: dict[str, Any]) -> dict[str, str]:
    expected_names = {"raw_timings.csv", "summary.csv", "environment.json", "ctest.txt"}
    require(set(manifest["files"]) == expected_names, "manifest file set mismatch")
    hashes: dict[str, str] = {}
    for name in expected_names:
        path = evidence / name
        require(path.is_file(), f"missing artifact: {name}")
        actual = sha256(path)
        require(actual == manifest["files"][name]["sha256"], f"SHA-256 mismatch: {name}")
        require(path.stat().st_size == manifest["files"][name]["bytes"], f"size mismatch: {name}")
        hashes[name] = actual
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = args.evidence_dir.resolve()
    require(evidence.is_dir(), f"not an evidence directory: {evidence}")
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    hashes = verify_manifest(evidence, manifest)
    environment = json.loads((evidence / "environment.json").read_text(encoding="utf-8"))
    require(environment["protocol_version"] == PROTOCOL_VERSION, "environment protocol mismatch")
    require(manifest["protocol_version"] == PROTOCOL_VERSION, "manifest protocol mismatch")
    require(environment["source_provenance"]["source_commit"] == manifest["source_commit"], "source commit mismatch")
    require(environment["source_provenance"]["git_head"] == manifest["source_commit"], "Git HEAD mismatch")
    require(environment["source_provenance"]["mode"] == "git-verified", "provenance is not Git-verified")
    require(environment["source_provenance"]["status_before_output"] == "", "source tree was dirty")
    require(environment["platform"]["system"] == "Windows", "Stage 5 evidence is not Windows")
    require(environment["build"]["cmake_cache"].get("NBODY_ENABLE_OPENMP") == "ON", "OpenMP build flag missing")
    require(environment["build"]["cmake_cache"].get("NBODY_ENABLE_CUDA") == "ON", "CUDA build flag missing")
    ctest = (evidence / "ctest.txt").read_text(encoding="utf-8", errors="replace")
    require("100% tests passed" in ctest and "0 tests failed" in ctest, "CTest evidence failed")

    with (evidence / "raw_timings.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (evidence / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    require(rows, "raw evidence is empty")
    protocol = environment["protocol"]
    mode = environment["mode"]
    require(mode in ("pilot", "full"), f"invalid mode: {mode}")
    bodies = PILOT_BODIES if mode == "pilot" else FULL_BODIES
    repetitions = 3 if mode == "pilot" else 11
    steps = 10 if mode == "pilot" else 100
    require(tuple(protocol["measured_bodies"]) == bodies, "measured N matrix mismatch")
    require(tuple(protocol["calibration_bodies"]) == CALIBRATION_BODIES, "calibration split mismatch")
    require(tuple(protocol["held_out_bodies"]) == HELD_OUT_BODIES, "held-out split mismatch")
    require(int(protocol["repetitions"]) == repetitions, "repetition count mismatch")
    require(int(protocol["steps"]) == steps and float(protocol["dt_s"]) == 10.0, "workload mismatch")
    threads = tuple(environment["openmp_environment"]["requested_threads"])
    require(threads == tuple(sorted(set(threads))) and 1 in threads, "invalid OpenMP thread list")
    schedule = expected_schedule(bodies, threads, mode, repetitions, int(protocol["order_seed"]))

    by_order: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_order[int(row["order_index"])].append(row)
    require(set(by_order) == set(range(1, len(schedule) + 1)), "missing/duplicate schedule order indices")
    checksum_groups: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    cpu_checksums: dict[int, set[str]] = defaultdict(set)
    openmp_specification_dates: set[int] = set()
    max_energy = 0.0
    max_angular = 0.0
    for order_index, expected in enumerate(schedule, start=1):
        selected = by_order[order_index]
        expected_metrics = (
            {"device_compute", "transfer_inclusive", "selector_steady_state"}
            if expected["backend"] == "cuda"
            else {"integration_compute", "selector_steady_state"}
        )
        require({row["timing_metric"] for row in selected} == expected_metrics, f"metric set mismatch at order {order_index}")
        require(len(selected) == len(expected_metrics), f"duplicate metric at order {order_index}")
        first = selected[0]
        requested = int(first["requested_threads"])
        expected_threads = int(expected["threads"])
        for row in selected:
            require(row["protocol_version"] == PROTOCOL_VERSION, "raw protocol mismatch")
            require(row["source_commit"] == manifest["source_commit"], "raw source mismatch")
            require(row["environment_id"] == environment["environment_id"], "raw environment mismatch")
            require(row["mode"] == mode and row["dataset_role"] == expected["role"], "raw role mismatch")
            require(row["backend"] == expected["backend"], f"backend schedule mismatch at order {order_index}")
            require(int(row["bodies"]) == expected["bodies"], f"N schedule mismatch at order {order_index}")
            require(int(row["block_id"]) == expected["block_id"], f"block mismatch at order {order_index}")
            require(int(row["repetition"]) == expected["block_id"], "repetition/block mismatch")
            require(int(row["steps"]) == steps and float(row["dt_s"]) == 10.0, "raw workload mismatch")
            require(row["correctness_status"] == "pass", "failed correctness row")
            elapsed = finite(row["elapsed_ms"], "elapsed_ms")
            require(elapsed > 0.0, "non-positive timing")
            energy = finite(row["relative_energy_drift"], "relative_energy_drift")
            angular = finite(row["relative_angular_momentum_drift"], "relative_angular_momentum_drift")
            require(energy <= ENERGY_BOUND and angular <= ANGULAR_BOUND, "conservation gate failure")
            require(float(row["energy_drift_bound"]) == ENERGY_BOUND, "energy bound changed")
            require(float(row["angular_momentum_drift_bound"]) == ANGULAR_BOUND, "angular bound changed")
            require(row["build_configuration"].casefold() == "release", "non-Release result")
            max_energy = max(max_energy, energy)
            max_angular = max(max_angular, angular)
        require(requested == expected_threads, f"requested thread mismatch at order {order_index}")
        require(int(first["effective_threads"]) == expected_threads, f"effective thread mismatch at order {order_index}")
        if expected["backend"] == "cuda":
            require("rtx 4050" in first["gpu_name"].casefold(), "unexpected GPU identity")
            metric_values = {row["timing_metric"]: float(row["elapsed_ms"]) for row in selected}
            require(metric_values["transfer_inclusive"] >= metric_values["device_compute"], "invalid CUDA timing ordering")
        elif expected["backend"] == "openmp":
            specification_date = int(first["openmp_specification_date"])
            require(specification_date > 0, "missing OpenMP specification date")
            openmp_specification_dates.add(specification_date)
            metric_values = {row["timing_metric"]: float(row["elapsed_ms"]) for row in selected}
            require(metric_values["selector_steady_state"] >= metric_values["integration_compute"], "invalid OpenMP timing ordering")
        else:
            metric_values = {row["timing_metric"]: float(row["elapsed_ms"]) for row in selected}
            require(metric_values["selector_steady_state"] >= metric_values["integration_compute"], "invalid sequential timing ordering")
        key = (expected["backend"], expected_threads, int(expected["bodies"]))
        checksum_groups[key].add(first["checksum"])
        if expected["backend"] != "cuda":
            cpu_checksums[int(expected["bodies"])].add(first["checksum"])
    require(all(len(values) == 1 for values in checksum_groups.values()), "repeat checksum mismatch")
    require(all(len(values) == 1 for values in cpu_checksums.values()), "sequential/OpenMP checksum mismatch")
    require(len(openmp_specification_dates) == 1, "inconsistent OpenMP specification date")
    require(
        environment["openmp_environment"]["openmp_specification_date"] == next(iter(openmp_specification_dates)),
        "environment OpenMP specification mismatch",
    )
    observed_gpu = environment["observed_cuda_device"]
    require("rtx 4050" in observed_gpu["name"].casefold(), "environment GPU identity mismatch")
    require(int(observed_gpu["multiprocessor_count"]) > 0, "missing GPU multiprocessor count")
    require(int(observed_gpu["total_global_memory_bytes"]) > 0, "missing GPU memory")
    require(int(protocol["raw_rows"]) == len(rows), "environment raw-row count mismatch")
    expected_rows = sum(3 if item["backend"] == "cuda" else 2 for item in schedule)
    require(len(rows) == expected_rows, "raw-row count mismatch")
    verify_summary(rows, summary, environment)
    if mode == "full":
        gate = environment["full_run_pilot_gate"]
        require(gate and gate["approval"]["approved"] is True, "full run lacks pilot approval")

    report = {
        "verification_status": "pass",
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "source_commit": manifest["source_commit"],
        "environment_id": environment["environment_id"],
        "experiment_id": environment["experiment_id"],
        "openmp_threads": list(threads),
        "body_counts": list(bodies),
        "calibration_bodies": list(CALIBRATION_BODIES),
        "held_out_bodies": list(HELD_OUT_BODIES),
        "repetitions": repetitions,
        "underlying_runs": len(schedule),
        "raw_rows": len(rows),
        "schedule_matches_seeded_shuffle": True,
        "summary_matches_raw": True,
        "artifact_hashes_match_manifest": True,
        "artifact_sha256": hashes,
        "all_correctness_statuses_pass": True,
        "all_conservation_gates_pass": True,
        "requested_effective_threads_match": True,
        "openmp_specification_date": next(iter(openmp_specification_dates)),
        "observed_cuda_device": observed_gpu,
        "repeat_checksums_match": True,
        "cpu_cross_backend_checksums_match": True,
        "maximum_relative_energy_drift": max_energy,
        "maximum_relative_angular_momentum_drift": max_angular,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
