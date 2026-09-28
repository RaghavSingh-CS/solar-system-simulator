#!/usr/bin/env python3
"""Independently verify one Stage 5 v2.0 acquisition directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from stage5_common import (
    ANGULAR_BOUND,
    DEFAULT_ORDER_SEED,
    ENERGY_BOUND,
    MODE_BODIES,
    MODE_REPETITIONS,
    MODE_ROLE,
    MODE_STEPS,
    PROTOCOL_VERSION,
    Stage5Error,
    bootstrap_interval,
    canonical_sha256,
    finite_float,
    group_seed,
    ordered_runs,
    quantile,
    read_json,
    require,
    sha256_file,
    validate_openmp_threads,
    write_json,
)


EXPECTED_FILES = {
    "raw_timings.csv",
    "summary.csv",
    "environment.json",
    "build_manifest.json",
    "ctest.txt",
}


def verify_manifest(evidence: Path, manifest: dict[str, Any]) -> dict[str, str]:
    require(set(manifest.get("files", {})) == EXPECTED_FILES, "evidence manifest file set mismatch")
    hashes: dict[str, str] = {}
    for name in sorted(EXPECTED_FILES):
        path = evidence / name
        require(path.is_file(), f"missing evidence file: {name}")
        actual = sha256_file(path)
        expected = manifest["files"][name]
        require(actual == expected["sha256"], f"SHA-256 mismatch: {name}")
        require(path.stat().st_size == int(expected["bytes"]), f"size mismatch: {name}")
        hashes[name] = actual
    extras = {
        path.name
        for path in evidence.iterdir()
        if path.is_file() and path.name not in EXPECTED_FILES | {"manifest.json"}
    }
    require(not extras, f"unexpected files in completed evidence: {sorted(extras)}")
    return hashes


def summary_rows(
    rows: list[dict[str, str]], mode: str, order_seed: int, bootstrap_resamples: int
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset_role"],
            row["backend"],
            int(row["requested_threads"]),
            int(row["bodies"]),
            row["timing_metric"],
        )
        grouped[key].append(finite_float(row["elapsed_ms"], "elapsed_ms"))
    result: list[dict[str, str]] = []
    for key in sorted(grouped):
        role, backend, threads, bodies, metric = key
        values = grouped[key]
        mean = statistics.mean(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        low, high = bootstrap_interval(values, bootstrap_resamples, group_seed(order_seed, key))
        result.append(
            {
                "mode": mode,
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
            }
        )
    return result


def verify_summary(actual: list[dict[str, str]], expected: list[dict[str, str]]) -> None:
    require(len(actual) == len(expected), "summary row count mismatch")
    keys = ("mode", "dataset_role", "backend", "requested_threads", "bodies", "timing_metric", "samples")
    numeric = (
        "median_ms",
        "mean_ms",
        "stdev_ms",
        "coefficient_of_variation",
        "min_ms",
        "max_ms",
        "iqr_ms",
        "median_ci95_low_ms",
        "median_ci95_high_ms",
    )
    for index, (candidate, reference) in enumerate(zip(actual, expected, strict=True), start=1):
        for key in keys:
            require(candidate[key] == reference[key], f"summary {key} mismatch at row {index}")
        for key in numeric:
            left = finite_float(candidate[key], f"summary {key}")
            right = finite_float(reference[key], f"recomputed {key}")
            require(math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), f"summary {key} mismatch at row {index}")


def verify_evidence(evidence_dir: Path) -> dict[str, Any]:
    evidence = evidence_dir.resolve()
    require(evidence.is_dir(), f"evidence directory not found: {evidence}")
    manifest_path = evidence / "manifest.json"
    require(manifest_path.is_file(), "manifest.json missing; partial evidence is not complete")
    manifest = read_json(manifest_path)
    hashes = verify_manifest(evidence, manifest)
    environment = read_json(evidence / "environment.json")
    build_manifest = read_json(evidence / "build_manifest.json")

    require(manifest.get("protocol_version") == PROTOCOL_VERSION, "manifest protocol mismatch")
    mode = str(manifest.get("mode", ""))
    require(mode in MODE_BODIES, f"invalid acquisition mode: {mode}")
    require(environment.get("mode") == mode, "environment mode mismatch")
    require(environment.get("protocol_version") == PROTOCOL_VERSION, "environment protocol mismatch")
    require(build_manifest.get("protocol_version") == PROTOCOL_VERSION, "build protocol mismatch")
    require(canonical_sha256(build_manifest.get("environment")) == build_manifest.get("environment_sha256"), "build environment hash mismatch")
    require(not build_manifest.get("fixture_mode"), "fixture build cannot be controlled evidence")
    require(manifest.get("source_commit") == build_manifest.get("source_commit"), "source mismatch")
    require(environment.get("source_commit") == manifest.get("source_commit"), "environment source mismatch")
    require(environment.get("environment_id") == build_manifest.get("environment_id"), "environment identity mismatch")
    require(environment.get("environment_sha256") == build_manifest.get("environment_sha256"), "environment hash mismatch")
    require(manifest.get("build_manifest_sha256") == hashes["build_manifest.json"], "build-manifest binding mismatch")
    require(build_manifest.get("ctest", {}).get("passed") is True, "build manifest lacks passing CTest")
    require(build_manifest.get("ctest", {}).get("return_code") == 0, "build CTest return code is not zero")
    require(build_manifest.get("ctest", {}).get("log_sha256") == hashes["ctest.txt"], "CTest log binding mismatch")
    require(build_manifest.get("files", {}).get("ctest.txt", {}).get("sha256") == hashes["ctest.txt"], "build artifact CTest hash mismatch")

    topology = build_manifest.get("openmp_topology", {})
    physical = int(topology.get("physical_cores", 0))
    logical = int(topology.get("logical_processors", 0))
    threads = tuple(int(value) for value in topology.get("threads", []))
    validate_openmp_threads(threads, physical, logical)
    require(environment.get("openmp_threads") == list(threads), "evidence topology mismatch")

    with (evidence / "raw_timings.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (evidence / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))

    order_seed = int(environment["order_seed"])
    bootstrap_resamples = int(environment["bootstrap_resamples"])
    require(order_seed == DEFAULT_ORDER_SEED, "order seed mismatch")
    schedule = ordered_runs(mode, threads, MODE_REPETITIONS[mode], order_seed)
    expected_row_count = sum(3 if item["backend"] == "cuda" else 2 for item in schedule)
    require(len(rows) == expected_row_count, f"raw row count mismatch: expected {expected_row_count}, got {len(rows)}")

    by_order: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_order[int(row["order_index"])].append(row)
    require(set(by_order) == set(range(1, len(schedule) + 1)), "missing or duplicate order indices")

    checksums: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    cpu_checksums: dict[int, set[str]] = defaultdict(set)
    for order_index, expected in enumerate(schedule, start=1):
        selected = by_order[order_index]
        expected_metrics = (
            {"device_compute", "transfer_inclusive", "selector_steady_state"}
            if expected["backend"] == "cuda"
            else {"integration_compute", "selector_steady_state"}
        )
        require({row["timing_metric"] for row in selected} == expected_metrics, f"metric set mismatch at order {order_index}")
        first = selected[0]
        require(first["protocol_version"] == PROTOCOL_VERSION, "raw protocol mismatch")
        require(first["mode"] == mode, "raw mode mismatch")
        require(first["dataset_role"] == MODE_ROLE[mode], "mixed or wrong dataset role")
        require(int(first["bodies"]) in MODE_BODIES[mode], "N outside phase custody")
        require(int(first["bodies"]) == int(expected["bodies"]), f"N/order mismatch at {order_index}")
        require(first["backend"] == expected["backend"], f"backend/order mismatch at {order_index}")
        require(int(first["requested_threads"]) == int(expected["threads"]), f"thread/order mismatch at {order_index}")
        require(int(first["block_id"]) == int(expected["block_id"]), f"block/order mismatch at {order_index}")
        require(int(first["steps"]) == MODE_STEPS[mode], "step count mismatch")
        require(finite_float(first["dt_s"], "dt_s") == 10.0, "dt mismatch")
        require(int(first["benchmark_seed"]) == 24301, "benchmark seed mismatch")
        require(int(first["warmups"]) == 1, "warm-up mismatch")
        require(first["correctness_status"] == "pass", "failed correctness row")
        energy = finite_float(first["relative_energy_drift"], "energy drift")
        angular = finite_float(first["relative_angular_momentum_drift"], "angular drift")
        require(energy <= ENERGY_BOUND and angular <= ANGULAR_BOUND, "scientific gate failed")
        require(first["source_commit"] == manifest["source_commit"], "raw source mismatch")
        require(first["build_manifest_sha256"] == manifest["build_manifest_sha256"], "raw build mismatch")
        require(first["environment_sha256"] == environment["environment_sha256"], "raw environment mismatch")
        for row in selected:
            require(row["run_id"] == first["run_id"], "metrics from different invocations mixed")
            require(finite_float(row["elapsed_ms"], "elapsed_ms") > 0.0, "timing must be positive")
            require(row["checksum"] == first["checksum"], "checksum differs across timing rows")
        if expected["backend"] == "openmp":
            require(int(first["effective_threads"]) == int(expected["threads"]), "effective OpenMP team mismatch")
        if expected["backend"] == "cuda":
            metric_values = {row["timing_metric"]: float(row["elapsed_ms"]) for row in selected}
            require(metric_values["transfer_inclusive"] >= metric_values["device_compute"], "CUDA timing relationship violation")
        key = (first["backend"], int(first["requested_threads"]), int(first["bodies"]))
        checksums[key].add(first["checksum"])
        if first["backend"] != "cuda":
            cpu_checksums[int(first["bodies"])].add(first["checksum"])

    require(all(len(values) == 1 for values in checksums.values()), "repeat checksum instability")
    require(all(len(values) == 1 for values in cpu_checksums.values()), "Sequential/OpenMP checksum mismatch")
    recomputed = summary_rows(rows, mode, order_seed, bootstrap_resamples)
    verify_summary(summaries, recomputed)

    report = {
        "verification_status": "pass",
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "source_commit": manifest["source_commit"],
        "build_manifest_sha256": manifest["build_manifest_sha256"],
        "environment_id": environment["environment_id"],
        "environment_sha256": environment["environment_sha256"],
        "openmp_threads": list(threads),
        "manifest_sha256": sha256_file(manifest_path),
        "raw_rows": len(rows),
        "summary_rows": len(summaries),
        "measured_bodies": list(MODE_BODIES[mode]),
        "artifact_hashes": hashes,
        "all_correctness_statuses_pass": True,
        "all_scientific_gates_pass": True,
        "schedule_exact": True,
        "checksums_stable": True,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_evidence(args.evidence_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        write_json(args.report.resolve(), report)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Stage5Error, ValueError, FileNotFoundError, KeyError, csv.Error) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
