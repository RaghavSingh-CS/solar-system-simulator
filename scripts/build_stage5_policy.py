#!/usr/bin/env python3
"""Build and freeze the deterministic calibration-only Stage 5 v2.0 policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage5_common import (
    CALIBRATION_BODIES,
    HELD_OUT_BODIES,
    POLICY_ALGORITHM_VERSION,
    PRIMARY_METRIC,
    PROTOCOL_VERSION,
    Stage5Error,
    canonical_configuration_order,
    canonical_sha256,
    configuration_id,
    read_json,
    require,
    require_absent_directory,
    sha256_file,
    write_json,
)
from verify_stage5_v2 import verify_evidence


def interpolate_loglog(points: dict[int, float], query: int) -> float:
    sizes = sorted(points)
    require(sizes[0] <= query <= sizes[-1], f"query N={query} requires extrapolation")
    if query in points:
        return points[query]
    upper_index = next(index for index, size in enumerate(sizes) if size > query)
    lower = sizes[upper_index - 1]
    upper = sizes[upper_index]
    x0, x1, x = math.log(lower), math.log(upper), math.log(query)
    y0, y1 = math.log(points[lower]), math.log(points[upper])
    fraction = (x - x0) / (x1 - x0)
    return math.exp(y0 + fraction * (y1 - y0))


def calibration_medians(calibration_dir: Path) -> dict[str, dict[int, float]]:
    with (calibration_dir / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, dict[int, float]] = {}
    for row in rows:
        if row["timing_metric"] != PRIMARY_METRIC:
            continue
        require(row["dataset_role"] == "calibration", "policy input contains non-calibration summary")
        backend = row["backend"]
        threads = int(row["requested_threads"])
        config = configuration_id(backend, threads)
        bodies = int(row["bodies"])
        median = float(row["median_ms"])
        require(math.isfinite(median) and median > 0.0, "policy input has invalid median")
        require(bodies not in result.setdefault(config, {}), f"duplicate policy median: {config} N={bodies}")
        result[config][bodies] = median
    expected = set(CALIBRATION_BODIES)
    require(result, "no selector_steady_state calibration medians")
    for config, values in result.items():
        require(set(values) == expected, f"configuration {config} lacks complete calibration grid")
    return result


def build_policy_content(calibration_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    medians = calibration_medians(calibration_dir)
    canonical_order = canonical_configuration_order(report["openmp_threads"])
    require(set(canonical_order) == set(medians), "calibration configurations differ from canonical ordering")
    selections: dict[str, Any] = {}
    for bodies in HELD_OUT_BODIES:
        predictions = {
            config: interpolate_loglog(medians[config], bodies) for config in canonical_order
        }
        selected = min(canonical_order, key=lambda config: (predictions[config], canonical_order.index(config)))
        selections[str(bodies)] = {
            "selected_configuration": selected,
            "predicted_median_ms": f"{predictions[selected]:.17g}",
            "predictions_ms": {config: f"{predictions[config]:.17g}" for config in canonical_order},
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_algorithm_version": POLICY_ALGORITHM_VERSION,
        "source_commit": report["source_commit"],
        "build_manifest_sha256": report["build_manifest_sha256"],
        "calibration_manifest_sha256": report["manifest_sha256"],
        "environment_id": report["environment_id"],
        "environment_sha256": report["environment_sha256"],
        "openmp_threads": report["openmp_threads"],
        "metric": PRIMARY_METRIC,
        "fixed_workload": {
            "scenario": "deterministic_seeded_synthetic",
            "seed": 24301,
            "precision": "binary64",
            "force_law": "direct_newtonian_o_n_squared",
            "integrator": "velocity_verlet",
            "dt_s": 10,
            "steps": 100,
            "energy_drift_bound": 5.0e-6,
            "angular_momentum_vector_drift_bound": 5.0e-10,
        },
        "canonical_configuration_order": list(canonical_order),
        "interpolation": {
            "coordinates": ["log(N)", "log(median_selector_steady_state_ms)"],
            "method": "piecewise-linear between adjacent calibration N values",
            "extrapolation": "forbidden",
            "tie_break": "first configuration in canonical_configuration_order",
        },
        "calibration_medians_ms": {
            config: {str(size): f"{values[size]:.17g}" for size in CALIBRATION_BODIES}
            for config, values in medians.items()
        },
        "held_out_queries": selections,
    }


def verify_policy_chain(
    policy_path: Path,
    freeze_path: Path,
    calibration_report: dict[str, Any],
    build_manifest: dict[str, Any],
    build_manifest_hash: str,
) -> dict[str, Any]:
    policy = read_json(policy_path)
    freeze = read_json(freeze_path)
    content = policy.get("policy_content")
    require(isinstance(content, dict), "policy content missing")
    computed_content_hash = canonical_sha256(content)
    require(policy.get("policy_sha256") == computed_content_hash, "policy canonical hash mismatch")
    require(freeze.get("policy_sha256") == computed_content_hash, "policy freeze hash mismatch")
    require(freeze.get("policy_file_sha256") == sha256_file(policy_path), "frozen policy file hash mismatch")
    required = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": calibration_report["source_commit"],
        "build_manifest_sha256": build_manifest_hash,
        "calibration_manifest_sha256": calibration_report["manifest_sha256"],
        "environment_id": calibration_report["environment_id"],
        "environment_sha256": calibration_report["environment_sha256"],
        "openmp_threads": calibration_report["openmp_threads"],
    }
    for key, expected in required.items():
        require(content.get(key) == expected, f"policy {key} mismatch")
        require(freeze.get(key) == expected, f"policy freeze {key} mismatch")
    require(build_manifest.get("source_commit") == required["source_commit"], "policy/build source mismatch")
    require(build_manifest.get("environment_sha256") == required["environment_sha256"], "policy/build environment mismatch")
    require(content.get("metric") == PRIMARY_METRIC, "policy metric mismatch")
    require(content.get("policy_algorithm_version") == POLICY_ALGORITHM_VERSION, "policy algorithm mismatch")
    require(set(content.get("held_out_queries", {})) == {str(value) for value in HELD_OUT_BODIES}, "policy held-out query set mismatch")
    return {
        "policy_sha256": computed_content_hash,
        "policy_file_sha256": sha256_file(policy_path),
        "policy_freeze_sha256": sha256_file(freeze_path),
        "calibration_manifest_sha256": calibration_report["manifest_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration = args.calibration_dir.resolve()
    report = verify_evidence(calibration)
    require(report["mode"] == "calibration", "policy input must be verified calibration evidence")
    output = require_absent_directory(args.output, "policy output directory")
    output.mkdir(parents=True)
    content = build_policy_content(calibration, report)
    policy_hash = canonical_sha256(content)
    policy = {
        "policy_content": content,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": policy_hash,
    }
    policy_path = output / "policy.json"
    write_json(policy_path, policy)
    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": report["source_commit"],
        "build_manifest_sha256": report["build_manifest_sha256"],
        "calibration_manifest_sha256": report["manifest_sha256"],
        "environment_id": report["environment_id"],
        "environment_sha256": report["environment_sha256"],
        "openmp_threads": report["openmp_threads"],
        "policy_sha256": policy_hash,
        "policy_file_sha256": sha256_file(policy_path),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    freeze_path = output / "policy_freeze.json"
    write_json(freeze_path, freeze)
    build = read_json(calibration / "build_manifest.json")
    verify_policy_chain(policy_path, freeze_path, report, build, report["build_manifest_sha256"])
    print(f"frozen calibration-only policy: {policy_hash}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Stage5Error, ValueError, FileNotFoundError, KeyError, csv.Error) as error:
        print(f"policy generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
