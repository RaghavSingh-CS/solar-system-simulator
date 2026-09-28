#!/usr/bin/env python3
"""Evaluate a frozen Stage 5 v2.0 policy on verified held-out evidence."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

from stage5_common import (
    HELD_OUT_BODIES,
    PRIMARY_METRIC,
    PROTOCOL_VERSION,
    Stage5Error,
    canonical_sha256,
    configuration_id,
    read_json,
    require,
    sha256_file,
    write_json,
)
from verify_stage5_v2 import verify_evidence


def load_held_out_medians(evidence: Path) -> dict[int, dict[str, float]]:
    with (evidence / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["timing_metric"] != PRIMARY_METRIC:
            continue
        require(row["dataset_role"] == "held_out", "evaluation input contains non-held-out summary")
        bodies = int(row["bodies"])
        config = configuration_id(row["backend"], int(row["requested_threads"]))
        median = float(row["median_ms"])
        require(math.isfinite(median) and median > 0.0, "evaluation input has invalid median")
        require(config not in result.setdefault(bodies, {}), f"duplicate held-out median: N={bodies} {config}")
        result[bodies][config] = median
    require(set(result) == set(HELD_OUT_BODIES), "held-out median N set mismatch")
    return result


def validate_frozen_policy(
    policy_path: Path, freeze_path: Path, held_out_report: dict[str, Any]
) -> dict[str, Any]:
    policy = read_json(policy_path)
    freeze = read_json(freeze_path)
    content = policy.get("policy_content")
    require(isinstance(content, dict), "policy content missing")
    content_hash = canonical_sha256(content)
    require(policy.get("policy_sha256") == content_hash, "policy canonical hash mismatch")
    require(freeze.get("policy_sha256") == content_hash, "policy freeze canonical hash mismatch")
    require(freeze.get("policy_file_sha256") == sha256_file(policy_path), "policy file changed after freeze")
    for key in (
        "protocol_version",
        "source_commit",
        "build_manifest_sha256",
        "environment_id",
        "environment_sha256",
        "openmp_threads",
    ):
        expected = held_out_report[key]
        require(content.get(key) == expected, f"policy/held-out {key} mismatch")
        require(freeze.get(key) == expected, f"freeze/held-out {key} mismatch")
    require(content.get("protocol_version") == PROTOCOL_VERSION, "policy protocol mismatch")
    require(set(content.get("held_out_queries", {})) == {str(value) for value in HELD_OUT_BODIES}, "policy query set mismatch")
    return {"policy": policy, "freeze": freeze, "content": content, "policy_sha256": content_hash}


def evaluate(policy_content: dict[str, Any], medians: dict[int, dict[str, float]]) -> list[dict[str, Any]]:
    order = list(policy_content["canonical_configuration_order"])
    results: list[dict[str, Any]] = []
    for bodies in HELD_OUT_BODIES:
        values = medians[bodies]
        require(set(values) == set(order), f"held-out configurations differ from policy at N={bodies}")
        selected = policy_content["held_out_queries"][str(bodies)]["selected_configuration"]
        require(selected in values, f"policy selected unavailable configuration at N={bodies}")
        oracle = min(order, key=lambda config: (values[config], order.index(config)))
        selected_median = values[selected]
        oracle_median = values[oracle]
        results.append(
            {
                "bodies": bodies,
                "selected_configuration": selected,
                "selected_median_ms": selected_median,
                "oracle_configuration": oracle,
                "oracle_median_ms": oracle_median,
                "multiplicative_regret": selected_median / oracle_median,
                "selected_empirical_fastest": selected == oracle,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held_out_dir", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    held_out = args.held_out_dir.resolve()
    report = verify_evidence(held_out)
    require(report["mode"] == "held_out", "evaluation requires verified held-out evidence")
    chain = validate_frozen_policy(args.policy.resolve(), args.policy_freeze.resolve(), report)
    results = evaluate(chain["content"], load_held_out_medians(held_out))
    output = args.output.resolve()
    require(not output.exists(), f"evaluation output already exists: {output}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_only": True,
        "source_commit": report["source_commit"],
        "build_manifest_sha256": report["build_manifest_sha256"],
        "environment_id": report["environment_id"],
        "environment_sha256": report["environment_sha256"],
        "held_out_manifest_sha256": report["manifest_sha256"],
        "policy_sha256": chain["policy_sha256"],
        "policy_file_sha256": sha256_file(args.policy.resolve()),
        "policy_freeze_sha256": sha256_file(args.policy_freeze.resolve()),
        "metric": PRIMARY_METRIC,
        "results": results,
        "aggregate": {
            "held_out_points": len(results),
            "empirical_fastest_selections": sum(item["selected_empirical_fastest"] for item in results),
            "maximum_multiplicative_regret": max(item["multiplicative_regret"] for item in results),
        },
        "claim_boundary": "six fixed-workload held-out N values; no universal best-backend claim",
    }
    write_json(output, payload)
    print(f"held-out evaluation written without policy refit: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Stage5Error, ValueError, FileNotFoundError, KeyError, csv.Error) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
