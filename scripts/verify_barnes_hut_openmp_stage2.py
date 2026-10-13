#!/usr/bin/env python3
"""Verify custody, matrix completeness, and summaries for Stage 2 evidence."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "benchmarks" / "barnes_hut_openmp_stage2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    return list(csv.DictReader(lines))


def median_ratio(rows: list[dict[str, str]], numerator: str, denominator: str) -> float:
    return statistics.median(
        float(row[numerator]) / float(row[denominator]) for row in rows
    )


def main() -> int:
    metadata_path = EVIDENCE / "metadata.json"
    raw_path = EVIDENCE / "raw.csv"
    provenance_path = EVIDENCE / "build_provenance.json"
    manifest_path = EVIDENCE / "source_manifest.sha256"
    for path in (metadata_path, raw_path, provenance_path, manifest_path):
        require(path.is_file(), f"missing evidence file: {path.relative_to(ROOT)}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = metadata["outputs"]
    require(sha256(raw_path) == outputs["raw_csv_sha256"], "raw CSV hash mismatch")
    require(
        sha256(provenance_path) == outputs["build_provenance_sha256"],
        "build provenance hash mismatch",
    )
    require(
        sha256(manifest_path) == outputs["candidate_source_manifest_sha256"],
        "candidate source/evidence manifest hash mismatch",
    )

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(" *", 1)
        path = ROOT / relative
        require(path.is_file(), f"manifest file missing: {relative}")
        require(sha256(path) == expected, f"manifest hash mismatch: {relative}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    cache = provenance["cmake_cache"]
    require(cache["NBODY_ENABLE_OPENMP"] == "ON", "provenance is not OpenMP-enabled")
    require(cache["NBODY_ENABLE_CUDA"] == "OFF", "provenance unexpectedly enables CUDA")
    require(cache["CMAKE_BUILD_TYPE"] == "Release", "provenance is not Release")
    require(cache["OpenMP_CXX_FLAGS"], "OpenMP compile flags are absent")
    require(cache["OpenMP_CXX_LIB_NAMES"], "OpenMP runtime libraries are absent")
    compile_commands = provenance["effective_commands"]["compile"]
    require(
        cache["OpenMP_CXX_FLAGS"] in compile_commands["src/barnes_hut.cpp"],
        "Barnes-Hut compile command omits the recorded OpenMP flags",
    )
    link = provenance["effective_commands"]["benchmark_link"]
    for library in cache["OpenMP_CXX_LIB_NAMES"].split(";"):
        require(library in link, f"benchmark link command omits {library}")

    rows = read_rows(raw_path)
    require(len(rows) == 216 == outputs["raw_rows"], "raw matrix must contain 216 rows")
    actual_keys: set[tuple[str, int, float, int, int]] = set()
    timing_fields = (
        "sequential_direct_force_s",
        "openmp_direct_force_s",
        "sequential_bh_tree_build_s",
        "sequential_bh_traversal_s",
        "sequential_bh_force_wall_s",
        "openmp_bh_tree_build_s",
        "openmp_bh_traversal_s",
        "openmp_bh_force_wall_s",
        "openmp_bh_step_s",
    )
    for row in rows:
        key = (
            row["distribution"],
            int(row["bodies"]),
            float(row["theta"]),
            int(row["threads"]),
            int(row["repetition"]),
        )
        require(key not in actual_keys, f"duplicate benchmark row: {key}")
        actual_keys.add(key)
        require(int(row["effective_threads"]) == int(row["threads"]),
                f"effective thread mismatch: {key}")
        for field in timing_fields:
            value = float(row[field])
            require(math.isfinite(value) and value > 0.0,
                    f"invalid {field} for {key}")

    expected_keys = set(itertools.product(
        ("uniform", "clustered", "planar"),
        (256, 1024, 4096),
        (0.25, 0.5, 0.8),
        (1, 2, 4, 8),
        (0, 1),
    ))
    require(actual_keys == expected_keys, "benchmark matrix is incomplete or unexpected")

    at_4096 = [row for row in rows if int(row["bodies"]) == 4096]
    speedups: dict[str, dict[str, float]] = {}
    for threads in (4, 8):
        values = []
        for distribution in ("uniform", "clustered", "planar"):
            for theta in (0.25, 0.5, 0.8):
                selected = [
                    row for row in at_4096
                    if row["distribution"] == distribution
                    and float(row["theta"]) == theta
                    and int(row["threads"]) == threads
                ]
                values.append(median_ratio(
                    selected, "sequential_bh_force_wall_s", "openmp_bh_force_wall_s"
                ))
        speedups[str(threads)] = {"minimum": min(values), "maximum": max(values)}

    theta_quarter_direct: dict[str, float] = {}
    for distribution in ("uniform", "clustered", "planar"):
        selected = [
            row for row in at_4096
            if row["distribution"] == distribution
            and float(row["theta"]) == 0.25
            and int(row["threads"]) == 8
        ]
        theta_quarter_direct[distribution] = median_ratio(
            selected, "openmp_direct_force_s", "openmp_bh_force_wall_s"
        )

    recorded = metadata["derived_summary"]
    for threads, bounds in speedups.items():
        for label, value in bounds.items():
            require(
                math.isclose(value, recorded["bh_speedup_vs_sequential_at_n4096"][threads][label],
                             rel_tol=1e-12, abs_tol=0.0),
                f"recorded {threads}-thread {label} speedup is stale",
            )
    for distribution, value in theta_quarter_direct.items():
        require(
            math.isclose(
                value,
                recorded["theta_0_25_bh_speedup_vs_openmp_direct_at_n4096_threads8"][distribution],
                rel_tol=1e-12,
                abs_tol=0.0,
            ),
            f"recorded theta=0.25 direct crossover is stale for {distribution}",
        )

    print("OpenMP Barnes-Hut Stage 2 evidence verified")
    print(json.dumps({
        "raw_rows": len(rows),
        "raw_sha256": sha256(raw_path),
        "build_provenance_sha256": sha256(provenance_path),
        "source_manifest_sha256": sha256(manifest_path),
        "bh_speedup_vs_sequential_at_n4096": speedups,
        "theta_0_25_bh_speedup_vs_openmp_direct_at_n4096_threads8": theta_quarter_direct,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
