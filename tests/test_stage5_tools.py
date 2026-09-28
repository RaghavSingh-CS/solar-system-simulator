#!/usr/bin/env python3
"""Unit and synthetic-evidence tests for the Stage 5 orchestration tools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


RUNNER = load_module("run_stage5", ROOT / "scripts" / "run_stage5.py")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


@unittest.skip("historical stage5-v1.0 tests; superseded by test_stage5_v2_tools.py")
class Stage5ToolTests(unittest.TestCase):
    def test_frozen_split_is_disjoint_and_complete(self) -> None:
        calibration = set(RUNNER.CALIBRATION_BODIES)
        held_out = set(RUNNER.HELD_OUT_BODIES)
        self.assertFalse(calibration & held_out)
        self.assertEqual(tuple(sorted(calibration | held_out)), RUNNER.FULL_BODIES)
        self.assertEqual(
            RUNNER.FULL_BODIES,
            (64, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096),
        )

    def test_schedule_is_deterministic_and_block_complete(self) -> None:
        threads = (1, 2, 4)
        configs = RUNNER.configurations(RUNNER.FULL_BODIES, threads, "full")
        first = RUNNER.ordered_runs(configs, 11, RUNNER.DEFAULT_ORDER_SEED)
        second = RUNNER.ordered_runs(configs, 11, RUNNER.DEFAULT_ORDER_SEED)
        self.assertEqual(first, second)
        expected_keys = {
            (item["backend"], item["threads"], item["bodies"], item["role"])
            for item in configs
        }
        for block in range(1, 12):
            actual = {
                (item["backend"], item["threads"], item["bodies"], item["role"])
                for item in first if item["block_id"] == block
            }
            self.assertEqual(actual, expected_keys)
            self.assertEqual(sum(item["block_id"] == block for item in first), len(configs))

    def test_synthetic_pilot_passes_independent_verifier(self) -> None:
        threads = (1, 2)
        configurations = RUNNER.configurations(RUNNER.PILOT_BODIES, threads, "pilot")
        schedule = RUNNER.ordered_runs(configurations, 3, RUNNER.DEFAULT_ORDER_SEED)
        rows: list[dict[str, str]] = []
        for order_index, config in enumerate(schedule, start=1):
            backend = str(config["backend"])
            requested = int(config["threads"])
            metrics = (
                ("device_compute", 2.0),
                ("transfer_inclusive", 2.2),
                ("selector_steady_state", 2.4),
            ) if backend == "cuda" else (
                ("integration_compute", 3.0 + requested),
                ("selector_steady_state", 3.1 + requested),
            )
            common = {
                "environment_id": "synthetic-rtx4050",
                "source_commit": "a" * 40,
                "protocol_version": RUNNER.PROTOCOL_VERSION,
                "provenance_mode": "git-verified",
                "experiment_id": "synthetic-pilot",
                "mode": "pilot",
                "dataset_role": "pilot_only",
                "run_id": f"b{config['block_id']:02d}-o{order_index:04d}",
                "order_seed": str(RUNNER.DEFAULT_ORDER_SEED),
                "block_id": str(config["block_id"]),
                "order_index": str(order_index),
                "repetition": str(config["block_id"]),
                "backend": backend,
                "requested_threads": str(requested),
                "effective_threads": str(requested),
                "bodies": str(config["bodies"]),
                "steps": "10",
                "dt_s": "10",
                "benchmark_seed": str(RUNNER.DEFAULT_BENCHMARK_SEED),
                "warmups": "1",
                "correctness_status": "pass",
                "checksum": f"cuda-{config['bodies']}" if backend == "cuda" else f"cpu-{config['bodies']}",
                "relative_energy_drift": "1e-12",
                "relative_angular_momentum_drift": "1e-13",
                "energy_drift_bound": "5e-6",
                "angular_momentum_drift_bound": "5e-10",
                "build_configuration": "Release",
                "openmp_specification_date": "202111" if backend == "openmp" else "0" if backend == "sequential" else "",
                "setup_allocation_ms": "0.5" if backend == "cuda" else "",
                "host_staging_ms": "0.1" if backend == "cuda" else "",
                "host_to_device_ms": "0.1" if backend == "cuda" else "",
                "device_to_host_ms": "0.1" if backend == "cuda" else "",
                "device_index": "0" if backend == "cuda" else "",
                "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU" if backend == "cuda" else "",
                "compute_capability": "8.9" if backend == "cuda" else "",
                "total_global_memory_bytes": "6442450944" if backend == "cuda" else "",
                "multiprocessor_count": "20" if backend == "cuda" else "",
                "cuda_driver_version": "13000" if backend == "cuda" else "",
                "cuda_runtime_version": "13000" if backend == "cuda" else "",
            }
            for metric, elapsed in metrics:
                rows.append({**common, "timing_metric": metric, "elapsed_ms": str(elapsed)})

        RUNNER.validate_collected_rows(rows, schedule)

        summary_args = argparse.Namespace(
            full=False,
            order_seed=RUNNER.DEFAULT_ORDER_SEED,
            bootstrap_resamples=100,
        )
        summaries = RUNNER.summary_rows(rows, summary_args)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            RUNNER.write_csv(evidence / "raw_timings.csv", RUNNER.RAW_FIELDS, rows)
            RUNNER.write_csv(evidence / "summary.csv", RUNNER.SUMMARY_FIELDS, summaries)
            (evidence / "ctest.txt").write_text(
                "100% tests passed, 0 tests failed out of 3\n", encoding="utf-8"
            )
            environment = {
                "experiment_id": "synthetic-pilot",
                "environment_id": "synthetic-rtx4050",
                "protocol_version": RUNNER.PROTOCOL_VERSION,
                "mode": "pilot",
                "source_provenance": {
                    "mode": "git-verified", "source_commit": "a" * 40,
                    "git_head": "a" * 40, "branch": "main", "status_before_output": "",
                },
                "platform": {"system": "Windows"},
                "build": {"cmake_cache": {"NBODY_ENABLE_OPENMP": "ON", "NBODY_ENABLE_CUDA": "ON"}},
                "openmp_environment": {
                    "requested_threads": list(threads),
                    "openmp_specification_date": 202111,
                },
                "observed_cuda_device": {
                    "device_index": 0,
                    "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                    "compute_capability": "8.9",
                    "total_global_memory_bytes": 6442450944,
                    "multiprocessor_count": 20,
                    "cuda_driver_version": 13000,
                    "cuda_runtime_version": 13000,
                },
                "protocol": {
                    "calibration_bodies": list(RUNNER.CALIBRATION_BODIES),
                    "held_out_bodies": list(RUNNER.HELD_OUT_BODIES),
                    "measured_bodies": list(RUNNER.PILOT_BODIES),
                    "steps": 10, "dt_s": 10.0, "repetitions": 3,
                    "order_seed": RUNNER.DEFAULT_ORDER_SEED,
                    "bootstrap_resamples": 100, "raw_rows": len(rows),
                },
                "full_run_pilot_gate": {},
            }
            (evidence / "environment.json").write_text(
                json.dumps(environment, indent=2) + "\n", encoding="utf-8"
            )
            names = ("raw_timings.csv", "summary.csv", "environment.json", "ctest.txt")
            manifest = {
                "protocol_version": RUNNER.PROTOCOL_VERSION,
                "mode": "pilot",
                "source_commit": "a" * 40,
                "files": {
                    name: {"sha256": file_hash(evidence / name), "bytes": (evidence / name).stat().st_size}
                    for name in names
                },
            }
            (evidence / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "verify_stage5_evidence.py"), str(evidence)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["verification_status"], "pass")
            self.assertEqual(report["raw_rows"], len(rows))


if __name__ == "__main__":
    unittest.main()
