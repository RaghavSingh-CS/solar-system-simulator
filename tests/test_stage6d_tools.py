#!/usr/bin/env python3
"""Tests for the bounded Stage 6D large-N pilot schedule and parser."""

from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_large_n_pilot", ROOT / "scripts" / "run_large_n_pilot.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import Stage 6D runner")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Stage6DToolTests(unittest.TestCase):
    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            sizes=RUNNER.DEFAULT_SIZES,
            backends=("sequential", "openmp", "mpi", "mpi+openmp"),
            mpi_ranks=2, omp_threads=4, steps=1, dt=10.0, warmups=1,
            repetitions=1, seed=24301, mpiexec="mpiexec",
            cpu_executable=Path("cpu.exe"), mpi_executable=Path("mpi.exe"),
            runtime_budget_s=30.0,
        )

    def test_schedule_is_complete_and_geometric(self) -> None:
        args = self.args()
        schedule = RUNNER.configurations(args)
        self.assertEqual(len(schedule), len(RUNNER.DEFAULT_SIZES) * len(args.backends))
        self.assertEqual(RUNNER.DEFAULT_SIZES[-1], 100000)
        self.assertEqual(
            {(item["backend"], item["bodies"]) for item in schedule},
            {(backend, bodies) for bodies in args.sizes for backend in args.backends},
        )

    def test_commands_record_topology_and_fixed_work(self) -> None:
        args = self.args()
        mpi = RUNNER.command_for(args, {
            "backend": "mpi+openmp", "bodies": 4096, "ranks": 2, "threads": 4})
        self.assertIn("-n", mpi)
        self.assertIn("4096", mpi)
        self.assertEqual(mpi[-2:], ["--threads", "4"])
        cpu = RUNNER.command_for(args, {
            "backend": "sequential", "bodies": 100000, "ranks": 1, "threads": 1})
        self.assertIn("100000", cpu)
        self.assertIn("--warmups", cpu)
        self.assertIn("--repetitions", cpu)

    def test_single_row_parser_rejects_missing_or_multiple_rows(self) -> None:
        self.assertEqual(RUNNER.parse_single_csv("a,b\n1,2\n"), {"a": "1", "b": "2"})
        with self.assertRaises(ValueError):
            RUNNER.parse_single_csv("a,b\n")
        with self.assertRaises(ValueError):
            RUNNER.parse_single_csv("a,b\n1,2\n3,4\n")


if __name__ == "__main__":
    unittest.main()
