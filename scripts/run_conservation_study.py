#!/usr/bin/env python3
"""Run the bounded sequential conservation-characterization matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SCENARIOS = ("analytic_two_body", "seeded_32_body")
TIMESTEP_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
DURATION_MULTIPLIERS = (1.0, 10.0, 100.0)
REPEATABILITY_EXCLUDED_FIELDS = {"experiment_id", "order_index", "experiment_wall_ms"}


def arguments() -> argparse.Namespace:
    default_executable = Path("build-sequential/nbody_conservation_probe")
    if os.name == "nt":
        default_executable = default_executable.with_suffix(".exe")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=default_executable)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/experiments/conservation")
    )
    parser.add_argument("--experiment-id", default="conservation-study-2026-09-26")
    parser.add_argument("--cmake-executable", default="cmake")
    parser.add_argument("--compiler-executable", default="c++")
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


def probe_command(
    executable: Path,
    scenario: str,
    timestep_multiplier: float,
    duration_multiplier: float,
) -> list[str]:
    return [
        str(executable),
        "--scenario", scenario,
        "--dt-multiplier", str(timestep_multiplier),
        "--duration-multiplier", str(duration_multiplier),
    ]


def run_probe(command: list[str]) -> dict[str, str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or completed.stderr.strip():
        raise RuntimeError(
            f"probe failed: {' '.join(command)}\n"
            f"return_code={completed.returncode}\n{completed.stderr}"
        )
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"expected one probe row from {' '.join(command)}")
    return rows[0]


def require_finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {field} in {row['scenario']}")
    return value


def validate_rows(rows: list[dict[str, str]]) -> None:
    expected = len(SCENARIOS) * len(TIMESTEP_MULTIPLIERS) * len(DURATION_MULTIPLIERS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    keys: set[tuple[str, str, str]] = set()
    numeric_fields = (
        "base_dt_s", "dt_s", "dt_multiplier", "baseline_duration_s",
        "duration_multiplier", "simulated_duration_s", "initial_energy_j",
        "requested_duration_s", "time_accumulation_error_s", "final_energy_j",
        "maximum_relative_energy_drift",
        "final_relative_energy_drift", "initial_angular_x", "initial_angular_y",
        "initial_angular_z", "initial_angular_magnitude", "final_angular_x",
        "final_angular_y", "final_angular_z", "final_angular_magnitude",
        "maximum_relative_angular_momentum_drift",
        "final_relative_angular_momentum_drift", "energy_drift_bound",
        "angular_momentum_drift_bound", "experiment_wall_ms", "final_state_checksum",
    )
    for row in rows:
        key = (row["scenario"], row["dt_multiplier"], row["duration_multiplier"])
        if key in keys:
            raise RuntimeError(f"duplicate configuration: {key}")
        keys.add(key)
        for field in numeric_fields:
            require_finite(row, field)
        if row["backend"] != "sequential" or row["integrator"] != "velocity_verlet":
            raise RuntimeError("unexpected backend or integrator")
        if row["precision"] != "binary64" or row["overall_status"] not in {"pass", "fail"}:
            raise RuntimeError("unexpected precision or status")
        steps = int(row["steps"])
        if steps <= 0 or int(row["diagnostic_samples"]) != steps + 1:
            raise RuntimeError("invalid step or diagnostic-sample count")
        expected_duration = steps * float(row["dt_s"])
        if not math.isclose(
            float(row["requested_duration_s"]), expected_duration,
            rel_tol=1.0e-15, abs_tol=1.0e-9,
        ):
            raise RuntimeError("requested duration does not equal steps * dt")
        target_duration = (
            float(row["baseline_duration_s"]) * float(row["duration_multiplier"])
        )
        if not math.isclose(expected_duration, target_duration, rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise RuntimeError("configuration does not preserve its target physical duration")
        accumulated_duration = float(row["simulated_duration_s"])
        accumulation_error = float(row["time_accumulation_error_s"])
        if not math.isclose(
            accumulated_duration - expected_duration, accumulation_error,
            rel_tol=1.0e-12, abs_tol=1.0e-9,
        ) or not math.isclose(
            accumulated_duration, expected_duration, rel_tol=1.0e-9, abs_tol=1.0e-9,
        ):
            raise RuntimeError("state-time accumulation error is unexpectedly large")


def deterministic_projection(row: dict[str, str]) -> dict[str, str]:
    return {
        key: value for key, value in row.items()
        if key not in REPEATABILITY_EXCLUDED_FIELDS
    }


def write_summary(rows: list[dict[str, str]], path: Path) -> None:
    by_scenario: dict[str, dict[str, object]] = {}
    for scenario in SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        by_timestep = []
        for multiplier in TIMESTEP_MULTIPLIERS:
            timestep_rows = [
                row for row in selected if float(row["dt_multiplier"]) == multiplier
            ]
            by_timestep.append({
                "dt_multiplier": multiplier,
                "dt_s": float(timestep_rows[0]["dt_s"]),
                "maximum_energy_drift_across_durations": max(
                    float(row["maximum_relative_energy_drift"]) for row in timestep_rows
                ),
                "maximum_angular_drift_across_durations": max(
                    float(row["maximum_relative_angular_momentum_drift"])
                    for row in timestep_rows
                ),
                "all_configurations_pass": all(
                    row["overall_status"] == "pass" for row in timestep_rows
                ),
            })
        by_scenario[scenario] = {
            "rows": len(selected),
            "all_configurations_pass": all(row["overall_status"] == "pass" for row in selected),
            "maximum_energy_drift": max(
                float(row["maximum_relative_energy_drift"]) for row in selected
            ),
            "maximum_angular_drift": max(
                float(row["maximum_relative_angular_momentum_drift"]) for row in selected
            ),
            "by_timestep": by_timestep,
        }
    path.write_text(json.dumps({"scenarios": by_scenario}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = arguments()
    executable = args.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)

    git_metadata = {
        "head": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "status_before_output": git_value("status", "--porcelain=v1"),
    }
    started = datetime.now(timezone.utc)
    rows: list[dict[str, str]] = []
    order_index = 0
    for scenario in SCENARIOS:
        for duration_multiplier in DURATION_MULTIPLIERS:
            for timestep_multiplier in TIMESTEP_MULTIPLIERS:
                order_index += 1
                command = probe_command(
                    executable, scenario, timestep_multiplier, duration_multiplier
                )
                row = run_probe(command)
                rows.append({
                    "experiment_id": args.experiment_id,
                    "order_index": str(order_index),
                    **row,
                })
                print(
                    f"[{order_index:02d}/30] {scenario} dt={timestep_multiplier:g}x "
                    f"duration={duration_multiplier:g}x "
                    f"max_dE={row['maximum_relative_energy_drift']} "
                    f"max_dL={row['maximum_relative_angular_momentum_drift']}",
                    flush=True,
                )

    validate_rows(rows)
    repeat_command = probe_command(executable, "seeded_32_body", 1.0, 10.0)
    original = next(
        row for row in rows
        if row["scenario"] == "seeded_32_body"
        and float(row["dt_multiplier"]) == 1.0
        and float(row["duration_multiplier"]) == 10.0
    )
    repeated = run_probe(repeat_command)
    repeat_matches = deterministic_projection(original) == deterministic_projection(repeated)
    if not repeat_matches:
        raise RuntimeError("repeatability check changed a scientific output field")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, output_dir / "summary.json")

    completed = datetime.now(timezone.utc)
    environment = {
        "experiment_id": args.experiment_id,
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "git": git_metadata,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_processors": os.cpu_count(),
        },
        "tools": {
            "cmake": command_output([args.cmake_executable, "--version"]),
            "compiler": command_output([args.compiler_executable, "--version"]),
        },
        "protocol": {
            "backend": "deterministic sequential only",
            "integrator": "existing Velocity Verlet implementation",
            "precision": "binary64/double",
            "scenarios": list(SCENARIOS),
            "timestep_multipliers": list(TIMESTEP_MULTIPLIERS),
            "duration_multipliers": list(DURATION_MULTIPLIERS),
            "diagnostics": "initial state and every completed integration step",
            "energy_drift": "abs(E-E0) / max(abs(E), abs(E0), 1 joule)",
            "angular_momentum_drift": "norm(L-L0) / max(norm(L0), 1 kg m^2 / s)",
            "energy_bound": 5.0e-6,
            "angular_momentum_bound": 5.0e-10,
            "runtime_scope": "incidental wall time for integration plus per-step diagnostics; not a backend benchmark",
        },
        "repeatability": {
            "scenario": "seeded_32_body",
            "dt_multiplier": 1.0,
            "duration_multiplier": 10.0,
            "excluded_fields": sorted(REPEATABILITY_EXCLUDED_FIELDS),
            "all_scientific_fields_exactly_equal": repeat_matches,
        },
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
