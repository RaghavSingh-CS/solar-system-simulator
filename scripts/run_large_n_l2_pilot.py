#!/usr/bin/env python3
"""Run and transactionally publish the bounded Large-N L2 8192-body pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import run_large_n_l1_pilot as publication_core


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "benchmarks" / "large_n_l2_pilot"
DEFAULT_PROTOCOL = ROOT / "tools" / "large_n_l2_protocol.json"
RUN_FILES = (
    "raw.csv",
    "summary.json",
    "metadata.json",
    "build_provenance.json",
    "source_manifest.sha256",
    "protocol.json",
    "README.md",
)
TIMING_FIELDS = (
    "generation_s",
    "initial_direct_force_s",
    "direct_trajectory_s",
    "diagnostics_s",
    "complete_force_s",
    "complete_timestep_s",
    "steps_per_second",
    "total_experiment_s",
    "tree_build_s",
    "force_traversal_s",
)
SCIENTIFIC_FIELDS = (
    "max_abs_acceleration_error_m_s2",
    "rms_abs_acceleration_error_m_s2",
    "p50_relative_error",
    "p95_relative_error",
    "p99_relative_error",
    "max_relative_error",
    "relative_floor_m_s2",
    "cancellation_sensitive_count",
    "max_position_separation_m",
    "rms_position_separation_m",
    "max_velocity_separation_m_s",
    "rms_velocity_separation_m_s",
    "candidate_relative_energy_drift",
    "direct_relative_energy_drift",
    "candidate_relative_angular_momentum_drift",
    "direct_relative_angular_momentum_drift",
)
RESOURCE_FIELDS = (
    "solver_memory_bytes",
    "node_count",
    "visited_nodes",
    "approximated_nodes",
    "direct_interactions",
    "estimated_memory_bytes",
    "generation_memory_bytes",
    "direct_reference_memory_bytes",
    "warmup_memory_bytes",
    "retained_repetition_memory_bytes",
    "characterization_memory_bytes",
    "serialization_memory_bytes",
    "estimated_direct_interactions",
)
DERIVED_FIELDS = (
    "protocol_phase",
    "reference_mode",
    "reference_sample_size",
    "accuracy_gate",
    "accuracy_reason",
    "steps_per_second",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def csv_float(value: float) -> str:
    """Match the laboratory's max_digits10 binary64 CSV serialization."""
    return format(float(value), ".17g")


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == "large-n-l2-protocol-v1", "wrong L2 protocol schema")
    require(protocol.get("version") == "2.0.0-pilot-8192", "wrong L2 protocol version")
    require(protocol.get("study") == "large_n_laboratory_l2_controlled_scaling", "wrong study")
    require(protocol.get("pilot_body_counts") == [8192], "pilot must contain only N=8192")
    require(
        protocol.get("candidate_body_counts") == [8192, 16384, 32768, 50000]
        and protocol.get("optional_stretch_body_count") == 100000,
        "candidate scaling sizes changed",
    )
    require(protocol.get("distributions") == ["uniform", "clustered", "planar"], "distribution matrix changed")
    require(
        protocol.get("backends")
        == [{"name": "sequential", "threads": 1}, {"name": "openmp", "threads": 4}],
        "backend/thread matrix changed",
    )
    execution = protocol.get("execution", {})
    require(
        execution
        == {
            "hard_subprocess_timeout_s": 35.0,
            "memory_budget_bytes": 536870912,
            "maximum_direct_interactions": 500000000,
            "maximum_direct_reference_bodies": 8192,
            "per_configuration_wall_time_budget_s": 30.0,
            "retained_repetitions": 3,
            "steps": 1,
            "timestep_s": 10.0,
            "warmups": 1,
        },
        "execution/resource policy changed",
    )
    require(
        protocol.get("solvers")
        == {
            "barnes-hut": {
                "request_direct_reference": True,
                "theta_values": [0.25, 0.5, 0.8],
            },
            "direct": {
                "expected_pilot_outcome": "resource_skip",
                "request_direct_reference": False,
                "theta_values": [0.25],
            },
        },
        "solver matrix changed",
    )
    require(
        protocol.get("reference_policy", {}).get("sampled", "").startswith("not implemented"),
        "sampled reference policy must remain explicitly unavailable",
    )
    thresholds = protocol.get("accuracy_thresholds", {})
    require(set(thresholds) == {"0.25", "0.5", "0.8"}, "accuracy thresholds changed")
    for value in thresholds.values():
        require(
            set(value)
            == {
                "maximum_relative_error",
                "p95_relative_error",
                "relative_angular_momentum_drift",
                "relative_energy_drift",
            }
            and all(math.isfinite(float(item)) and float(item) > 0.0 for item in value.values()),
            "accuracy threshold is invalid",
        )
    return protocol


def configurations(protocol: dict) -> list[dict]:
    result = []
    base_seed = 0x4C3250494C4F5400
    bodies = protocol["pilot_body_counts"][0]
    for distribution_index, distribution in enumerate(protocol["distributions"]):
        seed = base_seed + distribution_index * 0x100000 + bodies
        for backend in protocol["backends"]:
            result.append(
                {
                    "distribution": distribution,
                    "bodies": bodies,
                    "seed": seed,
                    "backend": backend["name"],
                    "threads": backend["threads"],
                    "solver": "direct",
                    "theta": 0.25,
                    "request_direct_reference": False,
                    "reference_mode": "none",
                }
            )
            for theta in protocol["solvers"]["barnes-hut"]["theta_values"]:
                result.append(
                    {
                        "distribution": distribution,
                        "bodies": bodies,
                        "seed": seed,
                        "backend": backend["name"],
                        "threads": backend["threads"],
                        "solver": "barnes-hut",
                        "theta": theta,
                        "request_direct_reference": True,
                        "reference_mode": "full",
                    }
                )
    return result


def accuracy_gate(row: dict[str, str], protocol: dict) -> tuple[str, str]:
    if row["status"] != "completed" or row["solver"] != "barnes-hut":
        return "not_applicable", "accuracy gate applies only to completed Barnes-Hut rows"
    theta_key = format(float(row["theta"]), ".15g")
    threshold = protocol["accuracy_thresholds"][theta_key]
    checks = {
        "p95_relative_error": float(row["p95_relative_error"]),
        "maximum_relative_error": float(row["max_relative_error"]),
        "relative_energy_drift": float(row["candidate_relative_energy_drift"]),
        "relative_angular_momentum_drift": float(row["candidate_relative_angular_momentum_drift"]),
    }
    failures = [name for name, value in checks.items() if value > float(threshold[name])]
    if failures:
        return "failed", "threshold exceeded: " + ", ".join(failures)
    return "passed", "all pre-registered theta-specific thresholds satisfied"


def reconstruct_summary(rows: list[dict[str, str]], protocol: dict) -> dict:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["distribution"], row["bodies"], row["backend"], row["solver"],
            row["theta"], row["effective_threads"],
        )
        grouped.setdefault(key, []).append(row)
    groups = []
    for key in sorted(grouped):
        group = grouped[key]
        status = group[0]["status"]
        item = {
            "distribution": key[0],
            "bodies": int(key[1]),
            "backend": key[2],
            "solver": key[3],
            "theta": float(key[4]),
            "effective_threads": int(key[5]),
            "status": status,
            "message": group[0]["message"],
            "rows": len(group),
            "reference_mode": group[0]["reference_mode"],
            "reference_sample_size": int(group[0]["reference_sample_size"]),
            "accuracy_gate": group[0]["accuracy_gate"],
            "accuracy_reason": group[0]["accuracy_reason"],
        }
        if status == "completed":
            medians = {
                field: statistics.median(float(row[field]) for row in group)
                for field in TIMING_FIELDS
            }
            force = [float(row["complete_force_s"]) for row in group]
            item.update(
                {
                    "retained_repetitions": len(group),
                    "observed_worker_threads": int(group[0]["observed_worker_threads"]),
                    "worker_count_provenance": group[0]["worker_count_provenance"],
                    "medians": medians,
                    "force_range_over_median": (max(force) - min(force)) / medians["complete_force_s"],
                    "scientific": {field: float(group[0][field]) for field in SCIENTIFIC_FIELDS},
                    "resources": {field: int(group[0][field]) for field in RESOURCE_FIELDS},
                }
            )
        groups.append(item)
    completed = [item for item in groups if item["status"] == "completed"]
    direct_skips = [item for item in groups if item["solver"] == "direct" and item["status"] == "skipped"]
    accuracy_pass = all(item["accuracy_gate"] == "passed" for item in completed)
    max_budget_fraction = max(
        (item["medians"]["total_experiment_s"] / protocol["execution"]["per_configuration_wall_time_budget_s"] for item in completed),
        default=1.0,
    )
    max_memory_fraction = max(
        (item["resources"]["estimated_memory_bytes"] / protocol["execution"]["memory_budget_bytes"] for item in completed),
        default=1.0,
    )
    recommend = (
        len(completed) == 18
        and len(direct_skips) == 6
        and accuracy_pass
        and max_budget_fraction <= 0.8
        and max_memory_fraction <= 0.9
    )
    return {
        "schema": "large-n-l2-summary-v1",
        "definitions": {
            "complete_force_s": "one complete force call including Barnes-Hut tree build",
            "complete_timestep_s": "one Velocity Verlet step containing two force evaluations",
            "steps_per_second": "configured steps divided by complete timestep wall time; not rendering FPS",
            "reference_mode_full": "complete all-body Sequential Direct force and matched one-step trajectory",
            "reference_mode_none": "no Direct reference; never interpreted as sampled or exact comparison",
        },
        "groups": groups,
        "pilot_decision": {
            "accuracy_gate_passed": accuracy_pass,
            "all_expected_barnes_hut_groups_completed": len(completed) == 18,
            "all_direct_groups_resource_skipped": len(direct_skips) == 6,
            "maximum_wall_budget_fraction": max_budget_fraction,
            "maximum_memory_budget_fraction": max_memory_fraction,
            "recommend_16384_performance_only_preflight": recommend,
            "reason": (
                "8192 pilot gates passed; 16384 may proceed only after independent review and without claiming a full Direct reference"
                if recommend
                else "one or more pilot continuation gates failed; do not advance automatically"
            ),
        },
    }


def write_run_manifest(run_directory: Path, run_id: str) -> dict:
    require(publication_core.valid_run_id(run_id), "unsafe L2 run ID")
    manifest = {
        "schema": "large-n-l2-run-v1",
        "status": "complete",
        "run_id": run_id,
        "files": {name: publication_core.sha256(run_directory / name) for name in RUN_FILES},
    }
    publication_core.write_json(run_directory / "run_manifest.json", manifest)
    return manifest


def read_run_manifest(run_directory: Path) -> dict:
    path = run_directory / "run_manifest.json"
    require(path.is_file(), f"missing L2 run manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(
        manifest
        == {
            "schema": "large-n-l2-run-v1",
            "status": "complete",
            "run_id": manifest.get("run_id"),
            "files": manifest.get("files"),
        },
        "invalid L2 run manifest schema",
    )
    require(publication_core.valid_run_id(manifest["run_id"]), "unsafe L2 run ID")
    require(set(manifest["files"]) == set(RUN_FILES), "invalid L2 run artifact set")
    for name, expected in manifest["files"].items():
        require(publication_core.sha256(run_directory / name) == expected, f"L2 artifact hash mismatch: {name}")
    return manifest


def descriptor(run_directory: Path) -> dict:
    manifest = read_run_manifest(run_directory)
    return {
        "run_id": manifest["run_id"],
        "run_manifest_sha256": publication_core.sha256(run_directory / "run_manifest.json"),
        "files": manifest["files"],
    }


def read_pointer(evidence_root: Path) -> dict | None:
    path = evidence_root / "publication.json"
    if not path.is_file():
        return None
    pointer = json.loads(path.read_text(encoding="utf-8"))
    require(
        pointer
        == {
            "schema": "large-n-l2-pointer-v1",
            "status": "complete",
            "current": pointer.get("current"),
            "history": pointer.get("history"),
        },
        "invalid L2 publication pointer",
    )
    require(isinstance(pointer["history"], list), "L2 history is not a list")
    seen = set()
    for item in [pointer["current"], *pointer["history"]]:
        require(isinstance(item, dict) and publication_core.valid_run_id(item.get("run_id")), "unsafe L2 descriptor")
        require(item["run_id"] not in seen, "duplicate L2 run ID")
        seen.add(item["run_id"])
        require(item == descriptor(evidence_root / "runs" / item["run_id"]), "L2 descriptor mismatch")
    return pointer


def publish_run(staged: Path, evidence_root: Path, run_id: str) -> dict:
    with publication_core.PublicationLock(evidence_root):
        pointer = read_pointer(evidence_root)
        runs = evidence_root / "runs"
        runs.mkdir(exist_ok=True)
        incomplete = sorted(path.name for path in runs.iterdir() if path.is_dir() and path.name.startswith(".install-"))
        referenced = set() if pointer is None else {pointer["current"]["run_id"], *(item["run_id"] for item in pointer["history"])}
        orphans = sorted(path.name for path in runs.iterdir() if path.is_dir() and not path.name.startswith(".install-") and path.name not in referenced)
        require(not incomplete, "incomplete L2 run installation requires inspection")
        require(not orphans, "unreferenced L2 run requires explicit inspection")
        require(read_run_manifest(staged)["run_id"] == run_id, "staged L2 run ID mismatch")
        destination = runs / run_id
        require(not destination.exists(), "L2 run ID already exists")
        os.replace(staged, destination)
        current = descriptor(destination)
        history = [] if pointer is None else [pointer["current"], *pointer["history"]]
        next_pointer = {
            "schema": "large-n-l2-pointer-v1",
            "status": "complete",
            "current": current,
            "history": history,
        }
        publication_core.atomic_replace_json(evidence_root / "publication.json", next_pointer, None)
        require(read_pointer(evidence_root) == next_pointer, "L2 publication confirmation failed")
        return next_pointer


def evidence_readme(summary: dict) -> str:
    decision = summary["pilot_decision"]
    return f"""# Large-N Laboratory Phase L2 — 8192-body pilot

This immutable run is exploratory controlled-scaling evidence, not frozen Stage 5
evidence and not a final crossover study. It contains the pre-registered N=8192
matrix only. Direct configurations that exceed the unchanged 500,000,000
interaction budget are retained as explicit skips; no fallback occurred.

Barnes--Hut configurations use complete all-body Direct references. No sampled
reference was produced. The theta-specific accuracy gate is
`{'passed' if decision['accuracy_gate_passed'] else 'failed'}`. The recommendation
for a separately reviewed 16384 performance-only preflight is
`{decision['recommend_16384_performance_only_preflight']}`.

`complete_force_s`, `complete_timestep_s`, and `steps_per_second` measure
simulation compute only. They are not rendering FPS. Three retained repetitions
support variability reporting but do not establish a universal crossover or
accuracy guarantee. CUDA, graphical integration, 16384, 32768, 50000, and
100000 execution are outside this run.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=ROOT / "build" / "openmp-release" / "nbody_large_n_lab.exe")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol.resolve())
    executable = args.executable.resolve()
    output = args.output.resolve()
    require(executable.is_file(), f"large-N executable not found: {executable}")
    execution = protocol["execution"]
    scratch_root = ROOT / "build" / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    run_id = publication_core.new_run_id()
    with tempfile.TemporaryDirectory(prefix="large-n-l2-run-", dir=scratch_root) as temporary:
        temporary_root = Path(temporary)
        staged = temporary_root / "candidate"
        work = temporary_root / "work"
        staged.mkdir()
        work.mkdir()
        all_rows: list[dict[str, str]] = []
        input_fields: list[str] | None = None
        command_log = []
        for index, configuration in enumerate(configurations(protocol)):
            path = work / f"configuration_{index:03d}.csv"
            command = [
                str(executable), "--output", str(path),
                "--bodies", str(configuration["bodies"]),
                "--seed", str(configuration["seed"]),
                "--distribution", configuration["distribution"],
                "--backend", configuration["backend"],
                "--solver", configuration["solver"],
                "--theta", str(configuration["theta"]),
                "--threads", str(configuration["threads"]),
                "--dt", str(execution["timestep_s"]),
                "--steps", str(execution["steps"]),
                "--warmups", str(execution["warmups"]),
                "--repetitions", str(execution["retained_repetitions"]),
                "--direct-reference", str(configuration["request_direct_reference"]).lower(),
                "--max-direct-reference-bodies", str(execution["maximum_direct_reference_bodies"]),
                "--memory-budget-bytes", str(execution["memory_budget_bytes"]),
                "--max-direct-interactions", str(execution["maximum_direct_interactions"]),
                "--wall-time-budget-s", str(execution["per_configuration_wall_time_budget_s"]),
            ]
            command_log.append(command)
            try:
                completed = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True,
                    timeout=execution["hard_subprocess_timeout_s"], check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"L2 configuration {index} exceeded its hard timeout") from error
            require(completed.returncode == 0, f"L2 configuration {index} failed: {completed.stdout}\n{completed.stderr}")
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(line for line in stream if line.strip() and not line.startswith("#"))
                rows = list(reader)
                fields = list(reader.fieldnames or [])
            if input_fields is None:
                input_fields = fields
            require(fields == input_fields, "inconsistent L2 CSV fields")
            if configuration["solver"] == "direct":
                require(len(rows) == 1 and rows[0]["status"] == "skipped", "Direct pilot configuration must be explicitly resource-skipped")
                require("interaction estimate exceeds" in rows[0]["message"], "Direct skip reason is not the work budget")
            else:
                require(len(rows) == 3 and all(row["status"] == "completed" for row in rows), "Barnes-Hut configuration did not complete three repetitions")
                require(all(row["direct_reference_available"] == "1" for row in rows), "Barnes-Hut pilot lacks a full Direct reference")
            for row in rows:
                gate, reason = accuracy_gate(row, protocol)
                row.update(
                    {
                        "protocol_phase": "exploratory_pilot_8192",
                        "reference_mode": configuration["reference_mode"],
                        "reference_sample_size": str(configuration["bodies"] if configuration["reference_mode"] == "full" else 0),
                        "accuracy_gate": gate,
                        "accuracy_reason": reason,
                        "steps_per_second": (
                            format(int(row["steps"]) / float(row["complete_timestep_s"]), ".17g")
                            if row["status"] == "completed" else "0"
                        ),
                    }
                )
                all_rows.append(row)
        require(input_fields is not None, "L2 matrix produced no rows")
        raw = staged / "raw.csv"
        with raw.open("w", encoding="utf-8", newline="") as stream:
            stream.write("# study=large_n_laboratory_l2_controlled_scaling\n# stage5_evidence=false\n# phase=exploratory_pilot_8192\n# graphical_fps=false\n")
            writer = csv.DictWriter(stream, fieldnames=[*input_fields, *DERIVED_FIELDS], lineterminator="\n")
            writer.writeheader()
            writer.writerows(all_rows)
        summary = reconstruct_summary(all_rows, protocol)
        publication_core.write_json(staged / "summary.json", summary)
        (staged / "protocol.json").write_bytes(args.protocol.resolve().read_bytes())
        status_counts = {status: sum(row["status"] == status for row in all_rows) for status in ("completed", "skipped")}
        metadata = {
            "schema": "large-n-l2-metadata-v1",
            "study": protocol["study"],
            "protocol_version": protocol["version"],
            "publication_status": "staging",
            "run_id": run_id,
            "stage5_evidence": False,
            "classification": "exploratory_pilot",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_state": "uncommitted Phase L2 candidate; source manifest binds exact bytes",
            "platform": {
                "system": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
                "logical_processors": os.cpu_count(),
            },
            "matrix": {
                "body_counts": protocol["pilot_body_counts"],
                "distributions": protocol["distributions"],
                "theta_values": protocol["solvers"]["barnes-hut"]["theta_values"],
                "backends": [item["name"] for item in protocol["backends"]],
                "solvers": ["direct", "barnes-hut"],
                **execution,
                "configuration_groups": 24,
                "expected_completed_rows": 54,
                "expected_skipped_rows": 6,
                "seed_formula": protocol["seed_formula"],
            },
            "commands": {"runner": "python -B scripts/run_large_n_l2_pilot.py", "configuration_argv": command_log},
            "outputs": {
                "raw_csv": "raw.csv",
                "raw_csv_sha256": publication_core.sha256(raw),
                "raw_rows": len(all_rows),
                "status_counts": status_counts,
                "summary_json": "summary.json",
                "summary_json_sha256": publication_core.sha256(staged / "summary.json"),
                "protocol": "protocol.json",
                "protocol_sha256": publication_core.sha256(staged / "protocol.json"),
            },
            "pilot_decision": summary["pilot_decision"],
            "limitations": [
                "Exploratory single-machine CPU pilot; not frozen Stage 5 evidence.",
                "Three retained repetitions quantify limited variability but do not establish a stable crossover.",
                "Theta-specific gates are protocol decisions, not universal Barnes-Hut error bounds.",
                "No sampled Direct reference was used; 16384 and larger lack full Direct-reference authorization.",
                "Simulation throughput is not graphical FPS.",
                "CUDA, RTX 4050, viewer, 16384, 32768, 50000, and 100000 execution were not performed.",
            ],
        }
        publication_core.write_json(staged / "metadata.json", metadata)
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "capture_large_n_l2_provenance.py"), "--evidence", str(staged), "--build-dir", str(executable.parent)],
            cwd=ROOT, check=True,
        )
        metadata = json.loads((staged / "metadata.json").read_text(encoding="utf-8"))
        metadata["publication_status"] = "complete"
        publication_core.write_json(staged / "metadata.json", metadata)
        (staged / "README.md").write_text(evidence_readme(summary), encoding="utf-8", newline="\n")
        write_run_manifest(staged, run_id)
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "verify_large_n_l2_evidence.py"), "--run", str(staged), "--expected-run-id", run_id],
            cwd=ROOT, check=True,
        )
        publish_run(staged, output, run_id)
    print(f"published immutable Large-N L2 run {run_id} under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
