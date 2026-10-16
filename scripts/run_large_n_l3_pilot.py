#!/usr/bin/env python3
"""Run and transactionally publish the bounded Large-N L3 16384 checkpoint."""

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
from datetime import datetime, timezone
from pathlib import Path

import run_large_n_l1_pilot as publication_core


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "benchmarks" / "large_n_l3_pilot"
DEFAULT_PROTOCOL = ROOT / "tools" / "large_n_l3_protocol.json"
RUN_FILES = (
    "raw.csv", "summary.json", "metadata.json", "build_provenance.json",
    "source_manifest.sha256", "protocol.json", "README.md",
)
TIMING_FIELDS = (
    "generation_s", "complete_force_s", "complete_timestep_s",
    "steps_per_second", "total_experiment_s", "tree_build_s",
    "force_traversal_s",
)
OMITTED_WORK_FIELDS = (
    "initial_direct_force_s", "direct_trajectory_s", "diagnostics_s",
)
SCIENTIFIC_FIELDS = (
    "max_abs_acceleration_error_m_s2", "rms_abs_acceleration_error_m_s2",
    "p50_relative_error", "p95_relative_error", "p99_relative_error",
    "max_relative_error", "relative_floor_m_s2",
    "cancellation_sensitive_count", "max_position_separation_m",
    "rms_position_separation_m", "max_velocity_separation_m_s",
    "rms_velocity_separation_m_s", "candidate_relative_energy_drift",
    "direct_relative_energy_drift",
    "candidate_relative_angular_momentum_drift",
    "direct_relative_angular_momentum_drift",
)
RESOURCE_FIELDS = (
    "solver_memory_bytes", "node_count", "visited_nodes",
    "approximated_nodes", "direct_interactions", "estimated_memory_bytes",
    "generation_memory_bytes", "direct_reference_memory_bytes",
    "warmup_memory_bytes", "retained_repetition_memory_bytes",
    "characterization_memory_bytes", "serialization_memory_bytes",
    "estimated_direct_interactions",
)
DERIVED_FIELDS = (
    "protocol_phase", "reference_mode", "reference_sample_size",
    "accuracy_status", "accuracy_reason", "conservation_status",
    "conservation_reason", "steps_per_second",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def csv_float(value: float) -> str:
    return format(float(value), ".17g")


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == "large-n-l3-protocol-v1", "wrong L3 protocol schema")
    require(protocol.get("version") == "3.0.0-pilot-16384", "wrong L3 protocol version")
    require(protocol.get("study") == "large_n_laboratory_l3_performance_only_scaling", "wrong L3 study")
    require(protocol.get("classification") == "exploratory_performance_only_checkpoint", "wrong L3 classification")
    require(protocol.get("candidate_body_counts") == [16384, 32768, 50000], "L3 candidate sizes changed")
    require(protocol.get("pilot_body_counts") == [16384], "L3 pilot must contain only N=16384")
    require(protocol.get("distributions") == ["uniform", "clustered", "planar"], "L3 distributions changed")
    require(
        protocol.get("backends")
        == [{"name": "sequential", "threads": 1}, {"name": "openmp", "threads": 4}],
        "L3 backend matrix changed",
    )
    require(
        protocol.get("pilot_matrix")
        == [
            {"distribution": "uniform", "theta": 0.25},
            {"distribution": "clustered", "theta": 0.5},
            {"distribution": "planar", "theta": 0.8},
        ],
        "L3 representative pilot matrix changed",
    )
    require(
        protocol.get("execution")
        == {
            "hard_subprocess_timeout_s": 70.0,
            "memory_budget_bytes": 536870912,
            "maximum_direct_interactions": 500000000,
            "maximum_direct_reference_bodies": 0,
            "per_configuration_wall_time_budget_s": 60.0,
            "retained_repetitions": 3,
            "steps": 1,
            "timestep_s": 10.0,
            "warmups": 1,
        },
        "L3 execution/resource policy changed",
    )
    require(
        protocol.get("solver")
        == {
            "maximum_nodes": 1000000,
            "maximum_traversal_work": 1000000000,
            "memory_budget_bytes": 268435456,
            "name": "barnes-hut",
            "theta_values": [0.25, 0.5, 0.8],
        },
        "L3 Barnes-Hut policy changed",
    )
    require(
        protocol.get("reference_policy")
        == {
            "accuracy_status": "unavailable_performance_only",
            "conservation_status": "unavailable_performance_only",
            "direct_reference": "disabled; no complete or sampled Direct reference is executed",
            "reference_mode": "none",
        },
        "L3 missing-reference policy changed",
    )
    gate = protocol.get("progression_gate")
    require(
        gate
        == {
            "maximum_memory_budget_fraction": 0.9,
            "maximum_wall_budget_fraction": 0.8,
            "maximum_within_group_force_range_over_median": 0.3,
            "next_checkpoint": 32768,
            "requires_all_pilot_groups_completed": True,
            "requires_independent_review": True,
        },
        "L3 progression gate changed",
    )
    return protocol


def configurations(protocol: dict) -> list[dict]:
    bodies = protocol["pilot_body_counts"][0]
    base_seed = 0x4C33504552460000
    distribution_indices = {name: index for index, name in enumerate(protocol["distributions"])}
    result = []
    for item in protocol["pilot_matrix"]:
        distribution = item["distribution"]
        seed = base_seed + distribution_indices[distribution] * 0x100000 + bodies
        for backend in protocol["backends"]:
            result.append(
                {
                    "distribution": distribution,
                    "bodies": bodies,
                    "seed": seed,
                    "backend": backend["name"],
                    "threads": backend["threads"],
                    "solver": "barnes-hut",
                    "theta": item["theta"],
                    "request_direct_reference": False,
                }
            )
    return result


def exact_command(configuration: dict, protocol: dict, executable: Path, output: Path) -> list[str]:
    execution = protocol["execution"]
    solver = protocol["solver"]
    return [
        str(executable), "--output", str(output),
        "--bodies", str(configuration["bodies"]),
        "--seed", str(configuration["seed"]),
        "--distribution", configuration["distribution"],
        "--backend", configuration["backend"],
        "--solver", "barnes-hut",
        "--theta", str(configuration["theta"]),
        "--threads", str(configuration["threads"]),
        "--dt", str(execution["timestep_s"]),
        "--steps", str(execution["steps"]),
        "--warmups", str(execution["warmups"]),
        "--repetitions", str(execution["retained_repetitions"]),
        "--direct-reference", "false",
        "--max-direct-reference-bodies", str(execution["maximum_direct_reference_bodies"]),
        "--memory-budget-bytes", str(execution["memory_budget_bytes"]),
        "--max-direct-interactions", str(execution["maximum_direct_interactions"]),
        "--wall-time-budget-s", str(execution["per_configuration_wall_time_budget_s"]),
        "--maximum-nodes", str(solver["maximum_nodes"]),
        "--maximum-traversal-work", str(solver["maximum_traversal_work"]),
        "--solver-memory-budget-bytes", str(solver["memory_budget_bytes"]),
    ]


def normalize_performance_only_row(row: dict[str, str]) -> None:
    """Validate the legacy no-reference placeholders, then publish absence."""
    require(row["direct_reference_available"] == "0", "L3 unexpectedly executed a Direct reference")
    require(row["direct_reference_reason"] == "direct reference disabled by configuration", "L3 Direct-reference reason changed")
    require(row["estimated_direct_interactions"] == "0", "L3 charged hidden O(N^2) work")
    require(all(row[field] == "0" for field in OMITTED_WORK_FIELDS), "L3 performed Direct or conservation work")
    require(all(row[field] == "0" for field in SCIENTIFIC_FIELDS), "L3 legacy no-reference placeholders changed")
    for field in SCIENTIFIC_FIELDS:
        row[field] = ""


def reconstruct_summary(rows: list[dict[str, str]], protocol: dict) -> dict:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["distribution"], row["bodies"], row["theta"], row["backend"])
        grouped.setdefault(key, []).append(row)
    groups = []
    for key in sorted(grouped):
        group = grouped[key]
        medians = {field: statistics.median(float(row[field]) for row in group) for field in TIMING_FIELDS}
        force = [float(row["complete_force_s"]) for row in group]
        groups.append(
            {
                "distribution": key[0],
                "bodies": int(key[1]),
                "theta": float(key[2]),
                "backend": key[3],
                "status": group[0]["status"],
                "rows": len(group),
                "retained_repetitions": len(group),
                "observed_worker_threads": int(group[0]["observed_worker_threads"]),
                "worker_count_provenance": group[0]["worker_count_provenance"],
                "reference_mode": group[0]["reference_mode"],
                "accuracy_status": group[0]["accuracy_status"],
                "conservation_status": group[0]["conservation_status"],
                "medians": medians,
                "force_range_over_median": (max(force) - min(force)) / medians["complete_force_s"],
                "resources": {field: int(group[0][field]) for field in RESOURCE_FIELDS},
            }
        )
    matched = []
    for item in protocol["pilot_matrix"]:
        pair = [group for group in groups if group["distribution"] == item["distribution"] and math.isclose(group["theta"], item["theta"])]
        require(len(pair) == 2, "L3 summary lacks a matched backend pair")
        by_backend = {group["backend"]: group for group in pair}
        matched.append(
            {
                "distribution": item["distribution"],
                "theta": item["theta"],
                "complete_force_speedup_sequential_over_openmp": by_backend["sequential"]["medians"]["complete_force_s"] / by_backend["openmp"]["medians"]["complete_force_s"],
                "complete_timestep_speedup_sequential_over_openmp": by_backend["sequential"]["medians"]["complete_timestep_s"] / by_backend["openmp"]["medians"]["complete_timestep_s"],
            }
        )
    gate = protocol["progression_gate"]
    max_wall = max(group["medians"]["total_experiment_s"] / protocol["execution"]["per_configuration_wall_time_budget_s"] for group in groups)
    max_memory = max(group["resources"]["estimated_memory_bytes"] / protocol["execution"]["memory_budget_bytes"] for group in groups)
    max_variability = max(group["force_range_over_median"] for group in groups)
    eligible = (
        len(groups) == 6
        and all(group["status"] == "completed" for group in groups)
        and max_wall <= gate["maximum_wall_budget_fraction"]
        and max_memory <= gate["maximum_memory_budget_fraction"]
        and max_variability <= gate["maximum_within_group_force_range_over_median"]
    )
    return {
        "schema": "large-n-l3-summary-v1",
        "definitions": {
            "complete_force_s": "one complete Barnes-Hut force call including tree construction",
            "complete_timestep_s": "one Velocity Verlet step containing two Barnes-Hut force evaluations",
            "steps_per_second": "configured simulation steps divided by complete timestep wall time; not rendering FPS",
            "reference_mode_none": "no Direct force, trajectory, accuracy, or conservation characterization was executed",
        },
        "groups": groups,
        "matched_backend_comparisons": matched,
        "checkpoint_decision": {
            "all_expected_groups_completed": len(groups) == 6 and all(group["status"] == "completed" for group in groups),
            "maximum_wall_budget_fraction": max_wall,
            "maximum_memory_budget_fraction": max_memory,
            "maximum_force_range_over_median": max_variability,
            "eligible_to_propose_32768": eligible,
            "requires_separate_admission_and_independent_review": True,
            "reason": (
                "all predeclared performance gates passed; 32768 may only be proposed for separate review"
                if eligible else
                "one or more performance gates failed; do not propose automatic progression"
            ),
        },
    }


def write_run_manifest(run_directory: Path, run_id: str) -> dict:
    require(publication_core.valid_run_id(run_id), "unsafe L3 run ID")
    manifest = {
        "schema": "large-n-l3-run-v1",
        "status": "complete",
        "run_id": run_id,
        "files": {name: publication_core.sha256(run_directory / name) for name in RUN_FILES},
    }
    publication_core.write_json(run_directory / "run_manifest.json", manifest)
    return manifest


def read_run_manifest(run_directory: Path) -> dict:
    manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == "large-n-l3-run-v1", "wrong L3 run manifest schema")
    require(manifest.get("status") == "complete", "incomplete L3 run manifest")
    require(publication_core.valid_run_id(manifest.get("run_id")), "unsafe L3 run ID")
    require(set(manifest.get("files", {})) == set(RUN_FILES), "wrong L3 artifact set")
    for name, expected in manifest["files"].items():
        require(publication_core.sha256(run_directory / name) == expected, f"L3 artifact hash mismatch: {name}")
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
    require(pointer.get("schema") == "large-n-l3-pointer-v1", "wrong L3 pointer schema")
    require(pointer.get("status") == "complete", "incomplete L3 pointer")
    require(isinstance(pointer.get("history"), list), "invalid L3 pointer history")
    seen = set()
    for item in [pointer.get("current"), *pointer["history"]]:
        require(isinstance(item, dict) and publication_core.valid_run_id(item.get("run_id")), "unsafe L3 descriptor")
        require(item["run_id"] not in seen, "duplicate L3 run ID")
        seen.add(item["run_id"])
        require(item == descriptor(evidence_root / "runs" / item["run_id"]), "L3 descriptor mismatch")
    return pointer


def publish_run(staged: Path, evidence_root: Path, run_id: str) -> dict:
    with publication_core.PublicationLock(evidence_root):
        pointer = read_pointer(evidence_root)
        runs = evidence_root / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        incomplete = sorted(path.name for path in runs.iterdir() if path.is_dir() and path.name.startswith(".install-"))
        referenced = set() if pointer is None else {pointer["current"]["run_id"], *(item["run_id"] for item in pointer["history"])}
        orphans = sorted(path.name for path in runs.iterdir() if path.is_dir() and not path.name.startswith(".install-") and path.name not in referenced)
        require(not incomplete, "incomplete L3 run installation requires inspection")
        require(not orphans, "unreferenced L3 run requires inspection")
        require(read_run_manifest(staged)["run_id"] == run_id, "staged L3 run ID mismatch")
        destination = runs / run_id
        require(not destination.exists(), "L3 run ID already exists")
        os.replace(staged, destination)
        current = descriptor(destination)
        history = [] if pointer is None else [pointer["current"], *pointer["history"]]
        next_pointer = {
            "schema": "large-n-l3-pointer-v1",
            "status": "complete",
            "current": current,
            "history": history,
        }
        publication_core.atomic_replace_json(evidence_root / "publication.json", next_pointer, None)
        require(read_pointer(evidence_root) == next_pointer, "L3 publication confirmation failed")
        return next_pointer


def evidence_readme(summary: dict) -> str:
    decision = summary["checkpoint_decision"]
    return f"""# Large-N Laboratory Phase L3 — 16384 performance-only checkpoint

This immutable run is exploratory CPU performance evidence. It is not frozen
Stage 5 evidence and does not override the failed L2 full-characterization
continuation gate. No complete or sampled Direct reference and no O(N^2)
conservation diagnostic was executed. Accuracy and conservation are explicitly
unavailable rather than represented by numeric zero.

The run contains six representative Barnes--Hut groups: matched Sequential and
OpenMP-4 executions for uniform/theta 0.25, clustered/theta 0.5, and
planar/theta 0.8. Each group has one warm-up and three retained repetitions of
one 10-second Velocity Verlet step. Simulation throughput is not rendering FPS.

Eligibility to propose a separately admitted and independently reviewed 32768
checkpoint is `{decision['eligible_to_propose_32768']}`. This run never
authorizes automatic progression. No 32768, 50000, 100000, CUDA, RTX 4050, or
graphical workload is part of this publication.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=ROOT / "build" / "openmp-release" / "nbody_large_n_lab.exe")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    executable = args.executable.resolve()
    output = args.output.resolve()
    require(executable.is_file(), f"large-N executable not found: {executable}")
    execution = protocol["execution"]
    scratch_root = ROOT / "build" / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    run_id = publication_core.new_run_id()
    with tempfile.TemporaryDirectory(prefix="large-n-l3-run-", dir=scratch_root) as temporary:
        temporary_root = Path(temporary)
        staged = temporary_root / "candidate"
        work = temporary_root / "work"
        staged.mkdir()
        work.mkdir()
        all_rows: list[dict[str, str]] = []
        input_fields: list[str] | None = None
        command_log: list[list[str]] = []
        for index, configuration in enumerate(configurations(protocol)):
            path = work / f"configuration_{index:03d}.csv"
            command = exact_command(configuration, protocol, executable, path)
            command_log.append(command)
            try:
                completed = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True,
                    timeout=execution["hard_subprocess_timeout_s"], check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"L3 configuration {index} exceeded its hard timeout; pilot stopped") from error
            require(completed.returncode == 0, f"L3 configuration {index} failed; pilot stopped: {completed.stdout}\n{completed.stderr}")
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(line for line in stream if line.strip() and not line.startswith("#"))
                rows = list(reader)
                fields = list(reader.fieldnames or [])
            if input_fields is None:
                input_fields = fields
            require(fields == input_fields, "inconsistent L3 CSV fields")
            require(len(rows) == execution["retained_repetitions"], f"L3 configuration {index} returned wrong row count")
            require(all(row["status"] == "completed" for row in rows), f"L3 configuration {index} did not complete")
            for row in rows:
                require(row["solver"] == "barnes-hut", "L3 attempted a non-Barnes-Hut solver")
                normalize_performance_only_row(row)
                row.update(
                    {
                        "protocol_phase": "performance_only_checkpoint_16384",
                        "reference_mode": "none",
                        "reference_sample_size": "0",
                        "accuracy_status": "unavailable_performance_only",
                        "accuracy_reason": "no Direct reference was executed",
                        "conservation_status": "unavailable_performance_only",
                        "conservation_reason": "O(N^2) conservation diagnostics were disabled",
                        "steps_per_second": format(int(row["steps"]) / float(row["complete_timestep_s"]), ".17g"),
                    }
                )
                all_rows.append(row)
        require(input_fields is not None, "L3 pilot produced no rows")
        raw = staged / "raw.csv"
        with raw.open("w", encoding="utf-8", newline="") as stream:
            stream.write("# study=large_n_laboratory_l3_performance_only_scaling\n# stage5_evidence=false\n# phase=performance_only_checkpoint_16384\n# graphical_fps=false\n")
            writer = csv.DictWriter(stream, fieldnames=[*input_fields, *DERIVED_FIELDS], lineterminator="\n")
            writer.writeheader()
            writer.writerows(all_rows)
        summary = reconstruct_summary(all_rows, protocol)
        publication_core.write_json(staged / "summary.json", summary)
        (staged / "protocol.json").write_bytes(protocol_path.read_bytes())
        metadata = {
            "schema": "large-n-l3-metadata-v1",
            "study": protocol["study"],
            "protocol_version": protocol["version"],
            "publication_status": "staging",
            "run_id": run_id,
            "stage5_evidence": False,
            "classification": protocol["classification"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_state": "uncommitted Phase L3 candidate; source manifest binds exact bytes",
            "platform": {
                "system": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
                "logical_processors": os.cpu_count(),
            },
            "matrix": {
                "body_counts": protocol["pilot_body_counts"],
                "pilot_matrix": protocol["pilot_matrix"],
                "backends": protocol["backends"],
                **execution,
                "configuration_groups": 6,
                "expected_completed_rows": 18,
                "seed_formula": protocol["seed_formula"],
            },
            "commands": {"runner": "python -B scripts/run_large_n_l3_pilot.py", "configuration_argv": command_log},
            "outputs": {
                "raw_csv": "raw.csv",
                "raw_csv_sha256": publication_core.sha256(raw),
                "raw_rows": len(all_rows),
                "status_counts": {"completed": len(all_rows), "skipped": 0},
                "summary_json": "summary.json",
                "summary_json_sha256": publication_core.sha256(staged / "summary.json"),
                "protocol": "protocol.json",
                "protocol_sha256": publication_core.sha256(staged / "protocol.json"),
            },
            "checkpoint_decision": summary["checkpoint_decision"],
            "limitations": [
                "Performance-only single-machine CPU checkpoint; not frozen Stage 5 evidence.",
                "No complete or sampled Direct reference and no O(N^2) conservation diagnostics were executed.",
                "No scientific accuracy or conservation result is available at N=16384.",
                "Three retained repetitions provide limited variability evidence only.",
                "The representative six-group pilot is not the future full L3 matrix.",
                "Simulation throughput is not graphical FPS.",
                "No 32768, 50000, 100000, CUDA, RTX 4050, or viewer workload was executed.",
            ],
        }
        publication_core.write_json(staged / "metadata.json", metadata)
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "capture_large_n_l3_provenance.py"), "--evidence", str(staged), "--build-dir", str(executable.parent)],
            cwd=ROOT, check=True,
        )
        metadata = json.loads((staged / "metadata.json").read_text(encoding="utf-8"))
        metadata["publication_status"] = "complete"
        publication_core.write_json(staged / "metadata.json", metadata)
        (staged / "README.md").write_text(evidence_readme(summary), encoding="utf-8", newline="\n")
        write_run_manifest(staged, run_id)
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "verify_large_n_l3_evidence.py"), "--run", str(staged), "--expected-run-id", run_id],
            cwd=ROOT, check=True,
        )
        publish_run(staged, output, run_id)
    print(f"published immutable Large-N L3 run {run_id} under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
