#!/usr/bin/env python3
"""Strict read-only verifier for Large-N L3 performance-only evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import capture_large_n_l3_provenance as custody
import run_large_n_l1_pilot as publication_core


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "data" / "benchmarks" / "large_n_l3_pilot"
DEFAULT_PROTOCOL = ROOT / "tools" / "large_n_l3_protocol.json"
RUN_FILES = {
    "raw.csv", "summary.json", "metadata.json", "build_provenance.json",
    "source_manifest.sha256", "protocol.json", "README.md",
}
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
PHASE_MEMORY_FIELDS = (
    "generation_memory_bytes", "direct_reference_memory_bytes",
    "warmup_memory_bytes", "retained_repetition_memory_bytes",
    "characterization_memory_bytes", "serialization_memory_bytes",
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
    require(protocol.get("pilot_body_counts") == [16384], "L3 pilot size changed")
    require(protocol.get("distributions") == ["uniform", "clustered", "planar"], "L3 distributions changed")
    require(protocol.get("backends") == [{"name": "sequential", "threads": 1}, {"name": "openmp", "threads": 4}], "L3 backends changed")
    require(protocol.get("pilot_matrix") == [
        {"distribution": "uniform", "theta": 0.25},
        {"distribution": "clustered", "theta": 0.5},
        {"distribution": "planar", "theta": 0.8},
    ], "L3 pilot matrix changed")
    require(protocol.get("execution") == {
        "hard_subprocess_timeout_s": 70.0,
        "memory_budget_bytes": 536870912,
        "maximum_direct_interactions": 500000000,
        "maximum_direct_reference_bodies": 0,
        "per_configuration_wall_time_budget_s": 60.0,
        "retained_repetitions": 3,
        "steps": 1,
        "timestep_s": 10.0,
        "warmups": 1,
    }, "L3 execution policy changed")
    require(protocol.get("solver") == {
        "maximum_nodes": 1000000,
        "maximum_traversal_work": 1000000000,
        "memory_budget_bytes": 268435456,
        "name": "barnes-hut",
        "theta_values": [0.25, 0.5, 0.8],
    }, "L3 solver policy changed")
    require(protocol.get("reference_policy") == {
        "accuracy_status": "unavailable_performance_only",
        "conservation_status": "unavailable_performance_only",
        "direct_reference": "disabled; no complete or sampled Direct reference is executed",
        "reference_mode": "none",
    }, "L3 reference policy changed")
    require(protocol.get("progression_gate") == {
        "maximum_memory_budget_fraction": 0.9,
        "maximum_wall_budget_fraction": 0.8,
        "maximum_within_group_force_range_over_median": 0.3,
        "next_checkpoint": 32768,
        "requires_all_pilot_groups_completed": True,
        "requires_independent_review": True,
    }, "L3 progression gate changed")
    return protocol


def expected_configurations(protocol: dict) -> list[dict]:
    bodies = 16384
    base_seed = 0x4C33504552460000
    matrix = (("uniform", 0, 0.25), ("clustered", 1, 0.5), ("planar", 2, 0.8))
    result = []
    for distribution, distribution_index, theta in matrix:
        seed = base_seed + distribution_index * 0x100000 + bodies
        result.extend((
            {"distribution": distribution, "bodies": bodies, "seed": seed, "backend": "sequential", "threads": 1, "theta": theta},
            {"distribution": distribution, "bodies": bodies, "seed": seed, "backend": "openmp", "threads": 4, "theta": theta},
        ))
    return result


def exact_command(configuration: dict, protocol: dict, executable: Path, output: Path) -> list[str]:
    execution = protocol["execution"]
    solver = protocol["solver"]
    return [
        str(executable), "--output", str(output), "--bodies", str(configuration["bodies"]),
        "--seed", str(configuration["seed"]), "--distribution", configuration["distribution"],
        "--backend", configuration["backend"], "--solver", "barnes-hut",
        "--theta", str(configuration["theta"]), "--threads", str(configuration["threads"]),
        "--dt", str(execution["timestep_s"]), "--steps", str(execution["steps"]),
        "--warmups", str(execution["warmups"]), "--repetitions", str(execution["retained_repetitions"]),
        "--direct-reference", "false", "--max-direct-reference-bodies", "0",
        "--memory-budget-bytes", str(execution["memory_budget_bytes"]),
        "--max-direct-interactions", str(execution["maximum_direct_interactions"]),
        "--wall-time-budget-s", str(execution["per_configuration_wall_time_budget_s"]),
        "--maximum-nodes", str(solver["maximum_nodes"]),
        "--maximum-traversal-work", str(solver["maximum_traversal_work"]),
        "--solver-memory-budget-bytes", str(solver["memory_budget_bytes"]),
    ]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(line for line in stream if line.strip() and not line.startswith("#")))


def require_finite(row: dict[str, str], field: str, index: int, *, positive: bool = False) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"L3 row {index} has invalid {field}") from error
    require(math.isfinite(value), f"L3 row {index} has non-finite {field}")
    if positive:
        require(value > 0.0, f"L3 row {index} has non-positive {field}")
    return value


def reconstruct_summary(rows: list[dict[str, str]], protocol: dict) -> dict:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["distribution"], row["bodies"], row["theta"], row["backend"]), []).append(row)
    groups = []
    for key in sorted(grouped):
        group = grouped[key]
        medians = {field: statistics.median(float(row[field]) for row in group) for field in TIMING_FIELDS}
        force = [float(row["complete_force_s"]) for row in group]
        groups.append({
            "distribution": key[0], "bodies": int(key[1]), "theta": float(key[2]),
            "backend": key[3], "status": group[0]["status"], "rows": len(group),
            "retained_repetitions": len(group),
            "observed_worker_threads": int(group[0]["observed_worker_threads"]),
            "worker_count_provenance": group[0]["worker_count_provenance"],
            "reference_mode": group[0]["reference_mode"],
            "accuracy_status": group[0]["accuracy_status"],
            "conservation_status": group[0]["conservation_status"],
            "medians": medians,
            "force_range_over_median": (max(force) - min(force)) / medians["complete_force_s"],
            "resources": {field: int(group[0][field]) for field in RESOURCE_FIELDS},
        })
    comparisons = []
    for distribution, theta in (("uniform", 0.25), ("clustered", 0.5), ("planar", 0.8)):
        pair = [group for group in groups if group["distribution"] == distribution and math.isclose(group["theta"], theta)]
        require(len(pair) == 2, "L3 summary lacks a matched backend pair")
        by_backend = {group["backend"]: group for group in pair}
        comparisons.append({
            "distribution": distribution, "theta": theta,
            "complete_force_speedup_sequential_over_openmp": by_backend["sequential"]["medians"]["complete_force_s"] / by_backend["openmp"]["medians"]["complete_force_s"],
            "complete_timestep_speedup_sequential_over_openmp": by_backend["sequential"]["medians"]["complete_timestep_s"] / by_backend["openmp"]["medians"]["complete_timestep_s"],
        })
    max_wall = max(group["medians"]["total_experiment_s"] / 60.0 for group in groups)
    max_memory = max(group["resources"]["estimated_memory_bytes"] / 536870912 for group in groups)
    max_variability = max(group["force_range_over_median"] for group in groups)
    gate = protocol["progression_gate"]
    eligible = (
        len(groups) == 6 and all(group["status"] == "completed" for group in groups)
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
        "matched_backend_comparisons": comparisons,
        "checkpoint_decision": {
            "all_expected_groups_completed": len(groups) == 6 and all(group["status"] == "completed" for group in groups),
            "maximum_wall_budget_fraction": max_wall,
            "maximum_memory_budget_fraction": max_memory,
            "maximum_force_range_over_median": max_variability,
            "eligible_to_propose_32768": eligible,
            "requires_separate_admission_and_independent_review": True,
            "reason": "all predeclared performance gates passed; 32768 may only be proposed for separate review" if eligible else "one or more performance gates failed; do not propose automatic progression",
        },
    }


def read_run_manifest(run_directory: Path) -> dict:
    path = run_directory / "run_manifest.json"
    require(path.is_file(), f"missing L3 run manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "large-n-l3-run-v1", "wrong L3 run manifest schema")
    require(manifest.get("status") == "complete", "incomplete L3 run manifest")
    require(publication_core.valid_run_id(manifest.get("run_id")), "unsafe L3 run ID")
    require(set(manifest.get("files", {})) == RUN_FILES, "wrong L3 run artifact set")
    for name, expected in manifest["files"].items():
        require(publication_core.sha256(run_directory / name) == expected, f"L3 artifact hash mismatch: {name}")
    return manifest


def descriptor(run_directory: Path) -> dict:
    manifest = read_run_manifest(run_directory)
    return {"run_id": manifest["run_id"], "run_manifest_sha256": publication_core.sha256(run_directory / "run_manifest.json"), "files": manifest["files"]}


def inspect_publication(evidence: Path) -> dict:
    pointer_path = evidence / "publication.json"
    require(pointer_path.is_file(), "missing L3 publication pointer")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    require(pointer.get("schema") == "large-n-l3-pointer-v1", "wrong L3 pointer schema")
    require(pointer.get("status") == "complete", "incomplete L3 pointer")
    require(isinstance(pointer.get("history"), list), "invalid L3 pointer history")
    seen = set()
    for item in [pointer.get("current"), *pointer["history"]]:
        require(isinstance(item, dict) and publication_core.valid_run_id(item.get("run_id")), "unsafe L3 descriptor")
        require(item["run_id"] not in seen, "duplicate L3 run ID")
        seen.add(item["run_id"])
        require(item == descriptor(evidence / "runs" / item["run_id"]), "L3 descriptor mismatch")
    runs = evidence / "runs"
    referenced = seen
    incomplete = sorted(path.name for path in runs.iterdir() if path.is_dir() and path.name.startswith(".install-"))
    orphaned = sorted(path.name for path in runs.iterdir() if path.is_dir() and not path.name.startswith(".install-") and path.name not in referenced)
    require(not incomplete, "incomplete L3 installation detected")
    require(not orphaned, "unreferenced L3 run detected")
    return {"pointer": pointer, "current_run": runs / pointer["current"]["run_id"]}


def verify_source_manifest(run_directory: Path, root: Path) -> None:
    entries = {}
    for line in (run_directory / "source_manifest.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(" *", 1)
        require(relative not in entries, f"duplicate L3 source input: {relative}")
        entries[relative] = expected
    require(set(entries) == set(custody.SOURCES), "L3 source manifest input set is not exact")
    for relative, expected in entries.items():
        source = root / relative
        require(source.is_file(), f"L3 source input missing: {relative}")
        require(publication_core.sha256(source) == expected, f"L3 source hash mismatch: {relative}")


def verify_run(run_directory: Path, root: Path, expected_run_id: str | None = None) -> dict:
    run_directory = run_directory.resolve()
    manifest = read_run_manifest(run_directory)
    if expected_run_id is not None:
        require(manifest["run_id"] == expected_run_id, "unexpected L3 run ID")
    protocol = load_protocol(run_directory / "protocol.json")
    require((run_directory / "protocol.json").read_bytes() == DEFAULT_PROTOCOL.read_bytes(), "retained L3 protocol differs from repository protocol")
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    require(metadata.get("schema") == "large-n-l3-metadata-v1", "wrong L3 metadata schema")
    require(metadata.get("publication_status") == "complete", "L3 metadata is not complete")
    require(metadata.get("run_id") == manifest["run_id"], "L3 metadata/run ID mismatch")
    require(metadata.get("stage5_evidence") is False, "L3 mislabeled as Stage 5 evidence")
    require(metadata.get("classification") == "exploratory_performance_only_checkpoint", "wrong L3 metadata classification")
    expected_matrix = {
        "body_counts": [16384], "pilot_matrix": protocol["pilot_matrix"],
        "backends": protocol["backends"], **protocol["execution"],
        "configuration_groups": 6, "expected_completed_rows": 18,
        "seed_formula": protocol["seed_formula"],
    }
    require(metadata.get("matrix") == expected_matrix, "L3 metadata matrix differs from protocol")
    outputs = metadata.get("outputs", {})
    for name, key in (
        ("raw.csv", "raw_csv_sha256"), ("summary.json", "summary_json_sha256"),
        ("protocol.json", "protocol_sha256"),
        ("build_provenance.json", "build_provenance_sha256"),
        ("source_manifest.sha256", "source_manifest_sha256"),
    ):
        require(publication_core.sha256(run_directory / name) == outputs.get(key), f"L3 metadata hash mismatch: {name}")
    require(outputs.get("raw_rows") == 18 and outputs.get("status_counts") == {"completed": 18, "skipped": 0}, "wrong L3 output counts")
    verify_source_manifest(run_directory, root)
    provenance = json.loads((run_directory / "build_provenance.json").read_text(encoding="utf-8"))
    require(provenance.get("schema") == "large-n-l3-build-provenance-v1", "wrong L3 provenance schema")
    require(provenance.get("source_manifest_sha256") == outputs["source_manifest_sha256"], "L3 provenance/source mismatch")
    require(provenance.get("protocol_sha256") == outputs["protocol_sha256"], "L3 provenance/protocol mismatch")
    cache = provenance.get("cmake_cache", {})
    require(cache.get("CMAKE_BUILD_TYPE") == "Release" and cache.get("NBODY_ENABLE_OPENMP") == "ON" and cache.get("NBODY_ENABLE_CUDA") == "OFF" and cache.get("NBODY_ENABLE_MPI") == "OFF", "L3 build is not canonical OpenMP Release")
    require(provenance.get("build_directory") == "build/openmp-release", "wrong L3 build directory")
    executable_record = provenance.get("executable", {})
    require(executable_record.get("path") == "build/openmp-release/nbody_large_n_lab.exe", "wrong L3 executable path")
    executable = (root / executable_record["path"]).resolve()
    if executable.is_file():
        require(publication_core.sha256(executable) == executable_record.get("sha256"), "measured L3 executable hash changed")

    commands = metadata.get("commands", {}).get("configuration_argv")
    configurations = expected_configurations(protocol)
    require(metadata.get("commands", {}).get("runner") == "python -B scripts/run_large_n_l3_pilot.py", "wrong L3 runner command")
    require(isinstance(commands, list) and len(commands) == 6, "wrong L3 command count")
    scratch = (root / "build" / "scratch").resolve()
    common_parent = None
    for index, (configuration, command) in enumerate(zip(configurations, commands)):
        require(isinstance(command, list) and len(command) >= 3, f"malformed L3 command {index}")
        output = Path(command[2])
        require(output.is_absolute(), f"L3 command {index} output is not absolute")
        try:
            output.resolve().relative_to(scratch)
        except ValueError as error:
            raise RuntimeError(f"L3 command {index} output escaped build/scratch") from error
        require(output.name == f"configuration_{index:03d}.csv", f"wrong L3 command output identity {index}")
        common_parent = output.parent if common_parent is None else common_parent
        require(output.parent == common_parent, f"L3 command {index} uses a different work directory")
        require(command == exact_command(configuration, protocol, executable, output), f"L3 command arguments differ at configuration {index}")

    preamble = (run_directory / "raw.csv").read_text(encoding="utf-8").splitlines()[:4]
    require(preamble == [
        "# study=large_n_laboratory_l3_performance_only_scaling",
        "# stage5_evidence=false",
        "# phase=performance_only_checkpoint_16384",
        "# graphical_fps=false",
    ], "wrong L3 raw preamble")
    rows = read_rows(run_directory / "raw.csv")
    require(len(rows) == 18, "L3 pilot must contain exactly 18 rows")
    expected = [(configuration, repetition) for configuration in configurations for repetition in range(3)]
    for index, (row, (configuration, repetition)) in enumerate(zip(rows, expected)):
        require(row["status"] == "completed" and row["message"] == "completed", f"L3 row {index} did not complete")
        require(row["distribution"] == configuration["distribution"] and row["bodies"] == "16384" and row["seed"] == str(configuration["seed"]), f"L3 row {index} identity differs")
        require(row["backend"] == configuration["backend"] and row["solver"] == "barnes-hut" and row["theta"] == csv_float(configuration["theta"]), f"L3 row {index} solver identity differs")
        require(row["requested_threads"] == str(configuration["threads"]) and row["effective_threads"] == str(configuration["threads"]), f"L3 row {index} worker admission differs")
        require(row["repetition"] == str(repetition) and row["repetitions"] == "3" and row["warmups"] == "1" and row["steps"] == "1" and row["timestep_s"] == "10", f"L3 row {index} execution contract differs")
        expected_worker = ("1", "serial") if configuration["backend"] == "sequential" else ("4", "observed_in_timed_region")
        require((row["observed_worker_threads"], row["worker_count_provenance"]) == expected_worker, f"L3 row {index} worker provenance differs")
        require(row["direct_reference_available"] == "0" and row["direct_reference_reason"] == "direct reference disabled by configuration", f"L3 row {index} Direct-reference status differs")
        require(row["reference_mode"] == "none" and row["reference_sample_size"] == "0", f"L3 row {index} reference label differs")
        require(row["accuracy_status"] == "unavailable_performance_only" and row["accuracy_reason"] == "no Direct reference was executed", f"L3 row {index} accuracy label differs")
        require(row["conservation_status"] == "unavailable_performance_only" and row["conservation_reason"] == "O(N^2) conservation diagnostics were disabled", f"L3 row {index} conservation label differs")
        require(all(row[field] == "0" for field in OMITTED_WORK_FIELDS), f"L3 row {index} contains Direct/diagnostic timing")
        require(all(row[field] == "" for field in SCIENTIFIC_FIELDS), f"L3 row {index} fabricates unavailable science")
        for field in TIMING_FIELDS:
            require_finite(row, field, index, positive=True)
        for field in RESOURCE_FIELDS[:-1]:
            require(int(row[field]) > 0, f"L3 row {index} has non-positive resource {field}")
        require(row["estimated_direct_interactions"] == "0", f"L3 row {index} charged O(N^2) work")
        require(int(row["estimated_memory_bytes"]) == max(int(row[field]) for field in PHASE_MEMORY_FIELDS), f"L3 row {index} memory high-water is inconsistent")
        require(int(row["estimated_memory_bytes"]) <= 536870912, f"L3 row {index} exceeds memory admission")
        require(float(row["total_experiment_s"]) <= 60.0, f"L3 row {index} exceeds wall admission")
        require(math.isclose(float(row["steps_per_second"]), 1.0 / float(row["complete_timestep_s"]), rel_tol=1e-15), f"L3 row {index} throughput is wrong")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["distribution"], row["theta"], row["backend"]), []).append(row)
    require(len(groups) == 6 and all(len(group) == 3 for group in groups.values()), "L3 grouping differs from the six-group protocol")
    for key, group in groups.items():
        for field in (*RESOURCE_FIELDS, "accuracy_status", "conservation_status", "reference_mode"):
            require(len({row[field] for row in group}) == 1, f"L3 non-timing field changed across repetitions: {key} {field}")
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    require(summary == reconstruct_summary(rows, protocol), "L3 summary does not reconstruct exactly")
    require(metadata.get("checkpoint_decision") == summary["checkpoint_decision"], "L3 checkpoint decision mismatch")
    return metadata


def verify_evidence(evidence: Path = DEFAULT_EVIDENCE, root: Path = ROOT) -> dict:
    inspection = inspect_publication(evidence.resolve())
    return verify_run(inspection["current_run"], root, inspection["pointer"]["current"]["run_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", nargs="?", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--expected-run-id")
    args = parser.parse_args()
    if args.run is not None:
        metadata = verify_run(args.run, ROOT, args.expected_run_id)
    else:
        require(args.expected_run_id is None, "--expected-run-id requires --run")
        metadata = verify_evidence(args.evidence, ROOT)
    print(f"Large-N L3 evidence verified: run {metadata['run_id']}, {metadata['outputs']['raw_rows']} rows, raw SHA-256 {metadata['outputs']['raw_csv_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
