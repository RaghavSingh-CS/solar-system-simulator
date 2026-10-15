#!/usr/bin/env python3
"""Strict, read-only verification for versioned Large-N L2 pilot evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import capture_large_n_l2_provenance as custody
import run_large_n_l1_pilot as publication_core
import run_large_n_l2_pilot as l2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "data" / "benchmarks" / "large_n_l2_pilot"

SKIP_MESSAGE = "direct/reference interaction estimate exceeds the configured budget"
DIRECT_REFERENCE_DISABLED = "direct reference disabled by configuration"
COMPLETED_MESSAGE = "completed"
PHASE = "exploratory_pilot_8192"
MEASURED_RESOURCE_FIELDS = (
    "solver_memory_bytes",
    "node_count",
    "visited_nodes",
    "approximated_nodes",
    "direct_interactions",
)
ADMISSION_FIELDS = (
    "estimated_memory_bytes",
    "generation_memory_bytes",
    "direct_reference_memory_bytes",
    "warmup_memory_bytes",
    "retained_repetition_memory_bytes",
    "characterization_memory_bytes",
    "serialization_memory_bytes",
    "estimated_direct_interactions",
)
# These x64 ABI values are independently transcribed from the checked-in
# conservative phase-accounting equations for the versioned 8192-body pilot.
# They intentionally make source/layout changes fail closed until re-audited.
EXPECTED_ADMISSION = {
    "direct": {
        "estimated_memory_bytes": 3_736_616,
        "generation_memory_bytes": 2_228_864,
        "direct_reference_memory_bytes": 983_720,
        "warmup_memory_bytes": 2_556_840,
        "retained_repetition_memory_bytes": 3_736_616,
        "characterization_memory_bytes": 983_720,
        "serialization_memory_bytes": 21_096,
        "estimated_direct_interactions": 805_208_064,
    },
    "barnes-hut": {
        "estimated_memory_bytes": 273_351_848,
        "generation_memory_bytes": 2_228_864,
        "direct_reference_memory_bytes": 2_556_840,
        "warmup_memory_bytes": 272_172_072,
        "retained_repetition_memory_bytes": 273_351_848,
        "characterization_memory_bytes": 3_408_872,
        "serialization_memory_bytes": 21_096,
        "estimated_direct_interactions": 301_953_024,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(line for line in stream if line.strip() and not line.startswith("#")))


def exact_matrix(protocol: dict) -> dict:
    execution = protocol["execution"]
    return {
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
    }


def expected_rows(protocol: dict) -> list[tuple[dict, int, str]]:
    result: list[tuple[dict, int, str]] = []
    for configuration in l2.configurations(protocol):
        status = "skipped" if configuration["solver"] == "direct" else "completed"
        repetitions = (0,) if status == "skipped" else range(protocol["execution"]["retained_repetitions"])
        result.extend((configuration, repetition, status) for repetition in repetitions)
    return result


def canonical_row_identity(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["distribution"], row["bodies"], row["seed"], row["backend"],
        row["solver"], row["theta"], row["requested_threads"],
        row["effective_threads"], row["repetition"], row["status"],
    )


def expected_row_identity(configuration: dict, repetition: int, status: str) -> tuple[str, ...]:
    return (
        configuration["distribution"], str(configuration["bodies"]), str(configuration["seed"]),
        configuration["backend"], configuration["solver"], l2.csv_float(configuration["theta"]),
        str(configuration["threads"]), str(configuration["threads"]), str(repetition), status,
    )


def exact_command_arguments(
    configuration: dict, protocol: dict, executable: Path, output: Path
) -> list[str]:
    execution = protocol["execution"]
    return [
        str(executable), "--output", str(output),
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


def verify_commands(metadata: dict, protocol: dict, executable: Path, root: Path) -> None:
    commands = metadata.get("commands")
    require(isinstance(commands, dict), "missing L2 command custody")
    require(commands.get("runner") == "python -B scripts/run_large_n_l2_pilot.py", "wrong L2 runner command")
    recorded = commands.get("configuration_argv")
    configurations = l2.configurations(protocol)
    require(isinstance(recorded, list) and len(recorded) == len(configurations), "wrong L2 command count")
    common_work_parent: Path | None = None
    scratch = (root / "build" / "scratch").resolve()
    for index, (configuration, command) in enumerate(zip(configurations, recorded)):
        require(isinstance(command, list) and len(command) >= 3, f"malformed L2 command {index}")
        output = Path(command[2])
        require(output.is_absolute(), f"L2 command {index} output is not absolute")
        try:
            output.resolve().relative_to(scratch)
        except ValueError as error:
            raise RuntimeError(f"L2 command {index} output escaped build/scratch") from error
        require(output.name == f"configuration_{index:03d}.csv", f"wrong L2 command output identity {index}")
        if common_work_parent is None:
            common_work_parent = output.parent
        require(output.parent == common_work_parent, f"L2 command {index} uses a different work directory")
        require(
            command == exact_command_arguments(configuration, protocol, executable, output),
            f"L2 command arguments differ from protocol at configuration {index}",
        )


def require_exact_text(row: dict[str, str], field: str, expected: str, index: int) -> None:
    require(row[field] == expected, f"L2 row {index} has wrong {field}")


def require_finite(row: dict[str, str], field: str, index: int, *, positive: bool = False) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"L2 row {index} has invalid {field}") from error
    require(math.isfinite(value), f"L2 row {index} has non-finite {field}")
    if positive:
        require(value > 0.0, f"L2 row {index} has non-positive {field}")
    return value


def verify_admission(row: dict[str, str], configuration: dict, protocol: dict, index: int) -> None:
    expected = EXPECTED_ADMISSION[configuration["solver"]]
    actual = {field: int(row[field]) for field in ADMISSION_FIELDS}
    require(actual == expected, f"L2 row {index} admission fields differ from audited policy")
    require(actual["estimated_memory_bytes"] == max(
        actual[field] for field in ADMISSION_FIELDS[1:-1]
    ), f"L2 row {index} memory high-water mark is inconsistent")
    require(actual["estimated_memory_bytes"] <= protocol["execution"]["memory_budget_bytes"], f"L2 row {index} exceeds memory admission")
    count = configuration["bodies"]
    pair_work = count * (count - 1)
    if configuration["solver"] == "direct":
        expected_work = pair_work * (1 + 2 * protocol["execution"]["steps"]) * (
            protocol["execution"]["warmups"] + protocol["execution"]["retained_repetitions"]
        )
        require(expected_work == actual["estimated_direct_interactions"], f"L2 row {index} Direct work estimate is wrong")
        require(expected_work > protocol["execution"]["maximum_direct_interactions"], f"L2 row {index} Direct skip is not independently justified")
    else:
        reference_calls = 1 + 2 * protocol["execution"]["steps"]
        expected_work = pair_work * reference_calls + (pair_work // 2) * 3
        require(expected_work == actual["estimated_direct_interactions"], f"L2 row {index} reference work estimate is wrong")
        require(expected_work <= protocol["execution"]["maximum_direct_interactions"], f"L2 row {index} completed despite failed work admission")


def verify_source_manifest(run_directory: Path, root: Path) -> dict[str, str]:
    entries = {}
    for line in (run_directory / "source_manifest.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(" *", 1)
        require(relative not in entries, f"duplicate L2 source input: {relative}")
        entries[relative] = expected
    require(set(entries) == set(custody.SOURCES), "L2 source manifest input set is not exact")
    for relative, expected in entries.items():
        source = root / relative
        require(source.is_file(), f"L2 source input missing: {relative}")
        require(publication_core.sha256(source) == expected, f"L2 source hash mismatch: {relative}")
    return entries


def verify_run(run_directory: Path, root: Path, expected_run_id: str | None = None) -> dict:
    run_directory = run_directory.resolve()
    manifest = l2.read_run_manifest(run_directory)
    if expected_run_id is not None:
        require(manifest["run_id"] == expected_run_id, "unexpected L2 run ID")
    protocol = l2.load_protocol(run_directory / "protocol.json")
    require(
        (run_directory / "protocol.json").read_bytes() == l2.DEFAULT_PROTOCOL.read_bytes(),
        "retained L2 protocol differs from the repository protocol",
    )
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    require(metadata.get("schema") == "large-n-l2-metadata-v1", "wrong L2 metadata schema")
    require(metadata.get("publication_status") == "complete", "L2 metadata is not complete")
    require(metadata.get("run_id") == manifest["run_id"], "L2 metadata/run ID mismatch")
    require(metadata.get("stage5_evidence") is False, "L2 evidence is mislabeled as Stage 5")
    require(metadata.get("classification") == "exploratory_pilot", "wrong L2 classification")
    require(metadata.get("study") == protocol["study"], "wrong L2 metadata study")
    require(metadata.get("protocol_version") == protocol["version"], "wrong L2 metadata protocol version")
    require(metadata.get("matrix") == exact_matrix(protocol), "L2 metadata matrix differs from protocol")
    outputs = metadata.get("outputs", {})
    require(
        {key: outputs.get(key) for key in ("raw_csv", "summary_json", "protocol", "build_provenance", "source_manifest")}
        == {
            "raw_csv": "raw.csv",
            "summary_json": "summary.json",
            "protocol": "protocol.json",
            "build_provenance": "build_provenance.json",
            "source_manifest": "source_manifest.sha256",
        },
        "L2 metadata output identities are wrong",
    )
    for name, key in (
        ("raw.csv", "raw_csv_sha256"),
        ("summary.json", "summary_json_sha256"),
        ("protocol.json", "protocol_sha256"),
        ("build_provenance.json", "build_provenance_sha256"),
        ("source_manifest.sha256", "source_manifest_sha256"),
    ):
        require(publication_core.sha256(run_directory / name) == outputs.get(key), f"L2 metadata hash mismatch: {name}")
    verify_source_manifest(run_directory, root)

    provenance = json.loads((run_directory / "build_provenance.json").read_text(encoding="utf-8"))
    require(provenance.get("schema") == "large-n-l2-build-provenance-v1", "wrong L2 provenance schema")
    require(provenance.get("source_manifest_sha256") == outputs["source_manifest_sha256"], "L2 provenance/source mismatch")
    require(provenance.get("protocol_sha256") == outputs["protocol_sha256"], "L2 provenance/protocol mismatch")
    cache = provenance.get("cmake_cache", {})
    require(
        cache.get("CMAKE_BUILD_TYPE") == "Release"
        and cache.get("NBODY_ENABLE_OPENMP") == "ON"
        and cache.get("NBODY_ENABLE_CUDA") == "OFF"
        and cache.get("NBODY_ENABLE_MPI") == "OFF",
        "L2 build is not canonical OpenMP Release",
    )
    require(provenance.get("build_directory") == "build/openmp-release", "wrong L2 build directory")
    executable_record = provenance.get("executable", {})
    require(executable_record.get("path") == "build/openmp-release/nbody_large_n_lab.exe", "wrong L2 executable path")
    require(isinstance(executable_record.get("sha256"), str) and len(executable_record["sha256"]) == 64, "invalid L2 executable hash")
    executable = (root / executable_record["path"]).resolve()
    if executable.is_file():
        require(publication_core.sha256(executable) == executable_record["sha256"], "measured L2 executable hash changed")
    verify_commands(metadata, protocol, executable, root)

    rows = read_rows(run_directory / "raw.csv")
    require(len(rows) == 60, "L2 pilot must contain exactly 60 retained rows")
    require(outputs.get("raw_rows") == 60, "L2 metadata raw-row count is wrong")
    require(outputs.get("status_counts") == {"completed": 54, "skipped": 6}, "L2 status counts are wrong")
    preamble = (run_directory / "raw.csv").read_text(encoding="utf-8").splitlines()[:4]
    require(
        preamble
        == [
            "# study=large_n_laboratory_l2_controlled_scaling",
            "# stage5_evidence=false",
            "# phase=exploratory_pilot_8192",
            "# graphical_fps=false",
        ],
        "L2 raw preamble is wrong",
    )
    expected_sequence = expected_rows(protocol)
    expected_identities = [expected_row_identity(*item) for item in expected_sequence]
    actual_identities = [canonical_row_identity(row) for row in rows]
    require(actual_identities == expected_identities, "L2 row sequence differs from generated protocol matrix")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    execution = protocol["execution"]
    for index, (row, (configuration, repetition, status)) in enumerate(zip(rows, expected_sequence)):
        require_exact_text(row, "protocol_phase", PHASE, index)
        require_exact_text(row, "timestep_s", l2.csv_float(execution["timestep_s"]), index)
        require_exact_text(row, "steps", str(execution["steps"]), index)
        require_exact_text(row, "warmups", str(execution["warmups"]), index)
        require_exact_text(row, "repetitions", str(execution["retained_repetitions"]), index)
        require_exact_text(row, "repetition", str(repetition), index)
        require_exact_text(row, "wall_time_budget_s", l2.csv_float(execution["per_configuration_wall_time_budget_s"]), index)
        require_exact_text(row, "memory_budget_bytes", str(execution["memory_budget_bytes"]), index)
        require_exact_text(row, "maximum_direct_interactions", str(execution["maximum_direct_interactions"]), index)
        verify_admission(row, configuration, protocol, index)
        key = expected_row_identity(configuration, 0, status)[:-2]
        groups.setdefault(key, []).append(row)
        if status == "skipped":
            require_exact_text(row, "message", SKIP_MESSAGE, index)
            require_exact_text(row, "observed_worker_threads", "0", index)
            require_exact_text(row, "worker_count_provenance", "", index)
            require_exact_text(row, "direct_reference_available", "0", index)
            require_exact_text(row, "direct_reference_reason", DIRECT_REFERENCE_DISABLED, index)
            require_exact_text(row, "reference_mode", "none", index)
            require_exact_text(row, "reference_sample_size", "0", index)
            require_exact_text(row, "accuracy_gate", "not_applicable", index)
            require_exact_text(row, "accuracy_reason", "accuracy gate applies only to completed Barnes-Hut rows", index)
            for field in (*l2.TIMING_FIELDS, *l2.SCIENTIFIC_FIELDS, *MEASURED_RESOURCE_FIELDS):
                require_exact_text(row, field, "0", index)
        else:
            require_exact_text(row, "message", COMPLETED_MESSAGE, index)
            expected_worker = ("1", "serial") if configuration["backend"] == "sequential" else ("4", "observed_in_timed_region")
            require(
                (row["observed_worker_threads"], row["worker_count_provenance"]) == expected_worker,
                f"L2 row {index} worker provenance differs from backend semantics",
            )
            require_exact_text(row, "direct_reference_available", "1", index)
            require_exact_text(row, "direct_reference_reason", "", index)
            require_exact_text(row, "reference_mode", "full", index)
            require_exact_text(row, "reference_sample_size", str(configuration["bodies"]), index)
            for field in l2.TIMING_FIELDS:
                require_finite(row, field, index, positive=True)
            for field in l2.SCIENTIFIC_FIELDS:
                value = require_finite(row, field, index)
                # Every current L2 scientific field is explicitly a magnitude,
                # count, floor, or absolute relative drift. Do not generalize
                # this rule to future signed observables.
                require(value >= 0.0, f"L2 row {index} has negative magnitude field {field}")
            for field in MEASURED_RESOURCE_FIELDS:
                require(int(row[field]) > 0, f"L2 row {index} has non-positive measured resource {field}")
            require(float(row["total_experiment_s"]) <= execution["per_configuration_wall_time_budget_s"], f"L2 row {index} exceeded wall-time admission")
            require(math.isclose(
                float(row["steps_per_second"]),
                execution["steps"] / float(row["complete_timestep_s"]),
                rel_tol=1e-15,
            ), f"L2 row {index} steps-per-second derivation is wrong")
            gate, reason = l2.accuracy_gate(row, protocol)
            require((row["accuracy_gate"], row["accuracy_reason"]) == (gate, reason), f"L2 row {index} accuracy gate was not reconstructed")
    require(len(groups) == 24, "L2 matrix does not contain exactly 24 generated groups")
    for key, group in groups.items():
        if key[4] == "direct":
            require(len(group) == 1, f"Direct skip group must have one row: {key}")
        else:
            require(len(group) == 3, f"Barnes-Hut group must have three rows: {key}")
            for field in (*l2.SCIENTIFIC_FIELDS, *l2.RESOURCE_FIELDS, "accuracy_gate", "accuracy_reason"):
                require(len({row[field] for row in group}) == 1, f"non-timing L2 field changed across repetitions: {key} {field}")
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    require(summary == l2.reconstruct_summary(rows, protocol), "L2 summary does not reconstruct exactly")
    require(metadata.get("pilot_decision") == summary["pilot_decision"], "L2 pilot decision mismatch")
    return metadata


def inspect_publication(evidence: Path) -> dict:
    evidence = evidence.resolve()
    pointer = l2.read_pointer(evidence)
    require(pointer is not None, "missing L2 publication pointer")
    runs = evidence / "runs"
    referenced = {pointer["current"]["run_id"], *(item["run_id"] for item in pointer["history"])}
    incomplete = sorted(path.name for path in runs.iterdir() if path.is_dir() and path.name.startswith(".install-"))
    unreferenced = sorted(path.name for path in runs.iterdir() if path.is_dir() and not path.name.startswith(".install-") and path.name not in referenced)
    require(not incomplete, "incomplete L2 installation detected")
    require(not unreferenced, "unreferenced L2 run detected")
    return {"pointer": pointer, "current_run": runs / pointer["current"]["run_id"]}


def verify_evidence(evidence: Path = DEFAULT_EVIDENCE, root: Path = ROOT) -> dict:
    inspection = inspect_publication(evidence)
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
    print(
        "Large-N L2 evidence verified: "
        f"run {metadata['run_id']}, {metadata['outputs']['raw_rows']} rows, "
        f"raw SHA-256 {metadata['outputs']['raw_csv_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
