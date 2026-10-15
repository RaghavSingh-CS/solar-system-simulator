#!/usr/bin/env python3
"""Strict, read-only verification for versioned Large-N L1 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "data" / "benchmarks" / "large_n_l1_pilot"
RUN_FILES = (
    "raw.csv",
    "summary.json",
    "metadata.json",
    "build_provenance.json",
    "source_manifest.sha256",
    "README.md",
)
LEGACY_PUBLICATION_FILE = "legacy_publication.json"
SOURCE_CUSTODY_COMMIT = "3f665c2610f3d65214494ff18f2f06d3c063513a"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
MANIFEST_INPUTS = {
    ".gitattributes",
    "CMakeLists.txt",
    "CMakePresets.json",
    "include/nbody/body.hpp",
    "include/nbody/vector3.hpp",
    "include/nbody/direct_force.hpp",
    "include/nbody/simulation.hpp",
    "include/nbody/diagnostics.hpp",
    "include/nbody/openmp_backend.hpp",
    "include/nbody/barnes_hut.hpp",
    "include/nbody/large_n_laboratory.hpp",
    "src/simulation.cpp",
    "src/diagnostics.cpp",
    "backends/openmp/openmp_backend.cpp",
    "src/barnes_hut.cpp",
    "src/large_n_laboratory.cpp",
    "src/large_n_laboratory_main.cpp",
    "tests/test_large_n_laboratory.cpp",
    "tests/test_large_n_l1_tools.py",
    "scripts/run_large_n_l1_pilot.py",
    "scripts/capture_large_n_l1_provenance.py",
    "scripts/verify_large_n_l1_evidence.py",
}
MEDIAN_FIELDS = (
    "generation_s",
    "initial_direct_force_s",
    "direct_trajectory_s",
    "diagnostics_s",
    "complete_force_s",
    "complete_timestep_s",
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
NONNEGATIVE_FLOAT_FIELDS = set(MEDIAN_FIELDS + SCIENTIFIC_FIELDS)
NONNEGATIVE_INTEGER_FIELDS = set(RESOURCE_FIELDS) | {
    "bodies",
    "steps",
    "warmups",
    "repetitions",
    "repetition",
    "requested_threads",
    "effective_threads",
    "observed_worker_threads",
    "memory_budget_bytes",
    "maximum_direct_interactions",
    "cancellation_sensitive_count",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_run_id(run_id: object) -> bool:
    return (
        isinstance(run_id, str)
        and bool(RUN_ID_PATTERN.fullmatch(run_id))
        and run_id not in {".", ".."}
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    return list(
        csv.DictReader(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        )
    )


def reconstruct_summary(rows: list[dict[str, str]]) -> dict:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["distribution"],
            row["bodies"],
            row["backend"],
            row["solver"],
            row["theta"],
            row["effective_threads"],
        )
        grouped.setdefault(key, []).append(row)
    groups = []
    for key in sorted(grouped):
        group = grouped[key]
        medians = {
            field: statistics.median(float(row[field]) for row in group)
            for field in MEDIAN_FIELDS
        }
        force = [float(row["complete_force_s"]) for row in group]
        groups.append(
            {
                "distribution": key[0],
                "bodies": int(key[1]),
                "backend": key[2],
                "solver": key[3],
                "theta": float(key[4]),
                "effective_threads": int(key[5]),
                "observed_worker_threads": int(group[0]["observed_worker_threads"]),
                "worker_count_provenance": group[0]["worker_count_provenance"],
                "retained_repetitions": len(group),
                "medians": medians,
                "force_range_over_median":
                    (max(force) - min(force)) / medians["complete_force_s"],
                "scientific": {
                    field: float(group[0][field]) for field in SCIENTIFIC_FIELDS
                },
                "resources": {
                    field: int(group[0][field]) for field in RESOURCE_FIELDS
                },
            }
        )
    return {
        "definitions": {
            "complete_force_s":
                "one complete force call including Barnes-Hut tree build",
            "complete_timestep_s":
                "configured Velocity Verlet batch; two force calls per step",
            "initial_direct_force_s":
                "one initial Sequential Direct acceleration call",
            "direct_trajectory_s":
                "matched Sequential Direct Velocity Verlet trajectory only",
            "diagnostics_s": "three conservation-diagnostic evaluations only",
            "total_experiment_s":
                "generation through final characterization; excludes CSV serialization",
            "graphical_fps": "not measured",
        },
        "groups": groups,
    }


def read_run_manifest(run_directory: Path) -> dict:
    path = run_directory / "run_manifest.json"
    require(path.is_file(), f"missing run manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id")
    require(valid_run_id(run_id), "run manifest has an unsafe run ID")
    require(
        manifest == {
            "schema": "large-n-l1-run-v1",
            "status": "complete",
            "run_id": run_id,
            "files": manifest.get("files"),
        },
        "run manifest has an unknown or incomplete schema",
    )
    expected_files = set(RUN_FILES)
    if LEGACY_PUBLICATION_FILE in manifest.get("files", {}):
        expected_files.add(LEGACY_PUBLICATION_FILE)
    require(
        isinstance(manifest["files"], dict)
        and set(manifest["files"]) == expected_files,
        "run manifest file set is incomplete or extraneous",
    )
    for name, expected in manifest["files"].items():
        path = run_directory / name
        require(path.is_file(), f"missing run artifact: {name}")
        require(sha256(path) == expected, f"run artifact hash mismatch: {name}")
    return manifest


def validate_nonnegative_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    require(math.isfinite(value), f"non-finite raw field: {field}")
    require(value >= 0.0, f"negative raw field: {field}")
    return value


def validate_nonnegative_integer(row: dict[str, str], field: str) -> int:
    text = row[field]
    require(text and text.strip() == text, f"malformed integer field: {field}")
    try:
        value = int(text, 10)
    except ValueError as error:
        raise RuntimeError(f"malformed integer field: {field}") from error
    require(str(value) == text, f"noncanonical integer field: {field}")
    require(value >= 0, f"negative integer field: {field}")
    return value


def committed_source_bytes(root: Path, relative: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{SOURCE_CUSTODY_COMMIT}:{relative}"],
            cwd=root,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"committed L1 source custody input is unavailable: {relative}"
        ) from error


def verify_source_manifest(run_directory: Path, root: Path) -> dict[str, str]:
    manifest_entries: dict[str, str] = {}
    for line in (run_directory / "source_manifest.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, relative = line.split(" *", 1)
        require(relative not in manifest_entries, f"duplicate manifest input: {relative}")
        manifest_entries[relative] = expected
    require(set(manifest_entries) == MANIFEST_INPUTS, "source manifest input set is not exact")
    for relative, expected in manifest_entries.items():
        actual = hashlib.sha256(committed_source_bytes(root, relative)).hexdigest()
        require(actual == expected, f"committed source hash mismatch: {relative}")
    return manifest_entries


def verify_scientific_run(
    run_directory: Path,
    root: Path,
    expected_run_id: str | None = None,
    verify_source: bool = True,
    allow_legacy: bool = False,
) -> dict:
    run_directory = run_directory.resolve()
    root = root.resolve()
    manifest = read_run_manifest(run_directory)
    if expected_run_id is not None:
        require(manifest["run_id"] == expected_run_id, "unexpected run ID")
    is_legacy_run = LEGACY_PUBLICATION_FILE in manifest["files"]
    if allow_legacy:
        require(is_legacy_run, "legacy verification requires legacy publication custody")
        legacy_publication = legacy_backup_publication(
            run_directory, LEGACY_PUBLICATION_FILE
        )
        require(
            all(
                manifest["files"][name] == legacy_publication["files"][name]
                for name in RUN_FILES
            ),
            "legacy publication/run manifest artifact mismatch",
        )
    else:
        require(not is_legacy_run, "legacy run requires explicit legacy verification")

    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    require(metadata.get("publication_status") == "complete", "metadata is stale/incomplete")
    metadata_run_id = metadata.get("run_id")
    if allow_legacy:
        require(
            metadata_run_id in (None, manifest["run_id"]),
            "legacy metadata/run ID mismatch",
        )
    else:
        require(metadata_run_id == manifest["run_id"], "metadata/run ID mismatch")
    require(
        metadata.get("stage5_evidence") is False
        and metadata.get("exploratory") is True,
        "incorrect evidence classification",
    )
    matrix = metadata.get("matrix", {})
    expected_matrix = {
        "body_counts": [256, 1024, 4096],
        "distributions": ["uniform", "clustered", "planar"],
        "theta_values": [0.25, 0.5, 0.8],
        "backends": ["sequential", "openmp"],
        "solvers": ["direct", "barnes-hut"],
        "requested_openmp_threads": 4,
        "timestep_s": 10.0,
        "steps": 1,
        "warmups": 1,
        "retained_repetitions": 2,
        "per_configuration_wall_time_budget_s": 30.0,
        "hard_subprocess_timeout_s": 35.0,
        "memory_budget_bytes": 536870912,
        "maximum_direct_interactions": 500000000,
        "maximum_direct_reference_bodies": 4096,
        "seed_formula":
            "0x4c3150494c4f5400 + distribution_index*0x100000 + N",
    }
    require(matrix == expected_matrix, "metadata matrix or budget policy is not exact")
    outputs = metadata.get("outputs", {})
    require(
        outputs.get("raw_rows") == 144
        and outputs.get("status_counts") == {"completed": 144},
        "metadata row/status counts are not exact",
    )
    for name, key in (
        ("raw.csv", "raw_csv_sha256"),
        ("summary.json", "summary_json_sha256"),
        ("build_provenance.json", "build_provenance_sha256"),
        ("source_manifest.sha256", "source_manifest_sha256"),
    ):
        require(
            sha256(run_directory / name) == outputs.get(key),
            f"metadata hash mismatch: {name}",
        )
    if verify_source:
        verify_source_manifest(run_directory, root)

    provenance = json.loads(
        (run_directory / "build_provenance.json").read_text(encoding="utf-8")
    )
    require(
        provenance.get("source_manifest_sha256") == outputs["source_manifest_sha256"],
        "provenance/source manifest mismatch",
    )
    cache = provenance.get("cmake_cache", {})
    require(
        cache.get("CMAKE_BUILD_TYPE") == "Release"
        and cache.get("NBODY_ENABLE_OPENMP") == "ON"
        and cache.get("NBODY_ENABLE_CUDA") == "OFF"
        and cache.get("NBODY_ENABLE_MPI") == "OFF",
        "build cache is not canonical OpenMP Release",
    )
    commands = "\n".join(provenance.get("effective_commands", []))
    require(
        "large_n_laboratory.cpp" in commands and "openmp" in commands.lower(),
        "effective build commands are incomplete",
    )
    executable = root / provenance["executable"]["path"]
    if executable.is_file():
        require(
            sha256(executable) == provenance["executable"]["sha256"],
            "canonical executable hash differs from measured binary",
        )

    rows = read_rows(run_directory / "raw.csv")
    require(len(rows) == 144, "raw evidence must contain exactly 144 rows")
    preamble = (run_directory / "raw.csv").read_text(encoding="utf-8").splitlines()[:4]
    require(
        preamble
        == [
            "# study=large_n_laboratory_l1_cpu_pilot",
            "# stage5_evidence=false",
            "# exploratory=true",
            "# graphical_fps=false",
        ],
        "raw preamble is stale or mislabeled",
    )
    expected_keys = set()
    for distribution in ("uniform", "clustered", "planar"):
        for bodies in (256, 1024, 4096):
            for backend, threads in (("sequential", 1), ("openmp", 4)):
                expected_keys.add((distribution, bodies, backend, "direct", 0.25, threads))
                for theta in (0.25, 0.5, 0.8):
                    expected_keys.add(
                        (distribution, bodies, backend, "barnes-hut", theta, threads)
                    )
    groups: dict[tuple, list[dict[str, str]]] = {}
    for row in rows:
        require(row["status"] == "completed", "raw contains a non-completed result")
        for field in NONNEGATIVE_FLOAT_FIELDS:
            validate_nonnegative_float(row, field)
        for field in NONNEGATIVE_INTEGER_FIELDS:
            validate_nonnegative_integer(row, field)
        require(
            float(row["relative_floor_m_s2"]) > 0.0,
            "relative acceleration denominator floor must be positive",
        )
        require(
            float(row["complete_force_s"]) > 0.0,
            "complete force duration must be positive for variability",
        )
        timestep = float(row["timestep_s"])
        wall_budget = float(row["wall_time_budget_s"])
        require(
            math.isfinite(timestep) and timestep > 0.0,
            "row timestep must be finite and positive",
        )
        require(
            math.isfinite(wall_budget) and wall_budget > 0.0,
            "row wall-time budget must be finite and positive",
        )
        key = (
            row["distribution"],
            int(row["bodies"]),
            row["backend"],
            row["solver"],
            float(row["theta"]),
            int(row["effective_threads"]),
        )
        groups.setdefault(key, []).append(row)
        expected_threads = 1 if row["backend"] == "sequential" else 4
        require(int(row["requested_threads"]) == expected_threads, "requested worker mismatch")
        expected_worker = (
            (1, "serial")
            if row["backend"] == "sequential"
            else (
                (4, "observed_in_timed_region")
                if row["solver"] == "barnes-hut"
                else (0, "runtime_probe_only")
            )
        )
        require(
            (int(row["observed_worker_threads"]), row["worker_count_provenance"])
            == expected_worker,
            "worker-count provenance mismatch",
        )
        require(
            (
                int(row["steps"]),
                int(row["warmups"]),
                int(row["repetitions"]),
                timestep,
            )
            == (1, 1, 2, 10.0),
            "row timestep/repetition policy mismatch",
        )
        require(
            (
                int(row["memory_budget_bytes"]),
                int(row["maximum_direct_interactions"]),
                wall_budget,
            )
            == (536870912, 500000000, 30.0),
            "row resource policy mismatch",
        )
        require(
            row["direct_reference_available"] == "1"
            and row["direct_reference_reason"] == "",
            "Direct reference is not genuinely available",
        )
        expected_seed = (
            0x4C3150494C4F5400
            + ("uniform", "clustered", "planar").index(row["distribution"])
            * 0x100000
            + int(row["bodies"])
        )
        require(int(row["seed"]) == expected_seed, "seed formula mismatch")
        require(
            int(row["estimated_memory_bytes"])
            == max(
                int(row[field])
                for field in (
                    "generation_memory_bytes",
                    "direct_reference_memory_bytes",
                    "warmup_memory_bytes",
                    "retained_repetition_memory_bytes",
                    "characterization_memory_bytes",
                    "serialization_memory_bytes",
                )
            ),
            "memory high-water field mismatch",
        )
        require(
            int(row["estimated_memory_bytes"]) > 0
            and int(row["generation_memory_bytes"]) > 0
            and int(row["serialization_memory_bytes"]) > 0,
            "required memory estimate is not positive",
        )
    require(
        set(groups) == expected_keys and len(groups) == 72,
        "raw configuration matrix has missing or extraneous groups",
    )
    for key, group in groups.items():
        require(
            len(group) == 2
            and {int(row["repetition"]) for row in group} == {0, 1},
            f"repetition set is not exactly 0/1: {key}",
        )
        for field in SCIENTIFIC_FIELDS + RESOURCE_FIELDS + (
            "observed_worker_threads",
            "worker_count_provenance",
        ):
            require(
                len({row[field] for row in group}) == 1,
                f"non-timing group field changed across repetitions: {key} {field}",
            )
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    require(summary == reconstruct_summary(rows), "summary does not reconstruct exactly")
    return metadata


def descriptor_from_run(run_directory: Path) -> dict:
    manifest = read_run_manifest(run_directory)
    return {
        "run_id": manifest["run_id"],
        "run_manifest_sha256": sha256(run_directory / "run_manifest.json"),
        "files": manifest["files"],
    }


def legacy_backup_publication(
    directory: Path, pointer_name: str = "publication.json"
) -> dict:
    pointer_path = directory / pointer_name
    require(pointer_path.is_file(), f"legacy backup lacks publication.json: {directory}")
    try:
        publication = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"legacy backup publication is unreadable: {directory}") from error
    require(
        publication == {
            "schema": "large-n-l1-publication-v1",
            "status": "complete",
            "files": publication.get("files"),
        },
        f"legacy backup publication schema is invalid: {directory}",
    )
    require(
        isinstance(publication["files"], dict)
        and set(publication["files"]) == set(RUN_FILES),
        f"legacy backup artifact set is invalid: {directory}",
    )
    for name, expected in publication["files"].items():
        artifact = directory / name
        require(artifact.is_file(), f"legacy backup artifact is missing: {name}")
        require(sha256(artifact) == expected, f"legacy backup hash mismatch: {name}")
    publication["publication_sha256"] = sha256(pointer_path)
    return publication


def descriptor_adopts_backup(item: dict, publication: dict) -> bool:
    files = item.get("files", {}) if isinstance(item, dict) else {}
    return (
        files.get(LEGACY_PUBLICATION_FILE) == publication["publication_sha256"]
        and all(files.get(name) == publication["files"][name] for name in RUN_FILES)
    )


def inspect_publication(evidence: Path) -> dict:
    evidence = evidence.resolve()
    pointer_path = evidence / "publication.json"
    require(pointer_path.is_file(), "missing authoritative publication pointer")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    require(
        pointer == {
            "schema": "large-n-l1-pointer-v1",
            "status": "complete",
            "current": pointer.get("current"),
            "history": pointer.get("history"),
        },
        "publication pointer is malformed or incomplete",
    )
    require(isinstance(pointer["history"], list), "publication history is not a list")
    descriptors = [pointer["current"], *pointer["history"]]
    run_ids = []
    runs_root = evidence / "runs"
    require(runs_root.is_dir(), "versioned runs directory is missing")
    for item in descriptors:
        require(isinstance(item, dict), "publication descriptor is malformed")
        run_id = item.get("run_id")
        require(valid_run_id(run_id), "publication descriptor has an unsafe run ID")
        require(run_id not in run_ids, "publication pointer repeats a run ID")
        run_ids.append(run_id)
        require(
            item == descriptor_from_run(runs_root / run_id),
            f"publication descriptor/hash mismatch: {run_id}",
        )
    incomplete = sorted(
        path.name
        for path in runs_root.iterdir()
        if path.is_dir() and path.name.startswith(".install-")
    )
    unreferenced = sorted(
        path.name
        for path in runs_root.iterdir()
        if path.is_dir() and not path.name.startswith(".install-")
        and path.name not in set(run_ids)
    )
    legacy_flat = sorted(name for name in RUN_FILES if (evidence / name).exists())
    adopted_backups = []
    stranded_backups = []
    for path in sorted(
        evidence.parent.glob(f".{evidence.name}.previous-*"),
        key=lambda candidate: candidate.name,
    ):
        if not path.is_dir():
            stranded_backups.append(path.name)
            continue
        try:
            publication = legacy_backup_publication(path)
        except RuntimeError:
            stranded_backups.append(path.name)
            continue
        if any(descriptor_adopts_backup(item, publication) for item in descriptors):
            adopted_backups.append(path.name)
        else:
            stranded_backups.append(path.name)
    return {
        "pointer": pointer,
        "current_run": runs_root / pointer["current"]["run_id"],
        "incomplete": incomplete,
        "unreferenced": unreferenced,
        "legacy_flat": legacy_flat,
        "adopted_backups": adopted_backups,
        "stranded_backups": stranded_backups,
    }


def verify_evidence(
    evidence: Path = DEFAULT_EVIDENCE,
    root: Path = ROOT,
    allow_orphans: bool = False,
) -> dict:
    inspection = inspect_publication(evidence)
    require(not inspection["incomplete"], "incomplete installed run detected")
    if not allow_orphans:
        require(not inspection["unreferenced"], "unreferenced installed run detected")
    require(not inspection["legacy_flat"], "stranded legacy flat publication detected")
    require(not inspection["stranded_backups"], "stranded legacy backup detected")
    manifest = read_run_manifest(inspection["current_run"])
    legacy = LEGACY_PUBLICATION_FILE in manifest["files"]
    return verify_scientific_run(
        inspection["current_run"],
        root,
        verify_source=not legacy,
        allow_legacy=legacy,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", nargs="?", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--expected-run-id")
    parser.add_argument("--allow-orphans", action="store_true")
    parser.add_argument("--allow-legacy-run", action="store_true")
    args = parser.parse_args()
    if args.run is not None:
        metadata = verify_scientific_run(
            args.run,
            ROOT,
            expected_run_id=args.expected_run_id,
            verify_source=not args.allow_legacy_run,
            allow_legacy=args.allow_legacy_run,
        )
    else:
        require(args.expected_run_id is None, "--expected-run-id requires --run")
        require(not args.allow_legacy_run, "--allow-legacy-run requires --run")
        metadata = verify_evidence(args.evidence, ROOT, args.allow_orphans)
    print(
        "Large-N L1 evidence verified: "
        f"run {metadata.get('run_id', 'legacy-migrated')}, "
        f"{metadata['outputs']['raw_rows']} rows, "
        f"raw SHA-256 {metadata['outputs']['raw_csv_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
