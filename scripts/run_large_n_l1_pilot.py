#!/usr/bin/env python3
"""Run and transactionally publish the bounded Large-N L1 CPU pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "benchmarks" / "large_n_l1_pilot"
RUN_FILES = (
    "raw.csv",
    "summary.json",
    "metadata.json",
    "build_provenance.json",
    "source_manifest.sha256",
    "README.md",
)
LEGACY_PUBLICATION_FILE = "legacy_publication.json"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
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

Checkpoint = Callable[[str], None]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def valid_run_id(run_id: str) -> bool:
    return bool(RUN_ID_PATTERN.fullmatch(run_id)) and run_id not in {".", ".."}


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"run-{timestamp}-{uuid.uuid4().hex[:12]}"


class PublicationLock:
    """Kernel-owned, process-crash-released lock for one evidence root."""

    def __init__(self, evidence_root: Path, timeout_seconds: float = 30.0):
        self.evidence_root = evidence_root.resolve()
        self.path = self.evidence_root / ".publication.lock"
        self.timeout_seconds = timeout_seconds
        self._stream = None
        self._locked = False

    def _try_lock(self) -> bool:
        assert self._stream is not None
        self._stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def __enter__(self):
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+b", buffering=0)
        if self.path.stat().st_size == 0:
            self._stream.write(b"\0")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        deadline = time.monotonic() + self.timeout_seconds
        while not self._try_lock():
            if time.monotonic() >= deadline:
                owner = "owner metadata unavailable while Windows byte lock is held"
                try:
                    self._stream.seek(0)
                    owner = self._stream.read().decode("utf-8", errors="replace")
                except OSError:
                    pass
                finally:
                    self._stream.close()
                    self._stream = None
                raise RuntimeError(
                    "Large-N publication lock is held by another live process: "
                    + owner.strip()
                )
            time.sleep(0.025)
        self._locked = True
        owner = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_utc": datetime.now(timezone.utc).isoformat(),
        }
        encoded = (json.dumps(owner, sort_keys=True) + "\n").encode("utf-8")
        self._stream.seek(0)
        self._stream.write(encoded)
        self._stream.truncate(len(encoded))
        self._stream.flush()
        os.fsync(self._stream.fileno())
        return self

    def __exit__(self, _type, _value, _traceback):
        if self._stream is None:
            return False
        try:
            if self._locked:
                self._stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._locked = False
            self._stream.close()
            self._stream = None
        return False


def atomic_replace_json(path: Path, value: object, checkpoint: Checkpoint | None) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if checkpoint is not None:
            checkpoint("before_pointer_replace")
        try:
            os.replace(temporary, path)
        except OSError as error:
            raise RuntimeError(
                f"Windows-compatible publication pointer replacement failed: {error}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def run_manifest(run_directory: Path) -> dict:
    path = run_directory / "run_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"run manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id")
    expected_files = set(RUN_FILES)
    if LEGACY_PUBLICATION_FILE in manifest.get("files", {}):
        expected_files.add(LEGACY_PUBLICATION_FILE)
    if manifest != {
        "schema": "large-n-l1-run-v1",
        "status": "complete",
        "run_id": run_id,
        "files": manifest.get("files"),
    }:
        raise RuntimeError("run manifest has an unknown or incomplete schema")
    if not isinstance(run_id, str) or not valid_run_id(run_id):
        raise RuntimeError("run manifest has an unsafe run ID")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != expected_files:
        raise RuntimeError("run manifest file set is incomplete or extraneous")
    for name, expected_hash in manifest["files"].items():
        artifact = run_directory / name
        if not artifact.is_file() or sha256(artifact) != expected_hash:
            raise RuntimeError(f"run artifact hash mismatch: {name}")
    return manifest


def write_run_manifest(run_directory: Path, run_id: str, legacy: bool = False) -> dict:
    if not valid_run_id(run_id):
        raise RuntimeError("refusing unsafe run ID")
    names = list(RUN_FILES)
    if legacy:
        names.append(LEGACY_PUBLICATION_FILE)
    for name in names:
        if not (run_directory / name).is_file():
            raise RuntimeError(f"run artifact is missing: {name}")
    manifest = {
        "schema": "large-n-l1-run-v1",
        "status": "complete",
        "run_id": run_id,
        "files": {name: sha256(run_directory / name) for name in names},
    }
    write_json(run_directory / "run_manifest.json", manifest)
    return manifest


def descriptor(run_directory: Path) -> dict:
    manifest = run_manifest(run_directory)
    return {
        "run_id": manifest["run_id"],
        "run_manifest_sha256": sha256(run_directory / "run_manifest.json"),
        "files": manifest["files"],
    }


def pointer_document(current: dict, history: list[dict]) -> dict:
    return {
        "schema": "large-n-l1-pointer-v1",
        "status": "complete",
        "current": current,
        "history": history,
    }


def read_pointer(evidence_root: Path) -> dict | None:
    path = evidence_root / "publication.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "large-n-l1-pointer-v1":
        return None
    if value != pointer_document(value.get("current"), value.get("history")):
        raise RuntimeError("publication pointer has an unknown or incomplete schema")
    if not isinstance(value["history"], list):
        raise RuntimeError("publication history is not a list")
    descriptors = [value["current"], *value["history"]]
    seen = set()
    for item in descriptors:
        if not isinstance(item, dict) or not valid_run_id(item.get("run_id")):
            raise RuntimeError("publication pointer contains an unsafe descriptor")
        if item["run_id"] in seen:
            raise RuntimeError("publication pointer repeats a run ID")
        seen.add(item["run_id"])
        if item != descriptor(evidence_root / "runs" / item["run_id"]):
            raise RuntimeError(f"publication descriptor mismatch: {item['run_id']}")
    return value


def legacy_backup_paths(evidence_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in evidence_root.parent.glob(
                f".{evidence_root.name}.previous-*"
            )
            if path.is_dir()
        ),
        key=lambda path: path.name,
    )


def validate_legacy_flat_directory(
    directory: Path, pointer_name: str = "publication.json"
) -> dict:
    pointer_path = directory / pointer_name
    if not pointer_path.is_file():
        raise RuntimeError(f"legacy backup lacks publication.json: {directory}")
    try:
        publication = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"legacy backup publication is unreadable: {directory}") from error
    if publication != {
        "schema": "large-n-l1-publication-v1",
        "status": "complete",
        "files": publication.get("files"),
    }:
        raise RuntimeError(f"legacy backup publication schema is invalid: {directory}")
    if not isinstance(publication["files"], dict) or \
            set(publication["files"]) != set(RUN_FILES):
        raise RuntimeError(f"legacy backup artifact set is invalid: {directory}")
    for name, expected_hash in publication["files"].items():
        artifact = directory / name
        if not artifact.is_file() or sha256(artifact) != expected_hash:
            raise RuntimeError(f"legacy backup artifact hash mismatch: {name}")
    publication["publication_sha256"] = sha256(pointer_path)
    return publication


def legacy_backup_run_id(publication: dict) -> str:
    return f"legacy-backup-{publication['publication_sha256']}"


def descriptor_adopts_backup(item: dict, publication: dict) -> bool:
    files = item.get("files", {}) if isinstance(item, dict) else {}
    return (
        files.get(LEGACY_PUBLICATION_FILE) == publication["publication_sha256"]
        and all(files.get(name) == publication["files"][name] for name in RUN_FILES)
    )


def verify_run_with_strict_verifier(
    run_directory: Path, run_id: str, allow_legacy: bool = False
) -> None:
    command = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "verify_large_n_l1_evidence.py"),
        "--run", str(run_directory),
        "--expected-run-id", run_id,
    ]
    if allow_legacy:
        command.append("--allow-legacy-run")
    subprocess.run(command, cwd=ROOT, check=True)


def require_no_unadopted_legacy_backups(
    evidence_root: Path, pointer: dict | None
) -> None:
    descriptors = [] if pointer is None else [
        pointer["current"], *pointer["history"]
    ]
    unadopted = []
    for backup in legacy_backup_paths(evidence_root):
        try:
            publication = validate_legacy_flat_directory(backup)
        except RuntimeError:
            unadopted.append(backup.name)
            continue
        if not any(
            descriptor_adopts_backup(item, publication) for item in descriptors
        ):
            unadopted.append(backup.name)
    if unadopted:
        raise RuntimeError(
            "legacy backup requires explicit --recover-legacy-backup: "
            + ", ".join(unadopted)
        )


def cleanup_legacy_flat_duplicates_locked(
    evidence_root: Path, pointer: dict
) -> None:
    existing = [name for name in RUN_FILES if (evidence_root / name).exists()]
    if not existing:
        return
    legacy_descriptors = [
        item
        for item in [pointer["current"], *pointer["history"]]
        if LEGACY_PUBLICATION_FILE in item.get("files", {})
    ]
    if len(legacy_descriptors) != 1:
        raise RuntimeError(
            "stranded legacy flat files have no unique historical run"
        )
    legacy = legacy_descriptors[0]
    for name in existing:
        duplicate = evidence_root / name
        if sha256(duplicate) != legacy["files"][name]:
            raise RuntimeError(f"legacy flat duplicate differs from history: {name}")
    for name in existing:
        (evidence_root / name).unlink()


def migrate_legacy_layout_locked(
    evidence_root: Path, checkpoint: Checkpoint | None = None
) -> dict | None:
    pointer_path = evidence_root / "publication.json"
    if not pointer_path.exists():
        return None
    existing = json.loads(pointer_path.read_text(encoding="utf-8"))
    if existing.get("schema") == "large-n-l1-pointer-v1":
        pointer = read_pointer(evidence_root)
        cleanup_legacy_flat_duplicates_locked(evidence_root, pointer)
        return pointer
    if existing.get("schema") != "large-n-l1-publication-v1" or \
            existing.get("status") != "complete":
        raise RuntimeError("cannot migrate malformed legacy publication")
    if set(existing.get("files", {})) != set(RUN_FILES):
        raise RuntimeError("legacy publication file set is incomplete or extraneous")
    for name in RUN_FILES:
        artifact = evidence_root / name
        if not artifact.is_file() or sha256(artifact) != existing["files"][name]:
            raise RuntimeError(f"legacy publication hash mismatch: {name}")

    run_id = f"legacy-{existing['files']['raw.csv'][:16]}"
    runs = evidence_root / "runs"
    runs.mkdir(exist_ok=True)
    destination = runs / run_id
    if not destination.exists():
        installing = runs / f".install-{run_id}-{uuid.uuid4().hex}"
        installing.mkdir()
        try:
            for name in RUN_FILES:
                shutil.copy2(evidence_root / name, installing / name)
            shutil.copy2(pointer_path, installing / LEGACY_PUBLICATION_FILE)
            write_run_manifest(installing, run_id, legacy=True)
            run_manifest(installing)
            os.replace(installing, destination)
        finally:
            if installing.exists():
                shutil.rmtree(installing)
    legacy_descriptor = descriptor(destination)
    pointer = pointer_document(legacy_descriptor, [])
    atomic_replace_json(pointer_path, pointer, checkpoint)
    cleanup_legacy_flat_duplicates_locked(evidence_root, pointer)
    return pointer


def referenced_run_ids(pointer: dict | None) -> set[str]:
    if pointer is None:
        return set()
    return {
        pointer["current"]["run_id"],
        *(item["run_id"] for item in pointer["history"]),
    }


def find_unreferenced_runs(evidence_root: Path, pointer: dict | None) -> list[str]:
    runs = evidence_root / "runs"
    if not runs.exists():
        return []
    referenced = referenced_run_ids(pointer)
    return sorted(
        path.name
        for path in runs.iterdir()
        if path.is_dir() and not path.name.startswith(".install-")
        and path.name not in referenced
    )


def find_incomplete_installs(evidence_root: Path) -> list[str]:
    runs = evidence_root / "runs"
    if not runs.exists():
        return []
    return sorted(
        path.name
        for path in runs.iterdir()
        if path.is_dir() and path.name.startswith(".install-")
    )


def publish_versioned_run(
    staged: Path,
    evidence_root: Path,
    run_id: str,
    checkpoint: Checkpoint | None = None,
    lock_timeout_seconds: float = 30.0,
) -> dict:
    staged = staged.resolve()
    evidence_root = evidence_root.resolve()
    manifest = run_manifest(staged)
    if manifest["run_id"] != run_id:
        raise RuntimeError("staged run ID does not match publication request")

    with PublicationLock(evidence_root, lock_timeout_seconds):
        if checkpoint is not None:
            checkpoint("after_lock_acquired")
        pointer = migrate_legacy_layout_locked(evidence_root, checkpoint)
        pointer = read_pointer(evidence_root) if pointer is not None else None
        require_no_unadopted_legacy_backups(evidence_root, pointer)
        incomplete = find_incomplete_installs(evidence_root)
        if incomplete:
            raise RuntimeError(
                "incomplete run installation requires inspection: "
                + ", ".join(incomplete)
            )
        orphans = find_unreferenced_runs(evidence_root, pointer)
        if orphans:
            raise RuntimeError(
                "unreferenced run requires explicit recovery before publication: "
                + ", ".join(orphans)
            )

        # Recheck all bytes while the destination lock is held, closing the
        # verification-to-installation race for cooperating publishers.
        manifest = run_manifest(staged)
        runs = evidence_root / "runs"
        runs.mkdir(exist_ok=True)
        destination = runs / run_id
        if destination.exists():
            raise RuntimeError(f"run ID already exists: {run_id}")
        if checkpoint is not None:
            checkpoint("before_run_install")
        try:
            os.replace(staged, destination)
        except OSError as error:
            raise RuntimeError(f"immutable run installation failed: {error}") from error
        if checkpoint is not None:
            checkpoint("after_run_install")

        current = descriptor(destination)
        history = [] if pointer is None else [
            pointer["current"], *pointer["history"]
        ]
        next_pointer = pointer_document(current, history)
        atomic_replace_json(evidence_root / "publication.json", next_pointer, checkpoint)
        return next_pointer


def recover_orphan_run(
    evidence_root: Path,
    run_id: str,
    checkpoint: Checkpoint | None = None,
) -> dict:
    if not valid_run_id(run_id):
        raise RuntimeError("refusing unsafe recovery run ID")
    evidence_root = evidence_root.resolve()
    with PublicationLock(evidence_root):
        pointer = migrate_legacy_layout_locked(evidence_root, checkpoint)
        pointer = read_pointer(evidence_root) if pointer is not None else None
        incomplete = find_incomplete_installs(evidence_root)
        if incomplete:
            raise RuntimeError(
                "cannot recover while incomplete installations exist: "
                + ", ".join(incomplete)
            )
        orphans = find_unreferenced_runs(evidence_root, pointer)
        if orphans != [run_id]:
            raise RuntimeError(
                "recovery requires exactly the named complete orphan; found: "
                + ", ".join(orphans)
            )
        recovered = descriptor(evidence_root / "runs" / run_id)
        history = [] if pointer is None else [
            pointer["current"], *pointer["history"]
        ]
        next_pointer = pointer_document(recovered, history)
        atomic_replace_json(evidence_root / "publication.json", next_pointer, checkpoint)
        return next_pointer


def recover_legacy_backup(
    evidence_root: Path,
    checkpoint: Checkpoint | None = None,
) -> dict:
    """Explicitly adopt one verified legacy directory-swap backup."""
    evidence_root = evidence_root.resolve()
    with PublicationLock(evidence_root):
        backups = legacy_backup_paths(evidence_root)
        if not backups:
            return {"status": "no_recovery_needed"}
        if len(backups) != 1:
            raise RuntimeError(
                "legacy backup recovery requires exactly one candidate; found: "
                + ", ".join(path.name for path in backups)
            )
        backup = backups[0]
        publication = validate_legacy_flat_directory(backup)

        pointer = migrate_legacy_layout_locked(evidence_root)
        pointer = read_pointer(evidence_root) if pointer is not None else None
        if pointer is not None:
            current = pointer["current"]
            verify_run_with_strict_verifier(
                evidence_root / "runs" / current["run_id"],
                current["run_id"],
                allow_legacy=LEGACY_PUBLICATION_FILE in current["files"],
            )
            descriptors = [current, *pointer["history"]]
            if any(
                descriptor_adopts_backup(item, publication) for item in descriptors
            ):
                return {
                    "status": "already_adopted",
                    "run_id": next(
                        item["run_id"]
                        for item in descriptors
                        if descriptor_adopts_backup(item, publication)
                    ),
                }
        else:
            unexpected = [
                path.name
                for path in evidence_root.iterdir()
                if path.name not in {".publication.lock", "runs"}
            ]
            if unexpected:
                raise RuntimeError(
                    "cannot recover into an unrecognized primary evidence root: "
                    + ", ".join(sorted(unexpected))
                )

        incomplete = find_incomplete_installs(evidence_root)
        if incomplete:
            raise RuntimeError(
                "incomplete run installation requires inspection: "
                + ", ".join(incomplete)
            )

        run_id = legacy_backup_run_id(publication)
        runs = evidence_root / "runs"
        runs.mkdir(exist_ok=True)
        destination = runs / run_id
        other_orphans = [
            item for item in find_unreferenced_runs(evidence_root, pointer)
            if item != run_id
        ]
        if other_orphans:
            raise RuntimeError(
                "unrelated unreferenced run blocks legacy recovery: "
                + ", ".join(other_orphans)
            )

        if destination.exists():
            recovered = descriptor(destination)
            if not descriptor_adopts_backup(recovered, publication):
                raise RuntimeError(
                    "existing recovery run does not match the legacy backup"
                )
            verify_run_with_strict_verifier(
                destination, run_id, allow_legacy=True
            )
        else:
            installing = runs / f".install-{run_id}-{uuid.uuid4().hex}"
            installing.mkdir()
            try:
                for name in RUN_FILES:
                    shutil.copy2(backup / name, installing / name)
                shutil.copy2(
                    backup / "publication.json",
                    installing / LEGACY_PUBLICATION_FILE,
                )
                copied_publication = validate_legacy_flat_directory(
                    installing, LEGACY_PUBLICATION_FILE
                )
                if copied_publication != publication:
                    raise RuntimeError(
                        "legacy backup changed while its snapshot was copied"
                    )
                write_run_manifest(installing, run_id, legacy=True)
                verify_run_with_strict_verifier(
                    installing, run_id, allow_legacy=True
                )
                if checkpoint is not None:
                    checkpoint("before_recovery_install")
                os.replace(installing, destination)
            finally:
                if installing.exists():
                    shutil.rmtree(installing)
            if checkpoint is not None:
                checkpoint("after_recovery_install")
            recovered = descriptor(destination)
            if not descriptor_adopts_backup(recovered, publication):
                raise RuntimeError("installed recovery run does not match the backup")

        if pointer is None:
            next_pointer = pointer_document(recovered, [])
        else:
            next_pointer = pointer_document(
                pointer["current"], [*pointer["history"], recovered]
            )
        atomic_replace_json(
            evidence_root / "publication.json", next_pointer, checkpoint
        )
        confirmed = read_pointer(evidence_root)
        if confirmed != next_pointer:
            raise RuntimeError("legacy recovery pointer verification failed")
        return {"status": "recovered", "run_id": run_id}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executable",
        type=Path,
        default=ROOT / "build" / "openmp-release" / "nbody_large_n_lab.exe",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--recover-legacy-backup",
        action="store_true",
        help="adopt one verified .<evidence>.previous-* backup without measuring",
    )
    return parser.parse_args()


def data_lines(path: Path) -> tuple[str, list[str]]:
    content = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if len(content) < 2:
        raise RuntimeError(f"laboratory output has no data row: {path}")
    return content[0], content[1:]


def configurations(threads: int) -> list[dict[str, object]]:
    result = []
    for distribution_index, distribution in enumerate(
        ("uniform", "clustered", "planar")
    ):
        for bodies in (256, 1024, 4096):
            seed = 0x4C3150494C4F5400 + distribution_index * 0x100000 + bodies
            for backend, workers in (("sequential", 1), ("openmp", threads)):
                result.append(
                    dict(
                        distribution=distribution,
                        bodies=bodies,
                        seed=seed,
                        backend=backend,
                        solver="direct",
                        theta=0.25,
                        threads=workers,
                    )
                )
                for theta in (0.25, 0.5, 0.8):
                    result.append(
                        dict(
                            distribution=distribution,
                            bodies=bodies,
                            seed=seed,
                            backend=backend,
                            solver="barnes-hut",
                            theta=theta,
                            threads=workers,
                        )
                    )
    return result


def execute_configuration(
    command: list[str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def summarize(rows: list[dict[str, str]]) -> dict:
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


def evidence_readme(evidence: Path) -> str:
    metadata = json.loads((evidence / "metadata.json").read_text(encoding="utf-8"))
    outputs = metadata["outputs"]
    summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
    groups = {
        (g["distribution"], g["bodies"], g["backend"], g["solver"], g["theta"]): g
        for g in summary["groups"]
    }
    direct_speedups, bh_speedups, variability, tree_fractions = [], [], [], []
    p95 = {0.25: [], 0.5: [], 0.8: []}
    for distribution in ("uniform", "clustered", "planar"):
        sequential_direct = groups[
            (distribution, 4096, "sequential", "direct", 0.25)
        ]["medians"]["complete_force_s"]
        openmp_direct = groups[
            (distribution, 4096, "openmp", "direct", 0.25)
        ]["medians"]["complete_force_s"]
        direct_speedups.append(sequential_direct / openmp_direct)
        for theta in (0.25, 0.5, 0.8):
            sequential = groups[
                (distribution, 4096, "sequential", "barnes-hut", theta)
            ]
            openmp = groups[
                (distribution, 4096, "openmp", "barnes-hut", theta)
            ]
            bh_speedups.append(
                sequential["medians"]["complete_force_s"]
                / openmp["medians"]["complete_force_s"]
            )
            for group in (sequential, openmp):
                tree_fractions.append(
                    group["medians"]["tree_build_s"]
                    / group["medians"]["complete_force_s"]
                )
                p95[theta].append(group["scientific"]["p95_relative_error"])
    for group in summary["groups"]:
        variability.append(group["force_range_over_median"])
    return f"""# Large-N Laboratory L1 bounded CPU pilot

Exploratory, non-Stage-5 CPU evidence from the uncommitted Phase L1 candidate
based on `{metadata['source_base_commit']}`. Run ID: `{metadata['run_id']}`.
Exactly 144 completed rows cover 72 configurations at N=256/1024/4096,
three synthetic distributions, Sequential/OpenMP-4, Direct/Barnes--Hut, and
theta 0.25/0.5/0.8 for Barnes--Hut. Each used one warm-up, two retained
repetitions, one 10-second Velocity Verlet step, 512 MiB admission,
500,000,000 Direct interactions, a 30-second in-process boundary budget, and a
35-second subprocess timeout.

OpenMP Direct workers are runtime-probed but not observed inside the timed force
region; Barnes--Hut records actual timed-region workers. Results are simulation
throughput, not graphical FPS. No CUDA, 50K, or 100K execution is represented.
Two repetitions and uncontrolled system conditions do not establish a stable
crossover.

At 4096 bodies, configured/probed OpenMP-4 Direct speedup was
`{min(direct_speedups):.2f}x`--`{max(direct_speedups):.2f}x`; timed-region-observed
OpenMP-4 Barnes--Hut speedup was `{min(bh_speedups):.2f}x`--`{max(bh_speedups):.2f}x`.
Tree construction occupied `{100*min(tree_fractions):.2f}%`--`{100*max(tree_fractions):.2f}%`
of complete Barnes--Hut force time. P95 relative acceleration-error ranges were
`{min(p95[0.25]):.2e}`--`{max(p95[0.25]):.2e}` at theta 0.25,
`{min(p95[0.5]):.2e}`--`{max(p95[0.5]):.2e}` at theta 0.5, and
`{min(p95[0.8]):.2e}`--`{max(p95[0.8]):.2e}` at theta 0.8. Maximum retained
force range/median was `{100*max(variability):.1f}%`.

Custody hashes:
- raw.csv: `{outputs['raw_csv_sha256']}`
- summary.json: `{outputs['summary_json_sha256']}`
- metadata.json: `{sha256(evidence / 'metadata.json')}`
- build_provenance.json: `{outputs['build_provenance_sha256']}`
- source_manifest.sha256: `{outputs['source_manifest_sha256']}`

The enclosing `run_manifest.json` binds these immutable artifacts. The root
`publication.json` atomically selects the authoritative run and identifies all
historical published runs.

Legacy directory-swap backups are never adopted during ordinary measurement.
An operator may recover exactly one fully verified backup with
`python -B scripts/run_large_n_l1_pilot.py --recover-legacy-backup`; the source
backup is retained, and an existing current run remains current.

The OS lock and same-filesystem `os.replace` provide process-crash consistency
and atomic pointer visibility on the validated Windows filesystem. The pointer
temporary file is flushed and passed to `fsync`, but Python does not establish
directory-entry or storage-device flush durability here. Sudden-power-loss
durability is not certified.
"""


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.recover_legacy_backup:
        result = recover_legacy_backup(output)
        if result["status"] == "no_recovery_needed":
            print(f"no legacy backup requires recovery for {output}")
        elif result["status"] == "already_adopted":
            print(f"legacy backup already adopted as {result['run_id']}")
        else:
            print(f"recovered legacy backup as immutable run {result['run_id']}")
        return 0

    executable = args.executable.resolve()
    if not executable.is_file():
        raise RuntimeError(f"large-N executable not found: {executable}")
    if (args.threads, args.repetitions, args.warmups, args.steps) != (4, 2, 1, 1):
        raise RuntimeError(
            "L1 pilot requires threads=4, repetitions=2, warmups=1, steps=1"
        )
    if args.timeout_seconds != 30.0:
        raise RuntimeError("L1 pilot requires a 30-second budget")

    # Adopt any complete legacy flat publication before starting new work. A
    # failed measurement then leaves an already valid versioned pointer.
    with PublicationLock(output):
        pointer = migrate_legacy_layout_locked(output)
        pointer = read_pointer(output) if pointer is not None else None
        require_no_unadopted_legacy_backups(output, pointer)

    run_id = new_run_id()
    scratch_root = ROOT / "build" / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="large-n-l1-run-", dir=scratch_root
    ) as temporary:
        run_root = Path(temporary)
        staged = run_root / "candidate"
        work = run_root / "work"
        staged.mkdir()
        work.mkdir()
        command_log = []
        combined_rows = []
        header = None
        for index, configuration in enumerate(configurations(args.threads)):
            temporary_output = work / f"configuration_{index:03d}.csv"
            command = [
                str(executable),
                "--output", str(temporary_output),
                "--bodies", str(configuration["bodies"]),
                "--seed", str(configuration["seed"]),
                "--distribution", str(configuration["distribution"]),
                "--backend", str(configuration["backend"]),
                "--solver", str(configuration["solver"]),
                "--theta", str(configuration["theta"]),
                "--threads", str(configuration["threads"]),
                "--dt", "10",
                "--steps", "1",
                "--warmups", "1",
                "--repetitions", "2",
                "--direct-reference", "true",
                "--max-direct-reference-bodies", "4096",
                "--memory-budget-bytes", str(512 * 1024 * 1024),
                "--max-direct-interactions", "500000000",
                "--wall-time-budget-s", "30",
            ]
            command_log.append(command)
            try:
                completed = execute_configuration(command, 35.0)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"configuration {index} exceeded the hard subprocess timeout"
                ) from error
            if completed.returncode != 0:
                raise RuntimeError(
                    f"configuration {index} failed ({completed.returncode}): "
                    f"{completed.stdout}\n{completed.stderr}"
                )
            candidate_header, candidate_rows = data_lines(temporary_output)
            if header is None:
                header = candidate_header
            if candidate_header != header:
                raise RuntimeError("inconsistent CSV headers")
            parsed = list(csv.DictReader([candidate_header, *candidate_rows]))
            if len(parsed) != 2 or any(row["status"] != "completed" for row in parsed):
                raise RuntimeError(
                    f"configuration {index} did not produce two completed repetitions"
                )
            combined_rows.extend(candidate_rows)

        raw = staged / "raw.csv"
        with raw.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                "# study=large_n_laboratory_l1_cpu_pilot\n"
                "# stage5_evidence=false\n"
                "# exploratory=true\n"
                "# graphical_fps=false\n"
            )
            stream.write(header + "\n" + "\n".join(combined_rows) + "\n")
        parsed_rows = list(
            csv.DictReader(
                line
                for line in raw.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            )
        )
        write_json(staged / "summary.json", summarize(parsed_rows))
        metadata = {
            "study": "large_n_laboratory_l1_cpu_pilot",
            "publication_status": "staging",
            "run_id": run_id,
            "stage5_evidence": False,
            "exploratory": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_state": "uncommitted Phase L1 candidate; manifest binds exact bytes",
            "platform": {
                "system": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor()
                    or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
                "logical_processors": os.cpu_count(),
            },
            "matrix": {
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
                "memory_budget_bytes": 512 * 1024 * 1024,
                "maximum_direct_interactions": 500_000_000,
                "maximum_direct_reference_bodies": 4096,
                "seed_formula":
                    "0x4c3150494c4f5400 + distribution_index*0x100000 + N",
            },
            "commands": {
                "runner":
                    "python -B scripts/run_large_n_l1_pilot.py --threads 4 "
                    "--repetitions 2 --warmups 1 --steps 1 --timeout-seconds 30",
                "configuration_argv": command_log,
            },
            "outputs": {
                "raw_csv": "raw.csv",
                "raw_csv_sha256": sha256(raw),
                "raw_rows": len(parsed_rows),
                "status_counts": {"completed": len(parsed_rows)},
                "summary_json": "summary.json",
                "summary_json_sha256": sha256(staged / "summary.json"),
            },
            "limitations": [
                "Exploratory single-machine CPU pilot; not Stage 5 evidence.",
                "Two retained repetitions are not stable crossover evidence.",
                "OpenMP Direct workers are runtime-probed, not timed-region observed.",
                "Simulation throughput is not graphical FPS.",
                "CUDA and 50K/100K workloads were not executed.",
            ],
        }
        write_json(staged / "metadata.json", metadata)
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "capture_large_n_l1_provenance.py"),
                "--evidence", str(staged),
                "--build-dir", str(executable.parent),
            ],
            cwd=ROOT,
            check=True,
        )
        metadata = json.loads((staged / "metadata.json").read_text(encoding="utf-8"))
        metadata["publication_status"] = "complete"
        write_json(staged / "metadata.json", metadata)
        (staged / "README.md").write_text(
            evidence_readme(staged), encoding="utf-8", newline="\n"
        )
        write_run_manifest(staged, run_id)
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "verify_large_n_l1_evidence.py"),
                "--run", str(staged),
                "--expected-run-id", run_id,
            ],
            cwd=ROOT,
            check=True,
        )
        publish_versioned_run(staged, output, run_id)
    print(f"published immutable Large-N L1 run {run_id} under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
