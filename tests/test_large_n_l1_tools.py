#!/usr/bin/env python3
"""Adversarial tests for Large-N L1 evidence publication and verification."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "benchmarks" / "large_n_l1_pilot"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


RUNNER = load_module("large_n_l1_runner", ROOT / "scripts" / "run_large_n_l1_pilot.py")
VERIFY = load_module("large_n_l1_verify", ROOT / "scripts" / "verify_large_n_l1_evidence.py")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def authoritative_run() -> Path:
    pointer_path = EVIDENCE / "publication.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema") == "large-n-l1-pointer-v1":
        return EVIDENCE / "runs" / pointer["current"]["run_id"]
    return EVIDENCE


def copy_candidate(destination: Path, run_id: str) -> Path:
    source = authoritative_run()
    destination.mkdir(parents=True)
    for name in RUNNER.RUN_FILES:
        shutil.copy2(source / name, destination / name)
    with (destination / "source_manifest.sha256").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for relative in sorted(VERIFY.MANIFEST_INPUTS):
            digest = hashlib.sha256(VERIFY.committed_source_bytes(ROOT, relative)).hexdigest()
            stream.write(f"{digest} *{relative}\n")
    provenance_path = destination / "build_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_manifest_sha256"] = RUNNER.sha256(
        destination / "source_manifest.sha256"
    )
    write_json(provenance_path, provenance)
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["run_id"] = run_id
    metadata["outputs"].update(
        {
            "raw_csv_sha256": RUNNER.sha256(destination / "raw.csv"),
            "summary_json_sha256": RUNNER.sha256(destination / "summary.json"),
            "build_provenance_sha256": RUNNER.sha256(provenance_path),
            "source_manifest_sha256": RUNNER.sha256(
                destination / "source_manifest.sha256"
            ),
        }
    )
    write_json(metadata_path, metadata)
    RUNNER.write_run_manifest(destination, run_id)
    return destination


def publish_initial(root: Path, run_id: str = "initial") -> dict:
    candidate = copy_candidate(root.parent / f"candidate-{run_id}", run_id)
    return RUNNER.publish_versioned_run(candidate, root, run_id)


def create_legacy_backup(root: Path, suffix: str = "legacy") -> Path:
    backup = root.parent / f".{root.name}.previous-{suffix}"
    backup.mkdir()
    source = authoritative_run()
    for name in RUNNER.RUN_FILES:
        shutil.copy2(source / name, backup / name)
    metadata_path = backup / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("run_id", None)
    write_json(metadata_path, metadata)
    publication = {
        "schema": "large-n-l1-publication-v1",
        "status": "complete",
        "files": {
            name: RUNNER.sha256(backup / name) for name in RUNNER.RUN_FILES
        },
    }
    write_json(backup / "publication.json", publication)
    return backup


def refresh_run_and_pointer(root: Path) -> Path:
    pointer_path = root / "publication.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    run = root / "runs" / pointer["current"]["run_id"]
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = metadata["outputs"]
    outputs["raw_csv_sha256"] = RUNNER.sha256(run / "raw.csv")
    outputs["summary_json_sha256"] = RUNNER.sha256(run / "summary.json")
    outputs["build_provenance_sha256"] = RUNNER.sha256(run / "build_provenance.json")
    outputs["source_manifest_sha256"] = RUNNER.sha256(run / "source_manifest.sha256")
    write_json(metadata_path, metadata)
    legacy = (run / RUNNER.LEGACY_PUBLICATION_FILE).is_file()
    RUNNER.write_run_manifest(run, run.name, legacy=legacy)
    pointer["current"] = RUNNER.descriptor(run)
    write_json(pointer_path, pointer)
    return run


def mutate_raw_field(root: Path, field: str, value: str) -> None:
    inspection = VERIFY.inspect_publication(root)
    run = inspection["current_run"]
    raw = run / "raw.csv"
    lines = raw.read_text(encoding="utf-8").splitlines()
    preamble = lines[:4]
    reader = csv.DictReader(lines[4:])
    rows = list(reader)
    if field in VERIFY.SCIENTIFIC_FIELDS or field in VERIFY.RESOURCE_FIELDS:
        key = (
            rows[0]["distribution"], rows[0]["bodies"], rows[0]["backend"],
            rows[0]["solver"], rows[0]["theta"], rows[0]["effective_threads"],
        )
        for row in rows:
            row_key = (
                row["distribution"], row["bodies"], row["backend"],
                row["solver"], row["theta"], row["effective_threads"],
            )
            if row_key == key:
                row[field] = value
    else:
        rows[0][field] = value
    with raw.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(preamble) + "\n")
        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(run / "summary.json", RUNNER.summarize(rows))
    refresh_run_and_pointer(root)


def publisher_process(
    candidate: str,
    root: str,
    run_id: str,
    acquired,
    release,
    started,
    result_queue,
) -> None:
    try:
        started.set()

        def checkpoint(name: str) -> None:
            if name == "after_lock_acquired" and acquired is not None:
                acquired.set()
                if release is not None and not release.wait(15.0):
                    raise RuntimeError("test release event timed out")

        RUNNER.publish_versioned_run(
            Path(candidate), Path(root), run_id, checkpoint=checkpoint,
            lock_timeout_seconds=15.0,
        )
        result_queue.put((run_id, "ok"))
    except BaseException as error:  # child diagnostics must reach the parent
        result_queue.put((run_id, f"{type(error).__name__}: {error}"))


class ScratchCase(unittest.TestCase):
    def setUp(self) -> None:
        scratch = ROOT / "build" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="large-n-tools-", dir=scratch)
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def root_with_initial(self) -> Path:
        root = self.base / "evidence"
        publish_initial(root)
        return root


class PublicationProtocolTests(ScratchCase):
    def test_legacy_recovery_reports_when_no_backup_exists(self) -> None:
        root = self.base / "no-backup"
        result = RUNNER.recover_legacy_backup(root)
        self.assertEqual(result, {"status": "no_recovery_needed"})
        self.assertFalse((root / "publication.json").exists())

    def test_successive_publications_are_immutable_and_historical(self) -> None:
        root = self.root_with_initial()
        first = (root / "publication.json").read_bytes()
        second = copy_candidate(self.base / "candidate-second", "second")
        pointer = RUNNER.publish_versioned_run(second, root, "second")
        self.assertEqual(pointer["current"]["run_id"], "second")
        self.assertEqual([item["run_id"] for item in pointer["history"]], ["initial"])
        self.assertNotEqual((root / "publication.json").read_bytes(), first)
        self.assertTrue((root / "runs" / "initial" / "raw.csv").is_file())
        VERIFY.verify_evidence(root, ROOT)

    def test_live_lock_cannot_be_stolen_and_publishers_serialize(self) -> None:
        root = self.root_with_initial()
        first = copy_candidate(self.base / "candidate-a", "publisher-a")
        second = copy_candidate(self.base / "candidate-b", "publisher-b")
        context = multiprocessing.get_context("spawn")
        acquired = context.Event()
        release = context.Event()
        started_a = context.Event()
        started_b = context.Event()
        queue = context.Queue()
        process_a = context.Process(
            target=publisher_process,
            args=(str(first), str(root), "publisher-a", acquired, release, started_a, queue),
        )
        process_b = context.Process(
            target=publisher_process,
            args=(str(second), str(root), "publisher-b", None, None, started_b, queue),
        )
        process_a.start()
        self.assertTrue(acquired.wait(15.0), "first publisher never acquired the lock")
        process_b.start()
        self.assertTrue(started_b.wait(15.0), "second publisher did not start")
        with self.assertRaisesRegex(RuntimeError, "live process"):
            with RUNNER.PublicationLock(root, timeout_seconds=0.15):
                self.fail("a second process stole the live publication lock")
        release.set()
        process_a.join(20.0)
        process_b.join(20.0)
        self.assertEqual(process_a.exitcode, 0)
        self.assertEqual(process_b.exitcode, 0)
        results = {queue.get(timeout=2.0), queue.get(timeout=2.0)}
        self.assertEqual(results, {("publisher-a", "ok"), ("publisher-b", "ok")})
        inspection = VERIFY.inspect_publication(root)
        self.assertEqual(inspection["pointer"]["current"]["run_id"], "publisher-b")
        self.assertEqual(
            [item["run_id"] for item in inspection["pointer"]["history"]],
            ["publisher-a", "initial"],
        )
        VERIFY.verify_evidence(root, ROOT)

    def test_stale_lock_metadata_does_not_block_kernel_lock_recovery(self) -> None:
        root = self.base / "evidence"
        root.mkdir()
        (root / ".publication.lock").write_text(
            '{"pid":999999,"host":"stale"}\n', encoding="utf-8"
        )
        with RUNNER.PublicationLock(root, timeout_seconds=0.5):
            pass

    def test_crash_before_install_preserves_pointer_and_leaves_no_orphan(self) -> None:
        root = self.root_with_initial()
        before = (root / "publication.json").read_bytes()
        candidate = copy_candidate(self.base / "candidate-before", "before-install")

        def stop(name: str) -> None:
            if name == "before_run_install":
                raise RuntimeError("injected before install")

        with self.assertRaisesRegex(RuntimeError, "injected before install"):
            RUNNER.publish_versioned_run(candidate, root, "before-install", stop)
        self.assertEqual((root / "publication.json").read_bytes(), before)
        self.assertEqual(RUNNER.find_unreferenced_runs(root, RUNNER.read_pointer(root)), [])
        VERIFY.verify_evidence(root, ROOT)

    def test_crash_after_install_requires_explicit_orphan_recovery(self) -> None:
        root = self.root_with_initial()
        before = (root / "publication.json").read_bytes()
        candidate = copy_candidate(self.base / "candidate-after", "after-install")

        def stop(name: str) -> None:
            if name == "after_run_install":
                raise RuntimeError("injected after install")

        with self.assertRaisesRegex(RuntimeError, "injected after install"):
            RUNNER.publish_versioned_run(candidate, root, "after-install", stop)
        self.assertEqual((root / "publication.json").read_bytes(), before)
        with self.assertRaisesRegex(RuntimeError, "unreferenced"):
            VERIFY.verify_evidence(root, ROOT)
        VERIFY.verify_evidence(root, ROOT, allow_orphans=True)
        RUNNER.recover_orphan_run(root, "after-install")
        VERIFY.verify_evidence(root, ROOT)

    def test_pointer_replacement_failure_preserves_old_authority(self) -> None:
        root = self.root_with_initial()
        before = (root / "publication.json").read_bytes()
        candidate = copy_candidate(self.base / "candidate-replace", "replace-failure")
        real_replace = RUNNER.os.replace

        def fail_pointer(source, destination):
            if Path(destination).name == "publication.json":
                raise OSError("injected pointer replacement failure")
            return real_replace(source, destination)

        with mock.patch.object(RUNNER.os, "replace", side_effect=fail_pointer):
            with self.assertRaisesRegex(RuntimeError, "pointer replacement failed"):
                RUNNER.publish_versioned_run(candidate, root, "replace-failure")
        self.assertEqual((root / "publication.json").read_bytes(), before)
        VERIFY.verify_evidence(root, ROOT, allow_orphans=True)

    def test_incomplete_install_and_stranded_backup_are_rejected(self) -> None:
        root = self.root_with_initial()
        incomplete = root / "runs" / ".install-interrupted"
        incomplete.mkdir()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            VERIFY.verify_evidence(root, ROOT)
        incomplete.rmdir()
        backup = root.parent / f".{root.name}.previous-interrupted"
        backup.mkdir()
        with self.assertRaisesRegex(RuntimeError, "backup"):
            VERIFY.verify_evidence(root, ROOT)

    def test_malformed_and_incomplete_publication_pointers_fail_closed(self) -> None:
        cases = (
            {},
            {"schema": "large-n-l1-pointer-v1", "status": "complete"},
        )
        for index, value in enumerate(cases):
            with self.subTest(index=index):
                root = self.base / f"malformed-pointer-{index}"
                publish_initial(root, f"malformed-pointer-{index}")
                runs_before = {
                    path.name: (path / "run_manifest.json").read_bytes()
                    for path in (root / "runs").iterdir()
                }
                write_json(root / "publication.json", value)
                with self.assertRaises(RuntimeError):
                    VERIFY.verify_evidence(root, ROOT)
                self.assertEqual(
                    runs_before,
                    {
                        path.name: (path / "run_manifest.json").read_bytes()
                        for path in (root / "runs").iterdir()
                    },
                )

    def test_pointer_path_and_descriptor_hash_mismatches_fail_closed(self) -> None:
        for mode in ("run-id", "manifest-hash"):
            with self.subTest(mode=mode):
                root = self.base / f"pointer-mismatch-{mode}"
                publish_initial(root, f"pointer-mismatch-{mode}")
                pointer_path = root / "publication.json"
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                authoritative = pointer["current"]["run_id"]
                run_bytes = (root / "runs" / authoritative / "raw.csv").read_bytes()
                if mode == "run-id":
                    pointer["current"]["run_id"] = "different-safe-run-id"
                else:
                    pointer["current"]["run_manifest_sha256"] = "0" * 64
                write_json(pointer_path, pointer)
                with self.assertRaises(RuntimeError):
                    VERIFY.verify_evidence(root, ROOT)
                self.assertEqual(
                    (root / "runs" / authoritative / "raw.csv").read_bytes(),
                    run_bytes,
                )

    def test_legacy_backup_recovery_without_primary_is_verified_and_idempotent(self) -> None:
        root = self.base / "recovered-evidence"
        backup = create_legacy_backup(root)
        backup_bytes = {
            name: (backup / name).read_bytes()
            for name in (*RUNNER.RUN_FILES, "publication.json")
        }
        result = RUNNER.recover_legacy_backup(root)
        self.assertEqual(result["status"], "recovered")
        pointer = json.loads((root / "publication.json").read_text(encoding="utf-8"))
        self.assertEqual(pointer["current"]["run_id"], result["run_id"])
        self.assertEqual(pointer["history"], [])
        self.assertTrue((root / "runs" / result["run_id"]).is_dir())
        self.assertEqual(
            backup_bytes,
            {
                name: (backup / name).read_bytes()
                for name in (*RUNNER.RUN_FILES, "publication.json")
            },
        )
        inspection = VERIFY.inspect_publication(root)
        self.assertEqual(inspection["adopted_backups"], [backup.name])
        self.assertEqual(inspection["stranded_backups"], [])
        VERIFY.verify_evidence(root, ROOT)
        pointer_bytes = (root / "publication.json").read_bytes()
        repeated = RUNNER.recover_legacy_backup(root)
        self.assertEqual(repeated, {"status": "already_adopted", "run_id": result["run_id"]})
        self.assertEqual((root / "publication.json").read_bytes(), pointer_bytes)

    def test_legacy_backup_adoption_preserves_existing_current_authority(self) -> None:
        root = self.root_with_initial()
        current_before = json.loads((root / "publication.json").read_text(encoding="utf-8"))[
            "current"
        ]
        current_raw = (root / "runs" / current_before["run_id"] / "raw.csv").read_bytes()
        backup = create_legacy_backup(root, "historical")
        result = RUNNER.recover_legacy_backup(root)
        pointer = json.loads((root / "publication.json").read_text(encoding="utf-8"))
        self.assertEqual(pointer["current"], current_before)
        self.assertEqual(pointer["history"][-1]["run_id"], result["run_id"])
        self.assertEqual(
            (root / "runs" / current_before["run_id"] / "raw.csv").read_bytes(),
            current_raw,
        )
        self.assertTrue(backup.is_dir())
        VERIFY.verify_evidence(root, ROOT)

    def test_multiple_or_corrupt_legacy_backups_fail_without_adoption(self) -> None:
        multiple_root = self.base / "multiple-evidence"
        first = create_legacy_backup(multiple_root, "one")
        second = create_legacy_backup(multiple_root, "two")
        first_bytes = (first / "raw.csv").read_bytes()
        second_bytes = (second / "raw.csv").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            RUNNER.recover_legacy_backup(multiple_root)
        self.assertFalse((multiple_root / "publication.json").exists())
        self.assertEqual((first / "raw.csv").read_bytes(), first_bytes)
        self.assertEqual((second / "raw.csv").read_bytes(), second_bytes)

        corrupt_root = self.base / "corrupt-evidence"
        corrupt = create_legacy_backup(corrupt_root, "corrupt")
        corrupt_raw = (corrupt / "raw.csv").read_bytes()
        with (corrupt / "raw.csv").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            RUNNER.recover_legacy_backup(corrupt_root)
        self.assertFalse((corrupt_root / "publication.json").exists())
        self.assertTrue((corrupt / "raw.csv").read_bytes().startswith(corrupt_raw))

    def test_failed_recovery_keeps_backup_and_can_resume_installed_orphan(self) -> None:
        root = self.base / "interrupted-recovery"
        backup = create_legacy_backup(root)
        backup_pointer = (backup / "publication.json").read_bytes()

        def stop(name: str) -> None:
            if name == "after_recovery_install":
                raise RuntimeError("injected recovery interruption")

        with self.assertRaisesRegex(RuntimeError, "injected recovery"):
            RUNNER.recover_legacy_backup(root, checkpoint=stop)
        self.assertFalse((root / "publication.json").exists())
        self.assertEqual((backup / "publication.json").read_bytes(), backup_pointer)
        orphan_names = [
            path.name for path in (root / "runs").iterdir()
            if not path.name.startswith(".install-")
        ]
        self.assertEqual(len(orphan_names), 1)
        recovered = RUNNER.recover_legacy_backup(root)
        self.assertEqual(recovered["status"], "recovered")
        self.assertEqual(recovered["run_id"], orphan_names[0])
        VERIFY.verify_evidence(root, ROOT)

    def test_normal_pilot_refuses_unadopted_legacy_backup_before_measurement(self) -> None:
        root = self.base / "ordinary-pilot"
        backup = create_legacy_backup(root)
        backup_raw = (backup / "raw.csv").read_bytes()
        argv = [
            "run_large_n_l1_pilot.py",
            "--output", str(root),
            "--executable", sys.executable,
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            RUNNER, "execute_configuration"
        ) as execute:
            with self.assertRaisesRegex(RuntimeError, "recover-legacy-backup"):
                RUNNER.main()
        execute.assert_not_called()
        self.assertFalse((root / "publication.json").exists())
        self.assertEqual((backup / "raw.csv").read_bytes(), backup_raw)

    def test_missing_and_corrupt_authoritative_run_fail_closed(self) -> None:
        for mode in ("missing", "corrupt"):
            with self.subTest(mode=mode):
                root = self.base / f"evidence-{mode}"
                publish_initial(root, f"initial-{mode}")
                run = root / "runs" / f"initial-{mode}"
                if mode == "missing":
                    (run / "raw.csv").unlink()
                else:
                    with (run / "raw.csv").open("ab") as stream:
                        stream.write(b"corruption")
                with self.assertRaisesRegex(RuntimeError, "missing|mismatch"):
                    VERIFY.verify_evidence(root, ROOT)

    def test_legacy_flat_layout_migrates_without_byte_loss(self) -> None:
        root = self.base / "legacy"
        root.mkdir()
        source = authoritative_run()
        for name in RUNNER.RUN_FILES:
            shutil.copy2(source / name, root / name)
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.pop("run_id", None)
        write_json(metadata_path, metadata)
        legacy_pointer = {
            "schema": "large-n-l1-publication-v1",
            "status": "complete",
            "files": {name: RUNNER.sha256(root / name) for name in RUNNER.RUN_FILES},
        }
        write_json(root / "publication.json", legacy_pointer)
        original = {name: (root / name).read_bytes() for name in RUNNER.RUN_FILES}
        original_pointer = (root / "publication.json").read_bytes()
        with RUNNER.PublicationLock(root):
            pointer = RUNNER.migrate_legacy_layout_locked(root)
        migrated = root / "runs" / pointer["current"]["run_id"]
        for name, value in original.items():
            self.assertEqual((migrated / name).read_bytes(), value)
            self.assertFalse((root / name).exists())
        self.assertEqual(
            (migrated / RUNNER.LEGACY_PUBLICATION_FILE).read_bytes(), original_pointer
        )
        VERIFY.verify_evidence(root, ROOT)


class SemanticVerifierTests(ScratchCase):
    def root_with_mutable_copy(self) -> Path:
        return self.root_with_initial()

    def test_legacy_verification_mode_cannot_bypass_modern_source_custody(self) -> None:
        root = self.root_with_mutable_copy()
        run = VERIFY.inspect_publication(root)["current_run"]
        with self.assertRaisesRegex(RuntimeError, "legacy publication custody"):
            VERIFY.verify_scientific_run(
                run, ROOT, verify_source=False, allow_legacy=True
            )

    def test_negative_scientific_timing_and_resource_values_are_rejected(self) -> None:
        cases = (
            ("complete_force_s", "-1"),
            ("p95_relative_error", "-1"),
            ("candidate_relative_energy_drift", "-1"),
            ("solver_memory_bytes", "-1"),
        )
        for index, (field, value) in enumerate(cases):
            with self.subTest(field=field):
                root = self.base / f"negative-{index}"
                publish_initial(root, f"negative-{index}")
                mutate_raw_field(root, field, value)
                with self.assertRaisesRegex(RuntimeError, "negative"):
                    VERIFY.verify_evidence(root, ROOT)

    def test_zero_relative_floor_and_zero_force_duration_are_rejected(self) -> None:
        for index, field in enumerate(("relative_floor_m_s2", "complete_force_s")):
            with self.subTest(field=field):
                root = self.base / f"zero-{index}"
                publish_initial(root, f"zero-{index}")
                mutate_raw_field(root, field, "0")
                with self.assertRaisesRegex(RuntimeError, "must be positive"):
                    VERIFY.verify_evidence(root, ROOT)

    def test_coherently_rehashed_but_false_summary_is_rejected(self) -> None:
        root = self.root_with_mutable_copy()
        run = VERIFY.inspect_publication(root)["current_run"]
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        summary["groups"][0]["medians"]["complete_force_s"] += 1.0
        write_json(run / "summary.json", summary)
        refresh_run_and_pointer(root)
        with self.assertRaisesRegex(RuntimeError, "summary does not reconstruct"):
            VERIFY.verify_evidence(root, ROOT)

    def test_matrix_and_repetition_falsification_are_rejected(self) -> None:
        root = self.root_with_mutable_copy()
        run = VERIFY.inspect_publication(root)["current_run"]
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        metadata["matrix"]["body_counts"] = [256, 1024]
        write_json(run / "metadata.json", metadata)
        refresh_run_and_pointer(root)
        with self.assertRaisesRegex(RuntimeError, "matrix"):
            VERIFY.verify_evidence(root, ROOT)


class RunnerFailureTests(ScratchCase):
    def test_timeout_malformed_output_and_provenance_failure_do_not_publish(self) -> None:
        # These are bounded unit controls of the measurement pipeline. They do
        # not execute the 144-row pilot.
        root = self.root_with_initial()
        before_pointer = (root / "publication.json").read_bytes()
        before_runs = sorted(path.name for path in (root / "runs").iterdir())
        command = [str(sys.executable)]
        with self.assertRaises(subprocess.TimeoutExpired):
            with mock.patch.object(
                RUNNER.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(command, 0.01),
            ):
                RUNNER.execute_configuration(command, 0.01)
        malformed = self.base / "malformed.csv"
        malformed.write_text("not,the,expected,output\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            RUNNER.data_lines(malformed)
        with mock.patch.object(
            RUNNER.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["capture-provenance"]),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                RUNNER.subprocess.run(["capture-provenance"], check=True)
        self.assertEqual((root / "publication.json").read_bytes(), before_pointer)
        self.assertEqual(sorted(path.name for path in (root / "runs").iterdir()), before_runs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
