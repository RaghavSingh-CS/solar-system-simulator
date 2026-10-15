#!/usr/bin/env python3
"""Focused adversarial tests for the Large-N L2 protocol and evidence."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_large_n_l1_pilot as l1  # noqa: E402
import run_large_n_l2_pilot as l2  # noqa: E402
import verify_large_n_l1_evidence as verify_l1  # noqa: E402
import verify_large_n_l2_evidence as verify_l2  # noqa: E402


EVIDENCE = ROOT / "data" / "benchmarks" / "large_n_l2_pilot"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


class LargeNL2Tests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = ROOT / "build" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="large-n-l2-test-", dir=scratch)
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def copy_evidence(self) -> Path:
        destination = self.base / "evidence"
        shutil.copytree(EVIDENCE, destination, ignore=shutil.ignore_patterns(".publication.lock"))
        return destination

    def refresh(self, root: Path, *, reconstruct: bool = False) -> Path:
        pointer_path = root / "publication.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run = root / "runs" / pointer["current"]["run_id"]
        if reconstruct:
            rows = verify_l2.read_rows(run / "raw.csv")
            protocol = l2.load_protocol(run / "protocol.json")
            summary = l2.reconstruct_summary(rows, protocol)
            write_json(run / "summary.json", summary)
        metadata_path = run / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["outputs"]["raw_csv_sha256"] = l1.sha256(run / "raw.csv")
        metadata["outputs"]["summary_json_sha256"] = l1.sha256(run / "summary.json")
        if reconstruct:
            metadata["outputs"]["raw_rows"] = len(rows)
            metadata["outputs"]["status_counts"] = {
                status: sum(row["status"] == status for row in rows)
                for status in ("completed", "skipped")
            }
            metadata["pilot_decision"] = summary["pilot_decision"]
        write_json(metadata_path, metadata)
        l2.write_run_manifest(run, run.name)
        pointer["current"] = l2.descriptor(run)
        write_json(pointer_path, pointer)
        return run

    def raw_parts(self, root: Path) -> tuple[Path, list[str], list[str], list[dict[str, str]]]:
        run = verify_l2.inspect_publication(root)["current_run"]
        raw = run / "raw.csv"
        lines = raw.read_text(encoding="utf-8").splitlines()
        comments = [line for line in lines if line.startswith("#")]
        reader = csv.DictReader(line for line in lines if line and not line.startswith("#"))
        return raw, comments, list(reader.fieldnames or []), list(reader)

    def write_rows(
        self, raw: Path, comments: list[str], fields: list[str], rows: list[dict[str, str]]
    ) -> None:
        with raw.open("w", encoding="utf-8", newline="") as stream:
            for line in comments:
                stream.write(line + "\n")
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def mutate_row(
        self, root: Path, predicate, field: str, value: str, *, reconstruct: bool = True
    ) -> None:
        raw, comments, fields, rows = self.raw_parts(root)
        changed = False
        for row in rows:
            if predicate(row):
                row[field] = value
                changed = True
                break
        self.assertTrue(changed)
        self.write_rows(raw, comments, fields, rows)
        self.refresh(root, reconstruct=reconstruct)

    def mutate_metadata(self, root: Path, mutation) -> None:
        run = verify_l2.inspect_publication(root)["current_run"]
        path = run / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        mutation(metadata)
        write_json(path, metadata)
        self.refresh(root)

    def assert_semantic_rejection(self, root: Path, pattern: str) -> None:
        with self.assertRaisesRegex(RuntimeError, pattern):
            verify_l2.verify_evidence(root, ROOT)

    def test_protocol_matrix_is_exact_and_bounded_to_8192(self) -> None:
        protocol = l2.load_protocol()
        configurations = l2.configurations(protocol)
        self.assertEqual(len(configurations), 24)
        self.assertEqual({item["bodies"] for item in configurations}, {8192})
        self.assertEqual(sum(item["solver"] == "direct" for item in configurations), 6)
        self.assertEqual(sum(item["solver"] == "barnes-hut" for item in configurations), 18)
        self.assertTrue(all(not item["request_direct_reference"] for item in configurations if item["solver"] == "direct"))
        self.assertTrue(all(item["reference_mode"] == "full" for item in configurations if item["solver"] == "barnes-hut"))
        self.assertEqual(l2.csv_float(0.8), "0.80000000000000004")

    def test_l1_historical_source_custody_survives_l2_development(self) -> None:
        metadata = verify_l1.verify_evidence()
        self.assertEqual(metadata["outputs"]["raw_rows"], 144)
        self.assertEqual(verify_l1.SOURCE_CUSTODY_COMMIT, "3f665c2610f3d65214494ff18f2f06d3c063513a")

    def test_l2_reuses_reviewed_lock_and_atomic_pointer_primitives(self) -> None:
        self.assertIs(l2.publication_core.PublicationLock, l1.PublicationLock)
        self.assertIs(l2.publication_core.atomic_replace_json, l1.atomic_replace_json)

    def test_authoritative_evidence_verifies_and_has_expected_decision(self) -> None:
        metadata = verify_l2.verify_evidence(EVIDENCE, ROOT)
        self.assertEqual(metadata["outputs"]["status_counts"], {"completed": 54, "skipped": 6})
        summary = json.loads((verify_l2.inspect_publication(EVIDENCE)["current_run"] / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summary["groups"]), 24)
        self.assertEqual(summary["pilot_decision"], metadata["pilot_decision"])

    def test_sampled_label_cannot_masquerade_as_full_reference(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "barnes-hut", "reference_mode", "sampled")
        self.assert_semantic_rejection(root, "wrong reference_mode")

    def test_direct_admission_cannot_replace_required_skip(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(
            root, lambda row: row["solver"] == "direct", "status", "completed",
            reconstruct=False,
        )
        self.assert_semantic_rejection(root, "row sequence differs")

    def test_coherently_rehashed_changed_seed_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "barnes-hut", "seed", "1")
        self.assert_semantic_rejection(root, "row sequence differs")

    def test_coherently_rehashed_changed_requested_workers_are_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["backend"] == "openmp" and row["solver"] == "barnes-hut", "requested_threads", "3")
        self.assert_semantic_rejection(root, "row sequence differs")

    def test_coherently_rehashed_changed_effective_workers_are_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["backend"] == "openmp" and row["solver"] == "barnes-hut", "effective_threads", "3")
        self.assert_semantic_rejection(root, "row sequence differs")

    def test_coherently_rehashed_changed_observed_workers_are_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["backend"] == "openmp" and row["solver"] == "barnes-hut", "observed_worker_threads", "3")
        self.assert_semantic_rejection(root, "worker provenance")

    def test_coherently_rehashed_changed_wall_budget_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "barnes-hut", "wall_time_budget_s", "31")
        self.assert_semantic_rejection(root, "wrong wall_time_budget_s")

    def test_coherently_rehashed_changed_repetition_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "barnes-hut" and row["repetition"] == "1", "repetition", "7")
        self.assert_semantic_rejection(root, "row sequence differs")

    def test_coherently_rehashed_reference_reason_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "barnes-hut", "direct_reference_reason", "forged")
        self.assert_semantic_rejection(root, "wrong direct_reference_reason")

    def test_coherently_rehashed_direct_skip_timing_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "direct", "complete_force_s", "0.5")
        self.assert_semantic_rejection(root, "wrong complete_force_s")

    def test_coherently_rehashed_direct_skip_science_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "direct", "max_abs_acceleration_error_m_s2", "0.5")
        self.assert_semantic_rejection(root, "wrong max_abs_acceleration_error_m_s2")

    def test_coherently_rehashed_direct_skip_reference_claim_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "direct", "direct_reference_available", "1")
        self.assert_semantic_rejection(root, "wrong direct_reference_available")

    def test_coherently_rehashed_missing_configuration_is_rejected(self) -> None:
        root = self.copy_evidence()
        raw, comments, fields, rows = self.raw_parts(root)
        rows.pop(1)
        self.write_rows(raw, comments, fields, rows)
        self.refresh(root, reconstruct=True)
        self.assert_semantic_rejection(root, "exactly 60 retained rows")

    def test_coherently_rehashed_duplicate_configuration_is_rejected(self) -> None:
        root = self.copy_evidence()
        raw, comments, fields, rows = self.raw_parts(root)
        rows.append(dict(rows[0]))
        self.write_rows(raw, comments, fields, rows)
        self.refresh(root, reconstruct=True)
        self.assert_semantic_rejection(root, "exactly 60 retained rows")

    def test_coherently_rehashed_extra_configuration_is_rejected(self) -> None:
        root = self.copy_evidence()
        raw, comments, fields, rows = self.raw_parts(root)
        extra = dict(rows[-1])
        extra["distribution"] = "forged"
        rows.append(extra)
        self.write_rows(raw, comments, fields, rows)
        self.refresh(root, reconstruct=True)
        self.assert_semantic_rejection(root, "exactly 60 retained rows")

    def test_coherently_rehashed_forged_metadata_matrix_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_metadata(root, lambda metadata: metadata["matrix"].update({"retained_repetitions": 4}))
        self.assert_semantic_rejection(root, "metadata matrix differs")

    def test_coherently_rehashed_forged_command_arguments_are_rejected(self) -> None:
        root = self.copy_evidence()
        def forge(metadata):
            command = metadata["commands"]["configuration_argv"][0]
            command[command.index("--seed") + 1] = "1"
        self.mutate_metadata(root, forge)
        self.assert_semantic_rejection(root, "command arguments differ")

    def test_coherently_rehashed_changed_admission_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "direct", "estimated_direct_interactions", "500000001")
        self.assert_semantic_rejection(root, "admission fields differ")

    def test_coherently_rehashed_changed_status_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["solver"] == "barnes-hut", "status", "skipped")
        self.assert_semantic_rejection(root, "status counts are wrong")

    def test_coherently_rehashed_false_summary_is_rejected(self) -> None:
        root = self.copy_evidence()
        run = verify_l2.inspect_publication(root)["current_run"]
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        completed = next(item for item in summary["groups"] if item["status"] == "completed")
        completed["medians"]["complete_force_s"] += 1.0
        write_json(run / "summary.json", summary)
        self.refresh(root)
        with self.assertRaisesRegex(RuntimeError, "summary does not reconstruct"):
            verify_l2.verify_evidence(root, ROOT)

    def test_pointer_descriptor_tampering_fails_closed(self) -> None:
        root = self.copy_evidence()
        pointer = json.loads((root / "publication.json").read_text(encoding="utf-8"))
        pointer["current"]["run_manifest_sha256"] = "0" * 64
        write_json(root / "publication.json", pointer)
        with self.assertRaisesRegex(RuntimeError, "descriptor mismatch"):
            verify_l2.verify_evidence(root, ROOT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
