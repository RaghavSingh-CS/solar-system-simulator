#!/usr/bin/env python3
"""Focused adversarial tests for Large-N L3 performance-only evidence."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_large_n_l1_pilot as l1  # noqa: E402
import run_large_n_l3_pilot as l3  # noqa: E402
import verify_large_n_l1_evidence as verify_l1  # noqa: E402
import verify_large_n_l2_evidence as verify_l2  # noqa: E402
import verify_large_n_l3_evidence as verify_l3  # noqa: E402


EVIDENCE = ROOT / "data" / "benchmarks" / "large_n_l3_pilot"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


class LargeNL3Tests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = ROOT / "build" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="large-n-l3-test-", dir=scratch)
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def copy_evidence(self) -> Path:
        if not EVIDENCE.is_dir():
            self.skipTest("no completed L3 publication exists after the bounded timeout")
        destination = self.base / "evidence"
        shutil.copytree(EVIDENCE, destination, ignore=shutil.ignore_patterns(".publication.lock"))
        return destination

    def raw_parts(self, root: Path) -> tuple[Path, list[str], list[str], list[dict[str, str]]]:
        run = verify_l3.inspect_publication(root)["current_run"]
        raw = run / "raw.csv"
        lines = raw.read_text(encoding="utf-8").splitlines()
        comments = [line for line in lines if line.startswith("#")]
        reader = csv.DictReader(line for line in lines if line and not line.startswith("#"))
        return raw, comments, list(reader.fieldnames or []), list(reader)

    def write_rows(self, raw: Path, comments: list[str], fields: list[str], rows: list[dict[str, str]]) -> None:
        with raw.open("w", encoding="utf-8", newline="") as stream:
            for line in comments:
                stream.write(line + "\n")
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def refresh(self, root: Path, *, reconstruct: bool = True) -> Path:
        pointer_path = root / "publication.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run = root / "runs" / pointer["current"]["run_id"]
        rows = verify_l3.read_rows(run / "raw.csv")
        if reconstruct:
            protocol = verify_l3.load_protocol(run / "protocol.json")
            write_json(run / "summary.json", verify_l3.reconstruct_summary(rows, protocol))
        metadata_path = run / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["outputs"]["raw_csv_sha256"] = l1.sha256(run / "raw.csv")
        metadata["outputs"]["summary_json_sha256"] = l1.sha256(run / "summary.json")
        metadata["outputs"]["raw_rows"] = len(rows)
        metadata["outputs"]["status_counts"] = {
            "completed": sum(row["status"] == "completed" for row in rows),
            "skipped": sum(row["status"] == "skipped" for row in rows),
        }
        if reconstruct:
            metadata["checkpoint_decision"] = json.loads((run / "summary.json").read_text(encoding="utf-8"))["checkpoint_decision"]
        write_json(metadata_path, metadata)
        l3.write_run_manifest(run, run.name)
        pointer["current"] = l3.descriptor(run)
        write_json(pointer_path, pointer)
        return run

    def mutate_row(self, root: Path, predicate, field: str, value: str, *, reconstruct: bool = True) -> None:
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

    def assert_rejected(self, root: Path, pattern: str) -> None:
        with self.assertRaisesRegex(RuntimeError, pattern):
            verify_l3.verify_evidence(root, ROOT)

    def test_protocol_is_bounded_representative_16384_only(self) -> None:
        protocol = l3.load_protocol()
        configurations = l3.configurations(protocol)
        self.assertEqual(len(configurations), 6)
        self.assertEqual({item["bodies"] for item in configurations}, {16384})
        self.assertEqual({item["distribution"] for item in configurations}, {"uniform", "clustered", "planar"})
        self.assertEqual({item["theta"] for item in configurations}, {0.25, 0.5, 0.8})
        self.assertEqual({item["backend"] for item in configurations}, {"sequential", "openmp"})
        self.assertTrue(all(item["solver"] == "barnes-hut" and not item["request_direct_reference"] for item in configurations))
        self.assertNotIn(100000, protocol["candidate_body_counts"])

    def test_commands_disable_direct_and_bind_all_budgets(self) -> None:
        protocol = l3.load_protocol()
        command = l3.exact_command(l3.configurations(protocol)[0], protocol, Path("lab.exe"), Path("row.csv"))
        self.assertEqual(command[command.index("--solver") + 1], "barnes-hut")
        self.assertEqual(command[command.index("--direct-reference") + 1], "false")
        self.assertEqual(command[command.index("--max-direct-reference-bodies") + 1], "0")
        self.assertIn("--maximum-traversal-work", command)
        self.assertIn("--solver-memory-budget-bytes", command)

    def test_no_reference_placeholders_are_validated_then_removed(self) -> None:
        row = {
            "direct_reference_available": "0",
            "direct_reference_reason": "direct reference disabled by configuration",
            "estimated_direct_interactions": "0",
            **{field: "0" for field in l3.OMITTED_WORK_FIELDS},
            **{field: "0" for field in l3.SCIENTIFIC_FIELDS},
        }
        l3.normalize_performance_only_row(row)
        self.assertTrue(all(row[field] == "" for field in l3.SCIENTIFIC_FIELDS))
        forged = dict(row)
        forged.update({field: "0" for field in l3.SCIENTIFIC_FIELDS})
        forged["max_relative_error"] = "1"
        with self.assertRaisesRegex(RuntimeError, "placeholders changed"):
            l3.normalize_performance_only_row(forged)

    def test_real_bounded_execution_omits_direct_and_diagnostics(self) -> None:
        executable = ROOT / "build" / "openmp-release" / "nbody_large_n_lab.exe"
        if not executable.is_file():
            self.skipTest("canonical OpenMP laboratory executable is unavailable")
        output = self.base / "bounded.csv"
        protocol = l3.load_protocol()
        configuration = dict(l3.configurations(protocol)[0])
        configuration.update({"bodies": 32, "seed": 0x4C3354455354, "theta": 0.5})
        command = l3.exact_command(configuration, protocol, executable, output)
        command[command.index("--warmups") + 1] = "0"
        command[command.index("--repetitions") + 1] = "1"
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10.0)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        with output.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(line for line in stream if line.strip() and not line.startswith("#")))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "completed")
        self.assertTrue(all(row[field] == "0" for field in l3.OMITTED_WORK_FIELDS))
        self.assertEqual(row["estimated_direct_interactions"], "0")
        self.assertGreater(int(row["node_count"]), 0)
        l3.normalize_performance_only_row(row)
        self.assertTrue(all(row[field] == "" for field in l3.SCIENTIFIC_FIELDS))

    def test_reuses_reviewed_publication_primitives(self) -> None:
        self.assertIs(l3.publication_core.PublicationLock, l1.PublicationLock)
        self.assertIs(l3.publication_core.atomic_replace_json, l1.atomic_replace_json)

    def test_timeout_discards_partial_staging_without_publication(self) -> None:
        executable = self.base / "lab.exe"
        executable.write_bytes(b"test placeholder")
        output = self.base / "publication"
        scratch = ROOT / "build" / "scratch"
        before = {path.resolve() for path in scratch.glob("large-n-l3-run-*")}

        def timeout_after_partial_output(command, **kwargs):
            path = Path(command[command.index("--output") + 1])
            path.write_text("partial,untrusted\n", encoding="utf-8", newline="\n")
            self.assertEqual(kwargs["timeout"], 70.0)
            self.assertFalse(kwargs["check"])
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        arguments = SimpleNamespace(
            executable=executable,
            output=output,
            protocol=l3.DEFAULT_PROTOCOL,
        )
        with mock.patch.object(l3, "parse_args", return_value=arguments), mock.patch.object(
            l3.subprocess, "run", side_effect=timeout_after_partial_output
        ):
            with self.assertRaisesRegex(RuntimeError, "configuration 0 exceeded its hard timeout"):
                l3.main()

        self.assertFalse(output.exists())
        self.assertEqual(
            {path.resolve() for path in scratch.glob("large-n-l3-run-*")},
            before,
        )

    def test_authoritative_evidence_and_historical_evidence_verify(self) -> None:
        if not EVIDENCE.is_dir():
            self.skipTest("no completed L3 publication exists after the bounded timeout")
        metadata = verify_l3.verify_evidence(EVIDENCE, ROOT)
        self.assertEqual(metadata["outputs"]["status_counts"], {"completed": 18, "skipped": 0})
        self.assertEqual(verify_l1.verify_evidence()["outputs"]["raw_rows"], 144)
        self.assertEqual(verify_l2.verify_evidence()["outputs"]["raw_rows"], 60)

    def test_authoritative_rows_have_explicitly_unavailable_science(self) -> None:
        if not EVIDENCE.is_dir():
            self.skipTest("no completed L3 publication exists after the bounded timeout")
        rows = verify_l3.read_rows(verify_l3.inspect_publication(EVIDENCE)["current_run"] / "raw.csv")
        self.assertEqual(len(rows), 18)
        for row in rows:
            self.assertEqual(row["reference_mode"], "none")
            self.assertEqual(row["accuracy_status"], "unavailable_performance_only")
            self.assertEqual(row["conservation_status"], "unavailable_performance_only")
            self.assertTrue(all(row[field] == "" for field in verify_l3.SCIENTIFIC_FIELDS))
            self.assertTrue(all(row[field] == "0" for field in verify_l3.OMITTED_WORK_FIELDS))
            self.assertEqual(row["estimated_direct_interactions"], "0")

    def test_missing_publication_fails_closed(self) -> None:
        missing = self.base / "missing-evidence"
        missing.mkdir()
        with self.assertRaisesRegex(RuntimeError, "missing L3 publication pointer"):
            verify_l3.verify_evidence(missing, ROOT)

    def test_coherently_rehashed_body_count_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: True, "bodies", "32768")
        self.assert_rejected(root, "identity differs")

    def test_coherently_rehashed_seed_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: True, "seed", "1")
        self.assert_rejected(root, "identity differs")

    def test_coherently_rehashed_backend_or_theta_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: True, "theta", "0.5")
        self.assert_rejected(root, "solver identity differs")

    def test_coherently_rehashed_worker_count_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: row["backend"] == "openmp", "observed_worker_threads", "3")
        self.assert_rejected(root, "worker provenance differs")

    def test_direct_reference_claim_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: True, "direct_reference_available", "1")
        self.assert_rejected(root, "Direct-reference status differs")

    def test_fabricated_zero_accuracy_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: True, "max_abs_acceleration_error_m_s2", "0")
        self.assert_rejected(root, "fabricates unavailable science")

    def test_hidden_diagnostic_timing_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: True, "diagnostics_s", "0.1")
        self.assert_rejected(root, "Direct/diagnostic timing")

    def test_hidden_direct_work_is_rejected(self) -> None:
        root = self.copy_evidence()
        self.mutate_row(root, lambda row: True, "estimated_direct_interactions", "1")
        self.assert_rejected(root, r"charged O\(N\^2\) work")

    def test_missing_row_is_rejected(self) -> None:
        root = self.copy_evidence()
        raw, comments, fields, rows = self.raw_parts(root)
        rows.pop()
        self.write_rows(raw, comments, fields, rows)
        self.refresh(root)
        self.assert_rejected(root, "exactly 18 rows")

    def test_forged_command_enabling_reference_is_rejected(self) -> None:
        root = self.copy_evidence()
        run = verify_l3.inspect_publication(root)["current_run"]
        metadata_path = run / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        command = metadata["commands"]["configuration_argv"][0]
        command[command.index("--direct-reference") + 1] = "true"
        write_json(metadata_path, metadata)
        self.refresh(root)
        self.assert_rejected(root, "command arguments differ")

    def test_forged_summary_is_rejected(self) -> None:
        root = self.copy_evidence()
        run = verify_l3.inspect_publication(root)["current_run"]
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["groups"][0]["medians"]["complete_force_s"] += 1.0
        write_json(summary_path, summary)
        self.refresh(root, reconstruct=False)
        self.assert_rejected(root, "summary does not reconstruct")

    def test_pointer_tampering_fails_closed(self) -> None:
        root = self.copy_evidence()
        pointer_path = root / "publication.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        pointer["current"]["run_manifest_sha256"] = "0" * 64
        write_json(pointer_path, pointer)
        self.assert_rejected(root, "descriptor mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
