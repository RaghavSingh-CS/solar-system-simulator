#!/usr/bin/env python3
"""Adversarial unit and synthetic-evidence tests for Stage 5 v2.0."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_stage5_policy as POLICY  # noqa: E402
import create_stage5_deployment_packet as PACKET  # noqa: E402
import evaluate_stage5_policy as EVALUATE  # noqa: E402
import prepare_stage5_build as PREPARE  # noqa: E402
import run_stage5_v2 as RUNNER  # noqa: E402
import stage5_common as COMMON  # noqa: E402
import verify_stage5_v2 as VERIFIER  # noqa: E402


SOURCE_SHA = "a" * 40
ENVIRONMENT_ID = "synthetic-rtx4050"
THREADS = (1, 2)
ENVIRONMENT = {
    "environment_id": ENVIRONMENT_ID, "fixture_mode": False,
    "platform": {"system": "Windows", "machine": "AMD64"},
    "topology": {"physical_cores": 2, "logical_processors": 2, "openmp_threads": list(THREADS)},
}
ENVIRONMENT_HASH = COMMON.canonical_sha256(ENVIRONMENT)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_info(component: str) -> dict[str, object]:
    value: dict[str, object] = {
        "component": component,
        "source_commit": SOURCE_SHA,
        "protocol_version": COMMON.PROTOCOL_VERSION,
        "build_configuration": "Release",
    }
    if component == "nbody_benchmark":
        value.update(
            cxx_compiler=str(ROOT / "fake-cl.exe"), compiler_id="MSVC",
            compiler_version="19.44", generator="Ninja", openmp_flags="/openmp",
            openmp_libraries="vcomp", openmp_compiled=True, openmp_specification_date=200203,
        )
    if component == "nbody_cuda_benchmark":
        value.update(
            cxx_compiler=str(ROOT / "fake-cl.exe"), compiler_id="MSVC",
            compiler_version="19.44", generator="Ninja",
            cuda_compiler=str(ROOT / "fake-nvcc.exe"), cuda_compiler_id="NVIDIA",
            cuda_compiler_version="12.8", cuda_host_compiler=str(ROOT / "fake-cl.exe"),
            cuda_architectures="89",
        )
    return value


def make_build_bundle(root: Path) -> tuple[Path, dict[str, object]]:
    bundle = root / "build-evidence"
    bundle.mkdir(parents=True)
    for name, text in (
        ("configure.txt", "configured Release Ninja\n"),
        ("build.txt", "build complete\n"),
        ("ctest.txt", "100% tests passed, 0 tests failed out of 7\n"),
    ):
        (bundle / name).write_text(text, encoding="utf-8")
    names = {
        "nbody_benchmark", "nbody_tests", "nbody_solar_system_tests",
        "nbody_openmp_tests", "nbody_cuda_benchmark", "nbody_cuda_tests",
    }
    binaries: dict[str, object] = {}
    for name in names:
        path = bundle / f"{name}.exe"
        path.write_bytes(f"synthetic {name}\n".encode())
        binaries[name] = {
            "path": str(path.resolve()), "sha256": COMMON.sha256_file(path),
            "bytes": path.stat().st_size, "build_info": build_info(name),
        }
    environment = copy.deepcopy(ENVIRONMENT)
    manifest: dict[str, object] = {
        "protocol_version": COMMON.PROTOCOL_VERSION, "source_commit": SOURCE_SHA,
        "source_directory": str(ROOT.resolve()), "source_clean": True,
        "environment_id": ENVIRONMENT_ID, "environment_sha256": ENVIRONMENT_HASH,
        "environment": environment, "fixture_mode": False,
        "ctest": {
            "return_code": 0, "passed": True,
            "log_sha256": COMMON.sha256_file(bundle / "ctest.txt"),
            "log_bytes": (bundle / "ctest.txt").stat().st_size,
        },
        "cmake_cache": {
            "CMAKE_HOME_DIRECTORY": str(ROOT.resolve()), "CMAKE_GENERATOR": "Ninja",
            "CMAKE_BUILD_TYPE": "Release", "NBODY_ENABLE_OPENMP": "ON",
            "NBODY_ENABLE_CUDA": "ON", "NBODY_ENABLE_MPI": "OFF",
            "NBODY_BUILD_VISUALIZATION": "OFF", "NBODY_STAGE5_SOURCE_SHA": SOURCE_SHA,
            "NBODY_STAGE5_PROTOCOL_VERSION": COMMON.PROTOCOL_VERSION,
            "CMAKE_CUDA_ARCHITECTURES": "89",
        },
        "compiler_identity": build_info("nbody_benchmark"),
        "cuda_compiler_identity": build_info("nbody_cuda_benchmark"),
        "openmp_topology": {
            "physical_cores": 2, "logical_processors": 2,
            "derivation": "sorted unique {1, powers of two below P, P, L}, capped at L",
            "threads": list(THREADS),
        },
        "binaries": binaries,
        "tool_scripts": {
            name: COMMON.sha256_file(ROOT / "scripts" / name)
            for name in (
                "stage5_common.py", "run_stage5_v2.py", "verify_stage5_v2.py",
                "build_stage5_policy.py", "evaluate_stage5_policy.py",
            )
        },
        "files": {
            name: {"sha256": COMMON.sha256_file(bundle / name), "bytes": (bundle / name).stat().st_size}
            for name in ("configure.txt", "build.txt", "ctest.txt")
        },
    }
    path = bundle / "build_manifest.json"
    COMMON.write_json(path, manifest)
    return path, manifest


def elapsed_values(backend: str, threads: int, bodies: int, repetition: int, scale: float) -> tuple[tuple[str, float], ...]:
    base = scale * (bodies / 64.0) ** 2 * (1.0 + repetition * 0.001)
    if backend == "cuda":
        return (("device_compute", base * 0.45 + 0.4),
                ("transfer_inclusive", base * 0.55 + 0.5),
                ("selector_steady_state", base * 0.60 + 0.6))
    speed = 1.0 if backend == "sequential" else 0.85 / threads
    return (("integration_compute", base * speed + 0.2),
            ("selector_steady_state", base * speed + 0.3))


def make_evidence(root: Path, mode: str, *, scale: float = 1.0) -> Path:
    evidence = root / f"{mode}-{scale:g}"
    evidence.mkdir()
    build_path, build = make_build_bundle(root / f"bundle-{mode}-{scale:g}")
    build_hash = COMMON.sha256_file(build_path)
    schedule = COMMON.ordered_runs(mode, THREADS, COMMON.MODE_REPETITIONS[mode], COMMON.DEFAULT_ORDER_SEED)
    rows: list[dict[str, str]] = []
    for order_index, config in enumerate(schedule, start=1):
        backend, threads = str(config["backend"]), int(config["threads"])
        bodies, repetition = int(config["bodies"]), int(config["repetition"])
        common = {
            "environment_id": ENVIRONMENT_ID, "environment_sha256": ENVIRONMENT_HASH,
            "source_commit": SOURCE_SHA, "build_manifest_sha256": build_hash,
            "protocol_version": COMMON.PROTOCOL_VERSION, "provenance_mode": "source-build-test-bound",
            "experiment_id": f"synthetic-{mode}", "mode": mode, "dataset_role": COMMON.MODE_ROLE[mode],
            "run_id": f"b{repetition:02d}-o{order_index:04d}", "order_seed": str(COMMON.DEFAULT_ORDER_SEED),
            "block_id": str(repetition), "order_index": str(order_index), "repetition": str(repetition),
            "backend": backend,
            "requested_threads": str(threads if backend == "openmp" else 0 if backend == "cuda" else 1),
            "effective_threads": str(threads if backend == "openmp" else 0 if backend == "cuda" else 1),
            "bodies": str(bodies), "steps": str(COMMON.MODE_STEPS[mode]), "dt_s": "10",
            "benchmark_seed": str(COMMON.DEFAULT_BENCHMARK_SEED), "warmups": "1",
            "correctness_status": "pass",
            "checksum": f"cuda-{bodies}" if backend == "cuda" else f"cpu-{bodies}",
            "relative_energy_drift": "1e-12", "relative_angular_momentum_drift": "1e-13",
            "energy_drift_bound": "5e-6", "angular_momentum_drift_bound": "5e-10",
            "build_configuration": "Release",
            "openmp_specification_date": "200203" if backend == "openmp" else "0",
            "setup_allocation_ms": "0.1" if backend == "cuda" else "",
            "host_staging_ms": "0.1" if backend == "cuda" else "",
            "host_to_device_ms": "0.1" if backend == "cuda" else "",
            "device_to_host_ms": "0.1" if backend == "cuda" else "",
            "device_index": "0" if backend == "cuda" else "",
            "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU" if backend == "cuda" else "",
            "compute_capability": "8.9" if backend == "cuda" else "",
            "total_global_memory_bytes": "6442450944" if backend == "cuda" else "",
            "multiprocessor_count": "20" if backend == "cuda" else "",
            "cuda_driver_version": "13000" if backend == "cuda" else "",
            "cuda_runtime_version": "12080" if backend == "cuda" else "",
        }
        for metric, elapsed in elapsed_values(backend, threads, bodies, repetition, scale):
            rows.append({**common, "timing_metric": metric, "elapsed_ms": f"{elapsed:.17g}"})
    write_csv(evidence / "raw_timings.csv", RUNNER.RAW_FIELDS, rows)
    write_csv(evidence / "summary.csv", RUNNER.SUMMARY_FIELDS,
              VERIFIER.summary_rows(rows, mode, COMMON.DEFAULT_ORDER_SEED, 20))
    environment = {
        "protocol_version": COMMON.PROTOCOL_VERSION, "mode": mode,
        "dataset_role": COMMON.MODE_ROLE[mode], "source_commit": SOURCE_SHA,
        "build_manifest_sha256": build_hash, "environment_id": ENVIRONMENT_ID,
        "environment_sha256": ENVIRONMENT_HASH, "openmp_threads": list(THREADS),
        "topology": build["openmp_topology"], "measured_bodies": list(COMMON.MODE_BODIES[mode]),
        "steps": COMMON.MODE_STEPS[mode], "dt_s": 10.0,
        "repetitions": COMMON.MODE_REPETITIONS[mode], "warmups_before_each_measurement": 1,
        "benchmark_seed": COMMON.DEFAULT_BENCHMARK_SEED, "order_seed": COMMON.DEFAULT_ORDER_SEED,
        "bootstrap_resamples": 20, "phase_gate": {},
    }
    COMMON.write_json(evidence / "environment.json", environment)
    (evidence / "build_manifest.json").write_bytes(build_path.read_bytes())
    (evidence / "ctest.txt").write_bytes((build_path.parent / "ctest.txt").read_bytes())
    rewrite_evidence_manifest(evidence)
    return evidence


def rewrite_evidence_manifest(evidence: Path) -> None:
    previous = json.loads((evidence / "manifest.json").read_text()) if (evidence / "manifest.json").exists() else {}
    files = {
        name: {"sha256": COMMON.sha256_file(evidence / name), "bytes": (evidence / name).stat().st_size}
        for name in VERIFIER.EXPECTED_FILES
    }
    environment = json.loads((evidence / "environment.json").read_text())
    COMMON.write_json(evidence / "manifest.json", {
        "protocol_version": previous.get("protocol_version", COMMON.PROTOCOL_VERSION),
        "mode": previous.get("mode", environment["mode"]),
        "source_commit": previous.get("source_commit", SOURCE_SHA),
        "build_manifest_sha256": files["build_manifest.json"]["sha256"],
        "environment_sha256": previous.get("environment_sha256", ENVIRONMENT_HASH), "files": files,
    })


def mutate_raw(evidence: Path, mutation) -> None:
    rows = read_csv(evidence / "raw_timings.csv")
    mutation(rows)
    write_csv(evidence / "raw_timings.csv", RUNNER.RAW_FIELDS, rows)
    rewrite_evidence_manifest(evidence)


def make_policy(calibration: Path, output: Path) -> tuple[Path, Path, dict[str, object]]:
    report = VERIFIER.verify_evidence(calibration)
    content = POLICY.build_policy_content(calibration, report)
    digest = COMMON.canonical_sha256(content)
    output.mkdir()
    policy_path = output / "policy.json"
    COMMON.write_json(policy_path, {"policy_content": content, "generated_utc": "metadata-only", "policy_sha256": digest})
    freeze_path = output / "policy_freeze.json"
    COMMON.write_json(freeze_path, {
        "protocol_version": COMMON.PROTOCOL_VERSION, "source_commit": report["source_commit"],
        "build_manifest_sha256": report["build_manifest_sha256"],
        "calibration_manifest_sha256": report["manifest_sha256"],
        "environment_id": report["environment_id"], "environment_sha256": report["environment_sha256"],
        "openmp_threads": report["openmp_threads"], "policy_sha256": digest,
        "policy_file_sha256": COMMON.sha256_file(policy_path), "frozen_utc": "metadata-only",
    })
    return policy_path, freeze_path, report


class Stage5V2Tests(unittest.TestCase):
    def assert_rejected(self, function, *args, **kwargs) -> None:
        with self.assertRaises(COMMON.Stage5Error):
            function(*args, **kwargs)

    def test_phase_sets_are_disjoint_and_frozen(self) -> None:
        self.assertFalse(set(COMMON.CALIBRATION_BODIES) & set(COMMON.HELD_OUT_BODIES))
        self.assertEqual(COMMON.PILOT_BODIES, (64, 512, 4096))
        self.assertEqual(COMMON.CALIBRATION_BODIES, (64, 128, 192, 256, 512, 1024, 2048, 4096))
        self.assertEqual(COMMON.HELD_OUT_BODIES, (96, 160, 384, 768, 1536, 3072))

    def test_schedule_is_deterministic_and_phase_local(self) -> None:
        first = COMMON.ordered_runs("calibration", THREADS, 11, COMMON.DEFAULT_ORDER_SEED)
        self.assertEqual(first, COMMON.ordered_runs("calibration", THREADS, 11, COMMON.DEFAULT_ORDER_SEED))
        self.assertEqual({item["bodies"] for item in first}, set(COMMON.CALIBRATION_BODIES))
        self.assertFalse({item["bodies"] for item in first} & set(COMMON.HELD_OUT_BODIES))

    def test_topology_derivation_and_forgery(self) -> None:
        self.assertEqual(COMMON.derive_openmp_threads(10, 16), (1, 2, 4, 8, 10, 16))
        self.assert_rejected(COMMON.validate_openmp_threads, (1, 2, 4, 16), 10, 16)
        self.assert_rejected(COMMON.derive_openmp_threads, 17, 16)

    def test_existing_build_and_evidence_directories_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assert_rejected(COMMON.require_absent_directory, Path(temporary), "build directory")

    def test_valid_synthetic_phases_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("pilot", "calibration", "held_out"):
                self.assertEqual(VERIFIER.verify_evidence(make_evidence(root, mode))["mode"], mode)

    def test_wrong_protocol_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = make_evidence(Path(temporary), "pilot")
            manifest = json.loads((evidence / "manifest.json").read_text())
            manifest["protocol_version"] = "stage5-v1.0"
            COMMON.write_json(evidence / "manifest.json", manifest)
            self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_partial_run_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            (evidence / "raw_timings.partial.csv").write_text("partial\n", encoding="utf-8")
            self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_calibration_heldout_n_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = make_evidence(Path(temporary), "calibration")
            mutate_raw(evidence, lambda rows: [row.update(bodies="96") for row in rows if row["order_index"] == "1"])
            self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_missing_and_duplicate_calibration_rows_are_rejected(self) -> None:
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as temporary:
                evidence = make_evidence(Path(temporary), "calibration")
                def change(rows):
                    rows.append(dict(rows[0])) if duplicate else rows.pop()
                mutate_raw(evidence, change)
                self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_missing_and_duplicate_heldout_rows_are_rejected(self) -> None:
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as temporary:
                evidence = make_evidence(Path(temporary), "held_out")
                def change(rows):
                    rows.append(dict(rows[0])) if duplicate else rows.pop()
                mutate_raw(evidence, change)
                self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_nonfinite_timing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = make_evidence(Path(temporary), "pilot")
            mutate_raw(evidence, lambda rows: rows[0].update(elapsed_ms="nan"))
            self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_failed_correctness_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = make_evidence(Path(temporary), "pilot")
            mutate_raw(evidence, lambda rows: rows[0].update(correctness_status="fail"))
            self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_checksum_instability_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = make_evidence(Path(temporary), "pilot")
            def change(rows):
                target = next(row for row in rows if row["block_id"] == "2" and row["backend"] == "openmp")
                for row in rows:
                    if row["run_id"] == target["run_id"]:
                        row["checksum"] = "tampered"
            mutate_raw(evidence, change)
            self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_cuda_timing_relationship_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = make_evidence(Path(temporary), "pilot")
            def change(rows):
                run_id = next(row["run_id"] for row in rows if row["backend"] == "cuda")
                for row in rows:
                    if row["run_id"] == run_id and row["timing_metric"] == "transfer_inclusive":
                        row["elapsed_ms"] = "0.0001"
            mutate_raw(evidence, change)
            self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_tampered_ctest_log_and_missing_pass_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = make_evidence(Path(temporary), "pilot")
            (evidence / "ctest.txt").write_text("forged pass\n", encoding="utf-8")
            rewrite_evidence_manifest(evidence)
            self.assert_rejected(VERIFIER.verify_evidence, evidence)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = make_evidence(Path(temporary), "pilot")
            build = json.loads((evidence / "build_manifest.json").read_text())
            build["ctest"]["passed"] = False
            COMMON.write_json(evidence / "build_manifest.json", build)
            rewrite_evidence_manifest(evidence)
            self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_build_manifest_rejects_stale_source_debug_arch_hash_and_topology(self) -> None:
        cases = (
            ("stale source", lambda m: m["cmake_cache"].update(CMAKE_HOME_DIRECTORY=str(ROOT.parent))),
            ("debug", lambda m: m["cmake_cache"].update(CMAKE_BUILD_TYPE="Debug")),
            ("wrong architecture", lambda m: m["cmake_cache"].update(CMAKE_CUDA_ARCHITECTURES="86")),
            ("wrong executable hash", lambda m: m["binaries"]["nbody_benchmark"].update(sha256="0" * 64)),
            ("wrong topology", lambda m: m["openmp_topology"].update(threads=[1, 2, 8])),
            ("wrong environment", lambda m: m.update(environment_id="other-machine")),
        )
        for label, mutation in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                manifest_path, manifest = make_build_bundle(Path(temporary))
                candidate = copy.deepcopy(manifest)
                mutation(candidate)
                self.assert_rejected(RUNNER.validate_build_manifest_document, candidate,
                                     manifest_path.parent, SOURCE_SHA, ENVIRONMENT_ID, ROOT,
                                     execute_binaries=False)

    def test_build_manifest_accepts_complete_bound_fixture_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = make_build_bundle(Path(temporary))
            RUNNER.validate_build_manifest_document(manifest, path.parent, SOURCE_SHA,
                                                    ENVIRONMENT_ID, ROOT, execute_binaries=False)

    def test_ctest_pass_detection_accepts_combined_output_streams(self) -> None:
        completed = subprocess.CompletedProcess(
            ["ctest"], 0, "", "100% tests passed out of 5\n"
        )
        self.assertTrue(PREPARE.ctest_succeeded(completed))
        self.assertFalse(PREPARE.ctest_succeeded(subprocess.CompletedProcess(["ctest"], 1, completed.stderr, "")))
        self.assertFalse(PREPARE.ctest_succeeded(subprocess.CompletedProcess(["ctest"], 0, "no summary", "")))

    def test_wrong_and_dirty_git_source_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "build_manifest.json"
            path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(RUNNER, "git_value", return_value="b" * 40):
                self.assert_rejected(RUNNER.validate_build_manifest, path, SOURCE_SHA, ENVIRONMENT_ID)
            with mock.patch.object(RUNNER, "git_value", side_effect=[SOURCE_SHA, " M source.cpp"]):
                self.assert_rejected(RUNNER.validate_build_manifest, path, SOURCE_SHA, ENVIRONMENT_ID)

    def test_pilot_approval_binds_actual_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = make_evidence(root, "pilot")
            report = VERIFIER.verify_evidence(pilot)
            build = json.loads((pilot / "build_manifest.json").read_text())
            approval = {
                "approved": True, "protocol_version": COMMON.PROTOCOL_VERSION,
                "source_commit": SOURCE_SHA, "build_manifest_sha256": report["build_manifest_sha256"],
                "environment_id": ENVIRONMENT_ID, "environment_sha256": ENVIRONMENT_HASH,
                "pilot_manifest_sha256": report["manifest_sha256"], "openmp_threads": list(THREADS),
                "reviewer": "test reviewer", "reviewed_utc": "2026-09-28T00:00:00Z",
                "notes": "synthetic test only",
            }
            approval_path = root / "approval.json"
            COMMON.write_json(approval_path, approval)
            RUNNER.validate_pilot_gate(pilot, approval_path, SOURCE_SHA,
                                       report["build_manifest_sha256"], build)
            approval["openmp_threads"] = None
            COMMON.write_json(approval_path, approval)
            self.assert_rejected(RUNNER.validate_pilot_gate, pilot, approval_path, SOURCE_SHA,
                                 report["build_manifest_sha256"], build)
            approval["openmp_threads"] = list(THREADS)
            approval["pilot_manifest_sha256"] = "0" * 64
            COMMON.write_json(approval_path, approval)
            self.assert_rejected(RUNNER.validate_pilot_gate, pilot, approval_path, SOURCE_SHA,
                                 report["build_manifest_sha256"], build)

    def test_tampered_pilot_evidence_is_rejected_before_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = make_evidence(Path(temporary), "pilot")
            mutate_raw(evidence, lambda rows: rows[0].update(checksum="tampered"))
            self.assert_rejected(VERIFIER.verify_evidence, evidence)

    def test_policy_interpolation_and_canonical_tie_break_are_deterministic(self) -> None:
        self.assertAlmostEqual(POLICY.interpolate_loglog({64: 1.0, 256: 4.0}, 128), 2.0)
        self.assertEqual(COMMON.canonical_configuration_order(THREADS),
                         ("sequential:1", "openmp:1", "openmp:2", "cuda:0"))
        with tempfile.TemporaryDirectory() as temporary:
            calibration = make_evidence(Path(temporary), "calibration")
            report = VERIFIER.verify_evidence(calibration)
            first = POLICY.build_policy_content(calibration, report)
            second = POLICY.build_policy_content(calibration, report)
            self.assertEqual(first, second)
            self.assertEqual(COMMON.canonical_sha256(first), COMMON.canonical_sha256(second))

    def test_policy_generation_rejects_heldout_or_mixed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assert_rejected(POLICY.calibration_medians, make_evidence(root, "held_out"))
            calibration = make_evidence(root, "calibration")
            rows = read_csv(calibration / "summary.csv")
            next(row for row in rows if row["timing_metric"] == COMMON.PRIMARY_METRIC)["dataset_role"] = "held_out"
            write_csv(calibration / "summary.csv", RUNNER.SUMMARY_FIELDS, rows)
            self.assert_rejected(POLICY.calibration_medians, calibration)

    def test_changed_calibration_after_freeze_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = make_evidence(root, "calibration", scale=1.0)
            policy_path, freeze_path, _ = make_policy(original, root / "policy")
            changed = make_evidence(root, "calibration", scale=1.1)
            report = VERIFIER.verify_evidence(changed)
            build = json.loads((changed / "build_manifest.json").read_text())
            self.assert_rejected(POLICY.verify_policy_chain, policy_path, freeze_path,
                                 report, build, report["build_manifest_sha256"])

    def test_changed_policy_after_freeze_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = make_evidence(root, "calibration")
            policy_path, freeze_path, report = make_policy(calibration, root / "policy")
            policy = json.loads(policy_path.read_text())
            policy["policy_content"]["held_out_queries"]["96"]["selected_configuration"] = "cuda:0"
            policy["policy_sha256"] = COMMON.canonical_sha256(policy["policy_content"])
            COMMON.write_json(policy_path, policy)
            build = json.loads((calibration / "build_manifest.json").read_text())
            self.assert_rejected(POLICY.verify_policy_chain, policy_path, freeze_path,
                                 report, build, report["build_manifest_sha256"])

    def test_heldout_before_policy_freeze_is_rejected(self) -> None:
        self.assert_rejected(RUNNER.policy_gate,
                             argparse.Namespace(policy=None, policy_freeze=None), {}, {}, "0" * 64)

    def test_heldout_source_build_environment_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = make_evidence(root, "calibration")
            policy_path, freeze_path, _ = make_policy(calibration, root / "policy")
            report = VERIFIER.verify_evidence(make_evidence(root, "held_out"))
            for field, value in (("source_commit", "b" * 40),
                                 ("build_manifest_sha256", "0" * 64),
                                 ("environment_sha256", "F" * 64)):
                with self.subTest(field=field):
                    candidate = dict(report)
                    candidate[field] = value
                    self.assert_rejected(EVALUATE.validate_frozen_policy,
                                         policy_path, freeze_path, candidate)

    def test_evaluation_reports_oracle_and_regret_without_refit(self) -> None:
        order = ["sequential:1", "openmp:1", "openmp:2", "cuda:0"]
        content = {"canonical_configuration_order": order,
                   "held_out_queries": {str(n): {"selected_configuration": "openmp:2"}
                                        for n in COMMON.HELD_OUT_BODIES}}
        medians = {n: {"sequential:1": 4.0, "openmp:1": 3.0,
                       "openmp:2": 2.0, "cuda:0": 1.0}
                   for n in COMMON.HELD_OUT_BODIES}
        results = EVALUATE.evaluate(content, medians)
        self.assertTrue(all(row["multiplicative_regret"] == 2.0 for row in results))
        self.assertTrue(all(not row["selected_empirical_fastest"] for row in results))

    def test_deployment_packet_is_deterministic_and_exact_sha_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = []
            for index in (1, 2):
                output = root / f"packet-{index}"
                with mock.patch.object(sys, "argv", ["packet", "--source-sha", SOURCE_SHA,
                                                     "--output", str(output)]):
                    self.assertEqual(PACKET.main(), 0)
                outputs.append(output)
            first = json.loads((outputs[0] / "packet_manifest.json").read_text())
            second = json.loads((outputs[1] / "packet_manifest.json").read_text())
            self.assertEqual(first["files"], second["files"])
            runbook = (outputs[0] / "RUNBOOK.md").read_text(encoding="utf-8")
            self.assertIn(SOURCE_SHA, runbook)
            self.assertNotIn("STAGE5_FROZEN_SOURCE_COMMIT", runbook)
            self.assertIn("FIRST NVIDIA REHEARSAL STOP", runbook)

    def test_tracked_approval_template_is_unmistakably_uninitialized(self) -> None:
        template = json.loads(
            (ROOT / "docs" / "stage5_pilot_approval.template.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(template["openmp_threads"])
        self.assertFalse(template["approved"])
        self.assertIn("automatically derived topology", template["notes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
