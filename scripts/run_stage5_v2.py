#!/usr/bin/env python3
"""Run separated, provenance-bound Stage 5 v2.0 acquisition phases."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage5_common import (
    ANGULAR_BOUND,
    DEFAULT_BENCHMARK_SEED,
    DEFAULT_ORDER_SEED,
    ENERGY_BOUND,
    MODE_BODIES,
    MODE_REPETITIONS,
    MODE_ROLE,
    MODE_STEPS,
    PROTOCOL_VERSION,
    Stage5Error,
    bootstrap_interval,
    canonical_sha256,
    finite_float,
    git_value,
    group_seed,
    measurement_environment,
    ordered_runs,
    quantile,
    read_json,
    require,
    require_absent_directory,
    run_command,
    sha256_file,
    validate_openmp_threads,
    write_json,
)
from verify_stage5_v2 import verify_evidence


RAW_FIELDS = (
    "environment_id",
    "environment_sha256",
    "source_commit",
    "build_manifest_sha256",
    "protocol_version",
    "provenance_mode",
    "experiment_id",
    "mode",
    "dataset_role",
    "run_id",
    "order_seed",
    "block_id",
    "order_index",
    "repetition",
    "backend",
    "requested_threads",
    "effective_threads",
    "bodies",
    "steps",
    "dt_s",
    "benchmark_seed",
    "warmups",
    "timing_metric",
    "elapsed_ms",
    "correctness_status",
    "checksum",
    "relative_energy_drift",
    "relative_angular_momentum_drift",
    "energy_drift_bound",
    "angular_momentum_drift_bound",
    "build_configuration",
    "openmp_specification_date",
    "setup_allocation_ms",
    "host_staging_ms",
    "host_to_device_ms",
    "device_to_host_ms",
    "device_index",
    "gpu_name",
    "compute_capability",
    "total_global_memory_bytes",
    "multiprocessor_count",
    "cuda_driver_version",
    "cuda_runtime_version",
)
SUMMARY_FIELDS = (
    "mode",
    "dataset_role",
    "backend",
    "requested_threads",
    "bodies",
    "timing_metric",
    "samples",
    "median_ms",
    "mean_ms",
    "stdev_ms",
    "coefficient_of_variation",
    "min_ms",
    "max_ms",
    "iqr_ms",
    "median_ci95_low_ms",
    "median_ci95_high_ms",
)


def integer_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result) or tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError("expected sorted unique positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--calibration", action="store_true")
    mode.add_argument("--held-out", dest="held_out", action="store_true")
    mode.add_argument("--dry-run", choices=tuple(MODE_BODIES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--environment-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--pilot-approval", type=Path)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--policy-freeze", type=Path)
    parser.add_argument("--dry-run-threads", type=integer_list)
    return parser.parse_args()


def mode_name(args: argparse.Namespace) -> str:
    if args.pilot:
        return "pilot"
    if args.calibration:
        return "calibration"
    if args.held_out:
        return "held_out"
    if args.preflight:
        return "preflight"
    return str(args.dry_run)


def parse_csv_row(completed: Any, command: list[str]) -> dict[str, str]:
    if completed.returncode != 0:
        raise Stage5Error(f"benchmark failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}")
    if completed.stderr.strip():
        raise Stage5Error(f"benchmark wrote unexpected stderr: {' '.join(command)}\n{completed.stderr}")
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    require(len(rows) == 1, f"expected one benchmark row, got {len(rows)}")
    return rows[0]


def validate_build_manifest_document(
    manifest: dict[str, Any],
    manifest_dir: Path,
    source_commit: str,
    environment_id: str,
    source_dir: Path,
    *,
    execute_binaries: bool = True,
) -> None:
    source_dir = source_dir.resolve()
    require(manifest.get("protocol_version") == PROTOCOL_VERSION, "build protocol mismatch")
    require(manifest.get("source_commit") == source_commit, "build source mismatch")
    require(manifest.get("environment_id") == environment_id, "build environment mismatch")
    require(canonical_sha256(manifest.get("environment")) == manifest.get("environment_sha256"), "build environment hash mismatch")
    require(manifest.get("ctest", {}).get("passed") is True, "build manifest lacks passing CTest")
    require(manifest.get("ctest", {}).get("return_code") == 0, "build CTest return code is not zero")
    require(not manifest.get("fixture_mode"), "fixture build cannot run controlled acquisition")
    require(Path(str(manifest.get("source_directory", ""))).resolve() == source_dir, "build source directory mismatch")
    cache = manifest.get("cmake_cache", {})
    require(Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() == source_dir, "stale/wrong CMAKE_HOME_DIRECTORY")
    require(cache.get("CMAKE_GENERATOR") == "Ninja", "controlled build generator is not Ninja")
    require(cache.get("CMAKE_BUILD_TYPE") == "Release", "build is not Release")
    require(cache.get("NBODY_ENABLE_OPENMP") == "ON", "build lacks OpenMP")
    require(cache.get("NBODY_ENABLE_CUDA") == "ON", "controlled build lacks CUDA")
    require(cache.get("NBODY_ENABLE_MPI") == "OFF", "controlled build unexpectedly enables MPI")
    require(cache.get("NBODY_BUILD_VISUALIZATION") == "OFF", "controlled build unexpectedly enables visualization")
    require(cache.get("NBODY_STAGE5_SOURCE_SHA") == source_commit, "cache source SHA mismatch")
    require(cache.get("NBODY_STAGE5_PROTOCOL_VERSION") == PROTOCOL_VERSION, "cache protocol mismatch")
    require(cache.get("CMAKE_CUDA_ARCHITECTURES") == "89", "controlled build CUDA architecture is not 89")
    for name, expected in manifest.get("files", {}).items():
        artifact = manifest_dir / name
        require(artifact.is_file(), f"build artifact missing: {name}")
        require(sha256_file(artifact) == expected.get("sha256"), f"build artifact hash mismatch: {name}")
        require(artifact.stat().st_size == int(expected.get("bytes", -1)), f"build artifact size mismatch: {name}")
    require(set(manifest.get("files", {})) == {"configure.txt", "build.txt", "ctest.txt"}, "build artifact set mismatch")
    require(manifest["ctest"]["log_sha256"] == manifest["files"]["ctest.txt"]["sha256"], "CTest log binding mismatch")
    required_binaries = {
        "nbody_benchmark",
        "nbody_tests",
        "nbody_solar_system_tests",
        "nbody_openmp_tests",
        "nbody_cuda_benchmark",
        "nbody_cuda_tests",
    }
    require(set(manifest.get("binaries", {})) == required_binaries, "controlled binary set mismatch")
    topology = manifest.get("openmp_topology", {})
    validate_openmp_threads(
        tuple(int(value) for value in topology.get("threads", [])),
        int(topology.get("physical_cores", 0)),
        int(topology.get("logical_processors", 0)),
    )
    compiler = manifest.get("compiler_identity", {})
    require(compiler.get("source_commit") == source_commit, "compiler identity source mismatch")
    require(compiler.get("protocol_version") == PROTOCOL_VERSION, "compiler identity protocol mismatch")
    require(compiler.get("generator") == "Ninja", "embedded generator mismatch")
    require(Path(str(compiler.get("cxx_compiler", ""))).is_absolute(), "C++ compiler path is not absolute")
    require(str(compiler.get("compiler_id", "")), "C++ compiler ID missing")
    require(str(compiler.get("compiler_version", "")), "C++ compiler version missing")
    cuda = manifest.get("cuda_compiler_identity", {})
    require(cuda.get("cuda_compiler_id") == "NVIDIA", "CUDA compiler ID mismatch")
    require(Path(str(cuda.get("cuda_compiler", ""))).is_absolute(), "CUDA compiler path is not absolute")
    require(str(cuda.get("cuda_compiler_version", "")), "CUDA compiler version missing")
    require(Path(str(cuda.get("cuda_host_compiler", ""))).is_absolute(), "CUDA host compiler path is not absolute")
    require(cuda.get("cuda_architectures") == "89", "embedded CUDA architecture is not 89")
    expected_scripts = {
        "stage5_common.py",
        "run_stage5_v2.py",
        "verify_stage5_v2.py",
        "build_stage5_policy.py",
        "evaluate_stage5_policy.py",
    }
    require(set(manifest.get("tool_scripts", {})) == expected_scripts, "Stage 5 tool script set mismatch")
    for name, expected_hash in manifest["tool_scripts"].items():
        require(sha256_file(source_dir / "scripts" / name) == expected_hash, f"Stage 5 tool script hash mismatch: {name}")
    for name, record in manifest.get("binaries", {}).items():
        binary = Path(record["path"]).resolve()
        require(binary.is_file(), f"build binary missing: {name}")
        require(sha256_file(binary) == record["sha256"], f"build binary hash mismatch: {name}")
        info = parse_csv_or_json_build_info(binary) if execute_binaries else record.get("build_info", {})
        require(info.get("source_commit") == source_commit, f"binary source mismatch: {name}")
        require(info.get("protocol_version") == PROTOCOL_VERSION, f"binary protocol mismatch: {name}")
        require(str(info.get("build_configuration", "")).casefold() == "release", f"binary is not Release: {name}")


def validate_build_manifest(path: Path, source_commit: str, environment_id: str) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    require(resolved.is_file(), f"build manifest missing: {resolved}")
    manifest = read_json(resolved)
    source_dir = Path.cwd().resolve()
    require(git_value(source_dir, "rev-parse", "HEAD") == source_commit, "current source HEAD mismatch")
    require(git_value(source_dir, "status", "--porcelain=v1") == "", "current source checkout must be clean")
    validate_build_manifest_document(
        manifest, resolved.parent, source_commit, environment_id, source_dir
    )
    return manifest, sha256_file(resolved)


def parse_csv_or_json_build_info(binary: Path) -> dict[str, Any]:
    completed = run_command([str(binary), "--build-info"])
    require(completed.returncode == 0, f"--build-info failed: {binary}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise Stage5Error(f"invalid --build-info JSON: {binary}") from error
    require(isinstance(value, dict), f"invalid --build-info object: {binary}")
    return value


def validate_pilot_gate(
    pilot_dir: Path,
    approval_path: Path,
    source_commit: str,
    build_hash: str,
    build: dict[str, Any],
) -> dict[str, Any]:
    report = verify_evidence(pilot_dir)
    require(report["mode"] == "pilot", "approval input is not pilot evidence")
    approval = read_json(approval_path.resolve())
    required = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": source_commit,
        "build_manifest_sha256": build_hash,
        "environment_id": build["environment_id"],
        "environment_sha256": build["environment_sha256"],
        "pilot_manifest_sha256": report["manifest_sha256"],
        "openmp_threads": build["openmp_topology"]["threads"],
    }
    require(approval.get("approved") is True, "pilot approval must set approved=true")
    for key, expected in required.items():
        require(approval.get(key) == expected, f"pilot approval {key} mismatch")
    for field in ("reviewer", "reviewed_utc", "notes"):
        require(str(approval.get(field, "")).strip() != "", f"pilot approval missing {field}")
    return {
        "pilot_verification": report,
        "pilot_approval_sha256": sha256_file(approval_path.resolve()),
        "pilot_approval": approval,
    }


def benchmark_command(build: dict[str, Any], config: dict[str, Any], steps: int, seed: int, device: int) -> list[str]:
    binaries = build["binaries"]
    if config["backend"] == "cuda":
        return [
            binaries["nbody_cuda_benchmark"]["path"],
            "--bodies",
            str(config["bodies"]),
            "--steps",
            str(steps),
            "--dt",
            "10",
            "--warmups",
            "1",
            "--repetitions",
            "1",
            "--seed",
            str(seed),
            "--device",
            str(device),
        ]
    return [
        binaries["nbody_benchmark"]["path"],
        "--backend",
        str(config["backend"]),
        "--bodies",
        str(config["bodies"]),
        "--steps",
        str(steps),
        "--dt",
        "10",
        "--threads",
        str(config["threads"] or 1),
        "--warmups",
        "1",
        "--repetitions",
        "1",
        "--seed",
        str(seed),
    ]


def normalized_rows(
    benchmark: dict[str, str],
    config: dict[str, Any],
    mode: str,
    steps: int,
    block_id: int,
    order_index: int,
    args: argparse.Namespace,
    build: dict[str, Any],
    build_hash: str,
) -> list[dict[str, str]]:
    backend = str(config["backend"])
    energy = finite_float(benchmark["relative_energy_drift"], "relative_energy_drift")
    angular = finite_float(benchmark["relative_angular_momentum_drift"], "relative_angular_momentum_drift")
    require(energy <= ENERGY_BOUND and angular <= ANGULAR_BOUND, f"scientific gate failed for {backend} N={config['bodies']}")
    if backend == "cuda":
        require(benchmark.get("correctness_status") == "pass", "CUDA correctness failed")
        requested_threads = effective_threads = "0"
        metrics = (
            ("device_compute", finite_float(benchmark["device_compute_ms"], "device_compute_ms")),
            ("transfer_inclusive", finite_float(benchmark["transfer_inclusive_ms"], "transfer_inclusive_ms")),
            ("selector_steady_state", finite_float(benchmark["backend_wall_ms"], "backend_wall_ms")),
        )
        require(metrics[1][1] >= metrics[0][1], "CUDA transfer-inclusive timing is below device compute")
    else:
        requested_threads = str(config["threads"] if backend == "openmp" else 1)
        effective_threads = benchmark["effective_threads"]
        require(int(effective_threads) == int(requested_threads), "requested/effective OpenMP team mismatch")
        metrics = (
            ("integration_compute", finite_float(benchmark["median_s"], "median_s") * 1000.0),
            ("selector_steady_state", finite_float(benchmark["selector_steady_state_s"], "selector_steady_state_s") * 1000.0),
        )
    run_id = f"b{block_id:02d}-o{order_index:04d}"
    common = {
        "environment_id": build["environment_id"],
        "environment_sha256": build["environment_sha256"],
        "source_commit": str(args.source_commit),
        "build_manifest_sha256": build_hash,
        "protocol_version": PROTOCOL_VERSION,
        "provenance_mode": "source-build-test-bound",
        "experiment_id": str(args.experiment_id),
        "mode": mode,
        "dataset_role": MODE_ROLE[mode],
        "run_id": run_id,
        "order_seed": str(args.order_seed),
        "block_id": str(block_id),
        "order_index": str(order_index),
        "repetition": str(block_id),
        "backend": backend,
        "requested_threads": requested_threads,
        "effective_threads": effective_threads,
        "bodies": str(config["bodies"]),
        "steps": str(steps),
        "dt_s": "10",
        "benchmark_seed": str(args.seed),
        "warmups": "1",
        "correctness_status": "pass",
        "checksum": benchmark["checksum"],
        "relative_energy_drift": f"{energy:.17g}",
        "relative_angular_momentum_drift": f"{angular:.17g}",
        "energy_drift_bound": f"{ENERGY_BOUND:.17g}",
        "angular_momentum_drift_bound": f"{ANGULAR_BOUND:.17g}",
        "build_configuration": benchmark["build_configuration"],
        "openmp_specification_date": benchmark.get("openmp_specification_date", ""),
        "setup_allocation_ms": benchmark.get("allocation_ms", ""),
        "host_staging_ms": benchmark.get("host_staging_ms", ""),
        "host_to_device_ms": benchmark.get("host_to_device_ms", ""),
        "device_to_host_ms": benchmark.get("device_to_host_ms", ""),
        "device_index": benchmark.get("device_index", ""),
        "gpu_name": benchmark.get("gpu_name", ""),
        "compute_capability": benchmark.get("compute_capability", ""),
        "total_global_memory_bytes": benchmark.get("total_global_memory_bytes", ""),
        "multiprocessor_count": benchmark.get("multiprocessor_count", ""),
        "cuda_driver_version": benchmark.get("cuda_driver_version", ""),
        "cuda_runtime_version": benchmark.get("cuda_runtime_version", ""),
    }
    result = []
    for metric, elapsed_ms in metrics:
        require(math.isfinite(elapsed_ms) and elapsed_ms > 0.0, f"invalid {metric} timing")
        result.append({**common, "timing_metric": metric, "elapsed_ms": f"{elapsed_ms:.17g}"})
    return result


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_summaries(rows: list[dict[str, str]], mode: str, args: argparse.Namespace) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["dataset_role"], row["backend"], int(row["requested_threads"]), int(row["bodies"]), row["timing_metric"])
        grouped[key].append(float(row["elapsed_ms"]))
    result: list[dict[str, str]] = []
    for key in sorted(grouped):
        role, backend, threads, bodies, metric = key
        values = grouped[key]
        mean = statistics.mean(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        low, high = bootstrap_interval(values, args.bootstrap_resamples, group_seed(args.order_seed, key))
        result.append({
            "mode": mode,
            "dataset_role": role,
            "backend": backend,
            "requested_threads": str(threads),
            "bodies": str(bodies),
            "timing_metric": metric,
            "samples": str(len(values)),
            "median_ms": f"{statistics.median(values):.17g}",
            "mean_ms": f"{mean:.17g}",
            "stdev_ms": f"{deviation:.17g}",
            "coefficient_of_variation": f"{deviation / mean:.17g}",
            "min_ms": f"{min(values):.17g}",
            "max_ms": f"{max(values):.17g}",
            "iqr_ms": f"{quantile(values, 0.75) - quantile(values, 0.25):.17g}",
            "median_ci95_low_ms": f"{low:.17g}",
            "median_ci95_high_ms": f"{high:.17g}",
        })
    return result


def validate_collected_rows(rows: list[dict[str, str]], schedule: list[dict[str, Any]]) -> None:
    expected = sum(3 if item["backend"] == "cuda" else 2 for item in schedule)
    require(len(rows) == expected, f"expected {expected} tidy rows, got {len(rows)}")
    checksums: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    cpu_checksums: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        key = (row["backend"], int(row["requested_threads"]), int(row["bodies"]))
        checksums[key].add(row["checksum"])
        if row["backend"] != "cuda":
            cpu_checksums[int(row["bodies"])].add(row["checksum"])
    require(all(len(values) == 1 for values in checksums.values()), "repeat checksum instability")
    require(all(len(values) == 1 for values in cpu_checksums.values()), "Sequential/OpenMP checksum mismatch")


def policy_gate(args: argparse.Namespace, calibration_report: dict[str, Any], build: dict[str, Any], build_hash: str) -> dict[str, Any]:
    require(args.policy and args.policy_freeze, "held-out requires --policy and --policy-freeze")
    from build_stage5_policy import verify_policy_chain

    return verify_policy_chain(
        args.policy.resolve(),
        args.policy_freeze.resolve(),
        calibration_report,
        build,
        build_hash,
    )


def main() -> int:
    args = parse_args()
    mode = mode_name(args)
    if args.dry_run:
        require(args.dry_run_threads is not None, "dry-run requires --dry-run-threads")
        schedule = ordered_runs(mode, args.dry_run_threads, MODE_REPETITIONS[mode], args.order_seed)
        print(json.dumps({"protocol_version": PROTOCOL_VERSION, "mode": mode, "schedule": schedule}, indent=2))
        return 0

    require(args.output is not None, "--output is required")
    require(args.build_manifest is not None, "--build-manifest is required")
    require(args.source_commit and len(args.source_commit) == 40, "full --source-commit is required")
    require(args.environment_id, "--environment-id is required")
    require(args.bootstrap_resamples > 0, "bootstrap resamples must be positive")
    require(args.device >= 0, "device must be non-negative")
    output = require_absent_directory(args.output, "evidence output directory")
    build, build_hash = validate_build_manifest(args.build_manifest, args.source_commit, args.environment_id)
    threads = tuple(int(value) for value in build["openmp_topology"]["threads"])

    gate: dict[str, Any] = {}
    if mode == "calibration":
        require(args.pilot_dir and args.pilot_approval, "calibration requires actual --pilot-dir and --pilot-approval")
        gate = validate_pilot_gate(args.pilot_dir, args.pilot_approval, args.source_commit, build_hash, build)
    elif mode == "held_out":
        require(args.calibration_dir, "held-out requires --calibration-dir")
        calibration_report = verify_evidence(args.calibration_dir)
        require(calibration_report["mode"] == "calibration", "held-out gate input is not calibration evidence")
        require(calibration_report["source_commit"] == args.source_commit, "held-out source differs from calibration")
        require(calibration_report["build_manifest_sha256"] == build_hash, "held-out build differs from calibration")
        require(calibration_report["environment_sha256"] == build["environment_sha256"], "held-out environment differs from calibration")
        require(calibration_report["openmp_threads"] == list(threads), "held-out topology differs from calibration")
        gate = policy_gate(args, calibration_report, build, build_hash)

    output.mkdir(parents=True)
    if mode == "preflight":
        shutil.copyfile(args.build_manifest.resolve(), output / "build_manifest.json")
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "source_commit": args.source_commit,
            "build_manifest_sha256": build_hash,
            "environment_id": build["environment_id"],
            "environment_sha256": build["environment_sha256"],
            "openmp_topology": build["openmp_topology"],
            "binaries_rehashed": sorted(build["binaries"]),
            "status": "pass",
        }
        write_json(output / "preflight.json", payload)
        print(f"Stage 5 v2 preflight passed: {output / 'preflight.json'}")
        return 0

    require(args.experiment_id, f"--experiment-id is required for {mode}")
    steps = MODE_STEPS[mode]
    repetitions = MODE_REPETITIONS[mode]
    schedule = ordered_runs(mode, threads, repetitions, args.order_seed)
    rows: list[dict[str, str]] = []
    partial = output / "raw_timings.partial.csv"
    failure = output / "failure.json"
    started = datetime.now(timezone.utc)
    try:
        with partial.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for order_index, config in enumerate(schedule, start=1):
                command = benchmark_command(build, config, steps, args.seed, args.device)
                completed = run_command(command, measurement_environment())
                try:
                    benchmark = parse_csv_row(completed, command)
                    normalized = normalized_rows(
                        benchmark,
                        config,
                        mode,
                        steps,
                        int(config["block_id"]),
                        order_index,
                        args,
                        build,
                        build_hash,
                    )
                except Exception as error:
                    write_json(failure, {
                        "protocol_version": PROTOCOL_VERSION,
                        "mode": mode,
                        "order_index": order_index,
                        "configuration": config,
                        "command": command,
                        "return_code": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "error": str(error),
                    })
                    raise
                writer.writerows(normalized)
                stream.flush()
                rows.extend(normalized)
                selector = next(row for row in normalized if row["timing_metric"] == "selector_steady_state")
                print(
                    f"[{order_index:04d}/{len(schedule)}] {mode} block={config['block_id']} "
                    f"N={config['bodies']} {config['backend']} t={config['threads']} "
                    f"selector={selector['elapsed_ms']}ms",
                    flush=True,
                )
        validate_collected_rows(rows, schedule)
        partial.replace(output / "raw_timings.csv")
    except Exception:
        print(f"partial evidence retained at {partial}; failure detail retained at {failure}", file=sys.stderr)
        raise

    write_csv(output / "summary.csv", SUMMARY_FIELDS, make_summaries(rows, mode, args))
    shutil.copyfile(args.build_manifest.resolve(), output / "build_manifest.json")
    shutil.copyfile(args.build_manifest.resolve().parent / "ctest.txt", output / "ctest.txt")
    environment = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "dataset_role": MODE_ROLE[mode],
        "source_commit": args.source_commit,
        "build_manifest_sha256": build_hash,
        "environment_id": build["environment_id"],
        "environment_sha256": build["environment_sha256"],
        "openmp_threads": list(threads),
        "topology": build["openmp_topology"],
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "measured_bodies": list(MODE_BODIES[mode]),
        "steps": steps,
        "dt_s": 10.0,
        "repetitions": repetitions,
        "warmups_before_each_measurement": 1,
        "benchmark_seed": args.seed,
        "order_seed": args.order_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "phase_gate": gate,
    }
    write_json(output / "environment.json", environment)
    files = {}
    for name in ("raw_timings.csv", "summary.csv", "environment.json", "build_manifest.json", "ctest.txt"):
        path = output / name
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "source_commit": args.source_commit,
        "build_manifest_sha256": build_hash,
        "environment_sha256": build["environment_sha256"],
        "files": files,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Stage 5 v2 {mode} acquisition complete; verify before continuing")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Stage5Error, ValueError, FileNotFoundError, KeyError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
