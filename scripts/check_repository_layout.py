#!/usr/bin/env python3
"""Non-destructively enforce the repository's canonical build layout."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tools" / "repository_layout_baseline.json"
GENERATED_SUFFIXES = {
    ".a",
    ".csv",
    ".dll",
    ".exe",
    ".ilk",
    ".lib",
    ".log",
    ".o",
    ".obj",
    ".out",
    ".pdb",
    ".profraw",
    ".so",
    ".tmp",
}
TRACKED_BINARY_SUFFIXES = {
    ".a", ".dll", ".exe", ".ilk", ".lib", ".o", ".obj", ".pdb", ".so"
}
PRUNED_ROOT_DIRECTORIES = {
    ".git", ".tools", ".cache", ".vs", ".vscode", "build", "data"
}
UNIQUE_EVIDENCE_DIRECTORIES = {
    "build-stage3-pilot",
    "build-stage6d-pilot-smoke",
}


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def git_tracked_files(failures: list[str]) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        fail("git ls-files failed; repository layout could not be verified", failures)
        return set()
    return {
        value.decode("utf-8", errors="strict").replace("\\", "/")
        for value in completed.stdout.split(b"\0")
        if value
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    try:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"repository layout check failed: cannot read baseline: {error}", file=sys.stderr)
        return 1

    legacy_builds = baseline.get("legacy_root_build_directories")
    legacy_artifacts = baseline.get("legacy_root_artifacts")
    if not isinstance(legacy_builds, list) or not isinstance(legacy_artifacts, list):
        print("repository layout check failed: malformed baseline", file=sys.stderr)
        return 1
    if len(legacy_builds) != len(set(legacy_builds)):
        fail("baseline contains duplicate legacy build directory names", failures)
    for name in legacy_builds:
        if not isinstance(name, str) or not name.startswith("build-") or "/" in name or "\\" in name:
            fail(f"invalid legacy build baseline entry: {name!r}", failures)

    current_legacy_style = {
        entry.name
        for entry in ROOT.iterdir()
        if entry.is_dir() and entry.name.startswith("build-")
    }
    unexpected_builds = sorted(current_legacy_style - set(legacy_builds))
    for name in unexpected_builds:
        fail(
            f"unexpected root build directory {name!r}; use a canonical preset or build/scratch/<purpose>",
            failures,
        )

    for forbidden in ("CMakeFiles", "Testing", "out"):
        if (ROOT / forbidden).exists():
            fail(f"unexpected generated root directory {forbidden!r}", failures)
    for entry in ROOT.iterdir():
        if entry.is_dir() and entry.name.startswith("cmake-build-"):
            fail(f"unexpected generated root directory {entry.name!r}", failures)

    baseline_artifact_paths: set[str] = set()
    for record in legacy_artifacts:
        if not isinstance(record, dict):
            fail("legacy artifact baseline entry is not an object", failures)
            continue
        relative = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(relative, str) or Path(relative).name != relative:
            fail(f"invalid legacy root artifact path: {relative!r}", failures)
            continue
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            fail(f"invalid legacy root artifact hash for {relative!r}", failures)
            continue
        baseline_artifact_paths.add(relative)
        candidate = ROOT / relative
        if candidate.exists() and sha256(candidate) != expected_hash.lower():
            fail(f"grandfathered root artifact changed bytes: {relative}", failures)

    tracked = git_tracked_files(failures)
    for relative in sorted(tracked):
        if Path(relative).suffix.lower() in TRACKED_BINARY_SUFFIXES:
            fail(f"generated build binary is tracked by Git: {relative}", failures)

    pruned = PRUNED_ROOT_DIRECTORIES | set(legacy_builds)
    for entry in ROOT.iterdir():
        if entry.name in pruned or entry.name in baseline_artifact_paths:
            continue
        candidates = entry.rglob("*") if entry.is_dir() else (entry,)
        for candidate in candidates:
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(ROOT).as_posix()
            if relative in tracked:
                continue
            lower_name = candidate.name.lower()
            generated_name = lower_name.startswith("validation-") or lower_name.startswith("test-results")
            if candidate.suffix.lower() in GENERATED_SUFFIXES or generated_name:
                fail(f"untracked generated artifact outside an approved location: {relative}", failures)

    if failures:
        print("Repository layout check FAILED:", file=sys.stderr)
        for message in failures:
            print(f"- {message}", file=sys.stderr)
        print("The checker is non-destructive; no files were changed.", file=sys.stderr)
        return 1

    present_baseline = current_legacy_style & set(legacy_builds)
    evidence_present = sorted(present_baseline & UNIQUE_EVIDENCE_DIRECTORIES)
    print(
        "Repository layout check passed: canonical build root is build/; "
        f"{len(present_baseline)} grandfathered build directories remain."
    )
    if evidence_present:
        print(
            "Protected pending evidence review: " + ", ".join(evidence_present)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
