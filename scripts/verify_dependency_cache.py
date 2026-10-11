#!/usr/bin/env python3
"""Verify the optional persistent visualization dependency source cache."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tools" / "dependency_cache_manifest.json"


def git(directory: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed for {directory}: {detail}")
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-present",
        action="store_true",
        help="fail instead of reporting unavailable when the cache is absent",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cache_root = ROOT / manifest["storage_root"]
    dependencies = manifest["dependencies"]
    present = [(cache_root / item["cache_directory"]).is_dir() for item in dependencies]
    if not any(present):
        message = (
            "Persistent dependency cache is not installed. Because .tools is ignored, "
            "another machine must provision every manifest-named checkout before an "
            "offline build; otherwise pinned FetchContent network acquisition is required."
        )
        if args.require_present:
            print(message, file=sys.stderr)
            return 1
        print(message)
        return 0
    if not all(present):
        missing = [
            item["cache_directory"]
            for item, exists in zip(dependencies, present, strict=True)
            if not exists
        ]
        print(f"Dependency cache is incomplete; missing: {', '.join(missing)}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for item in dependencies:
        directory = cache_root / item["cache_directory"]
        try:
            head = git(directory, "rev-parse", "HEAD")
            tree = git(directory, "rev-parse", "HEAD^{tree}")
            requested = git(directory, "rev-parse", f"{item['requested_git_tag']}^{{commit}}")
            origin = git(directory, "remote", "get-url", "origin")
            status = git(directory, "status", "--porcelain", "--untracked-files=all")
        except RuntimeError as error:
            failures.append(str(error))
            continue
        if head != item["resolved_commit"]:
            failures.append(f"{item['name']}: HEAD {head} != {item['resolved_commit']}")
        if tree != item["git_tree"]:
            failures.append(f"{item['name']}: tree {tree} != {item['git_tree']}")
        if requested != item["resolved_commit"]:
            failures.append(
                f"{item['name']}: requested tag resolves to {requested}, expected {item['resolved_commit']}"
            )
        if origin != item["repository"]:
            failures.append(f"{item['name']}: origin {origin!r} != {item['repository']!r}")
        if status:
            failures.append(f"{item['name']}: checkout is not clean")

    if failures:
        print("Dependency cache verification FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        "Dependency cache verified: "
        + ", ".join(f"{item['name']} {item['version']}" for item in dependencies)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
