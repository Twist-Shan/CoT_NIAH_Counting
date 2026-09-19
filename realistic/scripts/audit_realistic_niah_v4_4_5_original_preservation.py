#!/usr/bin/env python3
"""Verify that a Top-6 extension did not mutate historical artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--path", type=Path, action="append", default=[])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.snapshot:
        if args.baseline is not None or not args.path:
            parser.error("--snapshot requires one or more --path and no --baseline")
        rows: list[dict[str, Any]] = []
        for path in args.path:
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            rows.append(
                {
                    "path": str(path.resolve()),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path),
                }
            )
        payload = {
            "schema_version": (
                "realistic_niah_v4_4_5_original_preservation_snapshot_v1"
            ),
            "status": "SNAPSHOT",
            "auditor_sha256": sha256_file(Path(__file__)),
            "files": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    if args.baseline is None or args.path:
        parser.error("comparison mode requires --baseline and no --path")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("status") != "SNAPSHOT" or not baseline.get("files"):
        raise RuntimeError("Invalid original-artifact preservation snapshot")

    comparisons: list[dict[str, Any]] = []
    all_pass = True
    for before in baseline["files"]:
        path = Path(before["path"])
        exists = path.is_file()
        after = (
            {
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256_file(path),
            }
            if exists
            else None
        )
        unchanged = bool(
            exists
            and after is not None
            and int(before["bytes"]) == int(after["bytes"])
            and int(before["mtime_ns"]) == int(after["mtime_ns"])
            and str(before["sha256"]) == str(after["sha256"])
        )
        all_pass &= unchanged
        comparisons.append(
            {
                "path": str(path),
                "exists": exists,
                "before": {
                    "bytes": int(before["bytes"]),
                    "mtime_ns": int(before["mtime_ns"]),
                    "sha256": str(before["sha256"]),
                },
                "after": after,
                "unchanged": unchanged,
            }
        )

    payload = {
        "schema_version": (
            "realistic_niah_v4_4_5_original_preservation_audit_v1"
        ),
        "status": "PASS" if all_pass else "FAIL",
        "definition": (
            "Every snapshotted historical file must retain identical bytes, "
            "mtime_ns, and SHA-256 after the isolated Top-6 campaign."
        ),
        "baseline": str(args.baseline),
        "auditor_sha256": sha256_file(Path(__file__)),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
