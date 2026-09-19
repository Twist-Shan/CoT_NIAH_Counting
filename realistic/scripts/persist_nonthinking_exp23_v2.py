#!/usr/bin/env python3
"""Persist one model's Exp23 V2 artifacts and verify every copied file by SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(root: Path, model: str) -> list[dict[str, object]]:
    model_dir = root / model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"missing model directory: {model_dir}")
    rows: list[dict[str, object]] = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": f"model/{path.relative_to(model_dir).as_posix()}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    for path in sorted(root.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "path": f"meta/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def destination_manifest(root: Path, model: str) -> list[dict[str, object]]:
    model_dir = root / model
    meta_dir = root / "_host_meta" / model
    rows: list[dict[str, object]] = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": f"model/{path.relative_to(model_dir).as_posix()}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    if meta_dir.is_dir():
        for path in sorted(meta_dir.iterdir()):
            if path.is_file():
                rows.append(
                    {
                        "path": f"meta/{path.name}",
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    return rows


def manifest_digest(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    source_model = source_root / args.model
    destination_model = destination_root / args.model
    destination_meta = destination_root / "_host_meta" / args.model
    destination_model.mkdir(parents=True, exist_ok=True)
    destination_meta.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["rsync", "-aH", "--partial", f"{source_model}/", f"{destination_model}/"],
        check=True,
    )
    for source_file in sorted(source_root.iterdir()):
        if source_file.is_file():
            subprocess.run(
                ["rsync", "-aH", "--partial", str(source_file), f"{destination_meta}/"],
                check=True,
            )

    source_rows = model_manifest(source_root, args.model)
    destination_rows = destination_manifest(destination_root, args.model)
    source_digest = manifest_digest(source_rows)
    destination_digest = manifest_digest(destination_rows)
    passed = source_rows == destination_rows and source_digest == destination_digest
    audit = {
        "schema_version": "realistic_niah_v4_4_5_exp23_v2_persistence_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "model": args.model,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "file_count": len(source_rows),
        "total_bytes": sum(int(row["bytes"]) for row in source_rows),
        "source_manifest_sha256": source_digest,
        "destination_manifest_sha256": destination_digest,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "copy_semantics": "rsync -aH --partial; no deletion",
    }
    audit_dir = destination_root / "_persistence" / args.model
    write_json(audit_dir / "source_manifest.json", source_rows)
    write_json(audit_dir / "destination_manifest.json", destination_rows)
    write_json(audit_dir / "persistence_audit.json", audit)
    print(json.dumps(audit, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
