#!/usr/bin/env python3
"""Create a portable N=10-only V5 capture bundle for local comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialize(source: Path, target: Path, *, hardlink: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"Existing target has wrong size: {target}")
        return "existing"
    if hardlink:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copy"


def export_bundle(
    capture_index: Path,
    output: Path,
    *,
    hardlink: bool,
) -> dict[str, Any]:
    rows = [
        row for row in _read_jsonl(capture_index) if int(row.get("gold_count", -1)) == 10
    ]
    if not rows:
        raise ValueError(f"No N=10 rows in {capture_index}")
    output.mkdir(parents=True, exist_ok=True)
    output_rows = []
    file_rows = []
    for row in rows:
        request_stem = str(row["request_id"]).replace("/", "__")
        copied = dict(row)
        for key, filename in (
            ("states_path", "states.npz"),
            ("manifest_path", "capture_manifest.json"),
        ):
            source = (capture_index.parent / str(row[key])).resolve()
            relative = Path("shards") / request_stem / filename
            target = output / relative
            mode = _materialize(source, target, hardlink=hardlink)
            copied[key] = relative.as_posix()
            file_rows.append(
                {
                    "request_id": row["request_id"],
                    "kind": key,
                    "path": relative.as_posix(),
                    "bytes": int(target.stat().st_size),
                    "sha256": _sha256(target),
                    "materialization": mode,
                }
            )
        output_rows.append(copied)
    index_path = output / "capture_index.jsonl"
    temporary = index_path.with_name(index_path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    temporary.replace(index_path)
    audit = {
        "schema_version": "realistic_niah_v5_n10_portable_capture_v1",
        "source_capture_index": str(capture_index.resolve()),
        "rows": len(output_rows),
        "seeds": sorted({int(row["seed"]) for row in output_rows}),
        "splits": sorted({str(row["split"]) for row in output_rows}),
        "files": file_rows,
        "index_sha256": _sha256(index_path),
    }
    audit_path = output / "export_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardlink", action="store_true")
    args = parser.parse_args()
    audit = export_bundle(args.capture_index, args.output, hardlink=args.hardlink)
    print(json.dumps({"rows": audit["rows"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
