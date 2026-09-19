#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "realistic_niah_v5_answer_query_capture_merge_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path, source: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL {path}:{line_number}: {error}") from error
        for field in ("manifest_path", "states_path"):
            value = Path(str(row[field]))
            resolved = value if value.is_absolute() else path.parent / value
            resolved = resolved.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Capture index target is missing: {path}:{line_number} "
                    f"{field}={resolved}"
                )
            # The combined index lives in a third directory, so source-relative
            # paths are invalid there.  Absolute paths preserve the isolated
            # primary/supplement shards without copying or merging state files.
            row[field] = str(resolved)
        row["analysis_sample_source"] = source
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    primary = read(args.primary, "primary_registered_300")
    supplement = read(args.supplement, "isolated_n10_one_to_one_supplement")
    rows = primary + supplement
    keys = [
        (
            str(row.get("request_id")),
            str(row.get("site_id")),
            int(row.get("layer", -1)),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Primary/supplement capture merge has duplicate request/site/layer")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    audit = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "primary": str(args.primary.resolve()),
        "primary_sha256": sha256(args.primary),
        "primary_rows": len(primary),
        "supplement": str(args.supplement.resolve()),
        "supplement_sha256": sha256(args.supplement),
        "supplement_rows": len(supplement),
        "combined_rows": len(rows),
        "unique_request_site_layer_keys": len(set(keys)),
        "supplement_not_written_into_primary": True,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
