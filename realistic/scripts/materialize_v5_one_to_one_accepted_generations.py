#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "realistic_niah_v5_accepted_generation_materialization_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL {path}:{line_number}: {error}"
                ) from error
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(root: Path, model: str, output: Path) -> dict[str, Any]:
    model_root = root / model
    ledger_path = model_root / "accepted_supplement.jsonl"
    ledger = read_jsonl(ledger_path)
    accepted = [row for row in ledger if bool(row.get("accepted"))]
    by_batch: dict[str, list[dict[str, Any]]] = {}
    for row in accepted:
        by_batch.setdefault(str(row["batch"]), []).append(row)
    generation_by_id: dict[str, dict[str, Any]] = {}
    input_files = []
    for batch in sorted(by_batch):
        generation_path = model_root / "batches" / batch / "generations.jsonl"
        input_files.append(
            {"path": str(generation_path.resolve()), "sha256": sha256(generation_path)}
        )
        for row in read_jsonl(generation_path):
            request_id = str(row.get("request_id", row.get("stimulus_id")))
            if request_id in generation_by_id:
                raise ValueError(f"Duplicate generation request_id: {request_id}")
            generation_by_id[request_id] = row
    rows = []
    for ledger_row in accepted:
        request_id = str(ledger_row["request_id"])
        if request_id not in generation_by_id:
            raise KeyError(f"Accepted request missing generation row: {request_id}")
        row = dict(generation_by_id[request_id])
        if int(row["seed"]) != int(ledger_row["seed"]):
            raise ValueError(f"Seed mismatch for {request_id}")
        row["supplement_materialization"] = {
            "schema_version": SCHEMA,
            "accepted_ledger": str(ledger_path.resolve()),
            "batch": str(ledger_row["batch"]),
            "merge_into_primary_300": False,
        }
        rows.append(row)
    seeds = [int(row["seed"]) for row in rows]
    request_ids = [str(row.get("request_id", row.get("stimulus_id"))) for row in rows]
    if len(seeds) != len(set(seeds)) or len(request_ids) != len(set(request_ids)):
        raise ValueError("Accepted supplement rows are not seed/request unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)
    audit = {
        "schema_version": SCHEMA,
        "model_label": model,
        "accepted_rows": len(rows),
        "seeds": sorted(seeds),
        "splits": {
            split: sum(str(row.get("split")) == split for row in rows)
            for split in ("discovery", "confirmation")
        },
        "gold_counts": sorted(
            {len(row.get("gold_records", [])) for row in rows}
        ),
        "merge_into_primary_300": False,
        "accepted_ledger": {
            "path": str(ledger_path.resolve()),
            "sha256": sha256(ledger_path),
        },
        "input_generation_files": input_files,
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(args.supplement_root, args.model, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
