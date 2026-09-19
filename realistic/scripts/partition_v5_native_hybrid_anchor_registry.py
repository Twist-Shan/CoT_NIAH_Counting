#!/usr/bin/env python3
"""Verify a frozen full registry and write mutually exclusive grammar views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    model_spec = spec["models"][args.model]
    rows = [
        json.loads(line)
        for line in args.registry.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observed_sha = hashlib.sha256(args.registry.read_bytes()).hexdigest()
    expected_rows = int(model_spec["registry_rows"])
    expected_sha = str(model_spec["registry_sha256"])
    if len(rows) != expected_rows:
        raise AssertionError(("registry_rows", len(rows), expected_rows))
    if observed_sha != expected_sha:
        raise AssertionError(("registry_sha256", observed_sha, expected_sha))

    args.output.mkdir(parents=True, exist_ok=True)
    grammars = list(model_spec["grammars"])
    written = 0
    counts = {}
    for grammar in grammars:
        selected = [
            row for row in rows if row["target_grammar_class"] == grammar
        ]
        path = args.output / f"{grammar}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
            encoding="utf-8",
        )
        counts[grammar] = len(selected)
        written += len(selected)
    if written != len(rows):
        unexpected = sorted(
            {row["target_grammar_class"] for row in rows} - set(grammars)
        )
        raise AssertionError(("registry_partition", written, len(rows), unexpected))
    audit = {
        "schema_version": "realistic_niah_v5_native_hybrid_registry_partition_v1",
        "status": "PASS",
        "model_label": args.model,
        "registry_rows": len(rows),
        "registry_sha256": observed_sha,
        "grammar_counts": counts,
    }
    (args.output / "partition_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
