#!/usr/bin/env python3
"""Audit whether every marker pair supports each registered pre-city query."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_tokenizer
from realistic_niah_v4.spec import resolve_model_spec
from realistic_niah_v5.pre_city import pre_city_token_queries


SCHEMA = "realistic_niah_v5_marker_query_constructability_audit_v1"
VARIANTS = ("pre_city_d1", "pre_city_d2", "pre_city_anchor")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def request_id(row: Mapping[str, Any]) -> str:
    return str(row.get("request_id", row.get("stimulus_id")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generations: dict[str, dict[str, Any]] = {}
    generation_inputs = []
    for path in args.generations:
        generation_inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
        for row in read_jsonl(path):
            if str(row.get("model_label", row.get("model"))) != args.model:
                continue
            generations.setdefault(request_id(row), row)
    tokenizer = load_registered_tokenizer(
        resolve_model_spec(args.model), cache_dir=args.cache_dir
    )
    audit_rows = []
    for pair in read_jsonl(args.pairs):
        pair_id = str(pair["pair_id"])
        full_id = str(pair["full_request_id"])
        if full_id not in generations:
            raise ValueError(f"Full generation missing for pair {pair_id}: {full_id}")
        queries, exclusions = pre_city_token_queries(generations[full_id], tokenizer)
        occurrence = int(pair["occurrence"])
        matches = {
            variant: [
                query
                for query in queries
                if query.query_variant == variant and int(query.occurrence) == occurrence
            ]
            for variant in VARIANTS
        }
        constructible = {variant: len(values) == 1 for variant, values in matches.items()}
        query_counts = {
            variant: int(values[0].query_output_token_count) if len(values) == 1 else None
            for variant, values in matches.items()
        }
        audit_rows.append(
            {
                "schema_version": SCHEMA,
                "model_label": args.model,
                "pair_id": pair_id,
                "split": str(pair["split"]),
                "seed": int(pair["seed"]),
                "counterfactual_count": int(pair["counterfactual_count"]),
                "full_count": int(pair["full_count"]),
                "occurrence": occurrence,
                "all_required_constructible": all(constructible.values()),
                "variant_constructible": constructible,
                "query_output_token_count": query_counts,
                "alias_groups": {
                    str(count): sorted(
                        variant for variant, value in query_counts.items() if value == count
                    )
                    for count in sorted({value for value in query_counts.values() if value is not None})
                },
                "relevant_exclusions": [
                    row for row in exclusions if int(row.get("occurrence", -1)) == occurrence
                ],
            }
        )
    atomic_jsonl(args.output, audit_rows)
    passed = [row for row in audit_rows if row["all_required_constructible"]]
    failed = [row for row in audit_rows if not row["all_required_constructible"]]
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "model_label": args.model,
        "required_variants": list(VARIANTS),
        "pairs": len(audit_rows),
        "fully_constructible_pairs": len(passed),
        "excluded_pairs": len(failed),
        "excluded_pair_ids": [row["pair_id"] for row in failed],
        "generation_inputs": generation_inputs,
        "source_pairs": str(args.pairs.resolve()),
        "source_pairs_sha256": sha256(args.pairs),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
