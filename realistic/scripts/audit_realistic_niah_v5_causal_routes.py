#!/usr/bin/env python3
"""Audit grammar-conditioned retrieval transitions and candidate query sites."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_tokenizer
from realistic_niah_v4.spec import resolve_model_spec
from realistic_niah_v5.causal_sites import (
    CausalSiteError,
    compile_causal_site_plan,
    flatten_anchor_rows,
    flatten_transition_rows,
)
from realistic_niah_v5.pipeline import read_jsonl


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _event_text(event: Mapping[str, Any]) -> str:
    site = event.get("sites", {}).get("semantic_item_span", {})
    return " ".join(str(site.get("char_text", "")).split())


def _group_summary(
    transitions: Iterable[Mapping[str, Any]],
    anchors: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    transition_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    anchor_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in transitions:
        key = (
            str(row.get("target_grammar_class", "")),
            str(row.get("target_retrieval_surface_variant", "")),
        )
        transition_groups[key].append(row)
    for row in anchors:
        key = (
            str(row.get("target_grammar_class", "")),
            str(row.get("target_retrieval_surface_variant", "")),
        )
        anchor_groups[key].append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(transition_groups):
        rows = transition_groups[key]
        candidate_rows = anchor_groups.get(key, [])
        role_summary: dict[str, dict[str, int]] = {}
        for role in sorted({str(row.get("anchor_role", "")) for row in candidate_rows}):
            selected = [row for row in candidate_rows if str(row.get("anchor_role", "")) == role]
            role_summary[role] = {
                "candidate_rows": len(selected),
                "resolved_rows": sum(str(row.get("status")) == "ok" for row in selected),
                "local_eligible_rows": sum(bool(row.get("local_anchor_eligible")) for row in selected),
                "primary_eligible_rows": sum(bool(row.get("primary_anchor_eligible")) for row in selected),
            }
        examples: list[dict[str, Any]] = []
        seen_examples: set[tuple[str, str]] = set()
        for row in rows:
            identity = (str(row.get("request_id")), str(row.get("target_item_text")))
            if identity in seen_examples:
                continue
            seen_examples.add(identity)
            examples.append(
                {
                    "request_id": row.get("request_id"),
                    "seed": row.get("seed"),
                    "gold_count": row.get("gold_count"),
                    "from_occurrence": row.get("from_occurrence"),
                    "to_occurrence": row.get("to_occurrence"),
                    "target_item_text": row.get("target_item_text"),
                }
            )
            if len(examples) == 3:
                break
        output.append(
            {
                "target_grammar_class": key[0],
                "target_retrieval_surface_variant": key[1],
                "transition_count": len(rows),
                "trajectory_count": len({str(row.get("request_id")) for row in rows}),
                "seed_count": len({int(row["seed"]) for row in rows}),
                "split_counts": {
                    split: sum(str(row.get("split")) == split for row in rows)
                    for split in sorted({str(row.get("split")) for row in rows})
                },
                "anchor_roles": role_summary,
                "examples": examples,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = load_registered_tokenizer(
        resolve_model_spec(args.model), cache_dir=args.cache_dir
    )
    transition_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in read_jsonl(args.generations):
        if row.get("model_label") not in {None, args.model}:
            continue
        try:
            plan = compile_causal_site_plan(row, tokenizer)
        except CausalSiteError as error:
            errors.append(
                {
                    "request_id": row.get("request_id", row.get("stimulus_id")),
                    "seed": row.get("seed"),
                    "gold_count": row.get("gold_count"),
                    "error": str(error),
                }
            )
            continue
        events = {int(event["occurrence"]): event for event in plan.get("events", [])}
        for value in flatten_transition_rows(plan):
            if value.get("transition_kind") != "continue_to_next_city":
                continue
            target = events.get(int(value["to_occurrence"]))
            transition_rows.append(
                {
                    **value,
                    "target_grammar_class": str(value.get("grammar_pair", "")).rsplit(" -> ", 1)[-1],
                    "target_item_text": "" if target is None else _event_text(target),
                }
            )
        anchor_rows.extend(flatten_anchor_rows(plan))

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "transitions.csv", transition_rows)
    _write_csv(args.output / "anchor_candidates.csv", anchor_rows)
    payload = {
        "schema_version": "realistic_niah_v5_causal_route_audit_v1",
        "model_label": args.model,
        "generations": str(args.generations.resolve()),
        "compiled_trajectory_count": len(
            {str(row.get("request_id")) for row in transition_rows}
        ),
        "transition_count": len(transition_rows),
        "compile_errors": errors,
        "groups": _group_summary(transition_rows, anchor_rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[v5 route audit] transitions={len(transition_rows)} "
        f"errors={len(errors)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
