#!/usr/bin/env python3
"""Audit frozen native traces for naturally occurring count/ordinal dissociations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _final_answer_exact(row: Mapping[str, Any], parsed: Mapping[str, Any]) -> bool:
    for value in (row.get("exact_count"), parsed.get("exact_count")):
        if value is not None:
            return bool(value)
    return int(parsed.get("parsed_count", -1)) == int(parsed.get("gold_count", -2))


def _flags(row: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    parsed = row.get("trace_parse") if isinstance(row.get("trace_parse"), Mapping) else {}
    parser = parsed.get("parser") if isinstance(parsed.get("parser"), Mapping) else {}
    gold_count = len(row.get("gold_records", row.get("gold_pairs", [])))
    item_count = int(parser.get("item_count", 0) or 0)
    raw_markers = [value for value in parser.get("item_markers", []) if value is not None]
    numeric_markers = [
        int(str(value)) for value in raw_markers if re.fullmatch(r"\d+", str(value))
    ]
    expected_markers = list(range(1, item_count + 1))
    reasoning = str(parsed.get("reasoning_text", row.get("raw_output_text", "")))
    cut = int(parser.get("cut_char", len(reasoning)) or len(reasoning))
    trailing = reasoning[max(0, min(cut, len(reasoning))) :]
    flags: list[str] = []
    if not bool(parser.get("detected")):
        flags.append("no_detected_list")
    if str(parser.get("trace_category", "")) != "one_to_one":
        flags.append("not_one_to_one")
    if item_count != gold_count:
        flags.append("item_count_vs_gold")
    if int(parser.get("duplicate_gold_city_items", 0) or 0) > 0:
        flags.append("duplicate_gold_city")
    if parser.get("missing_gold_cities"):
        flags.append("missing_gold_city")
    if parser.get("all_items_gold_city") is False:
        flags.append("non_gold_item")
    if numeric_markers and len(numeric_markers) == len(raw_markers) and numeric_markers != expected_markers:
        flags.append("nonconsecutive_explicit_markers")
    if int(parser.get("rejected_candidates", 0) or 0) > 0:
        flags.append("rejected_list_candidate")
    if int(parser.get("candidates_considered", 0) or 0) > 1:
        flags.append("multiple_list_candidates")
    if str(parser.get("trace_order_class", "")) not in ("", "forward"):
        flags.append("nonforward_order")
    if not _final_answer_exact(row, parsed):
        flags.append("wrong_final_answer")
    correction_hits = re.findall(
        r"(?i)\b(?:wait|actually|correction|correcting|duplicate|exclude|not count|"
        r"shouldn't count|should not count|missed|skip(?:ped)?|recount|check again|"
        r"double[- ]check)\b",
        reasoning,
    )
    trailing_item_lines = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+.+$", trailing)
    if correction_hits:
        flags.append("correction_language")
    if trailing_item_lines:
        flags.append("post_list_enumeration")
    audit = {
        "request_id": str(row.get("request_id", row.get("stimulus_id", ""))),
        "model_label": str(row.get("model_label", row.get("model", ""))),
        "seed": int(row.get("seed", -1)),
        "gold_count": int(gold_count),
        "final_answer_exact": _final_answer_exact(row, parsed),
        "detected": bool(parser.get("detected")),
        "trace_category": str(parser.get("trace_category", "")),
        "trace_one_to_one": bool(parser.get("trace_one_to_one")),
        "item_count": item_count,
        "item_markers": raw_markers,
        "numeric_item_markers": numeric_markers,
        "item_gold_cities": list(parser.get("item_gold_cities", [])),
        "duplicate_gold_city_items": int(parser.get("duplicate_gold_city_items", 0) or 0),
        "missing_gold_cities": list(parser.get("missing_gold_cities", [])),
        "all_items_gold_city": parser.get("all_items_gold_city"),
        "rejected_candidates": int(parser.get("rejected_candidates", 0) or 0),
        "candidates_considered": int(parser.get("candidates_considered", 0) or 0),
        "marker_kind": str(parser.get("marker_kind", "")),
        "trace_order_class": str(parser.get("trace_order_class", "")),
        "cut_char": cut,
        "reasoning_char_count": len(reasoning),
        "correction_terms": correction_hits,
        "trailing_item_line_count": len(trailing_item_lines),
        "flags": sorted(set(flags)),
        "list_excerpt": reasoning[max(0, cut - 700) : min(len(reasoning), cut + 700)],
        "reasoning_excerpt": reasoning[: min(len(reasoning), 1800)],
    }
    return sorted(set(flags)), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    joint_counts: Counter[str] = Counter()
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    total_by_model: Counter[str] = Counter()
    for path in args.generations:
        for row in read_jsonl(path):
            flags, audit = _flags(row)
            rows.append(audit)
            model = str(audit["model_label"])
            total_by_model[model] += 1
            for flag in flags:
                flag_counts[flag] += 1
                by_model[model][flag] += 1
            joint_counts["+".join(flags) if flags else "clean_no_flags"] += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "natural_anomaly_candidates.jsonl", rows)
    write_json(
        args.output_dir / "natural_anomaly_audit.json",
        {
            "schema_version": "realistic_niah_v5_natural_anomaly_audit_v1",
            "inputs": [str(path) for path in args.generations],
            "total_rows": len(rows),
            "total_by_model": dict(sorted(total_by_model.items())),
            "flag_counts": dict(sorted(flag_counts.items())),
            "flag_counts_by_model": {
                model: dict(sorted(counts.items())) for model, counts in sorted(by_model.items())
            },
            "joint_flag_counts": dict(joint_counts.most_common()),
        },
    )
    print(json.dumps({"rows": len(rows), "flag_counts": dict(flag_counts)}, indent=2))


if __name__ == "__main__":
    main()
