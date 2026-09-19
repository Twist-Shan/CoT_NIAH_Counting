#!/usr/bin/env python3
"""Freeze clean explicit-index cohorts for progress-state positive controls.

The selected surface grammars and 20/10 splits predate this intervention and
are read from their existing text-only selection files.  No hidden state,
patch score, attention value, or generated continuation is used here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "realistic_niah_v5_indexed_progress_control_cohort_v1"
CONTRACTS = {
    "Qwen3-8B": {
        "grammar_class": "adjacent_rank_before_city",
        "marker_kind": "indexed",
        "surface_template": "k. City - score",
        "endpoint_family": "score_digit",
    },
    "Gemma4-E4B": {
        "grammar_class": "same_unit_rank_before_city",
        "marker_kind": "inline_count",
        "surface_template": "*   Record k: (City, score)",
        "endpoint_family": "bare_closing_parenthesis_after_city_and_score",
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _selected_items(row: Mapping[str, Any]) -> tuple[str, ...]:
    parser = dict(row.get("trace_parse", {}).get("parser", {}))
    starts = tuple(int(value) for value in parser.get("item_start_chars", ()))
    ends = tuple(int(value) for value in parser.get("item_end_chars", ()))
    if len(starts) != 10 or len(ends) != 10:
        raise ValueError("Indexed control row does not expose exactly ten item spans")
    raw = str(row.get("raw_output_text", ""))
    return tuple(raw[start:end].strip() for start, end in zip(starts, ends))


def audit_row(row: Mapping[str, Any], *, model_label: str) -> dict[str, Any]:
    """Validate a row against one exact, outcome-blind surface contract."""

    if model_label not in CONTRACTS:
        raise KeyError(f"Unknown indexed-control model: {model_label}")
    contract = CONTRACTS[model_label]
    parser = dict(row.get("trace_parse", {}).get("parser", {}))
    gold = tuple(
        (str(value["city"]), int(value["score"]))
        for value in row.get("gold_records", ())
    )
    reasons: list[str] = []
    if str(row.get("model_label")) != model_label:
        reasons.append("model_mismatch")
    if int(row.get("gold_count", -1)) != 10 or len(gold) != 10:
        reasons.append("not_complete_N10")
    if int(parser.get("item_count", -1)) != 10:
        reasons.append("parser_item_count_not_10")
    if not bool(parser.get("trace_one_to_one")):
        reasons.append("trace_not_one_to_one")
    markers = tuple(int(value) for value in parser.get("item_markers", ()))
    if markers != tuple(range(1, 11)):
        reasons.append("visible_indices_not_exact_1_to_10")
    parsed_cities = tuple(str(value) for value in parser.get("item_gold_cities", ()))
    gold_by_city = {city: score for city, score in gold}
    if (
        len(parsed_cities) != 10
        or len(set(parsed_cities)) != 10
        or set(parsed_cities) != set(gold_by_city)
    ):
        reasons.append("parsed_cities_not_one_to_one_with_gold_records")
    try:
        items = _selected_items(row)
    except ValueError:
        items = ()
        reasons.append("item_spans_not_exactly_ten")
    if (
        len(items) == 10
        and len(parsed_cities) == 10
        and len(gold_by_city) == 10
        and set(parsed_cities) == set(gold_by_city)
    ):
        expected = tuple(
            (
                f"{index}. {city} - {score}"
                if model_label == "Qwen3-8B"
                else f"*   Record {index}: ({city}, {score})"
            )
            for index, city in enumerate(parsed_cities, start=1)
            for score in (gold_by_city[city],)
        )
        if items != expected:
            reasons.append("surface_template_mismatch")
    else:
        expected = ()
    return {
        "status": "PASS" if not reasons else "FAIL",
        "primary_eligible_indexed_positive_control": not reasons,
        "reasons": reasons,
        "model_label": model_label,
        "gold_count": int(row.get("gold_count", -1)),
        "grammar_class": str(contract["grammar_class"]),
        "marker_kind": str(contract["marker_kind"]),
        "surface_template": str(contract["surface_template"]),
        "endpoint_family": str(contract["endpoint_family"]),
        "visible_indices": list(range(1, 11)),
        "exact_surface_items_sha256": _json_sha256(items),
        "expected_surface_items_sha256": _json_sha256(expected),
        "selection_uses_hidden_states": False,
        "selection_uses_patch_outcomes": False,
        "internal_counter_without_visible_index_claim_allowed": False,
    }


def _selection_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _source_index(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            request_id = str(row.get("request_id", ""))
            if not request_id:
                continue
            previous = selected.get(request_id)
            if previous is not None and _json_sha256(previous) != _json_sha256(row):
                raise ValueError(f"Conflicting source rows for {request_id}")
            selected[request_id] = row
    return selected


def _freeze_model(
    *,
    model_label: str,
    selection: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    contract = CONTRACTS[model_label]
    if len(selection) != 30:
        raise ValueError(f"{model_label} selection is not 30 rows")
    split_counts = {split: 0 for split in ("discovery", "confirmation")}
    frozen: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for rank, selected in enumerate(selection, start=1):
        request_id = str(selected["request_id"])
        if request_id not in source_rows:
            raise ValueError(f"Selected request is absent from sources: {request_id}")
        row = dict(source_rows[request_id])
        seed = int(row["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate {model_label} seed: {seed}")
        seen_seeds.add(seed)
        split = str(selected.get("split", row.get("split", "")))
        if split not in split_counts:
            raise ValueError(f"Unknown split {split!r} for {request_id}")
        split_counts[split] += 1
        row["split"] = split
        audit = audit_row(row, model_label=model_label)
        if not audit["primary_eligible_indexed_positive_control"]:
            raise ValueError(f"{request_id} failed exact grammar audit: {audit['reasons']}")
        row["indexed_progress_control_format_audit"] = audit
        row["indexed_progress_control_cohort"] = {
            "schema_version": SCHEMA_VERSION,
            "selection_population": "indexed_positive_control",
            "model_label": model_label,
            "rank": rank,
            "split": split,
            "grammar_class": str(contract["grammar_class"]),
            "surface_template": str(contract["surface_template"]),
            "visible_progress_confound_allowed": True,
            "internal_counter_without_visible_index_claim_allowed": False,
            "selection_independent_of_hidden_states": True,
            "selection_independent_of_patch_outcomes": True,
        }
        frozen.append(row)
    if split_counts != {"discovery": 20, "confirmation": 10}:
        raise ValueError(f"{model_label} split is not 20/10: {split_counts}")
    frozen.sort(key=lambda row: (str(row["split"]), int(row["seed"])))
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-selection", type=Path, required=True)
    parser.add_argument("--qwen-generations", type=Path, nargs="+", required=True)
    parser.add_argument("--gemma-selection", type=Path, required=True)
    parser.add_argument("--gemma-generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qwen_selection = _selection_csv(args.qwen_selection)
    gemma_selection = _selection_csv(args.gemma_selection)
    qwen = _freeze_model(
        model_label="Qwen3-8B",
        selection=qwen_selection,
        source_rows=_source_index(tuple(args.qwen_generations)),
    )
    gemma = _freeze_model(
        model_label="Gemma4-E4B",
        selection=gemma_selection,
        source_rows=_source_index((args.gemma_generations,)),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "Qwen3-8B": args.output / "Qwen3-8B.jsonl",
        "Gemma4-E4B": args.output / "Gemma4-E4B.jsonl",
    }
    _write_jsonl(outputs["Qwen3-8B"], qwen)
    _write_jsonl(outputs["Gemma4-E4B"], gemma)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "claim_role": "explicit-index positive control only",
        "internal_counter_without_visible_index_claim_allowed": False,
        "selection_independent_of_hidden_states": True,
        "selection_independent_of_patch_outcomes": True,
        "models": {
            model: {
                **CONTRACTS[model],
                "seed_count": len(rows),
                "discovery_seeds": [
                    int(row["seed"]) for row in rows if row["split"] == "discovery"
                ],
                "confirmation_seeds": [
                    int(row["seed"])
                    for row in rows
                    if row["split"] == "confirmation"
                ],
                "output": str(outputs[model]),
                "output_sha256": _sha256(outputs[model]),
            }
            for model, rows in (("Qwen3-8B", qwen), ("Gemma4-E4B", gemma))
        },
        "sources": {
            "qwen_selection": str(args.qwen_selection),
            "qwen_selection_sha256": _sha256(args.qwen_selection),
            "qwen_generations": [str(path) for path in args.qwen_generations],
            "qwen_generation_sha256": {
                str(path): _sha256(path) for path in args.qwen_generations
            },
            "gemma_selection": str(args.gemma_selection),
            "gemma_selection_sha256": _sha256(args.gemma_selection),
            "gemma_generations": str(args.gemma_generations),
            "gemma_generations_sha256": _sha256(args.gemma_generations),
        },
    }
    _write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
