#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.parsing import parse_trace_record


SCHEMA = "realistic_niah_v5_marker_adjacent_count_patch_plan_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_id(row: dict[str, Any]) -> str:
    return str(row.get("request_id", row.get("stimulus_id")))


def build(
    generations: list[Path],
    model_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    inputs = []
    rows_by_seed_count: dict[tuple[str, int, int], tuple[dict[str, Any], dict[str, Any], str]] = {}
    exclusions = []
    for path in generations:
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
        for row in read_jsonl(path):
            if str(row.get("model_label", row.get("model"))) != model_label:
                continue
            parsed = parse_trace_record(row)
            key = (str(row["split"]), int(row["seed"]), int(parsed["gold_count"]))
            if key in rows_by_seed_count:
                prior = rows_by_seed_count[key][0]
                if request_id(prior) != request_id(row):
                    raise ValueError(f"Duplicate split/seed/count row: {key}")
                continue
            if not bool(parsed["parser"].get("trace_one_to_one")):
                exclusions.append(
                    {"request_id": request_id(row), "reason": "not_strict_one_to_one"}
                )
                continue
            if not bool(parsed.get("exact_count")):
                exclusions.append(
                    {"request_id": request_id(row), "reason": "baseline_answer_incorrect"}
                )
                continue
            cities = tuple(
                str(value) for value in parsed["parser"].get("item_gold_cities", [])
            )
            if len(cities) != int(parsed["gold_count"]):
                exclusions.append(
                    {
                        "request_id": request_id(row),
                        "reason": "incomplete_strict_trace",
                    }
                )
                continue
            rows_by_seed_count[key] = (row, parsed, str(path.resolve()))

    pairs = []
    unavailable = []
    for (split, seed, full_count), (full, full_parsed, full_source) in sorted(
        rows_by_seed_count.items()
    ):
        if full_count <= 1:
            continue
        counterfactual_key = (split, seed, full_count - 1)
        if counterfactual_key not in rows_by_seed_count:
            unavailable.append(
                {
                    "split": split,
                    "seed": seed,
                    "full_count": full_count,
                    "reason": "missing_correct_adjacent_counterfactual",
                }
            )
            continue
        counterfactual, counterfactual_parsed, counterfactual_source = (
            rows_by_seed_count[counterfactual_key]
        )
        full_cities = tuple(
            str(value) for value in full_parsed["parser"]["item_gold_cities"]
        )
        counterfactual_cities = tuple(
            str(value)
            for value in counterfactual_parsed["parser"]["item_gold_cities"]
        )
        if tuple(value.casefold() for value in full_cities[:-1]) != tuple(
            value.casefold() for value in counterfactual_cities
        ):
            unavailable.append(
                {
                    "split": split,
                    "seed": seed,
                    "full_count": full_count,
                    "reason": "adjacent_prompts_not_nested",
                }
            )
            continue
        full_ids = tuple(int(value) for value in full["input_ids"])
        counterfactual_ids = tuple(int(value) for value in counterfactual["input_ids"])
        if len(full_ids) != len(counterfactual_ids):
            unavailable.append(
                {
                    "split": split,
                    "seed": seed,
                    "full_count": full_count,
                    "reason": "prompt_token_length_mismatch",
                }
            )
            continue
        changed = [
            index
            for index, (left, right) in enumerate(zip(full_ids, counterfactual_ids))
            if left != right
        ]
        if not changed:
            unavailable.append(
                {
                    "split": split,
                    "seed": seed,
                    "full_count": full_count,
                    "reason": "no_prompt_counterfactual_change",
                }
            )
            continue
        target_city = full_cities[-1]
        target_spans = [
            span
            for span in full.get("prompt_record_spans", [])
            if str(span["city"]).casefold() == target_city.casefold()
        ]
        if len(target_spans) != 1:
            unavailable.append(
                {
                    "split": split,
                    "seed": seed,
                    "full_count": full_count,
                    "reason": "target_prompt_record_span_not_unique",
                }
            )
            continue
        target_span = target_spans[0]
        if not all(
            int(target_span["start"]) <= index < int(target_span["end"])
            for index in changed
        ):
            unavailable.append(
                {
                    "split": split,
                    "seed": seed,
                    "full_count": full_count,
                    "reason": "counterfactual_change_outside_added_record",
                }
            )
            continue
        full_id = request_id(full)
        counterfactual_id = request_id(counterfactual)
        pairs.append(
            {
                "schema_version": SCHEMA,
                "pair_id": (
                    f"{model_label}__{split}__seed{seed}__"
                    f"N{full_count - 1}_to_N{full_count}"
                ),
                "model_label": model_label,
                "split": split,
                "seed": seed,
                "full_count": full_count,
                "counterfactual_count": full_count - 1,
                "occurrence": full_count,
                "target_city": target_city,
                "full_request_id": full_id,
                "counterfactual_request_id": counterfactual_id,
                "full_generation_source": full_source,
                "counterfactual_generation_source": counterfactual_source,
                "prompt_token_count": len(full_ids),
                "prompt_changed_token_count": len(changed),
                "prompt_changed_token_indices": changed,
                "target_prompt_record_token_start": int(target_span["start"]),
                "target_prompt_record_token_end": int(target_span["end"]),
                "full_exact_count": True,
                "counterfactual_exact_count": True,
                "full_trace_one_to_one": True,
                "counterfactual_trace_one_to_one": True,
                "pair_eligibility": (
                    "same_seed_adjacent_count_equal_length_nested_prompts_and_"
                    "both_baseline_correct_strict_one_to_one"
                ),
                "construction_reference": "nonthinking_adjacent_count_pairing",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / f"{model_label}__marker_adjacent_pairs.jsonl"
    exclusion_path = output_dir / f"{model_label}__marker_adjacent_exclusions.jsonl"
    write_jsonl(pair_path, pairs)
    write_jsonl(exclusion_path, [*exclusions, *unavailable])
    if not pairs:
        raise ValueError("No correct equal-length adjacent-count marker pairs")
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "model_label": model_label,
        "inputs": inputs,
        "construction_reference": "nonthinking_adjacent_count_pairing",
        "pair_policy": (
            "same model/split/seed adjacent N=k and N=k-1 prompts; first k-1 "
            "needles nested; equal prompt token length; both actual baseline "
            "answers correct and traces strict one-to-one"
        ),
        "counterfactual_locality": (
            "every changed prompt token lies inside the added kth record span"
        ),
        "trace_policy": (
            "teacher-force the identical N=k trace prefix into both prompts up "
            "to the pre-city query for the added kth needle"
        ),
        "behavioral_endpoint": "actual_greedy_added_kth_needle_exact_sequence",
        "pairs": len(pairs),
        "pair_seed_clusters": len({(row["split"], row["seed"]) for row in pairs}),
        "excluded_rows": len(exclusions),
        "unavailable_adjacent_pairs": len(unavailable),
        "splits": {
            split: sum(row["split"] == split for row in pairs)
            for split in sorted({row["split"] for row in pairs})
        },
        "outputs": {
            "pairs": str(pair_path.resolve()),
            "pairs_sha256": sha256(pair_path),
            "exclusions": str(exclusion_path.resolve()),
            "exclusions_sha256": sha256(exclusion_path),
        },
    }
    (output_dir / f"{model_label}__marker_adjacent_plan_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.generations, args.model, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
