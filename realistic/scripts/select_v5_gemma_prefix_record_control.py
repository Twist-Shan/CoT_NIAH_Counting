#!/usr/bin/env python3
"""Freeze a 20/10 split for the controlled Gemma prefix-record cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCHEMA = "realistic_niah_v5_gemma_prefix_record_control_selection_v1"
SPLIT_SALT = "gemma_prefix_record_control_20_10_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=Path)
    parser.add_argument("--selected-generations-output", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.accepted.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 30:
        raise ValueError(f"Controlled cohort requires exactly 30 accepted rows, got {len(rows)}")
    if len({int(row["seed"]) for row in rows}) != 30:
        raise ValueError("Controlled cohort has duplicate seeds")
    for row in rows:
        row["split_hash"] = hashlib.sha256(
            f"{SPLIT_SALT}:{int(row['seed'])}".encode()
        ).hexdigest()
    ranked = sorted(rows, key=lambda row: (row["split_hash"], int(row["seed"])))
    confirmation = {int(row["seed"]) for row in ranked[:10]}
    for row in rows:
        row["split"] = (
            "confirmation" if int(row["seed"]) in confirmation else "discovery"
        )
        row["split_role"] = f"controlled_{row['split']}"
        row["schema_version"] = SCHEMA
    rows.sort(key=lambda row: (row["split"], int(row["seed"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "schema_version",
        "model_label",
        "request_id",
        "seed",
        "split",
        "split_role",
        "source_split",
        "split_hash",
        "gold_count",
        "grammar_class",
        "marker_kind",
        "surface_family",
        "endpoint_family",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    if (args.generations is None) != (args.selected_generations_output is None):
        raise ValueError(
            "--generations and --selected-generations-output must be supplied together"
        )
    selected_generation_output = None
    if args.generations is not None:
        split_by_id = {
            str(row["request_id"]): str(row["split"])
            for row in rows
        }
        with args.generations.open("r", encoding="utf-8") as handle:
            generation_rows = [json.loads(line) for line in handle if line.strip()]
        selected_generations = []
        for row in generation_rows:
            request_id = str(row.get("request_id"))
            if request_id not in split_by_id:
                continue
            value = dict(row)
            value["source_split"] = str(value.get("split"))
            value["split"] = split_by_id[request_id]
            selected_generations.append(value)
        if len(selected_generations) != 30:
            raise ValueError(
                "Selected generation materialization mismatch: "
                f"{len(selected_generations)} != 30"
            )
        selected_generations.sort(
            key=lambda row: (str(row["split"]), int(row["seed"]))
        )
        args.selected_generations_output.parent.mkdir(parents=True, exist_ok=True)
        args.selected_generations_output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in selected_generations
            ),
            encoding="utf-8",
        )
        selected_generation_output = str(args.selected_generations_output.resolve())
    summary = {
        "schema_version": SCHEMA,
        "split_salt": SPLIT_SALT,
        "selection_independent_of_hidden_states": True,
        "controlled_surface_grammar": (
            "exact nested bullet '*   Record k: (city, score)'; "
            "every item ends in bare )"
        ),
        "endpoint": "bare closing parenthesis after city and score",
        "split_counts": {"discovery": 20, "confirmation": 10},
        "discovery_seeds": [
            int(row["seed"]) for row in rows if row["split"] == "discovery"
        ],
        "confirmation_seeds": [
            int(row["seed"]) for row in rows if row["split"] == "confirmation"
        ],
        "selected_generations_output": selected_generation_output,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
