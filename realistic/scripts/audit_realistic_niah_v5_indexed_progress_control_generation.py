#!/usr/bin/env python3
"""Audit indexed progress-control continuations without reclassifying outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _city_position(text: str, city: str) -> int | None:
    match = re.search(
        rf"(?<![A-Za-z0-9]){re.escape(city)}(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    return None if match is None else int(match.start())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation_root", type=Path)
    parser.add_argument("cohort_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "schema_version": "indexed_progress_control_generation_audit_v1",
        "status": "PASS",
        "endpoint": (
            "The first gold city named before the reasoning-close marker equals "
            "the donor-implied successor. Classification is copied from frozen "
            "trial rows; this audit does not relabel completions."
        ),
        "models": {},
    }
    for model in MODELS:
        cohort_rows = _read_jsonl(args.cohort_root / f"{model}.jsonl")
        cohort = {int(row["seed"]): row for row in cohort_rows}
        legacy_trial_root = (
            args.confirmation_root / model / "confirmation10" / "item_span"
        )
        aligned_trial_root = (
            args.confirmation_root
            / model
            / "indexed_progress_control"
            / "confirmation_runs"
            / "confirmation"
            / "item_span"
        )
        trial_root = (
            aligned_trial_root if aligned_trial_root.is_dir() else legacy_trial_root
        )
        trial_paths = sorted(trial_root.glob("*/trials.jsonl"))
        if len(trial_paths) != 6:
            raise ValueError(f"{model}: expected six k-by-direction trial files")

        grouped: dict[tuple[int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for path in trial_paths:
            for row in _read_jsonl(path):
                direction = (
                    "forward_skip"
                    if int(row["receiver_occurrence_j"])
                    < int(row["donor_occurrence_k"])
                    else "backward_rewind"
                )
                key = (int(row["seed"]), int(row["donor_occurrence_k"]), direction)
                condition = str(row["condition"])
                if condition in grouped[key]:
                    raise ValueError(f"{model}: duplicate condition in cell {key}")
                grouped[key][condition] = row
        if len(grouped) != 60:
            raise ValueError(f"{model}: expected 60 confirmation cells")

        records: list[dict[str, Any]] = []
        seed_any_patched: dict[int, bool] = defaultdict(bool)
        seed_any_incremental: dict[int, bool] = defaultdict(bool)
        for (seed, donor_k, direction), conditions in sorted(grouped.items()):
            required = {"receiver_self", "native_donor", "donor_to_receiver"}
            if set(conditions) != required:
                raise ValueError(f"{model}: incomplete conditions in cell {(seed, donor_k, direction)}")
            donor_successor = donor_k + 1
            city = str(cohort[seed]["gold_records"][donor_successor - 1]["city"])
            receiver = conditions["receiver_self"]
            patched = conditions["donor_to_receiver"]
            patched_adoption = (
                patched.get("first_generated_known_city_ordinal") == donor_successor
            )
            receiver_adoption = (
                receiver.get("first_generated_known_city_ordinal") == donor_successor
            )
            seed_any_patched[seed] = seed_any_patched[seed] or patched_adoption
            seed_any_incremental[seed] = seed_any_incremental[seed] or (
                patched_adoption and not receiver_adoption
            )
            patched_text = str(patched.get("completion_text", ""))
            receiver_text = str(receiver.get("completion_text", ""))
            patched_position = _city_position(patched_text, city)
            receiver_position = _city_position(receiver_text, city)
            if patched_adoption and patched_position is None:
                raise ValueError(f"{model} seed {seed}: adopted donor city is absent")
            close_position = patched.get("reasoning_close_char_position")
            if (
                patched_adoption
                and close_position is not None
                and int(patched_position) >= int(close_position)
            ):
                raise ValueError(f"{model} seed {seed}: adopted city occurs after reasoning close")
            records.append(
                {
                    "seed": seed,
                    "donor_occurrence_k": donor_k,
                    "direction": direction,
                    "donor_successor_city": city,
                    "patched_first_known_city_ordinal": patched.get(
                        "first_generated_known_city_ordinal"
                    ),
                    "receiver_first_known_city_ordinal": receiver.get(
                        "first_generated_known_city_ordinal"
                    ),
                    "patched_donor_adoption": patched_adoption,
                    "receiver_donor_adoption": receiver_adoption,
                    "incremental_donor_adoption": (
                        patched_adoption and not receiver_adoption
                    ),
                    "patched_donor_city_char_position": patched_position,
                    "receiver_donor_city_char_position": receiver_position,
                    "patched_donor_city_within_first_80_chars": (
                        patched_adoption
                        and patched_position is not None
                        and patched_position <= 80
                    ),
                    "patched_first_bullet_city_ordinal": patched.get(
                        "first_generated_bullet_city_ordinal"
                    ),
                    "reasoning_close_char_position": close_position,
                    "equal_length_complete_item": bool(
                        patched["equal_length_complete_item"]
                    ),
                    "patched_completion_prefix_240": patched_text[:240],
                    "receiver_completion_prefix_240": receiver_text[:240],
                }
            )

        patched_records = [row for row in records if row["patched_donor_adoption"]]
        receiver_records = [row for row in records if row["receiver_donor_adoption"]]
        incremental_records = [
            row for row in records if row["incremental_donor_adoption"]
        ]
        payload["models"][model] = {
            "cell_count": len(records),
            "seed_count": len({row["seed"] for row in records}),
            "patched_donor_adoption_count": len(patched_records),
            "receiver_donor_adoption_count": len(receiver_records),
            "paired_adoption_gain_count": len(patched_records)
            - len(receiver_records),
            "paired_adoption_gain_rate": (
                len(patched_records) - len(receiver_records)
            )
            / len(records),
            "incremental_cell_count": len(incremental_records),
            "seed_with_any_patched_adoption_count": sum(seed_any_patched.values()),
            "seed_with_any_incremental_adoption_count": sum(
                seed_any_incremental.values()
            ),
            "adoption_within_first_80_chars_count": sum(
                bool(row["patched_donor_city_within_first_80_chars"])
                for row in patched_records
            ),
            "adoption_after_first_80_chars_count": sum(
                not bool(row["patched_donor_city_within_first_80_chars"])
                for row in patched_records
            ),
            "strict_bullet_adoption_count": sum(
                row["patched_first_bullet_city_ordinal"]
                == int(row["donor_occurrence_k"]) + 1
                for row in records
            ),
            "records": records,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                model: {
                    key: value
                    for key, value in result.items()
                    if key != "records"
                }
                for model, result in payload["models"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
