#!/usr/bin/env python3
"""Freeze small, design-matched marker and answer patch registries for V5."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "realistic_niah_v5_aligned_patch_registries_v1"
MODELS = ("Qwen3-8B", "Gemma4-E4B")
SPLITS = ("discovery", "confirmation")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def marker_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["split"]), int(row["counterfactual_count"]), int(row["full_count"])


def marker_seed_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    split, low, high = marker_key(row)
    return split, low, high, int(row["seed"])


def freeze_marker(
    paths: Mapping[str, Path],
    transitions: tuple[tuple[int, int], ...],
    per_stratum: int,
    constructability_paths: Mapping[str, Path] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = {model: read_jsonl(path) for model, path in paths.items()}
    allowed_pair_ids: dict[str, set[str]] | None = None
    if constructability_paths is not None:
        allowed_pair_ids = {
            model: {
                str(row["pair_id"])
                for row in read_jsonl(constructability_paths[model])
                if bool(row["all_required_constructible"])
            }
            for model in MODELS
        }
    selected: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    availability: dict[str, dict[str, int]] = {model: {} for model in MODELS}
    exact_seed_matches = 0
    for split in SPLITS:
        for low, high in transitions:
            key = (split, low, high)
            candidates = {
                model: sorted(
                    (
                        row
                        for row in rows[model]
                        if marker_key(row) == key
                        and (
                            allowed_pair_ids is None
                            or str(row["pair_id"]) in allowed_pair_ids[model]
                        )
                    ),
                    key=lambda row: (int(row["seed"]), str(row["pair_id"])),
                )
                for model in MODELS
            }
            label = f"{split}:N{low}_to_N{high}"
            for model in MODELS:
                availability[model][label] = len(candidates[model])
                if len(candidates[model]) < per_stratum:
                    raise ValueError(
                        f"Marker stratum lacks {per_stratum} eligible pairs: "
                        f"{model}/{label} has {len(candidates[model])}"
                    )
            common_seeds = sorted(
                {int(row["seed"]) for row in candidates[MODELS[0]]}
                & {int(row["seed"]) for row in candidates[MODELS[1]]}
            )
            chosen: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
            for seed in common_seeds[:per_stratum]:
                for model in MODELS:
                    chosen[model].append(
                        next(row for row in candidates[model] if int(row["seed"]) == seed)
                    )
                exact_seed_matches += 1
            for model in MODELS:
                used = {str(row["pair_id"]) for row in chosen[model]}
                for row in candidates[model]:
                    if len(chosen[model]) >= per_stratum:
                        break
                    if str(row["pair_id"]) not in used:
                        chosen[model].append(row)
                for row in chosen[model]:
                    frozen = dict(row)
                    frozen.update(
                        {
                            "alignment_schema_version": SCHEMA,
                            "alignment_role": "marker_transition_split_matched",
                            "alignment_stratum": label,
                            "alignment_per_model_stratum_n": per_stratum,
                        }
                    )
                    selected[model].append(frozen)
    for model in MODELS:
        selected[model].sort(key=lambda row: (SPLITS.index(str(row["split"])), int(row["full_count"]), int(row["seed"])))
    audit = {
        "transitions": [f"N{low}_to_N{high}" for low, high in transitions],
        "splits": list(SPLITS),
        "per_model_per_stratum": per_stratum,
        "pairs_per_model": {model: len(selected[model]) for model in MODELS},
        "availability": availability,
        "exact_cross_model_seed_strata": exact_seed_matches,
        "alignment_policy": (
            "same transition strata, split and pair count per model; prefer exact "
            "cross-model seed matches, otherwise use deterministic model-specific "
            "correct strict-one-to-one pairs"
        ),
        "query_constructability_filter": (
            "pre_city_d1/pre_city_d2/pre_city_anchor all required"
            if allowed_pair_ids is not None
            else "not supplied"
        ),
    }
    return selected, audit


def signed_gap(row: Mapping[str, Any]) -> int:
    return int(row["donor_count"]) - int(row["receiver_count"])


def freeze_answer(
    paths: Mapping[str, Path], magnitudes: tuple[int, ...], max_per_signed_gap: int | None
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = {model: read_jsonl(path) for model, path in paths.items()}
    requested = tuple(gap for magnitude in magnitudes for gap in (-magnitude, magnitude))
    selected: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    availability: dict[str, dict[str, int]] = {model: {} for model in MODELS}
    frozen_counts: dict[str, int] = {}
    for gap in requested:
        candidates = {
            model: sorted(
                (row for row in rows[model] if signed_gap(row) == gap),
                key=lambda row: (int(row["seed"]), str(row["pair_id"])),
            )
            for model in MODELS
        }
        for model in MODELS:
            availability[model][f"{gap:+d}"] = len(candidates[model])
        count = min(len(candidates[model]) for model in MODELS)
        if max_per_signed_gap is not None:
            count = min(count, max_per_signed_gap)
        if count < 1:
            raise ValueError(f"No common answer patch coverage for signed gap {gap:+d}")
        frozen_counts[f"{gap:+d}"] = count
        for model in MODELS:
            for row in candidates[model][:count]:
                frozen = dict(row)
                frozen.update(
                    {
                        "alignment_schema_version": SCHEMA,
                        "alignment_role": "answer_signed_gap_count_matched",
                        "signed_count_gap": gap,
                        "absolute_count_gap": abs(gap),
                        "alignment_per_model_signed_gap_n": count,
                    }
                )
                selected[model].append(frozen)
    for model in MODELS:
        selected[model].sort(key=lambda row: (signed_gap(row), int(row["seed"]), str(row["pair_id"])))
    audit = {
        "signed_gaps": [f"{gap:+d}" for gap in requested],
        "pairs_per_signed_gap_per_model": frozen_counts,
        "pairs_per_model": {model: len(selected[model]) for model in MODELS},
        "availability": availability,
        "alignment_policy": (
            "same signed count gaps and same pair count per signed gap; all source "
            "pairs remain correct-only; model-specific seeds are analyzed separately"
        ),
    }
    return selected, audit


def parse_transitions(value: str) -> tuple[tuple[int, int], ...]:
    transitions = tuple(tuple(int(part) for part in item.split(":")) for item in value.split(","))
    if any(len(pair) != 2 or pair[1] != pair[0] + 1 for pair in transitions):
        raise argparse.ArgumentTypeError("transitions must be comma-separated adjacent low:high pairs")
    return transitions


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("positive comma-separated integers required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-marker", type=Path, required=True)
    parser.add_argument("--gemma-marker", type=Path, required=True)
    parser.add_argument("--qwen-marker-query-audit", type=Path)
    parser.add_argument("--gemma-marker-query-audit", type=Path)
    parser.add_argument("--qwen-answer", type=Path, required=True)
    parser.add_argument("--gemma-answer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marker-transitions", type=parse_transitions, default=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))
    parser.add_argument("--marker-per-stratum", type=int, default=1)
    parser.add_argument("--answer-gap-magnitudes", type=parse_ints, default=(1, 2, 3))
    parser.add_argument("--answer-max-per-signed-gap", type=int)
    args = parser.parse_args()

    query_audits = None
    if args.qwen_marker_query_audit or args.gemma_marker_query_audit:
        if not args.qwen_marker_query_audit or not args.gemma_marker_query_audit:
            raise ValueError("Both marker query audits must be supplied together")
        query_audits = {
            MODELS[0]: args.qwen_marker_query_audit,
            MODELS[1]: args.gemma_marker_query_audit,
        }
    marker, marker_audit = freeze_marker(
        {MODELS[0]: args.qwen_marker, MODELS[1]: args.gemma_marker},
        args.marker_transitions,
        args.marker_per_stratum,
        query_audits,
    )
    answer, answer_audit = freeze_answer(
        {MODELS[0]: args.qwen_answer, MODELS[1]: args.gemma_answer},
        args.answer_gap_magnitudes,
        args.answer_max_per_signed_gap,
    )
    outputs: dict[str, Any] = {}
    for mechanism, frames in (("marker", marker), ("answer", answer)):
        outputs[mechanism] = {}
        for model, rows in frames.items():
            path = args.output_dir / mechanism / f"{model}__{mechanism}_aligned_pairs.jsonl"
            atomic_jsonl(path, rows)
            outputs[mechanism][model] = {
                "path": str(path.resolve()), "rows": len(rows), "sha256": sha256(path)
            }
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "marker": marker_audit,
        "answer": answer_audit,
        "outputs": outputs,
    }
    atomic_json(args.output_dir / "alignment_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
