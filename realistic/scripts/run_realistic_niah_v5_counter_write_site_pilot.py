#!/usr/bin/env python3
"""Run the exploratory native counter write-site ladder on frozen traces."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.counter_write_site import (  # noqa: E402
    REGISTERED_COUNTER_WRITE_SITES,
    run_counter_write_site_trials,
    run_counter_write_site_uninformative_restore_trials,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("row"), dict):
            value = value["row"]
        if not isinstance(value, dict):
            raise ValueError(f"Frozen trace row must be one JSON object: {path}")
        rows.append(value)
    if not rows:
        raise ValueError("At least one frozen trace row is required")
    return rows


def _directional_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[
            (
                str(row["site_kind"]),
                int(row["source_layer"]),
                int(row["patch_span_width"]),
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for (site, layer, width), cells in sorted(groups.items()):
        log_odds = [float(row["donor_vs_receiver_log_odds_effect"]) for row in cells]
        aligned = [float(row["donor_aligned_expected_count_shift"]) for row in cells]
        consistency = sum(
            bool(row["moves_expected_count_toward_donor"]) for row in cells
        ) / len(cells)
        token_match = statistics.fmean(
            float(row["donor_receiver_token_match_fraction"]) for row in cells
        )
        both_directions = {int(row["donor_direction"]) for row in cells} == {-1, 1}
        mean_log_odds = statistics.fmean(log_odds)
        mean_aligned = statistics.fmean(aligned)
        summary.append(
            {
                "site_kind": site,
                "source_layer": layer,
                "patch_span_width": width,
                "n": len(cells),
                "mean_donor_log_odds_effect": mean_log_odds,
                "median_donor_log_odds_effect": statistics.median(log_odds),
                "mean_donor_aligned_expected_shift": mean_aligned,
                "direction_consistency": consistency,
                "mean_surface_token_match_fraction": token_match,
                "both_direction_cells_present": both_directions,
                "exploratory_bidirectional_candidate": bool(
                    mean_log_odds > 0
                    and mean_aligned > 0
                    and consistency >= 0.75
                    and both_directions
                ),
            }
        )
    summary.sort(
        key=lambda row: (
            bool(row["exploratory_bidirectional_candidate"]),
            float(row["mean_donor_log_odds_effect"]),
            float(row["mean_donor_aligned_expected_shift"]),
        ),
        reverse=True,
    )
    return summary


def _restore_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[
            (
                str(row["site_kind"]),
                int(row["source_layer"]),
                int(row["patch_span_width"]),
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for (site, layer, width), cells in sorted(groups.items()):
        margin = [float(row["target_margin_effect"]) for row in cells]
        probability = [float(row["target_probability_effect"]) for row in cells]
        distance = [float(row["expected_distance_improvement"]) for row in cells]
        summary.append(
            {
                "site_kind": site,
                "source_layer": layer,
                "patch_span_width": width,
                "n": len(cells),
                "mean_target_margin_effect": statistics.fmean(margin),
                "median_target_margin_effect": statistics.median(margin),
                "target_margin_positive_fraction": sum(value > 0 for value in margin)
                / len(margin),
                "mean_target_probability_effect": statistics.fmean(probability),
                "target_probability_positive_fraction": sum(
                    value > 0 for value in probability
                )
                / len(probability),
                "mean_expected_distance_improvement": statistics.fmean(distance),
                "expected_distance_positive_fraction": sum(
                    value > 0 for value in distance
                )
                / len(distance),
                "exploratory_restore_candidate": bool(
                    statistics.fmean(margin) > 0
                    and statistics.fmean(probability) > 0
                    and sum(value > 0 for value in margin) / len(margin) >= 0.75
                    and sum(value > 0 for value in probability) / len(probability)
                    >= 0.75
                ),
            }
        )
    summary.sort(
        key=lambda row: (
            bool(row["exploratory_restore_candidate"]),
            float(row["mean_target_margin_effect"]),
            float(row["mean_target_probability_effect"]),
        ),
        reverse=True,
    )
    return summary


def _summary(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "donor_offset":
        return _directional_summary(records)
    if mode == "uninformative_restore":
        return _restore_summary(records)
    raise ValueError(f"Unknown pilot mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode",
        choices=("donor_offset", "uninformative_restore"),
        default="donor_offset",
    )
    parser.add_argument("--rows", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--source-layers", nargs="+", type=int, required=True)
    parser.add_argument(
        "--receiver-occurrences", nargs="+", type=int, default=[4, 6]
    )
    parser.add_argument("--donor-offsets", nargs="+", type=int, default=[-1, 1])
    parser.add_argument("--span-widths", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--random-seed", type=int, default=20260823)
    parser.add_argument(
        "--site-kinds",
        nargs="+",
        default=list(REGISTERED_COUNTER_WRITE_SITES),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    frozen_rows = _read_rows(args.rows)
    observed_models = {str(row.get("model_label")) for row in frozen_rows}
    if observed_models != {str(args.model)}:
        raise ValueError(
            f"Frozen row model labels {sorted(observed_models)} do not match {args.model}"
        )
    model_spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        model_spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )

    shard_paths: list[Path] = []
    for row in frozen_rows:
        seed = int(row["seed"])
        count = int(row["gold_count"])
        shard = args.output / "shards" / f"seed{seed}_N{count}.jsonl"
        shard_paths.append(shard)
        if args.resume and shard.exists():
            print(f"resume: {shard}", flush=True)
            continue
        if args.mode == "donor_offset":
            trials = run_counter_write_site_trials(
                model,
                tokenizer,
                adapter,
                row,
                source_layers=args.source_layers,
                receiver_occurrences=args.receiver_occurrences,
                donor_offsets=args.donor_offsets,
                span_widths=args.span_widths,
                site_kinds=args.site_kinds,
            )
        else:
            trials = run_counter_write_site_uninformative_restore_trials(
                model,
                tokenizer,
                adapter,
                row,
                source_layers=args.source_layers,
                target_occurrences=args.receiver_occurrences,
                span_widths=args.span_widths,
                site_kinds=args.site_kinds,
                random_seed=int(args.random_seed),
            )
        _atomic_jsonl(shard, trials)
        print(f"sealed {shard}: {len(trials)} trials", flush=True)

    records: list[dict[str, Any]] = []
    for path in shard_paths:
        if not path.exists():
            raise RuntimeError(f"Expected sealed shard is missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    summary_rows = _summary(records, args.mode)
    _atomic_json(args.output / "site_ladder_summary.json", summary_rows)
    _atomic_json(
        args.output / "pilot_manifest.json",
        {
            "status": "PASS",
            "scientific_status": "exploratory_pilot_not_confirmation",
            "model": args.model,
            "mode": args.mode,
            "seeds": sorted({int(row["seed"]) for row in frozen_rows}),
            "counts": sorted({int(row["gold_count"]) for row in frozen_rows}),
            "row_count": len(frozen_rows),
            "trial_count": len(records),
            "source_layers": list(args.source_layers),
            "receiver_occurrences": list(args.receiver_occurrences),
            "donor_offsets": list(args.donor_offsets),
            "random_seed": int(args.random_seed),
            "span_widths": list(args.span_widths),
            "site_kinds": list(args.site_kinds),
            "prompt_modified": False,
            "minimal_terminal_bridge_used": True,
            "outcome_blind": True,
            "selection_rank_used": False,
            "summary": summary_rows,
        },
    )
    print(f"PASS: {len(records)} trials", flush=True)


if __name__ == "__main__":
    main()
