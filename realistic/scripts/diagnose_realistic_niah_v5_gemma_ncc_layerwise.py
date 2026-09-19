#!/usr/bin/env python3
"""Audit NCC effects by layer and grammar timing without rerunning the model.

This is a post-hoc diagnostic over the frozen discovery and confirmation NPZ
shards.  It deliberately does not replace the preregistered pooled NCC result.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import analyze_realistic_niah_v5_targeted_counter_ncc as ncc


def _selected_bank(path: Path) -> tuple[dict[str, str], tuple[tuple[int, int], ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row.get("condition") == "selected_bank"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one selected_bank row")
    heads = tuple((int(layer), int(head)) for layer, head in json.loads(selected[0]["heads"]))
    if not heads:
        raise ValueError("Selected bank is empty")
    return selected[0], heads


def _phase_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = [row["metadata"] for row in rows]
    return {
        "seed_count": len(rows),
        "seeds": sorted(int(row["seed"]) for row in metadata),
        "gold_count_histogram": {
            str(key): int(value)
            for key, value in sorted(
                Counter(int(row["gold_count"]) for row in metadata).items()
            )
        },
        "final_timing_histogram": {
            str(key): int(value)
            for key, value in sorted(
                Counter(str(row["final_grammar_timing_stratum"]) for row in metadata).items()
            )
        },
        "experiment_ids": sorted({str(row["experiment_id"]) for row in metadata}),
        "teacher_forced_trace_tokens": sorted(
            {bool(row["teacher_forced_trace_tokens"]) for row in metadata}
        ),
        "carrier_pooling": sorted({str(row["carrier_pooling"]) for row in metadata}),
        "condition_orders": sorted(
            {tuple(str(value) for value in row["condition_names"]) for row in rows}
        ),
        "layer_registries": sorted(
            {tuple(int(value) for value in row["layers"]) for row in rows}
        ),
    }


def _confirmation_effects(
    discovery: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    *,
    layer_index: int,
    layer: int,
    timing: str,
) -> dict[str, Any]:
    x, y, groups = ncc._basis_rows(discovery, layer_index, timing)
    model = ncc._fit_ncc(x, y)
    selected_losses: list[float] = []
    specificities: list[float] = []
    clean_margins: list[float] = []
    clean_exact: list[float] = []
    selected_exact: list[float] = []
    projected_shifts: list[float] = []

    for row in confirmation:
        metadata = row["metadata"]
        if str(metadata["final_grammar_timing_stratum"]) != timing:
            continue
        gold = int(metadata["gold_count"])
        classes = model["classes"]
        gold_indices = np.where(classes == gold)[0]
        if len(gold_indices) != 1:
            raise ValueError(f"Gold count {gold} lacks a unique centroid")
        gold_index = int(gold_indices[0])
        vectors = row["final_vectors"][:, layer_index, :]
        distances = ncc._distances(model, vectors)
        margins: list[float] = []
        predictions: list[int] = []
        for active in distances:
            predictions.append(int(classes[int(np.argmin(active))]))
            margins.append(
                float(np.min(np.delete(active, gold_index)) - active[gold_index])
            )
        selected_loss = margins[0] - margins[1]
        random_loss = margins[0] - float(np.mean(margins[2:]))
        selected_losses.append(selected_loss)
        specificities.append(selected_loss - random_loss)
        clean_margins.append(margins[0])
        clean_exact.append(float(predictions[0] == gold))
        selected_exact.append(float(predictions[1] == gold))
        projected = (
            (vectors[:2] - model["mean"]) / model["scale"]
        ) @ model["components"].T
        projected_shifts.append(float(np.linalg.norm(projected[0] - projected[1])))

    losses = np.asarray(selected_losses, dtype=float)
    specificity = np.asarray(specificities, dtype=float)
    return {
        "layer": int(layer),
        "timing": timing,
        "discovery_grouped_oof_balanced_accuracy": ncc._grouped_oof_ba(x, y, groups),
        "confirmation_n": int(len(losses)),
        "clean_exact_accuracy": float(np.mean(clean_exact)),
        "selected_exact_accuracy": float(np.mean(selected_exact)),
        "clean_margin_mean": float(np.mean(clean_margins)),
        "selected_margin_loss_mean": float(np.mean(losses)),
        "selected_margin_loss_median": float(np.median(losses)),
        "selected_margin_loss_positive_n": int(np.sum(losses > 0)),
        "selected_margin_loss_values": losses.tolist(),
        "selected_margin_loss_summary": ncc._summary(
            losses, "selected_correct_centroid_margin_loss", 20260823 + int(layer)
        ),
        "selected_vs_random_specificity_mean": float(np.mean(specificity)),
        "selected_vs_random_specificity_values": specificity.tolist(),
        "selected_vs_random_specificity_summary": ncc._summary(
            specificity,
            "selected_vs_random_margin_loss_specificity",
            202608230 + int(layer),
        ),
        "selected_projected_shift_mean": float(np.mean(projected_shifts)),
    }


def _best_layer(
    rows: list[dict[str, Any]], *, timing: str, eligible: set[int]
) -> dict[str, Any]:
    candidates = [
        row for row in rows if row["timing"] == timing and int(row["layer"]) in eligible
    ]
    if not candidates:
        raise ValueError("No eligible layer for diagnostic selection")
    winner = max(
        candidates,
        key=lambda row: (
            float(row["discovery_grouped_oof_balanced_accuracy"]),
            -int(row["layer"]),
        ),
    )
    return {
        "layer": int(winner["layer"]),
        "discovery_grouped_oof_balanced_accuracy": float(
            winner["discovery_grouped_oof_balanced_accuracy"]
        ),
    }


def build_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    discovery = ncc._load(args.discovery)
    confirmation = ncc._load(args.confirmation)
    layers = tuple(int(value) for value in discovery[0]["layers"])
    if any(tuple(int(value) for value in row["layers"]) != layers for row in discovery + confirmation):
        raise ValueError("Layer registry differs across shards")
    selected_bank, heads = _selected_bank(args.bank_plan)
    head_layers = tuple(sorted({layer for layer, _head in heads}))

    rows = [
        _confirmation_effects(
            discovery,
            confirmation,
            layer_index=layer_index,
            layer=layer,
            timing=timing,
        )
        for layer_index, layer in enumerate(layers)
        for timing in ncc.TIMINGS
    ]
    by_layer = {
        layer: [row for row in rows if int(row["layer"]) == layer]
        for layer in layers
    }
    pooled_oof = {
        layer: float(
            np.mean(
                [
                    row["discovery_grouped_oof_balanced_accuracy"]
                    for row in by_layer[layer]
                ]
            )
        )
        for layer in layers
    }
    current_layer = max(layers, key=lambda layer: (pooled_oof[layer], -layer))
    active_head_count = {
        layer: sum(head_layer < layer for head_layer, _head in heads) for layer in layers
    }
    any_bank_reachable = {layer for layer in layers if active_head_count[layer] > 0}
    partial_bank_reachable = {
        layer for layer in layers if 0 < active_head_count[layer] < len(heads)
    }
    full_bank_reachable = {
        layer for layer in layers if active_head_count[layer] == len(heads)
    }
    rank_before = "rank_before_city"
    frozen = json.loads(args.frozen_analysis.read_text(encoding="utf-8"))
    current_rows = by_layer[current_layer]
    current_n = sum(int(row["confirmation_n"]) for row in current_rows)

    def pooled(key: str) -> float:
        return float(
            sum(float(row[key]) * int(row["confirmation_n"]) for row in current_rows)
            / current_n
        )

    frozen_conditions = {
        str(row["condition"]): row for row in frozen["condition_metrics"]
    }
    reproduction_differences = {
        "selected_margin_loss_mean": abs(
            pooled("selected_margin_loss_mean")
            - float(frozen["primary_estimand"]["mean_effect"])
        ),
        "selected_vs_random_specificity_mean": abs(
            pooled("selected_vs_random_specificity_mean")
            - float(frozen["specificity_estimand"]["mean_effect"])
        ),
        "clean_exact_accuracy": abs(
            pooled("clean_exact_accuracy")
            - float(frozen_conditions["clean"]["exact_accuracy"])
        ),
        "selected_exact_accuracy": abs(
            pooled("selected_exact_accuracy")
            - float(frozen_conditions["selected_mask"]["exact_accuracy"])
        ),
    }

    return {
        "schema_version": "realistic_niah_v5_ncc_layerwise_diagnostic_v1",
        "status": "PASS",
        "inferential_status": "post_hoc_diagnostic_not_confirmatory",
        "model_label": str(args.model_label),
        "discovery_seed_count": len(discovery),
        "confirmation_seed_count": len(confirmation),
        "selected_heads": [list(head) for head in heads],
        "selected_head_layers": list(head_layers),
        "bank_selection_contract": {
            "bank_size": int(selected_bank["bank_size"]),
            "selection_metric": str(selected_bank["selection_metric"]),
            "selection_anchor_role": str(selected_bank["selection_anchor_role"]),
            "selection_target_grammar_class": str(
                selected_bank["selection_target_grammar_class"]
            ),
            "random_control_matching": str(selected_bank["random_control_matching"]),
        },
        "phase_audit": {
            "discovery": _phase_audit(discovery),
            "confirmation": _phase_audit(confirmation),
        },
        "frozen_result_reproduction": {
            "frozen_analysis_path": str(args.frozen_analysis),
            "selected_layer_expected": int(frozen["selected_layer"]),
            "selected_layer_recomputed": int(current_layer),
            "selected_layer_matches": int(frozen["selected_layer"]) == int(current_layer),
            "absolute_differences": reproduction_differences,
            "maximum_absolute_difference": max(reproduction_differences.values()),
            "within_tolerance_0_002": max(reproduction_differences.values()) <= 0.002,
        },
        "causal_layer_semantics": {
            "query_head_at_layer_L_first_reaches_later_carrier_at": "post-block L+1",
            "active_selected_head_count_by_capture_layer": {
                str(layer): int(active_head_count[layer]) for layer in layers
            },
            "no_selected_head_reachable_layers": [
                layer for layer in layers if active_head_count[layer] == 0
            ],
            "partial_bank_reachable_layers": sorted(partial_bank_reachable),
            "full_bank_reachable_layers": sorted(full_bank_reachable),
        },
        "selection_audit": {
            "original_pooled_timing_layer": int(current_layer),
            "original_pooled_timing_oof_balanced_accuracy": pooled_oof[current_layer],
            "original_pooled_timing_active_selected_heads": int(
                active_head_count[current_layer]
            ),
            "original_pooled_timing_active_selected_head_fraction": float(
                active_head_count[current_layer] / len(heads)
            ),
            "rank_before_best_all_layers": _best_layer(
                rows, timing=rank_before, eligible=set(layers)
            ),
            "rank_before_best_any_bank_reachable": _best_layer(
                rows, timing=rank_before, eligible=any_bank_reachable
            ),
            "rank_before_best_full_bank_reachable": (
                _best_layer(rows, timing=rank_before, eligible=full_bank_reachable)
                if full_bank_reachable
                else None
            ),
            "note": (
                "Alternative windows are diagnostic. Confirmation outcomes were already "
                "observed, so they cannot replace the frozen primary analysis."
            ),
        },
        "timing_crossover": {
            "rank_after_first_oof_at_least_0_95": min(
                int(row["layer"])
                for row in rows
                if row["timing"] == "rank_after_city"
                and float(row["discovery_grouped_oof_balanced_accuracy"]) >= 0.95
            ),
            "rank_after_first_perfect_oof": min(
                (
                    int(row["layer"])
                    for row in rows
                    if row["timing"] == "rank_after_city"
                    and float(row["discovery_grouped_oof_balanced_accuracy"]) == 1.0
                ),
                default=None,
            ),
        },
        "layer_timing_rows": rows,
        "interpretation": {
            "frozen_primary_result_retained": True,
            "absence_of_count_state_supported": False,
            "layer_timing_effects_are_exploratory": True,
            "full_bank_to_late_ncc_confirmed": False,
            "cross_model_effect_size_comparison_allowed": False,
        },
        "input_paths": {
            "discovery": str(args.discovery),
            "confirmation": str(args.confirmation),
            "bank_plan": str(args.bank_plan),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--bank-plan", type=Path, required=True)
    parser.add_argument("--frozen-analysis", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_diagnostic(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": len(result["layer_timing_rows"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
