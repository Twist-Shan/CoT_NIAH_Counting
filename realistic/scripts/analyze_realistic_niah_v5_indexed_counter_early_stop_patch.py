#!/usr/bin/env python3
"""Analyze the frozen old-HTML explicit-progress full-item patch panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import bootstrap_seed_mean_ci, sign_flip_pvalue  # noqa: E402


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read(root: Path) -> pd.DataFrame:
    files = sorted((root / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No indexed patch shards under {root}")
    rows = [
        json.loads(line)
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    if set(frame["experiment_id"].astype(str)) != {
        "indexed_old_html_counter_early_stop_patch"
    }:
        raise ValueError("Indexed patch experiment id changed")
    if frame["prompt_modified"].astype(bool).any():
        raise ValueError("Indexed patch contains a modified prompt")
    if not frame["source_prompt_is_frozen_original"].astype(bool).all():
        raise ValueError("Indexed patch lost frozen-prompt provenance")
    if frame["selection_rank_used"].astype(bool).any() or not frame[
        "outcome_blind"
    ].astype(bool).all():
        raise ValueError("Indexed patch outcome-blind contract changed")
    if frame["future_trace_tokens_present"].astype(bool).any():
        raise ValueError("Indexed early-stop rows still contain future trace tokens")
    if not frame["natural_recap_removed"].astype(bool).all():
        raise ValueError("Indexed early-stop rows retained a natural recap")
    if frame["minimal_terminal_suffix_contains_candidate_digit"].astype(bool).any():
        raise ValueError("Indexed early-stop suffix leaks a candidate digit")
    if set(frame["patch_layer_mode"].astype(str)) != {
        "single_decoder_block_input"
    }:
        raise ValueError("Indexed patch is not the registered single-layer intervention")
    if not frame["upper_layers_recomputed_after_patch"].astype(bool).all():
        raise ValueError("Indexed patch did not freely recompute upper layers")
    patched = frame.loc[frame["source_layer"].astype(int).ge(0)]
    if not patched["patch_layer_count"].astype(int).eq(1).all():
        raise ValueError("An indexed treatment arm patches more than one layer")
    if set(patched["patch_site"].astype(str)) != {"decoder_block_input"}:
        raise ValueError("Indexed treatment arms do not patch decoder-block inputs")
    return frame


def _scores(value: Any) -> np.ndarray:
    result = np.asarray([float(piece) for piece in str(value).split(",")], dtype=float)
    if result.shape != (10,) or not np.isfinite(result).all():
        raise ValueError("Indexed patch candidate score vector changed")
    return result


def _target_metrics(scores: np.ndarray, target: int) -> dict[str, float]:
    index = int(target) - 1
    others = np.delete(scores, index)
    prediction = int(np.argmax(scores)) + 1
    return {
        "prediction": prediction,
        "exact": float(prediction == int(target)),
        "absolute_error": float(abs(prediction - int(target))),
        "target_margin": float(scores[index] - float(others.max())),
    }


def _summary(values: np.ndarray, name: str, seed: int) -> dict[str, Any]:
    result = bootstrap_seed_mean_ci(values.astype(float), samples=10_000, seed=seed)
    result.update(
        {
            "estimand": name,
            "p_value": sign_flip_pvalue(values.astype(float)),
            "higher_is_supportive": True,
        }
    )
    return result


def analyze(
    frame: pd.DataFrame,
    *,
    phase: str,
    frozen_layer: int | None,
    expected_seed_order: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected_count = 20 if phase == "discovery" else 10
    if len(expected_seed_order) != expected_count or len(set(expected_seed_order)) != expected_count:
        raise ValueError(f"{phase} indexed patch needs {expected_count} unique seeds")
    expected_seeds = set(expected_seed_order)
    observed_seeds = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
    if observed_seeds != expected_seeds:
        raise ValueError("Indexed patch seed contract changed")
    conditions = set(frame["condition"].astype(str))
    expected_conditions = {
        "clean_early_stop_reference",
        "corrupt_early_stop_reference",
        "clean_item_restore_into_corrupt",
        "corrupt_item_ablate_into_clean",
    }
    if conditions != expected_conditions:
        raise ValueError(f"Indexed patch conditions changed: {conditions}")

    reference = frame.loc[frame["source_layer"].astype(int).eq(-1)].copy()
    patched = frame.loc[frame["source_layer"].astype(int).ge(0)].copy()
    observed_layers = sorted(set(patched["source_layer"].astype(int)))
    if phase == "confirmation":
        if frozen_layer is None or observed_layers != [int(frozen_layer)]:
            raise ValueError("Indexed confirmation did not use the frozen layer")

    derived_rows: list[dict[str, Any]] = []
    for (seed, target), active in frame.groupby(["seed", "target_occurrence"], sort=True):
        seed = int(seed)
        target = int(target)
        clean_rows = active.loc[active["condition"].eq("clean_early_stop_reference")]
        corrupt_rows = active.loc[active["condition"].eq("corrupt_early_stop_reference")]
        if len(clean_rows) != 1 or len(corrupt_rows) != 1:
            raise ValueError("Every seed/occurrence needs one clean and corrupt reference")
        clean = _target_metrics(_scores(clean_rows.iloc[0]["candidate_log_scores"]), target)
        corrupt = _target_metrics(_scores(corrupt_rows.iloc[0]["candidate_log_scores"]), target)
        for layer in observed_layers:
            layer_rows = active.loc[active["source_layer"].astype(int).eq(layer)]
            restore_rows = layer_rows.loc[
                layer_rows["condition"].eq("clean_item_restore_into_corrupt")
            ]
            ablate_rows = layer_rows.loc[
                layer_rows["condition"].eq("corrupt_item_ablate_into_clean")
            ]
            if len(restore_rows) != 1 or len(ablate_rows) != 1:
                raise ValueError("Every seed/occurrence/layer needs one restore and ablate row")
            restored = _target_metrics(
                _scores(restore_rows.iloc[0]["candidate_log_scores"]), target
            )
            ablated = _target_metrics(
                _scores(ablate_rows.iloc[0]["candidate_log_scores"]), target
            )
            derived_rows.append(
                {
                    "seed": seed,
                    "source_layer": int(layer),
                    "target_occurrence": target,
                    "marker_kind": str(active.iloc[0]["marker_kind"]),
                    "clean_prediction": int(clean["prediction"]),
                    "corrupt_prediction": int(corrupt["prediction"]),
                    "restored_prediction": int(restored["prediction"]),
                    "ablated_prediction": int(ablated["prediction"]),
                    "clean_exact": clean["exact"],
                    "corrupt_exact": corrupt["exact"],
                    "restored_exact": restored["exact"],
                    "ablated_exact": ablated["exact"],
                    "restoration_exact_gain": restored["exact"] - corrupt["exact"],
                    "restoration_mae_reduction": (
                        corrupt["absolute_error"] - restored["absolute_error"]
                    ),
                    "restoration_target_margin_gain": (
                        restored["target_margin"] - corrupt["target_margin"]
                    ),
                    "ablation_exact_damage": clean["exact"] - ablated["exact"],
                    "ablation_mae_increase": (
                        ablated["absolute_error"] - clean["absolute_error"]
                    ),
                    "ablation_target_margin_damage": (
                        clean["target_margin"] - ablated["target_margin"]
                    ),
                }
            )
    derived = pd.DataFrame(derived_rows)

    layer_rows: list[dict[str, Any]] = []
    for layer, active in derived.groupby("source_layer", sort=True):
        seed_means = active.groupby("seed", as_index=False).agg(
            corrupt_exact=("corrupt_exact", "mean"),
            restored_exact=("restored_exact", "mean"),
            restoration_exact_gain=("restoration_exact_gain", "mean"),
            restoration_mae_reduction=("restoration_mae_reduction", "mean"),
            restoration_target_margin_gain=("restoration_target_margin_gain", "mean"),
            clean_exact=("clean_exact", "mean"),
            ablated_exact=("ablated_exact", "mean"),
            ablation_exact_damage=("ablation_exact_damage", "mean"),
            ablation_mae_increase=("ablation_mae_increase", "mean"),
            ablation_target_margin_damage=("ablation_target_margin_damage", "mean"),
        )
        occurrence = active.groupby("target_occurrence", as_index=False).agg(
            restoration_target_margin_gain=("restoration_target_margin_gain", "mean"),
            ablation_target_margin_damage=("ablation_target_margin_damage", "mean"),
        )
        layer_rows.append(
            {
                "source_layer": int(layer),
                "seed_count": int(seed_means["seed"].nunique()),
                "available_seed_occurrence_count": int(len(active)),
                "available_occurrence_count": int(active["target_occurrence"].nunique()),
                "corrupt_exact_accuracy": float(seed_means["corrupt_exact"].mean()),
                "restored_exact_accuracy": float(seed_means["restored_exact"].mean()),
                "restoration_exact_gain": float(seed_means["restoration_exact_gain"].mean()),
                "mean_restoration_mae_reduction": float(
                    seed_means["restoration_mae_reduction"].mean()
                ),
                "mean_restoration_target_margin_gain": float(
                    seed_means["restoration_target_margin_gain"].mean()
                ),
                "positive_restoration_occurrence_count": int(
                    (occurrence["restoration_target_margin_gain"] > 0.0).sum()
                ),
                "clean_exact_accuracy": float(seed_means["clean_exact"].mean()),
                "ablated_exact_accuracy": float(seed_means["ablated_exact"].mean()),
                "ablation_exact_damage": float(seed_means["ablation_exact_damage"].mean()),
                "mean_ablation_mae_increase": float(
                    seed_means["ablation_mae_increase"].mean()
                ),
                "mean_ablation_target_margin_damage": float(
                    seed_means["ablation_target_margin_damage"].mean()
                ),
                "positive_ablation_occurrence_count": int(
                    (occurrence["ablation_target_margin_damage"] > 0.0).sum()
                ),
            }
        )
    layers = pd.DataFrame(layer_rows).sort_values("source_layer").reset_index(drop=True)
    if phase == "discovery":
        chosen = max(
            layer_rows,
            key=lambda row: (
                float(row["restored_exact_accuracy"]),
                float(row["mean_restoration_target_margin_gain"]),
                int(row["source_layer"]),
            ),
        )
        selected_layer = int(chosen["source_layer"])
    else:
        selected_layer = int(frozen_layer)
        chosen = next(
            row for row in layer_rows if int(row["source_layer"]) == selected_layer
        )

    selected = derived.loc[derived["source_layer"].eq(selected_layer)].copy()
    seed_effects = selected.groupby("seed", as_index=False).agg(
        restoration_exact_gain=("restoration_exact_gain", "mean"),
        restoration_mae_reduction=("restoration_mae_reduction", "mean"),
        restoration_target_margin_gain=("restoration_target_margin_gain", "mean"),
        ablation_exact_damage=("ablation_exact_damage", "mean"),
        ablation_mae_increase=("ablation_mae_increase", "mean"),
        ablation_target_margin_damage=("ablation_target_margin_damage", "mean"),
    )
    primary_estimands = [
        _summary(seed_effects[column].to_numpy(float), column, 20260823 + index)
        for index, column in enumerate(
            (
                "restoration_exact_gain",
                "restoration_mae_reduction",
                "restoration_target_margin_gain",
                "ablation_exact_damage",
                "ablation_mae_increase",
                "ablation_target_margin_damage",
            )
        )
    ]
    occurrence = selected.groupby("target_occurrence", as_index=False).agg(
        seed_count=("seed", "nunique"),
        corrupt_exact_accuracy=("corrupt_exact", "mean"),
        restored_exact_accuracy=("restored_exact", "mean"),
        restoration_exact_gain=("restoration_exact_gain", "mean"),
        mean_restoration_target_margin_gain=("restoration_target_margin_gain", "mean"),
        clean_exact_accuracy=("clean_exact", "mean"),
        ablated_exact_accuracy=("ablated_exact", "mean"),
        ablation_exact_damage=("ablation_exact_damage", "mean"),
        mean_ablation_target_margin_damage=("ablation_target_margin_damage", "mean"),
    )
    restoration_pass = bool(
        float(chosen["restored_exact_accuracy"]) >= 0.50
        and float(chosen["restoration_exact_gain"]) >= 0.25
        and float(chosen["mean_restoration_target_margin_gain"]) > 0.0
        and int(chosen["positive_restoration_occurrence_count"]) >= 6
    )
    ablation_support = bool(
        float(chosen["mean_ablation_target_margin_damage"]) > 0.0
        and int(chosen["positive_ablation_occurrence_count"]) >= 6
    )
    marker_counts = (
        reference[["seed", "marker_kind"]]
        .drop_duplicates()["marker_kind"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    result = {
        "schema_version": "realistic_niah_v5_indexed_counter_patch_analysis_v2",
        "status": "PASS",
        "experiment_id": "indexed_old_html_counter_early_stop_patch",
        "phase": phase,
        "seed_count": len(expected_seed_order),
        "seeds": list(expected_seed_order),
        "source_gold_count": 10,
        "patch_layer_mode": "single_decoder_block_input",
        "upper_layers_recomputed_after_patch": True,
        "selected_layer": selected_layer,
        "selection_rule": (
            "discovery maximize seed-equal restored exact accuracy, then restoration "
            "target-margin gain, then prefer the later start layer; confirmation frozen"
        ),
        "selected_layer_metrics": chosen,
        "primary_estimands": primary_estimands,
        "marker_kind_counts": {str(key): int(value) for key, value in marker_counts.items()},
        "magnitude_gate": {
            "restored_exact_accuracy_at_least_0_50": (
                float(chosen["restored_exact_accuracy"]) >= 0.50
            ),
            "restoration_exact_gain_at_least_0_25": (
                float(chosen["restoration_exact_gain"]) >= 0.25
            ),
            "mean_restoration_target_margin_gain_positive": (
                float(chosen["mean_restoration_target_margin_gain"]) > 0.0
            ),
            "at_least_6_occurrences_have_positive_restoration_margin": (
                int(chosen["positive_restoration_occurrence_count"]) >= 6
            ),
        },
        "old_html_explicit_progress_state_restoration_pass": restoration_pass,
        "ablation_secondary_support": ablation_support,
        "prompt_modified": False,
        "future_trace_tokens_present": False,
        "natural_recap_removed": True,
        "visible_progress_confound_allowed": True,
        "internal_counter_without_visible_index_claim_allowed": False,
        "controlled_running_state_sufficiency_claim_allowed": True,
        "outcome_blind": True,
        "selection_rank_used": False,
        "allowed_claim_if_confirmation_passes": (
            "In the frozen original-prompt N=10 traces, the full hidden-state span at "
            "parsed item k is sufficient to restore an immediate preference for k in "
            "an otherwise uninformative prompt/visible-prefix receiver. This confirms "
            "an item-local running-state representation, but explicit index/count text "
            "inside the span prevents a no-visible-index internal-counter claim."
        ),
        "restriction": (
            "This is the former-HTML-style positive control. Numbered and inline-count "
            "markers are deliberately retained, so the patched state may encode the "
            "visible marker rather than an autonomous latent counter. Partial parser "
            "episodes contribute only their registered contiguous 1..M spans; every "
            "occurrence table reports its actual seed count."
        ),
    }
    return layers, derived, occurrence, seed_effects, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--frozen-layer", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers, derived, occurrence, seed_effects, result = analyze(
        _read(args.input),
        phase=args.phase,
        frozen_layer=args.frozen_layer,
        expected_seed_order=tuple(int(value) for value in args.expected_seeds),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    layers.to_csv(args.output / "layer_metrics.csv", index=False)
    derived.to_csv(args.output / "patch_rows_derived.csv", index=False)
    occurrence.to_csv(args.output / "occurrence_metrics.csv", index=False)
    seed_effects.to_csv(args.output / "seed_effects.csv", index=False)
    _atomic_json(args.output / "claim_gates.json", result)
    _atomic_json(
        args.output / "audit.json",
        {
            "status": "PASS",
            "phase": args.phase,
            "seed_count": result["seed_count"],
            "prompt_modified": False,
            "future_trace_tokens_present": False,
            "natural_recap_removed": True,
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )
    print(
        json.dumps(
            {
                "selected_layer": result["selected_layer"],
                "restoration_pass": result[
                    "old_html_explicit_progress_state_restoration_pass"
                ],
                "ablation_support": result["ablation_secondary_support"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
