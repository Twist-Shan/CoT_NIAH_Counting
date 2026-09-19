#!/usr/bin/env python3
"""Select discovery layers and analyze marker-scrubbed list restoration."""

from __future__ import annotations

import argparse
import hashlib
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

from realistic_niah_v5.causal import (  # noqa: E402
    bootstrap_seed_mean_ci,
    holm_adjust,
    sign_flip_pvalue,
)


CONDITIONS = {
    "source_reference",
    "blank_reference",
    "source_list_item_k_to_blank_restoration",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read(root: Path, *, phase: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "frozen_trial_plan.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or str(manifest.get("phase")) != str(phase)
        or str(plan.get("phase")) != str(phase)
    ):
        raise ValueError("Trial manifest/plan phase is not sealed")
    files = sorted((root / "shards").glob("*.jsonl"))
    if len(files) != int(plan["seed_count"]):
        raise ValueError("Trial shard count differs from the frozen seed count")
    rows = [
        json.loads(line)
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    if set(frame["experiment_id"].astype(str)) != {
        "marker_scrubbed_list_counterfactual_sufficiency"
    }:
        raise ValueError("Bullet restoration experiment id changed")
    if set(frame["condition"].astype(str)) != CONDITIONS:
        raise ValueError("Bullet restoration conditions changed")
    if frame["selection_uses_final_answer"].astype(bool).any():
        raise ValueError("Format cohort selection accessed final-answer outcomes")
    if not frame["outcome_blind"].astype(bool).all():
        raise ValueError("Outcome-blind flag changed")
    if frame["diagnostic_suffix_used"].astype(bool).any():
        raise ValueError("A forbidden diagnostic suffix was used")
    if frame["future_trace_tokens_present"].astype(bool).any():
        raise ValueError("An early-stop row retains future list items")
    if not frame["natural_recap_removed"].astype(bool).all():
        raise ValueError("A natural recap remains in the readout suffix")
    if not frame["source_base_scrubbed_before_state_capture"].astype(bool).all():
        raise ValueError("Source states were captured before the base scrub")
    if not frame["source_blank_base_scrub_identical"].astype(bool).all():
        raise ValueError("Source and Blank do not share the base scrub")
    if not frame["source_item_nonmarkers_preserved"].astype(bool).all():
        raise ValueError("Source non-marker list-item content was not preserved")
    if not frame["source_explicit_item_markers_scrubbed"].astype(bool).all():
        raise ValueError("Source explicit item markers were not scrubbed")
    if frame["source_items_contain_explicit_running_value"].astype(bool).any():
        raise ValueError("A Source item retains an explicit running value")
    if set(frame["patch_layer_mode"].astype(str)) != {
        "single_decoder_block_input_once"
    }:
        raise ValueError("Restoration is not the registered single-layer patch")
    patched = frame.loc[frame["source_layer"].astype(int).ge(0)]
    if not patched["patch_layer_count"].astype(int).eq(1).all():
        raise ValueError("A restoration row patched more than one layer")
    if not patched["upper_layers_recomputed_after_patch"].astype(bool).all():
        raise ValueError("Upper layers were not freely recomputed")
    expected_seeds = {int(value["seed"]) for value in plan["rows"]}
    observed_seeds = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
    if observed_seeds != expected_seeds:
        raise ValueError("Observed seeds differ from the frozen trial plan")
    expected_seed_count = 20 if phase == "discovery" else 10
    if len(expected_seeds) != expected_seed_count:
        raise ValueError(f"{phase} must contain {expected_seed_count} independent seeds")
    return frame, plan


def _scores(value: Any) -> np.ndarray:
    result = np.asarray([float(piece) for piece in str(value).split(",")], dtype=float)
    if result.shape != (10,) or not np.isfinite(result).all():
        raise ValueError("Candidate score vector must contain ten finite values")
    return result


def _metrics(row: pd.Series, target: int) -> dict[str, float | int]:
    scores = _scores(row["candidate_log_scores"])
    index = int(target) - 1
    prediction = int(np.argmax(scores)) + 1
    margin = float(scores[index] - np.max(np.delete(scores, index)))
    if prediction != int(row["predicted_running_count"]):
        raise ValueError("Stored running-count prediction disagrees with candidate scores")
    if not np.isclose(margin, float(row["running_target_margin"]), atol=2e-6):
        raise ValueError("Stored running target margin disagrees with candidate scores")
    return {
        "prediction": prediction,
        "exact": float(prediction == int(target)),
        "margin": margin,
        "probability": float(row["running_target_probability"]),
    }


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    layers = sorted(
        set(
            frame.loc[frame["source_layer"].astype(int).ge(0), "source_layer"]
            .astype(int)
            .tolist()
        )
    )
    rows: list[dict[str, Any]] = []
    for (seed, target), active in frame.groupby(
        ["seed", "target_occurrence"], sort=True
    ):
        target = int(target)
        source_rows = active.loc[active["condition"].eq("source_reference")]
        blank_rows = active.loc[active["condition"].eq("blank_reference")]
        if len(source_rows) != 1 or len(blank_rows) != 1:
            raise ValueError("Every seed/k needs one Source and one Blank reference")
        source = _metrics(source_rows.iloc[0], target)
        blank = _metrics(blank_rows.iloc[0], target)
        for layer in layers:
            restored_rows = active.loc[
                active["condition"].eq("source_list_item_k_to_blank_restoration")
                & active["source_layer"].astype(int).eq(layer)
            ]
            if len(restored_rows) != 1:
                raise ValueError("Every seed/k/layer needs one restoration row")
            restored = _metrics(restored_rows.iloc[0], target)
            rows.append(
                {
                    "seed": int(seed),
                    "target_occurrence": target,
                    "source_layer": int(layer),
                    "source_prediction": int(source["prediction"]),
                    "blank_prediction": int(blank["prediction"]),
                    "restored_prediction": int(restored["prediction"]),
                    "source_exact": float(source["exact"]),
                    "blank_exact": float(blank["exact"]),
                    "restored_exact": float(restored["exact"]),
                    "restoration_exact_gain": float(
                        restored["exact"] - blank["exact"]
                    ),
                    "source_target_margin": float(source["margin"]),
                    "blank_target_margin": float(blank["margin"]),
                    "restored_target_margin": float(restored["margin"]),
                    "restoration_target_margin_gain": float(
                        restored["margin"] - blank["margin"]
                    ),
                    "source_target_probability": float(source["probability"]),
                    "blank_target_probability": float(blank["probability"]),
                    "restored_target_probability": float(restored["probability"]),
                    "restoration_target_probability_gain": float(
                        restored["probability"] - blank["probability"]
                    ),
                }
            )
    derived = pd.DataFrame(rows)
    expected = (
        derived["seed"].nunique()
        * derived["target_occurrence"].nunique()
        * derived["source_layer"].nunique()
    )
    if len(derived) != expected or derived["target_occurrence"].nunique() != 10:
        raise ValueError("Derived seed/k/layer panel is incomplete")
    return derived


def _layer_metrics(derived: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_layer = derived.groupby(["seed", "source_layer"], as_index=False).agg(
        source_exact=("source_exact", "mean"),
        blank_exact=("blank_exact", "mean"),
        restored_exact=("restored_exact", "mean"),
        restoration_exact_gain=("restoration_exact_gain", "mean"),
        source_target_margin=("source_target_margin", "mean"),
        blank_target_margin=("blank_target_margin", "mean"),
        restored_target_margin=("restored_target_margin", "mean"),
        restoration_target_margin_gain=("restoration_target_margin_gain", "mean"),
        restoration_target_probability_gain=(
            "restoration_target_probability_gain",
            "mean",
        ),
    )
    layer = seed_layer.groupby("source_layer", as_index=False).agg(
        seed_count=("seed", "nunique"),
        source_exact_accuracy=("source_exact", "mean"),
        blank_exact_accuracy=("blank_exact", "mean"),
        restored_exact_accuracy=("restored_exact", "mean"),
        mean_restoration_exact_gain=("restoration_exact_gain", "mean"),
        mean_source_target_margin=("source_target_margin", "mean"),
        mean_blank_target_margin=("blank_target_margin", "mean"),
        mean_restored_target_margin=("restored_target_margin", "mean"),
        mean_restoration_target_margin_gain=(
            "restoration_target_margin_gain",
            "mean",
        ),
        mean_restoration_target_probability_gain=(
            "restoration_target_probability_gain",
            "mean",
        ),
    )
    return seed_layer, layer.sort_values("source_layer").reset_index(drop=True)


def _digit_proportions(derived: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    references = derived.drop_duplicates(["seed", "target_occurrence"])

    def append_distribution(
        active: pd.DataFrame,
        *,
        condition: str,
        source_layer: int,
        prediction_column: str,
        target_occurrence: int,
    ) -> None:
        counts = active[prediction_column].astype(int).value_counts()
        for digit in range(1, 11):
            records.append(
                {
                    "condition": condition,
                    "source_layer": int(source_layer),
                    # Zero is the explicitly labeled all-k pooled descriptive row.
                    "target_occurrence": int(target_occurrence),
                    "predicted_digit": digit,
                    "count": int(counts.get(digit, 0)),
                    "proportion": float(counts.get(digit, 0) / len(active)),
                }
            )

    for condition, column in (
        ("source_reference", "source_prediction"),
        ("blank_reference", "blank_prediction"),
    ):
        append_distribution(
            references,
            condition=condition,
            source_layer=-1,
            prediction_column=column,
            target_occurrence=0,
        )
        for target, active in references.groupby("target_occurrence", sort=True):
            append_distribution(
                active,
                condition=condition,
                source_layer=-1,
                prediction_column=column,
                target_occurrence=int(target),
            )
    for layer, active in derived.groupby("source_layer", sort=True):
        append_distribution(
            active,
            condition="source_list_item_k_to_blank_restoration",
            source_layer=int(layer),
            prediction_column="restored_prediction",
            target_occurrence=0,
        )
        for target, target_rows in active.groupby("target_occurrence", sort=True):
            append_distribution(
                target_rows,
                condition="source_list_item_k_to_blank_restoration",
                source_layer=int(layer),
                prediction_column="restored_prediction",
                target_occurrence=int(target),
            )
    return pd.DataFrame(records)


def _freeze_top_three(
    *, layer_metrics: pd.DataFrame, model_label: str, plan: dict[str, Any]
) -> dict[str, Any]:
    ranked = layer_metrics.sort_values(
        [
            "mean_restoration_target_margin_gain",
            "restored_exact_accuracy",
            "source_layer",
        ],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    if len(ranked) < 3:
        raise ValueError("Discovery exposed fewer than three layers")
    top = ranked.iloc[:3]
    return {
        "schema_version": "realistic_niah_v5_bullet_counter_frozen_layers_v1",
        "status": "FROZEN_FROM_DISCOVERY",
        "model_label": str(model_label),
        "source_layers": [int(value) for value in top["source_layer"]],
        "selection_order": "descending_seed_equal_mean_delta_target_margin",
        "tie_break_1": "descending_restored_exact_accuracy",
        "tie_break_2": "ascending_layer_index",
        "discovery_seed_count": int(plan["seed_count"]),
        "discovery_trial_plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "ranked_layers": [
            {
                "rank": rank,
                "source_layer": int(row.source_layer),
                "mean_restoration_target_margin_gain": float(
                    row.mean_restoration_target_margin_gain
                ),
                "restored_exact_accuracy": float(row.restored_exact_accuracy),
            }
            for rank, row in enumerate(ranked.itertuples(index=False), start=1)
        ],
        "confirmation_results_accessed": False,
    }


def _confirmation_statistics(
    derived: pd.DataFrame, *, frozen_layers: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(frozen_layers) != 3:
        raise ValueError("Confirmation needs exactly three frozen layers")
    observed = tuple(sorted(set(derived["source_layer"].astype(int))))
    if set(observed) != set(frozen_layers):
        raise ValueError("Confirmation layers differ from the discovery-frozen set")
    seed_layer = derived.groupby(["seed", "source_layer"], as_index=False).agg(
        delta_margin=("restoration_target_margin_gain", "mean"),
        source_exact=("source_exact", "mean"),
        blank_exact=("blank_exact", "mean"),
        restored_exact=("restored_exact", "mean"),
        exact_gain=("restoration_exact_gain", "mean"),
    )
    raw_p = []
    layer_rows = []
    for layer in frozen_layers:
        active = seed_layer.loc[seed_layer["source_layer"].astype(int).eq(int(layer))]
        values = active["delta_margin"].to_numpy(float)
        ci = bootstrap_seed_mean_ci(values, samples=10_000, seed=20260824 + int(layer))
        p_value = sign_flip_pvalue(values)
        raw_p.append(p_value)
        layer_rows.append(
            {
                "source_layer": int(layer),
                "seed_count": int(len(values)),
                "mean_delta_margin": float(ci["mean_effect"]),
                "bootstrap_ci_low": float(ci["ci_low"]),
                "bootstrap_ci_high": float(ci["ci_high"]),
                "two_sided_exact_sign_flip_p": float(p_value),
                "source_exact_accuracy": float(active["source_exact"].mean()),
                "blank_exact_accuracy": float(active["blank_exact"].mean()),
                "restored_exact_accuracy": float(active["restored_exact"].mean()),
                "mean_exact_gain": float(active["exact_gain"].mean()),
            }
        )
    adjusted = holm_adjust(raw_p)
    for row, value in zip(layer_rows, adjusted):
        row["holm_p_across_three_frozen_layers"] = float(value)
    per_layer = pd.DataFrame(layer_rows)

    seed_primary = seed_layer.groupby("seed", as_index=False).agg(
        three_layer_mean_delta_margin=("delta_margin", "mean"),
        source_exact_accuracy=("source_exact", "mean"),
        blank_exact_accuracy=("blank_exact", "mean"),
        three_layer_mean_restored_exact_accuracy=("restored_exact", "mean"),
        three_layer_mean_exact_gain=("exact_gain", "mean"),
    )
    values = seed_primary["three_layer_mean_delta_margin"].to_numpy(float)
    ci = bootstrap_seed_mean_ci(values, samples=10_000, seed=20260824)
    p_value = sign_flip_pvalue(values)
    summary = {
        "primary_estimand": (
            "within each confirmation seed, mean over k=1..10 and the three "
            "discovery-frozen layers of restored target margin minus blank target margin"
        ),
        "statistical_unit": "seed",
        "seed_count": int(len(values)),
        "mean_delta_margin": float(ci["mean_effect"]),
        "bootstrap_95ci_low": float(ci["ci_low"]),
        "bootstrap_95ci_high": float(ci["ci_high"]),
        "two_sided_exact_sign_flip_p": float(p_value),
        "source_exact_accuracy": float(seed_primary["source_exact_accuracy"].mean()),
        "blank_exact_accuracy": float(seed_primary["blank_exact_accuracy"].mean()),
        "three_layer_mean_restored_exact_accuracy": float(
            seed_primary["three_layer_mean_restored_exact_accuracy"].mean()
        ),
        "three_layer_mean_exact_gain": float(
            seed_primary["three_layer_mean_exact_gain"].mean()
        ),
        "positive_confirmation_support": bool(
            float(ci["ci_low"]) > 0.0 and float(p_value) < 0.05
        ),
    }
    return per_layer, seed_primary, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--frozen-layers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame, plan = _read(args.input, phase=str(args.phase))
    derived = _derive(frame)
    seed_layer, layer = _layer_metrics(derived)
    digits = _digit_proportions(derived)
    args.output.mkdir(parents=True, exist_ok=True)
    derived.to_csv(args.output / "seed_k_layer_metrics.csv", index=False)
    seed_layer.to_csv(args.output / "seed_layer_metrics.csv", index=False)
    layer.to_csv(args.output / "layer_metrics.csv", index=False)
    digits.to_csv(args.output / "predicted_digit_proportions.csv", index=False)

    if args.phase == "discovery":
        frozen = _freeze_top_three(
            layer_metrics=layer,
            model_label=str(plan["model_label"]),
            plan=plan,
        )
        _atomic_json(args.output / "frozen_layers.json", frozen)
        result = {
            "schema_version": "realistic_niah_v5_bullet_counter_analysis_v1",
            "status": "PASS",
            "phase": "discovery",
            "model_label": str(plan["model_label"]),
            "seed_count": int(plan["seed_count"]),
            "frozen_source_layers": frozen["source_layers"],
            "selection_primary": "seed-equal mean restoration target-margin gain",
            "confirmation_results_accessed": False,
        }
    else:
        if args.frozen_layers is None:
            raise ValueError("Confirmation analysis requires --frozen-layers")
        frozen = json.loads(args.frozen_layers.read_text(encoding="utf-8"))
        if (
            frozen.get("status") != "FROZEN_FROM_DISCOVERY"
            or str(frozen.get("model_label")) != str(plan["model_label"])
        ):
            raise ValueError("Frozen-layer registry is invalid for this model")
        frozen_layers = tuple(int(value) for value in frozen["source_layers"])
        per_layer, seed_primary, primary = _confirmation_statistics(
            derived, frozen_layers=frozen_layers
        )
        per_layer.to_csv(args.output / "confirmation_per_layer.csv", index=False)
        seed_primary.to_csv(
            args.output / "confirmation_seed_primary_effects.csv", index=False
        )
        result = {
            "schema_version": "realistic_niah_v5_bullet_counter_analysis_v1",
            "status": "PASS",
            "phase": "confirmation",
            "model_label": str(plan["model_label"]),
            "seed_count": int(plan["seed_count"]),
            "frozen_source_layers": list(frozen_layers),
            "primary_three_layer_average": primary,
            "per_layer_holm_family_size": 3,
            "allowed_claim_if_primary_positive": (
                "After same-length removal of prompt needles, non-item reasoning, "
                "and explicit item-progress markers, the complete hidden-state "
                "span of list item k is "
                "counterfactually sufficient to increase an immediate blank "
                "receiver's preference for k."
            ),
            "claim_limit": (
                "The list items were naturally generated with potentially explicit "
                "progress markers that are removed only on re-forward. This does "
                "not establish natural formation without "
                "visible indices, necessity, a counter update rule, or absence of "
                "readout-time retrieval. The estimand is conditional on the "
                "outcome-blind, format-selected complete-list trace stratum."
            ),
        }
    _atomic_json(args.output / "analysis.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
