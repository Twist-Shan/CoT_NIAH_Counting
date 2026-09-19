#!/usr/bin/env python3
"""Analyze the no-index old-HTML counter restoration and freeze its layer."""

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
        raise FileNotFoundError(f"No unnumbered restore shards under {root}")
    rows = []
    for path in files:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    frame = pd.DataFrame(rows)
    experiment_ids = set(frame["experiment_id"].astype(str))
    allowed_experiment_ids = {
        "unnumbered_old_html_counter_restore",
        "unnumbered_old_html_counter_early_stop_restore",
    }
    if len(experiment_ids) != 1 or not experiment_ids <= allowed_experiment_ids:
        raise ValueError("Unnumbered restore experiment id changed")
    if not frame["trace_has_no_explicit_running_index"].astype(bool).all():
        raise ValueError("Unnumbered restore contains an explicit running index")
    if not frame["receiver_has_no_prompt_or_trace_needles"].astype(bool).all():
        raise ValueError("Unnumbered restore receiver still contains a needle")
    panel_kinds = set(frame["trace_panel_kind"].astype(str))
    if panel_kinds not in (
        {"teacher_forced_unnumbered_gold_bullets"},
        {"model_generated_no_count_enumeration"},
    ):
        raise ValueError("Unnumbered restore panel provenance changed")
    natural_panel = panel_kinds == {"model_generated_no_count_enumeration"}
    prompt_conditioned_a7 = bool(
        "prompt_conditioned_a7_auxiliary" in frame
        and frame["prompt_conditioned_a7_auxiliary"].astype(bool).all()
    )
    if "prompt_conditioned_a7_auxiliary" in frame:
        auxiliary_flags = frame["prompt_conditioned_a7_auxiliary"].astype(bool)
        if bool(auxiliary_flags.any()) != bool(auxiliary_flags.all()):
            raise ValueError("A7 auxiliary provenance is mixed within one analysis")
    natural_flags = frame["natural_generation_claim_allowed"].astype(bool)
    expected_natural_claim = natural_panel and not prompt_conditioned_a7
    if (
        bool(natural_flags.all()) != expected_natural_claim
        or bool(natural_flags.any()) != expected_natural_claim
    ):
        raise ValueError("Unnumbered restore natural-generation provenance is inconsistent")
    if "trace_tokens_teacher_forced" in frame:
        teacher_flags = frame["trace_tokens_teacher_forced"].astype(bool)
        if bool(teacher_flags.any()) == natural_panel or bool(teacher_flags.all()) == natural_panel:
            raise ValueError("Unnumbered restore teacher-forcing provenance is inconsistent")
    if frame["selection_rank_used"].astype(bool).any() or not frame["outcome_blind"].astype(bool).all():
        raise ValueError("Unnumbered restore outcome-blind contract changed")
    return frame


def _scores(value: Any) -> np.ndarray:
    result = np.asarray([float(piece) for piece in str(value).split(",")], dtype=float)
    if result.shape != (10,) or not np.isfinite(result).all():
        raise ValueError("Candidate score vector changed")
    return result


def _target_metrics(scores: np.ndarray, target: int) -> dict[str, float]:
    index = int(target) - 1
    other = np.delete(scores, index)
    prediction = int(np.argmax(scores)) + 1
    return {
        "prediction": prediction,
        "exact": float(prediction == int(target)),
        "absolute_error": float(abs(prediction - int(target))),
        "target_margin": float(scores[index] - float(other.max())),
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
    expected_seed_order: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    default_seed_order = (
        tuple(range(1234, 1254))
        if phase == "discovery"
        else tuple(range(1254, 1264))
    )
    expected_seed_order = expected_seed_order or default_seed_order
    expected_count = 20 if phase == "discovery" else 10
    if (
        len(expected_seed_order) != expected_count
        or len(set(expected_seed_order)) != expected_count
    ):
        raise ValueError(f"{phase} analysis requires {expected_count} unique seeds")
    expected_seeds = set(expected_seed_order)
    observed = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
    if observed != expected_seeds:
        raise ValueError(f"{phase} unnumbered restore seed contract changed")
    experiment_id = str(frame["experiment_id"].astype(str).iloc[0])
    early_stop = experiment_id == "unnumbered_old_html_counter_early_stop_restore"
    baseline_condition = (
        "early_stop_uninformative" if early_stop else "fully_uninformative"
    )
    patch_condition = (
        "occurrence_state_restore_early_stop"
        if early_stop
        else "occurrence_state_restore"
    )
    baseline_rows = frame.loc[
        frame["condition"].astype(str).eq(baseline_condition)
    ]
    patch_rows = frame.loc[
        frame["condition"].astype(str).eq(patch_condition)
    ].copy()
    expected_baselines = len(expected_seeds) * (8 if early_stop else 1)
    if len(baseline_rows) != expected_baselines:
        raise ValueError(
            f"Unnumbered restore needs {expected_baselines} registered baselines"
        )
    targets = set(pd.to_numeric(patch_rows["target_occurrence"], errors="raise").astype(int))
    if targets != set(range(2, 10)):
        raise ValueError(f"Unnumbered restore target registry changed: {targets}")
    if early_stop:
        if "future_trace_tokens_present" not in frame or frame[
            "future_trace_tokens_present"
        ].astype(bool).any():
            raise ValueError("Early-stop restore still contains future trace tokens")
        if set(frame["readout_mode"].astype(str)) != {
            "immediate_item_k_early_stop_minimal_terminal_suffix"
        }:
            raise ValueError("Early-stop readout mode changed")
        baseline_by_seed_target = {
            (int(row.seed), int(row.target_occurrence)): _scores(
                row.candidate_log_scores
            )
            for row in baseline_rows.itertuples(index=False)
        }
        if len(baseline_by_seed_target) != expected_baselines:
            raise ValueError("Early-stop baselines duplicate seed/target cells")
    else:
        baseline_by_seed = {
            int(row.seed): _scores(row.candidate_log_scores)
            for row in baseline_rows.itertuples(index=False)
        }
    derived_rows: list[dict[str, Any]] = []
    for row in patch_rows.itertuples(index=False):
        seed = int(row.seed)
        target = int(row.target_occurrence)
        layer = int(row.source_layer)
        baseline_scores = (
            baseline_by_seed_target[(seed, target)]
            if early_stop
            else baseline_by_seed[seed]
        )
        baseline = _target_metrics(baseline_scores, target)
        patched = _target_metrics(_scores(row.candidate_log_scores), target)
        derived_rows.append(
            {
                "seed": seed,
                "source_layer": layer,
                "target_occurrence": target,
                "baseline_prediction": int(baseline["prediction"]),
                "patched_prediction": int(patched["prediction"]),
                "baseline_exact": baseline["exact"],
                "patched_exact": patched["exact"],
                "exact_accuracy_gain": patched["exact"] - baseline["exact"],
                "baseline_absolute_error": baseline["absolute_error"],
                "patched_absolute_error": patched["absolute_error"],
                "mae_reduction": baseline["absolute_error"] - patched["absolute_error"],
                "baseline_target_margin": baseline["target_margin"],
                "patched_target_margin": patched["target_margin"],
                "target_margin_gain": patched["target_margin"] - baseline["target_margin"],
            }
        )
    derived = pd.DataFrame(derived_rows)
    layer_rows = []
    for layer, active in derived.groupby("source_layer"):
        occurrence_means = active.groupby("target_occurrence")["target_margin_gain"].mean()
        layer_rows.append(
            {
                "source_layer": int(layer),
                "patched_exact_accuracy": float(active["patched_exact"].mean()),
                "baseline_exact_accuracy": float(active["baseline_exact"].mean()),
                "exact_accuracy_gain": float(active["exact_accuracy_gain"].mean()),
                "patched_mean_absolute_error": float(active["patched_absolute_error"].mean()),
                "baseline_mean_absolute_error": float(active["baseline_absolute_error"].mean()),
                "mean_mae_reduction": float(active["mae_reduction"].mean()),
                "mean_target_margin_gain": float(active["target_margin_gain"].mean()),
                "positive_occurrence_margin_gain_count": int((occurrence_means > 0.0).sum()),
            }
        )
    layers = pd.DataFrame(layer_rows).sort_values("source_layer").reset_index(drop=True)
    if phase == "discovery":
        chosen = max(
            layer_rows,
            key=lambda row: (
                float(row["patched_exact_accuracy"]),
                float(row["mean_target_margin_gain"]),
                int(row["source_layer"]),
            ),
        )
        selected_layer = int(chosen["source_layer"])
    else:
        if frozen_layer is None:
            raise ValueError("Confirmation requires --frozen-layer")
        if set(int(value) for value in derived["source_layer"]) != {int(frozen_layer)}:
            raise ValueError("Confirmation source layer differs from discovery decision")
        selected_layer = int(frozen_layer)
        chosen = next(row for row in layer_rows if int(row["source_layer"]) == selected_layer)
    selected = derived.loc[derived["source_layer"].eq(selected_layer)].copy()
    seed_effects = (
        selected.groupby("seed", as_index=False)[
            ["exact_accuracy_gain", "mae_reduction", "target_margin_gain"]
        ]
        .mean()
        .sort_values("seed")
    )
    summaries = [
        _summary(seed_effects[column].to_numpy(dtype=float), column, 20260823 + index)
        for index, column in enumerate(
            ("exact_accuracy_gain", "mae_reduction", "target_margin_gain")
        )
    ]
    occurrence = (
        selected.groupby("target_occurrence", as_index=False)
        .agg(
            patched_exact_accuracy=("patched_exact", "mean"),
            baseline_exact_accuracy=("baseline_exact", "mean"),
            patched_mean_absolute_error=("patched_absolute_error", "mean"),
            mean_target_margin_gain=("target_margin_gain", "mean"),
        )
        .sort_values("target_occurrence")
    )
    magnitude_pass = bool(
        float(chosen["patched_exact_accuracy"]) >= 0.50
        and float(chosen["exact_accuracy_gain"]) >= 0.25
        and float(chosen["mean_target_margin_gain"]) > 0.0
        and int(chosen["positive_occurrence_margin_gain_count"]) >= 6
    )
    panel_kind = str(frame["trace_panel_kind"].astype(str).iloc[0])
    natural_panel = panel_kind == "model_generated_no_count_enumeration"
    prompt_conditioned_a7 = bool(
        "prompt_conditioned_a7_auxiliary" in frame
        and frame["prompt_conditioned_a7_auxiliary"].astype(bool).all()
    )
    result = {
        "schema_version": "realistic_niah_v5_unnumbered_counter_restore_analysis_v3",
        "status": "PASS",
        "experiment_id": experiment_id,
        "readout_mode": (
            "immediate_item_k_early_stop_minimal_terminal_suffix"
            if early_stop
            else "full_original_trace_to_answer_query"
        ),
        "phase": phase,
        "seed_count": len(expected_seeds),
        "seeds": list(expected_seed_order),
        "selected_layer": selected_layer,
        "selection_rule": (
            "discovery maximize seed-equal patch exact accuracy, then margin gain, "
            "then prefer later layer; confirmation frozen"
        ),
        "selected_layer_metrics": chosen,
        "primary_estimands": summaries,
        "magnitude_gate": {
            "patched_exact_accuracy_at_least_0_50": float(chosen["patched_exact_accuracy"]) >= 0.50,
            "exact_accuracy_gain_at_least_0_25": float(chosen["exact_accuracy_gain"]) >= 0.25,
            "mean_target_margin_gain_positive": float(chosen["mean_target_margin_gain"]) > 0.0,
            "at_least_6_of_8_occurrences_positive": int(chosen["positive_occurrence_margin_gain_count"]) >= 6,
        },
        "old_html_internal_counter_magnitude_pass": magnitude_pass,
        "trace_has_no_explicit_running_index": True,
        "receiver_has_no_prompt_or_trace_needles": True,
        "score_digits_are_not_running_indices": True,
        "trace_panel_kind": panel_kind,
        "natural_generation_claim_allowed": natural_panel and not prompt_conditioned_a7,
        "trace_tokens_teacher_forced": not natural_panel,
        "prompt_conditioned_a7_auxiliary": prompt_conditioned_a7,
        "formal_frozen_prompt_claim_allowed": not prompt_conditioned_a7,
        "controlled_hidden_state_sufficiency_claim_allowed": True,
        "outcome_blind": True,
        "selection_rank_used": False,
        "allowed_claim_if_confirmation_passes": (
            "Within the explicitly prompt-conditioned A7 bullet grammar, after deleting "
            "every future trace item and retaining only the minimal terminal answer suffix, "
            "the clean full-item hidden state at item k is sufficient to make an otherwise "
            "no-needle receiver prefer k as the immediate early-stop count."
            if early_stop and prompt_conditioned_a7
            else "After deleting every future trace item and retaining only the minimal "
            "terminal answer suffix, the clean full-item hidden state at item k is "
            "sufficient to make an otherwise no-needle receiver prefer k as the "
            "immediate early-stop count."
            if early_stop
            else "In a model-generated trace whose item-k causal prefix has no explicit "
            "record enumeration or already-stated count, the clean hidden state at "
            "item k is sufficient to make an otherwise no-needle receiver prefer k "
            "as an early-stop count."
            if natural_panel
            else "In a teacher-forced trace with no explicit running index, the clean "
            "hidden state at item k is sufficient to make an otherwise no-needle "
            "receiver prefer k as an early-stop count."
        ),
        "restriction": (
            "A7 appends an explicit bullet-only reasoning instruction and a fixed '- ' "
            "assistant prefix. This auxiliary result tests state sufficiency conditional "
            "on that induced grammar and cannot establish a natural-prompt mechanism. "
            "The readout removes all k+1..N trace items and teacher-forces only the minimal "
            "terminal grammar suffix needed to reach Total:."
            if early_stop and prompt_conditioned_a7
            else "The readout removes all k+1..N trace items but teacher-forces the minimal "
            "terminal grammar suffix needed to reach Total:. It is a controlled "
            "immediate-readout sufficiency test, not unconstrained free generation."
            if early_stop
            else "This is a format-conditioned natural-generation sufficiency result under "
            "cumulative full-state clamping. A fixed dash prefix constrains grammar but "
            "contains no count information; the result does not claim that a single "
            "token/layer is necessary or estimate unconditional grammar prevalence."
            if natural_panel
            else "This is a controlled sufficiency result under cumulative full-state "
            "clamping. It does not estimate how often the model naturally emits this "
            "trace grammar and does not claim that a single token/layer is necessary."
        ),
    }
    return layers, derived, occurrence, seed_effects, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--frozen-layer", type=int)
    parser.add_argument("--expected-seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers, derived, occurrence, seed_effects, result = analyze(
        _read(args.input),
        phase=args.phase,
        frozen_layer=args.frozen_layer,
        expected_seed_order=(
            tuple(int(value) for value in args.expected_seeds)
            if args.expected_seeds
            else None
        ),
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
            "trace_has_no_explicit_running_index": True,
            "receiver_has_no_prompt_or_trace_needles": True,
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )
    print(
        json.dumps(
            {
                "selected_layer": result["selected_layer"],
                "magnitude_pass": result["old_html_internal_counter_magnitude_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
