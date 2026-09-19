#!/usr/bin/env python3
"""Select timing-specific NCC layers on discovery and test confirmation damage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import bootstrap_seed_mean_ci, sign_flip_pvalue  # noqa: E402
from realistic_niah_v5.stratified_targeted_counter_ncc import (  # noqa: E402
    STRATIFIED_NCC_ENDPOINTS,
)
from realistic_niah_v5.targeted_counter_ncc import NCC_CONDITIONS  # noqa: E402


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_path = root / "frozen_row_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"No frozen row plan under {root}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    files = sorted((root / "shards").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No stratified NCC shards under {root}")
    rows: list[dict[str, Any]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (
                metadata["experiment_id"]
                != "teacher_forced_stratified_targeted_counter_ncc"
            ):
                raise ValueError("Stratified NCC experiment ID changed")
            rows.append(
                {
                    "path": path,
                    "metadata": metadata,
                    "clean_basis": archive["clean_basis"].astype(np.float32),
                    "final_vectors": archive["final_vectors"].astype(np.float32),
                    "endpoint_names": tuple(
                        str(value) for value in archive["endpoint_names"]
                    ),
                    "layers": archive["layers"].astype(int),
                    "occurrences": archive["occurrences"].astype(int),
                    "condition_names": tuple(
                        str(value) for value in archive["condition_names"]
                    ),
                }
            )
    return rows, plan


def _audit(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    timing: str,
    phase: str,
) -> None:
    if str(plan["timing_branch"]) != timing or str(plan["seed_role"]) != phase:
        raise ValueError("Stratified NCC plan belongs to another timing or phase")
    expected_seeds = {int(value) for value in plan["seeds"]}
    seeds = {int(row["metadata"]["seed"]) for row in rows}
    if seeds != expected_seeds or len(rows) != len(seeds):
        raise ValueError(f"{phase} stratified NCC seed contract changed")
    expected_endpoints = STRATIFIED_NCC_ENDPOINTS[timing]
    expected_layers = tuple(int(value) for value in plan.get("layers", ()))
    for row in rows:
        meta = row["metadata"]
        if str(meta["timing_branch"]) != timing:
            raise ValueError("Stratified NCC shard timing changed")
        if not meta["outcome_blind"] or meta["selection_rank_used"]:
            raise ValueError("Stratified NCC outcome-blind contract changed")
        if not meta["all_capture_layers_strictly_above_all_ablated_heads"]:
            raise ValueError("Stratified NCC causal-reach contract changed")
        if tuple(row["condition_names"]) != NCC_CONDITIONS:
            raise ValueError("Stratified NCC condition order changed")
        if tuple(row["endpoint_names"]) != expected_endpoints:
            raise ValueError("Stratified NCC endpoint order changed")
        if row["clean_basis"].ndim != 4:
            raise ValueError("Stratified NCC clean basis must be event x endpoint x layer x hidden")
        if row["final_vectors"].ndim != 4:
            raise ValueError("Stratified NCC final vectors must be endpoint x condition x layer x hidden")
        if row["clean_basis"].shape[:3] != (
            len(row["occurrences"]),
            len(expected_endpoints),
            len(row["layers"]),
        ):
            raise ValueError("Stratified NCC clean-basis axes changed")
        if row["final_vectors"].shape[:3] != (
            len(expected_endpoints),
            len(NCC_CONDITIONS),
            len(row["layers"]),
        ):
            raise ValueError("Stratified NCC final-vector axes changed")
        if expected_layers and tuple(int(value) for value in row["layers"]) != expected_layers:
            raise ValueError("Stratified NCC planned layers changed")


def _fit_ncc(x: np.ndarray, y: np.ndarray, *, pca_dim: int = 16) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    _u, _s, vt = np.linalg.svd(z, full_matrices=False)
    dim = min(int(pca_dim), int(vt.shape[0]), max(1, int(x.shape[0]) - 1))
    components = vt[:dim]
    projected = z @ components.T
    classes = np.asarray(sorted(set(int(value) for value in y)), dtype=int)
    centroids = np.stack([projected[y == label].mean(axis=0) for label in classes])
    return {
        "mean": mean,
        "scale": scale,
        "components": components,
        "classes": classes,
        "centroids": centroids,
    }


def _distances(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    projected = ((x - model["mean"]) / model["scale"]) @ model["components"].T
    return ((projected[:, None, :] - model["centroids"][None, :, :]) ** 2).sum(
        axis=2
    )


def _balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(prediction[y == label] == label)
                for label in sorted(set(int(value) for value in y))
            ]
        )
    )


def _basis_rows(
    rows: list[dict[str, Any]], endpoint_index: int, layer_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    seeds: list[int] = []
    for row in rows:
        for occurrence, vector in zip(
            row["occurrences"],
            row["clean_basis"][:, endpoint_index, layer_index, :],
        ):
            vectors.append(vector)
            labels.append(int(occurrence))
            seeds.append(int(row["metadata"]["seed"]))
    if not vectors:
        raise ValueError("Stratified NCC has no clean discovery basis")
    return (
        np.stack(vectors),
        np.asarray(labels, dtype=int),
        np.asarray(seeds, dtype=int),
    )


def _correct_margin(
    distances: np.ndarray, classes: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    margins = np.empty(len(labels), dtype=float)
    for index, label in enumerate(labels):
        matched = np.where(classes == int(label))[0]
        if len(matched) != 1:
            raise ValueError(f"NCC label {label} has no unique centroid")
        gold_index = int(matched[0])
        other = np.delete(distances[index], gold_index)
        margins[index] = float(np.min(other) - distances[index, gold_index])
    return margins


def _grouped_oof(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    unique = sorted(set(int(value) for value in groups))
    folds = min(5, len(unique))
    if folds < 2:
        raise ValueError("Stratified NCC grouped OOF needs at least two seeds")
    fold_by_seed = {seed: index % folds for index, seed in enumerate(unique)}
    prediction = np.full(y.shape, -1, dtype=int)
    margin = np.full(y.shape, np.nan, dtype=float)
    for fold in range(folds):
        test = np.asarray([fold_by_seed[int(seed)] == fold for seed in groups])
        train = ~test
        if set(int(value) for value in y[test]) - set(int(value) for value in y[train]):
            raise ValueError("Stratified NCC OOF fold lacks a test class in training")
        model = _fit_ncc(x[train], y[train])
        distance = _distances(model, x[test])
        prediction[test] = model["classes"][distance.argmin(axis=1)]
        margin[test] = _correct_margin(distance, model["classes"], y[test])
    if (prediction < 0).any() or not np.isfinite(margin).all():
        raise RuntimeError("Stratified NCC OOF prediction is incomplete")
    return _balanced_accuracy(y, prediction), prediction, margin


def _summary(values: Iterable[float], name: str, random_seed: int) -> dict[str, Any]:
    active = np.asarray(list(values), dtype=float)
    result = bootstrap_seed_mean_ci(active, samples=10_000, seed=random_seed)
    result.update(
        {
            "estimand": name,
            "p_value_two_sided_sign_flip": sign_flip_pvalue(active),
            "higher_is_supportive": True,
        }
    )
    return result


def analyze(
    discovery: list[dict[str, Any]],
    discovery_plan: dict[str, Any],
    confirmation: list[dict[str, Any]],
    confirmation_plan: dict[str, Any],
    *,
    timing: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _audit(discovery, discovery_plan, timing=timing, phase="development")
    _audit(confirmation, confirmation_plan, timing=timing, phase="confirmation")
    layer_values = tuple(int(value) for value in discovery[0]["layers"])
    endpoint_names = tuple(discovery[0]["endpoint_names"])
    for row in discovery + confirmation:
        if tuple(int(value) for value in row["layers"]) != layer_values:
            raise ValueError("Stratified NCC layer registry changed across shards")
        if tuple(row["endpoint_names"]) != endpoint_names:
            raise ValueError("Stratified NCC endpoints changed across shards")

    layer_metrics: list[dict[str, Any]] = []
    selected_by_endpoint: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    margin_scales: dict[str, float] = {}
    basis_support: dict[str, dict[str, int]] = {}
    for endpoint_index, endpoint in enumerate(endpoint_names):
        endpoint_metrics: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(layer_values):
            x, y, seeds = _basis_rows(discovery, endpoint_index, layer_index)
            score, prediction, margins = _grouped_oof(x, y, seeds)
            active = {
                "endpoint": endpoint,
                "layer": int(layer),
                "grouped_oof_ncc_balanced_accuracy": score,
                "grouped_oof_exact_accuracy": float(np.mean(prediction == y)),
                "grouped_oof_correct_centroid_margin_mean": float(margins.mean()),
                "grouped_oof_correct_centroid_margin_sd": float(
                    margins.std(ddof=1)
                ),
                "basis_event_count": len(y),
                "basis_seed_count": len(set(int(value) for value in seeds)),
            }
            endpoint_metrics.append(active)
            layer_metrics.append(active)
        selected = max(
            endpoint_metrics,
            key=lambda row: (
                float(row["grouped_oof_ncc_balanced_accuracy"]),
                -int(row["layer"]),
            ),
        )
        selected_by_endpoint[endpoint] = selected
        layer_index = layer_values.index(int(selected["layer"]))
        x, y, seeds = _basis_rows(discovery, endpoint_index, layer_index)
        _score, _prediction, oof_margins = _grouped_oof(x, y, seeds)
        margin_scale = float(oof_margins.std(ddof=1))
        if not np.isfinite(margin_scale) or margin_scale <= 1e-8:
            raise ValueError("Discovery OOF margin scale is degenerate")
        margin_scales[endpoint] = margin_scale
        models[endpoint] = _fit_ncc(x, y)
        basis_support[endpoint] = {
            str(label): int(np.sum(y == label)) for label in sorted(set(y))
        }

    prediction_rows: list[dict[str, Any]] = []
    for row in confirmation:
        meta = row["metadata"]
        gold = int(meta["gold_count"])
        for endpoint_index, endpoint in enumerate(endpoint_names):
            model = models[endpoint]
            classes = model["classes"]
            if gold not in set(int(value) for value in classes):
                raise ValueError(f"Confirmation gold {gold} has no discovery centroid")
            layer_index = layer_values.index(int(selected_by_endpoint[endpoint]["layer"]))
            distance = _distances(
                model, row["final_vectors"][endpoint_index, :, layer_index, :]
            )
            labels = np.full(len(NCC_CONDITIONS), gold, dtype=int)
            margins = _correct_margin(distance, classes, labels)
            for condition_index, condition in enumerate(NCC_CONDITIONS):
                prediction = int(classes[int(np.argmin(distance[condition_index]))])
                prediction_rows.append(
                    {
                        "seed": int(meta["seed"]),
                        "request_id": str(meta["request_id"]),
                        "gold_count": gold,
                        "timing_branch": timing,
                        "endpoint": endpoint,
                        "selected_layer": int(selected_by_endpoint[endpoint]["layer"]),
                        "condition": condition,
                        "predicted_count": prediction,
                        "exact": float(prediction == gold),
                        "absolute_error": abs(prediction - gold),
                        "correct_centroid_margin": float(margins[condition_index]),
                        "standardized_correct_centroid_margin": float(
                            margins[condition_index] / margin_scales[endpoint]
                        ),
                    }
                )
    predictions = pd.DataFrame(prediction_rows)

    effect_frames: list[pd.DataFrame] = []
    endpoint_results: dict[str, dict[str, Any]] = {}
    randoms = ["random_mask_r1", "random_mask_r2", "random_mask_r3"]
    for endpoint_index, endpoint in enumerate(endpoint_names):
        active = predictions.loc[predictions["endpoint"].eq(endpoint)].copy()
        wide_margin = active.pivot(
            index="seed", columns="condition", values="correct_centroid_margin"
        )
        wide_mae = active.pivot(
            index="seed", columns="condition", values="absolute_error"
        )
        wide_exact = active.pivot(index="seed", columns="condition", values="exact")
        effects = pd.DataFrame(index=wide_margin.index)
        effects.index.name = "seed"
        effects["selected_correct_centroid_margin_loss"] = (
            wide_margin["clean"] - wide_margin["selected_mask"]
        )
        effects["random_mean_correct_centroid_margin_loss"] = (
            wide_margin["clean"] - wide_margin[randoms].mean(axis=1)
        )
        effects["selected_vs_random_margin_loss_specificity"] = (
            effects["selected_correct_centroid_margin_loss"]
            - effects["random_mean_correct_centroid_margin_loss"]
        )
        scale = margin_scales[endpoint]
        effects["standardized_selected_margin_loss"] = (
            effects["selected_correct_centroid_margin_loss"] / scale
        )
        effects["standardized_random_mean_margin_loss"] = (
            effects["random_mean_correct_centroid_margin_loss"] / scale
        )
        effects["standardized_selected_vs_random_specificity"] = (
            effects["selected_vs_random_margin_loss_specificity"] / scale
        )
        effects["selected_mae_increase"] = (
            wide_mae["selected_mask"] - wide_mae["clean"]
        )
        effects["selected_exact_accuracy_drop"] = (
            wide_exact["clean"] - wide_exact["selected_mask"]
        )
        effects = effects.reset_index()
        effects.insert(1, "timing_branch", timing)
        effects.insert(2, "endpoint", endpoint)
        effect_frames.append(effects)

        summaries = [
            _summary(
                effects[column].to_numpy(dtype=float),
                column,
                20260823 + endpoint_index * 100 + index,
            )
            for index, column in enumerate(effects.columns)
            if column not in {"seed", "timing_branch", "endpoint"}
        ]
        by_name = {row["estimand"]: row for row in summaries}
        primary = by_name["standardized_selected_margin_loss"]
        specificity = by_name["standardized_selected_vs_random_specificity"]
        directional = float(primary["mean_effect"]) > 0.0
        more_damaging_than_random = float(specificity["mean_effect"]) > 0.0
        interval_confirmed = (
            float(primary["ci_low"]) > 0.0 and float(specificity["ci_low"]) > 0.0
        )
        if interval_confirmed:
            status = "INTERVAL_CONFIRMED_DIRECTIONAL_SPECIFIC_SUPPORT"
        elif directional and more_damaging_than_random:
            status = "DIRECTIONAL_SPECIFIC_EVIDENCE"
        else:
            status = "NO_DIRECTIONAL_SPECIFIC_EVIDENCE"
        condition_metrics = []
        for condition in NCC_CONDITIONS:
            subset = active.loc[active["condition"].eq(condition)]
            condition_metrics.append(
                {
                    "condition": condition,
                    "exact_accuracy": float(subset["exact"].mean()),
                    "mean_absolute_error": float(subset["absolute_error"].mean()),
                    "mean_correct_centroid_margin": float(
                        subset["correct_centroid_margin"].mean()
                    ),
                    "mean_standardized_correct_centroid_margin": float(
                        subset["standardized_correct_centroid_margin"].mean()
                    ),
                }
            )
        clean_predictions = active.loc[active["condition"].eq("clean")]
        clean_balanced_accuracy = _balanced_accuracy(
            clean_predictions["gold_count"].to_numpy(dtype=int),
            clean_predictions["predicted_count"].to_numpy(dtype=int),
        )
        class_count = len(models[endpoint]["classes"])
        chance_balanced_accuracy = 1.0 / float(class_count)
        clean_mean_margin = next(
            row["mean_correct_centroid_margin"]
            for row in condition_metrics
            if row["condition"] == "clean"
        )
        # A positive intervention contrast is not evidence of damage to the
        # *correct-count* geometry when the clean confirmation state is, on
        # average, already closer to a wrong centroid.  Keep this validity gate
        # separate from the intervention gates so a large but uninterpretable
        # margin shift cannot be promoted to mechanistic confirmation.
        readout_validity = bool(
            float(selected_by_endpoint[endpoint]["grouped_oof_ncc_balanced_accuracy"])
            > chance_balanced_accuracy
            and clean_balanced_accuracy > chance_balanced_accuracy
            and float(clean_mean_margin) > 0.0
        )
        if directional and more_damaging_than_random and not readout_validity:
            status = "UNINTERPRETABLE_MARGIN_SHIFT_READOUT_VALIDITY_FAILURE"
        elif interval_confirmed and readout_validity:
            status = "VALID_READOUT_INTERVAL_DIRECTIONAL_SPECIFIC_SUPPORT"
        elif directional and more_damaging_than_random and readout_validity:
            status = "VALID_READOUT_DIRECTIONAL_SPECIFIC_EVIDENCE"
        else:
            status = "NO_DIRECTIONAL_SPECIFIC_EVIDENCE"
        endpoint_results[endpoint] = {
            "endpoint": endpoint,
            "is_primary_endpoint": endpoint == endpoint_names[0],
            "selected_layer": int(selected_by_endpoint[endpoint]["layer"]),
            "selected_layer_discovery_metrics": selected_by_endpoint[endpoint],
            "discovery_oof_margin_sd_for_standardization": scale,
            "discovery_basis_support_by_count": basis_support[endpoint],
            "condition_metrics": condition_metrics,
            "readout_validity": {
                "pass": readout_validity,
                "class_count": class_count,
                "chance_balanced_accuracy": chance_balanced_accuracy,
                "discovery_grouped_oof_balanced_accuracy": selected_by_endpoint[
                    endpoint
                ]["grouped_oof_ncc_balanced_accuracy"],
                "clean_confirmation_balanced_accuracy": clean_balanced_accuracy,
                "clean_confirmation_mean_correct_centroid_margin": float(
                    clean_mean_margin
                ),
                "rule": (
                    "discovery OOF BA > 1/K AND clean confirmation BA > 1/K "
                    "AND clean confirmation mean correct-centroid margin > 0"
                ),
                "post_analysis_validity_qualification": True,
            },
            "raw_primary_estimand": by_name[
                "selected_correct_centroid_margin_loss"
            ],
            "raw_specificity_estimand": by_name[
                "selected_vs_random_margin_loss_specificity"
            ],
            "standardized_primary_estimand": primary,
            "standardized_specificity_estimand": specificity,
            "all_estimands": summaries,
            "ncc_effect_status": status,
            "selected_mask_changes_ncc_directionally": directional,
            "selected_mask_more_damaging_than_random": more_damaging_than_random,
            "bootstrap_interval_excludes_zero_for_both_gates": interval_confirmed,
        }

    effects = pd.concat(effect_frames, ignore_index=True)
    primary_endpoint = endpoint_names[0]
    primary_result = endpoint_results[primary_endpoint]
    result = {
        "schema_version": "realistic_niah_v5_stratified_targeted_counter_ncc_analysis_v2",
        "status": "PASS",
        "model_label": str(discovery[0]["metadata"]["model_label"]),
        "timing_branch": timing,
        "development_seed_count": len(discovery),
        "confirmation_seed_count": len(confirmation),
        "development_seeds": sorted(
            int(row["metadata"]["seed"]) for row in discovery
        ),
        "confirmation_seeds": sorted(
            int(row["metadata"]["seed"]) for row in confirmation
        ),
        "cohorts_are_maximal_eligible_within_fixed_phase": True,
        "primary_endpoint": primary_endpoint,
        "endpoint_results": endpoint_results,
        "primary_endpoint_result": primary_result,
        "layer_selection_rule": (
            "within_endpoint maximize discovery grouped-OOF clean NCC balanced "
            "accuracy; tie earlier causally reachable layer"
        ),
        "capture_layer_rule": "strictly_above_all_ablated_head_layers",
        "pca_dim": 16,
        "fit_data": "branch discovery clean states from matching explicit events only",
        "margin_standardization": (
            "divide by discovery grouped-OOF correct-centroid-margin SD at the "
            "endpoint-selected layer"
        ),
        "readout_validity_is_a_separate_interpretability_gate": True,
        "readout_validity_gate_registered_after_initial_contrast_inspection": True,
        "confirmation_used_for_fit_or_layer_selection": False,
        "confirmation_status": "registered_fixed_split_retrospective_extension",
        "outcome_blind": True,
        "selection_rank_used": False,
        "allowed_claim": (
            "The timing-specific frozen retrieval-head intervention changed the "
            "registered confirmation counter-state geometry according to the "
            f"reported gates ({primary_result['ncc_effect_status']})."
        ),
        "restrictions": [
            "The two timing branches use maximal eligible, unpaired seed cohorts.",
            "Raw margins are endpoint- and model-scale dependent and must not be pooled.",
            "Confirmation is a registered-split retrospective extension because earlier NCC variants were inspected.",
            "NCC is a diagnostic readout and does not establish exclusive natural-answer use of the decoder.",
        ],
    }
    return pd.DataFrame(layer_metrics), effects, predictions, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument(
        "--timing", choices=tuple(STRATIFIED_NCC_ENDPOINTS), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    discovery, discovery_plan = _load(args.discovery)
    confirmation, confirmation_plan = _load(args.confirmation)
    layer_metrics, effects, predictions, result = analyze(
        discovery,
        discovery_plan,
        confirmation,
        confirmation_plan,
        timing=str(args.timing),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    layer_metrics.to_csv(args.output / "layer_metrics.csv", index=False)
    effects.to_csv(args.output / "seed_effects.csv", index=False)
    predictions.to_csv(args.output / "confirmation_predictions.csv", index=False)
    _atomic_json(args.output / "claim_gates.json", result)
    _atomic_json(
        args.output / "audit.json",
        {
            "status": "PASS",
            "model_label": result["model_label"],
            "timing_branch": str(args.timing),
            "development_seed_count": len(discovery),
            "confirmation_seed_count": len(confirmation),
            "confirmation_used_for_fit_or_layer_selection": False,
            "capture_layer_rule": "strictly_above_all_ablated_head_layers",
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )
    print(
        json.dumps(
            {
                "model_label": result["model_label"],
                "timing_branch": str(args.timing),
                "primary_endpoint": result["primary_endpoint"],
                "selected_layer": result["primary_endpoint_result"][
                    "selected_layer"
                ],
                "effect_status": result["primary_endpoint_result"][
                    "ncc_effect_status"
                ],
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
