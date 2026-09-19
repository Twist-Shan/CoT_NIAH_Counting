#!/usr/bin/env python3
"""Fit discovery-only NCCs and evaluate targeted-head damage on confirmation."""

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
from realistic_niah_v5.count_stream import (  # noqa: E402
    COUNT_STREAM_CONFIRMATION_SEEDS,
    COUNT_STREAM_DISCOVERY_SEEDS,
)


TIMINGS = ("rank_after_city", "rank_before_city")
CONDITIONS = (
    "clean",
    "selected_mask",
    "random_mask_r1",
    "random_mask_r2",
    "random_mask_r3",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / "shards").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NCC shards under {root}")
    rows: list[dict[str, Any]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata["experiment_id"] != "teacher_forced_targeted_counter_ncc":
                raise ValueError("NCC experiment id changed")
            rows.append(
                {
                    "path": path,
                    "metadata": metadata,
                    "clean_basis": archive["clean_basis"].astype(np.float32),
                    "final_vectors": archive["final_vectors"].astype(np.float32),
                    "layers": archive["layers"].astype(int),
                    "occurrences": archive["occurrences"].astype(int),
                    "condition_names": tuple(str(x) for x in archive["condition_names"]),
                }
            )
    return rows


def _audit(rows: list[dict[str, Any]], expected: Iterable[int], phase: str) -> None:
    seeds = {int(row["metadata"]["seed"]) for row in rows}
    if seeds != set(int(x) for x in expected) or len(rows) != len(seeds):
        raise ValueError(f"{phase} NCC seed contract changed: {sorted(seeds)}")
    for row in rows:
        meta = row["metadata"]
        if not meta["outcome_blind"] or meta["selection_rank_used"]:
            raise ValueError("NCC outcome-blind contract changed")
        if tuple(row["condition_names"]) != CONDITIONS:
            raise ValueError("NCC condition order changed")
        if row["clean_basis"].shape[0] != int(meta["gold_count"]):
            raise ValueError("NCC clean-basis occurrence axis changed")
        if row["final_vectors"].shape[0] != len(CONDITIONS):
            raise ValueError("NCC final condition axis changed")


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
    return ((projected[:, None, :] - model["centroids"][None, :, :]) ** 2).sum(axis=2)


def _balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> float:
    recalls = [float(np.mean(prediction[y == label] == label)) for label in sorted(set(y))]
    return float(np.mean(recalls))


def _basis_rows(rows: list[dict[str, Any]], layer_index: int, timing: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    seeds: list[int] = []
    for row in rows:
        meta = row["metadata"]
        for occurrence, active_timing, vector in zip(
            row["occurrences"],
            meta["grammar_timing_strata"],
            row["clean_basis"][:, layer_index, :],
        ):
            if str(active_timing) != timing:
                continue
            vectors.append(vector)
            labels.append(int(occurrence))
            seeds.append(int(meta["seed"]))
    if not vectors:
        raise ValueError(f"NCC has no discovery basis for {timing}")
    return np.stack(vectors), np.asarray(labels, dtype=int), np.asarray(seeds, dtype=int)


def _grouped_oof_ba(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    unique = sorted(set(int(value) for value in groups))
    fold_by_seed = {seed: index % 5 for index, seed in enumerate(unique)}
    prediction = np.full(y.shape, -1, dtype=int)
    for fold in range(5):
        test = np.asarray([fold_by_seed[int(seed)] == fold for seed in groups])
        train = ~test
        train_classes = set(int(value) for value in y[train])
        if set(int(value) for value in y[test]) - train_classes:
            raise ValueError("NCC grouped fold lacks a test class in training")
        model = _fit_ncc(x[train], y[train])
        distance = _distances(model, x[test])
        prediction[test] = model["classes"][distance.argmin(axis=1)]
    if (prediction < 0).any():
        raise RuntimeError("NCC OOF prediction is incomplete")
    return _balanced_accuracy(y, prediction)


def _summary(values: np.ndarray, name: str, random_seed: int) -> dict[str, Any]:
    result = bootstrap_seed_mean_ci(values.astype(float), samples=10_000, seed=random_seed)
    result.update(
        {
            "estimand": name,
            "p_value": sign_flip_pvalue(values.astype(float)),
            "higher_is_supportive": True,
        }
    )
    return result


def analyze(
    discovery: list[dict[str, Any]], confirmation: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _audit(discovery, COUNT_STREAM_DISCOVERY_SEEDS, "discovery")
    _audit(confirmation, COUNT_STREAM_CONFIRMATION_SEEDS, "confirmation")
    layer_values = tuple(int(x) for x in discovery[0]["layers"])
    if any(tuple(int(x) for x in row["layers"]) != layer_values for row in discovery + confirmation):
        raise ValueError("NCC layer registry changed across shards")

    layer_metrics: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layer_values):
        timing_scores = {}
        for timing in TIMINGS:
            x, y, seeds = _basis_rows(discovery, layer_index, timing)
            timing_scores[timing] = _grouped_oof_ba(x, y, seeds)
        layer_metrics.append(
            {
                "layer": int(layer),
                **{f"{timing}_oof_ncc_balanced_accuracy": timing_scores[timing] for timing in TIMINGS},
                "mean_timing_oof_ncc_balanced_accuracy": float(np.mean(list(timing_scores.values()))),
            }
        )
    selected_metric = max(
        layer_metrics,
        key=lambda row: (float(row["mean_timing_oof_ncc_balanced_accuracy"]), -int(row["layer"])),
    )
    selected_layer = int(selected_metric["layer"])
    layer_index = layer_values.index(selected_layer)
    models: dict[str, dict[str, Any]] = {}
    basis_support: dict[str, dict[str, int]] = {}
    for timing in TIMINGS:
        x, y, _seeds = _basis_rows(discovery, layer_index, timing)
        models[timing] = _fit_ncc(x, y)
        basis_support[timing] = {
            str(label): int(np.sum(y == label)) for label in sorted(set(y))
        }

    prediction_rows: list[dict[str, Any]] = []
    for row in confirmation:
        meta = row["metadata"]
        timing = str(meta["final_grammar_timing_stratum"])
        model = models[timing]
        gold = int(meta["gold_count"])
        classes = model["classes"]
        if gold not in set(int(value) for value in classes):
            raise ValueError(f"Confirmation gold {gold} has no discovery centroid")
        gold_index = int(np.where(classes == gold)[0][0])
        distance = _distances(model, row["final_vectors"][:, layer_index, :])
        for condition_index, condition in enumerate(CONDITIONS):
            active = distance[condition_index]
            prediction = int(classes[int(np.argmin(active))])
            other = np.delete(active, gold_index)
            margin = float(np.min(other) - active[gold_index])
            prediction_rows.append(
                {
                    "seed": int(meta["seed"]),
                    "request_id": str(meta["request_id"]),
                    "gold_count": gold,
                    "grammar_timing_stratum": timing,
                    "condition": condition,
                    "predicted_count": prediction,
                    "exact": float(prediction == gold),
                    "absolute_error": abs(prediction - gold),
                    "correct_centroid_margin": margin,
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    wide_margin = predictions.pivot(index="seed", columns="condition", values="correct_centroid_margin")
    wide_mae = predictions.pivot(index="seed", columns="condition", values="absolute_error")
    wide_exact = predictions.pivot(index="seed", columns="condition", values="exact")
    randoms = ["random_mask_r1", "random_mask_r2", "random_mask_r3"]
    effects = pd.DataFrame(index=wide_margin.index)
    effects.index.name = "seed"
    effects["selected_correct_centroid_margin_loss"] = wide_margin["clean"] - wide_margin["selected_mask"]
    effects["random_mean_correct_centroid_margin_loss"] = wide_margin["clean"] - wide_margin[randoms].mean(axis=1)
    effects["selected_vs_random_margin_loss_specificity"] = (
        effects["selected_correct_centroid_margin_loss"]
        - effects["random_mean_correct_centroid_margin_loss"]
    )
    effects["selected_mae_increase"] = wide_mae["selected_mask"] - wide_mae["clean"]
    effects["selected_exact_accuracy_drop"] = wide_exact["clean"] - wide_exact["selected_mask"]
    effects = effects.reset_index()
    summaries = [
        _summary(effects[column].to_numpy(dtype=float), column, 20260823 + index)
        for index, column in enumerate(effects.columns)
        if column != "seed"
    ]
    by_name = {row["estimand"]: row for row in summaries}
    condition_metrics = []
    for condition in CONDITIONS:
        active = predictions.loc[predictions["condition"].eq(condition)]
        condition_metrics.append(
            {
                "condition": condition,
                "exact_accuracy": float(active["exact"].mean()),
                "mean_absolute_error": float(active["absolute_error"].mean()),
                "mean_correct_centroid_margin": float(active["correct_centroid_margin"].mean()),
            }
        )
    primary = by_name["selected_correct_centroid_margin_loss"]
    specificity = by_name["selected_vs_random_margin_loss_specificity"]
    directional = bool(float(primary["mean_effect"]) > 0.0)
    more_damaging_than_random = bool(float(specificity["mean_effect"]) > 0.0)
    effect_status = (
        "DIRECTIONAL_SPECIFIC_SUPPORT"
        if directional and more_damaging_than_random
        else "NO_DIRECTIONAL_SPECIFIC_SUPPORT"
    )
    allowed_claim = (
        "At a discovery-selected grammar-carrier layer, masking the frozen "
        "targeted retrieval bank directionally moves the confirmation carrier "
        "away from the correct running-count centroid more than the mean "
        "layer-matched random bank."
        if effect_status == "DIRECTIONAL_SPECIFIC_SUPPORT"
        else "The frozen targeted retrieval bank did not directionally move the "
        "confirmation carrier away from the correct running-count centroid more "
        "than the mean layer-matched random bank."
    )
    result = {
        "schema_version": "realistic_niah_v5_targeted_counter_ncc_analysis_v1",
        "status": "PASS",
        "discovery_seed_count": 20,
        "confirmation_seed_count": 10,
        "selected_layer": selected_layer,
        "layer_selection_rule": "maximize mean grammar-stratum grouped-OOF clean NCC BA; tie earlier",
        "selected_layer_discovery_metrics": selected_metric,
        "discovery_basis_support_by_timing_and_count": basis_support,
        "pca_dim": 16,
        "fit_data": "discovery clean carrier states from all completed occurrences",
        "confirmation_used_for_fit_or_layer_selection": False,
        "condition_metrics": condition_metrics,
        "primary_estimand": primary,
        "specificity_estimand": specificity,
        "all_estimands": summaries,
        "ncc_effect_status": effect_status,
        "selected_mask_changes_ncc_directionally": directional,
        "selected_mask_more_damaging_than_random": more_damaging_than_random,
        "outcome_blind": True,
        "selection_rank_used": False,
        "allowed_claim": allowed_claim,
        "restriction": (
            "NCC is a discovery-fitted readout of carrier geometry; it does not "
            "by itself show that the model's natural answer exclusively uses this decoder."
        ),
    }
    return pd.DataFrame(layer_metrics), effects, predictions, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layer_metrics, effects, predictions, result = analyze(
        _load(args.discovery), _load(args.confirmation)
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
            "discovery_seed_count": 20,
            "confirmation_seed_count": 10,
            "confirmation_used_for_fit_or_layer_selection": False,
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )
    print(json.dumps({"selected_layer": result["selected_layer"], "status": "PASS"}))


if __name__ == "__main__":
    main()
