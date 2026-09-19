#!/usr/bin/env python3
"""Analyze Native answer-query full-state donor adoption across layers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "realistic_niah_v5_answer_query_layer_sweep_analysis_v1"
REGISTERED_COUNTS = set(range(1, 11))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return int(value)


def cluster_bootstrap(
    values: np.ndarray, *, label: str, repetitions: int = 10_000
) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    draws = values[
        rng.integers(0, len(values), size=(int(repetitions), len(values)))
    ].mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def analyze(
    trials_path: Path,
    pairs_path: Path,
    output_dir: Path,
    *,
    expected_layers: list[int],
) -> dict[str, Any]:
    trials = read_jsonl(trials_path)
    pairs = read_jsonl(pairs_path)
    pair_registry = {str(row["pair_id"]): row for row in pairs}
    if len(pair_registry) != len(pairs):
        raise ValueError("Pair registry contains duplicate pair_id values")

    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in trials:
        key = (str(row["pair_id"]), int(row["layer"]))
        condition = str(row["condition"])
        if condition in grouped.setdefault(key, {}):
            raise ValueError(f"Duplicate trial condition for {key}: {condition}")
        grouped[key][condition] = row
    frozen_layers = sorted(set(int(layer) for layer in expected_layers))
    if not frozen_layers or len(frozen_layers) != len(expected_layers):
        raise ValueError("Expected layers must be non-empty and unique")
    observed_layers = sorted({int(row["layer"]) for row in trials})
    if observed_layers != frozen_layers:
        raise ValueError(
            "Observed layers do not match the preregistered grid: "
            f"observed={observed_layers} expected={frozen_layers}"
        )
    expected_cells = {
        (pair_id, layer) for pair_id in pair_registry for layer in frozen_layers
    }
    if set(grouped) != expected_cells:
        missing = sorted(expected_cells - set(grouped))
        extra = sorted(set(grouped) - expected_cells)
        raise ValueError(
            f"Trial grid mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    detail_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for (pair_id, layer), conditions in sorted(grouped.items()):
        if pair_id not in pair_registry:
            raise ValueError(f"Trial pair is absent from frozen registry: {pair_id}")
        if set(conditions) != {"self_patch", "full_donor_patch"}:
            raise ValueError(
                f"Expected self/full conditions for {pair_id}/L{layer}: "
                f"{sorted(conditions)}"
            )
        pair = pair_registry[pair_id]
        receiver = int(pair["receiver_count"])
        donor = int(pair["donor_count"])
        if receiver == donor:
            raise ValueError(f"Zero-gap pair: {pair_id}")
        self_prediction = optional_int(conditions["self_patch"].get("prediction"))
        if self_prediction != receiver:
            raise ValueError(
                f"Self patch did not reproduce receiver gold for {pair_id}/L{layer}: "
                f"{self_prediction} != {receiver}"
            )
        full_prediction = optional_int(
            conditions["full_donor_patch"].get("prediction")
        )
        full_valid = full_prediction in REGISTERED_COUNTS
        full_adoption = bool(full_prediction == donor)
        self_adoption = bool(self_prediction == donor)
        transport = (
            float((full_prediction - receiver) / (donor - receiver))
            if full_valid and full_prediction is not None
            else 0.0
        )
        common = {
            "pair_id": pair_id,
            "model_label": str(pair["model_label"]),
            "seed": int(pair["seed"]),
            "layer": int(layer),
            "receiver_count": receiver,
            "donor_count": donor,
            "signed_count_gap": donor - receiver,
            "pair_direction": str(pair["pair_direction"]),
        }
        for condition, row in sorted(conditions.items()):
            prediction = optional_int(row.get("prediction"))
            detail_rows.append(
                {
                    **common,
                    "condition": condition,
                    "prediction": prediction,
                    "registered_numeric_valid": prediction in REGISTERED_COUNTS,
                    "donor_adoption": bool(prediction == donor),
                    "receiver_retention": bool(prediction == receiver),
                    "completion_text_raw": row.get("completion_text_raw"),
                    "generated_token_count": row.get("generated_token_count"),
                }
            )
        pair_rows.append(
            {
                **common,
                "self_prediction": self_prediction,
                "full_prediction": full_prediction,
                "full_registered_numeric_valid": bool(full_valid),
                "full_donor_adoption": full_adoption,
                "self_donor_adoption": self_adoption,
                "adoption_specificity": float(full_adoption) - float(self_adoption),
                "full_receiver_retention": bool(full_prediction == receiver),
                "full_changed_from_receiver": bool(
                    full_prediction is not None and full_prediction != receiver
                ),
                "strict_normalized_transport": transport,
            }
        )

    detail = pd.DataFrame(detail_rows)
    pair_frame = pd.DataFrame(pair_rows)
    seed = (
        pair_frame.groupby(["model_label", "layer", "seed"], as_index=False)
        .agg(
            pairs=("pair_id", "size"),
            full_donor_adoption=("full_donor_adoption", "mean"),
            adoption_specificity=("adoption_specificity", "mean"),
            registered_numeric_valid=("full_registered_numeric_valid", "mean"),
            receiver_retention=("full_receiver_retention", "mean"),
            changed_from_receiver=("full_changed_from_receiver", "mean"),
            strict_normalized_transport=("strict_normalized_transport", "mean"),
        )
    )
    layer_rows: list[dict[str, Any]] = []
    metrics = (
        "full_donor_adoption",
        "adoption_specificity",
        "registered_numeric_valid",
        "receiver_retention",
        "changed_from_receiver",
        "strict_normalized_transport",
    )
    for (model_label, layer), frame in seed.groupby(
        ["model_label", "layer"], sort=True
    ):
        row: dict[str, Any] = {
            "model_label": model_label,
            "layer": int(layer),
            "seed_clusters": int(frame["seed"].nunique()),
            "pairs": int(frame["pairs"].sum()),
        }
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            low, high = cluster_bootstrap(
                values, label=f"native-answer-layer:{model_label}:L{layer}:{metric}"
            )
            row[metric] = float(values.mean())
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        layer_rows.append(row)
    layers = pd.DataFrame(layer_rows).sort_values(["model_label", "layer"])

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "detail.csv"
    pair_path = output_dir / "pair_effects.csv"
    seed_path = output_dir / "seed_effects.csv"
    layer_path = output_dir / "layer_effects.csv"
    detail.to_csv(detail_path, index=False)
    pair_frame.to_csv(pair_path, index=False)
    seed.to_csv(seed_path, index=False)
    layers.to_csv(layer_path, index=False)

    onset: dict[str, int | None] = {}
    for model_label, frame in layers.groupby("model_label", sort=True):
        qualifying = frame.loc[
            (frame["full_donor_adoption"] >= 0.5)
            & (frame["full_donor_adoption_ci95_low"] > 0.0)
        ]
        onset[str(model_label)] = (
            None if qualifying.empty else int(qualifying.iloc[0]["layer"])
        )
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "trials": str(trials_path.resolve()),
        "trials_sha256": sha256(trials_path),
        "pairs": str(pairs_path.resolve()),
        "pairs_sha256": sha256(pairs_path),
        "conditions": ["self_patch", "full_donor_patch"],
        "pair_eligibility": (
            "strict one-to-one confirmation traces; donor and receiver clean "
            "final answers exact; every self patch must regenerate receiver gold"
        ),
        "behavioral_endpoint": (
            "actual deterministic greedy numeric continuation; unparsable or "
            "outside registered counts 1..10 remains in denominator as failure"
        ),
        "estimand": (
            "seed-equal mean probability that receiver output exactly adopts "
            "donor gold after a single post-block answer_query_v3 state replacement"
        ),
        "bootstrap_unit": "seed cluster",
        "bootstrap_repetitions": 10_000,
        "registered_pairs": len(pair_registry),
        "layers": frozen_layers,
        "completed_pair_layer_cells": len(pair_frame),
        "completed_trials": len(detail),
        "descriptive_onset_rule": (
            "first sampled layer with adoption >= 0.5 and seed-bootstrap CI lower > 0; "
            "not used to choose any tested layer"
        ),
        "descriptive_onset_layer": onset,
        "outputs": {
            "detail": str(detail_path.resolve()),
            "pair_effects": str(pair_path.resolve()),
            "seed_effects": str(seed_path.resolve()),
            "layer_effects": str(layer_path.resolve()),
        },
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.trials,
                args.pairs,
                args.output_dir,
                expected_layers=args.expected_layers,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
