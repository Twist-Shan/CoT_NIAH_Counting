#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "realistic_niah_v5_answer_execution_analysis_v1"


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


def bootstrap(values: np.ndarray, *, label: str) -> tuple[float, float]:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    draws = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def sign_flip(values: np.ndarray) -> tuple[float, str, int]:
    values = values[np.isfinite(values)]
    observed = abs(float(values.mean()))
    if len(values) <= 20:
        total = 1 << len(values)
        extreme = 0
        bits = np.arange(len(values), dtype=np.uint64)
        for start in range(0, total, 65_536):
            stop = min(total, start + 65_536)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = np.where(((masks >> bits) & 1) == 0, -1.0, 1.0)
            draws = np.abs((signs * values).mean(axis=1))
            extreme += int(np.count_nonzero(draws >= observed - 1e-15))
        return float(extreme / total), "exact_enumeration", total
    repetitions = 1_000_000
    seed = int.from_bytes(hashlib.sha256(values.tobytes()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(100):
        signs = rng.choice((-1.0, 1.0), size=(10_000, len(values)))
        draws = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(draws >= observed - 1e-15))
    return (
        float((extreme + 1) / (repetitions + 1)),
        "deterministic_monte_carlo",
        repetitions,
    )


def analyze(trials: Path, pairs_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(trials)
    pairs = read_jsonl(pairs_path)
    pair_registry = {str(row["pair_id"]): row for row in pairs}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), []).append(row)
    detail_rows = []
    for pair_id, frame in sorted(grouped.items()):
        if pair_id not in pair_registry:
            raise ValueError(f"Trial pair is absent from frozen registry: {pair_id}")
        pair = pair_registry[pair_id]
        if not bool(pair.get("receiver_exact_count")) or not bool(
            pair.get("donor_exact_count")
        ):
            raise ValueError(f"Non-correct answer execution pair: {pair_id}")
        conditions = {str(row["condition"]): row for row in frame}
        expected = {
            "self_patch",
            "full_donor_patch",
            "projected_donor_patch",
            "orthogonal_norm_matched",
        }
        if set(conditions) != expected:
            raise ValueError(
                f"Answer execution condition mismatch for {pair_id}: "
                f"{sorted(conditions)}"
            )
        receiver = int(pair["receiver_count"])
        donor = int(pair["donor_count"])
        gap = donor - receiver
        if gap == 0:
            raise ValueError(f"Zero-gap answer execution pair: {pair_id}")
        clean_prediction = optional_int(conditions["self_patch"].get("prediction"))
        if clean_prediction != receiver:
            raise ValueError(
                f"Correct-only self-patch regenerated {clean_prediction}, "
                f"expected receiver gold {receiver}: {pair_id}"
            )
        for condition, row in conditions.items():
            prediction = optional_int(row.get("prediction"))
            numeric_valid = prediction is not None
            strict_transport = (
                float((prediction - clean_prediction) / gap)
                if numeric_valid
                else 0.0
            )
            detail_rows.append(
                {
                    "pair_id": pair_id,
                    "model_label": str(row["model_label"]),
                    "seed": int(row["seed"]),
                    "receiver_count": receiver,
                    "donor_count": donor,
                    "signed_count_gap": gap,
                    "absolute_count_gap": abs(gap),
                    "pair_direction": str(pair["pair_direction"]),
                    "condition": condition,
                    "prediction": prediction,
                    "numeric_valid": numeric_valid,
                    "strict_normalized_transport": strict_transport,
                    "donor_adoption": bool(prediction == donor),
                    "receiver_retention": bool(prediction == receiver),
                    "completion_text_raw": row.get("completion_text_raw"),
                    "pair_eligibility": str(pair.get("pair_eligibility")),
                }
            )
    detail = pd.DataFrame(detail_rows)
    wide = detail.pivot(
        index=[
            "pair_id",
            "model_label",
            "seed",
            "receiver_count",
            "donor_count",
            "signed_count_gap",
            "absolute_count_gap",
            "pair_direction",
        ],
        columns="condition",
        values=["strict_normalized_transport", "donor_adoption", "numeric_valid"],
    )
    wide.columns = [f"{field}__{condition}" for field, condition in wide.columns]
    wide = wide.reset_index()
    wide["projected_minus_orthogonal_transport"] = (
        wide["strict_normalized_transport__projected_donor_patch"]
        - wide["strict_normalized_transport__orthogonal_norm_matched"]
    )
    wide["full_minus_orthogonal_transport"] = (
        wide["strict_normalized_transport__full_donor_patch"]
        - wide["strict_normalized_transport__orthogonal_norm_matched"]
    )
    wide["projected_minus_orthogonal_adoption"] = (
        wide["donor_adoption__projected_donor_patch"].astype(float)
        - wide["donor_adoption__orthogonal_norm_matched"].astype(float)
    )
    wide["full_minus_orthogonal_adoption"] = (
        wide["donor_adoption__full_donor_patch"].astype(float)
        - wide["donor_adoption__orthogonal_norm_matched"].astype(float)
    )
    seed = (
        wide.groupby(["model_label", "seed"], as_index=False)
        .agg(
            pairs=("pair_id", "size"),
            full_transport=("strict_normalized_transport__full_donor_patch", "mean"),
            projected_transport=("strict_normalized_transport__projected_donor_patch", "mean"),
            orthogonal_transport=("strict_normalized_transport__orthogonal_norm_matched", "mean"),
            full_specificity=("full_minus_orthogonal_transport", "mean"),
            projected_specificity=("projected_minus_orthogonal_transport", "mean"),
            full_donor_adoption=("donor_adoption__full_donor_patch", "mean"),
            projected_donor_adoption=("donor_adoption__projected_donor_patch", "mean"),
            orthogonal_donor_adoption=("donor_adoption__orthogonal_norm_matched", "mean"),
            full_adoption_specificity=("full_minus_orthogonal_adoption", "mean"),
            projected_adoption_specificity=("projected_minus_orthogonal_adoption", "mean"),
        )
    )
    gap_seed = (
        wide.groupby(["model_label", "signed_count_gap", "seed"], as_index=False)
        .agg(
            pairs=("pair_id", "size"),
            full_transport=("strict_normalized_transport__full_donor_patch", "mean"),
            projected_transport=("strict_normalized_transport__projected_donor_patch", "mean"),
            orthogonal_transport=("strict_normalized_transport__orthogonal_norm_matched", "mean"),
            full_specificity=("full_minus_orthogonal_transport", "mean"),
            projected_specificity=("projected_minus_orthogonal_transport", "mean"),
            full_donor_adoption=("donor_adoption__full_donor_patch", "mean"),
            projected_donor_adoption=("donor_adoption__projected_donor_patch", "mean"),
            orthogonal_donor_adoption=("donor_adoption__orthogonal_norm_matched", "mean"),
            full_adoption_specificity=("full_minus_orthogonal_adoption", "mean"),
            projected_adoption_specificity=("projected_minus_orthogonal_adoption", "mean"),
        )
    )
    statistics_rows = []
    analysis_frames = [
        ("overall", None, model, frame)
        for model, frame in seed.groupby("model_label", sort=True)
    ]
    analysis_frames.extend(
        ("signed_gap", int(gap), model, frame)
        for (model, gap), frame in gap_seed.groupby(
            ["model_label", "signed_count_gap"], sort=True
        )
    )
    for scope, gap, model, frame in analysis_frames:
        for metric in (
            "full_transport",
            "projected_transport",
            "full_specificity",
            "projected_specificity",
            "full_donor_adoption",
            "projected_donor_adoption",
            "full_adoption_specificity",
            "projected_adoption_specificity",
        ):
            values = frame[metric].to_numpy(dtype=float)
            low, high = bootstrap(
                values,
                label=f"v5-answer-execution:{scope}:{gap}:{model}:{metric}",
            )
            p_value, method, assignments = sign_flip(values)
            statistics_rows.append(
                {
                    "model_label": model,
                    "scope": scope,
                    "signed_count_gap": gap,
                    "metric": metric,
                    "seed_clusters": len(values),
                    "effect": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_seed_fraction": float(np.mean(values > 0)),
                    "sign_flip_p": p_value,
                    "sign_flip_method": method,
                    "sign_flip_assignments": assignments,
                    "bootstrap_repetitions": 10_000,
                }
            )
    statistics = pd.DataFrame(statistics_rows)
    statistics["holm_p_within_model"] = np.nan
    for _model, indices in statistics.groupby("model_label").groups.items():
        ordered = sorted(
            indices, key=lambda index: statistics.loc[index, "sign_flip_p"]
        )
        running = 0.0
        for rank, index in enumerate(ordered):
            running = max(
                running,
                min(
                    1.0,
                    (len(ordered) - rank)
                    * float(statistics.loc[index, "sign_flip_p"]),
                ),
            )
            statistics.loc[index, "holm_p_within_model"] = running
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "detail.csv"
    pair_path = output_dir / "paired_effects.csv"
    seed_path = output_dir / "seed_effects.csv"
    gap_seed_path = output_dir / "signed_gap_seed_effects.csv"
    stats_path = output_dir / "statistics.csv"
    detail.to_csv(detail_path, index=False)
    wide.to_csv(pair_path, index=False)
    seed.to_csv(seed_path, index=False)
    gap_seed.to_csv(gap_seed_path, index=False)
    statistics.to_csv(stats_path, index=False)
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "trials": str(trials.resolve()),
        "trials_sha256": sha256(trials),
        "pairs": str(pairs_path.resolve()),
        "pairs_sha256": sha256(pairs_path),
        "pair_eligibility": (
            "receiver and donor original final answers exact; self-patch "
            "must regenerate receiver gold"
        ),
        "behavioral_endpoint": "strict greedy generated numeric answer",
        "count_gap_policy": (
            "signed_count_gap=donor_count-receiver_count; effects are reported "
            "both overall and separately for every frozen signed gap"
        ),
        "strict_invalid_policy": "invalid patched numeric generation has zero transport and zero adoption",
        "primary_specificity_controls": (
            "full/projected donor effects are each contrasted with the "
            "norm-matched orthogonal control; p values use seed-level sign flips"
        ),
        "registered_pairs": len(pairs),
        "completed_pairs": len(wide),
        "conditions_per_pair": 4,
        "outputs": {
            "detail": str(detail_path.resolve()),
            "paired": str(pair_path.resolve()),
            "seed_effects": str(seed_path.resolve()),
            "signed_gap_seed_effects": str(gap_seed_path.resolve()),
            "statistics": str(stats_path.resolve()),
        },
    }
    if len(wide) != len(pairs):
        raise ValueError("Answer execution trials do not cover every frozen pair")
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trials, args.pairs, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
