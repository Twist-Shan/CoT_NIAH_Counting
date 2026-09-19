#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA = "realistic_niah_v5_next_needle_ablation_analysis_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sign_flip(values: Iterable[float]) -> tuple[float, str, int]:
    vector = np.asarray(list(values), dtype=float)
    vector = vector[np.isfinite(vector)]
    observed = abs(float(vector.mean()))
    if len(vector) <= 20:
        total = 1 << len(vector)
        extreme = 0
        bits = np.arange(len(vector), dtype=np.uint64)
        for start in range(0, total, 65_536):
            stop = min(total, start + 65_536)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = np.where(((masks >> bits) & 1) == 0, -1.0, 1.0)
            draws = np.abs((signs * vector).mean(axis=1))
            extreme += int(np.count_nonzero(draws >= observed - 1e-15))
        return float(extreme / total), "exact_enumeration", total
    repetitions = 1_000_000
    seed = int.from_bytes(hashlib.sha256(vector.tobytes()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(100):
        signs = rng.choice((-1.0, 1.0), size=(10_000, len(vector)))
        draws = np.abs((signs * vector).mean(axis=1))
        extreme += int(np.count_nonzero(draws >= observed - 1e-15))
    return (
        float((extreme + 1) / (repetitions + 1)),
        "deterministic_monte_carlo",
        repetitions,
    )


def bootstrap(values: np.ndarray, *, label: str) -> tuple[float, float]:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    draws = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def condition_kind(condition: str) -> str:
    if condition == "clean":
        return "clean"
    if "layer_matched_random" in condition:
        return "random"
    if condition.endswith("_ranked"):
        return "ranked"
    raise ValueError(f"Unknown next-needle condition: {condition}")


def analyze(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    inputs = []
    rows = []
    for path in paths:
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
        rows.extend(read_jsonl(path))
    rows = [
        row
        for row in rows
        if row.get("status", "ok") == "ok"
        and row.get("behavioral_endpoint")
        == "actual_greedy_next_needle_token_sequence"
    ]
    clean: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    interventions: dict[
        tuple[str, str, str, str, int, int], list[dict[str, Any]]
    ] = {}
    for row in rows:
        key = (
            str(row["model_label"]),
            str(row["query_variant"]),
            str(row.get("response_type", "all")),
            str(row.get("request_id", row.get("stimulus_id"))),
            int(row["occurrence"]),
        )
        kind = condition_kind(str(row["condition"]))
        if kind == "clean":
            if key in clean and bool(clean[key]["next_needle_exact"]) != bool(
                row["next_needle_exact"]
            ):
                raise ValueError(f"Conflicting clean next-needle row: {key}")
            clean[key] = row
        else:
            interventions.setdefault((*key, int(row["bank_size"])), []).append(row)
    detail_rows = []
    unmatched = []
    for key, frame in sorted(interventions.items()):
        base_key = key[:5]
        bank_size = key[5]
        ranked = [row for row in frame if condition_kind(str(row["condition"])) == "ranked"]
        random = [row for row in frame if condition_kind(str(row["condition"])) == "random"]
        if len(ranked) != 1:
            raise ValueError(f"Expected one ranked row: {key}")
        if base_key not in clean:
            raise ValueError(f"Missing clean next-needle row: {base_key}")
        if len(random) != 3:
            unmatched.append(
                {
                    "model_label": key[0],
                    "query_variant": key[1],
                    "response_type": key[2],
                    "request_id": key[3],
                    "occurrence": key[4],
                    "bank_size": bank_size,
                    "controls": len(random),
                }
            )
            continue
        baseline = clean[base_key]
        treatment = ranked[0]
        clean_exact = float(bool(baseline["next_needle_exact"]))
        ranked_exact = float(bool(treatment["next_needle_exact"]))
        random_exact = [float(bool(row["next_needle_exact"])) for row in random]
        ranked_damage = clean_exact - ranked_exact
        random_damage = [clean_exact - value for value in random_exact]
        detail_rows.append(
            {
                "model_label": key[0],
                "query_variant": key[1],
                "response_type": key[2],
                "request_id": key[3],
                "occurrence": key[4],
                "bank_size": bank_size,
                "seed": int(treatment["seed"]),
                "split": str(treatment["split"]),
                "gold_count": int(treatment["gold_count"]),
                "clean_exact": clean_exact,
                "ranked_exact": ranked_exact,
                "random_exact_mean": float(np.mean(random_exact)),
                "ranked_damage": ranked_damage,
                "random_damage_mean": float(np.mean(random_damage)),
                "all_accuracy_damage_specificity": (
                    ranked_damage - float(np.mean(random_damage))
                ),
                "clean_correct_to_wrong_specificity": (
                    ranked_damage - float(np.mean(random_damage))
                    if bool(baseline["next_needle_exact"])
                    else np.nan
                ),
                "ranked_token_accuracy": float(treatment["next_needle_token_accuracy"]),
                "random_token_accuracy_mean": float(
                    np.mean([row["next_needle_token_accuracy"] for row in random])
                ),
                "token_accuracy_specificity": float(
                    np.mean([row["next_needle_token_accuracy"] for row in random])
                    - float(treatment["next_needle_token_accuracy"])
                ),
            }
        )
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise ValueError("No next-needle ranked rows had three exact controls")
    seed = (
        detail.groupby(
            [
                "model_label",
                "query_variant",
                "response_type",
                "bank_size",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            occurrences=("request_id", "size"),
            clean_accuracy=("clean_exact", "mean"),
            ranked_accuracy=("ranked_exact", "mean"),
            random_accuracy=("random_exact_mean", "mean"),
            all_accuracy_damage_specificity=(
                "all_accuracy_damage_specificity", "mean"
            ),
            clean_correct_to_wrong_specificity=(
                "clean_correct_to_wrong_specificity", "mean"
            ),
            token_accuracy_specificity=("token_accuracy_specificity", "mean"),
        )
    )
    # The primary position test pools response formats while retaining the
    # equal-weight seed as the independent unit.  Per-format rows remain in the
    # table as a heterogeneity/robustness audit rather than defining separate
    # head banks.
    pooled_seed = (
        seed.groupby(
            ["model_label", "query_variant", "bank_size", "seed"],
            as_index=False,
        )
        .agg(
            occurrences=("occurrences", "sum"),
            clean_accuracy=("clean_accuracy", "mean"),
            ranked_accuracy=("ranked_accuracy", "mean"),
            random_accuracy=("random_accuracy", "mean"),
            all_accuracy_damage_specificity=(
                "all_accuracy_damage_specificity", "mean"
            ),
            clean_correct_to_wrong_specificity=(
                "clean_correct_to_wrong_specificity", "mean"
            ),
            token_accuracy_specificity=("token_accuracy_specificity", "mean"),
        )
    )
    pooled_seed["response_type"] = "pooled_cross_response_types"
    seed = pd.concat([seed, pooled_seed], ignore_index=True, sort=False)
    stats_rows = []
    for (model, variant, response_type, bank_size), frame in seed.groupby(
        ["model_label", "query_variant", "response_type", "bank_size"],
        sort=True,
    ):
        for metric in (
            "all_accuracy_damage_specificity",
            "clean_correct_to_wrong_specificity",
            "token_accuracy_specificity",
        ):
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(
                dtype=float
            )
            if not len(values):
                continue
            low, high = bootstrap(
                values,
                label=(
                    f"next-needle:{model}:{variant}:{response_type}:"
                    f"K{bank_size}:{metric}"
                ),
            )
            p_value, method, assignments = sign_flip(values)
            stats_rows.append(
                {
                    "model_label": model,
                    "query_variant": variant,
                    "response_type": response_type,
                    "bank_size": int(bank_size),
                    "metric": metric,
                    "primary_endpoint": metric
                    in {
                        "all_accuracy_damage_specificity",
                        "clean_correct_to_wrong_specificity",
                    },
                    "seed_clusters": len(values),
                    "effect": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "sign_flip_p": p_value,
                    "sign_flip_method": method,
                    "sign_flip_assignments": assignments,
                }
            )
    statistics = pd.DataFrame(stats_rows)
    statistics["holm_p_within_model"] = np.nan
    for _model, indices in statistics.groupby("model_label").groups.items():
        ordered = sorted(indices, key=lambda index: statistics.loc[index, "sign_flip_p"])
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
    statistics["holm_p_pooled_primary_family"] = np.nan
    pooled_family = statistics.loc[
        statistics["response_type"].eq("pooled_cross_response_types")
        & statistics["primary_endpoint"].astype(bool)
    ]
    for _model, indices in pooled_family.groupby("model_label").groups.items():
        ordered = sorted(indices, key=lambda index: statistics.loc[index, "sign_flip_p"])
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
            statistics.loc[index, "holm_p_pooled_primary_family"] = running
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "paired_occurrence_effects.csv"
    seed_path = output_dir / "seed_effects.csv"
    stats_path = output_dir / "statistics.csv"
    unmatched_path = output_dir / "unmatched_controls.csv"
    detail.to_csv(detail_path, index=False)
    seed.to_csv(seed_path, index=False)
    statistics.to_csv(stats_path, index=False)
    pd.DataFrame(unmatched).to_csv(unmatched_path, index=False)
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "inputs": inputs,
        "behavioral_endpoint": "actual greedy immediate next-needle exact sequence",
        "final_count_evaluated": False,
        "comparison": (
            "(clean-ranked failure) minus mean of three disjoint exact "
            "layer-matched random-control failures"
        ),
        "reported_exact_endpoints": [
            "all_accuracy_damage_specificity",
            "clean_correct_to_wrong_specificity",
        ],
        "independent_unit": "equal-weight seed cluster",
        "primary_cross_response_type_inference": (
            "response types are equal-weighted within seed; Holm family spans "
            "all registered K, supplied query positions, and both exact primary "
            "endpoints within model"
        ),
        "paired_occurrences": len(detail),
        "unmatched": len(unmatched),
        "outputs": {
            "detail": str(detail_path.resolve()),
            "seed_effects": str(seed_path.resolve()),
            "statistics": str(stats_path.resolve()),
            "unmatched": str(unmatched_path.resolve()),
        },
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trials, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
