#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA = "realistic_niah_v5_marker_adjacent_interchange_analysis_v1"


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


def analyze(
    paths: list[Path],
    output_dir: Path,
    *,
    require_confirmation: bool = True,
) -> dict[str, Any]:
    rows = []
    inputs = []
    for path in paths:
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
        rows.extend(read_jsonl(path))
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("experiment_id") != "marker_adjacent_count_interchange_patch_v1":
            continue
        key = (str(row["pair_id"]), str(row["query_variant"]), int(row["layer"]))
        groups.setdefault(key, []).append(row)
    expected = {
        "full_self_patch",
        "counterfactual_self_patch",
        "counterfactual_from_full_patch",
        "full_from_counterfactual_patch",
        "counterfactual_from_orthogonal_patch",
        "full_from_orthogonal_patch",
    }
    detail_rows = []
    for (pair_id, variant, layer), frame in sorted(groups.items()):
        conditions = {str(row["condition"]): row for row in frame}
        if set(conditions) != expected or len(frame) != 6:
            raise ValueError(f"Incomplete interchange group: {pair_id}/{variant}/L{layer}")
        if not all(
            bool(row.get("full_exact_count"))
            and bool(row.get("counterfactual_exact_count"))
            for row in frame
        ):
            raise ValueError(f"Interchange pair is not correct-only: {pair_id}")
        if not all(
            row.get("prompt_length_match_audit") == "PASS_EQUAL_TOKEN_LENGTH"
            and row.get("trace_prefix_identity_audit")
            == "PASS_SAME_TEACHER_FORCED_PREFIX"
            and row.get("semantic_query_alignment_audit")
            == "PASS_SAME_QUERY_POSITION"
            and row.get("prompt_counterfactual_locality_audit")
            == "PASS_CHANGED_TOKENS_INSIDE_ADDED_RECORD_ONLY"
            for row in frame
        ):
            raise ValueError(f"Interchange construction audit failed: {pair_id}")
        control_audits = {
            str(row.get("orthogonal_control_audit")) for row in frame
        }
        allowed_control_audits = {
            "PASS_NORM_MATCHED_ORTHOGONAL_CONTROL",
            "PASS_DEGENERATE_ZERO_NORM_IDENTITY_CONTROL",
        }
        if not control_audits <= allowed_control_audits:
            raise ValueError(
                f"Interchange orthogonal audit failed: {pair_id}: "
                f"{sorted(control_audits)}"
            )
        exact = {
            condition: float(bool(row["target_needle_exact"]))
            for condition, row in conditions.items()
        }
        restoration = (
            exact["counterfactual_from_full_patch"]
            - exact["counterfactual_self_patch"]
        )
        suppression = (
            exact["full_self_patch"]
            - exact["full_from_counterfactual_patch"]
        )
        restoration_specificity = (
            exact["counterfactual_from_full_patch"]
            - exact["counterfactual_from_orthogonal_patch"]
        )
        suppression_specificity = (
            exact["full_from_orthogonal_patch"]
            - exact["full_from_counterfactual_patch"]
        )
        first = frame[0]
        detail_rows.append(
            {
                "pair_id": pair_id,
                "model_label": str(first["model_label"]),
                "split": str(first["split"]),
                "seed": int(first["seed"]),
                "full_count": int(first["full_count"]),
                "counterfactual_count": int(first["counterfactual_count"]),
                "occurrence": int(first["occurrence"]),
                "query_variant": variant,
                "query_alias_key": str(first["query_alias_key"]),
                "layer": int(layer),
                "donor_receiver_delta_norm": float(
                    first["donor_receiver_delta_norm"]
                ),
                "zero_delta_state": bool(
                    float(first["donor_receiver_delta_norm"]) == 0.0
                ),
                **{f"exact__{key}": value for key, value in exact.items()},
                "restoration": restoration,
                "suppression": suppression,
                "bidirectional_transport": 0.5 * (restoration + suppression),
                "restoration_specificity": restoration_specificity,
                "suppression_specificity": suppression_specificity,
                "bidirectional_specificity": 0.5
                * (restoration_specificity + suppression_specificity),
            }
        )
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise ValueError("No complete adjacent-count interchange groups")
    discovery = detail.loc[detail["split"].eq("discovery")].copy()
    if discovery.empty:
        raise ValueError("No discovery interchange trials")
    discovery_seed = (
        discovery.groupby(
            ["model_label", "query_variant", "layer", "seed"], as_index=False
        )
        .agg(
            pairs=("pair_id", "size"),
            restoration=("restoration", "mean"),
            suppression=("suppression", "mean"),
            bidirectional_transport=("bidirectional_transport", "mean"),
            bidirectional_specificity=("bidirectional_specificity", "mean"),
        )
    )
    selection_rows = []
    for (model, variant), frame in discovery_seed.groupby(
        ["model_label", "query_variant"], sort=True
    ):
        scored = (
            frame.groupby("layer", as_index=False)
            .agg(
                discovery_bidirectional_transport=(
                    "bidirectional_transport",
                    "mean",
                ),
                discovery_bidirectional_specificity=(
                    "bidirectional_specificity",
                    "mean",
                ),
                discovery_seed_clusters=("seed", "size"),
            )
            .sort_values(
                [
                    "discovery_bidirectional_specificity",
                    "discovery_bidirectional_transport",
                    "layer",
                ],
                ascending=[False, False, True],
            )
        )
        best = scored.iloc[0]
        selection_rows.append(
            {
                "model_label": model,
                "query_variant": variant,
                "selected_layer": int(best["layer"]),
                "selection_split": "discovery",
                "confirmation_used_for_selection": False,
                "discovery_bidirectional_transport": float(
                    best["discovery_bidirectional_transport"]
                ),
                "discovery_bidirectional_specificity": float(
                    best["discovery_bidirectional_specificity"]
                ),
                "discovery_seed_clusters": int(best["discovery_seed_clusters"]),
            }
        )
    selection = pd.DataFrame(selection_rows)
    confirmation_frames = []
    for row in selection.itertuples(index=False):
        active = detail.loc[
            detail["model_label"].eq(row.model_label)
            & detail["query_variant"].eq(row.query_variant)
            & detail["layer"].eq(row.selected_layer)
            & detail["split"].eq("confirmation")
        ].copy()
        if active.empty and require_confirmation:
            raise ValueError(
                f"Missing confirmation for {row.model_label}/"
                f"{row.query_variant}/L{row.selected_layer}"
            )
        if not active.empty:
            confirmation_frames.append(active)
    confirmation = (
        pd.concat(confirmation_frames, ignore_index=True)
        if confirmation_frames
        else pd.DataFrame(columns=detail.columns)
    )
    seed_effects = (
        confirmation.groupby(
            ["model_label", "query_variant", "layer", "seed"], as_index=False
        )
        .agg(
            pairs=("pair_id", "size"),
            restoration=("restoration", "mean"),
            suppression=("suppression", "mean"),
            bidirectional_transport=("bidirectional_transport", "mean"),
            restoration_specificity=("restoration_specificity", "mean"),
            suppression_specificity=("suppression_specificity", "mean"),
            bidirectional_specificity=("bidirectional_specificity", "mean"),
        )
        if not confirmation.empty
        else pd.DataFrame()
    )
    stats_rows = []
    if not seed_effects.empty:
        for (model, variant, layer), frame in seed_effects.groupby(
            ["model_label", "query_variant", "layer"], sort=True
        ):
            for metric in (
                "restoration",
                "suppression",
                "bidirectional_transport",
                "restoration_specificity",
                "suppression_specificity",
                "bidirectional_specificity",
            ):
                values = frame[metric].to_numpy(dtype=float)
                low, high = bootstrap(
                    values, label=f"marker:{model}:{variant}:L{layer}:{metric}"
                )
                p_value, method, assignments = sign_flip(values)
                stats_rows.append(
                    {
                        "model_label": model,
                        "query_variant": variant,
                        "selected_layer": int(layer),
                        "metric": metric,
                        "primary_endpoint": metric
                        == "bidirectional_specificity",
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
    if not statistics.empty:
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
    outputs = {
        "detail": output_dir / "paired_interchange_effects.csv",
        "discovery_seed": output_dir / "discovery_seed_effects.csv",
        "selection": output_dir / "discovery_frozen_layer_selection.csv",
        "confirmation": output_dir / "confirmation_selected_layer_effects.csv",
        "seed_effects": output_dir / "confirmation_seed_effects.csv",
        "statistics": output_dir / "confirmation_statistics.csv",
    }
    detail.to_csv(outputs["detail"], index=False)
    discovery_seed.to_csv(outputs["discovery_seed"], index=False)
    selection.to_csv(outputs["selection"], index=False)
    confirmation.to_csv(outputs["confirmation"], index=False)
    seed_effects.to_csv(outputs["seed_effects"], index=False)
    statistics.to_csv(outputs["statistics"], index=False)
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "inputs": inputs,
        "construction_reference": "nonthinking_adjacent_count_pairing",
        "selection_split": "discovery",
        "confirmation_used_for_selection": False,
        "selection_metric": "seed-mean bidirectional specificity vs orthogonal",
        "behavioral_endpoint": "actual greedy added-kth-needle exact sequence",
        "correct_only": True,
        "equal_prompt_token_length_required": True,
        "identical_teacher_forced_trace_prefix_required": True,
        "groups": len(detail),
        "zero_delta_groups": int(detail["zero_delta_state"].sum()),
        "selected_model_variants": len(selection),
        "confirmation_complete": bool(not confirmation.empty),
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.trials,
                args.output_dir,
                require_confirmation=not args.selection_only,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
