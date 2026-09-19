from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return pd.DataFrame(rows)


def paired_effects(detail: pd.DataFrame) -> pd.DataFrame:
    index = [
        "model_label",
        "seed",
        "gold_count",
        "source_patch_layer",
        "retrieval_layer",
    ]
    metrics = ["expected_count", "strict_absolute_error", "strict_correct"]
    pivot = detail.pivot_table(
        index=index, columns="condition", values=metrics, aggfunc="first"
    )
    required = {
        (metric, condition)
        for metric in metrics
        for condition in (
            "clean_aligned_block",
            "clean_orthogonal_block",
            "restored_aligned_block",
            "restored_orthogonal_block",
        )
    }
    missing = sorted(required - set(pivot.columns))
    if missing:
        raise RuntimeError(f"Retrieval subspace result grid is incomplete: {missing}")
    rows: list[dict[str, Any]] = []
    for values, row in pivot.iterrows():
        model, seed, gold, source_layer, retrieval_layer = values
        gold = int(gold)
        clean_aligned_error = abs(float(row[("expected_count", "clean_aligned_block")]) - gold)
        clean_orth_error = abs(float(row[("expected_count", "clean_orthogonal_block")]) - gold)
        restored_aligned_error = abs(
            float(row[("expected_count", "restored_aligned_block")]) - gold
        )
        restored_orth_error = abs(
            float(row[("expected_count", "restored_orthogonal_block")]) - gold
        )
        rows.append(
            {
                "model_label": model,
                "seed": int(seed),
                "gold_count": gold,
                "source_patch_layer": int(source_layer),
                "retrieval_layer": int(retrieval_layer),
                "natural_expected_error_specificity": (
                    clean_aligned_error - clean_orth_error
                ),
                "natural_strict_error_specificity": float(
                    row[("strict_absolute_error", "clean_aligned_block")]
                    - row[("strict_absolute_error", "clean_orthogonal_block")]
                ),
                "natural_accuracy_damage_specificity": float(
                    row[("strict_correct", "clean_orthogonal_block")]
                    - row[("strict_correct", "clean_aligned_block")]
                ),
                "restoration_mediation_expected_error": (
                    restored_aligned_error - restored_orth_error
                ),
                "restoration_mediation_strict_error": float(
                    row[("strict_absolute_error", "restored_aligned_block")]
                    - row[("strict_absolute_error", "restored_orthogonal_block")]
                ),
                "restoration_mediation_accuracy_damage": float(
                    row[("strict_correct", "restored_orthogonal_block")]
                    - row[("strict_correct", "restored_aligned_block")]
                ),
                "clean_aligned_expected_count": float(
                    row[("expected_count", "clean_aligned_block")]
                ),
                "clean_orthogonal_expected_count": float(
                    row[("expected_count", "clean_orthogonal_block")]
                ),
                "restored_aligned_expected_count": float(
                    row[("expected_count", "restored_aligned_block")]
                ),
                "restored_orthogonal_expected_count": float(
                    row[("expected_count", "restored_orthogonal_block")]
                ),
            }
        )
    return pd.DataFrame(rows)


def attach_restoration_fraction(
    effects: pd.DataFrame, restoration_root: Path
) -> pd.DataFrame:
    frames = []
    for model in effects["model_label"].unique():
        path = restoration_root / str(model) / "detail.jsonl"
        if path.exists():
            frames.append(read_jsonl(path))
    if not frames:
        effects["unblocked_restoration_error_repair"] = np.nan
        effects["estimated_mediated_fraction"] = np.nan
        return effects
    detail = pd.concat(frames, ignore_index=True)
    baseline = detail[
        detail["condition"].eq("needle_corrupt")
        & detail["patch_layer"].astype(int).eq(-1)
    ][["model_label", "seed", "gold_count", "expected_count"]].rename(
        columns={"expected_count": "corrupt_expected_count"}
    )
    clean = detail[
        detail["condition"].eq("clean")
        & detail["patch_layer"].astype(int).eq(-1)
    ][["model_label", "seed", "gold_count", "strict_correct"]].rename(
        columns={"strict_correct": "clean_strict_correct"}
    )
    restored = detail[
        detail["patch_kind"].eq("needle_full")
        & detail["patch_layer"].astype(int).ge(0)
    ][
        ["model_label", "seed", "gold_count", "patch_layer", "expected_count"]
    ].rename(
        columns={
            "patch_layer": "source_patch_layer",
            "expected_count": "unblocked_restored_expected_count",
        }
    )
    denominator = restored.merge(
        baseline,
        on=["model_label", "seed", "gold_count"],
        how="left",
        validate="many_to_one",
    )
    denominator = denominator.merge(
        clean,
        on=["model_label", "seed", "gold_count"],
        how="left",
        validate="many_to_one",
    )
    gold = denominator["gold_count"].astype(float)
    denominator["unblocked_restoration_error_repair"] = (
        (denominator["corrupt_expected_count"].astype(float) - gold).abs()
        - (denominator["unblocked_restored_expected_count"].astype(float) - gold).abs()
    )
    merged = effects.merge(
        denominator[
            [
                "model_label",
                "seed",
                "gold_count",
                "source_patch_layer",
                "unblocked_restoration_error_repair",
                "clean_strict_correct",
            ]
        ],
        on=["model_label", "seed", "gold_count", "source_patch_layer"],
        how="left",
        validate="one_to_one",
    )
    denominator_value = merged["unblocked_restoration_error_repair"].astype(float)
    merged["estimated_mediated_fraction"] = np.where(
        denominator_value.abs() > 1e-8,
        merged["restoration_mediation_expected_error"] / denominator_value,
        np.nan,
    )
    return merged


def summarize(effects: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "natural_expected_error_specificity",
        "natural_strict_error_specificity",
        "natural_accuracy_damage_specificity",
        "restoration_mediation_expected_error",
        "restoration_mediation_strict_error",
        "restoration_mediation_accuracy_damage",
        "unblocked_restoration_error_repair",
        "estimated_mediated_fraction",
    ]
    rows: list[dict[str, Any]] = []
    populations = [("all", effects)]
    if "clean_strict_correct" in effects:
        populations.append(
            ("clean_correct", effects[effects["clean_strict_correct"].fillna(False)])
        )
    for population, frame in populations:
        for keys, group in frame.groupby(
            ["model_label", "source_patch_layer", "retrieval_layer"], sort=True
        ):
            model, source, retrieval = keys
            row: dict[str, Any] = {
                "model_label": model,
                "population": population,
                "source_patch_layer": int(source),
                "retrieval_layer": int(retrieval),
                "rows": int(len(group)),
                "seeds": int(group["seed"].nunique()),
            }
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                row[f"mean_{metric}"] = float(values.mean()) if len(values) else math.nan
                row[f"median_{metric}"] = (
                    float(np.median(values)) if len(values) else math.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_effects(effects: pd.DataFrame, output: Path) -> None:
    models = list(effects["model_label"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    width = 0.34
    positions = np.arange(len(models))
    natural = [
        effects[effects["model_label"].eq(model)][
            "natural_expected_error_specificity"
        ].mean()
        for model in models
    ]
    mediation = [
        effects[effects["model_label"].eq(model)][
            "restoration_mediation_expected_error"
        ].mean()
        for model in models
    ]
    axes[0].bar(positions - width / 2, natural, width, label="Natural necessity")
    axes[0].bar(positions + width / 2, mediation, width, label="Restoration mediation")
    axes[0].set_xticks(positions, labels=models)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Aligned-minus-orthogonal expected-error damage")
    axes[0].legend(frameon=False)
    axes[0].set_title("Direction-specific causal effects")
    for index, model in enumerate(models):
        frame = effects[effects["model_label"].eq(model)]
        axes[1].scatter(
            frame["unblocked_restoration_error_repair"],
            frame["restoration_mediation_expected_error"],
            s=24,
            alpha=0.65,
            label=model,
        )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Unblocked full-span error repair")
    axes[1].set_ylabel("Repair lost under retrieval-subspace block")
    axes[1].legend(frameon=False)
    axes[1].set_title("Mediation versus available repair")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--restoration-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    detail = pd.concat(
        [read_jsonl(root / model / "detail.jsonl") for model in args.models],
        ignore_index=True,
    )
    if not detail["norm_match_max_abs_delta"].astype(float).le(1e-4).all():
        raise RuntimeError("An orthogonal control failed realized-norm matching")
    orth = detail[detail["block_mode"].eq("orthogonal")]
    if not orth["orthogonality_max_abs_dot"].astype(float).le(1e-4).all():
        raise RuntimeError("An orthogonal control overlaps the retrieval basis")
    effects = paired_effects(detail)
    effects = attach_restoration_fraction(effects, Path(args.restoration_root))
    summary = summarize(effects)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "raw_detail.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)
    summary.to_csv(output / "effect_summary.csv", index=False)
    plot_effects(effects, output / "retrieval_subspace_causal_effects.png")
    audit = {
        "schema_version": "realistic_niah_v4_4_5_retrieval_subspace_analysis_v1",
        "models": list(args.models),
        "rows": int(len(detail)),
        "paired_units": int(len(effects)),
        "norm_match_pass": True,
        "orthogonality_pass": True,
        "definitions": {
            "natural_expected_error_specificity": (
                "|E_clean+aligned-block-gold| minus |E_clean+orth-block-gold|"
            ),
            "restoration_mediation_expected_error": (
                "|E_restored+aligned-block-gold| minus "
                "|E_restored+orth-block-gold|"
            ),
            "estimated_mediated_fraction": (
                "restoration mediation expected-error damage divided by "
                "unblocked full-span expected-error repair; left unclipped"
            ),
        },
        "status": "PASS",
    }
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
