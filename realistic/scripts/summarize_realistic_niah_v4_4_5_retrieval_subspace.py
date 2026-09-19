from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_RUNS = (
    ("Qwen_L21", "Qwen3-8B", 21),
    ("Qwen_L23", "Qwen3-8B", 23),
    ("Qwen_L24", "Qwen3-8B", 24),
    ("Qwen_L26", "Qwen3-8B", 26),
    ("Qwen_L27", "Qwen3-8B", 27),
    ("Gemma_L29", "Gemma4-E4B", 29),
    ("Gemma_L35", "Gemma4-E4B", 35),
)

PRIMARY_METRICS = (
    "natural_expected_error_specificity",
    "restoration_mediation_expected_error",
    "estimated_mediated_fraction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_audit(analysis_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    source_audits: list[dict[str, Any]] = []
    for label, expected_model, expected_layer in EXPECTED_RUNS:
        directory = analysis_root / label
        audit_path = directory / "analysis_audit.json"
        summary_path = directory / "effect_summary.csv"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise RuntimeError(f"{label} analysis audit is not PASS: {audit}")
        if not audit.get("norm_match_pass") or not audit.get("orthogonality_pass"):
            raise RuntimeError(f"{label} failed a matched-control audit")
        if int(audit.get("rows", -1)) != 400 or int(audit.get("paired_units", -1)) != 100:
            raise RuntimeError(f"{label} has unexpected audit counts: {audit}")

        frame = pd.read_csv(summary_path)
        if set(frame["population"]) != {"all", "clean_correct"}:
            raise RuntimeError(f"{label} has unexpected populations")
        if set(frame["model_label"]) != {expected_model}:
            raise RuntimeError(f"{label} has the wrong model label")
        if set(frame["retrieval_layer"].astype(int)) != {expected_layer}:
            raise RuntimeError(f"{label} has the wrong retrieval layer")
        if set(frame["source_patch_layer"].astype(int)) != {8}:
            raise RuntimeError(f"{label} has the wrong source patch layer")
        primary = frame[frame["population"].eq("all")]
        if len(primary) != 1 or int(primary.iloc[0]["rows"]) != 100:
            raise RuntimeError(f"{label} primary population is incomplete")
        for metric in PRIMARY_METRICS:
            for prefix in ("mean_", "median_"):
                column = prefix + metric
                if column not in frame:
                    raise RuntimeError(f"{label} is missing {column}")

        frame.insert(0, "run_label", label)
        frames.append(frame)
        source_audits.append(
            {
                "run_label": label,
                "model_label": expected_model,
                "retrieval_layer": expected_layer,
                "analysis_audit_sha256": sha256(audit_path),
                "effect_summary_sha256": sha256(summary_path),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined, source_audits


def plot_primary(primary: pd.DataFrame, output: Path) -> None:
    models = ("Qwen3-8B", "Gemma4-E4B")
    colors = {
        "natural": "#237f78",
        "mediation": "#7048e8",
        "fraction": "#c96712",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex="col")
    for column, model in enumerate(models):
        frame = primary[primary["model_label"].eq(model)].sort_values("retrieval_layer")
        layers = frame["retrieval_layer"].astype(int)
        axes[0, column].plot(
            layers,
            frame["mean_natural_expected_error_specificity"],
            marker="o",
            color=colors["natural"],
            label="Natural aligned-vs-control damage",
        )
        axes[0, column].plot(
            layers,
            frame["mean_restoration_mediation_expected_error"],
            marker="o",
            color=colors["mediation"],
            label="Restoration mediation damage",
        )
        axes[0, column].axhline(0, color="black", linewidth=0.9)
        axes[0, column].set_title(model)
        axes[0, column].set_ylabel("Mean expected-count error difference")
        axes[0, column].legend(frameon=False, fontsize=8)

        axes[1, column].plot(
            layers,
            frame["mean_estimated_mediated_fraction"],
            marker="o",
            color=colors["fraction"],
            label="Mean",
        )
        axes[1, column].plot(
            layers,
            frame["median_estimated_mediated_fraction"],
            marker="s",
            linestyle="--",
            color="#505050",
            label="Median",
        )
        axes[1, column].axhline(0, color="black", linewidth=0.9)
        axes[1, column].set_xlabel("Retrieval layer")
        axes[1, column].set_ylabel("Estimated mediated fraction")
        axes[1, column].legend(frameon=False, fontsize=8)
        axes[1, column].set_xticks(list(layers))
    figure.suptitle("Retrieval-subspace causal effects across frozen layers", y=1.01)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--combined-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root).resolve()
    combined_audit_path = Path(args.combined_audit).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    combined_audit = json.loads(combined_audit_path.read_text(encoding="utf-8"))
    if combined_audit.get("status") != "PASS" or int(combined_audit.get("total_rows", -1)) != 2800:
        raise RuntimeError(f"Combined retrieval-subspace audit failed: {combined_audit}")
    combined, source_audits = load_and_audit(analysis_root)
    primary = combined[combined["population"].eq("all")].copy()
    if len(combined) != 14 or len(primary) != 7:
        raise RuntimeError("Cross-layer table has unexpected row counts")

    combined.to_csv(output / "cross_layer_effect_summary.csv", index=False)
    primary.to_csv(output / "cross_layer_primary.csv", index=False)
    plot_primary(primary, output / "retrieval_subspace_cross_layer.png")
    audit = {
        "schema_version": "realistic_niah_v4_4_5_retrieval_subspace_cross_layer_v1",
        "status": "PASS",
        "combined_gpu_audit_sha256": sha256(combined_audit_path),
        "models": sorted(primary["model_label"].unique().tolist()),
        "model_layers": {
            model: sorted(
                primary.loc[primary["model_label"].eq(model), "retrieval_layer"]
                .astype(int)
                .tolist()
            )
            for model in sorted(primary["model_label"].unique())
        },
        "primary_rows": int(len(primary)),
        "all_summary_rows": int(len(combined)),
        "paired_units_per_layer": 100,
        "source_audits": source_audits,
        "interpretation_guardrails": [
            "Positive aligned-minus-orthogonal damage indicates direction-specific necessity or mediation.",
            "Estimated mediated fractions are unclipped ratios and are descriptive, not probabilities.",
            "Clean-correct rows are a labeled robustness subset, not the primary population.",
        ],
    }
    (output / "cross_layer_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
