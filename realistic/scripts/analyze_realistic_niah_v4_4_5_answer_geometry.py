from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from scripts.analyze_realistic_niah_v4_4_5_retrieval_geometry import (
        geometry_metrics,
        project_isometric,
        read_jsonl,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    from analyze_realistic_niah_v4_4_5_retrieval_geometry import (
        geometry_metrics,
        project_isometric,
        read_jsonl,
    )


def load_answer_states(run_root: Path, models: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        detail_path = run_root / model / "detail.jsonl"
        if not detail_path.exists():
            raise FileNotFoundError(detail_path)
        for row in read_jsonl(detail_path):
            if row["condition"] != "clean" or int(row["patch_layer"]) != -1:
                continue
            state_path = run_root / model / str(row["state_path"])
            payload = torch.load(state_path, map_location="cpu", weights_only=True)
            answer_states = payload.get("answer_states", {})
            if not answer_states:
                raise RuntimeError(f"Missing answer states in {state_path}")
            for layer, value in answer_states.items():
                rows.append(
                    {
                        "model_label": str(model),
                        "seed": int(row["seed"]),
                        "gold_count": int(row["gold_count"]),
                        "layer": int(layer),
                        "strict_correct": bool(row["strict_correct"]),
                        "vector": value.detach().float().numpy(),
                    }
                )
    if not rows:
        raise ValueError("No clean answer-query states were found")
    states = pd.DataFrame(rows)
    duplicated = states.duplicated(
        ["model_label", "seed", "gold_count", "layer"], keep=False
    )
    if duplicated.any():
        raise RuntimeError("Duplicate clean answer-query state keys")
    return states


def plot_layerwise_metrics(metrics: pd.DataFrame, output: Path) -> None:
    models = list(metrics["model_label"].drop_duplicates())
    figure, axes = plt.subplots(
        len(models), 3, figsize=(15, max(4.2, 4.0 * len(models))), squeeze=False
    )
    for row_axes, model in zip(axes, models):
        frame = metrics[metrics["model_label"].eq(model)].sort_values("layer")
        layer = frame["layer"]

        row_axes[0].plot(
            layer,
            frame["exact_classifier_accuracy"],
            marker="o",
            markersize=3,
            label="logistic exact",
            color="#2563eb",
        )
        row_axes[0].plot(
            layer,
            frame["nearest_centroid_accuracy"],
            marker="o",
            markersize=3,
            label="cosine nearest centroid",
            color="#7c3aed",
        )
        row_axes[0].axhline(0.1, color="#6b7280", linestyle="--", linewidth=1)
        row_axes[0].set_ylim(0, 1.02)
        row_axes[0].set_ylabel("Confirmation exact-count accuracy")
        row_axes[0].legend(frameon=False, fontsize=8)

        row_axes[1].plot(
            layer,
            frame["exact_classifier_mad"],
            label="logistic MAD",
            color="#2563eb",
        )
        row_axes[1].plot(
            layer,
            frame["nearest_centroid_mad"],
            label="centroid MAD",
            color="#7c3aed",
        )
        row_axes[1].plot(
            layer,
            frame["ridge_mad"],
            label="ridge MAD",
            color="#d97706",
        )
        row_axes[1].set_ylabel("Confirmation absolute count error")
        row_axes[1].legend(frameon=False, fontsize=8)

        row_axes[2].plot(
            layer,
            frame["rank3_all"],
            label="rank-3: all discovery states",
            color="#0f766e",
        )
        row_axes[2].plot(
            layer,
            frame["rank3_centroids"],
            label="rank-3: discovery centroids",
            color="#14b8a6",
        )
        row_axes[2].plot(
            layer,
            frame["eta2_count"],
            label="confirmation count $\\eta^2$",
            color="#dc2626",
        )
        row_axes[2].set_ylim(-0.02, 1.02)
        row_axes[2].set_ylabel("Geometry fraction")
        row_axes[2].legend(frameon=False, fontsize=8)

        for axis in row_axes:
            axis.set_title(f"{model}: {axis.get_ylabel()}")
            axis.set_xlabel("Post-block layer (zero-based)")
            axis.grid(alpha=0.22)
    figure.suptitle(
        "Answer-query count geometry across depth: discovery fit, confirmation readout"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def select_display_layers(
    fitted: dict[tuple[str, int], dict[str, Any]], metrics: pd.DataFrame
) -> tuple[dict[str, int], pd.DataFrame]:
    """Select one visually separable 3-D layer per model for display only."""
    rows: list[dict[str, Any]] = []
    rank3_lookup = metrics.set_index(["model_label", "layer"])["rank3_all"]
    for (model, layer), payload in fitted.items():
        pca = payload["pca"]
        discovery = pca.transform(payload["x_train"])
        confirmation = pca.transform(payload["x_test"])
        counts = np.unique(payload["y_train"])
        centroids = np.stack(
            [discovery[payload["y_train"] == count].mean(axis=0) for count in counts]
        )
        distances = np.linalg.norm(
            confirmation[:, None, :] - centroids[None, :, :], axis=2
        )
        prediction = counts[np.argmin(distances, axis=1)]
        rows.append(
            {
                "model_label": model,
                "layer": int(layer),
                "confirmation_3d_nearest_centroid_accuracy": float(
                    np.mean(prediction == payload["y_test"])
                ),
                "confirmation_3d_nearest_centroid_mad": float(
                    np.mean(np.abs(prediction - payload["y_test"]))
                ),
                "discovery_rank3_all": float(rank3_lookup.loc[(model, layer)]),
            }
        )
    candidates = pd.DataFrame(rows)
    selected: dict[str, int] = {}
    candidates["selected_for_display"] = False
    for model, frame in candidates.groupby("model_label", sort=True):
        chosen = frame.sort_values(
            [
                "confirmation_3d_nearest_centroid_accuracy",
                "confirmation_3d_nearest_centroid_mad",
                "discovery_rank3_all",
                "layer",
            ],
            ascending=[False, True, False, True],
        ).iloc[0]
        layer = int(chosen["layer"])
        selected[str(model)] = layer
        candidates.loc[
            candidates["model_label"].eq(model) & candidates["layer"].eq(layer),
            "selected_for_display",
        ] = True
    return selected, candidates


def plot_selected_frozen_pca(
    fitted: dict[tuple[str, int], dict[str, Any]],
    selected_layers: dict[str, int],
    output: Path,
) -> None:
    keys = [(model, layer) for model, layer in selected_layers.items()]
    if not keys:
        return
    figure, axes = plt.subplots(
        1, len(keys), figsize=(7.2 * len(keys), 5.8), squeeze=False
    )
    palette = plt.get_cmap("turbo", 10)
    for axis, key in zip(axes.flat, keys):
        payload = fitted[key]
        pca = payload["pca"]
        discovery = pca.transform(payload["x_train"])
        confirmation = pca.transform(payload["x_test"])
        discovery_view = project_isometric(discovery)
        confirmation_view = project_isometric(confirmation)
        for count in range(1, 11):
            train_mask = payload["y_train"] == count
            test_mask = payload["y_test"] == count
            axis.scatter(
                discovery_view[train_mask, 0],
                discovery_view[train_mask, 1],
                s=10,
                alpha=0.22,
                color=palette(count - 1),
            )
            axis.scatter(
                confirmation_view[test_mask, 0],
                confirmation_view[test_mask, 1],
                s=18,
                alpha=0.78,
                facecolors="none",
                edgecolors=[palette(count - 1)],
                label=str(count) if key == keys[0] else None,
            )
        variance = 100 * pca.explained_variance_ratio_
        axis.set_title(
            f"{key[0]} L{key[1]}\nPC1/2/3 = "
            f"{variance[0]:.1f}/{variance[1]:.1f}/{variance[2]:.1f}%"
        )
        axis.set_xlabel("Isometric horizontal (rotated PC1/PC2)")
        axis.set_ylabel("Isometric vertical (rotated PC1/PC2/PC3)")
        axis.grid(alpha=0.18)
    axes.flat[0].legend(
        title="Gold count",
        ncol=5,
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.42),
    )
    figure.suptitle(
        "Display-only best 3-PC layer: fixed isometric 3-D view; rings are confirmation"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_span_restoration.json",
    )
    parser.add_argument(
        "--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"]
    )
    args = parser.parse_args()

    config = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    discovery = {int(value) for value in config["discovery_seeds"]}
    confirmation = {int(value) for value in config["confirmation_seeds"]}
    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    states = load_answer_states(root, args.models)
    metrics, fitted = geometry_metrics(
        states, discovery, confirmation, bootstrap_draws=0
    )
    selected_layers, display_selection = select_display_layers(fitted, metrics)
    metrics.to_csv(output / "answer_geometry_layerwise.csv", index=False)
    display_selection.to_csv(
        output / "answer_geometry_display_layer_selection.csv", index=False
    )
    plot_layerwise_metrics(metrics, output / "answer_geometry_layerwise.png")
    plot_selected_frozen_pca(
        fitted, selected_layers, output / "answer_frozen_pca_selected_layers.png"
    )
    audit = {
        "schema_version": "realistic_niah_v4_4_5_answer_geometry_analysis_v1",
        "models": list(args.models),
        "discovery_seeds": sorted(discovery),
        "confirmation_seeds": sorted(confirmation),
        "state_rows": int(len(states)),
        "layer_rows": int(len(metrics)),
        "fit_population": "clean natural forward; discovery seeds only",
        "evaluation_population": "clean natural forward; confirmation seeds only",
        "display_layers": selected_layers,
        "display_layer_status": (
            "post-hoc visualization only; not used for layerwise inference or causal claims"
        ),
        "display_layer_rule": (
            "maximize confirmation nearest-centroid accuracy in the frozen discovery "
            "three-PC space; break ties by lower count MAD, higher discovery rank-3 "
            "variance capture, then shallower layer"
        ),
        "predictive_space": (
            "discovery PCA up to 32 components, numerically degenerate components "
            "removed, then discovery StandardScaler"
        ),
        "definitions": {
            "exact_classifier_mad": "mean absolute integer-count error",
            "nearest_centroid": (
                "largest cosine similarity to a discovery count centroid in the "
                "standardized predictive space"
            ),
            "rank3_all": "discovery all-state centered variance captured by three PCs",
            "rank3_centroids": (
                "discovery count-centroid centered variance captured by three PCs"
            ),
            "eta2_count": (
                "confirmation between-count sum of squares divided by total sum of squares"
            ),
            "cosine_silhouette": (
                "confirmation within-count versus nearest other-count cosine distance"
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
