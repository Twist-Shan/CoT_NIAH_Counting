from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ExperimentConfig


sns.set_theme(style="whitegrid", context="notebook")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bin_key(value: str) -> tuple[int, int]:
    if str(value) == "all":
        return (10**9, 10**9)
    parts = str(value).split("-")
    return (int(parts[0]), int(parts[-1]))


def _heatmap(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    title: str,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    if frame.empty or value not in frame.columns:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_axis_off()
        return
    pivot = frame.pivot(index="layer", columns="head", values=value).sort_index()
    sns.heatmap(
        pivot,
        ax=axis,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        square=True,
    )
    axis.set_title(title)
    axis.set_xlabel("head (0-based)")
    axis.set_ylabel("Layer (1-based)")


def plot_learning(cfg: ExperimentConfig, run_dir: Path) -> None:
    figures = run_dir / "figures"
    train = _read(run_dir / "tables" / "train_metrics.csv")
    if not train.empty:
        fig, axes = plt.subplots(
            1,
            len(cfg.position_encodings),
            figsize=(5.2 * len(cfg.position_encodings), 4.2),
            squeeze=False,
            sharey=True,
        )
        for axis, position in zip(axes[0], cfg.position_encodings):
            subset = train[train["position_encoding"] == position]
            sns.lineplot(
                data=subset,
                x="step",
                y="train_total_loss",
                hue="mode",
                ax=axis,
            )
            axis.set_title(position.upper())
            axis.set_xlabel("training step")
            axis.set_ylabel(
                "all-sequence next-token cross-entropy"
                if cfg.loss_scope == "all_sequence"
                else "completion next-token cross-entropy"
            )
        objective = "all-sequence" if cfg.loss_scope == "all_sequence" else "completion-only"
        fig.suptitle(
            f"Training loss ({objective} teacher-forced causal objective)",
            y=1.03,
        )
        _save(fig, figures / "learning_loss.png")

    distribution = _read(run_dir / "tables" / "training_count_distribution.csv")
    if not distribution.empty:
        fig, axis = plt.subplots(figsize=(9.0, 4.3))
        axis.bar(distribution["count"], distribution["probability"], color="#2f6fed")
        axis.set_title(f"Training count distribution: {cfg.count_sampling}")
        axis.set_xlabel("needle count sampled for a training example")
        axis.set_ylabel("sampling probability")
        axis.set_xticks(distribution["count"])
        axis.tick_params(axis="x", labelrotation=90 if len(distribution) > 20 else 0)
        _save(fig, figures / "training_count_distribution.png")

    by_bin = _read(run_dir / "tables" / "eval_by_bin.csv")
    if not by_bin.empty:
        bins = sorted(by_bin["count_bin"].astype(str).unique(), key=_bin_key)
        fig, axes = plt.subplots(
            len(cfg.position_encodings),
            2,
            figsize=(12.5, 4.1 * len(cfg.position_encodings)),
            squeeze=False,
            sharex=False,
            sharey=True,
        )
        palette = sns.color_palette("tab10", n_colors=max(3, len(bins)))
        for row, position in enumerate(cfg.position_encodings):
            subset = by_bin[by_bin["position_encoding"] == position]
            for color, count_bin in zip(palette, bins):
                part = subset[subset["count_bin"].astype(str) == count_bin]
                for mode, linestyle in (("nonthinking", "-"), ("thinking", "--")):
                    line = part[part["mode"] == mode]
                    if not line.empty:
                        axes[row, 0].plot(
                            line["step"],
                            line["tf_final_accuracy"],
                            color=color,
                            linestyle=linestyle,
                            marker="o",
                            markersize=3,
                            label=f"{count_bin} | {mode}",
                        )
                trace = part[part["mode"] == "thinking"]
                if not trace.empty:
                    axes[row, 1].plot(
                        trace["step"],
                        trace["tf_trace_marker_accuracy"],
                        color=color,
                        marker="o",
                        markersize=3,
                        label=count_bin,
                    )
            axes[row, 0].set_title(f"{position.upper()}: teacher-forced final-count accuracy")
            axes[row, 1].set_title(f"{position.upper()}: teacher-forced CoT trace-marker accuracy")
            for axis in axes[row]:
                axis.set_ylim(-0.03, 1.03)
                axis.set_xlabel("training step")
                axis.set_ylabel("accuracy")
                axis.legend(fontsize=8, ncol=2)
        fig.suptitle("Learning dynamics by exact count range", y=1.01)
        _save(fig, figures / "learning_accuracy_by_bin.png")

    by_count = _read(run_dir / "tables" / "eval_by_count.csv")
    if not by_count.empty:
        final = by_count[by_count["step"] == by_count.groupby(
            ["position_encoding", "mode"]
        )["step"].transform("max")]
        fig, axes = plt.subplots(
            1,
            len(cfg.position_encodings),
            figsize=(5.2 * len(cfg.position_encodings), 4.2),
            squeeze=False,
            sharey=True,
        )
        for axis, position in zip(axes[0], cfg.position_encodings):
            subset = final[final["position_encoding"] == position]
            sns.lineplot(
                data=subset,
                x="count",
                y="tf_final_accuracy",
                hue="mode",
                marker="o",
                ax=axis,
            )
            axis.set_ylim(-0.03, 1.03)
            axis.set_title(position.upper())
            axis.set_xlabel("gold needle count")
            axis.set_ylabel("teacher-forced final-count accuracy")
        fig.suptitle("Final checkpoint accuracy by exact count", y=1.03)
        _save(fig, figures / "final_accuracy_by_count.png")


def plot_attention(cfg: ExperimentConfig, run_dir: Path) -> None:
    summary = _read(run_dir / "tables" / "attention_summary.csv")
    if summary.empty:
        return
    figures = run_dir / "figures"
    for position in cfg.position_encodings:
        current = summary[
            (summary["position_encoding"] == position) & (summary["count_bin"].astype(str) == "all")
        ]
        nonthinking = current[
            (current["mode"] == "nonthinking") & (current["query_kind"] == "final_answer")
        ]
        targeted = current[
            (current["mode"] == "thinking") & (current["query_kind"] == "trace_index")
        ]
        readout = current[
            (current["mode"] == "thinking") & (current["query_kind"] == "final_answer")
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6))
        _heatmap(
            axes[0],
            nonthinking,
            "broad_attention_score",
            "Non-thinking final query\nbroad needle score",
        )
        _heatmap(
            axes[1],
            targeted,
            "correct_prompt_needle_mass",
            "CoT trace index query\nraw matching-needle mass",
        )
        _heatmap(
            axes[2],
            readout,
            "trace_markers_mass",
            "CoT final query\nmass on all trace markers",
        )
        fig.suptitle(f"{position.upper()} descriptive attention signatures", y=1.03)
        _save(fig, figures / f"attention_signatures_{position}.png")

        bins = sorted(
            [value for value in summary["count_bin"].astype(str).unique() if value != "all"],
            key=_bin_key,
        )
        fig, axes = plt.subplots(3, len(bins), figsize=(5.0 * len(bins), 12.5), squeeze=False)
        for column, count_bin in enumerate(bins):
            part = summary[
                (summary["position_encoding"] == position)
                & (summary["mode"] == "thinking")
                & (summary["query_kind"] == "trace_index")
                & (summary["count_bin"].astype(str) == count_bin)
            ]
            for row, (metric, title) in enumerate(
                (
                    ("correct_prompt_needle_mass", "raw k-to-k mass"),
                    ("diagonal_dominance", "diagonal dominance within needles"),
                    ("correct_top1", "top-1 among prompt needles"),
                )
            ):
                _heatmap(axes[row, column], part, metric, f"count {count_bin}\n{title}")
        fig.suptitle(f"{position.upper()} targeted retrieval diagnostics by count range", y=1.01)
        _save(fig, figures / f"targeted_retrieval_by_bin_{position}.png")


def plot_state(cfg: ExperimentConfig, run_dir: Path) -> None:
    probes = _read(run_dir / "tables" / "state_probe_summary.csv")
    centroids = _read(run_dir / "tables" / "state_centroids_pca.csv")
    variance = _read(run_dir / "tables" / "state_pca_variance.csv")
    if probes.empty:
        return
    figures = run_dir / "figures"
    for position in cfg.position_encodings:
        current = probes[probes["position_encoding"] == position].copy()
        current["site_label"] = current["mode"] + " | " + current["site"]
        sites = list(dict.fromkeys(current["site_label"].tolist()))
        fig, axes = plt.subplots(1, 3, figsize=(16.2, max(4.1, 0.58 * len(sites) + 2.2)))
        for axis, (metric, title, vmin, vmax) in zip(
            axes,
            (
                ("nearest_centroid_accuracy", "Held-out nearest-centroid accuracy", 0.0, 1.0),
                ("position_only_accuracy", "Absolute-position-only baseline", 0.0, 1.0),
                ("ridge_r2", "Held-out ridge count R²", -0.2, 1.0),
            ),
        ):
            pivot = current.pivot(index="site_label", columns="layer", values=metric).reindex(sites)
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=vmin, vmax=vmax, ax=axis)
            axis.set_title(title)
            axis.set_xlabel("residual state: 0=embedding, 1–4=after Layer")
            axis.set_ylabel("mode | semantic site")
        fig.suptitle(f"{position.upper()} count-state decodability and position control", y=1.03)
        _save(fig, figures / f"state_probe_{position}.png")

        current_variance = variance[variance["position_encoding"] == position]
        if not current_variance.empty:
            fig, axes = plt.subplots(
                1,
                len(sites),
                figsize=(4.7 * len(sites), 4.0),
                squeeze=False,
                sharey=True,
            )
            for axis, site_label in zip(axes[0], sites):
                mode, site = [part.strip() for part in site_label.split("|", maxsplit=1)]
                subset = current_variance[
                    (current_variance["mode"] == mode) & (current_variance["site"] == site)
                ]
                for layer in sorted(subset["layer"].unique()):
                    line = subset[subset["layer"] == layer]
                    axis.plot(
                        line["component"],
                        line["cumulative_explained_variance"],
                        marker="o",
                        label=f"state {int(layer)}",
                    )
                axis.set_title(site_label)
                axis.set_xlabel("number of centroid PCs")
                axis.set_ylabel("cumulative explained variance")
                axis.set_ylim(0.0, 1.04)
                axis.legend(fontsize=8)
            fig.suptitle(f"{position.upper()} count-centroid PCA coverage", y=1.03)
            _save(fig, figures / f"state_pca_variance_{position}.png")

        current_centroids = centroids[centroids["position_encoding"] == position]
        for (mode, site), subset in current_centroids.groupby(["mode", "site"], sort=False):
            layers = sorted(int(value) for value in subset["layer"].unique())
            columns = min(3, len(layers))
            rows = math.ceil(len(layers) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.2 * rows), squeeze=False)
            for axis, layer in zip(axes.flat, layers):
                part = subset[subset["layer"] == layer].sort_values("state_label")
                axis.plot(part["pc1"], part["pc2"], color="#8ca0b8", linewidth=1.2, alpha=0.8)
                scatter = axis.scatter(
                    part["pc1"],
                    part["pc2"],
                    c=part["state_label"],
                    cmap="viridis",
                    s=38,
                    edgecolor="white",
                    linewidth=0.35,
                )
                axis.set_title("embedding" if layer == 0 else f"after Layer {layer}")
                axis.set_xlabel("centroid PC1")
                axis.set_ylabel("centroid PC2")
                fig.colorbar(scatter, ax=axis, label="count / trace progress")
            for axis in list(axes.flat)[len(layers) :]:
                axis.set_axis_off()
            fig.suptitle(
                f"{position.upper()} | {mode} | {site}: mean state per count/progress label",
                y=1.01,
            )
            _save(fig, figures / f"state_centroids_{position}_{mode}_{site}.png")


def make_all_plots(cfg: ExperimentConfig, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    plot_learning(cfg, run_dir)
    plot_attention(cfg, run_dir)
    plot_state(cfg, run_dir)
