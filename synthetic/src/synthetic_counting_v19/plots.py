from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COUNT_BANDS = ["count_1_32", "count_33_64", "count_65_96", "count_97_128"]
COUNT_LABELS = {
    "count_1_32": "count 1-32",
    "count_33_64": "count 33-64",
    "count_65_96": "count 65-96",
    "count_97_128": "count 97-128",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _save(figure: plt.Figure, path: Path) -> None:
    if figure.get_layout_engine() is None:
        figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _heatmap(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    title: str,
    *,
    limit: tuple[float, float] = (0.0, 1.0),
) -> None:
    if frame.empty or value not in frame:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_axis_off()
        return
    pivot = frame.pivot_table(index="layer", columns="head", values=value, aggfunc="mean")
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=limit[0],
        vmax=limit[1],
        cbar=False,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("attention head (0-based)")
    axis.set_ylabel("Transformer layer (1-based)")


def plot_learning(run_dir: Path, figures: Path) -> None:
    training = _read(run_dir / "tables" / "training_metrics.csv")
    if not training.empty:
        figure, axis = plt.subplots(figsize=(11, 6))
        for name, group in training.groupby("run_name"):
            axis.plot(group["step"], group["loss"], label=name, alpha=0.85)
        axis.set(
            xlabel="training step",
            ylabel="completion next-token cross-entropy",
            title="v19 training loss: shared decimal digits for trace indices and final counts",
        )
        axis.legend(fontsize=8, ncol=2)
        _save(figure, figures / "training_loss.png")

    dynamics = _read(run_dir / "tables" / "dynamics_by_band.csv")
    if dynamics.empty:
        return
    distributions = list(dict.fromkeys(dynamics["distribution"].astype(str)))
    figure, axes = plt.subplots(
        len(distributions),
        len(COUNT_BANDS),
        figsize=(18, 4.2 * len(distributions)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = {"direct": "#2563eb", "cot": "#ea580c"}
    for row, distribution in enumerate(distributions):
        for column, band in enumerate(COUNT_BANDS):
            axis = axes[row, column]
            subset = dynamics[
                (dynamics["distribution"] == distribution)
                & (dynamics["count_band"] == band)
            ]
            for mode, group in subset.groupby("mode"):
                group = group.sort_values("step")
                axis.plot(
                    group["step"],
                    group["primary_accuracy"],
                    marker="o",
                    ms=3,
                    label=mode,
                    color=colors.get(str(mode)),
                )
            axis.set_title(f"{distribution}: {COUNT_LABELS[band]}")
            axis.set_ylim(-0.03, 1.03)
            axis.set_xlabel("training step")
            if column == 0:
                axis.set_ylabel("free-running final-count accuracy")
            if row == 0 and column == len(COUNT_BANDS) - 1:
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(title="model mode")
    _save(figure, figures / "learning_primary_accuracy_by_count_band.png")

    cot = dynamics[dynamics["mode"] == "cot"]
    if cot.empty:
        return
    figure, axes = plt.subplots(
        len(distributions),
        len(COUNT_BANDS),
        figsize=(18, 4.2 * len(distributions)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, distribution in enumerate(distributions):
        for column, band in enumerate(COUNT_BANDS):
            axis = axes[row, column]
            subset = cot[
                (cot["distribution"] == distribution) & (cot["count_band"] == band)
            ].sort_values("step")
            axis.plot(subset["step"], subset["enumeration_accuracy"], label="index enumeration", marker="o", ms=3)
            axis.plot(subset["step"], subset["trace_marker_accuracy"], label="marker identity", marker="o", ms=3)
            axis.plot(subset["step"], subset["trace_exact_accuracy"], label="exact full trace", marker="o", ms=3)
            axis.set_title(f"{distribution}: {COUNT_LABELS[band]}")
            axis.set_ylim(-0.03, 1.03)
            axis.set_xlabel("training step")
            if column == 0:
                axis.set_ylabel("free-running CoT trace accuracy")
            if row == 0 and column == len(COUNT_BANDS) - 1:
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(fontsize=8)
    _save(figure, figures / "learning_cot_trace_by_count_band.png")


def plot_final(run_dir: Path, figures: Path) -> None:
    summary = _read(run_dir / "tables" / "final_summary.csv")
    if not summary.empty:
        figure, axis = plt.subplots(figsize=(12, max(5, 0.5 * len(summary))))
        ordered = summary.sort_values("primary_accuracy")
        sns.barplot(data=ordered, y="run_name", x="primary_accuracy", hue="mode", dodge=False, ax=axis)
        axis.set(xlim=(0, 1.0), xlabel="free-running final-count accuracy", ylabel="run specification")
        _save(figure, figures / "final_primary_accuracy.png")

    by_count = _read(run_dir / "tables" / "final_by_count.csv")
    if by_count.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, sharey=True)
    for name, group in by_count.groupby("run_name"):
        axes[0].plot(group["gold_count"], group["token_accuracy"], label=name)
    axes[0].set(
        xlabel="gold needle count",
        ylabel="free-running accuracy",
        title="Exact decoded final count (shared decimal digits)",
        ylim=(-0.02, 1.02),
    )
    cot = by_count[by_count["mode"] == "cot"]
    for name, group in cot.groupby("run_name"):
        axes[1].plot(group["gold_count"], group["trace_exact_accuracy"], label=name)
    axes[1].set(
        xlabel="gold needle count",
        title="Exact CoT trace: all digit-tokenized indices and marker identities",
        ylim=(-0.02, 1.02),
    )
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    _save(figure, figures / "final_accuracy_by_count.png")


def plot_attention(run_dir: Path, figures: Path) -> None:
    frame = _read(run_dir / "tables" / "attention_summary.csv")
    if frame.empty:
        return
    for distribution in sorted(frame["distribution"].unique()):
        subset = frame[
            (frame["distribution"] == distribution) & (frame["count_band"] == "all")
        ]
        direct = subset[(subset["mode"] == "direct") & (subset["query_kind"] == "final_answer")]
        targeted = subset[(subset["mode"] == "cot") & (subset["query_kind"] == "trace_index")]
        readout = subset[(subset["mode"] == "cot") & (subset["query_kind"] == "final_answer")]
        successor = subset[(subset["mode"] == "cot") & (subset["query_kind"] == "trace_marker")]
        figure, axes = plt.subplots(1, 4, figsize=(20, 4.8))
        _heatmap(axes[0], direct, "broad_attention_score", "Direct: broad prompt-needle score")
        _heatmap(axes[1], targeted, "correct_prompt_needle_mass", "CoT index final digit: matching-needle mass")
        _heatmap(axes[2], successor, "next_prompt_needle_mass", "CoT M_k: next prompt-needle mass")
        _heatmap(axes[3], readout, "trace_markers_mass", "CoT <Count>: mass on all trace markers")
        figure.suptitle(f"v19 descriptive attention signatures: {distribution}", fontsize=16, fontweight="bold")
        _save(figure, figures / f"attention_signatures_{distribution}.png")

        for metric, title, filename in (
            ("correct_prompt_needle_mass", "Raw k-to-k attention mass", "ktok_raw_mass"),
            ("diagonal_dominance", "Needle-conditional diagonal dominance", "ktok_diagonal_dominance"),
            ("correct_top1", "Correct top-1 within prompt needles", "ktok_correct_top1"),
        ):
            figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
            for axis, band in zip(axes, COUNT_BANDS):
                band_frame = frame[
                    (frame["distribution"] == distribution)
                    & (frame["mode"] == "cot")
                    & (frame["query_kind"] == "trace_index")
                    & (frame["count_band"] == band)
                ]
                _heatmap(axis, band_frame, metric, COUNT_LABELS[band])
            figure.suptitle(f"{distribution} CoT targeted retrieval: {title}", fontsize=15, fontweight="bold")
            _save(figure, figures / f"attention_{filename}_by_band_{distribution}.png")


def plot_state(run_dir: Path, figures: Path) -> None:
    probes = _read(run_dir / "tables" / "state_probe_summary.csv")
    centroids = _read(run_dir / "tables" / "state_centroids_pca.csv")
    variance = _read(run_dir / "tables" / "state_pca_variance.csv")
    if not probes.empty:
        labels = (
            probes["distribution"].astype(str)
            + " / "
            + probes["mode"].astype(str)
            + "\n"
            + probes["site"].astype(str)
        )
        probes = probes.assign(series_label=labels)
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(22, max(6.5, 0.62 * probes.series_label.nunique())),
            constrained_layout=True,
        )
        for axis, metric, title, limits in (
            (axes[0], "nearest_centroid_accuracy", "Exact count/progress nearest-centroid accuracy", (0, 1)),
            (axes[1], "ridge_r2", "Ridge count/progress R²", (-0.25, 1)),
            (axes[2], "position_only_accuracy", "Absolute-position-only baseline accuracy", (0, 1)),
        ):
            pivot = probes.pivot_table(index="series_label", columns="layer", values=metric)
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                vmin=limits[0],
                vmax=limits[1],
                cbar=False,
                ax=axis,
            )
            axis.set_title(title)
            axis.set_xlabel("state index: 0=embedding; 1-4=after Layers 1-4")
            axis.tick_params(axis="y", labelsize=8)
            axis.set_ylabel("run and semantic site" if axis is axes[0] else "")
        _save(figure, figures / "state_probe_summary.png")

    if not variance.empty:
        groups = list(variance.groupby(["run_name", "site"]))
        columns = 2
        rows = math.ceil(len(groups) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(13, 4 * rows), squeeze=False)
        for axis, ((run_name, site), group) in zip(axes.flat, groups):
            for layer, layer_frame in group.groupby("layer"):
                layer_frame = layer_frame.sort_values("component")
                axis.plot(
                    layer_frame["component"],
                    layer_frame["cumulative_explained_variance"],
                    marker="o",
                    label=f"state index {layer}",
                )
            axis.set(
                title=f"{run_name}\n{site}",
                xlabel="number of principal components",
                ylabel="cumulative variance of count centroids",
                ylim=(0, 1.03),
                xticks=range(1, 7),
            )
            axis.legend(fontsize=7, ncol=2)
        for axis in axes.flat[len(groups):]:
            axis.set_axis_off()
        _save(figure, figures / "state_pca_cumulative_variance.png")

    if centroids.empty or "pc1" not in centroids or "pc2" not in centroids:
        return
    for (run_name, site), group in centroids.groupby(["run_name", "site"]):
        layers = sorted(group["layer"].unique())
        columns = min(3, len(layers))
        rows = math.ceil(len(layers) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.2 * columns, 4.5 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, layer in zip(axes.flat, layers):
            current = group[group["layer"] == layer].sort_values("state_label")
            scatter = axis.scatter(
                current["pc1"], current["pc2"], c=current["state_label"], cmap="viridis", s=28
            )
            axis.plot(current["pc1"], current["pc2"], color="#64748b", linewidth=0.8, alpha=0.6)
            axis.set(
                title=f"state index {layer}",
                xlabel="PC1 of exact-count centroids",
                ylabel="PC2 of exact-count centroids",
            )
        for axis in axes.flat[len(layers):]:
            axis.set_axis_off()
        figure.colorbar(scatter, ax=axes.ravel().tolist(), label="count or trace-progress label", shrink=0.75)
        safe_name = run_name.replace("/", "_")
        _save(figure, figures / f"state_centroid_pc12_{safe_name}_{site}.png")


def make_plots(run_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_learning(run_dir, figures)
    plot_final(run_dir, figures)
    plot_attention(run_dir, figures)
    plot_state(run_dir, figures)
