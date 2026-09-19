from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import V10Config


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def plot_training_dynamics(run_dir: Path) -> None:
    figures = run_dir / "figures" / "training"
    figures.mkdir(parents=True, exist_ok=True)
    train_frames = []
    for mode in ("nonthinking", "thinking"):
        path = run_dir / "tables" / f"train_{mode}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["mode"] = mode
            train_frames.append(frame)
    if train_frames:
        train = pd.concat(train_frames, ignore_index=True)
        loss_columns = [column for column in train.columns if column.endswith("_loss")]
        long = train.melt(
            id_vars=["step", "mode"],
            value_vars=loss_columns,
            var_name="component",
            value_name="loss",
        ).dropna()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        sns.lineplot(data=long[long.component == "total_loss"], x="step", y="loss", hue="mode", ax=axes[0])
        axes[0].set_title("Training total supervised loss")
        sns.lineplot(data=long[long.component != "total_loss"], x="step", y="loss", hue="component", style="mode", ax=axes[1])
        axes[1].set_title("Training loss by supervised component")
        _save(fig, figures / "training_loss_components.png")

    by_bin = _read(run_dir / "tables" / "eval_dynamics_by_bin.csv")
    if not by_bin.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharey=True)
        sns.lineplot(data=by_bin, x="step", y="tf_accuracy", hue="mode", style="count_bin", marker="o", ax=axes[0])
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].set_title("Teacher-forced final-count accuracy")
        ar = by_bin.dropna(subset=["ar_accuracy"])
        if not ar.empty:
            sns.lineplot(data=ar, x="step", y="ar_accuracy", hue="mode", style="count_bin", marker="o", ax=axes[1])
        axes[1].set_ylim(-0.03, 1.03)
        axes[1].set_title("Autoregressive final-count accuracy")
        _save(fig, figures / "accuracy_dynamics_by_count_bin.png")

    by_count = _read(run_dir / "tables" / "eval_dynamics_by_count.csv")
    if not by_count.empty:
        final_step = int(by_count.step.max())
        final = by_count[by_count.step == final_step]
        fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
        metric = "ar_accuracy" if final.ar_accuracy.notna().any() else "tf_accuracy"
        sns.lineplot(data=final, x="count", y=metric, hue="mode", marker="o", ax=ax)
        ax.axvspan(0.5, 10.5, color="#4c78a8", alpha=0.05)
        ax.axvspan(10.5, 20.5, color="#f58518", alpha=0.05)
        ax.axvspan(20.5, 30.5, color="#54a24b", alpha=0.05)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"Final checkpoint accuracy by exact count (step {final_step})")
        ax.set_ylabel(metric)
        _save(fig, figures / "final_accuracy_by_exact_count.png")


def _heatmap_from_heads(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ax: plt.Axes,
    cfg: V10Config,
    *,
    vmin: float = 0.0,
    vmax: float | None = None,
) -> None:
    if frame.empty or metric not in frame.columns or not np.isfinite(frame[metric].to_numpy(dtype=float)).any():
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_axis_off()
        return
    pivot = frame.pivot(index="layer", columns="head", values=metric).reindex(
        index=range(cfg.n_layer), columns=range(cfg.n_head)
    )
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=vmin, vmax=vmax, ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xlabel("Head (0-based)")
    ax.set_ylabel("Layer (0-based)")


def plot_attention(run_dir: Path, cfg: V10Config) -> None:
    root = run_dir / "analysis" / "attention_causal"
    tables = root / "tables"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = _read(tables / "attention_head_summary.csv")
    if not summary.empty:
        panels = [
            (
                summary[(summary["mode"] == "nonthinking") & (summary["query_kind"] == "final_count_query")],
                "broad_attention_score",
                "Non-thinking: broad needle attention",
                None,
            ),
            (
                summary[(summary["mode"] == "thinking") & (summary["query_kind"] == "targeted_retrieval_query")],
                "correct_prompt_needle_mass",
                "Thinking: targeted k-to-k retrieval",
                1.0,
            ),
            (
                summary[(summary["mode"] == "thinking") & (summary["query_kind"] == "final_count_query")],
                "trace_markers_mass",
                "Thinking final readout: trace-marker mass",
                1.0,
            ),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        for ax, (frame, metric, title, vmax) in zip(axes, panels):
            _heatmap_from_heads(frame, metric, title, ax, cfg, vmax=vmax)
        _save(fig, figures / "attention_candidate_signatures.png")

        categories = [
            "bos_mass",
            "prompt_needles_mass",
            "prompt_noise_mass",
            "think_open_mass",
            "trace_indices_mass",
            "trace_markers_mass",
            "think_close_mass",
            "ans_mass",
            "other_or_query_self_mass",
        ]
        final_rows = summary[summary.query_kind == "final_count_query"].copy()
        final_rows["head_label"] = (
            "L"
            + (final_rows["layer"].astype(int) + 1).astype(str)
            + "H"
            + final_rows["head"].astype(int).astype(str)
        )
        for mode in ("nonthinking", "thinking"):
            frame = final_rows[final_rows["mode"] == mode]
            if frame.empty:
                continue
            long = frame.melt(id_vars=["head_label"], value_vars=categories, var_name="category", value_name="mass")
            pivot = long.pivot(index="head_label", columns="category", values="mass")
            fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako", vmin=0, ax=ax)
            ax.set_title(f"{mode}: complete final-query attention decomposition")
            _save(fig, figures / f"{mode}_final_query_attention_categories.png")

    ablation = _read(tables / "topn_ablation_summary.csv")
    if not ablation.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, mode in zip(axes, ("nonthinking", "thinking")):
            frame = ablation[(ablation["mode"] == mode) & (ablation.top_n > 0)]
            sns.lineplot(data=frame, x="top_n", y="drop_final_count_margin", hue="ranking", marker="o", ax=ax)
            ax.axhline(0, color="black", lw=1)
            ax.set_title(f"{mode}: cumulative top-n head ablation")
            ax.set_ylabel("drop in final-count logit margin")
        _save(fig, figures / "topn_head_ablation.png")

        thinking = ablation[(ablation["mode"] == "thinking") & (ablation.top_n > 0)]
        if not thinking.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            sns.lineplot(data=thinking, x="top_n", y="drop_trace_marker_margin", hue="ranking", marker="o", ax=axes[0])
            axes[0].set_title("Thinking: trace-marker margin ablation")
            sns.lineplot(data=thinking, x="top_n", y="drop_trace_index_accuracy", hue="ranking", marker="o", ax=axes[1])
            axes[1].set_title("Thinking: successor-index accuracy ablation")
            _save(fig, figures / "thinking_topn_trace_ablation.png")

    retrieval = _read(tables / "retrieval_patching_summary.csv")
    if not retrieval.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        sns.lineplot(data=retrieval, x="top_n", y="normalized_recovery", hue="ranking", marker="o", ax=ax)
        ax.axhline(0, color="black", lw=1)
        ax.axhline(1, color="gray", ls="--", lw=1)
        ax.set_title("Clean-to-corrupt targeted-head patching")
        _save(fig, figures / "retrieval_patching_topn.png")

    offsets = _read(tables / "count_offset_head_patching_summary.csv")
    if not offsets.empty:
        for mode in ("nonthinking", "thinking"):
            primary = "direct_broad" if mode == "nonthinking" else "trace_readout"
            frame = offsets[(offsets["mode"] == mode) & (offsets.ranking == primary)]
            pivot = frame.pivot(index="top_n", columns="donor_offset", values="causal_expected_shift")
            fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
            limit = max(abs(pivot.to_numpy()).max(), 1e-6)
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-limit, vmax=limit, ax=ax)
            ax.set_title(f"{mode}: head-output patch m-to-n expected-count shift")
            ax.set_xlabel("donor count minus receiver count")
            ax.set_ylabel("number of patched top-ranked heads")
            _save(fig, figures / f"{mode}_count_offset_head_patching.png")


def plot_state(run_dir: Path, cfg: V10Config) -> None:
    root = run_dir / "analysis" / "state_causal"
    tables = root / "tables"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    geometry = _read(tables / "direction_geometry.csv")
    if not geometry.empty:
        geometry = geometry.copy()
        geometry["layer_display"] = geometry.layer.astype(int) + 1
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        sns.lineplot(data=geometry, x="layer_display", y="projection_r2_heldout", hue="site", style="method", marker="o", ax=axes[0])
        axes[0].set_title("Held-out count-direction readability")
        axes[0].set_xlabel("Layer")
        sns.lineplot(data=geometry, x="layer_display", y="adjacent_delta_cosine_mean", hue="site", style="method", marker="o", ax=axes[1])
        axes[1].set_title("Parallelism of adjacent count differences")
        axes[1].set_xlabel("Layer")
        _save(fig, figures / "direction_geometry_by_layer.png")

    manifold_geometry = _read(tables / "manifold_geometry.csv")
    points = _read(tables / "manifold_points.csv")
    if not manifold_geometry.empty:
        frame = manifold_geometry.copy()
        frame["layer_display"] = frame.layer.astype(int) + 1
        long = frame.melt(
            id_vars=["site", "layer_display"],
            value_vars=["pc2_cumulative", "pc3_cumulative", "pc6_cumulative"],
            var_name="components",
            value_name="explained_variance",
        )
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        sns.lineplot(data=long, x="layer_display", y="explained_variance", hue="site", style="components", marker="o", ax=ax)
        ax.set_ylim(0, 1.03)
        ax.set_title("How many PCs explain the count-state manifold?")
        ax.set_xlabel("Layer")
        _save(fig, figures / "pca_2_3_6_component_variance.png")

    if not points.empty:
        for site in points.site.unique():
            site_points = points[points.site == site]
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
            for layer, ax in enumerate(axes.flat):
                frame = site_points[site_points.layer == layer]
                scatter = ax.scatter(frame.pc1, frame.pc2, c=frame.state_label, cmap="viridis", s=12, alpha=0.65)
                ax.set_title(f"Layer {layer + 1}")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
            fig.colorbar(scatter, ax=axes, label="count / trace progress label", shrink=0.75)
            fig.suptitle(f"{site}: per-Layer 2D PCA")
            _save(fig, figures / f"{site}_pca2d_by_layer.png")

            fig = plt.figure(figsize=(13, 10), constrained_layout=True)
            axes3d = [fig.add_subplot(2, 2, layer + 1, projection="3d") for layer in range(cfg.n_layer)]
            for layer, ax in enumerate(axes3d):
                frame = site_points[site_points.layer == layer]
                scatter = ax.scatter(frame.pc1, frame.pc2, frame.pc3, c=frame.state_label, cmap="viridis", s=10, alpha=0.65)
                ax.set_title(f"Layer {layer + 1}")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
            fig.colorbar(scatter, ax=axes3d, label="count / trace progress label", shrink=0.65)
            fig.suptitle(f"{site}: per-Layer 3D PCA")
            _save(fig, figures / f"{site}_pca3d_by_layer.png")

    steering = _read(tables / "steering_summary.csv")
    if not steering.empty:
        sites = ("nonthinking_final_answer", "thinking_final_answer", "thinking_fixed_trace_answer")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True)
        for ax, site in zip(axes, sites):
            sns.lineplot(data=steering[steering.site == site], x="alpha", y="causal_expected_shift", hue="layer", marker="o", ax=ax)
            ax.axhline(0, color="black", lw=1)
            ax.set_title(site)
            ax.set_ylabel("mean expected-count shift")
        _save(fig, figures / "geometry_steering_by_layer.png")

    final_patch = _read(tables / "final_state_transplant_summary.csv")
    if not final_patch.empty:
        sites = ("nonthinking_final_answer", "thinking_final_answer", "thinking_fixed_trace_answer")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        for ax, site in zip(axes, sites):
            frame = final_patch[final_patch.site == site]
            pivot = frame.pivot(index="layer", columns="donor_offset", values="causal_expected_shift")
            limit = max(abs(pivot.to_numpy()).max(), 1e-6)
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-limit, vmax=limit, ax=ax)
            ax.set_title(site)
            ax.set_xlabel("donor count minus receiver count")
            ax.set_ylabel("patched residual after Layer (0-based)")
        _save(fig, figures / "final_state_m_to_n_transplant.png")

    trace_patch = _read(tables / "trace_progress_transplant_summary.csv")
    if not trace_patch.empty:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
        for ax, metric, title in zip(
            axes,
            ("follows_donor_successor", "early_close_induced", "continuation_induced"),
            ("Follows donor successor", "Final donor induces early close", "Earlier donor induces continuation"),
        ):
            pivot = trace_patch.pivot(index="layer", columns="donor_offset", values=metric)
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("donor progress minus receiver progress")
            ax.set_ylabel("patched residual after Layer (0-based)")
        _save(fig, figures / "trace_progress_m_to_n_transplant.png")


def write_analysis_manifest(run_dir: Path, cfg: V10Config) -> None:
    manifest = {
        "definitions": {
            "broad_attention_score": "sum attention to prompt needles multiplied by normalized entropy within needle positions",
            "targeted_retrieval_score": "raw attention from trace index <k> to the kth prompt needle",
            "topn_ablation": "mask the cumulatively top-ranked 1..16 attention heads and compare against a matched random ordering",
            "normalized_recovery": "(patched margin - corrupt margin) / (clean margin - corrupt margin)",
            "state_label": "gold total count at final-answer anchors; prefix progress k at trace anchors",
            "causal_expected_shift": "softmax expected count after intervention minus the unmodified expected count",
        },
        "position_confound_warning": (
            "thinking natural-trace donor and receiver positions differ with count/progress under learned absolute positions; "
            "all transplant tables therefore retain donor_position, receiver_position, and position_delta"
        ),
        "config": cfg.to_dict(),
    }
    (run_dir / "analysis" / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def make_all_plots(run_dir: str | Path, cfg: V10Config) -> None:
    path = Path(run_dir)
    sns.set_theme(style="whitegrid", context="notebook")
    plot_training_dynamics(path)
    plot_attention(path, cfg)
    plot_state(path, cfg)
    write_analysis_manifest(path, cfg)
