from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import V20Config


sns.set_theme(style="whitegrid", context="notebook")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_v20_loss_suites(cfg: V20Config, run_dir: Path) -> None:
    frame = _read(run_dir / "tables" / "eval_loss_curves.csv")
    if frame.empty:
        return
    suites = [suite for suite in ("raw", "task", "mixture") if suite in set(frame["suite"])]
    positions = [value for value in cfg.position_encodings if value in set(frame["position_encoding"])]
    modes = [value for value in ("nonthinking", "thinking") if value in set(frame["mode"])]
    colors = {"train": "#2864b7", "heldout": "#d1543f"}
    styles = {"nonthinking": "-", "thinking": "--"}
    figure, axes = plt.subplots(
        len(positions), len(suites),
        figsize=(5.0 * len(suites), 4.0 * len(positions)),
        squeeze=False,
    )
    for row, position in enumerate(positions):
        for column, suite in enumerate(suites):
            axis = axes[row, column]
            subset = frame[(frame.position_encoding == position) & (frame.suite == suite)]
            for source in ("train", "heldout"):
                for mode in modes:
                    line = subset[(subset.curve_source == source) & (subset["mode"] == mode)].sort_values("step")
                    if line.empty:
                        continue
                    axis.plot(
                        line.step,
                        line.example_mean_cross_entropy,
                        color=colors[source],
                        linestyle=styles[mode],
                        marker="o",
                        markersize=3,
                        label=f"{source} | {mode}",
                    )
            axis.set_title(f"{position.upper()} — {suite}")
            axis.set_xlabel("training step")
            axis.set_ylabel("mean per-sequence cross-entropy")
            axis.legend(fontsize=8)
    figure.suptitle(
        f"{cfg.version} fixed-suite loss: train versus held-out/test (validation region)",
        y=1.01,
    )
    _save(figure, run_dir / "figures" / "learning_loss_suites_train_vs_heldout.png")


def plot_v20_token_weighted_suites(cfg: V20Config, run_dir: Path) -> None:
    frame = _read(run_dir / "tables" / "eval_loss_curves.csv")
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for axis, suite in zip(axes[0], ("raw", "task", "mixture")):
        subset = frame[frame.suite == suite]
        if not subset.empty:
            sns.lineplot(
                data=subset,
                x="step",
                y="token_weighted_cross_entropy",
                hue="curve_source",
                style="mode",
                units="position_encoding",
                estimator=None,
                ax=axis,
            )
        axis.set_title(suite)
        axis.set_ylabel("token-weighted cross-entropy")
    figure.suptitle("Secondary token-weighted loss diagnostic", y=1.02)
    _save(figure, run_dir / "figures" / "learning_loss_suites_token_weighted.png")


def plot_v20_training(cfg: V20Config, run_dir: Path) -> None:
    frame = _read(run_dir / "tables" / "train_metrics.csv")
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.lineplot(data=frame, x="step", y="train_total_loss", hue="mode", style="position_encoding", ax=axes[0])
    axes[0].set_title("Stochastic optimizer minibatch loss")
    sns.lineplot(
        data=frame,
        x="step",
        y="batch_realized_task_example_ratio",
        hue="mode",
        style="position_encoding",
        ax=axes[1],
    )
    axes[1].axhline(cfg.task_occurrence_ratio, color="black", linewidth=1, linestyle=":")
    axes[1].set_title("Realized task-example ratio")
    _save(figure, run_dir / "figures" / "training_diagnostics.png")


def plot_v20_accuracy(run_dir: Path) -> None:
    frame = _read(run_dir / "tables" / "eval_by_count.csv")
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(
        data=frame,
        x="step",
        y="tf_final_accuracy",
        hue="mode",
        style="position_encoding",
        marker="o",
        ax=axis,
    )
    axis.set_ylim(-0.03, 1.03)
    axis.set_title("Held-out teacher-forced counting accuracy")
    _save(figure, run_dir / "figures" / "learning_count_accuracy.png")


def plot_v20_analysis(run_dir: Path) -> None:
    attention = _read(run_dir / "tables" / "attention_summary.csv")
    if not attention.empty:
        final = attention[attention.query_kind == "final_answer"]
        figure, axis = plt.subplots(figsize=(9, 4.5))
        sns.lineplot(
            data=final,
            x="layer",
            y="prompt_needles_mass",
            hue="mode",
            style="position_encoding",
            marker="o",
            ax=axis,
        )
        axis.set_title("Final-answer attention mass on matching prompt characters")
        _save(figure, run_dir / "figures" / "attention_needles_by_layer.png")
    states = _read(run_dir / "tables" / "state_probe_summary.csv")
    if not states.empty:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        sns.lineplot(
            data=states[states.site == "final_answer"],
            x="layer",
            y="nearest_centroid_accuracy",
            hue="mode",
            style="position_encoding",
            marker="o",
            ax=axis,
        )
        axis.set_title("Held-out count decoding from final-answer states")
        _save(figure, run_dir / "figures" / "state_count_decoding.png")


def make_all_v20_plots(cfg: V20Config, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    plot_v20_loss_suites(cfg, run_dir)
    plot_v20_token_weighted_suites(cfg, run_dir)
    plot_v20_training(cfg, run_dir)
    plot_v20_accuracy(run_dir)
    plot_v20_analysis(run_dir)


def _variant(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not result.empty:
        result["variant"] = result["position_encoding"].str.upper() + "/" + result["mode"]
    return result


def _line_plot(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    path: Path,
    hue: str = "variant",
    style: str | None = None,
    ylim: tuple[float, float] | None = None,
    phase_boundary: int | None = None,
) -> None:
    if frame.empty or y not in frame:
        return
    figure, axis = plt.subplots(figsize=(10.8, 4.8), layout="constrained")
    sns.lineplot(data=frame, x=x, y=y, hue=hue, style=style, marker="o", ax=axis)
    axis.set_title(title)
    axis_labels = {
        "step": "Training step",
        "needle_attention_enrichment": "Target attention enrichment (× uniform)",
        "top_n_needle_recall": "Top-n target recall",
        "trace_readout_mass": "Attention mass on trace tokens",
        "correct_top1_minus_chance": "Correct-k top-1 minus 1/count",
        "r2": "Held-out cross-site R²",
        "pc1_label_r2": "Count-label R² from centroid PC1",
        "linear_cka": "Linear CKA vs. final checkpoint",
        "expected_count_delta": "Expected-count change",
    }
    axis.set_xlabel(axis_labels.get(x, x.replace("_", " ")))
    axis.set_ylabel(axis_labels.get(y, y.replace("_", " ")))
    if phase_boundary is not None:
        axis.axvline(
            phase_boundary, color="black", linestyle=":", linewidth=1.2,
            label="task-output-only boundary",
        )
    if ylim is not None:
        axis.set_ylim(*ylim)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles,
            labels,
            fontsize=7.5,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            ncol=2 if len(labels) > 8 else 1,
        )
    _save(figure, path)


def plot_v20_runtime(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    frame = _read(run_dir / "tables" / "runtime_events.csv")
    if frame.empty or "duration_seconds" not in frame:
        return
    complete = frame[(frame.status == "complete") & frame.duration_seconds.notna()].copy()
    if complete.empty:
        return
    summary = complete.groupby(["scope", "block"], as_index=False).duration_seconds.sum()
    summary["label"] = summary.scope + ": " + summary.block
    summary = summary.sort_values("duration_seconds", ascending=True).tail(24)
    figure, axis = plt.subplots(figsize=(10, max(4.5, 0.3 * len(summary))))
    axis.barh(summary.label, summary.duration_seconds, color="#466f9f")
    axis.set_xlabel("cumulative wall time (seconds)")
    axis.set_title("Runtime breakdown by instrumented block")
    _save(figure, run_dir / "figures" / "runtime_breakdown.png")


def plot_v20_checkpoint_dynamics(run_dir: str | Path) -> None:
    """Render revision-5 checkpoint plots; every plot tolerates disabled metric families."""

    run_dir = Path(run_dir)
    figures = run_dir / "figures"
    dynamics_figures = (
        "checkpoint_attention_retrieval_emergence.png",
        "checkpoint_answer_routing.png",
        "checkpoint_nonthinking_needle_coverage.png",
        "checkpoint_head_role_stability.png",
        "checkpoint_final_count_probe_heatmap.png",
        "checkpoint_trace_progress_probe_heatmap.png",
        "checkpoint_cross_site_counter_transfer.png",
        "checkpoint_counterfactual_trace_readout.png",
        "checkpoint_state_geometry_emergence.png",
        "checkpoint_representation_stability.png",
        "checkpoint_mechanism_overview.png",
        "checkpoint_ordered_trace_retrieval.png",
    )
    for name in dynamics_figures:
        (figures / name).unlink(missing_ok=True)
    config_path = run_dir / "config.json"
    phase_boundary = None
    if config_path.exists():
        phase_boundary = int(
            json.loads(config_path.read_text(encoding="utf-8")).get(
                "max_steps_for_language_pred", 0
            )
        )
    attention = _variant(_read(run_dir / "tables" / "checkpoint_attention_summary.csv"))
    attention_count = _variant(_read(run_dir / "tables" / "checkpoint_attention_by_count.csv"))
    attention_k = _variant(_read(run_dir / "tables" / "checkpoint_attention_by_k.csv"))
    heads = _variant(_read(run_dir / "tables" / "checkpoint_head_stability.csv"))
    probes = _variant(_read(run_dir / "tables" / "checkpoint_state_probe_summary.csv"))
    cross = _variant(_read(run_dir / "tables" / "checkpoint_state_cross_site.csv"))
    counter = _variant(_read(run_dir / "tables" / "checkpoint_counterfactual_trace_readout.csv"))
    geometry = _variant(_read(run_dir / "tables" / "checkpoint_state_geometry.csv"))
    similarity = _variant(_read(run_dir / "tables" / "checkpoint_state_similarity.csv"))
    ar = _variant(_read(run_dir / "tables" / "checkpoint_dynamics_autoregressive.csv"))

    final_attention = attention[
        (attention.get("diagnostic_split") == "heldout_reporting")
        & (attention.get("query_kind") == "final_answer")
    ] if not attention.empty else attention
    _line_plot(
        final_attention, x="step", y="needle_attention_enrichment",
        title="Emergence of final-answer retrieval from prompt needles",
        path=figures / "checkpoint_attention_retrieval_emergence.png",
        phase_boundary=phase_boundary,
    )
    _line_plot(
        final_attention, x="step", y="trace_readout_mass",
        title="Final-answer routing through the generated reasoning trace",
        path=figures / "checkpoint_answer_routing.png",
        phase_boundary=phase_boundary,
    )
    coverage = attention_count[
        (attention_count.get("diagnostic_split") == "heldout_reporting")
        & (attention_count.get("query_kind") == "final_answer")
        & (attention_count.get("mode") == "nonthinking")
    ] if not attention_count.empty else attention_count
    _line_plot(
        coverage, x="step", y="top_n_needle_recall", style="count",
        title="Nonthinking prompt-needle coverage (line style = true count)",
        path=figures / "checkpoint_nonthinking_needle_coverage.png", ylim=(0, 1.03),
        phase_boundary=phase_boundary,
    )
    head_curves = pd.DataFrame()
    if not heads.empty and "heldout_best_current_value" in heads:
        head_curves = heads.melt(
            id_vars=["variant", "step", "role"],
            value_vars=["heldout_value", "heldout_best_current_value"],
            var_name="selection", value_name="metric_value",
        )
        head_curves["curve"] = head_curves.role + " | " + head_curves.selection.map({
            "heldout_value": "fixed final-selected head",
            "heldout_best_current_value": "best current head",
        })
    _line_plot(
        head_curves, x="step", y="metric_value", style="curve",
        title="Fixed final-selected heads versus the best head at each checkpoint",
        path=figures / "checkpoint_head_role_stability.png",
        phase_boundary=phase_boundary,
    )

    final_probe = probes[(probes.get("site") == "final_answer") & (probes.get("context") == "teacher_forced")]
    if not final_probe.empty:
        pivot = final_probe.pivot_table(
            index=["variant", "layer"], columns="step", values="nearest_centroid_accuracy", aggfunc="mean"
        )
        figure, axis = plt.subplots(figsize=(10, max(4, 0.25 * len(pivot))))
        sns.heatmap(pivot, vmin=0, vmax=1, cmap="viridis", ax=axis)
        axis.set_title("Final-count nearest-centroid decoding across checkpoints")
        _save(figure, figures / "checkpoint_final_count_probe_heatmap.png")

    trace_probe = probes[(probes.get("site").isin(["trace_index", "trace_marker"])) & (probes.get("context") == "teacher_forced")]
    if not trace_probe.empty:
        pivot = trace_probe.pivot_table(
            index=["variant", "site", "layer"], columns="step", values="ridge_r2", aggfunc="mean"
        )
        figure, axis = plt.subplots(figsize=(10, max(4, 0.22 * len(pivot))))
        sns.heatmap(pivot, vmin=-0.2, vmax=1, center=0, cmap="vlag", ax=axis)
        axis.set_title("Trace-progress ridge decoding across checkpoints")
        _save(figure, figures / "checkpoint_trace_progress_probe_heatmap.png")

    _line_plot(
        cross, x="step", y="r2", style="direction",
        title="Transfer between trace-progress and final-count directions",
        path=figures / "checkpoint_cross_site_counter_transfer.png",
        phase_boundary=phase_boundary,
    )
    shortened = counter[counter.get("condition") == "remove_final_pair"] if not counter.empty else counter
    _line_plot(
        shortened, x="step", y="delta_gold_logit_margin_vs_count_minus_one", style="layer",
        title="Effect of removing the final trace index/marker pair",
        path=figures / "checkpoint_counterfactual_trace_readout.png",
        phase_boundary=phase_boundary,
    )
    state_geometry = geometry[
        (geometry.get("site") == "final_answer") & (geometry.get("context") == "teacher_forced")
    ] if not geometry.empty else geometry
    _line_plot(
        state_geometry, x="step", y="pc1_label_r2", style="layer",
        title="Emergence of ordered count geometry at the answer site",
        path=figures / "checkpoint_state_geometry_emergence.png", ylim=(-0.03, 1.03),
        phase_boundary=phase_boundary,
    )
    final_similarity = similarity[
        (similarity.get("reference") == "final") & (similarity.get("site") == "final_answer")
    ] if not similarity.empty else similarity
    _line_plot(
        final_similarity, x="step", y="linear_cka", style="layer",
        title="Representation similarity to the final checkpoint",
        path=figures / "checkpoint_representation_stability.png", ylim=(-0.03, 1.03),
        phase_boundary=phase_boundary,
    )

    if not ar.empty:
        overview = ar.groupby(["variant", "step"], as_index=False).ar_accuracy.mean()
        figure, axes_grid = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes_grid.ravel()
        sns.lineplot(data=overview, x="step", y="ar_accuracy", hue="variant", marker="o", ax=axes[0])
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].set_xlabel("Training step")
        axes[0].set_ylabel("AR exact-count accuracy")
        if phase_boundary is not None:
            axes[0].axvline(phase_boundary, color="black", linestyle=":", linewidth=1.2)
        axes[0].set_title("Autoregressive count accuracy")
        if not final_probe.empty:
            summary = final_probe.groupby(["variant", "step"], as_index=False).nearest_centroid_accuracy.max()
            sns.lineplot(data=summary, x="step", y="nearest_centroid_accuracy", hue="variant", marker="o", ax=axes[1])
            axes[1].set_ylim(-0.03, 1.03)
            axes[1].set_xlabel("Training step")
            axes[1].set_ylabel("Best-layer nearest-centroid accuracy")
            if phase_boundary is not None:
                axes[1].axvline(phase_boundary, color="black", linestyle=":", linewidth=1.2)
        axes[1].set_title("Best-layer final-count decoding")
        retrieval = heads[heads.role == "needle_retrieval"] if not heads.empty else heads
        if not retrieval.empty:
            sns.lineplot(
                data=retrieval, x="step", y="heldout_value", hue="variant",
                marker="o", ax=axes[2],
            )
        axes[2].set_title("Fixed-head prompt-needle retrieval")
        axes[2].set_xlabel("Training step")
        axes[2].set_ylabel("Target attention enrichment (× uniform)")
        routing = pd.DataFrame()
        if not final_attention.empty:
            routing = final_attention.groupby(
                ["variant", "step"], as_index=False
            ).trace_readout_mass.mean()
        if not routing.empty:
            sns.lineplot(
                data=routing, x="step", y="trace_readout_mass", hue="variant",
                marker="o", ax=axes[3],
            )
        axes[3].set_title("Final-answer attention on trace tokens")
        axes[3].set_xlabel("Training step")
        axes[3].set_ylabel("Trace-token attention mass")
        if phase_boundary is not None:
            for axis in axes[2:]:
                axis.axvline(phase_boundary, color="black", linestyle=":", linewidth=1.2)
        legend_handles, legend_labels = axes[0].get_legend_handles_labels()
        for axis in axes:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
        if legend_handles:
            figure.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.925),
                ncol=2,
                frameon=False,
                title="Model",
            )
        figure.suptitle("Checkpoint mechanism overview", y=0.985)
        figure.subplots_adjust(
            left=0.11, right=0.98, bottom=0.08, top=0.82, hspace=0.48, wspace=0.32
        )
        _save(figure, figures / "checkpoint_mechanism_overview.png")

    # A compact ordered-trace plot is useful even though it is not part of the
    # required headline figure list.
    ordered = attention_k[attention_k.get("diagnostic_split") == "heldout_reporting"] if not attention_k.empty else attention_k
    _line_plot(
        ordered, x="step", y="correct_top1_minus_chance", style="query_k",
        title="Ordered trace-to-prompt retrieval above chance",
        path=figures / "checkpoint_ordered_trace_retrieval.png",
        phase_boundary=phase_boundary,
    )
    plot_v20_runtime(run_dir)
