from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _empty_plot(path: Path, title: str) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.title(title)
    plt.text(0.5, 0.5, "No data", ha="center", va="center")
    plt.xticks([])
    plt.yticks([])
    _save(path)


def make_plots(run_dir: Path) -> None:
    figures = run_dir / "figures"
    tables = run_dir / "tables"
    sns.set_theme(style="whitegrid", context="notebook")
    train = pd.read_csv(tables / "train_log.csv") if (tables / "train_log.csv").exists() else pd.DataFrame()
    eval_df = pd.read_csv(tables / "eval_by_step.csv") if (tables / "eval_by_step.csv").exists() else pd.DataFrame()
    switch = pd.read_csv(tables / "mode_switch.csv") if (tables / "mode_switch.csv").exists() else pd.DataFrame()
    sim = pd.read_csv(tables / "mode_hidden_similarity.csv") if (tables / "mode_hidden_similarity.csv").exists() else pd.DataFrame()
    attn = pd.read_csv(tables / "attention_metrics.csv") if (tables / "attention_metrics.csv").exists() else pd.DataFrame()
    examples = pd.read_csv(tables / "eval_examples.csv") if (tables / "eval_examples.csv").exists() else pd.DataFrame()

    if not train.empty:
        long = train.melt(
            id_vars=["step"],
            value_vars=["loss_total", "loss_thinking_trace", "loss_thinking_final_count", "loss_nonthinking_close", "loss_nonthinking_final_count"],
            var_name="loss_name",
            value_name="loss",
        )
        plt.figure(figsize=(9, 4.5))
        sns.lineplot(data=long, x="step", y="loss", hue="loss_name", marker="o", errorbar=None)
        plt.title("v5 train loss by step and mode component")
        _save(figures / "train_loss_by_step_and_mode.png")
    else:
        _empty_plot(figures / "train_loss_by_step_and_mode.png", "v5 train loss")

    if not eval_df.empty:
        by_step = eval_df.groupby(["step", "mode"], as_index=False)["final_accuracy"].mean()
        plt.figure(figsize=(7.5, 4.2))
        sns.lineplot(data=by_step, x="step", y="final_accuracy", hue="mode", marker="o", errorbar=None)
        plt.ylim(-0.03, 1.03)
        plt.title("v5 final accuracy by step and mode")
        _save(figures / "final_accuracy_by_step_mode.png")

        final_step = int(eval_df["step"].max())
        final = eval_df[eval_df["step"].eq(final_step)]
        plt.figure(figsize=(8, 4.2))
        sns.lineplot(data=final, x="count", y="final_accuracy", hue="mode", marker="o", errorbar=None)
        plt.ylim(-0.03, 1.03)
        plt.title(f"v5 final accuracy by count at step {final_step}")
        _save(figures / "final_accuracy_by_count_mode.png")

        trace = final[final["mode"].eq("thinking")]
        if not trace.empty:
            trace_long = trace.melt(
                id_vars=["count"],
                value_vars=["trace_exact", "trace_marker_precision", "trace_marker_recall", "premature_close_rate", "missing_close_rate"],
                var_name="metric",
                value_name="value",
            )
            plt.figure(figsize=(10, 4.5))
            sns.lineplot(data=trace_long, x="count", y="value", hue="metric", marker="o", errorbar=None)
            plt.ylim(-0.03, 1.03)
            plt.title("v5 thinking trace metrics by count")
            _save(figures / "trace_metrics_by_count.png")
        else:
            _empty_plot(figures / "trace_metrics_by_count.png", "v5 trace metrics")
    else:
        _empty_plot(figures / "final_accuracy_by_step_mode.png", "v5 final accuracy by step")
        _empty_plot(figures / "final_accuracy_by_count_mode.png", "v5 final accuracy by count")
        _empty_plot(figures / "trace_metrics_by_count.png", "v5 trace metrics")

    if not switch.empty:
        switch_step = switch.groupby(["step", "mode"], as_index=False)["argmax_is_desired"].mean()
        plt.figure(figsize=(8, 4.2))
        sns.lineplot(data=switch_step, x="step", y="argmax_is_desired", hue="mode", marker="o", errorbar=None)
        plt.ylim(-0.03, 1.03)
        plt.title("v5 explicit mode-switch accuracy")
        _save(figures / "mode_switch_accuracy_by_step.png")
    else:
        _empty_plot(figures / "mode_switch_accuracy_by_step.png", "v5 mode-switch accuracy")

    for mode, filename in [("thinking", "confusion_matrix_thinking.png"), ("nonthinking", "confusion_matrix_nonthinking.png")]:
        if not examples.empty and mode in set(examples["mode"]):
            final_step = examples["step"].max()
            sub = examples[(examples["mode"].eq(mode)) & (examples["step"].eq(final_step))]
            mat = pd.crosstab(sub["count"], sub["pred_count"], normalize="index")
            plt.figure(figsize=(5.5, 4.5))
            sns.heatmap(mat, vmin=0, vmax=1, cmap="viridis", annot=False)
            plt.title(f"v5 {mode} confusion matrix")
            _save(figures / filename)
        else:
            _empty_plot(figures / filename, f"v5 {mode} confusion matrix")

    if not sim.empty:
        by_layer = sim.groupby("layer", as_index=False)["cosine_similarity"].mean()
        plt.figure(figsize=(6.5, 3.8))
        sns.lineplot(data=by_layer, x="layer", y="cosine_similarity", marker="o", errorbar=None)
        plt.title("v5 thinking vs non-thinking hidden similarity")
        _save(figures / "mode_hidden_similarity.png")
    else:
        _empty_plot(figures / "mode_hidden_similarity.png", "v5 hidden similarity")

    if not attn.empty and "thinking" in set(attn["mode"]):
        thinking = attn[attn["mode"].eq("thinking")]
        mat = thinking.pivot_table(index="layer", columns="head", values="diagonal_dominance", aggfunc="mean")
        plt.figure(figsize=(5.5, 4.2))
        sns.heatmap(mat, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis")
        plt.title("v5 attention trace-to-prompt best-head diagnostic")
        _save(figures / "attention_trace_to_prompt_best_head.png")
    else:
        _empty_plot(figures / "attention_trace_to_prompt_best_head.png", "v5 attention trace-to-prompt")
