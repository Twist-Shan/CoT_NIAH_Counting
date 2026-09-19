from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_round1_plots(train_log: pd.DataFrame, eval_df: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    if not train_log.empty:
        plt.figure(figsize=(9, 4.5))
        sns.lineplot(data=train_log, x="step", y="train_total_loss", hue="model_type", style="seed", errorbar=None)
        plt.title("Round 1 training loss by step")
        _save(figures_dir / "round1_train_loss_by_step.png")
    if not eval_df.empty:
        generated = eval_df[eval_df["eval_mode"].isin(["direct", "generated_trace"])]
        agg = generated.groupby(
            ["model_type", "seed", "checkpoint_step", "seq_len_eval", "count_bin"], as_index=False
        )["final_accuracy"].mean()
        plt.figure(figsize=(11, 5))
        sns.lineplot(
            data=agg,
            x="checkpoint_step",
            y="final_accuracy",
            hue="seq_len_eval",
            style="model_type",
            markers=True,
            errorbar=None,
        )
        plt.ylim(-0.03, 1.03)
        plt.title("Round 1 final accuracy by step and eval length")
        _save(figures_dir / "round1_final_accuracy_by_step_and_seq_len.png")

        final_step = generated["checkpoint_step"].max()
        final = generated[generated["checkpoint_step"].eq(final_step)]
        by_count = final.groupby(["model_type", "seq_len_eval", "count"], as_index=False)["final_accuracy"].mean()
        plt.figure(figsize=(10, 4.8))
        sns.lineplot(data=by_count, x="count", y="final_accuracy", hue="model_type", style="seq_len_eval", markers=True)
        plt.ylim(-0.03, 1.03)
        plt.title(f"Round 1 final checkpoint accuracy by exact count (step {final_step})")
        _save(figures_dir / "round1_accuracy_by_count_final.png")

        pivot = final.groupby(["model_type", "seq_len_eval"], as_index=False)["final_accuracy"].mean()
        heat = pivot.pivot(index="model_type", columns="seq_len_eval", values="final_accuracy")
        plt.figure(figsize=(7, 3))
        sns.heatmap(heat, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Round 1 accuracy heatmap: model x eval length")
        _save(figures_dir / "round1_accuracy_heatmap_count_x_seq_len.png")

        trace = eval_df[(eval_df["model_type"].eq("thinking")) & (eval_df["eval_mode"].eq("generated_trace"))]
        if not trace.empty:
            trace_agg = trace.groupby(["checkpoint_step", "seq_len_eval"], as_index=False)[
                ["trace_exact_rate", "trace_marker_recall", "invalid_generation_rate"]
            ].mean()
            plt.figure(figsize=(10, 4.8))
            long = trace_agg.melt(
                id_vars=["checkpoint_step", "seq_len_eval"],
                value_vars=["trace_exact_rate", "trace_marker_recall", "invalid_generation_rate"],
                var_name="metric",
                value_name="value",
            )
            sns.lineplot(data=long, x="seq_len_eval", y="value", hue="metric", style="checkpoint_step", markers=True, errorbar=None)
            plt.ylim(-0.03, 1.03)
            plt.title("Round 1 thinking trace metrics by eval length")
            _save(figures_dir / "round1_trace_metrics_by_seq_len.png")

            oracle = eval_df[eval_df["model_type"].eq("thinking") & eval_df["eval_mode"].isin(["generated_trace", "oracle_trace"])]
            oracle_agg = oracle.groupby(["eval_mode", "seq_len_eval"], as_index=False)["final_accuracy"].mean()
            plt.figure(figsize=(7, 4))
            sns.barplot(data=oracle_agg, x="seq_len_eval", y="final_accuracy", hue="eval_mode", errorbar=None)
            plt.ylim(0, 1)
            plt.title("Round 1 generated trace vs oracle trace final accuracy")
            _save(figures_dir / "round1_oracle_vs_generated_trace_accuracy.png")


def make_round2_plots(corrupt_df: pd.DataFrame, figures_dir: Path) -> None:
    if corrupt_df.empty:
        return
    plt.figure(figsize=(11, 4.8))
    sns.barplot(data=corrupt_df, x="corruption_type", y="correct_prompt_count", hue="seq_len_eval", errorbar=None)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.title("Round 2 corrupted trace: accuracy against prompt count")
    _save(figures_dir / "round2_corruption_accuracy_by_type.png")

    follow_cols = [col for col in corrupt_df.columns if col.startswith("follows_")]
    follow_long = corrupt_df.melt(
        id_vars=["corruption_type"],
        value_vars=follow_cols,
        var_name="rule",
        value_name="rate",
    )
    plt.figure(figsize=(11, 4.8))
    sns.barplot(data=follow_long, x="corruption_type", y="rate", hue="rule", errorbar=None)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.title("Round 2 follow-rule breakdown")
    _save(figures_dir / "round2_follow_rule_breakdown.png")

    for target, filename in [
        ("prompt_count", "round2_confusion_pred_vs_prompt_count.png"),
        ("trace_pair_count", "round2_confusion_pred_vs_trace_pair_count.png"),
        ("last_index_value", "round2_confusion_pred_vs_last_index.png"),
    ]:
        sub = corrupt_df.dropna(subset=[target])
        if sub.empty:
            continue
        mat = pd.crosstab(sub[target], sub["pred_count"], normalize="index")
        plt.figure(figsize=(6.5, 5))
        sns.heatmap(mat, vmin=0, vmax=1, cmap="viridis")
        plt.title(f"Round 2 prediction vs {target}")
        _save(figures_dir / filename)

    by_len = corrupt_df.groupby(["corruption_type", "seq_len_eval"], as_index=False)["correct_prompt_count"].mean()
    plt.figure(figsize=(9, 4.5))
    sns.lineplot(data=by_len, x="seq_len_eval", y="correct_prompt_count", hue="corruption_type", marker="o", errorbar=None)
    plt.ylim(0, 1)
    plt.title("Round 2 corruption robustness by eval length")
    _save(figures_dir / "round2_corruption_by_seq_len.png")


def make_round3_plots(
    probe_df: pd.DataFrame,
    attention_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    if not probe_df.empty:
        sub = probe_df[probe_df["target_type"].isin(["final_count", "prefix_count"])]
        plt.figure(figsize=(11, 4.8))
        sns.barplot(data=sub, x="anchor_type", y="test_accuracy", hue="layer", errorbar=None)
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1)
        plt.title("Round 3 probe accuracy by layer and anchor")
        _save(figures_dir / "round3_probe_accuracy_layer_by_anchor.png")

        plt.figure(figsize=(11, 4.8))
        sns.barplot(data=sub, x="anchor_type", y="r2", hue="layer", errorbar=None)
        plt.xticks(rotation=35, ha="right")
        plt.title("Round 3 ridge R2 by layer and anchor")
        _save(figures_dir / "round3_probe_r2_layer_by_anchor.png")

        plt.figure(figsize=(6.2, 4.8))
        sns.scatterplot(data=sub, x="position_only_accuracy", y="test_accuracy", hue="model_type", style="target_type")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.title("Round 3 probe vs position baseline")
        _save(figures_dir / "round3_probe_vs_position_baseline.png")

    if not attention_df.empty:
        board = attention_df.groupby(["model_type", "layer", "head", "query_anchor"], as_index=False)[
            ["correct_top1_rate", "diagonal_dominance", "needle_mass", "top_n_recall"]
        ].mean()
        plot_col = "correct_top1_rate"
        board["score"] = board[["correct_top1_rate", "diagonal_dominance", "needle_mass"]].mean(axis=1, skipna=True)
        plt.figure(figsize=(10, 4.8))
        sns.barplot(data=board.sort_values("score", ascending=False).head(24), x="query_anchor", y="score", hue="model_type", errorbar=None)
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 1)
        plt.title("Round 3 attention head leaderboard")
        _save(figures_dir / "round3_attention_head_leaderboard.png")

        thinking = attention_df[attention_df["model_type"].eq("thinking")]
        if not thinking.empty:
            mat = thinking.pivot_table(index="layer", columns="head", values="correct_top1_rate", aggfunc="mean")
            plt.figure(figsize=(5.4, 4.2))
            sns.heatmap(mat, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis")
            plt.title("Round 3 thinking trace-to-prompt top1")
            _save(figures_dir / "round3_thinking_trace_to_prompt_heatmap_best_head.png")

        non = attention_df[attention_df["model_type"].eq("non_thinking")]
        if not non.empty:
            mat = non.pivot_table(index="layer", columns="head", values="top_n_recall", aggfunc="mean")
            plt.figure(figsize=(5.4, 4.2))
            sns.heatmap(mat, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis")
            plt.title("Round 3 non-thinking <Ans> top-n retrieval")
            _save(figures_dir / "round3_nonthinking_ans_to_prompt_attention.png")

        by_bin = attention_df.groupby(["model_type", "count_bin"], as_index=False)[["needle_mass", "needle_to_noise_ratio"]].mean()
        plt.figure(figsize=(7, 4.5))
        sns.barplot(data=by_bin, x="count_bin", y="needle_mass", hue="model_type", errorbar=None)
        plt.title("Round 3 attention needle mass by count bin")
        _save(figures_dir / "round3_attention_metrics_by_count_bin.png")

        by_len = attention_df.groupby(["model_type", "seq_len_eval"], as_index=False)[["needle_mass", "needle_to_noise_ratio"]].mean()
        plt.figure(figsize=(7, 4.5))
        sns.lineplot(data=by_len, x="seq_len_eval", y="needle_mass", hue="model_type", marker="o", errorbar=None)
        plt.title("Round 3 attention needle mass by eval length")
        _save(figures_dir / "round3_attention_metrics_by_seq_len.png")

    if not ablation_df.empty:
        plt.figure(figsize=(8, 4.5))
        sns.barplot(data=ablation_df, x="count_bin", y="delta_final_accuracy", hue="model_type", errorbar=None)
        plt.title("Round 3 single-head ablation: delta final accuracy")
        _save(figures_dir / "round3_head_ablation_effects.png")

        trace = ablation_df[ablation_df["model_type"].eq("thinking")]
        if not trace.empty:
            plt.figure(figsize=(8, 4.5))
            sns.barplot(data=trace, x="count_bin", y="delta_trace_exact", hue="eval_mode", errorbar=None)
            plt.title("Round 3 single-head ablation: delta trace exact")
            _save(figures_dir / "round3_attention_masking_effects.png")
