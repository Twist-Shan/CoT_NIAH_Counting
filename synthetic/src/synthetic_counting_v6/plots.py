from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("default")


def _lineplot(data: pd.DataFrame, x: str, y: str, hue: str | None = None, marker: str | None = None, ax=None) -> None:
    ax = ax or plt.gca()
    if data.empty:
        return
    if hue is None:
        sub = data.sort_values(x)
        ax.plot(sub[x], sub[y], marker=marker)
        return
    for label, sub in data.groupby(hue, sort=False):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], marker=marker, label=str(label))
    ax.legend()


def _barplot(data: pd.DataFrame, x: str, y: str, ax=None) -> None:
    ax = ax or plt.gca()
    if data.empty:
        return
    labels = [str(v) for v in data[x].tolist()]
    positions = np.arange(len(labels))
    ax.bar(positions, data[y].to_numpy(dtype=float))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def _heatmap(
    data: pd.DataFrame,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    annot: bool = False,
    fmt: str = ".2f",
    ax=None,
) -> None:
    ax = ax or plt.gca()
    values = data.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels([str(c) for c in data.columns], rotation=0)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels([str(i) for i in data.index])
    if annot:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", color="white" if values[i, j] < 0.45 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _placeholder(path: Path, title: str, text: str = "No data") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14)
    ax.axis("off")
    _save(fig, path)


def plot_train_loss(run_dir: Path) -> None:
    path = run_dir / "metrics" / "metrics_train.csv"
    out = run_dir / "plots" / "train_loss_vs_step.png"
    if not path.exists():
        _placeholder(out, "Training loss over steps")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    _lineplot(data=df, x="step", y="train_completion_loss", hue="model_type", ax=ax)
    ax.set_title("Training masked completion loss; raw loss lengths differ across models")
    ax.set_xlabel("training step")
    ax.set_ylabel("masked completion loss")
    _save(fig, out)


def plot_eval_final_loss(run_dir: Path) -> None:
    path = run_dir / "metrics" / "metrics_eval_by_bin.csv"
    out = run_dir / "plots" / "eval_final_answer_loss_vs_step.png"
    if not path.exists():
        _placeholder(out, "Test final-answer loss over steps")
        return
    df = pd.read_csv(path)
    df = df[
        ((df["model_type"] == "non_thinking") & (df["eval_mode"] == "direct"))
        | ((df["model_type"] == "thinking_sep_trace") & (df["eval_mode"] == "oracle_trace_final_readout"))
    ]
    if df.empty:
        _placeholder(out, "Test final-answer loss over steps")
        return
    df = df.groupby(["step", "model_type"], as_index=False)["eval_final_answer_loss"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    _lineplot(data=df, x="step", y="eval_final_answer_loss", hue="model_type", marker="o", ax=ax)
    ax.set_title("Comparable final-answer CE: direct non-thinking vs oracle-trace thinking")
    ax.set_xlabel("training step")
    ax.set_ylabel("final-answer cross-entropy")
    _save(fig, out)


def plot_eval_accuracy_by_bin(run_dir: Path) -> None:
    path = run_dir / "metrics" / "metrics_eval_by_bin.csv"
    out = run_dir / "plots" / "eval_accuracy_by_bin_vs_step.png"
    if not path.exists():
        _placeholder(out, "Test accuracy by count bin")
        return
    df = pd.read_csv(path)
    df = df[df["eval_mode"].isin(["direct", "generated_trace", "oracle_trace_final_readout"])].copy()
    df["series"] = df["model_type"] + " / " + df["eval_mode"] + " / " + df["count_bin"]
    fig, ax = plt.subplots(figsize=(11, 6))
    _lineplot(data=df, x="step", y="accuracy", hue="series", marker="o", ax=ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Exact final-count accuracy by low/mid/high count bin")
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    _save(fig, out)


def plot_final_accuracy_by_count(run_dir: Path) -> None:
    path = run_dir / "metrics" / "metrics_eval_by_count.csv"
    out = run_dir / "plots" / "final_accuracy_by_count.png"
    if not path.exists():
        _placeholder(out, "Final checkpoint accuracy by count")
        return
    df = pd.read_csv(path)
    step = df["step"].max()
    df = df[df["step"] == step].copy()
    df["series"] = df["model_type"] + " / " + df["eval_mode"]
    fig, ax = plt.subplots(figsize=(9, 5))
    _lineplot(data=df, x="count", y="accuracy", hue="series", marker="o", ax=ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Final checkpoint exact-count accuracy by gold count (step {step})")
    ax.set_xlabel("gold count")
    ax.set_ylabel("accuracy")
    _save(fig, out)


def plot_accuracy_heatmaps(run_dir: Path) -> None:
    path = run_dir / "metrics" / "metrics_eval_by_count.csv"
    if not path.exists():
        for name in [
            "accuracy_heatmap_by_count_and_step_non_thinking.png",
            "accuracy_heatmap_by_count_and_step_thinking_generated_trace.png",
            "accuracy_heatmap_by_count_and_step_thinking_oracle_trace.png",
        ]:
            _placeholder(run_dir / "plots" / name, name)
        return
    df = pd.read_csv(path)
    specs = [
        ("non_thinking", "direct", "accuracy_heatmap_by_count_and_step_non_thinking.png", "Accuracy heatmap: non-thinking direct"),
        (
            "thinking_sep_trace",
            "generated_trace",
            "accuracy_heatmap_by_count_and_step_thinking_generated_trace.png",
            "Accuracy heatmap: separator-trace generated",
        ),
        (
            "thinking_sep_trace",
            "oracle_trace_final_readout",
            "accuracy_heatmap_by_count_and_step_thinking_oracle_trace.png",
            "Accuracy heatmap: separator-trace oracle final readout",
        ),
    ]
    for model_type, eval_mode, filename, title in specs:
        sub = df[(df["model_type"] == model_type) & (df["eval_mode"] == eval_mode)]
        out = run_dir / "plots" / filename
        if sub.empty:
            _placeholder(out, title)
            continue
        pivot = sub.pivot(index="count", columns="step", values="accuracy")
        fig, ax = plt.subplots(figsize=(10, 5))
        _heatmap(pivot, vmin=0, vmax=1, cmap="viridis", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.set_ylabel("gold count")
        _save(fig, out)


def plot_trace_quality(run_dir: Path) -> None:
    path = run_dir / "metrics" / "metrics_eval_by_count.csv"
    if not path.exists():
        for name in [
            "trace_exact_by_count.png",
            "trace_marker_precision_recall_by_count.png",
            "trace_delimiter_count_accuracy_by_count.png",
            "premature_close_missing_close_by_count.png",
        ]:
            _placeholder(run_dir / "plots" / name, name)
        return
    df = pd.read_csv(path)
    step = df["step"].max()
    sub = df[(df["step"] == step) & (df["model_type"] == "thinking_sep_trace") & (df["eval_mode"] == "generated_trace")]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    _lineplot(data=sub, x="count", y="trace_exact_match_rate", marker="o", ax=ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Generated separator trace exact match by gold count")
    ax.set_xlabel("gold count")
    ax.set_ylabel("trace exact match")
    _save(fig, run_dir / "plots" / "trace_exact_by_count.png")

    long = sub.melt(
        id_vars=["count"],
        value_vars=["trace_marker_precision", "trace_marker_recall"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    _lineplot(data=long, x="count", y="value", hue="metric", marker="o", ax=ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Generated trace marker precision/recall by gold count")
    ax.set_xlabel("gold count")
    ax.set_ylabel("rate")
    _save(fig, run_dir / "plots" / "trace_marker_precision_recall_by_count.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    _lineplot(data=sub, x="count", y="trace_delimiter_count_accuracy", marker="o", ax=ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Generated trace <Sep> count accuracy by gold count")
    ax.set_xlabel("gold count")
    ax.set_ylabel("<Sep> count accuracy")
    _save(fig, run_dir / "plots" / "trace_delimiter_count_accuracy_by_count.png")

    close = sub.melt(
        id_vars=["count"],
        value_vars=["premature_close_rate", "missing_close_rate"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    _lineplot(data=close, x="count", y="value", hue="metric", marker="o", ax=ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Generated trace close-token failure modes by gold count")
    ax.set_xlabel("gold count")
    ax.set_ylabel("rate")
    _save(fig, run_dir / "plots" / "premature_close_missing_close_by_count.png")


def plot_probe_metrics(run_dir: Path) -> None:
    path = run_dir / "probes" / "probe_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    specs = [
        ("non_thinking", "final_count", "probe_final_count_accuracy_heatmap_non_thinking.png", "Final-count probe accuracy: non-thinking"),
        (
            "thinking_sep_trace",
            "final_count",
            "probe_final_count_accuracy_heatmap_thinking_sep_trace.png",
            "Final-count probe accuracy: separator-trace thinking",
        ),
        (
            "thinking_sep_trace",
            "prefix_count",
            "probe_prefix_count_accuracy_heatmap_thinking_sep_trace.png",
            "Prefix-count probe accuracy: separator-trace thinking",
        ),
    ]
    for model_type, label_type, filename, title in specs:
        sub = df[(df["model_type"] == model_type) & (df["label_type"] == label_type)]
        out = run_dir / "probes" / filename
        if sub.empty:
            _placeholder(out, title)
            continue
        pivot = sub.pivot_table(index="anchor_type", columns="layer", values="probe_accuracy", aggfunc="max")
        fig, ax = plt.subplots(figsize=(9, 5))
        _heatmap(pivot, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2f", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("layer (-1 = embedding)")
        ax.set_ylabel("anchor")
        _save(fig, out)
    sub = df[(df["model_type"] == "thinking_sep_trace") & (df["label_type"] == "prefix_count") & (df["anchor_type"] == "sep_token_k")]
    if not sub.empty:
        best = sub.groupby("layer", as_index=False)[["probe_accuracy", "position_only_accuracy"]].max()
        best["probe_minus_position"] = best["probe_accuracy"] - best["position_only_accuracy"]
        fig, ax = plt.subplots(figsize=(8, 4))
        _barplot(data=best, x="layer", y="probe_minus_position", ax=ax)
        ax.set_title("sep_token_k prefix probe advantage over position baseline")
        ax.set_xlabel("layer")
        ax.set_ylabel("probe accuracy - position baseline")
        _save(fig, run_dir / "probes" / "probe_sep_token_prefix_probe_minus_position_baseline.png")


def plot_attention_metrics(run_dir: Path) -> None:
    path = run_dir / "attention" / "attention_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    think = df[(df["model_type"] == "thinking_sep_trace") & (df["query_anchor"] == "sep_token_k") & (df["subset"] == "all_examples")]
    for metric, filename, title in [
        ("diagonal_dominance", "attention_thinking_sep_diagonal_dominance_by_layer_head.png", "Thinking sep_token_k diagonal dominance"),
        ("correct_top1_rate", "attention_thinking_sep_correct_top1_by_layer_head.png", "Thinking sep_token_k correct top-1 retrieval"),
        ("needle_attention_mass", "attention_thinking_sep_needle_mass_by_layer_head.png", "Thinking sep_token_k needle attention mass"),
    ]:
        out = run_dir / "attention" / filename
        if think.empty or metric not in think:
            _placeholder(out, title)
            continue
        pivot = think.groupby(["layer", "head"], as_index=False)[metric].mean().pivot(index="layer", columns="head", values=metric)
        fig, ax = plt.subplots(figsize=(7, 4))
        _heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title(title)
        _save(fig, out)
    non = df[df["model_type"] == "non_thinking"]
    for metric, filename, title in [
        ("ans_to_all_needles_mass", "attention_nonthinking_ans_needle_mass_by_layer_head.png", "Non-thinking <Ans> attention mass to needles"),
        ("top_n_retrieval_recall", "attention_nonthinking_topn_recall_by_layer_head.png", "Non-thinking <Ans> top-n needle recall"),
    ]:
        out = run_dir / "attention" / filename
        if non.empty or metric not in non:
            _placeholder(out, title)
            continue
        pivot = non.groupby(["layer", "head"], as_index=False)[metric].mean().pivot(index="layer", columns="head", values=metric)
        fig, ax = plt.subplots(figsize=(7, 4))
        _heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title(title)
        _save(fig, out)
    if not think.empty:
        diag = df[(df["model_type"] == "thinking_sep_trace") & (df["query_anchor"] == "sep_token_k")]
        diag = diag.groupby(["subset"], as_index=False)["correct_top1_rate"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        _barplot(data=diag, x="subset", y="correct_top1_rate", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("sep_token_k retrieval: all vs unique vs repeated marker examples")
        ax.tick_params(axis="x", rotation=20)
        _save(fig, run_dir / "attention" / "attention_unique_vs_repeated_marker_diagnostic.png")
        best = think.groupby(["layer", "head"], as_index=False)["correct_top1_rate"].mean().sort_values("correct_top1_rate", ascending=False)
        matrix_path = run_dir / "attention" / "attention_trace_matrices_long.csv"
        if not best.empty and matrix_path.exists():
            best_layer = int(best.iloc[0]["layer"])
            best_head = int(best.iloc[0]["head"])
            matrices = pd.read_csv(matrix_path)
            matrices = matrices[
                (matrices["query_anchor"] == "sep_token_k")
                & (matrices["layer"] == best_layer)
                & (matrices["head"] == best_head)
            ]
            needle_cols = [col for col in matrices.columns if col.startswith("needle_j_")]
            for count_bin, filename in [
                ("low", "attention_matrix_thinking_sep_best_head_low.png"),
                ("mid", "attention_matrix_thinking_sep_best_head_mid.png"),
                ("high", "attention_matrix_thinking_sep_best_head_high.png"),
            ]:
                sub = matrices[matrices["count_bin"] == count_bin]
                out = run_dir / "attention" / filename
                if sub.empty:
                    _placeholder(out, f"Best sep-token attention matrix: {count_bin}")
                    continue
                long = sub.melt(
                    id_vars=["trace_item_k"],
                    value_vars=needle_cols,
                    var_name="needle_j",
                    value_name="attention_mass",
                )
                long["needle_j"] = long["needle_j"].str.replace("needle_j_", "", regex=False).astype(int)
                pivot = long.groupby(["trace_item_k", "needle_j"], as_index=False)["attention_mass"].mean().pivot(
                    index="trace_item_k", columns="needle_j", values="attention_mass"
                )
                fig, ax = plt.subplots(figsize=(6, 5))
                _heatmap(pivot, cmap="viridis", ax=ax)
                ax.set_title(f"Best sep_token_k head L{best_layer}H{best_head}: {count_bin}")
                ax.set_xlabel("prompt needle index j")
                ax.set_ylabel("trace item index k")
                _save(fig, out)


def make_all_plots(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    plot_train_loss(run_dir)
    plot_eval_final_loss(run_dir)
    plot_eval_accuracy_by_bin(run_dir)
    plot_final_accuracy_by_count(run_dir)
    plot_accuracy_heatmaps(run_dir)
    plot_trace_quality(run_dir)
    plot_probe_metrics(run_dir)
    plot_attention_metrics(run_dir)
