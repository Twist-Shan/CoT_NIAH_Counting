from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .io_utils import ensure_dir, load_json, read_jsonl
from .loss_masks import SEGMENT_NAMES


def _save_lineplot(df: pd.DataFrame, x: str, y: str, path: Path, *, hue: str | None = None, title: str | None = None) -> None:
    if df.empty or y not in df.columns:
        return
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_training(run_dir: Path, plots_dir: Path) -> None:
    log_path = run_dir / "train_log.jsonl"
    if not log_path.exists():
        return
    rows = read_jsonl(log_path)
    if not rows:
        return
    df = pd.DataFrame(rows)
    _save_lineplot(df, "step", "total_weighted_loss", plots_dir / "training_loss.png", title="Training weighted loss")
    _save_lineplot(df, "step", "val_total_weighted_loss", plots_dir / "validation_loss.png", title="Validation weighted loss")
    _save_lineplot(df, "step", "val_tf_count_acc", plots_dir / "tf_accuracy_by_step.png", title="Teacher-forced count accuracy")

    segment_cols = [name for name in SEGMENT_NAMES if name in df.columns]
    if segment_cols:
        long_df = df[["step", *segment_cols]].melt(id_vars="step", var_name="segment", value_name="loss").dropna()
        _save_lineplot(long_df, "step", "loss", plots_dir / "loss_breakdown_by_segment.png", hue="segment")


def _group_metric_rows(metrics: dict, mode: str, group_key: str) -> pd.DataFrame:
    section = metrics.get(mode, {})
    groups = section.get(group_key, {})
    rows = []
    for key, values in groups.items():
        row = {"group": key, **values}
        try:
            row["group_num"] = int(key)
        except ValueError:
            row["group_num"] = key
        rows.append(row)
    return pd.DataFrame(rows)


def plot_eval(run_dir: Path, plots_dir: Path) -> None:
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        return
    tf_rows = []
    ar_rows = []
    trace_rows = []
    for metric_path in sorted(eval_dir.glob("*_metrics.json")):
        if metric_path.name == "summary_metrics.json":
            continue
        split = metric_path.name.removesuffix("_metrics.json")
        metrics = load_json(metric_path)
        tf_df = _group_metric_rows(metrics, "teacher_forced", "accuracy_by_count")
        if not tf_df.empty:
            tf_df["split"] = split
            tf_rows.append(tf_df)
        ar_df = _group_metric_rows(metrics, "autoregressive", "accuracy_by_count")
        if not ar_df.empty:
            ar_df["split"] = split
            ar_rows.append(ar_df)
        trace_groups = metrics.get("autoregressive", {}).get("trace_by_count", {})
        if trace_groups:
            trace_df = pd.DataFrame(
                [
                    {
                        "group": key,
                        "group_num": int(key),
                        "split": split,
                        **values,
                    }
                    for key, values in trace_groups.items()
                ]
            )
            trace_rows.append(trace_df)

    if tf_rows:
        df = pd.concat(tf_rows, ignore_index=True).sort_values(["split", "group_num"])
        _save_lineplot(df, "group_num", "count_accuracy", plots_dir / "tf_accuracy_by_count.png", hue="split")
    if ar_rows:
        df = pd.concat(ar_rows, ignore_index=True).sort_values(["split", "group_num"])
        _save_lineplot(df, "group_num", "count_accuracy", plots_dir / "ar_accuracy_by_count.png", hue="split")
    if trace_rows:
        df = pd.concat(trace_rows, ignore_index=True).sort_values(["split", "group_num"])
        _save_lineplot(df, "group_num", "trace_exact_match", plots_dir / "trace_exact_by_count.png", hue="split")


def make_plots(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    plots_dir = ensure_dir(args.out_dir or run_dir / "plots")
    sns.set_theme(style="whitegrid")
    plot_training(run_dir, plots_dir)
    plot_eval(run_dir, plots_dir)
    print(f"saved plots to {plots_dir}")
    return plots_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create standard plots from a trace-counting run directory.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    return parser


def main() -> None:
    make_plots(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
