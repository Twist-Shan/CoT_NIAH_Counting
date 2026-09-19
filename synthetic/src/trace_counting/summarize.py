from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, load_json, save_json

WEIGHTED_MASKS = {"full_sequence_final_weighted", "completion_final_weighted"}


def summarize_runs(args: argparse.Namespace) -> pd.DataFrame:
    runs_dir = Path(args.runs_dir)
    rows = []
    for metrics_path in sorted(runs_dir.rglob("eval/*_metrics.json")):
        if metrics_path.name == "summary_metrics.json":
            continue
        run_dir = metrics_path.parent.parent
        config_path = run_dir / "config.json"
        config = load_json(config_path) if config_path.exists() else {}
        loss_mask = config.get("loss_mask")
        final_weight = config.get("final_weight") if loss_mask in WEIGHTED_MASKS else None
        metrics = load_json(metrics_path)
        split = metrics.get("split", metrics_path.name.removesuffix("_metrics.json"))
        tf = metrics.get("teacher_forced", {})
        ar = metrics.get("autoregressive", {})
        rows.append(
            {
                "run_dir": str(run_dir),
                "split": split,
                "model_name": config.get("model", {}).get("model_name"),
                "seed": config.get("seed"),
                "loss_mask": loss_mask,
                "final_weight": final_weight,
                "tf_count_acc": tf.get("count_accuracy"),
                "tf_mae": tf.get("mean_absolute_error"),
                "tf_under": tf.get("undercount_rate"),
                "tf_over": tf.get("overcount_rate"),
                "ar_count_acc": ar.get("count_accuracy"),
                "ar_mae": ar.get("mean_absolute_error"),
                "trace_exact": ar.get("trace_exact_match"),
                "format_valid": ar.get("format_validity"),
            }
        )
    df = pd.DataFrame(rows)
    if args.out_csv:
        out_csv = Path(args.out_csv)
        ensure_dir(out_csv.parent)
        df.to_csv(out_csv, index=False)
    if args.out_json:
        save_json(rows, args.out_json)
    if args.print_markdown and not df.empty:
        print(df.to_markdown(index=False))
    elif df.empty:
        print(f"No eval metrics found under {runs_dir}")
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate eval metrics across run directories.")
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--print_markdown", action="store_true")
    return parser


def main() -> None:
    summarize_runs(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
