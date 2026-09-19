from __future__ import annotations

from pathlib import Path

import pandas as pd


def _mean_metric(df: pd.DataFrame, mode: str, metric: str) -> float:
    if df.empty or metric not in df or mode not in set(df["mode"]):
        return float("nan")
    return float(df[df["mode"].eq(mode)][metric].mean())


def classify_outcome(eval_df: pd.DataFrame, mode_switch: pd.DataFrame) -> str:
    think_acc = _mean_metric(eval_df, "thinking", "final_accuracy")
    non_acc = _mean_metric(eval_df, "nonthinking", "final_accuracy")
    trace_recall = _mean_metric(eval_df, "thinking", "trace_marker_recall")
    final_switch = mode_switch[mode_switch["step"].eq(mode_switch["step"].max())] if not mode_switch.empty else mode_switch
    switch_acc = float(final_switch["argmax_is_desired"].mean()) if not final_switch.empty else float("nan")
    if switch_acc < 0.5:
        return "D. switch failure"
    if think_acc > 0.8 and non_acc > 0.8 and trace_recall > 0.8:
        return "A. successful toggle"
    if max(think_acc, non_acc) > 0.5:
        return "B. partial toggle"
    return "C. mode collapse"


def make_report(run_dir: Path) -> tuple[Path, Path]:
    tables = run_dir / "tables"
    figures = run_dir / "figures"
    eval_df = pd.read_csv(tables / "eval_by_step.csv") if (tables / "eval_by_step.csv").exists() else pd.DataFrame()
    mode_switch = pd.read_csv(tables / "mode_switch.csv") if (tables / "mode_switch.csv").exists() else pd.DataFrame()
    final = eval_df[eval_df["step"].eq(eval_df["step"].max())] if not eval_df.empty else eval_df
    outcome = classify_outcome(final, mode_switch)
    think_acc = _mean_metric(final, "thinking", "final_accuracy")
    non_acc = _mean_metric(final, "nonthinking", "final_accuracy")
    trace_recall = _mean_metric(final, "thinking", "trace_marker_recall")
    final_switch = mode_switch[mode_switch["step"].eq(mode_switch["step"].max())] if not mode_switch.empty else mode_switch
    switch_acc = float(final_switch["argmax_is_desired"].mean()) if not final_switch.empty else float("nan")
    md = f"""# Synthetic NIAH Counting v5 Report

Conclusion: **{outcome}**

## Toggle Questions

1. Can one transformer learn both formats? Final debug accuracy is thinking={think_acc:.3f}, non-thinking={non_acc:.3f}.
2. Does thinking-on generation produce marker traces before `</Think>`? Mean trace recall is {trace_recall:.3f}.
3. Does `<THINK_OFF>` make the model generate `</Think>` before the count? Mean mode-conditioned next-token accuracy is {switch_acc:.3f}.
4. Does `<THINK_ON>` start a trace while `<THINK_OFF>` closes the block? See `mode_switch.csv`.
5. Do the modes show different retrieval patterns? See `attention_metrics.csv` and `mode_hidden_similarity.csv`.

## Key Figures

- `figures/train_loss_by_step_and_mode.png`
- `figures/final_accuracy_by_step_mode.png`
- `figures/final_accuracy_by_count_mode.png`
- `figures/trace_metrics_by_count.png`
- `figures/mode_switch_accuracy_by_step.png`
- `figures/attention_trace_to_prompt_best_head.png`
"""
    md_path = run_dir / "report.md"
    html_path = run_dir / "report.html"
    md_path.write_text(md, encoding="utf-8")
    html_figs = "\n".join(
        f'<h3>{path.name}</h3><img src="figures/{path.name}" style="max-width: 900px; width: 100%;">'
        for path in sorted(figures.glob("*.png"))
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Synthetic NIAH v5 Report</title></head>
<body>
<pre>{md}</pre>
{html_figs}
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")
    return md_path, html_path
