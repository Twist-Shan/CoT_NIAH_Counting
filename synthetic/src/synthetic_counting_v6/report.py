from __future__ import annotations

from pathlib import Path

import pandas as pd


def _img(rel_path: str, caption: str) -> str:
    return f'<figure><img src="{rel_path}" style="max-width: 100%; border: 1px solid #ddd;"><figcaption>{caption}</figcaption></figure>'


def _metric(df: pd.DataFrame, model_type: str, eval_mode: str, metric: str = "accuracy") -> float | None:
    sub = df[(df["model_type"] == model_type) & (df["eval_mode"] == eval_mode)]
    if sub.empty or metric not in sub:
        return None
    step = sub["step"].max()
    return float(sub[sub["step"] == step][metric].mean())


def _fmt(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.3f}"


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    by_bin_path = run_dir / "metrics" / "metrics_eval_by_bin.csv"
    by_count_path = run_dir / "metrics" / "metrics_eval_by_count.csv"
    by_bin = pd.read_csv(by_bin_path) if by_bin_path.exists() else pd.DataFrame()
    by_count = pd.read_csv(by_count_path) if by_count_path.exists() else pd.DataFrame()
    non_acc = _metric(by_bin, "non_thinking", "direct")
    think_gen_acc = _metric(by_bin, "thinking_sep_trace", "generated_trace")
    think_oracle_acc = _metric(by_bin, "thinking_sep_trace", "oracle_trace_final_readout")
    trace_exact = _metric(by_count, "thinking_sep_trace", "generated_trace", "trace_exact_match_rate")
    sep_acc = _metric(by_count, "thinking_sep_trace", "generated_trace", "trace_delimiter_count_accuracy")
    interpretation = "D. Probe/attention-only evidence or incomplete run."
    if think_gen_acc is not None and trace_exact is not None:
        if think_gen_acc >= 0.99 and trace_exact >= 0.99:
            interpretation = "A. Separator trace matches indexed trace behaviorally: numeric indices were not necessary for final accuracy or trace generation."
        elif think_gen_acc >= 0.99:
            interpretation = "B. Separator trace solves final count but trace/retrieval quality is weaker than a fully exact indexed trace."
        else:
            interpretation = "C. Separator trace did not fully learn reliable generated traces; v2 numeric indices may have acted as useful scaffold."

    figures = [
        ("plots/train_loss_vs_step.png", "Training masked completion loss. Raw completion loss is not directly comparable because thinking has more supervised tokens."),
        ("plots/eval_final_answer_loss_vs_step.png", "Comparable final-answer cross-entropy: non-thinking direct vs separator-thinking with gold oracle trace."),
        ("plots/eval_accuracy_by_bin_vs_step.png", "Exact final-count accuracy over training, split by low/mid/high count bins."),
        ("plots/final_accuracy_by_count.png", "Final checkpoint accuracy by exact gold count."),
        ("plots/accuracy_heatmap_by_count_and_step_non_thinking.png", "Non-thinking accuracy heatmap across count and training step."),
        ("plots/accuracy_heatmap_by_count_and_step_thinking_generated_trace.png", "Separator-thinking free-generation accuracy heatmap."),
        ("plots/accuracy_heatmap_by_count_and_step_thinking_oracle_trace.png", "Separator-thinking oracle-trace final-readout accuracy heatmap."),
        ("plots/trace_exact_by_count.png", "Whether the generated separator-marker trace exactly matches the gold trace."),
        ("plots/trace_delimiter_count_accuracy_by_count.png", "Whether the generated number of <Sep> delimiters equals the gold count."),
        ("attention/attention_thinking_sep_correct_top1_by_layer_head.png", "For sep_token_k queries, whether the top-attended prompt needle is the kth needle."),
        ("attention/attention_matrix_thinking_sep_best_head_mid.png", "Average k-to-j attention matrix for the best sep_token_k retrieval head."),
        ("probes/probe_prefix_count_accuracy_heatmap_thinking_sep_trace.png", "Linear prefix-count decodability from separator-trace hidden states."),
        ("probes/probe_sep_token_prefix_probe_minus_position_baseline.png", "sep_token_k probe advantage after subtracting the position-only baseline."),
    ]
    existing_figures = [_img(path, caption) for path, caption in figures if (run_dir / path).exists()]
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Trace Count v6 Separator-Trace Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 32px auto; line-height: 1.5; color: #222; }}
h1, h2 {{ color: #111; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
td, th {{ border: 1px solid #ddd; padding: 6px 10px; }}
figcaption {{ color: #444; font-size: 0.95em; margin-top: 6px; }}
figure {{ margin: 24px 0; }}
code {{ background: #f3f3f3; padding: 2px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Trace Count v6: Separator Trace, No Numeric Indices</h1>
<p>v6 copies the v2 synthetic NIAH-style counting setup, with one intentional change: the thinking trace uses repeated <code>&lt;Sep&gt;</code> delimiters instead of numeric trace indices.</p>
<pre>v2: &lt;Think/&gt; &lt;1&gt; &lt;A&gt; &lt;2&gt; &lt;B&gt; &lt;3&gt; &lt;C&gt; &lt;/Think&gt; &lt;Ans&gt; &lt;3&gt;
v6: &lt;Think/&gt; &lt;Sep&gt; &lt;A&gt; &lt;Sep&gt; &lt;B&gt; &lt;Sep&gt; &lt;C&gt; &lt;/Think&gt; &lt;Ans&gt; &lt;3&gt;</pre>
<h2>Headline Metrics</h2>
<table>
<tr><th>Question</th><th>Metric</th><th>Value</th></tr>
<tr><td>Did non-thinking solve final count?</td><td>direct final accuracy</td><td>{_fmt(non_acc)}</td></tr>
<tr><td>Did separator-thinking solve generated trace + answer?</td><td>generated-trace final accuracy</td><td>{_fmt(think_gen_acc)}</td></tr>
<tr><td>Can separator-thinking read count from a gold trace?</td><td>oracle-trace final accuracy</td><td>{_fmt(think_oracle_acc)}</td></tr>
<tr><td>Is generated trace exact?</td><td>trace exact match</td><td>{_fmt(trace_exact)}</td></tr>
<tr><td>Does delimiter count match gold count?</td><td>&lt;Sep&gt; count accuracy</td><td>{_fmt(sep_acc)}</td></tr>
</table>
<h2>Interpretation</h2>
<p>{interpretation}</p>
<p>Probe and attention evidence should be read diagnostically, not causally. In v6, numeric trace-token identity leakage is removed, but absolute position and trace length can still carry count information, so the position and trace-length baselines matter.</p>
<h2>Figures</h2>
{''.join(existing_figures)}
</body>
</html>"""
    out = run_dir / "report" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    md = run_dir / "report" / "report.md"
    md.write_text(
        f"# Trace Count v6 Report\n\nInterpretation: {interpretation}\n\n"
        f"- non-thinking direct accuracy: {_fmt(non_acc)}\n"
        f"- thinking generated-trace accuracy: {_fmt(think_gen_acc)}\n"
        f"- thinking oracle-trace final accuracy: {_fmt(think_oracle_acc)}\n",
        encoding="utf-8",
    )
    return out

