from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from synthetic_niah_v5.data import balanced_examples, render_nonthinking, render_thinking, trace_prediction_queries
from synthetic_niah_v5.model import make_model
from synthetic_niah_v5.train import load_checkpoint
from synthetic_niah_v5.vocab import MARKER_TOKENS, Vocab, count_token, index_token


def _is_v5_run_dir(path: Path) -> bool:
    has_files = all(
        candidate.is_file()
        for candidate in (
            path / "config.json",
            path / "vocab.json",
            path / "checkpoints" / "final.pt",
        )
    )
    if not has_files:
        return False
    try:
        vocab_obj = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    tokens = set(vocab_obj.get("id_to_token", []))
    return {"<THINK_ON>", "<THINK_OFF>"}.issubset(tokens)


def resolve_v5_run_dir(path: str | Path) -> Path:
    """Resolve a v5 run from either the run itself or a containing result folder."""
    requested = Path(path).expanduser()
    direct = requested
    if requested.is_file():
        if requested.name == "config.json":
            direct = requested.parent
        elif requested.name == "final.pt" and requested.parent.name == "checkpoints":
            direct = requested.parent.parent

    if _is_v5_run_dir(direct):
        return direct.resolve()

    candidates: list[Path] = []
    if direct.is_dir():
        for config_path in direct.rglob("config.json"):
            candidate = config_path.parent
            if _is_v5_run_dir(candidate):
                candidates.append(candidate.resolve())

    unique = list(dict.fromkeys(candidates))
    if unique:
        return max(
            unique,
            key=lambda candidate: (
                (candidate / "checkpoints" / "final.pt").stat().st_mtime,
                str(candidate),
            ),
        )

    required = "config.json, vocab.json with <THINK_ON>/<THINK_OFF>, checkpoints/final.pt"
    raise FileNotFoundError(
        f"Could not resolve a completed v5 run under '{requested}'. "
        f"A valid run must contain: {required}. If this is a fresh Colab runtime, "
        "mount Google Drive or upload/copy the saved v5 result bundle first."
    )


def _softmax_np(logits: torch.Tensor) -> np.ndarray:
    return F.softmax(logits.detach().float().cpu(), dim=-1).numpy()


def _safe_entropy(x: np.ndarray) -> float:
    total = float(x.sum())
    if total <= 0:
        return math.nan
    p = x.astype(float) / total
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum() / max(math.log(len(p)), 1e-12))


def _margin(logits: torch.Tensor, target_id: int, competitor_ids: list[int]) -> float:
    values = logits.detach().float().cpu()
    target = float(values[target_id])
    competitors = [idx for idx in competitor_ids if int(idx) != int(target_id)]
    if not competitors:
        return math.nan
    return target - float(values[competitors].max())


def load_v5_state(run_dir: Path, device: str | None = None):
    run_dir = resolve_v5_run_dir(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if device is not None:
        cfg["device"] = device
    vocab = Vocab.load(run_dir / "vocab.json")
    model = make_model(cfg["model"], cfg["device"])
    load_checkpoint(model, run_dir / "checkpoints" / "final.pt", cfg["device"])
    model.eval()
    return cfg, vocab, model


def _prediction_query_positions(rendered, trace_indices: bool) -> list[dict[str, Any]]:
    rows = trace_prediction_queries(rendered)
    has_indices = bool(rendered.spans.trace_index_positions)
    if has_indices != bool(trace_indices):
        raise AssertionError("trace_indices disagrees with rendered trace spans.")
    return rows


def _attention_metrics(
    row: np.ndarray,
    prompt_needles: list[int],
    prompt_noise: list[int],
    correct_idx: int,
    bos_pos: int,
    think_open_pos: int,
    prev_trace_marker_pos: int | None,
    prev_trace_index_pos: int | None,
) -> dict[str, float]:
    needle_weights = row[prompt_needles] if prompt_needles else np.array([], dtype=float)
    correct_mass = float(row[prompt_needles[correct_idx]]) if prompt_needles and correct_idx < len(prompt_needles) else 0.0
    all_needle_mass = float(needle_weights.sum()) if len(needle_weights) else 0.0
    noise_mass = float(row[prompt_noise].sum()) if prompt_noise else 0.0
    if len(needle_weights):
        correct_top1 = float(int(np.argmax(needle_weights) == correct_idx))
        diag_share = correct_mass / max(all_needle_mass, 1e-12)
    else:
        correct_top1 = math.nan
        diag_share = math.nan
    return {
        "correct_top1": correct_top1,
        "diag_share_of_needle_mass": diag_share,
        "correct_prompt_needle_mass": correct_mass,
        "all_prompt_needles_mass": all_needle_mass,
        "prompt_noise_mass": noise_mass,
        "needle_entropy_normalized": _safe_entropy(needle_weights) if len(needle_weights) else math.nan,
        "bos_mass": float(row[bos_pos]),
        "think_open_mass": float(row[think_open_pos]),
        "previous_trace_marker_mass": float(row[prev_trace_marker_pos]) if prev_trace_marker_pos is not None else 0.0,
        "previous_trace_index_mass": float(row[prev_trace_index_pos]) if prev_trace_index_pos is not None else 0.0,
    }


@torch.no_grad()
def run_switch_and_retrieval_diagnostics(
    run_dir: str | Path,
    *,
    examples_per_count: int = 100,
    seed_offset: int = 77_000,
    device: str | None = None,
) -> dict[str, pd.DataFrame]:
    run_dir = resolve_v5_run_dir(run_dir)
    cfg, vocab, model = load_v5_state(run_dir, device=device)
    device = cfg["device"]
    train = cfg["train"]
    trace_indices = bool(cfg.get("trace_indices", False))
    examples = balanced_examples(
        int(train["seq_len"]),
        int(examples_per_count),
        int(train["seed"]) + int(seed_offset),
        int(train["count_min"]),
        int(train["count_max"]),
    )
    out_dir = run_dir / "v5_2_switch_diagnostics"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    switch_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    marker_ids = vocab.marker_ids
    count_ids = vocab.count_ids
    trace_start_ids = [vocab.token_to_id[index_token(1)]] if trace_indices else marker_ids
    for ex_idx, ex in enumerate(examples):
        rendered_think = render_thinking(ex, vocab, trace_indices=trace_indices)
        rendered_non = render_nonthinking(ex, vocab)
        for rendered in [rendered_think, rendered_non]:
            input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
            out = model(input_ids=input_ids, output_attentions=True, output_hidden_states=True)
            logits = out.logits[0]
            probs_open = _softmax_np(logits[rendered.spans.think_open_pos])
            probs_close = _softmax_np(logits[rendered.spans.think_close_pos])
            switch_rows.append(
                {
                    "example_idx": ex_idx,
                    "variant": rendered.variant,
                    "count": ex.count,
                    "p_close_after_think_open": float(probs_open[vocab.think_close_id]),
                    "p_any_marker_after_think_open": float(probs_open[marker_ids].sum()),
                    "p_any_trace_start_after_think_open": float(probs_open[trace_start_ids].sum()),
                    "p_any_count_after_think_close": float(probs_close[count_ids].sum()),
                    "p_gold_count_after_think_close": float(probs_close[vocab.count_id(ex.count)]),
                    "margin_close_vs_trace_start_after_open": _margin(logits[rendered.spans.think_open_pos], vocab.think_close_id, trace_start_ids + [vocab.think_close_id]),
                    "margin_gold_count_vs_other_counts_after_close": _margin(logits[rendered.spans.think_close_pos], vocab.count_id(ex.count), count_ids),
                }
            )
            hidden = out.hidden_states[-1][0].detach().float().cpu()
            hidden_rows.extend(
                [
                    {
                        "example_idx": ex_idx,
                        "variant": rendered.variant,
                        "anchor": "think_open",
                        "count": ex.count,
                        "hidden_norm": float(hidden[rendered.spans.think_open_pos].norm()),
                    },
                    {
                        "example_idx": ex_idx,
                        "variant": rendered.variant,
                        "anchor": "think_close",
                        "count": ex.count,
                        "hidden_norm": float(hidden[rendered.spans.think_close_pos].norm()),
                    },
                ]
            )
        input_ids = torch.tensor([rendered_think.input_ids], dtype=torch.long, device=device)
        out = model(input_ids=input_ids, output_attentions=True)
        attentions = [a[0].detach().cpu().numpy() for a in out.attentions or []]
        logits = out.logits[0]
        prompt_start = rendered_think.spans.seq_start
        prompt_end = rendered_think.spans.seq_end_exclusive
        prompt_needles = rendered_think.prompt_needle_token_positions
        prompt_needles_set = set(prompt_needles)
        prompt_noise = [pos for pos in range(prompt_start, prompt_end) if pos not in prompt_needles_set]
        query_rows = _prediction_query_positions(rendered_think, trace_indices)
        for q in query_rows:
            k = int(q["k"])
            correct_idx = k - 1
            target_marker_id = vocab.token_to_id[ex.needle_markers[correct_idx]]
            competitor_marker_ids = marker_ids
            prev_marker = None if k == 1 else query_rows[k - 2]["target_marker_pos"]
            prev_index = None
            if trace_indices and k > 1:
                prev_index = rendered_think.spans.trace_index_positions[k - 2]
            for layer, probs in enumerate(attentions, start=1):
                for head in range(probs.shape[0]):
                    pred_row = probs[head, int(q["prediction_query_pos"])]
                    post_row = probs[head, int(q["post_marker_query_pos"])]
                    base = {
                        "example_idx": ex_idx,
                        "count": ex.count,
                        "k": k,
                        "layer": layer,
                        "head": head,
                        "query_kind": q["query_kind"],
                    }
                    retrieval_rows.append(
                        {
                            **base,
                            "query_anchor": "prediction_query",
                            "target_marker_logit_margin_vs_markers": _margin(logits[int(q["prediction_query_pos"])], target_marker_id, competitor_marker_ids),
                            **_attention_metrics(
                                pred_row,
                                prompt_needles,
                                prompt_noise,
                                correct_idx,
                                rendered_think.spans.bos_pos,
                                rendered_think.spans.think_open_pos,
                                prev_marker,
                                prev_index,
                            ),
                        }
                    )
                    post_rows.append(
                        {
                            **base,
                            "query_anchor": "post_marker_token",
                            **_attention_metrics(
                                post_row,
                                prompt_needles,
                                prompt_noise,
                                correct_idx,
                                rendered_think.spans.bos_pos,
                                rendered_think.spans.think_open_pos,
                                prev_marker,
                                prev_index,
                            ),
                        }
                    )

    switch_df = pd.DataFrame(switch_rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    post_df = pd.DataFrame(post_rows)
    hidden_df = pd.DataFrame(hidden_rows)
    retrieval_head = retrieval_df.groupby(["query_anchor", "layer", "head", "query_kind"], as_index=False).mean(numeric_only=True)
    post_head = post_df.groupby(["query_anchor", "layer", "head", "query_kind"], as_index=False).mean(numeric_only=True)
    switch_summary = switch_df.groupby(["variant"], as_index=False).mean(numeric_only=True)
    hidden_summary = hidden_df.groupby(["variant", "anchor"], as_index=False).mean(numeric_only=True)

    outputs = {
        "switch_examples": switch_df,
        "switch_summary": switch_summary,
        "prediction_query_rows": retrieval_df,
        "prediction_query_head_summary": retrieval_head,
        "post_marker_rows": post_df,
        "post_marker_head_summary": post_head,
        "hidden_norm_summary": hidden_summary,
    }
    for name, df in outputs.items():
        df.to_csv(table_dir / f"{name}.csv", index=False)
    make_plots(outputs, out_dir)
    make_report(outputs, out_dir, cfg)
    return outputs


def _heatmap(df: pd.DataFrame, metric: str, title: str, path: Path, vmin: float | None = 0, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt

    mat = df.pivot(index="layer", columns="head", values=metric)
    values = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)), labels=[str(c) for c in mat.columns])
    ax.set_yticks(range(len(mat.index)), labels=[str(i) for i in mat.index])
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    finite = values[np.isfinite(values)]
    midpoint = float(np.nanmean(finite)) if finite.size else 0.5
    for y, row in enumerate(values):
        for x, val in enumerate(row):
            if np.isfinite(val):
                ax.text(x, y, f"{val:.2f}", ha="center", va="center", color="white" if val < midpoint else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_plots(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred = outputs["prediction_query_head_summary"]
    post = outputs["post_marker_head_summary"]
    _heatmap(pred, "correct_top1", "v5.2 prediction-query correct top-1 retrieval", fig_dir / "prediction_query_correct_top1.png", vmax=1)
    _heatmap(pred, "correct_prompt_needle_mass", "v5.2 prediction-query correct needle mass", fig_dir / "prediction_query_correct_mass.png")
    _heatmap(pred, "target_marker_logit_margin_vs_markers", "v5.2 marker logit margin at prediction query", fig_dir / "prediction_query_marker_margin.png", vmin=None)
    _heatmap(post, "correct_top1", "v5.2 post-marker-token correct top-1 retrieval", fig_dir / "post_marker_correct_top1.png", vmax=1)

    switch = outputs["switch_summary"].copy()
    cols = [
        "p_close_after_think_open",
        "p_any_trace_start_after_think_open",
        "p_any_count_after_think_close",
        "p_gold_count_after_think_close",
    ]
    variants = list(switch["variant"])
    x = np.arange(len(cols))
    width = 0.8 / max(1, len(variants))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, variant in enumerate(variants):
        vals = [float(switch.loc[switch["variant"] == variant, col].iloc[0]) for col in cols]
        xpos = x + (i - (len(variants) - 1) / 2) * width
        ax.bar(xpos, vals, width=width, label=variant)
    ax.set_xticks(x, labels=cols, rotation=25, ha="right")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("v5.2 switch-token probability diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "switch_probability_summary.png", dpi=180)
    plt.close(fig)


def _img(name: str) -> str:
    return f"<img src='../figures/{name}' style='max-width:100%;border:1px solid #ddd;border-radius:8px'>"


def make_report(outputs: dict[str, pd.DataFrame], out_dir: Path, cfg: dict[str, Any]) -> None:
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    pred = outputs["prediction_query_head_summary"].sort_values(["correct_top1", "correct_prompt_needle_mass"], ascending=False)
    post = outputs["post_marker_head_summary"].sort_values(["correct_top1", "correct_prompt_needle_mass"], ascending=False)
    best_pred = pred.iloc[0].to_dict() if not pred.empty else {}
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v5.2 switch diagnostics</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;line-height:1.55;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f4f7fb}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}code{{background:#eef2f7;padding:2px 5px;border-radius:5px}}</style>
</head><body>
<h1>v5.2 switch / retrieval diagnostics</h1>
<p>问题：同一个 mixed-transformer 中，开关 token 学好了是否应该自动产生显著 retrieval？本诊断区分两件事：switch logits 是否分离模式，以及 retrieval metric 是否测在正确的 prediction query 上。</p>
<h2>Setting</h2>
<table><tr><th>trace_indices</th><td>{cfg.get('trace_indices')}</td></tr><tr><th>thinking_fraction</th><td>{cfg['train'].get('thinking_fraction')}</td></tr><tr><th>seq_len</th><td>{cfg['train'].get('seq_len')}</td></tr></table>
<h2>Main interpretation</h2>
<p>如果 switch 概率已经分离，但 <code>prediction_query_correct_top1</code> 仍低，说明问题不是“不会开关”，而是 marker-only trace 缺少显式 <code>index_token_k</code> query，模型未必需要形成 v2 那种 k-to-k targeted retrieval head。</p>
<p>Best prediction-query retrieval head: layer={best_pred.get('layer','NA')}, head={best_pred.get('head','NA')}, correct_top1={best_pred.get('correct_top1','NA')}, correct_mass={best_pred.get('correct_prompt_needle_mass','NA')}.</p>
<div class="grid"><div>{_img('switch_probability_summary.png')}</div><div>{_img('prediction_query_correct_top1.png')}</div><div>{_img('prediction_query_correct_mass.png')}</div><div>{_img('post_marker_correct_top1.png')}</div></div>
<h2>Switch summary</h2>
{outputs['switch_summary'].to_html(index=False)}
<h2>Prediction-query head summary, top 12</h2>
{pred.head(12).to_html(index=False)}
<h2>Post-marker head summary, top 12</h2>
{post.head(12).to_html(index=False)}
</body></html>"""
    (report_dir / "report.html").write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v5.2 switch/retrieval diagnostics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--examples-per-count", type=int, default=100)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_switch_and_retrieval_diagnostics(args.run_dir, examples_per_count=args.examples_per_count, device=args.device)


if __name__ == "__main__":
    main()
