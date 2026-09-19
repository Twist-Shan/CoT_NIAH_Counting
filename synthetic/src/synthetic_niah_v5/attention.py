from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .cache import final_checkpoint_path
from .data import balanced_examples, render_nonthinking, render_thinking, trace_prediction_queries
from .model import make_model
from .train import load_checkpoint
from .vocab import Vocab


def _entropy(probs: np.ndarray) -> float:
    probs = probs.astype(float)
    probs = probs / max(float(probs.sum()), 1e-12)
    return float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())


def _row_metrics(row: np.ndarray, needle_positions: list[int], noise_positions: list[int], correct_idx: int | None) -> dict[str, float]:
    needle = row[needle_positions] if needle_positions else np.array([], dtype=float)
    needle_mass = float(needle.sum()) if len(needle) else 0.0
    noise_mass = float(row[noise_positions].sum()) if noise_positions else 0.0
    if correct_idx is not None and len(needle) and correct_idx < len(needle):
        correct_top1 = float(int(np.argmax(needle) == correct_idx))
        diagonal_dominance = float(needle[correct_idx] / max(needle_mass, 1e-12))
    else:
        correct_top1 = math.nan
        diagonal_dominance = math.nan
    return {
        "correct_top1": correct_top1,
        "diagonal_dominance": diagonal_dominance,
        "needle_mass": needle_mass,
        "needle_vs_noise_ratio": needle_mass / max(noise_mass, 1e-12),
        "entropy": _entropy(row[needle_positions + noise_positions]) if needle_positions or noise_positions else math.nan,
    }


@torch.no_grad()
def run_attention(cfg: dict[str, Any], vocab: Vocab, run_dir: Path) -> pd.DataFrame:
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    examples = balanced_examples(
        int(cfg["train"]["seq_len"]),
        int(cfg["train"]["attention_examples_per_count"]),
        int(cfg["train"]["seed"]) + 9000,
        int(cfg["train"]["count_min"]),
        int(cfg["train"]["count_max"]),
    )
    model = make_model(cfg["model"], cfg["device"])
    load_checkpoint(model, final_checkpoint_path(run_dir), cfg["device"])
    model.eval()
    rows: list[dict[str, Any]] = []
    matrices: dict[tuple[int, int], list[np.ndarray]] = {}
    for ex in examples:
        for rendered in [
            render_thinking(ex, vocab, trace_indices=bool(cfg["trace_indices"])),
            render_nonthinking(ex, vocab),
        ]:
            input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=cfg["device"])
            out = model(input_ids=input_ids, output_attentions=True)
            attentions = list(out.attentions or [])
            needle_positions = rendered.prompt_needle_token_positions
            prompt_positions = list(range(rendered.spans.seq_start, rendered.spans.seq_end_exclusive))
            needle_set = set(needle_positions)
            noise_positions = [pos for pos in prompt_positions if pos not in needle_set]
            for layer, attn in enumerate(attentions):
                attn_np = attn[0].detach().cpu().numpy()
                for head in range(attn_np.shape[0]):
                    probs = attn_np[head]
                    if rendered.variant == "thinking":
                        per_anchor: dict[str, list[dict[str, float]]] = {}
                        mat = np.zeros((len(rendered.spans.trace_marker_positions), len(needle_positions)), dtype=float)
                        query_rows = trace_prediction_queries(rendered)
                        for k, query in enumerate(query_rows):
                            query_pos = int(query["prediction_query_pos"])
                            row = probs[query_pos]
                            if needle_positions:
                                mat[k, : len(needle_positions)] = row[needle_positions]
                            metrics = _row_metrics(row, needle_positions, noise_positions, k)
                            per_anchor.setdefault(str(query["query_kind"]), []).append(metrics)
                        if mat.size:
                            matrices.setdefault((layer, head), []).append(mat)
                        for query_anchor, vals in per_anchor.items():
                            rows.append(
                                {
                                    "mode": rendered.variant,
                                    "layer": layer,
                                    "head": head,
                                    "query_anchor": query_anchor,
                                    **{key: float(np.nanmean([v[key] for v in vals])) for key in vals[0]},
                                }
                            )
                    else:
                        for query_anchor, query_pos in [("think_close_pos", rendered.spans.think_close_pos), ("pre_count_pos", rendered.spans.pre_count_pos)]:
                            rows.append(
                                {
                                    "mode": rendered.variant,
                                    "layer": layer,
                                    "head": head,
                                    "query_anchor": query_anchor,
                                    **_row_metrics(probs[query_pos], needle_positions, noise_positions, None),
                                }
                            )
    attn_df = pd.DataFrame(rows)
    if not attn_df.empty:
        attn_df = attn_df.groupby(["mode", "layer", "head", "query_anchor"], as_index=False).mean(numeric_only=True)
    attn_df.to_csv(tables / "attention_metrics.csv", index=False)
    matrix_rows: list[dict[str, Any]] = []
    for (layer, head), mats in matrices.items():
        max_rows = max(mat.shape[0] for mat in mats)
        max_cols = max(mat.shape[1] for mat in mats)
        padded = []
        for mat in mats:
            arr = np.full((max_rows, max_cols), np.nan)
            arr[: mat.shape[0], : mat.shape[1]] = mat
            padded.append(arr)
        avg = np.nanmean(np.stack(padded), axis=0)
        for i in range(avg.shape[0]):
            for j in range(avg.shape[1]):
                matrix_rows.append({"layer": layer, "head": head, "trace_index": i + 1, "needle_index": j + 1, "attention": float(avg[i, j])})
    pd.DataFrame(matrix_rows).to_csv(tables / "attention_trace_matrix.csv", index=False)
    return attn_df
