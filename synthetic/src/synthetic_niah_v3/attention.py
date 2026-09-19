from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import BaseExample, count_bin
from .render import render_for_model
from .vocab import Vocab


def _entropy(probs: np.ndarray) -> float:
    probs = probs.astype(float)
    probs = probs / max(probs.sum(), 1e-12)
    return float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())


@torch.no_grad()
def run_attention_analysis(
    model_by_type: dict[str, Any],
    examples_by_len: dict[int, list[BaseExample]],
    vocab: Vocab,
    cfg: dict[str, Any],
    seed: int,
    checkpoint_step: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    device = cfg["device"]
    examples: list[BaseExample] = []
    for seq_examples in examples_by_len.values():
        examples.extend(seq_examples[: int(cfg["attention_examples_per_count"])])
    for model_type, model in model_by_type.items():
        model.eval()
        for example in examples:
            rendered = render_for_model(example, vocab, model_type)
            input_ids = torch.tensor([rendered.tokens], dtype=torch.long, device=device)
            out = model(input_ids, output_attentions=True)
            if out.attentions is None:
                continue
            needle_positions = rendered.prompt_needle_token_positions
            all_prompt_positions = list(range(rendered.spans.seq_start, rendered.spans.seq_end_exclusive))
            noise_positions = [pos for pos in all_prompt_positions if pos not in set(needle_positions)]
            for layer_idx, attn in enumerate(out.attentions):
                attn_np = attn[0].detach().cpu().numpy()
                for head_idx in range(attn_np.shape[0]):
                    probs = attn_np[head_idx]
                    if model_type == "thinking":
                        query_specs = []
                        for k, pos in enumerate(rendered.spans.trace_index_positions):
                            query_specs.append(("index_k_pos", k, pos))
                        for k, pos in enumerate(rendered.spans.trace_marker_positions):
                            query_specs.append(("marker_k_pos", k, pos))
                        for k, pos in enumerate(rendered.spans.trace_marker_positions):
                            if pos + 1 < probs.shape[0]:
                                query_specs.append(("post_marker_k", k, pos + 1))
                        by_anchor: dict[str, list[dict[str, float]]] = {}
                        for anchor, k, query_pos in query_specs:
                            row = probs[query_pos]
                            needle_attention = row[needle_positions] if needle_positions else np.array([])
                            correct_top1 = (
                                int(np.argmax(needle_attention) == k) if len(needle_attention) and k < len(needle_positions) else 0
                            )
                            needle_mass = float(needle_attention.sum()) if len(needle_attention) else 0.0
                            noise_mass = float(row[noise_positions].sum()) if noise_positions else 0.0
                            by_anchor.setdefault(anchor, []).append(
                                {
                                    "correct_top1_rate": correct_top1,
                                    "diagonal_dominance": float(
                                        row[needle_positions[k]] / max(needle_mass, 1e-12)
                                    )
                                    if needle_positions and k < len(needle_positions)
                                    else 0.0,
                                    "needle_mass": needle_mass,
                                    "noise_mass": noise_mass,
                                    "needle_to_noise_ratio": needle_mass / max(noise_mass, 1e-12),
                                    "entropy": _entropy(row[all_prompt_positions]),
                                    "off_diagonal_mass": float(needle_mass - (row[needle_positions[k]] if k < len(needle_positions) else 0.0)),
                                }
                            )
                        for anchor, values in by_anchor.items():
                            rows.append(
                                {
                                    "model_type": model_type,
                                    "seed": seed,
                                    "checkpoint_step": checkpoint_step,
                                    "seq_len_eval": example.seq_len,
                                    "layer": layer_idx,
                                    "head": head_idx,
                                    "query_anchor": anchor,
                                    "count_bin": count_bin(example.count),
                                    "top_n_recall": math.nan,
                                    **{key: float(np.mean([v[key] for v in values])) for key in values[0]},
                                }
                            )
                    else:
                        query_pos = rendered.spans.ans_pos
                        row = probs[query_pos]
                        prompt_attention = row[all_prompt_positions]
                        top_prompt_positions = np.array(all_prompt_positions)[np.argsort(prompt_attention)[-example.count :]]
                        top_n_recall = float(set(needle_positions).issubset(set(top_prompt_positions.tolist())))
                        needle_mass = float(row[needle_positions].sum())
                        noise_mass = float(row[noise_positions].sum()) if noise_positions else 0.0
                        rows.append(
                            {
                                "model_type": model_type,
                                "seed": seed,
                                "checkpoint_step": checkpoint_step,
                                "seq_len_eval": example.seq_len,
                                "layer": layer_idx,
                                "head": head_idx,
                                "query_anchor": "ans_pos",
                                "count_bin": count_bin(example.count),
                                "correct_top1_rate": math.nan,
                                "diagonal_dominance": math.nan,
                                "needle_mass": needle_mass,
                                "noise_mass": noise_mass,
                                "needle_to_noise_ratio": needle_mass / max(noise_mass, 1e-12),
                                "entropy": _entropy(prompt_attention),
                                "off_diagonal_mass": math.nan,
                                "top_n_recall": top_n_recall,
                            }
                        )
    return pd.DataFrame(rows)


def attention_leaderboard(attn_df: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
    if attn_df.empty:
        return pd.DataFrame()
    thinking = attn_df[attn_df["model_type"].eq("thinking")]
    if thinking.empty:
        return pd.DataFrame()
    score_cols = ["correct_top1_rate", "diagonal_dominance", "needle_mass"]
    board = thinking.groupby(["layer", "head", "query_anchor"], as_index=False)[score_cols].mean()
    board["score"] = board[score_cols].mean(axis=1)
    return board.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
