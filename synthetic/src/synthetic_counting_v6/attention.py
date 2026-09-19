from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import count_bin
from .data import BaseExample, balanced_examples, has_repeated_markers
from .generation import pad_sequences
from .render import non_thinking_eval_prefix, render_thinking_sep_trace
from .vocab import Vocab


THINKING_QUERY_ANCHORS = ("sep_token_k", "marker_token_k", "pre_sep_k", "post_marker_k")


def _entropy(weights: np.ndarray) -> float:
    weights = weights.astype(float)
    total = weights.sum()
    if total <= 0:
        return math.nan
    p = weights / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _subset_name(example: BaseExample) -> str:
    return "repeated_marker_examples" if has_repeated_markers(example) else "unique_marker_examples"


@torch.no_grad()
def run_attention(models: dict[str, Any], vocab: Vocab, cfg, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples = balanced_examples(cfg.seq_len, cfg.eval.attention_examples_per_count, cfg.seed + 40_000)
    rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    batch_size = max(1, min(16, int(cfg.train.batch_size)))

    non_model = models["non_thinking"]
    non_model.eval()
    for start in tqdm(range(0, len(examples), batch_size), desc="attention non-thinking", leave=False):
        chunk = examples[start : start + batch_size]
        prefixes = [non_thinking_eval_prefix(example, vocab) for example in chunk]
        input_ids, lengths = pad_sequences(prefixes, vocab.pad_id, cfg.device)
        out = non_model(
            input_ids=input_ids,
            attention_mask=(input_ids != vocab.pad_id).long(),
            output_attentions=True,
            use_cache=False,
        )
        for local_idx, example in enumerate(chunk):
            ans_pos = int(lengths[local_idx].item()) - 1
            needle_positions = np.array([1 + pos for pos in example.needle_positions], dtype=int)
            body_positions = np.arange(1, 1 + example.seq_len)
            noise_positions = np.array([pos for pos in body_positions if pos not in set(needle_positions.tolist())], dtype=int)
            for layer_idx, attn in enumerate(out.attentions):
                weights = attn[local_idx, :, ans_pos, :].detach().cpu().numpy()
                for head in range(weights.shape[0]):
                    prompt_weights = weights[head, body_positions]
                    needle_mass = float(weights[head, needle_positions].sum())
                    noise_mass = float(weights[head, noise_positions].sum())
                    top_positions = body_positions[np.argsort(prompt_weights)[-example.count :]]
                    recall = len(set(top_positions.tolist()) & set(needle_positions.tolist())) / example.count
                    rows.append(
                        {
                            "model_type": "non_thinking",
                            "query_anchor": "ans_token",
                            "subset": "all_examples",
                            "count": example.count,
                            "count_bin": count_bin(example.count),
                            "layer": layer_idx + 1,
                            "head": head,
                            "ans_to_all_needles_mass": needle_mass,
                            "ans_to_noise_mass": noise_mass,
                            "needle_vs_noise_ratio": needle_mass / (noise_mass + 1e-9),
                            "top_n_retrieval_recall": float(recall),
                            "attention_entropy_over_prompt_body": _entropy(prompt_weights),
                        }
                    )

    think_model = models["thinking_sep_trace"]
    think_model.eval()
    for start in tqdm(range(0, len(examples), batch_size), desc="attention sep-thinking", leave=False):
        chunk = examples[start : start + batch_size]
        rendered = [render_thinking_sep_trace(example, vocab) for example in chunk]
        input_ids, _lengths = pad_sequences([item.input_ids for item in rendered], vocab.pad_id, cfg.device)
        out = think_model(
            input_ids=input_ids,
            attention_mask=(input_ids != vocab.pad_id).long(),
            output_attentions=True,
            use_cache=False,
        )
        for local_idx, (example, item) in enumerate(zip(chunk, rendered)):
            needle_positions = np.array(item.prompt_needle_token_positions, dtype=int)
            body_positions = np.arange(item.spans.seq_start, item.spans.seq_end_exclusive)
            noise_positions = np.array([pos for pos in body_positions if pos not in set(needle_positions.tolist())], dtype=int)
            subset_names = ["all_examples", _subset_name(example)]
            for query_anchor in THINKING_QUERY_ANCHORS:
                for k in range(1, example.count + 1):
                    if query_anchor == "sep_token_k":
                        query_pos = item.spans.sep_positions[k - 1]
                    elif query_anchor == "marker_token_k":
                        query_pos = item.spans.trace_marker_positions[k - 1]
                    elif query_anchor == "pre_sep_k":
                        query_pos = item.spans.sep_positions[k - 1] - 1
                    elif query_anchor == "post_marker_k":
                        query_pos = item.spans.trace_marker_positions[k - 1] + 1
                    else:
                        continue
                    if query_pos >= len(item.input_ids):
                        continue
                    correct_pos = needle_positions[k - 1]
                    for layer_idx, attn in enumerate(out.attentions):
                        weights = attn[local_idx, :, query_pos, :].detach().cpu().numpy()
                        for head in range(weights.shape[0]):
                            needle_weights = weights[head, needle_positions]
                            noise_mass = float(weights[head, noise_positions].sum())
                            diag_mass = float(weights[head, correct_pos])
                            off_mass = float((needle_weights.sum() - diag_mass) / max(1, len(needle_weights) - 1))
                            top_idx = int(np.argmax(needle_weights))
                            for subset in subset_names:
                                rows.append(
                                    {
                                        "model_type": "thinking_sep_trace",
                                        "query_anchor": query_anchor,
                                        "subset": subset,
                                        "count": example.count,
                                        "count_bin": count_bin(example.count),
                                        "layer": layer_idx + 1,
                                        "head": head,
                                        "trace_item_k": k,
                                        "diagonal_mass": diag_mass,
                                        "off_diagonal_mass": off_mass,
                                        "diagonal_dominance": diag_mass / (diag_mass + off_mass + 1e-9),
                                        "correct_top1_rate": float(top_idx == k - 1),
                                        "needle_attention_mass": float(needle_weights.sum()),
                                        "noise_attention_mass": noise_mass,
                                        "needle_vs_noise_ratio": float(needle_weights.sum() / (noise_mass + 1e-9)),
                                        "entropy_over_prompt_body": _entropy(weights[head, body_positions]),
                                    }
                                )
                            matrix_rows.append(
                                {
                                    "query_anchor": query_anchor,
                                    "count": example.count,
                                    "count_bin": count_bin(example.count),
                                    "layer": layer_idx + 1,
                                    "head": head,
                                    "trace_item_k": k,
                                    **{f"needle_j_{j+1}": float(value) for j, value in enumerate(needle_weights.tolist())},
                                }
                            )
    metrics = pd.DataFrame(rows)
    matrices = pd.DataFrame(matrix_rows)
    out_dir = run_dir / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "attention_metrics.csv", index=False)
    matrices.to_csv(out_dir / "attention_trace_matrices_long.csv", index=False)
    return metrics, matrices

