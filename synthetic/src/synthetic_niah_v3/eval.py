from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import BaseExample, count_bin
from .render import non_thinking_eval_prefix, thinking_generation_prefix, thinking_oracle_trace_prefix
from .trace_parse import parse_thinking_generation, trace_metrics
from .vocab import Vocab


def pad_sequences(sequences: list[list[int]], pad_id: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long, device=device)
    for idx, seq in enumerate(sequences):
        input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return input_ids, lengths


@torch.no_grad()
def predict_next_number(
    model,
    prefixes: list[list[int]],
    gold_counts: list[int],
    vocab: Vocab,
    device: str | torch.device,
    batch_size: int = 128,
    ablate_heads: set[tuple[int, int]] | None = None,
) -> list[dict]:
    model.eval()
    rows: list[dict] = []
    number_ids = vocab.number_ids
    for start in range(0, len(prefixes), batch_size):
        chunk_prefixes = prefixes[start : start + batch_size]
        chunk_gold = gold_counts[start : start + batch_size]
        input_ids, lengths = pad_sequences(chunk_prefixes, vocab.pad_id, device)
        out = model(input_ids, ablate_heads=ablate_heads)
        logits = out.logits[torch.arange(input_ids.size(0), device=input_ids.device), lengths - 1]
        restricted = logits[:, number_ids]
        pred_offsets = restricted.argmax(dim=-1).detach().cpu().numpy()
        ce = F.cross_entropy(
            restricted,
            torch.tensor([g - 1 for g in chunk_gold], dtype=torch.long, device=restricted.device),
            reduction="none",
        ).detach().cpu().numpy()
        for local_idx, (gold, pred_offset, one_ce) in enumerate(zip(chunk_gold, pred_offsets, ce)):
            pred = int(pred_offset) + 1
            pred_logit = restricted[local_idx, int(pred_offset)].detach().cpu()
            gold_logit = restricted[local_idx, gold - 1].detach().cpu()
            rows.append(
                {
                    "pred_count": pred,
                    "final_accuracy": float(pred == gold),
                    "final_mae": abs(pred - gold),
                    "undercount_rate": float(pred < gold),
                    "overcount_rate": float(pred > gold),
                    "final_answer_ce": float(one_ce),
                    "final_answer_logit_margin": float(gold_logit - pred_logit),
                }
            )
    return rows


@torch.no_grad()
def greedy_generate(
    model,
    prefixes: list[list[int]],
    vocab: Vocab,
    device: str | torch.device,
    max_new_tokens: int,
    batch_size: int = 64,
    ablate_heads: set[tuple[int, int]] | None = None,
) -> list[list[int]]:
    model.eval()
    outputs: list[list[int]] = []
    for start in range(0, len(prefixes), batch_size):
        chunk = prefixes[start : start + batch_size]
        input_ids, _ = pad_sequences(chunk, vocab.pad_id, device)
        lengths = torch.tensor([len(seq) for seq in chunk], dtype=torch.long, device=device)
        active = torch.ones(input_ids.size(0), dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in chunk]
        for _ in range(max_new_tokens):
            out = model(input_ids, ablate_heads=ablate_heads)
            logits = out.logits[torch.arange(input_ids.size(0), device=input_ids.device), lengths - 1]
            next_ids = logits.argmax(dim=-1)
            for row_idx, next_id in enumerate(next_ids.detach().cpu().tolist()):
                if active[row_idx]:
                    generated[row_idx].append(int(next_id))
            active = active & (next_ids != vocab.eos_id)
            input_ids = torch.cat([input_ids, next_ids[:, None]], dim=1)
            lengths = lengths + 1
            if not active.any():
                break
        outputs.extend(generated)
    return outputs


def summarize_example_rows(rows: list[dict] | pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return pd.DataFrame()
        df = rows.copy()
    else:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
    numeric_cols = [col for col in df.columns if col not in group_cols and pd.api.types.is_numeric_dtype(df[col])]
    return df.groupby(group_cols, as_index=False)[numeric_cols].mean()


def evaluate_model(
    model,
    model_type: str,
    examples_by_len: dict[int, list[BaseExample]],
    vocab: Vocab,
    device: str | torch.device,
    seed: int,
    checkpoint_step: int,
    batch_size: int = 128,
    ablate_heads: set[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for seq_len, examples in examples_by_len.items():
        if model_type == "non_thinking":
            prefixes = [non_thinking_eval_prefix(ex, vocab) for ex in examples]
            base_rows = predict_next_number(
                model, prefixes, [ex.count for ex in examples], vocab, device, batch_size=batch_size, ablate_heads=ablate_heads
            )
            for ex, out in zip(examples, base_rows):
                rows.append(
                    {
                        "model_type": model_type,
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "seq_len_eval": seq_len,
                        "count": ex.count,
                        "count_bin": count_bin(ex.count),
                        "eval_mode": "direct",
                        "trace_exact_rate": math.nan,
                        "trace_marker_recall": math.nan,
                        "invalid_generation_rate": math.nan,
                        **out,
                    }
                )
        else:
            prefixes = [thinking_generation_prefix(ex, vocab) for ex in examples]
            generated_ids = greedy_generate(
                model,
                prefixes,
                vocab,
                device,
                max_new_tokens=2 * 10 + 8,
                batch_size=max(8, min(64, batch_size // 2)),
                ablate_heads=ablate_heads,
            )
            for ex, gen in zip(examples, generated_ids):
                gen_tokens = vocab.decode(gen)
                parsed = parse_thinking_generation(gen_tokens)
                metrics = trace_metrics(parsed.trace_tokens, ex.needle_markers)
                pred = parsed.final_count
                rows.append(
                    {
                        "model_type": model_type,
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "seq_len_eval": seq_len,
                        "count": ex.count,
                        "count_bin": count_bin(ex.count),
                        "eval_mode": "generated_trace",
                        "pred_count": pred if pred is not None else -1,
                        "final_accuracy": float(pred == ex.count),
                        "final_mae": abs(pred - ex.count) if pred is not None else math.nan,
                        "undercount_rate": float(pred < ex.count) if pred is not None else math.nan,
                        "overcount_rate": float(pred > ex.count) if pred is not None else math.nan,
                        "final_answer_ce": math.nan,
                        "final_answer_logit_margin": math.nan,
                        "invalid_generation_rate": float(parsed.invalid),
                        **metrics,
                    }
                )
            oracle_prefixes = [thinking_oracle_trace_prefix(ex, vocab) for ex in examples]
            oracle_rows = predict_next_number(
                model,
                oracle_prefixes,
                [ex.count for ex in examples],
                vocab,
                device,
                batch_size=batch_size,
                ablate_heads=ablate_heads,
            )
            for ex, out in zip(examples, oracle_rows):
                rows.append(
                    {
                        "model_type": model_type,
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "seq_len_eval": seq_len,
                        "count": ex.count,
                        "count_bin": count_bin(ex.count),
                        "eval_mode": "oracle_trace",
                        "trace_exact_rate": 1.0,
                        "trace_marker_recall": 1.0,
                        "invalid_generation_rate": 0.0,
                        **out,
                    }
                )
    return pd.DataFrame(rows)


def threshold_table(eval_by_step: pd.DataFrame) -> pd.DataFrame:
    if eval_by_step.empty:
        return pd.DataFrame()
    rows = []
    grouped = eval_by_step.groupby(["model_type", "seed", "eval_mode", "seq_len_eval", "count_bin"], dropna=False)
    for keys, group in grouped:
        group = group.sort_values("checkpoint_step")
        for threshold in [0.90, 0.95, 0.99]:
            hit = group[group["final_accuracy"] >= threshold]
            step = int(hit.iloc[0]["checkpoint_step"]) if not hit.empty else math.nan
            rows.append(
                {
                    "model_type": keys[0],
                    "seed": keys[1],
                    "eval_mode": keys[2],
                    "seq_len_eval": keys[3],
                    "count_bin": keys[4],
                    "threshold": threshold,
                    "step_to_threshold": step,
                    "auc_accuracy_over_training": float(np.trapz(group["final_accuracy"], group["checkpoint_step"]))
                    if len(group) > 1
                    else float(group["final_accuracy"].mean()),
                }
            )
    return pd.DataFrame(rows)
