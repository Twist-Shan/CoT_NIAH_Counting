from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import count_bin
from .data import BaseExample
from .generation import generation_row, greedy_generate, pad_sequences
from .model import count_logits
from .render import non_thinking_eval_prefix, render_for_model, thinking_generation_prefix, thinking_oracle_trace_prefix
from .vocab import Vocab


def _nan_trace_metrics() -> dict[str, float]:
    return {
        "trace_exact_match_rate": math.nan,
        "trace_marker_precision": math.nan,
        "trace_marker_recall": math.nan,
        "trace_delimiter_count_accuracy": math.nan,
        "premature_close_rate": math.nan,
        "missing_close_rate": math.nan,
        "ans_generated_rate": math.nan,
        "think_close_generated_rate": math.nan,
    }


@torch.no_grad()
def predict_counts_from_prefixes(
    model,
    prefixes: list[list[int]],
    gold_counts: list[int],
    vocab: Vocab,
    device: str | torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(prefixes), batch_size):
        chunk = prefixes[start : start + batch_size]
        gold = gold_counts[start : start + batch_size]
        input_ids, lengths = pad_sequences(chunk, vocab.pad_id, device)
        attention_mask = (input_ids != vocab.pad_id).long()
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[torch.arange(input_ids.size(0), device=device), lengths - 1]
        restricted = count_logits(logits, vocab)
        target = torch.tensor([count - 1 for count in gold], dtype=torch.long, device=device)
        pred = restricted.argmax(dim=-1) + 1
        ce = F.cross_entropy(restricted, target, reduction="none")
        for local_idx, pred_count_t in enumerate(pred.detach().cpu().tolist()):
            gold_count = int(gold[local_idx])
            pred_count = int(pred_count_t)
            rows.append(
                {
                    "pred_count": pred_count,
                    "accuracy": float(pred_count == gold_count),
                    "mae": abs(pred_count - gold_count),
                    "under_rate": float(pred_count < gold_count),
                    "over_rate": float(pred_count > gold_count),
                    "invalid_rate": 0.0,
                    "eval_final_answer_loss": float(ce[local_idx].detach().cpu()),
                }
            )
    return rows


@torch.no_grad()
def teacher_forced_losses(
    model,
    examples: list[BaseExample],
    vocab: Vocab,
    model_type: str,
    device: str | torch.device,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    rendered = [render_for_model(example, vocab, model_type) for example in examples]
    by_id: dict[str, dict[str, float]] = {}
    model.eval()
    for start in range(0, len(rendered), batch_size):
        chunk = rendered[start : start + batch_size]
        max_len = max(len(item.input_ids) for item in chunk)
        input_ids = torch.full((len(chunk), max_len), vocab.pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(chunk), max_len), -100, dtype=torch.long, device=device)
        for row_idx, item in enumerate(chunk):
            input_ids[row_idx, : len(item.input_ids)] = torch.tensor(item.input_ids, dtype=torch.long, device=device)
            labels[row_idx, : len(item.labels)] = torch.tensor(item.labels, dtype=torch.long, device=device)
        out = model(input_ids=input_ids, attention_mask=(input_ids != vocab.pad_id).long(), use_cache=False)
        log_probs = F.log_softmax(out.logits[:, :-1, :], dim=-1)
        for row_idx, item in enumerate(chunk):
            positions = [pos for pos, label in enumerate(item.labels) if label != -100 and pos > 0]
            ces = [-float(log_probs[row_idx, pos - 1, item.labels[pos]].detach().cpu()) for pos in positions]
            final_ce = -float(log_probs[row_idx, item.spans.final_count_pos - 1, item.input_ids[item.spans.final_count_pos]].detach().cpu())
            trace_positions = [pos for pos in positions if pos < item.spans.ans_pos]
            trace_ces = [-float(log_probs[row_idx, pos - 1, item.labels[pos]].detach().cpu()) for pos in trace_positions]
            by_id[examples[start + row_idx].example_id] = {
                "eval_completion_loss": float(np.mean(ces)) if ces else math.nan,
                "eval_trace_loss": float(np.mean(trace_ces)) if trace_ces else math.nan,
                "eval_final_answer_loss_full_vocab": final_ce,
            }
    return by_id


def _aggregate_rows(rows: list[dict[str, Any]], group_key: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [
        "accuracy",
        "invalid_rate",
        "eval_completion_loss",
        "eval_trace_loss",
        "eval_final_answer_loss",
        "mae",
        "under_rate",
        "over_rate",
        "trace_exact_match_rate",
        "trace_marker_precision",
        "trace_marker_recall",
        "trace_delimiter_count_accuracy",
        "premature_close_rate",
        "missing_close_rate",
        "ans_generated_rate",
        "think_close_generated_rate",
    ]
    grouped = df.groupby(["step", "model_type", "eval_mode", group_key], dropna=False)
    out = grouped[metric_cols].mean(numeric_only=True).reset_index()
    out["n_examples"] = grouped.size().to_numpy()
    return out


def evaluate_all(
    models: dict[str, Any],
    examples: list[BaseExample],
    vocab: Vocab,
    device: str | torch.device,
    step: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for model_type, model in models.items():
        if model_type == "non_thinking":
            prefixes = [non_thinking_eval_prefix(example, vocab) for example in examples]
            pred_rows = predict_counts_from_prefixes(model, prefixes, [example.count for example in examples], vocab, device, batch_size)
            losses = teacher_forced_losses(model, examples, vocab, model_type, device, batch_size)
            for example, pred_row in zip(examples, pred_rows):
                rows.append(
                    {
                        "step": step,
                        "model_type": model_type,
                        "eval_mode": "direct",
                        "example_id": example.example_id,
                        "count": example.count,
                        "count_bin": count_bin(example.count),
                        **losses[example.example_id],
                        **_nan_trace_metrics(),
                        **pred_row,
                    }
                )
            continue

        oracle_prefixes = [thinking_oracle_trace_prefix(example, vocab) for example in examples]
        oracle_rows = predict_counts_from_prefixes(model, oracle_prefixes, [example.count for example in examples], vocab, device, batch_size)
        losses = teacher_forced_losses(model, examples, vocab, model_type, device, batch_size)
        for example, pred_row in zip(examples, oracle_rows):
            rows.append(
                {
                    "step": step,
                    "model_type": model_type,
                    "eval_mode": "oracle_trace_final_readout",
                    "example_id": example.example_id,
                    "count": example.count,
                    "count_bin": count_bin(example.count),
                    **losses[example.example_id],
                    "trace_exact_match_rate": 1.0,
                    "trace_marker_precision": 1.0,
                    "trace_marker_recall": 1.0,
                    "trace_delimiter_count_accuracy": 1.0,
                    "premature_close_rate": 0.0,
                    "missing_close_rate": 0.0,
                    "ans_generated_rate": 1.0,
                    "think_close_generated_rate": 1.0,
                    **pred_row,
                }
            )

        gen_prefixes = [thinking_generation_prefix(example, vocab) for example in examples]
        gen_ids = greedy_generate(model, gen_prefixes, vocab, device, max_new_tokens=2 * 10 + 4, batch_size=max(1, min(32, batch_size)))
        for example, ids in zip(examples, gen_ids):
            tokens = vocab.decode(ids, skip_pad=True)
            gen_row = generation_row(tokens, example)
            rows.append(
                {
                    "step": step,
                    "model_type": model_type,
                    "eval_mode": "generated_trace",
                    "example_id": example.example_id,
                    "count": example.count,
                    "count_bin": count_bin(example.count),
                    "eval_completion_loss": losses[example.example_id]["eval_completion_loss"],
                    "eval_trace_loss": losses[example.example_id]["eval_trace_loss"],
                    "eval_final_answer_loss": math.nan,
                    "eval_final_answer_loss_full_vocab": losses[example.example_id]["eval_final_answer_loss_full_vocab"],
                    **gen_row,
                }
            )
    detail = pd.DataFrame(rows)
    by_count = _aggregate_rows(rows, "count")
    by_bin = _aggregate_rows(rows, "count_bin")
    return detail, by_count, by_bin

