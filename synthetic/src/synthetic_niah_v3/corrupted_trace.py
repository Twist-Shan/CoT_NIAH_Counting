from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .data import BaseExample, count_bin
from .eval import predict_next_number
from .render import thinking_corrupted_trace_prefix, trace_tokens_for_example
from .trace_parse import token_to_number
from .vocab import MARKER_TOKENS, Vocab, number_token


CORRUPTION_TYPES = [
    "oracle_trace",
    "wrong_indices_correct_markers",
    "correct_indices_wrong_markers",
    "shuffled_trace_order",
    "deleted_one_item",
    "duplicated_one_item",
    "extra_random_item",
    "last_index_replaced",
    "indices_removed",
    "markers_removed",
    "empty_trace",
]


def split_trace_pairs(trace_tokens: list[str]) -> list[tuple[str | None, str | None]]:
    pairs: list[tuple[str | None, str | None]] = []
    idx = 0
    while idx < len(trace_tokens):
        first = trace_tokens[idx]
        second = trace_tokens[idx + 1] if idx + 1 < len(trace_tokens) else None
        if token_to_number(first) is not None:
            pairs.append((first, second))
            idx += 2
        else:
            pairs.append((None, first))
            idx += 1
    return pairs


def corrupt_trace(example: BaseExample, corruption_type: str, rng: random.Random) -> tuple[list[str], bool]:
    gold = trace_tokens_for_example(example)
    if corruption_type == "oracle_trace":
        return gold, False
    if corruption_type == "wrong_indices_correct_markers":
        out = []
        for _, marker in split_trace_pairs(gold):
            wrong = rng.randint(1, 10)
            out.extend([number_token(wrong), marker or rng.choice(MARKER_TOKENS)])
        return out, False
    if corruption_type == "correct_indices_wrong_markers":
        out = []
        for idx, _ in split_trace_pairs(gold):
            out.extend([idx or "<1>", rng.choice(MARKER_TOKENS)])
        return out, False
    if corruption_type == "shuffled_trace_order":
        pairs = split_trace_pairs(gold)
        rng.shuffle(pairs)
        return [tok for pair in pairs for tok in pair if tok is not None], False
    if corruption_type == "deleted_one_item":
        pairs = split_trace_pairs(gold)
        if not pairs:
            return [], True
        del pairs[rng.randrange(len(pairs))]
        return [tok for pair in pairs for tok in pair if tok is not None], False
    if corruption_type == "duplicated_one_item":
        if example.count >= 10:
            return gold, True
        pairs = split_trace_pairs(gold)
        pair = pairs[rng.randrange(len(pairs))]
        insert_at = rng.randrange(len(pairs) + 1)
        pairs.insert(insert_at, pair)
        return [tok for pair in pairs for tok in pair if tok is not None], False
    if corruption_type == "extra_random_item":
        if example.count >= 10:
            return gold, True
        return gold + [number_token(example.count + 1), rng.choice(MARKER_TOKENS)], False
    if corruption_type == "last_index_replaced":
        out = list(gold)
        index_positions = list(range(0, len(out), 2))
        if index_positions:
            last = index_positions[-1]
            choices = [i for i in range(1, 11) if i != example.count]
            out[last] = number_token(rng.choice(choices))
        return out, False
    if corruption_type == "indices_removed":
        return [tok for i, tok in enumerate(gold) if i % 2 == 1], False
    if corruption_type == "markers_removed":
        return [tok for i, tok in enumerate(gold) if i % 2 == 0], False
    if corruption_type == "empty_trace":
        return [], False
    raise ValueError(f"Unknown corruption type: {corruption_type}")


def trace_rule_labels(example: BaseExample, trace_tokens: list[str]) -> dict[str, int | None]:
    pairs = split_trace_pairs(trace_tokens)
    index_values = [token_to_number(tok) for tok in trace_tokens if token_to_number(tok) is not None]
    marker_count = sum(1 for tok in trace_tokens if tok in MARKER_TOKENS)
    pair_count = sum(1 for idx, marker in pairs if idx is not None or marker is not None)
    return {
        "prompt_count": example.count,
        "trace_pair_count": pair_count,
        "last_index_value": index_values[-1] if index_values else None,
        "max_index_value": max(index_values) if index_values else None,
        "marker_count_in_trace": marker_count,
    }


def classify_follow_rules(pred_count: int | None, labels: dict[str, int | None]) -> dict[str, bool]:
    return {
        "follows_prompt_count": pred_count == labels["prompt_count"],
        "follows_trace_pair_count": pred_count == labels["trace_pair_count"],
        "follows_last_index": pred_count is not None and labels["last_index_value"] is not None and pred_count == labels["last_index_value"],
        "follows_max_index": pred_count is not None and labels["max_index_value"] is not None and pred_count == labels["max_index_value"],
        "follows_marker_count": pred_count == labels["marker_count_in_trace"],
    }


def run_corrupted_trace_eval(
    model,
    examples_by_len: dict[int, list[BaseExample]],
    vocab: Vocab,
    device: str,
    seed: int,
    checkpoint_step: int,
    batch_size: int,
) -> pd.DataFrame:
    rng = random.Random(seed + 9917)
    rows: list[dict[str, Any]] = []
    for seq_len, examples in examples_by_len.items():
        for corruption_type in CORRUPTION_TYPES:
            prefixes = []
            gold_counts = []
            metadata = []
            for example in examples:
                trace_tokens, skipped = corrupt_trace(example, corruption_type, rng)
                if skipped:
                    continue
                labels = trace_rule_labels(example, trace_tokens)
                prefixes.append(thinking_corrupted_trace_prefix(example, trace_tokens, vocab))
                gold_counts.append(example.count)
                metadata.append((example, labels))
            if not prefixes:
                continue
            pred_rows = predict_next_number(model, prefixes, gold_counts, vocab, device, batch_size=batch_size)
            for (example, labels), pred_row in zip(metadata, pred_rows):
                pred = pred_row["pred_count"]
                rows.append(
                    {
                        "model_type": "thinking",
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "seq_len_eval": seq_len,
                        "count": example.count,
                        "count_bin": count_bin(example.count),
                        "corruption_type": corruption_type,
                        **labels,
                        "pred_count": pred,
                        "correct_prompt_count": pred == example.count,
                        "invalid": False,
                        **classify_follow_rules(pred, labels),
                    }
                )
    return pd.DataFrame(rows)


def summarize_follow_rules(corrupt_df: pd.DataFrame) -> pd.DataFrame:
    if corrupt_df.empty:
        return pd.DataFrame()
    follow_cols = [col for col in corrupt_df.columns if col.startswith("follows_")]
    value_cols = ["correct_prompt_count", *follow_cols]
    return corrupt_df.groupby(["corruption_type", "seq_len_eval", "count_bin"], as_index=False)[value_cols].mean()
