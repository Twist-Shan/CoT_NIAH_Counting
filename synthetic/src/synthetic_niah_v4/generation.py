from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from .data import BaseExample, count_bin
from .model import count_logits
from .render import non_thinking_eval_prefix, thinking_generation_prefix, thinking_oracle_trace_prefix, trace_tokens_for_example
from .vocab import Vocab


def pad_sequences(sequences: list[list[int]], pad_id: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long, device=device)
    for row_idx, seq in enumerate(sequences):
        input_ids[row_idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return input_ids, lengths


@torch.no_grad()
def predict_count_from_prefixes(
    model,
    prefixes: list[list[int]],
    gold_counts: list[int],
    vocab: Vocab,
    device: str | torch.device,
    batch_size: int = 128,
    hook_fn=None,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(prefixes), batch_size):
        chunk = prefixes[start : start + batch_size]
        gold = gold_counts[start : start + batch_size]
        input_ids, lengths = pad_sequences(chunk, vocab.pad_id, device)
        attention_mask = (input_ids != vocab.pad_id).long()
        out = model(input_ids, attention_mask=attention_mask, hook_fn=hook_fn)
        logits = out.logits[torch.arange(input_ids.size(0), device=input_ids.device), lengths - 1]
        restricted = count_logits(logits, vocab.count_ids)
        pred_idx = restricted.argmax(dim=-1)
        ce = F.cross_entropy(
            restricted,
            torch.tensor([value - 1 for value in gold], dtype=torch.long, device=restricted.device),
            reduction="none",
        )
        log_probs = F.log_softmax(restricted, dim=-1)
        for local_idx, pred_offset in enumerate(pred_idx.detach().cpu().tolist()):
            pred = int(pred_offset) + 1
            gold_count = int(gold[local_idx])
            target_plus = min(10, gold_count + 1)
            rows.append(
                {
                    "pred_count": pred,
                    "final_accuracy": float(pred == gold_count),
                    "final_mae": abs(pred - gold_count),
                    "gold_logit": float(restricted[local_idx, gold_count - 1].detach().cpu()),
                    "gold_logprob": float(log_probs[local_idx, gold_count - 1].detach().cpu()),
                    "target_plus_one_logprob": float(log_probs[local_idx, target_plus - 1].detach().cpu()),
                    "final_answer_ce": float(ce[local_idx].detach().cpu()),
                    "count_token_rate": 1.0,
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
) -> list[list[int]]:
    model.eval()
    outputs: list[list[int]] = []
    for start in range(0, len(prefixes), batch_size):
        chunk = prefixes[start : start + batch_size]
        input_ids, lengths = pad_sequences(chunk, vocab.pad_id, device)
        generated = [[] for _ in chunk]
        active = torch.ones(len(chunk), dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            attention_mask = (input_ids != vocab.pad_id).long()
            out = model(input_ids, attention_mask=attention_mask)
            logits = out.logits[torch.arange(input_ids.size(0), device=device), lengths - 1]
            next_ids = logits.argmax(dim=-1)
            for row_idx, token_id in enumerate(next_ids.detach().cpu().tolist()):
                if active[row_idx]:
                    generated[row_idx].append(int(token_id))
            active = active & (next_ids != vocab.eos_id)
            input_ids = torch.cat([input_ids, next_ids[:, None]], dim=1)
            lengths = lengths + 1
            if not bool(active.any()):
                break
        outputs.extend(generated)
    return outputs


def parse_generated_count(tokens: list[str]) -> int | None:
    if "<Ans>" not in tokens:
        return None
    ans_idx = tokens.index("<Ans>")
    if ans_idx + 1 >= len(tokens):
        return None
    token = tokens[ans_idx + 1]
    if token.startswith("<") and token.endswith(">") and token[1:-1].isdigit():
        value = int(token[1:-1])
        if 1 <= value <= 10:
            return value
    return None


def trace_exact(tokens: list[str], example: BaseExample) -> float:
    expected = trace_tokens_for_example(example) + ["</Think>", "<Ans>", f"<{example.count}>"]
    return float(tokens[: len(expected)] == expected)


def evaluate_behavior(
    models: dict[str, Any],
    examples: list[BaseExample],
    vocab: Vocab,
    device: str | torch.device,
    batch_size: int = 128,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_type, model in models.items():
        if model_type == "non_thinking":
            prefixes = [non_thinking_eval_prefix(ex, vocab) for ex in examples]
            out_rows = predict_count_from_prefixes(model, prefixes, [ex.count for ex in examples], vocab, device, batch_size)
            for ex, out in zip(examples, out_rows):
                rows.append(
                    {
                        "model_type": model_type,
                        "eval_mode": "direct",
                        "example_id": ex.example_id,
                        "count": ex.count,
                        "count_bin": count_bin(ex.count),
                        "invalid_rate": 0.0,
                        "trace_exact_rate": float("nan"),
                        **out,
                    }
                )
        else:
            oracle_prefixes = [thinking_oracle_trace_prefix(ex, vocab) for ex in examples]
            oracle_rows = predict_count_from_prefixes(model, oracle_prefixes, [ex.count for ex in examples], vocab, device, batch_size)
            for ex, out in zip(examples, oracle_rows):
                rows.append(
                    {
                        "model_type": model_type,
                        "eval_mode": "oracle_trace",
                        "example_id": ex.example_id,
                        "count": ex.count,
                        "count_bin": count_bin(ex.count),
                        "invalid_rate": 0.0,
                        "trace_exact_rate": 1.0,
                        **out,
                    }
                )
            gen_prefixes = [thinking_generation_prefix(ex, vocab) for ex in examples]
            generated = greedy_generate(model, gen_prefixes, vocab, device, max_new_tokens=2 * 10 + 4, batch_size=max(1, min(32, batch_size)))
            for ex, gen_ids in zip(examples, generated):
                gen_tokens = vocab.decode(gen_ids, skip_pad=True)
                pred = parse_generated_count(gen_tokens)
                rows.append(
                    {
                        "model_type": model_type,
                        "eval_mode": "generated_trace",
                        "example_id": ex.example_id,
                        "count": ex.count,
                        "count_bin": count_bin(ex.count),
                        "pred_count": pred if pred is not None else -1,
                        "final_accuracy": float(pred == ex.count),
                        "final_mae": abs(pred - ex.count) if pred is not None else float("nan"),
                        "invalid_rate": float(pred is None),
                        "trace_exact_rate": trace_exact(gen_tokens, ex),
                        "count_token_rate": float(pred is not None),
                    }
                )
    return pd.DataFrame(rows)
