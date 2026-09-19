from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .attention import attention_leaderboard
from .data import BaseExample
from .eval import evaluate_model, summarize_example_rows
from .vocab import Vocab


def run_head_ablation(
    model_by_type: dict[str, Any],
    attention_df: pd.DataFrame,
    examples_by_len: dict[int, list[BaseExample]],
    vocab: Vocab,
    cfg: dict[str, Any],
    seed: int,
    checkpoint_step: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if attention_df.empty:
        return pd.DataFrame(), pd.DataFrame(
            [{"intervention_type": "targeted_attention_masking", "status": "TODO", "reason": "attention analysis not available"}]
        )
    board = attention_leaderboard(attention_df, top_k=4)
    if board.empty:
        return pd.DataFrame(), pd.DataFrame(
            [{"intervention_type": "targeted_attention_masking", "status": "TODO", "reason": "no thinking retrieval heads found"}]
        )
    rows: list[dict[str, Any]] = []
    eval_examples_by_len = {k: v[: min(len(v), 50)] for k, v in examples_by_len.items()}
    for _, head_row in board.iterrows():
        layer = int(head_row["layer"])
        head = int(head_row["head"])
        for model_type, model in model_by_type.items():
            baseline = evaluate_model(
                model,
                model_type,
                eval_examples_by_len,
                vocab,
                cfg["device"],
                seed,
                checkpoint_step,
                batch_size=min(64, int(cfg["batch_size"])),
            )
            intervened = evaluate_model(
                model,
                model_type,
                eval_examples_by_len,
                vocab,
                cfg["device"],
                seed,
                checkpoint_step,
                batch_size=min(64, int(cfg["batch_size"])),
                ablate_heads={(layer, head)},
            )
            base_sum = summarize_example_rows(
                baseline,
                ["model_type", "seed", "checkpoint_step", "seq_len_eval", "count_bin", "eval_mode"],
            )
            int_sum = summarize_example_rows(
                intervened,
                ["model_type", "seed", "checkpoint_step", "seq_len_eval", "count_bin", "eval_mode"],
            )
            merged = base_sum.merge(
                int_sum,
                on=["model_type", "seed", "checkpoint_step", "seq_len_eval", "count_bin", "eval_mode"],
                suffixes=("_baseline", "_intervened"),
            )
            for _, merged_row in merged.iterrows():
                rows.append(
                    {
                        "model_type": model_type,
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "seq_len_eval": merged_row["seq_len_eval"],
                        "intervention_type": "single_head_zero_ablation",
                        "layer": layer,
                        "head": head,
                        "query_anchor": head_row["query_anchor"],
                        "mask_type": "head_output_zero",
                        "count_bin": merged_row["count_bin"],
                        "eval_mode": merged_row["eval_mode"],
                        "baseline_final_accuracy": merged_row.get("final_accuracy_baseline", math.nan),
                        "intervened_final_accuracy": merged_row.get("final_accuracy_intervened", math.nan),
                        "delta_final_accuracy": merged_row.get("final_accuracy_intervened", math.nan)
                        - merged_row.get("final_accuracy_baseline", math.nan),
                        "baseline_trace_exact": merged_row.get("trace_exact_rate_baseline", math.nan),
                        "intervened_trace_exact": merged_row.get("trace_exact_rate_intervened", math.nan),
                        "delta_trace_exact": merged_row.get("trace_exact_rate_intervened", math.nan)
                        - merged_row.get("trace_exact_rate_baseline", math.nan),
                        "baseline_logit_margin": merged_row.get("final_answer_logit_margin_baseline", math.nan),
                        "intervened_logit_margin": merged_row.get("final_answer_logit_margin_intervened", math.nan),
                        "delta_logit_margin": merged_row.get("final_answer_logit_margin_intervened", math.nan)
                        - merged_row.get("final_answer_logit_margin_baseline", math.nan),
                    }
                )
    masking_todo = pd.DataFrame(
        [
            {
                "intervention_type": "targeted_attention_masking",
                "status": "TODO",
                "reason": "single-head ablation is implemented; targeted masking requires a custom per-query attention mask hook",
            }
        ]
    )
    return pd.DataFrame(rows), masking_todo
