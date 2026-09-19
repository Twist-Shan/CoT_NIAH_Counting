from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V25Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V25Config:
    """Build the v24.7-objective retrieval-pressure setting.

    Main uses a 1,024-character prompt while retaining counts 1..10 and the
    unchanged no-index trace.  The 1,056-token model budget is exactly enough
    for the longest rendered Thinking example.  Debug keeps the same semantic
    design on a small sequence/count support so CPU smoke tests remain cheap.
    """

    if preset == "main":
        length = 1_024
        count_max = 10
        positions = 1_056
        batch_size = 32
    elif preset == "debug":
        length = 64
        count_max = 4
        positions = 96
        batch_size = 4
    else:
        raise ValueError(f"unknown preset: {preset}")
    fixed = {
        "version": "v25",
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "seq_len": length,
        "n_positions": positions,
        "count_max_threshold": count_max,
        "needle_pool_size": 20,
        "needle_pool_frequency_threshold": float(count_max) / float(length),
        "training_count_distribution": "maxent_set_count",
        "final_count_loss_weight": 1.0,
        "cot_trace_loss_weight": 1.0,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 1.0,
        "task_output_trace_weight": 1.0,
        "task_output_structure_weight": 0.1,
        "tie_word_embeddings": False,
        "answer_query_contrastive_weight": 0.1,
        "answer_query_contrastive_temperature": 0.1,
        "batch_size": batch_size,
        "enabled_model_variants": ("rope/nonthinking", "rope/thinking"),
    }
    fixed.update(overrides)
    return _v20_preset(preset, **fixed)


__all__ = ["V25Config", "preset_config"]
