from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V29Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V29Config:
    """Build the v28-matched count-region-weight experiment."""

    if preset == "debug":
        overrides.update(seq_len=256, n_positions=384)
    overrides.update(
        version="v29",
        count_tokenization="atomic",
        trace_format="separator",
        count_max_threshold=10,
        needle_pool_size=100,
        needle_pool_frequency_threshold=10.0 / 256.0,
        training_count_distribution="uniform",
        final_count_loss_weight=1.0,
        cot_trace_loss_weight=1.0,
        task_output_loss_reduction="component_normalized",
        task_output_count_weight=4.0,
        task_output_trace_weight=1.0,
        task_output_structure_weight=0.1,
        tie_word_embeddings=True,
        untie_atomic_count_readout=True,
        answer_query_contrastive_weight=0.0,
        enabled_model_variants=("rope/nonthinking", "rope/thinking"),
    )
    return _v20_preset(preset, **overrides)


__all__ = ["V29Config", "preset_config"]
