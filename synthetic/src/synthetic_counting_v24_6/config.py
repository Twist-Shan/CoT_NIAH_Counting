from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V246Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V246Config:
    """Build the v24.5-matched untied-LM-head readout control."""

    overrides.update(
        version="v24.6",
        count_tokenization="atomic",
        trace_format="separator",
        count_max_threshold=10,
        needle_pool_size=20,
        needle_pool_frequency_threshold=10.0 / 256.0,
        training_count_distribution="maxent_set_count",
        final_count_loss_weight=1.0,
        cot_trace_loss_weight=1.0,
        task_output_loss_reduction="component_normalized",
        task_output_count_weight=1.0,
        task_output_trace_weight=1.0,
        task_output_structure_weight=0.1,
        tie_word_embeddings=False,
        enabled_model_variants=("rope/nonthinking", "rope/thinking"),
    )
    return _v20_preset(preset, **overrides)


__all__ = ["V246Config", "preset_config"]
