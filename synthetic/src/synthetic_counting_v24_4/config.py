from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V244Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V244Config:
    """Build the v24.3-matched maximum-entropy set/count sampler control."""

    overrides.update(
        version="v24.4",
        count_tokenization="atomic",
        trace_format="separator",
        count_max_threshold=10,
        needle_pool_frequency_threshold=10.0 / 256.0,
        training_count_distribution="maxent_set_count",
        final_count_loss_weight=1.0,
        cot_trace_loss_weight=1.0,
        task_output_loss_reduction="component_normalized",
        task_output_count_weight=1.0,
        task_output_trace_weight=1.0,
        task_output_structure_weight=0.1,
        enabled_model_variants=("rope/nonthinking", "rope/thinking"),
    )
    return _v20_preset(preset, **overrides)


__all__ = ["V244Config", "preset_config"]
