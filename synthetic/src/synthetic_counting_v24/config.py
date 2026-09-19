from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V24Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V24Config:
    """Build the paired v22-grammar experiment with counts restricted to 1..10."""

    overrides.update(
        version="v24",
        count_tokenization="atomic",
        trace_format="separator",
        count_max_threshold=10,
        needle_pool_frequency_threshold=10.0 / 256.0,
        final_count_loss_weight=1.0,
        cot_trace_loss_weight=1.0,
        enabled_model_variants=("rope/nonthinking", "rope/thinking"),
    )
    return _v20_preset(preset, **overrides)


__all__ = ["V24Config", "preset_config"]
