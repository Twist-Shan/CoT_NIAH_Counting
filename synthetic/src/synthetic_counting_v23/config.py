from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V23Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V23Config:
    """Build the paired v23 no-index configuration.

    v23 preserves the v22 separator trace but retrains both Non-thinking and
    Thinking with an 8x final-count loss weight.  The weight is part of the
    version definition and therefore cannot be changed through overrides.
    """

    overrides.update(
        version="v23",
        count_tokenization="atomic",
        trace_format="separator",
        final_count_loss_weight=8.0,
    )
    overrides.setdefault(
        "enabled_model_variants",
        ("rope/nonthinking", "rope/thinking"),
    )
    return _v20_preset(preset, **overrides)


__all__ = ["V23Config", "preset_config"]
