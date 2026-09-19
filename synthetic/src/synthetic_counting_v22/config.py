from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V22Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V22Config:
    """Build the matched v20 no-index configuration.

    Only the Thinking model changes, so the canonical v22 run omits the
    unchanged Non-thinking model.  Callers may explicitly override
    ``enabled_model_variants`` when a standalone paired rerun is desired.
    """

    overrides.setdefault("enabled_model_variants", ("rope/thinking",))
    overrides.update(
        version="v22",
        count_tokenization="atomic",
        trace_format="separator",
    )
    return _v20_preset(preset, **overrides)


__all__ = ["V22Config", "preset_config"]
