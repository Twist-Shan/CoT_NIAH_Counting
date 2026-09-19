from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V21Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V21Config:
    overrides.update(version="v21", count_tokenization="digitwise")
    return _v20_preset(preset, **overrides)


__all__ = ["V21Config", "preset_config"]
