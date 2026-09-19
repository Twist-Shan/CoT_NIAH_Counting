"""v21 digit-wise control for the v20 query-first RoPE experiment.

All training and analysis code is shared deliberately; only the vocabulary and
number rendering selected by ``count_tokenization='digitwise'`` differ.
"""

from .config import V21Config, preset_config


def run_v21_pipeline(*args, **kwargs):
    """Import the plotting/training stack only when a run is actually started."""

    from .pipeline import run_v21_pipeline as implementation

    return implementation(*args, **kwargs)

__all__ = ["V21Config", "preset_config", "run_v21_pipeline"]
