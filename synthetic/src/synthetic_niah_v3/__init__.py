"""Synthetic NIAH-style symbolic counting v3.

v3 intentionally removes the old loss-mask ablation. It trains exactly two
model types per seed: a direct non-thinking counter and an indexed-thinking
trace model.
"""

__all__ = ["__version__"]

__version__ = "0.3.0"
