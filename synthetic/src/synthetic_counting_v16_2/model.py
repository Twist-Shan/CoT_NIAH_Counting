"""Task-independent Transformer wrapper for v16_2.

The causal Transformer implementation is unchanged from v11-v17. All task-specific
configuration, data, training, and evaluation remain isolated in this package.
"""

from synthetic_counting_v11.model import (  # noqa: F401
    CausalLMOutput,
    TinyPositionCausalLM,
    build_model,
)

__all__ = ["CausalLMOutput", "TinyPositionCausalLM", "build_model"]
