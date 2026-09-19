from __future__ import annotations

from realistic_niah.runner import EngineConfig

from .spec import MAX_MODEL_LEN, MODEL_LABELS


def formal_engine_config(model_label: str) -> EngineConfig:
    """Return the only registered engine configuration for this extension."""

    if model_label not in MODEL_LABELS:
        raise ValueError(f"Unregistered long-context model: {model_label}")
    return EngineConfig(
        tensor_parallel_size=2,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        max_num_seqs=1,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
        request_batch_size=1,
    )
