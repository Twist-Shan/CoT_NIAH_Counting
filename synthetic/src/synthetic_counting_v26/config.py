from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V26Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V26Config:
    """Build the v24.3-matched untied-native-head control.

    Both modes keep the 256-character prompt, 100 three-character marker
    sets, exactly balanced count sampling, separator trace, model capacity,
    seed, optimizer, and component-normalized objective used by v24.3.  The
    only changed training field is ``tie_word_embeddings=False``.
    """

    # Keep the debug data geometry viable: the 100-set/count-10 pool cannot
    # reliably supply every held-out count bucket in the generic 48-character
    # debug window.  Debug therefore retains L=256 but keeps the six-step,
    # batch-four CPU schedule from the shared debug preset.
    if preset == "debug":
        overrides.update(seq_len=256, n_positions=384)
    overrides.update(
        version="v26",
        count_tokenization="atomic",
        trace_format="separator",
        count_max_threshold=10,
        needle_pool_size=100,
        needle_pool_frequency_threshold=10.0 / 256.0,
        training_count_distribution="uniform",
        final_count_loss_weight=1.0,
        cot_trace_loss_weight=1.0,
        task_output_loss_reduction="component_normalized",
        task_output_count_weight=1.0,
        task_output_trace_weight=1.0,
        task_output_structure_weight=0.1,
        tie_word_embeddings=False,
        answer_query_contrastive_weight=0.0,
        enabled_model_variants=("rope/nonthinking", "rope/thinking"),
    )
    return _v20_preset(preset, **overrides)


__all__ = ["V26Config", "preset_config"]
