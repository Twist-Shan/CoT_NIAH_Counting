from __future__ import annotations

from typing import Any

from synthetic_counting_v20.config import V20Config, preset_config as _v20_preset


V37Config = V20Config


def preset_config(preset: str = "debug", **overrides: Any) -> V37Config:
    """Build the independent equal-component low-LR-tail screen."""

    if preset == "debug":
        overrides.update(seq_len=256, n_positions=384)
    overrides.update(
        version="v37",
        count_tokenization="atomic",
        trace_format="separator",
        count_max_threshold=10,
        needle_pool_size=100,
        needle_pool_frequency_threshold=10.0 / 256.0,
        training_count_distribution="maxent_set_count",
        final_count_loss_weight=1.0,
        cot_trace_loss_weight=1.0,
        task_output_loss_reduction="component_normalized",
        task_output_count_weight=8.0,
        task_output_trace_weight=8.0,
        task_output_structure_weight=8.0,
        answer_query_contrastive_weight=0.0,
        task_output_scheduled_sampling_max_probability=0.0,
        tie_word_embeddings=True,
        untie_atomic_count_readout=True,
        train_steps=8_000,
        lr_decay_steps=6_000,
        min_lr=1e-5,
        phase_cloud_steps=(
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
        ),
        n_layer=4,
        n_head=4,
        n_embd=256,
        n_inner=1024,
        enabled_model_variants=("rope/nonthinking", "rope/thinking"),
    )
    return _v20_preset(preset, **overrides)


__all__ = ["V37Config", "preset_config"]
