from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .spec import (
    PROTOCOL_VERSION as SINGLE_PROTOCOL_VERSION,
    SCHEMA_VERSION as SINGLE_SCHEMA_VERSION,
    V443Config,
)


SET_SCHEMA_VERSION = "realistic_niah_v4_4_3_ov_set_causal_v1"
SET_PROTOCOL_VERSION = "realistic_niah_v4_4_3_ov_set_causal_preregistered_v1"


@dataclass(frozen=True)
class V443SetConfig(V443Config):
    """Frozen small-head-set extension of V4.4.3.

    K=1 remains the earlier single-head baseline.  This protocol selects
    model-capacity-specific nested set sizes using discovery fit counts only, then evaluates
    held-out geometry, screen patches, and disjoint confirmation interventions.
    """

    schema_version: str = SET_SCHEMA_VERSION
    protocol_version: str = SET_PROTOCOL_VERSION
    set_sizes_qwen: tuple[int, ...] = (2, 3, 4, 6, 8)
    set_sizes_gemma: tuple[int, ...] = (2, 3, 4)
    set_selection_metric: str = "greedy_nested_fit_cosine_of_summed_ov_vectors"
    set_null_samples: int = 10_000
    set_control_norm_pool: int = 128
    set_injection_boundary: str = (
        "post_o_direction_projected_into_selected_set_output_span"
    )
    single_head_baseline_run_root: str = (
        "runs/v4_4_3_ov_causal/"
        "run_20260803_v4_4_3_ov_causal_a100_1501368870_v3"
    )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "V443SetConfig":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown V4.4.3-Set config keys: {unknown}")
        values = dict(payload)
        tuple_fields = {
            "model_labels",
            "discovery_seeds",
            "screen_seeds",
            "confirmation_seeds",
            "fit_counts",
            "heldout_counts",
            "target_output_layers_qwen",
            "target_output_layers_gemma",
            "patch_interventions",
            "injection_counts",
            "injection_betas",
            "direction_interventions",
            "generation_beta_magnitudes",
            "set_sizes_qwen",
            "set_sizes_gemma",
        }
        for name in tuple_fields:
            if name in values:
                values[name] = tuple(values[name])
        for name in ("qwen_sentinel_heads", "gemma_sentinel_heads", "patch_pairs"):
            if name in values:
                values[name] = tuple(
                    tuple(int(item) for item in pair) for pair in values[name]
                )
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "V443SetConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def set_sizes_for(self, model_label: str) -> tuple[int, ...]:
        if model_label == "Qwen3-8B":
            return self.set_sizes_qwen
        if model_label == "Gemma4-E4B":
            return self.set_sizes_gemma
        raise KeyError(f"Unregistered V4.4.3-Set model: {model_label}")

    def validate(self) -> None:
        base_names = {field.name for field in fields(V443Config)}
        base_payload = {
            name: getattr(self, name)
            for name in base_names
        }
        base_payload["schema_version"] = SINGLE_SCHEMA_VERSION
        base_payload["protocol_version"] = SINGLE_PROTOCOL_VERSION
        V443Config(**base_payload).validate()
        if self.schema_version != SET_SCHEMA_VERSION:
            raise ValueError("Unexpected V4.4.3-Set schema version")
        if self.protocol_version != SET_PROTOCOL_VERSION:
            raise ValueError("Unexpected V4.4.3-Set protocol version")
        for model_label in self.model_labels:
            sizes = self.set_sizes_for(model_label)
            if tuple(sorted(set(sizes))) != sizes:
                raise ValueError(f"{model_label} set sizes must be unique and sorted")
            if not sizes or any(int(size) <= 1 for size in sizes):
                raise ValueError("V4.4.3-Set sizes must all exceed one")
        if self.set_selection_metric != (
            "greedy_nested_fit_cosine_of_summed_ov_vectors"
        ):
            raise ValueError("Unexpected set-selection metric")
        if self.set_null_samples < 100:
            raise ValueError("set_null_samples must be at least 100")
        if self.set_control_norm_pool < 2:
            raise ValueError("set_control_norm_pool must be at least two")
        if self.set_injection_boundary != (
            "post_o_direction_projected_into_selected_set_output_span"
        ):
            raise ValueError("Unexpected set-injection boundary")
        if not str(self.single_head_baseline_run_root).strip():
            raise ValueError("single_head_baseline_run_root is required")
