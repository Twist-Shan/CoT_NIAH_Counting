from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "realistic_niah_v4_4_3_ov_causal_v1"
PROTOCOL_VERSION = "realistic_niah_v4_4_3_ov_causal_preregistered_v1"


@dataclass(frozen=True)
class V443Config:
    """Frozen V4.4.3 design.

    Layer and head indices are zero based, matching the existing V4 artifacts.
    Geometry fitting never uses the screen or confirmation seeds.  The staged
    patch screen and directed-intervention confirmation use disjoint seeds.
    """

    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    source_variant: str = "v4.4"
    source_answer_format: str = "numeric"
    model_labels: tuple[str, ...] = ("Qwen3-8B", "Gemma4-E4B")
    discovery_seeds: tuple[int, ...] = tuple(range(1234, 1254))
    screen_seeds: tuple[int, ...] = tuple(range(1254, 1259))
    confirmation_seeds: tuple[int, ...] = tuple(range(1259, 1264))
    fit_counts: tuple[int, ...] = (1, 3, 5, 7, 9)
    heldout_counts: tuple[int, ...] = (2, 4, 6, 8, 10)
    prompt_pooling: str = "span_end"
    target_output_layers_qwen: tuple[int, ...] = (28, 29, 30)
    target_output_layers_gemma: tuple[int, ...] = (36, 37, 38)
    heads_per_layer: int = 1
    qwen_sentinel_heads: tuple[tuple[int, int], ...] = ((29, 3),)
    gemma_sentinel_heads: tuple[tuple[int, int], ...] = ()
    mapping_null_repetitions: int = 10_000
    mapping_norm_match_pool: int = 8
    patch_pairs: tuple[tuple[int, int], ...] = ((1, 6), (3, 8), (5, 10))
    patch_interventions: tuple[str, ...] = (
        "alpha_receiver_v",
        "alpha_position_scramble",
        "z_donor",
        "o_donor",
        "output_norm_control",
    )
    injection_counts: tuple[int, ...] = (2, 5, 8)
    injection_betas: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    direction_interventions: tuple[str, ...] = (
        "answer_direction_removal",
        "equal_norm_orthogonal_removal",
        "signed_answer_direction_injection",
    )
    position_scramble_fraction: float = 0.37
    generation_beta_magnitudes: tuple[float, ...] = (-2.0, 2.0)
    generation_max_new_tokens: int = 16
    candidate_score_reduction: str = (
        "sum_log_probability_over_answer_and_chat_termination"
    )
    causal_primary_metric: str = "delta_expected_count"
    local_selectivity_metric: str = "non_count_token_kl"
    hidden_source_dtype: str = "float16"
    compute_dtype: str = "float32"
    model_torch_dtype: str = "bfloat16"
    attention_prefix_backend: str = "sdpa"
    attention_cache_logit_tolerance: float = 0.5
    write_raw_attention_rows: bool = False
    write_full_hidden_states: bool = False
    atomic_shards: bool = True
    strict_zo_equivalence_tolerance: float = 0.05

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "V443Config":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown V4.4.3 config keys: {unknown}")
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
    def from_json(cls, path: str | Path) -> "V443Config":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def target_layers(self, model_label: str) -> tuple[int, ...]:
        if model_label == "Qwen3-8B":
            return self.target_output_layers_qwen
        if model_label == "Gemma4-E4B":
            return self.target_output_layers_gemma
        raise KeyError(f"Unregistered V4.4.3 model: {model_label}")

    def sentinel_heads(self, model_label: str) -> tuple[tuple[int, int], ...]:
        if model_label == "Qwen3-8B":
            return self.qwen_sentinel_heads
        if model_label == "Gemma4-E4B":
            return self.gemma_sentinel_heads
        raise KeyError(f"Unregistered V4.4.3 model: {model_label}")

    @property
    def directed_patch_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            directed
            for low, high in self.patch_pairs
            for directed in ((int(low), int(high)), (int(high), int(low)))
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unexpected V4.4.3 schema version")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected V4.4.3 protocol version")
        if self.source_variant != "v4.4" or self.source_answer_format != "numeric":
            raise ValueError("V4.4.3 is frozen to the numeric V4.4 condition")
        seed_sets = tuple(
            set(values)
            for values in (
                self.discovery_seeds,
                self.screen_seeds,
                self.confirmation_seeds,
            )
        )
        if any(not values for values in seed_sets):
            raise ValueError("Every V4.4.3 seed split must be nonempty")
        if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("V4.4.3 seed splits must be disjoint")
        if set(self.fit_counts) & set(self.heldout_counts):
            raise ValueError("Direction fit and held-out counts must be disjoint")
        if set(self.fit_counts) | set(self.heldout_counts) != set(range(1, 11)):
            raise ValueError("Direction count splits must partition counts 1 through 10")
        if self.prompt_pooling not in {"span_end", "span_mean"}:
            raise ValueError("Unsupported prompt counter pooling")
        if self.heads_per_layer < 1:
            raise ValueError("heads_per_layer must be positive")
        if any(layer < 1 for layer in self.target_output_layers_qwen):
            raise ValueError("OV mapping needs a saved previous-layer prompt state")
        if any(layer < 1 for layer in self.target_output_layers_gemma):
            raise ValueError("OV mapping needs a saved previous-layer prompt state")
        if self.mapping_null_repetitions < 100:
            raise ValueError("mapping_null_repetitions must be at least 100")
        if self.mapping_norm_match_pool < 2:
            raise ValueError("mapping_norm_match_pool must be at least two")
        if not self.patch_pairs or any(low >= high for low, high in self.patch_pairs):
            raise ValueError("patch_pairs must be unique low-to-high count pairs")
        if len(set(self.patch_pairs)) != len(self.patch_pairs):
            raise ValueError("patch_pairs must be unique")
        expected_patch = {
            "alpha_receiver_v",
            "alpha_position_scramble",
            "z_donor",
            "o_donor",
            "output_norm_control",
        }
        if set(self.patch_interventions) != expected_patch:
            raise ValueError("Unexpected staged-patch intervention registry")
        if 0.0 not in self.injection_betas:
            raise ValueError("Injection grid must contain the exact zero baseline")
        if tuple(sorted(self.injection_betas)) != self.injection_betas:
            raise ValueError("Injection betas must be sorted")
        if not 0.0 < float(self.position_scramble_fraction) < 1.0:
            raise ValueError("position_scramble_fraction must lie strictly in (0, 1)")
        if self.causal_primary_metric != "delta_expected_count":
            raise ValueError("Unexpected V4.4.3 primary causal metric")
        if self.candidate_score_reduction != (
            "sum_log_probability_over_answer_and_chat_termination"
        ):
            raise ValueError("V4.4.3 requires frozen joint sequence scoring")
        if self.local_selectivity_metric != "non_count_token_kl":
            raise ValueError("Unexpected local selectivity metric")
        if self.write_raw_attention_rows or self.write_full_hidden_states:
            raise ValueError("V4.4.3 must not persist raw attention rows or full states")
        if not self.atomic_shards:
            raise ValueError("Filestream shards must be written atomically")
        if self.strict_zo_equivalence_tolerance <= 0:
            raise ValueError("Z/O equivalence tolerance must be positive")
        if self.attention_cache_logit_tolerance <= 0:
            raise ValueError("Attention cache/full logit tolerance must be positive")
