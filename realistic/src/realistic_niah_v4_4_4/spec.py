from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "realistic_niah_v4_4_4_natural_ov_v1"
PROTOCOL_VERSION = "realistic_niah_v4_4_4_natural_ov_preregistered_v1"


@dataclass(frozen=True)
class V444Config:
    """Frozen single-candidate confirmation of a natural OV transporter.

    The layer and heads come from V4.4.3 discovery.  New center/control-selection
    seeds are disjoint from the new causal confirmation seeds.  No layer, K, or
    head search is permitted after causal outcomes are observed.
    """

    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    model_label: str = "Qwen3-8B"
    layer: int = 28
    candidate_heads: tuple[int, ...] = (16, 19)
    secondary_nested_head_sets: tuple[tuple[int, ...], ...] = (
        (16, 19, 31),
        (16, 18, 19, 31),
        (1, 3, 16, 18, 19, 31),
        (1, 3, 12, 14, 16, 18, 19, 31),
    )
    direction_discovery_seeds: tuple[int, ...] = tuple(range(1234, 1254))
    center_seeds: tuple[int, ...] = tuple(range(1264, 1274))
    confirmation_seeds: tuple[int, ...] = tuple(range(1274, 1294))
    counts: tuple[int, ...] = tuple(range(1, 11))
    fit_counts: tuple[int, ...] = (1, 3, 5, 7, 9)
    heldout_counts: tuple[int, ...] = (2, 4, 6, 8, 10)
    causal_counts: tuple[int, ...] = (2, 5, 8)
    injection_betas: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    mediation_pairs: tuple[tuple[int, int], ...] = ((1, 6), (3, 8), (5, 10))
    matched_control_count: int = 4
    control_match_features: tuple[str, ...] = (
        "log_natural_output_step_norm",
        "natural_output_answer_cosine",
        "reachable_answer_cosine",
        "log_discovery_baseline_output_norm",
    )
    control_requires_same_gqa_relative_positions: bool = True
    include_factorial_single_heads: bool = True
    center_definition: str = "per_head_z_ols_intercept_at_count_zero"
    natural_carrier_definition: str = (
        "dot(WO_set(z-z0),normalize(WO_set(dz)))/norm(WO_set(dz))"
    )
    removal_boundary: str = "pre_o_z_set_realizable_natural_axis"
    mediation_boundary: str = "donor_z_patch_then_pre_o_natural_axis_block"
    primary_decision_rule: str = (
        "intersection_union_natural_injection_removal_mediation_all_p_le_alpha"
    )
    secondary_k_multiplicity_rule: str = "holm_across_registered_k_3_4_6_8"
    primary_alpha: float = 0.05
    bootstrap_repetitions: int = 10_000
    model_torch_dtype: str = "bfloat16"
    attention_prefix_backend: str = "sdpa"
    attention_cache_logit_tolerance: float = 0.5
    pre_o_output_equivalence_tolerance: float = 0.05
    write_raw_attention_rows: bool = False
    write_full_hidden_states: bool = False
    atomic_shards: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "V444Config":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown V4.4.4 config keys: {unknown}")
        values = dict(payload)
        tuple_fields = {
            "candidate_heads",
            "direction_discovery_seeds",
            "center_seeds",
            "confirmation_seeds",
            "counts",
            "fit_counts",
            "heldout_counts",
            "causal_counts",
            "injection_betas",
            "control_match_features",
        }
        for name in tuple_fields:
            if name in values:
                values[name] = tuple(values[name])
        if "mediation_pairs" in values:
            values["mediation_pairs"] = tuple(
                tuple(int(item) for item in pair)
                for pair in values["mediation_pairs"]
            )
        if "secondary_nested_head_sets" in values:
            values["secondary_nested_head_sets"] = tuple(
                tuple(int(item) for item in heads)
                for heads in values["secondary_nested_head_sets"]
            )
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "V444Config":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def dataset_seeds(self) -> tuple[int, ...]:
        return self.center_seeds + self.confirmation_seeds

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unexpected V4.4.4 schema version")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected V4.4.4 protocol version")
        if self.model_label != "Qwen3-8B" or self.layer != 28:
            raise ValueError("V4.4.4 is frozen to Qwen3-8B layer 28")
        if self.candidate_heads != (16, 19):
            raise ValueError("V4.4.4 is frozen to the L28 heads {16,19} core")
        expected_nested = (
            (16, 19, 31),
            (16, 18, 19, 31),
            (1, 3, 16, 18, 19, 31),
            (1, 3, 12, 14, 16, 18, 19, 31),
        )
        if self.secondary_nested_head_sets != expected_nested:
            raise ValueError("Unexpected registered L28 nested-K robustness sets")
        if tuple(len(heads) for heads in self.secondary_nested_head_sets) != (
            3,
            4,
            6,
            8,
        ):
            raise ValueError("Registered nested sets must have K=3,4,6,8")
        if any(
            not set(self.candidate_heads).issubset(heads)
            or len(set(heads)) != len(heads)
            for heads in self.secondary_nested_head_sets
        ):
            raise ValueError("Every registered nested set must contain the K=2 core")
        seed_groups = (
            set(self.direction_discovery_seeds),
            set(self.center_seeds),
            set(self.confirmation_seeds),
        )
        if any(not group for group in seed_groups):
            raise ValueError("Every seed split must be nonempty")
        if any(
            seed_groups[i] & seed_groups[j]
            for i in range(3)
            for j in range(i + 1, 3)
        ):
            raise ValueError("Direction, center, and confirmation seeds must be disjoint")
        if tuple(sorted(set(self.counts))) != self.counts or self.counts != tuple(
            range(1, 11)
        ):
            raise ValueError("V4.4.4 counts must be exactly 1 through 10")
        if set(self.fit_counts) & set(self.heldout_counts):
            raise ValueError("Fit and held-out counts overlap")
        if set(self.fit_counts) | set(self.heldout_counts) != set(self.counts):
            raise ValueError("Fit and held-out counts must partition 1 through 10")
        if not set(self.causal_counts).issubset(self.counts):
            raise ValueError("Causal counts must be registered counts")
        if tuple(sorted(self.injection_betas)) != self.injection_betas:
            raise ValueError("Injection betas must be sorted")
        if 0.0 not in self.injection_betas or not any(
            beta < 0 for beta in self.injection_betas
        ) or not any(beta > 0 for beta in self.injection_betas):
            raise ValueError("Injection grid must contain zero and both signs")
        if not self.mediation_pairs or any(
            low >= high or low not in self.counts or high not in self.counts
            for low, high in self.mediation_pairs
        ):
            raise ValueError("Mediation pairs must be valid low-to-high count pairs")
        if self.matched_control_count < 2:
            raise ValueError("At least two matched controls are required")
        expected_features = {
            "log_natural_output_step_norm",
            "natural_output_answer_cosine",
            "reachable_answer_cosine",
            "log_discovery_baseline_output_norm",
        }
        if set(self.control_match_features) != expected_features:
            raise ValueError("Unexpected control matching feature registry")
        if self.center_definition != "per_head_z_ols_intercept_at_count_zero":
            raise ValueError("Unexpected Z-center definition")
        if self.primary_decision_rule != (
            "intersection_union_natural_injection_removal_mediation_all_p_le_alpha"
        ):
            raise ValueError("Unexpected primary decision rule")
        if self.secondary_k_multiplicity_rule != (
            "holm_across_registered_k_3_4_6_8"
        ):
            raise ValueError("Unexpected nested-K multiplicity rule")
        if not 0.0 < float(self.primary_alpha) < 1.0:
            raise ValueError("primary_alpha must lie in (0,1)")
        if self.bootstrap_repetitions < 1_000:
            raise ValueError("bootstrap_repetitions must be at least 1000")
        if self.model_torch_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Unsupported model dtype")
        if self.attention_prefix_backend not in {"sdpa", "flash_attention_2"}:
            raise ValueError("Unsupported attention backend")
        if self.attention_cache_logit_tolerance <= 0:
            raise ValueError("Attention cache tolerance must be positive")
        if self.pre_o_output_equivalence_tolerance <= 0:
            raise ValueError("Pre-O equivalence tolerance must be positive")
        if self.write_raw_attention_rows or self.write_full_hidden_states:
            raise ValueError("V4.4.4 must not persist raw attention/full hidden states")
        if not self.atomic_shards:
            raise ValueError("V4.4.4 requires atomic shards")


V444Config().validate()
