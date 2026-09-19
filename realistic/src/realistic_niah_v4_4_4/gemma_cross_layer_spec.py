from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "realistic_niah_v4_4_4_gemma_cross_layer_ov_v1"
PROTOCOL_VERSION = "realistic_niah_v4_4_4_gemma_frozen_cross_layer_set_v1"


@dataclass(frozen=True, order=True)
class FrozenSite:
    layer: int
    head: int

    @classmethod
    def from_value(cls, value: Any) -> "FrozenSite":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("A frozen site must be [layer, head]")
        return cls(int(value[0]), int(value[1]))

    def to_list(self) -> list[int]:
        return [int(self.layer), int(self.head)]


@dataclass(frozen=True)
class GemmaCrossLayerConfig:
    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    model_label: str = "Gemma4-E4B"
    candidate_sites: tuple[FrozenSite, ...] = (
        FrozenSite(29, 4),
        FrozenSite(35, 2),
    )
    matched_control_sets: tuple[tuple[FrozenSite, ...], ...] = (
        (FrozenSite(29, 1), FrozenSite(35, 3)),
        (FrozenSite(29, 6), FrozenSite(35, 2)),
        (FrozenSite(29, 7), FrozenSite(35, 6)),
    )
    direction_discovery_seeds: tuple[int, ...] = tuple(range(1234, 1254))
    center_seeds: tuple[int, ...] = tuple(range(1426, 1436))
    confirmation_seeds: tuple[int, ...] = tuple(range(1436, 1456))
    counts: tuple[int, ...] = tuple(range(1, 11))
    fit_counts: tuple[int, ...] = (1, 3, 5, 7, 9)
    heldout_counts: tuple[int, ...] = (2, 4, 6, 8, 10)
    causal_counts: tuple[int, ...] = (2, 5, 8)
    injection_betas: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    mediation_pairs: tuple[tuple[int, int], ...] = ((1, 6), (3, 8), (5, 10))
    relay_mediator_layer: int = 35
    terminal_trace_layer: int = 41
    primary_alpha: float = 0.025
    bootstrap_repetitions: int = 10_000
    model_torch_dtype: str = "bfloat16"
    attention_prefix_backend: str = "sdpa"
    attention_cache_logit_tolerance: float = 0.5
    write_raw_attention_rows: bool = False
    write_full_hidden_states: bool = False
    atomic_shards: bool = True
    selection_source: str = (
        "correct-only broad-retrieval K2 ranked set and three frozen "
        "layer-matched random replicates"
    )
    selection_status: str = "frozen_before_single_head_natural_ov_outcomes"

    @property
    def dataset_seeds(self) -> tuple[int, ...]:
        return (
            self.direction_discovery_seeds
            + self.center_seeds
            + self.confirmation_seeds
        )

    @property
    def dataset_discovery_seeds(self) -> tuple[int, ...]:
        return self.direction_discovery_seeds + self.center_seeds

    @property
    def matched_control_count(self) -> int:
        return len(self.matched_control_sets)

    @property
    def secondary_nested_head_sets(self) -> tuple[tuple[int, ...], ...]:
        return ()

    @property
    def include_factorial_single_heads(self) -> bool:
        return False

    @property
    def primary_decision_rule(self) -> str:
        return (
            "intersection_union_joint_natural_injection_removal_mediation_"
            "plus_l29_to_l35_relay_all_p_le_alpha"
        )

    @property
    def candidate_heads(self) -> tuple[int, ...]:
        # Compatibility only for the generic statistical analysis.  Site
        # identity is always serialized separately and never inferred here.
        return tuple(site.head for site in self.candidate_sites)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GemmaCrossLayerConfig":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown cross-layer config keys: {unknown}")
        values = dict(payload)
        if "candidate_sites" in values:
            values["candidate_sites"] = tuple(
                FrozenSite.from_value(value) for value in values["candidate_sites"]
            )
        if "matched_control_sets" in values:
            values["matched_control_sets"] = tuple(
                tuple(FrozenSite.from_value(value) for value in site_set)
                for site_set in values["matched_control_sets"]
            )
        integer_tuple_fields = {
            "direction_discovery_seeds",
            "center_seeds",
            "confirmation_seeds",
            "counts",
            "fit_counts",
            "heldout_counts",
            "causal_counts",
        }
        for name in integer_tuple_fields:
            if name in values:
                values[name] = tuple(int(value) for value in values[name])
        if "injection_betas" in values:
            values["injection_betas"] = tuple(
                float(value) for value in values["injection_betas"]
            )
        if "mediation_pairs" in values:
            values["mediation_pairs"] = tuple(
                tuple(int(value) for value in pair)
                for pair in values["mediation_pairs"]
            )
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "GemmaCrossLayerConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_sites"] = [site.to_list() for site in self.candidate_sites]
        payload["matched_control_sets"] = [
            [site.to_list() for site in site_set]
            for site_set in self.matched_control_sets
        ]
        return payload

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unexpected Gemma cross-layer schema")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected Gemma cross-layer protocol")
        if self.model_label != "Gemma4-E4B":
            raise ValueError("Cross-layer fallback is frozen to Gemma4-E4B")
        if self.candidate_sites != (FrozenSite(29, 4), FrozenSite(35, 2)):
            raise ValueError("Candidate sites changed from frozen K2")
        expected_controls = (
            (FrozenSite(29, 1), FrozenSite(35, 3)),
            (FrozenSite(29, 6), FrozenSite(35, 2)),
            (FrozenSite(29, 7), FrozenSite(35, 6)),
        )
        if self.matched_control_sets != expected_controls:
            raise ValueError("Matched controls changed from frozen K2 random sets")
        all_sets = (self.candidate_sites, *self.matched_control_sets)
        if any(tuple(site.layer for site in site_set) != (29, 35) for site_set in all_sets):
            raise ValueError("Every cross-layer set must contain one L29 and one L35 site")
        if any(len(set(site_set)) != len(site_set) for site_set in all_sets):
            raise ValueError("A cross-layer site set contains duplicates")
        seed_sets = (
            set(self.direction_discovery_seeds),
            set(self.center_seeds),
            set(self.confirmation_seeds),
        )
        if any(not values for values in seed_sets) or any(
            seed_sets[left] & seed_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("Direction, center, and confirmation seeds must be disjoint")
        if self.counts != tuple(range(1, 11)):
            raise ValueError("Counts must be exactly 1 through 10")
        if set(self.fit_counts) | set(self.heldout_counts) != set(self.counts):
            raise ValueError("Fit and held-out counts must cover the registry")
        if set(self.fit_counts) & set(self.heldout_counts):
            raise ValueError("Fit and held-out counts overlap")
        if not set(self.causal_counts).issubset(self.counts):
            raise ValueError("A causal count lies outside the registry")
        if 0.0 not in self.injection_betas or len(set(self.injection_betas)) < 3:
            raise ValueError("Injection betas need zero and at least three doses")
        if any(
            receiver == donor
            or receiver not in self.counts
            or donor not in self.counts
            for receiver, donor in self.mediation_pairs
        ):
            raise ValueError("Invalid mediation pair")
        if self.relay_mediator_layer != 35 or self.terminal_trace_layer != 41:
            raise ValueError("Relay/terminal layers changed from the frozen protocol")
        if not 0 < self.primary_alpha <= 0.025:
            raise ValueError("Fallback alpha must be in (0, .025]")
        if self.bootstrap_repetitions < 1_000:
            raise ValueError("At least 1,000 bootstrap repetitions are required")
        if self.write_raw_attention_rows or self.write_full_hidden_states:
            raise ValueError("Cross-layer fallback must not persist raw tensors")
        if not self.atomic_shards:
            raise ValueError("Cross-layer fallback requires atomic shards")


GemmaCrossLayerConfig().validate()
