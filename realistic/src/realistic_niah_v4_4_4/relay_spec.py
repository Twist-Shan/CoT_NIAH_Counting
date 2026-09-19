from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .spec import V444Config


RELAY_SCHEMA_VERSION = "realistic_niah_v4_4_4_relay_v1"
RELAY_PROTOCOL_VERSION = "realistic_niah_v4_4_4_relay_serial_mediation_v1"


@dataclass(frozen=True)
class V444RelayConfig:
    """Frozen relay-to-OV extension of the completed V4.4.4 campaign.

    Relay discovery uses only natural source contributions on the original
    center seeds.  The selected position-set rule is frozen before the causal
    confirmation rows are generated.  The downstream layer, head set, and OV
    axis are inherited unchanged from V4.4.4.
    """

    schema_version: str = RELAY_SCHEMA_VERSION
    protocol_version: str = RELAY_PROTOCOL_VERSION
    model_label: str = "Qwen3-8B"
    layer: int = 28
    heads: tuple[int, ...] = (16, 19)
    discovery_seeds: tuple[int, ...] = tuple(range(1264, 1274))
    confirmation_seeds: tuple[int, ...] = tuple(range(1274, 1294))
    counts: tuple[int, ...] = tuple(range(1, 11))
    fit_counts: tuple[int, ...] = (1, 3, 5, 7, 9)
    heldout_counts: tuple[int, ...] = (2, 4, 6, 8, 10)
    removal_counts: tuple[int, ...] = (2, 5, 8)
    relay_pairs: tuple[tuple[int, int], ...] = (
        (1, 6),
        (6, 1),
        (3, 8),
        (8, 3),
        (5, 10),
        (10, 5),
    )
    semantic_relay_sets: tuple[str, ...] = (
        "answer_query_self",
        "pre_query_non_slot_tail_16",
        "pre_query_non_slot_tail_64",
    )
    ranked_non_slot_sizes: tuple[int, ...] = (4, 8, 16, 32, 64)
    source_control_sets: tuple[str, ...] = (
        "active_needle_endpoints",
        "active_needle_spans",
    )
    selection_rule: str = (
        "max_positive_discovery_seed_t_score_then_mean_slope_then_name"
    )
    selection_min_positive_seed_fraction: float = 0.7
    primary_decision_rule: str = (
        "intersection_union_natural_edge_patch_mediation_removal_all_p_le_alpha"
    )
    primary_alpha: float = 0.05
    bootstrap_repetitions: int = 10_000
    attention_cache_logit_tolerance: float = 0.5
    contribution_reconstruction_relative_tolerance: float = 0.05
    pre_o_output_equivalence_tolerance: float = 0.05
    persist_raw_position_contributions: bool = False
    persist_full_value_states: bool = False
    atomic_shards: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "V444RelayConfig":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown V4.4.4 relay config keys: {unknown}")
        values = dict(payload)
        tuple_fields = {
            "heads",
            "discovery_seeds",
            "confirmation_seeds",
            "counts",
            "fit_counts",
            "heldout_counts",
            "removal_counts",
            "semantic_relay_sets",
            "ranked_non_slot_sizes",
            "source_control_sets",
        }
        for name in tuple_fields:
            if name in values:
                values[name] = tuple(values[name])
        if "relay_pairs" in values:
            values["relay_pairs"] = tuple(
                tuple(int(item) for item in pair) for pair in values["relay_pairs"]
            )
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "V444RelayConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ranked_relay_sets(self) -> tuple[str, ...]:
        return tuple(f"non_slot_top_{size}" for size in self.ranked_non_slot_sizes)

    @property
    def eligible_relay_sets(self) -> tuple[str, ...]:
        return self.semantic_relay_sets + self.ranked_relay_sets

    @property
    def all_position_sets(self) -> tuple[str, ...]:
        return self.eligible_relay_sets + self.source_control_sets

    def validate_against_base(self, base: V444Config) -> None:
        base.validate()
        self.validate()
        if self.model_label != base.model_label:
            raise ValueError("Relay/base model labels disagree")
        if self.layer != base.layer or self.heads != base.candidate_heads:
            raise ValueError("Relay must inherit the frozen V4.4.4 layer/head set")
        if self.discovery_seeds != base.center_seeds:
            raise ValueError("Relay discovery must reuse the frozen center seeds")
        if self.confirmation_seeds != base.confirmation_seeds:
            raise ValueError("Relay confirmation must reuse the frozen confirmation seeds")
        if self.counts != base.counts:
            raise ValueError("Relay/base count registries disagree")
        if self.fit_counts != base.fit_counts or self.heldout_counts != base.heldout_counts:
            raise ValueError("Relay/base fit-count partition disagrees")
        if self.removal_counts != base.causal_counts:
            raise ValueError("Relay removal counts must inherit V4.4.4 causal counts")

    def validate(self) -> None:
        if self.schema_version != RELAY_SCHEMA_VERSION:
            raise ValueError("Unexpected V4.4.4 relay schema version")
        if self.protocol_version != RELAY_PROTOCOL_VERSION:
            raise ValueError("Unexpected V4.4.4 relay protocol version")
        if self.model_label != "Qwen3-8B" or self.layer != 28:
            raise ValueError("Relay extension is frozen to Qwen3-8B L28")
        if self.heads != (16, 19):
            raise ValueError("Relay extension is frozen to heads {16,19}")
        if not self.discovery_seeds or not self.confirmation_seeds:
            raise ValueError("Relay seed splits must be nonempty")
        if set(self.discovery_seeds) & set(self.confirmation_seeds):
            raise ValueError("Relay discovery and confirmation seeds overlap")
        if self.counts != tuple(range(1, 11)):
            raise ValueError("Relay counts must be exactly 1 through 10")
        if set(self.fit_counts) & set(self.heldout_counts):
            raise ValueError("Relay fit and held-out counts overlap")
        if set(self.fit_counts) | set(self.heldout_counts) != set(self.counts):
            raise ValueError("Relay fit/held-out counts must partition all counts")
        if not set(self.removal_counts).issubset(self.counts):
            raise ValueError("Relay removal counts are outside the count registry")
        if not self.relay_pairs or any(
            receiver == donor
            or receiver not in self.counts
            or donor not in self.counts
            for receiver, donor in self.relay_pairs
        ):
            raise ValueError("Relay pairs must be directed unequal registered counts")
        reverse = {(donor, receiver) for receiver, donor in self.relay_pairs}
        if reverse != set(self.relay_pairs):
            raise ValueError("Every relay pair must include its reverse direction")
        allowed_semantic = {
            "answer_query_self",
            "pre_query_non_slot_tail_16",
            "pre_query_non_slot_tail_64",
        }
        if not self.semantic_relay_sets or not set(self.semantic_relay_sets).issubset(
            allowed_semantic
        ):
            raise ValueError("Unexpected semantic relay position set")
        if (
            not self.ranked_non_slot_sizes
            or tuple(sorted(set(self.ranked_non_slot_sizes)))
            != self.ranked_non_slot_sizes
            or any(size <= 0 for size in self.ranked_non_slot_sizes)
        ):
            raise ValueError("Ranked non-slot sizes must be sorted unique positives")
        if set(self.source_control_sets) != {
            "active_needle_endpoints",
            "active_needle_spans",
        }:
            raise ValueError("Unexpected relay source-control registry")
        if len(set(self.all_position_sets)) != len(self.all_position_sets):
            raise ValueError("Relay position-set names must be unique")
        if self.selection_rule != (
            "max_positive_discovery_seed_t_score_then_mean_slope_then_name"
        ):
            raise ValueError("Unexpected relay selection rule")
        if not 0.5 <= self.selection_min_positive_seed_fraction <= 1.0:
            raise ValueError("Relay positive-seed threshold must lie in [0.5,1]")
        if self.primary_decision_rule != (
            "intersection_union_natural_edge_patch_mediation_removal_all_p_le_alpha"
        ):
            raise ValueError("Unexpected relay primary decision rule")
        if not 0.0 < self.primary_alpha < 1.0:
            raise ValueError("Relay alpha must lie in (0,1)")
        if self.bootstrap_repetitions < 1_000:
            raise ValueError("Relay bootstrap repetitions must be at least 1000")
        if self.attention_cache_logit_tolerance <= 0:
            raise ValueError("Relay cache-logit tolerance must be positive")
        if self.contribution_reconstruction_relative_tolerance <= 0:
            raise ValueError("Relay reconstruction tolerance must be positive")
        if self.pre_o_output_equivalence_tolerance <= 0:
            raise ValueError("Relay pre-O tolerance must be positive")
        if self.persist_raw_position_contributions or self.persist_full_value_states:
            raise ValueError("Relay extension must not persist raw maps/full V states")
        if not self.atomic_shards:
            raise ValueError("Relay extension requires atomic shards")


V444RelayConfig().validate()
