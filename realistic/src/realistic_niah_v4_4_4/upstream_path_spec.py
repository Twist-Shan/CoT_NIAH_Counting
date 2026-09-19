from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from realistic_niah_v4_4_4.spec import V444Config


SCHEMA_VERSION = "realistic_niah_v4_4_4_upstream_path_v1"
PROTOCOL_VERSION = "realistic_niah_v4_4_4_frozen_broad_to_ov_path_v1"
ROUTES = ("slot_edge_qk", "answer_query_full", "slot_state")


@dataclass(frozen=True)
class FrozenBroadHead:
    layer: int
    head: int
    cue_present_score: float
    cue_absent_score: float

    @property
    def stable_score(self) -> float:
        return min(self.cue_present_score, self.cue_absent_score)

    @classmethod
    def from_sequence(cls, value: Any) -> "FrozenBroadHead":
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("A frozen broad head must be [layer, head, present, absent]")
        return cls(int(value[0]), int(value[1]), float(value[2]), float(value[3]))


@dataclass(frozen=True)
class V444UpstreamPathConfig:
    """Append-only V4.4.4 upstream-read -> L28-write path experiment."""

    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    model_label: str = "Qwen3-8B"
    mediator_layer: int = 28
    evaluation_seeds: tuple[int, ...] = tuple(range(1284, 1294))
    counts: tuple[int, ...] = tuple(range(1, 11))
    donor_pairs: tuple[tuple[int, int], ...] = (
        (1, 6), (6, 1), (3, 8), (8, 3), (5, 10), (10, 5)
    )
    early_candidates: tuple[FrozenBroadHead, ...] = ()
    early_set_sizes: tuple[int, ...] = (2, 4, 8)
    routes: tuple[str, ...] = ROUTES
    late_head_sets: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("base_h16_h19", (16, 19)),
        ("gqa_h16_h19", (16, 17, 18, 19)),
        ("broad_top4", (19, 16, 17, 2)),
        ("broad_top8", (19, 16, 17, 2, 18, 23, 1, 11)),
    )
    primary_late_set: str = "base_h16_h19"
    primary_behavior_metric: str = "donor_log_odds_gain"
    primary_alpha: float = 0.05
    bootstrap_repetitions: int = 10_000
    expand_late_if_base_insufficient: bool = True
    block_closure_relative_tolerance: float = 1e-5
    control_orthogonality_tolerance: float = 1e-4
    persist_full_states: bool = False
    persist_raw_attention: bool = False
    atomic_shards: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "V444UpstreamPathConfig":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown upstream-path config keys: {unknown}")
        values = dict(payload)
        for name in ("evaluation_seeds", "counts", "early_set_sizes"):
            if name in values:
                values[name] = tuple(int(item) for item in values[name])
        if "donor_pairs" in values:
            values["donor_pairs"] = tuple(
                tuple(int(item) for item in pair) for pair in values["donor_pairs"]
            )
        if "early_candidates" in values:
            values["early_candidates"] = tuple(
                FrozenBroadHead.from_sequence(item) for item in values["early_candidates"]
            )
        if "routes" in values:
            values["routes"] = tuple(str(item) for item in values["routes"])
        if "late_head_sets" in values:
            values["late_head_sets"] = tuple(
                (str(name), tuple(int(head) for head in heads))
                for name, heads in values["late_head_sets"]
            )
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "V444UpstreamPathConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["early_candidates"] = [
            [item.layer, item.head, item.cue_present_score, item.cue_absent_score]
            for item in self.early_candidates
        ]
        payload["late_head_sets"] = [
            [name, list(heads)] for name, heads in self.late_head_sets
        ]
        return payload

    @property
    def late_sets(self) -> dict[str, tuple[int, ...]]:
        return dict(self.late_head_sets)

    @property
    def early_layers(self) -> tuple[int, ...]:
        return tuple(sorted({item.layer for item in self.early_candidates}))

    def early_set(self, size: int) -> tuple[FrozenBroadHead, ...]:
        if int(size) not in self.early_set_sizes:
            raise KeyError(size)
        return self.early_candidates[: int(size)]

    def validate_against_base(self, base: V444Config) -> None:
        base.validate()
        self.validate()
        if self.model_label != base.model_label or self.mediator_layer != base.layer:
            raise ValueError("Upstream-path/base model or mediator layer disagrees")
        if self.late_sets[self.primary_late_set] != base.candidate_heads:
            raise ValueError("Primary late set must remain frozen V4.4.4 L28 {H16,H19}")
        if self.counts != base.counts:
            raise ValueError("Upstream-path/base count registries disagree")
        if not set(self.evaluation_seeds).issubset(base.confirmation_seeds):
            raise ValueError("Pilot seeds must be a declared subset of V4.4.4 confirmation seeds")

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected upstream-path schema/protocol version")
        if self.model_label != "Qwen3-8B" or self.mediator_layer != 28:
            raise ValueError("Upstream-path pilot is frozen to Qwen3-8B L28")
        if len(self.evaluation_seeds) != 10 or len(set(self.evaluation_seeds)) != 10:
            raise ValueError("Upstream-path pilot requires ten unique seeds")
        if self.counts != tuple(range(1, 11)):
            raise ValueError("Counts must be exactly 1 through 10")
        if not self.donor_pairs or any(
            receiver == donor or receiver not in self.counts or donor not in self.counts
            for receiver, donor in self.donor_pairs
        ):
            raise ValueError("Donor pairs must be unequal registered counts")
        if {(b, a) for a, b in self.donor_pairs} != set(self.donor_pairs):
            raise ValueError("Every donor pair needs its reverse direction")
        if len(self.early_candidates) < max(self.early_set_sizes):
            raise ValueError("Frozen candidate registry is shorter than a requested top-k")
        keys = [(item.layer, item.head) for item in self.early_candidates]
        if len(keys) != len(set(keys)) or any(not 0 <= layer < 28 for layer, _ in keys):
            raise ValueError("Frozen early heads must be unique and strictly upstream of L28")
        stable = [item.stable_score for item in self.early_candidates]
        if any(left + 5e-5 < right for left, right in zip(stable, stable[1:])):
            raise ValueError("Frozen early candidates are not sorted by stable broad score")
        if tuple(sorted(set(self.early_set_sizes))) != self.early_set_sizes:
            raise ValueError("Early top-k sizes must be sorted and unique")
        if self.routes != ROUTES:
            raise ValueError(f"Routes must remain frozen to {ROUTES}")
        late_names = [name for name, _heads in self.late_head_sets]
        if len(late_names) != len(set(late_names)) or self.primary_late_set not in late_names:
            raise ValueError("Late head-set names must be unique and include the primary")
        for _name, heads in self.late_head_sets:
            if not heads or len(heads) != len(set(heads)) or any(not 0 <= h < 32 for h in heads):
                raise ValueError("Every L28 set must contain unique heads in [0,32)")
        if self.late_sets[self.primary_late_set] != (16, 19):
            raise ValueError("Primary L28 set is frozen to H16/H19")
        if self.primary_behavior_metric != "donor_log_odds_gain":
            raise ValueError("Primary behavior metric must be donor sequence log-odds gain")
        if not 0 < self.primary_alpha < 1 or self.bootstrap_repetitions < 1000:
            raise ValueError("Invalid inferential settings")
        if self.block_closure_relative_tolerance <= 0 or self.control_orthogonality_tolerance <= 0:
            raise ValueError("Causal audit tolerances must be positive")
        if self.persist_full_states or self.persist_raw_attention or not self.atomic_shards:
            raise ValueError("Only summary rows in atomic shards may be persisted")
