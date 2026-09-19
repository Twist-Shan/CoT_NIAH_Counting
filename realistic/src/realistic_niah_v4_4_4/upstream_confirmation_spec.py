from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from realistic_niah_v4_4_4.spec import V444Config
from realistic_niah_v4_4_4.upstream_path_spec import FrozenBroadHead


SCHEMA_VERSION = "realistic_niah_v4_4_4_upstream_confirmation_v1"
PROTOCOL_VERSION = "realistic_niah_v4_4_4_early_top4_slot_state_to_l28_confirm_v1"


FROZEN_EARLY_CANDIDATES = (
    FrozenBroadHead(27, 18, 0.4892, 0.4059),
    FrozenBroadHead(23, 28, 0.4359, 0.3472),
    FrozenBroadHead(23, 29, 0.4878, 0.3217),
    FrozenBroadHead(26, 20, 0.2933, 0.3369),
)

FROZEN_LATE_SETS = (
    ("full_h16_h19", (16, 17, 18, 19)),
    ("minus_h16", (17, 18, 19)),
    ("minus_h17", (16, 18, 19)),
    ("minus_h18", (16, 17, 19)),
    ("minus_h19", (16, 17, 18)),
)


@dataclass(frozen=True)
class V444UpstreamConfirmationConfig:
    """Frozen independent-seed confirmation of the exploratory upstream path.

    The primary claim is deliberately narrow: patching the already frozen early
    broad-retrieval top-4 at registered slot-query positions changes donor-vs-
    receiver answer evidence, and that change is specifically mediated by the
    L28 H16--H19 pre-O output channel rather than an equal-norm direction in the
    same W_O span.  Leave-one-out sets are secondary member analyses.
    """

    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    model_label: str = "Qwen3-8B"
    mediator_layer: int = 28
    evaluation_seeds: tuple[int, ...] = tuple(range(1294, 1314))
    counts: tuple[int, ...] = tuple(range(1, 11))
    donor_pairs: tuple[tuple[int, int], ...] = (
        (1, 6),
        (6, 1),
        (3, 8),
        (8, 3),
        (5, 10),
        (10, 5),
    )
    early_candidates: tuple[FrozenBroadHead, ...] = FROZEN_EARLY_CANDIDATES
    early_set_sizes: tuple[int, ...] = (4,)
    routes: tuple[str, ...] = ("slot_state",)
    late_head_sets: tuple[tuple[str, tuple[int, ...]], ...] = FROZEN_LATE_SETS
    primary_late_set: str = "full_h16_h19"
    primary_behavior_metric: str = "donor_log_odds_gain"
    primary_alpha: float = 0.05
    bootstrap_repetitions: int = 20_000
    expand_late_if_base_insufficient: bool = False
    block_closure_relative_tolerance: float = 1e-5
    control_orthogonality_tolerance: float = 1e-4
    persist_full_states: bool = False
    persist_raw_attention: bool = False
    atomic_shards: bool = True

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "V444UpstreamConfirmationConfig":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown upstream-confirmation config keys: {unknown}")
        values = dict(payload)
        for name in ("evaluation_seeds", "counts", "early_set_sizes", "routes"):
            if name in values:
                converter = str if name == "routes" else int
                values[name] = tuple(converter(item) for item in values[name])
        if "donor_pairs" in values:
            values["donor_pairs"] = tuple(
                tuple(int(item) for item in pair) for pair in values["donor_pairs"]
            )
        if "early_candidates" in values:
            values["early_candidates"] = tuple(
                FrozenBroadHead.from_sequence(item)
                for item in values["early_candidates"]
            )
        if "late_head_sets" in values:
            values["late_head_sets"] = tuple(
                (str(name), tuple(int(head) for head in heads))
                for name, heads in values["late_head_sets"]
            )
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_json(
        cls, path: str | Path
    ) -> "V444UpstreamConfirmationConfig":
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
        if int(size) != 4:
            raise KeyError(size)
        return self.early_candidates

    def validate_against_base(self, base: V444Config) -> None:
        base.validate()
        self.validate()
        if self.model_label != base.model_label or self.mediator_layer != base.layer:
            raise ValueError("Confirmation/base model or mediator layer disagrees")
        if self.counts != base.counts:
            raise ValueError("Confirmation/base count registries disagree")
        if self.evaluation_seeds != base.confirmation_seeds:
            raise ValueError("The derivative run must freeze exactly the new seed registry")
        prior = set(base.direction_discovery_seeds) | set(base.center_seeds)
        if prior & set(self.evaluation_seeds):
            raise ValueError("Confirmation seeds overlap direction or center fitting")

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unexpected upstream-confirmation schema")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected upstream-confirmation protocol")
        if self.model_label != "Qwen3-8B" or self.mediator_layer != 28:
            raise ValueError("Confirmation is frozen to Qwen3-8B L28")
        if self.evaluation_seeds != tuple(range(1294, 1314)):
            raise ValueError("Confirmation seeds must remain frozen to 1294--1313")
        if self.counts != tuple(range(1, 11)):
            raise ValueError("Counts must remain 1--10")
        expected_pairs = ((1, 6), (6, 1), (3, 8), (8, 3), (5, 10), (10, 5))
        if self.donor_pairs != expected_pairs:
            raise ValueError("Directed donor pairs changed after preregistration")
        if self.early_candidates != FROZEN_EARLY_CANDIDATES:
            raise ValueError("Frozen V4.4.2 broad-retrieval top-4 changed")
        if self.early_set_sizes != (4,) or self.routes != ("slot_state",):
            raise ValueError("Only the frozen top-4 slot-state route may be tested")
        if self.late_head_sets != FROZEN_LATE_SETS:
            raise ValueError("Frozen full/leave-one-out L28 sets changed")
        if self.primary_late_set != "full_h16_h19":
            raise ValueError("Primary L28 set must be H16--H19")
        if self.primary_behavior_metric != "donor_log_odds_gain":
            raise ValueError("Primary endpoint must remain donor log-odds gain")
        if not 0 < self.primary_alpha < 1 or self.bootstrap_repetitions < 1_000:
            raise ValueError("Invalid inferential settings")
        if self.expand_late_if_base_insufficient:
            raise ValueError("Leave-one-out stage is mandatory, not outcome-triggered")
        if self.block_closure_relative_tolerance <= 0:
            raise ValueError("Block-closure tolerance must be positive")
        if self.control_orthogonality_tolerance <= 0:
            raise ValueError("Orthogonality tolerance must be positive")
        if self.persist_full_states or self.persist_raw_attention:
            raise ValueError("Raw states/attention must not be persisted")
        if not self.atomic_shards:
            raise ValueError("Confirmation requires atomic shards")


V444UpstreamConfirmationConfig().validate()
