from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from realistic_niah_v4_4_4.spec import V444Config


SCHEMA_VERSION = "realistic_niah_v4_4_4_read_write_supplement_v1"
PROTOCOL_VERSION = "realistic_niah_v4_4_4_factorized_read_write_supplement_v1"


@dataclass(frozen=True)
class V444ReadWriteConfig:
    """Configuration for the additive V4.4.4 read/write supplement."""

    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    model_label: str = "Qwen3-8B"
    mediator_layer: int = 28
    heads: tuple[int, ...] = (16, 19)
    discovery_seeds: tuple[int, ...] = tuple(range(1264, 1274))
    evaluation_seeds: tuple[int, ...] = tuple(range(1274, 1294))
    counts: tuple[int, ...] = tuple(range(1, 11))
    fit_counts: tuple[int, ...] = (1, 3, 5, 7, 9)
    heldout_counts: tuple[int, ...] = (2, 4, 6, 8, 10)
    write_counts: tuple[int, ...] = (2, 5, 8)
    donor_pairs: tuple[tuple[int, int], ...] = (
        (1, 6),
        (6, 1),
        (3, 8),
        (8, 3),
        (5, 10),
        (10, 5),
    )
    downstream_layers: tuple[int, ...] = tuple(range(28, 36))
    tail_width: int = 64
    write_beta: float = 1.0
    primary_alpha: float = 0.05
    bootstrap_repetitions: int = 10_000
    closure_relative_tolerance: float = 1e-5
    edge_reconstruction_relative_tolerance: float = 0.05
    pre_o_output_equivalence_tolerance: float = 0.05
    persist_full_states: bool = False
    persist_raw_attention: bool = False
    atomic_shards: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "V444ReadWriteConfig":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown V4.4.4 read/write config keys: {unknown}")
        values = dict(payload)
        tuple_fields = {
            "heads",
            "discovery_seeds",
            "evaluation_seeds",
            "counts",
            "fit_counts",
            "heldout_counts",
            "write_counts",
            "downstream_layers",
        }
        for name in tuple_fields:
            if name in values:
                values[name] = tuple(int(item) for item in values[name])
        if "donor_pairs" in values:
            values["donor_pairs"] = tuple(
                tuple(int(item) for item in pair) for pair in values["donor_pairs"]
            )
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "V444ReadWriteConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_against_base(self, base: V444Config) -> None:
        base.validate()
        self.validate()
        if self.model_label != base.model_label:
            raise ValueError("V4.4.4 read/write/base model labels disagree")
        if self.mediator_layer != base.layer or self.heads != base.candidate_heads:
            raise ValueError("Read/write supplement must inherit the frozen V4.4.4 layer/head set")
        if self.discovery_seeds != base.center_seeds:
            raise ValueError("Read/write discovery must reuse the frozen center seeds")
        if self.evaluation_seeds != base.confirmation_seeds:
            raise ValueError("Read/write evaluation must reuse the frozen confirmation seeds")
        if self.counts != base.counts:
            raise ValueError("V4.4.4 read/write/base count registries disagree")
        if self.fit_counts != base.fit_counts or self.heldout_counts != base.heldout_counts:
            raise ValueError("V4.4.4 read/write/base fit-count partition disagrees")

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unexpected V4.4.4 read/write schema version")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected V4.4.4 read/write protocol version")
        if self.model_label != "Qwen3-8B" or self.mediator_layer != 28:
            raise ValueError("V4.4.4 read/write is frozen to Qwen3-8B L28")
        if self.heads != (16, 19):
            raise ValueError("V4.4.4 read/write is frozen to heads {16,19}")
        if not self.discovery_seeds or not self.evaluation_seeds:
            raise ValueError("Read/write seed registries must be nonempty")
        if set(self.discovery_seeds) & set(self.evaluation_seeds):
            raise ValueError("Read/write discovery and evaluation seeds overlap")
        if self.counts != tuple(range(1, 11)):
            raise ValueError("Read/write counts must be exactly 1 through 10")
        if set(self.fit_counts) & set(self.heldout_counts):
            raise ValueError("Read/write fit and held-out counts overlap")
        if set(self.fit_counts) | set(self.heldout_counts) != set(self.counts):
            raise ValueError("Read/write fit/held-out counts must partition all counts")
        if not set(self.write_counts).issubset(self.counts):
            raise ValueError("Read/write intervention counts lie outside the registry")
        if not self.donor_pairs or any(
            receiver == donor
            or receiver not in self.counts
            or donor not in self.counts
            for receiver, donor in self.donor_pairs
        ):
            raise ValueError("Read/write donor pairs must be unequal registered counts")
        if {(donor, receiver) for receiver, donor in self.donor_pairs} != set(
            self.donor_pairs
        ):
            raise ValueError("Every read/write donor pair needs its reverse direction")
        if (
            not self.downstream_layers
            or tuple(sorted(set(self.downstream_layers))) != self.downstream_layers
            or self.downstream_layers[0] != self.mediator_layer
        ):
            raise ValueError("Downstream layers must be sorted, unique, and start at L28")
        if self.tail_width <= 0:
            raise ValueError("Read/write tail width must be positive")
        if self.write_beta <= 0:
            raise ValueError("Read/write beta must be positive")
        if not 0 < self.primary_alpha < 1:
            raise ValueError("Read/write alpha must lie in (0,1)")
        if self.bootstrap_repetitions < 1_000:
            raise ValueError("Read/write bootstrap repetitions must be at least 1000")
        if self.closure_relative_tolerance <= 0:
            raise ValueError("Read/write closure tolerance must be positive")
        if self.edge_reconstruction_relative_tolerance <= 0:
            raise ValueError("Read/write edge reconstruction tolerance must be positive")
        if self.pre_o_output_equivalence_tolerance <= 0:
            raise ValueError("Read/write pre-O output tolerance must be positive")
        if self.persist_full_states or self.persist_raw_attention:
            raise ValueError("Read/write supplement must not persist full states/raw attention")
        if not self.atomic_shards:
            raise ValueError("Read/write supplement requires atomic seed shards")


V444ReadWriteConfig().validate()
