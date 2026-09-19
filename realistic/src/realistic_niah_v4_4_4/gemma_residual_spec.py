from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .gemma_cross_layer_spec import FrozenSite


SCHEMA_VERSION = "realistic_niah_v4_4_4_gemma_residual_mediation_v1"
PROTOCOL_VERSION = "realistic_niah_v4_4_4_gemma_frozen_bank_residual_path_v1"


@dataclass(frozen=True)
class GemmaResidualConfig:
    """Frozen fallback for a distributed residual-state counting path.

    This protocol is deliberately weaker than a localized OV-transporter
    claim.  It asks whether the independently frozen L29/L35 bank writes a
    count-aligned answer-query residual state that causally mediates the
    donor-count shift.  Discovery chooses one residual boundary; all effects
    are then evaluated on disjoint confirmation seeds.
    """

    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    model_label: str = "Gemma4-E4B"
    mechanism_variant: str = "k2"
    candidate_sites: tuple[FrozenSite, ...] = (
        FrozenSite(29, 4),
        FrozenSite(35, 2),
    )
    matched_control_sets: tuple[tuple[FrozenSite, ...], ...] = (
        (FrozenSite(29, 1), FrozenSite(35, 3)),
        (FrozenSite(29, 6), FrozenSite(35, 2)),
        (FrozenSite(29, 7), FrozenSite(35, 6)),
    )
    discovery_seeds: tuple[int, ...] = tuple(range(1456, 1466))
    confirmation_seeds: tuple[int, ...] = tuple(range(1466, 1486))
    counts: tuple[int, ...] = tuple(range(1, 11))
    fit_counts: tuple[int, ...] = (1, 3, 5, 7, 9)
    heldout_counts: tuple[int, ...] = (2, 4, 6, 8, 10)
    donor_pairs: tuple[tuple[int, int], ...] = ((1, 6), (3, 8), (5, 10))
    candidate_mediator_layers: tuple[int, ...] = (36, 37, 38, 39, 40)
    terminal_trace_layer: int = 41
    primary_alpha: float = 0.025
    bootstrap_repetitions: int = 10_000
    model_torch_dtype: str = "bfloat16"
    attention_prefix_backend: str = "sdpa"
    attention_cache_logit_tolerance: float = 0.5
    residual_closure_relative_tolerance: float = 0.05
    control_orthogonality_tolerance: float = 1e-4
    write_raw_attention_rows: bool = False
    write_full_hidden_states: bool = False
    atomic_shards: bool = True
    require_clean_necessity: bool = False
    matched_control_sampling_seed: int | None = None
    selection_source: str = (
        "correct-only broad-retrieval K2 ranked set and three frozen "
        "layer-matched random replicates"
    )
    selection_status: str = (
        "frozen_before_single_head_and_cross_layer_natural_ov_outcomes"
    )

    @property
    def direction_discovery_seeds(self) -> tuple[int, ...]:
        return self.discovery_seeds

    @property
    def center_seeds(self) -> tuple[int, ...]:
        return self.discovery_seeds

    @property
    def dataset_discovery_seeds(self) -> tuple[int, ...]:
        return self.discovery_seeds

    @property
    def dataset_seeds(self) -> tuple[int, ...]:
        return self.discovery_seeds + self.confirmation_seeds

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GemmaResidualConfig":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown residual-path config keys: {unknown}")
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
            "discovery_seeds",
            "confirmation_seeds",
            "counts",
            "fit_counts",
            "heldout_counts",
            "candidate_mediator_layers",
        }
        for name in integer_tuple_fields:
            if name in values:
                values[name] = tuple(int(value) for value in values[name])
        if "donor_pairs" in values:
            values["donor_pairs"] = tuple(
                tuple(int(value) for value in pair) for pair in values["donor_pairs"]
            )
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "GemmaResidualConfig":
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
            raise ValueError("Unexpected Gemma residual-path schema")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected Gemma residual-path protocol")
        if self.model_label != "Gemma4-E4B":
            raise ValueError("Residual fallback is model-locked to Gemma4-E4B")
        frozen_k2 = (
            (FrozenSite(29, 4), FrozenSite(35, 2)),
            (
                (FrozenSite(29, 1), FrozenSite(35, 3)),
                (FrozenSite(29, 6), FrozenSite(35, 2)),
                (FrozenSite(29, 7), FrozenSite(35, 6)),
            ),
        )
        frozen_k6 = (
            (
                FrozenSite(29, 4),
                FrozenSite(35, 2),
                FrozenSite(35, 7),
                FrozenSite(35, 1),
                FrozenSite(35, 3),
                FrozenSite(29, 2),
            ),
            (
                (
                    FrozenSite(29, 3),
                    FrozenSite(29, 7),
                    FrozenSite(35, 0),
                    FrozenSite(35, 4),
                    FrozenSite(35, 5),
                    FrozenSite(35, 6),
                ),
                (
                    FrozenSite(29, 0),
                    FrozenSite(29, 5),
                    FrozenSite(35, 0),
                    FrozenSite(35, 4),
                    FrozenSite(35, 5),
                    FrozenSite(35, 7),
                ),
                (
                    FrozenSite(29, 5),
                    FrozenSite(29, 6),
                    FrozenSite(35, 0),
                    FrozenSite(35, 1),
                    FrozenSite(35, 2),
                    FrozenSite(35, 4),
                ),
            ),
        )
        # Independent continuation requested after the registered K2 residual
        # path passed.  The candidate is still the pre-existing causal-v2
        # broad-aggregation K6 prefix, but the controls are now strictly
        # disjoint from the candidate.  At L35 only four non-candidate heads
        # remain, so those four necessarily recur; the L29 controls form a
        # deterministic disjoint partition of the six remaining heads.
        frozen_k6_retrieval_v2 = (
            frozen_k6[0],
            (
                (
                    FrozenSite(29, 0),
                    FrozenSite(29, 1),
                    FrozenSite(35, 0),
                    FrozenSite(35, 4),
                    FrozenSite(35, 5),
                    FrozenSite(35, 6),
                ),
                (
                    FrozenSite(29, 3),
                    FrozenSite(29, 5),
                    FrozenSite(35, 0),
                    FrozenSite(35, 4),
                    FrozenSite(35, 5),
                    FrozenSite(35, 6),
                ),
                (
                    FrozenSite(29, 6),
                    FrozenSite(29, 7),
                    FrozenSite(35, 0),
                    FrozenSite(35, 4),
                    FrozenSite(35, 5),
                    FrozenSite(35, 6),
                ),
            ),
        )
        if self.mechanism_variant == "k2":
            expected_candidate, expected_controls = frozen_k2
            expected_discovery = tuple(range(1456, 1466))
            expected_confirmation = tuple(range(1466, 1486))
            expected_layers = Counter({29: 1, 35: 1})
            if self.require_clean_necessity:
                raise ValueError(
                    "The frozen K2 protocol did not register clean necessity"
                )
            if self.matched_control_sampling_seed is not None:
                raise ValueError("The frozen K2 controls have no sampling seed field")
        elif self.mechanism_variant == "k6":
            expected_candidate, expected_controls = frozen_k6
            expected_discovery = tuple(range(1486, 1496))
            expected_confirmation = tuple(range(1496, 1516))
            expected_layers = Counter({29: 2, 35: 4})
            if not self.require_clean_necessity:
                raise ValueError("The frozen K6 protocol requires clean necessity")
            if self.matched_control_sampling_seed != 4446:
                raise ValueError("The frozen K6 controls require sampling seed 4446")
        elif self.mechanism_variant == "k6_retrieval_v2":
            expected_candidate, expected_controls = frozen_k6_retrieval_v2
            expected_discovery = tuple(range(1600, 1610))
            expected_confirmation = tuple(range(1610, 1630))
            expected_layers = Counter({29: 2, 35: 4})
            if not self.require_clean_necessity:
                raise ValueError("The K6 retrieval continuation requires clean necessity")
            if self.matched_control_sampling_seed != 4447:
                raise ValueError(
                    "The K6 retrieval continuation controls require registry seed 4447"
                )
        else:
            raise ValueError("Unknown residual mechanism variant")
        if self.candidate_sites != expected_candidate:
            raise ValueError(
                f"Residual source bank changed from the frozen {self.mechanism_variant.upper()}"
            )
        if self.matched_control_sets != expected_controls:
            raise ValueError("Residual matched controls changed from the frozen sets")
        all_sets = (self.candidate_sites, *self.matched_control_sets)
        if any(
            Counter(site.layer for site in site_set) != expected_layers
            for site_set in all_sets
        ):
            raise ValueError(
                "A residual source set changed its frozen layer composition"
            )
        discovery = set(self.discovery_seeds)
        confirmation = set(self.confirmation_seeds)
        if not discovery or not confirmation or discovery & confirmation:
            raise ValueError(
                "Residual discovery and confirmation seeds must be disjoint"
            )
        if self.discovery_seeds != expected_discovery:
            raise ValueError("Residual discovery seeds changed from the frozen split")
        if self.confirmation_seeds != expected_confirmation:
            raise ValueError(
                "Residual confirmation seeds changed from the frozen split"
            )
        if self.counts != tuple(range(1, 11)):
            raise ValueError("Residual counts must be exactly 1 through 10")
        if set(self.fit_counts) | set(self.heldout_counts) != set(self.counts):
            raise ValueError("Fit and held-out counts must cover the count grid")
        if set(self.fit_counts) & set(self.heldout_counts):
            raise ValueError("Fit and held-out counts overlap")
        if not self.donor_pairs or any(
            receiver >= donor or receiver not in self.counts or donor not in self.counts
            for receiver, donor in self.donor_pairs
        ):
            raise ValueError("Residual donor pairs are invalid")
        if self.candidate_mediator_layers != (36, 37, 38, 39, 40):
            raise ValueError("Residual discovery layer registry changed")
        if self.terminal_trace_layer != 41:
            raise ValueError("Residual terminal trace layer changed")
        if max(site.layer for site in self.candidate_sites) >= min(
            self.candidate_mediator_layers
        ):
            raise ValueError("Residual mediators must follow the source bank")
        if max(self.candidate_mediator_layers) >= self.terminal_trace_layer:
            raise ValueError("Terminal layer must follow every mediator candidate")
        if not 0 < self.primary_alpha <= 0.025:
            raise ValueError("Residual fallback alpha must lie in (0,.025]")
        if self.bootstrap_repetitions < 1_000:
            raise ValueError("Residual fallback needs at least 1,000 bootstraps")
        if self.residual_closure_relative_tolerance <= 0:
            raise ValueError("Residual closure tolerance must be positive")
        if self.control_orthogonality_tolerance <= 0:
            raise ValueError("Residual orthogonality tolerance must be positive")
        if self.write_raw_attention_rows or self.write_full_hidden_states:
            raise ValueError("Residual fallback must not persist raw tensors")
        if not self.atomic_shards:
            raise ValueError("Residual fallback requires atomic shards")


GemmaResidualConfig().validate()
