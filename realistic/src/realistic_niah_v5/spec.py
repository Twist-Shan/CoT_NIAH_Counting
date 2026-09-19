from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "realistic_niah_v5_config_v2"
PRIMARY_TRACE_SITE = "item_end"
DISCOVERY_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
REGISTERED_TRACE_SITES = (
    "pre_marker",
    "marker_end",
    "pre_city",
    "city_end",
    "city_unit_end",
    "item_end",
    "post_boundary",
    "list_cut",
    "answer_query",
    "answer_query_v3",
)
REGISTERED_COHORTS = (
    "parser_hit",
    "one_to_one",
    "one_to_one_correct",
)
REGISTERED_CAUSAL_HEAD_MECHANISMS = (
    "retrieval_anchor_localization",
)
REGISTERED_CAUSAL_HEAD_SELECTION_METRICS = (
    "seed_first_equal_anchor_mean_target_source_attention_mass",
    "seed_first_equal_anchor_mean_target_source_relative_attention_mass",
    "seed_first_equal_anchor_mean_source_attention_mass",
    "seed_first_equal_anchor_mean_target_minus_max_wrong_source_attention_mass",
    # Retained only so historical manifests/configs remain readable.
    "seed_first_equal_anchor_mean_source_specific_ov_write_norm",
)


@dataclass(frozen=True)
class DecodingSpec:
    max_new_tokens: int = 4096
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")


@dataclass(frozen=True)
class V5Config:
    schema_version: str = SCHEMA_VERSION
    design_variant: str = "v4.4"
    prompt_mode: str = "native_thinking"
    model_labels: tuple[str, ...] = ("Qwen3-8B", "Gemma4-E4B")
    counts: tuple[int, ...] = tuple(range(1, 11))
    discovery_seeds: tuple[int, ...] = DISCOVERY_SEEDS
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    primary_trace_site: str = PRIMARY_TRACE_SITE
    sensitivity_trace_sites: tuple[str, ...] = (
        "pre_marker",
        "marker_end",
        "pre_city",
        "city_end",
        "city_unit_end",
        "post_boundary",
        "list_cut",
        "answer_query",
        "answer_query_v3",
    )
    primary_parser_cohort: str = "parser_hit"
    representation_n10_only: bool = False
    hidden_save_dtype: str = "float16"
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    bootstrap_samples: int = 10000
    causal_development_seeds: tuple[int, ...] = DISCOVERY_SEEDS
    causal_confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    causal_head_mechanisms: tuple[str, ...] = REGISTERED_CAUSAL_HEAD_MECHANISMS
    causal_head_selection_metric: str = (
        "seed_first_equal_anchor_mean_target_source_attention_mass"
    )
    causal_primary_bank_size: int = 8
    # Retained for the legacy attention-plan and answer-query workflows.
    causal_head_bank_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    causal_crossfit_folds: int = 5
    causal_random_controls: int = 3
    candidate_counts: tuple[int, ...] = tuple(range(1, 11))
    decoding: DecodingSpec = field(default_factory=DecodingSpec)

    @property
    def all_seeds(self) -> tuple[int, ...]:
        return self.discovery_seeds + self.confirmation_seeds

    @property
    def registered_sites(self) -> tuple[str, ...]:
        return (self.primary_trace_site,) + tuple(self.sensitivity_trace_sites)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported V5 config schema: {self.schema_version}")
        if self.design_variant != "v4.4":
            raise ValueError("V5 mechanism replication is fixed to V4.4 stimuli")
        if self.prompt_mode != "native_thinking":
            raise ValueError("V5 is registered only for native_thinking")
        if not self.model_labels or len(set(self.model_labels)) != len(self.model_labels):
            raise ValueError("model_labels must be non-empty and unique")
        if self.counts != tuple(range(1, 11)):
            raise ValueError("V5 counts must be exactly 1 through 10")
        if not self.discovery_seeds or not self.confirmation_seeds:
            raise ValueError("Both discovery and confirmation seeds are required")
        if set(self.discovery_seeds) & set(self.confirmation_seeds):
            raise ValueError("Discovery and confirmation seeds must be disjoint")
        if self.discovery_seeds != DISCOVERY_SEEDS:
            raise ValueError("V5 discovery seeds are frozen to 1234..1253")
        if self.confirmation_seeds != CONFIRMATION_SEEDS:
            raise ValueError("V5 confirmation seeds are frozen to 1254..1263")
        if self.primary_trace_site != PRIMARY_TRACE_SITE:
            raise ValueError(
                "The registered primary site is item_end; change requires a new schema"
            )
        sites = self.registered_sites
        if len(set(sites)) != len(sites):
            raise ValueError("Trace sites must be unique")
        unknown = sorted(set(sites) - set(REGISTERED_TRACE_SITES))
        if unknown:
            raise ValueError(f"Unknown trace sites: {unknown}")
        if self.primary_parser_cohort not in REGISTERED_COHORTS:
            raise ValueError("Unknown primary parser cohort")
        if self.hidden_save_dtype not in {"float16", "float32"}:
            raise ValueError("hidden_save_dtype must be float16 or float32")
        if not self.ridge_alphas or any(value <= 0 for value in self.ridge_alphas):
            raise ValueError("ridge_alphas must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not self.causal_development_seeds:
            raise ValueError("causal_development_seeds cannot be empty")
        if len(set(self.causal_development_seeds)) != len(
            self.causal_development_seeds
        ):
            raise ValueError("causal_development_seeds must be unique")
        if len(set(self.causal_confirmation_seeds)) != len(
            self.causal_confirmation_seeds
        ):
            raise ValueError("causal_confirmation_seeds must be unique")
        if set(self.causal_development_seeds) & set(
            self.causal_confirmation_seeds
        ):
            raise ValueError(
                "Causal development and confirmation seeds must be disjoint"
            )
        if self.causal_development_seeds != DISCOVERY_SEEDS:
            raise ValueError(
                "V5 causal development seeds must equal the 20 discovery seeds"
            )
        if self.causal_confirmation_seeds != CONFIRMATION_SEEDS:
            raise ValueError(
                "V5 causal confirmation seeds must equal the 10 confirmation seeds"
            )
        causal_seed_roles = set(self.causal_development_seeds) | set(
            self.causal_confirmation_seeds
        )
        if set(self.all_seeds) != causal_seed_roles:
            raise ValueError(
                "Dataset seeds must be partitioned exactly into causal "
                "development and causal confirmation roles"
            )
        if self.causal_head_mechanisms != REGISTERED_CAUSAL_HEAD_MECHANISMS:
            raise ValueError(
                "V5 causal head smoke is retrieval_anchor_localization"
            )
        if self.causal_head_selection_metric not in (
            REGISTERED_CAUSAL_HEAD_SELECTION_METRICS
        ):
            raise ValueError(
                "Unknown V5 causal head selection metric"
            )
        if self.causal_primary_bank_size < 1:
            raise ValueError("causal_primary_bank_size must be positive")
        if tuple(sorted(set(self.causal_head_bank_sizes))) != (
            self.causal_head_bank_sizes
        ):
            raise ValueError("causal_head_bank_sizes must be unique and increasing")
        if not self.causal_head_bank_sizes or min(self.causal_head_bank_sizes) < 1:
            raise ValueError("causal_head_bank_sizes must be positive")
        if not 2 <= self.causal_crossfit_folds <= len(
            self.causal_development_seeds
        ):
            raise ValueError(
                "causal_crossfit_folds must be between 2 and development n"
            )
        if self.causal_random_controls < 1:
            raise ValueError("At least one random causal control is required")
        if len(set(self.candidate_counts)) != len(self.candidate_counts):
            raise ValueError("candidate_counts must be unique")
        if self.candidate_counts != tuple(range(1, 11)):
            raise ValueError("V5 causal candidate counts must be exactly 1 through 10")
        self.decoding.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "V5Config":
        payload = dict(value)
        decoding = DecodingSpec(**dict(payload.pop("decoding", {})))
        tuple_fields = {
            "model_labels",
            "counts",
            "discovery_seeds",
            "confirmation_seeds",
            "causal_development_seeds",
            "causal_confirmation_seeds",
            "sensitivity_trace_sites",
            "ridge_alphas",
            "causal_head_mechanisms",
            "causal_head_bank_sizes",
            "candidate_counts",
        }
        for name in tuple_fields:
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(decoding=decoding, **payload)
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "V5Config":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
