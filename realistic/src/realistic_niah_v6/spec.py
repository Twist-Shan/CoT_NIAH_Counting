from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "realistic_niah_v6_config_v1"
PROTOCOL_VERSION = "realistic_niah_v6_structured_enumeration_replication_v1"
GENERATION_SCHEMA_VERSION = "realistic_niah_v6_enumeration_generation_v1"
PROMPT_MODES = ("enumeration_index", "enumeration_bullet")
MODEL_LABELS = ("Qwen3-8B", "Gemma4-E4B")
DISCOVERY_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
COUNTS = tuple(range(1, 11))
PRIMARY_TRACE_SITE = "item_end"
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
    "strict_format_correct",
)


@dataclass(frozen=True)
class DecodingSpec:
    max_new_tokens: int = 4096
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    def validate(self) -> None:
        if self.max_new_tokens < 64:
            raise ValueError("V6 enumeration max_new_tokens must be at least 64")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")


@dataclass(frozen=True)
class V6Config:
    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    design_variant: str = "v4.4"
    prompt_mode: str = "enumeration_index"
    model_labels: tuple[str, ...] = MODEL_LABELS
    counts: tuple[int, ...] = COUNTS
    discovery_seeds: tuple[int, ...] = DISCOVERY_SEEDS
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    primary_trace_site: str = PRIMARY_TRACE_SITE
    sensitivity_trace_sites: tuple[str, ...] = tuple(
        site for site in REGISTERED_TRACE_SITES if site != PRIMARY_TRACE_SITE
    )
    primary_parser_cohort: str = "strict_format_correct"
    representation_n10_only: bool = False
    hidden_save_dtype: str = "float16"
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    bootstrap_samples: int = 10_000
    candidate_counts: tuple[int, ...] = COUNTS
    causal_development_seeds: tuple[int, ...] = DISCOVERY_SEEDS
    causal_confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    causal_head_mechanisms: tuple[str, ...] = (
        "retrieval_anchor_localization",
    )
    causal_head_selection_metric: str = (
        "seed_first_equal_anchor_mean_target_source_attention_mass"
    )
    causal_primary_bank_size: int = 128
    causal_head_bank_sizes: tuple[int, ...] = (
        1, 2, 4, 6, 8, 16, 32, 64, 80, 96, 112, 125, 128
    )
    qwen_targeted_bank_grid: tuple[int, ...] = (32, 64, 80, 96, 112, 125, 128)
    gemma_targeted_bank_grid: tuple[int, ...] = (1, 2, 4, 6, 8)
    qwen_report_reference_bank_size: int = 128
    gemma_report_reference_bank_size: int = 6
    qwen_targeted_selection_metric: str = (
        "seed_first_equal_anchor_mean_target_source_attention_mass"
    )
    gemma_targeted_selection_metric: str = (
        "seed_first_equal_anchor_mean_source_attention_mass"
    )
    causal_crossfit_folds: int = 5
    causal_random_controls: int = 3
    require_exact_passage_order: bool = True
    require_strict_output_format: bool = True
    decoding: DecodingSpec = field(default_factory=DecodingSpec)

    @property
    def all_seeds(self) -> tuple[int, ...]:
        return self.discovery_seeds + self.confirmation_seeds

    @property
    def registered_sites(self) -> tuple[str, ...]:
        return (self.primary_trace_site,) + tuple(self.sensitivity_trace_sites)

    @property
    def expected_marker_kind(self) -> str:
        return "indexed" if self.prompt_mode == "enumeration_index" else "bullet"

    @property
    def mode_slug(self) -> str:
        return self.prompt_mode.removeprefix("enumeration_")

    def targeted_bank_grid(self, model_label: str) -> tuple[int, ...]:
        if model_label == "Qwen3-8B":
            return self.qwen_targeted_bank_grid
        if model_label == "Gemma4-E4B":
            return self.gemma_targeted_bank_grid
        raise ValueError(f"Unknown V6 model: {model_label}")

    def report_reference_bank_size(self, model_label: str) -> int:
        if model_label == "Qwen3-8B":
            return self.qwen_report_reference_bank_size
        if model_label == "Gemma4-E4B":
            return self.gemma_report_reference_bank_size
        raise ValueError(f"Unknown V6 model: {model_label}")

    def targeted_selection_metric(self, model_label: str) -> str:
        if model_label == "Qwen3-8B":
            return self.qwen_targeted_selection_metric
        if model_label == "Gemma4-E4B":
            return self.gemma_targeted_selection_metric
        raise ValueError(f"Unknown V6 model: {model_label}")

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported V6 config schema: {self.schema_version}")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported V6 protocol version: {self.protocol_version}"
            )
        if self.design_variant != "v4.4":
            raise ValueError("V6 is frozen to V4.4 stimuli")
        if self.prompt_mode not in PROMPT_MODES:
            raise ValueError(f"V6 prompt_mode must be one of {PROMPT_MODES}")
        if tuple(self.model_labels) != MODEL_LABELS:
            raise ValueError(f"V6 models are frozen to {MODEL_LABELS}")
        if tuple(self.counts) != COUNTS or tuple(self.candidate_counts) != COUNTS:
            raise ValueError("V6 count and candidate grids must be exactly 1..10")
        if tuple(self.discovery_seeds) != DISCOVERY_SEEDS:
            raise ValueError("V6 discovery seeds are frozen to 1234..1253")
        if tuple(self.confirmation_seeds) != CONFIRMATION_SEEDS:
            raise ValueError("V6 confirmation seeds are frozen to 1254..1263")
        if set(self.discovery_seeds) & set(self.confirmation_seeds):
            raise ValueError("V6 discovery and confirmation seeds overlap")
        if tuple(self.causal_development_seeds) != DISCOVERY_SEEDS:
            raise ValueError("V6 causal development seeds must equal discovery seeds")
        if tuple(self.causal_confirmation_seeds) != CONFIRMATION_SEEDS:
            raise ValueError(
                "V6 causal confirmation seeds must equal confirmation seeds"
            )
        if self.primary_trace_site != PRIMARY_TRACE_SITE:
            raise ValueError("V6 primary progress site is frozen to item_end")
        if len(set(self.registered_sites)) != len(self.registered_sites):
            raise ValueError("V6 registered trace sites are not unique")
        unknown = sorted(set(self.registered_sites) - set(REGISTERED_TRACE_SITES))
        if unknown:
            raise ValueError(f"Unknown V6 trace sites: {unknown}")
        if self.primary_parser_cohort not in REGISTERED_COHORTS:
            raise ValueError("Unknown V6 parser cohort")
        if self.hidden_save_dtype not in {"float16", "float32"}:
            raise ValueError("hidden_save_dtype must be float16 or float32")
        if not self.ridge_alphas or any(value <= 0 for value in self.ridge_alphas):
            raise ValueError("ridge_alphas must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.causal_head_mechanisms != ("retrieval_anchor_localization",):
            raise ValueError("V6 targeted-head mechanism contract changed")
        if self.causal_primary_bank_size < 1:
            raise ValueError("causal_primary_bank_size must be positive")
        if tuple(sorted(set(self.causal_head_bank_sizes))) != tuple(
            self.causal_head_bank_sizes
        ):
            raise ValueError("causal_head_bank_sizes must be unique and increasing")
        if tuple(self.qwen_targeted_bank_grid) != (32, 64, 80, 96, 112, 125, 128):
            raise ValueError("Qwen V6 targeted-bank grid changed from the report")
        if tuple(self.gemma_targeted_bank_grid) != (1, 2, 4, 6, 8):
            raise ValueError("Gemma V6 targeted-bank grid changed from the report")
        if self.qwen_report_reference_bank_size != 128:
            raise ValueError("Qwen report-reference bank must remain K128")
        if self.gemma_report_reference_bank_size != 6:
            raise ValueError("Gemma report-reference bank must remain K6")
        if self.qwen_targeted_selection_metric != (
            "seed_first_equal_anchor_mean_target_source_attention_mass"
        ):
            raise ValueError("Qwen targeted selection metric changed")
        if self.gemma_targeted_selection_metric != (
            "seed_first_equal_anchor_mean_source_attention_mass"
        ):
            raise ValueError("Gemma targeted selection metric changed")
        if not 2 <= self.causal_crossfit_folds <= len(self.discovery_seeds):
            raise ValueError("causal_crossfit_folds is outside the discovery panel")
        if self.causal_random_controls < 1:
            raise ValueError("V6 requires at least one random control")
        if not self.require_exact_passage_order:
            raise ValueError("V6 formal cohort requires exact passage order")
        if not self.require_strict_output_format:
            raise ValueError("V6 formal cohort requires strict output format")
        self.decoding.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "V6Config":
        payload = dict(value)
        decoding = DecodingSpec(**dict(payload.pop("decoding", {})))
        tuple_fields = {
            "model_labels",
            "counts",
            "discovery_seeds",
            "confirmation_seeds",
            "sensitivity_trace_sites",
            "ridge_alphas",
            "candidate_counts",
            "causal_development_seeds",
            "causal_confirmation_seeds",
            "causal_head_mechanisms",
            "causal_head_bank_sizes",
            "qwen_targeted_bank_grid",
            "gemma_targeted_bank_grid",
        }
        for name in tuple_fields:
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(decoding=decoding, **payload)
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "V6Config":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("V6 config must contain one JSON object")
        return cls.from_mapping(value)
