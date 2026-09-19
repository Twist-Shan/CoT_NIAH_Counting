from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "realistic_niah_v4_nonthinking_v3"
CONFIG_SCHEMA_VERSION = "realistic_niah_v4_config_v3"
STIMULUS_SCHEMA_VERSION = "realistic_niah_v4_stimulus_v1"
MANIFEST_SCHEMA_VERSION = "realistic_niah_v4_manifest_v1"
CAPTURE_SCHEMA_VERSION = "realistic_niah_v4_capture_v1"

CANONICAL_TOKENIZER = "Qwen/Qwen3-8B"
CANONICAL_TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TARGET_PASSAGE_TOKENS = 10_000
NEEDLE_COUNTS = tuple(range(1, 11))
COUNT_CANDIDATE_WORDS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)
ANSWER_FORMATS = ("numeric",)
SEEDS = tuple(range(1234, 1264))
DISCOVERY_SEEDS = SEEDS[:20]
CONFIRMATION_SEEDS = SEEDS[20:]
FIXED_SLOT_DEPTHS = (
    0.08,
    0.17,
    0.26,
    0.35,
    0.44,
    0.56,
    0.65,
    0.74,
    0.83,
    0.92,
)
FIXED_NEEDLE_SEED = 4404
DESIGN_VARIANTS = ("v4.1", "v4.2", "v4.3", "v4.4")
DESIGN_VARIANT_CONTROLS = {
    "v4.1": {
        "positions_fixed_across_seeds": True,
        "city_score_order_fixed_across_seeds": True,
        "city_score_content_fixed_across_seeds": True,
    },
    "v4.2": {
        "positions_fixed_across_seeds": False,
        "city_score_order_fixed_across_seeds": True,
        "city_score_content_fixed_across_seeds": True,
    },
    "v4.3": {
        "positions_fixed_across_seeds": False,
        "city_score_order_fixed_across_seeds": False,
        "city_score_content_fixed_across_seeds": True,
    },
    "v4.4": {
        "positions_fixed_across_seeds": False,
        "city_score_order_fixed_across_seeds": False,
        "city_score_content_fixed_across_seeds": False,
    },
}


@dataclass(frozen=True)
class V4ModelSpec:
    label: str
    model_id: str
    revision: str
    family: str
    loader_class: str
    chat_template_control: str = "enable_thinking_kwarg"


MODEL_SPECS = {
    spec.label: spec
    for spec in (
        V4ModelSpec(
            label="Qwen3-8B",
            model_id="Qwen/Qwen3-8B",
            revision=CANONICAL_TOKENIZER_REVISION,
            family="qwen3",
            loader_class="AutoModelForCausalLM",
        ),
        V4ModelSpec(
            label="Gemma4-E4B",
            model_id="google/gemma-4-E4B-it",
            revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
            family="gemma4",
            loader_class="AutoModelForMultimodalLM",
        ),
    )
}


@dataclass(frozen=True)
class V4Config:
    schema_version: str = CONFIG_SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    target_passage_tokens: int = TARGET_PASSAGE_TOKENS
    needle_counts: tuple[int, ...] = NEEDLE_COUNTS
    seeds: tuple[int, ...] = SEEDS
    discovery_seeds: tuple[int, ...] = DISCOVERY_SEEDS
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    canonical_tokenizer: str = CANONICAL_TOKENIZER
    canonical_tokenizer_revision: str = CANONICAL_TOKENIZER_REVISION
    fixed_slot_depths: tuple[float, ...] = FIXED_SLOT_DEPTHS
    fixed_needle_seed: int = FIXED_NEEDLE_SEED
    design_variants: tuple[str, ...] = DESIGN_VARIANTS
    randomized_position_min_separation_tokens: int = 256
    model_labels: tuple[str, ...] = tuple(MODEL_SPECS)
    prompt_mode: str = "direct"
    answer_prefix: str = "Total:"
    answer_formats: tuple[str, ...] = ANSWER_FORMATS
    count_candidate_words: tuple[str, ...] = COUNT_CANDIDATE_WORDS
    candidate_separator: str = ""
    candidate_score_reduction: str = "sum_log_probability"
    candidate_score_include_termination: bool = True
    hidden_state_poolings: tuple[str, ...] = ("span_end", "span_mean")
    representation_count: int = 10
    pca_components: int = 3
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    attention_primary_metric: str = "broad_primary"
    ablation_top_ns: tuple[int, ...] = (1, 2, 4, 8)
    ablation_random_replicates: int = 3
    ablation_scope: str = "answer_query"
    ablation_scopes: tuple[str, ...] = ("answer_query", "global")
    causal_layer_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    patch_count_pairs: tuple[tuple[int, int], ...] = (
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
    )
    patch_sites: tuple[str, ...] = (
        "answer_query",
        "toggled_needle_end",
        "toggled_needle_span",
    )
    residual_patch_protocols: tuple[str, ...] = (
        "single_layer",
        "cumulative_from_layer",
    )
    steering_count_pairs: tuple[tuple[int, int], ...] = (
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (1, 3),
        (3, 6),
        (5, 10),
    )
    steering_methods: tuple[str, ...] = (
        "centroid_transplant",
        "centroid_delta",
        "chord",
        "polyline",
    )
    steering_alphas: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    steering_random_replicates: int = 1
    causal_bootstrap_repetitions: int = 10_000
    require_exact_offset_mapping: bool = True
    hidden_save_dtype: str = "float16"
    model_torch_dtype: str = "bfloat16"
    attention_prefix_backend: str = "sdpa"
    save_raw_attention_rows: bool = True
    attention_save_dtype: str = "float16"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> V4Config:
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"Unknown V4 config keys: {unknown}")
        values = dict(payload)
        tuple_fields = {
            "needle_counts",
            "seeds",
            "discovery_seeds",
            "confirmation_seeds",
            "fixed_slot_depths",
            "design_variants",
            "model_labels",
            "hidden_state_poolings",
            "answer_formats",
            "count_candidate_words",
            "ridge_alphas",
            "ablation_top_ns",
            "ablation_scopes",
            "causal_layer_fractions",
            "patch_sites",
            "residual_patch_protocols",
            "steering_methods",
            "steering_alphas",
        }
        for name in tuple_fields:
            if name in values:
                values[name] = tuple(values[name])
        if "patch_count_pairs" in values:
            values["patch_count_pairs"] = tuple(
                tuple(int(item) for item in pair)
                for pair in values["patch_count_pairs"]
            )
        if "steering_count_pairs" in values:
            values["steering_count_pairs"] = tuple(
                tuple(int(item) for item in pair)
                for pair in values["steering_count_pairs"]
            )
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> V4Config:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError("Unexpected V4 config schema")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected V4 protocol")
        if self.target_passage_tokens <= 0:
            raise ValueError("target_passage_tokens must be positive")
        if self.needle_counts != tuple(range(1, len(self.needle_counts) + 1)):
            raise ValueError("V4 needle_counts must be consecutive and start at one")
        if len(self.fixed_slot_depths) != max(self.needle_counts):
            raise ValueError("V4 requires one fixed depth per maximum-count slot")
        if any(not 0.05 <= float(depth) <= 0.95 for depth in self.fixed_slot_depths):
            raise ValueError("Every fixed V4 slot must lie in the 5%-95% interval")
        if any(
            right <= left
            for left, right in zip(self.fixed_slot_depths, self.fixed_slot_depths[1:])
        ):
            raise ValueError("V4 slot depths must be strictly increasing")
        if self.design_variants != DESIGN_VARIANTS:
            raise ValueError("V4 requires the registered v4.1-v4.4 panels")
        if self.randomized_position_min_separation_tokens <= 0:
            raise ValueError(
                "randomized_position_min_separation_tokens must be positive"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("V4 seeds must be unique")
        if not self.discovery_seeds or not self.confirmation_seeds:
            raise ValueError("Both V4 seed splits must be nonempty")
        if set(self.discovery_seeds).intersection(self.confirmation_seeds):
            raise ValueError("Discovery and confirmation seeds must be disjoint")
        if set(self.discovery_seeds).union(self.confirmation_seeds) != set(self.seeds):
            raise ValueError("Discovery and confirmation seeds must partition seeds")
        if self.representation_count != max(self.needle_counts):
            raise ValueError(
                "Primary prompt-reading capture must use the maximum-count passage"
            )
        if set(self.hidden_state_poolings) != {"span_end", "span_mean"}:
            raise ValueError("V4 requires both span_end and span_mean pooling")
        if self.pca_components < 2:
            raise ValueError("pca_components must be at least two")
        if not self.ridge_alphas or any(
            float(alpha) <= 0 for alpha in self.ridge_alphas
        ):
            raise ValueError("ridge_alphas must be nonempty and positive")
        if self.prompt_mode != "direct":
            raise ValueError("Registered V4 is non-thinking direct mode only")
        if self.ablation_scope not in {"answer_query", "global"}:
            raise ValueError("ablation_scope must be answer_query or global")
        if (
            not self.ablation_scopes
            or any(
                scope not in {"answer_query", "global"}
                for scope in self.ablation_scopes
            )
            or len(set(self.ablation_scopes)) != len(self.ablation_scopes)
        ):
            raise ValueError("ablation_scopes must be unique registered scopes")
        if (
            not self.ablation_top_ns
            or any(int(value) <= 0 for value in self.ablation_top_ns)
            or tuple(sorted(set(self.ablation_top_ns))) != self.ablation_top_ns
        ):
            raise ValueError("ablation_top_ns must be positive and increasing")
        if self.ablation_random_replicates <= 0:
            raise ValueError("ablation_random_replicates must be positive")
        if any(
            not 0.0 <= float(fraction) <= 1.0
            for fraction in self.causal_layer_fractions
        ):
            raise ValueError("causal_layer_fractions must lie in [0, 1]")
        if self.patch_sites != (
            "answer_query",
            "toggled_needle_end",
            "toggled_needle_span",
        ):
            raise ValueError("Registered V4 requires all three patch sites")
        if self.residual_patch_protocols != (
            "single_layer",
            "cumulative_from_layer",
        ):
            raise ValueError("Registered V4 requires both residual patch protocols")
        registered_steering_methods = {
            "centroid_transplant",
            "centroid_delta",
            "chord",
            "polyline",
        }
        if (
            not self.steering_methods
            or set(self.steering_methods) != registered_steering_methods
            or len(set(self.steering_methods)) != len(self.steering_methods)
        ):
            raise ValueError("Registered V4 requires all geometric steering methods")
        if (
            not self.steering_alphas
            or tuple(sorted(set(self.steering_alphas))) != self.steering_alphas
            or any(not 0.0 < float(alpha) <= 1.0 for alpha in self.steering_alphas)
            or 1.0 not in self.steering_alphas
        ):
            raise ValueError(
                "steering_alphas must be unique, increasing, and include 1"
            )
        if self.steering_random_replicates < 0:
            raise ValueError("steering_random_replicates must be nonnegative")
        if self.causal_bootstrap_repetitions <= 0:
            raise ValueError("causal_bootstrap_repetitions must be positive")
        if any(label not in MODEL_SPECS for label in self.model_labels):
            raise ValueError("Unknown registered V4 model label")
        if not self.answer_prefix:
            raise ValueError("answer_prefix must be non-empty")
        if self.answer_formats != ANSWER_FORMATS:
            raise ValueError("Registered V4 currently requires the numeric arm")
        expected_words = COUNT_CANDIDATE_WORDS[: len(self.needle_counts)]
        if self.count_candidate_words[: len(self.needle_counts)] != expected_words:
            raise ValueError(
                "Registered V4 requires the lowercase one-through-ten vocabulary"
            )
        if len(set(expected_words)) != len(self.needle_counts):
            raise ValueError("V4 count candidate words must be unique")
        if self.candidate_separator:
            raise ValueError("Registered V4 candidates must immediately follow Total:")
        if self.candidate_score_reduction != "sum_log_probability":
            raise ValueError("Registered V4 requires joint sequence log-probability")
        if not self.candidate_score_include_termination:
            raise ValueError("Registered V4 sequence scores must include termination")
        if not self.require_exact_offset_mapping:
            raise ValueError("Registered V4 requires exact offset mappings")
        if self.hidden_save_dtype not in {"float16", "float32"}:
            raise ValueError("hidden_save_dtype must be float16 or float32")
        if self.model_torch_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Unsupported V4 model_torch_dtype")
        if self.attention_prefix_backend not in {"sdpa", "flash_attention_2"}:
            raise ValueError("Unsupported V4 attention_prefix_backend")
        if not isinstance(self.save_raw_attention_rows, bool):
            raise TypeError("save_raw_attention_rows must be boolean")
        if self.attention_save_dtype not in {"float16", "float32"}:
            raise ValueError("attention_save_dtype must be float16 or float32")
        for low, high in self.patch_count_pairs:
            if (
                low >= high
                or low not in self.needle_counts
                or high not in self.needle_counts
            ):
                raise ValueError(f"Invalid V4 patch pair: {(low, high)}")
        for low, high in self.steering_count_pairs:
            if (
                low >= high
                or low not in self.needle_counts
                or high not in self.needle_counts
            ):
                raise ValueError(f"Invalid V4 steering pair: {(low, high)}")


def resolve_model_spec(label_or_id: str) -> V4ModelSpec:
    if label_or_id in MODEL_SPECS:
        return MODEL_SPECS[label_or_id]
    matches = [spec for spec in MODEL_SPECS.values() if spec.model_id == label_or_id]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unknown registered V4 model: {label_or_id}")


def resolve_fractional_layers(
    num_layers: int,
    fractions: Iterable[float],
) -> tuple[int, ...]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    resolved: set[int] = set()
    for fraction in fractions:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Layer fraction is outside [0, 1]: {value}")
        resolved.add(round(value * (num_layers - 1)))
    return tuple(sorted(resolved))


V4Config().validate()
