"""Native-thinking count-stream versus answer-time retrieval experiments.

The module deliberately separates four scientific objects:

1. count-related states that are written at parser-observed ``item_end:k`` sites;
2. donor-to-receiver patching between intermediate trace occurrences;
3. persistence of an intervention on those states to later trace/answer sites; and
4. independently ranked prompt- and trace-broad head banks at answer time.

Within each comparison, every condition uses the same frozen token prefix.
Answer-time trials branch at the final query; trace-patch trials branch at a
registered intermediate ``item_end`` and separately retain a full clean suffix
for downstream readouts.  No intervention silently retokenizes its receiver.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah_v4.counter_channel_interventions import (
    norm_matched_orthogonal_delta,
)
from realistic_niah_v4.layerwise_removal import (
    _closest_realized_norm_replacement,
    _realized_replacement,
)
from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _extract_shared_kv_states,
    _replace_output_tensor,
    _tensor_from_output,
    capture_post_block_states,
    position_attention_outputs,
)
from realistic_niah_v4_4_3.interventions import (
    _score_candidate_sequences,
    candidate_sequence_metrics,
    clone_prefill_output_for_scoring,
)
from realistic_niah_v4_4_5.restoration import (
    generate_answer_completion_from_prefill,
)

from .causal import (
    _first_generated_city_record,
    bootstrap_seed_mean_ci,
    completion_metrics,
    fit_centroid_subspace,
    holm_adjust,
    mechanism_continuations,
    sign_flip_pvalue,
)
from .causal_sites import build_output_token_map
from .encoding import (
    NativeTraceEncoding,
    build_native_causal_encoding,
    build_native_trace_encoding,
)
from .parsing import (
    align_trace_sites,
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    output_token_ids,
    raw_output_text,
    trace_char_sites,
)


COUNT_STREAM_SCHEMA_VERSION = "realistic_niah_v5_count_stream_v1"

# Every native-thinking/count-stream experiment uses one canonical seed
# contract.  Discovery-only subphases may partition the first 20 seeds, but
# they may never borrow from the 10 confirmation seeds or introduce a second
# confirmation registry.
COUNT_STREAM_DISCOVERY_SEEDS = tuple(range(1234, 1254))
COUNT_STREAM_CONFIRMATION_SEEDS = tuple(range(1254, 1264))
COUNT_STREAM_BROAD_RANKING_SEEDS = tuple(range(1234, 1244))
COUNT_STREAM_BROAD_K_SELECTION_SEEDS = tuple(range(1244, 1254))

REGISTERED_SOURCE_GROUPS = (
    "prompt_records",
    "trace_context",
    "trace_items",
    "trace_other",
    "trace_markers",
    "trace_nonmarkers",
    "earlier_trace_items",
    "terminal_trace_item",
)

REGISTERED_MASK_CONDITIONS = (
    "clean",
    "block_trace_context",
    "block_trace_items",
    "block_trace_other",
    "block_prompt_records",
    "block_trace_and_prompt",
    "block_terminal_trace",
    "block_earlier_trace",
    "block_trace_markers",
    "block_trace_nonmarkers",
    "block_trace_context_matched_control",
    "block_trace_items_matched_control",
    "block_prompt_records_matched_control",
    "block_trace_markers_matched_control",
)

REGISTERED_STREAM_CONDITIONS = (
    "clean",
    "aligned_running_state_removal",
    "norm_matched_orthogonal_removal",
)

REGISTERED_TRACE_PATCH_CONDITIONS = (
    "clean",
    "self_patch",
    "full_donor_patch",
    "progress_projected_patch",
    "norm_matched_orthogonal_patch",
)

REGISTERED_TRACE_FULL_STATE_CONDITIONS = (
    "clean",
    "self_patch",
    "full_donor_patch",
)

REGISTERED_TRACE_PATCH_GEOMETRIES = (
    "endpoint",
    "suffix4",
    "suffix8",
    "full_span",
)

# Capped suffixes are opt-in geometries for the prospective query-mediation
# panel.  They must not expand the defaults of older full-state protocols.
SUPPORTED_TRACE_PATCH_GEOMETRIES = REGISTERED_TRACE_PATCH_GEOMETRIES + (
    "suffix_cap4",
    "suffix_cap8",
)

REGISTERED_TRACE_PATCH_LAYER_MODES = (
    "one_shot",
    "cumulative_clamp",
)

REGISTERED_HTML_ALIGNED_SPAN_CONDITIONS = (
    "clean",
    "uninformative",
    "clean_target_ablation",
    "uninformative_target_restore",
)

REGISTERED_RELAY_RESET_CONDITIONS = (
    "natural_relay",
    "answer_query_clean_reset",
    "post_terminal_suffix_clean_reset",
)

REGISTERED_COMPLEMENTARY_RELAY_CONDITIONS = (
    "natural_relay",
    "post_terminal_suffix_clean_reset",
)

REGISTERED_COMPLEMENTARY_SOURCE_CONDITIONS = (
    "clean",
    "block_trace_items",
    "block_trace_items_matched_control",
)

REGISTERED_HEAD_READOUT_CONDITIONS = (
    "clean",
    "selected_bank",
    "layer_matched_random",
)

# Frozen development-only terminal-receiver panel requested after the first
# endpoint patch experiment.  The count minima deliberately match the agreed
# panel rather than merely using the first mathematically possible count.
TERMINAL_LAST_COUNT_RANGES: dict[int, tuple[int, ...]] = {
    -1: tuple(range(2, 11)),
    -3: tuple(range(5, 11)),
    -5: tuple(range(7, 11)),
}

REGISTERED_RESTORATION_CONDITIONS = (
    "clean",
    "trace_token_corrupt",
    "ordinary_token_corrupt",
    "trace_corrupt_full_span_restore",
    "trace_corrupt_endpoint_restore",
    "trace_corrupt_marker_restore",
    "trace_corrupt_ordinary_state_patch",
    "ordinary_corrupt_ordinary_state_restore",
)

REGISTERED_HIGHER_IS_BETTER_OUTCOMES = (
    "correct_count_margin",
    "correct_count_probability",
    "correct_count_log_score",
    "expected_count_utility",
    "exact_count",
    "strict_count_utility",
    "downstream_item_progress_subspace_retention_score",
    "donor_vs_receiver_path_log_odds",
    "donor_vs_receiver_city_log_odds",
    "donor_vs_receiver_mean_city_log_probability",
    "donor_city_adoption",
    "downstream_progress_transport_magnitude",
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_spans(
    spans: Iterable[tuple[int, int]], *, upper: int
) -> tuple[tuple[int, int], ...]:
    values = sorted((int(start), int(end)) for start, end in spans)
    if any(not 0 <= start < end <= int(upper) for start, end in values):
        raise ValueError(f"Token span is outside [0, {upper})")
    occupied: set[int] = set()
    for start, end in values:
        positions = set(range(start, end))
        if occupied.intersection(positions):
            raise ValueError("Registered spans overlap within one source group")
        occupied.update(positions)
    return tuple(values)


def _span_positions(spans: Iterable[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(
        position for start, end in spans for position in range(int(start), int(end))
    )


def _positions_to_spans(positions: Iterable[int]) -> tuple[tuple[int, int], ...]:
    values = sorted({int(value) for value in positions})
    if not values:
        return ()
    result: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        result.append((start, previous + 1))
        start = previous = value
    result.append((start, previous + 1))
    return tuple(result)


@dataclass(frozen=True)
class NativeCountMechanismSpec:
    """Frozen design controls that are independent of the base V5 config."""

    experiment_id: str = "realistic_niah_v5_native_count_stream_v1"
    status: str = "development_only"
    answer_site_id: str = "answer_query_v3"
    running_site_kind: str = "item_end"
    development_seeds: tuple[int, ...] = COUNT_STREAM_DISCOVERY_SEEDS
    broad_ranking_seeds: tuple[int, ...] = COUNT_STREAM_BROAD_RANKING_SEEDS
    broad_k_selection_seeds: tuple[int, ...] = (
        COUNT_STREAM_BROAD_K_SELECTION_SEEDS
    )
    confirmation_seeds: tuple[int, ...] = COUNT_STREAM_CONFIRMATION_SEEDS
    candidate_counts: tuple[int, ...] = tuple(range(1, 11))
    broad_selection_metric: str = "trace_items_broad_score"
    development_bank_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    boundary_extension_bank_size: int = 64
    broad_k_selection_rule: str = "smallest_within_one_se_of_max_positive"
    broad_panel_counts_per_seed: int = 5
    broad_count_assignment: str = "odd_even_balanced"
    random_controls: int = 3
    random_control_overlap_policy: str = "nonthinking_allow_treatment_overlap"
    bootstrap_samples: int = 10_000
    trace_patch_donor_offsets: tuple[int, ...] = (-5, -3, -1, 1, 3, 5)
    trace_patch_seeds_per_cell: int = 10
    trace_patch_include_count2_terminal_panel: bool = True
    trace_patch_sampling_seed: int = 20260820
    trace_patch_conditions: tuple[str, ...] = REGISTERED_TRACE_PATCH_CONDITIONS
    trace_patch_primary_outcome: str = "donor_vs_receiver_city_log_odds"
    trace_patch_primary_direction: str = "past_to_later_receiver"

    @property
    def formal_inference_eligible(self) -> bool:
        return bool(
            self.status == "frozen_confirmation"
            and self.confirmation_seeds
            and not (set(self.development_seeds) & set(self.confirmation_seeds))
        )

    def validate(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id cannot be empty")
        if self.status not in {"development_only", "frozen_confirmation"}:
            raise ValueError("status must be development_only or frozen_confirmation")
        if self.running_site_kind != "item_end":
            raise ValueError("The primary running site is frozen to item_end")
        if self.answer_site_id not in {"answer_query", "answer_query_v3"}:
            raise ValueError("Unsupported final answer-query site")
        if not self.development_seeds:
            raise ValueError("development_seeds cannot be empty")
        if len(set(self.development_seeds)) != len(self.development_seeds):
            raise ValueError("development_seeds must be unique")
        if len(set(self.confirmation_seeds)) != len(self.confirmation_seeds):
            raise ValueError("confirmation_seeds must be unique")
        if set(self.development_seeds) & set(self.confirmation_seeds):
            raise ValueError("Development and confirmation seeds must be disjoint")
        if tuple(self.development_seeds) != COUNT_STREAM_DISCOVERY_SEEDS:
            raise ValueError(
                "Native count-stream discovery_seeds are frozen to 1234..1253 "
                "(20 seeds)"
            )
        if tuple(self.confirmation_seeds) != COUNT_STREAM_CONFIRMATION_SEEDS:
            raise ValueError(
                "Native count-stream confirmation_seeds are frozen to "
                "1254..1263 (10 seeds)"
            )
        ranking = set(int(value) for value in self.broad_ranking_seeds)
        k_selection = set(int(value) for value in self.broad_k_selection_seeds)
        development = set(int(value) for value in self.development_seeds)
        if not ranking or not k_selection:
            raise ValueError("Broad ranking and K-selection seeds cannot be empty")
        if ranking & k_selection:
            raise ValueError("Broad ranking and K-selection seeds must be disjoint")
        if tuple(self.broad_ranking_seeds) != COUNT_STREAM_BROAD_RANKING_SEEDS:
            raise ValueError("Broad-head ranking seeds are frozen to 1234..1243")
        if (
            tuple(self.broad_k_selection_seeds)
            != COUNT_STREAM_BROAD_K_SELECTION_SEEDS
        ):
            raise ValueError("Broad-head K-selection seeds are frozen to 1244..1253")
        if ranking | k_selection != development:
            raise ValueError(
                "Broad ranking and K-selection must partition all 20 discovery seeds"
            )
        if self.status == "frozen_confirmation" and not self.confirmation_seeds:
            raise ValueError("A frozen confirmation design needs fresh seeds")
        if self.candidate_counts != tuple(range(1, 11)):
            raise ValueError("Candidate count scoring is frozen to counts 1..10")
        allowed_metrics = {
            f"{source}_broad_score" for source in REGISTERED_SOURCE_GROUPS
        }
        if self.broad_selection_metric not in allowed_metrics:
            raise ValueError(
                "broad_selection_metric must name one registered source broad score"
            )
        if (
            not self.development_bank_sizes
            or tuple(sorted(set(self.development_bank_sizes)))
            != self.development_bank_sizes
            or min(self.development_bank_sizes) < 1
        ):
            raise ValueError("development_bank_sizes must be positive and increasing")
        if int(self.boundary_extension_bank_size) <= max(self.development_bank_sizes):
            raise ValueError("boundary_extension_bank_size must exceed the frozen grid")
        if self.broad_k_selection_rule != "smallest_within_one_se_of_max_positive":
            raise ValueError("Unsupported broad K-selection rule")
        if not 1 <= int(self.broad_panel_counts_per_seed) <= len(
            self.candidate_counts
        ):
            raise ValueError("broad_panel_counts_per_seed is outside the count grid")
        if self.broad_count_assignment != "odd_even_balanced":
            raise ValueError("Unsupported broad count-assignment rule")
        if self.random_controls < 1:
            raise ValueError("At least one random head-bank control is required")
        if self.random_control_overlap_policy not in {
            "nonthinking_allow_treatment_overlap",
            "exclude_treatment_heads",
        }:
            raise ValueError("Unsupported random-control overlap policy")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if (
            not self.trace_patch_donor_offsets
            or len(set(self.trace_patch_donor_offsets))
            != len(self.trace_patch_donor_offsets)
            or any(int(value) == 0 for value in self.trace_patch_donor_offsets)
        ):
            raise ValueError(
                "trace_patch_donor_offsets must be unique, nonzero offsets"
            )
        if tuple(sorted(self.trace_patch_donor_offsets)) != tuple(
            self.trace_patch_donor_offsets
        ):
            raise ValueError("trace_patch_donor_offsets must be increasing")
        if int(self.trace_patch_seeds_per_cell) < 1:
            raise ValueError("trace_patch_seeds_per_cell must be positive")
        if tuple(self.trace_patch_conditions) != REGISTERED_TRACE_PATCH_CONDITIONS:
            raise ValueError(
                "trace_patch_conditions must retain every registered control arm"
            )
        if self.trace_patch_primary_outcome != "donor_vs_receiver_city_log_odds":
            raise ValueError(
                "The trace patch primary outcome is frozen to city log odds"
            )
        if self.trace_patch_primary_direction != "past_to_later_receiver":
            raise ValueError(
                "The trace patch primary direction is frozen to past-to-later"
            )

    def seed_role(self, seed: int) -> str | None:
        value = int(seed)
        if value in set(self.development_seeds):
            return "development"
        if value in set(self.confirmation_seeds):
            return "confirmation"
        return None

    def broad_phase(self, seed: int) -> str | None:
        """Return the non-overlapping broad-ablation role for one seed."""

        value = int(seed)
        if value in set(self.broad_ranking_seeds):
            return "ranking_discovery"
        if value in set(self.broad_k_selection_seeds):
            return "k_selection_discovery"
        if value in set(self.confirmation_seeds):
            return "confirmation"
        return None

    def broad_counts_for_seed(self, seed: int, *, phase: str) -> tuple[int, ...]:
        """Freeze the five-count odd/even panel while covering counts 1..10."""

        value = int(seed)
        if phase == "ranking_discovery":
            if value not in set(self.broad_ranking_seeds):
                return ()
            return self.candidate_counts
        if phase == "k_selection_discovery":
            seeds = tuple(sorted(int(item) for item in self.broad_k_selection_seeds))
        elif phase == "confirmation":
            seeds = tuple(sorted(int(item) for item in self.confirmation_seeds))
        else:
            raise ValueError(f"Unknown broad phase: {phase}")
        if value not in set(seeds):
            return ()
        if int(self.broad_panel_counts_per_seed) != 5:
            raise ValueError("odd_even_balanced currently requires five counts/seed")
        # Alternate parity by sorted seed so count parity is not confounded
        # with an early/late contiguous seed block.  With 10 K-selection or
        # confirmation seeds, every count is represented by exactly 5 seeds.
        parity = 1 if seeds.index(value) % 2 == 0 else 0
        return tuple(count for count in self.candidate_counts if count % 2 == parity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COUNT_STREAM_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "answer_site_id": self.answer_site_id,
            "running_site_kind": self.running_site_kind,
            "development_seeds": list(self.development_seeds),
            "broad_ranking_seeds": list(self.broad_ranking_seeds),
            "broad_k_selection_seeds": list(self.broad_k_selection_seeds),
            "confirmation_seeds": list(self.confirmation_seeds),
            "candidate_counts": list(self.candidate_counts),
            "broad_selection_metric": self.broad_selection_metric,
            "development_bank_sizes": list(self.development_bank_sizes),
            "boundary_extension_bank_size": self.boundary_extension_bank_size,
            "broad_k_selection_rule": self.broad_k_selection_rule,
            "broad_panel_counts_per_seed": self.broad_panel_counts_per_seed,
            "broad_count_assignment": self.broad_count_assignment,
            "random_controls": self.random_controls,
            "random_control_overlap_policy": self.random_control_overlap_policy,
            "bootstrap_samples": self.bootstrap_samples,
            "trace_patch_donor_offsets": list(self.trace_patch_donor_offsets),
            "trace_patch_seeds_per_cell": self.trace_patch_seeds_per_cell,
            "trace_patch_include_count2_terminal_panel": (
                self.trace_patch_include_count2_terminal_panel
            ),
            "trace_patch_sampling_seed": self.trace_patch_sampling_seed,
            "trace_patch_conditions": list(self.trace_patch_conditions),
            "trace_patch_primary_outcome": self.trace_patch_primary_outcome,
            "trace_patch_primary_direction": self.trace_patch_primary_direction,
            "formal_inference_eligible": self.formal_inference_eligible,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NativeCountMechanismSpec":
        schema = str(value.get("schema_version", COUNT_STREAM_SCHEMA_VERSION))
        if schema != COUNT_STREAM_SCHEMA_VERSION:
            raise ValueError(f"Unsupported count-stream schema: {schema}")
        tuple_fields = {
            "development_seeds",
            "broad_ranking_seeds",
            "broad_k_selection_seeds",
            "confirmation_seeds",
            "candidate_counts",
            "development_bank_sizes",
            "trace_patch_donor_offsets",
            "trace_patch_conditions",
        }
        kwargs = {
            key: (tuple(raw) if key in tuple_fields else raw)
            for key, raw in value.items()
            if key not in {"schema_version", "formal_inference_eligible"}
        }
        result = cls(**kwargs)
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "NativeCountMechanismSpec":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("Count-stream config must contain one JSON object")
        return cls.from_mapping(value)


@dataclass(frozen=True)
class AnswerSourceRegistry:
    """Exact disjoint token partitions used at the final answer query."""

    request_id: str
    answer_site_id: str
    sequence_length: int
    prompt_token_count: int
    query_position: int
    prompt_records: tuple[tuple[int, int], ...]
    trace_context: tuple[tuple[int, int], ...]
    trace_items: tuple[tuple[int, int], ...]
    trace_other: tuple[tuple[int, int], ...]
    trace_markers: tuple[tuple[int, int], ...]
    trace_nonmarkers: tuple[tuple[int, int], ...]
    earlier_trace_items: tuple[tuple[int, int], ...]
    terminal_trace_item: tuple[tuple[int, int], ...]

    def validate(self) -> None:
        upper = int(self.sequence_length)
        if self.query_position != upper - 1:
            raise ValueError("Answer query must be the final teacher-forced token")
        if not 0 < self.prompt_token_count <= self.query_position:
            raise ValueError("Invalid prompt/query boundary")
        groups = {
            name: _normalize_spans(getattr(self, name), upper=upper)
            for name in REGISTERED_SOURCE_GROUPS
        }
        if not groups["prompt_records"]:
            raise ValueError("No registered active prompt-record spans")
        if not groups["trace_items"]:
            raise ValueError("No parser-observed trace-item spans")
        prompt_positions = set(_span_positions(groups["prompt_records"]))
        context_positions = set(_span_positions(groups["trace_context"]))
        trace_positions = set(_span_positions(groups["trace_items"]))
        other_positions = set(_span_positions(groups["trace_other"]))
        marker_positions = set(_span_positions(groups["trace_markers"]))
        nonmarker_positions = set(_span_positions(groups["trace_nonmarkers"]))
        earlier_positions = set(_span_positions(groups["earlier_trace_items"]))
        terminal_positions = set(_span_positions(groups["terminal_trace_item"]))
        if any(position >= self.prompt_token_count for position in prompt_positions):
            raise ValueError("Prompt-record span crosses the prompt boundary")
        expected_context = set(range(self.prompt_token_count, self.query_position))
        if context_positions != expected_context:
            raise ValueError(
                "Trace context must cover every generated token before the answer query"
            )
        if any(
            position < self.prompt_token_count or position >= self.query_position
            for position in trace_positions
        ):
            raise ValueError("Trace-item span crosses prompt/query boundaries")
        if prompt_positions & context_positions:
            raise ValueError("Prompt and trace sources overlap")
        if not trace_positions <= context_positions:
            raise ValueError("Trace items must be a subset of the trace context")
        if trace_positions & other_positions:
            raise ValueError("Trace item and trace-other partitions overlap")
        if trace_positions | other_positions != context_positions:
            raise ValueError("Trace item/other partitions do not cover trace context")
        if not marker_positions <= trace_positions:
            raise ValueError("Trace markers must be a subset of trace items")
        if marker_positions & nonmarker_positions:
            raise ValueError("Marker and non-marker trace partitions overlap")
        if marker_positions | nonmarker_positions != trace_positions:
            raise ValueError("Marker/non-marker partitions do not cover trace items")
        if earlier_positions & terminal_positions:
            raise ValueError("Earlier and terminal trace partitions overlap")
        if earlier_positions | terminal_positions != trace_positions:
            raise ValueError("Earlier/terminal partitions do not cover trace items")
        if len(groups["terminal_trace_item"]) != 1:
            raise ValueError("Exactly one terminal trace item is required")

    def spans(self, source_group: str) -> tuple[tuple[int, int], ...]:
        if source_group not in REGISTERED_SOURCE_GROUPS:
            raise KeyError(f"Unknown answer source group: {source_group}")
        return tuple(getattr(self, source_group))

    def positions(self, source_group: str) -> tuple[int, ...]:
        return _span_positions(self.spans(source_group))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": COUNT_STREAM_SCHEMA_VERSION,
            "request_id": self.request_id,
            "answer_site_id": self.answer_site_id,
            "sequence_length": self.sequence_length,
            "prompt_token_count": self.prompt_token_count,
            "query_position": self.query_position,
        }
        for name in REGISTERED_SOURCE_GROUPS:
            spans = [list(value) for value in self.spans(name)]
            payload[name] = spans
            payload[f"{name}_token_count"] = len(self.positions(name))
        payload["registry_sha256"] = _sha256_json(payload)
        return payload


def _offset_mapped_trace_item_spans(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    prompt_token_count: int,
    query_position: int,
    parser: Any,
) -> tuple[tuple[int, int], ...]:
    """Map parser-observed item characters to frozen output-token intervals.

    ``literal_token_start``/``literal_token_end`` are intentionally strict:
    either value is absent when a parser boundary falls inside a tokenizer
    token.  That does not make the semantic item unobservable.  The causal
    compiler's audited offset map instead selects the smallest frozen-token
    interval overlapping each item, while the checks below prohibit guessing
    across items or beyond the final answer query.
    """

    token_map = build_output_token_map(row, tokenizer)
    item_sites = sorted(
        (
            site
            for site in trace_char_sites(raw_output_text(row), parser)
            if site.site_kind == "item_end"
        ),
        key=lambda site: int(site.occurrence or 0),
    )
    expected_count = int(parser.item_count)
    if len(item_sites) != expected_count:
        raise ValueError(
            "Parser item-site count differs from the selected trace count: "
            f"sites={len(item_sites)} expected={expected_count}"
        )
    spans: list[tuple[int, int]] = []
    for expected_occurrence, site in enumerate(item_sites, start=1):
        if int(site.occurrence or 0) != expected_occurrence:
            raise ValueError("Trace item occurrences are not the ordered sequence 1..M")
        mapped = token_map.span(
            f"answer_source_trace_item:{expected_occurrence}",
            int(site.char_start),
            int(site.char_end),
        )
        if mapped.get("status") != "ok":
            raise ValueError(
                "Cannot map parser-observed trace item to frozen output tokens: "
                f"occurrence={expected_occurrence} status={mapped.get('status')}"
            )
        start = int(prompt_token_count) + int(mapped["output_token_start"])
        end = int(prompt_token_count) + int(mapped["output_token_end"])
        if not int(prompt_token_count) <= start < end <= int(query_position):
            raise ValueError(
                "Mapped trace item crosses the prompt or answer-query boundary: "
                f"occurrence={expected_occurrence} span=[{start}, {end})"
            )
        if spans and start < spans[-1][1]:
            raise ValueError(
                "Mapped trace-item token intervals overlap; no implicit token "
                f"assignment is permitted: previous={spans[-1]} current={(start, end)}"
            )
        spans.append((start, end))
    return tuple(spans)


def build_answer_source_registry(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    answer_site_id: str = "answer_query_v3",
    candidate_counts: Sequence[int] = tuple(range(1, 11)),
) -> tuple[NativeTraceEncoding, AnswerSourceRegistry]:
    """Compile exact prompt/trace/marker partitions for one answer prefix."""

    encoding = build_native_trace_encoding(
        row,
        tokenizer,
        site_id=answer_site_id,
        candidate_counts=tuple(int(value) for value in candidate_counts),
    )
    family = infer_model_family(row)
    raw = raw_output_text(row)
    parser = find_trace_count_sequence(
        raw, model_family=family, gold_records=gold_records(row)
    )
    token_sites = align_trace_sites(
        tokenizer,
        raw_text=raw,
        baseline_output_token_ids=output_token_ids(row),
        sites=trace_char_sites(raw, parser),
    )
    selected_shared_prefix = int(
        encoding.selected_site.get("shared_baseline_prefix_tokens", 0)
    )
    prompt_count = int(encoding.prompt_token_count)
    item_spans = tuple(
        (int(span.start), int(span.end))
        for span in sorted(
            encoding.trace_item_spans, key=lambda value: value.slot_index
        )
    )
    if len(item_spans) != int(parser.item_count):
        item_spans = _offset_mapped_trace_item_spans(
            row,
            tokenizer,
            prompt_token_count=prompt_count,
            query_position=encoding.query_position,
            parser=parser,
        )
    item_positions = set(_span_positions(item_spans))
    trace_context_positions = set(range(prompt_count, encoding.query_position))
    marker_positions: set[int] = set()
    for site in token_sites:
        if site.char_site.site_kind != "marker_end" or not site.alignment_eligible:
            continue
        start = site.literal_token_start
        end = site.literal_token_end
        if start is None or end is None or not 0 <= start < end:
            continue
        if int(end) > selected_shared_prefix:
            continue
        full_positions = set(range(prompt_count + int(start), prompt_count + int(end)))
        marker_positions.update(full_positions & item_positions)
    nonmarker_positions = item_positions - marker_positions
    terminal = (item_spans[-1],) if item_spans else ()
    earlier = item_spans[:-1]
    registry = AnswerSourceRegistry(
        request_id=encoding.request_id,
        answer_site_id=answer_site_id,
        sequence_length=encoding.sequence_length,
        prompt_token_count=prompt_count,
        query_position=encoding.query_position,
        prompt_records=tuple(
            (int(span.start), int(span.end)) for span in encoding.prompt_record_spans
        ),
        trace_context=((prompt_count, encoding.query_position),),
        trace_items=item_spans,
        trace_other=_positions_to_spans(trace_context_positions - item_positions),
        trace_markers=_positions_to_spans(marker_positions),
        trace_nonmarkers=_positions_to_spans(nonmarker_positions),
        earlier_trace_items=earlier,
        terminal_trace_item=terminal,
    )
    registry.validate()
    return encoding, registry


def _depth_matched_positions(
    targets: Sequence[int], candidates: Sequence[int]
) -> tuple[int, ...]:
    """Choose a deterministic one-to-one position/depth-matched control."""

    target_values = tuple(sorted({int(value) for value in targets}))
    available = set(int(value) for value in candidates)
    if len(available) < len(target_values):
        raise RuntimeError(
            "Not enough non-source tokens for an equal-budget matched control"
        )
    chosen: list[int] = []
    for target in target_values:
        value = min(
            available, key=lambda candidate: (abs(candidate - target), candidate)
        )
        chosen.append(value)
        available.remove(value)
    return tuple(sorted(chosen))


def _matched_control_positions(
    registry: AnswerSourceRegistry, source_group: str
) -> tuple[int, ...]:
    targets = registry.positions(source_group)
    if source_group == "prompt_records":
        forbidden = set(registry.positions("prompt_records"))
        candidates = [
            position
            for position in range(1, registry.prompt_token_count)
            if position not in forbidden
        ]
    elif source_group == "trace_markers":
        candidates = list(registry.positions("trace_nonmarkers"))
        if len(candidates) < len(targets):
            forbidden = set(registry.positions("prompt_records"))
            candidates.extend(
                position
                for position in range(1, registry.prompt_token_count)
                if position not in forbidden
            )
    elif source_group == "trace_items":
        candidates = list(registry.positions("trace_other"))
        if len(candidates) < len(targets):
            forbidden = set(registry.positions("prompt_records"))
            candidates.extend(
                position
                for position in range(1, registry.prompt_token_count)
                if position not in forbidden
            )
    else:
        # The entire trace-context mask has no same-region non-source keys.
        # Match it to ordinary prompt tokens while excluding active records.
        forbidden = set(registry.positions("prompt_records"))
        candidates = [
            position
            for position in range(1, registry.prompt_token_count)
            if position not in forbidden
        ]
    return _depth_matched_positions(targets, candidates)


def answer_source_mask(
    registry: AnswerSourceRegistry,
    *,
    condition: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return an exact key mask for one registered answer-source arm.

    The mask is applied to the final query and retained for every scored or
    greedily decoded answer token.  Earlier trace construction is always clean.
    """

    if condition not in REGISTERED_MASK_CONDITIONS:
        raise ValueError(f"Unknown answer-source condition: {condition}")
    blocked: set[int] = set()
    source_groups: list[str] = []
    matched_for: str | None = None
    if condition == "block_trace_context":
        source_groups = ["trace_context"]
    elif condition == "block_trace_items":
        source_groups = ["trace_items"]
    elif condition == "block_trace_other":
        source_groups = ["trace_other"]
    elif condition == "block_prompt_records":
        source_groups = ["prompt_records"]
    elif condition == "block_trace_and_prompt":
        source_groups = ["trace_context", "prompt_records"]
    elif condition == "block_terminal_trace":
        source_groups = ["terminal_trace_item"]
    elif condition == "block_earlier_trace":
        source_groups = ["earlier_trace_items"]
    elif condition == "block_trace_markers":
        source_groups = ["trace_markers"]
    elif condition == "block_trace_nonmarkers":
        source_groups = ["trace_nonmarkers"]
    elif condition == "block_trace_context_matched_control":
        matched_for = "trace_context"
    elif condition == "block_trace_items_matched_control":
        matched_for = "trace_items"
    elif condition == "block_prompt_records_matched_control":
        matched_for = "prompt_records"
    elif condition == "block_trace_markers_matched_control":
        matched_for = "trace_markers"
    for group in source_groups:
        positions = registry.positions(group)
        if not positions:
            raise ValueError(
                f"Condition is not applicable: source group {group} is empty"
            )
        blocked.update(positions)
    if matched_for is not None:
        targets = registry.positions(matched_for)
        if not targets:
            raise ValueError(
                f"Matched control is not applicable: {matched_for} is empty"
            )
        blocked.update(_matched_control_positions(registry, matched_for))
    mask = torch.ones((1, registry.sequence_length), dtype=torch.long)
    if blocked:
        mask[:, sorted(blocked)] = 0
    mask[:, registry.query_position] = 1
    if int(mask.sum()) < 1:
        raise RuntimeError("Answer-source mask removed every key")
    expected_budget = None
    if matched_for is not None:
        expected_budget = len(registry.positions(matched_for))
        if len(blocked) != expected_budget:
            raise RuntimeError("Matched source mask changed the registered budget")
    audit = {
        "condition": condition,
        "blocked_source_groups": source_groups,
        "matched_control_for": matched_for,
        "blocked_token_count": len(blocked),
        "expected_matched_token_count": expected_budget,
        "allowed_key_count": int(mask.sum().item()),
        "blocked_positions_sha256": _sha256_json(sorted(blocked)),
        "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
    }
    return mask, audit


def source_attention_metrics(
    attention_row: torch.Tensor,
    *,
    key_start: int,
    spans: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Compute mass, normalized entropy coverage, and broad score per head."""

    row = attention_row.detach().float().cpu()
    if row.ndim != 2:
        raise ValueError("attention_row must have shape [heads, keys]")
    registered = tuple((int(start), int(end)) for start, end in spans)
    if not registered:
        return [
            {
                "mass": 0.0,
                "coverage": 0.0,
                "broad_score": 0.0,
                "span_masses": (),
            }
            for _head in range(int(row.shape[0]))
        ]
    result: list[dict[str, Any]] = []
    for head in range(int(row.shape[0])):
        masses: list[float] = []
        for start, end in registered:
            local_start = max(0, start - int(key_start))
            local_end = min(int(row.shape[1]), end - int(key_start))
            masses.append(
                float(row[head, local_start:local_end].sum())
                if local_start < local_end
                else 0.0
            )
        total = float(sum(masses))
        if total <= 0:
            coverage = 0.0
        else:
            probabilities = [mass / total for mass in masses if mass > 0]
            entropy = -sum(value * math.log(value) for value in probabilities)
            coverage = math.exp(entropy) / len(masses)
        result.append(
            {
                "mass": total,
                "coverage": float(coverage),
                "broad_score": float(total * coverage),
                "span_masses": tuple(masses),
            }
        )
    return result


@torch.inference_mode()
def capture_answer_source_attention(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    answer_site_id: str = "answer_query_v3",
    layers: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Capture natural answer-query routing to prompt and trace partitions."""

    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    selected_layers = (
        tuple(range(adapter.num_layers))
        if layers is None
        else tuple(sorted({int(value) for value in layers}))
    )
    if not selected_layers or any(
        layer < 0 or layer >= adapter.num_layers for layer in selected_layers
    ):
        raise ValueError("Invalid answer-source attention layers")
    attention_rows, key_starts, logits = position_attention_outputs(
        model, adapter, encoding, encoding.query_position
    )
    first_gold_ids = dict(encoding.count_candidate_answer_token_ids)[encoding.count]
    if not first_gold_ids:
        raise RuntimeError("Gold answer candidate has no token IDs")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gold_first_log_probability = float(log_probs[int(first_gold_ids[0])])
    output_rows: list[dict[str, Any]] = []
    for layer in selected_layers:
        group_metrics = {
            group: source_attention_metrics(
                attention_rows[layer],
                key_start=key_starts[layer],
                spans=registry.spans(group),
            )
            for group in REGISTERED_SOURCE_GROUPS
        }
        for head in range(adapter.num_heads[layer]):
            payload: dict[str, Any] = {
                "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                "experiment_id": "answer_source_attention_capture",
                "request_id": encoding.request_id,
                "model_label": encoding.model_label,
                "seed": encoding.seed,
                "dataset_split": encoding.split,
                "gold_count": encoding.count,
                "answer_site_id": answer_site_id,
                "layer": int(layer),
                "head": int(head),
                "layer_head_count": int(adapter.num_heads[layer]),
                "layer_type": str(adapter.layer_types[layer]),
                "attention_key_start": int(key_starts[layer]),
                "gold_first_token_log_probability": gold_first_log_probability,
            }
            for group, metrics in group_metrics.items():
                metric = metrics[head]
                payload[f"{group}_mass"] = metric["mass"]
                payload[f"{group}_coverage"] = metric["coverage"]
                payload[f"{group}_broad_score"] = metric["broad_score"]
                payload[f"{group}_span_masses"] = json.dumps(metric["span_masses"])
            output_rows.append(payload)
    return pd.DataFrame(output_rows), registry.to_dict()


def rank_answer_broad_heads(
    captures: pd.DataFrame,
    *,
    source_group: str,
    development_seeds: Sequence[int],
    model_label: str | None = None,
) -> pd.DataFrame:
    """Rank answer-query heads with request-first, seed-equal aggregation."""

    if source_group not in REGISTERED_SOURCE_GROUPS:
        raise ValueError(f"Unknown broad source group: {source_group}")
    metric = f"{source_group}_broad_score"
    needed = {
        "request_id",
        "model_label",
        "seed",
        "layer",
        "head",
        "layer_head_count",
        metric,
    }
    missing = sorted(needed - set(captures.columns))
    if missing:
        raise ValueError(f"Broad capture table is missing {missing}")
    allowed = {int(value) for value in development_seeds}
    selected = captures.loc[
        pd.to_numeric(captures["seed"], errors="coerce").isin(allowed)
    ].copy()
    if model_label is not None:
        selected = selected.loc[selected["model_label"].eq(str(model_label))]
    if selected.empty:
        raise ValueError("No development broad-capture rows remain")
    selected[metric] = pd.to_numeric(selected[metric], errors="raise")
    request = selected.groupby(
        [
            "model_label",
            "request_id",
            "seed",
            "layer",
            "head",
            "layer_head_count",
        ],
        as_index=False,
    )[metric].mean()
    seed = request.groupby(
        ["model_label", "seed", "layer", "head", "layer_head_count"],
        as_index=False,
    ).agg(selection_score=(metric, "mean"), request_count=("request_id", "nunique"))
    ranked = (
        seed.groupby(
            ["model_label", "layer", "head", "layer_head_count"],
            as_index=False,
        )
        .agg(
            discovery_score=("selection_score", "mean"),
            discovery_seed_count=("seed", "nunique"),
            discovery_request_count=("request_count", "sum"),
        )
        .sort_values(
            ["model_label", "discovery_score", "layer", "head"],
            ascending=[True, False, True, True],
        )
        .reset_index(drop=True)
    )
    ranked["discovery_rank"] = ranked.groupby("model_label").cumcount() + 1
    ranked["source_group"] = source_group
    ranked["selection_metric"] = metric
    ranked["selection_aggregation"] = "request_first_then_seed_equal"
    return ranked


def build_answer_broad_head_plan(
    ranking: pd.DataFrame,
    *,
    bank_size: int,
    random_controls: int = 3,
    random_seed: int = 0,
    allow_selected_random_overlap: bool = False,
) -> pd.DataFrame:
    """Freeze a global top-K bank and exact layer-matched random controls."""

    needed = {
        "model_label",
        "layer",
        "head",
        "layer_head_count",
        "discovery_rank",
        "source_group",
        "selection_metric",
        "discovery_seed_count",
    }
    missing = sorted(needed - set(ranking.columns))
    if missing:
        raise ValueError(f"Answer broad ranking is missing {missing}")
    if int(bank_size) < 1 or int(random_controls) < 1:
        raise ValueError("bank_size and random_controls must be positive")
    models = sorted(ranking["model_label"].astype(str).unique())
    plan_rows: list[dict[str, Any]] = []
    for model_label in models:
        model = ranking.loc[ranking["model_label"].eq(model_label)].copy()
        model = model.sort_values("discovery_rank")
        if len(model) < int(bank_size):
            raise ValueError(f"{model_label} has fewer than K={bank_size} heads")
        source_groups = set(model["source_group"].astype(str))
        metrics = set(model["selection_metric"].astype(str))
        if len(source_groups) != 1 or len(metrics) != 1:
            raise ValueError("One plan cannot mix source groups or metrics")
        selected_rows = model.head(int(bank_size))
        selected = [
            [int(row.layer), int(row.head)]
            for row in selected_rows.itertuples(index=False)
        ]
        per_layer: dict[int, int] = {}
        for layer, _head in selected:
            per_layer[layer] = per_layer.get(layer, 0) + 1
        observed_heads = {
            int(layer): sorted(
                int(value)
                for value in model.loc[model["layer"].eq(layer), "head"].unique()
            )
            for layer in sorted(per_layer)
        }
        selected_set = {tuple(value) for value in selected}
        for layer, count in per_layer.items():
            available = [
                head
                for head in observed_heads[layer]
                if allow_selected_random_overlap or (layer, head) not in selected_set
            ]
            if len(available) < count:
                raise ValueError(
                    f"{model_label} L{layer}: selected {count} heads but only "
                    f"{len(available)} disjoint controls exist; reduce K"
                )
        common = {
            "schema_version": COUNT_STREAM_SCHEMA_VERSION,
            "experiment_id": "answer_broad_head_ablation",
            "model_label": model_label,
            "source_group": next(iter(source_groups)),
            "selection_metric": next(iter(metrics)),
            "selection_split": "ranking_discovery",
            "selection_aggregation": "request_first_then_seed_equal",
            "selection_seed_count": int(selected_rows["discovery_seed_count"].min()),
            "bank_size": int(bank_size),
            "layer_composition": json.dumps(per_layer, sort_keys=True),
            "random_control_overlap_policy": (
                "nonthinking_allow_treatment_overlap"
                if allow_selected_random_overlap
                else "exclude_treatment_heads"
            ),
        }
        clean_heads: list[list[int]] = []
        for condition, repeat, heads in (
            ("clean", 0, clean_heads),
            ("selected_bank", 0, selected),
        ):
            plan_rows.append(
                {
                    **common,
                    "condition": condition,
                    "repeat": repeat,
                    "heads": json.dumps(heads),
                    "bank_sha256": _sha256_json(heads),
                }
            )
        seen_controls: set[str] = set()
        for repeat in range(1, int(random_controls) + 1):
            control: list[list[int]] | None = None
            for attempt in range(256):
                rng = np.random.default_rng(
                    int(random_seed)
                    + int.from_bytes(
                        hashlib.sha256(
                            f"{model_label}:{repeat}:{attempt}".encode("utf-8")
                        ).digest()[:8],
                        "big",
                    )
                )
                candidate: list[list[int]] = []
                for layer, count in sorted(per_layer.items()):
                    pool = np.asarray(
                        [
                            head
                            for head in observed_heads[layer]
                            if allow_selected_random_overlap
                            or (layer, head) not in selected_set
                        ],
                        dtype=np.int64,
                    )
                    chosen = sorted(
                        int(value)
                        for value in rng.choice(pool, size=count, replace=False)
                    )
                    candidate.extend([[layer, head] for head in chosen])
                digest = _sha256_json(candidate)
                if digest not in seen_controls:
                    control = candidate
                    seen_controls.add(digest)
                    break
            if control is None:
                raise RuntimeError("Could not construct distinct matched random banks")
            plan_rows.append(
                {
                    **common,
                    "condition": "layer_matched_random",
                    "repeat": repeat,
                    "heads": json.dumps(control),
                    "bank_sha256": _sha256_json(control),
                }
            )
    return pd.DataFrame(plan_rows)


def select_answer_broad_bank_size(
    trials: pd.DataFrame,
    *,
    model_label: str,
    source_group: str,
    expected_seeds: Sequence[int],
    expected_bank_sizes: Sequence[int] | None = None,
    expected_requests_per_seed: int = 5,
    expected_random_controls: int = 3,
    boundary_extension_bank_size: int = 64,
    bootstrap_samples: int = 10_000,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select a model/source-specific K on development outcomes only.

    Head membership is already frozen by the attention-only ranking.  This
    function chooses only the length of that nested ranking.  The primary
    discovery estimand is the non-thinking-style absolute output shift,
    evaluated on continuous candidate expected count for stability:

    ``|E[c]_topK-E[c]_clean| - mean_r |E[c]_random_r-E[c]_clean|``.

    Correct-count-margin damage must have the same sign.  Among positive
    candidates, the smallest K within one seed-level standard error of the
    maximum is selected.  A rising maximum at the largest observed K triggers
    a mandatory discovery-only boundary extension instead of silently freezing
    a truncated curve.
    """

    needed = {
        "experiment_id",
        "condition",
        "model_label",
        "source_group",
        "bank_size",
        "repeat",
        "request_id",
        "seed",
        "expected_count",
        "correct_count_margin",
    }
    missing = sorted(needed - set(trials.columns))
    if missing:
        raise ValueError(f"Broad K-selection trials are missing {missing}")
    selected = trials.loc[
        trials["experiment_id"].eq("answer_broad_head_ablation")
        & trials["model_label"].eq(str(model_label))
        & trials["source_group"].eq(str(source_group))
    ].copy()
    if "status" in selected.columns:
        selected = selected.loc[selected["status"].fillna("ok").eq("ok")]
    if selected.empty:
        raise ValueError("No broad-head K-selection trials remain")
    observed_seeds = set(pd.to_numeric(selected["seed"], errors="raise").astype(int))
    registered_seeds = {int(value) for value in expected_seeds}
    if observed_seeds != registered_seeds:
        raise ValueError(
            "K-selection outcomes must cover exactly the frozen discovery seeds; "
            f"missing={sorted(registered_seeds-observed_seeds)}, "
            f"unexpected={sorted(observed_seeds-registered_seeds)}"
        )
    for column in ("bank_size", "repeat", "seed", "expected_count", "correct_count_margin"):
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    observed_bank_sizes = set(selected["bank_size"].astype(int))
    if expected_bank_sizes is not None:
        base_bank_sizes = {int(value) for value in expected_bank_sizes}
        allowed_bank_size_sets = {
            frozenset(base_bank_sizes),
            frozenset(base_bank_sizes | {int(boundary_extension_bank_size)}),
        }
        if frozenset(observed_bank_sizes) not in allowed_bank_size_sets:
            raise ValueError(
                "K-selection trials must cover the complete frozen K grid, "
                "optionally plus only the registered boundary extension"
            )
    request_coverage = selected[
        ["bank_size", "seed", "request_id"]
    ].drop_duplicates()
    coverage_counts = request_coverage.groupby(
        ["bank_size", "seed"], as_index=False
    )["request_id"].nunique()
    if (
        len(coverage_counts)
        != len(observed_bank_sizes) * len(registered_seeds)
        or not coverage_counts["request_id"].eq(
            int(expected_requests_per_seed)
        ).all()
    ):
        raise ValueError(
            "Every frozen K-selection seed/K cell must contain exactly "
            f"{int(expected_requests_per_seed)} requests"
        )
    if "prediction" in selected.columns:
        selected["prediction"] = pd.to_numeric(
            selected["prediction"], errors="coerce"
        )

    request_effect_rows: list[dict[str, Any]] = []
    unit_columns = ["bank_size", "request_id", "seed", "condition", "repeat"]
    aggregations: dict[str, tuple[str, str]] = {
        "expected_count": ("expected_count", "mean"),
        "correct_count_margin": ("correct_count_margin", "mean"),
    }
    if "prediction" in selected.columns:
        aggregations["prediction"] = ("prediction", "mean")
    request_conditions = selected.groupby(unit_columns, as_index=False).agg(
        **aggregations
    )
    for (bank_size, request_id, seed), frame in request_conditions.groupby(
        ["bank_size", "request_id", "seed"], sort=True
    ):
        clean = frame.loc[frame["condition"].eq("clean")]
        treatment = frame.loc[frame["condition"].eq("selected_bank")]
        controls = frame.loc[frame["condition"].eq("layer_matched_random")]
        if len(clean) != 1 or len(treatment) != 1:
            raise ValueError("Every request/K needs one clean and one selected arm")
        if set(controls["repeat"].astype(int)) != set(
            range(1, int(expected_random_controls) + 1)
        ):
            raise ValueError(
                "Every request/K needs the exact frozen matched-random repeats"
            )
        clean_row = clean.iloc[0]
        treatment_row = treatment.iloc[0]
        clean_expected = float(clean_row["expected_count"])
        selected_expected_shift = abs(
            float(treatment_row["expected_count"]) - clean_expected
        )
        random_expected_shift = float(
            np.mean(np.abs(controls["expected_count"].to_numpy(float) - clean_expected))
        )
        selected_margin_damage = float(
            clean_row["correct_count_margin"]
            - treatment_row["correct_count_margin"]
        )
        random_margin_damage = float(
            np.mean(
                float(clean_row["correct_count_margin"])
                - controls["correct_count_margin"].to_numpy(float)
            )
        )
        payload: dict[str, Any] = {
            "model_label": str(model_label),
            "source_group": str(source_group),
            "bank_size": int(bank_size),
            "request_id": str(request_id),
            "seed": int(seed),
            "expected_shift_specificity": float(
                selected_expected_shift - random_expected_shift
            ),
            "margin_damage_specificity": float(
                selected_margin_damage - random_margin_damage
            ),
            "selected_expected_shift": float(selected_expected_shift),
            "random_expected_shift": float(random_expected_shift),
            "random_repeat_count": int(controls["repeat"].nunique()),
        }
        if "prediction" in request_conditions.columns:
            clean_prediction = float(clean_row["prediction"])
            treatment_prediction = float(treatment_row["prediction"])
            control_predictions = controls["prediction"].to_numpy(float)
            if np.isfinite(clean_prediction) and np.isfinite(treatment_prediction):
                selected_strict_shift = abs(treatment_prediction - clean_prediction)
            else:
                selected_strict_shift = math.nan
            valid_controls = control_predictions[np.isfinite(control_predictions)]
            random_strict_shift = (
                float(np.mean(np.abs(valid_controls - clean_prediction)))
                if np.isfinite(clean_prediction) and len(valid_controls)
                else math.nan
            )
            payload["strict_shift_specificity"] = (
                float(selected_strict_shift - random_strict_shift)
                if np.isfinite(selected_strict_shift)
                and np.isfinite(random_strict_shift)
                else math.nan
            )
        request_effect_rows.append(payload)

    requests = pd.DataFrame(request_effect_rows)
    metric_columns = [
        "expected_shift_specificity",
        "margin_damage_specificity",
        "selected_expected_shift",
        "random_expected_shift",
    ]
    if "strict_shift_specificity" in requests.columns:
        metric_columns.append("strict_shift_specificity")
    seed_effects = requests.groupby(
        ["model_label", "source_group", "bank_size", "seed"], as_index=False
    ).agg(
        **{column: (column, "mean") for column in metric_columns},
        request_count=("request_id", "nunique"),
    )
    curve_rows: list[dict[str, Any]] = []
    for bank_size, frame in seed_effects.groupby("bank_size", sort=True):
        values = frame["expected_shift_specificity"].to_numpy(float)
        interval = bootstrap_seed_mean_ci(
            values,
            samples=int(bootstrap_samples),
            seed=int(random_seed) + int(bank_size) * 1009,
        )
        se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else math.inf
        curve_rows.append(
            {
                "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                "model_label": str(model_label),
                "source_group": str(source_group),
                "bank_size": int(bank_size),
                "mean_expected_shift_specificity": float(np.mean(values)),
                "expected_shift_standard_error": se,
                "ci_low": float(interval["ci_low"]),
                "ci_high": float(interval["ci_high"]),
                "p_value": float(sign_flip_pvalue(values)),
                "positive_seed_count": int(np.sum(values > 0)),
                "seed_count": int(len(values)),
                "request_count": int(frame["request_count"].sum()),
                "mean_margin_damage_specificity": float(
                    frame["margin_damage_specificity"].mean()
                ),
                "mean_strict_shift_specificity": (
                    float(frame["strict_shift_specificity"].mean())
                    if "strict_shift_specificity" in frame.columns
                    else math.nan
                ),
            }
        )
    curve = pd.DataFrame(curve_rows).sort_values("bank_size").reset_index(drop=True)
    eligible = curve.loc[
        curve["mean_expected_shift_specificity"].gt(0)
        & curve["mean_margin_damage_specificity"].gt(0)
    ].copy()
    decision: dict[str, Any] = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "model_label": str(model_label),
        "source_group": str(source_group),
        "selection_split": "k_selection_discovery",
        "selection_rule": "smallest_within_one_se_of_max_positive",
        "candidate_bank_sizes": [int(value) for value in curve["bank_size"]],
        "boundary_extension_bank_size": int(boundary_extension_bank_size),
        "selected_bank_size": None,
        "status": "no_positive_discovery_bank",
        "confirmation_outcomes_used": False,
    }
    if not eligible.empty:
        best = eligible.loc[eligible["mean_expected_shift_specificity"].idxmax()]
        best_mean = float(best["mean_expected_shift_specificity"])
        best_se = float(best["expected_shift_standard_error"])
        threshold = best_mean - best_se
        one_se = eligible.loc[
            eligible["mean_expected_shift_specificity"].ge(threshold)
        ]
        chosen = one_se.sort_values("bank_size").iloc[0]
        largest_observed = int(curve["bank_size"].max())
        best_k = int(best["bank_size"])
        previous = curve.loc[curve["bank_size"].lt(largest_observed)].tail(1)
        rising_at_boundary = bool(
            best_k == largest_observed
            and not previous.empty
            and float(best["mean_expected_shift_specificity"])
            > float(previous.iloc[0]["mean_expected_shift_specificity"])
            and largest_observed < int(boundary_extension_bank_size)
        )
        decision.update(
            {
                "best_observed_bank_size": best_k,
                "best_observed_effect": best_mean,
                "one_se_threshold": float(threshold),
                "selected_bank_size": (
                    None if rising_at_boundary else int(chosen["bank_size"])
                ),
                "status": (
                    "requires_boundary_extension"
                    if rising_at_boundary
                    else "frozen_for_confirmation"
                ),
                "required_next_bank_size": (
                    int(boundary_extension_bank_size) if rising_at_boundary else None
                ),
            }
        )
    decision["decision_sha256"] = _sha256_json(decision)
    return curve, seed_effects, decision


def deterministic_control_basis(
    basis: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Construct a deterministic orthonormal control basis in the complement."""

    axes = np.asarray(basis, dtype=np.float64)
    if axes.ndim != 2 or axes.shape[1] < 1:
        raise ValueError("basis must have shape [hidden, rank]")
    gram = axes.T @ axes
    if not np.allclose(gram, np.eye(axes.shape[1]), atol=2e-4, rtol=2e-4):
        raise ValueError("basis must be orthonormal")
    if axes.shape[0] < 2 * axes.shape[1]:
        raise ValueError("Hidden width is too small for a disjoint control basis")
    rng = np.random.default_rng(int(seed))
    candidates = rng.standard_normal((axes.shape[0], axes.shape[1] * 3))
    candidates -= axes @ (axes.T @ candidates)
    q, _r = np.linalg.qr(candidates, mode="reduced")
    control = q[:, : axes.shape[1]]
    if control.shape != axes.shape:
        raise RuntimeError("Could not construct the requested control rank")
    if not np.allclose(control.T @ control, np.eye(control.shape[1]), atol=2e-5):
        raise RuntimeError("Control basis is not orthonormal")
    if float(np.max(np.abs(axes.T @ control))) > 2e-5:
        raise RuntimeError("Control basis is not orthogonal to the count basis")
    return control.astype(np.float32)


def fit_count_stream_basis(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a centroid count/progress basis and its deterministic control."""

    center, basis = fit_centroid_subspace(states, labels, rank=int(rank))
    control = deterministic_control_basis(basis, seed=int(seed))
    return center, basis, control


def load_count_stream_capture_dataset(
    capture_index: str | Path,
    *,
    site_kinds: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load only the capture fields needed for count-stream basis fitting.

    This lightweight loader intentionally avoids importing the full
    representation-analysis stack (SciPy/sklearn) on GPU runners.
    """

    index_path = Path(capture_index)
    root = index_path.parent
    allowed = set(str(value) for value in site_kinds) if site_kinds else None
    metadata_rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    with index_path.open("r", encoding="utf-8") as handle:
        index_rows = [json.loads(line) for line in handle if line.strip()]
    for index_row in index_rows:
        manifest_path = root / str(index_row["manifest_path"])
        states_path = root / str(index_row["states_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(states_path, allow_pickle=False) as archive:
            states = np.asarray(archive["site_states"], dtype=np.float32)
            layers = np.asarray(archive["layer_indices"], dtype=int)
        site_rows = list(manifest["site_rows"])
        if states.shape[:2] != (len(site_rows), len(layers)):
            raise ValueError(f"Capture shape mismatch in {states_path}")
        parser = manifest["parser"]
        for site_axis, site in enumerate(site_rows):
            if allowed is not None and str(site["site_kind"]) not in allowed:
                continue
            for layer_axis, layer in enumerate(layers):
                metadata_rows.append(
                    {
                        "request_id": str(manifest["request_id"]),
                        "model_label": str(manifest["model_label"]),
                        "seed": int(manifest["seed"]),
                        "split": str(
                            manifest.get("split", index_row.get("split", ""))
                        ),
                        "gold_count": int(manifest["gold_count"]),
                        "occurrence": int(site["occurrence"]),
                        "layer": int(layer),
                        "site_id": str(site["site_id"]),
                        "site_kind": str(site["site_kind"]),
                        "parser_hit": bool(parser.get("detected")),
                        "trace_one_to_one": bool(parser.get("trace_one_to_one")),
                        "exact_count": bool(manifest.get("exact_count")),
                    }
                )
                vectors.append(states[site_axis, layer_axis])
    if not vectors:
        raise ValueError("No registered states matched the requested site kinds")
    metadata = pd.DataFrame(metadata_rows)
    state_array = np.stack(vectors, axis=0)
    if len(metadata) != len(state_array) or state_array.ndim != 2:
        raise ValueError("Count-stream capture metadata/state shape mismatch")
    if not np.isfinite(state_array).all():
        raise ValueError("Count-stream capture states contain non-finite values")
    return metadata, state_array


def count_stream_cohort_mask(metadata: pd.DataFrame, cohort: str) -> np.ndarray:
    """Apply the registered native trace cohort without outcome leakage."""

    if cohort not in {"parser_hit", "one_to_one", "one_to_one_correct"}:
        raise ValueError(f"Unknown native trace cohort: {cohort}")
    mask = metadata["parser_hit"].astype(bool).to_numpy(copy=True)
    if cohort in {"one_to_one", "one_to_one_correct"}:
        mask &= metadata["trace_one_to_one"].astype(bool).to_numpy()
    if cohort == "one_to_one_correct":
        mask &= metadata["exact_count"].astype(bool).to_numpy()
    return mask


def _uses_shared_kv(adapter: DecoderAdapter) -> bool:
    return any(
        bool(getattr(attention, "is_kv_shared_layer", False))
        for attention in adapter.attentions
    )


def _prefix_forward(
    model: Any,
    adapter: DecoderAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _uses_shared_kv(adapter):
        kwargs["return_shared_kv_states"] = True
    return model(**kwargs)


def _query_forward_from_prefix(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    prefix_output: Any,
    query_attention_mask: torch.Tensor,
) -> Any:
    input_ids, _attention_mask = _encoding_tensors(model, encoding)
    query = int(encoding.query_position)
    past = getattr(prefix_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Answer-query prefix returned no KV cache")
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": query_attention_mask.to(input_ids.device),
        "past_key_values": past,
        "use_cache": True,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.tensor(
            [[query]], dtype=torch.long, device=input_ids.device
        )
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.tensor(
            [query], dtype=torch.long, device=input_ids.device
        )
    shared = _extract_shared_kv_states(prefix_output)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = shared
    return model(**kwargs)


def _score_and_generate_prefill(
    model: Any,
    tokenizer: Any,
    encoding: NativeTraceEncoding,
    prefill_output: Any,
    *,
    run_greedy: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    scoring_output = clone_prefill_output_for_scoring(prefill_output)
    scored = _score_candidate_sequences(model, encoding, scoring_output)
    result = candidate_sequence_metrics(scored.candidate_log_scores, encoding)
    expected_error = abs(float(result["expected_count"]) - float(encoding.count))
    result["expected_count_absolute_error"] = float(expected_error)
    result["expected_count_utility"] = -float(expected_error)
    if run_greedy:
        completion = generate_answer_completion_from_prefill(
            model,
            tokenizer,
            encoding,
            prefill_output,
            max_new_tokens=max_new_tokens,
        )
        result.update(completion_metrics(completion, gold_count=encoding.count))
        absolute_error = result.get("absolute_error")
        result["invalid_count_output"] = absolute_error is None
        result["strict_count_utility"] = (
            -float(absolute_error) if absolute_error is not None else -10.0
        )
    return result


@torch.inference_mode()
def run_answer_source_mask_trial(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    condition: str,
    answer_site_id: str = "answer_query_v3",
    mask_application: str = "answer_query_and_answer_tokens",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Mask source edges in every head at the final answer query.

    ``answer_query_only`` is directly comparable to the top-K answer-query
    head ablation: later numeric answer tokens regain their clean source edges.
    The persistent option is retained as a stronger sensitivity analysis.
    """

    application = str(mask_application)
    if application not in {
        "answer_query_only",
        "answer_query_and_answer_tokens",
    }:
        raise ValueError(f"Unknown answer source-mask application: {application}")

    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    mask, mask_audit = answer_source_mask(registry, condition=condition)
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    query = int(encoding.query_position)
    prefix = _prefix_forward(
        model, adapter, input_ids[:, :query], attention_mask[:, :query]
    )
    query_output = _query_forward_from_prefix(
        model,
        adapter,
        encoding,
        prefix_output=prefix,
        query_attention_mask=mask,
    )
    scoring_encoding = encoding
    if application == "answer_query_and_answer_tokens":
        scoring_encoding = replace(
            encoding,
            attention_mask=tuple(int(value) for value in mask[0].tolist()),
        )
    outcomes = _score_and_generate_prefill(
        model,
        tokenizer,
        scoring_encoding,
        query_output,
        run_greedy=run_greedy,
        max_new_tokens=max_new_tokens,
    )
    return {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "answer_source_mask_factorial",
        "condition": condition,
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "dataset_split": encoding.split,
        "gold_count": encoding.count,
        "answer_site_id": answer_site_id,
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "mask_scope": application,
        "source_edge_mask_head_scope": "all_attention_heads",
        "source_edge_mask_layer_scope": "all_decoder_layers",
        "decoder_layer_count": int(adapter.num_layers),
        "attention_head_instances_per_query": int(sum(adapter.num_heads)),
        **mask_audit,
        **outcomes,
    }


def _validate_head_bank(
    adapter: DecoderAdapter, heads: Sequence[tuple[int, int]]
) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < adapter.num_layers:
            raise ValueError(f"Invalid answer-query layer: {layer}")
        if not 0 <= head < adapter.num_heads[layer]:
            raise ValueError(f"Invalid answer-query head L{layer}H{head}")
        grouped.setdefault(layer, []).append(head)
    normalized = {
        layer: tuple(sorted(set(values))) for layer, values in grouped.items()
    }
    if sum(len(values) for values in normalized.values()) != len(heads):
        raise ValueError("Answer-query head bank contains duplicate heads")
    return normalized


def _normalize_head_readout_arm(
    adapter: DecoderAdapter, raw_arm: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a frozen arm without changing its serialized bank identity.

    Discovery stores selected heads in ranking order, whereas execution groups
    them canonically by ``(layer, head)``.  The frozen ``bank_sha256`` protects
    the serialized discovery plan and must therefore be checked against the
    original order.  A separate canonical digest is used for set-level checks
    after validating the bank against the active model adapter.
    """

    condition = str(raw_arm["condition"])
    if condition not in REGISTERED_HEAD_READOUT_CONDITIONS:
        raise ValueError(f"Unknown head-readout condition: {condition}")
    repeat = int(raw_arm.get("repeat", 0))
    raw_heads = tuple(
        (int(pair[0]), int(pair[1])) for pair in raw_arm.get("heads", ())
    )
    serialized_heads = [[layer_index, head] for layer_index, head in raw_heads]
    serialized_sha = _sha256_json(serialized_heads)
    planned_sha = str(raw_arm.get("bank_sha256", serialized_sha))
    if planned_sha != serialized_sha:
        raise ValueError("Head-readout bank hash disagrees with its head list")

    grouped = _validate_head_bank(adapter, raw_heads)
    heads = tuple(
        (layer_index, head)
        for layer_index, values in sorted(grouped.items())
        for head in values
    )
    normalized_heads = [[layer_index, head] for layer_index, head in heads]
    if condition == "clean" and heads:
        raise ValueError("Intact head-readout arm must have an empty bank")
    if condition != "clean" and not heads:
        raise ValueError("Ablated head-readout arm requires a nonempty bank")
    return {
        "condition": condition,
        "repeat": repeat,
        "heads": heads,
        "normalized_heads": normalized_heads,
        "bank_size": len(heads),
        "bank_sha256": planned_sha,
        "canonical_bank_sha256": _sha256_json(normalized_heads),
    }


def _query_from_prefix_with_head_ablation(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    prefix_output: Any,
    query_attention_mask: torch.Tensor,
    heads: Sequence[tuple[int, int]],
) -> tuple[Any, list[list[int]], dict[int, int]]:
    """Compute one final query while zeroing exact pre-O head slices.

    The prefix cache has already been built, so these hooks can only affect the
    final answer query.  An empty head bank is the intact readout arm.
    """

    grouped = _validate_head_bank(adapter, heads)
    applications = {layer: 0 for layer in grouped}
    handles = []
    for layer, layer_heads in grouped.items():
        width = int(adapter.head_dims[layer])
        expected = int(adapter.num_heads[layer]) * width

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            layer_heads: tuple[int, ...] = layer_heads,
            width: int = width,
            expected: int = expected,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Answer-query o_proj received no tensor")
            value = args[0]
            if value.ndim != 3 or value.shape[1] != 1 or value.shape[-1] != expected:
                raise RuntimeError("Answer-query head hook saw an unexpected shape")
            patched = value.clone()
            for head in layer_heads:
                start = int(head) * width
                patched[:, 0, start : start + width] = 0
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    try:
        query_output = _query_forward_from_prefix(
            model,
            adapter,
            encoding,
            prefix_output=prefix_output,
            query_attention_mask=query_attention_mask,
        )
    finally:
        for handle in handles:
            handle.remove()
    violations = sorted(layer for layer, count in applications.items() if count != 1)
    if violations:
        raise RuntimeError(
            "Answer-query head ablation must apply exactly once per layer: "
            f"{violations}"
        )
    normalized_heads = [
        [layer, head] for layer, values in sorted(grouped.items()) for head in values
    ]
    return query_output, normalized_heads, applications


@torch.inference_mode()
def run_answer_broad_head_trial(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    heads: Sequence[tuple[int, int]],
    condition: str,
    source_group: str,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Ablate one frozen bank only while the final query is computed."""

    if condition not in {"clean", "selected_bank", "layer_matched_random"}:
        raise ValueError(f"Unknown answer broad-head condition: {condition}")
    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    grouped = _validate_head_bank(adapter, heads)
    if condition == "clean" and grouped:
        raise ValueError("Clean answer-query arm must have an empty head bank")
    if condition != "clean" and not grouped:
        raise ValueError("Ablated answer-query arm requires at least one head")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    query = int(encoding.query_position)
    prefix = _prefix_forward(
        model, adapter, input_ids[:, :query], attention_mask[:, :query]
    )
    query_output, normalized_heads, applications = (
        _query_from_prefix_with_head_ablation(
            model,
            adapter,
            encoding,
            prefix_output=prefix,
            query_attention_mask=attention_mask,
            heads=heads,
        )
    )
    outcomes = _score_and_generate_prefill(
        model,
        tokenizer,
        encoding,
        query_output,
        run_greedy=run_greedy,
        max_new_tokens=max_new_tokens,
    )
    return {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "answer_broad_head_ablation",
        "condition": condition,
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "dataset_split": encoding.split,
        "gold_count": encoding.count,
        "answer_site_id": answer_site_id,
        "source_group": source_group,
        "heads": normalized_heads,
        "bank_size": len(normalized_heads),
        "bank_sha256": _sha256_json(normalized_heads),
        "intervention_scope": "answer_query_only",
        "intervention_hook_applications": {
            str(layer): count for layer, count in sorted(applications.items())
        },
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **outcomes,
    }


def build_trace_patch_pair_plan(
    specifications: Sequence[Mapping[str, Any]],
    *,
    gold_count: int,
    receiver_occurrences: Sequence[int] | None = None,
    donor_offsets: Sequence[int] = (-1, 1),
) -> list[dict[str, Any]]:
    """Build directed donor/receiver pairs at strictly intermediate trace sites.

    Both endpoints must own an eligible ``item_end -> next city`` transition.
    The default adjacent offsets therefore create the two directions of every
    interior pair without using the terminal answer state as a donor.
    """

    count = int(gold_count)
    if count < 1:
        raise ValueError("gold_count must be positive")
    offsets = tuple(int(value) for value in donor_offsets)
    if not offsets or len(set(offsets)) != len(offsets) or 0 in offsets:
        raise ValueError("donor_offsets must be unique and nonzero")
    by_occurrence: dict[int, Mapping[str, Any]] = {}
    for specification in specifications:
        occurrence = int(specification["from_occurrence"])
        if occurrence in by_occurrence:
            raise ValueError(
                f"Progress transition {occurrence} has multiple item-end anchors"
            )
        by_occurrence[occurrence] = specification
    expected = set(range(1, count))
    missing = sorted(expected - set(by_occurrence))
    if missing:
        raise ValueError(
            "Trace patching needs every nonterminal item-end transition; "
            f"missing {missing}"
        )
    receivers = (
        tuple(range(2, count))
        if receiver_occurrences is None
        else tuple(int(value) for value in receiver_occurrences)
    )
    if len(set(receivers)) != len(receivers):
        raise ValueError("receiver_occurrences must be unique")
    invalid_receivers = [value for value in receivers if not 1 < value < count]
    if invalid_receivers:
        raise ValueError(
            "Trace patch receivers must be strictly intermediate occurrences; "
            f"invalid {invalid_receivers}"
        )
    result: list[dict[str, Any]] = []
    for receiver in receivers:
        for offset in offsets:
            donor = receiver + offset
            if donor not in by_occurrence:
                continue
            receiver_specification = dict(by_occurrence[receiver])
            donor_specification = dict(by_occurrence[donor])
            result.append(
                {
                    "receiver_occurrence": int(receiver),
                    "donor_occurrence": int(donor),
                    "donor_offset": int(offset),
                    "donor_direction": (
                        "past_to_later_receiver"
                        if donor < receiver
                        else "future_to_earlier_receiver"
                    ),
                    "receiver_specification": receiver_specification,
                    "donor_specification": donor_specification,
                    "pair_sha256": _sha256_json(
                        {
                            "receiver": receiver,
                            "donor": donor,
                            "receiver_anchor": receiver_specification.get(
                                "anchor_equivalence_id"
                            ),
                            "donor_anchor": donor_specification.get(
                                "anchor_equivalence_id"
                            ),
                        }
                    ),
                }
            )
    return result


def valid_trace_patch_receivers(
    gold_count: int,
    donor_offset: int,
) -> tuple[int, ...]:
    """Return receivers for which both local next-city transitions exist.

    The donor offset is defined as ``donor - receiver``.  Local receivers are
    strictly intermediate and donors are nonterminal.  Thus the standard
    offsets imply the asymmetric valid-count ranges used by the registered
    panel: -1 at 3..10, +1 at 4..10, -3 at 5..10, +3 at 6..10, -5 at 7..10,
    and +5 at 8..10.
    """

    count = int(gold_count)
    offset = int(donor_offset)
    if count < 1:
        raise ValueError("gold_count must be positive")
    if offset == 0:
        raise ValueError("donor_offset cannot be zero")
    return tuple(
        receiver
        for receiver in range(2, count)
        if 1 <= receiver + offset < count
    )


def build_sparse_trace_patch_sample_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
    donor_offsets: Sequence[int] = (-5, -3, -1, 1, 3, 5),
    seeds_per_cell: int = 10,
    sampling_seed: int = 20260820,
    include_count2_terminal_panel: bool = True,
    candidate_counts: Sequence[int] = tuple(range(1, 11)),
) -> pd.DataFrame:
    """Freeze the outcome-blind sparse donor/receiver sampling panel.

    One row is chosen per seed within every valid ``count x signed-offset``
    cell.  Selection hashes use only registry identity fields; neither model
    correctness nor any behavioral outcome is inspected.  Receiver positions
    are then assigned cyclically, with a deterministic cell-specific rotation,
    so their frequencies differ by at most one within a cell.

    Count 2 has no strictly intermediate receiver.  Its optional two-direction
    panel is therefore marked ``terminal`` and is evaluated only at the final
    answer after patching occurrences 1 and 2.
    """

    label = str(model_label)
    offsets = tuple(int(value) for value in donor_offsets)
    counts = tuple(int(value) for value in candidate_counts)
    per_cell = int(seeds_per_cell)
    if not label:
        raise ValueError("model_label cannot be empty")
    if not offsets or len(set(offsets)) != len(offsets) or 0 in offsets:
        raise ValueError("donor_offsets must be unique and nonzero")
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("candidate_counts must be nonempty and unique")
    if per_cell < 1:
        raise ValueError("seeds_per_cell must be positive")

    registry: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in rows:
        seed = int(raw["seed"])
        count = int(
            raw.get(
                "gold_count",
                len(raw.get("gold_records", raw.get("gold_pairs", ()))),
            )
        )
        if count not in set(counts):
            continue
        row_model = raw.get("model_label")
        if row_model not in {None, label}:
            continue
        key = (seed, count)
        if key in registry:
            raise ValueError(
                f"Trace-patch registry has duplicate seed/count row {key}"
            )
        registry[key] = {
            "request_id": str(raw["request_id"]),
            "seed": seed,
            "gold_count": count,
        }
    if not registry:
        raise ValueError("No rows remain for the sparse trace-patch plan")

    def priority(
        *, panel: str, count: int, offset: int, row: Mapping[str, Any]
    ) -> str:
        payload = {
            "sampling_seed": int(sampling_seed),
            "model_label": label,
            "panel_kind": str(panel),
            "gold_count": int(count),
            "donor_offset": int(offset),
            "seed": int(row["seed"]),
            "request_id": str(row["request_id"]),
        }
        return _sha256_json(payload)

    plan_rows: list[dict[str, Any]] = []

    def add_cell(
        *,
        panel: str,
        count: int,
        offset: int,
        receivers: Sequence[int],
    ) -> None:
        receiver_values = tuple(int(value) for value in receivers)
        if not receiver_values:
            raise ValueError("A trace-patch sampling cell has no receiver")
        candidates = [
            dict(value)
            for (seed, observed_count), value in registry.items()
            if observed_count == int(count)
        ]
        ranked = sorted(
            (
                priority(
                    panel=panel,
                    count=int(count),
                    offset=int(offset),
                    row=value,
                ),
                int(value["seed"]),
                str(value["request_id"]),
                value,
            )
            for value in candidates
        )
        if len(ranked) < per_cell:
            raise ValueError(
                f"Trace-patch cell count={count}, offset={offset} needs "
                f"{per_cell} eligible seeds but has {len(ranked)}"
            )
        cell_id = f"{panel}:count={int(count)}:offset={int(offset):+d}"
        rotation = int(
            hashlib.sha256(
                f"{sampling_seed}:{label}:{cell_id}:receiver".encode("utf-8")
            ).hexdigest()[:16],
            16,
        ) % len(receiver_values)
        for selection_index, (digest, _seed, _request, value) in enumerate(
            ranked[:per_cell]
        ):
            receiver_cycle = (selection_index + rotation) % len(receiver_values)
            receiver = int(receiver_values[receiver_cycle])
            donor = int(receiver + int(offset))
            pair_identity = {
                "model_label": label,
                "panel_kind": panel,
                "request_id": str(value["request_id"]),
                "seed": int(value["seed"]),
                "gold_count": int(count),
                "receiver_occurrence": receiver,
                "donor_occurrence": donor,
                "donor_offset": int(offset),
            }
            plan_rows.append(
                {
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                    "experiment_id": "trace_state_patching_pair_plan",
                    "model_label": label,
                    "panel_kind": panel,
                    "selection_cell_id": cell_id,
                    "selection_rank": selection_index + 1,
                    "outcome_blind_priority_sha256": digest,
                    "selection_input_fields": (
                        "sampling_seed,model_label,panel_kind,gold_count,"
                        "donor_offset,seed,request_id"
                    ),
                    "sampling_seed": int(sampling_seed),
                    "seeds_per_cell": per_cell,
                    "receiver_balance_rotation": rotation,
                    "receiver_balance_cycle_position": receiver_cycle,
                    **pair_identity,
                    "donor_direction": (
                        "past_to_later_receiver"
                        if donor < receiver
                        else "future_to_earlier_receiver"
                    ),
                    "receiver_is_terminal": bool(receiver == int(count)),
                    "donor_is_terminal": bool(donor == int(count)),
                    "local_next_city_outcome_registered": bool(panel == "local"),
                    "final_answer_outcome_registered": True,
                    "pair_sha256": _sha256_json(pair_identity),
                }
            )

    for count in sorted(counts):
        for offset in offsets:
            receivers = valid_trace_patch_receivers(count, offset)
            if receivers:
                add_cell(
                    panel="local",
                    count=count,
                    offset=offset,
                    receivers=receivers,
                )
    if include_count2_terminal_panel:
        if 2 not in set(counts):
            raise ValueError("The terminal panel requires count 2")
        add_cell(panel="terminal", count=2, offset=-1, receivers=(2,))
        add_cell(panel="terminal", count=2, offset=1, receivers=(1,))

    plan = pd.DataFrame(plan_rows)
    if plan.empty:
        raise ValueError("Sparse trace-patch plan has no registered cells")
    return plan.sort_values(
        ["panel_kind", "gold_count", "donor_offset", "selection_rank"],
        kind="stable",
    ).reset_index(drop=True)


def build_terminal_last_trace_patch_sample_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
    seeds_per_cell: int = 10,
    sampling_seed: int = 20260820,
) -> pd.DataFrame:
    """Freeze the natural past-to-final receiver panel without outcomes.

    Every receiver is the last observed count occurrence, immediately before
    the answer query in the frozen teacher-forced trace.  Donors are earlier by
    1, 3, or 5 occurrences.  The asymmetric count ranges are intentional:
    ``-1`` uses counts 2..10, ``-3`` uses 5..10, and ``-5`` uses 7..10.
    Selection depends only on registry identity and the sampling seed.
    """

    label = str(model_label)
    per_cell = int(seeds_per_cell)
    if not label:
        raise ValueError("model_label cannot be empty")
    if per_cell < 1:
        raise ValueError("seeds_per_cell must be positive")

    allowed_counts = {
        count for values in TERMINAL_LAST_COUNT_RANGES.values() for count in values
    }
    registry: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in rows:
        seed = int(raw["seed"])
        count = int(
            raw.get(
                "gold_count",
                len(raw.get("gold_records", raw.get("gold_pairs", ()))),
            )
        )
        if count not in allowed_counts:
            continue
        row_model = raw.get("model_label")
        if row_model not in {None, label}:
            continue
        key = (seed, count)
        if key in registry:
            raise ValueError(f"Terminal-last registry has duplicate seed/count {key}")
        registry[key] = {
            "request_id": str(raw["request_id"]),
            "seed": seed,
            "gold_count": count,
        }
    if not registry:
        raise ValueError("No rows remain for the terminal-last trace-patch plan")

    plan_rows: list[dict[str, Any]] = []
    for offset in sorted(TERMINAL_LAST_COUNT_RANGES):
        for count in TERMINAL_LAST_COUNT_RANGES[offset]:
            candidates = [
                dict(value)
                for (_seed, observed_count), value in registry.items()
                if observed_count == int(count)
            ]
            ranked: list[tuple[str, int, str, dict[str, Any]]] = []
            for value in candidates:
                priority_payload = {
                    "sampling_seed": int(sampling_seed),
                    "model_label": label,
                    "panel_kind": "terminal_last",
                    "gold_count": int(count),
                    "donor_offset": int(offset),
                    "seed": int(value["seed"]),
                    "request_id": str(value["request_id"]),
                }
                ranked.append(
                    (
                        _sha256_json(priority_payload),
                        int(value["seed"]),
                        str(value["request_id"]),
                        value,
                    )
                )
            ranked.sort()
            if len(ranked) < per_cell:
                raise ValueError(
                    f"Terminal-last cell count={count}, offset={offset} needs "
                    f"{per_cell} eligible seeds but has {len(ranked)}"
                )
            cell_id = f"terminal_last:count={count}:offset={offset:+d}"
            for selection_index, (digest, _seed, _request, value) in enumerate(
                ranked[:per_cell]
            ):
                receiver = int(count)
                donor = receiver + int(offset)
                if not 1 <= donor < receiver:
                    raise RuntimeError("Terminal-last donor is not a natural past state")
                pair_identity = {
                    "model_label": label,
                    "panel_kind": "terminal_last",
                    "request_id": str(value["request_id"]),
                    "seed": int(value["seed"]),
                    "gold_count": int(count),
                    "receiver_occurrence": receiver,
                    "donor_occurrence": donor,
                    "donor_offset": int(offset),
                }
                plan_rows.append(
                    {
                        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                        "experiment_id": "trace_terminal_last_pair_plan",
                        "model_label": label,
                        "panel_kind": "terminal_last",
                        "selection_cell_id": cell_id,
                        "selection_rank": selection_index + 1,
                        "outcome_blind_priority_sha256": digest,
                        "selection_input_fields": (
                            "sampling_seed,model_label,panel_kind,gold_count,"
                            "donor_offset,seed,request_id"
                        ),
                        "sampling_seed": int(sampling_seed),
                        "seeds_per_cell": per_cell,
                        **pair_identity,
                        "donor_direction": "past_to_later_receiver",
                        "receiver_is_terminal": True,
                        "donor_is_terminal": False,
                        "local_next_city_outcome_registered": False,
                        "final_answer_outcome_registered": True,
                        "pair_sha256": _sha256_json(pair_identity),
                    }
                )
    plan = pd.DataFrame(plan_rows)
    expected_cells = sum(len(values) for values in TERMINAL_LAST_COUNT_RANGES.values())
    if len(plan) != expected_cells * per_cell:
        raise RuntimeError("Terminal-last plan did not realize every frozen cell")
    return plan.sort_values(
        ["gold_count", "donor_offset", "selection_rank"], kind="stable"
    ).reset_index(drop=True)


def build_terminal_serial_pair_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
) -> pd.DataFrame:
    """Use every eligible registered seed in the terminal serial-readout panel.

    Unlike the older fixed-budget patch sampler, this plan has no hash rank and
    no pair-level discovery/confirmation split.  Discovery versus confirmation
    is determined only by the row registry's canonical seed role.  Within each
    count/offset cell, every eligible row appears exactly once in seed order.
    """

    label = str(model_label)
    if not label:
        raise ValueError("model_label cannot be empty")
    allowed_counts = {
        count for values in TERMINAL_LAST_COUNT_RANGES.values() for count in values
    }
    registry: dict[tuple[int, int], dict[str, Any]] = {}
    observed_roles: set[str] = set()
    for raw in rows:
        seed = int(raw["seed"])
        count = int(
            raw.get(
                "gold_count",
                len(raw.get("gold_records", raw.get("gold_pairs", ()))),
            )
        )
        if count not in allowed_counts:
            continue
        row_model = raw.get("model_label")
        if row_model not in {None, label}:
            continue
        role = str(raw.get("mechanism_split", raw.get("split", "")))
        if role:
            observed_roles.add(role)
        key = (seed, count)
        if key in registry:
            raise ValueError(f"Terminal serial registry has duplicate seed/count {key}")
        registry[key] = {
            "request_id": str(raw["request_id"]),
            "seed": seed,
            "gold_count": count,
            "seed_role": role,
        }
    if not registry:
        raise ValueError("No rows remain for the terminal serial-readout plan")
    if len(observed_roles) > 1:
        raise ValueError("Terminal serial plan cannot mix discovery and confirmation")

    plan_rows: list[dict[str, Any]] = []
    for offset in sorted(TERMINAL_LAST_COUNT_RANGES):
        for count in TERMINAL_LAST_COUNT_RANGES[offset]:
            candidates = sorted(
                (
                    dict(value)
                    for (_seed, observed_count), value in registry.items()
                    if observed_count == int(count)
                ),
                key=lambda value: (int(value["seed"]), str(value["request_id"])),
            )
            if not candidates:
                raise ValueError(
                    f"Terminal serial cell count={count}, offset={offset} has no "
                    "eligible registered rows"
                )
            cell_id = f"terminal_serial:count={count}:offset={offset:+d}"
            eligible_count = len(candidates)
            for within_cell_index, value in enumerate(candidates, start=1):
                receiver = int(count)
                donor = receiver + int(offset)
                if not 1 <= donor < receiver:
                    raise RuntimeError("Terminal serial donor is not a natural past state")
                pair_identity = {
                    "model_label": label,
                    "panel_kind": "terminal_serial",
                    "request_id": str(value["request_id"]),
                    "seed": int(value["seed"]),
                    "gold_count": int(count),
                    "receiver_occurrence": receiver,
                    "donor_occurrence": donor,
                    "donor_offset": int(offset),
                }
                plan_rows.append(
                    {
                        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                        "experiment_id": "trace_terminal_serial_pair_plan",
                        "model_label": label,
                        "panel_kind": "terminal_serial",
                        "selection_policy": "all_eligible_registered_seeds",
                        "selection_cell_id": cell_id,
                        "within_cell_index": within_cell_index,
                        "eligible_seed_count": eligible_count,
                        "seed_role": str(value["seed_role"]),
                        **pair_identity,
                        "donor_direction": "past_to_later_receiver",
                        "receiver_is_terminal": True,
                        "donor_is_terminal": False,
                        "final_answer_outcome_registered": True,
                        "pair_sha256": _sha256_json(pair_identity),
                    }
                )
    plan = pd.DataFrame(plan_rows)
    if plan["pair_sha256"].duplicated().any():
        raise RuntimeError("Terminal serial plan emitted duplicate pairs")
    return plan.sort_values(
        ["gold_count", "donor_offset", "seed"], kind="stable"
    ).reset_index(drop=True)


def trace_intermediate_patch_pairs(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    receiver_occurrences: Sequence[int] | None = None,
    donor_offsets: Sequence[int] = (-1, 1),
) -> list[dict[str, Any]]:
    """Compile the registered directed patch pairs for one native trace."""

    specifications, _excluded = mechanism_continuations(
        row, tokenizer, mechanism="progress_transition"
    )
    return build_trace_patch_pair_plan(
        specifications,
        gold_count=len(gold_records(row)),
        receiver_occurrences=receiver_occurrences,
        donor_offsets=donor_offsets,
    )


def trace_patch_condition_states(
    receiver_state: np.ndarray | torch.Tensor,
    donor_state: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
    *,
    random_seed: int,
) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
    """Construct full, progress-only, and norm-matched control patches."""

    receiver = (
        torch.as_tensor(receiver_state).detach().to(dtype=torch.float32, device="cpu")
    ).reshape(-1)
    donor = (
        torch.as_tensor(donor_state).detach().to(dtype=torch.float32, device="cpu")
    ).reshape(-1)
    axes = torch.as_tensor(basis).detach().to(dtype=torch.float32, device="cpu")
    if receiver.shape != donor.shape:
        raise ValueError("Receiver and donor states must have the same width")
    if axes.ndim != 2 or axes.shape[0] != receiver.numel() or axes.shape[1] < 1:
        raise ValueError("Trace patch basis must have shape [hidden, rank]")
    gram = axes.T @ axes
    if not torch.allclose(gram, torch.eye(axes.shape[1]), atol=2e-4, rtol=2e-4):
        raise ValueError("Trace patch basis must be orthonormal")
    full_delta = donor - receiver
    projected_delta = (full_delta @ axes) @ axes.T
    orthogonal_delta = norm_matched_orthogonal_delta(
        projected_delta,
        axes,
        seed=int(random_seed),
    ).reshape(-1)
    projected_norm = float(torch.linalg.vector_norm(projected_delta))
    orthogonal_norm = float(torch.linalg.vector_norm(orthogonal_delta))
    tolerance = max(1e-6, projected_norm * 2e-5)
    if abs(projected_norm - orthogonal_norm) > tolerance:
        raise RuntimeError("Trace patch control is not norm matched")
    orthogonal_overlap = float(torch.max(torch.abs(orthogonal_delta @ axes)))
    if orthogonal_overlap > max(1e-5, orthogonal_norm * 2e-5):
        raise RuntimeError("Trace patch control is not orthogonal to progress basis")
    states: dict[str, torch.Tensor | None] = {
        "clean": None,
        "self_patch": receiver.clone(),
        "full_donor_patch": donor.clone(),
        "progress_projected_patch": receiver + projected_delta,
        "norm_matched_orthogonal_patch": receiver + orthogonal_delta,
    }
    full_norm = float(torch.linalg.vector_norm(full_delta))
    audit = {
        "basis_rank": int(axes.shape[1]),
        "full_donor_delta_norm": full_norm,
        "progress_projected_delta_norm": projected_norm,
        "orthogonal_control_delta_norm": orthogonal_norm,
        "progress_fraction_of_full_delta_norm": (
            projected_norm / full_norm if full_norm > 1e-12 else 0.0
        ),
        "orthogonal_control_progress_max_abs_coordinate": orthogonal_overlap,
        "condition_patch_delta_norms": {
            condition: (
                0.0
                if state is None
                else float(torch.linalg.vector_norm(state - receiver))
            )
            for condition, state in states.items()
        },
    }
    return states, audit


def _fixed_state_transform(state: torch.Tensor):
    value = state.detach().float().reshape(1, 1, -1)

    def transform(selected: torch.Tensor) -> torch.Tensor:
        if selected.ndim != 3 or selected.shape[1] != 1:
            raise ValueError("A trace patch must target exactly one occurrence")
        if selected.shape[-1] != value.shape[-1]:
            raise ValueError("Trace patch state width disagrees with the model")
        return value.to(device=selected.device, dtype=selected.dtype).expand_as(
            selected
        )

    return transform


@torch.inference_mode()
def _score_trace_continuation(
    model: Any,
    encoding: NativeTraceEncoding,
    prefill_output: Any,
    token_ids: Sequence[int],
    *,
    city_token_offset: int,
) -> dict[str, Any]:
    """Autoregressively score one interstitial-plus-city continuation."""

    tokens = tuple(int(value) for value in token_ids)
    city_offset = int(city_token_offset)
    if not tokens or not 0 <= city_offset < len(tokens):
        raise ValueError("Trace continuation has invalid city-token bounds")
    scoring_output = clone_prefill_output_for_scoring(prefill_output)
    logits = getattr(scoring_output, "logits", None)
    past = getattr(scoring_output, "past_key_values", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or past is None:
        raise RuntimeError("Trace continuation prefill exposes no logits/cache")
    first_log_probabilities = torch.log_softmax(logits[0, -1].float(), dim=-1)
    token_log_probabilities = [first_log_probabilities[tokens[0]]]
    if len(tokens) > 1:
        device = logits.device
        continuation_inputs = torch.tensor(
            [tokens[:-1]], dtype=torch.long, device=device
        )
        continuation_mask = torch.ones_like(continuation_inputs)
        base_mask = torch.tensor(
            [encoding.attention_mask], dtype=torch.long, device=device
        )
        kwargs: dict[str, Any] = {
            "input_ids": continuation_inputs,
            "attention_mask": torch.cat((base_mask, continuation_mask), dim=1),
            "past_key_values": past,
            "use_cache": False,
        }
        positions = torch.arange(
            encoding.sequence_length,
            encoding.sequence_length + len(tokens) - 1,
            dtype=torch.long,
            device=device,
        )
        if _accepts_keyword(model, "position_ids"):
            kwargs["position_ids"] = positions.unsqueeze(0)
        if _accepts_keyword(model, "cache_position"):
            kwargs["cache_position"] = positions
        shared = _extract_shared_kv_states(scoring_output)
        if shared is not None and _accepts_keyword(model, "shared_kv_states"):
            kwargs["shared_kv_states"] = shared
        continuation_output = model(**kwargs)
        continuation_logits = getattr(continuation_output, "logits", None)
        if (
            not isinstance(continuation_logits, torch.Tensor)
            or continuation_logits.ndim != 3
            or continuation_logits.shape[1] != len(tokens) - 1
        ):
            raise RuntimeError("Trace continuation returned unexpected logits")
        continuation_log_probabilities = torch.log_softmax(
            continuation_logits[0].float(), dim=-1
        )
        token_log_probabilities.extend(
            continuation_log_probabilities[index, token]
            for index, token in enumerate(tokens[1:])
        )
    values = torch.stack(token_log_probabilities).detach().float().cpu()
    city_values = values[city_offset:]
    return {
        "sequence_log_probability": float(values.sum()),
        "mean_token_log_probability": float(values.mean()),
        "city_log_probability": float(city_values.sum()),
        "mean_city_token_log_probability": float(city_values.mean()),
        "token_count": len(tokens),
        "city_token_count": len(tokens) - city_offset,
    }


@torch.inference_mode()
def run_trace_intermediate_patch_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    layer: int,
    basis: np.ndarray | torch.Tensor,
    readout_layers: Sequence[int],
    readout_bases: Mapping[int, np.ndarray],
    conditions: Sequence[str] = REGISTERED_TRACE_PATCH_CONDITIONS,
    answer_site_id: str = "answer_query_v3",
    random_seed: int = 0,
    run_greedy: bool = True,
    max_new_tokens: int = 48,
) -> list[dict[str, Any]]:
    """Patch one intermediate trace occurrence with another occurrence state.

    The local branch is teacher-forced only through the receiver ``item_end``.
    It scores both the receiver's clean successor and the donor's successor at
    the same grammar slot, then optionally free-generates the next record.  A
    second full-trace branch measures whether the perturbation survives the
    remaining clean suffix and reaches later item endpoints or the answer.
    """

    requested = tuple(str(value) for value in conditions)
    if len(set(requested)) != len(requested):
        raise ValueError("Trace patch conditions must be unique")
    unknown = sorted(set(requested) - set(REGISTERED_TRACE_PATCH_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown trace patch conditions: {unknown}")
    if "clean" not in requested or "self_patch" not in requested:
        raise ValueError("Trace patch trials require clean and self_patch controls")
    source_layer = int(layer)
    if not 0 <= source_layer < int(adapter.num_layers) - 1:
        raise ValueError("Trace patch layer must leave a downstream decoder layer")
    count = len(gold_records(row))
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    if not 1 < receiver < count:
        raise ValueError("Receiver must be a strictly intermediate occurrence")
    if donor == receiver or not 1 <= donor < count:
        raise ValueError("Donor must be a distinct nonterminal occurrence")
    specifications, _excluded = mechanism_continuations(
        row, tokenizer, mechanism="progress_transition"
    )
    by_occurrence = {
        int(value["from_occurrence"]): dict(value) for value in specifications
    }
    if receiver not in by_occurrence or donor not in by_occurrence:
        raise ValueError("Receiver/donor lacks a registered progress transition")
    receiver_specification = by_occurrence[receiver]
    donor_specification = by_occurrence[donor]

    answer_encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    endpoints = tuple(end - 1 for _start, end in registry.trace_items)
    if len(endpoints) != count:
        raise RuntimeError("Answer registry and oracle count disagree")
    receiver_position = endpoints[receiver - 1]
    donor_position = endpoints[donor - 1]
    _last_logits, captured = capture_post_block_states(
        model,
        adapter,
        answer_encoding,
        [receiver_position, donor_position],
        layers=[source_layer],
    )
    receiver_state, donor_state = captured[source_layer]
    condition_states, patch_audit = trace_patch_condition_states(
        receiver_state,
        donor_state,
        basis,
        random_seed=int(random_seed),
    )

    active_readout_layers = tuple(sorted({int(value) for value in readout_layers}))
    if not active_readout_layers:
        active_readout_layers = (source_layer + 1,)
    missing_bases = sorted(set(active_readout_layers) - set(readout_bases))
    if missing_bases:
        raise ValueError(f"Trace patch readout bases are missing {missing_bases}")
    readout_positions = tuple(
        position for position in endpoints if position > receiver_position
    ) + (registry.query_position,)

    query_output_index = int(receiver_specification["query_output_token_index"])
    target_output_start = int(receiver_specification["target_output_token_start"])
    target_output_end = int(receiver_specification["target_output_token_end"])
    baseline_output_ids = output_token_ids(row)
    receiver_path = tuple(
        baseline_output_ids[query_output_index + 1 : target_output_end]
    )
    city_offset = target_output_start - query_output_index - 1
    receiver_city_ids = tuple(
        int(value) for value in receiver_specification["target_token_ids"]
    )
    if receiver_path[city_offset:] != receiver_city_ids:
        raise RuntimeError("Receiver path disagrees with its registered target city")
    donor_city_ids = tuple(
        int(value) for value in donor_specification["target_token_ids"]
    )
    donor_path = receiver_path[:city_offset] + donor_city_ids
    receiver_path_text = tokenizer.decode(
        list(receiver_path),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    donor_path_text = tokenizer.decode(
        list(donor_path),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    local_encoding = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=query_output_index,
        sequence_output_token_end=query_output_index + 1,
        selected_site=receiver_specification,
    )
    if local_encoding.query_position != (
        local_encoding.prompt_token_count + query_output_index
    ):
        raise RuntimeError("Local receiver anchor moved during causal compilation")
    if local_encoding.query_position != receiver_position:
        raise RuntimeError(
            "Causal p0 receiver and answer-registry item endpoint disagree"
        )
    donor_causal_position = local_encoding.prompt_token_count + int(
        donor_specification["query_output_token_index"]
    )
    if donor_causal_position != donor_position:
        raise RuntimeError("Causal p0 donor and answer-registry item endpoint disagree")

    known_cities = tuple(str(value["city"]) for value in gold_records(row))
    receiver_next_city = str(receiver_specification["target_city"])
    donor_next_city = str(donor_specification["target_city"])
    if receiver_next_city.casefold() == donor_next_city.casefold():
        raise ValueError("Donor and receiver successors must be distinct cities")
    if receiver_next_city.casefold() not in str(receiver_path_text).casefold():
        raise RuntimeError("Decoded receiver continuation lost its target city")
    if donor_next_city.casefold() not in str(donor_path_text).casefold():
        raise RuntimeError("Decoded donor continuation lost its target city")
    marker_positions = set(registry.positions("trace_markers"))
    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "trace_intermediate_state_patching",
        "request_id": answer_encoding.request_id,
        "model_label": answer_encoding.model_label,
        "seed": answer_encoding.seed,
        "dataset_split": answer_encoding.split,
        "gold_count": answer_encoding.count,
        "answer_site_id": answer_site_id,
        "layer": source_layer,
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "donor_offset": donor - receiver,
        "donor_direction": (
            "past_to_later_receiver"
            if donor < receiver
            else "future_to_earlier_receiver"
        ),
        "future_donor_is_counterfactual_not_natural_stream": bool(donor > receiver),
        "receiver_position": int(receiver_position),
        "donor_position": int(donor_position),
        "receiver_is_visible_marker_token": receiver_position in marker_positions,
        "donor_is_visible_marker_token": donor_position in marker_positions,
        "receiver_anchor_equivalence_id": str(
            receiver_specification["anchor_equivalence_id"]
        ),
        "donor_anchor_equivalence_id": str(
            donor_specification["anchor_equivalence_id"]
        ),
        "receiver_expected_next_city": receiver_next_city,
        "donor_expected_next_city": donor_next_city,
        "receiver_path_token_count": len(receiver_path),
        "donor_path_token_count": len(donor_path),
        "receiver_path_text": str(receiver_path_text),
        "donor_path_text": str(donor_path_text),
        "teacher_forced_interstitial_token_count": city_offset,
        "local_branch_policy": (
            "teacher_force_through_receiver_item_end_then_score_or_generate"
        ),
        "full_branch_policy": (
            "patch_receiver_item_end_in_complete_teacher_forced_answer_prefix"
        ),
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **patch_audit,
    }
    result_rows: list[dict[str, Any]] = []
    states_by_condition: dict[str, np.ndarray] = {}
    for condition in requested:
        replacement_state = condition_states[condition]
        local_prefill, local_applications, local_realized_norm = (
            _prefill_with_state_replacements(
                model,
                adapter,
                local_encoding,
                layer=source_layer,
                positions=(local_encoding.query_position,),
                states=(
                    None
                    if replacement_state is None
                    else replacement_state.reshape(1, -1)
                ),
            )
        )
        receiver_score = _score_trace_continuation(
            model,
            local_encoding,
            local_prefill,
            receiver_path,
            city_token_offset=city_offset,
        )
        donor_score = _score_trace_continuation(
            model,
            local_encoding,
            local_prefill,
            donor_path,
            city_token_offset=city_offset,
        )
        transform = (
            None
            if replacement_state is None
            else _fixed_state_transform(replacement_state)
        )
        full_prefill, readout_capture, full_applications = (
            _full_prefill_with_stream_transform(
                model,
                adapter,
                answer_encoding,
                source_layer=source_layer,
                source_positions=(receiver_position,),
                transform=transform,
                readout_layers=active_readout_layers,
                readout_positions=readout_positions,
            )
        )
        final_outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            answer_encoding,
            full_prefill,
            run_greedy=run_greedy,
            max_new_tokens=max_new_tokens,
        )
        ordered_layers = sorted(readout_capture)
        readout_states = np.stack(
            [readout_capture[value].numpy() for value in ordered_layers]
        )
        states_by_condition[condition] = readout_states
        local_outcomes: dict[str, Any] = {
            "receiver_path_log_probability": receiver_score["sequence_log_probability"],
            "donor_path_log_probability": donor_score["sequence_log_probability"],
            "donor_vs_receiver_path_log_odds": float(
                donor_score["sequence_log_probability"]
                - receiver_score["sequence_log_probability"]
            ),
            "receiver_city_log_probability": receiver_score["city_log_probability"],
            "donor_city_log_probability": donor_score["city_log_probability"],
            "donor_vs_receiver_city_log_odds": float(
                donor_score["city_log_probability"]
                - receiver_score["city_log_probability"]
            ),
            "donor_vs_receiver_mean_city_log_probability": float(
                donor_score["mean_city_token_log_probability"]
                - receiver_score["mean_city_token_log_probability"]
            ),
        }
        if run_greedy:
            local_completion = generate_answer_completion_from_prefill(
                model,
                tokenizer,
                local_encoding,
                local_prefill,
                max_new_tokens=max_new_tokens,
            )
            generated_ids = tuple(
                int(value) for value in local_completion["generated_token_ids"]
            )
            generated_city, generated_city_start, generated_city_evidence = (
                _first_generated_city_record(
                    str(local_completion["completion_text"]), known_cities
                )
            )
            local_outcomes.update(
                {
                    "local_completion_text": str(local_completion["completion_text"]),
                    "local_generated_token_ids": list(generated_ids),
                    "local_generation_truncated": bool(
                        local_completion["generation_truncated"]
                    ),
                    "first_generated_city_record": generated_city,
                    "first_generated_city_record_char_start": generated_city_start,
                    "first_generated_city_record_evidence": generated_city_evidence,
                    "donor_city_adoption": bool(
                        generated_city is not None
                        and generated_city.casefold() == donor_next_city.casefold()
                    ),
                    "receiver_city_retention": bool(
                        generated_city is not None
                        and generated_city.casefold() == receiver_next_city.casefold()
                    ),
                    "receiver_path_exact_prefix": bool(
                        generated_ids[: len(receiver_path)] == receiver_path
                    ),
                    "donor_path_exact_prefix": bool(
                        generated_ids[: len(donor_path)] == donor_path
                    ),
                }
            )
        result_rows.append(
            {
                **common,
                "condition": condition,
                "status": "ok",
                "local_patch_hook_applications": local_applications,
                "local_patch_realized_fro_norm": local_realized_norm,
                "full_patch_hook_applications": full_applications,
                "condition_patch_delta_norm": patch_audit[
                    "condition_patch_delta_norms"
                ][condition],
                "readout_layers": ordered_layers,
                "readout_positions": list(readout_positions),
                **local_outcomes,
                **final_outcomes,
            }
        )
    reference = states_by_condition["self_patch"]
    for result in result_rows:
        metrics = stream_state_retention_metrics(
            reference,
            states_by_condition[str(result["condition"])],
            readout_layers=result["readout_layers"],
            readout_positions=result["readout_positions"],
            query_position=registry.query_position,
            count_bases=readout_bases,
        )
        result.update(metrics)
        result["downstream_progress_transport_magnitude"] = metrics[
            "downstream_item_progress_subspace_displacement_rms"
        ]
    return result_rows


@torch.inference_mode()
def run_trace_terminal_patch_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    layer: int,
    basis: np.ndarray | torch.Tensor,
    readout_layers: Sequence[int],
    readout_bases: Mapping[int, np.ndarray],
    conditions: Sequence[str] = REGISTERED_TRACE_PATCH_CONDITIONS,
    answer_site_id: str = "answer_query_v3",
    random_seed: int = 0,
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Run the count-2 endpoint panel using final-answer outcomes only.

    Count 2 has no strictly intermediate receiver with a local successor on
    both sides.  The registered auxiliary panel therefore transports the two
    item-end states in both directions while keeping the complete answer
    prefix teacher-forced.  It never fabricates a local next-city outcome.
    """

    requested = tuple(str(value) for value in conditions)
    if len(set(requested)) != len(requested):
        raise ValueError("Trace patch conditions must be unique")
    unknown = sorted(set(requested) - set(REGISTERED_TRACE_PATCH_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown trace patch conditions: {unknown}")
    if "clean" not in requested or "self_patch" not in requested:
        raise ValueError("Trace patch trials require clean and self_patch controls")
    if len(gold_records(row)) != 2:
        raise ValueError("The terminal trace-patch panel is registered only at count 2")
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    if (receiver, donor) not in {(2, 1), (1, 2)}:
        raise ValueError("Count-2 terminal patching requires the directed pair 1<->2")
    source_layer = int(layer)
    if not 0 <= source_layer < int(adapter.num_layers) - 1:
        raise ValueError("Trace patch layer must leave a downstream decoder layer")

    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    endpoints = tuple(end - 1 for _start, end in registry.trace_items)
    if len(endpoints) != 2:
        raise RuntimeError("Count-2 answer registry must expose two item endpoints")
    receiver_position = int(endpoints[receiver - 1])
    donor_position = int(endpoints[donor - 1])
    _last_logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        [receiver_position, donor_position],
        layers=[source_layer],
    )
    receiver_state, donor_state = captured[source_layer]
    condition_states, patch_audit = trace_patch_condition_states(
        receiver_state,
        donor_state,
        basis,
        random_seed=int(random_seed),
    )
    active_readout_layers = tuple(sorted({int(value) for value in readout_layers}))
    if not active_readout_layers:
        active_readout_layers = (source_layer + 1,)
    missing_bases = sorted(set(active_readout_layers) - set(readout_bases))
    if missing_bases:
        raise ValueError(f"Trace patch readout bases are missing {missing_bases}")
    readout_positions = tuple(
        position for position in endpoints if position > receiver_position
    ) + (registry.query_position,)
    marker_positions = set(registry.positions("trace_markers"))
    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "trace_terminal_state_patching",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "dataset_split": encoding.split,
        "gold_count": encoding.count,
        "answer_site_id": answer_site_id,
        "layer": source_layer,
        "panel_kind": "terminal",
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "donor_offset": donor - receiver,
        "donor_direction": (
            "past_to_later_receiver"
            if donor < receiver
            else "future_to_earlier_receiver"
        ),
        "future_donor_is_counterfactual_not_natural_stream": bool(donor > receiver),
        "receiver_position": receiver_position,
        "donor_position": donor_position,
        "receiver_is_terminal": bool(receiver == 2),
        "donor_is_terminal": bool(donor == 2),
        "receiver_is_visible_marker_token": receiver_position in marker_positions,
        "donor_is_visible_marker_token": donor_position in marker_positions,
        "local_next_city_outcome_registered": False,
        "final_answer_outcome_registered": True,
        "terminal_panel_answer_only": True,
        "full_branch_policy": (
            "patch_count2_item_end_in_complete_teacher_forced_answer_prefix"
        ),
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **patch_audit,
    }
    result_rows: list[dict[str, Any]] = []
    states_by_condition: dict[str, np.ndarray] = {}
    for condition in requested:
        replacement_state = condition_states[condition]
        transform = (
            None
            if replacement_state is None
            else _fixed_state_transform(replacement_state)
        )
        prefill, captures, applications = _full_prefill_with_stream_transform(
            model,
            adapter,
            encoding,
            source_layer=source_layer,
            source_positions=(receiver_position,),
            transform=transform,
            readout_layers=active_readout_layers,
            readout_positions=readout_positions,
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            encoding,
            prefill,
            run_greedy=run_greedy,
            max_new_tokens=max_new_tokens,
        )
        ordered_layers = sorted(captures)
        readout_states = np.stack(
            [captures[value].numpy() for value in ordered_layers]
        )
        states_by_condition[condition] = readout_states
        result_rows.append(
            {
                **common,
                "condition": condition,
                "status": "ok",
                "full_patch_hook_applications": applications,
                "condition_patch_delta_norm": patch_audit[
                    "condition_patch_delta_norms"
                ][condition],
                "readout_layers": ordered_layers,
                "readout_positions": list(readout_positions),
                **outcomes,
            }
        )
    reference = states_by_condition["self_patch"]
    for result in result_rows:
        metrics = stream_state_retention_metrics(
            reference,
            states_by_condition[str(result["condition"])],
            readout_layers=result["readout_layers"],
            readout_positions=result["readout_positions"],
            query_position=registry.query_position,
            count_bases=readout_bases,
        )
        result.update(metrics)
        result["downstream_progress_transport_magnitude"] = metrics[
            "downstream_item_progress_subspace_displacement_rms"
        ]
    return result_rows


def trace_patch_geometry_positions(
    registry: AnswerSourceRegistry,
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    geometry: str,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    """Right-align one registered multi-token donor/receiver geometry."""

    name = str(geometry)
    if name not in SUPPORTED_TRACE_PATCH_GEOMETRIES:
        raise ValueError(f"Unknown trace patch geometry: {name}")
    count = len(registry.trace_items)
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    if not 1 <= receiver <= count or not 1 <= donor <= count or donor == receiver:
        raise ValueError("Trace geometry needs two distinct valid occurrences")
    receiver_span = registry.trace_items[receiver - 1]
    donor_span = registry.trace_items[donor - 1]
    receiver_length = int(receiver_span[1]) - int(receiver_span[0])
    donor_length = int(donor_span[1]) - int(donor_span[0])
    requested_width: int | None = None
    capped_by_shorter_span = False
    if name == "endpoint":
        width = 1
    elif name.startswith("suffix_cap"):
        requested_width = int(name.removeprefix("suffix_cap"))
        width = min(requested_width, receiver_length, donor_length)
        capped_by_shorter_span = width < requested_width
    elif name.startswith("suffix"):
        width = int(name.removeprefix("suffix"))
        requested_width = width
        if min(receiver_length, donor_length) < width:
            raise ValueError(
                "not applicable: a trace item is shorter than the requested "
                f"{name} geometry"
            )
    else:
        if receiver_length != donor_length:
            raise ValueError(
                "not applicable: full_span requires exactly equal donor and "
                "receiver token lengths"
            )
        width = receiver_length
    receiver_positions = tuple(
        range(int(receiver_span[1]) - width, int(receiver_span[1]))
    )
    donor_positions = tuple(range(int(donor_span[1]) - width, int(donor_span[1])))
    if len(receiver_positions) != len(donor_positions) or not receiver_positions:
        raise RuntimeError("Trace geometry changed donor/receiver token alignment")
    audit = {
        "patch_geometry": name,
        "patch_position_alignment": "right_aligned_relative_token_index",
        "patch_token_count": len(receiver_positions),
        "requested_patch_token_count": requested_width,
        "patch_token_count_capped_by_shorter_span": capped_by_shorter_span,
        "receiver_span": list(receiver_span),
        "donor_span": list(donor_span),
        "receiver_span_token_count": receiver_length,
        "donor_span_token_count": donor_length,
        "donor_receiver_span_lengths_equal": bool(receiver_length == donor_length),
        "receiver_patch_positions": list(receiver_positions),
        "donor_patch_positions": list(donor_positions),
        "receiver_patch_positions_sha256": _sha256_json(receiver_positions),
        "donor_patch_positions_sha256": _sha256_json(donor_positions),
    }
    return receiver_positions, donor_positions, audit


def _full_state_patch_layers(
    *, source_layer: int, num_layers: int, layer_mode: str
) -> tuple[int, ...]:
    mode = str(layer_mode)
    if mode not in REGISTERED_TRACE_PATCH_LAYER_MODES:
        raise ValueError(f"Unknown trace patch layer mode: {mode}")
    source = int(source_layer)
    total = int(num_layers)
    if not 0 <= source < total - 1:
        raise ValueError("Full-state patching must leave a downstream decoder layer")
    if mode == "one_shot":
        return (source,)
    return tuple(range(source, total))


def build_html_aligned_uninformative_trace_encoding(
    encoding: NativeTraceEncoding,
    registry: AnswerSourceRegistry,
    tokenizer: Any,
    *,
    random_seed: int,
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    """Replace every parsed trace item with same-length prompt-background tokens.

    The replacement is deliberately token-level: no retokenization, padding, or
    interpolation is allowed.  Each control carrier is a contiguous window from
    the prompt outside registered prompt records and special tokens.  This
    mirrors the original HTML's all-needle-uninformative counterfactual while
    preserving every absolute position in the native trace.
    """

    original_ids = tuple(int(value) for value in encoding.input_ids)
    if len(original_ids) != int(registry.sequence_length):
        raise ValueError("Registry length disagrees with the trace encoding")
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    forbidden = set(registry.positions("prompt_records"))
    max_width = max(end - start for start, end in registry.trace_items)
    candidate_windows: list[tuple[int, ...]] = []
    candidate_starts: list[int] = []
    for start in range(1, int(registry.prompt_token_count) - max_width + 1):
        positions = tuple(range(start, start + max_width))
        if set(positions) & forbidden:
            continue
        window = tuple(original_ids[position] for position in positions)
        if any(token_id in special_ids for token_id in window):
            continue
        candidate_windows.append(window)
        candidate_starts.append(start)
    if not candidate_windows:
        raise ValueError(
            "No same-length ordinary prompt window can construct the "
            "uninformative trace"
        )

    control_ids = list(original_ids)
    selected_starts: list[int] = []
    changed_counts: list[int] = []
    for occurrence, (span_start, span_end) in enumerate(
        registry.trace_items, start=1
    ):
        width = int(span_end) - int(span_start)
        digest = hashlib.sha256(
            (
                f"{encoding.request_id}|{int(random_seed)}|{occurrence}|"
                f"{span_start}|{span_end}"
            ).encode("utf-8")
        ).digest()
        initial = int.from_bytes(digest[:8], "big") % len(candidate_windows)
        original_span = original_ids[int(span_start) : int(span_end)]
        selected_index = initial
        replacement = candidate_windows[selected_index][:width]
        for offset in range(len(candidate_windows)):
            candidate_index = (initial + offset) % len(candidate_windows)
            candidate = candidate_windows[candidate_index][:width]
            if candidate != original_span:
                selected_index = candidate_index
                replacement = candidate
                break
        if replacement == original_span:
            raise ValueError("Every ordinary control window equals one trace item")
        control_ids[int(span_start) : int(span_end)] = replacement
        selected_starts.append(candidate_starts[selected_index])
        changed_counts.append(
            sum(left != right for left, right in zip(original_span, replacement))
        )

    trace_positions = registry.positions("trace_items")
    changed_total = sum(
        original_ids[position] != control_ids[position]
        for position in trace_positions
    )
    if changed_total != sum(changed_counts) or changed_total <= 0:
        raise RuntimeError("Uninformative trace replacement audit failed")
    control = replace(encoding, input_ids=tuple(control_ids))
    audit = {
        "control_construction": "same_length_prompt_background_windows",
        "control_retokenized": False,
        "control_sequence_length_equal": True,
        "control_attention_mask_equal": bool(
            tuple(control.attention_mask) == tuple(encoding.attention_mask)
        ),
        "all_trace_items_replaced": True,
        "trace_item_count": len(registry.trace_items),
        "trace_item_token_count": len(trace_positions),
        "changed_trace_item_token_count": int(changed_total),
        "changed_trace_item_token_fraction": float(
            changed_total / len(trace_positions)
        ),
        "control_prompt_window_starts": selected_starts,
        "control_prompt_window_starts_sha256": _sha256_json(selected_starts),
        "control_input_ids_sha256": _sha256_json(control.input_ids),
        "outcome_fields_accessed": False,
    }
    return control, audit


@torch.inference_mode()
def run_html_aligned_terminal_span_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    layer: int,
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Run the original-HTML full-span ablation/restoration at the terminal item."""

    clean_encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    control_encoding, control_audit = (
        build_html_aligned_uninformative_trace_encoding(
            clean_encoding,
            registry,
            tokenizer,
            random_seed=int(random_seed),
        )
    )
    source_layer = int(layer)
    patch_layers = _full_state_patch_layers(
        source_layer=source_layer,
        num_layers=int(adapter.num_layers),
        layer_mode="cumulative_clamp",
    )
    target_occurrence = len(registry.trace_items)
    span_start, span_end = registry.trace_items[target_occurrence - 1]
    target_positions = tuple(range(int(span_start), int(span_end)))
    if not target_positions or max(target_positions) >= registry.query_position:
        raise RuntimeError("Terminal item span is not strictly before the answer query")

    _clean_logits, clean_capture = capture_post_block_states(
        model,
        adapter,
        clean_encoding,
        target_positions,
        layers=patch_layers,
    )
    _control_logits, control_capture = capture_post_block_states(
        model,
        adapter,
        control_encoding,
        target_positions,
        layers=patch_layers,
    )
    clean_states = {
        patch_layer: clean_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    control_states = {
        patch_layer: control_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    delta_norms = {
        str(patch_layer): float(
            torch.linalg.vector_norm(
                clean_states[patch_layer] - control_states[patch_layer]
            )
        )
        for patch_layer in patch_layers
    }
    conditions: dict[
        str, tuple[NativeTraceEncoding, Mapping[int, torch.Tensor] | None]
    ] = {
        "clean": (clean_encoding, None),
        "uninformative": (control_encoding, None),
        "clean_target_ablation": (clean_encoding, control_states),
        "uninformative_target_restore": (control_encoding, clean_states),
    }
    result_rows: list[dict[str, Any]] = []
    for condition in REGISTERED_HTML_ALIGNED_SPAN_CONDITIONS:
        active_encoding, replacements = conditions[condition]
        prefill, _captures, applications, realized_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                active_encoding,
                positions=target_positions,
                replacements=replacements,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(registry.query_position,),
            )
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            active_encoding,
            prefill,
            run_greedy=run_greedy,
            max_new_tokens=int(max_new_tokens),
        )
        result_rows.append(
            {
                "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                "experiment_id": "html_aligned_terminal_full_span",
                "condition": condition,
                "status": "ok",
                "request_id": clean_encoding.request_id,
                "model_label": clean_encoding.model_label,
                "seed": clean_encoding.seed,
                "dataset_split": clean_encoding.split,
                "gold_count": clean_encoding.count,
                "answer_site_id": answer_site_id,
                "target_occurrence": target_occurrence,
                "target_is_terminal": True,
                "patch_geometry": "full_item_span_same_position",
                "patch_layer_mode": "cumulative_clamp",
                "layer": source_layer,
                "patch_layers": list(patch_layers),
                "patch_layer_count": len(patch_layers),
                "patch_token_count": len(target_positions),
                "target_span": [int(span_start), int(span_end)],
                "target_positions_sha256": _sha256_json(target_positions),
                "clean_control_state_delta_norm_by_layer": delta_norms,
                "patch_hook_applications": {
                    str(key): value for key, value in sorted(applications.items())
                },
                "patch_realized_fro_norm_by_layer": {
                    str(key): value for key, value in sorted(realized_norms.items())
                },
                "final_answer_outcome_registered": True,
                "causal_claim_scope": (
                    "same_position_full_span_terminal_state_necessity_and_sufficiency"
                ),
                "registry_sha256": registry.to_dict()["registry_sha256"],
                **control_audit,
                **outcomes,
            }
        )
    return result_rows


def _prefill_with_layerwise_state_replacements(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    positions: Sequence[int],
    replacements: Mapping[int, torch.Tensor] | None,
    readout_layers: Sequence[int],
    readout_positions: Sequence[int],
) -> tuple[Any, dict[int, torch.Tensor], dict[int, int], dict[int, float]]:
    """Patch the same positions at one or many post-block layers."""

    selected_positions = tuple(int(value) for value in positions)
    if not selected_positions:
        raise ValueError("A full-state patch needs at least one receiver position")
    if len(set(selected_positions)) != len(selected_positions):
        raise ValueError("Full-state patch positions must be unique")
    replacement_map = {
        int(layer): torch.as_tensor(states).detach().float().cpu()
        for layer, states in (replacements or {}).items()
    }
    for layer, states in replacement_map.items():
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Full-state patch layer L{layer} is outside the decoder")
        if states.ndim != 2 or states.shape[0] != len(selected_positions):
            raise ValueError("Full-state replacement must have [positions, hidden]")
    active_readouts = tuple(sorted({int(value) for value in readout_layers}))
    if not active_readouts or any(
        not 0 <= layer < int(adapter.num_layers) for layer in active_readouts
    ):
        raise ValueError("Full-state patch readout layers are invalid")
    selected_readouts = tuple(int(value) for value in readout_positions)
    if not selected_readouts:
        raise ValueError("Full-state patch needs at least one downstream readout")

    applications = {layer: 0 for layer in replacement_map}
    realized_norms = {layer: 0.0 for layer in replacement_map}
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for layer, states in sorted(replacement_map.items()):

        def patch_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
            states: torch.Tensor = states,
        ) -> Any:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return output
            before = hidden[:, list(selected_positions), :]
            replacement = states.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
            if replacement.shape != before.shape:
                raise RuntimeError("Full-state replacement width disagrees with model")
            patched = hidden.clone()
            patched[:, list(selected_positions), :] = replacement
            realized_norms[layer] = float(
                torch.linalg.vector_norm(before.float() - replacement.float())
                .detach()
                .cpu()
            )
            applications[layer] += 1
            return _replace_output_tensor(output, patched)

        handles.append(adapter.layers[layer].register_forward_hook(patch_hook))
    # Register captures after patches so a readout on a clamped layer observes
    # the realized post-patch tensor rather than the pre-hook output.
    for layer in active_readouts:

        def readout_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return
            captures[layer] = (
                hidden[0, list(selected_readouts)].detach().float().cpu()
            )

        handles.append(adapter.layers[layer].register_forward_hook(readout_hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    violations = sorted(layer for layer, count in applications.items() if count != 1)
    if violations:
        raise RuntimeError(
            "Every full-state patch hook must apply exactly once; bad layers "
            f"{violations}"
        )
    missing_readouts = sorted(set(active_readouts) - set(captures))
    if missing_readouts:
        raise RuntimeError(f"Full-state patch missed readout layers {missing_readouts}")
    return prefill, captures, applications, realized_norms


def _prefix_with_layerwise_state_replacements(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    positions: Sequence[int],
    replacements: Mapping[int, torch.Tensor] | None,
) -> tuple[Any, dict[int, int], dict[int, float]]:
    """Patch a teacher-forced trace, stopping before the final answer query."""

    selected_positions = tuple(int(value) for value in positions)
    query = int(encoding.query_position)
    if not selected_positions or len(set(selected_positions)) != len(
        selected_positions
    ):
        raise ValueError("Full-state prefix positions must be unique and nonempty")
    if min(selected_positions) < 0 or max(selected_positions) >= query:
        raise ValueError("Full-state prefix positions must precede the answer query")
    replacement_map = {
        int(layer): torch.as_tensor(states).detach().float().cpu()
        for layer, states in (replacements or {}).items()
    }
    applications = {layer: 0 for layer in replacement_map}
    realized_norms = {layer: 0.0 for layer in replacement_map}
    handles = []
    for layer, states in sorted(replacement_map.items()):
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Full-state patch layer L{layer} is outside the decoder")
        if states.ndim != 2 or states.shape[0] != len(selected_positions):
            raise ValueError("Full-state replacement must have [positions, hidden]")

        def patch_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
            states: torch.Tensor = states,
        ) -> Any:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != query:
                return output
            before = hidden[:, list(selected_positions), :]
            replacement = states.to(
                device=hidden.device, dtype=hidden.dtype
            ).unsqueeze(0)
            if replacement.shape != before.shape:
                raise RuntimeError("Full-state replacement width disagrees with model")
            patched = hidden.clone()
            patched[:, list(selected_positions), :] = replacement
            realized_norms[layer] = float(
                torch.linalg.vector_norm(before.float() - replacement.float())
                .detach()
                .cpu()
            )
            applications[layer] += 1
            return _replace_output_tensor(output, patched)

        handles.append(adapter.layers[layer].register_forward_hook(patch_hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefix = _prefix_forward(
            model,
            adapter,
            input_ids[:, :query],
            attention_mask[:, :query],
        )
    finally:
        for handle in handles:
            handle.remove()
    violations = sorted(layer for layer, count in applications.items() if count != 1)
    if violations:
        raise RuntimeError(
            "Every full-state prefix patch hook must apply exactly once; bad "
            f"layers {violations}"
        )
    return prefix, applications, realized_norms


@torch.inference_mode()
def run_full_state_patch_answer_source_factorial_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    layer: int,
    geometry: str,
    layer_mode: str,
    patch_conditions: Sequence[str] = ("self_patch", "full_donor_patch"),
    mask_conditions: Sequence[str] = (
        "clean",
        "block_trace_items",
        "block_trace_items_matched_control",
        "block_prompt_records",
        "block_prompt_records_matched_control",
    ),
    answer_site_id: str = "answer_query_v3",
    mask_application: str = "answer_query_only",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Test whether answer-time source retrieval corrects a patched trace state.

    Positive prompt-mask interaction means blocking prompt evidence amplifies
    donor-state adoption (parallel/redundant recount).  Negative trace-mask
    interaction means blocking trace sources occludes donor-state adoption
    (serial readout/mediation through attention to the written trace state).
    """

    requested_patches = tuple(str(value) for value in patch_conditions)
    requested_masks = tuple(str(value) for value in mask_conditions)
    if len(set(requested_patches)) != len(requested_patches):
        raise ValueError("Full-state factorial patch conditions must be unique")
    if len(set(requested_masks)) != len(requested_masks):
        raise ValueError("Full-state factorial mask conditions must be unique")
    if not {"self_patch", "full_donor_patch"} <= set(requested_patches):
        raise ValueError("The factorial requires self_patch and full_donor_patch")
    unknown_patches = sorted(
        set(requested_patches) - set(REGISTERED_TRACE_FULL_STATE_CONDITIONS)
    )
    unknown_masks = sorted(set(requested_masks) - set(REGISTERED_MASK_CONDITIONS))
    if unknown_patches:
        raise ValueError(f"Unknown full-state patch conditions: {unknown_patches}")
    if unknown_masks:
        raise ValueError(f"Unknown answer source masks: {unknown_masks}")
    if mask_application not in {
        "answer_query_only",
        "answer_query_and_answer_tokens",
    }:
        raise ValueError(f"Unknown answer source-mask application: {mask_application}")

    source_layer = int(layer)
    patch_layers = _full_state_patch_layers(
        source_layer=source_layer,
        num_layers=int(adapter.num_layers),
        layer_mode=layer_mode,
    )
    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    receiver_positions, donor_positions, geometry_audit = (
        trace_patch_geometry_positions(
            registry,
            receiver_occurrence=receiver,
            donor_occurrence=donor,
            geometry=geometry,
        )
    )
    capture_positions = receiver_positions + donor_positions
    _last_logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        capture_positions,
        layers=patch_layers,
    )
    width = len(receiver_positions)
    receiver_states = {
        patch_layer: captured[patch_layer][:width].clone()
        for patch_layer in patch_layers
    }
    donor_states = {
        patch_layer: captured[patch_layer][width:].clone()
        for patch_layer in patch_layers
    }
    condition_replacements: dict[str, Mapping[int, torch.Tensor] | None] = {
        "clean": None,
        "self_patch": receiver_states,
        "full_donor_patch": donor_states,
    }
    receiver_token_ids = tuple(
        int(encoding.input_ids[position]) for position in receiver_positions
    )
    donor_token_ids = tuple(
        int(encoding.input_ids[position]) for position in donor_positions
    )
    token_matches = sum(
        left == right for left, right in zip(receiver_token_ids, donor_token_ids)
    )
    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "full_state_patch_answer_source_factorial",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "dataset_split": encoding.split,
        "gold_count": encoding.count,
        "answer_site_id": answer_site_id,
        "layer": source_layer,
        "patch_layer_mode": str(layer_mode),
        "patch_layers": list(patch_layers),
        "patch_layer_count": len(patch_layers),
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "donor_offset": donor - receiver,
        "receiver_is_terminal": bool(receiver == len(registry.trace_items)),
        "donor_is_terminal": bool(donor == len(registry.trace_items)),
        "later_trace_self_correction_possible": bool(
            max(receiver_positions) < max(end - 1 for _start, end in registry.trace_items)
        ),
        "receiver_patch_token_ids": list(receiver_token_ids),
        "donor_patch_token_ids": list(donor_token_ids),
        "donor_receiver_token_id_match_count": token_matches,
        "donor_receiver_token_id_match_fraction": token_matches / width,
        "donor_receiver_surface_tokens_identical": bool(token_matches == width),
        "causal_claim_scope": (
            "full_state_sufficiency_and_source_mediation_not_count_subspace_specificity"
        ),
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **geometry_audit,
    }
    rows: list[dict[str, Any]] = []
    for patch_condition in requested_patches:
        prefix, applications, realized_norms = (
            _prefix_with_layerwise_state_replacements(
                model,
                adapter,
                encoding,
                positions=receiver_positions,
                replacements=condition_replacements[patch_condition],
            )
        )
        aggregate_realized_norm = float(
            math.sqrt(sum(value * value for value in realized_norms.values()))
        )
        for mask_condition in requested_masks:
            mask, mask_audit = answer_source_mask(
                registry, condition=mask_condition
            )
            prefix_branch = clone_prefill_output_for_scoring(prefix)
            query_output = _query_forward_from_prefix(
                model,
                adapter,
                encoding,
                prefix_output=prefix_branch,
                query_attention_mask=mask,
            )
            scoring_encoding = encoding
            if mask_application == "answer_query_and_answer_tokens":
                scoring_encoding = replace(
                    encoding,
                    attention_mask=tuple(int(value) for value in mask[0].tolist()),
                )
            outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                scoring_encoding,
                query_output,
                run_greedy=run_greedy,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    **common,
                    "patch_condition": patch_condition,
                    "mask_condition": mask_condition,
                    "patch_hook_applications": {
                        str(key): value for key, value in sorted(applications.items())
                    },
                    "patch_realized_fro_norm_by_layer": {
                        str(key): value for key, value in sorted(realized_norms.items())
                    },
                    "patch_realized_aggregate_fro_norm": aggregate_realized_norm,
                    "mask_scope": mask_application,
                    "source_edge_mask_head_scope": "all_attention_heads",
                    "source_edge_mask_layer_scope": "all_decoder_layers",
                    **mask_audit,
                    **outcomes,
                }
            )
    expected = len(requested_patches) * len(requested_masks)
    if len(rows) != expected:
        raise RuntimeError(f"Full-state factorial emitted {len(rows)}, expected {expected}")
    return rows


def _prefill_with_terminal_source_and_relay_reset(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    source_positions: Sequence[int],
    source_replacements: Mapping[int, torch.Tensor],
    relay_layer: int,
    relay_positions: Sequence[int],
    relay_replacement: torch.Tensor | None,
) -> tuple[Any, dict[int, int], int, float]:
    """Clamp a terminal state before a later, optionally clean-reset relay."""

    sources = tuple(int(value) for value in source_positions)
    relays = tuple(int(value) for value in relay_positions)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Relay mediation needs unique terminal source positions")
    if not relays or len(set(relays)) != len(relays):
        raise ValueError("Relay mediation needs unique relay positions")
    replacement_map = {
        int(layer): torch.as_tensor(value).detach().float().cpu()
        for layer, value in source_replacements.items()
    }
    if not replacement_map:
        raise ValueError("Relay mediation needs at least one source patch layer")
    if max(replacement_map) >= int(relay_layer):
        raise ValueError("Every terminal source patch layer must precede the relay")
    if min(relays) <= max(sources):
        raise ValueError("Every relay position must follow the terminal source patch")
    if max(relays) > int(encoding.query_position):
        raise ValueError("A relay position lies after the answer query")

    source_applications = {layer: 0 for layer in replacement_map}
    relay_applications = 0
    relay_realized_norm = 0.0
    handles = []
    for layer, states in sorted(replacement_map.items()):
        if states.ndim != 2 or states.shape[0] != len(sources):
            raise ValueError("Source replacement must be [positions, hidden]")

        def source_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
            states: torch.Tensor = states,
        ) -> Any:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return output
            replacement = states.to(
                device=hidden.device, dtype=hidden.dtype
            ).unsqueeze(0)
            before = hidden[:, list(sources), :]
            if replacement.shape != before.shape:
                raise RuntimeError("Terminal source replacement shape changed")
            patched = hidden.clone()
            patched[:, list(sources), :] = replacement
            source_applications[layer] += 1
            return _replace_output_tensor(output, patched)

        handles.append(adapter.layers[layer].register_forward_hook(source_hook))

    if relay_replacement is not None:
        clean_relay = torch.as_tensor(relay_replacement).detach().float().cpu()
        if clean_relay.ndim != 2 or clean_relay.shape[0] != len(relays):
            raise ValueError("Relay replacement must be [positions, hidden]")

        def relay_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            nonlocal relay_applications, relay_realized_norm
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return output
            replacement = clean_relay.to(
                device=hidden.device, dtype=hidden.dtype
            ).unsqueeze(0)
            before = hidden[:, list(relays), :]
            if replacement.shape != before.shape:
                raise RuntimeError("Relay replacement shape changed")
            patched = hidden.clone()
            patched[:, list(relays), :] = replacement
            relay_realized_norm = float(
                torch.linalg.vector_norm(before.float() - replacement.float())
                .detach()
                .cpu()
            )
            relay_applications += 1
            return _replace_output_tensor(output, patched)

        handles.append(
            adapter.layers[int(relay_layer)].register_forward_hook(relay_hook)
        )
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    bad_sources = {
        layer: count for layer, count in source_applications.items() if count != 1
    }
    if bad_sources:
        raise RuntimeError(f"Terminal source hooks did not apply once: {bad_sources}")
    expected_relay = 0 if relay_replacement is None else 1
    if relay_applications != expected_relay:
        raise RuntimeError(
            f"Relay reset applied {relay_applications}, expected {expected_relay}"
        )
    return prefill, source_applications, relay_applications, relay_realized_norm


@torch.inference_mode()
def run_terminal_state_relay_reset_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    source_layer: int,
    relay_layer: int,
    geometry: str = "suffix8",
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Test terminal-state propagation through a frozen pre-answer relay.

    The terminal state is clamped from ``source_layer`` through the layer just
    before ``relay_layer``.  At the frozen relay layer, either the answer query
    alone or the complete post-terminal suffix is reset to its clean receiver
    state.  Source clamping never continues at or after the relay layer.
    """

    source = int(source_layer)
    relay = int(relay_layer)
    if not 0 <= source < relay < int(adapter.num_layers):
        raise ValueError("Relay layers must satisfy 0 <= source < relay < depth")
    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    if receiver != len(registry.trace_items):
        raise ValueError("Relay mediation is registered only at the terminal item")
    receiver_positions, donor_positions, geometry_audit = (
        trace_patch_geometry_positions(
            registry,
            receiver_occurrence=receiver,
            donor_occurrence=donor,
            geometry=geometry,
        )
    )
    terminal_end = int(registry.trace_items[-1][1])
    query = int(registry.query_position)
    if terminal_end > query:
        raise RuntimeError("Terminal trace item ends after the answer query")
    relay_positions = {
        "natural_relay": (query,),
        "answer_query_clean_reset": (query,),
        "post_terminal_suffix_clean_reset": tuple(range(terminal_end, query + 1)),
    }
    if not relay_positions["post_terminal_suffix_clean_reset"]:
        raise ValueError("No post-terminal relay suffix exists")

    source_layers = tuple(range(source, relay))
    all_capture_positions = tuple(
        dict.fromkeys(
            receiver_positions
            + donor_positions
            + relay_positions["post_terminal_suffix_clean_reset"]
        )
    )
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        all_capture_positions,
        layers=source_layers + (relay,),
    )
    position_lookup = {
        int(position): index for index, position in enumerate(all_capture_positions)
    }

    def states(layer: int, positions: Sequence[int]) -> torch.Tensor:
        return captured[int(layer)][
            [position_lookup[int(position)] for position in positions]
        ].clone()

    self_sources = {
        layer: states(layer, receiver_positions) for layer in source_layers
    }
    donor_sources = {
        layer: states(layer, donor_positions) for layer in source_layers
    }
    clean_relays = {
        condition: states(relay, positions)
        for condition, positions in relay_positions.items()
        if condition != "natural_relay"
    }
    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "terminal_state_pre_answer_relay_mediation",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "dataset_split": encoding.split,
        "gold_count": encoding.count,
        "answer_site_id": answer_site_id,
        "source_layer": source,
        "source_patch_layers": list(source_layers),
        "source_patch_last_layer": relay - 1,
        "source_patch_stops_before_relay": True,
        "relay_layer": relay,
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "donor_offset": donor - receiver,
        "receiver_is_terminal": True,
        "patch_geometry": str(geometry),
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **geometry_audit,
    }
    rows: list[dict[str, Any]] = []
    for source_condition, source_states in (
        ("self_patch", self_sources),
        ("full_donor_patch", donor_sources),
    ):
        for relay_condition in REGISTERED_RELAY_RESET_CONDITIONS:
            active_relay_positions = relay_positions[relay_condition]
            replacement = clean_relays.get(relay_condition)
            prefill, source_applications, relay_applications, relay_norm = (
                _prefill_with_terminal_source_and_relay_reset(
                    model,
                    adapter,
                    encoding,
                    source_positions=receiver_positions,
                    source_replacements=source_states,
                    relay_layer=relay,
                    relay_positions=active_relay_positions,
                    relay_replacement=replacement,
                )
            )
            outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                encoding,
                prefill,
                run_greedy=run_greedy,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    **common,
                    "source_condition": source_condition,
                    "relay_condition": relay_condition,
                    "relay_positions": list(active_relay_positions),
                    "relay_position_count": len(active_relay_positions),
                    "relay_includes_answer_query": query
                    in set(active_relay_positions),
                    "relay_reset_hook_applications": relay_applications,
                    "relay_reset_realized_fro_norm": relay_norm,
                    "source_patch_hook_applications": {
                        str(key): int(value)
                        for key, value in sorted(source_applications.items())
                    },
                    "status": "ok",
                    **outcomes,
                }
            )
    if len(rows) != 2 * len(REGISTERED_RELAY_RESET_CONDITIONS):
        raise RuntimeError("Relay mediation did not emit the registered 2x3 panel")
    return rows


@torch.inference_mode()
def run_terminal_state_complementary_readout_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    source_layer: int,
    relay_layer: int,
    geometry: str,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Cross Qwen's residual relay with its direct trace-source reread.

    The terminal patch is clamped strictly before ``relay_layer``.  The
    post-terminal residual suffix can then be reset to its clean state at the
    relay, while an independent all-head attention mask blocks direct reads of
    trace-item keys at the answer query and all scored/generated answer tokens.
    Thus the joint arm removes the two routes without changing the trace prefix.
    """

    source = int(source_layer)
    relay = int(relay_layer)
    if not 0 <= source < relay < int(adapter.num_layers):
        raise ValueError("Complementary layers must satisfy source < relay < depth")
    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    if receiver != len(registry.trace_items):
        raise ValueError("Complementary readout is registered only at terminal item")
    receiver_positions, donor_positions, geometry_audit = (
        trace_patch_geometry_positions(
            registry,
            receiver_occurrence=receiver,
            donor_occurrence=donor,
            geometry=geometry,
        )
    )
    query = int(registry.query_position)
    terminal_end = int(registry.trace_items[-1][1])
    relay_positions = tuple(range(terminal_end, query + 1))
    if not relay_positions or relay_positions[-1] != query:
        raise ValueError("No complete post-terminal suffix reaches answer query")
    prefix_relay_positions = relay_positions[:-1]
    source_layers = tuple(range(source, relay))
    capture_positions = tuple(
        dict.fromkeys(
            receiver_positions + donor_positions + relay_positions
        )
    )
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        capture_positions,
        layers=source_layers + (relay,),
    )
    lookup = {
        int(position): index for index, position in enumerate(capture_positions)
    }

    def states(layer: int, positions: Sequence[int]) -> torch.Tensor:
        return captured[int(layer)][
            [lookup[int(position)] for position in positions]
        ].clone()

    condition_sources = {
        "self_patch": {
            layer: states(layer, receiver_positions) for layer in source_layers
        },
        "full_donor_patch": {
            layer: states(layer, donor_positions) for layer in source_layers
        },
    }
    clean_prefix_relay = states(relay, prefix_relay_positions)
    clean_query_relay = states(relay, (query,))
    input_ids, attention_mask = _encoding_tensors(model, encoding)

    def patched_prefix(
        replacements: Mapping[int, torch.Tensor], *, reset_relay: bool
    ) -> tuple[Any, dict[int, int], int, float]:
        source_applications = {int(layer): 0 for layer in replacements}
        relay_applications = 0
        relay_norm = 0.0
        handles = []
        for active_layer, replacement_states in sorted(replacements.items()):

            def source_hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                active_layer: int = int(active_layer),
                replacement_states: torch.Tensor = replacement_states,
            ) -> Any:
                hidden = _tensor_from_output(output)
                if hidden.ndim != 3 or hidden.shape[1] != query:
                    return output
                replacement = replacement_states.to(
                    device=hidden.device, dtype=hidden.dtype
                ).unsqueeze(0)
                before = hidden[:, list(receiver_positions), :]
                if replacement.shape != before.shape:
                    raise RuntimeError("Complementary source replacement shape changed")
                patched = hidden.clone()
                patched[:, list(receiver_positions), :] = replacement
                source_applications[active_layer] += 1
                return _replace_output_tensor(output, patched)

            handles.append(
                adapter.layers[int(active_layer)].register_forward_hook(source_hook)
            )
        if reset_relay and prefix_relay_positions:

            def prefix_relay_hook(
                _module: Any, _args: tuple[Any, ...], output: Any
            ) -> Any:
                nonlocal relay_applications, relay_norm
                hidden = _tensor_from_output(output)
                if hidden.ndim != 3 or hidden.shape[1] != query:
                    return output
                replacement = clean_prefix_relay.to(
                    device=hidden.device, dtype=hidden.dtype
                ).unsqueeze(0)
                before = hidden[:, list(prefix_relay_positions), :]
                if replacement.shape != before.shape:
                    raise RuntimeError("Complementary prefix relay shape changed")
                patched = hidden.clone()
                patched[:, list(prefix_relay_positions), :] = replacement
                relay_norm = float(
                    torch.linalg.vector_norm(before.float() - replacement.float())
                    .detach()
                    .cpu()
                )
                relay_applications += 1
                return _replace_output_tensor(output, patched)

            handles.append(
                adapter.layers[relay].register_forward_hook(prefix_relay_hook)
            )
        try:
            prefix = _prefix_forward(
                model,
                adapter,
                input_ids[:, :query],
                attention_mask[:, :query],
            )
        finally:
            for handle in handles:
                handle.remove()
        bad = {
            layer: count
            for layer, count in source_applications.items()
            if count != 1
        }
        if bad:
            raise RuntimeError(f"Complementary source hooks did not apply once: {bad}")
        expected_relay = int(bool(reset_relay and prefix_relay_positions))
        if relay_applications != expected_relay:
            raise RuntimeError("Complementary prefix relay hook count changed")
        return prefix, source_applications, relay_applications, relay_norm

    def query_branch(
        prefix: Any,
        *,
        query_mask: torch.Tensor,
        reset_relay: bool,
    ) -> tuple[Any, int, float]:
        applications = 0
        realized_norm = 0.0
        handle = None
        if reset_relay:

            def query_relay_hook(
                _module: Any, _args: tuple[Any, ...], output: Any
            ) -> Any:
                nonlocal applications, realized_norm
                hidden = _tensor_from_output(output)
                if hidden.ndim != 3 or hidden.shape[1] != 1:
                    return output
                replacement = clean_query_relay.to(
                    device=hidden.device, dtype=hidden.dtype
                ).unsqueeze(0)
                before = hidden[:, :1, :]
                if replacement.shape != before.shape:
                    raise RuntimeError("Complementary query relay shape changed")
                patched = hidden.clone()
                patched[:, :1, :] = replacement
                realized_norm = float(
                    torch.linalg.vector_norm(before.float() - replacement.float())
                    .detach()
                    .cpu()
                )
                applications += 1
                return _replace_output_tensor(output, patched)

            handle = adapter.layers[relay].register_forward_hook(query_relay_hook)
        try:
            output = _query_forward_from_prefix(
                model,
                adapter,
                encoding,
                prefix_output=prefix,
                query_attention_mask=query_mask,
            )
        finally:
            if handle is not None:
                handle.remove()
        if applications != int(reset_relay):
            raise RuntimeError("Complementary query relay hook count changed")
        return output, applications, realized_norm

    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "terminal_state_complementary_readout",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "dataset_split": encoding.split,
        "gold_count": encoding.count,
        "answer_site_id": answer_site_id,
        "source_layer": source,
        "source_patch_layers": list(source_layers),
        "source_patch_stops_before_relay": True,
        "relay_layer": relay,
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "donor_offset": donor - receiver,
        "receiver_is_terminal": True,
        "patch_geometry": str(geometry),
        "relay_positions": list(relay_positions),
        "relay_position_count": len(relay_positions),
        "mask_application": "answer_query_and_answer_tokens",
        "source_edge_mask_head_scope": "all_attention_heads",
        "source_edge_mask_layer_scope": "all_decoder_layers",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **geometry_audit,
    }
    rows: list[dict[str, Any]] = []
    for patch_condition, source_states in condition_sources.items():
        for relay_condition in REGISTERED_COMPLEMENTARY_RELAY_CONDITIONS:
            reset_relay = relay_condition == "post_terminal_suffix_clean_reset"
            prefix, source_applications, prefix_relay_applications, prefix_norm = (
                patched_prefix(source_states, reset_relay=reset_relay)
            )
            for mask_condition in REGISTERED_COMPLEMENTARY_SOURCE_CONDITIONS:
                mask, mask_audit = answer_source_mask(
                    registry, condition=mask_condition
                )
                branch_prefix = clone_prefill_output_for_scoring(prefix)
                query_output, query_relay_applications, query_norm = query_branch(
                    branch_prefix,
                    query_mask=mask,
                    reset_relay=reset_relay,
                )
                scoring_encoding = replace(
                    encoding,
                    attention_mask=tuple(int(value) for value in mask[0].tolist()),
                )
                outcomes = _score_and_generate_prefill(
                    model,
                    tokenizer,
                    scoring_encoding,
                    query_output,
                    run_greedy=run_greedy,
                    max_new_tokens=max_new_tokens,
                )
                rows.append(
                    {
                        **common,
                        "patch_condition": patch_condition,
                        "relay_condition": relay_condition,
                        "mask_condition": mask_condition,
                        "source_patch_hook_applications": {
                            str(key): int(value)
                            for key, value in sorted(source_applications.items())
                        },
                        "prefix_relay_reset_hook_applications": int(
                            prefix_relay_applications
                        ),
                        "query_relay_reset_hook_applications": int(
                            query_relay_applications
                        ),
                        "relay_reset_realized_fro_norm": float(
                            math.sqrt(prefix_norm * prefix_norm + query_norm * query_norm)
                        ),
                        "status": "ok",
                        **mask_audit,
                        **outcomes,
                    }
                )
    expected = (
        2
        * len(REGISTERED_COMPLEMENTARY_RELAY_CONDITIONS)
        * len(REGISTERED_COMPLEMENTARY_SOURCE_CONDITIONS)
    )
    if len(rows) != expected:
        raise RuntimeError(
            f"Complementary readout emitted {len(rows)} rows, expected {expected}"
        )
    return rows


@torch.inference_mode()
def run_full_state_patch_head_readout_factorial_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    layer: int,
    geometry: str,
    layer_mode: str,
    head_arms: Sequence[Mapping[str, Any]],
    patch_conditions: Sequence[str] = ("self_patch", "full_donor_patch"),
    source_group: str = "trace_items",
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Cross a written terminal state with frozen answer-query head ablations.

    The teacher-forced trace prefix is built once for each patch arm.  Cloned
    caches then branch into intact, selected-bank, and layer-matched-random
    final-query readouts.  Head hooks run strictly after the terminal state has
    been written, so a patch-by-head interaction has the temporal order needed
    for a serial ``state -> heads -> answer`` interpretation.
    """

    requested_patches = tuple(str(value) for value in patch_conditions)
    if set(requested_patches) != {"self_patch", "full_donor_patch"} or len(
        requested_patches
    ) != 2:
        raise ValueError(
            "Serial head-readout factorial requires exactly self_patch and "
            "full_donor_patch"
        )
    if str(source_group) != "trace_items":
        raise ValueError("Serial head readout is registered only for trace_items")

    normalized_arms: list[dict[str, Any]] = []
    seen_arm_ids: set[tuple[str, int]] = set()
    for raw_arm in head_arms:
        normalized_arm = _normalize_head_readout_arm(adapter, raw_arm)
        condition = str(normalized_arm["condition"])
        repeat = int(normalized_arm["repeat"])
        arm_id = (condition, repeat)
        if arm_id in seen_arm_ids:
            raise ValueError(f"Duplicate head-readout arm: {arm_id}")
        seen_arm_ids.add(arm_id)
        normalized_arms.append(normalized_arm)

    condition_counts = {
        condition: sum(arm["condition"] == condition for arm in normalized_arms)
        for condition in REGISTERED_HEAD_READOUT_CONDITIONS
    }
    if condition_counts["clean"] != 1 or condition_counts["selected_bank"] != 1:
        raise ValueError("Serial factorial needs one intact and one selected head arm")
    if condition_counts["layer_matched_random"] < 1:
        raise ValueError("Serial factorial needs at least one matched-random head arm")
    selected_arm = next(
        arm for arm in normalized_arms if arm["condition"] == "selected_bank"
    )
    selected_heads = set(selected_arm["heads"])

    def layer_counts(heads: Sequence[tuple[int, int]]) -> dict[int, int]:
        counts: dict[int, int] = {}
        for layer_index, _head in heads:
            counts[layer_index] = counts.get(layer_index, 0) + 1
        return counts

    selected_layers = layer_counts(selected_arm["heads"])
    random_hashes: set[str] = set()
    for arm in normalized_arms:
        if arm["condition"] != "layer_matched_random":
            continue
        if layer_counts(arm["heads"]) != selected_layers:
            raise ValueError("Random head bank is not layer-matched to selected bank")
        if set(arm["heads"]) == selected_heads:
            raise ValueError("Random head bank is identical to the selected bank")
        if arm["canonical_bank_sha256"] in random_hashes:
            raise ValueError("Matched-random head banks must be distinct")
        random_hashes.add(arm["canonical_bank_sha256"])

    source_layer = int(layer)
    patch_layers = _full_state_patch_layers(
        source_layer=source_layer,
        num_layers=int(adapter.num_layers),
        layer_mode=layer_mode,
    )
    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    receiver_positions, donor_positions, geometry_audit = (
        trace_patch_geometry_positions(
            registry,
            receiver_occurrence=receiver,
            donor_occurrence=donor,
            geometry=geometry,
        )
    )
    capture_positions = receiver_positions + donor_positions
    _last_logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        capture_positions,
        layers=patch_layers,
    )
    width = len(receiver_positions)
    receiver_states = {
        patch_layer: captured[patch_layer][:width].clone()
        for patch_layer in patch_layers
    }
    donor_states = {
        patch_layer: captured[patch_layer][width:].clone()
        for patch_layer in patch_layers
    }
    condition_replacements = {
        "self_patch": receiver_states,
        "full_donor_patch": donor_states,
    }
    receiver_token_ids = tuple(
        int(encoding.input_ids[position]) for position in receiver_positions
    )
    donor_token_ids = tuple(
        int(encoding.input_ids[position]) for position in donor_positions
    )
    token_matches = sum(
        left == right for left, right in zip(receiver_token_ids, donor_token_ids)
    )
    _input_ids, attention_mask = _encoding_tensors(model, encoding)
    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "full_state_patch_head_readout_factorial",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "dataset_split": encoding.split,
        "gold_count": encoding.count,
        "answer_site_id": answer_site_id,
        "source_group": str(source_group),
        "layer": source_layer,
        "patch_layer_mode": str(layer_mode),
        "patch_layers": list(patch_layers),
        "patch_layer_count": len(patch_layers),
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "donor_offset": donor - receiver,
        "receiver_is_terminal": bool(receiver == len(registry.trace_items)),
        "donor_is_terminal": bool(donor == len(registry.trace_items)),
        "later_trace_self_correction_possible": bool(
            max(receiver_positions)
            < max(end - 1 for _start, end in registry.trace_items)
        ),
        "receiver_patch_token_ids": list(receiver_token_ids),
        "donor_patch_token_ids": list(donor_token_ids),
        "donor_receiver_token_id_match_count": token_matches,
        "donor_receiver_token_id_match_fraction": token_matches / width,
        "readout_intervention_scope": "answer_query_only",
        "causal_claim_scope": "serial_trace_state_to_frozen_heads_to_answer",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **geometry_audit,
    }
    rows: list[dict[str, Any]] = []
    for patch_condition in requested_patches:
        prefix, patch_applications, realized_norms = (
            _prefix_with_layerwise_state_replacements(
                model,
                adapter,
                encoding,
                positions=receiver_positions,
                replacements=condition_replacements[patch_condition],
            )
        )
        aggregate_realized_norm = float(
            math.sqrt(sum(value * value for value in realized_norms.values()))
        )
        for arm in normalized_arms:
            prefix_branch = clone_prefill_output_for_scoring(prefix)
            query_output, realized_heads, head_applications = (
                _query_from_prefix_with_head_ablation(
                    model,
                    adapter,
                    encoding,
                    prefix_output=prefix_branch,
                    query_attention_mask=attention_mask,
                    heads=arm["heads"],
                )
            )
            outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                encoding,
                query_output,
                run_greedy=run_greedy,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    **common,
                    "patch_condition": patch_condition,
                    "head_condition": arm["condition"],
                    "head_repeat": int(arm["repeat"]),
                    "heads": realized_heads,
                    "bank_size": int(arm["bank_size"]),
                    "bank_sha256": str(arm["bank_sha256"]),
                    "selected_bank_overlap_count": len(
                        selected_heads & set(arm["heads"])
                    ),
                    "selected_bank_overlap_fraction": (
                        len(selected_heads & set(arm["heads"]))
                        / len(selected_heads)
                    ),
                    "patch_hook_applications": {
                        str(key): value
                        for key, value in sorted(patch_applications.items())
                    },
                    "patch_realized_fro_norm_by_layer": {
                        str(key): value
                        for key, value in sorted(realized_norms.items())
                    },
                    "patch_realized_aggregate_fro_norm": aggregate_realized_norm,
                    "head_hook_applications": {
                        str(key): value
                        for key, value in sorted(head_applications.items())
                    },
                    **outcomes,
                }
            )
    expected = len(requested_patches) * len(normalized_arms)
    if len(rows) != expected:
        raise RuntimeError(
            f"Serial head-readout factorial emitted {len(rows)}, expected {expected}"
        )
    return rows


@torch.inference_mode()
def run_trace_full_state_patch_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    layer: int,
    geometry: str,
    layer_mode: str,
    readout_layers: Sequence[int],
    readout_bases: Mapping[int, np.ndarray],
    conditions: Sequence[str] = REGISTERED_TRACE_FULL_STATE_CONDITIONS,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Patch a multi-token full hidden-state tensor within one clean trace.

    This is a sufficiency assay.  Because donor and receiver items can contain
    different surface tokens, it does not by itself identify a count-specific
    subspace.  ``cumulative_clamp`` mirrors the referenced HTML by writing the
    donor tensor at every post-block layer from ``layer`` through the last;
    ``one_shot`` writes it only once and lets all later computation evolve.
    """

    requested = tuple(str(value) for value in conditions)
    if len(set(requested)) != len(requested):
        raise ValueError("Full-state trace patch conditions must be unique")
    unknown = sorted(set(requested) - set(REGISTERED_TRACE_FULL_STATE_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown full-state trace patch conditions: {unknown}")
    if not {"clean", "self_patch"} <= set(requested):
        raise ValueError("Full-state patching requires clean and self_patch controls")

    source_layer = int(layer)
    patch_layers = _full_state_patch_layers(
        source_layer=source_layer,
        num_layers=int(adapter.num_layers),
        layer_mode=layer_mode,
    )
    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    receiver_positions, donor_positions, geometry_audit = (
        trace_patch_geometry_positions(
            registry,
            receiver_occurrence=receiver,
            donor_occurrence=donor,
            geometry=geometry,
        )
    )
    capture_positions = receiver_positions + donor_positions
    _last_logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        capture_positions,
        layers=patch_layers,
    )
    width = len(receiver_positions)
    receiver_states = {
        patch_layer: captured[patch_layer][:width].clone()
        for patch_layer in patch_layers
    }
    donor_states = {
        patch_layer: captured[patch_layer][width:].clone()
        for patch_layer in patch_layers
    }
    if any(
        receiver_states[value].shape != donor_states[value].shape
        for value in patch_layers
    ):
        raise RuntimeError("Captured donor and receiver tensors have different shapes")

    active_readout_layers = tuple(sorted({int(value) for value in readout_layers}))
    if not active_readout_layers:
        active_readout_layers = (source_layer + 1,)
    if any(value <= source_layer for value in active_readout_layers):
        raise ValueError("Full-state readout layers must be after the source layer")
    missing_bases = sorted(set(active_readout_layers) - set(readout_bases))
    if missing_bases:
        raise ValueError(f"Full-state readout bases are missing {missing_bases}")
    endpoints = tuple(end - 1 for _start, end in registry.trace_items)
    downstream_endpoints = tuple(
        position for position in endpoints if position > max(receiver_positions)
    )
    readout_positions = downstream_endpoints + (registry.query_position,)
    if any(position >= registry.query_position for position in receiver_positions):
        raise RuntimeError("A full-state receiver is not before the answer query")

    receiver_token_ids = tuple(int(encoding.input_ids[pos]) for pos in receiver_positions)
    donor_token_ids = tuple(int(encoding.input_ids[pos]) for pos in donor_positions)
    token_matches = sum(
        left == right for left, right in zip(receiver_token_ids, donor_token_ids)
    )
    marker_positions = set(registry.positions("trace_markers"))
    full_delta_norms = {
        str(patch_layer): float(
            torch.linalg.vector_norm(
                donor_states[patch_layer] - receiver_states[patch_layer]
            )
        )
        for patch_layer in patch_layers
    }
    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "trace_full_state_geometry_patching",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "dataset_split": encoding.split,
        "gold_count": encoding.count,
        "answer_site_id": answer_site_id,
        "layer": source_layer,
        "patch_layer_mode": str(layer_mode),
        "patch_layers": list(patch_layers),
        "patch_layer_count": len(patch_layers),
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "donor_offset": donor - receiver,
        "donor_direction": (
            "past_to_later_receiver"
            if donor < receiver
            else "future_to_earlier_receiver"
        ),
        "future_donor_is_counterfactual_not_natural_stream": bool(donor > receiver),
        "receiver_is_terminal": bool(receiver == encoding.count),
        "donor_is_terminal": bool(donor == encoding.count),
        "downstream_trace_item_count": len(downstream_endpoints),
        "later_trace_self_correction_possible": bool(downstream_endpoints),
        "final_answer_outcome_registered": True,
        "local_next_city_outcome_registered": False,
        "receiver_patch_marker_overlap_count": len(
            set(receiver_positions) & marker_positions
        ),
        "donor_patch_marker_overlap_count": len(set(donor_positions) & marker_positions),
        "receiver_patch_token_ids": list(receiver_token_ids),
        "donor_patch_token_ids": list(donor_token_ids),
        "donor_receiver_token_id_match_count": token_matches,
        "donor_receiver_token_id_match_fraction": token_matches / width,
        "donor_receiver_surface_tokens_identical": bool(token_matches == width),
        "causal_claim_scope": "full_state_sufficiency_not_count_specificity",
        "full_donor_delta_norm_by_layer": full_delta_norms,
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **geometry_audit,
    }
    condition_replacements: dict[str, Mapping[int, torch.Tensor] | None] = {
        "clean": None,
        "self_patch": receiver_states,
        "full_donor_patch": donor_states,
    }
    result_rows: list[dict[str, Any]] = []
    states_by_condition: dict[str, np.ndarray] = {}
    for condition in requested:
        prefill, captures, applications, realized_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                encoding,
                positions=receiver_positions,
                replacements=condition_replacements[condition],
                readout_layers=active_readout_layers,
                readout_positions=readout_positions,
            )
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            encoding,
            prefill,
            run_greedy=run_greedy,
            max_new_tokens=max_new_tokens,
        )
        ordered_layers = sorted(captures)
        readout_states = np.stack(
            [captures[value].numpy() for value in ordered_layers]
        )
        states_by_condition[condition] = readout_states
        aggregate_realized_norm = float(
            math.sqrt(sum(value * value for value in realized_norms.values()))
        )
        result_rows.append(
            {
                **common,
                "condition": condition,
                "status": "ok",
                "patch_hook_applications": {
                    str(key): value for key, value in sorted(applications.items())
                },
                "patch_realized_fro_norm_by_layer": {
                    str(key): value for key, value in sorted(realized_norms.items())
                },
                "patch_realized_aggregate_fro_norm": aggregate_realized_norm,
                "readout_layers": ordered_layers,
                "readout_positions": list(readout_positions),
                **outcomes,
            }
        )
    reference = states_by_condition["self_patch"]
    for result in result_rows:
        metrics = stream_state_retention_metrics(
            reference,
            states_by_condition[str(result["condition"])],
            readout_layers=result["readout_layers"],
            readout_positions=result["readout_positions"],
            query_position=registry.query_position,
            count_bases=readout_bases,
        )
        result.update(metrics)
        result["downstream_progress_transport_magnitude"] = metrics[
            "downstream_item_progress_subspace_displacement_rms"
        ]
    return result_rows


def _stream_source_positions(
    registry: AnswerSourceRegistry,
    *,
    scope: str,
    occurrence: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    item_spans = registry.trace_items
    endpoints = tuple(end - 1 for _start, end in item_spans)
    occurrences = tuple(range(1, len(endpoints) + 1))
    if scope == "all":
        if occurrence is not None:
            raise ValueError("occurrence is not used when scope=all")
        return endpoints, occurrences
    if occurrence is None or not 1 <= int(occurrence) <= len(endpoints):
        raise ValueError("A valid 1-based occurrence is required")
    index = int(occurrence) - 1
    if scope == "occurrence":
        return (endpoints[index],), (int(occurrence),)
    if scope == "prefix":
        return endpoints[: index + 1], occurrences[: index + 1]
    raise ValueError("scope must be occurrence, prefix, or all")


def _running_state_transform(
    *,
    condition: str,
    center: torch.Tensor,
    basis: torch.Tensor,
    random_seed: int,
    audit: dict[str, Any],
):
    if condition not in REGISTERED_STREAM_CONDITIONS[1:]:
        raise ValueError(f"Unsupported stream transform: {condition}")
    center_cpu = center.detach().float().cpu()
    basis_cpu = basis.detach().float().cpu()

    def transform(selected: torch.Tensor) -> torch.Tensor:
        if selected.ndim != 3:
            raise ValueError("Stream states must have shape [batch, positions, hidden]")
        active_center = center_cpu.to(selected.device).view(1, 1, -1)
        active_basis = basis_cpu.to(selected.device)
        if active_center.shape[-1] != selected.shape[-1]:
            raise ValueError("Stream center width disagrees with the model")
        if active_basis.shape[0] != selected.shape[-1]:
            raise ValueError("Stream basis width disagrees with the model")
        target = ((selected.float() - active_center) @ active_basis) @ active_basis.T
        target_replacement, realized_target = _realized_replacement(selected, target)
        target_norm = torch.linalg.vector_norm(realized_target)
        if condition == "aligned_running_state_removal":
            replacement = target_replacement
            realized = realized_target
        else:
            control = norm_matched_orthogonal_delta(
                target.detach().float().cpu(),
                basis_cpu,
                seed=int(random_seed),
            ).to(selected.device)
            replacement, realized = _closest_realized_norm_replacement(
                selected, control, target_norm
            )
        realized_cpu = realized.detach().float().cpu()
        audit.update(
            {
                "applications": int(audit.get("applications", 0)) + 1,
                "target_removed_fro_norm": float(target_norm.detach().cpu()),
                "removed_fro_norm": float(torch.linalg.vector_norm(realized_cpu)),
                "norm_ratio": float(
                    torch.linalg.vector_norm(realized_cpu)
                    / max(float(target_norm.detach().cpu()), 1e-12)
                ),
                "removed_count_subspace_max_abs_cosine": float(
                    torch.max(
                        torch.abs(
                            realized_cpu.reshape(-1, realized_cpu.shape[-1]) @ basis_cpu
                        )
                    )
                    / max(float(torch.linalg.vector_norm(realized_cpu)), 1e-12)
                ),
            }
        )
        return replacement

    return transform


def _full_prefill_with_stream_transform(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    source_layer: int,
    source_positions: Sequence[int],
    transform: Any | None,
    readout_layers: Sequence[int],
    readout_positions: Sequence[int],
) -> tuple[Any, dict[int, torch.Tensor], int]:
    if not 0 <= int(source_layer) < int(adapter.num_layers) - 1:
        raise ValueError("Stream source layer must leave at least one downstream layer")
    if any(
        int(layer) < 0 or int(layer) >= int(adapter.num_layers)
        for layer in readout_layers
    ):
        raise ValueError("A stream readout layer is outside the decoder")
    if any(int(layer) <= int(source_layer) for layer in readout_layers):
        raise ValueError(
            "Every downstream readout layer must be strictly after the source layer"
        )
    captures: dict[int, torch.Tensor] = {}
    applications = 0
    handles = []
    if transform is not None:

        def source_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            nonlocal applications
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return output
            selected = hidden[:, list(source_positions), :]
            replacement = transform(selected)
            if replacement.shape != selected.shape:
                raise RuntimeError("Stream transform changed the selected shape")
            patched = hidden.clone()
            patched[:, list(source_positions), :] = replacement.to(
                device=hidden.device, dtype=hidden.dtype
            )
            applications += 1
            return _replace_output_tensor(output, patched)

        handles.append(
            adapter.layers[int(source_layer)].register_forward_hook(source_hook)
        )
    for layer in sorted({int(value) for value in readout_layers}):

        def readout_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return
            captures[layer] = (
                hidden[0, torch.as_tensor(readout_positions, device=hidden.device)]
                .detach()
                .float()
                .cpu()
            )

        handles.append(adapter.layers[layer].register_forward_hook(readout_hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(int(value) for value in readout_layers) - set(captures))
    if missing:
        raise RuntimeError(f"Stream trial missed readout layers {missing}")
    if transform is not None and applications != 1:
        raise RuntimeError(f"Stream transform must apply once, observed {applications}")
    return prefill, captures, applications


def _prefix_forward_with_stream_transform(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    source_layer: int,
    source_positions: Sequence[int],
    transform: Any | None,
) -> tuple[Any, int]:
    """Build an answer-prefix cache after an optional registered state removal.

    Unlike ``_full_prefill_with_stream_transform``, this helper stops immediately
    before the answer query.  The resulting cache can therefore be cloned into
    several source-mask branches without recomputing the 10K-token prefix.
    """

    layer = int(source_layer)
    query = int(encoding.query_position)
    positions = tuple(int(value) for value in source_positions)
    if not 0 <= layer < int(adapter.num_layers):
        raise ValueError("Stream source layer is outside the decoder")
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Stream source positions must be unique and nonempty")
    if min(positions) < 0 or max(positions) >= query:
        raise ValueError("Every stream source must precede the answer query")
    applications = 0
    handle = None
    if transform is not None:

        def source_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            nonlocal applications
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != query:
                return output
            selected = hidden[:, list(positions), :]
            replacement = transform(selected)
            if replacement.shape != selected.shape:
                raise RuntimeError("Stream transform changed the selected shape")
            patched = hidden.clone()
            patched[:, list(positions), :] = replacement.to(
                device=hidden.device, dtype=hidden.dtype
            )
            applications += 1
            return _replace_output_tensor(output, patched)

        handle = adapter.layers[layer].register_forward_hook(source_hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefix = _prefix_forward(
            model,
            adapter,
            input_ids[:, :query],
            attention_mask[:, :query],
        )
    finally:
        if handle is not None:
            handle.remove()
    expected = 0 if transform is None else 1
    if applications != expected:
        raise RuntimeError(
            f"Prefix stream transform must apply {expected} times, observed "
            f"{applications}"
        )
    return prefix, applications


@torch.inference_mode()
def run_count_state_answer_source_factorial_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layer: int,
    center: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
    state_scope: str = "all",
    state_occurrence: int | None = None,
    state_conditions: Sequence[str] = REGISTERED_STREAM_CONDITIONS,
    mask_conditions: Sequence[str] = (
        "clean",
        "block_trace_items",
        "block_trace_items_matched_control",
        "block_prompt_records",
        "block_prompt_records_matched_control",
    ),
    answer_site_id: str = "answer_query_v3",
    mask_application: str = "answer_query_only",
    random_seed: int = 0,
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Cross a written count-state intervention with answer-time read masks.

    The long prefix is evaluated once per state arm and cloned into all mask
    arms.  Consequently every source comparison has exactly the same residual
    intervention and every residual comparison has exactly the same source
    mask.  ``all`` removes the fitted occurrence subspace at every registered
    item endpoint; ``occurrence`` can target one endpoint (including terminal).
    """

    active_state_conditions = tuple(str(value) for value in state_conditions)
    active_mask_conditions = tuple(str(value) for value in mask_conditions)
    if len(set(active_state_conditions)) != len(active_state_conditions):
        raise ValueError("State conditions must be unique")
    if len(set(active_mask_conditions)) != len(active_mask_conditions):
        raise ValueError("Mask conditions must be unique")
    if not active_state_conditions or "clean" not in active_state_conditions:
        raise ValueError("The joint factorial requires a clean state arm")
    unknown_states = sorted(
        set(active_state_conditions) - set(REGISTERED_STREAM_CONDITIONS)
    )
    unknown_masks = sorted(
        set(active_mask_conditions) - set(REGISTERED_MASK_CONDITIONS)
    )
    if unknown_states:
        raise ValueError(f"Unknown state conditions: {unknown_states}")
    if unknown_masks:
        raise ValueError(f"Unknown mask conditions: {unknown_masks}")
    if mask_application not in {
        "answer_query_only",
        "answer_query_and_answer_tokens",
    }:
        raise ValueError(f"Unknown answer source-mask application: {mask_application}")

    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    active_state_scope = str(state_scope)
    active_state_occurrence = state_occurrence
    if active_state_scope == "terminal":
        active_state_scope = "occurrence"
        active_state_occurrence = len(registry.trace_items)
    source_positions, source_occurrences = _stream_source_positions(
        registry,
        scope=active_state_scope,
        occurrence=active_state_occurrence,
    )
    marker_positions = set(registry.positions("trace_markers"))
    marker_overlap = tuple(
        position for position in source_positions if position in marker_positions
    )
    center_tensor = torch.as_tensor(center, dtype=torch.float32)
    basis_tensor = torch.as_tensor(basis, dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    for state_condition in active_state_conditions:
        transform_audit: dict[str, Any] = {"applications": 0}
        transform = None
        if state_condition != "clean":
            transform = _running_state_transform(
                condition=state_condition,
                center=center_tensor,
                basis=basis_tensor,
                random_seed=int(random_seed),
                audit=transform_audit,
            )
        prefix, applications = _prefix_forward_with_stream_transform(
            model,
            adapter,
            encoding,
            source_layer=int(source_layer),
            source_positions=source_positions,
            transform=transform,
        )
        for mask_condition in active_mask_conditions:
            mask, mask_audit = answer_source_mask(
                registry, condition=mask_condition
            )
            prefix_branch = clone_prefill_output_for_scoring(prefix)
            query_output = _query_forward_from_prefix(
                model,
                adapter,
                encoding,
                prefix_output=prefix_branch,
                query_attention_mask=mask,
            )
            scoring_encoding = encoding
            if mask_application == "answer_query_and_answer_tokens":
                scoring_encoding = replace(
                    encoding,
                    attention_mask=tuple(int(value) for value in mask[0].tolist()),
                )
            outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                scoring_encoding,
                query_output,
                run_greedy=run_greedy,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                    "experiment_id": "count_state_answer_source_factorial",
                    "request_id": encoding.request_id,
                    "model_label": encoding.model_label,
                    "seed": encoding.seed,
                    "dataset_split": encoding.split,
                    "gold_count": encoding.count,
                    "answer_site_id": answer_site_id,
                    "state_condition": state_condition,
                    "state_source_layer": int(source_layer),
                    "state_source_scope": state_scope,
                    "state_source_occurrences": list(source_occurrences),
                    "state_source_positions": list(source_positions),
                    "state_source_positions_sha256": _sha256_json(source_positions),
                    "state_basis_site_kind": "item_end",
                    "state_source_marker_overlap_count": len(marker_overlap),
                    "state_source_marker_overlap_fraction": len(marker_overlap)
                    / len(source_positions),
                    "state_source_marker_confounded": bool(marker_overlap),
                    "state_intervention_hook_applications": applications,
                    "state_target_removed_fro_norm": float(
                        transform_audit.get("target_removed_fro_norm", 0.0)
                    ),
                    "state_removed_fro_norm": float(
                        transform_audit.get("removed_fro_norm", 0.0)
                    ),
                    "state_removal_norm_ratio": float(
                        transform_audit.get("norm_ratio", 1.0)
                    ),
                    "mask_condition": mask_condition,
                    "mask_scope": mask_application,
                    "source_edge_mask_head_scope": "all_attention_heads",
                    "source_edge_mask_layer_scope": "all_decoder_layers",
                    "decoder_layer_count": int(adapter.num_layers),
                    "attention_head_instances_per_query": int(
                        sum(adapter.num_heads)
                    ),
                    "registry_sha256": registry.to_dict()["registry_sha256"],
                    **mask_audit,
                    **outcomes,
                }
            )
    expected_rows = len(active_state_conditions) * len(active_mask_conditions)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Joint factorial emitted {len(rows)} rows, expected {expected_rows}"
        )
    return rows


def stream_state_retention_metrics(
    clean_states: np.ndarray,
    intervention_states: np.ndarray,
    *,
    readout_layers: Sequence[int],
    readout_positions: Sequence[int],
    query_position: int,
    count_bases: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """Measure downstream displacement in independently fitted item-end bases.

    Item-end bases are applied only to later item-end states.  The answer-query
    vector is retained as a full-state diagnostic because projecting it into an
    item-end basis would silently assume cross-site basis equivalence.
    """

    clean = np.asarray(clean_states, dtype=np.float64)
    active = np.asarray(intervention_states, dtype=np.float64)
    layers = tuple(int(value) for value in readout_layers)
    positions = tuple(int(value) for value in readout_positions)
    if clean.shape != active.shape or clean.ndim != 3:
        raise ValueError(
            "Stream readout states must share [layers, positions, hidden] shape"
        )
    if clean.shape[:2] != (len(layers), len(positions)):
        raise ValueError("Stream readout metadata disagrees with state shape")
    query_indices = [
        index for index, position in enumerate(positions) if position == query_position
    ]
    if len(query_indices) != 1:
        raise ValueError("Exactly one answer-query readout is required")
    query_index = query_indices[0]
    item_indices = [
        index for index, position in enumerate(positions) if position < query_position
    ]
    if any(position > query_position for position in positions):
        raise ValueError("A stream readout lies after the registered answer query")
    missing_bases = sorted(set(layers) - {int(value) for value in count_bases})
    if missing_bases:
        raise ValueError(f"Readout count bases are missing layers {missing_bases}")

    item_full_squared: list[float] = []
    item_projected_squared: list[float] = []
    query_full_squared: list[float] = []
    layer_rows: list[dict[str, Any]] = []
    delta = active - clean
    for layer_index, layer in enumerate(layers):
        basis = np.asarray(count_bases[layer], dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != clean.shape[-1]:
            raise ValueError(f"L{layer} readout basis has the wrong hidden width")
        if basis.shape[1] < 1 or not np.allclose(
            basis.T @ basis,
            np.eye(basis.shape[1]),
            atol=2e-4,
            rtol=2e-4,
        ):
            raise ValueError(f"L{layer} readout basis is not orthonormal")
        query_delta = delta[layer_index, query_index]
        query_squared = float(np.sum(query_delta * query_delta))
        query_full_squared.append(query_squared)
        if item_indices:
            item_delta = delta[layer_index, item_indices]
            projected = item_delta @ basis
            full_values = np.sum(item_delta * item_delta, axis=-1)
            projected_values = np.sum(projected * projected, axis=-1)
            item_full_squared.extend(float(value) for value in full_values)
            item_projected_squared.extend(float(value) for value in projected_values)
            full_rms = float(np.sqrt(np.mean(full_values)))
            projected_rms = float(np.sqrt(np.mean(projected_values)))
        else:
            full_rms = math.nan
            projected_rms = math.nan
        layer_rows.append(
            {
                "layer": int(layer),
                "downstream_item_count": len(item_indices),
                "downstream_item_full_vector_rms": full_rms,
                "downstream_item_progress_subspace_rms": projected_rms,
                "answer_query_full_vector_norm": float(np.sqrt(query_squared)),
            }
        )

    item_full_rms = (
        float(np.sqrt(np.mean(item_full_squared))) if item_full_squared else math.nan
    )
    item_projected_rms = (
        float(np.sqrt(np.mean(item_projected_squared)))
        if item_projected_squared
        else math.nan
    )
    return {
        "downstream_item_readout_count": len(item_indices) * len(layers),
        "downstream_item_full_state_displacement_rms": item_full_rms,
        "downstream_item_progress_subspace_displacement_rms": item_projected_rms,
        # Higher is better/cleaner, so the standard orthogonal-minus-aligned
        # specificity contrast retains its registered positive orientation.
        "downstream_item_progress_subspace_retention_score": (
            -item_projected_rms if np.isfinite(item_projected_rms) else math.nan
        ),
        "answer_query_full_state_displacement_rms": float(
            np.sqrt(np.mean(query_full_squared))
        ),
        "readout_metric_basis_site_kind": "item_end",
        "answer_query_projected_into_item_basis": False,
        "readout_layer_metrics": layer_rows,
    }


@torch.inference_mode()
def run_stream_state_trial(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    condition: str,
    source_layer: int,
    center: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
    scope: str,
    occurrence: int | None = None,
    readout_layers: Sequence[int] = (),
    answer_site_id: str = "answer_query_v3",
    random_seed: int = 0,
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Remove a running-index state and follow it through the clean suffix."""

    if condition not in REGISTERED_STREAM_CONDITIONS:
        raise ValueError(f"Unknown stream-state condition: {condition}")
    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    source_positions, source_occurrences = _stream_source_positions(
        registry, scope=scope, occurrence=occurrence
    )
    marker_positions = set(registry.positions("trace_markers"))
    source_marker_overlap = tuple(
        position for position in source_positions if position in marker_positions
    )
    if any(position >= registry.query_position for position in source_positions):
        raise RuntimeError("A stream source is not earlier than the answer query")
    downstream_positions = tuple(
        endpoint
        for endpoint in (end - 1 for _start, end in registry.trace_items)
        if endpoint > max(source_positions)
    ) + (registry.query_position,)
    active_readout_layers = tuple(sorted({int(value) for value in readout_layers}))
    if not active_readout_layers:
        active_readout_layers = (int(source_layer) + 1,)
    transform_audit: dict[str, Any] = {"applications": 0}
    transform = None
    if condition != "clean":
        transform = _running_state_transform(
            condition=condition,
            center=torch.as_tensor(center, dtype=torch.float32),
            basis=torch.as_tensor(basis, dtype=torch.float32),
            random_seed=int(random_seed),
            audit=transform_audit,
        )
    prefill, captures, applications = _full_prefill_with_stream_transform(
        model,
        adapter,
        encoding,
        source_layer=int(source_layer),
        source_positions=source_positions,
        transform=transform,
        readout_layers=active_readout_layers,
        readout_positions=downstream_positions,
    )
    outcomes = _score_and_generate_prefill(
        model,
        tokenizer,
        encoding,
        prefill,
        run_greedy=run_greedy,
        max_new_tokens=max_new_tokens,
    )
    ordered_layers = sorted(captures)
    states = np.stack([captures[layer].numpy() for layer in ordered_layers])
    return {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "stream_state_retention",
        "condition": condition,
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "dataset_split": encoding.split,
        "gold_count": encoding.count,
        "answer_site_id": answer_site_id,
        "source_layer": int(source_layer),
        "source_scope": scope,
        "source_occurrences": list(source_occurrences),
        "source_positions": list(source_positions),
        "source_positions_sha256": _sha256_json(source_positions),
        "source_marker_overlap_count": len(source_marker_overlap),
        "source_marker_overlap_fraction": (
            len(source_marker_overlap) / len(source_positions)
        ),
        "source_marker_confounded": bool(source_marker_overlap),
        "no_future_read_guarantee": bool(
            max(source_positions) < min(downstream_positions)
        ),
        "readout_layers": ordered_layers,
        "readout_positions": list(downstream_positions),
        "readout_states": states,
        "intervention_hook_applications": applications,
        **transform_audit,
        **outcomes,
    }


def _states_for_positions(
    captured: torch.Tensor,
    ordered_positions: Sequence[int],
    selected_positions: Sequence[int],
) -> torch.Tensor:
    lookup = {int(position): index for index, position in enumerate(ordered_positions)}
    missing = sorted(set(int(value) for value in selected_positions) - set(lookup))
    if missing:
        raise RuntimeError(f"Clean restoration capture missed positions {missing}")
    return captured[[lookup[int(position)] for position in selected_positions]]


def _registered_ordinary_corruption_banks(
    registry: AnswerSourceRegistry,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    """Allocate three length-matched ordinary banks outside active records.

    The banks respectively supply trace-replacement tokens, ordinary control
    targets, and ordinary-control replacement tokens.  Every span is inside
    the original prompt, and active prompt records are excluded so the control
    cannot silently destroy the same evidence tested by the prompt-source arm.
    """

    lengths = tuple(end - start for start, end in registry.trace_items)
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("Trace corruption requires nonempty trace-item spans")
    prompt_records = registry.prompt_records
    forbidden = set(registry.positions("prompt_records"))
    used = set(forbidden)
    if prompt_records:
        local_lower = max(1, min(start for start, _end in prompt_records) - 64)
        local_upper = min(
            registry.prompt_token_count,
            max(end for _start, end in prompt_records) + 64,
        )
    else:  # Registry validation currently makes this unreachable.
        local_lower, local_upper = 1, registry.prompt_token_count

    def allocate(length: int) -> tuple[int, int]:
        windows = (
            (local_lower, local_upper),
            (1, registry.prompt_token_count),
        )
        for lower, upper in windows:
            for candidate in range(int(lower), int(upper) - int(length) + 1):
                positions = set(range(candidate, candidate + int(length)))
                if not positions.intersection(used):
                    used.update(positions)
                    return candidate, candidate + int(length)
        raise RuntimeError(
            "Could not allocate three prompt-record-free ordinary token banks"
        )

    sources = tuple(allocate(length) for length in lengths)
    controls = tuple(allocate(length) for length in lengths)
    control_sources = tuple(allocate(length) for length in lengths)
    return sources, controls, control_sources


def corrupt_registered_trace_tokens(
    encoding: NativeTraceEncoding,
    registry: AnswerSourceRegistry,
) -> tuple[NativeTraceEncoding, NativeTraceEncoding, dict[str, Any]]:
    """Build trace and ordinary corruptions with identical span budgets."""

    trace_targets = tuple(registry.trace_items)
    sources, control_targets, control_sources = _registered_ordinary_corruption_banks(
        registry
    )
    clean = tuple(int(value) for value in encoding.input_ids)

    def replace_segments(
        targets: Sequence[tuple[int, int]],
        donors: Sequence[tuple[int, int]],
    ) -> tuple[tuple[int, ...], int]:
        if len(targets) != len(donors):
            raise ValueError("Corruption target/source counts disagree")
        changed = list(clean)
        changed_tokens = 0
        for (target_start, target_end), (source_start, source_end) in zip(
            targets, donors
        ):
            replacement = clean[int(source_start) : int(source_end)]
            if len(replacement) != int(target_end) - int(target_start):
                raise RuntimeError("Corruption donor changed a target span length")
            before = changed[int(target_start) : int(target_end)]
            changed[int(target_start) : int(target_end)] = replacement
            changed_tokens += sum(
                left != right for left, right in zip(before, replacement)
            )
        return tuple(changed), int(changed_tokens)

    trace_ids, trace_changed = replace_segments(trace_targets, sources)
    ordinary_ids, ordinary_changed = replace_segments(control_targets, control_sources)
    token_budget = sum(end - start for start, end in trace_targets)
    if any(
        sum(end - start for start, end in bank) != token_budget
        for bank in (sources, control_targets, control_sources)
    ):
        raise RuntimeError("Ordinary corruption banks changed the token budget")
    audit = {
        "token_budget": int(token_budget),
        "trace_changed_tokens": int(trace_changed),
        "control_changed_tokens": int(ordinary_changed),
        "trace_targets": trace_targets,
        "trace_sources": sources,
        "control_targets": control_targets,
        "control_sources": control_sources,
        "active_prompt_records_excluded": True,
        "corruption_plan_sha256": _sha256_json(
            {
                "trace_targets": trace_targets,
                "trace_sources": sources,
                "control_targets": control_targets,
                "control_sources": control_sources,
            }
        ),
    }
    return (
        replace(encoding, input_ids=trace_ids),
        replace(encoding, input_ids=ordinary_ids),
        audit,
    )


def _prefill_with_state_replacements(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    layer: int,
    positions: Sequence[int],
    states: torch.Tensor | None,
) -> tuple[Any, int, float]:
    applications = 0
    realized_norm = 0.0
    handle = None
    if states is not None:
        if len(positions) != int(states.shape[0]):
            raise ValueError("Restoration position/state counts disagree")

        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            nonlocal applications, realized_norm
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return output
            before = hidden[:, list(positions), :]
            replacement = states.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(
                0
            )
            patched = hidden.clone()
            patched[:, list(positions), :] = replacement
            realized_norm = float(
                torch.linalg.vector_norm(before.float() - replacement.float())
                .detach()
                .cpu()
            )
            applications += 1
            return _replace_output_tensor(output, patched)

        handle = adapter.layers[int(layer)].register_forward_hook(hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        if handle is not None:
            handle.remove()
    if states is not None and applications != 1:
        raise RuntimeError(f"Restoration hook must apply once, observed {applications}")
    return prefill, applications, realized_norm


@torch.inference_mode()
def run_trace_restoration_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    layer: int,
    conditions: Sequence[str] = REGISTERED_RESTORATION_CONDITIONS,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Run the native analogue of full-span source corruption/restoration."""

    requested = tuple(dict.fromkeys(str(value) for value in conditions))
    unknown = sorted(set(requested) - set(REGISTERED_RESTORATION_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown trace restoration conditions: {unknown}")
    if not 0 <= int(layer) < int(adapter.num_layers) - 1:
        raise ValueError(
            "Restoration layer must leave at least one downstream decoder layer"
        )
    clean, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    trace_corrupt, ordinary_corrupt, corruption_audit = corrupt_registered_trace_tokens(
        clean, registry
    )
    trace_positions = registry.positions("trace_items")
    endpoint_positions = tuple(end - 1 for _start, end in registry.trace_items)
    marker_positions = registry.positions("trace_markers")
    ordinary_positions = tuple(
        position
        for start, end in corruption_audit["control_targets"]
        for position in range(int(start), int(end))
    )
    capture_positions = tuple(sorted(set(trace_positions) | set(ordinary_positions)))
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        clean,
        capture_positions,
        layers=[int(layer)],
    )
    clean_states = captured[int(layer)]
    rows: list[dict[str, Any]] = []
    for condition in requested:
        if condition == "trace_corrupt_marker_restore" and not marker_positions:
            rows.append(
                {
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                    "experiment_id": "trace_source_restoration",
                    "condition": condition,
                    "status": "not_applicable",
                    "exclusion_reason": "no_parser_aligned_visible_marker_tokens",
                    "request_id": clean.request_id,
                    "model_label": clean.model_label,
                    "seed": clean.seed,
                    "dataset_split": clean.split,
                    "gold_count": clean.count,
                    "layer": int(layer),
                }
            )
            continue
        active = clean
        positions: tuple[int, ...] = ()
        replacement_states: torch.Tensor | None = None
        if condition == "trace_token_corrupt":
            active = trace_corrupt
        elif condition == "ordinary_token_corrupt":
            active = ordinary_corrupt
        elif condition == "trace_corrupt_full_span_restore":
            active = trace_corrupt
            positions = trace_positions
        elif condition == "trace_corrupt_endpoint_restore":
            active = trace_corrupt
            positions = endpoint_positions
        elif condition == "trace_corrupt_marker_restore":
            active = trace_corrupt
            positions = marker_positions
        elif condition == "trace_corrupt_ordinary_state_patch":
            active = trace_corrupt
            positions = ordinary_positions
        elif condition == "ordinary_corrupt_ordinary_state_restore":
            active = ordinary_corrupt
            positions = ordinary_positions
        if positions:
            replacement_states = _states_for_positions(
                clean_states, capture_positions, positions
            )
        prefill, applications, realized_norm = _prefill_with_state_replacements(
            model,
            adapter,
            active,
            layer=int(layer),
            positions=positions,
            states=replacement_states,
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            active,
            prefill,
            run_greedy=run_greedy,
            max_new_tokens=max_new_tokens,
        )
        rows.append(
            {
                "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                "experiment_id": "trace_source_restoration",
                "condition": condition,
                "status": "ok",
                "request_id": clean.request_id,
                "model_label": clean.model_label,
                "seed": clean.seed,
                "dataset_split": clean.split,
                "gold_count": clean.count,
                "answer_site_id": answer_site_id,
                "layer": int(layer),
                "token_budget": int(corruption_audit["token_budget"]),
                "trace_changed_tokens": int(corruption_audit["trace_changed_tokens"]),
                "ordinary_changed_tokens": int(
                    corruption_audit["control_changed_tokens"]
                ),
                "restored_token_count": len(positions),
                "restored_positions_sha256": _sha256_json(positions),
                "restoration_hook_applications": applications,
                "restoration_realized_fro_norm": realized_norm,
                "registry_sha256": registry.to_dict()["registry_sha256"],
                **outcomes,
            }
        )
    return rows


REGISTERED_CONTRASTS: dict[str, dict[str, dict[str, float]]] = {
    "answer_source_mask_factorial": {
        "trace_damage": {"clean": 1.0, "block_trace_context": -1.0},
        "trace_item_damage": {"clean": 1.0, "block_trace_items": -1.0},
        "trace_other_damage": {"clean": 1.0, "block_trace_other": -1.0},
        "prompt_damage": {"clean": 1.0, "block_prompt_records": -1.0},
        "joint_damage": {"clean": 1.0, "block_trace_and_prompt": -1.0},
        "trace_prompt_interaction": {
            "clean": 1.0,
            "block_trace_context": -1.0,
            "block_prompt_records": -1.0,
            "block_trace_and_prompt": 1.0,
        },
        "trace_source_specificity": {
            "block_trace_context_matched_control": 1.0,
            "block_trace_context": -1.0,
        },
        "trace_item_source_specificity": {
            "block_trace_items_matched_control": 1.0,
            "block_trace_items": -1.0,
        },
        "prompt_source_specificity": {
            "block_prompt_records_matched_control": 1.0,
            "block_prompt_records": -1.0,
        },
        "terminal_trace_damage": {
            "clean": 1.0,
            "block_terminal_trace": -1.0,
        },
        "earlier_trace_damage": {
            "clean": 1.0,
            "block_earlier_trace": -1.0,
        },
        "marker_damage": {"clean": 1.0, "block_trace_markers": -1.0},
        "marker_source_specificity": {
            "block_trace_markers_matched_control": 1.0,
            "block_trace_markers": -1.0,
        },
        "nonmarker_trace_damage": {
            "clean": 1.0,
            "block_trace_nonmarkers": -1.0,
        },
    },
    "answer_broad_head_ablation": {
        "selected_damage": {"clean": 1.0, "selected_bank": -1.0},
        "selected_vs_layer_matched_random": {
            "layer_matched_random": 1.0,
            "selected_bank": -1.0,
        },
    },
    "trace_intermediate_state_patching": {
        "full_donor_vs_self_transport": {
            "full_donor_patch": 1.0,
            "self_patch": -1.0,
        },
        "projected_donor_vs_self_transport": {
            "progress_projected_patch": 1.0,
            "self_patch": -1.0,
        },
        "projected_vs_orthogonal_transport": {
            "progress_projected_patch": 1.0,
            "norm_matched_orthogonal_patch": -1.0,
        },
    },
    "trace_terminal_state_patching": {
        "full_donor_vs_self_transport": {
            "full_donor_patch": 1.0,
            "self_patch": -1.0,
        },
        "projected_donor_vs_self_transport": {
            "progress_projected_patch": 1.0,
            "self_patch": -1.0,
        },
        "projected_vs_orthogonal_transport": {
            "progress_projected_patch": 1.0,
            "norm_matched_orthogonal_patch": -1.0,
        },
    },
    "stream_state_retention": {
        "aligned_damage": {
            "clean": 1.0,
            "aligned_running_state_removal": -1.0,
        },
        "aligned_vs_orthogonal_specificity": {
            "norm_matched_orthogonal_removal": 1.0,
            "aligned_running_state_removal": -1.0,
        },
    },
    "trace_source_restoration": {
        "trace_token_damage": {"clean": 1.0, "trace_token_corrupt": -1.0},
        "trace_vs_ordinary_token_specificity": {
            "ordinary_token_corrupt": 1.0,
            "trace_token_corrupt": -1.0,
        },
        "full_span_repair": {
            "trace_corrupt_full_span_restore": 1.0,
            "trace_token_corrupt": -1.0,
        },
        "endpoint_repair": {
            "trace_corrupt_endpoint_restore": 1.0,
            "trace_token_corrupt": -1.0,
        },
        "marker_repair": {
            "trace_corrupt_marker_restore": 1.0,
            "trace_token_corrupt": -1.0,
        },
        "full_span_vs_ordinary_repair_specificity": {
            "trace_corrupt_full_span_restore": 1.0,
            "trace_token_corrupt": -1.0,
            "ordinary_corrupt_ordinary_state_restore": -1.0,
            "ordinary_token_corrupt": 1.0,
        },
        "endpoint_vs_ordinary_repair_specificity": {
            "trace_corrupt_endpoint_restore": 1.0,
            "trace_token_corrupt": -1.0,
            "ordinary_corrupt_ordinary_state_restore": -1.0,
            "ordinary_token_corrupt": 1.0,
        },
        "marker_vs_ordinary_repair_specificity": {
            "trace_corrupt_marker_restore": 1.0,
            "trace_token_corrupt": -1.0,
            "ordinary_corrupt_ordinary_state_restore": -1.0,
            "ordinary_token_corrupt": 1.0,
        },
        "full_span_vs_endpoint_restore": {
            "trace_corrupt_full_span_restore": 1.0,
            "trace_corrupt_endpoint_restore": -1.0,
        },
        "full_span_vs_ordinary_state_patch": {
            "trace_corrupt_full_span_restore": 1.0,
            "trace_corrupt_ordinary_state_patch": -1.0,
        },
        "endpoint_vs_ordinary_state_patch": {
            "trace_corrupt_endpoint_restore": 1.0,
            "trace_corrupt_ordinary_state_patch": -1.0,
        },
    },
}


def registered_contrasts(experiment_id: str) -> dict[str, dict[str, float]]:
    if experiment_id not in REGISTERED_CONTRASTS:
        raise KeyError(f"No registered contrasts for {experiment_id}")
    return {
        name: dict(coefficients)
        for name, coefficients in REGISTERED_CONTRASTS[experiment_id].items()
    }


def summarize_linear_contrasts(
    trials: pd.DataFrame,
    *,
    experiment_id: str,
    outcome: str,
    bootstrap_samples: int = 10_000,
    random_seed: int = 0,
    unit_columns: Sequence[str] = ("model_label", "request_id", "seed"),
    stratum_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate occurrences within request, then requests within seed.

    Positive effects always have the registered interpretation: damage,
    repair, or specificity as named by the contrast.
    """

    needed = {
        "experiment_id",
        "condition",
        outcome,
        *unit_columns,
        *stratum_columns,
    }
    missing = sorted(needed - set(trials.columns))
    if missing:
        raise ValueError(f"Count-stream trial table is missing {missing}")
    selected = trials.loc[trials["experiment_id"].eq(experiment_id)].copy()
    for column in stratum_columns:
        selected[column] = selected[column].map(
            lambda value: (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (list, tuple, dict, set))
                else value
            )
        )
    if "status" in selected.columns:
        selected = selected.loc[selected["status"].fillna("ok").eq("ok")]
    selected[outcome] = pd.to_numeric(selected[outcome], errors="coerce")
    selected = selected.loc[np.isfinite(selected[outcome])]
    if selected.empty:
        raise ValueError("No finite registered trial outcomes remain")
    grouping = [*stratum_columns, *unit_columns, "condition"]
    request = selected.groupby(grouping, as_index=False)[outcome].mean()
    index_columns = [*stratum_columns, *unit_columns]
    pivot = request.pivot(index=index_columns, columns="condition", values=outcome)
    contrasts = registered_contrasts(experiment_id)
    seed_effect_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    strata: Iterable[tuple[Any, pd.DataFrame]]
    if stratum_columns:
        reset = pivot.reset_index()
        group_key: str | list[str] = (
            stratum_columns[0] if len(stratum_columns) == 1 else list(stratum_columns)
        )
        strata = reset.groupby(group_key, dropna=False, sort=True)
    else:
        strata = [((), pivot.reset_index())]
    for raw_stratum, frame in strata:
        stratum_values = (
            raw_stratum if isinstance(raw_stratum, tuple) else (raw_stratum,)
        )
        stratum_payload = dict(zip(stratum_columns, stratum_values))
        for contrast_name, coefficients in contrasts.items():
            required = set(coefficients)
            if not required <= set(frame.columns):
                continue
            complete = frame.dropna(subset=sorted(required)).copy()
            if complete.empty:
                continue
            complete["request_effect"] = sum(
                float(coefficient) * complete[condition]
                for condition, coefficient in coefficients.items()
            )
            seed_effects = complete.groupby(
                ["model_label", "seed"], as_index=False
            ).agg(
                effect=("request_effect", "mean"),
                request_count=("request_id", "nunique"),
            )
            for value in seed_effects.itertuples(index=False):
                seed_effect_rows.append(
                    {
                        **stratum_payload,
                        "experiment_id": experiment_id,
                        "contrast": contrast_name,
                        "outcome": outcome,
                        "model_label": str(value.model_label),
                        "seed": int(value.seed),
                        "effect": float(value.effect),
                        "request_count": int(value.request_count),
                    }
                )
            for model_label, model_effects in seed_effects.groupby("model_label"):
                values = model_effects["effect"].to_numpy(dtype=float)
                interval = bootstrap_seed_mean_ci(
                    values,
                    samples=int(bootstrap_samples),
                    seed=int(random_seed)
                    + int.from_bytes(
                        hashlib.sha256(
                            f"{experiment_id}:{contrast_name}:{model_label}".encode(
                                "utf-8"
                            )
                        ).digest()[:4],
                        "big",
                    ),
                )
                summary_rows.append(
                    {
                        **stratum_payload,
                        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                        "experiment_id": experiment_id,
                        "contrast": contrast_name,
                        "contrast_coefficients": json.dumps(
                            coefficients, sort_keys=True
                        ),
                        "outcome": outcome,
                        "model_label": str(model_label),
                        "mean_effect": float(np.mean(values)),
                        "ci_low": float(interval["ci_low"]),
                        "ci_high": float(interval["ci_high"]),
                        "p_value": float(sign_flip_pvalue(values)),
                        "seed_count": int(len(values)),
                        "request_count": int(model_effects["request_count"].sum()),
                    }
                )
    if not summary_rows:
        raise ValueError("No registered contrast had complete condition support")
    summary = pd.DataFrame(summary_rows)
    family_columns = [
        *stratum_columns,
        "experiment_id",
        "contrast",
        "outcome",
    ]
    summary["p_value_holm_across_models"] = np.nan
    for _key, family in summary.groupby(family_columns, dropna=False, sort=False):
        adjusted = holm_adjust(family["p_value"].to_numpy(dtype=float))
        summary.loc[family.index, "p_value_holm_across_models"] = adjusted
    return summary, pd.DataFrame(seed_effect_rows)


def mechanism_decision_ledger(summary: pd.DataFrame) -> pd.DataFrame:
    """Produce claim gates without converting statistical code into a claim."""

    needed = {"experiment_id", "contrast", "model_label", "mean_effect", "ci_low"}
    missing = sorted(needed - set(summary.columns))
    if missing:
        raise ValueError(f"Mechanism summary is missing {missing}")
    # The optional coordinates keep source-specific answer banks and temporal
    # patch directions separate.  In particular, a future-to-past transplant
    # is a useful representational diagnostic but cannot establish natural
    # forward propagation through the trace stream.
    required_gates: dict[str, tuple[tuple[str, str, str | None, str | None], ...]] = {
        "stream_written_state": (
            (
                "trace_intermediate_state_patching",
                "full_donor_vs_self_transport",
                None,
                "past_to_later_receiver",
            ),
            (
                "trace_intermediate_state_patching",
                "projected_vs_orthogonal_transport",
                None,
                "past_to_later_receiver",
            ),
        ),
        "answer_time_trace_retrieval": (
            (
                "answer_broad_head_ablation",
                "selected_vs_layer_matched_random",
                "trace_items",
                None,
            ),
        ),
        "answer_time_prompt_retrieval": (
            (
                "answer_broad_head_ablation",
                "selected_vs_layer_matched_random",
                "prompt_records",
                None,
            ),
        ),
    }

    def _source_group(row: Any) -> str | None:
        value = getattr(row, "source_group", None)
        if value is None or pd.isna(value):
            return None
        return str(value)

    def _donor_direction(row: Any) -> str | None:
        value = getattr(row, "donor_direction", None)
        if value is None or pd.isna(value):
            return None
        return str(value)

    rows: list[dict[str, Any]] = []
    for model_label, model in summary.groupby("model_label"):
        candidates: dict[tuple[str, str, str | None, str | None], list[Any]] = {}
        for row in model.itertuples(index=False):
            key = (
                str(row.experiment_id),
                str(row.contrast),
                _source_group(row),
                _donor_direction(row),
            )
            candidates.setdefault(key, []).append(row)
        lookup = {
            key: values[0] for key, values in candidates.items() if len(values) == 1
        }
        for claim, gates in required_gates.items():
            observed = [gate for gate in gates if gate in lookup]
            missing_gates = [gate for gate in gates if gate not in candidates]
            ambiguous_gates = [
                gate for gate in gates if len(candidates.get(gate, ())) > 1
            ]
            directional = all(float(lookup[gate].mean_effect) > 0 for gate in observed)
            ci_positive = bool(observed) and all(
                float(lookup[gate].ci_low) > 0 for gate in observed
            )
            holm_positive = bool(observed) and all(
                not hasattr(lookup[gate], "p_value_holm_across_models")
                or float(lookup[gate].p_value_holm_across_models) < 0.05
                for gate in observed
            )
            rows.append(
                {
                    "model_label": str(model_label),
                    "claim": claim,
                    "status": (
                        "passes_registered_gate"
                        if not missing_gates
                        and not ambiguous_gates
                        and ci_positive
                        and holm_positive
                        else (
                            "directional_only"
                            if not missing_gates and not ambiguous_gates and directional
                            else "not_established"
                        )
                    ),
                    "observed_gates": json.dumps(observed),
                    "missing_gates": json.dumps(missing_gates),
                    "ambiguous_gates": json.dumps(ambiguous_gates),
                    "all_mean_effects_positive": bool(directional),
                    "all_ci_lows_positive": bool(ci_positive),
                    "all_holm_p_below_0_05": bool(holm_positive),
                    "interpretation_guard": (
                        "A passing component gate supports that component only; "
                        "it does not establish a unique circuit or scalar counter."
                    ),
                }
            )
    return pd.DataFrame(rows)
