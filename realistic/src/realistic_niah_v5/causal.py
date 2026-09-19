from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah.parsing import parse_total
from realistic_niah_v4.counter_channel_interventions import removal_transform
from realistic_niah_v4.counter_channel_interventions import (
    run_counter_subspace_conditions,
)
from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _replace_output_tensor,
    _tensor_from_output,
    capture_post_block_states,
    generate_answer_completion,
    generate_with_residual_interventions,
    generate_with_residual_transforms,
    position_attention_outputs,
)
from .causal_sites import compile_causal_site_plan
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
    prompt_token_ids,
    parse_trace_record,
    raw_output_text,
    trace_char_sites,
)
from .spec import V5Config


CAUSAL_SCHEMA_VERSION = "realistic_niah_v5_causal_v3"

TARGET_CITY_HEAD_ABLATION_SCOPES = (
    "registered_query_only",
    "last_pre_city_only",
    "registered_query_through_last_pre_city",
    "city_predictor_chain",
    "registered_query_through_city_prefix",
)


CAUSAL_HEAD_SELECTION_COLUMNS = (
    "target_source_attention_mass",
    "target_source_relative_attention_mass",
    # Historical alias retained for previously captured source-write tables.
    "source_attention_mass",
    "target_minus_max_wrong_source_attention_mass",
    "source_specific_ov_write_norm",
)


def _head_selection_column(selection_metric: str) -> str:
    """Resolve a registered selection policy or an explicit source column."""

    value = str(selection_metric)
    if value in CAUSAL_HEAD_SELECTION_COLUMNS:
        return value
    prefix = "seed_first_equal_anchor_mean_"
    if value.startswith(prefix):
        column = value.removeprefix(prefix)
        if column in CAUSAL_HEAD_SELECTION_COLUMNS:
            return column
    raise ValueError(
        "Unknown causal head selection metric: "
        f"{selection_metric!r}; expected one of "
        f"{list(CAUSAL_HEAD_SELECTION_COLUMNS)}"
    )


def _target_grammar_class(value: Any) -> str:
    """Return the target-side grammar label from a stored grammar pair."""

    text = str(value)
    return text.rsplit(" -> ", 1)[-1]


@dataclass(frozen=True)
class CausalExperiment:
    experiment_id: str
    report_role: str
    intervention: str
    primary_endpoint: str
    required_controls: tuple[str, ...]
    implementation: str


V5_CAUSAL_LEDGER: tuple[CausalExperiment, ...] = (
    CausalExperiment(
        "earlier_span_attention",
        "Step 1: evidence for accumulation across earlier trace items",
        "restrict item-end query context to prior accepted-item spans",
        "downstream residual displacement and attention mass",
        ("clean", "length/depth-matched generated-text context"),
        "native query-context mask runner",
    ),
    CausalExperiment(
        "trace_token_corruption",
        "Step 1: token-level necessity of enumerated trace items",
        "replace trace-item tokens with equal-length non-trace tokens",
        "count accuracy, signed error, answer-query displacement",
        ("clean", "equal-token-count non-trace corruption"),
        "corrupt_trace_tokens",
    ),
    CausalExperiment(
        "trace_subspace_ablation",
        "Step 1/3: necessity of the discovery-frozen running-index subspace",
        "remove count-subspace component at trace endpoints",
        "count accuracy and answer-query count-axis coefficient",
        ("clean", "equal-norm orthogonal residual removal"),
        "run_subspace_ablation",
    ),
    CausalExperiment(
        "retrieval_anchor_head_ablation",
        "Step 2a: localize the k-to-next-city retrieval transition",
        "zero one frozen head bank at grammar-aware P0--P3 anchors",
        "fixed target-city-token likelihood under a shared teacher-forced path",
        ("clean", "layer-matched random heads", "same bank/K at every anchor"),
        "run_mechanism_head_ablation_trials",
    ),
    CausalExperiment(
        "retrieval_head_free_continuation",
        "Step 2a behavioral endpoint: observe the needle actually emitted after head ablation",
        "teacher-force through a registered pre-city anchor or fixed window, zero selected pre-O head slices once in prefill, then greedily free-generate",
        "first emitted gold city: correct next needle, wrong gold needle, or none",
        ("clean", "three layer-matched random head banks"),
        "run_retrieval_head_behavior_trial",
    ),
    CausalExperiment(
        "retrieval_source_edge_intervention",
        "Step 2a follow-up: test target-record specificity at the localized city-pre query",
        "remove or restore the clean-state target-record OV contribution within one frozen selected-head layer group",
        "fixed target-city-token likelihood under a shared teacher-forced path",
        (
            "clean",
            "closest-natural-norm wrong-record source contribution, natural and norm-matched",
            "selected-layer head ablation",
        ),
        "run_retrieval_source_edge_trials",
    ),
    CausalExperiment(
        "progress_transition_head_ablation",
        "Step 2b: necessity of progress-transition heads",
        "zero discovery-frozen head outputs at accepted item-end P0 queries",
        "fixed next-city token likelihood; stop is analyzed separately",
        ("clean", "layer-matched random heads", "one frozen K"),
        "run_mechanism_head_ablation_trials",
    ),
    CausalExperiment(
        "answer_query_patch",
        "Step 3: sufficiency of answer-query count state",
        "patch matched donor state into receiver answer-query residual",
        "transport toward donor count",
        ("self patch", "mismatched donor", "orthogonal norm-matched delta"),
        "run_residual_patch_trial",
    ),
    CausalExperiment(
        "trace_endpoint_patch",
        "Step 1 -> Step 3 causal transport",
        "patch discovery-frozen trace endpoint state/subspace",
        "answer-query representation and count transport",
        ("self patch", "orthogonal norm-matched delta"),
        "run_residual_patch_trial/run_counter_subspace_conditions",
    ),
    CausalExperiment(
        "qwen_natural_ov_write",
        "Step 2 optional delayed OV rewriting (Qwen)",
        "trace natural attention-weighted OV contribution",
        "count-axis write and candidate-score change",
        ("selected heads", "layer-matched controls", "read/write partition"),
        "capture_natural_head_writes",
    ),
    CausalExperiment(
        "gemma_distributed_residual_write",
        "Step 2 optional distributed residual write (Gemma)",
        "trace/patch the registered cross-layer residual contribution",
        "count-axis write and candidate-score change",
        ("equal-norm orthogonal delta", "site-set specificity"),
        "capture_natural_head_writes with cross-layer aggregation",
    ),
)


def causal_ledger_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **experiment.__dict__,
                "required_controls": ";".join(experiment.required_controls),
            }
            for experiment in V5_CAUSAL_LEDGER
        ]
    )


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def rank_mechanism_heads(
    attention: pd.DataFrame,
    *,
    mechanism: str,
    split: str = "discovery",
) -> pd.DataFrame:
    """Freeze heads by query-weighted semantic target mass on discovery."""

    needed = {
        "model_label",
        "split",
        "mechanism",
        "gold_count",
        "trace_one_to_one",
        "layer",
        "head",
        "target_needle_raw_mass",
        "target_needle_relative_mass",
        "target_needle_top1",
    }
    missing = sorted(needed - set(attention.columns))
    if missing:
        raise ValueError(f"Attention table is missing {missing}")
    if mechanism not in {"targeted_retrieval", "progress_transition"}:
        raise ValueError(f"Unknown head mechanism: {mechanism}")
    mask = attention["split"].astype(str).str.lower().eq(split.lower())
    mask &= attention["mechanism"].astype(str).eq(mechanism)
    one_to_one = attention["trace_one_to_one"]
    if one_to_one.dtype == bool:
        mask &= one_to_one
    else:
        mask &= one_to_one.astype(str).str.lower().isin({"true", "1"})
    selected = attention.loc[mask].copy()
    if selected.empty:
        raise ValueError(
            f"No attention rows matched {split}/{mechanism}"
        )
    grouped = (
        selected.groupby(["model_label", "layer", "head"], as_index=False)
        .agg(
            discovery_target_raw_mass=("target_needle_raw_mass", "mean"),
            discovery_target_relative_mass=("target_needle_relative_mass", "mean"),
            discovery_target_top1=("target_needle_top1", "mean"),
            n_queries=("target_needle_raw_mass", "size"),
        )
        .sort_values(
            ["model_label", "discovery_target_raw_mass", "layer", "head"],
            ascending=[True, False, True, True],
        )
    )
    grouped["discovery_rank"] = (
        grouped.groupby("model_label").cumcount() + 1
    )
    grouped["mechanism"] = mechanism
    grouped["query_site_kind"] = (
        "marker_end" if mechanism == "targeted_retrieval" else "item_end"
    )
    grouped["selection_metric"] = "query_weighted_mean_target_needle_raw_mass"
    grouped["selection_split"] = split
    grouped["selection_cohort"] = "one_to_one"
    grouped["selection_counts"] = "1-10"
    return grouped.reset_index(drop=True)


def rank_retrieval_heads(
    attention: pd.DataFrame, *, split: str = "discovery"
) -> pd.DataFrame:
    """Compatibility name for the registered targeted-retrieval ranking."""

    return rank_mechanism_heads(
        attention, mechanism="targeted_retrieval", split=split
    )


def rank_answer_query_heads(
    attention: pd.DataFrame,
    *,
    mechanism: str = "answer_prompt_aggregation",
    split: str = "discovery",
) -> pd.DataFrame:
    """Freeze prompt- and trace-aggregation answer-query heads separately.

    Confirmation rows are retained only for later descriptive bank-mass
    evaluation; they never influence ordering or membership.
    """

    mechanisms = {
        "answer_prompt_aggregation": (
            "target_needle_raw_mass",
            "target_needle_relative_mass",
            "prompt_broad_score",
            "prompt_broad_coverage",
            "exact_prompt_needle_span_broad_score",
        ),
        "answer_trace_aggregation": (
            "trace_item_raw_mass",
            "trace_item_relative_mass",
            "trace_broad_score",
            "trace_broad_coverage",
            "registered_trace_item_span_broad_score",
        ),
    }
    if mechanism not in mechanisms:
        raise ValueError(f"Unsupported answer-query mechanism: {mechanism}")
    (
        selection_raw,
        selection_relative,
        selection_broad,
        selection_coverage,
        selection_name,
    ) = mechanisms[mechanism]
    needed = {
        "model_label",
        "split",
        "trace_one_to_one",
        "layer",
        "head",
        "target_needle_raw_mass",
        "target_needle_relative_mass",
        "trace_item_raw_mass",
        "trace_item_relative_mass",
        "prompt_broad_score",
        "prompt_broad_coverage",
        "trace_broad_score",
        "trace_broad_coverage",
    }
    missing = sorted(needed - set(attention.columns))
    if missing:
        raise ValueError(f"Answer-query attention table is missing {missing}")
    selected = attention.copy()
    mask = selected["split"].astype(str).str.lower().eq(split.lower())
    one_to_one = selected["trace_one_to_one"]
    if one_to_one.dtype == bool:
        mask &= one_to_one
    else:
        mask &= one_to_one.astype(str).str.lower().isin({"true", "1"})
    selected = selected.loc[mask]
    if selected.empty:
        raise ValueError(f"No answer-query attention rows matched {split}")
    grouped = (
        selected.groupby(["model_label", "layer", "head"], as_index=False)
        .agg(
            discovery_target_raw_mass=("target_needle_raw_mass", "mean"),
            discovery_target_relative_mass=(
                "target_needle_relative_mass", "mean"
            ),
            discovery_trace_raw_mass=("trace_item_raw_mass", "mean"),
            discovery_trace_relative_mass=("trace_item_relative_mass", "mean"),
            discovery_prompt_broad_score=("prompt_broad_score", "mean"),
            discovery_prompt_broad_coverage=("prompt_broad_coverage", "mean"),
            discovery_trace_broad_score=("trace_broad_score", "mean"),
            discovery_trace_broad_coverage=("trace_broad_coverage", "mean"),
            n_queries=("target_needle_raw_mass", "size"),
        )
    )
    grouped["discovery_selection_raw_mass"] = grouped[
        "discovery_target_raw_mass"
        if selection_raw == "target_needle_raw_mass"
        else "discovery_trace_raw_mass"
    ]
    grouped["discovery_selection_relative_mass"] = grouped[
        "discovery_target_relative_mass"
        if selection_relative == "target_needle_relative_mass"
        else "discovery_trace_relative_mass"
    ]
    grouped["discovery_selection_broad_score"] = grouped[
        "discovery_prompt_broad_score"
        if selection_broad == "prompt_broad_score"
        else "discovery_trace_broad_score"
    ]
    grouped["discovery_selection_broad_coverage"] = grouped[
        "discovery_prompt_broad_coverage"
        if selection_coverage == "prompt_broad_coverage"
        else "discovery_trace_broad_coverage"
    ]
    grouped = grouped.sort_values(
        [
            "model_label",
            "discovery_selection_broad_score",
            "discovery_selection_raw_mass",
            "layer",
            "head",
        ],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    grouped["discovery_rank"] = grouped.groupby("model_label").cumcount() + 1
    grouped["mechanism"] = mechanism
    grouped["query_site_kind"] = "answer_query_v3"
    grouped["selection_metric"] = f"query_weighted_mean_{selection_name}"
    grouped["selection_split"] = split
    grouped["selection_cohort"] = "one_to_one"
    return grouped
def _source_row_has_anchor_role(row: Mapping[str, Any], role: str) -> bool:
    values: list[str] = []
    raw_roles = row.get("anchor_roles")
    if isinstance(raw_roles, (list, tuple, set)):
        values.extend(str(value) for value in raw_roles)
    elif isinstance(raw_roles, str) and raw_roles.strip():
        try:
            parsed = json.loads(raw_roles)
        except json.JSONDecodeError:
            parsed = [raw_roles]
        if isinstance(parsed, list):
            values.extend(str(value) for value in parsed)
        else:
            values.append(str(parsed))
    raw_primary = row.get("anchor_role")
    if raw_primary is not None and not pd.isna(raw_primary):
        values.append(str(raw_primary))
    return str(role) in set(values)


def rank_pooled_source_specific_heads(
    writes: pd.DataFrame,
    *,
    development_seeds: Sequence[int] | None = None,
    anchor_role: str | None = None,
    minimum_layer: int | None = None,
    maximum_layer: int | None = None,
    selection_metric: str = "source_specific_ov_write_norm",
    target_grammar_class: str | None = None,
    target_retrieval_surface_variant: str | None = None,
    selection_eligibility_scope: str = "primary",
    selection_aggregation: str = "request_equal",
) -> pd.DataFrame:
    """Rank heads without allowing the candidate anchor to select its own bank.

    Events are averaged within request, then anchor roles within seed, and only
    then across seeds.  Seeds are therefore equally weighted, while grammar
    classes retain their natural within-seed event proportions.
    """

    selection_eligibility_scope = str(selection_eligibility_scope)
    if selection_eligibility_scope not in {"primary", "local"}:
        raise ValueError(
            "selection_eligibility_scope must be 'primary' or 'local'"
        )
    eligibility_column = (
        "primary_anchor_eligible"
        if selection_eligibility_scope == "primary"
        else "local_anchor_eligible"
    )
    selection_aggregation = str(selection_aggregation)
    if selection_aggregation not in {"request_equal", "seed_event_mean"}:
        raise ValueError(
            "selection_aggregation must be 'request_equal' or 'seed_event_mean'"
        )
    selection_column = _head_selection_column(selection_metric)
    needed = {
        "model_label",
        "request_id",
        "seed",
        "anchor_role",
        eligibility_column,
        "layer",
        "head",
        selection_column,
    }
    if target_grammar_class is not None and not {
        "target_grammar_class",
        "grammar_pair",
    }.intersection(writes.columns):
        needed.add("grammar_pair")
    if (
        target_retrieval_surface_variant is not None
        and "target_retrieval_surface_variant" not in writes.columns
    ):
        needed.add("target_retrieval_surface_variant")
    missing = sorted(needed - set(writes.columns))
    if missing:
        raise ValueError(f"Source-specific write table is missing {missing}")
    selected = writes.copy()
    if anchor_role is not None:
        selected = selected.loc[
            selected.apply(
                lambda row: _source_row_has_anchor_role(row, str(anchor_role)),
                axis=1,
            )
        ]
    if target_grammar_class is not None:
        if "target_grammar_class" in selected:
            target_grammar = selected["target_grammar_class"].astype(str)
        else:
            target_grammar = selected["grammar_pair"].map(
                _target_grammar_class
            )
        selected = selected.loc[
            target_grammar.eq(str(target_grammar_class))
        ]
    if target_retrieval_surface_variant is not None:
        selected = selected.loc[
            selected["target_retrieval_surface_variant"]
            .astype(str)
            .eq(str(target_retrieval_surface_variant))
        ]
    if minimum_layer is not None:
        selected = selected.loc[
            pd.to_numeric(selected["layer"], errors="coerce")
            >= int(minimum_layer)
        ]
    if maximum_layer is not None:
        selected = selected.loc[
            pd.to_numeric(selected["layer"], errors="coerce")
            <= int(maximum_layer)
        ]
    if development_seeds is not None:
        allowed = {int(value) for value in development_seeds}
        selected = selected.loc[
            pd.to_numeric(selected["seed"], errors="coerce").isin(allowed)
        ]
    if "status" in selected:
        selected = selected.loc[selected["status"].astype(str).eq("ok")]
    if "event_specific" in selected:
        values = selected["event_specific"]
        mask = values if values.dtype == bool else values.astype(str).str.lower().isin(
            {"true", "1"}
        )
        selected = selected.loc[mask]
    eligibility_values = selected[eligibility_column]
    eligibility_mask = (
        eligibility_values
        if eligibility_values.dtype == bool
        else eligibility_values.astype(str).str.lower().isin({"true", "1"})
    )
    selected = selected.loc[eligibility_mask]
    selected["_selection_value"] = pd.to_numeric(
        selected[selection_column], errors="coerce"
    )
    selected = selected.loc[
        np.isfinite(selected["_selection_value"])
    ].copy()
    if selected.empty:
        raise ValueError("No development source-specific write rows remain")
    if selection_aggregation == "seed_event_mean":
        seed_anchor = (
            selected.groupby(
                ["model_label", "seed", "anchor_role", "layer", "head"],
                as_index=False,
            )
            .agg(
                seed_anchor_selection_value=("_selection_value", "mean"),
                n_requests=("request_id", "nunique"),
            )
        )
    else:
        request_anchor = (
            selected.groupby(
                [
                    "model_label",
                    "seed",
                    "request_id",
                    "anchor_role",
                    "layer",
                    "head",
                ],
                as_index=False,
            )
            .agg(
                request_anchor_selection_value=("_selection_value", "mean"),
                n_transition_rows=("_selection_value", "size"),
            )
        )
        seed_anchor = (
            request_anchor.groupby(
                ["model_label", "seed", "anchor_role", "layer", "head"],
                as_index=False,
            )
            .agg(
                seed_anchor_selection_value=(
                    "request_anchor_selection_value",
                    "mean",
                ),
                n_requests=("request_id", "nunique"),
            )
        )
    seed_head = (
        seed_anchor.groupby(
            ["model_label", "seed", "layer", "head"], as_index=False
        )
        .agg(
            seed_pooled_selection_value=(
                "seed_anchor_selection_value",
                "mean",
            ),
            n_anchor_roles=("anchor_role", "nunique"),
        )
    )
    ranking = (
        seed_head.groupby(["model_label", "layer", "head"], as_index=False)
        .agg(
            discovery_selection_value=(
                "seed_pooled_selection_value",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
            mean_anchor_roles_per_seed=("n_anchor_roles", "mean"),
        )
        .sort_values(
            [
                "model_label",
                "discovery_selection_value",
                "layer",
                "head",
            ],
            ascending=[True, False, True, True],
        )
        .reset_index(drop=True)
    )
    ranking[f"discovery_{selection_column}"] = ranking[
        "discovery_selection_value"
    ]
    diagnostic_columns = [
        column
        for column in (
            "target_source_attention_mass",
            "target_source_relative_attention_mass",
            "target_source_attention_top1",
            "target_source_attention_unique_top1",
            "target_minus_max_wrong_source_attention_mass",
            "source_specific_ov_write_norm",
        )
        if column in selected.columns and column != selection_column
    ]
    diagnostic_keys = [
        "model_label",
        "seed",
        "request_id",
        "anchor_role",
        "layer",
        "head",
    ]
    for column in diagnostic_columns:
        diagnostic = selected[diagnostic_keys + [column]].copy()
        diagnostic["_diagnostic_value"] = pd.to_numeric(
            diagnostic[column], errors="coerce"
        )
        diagnostic = diagnostic.loc[
            np.isfinite(diagnostic["_diagnostic_value"])
        ]
        if diagnostic.empty:
            continue
        if selection_aggregation == "seed_event_mean":
            seed_anchor_diagnostic = (
                diagnostic.groupby(
                    ["model_label", "seed", "anchor_role", "layer", "head"],
                    as_index=False,
                )
                .agg(_seed_anchor_diagnostic=("_diagnostic_value", "mean"))
            )
        else:
            request_diagnostic = (
                diagnostic.groupby(diagnostic_keys, as_index=False)
                .agg(_request_diagnostic=("_diagnostic_value", "mean"))
            )
            seed_anchor_diagnostic = (
                request_diagnostic.groupby(
                    ["model_label", "seed", "anchor_role", "layer", "head"],
                    as_index=False,
                )
                .agg(_seed_anchor_diagnostic=("_request_diagnostic", "mean"))
            )
        seed_head_diagnostic = (
            seed_anchor_diagnostic.groupby(
                ["model_label", "seed", "layer", "head"],
                as_index=False,
            )
            .agg(_seed_head_diagnostic=("_seed_anchor_diagnostic", "mean"))
        )
        pooled_diagnostic = (
            seed_head_diagnostic.groupby(
                ["model_label", "layer", "head"], as_index=False
            )
            .agg(**{f"discovery_{column}": ("_seed_head_diagnostic", "mean")})
        )
        ranking = ranking.merge(
            pooled_diagnostic,
            on=["model_label", "layer", "head"],
            how="left",
            validate="one_to_one",
        )
    ranking["discovery_rank"] = ranking.groupby("model_label").cumcount() + 1
    ranking["mechanism"] = "retrieval_anchor_localization"
    ranking["query_site_kind"] = (
        "pooled_transition_anchors"
        if anchor_role is None
        else f"expanded_anchor_role:{anchor_role}"
    )
    ranking["selection_metric"] = (
        f"{selection_aggregation}_{selection_column}"
        + (
            ""
            if minimum_layer is None
            else f"_with_minimum_layer_{int(minimum_layer)}"
        )
        + (
            ""
            if maximum_layer is None
            else f"_with_maximum_layer_{int(maximum_layer)}"
        )
    )
    ranking["selection_metric_column"] = selection_column
    ranking["selection_anchor_role"] = anchor_role
    ranking["selection_target_grammar_class"] = target_grammar_class
    ranking["selection_target_retrieval_surface_variant"] = (
        target_retrieval_surface_variant
    )
    ranking["selection_eligibility_scope"] = selection_eligibility_scope
    ranking["selection_aggregation"] = selection_aggregation
    return ranking


def layer_matched_random_controls(
    head_ranking: pd.DataFrame,
    selected_heads: Sequence[tuple[int, int]],
    *,
    repeats: int,
    seed_text: str,
) -> list[list[tuple[int, int]]]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    available = {
        (int(row.layer), int(row.head))
        for row in head_ranking.itertuples(index=False)
    }
    selected = [(int(layer), int(head)) for layer, head in selected_heads]
    by_layer: dict[int, list[int]] = {}
    for layer, head in available - set(selected):
        by_layer.setdefault(layer, []).append(head)
    target_counts: dict[int, int] = {}
    for layer, _head in selected:
        target_counts[layer] = target_counts.get(layer, 0) + 1
    rng = np.random.default_rng(_stable_seed(seed_text))
    controls: list[list[tuple[int, int]]] = []
    combinations = 1
    for layer, count in sorted(target_counts.items()):
        candidate_count = len(by_layer.get(layer, []))
        if candidate_count < count:
            raise ValueError(
                f"Not enough non-selected heads for layer-matched L{layer} control"
            )
        combinations *= math.comb(candidate_count, count)
    if combinations < repeats:
        raise ValueError(
            f"Only {combinations} distinct layer-matched control banks exist; "
            f"requested {repeats}"
        )
    seen: set[tuple[tuple[int, int], ...]] = set()
    attempts = 0
    while len(controls) < repeats:
        attempts += 1
        if attempts > max(1000, repeats * 100):
            raise RuntimeError("Could not draw distinct layer-matched controls")
        bank: list[tuple[int, int]] = []
        for layer, count in sorted(target_counts.items()):
            candidates = np.asarray(sorted(by_layer[layer]), dtype=int)
            chosen = rng.choice(candidates, size=count, replace=False)
            bank.extend((layer, int(head)) for head in chosen)
        key = tuple(sorted(bank))
        if key in seen:
            continue
        seen.add(key)
        controls.append(list(key))
    return controls


def _legacy_layer_matched_random_controls(
    head_ranking: pd.DataFrame,
    selected_heads: Sequence[tuple[int, int]],
    *,
    repeats: int,
    seed_text: str,
) -> list[list[tuple[int, int]]]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    available = {
        (int(row.layer), int(row.head))
        for row in head_ranking.itertuples(index=False)
    }
    selected = [(int(layer), int(head)) for layer, head in selected_heads]
    by_layer: dict[int, list[int]] = {}
    for layer, head in available - set(selected):
        by_layer.setdefault(layer, []).append(head)
    target_counts: dict[int, int] = {}
    for layer, _head in selected:
        target_counts[layer] = target_counts.get(layer, 0) + 1
    rng = np.random.default_rng(_stable_seed(seed_text))
    controls: list[list[tuple[int, int]]] = []
    for _repeat in range(repeats):
        bank: list[tuple[int, int]] = []
        for layer, count in sorted(target_counts.items()):
            candidates = np.asarray(sorted(by_layer.get(layer, [])), dtype=int)
            if len(candidates) < count:
                raise ValueError(
                    f"Not enough non-selected heads for layer-matched L{layer} control"
                )
            chosen = rng.choice(candidates, size=count, replace=False)
            bank.extend((layer, int(head)) for head in chosen)
        controls.append(sorted(bank))
    return controls
def _control_constructible_ranked_heads(
    head_ranking: pd.DataFrame,
    ordered_heads: Sequence[tuple[int, int]],
    *,
    bank_size: int,
) -> list[tuple[int, int]]:
    """Greedily preserve discovery rank while keeping exact controls feasible.

    A layer-matched control must draw distinct heads from the complement of the
    ranked bank.  Consequently no ranked bank may occupy more than half of the
    available heads in any layer.  This constraint is architectural (not a
    post-hoc outcome filter), so it belongs in the frozen discovery plan.
    """

    available_by_layer: dict[int, int] = {}
    for row in head_ranking.itertuples(index=False):
        layer = int(row.layer)
        available_by_layer[layer] = available_by_layer.get(layer, 0) + 1
    layer_counts: dict[int, int] = {}
    selected: list[tuple[int, int]] = []
    for raw_layer, raw_head in ordered_heads:
        layer, head = int(raw_layer), int(raw_head)
        capacity = available_by_layer.get(layer, 0) // 2
        if layer_counts.get(layer, 0) >= capacity:
            continue
        selected.append((layer, head))
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if len(selected) == int(bank_size):
            return selected
    raise ValueError(
        f"Cannot construct ranked bank K={bank_size} under exact-control "
        "per-layer capacity"
    )


def _control_constructible_joint_ranked_heads(
    head_ranking: pd.DataFrame,
    prompt_ordered: Sequence[tuple[int, int]],
    trace_ordered: Sequence[tuple[int, int]],
    *,
    bank_size: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Select equal-weight prompt/trace banks whose union admits controls."""

    available_by_layer: dict[int, int] = {}
    for row in head_ranking.itertuples(index=False):
        layer = int(row.layer)
        available_by_layer[layer] = available_by_layer.get(layer, 0) + 1
    prompt: list[tuple[int, int]] = []
    trace: list[tuple[int, int]] = []
    union: set[tuple[int, int]] = set()
    layer_counts: dict[int, int] = {}

    def add_next(
        ordered: Sequence[tuple[int, int]],
        selected: list[tuple[int, int]],
    ) -> bool:
        selected_set = set(selected)
        for raw_layer, raw_head in ordered:
            head = (int(raw_layer), int(raw_head))
            if head in selected_set:
                continue
            layer = head[0]
            capacity = available_by_layer.get(layer, 0) // 2
            if head not in union and layer_counts.get(layer, 0) >= capacity:
                continue
            selected.append(head)
            if head not in union:
                union.add(head)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            return True
        return False

    # Alternating additions give the two independently ranked mechanisms equal
    # weight in the joint bank while allowing a shared head to count for both.
    while len(prompt) < int(bank_size) or len(trace) < int(bank_size):
        progressed = False
        if len(prompt) < int(bank_size):
            progressed = add_next(prompt_ordered, prompt) or progressed
        if len(trace) < int(bank_size):
            progressed = add_next(trace_ordered, trace) or progressed
        if not progressed:
            raise ValueError(
                f"Cannot construct joint ranked banks K={bank_size} under "
                "exact-control per-layer capacity"
            )
    return prompt, trace


def _bank_attention_mass(
    head_ranking: pd.DataFrame,
    heads: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Summarize the frozen discovery exact-span mass for one head bank."""

    lookup = {
        (int(row.layer), int(row.head)): (
            float(row.discovery_target_raw_mass),
            float(row.discovery_target_relative_mass),
        )
        for row in head_ranking.itertuples(index=False)
    }
    values = [lookup[(int(layer), int(head))] for layer, head in heads]
    raw = np.asarray([value[0] for value in values], dtype=float)
    relative = np.asarray([value[1] for value in values], dtype=float)
    finite_relative = relative[np.isfinite(relative)]
    return {
        "target_needle_raw_mass": float(raw.mean()),
        "target_needle_relative_mass": (
            float(finite_relative.mean()) if len(finite_relative) else float("nan")
        ),
        "relative_mass_defined_heads": int(len(finite_relative)),
        "attention_mass_split": "discovery",
        "attention_mass_aggregation": (
            "mean_across_head_query_weighted_exact_span_scores"
        ),
    }


def _bank_answer_selection_mass(
    head_ranking: pd.DataFrame,
    heads: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Summarize the mechanism-specific frozen answer-query ranking mass."""

    lookup = {
        (int(row.layer), int(row.head)): (
            float(row.discovery_selection_raw_mass),
            float(row.discovery_selection_relative_mass),
            float(row.discovery_selection_broad_score),
            float(row.discovery_selection_broad_coverage),
        )
        for row in head_ranking.itertuples(index=False)
    }
    values = [lookup[(int(layer), int(head))] for layer, head in heads]
    raw = np.asarray([value[0] for value in values], dtype=float)
    relative = np.asarray([value[1] for value in values], dtype=float)
    broad = np.asarray([value[2] for value in values], dtype=float)
    coverage = np.asarray([value[3] for value in values], dtype=float)
    finite_relative = relative[np.isfinite(relative)]
    return {
        "selected_aggregation_raw_mass": float(raw.mean()),
        "selected_aggregation_relative_mass": (
            float(finite_relative.mean()) if len(finite_relative) else float("nan")
        ),
        "selected_aggregation_relative_defined_heads": int(len(finite_relative)),
        "selected_aggregation_broad_score": float(np.nanmean(broad)),
        "selected_aggregation_broad_coverage": float(np.nanmean(coverage)),
        "selected_aggregation_metric": str(
            head_ranking["selection_metric"].iloc[0]
        ),
    }
def global_random_controls(
    head_ranking: pd.DataFrame,
    selected_heads: Sequence[tuple[int, int]],
    *,
    repeats: int,
    seed_text: str,
) -> list[list[tuple[int, int]]]:
    """Draw distinct same-K controls globally, excluding selected heads.

    Unlike exact layer matching, this control does not constrain the layer
    histogram.  It is useful when a literal top-K treatment is sufficiently
    layer-concentrated that exact matching would be combinatorially impossible.
    """

    if repeats < 1:
        raise ValueError("repeats must be positive")
    available = {
        (int(row.layer), int(row.head))
        for row in head_ranking.itertuples(index=False)
    }
    selected = {(int(layer), int(head)) for layer, head in selected_heads}
    candidates = sorted(available - selected)
    bank_size = len(selected)
    if len(candidates) < bank_size:
        raise ValueError(
            "Not enough non-selected heads for a same-K global random control"
        )
    combinations = math.comb(len(candidates), bank_size)
    if combinations < repeats:
        raise ValueError(
            f"Only {combinations} distinct global control banks exist; "
            f"requested {repeats}"
        )
    rng = np.random.default_rng(_stable_seed(seed_text))
    controls: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    attempts = 0
    candidate_indices = np.arange(len(candidates), dtype=int)
    while len(controls) < repeats:
        attempts += 1
        if attempts > max(1000, repeats * 100):
            raise RuntimeError("Could not draw distinct global random controls")
        chosen_indices = rng.choice(
            candidate_indices,
            size=bank_size,
            replace=False,
        )
        bank = tuple(sorted(candidates[int(index)] for index in chosen_indices))
        if bank in seen:
            continue
        seen.add(bank)
        controls.append(list(bank))
    return controls


def control_feasible_ranked_bank(
    head_ranking: pd.DataFrame,
    *,
    bank_size: int,
    control_repeats: int = 1,
) -> list[tuple[int, int]]:
    """Take the highest ranked heads subject to exact control feasibility.

    Exact layer matching without reusing selected heads requires enough
    non-selected combinations in every represented layer to construct all
    requested *distinct* control banks.  Applying that constraint while
    traversing the frozen global ranking prevents a concentrated top-K from
    making its own preregistered controls impossible or duplicated.
    """

    if int(bank_size) < 1:
        raise ValueError("bank_size must be positive")
    if int(control_repeats) < 1:
        raise ValueError("control_repeats must be positive")
    available_per_layer = (
        head_ranking.groupby("layer")["head"].nunique().astype(int).to_dict()
    )
    selected_per_layer: dict[int, int] = {}
    selected: list[tuple[int, int]] = []
    ordered = head_ranking.sort_values(
        ["discovery_rank", "layer", "head"]
    )
    for row in ordered.itertuples(index=False):
        layer = int(row.layer)
        head = int(row.head)
        available = int(available_per_layer[layer])
        capacity = max(
            (
                count
                for count in range(1, available // 2 + 1)
                if math.comb(available - count, count)
                >= int(control_repeats)
            ),
            default=0,
        )
        if selected_per_layer.get(layer, 0) >= capacity:
            continue
        selected.append((layer, head))
        selected_per_layer[layer] = selected_per_layer.get(layer, 0) + 1
        if len(selected) == int(bank_size):
            return selected
    raise ValueError(
        "Not enough ranked heads under distinct exact layer-matched-control "
        f"capacity for K={bank_size}, repeats={control_repeats}"
    )


def strict_ranked_bank(
    head_ranking: pd.DataFrame, *, bank_size: int
) -> list[tuple[int, int]]:
    """Return the literal top-K heads; controls never alter treatment."""

    if int(bank_size) < 1:
        raise ValueError("bank_size must be positive")
    needed = {"layer", "head", "discovery_rank"}
    missing = sorted(needed - set(head_ranking.columns))
    if missing:
        raise ValueError(f"Head ranking is missing {missing}")
    ordered = head_ranking.sort_values(
        ["discovery_rank", "layer", "head"]
    ).drop_duplicates(["layer", "head"])
    if len(ordered) < int(bank_size):
        raise ValueError(
            f"Only {len(ordered)} distinct ranked heads exist for K={bank_size}"
        )
    return [
        (int(row.layer), int(row.head))
        for row in ordered.head(int(bank_size)).itertuples(index=False)
    ]


def ranked_bank_with_layer_profile(
    head_ranking: pd.DataFrame,
    *,
    layer_profile: dict[int, int],
) -> list[tuple[int, int]]:
    """Select the best ranked heads subject to an exact per-layer quota.

    This is a diagnostic treatment rather than a random control: scores come
    from the supplied ranking, while the layer histogram is frozen from a
    reference bank selected at another query site.
    """

    needed = {"layer", "head", "discovery_rank"}
    missing = sorted(needed - set(head_ranking.columns))
    if missing:
        raise ValueError(f"Head ranking is missing {missing}")
    normalized = {
        int(layer): int(count) for layer, count in layer_profile.items()
    }
    if not normalized or any(count < 0 for count in normalized.values()):
        raise ValueError("layer_profile must contain non-negative quotas")
    if sum(normalized.values()) < 1:
        raise ValueError("layer_profile must select at least one head")
    ordered = head_ranking.sort_values(
        ["discovery_rank", "layer", "head"]
    ).drop_duplicates(["layer", "head"])
    chosen: list[tuple[int, int]] = []
    for layer in sorted(normalized):
        quota = normalized[layer]
        available = ordered.loc[ordered["layer"].astype(int).eq(layer)]
        if len(available) < quota:
            raise ValueError(
                f"Layer {layer} has {len(available)} ranked heads for quota {quota}"
            )
        chosen.extend(
            (int(row.layer), int(row.head))
            for row in available.head(quota).itertuples(index=False)
        )
    rank_lookup = {
        (int(row.layer), int(row.head)): int(row.discovery_rank)
        for row in ordered.itertuples(index=False)
    }
    return sorted(chosen, key=lambda head: (rank_lookup[head], *head))


def _build_legacy_causal_plan(
    attention_csv: str | Path,
    output_dir: str | Path,
    *,
    config: V5Config,
) -> dict[str, Path]:
    config.validate()
    attention = pd.read_csv(attention_csv)
    legacy_mechanisms = tuple(
        mechanism
        for mechanism in ("targeted_retrieval", "progress_transition")
        if mechanism in set(attention["mechanism"].astype(str))
    )
    if not legacy_mechanisms:
        raise ValueError("Legacy attention plan has no registered mechanisms")
    ranking = pd.concat(
        [
            rank_mechanism_heads(attention, mechanism=mechanism)
            for mechanism in legacy_mechanisms
        ],
        ignore_index=True,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_rows: list[dict[str, Any]] = []
    skipped_banks: list[dict[str, Any]] = []
    for (model_label, mechanism), model_frame in ranking.groupby(
        ["model_label", "mechanism"], sort=True
    ):
        ordered = [
            (int(row.layer), int(row.head))
            for row in model_frame.sort_values("discovery_rank").itertuples()
        ]
        for bank_size in config.causal_head_bank_sizes:
            if bank_size > len(ordered):
                continue
            chosen = ordered[:bank_size]
            # The registered ranked treatment is scientifically meaningful at
            # every available K even when a disjoint, exactly layer-matched
            # random bank is combinatorially impossible.  Do not silently
            # discard the treatment together with an unavailable control.
            plan_rows.append(
                {
                    "model_label": model_label,
                    "mechanism": mechanism,
                    "query_site_kind": str(
                        model_frame["query_site_kind"].iloc[0]
                    ),
                    "experiment_id": f"{mechanism}_head_ablation",
                    "condition": f"{mechanism}_ranked",
                    "bank_size": bank_size,
                    "repeat": 0,
                    "heads": json.dumps(chosen),
                    **_bank_attention_mass(model_frame, chosen),
                }
            )
            try:
                controls = _legacy_layer_matched_random_controls(
                    model_frame,
                    chosen,
                    repeats=config.causal_random_controls,
                    seed_text=f"v5:{model_label}:{mechanism}:K{bank_size}",
                )
            except ValueError as error:
                skipped_banks.append(
                    {
                        "model_label": str(model_label),
                        "mechanism": str(mechanism),
                        "bank_size": int(bank_size),
                        "ranked_treatment_included": True,
                        "control_status": (
                            "not_constructible_disjoint_exact_layer_match"
                        ),
                        "reason": str(error),
                    }
                )
                continue
            for repeat, control in enumerate(controls, start=1):
                plan_rows.append(
                    {
                        "model_label": model_label,
                        "mechanism": mechanism,
                        "query_site_kind": str(
                            model_frame["query_site_kind"].iloc[0]
                        ),
                        "experiment_id": f"{mechanism}_head_ablation",
                        "condition": "layer_matched_random",
                        "bank_size": bank_size,
                        "repeat": repeat,
                        "heads": json.dumps(control),
                        **_bank_attention_mass(model_frame, control),
                    }
                )
    paths = {
        "ranking": output / "discovery_head_ranking.csv",
        "plan": output / "causal_plan.csv",
        "ledger": output / "causal_ledger.csv",
        "audit": output / "causal_plan_audit.json",
    }
    ranking.to_csv(paths["ranking"], index=False)
    pd.DataFrame(
        plan_rows,
        columns=[
            "model_label",
            "mechanism",
            "query_site_kind",
            "experiment_id",
            "condition",
            "bank_size",
            "repeat",
            "heads",
            "target_needle_raw_mass",
            "target_needle_relative_mass",
            "relative_mass_defined_heads",
            "attention_mass_split",
            "attention_mass_aggregation",
        ],
    ).to_csv(paths["plan"], index=False)
    causal_ledger_frame().to_csv(paths["ledger"], index=False)
    paths["audit"].write_text(
        json.dumps(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "selection_split": "discovery",
                "selection_cohort": "one_to_one",
                "selection_counts": list(config.counts),
                "confirmation_used_for_selection": False,
                "selection_metric": (
                    "query-weighted mean raw attention mass to the semantic "
                    "target prompt-record span"
                ),
                "mechanisms": {
                    "targeted_retrieval": {
                        "query": "marker_end:k",
                        "target": "prompt record matching accepted city k",
                    },
                    "progress_transition": {
                        "query": "item_end:k",
                        "target": "prompt record matching accepted city k+1",
                    },
                },
                "primary_ablation_scope": "mechanism query position only",
                "registered_bank_sizes": list(config.causal_head_bank_sizes),
                "ranked_treatment_policy": (
                    "include every registered K with enough ranked heads"
                ),
                "random_control": (
                    "disjoint random heads with the same selected-head count "
                    "in every layer; audited as structurally unavailable "
                    "rather than replaced by overlapping controls"
                ),
                "skipped_banks": skipped_banks,
                "attention_source": str(Path(attention_csv).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _build_source_write_causal_plan(
    source_writes_csv: str | Path,
    output_dir: str | Path,
    *,
    config: V5Config,
    bank_size: int | None = None,
    anchor_role: str | None = None,
    minimum_layer: int | None = None,
    maximum_layer: int | None = None,
    selection_metric: str | None = None,
    target_grammar_class: str | None = None,
    target_retrieval_surface_variant: str | None = None,
    selection_eligibility_scope: str = "primary",
    selection_aggregation: str = "request_equal",
    allow_incomplete_development_smoke: bool = False,
    include_random_controls: bool = True,
    random_control_matching: str = "layer_matched",
    confirmation_plan: bool = False,
    full_panel_plan: bool = False,
) -> dict[str, Path]:
    """Freeze cross-fitted banks at one audited retrieval query site."""

    config.validate()
    registered_bank_size = (
        int(config.causal_primary_bank_size)
        if bank_size is None
        else int(bank_size)
    )
    if registered_bank_size < 1:
        raise ValueError("bank_size must be positive")
    selection_eligibility_scope = str(selection_eligibility_scope)
    if selection_eligibility_scope not in {"primary", "local"}:
        raise ValueError(
            "selection_eligibility_scope must be 'primary' or 'local'"
        )
    selection_aggregation = str(selection_aggregation)
    if selection_aggregation not in {"request_equal", "seed_event_mean"}:
        raise ValueError(
            "selection_aggregation must be 'request_equal' or 'seed_event_mean'"
        )
    random_control_matching = str(random_control_matching)
    if random_control_matching not in {"layer_matched", "global"}:
        raise ValueError(
            "random_control_matching must be 'layer_matched' or 'global'"
        )
    confirmation_seeds = sorted(
        {int(value) for value in config.causal_confirmation_seeds}
    )
    if confirmation_plan and not confirmation_seeds:
        raise ValueError(
            "A confirmation plan requires non-empty causal_confirmation_seeds"
        )
    if confirmation_plan and allow_incomplete_development_smoke:
        raise ValueError(
            "A frozen confirmation plan cannot also be labeled a development smoke"
        )
    if full_panel_plan and confirmation_plan:
        raise ValueError(
            "A full-panel development plan cannot also be a confirmation plan"
        )
    if full_panel_plan and allow_incomplete_development_smoke:
        raise ValueError(
            "A full-panel development plan cannot also be labeled a smoke"
        )
    if minimum_layer is not None and int(minimum_layer) < 0:
        raise ValueError("minimum_layer must be non-negative")
    if maximum_layer is not None and int(maximum_layer) < 0:
        raise ValueError("maximum_layer must be non-negative")
    if (
        minimum_layer is not None
        and maximum_layer is not None
        and int(minimum_layer) > int(maximum_layer)
    ):
        raise ValueError("minimum_layer cannot exceed maximum_layer")
    selection_policy = (
        str(config.causal_head_selection_metric)
        if selection_metric is None
        else str(selection_metric)
    )
    selection_column = _head_selection_column(selection_policy)
    if (
        selection_column != "source_specific_ov_write_norm"
        and anchor_role is None
    ):
        raise ValueError(
            "Attention-selected head banks require one explicit anchor_role "
            "so selection and ablation cannot drift across query positions"
        )
    selection_site_scope = (
        "exact_single_query_anchor"
        if anchor_role is not None
        else "legacy_pooled_query_anchors"
    )
    source = Path(source_writes_csv)
    if source.is_dir():
        shard_files = sorted((source / "shards").glob("*.jsonl"))
        if not shard_files:
            raise ValueError(f"Source-specific write directory has no shards: {source}")
        records = []
        for shard in shard_files:
            with shard.open("r", encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
        writes = pd.DataFrame(records)
    elif source.suffix.lower() in {".jsonl", ".ndjson"}:
        with source.open("r", encoding="utf-8") as handle:
            writes = pd.DataFrame(
                json.loads(line) for line in handle if line.strip()
            )
    else:
        writes = pd.read_csv(source)
    if writes.empty:
        raise ValueError("Source-specific write table is empty")
    eligibility_column = (
        "primary_anchor_eligible"
        if selection_eligibility_scope == "primary"
        else "local_anchor_eligible"
    )
    if eligibility_column not in writes:
        raise ValueError(
            f"Source-specific writes must record {eligibility_column}"
        )
    eligibility_values = writes[eligibility_column]
    eligibility_mask = (
        eligibility_values
        if eligibility_values.dtype == bool
        else eligibility_values.astype(str).str.lower().isin({"true", "1"})
    )
    writes = writes.loc[eligibility_mask].copy()
    if writes.empty:
        raise ValueError(
            f"No {selection_eligibility_scope} causal-anchor source writes remain"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ranking_frames: list[pd.DataFrame] = []
    plan_rows: list[dict[str, Any]] = []
    skipped_banks: list[dict[str, Any]] = []
    fold_registry: dict[str, dict[int, int]] = {}
    source_seed_coverage: dict[str, list[int]] = {}
    missing_source_seeds: dict[str, list[int]] = {}
    selection_scope_seed_coverage: dict[str, list[int]] = {}
    missing_selection_scope_seeds: dict[str, list[int]] = {}
    development = {int(value) for value in config.causal_development_seeds}
    for model_label, raw_model_frame in writes.groupby("model_label", sort=True):
        model_frame = raw_model_frame.loc[
            pd.to_numeric(raw_model_frame["seed"], errors="coerce").isin(development)
        ].copy()
        observed_seeds = {
            int(value)
            for value in pd.to_numeric(model_frame["seed"], errors="coerce")
            if pd.notna(value)
        }
        missing_seeds = sorted(development - observed_seeds)
        missing_source_seeds[str(model_label)] = missing_seeds
        if (
            missing_seeds
            and not allow_incomplete_development_smoke
            and not confirmation_plan
            and not full_panel_plan
        ):
            raise ValueError(
                f"Source-specific writes for {model_label} do not cover every "
                f"causal development seed; missing={missing_seeds}"
            )
        seeds = sorted(
            observed_seeds
            if (
                allow_incomplete_development_smoke
                or confirmation_plan
                or full_panel_plan
            )
            else development
        )
        if len(seeds) < 2:
            raise ValueError(
                f"At least two source-write seeds are required for {model_label}"
            )
        source_seed_coverage[str(model_label)] = sorted(observed_seeds)
        scope_frame = model_frame
        if anchor_role is not None:
            scope_frame = scope_frame.loc[
                scope_frame.apply(
                    lambda row: _source_row_has_anchor_role(
                        row, str(anchor_role)
                    ),
                    axis=1,
                )
            ]
        if target_grammar_class is not None:
            if "target_grammar_class" in scope_frame:
                grammar_values = scope_frame["target_grammar_class"].astype(
                    str
                )
            elif "grammar_pair" in scope_frame:
                grammar_values = scope_frame["grammar_pair"].map(
                    _target_grammar_class
                )
            else:
                raise ValueError(
                    "Target-grammar selection requires target_grammar_class "
                    "or grammar_pair in source writes"
                )
            scope_frame = scope_frame.loc[
                grammar_values.eq(str(target_grammar_class))
            ]
        if target_retrieval_surface_variant is not None:
            if "target_retrieval_surface_variant" not in scope_frame:
                raise ValueError(
                    "Target-surface selection requires "
                    "target_retrieval_surface_variant in source writes"
                )
            scope_frame = scope_frame.loc[
                scope_frame["target_retrieval_surface_variant"]
                .astype(str)
                .eq(str(target_retrieval_surface_variant))
            ]
        scope_observed_seeds = {
            int(value)
            for value in pd.to_numeric(scope_frame["seed"], errors="coerce")
            if pd.notna(value)
        }
        selection_scope_seed_coverage[str(model_label)] = sorted(
            scope_observed_seeds
        )
        missing_selection_scope_seeds[str(model_label)] = sorted(
            development - scope_observed_seeds
        )
        if confirmation_plan:
            fold_definitions = [(0, seeds, confirmation_seeds)]
            fold_registry[str(model_label)] = {
                seed: 0 for seed in confirmation_seeds
            }
        elif full_panel_plan:
            # Rank strictly on the causal-development cohort, but allow one
            # frozen bank to be evaluated on both the discovery and registered
            # confirmation cohorts.  This preserves the split while letting
            # the full-panel report reuse exactly the same intervention arms.
            full_panel_seeds = sorted(development | set(confirmation_seeds))
            fold_definitions = [(0, seeds, full_panel_seeds)]
            fold_registry[str(model_label)] = {
                seed: 0 for seed in full_panel_seeds
            }
        else:
            fold_count = min(int(config.causal_crossfit_folds), len(seeds))
            seed_folds = {
                seed: (index % fold_count if fold_count > 1 else 0)
                for index, seed in enumerate(seeds)
            }
            fold_registry[str(model_label)] = seed_folds
            fold_definitions = [
                (
                    fold,
                    (
                        seeds
                        if fold_count == 1
                        else [seed for seed in seeds if seed_folds[seed] != fold]
                    ),
                    [seed for seed in seeds if seed_folds[seed] == fold],
                )
                for fold in range(fold_count)
            ]
        for fold, training, validation in fold_definitions:
            ranking = rank_pooled_source_specific_heads(
                model_frame,
                development_seeds=training,
                anchor_role=anchor_role,
                minimum_layer=minimum_layer,
                maximum_layer=maximum_layer,
                selection_metric=selection_policy,
                target_grammar_class=target_grammar_class,
                target_retrieval_surface_variant=(
                    target_retrieval_surface_variant
                ),
                selection_eligibility_scope=selection_eligibility_scope,
                selection_aggregation=selection_aggregation,
            )
            ranking["fold"] = int(fold)
            ranking["training_seeds"] = json.dumps(training)
            ranking["validation_seeds"] = json.dumps(validation)
            ranking_frames.append(ranking)
            try:
                chosen = strict_ranked_bank(
                    ranking, bank_size=registered_bank_size
                )
            except ValueError as error:
                skipped_banks.append(
                    {
                        "model_label": str(model_label),
                        "fold": int(fold),
                        "bank_size": registered_bank_size,
                        "reason": str(error),
                    }
                )
                continue
            controls: list[list[tuple[int, int]]] = []
            if include_random_controls:
                try:
                    control_builder = (
                        layer_matched_random_controls
                        if random_control_matching == "layer_matched"
                        else global_random_controls
                    )
                    controls = control_builder(
                        ranking,
                        chosen,
                        repeats=config.causal_random_controls,
                        seed_text=(
                            f"v5-reboot:{model_label}:"
                            f"confirmation={confirmation_plan}:"
                            f"full_panel={full_panel_plan}:fold{fold}:"
                            f"K{registered_bank_size}:role={anchor_role}:"
                            f"grammar={target_grammar_class}:"
                            f"surface={target_retrieval_surface_variant}:"
                            f"metric={selection_column}:min_layer={minimum_layer}:"
                            f"max_layer={maximum_layer}"
                        ),
                    )
                except ValueError as error:
                    skipped_banks.append(
                        {
                            "model_label": str(model_label),
                            "fold": int(fold),
                            "bank_size": registered_bank_size,
                            "reason": str(error),
                        }
                    )
                    continue
            common = {
                "model_label": model_label,
                "mechanism": "retrieval_anchor_localization",
                "query_site_kind": (
                    "grammar_aware_transition_anchor"
                    if anchor_role is None
                    else f"grammar_aware_transition_anchor:{anchor_role}"
                ),
                "experiment_id": (
                    "retrieval_anchor_head_ablation_confirmation"
                    if confirmation_plan
                    else (
                        "retrieval_anchor_head_ablation_full_panel"
                        if full_panel_plan
                        else "retrieval_anchor_head_ablation"
                    )
                ),
                "fold": int(fold),
                "training_seeds": json.dumps(training),
                "validation_seeds": json.dumps(validation),
                "bank_size": registered_bank_size,
                "bank_selection_policy": (
                    "literal_global_top_k_controls_cannot_change_treatment"
                ),
                "selection_metric": (
                    f"{selection_aggregation}_{selection_column}"
                ),
                "selection_metric_column": selection_column,
                "selection_anchor_role": anchor_role,
                "selection_target_grammar_class": target_grammar_class,
                "selection_target_retrieval_surface_variant": (
                    target_retrieval_surface_variant
                ),
                "selection_site_scope": selection_site_scope,
                "selection_eligibility_scope": selection_eligibility_scope,
                "selection_aggregation": selection_aggregation,
                "random_control_matching": random_control_matching,
            }
            plan_rows.append(
                {
                    **common,
                    "condition": "selected_bank",
                    "repeat": 0,
                    "heads": json.dumps(chosen),
                    "bank_sha256": hashlib.sha256(
                        json.dumps(chosen).encode("utf-8")
                    ).hexdigest(),
                }
            )
            for repeat, control in enumerate(controls, start=1):
                plan_rows.append(
                    {
                        **common,
                        "condition": (
                            "layer_matched_random"
                            if random_control_matching == "layer_matched"
                            else "global_random"
                        ),
                        "repeat": repeat,
                        "heads": json.dumps(control),
                        "bank_sha256": hashlib.sha256(
                            json.dumps(control).encode("utf-8")
                        ).hexdigest(),
                    }
                )
    if skipped_banks:
        raise ValueError(f"Could not construct every registered bank: {skipped_banks}")
    ranking = pd.concat(ranking_frames, ignore_index=True)
    paths = {
        "ranking": output / "crossfit_source_specific_head_ranking.csv",
        "plan": output / "retrieval_anchor_bank_plan.csv",
        "ledger": output / "causal_ledger.csv",
        "audit": output / "causal_plan_audit.json",
    }
    ranking.to_csv(paths["ranking"], index=False)
    pd.DataFrame(
        plan_rows,
        columns=[
            "model_label",
            "mechanism",
            "query_site_kind",
            "experiment_id",
            "fold",
            "training_seeds",
            "validation_seeds",
            "condition",
            "bank_size",
            "bank_selection_policy",
            "selection_metric",
            "selection_metric_column",
            "selection_anchor_role",
            "selection_target_grammar_class",
            "selection_target_retrieval_surface_variant",
            "selection_site_scope",
            "selection_eligibility_scope",
            "selection_aggregation",
            "random_control_matching",
            "repeat",
            "heads",
            "bank_sha256",
        ],
    ).to_csv(paths["plan"], index=False)
    causal_ledger_frame().to_csv(paths["ledger"], index=False)
    paths["audit"].write_text(
        json.dumps(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "selection_split": (
                    "frozen_development_ranking_to_fresh_causal_confirmation"
                    if confirmation_plan
                    else (
                        "frozen_development_ranking_reused_on_full_"
                        "registered_panel"
                        if full_panel_plan
                        else (
                            "incomplete_development_technical_smoke"
                            if allow_incomplete_development_smoke
                            else "all_previously_inspected_seeds_are_development"
                        )
                    )
                ),
                "selection_cohort": "grammar_aware_local_transition",
                "selection_counts": list(config.counts),
                "confirmation_used_for_selection": False,
                "confirmation_plan": bool(confirmation_plan),
                "full_panel_plan": bool(full_panel_plan),
                "confirmation_validation_seeds": (
                    confirmation_seeds if confirmation_plan else []
                ),
                "registered_confirmation_seeds": confirmation_seeds,
                "selection_metric": (
                    f"seed_first_equal_anchor_mean_{selection_column}"
                ),
                "selection_metric_column": selection_column,
                "registered_bank_size": registered_bank_size,
                "selection_anchor_role": anchor_role,
                "selection_target_grammar_class": target_grammar_class,
                "selection_target_retrieval_surface_variant": (
                    target_retrieval_surface_variant
                ),
                "selection_site_scope": selection_site_scope,
                "selection_eligibility_scope": selection_eligibility_scope,
                "selection_aggregation": selection_aggregation,
                "representation_guided_minimum_layer": minimum_layer,
                "representation_guided_maximum_layer": maximum_layer,
                "mechanism": {
                    "name": "retrieval_anchor_localization",
                    "queries": (
                        "one exact expanded anchor role per frozen plan"
                        if anchor_role is not None
                        else "pooled deduplicated grammar-aware anchors"
                    ),
                    "target": "identical next-city token span at every anchor",
                },
                "development_seeds": sorted(development),
                "source_seed_coverage": source_seed_coverage,
                "missing_source_seeds": missing_source_seeds,
                "selection_scope_seed_coverage": selection_scope_seed_coverage,
                "missing_selection_scope_seeds": missing_selection_scope_seeds,
                "formal_inference_eligible": bool(
                    include_random_controls
                    and not full_panel_plan
                    and (
                        (
                            confirmation_plan
                            and bool(confirmation_seeds)
                            and not (set(confirmation_seeds) & development)
                            and all(
                                len(values) >= 2
                                for values in selection_scope_seed_coverage.values()
                            )
                        )
                        or (
                            not confirmation_plan
                            and not allow_incomplete_development_smoke
                            and not any(missing_source_seeds.values())
                        )
                    )
                ),
                "registered_confirmation_subcohort_eligible": bool(
                    include_random_controls
                    and bool(confirmation_seeds)
                    and not (set(confirmation_seeds) & development)
                    and all(
                        len(values) >= 2
                        for values in selection_scope_seed_coverage.values()
                    )
                ),
                "crossfit_folds": fold_registry,
                "primary_ablation_scope": "mechanism query position only",
                "bank_selection_policy": (
                    f"literal global top-K by same-site {selection_column}; "
                    "control construction is attempted only after treatment "
                    "is frozen and can never substitute a selected head"
                ),
                "random_control": (
                    (
                        "distinct banks with exact selected-head counts per layer; "
                        "selected heads excluded"
                        if random_control_matching == "layer_matched"
                        else "distinct same-K banks sampled globally from all "
                        "non-selected heads; selected heads excluded"
                    )
                    if include_random_controls
                    else "deferred during explicitly non-inferential exact-site localization"
                ),
                "random_control_matching": random_control_matching,
                "random_controls_included": bool(include_random_controls),
                "skipped_banks": skipped_banks,
                "source_writes": str(source.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def build_causal_plan(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    config: V5Config,
    bank_size: int | None = None,
    anchor_role: str | None = None,
    minimum_layer: int | None = None,
    maximum_layer: int | None = None,
    selection_metric: str | None = None,
    target_grammar_class: str | None = None,
    target_retrieval_surface_variant: str | None = None,
    selection_eligibility_scope: str = "primary",
    selection_aggregation: str = "request_equal",
    allow_incomplete_development_smoke: bool = False,
    include_random_controls: bool = True,
    random_control_matching: str = "layer_matched",
    confirmation_plan: bool = False,
    full_panel_plan: bool = False,
) -> dict[str, Path]:
    """Build either a legacy attention plan or a source-write causal plan.

    The two research branches evolved the public entry point around different
    audited inputs.  Dispatching from the registered source-write column keeps
    existing attention-plan callers reproducible while preserving the rebooted
    source-specific workflow.
    """

    columns = set(pd.read_csv(source_csv, nrows=0).columns)
    if "source_specific_ov_write_norm" not in columns:
        return _build_legacy_causal_plan(
            source_csv,
            output_dir,
            config=config,
        )
    return _build_source_write_causal_plan(
        source_csv,
        output_dir,
        config=config,
        bank_size=bank_size,
        anchor_role=anchor_role,
        minimum_layer=minimum_layer,
        maximum_layer=maximum_layer,
        selection_metric=selection_metric,
        target_grammar_class=target_grammar_class,
        target_retrieval_surface_variant=target_retrieval_surface_variant,
        selection_eligibility_scope=selection_eligibility_scope,
        selection_aggregation=selection_aggregation,
        allow_incomplete_development_smoke=allow_incomplete_development_smoke,
        include_random_controls=include_random_controls,
        random_control_matching=random_control_matching,
        confirmation_plan=confirmation_plan,
        full_panel_plan=full_panel_plan,
    )


def build_answer_query_causal_plan(
    attention_csv: str | Path,
    output_dir: str | Path,
    *,
    config: V5Config,
) -> dict[str, Path]:
    """Build discovery-frozen answer-query banks and exact controls."""

    config.validate()
    attention = pd.read_csv(attention_csv)
    answer_mechanisms = (
        "answer_prompt_aggregation",
        "answer_trace_aggregation",
    )
    ranking = pd.concat(
        [
            rank_answer_query_heads(
                attention, mechanism=mechanism, split="discovery"
            )
            for mechanism in answer_mechanisms
        ],
        ignore_index=True,
    )
    confirmation = attention.loc[
        attention["split"].astype(str).str.lower().eq("confirmation")
    ].copy()
    confirmation_ranking = (
        confirmation.groupby(["model_label", "layer", "head"], as_index=False)
        .agg(
            confirmation_target_raw_mass=("target_needle_raw_mass", "mean"),
            confirmation_target_relative_mass=(
                "target_needle_relative_mass", "mean"
            ),
            confirmation_trace_raw_mass=("trace_item_raw_mass", "mean"),
            confirmation_trace_relative_mass=("trace_item_relative_mass", "mean"),
            confirmation_prompt_broad_score=("prompt_broad_score", "mean"),
            confirmation_prompt_broad_coverage=(
                "prompt_broad_coverage",
                "mean",
            ),
            confirmation_trace_broad_score=("trace_broad_score", "mean"),
            confirmation_trace_broad_coverage=(
                "trace_broad_coverage",
                "mean",
            ),
            confirmation_queries=("target_needle_raw_mass", "size"),
        )
        if not confirmation.empty
        else pd.DataFrame()
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_rows: list[dict[str, Any]] = []
    skipped_banks: list[dict[str, Any]] = []
    for (model_label, mechanism), model_frame in ranking.groupby(
        ["model_label", "mechanism"], sort=True
    ):
        ordered = [
            (int(row.layer), int(row.head))
            for row in model_frame.sort_values("discovery_rank").itertuples()
        ]
        confirmation_lookup: dict[
            tuple[int, int], tuple[float, float, float, float, float, float, float, float]
        ] = {}
        if not confirmation_ranking.empty:
            model_confirmation = confirmation_ranking.loc[
                confirmation_ranking["model_label"].astype(str).eq(
                    str(model_label)
                )
            ]
            confirmation_lookup = {
                (int(row.layer), int(row.head)): (
                    float(row.confirmation_target_raw_mass),
                    float(row.confirmation_target_relative_mass),
                    float(row.confirmation_trace_raw_mass),
                    float(row.confirmation_trace_relative_mass),
                    float(row.confirmation_prompt_broad_score),
                    float(row.confirmation_prompt_broad_coverage),
                    float(row.confirmation_trace_broad_score),
                    float(row.confirmation_trace_broad_coverage),
                )
                for row in model_confirmation.itertuples(index=False)
            }

        def bank_row(
            heads: Sequence[tuple[int, int]],
            *,
            condition: str,
            bank_size: int,
            repeat: int,
        ) -> dict[str, Any]:
            discovery_mass = _bank_attention_mass(model_frame, heads)
            confirmation_values = [
                confirmation_lookup.get(
                    (int(layer), int(head)),
                    (np.nan,) * 8,
                )
                for layer, head in heads
            ]
            confirmation_raw = np.asarray(
                [value[0] for value in confirmation_values], dtype=float
            )
            confirmation_relative = np.asarray(
                [value[1] for value in confirmation_values], dtype=float
            )
            confirmation_trace_raw = np.asarray(
                [value[2] for value in confirmation_values], dtype=float
            )
            confirmation_trace_relative = np.asarray(
                [value[3] for value in confirmation_values], dtype=float
            )
            confirmation_prompt_broad = np.asarray(
                [value[4] for value in confirmation_values], dtype=float
            )
            confirmation_prompt_coverage = np.asarray(
                [value[5] for value in confirmation_values], dtype=float
            )
            confirmation_trace_broad = np.asarray(
                [value[6] for value in confirmation_values], dtype=float
            )
            confirmation_trace_coverage = np.asarray(
                [value[7] for value in confirmation_values], dtype=float
            )
            selection_confirmation_raw = (
                confirmation_raw
                if mechanism == "answer_prompt_aggregation"
                else confirmation_trace_raw
            )
            selection_confirmation_relative = (
                confirmation_relative
                if mechanism == "answer_prompt_aggregation"
                else confirmation_trace_relative
            )
            selection_confirmation_broad = (
                confirmation_prompt_broad
                if mechanism == "answer_prompt_aggregation"
                else confirmation_trace_broad
            )
            selection_confirmation_coverage = (
                confirmation_prompt_coverage
                if mechanism == "answer_prompt_aggregation"
                else confirmation_trace_coverage
            )
            return {
                "model_label": model_label,
                "mechanism": mechanism,
                "query_site_kind": "answer_query_v3",
                "experiment_id": "answer_query_head_ablation_v5_broad_factorial",
                "condition": condition,
                "bank_size": int(bank_size),
                "repeat": int(repeat),
                "heads": json.dumps(heads),
                **discovery_mass,
                **_bank_answer_selection_mass(model_frame, heads),
                "confirmation_target_needle_raw_mass": (
                    float(np.nanmean(confirmation_raw))
                    if np.isfinite(confirmation_raw).any()
                    else np.nan
                ),
                "confirmation_target_needle_relative_mass": (
                    float(np.nanmean(confirmation_relative))
                    if np.isfinite(confirmation_relative).any()
                    else np.nan
                ),
                "confirmation_mass_split": "confirmation_descriptive_only",
                "confirmation_selected_aggregation_raw_mass": (
                    float(np.nanmean(selection_confirmation_raw))
                    if np.isfinite(selection_confirmation_raw).any()
                    else np.nan
                ),
                "confirmation_selected_aggregation_relative_mass": (
                    float(np.nanmean(selection_confirmation_relative))
                    if np.isfinite(selection_confirmation_relative).any()
                    else np.nan
                ),
                "confirmation_selected_aggregation_broad_score": (
                    float(np.nanmean(selection_confirmation_broad))
                    if np.isfinite(selection_confirmation_broad).any()
                    else np.nan
                ),
                "confirmation_selected_aggregation_broad_coverage": (
                    float(np.nanmean(selection_confirmation_coverage))
                    if np.isfinite(selection_confirmation_coverage).any()
                    else np.nan
                ),
            }

        for bank_size in config.causal_head_bank_sizes:
            if bank_size > len(ordered):
                continue
            chosen = _control_constructible_ranked_heads(
                model_frame,
                ordered,
                bank_size=int(bank_size),
            )
            plan_rows.append(
                bank_row(
                    chosen,
                    condition=f"{mechanism}_ranked",
                    bank_size=bank_size,
                    repeat=0,
                )
            )
            try:
                controls = _legacy_layer_matched_random_controls(
                    model_frame,
                    chosen,
                    repeats=config.causal_random_controls,
                    seed_text=(
                        f"v5-answer-query:{model_label}:{mechanism}:K{bank_size}"
                    ),
                )
            except ValueError as error:
                skipped_banks.append(
                    {
                        "model_label": str(model_label),
                        "mechanism": str(mechanism),
                        "bank_size": int(bank_size),
                        "ranked_treatment_included": True,
                        "control_status": (
                            "not_constructible_disjoint_exact_layer_match"
                        ),
                        "reason": str(error),
                    }
                )
                continue
            for repeat, control in enumerate(controls, start=1):
                plan_rows.append(
                    bank_row(
                        control,
                        condition=f"{mechanism}_layer_matched_random",
                        bank_size=bank_size,
                        repeat=repeat,
                    )
                )

    # The answer readout has two potentially redundant routes: direct prompt
    # aggregation and aggregation over the generated reasoning trace.  Report
    # their joint intervention as a third frozen treatment rather than asking
    # readers to infer it from two independent runs.
    for model_label, model_ranking in ranking.groupby("model_label", sort=True):
        mechanism_frames = {
            str(mechanism): frame.sort_values("discovery_rank").reset_index(
                drop=True
            )
            for mechanism, frame in model_ranking.groupby("mechanism", sort=True)
        }
        if set(mechanism_frames) != set(answer_mechanisms):
            raise ValueError(
                f"Answer aggregation mechanisms are incomplete for {model_label}: "
                f"{sorted(mechanism_frames)}"
            )
        prompt_frame = mechanism_frames["answer_prompt_aggregation"]
        trace_frame = mechanism_frames["answer_trace_aggregation"]
        prompt_ordered = [
            (int(row.layer), int(row.head))
            for row in prompt_frame.itertuples(index=False)
        ]
        trace_ordered = [
            (int(row.layer), int(row.head))
            for row in trace_frame.itertuples(index=False)
        ]
        confirmation_lookup = {}
        if not confirmation_ranking.empty:
            active_confirmation = confirmation_ranking.loc[
                confirmation_ranking["model_label"].astype(str).eq(
                    str(model_label)
                )
            ]
            confirmation_lookup = {
                (int(row.layer), int(row.head)): (
                    float(row.confirmation_target_raw_mass),
                    float(row.confirmation_target_relative_mass),
                    float(row.confirmation_trace_raw_mass),
                    float(row.confirmation_trace_relative_mass),
                    float(row.confirmation_prompt_broad_score),
                    float(row.confirmation_prompt_broad_coverage),
                    float(row.confirmation_trace_broad_score),
                    float(row.confirmation_trace_broad_coverage),
                )
                for row in active_confirmation.itertuples(index=False)
            }

        def joint_row(
            active_heads: Sequence[tuple[int, int]],
            *,
            condition: str,
            bank_size: int,
            repeat: int,
            prompt_heads: Sequence[tuple[int, int]],
            trace_heads: Sequence[tuple[int, int]],
        ) -> dict[str, Any]:
            generic_mass = _bank_attention_mass(prompt_frame, active_heads)
            prompt_mass = _bank_answer_selection_mass(
                prompt_frame, active_heads
            )
            trace_mass = _bank_answer_selection_mass(trace_frame, active_heads)
            confirmation_values = [
                confirmation_lookup.get(
                    (int(layer), int(head)),
                    (np.nan,) * 8,
                )
                for layer, head in active_heads
            ]
            confirmation_array = np.asarray(confirmation_values, dtype=float)
            prompt_set = set(prompt_heads)
            trace_set = set(trace_heads)
            return {
                "model_label": model_label,
                "mechanism": "answer_prompt_and_trace_aggregation",
                "query_site_kind": "answer_query_v3",
                "experiment_id": "answer_query_head_ablation_v5_broad_factorial",
                "condition": condition,
                "bank_size": int(bank_size),
                "repeat": int(repeat),
                "heads": json.dumps(list(active_heads)),
                "selected_head_count": int(len(active_heads)),
                "prompt_bank_size": int(len(prompt_heads)),
                "trace_bank_size": int(len(trace_heads)),
                "prompt_trace_head_overlap": int(len(prompt_set & trace_set)),
                "prompt_bank_heads": json.dumps(list(prompt_heads)),
                "trace_bank_heads": json.dumps(list(trace_heads)),
                **generic_mass,
                "selected_aggregation_raw_mass": np.nan,
                "selected_aggregation_relative_mass": np.nan,
                "selected_aggregation_broad_score": np.nan,
                "selected_aggregation_broad_coverage": np.nan,
                "selected_aggregation_relative_defined_heads": 0,
                "selected_aggregation_metric": (
                    "joint_prompt_and_trace_reported_separately"
                ),
                "prompt_aggregation_raw_mass": prompt_mass[
                    "selected_aggregation_raw_mass"
                ],
                "prompt_aggregation_relative_mass": prompt_mass[
                    "selected_aggregation_relative_mass"
                ],
                "prompt_aggregation_broad_score": prompt_mass[
                    "selected_aggregation_broad_score"
                ],
                "prompt_aggregation_broad_coverage": prompt_mass[
                    "selected_aggregation_broad_coverage"
                ],
                "trace_aggregation_raw_mass": trace_mass[
                    "selected_aggregation_raw_mass"
                ],
                "trace_aggregation_relative_mass": trace_mass[
                    "selected_aggregation_relative_mass"
                ],
                "trace_aggregation_broad_score": trace_mass[
                    "selected_aggregation_broad_score"
                ],
                "trace_aggregation_broad_coverage": trace_mass[
                    "selected_aggregation_broad_coverage"
                ],
                "confirmation_target_needle_raw_mass": (
                    float(np.nanmean(confirmation_array[:, 0]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 0]).any()
                    else np.nan
                ),
                "confirmation_target_needle_relative_mass": (
                    float(np.nanmean(confirmation_array[:, 1]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 1]).any()
                    else np.nan
                ),
                "confirmation_selected_aggregation_raw_mass": np.nan,
                "confirmation_selected_aggregation_relative_mass": np.nan,
                "confirmation_selected_aggregation_broad_score": np.nan,
                "confirmation_selected_aggregation_broad_coverage": np.nan,
                "confirmation_prompt_aggregation_raw_mass": (
                    float(np.nanmean(confirmation_array[:, 0]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 0]).any()
                    else np.nan
                ),
                "confirmation_prompt_aggregation_relative_mass": (
                    float(np.nanmean(confirmation_array[:, 1]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 1]).any()
                    else np.nan
                ),
                "confirmation_trace_aggregation_raw_mass": (
                    float(np.nanmean(confirmation_array[:, 2]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 2]).any()
                    else np.nan
                ),
                "confirmation_trace_aggregation_relative_mass": (
                    float(np.nanmean(confirmation_array[:, 3]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 3]).any()
                    else np.nan
                ),
                "confirmation_prompt_aggregation_broad_score": (
                    float(np.nanmean(confirmation_array[:, 4]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 4]).any()
                    else np.nan
                ),
                "confirmation_prompt_aggregation_broad_coverage": (
                    float(np.nanmean(confirmation_array[:, 5]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 5]).any()
                    else np.nan
                ),
                "confirmation_trace_aggregation_broad_score": (
                    float(np.nanmean(confirmation_array[:, 6]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 6]).any()
                    else np.nan
                ),
                "confirmation_trace_aggregation_broad_coverage": (
                    float(np.nanmean(confirmation_array[:, 7]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 7]).any()
                    else np.nan
                ),
                "confirmation_mass_split": "confirmation_descriptive_only",
            }

        for bank_size in config.causal_head_bank_sizes:
            if bank_size > len(prompt_ordered) or bank_size > len(trace_ordered):
                continue
            prompt_heads, trace_heads = _control_constructible_joint_ranked_heads(
                prompt_frame,
                prompt_ordered,
                trace_ordered,
                bank_size=int(bank_size),
            )
            joint_heads = list(dict.fromkeys([*prompt_heads, *trace_heads]))
            plan_rows.append(
                joint_row(
                    joint_heads,
                    condition="answer_prompt_and_trace_aggregation_ranked",
                    bank_size=int(bank_size),
                    repeat=0,
                    prompt_heads=prompt_heads,
                    trace_heads=trace_heads,
                )
            )
            try:
                controls = _legacy_layer_matched_random_controls(
                    prompt_frame,
                    joint_heads,
                    repeats=config.causal_random_controls,
                    seed_text=(
                        f"v5-answer-query:{model_label}:joint:K{bank_size}"
                    ),
                )
            except ValueError as error:
                skipped_banks.append(
                    {
                        "model_label": str(model_label),
                        "mechanism": "answer_prompt_and_trace_aggregation",
                        "bank_size": int(bank_size),
                        "ranked_treatment_included": True,
                        "control_status": (
                            "not_constructible_disjoint_exact_layer_match"
                        ),
                        "reason": str(error),
                    }
                )
                continue
            for repeat, control in enumerate(controls, start=1):
                plan_rows.append(
                    joint_row(
                        control,
                        condition=(
                            "answer_prompt_and_trace_aggregation_"
                            "layer_matched_random"
                        ),
                        bank_size=int(bank_size),
                        repeat=repeat,
                        prompt_heads=prompt_heads,
                        trace_heads=trace_heads,
                    )
                )
    paths = {
        "ranking": output / "discovery_answer_query_head_ranking.csv",
        "plan": output / "answer_query_causal_plan.csv",
        "audit": output / "answer_query_causal_plan_audit.json",
    }
    ranking.to_csv(paths["ranking"], index=False)
    pd.DataFrame(plan_rows).to_csv(paths["plan"], index=False)
    paths["audit"].write_text(
        json.dumps(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": "answer_query_head_ablation_v5_broad_factorial",
                "site_id": "answer_query_v3",
                "query_definition": (
                    "literal baseline token immediately before the first "
                    "numeric answer token"
                ),
                "mechanisms": [
                    *answer_mechanisms,
                    "answer_prompt_and_trace_aggregation",
                ],
                "reported_behavioral_conditions": [
                    "clean",
                    "answer_prompt_aggregation_ranked",
                    "answer_trace_aggregation_ranked",
                    "answer_prompt_and_trace_aggregation_ranked",
                ],
                "selection_split": "discovery",
                "confirmation_used_for_selection": False,
                "selection_cohort": "one_to_one",
                "ranked_treatment_policy": (
                    "preserve discovery rank subject to per-layer occupancy "
                    "<= floor(available_heads/2), guaranteeing a disjoint "
                    "exact layer-matched control bank"
                ),
                "joint_ranked_treatment_policy": (
                    "equal-weight alternating greedy selection over the "
                    "independent prompt and trace discovery rankings under "
                    "the same control-constructability constraint"
                ),
                "selection_metrics": {
                    "answer_prompt_aggregation": (
                        "mean(prompt exact-span total mass * normalized "
                        "effective-span coverage), identical to non-thinking "
                        "broad_primary"
                    ),
                    "answer_trace_aggregation": (
                        "mean(trace registered-item total mass * normalized "
                        "effective-span coverage), identical broad-primary "
                        "definition applied independently to trace items"
                    ),
                },
                "relative_mass_denominator": "all_prompt_attention_mass",
                "registered_bank_sizes": list(config.causal_head_bank_sizes),
                "random_control": (
                    "disjoint exact layer-matched random heads; structural "
                    "unavailability is reported, never silently substituted"
                ),
                "skipped_banks": skipped_banks,
                "attention_source": str(Path(attention_csv).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def completion_metrics(result: Mapping[str, Any], *, gold_count: int) -> dict[str, Any]:
    full = str(result.get("full_answer_text", ""))
    prediction = parse_total(full)
    return {
        "prediction": prediction,
        "exact_count": prediction == int(gold_count),
        "signed_error": prediction - int(gold_count) if prediction is not None else None,
        "absolute_error": (
            abs(prediction - int(gold_count)) if prediction is not None else None
        ),
        "completion_text_raw": result.get("completion_text_raw"),
        "generated_token_count": result.get("generated_token_count"),
        "generation_truncated": result.get("generation_truncated"),
    }


@torch.inference_mode()
def capture_natural_head_writes(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    layers: Iterable[int] | None = None,
    count_direction: (
        np.ndarray
        | torch.Tensor
        | Mapping[int, np.ndarray | torch.Tensor]
        | None
    ) = None,
) -> tuple[pd.DataFrame, dict[tuple[int, int], torch.Tensor]]:
    """Decompose the natural attention output at one registered query.

    The input to each layer's output projection is the concatenated,
    attention-weighted head output. For a linear output projection, evaluating
    one head slice at a time and subtracting the zero-input output gives its
    exact natural residual write, including GQA/global-layer differences
    already resolved by ``DecoderAdapter``. This implementation is native to
    V5 and has no V4.4.2+ experiment dependency.
    """

    active_layers = (
        tuple(range(int(adapter.num_layers)))
        if layers is None
        else tuple(sorted({int(layer) for layer in layers}))
    )
    if not active_layers or any(
        layer < 0 or layer >= int(adapter.num_layers) for layer in active_layers
    ):
        raise ValueError("Invalid natural-write layers")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in active_layers:

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention output projection received no tensor")
            value = args[0]
            if value.ndim != 3:
                raise RuntimeError("Expected [batch,time,heads*head_dim] before o_proj")
            captured[layer] = value[0, int(encoding.query_position)].detach()

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(active_layers) - set(captured))
    if missing:
        raise RuntimeError(f"Natural-write capture missed layers {missing}")
    def normalized_direction(
        value: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        direction = torch.as_tensor(value, dtype=torch.float32)
        if direction.ndim != 1:
            raise ValueError("count_direction must be one residual vector")
        norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise ValueError("count_direction must be finite and nonzero")
        return direction / norm

    directions: dict[int, torch.Tensor] = {}
    if isinstance(count_direction, Mapping):
        directions = {
            int(layer): normalized_direction(value)
            for layer, value in count_direction.items()
        }
        missing_directions = sorted(set(active_layers) - set(directions))
        if missing_directions:
            raise ValueError(
                f"count_direction mapping is missing layers {missing_directions}"
            )
    elif count_direction is not None:
        shared_direction = normalized_direction(count_direction)
        directions = {layer: shared_direction for layer in active_layers}
    rows: list[dict[str, Any]] = []
    writes: dict[tuple[int, int], torch.Tensor] = {}
    for layer in active_layers:
        direction = directions.get(layer)
        projection = adapter.output_projections[layer]
        aggregate = captured[layer]
        heads = int(adapter.num_heads[layer])
        head_dim = int(adapter.head_dims[layer])
        if aggregate.numel() != heads * head_dim:
            raise RuntimeError(
                f"L{layer} o_proj input width disagrees with decoder adapter"
            )
        zero = torch.zeros_like(aggregate)
        zero_output = projection(zero.reshape(1, 1, -1))[0, 0]
        full_delta = projection(aggregate.reshape(1, 1, -1))[0, 0] - zero_output
        summed = torch.zeros_like(full_delta)
        layer_writes: list[torch.Tensor] = []
        for head in range(heads):
            left = head * head_dim
            right = left + head_dim
            isolated = zero.clone()
            isolated[left:right] = aggregate[left:right]
            write = (
                projection(isolated.reshape(1, 1, -1))[0, 0] - zero_output
            ).detach().float().cpu()
            writes[(int(layer), int(head))] = write
            layer_writes.append(write)
            summed = summed + write.to(device=summed.device, dtype=summed.dtype)
        reconstruction_error = float(
            torch.linalg.vector_norm(summed - full_delta).detach().float().cpu()
        )
        for head, write in enumerate(layer_writes):
            norm = float(torch.linalg.vector_norm(write))
            if direction is None:
                coefficient = np.nan
                cosine = np.nan
            else:
                active_direction = direction.to(write)
                coefficient = float(torch.dot(write, active_direction))
                cosine = coefficient / norm if norm > 0 else np.nan
            rows.append(
                {
                    "schema_version": CAUSAL_SCHEMA_VERSION,
                    "request_id": encoding.request_id,
                    "model_label": encoding.model_label,
                    "seed": encoding.seed,
                    "split": encoding.split,
                    "gold_count": encoding.count,
                    "site_id": encoding.selected_site.get("site_id"),
                    "layer": int(layer),
                    "head": int(head),
                    "layer_type": adapter.layer_types[layer],
                    "write_norm": norm,
                    "count_direction_coefficient": coefficient,
                    "count_direction_cosine": cosine,
                    "layer_reconstruction_error": reconstruction_error,
                }
            )
    return pd.DataFrame(rows), writes


def _legacy_mechanism_continuations(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    mechanism: str,
    boundary_policy: str = "strict_registered",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if boundary_policy not in {"strict_registered", "item_end_fallback_v2"}:
        raise ValueError(f"Unknown causal boundary policy: {boundary_policy}")
    family = infer_model_family(row)
    parser = find_trace_count_sequence(
        raw_output_text(row),
        model_family=family,
        gold_records=gold_records(row),
    )
    token_sites = align_trace_sites(
        tokenizer,
        raw_text=raw_output_text(row),
        baseline_output_token_ids=output_token_ids(row),
        sites=trace_char_sites(raw_output_text(row), parser),
    )
    by_id = {site.char_site.site_id: site for site in token_sites}
    cities = [str(value) for value in parser.item_gold_cities]
    specifications: list[
        tuple[str, tuple[str, ...], int, str, str | None]
    ] = []
    if mechanism == "targeted_retrieval":
        specifications = [
            (
                f"marker_end:{k}",
                (
                    (f"city_end:{k}", f"item_end:{k}")
                    if boundary_policy == "item_end_fallback_v2"
                    else (f"city_end:{k}",)
                ),
                k,
                "retrieve",
                city,
            )
            for k, city in enumerate(cities, start=1)
        ]
    elif mechanism == "progress_transition":
        specifications = [
            (
                f"item_end:{k}",
                (
                    (f"marker_end:{k + 1}", f"item_end:{k + 1}")
                    if boundary_policy == "item_end_fallback_v2"
                    else (f"marker_end:{k + 1}",)
                ),
                k,
                "continue",
                cities[k],
            )
            for k in range(1, len(cities))
        ]
        if cities and "answer_query" in by_id:
            specifications.append(
                (
                    f"item_end:{len(cities)}",
                    ("answer_query",),
                    len(cities),
                    "stop",
                    None,
                )
            )
    else:
        raise ValueError(f"Unknown head mechanism: {mechanism}")
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for query_id, target_ids, occurrence, phase, target_city in specifications:
        query = by_id.get(query_id)
        primary_target_id = target_ids[0]
        reason = None
        if query is None:
            reason = "missing_registered_boundary"
        elif not query.alignment_eligible:
            reason = "ineligible_text_exact_alignment"
        elif query.alignment_strategy != "literal_baseline_token_prefix":
            reason = "local_causal_trial_requires_literal_baseline_alignment"
        elif query.prefix_token_count is None:
            reason = "empty_or_reversed_target_continuation"

        candidate_audit: list[dict[str, str]] = []
        selected_target_id: str | None = None
        target = None
        if reason is None:
            for target_id in target_ids:
                candidate = by_id.get(target_id)
                candidate_reason = None
                if candidate is None:
                    candidate_reason = "missing_registered_boundary"
                elif not candidate.alignment_eligible:
                    candidate_reason = "ineligible_text_exact_alignment"
                elif candidate.alignment_strategy != "literal_baseline_token_prefix":
                    candidate_reason = (
                        "local_causal_trial_requires_literal_baseline_alignment"
                    )
                elif (
                    candidate.prefix_token_count is None
                    or int(candidate.prefix_token_count)
                    <= int(query.prefix_token_count)
                ):
                    candidate_reason = "empty_or_reversed_target_continuation"
                candidate_audit.append(
                    {
                        "site_id": target_id,
                        "status": candidate_reason or "selected",
                    }
                )
                if candidate_reason is None:
                    selected_target_id = target_id
                    target = candidate
                    break
            if selected_target_id is None:
                if boundary_policy == "strict_registered":
                    reason = candidate_audit[0]["status"]
                else:
                    reason = "no_eligible_registered_target_boundary"
        payload = {
            "mechanism": mechanism,
            "query_site_id": query_id,
            "target_site_id": selected_target_id or primary_target_id,
            "primary_target_site_id": primary_target_id,
            "target_site_candidates": list(target_ids),
            "target_boundary_policy": boundary_policy,
            "target_boundary_variant": (
                "item_end_fallback"
                if selected_target_id is not None
                and selected_target_id != primary_target_id
                else "primary"
            ),
            "target_site_fallback": bool(
                selected_target_id is not None
                and selected_target_id != primary_target_id
            ),
            "target_candidate_audit": candidate_audit,
            "occurrence": int(occurrence),
            "transition_phase": phase,
            "target_city": target_city,
        }
        if reason is not None:
            excluded.append({**payload, "status": reason})
            continue
        eligible.append(
            {
                **payload,
                "query_output_token_count": int(query.prefix_token_count),
                "target_output_token_count": int(target.prefix_token_count),
            }
        )
    return eligible, excluded


def _value_capture_module(attention: Any) -> Any:
    """Return the module whose output is the value tensor used by attention.

    Gemma 4 normalizes values after projection and may use its raw key
    projection when ``attention_k_eq_v`` is active.  Capturing ``v_norm``
    therefore observes the actual value states in both cases.  Architectures
    without a value normalizer continue to use their ordinary value
    projection.
    """

    for name in ("v_norm", "v_proj", "value", "value_proj"):
        module = getattr(attention, name, None)
        if module is not None:
            return module
    raise RuntimeError(
        f"Cannot find a value capture module in {type(attention).__name__}"
    )


def _shared_source_value_slice(
    attention: Any,
    shared_kv_states: Any,
    *,
    source_start: int,
    source_end: int,
) -> torch.Tensor | None:
    """Read Gemma-style shared values as ``[source, kv_heads, head_dim]``."""

    layer_type = getattr(attention, "layer_type", None)
    if layer_type is None or not isinstance(shared_kv_states, Mapping):
        return None
    pair = shared_kv_states.get(layer_type)
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        return None
    values = pair[1]
    if not isinstance(values, torch.Tensor) or values.ndim != 4:
        return None
    if int(values.shape[0]) != 1 or int(values.shape[2]) < int(source_end):
        return None
    return (
        values[0, :, int(source_start) : int(source_end), :]
        .transpose(0, 1)
        .detach()
    )


def _source_specific_write_decomposition(
    adapter: DecoderAdapter,
    *,
    attention_rows: Mapping[int, torch.Tensor],
    key_starts: Mapping[int, int],
    source_values: Mapping[int, torch.Tensor],
    source_start: int,
    source_end: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], torch.Tensor]]:
    """Apply each query head's source-weighted value through its O slice."""

    metric_rows: list[dict[str, Any]] = []
    writes: dict[tuple[int, int], torch.Tensor] = {}
    for layer in sorted(attention_rows):
        attention = attention_rows[layer]
        values = source_values[layer]
        if attention.ndim != 2:
            raise RuntimeError("Expected [query_heads,key] attention rows")
        query_heads = int(attention.shape[0])
        head_dim = int(adapter.head_dims[layer])
        if values.ndim == 2:
            if int(values.shape[-1]) % head_dim:
                raise RuntimeError(f"L{layer} value width is not divisible by head dim")
            values = values.reshape(
                int(values.shape[0]), int(values.shape[-1]) // head_dim, head_dim
            )
        if values.ndim != 3 or int(values.shape[-1]) != head_dim:
            raise RuntimeError(
                f"L{layer} expected [source,kv_heads,head_dim] values"
            )
        kv_heads = int(values.shape[1])
        if query_heads % kv_heads:
            raise RuntimeError(
                f"L{layer} query heads are not divisible by value heads"
            )
        groups = query_heads // kv_heads
        key_start = int(key_starts[layer])
        key_end = key_start + int(attention.shape[-1])
        overlap_start = max(int(source_start), key_start)
        overlap_end = min(int(source_end), key_end)
        projection = adapter.output_projections[layer]
        projection_device = next(
            (value.device for value in projection.parameters()), values.device
        )
        projection_dtype = next(
            (value.dtype for value in projection.parameters()), values.dtype
        )
        zero = torch.zeros(
            query_heads * head_dim,
            device=projection_device,
            dtype=projection_dtype,
        )
        zero_output = projection(zero.reshape(1, 1, -1))[0, 0]
        summed = torch.zeros_like(zero_output)
        source_pre_o = zero.clone()
        if overlap_end > overlap_start:
            attention_slice = attention[
                :, overlap_start - key_start : overlap_end - key_start
            ]
            value_slice = values[
                overlap_start - int(source_start) : overlap_end - int(source_start)
            ]
        else:
            attention_slice = attention[:, :0]
            value_slice = values[:0]
        for head in range(query_heads):
            kv_head = head // groups
            if overlap_end > overlap_start:
                weights = attention_slice[head].to(
                    device=value_slice.device, dtype=value_slice.dtype
                )
                source_vector = torch.einsum(
                    "s,sd->d", weights, value_slice[:, kv_head]
                )
                source_mass = float(attention_slice[head].sum().item())
            else:
                source_vector = torch.zeros(
                    head_dim, device=values.device, dtype=values.dtype
                )
                source_mass = 0.0
            isolated = zero.clone()
            left = head * head_dim
            isolated[left : left + head_dim] = source_vector.to(
                device=isolated.device, dtype=isolated.dtype
            )
            source_pre_o[left : left + head_dim] = isolated[
                left : left + head_dim
            ]
            write = (
                projection(isolated.reshape(1, 1, -1))[0, 0] - zero_output
            ).detach().float().cpu()
            writes[(int(layer), int(head))] = write
            summed = summed + write.to(device=summed.device, dtype=summed.dtype)
            metric_rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "source_attention_mass": source_mass,
                    "source_specific_ov_write_norm": float(
                        torch.linalg.vector_norm(write)
                    ),
                    "value_head": int(kv_head),
                    "query_to_value_group_size": int(groups),
                }
            )
        full_source_delta = (
            projection(source_pre_o.reshape(1, 1, -1))[0, 0] - zero_output
        )
        reconstruction_error = float(
            torch.linalg.vector_norm(
                summed - full_source_delta
            )
            .detach()
            .float()
            .cpu()
        )
        for row in metric_rows[-query_heads:]:
            row["layer_source_write_reconstruction_error"] = reconstruction_error
            row["source_key_overlap_count"] = max(0, overlap_end - overlap_start)
    return metric_rows, writes


def _source_attention_concentration_frame(
    attention_rows: Mapping[int, torch.Tensor],
    key_starts: Mapping[int, int],
    *,
    source_spans: Mapping[str, Any],
    target_city: str,
) -> pd.DataFrame:
    """Audit whether each head attends preferentially to the target record.

    All masses are measured at the same query token used by the subsequent
    intervention.  The denominator contains every registered gold prompt
    record (and no generated trace span), so the relative mass distinguishes
    target-selective retrieval from broad record attention.
    """

    target_key = str(target_city).casefold()
    spans_by_key = {
        str(city).casefold(): span for city, span in source_spans.items()
    }
    if target_key not in spans_by_key:
        raise ValueError(f"Target city {target_city!r} is absent from audit spans")
    rows: list[dict[str, Any]] = []
    for layer in sorted(attention_rows):
        attention = attention_rows[layer]
        if attention.ndim != 2:
            raise RuntimeError("Expected [query_heads,key] attention rows")
        key_start = int(key_starts[layer])
        key_end = key_start + int(attention.shape[-1])
        for head in range(int(attention.shape[0])):
            head_row = attention[head]
            masses: dict[str, float] = {}
            for city_key, span in spans_by_key.items():
                overlap_start = max(int(span.start), key_start)
                overlap_end = min(int(span.end), key_end)
                if overlap_end <= overlap_start:
                    mass = 0.0
                else:
                    mass = float(
                        head_row[
                            overlap_start - key_start : overlap_end - key_start
                        ]
                        .sum()
                        .item()
                    )
                masses[city_key] = mass
            target_mass = float(masses[target_key])
            wrong_masses = [
                mass for city, mass in masses.items() if city != target_key
            ]
            total_mass = float(sum(masses.values()))
            max_wrong = float(max(wrong_masses, default=0.0))
            target_rank = 1 + sum(
                mass > target_mass for mass in wrong_masses
            )
            rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "target_source_attention_mass": target_mass,
                    "all_gold_source_attention_mass": total_mass,
                    "target_source_relative_attention_mass": (
                        target_mass / total_mass
                        if total_mass > 0.0
                        else np.nan
                    ),
                    "max_wrong_source_attention_mass": max_wrong,
                    "target_minus_max_wrong_source_attention_mass": (
                        target_mass - max_wrong
                    ),
                    "target_source_attention_rank": int(target_rank),
                    "target_source_attention_top1": bool(
                        total_mass > 0.0 and target_rank == 1
                    ),
                    "target_source_attention_unique_top1": bool(
                        total_mass > 0.0 and target_mass > max_wrong
                    ),
                    "attention_audit_gold_source_count": len(masses),
                }
            )
    return pd.DataFrame(rows)


def capture_source_specific_head_writes_multi(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    source_cities: Sequence[str],
    attention_audit_cities: Sequence[str] | None = None,
    layers: Iterable[int] | None = None,
) -> dict[str, tuple[pd.DataFrame, dict[tuple[int, int], torch.Tensor]]]:
    """Capture source-specific writes for several prompt records in one query."""

    cities = tuple(str(value) for value in source_cities)
    if not cities:
        raise ValueError("At least one prompt source city is required")
    if len({city.casefold() for city in cities}) != len(cities):
        raise ValueError("Prompt source cities must be unique ignoring case")
    audit_cities = (
        cities
        if attention_audit_cities is None
        else tuple(str(value) for value in attention_audit_cities)
    )
    if not audit_cities:
        raise ValueError("At least one attention-audit city is required")
    if len({city.casefold() for city in audit_cities}) != len(audit_cities):
        raise ValueError("Attention-audit cities must be unique ignoring case")
    missing_audit_targets = sorted(
        {city.casefold() for city in cities}
        - {city.casefold() for city in audit_cities}
    )
    if missing_audit_targets:
        raise ValueError(
            "Every source-write city must also be attention-audited: "
            f"{missing_audit_targets}"
        )
    source_spans = {}
    for city in tuple(dict.fromkeys((*cities, *audit_cities))):
        matches = [
            span
            for span in encoding.prompt_record_spans
            if span.city.casefold() == city.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one registered prompt source record for {city}, "
                f"found {len(matches)}"
            )
        source_spans[city] = matches[0]
    active_layers = (
        tuple(range(int(adapter.num_layers)))
        if layers is None
        else tuple(sorted({int(value) for value in layers}))
    )
    if not active_layers:
        raise ValueError("Source-specific writes require at least one layer")
    captured_values: dict[str, dict[int, torch.Tensor]] = {
        city: {} for city in cities
    }
    handles = []
    for layer in active_layers:
        attention = adapter.attentions[layer]

        if bool(getattr(attention, "is_kv_shared_layer", False)):

            def shared_hook(
                module: Any,
                _args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                layer: int = layer,
            ) -> None:
                for city in cities:
                    span = source_spans[city]
                    if layer in captured_values[city]:
                        continue
                    value = _shared_source_value_slice(
                        module,
                        kwargs.get("shared_kv_states"),
                        source_start=int(span.start),
                        source_end=int(span.end),
                    )
                    if value is not None:
                        captured_values[city][layer] = value

            handles.append(
                attention.register_forward_pre_hook(shared_hook, with_kwargs=True)
            )
            continue

        projection = _value_capture_module(attention)

        def hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            value = _tensor_from_output(output)
            if value.ndim < 3 or int(value.shape[0]) != 1:
                raise RuntimeError("Value projection returned an unsupported tensor")
            for city in cities:
                span = source_spans[city]
                if (
                    int(value.shape[1]) >= int(span.end)
                    and layer not in captured_values[city]
                ):
                    captured_values[city][layer] = value[
                        0, int(span.start) : int(span.end)
                    ].detach()

        handles.append(projection.register_forward_hook(hook))
    try:
        rows, starts, _logits = position_attention_outputs(
            model,
            adapter,
            encoding,
            int(encoding.query_position),
        )
    finally:
        for handle in handles:
            handle.remove()
    selected_rows = {layer: rows[layer] for layer in active_layers}
    selected_starts = {layer: starts[layer] for layer in active_layers}
    audit_spans = {city: source_spans[city] for city in audit_cities}
    result = {}
    for city in cities:
        missing = sorted(set(active_layers) - set(captured_values[city]))
        if missing:
            raise RuntimeError(f"Value capture for {city} missed layers {missing}")
        span = source_spans[city]
        metrics, writes = _source_specific_write_decomposition(
            adapter,
            attention_rows=selected_rows,
            key_starts=selected_starts,
            source_values=captured_values[city],
            source_start=int(span.start),
            source_end=int(span.end),
        )
        concentration = _source_attention_concentration_frame(
            selected_rows,
            selected_starts,
            source_spans=audit_spans,
            target_city=city,
        )
        metric_frame = pd.DataFrame(metrics).merge(
            concentration,
            on=["layer", "head"],
            how="inner",
            validate="one_to_one",
        )
        if not np.allclose(
            metric_frame["source_attention_mass"].to_numpy(dtype=float),
            metric_frame["target_source_attention_mass"].to_numpy(dtype=float),
            rtol=1e-5,
            atol=1e-7,
        ):
            raise RuntimeError(
                "Target source-write mass disagrees with attention audit mass"
            )
        metadata = {
            "schema_version": CAUSAL_SCHEMA_VERSION,
            "status": "ok",
            "request_id": encoding.request_id,
            "model_label": encoding.model_label,
            "seed": encoding.seed,
            "split": encoding.split,
            "gold_count": encoding.count,
            "target_city": city,
            "source_record_token_start": int(span.start),
            "source_record_token_end": int(span.end),
            "query_full_sequence_token": int(encoding.query_position),
            "attention_selection_query_full_sequence_token": int(
                encoding.query_position
            ),
            "attention_audit_scope": "all_registered_gold_prompt_records",
            **dict(encoding.selected_site),
        }
        result[city] = (
            pd.DataFrame(
                [{**metadata, **row} for row in metric_frame.to_dict("records")]
            ),
            writes,
        )
    return result


def capture_source_specific_head_writes(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    target_city: str,
    attention_audit_cities: Sequence[str] | None = None,
    layers: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[int, int], torch.Tensor]]:
    """Capture ``W_O^h sum_(j in R(city)) A(q,j)V(j)`` at one anchor."""

    return capture_source_specific_head_writes_multi(
        model,
        adapter,
        encoding,
        source_cities=[str(target_city)],
        attention_audit_cities=attention_audit_cities,
        layers=layers,
    )[str(target_city)]


def _registered_mechanism_continuations(
    row: Mapping[str, Any], tokenizer: Any, *, mechanism: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize registry-defined transition anchors and fixed city targets.

    The old V5 implementation treated ``marker_end:k -> city_end:k`` as one
    continuation and therefore changed both the scored sequence and its length
    whenever the query site changed.  The grammar-aware compiler now owns all
    site decisions.  Every eligible anchor for transition ``k -> k+1`` scores
    exactly the stored token span of city ``k+1``; intervening text remains in
    the teacher-forced input but is not part of the endpoint.
    """

    if mechanism == "targeted_retrieval":
        raise ValueError(
            "The legacy targeted_retrieval estimand was removed; use "
            "retrieval_anchor_localization"
        )
    if mechanism not in {"retrieval_anchor_localization", "progress_transition"}:
        raise ValueError(f"Unknown head mechanism: {mechanism}")
    plan = compile_causal_site_plan(row, tokenizer)
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for transition in plan["transitions"]:
        base = {
            "mechanism": mechanism,
            "causal_cohort": plan["causal_cohort"],
            "transition_phase": "continue",
            "transition_kind": transition["transition_kind"],
            "from_occurrence": int(transition["from_occurrence"]),
            "to_occurrence": int(transition["to_occurrence"]),
            "occurrence": int(transition["from_occurrence"]),
            "source_control_city": str(transition["current_city"]),
            "target_city": str(transition["target_city"]),
            "grammar_pair": str(transition["grammar_pair"]),
            "target_retrieval_surface_variant": str(
                transition.get("target_retrieval_surface_variant", "")
            ),
            "target_rank_to_city_interstitial_char_count": transition.get(
                "target_rank_to_city_interstitial_char_count"
            ),
            "target_rank_to_city_interstitial_token_count": transition.get(
                "target_rank_to_city_interstitial_token_count"
            ),
            "target_rank_to_city_has_lexical_content": transition.get(
                "target_rank_to_city_has_lexical_content"
            ),
            "target_output_token_start": transition.get(
                "next_city_output_token_start"
            ),
            "target_output_token_end": transition.get(
                "next_city_output_token_end"
            ),
            "target_token_ids": list(transition.get("next_city_token_ids", [])),
            "target_token_text": str(
                transition.get("next_city_token_text", "")
            ),
            "scored_target_policy": "fixed_next_city_token_span",
        }
        if mechanism == "progress_transition":
            candidates = [
                anchor
                for anchor in transition["anchors"]
                if "p0_item_end" in anchor.get("anchor_roles", [])
            ]
        else:
            candidates = list(transition["anchors"])

        if not transition["local_transition_eligible"]:
            reasons = list(transition.get("exclusion_reasons", [])) or [
                "transition_cohort_or_event_ineligible"
            ]
            for anchor in candidates or [{"anchor_role": "no_resolved_anchor"}]:
                excluded.append(
                    {
                        **base,
                        **dict(anchor),
                        "status": "excluded",
                        "exclusion_reason": ";".join(reasons),
                    }
                )
            continue

        for anchor in candidates:
            if not anchor.get("local_anchor_eligible"):
                excluded.append(
                    {
                        **base,
                        **dict(anchor),
                        "status": "excluded",
                        "exclusion_reason": "anchor_not_local_causal_eligible",
                    }
                )
                continue
            eligible.append(
                {
                    **base,
                    **dict(anchor),
                    "query_site_id": str(anchor["anchor_equivalence_id"]),
                    "query_output_token_index": int(
                        anchor["output_token_index"]
                    ),
                }
            )

        if mechanism == "retrieval_anchor_localization":
            resolved_roles = {
                str(role)
                for anchor in transition["anchors"]
                for role in anchor.get("anchor_roles", [])
            }
            for candidate in transition.get("anchor_candidates", []):
                role = str(candidate["anchor_role"])
                if candidate.get("status") == "ok" or role in resolved_roles:
                    continue
                excluded.append(
                    {
                        **base,
                        **dict(candidate),
                        "status": "not_applicable",
                        "exclusion_reason": str(
                            candidate.get(
                                "not_applicable_reason",
                                "semantic_anchor_unresolved",
                            )
                        ),
                    }
                )
    return eligible, excluded


def mechanism_continuations(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    mechanism: str,
    boundary_policy: str = "strict_registered",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile legacy or parser-registered causal continuations by estimand."""

    if mechanism in {"retrieval_anchor_localization", "progress_transition"}:
        if boundary_policy != "strict_registered":
            if mechanism == "retrieval_anchor_localization":
                raise ValueError(
                    "retrieval_anchor_localization uses only registered boundaries"
                )
            return _legacy_mechanism_continuations(
                row,
                tokenizer,
                mechanism=mechanism,
                boundary_policy=boundary_policy,
            )
        return _registered_mechanism_continuations(
            row,
            tokenizer,
            mechanism=mechanism,
        )
    return _legacy_mechanism_continuations(
        row,
        tokenizer,
        mechanism=mechanism,
        boundary_policy=boundary_policy,
    )


def _norm_matched_vector(
    control: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, float]:
    """Scale a control write to the reference residual-stream norm."""

    if control.shape != reference.shape:
        raise ValueError("Control/reference writes must have the same shape")
    control_norm = float(torch.linalg.vector_norm(control.detach().float()))
    reference_norm = float(torch.linalg.vector_norm(reference.detach().float()))
    if reference_norm <= float(epsilon):
        return torch.zeros_like(control), 0.0
    if control_norm <= float(epsilon):
        raise ValueError(
            "Cannot norm-match a zero previous-record source write to a "
            "nonzero target-record write"
        )
    scale = reference_norm / control_norm
    return control * scale, float(scale)


def _summed_source_write(
    writes: Mapping[tuple[int, int], torch.Tensor],
    heads: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Sum post-``W_O`` source contributions for one non-empty head group."""

    selected = [writes[(int(layer), int(head))] for layer, head in heads]
    if not selected:
        raise ValueError("A source-write group cannot be empty")
    result = torch.zeros_like(selected[0])
    for value in selected:
        if value.shape != result.shape:
            raise RuntimeError("Selected source-write vectors disagree in width")
        result = result + value.to(device=result.device, dtype=result.dtype)
    return result


def _closest_natural_norm_source(
    reference: torch.Tensor,
    controls: Mapping[str, torch.Tensor],
    *,
    epsilon: float = 1e-12,
) -> tuple[str, torch.Tensor]:
    """Choose a wrong-record write using norm only, never an outcome."""

    reference_norm = float(torch.linalg.vector_norm(reference.detach().float()))
    if reference_norm <= float(epsilon):
        raise ValueError("Target-record source write has zero norm")
    candidates = []
    for city, vector in controls.items():
        if vector.shape != reference.shape:
            raise ValueError("Wrong-record writes disagree with target width")
        norm = float(torch.linalg.vector_norm(vector.detach().float()))
        if norm <= float(epsilon):
            continue
        candidates.append(
            (abs(math.log(norm / reference_norm)), str(city).casefold(), str(city), vector)
        )
    if not candidates:
        raise ValueError("No nonzero wrong-record source write is available")
    _distance, _folded, city, vector = min(candidates, key=lambda value: value[:2])
    return city, vector


@torch.inference_mode()
def _fixed_target_head_write_intervention_logits(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    zero_heads: Sequence[tuple[int, int]] = (),
    output_deltas: Mapping[int, torch.Tensor] | None = None,
    *,
    hook_position: int,
    target_full_sequence_token_start: int,
    target_full_sequence_token_end: int,
) -> torch.Tensor:
    """Edit selected attention writes at one anchor and score a fixed target.

    ``zero_heads`` operates before ``W_O``. ``output_deltas`` operates after
    ``W_O`` in residual-stream coordinates. Keeping these two operations in one
    primitive makes source restoration auditable: a selected layer group can be
    zeroed and then receive only a frozen clean-state source-record write.
    """

    target_start = int(target_full_sequence_token_start)
    target_end = int(target_full_sequence_token_end)
    if not 0 <= int(hook_position) < target_start < target_end:
        raise ValueError("Hook must be strictly before a non-empty target span")
    if target_end != int(encoding.sequence_length):
        raise ValueError("Fixed target span must end at the teacher-forced prefix")
    target_token_count = target_end - target_start
    by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in zero_heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid ablation layer: {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    delta_by_layer = {
        int(layer): delta.detach()
        for layer, delta in (output_deltas or {}).items()
    }
    invalid_delta_layers = sorted(
        layer
        for layer in delta_by_layer
        if not 0 <= layer < int(adapter.num_layers)
    )
    if invalid_delta_layers:
        raise ValueError(f"Invalid output-delta layers: {invalid_delta_layers}")
    handles = []
    zero_applications = {layer: 0 for layer in by_layer}
    delta_applications = {layer: 0 for layer in delta_by_layer}
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            layer_heads: tuple[int, ...] = tuple(sorted(set(layer_heads))),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention projection received no head tensor")
            value = args[0]
            if value.ndim != 3 or hook_position >= int(value.shape[1]):
                raise RuntimeError("Mechanism query is outside the prefill tensor")
            patched = value.clone()
            for head in layer_heads:
                left = head * head_dim
                patched[:, hook_position, left : left + head_dim] = 0
            zero_applications[layer] += 1
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    for layer, raw_delta in delta_by_layer.items():

        def delta_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
            raw_delta: torch.Tensor = raw_delta,
        ) -> Any:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hook_position >= int(hidden.shape[1]):
                raise RuntimeError("Mechanism query is outside the projection output")
            delta = raw_delta.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
            if int(delta.numel()) != int(hidden.shape[-1]):
                raise RuntimeError(
                    f"L{layer} output delta has width {delta.numel()}, "
                    f"expected {hidden.shape[-1]}"
                )
            patched = hidden.clone()
            patched[:, hook_position, :] = patched[:, hook_position, :] + delta
            delta_applications[layer] += 1
            return _replace_output_tensor(output, patched)

        handles.append(
            adapter.output_projections[layer].register_forward_hook(delta_hook)
        )
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    keep = int(target_token_count) + 1
    if _accepts_keyword(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = keep
    try:
        output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    zero_violations = {
        layer: count for layer, count in zero_applications.items() if count != 1
    }
    delta_violations = {
        layer: count for layer, count in delta_applications.items() if count != 1
    }
    if zero_violations or delta_violations:
        raise RuntimeError(
            "Attention-write intervention must apply exactly once in every "
            f"selected layer: zero={zero_violations}, delta={delta_violations}"
        )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Mechanism trial returned no sequence logits")
    if int(logits.shape[1]) == int(encoding.sequence_length):
        selected = logits[0, target_start - 1 : target_end - 1]
    elif int(logits.shape[1]) == keep:
        selected = logits[0, :target_token_count]
    else:
        raise RuntimeError(
            f"Unexpected kept-logit length {logits.shape[1]} (expected {keep})"
        )
    return selected.detach().float().cpu()


@torch.inference_mode()
def _local_head_ablation_logits(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    hook_position: int,
    target_token_count: int,
) -> torch.Tensor:
    by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid ablation layer: {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    handles = []
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer_heads: tuple[int, ...] = tuple(sorted(set(layer_heads))),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention projection received no head tensor")
            value = args[0]
            if value.ndim != 3 or hook_position >= int(value.shape[1]):
                raise RuntimeError("Mechanism query is outside the prefill tensor")
            patched = value.clone()
            for head in layer_heads:
                left = head * head_dim
                patched[:, hook_position, left : left + head_dim] = 0
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    keep = int(target_token_count) + 1
    if _accepts_keyword(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = keep
    try:
        output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Mechanism trial returned no sequence logits")
    if int(logits.shape[1]) == int(encoding.sequence_length):
        selected = logits[0, hook_position : hook_position + target_token_count]
    elif int(logits.shape[1]) == keep:
        selected = logits[0, :target_token_count]
    else:
        raise RuntimeError(
            f"Unexpected kept-logit length {logits.shape[1]} (expected {keep})"
        )
    return selected.detach().float().cpu()
@torch.inference_mode()
def _scheduled_head_ablation_logits(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    hook_positions: Sequence[int],
    score_positions: Sequence[int],
) -> torch.Tensor:
    """Jointly ablate one bank at many baseline positions in one prefill.

    ``score_positions`` are autoregressive predictor positions (the token at
    position ``p`` predicts the target token at ``p + 1``).  Returning only
    those vocabulary rows avoids copying the full trace-by-vocabulary tensor
    to CPU while preserving exact teacher-forced downstream scoring.
    """

    scheduled = tuple(sorted({int(value) for value in hook_positions}))
    scored = tuple(int(value) for value in score_positions)
    if not scheduled:
        raise ValueError("Scheduled head ablation needs at least one query")
    if not scored:
        raise ValueError("Scheduled head ablation needs at least one endpoint")
    sequence_length = int(encoding.sequence_length)
    if any(value < 0 or value >= sequence_length for value in scheduled):
        raise ValueError("A scheduled query is outside the teacher-forced trace")
    if any(value < 0 or value >= sequence_length for value in scored):
        raise ValueError("A score position is outside the teacher-forced trace")
    by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid ablation layer: {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    handles = []
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer_heads: tuple[int, ...] = tuple(sorted(set(layer_heads))),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention projection received no head tensor")
            value = args[0]
            if value.ndim != 3 or int(value.shape[1]) != sequence_length:
                raise RuntimeError(
                    "Scheduled damage requires one full teacher-forced prefill"
                )
            patched = value.clone()
            for head in layer_heads:
                left = int(head) * head_dim
                right = left + head_dim
                patched[:, scheduled, left:right] = 0
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Scheduled damage returned no sequence logits")
    if int(logits.shape[1]) != sequence_length:
        raise RuntimeError("Scheduled damage did not return full-sequence logits")
    position_tensor = torch.as_tensor(scored, dtype=torch.long, device=logits.device)
    return logits[0].index_select(0, position_tensor).detach().float().cpu()


@torch.inference_mode()
def generate_with_head_ablation_at_positions(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    hook_positions: Sequence[int],
    max_new_tokens: int = 16,
    score_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Greedily decode after damaging registered positions in the prefill.

    Generation is the primary behavioral count endpoint.  Hooks apply during
    the initial teacher-forced prefill only; the patched KV/residual history is
    then carried through ordinary greedy decoding.
    """

    scheduled = tuple(sorted({int(value) for value in hook_positions}))
    if not scheduled:
        raise ValueError("Head-ablation generation needs at least one position")
    if any(value < 0 or value >= int(encoding.sequence_length) for value in scheduled):
        raise ValueError("A generation hook position is outside the input prefix")
    by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid ablation layer: {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    applied: dict[int, set[int]] = {layer: set() for layer in by_layer}
    handles = []
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            layer_heads: tuple[int, ...] = tuple(sorted(set(layer_heads))),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention projection received no head tensor")
            value = args[0]
            if value.ndim != 3:
                raise RuntimeError("Expected [batch,time,heads*head_dim]")
            active = [position for position in scheduled if position < value.shape[1]]
            if not active:
                return args
            patched = value.clone()
            for head in layer_heads:
                left = int(head) * head_dim
                right = left + head_dim
                patched[:, active, left:right] = 0
            applied[layer].update(active)
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    scored = tuple(int(value) for value in (score_positions or ()))
    if any(
        value < 0 or value >= int(encoding.sequence_length)
        for value in scored
    ):
        raise ValueError("A generation score position is outside the input prefix")
    captured_prefill_logits: list[torch.Tensor] = []
    capture_handle = None
    if scored:
        sequence_length = int(encoding.sequence_length)

        def capture_prefill_logits(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
        ) -> None:
            logits = getattr(output, "logits", None)
            if (
                captured_prefill_logits
                or not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or int(logits.shape[0]) != 1
                or int(logits.shape[1]) != sequence_length
            ):
                return
            positions = torch.as_tensor(
                scored,
                dtype=torch.long,
                device=logits.device,
            )
            captured_prefill_logits.append(
                logits[0]
                .index_select(0, positions)
                .detach()
                .float()
                .cpu()
            )

        capture_handle = model.register_forward_hook(capture_prefill_logits)
    try:
        if not scored:
            result = generate_answer_completion(
                model,
                tokenizer,
                encoding,
                max_new_tokens=max_new_tokens,
            )
        else:
            # A single generate() prefill now supplies both the frozen-trace
            # diagnostic rows and the KV cache used for actual greedy answer
            # generation.  ``logits_to_keep=0`` is required on architectures
            # that otherwise retain only the final prefill logit.
            if int(max_new_tokens) < 1:
                raise ValueError("max_new_tokens must be positive")
            input_ids, attention_mask = _encoding_tensors(model, encoding)
            generation_config = getattr(model, "generation_config", None)
            eos_value = (
                getattr(generation_config, "eos_token_id", None)
                if generation_config is not None
                else None
            )
            if eos_value is None:
                eos_value = getattr(tokenizer, "eos_token_id", None)
            if eos_value is None:
                eos_ids: list[int] = []
            elif isinstance(eos_value, (tuple, list, set)):
                eos_ids = [int(value) for value in eos_value]
            else:
                eos_ids = [int(eos_value)]
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None and eos_ids:
                pad_token_id = eos_ids[0]
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "do_sample": False,
                "max_new_tokens": int(max_new_tokens),
                "use_cache": True,
            }
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = int(pad_token_id)
            if _accepts_keyword(model, "logits_to_keep"):
                generation_kwargs["logits_to_keep"] = 0
            generated = model.generate(**generation_kwargs)
            if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
                sequences = getattr(generated, "sequences", None)
                if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
                    raise RuntimeError(
                        "Model.generate did not return [batch, time] sequences"
                    )
                generated = sequences
            if generated.shape[0] != 1 or generated.shape[1] < input_ids.shape[1]:
                raise RuntimeError("Unexpected generated sequence shape")
            continuation = [
                int(value)
                for value in generated[
                    0, input_ids.shape[1] :
                ].detach().cpu().tolist()
            ]
            if not continuation:
                raise RuntimeError("Greedy V5 generation returned an empty continuation")
            eos_set = set(eos_ids)
            stopped_on_eos = bool(eos_set and continuation[-1] in eos_set)
            raw_text = tokenizer.decode(
                continuation,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            clean_text = tokenizer.decode(
                continuation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            result = {
                "generated_token_ids": continuation,
                "generated_token_count": len(continuation),
                "generation_eos_token_ids": eos_ids,
                "stopped_on_eos": stopped_on_eos,
                "generation_truncated": bool(
                    len(continuation) >= int(max_new_tokens)
                    and not stopped_on_eos
                ),
                "completion_text_raw": str(raw_text),
                "completion_text": str(clean_text),
                "full_answer_text": "Total:" + str(clean_text),
            }
    finally:
        if capture_handle is not None:
            capture_handle.remove()
        for handle in handles:
            handle.remove()
    missed = {
        int(layer): sorted(set(scheduled) - positions)
        for layer, positions in applied.items()
        if set(scheduled) - positions
    }
    if missed:
        raise RuntimeError(f"Generation head hooks missed positions: {missed}")
    if scored and len(captured_prefill_logits) != 1:
        raise RuntimeError(
            "Fused generation did not capture exactly one full prefill logit tensor"
        )
    combined = {
        **result,
        "head_ablation_prefill_positions": list(scheduled),
        "head_ablation_prefill_position_count": len(scheduled),
        "head_ablation_layers": sorted(by_layer),
        "head_ablation_hook_audit": "PASS",
    }
    if scored:
        combined["prefill_selected_logits"] = captured_prefill_logits[0]
        combined["prefill_reuse_audit"] = "PASS_SINGLE_PREFILL"
    return combined


@torch.inference_mode()
def _fixed_target_head_ablation_logits(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    hook_position: int,
    target_full_sequence_token_start: int,
    target_full_sequence_token_end: int,
) -> torch.Tensor:
    """Ablate at one anchor and return only logits predicting target tokens."""

    return _fixed_target_head_write_intervention_logits(
        model,
        adapter,
        encoding,
        zero_heads=heads,
        hook_position=hook_position,
        target_full_sequence_token_start=target_full_sequence_token_start,
        target_full_sequence_token_end=target_full_sequence_token_end,
    )


def target_city_head_ablation_geometry(
    *,
    prompt_token_count: int,
    query_output_token_index: int,
    target_output_token_start: int,
    target_output_token_end: int,
    scope: str,
) -> dict[str, Any]:
    """Compile the exact predictor-position support of a city-token lesion.

    Full-sequence position ``p`` predicts token ``p + 1``.  The returned audit
    therefore exposes when the registered query is exactly the predictor
    immediately before the first city token instead of relabeling that equality
    as an apparent site change.
    """

    requested = str(scope)
    if requested not in TARGET_CITY_HEAD_ABLATION_SCOPES:
        raise ValueError(f"Unknown target-city head-ablation scope: {requested}")
    prompt = int(prompt_token_count)
    query = prompt + int(query_output_token_index)
    target_start = prompt + int(target_output_token_start)
    target_end = prompt + int(target_output_token_end)
    if target_end <= target_start:
        raise ValueError("Target city must contain at least one token")
    last_pre_city = target_start - 1
    last_city_predictor = target_end - 2
    if query > last_pre_city:
        raise ValueError(
            "Registered retrieval query lies after the first target-city predictor"
        )
    if requested == "registered_query_only":
        hook_positions = (query,)
    elif requested == "last_pre_city_only":
        hook_positions = (last_pre_city,)
    elif requested == "registered_query_through_last_pre_city":
        hook_positions = tuple(range(query, last_pre_city + 1))
    elif requested == "city_predictor_chain":
        hook_positions = tuple(range(last_pre_city, last_city_predictor + 1))
    else:
        hook_positions = tuple(range(query, last_city_predictor + 1))
    score_positions = tuple(range(last_pre_city, last_city_predictor + 1))
    if not hook_positions or not score_positions:
        raise RuntimeError("Compiled target-city support is empty")

    def digest(values: Sequence[int]) -> str:
        payload = json.dumps(
            [int(value) for value in values], separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return {
        "head_ablation_scope": requested,
        "head_ablation_positions": list(hook_positions),
        "head_ablation_position_count": len(hook_positions),
        "head_ablation_positions_sha256": digest(hook_positions),
        "score_positions": list(score_positions),
        "score_position_count": len(score_positions),
        "score_positions_sha256": digest(score_positions),
        "registered_query_full_sequence_token": query,
        "last_pre_city_predictor_full_sequence_token": last_pre_city,
        "last_city_predictor_full_sequence_token": last_city_predictor,
        "registered_query_equals_last_pre_city_predictor": bool(
            query == last_pre_city
        ),
        "registered_query_to_last_pre_city_distance": last_pre_city - query,
        "target_city_full_sequence_token_start": target_start,
        "target_city_full_sequence_token_end": target_end,
        "autoregressive_position_definition": (
            "full-sequence token position p predicts token p+1"
        ),
    }


@torch.inference_mode()


def _continuation_metrics(
    logits: torch.Tensor, target_ids: Sequence[int]
) -> dict[str, Any]:
    targets = torch.as_tensor(tuple(target_ids), dtype=torch.long)
    if logits.ndim != 2 or len(logits) != len(targets) or not len(targets):
        raise ValueError("Continuation logits/targets disagree")
    log_probabilities = torch.log_softmax(logits, dim=-1)
    token_logp = log_probabilities[
        torch.arange(len(targets), dtype=torch.long), targets
    ]
    row_indices = torch.arange(len(targets), dtype=torch.long)
    target_logits = logits[row_indices, targets]
    non_target_logits = logits.clone()
    non_target_logits[row_indices, targets] = -torch.inf
    nearest_competitor_logits = non_target_logits.max(dim=-1).values
    other_logsumexp = torch.logsumexp(non_target_logits, dim=-1)
    token_logit_margins = target_logits - nearest_competitor_logits
    token_log_odds = target_logits - other_logsumexp
    predictions = logits.argmax(dim=-1)
    first_target = int(targets[0])
    competing = logits[0].clone()
    competing[first_target] = -torch.inf
    top_competing_logit = float(competing.max())
    target_first_logit = float(logits[0, first_target])
    first_rank = int(
        1
        + torch.count_nonzero(
            logits[0] > logits[0, int(targets[0])]
        ).item()
    )
    return {
        "target_sequence_log_probability": float(token_logp.sum()),
        "target_mean_token_log_probability": float(token_logp.mean()),
        "target_sequence_teacher_forced_exact": bool(
            torch.equal(predictions, targets)
        ),
        "target_first_token_exact": bool(predictions[0] == targets[0]),
        "target_first_token_rank": first_rank,
        "target_sequence_target_logit_sum": float(target_logits.sum()),
        "target_mean_target_token_logit": float(target_logits.mean()),
        "target_first_token_logit": float(target_logits[0]),
        "top_non_target_first_token_logit": top_competing_logit,
        "target_sequence_logit_margin": float(token_logit_margins.sum()),
        "target_mean_token_logit_margin": float(token_logit_margins.mean()),
        "target_first_token_logit_margin": float(token_logit_margins[0]),
        "target_sequence_log_odds": float(token_log_odds.sum()),
        "target_mean_token_log_odds": float(token_log_odds.mean()),
        "target_first_token_log_odds": float(token_log_odds[0]),
        "target_city_log_probability": float(token_logp.sum()),
        "target_city_mean_token_log_probability": float(token_logp.mean()),
        "target_city_teacher_forced_exact": bool(
            torch.equal(predictions, targets)
        ),
        "target_city_first_token_exact": bool(predictions[0] == targets[0]),
        "target_city_first_token_rank": first_rank,
        "target_city_target_logit_sum": float(target_logits.sum()),
        "target_city_mean_target_token_logit": float(target_logits.mean()),
        "target_city_first_token_logit": float(target_logits[0]),
        "target_city_logit_margin": float(token_logit_margins.sum()),
        "target_city_mean_token_logit_margin": float(token_logit_margins.mean()),
        "target_city_first_token_logit_margin": float(token_logit_margins[0]),
        "target_city_log_odds": float(token_log_odds.sum()),
        "target_city_mean_token_log_odds": float(token_log_odds.mean()),
        "target_city_first_token_log_odds": float(token_log_odds[0]),
    }


@torch.inference_mode()
def run_mechanism_head_ablation_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    mechanism: str,
    heads: Sequence[tuple[int, int]],
    condition: str,
    boundary_policy: str = "strict_registered",
    anchor_equivalence_ids: Iterable[str] | None = None,
    head_ablation_scope: str = "registered_query_only",
) -> list[dict[str, Any]]:
    """Run an audited head lesion against one unchanged fixed city span."""

    specifications, excluded = mechanism_continuations(
        row,
        tokenizer,
        mechanism=mechanism,
        boundary_policy=boundary_policy,
    )
    selected_anchor_ids = (
        None
        if anchor_equivalence_ids is None
        else {str(value) for value in anchor_equivalence_ids}
    )
    if selected_anchor_ids is not None:
        specifications = [
            value
            for value in specifications
            if str(value.get("anchor_equivalence_id")) in selected_anchor_ids
        ]
        excluded = [
            value
            for value in excluded
            if str(value.get("anchor_equivalence_id")) in selected_anchor_ids
        ]
        found = {
            str(value.get("anchor_equivalence_id")) for value in specifications
        }
        missing = sorted(selected_anchor_ids - found)
        if missing:
            raise ValueError(f"Requested causal anchors are unavailable: {missing}")
    output: list[dict[str, Any]] = []
    for exclusion in excluded:
        output.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": f"{mechanism}_head_ablation",
                "condition": condition,
                "request_id": row.get("request_id", row.get("stimulus_id")),
                "model_label": row.get("model_label"),
                "seed": row.get("seed"),
                "split": row.get("split"),
                "gold_count": len(gold_records(row)),
                "heads": [[int(layer), int(head)] for layer, head in heads],
                "head_ablation_scope": str(head_ablation_scope),
                **exclusion,
            }
        )
    baseline_ids = output_token_ids(row)
    prompt_count = len(prompt_token_ids(row))
    for specification in specifications:
        query_output_index = int(specification["query_output_token_index"])
        target_output_start = int(specification["target_output_token_start"])
        target_output_end = int(specification["target_output_token_end"])
        target_encoding = build_native_causal_encoding(
            row,
            tokenizer,
            query_output_token_index=query_output_index,
            sequence_output_token_end=target_output_end,
            selected_site=specification,
        )
        target_ids = baseline_ids[target_output_start:target_output_end]
        registered_ids = tuple(int(value) for value in specification["target_token_ids"])
        if tuple(target_ids) != registered_ids:
            raise RuntimeError("Compiler target IDs differ from frozen output IDs")
        hook_position = prompt_count + query_output_index
        target_full_start = prompt_count + target_output_start
        target_full_end = prompt_count + target_output_end
        geometry = target_city_head_ablation_geometry(
            prompt_token_count=prompt_count,
            query_output_token_index=query_output_index,
            target_output_token_start=target_output_start,
            target_output_token_end=target_output_end,
            scope=str(head_ablation_scope),
        )
        if int(geometry["registered_query_full_sequence_token"]) != hook_position:
            raise RuntimeError("Compiled target-city query position changed")
        if str(head_ablation_scope) == "registered_query_only":
            # Preserve the original query-local numerical path byte-for-byte.
            logits = _fixed_target_head_ablation_logits(
                model,
                adapter,
                target_encoding,
                heads,
                hook_position=hook_position,
                target_full_sequence_token_start=target_full_start,
                target_full_sequence_token_end=target_full_end,
            )
        else:
            logits = _scheduled_head_ablation_logits(
                model,
                adapter,
                target_encoding,
                heads,
                hook_positions=geometry["head_ablation_positions"],
                score_positions=geometry["score_positions"],
            )
        output.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": f"{mechanism}_head_ablation",
                "condition": condition,
                "request_id": target_encoding.request_id,
                "model_label": target_encoding.model_label,
                "seed": target_encoding.seed,
                "split": target_encoding.split,
                "gold_count": target_encoding.count,
                "heads": [[int(layer), int(head)] for layer, head in heads],
                "bank_size": len(heads),
                "status": "ok",
                "target_token_count": len(target_ids),
                "teacher_forced_interstitial_token_count": (
                    target_output_start - query_output_index - 1
                ),
                **geometry,
                "target_text": tokenizer.decode(
                    list(target_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                **specification,
                **_continuation_metrics(logits, target_ids),
            }
        )
    return output


@torch.inference_mode()
def run_answer_query_head_ablation_trial(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    heads: Sequence[tuple[int, int]],
    condition: str,
    site_id: str = "answer_query_v3",
) -> dict[str, Any]:
    """Test one frozen head bank at the true final answer-query token."""

    encoding = build_native_trace_encoding(
        row,
        tokenizer,
        site_id=site_id,
        candidate_counts=tuple(range(1, 11)),
    )
    answer_ids = dict(encoding.count_candidate_answer_token_ids)[encoding.count]
    # ``_local_head_ablation_logits`` expects the teacher-forced target tokens
    # to be present after the query position.  Without this extension,
    # ``logits_to_keep=target_count+1`` returns the query's predecessor as the
    # first kept position for a prefix-only answer encoding, silently scoring
    # an off-by-one logit.  Keep ``query_position`` fixed at the literal token
    # immediately before the numeric answer and append only the frozen gold
    # continuation used for scoring.
    scoring_encoding = replace(
        encoding,
        input_ids=encoding.input_ids + tuple(int(value) for value in answer_ids),
        attention_mask=encoding.attention_mask + (1,) * len(answer_ids),
    )
    logits = _local_head_ablation_logits(
        model,
        adapter,
        scoring_encoding,
        heads,
        hook_position=encoding.query_position,
        target_token_count=len(answer_ids),
    )
    metrics = _continuation_metrics(logits, answer_ids)
    generation = generate_with_head_ablation_at_positions(
        model,
        tokenizer,
        adapter,
        encoding,
        heads,
        hook_positions=[int(encoding.query_position)],
        max_new_tokens=16,
    )
    behavior = completion_metrics(generation, gold_count=encoding.count)
    probabilities = torch.softmax(logits[0], dim=-1)
    first_id = int(answer_ids[0])
    candidate_first_ids = {
        int(count): int(ids[0])
        for count, ids in encoding.count_candidate_answer_token_ids
    }
    candidate_logits = {
        count: float(logits[0, token_id])
        for count, token_id in candidate_first_ids.items()
    }
    competing_counts = {
        count: value
        for count, value in candidate_logits.items()
        if int(candidate_first_ids[count]) != first_id
    }
    best_candidate_logit = max(candidate_logits.values())
    candidate_argmax_counts = sorted(
        count
        for count, value in candidate_logits.items()
        if value == best_candidate_logit
    )
    target_token_alias_counts = sorted(
        count
        for count, token_id in candidate_first_ids.items()
        if int(token_id) == first_id
    )
    candidate_margin = (
        float(candidate_logits[encoding.count])
        - float(max(competing_counts.values()))
        if competing_counts
        else float("nan")
    )
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "answer_query_head_ablation_v5_broad_factorial",
        "condition": condition,
        "request_id": encoding.request_id,
        "stimulus_id": encoding.stimulus_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "split": encoding.split,
        "gold_count": encoding.count,
        "baseline_exact_count": bool(parse_trace_record(row)["exact_count"]),
        "site_id": site_id,
        "query_position": int(encoding.query_position),
        "query_definition": (
            "literal baseline token immediately before the first numeric "
            "answer token"
        ),
        "heads": [[int(layer), int(head)] for layer, head in heads],
        "bank_size": int(len(heads)),
        "status": "ok",
        "target_token_count": int(len(answer_ids)),
        "target_text": tokenizer.decode(
            list(answer_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "target_first_token_probability": float(probabilities[first_id]),
        "behavioral_endpoint": "greedy_generated_numeric_answer",
        "prediction": behavior["prediction"],
        "exact_count": behavior["exact_count"],
        "signed_error": behavior["signed_error"],
        "absolute_error": behavior["absolute_error"],
        "completion_text_raw": behavior["completion_text_raw"],
        "generated_token_count": behavior["generated_token_count"],
        "generation_truncated": behavior["generation_truncated"],
        "head_ablation_hook_audit": generation[
            "head_ablation_hook_audit"
        ],
        "answer_candidate_argmax_counts": candidate_argmax_counts,
        "answer_candidate_argmax_contains_gold": bool(
            int(encoding.count) in candidate_argmax_counts
        ),
        "answer_gold_first_token_alias_counts": target_token_alias_counts,
        "answer_gold_first_token_identifies_unique_count": bool(
            len(target_token_alias_counts) == 1
        ),
        "answer_candidate_first_token_logits": candidate_logits,
        "answer_gold_vs_best_distinct_count_token_logit_margin": (
            candidate_margin
        ),
        **metrics,
    }


@torch.inference_mode()
def run_retrieval_source_edge_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    heads: Sequence[tuple[int, int]],
    anchor_equivalence_id: str,
) -> list[dict[str, Any]]:
    """Remove and restore record-specific OV writes at one city-pre anchor.

    Source vectors are frozen from the unperturbed query. Each selected layer
    group is intervened on separately, so a write captured at layer ``l`` is
    not made stale by an earlier-layer intervention. At each layer, the wrong
    gold record with the closest *natural* source-write norm is selected without
    consulting any outcome. Both its natural write and an exactly norm-matched
    version are retained as controls.
    """

    specifications, _excluded = mechanism_continuations(
        row, tokenizer, mechanism="retrieval_anchor_localization"
    )
    matches = [
        value
        for value in specifications
        if str(value.get("anchor_equivalence_id")) == str(anchor_equivalence_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source-edge anchor {anchor_equivalence_id!r}, "
            f"found {len(matches)}"
        )
    specification = matches[0]
    if "city_pre_d1" not in set(specification.get("anchor_roles", [])):
        raise ValueError("Source-edge trials are registered only at city_pre_d1")
    target_city = str(specification["target_city"])
    previous_city = str(specification["source_control_city"])
    if not previous_city or previous_city.casefold() == target_city.casefold():
        raise ValueError("Previous-city metadata is missing or equals target")

    by_layer: dict[int, list[tuple[int, int]]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid source-edge layer: {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid source-edge head L{layer}H{head}")
        by_layer.setdefault(layer, []).append((layer, head))
    if not by_layer:
        raise ValueError("Source-edge trials require a non-empty frozen head bank")
    restoration_layer = int(heads[0][0])

    baseline_ids = output_token_ids(row)
    prompt_count = len(prompt_token_ids(row))
    query_output_index = int(specification["query_output_token_index"])
    target_output_start = int(specification["target_output_token_start"])
    target_output_end = int(specification["target_output_token_end"])
    target_encoding = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=query_output_index,
        sequence_output_token_end=target_output_end,
        selected_site=specification,
    )
    target_ids = baseline_ids[target_output_start:target_output_end]
    registered_ids = tuple(int(value) for value in specification["target_token_ids"])
    if tuple(target_ids) != registered_ids:
        raise RuntimeError("Compiler target IDs differ from frozen output IDs")
    hook_position = prompt_count + query_output_index
    target_full_start = prompt_count + target_output_start
    target_full_end = prompt_count + target_output_end
    active_layers = tuple(sorted(by_layer))
    wrong_cities = sorted(
        (
            span.city
            for span in target_encoding.prompt_record_spans
            if span.city.casefold() != target_city.casefold()
        ),
        key=str.casefold,
    )
    captures = capture_source_specific_head_writes_multi(
        model,
        adapter,
        target_encoding,
        source_cities=[target_city, *wrong_cities],
        layers=active_layers,
    )
    correct_metrics, correct_writes = captures[target_city]
    clean_logits = _fixed_target_head_write_intervention_logits(
        model,
        adapter,
        target_encoding,
        hook_position=hook_position,
        target_full_sequence_token_start=target_full_start,
        target_full_sequence_token_end=target_full_end,
    )
    clean_outcomes = _continuation_metrics(clean_logits, target_ids)
    common = {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "retrieval_source_edge_intervention",
        "mechanism": "retrieval_anchor_localization",
        "request_id": target_encoding.request_id,
        "model_label": target_encoding.model_label,
        "seed": target_encoding.seed,
        "split": target_encoding.split,
        "gold_count": target_encoding.count,
        "status": "ok",
        "trial_complete": True,
        "heads": [[int(layer), int(head)] for layer, head in heads],
        "bank_size": len(heads),
        "target_token_count": len(target_ids),
        "teacher_forced_interstitial_token_count": (
            target_output_start - query_output_index - 1
        ),
        "target_text": tokenizer.decode(
            list(target_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "correct_source_city": target_city,
        "previous_source_city": previous_city,
        "source_control_policy": (
            "per_layer_closest_natural_write_norm_wrong_gold_record_with_"
            "natural_and_exact_residual_norm_matched_arms"
        ),
        "source_vector_state": "unperturbed_query_eager_attention",
        "source_vector_application": "post_o_residual_delta_at_query",
        "intervention_scope": "one_selected_layer_group_at_a_time",
        "restoration_layer": restoration_layer,
        "restoration_layer_policy": "first_head_layer_in_crossfit_ranked_bank",
        **specification,
    }
    output: list[dict[str, Any]] = []
    for layer in active_layers:
        layer_heads = tuple(sorted(set(by_layer[layer])))
        correct = _summed_source_write(correct_writes, layer_heads).detach().float()
        wrong_vectors = {
            city: _summed_source_write(captures[city][1], layer_heads)
            .detach()
            .float()
            for city in wrong_cities
        }
        control_city, control = _closest_natural_norm_source(correct, wrong_vectors)
        control_metrics, _control_writes = captures[control_city]
        matched_control, scale = _norm_matched_vector(control, correct)
        correct_norm = float(torch.linalg.vector_norm(correct))
        control_norm = float(torch.linalg.vector_norm(control))
        matched_norm = float(torch.linalg.vector_norm(matched_control.float()))
        correct_rows = correct_metrics.loc[
            correct_metrics["layer"].eq(layer)
            & correct_metrics["head"].isin([head for _layer, head in layer_heads])
        ]
        control_rows = control_metrics.loc[
            control_metrics["layer"].eq(layer)
            & control_metrics["head"].isin([head for _layer, head in layer_heads])
        ]
        layer_metadata = {
            "intervention_layer": int(layer),
            "control_source_city": control_city,
            "intervention_layer_heads": [
                [int(item_layer), int(head)] for item_layer, head in layer_heads
            ],
            "intervention_layer_head_count": len(layer_heads),
            "correct_source_write_norm": correct_norm,
            "control_source_write_norm_unscaled": control_norm,
            "control_source_write_norm_matched": matched_norm,
            "control_source_norm_scale": float(scale),
            "correct_source_attention_mass_sum": float(
                correct_rows["source_attention_mass"].sum()
            ),
            "control_source_attention_mass_sum": float(
                control_rows["source_attention_mass"].sum()
            ),
        }
        output.append(
            {
                **common,
                **layer_metadata,
                "condition": "clean",
                "zero_selected_layer_heads": False,
                "applied_source_delta": "none",
                **clean_outcomes,
            }
        )
        conditions: tuple[
            tuple[str, Sequence[tuple[int, int]], Mapping[int, torch.Tensor], str],
            ...,
        ] = (
            (
                "correct_source_removal",
                (),
                {layer: -correct},
                "negative_correct_source_write",
            ),
            (
                "matched_wrong_source_removal_natural",
                (),
                {layer: -control},
                "negative_natural_wrong_source_write",
            ),
            (
                "matched_wrong_source_removal_norm_matched",
                (),
                {layer: -matched_control},
                "negative_norm_matched_wrong_source_write",
            ),
        )
        if layer == restoration_layer:
            conditions += (
                (
                    "selected_layer_head_ablation",
                    layer_heads,
                    {},
                    "none",
                ),
                (
                    "selected_layer_head_ablation_correct_restore",
                    layer_heads,
                    {layer: correct},
                    "positive_correct_source_write",
                ),
                (
                    "selected_layer_head_ablation_matched_wrong_restore_natural",
                    layer_heads,
                    {layer: control},
                    "positive_natural_wrong_source_write",
                ),
                (
                    "selected_layer_head_ablation_matched_wrong_restore_norm_matched",
                    layer_heads,
                    {layer: matched_control},
                    "positive_norm_matched_wrong_source_write",
                ),
            )
        for condition, zero_heads, deltas, delta_label in conditions:
            logits = _fixed_target_head_write_intervention_logits(
                model,
                adapter,
                target_encoding,
                zero_heads=zero_heads,
                output_deltas=deltas,
                hook_position=hook_position,
                target_full_sequence_token_start=target_full_start,
                target_full_sequence_token_end=target_full_end,
            )
            output.append(
                {
                    **common,
                    **layer_metadata,
                    "condition": condition,
                    "zero_selected_layer_heads": bool(zero_heads),
                    "applied_source_delta": delta_label,
                    **_continuation_metrics(logits, target_ids),
                }
            )
    return output


def _first_generated_gold_city(
    text: str, cities: Sequence[str]
) -> tuple[str | None, int | None]:
    """Return the earliest whole-name gold city in a free continuation."""

    hits: list[tuple[int, int, str]] = []
    for city in cities:
        match = re.search(
            rf"(?<![\w]){re.escape(str(city))}(?![\w])",
            str(text),
            flags=re.IGNORECASE,
        )
        if match is not None:
            hits.append((int(match.start()), -len(str(city)), str(city)))
    if not hits:
        return None, None
    start, _negative_length, city = min(hits)
    return city, start


_CITY_NAME_FRAGMENT = (
    r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*"
    r"(?:[ \t]+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*){0,4}"
)
_GENERATED_CITY_RECORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "received_score",
        re.compile(
            rf"(?P<city>{_CITY_NAME_FRAGMENT})[ \t]+received[ \t]+a[ \t]+"
            r"(?:numeric[ \t]+)?score\b"
        ),
    ),
    (
        "entry_or_record_for",
        re.compile(
            rf"\b(?:entry|record)[ \t]+for[ \t]+"
            rf"(?P<city>{_CITY_NAME_FRAGMENT})[ \t]+with\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "city_with_score",
        re.compile(
            rf"(?P<city>{_CITY_NAME_FRAGMENT})[ \t]+with[ \t]+"
            r"(?:a[ \t]+)?(?:numeric[ \t]+)?score\b"
        ),
    ),
    (
        "city_dash_or_colon_score",
        re.compile(
            rf"(?P<city>{_CITY_NAME_FRAGMENT})[ \t]*[-:][ \t]*"
            r"(?:score[ \t]*[:=][ \t]*)?\d{1,3}\b"
        ),
    ),
)
_NON_CITY_RECORD_LABELS = frozenset(
    {"answer", "count", "final answer", "score", "total"}
)


def _first_generated_city_record(
    text: str, known_cities: Sequence[str]
) -> tuple[str | None, int | None, str | None]:
    """Return the first semantic city record, including non-gold cities.

    Gold-only matching can incorrectly score a continuation that first emits
    a distractor or hallucinated city and later self-corrects to the target.
    The generic patterns therefore recognize the task's common record
    surfaces without requiring the emitted city to be in the oracle registry.
    """

    candidates: list[tuple[int, int, int, str, str]] = []
    known_city, known_start = _first_generated_gold_city(text, known_cities)
    if known_city is not None and known_start is not None:
        candidates.append(
            (
                int(known_start),
                1,
                -len(known_city),
                str(known_city),
                "known_city",
            )
        )
    for evidence_kind, pattern in _GENERATED_CITY_RECORD_PATTERNS:
        for match in pattern.finditer(str(text)):
            city = str(match.group("city")).strip(" \t\"'“”‘’[]()")
            if not city or city.casefold() in _NON_CITY_RECORD_LABELS:
                continue
            candidates.append(
                (
                    int(match.start("city")),
                    0,
                    -len(city),
                    city,
                    evidence_kind,
                )
            )
    if not candidates:
        return None, None, None
    start, _evidence_priority, _negative_length, city, evidence_kind = min(
        candidates
    )
    return city, start, evidence_kind


def _retrieval_behavior_score(
    text: str,
    *,
    expected_city: str,
    gold_cities: Sequence[str],
    exact_target_prefix: bool,
) -> dict[str, Any]:
    """Score the first emitted semantic record, retaining the legacy audit."""

    legacy_city, legacy_start = _first_generated_gold_city(text, gold_cities)
    legacy_correct = bool(
        (
            legacy_city is not None
            and legacy_city.casefold() == str(expected_city).casefold()
        )
        or (legacy_city is None and exact_target_prefix)
    )
    first_city, first_start, evidence_kind = _first_generated_city_record(
        text, gold_cities
    )
    correct = bool(
        (
            first_city is not None
            and first_city.casefold() == str(expected_city).casefold()
        )
        or (first_city is None and exact_target_prefix)
    )
    gold_keys = {str(city).casefold() for city in gold_cities}
    if correct:
        outcome = "correct_next_needle"
    elif first_city is None:
        outcome = "no_identifiable_city_record"
    elif first_city.casefold() in gold_keys:
        outcome = "wrong_gold_needle"
    else:
        outcome = "wrong_non_gold_city_record"
    return {
        "behavior_scoring_policy": (
            "first_semantic_city_record_v3_reserved_label_exclusion"
        ),
        "first_generated_city_record": first_city,
        "first_generated_city_record_char_start": first_start,
        "first_generated_city_record_evidence": evidence_kind,
        "first_generated_city_record_in_gold_registry": bool(
            first_city is not None and first_city.casefold() in gold_keys
        ),
        "correct_next_needle": correct,
        "behavior_outcome": outcome,
        "legacy_first_generated_gold_city": legacy_city,
        "legacy_first_generated_gold_city_char_start": legacy_start,
        "legacy_correct_next_needle": legacy_correct,
    }


@torch.inference_mode()
def _generate_with_prefill_head_ablation(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    hook_position: int | Sequence[int],
    max_new_tokens: int,
    decode_head_ablation_steps: int = 0,
) -> dict[str, Any]:
    """Ablate selected prefill sites and an optional fixed decode window.

    Positive ``decode_head_ablation_steps`` counts one-token cached decode
    forwards *after* the prefill; ``-1`` keeps the bank off for every decode
    forward.  The prefill itself controls the first generated token; decode
    forward 1 controls generated token 2, and so on.  Keeping this policy
    explicit distinguishes immediate path disruption from a model that simply
    retries retrieval at a newly generated query token.
    """

    if isinstance(hook_position, int):
        hook_positions = (int(hook_position),)
    else:
        hook_positions = tuple(
            sorted({int(position) for position in hook_position})
        )
    if not hook_positions:
        raise ValueError("Behavioral ablation requires a prefill position")
    if hook_positions[0] < 0:
        raise ValueError("Behavioral ablation positions must be non-negative")
    decode_steps = int(decode_head_ablation_steps)
    if decode_steps < -1:
        raise ValueError(
            "decode_head_ablation_steps must be -1 or non-negative"
        )
    persistent_decode_ablation = decode_steps == -1

    by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid behavioral ablation layer: {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid behavioral head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    handles = []
    prefill_applications = {layer: 0 for layer in by_layer}
    decode_applications = {layer: 0 for layer in by_layer}
    selected_post_zero_max_abs = {layer: 0.0 for layer in by_layer}
    expected_prefill_length = int(encoding.sequence_length)
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])
        expected_width = int(adapter.num_heads[layer]) * head_dim

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            layer_heads: tuple[int, ...] = tuple(sorted(set(layer_heads))),
            head_dim: int = head_dim,
            expected_width: int = expected_width,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention projection received no head tensor")
            value = args[0]
            if value.ndim != 3:
                raise RuntimeError("Behavioral head tensor must be [batch,time,width]")
            if int(value.shape[-1]) != expected_width:
                raise RuntimeError(
                    f"L{layer} o_proj input width {value.shape[-1]} disagrees "
                    f"with decoder adapter width {expected_width}"
                )
            sequence_length = int(value.shape[1])
            if (
                sequence_length == expected_prefill_length
                and prefill_applications[layer] == 0
            ):
                positions = hook_positions
                prefill_applications[layer] += 1
            elif (
                sequence_length == 1
                and prefill_applications[layer] == 1
                and (
                    persistent_decode_ablation
                    or decode_applications[layer] < decode_steps
                )
            ):
                positions = (0,)
                decode_applications[layer] += 1
            else:
                return None
            patched = value.clone()
            for head in layer_heads:
                left = head * head_dim
                patched[
                    :, list(positions), left : left + head_dim
                ] = 0
            selected = torch.cat(
                [
                    patched[:, list(positions), head * head_dim : (head + 1) * head_dim]
                    .detach()
                    .reshape(-1)
                    for head in layer_heads
                ]
            )
            if selected.numel():
                selected_post_zero_max_abs[layer] = max(
                    selected_post_zero_max_abs[layer],
                    float(selected.abs().max().float().cpu()),
                )
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    try:
        completion = generate_answer_completion(
            model,
            tokenizer,
            encoding,
            max_new_tokens=max_new_tokens,
        )
    finally:
        for handle in handles:
            handle.remove()
    prefill_violations = {
        layer: count
        for layer, count in prefill_applications.items()
        if count != 1
    }
    if prefill_violations:
        raise RuntimeError(
            "Behavioral ablation must apply exactly once in the full prefill: "
            f"{prefill_violations}"
        )
    decode_counts = set(decode_applications.values())
    if len(decode_counts) > 1:
        raise RuntimeError(
            "Behavioral decode-window applications disagree across layers: "
            f"{decode_applications}"
        )
    if any(value != 0.0 for value in selected_post_zero_max_abs.values()):
        raise RuntimeError(
            "Selected pre-O head slices were not exactly zero after ablation"
        )
    completion.pop("full_answer_text", None)
    observed_decode_steps = (
        next(iter(decode_counts)) if decode_counts else 0
    )
    completion.update(
        {
            "head_ablation_prefill_layer_applications": {
                str(layer): int(count)
                for layer, count in sorted(prefill_applications.items())
            },
            "head_ablation_decode_layer_applications": {
                str(layer): int(count)
                for layer, count in sorted(decode_applications.items())
            },
            "head_ablation_decode_steps_requested": decode_steps,
            "head_ablation_decode_steps_observed": int(observed_decode_steps),
            "head_ablation_decode_policy": (
                "all_one_token_cached_decode_forwards"
                if persistent_decode_ablation
                else "first_n_one_token_cached_decode_forwards"
                if decode_steps > 0
                else "prefill_only"
            ),
            "head_ablation_o_proj_input_width_validated": bool(by_layer),
            "head_ablation_selected_post_zero_max_abs": (
                max(selected_post_zero_max_abs.values())
                if selected_post_zero_max_abs
                else None
            ),
        }
    )
    return completion


@torch.inference_mode()
def run_retrieval_head_behavior_trial(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    heads: Sequence[tuple[int, int]],
    condition: str,
    anchor_equivalence_id: str | Sequence[str],
    max_new_tokens: int = 32,
    decode_head_ablation_steps: int = 0,
) -> dict[str, Any]:
    """Ablate registered pre-city sites and record the emitted needle.

    A sequence of anchor IDs defines a fixed multi-site retrieval window for
    one transition.  Prefill interventions occur together.  Generation
    branches after the latest site; an explicitly requested fixed decode
    window can keep the same bank ablated while the model attempts to retry.
    """

    specifications, _excluded = mechanism_continuations(
        row, tokenizer, mechanism="retrieval_anchor_localization"
    )
    if isinstance(anchor_equivalence_id, str):
        requested_anchor_ids = (str(anchor_equivalence_id),)
    else:
        requested_anchor_ids = tuple(
            dict.fromkeys(str(value) for value in anchor_equivalence_id)
        )
    if not requested_anchor_ids:
        raise ValueError("At least one behavioral anchor is required")
    by_anchor_id = {
        str(value.get("anchor_equivalence_id")): value
        for value in specifications
    }
    missing_anchor_ids = [
        value for value in requested_anchor_ids if value not in by_anchor_id
    ]
    if missing_anchor_ids:
        raise ValueError(
            "Behavioral anchors are absent from the compiled transition: "
            f"{missing_anchor_ids}"
        )
    matches = [by_anchor_id[value] for value in requested_anchor_ids]
    transition_keys = {
        (
            int(value["from_occurrence"]),
            int(value["to_occurrence"]),
            int(value["target_output_token_start"]),
            int(value["target_output_token_end"]),
            str(value["target_city"]).casefold(),
        )
        for value in matches
    }
    if len(transition_keys) != 1:
        raise ValueError(
            "A multi-site behavioral window must stay within one transition"
        )
    matches.sort(key=lambda value: int(value["query_output_token_index"]))
    specification = matches[-1]
    query_output_indices = tuple(
        sorted(
            {
                int(value["query_output_token_index"])
                for value in matches
            }
        )
    )
    query_output_index = int(specification["query_output_token_index"])
    target_output_start = int(specification["target_output_token_start"])
    if query_output_index >= target_output_start:
        raise ValueError(
            "Retrieval behavior requires an anchor strictly before target city"
        )
    prefix_encoding = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=query_output_index,
        sequence_output_token_end=query_output_index + 1,
        selected_site=specification,
    )
    prompt_count = len(prompt_token_ids(row))
    hook_positions = tuple(
        prompt_count + output_index for output_index in query_output_indices
    )
    if hook_positions[-1] != int(prefix_encoding.sequence_length) - 1:
        raise RuntimeError("Behavioral city-pre anchor is not the branch-prefix end")
    completion = _generate_with_prefill_head_ablation(
        model,
        tokenizer,
        adapter,
        prefix_encoding,
        heads,
        hook_position=hook_positions,
        max_new_tokens=max_new_tokens,
        decode_head_ablation_steps=decode_head_ablation_steps,
    )
    generated_ids = tuple(int(value) for value in completion["generated_token_ids"])
    target_ids = tuple(int(value) for value in specification["target_token_ids"])
    target_token_offset = target_output_start - (query_output_index + 1)
    exact_target_at_registered_offset = (
        generated_ids[
            target_token_offset : target_token_offset + len(target_ids)
        ]
        == target_ids
    )
    exact_target_prefix = bool(
        target_token_offset == 0 and exact_target_at_registered_offset
    )
    gold_cities = tuple(span.city for span in prefix_encoding.prompt_record_spans)
    expected_city = str(specification["target_city"])
    behavior = _retrieval_behavior_score(
        str(completion["completion_text"]),
        expected_city=expected_city,
        gold_cities=gold_cities,
        exact_target_prefix=exact_target_prefix,
    )
    intervention_roles = tuple(
        dict.fromkeys(
            str(role)
            for value in matches
            for role in value.get("anchor_roles", [])
        )
    )
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "retrieval_head_free_continuation",
        "mechanism": "retrieval_anchor_localization",
        "condition": str(condition),
        "request_id": prefix_encoding.request_id,
        "model_label": prefix_encoding.model_label,
        "seed": prefix_encoding.seed,
        "split": prefix_encoding.split,
        "gold_count": prefix_encoding.count,
        "status": "ok",
        "trial_complete": True,
        "heads": [[int(layer), int(head)] for layer, head in heads],
        "bank_size": len(heads),
        "branch_policy": (
            (
                "teacher_force_through_latest_registered_anchor_then_"
                "persistent_decode_head_ablation"
            )
            if int(decode_head_ablation_steps) == -1
            and len(query_output_indices) > 1
            else (
                "teacher_force_through_registered_anchor_then_"
                "persistent_decode_head_ablation"
            )
            if int(decode_head_ablation_steps) == -1
            else
            (
                "teacher_force_through_latest_registered_anchor_then_"
                "fixed_decode_head_ablation_window_then_free_generate"
            )
            if int(decode_head_ablation_steps) > 0
            and len(query_output_indices) > 1
            else (
                "teacher_force_through_registered_anchor_then_"
                "fixed_decode_head_ablation_window_then_free_generate"
            )
            if int(decode_head_ablation_steps) > 0
            else "teacher_force_through_latest_registered_anchor_then_free_generate"
            if len(query_output_indices) > 1
            else "teacher_force_through_registered_anchor_then_free_generate"
        ),
        "head_ablation_tensor_site": "attention_output_projection_input_pre_o",
        "head_ablation_operation": "zero_selected_head_slices",
        "head_ablation_prefill_only": int(decode_head_ablation_steps) == 0,
        "head_ablation_decode_steps_untouched": (
            int(decode_head_ablation_steps) == 0
        ),
        "head_ablation_decode_window_policy": (
            "all_one_token_cached_decode_forwards_after_prefill"
            if int(decode_head_ablation_steps) == -1
            else "first_n_one_token_cached_decode_forwards_after_prefill"
        ),
        "same_selected_bank_at_all_intervention_sites": True,
        "branch_anchor_roles": list(specification.get("anchor_roles", [])),
        "intervention_anchor_equivalence_ids": list(requested_anchor_ids),
        "intervention_anchor_roles": list(intervention_roles),
        "intervention_output_token_indices": list(query_output_indices),
        "intervention_full_sequence_token_indices": list(hook_positions),
        "intervention_site_count": len(query_output_indices),
        "branch_to_target_token_distance": target_token_offset,
        "free_generation_max_new_tokens": int(max_new_tokens),
        "expected_next_city": expected_city,
        "expected_next_city_token_ids": list(target_ids),
        "generated_exact_target_city_token_prefix": exact_target_prefix,
        "generated_target_city_exact_at_registered_path_offset": (
            exact_target_at_registered_offset
        ),
        "first_generated_gold_city": behavior[
            "legacy_first_generated_gold_city"
        ],
        "first_generated_gold_city_char_start": behavior[
            "legacy_first_generated_gold_city_char_start"
        ],
        **behavior,
        **specification,
        **completion,
    }


@torch.inference_mode()
def run_projected_patch_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    receiver_row: Mapping[str, Any],
    donor_row: Mapping[str, Any],
    *,
    receiver_site_id: str,
    donor_site_id: str,
    layer: int,
    basis: np.ndarray | torch.Tensor,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Run a subspace donor patch with preregistered matched controls."""

    receiver, receiver_state = capture_site_state(
        model,
        adapter,
        tokenizer,
        receiver_row,
        site_id=receiver_site_id,
        layer=layer,
    )
    donor, donor_state = capture_site_state(
        model,
        adapter,
        tokenizer,
        donor_row,
        site_id=donor_site_id,
        layer=layer,
    )
    return run_projected_patch_trials_from_states(
        model,
        tokenizer,
        adapter,
        receiver,
        receiver_state,
        donor,
        donor_state,
        receiver_site_id=receiver_site_id,
        donor_site_id=donor_site_id,
        layer=layer,
        basis=basis,
        max_new_tokens=max_new_tokens,
    )


@torch.inference_mode()
def run_projected_patch_trials_from_states(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    receiver: NativeTraceEncoding,
    receiver_state: torch.Tensor,
    donor: NativeTraceEncoding,
    donor_state: torch.Tensor,
    *,
    receiver_site_id: str,
    donor_site_id: str,
    layer: int,
    basis: np.ndarray | torch.Tensor | None,
    max_new_tokens: int = 16,
    self_patch_result: Mapping[str, Any] | None = None,
    requested_conditions: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run requested answer-query patch conditions from reusable states.

    The historical four-arm protocol remains the default.  A full-state-only
    layer sweep may request just ``self_patch`` and ``full_donor_patch``; in
    that case no fitted subspace is needed.  Keeping that path explicit avoids
    spending two additional long-context generations per pair/layer on
    projected and orthogonal controls that answer a different question.
    """

    if receiver.model_label != donor.model_label:
        raise ValueError("Receiver and donor must use the same registered model")
    if receiver.count == donor.count:
        raise ValueError(
            "Answer-query donor trials require different receiver/donor counts; "
            "a self patch may be requested as the regeneration control"
        )
    allowed_conditions = {
        "self_patch",
        "full_donor_patch",
        "projected_donor_patch",
        "orthogonal_norm_matched",
    }
    requested = tuple(
        requested_conditions
        or (
            "self_patch",
            "full_donor_patch",
            "projected_donor_patch",
            "orthogonal_norm_matched",
        )
    )
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Patch conditions must be non-empty and unique")
    unknown = sorted(set(requested) - allowed_conditions)
    if unknown:
        raise ValueError(f"Unknown answer-query patch conditions: {unknown}")
    needs_subspace = bool(
        {"projected_donor_patch", "orthogonal_norm_matched"} & set(requested)
    )
    basis_tensor: torch.Tensor | None = None
    if needs_subspace:
        if basis is None:
            raise ValueError("Projected/orthogonal patch conditions require a basis")
        basis_tensor = torch.as_tensor(basis, dtype=torch.float32)
        if (
            basis_tensor.ndim != 2
            or basis_tensor.shape[0] != receiver_state.numel()
        ):
            raise ValueError("Patch basis must have shape [hidden, rank]")
    reused_self_patch = self_patch_result is not None
    results: dict[str, dict[str, Any]] = {}
    if "self_patch" in requested:
        results["self_patch"] = (
            dict(self_patch_result)
            if self_patch_result is not None
            else generate_with_residual_interventions(
                model,
                tokenizer,
                adapter,
                receiver,
                {int(layer): ([receiver.query_position], receiver_state)},
                max_new_tokens=max_new_tokens,
            )
        )
    if "full_donor_patch" in requested:
        results["full_donor_patch"] = generate_with_residual_interventions(
            model,
            tokenizer,
            adapter,
            receiver,
            {int(layer): ([receiver.query_position], donor_state)},
            max_new_tokens=max_new_tokens,
        )
    audit: dict[str, Any] = {
        "source_layer": int(layer),
        "source_positions": [int(receiver.query_position)],
        "full_delta_norm": float(
            torch.linalg.vector_norm(
                donor_state.detach().float() - receiver_state.detach().float()
            )
        ),
        "requested_conditions": list(requested),
    }
    if needs_subspace:
        assert basis_tensor is not None
        subspace_results = run_counter_subspace_conditions(
            model,
            tokenizer,
            adapter,
            receiver,
            source_layer=int(layer),
            source_positions=[receiver.query_position],
            receiver_source_state=receiver_state,
            donor_source_state=donor_state,
            source_basis=basis_tensor,
            random_seed=_stable_seed(
                f"v5-patch:{receiver.request_id}:{donor.request_id}:L{layer}"
            )
            % (2**31 - 1),
            max_new_tokens=max_new_tokens,
        )
        audit.update(dict(subspace_results.pop("_audit")))
        if "projected_donor_patch" in requested:
            results["projected_donor_patch"] = subspace_results["projected_patch"]
        if "orthogonal_norm_matched" in requested:
            results["orthogonal_norm_matched"] = subspace_results[
                "orthogonal_norm_matched"
            ]
    result_rows: list[dict[str, Any]] = []
    for condition in requested:
        result = results[condition]
        result_rows.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": "answer_query_patch",
                "condition": condition,
                "request_id": receiver.request_id,
                "donor_request_id": donor.request_id,
                "model_label": receiver.model_label,
                "seed": receiver.seed,
                "split": receiver.split,
                "gold_count": receiver.count,
                "donor_count": donor.count,
                "receiver_site_id": receiver_site_id,
                "donor_site_id": donor_site_id,
                "layer": int(layer),
                "rank": (
                    None if basis_tensor is None else int(basis_tensor.shape[1])
                ),
                "captured_state_cache_audit": "PASS_REUSED_SITE_STATE",
                "self_patch_cache_reused": bool(reused_self_patch),
                **audit,
                **completion_metrics(result, gold_count=receiver.count),
            }
        )
    return result_rows


@torch.inference_mode()
def capture_site_state(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    site_id: str,
    layer: int,
) -> tuple[NativeTraceEncoding, torch.Tensor]:
    encoding = build_native_trace_encoding(row, tokenizer, site_id=site_id)
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        [encoding.query_position],
        layers=[layer],
    )
    return encoding, captured[int(layer)][0]


@torch.inference_mode()
def run_residual_patch_trial(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    receiver_row: Mapping[str, Any],
    donor_row: Mapping[str, Any],
    *,
    receiver_site_id: str,
    donor_site_id: str,
    layer: int,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    receiver = build_native_trace_encoding(
        receiver_row, tokenizer, site_id=receiver_site_id
    )
    donor, donor_state = capture_site_state(
        model,
        adapter,
        tokenizer,
        donor_row,
        site_id=donor_site_id,
        layer=layer,
    )
    result = generate_with_residual_interventions(
        model,
        tokenizer,
        adapter,
        receiver,
        {int(layer): ([receiver.query_position], donor_state)},
        max_new_tokens=max_new_tokens,
    )
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "answer_query_patch",
        "condition": "donor_patch",
        "request_id": receiver.request_id,
        "donor_request_id": donor.request_id,
        "model_label": receiver.model_label,
        "seed": receiver.seed,
        "split": receiver.split,
        "gold_count": receiver.count,
        "donor_count": donor.count,
        "receiver_site_id": receiver_site_id,
        "donor_site_id": donor_site_id,
        "layer": int(layer),
        **completion_metrics(result, gold_count=receiver.count),
    }


def fit_centroid_subspace(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    if states.ndim != 2 or len(states) != len(labels):
        raise ValueError("states/labels must be [observations, hidden]/[observations]")
    if int(rank) < 1:
        raise ValueError("rank must be positive")
    values = np.unique(labels)
    if len(values) < 2:
        raise ValueError("At least two labels are needed to fit a count subspace")
    centroids = np.stack([states[labels == value].mean(axis=0) for value in values])
    center = centroids.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centroids - center, full_matrices=False)
    effective_rank = min(int(rank), len(values) - 1, states.shape[1])
    basis = vh[:effective_rank].T
    centered_values = values.astype(float) - float(np.mean(values))
    scores = (centroids - center) @ basis
    for component in range(effective_rank):
        orientation = float(np.dot(centered_values, scores[:, component]))
        if orientation < 0:
            basis[:, component] *= -1
        elif abs(orientation) <= 1e-12:
            anchor = int(np.argmax(np.abs(basis[:, component])))
            if basis[anchor, component] < 0:
                basis[:, component] *= -1
    return center.astype(np.float32), basis.astype(np.float32)


@torch.inference_mode()
def run_subspace_ablation(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    site_id: str,
    layer: int,
    center: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
    dose: float = 1.0,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    encoding = build_native_trace_encoding(row, tokenizer, site_id=site_id)
    center_tensor = torch.as_tensor(center, dtype=torch.float32)
    basis_tensor = torch.as_tensor(basis, dtype=torch.float32)
    result = generate_with_residual_transforms(
        model,
        tokenizer,
        adapter,
        encoding,
        {
            int(layer): (
                [encoding.query_position],
                removal_transform(center_tensor, basis_tensor, dose=dose),
            )
        },
        max_new_tokens=max_new_tokens,
    )
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "trace_subspace_ablation",
        "condition": "count_subspace_removal",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "split": encoding.split,
        "gold_count": encoding.count,
        "site_id": site_id,
        "layer": int(layer),
        "rank": int(basis_tensor.shape[1]),
        "dose": float(dose),
        **completion_metrics(result, gold_count=encoding.count),
    }


def _ordinary_segments(
    encoding: NativeTraceEncoding,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    spans = sorted(encoding.needle_spans, key=lambda span: span.start)
    if not spans:
        raise ValueError("Token corruption needs visible trace-item spans")
    lengths = [int(span.end) - int(span.start) for span in spans]
    forbidden = {
        position
        for span in spans
        for position in range(int(span.start), int(span.end))
    }
    used = set(forbidden)
    start = 1
    end = int(encoding.prompt_token_count)

    def allocate(length: int) -> tuple[int, int]:
        for candidate in range(start, end - length + 1):
            positions = set(range(candidate, candidate + length))
            if not positions.intersection(used):
                used.update(positions)
                return candidate, candidate + length
        raise RuntimeError("Could not allocate equal-length prompt control segment")

    sources = [allocate(length) for length in lengths]
    controls = [allocate(length) for length in lengths]
    control_sources = [allocate(length) for length in lengths]
    return sources, controls, control_sources


def corrupt_trace_tokens(
    encoding: NativeTraceEncoding,
) -> tuple[NativeTraceEncoding, NativeTraceEncoding, dict[str, Any]]:
    """Return trace corruption and equal-budget ordinary-token control."""

    sources, controls, control_sources = _ordinary_segments(encoding)
    clean = list(encoding.input_ids)
    trace_ids = clean.copy()
    control_ids = clean.copy()
    trace_changed = 0
    control_changed = 0
    for span, source in zip(encoding.needle_spans, sources):
        target = (int(span.start), int(span.end))
        replacement = clean[source[0] : source[1]]
        before = trace_ids[target[0] : target[1]]
        trace_ids[target[0] : target[1]] = replacement
        trace_changed += sum(left != right for left, right in zip(before, replacement))
    for target, source in zip(controls, control_sources):
        replacement = clean[source[0] : source[1]]
        before = control_ids[target[0] : target[1]]
        control_ids[target[0] : target[1]] = replacement
        control_changed += sum(left != right for left, right in zip(before, replacement))
    return (
        replace(encoding, input_ids=tuple(trace_ids)),
        replace(encoding, input_ids=tuple(control_ids)),
        {
            "token_budget": sum(
                int(span.end) - int(span.start) for span in encoding.needle_spans
            ),
            "trace_changed_tokens": trace_changed,
            "control_changed_tokens": control_changed,
            "trace_sources": sources,
            "control_targets": controls,
            "control_sources": control_sources,
        },
    )


def _matched_nontrace_positions(
    encoding: NativeTraceEncoding,
    *,
    token_budget: int,
) -> list[int]:
    forbidden = {
        position
        for span in encoding.needle_spans
        for position in range(int(span.start), int(span.end))
    }
    candidates = [
        position
        for position in range(1, int(encoding.query_position))
        if position not in forbidden
    ]
    if len(candidates) < token_budget:
        raise RuntimeError("Not enough non-trace keys for a matched context mask")
    # Evenly cover the prompt/trace depth instead of taking one local window.
    indices = np.linspace(0, len(candidates) - 1, token_budget, dtype=int)
    return [candidates[index] for index in indices]


def query_context_mask(
    encoding: NativeTraceEncoding,
    *,
    condition: str,
) -> torch.Tensor:
    query = int(encoding.query_position)
    mask = torch.zeros((1, query + 1), dtype=torch.long)
    if condition == "clean":
        mask[:] = 1
    elif condition == "trace_only":
        for span in encoding.needle_spans:
            mask[:, int(span.start) : min(int(span.end), query + 1)] = 1
        mask[:, query] = 1
    elif condition == "matched_nontrace_only":
        trace_positions = {
            position
            for span in encoding.needle_spans
            for position in range(
                int(span.start), min(int(span.end), query + 1)
            )
        }
        trace_positions.add(query)
        # The query itself is always available in both sparse conditions.
        # Match the number of *other* visible keys exactly.
        budget = len(trace_positions - {query})
        if budget < 1:
            raise RuntimeError("No earlier trace keys exist for a matched mask")
        positions = _matched_nontrace_positions(encoding, token_budget=budget)
        mask[:, positions] = 1
        mask[:, query] = 1
    else:
        raise ValueError(f"Unknown query context condition: {condition}")
    if int(mask.sum()) < 1:
        raise RuntimeError("Query context mask is empty")
    return mask


@torch.inference_mode()
def run_query_context_mask_trial(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    site_id: str,
    condition: str,
    layers: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Capture item-end/answer-query states under a key-context intervention."""

    encoding = build_native_trace_encoding(row, tokenizer, site_id=site_id)
    query = int(encoding.query_position)
    if query < 1:
        raise ValueError("A context-mask trial needs a non-initial query")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    prefix_output = model(
        input_ids=input_ids[:, :query],
        attention_mask=attention_mask[:, :query],
        use_cache=True,
        output_attentions=False,
        **_bounded_logits_kwargs(model),
    )
    past = getattr(prefix_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Context-mask prefix forward returned no KV cache")
    active_layers = (
        tuple(range(int(adapter.num_layers)))
        if layers is None
        else tuple(sorted({int(layer) for layer in layers}))
    )
    if not active_layers or any(
        layer < 0 or layer >= int(adapter.num_layers) for layer in active_layers
    ):
        raise ValueError("Invalid context-mask capture layers")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in active_layers:

        def hook(_module: Any, _args: tuple[Any, ...], output: Any, *, layer: int = layer):
            hidden = _tensor_from_output(output)
            captured[layer] = hidden[0, -1].detach().float().cpu()

        handles.append(adapter.layers[layer].register_forward_hook(hook))
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": query_context_mask(
            encoding, condition=condition
        ).to(input_ids.device),
        "past_key_values": past,
        "use_cache": False,
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
    shared = getattr(prefix_output, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = shared
    try:
        model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(active_layers) - set(captured))
    if missing:
        raise RuntimeError(f"Context-mask trial missed layers {missing}")
    states = np.stack([captured[layer].numpy() for layer in active_layers])
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "earlier_span_attention",
        "condition": condition,
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "split": encoding.split,
        "gold_count": encoding.count,
        "site_id": site_id,
        "occurrence": encoding.selected_site.get("occurrence"),
        "layers": list(active_layers),
        "states": states,
        "allowed_key_count": int(
            query_context_mask(encoding, condition=condition).sum().item()
        ),
    }


@torch.inference_mode()
def run_token_corruption_trial(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    config: V5Config,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    encoding = build_native_trace_encoding(
        row,
        tokenizer,
        site_id="answer_query",
        candidate_counts=config.candidate_counts,
    )
    corrupted, control, audit = corrupt_trace_tokens(encoding)
    rows: list[dict[str, Any]] = []
    for condition, active in (
        ("clean", encoding),
        ("trace_corrupt", corrupted),
        ("ordinary_control", control),
    ):
        result = generate_answer_completion(
            model, tokenizer, active, max_new_tokens=max_new_tokens
        )
        rows.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": "trace_token_corruption",
                "condition": condition,
                "request_id": encoding.request_id,
                "model_label": encoding.model_label,
                "seed": encoding.seed,
                "split": encoding.split,
                "gold_count": encoding.count,
                **audit,
                **completion_metrics(result, gold_count=encoding.count),
            }
        )
    return rows


def paired_seed_effects(
    trials: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    outcome: str,
) -> pd.DataFrame:
    required = {"model_label", "seed", "condition", outcome}
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Causal trials are missing {missing}")
    selected = trials.loc[trials["condition"].isin([treatment, control])].copy()
    if "request_id" not in selected:
        # Legacy aggregate trials have one registered unit per seed. Promote
        # that unit to an explicit request ID so the stricter request-first
        # pairing estimator remains backward compatible.
        selected["request_id"] = selected["seed"].map(
            lambda value: f"legacy-seed-{int(value)}"
        )
    selected[outcome] = pd.to_numeric(selected[outcome], errors="coerce")
    unit_columns = ["model_label", "seed", "request_id"]
    if "anchor_equivalence_id" in selected:
        unit_columns.append("anchor_equivalence_id")
    elif "site_id" in selected:
        unit_columns.append("site_id")
    # Repeated random banks are first averaged inside exactly the same
    # request-anchor-condition. Pairing then occurs before any request/seed
    # pooling, so missing or duplicated arms cannot alter the estimand.
    unit_conditions = (
        selected.groupby([*unit_columns, "condition"], as_index=False, dropna=False)
        .agg(outcome_mean=(outcome, "mean"), n_repeats=(outcome, "count"))
    )
    pivot = unit_conditions.pivot(
        index=unit_columns,
        columns="condition",
        values="outcome_mean",
    )
    if treatment not in pivot or control not in pivot:
        raise ValueError("Treatment/control pairing is incomplete")
    incomplete = pivot[[treatment, control]].isna().any(axis=1)
    if bool(incomplete.any()):
        examples = pivot.loc[incomplete].reset_index()[unit_columns].head(10)
        raise ValueError(
            "Treatment/control pairing is incomplete for request-anchor units: "
            f"{examples.to_dict(orient='records')}"
        )
    pivot = pivot.reset_index()
    pivot["effect"] = pivot[treatment] - pivot[control]
    request_effects = (
        pivot.groupby(
            ["model_label", "seed", "request_id"], as_index=False, dropna=False
        )
        .agg(
            request_effect=("effect", "mean"),
            n_paired_anchor_units=("effect", "size"),
        )
    )
    return (
        request_effects.groupby(["model_label", "seed"], as_index=False)
        .agg(
            mean_effect=("request_effect", "mean"),
            n_requests=("request_id", "nunique"),
            n_paired_anchor_units=("n_paired_anchor_units", "sum"),
        )
    )


def bootstrap_seed_mean_ci(
    seed_effects: Sequence[float],
    *,
    samples: int,
    seed: int = 5,
) -> dict[str, float]:
    values = np.asarray(seed_effects, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("Bootstrap needs finite seed effects")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(samples), len(values)), replace=True).mean(axis=1)
    return {
        "mean_effect": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_seeds": int(len(values)),
    }


def sign_flip_pvalue(seed_effects: Sequence[float]) -> float:
    values = np.asarray(seed_effects, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        raise ValueError("Sign-flip test needs finite seed effects")
    observed = abs(float(values.mean()))
    if n <= 20:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
        permuted = np.abs((signs * values).mean(axis=1))
    else:
        rng = np.random.default_rng(544)
        signs = rng.choice((-1, 1), size=(200_000, n))
        permuted = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (len(permuted) + 1))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values in original order."""

    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Holm adjustment requires finite one-dimensional p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def analyze_paired_causal_results(
    trials_csv: str | Path,
    output_csv: str | Path,
    *,
    treatment: str,
    control: str,
    outcome: str,
    config: V5Config,
    mechanism: str | None = None,
    bank_size: int | None = None,
    transition_phase: str | None = None,
    anchor_role: str | None = None,
) -> pd.DataFrame:
    source = Path(trials_csv)
    if source.is_dir():
        records = []
        for shard in sorted((source / "shards").glob("*.jsonl")):
            with shard.open("r", encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
        trials = pd.DataFrame(records)
    elif source.suffix.lower() in {".jsonl", ".ndjson"}:
        with source.open("r", encoding="utf-8") as handle:
            trials = pd.DataFrame(
                json.loads(line) for line in handle if line.strip()
            )
    else:
        trials = pd.read_csv(source)
    if "status" in trials:
        trials = trials.loc[trials["status"].astype(str).eq("ok")]
    filters = {
        "mechanism": mechanism,
        "transition_phase": transition_phase,
    }
    for column, value in filters.items():
        if value is None:
            if column in trials and trials[column].dropna().astype(str).nunique() > 1:
                raise ValueError(
                    f"Causal analysis must select one {column}; found multiple"
                )
            continue
        if column not in trials:
            raise ValueError(f"Causal trials have no filter column {column}")
        trials = trials.loc[trials[column].astype(str).eq(str(value))]
    if bank_size is not None:
        column = (
            "planned_bank_size"
            if "planned_bank_size" in trials
            else "bank_size"
        )
        if column not in trials:
            raise ValueError("Causal trials have no bank-size column")
        value_mask = pd.to_numeric(trials[column], errors="coerce").eq(
            float(bank_size)
        )
        if column == "bank_size" and "condition" in trials:
            # Legacy clean rows are shared K=0 baselines. Keep them when a
            # registered treatment K is selected for treatment-vs-clean tests.
            value_mask |= trials["condition"].astype(str).eq("clean")
        trials = trials.loc[value_mask]
    if anchor_role is not None:
        if "anchor_roles" not in trials:
            raise ValueError("Causal trials have no anchor_roles column")

        def has_role(value: Any) -> bool:
            if isinstance(value, (list, tuple, set)):
                return str(anchor_role) in {str(item) for item in value}
            return str(anchor_role) in str(value).split("|")

        trials = trials.loc[trials["anchor_roles"].map(has_role)]
    if trials.empty:
        raise ValueError("No causal trial rows remain after registered filters")
    seed_frame = paired_seed_effects(
        trials,
        treatment=treatment,
        control=control,
        outcome=outcome,
    )
    rows: list[dict[str, Any]] = []
    for model_label, frame in seed_frame.groupby("model_label"):
        statistics = bootstrap_seed_mean_ci(
            frame["mean_effect"], samples=config.bootstrap_samples
        )
        rows.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "model_label": model_label,
                "treatment": treatment,
                "control": control,
                "outcome": outcome,
                **{key: value for key, value in filters.items() if value is not None},
                **({"bank_size": bank_size} if bank_size is not None else {}),
                **({"anchor_role": anchor_role} if anchor_role is not None else {}),
                **statistics,
                "sign_flip_pvalue": sign_flip_pvalue(frame["mean_effect"]),
                "unit_of_inference": "seed_after_request_anchor_pairing",
                "n_requests": int(frame["n_requests"].sum()),
                "n_paired_anchor_units": int(
                    frame["n_paired_anchor_units"].sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_sign_flip_pvalue_across_models"] = holm_adjust(
        result["sign_flip_pvalue"].to_numpy(dtype=float)
    )
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    return result
