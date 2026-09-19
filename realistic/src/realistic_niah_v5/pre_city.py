from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah_v3.city_list_termination import (
    find_first_terminated_gold_city_list,
)
from realistic_niah_v4.modeling import (
    DecoderAdapter,
    capture_post_block_states,
    generate_with_residual_interventions,
    position_attention_outputs,
)

from .encoding import (
    NativeTraceEncoding,
    PromptRecordSpan,
    build_native_trace_encoding,
)
from .parsing import (
    align_trace_sites,
    gold_records,
    infer_model_family,
    output_token_ids,
    parse_trace_record,
    prompt_token_ids,
    raw_output_text,
    trace_char_sites,
)


PRE_CITY_SCHEMA_VERSION = "realistic_niah_v5_pre_city_token_v1"


@dataclass(frozen=True)
class PreCityQuery:
    occurrence: int
    city: str
    query_variant: str
    query_output_token_count: int
    city_first_token: int
    city_after_token: int
    item_start_char: int
    marker_end_char: int | None
    anchor_kind: str

    @property
    def token_distance_before_city(self) -> int:
        return int(self.city_first_token - self.query_output_token_count + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence": self.occurrence,
            "target_city": self.city,
            "query_variant": self.query_variant,
            "query_output_token_count": self.query_output_token_count,
            "city_first_token": self.city_first_token,
            "city_after_token": self.city_after_token,
            "item_start_char": self.item_start_char,
            "marker_end_char": self.marker_end_char,
            "anchor_kind": self.anchor_kind,
            "token_distance_before_city": self.token_distance_before_city,
        }


def _flat_ints(value: Any) -> tuple[int, ...]:
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected one unbatched tokenizer row")
        value = value[0]
    return tuple(int(item) for item in value)


def _flat_offsets(value: Any) -> tuple[tuple[int, int], ...]:
    if value and isinstance(value[0], (list, tuple)) and value[0] and isinstance(
        value[0][0], (list, tuple)
    ):
        if len(value) != 1:
            raise ValueError("Expected one unbatched offset row")
        value = value[0]
    return tuple((int(left), int(right)) for left, right in value)


def _boundary_at_or_left(
    offsets: Sequence[tuple[int, int]], char_position: int
) -> int | None:
    candidates = [
        index + 1
        for index, (left, right) in enumerate(offsets)
        if right > left and right <= int(char_position)
    ]
    return max(candidates) if candidates else None


def pre_city_token_queries(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    depths: Sequence[int] = (1, 2),
    include_anchor: bool = True,
) -> tuple[list[PreCityQuery], list[dict[str, Any]]]:
    requested_depths = tuple(sorted({int(value) for value in depths}))
    if not requested_depths or requested_depths[0] < 1:
        raise ValueError("pre-city depths must be positive")
    raw = raw_output_text(row)
    baseline = output_token_ids(row)
    encoded = tokenizer(
        raw,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    retokenized = _flat_ints(encoded["input_ids"])
    offsets = _flat_offsets(encoded["offset_mapping"])
    if len(retokenized) != len(offsets):
        raise RuntimeError("Tokenizer IDs and offset mapping have different lengths")
    if baseline[: len(retokenized)] != retokenized:
        return [], [
            {
                "status": "raw_retokenization_not_baseline_prefix",
                "baseline_token_count": len(baseline),
                "retokenized_token_count": len(retokenized),
            }
        ]

    parser = find_first_terminated_gold_city_list(
        raw,
        model_family=infer_model_family(row),
        gold_records=gold_records(row),
    )
    sites = trace_char_sites(raw, parser)
    by_id = {site.site_id: site for site in sites}
    queries: list[PreCityQuery] = []
    exclusions: list[dict[str, Any]] = []
    for occurrence, city in enumerate(parser.item_gold_cities, start=1):
        city_site = by_id.get(f"city_end:{occurrence}")
        item_site = by_id.get(f"item_end:{occurrence}")
        marker_site = by_id.get(f"marker_end:{occurrence}")
        if city_site is None or item_site is None:
            exclusions.append(
                {
                    "occurrence": occurrence,
                    "target_city": str(city),
                    "status": "missing_city_or_item_char_span",
                }
            )
            continue
        city_tokens = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > int(city_site.char_start)
            and left < int(city_site.char_end)
            and right > left
        ]
        if not city_tokens:
            exclusions.append(
                {
                    "occurrence": occurrence,
                    "target_city": str(city),
                    "status": "no_baseline_token_overlaps_city",
                }
            )
            continue
        city_first = min(city_tokens)
        city_after = max(city_tokens) + 1
        item_left = _boundary_at_or_left(offsets, int(item_site.char_start))
        minimum_query_count = max(1, int(item_left or 1))
        specs: list[tuple[str, int, str]] = []
        for depth in requested_depths:
            query_count = city_first - depth + 1
            if query_count >= minimum_query_count:
                specs.append((f"pre_city_d{depth}", query_count, "city_relative"))
            else:
                exclusions.append(
                    {
                        "occurrence": occurrence,
                        "target_city": str(city),
                        "query_variant": f"pre_city_d{depth}",
                        "status": "query_would_precede_item_start",
                    }
                )
        marker_end = (
            int(marker_site.char_end) if marker_site is not None else None
        )
        if include_anchor:
            anchor_char = (
                marker_end if marker_end is not None else int(item_site.char_start)
            )
            anchor_count = _boundary_at_or_left(offsets, anchor_char)
            anchor_kind = (
                "marker_end_left_token_boundary"
                if marker_end is not None
                else "item_start_left_token_boundary"
            )
            if anchor_count is not None and 1 <= anchor_count <= city_first:
                specs.append(("pre_city_anchor", anchor_count, anchor_kind))
            else:
                exclusions.append(
                    {
                        "occurrence": occurrence,
                        "target_city": str(city),
                        "query_variant": "pre_city_anchor",
                        "status": "no_left_baseline_token_anchor",
                        "anchor_kind": anchor_kind,
                    }
                )
        for variant, query_count, anchor_kind in specs:
            queries.append(
                PreCityQuery(
                    occurrence=occurrence,
                    city=str(city),
                    query_variant=variant,
                    query_output_token_count=int(query_count),
                    city_first_token=int(city_first),
                    city_after_token=int(city_after),
                    item_start_char=int(item_site.char_start),
                    marker_end_char=marker_end,
                    anchor_kind=anchor_kind,
                )
            )
    return queries, exclusions


def _prompt_spans(row: Mapping[str, Any]) -> tuple[PromptRecordSpan, ...]:
    result = []
    for value in row.get("prompt_record_spans", ()):
        result.append(
            PromptRecordSpan(
                slot_index=int(value["slot_index"]),
                city=str(value["city"]),
                score=(None if value.get("score") is None else int(value["score"])),
                start=int(value["start"]),
                end=int(value["end"]),
            )
        )
    return tuple(sorted(result, key=lambda span: span.slot_index))


def baseline_prefix_encoding(
    row: Mapping[str, Any],
    tokenizer: Any,
    query: PreCityQuery,
    *,
    prefix_output_token_count: int | None = None,
) -> NativeTraceEncoding:
    prompt = prompt_token_ids(row)
    output = output_token_ids(row)
    count = int(
        query.query_output_token_count
        if prefix_output_token_count is None
        else prefix_output_token_count
    )
    if not 1 <= count <= len(output):
        raise ValueError("Pre-city prefix token count is out of bounds")
    mask = row.get("attention_mask")
    prompt_mask = (
        tuple(int(value) for value in mask)
        if mask is not None
        else (1,) * len(prompt)
    )
    raw_prefix = tokenizer.decode(
        list(output[:count]),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    prompt_text = str(row.get("rendered_prompt", row.get("generation_prompt", "")))
    stimulus_id = str(row.get("stimulus_id", row.get("request_id", "unknown")))
    request_id = str(row.get("request_id", stimulus_id))
    full = prompt + output[:count]
    return NativeTraceEncoding(
        stimulus_id=stimulus_id,
        request_id=request_id,
        design_variant=str(row.get("design_variant", "v4.4")),
        seed=int(row.get("seed", -1)),
        split=str(row.get("split") or "unregistered"),
        count=len(gold_records(row)),
        model_label=str(row.get("model_label", row.get("model", "unknown"))),
        model_family=infer_model_family(row),
        answer_format="numeric",
        text=prompt_text + raw_prefix,
        generation_prompt=prompt_text + raw_prefix,
        input_ids=full,
        attention_mask=prompt_mask + (1,) * count,
        query_position=len(full) - 1,
        prompt_token_count=len(prompt),
        raw_prefix_text=raw_prefix,
        selected_site={
            "site_id": f"{query.query_variant}:{query.occurrence}",
            "site_kind": "pre_city_token",
            **query.to_dict(),
        },
        prompt_record_spans=_prompt_spans(row),
        trace_item_spans=(),
        slot_spans=(),
        needle_spans=(),
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )


def _span_mass(
    attention: Any,
    *,
    key_start: int,
    span: PromptRecordSpan,
) -> float:
    left = max(0, int(span.start) - int(key_start))
    right = min(int(attention.shape[-1]), int(span.end) - int(key_start))
    return float(attention[left:right].sum().item()) if right > left else 0.0


def capture_pre_city_attention_metrics(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    depths: Sequence[int] = (1, 2),
    include_anchor: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    queries, exclusions = pre_city_token_queries(
        row, tokenizer, depths=depths, include_anchor=include_anchor
    )
    parsed_family = infer_model_family(row)
    rows: list[dict[str, Any]] = []
    query_cache: dict[int, tuple[NativeTraceEncoding, Any, Any]] = {}
    for query in queries:
        cached = query_cache.get(query.query_output_token_count)
        if cached is None:
            encoding = baseline_prefix_encoding(row, tokenizer, query)
            attention_rows, key_starts, _ = position_attention_outputs(
                model, adapter, encoding, encoding.query_position
            )
            query_cache[query.query_output_token_count] = (
                encoding,
                attention_rows,
                key_starts,
            )
        else:
            encoding, attention_rows, key_starts = cached
        prompt_by_city = {
            span.city.casefold(): span for span in encoding.prompt_record_spans
        }
        target_span = prompt_by_city.get(query.city.casefold())
        if target_span is None:
            raise RuntimeError(f"No prompt record span for target city {query.city}")
        for layer, (attention, key_start) in enumerate(
            zip(attention_rows, key_starts)
        ):
            for head in range(attention.shape[0]):
                head_row = attention[head]
                record_masses = [
                    _span_mass(head_row, key_start=key_start, span=span)
                    for span in encoding.prompt_record_spans
                ]
                total = float(sum(record_masses))
                target = _span_mass(
                    head_row, key_start=key_start, span=target_span
                )
                relative = target / total if total > 0 else float("nan")
                rows.append(
                    {
                        "schema_version": PRE_CITY_SCHEMA_VERSION,
                        "request_id": encoding.request_id,
                        "stimulus_id": encoding.stimulus_id,
                        "model_label": encoding.model_label,
                        "model_family": parsed_family,
                        "seed": encoding.seed,
                        "split": encoding.split,
                        "gold_count": encoding.count,
                        "mechanism": "targeted_retrieval",
                        "site_kind": "pre_city_token",
                        **query.to_dict(),
                        "query_position": encoding.query_position,
                        "causal_target_token_count": int(
                            query.city_after_token - query.query_output_token_count
                        ),
                        "city_token_count": int(
                            query.city_after_token - query.city_first_token
                        ),
                        "layer": int(layer),
                        "head": int(head),
                        "key_start": int(key_start),
                        "target_slot_index": int(target_span.slot_index),
                        "target_needle_raw_mass": target,
                        "all_active_needles_raw_mass": total,
                        "target_needle_relative_mass": relative,
                        "target_needle_top1": bool(
                            total > 0
                            and record_masses
                            and int(np.argmax(record_masses))
                            == list(encoding.prompt_record_spans).index(target_span)
                        ),
                        "row_sum": float(head_row.sum().item()),
                    }
                )
    request_id = str(row.get("request_id", row.get("stimulus_id", "unknown")))
    enriched_exclusions = [
        {
            "schema_version": PRE_CITY_SCHEMA_VERSION,
            "request_id": request_id,
            "model_label": row.get("model_label"),
            "seed": row.get("seed"),
            "split": row.get("split"),
            **value,
        }
        for value in exclusions
    ]
    return pd.DataFrame(rows), enriched_exclusions


def write_pre_city_audit(
    attention: pd.DataFrame,
    exclusions: Sequence[Mapping[str, Any]],
    output: str | Path,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    variants = (
        attention.groupby("query_variant", dropna=False)
        .agg(
            attention_rows=("query_variant", "size"),
            requests=("request_id", "nunique"),
            occurrences=("occurrence", "count"),
        )
        .reset_index()
        .to_dict(orient="records")
        if not attention.empty
        else []
    )
    payload = {
        "schema_version": PRE_CITY_SCHEMA_VERSION,
        "mechanism": "targeted_retrieval",
        "selection_policy": "discovery_only_per_query_variant",
        "query_variants": variants,
        "attention_rows": int(len(attention)),
        "requests": int(attention["request_id"].nunique()) if not attention.empty else 0,
        "exclusion_rows": int(len(exclusions)),
        "exact_span_metrics": [
            "target_needle_raw_mass",
            "target_needle_relative_mass",
        ],
        "broad_aggregation_used": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def rank_pre_city_heads(
    attention: pd.DataFrame,
    *,
    split: str = "discovery",
) -> pd.DataFrame:
    """Freeze a separate targeted-retrieval head ranking for each query variant."""

    needed = {
        "model_label",
        "split",
        "query_variant",
        "layer",
        "head",
        "target_needle_raw_mass",
        "target_needle_relative_mass",
        "target_needle_top1",
    }
    missing = sorted(needed - set(attention.columns))
    if missing:
        raise ValueError(f"Pre-city attention table is missing {missing}")
    selected = attention.loc[
        attention["split"].astype(str).str.lower().eq(split.lower())
    ].copy()
    if selected.empty:
        raise ValueError(f"No pre-city attention rows matched split={split}")
    grouped = (
        selected.groupby(
            ["model_label", "query_variant", "layer", "head"],
            as_index=False,
        )
        .agg(
            discovery_target_raw_mass=("target_needle_raw_mass", "mean"),
            discovery_target_relative_mass=(
                "target_needle_relative_mass",
                "mean",
            ),
            discovery_target_top1=("target_needle_top1", "mean"),
            discovery_query_rows=("target_needle_raw_mass", "size"),
        )
        .sort_values(
            [
                "model_label",
                "query_variant",
                "discovery_target_raw_mass",
                "layer",
                "head",
            ],
            ascending=[True, True, False, True, True],
        )
    )
    grouped["discovery_rank"] = (
        grouped.groupby(["model_label", "query_variant"]).cumcount() + 1
    )
    grouped["mechanism"] = "targeted_retrieval"
    grouped["query_site_kind"] = "pre_city_token"
    grouped["selection_metric"] = (
        "query_weighted_mean_target_needle_raw_mass"
    )
    grouped["selection_split"] = split
    grouped["selection_cohort"] = "one_to_one"
    return grouped.reset_index(drop=True)


def _confirmation_head_metrics(attention: pd.DataFrame) -> pd.DataFrame:
    selected = attention.loc[
        attention["split"].astype(str).str.lower().eq("confirmation")
    ].copy()
    if selected.empty:
        raise ValueError("No pre-city attention rows matched split=confirmation")
    return (
        selected.groupby(
            ["model_label", "query_variant", "layer", "head"],
            as_index=False,
        )
        .agg(
            confirmation_target_raw_mass=("target_needle_raw_mass", "mean"),
            confirmation_target_relative_mass=(
                "target_needle_relative_mass",
                "mean",
            ),
            confirmation_target_top1=("target_needle_top1", "mean"),
            confirmation_query_rows=("target_needle_raw_mass", "size"),
        )
    )


def _pre_city_bank_mass(
    head_frame: pd.DataFrame,
    heads: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    lookup = {
        (int(row.layer), int(row.head)): row
        for row in head_frame.itertuples(index=False)
    }
    selected = [lookup[(int(layer), int(head))] for layer, head in heads]

    def finite_mean(field: str) -> tuple[float, int]:
        values = np.asarray([getattr(row, field) for row in selected], dtype=float)
        finite = values[np.isfinite(values)]
        return (
            float(finite.mean()) if len(finite) else float("nan"),
            int(len(finite)),
        )

    raw, raw_n = finite_mean("discovery_target_raw_mass")
    relative, relative_n = finite_mean("discovery_target_relative_mass")
    confirmation_raw, confirmation_raw_n = finite_mean(
        "confirmation_target_raw_mass"
    )
    confirmation_relative, confirmation_relative_n = finite_mean(
        "confirmation_target_relative_mass"
    )
    return {
        "target_needle_raw_mass": raw,
        "target_needle_relative_mass": relative,
        "relative_mass_defined_heads": relative_n,
        "raw_mass_defined_heads": raw_n,
        "confirmation_target_needle_raw_mass": confirmation_raw,
        "confirmation_target_needle_relative_mass": confirmation_relative,
        "confirmation_raw_mass_defined_heads": confirmation_raw_n,
        "confirmation_relative_mass_defined_heads": confirmation_relative_n,
        "attention_mass_split": "discovery",
        "confirmation_mass_split": "confirmation",
        "attention_mass_aggregation": (
            "mean_across_head_query_weighted_exact_span_scores"
        ),
    }


def build_pre_city_causal_plan(
    attention_csv: str | Path,
    output_dir: str | Path,
    *,
    config: Any,
) -> dict[str, Path]:
    """Build variant-specific discovery-frozen E4 head banks and controls."""

    from .causal import layer_matched_random_controls

    config.validate()
    attention = pd.read_csv(attention_csv)
    ranking = rank_pre_city_heads(attention)
    confirmation = _confirmation_head_metrics(attention)
    evaluation = ranking.merge(
        confirmation,
        on=["model_label", "query_variant", "layer", "head"],
        how="left",
        validate="one_to_one",
    )
    expected_variants = {"pre_city_d1", "pre_city_d2", "pre_city_anchor"}
    observed_variants = set(evaluation["query_variant"].astype(str))
    if observed_variants != expected_variants:
        raise ValueError(
            "Pre-city query variant contract mismatch: "
            f"expected={sorted(expected_variants)} observed={sorted(observed_variants)}"
        )

    rows: list[dict[str, Any]] = []
    skipped_controls: list[dict[str, Any]] = []
    for (model_label, variant), frame in evaluation.groupby(
        ["model_label", "query_variant"], sort=True
    ):
        ordered = [
            (int(row.layer), int(row.head))
            for row in frame.sort_values("discovery_rank").itertuples(index=False)
        ]
        for bank_size in config.causal_head_bank_sizes:
            if int(bank_size) > len(ordered):
                continue
            chosen = ordered[: int(bank_size)]
            common = {
                "model_label": str(model_label),
                "mechanism": "targeted_retrieval",
                "query_variant": str(variant),
                "query_site_kind": "pre_city_token",
                "experiment_id": "pre_city_targeted_retrieval_head_ablation",
                "bank_size": int(bank_size),
            }
            rows.append(
                {
                    **common,
                    "condition": "pre_city_targeted_retrieval_ranked",
                    "repeat": 0,
                    "heads": json.dumps(chosen),
                    **_pre_city_bank_mass(frame, chosen),
                }
            )
            try:
                controls = layer_matched_random_controls(
                    frame,
                    chosen,
                    repeats=config.causal_random_controls,
                    seed_text=(
                        f"v5:e4:{model_label}:{variant}:targeted_retrieval:"
                        f"K{bank_size}"
                    ),
                )
            except ValueError as error:
                skipped_controls.append(
                    {
                        **common,
                        "ranked_treatment_included": True,
                        "control_status": (
                            "not_constructible_disjoint_exact_layer_match"
                        ),
                        "reason": str(error),
                    }
                )
                continue
            for repeat, control in enumerate(controls, start=1):
                rows.append(
                    {
                        **common,
                        "condition": "layer_matched_random",
                        "repeat": int(repeat),
                        "heads": json.dumps(control),
                        **_pre_city_bank_mass(frame, control),
                    }
                )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "ranking": output / "discovery_head_ranking.csv",
        "confirmation": output / "confirmation_head_evaluation.csv",
        "plan": output / "causal_plan.csv",
        "audit": output / "causal_plan_audit.json",
    }
    ranking.to_csv(paths["ranking"], index=False)
    evaluation.to_csv(paths["confirmation"], index=False)
    pd.DataFrame(rows).to_csv(paths["plan"], index=False)
    paths["audit"].write_text(
        json.dumps(
            {
                "schema_version": "realistic_niah_v5_pre_city_causal_plan_v1",
                "mechanism": "targeted_retrieval",
                "query_site_kind": "pre_city_token",
                "query_variants": sorted(expected_variants),
                "variant_specific_discovery_selection": True,
                "confirmation_used_for_selection": False,
                "selection_split": "discovery",
                "evaluation_split": "confirmation",
                "selection_cohort": "one_to_one",
                "selection_metric": (
                    "query-weighted mean exact prompt-needle raw mass"
                ),
                "reported_exact_span_metrics": [
                    "target_needle_raw_mass",
                    "target_needle_relative_mass",
                    "confirmation_target_needle_raw_mass",
                    "confirmation_target_needle_relative_mass",
                ],
                "registered_bank_sizes": list(config.causal_head_bank_sizes),
                "random_control": (
                    "disjoint exact layer-matched random heads; unavailable "
                    "controls are audited without dropping ranked treatment"
                ),
                "skipped_controls": skipped_controls,
                "attention_source": str(Path(attention_csv).resolve()),
                "broad_aggregation_used": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def run_pre_city_head_ablation_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    query_variant: str,
    heads: Sequence[tuple[int, int]],
    condition: str,
    occurrences: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Ablate a frozen bank and greedily generate only the next city.

    The primary endpoint is an actual greedy token-sequence match for the city
    immediately following the registered query.  Teacher-forced likelihood is
    retained as a diagnostic, but no final-answer/count endpoint is computed.
    """

    from .causal import (
        CAUSAL_SCHEMA_VERSION,
        _continuation_metrics,
        _local_head_ablation_logits,
        generate_with_head_ablation_at_positions,
    )

    allowed = {"pre_city_d1", "pre_city_d2", "pre_city_anchor"}
    if query_variant not in allowed:
        raise ValueError(f"Unknown pre-city query variant: {query_variant}")
    queries, raw_exclusions = pre_city_token_queries(
        row,
        tokenizer,
        depths=(1, 2),
        include_anchor=True,
    )
    selected_queries = [
        query for query in queries if query.query_variant == query_variant
    ]
    occurrence_filter = (
        None if occurrences is None else {int(value) for value in occurrences}
    )
    if occurrence_filter is not None:
        selected_queries = [
            query
            for query in selected_queries
            if int(query.occurrence) in occurrence_filter
        ]
    exclusions = [
        value
        for value in raw_exclusions
        if value.get("query_variant") in {None, query_variant}
        and (
            occurrence_filter is None
            or value.get("occurrence") is None
            or int(value["occurrence"]) in occurrence_filter
        )
    ]
    request_id = row.get("request_id", row.get("stimulus_id"))
    common = {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "pre_city_next_needle_head_ablation_v2",
        "condition": condition,
        "request_id": request_id,
        "model_label": row.get("model_label"),
        "seed": row.get("seed"),
        "split": row.get("split"),
        "gold_count": len(gold_records(row)),
        "mechanism": "targeted_retrieval",
        "query_variant": query_variant,
        "query_site_kind": "pre_city_token",
        "transition_phase": "retrieve",
        "heads": [[int(layer), int(head)] for layer, head in heads],
        "bank_size": len(heads),
        "behavioral_endpoint": "actual_greedy_next_needle_token_sequence",
        "final_count_evaluated": False,
    }
    output = [{**common, **value} for value in exclusions]
    baseline_ids = output_token_ids(row)
    prompt_count = len(prompt_token_ids(row))
    for query in selected_queries:
        query_count = int(query.query_output_token_count)
        target_count = int(query.city_after_token)
        target_ids = baseline_ids[query_count:target_count]
        if not target_ids:
            output.append(
                {
                    **common,
                    **query.to_dict(),
                    "status": "empty_pre_city_target_continuation",
                }
            )
            continue
        target_encoding = baseline_prefix_encoding(
            row,
            tokenizer,
            query,
            prefix_output_token_count=target_count,
        )
        logits = _local_head_ablation_logits(
            model,
            adapter,
            target_encoding,
            heads,
            hook_position=prompt_count + query_count - 1,
            target_token_count=len(target_ids),
        )
        generation_encoding = baseline_prefix_encoding(
            row,
            tokenizer,
            query,
            prefix_output_token_count=query_count,
        )
        generation = generate_with_head_ablation_at_positions(
            model,
            tokenizer,
            adapter,
            generation_encoding,
            heads,
            hook_positions=[generation_encoding.query_position],
            max_new_tokens=len(target_ids),
        )
        generated_ids = tuple(
            int(value) for value in generation["generated_token_ids"]
        )
        target_tuple = tuple(int(value) for value in target_ids)
        matched_tokens = sum(
            int(index < len(generated_ids) and generated_ids[index] == token_id)
            for index, token_id in enumerate(target_tuple)
        )
        output.append(
            {
                **common,
                **query.to_dict(),
                "status": "ok",
                "target_token_count": len(target_ids),
                "target_text": tokenizer.decode(
                    list(target_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "generated_next_needle_token_ids": list(generated_ids),
                "generated_next_needle_text": tokenizer.decode(
                    list(generated_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "next_needle_exact": bool(generated_ids == target_tuple),
                "next_needle_first_token_exact": bool(
                    generated_ids and generated_ids[0] == target_tuple[0]
                ),
                "next_needle_token_accuracy": float(
                    matched_tokens / len(target_tuple)
                ),
                "next_needle_generated_token_count": len(generated_ids),
                "head_ablation_hook_audit": generation[
                    "head_ablation_hook_audit"
                ],
                **_continuation_metrics(logits, target_ids),
            }
        )
    return output


def _capture_query_state_from_encoding(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    layer: int,
) -> torch.Tensor:
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        [encoding.query_position],
        layers=[int(layer)],
    )
    return captured[int(layer)][0]


def orthogonal_norm_matched_patch_state(
    receiver_state: torch.Tensor,
    donor_state: torch.Tensor,
    *,
    seed_text: str,
) -> torch.Tensor:
    """Construct ``receiver + delta_control`` orthogonal to the donor delta."""

    receiver = receiver_state.detach().float().cpu().reshape(-1)
    donor = donor_state.detach().float().cpu().reshape(-1)
    if receiver.shape != donor.shape:
        raise ValueError("Receiver and donor marker states have different shapes")
    delta = donor - receiver
    norm = torch.linalg.vector_norm(delta)
    if not torch.isfinite(norm):
        raise ValueError("Donor-minus-receiver marker delta has invalid norm")
    # In local-attention early layers the added distant prompt record may be
    # causally unreachable, making the two semantic query states identical.
    # The exact norm-matched control then has zero norm: the identity patch.
    if float(norm) == 0.0:
        return receiver.clone()
    seed = int.from_bytes(
        hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "little"
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    random = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
    random = random - torch.dot(random, delta) / torch.dot(delta, delta) * delta
    random_norm = torch.linalg.vector_norm(random)
    if not torch.isfinite(random_norm) or float(random_norm) <= 0:
        raise RuntimeError("Could not construct an orthogonal marker control")
    return receiver + random * (norm / random_norm)


def _next_needle_generation_metrics(
    result: Mapping[str, Any],
    *,
    target_ids: Sequence[int],
    wrong_ids: Sequence[int],
    tokenizer: Any,
) -> dict[str, Any]:
    generated = tuple(int(value) for value in result["generated_token_ids"])
    target = tuple(int(value) for value in target_ids)
    wrong = tuple(int(value) for value in wrong_ids)

    def prefix_exact(reference: tuple[int, ...]) -> bool:
        return bool(
            reference
            and len(generated) >= len(reference)
            and generated[: len(reference)] == reference
        )

    matched = sum(
        int(index < len(generated) and generated[index] == token_id)
        for index, token_id in enumerate(target)
    )
    return {
        "generated_next_needle_token_ids": list(generated),
        "generated_next_needle_text": tokenizer.decode(
            list(generated),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "target_needle_exact": prefix_exact(target),
        "wrong_donor_needle_exact": prefix_exact(wrong),
        "target_needle_first_token_exact": bool(
            generated and target and generated[0] == target[0]
        ),
        "target_needle_token_accuracy": float(matched / len(target)),
        "generated_token_count": len(generated),
    }


def run_marker_needle_patch_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    full_row: Mapping[str, Any],
    counterfactual_row: Mapping[str, Any],
    *,
    query_variant: str,
    occurrence: int,
    layer: int,
) -> list[dict[str, Any]]:
    """Bidirectionally patch a marker state across adjacent count prompts.

    This mirrors the non-thinking counterfactual construction: the full N=k
    prompt and same-seed N=k-1 prompt must have equal token length and an
    identical prefix of k-1 oracle records.  Both receive the *same* N=k trace
    prefix up to the marker immediately before the added kth needle.  Only the
    prompt differs, so patching the query residual tests whether that state
    carries the missing needle into subsequent trace generation.
    """

    from .causal import CAUSAL_SCHEMA_VERSION

    full_parsed = parse_trace_record(full_row)
    counterfactual_parsed = parse_trace_record(counterfactual_row)
    for label, parsed in (
        ("full", full_parsed),
        ("counterfactual", counterfactual_parsed),
    ):
        if not bool(parsed["parser"].get("trace_one_to_one")) or not bool(
            parsed.get("exact_count")
        ):
            raise ValueError(f"{label} marker-patch row is not correct one-to-one")
    if str(full_row.get("model_label")) != str(counterfactual_row.get("model_label")):
        raise ValueError("Marker-patch prompts have different model labels")
    if int(full_row["seed"]) != int(counterfactual_row["seed"]):
        raise ValueError("Marker-patch prompts must share a seed")
    if str(full_row["split"]) != str(counterfactual_row["split"]):
        raise ValueError("Marker-patch prompts must share a split")
    full_records = gold_records(full_row)
    counterfactual_records = gold_records(counterfactual_row)
    if len(full_records) != len(counterfactual_records) + 1:
        raise ValueError("Marker-patch prompts must be adjacent counts N=k/N=k-1")
    full_prefix = [str(row["city"]).casefold() for row in full_records[:-1]]
    counterfactual_cities = [
        str(row["city"]).casefold() for row in counterfactual_records
    ]
    if full_prefix != counterfactual_cities:
        raise ValueError("Adjacent prompts do not share the first k-1 needles")
    if int(occurrence) != len(full_records):
        raise ValueError("Marker patch occurrence must be the added kth needle")
    full_prompt_ids = prompt_token_ids(full_row)
    counterfactual_prompt_ids = prompt_token_ids(counterfactual_row)
    if len(full_prompt_ids) != len(counterfactual_prompt_ids):
        raise ValueError("Adjacent prompt pair is not token-length matched")
    changed_prompt_tokens = [
        index
        for index, (left, right) in enumerate(
            zip(full_prompt_ids, counterfactual_prompt_ids)
        )
        if int(left) != int(right)
    ]
    if not changed_prompt_tokens:
        raise ValueError("Adjacent prompts have no counterfactual token change")
    target_city = str(full_records[-1]["city"])
    target_spans = [
        span
        for span in _prompt_spans(full_row)
        if span.city.casefold() == target_city.casefold()
    ]
    if len(target_spans) != 1 or not all(
        target_spans[0].start <= index < target_spans[0].end
        for index in changed_prompt_tokens
    ):
        raise ValueError(
            "Adjacent prompt changes are not confined to the added kth record"
        )

    full_queries, query_exclusions = pre_city_token_queries(
        full_row, tokenizer, depths=(1, 2), include_anchor=True
    )
    matches = [
        query
        for query in full_queries
        if query.query_variant == query_variant
        and int(query.occurrence) == int(occurrence)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Marker patch query is not uniquely constructible: "
            f"matches={len(matches)} exclusions={query_exclusions}"
        )
    query = matches[0]
    full_output_ids = output_token_ids(full_row)
    target_ids = tuple(
        int(value)
        for value in full_output_ids[
            query.query_output_token_count : query.city_after_token
        ]
    )
    if not target_ids:
        raise ValueError("Marker patch has an empty city continuation")

    full_encoding = baseline_prefix_encoding(
        full_row,
        tokenizer,
        query,
        prefix_output_token_count=query.query_output_token_count,
    )
    counterfactual_context = copy.deepcopy(dict(counterfactual_row))
    counterfactual_context["output_token_ids"] = list(full_output_ids)
    counterfactual_context["raw_output_text"] = full_row.get("raw_output_text")
    counterfactual_context["request_id"] = (
        f"{counterfactual_row.get('request_id')}__teacher_forced_from__"
        f"{full_row.get('request_id')}"
    )
    counterfactual_encoding = baseline_prefix_encoding(
        counterfactual_context,
        tokenizer,
        query,
        prefix_output_token_count=query.query_output_token_count,
    )
    if full_encoding.sequence_length != counterfactual_encoding.sequence_length:
        raise ValueError("Counterfactual shifted the semantic marker query")
    if full_encoding.query_position != counterfactual_encoding.query_position:
        raise ValueError("Counterfactual marker query positions are not aligned")
    full_state = _capture_query_state_from_encoding(
        model, adapter, full_encoding, layer=layer
    )
    counterfactual_state = _capture_query_state_from_encoding(
        model, adapter, counterfactual_encoding, layer=layer
    )
    donor_receiver_delta_norm = float(
        torch.linalg.vector_norm(
            full_state.detach().float().cpu()
            - counterfactual_state.detach().float().cpu()
        )
    )
    if not np.isfinite(donor_receiver_delta_norm):
        raise ValueError("Marker donor-receiver state delta has invalid norm")
    orthogonal_control_audit = (
        "PASS_DEGENERATE_ZERO_NORM_IDENTITY_CONTROL"
        if donor_receiver_delta_norm == 0.0
        else "PASS_NORM_MATCHED_ORTHOGONAL_CONTROL"
    )
    counterfactual_orthogonal = orthogonal_norm_matched_patch_state(
        counterfactual_state,
        full_state,
        seed_text=(
            f"v5-marker-adjacent:cf:{full_encoding.request_id}:"
            f"{counterfactual_encoding.request_id}:{query_variant}:L{layer}"
        ),
    )
    full_orthogonal = orthogonal_norm_matched_patch_state(
        full_state,
        counterfactual_state,
        seed_text=(
            f"v5-marker-adjacent:full:{full_encoding.request_id}:"
            f"{counterfactual_encoding.request_id}:{query_variant}:L{layer}"
        ),
    )
    conditions = (
        ("full_self_patch", full_encoding, full_state),
        (
            "counterfactual_self_patch",
            counterfactual_encoding,
            counterfactual_state,
        ),
        (
            "counterfactual_from_full_patch",
            counterfactual_encoding,
            full_state,
        ),
        ("full_from_counterfactual_patch", full_encoding, counterfactual_state),
        (
            "counterfactual_from_orthogonal_patch",
            counterfactual_encoding,
            counterfactual_orthogonal,
        ),
        ("full_from_orthogonal_patch", full_encoding, full_orthogonal),
    )
    output: list[dict[str, Any]] = []
    for condition, encoding, state in conditions:
        result = generate_with_residual_interventions(
            model,
            tokenizer,
            adapter,
            encoding,
            {int(layer): ([encoding.query_position], state)},
            max_new_tokens=len(target_ids),
        )
        output.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": "marker_adjacent_count_interchange_patch_v1",
                "condition": condition,
                "request_id": full_encoding.request_id,
                "counterfactual_request_id": str(
                    counterfactual_row.get("request_id")
                ),
                "model_label": full_encoding.model_label,
                "seed": full_encoding.seed,
                "split": full_encoding.split,
                "full_count": full_encoding.count,
                "counterfactual_count": counterfactual_encoding.count,
                "query_variant": query_variant,
                "occurrence": int(occurrence),
                "query_output_token_count": int(query.query_output_token_count),
                "query_position": int(encoding.query_position),
                "query_alias_key": (
                    f"occ{int(occurrence)}__q{int(query.query_output_token_count)}"
                ),
                "layer": int(layer),
                "target_city": query.city,
                "target_needle_token_ids": list(target_ids),
                "counterfactual_construction": (
                    "same_seed_adjacent_count_nonthinking_matched_prompt"
                ),
                "prompt_token_count": len(full_prompt_ids),
                "prompt_changed_token_count": len(changed_prompt_tokens),
                "prompt_changed_token_indices": changed_prompt_tokens,
                "prompt_length_match_audit": "PASS_EQUAL_TOKEN_LENGTH",
                "prompt_counterfactual_locality_audit": (
                    "PASS_CHANGED_TOKENS_INSIDE_ADDED_RECORD_ONLY"
                ),
                "trace_prefix_identity_audit": "PASS_SAME_TEACHER_FORCED_PREFIX",
                "semantic_query_alignment_audit": "PASS_SAME_QUERY_POSITION",
                "donor_receiver_delta_norm": donor_receiver_delta_norm,
                "orthogonal_control_audit": orthogonal_control_audit,
                "behavioral_endpoint": "actual_greedy_next_needle_token_sequence",
                **_next_needle_generation_metrics(
                    result,
                    target_ids=target_ids,
                    wrong_ids=(),
                    tokenizer=tokenizer,
                ),
            }
        )
    return output


def run_all_site_pre_city_damage_trial(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    query_variant: str,
    heads: Sequence[tuple[int, int]],
    condition: str,
) -> dict[str, Any]:
    """Jointly damage every registered pre-city query in a full trace.

    The baseline trace tokens are teacher forced.  This deliberately isolates
    the downstream routing effect from free-rollout trajectory drift: the same
    frozen head bank is zeroed at every baseline query for one variant.  Marker
    continuations are retained as teacher-forced mechanism diagnostics, while
    the final count is evaluated by an actual greedy continuation, matching the
    V4.4 non-thinking behavioral endpoint.
    """

    from .causal import (
        CAUSAL_SCHEMA_VERSION,
        _continuation_metrics,
        completion_metrics,
        generate_with_head_ablation_at_positions,
    )

    allowed = {"pre_city_d1", "pre_city_d2", "pre_city_anchor"}
    if query_variant not in allowed:
        raise ValueError(f"Unknown pre-city query variant: {query_variant}")
    queries, exclusions = pre_city_token_queries(
        row,
        tokenizer,
        depths=(1, 2),
        include_anchor=True,
    )
    selected = [
        query for query in queries if query.query_variant == query_variant
    ]
    relevant_exclusions = [
        value
        for value in exclusions
        if value.get("query_variant") in {None, query_variant}
    ]
    common = {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "all_site_pre_city_targeted_retrieval_damage_v1",
        "condition": condition,
        "request_id": row.get("request_id", row.get("stimulus_id")),
        "model_label": row.get("model_label"),
        "seed": row.get("seed"),
        "split": row.get("split"),
        "gold_count": len(gold_records(row)),
        "mechanism": "targeted_retrieval",
        "query_variant": query_variant,
        "query_site_kind": "pre_city_token",
        "ablation_scope": "all_registered_variant_queries_jointly",
        "evaluation_mode": (
            "fixed_baseline_trace_plus_greedy_numeric_answer_generation"
        ),
        "heads": [[int(layer), int(head)] for layer, head in heads],
        "bank_size": len(heads),
        "broad_aggregation_used": False,
    }
    if relevant_exclusions:
        return {
            **common,
            "status": "registered_query_exclusion",
            "exclusions": relevant_exclusions,
            "scheduled_query_count": len(selected),
        }
    if not selected:
        return {
            **common,
            "status": "no_registered_variant_queries",
            "scheduled_query_count": 0,
        }

    raw = raw_output_text(row)
    family = infer_model_family(row)
    parser = find_first_terminated_gold_city_list(
        raw,
        model_family=family,
        gold_records=gold_records(row),
    )
    token_sites = align_trace_sites(
        tokenizer,
        raw_text=raw,
        baseline_output_token_ids=output_token_ids(row),
        sites=trace_char_sites(raw, parser),
    )
    by_id = {site.char_site.site_id: site for site in token_sites}
    answer_encoding = build_native_trace_encoding(
        row,
        tokenizer,
        site_id="answer_query_v3",
        candidate_counts=tuple(range(1, 11)),
    )
    output_ids = output_token_ids(row)
    prompt_ids = prompt_token_ids(row)
    prompt_count = len(prompt_ids)
    hook_positions = [
        prompt_count + int(query.query_output_token_count) - 1
        for query in selected
    ]

    marker_specs: list[dict[str, Any]] = []
    score_positions: list[int] = []
    target_ids: list[int] = []
    for query in selected:
        endpoint = by_id.get(f"item_end:{int(query.occurrence)}")
        if (
            endpoint is None
            or not endpoint.alignment_eligible
            or endpoint.prefix_token_count is None
        ):
            return {
                **common,
                "status": "missing_aligned_item_end_endpoint",
                "occurrence": int(query.occurrence),
                "scheduled_query_count": len(selected),
            }
        start = int(query.query_output_token_count)
        end = int(endpoint.prefix_token_count)
        if not start < end <= len(output_ids):
            return {
                **common,
                "status": "invalid_marker_continuation_bounds",
                "occurrence": int(query.occurrence),
                "marker_start": start,
                "marker_end": end,
                "scheduled_query_count": len(selected),
            }
        left = len(score_positions)
        for output_index in range(start, end):
            score_positions.append(prompt_count + output_index - 1)
            target_ids.append(int(output_ids[output_index]))
        marker_specs.append(
            {
                "occurrence": int(query.occurrence),
                "target_city": str(query.city),
                "query_output_token_count": start,
                "item_end_output_token_count": end,
                "score_left": left,
                "score_right": len(score_positions),
            }
        )

    answer_ids = dict(
        answer_encoding.count_candidate_answer_token_ids
    )[answer_encoding.count]
    answer_score_index = len(score_positions)
    score_positions.append(int(answer_encoding.query_position))
    target_ids.append(int(answer_ids[0]))
    generation = generate_with_head_ablation_at_positions(
        model,
        tokenizer,
        adapter,
        answer_encoding,
        heads,
        hook_positions=hook_positions,
        score_positions=score_positions,
        max_new_tokens=16,
    )
    selected_logits = generation.pop("prefill_selected_logits")
    if len(selected_logits) != len(target_ids):
        raise RuntimeError("Scheduled damage endpoint row count mismatch")

    marker_rows: list[dict[str, Any]] = []
    marker_logps: list[float] = []
    marker_first_exact: list[bool] = []
    for specification in marker_specs:
        left = int(specification.pop("score_left"))
        right = int(specification.pop("score_right"))
        active_logits = selected_logits[left:right]
        active_targets = target_ids[left:right]
        metrics = _continuation_metrics(active_logits, active_targets)
        marker_logps.append(float(metrics["target_sequence_log_probability"]))
        marker_first_exact.append(bool(metrics["target_first_token_exact"]))
        marker_rows.append(
            {
                **specification,
                "target_token_count": len(active_targets),
                "target_text": tokenizer.decode(
                    active_targets,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                **metrics,
            }
        )

    answer_logits = selected_logits[answer_score_index]
    answer_probabilities = torch.softmax(answer_logits, dim=-1)
    gold_first_id = int(answer_ids[0])
    gold_rank = int(
        1 + torch.count_nonzero(answer_logits > answer_logits[gold_first_id]).item()
    )
    candidate_first_ids = {
        int(count): int(ids[0])
        for count, ids in answer_encoding.count_candidate_answer_token_ids
    }
    candidate_logits = {
        count: float(answer_logits[token_id])
        for count, token_id in candidate_first_ids.items()
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
        if int(token_id) == gold_first_id
    )
    unconstrained_id = int(answer_logits.argmax().item())
    non_target_logits = answer_logits.clone()
    non_target_logits[gold_first_id] = -torch.inf
    top_non_target_logit = float(non_target_logits.max())
    gold_first_logit = float(answer_logits[gold_first_id])
    behavior = completion_metrics(
        generation,
        gold_count=answer_encoding.count,
    )
    return {
        **common,
        "status": "ok",
        "scheduled_query_count": len(selected),
        "scheduled_query_positions": hook_positions,
        "marker_endpoint_definition": (
            "tokens after each registered pre-city query through item_end:k"
        ),
        "marker_occurrences": marker_rows,
        "marker_sequence_log_probability_sum": float(sum(marker_logps)),
        "marker_mean_sequence_log_probability": float(np.mean(marker_logps)),
        "marker_first_token_exact_rate": float(np.mean(marker_first_exact)),
        "behavioral_endpoint": "strict_greedy_complete_numeric_generation",
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
        "prefill_reuse_audit": generation["prefill_reuse_audit"],
        "head_ablation_prefill_position_count": generation[
            "head_ablation_prefill_position_count"
        ],
        "answer_query_site_id": "answer_query_v3",
        "answer_gold_first_token_id": gold_first_id,
        "answer_gold_first_token_text": tokenizer.decode(
            [gold_first_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "answer_gold_first_token_probability": float(
            answer_probabilities[gold_first_id]
        ),
        "answer_gold_first_token_rank": gold_rank,
        "answer_gold_first_token_logit": gold_first_logit,
        "answer_top_non_gold_first_token_logit": top_non_target_logit,
        "answer_gold_first_token_logit_margin": (
            gold_first_logit - top_non_target_logit
        ),
        "answer_gold_first_token_exact": bool(unconstrained_id == gold_first_id),
        "answer_candidate_argmax_counts": candidate_argmax_counts,
        "answer_candidate_argmax_contains_gold": bool(
            int(answer_encoding.count) in candidate_argmax_counts
        ),
        "answer_gold_first_token_alias_counts": target_token_alias_counts,
        "answer_gold_first_token_identifies_unique_count": bool(
            len(target_token_alias_counts) == 1
        ),
        "answer_candidate_first_token_logits": candidate_logits,
        "answer_unconstrained_first_token_id": unconstrained_id,
        "answer_unconstrained_first_token_text": tokenizer.decode(
            [unconstrained_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }
