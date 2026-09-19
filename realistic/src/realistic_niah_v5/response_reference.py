from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from realistic_niah_v3.city_list_termination import (
    find_first_terminated_gold_city_list,
)

from .parsing import (
    gold_records,
    infer_model_family,
    output_token_ids,
    prompt_token_ids,
    raw_output_text,
    trace_char_sites,
)
from .pre_city import (
    PreCityQuery,
    _flat_ints,
    _flat_offsets,
    _span_mass,
    baseline_prefix_encoding,
    pre_city_token_queries,
)


RESPONSE_REFERENCE_SCHEMA_VERSION = (
    "realistic_niah_v5_response_reference_parser_v1"
)
REFERENCE_TYPES = ("bare_or_list", "record_template", "semantic_cue")
POSITION_VARIANTS = ("pre_city_d1", "pre_city_d2", "pre_city_anchor")
TARGETED_REFERENCE_VARIANT = "pre_reference_d1"


@dataclass(frozen=True)
class ResponseReferenceSite:
    occurrence: int
    city: str
    response_type: str
    item_start_char: int
    item_end_char: int
    city_start_char: int
    city_end_char: int
    citation_start_char: int
    citation_start_kind: str
    raw_prefix: str
    parser_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence": int(self.occurrence),
            "target_city": self.city,
            "response_type": self.response_type,
            "item_start_char": int(self.item_start_char),
            "item_end_char": int(self.item_end_char),
            "city_start_char": int(self.city_start_char),
            "city_end_char": int(self.city_end_char),
            "citation_start_char": int(self.citation_start_char),
            "citation_start_kind": self.citation_start_kind,
            "reference_prefix": self.raw_prefix,
            "response_reference_parser": self.parser_name,
            "response_reference_parser_schema": RESPONSE_REFERENCE_SCHEMA_VERSION,
        }


@dataclass(frozen=True)
class ResponseReferenceQuery:
    base: PreCityQuery
    site: ResponseReferenceSite
    target_after_token: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = self.base.to_dict()
        result.update(self.site.to_dict())
        result["position_variant"] = self.base.query_variant
        result["query_variant"] = f"response_reference_{self.base.query_variant}"
        result["source_query_variant"] = self.base.query_variant
        result["target_after_token"] = int(
            self.base.city_after_token
            if self.target_after_token is None
            else self.target_after_token
        )
        if self.base.query_variant == TARGETED_REFERENCE_VARIANT:
            result["citation_first_token"] = int(
                self.base.query_output_token_count
            )
            result["token_distance_before_citation"] = 1
        result["query_definition"] = (
            "registered real baseline generation token before the exact "
            "parser-registered citation span"
        )
        return result


_NUMBERED_ONLY = re.compile(r"\d+\s*[.)]?\s*\Z")
_BULLET_ONLY = re.compile(r"[-*\u2022]+\s*\Z")
_AUDIT_TEMPLATE_END = re.compile(
    r"city\s+score\s+audit\s*[,;:]?\s*[\"'\u201c\u2018]?\s*\Z",
    re.IGNORECASE,
)
_EXACT_NEEDLE_PREFIX = re.compile(
    r"\bIn\s+the\s+2024\s+city\s+score\s+audit\b",
    re.IGNORECASE,
)


def _citation_start(
    raw: str,
    *,
    item_start: int,
    city_start: int,
) -> tuple[int, str]:
    """Locate the semantic start of one cited needle in the trace.

    A copied prompt record begins at its canonical record prefix.  List
    numbering, bullets, ``Found:``, ``Excerpt:``, and quotation punctuation are
    response scaffolding and are not part of the cited needle.  When the model
    does not copy the record template, the exact city span is the only stable
    citation identity and therefore defines the citation start.
    """

    prefix = raw[int(item_start) : int(city_start)]
    matches = list(_EXACT_NEEDLE_PREFIX.finditer(prefix))
    if matches:
        return int(item_start) + int(matches[-1].start()), "exact_record_prefix"
    return int(city_start), "exact_city_fallback"


def _parse_qwen_reference_type(prefix: str) -> str:
    """Classify Qwen's dominant bare/list, copied-record, and cue formats."""

    stripped = prefix.strip()
    if not stripped or _NUMBERED_ONLY.fullmatch(stripped) or _BULLET_ONLY.fullmatch(
        stripped
    ):
        return "bare_or_list"
    if _AUDIT_TEMPLATE_END.search(stripped):
        return "record_template"
    return "semantic_cue"


def _parse_gemma_reference_type(prefix: str) -> str:
    """Classify Gemma's copied-record, structured-cue, and rare bare formats."""

    stripped = prefix.strip()
    if _AUDIT_TEMPLATE_END.search(stripped):
        return "record_template"
    if not stripped or _NUMBERED_ONLY.fullmatch(stripped) or _BULLET_ONLY.fullmatch(
        stripped
    ):
        return "bare_or_list"
    return "semantic_cue"


def parse_response_reference_sites(
    row: Mapping[str, Any],
) -> tuple[list[ResponseReferenceSite], list[dict[str, Any]]]:
    """Register every exact city citation and its model-specific response type."""

    raw = raw_output_text(row)
    family = infer_model_family(row)
    parser = find_first_terminated_gold_city_list(
        raw,
        model_family=family,
        gold_records=gold_records(row),
    )
    if family == "qwen3":
        classifier = _parse_qwen_reference_type
        parser_name = "qwen3_response_reference_parser_v1"
    elif family in {"gemma", "gemma4"}:
        classifier = _parse_gemma_reference_type
        parser_name = "gemma4_response_reference_parser_v1"
    else:
        raise ValueError(f"Unsupported response-reference model family: {family}")

    by_id = {site.site_id: site for site in trace_char_sites(raw, parser)}
    sites: list[ResponseReferenceSite] = []
    exclusions: list[dict[str, Any]] = []
    for occurrence, city in enumerate(parser.item_gold_cities, start=1):
        city_site = by_id.get(f"city_end:{occurrence}")
        item_site = by_id.get(f"item_end:{occurrence}")
        if city_site is None or item_site is None:
            exclusions.append(
                {
                    "occurrence": int(occurrence),
                    "target_city": str(city),
                    "status": "missing_response_reference_char_span",
                    "response_reference_parser": parser_name,
                }
            )
            continue
        prefix = raw[int(item_site.char_start) : int(city_site.char_start)]
        response_type = classifier(prefix)
        if response_type not in REFERENCE_TYPES:
            raise AssertionError(f"Unregistered response type: {response_type}")
        citation_start, citation_start_kind = _citation_start(
            raw,
            item_start=int(item_site.char_start),
            city_start=int(city_site.char_start),
        )
        sites.append(
            ResponseReferenceSite(
                occurrence=int(occurrence),
                city=str(city),
                response_type=response_type,
                item_start_char=int(item_site.char_start),
                item_end_char=int(item_site.char_end),
                city_start_char=int(city_site.char_start),
                city_end_char=int(city_site.char_end),
                citation_start_char=citation_start,
                citation_start_kind=citation_start_kind,
                raw_prefix=prefix,
                parser_name=parser_name,
            )
        )
    return sites, exclusions


def targeted_retrieval_queries(
    row: Mapping[str, Any], tokenizer: Any
) -> tuple[list[ResponseReferenceQuery], list[dict[str, Any]]]:
    """Register a separate pre-citation query for every k-to-k retrieval.

    The query is the real baseline token immediately before the first token
    overlapping the parser-registered citation.  The target prompt needle is
    still resolved by exact city identity; the occurrence index records the
    response-side ``k`` and the prompt slot records its matched needle.
    """

    raw = raw_output_text(row)
    baseline = output_token_ids(row)
    encoded = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
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

    sites, exclusions = parse_response_reference_sites(row)
    queries: list[ResponseReferenceQuery] = []
    for site in sites:
        citation_tokens = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > int(site.citation_start_char)
            and left < int(site.city_end_char)
            and right > left
        ]
        city_tokens = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > int(site.city_start_char)
            and left < int(site.city_end_char)
            and right > left
        ]
        target_tokens = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > int(site.citation_start_char)
            and left < int(site.item_end_char)
            and right > left
        ]
        if not citation_tokens or not city_tokens or not target_tokens:
            exclusions.append(
                {
                    "occurrence": int(site.occurrence),
                    "target_city": site.city,
                    "query_variant": TARGETED_REFERENCE_VARIANT,
                    "status": "no_baseline_token_overlaps_registered_citation",
                    "citation_start_kind": site.citation_start_kind,
                }
            )
            continue
        citation_first = min(citation_tokens)
        city_first = min(city_tokens)
        city_after = max(city_tokens) + 1
        if citation_first < 1:
            exclusions.append(
                {
                    "occurrence": int(site.occurrence),
                    "target_city": site.city,
                    "query_variant": TARGETED_REFERENCE_VARIANT,
                    "status": "registered_citation_has_no_previous_baseline_token",
                    "citation_start_kind": site.citation_start_kind,
                }
            )
            continue
        base = PreCityQuery(
            occurrence=int(site.occurrence),
            city=site.city,
            query_variant=TARGETED_REFERENCE_VARIANT,
            query_output_token_count=int(citation_first),
            city_first_token=int(city_first),
            city_after_token=int(city_after),
            item_start_char=int(site.item_start_char),
            marker_end_char=None,
            anchor_kind=f"parser_{site.citation_start_kind}_left_token",
        )
        queries.append(
            ResponseReferenceQuery(
                base=base,
                site=site,
                target_after_token=max(target_tokens) + 1,
            )
        )
    return queries, exclusions


def capture_targeted_retrieval_attention_metrics(
    model: Any,
    adapter: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Capture exact k-to-k prompt-needle mass at every pre-citation query."""

    from realistic_niah_v4.modeling import position_attention_outputs

    queries, exclusions = targeted_retrieval_queries(row, tokenizer)
    rows: list[dict[str, Any]] = []
    query_cache: dict[int, tuple[Any, Any, Any]] = {}
    for query in queries:
        base = query.base
        cached = query_cache.get(int(base.query_output_token_count))
        if cached is None:
            encoding = baseline_prefix_encoding(row, tokenizer, base)
            attention_rows, key_starts, _ = position_attention_outputs(
                model, adapter, encoding, encoding.query_position
            )
            cached = (encoding, attention_rows, key_starts)
            query_cache[int(base.query_output_token_count)] = cached
        encoding, attention_rows, key_starts = cached
        matching_spans = [
            span
            for span in encoding.prompt_record_spans
            if span.city.casefold() == base.city.casefold()
        ]
        if len(matching_spans) != 1:
            exclusions.append(
                {
                    "occurrence": int(base.occurrence),
                    "target_city": base.city,
                    "query_variant": TARGETED_REFERENCE_VARIANT,
                    "status": "prompt_exact_needle_identity_not_unique",
                    "prompt_span_matches": len(matching_spans),
                }
            )
            continue
        target_span = matching_spans[0]
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
                rows.append(
                    {
                        "schema_version": RESPONSE_REFERENCE_SCHEMA_VERSION,
                        "experiment_id": (
                            "trace_pre_reference_k_to_k_targeted_retrieval_v1"
                        ),
                        "request_id": encoding.request_id,
                        "stimulus_id": encoding.stimulus_id,
                        "model_label": encoding.model_label,
                        "model_family": encoding.model_family,
                        "seed": encoding.seed,
                        "split": encoding.split,
                        "gold_count": encoding.count,
                        "mechanism": "targeted_retrieval",
                        "site_kind": "pre_reference_token",
                        **query.to_dict(),
                        "position_variant": TARGETED_REFERENCE_VARIANT,
                        "query_variant": TARGETED_REFERENCE_VARIANT,
                        "query_position": int(encoding.query_position),
                        "causal_target_token_count": int(
                            int(query.target_after_token)
                            - base.query_output_token_count
                        ),
                        "city_token_count": int(
                            base.city_after_token - base.city_first_token
                        ),
                        "layer": int(layer),
                        "head": int(head),
                        "key_start": int(key_start),
                        "target_slot_index": int(target_span.slot_index),
                        "target_needle_raw_mass": target,
                        "all_active_needles_raw_mass": total,
                        "target_needle_relative_mass": (
                            target / total if total > 0 else float("nan")
                        ),
                        "target_needle_top1": bool(
                            total > 0
                            and record_masses
                            and int(np.argmax(record_masses))
                            == list(encoding.prompt_record_spans).index(target_span)
                        ),
                        "row_sum": float(head_row.sum().item()),
                        "k_to_k_registry_audit": (
                            "PASS_RESPONSE_OCCURRENCE_TO_EXACT_CITY_PROMPT_SPAN"
                        ),
                    }
                )
    request_id = str(row.get("request_id", row.get("stimulus_id", "unknown")))
    enriched_exclusions = [
        {
            "schema_version": RESPONSE_REFERENCE_SCHEMA_VERSION,
            "request_id": request_id,
            "model_label": row.get("model_label", row.get("model")),
            "seed": row.get("seed"),
            "split": row.get("split"),
            **value,
        }
        for value in exclusions
    ]
    return pd.DataFrame(rows), enriched_exclusions


def response_reference_queries(
    row: Mapping[str, Any], tokenizer: Any
) -> tuple[list[ResponseReferenceQuery], list[dict[str, Any]]]:
    """Resolve model-specific response types onto exact real-token d1 queries."""

    base_queries, token_exclusions = pre_city_token_queries(
        row, tokenizer, depths=(1, 2), include_anchor=True
    )
    sites, parser_exclusions = parse_response_reference_sites(row)
    by_occurrence = {site.occurrence: site for site in sites}
    queries: list[ResponseReferenceQuery] = []
    exclusions = [*token_exclusions, *parser_exclusions]
    for base in base_queries:
        site = by_occurrence.get(int(base.occurrence))
        if site is None:
            exclusions.append(
                {
                    "occurrence": int(base.occurrence),
                    "target_city": base.city,
                    "status": "response_reference_site_not_registered",
                }
            )
            continue
        if site.city.casefold() != base.city.casefold():
            raise RuntimeError(
                "Response parser and token query disagree on cited city: "
                f"{site.city!r} != {base.city!r}"
            )
        if base.query_variant not in POSITION_VARIANTS:
            raise RuntimeError(
                f"Unregistered response-reference position: {base.query_variant}"
            )
        if (
            base.query_variant == "pre_city_d1"
            and base.token_distance_before_city != 1
        ):
            raise RuntimeError("Response-reference d1 is not exactly one token left")
        queries.append(ResponseReferenceQuery(base=base, site=site))
    return queries, exclusions


def attach_response_reference_types(
    attention: pd.DataFrame,
    generations: Iterable[Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Reuse d1 attention forwards after exact query-position identity checks."""

    frame = attention.loc[
        attention["query_variant"].astype(str).isin(POSITION_VARIANTS)
    ].copy()
    if frame.empty:
        raise ValueError("No registered pre-city attention rows are available")
    d1 = frame["query_variant"].astype(str).eq("pre_city_d1")
    if not pd.to_numeric(frame.loc[d1, "token_distance_before_city"]).eq(1).all():
        raise ValueError("Source attention contains a non-d1 query position")

    metadata: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in generations:
        request_id = str(row.get("request_id", row.get("stimulus_id")))
        sites, row_exclusions = parse_response_reference_sites(row)
        metadata.extend(
            {"request_id": request_id, **site.to_dict()} for site in sites
        )
        exclusions.extend(
            {"request_id": request_id, **value} for value in row_exclusions
        )
    registry = pd.DataFrame(metadata)
    if registry.empty:
        raise ValueError("Response-reference parser registered no sites")
    key = ["request_id", "occurrence", "target_city"]
    if registry.duplicated(key).any():
        raise ValueError("Response-reference registry contains duplicate sites")
    merged = frame.merge(registry, on=key, how="left", validate="many_to_one")
    missing = merged["response_type"].isna()
    if missing.any():
        examples = merged.loc[missing, key].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"Attention rows lack response-reference metadata: {examples}")
    merged["source_query_variant"] = merged["query_variant"].astype(str)
    merged["position_variant"] = merged["source_query_variant"]
    merged["query_variant"] = (
        "response_reference_" + merged["source_query_variant"].astype(str)
    )
    merged["query_position_reuse_audit"] = "PASS_IDENTICAL_SOURCE_PRE_CITY"
    merged["experiment_id"] = "response_typed_targeted_retrieval_attention_v1"
    return merged, exclusions


def rank_response_reference_heads(
    attention: pd.DataFrame, *, split: str = "discovery"
) -> pd.DataFrame:
    needed = {
        "model_label",
        "split",
        "response_type",
        "position_variant",
        "layer",
        "head",
        "target_needle_raw_mass",
        "target_needle_relative_mass",
        "target_needle_top1",
    }
    missing = sorted(needed - set(attention.columns))
    if missing:
        raise ValueError(f"Response-reference attention is missing {missing}")
    selected = attention.loc[
        attention["split"].astype(str).str.lower().eq(split.lower())
    ].copy()
    if selected.empty:
        raise ValueError(f"No response-reference rows matched split={split}")
    grouped = (
        selected.groupby(
            [
                "model_label",
                "response_type",
                "position_variant",
                "layer",
                "head",
            ],
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
            discovery_requests=("request_id", "nunique"),
        )
        .sort_values(
            [
                "model_label",
                "response_type",
                "position_variant",
                "discovery_target_raw_mass",
                "layer",
                "head",
            ],
            ascending=[True, True, True, False, True, True],
        )
        .reset_index(drop=True)
    )
    grouped["discovery_rank"] = (
        grouped.groupby(
            ["model_label", "response_type", "position_variant"]
        ).cumcount()
        + 1
    )
    grouped["mechanism"] = "targeted_retrieval"
    grouped["query_site_kind"] = "response_reference_pre_city"
    grouped["selection_metric"] = (
        "response_type_specific_mean_exact_prompt_needle_raw_mass"
    )
    grouped["selection_split"] = split
    return grouped


def rank_response_reference_consensus_heads(
    stratum_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Equal-weight every response-type/position stratum for one common bank."""

    stratum_count = len(REFERENCE_TYPES) * int(
        stratum_ranking["position_variant"].nunique()
    )
    consensus = (
        stratum_ranking.groupby(
            ["model_label", "layer", "head"], as_index=False
        )
        .agg(
            consensus_target_raw_mass=("discovery_target_raw_mass", "mean"),
            consensus_target_relative_mass=(
                "discovery_target_relative_mass",
                "mean",
            ),
            consensus_target_top1=("discovery_target_top1", "mean"),
            consensus_worst_stratum_raw_mass=(
                "discovery_target_raw_mass",
                "min",
            ),
            consensus_strata=("discovery_target_raw_mass", "size"),
        )
    )
    if not consensus["consensus_strata"].eq(stratum_count).all():
        raise ValueError("Consensus targeted-retrieval ranking lacks full strata")
    consensus = consensus.sort_values(
        [
            "model_label",
            "consensus_target_raw_mass",
            "consensus_worst_stratum_raw_mass",
            "layer",
            "head",
        ],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    consensus["consensus_rank"] = consensus.groupby("model_label").cumcount() + 1
    consensus["selection_metric"] = (
        "equal_stratum_mean_k_to_k_exact_prompt_needle_raw_mass"
    )
    return consensus


def rank_response_reference_position_consensus_heads(
    stratum_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Build one cross-response-type bank independently for each query position."""

    response_type_count = len(REFERENCE_TYPES)
    consensus = (
        stratum_ranking.groupby(
            ["model_label", "position_variant", "layer", "head"],
            as_index=False,
        )
        .agg(
            consensus_target_raw_mass=("discovery_target_raw_mass", "mean"),
            consensus_target_relative_mass=(
                "discovery_target_relative_mass",
                "mean",
            ),
            consensus_target_top1=("discovery_target_top1", "mean"),
            consensus_worst_response_type_raw_mass=(
                "discovery_target_raw_mass",
                "min",
            ),
            consensus_response_types=("discovery_target_raw_mass", "size"),
        )
    )
    if not consensus["consensus_response_types"].eq(response_type_count).all():
        raise ValueError(
            "Position-consensus targeted-retrieval ranking lacks all response types"
        )
    consensus = consensus.sort_values(
        [
            "model_label",
            "position_variant",
            "consensus_target_raw_mass",
            "consensus_worst_response_type_raw_mass",
            "layer",
            "head",
        ],
        ascending=[True, True, False, False, True, True],
    ).reset_index(drop=True)
    consensus["consensus_rank"] = (
        consensus.groupby(["model_label", "position_variant"]).cumcount() + 1
    )
    consensus["selection_metric"] = (
        "equal_response_type_mean_k_to_k_exact_prompt_needle_raw_mass_"
        "within_position"
    )
    return consensus


def _bank_metrics(
    frame: pd.DataFrame, heads: Sequence[tuple[int, int]], prefix: str
) -> dict[str, Any]:
    lookup = {
        (int(row.layer), int(row.head)): row for row in frame.itertuples(index=False)
    }
    selected = [lookup[(int(layer), int(head))] for layer, head in heads]
    raw = np.asarray([getattr(row, f"{prefix}_target_raw_mass") for row in selected])
    relative = np.asarray(
        [getattr(row, f"{prefix}_target_relative_mass") for row in selected],
        dtype=float,
    )
    finite = relative[np.isfinite(relative)]
    return {
        f"{prefix}_target_needle_raw_mass": float(raw.mean()),
        f"{prefix}_target_needle_relative_mass": (
            float(finite.mean()) if len(finite) else float("nan")
        ),
    }


def _constructible_rank_order(
    frame: pd.DataFrame, *, rank_column: str
) -> list[tuple[int, int]]:
    """Preserve ranking while reserving an exact same-layer control pool.

    Selecting at most half of a layer's heads guarantees that a disjoint bank
    with the identical layer histogram remains constructible at every prefix K.
    """

    layer_sizes = frame.groupby("layer")["head"].nunique().to_dict()
    capacities = {int(layer): int(size) // 2 for layer, size in layer_sizes.items()}
    selected_per_layer: dict[int, int] = {}
    ordered: list[tuple[int, int]] = []
    for row in frame.sort_values(rank_column).itertuples(index=False):
        layer = int(row.layer)
        if selected_per_layer.get(layer, 0) >= capacities[layer]:
            continue
        ordered.append((layer, int(row.head)))
        selected_per_layer[layer] = selected_per_layer.get(layer, 0) + 1
    return ordered


def build_response_reference_causal_plan(
    attention_csv: str | Path,
    output_dir: str | Path,
    *,
    config: Any,
) -> dict[str, Path]:
    from .causal import layer_matched_random_controls

    config.validate()
    attention = pd.read_csv(attention_csv)
    ranking = rank_response_reference_heads(attention)
    position_consensus = rank_response_reference_position_consensus_heads(ranking)
    consensus = rank_response_reference_consensus_heads(ranking)
    confirmation = attention.loc[
        attention["split"].astype(str).str.lower().eq("confirmation")
    ].copy()
    if confirmation.empty:
        raise ValueError("Response-reference plan requires confirmation attention")
    confirmation_summary = (
        confirmation.groupby(
            [
                "model_label",
                "response_type",
                "position_variant",
                "layer",
                "head",
            ],
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
    evaluation = ranking.merge(
        confirmation_summary,
        on=[
            "model_label",
            "response_type",
            "position_variant",
            "layer",
            "head",
        ],
        how="left",
        validate="one_to_one",
    )
    observed = {
        (str(row.response_type), str(row.position_variant))
        for row in evaluation[["response_type", "position_variant"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    observed_positions = set(evaluation["position_variant"].astype(str))
    allowed_positions = {*POSITION_VARIANTS, TARGETED_REFERENCE_VARIANT}
    if not observed_positions or not observed_positions.issubset(allowed_positions):
        raise ValueError(
            f"Unsupported response-reference positions: {sorted(observed_positions)}"
        )
    expected = {
        (response_type, position)
        for response_type in REFERENCE_TYPES
        for position in observed_positions
    }
    if observed != expected:
        raise ValueError(
            f"Response/position coverage mismatch: expected={sorted(expected)} "
            f"observed={sorted(observed)}"
        )

    plan_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for (model, response_type, position_variant), group in evaluation.groupby(
        ["model_label", "response_type", "position_variant"], sort=True
    ):
        frame = group.sort_values("discovery_rank").reset_index(drop=True)
        ordered = _constructible_rank_order(frame, rank_column="discovery_rank")
        for bank_size in config.causal_head_bank_sizes:
            if int(bank_size) > len(ordered):
                continue
            ranked = ordered[: int(bank_size)]
            common = {
                "model_label": str(model),
                "mechanism": "targeted_retrieval",
                "response_type": str(response_type),
                "position_variant": str(position_variant),
                "query_variant": f"response_reference_{position_variant}",
                "bank_scope": "response_type_and_position_specific",
                "experiment_id": "response_typed_targeted_retrieval_ablation_v1",
                "bank_size": int(bank_size),
            }

            def make_row(
                heads: Sequence[tuple[int, int]], condition: str, repeat: int
            ) -> dict[str, Any]:
                return {
                    **common,
                    "condition": condition,
                    "repeat": int(repeat),
                    "heads": json.dumps(list(heads)),
                    **_bank_metrics(frame, heads, "discovery"),
                    **_bank_metrics(frame, heads, "confirmation"),
                    "target_needle_raw_mass": _bank_metrics(
                        frame, heads, "discovery"
                    )["discovery_target_needle_raw_mass"],
                    "target_needle_relative_mass": _bank_metrics(
                        frame, heads, "discovery"
                    )["discovery_target_needle_relative_mass"],
                }

            plan_rows.append(
                make_row(ranked, "response_reference_targeted_retrieval_ranked", 0)
            )
            try:
                controls = layer_matched_random_controls(
                    frame,
                    ranked,
                    repeats=config.causal_random_controls,
                    seed_text=(
                        f"v5-response-reference:{model}:{response_type}:"
                        f"{position_variant}:K{bank_size}"
                    ),
                )
            except ValueError as error:
                skipped.append(
                    {
                        **common,
                        "status": "not_constructible_disjoint_exact_layer_match",
                        "reason": str(error),
                    }
                )
                continue
            for repeat, control in enumerate(controls, start=1):
                plan_rows.append(
                    make_row(control, "layer_matched_random", repeat)
                )

    # Primary targeted-retrieval banks generalize across response formats but
    # remain separate across query positions.  Each position-consensus ranking
    # equal-weights the three response types in discovery; the frozen bank is
    # then applied to all three types in confirmation.
    for (model, position_variant), consensus_frame in position_consensus.groupby(
        ["model_label", "position_variant"], sort=True
    ):
        consensus_frame = consensus_frame.sort_values("consensus_rank").reset_index(
            drop=True
        )
        ordered = _constructible_rank_order(
            consensus_frame, rank_column="consensus_rank"
        )
        applications = evaluation.loc[
            evaluation["model_label"].astype(str).eq(str(model))
            & evaluation["position_variant"].astype(str).eq(str(position_variant))
        ]
        for bank_size in config.causal_head_bank_sizes:
            if int(bank_size) > len(ordered):
                continue
            ranked = ordered[: int(bank_size)]
            controls = layer_matched_random_controls(
                consensus_frame,
                ranked,
                repeats=config.causal_random_controls,
                seed_text=(
                    f"v5-response-reference:{model}:position-consensus:"
                    f"{position_variant}:K{bank_size}"
                ),
            )
            for response_type, stratum in applications.groupby(
                "response_type", sort=True
            ):
                common = {
                    "model_label": str(model),
                    "mechanism": "targeted_retrieval",
                    "response_type": str(response_type),
                    "position_variant": str(position_variant),
                    "query_variant": f"response_reference_{position_variant}",
                    "bank_scope": "position_consensus",
                    "experiment_id": (
                        "response_typed_targeted_retrieval_ablation_v1"
                    ),
                    "bank_size": int(bank_size),
                }

                def position_row(
                    heads: Sequence[tuple[int, int]],
                    condition: str,
                    repeat: int,
                ) -> dict[str, Any]:
                    discovery = _bank_metrics(stratum, heads, "discovery")
                    return {
                        **common,
                        "condition": condition,
                        "repeat": int(repeat),
                        "heads": json.dumps(list(heads)),
                        **discovery,
                        **_bank_metrics(stratum, heads, "confirmation"),
                        "target_needle_raw_mass": discovery[
                            "discovery_target_needle_raw_mass"
                        ],
                        "target_needle_relative_mass": discovery[
                            "discovery_target_needle_relative_mass"
                        ],
                    }

                plan_rows.append(
                    position_row(
                        ranked, "response_reference_position_consensus_ranked", 0
                    )
                )
                for repeat, control in enumerate(controls, start=1):
                    plan_rows.append(
                        position_row(control, "layer_matched_random", repeat)
                    )

    # The cross-position common bank is retained only as a supplementary
    # robustness analysis; it is not used to choose the primary query position.
    # exact-span enrichment across response formats and token positions.  The
    # consensus ranking equal-weights all nine discovery strata; the same
    # frozen heads and the same control banks are then applied separately at
    # every response-type/position stratum in confirmation.
    for model, consensus_frame in consensus.groupby("model_label", sort=True):
        consensus_frame = consensus_frame.sort_values("consensus_rank").reset_index(
            drop=True
        )
        ordered = _constructible_rank_order(
            consensus_frame, rank_column="consensus_rank"
        )
        applications = evaluation.loc[
            evaluation["model_label"].astype(str).eq(str(model))
        ]
        for bank_size in config.causal_head_bank_sizes:
            if int(bank_size) > len(ordered):
                continue
            ranked = ordered[: int(bank_size)]
            try:
                controls = layer_matched_random_controls(
                    consensus_frame,
                    ranked,
                    repeats=config.causal_random_controls,
                    seed_text=f"v5-response-reference:{model}:consensus:K{bank_size}",
                )
            except ValueError as error:
                skipped.append(
                    {
                        "model_label": str(model),
                        "bank_scope": "unified_consensus",
                        "bank_size": int(bank_size),
                        "status": "not_constructible_disjoint_exact_layer_match",
                        "reason": str(error),
                    }
                )
                controls = []
            for (response_type, position_variant), stratum in applications.groupby(
                ["response_type", "position_variant"], sort=True
            ):
                common = {
                    "model_label": str(model),
                    "mechanism": "targeted_retrieval",
                    "response_type": str(response_type),
                    "position_variant": str(position_variant),
                    "query_variant": f"response_reference_{position_variant}",
                    "bank_scope": "unified_consensus",
                    "experiment_id": (
                        "response_typed_targeted_retrieval_ablation_v1"
                    ),
                    "bank_size": int(bank_size),
                }

                def consensus_row(
                    heads: Sequence[tuple[int, int]],
                    condition: str,
                    repeat: int,
                ) -> dict[str, Any]:
                    discovery = _bank_metrics(stratum, heads, "discovery")
                    return {
                        **common,
                        "condition": condition,
                        "repeat": int(repeat),
                        "heads": json.dumps(list(heads)),
                        **discovery,
                        **_bank_metrics(stratum, heads, "confirmation"),
                        "target_needle_raw_mass": discovery[
                            "discovery_target_needle_raw_mass"
                        ],
                        "target_needle_relative_mass": discovery[
                            "discovery_target_needle_relative_mass"
                        ],
                    }

                plan_rows.append(
                    consensus_row(
                        ranked, "response_reference_consensus_ranked", 0
                    )
                )
                for repeat, control in enumerate(controls, start=1):
                    plan_rows.append(
                        consensus_row(control, "layer_matched_random", repeat)
                    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "ranking": output / "discovery_head_ranking.csv",
        "position_consensus_ranking": (
            output / "discovery_position_consensus_head_ranking.csv"
        ),
        "consensus_ranking": output / "discovery_consensus_head_ranking.csv",
        "confirmation": output / "confirmation_head_evaluation.csv",
        "plan": output / "causal_plan.csv",
        "audit": output / "causal_plan_audit.json",
    }
    ranking.to_csv(paths["ranking"], index=False)
    position_consensus.to_csv(paths["position_consensus_ranking"], index=False)
    consensus.to_csv(paths["consensus_ranking"], index=False)
    evaluation.to_csv(paths["confirmation"], index=False)
    pd.DataFrame(plan_rows).to_csv(paths["plan"], index=False)
    paths["audit"].write_text(
        json.dumps(
            {
                "schema_version": RESPONSE_REFERENCE_SCHEMA_VERSION,
                "experiment_id": "response_typed_targeted_retrieval_ablation_v1",
                "response_types": list(REFERENCE_TYPES),
                "position_variants": sorted(observed_positions),
                "model_specific_parsers": {
                    "Qwen3-8B": "qwen3_response_reference_parser_v1",
                    "Gemma4-E4B": "gemma4_response_reference_parser_v1",
                },
                "query_definition": (
                    "pre_reference_d1 is the real baseline token immediately "
                    "before each parser-registered citation start; legacy pre-city "
                    "positions remain supported for archived robustness runs"
                ),
                "source_forward_reuse": (
                    "none for pre_reference_d1; its k-to-k attention is captured "
                    "from the dedicated response-reference parser registry"
                ),
                "selection_split": "discovery",
                "confirmation_used_for_selection": False,
                "selection_metric": (
                    "primary: within each query position, equal-response-type "
                    "mean k-to-k exact prompt-needle raw mass"
                ),
                "primary_bank_scope": "position_consensus",
                "primary_position_policy": (
                    "the trace pre-reference experiment pre-registers only "
                    "pre_reference_d1 before ablation"
                ),
                "consensus_selection_metric": (
                    "equal-weight mean k-to-k mass across all nine "
                    "response-type/position strata; supplementary robustness only"
                ),
                "registered_bank_sizes": list(config.causal_head_bank_sizes),
                "constructibility_constraint": (
                    "rank-preserving per-layer cap=floor(layer_head_count/2), "
                    "which reserves a disjoint exact layer-matched pool for "
                    "every registered K"
                ),
                "random_control": "disjoint exact layer-matched, three repeats",
                "skipped_controls": skipped,
                "attention_source": str(Path(attention_csv).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def run_response_reference_head_ablation_trials(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    row: Mapping[str, Any],
    *,
    response_type: str,
    position_variant: str,
    heads: Sequence[tuple[int, int]],
    condition: str,
) -> list[dict[str, Any]]:
    """Ablate one frozen bank at parser-matched response citations."""

    from .pre_city import run_pre_city_head_ablation_trials

    if response_type not in REFERENCE_TYPES:
        raise ValueError(f"Unknown response-reference type: {response_type}")
    if position_variant not in {*POSITION_VARIANTS, TARGETED_REFERENCE_VARIANT}:
        raise ValueError(f"Unknown response-reference position: {position_variant}")
    queries, exclusions = (
        targeted_retrieval_queries(row, tokenizer)
        if position_variant == TARGETED_REFERENCE_VARIANT
        else response_reference_queries(row, tokenizer)
    )
    active = [
        query
        for query in queries
        if query.site.response_type == response_type
        and query.base.query_variant == position_variant
    ]
    if not active:
        return [
            {
                "schema_version": RESPONSE_REFERENCE_SCHEMA_VERSION,
                "experiment_id": "response_typed_targeted_retrieval_ablation_v1",
                "condition": condition,
                "request_id": row.get("request_id", row.get("stimulus_id")),
                "model_label": row.get("model_label", row.get("model")),
                "seed": row.get("seed"),
                "split": row.get("split"),
                "gold_count": len(gold_records(row)),
                "mechanism": "targeted_retrieval",
                "query_variant": f"response_reference_{position_variant}",
                "position_variant": position_variant,
                "response_type": response_type,
                "status": "no_matching_response_reference_type",
                "parser_exclusions": exclusions,
                "heads": [[int(layer), int(head)] for layer, head in heads],
                "bank_size": int(len(heads)),
            }
        ]
    by_occurrence = {query.base.occurrence: query for query in active}
    if position_variant == TARGETED_REFERENCE_VARIANT:
        results = _run_pre_reference_head_ablation_trials(
            model,
            tokenizer,
            adapter,
            row,
            queries=tuple(by_occurrence.values()),
            heads=heads,
            condition=condition,
        )
    else:
        results = run_pre_city_head_ablation_trials(
            model,
            tokenizer,
            adapter,
            row,
            query_variant=position_variant,
            heads=heads,
            condition=condition,
            occurrences=tuple(by_occurrence),
        )
    enriched: list[dict[str, Any]] = []
    for result in results:
        occurrence = result.get("occurrence")
        query = (
            by_occurrence.get(int(occurrence))
            if occurrence is not None
            else None
        )
        payload = dict(result)
        payload.update(
            {
                "schema_version": RESPONSE_REFERENCE_SCHEMA_VERSION,
                "experiment_id": "response_typed_targeted_retrieval_ablation_v1",
                "query_variant": f"response_reference_{position_variant}",
                "source_query_variant": position_variant,
                "position_variant": position_variant,
                "response_type": response_type,
                "query_position_reuse_audit": (
                    "NOT_REUSED_DEDICATED_PRE_REFERENCE_PARSER"
                    if position_variant == TARGETED_REFERENCE_VARIANT
                    else "PASS_IDENTICAL_SOURCE_PRE_CITY"
                ),
            }
        )
        if query is not None:
            payload.update(query.to_dict())
        enriched.append(payload)
    return enriched


def _run_pre_reference_head_ablation_trials(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    row: Mapping[str, Any],
    *,
    queries: Sequence[ResponseReferenceQuery],
    heads: Sequence[tuple[int, int]],
    condition: str,
) -> list[dict[str, Any]]:
    """Generate the parser-registered citation after each pre-reference query."""

    from .causal import (
        _continuation_metrics,
        _local_head_ablation_logits,
        generate_with_head_ablation_at_positions,
    )

    baseline_ids = output_token_ids(row)
    prompt_count = len(prompt_token_ids(row))
    common = {
        "schema_version": RESPONSE_REFERENCE_SCHEMA_VERSION,
        "experiment_id": "trace_pre_reference_k_to_k_targeted_retrieval_v1",
        "condition": condition,
        "request_id": row.get("request_id", row.get("stimulus_id")),
        "model_label": row.get("model_label", row.get("model")),
        "seed": row.get("seed"),
        "split": row.get("split"),
        "gold_count": len(gold_records(row)),
        "mechanism": "targeted_retrieval",
        "query_variant": TARGETED_REFERENCE_VARIANT,
        "query_site_kind": "pre_reference_token",
        "transition_phase": "retrieve",
        "heads": [[int(layer), int(head)] for layer, head in heads],
        "bank_size": len(heads),
        "behavioral_endpoint": "actual_greedy_next_needle_token_sequence",
        "final_count_evaluated": False,
    }
    output: list[dict[str, Any]] = []
    for query in queries:
        start = int(query.base.query_output_token_count)
        end = int(query.target_after_token or query.base.city_after_token)
        target_ids = baseline_ids[start:end]
        if not target_ids:
            output.append(
                {**common, **query.to_dict(), "status": "empty_citation_target"}
            )
            continue
        target_encoding = baseline_prefix_encoding(
            row, tokenizer, query.base, prefix_output_token_count=end
        )
        logits = _local_head_ablation_logits(
            model,
            adapter,
            target_encoding,
            heads,
            hook_position=prompt_count + start - 1,
            target_token_count=len(target_ids),
        )
        generation_encoding = baseline_prefix_encoding(row, tokenizer, query.base)
        generation = generate_with_head_ablation_at_positions(
            model,
            tokenizer,
            adapter,
            generation_encoding,
            heads,
            hook_positions=[generation_encoding.query_position],
            max_new_tokens=len(target_ids),
        )
        generated = tuple(int(value) for value in generation["generated_token_ids"])
        target = tuple(int(value) for value in target_ids)
        matched = sum(
            int(index < len(generated) and generated[index] == token_id)
            for index, token_id in enumerate(target)
        )
        output.append(
            {
                **common,
                **query.to_dict(),
                "status": "ok",
                "target_token_count": len(target),
                "target_text": tokenizer.decode(
                    list(target),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "generated_next_needle_token_ids": list(generated),
                "generated_next_needle_text": tokenizer.decode(
                    list(generated),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "next_needle_exact": bool(generated == target),
                "next_needle_first_token_exact": bool(
                    generated and generated[0] == target[0]
                ),
                "next_needle_token_accuracy": float(matched / len(target)),
                "next_needle_generated_token_count": len(generated),
                "head_ablation_hook_audit": generation["head_ablation_hook_audit"],
                "k_to_k_registry_audit": (
                    "PASS_RESPONSE_OCCURRENCE_TO_EXACT_CITY_PROMPT_SPAN"
                ),
                **_continuation_metrics(logits, target_ids),
            }
        )
    return output
