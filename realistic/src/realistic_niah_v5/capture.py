from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    capture_post_block_states,
    capture_span_states,
    position_attention_outputs,
)
from realistic_niah_v4.prompts import TokenSpan
from .encoding import build_native_trace_encoding
from .parsing import (
    PARSER_IMPLEMENTATION,
    PARSER_SCHEMA_VERSION,
    PARSER_SELECTION_RULE,
    SITE_SCHEMA_VERSION,
    TraceTokenSite,
    align_trace_sites,
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    output_token_ids,
    parse_trace_record,
    prompt_token_ids,
    raw_output_text,
    trace_char_sites,
)
from .spec import V5Config


CAPTURE_SCHEMA_VERSION = "realistic_niah_v5_trace_capture_v4"
ATTENTION_SCHEMA_VERSION = "realistic_niah_v5_mechanism_attention_v3"
NO_ALIGNED_TRACE_SITES_REASON = "no_aligned_registered_v5_trace_sites"


class NoAlignedTraceSitesError(ValueError):
    """A record has no registered trace site that can be captured."""


@dataclass(frozen=True)
class _ModelInput:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _selected_sites(
    token_sites: Sequence[TraceTokenSite],
    config: V5Config,
    site_kinds: Sequence[str] | None = None,
    *,
    site_ids: Sequence[str] | None = None,
) -> list[TraceTokenSite]:
    if site_kinds is not None and site_ids is not None:
        raise ValueError("site_kinds and site_ids are mutually exclusive")
    registered_ids = None if site_ids is None else {str(value) for value in site_ids}
    if registered_ids is not None and not registered_ids:
        raise ValueError("site_ids cannot be empty")
    registered = set(config.registered_sites)
    if site_kinds is not None:
        requested = set(map(str, site_kinds))
        unknown = sorted(requested - registered)
        if unknown:
            raise ValueError(f"Unregistered V5 trace sites requested: {unknown}")
        if not requested:
            raise ValueError("site_kinds cannot be empty")
        registered = requested
    result = [
        site
        for site in token_sites
        if (
            (
                site.char_site.site_kind in registered
                if registered_ids is None
                else site.char_site.site_id in registered_ids
            )
            and site.alignment_eligible
        )
    ]
    return sorted(
        result,
        key=lambda site: (
            site.char_site.char_end,
            site.char_site.site_kind,
            site.char_site.site_id,
        ),
    )


def _full_input(row: Mapping[str, Any]) -> _ModelInput:
    prompt = prompt_token_ids(row)
    output = output_token_ids(row)
    mask = row.get("attention_mask")
    prompt_mask = (
        tuple(int(value) for value in mask)
        if mask is not None
        else (1,) * len(prompt)
    )
    if len(prompt_mask) != len(prompt):
        raise ValueError("Prompt attention-mask length mismatch")
    return _ModelInput(
        input_ids=prompt + output,
        attention_mask=prompt_mask + (1,) * len(output),
    )


def _literal_item_spans(
    token_sites: Sequence[TraceTokenSite], prompt_count: int
) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for site in token_sites:
        if site.char_site.site_kind != "item_end":
            continue
        start = site.literal_token_start
        end = site.literal_token_end
        if start is None or end is None or end <= start:
            continue
        occurrence = int(site.char_site.occurrence or 0)
        spans.append(
            TokenSpan(
                slot_index=occurrence,
                start=prompt_count + int(start),
                end=prompt_count + int(end),
                active=True,
                kind="native_trace_item",
                canonical_length=end - start,
                model_token_length=end - start,
            )
        )
    return spans


@torch.inference_mode()
def capture_trace_record(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    config: V5Config,
    output_dir: str | Path,
    layers: Iterable[int] | None = None,
    site_kinds: Sequence[str] | None = None,
    capture_span_pooling: bool = True,
    overwrite: bool = False,
    site_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    config.validate()
    if site_kinds is not None and site_ids is not None:
        raise ValueError("site_kinds and site_ids are mutually exclusive")
    output = Path(output_dir)
    manifest_path = output / "capture_manifest.json"
    if manifest_path.exists() and not overwrite:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            saved.get("schema_version") != CAPTURE_SCHEMA_VERSION
            or saved.get("parser_implementation") != PARSER_IMPLEMENTATION
        ):
            raise RuntimeError(
                f"Incompatible existing V5 capture shard: {manifest_path}. "
                "Use --overwrite or a new output directory."
            )
        if layers is not None and saved.get("layers") != sorted(
            {int(layer) for layer in layers}
        ):
            raise RuntimeError(
                f"Existing V5 layer grid differs in {manifest_path}; "
                "use --overwrite or a new output directory."
            )
        expected_site_kinds = sorted(
            set(config.registered_sites if site_kinds is None else site_kinds)
        )
        if saved.get("selected_site_kinds") != expected_site_kinds:
            raise RuntimeError(
                f"Existing V5 site grid differs in {manifest_path}; "
                "use --overwrite or a new output directory."
            )
        expected_site_ids = (
            None if site_ids is None else sorted({str(value) for value in site_ids})
        )
        if saved.get("selected_site_ids") != expected_site_ids:
            raise RuntimeError(
                f"Existing V5 site-id grid differs in {manifest_path}; "
                "use --overwrite or a new output directory."
            )
        if bool(saved.get("capture_span_pooling")) != bool(capture_span_pooling):
            raise RuntimeError(
                f"Existing V5 span-pooling policy differs in {manifest_path}; "
                "use --overwrite or a new output directory."
            )
        return saved
    family = infer_model_family(row)
    raw = raw_output_text(row)
    parser = find_trace_count_sequence(
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
    sites = _selected_sites(token_sites, config, site_kinds, site_ids=site_ids)
    if not sites:
        raise NoAlignedTraceSitesError("No aligned registered V5 trace sites")
    started = time.perf_counter()
    full = _full_input(row)
    prompt_count = len(prompt_token_ids(row))
    literal_indices = [
        index
        for index, site in enumerate(sites)
        if site.alignment_strategy == "literal_baseline_token_prefix"
    ]
    state_by_site: dict[int, torch.Tensor] = {}
    layer_indices: torch.Tensor | None = None
    if literal_indices:
        positions = [
            prompt_count + int(sites[index].prefix_token_count or 0) - 1
            for index in literal_indices
        ]
        _logits, captured = capture_post_block_states(
            model,
            adapter,
            full,
            positions,
            layers=layers,
        )
        ordered_layers = tuple(sorted(captured))
        layer_indices = torch.tensor(ordered_layers, dtype=torch.long)
        stacked = torch.stack([captured[layer] for layer in ordered_layers], dim=0)
        for position_axis, site_index in enumerate(literal_indices):
            state_by_site[site_index] = stacked[:, position_axis, :]
    for site_index, site in enumerate(sites):
        if site_index in state_by_site:
            continue
        encoding = build_native_trace_encoding(
            row,
            tokenizer,
            site_id=site.char_site.site_id,
            candidate_counts=config.candidate_counts,
        )
        _logits, captured = capture_post_block_states(
            model,
            adapter,
            encoding,
            [encoding.query_position],
            layers=layers,
        )
        ordered_layers = tuple(sorted(captured))
        current_layers = torch.tensor(ordered_layers, dtype=torch.long)
        if layer_indices is None:
            layer_indices = current_layers
        elif not torch.equal(layer_indices, current_layers):
            raise RuntimeError("V5 site captures disagree on decoder layers")
        state_by_site[site_index] = torch.stack(
            [captured[layer][0] for layer in ordered_layers], dim=0
        )
    if layer_indices is None or len(state_by_site) != len(sites):
        raise RuntimeError("V5 did not capture every registered trace site")
    states = torch.stack(
        [state_by_site[index] for index in range(len(sites))], dim=0
    )
    numpy_dtype = np.float16 if config.hidden_save_dtype == "float16" else np.float32
    arrays: dict[str, np.ndarray] = {
        "layer_indices": layer_indices.numpy(),
        "site_states": states.numpy().astype(numpy_dtype, copy=False),
    }
    item_spans = (
        _literal_item_spans(token_sites, prompt_count)
        if capture_span_pooling
        else []
    )
    span_site_ids: list[str] = []
    if item_spans:
        pooled = capture_span_states(
            model,
            adapter,
            full,
            item_spans,
            layers=layer_indices.tolist(),
        )
        arrays["item_span_mean"] = (
            pooled["span_mean"].permute(1, 0, 2).numpy().astype(numpy_dtype, copy=False)
        )
        arrays["item_span_end_literal"] = (
            pooled["span_end"].permute(1, 0, 2).numpy().astype(numpy_dtype, copy=False)
        )
        span_site_ids = [f"item_span:{span.slot_index}" for span in item_spans]
    _save_npz(output / "states.npz", **arrays)
    parsed = parse_trace_record(row)
    manifest = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "stimulus_id": row.get("stimulus_id"),
        "request_id": row.get("request_id"),
        "model_label": row.get("model_label"),
        "model_family": family,
        "entity_domain": str(row.get("entity_domain", "city")),
        "seed": row.get("seed"),
        "split": row.get("split"),
        "gold_count": len(gold_records(row)),
        "parsed_count": parsed["parsed_count"],
        "exact_count": parsed["exact_count"],
        "parser_implementation": parsed["parser_implementation"],
        "parser_schema_version": parsed["schema_version"],
        "parser_selection_rule": parsed["parser_selection_rule"],
        "rank_episode_schema_version": parsed["rank_episode_schema_version"],
        "rank_episode_selection_policy": parsed[
            "rank_episode_selection_policy"
        ],
        "site_schema_version": SITE_SCHEMA_VERSION,
        "parser_file_sha256": parsed["parser_file_sha256"],
        "sequence_source": parsed["sequence_source"],
        "episode_parse": parsed["episode_parse"],
        "parser": parsed["parser"],
        "prompt_token_count": prompt_count,
        "output_token_count": len(output_token_ids(row)),
        "layers": layer_indices.tolist(),
        "selected_site_kinds": sorted(
            set(config.registered_sites if site_kinds is None else site_kinds)
        ),
        "selected_site_ids": (
            None if site_ids is None else sorted({str(value) for value in site_ids})
        ),
        "capture_span_pooling": bool(capture_span_pooling),
        "site_rows": [site.to_dict() for site in sites],
        "span_site_ids": span_site_ids,
        "states_file": "states.npz",
        "site_states_shape": list(states.shape),
        "save_dtype": config.hidden_save_dtype,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _mass(row: torch.Tensor, *, key_start: int, spans: Sequence[Any]) -> float:
    total = 0.0
    for span in spans:
        left = max(int(span.start), int(key_start)) - int(key_start)
        right = min(int(span.end), int(key_start) + int(row.shape[-1])) - int(key_start)
        if right > left:
            total += float(row[left:right].sum().item())
    return total


@torch.inference_mode()
def capture_trace_attention_metrics(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    config: V5Config,
    mechanisms: Sequence[str] = (
        "targeted_retrieval",
        "progress_transition",
    ),
) -> pd.DataFrame:
    """Capture native-thinking head scores aligned to semantic targets.

    ``targeted_retrieval`` queries the parser marker boundary for item k and
    scores attention to the matching prompt city-record span.  The
    ``progress_transition`` query is the accepted item-k endpoint and scores
    attention to the prompt span of the next city that the trace actually
    enumerates.  The latter is defined only for nonterminal k.
    """

    allowed = {"targeted_retrieval", "progress_transition"}
    unknown = sorted(set(mechanisms) - allowed)
    if unknown:
        raise ValueError(f"Unknown V5 attention mechanisms: {unknown}")
    parsed = parse_trace_record(row)
    if not bool(parsed["parser"]["detected"]):
        return pd.DataFrame()
    raw = raw_output_text(row)
    parser = find_trace_count_sequence(
        raw,
        model_family=parsed["model_family"],
        gold_records=gold_records(row),
    )
    token_sites = align_trace_sites(
        tokenizer,
        raw_text=raw,
        baseline_output_token_ids=output_token_ids(row),
        sites=trace_char_sites(raw, parser),
    )
    by_id = {site.char_site.site_id: site for site in token_sites}
    trace_cities = [str(value) for value in parser.item_gold_cities]
    queries: list[tuple[str, Any, str]] = []
    if "targeted_retrieval" in mechanisms:
        for occurrence, city in enumerate(trace_cities, start=1):
            site = by_id.get(f"marker_end:{occurrence}")
            if site is not None and site.alignment_eligible:
                queries.append(("targeted_retrieval", site, city))
    if "progress_transition" in mechanisms:
        for occurrence, target_city in enumerate(trace_cities[1:], start=1):
            site = by_id.get(f"item_end:{occurrence}")
            if site is not None and site.alignment_eligible:
                queries.append(("progress_transition", site, target_city))
    rows: list[dict[str, Any]] = []
    for mechanism, site, target_city in queries:
        encoding = build_native_trace_encoding(
            row,
            tokenizer,
            site_id=site.char_site.site_id,
            candidate_counts=config.candidate_counts,
        )
        if not encoding.prompt_record_spans:
            raise ValueError(
                "Mechanism-specific attention requires prompt_record_spans; "
                "regenerate V5 rows from frozen V4.4 stimuli"
            )
        prompt_by_city = {
            span.city.casefold(): span for span in encoding.prompt_record_spans
        }
        target_span = prompt_by_city.get(target_city.casefold())
        if target_span is None:
            raise RuntimeError(f"No prompt record span for target city {target_city}")
        attention_rows, key_starts, _logits = position_attention_outputs(
            model,
            adapter,
            encoding,
            encoding.query_position,
        )
        occurrence = int(site.char_site.occurrence or 0)
        for layer, (attention, key_start) in enumerate(
            zip(attention_rows, key_starts)
        ):
            for head in range(attention.shape[0]):
                head_row = attention[head]
                record_masses = [
                    _mass(head_row, key_start=key_start, spans=[span])
                    for span in encoding.prompt_record_spans
                ]
                prompt_records_mass = float(sum(record_masses))
                target_mass = _mass(
                    head_row, key_start=key_start, spans=[target_span]
                )
                target_relative_mass = (
                    target_mass / prompt_records_mass
                    if prompt_records_mass > 0
                    else float("nan")
                )
                target_top1 = bool(
                    prompt_records_mass > 0
                    and record_masses
                    and int(np.argmax(record_masses))
                    == list(encoding.prompt_record_spans).index(target_span)
                )
                trace_mass = _mass(
                    head_row, key_start=key_start, spans=encoding.needle_spans
                )
                prompt_right = min(
                    encoding.prompt_token_count - key_start,
                    int(head_row.shape[-1]),
                )
                prompt_mass = (
                    float(head_row[: max(0, prompt_right)].sum().item())
                    if prompt_right > 0
                    else 0.0
                )
                rows.append(
                    {
                        "schema_version": ATTENTION_SCHEMA_VERSION,
                        "request_id": encoding.request_id,
                        "stimulus_id": encoding.stimulus_id,
                        "model_label": encoding.model_label,
                        "seed": encoding.seed,
                        "split": encoding.split,
                        "gold_count": encoding.count,
                        "trace_one_to_one": bool(
                            parsed["parser"].get("trace_one_to_one")
                        ),
                        "trace_category": parsed["parser"].get("trace_category"),
                        "final_exact_count": bool(parsed.get("exact_count")),
                        "mechanism": mechanism,
                        "site_id": site.char_site.site_id,
                        "site_kind": site.char_site.site_kind,
                        "occurrence": occurrence,
                        "query_city": site.char_site.city,
                        "target_city": target_city,
                        "target_slot_index": int(target_span.slot_index),
                        "layer": layer,
                        "head": head,
                        "key_start": key_start,
                        # Canonical V5 result contract.  These are computed on
                        # the exact semantic needle spans in the frozen V4.4
                        # prompt, at one registered trace query row.
                        "target_needle_raw_mass": target_mass,
                        "all_active_needles_raw_mass": prompt_records_mass,
                        "target_needle_relative_mass": target_relative_mass,
                        "target_needle_top1": target_top1,
                        # Compatibility aliases for pre-freeze V5 consumers.
                        "target_prompt_record_mass": target_mass,
                        "prompt_records_total_mass": prompt_records_mass,
                        "target_within_records_fraction": target_relative_mass,
                        "target_record_top1": target_top1,
                        "trace_item_mass": trace_mass,
                        "full_prompt_mass": prompt_mass,
                        "other_generated_mass": max(
                            0.0, 1.0 - trace_mass - prompt_mass
                        ),
                        "row_sum": float(head_row.sum().item()),
                    }
                )
    return pd.DataFrame(rows)


@torch.inference_mode()
def capture_answer_query_attention_metrics(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    site_id: str = "answer_query_v3",
) -> pd.DataFrame:
    """Measure prompt and registered-trace aggregation at the answer query.

    Prompt aggregation is computed over the literal oracle needle-record spans.
    Trace aggregation is computed independently over the frozen,
    parser-registered reasoning item spans.  For each route we retain the full
    per-span mass vector and the V4.4 non-thinking broad score
    ``total_mass * exp(entropy(per_span_mass)) / span_count``.  The two routes
    are ranked separately downstream; neither is replaced by an unregistered
    broad-context aggregate.  The table always preserves the exact prompt
    needle raw/relative mass required by the V5 result contract.
    """

    parsed = parse_trace_record(row)
    if not bool(parsed["parser"]["detected"]):
        return pd.DataFrame()
    encoding = build_native_trace_encoding(
        row,
        tokenizer,
        site_id=site_id,
        candidate_counts=tuple(range(1, 11)),
    )
    if not encoding.prompt_record_spans:
        raise ValueError(
            "Answer-query attention requires frozen prompt_record_spans"
        )
    raw = raw_output_text(row)
    parser = find_trace_count_sequence(
        raw,
        model_family=parsed["model_family"],
        gold_records=gold_records(row),
    )
    aligned_sites = align_trace_sites(
        tokenizer,
        raw_text=raw,
        baseline_output_token_ids=output_token_ids(row),
        sites=trace_char_sites(raw, parser),
    )
    aligned_by_id = {site.char_site.site_id: site for site in aligned_sites}
    active_site = aligned_by_id.get(site_id)
    if active_site is None or not active_site.alignment_eligible:
        raise RuntimeError(f"Answer-query site {site_id!r} is not token aligned")
    legacy_site = aligned_by_id.get("answer_query")
    legacy_alias_same_endpoint = bool(
        legacy_site is not None
        and legacy_site.alignment_eligible
        and legacy_site.endpoint_token == active_site.endpoint_token
    )
    attention_rows, key_starts, _logits = position_attention_outputs(
        model,
        adapter,
        encoding,
        encoding.query_position,
    )
    rows: list[dict[str, Any]] = []
    for layer, (attention, key_start) in enumerate(
        zip(attention_rows, key_starts)
    ):
        for head in range(attention.shape[0]):
            head_row = attention[head]
            span_masses = [
                _mass(head_row, key_start=key_start, spans=[span])
                for span in encoding.prompt_record_spans
            ]
            target_mass = float(sum(span_masses))
            prompt_right = min(
                encoding.prompt_token_count - int(key_start),
                int(head_row.shape[-1]),
            )
            prompt_mass = (
                float(head_row[: max(0, prompt_right)].sum().item())
                if prompt_right > 0
                else 0.0
            )
            relative_mass = (
                target_mass / prompt_mass if prompt_mass > 0 else float("nan")
            )
            trace_span_masses = [
                _mass(head_row, key_start=key_start, spans=[span])
                for span in encoding.trace_item_spans
            ]
            trace_mass = float(sum(trace_span_masses))
            generated_left = max(
                0, int(encoding.prompt_token_count) - int(key_start)
            )
            generated_right = min(
                int(encoding.query_position) - int(key_start),
                int(head_row.shape[-1]),
            )
            generated_prior_mass = (
                float(head_row[generated_left:max(generated_left, generated_right)].sum().item())
                if generated_right > generated_left
                else 0.0
            )
            trace_relative_mass = (
                trace_mass / generated_prior_mass
                if generated_prior_mass > 0
                else float("nan")
            )
            prompt_broad = _broad_span_metrics(span_masses)
            trace_broad = _broad_span_metrics(trace_span_masses)
            rows.append(
                {
                    "schema_version": ATTENTION_SCHEMA_VERSION,
                    "experiment_id": "answer_query_dual_broad_aggregation_attention_v4",
                    "request_id": encoding.request_id,
                    "stimulus_id": encoding.stimulus_id,
                    "model_label": encoding.model_label,
                    "seed": encoding.seed,
                    "split": encoding.split,
                    "gold_count": encoding.count,
                    "trace_one_to_one": bool(
                        parsed["parser"].get("trace_one_to_one")
                    ),
                    "trace_category": parsed["parser"].get("trace_category"),
                    "final_exact_count": bool(parsed.get("exact_count")),
                    "mechanism": "answer_execution",
                    "site_id": site_id,
                    "site_kind": str(
                        encoding.selected_site.get("site_kind", site_id)
                    ),
                    "alignment_strategy": str(active_site.alignment_strategy),
                    "baseline_endpoint_token": int(active_site.endpoint_token),
                    "legacy_answer_query_present": bool(legacy_site is not None),
                    "legacy_answer_query_same_endpoint": legacy_alias_same_endpoint,
                    "layer": int(layer),
                    "head": int(head),
                    "key_start": int(key_start),
                    "active_needle_spans": int(len(span_masses)),
                    "target_needle_raw_mass": target_mass,
                    "target_needle_relative_mass": relative_mass,
                    "target_needle_relative_denominator": "all_prompt_attention_mass",
                    "all_active_needles_raw_mass": target_mass,
                    "prompt_span_masses_json": json.dumps(span_masses),
                    "prompt_broad_coverage": prompt_broad["coverage"],
                    "prompt_broad_effective_spans": prompt_broad[
                        "effective_spans"
                    ],
                    "prompt_broad_score": prompt_broad["score"],
                    "full_prompt_mass": prompt_mass,
                    "registered_trace_item_spans": int(len(trace_span_masses)),
                    "trace_item_raw_mass": trace_mass,
                    "trace_span_masses_json": json.dumps(trace_span_masses),
                    "trace_broad_coverage": trace_broad["coverage"],
                    "trace_broad_effective_spans": trace_broad[
                        "effective_spans"
                    ],
                    "trace_broad_score": trace_broad["score"],
                    "trace_item_relative_mass": trace_relative_mass,
                    "trace_item_relative_denominator": (
                        "all_prior_generated_trace_attention_mass"
                    ),
                    "all_prior_generated_trace_attention_mass": generated_prior_mass,
                    "other_generated_raw_mass": max(
                        0.0, generated_prior_mass - trace_mass
                    ),
                    "row_sum": float(head_row.sum().item()),
                }
            )
    return pd.DataFrame(rows)


def _broad_span_metrics(
    masses: Sequence[float], *, epsilon: float = 1e-12
) -> dict[str, float]:
    """Return the frozen V4.4 broad-aggregation score for exact spans.

    This is deliberately local to the V5 capture path because prompt record
    spans and parser-registered trace item spans have different dataclasses
    from the V4 attention atlas.  The numerical definition is identical.
    """

    values = np.asarray(tuple(float(value) for value in masses), dtype=float)
    if values.size == 0:
        return {"coverage": 0.0, "effective_spans": 0.0, "score": 0.0}
    total = float(values.sum())
    if total <= float(epsilon):
        return {"coverage": 0.0, "effective_spans": 0.0, "score": 0.0}
    probabilities = values / (total + float(epsilon))
    entropy = float(
        -np.sum(probabilities * np.log(probabilities + float(epsilon)))
    )
    effective = float(math.exp(entropy))
    coverage = float(effective / len(values))
    return {
        "coverage": coverage,
        "effective_spans": effective,
        "score": float(total * coverage),
    }


def capture_trace_shards(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    records: Iterable[Mapping[str, Any]],
    *,
    config: V5Config,
    output_dir: str | Path,
    layers: Iterable[int] | None = None,
    site_kinds: Sequence[str] | None = None,
    capture_span_pooling: bool = True,
    overwrite: bool = False,
    site_ids: Sequence[str] | None = None,
) -> Path:
    output = Path(output_dir)
    index_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        request_id = str(row.get("request_id", row.get("stimulus_id", row_index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        relative = Path("shards") / safe_id
        try:
            manifest = capture_trace_record(
                model,
                adapter,
                tokenizer,
                row,
                config=config,
                output_dir=output / relative,
                layers=layers,
                site_kinds=site_kinds,
                capture_span_pooling=capture_span_pooling,
                overwrite=overwrite,
                site_ids=site_ids,
            )
        except NoAlignedTraceSitesError as error:
            parsed = parse_trace_record(row)
            parser = dict(parsed["parser"])
            exclusion_rows.append(
                {
                    "schema_version": CAPTURE_SCHEMA_VERSION,
                    "row_index": row_index,
                    "request_id": request_id,
                    "stimulus_id": parsed.get("stimulus_id", row.get("stimulus_id")),
                    "model_label": parsed.get("model_label", row.get("model_label")),
                    "seed": parsed.get("seed", row.get("seed")),
                    "split": parsed.get("split", row.get("split")),
                    "gold_count": parsed.get("gold_count"),
                    "parsed_count": parsed.get("parsed_count"),
                    "exact_count": parsed.get("exact_count"),
                    "parser_detected": bool(parser.get("detected", False)),
                    "parser_status": parser.get("status"),
                    "trace_one_to_one": bool(parser.get("trace_one_to_one", False)),
                    "trace_category": parser.get("trace_category"),
                    "reason_code": NO_ALIGNED_TRACE_SITES_REASON,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )
            print(
                f"[v5 capture] skip {row_index + 1} {request_id} "
                f"reason={NO_ALIGNED_TRACE_SITES_REASON} "
                f"parser_status={parser.get('status')}",
                flush=True,
            )
            continue
        index_rows.append(
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "row_index": row_index,
                "request_id": request_id,
                "stimulus_id": manifest["stimulus_id"],
                "model_label": manifest["model_label"],
                "entity_domain": manifest.get("entity_domain", "city"),
                "seed": manifest["seed"],
                "split": manifest["split"],
                "gold_count": manifest["gold_count"],
                "parsed_count": manifest["parsed_count"],
                "exact_count": manifest["exact_count"],
                "trace_one_to_one": manifest["parser"]["trace_one_to_one"],
                "trace_category": manifest["parser"]["trace_category"],
                "marker_kind": manifest["parser"]["marker_kind"],
                "trace_item_count": int(manifest["parser"].get("item_count", 0)),
                "sequence_source": str(manifest.get("sequence_source", "unknown")),
                "rank_supported_event_count": int(
                    manifest.get("episode_parse", {}).get(
                        "rank_supported_event_count", 0
                    )
                ),
                "raw_sequence_count": int(
                    manifest.get("episode_parse", {}).get("raw_sequence_count", 0)
                ),
                "manifest_path": (relative / "capture_manifest.json").as_posix(),
                "states_path": (relative / "states.npz").as_posix(),
            }
        )
        print(
            f"[v5 capture] {row_index + 1} {request_id} "
            f"sites={len(manifest['site_rows'])} elapsed={manifest['elapsed_seconds']:.2f}s",
            flush=True,
        )
    exclusions = output / "capture_exclusions.jsonl"
    _atomic_jsonl(exclusions, exclusion_rows)
    if not index_rows:
        raise ValueError("No eligible V5 records were captured")
    index = output / "capture_index.jsonl"
    _atomic_jsonl(index, index_rows)
    _atomic_json(
        output / "capture_manifest.json",
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "rows": len(index_rows),
            "input_rows": len(index_rows) + len(exclusion_rows),
            "excluded_rows": len(exclusion_rows),
            "exclusions_path": exclusions.name,
            "exclusion_reason_codes": sorted(
                {str(row["reason_code"]) for row in exclusion_rows}
            ),
            "parser_implementation": PARSER_IMPLEMENTATION,
            "parser_schema_version": PARSER_SCHEMA_VERSION,
            "parser_selection_rule": PARSER_SELECTION_RULE,
            "site_schema_version": SITE_SCHEMA_VERSION,
            "primary_trace_site": config.primary_trace_site,
            "registered_sites": list(config.registered_sites),
            "selected_site_kinds": sorted(
                set(config.registered_sites if site_kinds is None else site_kinds)
            ),
            "selected_site_ids": (
                None
                if site_ids is None
                else sorted({str(value) for value in site_ids})
            ),
            "capture_span_pooling": bool(capture_span_pooling),
            "counts": sorted({int(row["gold_count"]) for row in index_rows}),
            "entity_domains": sorted(
                {str(row.get("entity_domain", "city")) for row in index_rows}
            ),
            "split_trajectory_counts": {
                split: sum(str(row["split"]) == split for row in index_rows)
                for split in ("discovery", "confirmation")
            },
            "trace_item_states": sum(
                int(row["trace_item_count"]) for row in index_rows
            ),
            "sequence_source_counts": {
                source: sum(row["sequence_source"] == source for row in index_rows)
                for source in sorted({str(row["sequence_source"]) for row in index_rows})
            },
            "restartable_shards": True,
            "full_sequence_hidden_states_materialized": False,
        },
    )
    print(
        f"[v5 capture] completed captured={len(index_rows)} "
        f"excluded={len(exclusion_rows)} input={len(index_rows) + len(exclusion_rows)}",
        flush=True,
    )
    return index
