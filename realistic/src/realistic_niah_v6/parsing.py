from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from realistic_niah.parsing import evaluate_generation
from realistic_niah_v5.parsing import (
    EPISODE_SCHEMA_VERSION,
    EPISODE_SELECTION_POLICY,
    TraceCharSite,
    TraceTokenSite,
    align_trace_sites,
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    output_token_ids,
    parse_hybrid_trace,
    prompt_token_ids,
    raw_output_text,
    trace_char_sites,
)

from .spec import PROMPT_MODES


PARSER_SCHEMA_VERSION = "realistic_niah_v6_structured_enumeration_parser_v1"
SITE_SCHEMA_VERSION = "realistic_niah_v6_structured_enumeration_sites_v1"
PARSER_IMPLEMENTATION = "realistic_niah_v6.parse_structured_enumeration_trace"
PARSER_SELECTION_RULE = (
    "Parse the model output under its registered enumeration grammar, retain "
    "the V5 hybrid parser's exact semantic item spans, and require strict "
    "marker syntax, contiguous index labels when applicable, exact city-score "
    "pairs in passage order, a matching final Total, and no extra text for "
    "the formal causal cohort. Gold N and final Total never construct or pad "
    "the item sequence."
)
PARSER_FILE_SHA256 = {
    "parsing.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}


def _prompt_mode(row: Mapping[str, Any]) -> str:
    mode = str(row.get("prompt_mode", ""))
    if mode not in PROMPT_MODES:
        raise ValueError(
            f"V6 row prompt_mode must be one of {PROMPT_MODES}, got {mode!r}"
        )
    return mode


def parse_trace_record(
    row: Mapping[str, Any], *, model_family: str | None = None
) -> dict[str, Any]:
    mode = _prompt_mode(row)
    family = infer_model_family(row, model_family)
    raw = raw_output_text(row)
    gold = gold_records(row)
    parser, episode_parse = parse_hybrid_trace(
        raw,
        model_family=family,
        gold_records=gold,
    )
    finish_reason = str(
        row.get(
            "finish_reason",
            "length" if bool(row.get("generation_truncated")) else "stop",
        )
    )
    evaluation = evaluate_generation(
        raw,
        prompt_mode=mode,
        reasoning_expected=False,
        gold_pairs=[dict(value) for value in gold],
        finish_reason=finish_reason,
        output_tokens=(
            int(row["output_tokens"]) if row.get("output_tokens") is not None else None
        ),
        max_output_tokens=(
            int(row.get("decoding", {}).get("max_new_tokens"))
            if isinstance(row.get("decoding"), Mapping)
            and row.get("decoding", {}).get("max_new_tokens") is not None
            else None
        ),
    )
    listed = [
        (str(value["city"]), int(value["score"]))
        for value in evaluation["strict_listed_records"]
    ]
    gold_pairs = [(str(value["city"]), int(value["score"])) for value in gold]
    expected_marker = "indexed" if mode == "enumeration_index" else "bullet"
    marker_ok = str(parser.marker_kind) == expected_marker
    exact_ordered_pairs = listed == gold_pairs
    parser_forward = bool(
        parser.trace_one_to_one and parser.trace_order_class == "forward"
    )
    strict_causal_eligible = bool(
        evaluation["registered_success"]
        and evaluation["enumeration_format_compliant"]
        and evaluation["strict_listed_total_matches_length"]
        and exact_ordered_pairs
        and marker_ok
        and parser_forward
        and int(parser.item_count) == len(gold_pairs)
    )
    parser_payload = parser.to_dict()
    parser_payload["strict_causal_eligible"] = strict_causal_eligible
    parser_payload["enumeration_format_compliant"] = bool(
        evaluation["enumeration_format_compliant"]
    )
    parser_payload["exact_ordered_gold_pairs"] = exact_ordered_pairs
    return {
        "schema_version": PARSER_SCHEMA_VERSION,
        "site_schema_version": SITE_SCHEMA_VERSION,
        "parser_implementation": PARSER_IMPLEMENTATION,
        "parser_selection_rule": PARSER_SELECTION_RULE,
        "parser_file_sha256": dict(PARSER_FILE_SHA256),
        "request_id": row.get("request_id", row.get("stimulus_id")),
        "stimulus_id": row.get("stimulus_id"),
        "model_label": row.get("model_label", row.get("model")),
        "model_family": family,
        "prompt_mode": mode,
        "seed": row.get("seed"),
        "split": row.get("split"),
        "gold_count": len(gold_pairs),
        "parsed_count": evaluation["predicted_count"],
        "exact_count": bool(evaluation["exact_count"]),
        "reasoning_text": evaluation["reasoning_text"],
        "final_text": evaluation["final_text"],
        "response_format_compliant": bool(
            evaluation["response_format_compliant"]
        ),
        "enumeration_format_status": evaluation["enumeration_format_status"],
        "enumeration_format_compliant": bool(
            evaluation["enumeration_format_compliant"]
        ),
        "listed_total_matches_length": evaluation[
            "strict_listed_total_matches_length"
        ],
        "listed_records": evaluation["strict_listed_records"],
        "exact_ordered_gold_pairs": exact_ordered_pairs,
        "expected_marker_kind": expected_marker,
        "marker_kind_compliant": marker_ok,
        "parser_forward_one_to_one": parser_forward,
        "strict_causal_eligible": strict_causal_eligible,
        "sequence_source": episode_parse["sequence_source"],
        "rank_episode_schema_version": EPISODE_SCHEMA_VERSION,
        "rank_episode_selection_policy": EPISODE_SELECTION_POLICY,
        "episode_parse": episode_parse,
        "parser": parser_payload,
        "char_sites": [site.to_dict() for site in trace_char_sites(raw, parser)],
        "generation_eval": evaluation,
    }


def parse_and_align_record(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    model_family: str | None = None,
    include_token_ids: bool = False,
) -> dict[str, Any]:
    parsed = parse_trace_record(row, model_family=model_family)
    sites = [TraceCharSite(**value) for value in parsed["char_sites"]]
    aligned = align_trace_sites(
        tokenizer,
        raw_text=raw_output_text(row),
        baseline_output_token_ids=output_token_ids(row),
        sites=sites,
    )
    token_sites = [
        site.to_dict(include_token_ids=include_token_ids) for site in aligned
    ]
    return {
        **parsed,
        "token_sites": token_sites,
        "alignment_summary": {
            "total": len(token_sites),
            "eligible": sum(bool(site["alignment_eligible"]) for site in token_sites),
            "ineligible": sum(
                not bool(site["alignment_eligible"]) for site in token_sites
            ),
        },
    }


def formal_cohort_eligible(row: Mapping[str, Any]) -> bool:
    parsed = row.get("trace_parse")
    if not isinstance(parsed, Mapping) or parsed.get("schema_version") != PARSER_SCHEMA_VERSION:
        parsed = parse_trace_record(row)
    return bool(parsed.get("strict_causal_eligible"))


__all__ = [
    "PARSER_FILE_SHA256",
    "PARSER_IMPLEMENTATION",
    "PARSER_SCHEMA_VERSION",
    "PARSER_SELECTION_RULE",
    "SITE_SCHEMA_VERSION",
    "TraceCharSite",
    "TraceTokenSite",
    "align_trace_sites",
    "find_trace_count_sequence",
    "formal_cohort_eligible",
    "gold_records",
    "infer_model_family",
    "output_token_ids",
    "parse_and_align_record",
    "parse_trace_record",
    "prompt_token_ids",
    "raw_output_text",
    "trace_char_sites",
]
