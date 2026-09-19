from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .causal_sites import build_output_token_map
from .parsing import output_token_ids, prompt_token_ids, raw_output_text


TSTAR_PREFIX_SCHEMA = "realistic_niah_v5_tstar_first_pass_context_v2"


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prompt_attention_mask(row: Mapping[str, Any], prompt_length: int) -> tuple[int, ...]:
    raw_mask = row.get("attention_mask")
    if raw_mask is None:
        return (1,) * int(prompt_length)
    mask = tuple(int(value) for value in raw_mask)
    if len(mask) != int(prompt_length):
        raise ValueError("Prompt attention mask length does not match prompt input IDs")
    if any(value not in {0, 1} for value in mask):
        raise ValueError("Prompt attention mask must be binary")
    return mask


def _token_end_covering_char(
    offsets: Sequence[tuple[int, int]], *, char_end: int
) -> int:
    """Return the smallest exclusive token end whose decoded text covers char_end."""

    endpoint = int(char_end)
    if endpoint <= 0:
        raise ValueError("t_star_char must be positive")
    hits = [
        index
        for index, (left, right) in enumerate(offsets)
        if int(right) > int(left) and int(left) < endpoint
    ]
    if not hits:
        raise ValueError("No output token covers the requested t_star_char")
    token_end = hits[-1] + 1
    if int(offsets[token_end - 1][1]) < endpoint:
        raise ValueError("Output-token prefix does not reach t_star_char")
    return token_end


def build_tstar_prefix_context(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    audit_key: str,
    cohort_key: str,
    cohort_split: str | None = None,
    eligibility_field: str = "primary_eligible_prefix_clean",
) -> dict[str, Any]:
    """Compile one frozen generation into an exact first-pass early-stop context.

    The source generation is not changed or regenerated.  The returned model
    input is the archived prompt followed by the smallest whole-token output
    prefix covering ``t_star_char``.  Consequently, text generated after the
    first complete evidence pass cannot enter any downstream causal analysis.
    """

    audit_value = row.get(audit_key)
    if not isinstance(audit_value, Mapping):
        raise ValueError(f"Missing format audit: {audit_key}")
    audit = dict(audit_value)
    if not bool(audit.get(str(eligibility_field))):
        raise ValueError("Row is not first-occurrence prefix-clean eligible")
    if not bool(audit.get("coverage_complete")):
        raise ValueError("Row does not contain a complete first evidence pass")
    if not bool(audit.get("first_pass_complete")):
        raise ValueError("Row repeats score-supported evidence before t_star")
    t_star_value = audit.get("t_star_char")
    if t_star_value is None:
        raise ValueError("Prefix-clean audit has no t_star_char")
    t_star_char = int(t_star_value)

    raw = raw_output_text(row)
    if not 0 < t_star_char <= len(raw):
        raise ValueError("t_star_char is outside the archived output")
    token_map = build_output_token_map(row, tokenizer)
    output_token_end = _token_end_covering_char(
        token_map.offsets, char_end=t_star_char
    )
    stop_char_end = int(token_map.offsets[output_token_end - 1][1])
    prefix_ids = tuple(int(value) for value in token_map.token_ids[:output_token_end])
    raw_prefix = raw[:stop_char_end]
    if token_map.decode(prefix_ids) != raw_prefix:
        raise ValueError("Decoded early-stop tokens are not an exact raw-text prefix")
    if tuple(output_token_ids(row)[:output_token_end]) != prefix_ids:
        raise ValueError("Early-stop IDs are not a strict prefix of archived output IDs")

    occurrence_rows = list(audit.get("first_occurrences") or ())
    gold_count = int(row.get("gold_count", 0))
    if len(occurrence_rows) != gold_count or gold_count <= 0:
        raise ValueError("First-occurrence registry does not match gold_count")
    token_occurrences: list[dict[str, Any]] = []
    previous_end = -1
    for value in occurrence_rows:
        occurrence = dict(value)
        char_start = int(occurrence["char_start"])
        char_end = int(occurrence["char_end"])
        span = token_map.span(
            f"first_occurrence_{int(occurrence['occurrence'])}",
            char_start,
            char_end,
        )
        if span.get("status") != "ok":
            raise ValueError("Cannot align one first-occurrence evidence span")
        token_start = int(span["output_token_start"])
        token_end = int(span["output_token_end"])
        if token_start < previous_end or token_end > output_token_end:
            raise ValueError("First-occurrence token spans are nonmonotone or post-stop")
        previous_end = token_end
        token_occurrences.append(
            {
                "occurrence": int(occurrence["occurrence"]),
                "city": str(occurrence["city"]),
                "char_start": char_start,
                "char_end": char_end,
                "output_token_start": token_start,
                "output_token_end": token_end,
                "offset_char_start": int(span["offset_char_start"]),
                "offset_char_end": int(span["offset_char_end"]),
            }
        )

    cohort_value = row.get(cohort_key)
    if not isinstance(cohort_value, Mapping):
        raise ValueError(f"Missing frozen cohort metadata: {cohort_key}")
    registered_split = str(cohort_value.get("split", ""))
    split = registered_split if cohort_split is None else str(cohort_split)
    if split not in {"discovery", "confirmation"}:
        raise ValueError("Cohort split must be discovery or confirmation")
    if registered_split != split:
        raise ValueError("Manifest split disagrees with nested cohort split")

    prompt_ids = tuple(int(value) for value in prompt_token_ids(row))
    prompt_mask = _prompt_attention_mask(row, len(prompt_ids))
    model_input_ids = prompt_ids + prefix_ids
    attention_mask = prompt_mask + (1,) * len(prefix_ids)
    if len(model_input_ids) != len(attention_mask):
        raise AssertionError("Compiled input and attention mask lengths differ")

    spill_text = raw[t_star_char:stop_char_end]
    full_output_ids = output_token_ids(row)
    return {
        "schema_version": TSTAR_PREFIX_SCHEMA,
        "status": "PASS",
        "model_label": str(row.get("model_label", "")),
        "model_family": str(row.get("model_family", "")),
        "seed": int(row["seed"]),
        "split": split,
        "fixed_count": gold_count,
        "request_id": str(row.get("request_id", "")),
        "stimulus_id": str(row.get("stimulus_id", "")),
        "source_schema_version": str(row.get("schema_version", "")),
        "source_row_sha256": sha256_json(dict(row)),
        "source_raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "source_output_token_ids_sha256": sha256_json(full_output_ids),
        "stopping_rule": (
            "smallest whole-output-token prefix covering the end of the K-th "
            "unique locally score-supported gold-record first occurrence"
        ),
        "selection_population": str(
            cohort_value.get("selection_population", "first_pass_noindex_enumeration")
        ),
        "selection_used_final_answer": False,
        "mechanism_outcomes_accessed": False,
        "future_recap_available_to_context": False,
        "t_star_char": t_star_char,
        "stop_char_end": stop_char_end,
        "token_boundary_right_spill_chars": stop_char_end - t_star_char,
        "token_boundary_right_spill_text": spill_text,
        "output_token_end": output_token_end,
        "output_prefix_token_count": len(prefix_ids),
        "full_output_token_count": len(full_output_ids),
        "removed_output_token_count": len(full_output_ids) - len(prefix_ids),
        "removed_output_char_count": len(raw) - stop_char_end,
        "prompt_token_count": len(prompt_ids),
        "sequence_token_count": len(model_input_ids),
        "query_position": len(model_input_ids) - 1,
        "raw_prefix_text": raw_prefix,
        "output_prefix_token_ids": list(prefix_ids),
        "input_ids": list(model_input_ids),
        "attention_mask": list(attention_mask),
        "first_occurrences": token_occurrences,
        "gold_records": [dict(value) for value in row.get("gold_records", ())],
        "prompt_record_spans": [
            dict(value) for value in row.get("prompt_record_spans", ())
        ],
    }
