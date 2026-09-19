#!/usr/bin/env python3
"""Generate and freeze an isolated Gemma prompt-conditioned no-index cohort.

This auxiliary condition deliberately changes the task tail so that every
reasoning event uses the same literal ``FOUND:`` marker.  It is not part of the
byte-identical Native-thinking prompt cohort and must never be described as
spontaneous/natural no-index behavior.  Selection is based only on an exact
surface-format audit; hidden states and intervention outcomes are unavailable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
)
from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.generation import (  # noqa: E402
    NativePrompt,
    _chat_template_kwargs,
    generate_native_trace,
    render_native_prompt,
)
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402
from realistic_niah_v5.parsing import parse_trace_record  # noqa: E402
from realistic_niah_v5.spec import DecodingSpec  # noqa: E402


SCHEMA_VERSION = "realistic_niah_v5_gemma_prompt_conditioned_noindex_v3"
MODEL_LABEL = "Gemma4-E4B"
SELECTION_POPULATION = "gemma_prompt_conditioned_noindex_found_v3"
AUDIT_KEY = "gemma_prompt_conditioned_noindex_format_audit"
COHORT_KEY = "gemma_prompt_conditioned_noindex_cohort"

_ORIGINAL_TASK_TAIL = """How many city-score audit records are in the passage?
Reason concisely without repeating or restarting.
Stop as soon as you determine the count, then output exactly one line:
Total: <integer>"""

PROMPT_CONDITIONED_TASK_TAIL = """How many city-score audit records are in the passage?
During reasoning, write each matching record on one line using exactly:
FOUND: <city> | score <score>
Always use the same literal marker FOUND:. Do not number records, use ordinal
words, item/record/excerpt indices, mention how many have been found, or write a
running subtotal. Do not write a preamble, recap, verification, or explanation.
Immediately after the final FOUND: line, output exactly one line:
Total: <integer>
Your assistant response has already begun with the literal text 'FOUND: ';
continue that first line directly with the first city and score."""

ASSISTANT_PREFIX = "FOUND: "

_FOUND_LINE_RE = re.compile(r"(?m)^FOUND: (?P<payload>.+?)\r?$")
_TERMINAL_TOTAL_RE = re.compile(
    r"(?m)^Total:\s*(?P<total>\d+)\s*"
    r"(?:(?:<turn\|>|<end_of_turn>|<eos>)\s*)?$"
)
_ORDINAL_WORD_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)
_LABELED_INDEX_RE = re.compile(
    r"\b(?:record|match|excerpt|item|number|count)\s*#?\s*"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_RUNNING_TOTAL_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:records?|matches?|items?)\s+(?:so far|found|identified|total)\b",
    re.IGNORECASE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat(values: Any) -> tuple[int, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("Prompt-conditioned generation requires batch size one")
        values = values[0]
    return tuple(int(value) for value in values)


def _decoding(source: Mapping[str, Any]) -> DecodingSpec:
    raw = dict(source.get("decoding", {}))
    names = {field.name for field in fields(DecodingSpec)}
    return DecodingSpec(**{name: raw[name] for name in names if name in raw})


def rewrite_task_tail(user_text: str) -> str:
    """Replace only the original post-passage task tail."""

    text = str(user_text).rstrip()
    if not text.endswith(_ORIGINAL_TASK_TAIL):
        raise ValueError("Frozen native task tail was not found exactly once at the end")
    return text[: -len(_ORIGINAL_TASK_TAIL)] + PROMPT_CONDITIONED_TASK_TAIL


def prompt_from_source(
    source: Mapping[str, Any], tokenizer: Any, model_spec: Any
) -> tuple[NativePrompt, dict[str, Any]]:
    """Render the modified tail while proving passage tokens are unchanged."""

    source_row = dict(source)
    if "input_ids" not in source_row:
        rendered_source = render_native_prompt(
            source_row, tokenizer=tokenizer, model_spec=model_spec
        )
        source_row.update(
            {
                "stimulus_id": rendered_source.stimulus_id,
                "design_variant": rendered_source.design_variant,
                "seed": rendered_source.seed,
                "split": rendered_source.split,
                "gold_count": rendered_source.gold_count,
                "model_label": rendered_source.model_label,
                "model_family": rendered_source.model_family,
                "entity_domain": rendered_source.entity_domain,
                "user_text": rendered_source.user_text,
                "rendered_prompt": rendered_source.rendered_prompt,
                "input_ids": list(rendered_source.input_ids),
                "attention_mask": list(rendered_source.attention_mask),
                "gold_records": list(rendered_source.gold_records),
                "prompt_record_spans": list(rendered_source.prompt_record_spans),
            }
        )

    old_user_text = str(source_row["user_text"])
    user_text = rewrite_task_tail(old_user_text)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        **_chat_template_kwargs(model_spec),
    )
    encoded = tokenizer(rendered, add_special_tokens=False)
    base_input_ids = _flat(encoded["input_ids"])
    base_attention_mask = _flat(
        encoded.get("attention_mask", [1] * len(base_input_ids))
    )
    prefix_ids = _flat(
        tokenizer(ASSISTANT_PREFIX, add_special_tokens=False)["input_ids"]
    )
    if not prefix_ids:
        raise RuntimeError("The fixed FOUND assistant prefix tokenized empty")
    input_ids = base_input_ids + prefix_ids
    attention_mask = base_attention_mask + (1,) * len(prefix_ids)
    spans = tuple(dict(value) for value in source_row["prompt_record_spans"])
    passage_prefix_end = max(int(span["end"]) for span in spans)
    old_ids = tuple(int(value) for value in source_row["input_ids"])
    if old_ids[:passage_prefix_end] != base_input_ids[:passage_prefix_end]:
        raise RuntimeError(
            "PROMPT_INTEGRITY_FAILURE: changing the task tail changed passage tokens"
        )

    prompt = NativePrompt(
        stimulus_id=f"{source_row['stimulus_id']}_GEMMA_FOUND_NOINDEX_V1",
        design_variant=str(source_row["design_variant"]),
        seed=int(source_row["seed"]),
        split=str(source_row["split"]),
        gold_count=int(source_row["gold_count"]),
        model_label=str(source_row["model_label"]),
        model_family=str(source_row["model_family"]),
        entity_domain=str(source_row.get("entity_domain", "city")),
        user_text=user_text,
        rendered_prompt=str(rendered),
        input_ids=input_ids,
        attention_mask=attention_mask,
        gold_records=tuple(dict(value) for value in source_row["gold_records"]),
        prompt_record_spans=spans,
    )
    return prompt, {
        "status": "PASS",
        "prompt_modified": True,
        "modification_scope": "post_passage_task_tail_only",
        "extra_system_message": False,
        "extra_assistant_prefix": True,
        "assistant_prefix_text": ASSISTANT_PREFIX,
        "assistant_prefix_token_ids": list(prefix_ids),
        "assistant_prefix_contains_count_information": False,
        "base_rendered_prompt": str(rendered),
        "base_input_ids": list(base_input_ids),
        "base_attention_mask": list(base_attention_mask),
        "passage_prefix_token_count": passage_prefix_end,
        "passage_prefix_tokens_byte_identical": True,
        "original_user_text_sha256": hashlib.sha256(
            old_user_text.encode("utf-8")
        ).hexdigest(),
        "modified_user_text_sha256": hashlib.sha256(
            user_text.encode("utf-8")
        ).hexdigest(),
        "fixed_marker": "FOUND:",
        "fixed_marker_contains_count_information": False,
    }


def _gold_record_in_payload(
    payload: str,
    gold_records: Sequence[tuple[str, int]],
) -> tuple[str, int] | None:
    city_hits = [
        (city, score)
        for city, score in gold_records
        if re.search(rf"(?<!\w){re.escape(city)}(?!\w)", payload, re.IGNORECASE)
    ]
    if len(city_hits) != 1:
        return None
    city, score = city_hits[0]
    if re.search(rf"(?<!\d){int(score)}(?!\d)", payload) is None:
        return None
    return city, score


def audit_prompt_conditioned_noindex_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one score-supported event per FOUND line with no count labels."""

    raw = str(row.get("raw_output_text", ""))
    gold_records = tuple(
        (str(value["city"]), int(value["score"]))
        for value in row.get("gold_records", ())
    )
    gold_count = int(row.get("gold_count", 0))
    reasons: list[str] = []
    total_matches = list(_TERMINAL_TOTAL_RE.finditer(raw))
    if len(total_matches) != 1:
        reasons.append("terminal_total_line_not_unique_or_missing")
        total_match = None
        body = raw
        final_total = None
    else:
        total_match = total_matches[0]
        body = raw[: total_match.start()]
        final_total = int(total_match.group("total"))

    found_matches = list(_FOUND_LINE_RE.finditer(body))
    if len(found_matches) != gold_count:
        reasons.append("found_line_count_mismatch")
    residue_parts: list[str] = []
    cursor = 0
    for match in found_matches:
        residue_parts.append(body[cursor : match.start()])
        cursor = match.end()
    residue_parts.append(body[cursor:])
    if "".join(residue_parts).strip():
        reasons.append("non_found_reasoning_text")

    canonical = {(city.casefold(), score): (city, score) for city, score in gold_records}
    observed: list[tuple[str, int]] = []
    exact_payload_count = 0
    for occurrence, match in enumerate(found_matches, start=1):
        payload = str(match.group("payload")).strip()
        record = _gold_record_in_payload(payload, gold_records)
        if record is None:
            reasons.append(f"occurrence_{occurrence}:record_identification_failure")
            observed.append((f"__invalid_{occurrence}", -1))
        else:
            observed.append((record[0].casefold(), int(record[1])))
            if payload == f"{record[0]} | score {int(record[1])}":
                exact_payload_count += 1
        if _ORDINAL_WORD_RE.search(payload):
            reasons.append(f"occurrence_{occurrence}:ordinal_word")
        if _LABELED_INDEX_RE.search(payload):
            reasons.append(f"occurrence_{occurrence}:labeled_index")
        if _RUNNING_TOTAL_RE.search(payload):
            reasons.append(f"occurrence_{occurrence}:running_total")

    observed_keys = tuple(observed)
    gold_key_counts = Counter(canonical.keys())
    observed_key_counts = Counter(observed_keys)
    if observed_key_counts != gold_key_counts:
        reasons.append("found_records_not_one_to_one_with_gold")
    if len(set(observed_keys)) != len(observed_keys):
        reasons.append("duplicate_found_record")

    first_occurrences: list[dict[str, Any]] = []
    for occurrence, (match, key) in enumerate(
        zip(found_matches, observed_keys), start=1
    ):
        canonical_pair = canonical.get(key)
        first_occurrences.append(
            {
                "occurrence": occurrence,
                "city": (
                    str(match.group("payload").strip())
                    if canonical_pair is None
                    else canonical_pair[0]
                ),
                "score": int(key[1]),
                "char_start": int(match.start()),
                "char_end": int(match.end()),
                "surface_text_sha256": hashlib.sha256(
                    match.group(0).encode("utf-8")
                ).hexdigest(),
            }
        )

    format_eligible = not reasons
    return {
        "status": "PASS" if format_eligible else "FAIL",
        "primary_eligible_prompt_conditioned_noindex": format_eligible,
        # Standard first-pass fields are duplicated so the generic t* compiler
        # can consume this explicitly labeled auxiliary population.
        "primary_eligible_prefix_clean": format_eligible,
        "coverage_complete": observed_key_counts == gold_key_counts,
        "first_pass_complete": (
            observed_key_counts == gold_key_counts
            and len(set(observed_keys)) == len(observed_keys)
        ),
        "reasons": list(dict.fromkeys(reasons)),
        "gold_count": gold_count,
        "found_line_count": len(found_matches),
        "grammar_class": "literal_found_marker_score_supported_noindex",
        "marker_kind": "literal_found_marker",
        "fixed_marker": "FOUND:",
        "fixed_marker_contains_count_information": False,
        "first_occurrences": first_occurrences,
        "exact_city_pipe_score_payload_count": exact_payload_count,
        "exact_city_pipe_score_payload_rate": (
            exact_payload_count / len(found_matches) if found_matches else 0.0
        ),
        "t_star_char": (
            int(found_matches[-1].end()) if found_matches else None
        ),
        "terminal_total_present": total_match is not None,
        "terminal_total_value": final_total,
        "terminal_total_correct": final_total == gold_count,
        "terminal_total_correctness_used_for_selection": False,
        "prompt_conditioned": True,
        "spontaneous_natural_noindex_claim_allowed": False,
        "selection_uses_hidden_states": False,
        "selection_uses_patch_outcomes": False,
    }


def _source_rows(
    paths: Sequence[Path], *, seed_start: int
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_seed: dict[int, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        hashes[str(path)] = _sha256_file(path)
        for row in read_jsonl(path):
            if str(row.get("model_label")) != MODEL_LABEL:
                continue
            if int(row.get("gold_count", -1)) != 10:
                continue
            seed = int(row["seed"])
            if seed < int(seed_start):
                continue
            previous = by_seed.get(seed)
            if previous is not None:
                if str(previous.get("stimulus_id")) != str(row.get("stimulus_id")):
                    raise ValueError(f"Conflicting source stimuli for seed {seed}")
                continue
            by_seed[seed] = dict(row)
    rows = [by_seed[seed] for seed in sorted(by_seed)]
    if not rows:
        raise ValueError("No Gemma N=10 source rows were found")
    return rows, hashes


def _freeze_rows(
    eligible: Sequence[Mapping[str, Any]], *, target_eligible: int
) -> list[dict[str, Any]]:
    if len(eligible) < int(target_eligible):
        raise ValueError("Cannot freeze before the requested format quota is met")
    frozen: list[dict[str, Any]] = []
    for rank, source in enumerate(eligible[: int(target_eligible)], start=1):
        row = dict(source)
        source_split = str(row.get("split", ""))
        if int(target_eligible) == 30:
            split = "discovery" if rank <= 20 else "confirmation"
        else:
            split = "discovery"
        row["source_split_before_prompt_conditioned_freeze"] = source_split
        row["split"] = split
        row[COHORT_KEY] = {
            "schema_version": SCHEMA_VERSION,
            "selection_population": SELECTION_POPULATION,
            "rank": rank,
            "split": split,
            "prompt_conditioned": True,
            "prompt_modified": True,
            "grammar_class": "literal_found_marker_score_supported_noindex",
            "fixed_marker": "FOUND:",
            "fixed_marker_contains_count_information": False,
            "selection_independent_of_hidden_states": True,
            "selection_independent_of_patch_outcomes": True,
            "spontaneous_natural_noindex_claim_allowed": False,
        }
        frozen.append(row)
    return frozen


def _write_scan_snapshot(
    output: Path,
    scanned: Sequence[Mapping[str, Any]],
    eligible: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_jsonl(output / "scanned_generations.jsonl", scanned)
    _atomic_json(
        output / "scan_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "scanned_seed_count": len(scanned),
            "eligible_seed_count": len(eligible),
            "latest_seed": int(scanned[-1]["seed"]) if scanned else None,
            "eligible_seeds": [int(row["seed"]) for row in eligible],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--source-generations", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, default=1234)
    parser.add_argument("--target-eligible", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if int(args.target_eligible) <= 0:
        raise ValueError("target-eligible must be positive")
    sources, source_hashes = _source_rows(
        tuple(args.source_generations), seed_start=int(args.seed_start)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    shards = args.output / "generation_shards"
    shards.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_GENERATION",
        "model_label": MODEL_LABEL,
        "source_gold_count": 10,
        "seed_start": int(args.seed_start),
        "seed_order": "ascending_seed",
        "target_eligible": int(args.target_eligible),
        "formal_split": (
            {"discovery": 20, "confirmation": 10}
            if int(args.target_eligible) == 30
            else {"pilot_discovery": int(args.target_eligible)}
        ),
        "selection_rule": "earliest exact-format eligible seeds",
        "eligibility_uses_final_count_correctness": False,
        "prompt_conditioned": True,
        "prompt_modified": True,
        "prompt_tail": PROMPT_CONDITIONED_TASK_TAIL,
        "fixed_marker": "FOUND:",
        "fixed_marker_contains_count_information": False,
        "extra_system_message": False,
        "extra_assistant_prefix": True,
        "assistant_prefix_text": ASSISTANT_PREFIX,
        "assistant_prefix_contains_count_information": False,
        "spontaneous_natural_noindex_claim_allowed": False,
        "hidden_states_available_during_selection": False,
        "patch_outcomes_available_during_selection": False,
        "source_paths_sha256": source_hashes,
    }
    plan_path = args.output / "frozen_generation_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing prompt-conditioned generation plan changed")
    else:
        _atomic_json(plan_path, plan)

    # Reuse audited shards first.  Load model only if more rows are needed.
    scanned: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for source in sources:
        seed = int(source["seed"])
        shard = shards / f"seed{seed}_N10.json"
        if args.resume and shard.exists():
            row = json.loads(shard.read_text(encoding="utf-8"))
            audit = row.get(AUDIT_KEY)
            if not isinstance(audit, Mapping):
                raise ValueError(f"Resumed seed {seed} lacks the registered audit")
            scanned.append(row)
            if bool(audit.get("primary_eligible_prompt_conditioned_noindex")):
                eligible.append(row)
            if len(eligible) >= int(args.target_eligible):
                break
        else:
            pending.append(source)
            break

    if len(eligible) < int(args.target_eligible):
        # Continue from the first missing shard, including all later sources.
        if pending:
            first_pending_seed = int(pending[0]["seed"])
        elif len(scanned) < len(sources):
            first_pending_seed = int(sources[len(scanned)]["seed"])
        else:
            raise RuntimeError(
                "All registered source seeds were scanned before the format quota was met"
            )
        remaining = [row for row in sources if int(row["seed"]) >= first_pending_seed]
        spec = resolve_model_spec(MODEL_LABEL)
        model, tokenizer, _adapter = load_registered_model(
            spec,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            attention_backend=args.attention_backend,
        )
        for source in remaining:
            seed = int(source["seed"])
            shard = shards / f"seed{seed}_N10.json"
            if args.resume and shard.exists():
                row = json.loads(shard.read_text(encoding="utf-8"))
            else:
                prompt, prompt_audit = prompt_from_source(source, tokenizer, spec)
                row = generate_native_trace(
                    model,
                    tokenizer,
                    prompt,
                    decoding=_decoding(source),
                    sampling_seed=seed,
                )
                continuation_ids = tuple(int(value) for value in row["output_token_ids"])
                prefix_ids = tuple(int(value) for value in prompt_audit["assistant_prefix_token_ids"])
                output_ids = prefix_ids + continuation_ids
                row["rendered_prompt"] = str(prompt_audit["base_rendered_prompt"])
                row["input_ids"] = list(prompt_audit["base_input_ids"])
                row["attention_mask"] = list(prompt_audit["base_attention_mask"])
                row["prompt_token_count"] = len(prompt_audit["base_input_ids"])
                row["output_token_ids"] = list(output_ids)
                row["output_tokens"] = len(output_ids)
                row["raw_output_text"] = tokenizer.decode(
                    list(output_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                row["clean_output_text"] = tokenizer.decode(
                    list(output_ids),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                row["trace_parse"] = parse_trace_record(row)
                row["gemma_prompt_conditioned_noindex_prompt_audit"] = prompt_audit
                row[AUDIT_KEY] = audit_prompt_conditioned_noindex_row(row)
                _atomic_json(shard, row)
            if any(int(value["seed"]) == seed for value in scanned):
                continue
            scanned.append(row)
            audit = row[AUDIT_KEY]
            if bool(audit["primary_eligible_prompt_conditioned_noindex"]):
                eligible.append(row)
            print(
                f"[gemma-found-noindex] seed={seed} "
                f"eligible={int(bool(audit['primary_eligible_prompt_conditioned_noindex']))} "
                f"hits={len(eligible)}/{int(args.target_eligible)} "
                f"reasons={','.join(audit['reasons']) or 'none'}",
                flush=True,
            )
            _write_scan_snapshot(args.output, scanned, eligible)
            if len(eligible) >= int(args.target_eligible):
                break

    _write_scan_snapshot(args.output, scanned, eligible)
    if len(eligible) < int(args.target_eligible):
        raise RuntimeError(
            f"Source registry ended at {len(eligible)}/{int(args.target_eligible)} "
            "exact-format rows"
        )

    frozen = _freeze_rows(eligible, target_eligible=int(args.target_eligible))
    cohort_path = args.output / "frozen_cohort.jsonl"
    _atomic_jsonl(cohort_path, frozen)
    discovery = [int(row["seed"]) for row in frozen if row["split"] == "discovery"]
    confirmation = [
        int(row["seed"]) for row in frozen if row["split"] == "confirmation"
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "claim_scope": "prompt-conditioned no-index auxiliary only",
        "model_label": MODEL_LABEL,
        "selection_population": SELECTION_POPULATION,
        "grammar_class": "literal_found_marker_score_supported_noindex",
        "fixed_marker": "FOUND:",
        "fixed_marker_contains_count_information": False,
        "prompt_conditioned": True,
        "prompt_modified": True,
        "spontaneous_natural_noindex_claim_allowed": False,
        "scanned_seed_count": len(scanned),
        "format_eligible_seed_count": len(eligible),
        "format_success_rate_over_scanned": len(eligible) / len(scanned),
        "selected_seed_count": len(frozen),
        "discovery_seeds": discovery,
        "confirmation_seeds": confirmation,
        "selected_seeds": [int(row["seed"]) for row in frozen],
        "terminal_total_correct_rate_selected": sum(
            bool(row[AUDIT_KEY]["terminal_total_correct"]) for row in frozen
        )
        / len(frozen),
        "terminal_total_correctness_used_for_selection": False,
        "selection_independent_of_hidden_states": True,
        "selection_independent_of_patch_outcomes": True,
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256_file(cohort_path),
        "source_paths_sha256": source_hashes,
    }
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
