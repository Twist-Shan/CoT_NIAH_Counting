#!/usr/bin/env python3
"""Historical prompt-conditioned no-record-enumeration generator.

This entry point is deliberately disabled.  It appended a trace-format
instruction to ``user_text`` and, for the final attempt set, an assistant
``"- "`` prefix.  Those interventions violate the frozen Native-thinking
prompt contract and must not be used for formal mechanism evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_count_stream import _atomic_json, _atomic_jsonl  # noqa: E402
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
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_no_count_enumeration_trace,
    audit_qwen_thinking_bullets_final_total,
    inject_unnumbered_instruction,
)


_ORIGINAL_NATIVE_TASK_TAIL = """How many city-score audit records are in the passage?
Reason concisely without repeating or restarting.
Stop as soon as you determine the count, then output exactly one line:
Total: <integer>"""

_QWEN_EXTRACTION_ANSWER_TAILS = (
    """How many city-score audit records are in the passage?
This response has two jobs, performed in different channels.
THINKING JOB — EXTRACT ONLY: do not answer the question and do not determine,
calculate, or state how many records there are. Copy each matching record exactly
once as an unnumbered '- <city>: <score>' bullet, with no other thinking text.
When extraction is complete, end thinking immediately after the last bullet.
FINAL JOB — COUNT: only in the final answer, count the extracted bullet lines and
output exactly one line 'Total: <integer>'. Do not repeat any bullet or evidence.""",
    """How many city-score audit records are in the passage?
During thinking, act only as an evidence extractor, not as a counter. Emit exactly
one unnumbered '- <city>: <score>' bullet per matching record. No thinking line may
be an answer, count, subtotal, summary, verification, or Total line. End the
thinking channel directly after the last extracted record. In the final answer
channel, act as the counter and output only 'Total: <integer>' on one line. Never
copy the evidence list into the final answer.""",
    """How many city-score audit records are in the passage?
Use this channel-local grammar. THINKING permits only unnumbered lines shaped
'- <city>: <score>', one per match. The word Total and any standalone count are
illegal in THINKING; finishing the extraction means ending THINKING immediately.
FINAL permits exactly one line shaped 'Total: <integer>' and then end-of-response.
Bullets, cities, scores, explanations, and recaps are illegal in FINAL.""",
)


def _rewrite_qwen_extraction_answer_task(user_text: str, *, attempt: int) -> str:
    """Replace only the post-passage task tail for the prompt-only Qwen pilot."""

    index = int(attempt) - 16
    if not 0 <= index < len(_QWEN_EXTRACTION_ANSWER_TAILS):
        raise ValueError("Qwen extraction-answer attempt is outside the registry")
    text = str(user_text).rstrip()
    if not text.endswith(_ORIGINAL_NATIVE_TASK_TAIL):
        raise RuntimeError("Frozen native task tail was not found for prompt rewrite")
    return text[: -len(_ORIGINAL_NATIVE_TASK_TAIL)] + _QWEN_EXTRACTION_ANSWER_TAILS[index]


def _flat(values: Any) -> tuple[int, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("Natural unnumbered generation requires batch size one")
        values = values[0]
    return tuple(int(value) for value in values)


def _decoding(source: Mapping[str, Any]) -> DecodingSpec:
    raw = dict(source.get("decoding", {}))
    names = {field.name for field in fields(DecodingSpec)}
    return DecodingSpec(**{name: raw[name] for name in names if name in raw})


def _prompt_from_source(
    source: Mapping[str, Any], tokenizer: Any, model_spec: Any, *, attempt: int
) -> tuple[NativePrompt, dict[str, Any]]:
    if "input_ids" not in source:
        rendered_source = render_native_prompt(
            source, tokenizer=tokenizer, model_spec=model_spec
        )
        source = {
            **dict(source),
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
    assistant_prefix = 7 <= int(attempt) <= 18
    if 16 <= int(attempt) <= 18:
        user_text = _rewrite_qwen_extraction_answer_task(
            str(source["user_text"]), attempt=int(attempt)
        )
    elif 19 <= int(attempt) <= 21:
        user_text = _rewrite_qwen_extraction_answer_task(
            str(source["user_text"]), attempt=int(attempt) - 3
        )
    else:
        instruction_attempt = int(attempt) - 3 if assistant_prefix else int(attempt)
        user_text = inject_unnumbered_instruction(
            str(source["user_text"]), attempt=instruction_attempt
        )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        **_chat_template_kwargs(model_spec),
    )
    encoded = tokenizer(rendered, add_special_tokens=False)
    base_input_ids = _flat(encoded["input_ids"])
    base_attention_mask = _flat(
        encoded.get("attention_mask", [1] * len(base_input_ids))
    )
    prefix_ids = (
        _flat(tokenizer("- ", add_special_tokens=False)["input_ids"])
        if assistant_prefix
        else ()
    )
    if assistant_prefix and not prefix_ids:
        raise RuntimeError("The registered reasoning bullet prefix tokenized empty")
    input_ids = base_input_ids + prefix_ids
    attention_mask = base_attention_mask + tuple(1 for _ in prefix_ids)
    spans = tuple(dict(value) for value in source["prompt_record_spans"])
    prefix_end = max(int(span["end"]) for span in spans)
    old_ids = tuple(int(value) for value in source["input_ids"])
    if old_ids[:prefix_end] != base_input_ids[:prefix_end]:
        raise RuntimeError("Post-passage instruction changed frozen prompt-record tokens")
    prompt = NativePrompt(
        stimulus_id=f"{source['stimulus_id']}_NATURAL_UNNUMBERED_A{attempt}",
        design_variant=str(source["design_variant"]),
        seed=int(source["seed"]),
        split=str(source["split"]),
        gold_count=int(source["gold_count"]),
        model_label=str(source["model_label"]),
        model_family=str(source["model_family"]),
        entity_domain=str(source.get("entity_domain", "city")),
        user_text=user_text,
        rendered_prompt=str(rendered),
        input_ids=input_ids,
        attention_mask=attention_mask,
        gold_records=tuple(dict(value) for value in source["gold_records"]),
        prompt_record_spans=spans,
    )
    return prompt, {
        "base_rendered_prompt": str(rendered),
        "base_input_ids": base_input_ids,
        "base_attention_mask": base_attention_mask,
        "assistant_prefix_ids": prefix_ids,
        "assistant_prefix_text": "- " if assistant_prefix else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-generations", type=Path)
    source.add_argument("--source-stimuli", type=Path)
    parser.add_argument("--source-counts", type=int, nargs="+", default=(10,))
    parser.add_argument("--planned-seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument(
        "--attempt-set",
        choices=(
            "after_passage",
            "end_priority",
            "reasoning_bullet_prefix",
            "a7_only",
            "qwen_channel_contract",
            "qwen_channel_boundary",
            "qwen_extraction_answer",
            "qwen_extraction_answer_no_prefill",
        ),
        default="reasoning_bullet_prefix",
    )
    parser.add_argument("--allow-prompt-conditioned-a7-auxiliary", action="store_true")
    parser.add_argument("--allow-prompt-conditioned-auxiliary", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if not (
        args.allow_prompt_conditioned_a7_auxiliary
        or args.allow_prompt_conditioned_auxiliary
    ):
        raise RuntimeError(
            "DISABLED_PROMPT_MODIFICATION: this historical generator changes "
            "user_text and/or the assistant prefix. Formal Native-thinking "
            "experiments must use the byte-identical frozen prompt."
        )
    if args.attempt_set not in {
        "a7_only",
        "qwen_channel_contract",
        "qwen_channel_boundary",
        "qwen_extraction_answer",
        "qwen_extraction_answer_no_prefill",
    }:
        raise ValueError(
            "The auxiliary prompt-conditioning override is restricted to registered pilot sets"
        )

    expected_seeds = (
        tuple(int(value) for value in args.planned_seeds)
        if args.planned_seeds
        else tuple(range(1234, 1264))
    )
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Planned natural-generation seeds must be nonempty and unique")
    source_counts = tuple(int(value) for value in args.source_counts)
    if not source_counts or len(set(source_counts)) != len(source_counts):
        raise ValueError("Natural source-count order must be non-empty and unique")
    if min(source_counts) < 9 or max(source_counts) > 10:
        raise ValueError("Natural restore sources must be N=9 or N=10")
    source_path = args.source_generations or args.source_stimuli
    source_kind = "registered_generation" if args.source_generations else "frozen_stimulus"
    rows = [
        row
        for row in read_jsonl(source_path)
        if (
            args.source_stimuli
            or str(row.get("model_label")) == str(args.model)
        )
        and int(row.get("seed", -1)) in expected_seeds
        and int(row.get("gold_count", -1)) in set(source_counts)
    ]
    count_rank = {count: rank for rank, count in enumerate(source_counts)}
    rows.sort(key=lambda row: (int(row["seed"]), count_rank[int(row["gold_count"])]))
    expected_pairs = {(seed, count) for seed in expected_seeds for count in source_counts}
    actual_pairs = {(int(row["seed"]), int(row["gold_count"])) for row in rows}
    if actual_pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise ValueError("Natural unnumbered generation source seed/count panel changed")
    rows_by_seed = {
        seed: [row for row in rows if int(row["seed"]) == seed]
        for seed in expected_seeds
    }
    if args.max_seeds is not None:
        planned_seeds = expected_seeds[: int(args.max_seeds)]
    else:
        planned_seeds = expected_seeds

    attempt_orders = {
        "after_passage": [1, 2, 3],
        "end_priority": [4, 5, 6],
        "reasoning_bullet_prefix": [7, 8, 9],
        "a7_only": [7],
        "qwen_channel_contract": [10, 11, 12],
        "qwen_channel_boundary": [13, 14, 15],
        "qwen_extraction_answer": [16, 17, 18],
        "qwen_extraction_answer_no_prefill": [19, 20, 21],
    }
    attempt_order = attempt_orders[str(args.attempt_set)]
    plan = {
        "schema_version": "realistic_niah_v5_natural_unnumbered_generation_plan_v5",
        "model_label": str(args.model),
        "source_kind": source_kind,
        "source_path": str(source_path),
        "source_count_order": list(source_counts),
        "planned_seeds": list(planned_seeds),
        "attempt_set": str(args.attempt_set),
        "attempt_order": attempt_order,
        "fixed_reasoning_prefix": (
            "- "
            if args.attempt_set
            in {
                "reasoning_bullet_prefix",
                "a7_only",
                "qwen_channel_contract",
                "qwen_channel_boundary",
                "qwen_extraction_answer",
            }
            else ""
        ),
        "fixed_reasoning_prefix_contains_count_information": False,
        "selection_rule": "first causal-prefix format-eligible attempt; patch outcomes unavailable",
        "eligibility_rule": (
            "plain bullets allowed; no explicit record-number enumeration or already-stated "
            "count in any registered item causal prefix"
        ),
        "trace_tokens_model_generated": True,
        "teacher_forcing": False,
        "outcome_blind": True,
        "selection_rank_used": False,
        "prompt_conditioned_a7_auxiliary": True,
        "formal_frozen_prompt_claim_allowed": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "frozen_generation_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing natural unnumbered generation plan changed")
    else:
        _atomic_json(plan_path, plan)

    spec = resolve_model_spec(str(args.model))
    model, tokenizer, _adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    shard_dir = args.output / "attempts"
    shard_dir.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    seed_audits: list[dict[str, Any]] = []
    for seed in planned_seeds:
        selected: dict[str, Any] | None = None
        attempts = []
        for source in rows_by_seed[int(seed)]:
            source_count = int(source["gold_count"])
            for attempt in attempt_order:
                path = shard_dir / f"seed{seed}_N{source_count}_attempt{attempt}.json"
                if args.resume and path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                else:
                    prompt, prompt_control = _prompt_from_source(
                        source, tokenizer, spec, attempt=attempt
                    )
                    row = generate_native_trace(
                        model,
                        tokenizer,
                        prompt,
                        decoding=_decoding(source),
                        sampling_seed=int(seed),
                    )
                    prefix_ids = tuple(prompt_control["assistant_prefix_ids"])
                    if prefix_ids:
                        continuation_ids = tuple(
                            int(value) for value in row["output_token_ids"]
                        )
                        output_ids = prefix_ids + continuation_ids
                        row["rendered_prompt"] = prompt_control["base_rendered_prompt"]
                        row["input_ids"] = list(prompt_control["base_input_ids"])
                        row["attention_mask"] = list(
                            prompt_control["base_attention_mask"]
                        )
                        row["prompt_token_count"] = len(
                            prompt_control["base_input_ids"]
                        )
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
                    row["natural_unnumbered_attempt"] = int(attempt)
                    row["natural_unnumbered_teacher_forced"] = False
                    row["natural_unnumbered_fixed_reasoning_prefix"] = str(
                        prompt_control["assistant_prefix_text"]
                    )
                    row[
                        "natural_unnumbered_fixed_prefix_contains_count_information"
                    ] = False
                    row["no_count_enumeration_audit"] = (
                        audit_qwen_thinking_bullets_final_total(row)
                        if args.attempt_set
                        in {
                            "qwen_channel_contract",
                            "qwen_channel_boundary",
                            "qwen_extraction_answer",
                            "qwen_extraction_answer_no_prefill",
                        }
                        else audit_no_count_enumeration_trace(row)
                    )
                    _atomic_json(path, row)
                audit = row["no_count_enumeration_audit"]
                attempts.append(
                    {
                        "gold_count": source_count,
                        "attempt": attempt,
                        "eligible": bool(audit["eligible"]),
                        "marker_kind": str(audit["marker_kind"]),
                        "reasons": list(audit["reasons"]),
                    }
                )
                if bool(audit["eligible"]):
                    selected = row
                    break
            if selected is not None:
                break
        seed_audits.append({"seed": seed, "attempts": attempts, "eligible": selected is not None})
        if selected is not None:
            selected_rows.append(selected)
        print(
            f"[natural-unnumbered] seed={seed} eligible={selected is not None} "
            f"attempts={len(attempts)}",
            flush=True,
        )

    _atomic_jsonl(args.output / "selected_generations.jsonl", selected_rows)
    eligible_seeds = {int(row["seed"]) for row in selected_rows}
    eligible_discovery_seeds = {
        int(row["seed"])
        for row in selected_rows
        if str(row.get("split")) == "discovery"
    }
    eligible_confirmation_seeds = {
        int(row["seed"])
        for row in selected_rows
        if str(row.get("split")) == "confirmation"
    }
    complete = eligible_seeds == set(planned_seeds)
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_natural_unnumbered_manifest_v5",
            "status": "PASS" if complete else "INCOMPLETE_FORMAT_COHORT",
            "model_label": str(args.model),
            "source_kind": source_kind,
            "planned_seed_count": len(planned_seeds),
            "source_count_order": list(source_counts),
            "eligible_seed_count": len(eligible_seeds),
            "eligible_discovery_seed_count": len(eligible_discovery_seeds),
            "eligible_confirmation_seed_count": len(eligible_confirmation_seeds),
            "trace_tokens_model_generated": True,
            "teacher_forcing": False,
            "attempt_set": str(args.attempt_set),
            "fixed_reasoning_prefix": (
                "- "
                if args.attempt_set
                in {
                    "reasoning_bullet_prefix",
                    "a7_only",
                    "qwen_channel_contract",
                    "qwen_channel_boundary",
                    "qwen_extraction_answer",
                }
                else ""
            ),
            "fixed_reasoning_prefix_contains_count_information": False,
            "plain_bullets_allowed": True,
            "outcome_blind": True,
            "selection_rank_used": False,
            "prompt_conditioned_a7_auxiliary": True,
            "formal_frozen_prompt_claim_allowed": False,
            "seed_audits": seed_audits,
        },
    )
    if args.require_complete and not complete:
        raise RuntimeError("Natural no-enumeration format cohort is incomplete")


if __name__ == "__main__":
    main()
