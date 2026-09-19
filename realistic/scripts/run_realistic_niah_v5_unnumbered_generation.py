#!/usr/bin/env python3
"""Materialize a frozen 20/10 no-running-index counterfactual trace panel.

The archived native generations contain explicit running indices in every
formally registered seed. Retrying generation until a model happens to obey a
format instruction would select on generated text. Instead, this script makes
the weaker controlled intervention explicit: it teacher-forces one invariant
unnumbered city/score bullet per gold record while preserving the original
native-thinking container, prompt, and formal seed split. The final channel is
standardized to ``Total: <gold>`` so an erroneous archived final-answer sentence
cannot leak a different count into the answer-query prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
)
from realistic_niah_v4.modeling import load_registered_tokenizer  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.parsing import parse_trace_record  # noqa: E402
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_unnumbered_trace,
)


def unnumbered_reasoning_text(row: Mapping[str, Any]) -> str:
    """Return the registered no-running-index reasoning intervention.

    Numeric city scores are task evidence and intentionally remain visible.
    No occurrence number, ordinal, running subtotal, or pre-answer total is
    inserted. Leading/trailing newlines keep the native reasoning delimiters.
    """

    records = tuple(row.get("gold_records", ()))
    if not records:
        raise ValueError("Cannot build an unnumbered trace without gold records")
    lines: list[str] = []
    for record in records:
        city = str(record.get("city", "")).strip()
        score = record.get("score")
        if not city or not isinstance(score, int):
            raise ValueError("Every gold record needs a city and integer score")
        lines.append(f"- {city}: score {score}")
    return "\n" + "\n".join(lines) + "\n"


def materialize_unnumbered_counterfactual(
    source: Mapping[str, Any], tokenizer: Any
) -> dict[str, Any]:
    """Replace only the native reasoning body and re-tokenize it exactly."""

    parser = source.get("trace_parse", {}).get("parser", {})
    start = int(parser.get("reasoning_start_char", -1))
    end = int(parser.get("reasoning_end_char", -1))
    raw = str(source.get("raw_output_text", ""))
    if not 0 <= start <= end <= len(raw):
        raise ValueError("Source reasoning boundaries are unavailable")
    reasoning = unnumbered_reasoning_text(source)
    model_label = str(source.get("model_label", ""))
    if model_label == "Qwen3-8B":
        final_suffix = f"</think>\n\nTotal: {int(source['gold_count'])}<|im_end|>"
    elif model_label == "Gemma4-E4B":
        final_suffix = f"<channel|>Total: {int(source['gold_count'])}<turn|>"
    else:
        raise ValueError(f"Unsupported model label: {model_label}")
    new_raw = raw[:start] + reasoning + final_suffix
    encoded = tokenizer(new_raw, add_special_tokens=False)
    output_ids = tuple(int(value) for value in encoded["input_ids"])
    output_tokens = tokenizer.convert_ids_to_tokens(list(output_ids))

    suffix = "_TF_UNNUMBERED_COUNTERFACTUAL"
    row = dict(source)
    row.update(
        {
            "schema_version": "realistic_niah_v5_teacher_forced_unnumbered_trace_v1",
            "stimulus_id": f"{source['stimulus_id']}{suffix}",
            "request_id": f"{source['request_id']}{suffix}",
            "raw_output_text": new_raw,
            "clean_output_text": new_raw,
            "output_token_ids": list(output_ids),
            "output_tokens": list(output_tokens),
            "generation_truncated": False,
            "stopped_on_eos": True,
            "elapsed_seconds": 0.0,
            "counterfactual_trace_kind": "teacher_forced_unnumbered_gold_bullets",
            "counterfactual_reasoning_only": True,
            "counterfactual_preserves_prompt": True,
            "counterfactual_preserves_native_container": True,
            "counterfactual_preserves_original_final_answer": False,
            "counterfactual_standardizes_final_answer_to_gold": True,
            "counterfactual_selected_by_patch_outcome": False,
            "counterfactual_claim_scope": (
                "controlled_no_running_index_hidden_state_sufficiency; "
                "not natural-generation prevalence"
            ),
        }
    )
    row["trace_parse"] = parse_trace_record(row)
    audit = audit_unnumbered_trace(row)
    if not audit["eligible"]:
        raise RuntimeError(
            f"Teacher-forced unnumbered trace failed audit: {audit['reasons']}"
        )
    row["unnumbered_trace_audit"] = audit
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    expected_seeds = tuple(range(1234, 1264))
    rows = [
        row
        for row in read_jsonl(args.source_generations)
        if str(row.get("model_label")) == str(args.model)
        and int(row.get("seed", -1)) in expected_seeds
        and int(row.get("gold_count", -1)) == 10
    ]
    if len(rows) != 30 or {int(row["seed"]) for row in rows} != set(expected_seeds):
        raise ValueError("Unnumbered supplement needs one frozen N=10 prompt per formal seed")
    rows.sort(key=lambda row: int(row["seed"]))
    plan = {
        "schema_version": "realistic_niah_v5_unnumbered_counterfactual_plan_v1",
        "model_label": str(args.model),
        "primary_seed_count": 30,
        "discovery_seeds": list(range(1234, 1254)),
        "confirmation_seeds": list(range(1254, 1264)),
        "source_gold_count": 10,
        "trace_intervention": "teacher_forced_unnumbered_gold_bullets",
        "final_channel_intervention": "standardized_Total_gold_without_preamble",
        "bullet_template": "- <city>: score <score>",
        "selection_rule": "one N=10 archive row per frozen seed; no retries",
        "eligibility_rule": (
            "one-to-one bullet sequence with no occurrence number, ordinal, "
            "running subtotal, or pre-answer total"
        ),
        "score_digits_retained_as_task_evidence": True,
        "natural_generation_claim_allowed": False,
        "controlled_hidden_state_sufficiency_claim_allowed": True,
        "outcome_blind": True,
        "selection_rank_used": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "frozen_counterfactual_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing unnumbered counterfactual plan changed")
    else:
        _atomic_json(plan_path, plan)

    tokenizer = load_registered_tokenizer(
        resolve_model_spec(str(args.model)), cache_dir=args.cache_dir
    )
    shard_dir = args.output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        seed = int(source["seed"])
        selected_path = shard_dir / f"seed{seed}.json"
        if args.resume and selected_path.exists():
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            if not selected["unnumbered_trace_audit"]["eligible"]:
                raise ValueError("A resumed counterfactual shard is not eligible")
        else:
            selected = materialize_unnumbered_counterfactual(source, tokenizer)
            _atomic_json(selected_path, selected)
        selected_rows.append(selected)
        print(f"[unnumbered-counterfactual] {index}/30 seed={seed}", flush=True)
    _atomic_jsonl(args.output / "selected_generations.jsonl", selected_rows)
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_unnumbered_counterfactual_manifest_v1",
            "status": "PASS",
            "model_label": str(args.model),
            "row_count": len(selected_rows),
            "discovery_seed_count": 20,
            "confirmation_seed_count": 10,
            "all_rows_format_eligible": all(
                row["unnumbered_trace_audit"]["eligible"] for row in selected_rows
            ),
            "counterfactual_trace_kind": "teacher_forced_unnumbered_gold_bullets",
            "natural_generation_claim_allowed": False,
            "controlled_hidden_state_sufficiency_claim_allowed": True,
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )


if __name__ == "__main__":
    main()
