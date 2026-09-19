"""Shared prompt rendering and generation scoring for Appendix H.

The historical pilot runner is excluded from the paper release. Keep this module
name because task preparation snapshots and natural-output stages import it.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "src"))

from diagnostics import strip_end_tokens
from protocol import encode_ids, parse_answer, text_hash, user_prompt
from realistic_niah.parsing import split_reasoning_and_final


def render(case: dict, mode: str, tokenizer):
    user = user_prompt(case, mode)
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": mode == "native_thinking"}
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": user}], **kwargs)
    prefill = case["answer_prefix"] if mode == "nonthinking" else ""
    rendered += prefill
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if rendered.count(case["passage"]) != 1:
        raise ValueError("Rendered passage not unique")
    return encode_ids(encoded["input_ids"]), encoded["offset_mapping"], {
        "user_text": user, "rendered_prompt": rendered,
        "user_text_sha256": text_hash(user), "rendered_prompt_sha256": text_hash(rendered),
        "chat_template_kwargs": kwargs, "assistant_prefill_text": prefill,
        "input_ids": encoded["input_ids"], "attention_mask": [1] * len(encoded["input_ids"]),
    }


def summarize_generation(result: dict, case: dict, *, mode: str, prefixed: bool = False) -> dict:
    raw = result["completion_text_raw"]
    if prefixed:
        final, reasoning = case["answer_prefix"] + strip_end_tokens(raw), ""
    else:
        reasoning, final = split_reasoning_and_final(raw, prompt_mode="native_thinking", reasoning_expected=True)
        final = strip_end_tokens(final)
    parsed = parse_answer(final, case)
    ordinal = None
    if case["task"] == "kth_needle" and parsed["prediction"]:
        ordinal = next((r["ordinal"] for r in case["records"]
                        if parsed["prediction"] == f"{r['city']}|{r['score']}"), None)
    numeric = ordinal if case["task"] == "kth_needle" else (
        int(parsed["prediction"]) if parsed["prediction"] is not None else None)
    return {**parsed, "final_text": final, "reasoning_text": reasoning,
            "predicted_ordinal_or_count": numeric,
            "absolute_error": abs(numeric - case["level"]) if numeric is not None else None,
            "generation_truncated": result["generation_truncated"],
            "generated_token_count": result["generated_token_count"]}
