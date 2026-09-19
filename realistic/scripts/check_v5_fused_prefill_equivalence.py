#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_model
from realistic_niah_v4.spec import resolve_model_spec
from realistic_niah_v5.causal import (
    _scheduled_head_ablation_logits,
    generate_with_head_ablation_at_positions,
)
from realistic_niah_v5.encoding import build_native_trace_encoding
from realistic_niah_v5.parsing import parse_trace_record
from realistic_niah_v5.pipeline import read_jsonl


def select_row(path: Path, model_label: str) -> dict[str, Any]:
    for row in read_jsonl(path):
        if str(row.get("model_label", row.get("model"))) != model_label:
            continue
        parsed = parse_trace_record(row)
        if (
            str(row.get("split")) == "confirmation"
            and bool(parsed["parser"].get("trace_one_to_one"))
            and bool(parsed.get("exact_count"))
        ):
            return row
    raise ValueError(f"No clean-correct confirmation row for {model_label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    args = parser.parse_args()

    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    row = select_row(args.generations, args.model)
    encoding = build_native_trace_encoding(
        row,
        tokenizer,
        site_id="answer_query_v3",
        candidate_counts=tuple(range(1, 11)),
    )
    position = int(encoding.query_position)
    conditions = {
        "clean": [],
        "single_head_L0H0": [(0, 0)],
    }
    rows = []
    for condition, heads in conditions.items():
        reference_logits = _scheduled_head_ablation_logits(
            model,
            adapter,
            encoding,
            heads,
            hook_positions=[position],
            score_positions=[position],
        )
        reference_generation = generate_with_head_ablation_at_positions(
            model,
            tokenizer,
            adapter,
            encoding,
            heads,
            hook_positions=[position],
            max_new_tokens=8,
        )
        fused = generate_with_head_ablation_at_positions(
            model,
            tokenizer,
            adapter,
            encoding,
            heads,
            hook_positions=[position],
            score_positions=[position],
            max_new_tokens=8,
        )
        fused_logits = fused.pop("prefill_selected_logits")
        difference = (reference_logits - fused_logits).abs()
        generated_equal = (
            reference_generation["generated_token_ids"]
            == fused["generated_token_ids"]
        )
        argmax_equal = bool(
            torch.equal(reference_logits.argmax(dim=-1), fused_logits.argmax(dim=-1))
        )
        allclose = bool(
            torch.allclose(reference_logits, fused_logits, rtol=0.05, atol=0.05)
        )
        rows.append(
            {
                "condition": condition,
                "generated_token_ids_equal": generated_equal,
                "diagnostic_argmax_equal": argmax_equal,
                "diagnostic_logits_allclose": allclose,
                "max_abs_logit_difference": float(difference.max()),
                "mean_abs_logit_difference": float(difference.mean()),
                "prefill_reuse_audit": fused.get("prefill_reuse_audit"),
                "head_ablation_hook_audit": fused.get(
                    "head_ablation_hook_audit"
                ),
            }
        )
    passed = all(
        row["generated_token_ids_equal"]
        and row["diagnostic_argmax_equal"]
        and row["diagnostic_logits_allclose"]
        and row["prefill_reuse_audit"] == "PASS_SINGLE_PREFILL"
        and row["head_ablation_hook_audit"] == "PASS"
        for row in rows
    )
    payload = {
        "schema_version": "v5_fused_prefill_equivalence_v1",
        "model_label": args.model,
        "request_id": row.get("request_id", row.get("stimulus_id")),
        "query_position": position,
        "conditions": rows,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
