#!/usr/bin/env python3
"""Pilot counted versus no-op transitions from a transplanted boundary state."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    build_transition_attention_bottleneck_mask,
    full_native_terminal_suffix_positions,
    greedy_integer_from_bottleneck_prefill,
    prefill_with_custom_attention_mask,
    select_post_item_boundary_position,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)


_EXPLICIT_COUNT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:10|[0-9]|zero|one|two|three|four|five|six|seven|eight|nine|ten)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--receiver-occurrence", type=int, default=5)
    parser.add_argument("--donor-occurrences", type=int, nargs="+", default=(2, 5, 8))
    parser.add_argument("--layers", type=int, nargs="+", default=(28, 32, 33))
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "increment-noop-transition-smoke"

    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if int(row["seed"]) == int(args.seed)]
    if len(selected) != 1:
        raise ValueError("Transition smoke seed is absent or duplicated")
    row = selected[0]
    receiver_k = int(args.receiver_occurrence)
    donors = tuple(int(value) for value in args.donor_occurrences)
    if not 1 <= receiver_k < 10 or min(donors) < 1 or max(donors) >= 10:
        raise ValueError("Transition occurrences must leave room for an increment")

    model, tokenizer, adapter = _model(args)
    layers = tuple(sorted({int(value) for value in args.layers}))
    source, blank, registry, audit = build_diagnostic_bases(
        row,
        tokenizer,
        random_seed=20260829 + int(args.seed),
        construction="targeted_explicit_count_scrub",
    )
    suffix_positions, suffix_audit = full_native_terminal_suffix_positions(
        row, tokenizer, source
    )
    scaffold_end = int(registry.trace_items[0][0])
    donor_boundaries: dict[int, int] = {}
    for donor in set(donors) | {receiver_k}:
        donor_boundaries[donor] = select_post_item_boundary_position(
            source, registry, tokenizer, occurrence=donor
        )[0]
    receiver_boundary = donor_boundaries[receiver_k]
    next_boundary, next_boundary_audit = select_post_item_boundary_position(
        source, registry, tokenizer, occurrence=receiver_k + 1
    )
    next_start, _next_end = (
        int(value) for value in registry.trace_items[receiver_k]
    )
    transition_positions = tuple(range(next_start, next_boundary + 1))

    counted_ids = list(blank.input_ids)
    for position in transition_positions:
        counted_ids[position] = int(source.input_ids[position])
    counted = replace(blank, input_ids=tuple(counted_ids))
    noop = blank
    counted_text = tokenizer.decode(
        [int(counted.input_ids[position]) for position in transition_positions],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    noop_text = tokenizer.decode(
        [int(noop.input_ids[position]) for position in transition_positions],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if _EXPLICIT_COUNT_RE.search(counted_text) or _EXPLICIT_COUNT_RE.search(noop_text):
        raise ValueError("A transition span contains an explicit candidate count")

    capture_positions = tuple(donor_boundaries[donor] for donor in donors)
    captures = capture_decoder_block_input_states(
        model, adapter, source, capture_positions, layers=layers
    )
    captured_by_layer = {
        layer: {
            donor: captures[layer][index : index + 1]
            for index, donor in enumerate(donors)
        }
        for layer in layers
    }
    device = next(model.parameters()).device
    results: list[dict[str, Any]] = []
    for transition_kind, encoding, increment in (
        ("counted", counted, 1),
        ("noop", noop, 0),
    ):
        mask = build_transition_attention_bottleneck_mask(
            encoding,
            scaffold_end=scaffold_end,
            donor_boundary_positions=(receiver_boundary,),
            transition_positions=transition_positions,
            next_boundary_positions=(next_boundary,),
            suffix_positions=suffix_positions,
            device=device,
        )
        for layer in layers:
            for donor in donors:
                expected = donor + increment
                prefill, applications, norm = prefill_with_custom_attention_mask(
                    model,
                    adapter,
                    encoding,
                    attention_mask_4d=mask,
                    patch_positions=(receiver_boundary,),
                    patch_layer=layer,
                    replacement_states=captured_by_layer[layer][donor],
                )
                outcome = greedy_integer_from_bottleneck_prefill(
                    model,
                    tokenizer,
                    encoding,
                    prefill,
                    boundary_positions=(next_boundary,),
                    suffix_positions=suffix_positions,
                    scaffold_end=scaffold_end,
                    target_count=expected,
                    max_new_tokens=int(args.max_new_tokens),
                )
                results.append(
                    {
                        "schema_version": "increment_noop_transition_smoke_v1",
                        "model_label": str(args.model),
                        "seed": int(args.seed),
                        "request_id": str(row["request_id"]),
                        "condition": f"{transition_kind}_transition",
                        "transition_kind": transition_kind,
                        "source_layer": layer,
                        "receiver_occurrence": receiver_k,
                        "donor_occurrence": donor,
                        "expected_count": expected,
                        "transition_delta": increment,
                        "receiver_boundary_position": receiver_boundary,
                        "next_boundary_position": next_boundary,
                        "transition_positions": list(transition_positions),
                        "transition_token_count": len(transition_positions),
                        "future_items_deleted": False,
                        "post_list_recap_present_but_unreachable": True,
                        "transition_reads_old_list_history": False,
                        "suffix_reads_donor_boundary_directly": False,
                        "patch_hook_applications": applications,
                        "patch_realized_fro_norm": norm,
                        **outcome,
                    }
                )
                del prefill
            print(
                f"[transition] kind={transition_kind} layer={layer} complete",
                flush=True,
            )
        del mask

    _atomic_jsonl(args.output, results)
    grouped: dict[str, Any] = {}
    for kind in ("counted_transition", "noop_transition"):
        for layer in layers:
            values = [
                result
                for result in results
                if result["condition"] == kind and result["source_layer"] == layer
            ]
            grouped[f"{kind}|L{layer}"] = {
                "n": len(values),
                "greedy_exact": sum(bool(value["greedy_exact"]) for value in values),
                "predictions": [value["greedy_prediction"] for value in values],
                "expected": [int(value["expected_count"]) for value in values],
            }
    _atomic_json(
        args.summary,
        {
            "schema_version": "increment_noop_transition_smoke_v1",
            "seed": int(args.seed),
            "receiver_occurrence": receiver_k,
            "donor_occurrences": list(donors),
            "layers": list(layers),
            "counted_transition_text": counted_text,
            "noop_transition_text": noop_text,
            "source_boundary_direct_readout_ceiling_established": False,
            "interpretation_gate": "transition_is_diagnostic_only_until_boundary_source_ceiling_passes",
            "conditions": grouped,
            "construction_audit": audit,
            "suffix_audit": suffix_audit,
            "next_boundary_audit": next_boundary_audit,
        },
    )
    print(f"[transition] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
