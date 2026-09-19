#!/usr/bin/env python3
"""Scan local, cumulative, boundary, suffix, and readout counter sites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
    build_item_early_stop_with_suffix,
    capture_last_decoder_block_output_states,
    prefill_with_last_decoder_block_output_replacement,
    terminal_suffix_with_optional_newline,
)
from realistic_niah_v5.bullet_greedy_restore import (  # noqa: E402
    _greedy_integer_outcomes,
    _score_greedy_encoding,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
    prefill_with_single_decoder_block_input_replacement,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_jsonl,
    _model,
)


def _unique_positions(values: list[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--target-occurrences", type=int, nargs="+", required=True)
    parser.add_argument("--source-layers", type=int, nargs="+", required=True)
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=(
            "item_k",
            "item_k_last_token",
            "newline_after_item_k",
            "item_k_plus_newline",
            "cumulative_items_plus_newline",
            "visible_list_envelope_plus_newline",
            "native_terminal_suffix",
            "Total_query_token",
            "item_last_to_Total_query",
            "newline_to_Total_query",
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.command = "bullet-counter-site-diagnostics"

    wanted = {int(value) for value in args.seeds}
    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if int(row["seed"]) in wanted]
    if {int(row["seed"]) for row in selected} != wanted:
        raise ValueError("One or more requested diagnostic seeds are absent")
    targets = tuple(sorted({int(value) for value in args.target_occurrences}))
    layers = tuple(sorted({int(value) for value in args.source_layers}))
    if not targets or min(targets) < 1 or max(targets) > 10:
        raise ValueError("Diagnostic target occurrences must lie in 1..10")

    model, tokenizer, adapter = _model(args)
    if not layers or min(layers) < 0 or max(layers) >= int(adapter.num_layers):
        raise ValueError("Diagnostic source layer is outside the decoder")
    newline_count = len(tokenizer.encode("\n", add_special_tokens=False))
    if newline_count < 1:
        raise RuntimeError("No newline token is available")
    results: list[dict[str, Any]] = []

    for row in selected:
        source_full, blank_full, registry, audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260826 + int(row["seed"]),
            construction="structural_indices_scrubbed",
        )
        suffix = terminal_suffix_with_optional_newline(
            row, tokenizer, prepend_newline=True
        )
        for occurrence in targets:
            source, early_audit = build_item_early_stop_with_suffix(
                source_full,
                registry,
                target_occurrence=occurrence,
                terminal_suffix_token_ids=suffix,
            )
            blank, blank_early_audit = build_item_early_stop_with_suffix(
                blank_full,
                registry,
                target_occurrence=occurrence,
                terminal_suffix_token_ids=suffix,
            )
            if early_audit != blank_early_audit:
                raise RuntimeError("Source/Blank early-stop geometry differs")
            item_start, item_end = registry.trace_items[occurrence - 1]
            newline_positions = tuple(
                range(int(item_end), int(item_end) + int(newline_count))
            )
            cumulative_item_positions = _unique_positions(
                [
                    position
                    for start, end in registry.trace_items[:occurrence]
                    for position in range(int(start), int(end))
                ]
            )
            first_item_start = int(registry.trace_items[0][0])
            query = int(source.query_position)
            sites = {
                "item_k": (
                    tuple(range(int(item_start), int(item_end))),
                    tuple(range(int(item_start), int(item_end))),
                ),
                "item_k_last_token": ((int(item_end) - 1,), (int(item_end) - 1,)),
                "newline_after_item_k": (newline_positions, newline_positions),
                "item_k_plus_newline": (
                    tuple(range(int(item_start), int(item_end))) + newline_positions,
                    tuple(range(int(item_start), int(item_end))) + newline_positions,
                ),
                "cumulative_items_plus_newline": (
                    cumulative_item_positions + newline_positions,
                    cumulative_item_positions + newline_positions,
                ),
                "visible_list_envelope_plus_newline": (
                    tuple(range(first_item_start, int(item_end) + int(newline_count))),
                    tuple(range(first_item_start, int(item_end) + int(newline_count))),
                ),
                "native_terminal_suffix": (
                    tuple(range(int(item_end), query + 1)),
                    tuple(range(int(item_end), query + 1)),
                ),
                "Total_query_token": ((query,), (query,)),
                "item_last_to_Total_query": ((int(item_end) - 1,), (query,)),
                "newline_to_Total_query": ((int(item_end),), (query,)),
            }
            if args.sites:
                requested_sites = tuple(dict.fromkeys(str(value) for value in args.sites))
                sites = {site: sites[site] for site in requested_sites}
            capture_positions = tuple(range(first_item_start, query + 1))
            position_to_row = {
                position: index for index, position in enumerate(capture_positions)
            }
            captured = capture_decoder_block_input_states(
                model,
                adapter,
                source,
                capture_positions,
                layers=layers,
            )
            source_outcomes = _score_greedy_encoding(
                model,
                tokenizer,
                adapter,
                source,
                target_k=occurrence,
                max_new_tokens=int(args.max_new_tokens),
            )
            blank_outcomes = _score_greedy_encoding(
                model,
                tokenizer,
                adapter,
                blank,
                target_k=occurrence,
                max_new_tokens=int(args.max_new_tokens),
            )
            common = {
                "schema_version": "bullet_counter_site_diagnostic_v1",
                "model_label": str(args.model),
                "seed": int(row["seed"]),
                "request_id": str(row["request_id"]),
                "marker_kind": str(audit["marker_kind"]),
                "construction": "structural_indices_scrubbed",
                "prepend_newline": True,
                "target_occurrence": int(occurrence),
                **early_audit,
            }
            results.extend(
                [
                    {
                        **common,
                        "condition": "source_reference",
                        "site": "reference",
                        "source_layer": -1,
                        **source_outcomes,
                    },
                    {
                        **common,
                        "condition": "blank_reference",
                        "site": "reference",
                        "source_layer": -1,
                        **blank_outcomes,
                    },
                ]
            )
            for layer in layers:
                layer_states = captured[int(layer)]
                for site, (donor_positions, receiver_positions) in sites.items():
                    indices = [position_to_row[position] for position in donor_positions]
                    replacement = layer_states[indices]
                    prefill, applications, norm = (
                        prefill_with_single_decoder_block_input_replacement(
                            model,
                            adapter,
                            blank,
                            positions=receiver_positions,
                            layer=int(layer),
                            replacement_states=replacement,
                        )
                    )
                    outcomes = _greedy_integer_outcomes(
                        model,
                        tokenizer,
                        blank,
                        prefill,
                        target_k=occurrence,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    results.append(
                        {
                            **common,
                            "condition": "restoration",
                            "site": site,
                            "source_layer": int(layer),
                            "patch_token_count": len(receiver_positions),
                            "donor_receiver_positions_identical": bool(
                                donor_positions == receiver_positions
                            ),
                            "patch_hook_applications": int(applications),
                            "patch_realized_fro_norm": float(norm),
                            **outcomes,
                        }
                    )

            final_positions = (query,)
            final_states = capture_last_decoder_block_output_states(
                model, adapter, source, final_positions
            )
            final_prefill, final_applications = (
                prefill_with_last_decoder_block_output_replacement(
                    model,
                    adapter,
                    blank,
                    positions=final_positions,
                    replacement_states=final_states,
                )
            )
            final_outcomes = _greedy_integer_outcomes(
                model,
                tokenizer,
                blank,
                final_prefill,
                target_k=occurrence,
                max_new_tokens=int(args.max_new_tokens),
            )
            results.append(
                {
                    **common,
                    "condition": "restoration",
                    "site": "Total_query_token_last_block_output",
                    "source_layer": int(adapter.num_layers),
                    "patch_token_count": 1,
                    "patch_hook_applications": int(final_applications),
                    **final_outcomes,
                }
            )
            print(
                f"[site-diagnostic] model={args.model} seed={row['seed']} "
                f"k={occurrence}",
                flush=True,
            )
    _atomic_jsonl(args.output, results)
    print(f"[site-diagnostic] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
