#!/usr/bin/env python3
"""Clamp one local trace site through later layers and read the running count."""

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
    prefill_with_layerwise_decoder_block_input_replacements,
    terminal_suffix_with_optional_newline,
)
from realistic_niah_v5.bullet_greedy_restore import (  # noqa: E402
    _greedy_integer_outcomes,
    _score_greedy_encoding,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_jsonl,
    _model,
)


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
    parser.add_argument("--start-layers", type=int, nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.command = "bullet-multilayer-clamp-diagnostics"

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
    starts = tuple(sorted({int(value) for value in args.start_layers}))
    model, tokenizer, adapter = _model(args)
    if not starts or min(starts) < 0 or max(starts) >= int(adapter.num_layers) - 1:
        raise ValueError("Clamp start layers must precede the last decoder block")
    newline_count = len(tokenizer.encode("\n", add_special_tokens=False))
    all_layers = tuple(range(int(adapter.num_layers)))
    results: list[dict[str, Any]] = []

    for row in selected:
        source_full, blank_full, registry, audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260827 + int(row["seed"]),
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
            item_positions = tuple(range(int(item_start), int(item_end)))
            newline_positions = tuple(
                range(int(item_end), int(item_end) + int(newline_count))
            )
            capture_positions = item_positions + newline_positions
            source_states = capture_decoder_block_input_states(
                model,
                adapter,
                source,
                capture_positions,
                layers=all_layers,
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
                "schema_version": "bullet_multilayer_clamp_diagnostic_v1",
                "model_label": str(args.model),
                "seed": int(row["seed"]),
                "request_id": str(row["request_id"]),
                "marker_kind": str(audit["marker_kind"]),
                "construction": "structural_indices_scrubbed",
                "target_occurrence": int(occurrence),
                **early_audit,
            }
            results.extend(
                [
                    {
                        **common,
                        "condition": "source_reference",
                        "site": "reference",
                        "start_layer": -1,
                        **source_outcomes,
                    },
                    {
                        **common,
                        "condition": "blank_reference",
                        "site": "reference",
                        "start_layer": -1,
                        **blank_outcomes,
                    },
                ]
            )
            site_indices = {
                "item_k": tuple(range(len(item_positions))),
                "newline_after_item_k": tuple(
                    range(len(item_positions), len(capture_positions))
                ),
                "item_k_plus_newline": tuple(range(len(capture_positions))),
            }
            site_positions = {
                "item_k": item_positions,
                "newline_after_item_k": newline_positions,
                "item_k_plus_newline": capture_positions,
            }
            for start_layer in starts:
                active_layers = tuple(range(int(start_layer), int(adapter.num_layers)))
                for site in site_indices:
                    indices = list(site_indices[site])
                    positions = site_positions[site]
                    replacements = {
                        layer: source_states[layer][indices].clone()
                        for layer in active_layers
                    }
                    prefill, applications, norms = (
                        prefill_with_layerwise_decoder_block_input_replacements(
                            model,
                            adapter,
                            blank,
                            positions=positions,
                            replacements=replacements,
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
                            "condition": "multilayer_clamp_restoration",
                            "site": site,
                            "start_layer": int(start_layer),
                            "patch_layers": list(active_layers),
                            "patch_layer_count": len(active_layers),
                            "patch_token_count": len(positions),
                            "patch_hook_applications": {
                                str(layer): int(value)
                                for layer, value in applications.items()
                            },
                            "patch_realized_fro_norms": {
                                str(layer): float(value) for layer, value in norms.items()
                            },
                            **outcomes,
                        }
                    )
            print(
                f"[multilayer-clamp] model={args.model} seed={row['seed']} "
                f"k={occurrence}",
                flush=True,
            )
    _atomic_jsonl(args.output, results)
    print(f"[multilayer-clamp] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
