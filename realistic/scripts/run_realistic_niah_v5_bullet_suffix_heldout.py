#!/usr/bin/env python3
"""Held-out test of count transfer from the native closing suffix.

The site and layers must be frozen before this script is run.  Alongside the
matched-k restoration, a cyclic wrong-k suffix donor tests count specificity.
"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--frozen-layers", type=int, nargs="+", required=True)
    parser.add_argument(
        "--construction",
        choices=(
            "structural_indices_scrubbed",
            "targeted_explicit_count_scrub",
        ),
        default="targeted_explicit_count_scrub",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--omit-wrong-control", action="store_true")
    parser.add_argument(
        "--selection-status",
        choices=(
            "site_and_layers_frozen_before_heldout_run",
            "discovery_layer_scan",
        ),
        default="site_and_layers_frozen_before_heldout_run",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.command = "bullet-suffix-heldout"

    wanted = tuple(int(value) for value in args.seeds)
    if len(set(wanted)) != len(wanted):
        raise ValueError("Held-out seeds must be unique")
    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected_by_seed = {
        int(row["seed"]): row for row in rows if int(row["seed"]) in set(wanted)
    }
    if set(selected_by_seed) != set(wanted):
        raise ValueError("One or more held-out seeds are absent")

    model, tokenizer, adapter = _model(args)
    layers = tuple(sorted({int(value) for value in args.frozen_layers}))
    if not layers or layers[0] < 0 or layers[-1] >= int(adapter.num_layers):
        raise ValueError("Frozen suffix layer is outside the decoder")
    targets = tuple(range(1, 11))
    results: list[dict[str, Any]] = []

    for seed in wanted:
        row = selected_by_seed[seed]
        source_full, blank_full, registry, audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260826 + int(seed),
            construction=str(args.construction),
        )
        suffix = terminal_suffix_with_optional_newline(
            row, tokenizer, prepend_newline=True
        )
        source_encodings: dict[int, Any] = {}
        blank_encodings: dict[int, Any] = {}
        suffix_positions: dict[int, tuple[int, ...]] = {}
        captured: dict[int, dict[int, Any]] = {}

        for occurrence in targets:
            source, source_geometry = build_item_early_stop_with_suffix(
                source_full,
                registry,
                target_occurrence=occurrence,
                terminal_suffix_token_ids=suffix,
            )
            blank, blank_geometry = build_item_early_stop_with_suffix(
                blank_full,
                registry,
                target_occurrence=occurrence,
                terminal_suffix_token_ids=suffix,
            )
            if source_geometry != blank_geometry:
                raise RuntimeError("Source/Blank held-out geometry differs")
            positions = tuple(
                range(int(source_geometry["suffix_start"]), int(source.query_position) + 1)
            )
            source_encodings[occurrence] = source
            blank_encodings[occurrence] = blank
            suffix_positions[occurrence] = positions
            captured[occurrence] = capture_decoder_block_input_states(
                model, adapter, source, positions, layers=layers
            )
            common = {
                "schema_version": "bullet_suffix_heldout_v1",
                "model_label": str(args.model),
                "seed": int(seed),
                "request_id": str(row["request_id"]),
                "target_occurrence": occurrence,
                "marker_kind": str(audit["marker_kind"]),
                "construction": str(args.construction),
                "site": "native_terminal_suffix_with_leading_newline",
                "selection_status": str(args.selection_status),
                **source_geometry,
            }
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
            results.extend(
                [
                    {
                        **common,
                        "condition": "source_reference",
                        "source_layer": -1,
                        "donor_occurrence": occurrence,
                        **source_outcomes,
                    },
                    {
                        **common,
                        "condition": "blank_reference",
                        "source_layer": -1,
                        "donor_occurrence": None,
                        **blank_outcomes,
                    },
                ]
            )

        for occurrence in targets:
            blank = blank_encodings[occurrence]
            positions = suffix_positions[occurrence]
            wrong_occurrence = occurrence % 10 + 1
            for layer in layers:
                patch_conditions = [("matched_k_restoration", occurrence)]
                if not bool(args.omit_wrong_control):
                    patch_conditions.append(
                        ("wrong_k_suffix_control", wrong_occurrence)
                    )
                for condition, donor_occurrence in patch_conditions:
                    replacement = captured[donor_occurrence][layer]
                    if int(replacement.shape[0]) != len(positions):
                        raise RuntimeError("Donor/receiver suffix lengths differ")
                    prefill, applications, norm = (
                        prefill_with_single_decoder_block_input_replacement(
                            model,
                            adapter,
                            blank,
                            positions=positions,
                            layer=layer,
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
                    prediction = outcomes["greedy_prediction"]
                    results.append(
                        {
                            "schema_version": "bullet_suffix_heldout_v1",
                            "model_label": str(args.model),
                            "seed": int(seed),
                            "request_id": str(row["request_id"]),
                            "target_occurrence": occurrence,
                            "marker_kind": str(audit["marker_kind"]),
                            "construction": str(args.construction),
                            "site": "native_terminal_suffix_with_leading_newline",
                            "selection_status": str(args.selection_status),
                            "condition": condition,
                            "source_layer": layer,
                            "donor_occurrence": donor_occurrence,
                            "donor_occurrence_exact": bool(
                                prediction is not None
                                and int(prediction) == int(donor_occurrence)
                            ),
                            "patch_token_count": len(positions),
                            "patch_hook_applications": int(applications),
                            "patch_realized_fro_norm": float(norm),
                            **outcomes,
                        }
                    )
        print(f"[suffix-heldout] seed={seed} complete", flush=True)

    _atomic_jsonl(args.output, results)
    print(f"[suffix-heldout] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
