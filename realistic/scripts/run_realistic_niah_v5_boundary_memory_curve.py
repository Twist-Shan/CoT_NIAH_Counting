#!/usr/bin/env python3
"""Measure how much full-trace memory the isolated native suffix needs."""

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

from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    build_suffix_attention_bottleneck_mask,
    full_native_terminal_suffix_positions,
    greedy_integer_from_bottleneck_prefill,
    memory_geometry_positions,
    prefill_with_custom_attention_mask,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)


GEOMETRIES = (
    "post_item_boundary",
    "item_endpoint",
    "item_suffix4",
    "full_item",
    "all_boundaries_through_k",
    "all_items_through_k",
    "list_prefix_through_k",
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
    parser.add_argument("--targets", type=int, nargs="+", default=(2, 5, 9))
    parser.add_argument(
        "--geometries", choices=GEOMETRIES, nargs="+", default=GEOMETRIES
    )
    parser.add_argument("--include-blank", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "boundary-memory-curve"

    wanted = tuple(int(value) for value in args.seeds)
    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {int(row["seed"]): row for row in rows if int(row["seed"]) in wanted}
    if set(selected) != set(wanted):
        raise ValueError("One or more memory-curve seeds are absent")
    targets = tuple(int(value) for value in args.targets)
    geometries = tuple(str(value) for value in args.geometries)
    model, tokenizer, adapter = _model(args)
    model_device = next(model.parameters()).device
    results: list[dict[str, Any]] = []

    for seed in wanted:
        row = selected[seed]
        source, blank, registry, construction_audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260828 + int(seed),
            construction="targeted_explicit_count_scrub",
        )
        suffix_positions, suffix_audit = full_native_terminal_suffix_positions(
            row, tokenizer, source
        )
        scaffold_end = int(registry.trace_items[0][0])
        for k in targets:
            for geometry in geometries:
                memory_positions, geometry_audit = memory_geometry_positions(
                    source,
                    registry,
                    tokenizer,
                    occurrence=k,
                    geometry=geometry,
                )
                conditions = [("source_bottleneck", source)]
                if bool(args.include_blank):
                    conditions.append(("blank_bottleneck", blank))
                for condition, encoding in conditions:
                    mask = build_suffix_attention_bottleneck_mask(
                        encoding,
                        boundary_positions=memory_positions,
                        suffix_positions=suffix_positions,
                        scaffold_end=scaffold_end,
                        device=model_device,
                    )
                    prefill, applications, norm = prefill_with_custom_attention_mask(
                        model,
                        adapter,
                        encoding,
                        attention_mask_4d=mask,
                    )
                    outcome = greedy_integer_from_bottleneck_prefill(
                        model,
                        tokenizer,
                        encoding,
                        prefill,
                        boundary_positions=memory_positions,
                        suffix_positions=suffix_positions,
                        scaffold_end=scaffold_end,
                        target_count=k,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    results.append(
                        {
                            "schema_version": "boundary_memory_curve_v1",
                            "model_label": str(args.model),
                            "seed": seed,
                            "request_id": str(row["request_id"]),
                            "condition": condition,
                            "future_items_deleted": False,
                            "post_list_recap_present_but_unreachable": True,
                            "suffix_scaffold_end": scaffold_end,
                            "full_trace_sequence_length": int(encoding.sequence_length),
                            "native_query_position": int(encoding.query_position),
                            "patch_hook_applications": applications,
                            "patch_realized_fro_norm": norm,
                            **geometry_audit,
                            **outcome,
                        }
                    )
                    del prefill, mask
                print(
                    f"[boundary-memory] seed={seed} k={k} geometry={geometry} complete",
                    flush=True,
                )

    _atomic_jsonl(args.output, results)
    grouped: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row["condition"]) for row in results}):
        for geometry in geometries:
            values = [
                row
                for row in results
                if row["condition"] == condition
                and row["memory_geometry"] == geometry
            ]
            grouped[f"{condition}|{geometry}"] = {
                "n": len(values),
                "greedy_exact": sum(bool(row["greedy_exact"]) for row in values),
                "predictions": [row["greedy_prediction"] for row in values],
                "targets": [int(row["target_occurrence"]) for row in values],
                "mean_memory_token_count": (
                    sum(int(row["memory_token_count"]) for row in values) / len(values)
                    if values
                    else None
                ),
            }
    _atomic_json(
        args.summary,
        {
            "schema_version": "boundary_memory_curve_v1",
            "seeds": list(wanted),
            "targets": list(targets),
            "geometries": list(geometries),
            "conditions": grouped,
            "construction_audit_last_seed": construction_audit,
            "suffix_audit_last_seed": suffix_audit,
        },
    )
    print(f"[boundary-memory] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
