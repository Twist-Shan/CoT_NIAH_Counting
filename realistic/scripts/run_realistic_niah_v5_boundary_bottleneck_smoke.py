#!/usr/bin/env python3
"""Smoke test full-native-trace boundary-state transplant under a graph cut."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import (  # noqa: E402
    _bounded_logits_kwargs,
    _encoding_tensors,
)
from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    build_standard_4d_causal_mask,
    build_suffix_attention_bottleneck_mask,
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


def _last_logits(prefill: Any) -> torch.Tensor:
    return prefill.logits[0, -1].detach().float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=(28, 32, 33))
    parser.add_argument("--targets", type=int, nargs="+", default=(2, 5, 9))
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "boundary-attention-bottleneck-smoke"

    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = tuple(int(value) for value in args.seeds)
    selected = {int(row["seed"]): row for row in rows if int(row["seed"]) in wanted}
    if set(selected) != set(wanted):
        raise ValueError("One or more smoke seeds are absent from generations")
    targets = tuple(int(value) for value in args.targets)
    if not targets or min(targets) < 1 or max(targets) > 10:
        raise ValueError("Targets must be occurrences in 1..10")

    model, tokenizer, adapter = _model(args)
    layers = tuple(sorted({int(value) for value in args.layers}))
    if not layers or layers[0] < 0 or layers[-1] >= int(adapter.num_layers):
        raise ValueError("A requested layer is outside the decoder")

    results: list[dict[str, Any]] = []
    sanity: list[dict[str, Any]] = []
    for seed in wanted:
        row = selected[seed]
        source, blank, registry, construction_audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260827 + int(seed),
            construction="targeted_explicit_count_scrub",
        )
        suffix_positions, suffix_audit = full_native_terminal_suffix_positions(
            row, tokenizer, source
        )
        scaffold_end = int(registry.trace_items[0][0])
        boundary_by_k: dict[int, int] = {}
        boundary_audit_by_k: dict[int, dict[str, Any]] = {}
        for k in targets:
            boundary, boundary_audit = select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=k
            )
            boundary_by_k[k] = boundary
            boundary_audit_by_k[k] = boundary_audit

        captures = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            tuple(boundary_by_k[k] for k in targets),
            layers=layers,
        )
        captured_by_k = {
            layer: {
                k: captures[layer][index : index + 1]
                for index, k in enumerate(targets)
            }
            for layer in layers
        }

        # A dense standard causal mask must preserve the standard forward's
        # answer-site decision before the graph-cut intervention is trusted.
        input_ids, attention_mask_2d = _encoding_tensors(model, source)
        with torch.inference_mode():
            standard_prefill = model(
                input_ids=input_ids,
                attention_mask=attention_mask_2d,
                use_cache=False,
                **_bounded_logits_kwargs(model),
            )
        standard_logits = _last_logits(standard_prefill)
        del standard_prefill
        standard_4d = build_standard_4d_causal_mask(
            source, device=input_ids.device
        )
        causal_4d_prefill, _, _ = prefill_with_custom_attention_mask(
            model,
            adapter,
            source,
            attention_mask_4d=standard_4d,
        )
        causal_4d_logits = _last_logits(causal_4d_prefill)
        sanity_row = {
            "seed": seed,
            "standard_top_token": int(torch.argmax(standard_logits)),
            "causal_4d_top_token": int(torch.argmax(causal_4d_logits)),
            "top_token_match": bool(
                int(torch.argmax(standard_logits)) == int(torch.argmax(causal_4d_logits))
            ),
            "last_logit_max_abs_diff": float(
                torch.max(torch.abs(standard_logits - causal_4d_logits))
            ),
        }
        sanity.append(sanity_row)
        del causal_4d_prefill, standard_4d, standard_logits, causal_4d_logits

        wrong_donor = {
            k: targets[(index + 1) % len(targets)]
            for index, k in enumerate(targets)
        }
        for k in targets:
            boundary_position = boundary_by_k[k]
            source_mask = build_suffix_attention_bottleneck_mask(
                source,
                boundary_positions=(boundary_position,),
                suffix_positions=suffix_positions,
                scaffold_end=scaffold_end,
                device=input_ids.device,
            )
            blank_mask = build_suffix_attention_bottleneck_mask(
                blank,
                boundary_positions=(boundary_position,),
                suffix_positions=suffix_positions,
                scaffold_end=scaffold_end,
                device=input_ids.device,
            )
            common = {
                "schema_version": "boundary_attention_bottleneck_smoke_v1",
                "model_label": str(args.model),
                "seed": seed,
                "request_id": str(row["request_id"]),
                "target_occurrence": k,
                "full_trace_sequence_length": int(source.sequence_length),
                "native_query_position": int(source.query_position),
                "suffix_attention_access": "scrubbed_prompt+scrubbed_prelist_scaffold+selected_boundary+suffix_self",
                "suffix_scaffold_end": scaffold_end,
                "future_items_deleted": False,
                "post_list_recap_present_but_unreachable": True,
                **boundary_audit_by_k[k],
            }

            for condition, active_encoding, active_mask in (
                ("source_bottleneck", source, source_mask),
                ("blank_bottleneck", blank, blank_mask),
            ):
                prefill, applications, norm = prefill_with_custom_attention_mask(
                    model,
                    adapter,
                    active_encoding,
                    attention_mask_4d=active_mask,
                )
                outcome = greedy_integer_from_bottleneck_prefill(
                    model,
                    tokenizer,
                    active_encoding,
                    prefill,
                    boundary_positions=(boundary_position,),
                    suffix_positions=suffix_positions,
                    scaffold_end=scaffold_end,
                    target_count=k,
                    max_new_tokens=int(args.max_new_tokens),
                )
                results.append(
                    {
                        **common,
                        "condition": condition,
                        "source_layer": -1,
                        "donor_occurrence": k if condition == "source_bottleneck" else None,
                        "patch_hook_applications": applications,
                        "patch_realized_fro_norm": norm,
                        **outcome,
                    }
                )
                del prefill

            for layer in layers:
                for condition, donor in (
                    ("matched_boundary_transplant", k),
                    ("wrong_boundary_transplant", wrong_donor[k]),
                ):
                    prefill, applications, norm = prefill_with_custom_attention_mask(
                        model,
                        adapter,
                        blank,
                        attention_mask_4d=blank_mask,
                        patch_positions=(boundary_position,),
                        patch_layer=layer,
                        replacement_states=captured_by_k[layer][donor],
                    )
                    outcome = greedy_integer_from_bottleneck_prefill(
                        model,
                        tokenizer,
                        blank,
                        prefill,
                        boundary_positions=(boundary_position,),
                        suffix_positions=suffix_positions,
                        scaffold_end=scaffold_end,
                        target_count=k,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    prediction = outcome["greedy_prediction"]
                    results.append(
                        {
                            **common,
                            "condition": condition,
                            "source_layer": layer,
                            "donor_occurrence": donor,
                            "donor_occurrence_exact": bool(
                                prediction is not None and int(prediction) == donor
                            ),
                            "patch_hook_applications": applications,
                            "patch_realized_fro_norm": norm,
                            **outcome,
                        }
                    )
                    del prefill
            del source_mask, blank_mask
            print(f"[boundary-bottleneck] seed={seed} k={k} complete", flush=True)

        print(f"[boundary-bottleneck] seed={seed} complete", flush=True)

    _atomic_jsonl(args.output, results)
    grouped: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row["condition"]) for row in results}):
        for layer in sorted(
            {int(row["source_layer"]) for row in results if row["condition"] == condition}
        ):
            values = [
                row
                for row in results
                if row["condition"] == condition and int(row["source_layer"]) == layer
            ]
            grouped[f"{condition}|L{layer}"] = {
                "n": len(values),
                "greedy_exact": sum(bool(row["greedy_exact"]) for row in values),
                "candidate_argmax_exact": sum(
                    bool(row["candidate_argmax_exact"]) for row in values
                ),
                "donor_occurrence_exact": sum(
                    bool(row.get("donor_occurrence_exact")) for row in values
                ),
            }
    _atomic_json(
        args.summary,
        {
            "schema_version": "boundary_attention_bottleneck_smoke_v1",
            "seeds": list(wanted),
            "targets": list(targets),
            "layers": list(layers),
            "sanity": sanity,
            "conditions": grouped,
            "construction_audit_last_seed": construction_audit,
            "suffix_audit_last_seed": suffix_audit,
        },
    )
    print(f"[boundary-bottleneck] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
