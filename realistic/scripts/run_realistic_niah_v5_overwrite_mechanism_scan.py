#!/usr/bin/env python3
"""Diagnose why a patched native count returns to the clean next count.

The scan crosses three factors while leaving the list grammar intact:

1. donor count state at boundary B_r;
2. the first layer from which that boundary is repeatedly clamped;
3. native next-item content, either original or swapped with an equal-token-
   length future item from the same trace.

Frozen probes at several layers read both B_r and B_(r+1).  The three
competing discrete predictions are therefore directly distinguishable:

* recurrent successor: donor + 1;
* position reset: receiver + 1;
* content lookup: the original occurrence of the swapped-in item.

The content counterfactual swaps complete equal-length item spans.  Thus all
token positions, attention masks, separators, and total sequence length are
preserved, and the second half of the swap lies causally after B_(r+1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from realistic_niah_v5.unified_carrier_transition import (  # noqa: E402
    carrier_capture_layer_positions,
)
from scripts.run_realistic_niah_v5_boundary_equivariance import (  # noqa: E402
    decode_count_probe,
    through_origin_slope,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_unified_carrier_transition import (  # noqa: E402
    _read_rows,
)


SCHEMA_VERSION = "overwrite_mechanism_scan_v1"


def _token_digest(values: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def swap_equal_length_item_contents(
    encoding: Any,
    registry: Any,
    *,
    left_occurrence: int,
    right_occurrence: int,
) -> tuple[Any, dict[str, Any]]:
    """Swap two complete item spans without moving any token position."""

    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    left = int(left_occurrence)
    right = int(right_occurrence)
    if not 1 <= left < right <= len(items):
        raise ValueError("Content swap occurrences must be ordered list indices")
    left_start, left_end = items[left - 1]
    right_start, right_end = items[right - 1]
    width = left_end - left_start
    if width < 1 or right_end - right_start != width:
        raise ValueError("Content swap requires equal nonempty token spans")
    if left_end > right_start:
        raise ValueError("Content swap item spans overlap")
    original = tuple(int(value) for value in encoding.input_ids)
    output = list(original)
    left_ids = original[left_start:left_end]
    right_ids = original[right_start:right_end]
    output[left_start:left_end] = right_ids
    output[right_start:right_end] = left_ids
    changed = sum(
        int(before != after) for before, after in zip(original, tuple(output))
    )
    if len(output) != len(original) or changed < 1:
        raise RuntimeError("Content swap failed to make a length-preserving edit")
    counterfactual = replace(encoding, input_ids=tuple(output))
    if tuple(counterfactual.attention_mask) != tuple(encoding.attention_mask):
        raise RuntimeError("Content swap changed the attention mask")
    return counterfactual, {
        "left_occurrence": left,
        "right_occurrence": right,
        "item_token_width": width,
        "changed_token_count": changed,
        "left_original_token_sha256": _token_digest(left_ids),
        "right_original_token_sha256": _token_digest(right_ids),
        "sequence_length_preserved": True,
        "attention_mask_preserved": True,
        "positions_preserved": True,
        "second_swap_is_after_left_boundary": True,
    }


def choose_future_equal_length_content(
    registry: Any,
    *,
    physical_next_occurrence: int,
    excluded_occurrences: Sequence[int] = (),
) -> int | None:
    """Choose the latest distinguishable future item of the same width."""

    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    target = int(physical_next_occurrence)
    if not 1 <= target < len(items):
        raise ValueError("A future content donor requires a nonterminal target")
    width = items[target - 1][1] - items[target - 1][0]
    excluded = {int(value) for value in excluded_occurrences} | {target}
    candidates = [
        occurrence
        for occurrence in range(target + 1, len(items) + 1)
        if occurrence not in excluded
        and items[occurrence - 1][1] - items[occurrence - 1][0] == width
    ]
    return max(candidates) if candidates else None


def replace_positions_from_encoding(
    encoding: Any,
    replacement_source: Any,
    *,
    positions: Sequence[int],
    variant: str,
) -> tuple[Any, dict[str, Any]]:
    """Replace fixed positions from a same-geometry neutral encoding."""

    selected = tuple(sorted({int(value) for value in positions}))
    if not selected or selected[0] < 0 or selected[-1] >= int(
        encoding.sequence_length
    ):
        raise ValueError("Structural scrub positions are invalid")
    if int(replacement_source.sequence_length) != int(encoding.sequence_length):
        raise ValueError("Structural scrub source changed sequence length")
    if tuple(replacement_source.attention_mask) != tuple(encoding.attention_mask):
        raise ValueError("Structural scrub source changed the attention mask")
    original = tuple(int(value) for value in encoding.input_ids)
    donor = tuple(int(value) for value in replacement_source.input_ids)
    output = list(original)
    for position in selected:
        output[position] = donor[position]
    changed = sum(original[position] != output[position] for position in selected)
    if changed < 1:
        raise RuntimeError("Structural scrub made no token change")
    return replace(encoding, input_ids=tuple(output)), {
        "structural_variant": str(variant),
        "selected_position_count": len(selected),
        "changed_token_count": int(changed),
        "selected_position_min": selected[0],
        "selected_position_max": selected[-1],
        "sequence_length_preserved": True,
        "attention_mask_preserved": True,
        "positions_preserved": True,
        "separators_preserved": True,
    }


def build_structure_scrub_variants(
    source: Any,
    neutral: Any,
    registry: Any,
    *,
    physical_next_occurrence: int,
) -> tuple[tuple[str, Any, dict[str, Any]], ...]:
    """Build marker/payload interventions through the next native item."""

    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    physical_next = int(physical_next_occurrence)
    if not 1 <= physical_next <= len(items):
        raise ValueError("Structural scrub target is outside the native list")
    active = items[:physical_next]
    if any(end - start < 3 for start, end in active):
        raise ValueError("Every scrubbed item needs a two-token marker prefix")
    next_start, next_end = items[physical_next - 1]
    prior = items[: physical_next - 1]
    plans = {
        "next_item_payload_scrub": tuple(range(next_start + 2, next_end)),
        "prior_payload_scrub": tuple(
            position
            for start, end in prior
            for position in range(start + 2, end)
        ),
        "history_payload_scrub": tuple(
            position
            for start, end in active
            for position in range(start + 2, end)
        ),
        "next_item_marker_scrub": tuple(range(next_start, next_start + 2)),
        "prior_marker_scrub": tuple(
            position for start, _end in prior for position in range(start, start + 2)
        ),
        "history_marker_scrub": tuple(
            position for start, _end in active for position in range(start, start + 2)
        ),
        "history_full_item_scrub": tuple(
            position for start, end in active for position in range(start, end)
        ),
    }
    output = []
    for variant, positions in plans.items():
        encoding, audit = replace_positions_from_encoding(
            source,
            neutral,
            positions=positions,
            variant=variant,
        )
        output.append((variant, encoding, audit))
    return tuple(output)


def _load_probes(path: Path, read_layers: Sequence[int]) -> dict[int, dict[str, Any]]:
    payload = np.load(path)
    available = {
        int(value) for value in np.asarray(payload["frozen_layers"]).reshape(-1)
    }
    requested = tuple(sorted({int(value) for value in read_layers}))
    if not requested or any(layer not in available for layer in requested):
        raise ValueError(
            f"Requested read layers {requested} are not all frozen in {sorted(available)}"
        )
    alpha = float(np.asarray(payload["alpha"]).reshape(-1)[0])
    return {
        layer: {
            "mean": np.asarray(payload[f"layer_{layer}_mean"], dtype=np.float32),
            "weights": np.asarray(
                payload[f"layer_{layer}_weights"], dtype=np.float32
            ),
            "alpha": alpha,
        }
        for layer in requested
    }


def summarize_trials(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize each layer-start/content cell without pooling token reads."""

    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["content_variant"]),
                int(row["clamp_start_layer"]),
                int(row["read_layer"]),
            )
        ].append(row)
    cells = []
    for (content_variant, clamp_start, read_layer), active in sorted(
        grouped.items()
    ):
        current_shift = [float(row["current_soft_shift"]) for row in active]
        next_shift = [float(row["next_soft_shift"]) for row in active]
        signed_current = [
            float(row["dose"]) * float(row["current_soft_shift"]) for row in active
        ]
        signed_next = [
            float(row["dose"]) * float(row["next_soft_shift"]) for row in active
        ]
        confusion = Counter(int(row["next_prediction"]) for row in active)
        cells.append(
            {
                "content_variant": content_variant,
                "clamp_start_layer": clamp_start,
                "read_layer": read_layer,
                "n": len(active),
                "n_seeds": len({int(row["seed"]) for row in active}),
                "current_donor_accuracy": float(
                    np.mean([bool(row["current_donor_exact"]) for row in active])
                ),
                "next_recurrent_successor_accuracy": float(
                    np.mean(
                        [bool(row["next_recurrent_successor_exact"]) for row in active]
                    )
                ),
                "next_position_reset_accuracy": float(
                    np.mean([bool(row["next_position_reset_exact"]) for row in active])
                ),
                "next_content_lookup_accuracy": float(
                    np.mean([bool(row["next_content_lookup_exact"]) for row in active])
                ),
                "mean_current_soft_shift": float(np.mean(current_shift)),
                "mean_next_soft_shift": float(np.mean(next_shift)),
                "mean_donor_aligned_current_soft_shift": float(
                    np.mean(signed_current)
                ),
                "mean_donor_aligned_next_soft_shift": float(np.mean(signed_next)),
                "current_to_next_soft_retention": through_origin_slope(
                    current_shift, next_shift
                ),
                "next_prediction_counts": {
                    str(label): int(count) for label, count in sorted(confusion.items())
                },
                "content_occurrences": sorted(
                    {int(row["content_occurrence"]) for row in active}
                ),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "cells": cells}


def _decoded_item(tokenizer: Any, encoding: Any, span: Sequence[int]) -> str:
    start, end = (int(value) for value in span)
    return tokenizer.decode(
        list(encoding.input_ids[start:end]),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--receivers", type=int, nargs="+", default=[5])
    parser.add_argument("--doses", type=int, nargs="+", default=[-1, 1])
    parser.add_argument(
        "--clamp-start-layers", type=int, nargs="+", default=[0, 4, 8, 14, 20]
    )
    parser.add_argument("--clamp-end-layer", type=int, default=23)
    parser.add_argument("--read-layers", type=int, nargs="+", default=[15, 16, 24])
    parser.add_argument("--include-content-swap", action="store_true")
    parser.add_argument("--include-structure-scrubs", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "overwrite-mechanism-scan"

    seeds = tuple(dict.fromkeys(int(value) for value in args.evaluation_seeds))
    receivers = tuple(dict.fromkeys(int(value) for value in args.receivers))
    doses = tuple(dict.fromkeys(int(value) for value in args.doses))
    starts = tuple(sorted({int(value) for value in args.clamp_start_layers}))
    clamp_end = int(args.clamp_end_layer)
    read_layers = tuple(sorted({int(value) for value in args.read_layers}))
    if not seeds or not receivers or not doses or not starts:
        raise ValueError("Every mechanism-scan factor must be nonempty")
    if any(not 1 <= receiver < 9 for receiver in receivers):
        raise ValueError("Receivers must leave a later item available for swapping")
    if any(dose == 0 for dose in doses):
        raise ValueError("State donors must differ from their receiver")
    if min(starts) < 0 or max(starts) > clamp_end:
        raise ValueError("Clamp-start layers must lie inside the clamp band")
    if clamp_end >= max(read_layers):
        raise ValueError("At least the latest read must follow the clamp band")

    probes = _load_probes(args.frozen_probes, read_layers)
    rows_by_seed = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    if clamp_end >= int(adapter.num_layers) or max(read_layers) >= int(
        adapter.num_layers
    ):
        raise ValueError("A requested clamp/read layer is outside the decoder")

    trials: list[dict[str, Any]] = []
    content_swap_skips: list[dict[str, int]] = []
    for seed in seeds:
        row = rows_by_seed[seed]
        source, blank, registry, scrub_audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        boundary_positions = {
            occurrence: select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=occurrence
            )[0]
            for occurrence in range(1, 11)
        }
        natural_layers = tuple(range(min(starts), clamp_end + 1)) + tuple(
            layer for layer in read_layers if layer > clamp_end
        )
        natural = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            tuple(boundary_positions.values()),
            layers=natural_layers,
        )
        items = tuple((int(start), int(end)) for start, end in registry.trace_items)

        for receiver in receivers:
            physical_next = receiver + 1
            donors = tuple(receiver + dose for dose in doses)
            if any(not 1 <= donor < 10 for donor in donors):
                raise ValueError("Every donor must have a recurrent successor")
            variants: list[tuple[str, int, Any, dict[str, Any]]] = [
                (
                    "original",
                    physical_next,
                    source,
                    {
                        "changed_token_count": 0,
                        "sequence_length_preserved": True,
                        "attention_mask_preserved": True,
                        "positions_preserved": True,
                    },
                )
            ]
            if bool(args.include_content_swap):
                content_occurrence = choose_future_equal_length_content(
                    registry,
                    physical_next_occurrence=physical_next,
                    # A content label may coincide with one donor successor;
                    # the opposite donor direction remains identifiable and
                    # retaining the pair avoids geometry-dependent attrition.
                    excluded_occurrences=(),
                )
                if content_occurrence is None:
                    content_swap_skips.append(
                        {"seed": seed, "receiver": receiver, "physical_next": physical_next}
                    )
                else:
                    swapped, swap_audit = swap_equal_length_item_contents(
                        source,
                        registry,
                        left_occurrence=physical_next,
                        right_occurrence=content_occurrence,
                    )
                    swap_audit = {
                        **swap_audit,
                        "physical_item_before": _decoded_item(
                            tokenizer, source, items[physical_next - 1]
                        ),
                        "content_donor_before": _decoded_item(
                            tokenizer, source, items[content_occurrence - 1]
                        ),
                        "physical_item_after": _decoded_item(
                            tokenizer, swapped, items[physical_next - 1]
                        ),
                    }
                    variants.append(
                        ("future_equal_length_swap", content_occurrence, swapped, swap_audit)
                    )
            if bool(args.include_structure_scrubs):
                for variant, scrubbed, structure_audit in build_structure_scrub_variants(
                    source,
                    blank,
                    registry,
                    physical_next_occurrence=physical_next,
                ):
                    variants.append(
                        (variant, physical_next, scrubbed, structure_audit)
                    )

            for content_variant, content_occurrence, encoding, content_audit in variants:
                if content_variant == "original":
                    clean_states = {
                        layer: natural[layer][
                            [receiver - 1, physical_next - 1]
                        ]
                        for layer in read_layers
                    }
                else:
                    clean_states = capture_decoder_block_input_states(
                        model,
                        adapter,
                        encoding,
                        (
                            boundary_positions[receiver],
                            boundary_positions[physical_next],
                        ),
                        layers=read_layers,
                    )
                clean_decoded = {
                    layer: (
                        decode_count_probe(probes[layer], clean_states[layer][0].numpy()),
                        decode_count_probe(probes[layer], clean_states[layer][1].numpy()),
                    )
                    for layer in read_layers
                }

                for donor, dose in zip(donors, doses):
                    for clamp_start in starts:
                        targets = {
                            layer: natural[layer][donor - 1].numpy()
                            for layer in range(clamp_start, clamp_end + 1)
                        }
                        captured, intervention_audit = carrier_capture_layer_positions(
                            model,
                            adapter,
                            encoding,
                            boundary_position=boundary_positions[receiver],
                            boundary_targets=targets,
                            kv_directions={},
                            read_positions=(
                                boundary_positions[receiver],
                                boundary_positions[physical_next],
                            ),
                            read_layers=read_layers,
                        )
                        for read_layer in read_layers:
                            current = decode_count_probe(
                                probes[read_layer], captured[read_layer][0].numpy()
                            )
                            later = decode_count_probe(
                                probes[read_layer], captured[read_layer][1].numpy()
                            )
                            clean_current, clean_next = clean_decoded[read_layer]
                            current_soft = float(
                                current["probe_softmax_expected_count"]
                            )
                            next_soft = float(later["probe_softmax_expected_count"])
                            clean_current_soft = float(
                                clean_current["probe_softmax_expected_count"]
                            )
                            clean_next_soft = float(
                                clean_next["probe_softmax_expected_count"]
                            )
                            trials.append(
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "model_label": str(args.model),
                                    "seed": seed,
                                    "request_id": str(row["request_id"]),
                                    "receiver": receiver,
                                    "donor": donor,
                                    "dose": dose,
                                    "physical_next_occurrence": physical_next,
                                    "content_variant": content_variant,
                                    "content_occurrence": content_occurrence,
                                    "recurrent_successor_occurrence": donor + 1,
                                    "clamp_start_layer": clamp_start,
                                    "clamp_end_layer": clamp_end,
                                    "read_layer": read_layer,
                                    "clean_current_prediction": int(
                                        clean_current["probe_prediction"]
                                    ),
                                    "clean_next_prediction": int(
                                        clean_next["probe_prediction"]
                                    ),
                                    "current_prediction": int(current["probe_prediction"]),
                                    "next_prediction": int(later["probe_prediction"]),
                                    "clean_current_soft": clean_current_soft,
                                    "clean_next_soft": clean_next_soft,
                                    "current_soft": current_soft,
                                    "next_soft": next_soft,
                                    "current_soft_shift": current_soft
                                    - clean_current_soft,
                                    "next_soft_shift": next_soft - clean_next_soft,
                                    "current_donor_exact": bool(
                                        int(current["probe_prediction"]) == donor
                                    ),
                                    "next_recurrent_successor_exact": bool(
                                        int(later["probe_prediction"]) == donor + 1
                                    ),
                                    "next_position_reset_exact": bool(
                                        int(later["probe_prediction"]) == physical_next
                                    ),
                                    "next_content_lookup_exact": bool(
                                        int(later["probe_prediction"])
                                        == content_occurrence
                                    ),
                                    "probe_scores_current": current["probe_scores"],
                                    "probe_scores_next": later["probe_scores"],
                                    "scrub_construction": scrub_audit["construction"],
                                    "content_audit": content_audit,
                                    "intervention_audit": intervention_audit,
                                    "input_tokens_changed_by_state_intervention": False,
                                    "diagnostic_suffix_used": False,
                                }
                            )
        print(f"[overwrite-mechanism] seed={seed} complete", flush=True)

    summary = {
        **summarize_trials(trials),
        "model_label": str(args.model),
        "evaluation_seeds": list(seeds),
        "receivers": list(receivers),
        "doses": list(doses),
        "clamp_start_layers": list(starts),
        "clamp_end_layer": clamp_end,
        "read_layers": list(read_layers),
        "include_content_swap": bool(args.include_content_swap),
        "include_structure_scrubs": bool(args.include_structure_scrubs),
        "trial_count": len(trials),
        "content_swap_skips": content_swap_skips,
    }
    _atomic_jsonl(args.output, trials)
    _atomic_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
