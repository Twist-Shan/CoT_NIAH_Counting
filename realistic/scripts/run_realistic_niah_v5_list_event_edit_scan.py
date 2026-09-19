#!/usr/bin/env python3
"""Calibrate native boundary count against inserted/deleted valid list events.

The marker-scrubbed trace contains natural unnumbered city-score items.  Before
the original item r+1, this scan either inserts a copied complete item plus its
separator, or deletes one complete prior item plus its separator.  It then
reads the boundary after the same original item r+1.  Event recount predicts
r+2 / r / r+1 for insert / delete / original, whereas content identity keeps
the original label r+1.  A donor-state clamp adds a separate recurrent-state
prediction: donor + the number of intervening valid item events.
"""

from __future__ import annotations

import argparse
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
from scripts.run_realistic_niah_v5_overwrite_mechanism_scan import (  # noqa: E402
    _load_probes,
)
from scripts.run_realistic_niah_v5_unified_carrier_transition import (  # noqa: E402
    _read_rows,
)


SCHEMA_VERSION = "list_event_edit_scan_v1"


def splice_encoding(
    encoding: Any,
    *,
    start: int,
    end: int,
    replacement_ids: Sequence[int],
    replacement_mask: Sequence[int],
) -> tuple[Any, int]:
    """Replace one post-prompt token interval and update causal query geometry."""

    left = int(start)
    right = int(end)
    if not int(encoding.prompt_token_count) <= left <= right <= int(
        encoding.query_position
    ):
        raise ValueError("List-event splice must lie after the prompt and before query")
    ids = tuple(int(value) for value in encoding.input_ids)
    mask = tuple(int(value) for value in encoding.attention_mask)
    inserted_ids = tuple(int(value) for value in replacement_ids)
    inserted_mask = tuple(int(value) for value in replacement_mask)
    if len(inserted_ids) != len(inserted_mask):
        raise ValueError("Replacement ids and mask lengths disagree")
    output_ids = ids[:left] + inserted_ids + ids[right:]
    output_mask = mask[:left] + inserted_mask + mask[right:]
    delta = len(inserted_ids) - (right - left)
    if len(output_ids) != len(output_mask) or len(output_ids) != len(ids) + delta:
        raise RuntimeError("List-event splice produced inconsistent geometry")
    return (
        replace(
            encoding,
            input_ids=output_ids,
            attention_mask=output_mask,
            query_position=int(encoding.query_position) + delta,
            # These semantic spans describe the original trace.  Clear the
            # trace-local copies instead of silently exposing stale geometry;
            # prompt record spans remain valid because every edit is post-prompt.
            trace_item_spans=(),
            slot_spans=(),
            needle_spans=(),
        ),
        delta,
    )


def build_list_event_variants(
    encoding: Any,
    neutral_encoding: Any,
    registry: Any,
    *,
    receiver: int,
    current_boundary: int,
    target_boundary: int,
    insert_source_occurrence: int,
    delete_occurrence: int,
) -> tuple[dict[str, Any], ...]:
    """Return original, +1-valid-event, and -1-valid-event geometries."""

    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    active_receiver = int(receiver)
    physical_target = active_receiver + 1
    insert_source = int(insert_source_occurrence)
    deleted = int(delete_occurrence)
    if not 1 <= deleted < active_receiver < physical_target <= len(items):
        raise ValueError("Delete/receiver/target occurrences are misordered")
    if not 1 <= insert_source < len(items):
        raise ValueError("Insert source must have a following separator")

    ids = tuple(int(value) for value in encoding.input_ids)
    mask = tuple(int(value) for value in encoding.attention_mask)
    neutral_ids = tuple(int(value) for value in neutral_encoding.input_ids)
    if len(neutral_ids) != len(ids) or tuple(neutral_encoding.attention_mask) != mask:
        raise ValueError("Neutral insertion control must preserve source geometry")
    insertion_point = items[physical_target - 1][0]
    insert_start = items[insert_source - 1][0]
    insert_item_end = items[insert_source - 1][1]
    insert_end = items[insert_source][0]
    source_segment = ids[insert_start:insert_end]
    source_segment_mask = mask[insert_start:insert_end]
    if insert_item_end - insert_start < 2 or insert_item_end > insert_end:
        raise ValueError("Inserted item has invalid token geometry")
    marker_positions = {
        position
        for start, end in registry.trace_markers
        for position in range(int(start), int(end))
        if insert_start <= position < insert_item_end
    }
    marker_local = {position - insert_start for position in marker_positions}
    item_local = set(range(insert_item_end - insert_start))
    payload_local = item_local - marker_local
    if not marker_local or not payload_local:
        raise ValueError("Inserted item lacks registered marker or payload tokens")
    neutral_segment = list(source_segment)
    # Positions outside the registered item (when tokenization exposes a
    # separate separator) always remain natural and identical across controls.
    for local_position in item_local:
        neutral_segment[local_position] = neutral_ids[insert_start + local_position]
    markerless_segment = list(source_segment)
    for local_position in marker_local:
        markerless_segment[local_position] = neutral_ids[
            insert_start + local_position
        ]
    payloadless_segment = list(source_segment)
    for local_position in payload_local:
        payloadless_segment[local_position] = neutral_ids[
            insert_start + local_position
        ]
    insertion_segments = {
        "insert_valid_item": source_segment,
        "insert_markerless_valid_payload": tuple(markerless_segment),
        "insert_marker_neutral_payload": tuple(payloadless_segment),
        "insert_neutral_line": tuple(neutral_segment),
    }
    inserted_variants: dict[str, tuple[Any, int]] = {}
    for label, segment in insertion_segments.items():
        inserted_variants[label] = splice_encoding(
            encoding,
            start=insertion_point,
            end=insertion_point,
            replacement_ids=segment,
            replacement_mask=source_segment_mask,
        )
    insert_deltas = {delta for _value, delta in inserted_variants.values()}
    if len(insert_deltas) != 1:
        raise RuntimeError("Insertion controls do not have identical token geometry")
    insert_delta = next(iter(insert_deltas))
    delete_start = items[deleted - 1][0]
    delete_end = items[deleted][0]
    removed, delete_delta = splice_encoding(
        encoding,
        start=delete_start,
        end=delete_end,
        replacement_ids=(),
        replacement_mask=(),
    )
    if insert_delta <= 0 or delete_delta >= 0:
        raise RuntimeError("Insert/delete event deltas have the wrong sign")
    base_variants: list[dict[str, Any]] = [
        {
            "event_variant": "original",
            "encoding": encoding,
            "current_boundary": int(current_boundary),
            "target_boundary": int(target_boundary),
            "event_count_target": physical_target,
            "transition_horizon": 1,
            "token_delta": 0,
        },
        {
            "event_variant": "delete_prior_valid_item",
            "encoding": removed,
            "current_boundary": int(current_boundary) + delete_delta,
            "target_boundary": int(target_boundary) + delete_delta,
            "event_count_target": physical_target - 1,
            "transition_horizon": 1,
            "token_delta": delete_delta,
            "deleted_occurrence": deleted,
        },
    ]
    for label, (inserted, delta) in inserted_variants.items():
        valid_event = label == "insert_valid_item"
        base_variants.append(
            {
                "event_variant": label,
                "encoding": inserted,
                "current_boundary": int(current_boundary),
                "target_boundary": int(target_boundary) + delta,
                "event_count_target": physical_target + int(valid_event),
                "transition_horizon": 1 + int(valid_event),
                "token_delta": delta,
                "insert_source_occurrence": insert_source,
                "inserted_marker_valid": label
                in {"insert_valid_item", "insert_marker_neutral_payload"},
                "inserted_payload_valid": label
                in {"insert_valid_item", "insert_markerless_valid_payload"},
                "matched_insertion_token_delta": insert_delta,
                "inserted_marker_token_count": len(marker_local),
                "inserted_payload_token_count": len(payload_local),
                "separate_separator_token_count": insert_end - insert_item_end,
            }
        )
    return tuple(base_variants)


def summarize_trials(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["event_variant"]), int(row["read_layer"]))].append(row)
    cells = []
    for (variant, layer), active in sorted(grouped.items()):
        current_shift = [float(row["current_soft_shift"]) for row in active]
        next_shift = [float(row["next_soft_shift"]) for row in active]
        cells.append(
            {
                "event_variant": variant,
                "read_layer": layer,
                "n": len(active),
                "n_seeds": len({int(row["seed"]) for row in active}),
                "clean_event_count_accuracy": float(
                    np.mean([bool(row["clean_event_count_exact"]) for row in active])
                ),
                "current_donor_accuracy": float(
                    np.mean([bool(row["current_donor_exact"]) for row in active])
                ),
                "next_event_count_accuracy": float(
                    np.mean([bool(row["next_event_count_exact"]) for row in active])
                ),
                "next_recurrent_accuracy": float(
                    np.mean([bool(row["next_recurrent_exact"]) for row in active])
                ),
                "next_original_identity_accuracy": float(
                    np.mean([bool(row["next_original_identity_exact"]) for row in active])
                ),
                "mean_donor_aligned_next_shift": float(
                    np.mean(
                        [
                            float(row["dose"]) * float(row["next_soft_shift"])
                            for row in active
                        ]
                    )
                ),
                "current_to_next_retention": through_origin_slope(
                    current_shift, next_shift
                ),
                "next_prediction_counts": {
                    str(label): int(count)
                    for label, count in sorted(
                        Counter(int(row["next_prediction"]) for row in active).items()
                    )
                },
                "clean_target_prediction_counts": {
                    str(label): int(count)
                    for label, count in sorted(
                        Counter(
                            int(row["clean_target_prediction"]) for row in active
                        ).items()
                    )
                },
            }
        )
    return {"schema_version": SCHEMA_VERSION, "cells": cells}


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
    parser.add_argument("--receiver", type=int, default=5)
    parser.add_argument("--doses", type=int, nargs="+", default=[-1, 1])
    parser.add_argument("--clamp-start-layer", type=int, default=0)
    parser.add_argument("--clamp-end-layer", type=int, default=23)
    parser.add_argument("--read-layers", type=int, nargs="+", default=[15, 16, 24])
    parser.add_argument("--insert-source-occurrence", type=int, default=4)
    parser.add_argument("--delete-occurrence", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "list-event-edit-scan"

    seeds = tuple(dict.fromkeys(int(value) for value in args.evaluation_seeds))
    receiver = int(args.receiver)
    doses = tuple(dict.fromkeys(int(value) for value in args.doses))
    donors = tuple(receiver + dose for dose in doses)
    clamp_start = int(args.clamp_start_layer)
    clamp_end = int(args.clamp_end_layer)
    read_layers = tuple(sorted({int(value) for value in args.read_layers}))
    if not seeds or any(dose == 0 for dose in doses):
        raise ValueError("Event edit scan needs seeds and nonzero donor doses")
    if not 2 <= receiver <= 8 or any(not 1 <= donor < 10 for donor in donors):
        raise ValueError("Receiver/donor geometry is outside the ten-item list")
    if not 0 <= clamp_start <= clamp_end < max(read_layers):
        raise ValueError("Clamp band must precede the latest read layer")

    probes = _load_probes(args.frozen_probes, read_layers)
    source_rows = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    trials: list[dict[str, Any]] = []
    clamp_layers = tuple(range(clamp_start, clamp_end + 1))
    natural_layers = tuple(sorted(set(clamp_layers) | set(read_layers)))

    for seed in seeds:
        row = source_rows[seed]
        source, blank, registry, scrub_audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        boundaries = {
            occurrence: select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=occurrence
            )[0]
            for occurrence in range(1, 11)
        }
        natural = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            tuple(boundaries.values()),
            layers=natural_layers,
        )
        variants = build_list_event_variants(
            source,
            blank,
            registry,
            receiver=receiver,
            current_boundary=boundaries[receiver],
            target_boundary=boundaries[receiver + 1],
            insert_source_occurrence=int(args.insert_source_occurrence),
            delete_occurrence=int(args.delete_occurrence),
        )
        for variant in variants:
            encoding = variant["encoding"]
            current_position = int(variant["current_boundary"])
            target_position = int(variant["target_boundary"])
            event_target = int(variant["event_count_target"])
            horizon = int(variant["transition_horizon"])
            clean = capture_decoder_block_input_states(
                model,
                adapter,
                encoding,
                (current_position, target_position),
                layers=read_layers,
            )
            clean_decoded = {
                layer: (
                    decode_count_probe(probes[layer], clean[layer][0].numpy()),
                    decode_count_probe(probes[layer], clean[layer][1].numpy()),
                )
                for layer in read_layers
            }
            for donor, dose in zip(donors, doses):
                targets = {
                    layer: natural[layer][donor - 1].numpy() for layer in clamp_layers
                }
                captured, audit = carrier_capture_layer_positions(
                    model,
                    adapter,
                    encoding,
                    boundary_position=current_position,
                    boundary_targets=targets,
                    kv_directions={},
                    read_positions=(current_position, target_position),
                    read_layers=read_layers,
                )
                recurrent_target = donor + horizon
                for layer in read_layers:
                    current = decode_count_probe(
                        probes[layer], captured[layer][0].numpy()
                    )
                    later = decode_count_probe(probes[layer], captured[layer][1].numpy())
                    clean_current, clean_target = clean_decoded[layer]
                    current_soft = float(current["probe_softmax_expected_count"])
                    next_soft = float(later["probe_softmax_expected_count"])
                    clean_current_soft = float(
                        clean_current["probe_softmax_expected_count"]
                    )
                    clean_target_soft = float(
                        clean_target["probe_softmax_expected_count"]
                    )
                    trials.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "model_label": str(args.model),
                            "seed": seed,
                            "request_id": str(row["request_id"]),
                            "event_variant": str(variant["event_variant"]),
                            "token_delta": int(variant["token_delta"]),
                            "receiver": receiver,
                            "donor": donor,
                            "dose": dose,
                            "transition_horizon": horizon,
                            "event_count_target": event_target,
                            "original_identity_target": receiver + 1,
                            "recurrent_target": recurrent_target,
                            "current_boundary": current_position,
                            "target_boundary": target_position,
                            "clamp_layers": list(clamp_layers),
                            "read_layer": layer,
                            "clean_current_prediction": int(
                                clean_current["probe_prediction"]
                            ),
                            "clean_target_prediction": int(
                                clean_target["probe_prediction"]
                            ),
                            "current_prediction": int(current["probe_prediction"]),
                            "next_prediction": int(later["probe_prediction"]),
                            "clean_current_soft": clean_current_soft,
                            "clean_target_soft": clean_target_soft,
                            "current_soft": current_soft,
                            "next_soft": next_soft,
                            "current_soft_shift": current_soft - clean_current_soft,
                            "next_soft_shift": next_soft - clean_target_soft,
                            "clean_event_count_exact": bool(
                                int(clean_target["probe_prediction"]) == event_target
                            ),
                            "current_donor_exact": bool(
                                int(current["probe_prediction"]) == donor
                            ),
                            "next_event_count_exact": bool(
                                int(later["probe_prediction"]) == event_target
                            ),
                            "next_recurrent_exact": bool(
                                int(later["probe_prediction"]) == recurrent_target
                            ),
                            "next_original_identity_exact": bool(
                                int(later["probe_prediction"]) == receiver + 1
                            ),
                            "probe_scores_current": current["probe_scores"],
                            "probe_scores_next": later["probe_scores"],
                            "scrub_construction": scrub_audit["construction"],
                            "intervention_audit": audit,
                            "variant_audit": {
                                key: value
                                for key, value in variant.items()
                                if key != "encoding"
                            },
                            "state_intervention_changed_tokens": False,
                            "diagnostic_suffix_used": False,
                        }
                    )
        print(f"[list-event-edit] seed={seed} complete", flush=True)

    summary = {
        **summarize_trials(trials),
        "model_label": str(args.model),
        "evaluation_seeds": list(seeds),
        "receiver": receiver,
        "doses": list(doses),
        "clamp_layers": list(clamp_layers),
        "read_layers": list(read_layers),
        "insert_source_occurrence": int(args.insert_source_occurrence),
        "delete_occurrence": int(args.delete_occurrence),
        "trial_count": len(trials),
    }
    _atomic_jsonl(args.output, trials)
    _atomic_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
