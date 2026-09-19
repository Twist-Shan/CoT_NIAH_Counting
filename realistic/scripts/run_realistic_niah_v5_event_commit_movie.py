#!/usr/bin/env python3
"""Trace a tentative-list-event signal from marker through boundary commit.

The scan reuses the matched-token insertion factorial from the overwrite audit.
For each variant it decodes the frozen boundary-count probe at every token from
the receiver boundary through the next original item.  The complete valid-item
variant is additionally run with donor counts 4 and 6 clamped at the receiver,
so we can locate where the donor separation disappears.

Intermediate-token probe values are descriptive because the probe was trained
at post-item boundaries.  The current, inserted-event, and target boundaries are
the formal read sites.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


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
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_list_event_edit_scan import (  # noqa: E402
    build_list_event_variants,
)
from scripts.run_realistic_niah_v5_overwrite_mechanism_scan import (  # noqa: E402
    _load_probes,
)
from scripts.run_realistic_niah_v5_unified_carrier_transition import (  # noqa: E402
    _read_rows,
)


SCHEMA_VERSION = "event_commit_movie_v1"
INSERT_VARIANTS = (
    "insert_valid_item",
    "insert_markerless_valid_payload",
    "insert_marker_neutral_payload",
    "insert_neutral_line",
)
MOVIE_VARIANTS = ("original", *INSERT_VARIANTS)


def _positions_in_spans(
    spans: Sequence[Sequence[int]], *, start: int, end: int
) -> set[int]:
    return {
        position
        for span_start, span_end in spans
        for position in range(int(span_start), int(span_end))
        if start <= position < end
    }


def _middle(positions: Sequence[int]) -> int:
    if not positions:
        raise ValueError("A semantic movie region is empty")
    return int(positions[(len(positions) - 1) // 2])


def build_event_movie_geometry(
    variant: Mapping[str, Any],
    registry: Any,
    boundaries: Mapping[int, int],
    *,
    receiver: int,
    insert_source_occurrence: int,
) -> dict[str, Any]:
    """Assign semantic roles and landmarks to the receiver-to-target path."""

    label = str(variant["event_variant"])
    if label not in MOVIE_VARIANTS:
        raise ValueError(f"Unsupported movie variant: {label}")
    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    markers = tuple((int(start), int(end)) for start, end in registry.trace_markers)
    physical_target = int(receiver) + 1
    target_start_original, target_end_original = items[physical_target - 1]
    target_marker_original = _positions_in_spans(
        markers, start=target_start_original, end=target_end_original
    )
    target_payload_original = sorted(
        set(range(target_start_original, target_end_original))
        - target_marker_original
    )
    if not target_marker_original or not target_payload_original:
        raise ValueError("Target item lacks registered marker or payload tokens")

    delta = int(variant["token_delta"])
    current_boundary = int(variant["current_boundary"])
    target_boundary = int(variant["target_boundary"])
    target_start = target_start_original + delta
    target_end = target_end_original + delta
    target_markers = {position + delta for position in target_marker_original}
    target_payload = {position + delta for position in target_payload_original}

    inserted = label in INSERT_VARIANTS
    inserted_start: int | None = None
    inserted_item_end: int | None = None
    inserted_segment_end: int | None = None
    inserted_boundary: int | None = None
    inserted_markers: set[int] = set()
    inserted_payload: set[int] = set()
    if inserted:
        source_index = int(insert_source_occurrence) - 1
        source_start, source_item_end = items[source_index]
        source_segment_end = items[source_index + 1][0]
        source_markers = _positions_in_spans(
            markers, start=source_start, end=source_item_end
        )
        source_payload = sorted(
            set(range(source_start, source_item_end)) - source_markers
        )
        if not source_markers or not source_payload:
            raise ValueError("Insertion source lacks marker or payload tokens")
        inserted_start = target_start_original
        inserted_item_end = inserted_start + (source_item_end - source_start)
        inserted_segment_end = inserted_start + (source_segment_end - source_start)
        inserted_boundary = inserted_start + (
            int(boundaries[int(insert_source_occurrence)]) - source_start
        )
        inserted_markers = {
            inserted_start + (position - source_start) for position in source_markers
        }
        inserted_payload = {
            inserted_start + (position - source_start) for position in source_payload
        }
        if inserted_segment_end != target_start:
            raise RuntimeError("Matched insertion geometry does not meet target item")
        if not inserted_start <= inserted_boundary < inserted_segment_end:
            raise RuntimeError("Inserted event boundary is outside its copied segment")

    if not current_boundary <= target_boundary:
        raise ValueError("Movie path is reversed")
    path = tuple(range(current_boundary, target_boundary + 1))
    roles: dict[int, str] = {}
    for position in path:
        if inserted and position in inserted_markers:
            role = "inserted_marker"
        elif inserted and position in inserted_payload:
            role = "inserted_payload"
        elif (
            inserted
            and inserted_item_end is not None
            and inserted_segment_end is not None
            and inserted_item_end <= position < inserted_segment_end
        ):
            role = "inserted_separator"
        elif position in target_markers:
            role = "target_marker"
        elif position in target_payload:
            role = "target_payload"
        elif target_end <= position <= target_boundary:
            role = "target_separator"
        else:
            role = "between_events"
        roles[position] = role
    roles[current_boundary] = "current_boundary"
    if inserted_boundary is not None:
        roles[inserted_boundary] = "inserted_event_boundary"
    roles[target_boundary] = "target_boundary"

    landmarks: dict[str, int] = {"current_boundary": current_boundary}
    if inserted:
        assert inserted_item_end is not None and inserted_boundary is not None
        landmarks.update(
            {
                "inserted_marker_end": max(inserted_markers),
                "inserted_payload_mid": _middle(sorted(inserted_payload)),
                "inserted_item_end": inserted_item_end - 1,
                "inserted_event_boundary": inserted_boundary,
            }
        )
    landmarks.update(
        {
            "target_marker_end": max(target_markers),
            "target_payload_mid": _middle(sorted(target_payload)),
            "target_item_end": target_end - 1,
            "target_boundary": target_boundary,
        }
    )
    position_landmarks: dict[int, list[str]] = defaultdict(list)
    for name, position in landmarks.items():
        position_landmarks[position].append(name)
    return {
        "path_positions": path,
        "roles": roles,
        "landmarks": landmarks,
        "position_landmarks": dict(position_landmarks),
        "inserted_item_valid": label == "insert_valid_item",
        "inserted_marker_valid": label
        in {"insert_valid_item", "insert_marker_neutral_payload"},
        "inserted_payload_valid": label
        in {"insert_valid_item", "insert_markerless_valid_payload"},
    }


def _prediction_counts(values: Sequence[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def summarize_event_movie(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize clean landmark states and valid-item donor-pair decay."""

    clean_groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    donor_index: dict[tuple[int, int, str, int], Mapping[str, Any]] = {}
    for row in rows:
        for landmark in row["landmarks"]:
            if str(row["condition"]) == "clean":
                clean_groups[
                    (str(row["event_variant"]), int(row["read_layer"]), str(landmark))
                ].append(row)
            elif str(row["event_variant"]) == "insert_valid_item":
                key = (
                    int(row["seed"]),
                    int(row["read_layer"]),
                    str(landmark),
                    int(row["donor"]),
                )
                if key in donor_index:
                    raise ValueError(f"Duplicate donor landmark row: {key}")
                donor_index[key] = row

    clean_cells = []
    for (variant, layer, landmark), active in sorted(clean_groups.items()):
        predictions = [int(row["probe_prediction"]) for row in active]
        clean_cells.append(
            {
                "event_variant": variant,
                "read_layer": layer,
                "landmark": landmark,
                "n_seeds": len(active),
                "prediction_counts": _prediction_counts(predictions),
                "mean_soft_count": fmean(
                    float(row["probe_softmax_expected_count"]) for row in active
                ),
            }
        )

    pair_groups: dict[tuple[int, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    seeds_layers_landmarks = sorted(
        {(seed, layer, landmark) for seed, layer, landmark, donor in donor_index}
    )
    for seed, layer, landmark in seeds_layers_landmarks:
        low = donor_index.get((seed, layer, landmark, 4))
        high = donor_index.get((seed, layer, landmark, 6))
        if low is None or high is None:
            raise ValueError(
                f"Incomplete donor pair at seed={seed}, layer={layer}, landmark={landmark}"
            )
        pair_groups[(layer, landmark)].append((low, high))

    current_separation: dict[tuple[int, int], float] = {}
    for (layer, landmark), pairs in pair_groups.items():
        if landmark != "current_boundary":
            continue
        for low, high in pairs:
            current_separation[(int(low["seed"]), layer)] = float(
                high["probe_softmax_expected_count"]
            ) - float(low["probe_softmax_expected_count"])

    donor_cells = []
    for (layer, landmark), pairs in sorted(pair_groups.items()):
        separations = [
            float(high["probe_softmax_expected_count"])
            - float(low["probe_softmax_expected_count"])
            for low, high in pairs
        ]
        retention = []
        for (low, high), separation in zip(pairs, separations):
            baseline = current_separation[(int(low["seed"]), layer)]
            if abs(baseline) > 1e-12:
                retention.append(separation / baseline)
        donor_cells.append(
            {
                "read_layer": layer,
                "landmark": landmark,
                "n_seed_pairs": len(pairs),
                "donor_invariant_count": sum(
                    int(low["probe_prediction"]) == int(high["probe_prediction"])
                    for low, high in pairs
                ),
                "recurrent_separation_2_count": sum(
                    int(high["probe_prediction"]) - int(low["probe_prediction"]) == 2
                    for low, high in pairs
                ),
                "mean_soft_donor_separation": fmean(separations),
                "mean_within_seed_retention": fmean(retention) if retention else None,
                "donor4_prediction_counts": _prediction_counts(
                    [int(low["probe_prediction"]) for low, _high in pairs]
                ),
                "donor6_prediction_counts": _prediction_counts(
                    [int(high["probe_prediction"]) for _low, high in pairs]
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "clean_landmark_cells": clean_cells,
        "valid_item_donor_pair_cells": donor_cells,
    }


def _append_capture_rows(
    output: list[dict[str, Any]],
    *,
    seed: int,
    request_id: str,
    variant: Mapping[str, Any],
    geometry: Mapping[str, Any],
    captured: Mapping[int, Any],
    probes: Mapping[int, Any],
    tokenizer: Any,
    condition: str,
    donor: int | None,
    intervention_audit: Mapping[str, Any] | None,
) -> None:
    encoding = variant["encoding"]
    positions = tuple(int(value) for value in geometry["path_positions"])
    for layer, states in captured.items():
        for offset, position in enumerate(positions):
            decoded = decode_count_probe(probes[int(layer)], states[offset].numpy())
            landmarks = list(geometry["position_landmarks"].get(position, ()))
            token_id = int(encoding.input_ids[position])
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "seed": int(seed),
                    "request_id": str(request_id),
                    "event_variant": str(variant["event_variant"]),
                    "condition": str(condition),
                    "donor": int(donor) if donor is not None else None,
                    "read_layer": int(layer),
                    "position": position,
                    "position_offset": position - positions[0],
                    "role": str(geometry["roles"][position]),
                    "landmarks": landmarks,
                    "formal_boundary_read": bool(
                        any(
                            name
                            in {
                                "current_boundary",
                                "inserted_event_boundary",
                                "target_boundary",
                            }
                            for name in landmarks
                        )
                    ),
                    "token_id": token_id,
                    "token_text": tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    ),
                    "probe_prediction": int(decoded["probe_prediction"]),
                    "probe_softmax_expected_count": float(
                        decoded["probe_softmax_expected_count"]
                    ),
                    "probe_scores": decoded["probe_scores"],
                    "token_delta": int(variant["token_delta"]),
                    "event_count_target": int(variant["event_count_target"]),
                    "inserted_item_valid": bool(geometry["inserted_item_valid"]),
                    "inserted_marker_valid": bool(geometry["inserted_marker_valid"]),
                    "inserted_payload_valid": bool(geometry["inserted_payload_valid"]),
                    "state_intervention_changed_tokens": False,
                    "diagnostic_suffix_used": False,
                    "intervention_audit": intervention_audit,
                }
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
    parser.add_argument("--receiver", type=int, default=5)
    parser.add_argument("--read-layers", type=int, nargs="+", default=[15, 16, 24])
    parser.add_argument("--clamp-start-layer", type=int, default=0)
    parser.add_argument("--clamp-end-layer", type=int, default=23)
    parser.add_argument("--insert-source-occurrence", type=int, default=4)
    parser.add_argument("--delete-occurrence", type=int, default=3)
    parser.add_argument("--skip-donor-pair", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "event-commit-movie"

    seeds = tuple(dict.fromkeys(int(value) for value in args.evaluation_seeds))
    receiver = int(args.receiver)
    read_layers = tuple(sorted({int(value) for value in args.read_layers}))
    clamp_layers = tuple(
        range(int(args.clamp_start_layer), int(args.clamp_end_layer) + 1)
    )
    if not seeds or not 2 <= receiver <= 8:
        raise ValueError("Event movie seed/receiver geometry is invalid")
    if not clamp_layers or clamp_layers[-1] >= max(read_layers):
        raise ValueError("Clamp band must end before the latest read layer")

    probes = _load_probes(args.frozen_probes, read_layers)
    source_rows = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    trials: list[dict[str, Any]] = []
    natural_layers = tuple(sorted(set(clamp_layers) | set(read_layers)))

    for seed in seeds:
        row = source_rows[seed]
        source, blank, registry, _scrub_audit = build_diagnostic_bases(
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
        variants = {
            str(variant["event_variant"]): variant
            for variant in build_list_event_variants(
                source,
                blank,
                registry,
                receiver=receiver,
                current_boundary=boundaries[receiver],
                target_boundary=boundaries[receiver + 1],
                insert_source_occurrence=int(args.insert_source_occurrence),
                delete_occurrence=int(args.delete_occurrence),
            )
            if str(variant["event_variant"]) in MOVIE_VARIANTS
        }
        if set(variants) != set(MOVIE_VARIANTS):
            raise RuntimeError("Event movie variant construction is incomplete")

        geometry_by_variant = {
            label: build_event_movie_geometry(
                variant,
                registry,
                boundaries,
                receiver=receiver,
                insert_source_occurrence=int(args.insert_source_occurrence),
            )
            for label, variant in variants.items()
        }
        for label in MOVIE_VARIANTS:
            variant = variants[label]
            geometry = geometry_by_variant[label]
            clean = capture_decoder_block_input_states(
                model,
                adapter,
                variant["encoding"],
                geometry["path_positions"],
                layers=read_layers,
            )
            _append_capture_rows(
                trials,
                seed=seed,
                request_id=str(row["request_id"]),
                variant=variant,
                geometry=geometry,
                captured=clean,
                probes=probes,
                tokenizer=tokenizer,
                condition="clean",
                donor=None,
                intervention_audit=None,
            )

        if not args.skip_donor_pair:
            natural = capture_decoder_block_input_states(
                model,
                adapter,
                source,
                tuple(boundaries.values()),
                layers=natural_layers,
            )
            variant = variants["insert_valid_item"]
            geometry = geometry_by_variant["insert_valid_item"]
            for donor in (4, 6):
                targets = {
                    layer: natural[layer][donor - 1].numpy()
                    for layer in clamp_layers
                }
                captured, audit = carrier_capture_layer_positions(
                    model,
                    adapter,
                    variant["encoding"],
                    boundary_position=int(variant["current_boundary"]),
                    boundary_targets=targets,
                    kv_directions={},
                    read_positions=geometry["path_positions"],
                    read_layers=read_layers,
                )
                _append_capture_rows(
                    trials,
                    seed=seed,
                    request_id=str(row["request_id"]),
                    variant=variant,
                    geometry=geometry,
                    captured=captured,
                    probes=probes,
                    tokenizer=tokenizer,
                    condition="donor_clamp",
                    donor=donor,
                    intervention_audit=audit,
                )
        print(f"[event-commit-movie] seed={seed} complete", flush=True)

    summary = {
        **summarize_event_movie(trials),
        "model_label": str(args.model),
        "evaluation_seeds": list(seeds),
        "receiver": receiver,
        "read_layers": list(read_layers),
        "clamp_layers": list(clamp_layers),
        "movie_variants": list(MOVIE_VARIANTS),
        "donor_pair_included": not bool(args.skip_donor_pair),
        "intermediate_probe_values_are_descriptive": True,
        "trial_count": len(trials),
    }
    _atomic_jsonl(args.output, trials)
    _atomic_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
