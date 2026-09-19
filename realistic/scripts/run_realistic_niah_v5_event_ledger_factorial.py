#!/usr/bin/env python3
"""Test whether independent marker-keyed cache entries support late counting.

The textual arm is a same-geometry 2^3 factorial: three copied list events have
fixed payloads and separators while each event marker is independently valid or
markerless.  The cache arm materializes the 000 and 111 histories, then moves
arbitrary subsets of the three same-position cache entries from 111 into 000.

The assay is intentionally asymmetric.  It asks whether marker cache fields are
sufficient to reconstruct graded late readout, not whether the full 111 history
can be reconstructed by an unconstrained cache patch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np
import torch


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
from realistic_niah_v5.causal import sign_flip_pvalue  # noqa: E402
from realistic_niah_v5.event_cache_splice import (  # noqa: E402
    clone_cache,
    splice_cache_positions,
)
from realistic_niah_v5.event_ledger import build_marker_event_factorial  # noqa: E402
from scripts.run_realistic_niah_v5_boundary_equivariance import (  # noqa: E402
    decode_count_probe,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_event_cache_splice import (  # noqa: E402
    _forward_from_cache,
    advance_event_cache,
    prefill_common_prefix,
    state_axis_metrics,
    state_equivalence,
)
from scripts.run_realistic_niah_v5_overwrite_mechanism_scan import (  # noqa: E402
    _load_probes,
)
from scripts.run_realistic_niah_v5_unified_carrier_transition import (  # noqa: E402
    _read_rows,
)


SCHEMA_VERSION = "event_ledger_factorial_v1"
PRIMARY_READ_LAYER = 24
PRIMARY_LANDMARK = "target_boundary"
PRIMARY_MARKER_FAMILIES = (
    "marker_V_all_layers",
    "marker_KV_L20_23",
)
CONTROL_FAMILY = "closing_KV_all_layers"


def binary_cells(width: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in bits) for bits in product((0, 1), repeat=width))


def _compact_splice_audit(audit: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if audit is None:
        return None
    return {key: value for key, value in audit.items() if key != "per_layer"}


@torch.inference_mode()
def capture_landmark_states_from_cache(
    model: Any,
    adapter: Any,
    encoding: Any,
    past: Any,
    *,
    prefix_length: int,
    landmarks: Mapping[str, int],
    read_layers: Sequence[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Capture several absolute suffix positions in one cached forward."""

    start = int(prefix_length)
    positions = {str(label): int(position) for label, position in landmarks.items()}
    layers = tuple(sorted({int(value) for value in read_layers}))
    if not positions or not layers:
        raise ValueError("Landmarks and read layers must be nonempty")
    if any(not start <= position < int(encoding.sequence_length) for position in positions.values()):
        raise ValueError("A captured landmark lies outside the cached suffix")
    if any(not 0 <= layer < int(adapter.num_layers) for layer in layers):
        raise ValueError("A captured read layer lies outside the decoder")
    end = max(positions.values()) + 1
    suffix_width = end - start
    captured: dict[str, dict[int, torch.Tensor]] = {
        label: {} for label in positions
    }
    handles = []
    for layer in layers:

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Decoder block input is not a positional tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != suffix_width:
                return
            for label, absolute in positions.items():
                captured[label][layer] = (
                    hidden[0, absolute - start].detach().float().cpu()
                )

        handles.append(adapter.layers[layer].register_forward_pre_hook(hook))
    try:
        _forward_from_cache(
            model,
            encoding,
            past,
            start=start,
            end=end,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = {
        label: sorted(set(layers) - set(by_layer))
        for label, by_layer in captured.items()
        if set(by_layer) != set(layers)
    }
    if missing:
        raise RuntimeError(f"Cached landmark capture missed states: {missing}")
    return captured


def _landmark_expected_count(
    landmark: str,
    bits: Sequence[int],
    *,
    receiver: int,
    physical_target: int,
) -> int:
    if landmark.startswith("inserted_boundary_"):
        slot = int(landmark.rsplit("_", 1)[-1])
        return int(receiver) + sum(int(value) for value in bits[: slot + 1])
    if landmark == "target_marker":
        return int(receiver) + sum(int(value) for value in bits)
    if landmark == "target_boundary":
        return int(physical_target) + sum(int(value) for value in bits)
    raise ValueError(f"Unknown ledger landmark: {landmark}")


def _decoded_fields(probe: Mapping[str, Any], state: torch.Tensor) -> dict[str, Any]:
    decoded = decode_count_probe(probe, state.numpy())
    return {
        "probe_prediction": int(decoded["probe_prediction"]),
        "probe_softmax_expected_count": float(decoded["probe_softmax_expected_count"]),
        "probe_scores": [float(value) for value in decoded["probe_scores"]],
    }


def append_textual_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    request_id: str,
    variants: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    probes: Mapping[int, Mapping[str, Any]],
    receiver: int,
    physical_target: int,
) -> None:
    endpoints = (states["markers_000"], states["markers_111"])
    for variant in variants:
        variant_id = str(variant["variant_id"])
        bits = tuple(int(value) for value in variant["marker_bits"])
        for landmark, by_layer in states[variant_id].items():
            expected = _landmark_expected_count(
                landmark,
                bits,
                receiver=receiver,
                physical_target=physical_target,
            )
            causal_factors = (
                int(landmark.rsplit("_", 1)[-1]) + 1
                if landmark.startswith("inserted_boundary_")
                else len(bits)
            )
            for layer, state in sorted(by_layer.items()):
                decoded = _decoded_fields(probes[layer], state)
                axis = state_axis_metrics(
                    state,
                    endpoints[0][landmark][layer],
                    endpoints[1][landmark][layer],
                )
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "condition": "textual_factorial",
                        "seed": int(seed),
                        "request_id": str(request_id),
                        "variant_id": variant_id,
                        "marker_bits": list(bits),
                        "valid_marker_count": sum(bits),
                        "causally_available_factor_count": causal_factors,
                        "landmark": landmark,
                        "read_layer": int(layer),
                        "expected_count": int(expected),
                        **decoded,
                        "probe_prediction_exact": int(decoded["probe_prediction"]) == expected,
                        "probe_expected_absolute_error": abs(
                            float(decoded["probe_softmax_expected_count"]) - expected
                        ),
                        "endpoint_axis_progress": axis["donor_axis_progress"],
                        "off_axis_norm_over_endpoint_contrast": axis[
                            "off_axis_norm_over_clean_contrast"
                        ],
                        "clean_endpoint_contrast_l2": axis["clean_contrast_l2"],
                        "tokens_changed": True,
                        "only_marker_token_ids_vary_across_cells": True,
                        "attention_mask_changed": False,
                        "positions_changed": False,
                    }
                )


def append_cache_subset_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    request_id: str,
    family: str,
    bits: Sequence[int],
    positions: Sequence[int],
    layers: Sequence[int],
    components: Sequence[str],
    captured: Mapping[str, Mapping[int, torch.Tensor]],
    clean_states: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    probes: Mapping[int, Mapping[str, Any]],
    splice_audit: Mapping[str, Any] | None,
) -> None:
    subset_id = "".join(str(int(value)) for value in bits)
    for landmark, by_layer in captured.items():
        for layer, state in sorted(by_layer.items()):
            decoded = _decoded_fields(probes[layer], state)
            receiver = clean_states["markers_000"][landmark][layer]
            donor = clean_states["markers_111"][landmark][layer]
            receiver_decoded = _decoded_fields(probes[layer], receiver)
            donor_decoded = _decoded_fields(probes[layer], donor)
            axis = state_axis_metrics(state, receiver, donor)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "condition": "cache_subset_splice",
                    "seed": int(seed),
                    "request_id": str(request_id),
                    "family": str(family),
                    "subset_id": subset_id,
                    "marker_bits": [int(value) for value in bits],
                    "subset_size": sum(int(value) for value in bits),
                    "spliced_positions": [int(value) for value in positions],
                    "spliced_layers": [int(value) for value in layers],
                    "components": [str(value) for value in components],
                    "landmark": landmark,
                    "read_layer": int(layer),
                    **decoded,
                    "receiver_clean_prediction": int(receiver_decoded["probe_prediction"]),
                    "donor_clean_prediction": int(donor_decoded["probe_prediction"]),
                    **axis,
                    "splice_audit": _compact_splice_audit(splice_audit),
                    "tokens_changed": False,
                    "attention_mask_changed": False,
                    "positions_changed": False,
                }
            )


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    active_x = np.asarray(x, dtype=np.float64)
    active_y = np.asarray(y, dtype=np.float64)
    if len(active_x) < 2 or float(np.var(active_x)) <= 1e-12:
        raise ValueError("Slope requires at least two distinct x values")
    return float(np.polyfit(active_x, active_y, 1)[0])


def summarize_textual_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 8:
        raise ValueError("A textual seed summary requires all eight factorial cells")
    h = [int(row["valid_marker_count"]) for row in rows]
    axis = [float(row["endpoint_axis_progress"]) for row in rows]
    probe = [float(row["probe_softmax_expected_count"]) for row in rows]
    bits = [tuple(int(value) for value in row["marker_bits"]) for row in rows]
    hamming_means = {
        str(level): {
            "axis_progress": fmean(axis[i] for i in range(8) if h[i] == level),
            "probe_softmax_expected_count": fmean(
                probe[i] for i in range(8) if h[i] == level
            ),
        }
        for level in range(4)
    }
    axis_main = [
        fmean(axis[i] for i in range(8) if bits[i][slot] == 1)
        - fmean(axis[i] for i in range(8) if bits[i][slot] == 0)
        for slot in range(3)
    ]
    probe_main = [
        fmean(probe[i] for i in range(8) if bits[i][slot] == 1)
        - fmean(probe[i] for i in range(8) if bits[i][slot] == 0)
        for slot in range(3)
    ]
    level_axis = [float(hamming_means[str(level)]["axis_progress"]) for level in range(4)]
    return {
        "seed": int(rows[0]["seed"]),
        "landmark": str(rows[0]["landmark"]),
        "read_layer": int(rows[0]["read_layer"]),
        "axis_progress_per_valid_marker": _slope(h, axis),
        "probe_expected_count_per_valid_marker": _slope(h, probe),
        "axis_factorial_main_effects": axis_main,
        "probe_factorial_main_effects": probe_main,
        "all_axis_main_effects_positive": all(value > 0 for value in axis_main),
        "hamming_level_axis_monotone": all(
            right >= left for left, right in zip(level_axis, level_axis[1:])
        ),
        "probe_prediction_accuracy": fmean(
            float(bool(row["probe_prediction_exact"])) for row in rows
        ),
        "hamming_level_means": hamming_means,
    }


def summarize_cache_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 8:
        raise ValueError("A cache seed summary requires all eight subsets")
    by_bits = {
        tuple(int(value) for value in row["marker_bits"]): float(row["donor_axis_progress"])
        for row in rows
    }
    if set(by_bits) != set(binary_cells(3)):
        raise ValueError("A cache seed summary is missing factorial subsets")
    baseline = by_bits[(0, 0, 0)]
    singletons = [
        by_bits[tuple(1 if index == slot else 0 for index in range(3))] - baseline
        for slot in range(3)
    ]
    main_effects = [
        fmean(value for bits, value in by_bits.items() if bits[slot] == 1)
        - fmean(value for bits, value in by_bits.items() if bits[slot] == 0)
        for slot in range(3)
    ]
    hamming_means = {
        str(level): fmean(
            value for bits, value in by_bits.items() if sum(bits) == level
        )
        for level in range(4)
    }
    full = by_bits[(1, 1, 1)] - baseline
    return {
        "seed": int(rows[0]["seed"]),
        "family": str(rows[0]["family"]),
        "landmark": str(rows[0]["landmark"]),
        "read_layer": int(rows[0]["read_layer"]),
        "baseline_progress": baseline,
        "singleton_effects": singletons,
        "singleton_mean": fmean(singletons),
        "early_singleton_mean": fmean(singletons[:2]),
        "latest_singleton_effect": singletons[2],
        "all_singletons_positive": all(value > 0 for value in singletons),
        "factorial_main_effects": main_effects,
        "all_factorial_main_effects_positive": all(value > 0 for value in main_effects),
        "full_subset_progress": full,
        "additive_full_prediction": sum(singletons),
        "additivity_error": full - sum(singletons),
        "absolute_additivity_error": abs(full - sum(singletons)),
        "hamming_progress_per_entry": _slope(
            [sum(bits) for bits in by_bits], list(by_bits.values())
        ),
        "hamming_level_means": hamming_means,
    }


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    active = [float(value) for value in values]
    return {
        "n_seeds": len(active),
        "mean": fmean(active),
        "median": float(np.median(active)),
        "positive_rate": float(np.mean(np.asarray(active) > 0)),
        "two_sided_exact_sign_flip_pvalue": sign_flip_pvalue(active),
        "per_seed": active,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    textual_groups: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    cache_groups: dict[tuple[int, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "textual_factorial":
            textual_groups[(int(row["seed"]), str(row["landmark"]), int(row["read_layer"]))].append(row)
        elif row["condition"] == "cache_subset_splice":
            cache_groups[(int(row["seed"]), str(row["family"]), str(row["landmark"]), int(row["read_layer"]))].append(row)
    textual = [summarize_textual_seed(active) for _key, active in sorted(textual_groups.items())]
    cache = [summarize_cache_seed(active) for _key, active in sorted(cache_groups.items())]

    primary_textual = {
        int(row["seed"]): row
        for row in textual
        if row["landmark"] == PRIMARY_LANDMARK and int(row["read_layer"]) == PRIMARY_READ_LAYER
    }
    primary_cache = {
        (int(row["seed"]), str(row["family"])): row
        for row in cache
        if row["landmark"] == PRIMARY_LANDMARK and int(row["read_layer"]) == PRIMARY_READ_LAYER
    }
    seeds = sorted(primary_textual)
    seed_estimands = []
    for seed in seeds:
        required = [primary_cache.get((seed, family)) for family in (*PRIMARY_MARKER_FAMILIES, CONTROL_FAMILY)]
        if any(row is None for row in required):
            raise ValueError(f"Seed {seed} is missing a primary cache family")
        marker = [primary_cache[(seed, family)] for family in PRIMARY_MARKER_FAMILIES]
        control = primary_cache[(seed, CONTROL_FAMILY)]
        marker_singleton = fmean(float(row["singleton_mean"]) for row in marker)
        marker_early = fmean(float(row["early_singleton_mean"]) for row in marker)
        marker_full = fmean(float(row["full_subset_progress"]) for row in marker)
        slot_means = [
            fmean(float(row["singleton_effects"][slot]) for row in marker)
            for slot in range(3)
        ]
        seed_estimands.append(
            {
                "seed": seed,
                "marker_singleton_mean": marker_singleton,
                "closing_singleton_mean": float(control["singleton_mean"]),
                "marker_entry_specificity": marker_singleton - float(control["singleton_mean"]),
                "marker_early_singleton_mean": marker_early,
                "closing_early_singleton_mean": float(control["early_singleton_mean"]),
                "early_entry_specificity": marker_early - float(control["early_singleton_mean"]),
                "marker_full_subset_progress": marker_full,
                "marker_slot_singleton_means": slot_means,
                "all_marker_slot_means_positive": all(value > 0 for value in slot_means),
                "textual_axis_progress_per_valid_marker": float(
                    primary_textual[seed]["axis_progress_per_valid_marker"]
                ),
                "textual_probe_expected_count_per_valid_marker": float(
                    primary_textual[seed]["probe_expected_count_per_valid_marker"]
                ),
            }
        )
    formal = {
        "marker_entry_specificity": _aggregate(
            [row["marker_entry_specificity"] for row in seed_estimands]
        ),
        "early_entry_specificity": _aggregate(
            [row["early_entry_specificity"] for row in seed_estimands]
        ),
        "marker_full_subset_progress": _aggregate(
            [row["marker_full_subset_progress"] for row in seed_estimands]
        ),
        "textual_axis_progress_per_valid_marker": _aggregate(
            [row["textual_axis_progress_per_valid_marker"] for row in seed_estimands]
        ),
        "textual_probe_expected_count_per_valid_marker": _aggregate(
            [row["textual_probe_expected_count_per_valid_marker"] for row in seed_estimands]
        ),
        "all_three_marker_slots_positive_rate": float(
            np.mean([row["all_marker_slot_means_positive"] for row in seed_estimands])
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_seeds": seeds,
        "primary_landmark": PRIMARY_LANDMARK,
        "primary_read_layer": PRIMARY_READ_LAYER,
        "primary_marker_families": list(PRIMARY_MARKER_FAMILIES),
        "matched_control_family": CONTROL_FAMILY,
        "seed_estimands": seed_estimands,
        "formal_estimands": formal,
        "textual_seed_summaries": textual,
        "cache_seed_summaries": cache,
    }


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
    parser.add_argument("--source-occurrences", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--read-layers", type=int, nargs="+", default=[15, 16, 24])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "event-ledger-factorial"

    seeds = tuple(dict.fromkeys(int(value) for value in args.evaluation_seeds))
    sources = tuple(int(value) for value in args.source_occurrences)
    read_layers = tuple(sorted({int(value) for value in args.read_layers}))
    if not seeds or len(sources) != 3 or PRIMARY_READ_LAYER not in read_layers:
        raise ValueError("Nonempty seeds, exactly three sources, and L24 are required")
    probes = _load_probes(args.frozen_probes, read_layers)
    source_rows = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    if max(read_layers) >= int(adapter.num_layers) or int(adapter.num_layers) < 24:
        raise ValueError("Ledger read/splice layers lie outside the decoder")

    all_layers = tuple(range(int(adapter.num_layers)))
    families = {
        "marker_V_all_layers": {"role": "marker", "layers": all_layers, "components": ("value",)},
        "marker_K_all_layers": {"role": "marker", "layers": all_layers, "components": ("key",)},
        "marker_KV_L20_23": {"role": "marker", "layers": tuple(range(20, 24)), "components": ("key", "value")},
        "closing_KV_all_layers": {"role": "closing", "layers": all_layers, "components": ("key", "value")},
    }
    trials: list[dict[str, Any]] = []
    geometry_audits: list[dict[str, Any]] = []
    cache_equivalence: list[dict[str, Any]] = []

    for seed in seeds:
        row = source_rows[seed]
        source, blank, registry, _scrub = build_diagnostic_bases(
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
        variants, geometry = build_marker_event_factorial(
            source,
            blank,
            registry,
            boundaries,
            receiver=int(args.receiver),
            source_occurrences=sources,
        )
        by_id = {str(variant["variant_id"]): variant for variant in variants}
        if set(by_id) != {"markers_" + "".join(map(str, bits)) for bits in binary_cells(3)}:
            raise RuntimeError("Ledger construction did not produce the full 2^3 factorial")
        insertion_start = int(geometry["insertion_start"])
        event_end = int(geometry["event_end"])
        landmarks = {
            **{
                f"inserted_boundary_{slot['slot']}": int(slot["event_boundary"])
                for slot in geometry["inserted_slots"]
            },
            "target_marker": int(geometry["target_marker_position"]),
            "target_boundary": int(geometry["target_boundary"]),
        }
        suffix_landmarks = {
            key: value for key, value in landmarks.items() if value >= event_end
        }
        if set(suffix_landmarks) != {"target_marker", "target_boundary"}:
            raise RuntimeError("Cached suffix landmarks are not the two target reads")
        geometry_audits.append(
            {
                "seed": int(seed),
                **{key: value for key, value in geometry.items() if key != "inserted_marker_positions"},
                "landmarks": landmarks,
            }
        )

        common = prefill_common_prefix(model, by_id["markers_000"]["encoding"], end=insertion_start)
        textual_states: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
        for variant in variants:
            variant_id = str(variant["variant_id"])
            textual_states[variant_id] = capture_landmark_states_from_cache(
                model,
                adapter,
                variant["encoding"],
                clone_cache(common),
                prefix_length=insertion_start,
                landmarks=landmarks,
                read_layers=read_layers,
            )
        append_textual_rows(
            trials,
            seed=seed,
            request_id=str(row["request_id"]),
            variants=variants,
            states=textual_states,
            probes=probes,
            receiver=int(geometry["receiver"]),
            physical_target=int(geometry["physical_target"]),
        )

        endpoint_caches = {
            label: advance_event_cache(
                model,
                by_id[label]["encoding"],
                common,
                start=insertion_start,
                end=event_end,
            )
            for label in ("markers_000", "markers_111")
        }
        del common
        cached_clean = {
            label: capture_landmark_states_from_cache(
                model,
                adapter,
                by_id[label]["encoding"],
                clone_cache(endpoint_caches[label]),
                prefix_length=event_end,
                landmarks=suffix_landmarks,
                read_layers=read_layers,
            )
            for label in ("markers_000", "markers_111")
        }
        for label in ("markers_000", "markers_111"):
            for landmark in suffix_landmarks:
                for layer in read_layers:
                    equivalence = state_equivalence(
                        textual_states[label][landmark][layer],
                        cached_clean[label][landmark][layer],
                    )
                    textual_decoded = _decoded_fields(
                        probes[layer], textual_states[label][landmark][layer]
                    )
                    cached_decoded = _decoded_fields(
                        probes[layer], cached_clean[label][landmark][layer]
                    )
                    cache_equivalence.append(
                        {
                            "seed": int(seed),
                            "variant_id": label,
                            "landmark": landmark,
                            "read_layer": int(layer),
                            **equivalence,
                            "probe_prediction_match": int(textual_decoded["probe_prediction"])
                            == int(cached_decoded["probe_prediction"]),
                        }
                    )
                    if equivalence["cosine_similarity"] < 0.999:
                        raise RuntimeError(
                            "Chunked clean cache failed textual-state equivalence: "
                            + json.dumps(cache_equivalence[-1], sort_keys=True)
                        )

        slot_positions = {
            "marker": [tuple(int(value) for value in slot["marker_positions"]) for slot in geometry["inserted_slots"]],
            "closing": [(int(slot["event_boundary"]),) for slot in geometry["inserted_slots"]],
        }
        receiver_cache = endpoint_caches["markers_000"]
        donor_cache = endpoint_caches["markers_111"]
        receiver_encoding = by_id["markers_000"]["encoding"]
        for family, spec in families.items():
            for bits in binary_cells(3):
                positions = tuple(
                    position
                    for bit, active in zip(bits, slot_positions[str(spec["role"])])
                    if bit
                    for position in active
                )
                if positions:
                    hybrid, splice_audit = splice_cache_positions(
                        receiver_cache,
                        donor_cache,
                        positions=positions,
                        layers=spec["layers"],
                        components=spec["components"],
                    )
                    captured = capture_landmark_states_from_cache(
                        model,
                        adapter,
                        receiver_encoding,
                        hybrid,
                        prefix_length=event_end,
                        landmarks=suffix_landmarks,
                        read_layers=read_layers,
                    )
                    del hybrid
                else:
                    splice_audit = None
                    captured = cached_clean["markers_000"]
                append_cache_subset_rows(
                    trials,
                    seed=seed,
                    request_id=str(row["request_id"]),
                    family=family,
                    bits=bits,
                    positions=positions,
                    layers=spec["layers"],
                    components=spec["components"],
                    captured=captured,
                    clean_states=cached_clean,
                    probes=probes,
                    splice_audit=splice_audit,
                )
        del endpoint_caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[event-ledger-factorial] seed={seed} complete", flush=True)

    summary = {
        **summarize(trials),
        "model_label": str(args.model),
        "receiver": int(args.receiver),
        "source_occurrences": list(sources),
        "read_layers": list(read_layers),
        "families": {
            key: {
                "role": str(value["role"]),
                "layers": list(value["layers"]),
                "components": list(value["components"]),
            }
            for key, value in families.items()
        },
        "geometry_audits": geometry_audits,
        "cache_textual_equivalence": cache_equivalence,
        "cache_textual_all_probe_predictions_match": all(
            bool(row["probe_prediction_match"]) for row in cache_equivalence
        ),
        "cache_textual_probe_prediction_match_rate": float(
            np.mean([bool(row["probe_prediction_match"]) for row in cache_equivalence])
        ),
        "cache_textual_min_cosine_similarity": min(
            float(row["cosine_similarity"]) for row in cache_equivalence
        ),
        "trial_count": len(trials),
        "estimand_note": (
            "Cache rows move selected same-position fields from the clean 111 cache "
            "into clean 000. Donor-axis progress is normalized by independently "
            "materialized 000/111 target states. Closing fields are a matched "
            "one-position-per-event downstream control."
        ),
    }
    _atomic_jsonl(args.output, trials)
    _atomic_json(args.summary, summary)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "evaluation_seeds": list(seeds),
                "output": str(args.output),
                "summary": str(args.summary),
                "trial_count": len(trials),
                "cache_textual_min_cosine_similarity": summary[
                    "cache_textual_min_cosine_similarity"
                ],
                "formal_estimands": summary["formal_estimands"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
