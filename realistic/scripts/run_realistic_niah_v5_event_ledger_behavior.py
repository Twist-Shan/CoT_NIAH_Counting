#!/usr/bin/env python3
"""Early-stop behavioral readout of the three-entry marker ledger assay."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
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

from realistic_niah_v4_4_3.interventions import (  # noqa: E402
    _score_candidate_sequences,
    candidate_sequence_metrics,
)
from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.causal import sign_flip_pvalue  # noqa: E402
from realistic_niah_v5.event_cache_splice import clone_cache, splice_cache_positions  # noqa: E402
from realistic_niah_v5.event_ledger import build_marker_event_factorial  # noqa: E402
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    minimal_terminal_suffix_token_ids,
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
)
from scripts.run_realistic_niah_v5_event_ledger_factorial import (  # noqa: E402
    CONTROL_FAMILY,
    PRIMARY_MARKER_FAMILIES,
    binary_cells,
)
from scripts.run_realistic_niah_v5_unified_carrier_transition import _read_rows  # noqa: E402


SCHEMA_VERSION = "event_ledger_behavior_v1"


def build_ledger_early_stop_encoding(
    encoding: Any,
    registry: Any,
    geometry: Mapping[str, Any],
    *,
    terminal_suffix_token_ids: Sequence[int],
    expected_count: int,
) -> tuple[Any, dict[str, Any]]:
    """Stop after the physical target item and append the native Total query."""

    physical_target = int(geometry["physical_target"])
    delta = int(geometry["total_token_delta"])
    original_start, original_end = (
        int(value) for value in registry.trace_items[physical_target - 1]
    )
    target_start = original_start + delta
    target_end = original_end + delta
    if target_start != int(geometry["event_end"]):
        raise ValueError("Early-stop target does not immediately follow ledger events")
    suffix = tuple(int(value) for value in terminal_suffix_token_ids)
    if not suffix or not 1 <= int(expected_count) <= 10:
        raise ValueError("Early-stop suffix/count is invalid")
    ids = tuple(int(value) for value in encoding.input_ids)
    mask = tuple(int(value) for value in encoding.attention_mask)
    early_ids = ids[:target_end] + suffix
    early_mask = mask[:target_end] + (1,) * len(suffix)
    early = replace(
        encoding,
        count=int(expected_count),
        input_ids=early_ids,
        attention_mask=early_mask,
        query_position=len(early_ids) - 1,
        trace_item_spans=(),
        slot_spans=(),
        needle_spans=(),
    )
    if early.input_ids[:target_end] != ids[:target_end]:
        raise RuntimeError("Behavioral early stop changed the event/item prefix")
    return early, {
        "expected_count": int(expected_count),
        "target_start": target_start,
        "target_end": target_end,
        "query_position": int(early.query_position),
        "sequence_length": int(early.sequence_length),
        "future_native_items_removed": 10 - physical_target,
        "minimal_native_terminal_suffix": True,
        "future_trace_tokens_present": False,
    }


@torch.inference_mode()
def score_behavior(
    model: Any,
    encoding: Any,
    prefill_output: Any,
) -> dict[str, Any]:
    """Score registered answer-plus-termination sequences for counts 1..10."""

    scored = _score_candidate_sequences(model, encoding, prefill_output)
    metrics = candidate_sequence_metrics(scored.candidate_log_scores, encoding)
    return {
        "candidate_predicted_count": int(metrics["predicted_count_among_candidates"]),
        "candidate_expected_count": float(metrics["expected_count"]),
        "candidate_correct_probability": float(metrics["correct_count_probability"]),
        "candidate_correct_margin": float(metrics["correct_count_margin"]),
        "candidate_log_scores": str(metrics["candidate_log_scores"]),
        "candidate_probabilities": str(metrics["candidate_probabilities"]),
        "candidate_exact": int(metrics["predicted_count_among_candidates"])
        == int(encoding.count),
    }


def _scalar_progress(active: float, receiver: float, donor: float) -> float | None:
    denominator = float(donor) - float(receiver)
    if abs(denominator) <= 1e-8:
        return None
    return float((float(active) - float(receiver)) / denominator)


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    return float(np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)[0])


def summarize_textual_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 8:
        raise ValueError("Behavioral textual summary requires eight cells")
    bits = [tuple(int(value) for value in row["marker_bits"]) for row in rows]
    hamming = [sum(value) for value in bits]
    expected = [float(row["candidate_expected_count"]) for row in rows]
    by_hamming = {
        str(level): fmean(expected[i] for i in range(8) if hamming[i] == level)
        for level in range(4)
    }
    main_effects = [
        fmean(expected[i] for i in range(8) if bits[i][slot] == 1)
        - fmean(expected[i] for i in range(8) if bits[i][slot] == 0)
        for slot in range(3)
    ]
    level_values = [float(by_hamming[str(level)]) for level in range(4)]
    return {
        "seed": int(rows[0]["seed"]),
        "candidate_expected_count_per_valid_marker": _slope(hamming, expected),
        "clean_endpoint_expected_count_contrast": float(
            next(row["candidate_expected_count"] for row in rows if row["subset_id"] == "111")
            - next(row["candidate_expected_count"] for row in rows if row["subset_id"] == "000")
        ),
        "candidate_factorial_main_effects": main_effects,
        "all_candidate_main_effects_positive": all(value > 0 for value in main_effects),
        "hamming_expected_count_monotone": all(
            right >= left for left, right in zip(level_values, level_values[1:])
        ),
        "candidate_exact_accuracy": fmean(float(bool(row["candidate_exact"])) for row in rows),
        "hamming_expected_count_means": by_hamming,
    }


def summarize_cache_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 8:
        raise ValueError("Behavioral cache summary requires eight subsets")
    by_bits = {
        tuple(int(value) for value in row["marker_bits"]): float(row["behavior_axis_progress"])
        for row in rows
    }
    if set(by_bits) != set(binary_cells(3)):
        raise ValueError("Behavioral cache summary is missing subsets")
    baseline = by_bits[(0, 0, 0)]
    singletons = [
        by_bits[tuple(1 if index == slot else 0 for index in range(3))] - baseline
        for slot in range(3)
    ]
    full = by_bits[(1, 1, 1)] - baseline
    return {
        "seed": int(rows[0]["seed"]),
        "family": str(rows[0]["family"]),
        "singleton_effects": singletons,
        "singleton_mean": fmean(singletons),
        "early_singleton_mean": fmean(singletons[:2]),
        "all_singletons_positive": all(value > 0 for value in singletons),
        "full_subset_progress": full,
        "additivity_error": full - sum(singletons),
        "hamming_progress_per_entry": _slope(
            [sum(bits) for bits in by_bits], list(by_bits.values())
        ),
        "candidate_exact_accuracy": fmean(float(bool(row["candidate_exact"])) for row in rows),
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
    textual_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    cache_groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "textual_behavior":
            textual_groups[int(row["seed"])].append(row)
        elif row["condition"] == "cache_behavior":
            cache_groups[(int(row["seed"]), str(row["family"]))].append(row)
    textual = [summarize_textual_seed(active) for _seed, active in sorted(textual_groups.items())]
    cache = [summarize_cache_seed(active) for _key, active in sorted(cache_groups.items())]
    by_cache = {(int(row["seed"]), str(row["family"])): row for row in cache}
    by_textual = {int(row["seed"]): row for row in textual}
    seed_estimands = []
    for seed in sorted(by_textual):
        marker = [by_cache[(seed, family)] for family in PRIMARY_MARKER_FAMILIES]
        control = by_cache[(seed, CONTROL_FAMILY)]
        marker_singleton = fmean(float(row["singleton_mean"]) for row in marker)
        marker_early = fmean(float(row["early_singleton_mean"]) for row in marker)
        slot_means = [
            fmean(float(row["singleton_effects"][slot]) for row in marker)
            for slot in range(3)
        ]
        seed_estimands.append(
            {
                "seed": seed,
                "behavior_marker_entry_specificity": marker_singleton
                - float(control["singleton_mean"]),
                "behavior_early_entry_specificity": marker_early
                - float(control["early_singleton_mean"]),
                "behavior_marker_full_subset_progress": fmean(
                    float(row["full_subset_progress"]) for row in marker
                ),
                "behavior_marker_slot_singleton_means": slot_means,
                "all_behavior_marker_slots_positive": all(value > 0 for value in slot_means),
                **{
                    key: value
                    for key, value in by_textual[seed].items()
                    if key != "seed"
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_seeds": [int(row["seed"]) for row in seed_estimands],
        "seed_estimands": seed_estimands,
        "formal_estimands": {
            "behavior_marker_entry_specificity": _aggregate(
                [row["behavior_marker_entry_specificity"] for row in seed_estimands]
            ),
            "behavior_early_entry_specificity": _aggregate(
                [row["behavior_early_entry_specificity"] for row in seed_estimands]
            ),
            "behavior_marker_full_subset_progress": _aggregate(
                [row["behavior_marker_full_subset_progress"] for row in seed_estimands]
            ),
            "textual_candidate_expected_count_per_valid_marker": _aggregate(
                [row["candidate_expected_count_per_valid_marker"] for row in seed_estimands]
            ),
            "textual_clean_endpoint_expected_count_contrast": _aggregate(
                [row["clean_endpoint_expected_count_contrast"] for row in seed_estimands]
            ),
            "textual_candidate_exact_accuracy": {
                "n_seeds": len(seed_estimands),
                "mean": fmean(float(row["candidate_exact_accuracy"]) for row in seed_estimands),
                "per_seed": [float(row["candidate_exact_accuracy"]) for row in seed_estimands],
            },
            "all_three_behavior_marker_slots_positive_rate": float(
                np.mean([row["all_behavior_marker_slots_positive"] for row in seed_estimands])
            ),
        },
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
    parser.add_argument("--receiver", type=int, default=5)
    parser.add_argument("--source-occurrences", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "event-ledger-behavior"

    seeds = tuple(dict.fromkeys(int(value) for value in args.evaluation_seeds))
    sources = tuple(int(value) for value in args.source_occurrences)
    if not seeds or len(sources) != 3:
        raise ValueError("Seeds and exactly three sources are required")
    source_rows = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    all_layers = tuple(range(int(adapter.num_layers)))
    families = {
        "marker_V_all_layers": {"role": "marker", "layers": all_layers, "components": ("value",)},
        "marker_KV_L20_23": {"role": "marker", "layers": tuple(range(20, 24)), "components": ("key", "value")},
        "closing_KV_all_layers": {"role": "closing", "layers": all_layers, "components": ("key", "value")},
    }
    trials: list[dict[str, Any]] = []
    geometry_audits: list[dict[str, Any]] = []

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
        terminal_suffix, terminal_audit = minimal_terminal_suffix_token_ids(row, tokenizer)
        early_encodings: dict[str, Any] = {}
        early_audits: dict[str, Any] = {}
        for variant in variants:
            variant_id = str(variant["variant_id"])
            early, audit = build_ledger_early_stop_encoding(
                variant["encoding"],
                registry,
                geometry,
                terminal_suffix_token_ids=terminal_suffix,
                expected_count=int(variant["event_count_target"]),
            )
            early_encodings[variant_id] = early
            early_audits[variant_id] = audit
        lengths = {int(value.sequence_length) for value in early_encodings.values()}
        queries = {int(value.query_position) for value in early_encodings.values()}
        if len(lengths) != 1 or len(queries) != 1:
            raise RuntimeError("Behavioral factorial changed early-stop geometry")
        geometry_audits.append(
            {
                "seed": int(seed),
                "ledger_geometry": geometry,
                "early_stop_audits": early_audits,
                "terminal_suffix_audit": terminal_audit,
                "all_early_stop_cells_equal_length": True,
                "all_early_stop_queries_equal_position": True,
            }
        )

        insertion_start = int(geometry["insertion_start"])
        event_end = int(geometry["event_end"])
        common = prefill_common_prefix(model, early_encodings["markers_000"], end=insertion_start)
        textual_outcomes: dict[str, dict[str, Any]] = {}
        for variant in variants:
            variant_id = str(variant["variant_id"])
            early = early_encodings[variant_id]
            prefill = _forward_from_cache(
                model,
                early,
                clone_cache(common),
                start=insertion_start,
                end=int(early.sequence_length),
                use_cache=True,
            )
            outcome = score_behavior(
                model,
                early,
                prefill,
            )
            textual_outcomes[variant_id] = outcome
            bits = tuple(int(value) for value in variant["marker_bits"])
            trials.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "condition": "textual_behavior",
                    "seed": int(seed),
                    "request_id": str(row["request_id"]),
                    "subset_id": "".join(map(str, bits)),
                    "marker_bits": list(bits),
                    "valid_marker_count": sum(bits),
                    "expected_count": int(variant["event_count_target"]),
                    **outcome,
                    "tokens_changed": True,
                    "only_marker_token_ids_vary_across_cells": True,
                    "attention_mask_changed": False,
                    "positions_changed": False,
                }
            )

        receiver_expected = float(textual_outcomes["markers_000"]["candidate_expected_count"])
        donor_expected = float(textual_outcomes["markers_111"]["candidate_expected_count"])
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
        slot_positions = {
            "marker": [tuple(int(value) for value in slot["marker_positions"]) for slot in geometry["inserted_slots"]],
            "closing": [(int(slot["event_boundary"]),) for slot in geometry["inserted_slots"]],
        }
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
                        endpoint_caches["markers_000"],
                        endpoint_caches["markers_111"],
                        positions=positions,
                        layers=spec["layers"],
                        components=spec["components"],
                    )
                else:
                    hybrid = clone_cache(endpoint_caches["markers_000"])
                    splice_audit = None
                expected_count = int(geometry["physical_target"]) + sum(bits)
                active_encoding = replace(
                    early_encodings["markers_000"], count=expected_count
                )
                prefill = _forward_from_cache(
                    model,
                    active_encoding,
                    hybrid,
                    start=event_end,
                    end=int(active_encoding.sequence_length),
                    use_cache=True,
                )
                outcome = score_behavior(
                    model,
                    active_encoding,
                    prefill,
                )
                progress = _scalar_progress(
                    float(outcome["candidate_expected_count"]),
                    receiver_expected,
                    donor_expected,
                )
                if progress is None:
                    raise RuntimeError("Clean behavioral endpoints have zero contrast")
                trials.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "condition": "cache_behavior",
                        "seed": int(seed),
                        "request_id": str(row["request_id"]),
                        "family": family,
                        "subset_id": "".join(map(str, bits)),
                        "marker_bits": list(bits),
                        "subset_size": sum(bits),
                        "expected_count": expected_count,
                        **outcome,
                        "behavior_axis_progress": progress,
                        "clean_000_candidate_expected_count": receiver_expected,
                        "clean_111_candidate_expected_count": donor_expected,
                        "spliced_positions": list(positions),
                        "spliced_layers": list(spec["layers"]),
                        "components": list(spec["components"]),
                        "splice_audit": (
                            {key: value for key, value in splice_audit.items() if key != "per_layer"}
                            if splice_audit is not None
                            else None
                        ),
                        "tokens_changed": False,
                        "attention_mask_changed": False,
                        "positions_changed": False,
                    }
                )
        del endpoint_caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[event-ledger-behavior] seed={seed} complete", flush=True)

    summary = {
        **summarize(trials),
        "model_label": str(args.model),
        "receiver": int(args.receiver),
        "source_occurrences": list(sources),
        "families": {
            key: {
                "role": str(value["role"]),
                "layers": list(value["layers"]),
                "components": list(value["components"]),
            }
            for key, value in families.items()
        },
        "geometry_audits": geometry_audits,
        "trial_count": len(trials),
        "estimand_note": (
            "Every condition stops immediately after original item 6 and appends "
            "the token-exact native channel close plus Total query with recap text "
            "removed. Candidate expected count scores registered answer+termination "
            "sequences 1..10. Cache progress is normalized by textual 000/111 behavior."
        ),
    }
    _atomic_jsonl(args.output, trials)
    _atomic_json(args.summary, summary)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "evaluation_seeds": list(seeds),
                "trial_count": len(trials),
                "output": str(args.output),
                "summary": str(args.summary),
                "formal_estimands": summary["formal_estimands"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
