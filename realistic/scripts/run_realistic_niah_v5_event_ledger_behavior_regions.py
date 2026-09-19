#!/usr/bin/env python3
"""Post-hoc cache-region sufficiency scan for marker-ledger behavior."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

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
    cache_difference,
    prefill_common_prefix,
)
from scripts.run_realistic_niah_v5_event_ledger_behavior import (  # noqa: E402
    _scalar_progress,
    build_ledger_early_stop_encoding,
    score_behavior,
)
from scripts.run_realistic_niah_v5_unified_carrier_transition import _read_rows  # noqa: E402


SCHEMA_VERSION = "event_ledger_behavior_regions_v1"


def compile_region_positions(geometry: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    slots = tuple(geometry["inserted_slots"])
    marker = tuple(
        int(position) for slot in slots for position in slot["marker_positions"]
    )
    closing = tuple(int(slot["event_boundary"]) for slot in slots)
    full_event = tuple(
        position
        for slot in slots
        for position in range(int(slot["start"]), int(slot["end"]))
    )
    marker_set = set(marker)
    nonmarker = tuple(position for position in full_event if position not in marker_set)
    marker_closing = tuple(sorted(set(marker).union(closing)))
    if (
        not marker
        or not closing
        or set(marker).intersection(nonmarker)
        or set(marker).union(nonmarker) != set(full_event)
    ):
        raise RuntimeError("Ledger cache regions failed disjoint-union audit")
    return {
        "marker": marker,
        "closing": closing,
        "marker_closing": marker_closing,
        "nonmarker_event": nonmarker,
        "full_event": full_event,
    }


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    cells = []
    for family, active in sorted(grouped.items()):
        progress = [float(row["behavior_axis_progress"]) for row in active]
        cells.append(
            {
                "family": family,
                "n_seeds": len(active),
                "mean_behavior_axis_progress": fmean(progress),
                "median_behavior_axis_progress": float(np.median(progress)),
                "positive_rate": float(np.mean(np.asarray(progress) > 0)),
                "two_sided_exact_sign_flip_pvalue": sign_flip_pvalue(progress),
                "per_seed_behavior_axis_progress": progress,
                "candidate_exact_rate": fmean(
                    float(bool(row["candidate_exact"])) for row in active
                ),
                "all_splices_changed_cache": all(
                    int(row["splice_audit"]["changed_elements"]) > 0 for row in active
                ),
                "all_hybrid_caches_equal_donor": all(
                    bool(row["hybrid_vs_donor_cache"]["exactly_equal"])
                    for row in active
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "post_hoc": True,
        "cells": cells,
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
    args.command = "event-ledger-behavior-regions"

    seeds = tuple(dict.fromkeys(int(value) for value in args.evaluation_seeds))
    sources = tuple(int(value) for value in args.source_occurrences)
    if not seeds or len(sources) != 3:
        raise ValueError("Seeds and exactly three sources are required")
    source_rows = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    all_layers = tuple(range(int(adapter.num_layers)))
    families = {
        "marker_K_all_layers": ("marker", all_layers, ("key",)),
        "marker_V_all_layers": ("marker", all_layers, ("value",)),
        "marker_KV_all_layers": ("marker", all_layers, ("key", "value")),
        "marker_KV_L20_23": ("marker", tuple(range(20, 24)), ("key", "value")),
        "closing_KV_all_layers": ("closing", all_layers, ("key", "value")),
        "marker_closing_KV_all_layers": (
            "marker_closing",
            all_layers,
            ("key", "value"),
        ),
        "nonmarker_event_KV_all_layers": (
            "nonmarker_event",
            all_layers,
            ("key", "value"),
        ),
        "full_event_KV_all_layers": (
            "full_event",
            all_layers,
            ("key", "value"),
        ),
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
        by_id = {str(value["variant_id"]): value for value in variants}
        terminal_suffix, terminal_audit = minimal_terminal_suffix_token_ids(row, tokenizer)
        early: dict[str, Any] = {}
        for label in ("markers_000", "markers_111"):
            early[label], _audit = build_ledger_early_stop_encoding(
                by_id[label]["encoding"],
                registry,
                geometry,
                terminal_suffix_token_ids=terminal_suffix,
                expected_count=6 if label.endswith("000") else 9,
            )
        insertion_start = int(geometry["insertion_start"])
        event_end = int(geometry["event_end"])
        common = prefill_common_prefix(model, early["markers_000"], end=insertion_start)

        clean_outcomes: dict[str, dict[str, Any]] = {}
        for label in ("markers_000", "markers_111"):
            prefill = _forward_from_cache(
                model,
                early[label],
                clone_cache(common),
                start=insertion_start,
                end=int(early[label].sequence_length),
                use_cache=True,
            )
            clean_outcomes[label] = score_behavior(model, early[label], prefill)
        receiver_expected = float(clean_outcomes["markers_000"]["candidate_expected_count"])
        donor_expected = float(clean_outcomes["markers_111"]["candidate_expected_count"])

        caches = {
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
        regions = compile_region_positions(geometry)
        geometry_audits.append(
            {
                "seed": int(seed),
                "regions": {key: list(value) for key, value in regions.items()},
                "marker_nonmarker_disjoint_union_full_event": True,
                "terminal_suffix_audit": terminal_audit,
            }
        )
        for family, (region, layers, components) in families.items():
            hybrid, splice_audit = splice_cache_positions(
                caches["markers_000"],
                caches["markers_111"],
                positions=regions[region],
                layers=layers,
                components=components,
            )
            cache_audit = cache_difference(hybrid, caches["markers_111"])
            active_encoding = replace(early["markers_000"], count=9)
            prefill = _forward_from_cache(
                model,
                active_encoding,
                hybrid,
                start=event_end,
                end=int(active_encoding.sequence_length),
                use_cache=True,
            )
            outcome = score_behavior(model, active_encoding, prefill)
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
                    "post_hoc": True,
                    "seed": int(seed),
                    "request_id": str(row["request_id"]),
                    "family": family,
                    "region": region,
                    "components": list(components),
                    "spliced_layers": list(layers),
                    "spliced_positions": list(regions[region]),
                    **outcome,
                    "behavior_axis_progress": progress,
                    "clean_000_candidate_expected_count": receiver_expected,
                    "clean_111_candidate_expected_count": donor_expected,
                    "splice_audit": {
                        key: value for key, value in splice_audit.items() if key != "per_layer"
                    },
                    "hybrid_vs_donor_cache": cache_audit,
                    "tokens_changed": False,
                    "attention_mask_changed": False,
                    "positions_changed": False,
                }
            )
        del caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[event-ledger-behavior-regions] seed={seed} complete", flush=True)

    summary = {
        **summarize(trials),
        "model_label": str(args.model),
        "evaluation_seeds": list(seeds),
        "receiver": int(args.receiver),
        "source_occurrences": list(sources),
        "families": {
            family: {
                "region": region,
                "layers": list(layers),
                "components": list(components),
            }
            for family, (region, layers, components) in families.items()
        },
        "geometry_audits": geometry_audits,
        "trial_count": len(trials),
        "interpretation_guard": (
            "This scan was selected after the frozen behavioral confirmation was read. "
            "It can refine region sufficiency but cannot rescue or replace the failed "
            "confirmatory claim rule."
        ),
    }
    _atomic_jsonl(args.output, trials)
    _atomic_json(args.summary, summary)
    print(json.dumps(summary["cells"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
