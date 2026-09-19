#!/usr/bin/env python3
"""Cumulative one-boundary clamp for counted/no-op counter transitions."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.boundary_counter_probe import (  # noqa: E402
    clamp_boundary_layers_and_capture_later_state,
    count_probe_predictions,
    count_probe_scores,
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


def _decode(probe: dict[str, Any], state: np.ndarray) -> dict[str, Any]:
    values = np.asarray(state, dtype=np.float32).reshape(1, -1)
    scores = count_probe_scores(probe, values)[0]
    return {
        "probe_prediction": int(count_probe_predictions(probe, values)[0]),
        "probe_scores": [float(value) for value in scores],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--receiver-occurrence", type=int, default=5)
    parser.add_argument("--donor-occurrences", type=int, nargs="+", default=(2, 5, 8))
    parser.add_argument("--clamp-start-layer", type=int, default=14)
    parser.add_argument("--read-layer", type=int, default=24)
    parser.add_argument(
        "--read-site",
        choices=("next_boundary", "receiver_boundary"),
        default="next_boundary",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "boundary-probe-clamped-transition"

    wanted = tuple(int(value) for value in args.seeds)
    receiver_k = int(args.receiver_occurrence)
    donors = tuple(int(value) for value in args.donor_occurrences)
    read_layer = int(args.read_layer)
    clamp_layers = tuple(range(int(args.clamp_start_layer), read_layer))
    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {int(row["seed"]): row for row in rows if int(row["seed"]) in set(wanted)}
    if set(selected) != set(wanted):
        raise ValueError("One or more clamp seeds are absent")
    probe_npz = np.load(args.frozen_probes)
    frozen_layers = set(int(value) for value in np.asarray(probe_npz["frozen_layers"]))
    if read_layer not in frozen_layers:
        raise ValueError("Clamp read layer has no frozen probe")
    probe = {
        "mean": np.asarray(probe_npz[f"layer_{read_layer}_mean"], dtype=np.float32),
        "weights": np.asarray(probe_npz[f"layer_{read_layer}_weights"], dtype=np.float32),
        "alpha": float(np.asarray(probe_npz["alpha"])[0]),
    }
    model, tokenizer, adapter = _model(args)
    results: list[dict[str, Any]] = []

    for seed in wanted:
        row = selected[seed]
        source, blank, registry, audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        boundaries = {
            k: select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=k
            )[0]
            for k in set(donors) | {receiver_k, receiver_k + 1}
        }
        receiver_boundary = boundaries[receiver_k]
        next_boundary = boundaries[receiver_k + 1]
        active_read_position = (
            next_boundary
            if str(args.read_site) == "next_boundary"
            else receiver_boundary
        )
        next_start, next_end = (
            int(value) for value in registry.trace_items[receiver_k]
        )
        noop_ids = list(source.input_ids)
        noop_ids[next_start:next_end] = blank.input_ids[next_start:next_end]
        noop = replace(source, input_ids=tuple(noop_ids))
        donor_positions = tuple(boundaries[donor] for donor in donors)
        donor_captures = capture_decoder_block_input_states(
            model, adapter, source, donor_positions, layers=clamp_layers
        )
        transition_registry = (
            (("counted", source, 1), ("noop", noop, 0))
            if str(args.read_site) == "next_boundary"
            else (("persistence", source, 0),)
        )
        for transition_kind, encoding, increment in transition_registry:
            baseline = capture_decoder_block_input_states(
                model,
                adapter,
                encoding,
                (active_read_position,),
                layers=(read_layer,),
            )[read_layer][0]
            baseline_decoded = _decode(probe, baseline.numpy())
            baseline_expected = receiver_k + increment
            results.append(
                {
                    "schema_version": "boundary_probe_clamped_transition_v1",
                    "model_label": str(args.model),
                    "seed": seed,
                    "request_id": str(row["request_id"]),
                    "condition": f"unpatched_{transition_kind}",
                    "transition_kind": transition_kind,
                    "clamp_layers": list(clamp_layers),
                    "read_layer": read_layer,
                    "receiver_occurrence": receiver_k,
                    "donor_occurrence": None,
                    "expected_count": baseline_expected,
                    "exact": bool(baseline_decoded["probe_prediction"] == baseline_expected),
                    **baseline_decoded,
                }
            )
            for donor_index, donor in enumerate(donors):
                replacements = {
                    layer: donor_captures[layer][donor_index]
                    for layer in clamp_layers
                }
                captured, applications, norms, read_apps = (
                    clamp_boundary_layers_and_capture_later_state(
                        model,
                        adapter,
                        encoding,
                        patch_position=receiver_boundary,
                        replacement_states=replacements,
                        read_position=active_read_position,
                        read_layer=read_layer,
                    )
                )
                decoded = _decode(probe, captured.numpy())
                expected = donor + increment
                scores = decoded["probe_scores"]
                results.append(
                    {
                        "schema_version": "boundary_probe_clamped_transition_v1",
                        "model_label": str(args.model),
                        "seed": seed,
                        "request_id": str(row["request_id"]),
                        "condition": f"clamped_{transition_kind}",
                        "transition_kind": transition_kind,
                        "clamp_layers": list(clamp_layers),
                        "read_layer": read_layer,
                        "receiver_occurrence": receiver_k,
                        "donor_occurrence": donor,
                        "expected_count": expected,
                        "exact": bool(decoded["probe_prediction"] == expected),
                        "expected_score": float(scores[expected - 1]),
                        "receiver_default_score": float(scores[baseline_expected - 1]),
                        "clamp_hook_applications": applications,
                        "clamp_realized_l2_norms": norms,
                        "read_hook_applications": read_apps,
                        **decoded,
                    }
                )
            print(
                f"[probe-clamp] seed={seed} kind={transition_kind} complete",
                flush=True,
            )

    _atomic_jsonl(args.output, results)
    grouped: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in results}):
        values = [row for row in results if row["condition"] == condition]
        grouped[condition] = {
            "n": len(values),
            "exact": sum(bool(row["exact"]) for row in values),
            "predictions": [int(row["probe_prediction"]) for row in values],
            "expected": [int(row["expected_count"]) for row in values],
            "donors": [row["donor_occurrence"] for row in values],
        }
    _atomic_json(
        args.summary,
        {
            "schema_version": "boundary_probe_clamped_transition_v1",
            "seeds": list(wanted),
            "receiver_occurrence": receiver_k,
            "donor_occurrences": list(donors),
            "clamp_layers": list(clamp_layers),
            "read_layer": read_layer,
            "read_site": str(args.read_site),
            "conditions": grouped,
            "hard_suffix_bottleneck_used": False,
            "native_Total_readout_used": False,
            "only_receiver_boundary_clamped": True,
        },
    )
    print(f"[probe-clamp] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
