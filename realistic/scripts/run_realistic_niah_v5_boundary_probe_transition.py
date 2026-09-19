#!/usr/bin/env python3
"""Test donor-dependent counted/no-op dynamics with frozen boundary probes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import re
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
    count_probe_predictions,
    count_probe_scores,
    transplant_boundary_and_capture_later_state,
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


_EXPLICIT_COUNT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:10|[0-9]|zero|one|two|three|four|five|six|seven|eight|nine|ten)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _load_probe(npz: Any, layer: int, alpha: float) -> dict[str, Any]:
    return {
        "mean": np.asarray(npz[f"layer_{layer}_mean"], dtype=np.float32),
        "weights": np.asarray(npz[f"layer_{layer}_weights"], dtype=np.float32),
        "alpha": float(alpha),
    }


def _decode(probe: dict[str, Any], state: np.ndarray) -> dict[str, Any]:
    values = np.asarray(state, dtype=np.float32).reshape(1, -1)
    scores = count_probe_scores(probe, values)[0]
    prediction = int(count_probe_predictions(probe, values)[0])
    return {
        "probe_prediction": prediction,
        "probe_scores": [float(value) for value in scores],
        "probe_top_margin": float(
            np.partition(scores, -1)[-1] - np.partition(scores, -2)[-2]
        ),
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
    parser.add_argument("--patch-read-pairs", type=int, nargs="+", default=(14, 15, 15, 16, 23, 24))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "boundary-probe-transition"

    flat_pairs = tuple(int(value) for value in args.patch_read_pairs)
    if len(flat_pairs) % 2:
        raise ValueError("Patch/read layer registry must contain pairs")
    pairs = tuple(zip(flat_pairs[::2], flat_pairs[1::2]))
    receiver_k = int(args.receiver_occurrence)
    donors = tuple(int(value) for value in args.donor_occurrences)
    if not 1 <= receiver_k < 10 or min(donors) < 1 or max(donors) > 9:
        raise ValueError("Transition occurrence registry is invalid")
    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = tuple(int(value) for value in args.seeds)
    selected = {int(row["seed"]): row for row in rows if int(row["seed"]) in set(wanted)}
    if set(selected) != set(wanted):
        raise ValueError("One or more transition seeds are absent")
    probe_npz = np.load(args.frozen_probes)
    alpha = float(np.asarray(probe_npz["alpha"])[0])
    frozen_layers = set(int(value) for value in np.asarray(probe_npz["frozen_layers"]))
    if any(read not in frozen_layers for _patch, read in pairs):
        raise ValueError("Every transition read layer must have a frozen probe")
    probes = {read: _load_probe(probe_npz, read, alpha) for _patch, read in pairs}
    model, tokenizer, adapter = _model(args)
    results: list[dict[str, Any]] = []

    for seed in wanted:
        row = selected[seed]
        source, blank, registry, audit = build_diagnostic_bases(
            row,
            tokenizer,
            # Must match the frozen-probe capture construction exactly.  A
            # different salt changes hundreds of count-neutral replacement
            # tokens and invalidates the unpatched probe ceiling.
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
        next_start, next_end = (
            int(value) for value in registry.trace_items[receiver_k]
        )
        noop_ids = list(source.input_ids)
        for position in range(next_start, next_end):
            noop_ids[position] = int(blank.input_ids[position])
        noop = replace(source, input_ids=tuple(noop_ids))
        counted = source
        counted_text = tokenizer.decode(
            list(counted.input_ids[next_start:next_end]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        noop_text = tokenizer.decode(
            list(noop.input_ids[next_start:next_end]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if _EXPLICIT_COUNT_RE.search(noop_text):
            raise ValueError("No-op transition contains an explicit candidate count")
        patch_layers = tuple(sorted({patch for patch, _read in pairs}))
        donor_positions = tuple(boundaries[donor] for donor in donors)
        donor_captures = capture_decoder_block_input_states(
            model, adapter, source, donor_positions, layers=patch_layers
        )
        donor_states = {
            patch_layer: {
                donor: donor_captures[patch_layer][index]
                for index, donor in enumerate(donors)
            }
            for patch_layer in patch_layers
        }
        baseline_captures: dict[str, dict[int, Any]] = {}
        for transition_kind, encoding in (("counted", counted), ("noop", noop)):
            read_layers = tuple(sorted({read for _patch, read in pairs}))
            baseline_captures[transition_kind] = capture_decoder_block_input_states(
                model, adapter, encoding, (next_boundary,), layers=read_layers
            )
            for patch_layer, read_layer in pairs:
                baseline_decoded = _decode(
                    probes[read_layer],
                    baseline_captures[transition_kind][read_layer][0].numpy(),
                )
                baseline_expected = receiver_k + (1 if transition_kind == "counted" else 0)
                results.append(
                    {
                        "schema_version": "boundary_probe_transition_v1",
                        "model_label": str(args.model),
                        "seed": seed,
                        "request_id": str(row["request_id"]),
                        "condition": f"unpatched_{transition_kind}",
                        "transition_kind": transition_kind,
                        "patch_layer": patch_layer,
                        "read_layer": read_layer,
                        "receiver_occurrence": receiver_k,
                        "donor_occurrence": None,
                        "expected_count": baseline_expected,
                        "exact": bool(baseline_decoded["probe_prediction"] == baseline_expected),
                        "counted_transition_text": counted_text,
                        "noop_transition_text": noop_text,
                        **baseline_decoded,
                    }
                )
                for donor in donors:
                    expected = donor + (1 if transition_kind == "counted" else 0)
                    captured, patch_apps, read_apps, norm = (
                        transplant_boundary_and_capture_later_state(
                            model,
                            adapter,
                            encoding,
                            patch_position=receiver_boundary,
                            patch_layer=patch_layer,
                            replacement_state=donor_states[patch_layer][donor],
                            read_position=next_boundary,
                            read_layer=read_layer,
                        )
                    )
                    decoded = _decode(probes[read_layer], captured.numpy())
                    scores = decoded["probe_scores"]
                    results.append(
                        {
                            "schema_version": "boundary_probe_transition_v1",
                            "model_label": str(args.model),
                            "seed": seed,
                            "request_id": str(row["request_id"]),
                            "condition": f"transplanted_{transition_kind}",
                            "transition_kind": transition_kind,
                            "patch_layer": patch_layer,
                            "read_layer": read_layer,
                            "receiver_occurrence": receiver_k,
                            "donor_occurrence": donor,
                            "expected_count": expected,
                            "exact": bool(decoded["probe_prediction"] == expected),
                            "expected_score": float(scores[expected - 1]),
                            "receiver_default_score": float(
                                scores[
                                    receiver_k
                                    + (1 if transition_kind == "counted" else 0)
                                    - 1
                                ]
                            ),
                            "patch_hook_applications": patch_apps,
                            "read_hook_applications": read_apps,
                            "patch_realized_l2_norm": norm,
                            "counted_transition_text": counted_text,
                            "noop_transition_text": noop_text,
                            **decoded,
                        }
                    )
            print(
                f"[probe-transition] seed={seed} kind={transition_kind} complete",
                flush=True,
            )
        print(f"[probe-transition] seed={seed} complete", flush=True)

    _atomic_jsonl(args.output, results)
    grouped: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in results}):
        for patch_layer, read_layer in pairs:
            values = [
                row
                for row in results
                if row["condition"] == condition
                and row["patch_layer"] == patch_layer
                and row["read_layer"] == read_layer
            ]
            grouped[f"{condition}|L{patch_layer}->L{read_layer}"] = {
                "n": len(values),
                "exact": sum(bool(row["exact"]) for row in values),
                "predictions": [int(row["probe_prediction"]) for row in values],
                "expected": [int(row["expected_count"]) for row in values],
                "donors": [row["donor_occurrence"] for row in values],
            }
    _atomic_json(
        args.summary,
        {
            "schema_version": "boundary_probe_transition_v1",
            "seeds": list(wanted),
            "receiver_occurrence": receiver_k,
            "donor_occurrences": list(donors),
            "patch_read_pairs": [list(value) for value in pairs],
            "frozen_probe_layers": sorted(frozen_layers),
            "conditions": grouped,
            "hard_suffix_bottleneck_used": False,
            "native_Total_readout_used": False,
            "position_ordinal_confound_addressed_by_cross_position_donors": True,
        },
    )
    print(f"[probe-transition] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
