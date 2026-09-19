#!/usr/bin/env python3
"""Adjacent-donor count-subspace swaps on an unchanged native bullet trace."""

from __future__ import annotations

import argparse
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
    clamp_boundary_layers_and_capture_positions,
    count_probe_predictions,
    count_probe_scores,
    count_probe_subspace,
    fit_dual_ridge_count_probe,
    norm_matched_orthogonal_replacement,
    projected_donor_replacement,
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
    maximum = float(np.max(scores))
    probabilities = np.exp(scores - maximum)
    probabilities = probabilities / probabilities.sum()
    return {
        "probe_prediction": int(count_probe_predictions(probe, values)[0]),
        "probe_scores": [float(value) for value in scores],
        "probe_softmax_expected_count": float(
            np.sum(probabilities * np.arange(1, 11, dtype=np.float64))
        ),
    }


def _frozen_layer_probes_and_bases(
    state_path: Path,
    *,
    layers: tuple[int, ...],
    alpha: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray]]:
    payload = np.load(state_path)
    discovery = np.asarray(payload["discovery"], dtype=np.float32)
    stored_layers = [int(value) for value in np.asarray(payload["layers"])]
    counts = np.asarray(payload["counts"], dtype=np.int64)
    if discovery.ndim != 4 or counts.tolist() != list(range(1, 11)):
        raise ValueError("Boundary-state bank geometry changed")
    labels = np.tile(counts, int(discovery.shape[0]))
    probes: dict[int, dict[str, Any]] = {}
    bases: dict[int, np.ndarray] = {}
    for layer in layers:
        if layer not in stored_layers:
            raise ValueError(f"Boundary-state bank has no layer {layer}")
        layer_index = stored_layers.index(layer)
        states = discovery[:, layer_index].reshape(-1, discovery.shape[-1])
        probe = fit_dual_ridge_count_probe(states, labels, alpha=float(alpha))
        probes[layer] = probe
        bases[layer] = count_probe_subspace(probe)
    return probes, bases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--boundary-states", type=Path, required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--receiver-occurrence", type=int, default=5)
    parser.add_argument("--donor-occurrences", type=int, nargs="+", default=(4, 5, 6))
    parser.add_argument("--clamp-start-layer", type=int, default=14)
    parser.add_argument("--read-layer", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "boundary-subspace-transition"

    wanted = tuple(int(value) for value in args.seeds)
    receiver_k = int(args.receiver_occurrence)
    donors = tuple(int(value) for value in args.donor_occurrences)
    read_layer = int(args.read_layer)
    clamp_layers = tuple(range(int(args.clamp_start_layer), read_layer))
    if donors != (receiver_k - 1, receiver_k, receiver_k + 1):
        raise ValueError("Subspace pilot requires adjacent donors k-1,k,k+1")
    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {int(row["seed"]): row for row in rows if int(row["seed"]) in set(wanted)}
    if set(selected) != set(wanted):
        raise ValueError("One or more requested seeds are absent")

    probe_npz = np.load(args.frozen_probes)
    alpha = float(np.asarray(probe_npz["alpha"])[0])
    frozen_layers = set(int(value) for value in np.asarray(probe_npz["frozen_layers"]))
    if read_layer not in frozen_layers:
        raise ValueError("Read layer has no frozen confirmation probe")
    read_probe = {
        "mean": np.asarray(probe_npz[f"layer_{read_layer}_mean"], dtype=np.float32),
        "weights": np.asarray(probe_npz[f"layer_{read_layer}_weights"], dtype=np.float32),
        "alpha": alpha,
    }
    _layer_probes, bases = _frozen_layer_probes_and_bases(
        args.boundary_states,
        layers=clamp_layers,
        alpha=alpha,
    )
    model, tokenizer, adapter = _model(args)
    results: list[dict[str, Any]] = []

    for seed in wanted:
        row = selected[seed]
        source, _blank, registry, _audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        boundaries = {
            occurrence: select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=occurrence
            )[0]
            for occurrence in set(donors) | {receiver_k, receiver_k + 1}
        }
        receiver_position = boundaries[receiver_k]
        next_position = boundaries[receiver_k + 1]
        read_positions = (receiver_position, next_position)
        state_positions = tuple(boundaries[donor] for donor in donors)
        captures = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            state_positions,
            layers=clamp_layers,
        )
        clean = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            read_positions,
            layers=(read_layer,),
        )[read_layer]
        for site_index, (site, expected) in enumerate(
            (("receiver_boundary", receiver_k), ("next_boundary", receiver_k + 1))
        ):
            decoded = _decode(read_probe, clean[site_index].numpy())
            results.append(
                {
                    "schema_version": "boundary_subspace_transition_v1",
                    "model_label": str(args.model),
                    "seed": seed,
                    "request_id": str(row["request_id"]),
                    "condition": "clean",
                    "site": site,
                    "receiver_occurrence": receiver_k,
                    "donor_occurrence": None,
                    "expected_count": expected,
                    "exact": bool(decoded["probe_prediction"] == expected),
                    "clamp_layers": list(clamp_layers),
                    "read_layer": read_layer,
                    **decoded,
                }
            )

        receiver_index = donors.index(receiver_k)
        for donor_index, donor in enumerate(donors):
            projected_replacements: dict[int, np.ndarray] = {}
            control_replacements: dict[int, np.ndarray] = {}
            projected_norms: dict[int, float] = {}
            control_norms: dict[int, float] = {}
            subspace_ranks: dict[int, int] = {}
            for layer in clamp_layers:
                receiver_state = captures[layer][receiver_index].numpy()
                donor_state = captures[layer][donor_index].numpy()
                projected, projected_delta = projected_donor_replacement(
                    receiver_state,
                    donor_state,
                    bases[layer],
                )
                control, control_delta = norm_matched_orthogonal_replacement(
                    receiver_state,
                    projected_delta,
                    bases[layer],
                    seed=20260901 + seed * 1000 + donor * 100 + layer,
                )
                projected_replacements[layer] = projected
                control_replacements[layer] = control
                projected_norms[layer] = float(np.linalg.norm(projected_delta))
                control_norms[layer] = float(np.linalg.norm(control_delta))
                subspace_ranks[layer] = int(bases[layer].shape[1])

            for condition, replacements, delta_norms in (
                ("projected_count_swap", projected_replacements, projected_norms),
                ("orthogonal_norm_matched", control_replacements, control_norms),
            ):
                captured, applications, realized_norms, read_applications = (
                    clamp_boundary_layers_and_capture_positions(
                        model,
                        adapter,
                        source,
                        patch_position=receiver_position,
                        replacement_states=replacements,
                        read_positions=read_positions,
                        read_layer=read_layer,
                    )
                )
                for site_index, site in enumerate(("receiver_boundary", "next_boundary")):
                    expected = (
                        donor
                        if condition == "projected_count_swap" and site == "receiver_boundary"
                        else donor + 1
                        if condition == "projected_count_swap"
                        else receiver_k
                        if site == "receiver_boundary"
                        else receiver_k + 1
                    )
                    decoded = _decode(read_probe, captured[site_index].numpy())
                    results.append(
                        {
                            "schema_version": "boundary_subspace_transition_v1",
                            "model_label": str(args.model),
                            "seed": seed,
                            "request_id": str(row["request_id"]),
                            "condition": condition,
                            "site": site,
                            "receiver_occurrence": receiver_k,
                            "donor_occurrence": donor,
                            "expected_count": expected,
                            "exact": bool(decoded["probe_prediction"] == expected),
                            "clamp_layers": list(clamp_layers),
                            "read_layer": read_layer,
                            "subspace_ranks": subspace_ranks,
                            "planned_delta_l2_norms": delta_norms,
                            "realized_delta_l2_norms": realized_norms,
                            "clamp_hook_applications": applications,
                            "read_hook_applications": read_applications,
                            **decoded,
                        }
                    )
        print(f"[boundary-subspace] seed={seed} complete", flush=True)

    _atomic_jsonl(args.output, results)
    grouped: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in results}):
        for site in ("receiver_boundary", "next_boundary"):
            values = [
                row
                for row in results
                if row["condition"] == condition and row["site"] == site
            ]
            grouped[f"{condition}:{site}"] = {
                "n": len(values),
                "exact": sum(bool(row["exact"]) for row in values),
                "predictions": [int(row["probe_prediction"]) for row in values],
                "expected": [int(row["expected_count"]) for row in values],
                "donors": [row["donor_occurrence"] for row in values],
                "mean_softmax_expected_count": float(
                    np.mean([row["probe_softmax_expected_count"] for row in values])
                ),
            }
    _atomic_json(
        args.summary,
        {
            "schema_version": "boundary_subspace_transition_v1",
            "seeds": list(wanted),
            "receiver_occurrence": receiver_k,
            "donor_occurrences": list(donors),
            "clamp_layers": list(clamp_layers),
            "read_layer": read_layer,
            "conditions": grouped,
            "input_tokens_changed": False,
            "task_changed": False,
            "whole_state_swap_used": False,
            "count_subspace_frozen_from_discovery_only": True,
        },
    )
    print(f"[boundary-subspace] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
