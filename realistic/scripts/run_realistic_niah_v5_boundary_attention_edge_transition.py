#!/usr/bin/env python3
"""Intervene on the boundary-k value edge into boundary k+1 without text edits."""

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

from realistic_niah_v4.modeling import position_attention_outputs  # noqa: E402
from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.boundary_counter_probe import (  # noqa: E402
    add_attention_output_deltas_and_capture_positions,
    boundary_value_edge_write,
    capture_attention_value_states,
    count_probe_predictions,
    count_probe_scores,
    norm_matched_orthogonal_replacement,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from scripts.run_realistic_niah_v5_boundary_subspace_transition import (  # noqa: E402
    _frozen_layer_probes_and_bases,
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


def _project(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    active = np.asarray(basis, dtype=np.float64)
    return ((value @ active) @ active.T).astype(np.float32)


def _orthogonal_delta(
    target_delta: np.ndarray,
    basis: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    zero = np.zeros_like(np.asarray(target_delta, dtype=np.float32))
    _replacement, delta = norm_matched_orthogonal_replacement(
        zero,
        target_delta,
        basis,
        seed=seed,
    )
    return delta


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
    parser.add_argument("--edge-start-layer", type=int, default=14)
    parser.add_argument("--read-layer", type=int, default=24)
    parser.add_argument(
        "--target-scope",
        choices=("next_boundary", "next_item_span"),
        default="next_boundary",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "boundary-attention-edge-transition"

    wanted = tuple(int(value) for value in args.seeds)
    receiver_k = int(args.receiver_occurrence)
    donors = tuple(int(value) for value in args.donor_occurrences)
    read_layer = int(args.read_layer)
    edge_layers = tuple(range(int(args.edge_start_layer), read_layer))
    if donors != (receiver_k - 1, receiver_k, receiver_k + 1):
        raise ValueError("Attention-edge pilot requires adjacent donors k-1,k,k+1")
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
        layers=edge_layers,
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
        next_item_start, _next_item_end = (
            int(value) for value in registry.trace_items[receiver_k]
        )
        target_positions = (
            (next_position,)
            if str(args.target_scope) == "next_boundary"
            else tuple(
                range(max(receiver_position + 1, next_item_start), next_position + 1)
            )
        )
        if not target_positions or target_positions[-1] != next_position:
            raise RuntimeError("Attention-edge target scope does not end at boundary k+1")
        donor_positions = tuple(boundaries[donor] for donor in donors)
        value_states = capture_attention_value_states(
            model,
            adapter,
            source,
            donor_positions,
            layers=edge_layers,
        )
        attention_by_position = {}
        key_starts_by_position = {}
        for target_position in target_positions:
            attention_rows, key_starts, _query_logits = position_attention_outputs(
                model,
                adapter,
                source,
                target_position,
            )
            attention_by_position[target_position] = attention_rows
            key_starts_by_position[target_position] = key_starts
        writes: dict[int, dict[int, dict[int, np.ndarray]]] = {}
        write_audits: dict[
            int, dict[int, dict[int, dict[str, float | int]]]
        ] = {}
        for layer in edge_layers:
            writes[layer] = {}
            write_audits[layer] = {}
            for donor_index, donor in enumerate(donors):
                writes[layer][donor] = {}
                write_audits[layer][donor] = {}
                for target_position in target_positions:
                    write, audit = boundary_value_edge_write(
                        adapter,
                        layer=layer,
                        attention_row=attention_by_position[target_position][layer],
                        key_start=key_starts_by_position[target_position][layer],
                        source_position=receiver_position,
                        source_value=value_states[layer][donor_index],
                    )
                    writes[layer][donor][target_position] = write.numpy()
                    write_audits[layer][donor][target_position] = audit

        clean_state = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            (next_position,),
            layers=(read_layer,),
        )[read_layer][0]
        clean_decoded = _decode(read_probe, clean_state.numpy())
        results.append(
            {
                "schema_version": "boundary_attention_edge_transition_v1",
                "model_label": str(args.model),
                "seed": seed,
                "request_id": str(row["request_id"]),
                "condition": "clean",
                "receiver_occurrence": receiver_k,
                "donor_occurrence": None,
                "expected_count": receiver_k + 1,
                "exact": bool(clean_decoded["probe_prediction"] == receiver_k + 1),
                "edge_layers": list(edge_layers),
                "read_layer": read_layer,
                **clean_decoded,
            }
        )

        receiver_writes = {
            layer: writes[layer][receiver_k] for layer in edge_layers
        }
        condition_deltas: list[
            tuple[str, int | None, dict[int, dict[int, np.ndarray]], int]
        ] = []
        ablation = {
            layer: {
                target_position: -_project(
                    receiver_writes[layer][target_position], bases[layer]
                )
                for target_position in target_positions
            }
            for layer in edge_layers
        }
        ablation_control = {
            layer: {
                target_position: _orthogonal_delta(
                    ablation[layer][target_position],
                    bases[layer],
                    seed=(
                        20260911
                        + seed * 100000
                        + layer * 1000
                        + target_position
                    ),
                )
                for target_position in target_positions
            }
            for layer in edge_layers
        }
        condition_deltas.extend(
            (
                ("count_edge_ablation", None, ablation, receiver_k + 1),
                ("orthogonal_ablation_control", None, ablation_control, receiver_k + 1),
            )
        )
        for donor in donors:
            swap = {
                layer: {
                    target_position: _project(
                        writes[layer][donor][target_position]
                        - receiver_writes[layer][target_position],
                        bases[layer],
                    )
                    for target_position in target_positions
                }
                for layer in edge_layers
            }
            swap_control = {
                layer: {
                    target_position: _orthogonal_delta(
                        swap[layer][target_position],
                        bases[layer],
                        seed=(
                            20260921
                            + seed * 100000
                            + donor * 10000
                            + layer * 1000
                            + target_position
                        ),
                    )
                    for target_position in target_positions
                }
                for layer in edge_layers
            }
            condition_deltas.extend(
                (
                    ("count_edge_swap", donor, swap, donor + 1),
                    ("orthogonal_swap_control", donor, swap_control, receiver_k + 1),
                )
            )

        for condition, donor, deltas, expected in condition_deltas:
            captured, applications, realized, read_applications = (
                add_attention_output_deltas_and_capture_positions(
                    model,
                    adapter,
                    source,
                    output_deltas=deltas,
                    read_positions=(next_position,),
                    read_layer=read_layer,
                )
            )
            decoded = _decode(read_probe, captured[0].numpy())
            planned_norms = {
                layer: float(
                    np.sqrt(
                        sum(
                            float(np.linalg.norm(deltas[layer][position])) ** 2
                            for position in target_positions
                        )
                    )
                )
                for layer in edge_layers
            }
            results.append(
                {
                    "schema_version": "boundary_attention_edge_transition_v1",
                    "model_label": str(args.model),
                    "seed": seed,
                    "request_id": str(row["request_id"]),
                    "condition": condition,
                    "receiver_occurrence": receiver_k,
                    "donor_occurrence": donor,
                    "expected_count": expected,
                    "exact": bool(decoded["probe_prediction"] == expected),
                    "edge_layers": list(edge_layers),
                    "read_layer": read_layer,
                    "target_scope": str(args.target_scope),
                    "target_position_count": len(target_positions),
                    "planned_delta_l2_norms": planned_norms,
                    "realized_delta_l2_norms": realized,
                    "edge_hook_applications": applications,
                    "read_hook_applications": read_applications,
                    "receiver_edge_write_audits": {
                        layer: {
                            "target_position_count": len(target_positions),
                            "source_attention_mass_sum_across_targets": float(
                                sum(
                                    write_audits[layer][receiver_k][position][
                                        "source_attention_mass_sum"
                                    ]
                                    for position in target_positions
                                )
                            ),
                            "source_write_l2_norm_frobenius": float(
                                np.sqrt(
                                    sum(
                                        float(
                                            write_audits[layer][receiver_k][position][
                                                "source_write_l2_norm"
                                            ]
                                        )
                                        ** 2
                                        for position in target_positions
                                    )
                                )
                            ),
                        }
                        for layer in edge_layers
                    },
                    **decoded,
                }
            )
        print(f"[boundary-edge] seed={seed} complete", flush=True)

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
            "mean_softmax_expected_count": float(
                np.mean([row["probe_softmax_expected_count"] for row in values])
            ),
        }
    _atomic_json(
        args.summary,
        {
            "schema_version": "boundary_attention_edge_transition_v1",
            "seeds": list(wanted),
            "receiver_occurrence": receiver_k,
            "donor_occurrences": list(donors),
            "edge_layers": list(edge_layers),
            "read_layer": read_layer,
            "target_scope": str(args.target_scope),
            "conditions": grouped,
            "input_tokens_changed": False,
            "task_changed": False,
            "source_specific_edge": "boundary_k_value_to_boundary_k_plus_1_query",
            "count_subspace_frozen_from_discovery_only": True,
        },
    )
    print(f"[boundary-edge] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
