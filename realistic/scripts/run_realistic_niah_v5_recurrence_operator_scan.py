#!/usr/bin/env python3
"""Estimate causal count-state transition tables under several carriers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
from realistic_niah_v5.boundary_counter_probe import (  # noqa: E402
    norm_matched_orthogonal_replacement,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from realistic_niah_v5.kv_counter_transition import (  # noqa: E402
    item_bin_positions,
)
from realistic_niah_v5.unified_carrier_transition import (  # noqa: E402
    carrier_capture_positions,
    interpolated_boundary_targets,
    through_origin_slope,
)
from scripts.run_realistic_niah_v5_boundary_equivariance import (  # noqa: E402
    decode_count_probe,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_kv_counter_transition import (  # noqa: E402
    BankSpec,
    build_kv_directions,
    build_or_load_raw_kv_panel,
    orthogonal_kv_tangents,
)
from scripts.run_realistic_niah_v5_unified_carrier_transition import (  # noqa: E402
    _fit_fold_geometry,
    _read_rows,
    build_or_load_boundary_panel,
)


CARRIERS = (
    "whole_state",
    "residual_count_subspace",
    "residual_count_subspace_orthogonal",
    "residual_count_plus_kv",
    "residual_count_plus_kv_orthogonal",
)


def valid_scan_pairs(
    receivers: Sequence[int], doses: Sequence[int]
) -> tuple[tuple[int, int, int], ...]:
    """Return valid (receiver, donor, dose) pairs with a donor successor."""

    output: list[tuple[int, int, int]] = []
    for receiver in tuple(int(value) for value in receivers):
        if not 1 <= receiver < 10:
            raise ValueError("Scan receivers must have a native successor")
        for dose in tuple(int(value) for value in doses):
            if dose == 0:
                raise ValueError("Operator-scan doses must be nonzero")
            donor = receiver + dose
            if 1 <= donor < 10:
                output.append((receiver, donor, dose))
    if not output:
        raise ValueError("No valid receiver/donor pairs remain")
    return tuple(output)


def summarize_operator_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare fixed operator hypotheses and a fitted leaky-reset model."""

    output: dict[str, Any] = {}
    for carrier in sorted({str(row["carrier"]) for row in rows}):
        active = [row for row in rows if str(row["carrier"]) == carrier]
        current = np.asarray(
            [float(row["current_soft"]) for row in active], dtype=np.float64
        )
        later = np.asarray(
            [float(row["next_soft"]) for row in active], dtype=np.float64
        )
        clean_current = np.asarray(
            [float(row["clean_current_soft"]) for row in active], dtype=np.float64
        )
        clean_next = np.asarray(
            [float(row["clean_next_soft"]) for row in active], dtype=np.float64
        )
        delta_current = current - clean_current
        delta_next = later - clean_next
        retention = through_origin_slope(delta_current, delta_next)
        rho = 0.0 if retention is None else float(retention)
        predictions = {
            "reset_to_clean_next": clean_next,
            "plus_one": np.clip(current + 1.0, 1.0, 10.0),
            "identity": current,
            "fitted_leaky_reset": clean_next + rho * delta_current,
        }
        output[carrier] = {
            "trial_count": len(active),
            "current_target_exact": int(
                sum(int(row["current_prediction"]) == int(row["donor"]) for row in active)
            ),
            "next_target_exact": int(
                sum(
                    int(row["next_prediction"]) == int(row["donor"]) + 1
                    for row in active
                )
            ),
            "current_to_next_retention": retention,
            "operator_rmse": {
                label: float(np.sqrt(np.mean((later - prediction) ** 2)))
                for label, prediction in predictions.items()
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--discovery-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--fit-mode", choices=("oof", "full_discovery"), default="oof")
    parser.add_argument("--raw-kv-panel", type=Path, required=True)
    parser.add_argument("--boundary-panel", type=Path, required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--receivers", type=int, nargs="+", required=True)
    parser.add_argument("--doses", type=int, nargs="+", required=True)
    parser.add_argument("--clamp-start-layer", type=int, default=14)
    parser.add_argument("--read-layer", type=int, default=24)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--oof-folds", type=int, default=4)
    parser.add_argument("--whole-scale", type=float, default=1.0)
    parser.add_argument("--subspace-scale", type=float, default=1.0)
    parser.add_argument("--kv-scale", type=float, default=1.0)
    parser.add_argument("--carriers", nargs="+", choices=CARRIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "recurrence-operator-scan"

    discovery_seeds = tuple(int(value) for value in args.discovery_seeds)
    evaluation_seeds = tuple(int(value) for value in args.evaluation_seeds)
    receivers = tuple(dict.fromkeys(int(value) for value in args.receivers))
    doses = tuple(dict.fromkeys(int(value) for value in args.doses))
    carriers = tuple(dict.fromkeys(str(value) for value in args.carriers))
    pairs = valid_scan_pairs(receivers, doses)
    pairs_by_receiver = {
        receiver: tuple(pair for pair in pairs if pair[0] == receiver)
        for receiver in receivers
    }
    read_layer = int(args.read_layer)
    layers = tuple(range(int(args.clamp_start_layer), read_layer))
    folds = int(args.oof_folds)
    if len(discovery_seeds) < folds or folds < 2:
        raise ValueError("Discovery fitting requires at least two folds")
    if any(
        not np.isfinite(float(value)) or float(value) <= 0
        for value in (args.whole_scale, args.subspace_scale, args.kv_scale)
    ):
        raise ValueError("Every carrier scale must be finite and positive")

    all_seeds = tuple(dict.fromkeys(discovery_seeds + evaluation_seeds))
    source_rows = _read_rows(args.generations, all_seeds)
    discovery_rows = {seed: source_rows[seed] for seed in discovery_seeds}
    evaluation_rows = {seed: source_rows[seed] for seed in evaluation_seeds}
    probe_npz = np.load(args.frozen_probes)
    alpha = float(np.asarray(probe_npz["alpha"])[0])
    frozen_layers = set(int(value) for value in np.asarray(probe_npz["frozen_layers"]))
    if read_layer not in frozen_layers:
        raise ValueError("Read layer has no frozen confirmation probe")
    read_probe = {
        "mean": np.asarray(probe_npz[f"layer_{read_layer}_mean"], dtype=np.float32),
        "weights": np.asarray(
            probe_npz[f"layer_{read_layer}_weights"], dtype=np.float32
        ),
        "alpha": alpha,
    }

    model, tokenizer, adapter = _model(args)
    boundary_panel = build_or_load_boundary_panel(
        model,
        tokenizer,
        adapter,
        path=args.boundary_panel,
        rows=discovery_rows,
        seeds=discovery_seeds,
        layers=layers,
    )
    kv_panel = build_or_load_raw_kv_panel(
        model,
        tokenizer,
        adapter,
        path=args.raw_kv_panel,
        rows=discovery_rows,
        seeds=discovery_seeds,
        layers=layers,
        bins=int(args.bins),
    )
    fold_by_seed, residual_by_fold, kv_by_fold = _fit_fold_geometry(
        fit_mode=str(args.fit_mode),
        discovery_seeds=discovery_seeds,
        evaluation_seeds=evaluation_seeds,
        folds=folds,
        boundary_panel=boundary_panel,
        kv_panel=kv_panel,
        layers=layers,
        alpha=alpha,
    )

    need_residual = any(carrier != "whole_state" for carrier in carriers)
    need_kv = any("plus_kv" in carrier for carrier in carriers)
    need_residual_control = "residual_count_subspace_orthogonal" in carriers
    need_kv_control = "residual_count_plus_kv_orthogonal" in carriers
    kv_spec = BankSpec("all_history_kv", "all_history", "kv", layers)
    results: list[dict[str, Any]] = []
    for seed in evaluation_seeds:
        row = evaluation_rows[seed]
        fold = fold_by_seed[seed]
        residual_bases = residual_by_fold[fold]
        _kv_bases, kv_tangents = kv_by_fold[fold]
        orthogonal_kv = (
            orthogonal_kv_tangents(
                kv_by_fold[fold][0],
                kv_tangents,
                seed=20261225 + seed * 100,
            )
            if need_kv_control
            else {}
        )
        source, _blank, registry, scrub_audit = build_diagnostic_bases(
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
        captured = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            tuple(boundary_positions.values()),
            layers=layers + (read_layer,),
        )
        bins_by_occurrence = (
            item_bin_positions(registry.trace_items, bins=int(args.bins))
            if need_kv
            else {}
        )

        for receiver in receivers:
            clean_current = decode_count_probe(
                read_probe, captured[read_layer][receiver - 1].numpy()
            )
            clean_next = decode_count_probe(
                read_probe, captured[read_layer][receiver].numpy()
            )
            receiver_states = {
                layer: captured[layer][receiver - 1].numpy() for layer in layers
            }
            for _receiver, donor, dose in pairs_by_receiver[receiver]:
                donor_states = {
                    layer: captured[layer][donor - 1].numpy() for layer in layers
                }
                conditions: dict[
                    str,
                    tuple[
                        Mapping[int, np.ndarray],
                        Mapping[tuple[int, str], Mapping[int, np.ndarray]],
                        Mapping[int, np.ndarray],
                    ],
                ] = {}
                if "whole_state" in carriers:
                    targets, deltas = interpolated_boundary_targets(
                        receiver_states,
                        donor_states,
                        scale=float(args.whole_scale),
                    )
                    conditions["whole_state"] = (targets, {}, deltas)
                if need_residual:
                    subspace_targets, subspace_deltas = interpolated_boundary_targets(
                        receiver_states,
                        donor_states,
                        scale=float(args.subspace_scale),
                        bases=residual_bases,
                    )
                    conditions["residual_count_subspace"] = (
                        subspace_targets,
                        {},
                        subspace_deltas,
                    )
                    if need_residual_control or need_kv_control:
                        orthogonal_targets: dict[int, np.ndarray] = {}
                        orthogonal_deltas: dict[int, np.ndarray] = {}
                        for layer in layers:
                            target, delta = norm_matched_orthogonal_replacement(
                                receiver_states[layer],
                                subspace_deltas[layer],
                                residual_bases[layer],
                                seed=(
                                    20261301
                                    + seed * 10000
                                    + (dose + 20) * 100
                                    + layer
                                ),
                            )
                            orthogonal_targets[layer] = target
                            orthogonal_deltas[layer] = delta
                        conditions["residual_count_subspace_orthogonal"] = (
                            orthogonal_targets,
                            {},
                            orthogonal_deltas,
                        )
                    if need_kv:
                        aligned_kv = build_kv_directions(
                            kv_spec,
                            receiver=receiver,
                            dose=dose,
                            scale=float(args.kv_scale),
                            bins_by_occurrence=bins_by_occurrence,
                            tangents=kv_tangents,
                        )
                        conditions["residual_count_plus_kv"] = (
                            subspace_targets,
                            aligned_kv,
                            subspace_deltas,
                        )
                    if need_kv_control:
                        control_kv = build_kv_directions(
                            kv_spec,
                            receiver=receiver,
                            dose=dose,
                            scale=float(args.kv_scale),
                            bins_by_occurrence=bins_by_occurrence,
                            tangents=orthogonal_kv,
                        )
                        conditions["residual_count_plus_kv_orthogonal"] = (
                            orthogonal_targets,
                            control_kv,
                            orthogonal_deltas,
                        )

                for carrier in carriers:
                    boundary_targets, kv_directions, planned_deltas = conditions[carrier]
                    states, audit = carrier_capture_positions(
                        model,
                        adapter,
                        source,
                        boundary_position=boundary_positions[receiver],
                        boundary_targets=boundary_targets,
                        kv_directions=kv_directions,
                        read_positions=(
                            boundary_positions[receiver],
                            boundary_positions[receiver + 1],
                        ),
                        read_layer=read_layer,
                    )
                    current = decode_count_probe(read_probe, states[0].numpy())
                    later = decode_count_probe(read_probe, states[1].numpy())
                    current_soft = float(current["probe_softmax_expected_count"])
                    next_soft = float(later["probe_softmax_expected_count"])
                    clean_current_soft = float(
                        clean_current["probe_softmax_expected_count"]
                    )
                    clean_next_soft = float(clean_next["probe_softmax_expected_count"])
                    results.append(
                        {
                            "schema_version": "recurrence_operator_scan_v1",
                            "model_label": str(args.model),
                            "seed": seed,
                            "request_id": str(row["request_id"]),
                            "oof_fold": fold,
                            "fit_mode": str(args.fit_mode),
                            "receiver": receiver,
                            "donor": donor,
                            "dose": dose,
                            "carrier": carrier,
                            "whole_scale": float(args.whole_scale),
                            "subspace_scale": float(args.subspace_scale),
                            "kv_scale": float(args.kv_scale),
                            "read_layer": read_layer,
                            "clamp_layers": list(layers),
                            "scrub_construction": scrub_audit["construction"],
                            "clean_current_prediction": int(
                                clean_current["probe_prediction"]
                            ),
                            "clean_next_prediction": int(clean_next["probe_prediction"]),
                            "current_prediction": int(current["probe_prediction"]),
                            "next_prediction": int(later["probe_prediction"]),
                            "clean_current_soft": clean_current_soft,
                            "clean_next_soft": clean_next_soft,
                            "current_soft": current_soft,
                            "next_soft": next_soft,
                            "current_shift": current_soft - clean_current_soft,
                            "next_shift": next_soft - clean_next_soft,
                            "current_target_exact": bool(
                                int(current["probe_prediction"]) == donor
                            ),
                            "next_target_exact": bool(
                                int(later["probe_prediction"]) == donor + 1
                            ),
                            "probe_scores_current": current["probe_scores"],
                            "probe_scores_next": later["probe_scores"],
                            "planned_boundary_l2_norms": {
                                str(layer): float(np.linalg.norm(planned_deltas[layer]))
                                for layer in layers
                            },
                            "full_trace_audit": audit,
                            "tokens_changed_by_intervention": False,
                            "diagnostic_suffix_used": False,
                        }
                    )
            print(
                f"[operator-scan] seed={seed} receiver={receiver} complete",
                flush=True,
            )

    summary = {
        "schema_version": "recurrence_operator_scan_v1",
        "model_label": str(args.model),
        "fit_mode": str(args.fit_mode),
        "discovery_seeds": list(discovery_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "receivers": list(receivers),
        "doses": list(doses),
        "valid_pairs": [list(pair) for pair in pairs],
        "carriers": list(carriers),
        "whole_scale": float(args.whole_scale),
        "subspace_scale": float(args.subspace_scale),
        "kv_scale": float(args.kv_scale),
        "read_layer": read_layer,
        "clamp_layers": list(layers),
        "candidate_scoring_run": False,
        "outcomes": summarize_operator_rows(results),
        "input_tokens_changed_by_intervention": False,
        "diagnostic_suffix_used": False,
    }
    _atomic_jsonl(args.output, results)
    _atomic_json(args.summary, summary)
    print(json.dumps(summary["outcomes"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
