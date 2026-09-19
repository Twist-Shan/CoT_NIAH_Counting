#!/usr/bin/env python3
"""Token-resolved movie of a count-aligned offset across one native item.

The input is the same teacher-forced, explicit-count-scrubbed native trace used
by the distributed K/V transition assay.  At boundary B_k, a frozen +/-1
translation is applied either to the boundary alone or jointly to the full
historical K/V field.  Frozen boundary probes then read every token through
B_(k+1), exposing where the offset survives, collapses, or is reconstructed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
from realistic_niah_v5.kv_counter_transition import (  # noqa: E402
    add_boundary_and_kv_deltas_capture_layers,
    item_bin_positions,
)
from scripts.run_realistic_niah_v5_boundary_equivariance import (  # noqa: E402
    decode_count_probe,
    frozen_layer_geometry,
    through_origin_slope,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_kv_counter_transition import (  # noqa: E402
    BankSpec,
    _read_rows,
    build_kv_directions,
    build_or_load_raw_kv_panel,
    fit_kv_field,
)


def transition_position_metadata(
    trace_items: Sequence[Sequence[int]],
    *,
    receiver: int,
    current_boundary: int,
    next_boundary: int,
) -> tuple[dict[str, Any], ...]:
    """Label every native token from B_k through B_(k+1), inclusively."""

    items = tuple((int(value[0]), int(value[1])) for value in trace_items)
    active_receiver = int(receiver)
    left = int(current_boundary)
    right = int(next_boundary)
    if not 1 <= active_receiver < len(items):
        raise ValueError("Movie receiver must have a following item")
    next_start, next_end = items[active_receiver]
    if not left < right or not left < next_start < next_end or right < next_end - 1:
        raise ValueError("Movie boundary and next-item geometry are inconsistent")
    denominator = right - left
    output = []
    for position in range(left, right + 1):
        if position == left:
            role = "current_boundary"
        elif position == right:
            role = "next_boundary"
        elif next_start <= position < next_end:
            role = "next_item_token"
        elif position < next_start:
            role = "pre_item_separator"
        else:
            role = "post_item_separator"
        output.append(
            {
                "position": position,
                "transition_offset": position - left,
                "transition_progress": float((position - left) / denominator),
                "role": role,
                "is_current_boundary": position == left,
                "is_next_boundary": position == right,
            }
        )
    return tuple(output)


def progress_bin(progress: float, *, bins: int) -> int:
    """Map [0,1] to a grid that preserves both native boundaries exactly."""

    active_bins = int(bins)
    value = float(progress)
    if active_bins < 2 or not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Progress binning needs finite [0,1] progress and >=2 bins")
    if value == 0.0:
        return 0
    if value == 1.0:
        return active_bins - 1
    # Reserve endpoint bins for the two formal native boundary sites.
    return 1 + int(np.floor(value * (active_bins - 2)))


def _mean_ci_by_seed(
    values: Sequence[tuple[int, float]],
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, float | int]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for seed, value in values:
        grouped[int(seed)].append(float(value))
    seeds = tuple(sorted(grouped))
    if not seeds:
        raise ValueError("Cannot summarize an empty trajectory")

    def statistic(sampled: Sequence[int]) -> float:
        return float(
            np.mean([value for seed in sampled for value in grouped[int(seed)]])
        )

    point = statistic(seeds)
    rng = np.random.default_rng(int(random_seed))
    draws = np.asarray(
        [
            statistic(rng.choice(seeds, size=len(seeds), replace=True).tolist())
            for _ in range(int(bootstrap_samples))
        ],
        dtype=np.float64,
    )
    return {
        "estimate": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_seeds": len(seeds),
        "n_seed_receivers": len(values),
    }


def _slope_ci_by_seed(
    values: Sequence[tuple[int, float, float]],
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, float | int]:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for seed, x_value, y_value in values:
        grouped[int(seed)].append((float(x_value), float(y_value)))
    seeds = tuple(sorted(grouped))
    if not seeds:
        raise ValueError("Cannot summarize an empty retention trajectory")

    def statistic(sampled: Sequence[int]) -> float:
        pairs = [pair for seed in sampled for pair in grouped[int(seed)]]
        return float(
            through_origin_slope(
                [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            )
        )

    point = statistic(seeds)
    rng = np.random.default_rng(int(random_seed))
    draws = np.asarray(
        [
            statistic(rng.choice(seeds, size=len(seeds), replace=True).tolist())
            for _ in range(int(bootstrap_samples))
        ],
        dtype=np.float64,
    )
    return {
        "estimate": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_seeds": len(seeds),
        "n_seed_receiver_doses": len(values),
    }


def _sample_bin_means(
    rows: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], float],
) -> dict[tuple[int, int, int, str, int, int], float]:
    grouped: dict[tuple[int, int, int, str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["receiver"]),
            int(row["read_layer"]),
            str(row["bank"]),
            int(row["dose"]),
            int(row["progress_bin"]),
        )
        grouped[key].append(float(value(row)))
    return {key: float(np.mean(active)) for key, active in grouped.items()}


def summarize_movie(
    rows: Sequence[Mapping[str, Any]],
    *,
    progress_bins: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Aggregate variable-length token paths without treating tokens as samples."""

    active = list(rows)
    clean = [row for row in active if str(row["bank"]) == "clean"]
    interventions = [row for row in active if str(row["bank"]) != "clean"]
    layers = sorted({int(row["read_layer"]) for row in clean})
    banks = sorted({str(row["bank"]) for row in interventions})
    clean_by_position = {
        (
            int(row["seed"]),
            int(row["receiver"]),
            int(row["read_layer"]),
            int(row["position"]),
        ): float(row["probe_softmax_expected_count"])
        for row in clean
    }
    current_clean = {
        (int(row["seed"]), int(row["receiver"]), int(row["read_layer"])): float(
            row["probe_softmax_expected_count"]
        )
        for row in clean
        if bool(row["is_current_boundary"])
    }
    clean_bin = _sample_bin_means(
        clean, lambda row: float(row["probe_softmax_expected_count"])
    )
    clean_margin_bin = _sample_bin_means(
        clean,
        lambda row: float(row["probe_scores"][int(row["receiver"])])
        - float(row["probe_scores"][int(row["receiver"]) - 1]),
    )
    clean_next_rate_bin = _sample_bin_means(
        clean,
        lambda row: float(
            int(row["probe_prediction"]) == int(row["receiver"]) + 1
        ),
    )
    clean_current_rate_bin = _sample_bin_means(
        clean,
        lambda row: float(int(row["probe_prediction"]) == int(row["receiver"])),
    )
    intervention_bin = _sample_bin_means(
        interventions, lambda row: float(row["probe_softmax_expected_count"])
    )

    natural: dict[str, list[dict[str, Any]]] = {}
    natural_margin: dict[str, list[dict[str, Any]]] = {}
    natural_class_rates: dict[str, dict[str, list[dict[str, Any]]]] = {}
    paths: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    retention: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for layer in layers:
        layer_key = f"L{layer}"
        natural[layer_key] = []
        natural_margin[layer_key] = []
        natural_class_rates[layer_key] = {
            "current_k": [],
            "next_k_plus_1": [],
        }
        for bin_index in range(int(progress_bins)):
            values = []
            for key, soft_count in clean_bin.items():
                seed, receiver, active_layer, bank, dose, active_bin = key
                if active_layer == layer and bank == "clean" and dose == 0 and active_bin == bin_index:
                    values.append(
                        (
                            seed,
                            soft_count - current_clean[(seed, receiver, layer)],
                        )
                    )
            if values:
                natural[layer_key].append(
                    {
                        "progress_bin": bin_index,
                        "progress": float(bin_index / (int(progress_bins) - 1)),
                        **_mean_ci_by_seed(
                            values,
                            bootstrap_samples=int(bootstrap_samples),
                            random_seed=20260831 + layer * 101 + bin_index,
                        ),
                    }
                )
            margin_values = [
                (seed, margin)
                for (
                    seed,
                    _receiver,
                    active_layer,
                    bank,
                    dose,
                    active_bin,
                ), margin in clean_margin_bin.items()
                if active_layer == layer
                and bank == "clean"
                and dose == 0
                and active_bin == bin_index
            ]
            if margin_values:
                natural_margin[layer_key].append(
                    {
                        "progress_bin": bin_index,
                        "progress": float(bin_index / (int(progress_bins) - 1)),
                        **_mean_ci_by_seed(
                            margin_values,
                            bootstrap_samples=int(bootstrap_samples),
                            random_seed=20260832 + layer * 101 + bin_index,
                        ),
                    }
                )
            for label, panel in (
                ("current_k", clean_current_rate_bin),
                ("next_k_plus_1", clean_next_rate_bin),
            ):
                rate_values = [
                    (seed, rate)
                    for (
                        seed,
                        _receiver,
                        active_layer,
                        bank,
                        dose,
                        active_bin,
                    ), rate in panel.items()
                    if active_layer == layer
                    and bank == "clean"
                    and dose == 0
                    and active_bin == bin_index
                ]
                if rate_values:
                    natural_class_rates[layer_key][label].append(
                        {
                            "progress_bin": bin_index,
                            "progress": float(bin_index / (int(progress_bins) - 1)),
                            **_mean_ci_by_seed(
                                rate_values,
                                bootstrap_samples=int(bootstrap_samples),
                                random_seed=(
                                    20260833
                                    + layer * 101
                                    + bin_index
                                    + (100003 if label == "next_k_plus_1" else 0)
                                ),
                            ),
                        }
                    )

    for bank_index, bank in enumerate(banks):
        paths[bank] = {}
        retention[bank] = {}
        for layer in layers:
            layer_key = f"L{layer}"
            paths[bank][layer_key] = {}
            retention[bank][layer_key] = []
            for dose in (-1, 0, 1):
                dose_key = str(dose)
                paths[bank][layer_key][dose_key] = []
                source_bins = clean_bin if dose == 0 else intervention_bin
                for bin_index in range(int(progress_bins)):
                    values = []
                    for key, soft_count in source_bins.items():
                        seed, receiver, active_layer, active_bank, active_dose, active_bin = key
                        expected_bank = "clean" if dose == 0 else bank
                        if (
                            active_layer == layer
                            and active_bank == expected_bank
                            and active_dose == dose
                            and active_bin == bin_index
                        ):
                            values.append((seed, soft_count - receiver))
                    if values:
                        paths[bank][layer_key][dose_key].append(
                            {
                                "progress_bin": bin_index,
                                "progress": float(bin_index / (int(progress_bins) - 1)),
                                **_mean_ci_by_seed(
                                    values,
                                    bootstrap_samples=int(bootstrap_samples),
                                    random_seed=(
                                        20260901
                                        + bank_index * 100003
                                        + layer * 101
                                        + (dose + 1) * 1009
                                        + bin_index
                                    ),
                                ),
                            }
                        )

            for bin_index in range(int(progress_bins)):
                triples = []
                for seed, receiver, active_layer, active_bank, dose, active_bin in intervention_bin:
                    if (
                        active_layer != layer
                        or active_bank != bank
                        or active_bin != bin_index
                        or dose not in {-1, 1}
                    ):
                        continue
                    current_key = (seed, receiver, layer, bank, dose, 0)
                    if current_key not in intervention_bin:
                        continue
                    clean_current = clean_bin[(seed, receiver, layer, "clean", 0, 0)]
                    clean_target = clean_bin[(seed, receiver, layer, "clean", 0, bin_index)]
                    x_value = intervention_bin[current_key] - clean_current
                    y_value = (
                        intervention_bin[(seed, receiver, layer, bank, dose, bin_index)]
                        - clean_target
                    )
                    triples.append((seed, x_value, y_value))
                if triples:
                    retention[bank][layer_key].append(
                        {
                            "progress_bin": bin_index,
                            "progress": float(bin_index / (int(progress_bins) - 1)),
                            **_slope_ci_by_seed(
                                triples,
                                bootstrap_samples=int(bootstrap_samples),
                                random_seed=(
                                    20260902
                                    + bank_index * 100003
                                    + layer * 101
                                    + bin_index
                                ),
                            ),
                        }
                    )

    endpoint_discrete: dict[str, dict[str, dict[str, Any]]] = {}
    for bank in banks:
        endpoint_discrete[bank] = {}
        for layer in layers:
            selected = [
                row
                for row in interventions
                if str(row["bank"]) == bank
                and int(row["read_layer"]) == layer
                and bool(row["is_next_boundary"])
            ]
            endpoint_discrete[bank][f"L{layer}"] = {
                "exact": int(sum(bool(row["exact_transition_target"]) for row in selected)),
                "n": len(selected),
                "changed_from_clean": int(
                    sum(bool(row["prediction_changed_from_clean"]) for row in selected)
                ),
                "directionally_correct_changed": int(
                    sum(
                        bool(row["directionally_correct_change"])
                        for row in selected
                    )
                ),
            }
    return {
        "natural_increment": natural,
        "natural_adjacent_margin_k_plus_1_minus_k": natural_margin,
        "natural_class_rates": natural_class_rates,
        "dose_paths_centered_on_k": paths,
        "normalized_offset_retention": retention,
        "next_boundary_discrete": endpoint_discrete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--discovery-generations", type=Path, required=True)
    parser.add_argument("--discovery-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--evaluation-generations", type=Path, required=True)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--raw-field-cache", type=Path, required=True)
    parser.add_argument("--boundary-states", type=Path, required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--receivers", type=int, nargs="+", default=(3, 4, 5, 6, 7, 8))
    parser.add_argument("--clamp-start-layer", type=int, default=14)
    parser.add_argument("--clamp-end-layer", type=int, default=23)
    parser.add_argument("--read-layers", type=int, nargs="+", default=(15, 16, 24))
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--progress-bins", type=int, default=12)
    parser.add_argument("--oof-folds", type=int, default=4)
    parser.add_argument("--field-scale", type=float, default=1.0)
    parser.add_argument("--boundary-scale", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    discovery_seeds = tuple(int(value) for value in args.discovery_seeds)
    evaluation_seeds = tuple(int(value) for value in args.evaluation_seeds)
    receivers = tuple(int(value) for value in args.receivers)
    read_layers = tuple(sorted({int(value) for value in args.read_layers}))
    clamp_layers = tuple(
        range(int(args.clamp_start_layer), int(args.clamp_end_layer) + 1)
    )
    folds = int(args.oof_folds)
    if len(discovery_seeds) < folds or folds < 2:
        raise ValueError("OOF movie requires at least two folds")
    if any(receiver < 2 or receiver > 9 for receiver in receivers):
        raise ValueError("Receivers must permit a central tangent and next item")
    if max(clamp_layers) >= max(read_layers):
        raise ValueError("The final read layer must follow every clamp layer")
    if int(args.progress_bins) < 2:
        raise ValueError("Token movie needs at least two progress bins")

    discovery_rows = _read_rows(args.discovery_generations, discovery_seeds)
    evaluation_rows = _read_rows(args.evaluation_generations, evaluation_seeds)
    probe_npz = np.load(args.frozen_probes)
    alpha = float(np.asarray(probe_npz["alpha"])[0])
    frozen_layers = set(int(value) for value in np.asarray(probe_npz["frozen_layers"]))
    if any(layer not in frozen_layers for layer in read_layers):
        raise ValueError("One or more movie read layers have no frozen probe")
    probes = {
        layer: {
            "mean": np.asarray(probe_npz[f"layer_{layer}_mean"], dtype=np.float32),
            "weights": np.asarray(
                probe_npz[f"layer_{layer}_weights"], dtype=np.float32
            ),
            "alpha": alpha,
        }
        for layer in read_layers
    }

    model, tokenizer, adapter = _model(args)
    raw_panel = build_or_load_raw_kv_panel(
        model,
        tokenizer,
        adapter,
        path=args.raw_field_cache,
        rows=discovery_rows,
        seeds=discovery_seeds,
        layers=clamp_layers,
        bins=int(args.bins),
    )
    seed_to_index = {seed: index for index, seed in enumerate(discovery_seeds)}
    fold_by_seed = {
        seed: index % folds for index, seed in enumerate(sorted(discovery_seeds))
    }
    field_by_fold = {}
    for fold in range(folds):
        train_indices = [
            seed_to_index[seed]
            for seed in discovery_seeds
            if fold_by_seed[seed] != fold
        ]
        field_by_fold[fold] = fit_kv_field(
            raw_panel,
            layers=clamp_layers,
            train_indices=train_indices,
            alpha=alpha,
        )
        print(f"[transition-movie] fit OOF fold={fold} n={len(train_indices)}", flush=True)

    boundary_geometry = {
        receiver: frozen_layer_geometry(
            args.boundary_states,
            layers=clamp_layers,
            receiver_count=receiver,
            alpha=alpha,
        )
        for receiver in receivers
    }
    kv_spec = BankSpec("all_history_kv", "all_history", "kv", clamp_layers)
    results: list[dict[str, Any]] = []
    for seed in evaluation_seeds:
        if seed not in fold_by_seed:
            raise ValueError("Movie evaluation seeds must belong to the OOF discovery panel")
        _bases, tangents = field_by_fold[fold_by_seed[seed]]
        row = evaluation_rows[seed]
        source, _blank, registry, scrub_audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        bins_by_occurrence = item_bin_positions(registry.trace_items, bins=int(args.bins))
        for receiver in receivers:
            current_boundary, current_audit = select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=receiver
            )
            next_boundary, next_audit = select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=receiver + 1
            )
            metadata = transition_position_metadata(
                registry.trace_items,
                receiver=receiver,
                current_boundary=current_boundary,
                next_boundary=next_boundary,
            )
            positions = tuple(int(value["position"]) for value in metadata)
            token_metadata = []
            for value in metadata:
                position = int(value["position"])
                token_id = int(source.input_ids[position])
                token_metadata.append(
                    {
                        **value,
                        "progress_bin": progress_bin(
                            float(value["transition_progress"]),
                            bins=int(args.progress_bins),
                        ),
                        "token_id": token_id,
                        "token_text": tokenizer.decode(
                            [token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                    }
                )

            clean_states = capture_decoder_block_input_states(
                model, adapter, source, positions, layers=read_layers
            )
            clean_predictions: dict[tuple[int, int], int] = {}
            for layer in read_layers:
                for position_index, state in enumerate(clean_states[layer]):
                    decoded = decode_count_probe(probes[layer], state.numpy())
                    clean_predictions[(layer, position_index)] = int(
                        decoded["probe_prediction"]
                    )
                    results.append(
                        {
                            "schema_version": "kv_transition_movie_v1",
                            "model_label": str(args.model),
                            "seed": seed,
                            "request_id": str(row["request_id"]),
                            "oof_fold": fold_by_seed[seed],
                            "receiver": receiver,
                            "bank": "clean",
                            "dose": 0,
                            "read_layer": layer,
                            "expected_transition_target": receiver + (
                                1 if bool(token_metadata[position_index]["is_next_boundary"]) else 0
                            ),
                            "exact_transition_target": bool(
                                bool(token_metadata[position_index]["is_current_boundary"])
                                and int(decoded["probe_prediction"]) == receiver
                                or bool(token_metadata[position_index]["is_next_boundary"])
                                and int(decoded["probe_prediction"]) == receiver + 1
                            ),
                            "prediction_changed_from_clean": False,
                            "directionally_correct_change": False,
                            "scrub_construction": scrub_audit["construction"],
                            "current_boundary_kind": current_audit["boundary_kind"],
                            "next_boundary_kind": next_audit["boundary_kind"],
                            **token_metadata[position_index],
                            **decoded,
                        }
                    )

            _boundary_bases, boundary_tangents, _boundary_ranks = boundary_geometry[receiver]
            for dose in (-1, 1):
                boundary_directions = {
                    layer: (
                        float(dose)
                        * float(args.boundary_scale)
                        * boundary_tangents[layer]
                    ).astype(np.float32)
                    for layer in clamp_layers
                }
                runs = (
                    ("boundary_only", {}),
                    (
                        "all_history_kv",
                        build_kv_directions(
                            kv_spec,
                            receiver=receiver,
                            dose=dose,
                            scale=float(args.field_scale),
                            bins_by_occurrence=bins_by_occurrence,
                            tangents=tangents,
                        ),
                    ),
                )
                for bank, kv_directions in runs:
                    captured, boundary_apps, kv_apps, boundary_norms, kv_norms, read_apps = (
                        add_boundary_and_kv_deltas_capture_layers(
                            model,
                            adapter,
                            source,
                            boundary_position=current_boundary,
                            boundary_directions=boundary_directions,
                            kv_directions=kv_directions,
                            read_positions=positions,
                            read_layers=read_layers,
                        )
                    )
                    for layer in read_layers:
                        for position_index, state in enumerate(captured[layer]):
                            decoded = decode_count_probe(probes[layer], state.numpy())
                            prediction = int(decoded["probe_prediction"])
                            clean_prediction = clean_predictions[(layer, position_index)]
                            is_current = bool(token_metadata[position_index]["is_current_boundary"])
                            is_next = bool(token_metadata[position_index]["is_next_boundary"])
                            target = receiver + dose + (1 if is_next else 0)
                            changed = prediction != clean_prediction
                            results.append(
                                {
                                    "schema_version": "kv_transition_movie_v1",
                                    "model_label": str(args.model),
                                    "seed": seed,
                                    "request_id": str(row["request_id"]),
                                    "oof_fold": fold_by_seed[seed],
                                    "receiver": receiver,
                                    "bank": bank,
                                    "dose": dose,
                                    "read_layer": layer,
                                    "expected_transition_target": target,
                                    "exact_transition_target": bool(
                                        (is_current or is_next) and prediction == target
                                    ),
                                    "prediction_changed_from_clean": changed,
                                    "directionally_correct_change": bool(
                                        changed
                                        and np.sign(prediction - clean_prediction) == np.sign(dose)
                                    ),
                                    "scrub_construction": scrub_audit["construction"],
                                    "current_boundary_kind": current_audit["boundary_kind"],
                                    "next_boundary_kind": next_audit["boundary_kind"],
                                    "boundary_hooks_all_once": bool(
                                        boundary_apps
                                        and all(value == 1 for value in boundary_apps.values())
                                    ),
                                    "kv_hooks_all_once": bool(
                                        all(value == 1 for value in kv_apps.values())
                                    ),
                                    "read_hooks_all_once": bool(
                                        read_apps
                                        and all(value == 1 for value in read_apps.values())
                                    ),
                                    "boundary_realized_l2_norm_mean": float(
                                        np.mean(list(boundary_norms.values()))
                                    ),
                                    "kv_realized_l2_norm_mean": float(
                                        np.mean(list(kv_norms.values()))
                                    ) if kv_norms else 0.0,
                                    **token_metadata[position_index],
                                    **decoded,
                                }
                            )
            print(
                f"[transition-movie] seed={seed} receiver={receiver} tokens={len(positions)} complete",
                flush=True,
            )

    summary = {
        "schema_version": "kv_transition_movie_v1",
        "model_label": str(args.model),
        "discovery_seeds": list(discovery_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "oof_folds": folds,
        "receivers": list(receivers),
        "kv_item_bins": int(args.bins),
        "trajectory_progress_bins": int(args.progress_bins),
        "field_scale": float(args.field_scale),
        "boundary_scale": float(args.boundary_scale),
        "clamp_layers": list(clamp_layers),
        "read_layers": list(read_layers),
        "banks": ["boundary_only", "all_history_kv"],
        "outcomes": summarize_movie(
            results,
            progress_bins=int(args.progress_bins),
            bootstrap_samples=int(args.bootstrap_samples),
        ),
        "intermediate_token_probe_is_out_of_training_site": True,
        "formal_native_boundary_sites": ["B_k", "B_(k+1)"],
        "input_tokens_changed_by_intervention": False,
        "diagnostic_suffix_used": False,
        "teacher_forced_full_marker_scrubbed_native_trace": True,
        "selection_status": "frozen discovery diagnostic; no confirmation seeds consumed",
    }
    _atomic_jsonl(args.output, results)
    _atomic_json(args.summary, summary)
    print(f"[transition-movie] wrote {len(results)} rows", flush=True)


if __name__ == "__main__":
    main()
