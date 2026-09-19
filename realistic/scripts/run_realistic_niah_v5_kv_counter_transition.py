#!/usr/bin/env python3
"""OOF discovery screen for a distributed K/V-cache counter transition."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
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
    count_probe_subspace,
    fit_dual_ridge_count_probe,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from realistic_niah_v5.kv_counter_transition import (  # noqa: E402
    add_boundary_and_kv_deltas_capture,
    capture_kv_item_bin_means,
    history_occurrences,
    item_bin_positions,
    projection_kinds,
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


@dataclass(frozen=True)
class BankSpec:
    name: str
    scope: str
    projection: str
    layers: tuple[int, ...]


def default_bank_specs(layers: Sequence[int]) -> tuple[BankSpec, ...]:
    active = tuple(int(value) for value in layers)
    if len(active) < 6 or active != tuple(range(active[0], active[-1] + 1)):
        raise ValueError("The coarse KV screen expects a contiguous layer band")
    first_cut = active[0] + len(active) // 3
    second_cut = active[0] + (2 * len(active)) // 3
    early = tuple(layer for layer in active if layer < first_cut)
    middle = tuple(layer for layer in active if first_cut <= layer < second_cut)
    late = tuple(layer for layer in active if layer >= second_cut)
    return (
        BankSpec("all_history_k", "all_history", "k", active),
        BankSpec("all_history_v", "all_history", "v", active),
        BankSpec("all_history_kv", "all_history", "kv", active),
        BankSpec("last4_kv", "last_4", "kv", active),
        BankSpec("all_history_kv_early", "all_history", "kv", early),
        BankSpec("all_history_kv_middle", "all_history", "kv", middle),
        BankSpec("all_history_kv_late", "all_history", "kv", late),
    )


def _read_rows(path: Path, seeds: Sequence[int]) -> dict[int, dict[str, Any]]:
    wanted = {int(value) for value in seeds}
    selected: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seed = int(row["seed"])
        if seed in wanted:
            selected[seed] = row
    if set(selected) != wanted:
        raise ValueError(f"One or more requested seeds are absent from {path}")
    return selected


def adjacent_count_tangent(
    states: np.ndarray,
    *,
    occurrence: int,
    basis: np.ndarray,
) -> np.ndarray:
    """Projected one-count displacement, central except at endpoints."""

    panel = np.asarray(states, dtype=np.float64)
    active_basis = np.asarray(basis, dtype=np.float64)
    count = int(occurrence)
    if panel.ndim != 3 or int(panel.shape[1]) != 10:
        raise ValueError("KV field states must have shape [seed,10,width]")
    if not 1 <= count <= 10:
        raise ValueError("KV field occurrence is outside 1..10")
    centroids = panel.mean(axis=0)
    if count == 1:
        raw = centroids[1] - centroids[0]
    elif count == 10:
        raw = centroids[9] - centroids[8]
    else:
        raw = 0.5 * (centroids[count] - centroids[count - 2])
    return ((raw @ active_basis) @ active_basis.T).astype(np.float32)


def build_or_load_raw_kv_panel(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    path: Path,
    rows: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
    layers: Sequence[int],
    bins: int,
) -> np.ndarray:
    """Cache discovery K/V bin means without fitting on evaluation folds."""

    active_seeds = tuple(int(value) for value in seeds)
    active_layers = tuple(int(value) for value in layers)
    if path.exists():
        payload = np.load(path)
        if np.asarray(payload["seeds"]).tolist() != list(active_seeds):
            raise ValueError("Frozen raw KV panel seeds changed")
        if np.asarray(payload["layers"]).tolist() != list(active_layers):
            raise ValueError("Frozen raw KV panel layers changed")
        if int(np.asarray(payload["bins"])[0]) != int(bins):
            raise ValueError("Frozen raw KV panel bin count changed")
        values = np.asarray(payload["states"], dtype=np.float32)
        if values.shape[:5] != (
            len(active_seeds),
            len(active_layers),
            2,
            10,
            int(bins),
        ):
            raise ValueError("Frozen raw KV panel shape changed")
        return values

    values: np.ndarray | None = None
    for seed_index, seed in enumerate(active_seeds):
        source, _blank, registry, _audit = build_diagnostic_bases(
            rows[seed],
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        captured = capture_kv_item_bin_means(
            model,
            adapter,
            source,
            registry.trace_items,
            layers=active_layers,
            bins=int(bins),
        )
        if values is None:
            width = int(captured[(active_layers[0], "k")].shape[-1])
            values = np.empty(
                (
                    len(active_seeds),
                    len(active_layers),
                    2,
                    10,
                    int(bins),
                    width,
                ),
                dtype=np.float32,
            )
        for layer_index, layer in enumerate(active_layers):
            for kind_index, kind in enumerate(("k", "v")):
                panel = captured[(layer, kind)].numpy().astype(np.float32)
                if panel.shape != values[seed_index, layer_index, kind_index].shape:
                    raise ValueError("A KV projection width changed across layers")
                values[seed_index, layer_index, kind_index] = panel
        print(f"[kv-field] captured seed={seed}", flush=True)
    if values is None:
        raise RuntimeError("No discovery KV states were captured")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        states=values.astype(np.float16),
        seeds=np.asarray(active_seeds, dtype=np.int64),
        layers=np.asarray(active_layers, dtype=np.int64),
        bins=np.asarray([int(bins)], dtype=np.int64),
    )
    temporary.replace(path)
    return values


def fit_kv_field(
    panel: np.ndarray,
    *,
    layers: Sequence[int],
    train_indices: Sequence[int],
    alpha: float,
) -> tuple[
    dict[tuple[int, str, int], np.ndarray],
    dict[tuple[int, str, int, int], np.ndarray],
]:
    """Fit count bases/tangents using only the specified discovery rows."""

    values = np.asarray(panel, dtype=np.float32)
    indices = np.asarray(tuple(int(value) for value in train_indices), dtype=np.int64)
    active_layers = tuple(int(value) for value in layers)
    if indices.size < 2 or values.ndim != 6:
        raise ValueError("KV field fitting needs multiple discovery rows")
    labels = np.tile(np.arange(1, 11, dtype=np.int64), indices.size)
    bases: dict[tuple[int, str, int], np.ndarray] = {}
    tangents: dict[tuple[int, str, int, int], np.ndarray] = {}
    for layer_index, layer in enumerate(active_layers):
        for kind_index, kind in enumerate(("k", "v")):
            for bin_index in range(int(values.shape[4])):
                states = values[indices, layer_index, kind_index, :, bin_index, :]
                probe = fit_dual_ridge_count_probe(
                    states.reshape(-1, states.shape[-1]), labels, alpha=float(alpha)
                )
                basis = count_probe_subspace(probe)
                bases[(layer, kind, bin_index)] = basis
                for occurrence in range(1, 11):
                    tangents[(layer, kind, bin_index, occurrence)] = (
                        adjacent_count_tangent(
                            states, occurrence=occurrence, basis=basis
                        )
                    )
    return bases, tangents


def build_kv_directions(
    spec: BankSpec,
    *,
    receiver: int,
    dose: int,
    scale: float,
    bins_by_occurrence: Mapping[int, Sequence[Sequence[int]]],
    tangents: Mapping[tuple[int, str, int, int], np.ndarray],
) -> dict[tuple[int, str], dict[int, np.ndarray]]:
    output: dict[tuple[int, str], dict[int, np.ndarray]] = {}
    occurrences = history_occurrences(receiver, spec.scope)
    for layer in spec.layers:
        for kind in projection_kinds(spec.projection):
            position_values: dict[int, np.ndarray] = {}
            for occurrence in occurrences:
                for bin_index, positions in enumerate(bins_by_occurrence[occurrence]):
                    delta = (
                        float(dose)
                        * float(scale)
                        * tangents[(layer, kind, bin_index, occurrence)]
                    ).astype(np.float32)
                    for position in positions:
                        position_values[int(position)] = delta
            output[(layer, kind)] = position_values
    return output


def orthogonal_kv_tangents(
    bases: Mapping[tuple[int, str, int], np.ndarray],
    tangents: Mapping[tuple[int, str, int, int], np.ndarray],
    *,
    seed: int,
) -> dict[tuple[int, str, int, int], np.ndarray]:
    """Build deterministic equal-norm directions outside each count subspace."""

    output: dict[tuple[int, str, int, int], np.ndarray] = {}
    for index, (key, tangent_value) in enumerate(sorted(tangents.items())):
        layer, kind, bin_index, _occurrence = key
        basis = np.asarray(bases[(layer, kind, bin_index)], dtype=np.float64)
        tangent = np.asarray(tangent_value, dtype=np.float64).reshape(-1)
        rng = np.random.default_rng(int(seed) + index * 1009)
        random = rng.standard_normal(tangent.shape)
        random = random - (random @ basis) @ basis.T
        random = random - (
            float(random @ tangent) / max(float(tangent @ tangent), 1e-12)
        ) * tangent
        random_norm = float(np.linalg.norm(random))
        target_norm = float(np.linalg.norm(tangent))
        if random_norm <= 1e-12 or target_norm <= 1e-12:
            raise RuntimeError("Could not construct a nonzero orthogonal KV control")
        output[key] = (random * (target_norm / random_norm)).astype(np.float32)
    return output


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    banks = sorted({str(row["bank"]) for row in rows})
    for bank in banks:
        active = [row for row in rows if str(row["bank"]) == bank]
        clean = {
            (int(row["seed"]), int(row["receiver"]), int(row["horizon"])): row
            for row in active
            if int(row["dose"]) == 0
        }
        counterfactual = [row for row in active if int(row["dose"]) != 0]
        current_x: list[float] = []
        next_y: list[float] = []
        for row in counterfactual:
            key0 = (int(row["seed"]), int(row["receiver"]), 0)
            keyh = (int(row["seed"]), int(row["receiver"]), int(row["horizon"]))
            if int(row["horizon"]) == 0:
                current_x.append(
                    float(row["probe_softmax_expected_count"])
                    - float(clean[key0]["probe_softmax_expected_count"])
                )
            else:
                next_y.append(
                    float(row["probe_softmax_expected_count"])
                    - float(clean[keyh]["probe_softmax_expected_count"])
                )
        # Rows are emitted in dose-major order at each horizon, so regroup by key.
        shifts: dict[tuple[int, int, int], float] = {}
        for row in counterfactual:
            key = (int(row["seed"]), int(row["receiver"]), int(row["dose"]))
            clean_key = (int(row["seed"]), int(row["receiver"]), int(row["horizon"]))
            shift = float(row["probe_softmax_expected_count"]) - float(
                clean[clean_key]["probe_softmax_expected_count"]
            )
            shifts[(key[0], key[1], key[2], int(row["horizon"]))] = shift
        x = []
        y = []
        for seed, receiver, dose, horizon in sorted(shifts):
            if horizon != 1:
                continue
            x.append(shifts[(seed, receiver, dose, 0)])
            y.append(shifts[(seed, receiver, dose, 1)])
        discrete: dict[str, Any] = {}
        for horizon in (0, 1):
            values = [row for row in counterfactual if int(row["horizon"]) == horizon]
            changed = 0
            correct_changed = 0
            for row in values:
                natural = int(
                    clean[(int(row["seed"]), int(row["receiver"]), horizon)][
                        "probe_prediction"
                    ]
                )
                prediction = int(row["probe_prediction"])
                dose = int(row["dose"])
                changed += int(prediction != natural)
                correct_changed += int(
                    prediction != natural and (prediction - natural) * dose > 0
                )
            discrete[f"horizon_{horizon}"] = {
                "exact": int(sum(bool(row["exact"]) for row in values)),
                "n": len(values),
                "changed": changed,
                "directionally_correct_changed": correct_changed,
            }
        output[bank] = {
            "pooled_current_to_next_retention": through_origin_slope(x, y),
            "current_mean_abs_soft_shift": float(np.mean(np.abs(x))),
            "next_mean_abs_soft_shift": float(np.mean(np.abs(y))),
            "discrete": discrete,
        }
    return output


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
    parser.add_argument("--read-layer", type=int, default=24)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--oof-folds", type=int, default=4)
    parser.add_argument("--field-scale", type=float, default=1.0)
    parser.add_argument("--boundary-scale", type=float, default=1.0)
    parser.add_argument("--bank-names", nargs="+")
    parser.add_argument("--include-orthogonal-control", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "kv-counter-transition"

    discovery_seeds = tuple(int(value) for value in args.discovery_seeds)
    evaluation_seeds = tuple(int(value) for value in args.evaluation_seeds)
    receivers = tuple(int(value) for value in args.receivers)
    read_layer = int(args.read_layer)
    layers = tuple(range(int(args.clamp_start_layer), read_layer))
    folds = int(args.oof_folds)
    if len(discovery_seeds) < folds or folds < 2:
        raise ValueError("OOF discovery requires at least two folds")
    if any(receiver < 2 or receiver > 9 for receiver in receivers):
        raise ValueError("Receivers must permit central boundary tangents and a next item")
    if not np.isfinite(float(args.field_scale)) or float(args.field_scale) <= 0:
        raise ValueError("Field scale must be finite and positive")
    if not np.isfinite(float(args.boundary_scale)) or float(args.boundary_scale) <= 0:
        raise ValueError("Boundary scale must be finite and positive")

    discovery_rows = _read_rows(args.discovery_generations, discovery_seeds)
    evaluation_rows = _read_rows(args.evaluation_generations, evaluation_seeds)
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
    raw_panel = build_or_load_raw_kv_panel(
        model,
        tokenizer,
        adapter,
        path=args.raw_field_cache,
        rows=discovery_rows,
        seeds=discovery_seeds,
        layers=layers,
        bins=int(args.bins),
    )
    seed_to_index = {seed: index for index, seed in enumerate(discovery_seeds)}
    fold_by_seed = {
        seed: index % folds for index, seed in enumerate(sorted(discovery_seeds))
    }
    field_by_fold: dict[
        int,
        tuple[
            dict[tuple[int, str, int], np.ndarray],
            dict[tuple[int, str, int, int], np.ndarray],
        ],
    ] = {}
    for fold in range(folds):
        train_indices = [
            seed_to_index[seed]
            for seed in discovery_seeds
            if fold_by_seed[seed] != fold
        ]
        field_by_fold[fold] = fit_kv_field(
            raw_panel, layers=layers, train_indices=train_indices, alpha=alpha
        )
        print(f"[kv-field] fit OOF fold={fold} n={len(train_indices)}", flush=True)

    boundary_geometry = {
        receiver: frozen_layer_geometry(
            args.boundary_states,
            layers=layers,
            receiver_count=receiver,
            alpha=alpha,
        )
        for receiver in receivers
    }
    specs = default_bank_specs(layers)
    if args.bank_names:
        requested_banks = tuple(str(value) for value in args.bank_names)
        by_name = {spec.name: spec for spec in specs}
        if any(name not in by_name for name in requested_banks):
            raise ValueError("One or more requested KV banks are unknown")
        specs = tuple(by_name[name] for name in requested_banks)
    results: list[dict[str, Any]] = []
    for seed in evaluation_seeds:
        if seed not in fold_by_seed:
            raise ValueError("This discovery runner only accepts OOF evaluation seeds")
        bases, tangents = field_by_fold[fold_by_seed[seed]]
        orthogonal_tangents = (
            orthogonal_kv_tangents(
                bases,
                tangents,
                seed=20261201 + seed * 100,
            )
            if bool(args.include_orthogonal_control)
            else {}
        )
        row = evaluation_rows[seed]
        source, _blank, registry, scrub_audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        bins_by_occurrence = item_bin_positions(
            registry.trace_items, bins=int(args.bins)
        )
        for receiver in receivers:
            boundary_positions = {
                occurrence: select_post_item_boundary_position(
                    source, registry, tokenizer, occurrence=occurrence
                )[0]
                for occurrence in (receiver, receiver + 1)
            }
            read_positions = (
                boundary_positions[receiver],
                boundary_positions[receiver + 1],
            )
            clean_states = capture_decoder_block_input_states(
                model, adapter, source, read_positions, layers=(read_layer,)
            )[read_layer]
            aligned_banks = tuple(spec.name for spec in specs)
            control_banks = (
                tuple(f"{spec.name}_orthogonal" for spec in specs)
                if bool(args.include_orthogonal_control)
                else ()
            )
            all_banks = ("boundary_only",) + aligned_banks + control_banks
            for bank in all_banks:
                for horizon, state in enumerate(clean_states):
                    decoded = decode_count_probe(read_probe, state.numpy())
                    results.append(
                        {
                            "schema_version": "kv_counter_transition_v1",
                            "model_label": str(args.model),
                            "seed": seed,
                            "request_id": str(row["request_id"]),
                            "oof_fold": fold_by_seed[seed],
                            "receiver": receiver,
                            "bank": bank,
                            "condition": "aligned_count_translation",
                            "dose": 0,
                            "horizon": horizon,
                            "expected_count": receiver + horizon,
                            "exact": bool(decoded["probe_prediction"] == receiver + horizon),
                            "read_layer": read_layer,
                            "clamp_layers": list(layers),
                            "scrub_construction": scrub_audit["construction"],
                            "boundary_hook_applications": {},
                            "kv_hook_applications": {},
                            "boundary_realized_l2_norms": {},
                            "kv_realized_l2_norms": {},
                            "read_hook_applications": 1,
                            **decoded,
                        }
                    )
            _boundary_bases, boundary_tangents, _boundary_ranks = (
                boundary_geometry[receiver]
            )
            for dose in (-1, 1):
                boundary_directions = {
                    layer: (
                        float(dose)
                        * float(args.boundary_scale)
                        * boundary_tangents[layer]
                    ).astype(np.float32)
                    for layer in layers
                }
                runs: list[tuple[str, dict[tuple[int, str], dict[int, np.ndarray]]]] = [
                    ("boundary_only", {})
                ]
                for spec in specs:
                    runs.append(
                        (
                            spec.name,
                            build_kv_directions(
                                spec,
                                receiver=receiver,
                                dose=dose,
                                scale=float(args.field_scale),
                                bins_by_occurrence=bins_by_occurrence,
                                tangents=tangents,
                            ),
                        )
                    )
                    if bool(args.include_orthogonal_control):
                        runs.append(
                            (
                                f"{spec.name}_orthogonal",
                                build_kv_directions(
                                    spec,
                                    receiver=receiver,
                                    dose=dose,
                                    scale=float(args.field_scale),
                                    bins_by_occurrence=bins_by_occurrence,
                                    tangents=orthogonal_tangents,
                                ),
                            )
                        )
                for bank, kv_directions in runs:
                    run = add_boundary_and_kv_deltas_capture(
                        model,
                        adapter,
                        source,
                        boundary_position=boundary_positions[receiver],
                        boundary_directions=boundary_directions,
                        kv_directions=kv_directions,
                        read_positions=read_positions,
                        read_layer=read_layer,
                    )
                    states, boundary_apps, kv_apps, boundary_norms, kv_norms, read_apps = run
                    for horizon, state in enumerate(states):
                        expected = receiver + horizon + dose
                        decoded = decode_count_probe(read_probe, state.numpy())
                        results.append(
                            {
                                "schema_version": "kv_counter_transition_v1",
                                "model_label": str(args.model),
                                "seed": seed,
                                "request_id": str(row["request_id"]),
                                "oof_fold": fold_by_seed[seed],
                                "receiver": receiver,
                                "bank": bank,
                                "condition": "aligned_count_translation",
                                "dose": dose,
                                "horizon": horizon,
                                "expected_count": expected,
                                "exact": bool(decoded["probe_prediction"] == expected),
                                "read_layer": read_layer,
                                "clamp_layers": list(layers),
                                "scrub_construction": scrub_audit["construction"],
                                "boundary_hook_applications": boundary_apps,
                                "kv_hook_applications": kv_apps,
                                "boundary_realized_l2_norms": boundary_norms,
                                "kv_realized_l2_norms": kv_norms,
                                "read_hook_applications": read_apps,
                                **decoded,
                            }
                        )
            print(f"[kv-transition] seed={seed} receiver={receiver} complete", flush=True)

    summary = {
        "schema_version": "kv_counter_transition_v1",
        "model_label": str(args.model),
        "discovery_seeds": list(discovery_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "oof_folds": folds,
        "receivers": list(receivers),
        "bins": int(args.bins),
        "field_scale": float(args.field_scale),
        "boundary_scale": float(args.boundary_scale),
        "clamp_layers": list(layers),
        "read_layer": read_layer,
        "banks": {
            spec.name: {
                "scope": spec.scope,
                "projection": spec.projection,
                "layers": list(spec.layers),
            }
            for spec in specs
        },
        "orthogonal_controls_included": bool(args.include_orthogonal_control),
        "outcomes": summarize(results),
        "input_tokens_changed_by_intervention": False,
        "diagnostic_suffix_used": False,
        "teacher_forced_full_marker_scrubbed_native_trace": True,
    }
    _atomic_jsonl(args.output, results)
    _atomic_json(args.summary, summary)
    print(f"[kv-transition] wrote {len(results)} rows", flush=True)


if __name__ == "__main__":
    main()
