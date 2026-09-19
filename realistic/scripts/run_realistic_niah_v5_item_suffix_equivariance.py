#!/usr/bin/env python3
"""Test a distributed count offset in endpoint-aligned native item spans.

Discovery traces freeze an occurrence-specific count-tangent field for each of
the final W tokens of a list item.  Evaluation then adds coherent +/-1 fields
to the current item or to several historical items while leaving every input
token and the full teacher-forced continuation unchanged.  Later native item
boundaries read whether the induced count offset survives a +1 transition.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    norm_matched_orthogonal_replacement,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from scripts.run_realistic_niah_v5_boundary_equivariance import (  # noqa: E402
    decode_count_probe,
    frozen_layer_geometry,
    local_count_tangent,
)
from scripts.run_realistic_niah_v5_boundary_history_equivariance import (  # noqa: E402
    add_history_layer_deltas_and_capture_positions,
    summarize_history_slopes,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)


def item_suffix_geometries(receiver: int) -> dict[str, tuple[int, ...]]:
    receiver = int(receiver)
    if receiver < 5:
        raise ValueError("The item-suffix history sweep requires receiver >= 5")
    return {
        "current_item_suffix": (receiver,),
        "current_plus_item_2_suffix": (2, receiver),
        "last_2_item_suffixes": tuple(range(receiver - 1, receiver + 1)),
        "last_3_item_suffixes": tuple(range(receiver - 2, receiver + 1)),
        "last_4_item_suffixes": tuple(range(receiver - 3, receiver + 1)),
    }


def suffix_positions_for_occurrences(
    registry: Any,
    occurrences: Sequence[int],
    *,
    width: int,
) -> dict[int, tuple[int, ...]]:
    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    active_width = int(width)
    if active_width < 1:
        raise ValueError("Suffix width must be positive")
    output: dict[int, tuple[int, ...]] = {}
    for occurrence in occurrences:
        value = int(occurrence)
        if not 1 <= value <= len(items):
            raise ValueError("Suffix occurrence is outside the registered list")
        start, end = items[value - 1]
        if end - start < active_width:
            raise ValueError(
                f"Item {value} is shorter than suffix width {active_width}"
            )
        output[value] = tuple(range(end - active_width, end))
    flattened = [position for positions in output.values() for position in positions]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Endpoint-aligned item suffixes overlap")
    return output


def union_subspace_basis(*values: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the union of nonempty column spans."""

    active = [np.asarray(value, dtype=np.float64) for value in values]
    if not active or any(value.ndim != 2 for value in active):
        raise ValueError("Union subspaces must be nonempty matrices")
    widths = {int(value.shape[0]) for value in active}
    if len(widths) != 1:
        raise ValueError("Union subspaces disagree on hidden width")
    joined = np.concatenate(active, axis=1)
    if joined.shape[1] == 0:
        return np.zeros((joined.shape[0], 0), dtype=np.float32)
    left, singular, _right = np.linalg.svd(joined, full_matrices=False)
    tolerance = max(joined.shape) * np.finfo(np.float64).eps * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    return left[:, :rank].astype(np.float32)


def _read_rows(path: Path, seeds: Sequence[int]) -> dict[int, dict[str, Any]]:
    wanted = {int(value) for value in seeds}
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {int(row["seed"]): row for row in rows if int(row["seed"]) in wanted}
    if set(selected) != wanted:
        raise ValueError(f"One or more requested seeds are absent from {path}")
    return selected


def build_or_load_suffix_field(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    field_path: Path,
    discovery_rows: Mapping[int, Mapping[str, Any]],
    discovery_seeds: Sequence[int],
    layers: Sequence[int],
    width: int,
    alpha: float,
) -> tuple[
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int, int], np.ndarray],
    dict[tuple[int, int], int],
]:
    """Freeze/load offset-specific bases and local count tangents."""

    active_layers = tuple(int(value) for value in layers)
    active_seeds = tuple(int(value) for value in discovery_seeds)
    active_width = int(width)
    if field_path.exists():
        payload = np.load(field_path)
        if np.asarray(payload["layers"]).tolist() != list(active_layers):
            raise ValueError("Frozen suffix field layers changed")
        if np.asarray(payload["discovery_seeds"]).tolist() != list(active_seeds):
            raise ValueError("Frozen suffix field discovery seeds changed")
        if int(np.asarray(payload["width"])[0]) != active_width:
            raise ValueError("Frozen suffix field width changed")
        bases = {
            (layer, offset): np.asarray(
                payload[f"basis_L{layer}_O{offset}"], dtype=np.float32
            )
            for layer in active_layers
            for offset in range(active_width)
        }
        tangents = {
            (layer, offset, occurrence): np.asarray(
                payload[f"tangent_L{layer}_O{offset}_C{occurrence}"],
                dtype=np.float32,
            )
            for layer in active_layers
            for offset in range(active_width)
            for occurrence in range(2, 6)
        }
        ranks = {
            (layer, offset): int(np.asarray(payload[f"rank_L{layer}_O{offset}"])[0])
            for layer in active_layers
            for offset in range(active_width)
        }
        return bases, tangents, ranks

    captured_by_seed: list[np.ndarray] = []
    for seed in active_seeds:
        row = discovery_rows[seed]
        source, _blank, registry, _audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        suffixes = suffix_positions_for_occurrences(
            registry, range(1, 11), width=active_width
        )
        positions = tuple(
            position
            for occurrence in range(1, 11)
            for position in suffixes[occurrence]
        )
        captured = capture_decoder_block_input_states(
            model, adapter, source, positions, layers=active_layers
        )
        seed_panel = np.stack(
            [
                captured[layer]
                .numpy()
                .reshape(10, active_width, -1)
                .astype(np.float16)
                for layer in active_layers
            ],
            axis=0,
        )
        captured_by_seed.append(seed_panel)
        print(f"[item-suffix-field] discovery seed={seed} captured", flush=True)
    panel = np.stack(captured_by_seed, axis=0)
    labels = np.tile(np.arange(1, 11, dtype=np.int64), len(active_seeds))
    bases: dict[tuple[int, int], np.ndarray] = {}
    tangents: dict[tuple[int, int, int], np.ndarray] = {}
    ranks: dict[tuple[int, int], int] = {}
    payload: dict[str, np.ndarray] = {
        "layers": np.asarray(active_layers, dtype=np.int64),
        "discovery_seeds": np.asarray(active_seeds, dtype=np.int64),
        "width": np.asarray([active_width], dtype=np.int64),
        "alpha": np.asarray([float(alpha)], dtype=np.float64),
    }
    for layer_index, layer in enumerate(active_layers):
        for offset in range(active_width):
            active = np.asarray(panel[:, layer_index, :, offset, :], dtype=np.float32)
            probe = fit_dual_ridge_count_probe(
                active.reshape(-1, active.shape[-1]), labels, alpha=float(alpha)
            )
            basis = count_probe_subspace(probe).astype(np.float32)
            bases[(layer, offset)] = basis
            ranks[(layer, offset)] = int(basis.shape[1])
            payload[f"basis_L{layer}_O{offset}"] = basis
            payload[f"rank_L{layer}_O{offset}"] = np.asarray(
                [basis.shape[1]], dtype=np.int64
            )
            for occurrence in range(2, 6):
                tangent = local_count_tangent(
                    active, receiver_count=occurrence, basis=basis
                )
                tangents[(layer, offset, occurrence)] = tangent
                payload[f"tangent_L{layer}_O{offset}_C{occurrence}"] = tangent
    field_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(field_path, **payload)
    print(f"[item-suffix-field] wrote {field_path}", flush=True)
    return bases, tangents, ranks


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
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--boundary-states", type=Path)
    parser.add_argument("--field-cache", type=Path, required=True)
    parser.add_argument("--suffix-width", type=int, default=4)
    parser.add_argument("--field-scale", type=float, default=1.0)
    parser.add_argument("--add-receiver-boundary", action="store_true")
    parser.add_argument("--boundary-scale", type=float, default=1.0)
    parser.add_argument("--receiver-occurrence", type=int, default=5)
    parser.add_argument("--doses", type=int, nargs="+", default=(-1, 0, 1))
    parser.add_argument("--horizons", type=int, default=3)
    parser.add_argument("--clamp-start-layer", type=int, default=14)
    parser.add_argument("--read-layer", type=int, default=24)
    parser.add_argument("--geometries", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "item-suffix-equivariance"

    discovery_seeds = tuple(int(value) for value in args.discovery_seeds)
    evaluation_seeds = tuple(int(value) for value in args.evaluation_seeds)
    receiver = int(args.receiver_occurrence)
    horizons = int(args.horizons)
    read_layer = int(args.read_layer)
    layers = tuple(range(int(args.clamp_start_layer), read_layer))
    width = int(args.suffix_width)
    field_scale = float(args.field_scale)
    boundary_scale = float(args.boundary_scale)
    geometries = item_suffix_geometries(receiver)
    geometry_names = (
        tuple(str(value) for value in args.geometries)
        if args.geometries
        else tuple(geometries)
    )
    if tuple(sorted({int(value) for value in args.doses})) != (-1, 0, 1):
        raise ValueError("The frozen suffix sweep requires doses -1 0 1")
    if not np.isfinite(field_scale) or field_scale <= 0.0:
        raise ValueError("Field scale must be finite and positive")
    if not np.isfinite(boundary_scale) or boundary_scale <= 0.0:
        raise ValueError("Boundary scale must be finite and positive")
    if bool(args.add_receiver_boundary) and args.boundary_states is None:
        raise ValueError("The joint intervention requires --boundary-states")
    if any(name not in geometries for name in geometry_names):
        raise ValueError("Unknown suffix history geometry")
    if horizons < 1 or receiver + horizons > 10:
        raise ValueError("Requested native-boundary horizons are outside 1..10")

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
    bases, tangents, ranks = build_or_load_suffix_field(
        model,
        tokenizer,
        adapter,
        field_path=args.field_cache,
        discovery_rows=discovery_rows,
        discovery_seeds=discovery_seeds,
        layers=layers,
        width=width,
        alpha=alpha,
    )
    boundary_bases: dict[int, np.ndarray] = {}
    boundary_tangents: dict[int, np.ndarray] = {}
    boundary_ranks: dict[int, int] = {}
    if bool(args.add_receiver_boundary):
        boundary_bases, boundary_tangents, boundary_ranks = frozen_layer_geometry(
            args.boundary_states,
            layers=layers,
            receiver_count=receiver,
            alpha=alpha,
        )
    required_occurrences = sorted(
        {occurrence for name in geometry_names for occurrence in geometries[name]}
    )
    results: list[dict[str, Any]] = []

    for seed in evaluation_seeds:
        row = evaluation_rows[seed]
        source, _blank, registry, _audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        suffixes = suffix_positions_for_occurrences(
            registry, required_occurrences, width=width
        )
        receiver_boundary_position = select_post_item_boundary_position(
            source, registry, tokenizer, occurrence=receiver
        )[0]
        suffix_patch_positions = tuple(
            position
            for occurrence in required_occurrences
            for position in suffixes[occurrence]
        )
        all_patch_positions = tuple(
            sorted(
                set(suffix_patch_positions)
                | (
                    {receiver_boundary_position}
                    if bool(args.add_receiver_boundary)
                    else set()
                )
            )
        )
        live_states = capture_decoder_block_input_states(
            model, adapter, source, all_patch_positions, layers=layers
        )
        live_by_layer_position = {
            layer: {
                position: live_states[layer][index].numpy()
                for index, position in enumerate(all_patch_positions)
            }
            for layer in layers
        }
        boundary_positions = {
            occurrence: select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=occurrence
            )[0]
            for occurrence in range(receiver, receiver + horizons + 1)
        }
        read_positions = tuple(
            boundary_positions[occurrence]
            for occurrence in range(receiver, receiver + horizons + 1)
        )
        clean_states = capture_decoder_block_input_states(
            model, adapter, source, read_positions, layers=(read_layer,)
        )[read_layer]

        for geometry_name in geometry_names:
            geometry_occurrences = geometries[geometry_name]
            suffix_geometry_positions = tuple(
                position
                for occurrence in geometry_occurrences
                for position in suffixes[occurrence]
            )
            geometry_positions = tuple(
                sorted(
                    set(suffix_geometry_positions)
                    | (
                        {receiver_boundary_position}
                        if bool(args.add_receiver_boundary)
                        else set()
                    )
                )
            )
            for horizon, state in enumerate(clean_states):
                expected = receiver + horizon
                decoded = decode_count_probe(read_probe, state.numpy())
                results.append(
                    {
                        "schema_version": "item_suffix_equivariance_v1",
                        "model_label": str(args.model),
                        "seed": seed,
                        "request_id": str(row["request_id"]),
                        "history_geometry": geometry_name,
                        "history_occurrences": list(geometry_occurrences),
                        "history_positions": list(geometry_positions),
                        "suffix_width": width,
                        "field_scale": field_scale,
                        "receiver_boundary_added": bool(args.add_receiver_boundary),
                        "boundary_scale": boundary_scale,
                        "receiver_boundary_position": receiver_boundary_position,
                        "condition": "aligned_count_tangent",
                        "dose": 0,
                        "horizon": horizon,
                        "expected_count": expected,
                        "exact": bool(decoded["probe_prediction"] == expected),
                        "receiver_occurrence": receiver,
                        "clamp_layers": list(layers),
                        "read_layer": read_layer,
                        "planned_delta_l2_norms": {},
                        "realized_delta_l2_norms": {},
                        "clamp_hook_applications": {},
                        "read_hook_applications": 1,
                        **decoded,
                    }
                )

            for dose in (-1, 1):
                aligned: dict[int, dict[int, np.ndarray]] = {}
                orthogonal: dict[int, dict[int, np.ndarray]] = {}
                planned: dict[int, dict[int, float]] = {}
                for layer in layers:
                    aligned[layer] = {}
                    orthogonal[layer] = {}
                    planned[layer] = {}
                    position_bases: dict[int, np.ndarray] = {}
                    for occurrence in geometry_occurrences:
                        for offset, position in enumerate(suffixes[occurrence]):
                            delta = (
                                float(dose)
                                * field_scale
                                * tangents[(layer, offset, occurrence)]
                            ).astype(np.float32)
                            aligned[layer][position] = delta
                            position_bases[position] = bases[(layer, offset)]
                    if bool(args.add_receiver_boundary):
                        boundary_delta = (
                            float(dose)
                            * boundary_scale
                            * boundary_tangents[layer]
                        ).astype(np.float32)
                        if receiver_boundary_position in aligned[layer]:
                            aligned[layer][receiver_boundary_position] = (
                                aligned[layer][receiver_boundary_position]
                                + boundary_delta
                            ).astype(np.float32)
                            position_bases[receiver_boundary_position] = (
                                union_subspace_basis(
                                    position_bases[receiver_boundary_position],
                                    boundary_bases[layer],
                                )
                            )
                        else:
                            aligned[layer][receiver_boundary_position] = boundary_delta
                            position_bases[receiver_boundary_position] = boundary_bases[
                                layer
                            ]
                    for position, delta in aligned[layer].items():
                        planned[layer][position] = float(np.linalg.norm(delta))
                        _replacement, orthogonal_direction = (
                            norm_matched_orthogonal_replacement(
                                live_by_layer_position[layer][position],
                                delta,
                                position_bases[position],
                                seed=(
                                    20261124
                                    + seed * 100000
                                    + (dose + 2) * 10000
                                    + position * 100
                                    + layer
                                ),
                            )
                        )
                        orthogonal[layer][position] = orthogonal_direction
                aligned_run = add_history_layer_deltas_and_capture_positions(
                    model,
                    adapter,
                    source,
                    layer_position_directions=aligned,
                    read_positions=read_positions,
                    read_layer=read_layer,
                )
                orthogonal_run = add_history_layer_deltas_and_capture_positions(
                    model,
                    adapter,
                    source,
                    layer_position_directions=orthogonal,
                    read_positions=read_positions,
                    read_layer=read_layer,
                    target_realized_norms=aligned_run[2],
                )
                for condition, run, planned_norms in (
                    ("aligned_count_tangent", aligned_run, planned),
                    ("orthogonal_norm_matched", orthogonal_run, aligned_run[2]),
                ):
                    captured, applications, realized, read_applications = run
                    for horizon, state in enumerate(captured):
                        natural = receiver + horizon
                        expected = (
                            natural + dose
                            if condition == "aligned_count_tangent"
                            else natural
                        )
                        decoded = decode_count_probe(read_probe, state.numpy())
                        results.append(
                            {
                                "schema_version": "item_suffix_equivariance_v1",
                                "model_label": str(args.model),
                                "seed": seed,
                                "request_id": str(row["request_id"]),
                                "history_geometry": geometry_name,
                                "history_occurrences": list(geometry_occurrences),
                                "history_positions": list(geometry_positions),
                                "suffix_width": width,
                                "field_scale": field_scale,
                                "receiver_boundary_added": bool(
                                    args.add_receiver_boundary
                                ),
                                "boundary_scale": boundary_scale,
                                "receiver_boundary_position": (
                                    receiver_boundary_position
                                ),
                                "condition": condition,
                                "dose": dose,
                                "horizon": horizon,
                                "expected_count": expected,
                                "exact": bool(decoded["probe_prediction"] == expected),
                                "receiver_occurrence": receiver,
                                "clamp_layers": list(layers),
                                "read_layer": read_layer,
                                "offset_subspace_ranks": {
                                    f"L{layer}_O{offset}": ranks[(layer, offset)]
                                    for layer in layers
                                    for offset in range(width)
                                },
                                "planned_delta_l2_norms": planned_norms,
                                "realized_delta_l2_norms": realized,
                                "clamp_hook_applications": applications,
                                "read_hook_applications": read_applications,
                                **decoded,
                            }
                        )
            print(
                f"[item-suffix] seed={seed} geometry={geometry_name} complete",
                flush=True,
            )

    slope_summary = summarize_history_slopes(
        results, geometries=geometry_names, horizons=horizons
    )
    grouped: dict[str, Any] = {}
    for geometry in geometry_names:
        for condition in ("aligned_count_tangent", "orthogonal_norm_matched"):
            for horizon in range(horizons + 1):
                values = [
                    row
                    for row in results
                    if row["history_geometry"] == geometry
                    and row["condition"] == condition
                    and int(row["horizon"]) == horizon
                ]
                grouped[f"{geometry}|{condition}|horizon_{horizon}"] = {
                    "n": len(values),
                    "exact": sum(bool(row["exact"]) for row in values),
                    "predictions": [int(row["probe_prediction"]) for row in values],
                    "expected": [int(row["expected_count"]) for row in values],
                }
    _atomic_jsonl(args.output, results)
    _atomic_json(
        args.summary,
        {
            "schema_version": "item_suffix_equivariance_v1",
            "discovery_seeds": list(discovery_seeds),
            "evaluation_seeds": list(evaluation_seeds),
            "receiver_occurrence": receiver,
            "doses": [-1, 0, 1],
            "horizons": horizons,
            "suffix_width": width,
            "field_scale": field_scale,
            "receiver_boundary_added": bool(args.add_receiver_boundary),
            "boundary_scale": boundary_scale,
            "boundary_states": (
                str(args.boundary_states)
                if args.boundary_states is not None
                else None
            ),
            "boundary_subspace_ranks": boundary_ranks,
            "history_geometries": {
                name: list(geometries[name]) for name in geometry_names
            },
            "clamp_layers": list(layers),
            "read_layer": read_layer,
            "field_cache": str(args.field_cache),
            "conditions": grouped,
            "slopes": slope_summary,
            "input_tokens_changed": False,
            "diagnostic_suffix_used": False,
            "teacher_forced_full_native_trace": True,
            "intervention_mode": (
                "discovery-frozen layer-by-offset-by-occurrence item-suffix count field"
            ),
            "control_norm_matching": (
                "per-layer per-token runtime matching after model-dtype quantization"
            ),
            "recurrent_counter_reference_slope": 1.0,
        },
    )
    print(f"[item-suffix] wrote {len(results)} rows", flush=True)


if __name__ == "__main__":
    main()
