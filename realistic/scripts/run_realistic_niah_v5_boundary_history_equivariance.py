#!/usr/bin/env python3
"""Test whether a count offset is distributed over several native boundaries.

Every input token and the full teacher-forced native trace remain unchanged.  The
intervention coherently translates the count representation at the last 1--4
post-item boundaries before item 6.  Boundary i receives its own discovery-only
local count tangent, rather than broadcasting the item-5 tangent to unlike sites.
If a recurrent count offset is distributed over this boundary-memory bank, the
offset measured at item 5 should survive the native item-6 transition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import (  # noqa: E402
    _bounded_logits_kwargs,
    _encoding_tensors,
)
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


def boundary_history_geometries(receiver: int) -> dict[str, tuple[int, ...]]:
    """Return the predeclared local-to-distributed boundary-memory sweep."""

    receiver = int(receiver)
    if receiver < 5:
        raise ValueError("The four-boundary history sweep requires receiver >= 5")
    return {
        "current_boundary": (receiver,),
        "current_plus_boundary_2": (2, receiver),
        "current_plus_boundary_3": (3, receiver),
        "last_2_boundaries": tuple(range(receiver - 1, receiver + 1)),
        "last_3_boundaries": tuple(range(receiver - 2, receiver + 1)),
        "last_4_boundaries": tuple(range(receiver - 3, receiver + 1)),
    }


def _validate_layer_position_panel(
    panel: Mapping[int, Mapping[int, Any]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    layers = tuple(sorted(int(layer) for layer in panel))
    if not layers or layers != tuple(range(layers[0], layers[-1] + 1)):
        raise ValueError("Patch layers must be a nonempty contiguous band")
    position_sets = {
        tuple(sorted(int(position) for position in panel[layer])) for layer in layers
    }
    if len(position_sets) != 1:
        raise ValueError("Every patch layer must use the same token positions")
    positions = next(iter(position_sets))
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Patch positions must be unique and nonempty")
    return layers, positions


@torch.inference_mode()
def add_history_layer_deltas_and_capture_positions(
    model: Any,
    adapter: Any,
    encoding: Any,
    *,
    layer_position_directions: Mapping[int, Mapping[int, np.ndarray]],
    read_positions: Sequence[int],
    read_layer: int,
    target_realized_norms: Mapping[int, Mapping[int, float]] | None = None,
) -> tuple[torch.Tensor, dict[int, int], dict[int, dict[int, float]], int]:
    """Add position-specific deltas to live states and capture later read sites.

    Because the edited positions precede the transition, their altered states also
    alter the K/V memory exposed to later native tokens.  Controls can request exact
    per-layer, per-position realized norms after model-dtype quantization.
    """

    patch_layers, patch_positions = _validate_layer_position_panel(
        layer_position_directions
    )
    directions = {
        layer: {
            int(position): torch.as_tensor(direction).detach().float().cpu().reshape(-1)
            for position, direction in layer_position_directions[layer].items()
        }
        for layer in patch_layers
    }
    reads = tuple(int(value) for value in read_positions)
    read_layer = int(read_layer)
    if not reads or len(set(reads)) != len(reads):
        raise ValueError("Read positions must be unique and nonempty")
    if not 0 <= patch_layers[0] <= patch_layers[-1] < read_layer < int(adapter.num_layers):
        raise ValueError("Patch band must end before the read layer")
    if min(patch_positions) < 0 or max(patch_positions + reads) >= int(
        encoding.sequence_length
    ):
        raise ValueError("A patch/read position is outside the encoding")
    if max(patch_positions) > max(reads):
        raise ValueError("Patch history must not extend beyond every requested future")

    targets = None
    if target_realized_norms is not None:
        targets = {
            int(layer): {
                int(position): float(value) for position, value in values.items()
            }
            for layer, values in target_realized_norms.items()
        }
        target_layers, target_positions = _validate_layer_position_panel(targets)
        if target_layers != patch_layers or target_positions != patch_positions:
            raise ValueError("Control targets changed the patch geometry")

    applications = {layer: 0 for layer in patch_layers}
    realized = {
        layer: {position: 0.0 for position in patch_positions}
        for layer in patch_layers
    }
    captured: torch.Tensor | None = None
    read_applications = 0
    handles = []

    def closest_live_replacement(
        before: torch.Tensor,
        direction: torch.Tensor,
        target: float,
    ) -> tuple[torch.Tensor, float]:
        active = direction.to(device=before.device, dtype=torch.float32)
        active_norm = float(torch.linalg.vector_norm(active).detach().cpu())
        if active_norm <= 1e-12 or float(target) <= 1e-12:
            return before, 0.0
        unit = active / active_norm

        def candidate(scale: float) -> tuple[torch.Tensor, float]:
            replacement = (before.float() + float(scale) * unit).to(dtype=before.dtype)
            norm = float(
                torch.linalg.vector_norm(replacement.float() - before.float())
                .detach()
                .cpu()
            )
            return replacement, norm

        low = 0.0
        high = float(target)
        candidates = [candidate(low), candidate(high)]
        for _ in range(20):
            if candidates[-1][1] >= float(target):
                break
            high *= 2.0
            candidates.append(candidate(high))
        else:
            raise RuntimeError("Could not bracket a quantized control norm")
        for _ in range(32):
            midpoint = 0.5 * (low + high)
            current = candidate(midpoint)
            candidates.append(current)
            if current[1] < float(target):
                low = midpoint
            else:
                high = midpoint
        return min(candidates, key=lambda value: abs(value[1] - float(target)))

    for layer in patch_layers:
        def patch_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("History patch block input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
                return None
            patched = hidden.clone()
            for position in patch_positions:
                before = hidden[0, position]
                direction = directions[layer][position]
                if direction.numel() != before.numel():
                    raise RuntimeError("History direction hidden width mismatch")
                if targets is None:
                    replacement = (
                        before.float()
                        + direction.to(device=before.device, dtype=torch.float32)
                    ).to(dtype=before.dtype)
                    norm = float(
                        torch.linalg.vector_norm(replacement.float() - before.float())
                        .detach()
                        .cpu()
                    )
                else:
                    replacement, norm = closest_live_replacement(
                        before, direction, targets[layer][position]
                    )
                patched[0, position] = replacement
                realized[layer][position] = norm
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(patch_hook))

    def read_hook(_module: Any, args: tuple[Any, ...]) -> None:
        nonlocal captured, read_applications
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("History read block input is not a tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
            return
        captured = hidden[0, list(reads)].detach().float().cpu()
        read_applications += 1

    handles.append(adapter.layers[read_layer].register_forward_pre_hook(read_hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    if (
        any(value != 1 for value in applications.values())
        or read_applications != 1
        or captured is None
    ):
        raise RuntimeError(
            "Every history/read hook must apply once; "
            f"history={applications} read={read_applications}"
        )
    return captured, applications, realized, read_applications


def summarize_history_slopes(
    rows: Sequence[Mapping[str, Any]],
    *,
    geometries: Sequence[str],
    horizons: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for geometry in geometries:
        active = [
            row
            for row in rows
            if row["condition"] == "aligned_count_tangent"
            and row["history_geometry"] == geometry
        ]
        per_seed: dict[str, Any] = {}
        pooled_x = {horizon: [] for horizon in range(1, horizons + 1)}
        pooled_y = {horizon: [] for horizon in range(1, horizons + 1)}
        for seed in sorted({int(row["seed"]) for row in active}):
            seed_rows = [row for row in active if int(row["seed"]) == seed]
            by_site_dose = {
                (int(row["horizon"]), int(row["dose"])): float(
                    row["probe_softmax_expected_count"]
                )
                for row in seed_rows
            }
            doses = sorted({int(row["dose"]) for row in seed_rows})
            current_zero = by_site_dose[(0, 0)]
            current_shift = [by_site_dose[(0, dose)] - current_zero for dose in doses]
            seed_summary: dict[str, Any] = {
                "doses": doses,
                "current_shifts": current_shift,
                "dose_to_current_slope": through_origin_slope(doses, current_shift),
            }
            for horizon in range(1, horizons + 1):
                site_zero = by_site_dose[(horizon, 0)]
                site_shift = [
                    by_site_dose[(horizon, dose)] - site_zero for dose in doses
                ]
                seed_summary[f"horizon_{horizon}_shifts"] = site_shift
                seed_summary[f"current_to_horizon_{horizon}_slope"] = (
                    through_origin_slope(current_shift, site_shift)
                )
                pooled_x[horizon].extend(current_shift)
                pooled_y[horizon].extend(site_shift)
            per_seed[str(seed)] = seed_summary
        output[geometry] = {
            "per_seed": per_seed,
            "pooled_current_to_horizon_slopes": {
                str(horizon): through_origin_slope(
                    pooled_x[horizon], pooled_y[horizon]
                )
                for horizon in range(1, horizons + 1)
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
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--boundary-states", type=Path, required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--receiver-occurrence", type=int, default=5)
    parser.add_argument("--doses", type=int, nargs="+", default=(-1, 0, 1))
    parser.add_argument("--horizons", type=int, default=3)
    parser.add_argument("--clamp-start-layer", type=int, default=14)
    parser.add_argument("--read-layer", type=int, default=24)
    parser.add_argument("--geometries", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "boundary-history-equivariance"

    wanted = tuple(int(value) for value in args.seeds)
    receiver = int(args.receiver_occurrence)
    doses = tuple(sorted({int(value) for value in args.doses}))
    horizons = int(args.horizons)
    read_layer = int(args.read_layer)
    clamp_layers = tuple(range(int(args.clamp_start_layer), read_layer))
    available_geometries = boundary_history_geometries(receiver)
    geometry_names = (
        tuple(str(value) for value in args.geometries)
        if args.geometries
        else tuple(available_geometries)
    )
    if not wanted or doses != (-1, 0, 1):
        raise ValueError("The frozen history sweep requires doses -1 0 1")
    if any(name not in available_geometries for name in geometry_names):
        raise ValueError("Unknown history geometry")
    if horizons < 1 or receiver + horizons > 10:
        raise ValueError("Requested native-boundary horizons are outside 1..10")

    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {int(row["seed"]): row for row in rows if int(row["seed"]) in wanted}
    if set(selected) != set(wanted):
        raise ValueError("One or more requested seeds are absent")

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

    required_occurrences = sorted(
        {
            occurrence
            for name in geometry_names
            for occurrence in available_geometries[name]
        }
    )
    bases: dict[int, np.ndarray] = {}
    ranks: dict[int, int] = {}
    tangents_by_occurrence: dict[int, dict[int, np.ndarray]] = {}
    for occurrence in required_occurrences:
        active_bases, active_tangents, active_ranks = frozen_layer_geometry(
            args.boundary_states,
            layers=clamp_layers,
            receiver_count=occurrence,
            alpha=alpha,
        )
        if not bases:
            bases, ranks = active_bases, active_ranks
        tangents_by_occurrence[occurrence] = active_tangents

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
        occurrences_to_read = tuple(
            range(min(required_occurrences), receiver + horizons + 1)
        )
        boundary_positions = {
            occurrence: select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=occurrence
            )[0]
            for occurrence in occurrences_to_read
        }
        read_positions = tuple(boundary_positions[value] for value in occurrences_to_read)
        read_index = {occurrence: index for index, occurrence in enumerate(occurrences_to_read)}
        history_positions = tuple(
            boundary_positions[value] for value in required_occurrences
        )
        receiver_states = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            history_positions,
            layers=clamp_layers,
        )
        state_by_layer_occurrence = {
            layer: {
                occurrence: receiver_states[layer][index].numpy()
                for index, occurrence in enumerate(required_occurrences)
            }
            for layer in clamp_layers
        }
        clean_states = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            read_positions,
            layers=(read_layer,),
        )[read_layer]

        for geometry_name in geometry_names:
            geometry_occurrences = available_geometries[geometry_name]
            geometry_positions = tuple(
                boundary_positions[value] for value in geometry_occurrences
            )
            for horizon in range(horizons + 1):
                occurrence = receiver + horizon
                decoded = decode_count_probe(
                    read_probe, clean_states[read_index[occurrence]].numpy()
                )
                results.append(
                    {
                        "schema_version": "boundary_history_equivariance_v1",
                        "model_label": str(args.model),
                        "seed": seed,
                        "request_id": str(row["request_id"]),
                        "history_geometry": geometry_name,
                        "history_occurrences": list(geometry_occurrences),
                        "history_positions": list(geometry_positions),
                        "condition": "aligned_count_tangent",
                        "dose": 0,
                        "horizon": horizon,
                        "expected_count": occurrence,
                        "exact": bool(decoded["probe_prediction"] == occurrence),
                        "receiver_occurrence": receiver,
                        "clamp_layers": list(clamp_layers),
                        "read_layer": read_layer,
                        "subspace_ranks": ranks,
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
                for layer in clamp_layers:
                    aligned[layer] = {}
                    orthogonal[layer] = {}
                    planned[layer] = {}
                    for occurrence in geometry_occurrences:
                        position = boundary_positions[occurrence]
                        delta = (
                            float(dose) * tangents_by_occurrence[occurrence][layer]
                        ).astype(np.float32)
                        aligned[layer][position] = delta
                        planned[layer][position] = float(np.linalg.norm(delta))
                        _replacement, orthogonal_direction = (
                            norm_matched_orthogonal_replacement(
                                state_by_layer_occurrence[layer][occurrence],
                                delta,
                                bases[layer],
                                seed=(
                                    20261024
                                    + seed * 100000
                                    + (dose + 2) * 10000
                                    + occurrence * 100
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
                condition_runs = (
                    ("aligned_count_tangent", aligned_run, planned),
                    ("orthogonal_norm_matched", orthogonal_run, aligned_run[2]),
                )
                for condition, run, planned_norms in condition_runs:
                    captured, applications, realized, read_applications = run
                    for horizon in range(horizons + 1):
                        occurrence = receiver + horizon
                        decoded = decode_count_probe(
                            read_probe, captured[read_index[occurrence]].numpy()
                        )
                        expected = (
                            occurrence + dose
                            if condition == "aligned_count_tangent"
                            else occurrence
                        )
                        results.append(
                            {
                                "schema_version": "boundary_history_equivariance_v1",
                                "model_label": str(args.model),
                                "seed": seed,
                                "request_id": str(row["request_id"]),
                                "history_geometry": geometry_name,
                                "history_occurrences": list(geometry_occurrences),
                                "history_positions": list(geometry_positions),
                                "condition": condition,
                                "dose": dose,
                                "horizon": horizon,
                                "expected_count": expected,
                                "exact": bool(decoded["probe_prediction"] == expected),
                                "receiver_occurrence": receiver,
                                "clamp_layers": list(clamp_layers),
                                "read_layer": read_layer,
                                "subspace_ranks": ranks,
                                "planned_delta_l2_norms": planned_norms,
                                "realized_delta_l2_norms": realized,
                                "clamp_hook_applications": applications,
                                "read_hook_applications": read_applications,
                                **decoded,
                            }
                        )
            print(
                f"[boundary-history] seed={seed} geometry={geometry_name} complete",
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
            "schema_version": "boundary_history_equivariance_v1",
            "seeds": list(wanted),
            "receiver_occurrence": receiver,
            "doses": list(doses),
            "horizons": horizons,
            "history_geometries": {
                name: list(available_geometries[name]) for name in geometry_names
            },
            "clamp_layers": list(clamp_layers),
            "read_layer": read_layer,
            "conditions": grouped,
            "slopes": slope_summary,
            "input_tokens_changed": False,
            "diagnostic_suffix_used": False,
            "teacher_forced_full_native_trace": True,
            "intervention_mode": (
                "occurrence-specific discovery count tangents added to multiple "
                "live boundary states and their downstream K/V memory"
            ),
            "control_norm_matching": (
                "per-layer per-position runtime matching after model-dtype quantization"
            ),
            "recurrent_counter_reference_slope": 1.0,
        },
    )
    print(f"[boundary-history] wrote {len(results)} rows", flush=True)


if __name__ == "__main__":
    main()
