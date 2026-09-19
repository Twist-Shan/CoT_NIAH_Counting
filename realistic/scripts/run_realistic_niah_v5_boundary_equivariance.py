#!/usr/bin/env python3
"""Causal multi-dose test of count-state retention across native bullet transitions.

The input trace is unchanged.  At one native post-item boundary, the script
adds discovery-frozen local count tangents to a contiguous decoder-layer band,
then reads the same boundary and several later native boundaries with the
already frozen confirmation probe.  A recurrent counter predicts that the
probe displacement created at the intervention boundary is retained after
each subsequent counted item.
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

from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v4.modeling import (  # noqa: E402
    _bounded_logits_kwargs,
    _encoding_tensors,
)
from realistic_niah_v5.boundary_counter_probe import (  # noqa: E402
    count_probe_predictions,
    count_probe_scores,
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
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)


def decode_count_probe(probe: Mapping[str, Any], state: np.ndarray) -> dict[str, Any]:
    values = np.asarray(state, dtype=np.float32).reshape(1, -1)
    scores = count_probe_scores(dict(probe), values)[0]
    probabilities = np.exp(scores - float(np.max(scores)))
    probabilities = probabilities / float(probabilities.sum())
    return {
        "probe_prediction": int(count_probe_predictions(dict(probe), values)[0]),
        "probe_scores": [float(value) for value in scores],
        "probe_softmax_expected_count": float(
            probabilities @ np.arange(1, 11, dtype=np.float64)
        ),
    }


def local_count_tangent(
    states: np.ndarray,
    *,
    receiver_count: int,
    basis: np.ndarray,
) -> np.ndarray:
    """Discovery-centroid central difference, projected into the count span."""

    panel = np.asarray(states, dtype=np.float64)
    active_basis = np.asarray(basis, dtype=np.float64)
    receiver = int(receiver_count)
    if panel.ndim != 3 or int(panel.shape[1]) != 10:
        raise ValueError("Expected discovery states with shape [seed,10,hidden]")
    if not 2 <= receiver <= 9:
        raise ValueError("A central local tangent requires receiver_count in 2..9")
    if active_basis.ndim != 2 or active_basis.shape[0] != panel.shape[-1]:
        raise ValueError("Count basis and discovery-state widths disagree")
    centroids = panel.mean(axis=0)
    raw = 0.5 * (centroids[receiver] - centroids[receiver - 2])
    projected = (raw @ active_basis) @ active_basis.T
    return projected.astype(np.float32)


def through_origin_slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Slope inputs must be same-length vectors")
    denominator = float(left @ left)
    if denominator <= 1e-12:
        return None
    return float((left @ right) / denominator)


def quantized_delta(
    base: np.ndarray,
    replacement: np.ndarray,
    *,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Realize a replacement in model dtype and return its actual delta/norm."""

    base_tensor = torch.as_tensor(base, dtype=torch.float32).reshape(-1)
    replacement_tensor = torch.as_tensor(replacement, dtype=torch.float32).reshape(-1)
    if base_tensor.shape != replacement_tensor.shape:
        raise ValueError("Quantized replacement width changed")
    realized = replacement_tensor.to(dtype=dtype).to(dtype=torch.float32)
    delta = realized - base_tensor
    return (
        realized.numpy().astype(np.float32),
        delta.numpy().astype(np.float32),
        float(torch.linalg.vector_norm(delta)),
    )


def quantized_norm_matched_replacement(
    base: np.ndarray,
    direction: np.ndarray,
    *,
    target_norm: float,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale a direction until its model-dtype realized norm matches a target."""

    base_array = np.asarray(base, dtype=np.float32).reshape(-1)
    active = np.asarray(direction, dtype=np.float64).reshape(-1)
    if active.shape != base_array.shape:
        raise ValueError("Norm-match direction and base widths disagree")
    norm = float(np.linalg.norm(active))
    target = float(target_norm)
    if norm <= 1e-12 or target <= 1e-12:
        zero = np.zeros_like(base_array)
        return base_array.copy(), zero, 0.0
    unit = active / norm

    def candidate(scale: float) -> tuple[np.ndarray, np.ndarray, float]:
        return quantized_delta(
            base_array,
            base_array + float(scale) * unit,
            dtype=dtype,
        )

    low = 0.0
    high = target
    candidates = [candidate(low), candidate(high)]
    for _ in range(20):
        if candidates[-1][2] >= target:
            break
        high *= 2.0
        candidates.append(candidate(high))
    else:
        raise RuntimeError("Could not bracket a quantized norm-matched control")
    for _ in range(48):
        midpoint = 0.5 * (low + high)
        current = candidate(midpoint)
        candidates.append(current)
        if current[2] < target:
            low = midpoint
        else:
            high = midpoint
    return min(candidates, key=lambda value: abs(value[2] - target))


@torch.inference_mode()
def add_boundary_layer_deltas_and_capture_positions(
    model: Any,
    adapter: Any,
    encoding: Any,
    *,
    patch_position: int,
    layer_directions: Mapping[int, np.ndarray],
    read_positions: Sequence[int],
    read_layer: int,
    target_realized_norms: Mapping[int, float] | None = None,
) -> tuple[torch.Tensor, dict[int, int], dict[int, float], int]:
    """Add deltas to each live layer state; optionally quantization-match norms."""

    patch_position = int(patch_position)
    positions = tuple(int(value) for value in read_positions)
    read_layer = int(read_layer)
    directions = {
        int(layer): torch.as_tensor(direction).detach().float().cpu().reshape(-1)
        for layer, direction in layer_directions.items()
    }
    patch_layers = tuple(sorted(directions))
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Read positions must be nonempty and unique")
    if not patch_layers or patch_layers != tuple(range(patch_layers[0], patch_layers[-1] + 1)):
        raise ValueError("Additive patch layers must be a nonempty contiguous band")
    if not 0 <= patch_layers[0] <= patch_layers[-1] < read_layer < int(adapter.num_layers):
        raise ValueError("Additive patch band must end before the read layer")
    if not 0 <= patch_position <= min(positions):
        raise ValueError("Every read site must be at or after the additive patch")
    if max(positions) >= int(encoding.sequence_length):
        raise ValueError("An additive read site is outside the encoding")
    targets = (
        None
        if target_realized_norms is None
        else {int(layer): float(value) for layer, value in target_realized_norms.items()}
    )
    if targets is not None and set(targets) != set(patch_layers):
        raise ValueError("Every additive control layer needs a target realized norm")

    applications = {layer: 0 for layer in patch_layers}
    realized_norms = {layer: 0.0 for layer in patch_layers}
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
            raise RuntimeError("Could not bracket a live quantized control norm")
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
                raise RuntimeError("Additive block input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
                return None
            before = hidden[0, patch_position]
            direction = directions[layer]
            if direction.numel() != before.numel():
                raise RuntimeError("Additive direction hidden width mismatch")
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
                    before, direction, targets[layer]
                )
            patched = hidden.clone()
            patched[0, patch_position] = replacement
            applications[layer] += 1
            realized_norms[layer] = norm
            return (patched, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(patch_hook))

    def read_hook(_module: Any, args: tuple[Any, ...]) -> None:
        nonlocal captured, read_applications
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Additive read block input is not a tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
            return
        captured = hidden[0, list(positions)].detach().float().cpu()
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
            "Every additive/read hook must apply once; "
            f"additive={applications} read={read_applications}"
        )
    return captured, applications, realized_norms, read_applications


def frozen_layer_geometry(
    state_path: Path,
    *,
    layers: Sequence[int],
    receiver_count: int,
    alpha: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    payload = np.load(state_path)
    discovery = np.asarray(payload["discovery"], dtype=np.float32)
    stored_layers = [int(value) for value in np.asarray(payload["layers"])]
    counts = np.asarray(payload["counts"], dtype=np.int64)
    if discovery.ndim != 4 or counts.tolist() != list(range(1, 11)):
        raise ValueError("Boundary-state bank geometry changed")
    labels = np.tile(counts, int(discovery.shape[0]))
    bases: dict[int, np.ndarray] = {}
    tangents: dict[int, np.ndarray] = {}
    ranks: dict[int, int] = {}
    for layer in layers:
        active_layer = int(layer)
        if active_layer not in stored_layers:
            raise ValueError(f"Boundary-state bank has no layer {active_layer}")
        layer_index = stored_layers.index(active_layer)
        panel = discovery[:, layer_index]
        probe = fit_dual_ridge_count_probe(
            panel.reshape(-1, panel.shape[-1]), labels, alpha=float(alpha)
        )
        basis = count_probe_subspace(probe)
        bases[active_layer] = basis
        tangents[active_layer] = local_count_tangent(
            panel, receiver_count=int(receiver_count), basis=basis
        )
        ranks[active_layer] = int(basis.shape[1])
    return bases, tangents, ranks


def summarize_slopes(
    rows: Sequence[Mapping[str, Any]],
    *,
    receiver_count: int,
    horizons: int,
) -> dict[str, Any]:
    aligned = [row for row in rows if row["condition"] == "aligned_count_tangent"]
    output: dict[str, Any] = {"per_seed": {}}
    pooled_x: dict[int, list[float]] = {horizon: [] for horizon in range(1, horizons + 1)}
    pooled_y: dict[int, list[float]] = {horizon: [] for horizon in range(1, horizons + 1)}
    for seed in sorted({int(row["seed"]) for row in aligned}):
        seed_rows = [row for row in aligned if int(row["seed"]) == seed]
        by_site_dose = {
            (int(row["horizon"]), int(row["dose"])): float(
                row["probe_softmax_expected_count"]
            )
            for row in seed_rows
        }
        doses = sorted({int(row["dose"]) for row in seed_rows})
        if 0 not in doses:
            raise ValueError("Aligned slope panel has no zero-dose reference")
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
            slope = through_origin_slope(current_shift, site_shift)
            seed_summary[f"horizon_{horizon}_shifts"] = site_shift
            seed_summary[f"current_to_horizon_{horizon}_slope"] = slope
            pooled_x[horizon].extend(current_shift)
            pooled_y[horizon].extend(site_shift)
        output["per_seed"][str(seed)] = seed_summary
    output["pooled_current_to_horizon_slopes"] = {
        str(horizon): through_origin_slope(pooled_x[horizon], pooled_y[horizon])
        for horizon in range(1, horizons + 1)
    }
    output["recurrent_counter_reference_slope"] = 1.0
    output["receiver_count"] = int(receiver_count)
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
    parser.add_argument("--doses", type=int, nargs="+", default=(-2, -1, 0, 1, 2))
    parser.add_argument("--horizons", type=int, default=3)
    parser.add_argument("--clamp-start-layer", type=int, default=14)
    parser.add_argument("--read-layer", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "boundary-equivariance"

    wanted = tuple(int(value) for value in args.seeds)
    receiver = int(args.receiver_occurrence)
    doses = tuple(sorted({int(value) for value in args.doses}))
    horizons = int(args.horizons)
    read_layer = int(args.read_layer)
    clamp_layers = tuple(range(int(args.clamp_start_layer), read_layer))
    if not wanted or 0 not in doses or len(doses) < 3:
        raise ValueError("Need seeds and at least three doses including zero")
    if horizons < 1 or receiver + horizons > 10:
        raise ValueError("Requested native-boundary horizons are outside 1..10")

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
        "weights": np.asarray(
            probe_npz[f"layer_{read_layer}_weights"], dtype=np.float32
        ),
        "alpha": alpha,
    }
    bases, tangents, ranks = frozen_layer_geometry(
        args.boundary_states,
        layers=clamp_layers,
        receiver_count=receiver,
        alpha=alpha,
    )
    tangent_norms = {
        layer: float(np.linalg.norm(tangents[layer])) for layer in clamp_layers
    }
    if not any(value > 1e-12 for value in tangent_norms.values()):
        raise RuntimeError("Every discovery local count tangent in the clamp band is zero")
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
            for occurrence in range(receiver, receiver + horizons + 1)
        }
        read_positions = tuple(boundaries[value] for value in sorted(boundaries))
        receiver_position = boundaries[receiver]
        receiver_states = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            (receiver_position,),
            layers=clamp_layers,
        )
        clean_states = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            read_positions,
            layers=(read_layer,),
        )[read_layer]

        for horizon, state in enumerate(clean_states):
            decoded = decode_count_probe(read_probe, state.numpy())
            expected = receiver + horizon
            results.append(
                {
                    "schema_version": "boundary_equivariance_v1",
                    "model_label": str(args.model),
                    "seed": seed,
                    "request_id": str(row["request_id"]),
                    "condition": "aligned_count_tangent",
                    "dose": 0,
                    "horizon": horizon,
                    "expected_count": expected,
                    "exact": bool(decoded["probe_prediction"] == expected),
                    "receiver_occurrence": receiver,
                    "clamp_layers": list(clamp_layers),
                    "read_layer": read_layer,
                    "subspace_ranks": ranks,
                    "planned_delta_l2_norms": {
                        layer: 0.0 for layer in clamp_layers
                    },
                    "realized_delta_l2_norms": {
                        layer: 0.0 for layer in clamp_layers
                    },
                    "clamp_hook_applications": {
                        layer: 0 for layer in clamp_layers
                    },
                    "read_hook_applications": 1,
                    **decoded,
                }
            )

        for dose in (value for value in doses if value != 0):
            aligned_deltas: dict[int, np.ndarray] = {}
            orthogonal_directions: dict[int, np.ndarray] = {}
            aligned_norms: dict[int, float] = {}
            for layer in clamp_layers:
                receiver_state = receiver_states[layer][0].numpy()
                delta = float(dose) * tangents[layer]
                aligned_deltas[layer] = delta.astype(np.float32)
                aligned_norms[layer] = float(np.linalg.norm(delta))
                _orthogonal, orthogonal_direction = norm_matched_orthogonal_replacement(
                    receiver_state,
                    delta,
                    bases[layer],
                    seed=20260924 + seed * 10000 + (dose + 10) * 100 + layer,
                )
                orthogonal_directions[layer] = orthogonal_direction

            condition_runs: list[
                tuple[str, torch.Tensor, dict[int, int], dict[int, float], int, dict[int, float]]
            ] = []
            aligned_run = add_boundary_layer_deltas_and_capture_positions(
                model,
                adapter,
                source,
                patch_position=receiver_position,
                layer_directions=aligned_deltas,
                read_positions=read_positions,
                read_layer=read_layer,
            )
            condition_runs.append(
                ("aligned_count_tangent", *aligned_run, aligned_norms)
            )
            orthogonal_run = add_boundary_layer_deltas_and_capture_positions(
                model,
                adapter,
                source,
                patch_position=receiver_position,
                layer_directions=orthogonal_directions,
                read_positions=read_positions,
                read_layer=read_layer,
                target_realized_norms=aligned_run[2],
            )
            condition_runs.append(
                ("orthogonal_norm_matched", *orthogonal_run, aligned_run[2])
            )

            for (
                condition,
                captured,
                applications,
                realized_norms,
                read_applications,
                planned_norms,
            ) in condition_runs:
                for horizon, state in enumerate(captured):
                    decoded = decode_count_probe(read_probe, state.numpy())
                    expected = (
                        receiver + horizon + dose
                        if condition == "aligned_count_tangent"
                        else receiver + horizon
                    )
                    valid_expected = 1 <= expected <= 10
                    results.append(
                        {
                            "schema_version": "boundary_equivariance_v1",
                            "model_label": str(args.model),
                            "seed": seed,
                            "request_id": str(row["request_id"]),
                            "condition": condition,
                            "dose": dose,
                            "horizon": horizon,
                            "expected_count": expected if valid_expected else None,
                            "exact": bool(
                                valid_expected
                                and decoded["probe_prediction"] == expected
                            ),
                            "receiver_occurrence": receiver,
                            "clamp_layers": list(clamp_layers),
                            "read_layer": read_layer,
                            "subspace_ranks": ranks,
                            "planned_delta_l2_norms": planned_norms,
                            "realized_delta_l2_norms": realized_norms,
                            "clamp_hook_applications": applications,
                            "read_hook_applications": read_applications,
                            **decoded,
                        }
                    )
        print(f"[boundary-equivariance] seed={seed} complete", flush=True)

    slope_summary = summarize_slopes(
        results, receiver_count=receiver, horizons=horizons
    )
    grouped: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in results}):
        for horizon in range(horizons + 1):
            values = [
                row
                for row in results
                if row["condition"] == condition and int(row["horizon"]) == horizon
            ]
            grouped[f"{condition}:horizon_{horizon}"] = {
                "n": len(values),
                "exact": sum(bool(row["exact"]) for row in values),
                "doses": [int(row["dose"]) for row in values],
                "predictions": [int(row["probe_prediction"]) for row in values],
                "expected": [row["expected_count"] for row in values],
                "mean_softmax_expected_count": float(
                    np.mean(
                        [float(row["probe_softmax_expected_count"]) for row in values]
                    )
                ),
            }

    _atomic_jsonl(args.output, results)
    _atomic_json(
        args.summary,
        {
            "schema_version": "boundary_equivariance_v1",
            "seeds": list(wanted),
            "receiver_occurrence": receiver,
            "doses": list(doses),
            "horizons": horizons,
            "clamp_layers": list(clamp_layers),
            "read_layer": read_layer,
            "conditions": grouped,
            "slopes": slope_summary,
            "input_tokens_changed": False,
            "diagnostic_suffix_used": False,
            "teacher_forced_full_native_trace": True,
            "count_tangent_fit": "discovery-only central centroid difference projected into frozen probe span",
            "count_tangent_l2_norms": tangent_norms,
            "primary_estimand": "probe displacement retained from horizon 0 to each later native item boundary",
            "intervention_mode": "relative additive delta at each live layer state",
            "control_norm_matching": "per-layer runtime matching after model-dtype quantization",
        },
    )
    print(f"[boundary-equivariance] wrote {len(results)} rows", flush=True)


if __name__ == "__main__":
    main()
