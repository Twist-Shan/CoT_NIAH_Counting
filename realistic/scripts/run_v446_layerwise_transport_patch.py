from __future__ import annotations

"""Run answer-query transport-aligned causal patches across layer boundaries."""

import argparse
import csv
import gc
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from realistic_niah_v4.modeling import (  # noqa: E402
    _bounded_logits_kwargs,
    _encoding_tensors,
    _last_logits,
    _tensor_from_output,
    load_registered_model,
)
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt  # noqa: E402
from realistic_niah_v4.spec import V4Config, resolve_model_spec  # noqa: E402
from realistic_niah_v4.stimuli import load_stimuli  # noqa: E402
from run_v445_transport_aligned_patch import (  # noqa: E402
    aligned_geometry,
    donor_fraction,
    forward_with_patch_and_target,
    one_token_id,
)


CONDITIONS = ("aligned_dose_1", "aligned_dose_2", "matched_orthogonal")
FIELDS = (
    "model_label",
    "seed",
    "receiver_count",
    "donor_count",
    "support",
    "condition",
    "source_layer",
    "target_layer",
    "normalized_depth",
    "replacement_delta_norm",
    "planned_replacement_delta_norm",
    "aligned_dose_1_norm",
    "realized_norm_ratio_to_aligned_dose_1",
    "clean_donor_log_odds",
    "condition_donor_log_odds",
    "donor_log_odds_gain",
    "target_donor_fraction",
    "argmax_token_changed",
    "geometry_discovery_centroid_r2",
    "runtime_seconds",
)


@torch.inference_mode()
def forward_capture_query_layers(
    model: Any,
    adapter: Any,
    encoding: PromptEncoding,
    layers: list[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:

        def hook(_module, _args, output, *, layer: int = layer):
            hidden = _tensor_from_output(output)
            captured[layer] = (
                hidden[0, int(encoding.query_position)].detach().float().cpu()
            )

        handles.append(adapter.layers[layer].register_forward_hook(hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError(f"missing clean layer captures: {set(layers) - set(captured)}")
    return _last_logits(output).detach().float().cpu(), captured


def parse_boundaries(values: list[str]) -> list[tuple[int, int]]:
    boundaries = [tuple(map(int, value.split(":"))) for value in values]
    if any(len(value) != 2 or value[1] != value[0] + 1 for value in boundaries):
        raise ValueError("transport boundaries must be adjacent source:target pairs")
    return [(int(source), int(target)) for source, target in boundaries]


def quantized_additive_replacement(
    base: torch.Tensor, delta: torch.Tensor, *, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the model-dtype replacement and its realized fp32 delta."""

    base_quantized = base.to(dtype=dtype)
    replacement = (base_quantized.float() + delta.float()).to(dtype=dtype)
    realized = replacement.float() - base_quantized.float()
    norm = float(torch.linalg.vector_norm(realized).detach().cpu())
    return replacement, realized, norm


def closest_quantized_direction(
    base: torch.Tensor,
    direction: torch.Tensor,
    target_norm: float,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Search one direction for the closest realizable model-dtype norm."""

    unit = direction.float() / torch.clamp(
        torch.linalg.vector_norm(direction.float()), min=1e-12
    )

    def candidate(scale: float) -> tuple[torch.Tensor, torch.Tensor, float]:
        return quantized_additive_replacement(base, unit * float(scale), dtype=dtype)

    low = 0.0
    high = max(float(target_norm), 1e-12)
    candidates = [candidate(low), candidate(high)]
    for _ in range(20):
        if candidates[-1][2] >= target_norm:
            break
        high *= 2.0
        candidates.append(candidate(high))
    else:
        raise RuntimeError("could not bracket quantized transport norm")
    for _ in range(40):
        midpoint = (low + high) / 2.0
        current = candidate(midpoint)
        candidates.append(current)
        if current[2] < target_norm:
            low = midpoint
        else:
            high = midpoint
    return min(candidates, key=lambda value: abs(value[2] - target_norm))


def quantized_aligned_doses(
    base: torch.Tensor,
    direction: torch.Tensor,
    planned_norm: float,
    *,
    dtype: torch.dtype,
    realized_norm_tolerance: float,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, float],
    tuple[torch.Tensor, torch.Tensor, float],
    bool,
]:
    """Realize dose 1 and prefer a planned-norm dose 2 when ratio-matched."""

    dose_1 = closest_quantized_direction(
        base, direction, planned_norm, dtype=dtype
    )
    dose_2 = closest_quantized_direction(
        base, direction, 2.0 * planned_norm, dtype=dtype
    )
    realized_ratio = dose_2[2] / max(dose_1[2], 1e-12)
    used_paired_target = not np.isclose(
        realized_ratio,
        2.0,
        rtol=realized_norm_tolerance,
        atol=1e-6,
    )
    if used_paired_target:
        dose_2 = quantized_additive_replacement(
            base, 2.0 * dose_1[1], dtype=dtype
        )
        realized_ratio = dose_2[2] / max(dose_1[2], 1e-12)
        if not np.isclose(
            realized_ratio,
            2.0,
            rtol=realized_norm_tolerance,
            atol=1e-6,
        ):
            dose_2 = closest_quantized_direction(
                base, dose_1[1], 2.0 * dose_1[2], dtype=dtype
            )
    return dose_1, dose_2, used_paired_target


def existing_keys(path: Path) -> set[tuple[str, int, int, int, int, int, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int, int, int, int, int, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["model_label"],
                int(row["seed"]),
                int(row["receiver_count"]),
                int(row["donor_count"]),
                int(row["source_layer"]),
                int(row["target_layer"]),
                row["condition"],
            )
            if key in keys:
                raise RuntimeError(f"duplicate existing result key: {key}")
            keys.add(key)
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--design-config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--layer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--boundaries", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    design = json.loads(args.design_config.read_text(encoding="utf-8"))
    transport = design["answer_transport"]
    realized_norm_tolerance = float(
        transport["realized_norm_relative_tolerance"]
    )
    rank = int(design["rank"])
    seeds = args.seeds or [int(value) for value in design["confirmation_seeds"]]
    pairs = (
        [tuple(map(int, value.split(":"))) for value in args.pairs]
        if args.pairs
        else [tuple(map(int, value)) for value in transport["pairs"]]
    )
    boundaries = {
        model: (
            parse_boundaries(args.boundaries)
            if args.boundaries is not None
            else [tuple(map(int, value)) for value in transport["boundaries"][model]]
        )
        for model in args.models
    }
    config = V4Config.from_json(args.v4_config)
    stimuli = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(args.stimuli)
        if str(row.get("design_variant")) == "v4.4"
    }
    missing_stimuli = [
        (seed, receiver)
        for seed in seeds
        for receiver, _ in pairs
        if (seed, receiver) not in stimuli
    ]
    if missing_stimuli:
        raise RuntimeError(f"missing registered stimuli: {missing_stimuli[:5]}")

    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "layerwise_transport_patch.csv"
    seen = existing_keys(detail_path)
    expected = {
        (model, seed, receiver, donor, source, target, condition)
        for model in args.models
        for seed in seeds
        for receiver, donor in pairs
        for source, target in boundaries[model]
        for condition in CONDITIONS
    }
    geometry_audit: dict[str, dict[str, Any]] = {}
    with detail_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if detail_path.stat().st_size == 0:
            writer.writeheader()
        for model_label in args.models:
            geometries = {
                boundary: aligned_geometry(
                    args.layer_root,
                    model_label,
                    "answer_query",
                    boundary[0],
                    boundary[1],
                    rank=rank,
                )
                for boundary in boundaries[model_label]
            }
            geometry_audit[model_label] = {
                f"L{source}->L{target}": {
                    "paired_discovery_rows": value["paired_discovery_rows"],
                    "ridge": value["ridge"],
                    "discovery_centroid_r2": value["discovery_centroid_r2"],
                }
                for (source, target), value in geometries.items()
            }
            spec = resolve_model_spec(model_label)
            model, tokenizer, adapter = load_registered_model(
                spec,
                cache_dir=args.cache_dir,
                device_map=args.device_map,
                torch_dtype=config.model_torch_dtype,
                attention_backend=config.attention_prefix_backend,
            )
            capture_layers = sorted(
                {layer for boundary in boundaries[model_label] for layer in boundary}
            )
            encoding_cache: dict[tuple[int, int], PromptEncoding] = {}
            for seed in seeds:
                for receiver_count, donor_count in pairs:
                    pending = [
                        (source, target, condition)
                        for source, target in boundaries[model_label]
                        for condition in CONDITIONS
                        if (
                            model_label,
                            seed,
                            receiver_count,
                            donor_count,
                            source,
                            target,
                            condition,
                        )
                        not in seen
                    ]
                    if not pending:
                        continue
                    key = (seed, receiver_count)
                    if key not in encoding_cache:
                        encoding_cache[key] = render_v4_prompt(
                            stimuli[key],
                            tokenizer=tokenizer,
                            model_spec=spec,
                            config=config,
                            answer_format="numeric",
                        )
                    encoding = encoding_cache[key]
                    clean_logits, clean_states = forward_capture_query_layers(
                        model, adapter, encoding, capture_layers
                    )
                    receiver_token = one_token_id(tokenizer, receiver_count)
                    donor_token = one_token_id(tokenizer, donor_count)
                    clean_log_odds = float(
                        clean_logits[donor_token] - clean_logits[receiver_token]
                    )
                    for source_layer, target_layer, condition in pending:
                        geometry = geometries[(source_layer, target_layer)]
                        receiver_state = clean_states[source_layer]
                        source_delta = (
                            geometry["source_centroids"][donor_count - 1]
                            - geometry["source_centroids"][receiver_count - 1]
                        )
                        basis = geometry["source_basis"]
                        aligned_delta = (source_delta @ basis) @ basis.T
                        main_norm = float(torch.linalg.vector_norm(aligned_delta))
                        residual_dtype = next(
                            parameter.dtype
                            for parameter in adapter.layers[source_layer].parameters()
                            if parameter.is_floating_point()
                        )
                        dose_1, dose_2, _ = quantized_aligned_doses(
                            receiver_state,
                            aligned_delta,
                            main_norm,
                            dtype=residual_dtype,
                            realized_norm_tolerance=realized_norm_tolerance,
                        )
                        aligned_1_replacement, _, aligned_1_realized_norm = dose_1
                        aligned_2_replacement, _, aligned_2_realized_norm = dose_2
                        control_replacement, _, control_realized_norm = (
                            closest_quantized_direction(
                                receiver_state,
                                geometry["control_axis"],
                                aligned_1_realized_norm,
                                dtype=residual_dtype,
                            )
                        )
                        if condition == "aligned_dose_1":
                            replacement = aligned_1_replacement
                            replacement_norm = aligned_1_realized_norm
                            planned_norm = main_norm
                        elif condition == "aligned_dose_2":
                            replacement = aligned_2_replacement
                            replacement_norm = aligned_2_realized_norm
                            planned_norm = 2.0 * main_norm
                        else:
                            replacement = control_replacement
                            replacement_norm = control_realized_norm
                            planned_norm = main_norm
                        condition_started = time.perf_counter()
                        logits, target_state = forward_with_patch_and_target(
                            model,
                            adapter,
                            encoding,
                            source_layer=source_layer,
                            source_positions=[encoding.query_position],
                            replacement=replacement,
                            target_layer=target_layer,
                            target_position=encoding.query_position,
                        )
                        runtime = time.perf_counter() - condition_started
                        log_odds = float(logits[donor_token] - logits[receiver_token])
                        fraction = donor_fraction(
                            target_state,
                            clean_states[target_layer],
                            geometry["target_centroids"][receiver_count - 1],
                            geometry["target_centroids"][donor_count - 1],
                        )
                        expected_ratio = (
                            2.0 if condition == "aligned_dose_2" else 1.0
                        )
                        realized_ratio = replacement_norm / max(
                            aligned_1_realized_norm, 1e-12
                        )
                        planned_ratio = replacement_norm / max(planned_norm, 1e-12)
                        ratio_ok = np.isclose(
                            realized_ratio,
                            expected_ratio,
                            rtol=realized_norm_tolerance,
                            atol=1e-6,
                        )
                        if not ratio_ok:
                            raise RuntimeError(
                                "transport replacement norm audit failed: "
                                f"model={model_label} seed={seed} "
                                f"pair={receiver_count}->{donor_count} "
                                f"boundary=L{source_layer}->L{target_layer} "
                                f"condition={condition} "
                                f"planned_ratio={planned_ratio:.6f} "
                                f"dose1_ratio={realized_ratio:.6f}"
                            )
                        row = {
                            "model_label": model_label,
                            "seed": seed,
                            "receiver_count": receiver_count,
                            "donor_count": donor_count,
                            "support": "answer_query_relay",
                            "condition": condition,
                            "source_layer": source_layer,
                            "target_layer": target_layer,
                            "normalized_depth": float(
                                target_layer / max(adapter.num_layers - 1, 1)
                            ),
                            "replacement_delta_norm": replacement_norm,
                            "planned_replacement_delta_norm": planned_norm,
                            "aligned_dose_1_norm": aligned_1_realized_norm,
                            "realized_norm_ratio_to_aligned_dose_1": realized_ratio,
                            "clean_donor_log_odds": clean_log_odds,
                            "condition_donor_log_odds": log_odds,
                            "donor_log_odds_gain": log_odds - clean_log_odds,
                            "target_donor_fraction": fraction,
                            "argmax_token_changed": int(
                                torch.argmax(logits).item()
                                != torch.argmax(clean_logits).item()
                            ),
                            "geometry_discovery_centroid_r2": geometry[
                                "discovery_centroid_r2"
                            ],
                            "runtime_seconds": runtime,
                        }
                        writer.writerow(row)
                        handle.flush()
                        result_key = (
                            model_label,
                            seed,
                            receiver_count,
                            donor_count,
                            source_layer,
                            target_layer,
                            condition,
                        )
                        seen.add(result_key)
                        print(
                            f"[layer-transport] {model_label} seed={seed} "
                            f"{receiver_count}->{donor_count} L{source_layer}->L{target_layer} "
                            f"{condition} logodds={row['donor_log_odds_gain']:+.4f} "
                            f"target={fraction:+.4f}",
                            flush=True,
                        )
            del model, tokenizer, adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(f"incomplete transport grid: {missing[:5]}")
    elapsed = time.perf_counter() - started
    audit = {
        "schema_version": "realistic_niah_v4_4_layerwise_transport_patch_v1",
        "status": "PASS",
        "command": sys.argv,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": args.models,
        "seeds": seeds,
        "pairs": pairs,
        "boundaries": boundaries,
        "rank": rank,
        "conditions": list(CONDITIONS),
        "realized_norm_relative_tolerance": realized_norm_tolerance,
        "quantization_protocol": "direction-preserving scalar search to the closest realizable model-dtype norm",
        "dose2_quantization_policy": "prefer the continuous planned norm; when required by the frozen paired-dose tolerance, double the realized dose-1 vector and then search along that realized direction if needed",
        "planned_norm_policy": "diagnostic only; causal matching is audited on realized dose-2/dose-1 and control/dose-1 ratios",
        "basis_fit": "discovery count centroids; ridge prediction of adjacent downstream answer-query rank-3 count coordinates",
        "source_support": "answer query at the source post-block residual",
        "geometry": geometry_audit,
        "expected_cells": len(expected),
        "completed_cells": len(expected),
        "elapsed_seconds": elapsed,
    }
    (args.output / "transport_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
