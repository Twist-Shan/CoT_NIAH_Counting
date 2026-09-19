#!/usr/bin/env python3
"""Capture natural boundary states and fit frozen current-count probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
from realistic_niah_v5.boundary_counter_probe import (  # noqa: E402
    count_prediction_metrics,
    count_probe_predictions,
    fit_dual_ridge_count_probe,
    leave_one_seed_out_probe_metrics,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _model,
)


def _read_rows(path: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    wanted = set(seeds)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {int(row["seed"]): row for row in rows if int(row["seed"]) in wanted}
    if set(selected) != wanted:
        raise ValueError(f"One or more requested seeds are absent from {path}")
    return [selected[seed] for seed in seeds]


def _final_norm_and_unembedding(model: Any) -> tuple[Any, torch.Tensor]:
    base = getattr(model, "model", None)
    norm = getattr(base, "norm", None)
    head = getattr(model, "lm_head", None)
    if norm is None or head is None or not hasattr(head, "weight"):
        raise TypeError("Cannot locate final norm and LM head for raw logit lens")
    return norm, head.weight


@torch.inference_mode()
def _raw_digit_lens_predictions(
    model: Any,
    tokenizer: Any,
    states: torch.Tensor,
) -> list[int]:
    """Predict first digits 1..9; count 10 is excluded from this diagnostic."""

    digit_ids: list[int] = []
    for value in range(1, 10):
        ids = tuple(int(token) for token in tokenizer.encode(str(value), add_special_tokens=False))
        if len(ids) != 1:
            raise ValueError(f"Digit {value} is not one token")
        digit_ids.append(ids[0])
    norm, unembedding = _final_norm_and_unembedding(model)
    device = unembedding.device
    dtype = unembedding.dtype
    active = states.to(device=device, dtype=dtype)
    normalized = norm(active)
    selected_weights = unembedding[
        torch.as_tensor(digit_ids, dtype=torch.long, device=device)
    ]
    logits = normalized @ selected_weights.T
    return (torch.argmax(logits, dim=-1) + 1).detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--discovery-generations", type=Path, required=True)
    parser.add_argument("--confirmation-generations", type=Path, required=True)
    parser.add_argument("--discovery-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--confirmation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.command = "boundary-counter-probe"

    discovery_seeds = tuple(int(value) for value in args.discovery_seeds)
    confirmation_seeds = tuple(int(value) for value in args.confirmation_seeds)
    if set(discovery_seeds) & set(confirmation_seeds):
        raise ValueError("Discovery and confirmation seeds overlap")
    discovery_rows = _read_rows(args.discovery_generations, discovery_seeds)
    confirmation_rows = _read_rows(args.confirmation_generations, confirmation_seeds)
    model, tokenizer, adapter = _model(args)
    layers = tuple(range(int(adapter.num_layers)))
    splits = (
        ("discovery", discovery_rows),
        ("confirmation", confirmation_rows),
    )
    captured_by_split: dict[str, np.ndarray] = {}
    lens_rows: list[dict[str, Any]] = []
    boundary_audits: list[dict[str, Any]] = []

    lens_layers = set(range(0, int(adapter.num_layers), 4)) | {
        int(adapter.num_layers) - 1
    }
    for split, rows in splits:
        seed_states: list[np.ndarray] = []
        for row in rows:
            seed = int(row["seed"])
            source, _blank, registry, construction_audit = build_diagnostic_bases(
                row,
                tokenizer,
                random_seed=20260830 + seed,
                construction="targeted_explicit_count_scrub",
            )
            positions: list[int] = []
            per_seed_boundary_audit: list[dict[str, Any]] = []
            for k in range(1, 11):
                position, audit = select_post_item_boundary_position(
                    source, registry, tokenizer, occurrence=k
                )
                positions.append(position)
                per_seed_boundary_audit.append(audit)
            captures = capture_decoder_block_input_states(
                model, adapter, source, positions, layers=layers
            )
            stacked = np.stack(
                [captures[layer].numpy() for layer in layers], axis=0
            ).astype(np.float32)
            seed_states.append(stacked)
            for layer in sorted(lens_layers):
                predictions = _raw_digit_lens_predictions(
                    model, tokenizer, captures[layer][:9]
                )
                for k, prediction in enumerate(predictions, start=1):
                    lens_rows.append(
                        {
                            "split": split,
                            "seed": seed,
                            "layer": layer,
                            "target_count": k,
                            "prediction": int(prediction),
                            "exact": bool(prediction == k),
                            "readout": "raw_final_norm_unembedding_first_digit_1_to_9",
                        }
                    )
            boundary_audits.append(
                {
                    "split": split,
                    "seed": seed,
                    "request_id": str(row["request_id"]),
                    "construction": str(construction_audit["construction"]),
                    "marker_kind": str(construction_audit["marker_kind"]),
                    "boundaries": per_seed_boundary_audit,
                }
            )
            print(f"[boundary-probe] split={split} seed={seed} captured", flush=True)
        captured_by_split[split] = np.stack(seed_states, axis=0)

    discovery = captured_by_split["discovery"]
    confirmation = captured_by_split["confirmation"]
    labels_discovery = np.tile(np.arange(1, 11, dtype=np.int64), len(discovery_seeds))
    labels_confirmation = np.tile(
        np.arange(1, 11, dtype=np.int64), len(confirmation_seeds)
    )
    seed_ids_discovery = np.repeat(
        np.asarray(discovery_seeds, dtype=np.int64), 10
    )
    layer_rows: list[dict[str, Any]] = []
    probes: dict[int, dict[str, Any]] = {}
    for layer in layers:
        x_discovery = discovery[:, layer].reshape(-1, discovery.shape[-1])
        x_confirmation = confirmation[:, layer].reshape(-1, confirmation.shape[-1])
        loso = leave_one_seed_out_probe_metrics(
            x_discovery,
            labels_discovery,
            seed_ids_discovery,
            alpha=float(args.alpha),
        )
        probe = fit_dual_ridge_count_probe(
            x_discovery, labels_discovery, alpha=float(args.alpha)
        )
        confirmation_predictions = count_probe_predictions(probe, x_confirmation)
        confirmation_metrics = count_prediction_metrics(
            labels_confirmation, confirmation_predictions
        )
        layer_rows.append(
            {
                "layer": layer,
                "discovery_loso": loso,
                "confirmation": {
                    **confirmation_metrics,
                    "predictions": confirmation_predictions.tolist(),
                    "labels": labels_confirmation.tolist(),
                },
            }
        )
        probes[layer] = probe

    ranked = sorted(
        layer_rows,
        key=lambda row: (
            -float(row["discovery_loso"]["exact_accuracy"]),
            float(row["discovery_loso"]["mae"]),
            int(row["layer"]),
        ),
    )
    frozen_layers = tuple(int(row["layer"]) for row in ranked[:3])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "boundary_states_float16.npz",
        discovery=discovery.astype(np.float16),
        confirmation=confirmation.astype(np.float16),
        discovery_seeds=np.asarray(discovery_seeds, dtype=np.int64),
        confirmation_seeds=np.asarray(confirmation_seeds, dtype=np.int64),
        layers=np.asarray(layers, dtype=np.int64),
        counts=np.arange(1, 11, dtype=np.int64),
    )
    probe_payload: dict[str, np.ndarray] = {
        "frozen_layers": np.asarray(frozen_layers, dtype=np.int64),
        "alpha": np.asarray([float(args.alpha)], dtype=np.float64),
    }
    for layer in frozen_layers:
        probe_payload[f"layer_{layer}_mean"] = np.asarray(probes[layer]["mean"])
        probe_payload[f"layer_{layer}_weights"] = np.asarray(
            probes[layer]["weights"]
        )
    np.savez(args.output_dir / "frozen_probes.npz", **probe_payload)
    _atomic_json(
        args.output_dir / "probe_summary.json",
        {
            "schema_version": "boundary_counter_probe_v1",
            "model_label": str(args.model),
            "discovery_seeds": list(discovery_seeds),
            "confirmation_seeds": list(confirmation_seeds),
            "construction": "targeted_explicit_count_scrub",
            "site": "first_count_neutral_post_item_separator_else_item_endpoint",
            "layers": list(layers),
            "alpha": float(args.alpha),
            "selection_rule": "top_three_discovery_LOSO_exact_then_MAE_then_layer",
            "frozen_layers": list(frozen_layers),
            "layer_results": layer_rows,
            "raw_logit_lens": lens_rows,
            "boundary_audits": boundary_audits,
            "position_ordinal_confound": True,
            "probe_alone_establishes_counter": False,
        },
    )
    print(
        f"[boundary-probe] frozen_layers={list(frozen_layers)} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
