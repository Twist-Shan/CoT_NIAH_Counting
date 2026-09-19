from __future__ import annotations

"""Remove a discovery-frozen rank-3 count subspace from all active endpoints."""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch

FALLBACK_SRC = Path("runs/v4_4_counter_channel_20260806/code/src")
if FALLBACK_SRC.is_dir() and str(FALLBACK_SRC) not in sys.path:
    sys.path.insert(0, str(FALLBACK_SRC))

from realistic_niah_v4.modeling import (
    _is_prompt_prefill,
    _tensor_from_output,
    generate_answer_completion,
    generate_with_residual_transforms,
    load_registered_model,
)
from realistic_niah_v4.behavior import parse_numeric_completion
from realistic_niah_v4.prompts import render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli


PRIMARY_LAYER = {"Qwen3-8B": 8, "Gemma4-E4B": 9}
ANSWER_CAPTURE_LAYERS = {"Qwen3-8B": (29, 35), "Gemma4-E4B": (37, 41)}


def parsed_prediction(result: dict[str, Any]) -> int | None:
    """Apply the registered strict numeric parser to a generated continuation."""

    labels = parse_numeric_completion(str(result.get("completion_text", "")))
    value = labels["parsed_count"]
    return int(value) if value is not None else None


@torch.inference_mode()
def call_with_answer_capture(
    adapter: Any,
    encoding: Any,
    layers: tuple[int, ...],
    call: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """Run one generation and capture post-block answer-query residuals."""

    captured: dict[int, np.ndarray] = {}
    handles = []
    for layer in layers:
        def hook(_module, _args, output, *, layer=layer):
            hidden = _tensor_from_output(output)
            if _is_prompt_prefill(hidden, encoding):
                captured[layer] = (
                    hidden[0, int(encoding.query_position)]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

        handles.append(adapter.layers[layer].register_forward_hook(hook))
    try:
        result = call()
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError(f"Missing answer-query captures: {set(layers) - set(captured)}")
    return result, captured


def geometry(packed: Path, model: str, layer: int, rank: int = 3) -> dict[str, torch.Tensor]:
    with np.load(packed / "layers" / f"{model}__prompt_running__L{layer:02d}.npz", allow_pickle=False) as z:
        x = np.asarray(z["states"], dtype=np.float64)
        count = np.asarray(z["count"], dtype=int)
        split = np.asarray(z["split"]).astype(str)
    x, count = x[split == "discovery"], count[split == "discovery"]
    centroids = np.stack([x[count == value].mean(axis=0) for value in range(1, 11)])
    curve = centroids - centroids.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(curve, full_matrices=False)
    basis = vt[:rank].T
    residual = x - np.stack([centroids[value - 1] for value in count])
    residual -= (residual @ basis) @ basis.T
    _, _, residual_vt = np.linalg.svd(residual, full_matrices=False)
    control = residual_vt[:rank].T
    control -= basis @ (basis.T @ control)
    control, _ = np.linalg.qr(control)
    return {
        "basis": torch.from_numpy(basis.astype(np.float32)),
        "control": torch.from_numpy(control[:, :rank].astype(np.float32)),
        "centroids": torch.from_numpy(centroids.astype(np.float32)),
    }


def transforms(geo: dict[str, torch.Tensor], count: int) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    basis = geo["basis"]
    control = geo["control"]
    centroids = geo["centroids"][:count]

    def actual_remove(selected: torch.Tensor) -> torch.Tensor:
        b = basis.to(selected)
        centered = selected - selected.mean(dim=1, keepdim=True)
        projection = (centered @ b) @ b.T
        return selected - projection

    def centroid_remove(selected: torch.Tensor) -> torch.Tensor:
        b = basis.to(selected)
        c = centroids.to(selected)
        c = c - c.mean(dim=0, keepdim=True)
        delta = (c @ b) @ b.T
        return selected - delta.unsqueeze(0)

    def normmatched_control(selected: torch.Tensor) -> torch.Tensor:
        b = basis.to(selected)
        c = control.to(selected)
        centered = selected - selected.mean(dim=1, keepdim=True)
        target = (centered @ b) @ b.T
        nuisance = (centered @ c) @ c.T
        target_norm = torch.linalg.vector_norm(target)
        nuisance_norm = torch.clamp(torch.linalg.vector_norm(nuisance), min=1e-12)
        return selected - nuisance * (target_norm / nuisance_norm)

    return {
        "actual_rank3_remove": actual_remove,
        "centroid_curve_remove": centroid_remove,
        "normmatched_orthogonal": normmatched_control,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1254, 1264)))
    parser.add_argument("--counts", nargs="+", type=int, default=list(range(2, 11)))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    config = V4Config.from_json(args.v4_config)
    stimuli = {(int(row["seed"]), int(row["gold_count"])): row for row in load_stimuli(args.stimuli) if str(row.get("design_variant")) == "v4.4"}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "prompt_subspace_ablation_detail.csv"
    fields = ["model_label", "seed", "gold_count", "layer", "condition", "prediction", "correct", "absolute_error", "signed_error", "clean_prediction", "clean_correct", "clean_absolute_error", "completion", "answer_state_paths"]
    existing = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = {(row["model_label"], int(row["seed"]), int(row["gold_count"]), row["condition"]) for row in csv.DictReader(handle)}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        for model_label in args.models:
            layer = PRIMARY_LAYER[model_label]
            answer_layers = ANSWER_CAPTURE_LAYERS[model_label]
            geo = geometry(args.packed_root, model_label, layer)
            spec = resolve_model_spec(model_label)
            model, tokenizer, adapter = load_registered_model(spec, cache_dir=args.cache_dir, device_map=args.device_map, torch_dtype=config.model_torch_dtype, attention_backend=config.attention_prefix_backend)
            for seed in args.seeds:
                for count in args.counts:
                    encoding = render_v4_prompt(stimuli[(seed, count)], tokenizer=tokenizer, model_spec=spec, config=config, answer_format="numeric")
                    clean, clean_states = call_with_answer_capture(
                        adapter,
                        encoding,
                        answer_layers,
                        lambda: generate_answer_completion(
                            model, tokenizer, encoding, max_new_tokens=8
                        ),
                    )
                    clean_prediction = parsed_prediction(clean)
                    clean_error = abs(int(clean_prediction) - count) if clean_prediction is not None else 10
                    conditions: dict[str, tuple[dict[str, Any], dict[int, np.ndarray]]] = {
                        "clean": (clean, clean_states)
                    }
                    positions = [int(span.end) - 1 for span in encoding.needle_spans]
                    for name, transform in transforms(geo, count).items():
                        conditions[name] = call_with_answer_capture(
                            adapter,
                            encoding,
                            answer_layers,
                            lambda transform=transform: generate_with_residual_transforms(
                                model,
                                tokenizer,
                                adapter,
                                encoding,
                                {layer: (positions, transform)},
                                max_new_tokens=8,
                            ),
                        )
                    for condition, (result, answer_states) in conditions.items():
                        key = (model_label, seed, count, condition)
                        if key in existing:
                            continue
                        parsed = parsed_prediction(result)
                        state_paths = {}
                        for answer_layer, state in answer_states.items():
                            relative = (
                                Path("states")
                                / model_label
                                / f"seed_{seed}_count_{count}_{condition}_L{answer_layer}.npy"
                            )
                            state_path = args.output / relative
                            state_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(state_path, state.astype(np.float16))
                            state_paths[str(answer_layer)] = relative.as_posix()
                        writer.writerow({
                            "model_label": model_label, "seed": seed, "gold_count": count, "layer": layer,
                            "condition": condition, "prediction": "" if parsed is None else parsed,
                            "correct": int(parsed == count) if parsed is not None else 0,
                            "absolute_error": abs(parsed - count) if parsed is not None else 10,
                            "signed_error": parsed - count if parsed is not None else "",
                            "clean_prediction": "" if clean_prediction is None else int(clean_prediction),
                            "clean_correct": int(clean_prediction == count) if clean_prediction is not None else 0,
                            "clean_absolute_error": clean_error, "completion": result.get("completion_text", ""),
                            "answer_state_paths": json.dumps(state_paths, sort_keys=True),
                        })
                        handle.flush()
                        print(f"[subspace-ablation] {model_label} seed={seed} N={count} {condition} pred={parsed}", flush=True)
            del model, tokenizer, adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    (args.output / "audit.json").write_text(json.dumps({
        "schema_version": "realistic_niah_v4_4_prompt_subspace_ablation_v1",
        "models": args.models, "seeds": args.seeds, "counts": args.counts,
        "layers": PRIMARY_LAYER, "rank": 3,
        "actual_removal": "remove the actual within-prompt, across-endpoint component in the frozen count-centroid subspace",
        "centroid_removal": "subtract the discovery count-centroid curve after centering over active occurrences",
        "control": "remove an equal-Frobenius-norm component in an orthogonal within-count residual PCA subspace",
        "answer_capture_layers": ANSWER_CAPTURE_LAYERS,
        "status": "PASS",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
