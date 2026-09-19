from __future__ import annotations

"""Run discovery-frozen rank-3 removal over registered layer landmarks."""

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

from realistic_niah_v4.behavior import parse_numeric_completion  # noqa: E402
from realistic_niah_v4.layerwise_removal import (  # noqa: E402
    PromptRemovalGeometry,
    fit_prompt_removal_geometry,
    make_answer_query_removal_transform,
    make_prompt_removal_transform,
)
from realistic_niah_v4.modeling import (  # noqa: E402
    generate_answer_completion,
    generate_with_residual_transforms,
    load_registered_model,
)
from realistic_niah_v4.prompts import render_v4_prompt  # noqa: E402
from realistic_niah_v4.spec import V4Config, resolve_model_spec  # noqa: E402
from realistic_niah_v4.stimuli import load_stimuli  # noqa: E402


CONDITIONS = ("actual_rank3_remove", "actual_normmatched_orthogonal")
FIELDS = (
    "model_label",
    "seed",
    "gold_count",
    "layer",
    "normalized_depth",
    "condition",
    "prediction",
    "correct",
    "absolute_error",
    "signed_error",
    "clean_prediction",
    "clean_correct",
    "clean_absolute_error",
    "removed_fro_norm",
    "target_removed_fro_norm",
    "norm_ratio",
    "completion",
    "runtime_seconds",
)


def parsed_prediction(result: dict[str, Any]) -> int | None:
    labels = parse_numeric_completion(str(result.get("completion_text", "")))
    value = labels["parsed_count"]
    return int(value) if value is not None else None


def load_geometry(
    packed_root: Path, model_label: str, role: str, layer: int, rank: int
) -> PromptRemovalGeometry:
    path = packed_root / "layers" / f"{model_label}__{role}__L{layer:02d}.npz"
    with np.load(path, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float64)
        counts = np.asarray(data["count"], dtype=int)
        splits = np.asarray(data["split"]).astype(str)
    discovery = splits == "discovery"
    return fit_prompt_removal_geometry(
        states[discovery],
        counts[discovery],
        rank=rank,
        required_classes=np.arange(1, 11),
    )


def existing_keys(path: Path) -> set[tuple[str, int, int, int, str]]:
    if not path.exists():
        return set()
    seen: set[tuple[str, int, int, int, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["model_label"],
                int(row["seed"]),
                int(row["gold_count"]),
                int(row["layer"]),
                row["condition"],
            )
            if key in seen:
                raise RuntimeError(f"duplicate existing result key: {key}")
            seen.add(key)
    return seen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--design-config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument(
        "--support-role",
        choices=("prompt_running", "answer_query"),
        default="prompt_running",
    )
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--counts", nargs="+", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    design = json.loads(args.design_config.read_text(encoding="utf-8"))
    design_key = (
        "prompt_removal"
        if args.support_role == "prompt_running"
        else "answer_query_removal"
    )
    registered = design[design_key]
    realized_norm_tolerance = float(
        registered["realized_norm_relative_tolerance"]
    )
    rank = int(design["rank"])
    seeds = args.seeds or [int(value) for value in design["confirmation_seeds"]]
    counts = args.counts or [int(value) for value in registered["counts"]]
    config = V4Config.from_json(args.v4_config)
    stimuli = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(args.stimuli)
        if str(row.get("design_variant")) == "v4.4"
    }
    missing_stimuli = [
        (seed, count)
        for seed in seeds
        for count in counts
        if (seed, count) not in stimuli
    ]
    if missing_stimuli:
        raise RuntimeError(f"missing registered stimuli: {missing_stimuli[:5]}")

    args.output.mkdir(parents=True, exist_ok=True)
    detail_basename = (
        "layerwise_prompt_removal_detail.csv"
        if args.support_role == "prompt_running"
        else "layerwise_answer_query_removal_detail.csv"
    )
    detail_path = args.output / detail_basename
    seen = existing_keys(detail_path)
    resolved_layers = {
        model: (
            [int(value) for value in args.layers]
            if args.layers is not None
            else [int(value) for value in registered["layers"][model]]
        )
        for model in args.models
    }
    geometries = {
        (model, layer): load_geometry(
            args.packed_root, model, args.support_role, layer, rank
        )
        for model in args.models
        for layer in resolved_layers[model]
    }
    expected = {
        (model, seed, count, layer, condition)
        for model in args.models
        for seed in seeds
        for count in counts
        for layer in resolved_layers[model]
        for condition in CONDITIONS
    }

    with detail_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if detail_path.stat().st_size == 0:
            writer.writeheader()
        for model_label in args.models:
            spec = resolve_model_spec(model_label)
            model, tokenizer, adapter = load_registered_model(
                spec,
                cache_dir=args.cache_dir,
                device_map=args.device_map,
                torch_dtype=config.model_torch_dtype,
                attention_backend=config.attention_prefix_backend,
            )
            layers = resolved_layers[model_label]
            for seed in seeds:
                for count in counts:
                    pending = [
                        (layer, condition)
                        for layer in layers
                        for condition in CONDITIONS
                        if (model_label, seed, count, layer, condition) not in seen
                    ]
                    if not pending:
                        continue
                    encoding = render_v4_prompt(
                        stimuli[(seed, count)],
                        tokenizer=tokenizer,
                        model_spec=spec,
                        config=config,
                        answer_format="numeric",
                    )
                    clean = generate_answer_completion(
                        model,
                        tokenizer,
                        encoding,
                        max_new_tokens=args.max_new_tokens,
                    )
                    clean_prediction = parsed_prediction(clean)
                    clean_error = (
                        abs(clean_prediction - count)
                        if clean_prediction is not None
                        else 10
                    )
                    if args.support_role == "prompt_running":
                        positions = [int(span.end) - 1 for span in encoding.needle_spans]
                        if len(positions) != count:
                            raise RuntimeError(
                                f"active endpoint count mismatch: {len(positions)} != {count}"
                            )
                    else:
                        positions = [int(encoding.query_position)]
                    for layer, condition in pending:
                        measurements: dict[str, float] = {}
                        if args.support_role == "prompt_running":
                            transform = make_prompt_removal_transform(
                                geometries[(model_label, layer)], condition, measurements
                            )
                        else:
                            transform = make_answer_query_removal_transform(
                                geometries[(model_label, layer)],
                                condition,
                                measurements,
                            )
                        intervention_started = time.perf_counter()
                        result = generate_with_residual_transforms(
                            model,
                            tokenizer,
                            adapter,
                            encoding,
                            {layer: (positions, transform)},
                            max_new_tokens=args.max_new_tokens,
                        )
                        runtime = time.perf_counter() - intervention_started
                        prediction = parsed_prediction(result)
                        if "removed_fro_norm" not in measurements:
                            raise RuntimeError("removal hook did not record its norm")
                        if condition == "actual_normmatched_orthogonal" and not np.isclose(
                            measurements["norm_ratio"],
                            1.0,
                            rtol=realized_norm_tolerance,
                            atol=1e-6,
                        ):
                            raise RuntimeError(
                                f"norm-matched control failed: {measurements['norm_ratio']}"
                            )
                        row = {
                            "model_label": model_label,
                            "seed": seed,
                            "gold_count": count,
                            "layer": layer,
                            "normalized_depth": float(layer / max(adapter.num_layers - 1, 1)),
                            "condition": condition,
                            "prediction": "" if prediction is None else prediction,
                            "correct": int(prediction == count) if prediction is not None else 0,
                            "absolute_error": abs(prediction - count) if prediction is not None else 10,
                            "signed_error": "" if prediction is None else prediction - count,
                            "clean_prediction": "" if clean_prediction is None else clean_prediction,
                            "clean_correct": int(clean_prediction == count) if clean_prediction is not None else 0,
                            "clean_absolute_error": clean_error,
                            "removed_fro_norm": measurements["removed_fro_norm"],
                            "target_removed_fro_norm": measurements["target_removed_fro_norm"],
                            "norm_ratio": measurements["norm_ratio"],
                            "completion": result.get("completion_text", ""),
                            "runtime_seconds": runtime,
                        }
                        writer.writerow(row)
                        handle.flush()
                        key = (model_label, seed, count, layer, condition)
                        seen.add(key)
                        print(
                            f"[layer-remove:{args.support_role}] {model_label} seed={seed} N={count} "
                            f"L{layer} {condition} pred={prediction} "
                            f"norm={measurements['removed_fro_norm']:.3f}",
                            flush=True,
                        )
            del model, tokenizer, adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(f"incomplete output grid: {missing[:5]}")
    elapsed = time.perf_counter() - started
    audit = {
        "schema_version": f"realistic_niah_v4_4_layerwise_{design_key}_v1",
        "status": "PASS",
        "command": sys.argv,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "design_config": str(args.design_config),
        "v4_config": str(args.v4_config),
        "packed_root": str(args.packed_root),
        "models": args.models,
        "layers": resolved_layers,
        "seeds": seeds,
        "counts": counts,
        "rank": rank,
        "support_role": args.support_role,
        "conditions": list(CONDITIONS),
        "realized_norm_relative_tolerance": realized_norm_tolerance,
        "expected_cells": len(expected),
        "completed_cells": len(expected),
        "removal_definition": (
            "within-prompt needle-end deviations projected onto the discovery count-centroid rank-3 basis"
            if args.support_role == "prompt_running"
            else "answer-query state relative to the discovery global centroid, projected onto the discovery answer-query count-centroid rank-3 basis"
        ),
        "control_definition": (
            "within-prompt needle-end deviations projected onto a discovery within-count residual basis and rescaled per example to the candidate Frobenius norm"
            if args.support_role == "prompt_running"
            else "answer-query state relative to the discovery global centroid, projected onto an orthogonal discovery within-count residual basis and rescaled per example to the candidate realized Frobenius norm"
        ),
        "elapsed_seconds": elapsed,
    }
    audit_basename = (
        "prompt_removal_audit.json"
        if args.support_role == "prompt_running"
        else "answer_query_removal_audit.json"
    )
    (args.output / audit_basename).write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
