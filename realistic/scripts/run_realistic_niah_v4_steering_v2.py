from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from realistic_niah_v4.causal_generation import load_generation_labels
from realistic_niah_v4.geometric_steering import (
    LayerSetSteeringPlan,
    layer_set_steering_plan_scores,
    load_centroid_bundle,
    run_generation_layer_set_centroid_delta,
    select_layer_set_steering_plans,
    summarize_layer_set_steering,
)
from realistic_niah_v4.modeling import load_registered_model
from realistic_niah_v4.pipeline import render_encodings, select_stimuli
from realistic_niah_v4.spec import DESIGN_VARIANTS, V4Config, resolve_model_spec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_gzip_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def _parse_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def _parse_layer_sets(value: str) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        layers = tuple(int(part.strip()) for part in item.split("+") if part.strip())
        if not layers:
            raise argparse.ArgumentTypeError("empty layer set")
        if tuple(sorted(set(layers))) != layers:
            raise argparse.ArgumentTypeError(
                "each layer set must be unique and increasing"
            )
        result.append(layers)
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layer sets must be nonempty and unique")
    return tuple(result)


def _parse_count_pairs(value: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("count pairs use LOW:HIGH syntax")
        lower, upper = (int(part) for part in parts)
        if lower >= upper:
            raise argparse.ArgumentTypeError("count pairs must satisfy LOW < HIGH")
        pairs.append((lower, upper))
    if not pairs or len(set(pairs)) != len(pairs):
        raise argparse.ArgumentTypeError("count pairs must be nonempty and unique")
    return tuple(pairs)


def _directed_pairs(
    canonical: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(pair for lower, upper in canonical for pair in ((lower, upper), (upper, lower)))


def _plans_from_selection(path: Path) -> tuple[LayerSetSteeringPlan, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_split") != "discovery":
        raise ValueError("Confirmation plans must come from discovery selection")
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("Selection JSON has no selected plans")
    plans: list[LayerSetSteeringPlan] = []
    for protocol in ("single_layer", "multi_layer"):
        item = selected.get(protocol)
        if not isinstance(item, dict):
            raise ValueError(f"Selection JSON is missing {protocol}")
        plan = LayerSetSteeringPlan(
            layers=tuple(int(value) for value in item["layers"]),
            alpha=float(item["alpha"]),
        )
        if plan.protocol != protocol:
            raise ValueError(f"Selection protocol/layer mismatch for {protocol}")
        plans.append(plan)
    return tuple(plans)


def _git_head(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run restartable Realistic NIAH V4 discovery-screened single- and "
            "multi-layer answer-query centroid-delta steering."
        )
    )
    parser.add_argument("--phase", required=True, choices=("screen", "confirmation"))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--config", default="configs/realistic_niah_v4.json")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--seeds", required=True, type=_parse_ints)
    parser.add_argument(
        "--variants", type=_parse_strings, default=tuple(DESIGN_VARIANTS)
    )
    parser.add_argument(
        "--count-pairs", type=_parse_count_pairs, default=((7, 8), (9, 10), (5, 10))
    )
    parser.add_argument("--layer-sets", type=_parse_layer_sets)
    parser.add_argument("--alphas", type=_parse_floats, default=(0.25, 0.5, 1.0))
    parser.add_argument("--selection-json")
    parser.add_argument("--random-replicates", type=int, default=1)
    parser.add_argument("--generation-max-new-tokens", type=int, default=16)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    stimuli_path = Path(args.stimuli).resolve()
    run_root = Path(args.run_root).resolve()
    centroid_path = Path(args.centroids).resolve()
    config = V4Config.from_json(config_path)
    model_spec = resolve_model_spec(args.model)
    variants = tuple(str(value) for value in args.variants)
    unknown = sorted(set(variants) - set(DESIGN_VARIANTS))
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    seeds = tuple(int(value) for value in args.seeds)
    expected_split = "discovery" if args.phase == "screen" else "confirmation"
    registered = (
        set(config.discovery_seeds)
        if expected_split == "discovery"
        else set(config.confirmation_seeds)
    )
    if not seeds or not set(seeds).issubset(registered):
        parser.error(f"{args.phase} seeds must belong to the {expected_split} split")
    if int(args.random_replicates) < 1:
        parser.error("random-replicates must be at least one")

    centroids = load_centroid_bundle(centroid_path)
    if args.phase == "screen":
        if not args.layer_sets:
            parser.error("screen phase requires --layer-sets")
        plans = tuple(
            LayerSetSteeringPlan(layers=layers, alpha=float(alpha))
            for layers in args.layer_sets
            for alpha in args.alphas
        )
        selection_path: Path | None = None
    else:
        if not args.selection_json:
            parser.error("confirmation phase requires --selection-json")
        selection_path = Path(args.selection_json).resolve()
        plans = _plans_from_selection(selection_path)

    for plan in plans:
        plan.validate(centroids)
    canonical_pairs = tuple(tuple(int(value) for value in pair) for pair in args.count_pairs)
    valid_counts = set(int(value) for value in config.needle_counts)
    if any(left not in valid_counts or right not in valid_counts for left, right in canonical_pairs):
        parser.error("count pairs must use registered counts")
    directed_pairs = _directed_pairs(canonical_pairs)
    selected_counts = tuple(sorted({value for pair in directed_pairs for value in pair}))

    behavior_path = (
        run_root
        / model_spec.label
        / "numeric"
        / "behavior"
        / "capture"
        / "generation_labels.csv"
    )
    if not behavior_path.exists():
        raise FileNotFoundError(behavior_path)
    behavior_labels = load_generation_labels(behavior_path)
    selection_sha = _sha256_file(selection_path) if selection_path is not None else None
    design = {
        "schema_version": "realistic_niah_v4_layer_set_steering_v2",
        "family": "geometric_steering_v2",
        "phase": args.phase,
        "evaluation_split": expected_split,
        "model_label": model_spec.label,
        "answer_format": "numeric",
        "variants": list(variants),
        "seeds": list(seeds),
        "canonical_count_pairs": [list(pair) for pair in canonical_pairs],
        "directed_count_pairs": [list(pair) for pair in directed_pairs],
        "plans": [
            {
                "protocol": plan.protocol,
                "layers": list(plan.layers),
                "layer_set": plan.label,
                "alpha": float(plan.alpha),
            }
            for plan in plans
        ],
        "selection_json_sha256": selection_sha,
        "random_replicates": int(args.random_replicates),
        "generation_max_new_tokens": int(args.generation_max_new_tokens),
        "stimuli_sha256": _sha256_file(stimuli_path),
        "config_sha256": _sha256_file(config_path),
        "behavior_labels_sha256": _sha256_file(behavior_path),
        "centroids_sha256": _sha256_file(centroid_path),
    }
    design_hash = _json_hash(design)
    stage_root = (
        run_root
        / model_spec.label
        / "numeric"
        / "causal"
        / "geometric_steering_v2"
        / f"{args.phase}_{design_hash}"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    design_path = stage_root / "design.json"
    if design_path.exists():
        observed = json.loads(design_path.read_text(encoding="utf-8"))
        if observed != design:
            raise RuntimeError("Existing steering-v2 design does not match")
    else:
        _write_json_atomic(design_path, design)

    repo_root = Path(args.repo_root).resolve()
    runtime = {
        "unix_time": time.time(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_head": _git_head(repo_root),
        "pid": os.getpid(),
        "argv": sys.argv,
    }
    _write_json_atomic(stage_root / "runtime.json", runtime)

    rows = select_stimuli(
        stimuli_path,
        variants=variants,
        seeds=seeds,
        counts=selected_counts,
        split=expected_split,
    )
    model, tokenizer, adapter = load_registered_model(
        model_spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=config.model_torch_dtype,
        attention_backend=config.attention_prefix_backend,
    )
    encodings = render_encodings(
        rows,
        tokenizer=tokenizer,
        model_label=model_spec.label,
        config=config,
        answer_format="numeric",
    )
    grouped: dict[tuple[str, int], list[Any]] = {}
    for encoding in encodings:
        grouped.setdefault((encoding.design_variant, int(encoding.seed)), []).append(
            encoding
        )

    index_rows: list[dict[str, Any]] = []
    expected_rows = len(directed_pairs) * len(plans) * (1 + int(args.random_replicates))
    capture_root = stage_root / "capture"
    for (variant, seed), family in sorted(grouped.items()):
        shard = capture_root / "shards" / variant / f"seed{seed}.csv.gz"
        if shard.exists() and not args.overwrite:
            frame = pd.read_csv(shard, compression="gzip")
            if len(frame) != expected_rows:
                raise RuntimeError(
                    f"Existing shard {shard} has {len(frame)} rows; expected {expected_rows}"
                )
            if set(frame["design_variant"].astype(str)) != {variant}:
                raise RuntimeError(f"Existing shard variant mismatch: {shard}")
            if set(frame["seed"].astype(int)) != {seed}:
                raise RuntimeError(f"Existing shard seed mismatch: {shard}")
        else:
            frame = run_generation_layer_set_centroid_delta(
                model,
                tokenizer,
                adapter,
                family,
                baseline_labels=behavior_labels,
                centroids=centroids,
                count_pairs=directed_pairs,
                plans=plans,
                random_replicates=int(args.random_replicates),
                max_new_tokens=int(args.generation_max_new_tokens),
            )
            if len(frame) != expected_rows:
                raise RuntimeError(
                    f"Generated {len(frame)} rows for {variant}/seed{seed}; "
                    f"expected {expected_rows}"
                )
            frame["behavior_metric"] = "strict_greedy_complete_numeric_generation"
            _write_csv_gzip_atomic(frame, shard)
        index_rows.append(
            {
                "design_variant": variant,
                "seed": int(seed),
                "rows": int(len(frame)),
                "shard_path": shard.relative_to(capture_root).as_posix(),
                "sha256": _sha256_file(shard),
            }
        )
        print(
            f"[v4 steering v2] {args.phase} complete "
            f"{model_spec.label} {variant} seed={seed} rows={len(frame)}",
            flush=True,
        )

    index_path = capture_root / "capture_index.jsonl"
    index_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in index_rows
    )
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index.write_text(index_text, encoding="utf-8")
    temporary_index.replace(index_path)
    detail = pd.concat(
        [
            pd.read_csv(capture_root / row["shard_path"], compression="gzip")
            for row in index_rows
        ],
        ignore_index=True,
    )
    detail_path = stage_root / "detail.csv.gz"
    _write_csv_gzip_atomic(detail, detail_path)

    outputs: dict[str, Any] = {
        "stage_root": str(stage_root),
        "design": str(design_path),
        "capture_index": str(index_path),
        "detail": str(detail_path),
        "rows": int(len(detail)),
    }
    if args.phase == "screen":
        scores = layer_set_steering_plan_scores(detail)
        scores_path = stage_root / "plan_scores.csv"
        scores.to_csv(scores_path, index=False)
        selection = select_layer_set_steering_plans(detail)
        selection.update(
            {
                "schema_version": "realistic_niah_v4_steering_selection_v2",
                "model_label": model_spec.label,
                "screen_design_hash": design_hash,
                "screen_design_path": str(design_path),
                "discovery_seeds": list(seeds),
                "candidate_plan_count": len(plans),
            }
        )
        locked_path = stage_root / "selection.json"
        _write_json_atomic(locked_path, selection)
        outputs.update(
            {"plan_scores": str(scores_path), "selection": str(locked_path)}
        )
    else:
        summary = summarize_layer_set_steering(detail)
        summary_path = stage_root / "summary.csv"
        summary.to_csv(summary_path, index=False)
        outputs["summary"] = str(summary_path)
    complete_path = stage_root / "complete.json"
    _write_json_atomic(
        complete_path,
        {
            "status": "complete",
            "phase": args.phase,
            "model_label": model_spec.label,
            "design_hash": design_hash,
            "families": len(index_rows),
            "rows": len(detail),
            "completed_unix_time": time.time(),
        },
    )
    outputs["complete"] = str(complete_path)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
