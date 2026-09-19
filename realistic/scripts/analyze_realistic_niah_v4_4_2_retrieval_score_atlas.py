from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


MODES = ("nonthinking", "native_thinking")
CONDITIONS = ("cue_present", "cue_absent")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    width = value.shape[-1] // 2
    return torch.cat((-value[..., width:], value[..., :width]), dim=-1)


def apply_saved_rope(
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    if value.ndim != 3 or cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("Expected value [T,H,D] and matching cos/sin [T,R]")
    if value.shape[0] != cos.shape[0]:
        raise ValueError("RoPE time dimension does not match Q/K")
    rotated_width = min(int(cos.shape[-1]), int(value.shape[-1]))
    if rotated_width % 2:
        raise ValueError("RoPE width must be even")
    prefix = value[..., :rotated_width]
    cos_value = cos[:, None, :rotated_width].to(
        device=value.device, dtype=value.dtype
    )
    sin_value = sin[:, None, :rotated_width].to(
        device=value.device, dtype=value.dtype
    )
    rotated = prefix * cos_value + rotate_half(prefix) * sin_value
    if rotated_width == value.shape[-1]:
        return rotated
    return torch.cat((rotated, value[..., rotated_width:]), dim=-1)


def softcap(scores: torch.Tensor, value: float | None) -> torch.Tensor:
    if value is None:
        return scores
    return torch.tanh(scores / float(value)) * float(value)


def rounded_nested(array: np.ndarray, digits: int = 7) -> list[Any]:
    values = np.round(array.astype(np.float64, copy=False), digits)
    flat = [
        None if not np.isfinite(value) else float(value)
        for value in values.reshape(-1)
    ]
    return np.asarray(flat, dtype=object).reshape(array.shape).tolist()


def broad_primary_from_masses(masses: torch.Tensor) -> torch.Tensor:
    """Old V4.4 span-sum primary: total mass times entropy coverage."""

    epsilon = 1e-12
    total = masses.sum(dim=-1)
    probabilities = masses / total[:, None].clamp_min(epsilon)
    entropy = -torch.where(
        probabilities > 0,
        probabilities * probabilities.clamp_min(epsilon).log(),
        torch.zeros_like(probabilities),
    ).sum(dim=-1)
    coverage = torch.where(
        total > epsilon,
        entropy.exp() / masses.shape[-1],
        torch.zeros_like(total),
    )
    return total * coverage


@torch.inference_mode()
def nonthinking_broad_scores(
    capture_dir: Path,
    manifest: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    query_roles = [str(value) for value in manifest["query_roles"]]
    answer_indices = [
        index for index, role in enumerate(query_roles) if role == "answer_query"
    ]
    if not answer_indices:
        raise RuntimeError(f"No answer-query row in {capture_dir}")
    answer_index = answer_indices[-1]
    query_positions = torch.as_tensor(manifest["query_positions"], dtype=torch.long)
    absolute_query = int(query_positions[answer_index])
    spans = [(int(start), int(end)) for start, end in manifest["needle_spans"]]
    if not spans:
        raise RuntimeError(f"No active needle spans in {capture_dir}")

    file_counts = Counter(str(row["k_file"]) for row in manifest["layers"])
    shared_k_cache: dict[str, torch.Tensor] = {}
    layers: dict[str, Any] = {}
    for layer_row in manifest["layers"]:
        layer = int(layer_row["layer"])
        q_raw = torch.load(
            capture_dir / f"layer_{layer:02d}_q_norm.pt",
            map_location="cpu",
            weights_only=True,
        )
        q_raw = q_raw[answer_index : answer_index + 1]
        k_file = str(layer_row["k_file"])
        if k_file in shared_k_cache:
            k_raw = shared_k_cache[k_file]
        else:
            k_raw = torch.load(
                capture_dir / k_file,
                map_location="cpu",
                weights_only=True,
            )
            if file_counts[k_file] > 1:
                shared_k_cache[k_file] = k_raw
        rope = torch.load(
            capture_dir / f"layer_{layer:02d}_rope.pt",
            map_location="cpu",
            weights_only=True,
        )
        q = q_raw.to(device=device, dtype=torch.float32)
        k = k_raw.to(device=device, dtype=torch.float32)
        cos = rope["cos"].to(device=device, dtype=torch.float32)
        sin = rope["sin"].to(device=device, dtype=torch.float32)
        q = apply_saved_rope(
            q,
            cos[absolute_query : absolute_query + 1],
            sin[absolute_query : absolute_query + 1],
        )
        k = apply_saved_rope(k, cos, sin)
        _, num_heads, head_dim = q.shape
        sequence_length, num_kv_heads, k_head_dim = k.shape
        if head_dim != k_head_dim or num_heads % num_kv_heads:
            raise RuntimeError(
                f"Incompatible Q/K at {capture_dir}/L{layer}: "
                f"q={tuple(q.shape)} k={tuple(k.shape)}"
            )
        groups = num_heads // num_kv_heads
        kv_indices = torch.arange(num_heads, device=device) // groups
        keys_by_head = k[:, kv_indices].permute(1, 2, 0).contiguous()
        queries = q[0]
        scaling = layer_row.get("scaling")
        scaling = head_dim**-0.5 if scaling is None else float(scaling)
        scores = torch.matmul(queries[:, None, :], keys_by_head).squeeze(1)
        scores = softcap(scores * scaling, layer_row.get("softcap"))
        key_positions = torch.arange(sequence_length, device=device)
        allowed = key_positions <= absolute_query
        if layer_row.get("is_sliding") and layer_row.get("sliding_window"):
            allowed &= key_positions >= (
                absolute_query - int(layer_row["sliding_window"]) + 1
            )
        scores.masked_fill_(~allowed[None, :], -torch.inf)
        probabilities = torch.softmax(scores.float(), dim=-1)
        prefix = torch.cat(
            [
                torch.zeros(
                    num_heads, 1, dtype=probabilities.dtype, device=device
                ),
                probabilities.cumsum(dim=-1),
            ],
            dim=-1,
        )
        occurrence_masses = torch.stack(
            [prefix[:, end] - prefix[:, start] for start, end in spans],
            dim=-1,
        )
        primary = broad_primary_from_masses(occurrence_masses)
        layers[str(layer)] = {
            "score": primary.cpu(),
            "mass": occurrence_masses.sum(dim=-1).cpu(),
            "coverage": torch.where(
                occurrence_masses.sum(dim=-1) > 1e-12,
                primary / occurrence_masses.sum(dim=-1).clamp_min(1e-12),
                torch.zeros_like(primary),
            ).cpu(),
        }
        del q_raw, q, k, cos, sin, keys_by_head, queries, scores, probabilities
    return {"layers": layers}


def visible_uniform_baseline(
    manifest: dict[str, Any],
    layer_row: dict[str, Any],
) -> float:
    trace_queries = [
        int(position)
        for position, role in zip(
            manifest["query_positions"], manifest["query_roles"]
        )
        if str(role) == "trace"
    ]
    if not trace_queries:
        return float("nan")
    spans = [(int(start), int(end)) for start, end in manifest["needle_spans"]]
    sliding_window = (
        int(layer_row["sliding_window"])
        if layer_row.get("is_sliding") and layer_row.get("sliding_window")
        else None
    )
    baselines: list[float] = []
    for query in trace_queries:
        allowed_start = max(0, query - sliding_window + 1) if sliding_window else 0
        allowed_end = query + 1
        visible_targets = sum(
            max(0, min(end, allowed_end) - max(start, allowed_start))
            for start, end in spans
        )
        baselines.append(visible_targets / max(1, allowed_end - allowed_start))
    return float(np.mean(baselines))


def native_targeted_scores(
    capture_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    attention = torch.load(
        capture_dir / "attention_summary.pt",
        map_location="cpu",
        weights_only=True,
    )
    region_names = tuple(str(value) for value in attention["region_names"])
    needle_index = region_names.index("needle_span")
    layer_rows = {int(row["layer"]): row for row in manifest["layers"]}
    layers: dict[str, Any] = {}
    for layer, layer_data in attention["layers"].items():
        layer = int(layer)
        mass = layer_data["head_region_mean"][:, needle_index].float()
        baseline = visible_uniform_baseline(manifest, layer_rows[layer])
        score = (
            mass / baseline
            if math.isfinite(baseline) and baseline > 0
            else torch.full_like(mass, float("nan"))
        )
        layers[str(layer)] = {
            "score": score,
            "mass": mass,
            "baseline": baseline,
        }
    return {"layers": layers}


def capture_entries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    entries: list[tuple[Path, dict[str, Any]]] = []
    pattern = "conditions/*/*/*/**/capture/capture_manifest.json"
    for manifest_path in sorted(root.glob(pattern)):
        manifest = read_json(manifest_path)
        if str(manifest.get("mode")) not in MODES:
            continue
        if str(manifest.get("prompt_variant")) not in CONDITIONS:
            continue
        entries.append((manifest_path.parent, manifest))
    return entries


def cache_path(cache_root: Path, manifest: dict[str, Any]) -> Path:
    return (
        cache_root
        / str(manifest["mode"])
        / str(manifest["model_label"])
        / str(manifest["prompt_variant"])
        / f"{manifest['stimulus_id']}.pt"
    )


def build_cache(
    entries: list[tuple[Path, dict[str, Any]]],
    *,
    cache_root: Path,
    device: torch.device,
    overwrite: bool,
    model_filter: str | None,
    mode_filter: str | None,
    limit: int | None,
) -> None:
    selected = [
        row
        for row in entries
        if (model_filter is None or str(row[1]["model_label"]) == model_filter)
        and (mode_filter is None or str(row[1]["mode"]) == mode_filter)
    ]
    if limit is not None:
        selected = selected[:limit]
    for index, (capture_dir, manifest) in enumerate(selected, start=1):
        output = cache_path(cache_root, manifest)
        if output.exists() and not overwrite:
            print(
                f"[retrieval-score] skip {index}/{len(selected)} "
                f"{manifest['model_label']} {manifest['mode']} "
                f"{manifest['prompt_variant']} {manifest['stimulus_id']}",
                flush=True,
            )
            continue
        if str(manifest["mode"]) == "nonthinking":
            result = nonthinking_broad_scores(
                capture_dir,
                manifest,
                device=device,
            )
            score_name = "broad_retrieval_primary"
        else:
            result = native_targeted_scores(capture_dir, manifest)
            score_name = "targeted_retrieval_lift"
        payload = {
            "model": str(manifest["model_label"]),
            "mode": str(manifest["mode"]),
            "condition": str(manifest["prompt_variant"]),
            "stimulus_id": str(manifest["stimulus_id"]),
            "seed": int(manifest["seed"]),
            "gold_count": int(manifest["gold_count"]),
            "score_name": score_name,
            **result,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        print(
            f"[retrieval-score] done {index}/{len(selected)} "
            f"{manifest['model_label']} {manifest['mode']} "
            f"{manifest['prompt_variant']} {manifest['stimulus_id']}",
            flush=True,
        )


def aggregate(
    entries: list[tuple[Path, dict[str, Any]]],
    *,
    cache_root: Path,
    output: Path,
) -> dict[str, Any]:
    models = sorted({str(manifest["model_label"]) for _, manifest in entries})
    payload: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_4_2_retrieval_score_atlas_v1",
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "score_definitions": {
            "nonthinking": {
                "name": "broad retrieval primary",
                "site": "last answer-query row (Total:)",
                "formula": "S_broad = M_needle × exp(H(p_occurrence))/N",
                "pooling": "literal sum over each complete active needle span",
                "provenance": "frozen V4.4 span-sum evidence-times-coverage primary",
            },
            "native_thinking": {
                "name": "targeted retrieval lift",
                "site": "all saved native trace query rows",
                "formula": "S_targeted = mean(trace→needle-span mass) / mean(architecture-visible causal-uniform baseline)",
                "pooling": "union of complete active needle spans",
                "provenance": "target-span attention lift with exact sliding/full visibility",
            },
        },
        "models": {},
    }
    for model in models:
        model_payload: dict[str, Any] = {"modes": {}}
        for mode in MODES:
            mode_payload: dict[str, Any] = {"conditions": {}}
            layer_reference: list[int] | None = None
            head_reference: int | None = None
            for condition in CONDITIONS:
                caches = []
                for _, manifest in entries:
                    if (
                        str(manifest["model_label"]) == model
                        and str(manifest["mode"]) == mode
                        and str(manifest["prompt_variant"]) == condition
                    ):
                        path = cache_path(cache_root, manifest)
                        if not path.exists():
                            raise RuntimeError(f"Missing score cache: {path}")
                        caches.append(
                            torch.load(path, map_location="cpu", weights_only=True)
                        )
                caches.sort(key=lambda row: (int(row["gold_count"]), int(row["seed"])))
                if not caches:
                    continue
                layers = sorted(int(value) for value in caches[0]["layers"])
                if layer_reference is None:
                    layer_reference = layers
                elif layers != layer_reference:
                    raise RuntimeError(f"Layer schema changed for {model}/{mode}")
                score_maps: list[list[Any]] = []
                mass_maps: list[list[Any]] = []
                valid_samples: list[int] = []
                baseline_means: list[Any] = []
                for layer in layers:
                    scores = torch.stack(
                        [row["layers"][str(layer)]["score"].float() for row in caches],
                        dim=0,
                    ).numpy()
                    masses = torch.stack(
                        [row["layers"][str(layer)]["mass"].float() for row in caches],
                        dim=0,
                    ).numpy()
                    if head_reference is None:
                        head_reference = int(scores.shape[1])
                    finite = np.isfinite(scores)
                    count = finite.sum(axis=0)
                    score_mean = np.divide(
                        np.where(finite, scores, 0.0).sum(axis=0),
                        count,
                        out=np.full(scores.shape[1], np.nan, dtype=np.float64),
                        where=count > 0,
                    )
                    mass_mean = np.nanmean(masses, axis=0)
                    score_maps.append(rounded_nested(score_mean))
                    mass_maps.append(rounded_nested(mass_mean))
                    valid_samples.append(int(np.max(count)))
                    if mode == "native_thinking":
                        baselines = np.asarray(
                            [
                                float(row["layers"][str(layer)]["baseline"])
                                for row in caches
                            ],
                            dtype=np.float64,
                        )
                        baseline_means.append(
                            None
                            if not np.isfinite(baselines).any()
                            else round(float(np.nanmean(baselines)), 9)
                        )
                condition_payload = {
                    "samples": len(caches),
                    "layer_head_score": score_maps,
                    "layer_head_mass": mass_maps,
                    "valid_samples_by_layer": valid_samples,
                }
                if mode == "native_thinking":
                    condition_payload["mean_uniform_baseline_by_layer"] = baseline_means
                mode_payload["conditions"][condition] = condition_payload
            mode_payload["layers"] = layer_reference or []
            mode_payload["heads"] = head_reference or 0
            mode_payload["score_definition"] = payload["score_definitions"][mode]
            model_payload["modes"][mode] = mode_payload
        payload["models"][model] = model_payload
    write_json(output, payload)
    summary = {
        "models": models,
        "modes": list(MODES),
        "output": str(output),
        "bytes": output.stat().st_size,
        "cache_files": len(list(cache_root.glob("**/*.pt"))),
    }
    write_json(output.with_name("retrieval_score_atlas_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    entries = capture_entries(arguments.run_root)
    cache_root = arguments.cache_root or arguments.output.with_name(
        "retrieval_score_cache"
    )
    if not arguments.aggregate_only:
        build_cache(
            entries,
            cache_root=cache_root,
            device=torch.device(arguments.device),
            overwrite=arguments.overwrite,
            model_filter=arguments.model,
            mode_filter=arguments.mode,
            limit=arguments.limit,
        )
    if arguments.model is None:
        result = aggregate(
            entries,
            cache_root=cache_root,
            output=arguments.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
