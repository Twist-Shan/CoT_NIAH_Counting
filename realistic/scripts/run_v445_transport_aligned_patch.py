from __future__ import annotations

"""Diagnostic causal patch with a discovery-frozen transport-aligned subspace.

The source basis is not a PCA basis.  It is the rank-3 ridge/PLS row-space
that predicts the downstream answer-query count coordinates from discovery
count centroids.  Two supports are compared:

* legacy_endpoint: the last prompt needle endpoint used by V4.4.5;
* answer_query_relay: the same-position answer-query residual immediately
  upstream of the registered downstream target.

This separates a bad source support from a bad descriptive PCA direction.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    _bounded_logits_kwargs,
    _encoding_tensors,
    _last_logits,
    _replace_output_tensor,
    _tensor_from_output,
    capture_post_block_states,
    load_registered_model,
)
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt  # noqa: E402
from realistic_niah_v4.spec import V4Config, resolve_model_spec  # noqa: E402
from realistic_niah_v4.stimuli import load_stimuli  # noqa: E402


MODEL_SITES = {
    "Qwen3-8B": {
        "endpoint_layer": 27,
        "relay_layer": 28,
        "target_layer": 29,
    },
    "Gemma4-E4B": {
        "endpoint_layer": 36,
        "relay_layer": 36,
        "target_layer": 37,
    },
}


def needle_end(encoding: PromptEncoding) -> int:
    return int(encoding.needle_spans[-1].end) - 1


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def aligned_geometry(
    layer_root: Path,
    model_label: str,
    source_role: str,
    source_layer: int,
    target_layer: int,
    *,
    rank: int = 3,
) -> dict[str, Any]:
    source = load_npz(
        layer_root / f"{model_label}__{source_role}__L{source_layer:02d}.npz"
    )
    target = load_npz(
        layer_root / f"{model_label}__answer_query__L{target_layer:02d}.npz"
    )
    source_index = {
        str(sample_id): index
        for index, sample_id in enumerate(source["sample_id"])
        if str(source["split"][index]) == "discovery"
    }
    target_index = {
        str(sample_id): index
        for index, sample_id in enumerate(target["sample_id"])
        if str(target["split"][index]) == "discovery"
    }
    common = sorted(set(source_index) & set(target_index))
    if len(common) < 20:
        raise RuntimeError(f"too few paired discovery rows: {len(common)}")
    source_states = np.stack(
        [source["states"][source_index[sample_id]].astype(np.float32) for sample_id in common]
    )
    target_states = np.stack(
        [target["states"][target_index[sample_id]].astype(np.float32) for sample_id in common]
    )
    counts = np.asarray(
        [int(source["count"][source_index[sample_id]]) for sample_id in common],
        dtype=np.int64,
    )
    target_counts = np.asarray(
        [int(target["count"][target_index[sample_id]]) for sample_id in common],
        dtype=np.int64,
    )
    if not np.array_equal(counts, target_counts):
        raise RuntimeError("source/target count mismatch")
    levels = sorted(set(counts.tolist()))
    if levels != list(range(1, 11)):
        raise RuntimeError(f"unexpected discovery count levels: {levels}")
    source_centroids = np.stack([source_states[counts == count].mean(0) for count in levels])
    target_centroids = np.stack([target_states[counts == count].mean(0) for count in levels])

    target_centered = target_centroids - target_centroids.mean(0, keepdims=True)
    _, _, target_vt = np.linalg.svd(target_centered, full_matrices=False)
    target_basis = target_vt[:rank].T.astype(np.float64)
    target_coordinates = target_centered.astype(np.float64) @ target_basis
    target_scale = np.maximum(target_coordinates.std(0, ddof=1), 1e-8)
    target_coordinates /= target_scale

    source_centered = (
        source_centroids - source_centroids.mean(0, keepdims=True)
    ).astype(np.float64)
    gram = source_centered @ source_centered.T
    ridge = 1e-3 * max(float(np.trace(gram) / len(gram)), 1e-8)
    dual = np.linalg.solve(gram + ridge * np.eye(len(gram)), target_coordinates)
    weights = source_centered.T @ dual
    source_basis, _ = np.linalg.qr(weights, mode="reduced")
    source_basis = source_basis[:, :rank]

    within = source_states.astype(np.float64) - np.stack(
        [source_centroids[count - 1] for count in counts]
    ).astype(np.float64)
    within -= (within @ source_basis) @ source_basis.T
    within_gram = within @ within.T
    eigenvalues, eigenvectors = np.linalg.eigh(within_gram)
    top_value = max(float(eigenvalues[-1]), 1e-12)
    control_axis = within.T @ eigenvectors[:, -1] / np.sqrt(top_value)
    control_axis -= source_basis @ (source_basis.T @ control_axis)
    control_axis /= max(float(np.linalg.norm(control_axis)), 1e-12)

    prediction = source_centered @ weights
    ss_res = float(np.square(target_coordinates - prediction).sum())
    ss_tot = float(np.square(target_coordinates - target_coordinates.mean(0)).sum())
    return {
        "source_basis": torch.from_numpy(source_basis.astype(np.float32)),
        "control_axis": torch.from_numpy(control_axis.astype(np.float32)),
        "source_centroids": torch.from_numpy(source_centroids.astype(np.float32)),
        "target_centroids": torch.from_numpy(target_centroids.astype(np.float32)),
        "paired_discovery_rows": len(common),
        "ridge": ridge,
        "discovery_centroid_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
    }


@torch.inference_mode()
def forward_with_patch_and_target(
    model: Any,
    adapter: Any,
    encoding: PromptEncoding,
    *,
    target_layer: int,
    target_position: int,
    source_layer: int | None = None,
    source_positions: Sequence[int] = (),
    replacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_layer is not None and source_layer >= target_layer:
        raise ValueError("source layer must precede target layer")
    captured: dict[str, torch.Tensor] = {}
    applied = 0
    handles = []
    if source_layer is not None:
        positions = tuple(int(value) for value in source_positions)
        if replacement is None:
            raise ValueError("replacement is required")
        if replacement.ndim == 1:
            replacement = replacement.unsqueeze(0)

        def source_hook(_module, _args, output):
            nonlocal applied
            hidden = _tensor_from_output(output)
            value = replacement.to(device=hidden.device, dtype=hidden.dtype)
            expected = hidden[:, list(positions), :].shape[1:]
            if value.shape != expected:
                raise RuntimeError(f"replacement shape {value.shape} != {expected}")
            patched = hidden.clone()
            patched[:, list(positions), :] = value.unsqueeze(0)
            applied += 1
            return _replace_output_tensor(output, patched)

        handles.append(adapter.layers[source_layer].register_forward_hook(source_hook))

    def target_hook(_module, _args, output):
        hidden = _tensor_from_output(output)
        captured["target"] = hidden[0, int(target_position)].detach().float().cpu()

    handles.append(adapter.layers[target_layer].register_forward_hook(target_hook))
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
    if "target" not in captured:
        raise RuntimeError("target state missing")
    if source_layer is not None and applied != 1:
        raise RuntimeError(f"source hook applied {applied} times")
    return _last_logits(output).detach().float().cpu(), captured["target"]


def one_token_id(tokenizer: Any, count: int) -> int:
    ids = tokenizer.encode(str(int(count)), add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"count {count} is not one token: {ids}")
    return int(ids[0])


def donor_fraction(
    changed: torch.Tensor,
    clean: torch.Tensor,
    receiver_centroid: torch.Tensor,
    donor_centroid: torch.Tensor,
) -> float:
    direction = donor_centroid.float() - receiver_centroid.float()
    numerator = torch.dot(changed.float() - clean.float(), direction)
    denominator = torch.clamp(torch.dot(direction, direction), min=1e-12)
    return float(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--layer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1254])
    parser.add_argument("--pairs", nargs="+", default=["1:2", "2:1", "5:6", "6:5"])
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    config = V4Config.from_json(args.v4_config)
    pairs = [tuple(map(int, item.split(":"))) for item in args.pairs]
    stimuli = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(args.stimuli)
        if str(row.get("design_variant")) == "v4.4"
    }
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    geometry_audit: dict[str, Any] = {}

    for model_label in args.models:
        sites = MODEL_SITES[model_label]
        geometries = {
            "legacy_endpoint": aligned_geometry(
                args.layer_root,
                model_label,
                "prompt_running",
                sites["endpoint_layer"],
                sites["target_layer"],
            ),
            "answer_query_relay": aligned_geometry(
                args.layer_root,
                model_label,
                "answer_query",
                sites["relay_layer"],
                sites["target_layer"],
            ),
        }
        geometry_audit[model_label] = {
            name: {
                "paired_discovery_rows": value["paired_discovery_rows"],
                "ridge": value["ridge"],
                "discovery_centroid_r2": value["discovery_centroid_r2"],
            }
            for name, value in geometries.items()
        }
        model, tokenizer, adapter = load_registered_model(
            resolve_model_spec(model_label),
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            torch_dtype=config.model_torch_dtype,
            attention_backend=config.attention_prefix_backend,
        )
        encoding_cache: dict[tuple[int, int], PromptEncoding] = {}

        def encoding(seed: int, count: int) -> PromptEncoding:
            key = (seed, count)
            if key not in encoding_cache:
                encoding_cache[key] = render_v4_prompt(
                    stimuli[key],
                    tokenizer=tokenizer,
                    model_spec=resolve_model_spec(model_label),
                    config=config,
                    answer_format="numeric",
                )
            return encoding_cache[key]

        for seed in args.seeds:
            for receiver_count, donor_count in pairs:
                receiver = encoding(seed, receiver_count)
                clean_logits, clean_target = forward_with_patch_and_target(
                    model,
                    adapter,
                    receiver,
                    target_layer=sites["target_layer"],
                    target_position=receiver.query_position,
                )
                receiver_token = one_token_id(tokenizer, receiver_count)
                donor_token = one_token_id(tokenizer, donor_count)
                clean_log_odds = float(clean_logits[donor_token] - clean_logits[receiver_token])

                for support, geometry in geometries.items():
                    source_layer = (
                        sites["endpoint_layer"]
                        if support == "legacy_endpoint"
                        else sites["relay_layer"]
                    )
                    source_position = (
                        needle_end(receiver)
                        if support == "legacy_endpoint"
                        else receiver.query_position
                    )
                    _, capture = capture_post_block_states(
                        model,
                        adapter,
                        receiver,
                        [source_position],
                        layers=[source_layer],
                    )
                    receiver_state = capture[source_layer][0]
                    source_delta = (
                        geometry["source_centroids"][donor_count - 1]
                        - geometry["source_centroids"][receiver_count - 1]
                    )
                    basis = geometry["source_basis"]
                    aligned_delta = (source_delta @ basis) @ basis.T
                    main_norm = float(torch.linalg.vector_norm(aligned_delta))
                    conditions = [
                        ("aligned_dose_1", receiver_state + aligned_delta),
                        ("aligned_dose_2", receiver_state + 2.0 * aligned_delta),
                        (
                            "matched_orthogonal",
                            receiver_state + main_norm * geometry["control_axis"],
                        ),
                    ]
                    for condition, replacement in conditions:
                        logits, target = forward_with_patch_and_target(
                            model,
                            adapter,
                            receiver,
                            source_layer=source_layer,
                            source_positions=[source_position],
                            replacement=replacement,
                            target_layer=sites["target_layer"],
                            target_position=receiver.query_position,
                        )
                        log_odds = float(logits[donor_token] - logits[receiver_token])
                        fraction = donor_fraction(
                            target,
                            clean_target,
                            geometry["target_centroids"][receiver_count - 1],
                            geometry["target_centroids"][donor_count - 1],
                        )
                        row = {
                            "model_label": model_label,
                            "seed": seed,
                            "receiver_count": receiver_count,
                            "donor_count": donor_count,
                            "support": support,
                            "condition": condition,
                            "source_layer": source_layer,
                            "target_layer": sites["target_layer"],
                            "replacement_delta_norm": float(
                                torch.linalg.vector_norm(replacement - receiver_state)
                            ),
                            "clean_donor_log_odds": clean_log_odds,
                            "condition_donor_log_odds": log_odds,
                            "donor_log_odds_gain": log_odds - clean_log_odds,
                            "target_donor_fraction": fraction,
                            "argmax_token_changed": int(
                                torch.argmax(logits).item() != torch.argmax(clean_logits).item()
                            ),
                        }
                        rows.append(row)
                        print(
                            f"[{model_label}] {support} {receiver_count}->{donor_count} "
                            f"{condition} logodds={row['donor_log_odds_gain']:+.4f} "
                            f"target={fraction:+.4f}",
                            flush=True,
                        )
        del model, tokenizer, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = args.output / "transport_aligned_patch.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema_version": "realistic_niah_v4_4_5_transport_aligned_patch_v1",
        "models": args.models,
        "seeds": args.seeds,
        "pairs": pairs,
        "rank": 3,
        "basis_fit": "discovery count centroids; ridge prediction of downstream rank-3 count coordinates",
        "supports": ["legacy_endpoint", "answer_query_relay"],
        "geometry": geometry_audit,
        "rows": len(rows),
        "status": "PASS",
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
