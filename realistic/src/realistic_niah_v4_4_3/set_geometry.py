from __future__ import annotations

import hashlib
import itertools
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .geometry import DecoderAdapter
from .set_spec import V443SetConfig


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 0:
        return math.nan
    return float(torch.dot(left.float(), right.float()) / denominator)


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def _greedy_nested_heads(
    mapped_fit: torch.Tensor,
    answer_direction: torch.Tensor,
    *,
    sizes: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    if mapped_fit.ndim != 2:
        raise ValueError("mapped_fit must be [head, hidden]")
    if max(sizes) > mapped_fit.shape[0]:
        raise ValueError("A requested set is larger than the layer head count")
    selected: list[int] = []
    total = torch.zeros(mapped_fit.shape[1], dtype=torch.float32)
    snapshots: dict[int, tuple[int, ...]] = {}
    for step in range(1, max(sizes) + 1):
        candidates = []
        for head in range(mapped_fit.shape[0]):
            if head in selected:
                continue
            score = _cosine(total + mapped_fit[head].float(), answer_direction)
            candidates.append((score, -head, head))
        if not candidates:
            raise RuntimeError("Greedy set selection exhausted the head registry")
        head = max(candidates)[2]
        selected.append(int(head))
        total = total + mapped_fit[head].float()
        if step in sizes:
            snapshots[int(step)] = tuple(sorted(selected))
    return snapshots


def _combination_pool(
    heads: Sequence[int],
    size: int,
    *,
    limit: int,
    label: str,
) -> list[tuple[int, ...]]:
    universe = math.comb(len(heads), int(size))
    if universe <= int(limit):
        return list(itertools.combinations(tuple(int(h) for h in heads), int(size)))
    generator = np.random.default_rng(_stable_seed(label))
    values: set[tuple[int, ...]] = set()
    head_array = np.asarray(heads, dtype=int)
    while len(values) < int(limit):
        draw = tuple(
            sorted(
                int(value)
                for value in generator.choice(head_array, size=int(size), replace=False)
            )
        )
        values.add(draw)
    return sorted(values)


def _set_metrics(
    heads: Sequence[int],
    *,
    mapped_fit: torch.Tensor,
    mapped_holdout: torch.Tensor,
    answer_fit: torch.Tensor,
    answer_holdout: torch.Tensor,
) -> dict[str, float]:
    index = torch.as_tensor(tuple(int(head) for head in heads), dtype=torch.long)
    fit_vector = mapped_fit[index].sum(dim=0).float()
    heldout_vector = mapped_holdout[index].sum(dim=0).float()
    return {
        "fit_mapping_cosine": _cosine(fit_vector, answer_fit),
        "heldout_count_mapping_cosine": _cosine(
            heldout_vector, answer_holdout
        ),
        "fit_mapped_norm": float(torch.linalg.vector_norm(fit_vector)),
        "heldout_mapped_norm": float(torch.linalg.vector_norm(heldout_vector)),
        "fit_signed_answer_component": float(torch.dot(fit_vector, answer_fit)),
        "heldout_signed_answer_component": float(
            torch.dot(heldout_vector, answer_holdout)
        ),
    }


def select_candidate_and_control_sets(
    scores: pd.DataFrame,
    directions: Mapping[int, Mapping[str, Any]],
    *,
    model_label: str,
    config: V443SetConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate_sets: list[dict[str, Any]] = []
    control_sets: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for layer in config.target_layers(model_label):
        layer_direction = directions[int(layer)]
        mapped_fit = layer_direction["mapped_fit_by_head"].detach().float().cpu()
        mapped_holdout = (
            layer_direction["mapped_holdout_by_head"].detach().float().cpu()
        )
        answer_fit = layer_direction["u_answer_fit"].detach().float().cpu()
        answer_holdout = (
            layer_direction["u_answer_holdout"].detach().float().cpu()
        )
        model_set_sizes = config.set_sizes_for(model_label)
        nested = _greedy_nested_heads(
            mapped_fit,
            answer_fit,
            sizes=model_set_sizes,
        )
        all_heads = tuple(range(mapped_fit.shape[0]))
        for size in model_set_sizes:
            candidate_heads = nested[int(size)]
            set_id = f"L{int(layer)}K{int(size)}"
            candidate_metrics = _set_metrics(
                candidate_heads,
                mapped_fit=mapped_fit,
                mapped_holdout=mapped_holdout,
                answer_fit=answer_fit,
                answer_holdout=answer_holdout,
            )
            null_sets = _combination_pool(
                all_heads,
                int(size),
                limit=config.set_null_samples,
                label=f"{model_label}:L{layer}:K{size}:mapping-null",
            )
            null_sets = [item for item in null_sets if item != candidate_heads]
            null_scores = np.asarray(
                [
                    _set_metrics(
                        item,
                        mapped_fit=mapped_fit,
                        mapped_holdout=mapped_holdout,
                        answer_fit=answer_fit,
                        answer_holdout=answer_holdout,
                    )["fit_mapping_cosine"]
                    for item in null_sets
                ],
                dtype=float,
            )
            candidate_metrics["fit_empirical_tail_fraction"] = float(
                (1 + np.sum(null_scores >= candidate_metrics["fit_mapping_cosine"] - 1e-12))
                / (1 + len(null_scores))
            )
            candidate_metrics["null_set_count"] = int(len(null_sets))
            remaining = tuple(
                head for head in all_heads if head not in set(candidate_heads)
            )
            if len(remaining) < int(size):
                raise RuntimeError(f"No disjoint K={size} control for {set_id}")
            control_pool = _combination_pool(
                remaining,
                int(size),
                limit=config.set_null_samples,
                label=f"{model_label}:L{layer}:K{size}:control-pool",
            )
            control_records = []
            for heads in control_pool:
                metrics = _set_metrics(
                    heads,
                    mapped_fit=mapped_fit,
                    mapped_holdout=mapped_holdout,
                    answer_fit=answer_fit,
                    answer_holdout=answer_holdout,
                )
                control_records.append((heads, metrics))
            median_score = float(
                np.median(
                    [metrics["fit_mapping_cosine"] for _, metrics in control_records]
                )
            )
            candidate_norm = float(candidate_metrics["fit_mapped_norm"])
            norm_ranked = sorted(
                control_records,
                key=lambda item: (
                    abs(
                        math.log(float(item[1]["fit_mapped_norm"]) + 1e-12)
                        - math.log(candidate_norm + 1e-12)
                    ),
                    item[0],
                ),
            )[: min(int(config.set_control_norm_pool), len(control_records))]
            control_heads, control_metrics = min(
                norm_ranked,
                key=lambda item: (
                    abs(float(item[1]["fit_mapping_cosine"]) - median_score),
                    abs(
                        math.log(float(item[1]["fit_mapped_norm"]) + 1e-12)
                        - math.log(candidate_norm + 1e-12)
                    ),
                    item[0],
                ),
            )
            candidate = {
                "set_id": set_id,
                "layer": int(layer),
                "size": int(size),
                "heads": list(candidate_heads),
                "nested_selection": True,
                **candidate_metrics,
            }
            control = {
                "set_id": set_id,
                "layer": int(layer),
                "size": int(size),
                "heads": list(control_heads),
                "disjoint_from_candidate": True,
                **control_metrics,
            }
            candidate_sets.append(candidate)
            control_sets[set_id] = control
            for role, item in (("candidate_set", candidate), ("matched_set", control)):
                rows.append(
                    {
                        "model_label": model_label,
                        "set_id": set_id,
                        "set_role": role,
                        "layer": int(layer),
                        "set_size": int(size),
                        "heads": ",".join(str(head) for head in item["heads"]),
                        **{
                            key: value
                            for key, value in item.items()
                            if key
                            in {
                                "fit_mapping_cosine",
                                "heldout_count_mapping_cosine",
                                "fit_mapped_norm",
                                "heldout_mapped_norm",
                                "fit_signed_answer_component",
                                "heldout_signed_answer_component",
                                "fit_empirical_tail_fraction",
                                "null_set_count",
                            }
                        },
                    }
                )
    selection = {
        "schema_version": "realistic_niah_v4_4_3_set_selection_v1",
        "model_label": model_label,
        "selection_metric": config.set_selection_metric,
        "heldout_count_metric_used_for_selection": False,
        "set_sizes": list(config.set_sizes_for(model_label)),
        "candidate_sets": candidate_sets,
        "matched_control_sets": control_sets,
    }
    return selection, pd.DataFrame(rows)


@torch.inference_mode()
def set_reachable_answer_direction(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    answer_direction: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    projection = adapter.output_projections[int(layer)]
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError("Output projection has no two-dimensional weight")
    width = int(adapter.head_dims[int(layer)])
    blocks = [
        weight[:, int(head) * width : (int(head) + 1) * width].float()
        for head in heads
    ]
    basis = torch.cat(blocks, dim=1)
    answer = answer_direction.to(device=basis.device, dtype=torch.float32)
    answer = answer / torch.linalg.vector_norm(answer)
    coefficients = torch.linalg.lstsq(basis, answer.unsqueeze(1)).solution[:, 0]
    projected = basis @ coefficients
    norm = torch.linalg.vector_norm(projected)
    if not torch.isfinite(norm) or float(norm) <= 1e-8:
        raise RuntimeError("Selected set has no stable answer-direction projection")
    unit = projected / norm
    cosine = float(torch.dot(unit, answer))
    return unit.detach().float().cpu(), cosine


def build_set_direction_artifacts(
    adapter: DecoderAdapter,
    base_directions: Mapping[int, Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    controls = selection["matched_control_sets"]
    for candidate in selection["candidate_sets"]:
        set_id = str(candidate["set_id"])
        layer = int(candidate["layer"])
        layer_directions = base_directions[layer]
        z_count_steps = layer_directions.get("z_count_step_fit_by_head")
        if not isinstance(z_count_steps, torch.Tensor) or z_count_steps.ndim != 2:
            raise RuntimeError(
                "Base directions expose no [head, head_dim] natural V count steps"
            )
        output_projection = adapter.output_projections[layer]
        output_weight = getattr(output_projection, "weight", None)
        if not isinstance(output_weight, torch.Tensor) or output_weight.ndim != 2:
            raise RuntimeError("Output projection has no two-dimensional weight")
        head_width = int(adapter.head_dims[layer])
        for role, item in (
            ("candidate_set", candidate),
            ("matched_set", controls[set_id]),
        ):
            unit, cosine = set_reachable_answer_direction(
                adapter,
                layer=layer,
                heads=item["heads"],
                answer_direction=base_directions[layer]["u_answer_fit"],
            )
            heads = tuple(int(head) for head in item["heads"])
            index = torch.as_tensor(heads, dtype=torch.long)
            selected_z_steps = z_count_steps[index].detach().float().cpu()
            blocks = [
                output_weight[
                    :, int(head) * head_width : (int(head) + 1) * head_width
                ].float()
                for head in heads
            ]
            output_basis = torch.cat(blocks, dim=1)
            natural_output_step = output_basis @ selected_z_steps.reshape(-1).to(
                device=output_basis.device, dtype=torch.float32
            )
            natural_output_step = natural_output_step.detach().float().cpu()
            natural_output_norm = torch.linalg.vector_norm(natural_output_step)
            if (
                not torch.isfinite(natural_output_norm)
                or float(natural_output_norm) <= 1e-8
            ):
                raise RuntimeError(
                    "Selected set has a degenerate natural OV count step"
                )
            answer_unit = layer_directions["u_answer_fit"].detach().float().cpu()
            answer_unit = answer_unit / torch.linalg.vector_norm(answer_unit)
            natural_output_unit = natural_output_step / natural_output_norm
            artifacts[f"{set_id}:{role}"] = {
                "set_id": set_id,
                "set_role": role,
                "layer": layer,
                "heads": list(item["heads"]),
                "reachable_answer_direction": unit,
                "reachable_answer_cosine": cosine,
                # One prompt-count step at the actual pre-O head boundary.  For
                # ordinary attention this is W_V times the fitted prompt slope;
                # for nonlinear/shared-KV attention it is the empirical V-path
                # slope.  No arbitrary post-O steering direction is introduced.
                "natural_ov_z_count_step_by_head": selected_z_steps,
                "natural_ov_output_count_step": natural_output_step,
                "natural_ov_output_count_step_norm": float(natural_output_norm),
                "natural_ov_output_answer_cosine": float(
                    torch.dot(natural_output_unit, answer_unit)
                ),
                "answer_step_scale": float(
                    base_directions[layer]["answer_step_scale"]
                ),
                "u_answer_fit": base_directions[layer]["u_answer_fit"],
            }
    return artifacts
