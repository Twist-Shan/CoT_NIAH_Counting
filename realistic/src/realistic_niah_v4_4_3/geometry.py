from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from realistic_niah_v4.modeling import DecoderAdapter

from .io import sha256_file
from .spec import V443Config


@dataclass(frozen=True)
class DirectionFit:
    unit: torch.Tensor
    slope: torch.Tensor
    slope_norm: float
    projection_count_correlation: float


def _stimulus_stem(count: int, seed: int) -> str:
    return f"V4_4_T10000_N{int(count)}_seed{int(seed)}.npz"


def representation_paths(
    source_run_root: str | Path,
    model_label: str,
    *,
    seeds: Sequence[int],
    counts: Sequence[int],
) -> tuple[list[Path], list[Path]]:
    base = Path(source_run_root) / model_label / "numeric" / "representation"
    prompt_root = base / "capture" / "shards" / "v4.4"
    answer_root = base / "answer_query_all_layers_v1" / "shards" / "v4.4"
    prompt = [prompt_root / _stimulus_stem(10, seed) for seed in seeds]
    answer = [
        answer_root / _stimulus_stem(count, seed)
        for seed in seeds
        for count in counts
    ]
    missing = [str(path) for path in (*prompt, *answer) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing V4.4 representation inputs ({len(missing)}): {missing[:5]}"
        )
    return prompt, answer


def _load_npz_array(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive or "layer_indices" not in archive:
            raise KeyError(f"{path} lacks {key}/layer_indices")
        values = np.asarray(archive[key])
        layers = np.asarray(archive["layer_indices"], dtype=np.int64)
    if values.shape[0] != len(layers):
        raise ValueError(f"Layer axis mismatch in {path}")
    return layers, values


def load_direction_source_states(
    source_run_root: str | Path,
    model_label: str,
    config: V443Config,
    *,
    prompt_source_layers: Mapping[int, int] | None = None,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    list[dict[str, Any]],
]:
    """Load only the frozen discovery tensors needed for OV direction fitting.

    Prompt tensors have shape ``[seed, running_count, hidden]`` and use the
    post-block state immediately before the layer that actually produces V.
    For ordinary attention this is the candidate layer. Gemma4 shared-KV
    layers instead point to the earlier provider layer.
    Answer tensors have shape ``[seed, count, hidden]`` and use the candidate
    head's post-block output layer.
    """

    config.validate()
    seeds = tuple(config.discovery_seeds)
    counts = tuple(range(1, 11))
    target_layers = config.target_layers(model_label)
    source_layers = {
        int(layer): int(
            layer
            if prompt_source_layers is None
            else prompt_source_layers.get(int(layer), int(layer))
        )
        for layer in target_layers
    }
    prompt_paths, answer_paths = representation_paths(
        source_run_root,
        model_label,
        seeds=seeds,
        counts=counts,
    )
    prompt_by_layer: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in target_layers
    }
    answer_by_layer: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in target_layers
    }
    records: list[dict[str, Any]] = []
    for seed, path in zip(seeds, prompt_paths):
        layer_indices, values = _load_npz_array(path, config.prompt_pooling)
        index = {int(layer): offset for offset, layer in enumerate(layer_indices)}
        for layer in target_layers:
            previous = source_layers[int(layer)] - 1
            if previous not in index:
                raise ValueError(
                    f"Prompt shard lacks V-source prior layer {previous}: {path}"
                )
            selected = torch.from_numpy(values[index[previous]].astype(np.float32))
            if selected.ndim != 2 or selected.shape[0] != 10:
                raise ValueError(f"Unexpected prompt counter tensor in {path}")
            prompt_by_layer[layer].append(selected)
        records.append(
            {
                "role": "prompt_counter",
                "model_label": model_label,
                "seed": int(seed),
                "count": 10,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    answer_slots: dict[int, dict[int, list[torch.Tensor]]] = {
        layer: {seed: [] for seed in seeds} for layer in target_layers
    }
    for seed in seeds:
        for count in counts:
            path = (
                Path(source_run_root)
                / model_label
                / "numeric"
                / "representation"
                / "answer_query_all_layers_v1"
                / "shards"
                / "v4.4"
                / _stimulus_stem(count, seed)
            )
            layer_indices, values = _load_npz_array(path, "query_states")
            index = {int(layer): offset for offset, layer in enumerate(layer_indices)}
            for layer in target_layers:
                if layer not in index:
                    raise ValueError(f"Answer shard lacks layer {layer}: {path}")
                selected = torch.from_numpy(values[index[layer]].astype(np.float32))
                if selected.ndim != 1:
                    raise ValueError(f"Unexpected answer query tensor in {path}")
                answer_slots[layer][seed].append(selected)
            records.append(
                {
                    "role": "answer_query",
                    "model_label": model_label,
                    "seed": int(seed),
                    "count": int(count),
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    prompt = {
        layer: torch.stack(per_seed, dim=0)
        for layer, per_seed in prompt_by_layer.items()
    }
    answer = {
        layer: torch.stack(
            [torch.stack(answer_slots[layer][seed], dim=0) for seed in seeds],
            dim=0,
        )
        for layer in target_layers
    }
    return prompt, answer, records


def resolve_pre_attention_norm(block: nn.Module) -> nn.Module:
    for name in (
        "input_layernorm",
        "pre_attention_layernorm",
        "ln_1",
        "attention_norm",
    ):
        module = getattr(block, name, None)
        if isinstance(module, nn.Module):
            return module
    candidates = [
        module
        for name, module in block.named_children()
        if "norm" in name.lower() and isinstance(module, nn.Module)
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"Cannot identify the pre-attention norm in {type(block).__name__}"
    )


@torch.inference_mode()
def apply_pre_attention_norm(
    block: nn.Module,
    states: torch.Tensor,
    *,
    chunk_rows: int = 128,
) -> torch.Tensor:
    if states.ndim < 2:
        raise ValueError("LayerNorm input must include a hidden dimension")
    norm = resolve_pre_attention_norm(block)
    parameters = list(norm.parameters())
    if not parameters:
        raise RuntimeError("Pre-attention norm exposes no parameters")
    reference = parameters[0]
    flat = states.reshape(-1, states.shape[-1])
    outputs = []
    for start in range(0, len(flat), int(chunk_rows)):
        batch = flat[start : start + int(chunk_rows)].to(
            device=reference.device, dtype=reference.dtype
        )
        value = norm(batch)
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not isinstance(value, torch.Tensor) or value.shape != batch.shape:
            raise RuntimeError("Unexpected pre-attention norm output")
        outputs.append(value.detach().float().cpu())
    return torch.cat(outputs, dim=0).reshape(states.shape)


def fit_count_direction(
    states: torch.Tensor,
    *,
    all_counts: Sequence[int] = tuple(range(1, 11)),
    selected_counts: Sequence[int],
) -> DirectionFit:
    """Fit a within-seed OLS count slope and orient it toward larger counts."""

    values = states.detach().float().cpu()
    if values.ndim != 3:
        raise ValueError("states must have shape [seed, count, hidden]")
    if values.shape[1] != len(all_counts):
        raise ValueError("Count axis does not match all_counts")
    index = {int(count): offset for offset, count in enumerate(all_counts)}
    offsets = [index[int(count)] for count in selected_counts]
    selected = values[:, offsets, :]
    count_values = torch.tensor(
        [float(count) for count in selected_counts], dtype=torch.float32
    )
    centered_counts = count_values - count_values.mean()
    centered_states = selected - selected.mean(dim=1, keepdim=True)
    numerator = torch.einsum("c,scd->d", centered_counts, centered_states)
    denominator = float(len(selected)) * float(torch.sum(centered_counts.square()))
    if denominator <= 0:
        raise ValueError("Cannot fit a direction to constant counts")
    slope = numerator / denominator
    norm = float(torch.linalg.vector_norm(slope))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Count slope is degenerate")
    unit = slope / norm
    projections = torch.einsum("scd,d->sc", selected, unit)
    projections = projections - projections.mean(dim=1, keepdim=True)
    repeated_counts = centered_counts.repeat(len(selected)).numpy()
    correlation = float(
        np.corrcoef(repeated_counts, projections.reshape(-1).numpy())[0, 1]
    )
    return DirectionFit(
        unit=unit,
        slope=slope,
        slope_norm=norm,
        projection_count_correlation=correlation,
    )


def _projection_module(attention: nn.Module, name: str) -> nn.Module:
    module = getattr(attention, name, None)
    if not isinstance(module, nn.Module):
        raise RuntimeError(f"{type(attention).__name__} exposes no {name}")
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError(f"{name} is not a two-dimensional linear projection")
    return module


def resolve_value_source_layer(adapter: DecoderAdapter, layer: int) -> int:
    """Resolve the block that produced the value states consumed by ``layer``.

    Most decoder layers own a V projection. Gemma4's final shared-KV layers do
    not: they consume full-length values stored by the last non-shared layer of
    the same attention type. This resolver follows that explicit model graph
    rather than pretending that the target layer owns a missing matrix.
    """

    target = int(layer)
    if not 0 <= target < adapter.num_layers:
        raise ValueError("Value target layer is out of range")
    attention = adapter.attentions[target]
    if isinstance(getattr(attention, "v_proj", None), nn.Module):
        return target
    if not bool(getattr(attention, "is_kv_shared_layer", False)):
        raise RuntimeError(
            f"{type(attention).__name__} has no V projection and is not shared-KV"
        )
    layer_type = adapter.layer_types[target]
    for source in range(target - 1, -1, -1):
        candidate = adapter.attentions[source]
        if adapter.layer_types[source] != layer_type:
            continue
        if not isinstance(getattr(candidate, "v_proj", None), nn.Module):
            continue
        if bool(getattr(candidate, "store_full_length_kv", False)):
            return source
    raise RuntimeError(
        f"Cannot resolve shared value provider for layer {target} ({layer_type})"
    )


@torch.inference_mode()
def project_actual_value_states(
    attention: nn.Module,
    states: torch.Tensor,
    *,
    head_dim: int,
    chunk_rows: int = 128,
) -> torch.Tensor:
    """Apply the actual V path, including a model-specific value RMSNorm."""

    projection = _projection_module(attention, "v_proj")
    weight = projection.weight
    if weight.shape[0] % int(head_dim):
        raise RuntimeError("V projection width is not divisible by head_dim")
    kv_heads = int(weight.shape[0]) // int(head_dim)
    flat = states.reshape(-1, states.shape[-1])
    outputs = []
    value_norm = getattr(attention, "v_norm", None)
    for start in range(0, len(flat), int(chunk_rows)):
        batch = flat[start : start + int(chunk_rows)].to(
            device=weight.device, dtype=weight.dtype
        )
        value = projection(batch).reshape(len(batch), kv_heads, int(head_dim))
        if isinstance(value_norm, nn.Module):
            value = value_norm(value)
        outputs.append(value.detach().float().cpu())
    return torch.cat(outputs, dim=0).reshape(
        *states.shape[:-1], kv_heads, int(head_dim)
    )


@torch.inference_mode()
def o_map_value_direction(
    output_projection: nn.Module,
    *,
    query_head: int,
    head_dim: int,
    value_direction: torch.Tensor,
) -> torch.Tensor:
    weight = getattr(output_projection, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError("Output projection has no two-dimensional weight")
    start = int(query_head) * int(head_dim)
    if start + int(head_dim) > weight.shape[1]:
        raise RuntimeError("Query-head slice lies outside W_O")
    if value_direction.shape != (int(head_dim),):
        raise ValueError("Value direction has the wrong head width")
    result = weight[:, start : start + int(head_dim)].float() @ value_direction.to(
        device=weight.device, dtype=torch.float32
    )
    return result.detach().float().cpu()


def query_to_kv_head(
    *, query_head: int, query_heads: int, kv_heads: int
) -> int:
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("Query/KV head counts must have an integral GQA grouping")
    if not 0 <= int(query_head) < int(query_heads):
        raise ValueError("query_head is out of range")
    return int(query_head) // (int(query_heads) // int(kv_heads))


@torch.inference_mode()
def v_map_direction(
    attention: nn.Module,
    *,
    query_head: int,
    query_heads: int,
    head_dim: int,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Map a residual-space direction into one query head's natural V channel.

    The returned vector lives at the pre-O ``z_h`` boundary.  GQA query heads
    that share a KV head therefore receive the same V-space direction, while
    their later W_O slices can still map it to different residual directions.
    """

    v_projection = _projection_module(attention, "v_proj")
    v_weight = v_projection.weight.detach()
    if v_weight.shape[0] % int(head_dim):
        raise RuntimeError("V projection width is not divisible by head_dim")
    kv_heads = int(v_weight.shape[0]) // int(head_dim)
    kv_head = query_to_kv_head(
        query_head=int(query_head),
        query_heads=int(query_heads),
        kv_heads=kv_heads,
    )
    kv_start = int(kv_head) * int(head_dim)
    source = direction.to(device=v_weight.device, dtype=torch.float32)
    v_slice = v_weight[kv_start : kv_start + int(head_dim)].float()
    mapped = v_slice @ source
    return mapped.detach().float().cpu(), kv_head


@torch.inference_mode()
def ov_map_direction(
    attention: nn.Module,
    output_projection: nn.Module,
    *,
    query_head: int,
    query_heads: int,
    head_dim: int,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    v_projection = _projection_module(attention, "v_proj")
    v_weight = v_projection.weight.detach()
    o_weight = getattr(output_projection, "weight", None)
    if not isinstance(o_weight, torch.Tensor) or o_weight.ndim != 2:
        raise RuntimeError("Output projection has no two-dimensional weight")
    if v_weight.shape[0] % int(head_dim):
        raise RuntimeError("V projection width is not divisible by head_dim")
    kv_heads = int(v_weight.shape[0]) // int(head_dim)
    kv_head = query_to_kv_head(
        query_head=int(query_head),
        query_heads=int(query_heads),
        kv_heads=kv_heads,
    )
    q_start = int(query_head) * int(head_dim)
    kv_start = int(kv_head) * int(head_dim)
    if q_start + int(head_dim) > o_weight.shape[1]:
        raise RuntimeError("Query-head slice lies outside W_O")
    source = direction.to(device=v_weight.device, dtype=torch.float32)
    v_slice = v_weight[kv_start : kv_start + int(head_dim)].float()
    o_slice = o_weight[:, q_start : q_start + int(head_dim)].float()
    mapped = o_slice @ (v_slice @ source)
    return mapped.detach().float().cpu(), kv_head


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 0:
        return math.nan
    return float(torch.dot(left.float(), right.float()) / denominator)


def fit_and_score_ov_heads(
    model: nn.Module,
    adapter: DecoderAdapter,
    *,
    prompt_states: Mapping[int, torch.Tensor],
    answer_states: Mapping[int, torch.Tensor],
    model_label: str,
    config: V443Config,
) -> tuple[pd.DataFrame, dict[int, dict[str, torch.Tensor | float]]]:
    rows: list[dict[str, Any]] = []
    directions: dict[int, dict[str, torch.Tensor | float]] = {}
    for layer in config.target_layers(model_label):
        if not 0 <= int(layer) < adapter.num_layers:
            raise ValueError(f"Target layer {layer} is outside {model_label}")
        value_source_layer = resolve_value_source_layer(adapter, layer)
        value_source_attention = adapter.attentions[value_source_layer]
        normalized_prompt = apply_pre_attention_norm(
            adapter.layers[value_source_layer], prompt_states[layer]
        )
        p_fit = fit_count_direction(
            normalized_prompt, selected_counts=config.fit_counts
        )
        p_holdout = fit_count_direction(
            normalized_prompt, selected_counts=config.heldout_counts
        )
        a_fit = fit_count_direction(
            answer_states[layer], selected_counts=config.fit_counts
        )
        a_holdout = fit_count_direction(
            answer_states[layer], selected_counts=config.heldout_counts
        )
        nonlinear_or_shared_value_path = bool(
            value_source_layer != int(layer)
            or isinstance(getattr(value_source_attention, "v_norm", None), nn.Module)
        )
        value_states = None
        value_fit_by_kv: dict[int, DirectionFit] = {}
        value_holdout_by_kv: dict[int, DirectionFit] = {}
        if nonlinear_or_shared_value_path:
            value_states = project_actual_value_states(
                value_source_attention,
                normalized_prompt,
                head_dim=adapter.head_dims[layer],
            )
            for kv_head in range(value_states.shape[-2]):
                value_fit_by_kv[kv_head] = fit_count_direction(
                    value_states[:, :, kv_head, :],
                    selected_counts=config.fit_counts,
                )
                value_holdout_by_kv[kv_head] = fit_count_direction(
                    value_states[:, :, kv_head, :],
                    selected_counts=config.heldout_counts,
                )
        directions[layer] = {
            "u_prompt_fit": p_fit.unit,
            "u_prompt_holdout": p_holdout.unit,
            "u_answer_fit": a_fit.unit,
            "u_answer_holdout": a_holdout.unit,
            "prompt_slope_norm_fit": p_fit.slope_norm,
            "answer_slope_norm_fit": a_fit.slope_norm,
            "answer_step_scale": a_fit.slope_norm,
            "value_source_layer": int(value_source_layer),
            "value_path_estimand": (
                "empirical_v_proj_then_v_norm_slope_to_target_o"
                if nonlinear_or_shared_value_path
                else "linear_w_o_w_v_prompt_direction"
            ),
            "prompt_projection_count_correlation_fit": p_fit.projection_count_correlation,
            "answer_projection_count_correlation_fit": a_fit.projection_count_correlation,
            "prompt_projection_count_correlation_holdout": p_holdout.projection_count_correlation,
            "answer_projection_count_correlation_holdout": a_holdout.projection_count_correlation,
        }
        mapped_fit_rows: list[torch.Tensor] = []
        mapped_holdout_rows: list[torch.Tensor] = []
        z_count_step_fit_rows: list[torch.Tensor] = []
        z_count_step_holdout_rows: list[torch.Tensor] = []
        for head in range(adapter.num_heads[layer]):
            if nonlinear_or_shared_value_path:
                assert value_states is not None
                kv_head = query_to_kv_head(
                    query_head=head,
                    query_heads=adapter.num_heads[layer],
                    kv_heads=value_states.shape[-2],
                )
                heldout_kv_head = kv_head
                mapped_fit = o_map_value_direction(
                    adapter.output_projections[layer],
                    query_head=head,
                    head_dim=adapter.head_dims[layer],
                    value_direction=(
                        value_fit_by_kv[kv_head].slope / p_fit.slope_norm
                    ),
                )
                mapped_holdout = o_map_value_direction(
                    adapter.output_projections[layer],
                    query_head=head,
                    head_dim=adapter.head_dims[layer],
                    value_direction=(
                        value_holdout_by_kv[kv_head].slope
                        / p_holdout.slope_norm
                    ),
                )
                # The empirical V slope is already measured per one count step.
                # Keep it at the pre-O boundary for mechanism-specific injection
                # and removal instead of replacing it by an arbitrary residual
                # direction in the selected W_O span.
                z_count_step_fit = value_fit_by_kv[kv_head].slope
                z_count_step_holdout = value_holdout_by_kv[kv_head].slope
            else:
                mapped_fit, kv_head = ov_map_direction(
                    adapter.attentions[layer],
                    adapter.output_projections[layer],
                    query_head=head,
                    query_heads=adapter.num_heads[layer],
                    head_dim=adapter.head_dims[layer],
                    direction=p_fit.unit,
                )
                mapped_holdout, heldout_kv_head = ov_map_direction(
                    adapter.attentions[layer],
                    adapter.output_projections[layer],
                    query_head=head,
                    query_heads=adapter.num_heads[layer],
                    head_dim=adapter.head_dims[layer],
                    direction=p_holdout.unit,
                )
                z_count_step_fit, fit_step_kv_head = v_map_direction(
                    adapter.attentions[layer],
                    query_head=head,
                    query_heads=adapter.num_heads[layer],
                    head_dim=adapter.head_dims[layer],
                    direction=p_fit.slope,
                )
                z_count_step_holdout, heldout_step_kv_head = v_map_direction(
                    adapter.attentions[layer],
                    query_head=head,
                    query_heads=adapter.num_heads[layer],
                    head_dim=adapter.head_dims[layer],
                    direction=p_holdout.slope,
                )
                if fit_step_kv_head != kv_head or heldout_step_kv_head != kv_head:
                    raise AssertionError("OV and pre-O V-step GQA mappings disagree")
            if heldout_kv_head != kv_head:
                raise AssertionError("Fit/held-out GQA mapping disagreement")
            mapped_fit_rows.append(mapped_fit.detach().float().cpu())
            mapped_holdout_rows.append(mapped_holdout.detach().float().cpu())
            z_count_step_fit_rows.append(z_count_step_fit.detach().float().cpu())
            z_count_step_holdout_rows.append(
                z_count_step_holdout.detach().float().cpu()
            )
            rows.append(
                {
                    "model_label": model_label,
                    "layer": int(layer),
                    "head": int(head),
                    "kv_head": int(kv_head),
                    "layer_type": adapter.layer_types[layer],
                    "value_source_layer": int(value_source_layer),
                    "value_path_estimand": (
                        "empirical_v_proj_then_v_norm_slope_to_target_o"
                        if nonlinear_or_shared_value_path
                        else "linear_w_o_w_v_prompt_direction"
                    ),
                    "fit_mapping_cosine": _cosine(mapped_fit, a_fit.unit),
                    "heldout_count_mapping_cosine": _cosine(
                        mapped_holdout, a_holdout.unit
                    ),
                    "fit_mapped_norm": float(torch.linalg.vector_norm(mapped_fit)),
                    "heldout_mapped_norm": float(
                        torch.linalg.vector_norm(mapped_holdout)
                    ),
                    "fit_z_count_step_norm": float(
                        torch.linalg.vector_norm(z_count_step_fit)
                    ),
                    "heldout_z_count_step_norm": float(
                        torch.linalg.vector_norm(z_count_step_holdout)
                    ),
                    "fit_signed_answer_component": float(
                        torch.dot(mapped_fit, a_fit.unit)
                    ),
                    "heldout_signed_answer_component": float(
                        torch.dot(mapped_holdout, a_holdout.unit)
                    ),
                    "prompt_fit_count_correlation": p_fit.projection_count_correlation,
                    "answer_fit_count_correlation": a_fit.projection_count_correlation,
                    "prompt_holdout_count_correlation": p_holdout.projection_count_correlation,
                    "answer_holdout_count_correlation": a_holdout.projection_count_correlation,
                    "value_fit_count_correlation": (
                        value_fit_by_kv[kv_head].projection_count_correlation
                        if nonlinear_or_shared_value_path
                        else math.nan
                    ),
                    "value_holdout_count_correlation": (
                        value_holdout_by_kv[kv_head].projection_count_correlation
                        if nonlinear_or_shared_value_path
                        else math.nan
                    ),
                }
            )
        directions[layer]["mapped_fit_by_head"] = torch.stack(mapped_fit_rows)
        directions[layer]["mapped_holdout_by_head"] = torch.stack(
            mapped_holdout_rows
        )
        directions[layer]["z_count_step_fit_by_head"] = torch.stack(
            z_count_step_fit_rows
        )
        directions[layer]["z_count_step_holdout_by_head"] = torch.stack(
            z_count_step_holdout_rows
        )
    frame = pd.DataFrame(rows)
    enriched: list[pd.DataFrame] = []
    for _layer, group in frame.groupby("layer", sort=True):
        part = group.copy().sort_values("head").reset_index(drop=True)
        scores = part["fit_mapping_cosine"].to_numpy(float)
        norms = part["fit_mapped_norm"].to_numpy(float)
        empirical = []
        norm_matched = []
        same_kv = []
        for index, row in part.iterrows():
            empirical.append((1 + int(np.sum(scores >= scores[index]))) / (len(scores) + 1))
            candidates = np.asarray([i for i in range(len(part)) if i != index])
            distance = np.abs(np.log(norms[candidates] + 1e-12) - np.log(norms[index] + 1e-12))
            nearest = candidates[np.argsort(distance)[: config.mapping_norm_match_pool]]
            norm_matched.append(
                (1 + int(np.sum(scores[nearest] >= scores[index]))) / (len(nearest) + 1)
            )
            kv_candidates = part.index[
                part["kv_head"].astype(int).eq(int(row["kv_head"]))
                & ~part["head"].astype(int).eq(int(row["head"]))
            ].to_numpy(int)
            same_kv.append(
                math.nan
                if len(kv_candidates) == 0
                else (1 + int(np.sum(scores[kv_candidates] >= scores[index])))
                / (len(kv_candidates) + 1)
            )
        part["same_layer_empirical_p"] = empirical
        part["norm_matched_empirical_p"] = norm_matched
        part["same_kv_group_empirical_p"] = same_kv
        part["fit_rank_within_layer"] = (
            part["fit_mapping_cosine"].rank(method="min", ascending=False).astype(int)
        )
        enriched.append(part)
    return pd.concat(enriched, ignore_index=True), directions


def select_candidate_and_control_heads(
    scores: pd.DataFrame,
    *,
    model_label: str,
    config: V443Config,
) -> dict[str, Any]:
    required = {
        "model_label",
        "layer",
        "head",
        "fit_mapping_cosine",
        "heldout_count_mapping_cosine",
        "fit_mapped_norm",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"OV score table is missing columns: {missing}")
    frame = scores[scores["model_label"].astype(str).eq(model_label)].copy()
    selected: list[tuple[int, int]] = []
    for layer in config.target_layers(model_label):
        group = frame[frame["layer"].astype(int).eq(int(layer))]
        if group.empty:
            raise ValueError(f"No mapping scores for {model_label} layer {layer}")
        ranked = group.sort_values(
            ["fit_mapping_cosine", "head"], ascending=[False, True]
        )
        selected.extend(
            (int(row.layer), int(row.head))
            for row in ranked.head(config.heads_per_layer).itertuples()
        )
    for sentinel in config.sentinel_heads(model_label):
        if sentinel not in selected:
            selected.append(tuple(int(value) for value in sentinel))
    controls: dict[str, dict[str, int | float]] = {}
    for layer, head in selected:
        candidate = frame[
            frame["layer"].astype(int).eq(layer)
            & frame["head"].astype(int).eq(head)
        ]
        if len(candidate) != 1:
            raise ValueError(f"Missing or duplicate candidate L{layer}H{head}")
        candidate_row = candidate.iloc[0]
        pool = frame[
            frame["layer"].astype(int).eq(layer)
            & ~frame["head"].astype(int).eq(head)
        ].copy()
        if pool.empty:
            raise ValueError(f"No same-layer control pool for L{layer}H{head}")
        median = float(pool["fit_mapping_cosine"].median())
        pool["norm_distance"] = np.abs(
            np.log(pool["fit_mapped_norm"].astype(float) + 1e-12)
            - math.log(float(candidate_row["fit_mapped_norm"]) + 1e-12)
        )
        pool["score_median_distance"] = np.abs(
            pool["fit_mapping_cosine"].astype(float) - median
        )
        control = pool.sort_values(
            ["norm_distance", "score_median_distance", "head"]
        ).iloc[0]
        controls[f"L{layer}H{head}"] = {
            "layer": int(control["layer"]),
            "head": int(control["head"]),
            "fit_mapping_cosine": float(control["fit_mapping_cosine"]),
            "heldout_count_mapping_cosine": float(
                control["heldout_count_mapping_cosine"]
            ),
            "fit_mapped_norm": float(control["fit_mapped_norm"]),
        }
    return {
        "schema_version": "realistic_niah_v4_4_3_head_selection_v1",
        "model_label": model_label,
        "selection_metric": "fit_mapping_cosine",
        "heldout_count_metric_used_for_selection": False,
        "candidate_heads": [
            {
                "layer": layer,
                "head": head,
                "sentinel": (layer, head) in config.sentinel_heads(model_label),
            }
            for layer, head in selected
        ],
        "matched_control_heads": controls,
    }


def deterministic_orthogonal_direction(
    direction: torch.Tensor, *, label: str
) -> torch.Tensor:
    unit = direction.detach().float().cpu()
    unit = unit / torch.linalg.vector_norm(unit)
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidate = torch.randn(unit.shape, generator=generator, dtype=torch.float32)
    candidate = candidate - torch.dot(candidate, unit) * unit
    norm = torch.linalg.vector_norm(candidate)
    if float(norm) <= 1e-8:
        raise RuntimeError("Failed to construct a deterministic orthogonal control")
    return candidate / norm
