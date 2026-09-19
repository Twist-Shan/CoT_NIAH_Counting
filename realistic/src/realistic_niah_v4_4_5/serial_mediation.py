"""Frozen interventions for V4.4.5 same-forward serial mediation.

The source, retrieval, and late-answer stages are deliberately represented by
different frozen objects.  Nothing in this module assumes that one rank-3
basis is transported unchanged across layers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from realistic_niah_v4.layerwise_removal import (
    PromptRemovalGeometry,
    _closest_realized_norm_replacement,
    _realized_replacement,
    fit_prompt_removal_geometry,
    make_answer_query_removal_transform,
)
from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _is_prompt_prefill,
    _replace_output_tensor,
    _tensor_from_output,
)
from realistic_niah_v4.prompts import PromptEncoding


@dataclass(frozen=True)
class SerialArm:
    name: str
    source: str
    retrieval: str | None
    late: str | None


SERIAL_ARMS: tuple[SerialArm, ...] = (
    SerialArm("C", "none", None, None),
    SerialArm("O", "ordinary", None, None),
    SerialArm("S", "needle", None, None),
    SerialArm("S_Rorth", "needle", "orthogonal", None),
    SerialArm("S_Raligned", "needle", "aligned", None),
    SerialArm("S_Torth", "needle", None, "orthogonal"),
    SerialArm("S_Taligned", "needle", None, "aligned"),
    SerialArm("S_Rorth_Torth", "needle", "orthogonal", "orthogonal"),
    SerialArm("S_Raligned_Torth", "needle", "aligned", "orthogonal"),
    SerialArm("S_Rorth_Taligned", "needle", "orthogonal", "aligned"),
    SerialArm("S_Raligned_Taligned", "needle", "aligned", "aligned"),
)


def serial_arm_map() -> dict[str, SerialArm]:
    return {arm.name: arm for arm in SERIAL_ARMS}


def validate_serial_registry(
    configured_arms: Sequence[str], *, source: int, retrieval: int, late: int
) -> None:
    expected = tuple(arm.name for arm in SERIAL_ARMS)
    if tuple(configured_arms) != expected:
        raise ValueError(f"Serial arms differ from the frozen registry: {expected}")
    if not 0 <= int(source) < int(retrieval) < int(late):
        raise ValueError("Serial layers must satisfy source < retrieval < late")


def load_answer_geometry(
    path: str | Path,
    *,
    discovery_seeds: Sequence[int],
    rank: int = 3,
) -> PromptRemovalGeometry:
    """Fit only from the preregistered discovery rows in a packed NPZ shard."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        missing = {"states", "count", "seed"} - set(payload.files)
        if missing:
            raise ValueError(f"{source} is missing {sorted(missing)}")
        states = np.asarray(payload["states"], dtype=np.float64)
        counts = np.asarray(payload["count"], dtype=np.int64)
        seeds = np.asarray(payload["seed"], dtype=np.int64)
        split = (
            np.asarray(payload["split"]).astype(str)
            if "split" in payload.files
            else None
        )
    selected = np.isin(seeds, np.asarray(discovery_seeds, dtype=np.int64))
    if split is not None:
        selected &= split == "discovery"
    observed_seeds = sorted(set(seeds[selected].tolist()))
    required_seeds = sorted(set(int(value) for value in discovery_seeds))
    if observed_seeds != required_seeds:
        raise RuntimeError(
            f"Answer geometry discovery seeds differ: {observed_seeds} != {required_seeds}"
        )
    return fit_prompt_removal_geometry(
        states[selected],
        counts[selected],
        rank=int(rank),
        required_classes=np.arange(1, 11),
    )


def projection_result(projection: nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Evaluate a linear output projection without re-entering module hooks.

    Retrieval-bank reconstruction runs inside an attention forward hook while
    other capture hooks may already be installed on ``o_proj``.  Calling the
    module recursively would therefore look like a second, one-token model
    forward to those hooks.  Applying the frozen affine map directly is
    exactly equivalent and deliberately bypasses module-level hooks.
    """

    linear = projection
    weight = getattr(linear, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        wrapped = getattr(projection, "linear", None)
        if not isinstance(wrapped, nn.Module):
            raise RuntimeError("Output projection exposes no matrix weight")
        linear = wrapped
        weight = getattr(linear, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError("Output projection exposes no matrix weight")
    if value.ndim != 1 or int(value.numel()) != int(weight.shape[1]):
        raise ValueError("Output-projection input has the wrong width")
    bias = getattr(linear, "bias", None)
    tensor = value.to(device=weight.device, dtype=weight.dtype)
    result = torch.nn.functional.linear(tensor, weight, bias)
    return result.detach().float().cpu()


def selected_bank_output(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    z: torch.Tensor,
) -> torch.Tensor:
    projection = adapter.output_projections[int(layer)]
    width = int(adapter.head_dims[int(layer)])
    zeros = torch.zeros_like(z)
    base = projection_result(projection, zeros)
    writes: list[torch.Tensor] = []
    for head in heads:
        selected = zeros.clone()
        start = int(head) * width
        selected[start : start + width] = z[start : start + width]
        writes.append(projection_result(projection, selected) - base)
    if not writes:
        raise ValueError("The frozen retrieval bank cannot be empty")
    return torch.stack(writes).sum(dim=0)


def deterministic_bank_orthogonal_direction(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    basis: torch.Tensor,
    random_seed: int,
) -> torch.Tensor:
    """Construct a deterministic unit direction in the bank span, orthogonal to rank 3."""

    axes = basis.detach().float().cpu()
    if axes.ndim != 2:
        raise ValueError("Retrieval basis must have shape [rank, hidden]")
    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    width = int(adapter.head_dims[int(layer)])
    width_total = int(adapter.num_heads[int(layer)] * width)
    mask = torch.zeros(width_total, dtype=torch.float32)
    for head in heads:
        start = int(head) * width
        mask[start : start + width] = 1.0
    for _ in range(128):
        z = torch.randn(width_total, generator=generator) * mask
        vector = selected_bank_output(
            adapter, layer=int(layer), heads=heads, z=z
        )
        vector -= (vector @ axes.T) @ axes
        norm = torch.linalg.vector_norm(vector)
        if float(norm) > 1e-8:
            return vector / norm
    raise RuntimeError("Could not construct a retrieval-bank orthogonal control")


def _coordinates_row_major(
    value: torch.Tensor, center: torch.Tensor, basis_rows: torch.Tensor
) -> list[float]:
    coordinates = (value.detach().float().cpu() - center) @ basis_rows.T
    return [float(item) for item in coordinates]


def _coordinates_column_major(
    value: torch.Tensor, center: torch.Tensor, basis_columns: torch.Tensor
) -> list[float]:
    coordinates = (value.detach().float().cpu() - center) @ basis_columns
    return [float(item) for item in coordinates]


@contextmanager
def retrieval_path_hook(
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    heads: Sequence[int],
    mean: torch.Tensor,
    basis: torch.Tensor,
    control_direction: torch.Tensor,
    mode: str | None,
) -> Iterator[dict[str, Any]]:
    """Capture the frozen bank and optionally remove aligned/control signal once."""

    if mode not in {None, "aligned", "orthogonal"}:
        raise ValueError("retrieval mode must be None, aligned, or orthogonal")
    center = mean.detach().float().cpu()
    axes = basis.detach().float().cpu()
    control = control_direction.detach().float().cpu()
    captured_z: dict[str, torch.Tensor | None] = {"value": None}
    audit: dict[str, Any] = {
        "applications": 0,
        "mode": mode or "none",
        "coordinates_before": None,
        "coordinates_after": None,
        "aligned_component_norm": 0.0,
        "target_removed_fro_norm": 0.0,
        "removed_fro_norm": 0.0,
        "norm_ratio": 1.0,
        "orthogonality_max_abs_cosine": 0.0,
    }

    def pre_hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Retrieval hook saw no pre-O tensor")
        value = args[0]
        if _is_prompt_prefill(value, encoding):
            captured_z["value"] = (
                value[0, int(encoding.query_position)].detach().float().cpu()
            )

    def post_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = _tensor_from_output(output)
        if not _is_prompt_prefill(hidden, encoding):
            return output
        z = captured_z["value"]
        if z is None:
            raise RuntimeError("Retrieval post-hook lacks its pre-O capture")
        bank = selected_bank_output(
            adapter, layer=int(layer), heads=heads, z=z
        )
        aligned = ((bank - center) @ axes.T) @ axes
        audit["coordinates_before"] = _coordinates_row_major(bank, center, axes)
        audit["aligned_component_norm"] = float(torch.linalg.vector_norm(aligned))
        patched = hidden
        realized = torch.zeros_like(bank)
        if mode is not None:
            selected = hidden[:, int(encoding.query_position) : int(encoding.query_position) + 1, :]
            # Bank coordinates are intentionally audited on CPU, whereas the
            # causal residual stream lives on the model device.  Move only the
            # intervention vector back to that device before the bf16-realized
            # subtraction; retain the CPU copy above for all reported geometry.
            aligned_on_device = aligned.to(
                device=selected.device, dtype=torch.float32
            )
            target_replacement, realized_target = _realized_replacement(
                selected, aligned_on_device
            )
            target_norm = torch.linalg.vector_norm(realized_target)
            if mode == "aligned":
                replacement = target_replacement
                realized_tensor = realized_target
            else:
                replacement, realized_tensor = _closest_realized_norm_replacement(
                    selected,
                    control.to(device=selected.device, dtype=torch.float32),
                    target_norm,
                )
            patched = hidden.clone()
            patched[:, int(encoding.query_position) : int(encoding.query_position) + 1, :] = replacement
            realized = realized_tensor[0, 0].detach().float().cpu()
            audit["target_removed_fro_norm"] = float(target_norm.detach().float().cpu())
            audit["removed_fro_norm"] = float(torch.linalg.vector_norm(realized))
            audit["norm_ratio"] = float(
                audit["removed_fro_norm"]
                / max(audit["target_removed_fro_norm"], 1e-12)
            )
            if mode == "orthogonal":
                realized_norm = torch.clamp(torch.linalg.vector_norm(realized), min=1e-12)
                audit["orthogonality_max_abs_cosine"] = float(
                    torch.max(torch.abs(realized @ axes.T)) / realized_norm
                )
        audit["coordinates_after"] = _coordinates_row_major(
            bank - realized, center, axes
        )
        audit["applications"] += 1
        captured_z["value"] = None
        return _replace_output_tensor(output, patched) if patched is not hidden else output

    pre = adapter.output_projections[int(layer)].register_forward_pre_hook(pre_hook)
    post = adapter.attentions[int(layer)].register_forward_hook(post_hook)
    try:
        yield audit
    finally:
        pre.remove()
        post.remove()


@contextmanager
def late_answer_path_hook(
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    geometry: PromptRemovalGeometry,
    mode: str | None,
) -> Iterator[dict[str, Any]]:
    """Capture late count coordinates and optionally apply exact/control removal."""

    if mode not in {None, "aligned", "orthogonal"}:
        raise ValueError("late mode must be None, aligned, or orthogonal")
    basis = torch.from_numpy(geometry.basis.astype(np.float32))
    center = torch.from_numpy(geometry.centroids.mean(axis=0).astype(np.float32))
    measurements: dict[str, float] = {}
    transform = (
        None
        if mode is None
        else make_answer_query_removal_transform(
            geometry,
            "actual_rank3_remove"
            if mode == "aligned"
            else "actual_normmatched_orthogonal",
            measurements,
        )
    )
    audit: dict[str, Any] = {
        "applications": 0,
        "mode": mode or "none",
        "coordinates_before": None,
        "coordinates_after": None,
        "target_removed_fro_norm": 0.0,
        "removed_fro_norm": 0.0,
        "norm_ratio": 1.0,
        "orthogonality_max_abs_cosine": 0.0,
    }

    def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = _tensor_from_output(output)
        if not _is_prompt_prefill(hidden, encoding):
            return output
        query = int(encoding.query_position)
        selected = hidden[:, query : query + 1, :]
        before = selected[0, 0].detach().float().cpu()
        audit["coordinates_before"] = _coordinates_column_major(
            before, center, basis
        )
        if transform is None:
            replacement = selected
            realized = torch.zeros_like(before)
            patched = hidden
        else:
            replacement = transform(selected)
            realized = (selected.float() - replacement.float())[0, 0].detach().cpu()
            patched = hidden.clone()
            patched[:, query : query + 1, :] = replacement
            audit.update({key: float(value) for key, value in measurements.items()})
            if mode == "orthogonal":
                realized_norm = torch.clamp(torch.linalg.vector_norm(realized), min=1e-12)
                audit["orthogonality_max_abs_cosine"] = float(
                    torch.max(torch.abs(realized @ basis)) / realized_norm
                )
        after = replacement[0, 0].detach().float().cpu()
        audit["coordinates_after"] = _coordinates_column_major(after, center, basis)
        audit["applications"] += 1
        return _replace_output_tensor(output, patched) if patched is not hidden else output

    handle = adapter.layers[int(layer)].register_forward_hook(hook)
    try:
        yield audit
    finally:
        handle.remove()


def flatten_path_audits(
    retrieval: Mapping[str, Any], late: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix, source in (("retrieval", retrieval), ("late", late)):
        for key, value in source.items():
            result[f"{prefix}_{key}"] = value
    return result
