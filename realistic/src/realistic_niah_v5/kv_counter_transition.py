"""Projection-level interventions for distributed native-trace counter tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _replace_output_tensor,
    _tensor_from_output,
)

from .encoding import NativeTraceEncoding


def normalized_span_bins(
    start: int,
    end: int,
    *,
    bins: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition every token in a nonempty span into near-equal ordered bins."""

    left = int(start)
    right = int(end)
    count = int(bins)
    if left < 0 or right <= left:
        raise ValueError("A binned span must be nonempty and nonnegative")
    if count < 1 or right - left < count:
        raise ValueError("A binned span needs at least one token per bin")
    output = tuple(
        tuple(
            range(
                left + ((right - left) * index) // count,
                left + ((right - left) * (index + 1)) // count,
            )
        )
        for index in range(count)
    )
    flattened = tuple(position for group in output for position in group)
    if flattened != tuple(range(left, right)) or any(not group for group in output):
        raise RuntimeError("Normalized bins did not partition the span exactly")
    return output


def item_bin_positions(
    trace_items: Sequence[Sequence[int]],
    *,
    bins: int,
) -> dict[int, tuple[tuple[int, ...], ...]]:
    """Return one-indexed occurrence-to-normalized-bin token positions."""

    items = tuple((int(value[0]), int(value[1])) for value in trace_items)
    if len(items) != 10:
        raise ValueError("The counter transition assay requires exactly ten items")
    output = {
        occurrence: normalized_span_bins(start, end, bins=int(bins))
        for occurrence, (start, end) in enumerate(items, start=1)
    }
    flattened = [
        position
        for occurrence_bins in output.values()
        for group in occurrence_bins
        for position in group
    ]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Registered item spans overlap")
    return output


def projection_module(attention: torch.nn.Module, kind: str) -> torch.nn.Module:
    """Resolve a Qwen-style raw key or value projection."""

    active = str(kind).lower()
    names = {
        "k": ("k_proj", "key", "key_proj"),
        "v": ("v_proj", "value", "value_proj"),
    }
    if active not in names:
        raise ValueError("Projection kind must be 'k' or 'v'")
    for name in names[active]:
        module = getattr(attention, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    raise RuntimeError(f"Attention exposes no supported {active}-projection")


@torch.inference_mode()
def capture_kv_item_bin_means(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    trace_items: Sequence[Sequence[int]],
    *,
    layers: Sequence[int],
    bins: int,
) -> dict[tuple[int, str], torch.Tensor]:
    """Capture raw K/V projection means as [occurrence,bin,projection_width]."""

    active_layers = tuple(sorted({int(value) for value in layers}))
    if not active_layers or active_layers[0] < 0 or active_layers[-1] >= adapter.num_layers:
        raise ValueError("KV capture layers are outside the decoder")
    positions = item_bin_positions(trace_items, bins=int(bins))
    captured: dict[tuple[int, str], torch.Tensor] = {}
    applications = {
        (layer, kind): 0 for layer in active_layers for kind in ("k", "v")
    }
    handles = []
    for layer in active_layers:
        for kind in ("k", "v"):
            module = projection_module(adapter.attentions[layer], kind)

            def hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                layer: int = layer,
                kind: str = kind,
            ) -> None:
                value = _tensor_from_output(output)
                if value.ndim != 3 or int(value.shape[0]) != 1:
                    raise RuntimeError("KV projection returned an unsupported tensor")
                if int(value.shape[1]) != int(encoding.sequence_length):
                    return
                panels = []
                for occurrence in range(1, 11):
                    panels.append(
                        torch.stack(
                            [
                                value[0, list(group)].float().mean(dim=0)
                                for group in positions[occurrence]
                            ],
                            dim=0,
                        )
                    )
                captured[(layer, kind)] = torch.stack(panels, dim=0).detach().cpu()
                applications[(layer, kind)] += 1

            handles.append(module.register_forward_hook(hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    bad = {key: count for key, count in applications.items() if count != 1}
    if bad or set(captured) != set(applications):
        raise RuntimeError(f"KV capture hooks did not apply exactly once: {bad}")
    return captured


def history_occurrences(receiver: int, scope: str) -> tuple[int, ...]:
    """Resolve a frozen historical-token scope without reading outcomes."""

    active_receiver = int(receiver)
    if not 1 <= active_receiver <= 10:
        raise ValueError("Receiver occurrence is outside 1..10")
    active_scope = str(scope)
    if active_scope == "all_history":
        return tuple(range(1, active_receiver + 1))
    if active_scope == "last_4":
        return tuple(range(max(1, active_receiver - 3), active_receiver + 1))
    raise ValueError("Unknown KV history scope")


def projection_kinds(value: str) -> tuple[str, ...]:
    active = str(value).lower()
    if active == "k":
        return ("k",)
    if active == "v":
        return ("v",)
    if active == "kv":
        return ("k", "v")
    raise ValueError("Projection bank must be k, v, or kv")


@torch.inference_mode()
def add_boundary_and_kv_deltas_capture(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    boundary_position: int,
    boundary_directions: Mapping[int, np.ndarray | torch.Tensor],
    kv_directions: Mapping[tuple[int, str], Mapping[int, np.ndarray | torch.Tensor]],
    read_positions: Sequence[int],
    read_layer: int,
) -> tuple[
    torch.Tensor,
    dict[int, int],
    dict[str, int],
    dict[int, float],
    dict[str, float],
    int,
]:
    """Apply one joint residual/KV edit and capture later block-input states."""

    boundary = int(boundary_position)
    reads = tuple(int(value) for value in read_positions)
    active_read_layer = int(read_layer)
    residual = {
        int(layer): torch.as_tensor(value).detach().float().cpu().reshape(-1)
        for layer, value in boundary_directions.items()
    }
    kv = {
        (int(layer), str(kind)): {
            int(position): torch.as_tensor(value).detach().float().cpu().reshape(-1)
            for position, value in panel.items()
        }
        for (layer, kind), panel in kv_directions.items()
    }
    if not reads or len(set(reads)) != len(reads):
        raise ValueError("Joint-transition read positions must be unique and nonempty")
    if boundary > min(reads) or max(reads) >= int(encoding.sequence_length):
        raise ValueError("Joint-transition positions are inconsistent")
    if any(layer >= active_read_layer for layer in residual):
        raise ValueError("Boundary edits must precede the read layer")
    if any(layer >= active_read_layer or kind not in {"k", "v"} for layer, kind in kv):
        raise ValueError("KV edits must precede the read layer and use K/V projections")

    boundary_applications = {layer: 0 for layer in residual}
    kv_applications = {f"L{layer}_{kind}": 0 for layer, kind in kv}
    boundary_norms = {layer: 0.0 for layer in residual}
    kv_norms = {f"L{layer}_{kind}": 0.0 for layer, kind in kv}
    captured: torch.Tensor | None = None
    read_applications = 0
    handles = []

    for layer, direction in residual.items():
        def boundary_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            direction: torch.Tensor = direction,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Boundary block input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
                return None
            before = hidden[0, boundary]
            if before.numel() != direction.numel():
                raise RuntimeError("Boundary direction width mismatch")
            replacement = (
                before.float() + direction.to(device=before.device)
            ).to(dtype=before.dtype)
            patched = hidden.clone()
            patched[0, boundary] = replacement
            boundary_norms[layer] = float(
                torch.linalg.vector_norm(replacement.float() - before.float()).cpu()
            )
            boundary_applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(boundary_hook))

    for (layer, kind), panel in kv.items():
        module = projection_module(adapter.attentions[layer], kind)
        label = f"L{layer}_{kind}"

        def kv_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            panel: Mapping[int, torch.Tensor] = panel,
            label: str = label,
        ) -> Any:
            value = _tensor_from_output(output)
            if value.ndim != 3 or int(value.shape[0]) != 1:
                raise RuntimeError("KV projection returned an unsupported tensor")
            if int(value.shape[1]) != int(encoding.sequence_length):
                return None
            patched = value.clone()
            squared = 0.0
            for position, direction in panel.items():
                before = value[0, int(position)]
                if before.numel() != direction.numel():
                    raise RuntimeError("KV direction width mismatch")
                replacement = (
                    before.float() + direction.to(device=before.device)
                ).to(dtype=before.dtype)
                patched[0, int(position)] = replacement
                realized = replacement.float() - before.float()
                squared += float(torch.sum(realized * realized).detach().cpu())
            kv_norms[label] = float(np.sqrt(squared))
            kv_applications[label] += 1
            return _replace_output_tensor(output, patched)

        handles.append(module.register_forward_hook(kv_hook))

    def read_hook(_module: Any, args: tuple[Any, ...]) -> None:
        nonlocal captured, read_applications
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Transition read block input is not a tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
            return
        captured = hidden[0, list(reads)].detach().float().cpu()
        read_applications += 1

    handles.append(adapter.layers[active_read_layer].register_forward_pre_hook(read_hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    if (
        any(value != 1 for value in boundary_applications.values())
        or any(value != 1 for value in kv_applications.values())
        or read_applications != 1
        or captured is None
    ):
        raise RuntimeError(
            "Joint transition hooks must apply exactly once: "
            f"boundary={boundary_applications}, kv={kv_applications}, "
            f"read={read_applications}"
        )
    return (
        captured,
        boundary_applications,
        kv_applications,
        boundary_norms,
        kv_norms,
        read_applications,
    )


@torch.inference_mode()
def add_boundary_and_kv_deltas_capture_layers(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    boundary_position: int,
    boundary_directions: Mapping[int, np.ndarray | torch.Tensor],
    kv_directions: Mapping[tuple[int, str], Mapping[int, np.ndarray | torch.Tensor]],
    read_positions: Sequence[int],
    read_layers: Sequence[int],
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, int],
    dict[str, int],
    dict[int, float],
    dict[str, float],
    dict[int, int],
]:
    """Apply a joint edit and capture token trajectories at several block inputs.

    Read hooks are deliberately registered before edit hooks.  A capture at
    block input L therefore reflects edits completed through L-1, while an edit
    registered at L is applied only after that capture.  This keeps each frozen
    block-input probe on the same pre-edit site at which it was trained.
    """

    boundary = int(boundary_position)
    reads = tuple(int(value) for value in read_positions)
    active_read_layers = tuple(sorted({int(value) for value in read_layers}))
    residual = {
        int(layer): torch.as_tensor(value).detach().float().cpu().reshape(-1)
        for layer, value in boundary_directions.items()
    }
    kv = {
        (int(layer), str(kind)): {
            int(position): torch.as_tensor(value).detach().float().cpu().reshape(-1)
            for position, value in panel.items()
        }
        for (layer, kind), panel in kv_directions.items()
    }
    if not reads or len(set(reads)) != len(reads):
        raise ValueError("Trajectory read positions must be unique and nonempty")
    if boundary > min(reads) or min(reads) < 0 or max(reads) >= int(encoding.sequence_length):
        raise ValueError("Trajectory read positions are inconsistent")
    if not active_read_layers:
        raise ValueError("Trajectory capture needs at least one read layer")
    if active_read_layers[0] < 0 or active_read_layers[-1] >= adapter.num_layers:
        raise ValueError("Trajectory read layer is outside the decoder")
    if any(layer < 0 or layer >= active_read_layers[-1] for layer in residual):
        raise ValueError("Boundary edits must precede the final read layer")
    if any(
        layer < 0 or layer >= active_read_layers[-1] or kind not in {"k", "v"}
        for layer, kind in kv
    ):
        raise ValueError("KV edits must precede the final read layer and use K/V projections")

    boundary_applications = {layer: 0 for layer in residual}
    kv_applications = {f"L{layer}_{kind}": 0 for layer, kind in kv}
    boundary_norms = {layer: 0.0 for layer in residual}
    kv_norms = {f"L{layer}_{kind}": 0.0 for layer, kind in kv}
    captured: dict[int, torch.Tensor] = {}
    read_applications = {layer: 0 for layer in active_read_layers}
    handles = []

    # Register reads first so a read at L occurs before a residual edit at L.
    for read_layer in active_read_layers:
        def read_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            read_layer: int = read_layer,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Trajectory read block input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
                return
            captured[read_layer] = hidden[0, list(reads)].detach().float().cpu()
            read_applications[read_layer] += 1

        handles.append(
            adapter.layers[read_layer].register_forward_pre_hook(read_hook)
        )

    for layer, direction in residual.items():
        def boundary_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            direction: torch.Tensor = direction,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Boundary block input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
                return None
            before = hidden[0, boundary]
            if before.numel() != direction.numel():
                raise RuntimeError("Boundary direction width mismatch")
            replacement = (
                before.float() + direction.to(device=before.device)
            ).to(dtype=before.dtype)
            patched = hidden.clone()
            patched[0, boundary] = replacement
            boundary_norms[layer] = float(
                torch.linalg.vector_norm(replacement.float() - before.float()).cpu()
            )
            boundary_applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(boundary_hook))

    for (layer, kind), panel in kv.items():
        module = projection_module(adapter.attentions[layer], kind)
        label = f"L{layer}_{kind}"

        def kv_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            panel: Mapping[int, torch.Tensor] = panel,
            label: str = label,
        ) -> Any:
            value = _tensor_from_output(output)
            if value.ndim != 3 or int(value.shape[0]) != 1:
                raise RuntimeError("KV projection returned an unsupported tensor")
            if int(value.shape[1]) != int(encoding.sequence_length):
                return None
            patched = value.clone()
            squared = 0.0
            for position, direction in panel.items():
                before = value[0, int(position)]
                if before.numel() != direction.numel():
                    raise RuntimeError("KV direction width mismatch")
                replacement = (
                    before.float() + direction.to(device=before.device)
                ).to(dtype=before.dtype)
                patched[0, int(position)] = replacement
                realized = replacement.float() - before.float()
                squared += float(torch.sum(realized * realized).detach().cpu())
            kv_norms[label] = float(np.sqrt(squared))
            kv_applications[label] += 1
            return _replace_output_tensor(output, patched)

        handles.append(module.register_forward_hook(kv_hook))

    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    if (
        any(value != 1 for value in boundary_applications.values())
        or any(value != 1 for value in kv_applications.values())
        or any(value != 1 for value in read_applications.values())
        or set(captured) != set(active_read_layers)
    ):
        raise RuntimeError(
            "Trajectory hooks must apply exactly once: "
            f"boundary={boundary_applications}, kv={kv_applications}, "
            f"read={read_applications}"
        )
    return (
        captured,
        boundary_applications,
        kv_applications,
        boundary_norms,
        kv_norms,
        read_applications,
    )
