"""Utilities for position-local causal edits of a Transformers KV cache.

The helpers in this module deliberately operate on *materialized* cache
tensors.  They do not alter token ids, attention masks, or positions.  The
primary supported representation is the Transformers 5 ``Cache`` API, whose
layers expose ``keys`` and ``values`` with shape
``[batch, kv_heads, sequence, head_dim]``.  Legacy tuple caches are accepted so
the core splice can be unit-tested without importing a model implementation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


@dataclass(frozen=True)
class CacheLayerView:
    """Mutable key/value tensor references for one cache layer."""

    key: torch.Tensor
    value: torch.Tensor


def cache_layer_views(past_key_values: Any) -> tuple[CacheLayerView, ...]:
    """Return validated key/value tensor views from a modern or legacy cache."""

    raw_layers: Any
    if hasattr(past_key_values, "layers"):
        raw_layers = getattr(past_key_values, "layers")
        views = []
        for index, layer in enumerate(raw_layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if not isinstance(key, torch.Tensor) or not isinstance(
                value, torch.Tensor
            ):
                raise TypeError(
                    f"Cache layer {index} does not expose tensor keys/values"
                )
            views.append(CacheLayerView(key=key, value=value))
    elif isinstance(past_key_values, (tuple, list)):
        views = []
        for index, layer in enumerate(past_key_values):
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise TypeError(f"Legacy cache layer {index} is not a K/V pair")
            key, value = layer[:2]
            if not isinstance(key, torch.Tensor) or not isinstance(
                value, torch.Tensor
            ):
                raise TypeError(f"Legacy cache layer {index} is not tensor-valued")
            views.append(CacheLayerView(key=key, value=value))
    else:
        raise TypeError(
            "Unsupported KV cache representation; expected .layers or tuples"
        )
    if not views:
        raise ValueError("KV cache has no materialized layers")
    sequence_lengths: set[int] = set()
    for index, view in enumerate(views):
        if view.key.ndim < 3 or view.value.ndim < 3:
            raise ValueError(f"Cache layer {index} tensors have too few dimensions")
        if view.key.shape != view.value.shape:
            raise ValueError(f"Cache layer {index} key/value shapes disagree")
        sequence_lengths.add(int(view.key.shape[-2]))
    if len(sequence_lengths) != 1:
        raise ValueError("KV cache layers do not share one sequence length")
    return tuple(views)


def cache_sequence_length(past_key_values: Any) -> int:
    """Return the common materialized cache sequence length."""

    return int(cache_layer_views(past_key_values)[0].key.shape[-2])


def clone_cache(past_key_values: Any) -> Any:
    """Deep-copy a cache, rejecting implementations that alias the source."""

    try:
        cloned = copy.deepcopy(past_key_values)
    except Exception as exc:  # pragma: no cover - cache-version specific
        raise RuntimeError("Transformers KV cache could not be deep-copied") from exc
    if cloned is past_key_values:
        raise RuntimeError("KV cache deepcopy unexpectedly aliased the source")
    source_views = cache_layer_views(past_key_values)
    cloned_views = cache_layer_views(cloned)
    if len(source_views) != len(cloned_views):
        raise RuntimeError("KV cache deepcopy changed the layer count")
    if any(
        source.key.data_ptr() == target.key.data_ptr()
        or source.value.data_ptr() == target.value.data_ptr()
        for source, target in zip(source_views, cloned_views)
    ):
        raise RuntimeError("KV cache deepcopy retained tensor storage aliases")
    return cloned


def _normalize_indices(
    values: Iterable[int], *, upper_bound: int, label: str
) -> tuple[int, ...]:
    indices = tuple(sorted({int(value) for value in values}))
    if not indices:
        raise ValueError(f"{label} must be nonempty")
    if indices[0] < 0 or indices[-1] >= int(upper_bound):
        raise ValueError(f"{label} fall outside 0..{int(upper_bound) - 1}")
    return indices


@torch.inference_mode()
def splice_cache_positions(
    receiver_cache: Any,
    donor_cache: Any,
    *,
    positions: Sequence[int],
    layers: Sequence[int] | None = None,
    components: Sequence[str] = ("key", "value"),
) -> tuple[Any, dict[str, Any]]:
    """Clone ``receiver_cache`` and transplant selected donor K/V fields.

    Positions are absolute sequence indices.  The splice is same-position only:
    donor position ``p`` always replaces receiver position ``p``.  This keeps
    rotary phase and causal geometry fixed.
    """

    receiver_views = cache_layer_views(receiver_cache)
    donor_views = cache_layer_views(donor_cache)
    if len(receiver_views) != len(donor_views):
        raise ValueError("Donor and receiver cache layer counts disagree")
    receiver_length = int(receiver_views[0].key.shape[-2])
    donor_length = int(donor_views[0].key.shape[-2])
    if receiver_length != donor_length:
        raise ValueError("Donor and receiver cache sequence lengths disagree")
    active_positions = _normalize_indices(
        positions, upper_bound=receiver_length, label="Cache splice positions"
    )
    active_layers = (
        tuple(range(len(receiver_views)))
        if layers is None
        else _normalize_indices(
            layers, upper_bound=len(receiver_views), label="Cache splice layers"
        )
    )
    active_components = tuple(dict.fromkeys(str(value) for value in components))
    if not active_components or any(
        value not in {"key", "value"} for value in active_components
    ):
        raise ValueError("Cache components must be key and/or value")

    hybrid = clone_cache(receiver_cache)
    hybrid_views = cache_layer_views(hybrid)
    totals = {
        component: {
            "elements": 0,
            "changed_elements": 0,
            "absolute_delta_sum": 0.0,
            "squared_delta_sum": 0.0,
            "maximum_absolute_delta": 0.0,
        }
        for component in active_components
    }
    per_layer = []
    for layer in active_layers:
        layer_audit: dict[str, Any] = {"layer": int(layer)}
        for component in active_components:
            source_tensor = getattr(donor_views[layer], component)
            target_tensor = getattr(hybrid_views[layer], component)
            if source_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"Layer {layer} donor/receiver {component} shapes disagree"
                )
            dimension = int(target_tensor.ndim) - 2
            source_index = torch.tensor(
                active_positions, dtype=torch.long, device=source_tensor.device
            )
            target_index = source_index.to(device=target_tensor.device)
            source = source_tensor.index_select(dimension, source_index)
            before = target_tensor.index_select(dimension, target_index)
            if source.device != before.device:
                source = source.to(device=before.device)
            delta = source.float() - before.float()
            elements = int(delta.numel())
            changed = int(torch.count_nonzero(delta).item())
            absolute_sum = float(delta.abs().sum().item())
            squared_sum = float(delta.square().sum().item())
            maximum = float(delta.abs().max().item()) if elements else 0.0
            target_tensor.index_copy_(dimension, target_index, source)
            totals[component]["elements"] += elements
            totals[component]["changed_elements"] += changed
            totals[component]["absolute_delta_sum"] += absolute_sum
            totals[component]["squared_delta_sum"] += squared_sum
            totals[component]["maximum_absolute_delta"] = max(
                float(totals[component]["maximum_absolute_delta"]), maximum
            )
            layer_audit[component] = {
                "elements": elements,
                "changed_elements": changed,
                "mean_absolute_delta": absolute_sum / elements,
                "root_mean_square_delta": (squared_sum / elements) ** 0.5,
                "maximum_absolute_delta": maximum,
            }
        per_layer.append(layer_audit)

    component_audit = {}
    for component, values in totals.items():
        elements = int(values["elements"])
        component_audit[component] = {
            "elements": elements,
            "changed_elements": int(values["changed_elements"]),
            "mean_absolute_delta": float(values["absolute_delta_sum"]) / elements,
            "root_mean_square_delta": (
                float(values["squared_delta_sum"]) / elements
            )
            ** 0.5,
            "maximum_absolute_delta": float(values["maximum_absolute_delta"]),
        }
    changed_total = sum(
        int(values["changed_elements"]) for values in component_audit.values()
    )
    return hybrid, {
        "cache_type": type(receiver_cache).__name__,
        "sequence_length": receiver_length,
        "cache_layer_count": len(receiver_views),
        "spliced_layers": list(active_layers),
        "spliced_positions": list(active_positions),
        "spliced_position_count": len(active_positions),
        "components": list(active_components),
        "component_audit": component_audit,
        "changed_elements": int(changed_total),
        "exact_identity_splice": changed_total == 0,
        "per_layer": per_layer,
    }
