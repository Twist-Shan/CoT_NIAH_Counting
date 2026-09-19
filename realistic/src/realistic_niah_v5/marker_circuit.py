"""Geometry helpers for marker-cache component, layer, and edge scans."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def marker_layer_bands(
    *, num_layers: int, read_layer: int, band_width: int = 4
) -> tuple[dict[str, Any], ...]:
    """Partition causally relevant layers and append one post-read control band."""

    total = int(num_layers)
    read = int(read_layer)
    width = int(band_width)
    if total < 1 or not 1 <= read < total or width < 1:
        raise ValueError("Marker layer-band geometry is invalid")
    bands = []
    for start in range(0, read, width):
        end = min(read, start + width)
        bands.append(
            {
                "label": f"L{start:02d}_{end - 1:02d}",
                "layers": tuple(range(start, end)),
                "causally_precedes_read": True,
            }
        )
    bands.append(
        {
            "label": f"L{read:02d}_{total - 1:02d}_postread_control",
            "layers": tuple(range(read, total)),
            "causally_precedes_read": False,
        }
    )
    flattened = tuple(layer for band in bands for layer in band["layers"])
    if flattened != tuple(range(total)):
        raise RuntimeError("Marker layer bands do not partition the decoder")
    return tuple(bands)


def build_cached_suffix_causal_mask(
    attention_mask: Sequence[int],
    *,
    prefix_length: int,
    end: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build an allowed-edge boolean mask for cached suffix positions.

    The output shape is ``[1, 1, suffix_query, absolute_key]``.  ``True`` means
    the causal attention edge is allowed, matching the 4D-mask convention used
    by the pinned Transformers runtime.
    """

    active = torch.as_tensor(attention_mask, dtype=torch.bool, device=device)
    start = int(prefix_length)
    stop = int(end)
    if active.ndim != 1 or not 0 <= start < stop <= int(active.numel()):
        raise ValueError("Cached suffix mask geometry is invalid")
    query_positions = torch.arange(start, stop, device=device).view(-1, 1)
    key_positions = torch.arange(stop, device=device).view(1, -1)
    allowed = key_positions <= query_positions
    allowed &= active[:stop].view(1, -1)
    return allowed.view(1, 1, stop - start, stop)


def mask_single_attention_edge(
    allowed_mask: torch.Tensor,
    *,
    prefix_length: int,
    query_position: int,
    key_position: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Disable exactly one absolute query-to-key edge in a cached 4D mask."""

    if allowed_mask.dtype != torch.bool or allowed_mask.ndim != 4:
        raise ValueError("Single-edge masking requires a 4D boolean allowed mask")
    if tuple(allowed_mask.shape[:2]) != (1, 1):
        raise ValueError("Single-edge masking currently requires batch/head broadcast")
    start = int(prefix_length)
    query = int(query_position)
    key = int(key_position)
    relative_query = query - start
    if not 0 <= relative_query < int(allowed_mask.shape[-2]):
        raise ValueError("Masked query is outside the cached suffix")
    if not 0 <= key < int(allowed_mask.shape[-1]) or key > query:
        raise ValueError("Masked key is outside the query's causal history")
    if not bool(allowed_mask[0, 0, relative_query, key].item()):
        raise ValueError("Requested attention edge is not allowed in the clean mask")
    result = allowed_mask.clone()
    before_allowed = int(torch.count_nonzero(result).item())
    result[0, 0, relative_query, key] = False
    after_allowed = int(torch.count_nonzero(result).item())
    if before_allowed - after_allowed != 1:
        raise RuntimeError("Single-edge mask changed more than one allowed edge")
    return result, {
        "query_position": query,
        "query_relative_position": relative_query,
        "key_position": key,
        "allowed_edges_before": before_allowed,
        "allowed_edges_after": after_allowed,
        "masked_edge_count": 1,
        "query_key_edge_was_allowed": True,
        "query_key_edge_is_allowed": False,
    }

def edge_key_positions(
    geometry: Mapping[str, Any],
) -> dict[str, int]:
    """Choose deterministic one-key marker and semantic matched controls."""

    regions = geometry["regions"]
    marker = tuple(int(value) for value in regions["marker"])
    payload = tuple(int(value) for value in regions["payload"])
    closing = tuple(int(value) for value in regions["closing"])
    b5 = tuple(int(value) for value in regions["b5"])
    if len(marker) != 1 or not payload or len(closing) != 1 or len(b5) != 1:
        raise ValueError("Exact-edge key geometry is incomplete")
    return {
        "inserted_marker": marker[0],
        "inserted_payload_first": payload[0],
        "inserted_payload_mid": payload[(len(payload) - 1) // 2],
        "inserted_closing": closing[0],
        "pre_insertion_b5": b5[0],
    }
