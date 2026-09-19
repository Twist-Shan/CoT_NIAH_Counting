from __future__ import annotations

import torch

from realistic_niah_v5.marker_circuit import (
    build_cached_suffix_causal_mask,
    edge_key_positions,
    marker_layer_bands,
    mask_single_attention_edge,
)


def test_marker_layer_bands_partition_pre_and_post_read_layers() -> None:
    bands = marker_layer_bands(num_layers=36, read_layer=24, band_width=4)

    assert [band["label"] for band in bands[:2]] == ["L00_03", "L04_07"]
    assert bands[-1]["label"] == "L24_35_postread_control"
    assert tuple(layer for band in bands for layer in band["layers"]) == tuple(
        range(36)
    )


def test_single_edge_mask_changes_exactly_one_allowed_entry() -> None:
    mask = build_cached_suffix_causal_mask(
        (1,) * 12, prefix_length=7, end=12
    )
    original = mask.clone()

    masked, audit = mask_single_attention_edge(
        mask, prefix_length=7, query_position=11, key_position=5
    )

    assert tuple(masked.shape) == (1, 1, 5, 12)
    assert torch.equal(mask, original)
    assert bool(masked[0, 0, 4, 5]) is False
    assert bool(masked[0, 0, 3, 5]) is True
    assert int(torch.count_nonzero(mask ^ masked)) == 1
    assert audit["masked_edge_count"] == 1


def test_edge_key_positions_are_one_token_controls() -> None:
    positions = edge_key_positions(
        {
            "regions": {
                "marker": (10,),
                "payload": (11, 12, 13, 14),
                "closing": (15,),
                "b5": (9,),
            }
        }
    )

    assert positions == {
        "inserted_marker": 10,
        "inserted_payload_first": 11,
        "inserted_payload_mid": 12,
        "inserted_closing": 15,
        "pre_insertion_b5": 9,
    }
