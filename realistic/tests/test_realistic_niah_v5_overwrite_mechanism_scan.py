from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from realistic_niah_v5.unified_carrier_transition import (
    carrier_capture_layer_positions,
)
from scripts.run_realistic_niah_v5_list_event_edit_scan import (
    build_list_event_variants,
)
from scripts.run_realistic_niah_v5_overwrite_mechanism_scan import (
    build_structure_scrub_variants,
    choose_future_equal_length_content,
    swap_equal_length_item_contents,
)


@dataclass(frozen=True)
class _Encoding:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    query_position: int = 0
    prompt_token_count: int = 0
    trace_item_spans: tuple[object, ...] = ()
    slot_spans: tuple[object, ...] = ()
    needle_spans: tuple[object, ...] = ()

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


class _CausalBlock(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + torch.cumsum(hidden, dim=1)


class _CausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 3)
        self.layers = torch.nn.ModuleList([_CausalBlock() for _ in range(3)])

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = torch.zeros(
            hidden.shape[0], hidden.shape[1], 5, device=hidden.device
        )
        return SimpleNamespace(logits=logits[:, -int(logits_to_keep) :])


def test_equal_length_item_swap_preserves_geometry_and_swaps_both_spans() -> None:
    encoding = _Encoding(
        input_ids=(10, 11, 20, 21, 30, 31, 40, 41),
        attention_mask=(1,) * 8,
    )
    registry = SimpleNamespace(trace_items=((0, 2), (2, 4), (4, 6), (6, 8)))
    swapped, audit = swap_equal_length_item_contents(
        encoding, registry, left_occurrence=2, right_occurrence=4
    )
    assert swapped.input_ids == (10, 11, 40, 41, 30, 31, 20, 21)
    assert swapped.attention_mask == encoding.attention_mask
    assert audit["item_token_width"] == 2
    assert audit["changed_token_count"] == 4
    assert audit["positions_preserved"] is True


def test_future_content_choice_uses_equal_width_and_exclusions() -> None:
    registry = SimpleNamespace(
        trace_items=((0, 2), (2, 5), (5, 7), (7, 9), (9, 11), (11, 13))
    )
    assert choose_future_equal_length_content(
        registry,
        physical_next_occurrence=3,
        excluded_occurrences=(6,),
    ) == 5
    assert choose_future_equal_length_content(
        registry,
        physical_next_occurrence=5,
        excluded_occurrences=(6,),
    ) is None


def test_structure_scrubs_preserve_separators_and_factor_marker_from_payload() -> None:
    source = _Encoding(
        input_ids=(1, 2, 3, 9, 4, 5, 6, 9, 7, 8, 10),
        attention_mask=(1,) * 11,
    )
    neutral = _Encoding(
        input_ids=(11, 12, 13, 9, 14, 15, 16, 9, 17, 18, 20),
        attention_mask=(1,) * 11,
    )
    registry = SimpleNamespace(trace_items=((0, 3), (4, 7), (8, 11)))
    variants = {
        name: (encoding, audit)
        for name, encoding, audit in build_structure_scrub_variants(
            source,
            neutral,
            registry,
            physical_next_occurrence=2,
        )
    }
    payload = variants["history_payload_scrub"][0]
    markers = variants["history_marker_scrub"][0]
    assert payload.input_ids == (1, 2, 13, 9, 4, 5, 16, 9, 7, 8, 10)
    assert markers.input_ids == (11, 12, 3, 9, 14, 15, 6, 9, 7, 8, 10)
    assert variants["prior_payload_scrub"][0].input_ids == (
        1,
        2,
        13,
        9,
        4,
        5,
        6,
        9,
        7,
        8,
        10,
    )
    assert variants["next_item_marker_scrub"][0].input_ids == (
        1,
        2,
        3,
        9,
        14,
        15,
        6,
        9,
        7,
        8,
        10,
    )
    assert payload.input_ids[3] == payload.input_ids[7] == 9
    assert variants["history_full_item_scrub"][1]["separators_preserved"] is True


def test_list_event_insert_delete_shift_target_and_query_consistently() -> None:
    encoding = _Encoding(
        input_ids=tuple(range(31)),
        attention_mask=(1,) * 31,
        query_position=30,
        prompt_token_count=2,
        trace_item_spans=(object(),),
        slot_spans=(object(),),
        needle_spans=(object(),),
    )
    registry = SimpleNamespace(
        trace_items=((2, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 25), (26, 29)),
        trace_markers=((2, 3), (6, 7), (10, 11), (14, 15), (18, 19), (22, 23), (26, 27)),
    )
    neutral = _Encoding(
        input_ids=tuple(range(100, 131)),
        attention_mask=(1,) * 31,
        query_position=30,
        prompt_token_count=2,
    )
    variants = {
        value["event_variant"]: value
        for value in build_list_event_variants(
            encoding,
            neutral,
            registry,
            receiver=5,
            current_boundary=21,
            target_boundary=25,
            insert_source_occurrence=4,
            delete_occurrence=3,
        )
    }
    inserted = variants["insert_valid_item"]
    deleted = variants["delete_prior_valid_item"]
    assert inserted["token_delta"] == 4
    assert inserted["current_boundary"] == 21
    assert inserted["target_boundary"] == 29
    assert inserted["event_count_target"] == 7
    assert inserted["transition_horizon"] == 2
    assert inserted["encoding"].query_position == 34
    for label in (
        "insert_markerless_valid_payload",
        "insert_marker_neutral_payload",
        "insert_neutral_line",
    ):
        assert variants[label]["token_delta"] == inserted["token_delta"]
        assert variants[label]["target_boundary"] == inserted["target_boundary"]
        assert variants[label]["event_count_target"] == 6
        assert variants[label]["transition_horizon"] == 1
    assert deleted["token_delta"] == -4
    assert deleted["current_boundary"] == 17
    assert deleted["target_boundary"] == 21
    assert deleted["event_count_target"] == 5
    assert deleted["encoding"].query_position == 26
    assert deleted["encoding"].trace_item_spans == ()


def test_multi_layer_carrier_capture_reads_each_layer_once() -> None:
    torch.manual_seed(0)
    model = _CausalModel()
    adapter = SimpleNamespace(layers=model.layers, num_layers=len(model.layers))
    encoding = _Encoding(input_ids=(1, 2, 3, 4), attention_mask=(1, 1, 1, 1))
    targets = {
        0: torch.tensor([1.0, 2.0, 3.0]),
        1: torch.tensor([4.0, 5.0, 6.0]),
    }
    captured, audit = carrier_capture_layer_positions(
        model,
        adapter,
        encoding,
        boundary_position=1,
        boundary_targets=targets,
        kv_directions={},
        read_positions=(1, 3),
        read_layers=(0, 1, 2),
    )
    assert set(captured) == {0, 1, 2}
    assert all(value.shape == (2, 3) for value in captured.values())
    assert torch.allclose(captured[0][0], targets[0])
    assert torch.allclose(captured[1][0], targets[1])
    assert audit["boundary_hook_applications"] == {0: 1, 1: 1}
    assert audit["read_hook_applications"] == {0: 1, 1: 1, 2: 1}
