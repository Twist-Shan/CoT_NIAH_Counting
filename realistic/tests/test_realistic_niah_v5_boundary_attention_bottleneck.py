import torch

from realistic_niah_v5.boundary_attention_bottleneck import (
    build_standard_4d_causal_mask,
    build_suffix_attention_bottleneck_mask,
    build_transition_attention_bottleneck_mask,
    memory_geometry_positions,
)


class _Encoding:
    sequence_length = 8
    prompt_token_count = 3
    query_position = 7
    attention_mask = (1,) * 8


def test_standard_4d_mask_is_causal() -> None:
    mask = build_standard_4d_causal_mask(_Encoding(), device=torch.device("cpu"))
    assert mask.shape == (1, 1, 8, 8)
    assert bool(mask[0, 0, 5, 5])
    assert not bool(mask[0, 0, 5, 6])


def test_suffix_graph_cut_exposes_only_prompt_boundary_and_suffix() -> None:
    mask = build_suffix_attention_bottleneck_mask(
        _Encoding(),
        boundary_positions=(4,),
        suffix_positions=(6, 7),
        device=torch.device("cpu"),
    )
    assert torch.equal(
        torch.nonzero(mask[0, 0, 6], as_tuple=False).flatten(),
        torch.tensor([0, 1, 2, 4, 6]),
    )
    assert torch.equal(
        torch.nonzero(mask[0, 0, 7], as_tuple=False).flatten(),
        torch.tensor([0, 1, 2, 4, 6, 7]),
    )
    # Non-suffix queries keep ordinary causal attention.
    assert torch.equal(
        torch.nonzero(mask[0, 0, 5], as_tuple=False).flatten(),
        torch.tensor([0, 1, 2, 3, 4, 5]),
    )


class _Registry:
    trace_items = ((3, 5), (6, 9), (10, 14))


class _Tokenizer:
    def decode(self, ids, **_kwargs):
        return "\n" if ids[0] in {5, 9} else "x"


class _GeometryEncoding:
    query_position = 15
    input_ids = tuple(range(16))


def test_memory_geometry_positions_are_predeclared_and_causal() -> None:
    endpoint, _ = memory_geometry_positions(
        _GeometryEncoding(), _Registry(), _Tokenizer(),
        occurrence=2, geometry="item_endpoint"
    )
    suffix4, _ = memory_geometry_positions(
        _GeometryEncoding(), _Registry(), _Tokenizer(),
        occurrence=3, geometry="item_suffix4"
    )
    all_items, _ = memory_geometry_positions(
        _GeometryEncoding(), _Registry(), _Tokenizer(),
        occurrence=2, geometry="all_items_through_k"
    )
    list_prefix, _ = memory_geometry_positions(
        _GeometryEncoding(), _Registry(), _Tokenizer(),
        occurrence=2, geometry="list_prefix_through_k"
    )
    assert endpoint == (8,)
    assert suffix4 == (10, 11, 12, 13)
    assert all_items == (3, 4, 6, 7, 8)
    assert list_prefix == (3, 4, 5, 6, 7, 8, 9)


def test_transition_graph_forces_serial_boundary_relay() -> None:
    class Encoding:
        sequence_length = 10
        prompt_token_count = 2
        query_position = 9
        attention_mask = (1,) * 10

    mask = build_transition_attention_bottleneck_mask(
        Encoding(),
        scaffold_end=3,
        donor_boundary_positions=(4,),
        transition_positions=(5, 6, 7),
        next_boundary_positions=(7,),
        suffix_positions=(8, 9),
        device=torch.device("cpu"),
    )
    assert torch.equal(
        torch.nonzero(mask[0, 0, 6], as_tuple=False).flatten(),
        torch.tensor([0, 1, 2, 4, 5, 6]),
    )
    assert torch.equal(
        torch.nonzero(mask[0, 0, 9], as_tuple=False).flatten(),
        torch.tensor([0, 1, 2, 7, 8, 9]),
    )
