from __future__ import annotations

import torch

from realistic_niah_v5.event_cache_splice import (
    cache_sequence_length,
    splice_cache_positions,
)


class _Layer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class _Cache:
    def __init__(self, layers: list[_Layer]) -> None:
        self.layers = layers


def _cache(offset: float = 0.0) -> _Cache:
    base = torch.arange(2 * 1 * 5 * 3, dtype=torch.float32).reshape(2, 1, 5, 3)
    return _Cache(
        [
            _Layer(base[layer : layer + 1] + offset, base[layer : layer + 1] + 100 + offset)
            for layer in range(2)
        ]
    )


def test_splice_cache_positions_changes_only_requested_fields() -> None:
    receiver = _cache()
    donor = _cache(1000.0)

    hybrid, audit = splice_cache_positions(
        receiver,
        donor,
        positions=(1, 3),
        layers=(1,),
        components=("value",),
    )

    assert cache_sequence_length(hybrid) == 5
    assert torch.equal(hybrid.layers[0].values, receiver.layers[0].values)
    assert torch.equal(hybrid.layers[1].keys, receiver.layers[1].keys)
    assert torch.equal(
        hybrid.layers[1].values[..., (1, 3), :],
        donor.layers[1].values[..., (1, 3), :],
    )
    assert torch.equal(
        hybrid.layers[1].values[..., (0, 2, 4), :],
        receiver.layers[1].values[..., (0, 2, 4), :],
    )
    assert audit["changed_elements"] == 6
    assert not audit["exact_identity_splice"]
    # The receiver must remain reusable for another causal branch.
    assert torch.equal(receiver.layers[1].values, _cache().layers[1].values)


def test_identity_splice_is_exact_and_supports_legacy_cache() -> None:
    receiver = tuple(
        (torch.zeros(1, 2, 4, 3), torch.ones(1, 2, 4, 3)) for _ in range(2)
    )

    hybrid, audit = splice_cache_positions(
        receiver,
        receiver,
        positions=(0, 1, 2, 3),
    )

    assert hybrid is not receiver
    assert audit["changed_elements"] == 0
    assert audit["exact_identity_splice"]
    assert torch.equal(hybrid[0][0], receiver[0][0])


def test_splice_rejects_misaligned_cache_lengths() -> None:
    receiver = _cache()
    donor = _cache()
    for layer in donor.layers:
        layer.keys = layer.keys[..., :-1, :]
        layer.values = layer.values[..., :-1, :]

    try:
        splice_cache_positions(receiver, donor, positions=(1,))
    except ValueError as exc:
        assert "sequence lengths" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("Misaligned caches should be rejected")
