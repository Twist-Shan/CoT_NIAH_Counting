from __future__ import annotations

"""Pure registration and edge-contribution helpers for follow-up 22/23.

The causal intervention used by these follow-ups is deliberately narrower than
masking a whole span.  For a naturally observed attention row ``alpha`` and
the corresponding value states ``V``, it removes only

    z_edge = sum_{j in E} alpha(q, j) V(j)

at the true pre-output-projection head slice.  The operation therefore tests a
registered natural edge contribution.  It does *not* claim that all
counterfactual attention weights have been recomputed after deleting E.
"""

from dataclasses import replace
import hashlib
import random
from typing import Iterable, Mapping, Sequence

import torch

from realistic_niah_v4.prompts import PromptEncoding, TokenSpan
from realistic_niah_v4_4_3.geometry import query_to_kv_head


def span_positions(spans: Sequence[TokenSpan]) -> set[int]:
    return {
        position
        for span in spans
        for position in range(int(span.start), int(span.end))
    }


def registered_forbidden_positions(encoding: PromptEncoding) -> set[int]:
    """Positions that cannot be used as ordinary-token controls."""

    return span_positions(tuple(encoding.slot_spans) + tuple(encoding.hard_negative_spans)) | {
        int(encoding.query_position)
    }


def repeated_anchor_candidates(
    encoding: PromptEncoding,
    *,
    require_all_active_spans: bool = True,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Return template tokens that occur once per active span and have a successor.

    The mapping value contains ``(anchor_position, successor_position)`` for
    each occurrence in prompt order.  Requiring a unique in-span occurrence
    prevents ambiguous token registration when punctuation repeats.
    """

    spans = tuple(encoding.needle_spans)
    if len(spans) < 2:
        return {}
    ids = tuple(int(value) for value in encoding.input_ids)
    per_span: list[dict[int, tuple[int, int]]] = []
    for span in spans:
        occurrences: dict[int, list[int]] = {}
        for position in range(int(span.start), int(span.end) - 1):
            occurrences.setdefault(ids[position], []).append(position)
        per_span.append(
            {
                token: (positions[0], positions[0] + 1)
                for token, positions in occurrences.items()
                if len(positions) == 1
            }
        )
    tokens = set(per_span[0])
    if require_all_active_spans:
        for values in per_span[1:]:
            tokens.intersection_update(values)
    else:
        tokens.update(*(set(values) for values in per_span[1:]))
    result: dict[int, tuple[tuple[int, int], ...]] = {}
    for token in sorted(tokens):
        pairs = tuple(values[token] for values in per_span if token in values)
        if len(pairs) >= 2:
            result[int(token)] = pairs
    return result


def freeze_anchor_token(
    candidate_maps: Sequence[Mapping[int, Sequence[tuple[int, int]]]],
) -> int:
    """Freeze one model-specific anchor without looking at causal outcomes.

    We first require presence in every discovery prompt.  Ties are resolved by
    the median normalized within-span rank implicit in the absolute positions:
    the later stable template token wins, followed by the smaller token id.
    This deterministic rule is intentionally independent of attention or
    behavior.
    """

    if not candidate_maps:
        raise ValueError("At least one discovery candidate map is required")
    shared = set(int(value) for value in candidate_maps[0])
    for values in candidate_maps[1:]:
        shared.intersection_update(int(value) for value in values)
    if not shared:
        raise ValueError("No repeated anchor token is stable across discovery prompts")

    def score(token: int) -> tuple[float, int]:
        ranks = []
        for values in candidate_maps:
            pairs = tuple(values[token])
            starts = [int(anchor) for anchor, _successor in pairs]
            lo, hi = min(starts), max(starts)
            scale = max(1, hi - lo)
            ranks.extend((position - lo) / scale for position in starts)
        ranks.sort()
        median = ranks[len(ranks) // 2]
        return float(median), -int(token)

    return max(sorted(shared), key=score)


def freeze_anchor_token_from_encodings(
    encodings: Sequence[PromptEncoding],
) -> tuple[int, dict[str, float]]:
    """Freeze the stable anchor whose immediate successor is most variable.

    This favors a repeated template token immediately before record-specific
    content, which is the canonical setting for a previous-match-to-successor
    induction relation.  The rule uses token registration only—never attention
    or behavioral outcomes.
    """

    if not encodings:
        raise ValueError("At least one discovery encoding is required")
    maps = [repeated_anchor_candidates(encoding) for encoding in encodings]
    shared = set(maps[0])
    for values in maps[1:]:
        shared.intersection_update(values)
    if not shared:
        raise ValueError("No stable repeated anchor exists across discovery encodings")
    diagnostics: dict[int, tuple[int, float]] = {}
    for token in sorted(shared):
        successors = []
        offsets = []
        for encoding, values in zip(encodings, maps):
            for anchor, successor in values[token]:
                successors.append(int(encoding.input_ids[int(successor)]))
                containing = next(
                    span
                    for span in encoding.needle_spans
                    if int(span.start) <= int(anchor) < int(span.end)
                )
                offsets.append(
                    (int(anchor) - int(containing.start))
                    / max(1, int(containing.end) - int(containing.start) - 1)
                )
        diagnostics[int(token)] = (
            len(set(successors)),
            float(sum(offsets) / len(offsets)),
        )
    selected = max(
        sorted(shared),
        key=lambda token: (
            diagnostics[int(token)][0],
            diagnostics[int(token)][1],
            -int(token),
        ),
    )
    unique_successors, mean_offset = diagnostics[int(selected)]
    return int(selected), {
        "stable_candidate_count": float(len(shared)),
        "unique_successor_token_count": float(unique_successors),
        "mean_normalized_anchor_offset": float(mean_offset),
    }


def context_halo_positions(
    encoding: PromptEncoding,
    *,
    width: int,
) -> tuple[int, ...]:
    """Ordinary positions immediately surrounding active needle spans."""

    if int(width) <= 0:
        raise ValueError("Context halo width must be positive")
    forbidden = registered_forbidden_positions(encoding)
    query = int(encoding.query_position)
    values: set[int] = set()
    for span in encoding.needle_spans:
        candidates = range(
            max(0, int(span.start) - int(width)),
            min(query, int(span.end) + int(width)),
        )
        values.update(position for position in candidates if position not in forbidden)
    if not values:
        raise ValueError("No ordinary context-halo positions were available")
    return tuple(sorted(values))


def distance_bin(query: int, key: int, *, width: int) -> int:
    if int(width) <= 0:
        raise ValueError("Distance-bin width must be positive")
    if not 0 <= int(key) < int(query):
        raise ValueError("A causal edge requires 0 <= key < query")
    return (int(query) - int(key) - 1) // int(width)


def select_control_feasible_candidates(
    *,
    query: int,
    ranked_candidates: Iterable[int],
    allowed_controls: Iterable[int],
    bin_width: int,
    max_candidates: int,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Keep the highest-ranked candidates admitting unique exact-bin controls.

    Candidate rank is supplied by the caller and must be frozen without using
    any intervention outcome.  The per-bin capacity is the number of eligible
    ordinary control keys in that exact distance bin.  Capping selected
    candidates at this capacity guarantees that both matched-control arms can
    use distinct keys without falling back to another distance bin.
    """

    if int(max_candidates) <= 0:
        raise ValueError("Maximum candidate count must be positive")
    ranked = tuple(dict.fromkeys(int(value) for value in ranked_candidates))
    if not ranked:
        raise ValueError("No ranked candidate keys were supplied")
    for key in ranked:
        if not 0 <= key < int(query):
            raise ValueError("A candidate key lies outside the causal prefix")
    controls = tuple(
        sorted(
            {
                int(value)
                for value in allowed_controls
                if 0 <= int(value) < int(query)
            }
        )
    )
    capacity_by_bin: dict[int, int] = {}
    for key in controls:
        index = distance_bin(int(query), key, width=int(bin_width))
        capacity_by_bin[index] = capacity_by_bin.get(index, 0) + 1

    selected: list[int] = []
    selected_by_bin: dict[int, int] = {}
    omitted_no_capacity = 0
    omitted_capacity_exhausted = 0
    for key in ranked:
        if len(selected) >= int(max_candidates):
            break
        index = distance_bin(int(query), key, width=int(bin_width))
        capacity = capacity_by_bin.get(index, 0)
        used = selected_by_bin.get(index, 0)
        if capacity == 0:
            omitted_no_capacity += 1
            continue
        if used >= capacity:
            omitted_capacity_exhausted += 1
            continue
        selected.append(key)
        selected_by_bin[index] = used + 1

    if not selected:
        raise ValueError("No candidate key admits a unique exact-bin control")
    audit: dict[str, object] = {
        "policy": "natural-attention rank subject to unique exact-distance-bin control capacity",
        "ranked_candidate_count": len(ranked),
        "requested_max_candidates": int(max_candidates),
        "selected_candidate_count": len(selected),
        "omitted_no_exact_bin_capacity": omitted_no_capacity,
        "omitted_exhausted_bin_capacity": omitted_capacity_exhausted,
        "capacity_by_distance_bin": {
            str(index): int(value) for index, value in sorted(capacity_by_bin.items())
        },
        "selected_by_distance_bin": {
            str(index): int(value) for index, value in sorted(selected_by_bin.items())
        },
        "all_selected_have_unique_exact_bin_capacity": all(
            selected_by_bin[index] <= capacity_by_bin[index]
            for index in selected_by_bin
        ),
    }
    return tuple(selected), audit


def _candidate_pool(
    *,
    query: int,
    target_key: int,
    allowed: Iterable[int],
    excluded: Iterable[int],
    bin_width: int,
) -> tuple[list[int], bool]:
    excluded_set = {int(value) for value in excluded} | {int(target_key)}
    causal = sorted(
        int(value)
        for value in set(allowed)
        if 0 <= int(value) < int(query) and int(value) not in excluded_set
    )
    target_bin = distance_bin(query, target_key, width=bin_width)
    exact = [
        key
        for key in causal
        if distance_bin(query, key, width=bin_width) == target_bin
    ]
    return (exact if exact else causal), bool(exact)


def select_attention_mass_control(
    attention_row: torch.Tensor,
    *,
    query: int,
    target_key: int,
    allowed: Iterable[int],
    excluded: Iterable[int] = (),
    bin_width: int = 64,
) -> tuple[int, dict[str, float | bool]]:
    """Choose a deterministic distance-then-attention-mass matched key."""

    row = torch.as_tensor(attention_row, dtype=torch.float64).flatten()
    if int(target_key) >= len(row):
        raise ValueError("Target key lies outside the attention row")
    pool, exact_bin = _candidate_pool(
        query=int(query),
        target_key=int(target_key),
        allowed=allowed,
        excluded=excluded,
        bin_width=int(bin_width),
    )
    if not pool:
        raise ValueError("No eligible matched-control key exists")
    target_mass = float(row[int(target_key)])
    chosen = min(
        pool,
        key=lambda key: (
            abs(float(row[key]) - target_mass),
            abs((int(query) - key) - (int(query) - int(target_key))),
            key,
        ),
    )
    return chosen, {
        "exact_distance_bin": exact_bin,
        "target_attention_mass": target_mass,
        "control_attention_mass": float(row[chosen]),
        "absolute_attention_mass_difference": abs(float(row[chosen]) - target_mass),
    }


def select_deterministic_random_control(
    *,
    query: int,
    target_key: int,
    allowed: Iterable[int],
    excluded: Iterable[int] = (),
    bin_width: int = 64,
    label: str,
) -> tuple[int, dict[str, bool]]:
    """Choose an outcome-blind reproducible key from the same distance bin."""

    pool, exact_bin = _candidate_pool(
        query=int(query),
        target_key=int(target_key),
        allowed=allowed,
        excluded=excluded,
        bin_width=int(bin_width),
    )
    if not pool:
        raise ValueError("No eligible random-control key exists")
    seed = int.from_bytes(hashlib.sha256(str(label).encode("utf-8")).digest()[:8], "big")
    chosen = random.Random(seed).choice(pool)
    return int(chosen), {"exact_distance_bin": exact_bin}


def kv_slice_for_query_head(
    values: torch.Tensor,
    *,
    query_head: int,
    query_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Map a query head to its GQA value-head slice."""

    matrix = torch.as_tensor(values).detach().float().cpu()
    if matrix.ndim != 2 or matrix.shape[-1] % int(head_dim):
        raise ValueError("Value states must have shape [position, kv_heads*head_dim]")
    kv_heads = int(matrix.shape[-1]) // int(head_dim)
    kv_head = query_to_kv_head(
        query_head=int(query_head), query_heads=int(query_heads), kv_heads=kv_heads
    )
    start = kv_head * int(head_dim)
    return matrix[:, start : start + int(head_dim)]


def natural_edge_delta(
    attention_row: torch.Tensor,
    value_states: torch.Tensor,
    *,
    keys: Sequence[int],
    key_start: int,
    query_head: int,
    query_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``-sum(alpha*V)`` for registered natural edges."""

    row = torch.as_tensor(attention_row).detach().float().cpu().flatten()
    values = kv_slice_for_query_head(
        value_states,
        query_head=int(query_head),
        query_heads=int(query_heads),
        head_dim=int(head_dim),
    )
    absolute = tuple(sorted({int(value) for value in keys}))
    if not absolute:
        raise ValueError("At least one edge key is required")
    relative = torch.as_tensor(
        [position - int(key_start) for position in absolute], dtype=torch.long
    )
    if int(relative.min()) < 0 or int(relative.max()) >= len(row):
        raise ValueError("An edge key lies outside the captured attention row")
    source = values[
        torch.as_tensor(absolute, dtype=torch.long)
    ]
    contribution = torch.einsum("k,kd->d", row[relative], source)
    return -contribution, {
        "edge_count": float(len(absolute)),
        "edge_attention_mass": float(row[relative].sum()),
        "edge_contribution_norm": float(torch.linalg.vector_norm(contribution)),
    }


def _ordinary_context_windows(
    encoding: PromptEncoding,
    *,
    width: int,
    additionally_forbidden: Iterable[int] = (),
) -> tuple[tuple[int, ...], ...]:
    forbidden = registered_forbidden_positions(encoding) | {
        int(value) for value in additionally_forbidden
    }
    used = set(forbidden)
    windows: list[tuple[int, ...]] = []
    for span in encoding.needle_spans:
        found: tuple[int, ...] | None = None
        for gap in range(1, 512):
            end = int(span.start) - gap
            start = end - int(width)
            candidate = tuple(range(start, end))
            if start >= 1 and not set(candidate).intersection(used):
                found = candidate
                break
        if found is None:
            raise ValueError("Could not register disjoint ordinary context windows")
        used.update(found)
        windows.append(found)
    return tuple(windows)


def matched_position_carriers(
    encoding: PromptEncoding,
) -> tuple[tuple[int, ...], ...]:
    """Find monotone, disjoint ordinary carriers matching each span length."""

    forbidden = registered_forbidden_positions(encoding)
    used = set(forbidden)
    carriers: list[tuple[int, ...]] = []
    previous_end = 0
    for span in encoding.needle_spans:
        length = int(span.end) - int(span.start)
        found: tuple[int, ...] | None = None
        for gap in range(8, 1024):
            end = int(span.start) - gap
            start = end - length
            candidate = tuple(range(start, end))
            if (
                start >= max(1, previous_end)
                and not set(candidate).intersection(used)
            ):
                found = candidate
                break
        if found is None:
            raise ValueError("Could not register monotone position carriers")
        used.update(found)
        carriers.append(found)
        previous_end = found[-1] + 1
    return tuple(carriers)


def derive_factorial_encoding(
    encoding: PromptEncoding,
    *,
    identity_replacements: Sequence[Sequence[int]],
    identity: bool,
    context: bool,
    position: bool,
    context_width: int = 8,
) -> tuple[PromptEncoding, dict[str, object]]:
    """Create a length-preserving token-level 2x2x2 factorial encoding.

    Identity replaces each record by a tokenizer-length-matched record from a
    different canonical seed.  Context cyclically permutes equal-width
    ordinary windows.  Position swaps each record with a disjoint ordinary
    carrier of exactly the same length and updates the registered span sites.
    """

    spans = tuple(encoding.needle_spans)
    if len(identity_replacements) != len(spans):
        raise ValueError("Identity replacements must cover every active span")
    carriers = matched_position_carriers(encoding)
    carrier_positions = {position for carrier in carriers for position in carrier}
    windows = _ordinary_context_windows(
        encoding,
        width=int(context_width),
        additionally_forbidden=carrier_positions,
    )
    ids = list(int(value) for value in encoding.input_ids)
    if identity:
        for span, replacement_ids in zip(spans, identity_replacements):
            replacement_ids = tuple(int(value) for value in replacement_ids)
            if len(replacement_ids) != int(span.end) - int(span.start):
                raise ValueError("An identity replacement is not token-length matched")
            ids[int(span.start) : int(span.end)] = replacement_ids
    if context:
        payloads = [tuple(ids[position] for position in window) for window in windows]
        for index, window in enumerate(windows):
            source = payloads[(index - 1) % len(payloads)]
            for position_index, token in zip(window, source):
                ids[position_index] = token
    new_spans = spans
    if position:
        record_payloads = [
            tuple(ids[position] for position in range(int(span.start), int(span.end)))
            for span in spans
        ]
        carrier_payloads = [tuple(ids[position] for position in carrier) for carrier in carriers]
        for span, carrier, record_values, ordinary_values in zip(
            spans, carriers, record_payloads, carrier_payloads
        ):
            if len(record_values) != len(carrier):
                raise RuntimeError("Position carrier length changed unexpectedly")
            ids[int(span.start) : int(span.end)] = ordinary_values
            for carrier_position, token in zip(carrier, record_values):
                ids[carrier_position] = token
        new_spans = tuple(
            TokenSpan(
                slot_index=int(span.slot_index),
                start=int(carrier[0]),
                end=int(carrier[-1]) + 1,
                active=True,
                kind=str(span.kind),
                canonical_length=int(span.canonical_length),
                model_token_length=len(carrier),
            )
            for span, carrier in zip(spans, carriers)
        )
    if len(ids) != encoding.sequence_length or ids[int(encoding.query_position)] != int(
        encoding.input_ids[int(encoding.query_position)]
    ):
        raise RuntimeError("A factorial manipulation changed length or answer query")
    # N=10 is required, so every registered slot is active and can be replaced
    # by its position-adjusted counterpart without retaining inactive slots.
    if len(encoding.slot_spans) != len(spans):
        raise ValueError("Factorial encoding is registered only for final N=10 prompts")
    derived = replace(
        encoding,
        input_ids=tuple(ids),
        slot_spans=tuple(new_spans),
        needle_spans=tuple(new_spans),
    )
    audit = {
        "identity": bool(identity),
        "context": bool(context),
        "position": bool(position),
        "sequence_length": int(derived.sequence_length),
        "query_position": int(derived.query_position),
        "context_width": int(context_width),
        "span_lengths": [int(span.model_token_length) for span in new_spans],
        "carrier_starts": [int(carrier[0]) for carrier in carriers],
        "context_window_starts": [int(window[0]) for window in windows],
    }
    return derived, audit
