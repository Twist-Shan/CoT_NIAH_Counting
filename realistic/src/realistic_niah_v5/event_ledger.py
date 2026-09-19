"""Equal-geometry multi-event factorials for testing marker-keyed ledgers."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Mapping, Sequence


def _splice_encoding(
    encoding: Any,
    *,
    position: int,
    replacement_ids: Sequence[int],
    replacement_mask: Sequence[int],
) -> Any:
    left = int(position)
    ids = tuple(int(value) for value in encoding.input_ids)
    mask = tuple(int(value) for value in encoding.attention_mask)
    inserted_ids = tuple(int(value) for value in replacement_ids)
    inserted_mask = tuple(int(value) for value in replacement_mask)
    if not int(encoding.prompt_token_count) <= left <= int(encoding.query_position):
        raise ValueError("Multi-event insertion lies outside the native trace")
    if len(inserted_ids) != len(inserted_mask) or not inserted_ids:
        raise ValueError("Multi-event replacement ids/mask are invalid")
    delta = len(inserted_ids)
    return replace(
        encoding,
        input_ids=ids[:left] + inserted_ids + ids[left:],
        attention_mask=mask[:left] + inserted_mask + mask[left:],
        query_position=int(encoding.query_position) + delta,
        trace_item_spans=(),
        slot_spans=(),
        needle_spans=(),
    )


def build_marker_event_factorial(
    encoding: Any,
    neutral_encoding: Any,
    registry: Any,
    boundaries: Mapping[int, int],
    *,
    receiver: int,
    source_occurrences: Sequence[int],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Insert distinct copied events whose markers form a full binary factorial.

    Every cell has identical payloads, separators, attention mask, sequence
    length, and absolute target geometry.  A zero bit replaces only the copied
    source marker token(s) with their same-position neutral counterparts.
    """

    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    markers = tuple((int(start), int(end)) for start, end in registry.trace_markers)
    active_receiver = int(receiver)
    physical_target = active_receiver + 1
    sources = tuple(int(value) for value in source_occurrences)
    if not 1 <= len(sources) <= 4 or len(set(sources)) != len(sources):
        raise ValueError("Ledger factorial requires one to four distinct sources")
    if not 1 <= active_receiver < physical_target <= len(items):
        raise ValueError("Ledger receiver/target geometry is invalid")
    if any(not 1 <= source < len(items) for source in sources):
        raise ValueError("A ledger source lacks a following separator")

    ids = tuple(int(value) for value in encoding.input_ids)
    mask = tuple(int(value) for value in encoding.attention_mask)
    neutral_ids = tuple(int(value) for value in neutral_encoding.input_ids)
    if (
        len(neutral_ids) != len(ids)
        or tuple(int(value) for value in neutral_encoding.attention_mask) != mask
    ):
        raise ValueError("Neutral encoding does not preserve native geometry")

    source_specs: list[dict[str, Any]] = []
    for slot, source in enumerate(sources):
        start, item_end = items[source - 1]
        segment_end = items[source][0]
        if not start < item_end <= segment_end:
            raise ValueError(f"Source occurrence {source} has invalid item geometry")
        marker_positions = tuple(
            position
            for marker_start, marker_end in markers
            for position in range(marker_start, marker_end)
            if start <= position < item_end
        )
        marker_local = tuple(position - start for position in marker_positions)
        payload_local = tuple(
            position
            for position in range(item_end - start)
            if position not in set(marker_local)
        )
        if not marker_local or not payload_local:
            raise ValueError(
                f"Source occurrence {source} lacks marker or payload tokens"
            )
        valid_segment = tuple(ids[start:segment_end])
        markerless_segment = list(valid_segment)
        for local in marker_local:
            markerless_segment[local] = neutral_ids[start + local]
        boundary_local = int(boundaries[source]) - start
        if not 0 <= boundary_local < len(valid_segment):
            raise ValueError(f"Source occurrence {source} boundary leaves its segment")
        if any(
            valid_segment[local] == markerless_segment[local]
            for local in marker_local
        ):
            raise ValueError("A valid/markerless source marker is token-identical")
        source_specs.append(
            {
                "slot": slot,
                "source_occurrence": source,
                "source_start": start,
                "source_item_end": item_end,
                "source_segment_end": segment_end,
                "segment_length": len(valid_segment),
                "valid_segment": valid_segment,
                "markerless_segment": tuple(markerless_segment),
                "segment_mask": tuple(mask[start:segment_end]),
                "marker_local_positions": marker_local,
                "payload_local_positions": payload_local,
                "boundary_local_position": boundary_local,
            }
        )

    insertion_start = int(items[physical_target - 1][0])
    total_delta = sum(int(spec["segment_length"]) for spec in source_specs)
    event_end = insertion_start + total_delta
    target_start_original, target_item_end_original = items[physical_target - 1]
    target_marker_original = tuple(
        position
        for marker_start, marker_end in markers
        for position in range(marker_start, marker_end)
        if target_start_original <= position < target_item_end_original
    )
    if not target_marker_original:
        raise ValueError("The original target item lacks marker tokens")
    target_marker_positions = tuple(
        position + total_delta for position in target_marker_original
    )
    target_boundary = int(boundaries[physical_target]) + total_delta
    if event_end != target_start_original + total_delta:
        raise RuntimeError("Inserted ledger events do not meet the target item")

    slots: list[dict[str, Any]] = []
    offset = 0
    for spec in source_specs:
        absolute_start = insertion_start + offset
        slots.append(
            {
                "slot": int(spec["slot"]),
                "source_occurrence": int(spec["source_occurrence"]),
                "start": absolute_start,
                "end": absolute_start + int(spec["segment_length"]),
                "marker_positions": [
                    absolute_start + int(local)
                    for local in spec["marker_local_positions"]
                ],
                "event_boundary": absolute_start
                + int(spec["boundary_local_position"]),
            }
        )
        offset += int(spec["segment_length"])
    marker_positions = tuple(
        position for slot in slots for position in slot["marker_positions"]
    )
    if len(set(marker_positions)) != len(marker_positions):
        raise RuntimeError("Inserted ledger marker positions overlap")

    variants: list[dict[str, Any]] = []
    for bits_raw in product((0, 1), repeat=len(source_specs)):
        bits = tuple(int(value) for value in bits_raw)
        replacement_ids: list[int] = []
        replacement_mask: list[int] = []
        for bit, spec in zip(bits, source_specs):
            replacement_ids.extend(
                spec["valid_segment"] if bit else spec["markerless_segment"]
            )
            replacement_mask.extend(spec["segment_mask"])
        active = _splice_encoding(
            encoding,
            position=insertion_start,
            replacement_ids=replacement_ids,
            replacement_mask=replacement_mask,
        )
        variants.append(
            {
                "variant_id": "markers_" + "".join(str(value) for value in bits),
                "marker_bits": bits,
                "valid_marker_count": sum(bits),
                "event_count_target": physical_target + sum(bits),
                "encoding": active,
                "insertion_start": insertion_start,
                "event_end": event_end,
                "target_marker_position": max(target_marker_positions),
                "target_boundary": target_boundary,
            }
        )

    baseline_ids = tuple(int(value) for value in variants[0]["encoding"].input_ids)
    union = set(marker_positions)
    for variant in variants:
        active_encoding = variant["encoding"]
        if (
            int(active_encoding.sequence_length)
            != int(variants[0]["encoding"].sequence_length)
            or tuple(int(value) for value in active_encoding.attention_mask)
            != tuple(int(value) for value in variants[0]["encoding"].attention_mask)
        ):
            raise RuntimeError("Ledger factorial cells changed sequence geometry")
        active_ids = tuple(int(value) for value in active_encoding.input_ids)
        changed = {
            position
            for position, (left, right) in enumerate(zip(baseline_ids, active_ids))
            if left != right
        }
        expected = {
            position
            for bit, slot in zip(variant["marker_bits"], slots)
            if bit
            for position in slot["marker_positions"]
        }
        if changed != expected or not changed.issubset(union):
            raise RuntimeError("A ledger factorial cell changed non-marker tokens")

    geometry = {
        "receiver": active_receiver,
        "physical_target": physical_target,
        "source_occurrences": list(sources),
        "factor_count": len(sources),
        "insertion_start": insertion_start,
        "event_end": event_end,
        "total_token_delta": total_delta,
        "inserted_slots": slots,
        "inserted_marker_positions": list(marker_positions),
        "target_marker_positions": list(target_marker_positions),
        "target_marker_position": max(target_marker_positions),
        "target_boundary": target_boundary,
        "factorial_cell_count": len(variants),
        "all_cells_equal_length": True,
        "all_cells_equal_attention_mask": True,
        "only_marker_token_ids_vary": True,
    }
    return tuple(variants), geometry


def build_semantic_event_factorial(
    encoding: Any,
    neutral_encoding: Any,
    registry: Any,
    boundaries: Mapping[int, int],
    *,
    receiver: int,
    source_occurrences: Sequence[int],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Insert equal-length events whose semantic validity forms a factorial.

    This is the no-marker analogue of :func:`build_marker_event_factorial`.
    Each valid slot copies a native city-score event.  Its invalid counterpart
    keeps punctuation, whitespace, separator, closing token, attention mask,
    and length fixed while replacing only the event's alphanumeric token ids
    by same-position neutral ids supplied in ``neutral_encoding``.
    """

    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    active_receiver = int(receiver)
    physical_target = active_receiver + 1
    sources = tuple(int(value) for value in source_occurrences)
    if not 1 <= len(sources) <= 4 or len(set(sources)) != len(sources):
        raise ValueError("Semantic factorial requires one to four distinct sources")
    if not 1 <= active_receiver < physical_target <= len(items):
        raise ValueError("Semantic-factorial receiver/target geometry is invalid")
    if any(not 1 <= source < len(items) for source in sources):
        raise ValueError("A semantic-factorial source lacks a following separator")

    ids = tuple(int(value) for value in encoding.input_ids)
    mask = tuple(int(value) for value in encoding.attention_mask)
    neutral_ids = tuple(int(value) for value in neutral_encoding.input_ids)
    if (
        len(neutral_ids) != len(ids)
        or tuple(int(value) for value in neutral_encoding.attention_mask) != mask
    ):
        raise ValueError("Semantic neutral encoding does not preserve geometry")

    source_specs: list[dict[str, Any]] = []
    for slot, source in enumerate(sources):
        start, item_end = items[source - 1]
        segment_end = items[source][0]
        if not start < item_end <= segment_end:
            raise ValueError(f"Source occurrence {source} has invalid item geometry")
        valid_segment = tuple(ids[start:segment_end])
        invalid_segment = tuple(neutral_ids[start:segment_end])
        discriminative_local = tuple(
            index
            for index, (left, right) in enumerate(
                zip(valid_segment, invalid_segment)
            )
            if int(left) != int(right)
        )
        if not discriminative_local:
            raise ValueError(
                f"Source occurrence {source} has no semantic validity tokens"
            )
        if any(index >= item_end - start for index in discriminative_local):
            raise ValueError("Semantic neutralization changed a separator token")
        boundary_local = int(boundaries[source]) - start
        if not 0 <= boundary_local < len(valid_segment):
            raise ValueError(f"Source occurrence {source} boundary leaves its segment")
        if valid_segment[boundary_local] != invalid_segment[boundary_local]:
            raise ValueError("Semantic neutralization changed the shared commit token")
        source_specs.append(
            {
                "slot": slot,
                "source_occurrence": source,
                "source_start": start,
                "source_item_end": item_end,
                "source_segment_end": segment_end,
                "segment_length": len(valid_segment),
                "valid_segment": valid_segment,
                "invalid_segment": invalid_segment,
                "segment_mask": tuple(mask[start:segment_end]),
                "discriminative_local_positions": discriminative_local,
                "boundary_local_position": boundary_local,
            }
        )

    insertion_start = int(items[physical_target - 1][0])
    total_delta = sum(int(spec["segment_length"]) for spec in source_specs)
    event_end = insertion_start + total_delta
    target_boundary = int(boundaries[physical_target]) + total_delta
    slots: list[dict[str, Any]] = []
    offset = 0
    for spec in source_specs:
        absolute_start = insertion_start + offset
        slots.append(
            {
                "slot": int(spec["slot"]),
                "source_occurrence": int(spec["source_occurrence"]),
                "start": absolute_start,
                "end": absolute_start + int(spec["segment_length"]),
                "discriminative_positions": [
                    absolute_start + int(local)
                    for local in spec["discriminative_local_positions"]
                ],
                "event_boundary": absolute_start
                + int(spec["boundary_local_position"]),
            }
        )
        offset += int(spec["segment_length"])
    discriminative_positions = tuple(
        position
        for slot in slots
        for position in slot["discriminative_positions"]
    )
    if len(set(discriminative_positions)) != len(discriminative_positions):
        raise RuntimeError("Semantic-factorial discriminative positions overlap")

    variants: list[dict[str, Any]] = []
    for bits_raw in product((0, 1), repeat=len(source_specs)):
        bits = tuple(int(value) for value in bits_raw)
        replacement_ids: list[int] = []
        replacement_mask: list[int] = []
        for bit, spec in zip(bits, source_specs):
            replacement_ids.extend(
                spec["valid_segment"] if bit else spec["invalid_segment"]
            )
            replacement_mask.extend(spec["segment_mask"])
        active = _splice_encoding(
            encoding,
            position=insertion_start,
            replacement_ids=replacement_ids,
            replacement_mask=replacement_mask,
        )
        variants.append(
            {
                "variant_id": "events_" + "".join(str(value) for value in bits),
                "marker_bits": bits,
                "valid_marker_count": sum(bits),
                "valid_event_count": sum(bits),
                "event_count_at_inserted_commit": active_receiver + sum(bits),
                "encoding": active,
                "insertion_start": insertion_start,
                "event_end": event_end,
                "target_boundary": target_boundary,
            }
        )

    baseline_ids = tuple(int(value) for value in variants[0]["encoding"].input_ids)
    union = set(discriminative_positions)
    for variant in variants:
        active_encoding = variant["encoding"]
        if (
            int(active_encoding.sequence_length)
            != int(variants[0]["encoding"].sequence_length)
            or tuple(int(value) for value in active_encoding.attention_mask)
            != tuple(int(value) for value in variants[0]["encoding"].attention_mask)
        ):
            raise RuntimeError("Semantic-factorial cells changed sequence geometry")
        active_ids = tuple(int(value) for value in active_encoding.input_ids)
        changed = {
            position
            for position, (left, right) in enumerate(zip(baseline_ids, active_ids))
            if left != right
        }
        expected = {
            position
            for bit, slot in zip(variant["marker_bits"], slots)
            if bit
            for position in slot["discriminative_positions"]
        }
        if changed != expected or not changed.issubset(union):
            raise RuntimeError("Semantic-factorial cell changed an unregistered token")

    geometry = {
        "receiver": active_receiver,
        "physical_target": physical_target,
        "source_occurrences": list(sources),
        "factor_count": len(sources),
        "insertion_start": insertion_start,
        "event_end": event_end,
        "total_token_delta": total_delta,
        "inserted_slots": slots,
        "inserted_discriminative_positions": list(discriminative_positions),
        "target_boundary": target_boundary,
        "factorial_cell_count": len(variants),
        "all_cells_equal_length": True,
        "all_cells_equal_attention_mask": True,
        "only_event_semantic_token_ids_vary": True,
        "shared_commit_token_identical": True,
    }
    return tuple(variants), geometry
