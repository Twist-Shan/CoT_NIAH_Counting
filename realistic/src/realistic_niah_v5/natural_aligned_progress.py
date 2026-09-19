"""Pure geometry helpers for natural, position-aligned progress transplants."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence


SITE_POLICIES = (
    "latest_structural",
    "period_preferred",
    "non_whitespace_structural",
)

PATCH_SCOPES = (
    "fixed_suffix",
    "item_span",
)


def resolve_natural_patch_span(
    trace_items: Sequence[Sequence[int]],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    receiver_site: int,
    donor_site: int,
    patch_scope: str = "fixed_suffix",
    patch_width: int = 1,
) -> dict[str, Any]:
    """Resolve an endpoint-aligned causal patch span without using outcomes.

    ``fixed_suffix`` reproduces the original width-W tail intervention.
    ``item_span`` uses the largest endpoint-aligned span that stays inside both
    registered items.  When the two tokenized items have equal width this is a
    complete-item transplant; otherwise it is the full shorter item and the
    equally wide suffix of the longer item.  The latter is deliberately
    audited rather than resampling hidden states across unequal token grids.
    """

    items = tuple((int(value[0]), int(value[1])) for value in trace_items)
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    scope = str(patch_scope)
    requested_width = int(patch_width)
    if scope not in PATCH_SCOPES:
        raise ValueError(f"Unknown natural patch scope: {scope}")
    if not (
        1 <= receiver <= len(items)
        and 1 <= donor <= len(items)
        and receiver != donor
    ):
        raise ValueError("Natural patch spans require distinct valid items")
    if requested_width <= 0:
        raise ValueError("Patch width must be positive")

    receiver_start, receiver_end = items[receiver - 1]
    donor_start, donor_end = items[donor - 1]
    receiver_site = int(receiver_site)
    donor_site = int(donor_site)
    if not receiver_start <= receiver_site < receiver_end:
        raise ValueError("Receiver patch site falls outside its item")
    if not donor_start <= donor_site < donor_end:
        raise ValueError("Donor patch site falls outside its item")

    receiver_available = receiver_site - receiver_start + 1
    donor_available = donor_site - donor_start + 1
    if scope == "fixed_suffix":
        effective_width = requested_width
    else:
        if receiver_site != receiver_end - 1 or donor_site != donor_end - 1:
            raise ValueError("item_span requires the registered item endpoint")
        effective_width = min(receiver_available, donor_available)
    if effective_width > receiver_available or effective_width > donor_available:
        raise ValueError("Patch span crosses a natural item boundary")

    receiver_item_width = receiver_end - receiver_start
    donor_item_width = donor_end - donor_start
    return {
        "patch_scope": scope,
        "requested_patch_width": requested_width,
        "effective_patch_width": effective_width,
        "receiver_patch_start": receiver_site - effective_width + 1,
        "receiver_patch_end": receiver_site + 1,
        "donor_patch_start": donor_site - effective_width + 1,
        "donor_patch_end": donor_site + 1,
        "receiver_item_token_width": receiver_item_width,
        "donor_item_token_width": donor_item_width,
        "receiver_item_coverage": effective_width / receiver_item_width,
        "donor_item_coverage": effective_width / donor_item_width,
        "equal_length_complete_item": bool(
            scope == "item_span"
            and receiver_item_width == donor_item_width == effective_width
        ),
        "endpoint_aligned": True,
        "hidden_state_resampling": False,
    }


def matched_post_item_site_candidates(
    encoding: Any,
    trace_items: Sequence[Sequence[int]],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    tokenizer: Any,
    tail_window: int = 4,
) -> list[dict[str, Any]]:
    """Enumerate same-token site pairs inside two natural item tails.

    The returned table is deliberately outcome-blind.  It exposes the token
    identity and relative location needed to freeze a grammar-level site rule
    before any causal transplant is scored.
    """

    items = tuple((int(value[0]), int(value[1])) for value in trace_items)
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    window = int(tail_window)
    if not (
        1 <= receiver < len(items)
        and 1 <= donor < len(items)
        and receiver != donor
    ):
        raise ValueError("Natural aligned sites require distinct 1 <= j,k < N")
    if window <= 0:
        raise ValueError("tail_window must be positive")

    def positions(occurrence: int) -> list[int]:
        start, end = items[occurrence - 1]
        left = max(start, end - window)
        # Stay inside the registered natural item.  The gap before the next
        # item can contain recap/lead-in prose (for example ``Then later:``),
        # whose repeated whitespace is surface-matched but is not a progress
        # commit site.
        return list(range(left, end))

    receiver_start, receiver_end = items[receiver - 1]
    donor_start, donor_end = items[donor - 1]
    del receiver_start, donor_start
    candidates: list[dict[str, Any]] = []
    for left in positions(receiver):
        for right in positions(donor):
            token_id = int(encoding.input_ids[left])
            if token_id != int(encoding.input_ids[right]):
                continue
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            structural = not any(character.isalnum() for character in text)
            whitespace = not text or text.isspace()
            receiver_offset = receiver_end - 1 - left
            donor_offset = donor_end - 1 - right
            candidates.append(
                {
                    "receiver_site": left,
                    "donor_site_before_alignment": right,
                    "receiver_tail_offset": receiver_offset,
                    "donor_tail_offset": donor_offset,
                    "tail_offsets_match": receiver_offset == donor_offset,
                    "tail_offset_distance": abs(receiver_offset - donor_offset),
                    "shared_commit_token_id": token_id,
                    "shared_commit_token_text": text,
                    "is_structural": structural,
                    "is_whitespace": whitespace,
                    "is_exact_period": text == ".",
                    "is_period_like": "." in text and structural,
                }
            )
    return candidates


def post_item_sites_at_tail_offset(
    encoding: Any,
    trace_items: Sequence[Sequence[int]],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    tokenizer: Any,
    tail_offset: int,
) -> tuple[int, int, dict[str, Any]]:
    """Choose the same relative item-tail position without requiring token match."""

    items = tuple((int(value[0]), int(value[1])) for value in trace_items)
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    offset = int(tail_offset)
    if not (
        1 <= receiver < len(items)
        and 1 <= donor < len(items)
        and receiver != donor
    ):
        raise ValueError("Natural aligned sites require distinct 1 <= j,k < N")
    if offset < 0:
        raise ValueError("tail_offset must be nonnegative")
    receiver_start, receiver_end = items[receiver - 1]
    donor_start, donor_end = items[donor - 1]
    receiver_site = receiver_end - 1 - offset
    donor_site = donor_end - 1 - offset
    if receiver_site < receiver_start or donor_site < donor_start:
        raise ValueError("Requested tail offset falls outside a natural item")
    receiver_token_id = int(encoding.input_ids[receiver_site])
    donor_token_id = int(encoding.input_ids[donor_site])

    def decode(token_id: int) -> str:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    return receiver_site, donor_site, {
        "receiver_site": receiver_site,
        "donor_site_before_alignment": donor_site,
        "alignment_token_delta": donor_site - receiver_site,
        "site_policy": "fixed_tail_offset",
        "tail_offset": offset,
        "receiver_tail_offset": offset,
        "donor_tail_offset": offset,
        "tail_offsets_match": True,
        "receiver_commit_token_id": receiver_token_id,
        "donor_commit_token_id": donor_token_id,
        "receiver_commit_token_text": decode(receiver_token_id),
        "donor_commit_token_text": decode(donor_token_id),
        "surface_token_matched": receiver_token_id == donor_token_id,
    }


def matched_post_item_sites(
    encoding: Any,
    trace_items: Sequence[Sequence[int]],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    tokenizer: Any,
    tail_window: int = 4,
    site_policy: str = "latest_structural",
) -> tuple[int, int, dict[str, Any]]:
    """Choose same-token commit sites after two natural trace items."""
    policy = str(site_policy)
    if policy not in SITE_POLICIES:
        raise ValueError(f"Unknown natural aligned site policy: {policy}")
    candidates = matched_post_item_site_candidates(
        encoding,
        trace_items,
        receiver_occurrence=receiver_occurrence,
        donor_occurrence=donor_occurrence,
        tokenizer=tokenizer,
        tail_window=tail_window,
    )
    if not candidates:
        raise ValueError("Natural steps expose no same-token post-item commit site")

    def score(candidate: dict[str, Any]) -> tuple[int, ...]:
        left = int(candidate["receiver_site"])
        right = int(candidate["donor_site_before_alignment"])
        structural = int(bool(candidate["is_structural"]))
        non_whitespace_structural = int(
            bool(candidate["is_structural"]) and not bool(candidate["is_whitespace"])
        )
        relative_match = int(bool(candidate["tail_offsets_match"]))
        relative_distance = -int(candidate["tail_offset_distance"])
        if policy == "latest_structural":
            # Preserve the original frozen assay exactly.
            return structural, left, right
        if policy == "period_preferred":
            return (
                int(bool(candidate["is_exact_period"])),
                int(bool(candidate["is_period_like"])),
                non_whitespace_structural,
                relative_match,
                relative_distance,
                left,
                right,
            )
        return (
            non_whitespace_structural,
            structural,
            relative_match,
            relative_distance,
            left,
            right,
        )

    selected = max(candidates, key=score)
    receiver_site = int(selected["receiver_site"])
    donor_site = int(selected["donor_site_before_alignment"])
    token_id = int(selected["shared_commit_token_id"])
    return receiver_site, donor_site, {
        "alignment_token_delta": donor_site - receiver_site,
        "site_policy": policy,
        "tail_window": int(tail_window),
        "matched_site_candidate_count": len(candidates),
        "receiver_commit_token_id": token_id,
        "donor_commit_token_id": token_id,
        "receiver_commit_token_text": str(selected["shared_commit_token_text"]),
        "donor_commit_token_text": str(selected["shared_commit_token_text"]),
        "surface_token_matched": True,
        **selected,
    }


def align_natural_donor_prompt(
    encoding: Any,
    registry: Any,
    *,
    receiver_site: int,
    donor_site: int,
    tokenizer: Any,
    require_surface_match: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Delete one non-record filler window so donor site equals receiver site."""

    delta = int(donor_site) - int(receiver_site)
    if delta <= 0:
        raise ValueError("Donor alignment requires a positive token delta")
    prompt_end = int(registry.prompt_token_count)
    if prompt_end <= delta + 192:
        raise ValueError("Prompt is too short for protected filler alignment")
    forbidden = {
        position
        for start, end in registry.prompt_records
        for position in range(int(start), int(end))
    }
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    deletion_start = None
    for start in range(64, prompt_end - 128 - delta + 1):
        window = range(start, start + delta)
        if any(position in forbidden for position in window):
            continue
        if any(int(encoding.input_ids[position]) in special for position in window):
            continue
        deletion_start = start
        break
    if deletion_start is None:
        raise ValueError("No protected non-record filler window can align the donor")
    deletion_end = deletion_start + delta
    ids = (
        tuple(int(value) for value in encoding.input_ids[:deletion_start])
        + tuple(int(value) for value in encoding.input_ids[deletion_end:])
    )
    mask = (
        tuple(int(value) for value in encoding.attention_mask[:deletion_start])
        + tuple(int(value) for value in encoding.attention_mask[deletion_end:])
    )

    def shift_span(span: Any) -> Any:
        start = int(span.start)
        end = int(span.end)
        if end <= deletion_start:
            return span
        if start < deletion_end:
            raise RuntimeError("Alignment deletion unexpectedly overlaps a span")
        return replace(span, start=start - delta, end=end - delta)

    donor = replace(
        encoding,
        input_ids=ids,
        attention_mask=mask,
        query_position=int(encoding.query_position) - delta,
        prompt_token_count=int(encoding.prompt_token_count) - delta,
        prompt_record_spans=tuple(
            shift_span(span) for span in encoding.prompt_record_spans
        ),
        trace_item_spans=tuple(shift_span(span) for span in encoding.trace_item_spans),
        slot_spans=tuple(shift_span(span) for span in encoding.slot_spans),
        needle_spans=tuple(shift_span(span) for span in encoding.needle_spans),
        hard_negative_spans=tuple(
            shift_span(span) for span in encoding.hard_negative_spans
        ),
    )
    aligned_site = int(donor_site) - delta
    if aligned_site != int(receiver_site):
        raise RuntimeError("Natural donor alignment missed the receiver site")
    surface_token_matched = (
        int(donor.input_ids[aligned_site]) == int(encoding.input_ids[receiver_site])
    )
    if bool(require_surface_match) and not surface_token_matched:
        raise RuntimeError("Natural donor alignment changed the commit surface token")
    return donor, {
        "deletion_start": deletion_start,
        "deletion_end": deletion_end,
        "deleted_token_count": delta,
        "deletion_before_answer": deletion_end <= prompt_end,
        "deletion_avoids_prompt_records": True,
        "deletion_avoids_special_tokens": True,
        "aligned_donor_site": aligned_site,
        "alignment_surface_match_required": bool(require_surface_match),
        "alignment_surface_token_matched": surface_token_matched,
    }
