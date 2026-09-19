from __future__ import annotations

from typing import Literal

from .tokenizer import VocabTokenizer

LossMask = Literal[
    "full_sequence",
    "full_sequence_final_weighted",
    "completion_only",
    "completion_final_weighted",
    "final_count_only",
]

LOSS_MASKS: set[str] = {
    "full_sequence",
    "full_sequence_final_weighted",
    "completion_only",
    "completion_final_weighted",
    "final_count_only",
}

SEGMENT_NAMES = [
    "source_loss",
    "think_boundary_loss",
    "trace_index_loss",
    "trace_marker_loss",
    "answer_prefix_loss",
    "count_loss",
    "eos_loss",
]
SEGMENT_TO_ID = {name: idx for idx, name in enumerate(SEGMENT_NAMES)}
ID_TO_SEGMENT = {idx: name for name, idx in SEGMENT_TO_ID.items()}


def supervised_indices_for_mask(
    example: dict,
    loss_mask: str,
    *,
    final_count_only_include_eos: bool = False,
) -> set[int]:
    if loss_mask not in LOSS_MASKS:
        raise ValueError(f"Unknown loss mask {loss_mask!r}. Expected one of {sorted(LOSS_MASKS)}")
    spans = example["spans"]
    if loss_mask in {"full_sequence", "full_sequence_final_weighted"}:
        return set(range(1, spans["eos_idx"] + 1))
    if loss_mask in {"completion_only", "completion_final_weighted"}:
        return set(range(spans["think_open_idx"], spans["eos_idx"] + 1))
    count_start = spans.get("count_start_idx", spans["count_idx"])
    count_end = spans.get("count_end_exclusive", spans["count_idx"] + 1)
    indices = set(range(count_start, count_end))
    if final_count_only_include_eos:
        indices.add(spans["eos_idx"])
    return indices


def build_labels_and_weights(
    example: dict,
    tokenizer: VocabTokenizer,
    *,
    loss_mask: str,
    final_weight: float = 10.0,
    eos_weight: float = 1.0,
    final_count_only_include_eos: bool = False,
) -> tuple[list[int], list[float]]:
    input_ids = tokenizer.encode(example["full_tokens"])
    labels = [-100] * len(input_ids)
    weights = [0.0] * len(input_ids)
    supervised = supervised_indices_for_mask(
        example,
        loss_mask,
        final_count_only_include_eos=final_count_only_include_eos,
    )
    for idx in supervised:
        labels[idx] = input_ids[idx]
        weights[idx] = 1.0

    spans = example["spans"]
    if loss_mask in {"full_sequence_final_weighted", "completion_final_weighted"}:
        count_start = spans.get("count_start_idx", spans["count_idx"])
        count_end = spans.get("count_end_exclusive", spans["count_idx"] + 1)
        for idx in range(count_start, count_end):
            weights[idx] = float(final_weight)
        if spans["eos_idx"] in supervised:
            weights[spans["eos_idx"]] = float(eos_weight)
    elif loss_mask == "final_count_only" and final_count_only_include_eos:
        weights[spans["eos_idx"]] = float(eos_weight)

    labels[0] = -100
    weights[0] = 0.0
    return labels, weights


def token_segment(example: dict, idx: int) -> str | None:
    spans = example["spans"]
    token = example["full_tokens"][idx]
    if spans["source_start"] <= idx < spans["source_end_exclusive"]:
        return "source_loss"
    think_indices = {value for value in (spans.get("think_open_idx"), spans.get("think_close_idx")) if value is not None}
    if idx in think_indices:
        return "think_boundary_loss"
    if spans["trace_start"] <= idx < spans["trace_end_exclusive"]:
        trace_index_indices = {
            pair["index_idx"]
            for pair in spans.get("trace_pairs", [])
            if pair.get("index_idx") is not None
        }
        return "trace_index_loss" if idx in trace_index_indices or token.startswith("<I") else "trace_marker_loss"
    if idx == spans["ans_idx"]:
        return "answer_prefix_loss"
    count_start = spans.get("count_start_idx", spans["count_idx"])
    count_end = spans.get("count_end_exclusive", spans["count_idx"] + 1)
    if count_start <= idx < count_end:
        return "count_loss"
    if idx == spans["eos_idx"]:
        return "eos_loss"
    return None


def segment_ids_for_example(example: dict) -> list[int]:
    ids = []
    for idx in range(len(example["full_tokens"])):
        segment = token_segment(example, idx)
        ids.append(SEGMENT_TO_ID[segment] if segment is not None else -1)
    return ids
