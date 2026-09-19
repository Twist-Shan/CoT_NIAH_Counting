from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
import torch
import torch.nn.functional as F

from dataset_generation.token_match_utils import find_subsequence_positions
from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    generate_dynamic_niah_dataset_v2,
)

DEFAULT_OUTPUT_DIR = "analysis/hidden_comp"
MAX_STORED_PT_BYTES = 200 * 1024 * 1024


def prune_large_pt_files(
    paths: list[str | Path] | str | Path,
    *,
    max_bytes: int = MAX_STORED_PT_BYTES,
    delete: bool = False,
) -> list[Path]:
    """Report oversized .pt files, optionally delete them, and return affected paths."""
    if isinstance(paths, (str, Path)):
        candidates = sorted(Path(paths).glob("**/*.pt"))
    else:
        candidates = [Path(path) for path in paths]

    deleted: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.suffix != ".pt" or path in seen or not path.exists():
            continue
        seen.add(path)
        size = path.stat().st_size
        if size <= max_bytes:
            continue
        size_mb = size / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        if delete:
            path.unlink()
            action = "deleted .pt file"
        else:
            action = "kept .pt file; set DELETE_LARGE_FILE_WHEN_DONE=True in the notebook to delete after analysis"
        deleted.append(path)
        print(
            f"[hidden-analysis] {path} is larger than {max_mb:.0f} MB ({size_mb:.1f} MB); {action}"
        )
    return deleted


def generate_control_dataset_with_logging(
    cfg: DynamicNiahV2Config,
) -> list[dict[str, Any]]:
    control_count = sum(bool(x) for x in (cfg.control_switch or []))
    if control_count != 1:
        raise ValueError(
            "Expected exactly one True in control_switch for hidden-state analysis"
        )
    rows = generate_dynamic_niah_dataset_v2(cfg)
    if cfg.save_data:
        if cfg.data_save_path is None:
            raise ValueError(
                "data_save_path must be resolved before saving hidden-analysis generated data"
            )
        path = write_dataset_jsonl(rows, cfg.data_save_path)
        print(f"[hidden-analysis] saved dataset jsonl to {path}")
    print(
        f"[hidden-analysis] generated dataset: examples={len(rows)} num_needles={cfg.num_needles} "
        f"control_switch={cfg.control_switch} output_dir={cfg.output_dir}"
    )
    return rows


def compute_alignment_offset(
    inputs: torch.Tensor,
    inputs_control: torch.Tensor,
    insertion_position: int,
    max_search_offset: int = 128,
) -> int:
    normal = inputs[0]
    control = inputs_control[0]
    if normal.shape[0] == control.shape[0] and torch.equal(
        normal[insertion_position:], control[insertion_position:]
    ):
        return 0

    best_offset = 0
    best_match = -1
    for offset in range(-max_search_offset, max_search_offset + 1):
        n_start = insertion_position
        c_start = insertion_position + offset
        if c_start < 0:
            continue
        overlap = min(normal.shape[0] - n_start, control.shape[0] - c_start)
        if overlap <= 0:
            continue
        matches = int(
            (
                normal[n_start : n_start + overlap]
                == control[c_start : c_start + overlap]
            )
            .sum()
            .item()
        )
        if matches > best_match:
            best_match = matches
            best_offset = offset

    print(
        "[hidden-analysis] token-length mismatch/alignment ",
        f"normal_len={normal.shape[0]} control_len={control.shape[0]} insertion_position={insertion_position} offset={best_offset}",
    )
    return best_offset


def compare_hidden_states(
    hidden: torch.Tensor,
    hidden_control: torch.Tensor,
    insertion_position: int,
    offset: int,
    layer_indices: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Compare hidden states on the full final-input axis.

    The x-axis of returned metrics is the uncontrolled model-input position
    ``0..T-1``. Controlled positions are aligned by applying ``offset`` only at
    and after ``insertion_position``; invalid aligned positions are represented
    as NaN so plots and tables still preserve full-input coordinates.
    """

    if layer_indices is None:
        layer_indices = list(range(hidden.shape[0]))

    sequence_length = int(hidden.shape[1])
    valid_positions, valid_control_positions = valid_aligned_positions_for_input(
        sequence_length,
        int(hidden_control.shape[1]),
        insertion_position=insertion_position,
        offset=offset,
    )

    rel_norm_rows = []
    cos_rows = []
    for layer_idx in layer_indices:
        h = hidden[layer_idx].detach().to(device="cpu", dtype=torch.float32)
        hc = hidden_control[layer_idx].detach().to(device="cpu", dtype=torch.float32)
        rel_norm = torch.full((sequence_length,), torch.nan, dtype=torch.float32)
        cossim = torch.full((sequence_length,), torch.nan, dtype=torch.float32)
        if valid_positions.numel() > 0:
            h_seg = h[valid_positions]
            hc_seg = hc[valid_control_positions]
            delta = h_seg - hc_seg
            denom = torch.norm(h_seg, dim=-1).clamp_min(1e-8)
            rel_norm[valid_positions] = torch.norm(delta, dim=-1) / denom
            cossim[valid_positions] = F.cosine_similarity(h_seg, hc_seg, dim=-1)

        rel_norm_rows.append(rel_norm)
        cos_rows.append(cossim)

    return {
        "relative_norm_diff": torch.stack(rel_norm_rows, dim=0),
        "cosine_similarity": torch.stack(cos_rows, dim=0),
        "layers": torch.tensor(layer_indices, dtype=torch.int64),
        "positions": torch.arange(sequence_length, dtype=torch.int64),
        "control_positions": aligned_control_positions_for_input(
            sequence_length,
            int(hidden_control.shape[1]),
            insertion_position=insertion_position,
            offset=offset,
        ),
        "insertion_position": torch.tensor(insertion_position, dtype=torch.int64),
        "offset": torch.tensor(offset, dtype=torch.int64),
    }


def _aligned_post_insertion_segments(
    hidden_layer: torch.Tensor,
    hidden_control_layer: torch.Tensor,
    insertion_position: int,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c_start = insertion_position + offset
    if c_start < 0:
        raise ValueError(
            f"Control start position is negative after applying offset: {c_start}"
        )

    overlap = min(
        hidden_layer.shape[0] - insertion_position,
        hidden_control_layer.shape[0] - c_start,
    )
    if overlap <= 0:
        raise ValueError("No overlap after applying offset")

    normal_segment = hidden_layer[insertion_position : insertion_position + overlap]
    control_segment = hidden_control_layer[c_start : c_start + overlap]
    positions = torch.arange(
        insertion_position, insertion_position + overlap, dtype=torch.int64
    )
    return normal_segment, control_segment, positions


def aligned_control_positions_for_input(
    sequence_length: int,
    control_sequence_length: int,
    *,
    insertion_position: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """Map uncontrolled final-input positions to aligned controlled positions.

    Returned values are controlled-input positions for each uncontrolled position
    ``0..sequence_length-1``. Positions at or after ``insertion_position`` use
    ``offset``; invalid controlled positions are marked with ``-1``.
    """

    if sequence_length < 0 or control_sequence_length < 0:
        raise ValueError("sequence lengths must be non-negative")
    positions = torch.arange(int(sequence_length), dtype=torch.long)
    control_positions = positions.clone()
    control_positions[positions >= int(insertion_position)] += int(offset)
    invalid = (control_positions < 0) | (
        control_positions >= int(control_sequence_length)
    )
    control_positions[invalid] = -1
    return control_positions


def valid_aligned_positions_for_input(
    sequence_length: int,
    control_sequence_length: int,
    *,
    insertion_position: int = 0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid uncontrolled positions and their controlled counterparts."""

    control_positions = aligned_control_positions_for_input(
        sequence_length,
        control_sequence_length,
        insertion_position=insertion_position,
        offset=offset,
    )
    positions = torch.arange(int(sequence_length), dtype=torch.long)
    valid = control_positions >= 0
    return positions[valid], control_positions[valid]


def _flat_token_ids(tokens: torch.Tensor | list[int]) -> list[int]:
    if isinstance(tokens, torch.Tensor):
        return [int(x) for x in tokens.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in tokens]


def _flat_token_offsets(
    token_offsets: torch.Tensor | list[Any] | None,
) -> list[tuple[int, int]] | None:
    if token_offsets is None:
        return None
    if isinstance(token_offsets, torch.Tensor):
        values = token_offsets.detach().cpu().reshape(-1, 2).tolist()
    else:
        values = token_offsets
        if (
            values
            and isinstance(values[0], list)
            and values[0]
            and isinstance(values[0][0], (list, tuple))
        ):
            values = values[0]
    return [(int(start), int(end)) for start, end in values]


def _text_occurrence_spans(text: str, pattern: str) -> list[tuple[int, int]]:
    if not pattern:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx < 0:
            return spans
        spans.append((idx, idx + len(pattern)))
        start = idx + 1


def _token_span_for_char_span(
    token_offsets: list[tuple[int, int]], char_span: tuple[int, int]
) -> tuple[int, int] | None:
    char_start, char_end = char_span
    token_indices = [
        idx
        for idx, (tok_start, tok_end) in enumerate(token_offsets)
        if tok_end > tok_start and tok_end > char_start and tok_start < char_end
    ]
    if not token_indices:
        return None
    return (token_indices[0], token_indices[-1] + 1)


def build_uncontrolled_needle_insertions(
    realized_insertions: list[dict[str, Any]], needles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return needle spans whose positions are based on the uncontrolled input.

    Generated rows store ``realized_insertions`` for the primary context, which may
    contain a control replacement with a different token length than the original
    needle. Hidden-state plots use the all-needle/uncontrolled prompt as their
    x-axis, so span metadata must be rebuilt with original needle token lengths.
    """

    needle_by_id = {str(needle.get("needle_id")): needle for needle in needles}
    ordered_insertions = sorted(
        realized_insertions,
        key=lambda item: int(
            item.get("requested_position", item.get("final_position", 0))
        ),
    )

    restored_insertions: list[dict[str, Any]] = []
    shift = 0
    for insertion in ordered_insertions:
        needle_id = str(insertion.get("needle_id"))
        needle = needle_by_id.get(needle_id, {})
        requested_position = int(
            insertion.get("requested_position", insertion.get("final_position", 0))
        )
        tokens = [int(x) for x in needle.get("tokens", insertion.get("tokens", []))]
        restored = dict(insertion)
        restored.update(
            {
                "requested_position": requested_position,
                "final_position": requested_position + shift,
                "token_length": len(tokens),
                "tokens": tokens,
                "decoded_text": needle.get(
                    "decoded_text", insertion.get("decoded_text")
                ),
                "is_control": bool(needle.get("is_control", False)),
                "inserted_from": "needle",
                "control": needle.get("control", insertion.get("control")),
            }
        )
        restored_insertions.append(restored)
        shift += len(tokens)

    return restored_insertions


def build_prompt_needle_spans(
    input_ids: torch.Tensor | list[int],
    realized_insertions: list[dict[str, Any]],
    *,
    prompt_text: str | None = None,
    token_offsets: torch.Tensor | list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Find realized insertion spans in the final model input token sequence.

    Dataset insertion positions are context-local, while hidden states are indexed
    over the final chat-templated input. This helper locates each inserted token
    sequence in the actual prompt sent to the model and returns prompt-level spans
    using an exclusive ``end`` index.
    """

    sequence = _flat_token_ids(input_ids)
    offsets = _flat_token_offsets(token_offsets)
    text_span_matches_by_needle: dict[str, list[tuple[int, int]]] = {}
    if prompt_text is not None and offsets is not None:
        for insertion in realized_insertions:
            decoded_text = insertion.get("decoded_text")
            if isinstance(decoded_text, str) and decoded_text:
                text_span_matches_by_needle[str(insertion.get("needle_id"))] = (
                    _text_occurrence_spans(prompt_text, decoded_text)
                )
    used_text_spans: set[tuple[int, int]] = set()
    ordered_insertions = sorted(
        realized_insertions,
        key=lambda item: int(
            item.get("final_position", item.get("requested_position", 0))
        ),
    )
    spans: list[dict[str, Any]] = []
    last_start = -1
    context_offset: int | None = None
    used: set[tuple[int, int]] = set()

    for insertion in ordered_insertions:
        pattern = [int(x) for x in insertion.get("tokens", [])]
        final_position = int(
            insertion.get("final_position", insertion.get("requested_position", 0))
        )
        matches = find_subsequence_positions(sequence, pattern)
        available = [match for match in matches if match not in used]

        chosen: tuple[int, int] | None = None
        if pattern and available:
            monotonic = [match for match in available if match[0] >= last_start]
            candidates = monotonic or available
            if context_offset is not None:
                chosen = min(
                    candidates,
                    key=lambda match: (
                        abs((match[0] - final_position) - context_offset),
                        match[0] < last_start,
                        match[0],
                    ),
                )
            else:
                chosen = min(
                    candidates, key=lambda match: (match[0] < last_start, match[0])
                )
                context_offset = chosen[0] - final_position
        if chosen is None and offsets is not None:
            text_spans = text_span_matches_by_needle.get(
                str(insertion.get("needle_id")), []
            )
            available_text_spans = [
                span for span in text_spans if span not in used_text_spans
            ]
            if available_text_spans:
                token_span_candidates: list[tuple[tuple[int, int], tuple[int, int]]] = (
                    []
                )
                for text_span in available_text_spans:
                    token_span = _token_span_for_char_span(offsets, text_span)
                    if token_span is not None and token_span not in used:
                        token_span_candidates.append((token_span, text_span))
                if token_span_candidates:
                    monotonic = [
                        candidate
                        for candidate in token_span_candidates
                        if candidate[0][0] >= last_start
                    ]
                    candidates = monotonic or token_span_candidates
                    chosen, chosen_text_span = min(
                        candidates,
                        key=lambda candidate: (
                            candidate[0][0] < last_start,
                            candidate[0][0],
                        ),
                    )
                    used_text_spans.add(chosen_text_span)
                    if context_offset is None:
                        context_offset = chosen[0] - final_position

        if chosen is None and pattern and context_offset is not None:
            # The dataset stores insertion token ids before the full context is
            # decoded and wrapped in a chat template. Retokenizing that text can
            # merge a needle boundary with adjacent haystack text, so an exact
            # subsequence search may fail even though the insertion position is
            # known. Fall back to the established context-to-prompt offset so
            # every realized needle receives a prompt-level span for plotting.
            start = max(0, min(len(sequence), final_position + context_offset))
            end = max(start, min(len(sequence), start + len(pattern)))
            if end > start:
                chosen = (start, end)
                print(
                    "[hidden-analysis] warning: approximated inserted token span "
                    f"for {insertion.get('needle_id')} in final model input"
                )

        if chosen is None:
            print(
                "[hidden-analysis] warning: could not find inserted token sequence "
                f"for {insertion.get('needle_id')} in final model input"
            )
            continue

        used.add(chosen)
        if context_offset is None:
            context_offset = chosen[0] - final_position
        last_start = chosen[0]
        spans.append(
            {
                "needle_id": insertion.get("needle_id"),
                "start": int(chosen[0]),
                "end": int(chosen[1]),
                "length": int(chosen[1] - chosen[0]),
                "context_final_position": final_position,
                "is_control": bool(insertion.get("is_control", False)),
                "inserted_from": insertion.get("inserted_from"),
                "decoded_text": insertion.get("decoded_text"),
            }
        )
    return spans


def expand_needle_segments(
    needle_spans: list[dict[str, Any]],
    *,
    sequence_length: int,
    expansion: int = 5,
) -> list[dict[str, Any]]:
    """Expand prompt-level needle spans by ``expansion`` tokens after each needle."""

    if expansion < 0:
        raise ValueError("expansion must be non-negative")
    expanded: list[dict[str, Any]] = []
    for span in needle_spans:
        start = max(0, int(span["start"]))
        end = min(sequence_length, int(span["end"]))
        expanded_end = min(sequence_length, end + int(expansion))
        expanded.append(
            {
                **span,
                "start": start,
                "end": end,
                "expanded_end": expanded_end,
                "expansion": int(expansion),
            }
        )
    return expanded


def build_outside_segments_mask(
    sequence_length: int, expanded_segments: list[dict[str, Any]]
) -> torch.Tensor:
    """Return a boolean mask that is true outside every expanded needle segment."""

    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    mask = torch.ones(sequence_length, dtype=torch.bool)
    for segment in expanded_segments:
        start = max(0, int(segment["start"]))
        end_value = (
            segment["expanded_end"] if "expanded_end" in segment else segment["end"]
        )
        end = min(sequence_length, int(end_value))
        if start < end:
            mask[start:end] = False
    return mask


def compute_needle_sensitive_tokens(
    hidden: torch.Tensor,
    hidden_control: torch.Tensor,
    input_ids: torch.Tensor | list[int],
    *,
    layer_indices: list[int] | None = None,
    layer_labels: list[int] | None = None,
    expanded_segments: list[dict[str, Any]] | None = None,
    insertion_position: int = 0,
    offset: int = 0,
    top_m: int = 20,
    decode_token: Callable[[list[int]], str] | None = None,
) -> list[dict[str, Any]]:
    """Select low-cosine non-needle prompt tokens for each layer.

    Returned positions ``t`` are uncontrolled final-input positions. The
    corresponding controlled position is ``t`` before the controlled insertion and
    ``t + offset`` at/after the controlled insertion.
    """

    if top_m <= 0:
        raise ValueError("top_m must be positive")
    normal = normalize_hidden_tensor(hidden, name="hidden")
    control = normalize_hidden_tensor(hidden_control, name="hidden_control")
    token_ids = _flat_token_ids(input_ids)
    sequence_length = min(len(token_ids), normal.shape[1])
    if layer_indices is None:
        layer_indices = list(range(normal.shape[0]))
    if layer_labels is None:
        layer_labels = [int(x) for x in layer_indices]
    if len(layer_indices) != len(layer_labels):
        raise ValueError("layer_indices and layer_labels must have the same length")

    outside_mask = build_outside_segments_mask(sequence_length, expanded_segments or [])
    positions = torch.arange(sequence_length, dtype=torch.long)
    control_positions = aligned_control_positions_for_input(
        sequence_length,
        int(control.shape[1]),
        insertion_position=insertion_position,
        offset=offset,
    )
    valid_mask = outside_mask & (control_positions >= 0)
    candidate_positions = positions[valid_mask]
    candidate_control_positions = control_positions[valid_mask]

    rows: list[dict[str, Any]] = []
    for stored_layer_idx, layer_label in zip(layer_indices, layer_labels):
        if stored_layer_idx < 0 or stored_layer_idx >= normal.shape[0]:
            raise ValueError(
                f"layer index {stored_layer_idx} is outside available range 0..{normal.shape[0] - 1}"
            )
        if candidate_positions.numel() == 0:
            selected_positions = candidate_positions
            selected_control_positions = candidate_control_positions
            selected_scores = torch.empty(0, dtype=torch.float32)
        else:
            normal_rows = normal[stored_layer_idx, candidate_positions]
            control_rows = control[stored_layer_idx, candidate_control_positions]
            scores = F.cosine_similarity(normal_rows, control_rows, dim=-1)
            k = min(int(top_m), int(scores.numel()))
            selected_scores, order = torch.topk(scores, k=k, largest=False, sorted=True)
            selected_positions = candidate_positions[order]
            selected_control_positions = candidate_control_positions[order]

        tokens: list[dict[str, Any]] = []
        for pos, ctrl_pos, score in zip(
            selected_positions.tolist(),
            selected_control_positions.tolist(),
            selected_scores.tolist(),
        ):
            token_id = int(token_ids[int(pos)])
            token_text = (
                decode_token([token_id]) if decode_token is not None else str(token_id)
            )
            tokens.append(
                {
                    "token": token_text,
                    "token_id": token_id,
                    "position": int(pos),
                    "control_position": int(ctrl_pos),
                    "cosine_similarity": float(score),
                }
            )
        rows.append({"layer": int(layer_label), "tokens": tokens})
    return rows


def save_needle_sensitive_outputs(
    records: list[dict[str, Any]], output_dir: str | Path
) -> tuple[Path, Path]:
    """Save needle-sensitive token records as JSON and readable text."""

    out_dir = Path(output_dir)
    tables_dir = (
        out_dir.parent / "tables" if out_dir.name in {"figures", "tensors"} else out_dir
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        sample_idx = int(record["sample_idx"])
        for layer_record in record.get("layers", []):
            layer_key = str(int(layer_record["layer"]))
            for token_record in layer_record.get("tokens", []):
                by_layer.setdefault(layer_key, []).append(
                    {"sample_idx": sample_idx, **token_record}
                )
    for layer_key, rows in by_layer.items():
        rows.sort(key=lambda row: float(row["cosine_similarity"]))

    payload = {"samples": records, "by_layer": by_layer}
    json_path = tables_dir / "needle_sensitive_tokens.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines: list[str] = []
    for record in records:
        lines.append(f"Example ID {record['sample_idx']}")
        for layer_record in record.get("layers", []):
            lines.append(f"Layer {layer_record['layer']}")
            for token_record in layer_record.get("tokens", []):
                token = repr(token_record["token"])
                lines.append(
                    f"  token={token} t={token_record['position']} "
                    f"control_t={token_record['control_position']} "
                    f"C[t]={token_record['cosine_similarity']:.8f}"
                )
        lines.append("")
    txt_path = tables_dir / "needle_sensitive_tokens.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path


def save_model_input_ids_table(
    records: list[dict[str, Any]], output_dir: str | Path
) -> Path:
    """Write final uncontrolled/controlled model input token ids for inspection."""

    out_dir = Path(output_dir)
    tables_dir = (
        out_dir.parent / "tables" if out_dir.name in {"figures", "tensors"} else out_dir
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        lines.append(f"Example ID {record['sample_idx']}")
        lines.append("uncontrolled input ids")
        lines.append(" ".join(str(int(x)) for x in record["uncontrolled_input_ids"]))
        lines.append("controlled input ids")
        lines.append(" ".join(str(int(x)) for x in record["controlled_input_ids"]))
        lines.append("")
    path = tables_dir / "model_input_ids.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def normalize_hidden_tensor(
    hidden: torch.Tensor, *, name: str = "hidden"
) -> torch.Tensor:
    """Return a CPU float tensor with shape [layers, sequence_length, hidden_dim]."""
    if hidden.ndim == 4:
        if hidden.shape[0] != 1:
            raise ValueError(
                f"{name} has batch dimension {hidden.shape[0]}; expected batch_size=1"
            )
        hidden = hidden.squeeze(0)
    if hidden.ndim != 3:
        raise ValueError(
            f"{name} must have shape [layers, sequence_length, hidden_dim] or "
            "[1, layers, sequence_length, hidden_dim]"
        )
    if hidden.shape[0] <= 0 or hidden.shape[1] <= 0 or hidden.shape[2] <= 0:
        raise ValueError(
            f"{name} must have non-empty layer, sequence, and hidden dimensions"
        )
    return hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()


def save_hidden_states(
    hidden: torch.Tensor,
    hidden_control: torch.Tensor,
    output_dir: str | Path,
    sample_idx: int,
    *,
    layers: list[int] | None = None,
    input_ids: torch.Tensor | list[int] | None = None,
    input_ids_control: torch.Tensor | list[int] | None = None,
    insertion_position: int | None = None,
    offset: int | None = None,
    pca_start_position: int | None = None,
    needle_spans: list[dict[str, Any]] | None = None,
    expanded_needle_segments: list[dict[str, Any]] | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normal = normalize_hidden_tensor(hidden, name="hidden")
    control = normalize_hidden_tensor(hidden_control, name="hidden_control")
    if normal.shape[0] != control.shape[0] or normal.shape[2] != control.shape[2]:
        raise ValueError(
            "hidden and hidden_control must have matching layer and hidden dimensions; "
            f"got {tuple(normal.shape)} and {tuple(control.shape)}"
        )
    stored_layers = None if layers is None else [int(x) for x in layers]
    if stored_layers is not None:
        bad_layers = [
            layer for layer in stored_layers if layer < 0 or layer >= normal.shape[0]
        ]
        if bad_layers:
            raise ValueError(
                f"Cannot save requested hidden-state layers {bad_layers}; "
                f"available range is 0..{normal.shape[0] - 1}"
            )
        layer_index = torch.tensor(stored_layers, dtype=torch.long)
        normal = normal.index_select(0, layer_index)
        control = control.index_select(0, layer_index)
    path = out_dir / f"hidden_inputs_{sample_idx}.pt"
    torch.save(
        {
            "hidden": normal,
            "hidden_control": control,
            "sample_idx": int(sample_idx),
            "layers": stored_layers,
            "stored_layers": stored_layers,
            "input_ids": None if input_ids is None else _flat_token_ids(input_ids),
            "input_ids_control": (
                None
                if input_ids_control is None
                else _flat_token_ids(input_ids_control)
            ),
            "insertion_position": insertion_position,
            "offset": offset,
            "pca_start_position": pca_start_position,
            "needle_spans": needle_spans or [],
            "expanded_needle_segments": expanded_needle_segments or [],
        },
        path,
    )
    return path


def _hidden_record_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1]), path.name
    except (IndexError, ValueError):
        return 10**12, path.name


def _sample_idx_from_stem(path: Path, *, prefix: str, suffix: str = "") -> int | None:
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix) :]
    if suffix:
        if not tail.endswith(suffix):
            return None
        tail = tail[: -len(suffix)]
    try:
        return int(tail)
    except ValueError:
        return None


def _sample_indices_for_paths(
    paths: list[Path], *, prefix: str, suffix: str = ""
) -> set[int]:
    return {
        idx
        for path in paths
        if (idx := _sample_idx_from_stem(path, prefix=prefix, suffix=suffix))
        is not None
    }


def _validate_expected_sample_indices(
    *,
    artifact_name: str,
    found: set[int],
    expected_examples: int | None,
    run_dir: Path,
) -> None:
    if not found:
        raise FileNotFoundError(
            f"Hidden-state analysis did not produce any {artifact_name} in {run_dir}. "
            "Re-run Section 7 and check the script output above for the first error."
        )
    if expected_examples is None:
        return
    missing = [idx for idx in range(int(expected_examples)) if idx not in found]
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(
            f"Hidden-state analysis is missing {artifact_name} for sample(s): "
            f"{preview}{suffix} in {run_dir}. Re-run Section 7 for the same RUN_DIR."
        )


def validate_hidden_analysis_artifacts(
    run_dir: str | Path, *, expected_examples: int | None = None
) -> dict[str, list[Path]]:
    """Validate that Section 7 wrote its required figures and tensors."""

    root = Path(run_dir)
    tensors_dir = root / "tensors"
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    artifacts = {
        "measurement_tensors": sorted(tensors_dir.glob("inputs_*.pt")),
        "hidden_tensors": sorted(tensors_dir.glob("hidden_inputs_*.pt")),
        "measurement_figures": [
            path
            for path in sorted(figures_dir.glob("inputs_*.png"))
            if _sample_idx_from_stem(path, prefix="inputs_") is not None
        ],
        "measurement_tables": sorted(tables_dir.glob("inputs_*_measurements.csv")),
    }
    specs = {
        "measurement tensor inputs_*.pt": (
            artifacts["measurement_tensors"],
            "inputs_",
            "",
        ),
        "hidden tensor hidden_inputs_*.pt": (
            artifacts["hidden_tensors"],
            "hidden_inputs_",
            "",
        ),
        "measurement figure inputs_*.png": (
            artifacts["measurement_figures"],
            "inputs_",
            "",
        ),
        "measurement table inputs_*_measurements.csv": (
            artifacts["measurement_tables"],
            "inputs_",
            "_measurements",
        ),
    }
    for artifact_name, (paths, prefix, suffix) in specs.items():
        found = _sample_indices_for_paths(paths, prefix=prefix, suffix=suffix)
        _validate_expected_sample_indices(
            artifact_name=artifact_name,
            found=found,
            expected_examples=expected_examples,
            run_dir=root,
        )
    return artifacts


def load_hidden_state_records(input_dir: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        Path(input_dir).glob("hidden_inputs_*.pt"), key=_hidden_record_sort_key
    ):
        payload = torch.load(path, map_location="cpu")
        if (
            not isinstance(payload, dict)
            or "hidden" not in payload
            or "hidden_control" not in payload
        ):
            raise ValueError(
                f"Hidden-state file {path} must contain hidden and hidden_control"
            )
        hidden = normalize_hidden_tensor(payload["hidden"], name=f"{path.name}:hidden")
        hidden_control = normalize_hidden_tensor(
            payload["hidden_control"], name=f"{path.name}:hidden_control"
        )
        if (
            hidden.shape[0] != hidden_control.shape[0]
            or hidden.shape[2] != hidden_control.shape[2]
        ):
            raise ValueError(
                f"Hidden-state file {path} has mismatched normal/control shapes: "
                f"{tuple(hidden.shape)} vs {tuple(hidden_control.shape)}"
            )
        sample_idx = int(payload.get("sample_idx", _hidden_record_sort_key(path)[0]))
        records.append(
            {
                **payload,
                "hidden": hidden,
                "hidden_control": hidden_control,
                "sample_idx": sample_idx,
                "path": path,
            }
        )
    if not records:
        raise ValueError(f"No hidden_inputs_*.pt files found in {input_dir}")
    return records


def stored_layer_index(record: dict[str, Any], layer_idx: int) -> int:
    """Map an original model layer id to its stored tensor index."""

    hidden = normalize_hidden_tensor(record["hidden"], name="hidden")
    stored_layers = record.get("stored_layers", record.get("layers"))
    if stored_layers is None:
        if layer_idx < 0 or layer_idx >= hidden.shape[0]:
            raise ValueError(
                f"layer_idx {layer_idx} is outside available range 0..{hidden.shape[0] - 1}"
            )
        return int(layer_idx)
    stored = [int(x) for x in stored_layers]
    if len(stored) != hidden.shape[0]:
        raise ValueError(
            f"Stored layer metadata length {len(stored)} does not match hidden tensor layer count {hidden.shape[0]}"
        )
    try:
        return stored.index(int(layer_idx))
    except ValueError as exc:
        raise ValueError(
            f"Requested layer {layer_idx} is not present in stored layers {stored}"
        ) from exc


def _filter_largest_norm_fraction(
    hidden_layer: torch.Tensor, filter_top_frac: float
) -> torch.Tensor:
    if not 0 <= filter_top_frac < 1:
        raise ValueError("filter_top_frac must satisfy 0 <= filter_top_frac < 1")
    layer = hidden_layer.detach().to(device="cpu", dtype=torch.float32)
    if layer.ndim != 2:
        raise ValueError("hidden_layer must have shape [sequence_length, hidden_dim]")
    remove_count = int(layer.shape[0] * filter_top_frac)
    keep_count = max(1, layer.shape[0] - remove_count)
    keep_indices = torch.argsort(torch.norm(layer, dim=-1))[:keep_count]
    return layer[keep_indices]


def _pca_fit_start_position(record: dict[str, Any], sequence_length: int) -> int:
    start_position = int(record.get("pca_start_position", 0) or 0)
    if start_position < 0:
        raise ValueError("pca_start_position must be non-negative")
    if start_position >= sequence_length:
        raise ValueError(
            "pca_start_position must be within the hidden-state sequence length; "
            f"got {start_position} for length {sequence_length}"
        )
    return start_position


def _non_outlier_mask(points: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    total_count = int(points.shape[0])
    if total_count == 0:
        return torch.zeros(0, dtype=torch.bool), 0, 0

    points = points.detach().to(device="cpu", dtype=torch.float32)
    distances = torch.norm(points - points.mean(dim=0, keepdim=True), dim=-1)
    median_distance = torch.median(distances)
    keep_mask = distances <= 10 * median_distance
    outlier_count = int((~keep_mask).sum().item())
    return keep_mask, outlier_count, total_count


def _scatter_non_outlier_masks(
    normal: torch.Tensor, control: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    normal = normal.detach().to(device="cpu", dtype=torch.float32)
    control = control.detach().to(device="cpu", dtype=torch.float32)
    combined = torch.cat([normal, control], dim=0)
    keep_mask, outlier_count, total_count = _non_outlier_mask(combined)
    normal_keep = keep_mask[: normal.shape[0]]
    control_keep = keep_mask[normal.shape[0] :]
    return normal_keep, control_keep, outlier_count, total_count


def _compute_projection_from_matrix(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            "PCA input matrix must have shape [N, hidden_dim] and be non-empty"
        )
    matrix = matrix.detach().to(device="cpu", dtype=torch.float32)
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    hidden_dim = matrix.shape[1]
    if centered.shape[0] < 2 or torch.count_nonzero(centered).item() == 0:
        projection = torch.zeros((hidden_dim, 2), dtype=torch.float32)
        projection[0, 0] = 1.0
        if hidden_dim > 1:
            projection[1, 1] = 1.0
        return projection, mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    projection = vh[:2].T.contiguous()
    if projection.shape[1] < 2:
        padding = torch.zeros(
            (hidden_dim, 2 - projection.shape[1]), dtype=projection.dtype
        )
        projection = torch.cat([projection, padding], dim=1)
        if hidden_dim > 1:
            projection[1, 1] = 1.0
    return projection, mean


def split_pca_train_test_records(
    records: list[dict[str, Any]], test_count: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda r: int(r["sample_idx"]))
    if test_count is None:
        test_count = len(ordered) // 2
    if test_count <= 0:
        raise ValueError("test_count must be positive")
    if len(ordered) <= test_count:
        raise ValueError(
            f"Need more than {test_count} hidden-state records to reserve test examples; got {len(ordered)}"
        )
    # Reserve the earliest examples for PCA visualization and fit the projection on
    # the later examples. This keeps SELECT_EXAMPLE_ID constrained to the same
    # first-half examples used for hidden-state figures in the counting notebook.
    return ordered[test_count:], ordered[:test_count]


def fit_pca_projections_from_records(
    train_records: list[dict[str, Any]],
    layer_indices: list[int] | None = None,
    *,
    filter_top_frac: float = 0.10,
) -> dict[int, dict[str, torch.Tensor]]:
    if not train_records:
        raise ValueError("At least one training hidden-state record is required")
    first_hidden = normalize_hidden_tensor(train_records[0]["hidden"], name="hidden")
    available_layers = first_hidden.shape[0]
    if layer_indices is None:
        stored_layers = train_records[0].get(
            "stored_layers", train_records[0].get("layers")
        )
        layer_indices = (
            list(range(available_layers))
            if stored_layers is None
            else [int(x) for x in stored_layers]
        )
    projections: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in layer_indices:
        first_stored_idx = stored_layer_index(train_records[0], int(layer_idx))
        rows = []
        for record in train_records:
            hidden = normalize_hidden_tensor(record["hidden"], name="hidden")
            if (
                hidden.shape[0] != available_layers
                or hidden.shape[2] != first_hidden.shape[2]
            ):
                raise ValueError(
                    "All hidden-state records must share layer count and hidden dimension"
                )
            stored_idx = stored_layer_index(record, int(layer_idx))
            start_position = _pca_fit_start_position(record, hidden.shape[1])
            rows.append(
                _filter_largest_norm_fraction(
                    hidden[stored_idx, start_position:], filter_top_frac
                )
            )
        matrix = torch.cat(rows, dim=0)
        projection, mean = _compute_projection_from_matrix(matrix)
        projections[int(layer_idx)] = {
            "projection": projection,
            "mean": mean,
            "stored_layer_index": first_stored_idx,
        }
    return projections


def project_hidden_states_with_pca(
    hidden: torch.Tensor,
    hidden_control: torch.Tensor,
    layer_idx: int,
    projection: torch.Tensor,
    mean: torch.Tensor,
    *,
    start_position: int = 0,
    insertion_position: int = 0,
    offset: int = 0,
) -> dict[str, torch.Tensor]:
    """Project aligned normal/control hidden states onto a prefit PCA basis.

    Returned ``positions`` are uncontrolled final-input positions. The control
    trajectory is gathered from ``control_positions`` after applying ``offset``
    at and after ``insertion_position`` so both plotted trajectories share the
    uncontrolled input sequence as their reference x/color axis.
    """

    normal = normalize_hidden_tensor(hidden, name="hidden")
    control = normalize_hidden_tensor(hidden_control, name="hidden_control")
    if normal.shape[0] != control.shape[0] or normal.shape[2] != control.shape[2]:
        raise ValueError(
            "hidden and hidden_control must share layer and hidden dimensions"
        )
    if layer_idx < 0 or layer_idx >= normal.shape[0]:
        raise ValueError(
            f"layer_idx {layer_idx} is outside available range 0..{normal.shape[0] - 1}"
        )
    projection = projection.detach().to(device="cpu", dtype=torch.float32)
    mean = mean.detach().to(device="cpu", dtype=torch.float32)
    if projection.shape != (normal.shape[2], 2):
        raise ValueError(
            f"projection must have shape ({normal.shape[2]}, 2); got {tuple(projection.shape)}"
        )
    if mean.shape != (normal.shape[2],):
        raise ValueError(
            f"mean must have shape ({normal.shape[2]},); got {tuple(mean.shape)}"
        )
    if start_position < 0:
        raise ValueError("start_position must be non-negative")
    if start_position >= normal.shape[1]:
        raise ValueError(
            "start_position must be within the uncontrolled hidden-state sequence length; "
            f"got {start_position} for length {normal.shape[1]}"
        )
    valid_positions, valid_control_positions = valid_aligned_positions_for_input(
        int(normal.shape[1]),
        int(control.shape[1]),
        insertion_position=int(insertion_position),
        offset=int(offset),
    )
    keep = valid_positions >= int(start_position)
    normal_positions = valid_positions[keep]
    control_positions = valid_control_positions[keep]
    if normal_positions.numel() == 0:
        raise ValueError(
            "No aligned normal/control PCA positions remain after applying "
            f"start_position={start_position}, insertion_position={insertion_position}, offset={offset}"
        )
    normal_centered = normal[layer_idx, normal_positions] - mean
    control_centered = control[layer_idx, control_positions] - mean
    return {
        "normal": normal_centered @ projection,
        "control": control_centered @ projection,
        "positions": normal_positions,
        "control_positions": control_positions,
        "projection": projection,
        "mean": mean,
        "layer": torch.tensor(layer_idx, dtype=torch.int64),
    }


def compute_pca_projection_2d(
    hidden_layer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_layer.ndim != 2:
        raise ValueError("hidden_layer must have shape [sequence_length, hidden_dim]")
    if hidden_layer.shape[0] == 0 or hidden_layer.shape[1] == 0:
        raise ValueError("hidden_layer must be non-empty")

    layer = hidden_layer.detach().to(device="cpu", dtype=torch.float32)
    mean = layer.mean(dim=0, keepdim=True)
    centered = layer - mean

    if centered.shape[0] < 2 or torch.count_nonzero(centered).item() == 0:
        projection = torch.zeros((centered.shape[1], 2), dtype=torch.float32)
        projection[0, 0] = 1.0
        if centered.shape[1] > 1:
            projection[1, 1] = 1.0
        return projection, mean.squeeze(0)

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T.contiguous()
    if components.shape[1] < 2:
        padding = torch.zeros(
            (components.shape[0], 2 - components.shape[1]), dtype=components.dtype
        )
        components = torch.cat([components, padding], dim=1)
        if components.shape[0] > 1:
            components[1, 1] = 1.0
    return components, mean.squeeze(0)


def project_hidden_trajectories_pca(
    hidden: torch.Tensor,
    hidden_control: torch.Tensor,
    layer_idx: int,
    insertion_position: int,
    offset: int,
) -> dict[str, torch.Tensor]:
    hidden_layer = hidden[layer_idx]
    hidden_control_layer = hidden_control[layer_idx]
    normal_segment, control_segment, positions = _aligned_post_insertion_segments(
        hidden_layer, hidden_control_layer, insertion_position, offset
    )

    projection, mean = compute_pca_projection_2d(hidden_layer)
    normal_centered = (
        normal_segment.detach().to(device="cpu", dtype=torch.float32) - mean
    )
    control_centered = (
        control_segment.detach().to(device="cpu", dtype=torch.float32) - mean
    )

    return {
        "normal": normal_centered @ projection,
        "control": control_centered @ projection,
        "positions": positions,
        "projection": projection,
        "mean": mean,
        "layer": torch.tensor(layer_idx, dtype=torch.int64),
        "insertion_position": torch.tensor(insertion_position, dtype=torch.int64),
        "offset": torch.tensor(offset, dtype=torch.int64),
    }


def _projected_xy_for_plot(points: torch.Tensor) -> tuple[Any, Any, str, str]:
    """Return x/y arrays for plotting 1D or 2D projected hidden states."""
    if points.ndim != 2 or points.shape[1] not in (1, 2):
        raise ValueError(
            "projected hidden states must have shape [sequence_length, 1] "
            "or [sequence_length, 2]"
        )
    array = points.numpy()
    if points.shape[1] == 1:
        return array[:, 0], [0.0] * points.shape[0], "projected dimension 1", "0"
    return array[:, 0], array[:, 1], "PC 1", "PC 2"


def _plot_projected_hidden_state_categories(
    *,
    ax: Any,
    normal: torch.Tensor,
    control: torch.Tensor,
    positions: Any,
    control_positions: Any,
    cmap_name: str = "viridis",
) -> Any:
    """Plot categories with marker shape and token position with one color scale."""
    all_positions = list(positions) + list(control_positions)
    if not all_positions:
        raise ValueError(
            "No projected hidden-state points remain after outlier filtering"
        )
    norm = Normalize(vmin=min(all_positions), vmax=max(all_positions))
    cmap = plt.get_cmap(cmap_name)

    normal_x, normal_y, xlabel, ylabel = _projected_xy_for_plot(normal)
    control_x, control_y, _, _ = _projected_xy_for_plot(control)
    normal_scatter = ax.scatter(
        normal_x,
        normal_y,
        c=positions,
        cmap=cmap,
        norm=norm,
        marker="o",
        s=12,
        alpha=0.85,
        label="normal",
    )
    ax.scatter(
        control_x,
        control_y,
        c=control_positions,
        cmap=cmap,
        norm=norm,
        marker="^",
        s=18,
        alpha=0.85,
        label="control",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="normal",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                linestyle="None",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                label="control",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                linestyle="None",
            ),
        ],
        title="label category",
        loc="best",
    )
    return normal_scatter


def plot_pca_trajectory(
    hidden: torch.Tensor,
    hidden_control: torch.Tensor,
    layer_idx: int,
    insertion_position: int,
    offset: int,
    output_dir: str | Path,
    sample_idx: int,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    projected = project_hidden_trajectories_pca(
        hidden, hidden_control, layer_idx, insertion_position, offset
    )

    normal_tensor = projected["normal"].detach().to(torch.float32).cpu()
    control_tensor = projected["control"].detach().to(torch.float32).cpu()
    normal_keep, control_keep, outlier_count, total_count = _scatter_non_outlier_masks(
        normal_tensor, control_tensor
    )
    normal = normal_tensor[normal_keep]
    control = control_tensor[control_keep]
    positions_tensor = projected["positions"].detach().cpu()
    positions = positions_tensor[normal_keep].numpy()
    control_positions = positions_tensor[control_keep].numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    position_scatter = _plot_projected_hidden_state_categories(
        ax=ax,
        normal=normal,
        control=control,
        positions=positions,
        control_positions=control_positions,
    )
    ax.set_title(
        f"PCA hidden-state trajectory: layer {layer_idx}, inputs {sample_idx}\n"
        f"{outlier_count} of {total_count} outliers excluded"
    )
    colorbar = fig.colorbar(position_scatter, ax=ax)
    colorbar.set_label("uncontrolled input position t")
    fig.tight_layout()

    out_path = out_dir / f"PCA_layer_{layer_idx}_inputs_{sample_idx}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_pca_projection(
    projected: dict[str, torch.Tensor],
    output_dir: str | Path,
    sample_idx: int,
    layer_idx: int,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_tensor = projected["normal"].detach().to(torch.float32).cpu()
    control_tensor = projected["control"].detach().to(torch.float32).cpu()
    normal_keep, control_keep, outlier_count, total_count = _scatter_non_outlier_masks(
        normal_tensor, control_tensor
    )
    normal = normal_tensor[normal_keep]
    control = control_tensor[control_keep]
    positions_tensor = projected["positions"].detach().cpu()
    control_positions_tensor = (
        projected.get("control_positions", projected["positions"]).detach().cpu()
    )
    positions = positions_tensor[normal_keep].numpy()
    if positions_tensor.shape[0] == control_tensor.shape[0]:
        control_color_positions = positions_tensor[control_keep].numpy()
    else:
        control_color_positions = control_positions_tensor[control_keep].numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    position_scatter = _plot_projected_hidden_state_categories(
        ax=ax,
        normal=normal,
        control=control,
        positions=positions,
        control_positions=control_color_positions,
    )
    ax.set_title(
        f"PCA hidden states: layer {layer_idx}, inputs {sample_idx}\n"
        f"{outlier_count} of {total_count} outliers excluded"
    )
    colorbar = fig.colorbar(position_scatter, ax=ax)
    colorbar.set_label("uncontrolled input position t")
    fig.tight_layout()

    out_path = out_dir / f"PCA_layer_{layer_idx}_inputs_{sample_idx}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_saved_hidden_pca(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    layer_indices: list[int] | None = None,
    *,
    test_count: int | None = None,
    filter_top_frac: float = 0.10,
) -> list[Path]:
    train_records, test_records = split_pca_train_test_records(
        records, test_count=test_count
    )
    projections = fit_pca_projections_from_records(
        train_records, layer_indices, filter_top_frac=filter_top_frac
    )
    paths: list[Path] = []
    print(
        f"[hidden-analysis] PCA fit train_examples={len(train_records)} "
        f"test_examples={len(test_records)} layers={list(projections.keys())} "
        f"filter_top_frac={filter_top_frac}"
    )
    for record in test_records:
        sample_idx = int(record["sample_idx"])
        for layer_idx, params in projections.items():
            stored_idx = stored_layer_index(record, int(layer_idx))
            projected = project_hidden_states_with_pca(
                record["hidden"],
                record["hidden_control"],
                stored_idx,
                params["projection"],
                params["mean"],
                start_position=int(record.get("pca_start_position", 0) or 0),
                insertion_position=int(record.get("insertion_position", 0) or 0),
                offset=int(record.get("offset", 0) or 0),
            )
            projected["layer"] = torch.tensor(int(layer_idx), dtype=torch.int64)
            path = plot_pca_projection(projected, output_dir, sample_idx, layer_idx)
            paths.append(path)
            print(
                f"[hidden-analysis] saved PCA figure sample={sample_idx} layer={layer_idx} path={path}"
            )
    return paths


def save_measurements(
    measurements: dict[str, torch.Tensor], output_dir: str | Path, sample_idx: int
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"inputs_{sample_idx}.pt"
    torch.save(measurements, path)
    return path


def _measurement_plot_table_path(output_dir: str | Path, sample_idx: int) -> Path:
    out_dir = Path(output_dir)
    tables_dir = out_dir.parent / "tables" if out_dir.name == "figures" else out_dir
    return tables_dir / f"inputs_{sample_idx}_measurements.csv"


def save_measurement_plot_table(
    measurements: dict[str, torch.Tensor], output_dir: str | Path, sample_idx: int
) -> Path:
    """Save per-position measurement plot data used by ``inputs_{sample_idx}.png``."""

    metric_keys = [
        k
        for k in measurements.keys()
        if k in {"relative_norm_diff", "cosine_similarity"}
    ]
    layer_ids = [int(x) for x in measurements["layers"].tolist()]
    max_positions = max(
        (int(measurements[key].shape[1]) for key in metric_keys), default=0
    )
    if "positions" in measurements:
        positions = [int(x) for x in measurements["positions"].detach().cpu().tolist()]
    else:
        positions = list(range(max_positions))
    table_path = _measurement_plot_table_path(output_dir, sample_idx)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    layer_colors = {
        layer_id: (
            to_hex(color_cycle[layer_i % len(color_cycle)]) if color_cycle else ""
        )
        for layer_i, layer_id in enumerate(layer_ids)
    }
    fieldnames = ["position", "layer", "layer_color", *metric_keys]
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer_i, layer_id in enumerate(layer_ids):
            for value_idx, position in enumerate(positions):
                row: dict[str, Any] = {
                    "position": position,
                    "layer": layer_id,
                    "layer_color": layer_colors[layer_id],
                }
                for key in metric_keys:
                    arr = measurements[key]
                    if layer_i < arr.shape[0] and value_idx < arr.shape[1]:
                        value = (
                            arr[layer_i, value_idx]
                            .detach()
                            .to(torch.float32)
                            .cpu()
                            .item()
                        )
                        row[key] = float(value)
                writer.writerow(row)
    return table_path


def plot_measurements(
    measurements: dict[str, torch.Tensor],
    output_dir: str | Path,
    sample_idx: int,
    *,
    needle_spans: list[dict[str, Any]] | None = None,
    vertical_lines: list[dict[str, Any]] | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        k
        for k in measurements.keys()
        if k in {"relative_norm_diff", "cosine_similarity"}
    ]
    fig, axes = plt.subplots(
        len(metric_keys), 1, figsize=(12, 4 * len(metric_keys)), squeeze=False
    )
    layer_ids = measurements["layers"].tolist()

    for row_idx, key in enumerate(metric_keys):
        ax = axes[row_idx][0]
        arr = measurements[key]
        if "positions" in measurements:
            x = [int(v) for v in measurements["positions"].detach().cpu().tolist()]
        else:
            x = list(range(arr.shape[1]))
        if needle_spans:
            labeled_any = False
            labeled_control = False
            for span in needle_spans:
                start = int(span["start"])
                end = int(span["end"])
                if end <= min(x, default=0) or start > max(x, default=-1):
                    continue
                is_control = bool(span.get("is_control", False))
                label = None
                if is_control and not labeled_control:
                    label = "controlled needle span"
                    labeled_control = True
                elif not is_control and not labeled_any:
                    label = "needle span"
                    labeled_any = True
                ax.axvspan(
                    start,
                    end,
                    color="tab:red" if is_control else "tab:orange",
                    alpha=0.16 if is_control else 0.12,
                    label=label,
                    zorder=0,
                )
        if vertical_lines:
            labeled_lines: set[str] = set()
            for line in vertical_lines:
                position = int(line["position"])
                label_text = str(line.get("label", "reference"))
                label = label_text if label_text not in labeled_lines else None
                labeled_lines.add(label_text)
                ax.axvline(
                    position,
                    color=str(line.get("color", "cyan")),
                    linestyle=str(line.get("linestyle", "--")),
                    linewidth=float(line.get("linewidth", 1.2)),
                    alpha=float(line.get("alpha", 0.85)),
                    label=label,
                    zorder=1,
                )
        for layer_i, layer_id in enumerate(layer_ids):
            y = arr[layer_i].detach().to(torch.float32).cpu().numpy()
            ax.plot(x, y, label=f"layer {layer_id}")
        ax.set_title(key)
        ax.set_xlabel("model input token position")
        ax.set_ylabel(key)
        ax.legend(loc="best")

    save_measurement_plot_table(measurements, out_dir, sample_idx)
    fig.tight_layout()
    out_path = out_dir / f"inputs_{sample_idx}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def write_dataset_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
