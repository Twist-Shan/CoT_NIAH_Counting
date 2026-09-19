from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from dataset_generation.response_eval import build_response_result
from single_example.ablation_representation_analysis import resolve_decoder_layers
from single_example.single_example_analysis import (
    locate_uncontrolled_needle_segments,
    render_and_tokenize_messages,
)


@dataclass(frozen=True)
class SteeringExample:
    dataset_index: int
    row_id: str
    group: str
    row: dict[str, Any]
    scored_row: dict[str, Any] | None = None


@dataclass(frozen=True)
class SteeringVector:
    layer: int
    vector: torch.Tensor
    norm: float
    source_path: str
    source_space: str = "hidden"
    standardized_probe: bool = False


@dataclass(frozen=True)
class NeedleSpanSteeringTarget:
    dataset_index: int
    row_id: str
    group: str
    row: dict[str, Any]
    scored_row: dict[str, Any] | None
    input_ids: torch.Tensor
    needle_id: str
    needle_ordinal: int
    span_start: int
    span_end: int
    span_length: int
    decoded_text: str | None = None


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id", index))


def _scored_by_id(
    scored_rows: Sequence[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if scored_rows is None:
        return {}
    return {_row_id(row, idx): row for idx, row in enumerate(scored_rows)}


def _balanced_prefix(
    successes: list[SteeringExample],
    failures: list[SteeringExample],
    max_total: int,
) -> list[SteeringExample]:
    if max_total <= 0:
        raise ValueError(f"max_total must be positive, got {max_total}")
    first_successes = min(len(successes), max_total // 2)
    first_failures = min(len(failures), max_total - first_successes)
    selected = successes[:first_successes] + failures[:first_failures]
    remaining = max_total - len(selected)
    if remaining > 0:
        selected_ids = {(ex.group, ex.dataset_index) for ex in selected}
        extras = [
            ex
            for ex in successes[first_successes:] + failures[first_failures:]
            if (ex.group, ex.dataset_index) not in selected_ids
        ]
        selected.extend(extras[:remaining])
    return sorted(selected, key=lambda ex: ex.dataset_index)


def _matching_needle_ids_for_row(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for record in row.get("relevant_records", []) or []:
        needle_id = record.get("needle_id")
        if needle_id is not None:
            ids.append(str(needle_id))
    if ids:
        return ids

    try:
        count = int(row.get("gold_answer", {}).get("count"))
    except (TypeError, ValueError):
        return []
    inserted: list[str] = []
    for needle in row.get("needles", []) or []:
        if needle.get("is_inserted", True):
            needle_id = needle.get("needle_id")
            if needle_id is not None:
                inserted.append(str(needle_id))
    return inserted[:count]


def _expected_inserted_needle_count(row: dict[str, Any]) -> int | None:
    needles = row.get("needles")
    if not isinstance(needles, list):
        return None
    return sum(1 for needle in needles if needle.get("is_inserted", True))


def _matching_segments_for_row(
    row: dict[str, Any], segments: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    matching_ids = _matching_needle_ids_for_row(row)
    if matching_ids:
        wanted = {str(x) for x in matching_ids}
        matched = [
            dict(seg) for seg in segments if str(seg.get("needle_id")) in wanted
        ]
        if matched:
            return sorted(matched, key=lambda item: int(item["start"]))
    return sorted((dict(seg) for seg in segments), key=lambda item: int(item["start"]))


def select_steering_examples(
    dataset_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]] | None,
    *,
    max_total: int = 10,
) -> tuple[list[SteeringExample], dict[str, Any]]:
    """Select up to `max_total` examples, balancing successful and failed rows.

    The group labels come from scored `exact_match` values. Rows without scores
    are excluded because steering summaries need a clear successful/unsuccessful
    grouping.
    """

    scored_lookup = _scored_by_id(scored_rows)
    successes: list[SteeringExample] = []
    failures: list[SteeringExample] = []
    missing: list[str] = []
    for idx, row in enumerate(dataset_rows):
        row_id = _row_id(row, idx)
        scored = scored_lookup.get(row_id) if scored_lookup else row
        if scored is None or "exact_match" not in scored:
            missing.append(row_id)
            continue
        group = "successful" if bool(scored.get("exact_match")) else "unsuccessful"
        ex = SteeringExample(
            dataset_index=idx,
            row_id=row_id,
            group=group,
            row=dict(row),
            scored_row=dict(scored),
        )
        if group == "successful":
            successes.append(ex)
        else:
            failures.append(ex)

    selected = _balanced_prefix(successes, failures, int(max_total))
    summary = {
        "num_dataset_rows": len(dataset_rows),
        "num_scored_rows": len(scored_rows) if scored_rows is not None else len(dataset_rows),
        "num_successful": len(successes),
        "num_unsuccessful": len(failures),
        "num_missing_scores": len(missing),
        "missing_score_ids": missing[:50],
        "max_total": int(max_total),
        "num_selected": len(selected),
        "selected_dataset_indices": [ex.dataset_index for ex in selected],
        "selected_row_ids": [ex.row_id for ex in selected],
        "selected_groups": [ex.group for ex in selected],
    }
    return selected, summary


def build_needle_span_steering_targets(
    tokenizer: Any,
    examples: Sequence[SteeringExample],
    *,
    thinking_mode: bool = False,
) -> tuple[list[NeedleSpanSteeringTarget], dict[str, Any]]:
    """Locate actual matching needle spans for selected steering examples."""

    targets: list[NeedleSpanSteeringTarget] = []
    skipped: list[dict[str, Any]] = []
    for ex in examples:
        tokenized = render_and_tokenize_messages(
            tokenizer,
            ex.row.get("uncontrolled_messages") or ex.row.get("messages"),
            thinking_mode=thinking_mode,
        )
        expected_num_needles = _expected_inserted_needle_count(ex.row)
        try:
            segments = locate_uncontrolled_needle_segments(
                row=ex.row,
                uncontrolled_input_ids=tokenized.input_ids,
                prompt_text=tokenized.prompt_text,
                token_offsets=tokenized.token_offsets,
                expected_num_needles=expected_num_needles,
            )
        except Exception as exc:
            skipped.append(
                {
                    "dataset_index": ex.dataset_index,
                    "row_id": ex.row_id,
                    "reason": repr(exc),
                }
            )
            continue
        matching_segments = _matching_segments_for_row(ex.row, segments)
        flat_input_ids = tokenized.input_ids.detach().cpu()
        for ordinal, segment in enumerate(matching_segments):
            start = int(segment["start"])
            end = int(segment["end"])
            if end <= start:
                skipped.append(
                    {
                        "dataset_index": ex.dataset_index,
                        "row_id": ex.row_id,
                        "needle_id": segment.get("needle_id"),
                        "reason": f"empty span start={start} end={end}",
                    }
                )
                continue
            targets.append(
                NeedleSpanSteeringTarget(
                    dataset_index=ex.dataset_index,
                    row_id=ex.row_id,
                    group=ex.group,
                    row=ex.row,
                    scored_row=ex.scored_row,
                    input_ids=flat_input_ids,
                    needle_id=str(segment.get("needle_id", ordinal)),
                    needle_ordinal=ordinal,
                    span_start=start,
                    span_end=end,
                    span_length=end - start,
                    decoded_text=(
                        segment.get("decoded_text")
                        if isinstance(segment.get("decoded_text"), str)
                        else None
                    ),
                )
            )
    summary = {
        "num_examples": len(examples),
        "num_targets": len(targets),
        "num_skipped": len(skipped),
        "skipped": skipped[:50],
        "targets_by_example": {
            str(ex.dataset_index): sum(
                1 for target in targets if target.dataset_index == ex.dataset_index
            )
            for ex in examples
        },
    }
    return targets, summary


def load_ridge_counting_vector(path: str | Path, *, layer: int) -> SteeringVector:
    """Load a ridge counting direction and convert it to hidden-state space.

    Ridge probes may be trained on standardized features z=(h-mean)/scale. The
    saved coef then lives in z-space, but steering edits h directly, so the
    hidden-space count direction is coef / scale.
    """

    probe_path = Path(path)
    payload = torch.load(probe_path, map_location="cpu")
    if not isinstance(payload, dict) or "coef" not in payload:
        raise ValueError(f"Ridge probe at {probe_path} does not contain a 'coef' tensor")
    coef = torch.as_tensor(payload["coef"], dtype=torch.float32).flatten()
    if coef.ndim != 1 or coef.numel() == 0:
        raise ValueError(f"Ridge probe coefficient must be a non-empty vector: {tuple(coef.shape)}")

    standardized = bool(payload.get("standardize", False))
    if standardized and payload.get("feature_scale") is not None:
        scale = torch.as_tensor(payload["feature_scale"], dtype=torch.float32).flatten()
        if scale.shape != coef.shape:
            raise ValueError(
                f"feature_scale shape {tuple(scale.shape)} does not match coef shape {tuple(coef.shape)}"
            )
        scale = torch.where(scale.abs() < 1e-6, torch.ones_like(scale), scale)
        vector = coef / scale
        source_space = "hidden_from_standardized_ridge"
    else:
        vector = coef
        source_space = "hidden"

    norm = float(torch.linalg.vector_norm(vector).item())
    if not torch.isfinite(torch.tensor(norm)) or norm <= 0.0:
        raise ValueError(f"Ridge probe coefficient has invalid hidden-space norm: {norm}")
    return SteeringVector(
        layer=int(layer),
        vector=vector / norm,
        norm=norm,
        source_path=str(probe_path),
        source_space=source_space,
        standardized_probe=standardized,
    )


def load_contrastive_success_vector(path: str | Path, *, layer: int) -> SteeringVector:
    """Load a saved contrastive success direction for steering."""

    vector_path = Path(path)
    payload = torch.load(vector_path, map_location="cpu")
    if not isinstance(payload, dict) or "direction" not in payload:
        raise ValueError(
            f"Contrastive success vector at {vector_path} does not contain a 'direction' tensor"
        )
    direction = torch.as_tensor(payload["direction"], dtype=torch.float32).flatten()
    if direction.ndim != 1 or direction.numel() == 0:
        raise ValueError(
            f"Contrastive success direction must be a non-empty vector: {tuple(direction.shape)}"
        )
    norm = float(torch.linalg.vector_norm(direction).item())
    if not torch.isfinite(torch.tensor(norm)) or norm <= 0.0:
        raise ValueError(f"Contrastive success direction has invalid norm: {norm}")
    return SteeringVector(
        layer=int(layer),
        vector=direction / norm,
        norm=float(payload.get("raw_norm", norm)),
        source_path=str(vector_path),
        source_space="contrastive_success",
        standardized_probe=False,
    )


def load_counterfactual_count_vector(path: str | Path, *, layer: int) -> SteeringVector:
    """Load a saved counterfactual count direction for steering."""

    vector_path = Path(path)
    payload = torch.load(vector_path, map_location="cpu")
    if not isinstance(payload, dict) or "direction" not in payload:
        raise ValueError(
            f"Counterfactual count vector at {vector_path} does not contain a 'direction' tensor"
        )
    direction = torch.as_tensor(payload["direction"], dtype=torch.float32).flatten()
    if direction.ndim != 1 or direction.numel() == 0:
        raise ValueError(
            f"Counterfactual count direction must be a non-empty vector: {tuple(direction.shape)}"
        )
    norm = float(torch.linalg.vector_norm(direction).item())
    if not torch.isfinite(torch.tensor(norm)) or norm <= 0.0:
        raise ValueError(f"Counterfactual count direction has invalid norm: {norm}")
    return SteeringVector(
        layer=int(layer),
        vector=direction / norm,
        norm=float(payload.get("raw_norm", norm)),
        source_path=str(vector_path),
        source_space="counterfactual_count",
        standardized_probe=False,
    )


def load_counting_steering_vector(
    probe_dir: str | Path, *, layer: int, vector_source: str = "ridge"
) -> SteeringVector:
    """Load the selected counting-feature vector for steering."""

    root = Path(probe_dir)
    if vector_source == "ridge":
        return load_ridge_counting_vector(
            root / f"ridge_probe_layer_{int(layer)}.pt", layer=layer
        )
    if vector_source == "contrastive-success":
        candidates = [
            root / "contrastive_success" / f"contrastive_success_layer_{int(layer)}.pt",
            root / f"contrastive_success_layer_{int(layer)}.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return load_contrastive_success_vector(candidate, layer=layer)
        raise FileNotFoundError(
            "Missing contrastive success vector for layer "
            f"{int(layer)}; checked " + ", ".join(str(path) for path in candidates)
        )
    if vector_source == "counterfactual":
        candidates = [
            root / "counterfactual" / f"counterfactual_count_layer_{int(layer)}.pt",
            root / f"counterfactual_count_layer_{int(layer)}.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return load_counterfactual_count_vector(candidate, layer=layer)
        raise FileNotFoundError(
            "Missing counterfactual count vector for layer "
            f"{int(layer)}; checked " + ", ".join(str(path) for path in candidates)
        )
    raise ValueError(
        "vector_source must be 'ridge', 'contrastive-success', or 'counterfactual', "
        f"got {vector_source!r}"
    )


def input_ids_for_counting_row(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    thinking_mode: bool = False,
) -> torch.Tensor:
    messages = row.get("uncontrolled_messages") or row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Dataset row must contain uncontrolled_messages or messages")
    return render_and_tokenize_messages(
        tokenizer, messages, thinking_mode=thinking_mode
    ).input_ids


def _model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _hidden_state_layer_index(layer: int) -> int:
    layer = int(layer)
    if layer < 0:
        raise ValueError(f"Layer must be non-negative, got {layer}")
    return layer


def _decoder_block_index(layer: int) -> int:
    layer = int(layer)
    if layer <= 0:
        raise ValueError(
            "Steering layer 0 is the embedding output and cannot be patched with a decoder-block hook"
        )
    return layer - 1


def compute_last_token_probe_sigma(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[SteeringExample],
    steering_vector: SteeringVector,
    thinking_mode: bool = False,
) -> dict[str, Any]:
    """Compute std(h_last @ v_l) across selected examples for one layer."""

    if not examples:
        raise ValueError("Cannot compute steering sigma from zero examples")
    device = _model_input_device(model)
    v = steering_vector.vector.to(device=device, dtype=torch.float32)
    values: list[float] = []
    layer = _hidden_state_layer_index(steering_vector.layer)
    for ex in examples:
        input_ids = input_ids_for_counting_row(
            tokenizer, ex.row, thinking_mode=thinking_mode
        ).to(device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = list(output.hidden_states)
        if layer >= len(hidden_states):
            raise IndexError(
                f"Requested layer {layer}, but model returned {len(hidden_states)} hidden-state entries"
            )
        h_last = hidden_states[layer][0, -1, :].detach().to(dtype=torch.float32)
        if h_last.numel() != v.numel():
            raise ValueError(
                f"Hidden dim {h_last.numel()} does not match steering vector dim {v.numel()}"
            )
        values.append(float(torch.dot(h_last, v).item()))
    values_t = torch.tensor(values, dtype=torch.float32)
    sigma = float(values_t.std(unbiased=False).item()) if values_t.numel() > 1 else 0.0
    return {
        "layer": int(steering_vector.layer),
        "sigma": sigma,
        "num_examples": len(values),
        "projection_mean": float(values_t.mean().item()),
        "projection_std": sigma,
        "projection_values": values,
    }


def compute_needle_span_probe_sigma(
    *,
    model: Any,
    targets: Sequence[NeedleSpanSteeringTarget],
    steering_vector: SteeringVector,
) -> dict[str, Any]:
    """Compute std(h_t @ v_l) over selected needle-span token positions."""

    if not targets:
        raise ValueError("Cannot compute needle-span steering sigma from zero targets")
    device = _model_input_device(model)
    v = steering_vector.vector.to(device=device, dtype=torch.float32)
    layer = _hidden_state_layer_index(steering_vector.layer)
    values: list[float] = []
    positions_by_target: list[dict[str, Any]] = []
    for target in targets:
        input_ids = target.input_ids.to(device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = list(output.hidden_states)
        if layer >= len(hidden_states):
            raise IndexError(
                f"Requested layer {layer}, but model returned {len(hidden_states)} hidden-state entries"
            )
        span = hidden_states[layer][
            0, target.span_start : target.span_end, :
        ].detach().to(dtype=torch.float32)
        if span.ndim != 2 or span.shape[0] != target.span_length:
            raise ValueError(
                f"Invalid span shape {tuple(span.shape)} for target {target.row_id}/{target.needle_id}"
            )
        if span.shape[1] != v.numel():
            raise ValueError(
                f"Hidden dim {span.shape[1]} does not match steering vector dim {v.numel()}"
            )
        projections = span @ v
        values.extend(float(x) for x in projections.detach().cpu().tolist())
        positions_by_target.append(
            {
                "dataset_index": target.dataset_index,
                "row_id": target.row_id,
                "needle_id": target.needle_id,
                "needle_ordinal": target.needle_ordinal,
                "span_start": target.span_start,
                "span_end": target.span_end,
                "span_length": target.span_length,
                "projection_mean": float(projections.mean().item()),
                "projection_std": (
                    float(projections.std(unbiased=False).item())
                    if projections.numel() > 1
                    else 0.0
                ),
            }
        )
    values_t = torch.tensor(values, dtype=torch.float32)
    sigma = float(values_t.std(unbiased=False).item()) if values_t.numel() > 1 else 0.0
    return {
        "layer": int(steering_vector.layer),
        "sigma": sigma,
        "num_targets": len(targets),
        "num_positions": len(values),
        "projection_mean": float(values_t.mean().item()),
        "projection_std": sigma,
        "projection_min": float(values_t.min().item()),
        "projection_max": float(values_t.max().item()),
        "positions_by_target": positions_by_target,
    }


def last_token_projection(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    steering_vector: SteeringVector,
    thinking_mode: bool = False,
) -> float:
    """Return h_last @ v_l for one row without steering."""

    device = _model_input_device(model)
    input_ids = input_ids_for_counting_row(
        tokenizer, row, thinking_mode=thinking_mode
    ).to(device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    with torch.no_grad():
        output = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.long, device=device),
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = list(output.hidden_states)
    layer = _hidden_state_layer_index(steering_vector.layer)
    h_last = hidden_states[layer][0, -1, :].detach().to(dtype=torch.float32)
    v = steering_vector.vector.to(device=h_last.device, dtype=torch.float32)
    return float(torch.dot(h_last, v).item())


def _add_to_last_token_output(output: Any, delta: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        hidden = output[0].clone()
        rest = output[1:]
    else:
        hidden = output.clone()
        rest = None
    hidden[:, -1, :] = hidden[:, -1, :] + delta.to(
        device=hidden.device, dtype=hidden.dtype
    )
    if rest is None:
        return hidden
    return (hidden, *rest)


def _add_to_span_output(
    output: Any, delta: torch.Tensor, *, span_start: int, span_end: int
) -> Any:
    if isinstance(output, tuple):
        hidden = output[0].clone()
        rest = output[1:]
    else:
        hidden = output.clone()
        rest = None
    start = int(span_start)
    end = int(span_end)
    if start < 0 or end <= start or end > hidden.shape[1]:
        raise ValueError(
            f"Invalid steering span start={start} end={end} for sequence length {hidden.shape[1]}"
        )
    hidden[:, start:end, :] = hidden[:, start:end, :] + delta.to(
        device=hidden.device, dtype=hidden.dtype
    )
    if rest is None:
        return hidden
    return (hidden, *rest)


def generate_with_counting_steering(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    layer: int,
    steering_vector: torch.Tensor | None = None,
    beta: float = 0.0,
    sigma: float = 1.0,
    max_new_tokens: int = 20,
) -> str:
    """Greedy decode while adding beta * sigma * v to the current last token."""

    device = _model_input_device(model)
    generated = input_ids.to(device)
    if generated.ndim == 1:
        generated = generated.unsqueeze(0)
    decoder_layers = list(resolve_decoder_layers(model))
    block_idx = _decoder_block_index(int(layer))
    if block_idx >= len(decoder_layers):
        raise IndexError(
            f"Steering layer {layer} maps to decoder block {block_idx}, "
            f"but model has {len(decoder_layers)} decoder blocks"
        )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    delta: torch.Tensor | None = None
    if steering_vector is not None and float(beta) != 0.0 and float(sigma) != 0.0:
        delta = (
            torch.as_tensor(steering_vector, dtype=torch.float32, device=device).flatten()
            * float(beta)
            * float(sigma)
        )

    for _step in range(int(max_new_tokens)):
        hooks = []
        try:
            if delta is not None:

                def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                    return _add_to_last_token_output(output, delta)

                hooks.append(decoder_layers[block_idx].register_forward_hook(hook))
            with torch.no_grad():
                output = model(
                    generated,
                    attention_mask=torch.ones_like(
                        generated, dtype=torch.long, device=generated.device
                    ),
                    use_cache=False,
                )
        finally:
            for handle in hooks:
                handle.remove()
        next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and int(next_token[0, 0]) == int(eos_token_id):
            break
    prompt_len = int(input_ids.reshape(1, -1).shape[1])
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


def generate_with_needle_span_counting_steering(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    layer: int,
    span_start: int,
    span_end: int,
    steering_vector: torch.Tensor | None = None,
    beta: float = 0.0,
    sigma: float = 1.0,
    max_new_tokens: int = 20,
) -> str:
    """Greedy decode while adding beta * sigma * v to one prompt needle span."""

    device = _model_input_device(model)
    generated = input_ids.to(device)
    if generated.ndim == 1:
        generated = generated.unsqueeze(0)
    decoder_layers = list(resolve_decoder_layers(model))
    block_idx = _decoder_block_index(int(layer))
    if block_idx >= len(decoder_layers):
        raise IndexError(
            f"Steering layer {layer} maps to decoder block {block_idx}, "
            f"but model has {len(decoder_layers)} decoder blocks"
        )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    delta: torch.Tensor | None = None
    if steering_vector is not None and float(beta) != 0.0 and float(sigma) != 0.0:
        delta = (
            torch.as_tensor(steering_vector, dtype=torch.float32, device=device).flatten()
            * float(beta)
            * float(sigma)
        )

    for _step in range(int(max_new_tokens)):
        hooks = []
        try:
            if delta is not None:

                def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                    return _add_to_span_output(
                        output, delta, span_start=span_start, span_end=span_end
                    )

                hooks.append(decoder_layers[block_idx].register_forward_hook(hook))
            with torch.no_grad():
                output = model(
                    generated,
                    attention_mask=torch.ones_like(
                        generated, dtype=torch.long, device=generated.device
                    ),
                    use_cache=False,
                )
        finally:
            for handle in hooks:
                handle.remove()
        next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and int(next_token[0, 0]) == int(eos_token_id):
            break
    prompt_len = int(input_ids.reshape(1, -1).shape[1])
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


def _cached_prediction_count(scored_row: dict[str, Any] | None) -> int | None:
    if not scored_row:
        return None
    prediction = scored_row.get("prediction")
    if isinstance(prediction, dict) and prediction.get("count") is not None:
        try:
            return int(prediction["count"])
        except (TypeError, ValueError):
            return None
    return None


def run_counting_steering_sweep(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[SteeringExample],
    layers: Sequence[int],
    probe_dir: str | Path,
    betas: Sequence[float],
    max_new_tokens: int,
    thinking_mode: bool,
    vector_source: str = "ridge",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run layer/beta steering and return detailed rows, summaries, metadata."""

    if not examples:
        raise ValueError("No steering examples selected")
    selected_layers = [int(layer) for layer in layers]
    if not selected_layers:
        raise ValueError("At least one steering layer is required")
    probe_root = Path(probe_dir)
    input_by_key = {
        (ex.group, ex.dataset_index): input_ids_for_counting_row(
            tokenizer, ex.row, thinking_mode=thinking_mode
        )
        for ex in examples
    }
    baseline_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for ex in examples:
        key = (ex.group, ex.dataset_index)
        output_text = generate_with_counting_steering(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_by_key[key],
            layer=selected_layers[0],
            steering_vector=None,
            beta=0.0,
            sigma=0.0,
            max_new_tokens=int(max_new_tokens),
        )
        baseline_by_key[key] = build_response_result(ex.row, output_text)

    detail_rows: list[dict[str, Any]] = []
    sigma_by_layer: dict[int, dict[str, Any]] = {}
    for layer in selected_layers:
        vector = load_counting_steering_vector(
            probe_root, layer=layer, vector_source=vector_source
        )
        sigma_info = compute_last_token_probe_sigma(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            steering_vector=vector,
            thinking_mode=thinking_mode,
        )
        sigma_info["steering_vector_norm_before_unit_normalization"] = vector.norm
        sigma_info["steering_vector_source_space"] = vector.source_space
        sigma_info["standardized_probe"] = vector.standardized_probe
        sigma_by_layer[layer] = sigma_info
        sigma = float(sigma_info["sigma"])
        for beta in [float(x) for x in betas]:
            for ex in examples:
                key = (ex.group, ex.dataset_index)
                steered_text = generate_with_counting_steering(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_by_key[key],
                    layer=layer,
                    steering_vector=vector.vector,
                    beta=beta,
                    sigma=sigma,
                    max_new_tokens=int(max_new_tokens),
                )
                steered = build_response_result(ex.row, steered_text)
                baseline = baseline_by_key[key]
                delta_norm = abs(float(beta) * sigma)
                detail_rows.append(
                    {
                        "layer": layer,
                        "beta": beta,
                        "sigma": sigma,
                        "delta_norm": delta_norm,
                        "expected_projection_shift": float(beta) * sigma,
                        "steering_vector_source_space": vector.source_space,
                        "standardized_probe": vector.standardized_probe,
                        "dataset_index": ex.dataset_index,
                        "row_id": ex.row_id,
                        "example_group": ex.group,
                        "gold_count": ex.row.get("gold_answer", {}).get("count"),
                        "cached_prediction_count": _cached_prediction_count(
                            ex.scored_row
                        ),
                        "cached_exact_match": (
                            None
                            if ex.scored_row is None
                            else bool(ex.scored_row.get("exact_match"))
                        ),
                        "baseline_prediction_count": baseline.get(
                            "prediction", {}
                        ).get("count"),
                        "baseline_exact_match": bool(baseline.get("exact_match")),
                        "baseline_parse_mode": baseline.get("parse_mode"),
                        "baseline_output_text": baseline.get("model_output_text"),
                        "steered_prediction_count": steered.get("prediction", {}).get(
                            "count"
                        ),
                        "steered_exact_match": bool(steered.get("exact_match")),
                        "steered_parse_mode": steered.get("parse_mode"),
                        "steered_output_text": steered.get("model_output_text"),
                    }
                )
    summary_rows = summarize_steering_results(detail_rows)
    metadata = {
        "layers": selected_layers,
        "steering_vector_source": str(vector_source),
        "betas": [float(x) for x in betas],
        "max_new_tokens": int(max_new_tokens),
        "num_examples": len(examples),
        "sigma_by_layer": sigma_by_layer,
    }
    return detail_rows, summary_rows, metadata


def _tqdm(iterable: Any, *, total: int, desc: str) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def run_needle_span_counting_steering_sweep(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[SteeringExample],
    layers: Sequence[int],
    probe_dir: str | Path,
    betas: Sequence[float],
    max_new_tokens: int,
    thinking_mode: bool,
    vector_source: str = "ridge",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run layer/beta/needle-span steering and return details, summaries, metadata."""

    if not examples:
        raise ValueError("No steering examples selected")
    selected_layers = [int(layer) for layer in layers]
    if not selected_layers:
        raise ValueError("At least one steering layer is required")
    targets, target_summary = build_needle_span_steering_targets(
        tokenizer, examples, thinking_mode=thinking_mode
    )
    if not targets:
        raise ValueError("No inserted matching needle spans available for steering")

    probe_root = Path(probe_dir)
    input_by_key: dict[tuple[str, int], torch.Tensor] = {}
    row_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for target in targets:
        key = (target.group, target.dataset_index)
        input_by_key.setdefault(key, target.input_ids)
        row_by_key.setdefault(key, target.row)

    baseline_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for key, input_ids in input_by_key.items():
        output_text = generate_with_counting_steering(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            layer=selected_layers[0],
            steering_vector=None,
            beta=0.0,
            sigma=0.0,
            max_new_tokens=int(max_new_tokens),
        )
        baseline_by_key[key] = build_response_result(row_by_key[key], output_text)

    detail_rows: list[dict[str, Any]] = []
    sigma_by_layer: dict[int, dict[str, Any]] = {}
    tasks = [
        (layer, float(beta), target)
        for layer in selected_layers
        for beta in [float(x) for x in betas]
        for target in targets
    ]
    vector_by_layer: dict[int, SteeringVector] = {}
    for layer in selected_layers:
        vector = load_counting_steering_vector(
            probe_root, layer=layer, vector_source=vector_source
        )
        sigma_info = compute_needle_span_probe_sigma(
            model=model,
            targets=targets,
            steering_vector=vector,
        )
        sigma_info["steering_vector_norm_before_unit_normalization"] = vector.norm
        sigma_info["steering_vector_source_space"] = vector.source_space
        sigma_info["standardized_probe"] = vector.standardized_probe
        sigma_by_layer[layer] = sigma_info
        vector_by_layer[layer] = vector

    for layer, beta, target in _tqdm(
        tasks, total=len(tasks), desc="needle-span steering"
    ):
        vector = vector_by_layer[int(layer)]
        sigma = float(sigma_by_layer[int(layer)]["sigma"])
        steered_text = generate_with_needle_span_counting_steering(
            model=model,
            tokenizer=tokenizer,
            input_ids=target.input_ids,
            layer=int(layer),
            span_start=target.span_start,
            span_end=target.span_end,
            steering_vector=vector.vector,
            beta=beta,
            sigma=sigma,
            max_new_tokens=int(max_new_tokens),
        )
        steered = build_response_result(target.row, steered_text)
        key = (target.group, target.dataset_index)
        baseline = baseline_by_key[key]
        detail_rows.append(
            {
                "layer": int(layer),
                "beta": beta,
                "sigma": sigma,
                "delta_norm": abs(beta * sigma),
                "approx_ridge_count_shift": beta
                * sigma
                * float(vector.norm),
                "steering_vector_source_space": vector.source_space,
                "standardized_probe": vector.standardized_probe,
                "dataset_index": target.dataset_index,
                "row_id": target.row_id,
                "example_group": target.group,
                "needle_id": target.needle_id,
                "needle_ordinal": target.needle_ordinal,
                "needle_span_start": target.span_start,
                "needle_span_end": target.span_end,
                "needle_span_length": target.span_length,
                "gold_count": target.row.get("gold_answer", {}).get("count"),
                "cached_prediction_count": _cached_prediction_count(target.scored_row),
                "cached_exact_match": (
                    None
                    if target.scored_row is None
                    else bool(target.scored_row.get("exact_match"))
                ),
                "baseline_prediction_count": baseline.get("prediction", {}).get(
                    "count"
                ),
                "baseline_exact_match": bool(baseline.get("exact_match")),
                "baseline_parse_mode": baseline.get("parse_mode"),
                "baseline_output_text": baseline.get("model_output_text"),
                "steered_prediction_count": steered.get("prediction", {}).get(
                    "count"
                ),
                "steered_exact_match": bool(steered.get("exact_match")),
                "steered_parse_mode": steered.get("parse_mode"),
                "steered_output_text": steered.get("model_output_text"),
            }
        )

    summary_rows = summarize_needle_span_steering_results(detail_rows)
    metadata = {
        "steering_position_mode": "needle_span",
        "steering_vector_source": str(vector_source),
        "layers": selected_layers,
        "betas": [float(x) for x in betas],
        "max_new_tokens": int(max_new_tokens),
        "num_examples": len(examples),
        "target_summary": target_summary,
        "sigma_by_layer": sigma_by_layer,
    }
    return detail_rows, summary_rows, metadata


def summarize_steering_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["layer"]), float(row["beta"]), str(row["example_group"]))
        grouped.setdefault(key, []).append(dict(row))
    summaries: list[dict[str, Any]] = []
    for (layer, beta, group), group_rows in sorted(grouped.items()):
        n = len(group_rows)
        baseline_correct = sum(1 for row in group_rows if row["baseline_exact_match"])
        steered_correct = sum(1 for row in group_rows if row["steered_exact_match"])
        shifts = []
        for row in group_rows:
            before = row.get("baseline_prediction_count")
            after = row.get("steered_prediction_count")
            if before is not None and after is not None:
                shifts.append(float(after) - float(before))
        summaries.append(
            {
                "layer": layer,
                "beta": beta,
                "example_group": group,
                "num_examples": n,
                "baseline_exact_match_count": baseline_correct,
                "baseline_accuracy": baseline_correct / n if n else 0.0,
                "steered_exact_match_count": steered_correct,
                "steered_accuracy": steered_correct / n if n else 0.0,
                "mean_predicted_count_shift": (
                    sum(shifts) / len(shifts) if shifts else None
                ),
                "baseline_parse_failure_rate": (
                    sum(1 for row in group_rows if row.get("baseline_parse_mode") == "parse_fail")
                    / n
                    if n
                    else 0.0
                ),
                "steered_parse_failure_rate": (
                    sum(1 for row in group_rows if row.get("steered_parse_mode") == "parse_fail")
                    / n
                    if n
                    else 0.0
                ),
            }
        )
    return summaries


def _summarize_group_rows(
    group_rows: Sequence[dict[str, Any]],
    *,
    layer: int,
    beta: float,
    group: str,
    needle_ordinal: int | str,
    needle_id: str,
) -> dict[str, Any]:
    n = len(group_rows)
    baseline_correct = sum(1 for row in group_rows if row["baseline_exact_match"])
    steered_correct = sum(1 for row in group_rows if row["steered_exact_match"])
    shifts = []
    for row in group_rows:
        before = row.get("baseline_prediction_count")
        after = row.get("steered_prediction_count")
        if before is not None and after is not None:
            shifts.append(float(after) - float(before))
    return {
        "layer": layer,
        "beta": beta,
        "example_group": group,
        "needle_ordinal": needle_ordinal,
        "needle_id": needle_id,
        "num_interventions": n,
        "num_examples": len({int(row["dataset_index"]) for row in group_rows}),
        "baseline_exact_match_count": baseline_correct,
        "baseline_accuracy": baseline_correct / n if n else 0.0,
        "steered_exact_match_count": steered_correct,
        "steered_accuracy": steered_correct / n if n else 0.0,
        "mean_predicted_count_shift": (
            sum(shifts) / len(shifts) if shifts else None
        ),
        "baseline_parse_failure_rate": (
            sum(1 for row in group_rows if row.get("baseline_parse_mode") == "parse_fail")
            / n
            if n
            else 0.0
        ),
        "steered_parse_failure_rate": (
            sum(1 for row in group_rows if row.get("steered_parse_mode") == "parse_fail")
            / n
            if n
            else 0.0
        ),
    }


def summarize_needle_span_steering_results(
    rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Summarize needle-span steering per needle ordinal and over all needles."""

    by_needle: dict[tuple[int, float, str, int, str], list[dict[str, Any]]] = {}
    aggregate: dict[tuple[int, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        layer = int(row["layer"])
        beta = float(row["beta"])
        group = str(row["example_group"])
        needle_ordinal = int(row["needle_ordinal"])
        needle_id = str(row["needle_id"])
        by_needle.setdefault(
            (layer, beta, group, needle_ordinal, needle_id), []
        ).append(dict(row))
        aggregate.setdefault((layer, beta, group), []).append(dict(row))

    summaries: list[dict[str, Any]] = []
    for (layer, beta, group), group_rows in sorted(aggregate.items()):
        summaries.append(
            _summarize_group_rows(
                group_rows,
                layer=layer,
                beta=beta,
                group=group,
                needle_ordinal="all",
                needle_id="all",
            )
        )
    for (layer, beta, group, needle_ordinal, needle_id), group_rows in sorted(
        by_needle.items()
    ):
        summaries.append(
            _summarize_group_rows(
                group_rows,
                layer=layer,
                beta=beta,
                group=group,
                needle_ordinal=needle_ordinal,
                needle_id=needle_id,
            )
        )
    return summaries


def write_jsonl(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path
