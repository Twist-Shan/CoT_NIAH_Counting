from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _bounded_logits_kwargs,
    _is_prompt_prefill,
    _replace_output_tensor,
    _tensor_from_output,
)
from realistic_niah_v4.prompts import PromptEncoding


@dataclass(frozen=True)
class CorruptionPlan:
    """Equal-token-budget active-needle and ordinary-passage corruptions."""

    needle_targets: tuple[tuple[int, int], ...]
    needle_sources: tuple[tuple[int, int], ...]
    ordinary_targets: tuple[tuple[int, int], ...]
    ordinary_sources: tuple[tuple[int, int], ...]

    @property
    def token_budget(self) -> int:
        return sum(end - start for start, end in self.needle_targets)


def _positions(segments: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(
        position
        for start, end in segments
        for position in range(int(start), int(end))
    )


def build_corruption_plan(encoding: PromptEncoding) -> CorruptionPlan:
    """Allocate three disjoint ordinary segment banks near the registered slots.

    The first bank supplies replacement tokens for active needle spans. The
    second bank is the matched ordinary target, and the third supplies its
    replacement tokens. All target/source segments are disjoint from every
    slot and hard-negative span and match the per-needle token lengths.
    """

    needle_targets = tuple(
        (int(span.start), int(span.end)) for span in encoding.needle_spans
    )
    if not needle_targets:
        raise ValueError("Restoration requires at least one active needle span")
    lengths = tuple(end - start for start, end in needle_targets)
    if any(length <= 0 for length in lengths):
        raise ValueError("Needle spans must be nonempty")
    all_registered = tuple(encoding.slot_spans) + tuple(encoding.hard_negative_spans)
    if not all_registered:
        raise ValueError("No registered slot/hard-negative spans are available")
    forbidden: set[int] = set()
    for span in all_registered:
        forbidden.update(range(int(span.start), int(span.end)))
    lower = max(1, min(int(span.start) for span in all_registered) - 64)
    upper = min(
        int(encoding.query_position),
        max(int(span.end) for span in all_registered) + 64,
    )
    used = set(forbidden)

    def allocate(length: int, phase: int) -> tuple[int, int]:
        stride = max(1, int(length) // 2)
        stop = max(lower + phase, upper - int(length) + 1)
        for candidate in range(lower + phase, stop, stride):
            segment = set(range(candidate, candidate + int(length)))
            if candidate + int(length) <= upper and not segment.intersection(used):
                used.update(segment)
                return int(candidate), int(candidate + length)
        for candidate in range(lower, upper - int(length) + 1):
            segment = set(range(candidate, candidate + int(length)))
            if not segment.intersection(used):
                used.update(segment)
                return int(candidate), int(candidate + length)
        raise RuntimeError(
            "Could not allocate a length-matched ordinary passage segment"
        )

    needle_sources = tuple(allocate(length, 0) for length in lengths)
    ordinary_targets = tuple(allocate(length, 1) for length in lengths)
    ordinary_sources = tuple(allocate(length, 2) for length in lengths)
    plan = CorruptionPlan(
        needle_targets=needle_targets,
        needle_sources=needle_sources,
        ordinary_targets=ordinary_targets,
        ordinary_sources=ordinary_sources,
    )
    banks = (
        _positions(plan.needle_targets),
        _positions(plan.needle_sources),
        _positions(plan.ordinary_targets),
        _positions(plan.ordinary_sources),
    )
    if any(len(values) != plan.token_budget for values in banks):
        raise RuntimeError("Corruption banks do not share the token budget")
    if set(banks[1]).intersection(banks[2]) or set(banks[1]).intersection(banks[3]):
        raise RuntimeError("Ordinary corruption banks overlap")
    if set(banks[2]).intersection(banks[3]):
        raise RuntimeError("Ordinary target/source banks overlap")
    return plan


def _replace_segments(
    input_ids: Sequence[int],
    targets: Sequence[tuple[int, int]],
    sources: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], int]:
    if len(targets) != len(sources):
        raise ValueError("Target/source segment counts differ")
    original = tuple(int(value) for value in input_ids)
    changed = list(original)
    changed_tokens = 0
    for (target_start, target_end), (source_start, source_end) in zip(
        targets, sources
    ):
        target_length = int(target_end) - int(target_start)
        source = original[int(source_start) : int(source_end)]
        if len(source) != target_length:
            raise RuntimeError("A replacement segment changed token length")
        before = changed[int(target_start) : int(target_end)]
        changed[int(target_start) : int(target_end)] = source
        changed_tokens += sum(left != right for left, right in zip(before, source))
    return tuple(changed), int(changed_tokens)


def corrupt_encoding(
    encoding: PromptEncoding,
    plan: CorruptionPlan,
    *,
    condition: str,
) -> tuple[PromptEncoding, int]:
    """Return a length-preserving needle or ordinary-passage corruption."""

    if condition == "needle_corrupt":
        targets, sources = plan.needle_targets, plan.needle_sources
    elif condition == "ordinary_corrupt":
        targets, sources = plan.ordinary_targets, plan.ordinary_sources
    else:
        raise ValueError("condition must be needle_corrupt or ordinary_corrupt")
    input_ids, changed = _replace_segments(encoding.input_ids, targets, sources)
    if len(input_ids) != int(encoding.sequence_length):
        raise RuntimeError("Corruption changed prompt length")
    return replace(encoding, input_ids=input_ids), changed


def segment_positions(
    plan: CorruptionPlan, *, condition: str, endpoint_only: bool = False
) -> tuple[int, ...]:
    if condition == "needle":
        segments = plan.needle_targets
    elif condition == "ordinary":
        segments = plan.ordinary_targets
    else:
        raise ValueError("condition must be needle or ordinary")
    if endpoint_only:
        return tuple(int(end) - 1 for _start, end in segments)
    return _positions(segments)


def single_segment_positions(
    plan: CorruptionPlan, *, condition: str, segment_index: int
) -> tuple[int, ...]:
    segments = (
        plan.needle_targets if condition == "needle" else plan.ordinary_targets
    )
    index = int(segment_index)
    if not 0 <= index < len(segments):
        raise IndexError("segment_index lies outside the corruption plan")
    return _positions((segments[index],))


@contextmanager
def residual_patch_hook(
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    positions: Sequence[int],
    replacement: torch.Tensor,
) -> Iterator[dict[str, int]]:
    """Patch selected post-block positions in full/prefix forwards only."""

    layer = int(layer)
    if not 0 <= layer < adapter.num_layers:
        raise ValueError(f"Invalid patch layer: {layer}")
    selected = tuple(int(position) for position in positions)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Patch positions must be nonempty and unique")
    states = replacement
    if states.ndim == 1:
        states = states.unsqueeze(0)
    if states.ndim != 2 or int(states.shape[0]) != len(selected):
        raise ValueError("replacement must have shape [positions, hidden]")
    applications = {"count": 0}

    def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = _tensor_from_output(output)
        # Applies both to the full candidate-scoring prefill and to the prefix
        # used for a one-row attention reconstruction. It skips continuation
        # decoding, whose time dimension is too short to contain the targets.
        if hidden.ndim != 3 or int(hidden.shape[1]) <= max(selected):
            return output
        if int(hidden.shape[1]) > int(encoding.sequence_length):
            raise RuntimeError("Patch hook observed a sequence longer than the prompt")
        target = states.to(device=hidden.device, dtype=hidden.dtype)
        if int(target.shape[-1]) != int(hidden.shape[-1]):
            raise RuntimeError("Patch hidden width differs from model hidden width")
        patched = hidden.clone()
        patched[:, list(selected), :] = target.unsqueeze(0)
        applications["count"] += 1
        return _replace_output_tensor(output, patched)

    handle = adapter.layers[layer].register_forward_hook(hook)
    try:
        yield applications
    finally:
        handle.remove()


def _generation_eos_ids(model: nn.Module, tokenizer: Any) -> list[int]:
    generation_config = getattr(model, "generation_config", None)
    value = (
        getattr(generation_config, "eos_token_id", None)
        if generation_config is not None
        else None
    )
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, (tuple, list, set)):
        return [int(item) for item in value]
    return [int(value)]


@torch.inference_mode()
def generate_answer_completion_from_prefill(
    model: nn.Module,
    tokenizer: Any,
    encoding: PromptEncoding,
    prefill_output: Any,
    *,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Greedily decode from an already patched full-prompt KV cache.

    The first token comes directly from the retained prompt logits.  Each
    later token is a one-token cached forward, so a residual patch hook whose
    target positions lie in the prompt is not re-applied during decoding.
    """

    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be positive")
    logits = getattr(prefill_output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Reusable prefill exposes no [batch,time,vocab] logits")
    if int(logits.shape[0]) != 1:
        raise RuntimeError("Reusable strict generation requires batch size one")
    past = getattr(prefill_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Reusable strict generation received no KV cache")
    eos_ids = _generation_eos_ids(model, tokenizer)
    eos_set = set(eos_ids)
    continuation: list[int] = []
    next_logits = logits[0, -1].detach().float()
    base_mask = torch.tensor(
        [encoding.attention_mask], dtype=torch.long, device=next_logits.device
    )
    shared = getattr(prefill_output, "shared_kv_states", None)
    for step in range(int(max_new_tokens)):
        token = int(torch.argmax(next_logits).item())
        continuation.append(token)
        if token in eos_set:
            break
        device = next_logits.device
        token_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        generated_mask = torch.ones(
            (1, len(continuation)), dtype=base_mask.dtype, device=base_mask.device
        )
        attention_mask = torch.cat((base_mask, generated_mask), dim=1)
        position = int(encoding.sequence_length) + int(step)
        kwargs: dict[str, Any] = {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": True,
            **_bounded_logits_kwargs(model),
        }
        if _accepts_keyword(model, "position_ids"):
            kwargs["position_ids"] = torch.tensor(
                [[position]], dtype=torch.long, device=device
            )
        if _accepts_keyword(model, "cache_position"):
            kwargs["cache_position"] = torch.tensor(
                [position], dtype=torch.long, device=device
            )
        if shared is not None and _accepts_keyword(model, "shared_kv_states"):
            kwargs["shared_kv_states"] = shared
        output = model(**kwargs)
        updated_past = getattr(output, "past_key_values", None)
        if updated_past is None:
            raise RuntimeError("Cached strict-generation step returned no KV cache")
        past = updated_past
        updated_shared = getattr(output, "shared_kv_states", None)
        if updated_shared is not None:
            shared = updated_shared
        step_logits = getattr(output, "logits", None)
        if not isinstance(step_logits, torch.Tensor) or step_logits.ndim != 3:
            raise RuntimeError("Cached strict-generation step returned no logits")
        next_logits = step_logits[0, -1].detach().float()
    stopped_on_eos = bool(eos_ids and continuation[-1] in eos_set)
    raw_text = tokenizer.decode(
        continuation,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    clean_text = tokenizer.decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "generated_token_ids": continuation,
        "generated_token_count": len(continuation),
        "generation_eos_token_ids": eos_ids,
        "stopped_on_eos": stopped_on_eos,
        "generation_truncated": bool(
            len(continuation) >= int(max_new_tokens) and not stopped_on_eos
        ),
        "completion_text_raw": str(raw_text),
        "completion_text": str(clean_text),
        "full_answer_text": "Total:" + str(clean_text),
    }


def active_broad_metrics(
    attention_row: torch.Tensor,
    *,
    key_start: int,
    spans: Sequence[Any],
) -> list[dict[str, Any]]:
    """Compute per-head active mass, entropy coverage, and broad score."""

    row = attention_row.detach().float().cpu()
    if row.ndim != 2:
        raise ValueError("attention_row must have shape [heads, keys]")
    registered = tuple((int(span.start), int(span.end)) for span in spans)
    if not registered:
        raise ValueError("At least one active span is required")
    result: list[dict[str, Any]] = []
    for head in range(int(row.shape[0])):
        masses: list[float] = []
        for start, end in registered:
            local_start = max(0, start - int(key_start))
            local_end = min(int(row.shape[1]), end - int(key_start))
            mass = (
                float(row[head, local_start:local_end].sum())
                if local_start < local_end
                else 0.0
            )
            masses.append(mass)
        total = float(sum(masses))
        if total > 0:
            probabilities = [mass / total for mass in masses if mass > 0]
            entropy = -sum(value * math.log(value) for value in probabilities)
            coverage = math.exp(entropy) / len(masses)
        else:
            coverage = 0.0
        result.append(
            {
                "head": int(head),
                "needle_mass": total,
                "coverage": float(coverage),
                "broad_score": float(total * coverage),
                "span_masses": tuple(float(value) for value in masses),
            }
        )
    return result


def normalized_recovery(clean: float, corrupted: float, restored: float) -> float:
    """Fraction of the clean-minus-corrupted expected-count displacement restored."""

    denominator = float(clean) - float(corrupted)
    if abs(denominator) <= 1e-8:
        return math.nan
    return (float(restored) - float(corrupted)) / denominator
