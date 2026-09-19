from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _bounded_logits_kwargs,
    _encoding_tensors,
    query_attention_outputs,
)
from realistic_niah_v4.prompts import PromptEncoding

from .geometry import (
    deterministic_orthogonal_direction,
    query_to_kv_head,
    resolve_value_source_layer,
)


@dataclass(frozen=True)
class CausalOutput:
    logits: torch.Tensor
    candidate_log_scores: dict[int, float]
    attention_output: torch.Tensor | None = None


@dataclass(frozen=True)
class QueryBundle:
    logits: torch.Tensor
    candidate_log_scores: dict[int, float]
    z_by_layer: dict[int, torch.Tensor]
    value_by_layer: dict[int, torch.Tensor]
    attention_output_by_layer: dict[int, torch.Tensor]
    alpha_by_layer: dict[int, torch.Tensor]
    alpha_key_start_by_layer: dict[int, int]
    attention_cache_candidate_logit_max_abs_delta: float
    attention_cache_candidate_centered_logit_max_abs_delta: float
    attention_cache_candidate_argmax_agreement: bool = True
    attention_cache_candidate_probability_total_variation: float = 0.0
    attention_cache_equivalence_audited: bool = True
    reusable_prefill_output: Any | None = None


def _output_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Model forward did not return [batch, time, vocab] logits")
    return logits


def _repeat_batch_tree(value: Any, repeats: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == 1:
            return value.repeat_interleave(int(repeats), dim=0)
        return value
    if isinstance(value, tuple):
        return tuple(_repeat_batch_tree(item, repeats) for item in value)
    if isinstance(value, list):
        return [_repeat_batch_tree(item, repeats) for item in value]
    if isinstance(value, dict):
        return {
            key: _repeat_batch_tree(item, repeats) for key, item in value.items()
        }
    return value


def _clone_tensor_tree(value: Any) -> Any:
    """Clone tensors in a small auxiliary-state tree without changing type."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    return copy.deepcopy(value)


def clone_prefill_output_for_scoring(prefill_output: Any) -> Any:
    """Return an isolated cache branch for destructive candidate scoring.

    ``_score_candidate_sequences`` expands the cache batch from one prompt to
    ten candidate continuations in place.  A retained batch-1 prompt cache can
    therefore only be reused for strict generation when scoring receives a
    deep-copied cache object.  Logits are read-only and remain shared.
    """

    past = getattr(prefill_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Reusable candidate prefill returned no KV cache")
    try:
        cloned_past = copy.deepcopy(past)
    except Exception as exc:  # pragma: no cover - model/cache-version specific
        raise RuntimeError(
            "Transformers KV cache cannot be cloned for reusable prefill scoring"
        ) from exc
    if cloned_past is past:
        raise RuntimeError("KV cache deepcopy unexpectedly returned the source object")
    values: dict[str, Any] = {
        "logits": _output_logits(prefill_output),
        "past_key_values": cloned_past,
    }
    shared = getattr(prefill_output, "shared_kv_states", None)
    if shared is not None:
        values["shared_kv_states"] = _clone_tensor_tree(shared)
    return SimpleNamespace(**values)


@torch.inference_mode()
def _score_candidate_sequences(
    model: nn.Module,
    encoding: PromptEncoding,
    prefill_output: Any,
) -> CausalOutput:
    """Score all ten registered answer+termination sequences from one KV cache."""

    prefill_logits = _output_logits(prefill_output)[0, -1].detach().float()
    past = getattr(prefill_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Candidate scoring prefill returned no KV cache")
    candidates = sorted(
        (int(count), tuple(int(token) for token in tokens))
        for count, tokens in encoding.count_candidate_token_ids
    )
    if [count for count, _tokens in candidates] != list(range(1, 11)):
        raise RuntimeError("V4.4.3 expects candidate counts exactly 1 through 10")
    if any(len(tokens) < 2 for _count, tokens in candidates):
        raise RuntimeError("Every candidate must include answer and termination tokens")
    if len({tokens for _count, tokens in candidates}) != len(candidates):
        raise RuntimeError("Candidate token sequences must be unique")
    batch = len(candidates)
    max_inputs = max(len(tokens) - 1 for _count, tokens in candidates)
    device = prefill_logits.device
    continuation_ids = torch.zeros(
        (batch, max_inputs), dtype=torch.long, device=device
    )
    continuation_mask = torch.zeros_like(continuation_ids)
    for row, (_count, tokens) in enumerate(candidates):
        inputs = tokens[:-1]
        continuation_ids[row, : len(inputs)] = torch.tensor(
            inputs, dtype=torch.long, device=device
        )
        continuation_mask[row, : len(inputs)] = 1
    repeater = getattr(past, "batch_repeat_interleave", None)
    if not callable(repeater):
        raise RuntimeError("Transformers cache cannot branch candidate continuations")
    repeater(batch)
    base_mask = torch.tensor(
        [encoding.attention_mask], dtype=torch.long, device=device
    ).repeat(batch, 1)
    attention_mask = torch.cat((base_mask, continuation_mask), dim=1)
    kwargs: dict[str, Any] = {
        "input_ids": continuation_ids,
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": False,
    }
    positions = torch.arange(
        encoding.sequence_length,
        encoding.sequence_length + max_inputs,
        dtype=torch.long,
        device=device,
    )
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = positions.unsqueeze(0).expand(batch, -1)
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = positions
    shared = getattr(prefill_output, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = _repeat_batch_tree(shared, batch)
    continuation_output = model(**kwargs)
    continuation_logits = _output_logits(continuation_output).detach().float()
    if continuation_logits.shape[:2] != (batch, max_inputs):
        raise RuntimeError("Candidate continuation logits have the wrong shape")
    first_log_probs = torch.log_softmax(prefill_logits, dim=-1)
    continuation_log_probs = torch.log_softmax(continuation_logits, dim=-1)
    scores: dict[int, float] = {}
    for row, (count, tokens) in enumerate(candidates):
        score = first_log_probs[tokens[0]]
        for target_offset, token in enumerate(tokens[1:]):
            score = score + continuation_log_probs[row, target_offset, token]
        scores[count] = float(score.detach().cpu())
    return CausalOutput(
        logits=prefill_logits.detach().float().cpu(),
        candidate_log_scores=scores,
    )


def candidate_sequence_metrics(
    candidate_log_scores: Mapping[int, float],
    encoding: PromptEncoding,
) -> dict[str, Any]:
    counts = np.asarray(sorted(int(count) for count in candidate_log_scores), dtype=float)
    if counts.tolist() != list(range(1, 11)):
        raise ValueError("Candidate scores must cover counts 1 through 10")
    scores = np.asarray(
        [float(candidate_log_scores[int(count)]) for count in counts], dtype=float
    )
    if not np.isfinite(scores).all():
        raise ValueError("Candidate sequence scores must be finite")
    shifted = scores - float(scores.max())
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    correct_index = int(np.flatnonzero(counts == encoding.count)[0])
    other = np.delete(scores, correct_index)
    return {
        "gold_count": int(encoding.count),
        "predicted_count_among_candidates": int(counts[int(scores.argmax())]),
        "correct_count_log_score": float(scores[correct_index]),
        "correct_count_margin": float(scores[correct_index] - other.max()),
        "correct_count_probability": float(probabilities[correct_index]),
        "expected_count": float(np.sum(probabilities * counts)),
        "candidate_counts": ",".join(str(int(value)) for value in counts),
        "candidate_log_scores": ",".join(f"{value:.9g}" for value in scores),
        "candidate_probabilities": ",".join(
            f"{value:.9g}" for value in probabilities
        ),
    }


def _tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise RuntimeError(f"Unsupported module output: {type(output).__name__}")


def _replace_output_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise RuntimeError(
        "Attention output replacement supports tensor/tuple/list outputs only"
    )


def _v_projection(attention: nn.Module) -> nn.Module:
    module = getattr(attention, "v_proj", None)
    if not isinstance(module, nn.Module):
        raise RuntimeError(f"{type(attention).__name__} exposes no v_proj")
    return module


def _value_capture_module(
    adapter: DecoderAdapter, layer: int
) -> tuple[int, nn.Module]:
    source = resolve_value_source_layer(adapter, int(layer))
    attention = adapter.attentions[source]
    value_norm = getattr(attention, "v_norm", None)
    if isinstance(value_norm, nn.Module):
        return source, value_norm
    return source, _v_projection(attention)


def candidate_logit_cache_deltas(
    full_logits: torch.Tensor,
    cached_logits: torch.Tensor,
    candidate_ids: Sequence[int],
) -> tuple[float, float]:
    """Return raw and common-shift-invariant candidate-logit discrepancies."""

    ids = [int(value) for value in candidate_ids]
    if not ids:
        raise ValueError("Candidate logit audit needs at least one token ID")
    full = full_logits.detach().float().cpu()[ids]
    cached = cached_logits.detach().float().cpu()[ids]
    if full.shape != cached.shape or full.ndim != 1:
        raise ValueError("Candidate logit audit received incompatible tensors")
    raw = float(torch.max(torch.abs(full - cached)))
    centered = float(
        torch.max(torch.abs((full - full.mean()) - (cached - cached.mean())))
    )
    return raw, centered


def candidate_logit_cache_distribution_diagnostics(
    full_logits: torch.Tensor,
    cached_logits: torch.Tensor,
    candidate_ids: Sequence[int],
) -> tuple[bool, float]:
    """Compare unique first-token candidate distributions after softmax.

    This supplements the conservative max-logit audit with two scale-free
    quantities. It is diagnostic only: the causal score still comes from the
    full-prompt forward, never from the attention-reconstruction forward.
    """

    ids = sorted({int(value) for value in candidate_ids})
    if not ids:
        raise ValueError("Candidate distribution audit needs at least one token ID")
    full = full_logits.detach().float().cpu()[ids]
    cached = cached_logits.detach().float().cpu()[ids]
    full_probability = torch.softmax(full, dim=0)
    cached_probability = torch.softmax(cached, dim=0)
    agreement = bool(int(torch.argmax(full)) == int(torch.argmax(cached)))
    total_variation = float(
        0.5 * torch.sum(torch.abs(full_probability - cached_probability))
    )
    return agreement, total_variation


def _single_query_attention_row(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if not tensors:
            raise RuntimeError("Selected attention output contains no tensor")
        tensor = tensors[0]
    else:
        raise RuntimeError(
            f"Unsupported selected attention output: {type(value).__name__}"
        )
    if tensor.ndim == 5:
        batch, heads, blocks, queries, keys = tensor.shape
        if blocks * queries != 1:
            raise RuntimeError("Selected attention returned multiple query cells")
        tensor = tensor.reshape(batch, heads, 1, keys)
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[2] != 1:
        raise RuntimeError(
            f"Expected one [batch,head,query,key] row, got {tuple(tensor.shape)}"
        )
    return tensor[0, :, 0].detach().float().cpu()


@torch.inference_mode()
def selective_query_attention_outputs(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layers: Sequence[int],
) -> tuple[dict[int, torch.Tensor], dict[int, int], torch.Tensor]:
    """Capture only registered target rows while all other layers stay native.

    The earlier V4 helper switches every decoder layer to eager attention for
    the one-token query. That is unnecessary here and can accumulate a visible
    backend discrepancy in deep Gemma4. Each target module receives a shallow
    config copy with eager attention; every unselected layer remains on the
    model's configured backend.
    """

    selected = tuple(sorted({int(layer) for layer in layers}))
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    query = int(encoding.query_position)
    if query != int(encoding.sequence_length) - 1 or not 0 < query < input_ids.shape[1]:
        raise ValueError("V4.4.3 attention capture requires the final query token")
    prefix_output = model(
        input_ids=input_ids[:, :query],
        attention_mask=attention_mask[:, :query],
        use_cache=True,
        output_attentions=False,
        **_bounded_logits_kwargs(model),
    )
    past = getattr(prefix_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Selective attention prefix returned no KV cache")
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": attention_mask[:, : query + 1],
        "past_key_values": past,
        "use_cache": False,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.tensor(
            [[query]], dtype=torch.long, device=input_ids.device
        )
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.tensor(
            [query], dtype=torch.long, device=input_ids.device
        )
    shared = getattr(prefix_output, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = shared
    original_configs: dict[int, Any] = {}
    captured_rows: dict[int, torch.Tensor] = {}
    handles = []
    for layer in selected:
        attention = adapter.attentions[layer]
        original = getattr(attention, "config", None)
        if original is None or not hasattr(original, "_attn_implementation"):
            raise RuntimeError("Selected attention exposes no backend config")
        patched = copy.copy(original)
        patched._attn_implementation = "eager"
        original_configs[layer] = original
        attention.config = patched

        def capture_hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError("Selected attention returned no weight slot")
            captured_rows[layer] = _single_query_attention_row(output[1])

        handles.append(attention.register_forward_hook(capture_hook))
    try:
        query_output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
        for layer, original in original_configs.items():
            adapter.attentions[layer].config = original
    rows: dict[int, torch.Tensor] = {}
    key_starts: dict[int, int] = {}
    for layer in selected:
        if layer not in captured_rows:
            raise RuntimeError(f"Selected layer {layer} produced no attention row")
        row = captured_rows[layer]
        if row.shape[0] != adapter.num_heads[layer]:
            raise RuntimeError("Selected query attention head count mismatch")
        key_start = query + 1 - int(row.shape[-1])
        if key_start < 0:
            raise RuntimeError("Selected query attention key axis is too long")
        rows[layer] = row
        key_starts[layer] = key_start
    return rows, key_starts, _output_logits(query_output)[0, -1].detach().float().cpu()


@torch.inference_mode()
def capture_query_bundle(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layers: Sequence[int],
    capture_attention: bool,
    capture_values: bool = True,
    cache_logit_tolerance: float = 0.5,
    cache_probability_total_variation_tolerance: float | None = None,
    require_cache_candidate_argmax_agreement: bool = False,
    audit_cache_equivalence: bool = True,
    retain_prefill_output: bool = False,
) -> QueryBundle:
    selected = tuple(sorted({int(layer) for layer in layers}))
    if not selected:
        raise ValueError("At least one bundle layer is required")
    if any(not 0 <= layer < adapter.num_layers for layer in selected):
        raise ValueError("A bundle layer is out of range")
    z_by_layer: dict[int, torch.Tensor] = {}
    value_by_layer: dict[int, torch.Tensor] = {}
    attention_output_by_layer: dict[int, torch.Tensor] = {}
    z_calls = {layer: 0 for layer in selected}
    value_calls = {layer: 0 for layer in selected}
    attention_output_calls = {layer: 0 for layer in selected}
    handles = []
    for layer in selected:
        expected_width = adapter.num_heads[layer] * adapter.head_dims[layer]

        def z_hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            expected_width: int = expected_width,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("o_proj received no positional tensor")
            value = args[0]
            if value.ndim != 3 or value.shape[-1] != expected_width:
                raise RuntimeError(
                    f"Unexpected pre-O aggregate shape at layer {layer}: {value.shape}"
                )
            if value.shape[1] != encoding.sequence_length:
                raise RuntimeError("Bundle capture requires a full-prompt forward")
            z_by_layer[layer] = (
                value[0, encoding.query_position].detach().float().cpu()
            )
            z_calls[layer] += 1

        def value_hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            value = _tensor_from_output(output)
            if value.ndim not in (3, 4) or value.shape[0] != 1:
                raise RuntimeError(
                    f"Unexpected V projection shape at layer {layer}: {value.shape}"
                )
            if value.shape[1] != encoding.sequence_length:
                raise RuntimeError("V capture requires a full-prompt forward")
            if value.ndim == 4:
                if value.shape[-1] != adapter.head_dims[layer]:
                    raise RuntimeError("Normalized V capture has the wrong head width")
                value = value.flatten(start_dim=-2)
            value_by_layer[layer] = value[0].detach().to(
                device="cpu", dtype=torch.float16
            )
            value_calls[layer] += 1

        def attention_output_hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            value = _tensor_from_output(output)
            if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
                raise RuntimeError("Attention-output capture requires full prefill")
            attention_output_by_layer[layer] = (
                value[0, encoding.query_position].detach().float().cpu()
            )
            attention_output_calls[layer] += 1

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(z_hook)
        )
        if capture_values:
            _source_layer, capture_module = _value_capture_module(adapter, layer)
            handles.append(
                capture_module.register_forward_hook(value_hook)
            )
        handles.append(
            adapter.attentions[layer].register_forward_hook(attention_output_hook)
        )
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    violations = {
        "z": {layer: count for layer, count in z_calls.items() if count != 1},
        "value": (
            {layer: count for layer, count in value_calls.items() if count != 1}
            if capture_values
            else {}
        ),
        "attention_output": {
            layer: count
            for layer, count in attention_output_calls.items()
            if count != 1
        },
    }
    if violations["z"] or violations["value"] or violations["attention_output"]:
        raise RuntimeError(f"Bundle hook application mismatch: {violations}")
    scoring_output = (
        clone_prefill_output_for_scoring(prefill_output)
        if retain_prefill_output
        else prefill_output
    )
    causal_output = _score_candidate_sequences(model, encoding, scoring_output)
    logits = causal_output.logits
    alpha_by_layer: dict[int, torch.Tensor] = {}
    alpha_key_start: dict[int, int] = {}
    cache_logit_delta = 0.0
    cache_centered_logit_delta = 0.0
    argmax_agreement = True
    probability_total_variation = 0.0
    if capture_attention:
        if audit_cache_equivalence:
            rows, key_starts, cached_logits = selective_query_attention_outputs(
                model,
                adapter,
                encoding,
                layers=selected,
            )
            candidate_ids = sorted(
                {
                    int(token_ids[0])
                    for _count, token_ids in encoding.count_candidate_answer_token_ids
                }
            )
            delta, centered_delta = candidate_logit_cache_deltas(
                logits, cached_logits, candidate_ids
            )
            argmax_agreement, probability_total_variation = (
                candidate_logit_cache_distribution_diagnostics(
                    logits, cached_logits, candidate_ids
                )
            )
            cache_logit_delta = delta
            cache_centered_logit_delta = centered_delta
            if (
                not math.isfinite(centered_delta)
                or centered_delta > float(cache_logit_tolerance)
            ):
                raise RuntimeError(
                    "Full/cache relative candidate logits disagree before intervention: "
                    f"centered={centered_delta}, raw={delta}"
                )
            if require_cache_candidate_argmax_agreement and not argmax_agreement:
                raise RuntimeError(
                    "Full/cache first-token candidate argmax disagrees before "
                    f"intervention; total_variation={probability_total_variation}"
                )
            if (
                cache_probability_total_variation_tolerance is not None
                and (
                    not math.isfinite(probability_total_variation)
                    or probability_total_variation
                    > float(cache_probability_total_variation_tolerance)
                )
            ):
                raise RuntimeError(
                    "Full/cache first-token candidate probability distributions "
                    "disagree before intervention: "
                    f"total_variation={probability_total_variation}, "
                    f"tolerance={cache_probability_total_variation_tolerance}"
                )
        else:
            # Match the original V4 attention atlas: evaluate the long prefix
            # with the configured backend, run only the final answer-query
            # token with eager attention, and use those cache-reconstructed
            # query rows without comparing them with the causal full forward.
            all_rows, all_key_starts, _query_logits = query_attention_outputs(
                model, adapter, encoding
            )
            rows = {layer: all_rows[layer] for layer in selected}
            key_starts = {layer: all_key_starts[layer] for layer in selected}
        for layer in selected:
            alpha_by_layer[layer] = rows[layer].detach().to(
                device="cpu", dtype=torch.float32
            )
            alpha_key_start[layer] = int(key_starts[layer])
    return QueryBundle(
        logits=logits,
        candidate_log_scores=causal_output.candidate_log_scores,
        z_by_layer=z_by_layer,
        value_by_layer=value_by_layer,
        attention_output_by_layer=attention_output_by_layer,
        alpha_by_layer=alpha_by_layer,
        alpha_key_start_by_layer=alpha_key_start,
        attention_cache_candidate_logit_max_abs_delta=cache_logit_delta,
        attention_cache_candidate_centered_logit_max_abs_delta=(
            cache_centered_logit_delta
        ),
        attention_cache_candidate_argmax_agreement=(
            argmax_agreement if capture_attention else True
        ),
        attention_cache_candidate_probability_total_variation=(
            probability_total_variation if capture_attention else 0.0
        ),
        attention_cache_equivalence_audited=(
            bool(audit_cache_equivalence) if capture_attention else False
        ),
        reusable_prefill_output=(prefill_output if retain_prefill_output else None),
    )


def head_z(
    bundle: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    head: int,
) -> torch.Tensor:
    width = int(adapter.head_dims[layer])
    start = int(head) * width
    value = bundle.z_by_layer[layer]
    if not 0 <= start < start + width <= len(value):
        raise ValueError("Head Z slice lies outside the captured aggregate")
    return value[start : start + width].float()


@torch.inference_mode()
def head_output_from_z(
    adapter: DecoderAdapter,
    *,
    layer: int,
    head: int,
    z: torch.Tensor,
) -> torch.Tensor:
    projection = adapter.output_projections[layer]
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError("Output projection exposes no matrix weight")
    width = int(adapter.head_dims[layer])
    start = int(head) * width
    if z.ndim != 1 or len(z) != width:
        raise ValueError("Head Z has the wrong width")
    result = weight[:, start : start + width].float() @ z.to(
        device=weight.device, dtype=torch.float32
    )
    return result.detach().float().cpu()


def alpha_receiver_v_z(
    donor: QueryBundle,
    receiver: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    head: int,
    alpha_override: torch.Tensor | None = None,
    key_start_override: int | None = None,
) -> torch.Tensor:
    if layer not in donor.alpha_by_layer:
        raise ValueError("Donor bundle has no attention row")
    row = (
        donor.alpha_by_layer[layer][head].float()
        if alpha_override is None
        else alpha_override.detach().float().cpu()
    )
    if row.ndim != 1:
        raise ValueError("Attention row must be one dimensional")
    start = int(
        donor.alpha_key_start_by_layer[layer]
        if key_start_override is None
        else key_start_override
    )
    values = receiver.value_by_layer[layer]
    if not 0 <= start < start + len(row) <= len(values):
        raise RuntimeError("Donor attention key axis does not align to receiver V")
    width = int(adapter.head_dims[layer])
    if values.shape[-1] % width:
        raise RuntimeError("Captured V width is not divisible by head width")
    kv_heads = int(values.shape[-1]) // width
    kv_head = query_to_kv_head(
        query_head=head,
        query_heads=adapter.num_heads[layer],
        kv_heads=kv_heads,
    )
    kv_start = kv_head * width
    selected_values = values[start : start + len(row), kv_start : kv_start + width]
    if selected_values.shape != (len(row), width):
        raise RuntimeError("Receiver V slice has the wrong shape")
    return torch.einsum("k,kd->d", row, selected_values.float())


def align_attention_row_to_receiver(
    row: torch.Tensor,
    *,
    donor_key_start: int,
    donor_encoding: PromptEncoding,
    receiver_key_start: int,
    receiver_key_length: int,
    receiver_encoding: PromptEncoding,
) -> torch.Tensor:
    """Piecewise-align an attention row across length-matched semantic slots.

    Qwen V4.4 pairs normally align exactly.  Gemma may tokenize a replaced slot
    to a slightly different width, so absolute token indices after that slot
    are not interchangeable.  The mapping uses the ten registered slot-span
    boundaries and linearly splats donor mass into the corresponding receiver
    segment.  It never writes or stores a raw attention row.
    """

    values = row.detach().float().cpu()
    if values.ndim != 1 or receiver_key_length <= 0:
        raise ValueError("Attention alignment received an invalid key axis")
    if len(donor_encoding.slot_spans) != len(receiver_encoding.slot_spans):
        raise ValueError("Donor/receiver have different semantic slot counts")
    donor_bounds = [0]
    receiver_bounds = [0]
    for donor_span, receiver_span in zip(
        donor_encoding.slot_spans, receiver_encoding.slot_spans
    ):
        donor_bounds.extend((int(donor_span.start), int(donor_span.end)))
        receiver_bounds.extend((int(receiver_span.start), int(receiver_span.end)))
    donor_bounds.append(int(donor_encoding.query_position) + 1)
    receiver_bounds.append(int(receiver_encoding.query_position) + 1)
    if donor_bounds != sorted(donor_bounds) or receiver_bounds != sorted(receiver_bounds):
        raise RuntimeError("Semantic token boundaries are not monotonic")
    output = torch.zeros(int(receiver_key_length), dtype=torch.float32)
    receiver_stop = int(receiver_key_start) + int(receiver_key_length)
    for offset, mass in enumerate(values):
        absolute = int(donor_key_start) + offset
        if not 0 <= absolute < donor_bounds[-1]:
            raise RuntimeError("Donor attention key lies outside its prompt")
        segment = int(np.searchsorted(donor_bounds[1:], absolute, side="right"))
        segment = min(segment, len(donor_bounds) - 2)
        d0, d1 = donor_bounds[segment], donor_bounds[segment + 1]
        r0, r1 = receiver_bounds[segment], receiver_bounds[segment + 1]
        if d1 <= d0 or r1 <= r0:
            raise RuntimeError("Degenerate semantic alignment segment")
        coordinate = r0 + ((absolute + 0.5 - d0) / (d1 - d0)) * (r1 - r0) - 0.5
        lower = math.floor(coordinate)
        upper = lower + 1
        upper_weight = float(coordinate - lower)
        for target, weight in ((lower, 1.0 - upper_weight), (upper, upper_weight)):
            if receiver_key_start <= target < receiver_stop and weight > 0:
                output[target - receiver_key_start] += float(mass) * weight
    input_mass = float(values.sum())
    output_mass = float(output.sum())
    if output_mass <= 0 or not math.isfinite(output_mass):
        raise RuntimeError("Semantic alignment retained no attention mass")
    output *= input_mass / output_mass
    return output


def scramble_attention_row(row: torch.Tensor, *, fraction: float) -> torch.Tensor:
    values = row.detach().float().cpu()
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("Position scrambling needs a nontrivial attention row")
    prefix = values[:-1]
    shift = max(1, min(len(prefix) - 1, int(round(len(prefix) * float(fraction)))))
    scrambled = torch.cat((torch.roll(prefix, shifts=shift), values[-1:]))
    if not torch.allclose(scrambled.sum(), values.sum(), atol=1e-6, rtol=1e-6):
        raise AssertionError("Position scrambling failed to preserve attention mass")
    return scrambled


@torch.inference_mode()
def run_with_z_replacement(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    head: int,
    replacement_z: torch.Tensor,
) -> CausalOutput:
    width = int(adapter.head_dims[layer])
    start = int(head) * width
    applied = 0
    captured_attention_output: torch.Tensor | None = None

    def hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        nonlocal applied
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("o_proj received no positional tensor")
        value = args[0]
        if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
            raise RuntimeError("Z replacement requires a full-prompt forward")
        replacement = replacement_z.to(device=value.device, dtype=value.dtype)
        if replacement.shape != (width,):
            raise RuntimeError("Replacement Z has the wrong shape")
        patched = value.clone()
        patched[:, encoding.query_position, start : start + width] = replacement
        applied += 1
        return (patched, *args[1:])

    def capture_output_hook(
        _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> None:
        nonlocal captured_attention_output
        value = _tensor_from_output(output)
        if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
            raise RuntimeError("Z replacement output capture requires full prefill")
        captured_attention_output = (
            value[0, encoding.query_position].detach().float().cpu()
        )

    handle = adapter.output_projections[layer].register_forward_pre_hook(hook)
    output_handle = adapter.attentions[layer].register_forward_hook(capture_output_hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
    finally:
        handle.remove()
        output_handle.remove()
    if applied != 1:
        raise RuntimeError(f"Z replacement applied {applied} times instead of once")
    if captured_attention_output is None:
        raise RuntimeError("Z replacement did not capture a post-O output")
    scored = _score_candidate_sequences(model, encoding, prefill_output)
    return CausalOutput(
        logits=scored.logits,
        candidate_log_scores=scored.candidate_log_scores,
        attention_output=captured_attention_output,
    )


@torch.inference_mode()
def run_with_attention_output_replacement(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    replacement: torch.Tensor,
) -> CausalOutput:
    applied = 0

    def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        nonlocal applied
        value = _tensor_from_output(output)
        if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
            raise RuntimeError("Output replacement requires a full-prompt forward")
        target = replacement.to(device=value.device, dtype=value.dtype)
        if target.shape != (value.shape[-1],):
            raise RuntimeError("Output replacement has the wrong hidden width")
        patched = value.clone()
        patched[:, encoding.query_position, :] = target
        applied += 1
        return _replace_output_tensor(output, patched)

    handle = adapter.attentions[layer].register_forward_hook(hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
    finally:
        handle.remove()
    if applied != 1:
        raise RuntimeError(
            f"Attention output replacement applied {applied} times instead of once"
        )
    return _score_candidate_sequences(model, encoding, prefill_output)


@torch.inference_mode()
def run_with_attention_output_delta(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    delta: torch.Tensor,
) -> CausalOutput:
    applied = 0

    def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        nonlocal applied
        value = _tensor_from_output(output)
        if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
            raise RuntimeError("Output intervention requires a full-prompt forward")
        addition = delta.to(device=value.device, dtype=value.dtype)
        if addition.shape != (value.shape[-1],):
            raise RuntimeError("Output delta has the wrong hidden width")
        patched = value.clone()
        patched[:, encoding.query_position, :] += addition
        applied += 1
        return _replace_output_tensor(output, patched)

    handle = adapter.attentions[layer].register_forward_hook(hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
    finally:
        handle.remove()
    if applied != 1:
        raise RuntimeError(
            f"Attention output intervention applied {applied} times instead of once"
        )
    return _score_candidate_sequences(model, encoding, prefill_output)


def _first_count_token_ids(encoding: PromptEncoding) -> tuple[int, ...]:
    token_ids: list[int] = []
    for _count, values in encoding.count_candidate_answer_token_ids:
        if not values:
            raise ValueError("A numeric count candidate has no answer tokens")
        token_ids.append(int(values[0]))
    return tuple(sorted(set(token_ids)))


def local_logit_selectivity_metrics(
    baseline_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    encoding: PromptEncoding,
) -> dict[str, float]:
    baseline = baseline_logits.detach().float().cpu()
    intervened = intervened_logits.detach().float().cpu()
    if baseline.ndim != 1 or intervened.shape != baseline.shape:
        raise ValueError("Selectivity metrics require matching vocabulary vectors")
    count_ids = _first_count_token_ids(encoding)
    mask = torch.ones(len(baseline), dtype=torch.bool)
    mask[list(count_ids)] = False
    base_log_probs = torch.log_softmax(baseline[mask], dim=0)
    int_log_probs = torch.log_softmax(intervened[mask], dim=0)
    base_probs = torch.exp(base_log_probs)
    non_count_kl = torch.sum(base_probs * (base_log_probs - int_log_probs))
    delta = intervened - baseline
    count_norm = torch.linalg.vector_norm(delta[list(count_ids)])
    full_norm = torch.linalg.vector_norm(delta)
    return {
        "non_count_token_kl": float(non_count_kl),
        "count_token_logit_delta_l2": float(count_norm),
        "full_vocab_logit_delta_l2": float(full_norm),
        "count_subspace_delta_fraction": (
            float(count_norm / full_norm) if float(full_norm) > 0 else 0.0
        ),
    }


def intervention_metrics(
    *,
    baseline_output: QueryBundle | CausalOutput,
    intervened_output: CausalOutput,
    encoding: PromptEncoding,
    donor_count: int | None = None,
) -> dict[str, Any]:
    baseline = candidate_sequence_metrics(
        baseline_output.candidate_log_scores, encoding
    )
    intervened = candidate_sequence_metrics(
        intervened_output.candidate_log_scores, encoding
    )
    expected_delta = float(intervened["expected_count"] - baseline["expected_count"])
    payload: dict[str, Any] = {
        "baseline_expected_count": float(baseline["expected_count"]),
        "intervened_expected_count": float(intervened["expected_count"]),
        "delta_expected_count": expected_delta,
        "baseline_expected_count_absolute_error": abs(
            float(baseline["expected_count"]) - int(encoding.count)
        ),
        "intervened_expected_count_absolute_error": abs(
            float(intervened["expected_count"]) - int(encoding.count)
        ),
        "delta_expected_count_absolute_error": (
            abs(float(intervened["expected_count"]) - int(encoding.count))
            - abs(float(baseline["expected_count"]) - int(encoding.count))
        ),
        "baseline_correct_margin": float(baseline["correct_count_margin"]),
        "intervened_correct_margin": float(intervened["correct_count_margin"]),
        "delta_correct_margin": float(
            intervened["correct_count_margin"] - baseline["correct_count_margin"]
        ),
        "baseline_predicted_count": int(baseline["predicted_count_among_candidates"]),
        "intervened_predicted_count": int(
            intervened["predicted_count_among_candidates"]
        ),
        **local_logit_selectivity_metrics(
            baseline_output.logits, intervened_output.logits, encoding
        ),
    }
    if donor_count is not None:
        semantic_shift = int(donor_count) - int(encoding.count)
        if semantic_shift == 0:
            raise ValueError("Patch donor and receiver counts must differ")
        payload["semantic_count_shift"] = semantic_shift
        payload["continuous_normalized_transport"] = expected_delta / semantic_shift
    return payload


def staged_patch_logits(
    model: nn.Module,
    adapter: DecoderAdapter,
    receiver_encoding: PromptEncoding,
    donor_encoding: PromptEncoding,
    *,
    receiver: QueryBundle,
    donor: QueryBundle,
    layer: int,
    head: int,
    scramble_fraction: float,
    orthogonal_label: str,
) -> dict[str, tuple[CausalOutput, float]]:
    receiver_z = head_z(receiver, adapter, layer=layer, head=head)
    donor_z = head_z(donor, adapter, layer=layer, head=head)
    receiver_o = head_output_from_z(adapter, layer=layer, head=head, z=receiver_z)
    donor_row = donor.alpha_by_layer[layer][head]
    receiver_row = receiver.alpha_by_layer[layer][head]
    aligned_row = align_attention_row_to_receiver(
        donor_row,
        donor_key_start=donor.alpha_key_start_by_layer[layer],
        donor_encoding=donor_encoding,
        receiver_key_start=receiver.alpha_key_start_by_layer[layer],
        receiver_key_length=len(receiver_row),
        receiver_encoding=receiver_encoding,
    )
    alpha_z = alpha_receiver_v_z(
        donor,
        receiver,
        adapter,
        layer=layer,
        head=head,
        alpha_override=aligned_row,
        key_start_override=receiver.alpha_key_start_by_layer[layer],
    )
    scrambled_row = scramble_attention_row(
        aligned_row, fraction=scramble_fraction
    )
    scrambled_z = alpha_receiver_v_z(
        donor,
        receiver,
        adapter,
        layer=layer,
        head=head,
        alpha_override=scrambled_row,
        key_start_override=receiver.alpha_key_start_by_layer[layer],
    )
    alpha_output = run_with_z_replacement(
        model,
        adapter,
        receiver_encoding,
        layer=layer,
        head=head,
        replacement_z=alpha_z,
    )
    scrambled_output = run_with_z_replacement(
        model,
        adapter,
        receiver_encoding,
        layer=layer,
        head=head,
        replacement_z=scrambled_z,
    )
    z_output = run_with_z_replacement(
        model,
        adapter,
        receiver_encoding,
        layer=layer,
        head=head,
        replacement_z=donor_z,
    )
    baseline_output = receiver.attention_output_by_layer[layer]
    if any(
        output.attention_output is None
        for output in (alpha_output, scrambled_output, z_output)
    ):
        raise RuntimeError("A Z patch did not expose its actual post-O output")
    alpha_delta = alpha_output.attention_output - baseline_output
    scrambled_delta = scrambled_output.attention_output - baseline_output
    output_delta = z_output.attention_output - baseline_output
    o_output = run_with_attention_output_replacement(
        model,
        adapter,
        receiver_encoding,
        layer=layer,
        replacement=z_output.attention_output,
    )
    orthogonal = deterministic_orthogonal_direction(
        output_delta
        if float(torch.linalg.vector_norm(output_delta)) > 0
        else receiver_o,
        label=orthogonal_label,
    )
    norm_control_delta = orthogonal * torch.linalg.vector_norm(output_delta)
    results = {
        "alpha_receiver_v": (
            alpha_output,
            float(torch.linalg.vector_norm(alpha_delta)),
        ),
        "alpha_position_scramble": (
            scrambled_output,
            float(torch.linalg.vector_norm(scrambled_delta)),
        ),
        "z_donor": (
            z_output,
            float(torch.linalg.vector_norm(output_delta)),
        ),
        "o_donor": (
            o_output,
            float(torch.linalg.vector_norm(output_delta)),
        ),
        "output_norm_control": (
            run_with_attention_output_delta(
                model,
                adapter,
                receiver_encoding,
                layer=layer,
                delta=norm_control_delta,
            ),
            float(torch.linalg.vector_norm(norm_control_delta)),
        ),
    }
    return results


def directed_intervention_logits(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    bundle: QueryBundle,
    layer: int,
    head: int,
    answer_direction: torch.Tensor,
    answer_step_scale: float,
    injection_betas: Sequence[float],
    orthogonal_label: str,
) -> dict[str, tuple[CausalOutput, float, float | None]]:
    unit = answer_direction.detach().float().cpu()
    unit = unit / torch.linalg.vector_norm(unit)
    receiver_z = head_z(bundle, adapter, layer=layer, head=head)
    receiver_o = head_output_from_z(
        adapter, layer=layer, head=head, z=receiver_z
    )
    coefficient = float(torch.dot(receiver_o, unit))
    removal_delta = -coefficient * unit
    orthogonal = deterministic_orthogonal_direction(unit, label=orthogonal_label)
    orthogonal_delta = -coefficient * orthogonal
    results: dict[str, tuple[CausalOutput, float, float | None]] = {
        "answer_direction_removal": (
            run_with_attention_output_delta(
                model,
                adapter,
                encoding,
                layer=layer,
                delta=removal_delta,
            ),
            float(torch.linalg.vector_norm(removal_delta)),
            None,
        ),
        "equal_norm_orthogonal_removal": (
            run_with_attention_output_delta(
                model,
                adapter,
                encoding,
                layer=layer,
                delta=orthogonal_delta,
            ),
            float(torch.linalg.vector_norm(orthogonal_delta)),
            None,
        ),
    }
    if not math.isfinite(float(answer_step_scale)) or float(answer_step_scale) <= 0:
        raise ValueError("answer_step_scale must be finite and positive")
    for beta in injection_betas:
        beta = float(beta)
        key = f"signed_answer_direction_injection_beta_{beta:+g}"
        if beta == 0.0:
            output = CausalOutput(
                logits=bundle.logits.clone(),
                candidate_log_scores=dict(bundle.candidate_log_scores),
            )
        else:
            delta = beta * float(answer_step_scale) * unit
            output = run_with_attention_output_delta(
                model,
                adapter,
                encoding,
                layer=layer,
                delta=delta,
            )
        results[key] = (
            output,
            abs(beta) * float(answer_step_scale),
            beta,
        )
    return results
