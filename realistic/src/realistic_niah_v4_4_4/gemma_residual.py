from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _replace_output_tensor,
    _tensor_from_output,
)
from realistic_niah_v4.prompts import PromptEncoding
from realistic_niah_v4_4_3.geometry import deterministic_orthogonal_direction
from realistic_niah_v4_4_3.interventions import (
    CausalOutput,
    QueryBundle,
    _score_candidate_sequences,
    capture_query_bundle,
    head_z,
)

from .gemma_cross_layer import site_set_label
from .gemma_cross_layer_spec import FrozenSite


HeadKey = tuple[int, int]


@dataclass(frozen=True)
class ResidualBundle:
    query: QueryBundle
    residual_by_layer: dict[int, torch.Tensor]


@dataclass(frozen=True)
class ResidualCausalOutput:
    causal_output: CausalOutput
    mediator_before: torch.Tensor
    mediator_after: torch.Tensor
    terminal_state: torch.Tensor


def fit_residual_intercept_and_step(
    states: Sequence[torch.Tensor], counts: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack([state.detach().float().cpu() for state in states], dim=0)
    labels = torch.as_tensor(tuple(int(count) for count in counts), dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] != len(labels):
        raise ValueError("Residual states/count labels have incompatible shapes")
    centered = labels - labels.mean()
    denominator = torch.dot(centered, centered)
    if float(denominator) <= 0:
        raise ValueError("Residual count fit needs at least two distinct counts")
    step = torch.einsum("n,nd->d", centered, values) / denominator
    intercept = values.mean(dim=0) - labels.mean() * step
    if not torch.isfinite(step).all() or float(torch.linalg.vector_norm(step)) <= 1e-8:
        raise RuntimeError("Residual count step is non-finite or degenerate")
    return intercept, step


def source_replacements(
    bundle: QueryBundle,
    adapter: DecoderAdapter,
    sites: Sequence[FrozenSite],
) -> dict[HeadKey, torch.Tensor]:
    return {
        (int(site.layer), int(site.head)): head_z(
            bundle,
            adapter,
            layer=int(site.layer),
            head=int(site.head),
        ).detach().float().cpu()
        for site in sites
    }


def residual_projection_coefficient(delta: torch.Tensor, step: torch.Tensor) -> float:
    change = delta.detach().float().cpu().flatten()
    direction = step.detach().float().cpu().flatten()
    denominator = torch.dot(direction, direction)
    if float(denominator) <= 0:
        raise ValueError("Residual projection axis is degenerate")
    return float(torch.dot(change, direction) / denominator)


def residual_component(delta: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    return residual_projection_coefficient(delta, step) * step.detach().float().cpu()


def equal_norm_orthogonal(
    axis: torch.Tensor, *, norm: float, label: str
) -> torch.Tensor:
    if not torch.isfinite(torch.as_tensor(norm)) or norm < 0:
        raise ValueError("Orthogonal-control norm is invalid")
    if norm <= 1e-12:
        return torch.zeros_like(axis.detach().float().cpu())
    return deterministic_orthogonal_direction(axis, label=label) * float(norm)


@torch.inference_mode()
def capture_source_and_residual_bundle(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    source_layers: Sequence[int],
    residual_layers: Sequence[int],
    cache_logit_tolerance: float,
) -> ResidualBundle:
    selected_residuals = tuple(sorted({int(layer) for layer in residual_layers}))
    if not selected_residuals:
        raise ValueError("Residual capture needs at least one layer")
    residuals: dict[int, torch.Tensor] = {}
    calls = {layer: 0 for layer in selected_residuals}

    def make_hook(layer: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return
            residuals[layer] = (
                hidden[0, int(encoding.query_position)].detach().float().cpu()
            )
            calls[layer] += 1

        return hook

    handles = [
        adapter.layers[layer].register_forward_hook(make_hook(layer))
        for layer in selected_residuals
    ]
    try:
        query = capture_query_bundle(
            model,
            adapter,
            encoding,
            layers=tuple(sorted({int(layer) for layer in source_layers})),
            capture_attention=False,
            capture_values=False,
            cache_logit_tolerance=float(cache_logit_tolerance),
        )
    finally:
        for handle in handles:
            handle.remove()
    violations = {layer: count for layer, count in calls.items() if count != 1}
    if violations:
        raise RuntimeError(f"Residual bundle hook mismatch: {violations}")
    return ResidualBundle(query=query, residual_by_layer=residuals)


@torch.inference_mode()
def run_source_patch_with_residual_delta(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    replacements: Mapping[HeadKey, torch.Tensor],
    mediator_layer: int,
    terminal_layer: int,
    residual_delta: torch.Tensor | None = None,
) -> ResidualCausalOutput:
    """Patch a frozen pre-O source bank and optionally edit one residual state.

    Source replacements occur at the answer-query pre-O slices.  The residual
    intervention is applied to the post-block answer-query state at the frozen
    mediator layer.  L41 is captured after downstream computation.  Hooks are
    removed before candidate-sequence continuation scoring.
    """

    mediator_layer = int(mediator_layer)
    terminal_layer = int(terminal_layer)
    if mediator_layer >= terminal_layer:
        raise ValueError("Residual mediator must precede the terminal trace")
    if not replacements:
        raise ValueError("A residual-path run needs a source-bank patch")
    if any(int(layer) >= mediator_layer for layer, _head in replacements):
        raise ValueError("Every source site must precede the residual mediator")
    grouped: dict[int, dict[int, torch.Tensor]] = {}
    for (layer, head), state in replacements.items():
        grouped.setdefault(int(layer), {})[int(head)] = state.detach().float().cpu()
    source_calls = {layer: 0 for layer in grouped}
    mediator_calls = 0
    terminal_calls = 0
    mediator_before: torch.Tensor | None = None
    mediator_after: torch.Tensor | None = None
    terminal_state: torch.Tensor | None = None
    handles = []
    for layer, states in grouped.items():
        expected = int(adapter.num_heads[layer]) * int(adapter.head_dims[layer])
        width = int(adapter.head_dims[layer])

        def source_hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            states: dict[int, torch.Tensor] = states,
            expected: int = expected,
            width: int = width,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Source o_proj received no positional tensor")
            value = args[0]
            if (
                value.ndim != 3
                or value.shape[0] != 1
                or value.shape[1] != encoding.sequence_length
                or value.shape[-1] != expected
            ):
                raise RuntimeError("Residual source patch requires a full prefill")
            patched = value.clone()
            for head, state in states.items():
                replacement = state.to(device=value.device, dtype=value.dtype)
                if replacement.shape != (width,):
                    raise RuntimeError("A residual source state has the wrong width")
                start = int(head) * width
                patched[0, int(encoding.query_position), start : start + width] = replacement
            source_calls[layer] += 1
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(source_hook)
        )

    def mediator_hook(
        _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> Any:
        nonlocal mediator_calls, mediator_before, mediator_after
        hidden = _tensor_from_output(output)
        if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
            return output
        query = int(encoding.query_position)
        mediator_before = hidden[0, query].detach().float().cpu()
        patched = hidden
        if residual_delta is not None:
            addition = residual_delta.to(device=hidden.device, dtype=hidden.dtype)
            if addition.shape != hidden.shape[-1:]:
                raise RuntimeError("Residual intervention has the wrong hidden width")
            patched = hidden.clone()
            patched[0, query] += addition
        mediator_after = patched[0, query].detach().float().cpu()
        mediator_calls += 1
        return _replace_output_tensor(output, patched)

    def terminal_hook(
        _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> None:
        nonlocal terminal_calls, terminal_state
        hidden = _tensor_from_output(output)
        if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
            return
        terminal_state = (
            hidden[0, int(encoding.query_position)].detach().float().cpu()
        )
        terminal_calls += 1

    handles.extend(
        (
            adapter.layers[mediator_layer].register_forward_hook(mediator_hook),
            adapter.layers[terminal_layer].register_forward_hook(terminal_hook),
        )
    )
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    bad_sources = {layer: count for layer, count in source_calls.items() if count != 1}
    if (
        bad_sources
        or mediator_calls != 1
        or terminal_calls != 1
        or mediator_before is None
        or mediator_after is None
        or terminal_state is None
    ):
        raise RuntimeError(
            "Residual intervention hook mismatch: "
            f"source={bad_sources}, mediator={mediator_calls}, terminal={terminal_calls}"
        )
    scored = _score_candidate_sequences(model, encoding, prefill)
    return ResidualCausalOutput(
        causal_output=CausalOutput(
            logits=scored.logits,
            candidate_log_scores=scored.candidate_log_scores,
        ),
        mediator_before=mediator_before,
        mediator_after=mediator_after,
        terminal_state=terminal_state,
    )


def residual_set_label(sites: Sequence[FrozenSite]) -> str:
    return site_set_label(sites)
