from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _bounded_logits_kwargs,
    _encoding_tensors,
)
from realistic_niah_v4.prompts import PromptEncoding

from .geometry import deterministic_orthogonal_direction
from .interventions import (
    CausalOutput,
    QueryBundle,
    _replace_output_tensor,
    _score_candidate_sequences,
    _tensor_from_output,
    align_attention_row_to_receiver,
    alpha_receiver_v_z,
    head_output_from_z,
    head_z,
    run_with_attention_output_delta,
    run_with_attention_output_replacement,
    scramble_attention_row,
)


@torch.inference_mode()
def run_with_set_z_replacements(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    replacements: Mapping[int, torch.Tensor],
) -> CausalOutput:
    if not replacements:
        raise ValueError("A set Z intervention needs at least one head")
    width = int(adapter.head_dims[int(layer)])
    applied = 0
    captured_attention_output: torch.Tensor | None = None

    def z_hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        nonlocal applied
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("o_proj received no positional tensor")
        value = args[0]
        if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
            raise RuntimeError("Set Z replacement requires a full-prompt forward")
        patched = value.clone()
        for head, replacement_z in replacements.items():
            start = int(head) * width
            replacement = replacement_z.to(device=value.device, dtype=value.dtype)
            if replacement.shape != (width,):
                raise RuntimeError("A set replacement Z has the wrong shape")
            patched[:, encoding.query_position, start : start + width] = replacement
        applied += 1
        return (patched, *args[1:])

    def output_hook(
        _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> None:
        nonlocal captured_attention_output
        value = _tensor_from_output(output)
        if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
            raise RuntimeError("Set Z output capture requires a full prefill")
        captured_attention_output = (
            value[0, encoding.query_position].detach().float().cpu()
        )

    z_handle = adapter.output_projections[int(layer)].register_forward_pre_hook(z_hook)
    output_handle = adapter.attentions[int(layer)].register_forward_hook(output_hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
    finally:
        z_handle.remove()
        output_handle.remove()
    if applied != 1:
        raise RuntimeError(f"Set Z replacement applied {applied} times")
    if captured_attention_output is None:
        raise RuntimeError("Set Z replacement captured no post-O output")
    scored = _score_candidate_sequences(model, encoding, prefill_output)
    return CausalOutput(
        logits=scored.logits,
        candidate_log_scores=scored.candidate_log_scores,
        attention_output=captured_attention_output,
    )


@torch.inference_mode()
def run_with_set_z_deltas(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    deltas: Mapping[int, torch.Tensor],
) -> CausalOutput:
    """Add head-specific vectors at the true pre-O ``z_h`` boundary.

    This differs from residual steering: every delta is written into the
    selected head slice before the model's own O projection is evaluated.
    """

    if not deltas:
        raise ValueError("A set Z intervention needs at least one head")
    width = int(adapter.head_dims[int(layer)])
    applied = 0
    captured_attention_output: torch.Tensor | None = None

    def z_hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        nonlocal applied
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("o_proj received no positional tensor")
        value = args[0]
        if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
            raise RuntimeError("Set Z addition requires a full-prompt forward")
        patched = value.clone()
        for head, delta_z in deltas.items():
            start = int(head) * width
            addition = delta_z.to(device=value.device, dtype=value.dtype)
            if addition.shape != (width,):
                raise RuntimeError("A set Z delta has the wrong shape")
            patched[:, encoding.query_position, start : start + width] += addition
        applied += 1
        return (patched, *args[1:])

    def output_hook(
        _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> None:
        nonlocal captured_attention_output
        value = _tensor_from_output(output)
        if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
            raise RuntimeError("Set Z output capture requires a full prefill")
        captured_attention_output = (
            value[0, encoding.query_position].detach().float().cpu()
        )

    z_handle = adapter.output_projections[int(layer)].register_forward_pre_hook(z_hook)
    output_handle = adapter.attentions[int(layer)].register_forward_hook(output_hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
    finally:
        z_handle.remove()
        output_handle.remove()
    if applied != 1:
        raise RuntimeError(f"Set Z addition applied {applied} times")
    if captured_attention_output is None:
        raise RuntimeError("Set Z addition captured no post-O output")
    scored = _score_candidate_sequences(model, encoding, prefill_output)
    return CausalOutput(
        logits=scored.logits,
        candidate_log_scores=scored.candidate_log_scores,
        attention_output=captured_attention_output,
    )


def _set_output_from_bundle(
    bundle: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
) -> torch.Tensor:
    outputs = [
        head_output_from_z(
            adapter,
            layer=int(layer),
            head=int(head),
            z=head_z(bundle, adapter, layer=int(layer), head=int(head)),
        )
        for head in heads
    ]
    return torch.stack(outputs).sum(dim=0)


def _set_output_from_stacked_z(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    stacked_z: torch.Tensor,
) -> torch.Tensor:
    registered = tuple(int(head) for head in heads)
    width = int(adapter.head_dims[int(layer)])
    values = stacked_z.detach().float().cpu()
    if values.shape != (len(registered), width):
        raise ValueError("Stacked set Z has the wrong shape")
    outputs = [
        head_output_from_z(
            adapter,
            layer=int(layer),
            head=head,
            z=values[offset],
        )
        for offset, head in enumerate(registered)
    ]
    return torch.stack(outputs).sum(dim=0)


def natural_ov_removal_deltas(
    adapter: DecoderAdapter,
    *,
    bundle: QueryBundle,
    layer: int,
    heads: Sequence[int],
    z_count_steps: torch.Tensor,
    orthogonal_label: str,
    z_center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Construct a realizable natural-OV removal and a span-matched control.

    Let ``d_S`` be the stacked, one-count V-path step at the selected heads and
    ``m_S = W_O^S d_S``.  The primary delta removes the actual selected-set
    output component along ``m_S``.  Both the removal and its control are
    represented at pre-O Z, and the control has an orthogonal, equal-norm
    output inside the same selected-set W_O column span.
    """

    registered = tuple(int(head) for head in heads)
    if not registered or len(set(registered)) != len(registered):
        raise ValueError("Natural OV removal requires unique set heads")
    width = int(adapter.head_dims[int(layer)])
    steps = z_count_steps.detach().float().cpu()
    if steps.shape != (len(registered), width):
        raise ValueError("Natural OV count steps have the wrong shape")
    flat_steps = steps.reshape(-1)
    step_output = _set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=registered,
        stacked_z=steps,
    )
    step_output_norm = torch.linalg.vector_norm(step_output)
    if not torch.isfinite(step_output_norm) or float(step_output_norm) <= 1e-8:
        raise RuntimeError("Natural OV set output direction is degenerate")
    output_unit = step_output / step_output_norm
    actual_output = _set_output_from_bundle(
        bundle,
        adapter,
        layer=int(layer),
        heads=registered,
    )
    if z_center is None:
        centered_output = actual_output
    else:
        center = z_center.detach().float().cpu()
        if center.shape != steps.shape:
            raise ValueError("Natural OV Z center has the wrong shape")
        center_output = _set_output_from_stacked_z(
            adapter,
            layer=int(layer),
            heads=registered,
            stacked_z=center,
        )
        centered_output = actual_output - center_output
    coefficient = float(torch.dot(centered_output, output_unit))
    removal = (-coefficient / float(step_output_norm)) * steps

    # Start with a deterministic Z-space probe, then remove its output-space
    # component along m_S.  Because m_S itself is W_O^S d_S, this subtraction
    # has an explicit preimage and remains inside the selected set's span.
    probe = deterministic_orthogonal_direction(
        flat_steps, label=f"{orthogonal_label}:z-probe"
    ).reshape_as(steps)
    probe_output = _set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=registered,
        stacked_z=probe,
    )
    probe_parallel = float(torch.dot(probe_output, output_unit))
    control = probe - (probe_parallel / float(step_output_norm)) * steps
    control_output = _set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=registered,
        stacked_z=control,
    )
    control_output_norm = torch.linalg.vector_norm(control_output)
    target_output_norm = abs(coefficient)
    if target_output_norm <= 1e-12:
        control = torch.zeros_like(control)
        control_output = torch.zeros_like(control_output)
    elif not torch.isfinite(control_output_norm) or float(control_output_norm) <= 1e-8:
        raise RuntimeError("Cannot construct a stable set-span orthogonal control")
    else:
        control = control * (target_output_norm / float(control_output_norm))
        control_output = _set_output_from_stacked_z(
            adapter,
            layer=int(layer),
            heads=registered,
            stacked_z=control,
        )
    realized_removal = _set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=registered,
        stacked_z=removal,
    )
    diagnostics = {
        "natural_ov_output_step_norm": float(step_output_norm),
        "removed_output_coefficient": coefficient,
        "used_count_neutral_z_center": float(z_center is not None),
        "removal_output_norm": float(torch.linalg.vector_norm(realized_removal)),
        "control_output_norm": float(torch.linalg.vector_norm(control_output)),
        "control_output_cosine_with_removed_axis": (
            float(torch.dot(control_output, output_unit))
            / max(float(torch.linalg.vector_norm(control_output)), 1e-12)
        ),
    }
    return removal, control, diagnostics


def staged_set_patch_logits(
    model: nn.Module,
    adapter: DecoderAdapter,
    receiver_encoding: PromptEncoding,
    donor_encoding: PromptEncoding,
    *,
    receiver: QueryBundle,
    donor: QueryBundle,
    layer: int,
    heads: Sequence[int],
    scramble_fraction: float,
    orthogonal_label: str,
) -> dict[str, tuple[CausalOutput, float]]:
    registered = tuple(sorted({int(head) for head in heads}))
    if len(registered) != len(heads):
        raise ValueError("Set heads must be unique")
    receiver_z = {
        head: head_z(receiver, adapter, layer=int(layer), head=head)
        for head in registered
    }
    donor_z = {
        head: head_z(donor, adapter, layer=int(layer), head=head)
        for head in registered
    }
    alpha_z: dict[int, torch.Tensor] = {}
    scrambled_z: dict[int, torch.Tensor] = {}
    for head in registered:
        donor_row = donor.alpha_by_layer[int(layer)][head]
        receiver_row = receiver.alpha_by_layer[int(layer)][head]
        aligned = align_attention_row_to_receiver(
            donor_row,
            donor_key_start=donor.alpha_key_start_by_layer[int(layer)],
            donor_encoding=donor_encoding,
            receiver_key_start=receiver.alpha_key_start_by_layer[int(layer)],
            receiver_key_length=len(receiver_row),
            receiver_encoding=receiver_encoding,
        )
        alpha_z[head] = alpha_receiver_v_z(
            donor,
            receiver,
            adapter,
            layer=int(layer),
            head=head,
            alpha_override=aligned,
            key_start_override=receiver.alpha_key_start_by_layer[int(layer)],
        )
        scrambled_z[head] = alpha_receiver_v_z(
            donor,
            receiver,
            adapter,
            layer=int(layer),
            head=head,
            alpha_override=scramble_attention_row(
                aligned, fraction=float(scramble_fraction)
            ),
            key_start_override=receiver.alpha_key_start_by_layer[int(layer)],
        )
    alpha_output = run_with_set_z_replacements(
        model,
        adapter,
        receiver_encoding,
        layer=int(layer),
        replacements=alpha_z,
    )
    scrambled_output = run_with_set_z_replacements(
        model,
        adapter,
        receiver_encoding,
        layer=int(layer),
        replacements=scrambled_z,
    )
    z_output = run_with_set_z_replacements(
        model,
        adapter,
        receiver_encoding,
        layer=int(layer),
        replacements=donor_z,
    )
    baseline_output = receiver.attention_output_by_layer[int(layer)]
    if any(
        output.attention_output is None
        for output in (alpha_output, scrambled_output, z_output)
    ):
        raise RuntimeError("A set Z patch did not expose its actual post-O output")
    # Use the model's actual post-O vectors here.  Recomputing a summed W_O delta
    # outside the fused BF16 projection is algebraically equivalent but can differ
    # enough numerically to break the Z/O implementation audit.
    alpha_delta = alpha_output.attention_output - baseline_output
    scrambled_delta = scrambled_output.attention_output - baseline_output
    output_delta = z_output.attention_output - baseline_output
    o_output = run_with_attention_output_replacement(
        model,
        adapter,
        receiver_encoding,
        layer=int(layer),
        replacement=z_output.attention_output,
    )
    reference = (
        output_delta
        if float(torch.linalg.vector_norm(output_delta)) > 0
        else _set_output_from_bundle(
            receiver, adapter, layer=int(layer), heads=registered
        )
    )
    orthogonal = deterministic_orthogonal_direction(
        reference, label=orthogonal_label
    )
    norm_control_delta = orthogonal * torch.linalg.vector_norm(output_delta)
    return {
        "alpha_receiver_v": (
            alpha_output,
            float(torch.linalg.vector_norm(alpha_delta)),
        ),
        "alpha_position_scramble": (
            scrambled_output,
            float(torch.linalg.vector_norm(scrambled_delta)),
        ),
        "z_donor": (z_output, float(torch.linalg.vector_norm(output_delta))),
        "o_donor": (o_output, float(torch.linalg.vector_norm(output_delta))),
        "output_norm_control": (
            run_with_attention_output_delta(
                model,
                adapter,
                receiver_encoding,
                layer=int(layer),
                delta=norm_control_delta,
            ),
            float(torch.linalg.vector_norm(norm_control_delta)),
        ),
    }


def directed_set_intervention_logits(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    bundle: QueryBundle,
    layer: int,
    heads: Sequence[int],
    answer_direction: torch.Tensor,
    reachable_answer_direction: torch.Tensor,
    answer_step_scale: float,
    injection_betas: Sequence[float],
    orthogonal_label: str,
) -> dict[str, tuple[CausalOutput, float, float | None]]:
    answer_unit = answer_direction.detach().float().cpu()
    answer_unit = answer_unit / torch.linalg.vector_norm(answer_unit)
    reachable_unit = reachable_answer_direction.detach().float().cpu()
    reachable_unit = reachable_unit / torch.linalg.vector_norm(reachable_unit)
    set_output = _set_output_from_bundle(
        bundle,
        adapter,
        layer=int(layer),
        heads=tuple(int(head) for head in heads),
    )
    coefficient = float(torch.dot(set_output, answer_unit))
    removal_delta = -coefficient * answer_unit
    orthogonal = deterministic_orthogonal_direction(
        answer_unit, label=orthogonal_label
    )
    orthogonal_delta = -coefficient * orthogonal
    results: dict[str, tuple[CausalOutput, float, float | None]] = {
        "answer_direction_removal": (
            run_with_attention_output_delta(
                model,
                adapter,
                encoding,
                layer=int(layer),
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
                layer=int(layer),
                delta=orthogonal_delta,
            ),
            float(torch.linalg.vector_norm(orthogonal_delta)),
            None,
        ),
    }
    if not math.isfinite(float(answer_step_scale)) or float(answer_step_scale) <= 0:
        raise ValueError("answer_step_scale must be finite and positive")
    for beta_value in injection_betas:
        beta = float(beta_value)
        key = f"signed_answer_direction_injection_beta_{beta:+g}"
        if beta == 0.0:
            output = CausalOutput(
                logits=bundle.logits.clone(),
                candidate_log_scores=dict(bundle.candidate_log_scores),
            )
        else:
            output = run_with_attention_output_delta(
                model,
                adapter,
                encoding,
                layer=int(layer),
                delta=beta * float(answer_step_scale) * reachable_unit,
            )
        results[key] = (
            output,
            abs(beta) * float(answer_step_scale),
            beta,
        )
    return results


def natural_ov_set_intervention_logits(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    bundle: QueryBundle,
    layer: int,
    heads: Sequence[int],
    z_count_steps: torch.Tensor,
    injection_betas: Sequence[float],
    orthogonal_label: str,
    z_center: torch.Tensor | None = None,
) -> tuple[
    dict[str, tuple[CausalOutput, float, float | None]],
    dict[str, float],
]:
    """Run natural OV injection and set-realizable removal at pre-O Z.

    A beta of one adds the fitted one-count V-path step to every selected
    query-head channel. The model's own O projection then produces
    ``sum_h W_O^h delta_z_h``. No post-O answer direction is supplied.
    """

    registered = tuple(int(head) for head in heads)
    width = int(adapter.head_dims[int(layer)])
    steps = z_count_steps.detach().float().cpu()
    if steps.shape != (len(registered), width):
        raise ValueError("Natural OV count steps have the wrong shape")
    output_step = _set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=registered,
        stacked_z=steps,
    )
    output_step_norm = float(torch.linalg.vector_norm(output_step))
    if not math.isfinite(output_step_norm) or output_step_norm <= 1e-8:
        raise RuntimeError("Natural OV set output direction is degenerate")
    removal, control, diagnostics = natural_ov_removal_deltas(
        adapter,
        bundle=bundle,
        layer=int(layer),
        heads=registered,
        z_count_steps=steps,
        orthogonal_label=orthogonal_label,
        z_center=z_center,
    )

    def as_mapping(stacked: torch.Tensor) -> dict[int, torch.Tensor]:
        return {
            head: stacked[offset]
            for offset, head in enumerate(registered)
        }

    results: dict[str, tuple[CausalOutput, float, float | None]] = {
        "natural_ov_count_axis_removal": (
            run_with_set_z_deltas(
                model,
                adapter,
                encoding,
                layer=int(layer),
                deltas=as_mapping(removal),
            ),
            float(diagnostics["removal_output_norm"]),
            None,
        ),
        "equal_output_norm_set_span_orthogonal_removal": (
            run_with_set_z_deltas(
                model,
                adapter,
                encoding,
                layer=int(layer),
                deltas=as_mapping(control),
            ),
            float(diagnostics["control_output_norm"]),
            None,
        ),
    }
    for beta_value in injection_betas:
        beta = float(beta_value)
        key = f"natural_ov_z_injection_beta_{beta:+g}"
        if beta == 0.0:
            output = CausalOutput(
                logits=bundle.logits.clone(),
                candidate_log_scores=dict(bundle.candidate_log_scores),
                attention_output=bundle.attention_output_by_layer[int(layer)].clone(),
            )
        else:
            output = run_with_set_z_deltas(
                model,
                adapter,
                encoding,
                layer=int(layer),
                deltas=as_mapping(beta * steps),
            )
        results[key] = (output, abs(beta) * output_step_norm, beta)
    diagnostics = {
        **diagnostics,
        "natural_ov_z_count_step_norm": float(torch.linalg.vector_norm(steps)),
        "natural_ov_output_count_step_norm": output_step_norm,
    }
    return results, diagnostics
