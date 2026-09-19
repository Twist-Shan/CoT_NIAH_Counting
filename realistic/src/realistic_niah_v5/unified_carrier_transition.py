"""Shared interventions for count-carrier comparison experiments.

The functions in this module deliberately separate two operations:

* replacing the residual state at one native item boundary; and
* adding frozen directions to historical key/value projections.

Both operations can be applied to a complete native trace, where later
boundary states are captured, or to a prefix prefill, where native next-item
candidates are scored.  Keeping these two readouts under the same intervention
geometry is the main purpose of the module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _replace_output_tensor,
    _tensor_from_output,
)

from .count_stream import _prefix_forward
from .encoding import NativeTraceEncoding
from .kv_counter_transition import projection_module


def projected_donor_delta(
    receiver_state: np.ndarray,
    donor_state: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Project a natural donor-minus-receiver delta into a frozen subspace."""

    receiver = np.asarray(receiver_state, dtype=np.float64).reshape(-1)
    donor = np.asarray(donor_state, dtype=np.float64).reshape(-1)
    active_basis = np.asarray(basis, dtype=np.float64)
    if receiver.shape != donor.shape:
        raise ValueError("Receiver and donor state widths disagree")
    if active_basis.ndim != 2 or active_basis.shape[0] != receiver.size:
        raise ValueError("Subspace basis and residual-state widths disagree")
    raw = donor - receiver
    return ((raw @ active_basis) @ active_basis.T).astype(np.float32)


def interpolated_boundary_targets(
    receiver_states: Mapping[int, np.ndarray | torch.Tensor],
    donor_states: Mapping[int, np.ndarray | torch.Tensor],
    *,
    scale: float,
    bases: Mapping[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build absolute clamp targets and their planned deltas.

    With ``bases=None`` the complete natural donor delta is used.  Otherwise
    each layer delta is projected into the corresponding frozen count span.
    """

    active_scale = float(scale)
    if not np.isfinite(active_scale) or active_scale <= 0:
        raise ValueError("Carrier scale must be finite and positive")
    receiver_layers = {int(layer) for layer in receiver_states}
    donor_layers = {int(layer) for layer in donor_states}
    if not receiver_layers or receiver_layers != donor_layers:
        raise ValueError("Receiver and donor layer banks must match and be nonempty")
    if bases is not None and set(int(layer) for layer in bases) != receiver_layers:
        raise ValueError("Every projected carrier layer needs one frozen basis")

    targets: dict[int, np.ndarray] = {}
    deltas: dict[int, np.ndarray] = {}
    for layer in sorted(receiver_layers):
        receiver = np.asarray(receiver_states[layer], dtype=np.float32).reshape(-1)
        donor = np.asarray(donor_states[layer], dtype=np.float32).reshape(-1)
        if bases is None:
            delta = donor - receiver
        else:
            delta = projected_donor_delta(receiver, donor, bases[layer])
        planned = (active_scale * delta).astype(np.float32)
        targets[layer] = (receiver + planned).astype(np.float32)
        deltas[layer] = planned
    return targets, deltas


def through_origin_slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Return the least-squares through-origin slope, or ``None`` at zero input."""

    left = np.asarray(tuple(float(value) for value in x), dtype=np.float64)
    right = np.asarray(tuple(float(value) for value in y), dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Slope inputs must be same-length vectors")
    denominator = float(left @ left)
    if denominator <= 1e-12:
        return None
    return float((left @ right) / denominator)


def summarize_carrier_trials(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize first-stage displacement, retention, and identity preservation."""

    output: dict[str, Any] = {}
    carriers = sorted({str(row["carrier"]) for row in rows})
    for carrier in carriers:
        active = [row for row in rows if str(row["carrier"]) == carrier]
        current = [float(row["current_shift"]) for row in active]
        later = [float(row["next_shift"]) for row in active]
        receiver_argmax_mean = [
            bool(
                row.get(
                    "receiver_successor_argmax_mean_logprob",
                    row.get("receiver_successor_argmax", False),
                )
            )
            for row in active
        ]
        donor_argmax_mean = [
            bool(
                row.get(
                    "donor_successor_argmax_mean_logprob",
                    row.get("donor_successor_argmax", False),
                )
            )
            for row in active
        ]
        receiver_mean_changes = [
            float(
                row.get(
                    "receiver_successor_mean_logprob_change",
                    row.get("receiver_successor_logprob_change", 0.0),
                )
            )
            for row in active
        ]
        donor_receiver_mean_changes = [
            float(
                row.get(
                    "donor_vs_receiver_mean_logodds_change",
                    row.get("donor_vs_receiver_logodds_change", 0.0),
                )
            )
            for row in active
        ]
        output[carrier] = {
            "trial_count": len(active),
            "mean_abs_first_stage_shift": float(np.mean(np.abs(current))),
            "mean_abs_next_shift": float(np.mean(np.abs(later))),
            "pooled_current_to_next_retention": through_origin_slope(current, later),
            "current_exact": int(sum(bool(row["current_exact"]) for row in active)),
            "next_exact": int(sum(bool(row["next_exact"]) for row in active)),
            "receiver_successor_argmax_mean_logprob": int(sum(receiver_argmax_mean)),
            "donor_successor_argmax_mean_logprob": int(sum(donor_argmax_mean)),
            "mean_receiver_successor_mean_logprob_change": float(
                np.mean(receiver_mean_changes)
            ),
            "mean_donor_vs_receiver_mean_logodds_change": float(
                np.mean(donor_receiver_mean_changes)
            ),
        }
    return output


def _prepare_carrier(
    encoding: NativeTraceEncoding,
    *,
    boundary_position: int,
    boundary_targets: Mapping[int, np.ndarray | torch.Tensor],
    kv_directions: Mapping[
        tuple[int, str], Mapping[int, np.ndarray | torch.Tensor]
    ],
) -> tuple[
    int,
    dict[int, torch.Tensor],
    dict[tuple[int, str], dict[int, torch.Tensor]],
]:
    boundary = int(boundary_position)
    if not 0 <= boundary < int(encoding.sequence_length):
        raise ValueError("Carrier boundary is outside the encoding")
    targets = {
        int(layer): torch.as_tensor(value).detach().float().cpu().reshape(-1)
        for layer, value in boundary_targets.items()
    }
    kv = {
        (int(layer), str(kind)): {
            int(position): torch.as_tensor(value).detach().float().cpu().reshape(-1)
            for position, value in panel.items()
        }
        for (layer, kind), panel in kv_directions.items()
    }
    if not targets and not kv:
        raise ValueError("A carrier intervention must edit residual or K/V state")
    if any(kind not in {"k", "v"} for _layer, kind in kv):
        raise ValueError("Carrier K/V edits must use k or v projections")
    for panel in kv.values():
        if not panel or min(panel) < 0 or max(panel) >= int(encoding.sequence_length):
            raise ValueError("A carrier K/V position is outside the encoding")
    return boundary, targets, kv


def _install_carrier_hooks(
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    boundary: int,
    targets: Mapping[int, torch.Tensor],
    kv: Mapping[tuple[int, str], Mapping[int, torch.Tensor]],
) -> tuple[
    list[Any],
    dict[int, int],
    dict[str, int],
    dict[int, float],
    dict[str, float],
]:
    boundary_applications = {int(layer): 0 for layer in targets}
    kv_applications = {f"L{layer}_{kind}": 0 for layer, kind in kv}
    boundary_norms = {int(layer): 0.0 for layer in targets}
    kv_norms = {f"L{layer}_{kind}": 0.0 for layer, kind in kv}
    handles: list[Any] = []

    for layer, target in targets.items():

        def boundary_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            target: torch.Tensor = target,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Carrier block input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
                return None
            before = hidden[0, boundary]
            if before.numel() != target.numel():
                raise RuntimeError("Carrier residual target width mismatch")
            replacement = target.to(device=before.device, dtype=before.dtype)
            patched = hidden.clone()
            patched[0, boundary] = replacement
            boundary_norms[layer] = float(
                torch.linalg.vector_norm(replacement.float() - before.float())
                .detach()
                .cpu()
            )
            boundary_applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(boundary_hook))

    for (layer, kind), panel in kv.items():
        module = projection_module(adapter.attentions[layer], kind)
        label = f"L{layer}_{kind}"

        def kv_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            panel: Mapping[int, torch.Tensor] = panel,
            label: str = label,
        ) -> Any:
            value = _tensor_from_output(output)
            if value.ndim != 3 or int(value.shape[0]) != 1:
                raise RuntimeError("Carrier KV projection returned an unsupported tensor")
            if int(value.shape[1]) != int(encoding.sequence_length):
                return output
            patched = value.clone()
            squared = 0.0
            for position, direction in panel.items():
                before = value[0, int(position)]
                if before.numel() != direction.numel():
                    raise RuntimeError("Carrier KV direction width mismatch")
                replacement = (
                    before.float() + direction.to(device=before.device)
                ).to(dtype=before.dtype)
                patched[0, int(position)] = replacement
                realized = replacement.float() - before.float()
                squared += float(torch.sum(realized * realized).detach().cpu())
            kv_norms[label] = float(np.sqrt(squared))
            kv_applications[label] += 1
            return _replace_output_tensor(output, patched)

        handles.append(module.register_forward_hook(kv_hook))

    return handles, boundary_applications, kv_applications, boundary_norms, kv_norms


def _validate_applications(
    boundary_applications: Mapping[int, int],
    kv_applications: Mapping[str, int],
) -> None:
    if any(int(value) != 1 for value in boundary_applications.values()) or any(
        int(value) != 1 for value in kv_applications.values()
    ):
        raise RuntimeError(
            "Every carrier hook must apply exactly once: "
            f"boundary={dict(boundary_applications)}, kv={dict(kv_applications)}"
        )


@torch.inference_mode()
def no_cache_tail_logits(
    model: Any,
    encoding: NativeTraceEncoding,
    *,
    logits_to_keep: int,
) -> torch.Tensor:
    """Run a memory-bounded causal forward and return trailing logits on CPU.

    A cached prefill can exceed accelerator memory for the long realistic-NIAH
    prompts because it retains every layer's complete K/V history.  Candidate
    scoring only needs the logits that predict the candidate tokens, so this
    helper disables the cache and asks supported models to materialize only
    that trailing window.
    """

    keep = int(logits_to_keep)
    if keep < 1 or keep > int(encoding.sequence_length):
        raise ValueError("Trailing-logit count must fit inside the encoding")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    bounded = _bounded_logits_kwargs(model)
    if "logits_to_keep" in bounded:
        bounded["logits_to_keep"] = keep
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=False,
        **bounded,
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Candidate forward returned no [batch,time,vocab] logits")
    if int(logits.shape[0]) != 1 or int(logits.shape[1]) < keep:
        raise RuntimeError("Candidate forward returned too few trailing logits")
    return logits[0, -keep:].detach().float().cpu()


@torch.inference_mode()
def carrier_no_cache_tail_logits(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    boundary_position: int,
    boundary_targets: Mapping[int, np.ndarray | torch.Tensor],
    kv_directions: Mapping[
        tuple[int, str], Mapping[int, np.ndarray | torch.Tensor]
    ],
    logits_to_keep: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply a carrier during a no-cache candidate-scoring forward."""

    boundary, targets, kv = _prepare_carrier(
        encoding,
        boundary_position=boundary_position,
        boundary_targets=boundary_targets,
        kv_directions=kv_directions,
    )
    active_layers = set(targets) | {layer for layer, _kind in kv}
    if any(layer < 0 or layer >= int(adapter.num_layers) for layer in active_layers):
        raise ValueError("Carrier candidate-scoring layer is outside the decoder")
    handles, boundary_apps, kv_apps, boundary_norms, kv_norms = (
        _install_carrier_hooks(
            adapter,
            encoding,
            boundary=boundary,
            targets=targets,
            kv=kv,
        )
    )
    try:
        logits = no_cache_tail_logits(
            model,
            encoding,
            logits_to_keep=int(logits_to_keep),
        )
    finally:
        for handle in handles:
            handle.remove()
    _validate_applications(boundary_apps, kv_apps)
    return logits, {
        "boundary_hook_applications": boundary_apps,
        "kv_hook_applications": kv_apps,
        "boundary_realized_l2_norms": boundary_norms,
        "kv_realized_l2_norms": kv_norms,
    }


@torch.inference_mode()
def carrier_capture_positions(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    boundary_position: int,
    boundary_targets: Mapping[int, np.ndarray | torch.Tensor],
    kv_directions: Mapping[
        tuple[int, str], Mapping[int, np.ndarray | torch.Tensor]
    ],
    read_positions: Sequence[int],
    read_layer: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply a carrier intervention to a full trace and capture later states."""

    boundary, targets, kv = _prepare_carrier(
        encoding,
        boundary_position=boundary_position,
        boundary_targets=boundary_targets,
        kv_directions=kv_directions,
    )
    reads = tuple(int(value) for value in read_positions)
    active_read_layer = int(read_layer)
    if not reads or len(set(reads)) != len(reads):
        raise ValueError("Carrier read positions must be unique and nonempty")
    if boundary > min(reads) or max(reads) >= int(encoding.sequence_length):
        raise ValueError("Carrier boundary/read positions are inconsistent")
    active_layers = set(targets) | {layer for layer, _kind in kv}
    if any(layer >= active_read_layer for layer in active_layers):
        raise ValueError("Carrier edits must precede the read layer")

    handles, boundary_apps, kv_apps, boundary_norms, kv_norms = (
        _install_carrier_hooks(
            adapter,
            encoding,
            boundary=boundary,
            targets=targets,
            kv=kv,
        )
    )
    captured: torch.Tensor | None = None
    read_applications = 0

    def read_hook(_module: Any, args: tuple[Any, ...]) -> None:
        nonlocal captured, read_applications
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Carrier read block input is not a tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
            return
        captured = hidden[0, list(reads)].detach().float().cpu()
        read_applications += 1

    handles.append(adapter.layers[active_read_layer].register_forward_pre_hook(read_hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    _validate_applications(boundary_apps, kv_apps)
    if read_applications != 1 or captured is None:
        raise RuntimeError("Carrier read hook must apply exactly once")
    return captured, {
        "boundary_hook_applications": boundary_apps,
        "kv_hook_applications": kv_apps,
        "boundary_realized_l2_norms": boundary_norms,
        "kv_realized_l2_norms": kv_norms,
        "read_hook_applications": read_applications,
    }


@torch.inference_mode()
def carrier_capture_layer_positions(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    boundary_position: int,
    boundary_targets: Mapping[int, np.ndarray | torch.Tensor],
    kv_directions: Mapping[
        tuple[int, str], Mapping[int, np.ndarray | torch.Tensor]
    ],
    read_positions: Sequence[int],
    read_layers: Sequence[int],
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Apply one carrier and capture the same positions at several layers.

    Read hooks are registered after carrier hooks.  Consequently, if a read
    layer is also clamped, the current boundary is observed after that
    layer-input replacement.  Other positions at that layer are unchanged by
    the replacement itself and still expose how earlier layers propagated the
    intervention.  This ordering is useful for locating when a later boundary
    reacquires its native count state without repeating the expensive forward.
    """

    boundary, targets, kv = _prepare_carrier(
        encoding,
        boundary_position=boundary_position,
        boundary_targets=boundary_targets,
        kv_directions=kv_directions,
    )
    reads = tuple(int(value) for value in read_positions)
    layers = tuple(sorted({int(value) for value in read_layers}))
    if not reads or len(set(reads)) != len(reads):
        raise ValueError("Carrier read positions must be unique and nonempty")
    if not layers or any(
        layer < 0 or layer >= int(adapter.num_layers) for layer in layers
    ):
        raise ValueError("Carrier read layers are invalid")
    if boundary > min(reads) or max(reads) >= int(encoding.sequence_length):
        raise ValueError("Carrier boundary/read positions are inconsistent")
    active_layers = set(targets) | {layer for layer, _kind in kv}
    if active_layers and max(active_layers) >= int(adapter.num_layers):
        raise ValueError("Carrier edit layer is outside the decoder")

    handles, boundary_apps, kv_apps, boundary_norms, kv_norms = (
        _install_carrier_hooks(
            adapter,
            encoding,
            boundary=boundary,
            targets=targets,
            kv=kv,
        )
    )
    captured: dict[int, torch.Tensor] = {}
    read_applications = {layer: 0 for layer in layers}
    for layer in layers:

        def read_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Carrier read block input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != int(
                encoding.sequence_length
            ):
                return
            captured[layer] = hidden[0, list(reads)].detach().float().cpu()
            read_applications[layer] += 1

        handles.append(adapter.layers[layer].register_forward_pre_hook(read_hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    _validate_applications(boundary_apps, kv_apps)
    if any(read_applications[layer] != 1 for layer in layers) or set(captured) != set(
        layers
    ):
        raise RuntimeError(
            "Every carrier layer read must apply exactly once: "
            f"{read_applications}"
        )
    return captured, {
        "boundary_hook_applications": boundary_apps,
        "kv_hook_applications": kv_apps,
        "boundary_realized_l2_norms": boundary_norms,
        "kv_realized_l2_norms": kv_norms,
        "read_hook_applications": read_applications,
    }


@torch.inference_mode()
def carrier_prefill(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    boundary_position: int,
    boundary_targets: Mapping[int, np.ndarray | torch.Tensor],
    kv_directions: Mapping[
        tuple[int, str], Mapping[int, np.ndarray | torch.Tensor]
    ],
) -> tuple[Any, dict[str, Any]]:
    """Apply the identical carrier intervention during a cached prefix prefill."""

    boundary, targets, kv = _prepare_carrier(
        encoding,
        boundary_position=boundary_position,
        boundary_targets=boundary_targets,
        kv_directions=kv_directions,
    )
    active_layers = set(targets) | {layer for layer, _kind in kv}
    if any(layer < 0 or layer >= int(adapter.num_layers) for layer in active_layers):
        raise ValueError("Carrier prefill layer is outside the decoder")
    handles, boundary_apps, kv_apps, boundary_norms, kv_norms = (
        _install_carrier_hooks(
            adapter,
            encoding,
            boundary=boundary,
            targets=targets,
            kv=kv,
        )
    )
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    _validate_applications(boundary_apps, kv_apps)
    return prefill, {
        "boundary_hook_applications": boundary_apps,
        "kv_hook_applications": kv_apps,
        "boundary_realized_l2_norms": boundary_norms,
        "kv_realized_l2_norms": kv_norms,
    }
