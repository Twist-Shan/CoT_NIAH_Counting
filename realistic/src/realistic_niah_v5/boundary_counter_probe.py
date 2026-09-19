"""Frozen probes for current-count information at natural list boundaries."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _replace_output_tensor,
    _tensor_from_output,
)

from .encoding import NativeTraceEncoding


def _prepare_features(
    states: np.ndarray,
    *,
    mean: np.ndarray,
) -> np.ndarray:
    centered = np.asarray(states, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, 1e-12)


def fit_dual_ridge_count_probe(
    states: np.ndarray,
    labels: Sequence[int],
    *,
    alpha: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Fit a balanced 10-way ridge classifier in the sample-space dual."""

    x = np.asarray(states, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("Probe states/labels have incompatible shapes")
    if set(int(value) for value in np.unique(y)) != set(range(1, 11)):
        raise ValueError("Count probe requires balanced labels 1..10")
    mean = x.mean(axis=0, keepdims=True)
    z = _prepare_features(x, mean=mean)
    targets = np.eye(10, dtype=np.float64)[y - 1]
    gram = z @ z.T
    dual = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64),
        targets,
    )
    weights = z.T @ dual
    return {
        "mean": mean.astype(np.float32),
        "weights": weights.astype(np.float32),
        "alpha": float(alpha),
    }


def count_probe_scores(
    probe: dict[str, Any], states: np.ndarray
) -> np.ndarray:
    z = _prepare_features(
        np.asarray(states, dtype=np.float64),
        mean=np.asarray(probe["mean"], dtype=np.float64),
    )
    return z @ np.asarray(probe["weights"], dtype=np.float64)


def count_probe_predictions(
    probe: dict[str, Any], states: np.ndarray
) -> np.ndarray:
    return np.argmax(count_probe_scores(probe, states), axis=1).astype(np.int64) + 1


def count_probe_subspace(
    probe: Mapping[str, Any],
    *,
    relative_tolerance: float = 1e-6,
) -> np.ndarray:
    """Return the orthonormal discriminative span of a frozen count probe.

    The shared component of all ten class weights cannot affect an argmax, so
    it is removed before the SVD.  The resulting basis has at most rank nine
    and lives in residual-stream coordinates.
    """

    weights = np.asarray(probe["weights"], dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != 10:
        raise ValueError("Count-probe weights must have shape [hidden,10]")
    if not 0.0 < float(relative_tolerance) < 1.0:
        raise ValueError("Subspace tolerance must lie strictly between zero and one")
    discriminative = weights - weights.mean(axis=1, keepdims=True)
    left, singular, _right = np.linalg.svd(discriminative, full_matrices=False)
    if singular.size == 0 or float(singular[0]) <= 0.0:
        raise ValueError("Count probe has no nonzero discriminative direction")
    rank = int(np.sum(singular > float(relative_tolerance) * float(singular[0])))
    if not 1 <= rank <= 9:
        raise RuntimeError(f"Unexpected count-subspace rank: {rank}")
    basis = left[:, :rank]
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(rank), atol=1e-6, rtol=1e-6):
        raise RuntimeError("Count-subspace basis is not orthonormal")
    return basis.astype(np.float32)


def projected_donor_replacement(
    receiver_state: np.ndarray,
    donor_state: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Swap only the frozen count-subspace component of one state."""

    receiver = np.asarray(receiver_state, dtype=np.float64).reshape(-1)
    donor = np.asarray(donor_state, dtype=np.float64).reshape(-1)
    active_basis = np.asarray(basis, dtype=np.float64)
    if donor.shape != receiver.shape:
        raise ValueError("Receiver and donor states must have the same width")
    if active_basis.ndim != 2 or active_basis.shape[0] != receiver.size:
        raise ValueError("Count basis width does not match the hidden state")
    delta = donor - receiver
    projected = (delta @ active_basis) @ active_basis.T
    return (
        (receiver + projected).astype(np.float32),
        projected.astype(np.float32),
    )


def norm_matched_orthogonal_replacement(
    receiver_state: np.ndarray,
    target_delta: np.ndarray,
    basis: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a deterministic equal-norm delta orthogonal to the count span."""

    receiver = np.asarray(receiver_state, dtype=np.float64).reshape(-1)
    delta = np.asarray(target_delta, dtype=np.float64).reshape(-1)
    active_basis = np.asarray(basis, dtype=np.float64)
    if delta.shape != receiver.shape:
        raise ValueError("Target delta and receiver widths disagree")
    if active_basis.ndim != 2 or active_basis.shape[0] != receiver.size:
        raise ValueError("Count basis width does not match the hidden state")
    target_norm = float(np.linalg.norm(delta))
    if target_norm <= 1e-12:
        zero = np.zeros_like(receiver, dtype=np.float32)
        return receiver.astype(np.float32), zero
    rng = np.random.default_rng(int(seed))
    random = rng.standard_normal(receiver.shape)
    random = random - (random @ active_basis) @ active_basis.T
    # Also make the control orthogonal to the realized count-subspace delta.
    random = random - (float(random @ delta) / max(float(delta @ delta), 1e-12)) * delta
    random_norm = float(np.linalg.norm(random))
    if random_norm <= 1e-12:
        raise RuntimeError("Failed to sample a nonzero orthogonal control delta")
    matched = random * (target_norm / random_norm)
    return (
        (receiver + matched).astype(np.float32),
        matched.astype(np.float32),
    )


@torch.inference_mode()
def capture_attention_value_states(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    positions: Sequence[int],
    *,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Capture Qwen-style value states as [position,kv_head,head_dim]."""

    active_positions = tuple(int(value) for value in positions)
    active_layers = tuple(sorted({int(value) for value in layers}))
    if not active_positions or len(set(active_positions)) != len(active_positions):
        raise ValueError("Value-capture positions must be nonempty and unique")
    if min(active_positions) < 0 or max(active_positions) >= int(encoding.sequence_length):
        raise ValueError("A value-capture position is outside the encoding")
    if not active_layers or active_layers[0] < 0 or active_layers[-1] >= int(adapter.num_layers):
        raise ValueError("A value-capture layer is outside the decoder")
    captured: dict[int, torch.Tensor] = {}
    applications = {layer: 0 for layer in active_layers}
    handles = []
    for layer in active_layers:
        attention = adapter.attentions[layer]
        projection = next(
            (
                getattr(attention, name)
                for name in ("v_proj", "value", "value_proj")
                if isinstance(getattr(attention, name, None), torch.nn.Module)
            ),
            None,
        )
        if projection is None:
            raise RuntimeError(
                f"L{layer} attention exposes no supported value projection"
            )
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
            head_dim: int = head_dim,
        ) -> None:
            value = _tensor_from_output(output)
            if value.ndim != 3 or int(value.shape[0]) != 1:
                raise RuntimeError("Value projection returned an unsupported tensor")
            if int(value.shape[1]) != int(encoding.sequence_length):
                return
            width = int(value.shape[-1])
            if width % head_dim:
                raise RuntimeError("Value width is not divisible by the head dimension")
            selected = value[0, list(active_positions)]
            captured[layer] = selected.reshape(
                len(active_positions), width // head_dim, head_dim
            ).detach().float().cpu()
            applications[layer] += 1

        handles.append(projection.register_forward_hook(hook))
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
    bad = {layer: count for layer, count in applications.items() if count != 1}
    if bad or set(captured) != set(active_layers):
        raise RuntimeError(f"Value-capture hooks did not apply exactly once: {bad}")
    return captured


@torch.inference_mode()
def boundary_value_edge_write(
    adapter: DecoderAdapter,
    *,
    layer: int,
    attention_row: torch.Tensor,
    key_start: int,
    source_position: int,
    source_value: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Map one source token's attention-weighted value through W_O."""

    layer = int(layer)
    row = torch.as_tensor(attention_row).detach().float().cpu()
    value = torch.as_tensor(source_value).detach().float().cpu()
    if row.ndim != 2 or int(row.shape[0]) != int(adapter.num_heads[layer]):
        raise ValueError("Attention row must have shape [query_heads,key]")
    head_dim = int(adapter.head_dims[layer])
    if value.ndim != 2 or int(value.shape[-1]) != head_dim:
        raise ValueError("Source value must have shape [kv_heads,head_dim]")
    relative = int(source_position) - int(key_start)
    if not 0 <= relative < int(row.shape[-1]):
        raise ValueError("Source position is outside the attention key row")
    query_heads = int(row.shape[0])
    kv_heads = int(value.shape[0])
    if query_heads % kv_heads:
        raise ValueError("Query heads are not divisible by KV heads")
    groups = query_heads // kv_heads
    projection = adapter.output_projections[layer]
    parameter = next(iter(projection.parameters()), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    pre_o = torch.zeros(query_heads * head_dim, device=device, dtype=dtype)
    weights = row[:, relative]
    for head in range(query_heads):
        kv_head = head // groups
        left = head * head_dim
        pre_o[left : left + head_dim] = (
            weights[head].to(device=device, dtype=dtype)
            * value[kv_head].to(device=device, dtype=dtype)
        )
    zero = torch.zeros_like(pre_o)
    write = (
        projection(pre_o.reshape(1, 1, -1))[0, 0]
        - projection(zero.reshape(1, 1, -1))[0, 0]
    ).detach().float().cpu()
    return write, {
        "source_attention_mass_sum": float(weights.sum()),
        "source_attention_mass_max": float(weights.max()),
        "source_write_l2_norm": float(torch.linalg.vector_norm(write)),
        "query_heads": query_heads,
        "kv_heads": kv_heads,
    }


@torch.inference_mode()
def add_attention_output_deltas_and_capture_positions(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    output_deltas: Mapping[int, Mapping[int, torch.Tensor | np.ndarray]],
    read_positions: Sequence[int],
    read_layer: int,
) -> tuple[torch.Tensor, dict[int, int], dict[int, dict[int, float]], int]:
    """Add frozen residual deltas to attention outputs, then capture read sites."""

    positions = tuple(int(value) for value in read_positions)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Read positions must be nonempty and unique")
    read_layer = int(read_layer)
    deltas = {
        int(layer): {
            int(position): torch.as_tensor(delta).detach().float().cpu().reshape(-1)
            for position, delta in by_position.items()
        }
        for layer, by_position in output_deltas.items()
    }
    active_layers = tuple(sorted(deltas))
    if not active_layers or active_layers[0] < 0 or active_layers[-1] >= read_layer:
        raise ValueError("Attention-output intervention layers must precede the read layer")
    applications = {layer: 0 for layer in active_layers}
    realized = {layer: {} for layer in active_layers}
    read_applications = 0
    captured: torch.Tensor | None = None
    handles = []
    for layer in active_layers:
        by_position = deltas[layer]
        if not by_position:
            raise ValueError("Every intervention layer needs at least one target position")

        def hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
            by_position: Mapping[int, torch.Tensor] = by_position,
        ) -> Any:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
                return output
            patched = hidden.clone()
            for position, raw_delta in by_position.items():
                if not 0 <= position < int(hidden.shape[1]):
                    raise RuntimeError("Attention-output intervention position is invalid")
                delta = raw_delta.to(device=hidden.device, dtype=hidden.dtype)
                if delta.shape != hidden[0, position].shape:
                    raise RuntimeError("Attention-output delta width mismatch")
                patched[0, position] = patched[0, position] + delta
                realized[layer][position] = float(
                    torch.linalg.vector_norm(delta.float()).detach().cpu()
                )
            applications[layer] += 1
            return _replace_output_tensor(output, patched)

        handles.append(adapter.output_projections[layer].register_forward_hook(hook))

    def read_hook(_module: Any, args: tuple[Any, ...]) -> None:
        nonlocal captured, read_applications
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Attention-edge read hook saw no hidden tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
            return
        captured = hidden[0, list(positions)].detach().float().cpu()
        read_applications += 1

    handles.append(adapter.layers[read_layer].register_forward_pre_hook(read_hook))
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
    bad = {layer: count for layer, count in applications.items() if count != 1}
    if bad or read_applications != 1 or captured is None:
        raise RuntimeError(
            "Attention-output/read hooks must apply once; "
            f"delta={bad} read={read_applications}"
        )
    return captured, applications, realized, read_applications


def count_prediction_metrics(
    labels: Sequence[int], predictions: Sequence[int]
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if y.shape != pred.shape or y.ndim != 1:
        raise ValueError("Count metrics require aligned one-dimensional arrays")
    return {
        "n": int(y.size),
        "exact": int(np.sum(y == pred)),
        "exact_accuracy": float(np.mean(y == pred)),
        "mae": float(np.mean(np.abs(y - pred))),
        "within_one_accuracy": float(np.mean(np.abs(y - pred) <= 1)),
    }


def leave_one_seed_out_probe_metrics(
    states: np.ndarray,
    labels: Sequence[int],
    seed_ids: Sequence[int],
    *,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Evaluate a probe with entire seeds held out from every fit."""

    x = np.asarray(states, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    seeds = np.asarray(seed_ids, dtype=np.int64)
    if x.shape[0] != y.size or y.size != seeds.size:
        raise ValueError("LOSO probe arrays have incompatible lengths")
    predictions = np.zeros_like(y)
    per_seed: list[dict[str, Any]] = []
    for seed in sorted(int(value) for value in np.unique(seeds)):
        test = seeds == seed
        train = ~test
        probe = fit_dual_ridge_count_probe(x[train], y[train], alpha=float(alpha))
        predictions[test] = count_probe_predictions(probe, x[test])
        per_seed.append(
            {
                "seed": seed,
                **count_prediction_metrics(y[test], predictions[test]),
                "predictions": predictions[test].tolist(),
                "labels": y[test].tolist(),
            }
        )
    return {
        **count_prediction_metrics(y, predictions),
        "predictions": predictions.tolist(),
        "labels": y.tolist(),
        "per_seed": per_seed,
        "alpha": float(alpha),
    }


@torch.inference_mode()
def transplant_boundary_and_capture_later_state(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    patch_position: int,
    patch_layer: int,
    replacement_state: torch.Tensor,
    read_position: int,
    read_layer: int,
) -> tuple[torch.Tensor, int, int, float]:
    """Patch one natural boundary and capture a causally later boundary."""

    patch_position = int(patch_position)
    read_position = int(read_position)
    patch_layer = int(patch_layer)
    read_layer = int(read_layer)
    if not 0 <= patch_layer < read_layer < int(adapter.num_layers):
        raise ValueError("Transition read layer must be later than patch layer")
    if not 0 <= patch_position < read_position < int(encoding.sequence_length):
        raise ValueError("Transition read position must be causally after patch position")
    state = torch.as_tensor(replacement_state).detach().float().cpu()
    if state.ndim == 2 and int(state.shape[0]) == 1:
        state = state[0]
    if state.ndim != 1:
        raise ValueError("Boundary replacement must be one hidden vector")
    patch_applications = 0
    read_applications = 0
    realized_norm = 0.0
    captured: torch.Tensor | None = None

    def patch_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        nonlocal patch_applications, realized_norm
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Patch block input is not a tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != encoding.sequence_length:
            return None
        replacement = state.to(device=hidden.device, dtype=hidden.dtype)
        before = hidden[0, patch_position]
        if replacement.shape != before.shape:
            raise RuntimeError("Boundary replacement hidden width mismatch")
        patched = hidden.clone()
        patched[0, patch_position] = replacement
        realized_norm = float(
            torch.linalg.vector_norm(before.float() - replacement.float())
            .detach()
            .cpu()
        )
        patch_applications += 1
        return (patched, *args[1:])

    def read_hook(_module: Any, args: tuple[Any, ...]) -> None:
        nonlocal read_applications, captured
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Read block input is not a tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != encoding.sequence_length:
            return
        captured = hidden[0, read_position].detach().float().cpu()
        read_applications += 1

    patch_handle = adapter.layers[patch_layer].register_forward_pre_hook(patch_hook)
    read_handle = adapter.layers[read_layer].register_forward_pre_hook(read_hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        patch_handle.remove()
        read_handle.remove()
    if patch_applications != 1 or read_applications != 1 or captured is None:
        raise RuntimeError(
            "Transition patch/read hooks must each apply once; "
            f"patch={patch_applications} read={read_applications}"
        )
    return captured, patch_applications, read_applications, realized_norm


@torch.inference_mode()
def clamp_boundary_layers_and_capture_later_state(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    patch_position: int,
    replacement_states: dict[int, torch.Tensor],
    read_position: int,
    read_layer: int,
) -> tuple[torch.Tensor, dict[int, int], dict[int, float], int]:
    """Clamp one boundary through a layer band, then read a later state."""

    patch_position = int(patch_position)
    read_position = int(read_position)
    read_layer = int(read_layer)
    replacements = {
        int(layer): torch.as_tensor(state).detach().float().cpu().reshape(-1)
        for layer, state in replacement_states.items()
    }
    patch_layers = tuple(sorted(replacements))
    if not patch_layers or patch_layers != tuple(range(patch_layers[0], patch_layers[-1] + 1)):
        raise ValueError("Boundary clamp layers must be a nonempty contiguous band")
    if not 0 <= patch_layers[0] <= patch_layers[-1] < read_layer < int(adapter.num_layers):
        raise ValueError("Boundary clamp band must end before the read layer")
    if not 0 <= patch_position <= read_position < int(encoding.sequence_length):
        raise ValueError("Clamp read position must not causally precede the patch")
    applications = {layer: 0 for layer in patch_layers}
    norms = {layer: 0.0 for layer in patch_layers}
    read_applications = 0
    captured: torch.Tensor | None = None
    handles = []

    for layer in patch_layers:
        def patch_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Clamp block input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != encoding.sequence_length:
                return None
            replacement = replacements[layer].to(
                device=hidden.device, dtype=hidden.dtype
            )
            before = hidden[0, patch_position]
            if replacement.shape != before.shape:
                raise RuntimeError("Clamp replacement hidden width mismatch")
            patched = hidden.clone()
            patched[0, patch_position] = replacement
            norms[layer] = float(
                torch.linalg.vector_norm(before.float() - replacement.float())
                .detach()
                .cpu()
            )
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(patch_hook))

    def read_hook(_module: Any, args: tuple[Any, ...]) -> None:
        nonlocal captured, read_applications
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Clamp read block input is not a tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != encoding.sequence_length:
            return
        captured = hidden[0, read_position].detach().float().cpu()
        read_applications += 1

    handles.append(adapter.layers[read_layer].register_forward_pre_hook(read_hook))
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
    if any(value != 1 for value in applications.values()) or read_applications != 1 or captured is None:
        raise RuntimeError(
            "Every clamp/read hook must apply once; "
            f"clamp={applications} read={read_applications}"
        )
    return captured, applications, norms, read_applications


@torch.inference_mode()
def clamp_boundary_layers_and_capture_positions(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    patch_position: int,
    replacement_states: Mapping[int, torch.Tensor | np.ndarray],
    read_positions: Sequence[int],
    read_layer: int,
) -> tuple[torch.Tensor, dict[int, int], dict[int, float], int]:
    """Clamp one boundary through a layer band and capture several sites once."""

    patch_position = int(patch_position)
    positions = tuple(int(value) for value in read_positions)
    read_layer = int(read_layer)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Read positions must be nonempty and unique")
    replacements = {
        int(layer): torch.as_tensor(state).detach().float().cpu().reshape(-1)
        for layer, state in replacement_states.items()
    }
    patch_layers = tuple(sorted(replacements))
    if not patch_layers or patch_layers != tuple(range(patch_layers[0], patch_layers[-1] + 1)):
        raise ValueError("Boundary clamp layers must be a nonempty contiguous band")
    if not 0 <= patch_layers[0] <= patch_layers[-1] < read_layer < int(adapter.num_layers):
        raise ValueError("Boundary clamp band must end before the read layer")
    if not 0 <= patch_position <= min(positions):
        raise ValueError("Every captured position must be at or after the patch")
    if max(positions) >= int(encoding.sequence_length):
        raise ValueError("A captured position is outside the encoding")
    applications = {layer: 0 for layer in patch_layers}
    norms = {layer: 0.0 for layer in patch_layers}
    read_applications = 0
    captured: torch.Tensor | None = None
    handles = []

    for layer in patch_layers:
        def patch_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Clamp block input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != encoding.sequence_length:
                return None
            replacement = replacements[layer].to(
                device=hidden.device, dtype=hidden.dtype
            )
            before = hidden[0, patch_position]
            if replacement.shape != before.shape:
                raise RuntimeError("Clamp replacement hidden width mismatch")
            patched = hidden.clone()
            patched[0, patch_position] = replacement
            norms[layer] = float(
                torch.linalg.vector_norm(before.float() - replacement.float())
                .detach()
                .cpu()
            )
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(patch_hook))

    def read_hook(_module: Any, args: tuple[Any, ...]) -> None:
        nonlocal captured, read_applications
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Clamp read block input is not a tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != encoding.sequence_length:
            return
        captured = hidden[0, list(positions)].detach().float().cpu()
        read_applications += 1

    handles.append(adapter.layers[read_layer].register_forward_pre_hook(read_hook))
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
    if any(value != 1 for value in applications.values()) or read_applications != 1 or captured is None:
        raise RuntimeError(
            "Every clamp/read hook must apply once; "
            f"clamp={applications} read={read_applications}"
        )
    return captured, applications, norms, read_applications
