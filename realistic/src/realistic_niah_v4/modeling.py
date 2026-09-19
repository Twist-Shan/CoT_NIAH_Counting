from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .prompts import PromptEncoding, TokenSpan
from .spec import V4ModelSpec


def _tokenizer_compatibility_kwargs(model_spec: V4ModelSpec) -> dict[str, Any]:
    """Return narrow loader overrides for known tokenizer/config mismatches.

    The pinned Gemma 4 snapshot stores ``extra_special_tokens`` as a list,
    while transformers 4.57's Gemma tokenizer expects a name-to-token mapping.
    The extra entry is a video placeholder and is unused by the text-only NIAH
    experiments, so an empty mapping preserves the text vocabulary while
    avoiding a loader-time type error.
    """

    if model_spec.label == "Gemma4-E4B":
        return {"extra_special_tokens": {}}
    return {}


@dataclass(frozen=True)
class DecoderAdapter:
    layer_container_name: str
    layers: tuple[nn.Module, ...]
    layer_names: tuple[str, ...]
    attentions: tuple[nn.Module, ...]
    output_projections: tuple[nn.Module, ...]
    num_heads: tuple[int, ...]
    head_dims: tuple[int, ...]
    layer_types: tuple[str, ...]

    @property
    def num_layers(self) -> int:
        return len(self.layers)


def _text_config(model: nn.Module) -> Any:
    config = getattr(model, "config", None)
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return getter()
        except TypeError:
            return getter(decoder=True)
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def _attention_module(block: nn.Module) -> nn.Module:
    for name in ("self_attn", "attn", "attention"):
        module = getattr(block, name, None)
        if isinstance(module, nn.Module):
            return module
    raise ValueError(f"Cannot find a self-attention module in {type(block).__name__}")


def _output_projection(attention: nn.Module) -> nn.Module:
    for name in ("o_proj", "out_proj", "c_proj", "dense", "post"):
        module = getattr(attention, name, None)
        if isinstance(module, nn.Module):
            return module
    raise ValueError(
        f"Cannot find an attention output projection in {type(attention).__name__}"
    )


def _linear_input_features(module: nn.Module) -> int | None:
    value = getattr(module, "in_features", None)
    if value is not None:
        return int(value)
    wrapped = getattr(module, "linear", None)
    if isinstance(wrapped, nn.Module):
        value = getattr(wrapped, "in_features", None)
        if value is not None:
            return int(value)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[1])
    return None


def _module_num_heads(attention: nn.Module, text_config: Any) -> int:
    for source in (attention, getattr(attention, "config", None), text_config):
        for name in ("num_heads", "num_attention_heads", "n_head"):
            value = getattr(source, name, None)
            if value is not None:
                return int(value)
    raise ValueError(f"Cannot determine head count for {type(attention).__name__}")


def _module_head_dim(
    attention: nn.Module,
    output_projection: nn.Module,
    num_heads: int,
    text_config: Any,
) -> int:
    value = getattr(attention, "head_dim", None)
    projection_width = _linear_input_features(output_projection)
    if projection_width is not None:
        if projection_width % int(num_heads) != 0:
            raise ValueError(
                "Attention output width is not divisible by query-head count"
            )
        inferred = projection_width // int(num_heads)
        if value is not None and int(value) != inferred:
            # Gemma 4 may use a different head dimension in global layers.
            return int(inferred)
        return int(inferred)
    if value is not None:
        return int(value)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None or int(hidden_size) % int(num_heads) != 0:
        raise ValueError("Cannot infer attention head dimension")
    return int(hidden_size) // int(num_heads)


def discover_decoder_adapter(model: nn.Module) -> DecoderAdapter:
    text_config = _text_config(model)
    expected_layers = getattr(text_config, "num_hidden_layers", None)
    candidates: list[tuple[int, str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not module:
            continue
        try:
            attentions = [_attention_module(block) for block in module]
            [_output_projection(attention) for attention in attentions]
        except ValueError:
            continue
        score = len(module)
        lowered = name.lower()
        if expected_layers is not None and len(module) == int(expected_layers):
            score += 100
        if "language_model" in lowered or "text_model" in lowered:
            score += 30
        if lowered.endswith("model.layers") or lowered == "model.layers":
            score += 20
        if lowered.endswith("layers"):
            score += 5
        if "vision" in lowered or "audio" in lowered or "encoder" in lowered:
            score -= 100
        candidates.append((score, name, module))
    if not candidates:
        raise RuntimeError(
            "Could not discover decoder layers. Expected a ModuleList whose "
            "blocks expose self_attn/attn and o_proj/out_proj."
        )
    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    _, container_name, container = candidates[0]
    layers = tuple(container)
    attentions = tuple(_attention_module(block) for block in layers)
    projections = tuple(_output_projection(attention) for attention in attentions)
    heads = tuple(_module_num_heads(attention, text_config) for attention in attentions)
    head_dims = tuple(
        _module_head_dim(attention, projection, num_heads, text_config)
        for attention, projection, num_heads in zip(attentions, projections, heads)
    )
    layer_types = tuple(
        str(
            getattr(
                attention,
                "layer_type",
                getattr(attention, "attention_type", "unknown"),
            )
        )
        for attention in attentions
    )
    return DecoderAdapter(
        layer_container_name=container_name,
        layers=layers,
        layer_names=tuple(f"{container_name}.{index}" for index in range(len(layers))),
        attentions=attentions,
        output_projections=projections,
        num_heads=heads,
        head_dims=head_dims,
        layer_types=layer_types,
    )


def load_registered_model(
    model_spec: V4ModelSpec,
    *,
    cache_dir: str | Path | None = None,
    device_map: str | dict[str, Any] = "auto",
    torch_dtype: str = "bfloat16",
    attention_backend: str = "sdpa",
) -> tuple[Any, Any, DecoderAdapter]:
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=False,
        **_tokenizer_compatibility_kwargs(model_spec),
    )
    loader = getattr(transformers, model_spec.loader_class, None)
    if loader is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} does not expose "
            f"{model_spec.loader_class}; use the pinned V4 environment"
        )
    dtype = getattr(torch, str(torch_dtype), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    model = loader.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attention_backend,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    adapter = discover_decoder_adapter(model)
    return model, tokenizer, adapter


def load_registered_tokenizer(
    model_spec: V4ModelSpec,
    *,
    cache_dir: str | Path | None = None,
) -> Any:
    """Load only the pinned tokenizer for CPU-side V4 analysis."""

    import transformers

    return transformers.AutoTokenizer.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=False,
        **_tokenizer_compatibility_kwargs(model_spec),
    )


def _tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise TypeError(f"Cannot extract a hidden tensor from {type(output).__name__}")


def _replace_output_tensor(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Cannot replace a hidden tensor in {type(output).__name__}")


def _input_device(model: nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    return embeddings.weight.device


def _accepts_keyword(model: nn.Module, name: str) -> bool:
    signature = inspect.signature(model.forward)
    if name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _bounded_logits_kwargs(model: nn.Module) -> dict[str, int]:
    return {"logits_to_keep": 1} if _accepts_keyword(model, "logits_to_keep") else {}


def _encoding_tensors(
    model: nn.Module,
    encoding: PromptEncoding,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = _input_device(model)
    input_ids = torch.tensor([encoding.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [encoding.attention_mask], dtype=torch.long, device=device
    )
    return input_ids, attention_mask


def _last_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Model forward did not return [batch, time, vocab] logits")
    return logits[0, -1].detach().float().cpu()


@torch.inference_mode()
def run_last_logits(model: nn.Module, encoding: PromptEncoding) -> torch.Tensor:
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        **_bounded_logits_kwargs(model),
    )
    return _last_logits(output)


@torch.inference_mode()
def generate_answer_completion(
    model: nn.Module,
    tokenizer: Any,
    encoding: PromptEncoding,
    *,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Greedily generate the actual continuation after the final ``Total:``.

    This is behavioral evaluation, not candidate-probability scoring. The
    returned text excludes the prompt and is decoded both with and without
    special tokens so strict answer parsing remains auditable.
    """

    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be positive")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    generation_config = getattr(model, "generation_config", None)
    eos_value = (
        getattr(generation_config, "eos_token_id", None)
        if generation_config is not None
        else None
    )
    if eos_value is None:
        eos_value = getattr(tokenizer, "eos_token_id", None)
    if eos_value is None:
        eos_ids: list[int] = []
    elif isinstance(eos_value, (tuple, list, set)):
        eos_ids = [int(value) for value in eos_value]
    else:
        eos_ids = [int(eos_value)]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = eos_ids[0]
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)
    generated = model.generate(**kwargs)
    if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
        sequences = getattr(generated, "sequences", None)
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise RuntimeError("Model.generate did not return [batch, time] sequences")
        generated = sequences
    if generated.shape[0] != 1 or generated.shape[1] < input_ids.shape[1]:
        raise RuntimeError("Unexpected generated sequence shape")
    continuation = [
        int(value)
        for value in generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    ]
    if not continuation:
        raise RuntimeError("Greedy V4 generation returned an empty continuation")
    stopped_on_eos = bool(eos_ids and continuation[-1] in set(eos_ids))
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


def _normalized_layer_indices(
    adapter: DecoderAdapter,
    layers: Iterable[int] | None,
) -> tuple[int, ...]:
    if layers is None:
        return tuple(range(adapter.num_layers))
    result = tuple(sorted({int(layer) for layer in layers}))
    invalid = [layer for layer in result if not 0 <= layer < adapter.num_layers]
    if invalid:
        raise ValueError(f"Invalid decoder layer indices: {invalid}")
    return result


@torch.inference_mode()
def capture_span_states(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    spans: Sequence[TokenSpan] | None = None,
    *,
    layers: Iterable[int] | None = None,
) -> dict[str, torch.Tensor]:
    selected_spans = tuple(spans or encoding.needle_spans)
    if not selected_spans:
        raise ValueError("At least one span is required")
    selected_layers = _normalized_layer_indices(adapter, layers)
    captured_end: dict[int, torch.Tensor] = {}
    captured_mean: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            hidden = _tensor_from_output(output)
            ends = []
            means = []
            for span in selected_spans:
                if not 0 <= span.start < span.end <= hidden.shape[1]:
                    raise RuntimeError(
                        f"Span [{span.start}, {span.end}) is outside layer output"
                    )
                ends.append(hidden[0, span.end - 1].detach().float().cpu())
                means.append(
                    hidden[0, span.start : span.end].detach().float().mean(dim=0).cpu()
                )
            captured_end[layer] = torch.stack(ends)
            captured_mean[layer] = torch.stack(means)

        return hook

    for layer in selected_layers:
        handles.append(adapter.layers[layer].register_forward_hook(make_hook(layer)))
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
    missing = set(selected_layers) - set(captured_end)
    if missing:
        raise RuntimeError(f"Failed to capture decoder layers: {sorted(missing)}")
    return {
        "layer_indices": torch.tensor(selected_layers, dtype=torch.long),
        "span_end": torch.stack([captured_end[layer] for layer in selected_layers]),
        "span_mean": torch.stack([captured_mean[layer] for layer in selected_layers]),
    }


@torch.inference_mode()
def capture_post_block_states(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    positions: Sequence[int],
    *,
    layers: Iterable[int] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    selected_layers = _normalized_layer_indices(adapter, layers)
    selected_positions = tuple(int(position) for position in positions)
    if not selected_positions:
        raise ValueError("At least one capture position is required")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            hidden = _tensor_from_output(output)
            if any(
                position < 0 or position >= hidden.shape[1]
                for position in selected_positions
            ):
                raise RuntimeError("A capture position is outside layer output")
            captured[layer] = hidden[0, list(selected_positions)].detach().float().cpu()

        return hook

    for layer in selected_layers:
        handles.append(adapter.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = set(selected_layers) - set(captured)
    if missing:
        raise RuntimeError(f"Failed to capture decoder layers: {sorted(missing)}")
    return _last_logits(output), captured


def _validate_heads(
    adapter: DecoderAdapter,
    heads: Sequence[tuple[int, int]],
) -> dict[int, tuple[int, ...]]:
    by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < adapter.num_layers:
            raise ValueError(f"Invalid intervention layer: {layer}")
        if not 0 <= head < adapter.num_heads[layer]:
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    return {
        layer: tuple(sorted(set(layer_heads)))
        for layer, layer_heads in by_layer.items()
    }


def _is_prompt_prefill(value: torch.Tensor, encoding: PromptEncoding) -> bool:
    """Return whether a generation hook is seeing the original full prompt."""

    return value.ndim == 3 and int(value.shape[1]) == int(encoding.sequence_length)


@torch.inference_mode()
def generate_with_head_ablation(
    model: nn.Module,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    scope: str = "answer_query",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Greedily generate after zeroing selected attention-head outputs.

    ``answer_query`` is a one-shot, position-local intervention on the final
    prompt row. ``global`` also masks every prompt and decoding row, matching
    the global-head necessity test in synthetic V10.
    """

    if scope not in {"answer_query", "global"}:
        raise ValueError("scope must be answer_query or global")
    by_layer = _validate_heads(adapter, heads)
    if not by_layer:
        return generate_answer_completion(
            model, tokenizer, encoding, max_new_tokens=max_new_tokens
        )
    applied = {layer: 0 for layer in by_layer}
    handles = []
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])
        expected_width = int(adapter.num_heads[layer]) * head_dim

        def hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            layer_heads: tuple[int, ...] = layer_heads,
            head_dim: int = head_dim,
            expected_width: int = expected_width,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(
                    "Attention output projection did not receive a positional tensor"
                )
            value = args[0]
            if value.ndim != 3 or int(value.shape[-1]) != expected_width:
                raise RuntimeError(
                    "Expected [batch, time, query_heads * head_dim] at o_proj"
                )
            if scope == "answer_query" and not _is_prompt_prefill(value, encoding):
                return None
            patched = value.clone()
            positions: slice | list[int] = (
                slice(None)
                if scope == "global"
                else [int(encoding.query_position)]
            )
            for head in layer_heads:
                start = int(head) * head_dim
                patched[:, positions, start : start + head_dim] = 0
            applied[layer] += 1
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    try:
        result = generate_answer_completion(
            model, tokenizer, encoding, max_new_tokens=max_new_tokens
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(layer for layer, count in applied.items() if count == 0)
    if missing:
        raise RuntimeError(f"Head ablation never reached layers: {missing}")
    return {
        **result,
        "intervention_hook_applications": {
            str(layer): int(count) for layer, count in sorted(applied.items())
        },
    }


@torch.inference_mode()
def generate_with_residual_interventions(
    model: nn.Module,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    interventions: dict[int, tuple[Sequence[int], torch.Tensor]],
    *,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Greedily generate after one-shot post-block residual replacements.

    Each selected layer may replace one or more semantic prompt positions.
    Hooks deliberately skip single-token decoding forwards, so this implements
    a prefill-state intervention rather than repeatedly clamping generation.
    Supplying several layers implements the cumulative-from-layer protocol.
    """

    if not interventions:
        raise ValueError("At least one residual intervention is required")
    normalized: dict[int, tuple[tuple[int, ...], torch.Tensor]] = {}
    for raw_layer, (raw_positions, raw_states) in interventions.items():
        layer = int(raw_layer)
        if not 0 <= layer < adapter.num_layers:
            raise ValueError(f"Invalid residual intervention layer: {layer}")
        positions = tuple(int(position) for position in raw_positions)
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("Residual intervention positions must be unique")
        states = raw_states
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or int(states.shape[0]) != len(positions):
            raise ValueError(
                "Residual states must have shape [number_of_positions, hidden_size]"
            )
        normalized[layer] = (positions, states)
    applied = {layer: 0 for layer in normalized}
    handles = []
    for layer, (positions, states) in normalized.items():

        def hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
            positions: tuple[int, ...] = positions,
            states: torch.Tensor = states,
        ) -> Any:
            hidden = _tensor_from_output(output)
            if not _is_prompt_prefill(hidden, encoding):
                return output
            if any(
                position < 0 or position >= hidden.shape[1]
                for position in positions
            ):
                raise RuntimeError(
                    "A residual patch position is outside prefill output"
                )
            replacement = states.to(device=hidden.device, dtype=hidden.dtype)
            if int(replacement.shape[-1]) != int(hidden.shape[-1]):
                raise RuntimeError("Residual replacement hidden width mismatch")
            patched = hidden.clone()
            patched[:, list(positions), :] = replacement.unsqueeze(0)
            applied[layer] += 1
            return _replace_output_tensor(output, patched)

        handles.append(adapter.layers[layer].register_forward_hook(hook))
    try:
        result = generate_answer_completion(
            model, tokenizer, encoding, max_new_tokens=max_new_tokens
        )
    finally:
        for handle in handles:
            handle.remove()
    violations = sorted(layer for layer, count in applied.items() if count != 1)
    if violations:
        raise RuntimeError(
            "Residual intervention must apply exactly once per selected layer; "
            f"violations={violations}"
        )
    return {
        **result,
        "intervention_hook_applications": {
            str(layer): int(count) for layer, count in sorted(applied.items())
        },
    }


@torch.inference_mode()
def generate_with_residual_transforms(
    model: nn.Module,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    transforms: dict[
        int, tuple[Sequence[int], Callable[[torch.Tensor], torch.Tensor]]
    ],
    *,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Greedily generate after dynamic one-shot post-block transforms.

    Unlike :func:`generate_with_residual_interventions`, a transform receives
    the *actual* selected prefill activations at its layer.  This is required
    for mediation tests: an early projected donor patch may first alter the
    forward pass, after which a later hook removes only the downstream count
    subspace from the causally changed state.  Transform functions must map a
    tensor of shape ``[batch, positions, hidden]`` to the same shape.
    """

    if not transforms:
        raise ValueError("At least one residual transform is required")
    normalized: dict[
        int, tuple[tuple[int, ...], Callable[[torch.Tensor], torch.Tensor]]
    ] = {}
    for raw_layer, (raw_positions, transform) in transforms.items():
        layer = int(raw_layer)
        if not 0 <= layer < adapter.num_layers:
            raise ValueError(f"Invalid residual-transform layer: {layer}")
        positions = tuple(int(position) for position in raw_positions)
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("Residual-transform positions must be unique")
        if not callable(transform):
            raise TypeError("Residual transform must be callable")
        normalized[layer] = (positions, transform)

    applied = {layer: 0 for layer in normalized}
    handles = []
    for layer, (positions, transform) in normalized.items():

        def hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
            positions: tuple[int, ...] = positions,
            transform: Callable[[torch.Tensor], torch.Tensor] = transform,
        ) -> Any:
            hidden = _tensor_from_output(output)
            if not _is_prompt_prefill(hidden, encoding):
                return output
            if any(
                position < 0 or position >= hidden.shape[1]
                for position in positions
            ):
                raise RuntimeError(
                    "A residual-transform position is outside prefill output"
                )
            selected = hidden[:, list(positions), :]
            replacement = transform(selected)
            if not isinstance(replacement, torch.Tensor):
                raise TypeError("Residual transform must return a tensor")
            if replacement.shape != selected.shape:
                raise RuntimeError(
                    "Residual transform changed shape: "
                    f"{tuple(selected.shape)} -> {tuple(replacement.shape)}"
                )
            if replacement.device != hidden.device or replacement.dtype != hidden.dtype:
                replacement = replacement.to(device=hidden.device, dtype=hidden.dtype)
            if not torch.isfinite(replacement).all():
                raise RuntimeError("Residual transform produced non-finite values")
            patched = hidden.clone()
            patched[:, list(positions), :] = replacement
            applied[layer] += 1
            return _replace_output_tensor(output, patched)

        handles.append(adapter.layers[layer].register_forward_hook(hook))
    try:
        result = generate_answer_completion(
            model, tokenizer, encoding, max_new_tokens=max_new_tokens
        )
    finally:
        for handle in handles:
            handle.remove()
    violations = sorted(layer for layer, count in applied.items() if count != 1)
    if violations:
        raise RuntimeError(
            "Residual transform must apply exactly once per selected layer; "
            f"violations={violations}"
        )
    return {
        **result,
        "intervention_hook_applications": {
            str(layer): int(count) for layer, count in sorted(applied.items())
        },
    }


@torch.inference_mode()
def run_with_residual_patch(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    receiver_positions: Sequence[int],
    donor_states: torch.Tensor,
) -> torch.Tensor:
    layer = int(layer)
    if not 0 <= layer < adapter.num_layers:
        raise ValueError(f"Invalid patch layer: {layer}")
    positions = tuple(int(position) for position in receiver_positions)
    if donor_states.ndim == 1:
        donor_states = donor_states.unsqueeze(0)
    if donor_states.ndim != 2 or donor_states.shape[0] != len(positions):
        raise ValueError(
            "donor_states must have shape [number_of_positions, hidden_size]"
        )

    def patch(
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        hidden = _tensor_from_output(output)
        if any(position < 0 or position >= hidden.shape[1] for position in positions):
            raise RuntimeError("A residual patch position is outside layer output")
        replacement = donor_states.to(device=hidden.device, dtype=hidden.dtype)
        patched = hidden.clone()
        patched[:, list(positions), :] = replacement.unsqueeze(0)
        return _replace_output_tensor(output, patched)

    handle = adapter.layers[layer].register_forward_hook(patch)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
        return _last_logits(output)
    finally:
        handle.remove()


@torch.inference_mode()
def run_with_head_ablation(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    scope: str = "answer_query",
) -> torch.Tensor:
    if scope not in {"answer_query", "global"}:
        raise ValueError("scope must be answer_query or global")
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        layer = int(layer)
        head = int(head)
        if not 0 <= layer < adapter.num_layers:
            raise ValueError(f"Invalid ablation layer: {layer}")
        if not 0 <= head < adapter.num_heads[layer]:
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    handles = []

    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            layer: int = layer,
            layer_heads: tuple[int, ...] = tuple(layer_heads),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(
                    "Attention output projection did not receive a positional tensor"
                )
            value = args[0]
            if value.ndim != 3:
                raise RuntimeError("Expected [batch, time, head_width] at o_proj")
            patched = value.clone()
            positions: slice | list[int]
            if scope == "global":
                positions = slice(None)
            else:
                positions = [int(encoding.query_position)]
            for head in layer_heads:
                start = int(head) * head_dim
                end = start + head_dim
                patched[:, positions, start:end] = 0
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
        return _last_logits(output)
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _temporary_attention_backend(model: nn.Module, backend: str):
    configs: list[Any] = []
    seen: set[int] = set()

    def collect(config: Any) -> None:
        if config is None or id(config) in seen:
            return
        seen.add(id(config))
        configs.append(config)
        for name in getattr(config, "sub_configs", {}):
            collect(getattr(config, name, None))

    root = getattr(model, "config", None)
    collect(root)
    collect(_text_config(model))
    # Composite config setters also mutate children. Snapshot every original
    # value before changing a parent, then restore parents before children.
    saved = [(config, config._attn_implementation) for config in configs
             if hasattr(config, "_attn_implementation")]
    try:
        for config, _ in saved:
            setattr(config, "_attn_implementation", backend)
        yield
    finally:
        for config, value in saved:
            setattr(config, "_attn_implementation", value)


def _extract_attentions(output: Any) -> Sequence[Any]:
    attentions = getattr(output, "attentions", None)
    if attentions is not None:
        return attentions
    for name in (
        "language_model_output",
        "text_model_output",
        "model_output",
    ):
        nested = getattr(output, name, None)
        attentions = getattr(nested, "attentions", None)
        if attentions is not None:
            return attentions
    raise RuntimeError(
        "Query forward returned no attention tensors. Confirm that the pinned "
        "Transformers build records eager attentions."
    )


def _extract_shared_kv_states(output: Any) -> Any | None:
    shared = getattr(output, "shared_kv_states", None)
    if shared is not None:
        return shared
    for name in (
        "language_model_output",
        "text_model_output",
        "model_output",
    ):
        nested = getattr(output, name, None)
        shared = getattr(nested, "shared_kv_states", None)
        if shared is not None:
            return shared
    return None


def _attention_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if not tensors:
            raise RuntimeError("Attention output contains no tensor")
        tensor = tensors[0]
    else:
        raise RuntimeError(f"Unsupported attention output: {type(value).__name__}")
    if tensor.ndim == 5:
        # Some block/local-attention implementations expose
        # [batch, heads, blocks, query, key]. The V4 query step has exactly one
        # block-query cell, so it can be flattened without ambiguity.
        batch, heads, blocks, queries, keys = tensor.shape
        if blocks * queries != 1:
            raise RuntimeError(
                "A five-dimensional attention output contained more than one "
                f"query cell: {tuple(tensor.shape)}"
            )
        tensor = tensor.reshape(batch, heads, 1, keys)
    if tensor.ndim != 4:
        raise RuntimeError(
            f"Expected [batch, heads, query, key] attention, got {tuple(tensor.shape)}"
        )
    return tensor


@torch.inference_mode()
def position_attention_outputs(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    query_position: int,
) -> tuple[list[torch.Tensor], list[int], torch.Tensor]:
    """Return one prompt-position attention row per layer without a T-by-T map.

    The long prefix is evaluated with the configured efficient backend and a
    KV cache. Only the one-token query step switches to eager attention, so the
    code never materializes a 10k-by-10k attention matrix.
    """

    input_ids, attention_mask = _encoding_tensors(model, encoding)
    query = int(query_position)
    if not 0 < query < input_ids.shape[1]:
        raise ValueError(
            f"Attention query position {query} is outside the non-initial prompt"
        )
    prefix_ids = input_ids[:, :query]
    prefix_mask = attention_mask[:, :query]
    uses_shared_kv = any(
        bool(getattr(attention, "is_kv_shared_layer", False))
        for attention in adapter.attentions
    )
    prefix_kwargs: dict[str, Any] = {
        "input_ids": prefix_ids,
        "attention_mask": prefix_mask,
        "use_cache": True,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if uses_shared_kv:
        prefix_kwargs["return_shared_kv_states"] = True
    prefix_output = model(**prefix_kwargs)
    past = getattr(prefix_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Prefix forward did not return a KV cache")
    query_kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": attention_mask[:, : query + 1],
        "past_key_values": past,
        "use_cache": False,
        "output_attentions": True,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        query_kwargs["position_ids"] = torch.tensor(
            [[query]], dtype=torch.long, device=input_ids.device
        )
    if _accepts_keyword(model, "cache_position"):
        query_kwargs["cache_position"] = torch.tensor(
            [query], dtype=torch.long, device=input_ids.device
        )
    shared = _extract_shared_kv_states(prefix_output)
    if uses_shared_kv and shared is None:
        raise RuntimeError("Shared-KV model did not return its prefix shared states")
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        query_kwargs["shared_kv_states"] = shared
    with _temporary_attention_backend(model, "eager"):
        query_output = model(**query_kwargs)
    attentions = _extract_attentions(query_output)
    if len(attentions) != adapter.num_layers:
        raise RuntimeError(
            f"Expected {adapter.num_layers} attention layers, got {len(attentions)}"
        )
    rows: list[torch.Tensor] = []
    key_starts: list[int] = []
    for layer, value in enumerate(attentions):
        tensor = _attention_tensor(value)
        if tensor.shape[0] != 1 or tensor.shape[2] != 1:
            raise RuntimeError(
                f"Layer {layer} did not return a single query row: "
                f"{tuple(tensor.shape)}"
            )
        row = tensor[0, :, 0].detach().float().cpu()
        if row.shape[0] != adapter.num_heads[layer]:
            raise RuntimeError(f"Layer {layer} attention-head count mismatch")
        key_start = query + 1 - int(row.shape[-1])
        if key_start < 0:
            raise RuntimeError(f"Layer {layer} attention key axis is too long")
        rows.append(row)
        key_starts.append(key_start)
    return rows, key_starts, _last_logits(query_output)


@torch.inference_mode()
def query_attention_outputs(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
) -> tuple[list[torch.Tensor], list[int], torch.Tensor]:
    """Return final answer-query attention rows, absolute key starts, and logits."""

    if int(encoding.query_position) != int(encoding.sequence_length) - 1:
        raise ValueError("V4 answer query must be the final non-initial token")
    return position_attention_outputs(
        model,
        adapter,
        encoding,
        int(encoding.query_position),
    )


@torch.inference_mode()
def query_attention_rows(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
) -> tuple[list[torch.Tensor], list[int]]:
    """Return only the final answer-query attention row from every layer."""

    rows, key_starts, _logits = query_attention_outputs(model, adapter, encoding)
    return rows, key_starts
