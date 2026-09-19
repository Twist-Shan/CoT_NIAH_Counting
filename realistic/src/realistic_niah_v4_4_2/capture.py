from __future__ import annotations

import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from realistic_niah_v4.modeling import DecoderAdapter

from .prompts import TracePromptEncoding
from .spec import V442Config
from .trace import TraceBoundaries


def _torch_dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return value


def _tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise TypeError(f"Cannot extract tensor from {type(output).__name__}")


def _canonical_thd(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Normalize Q/K norm output to [time, heads, head_dim]."""

    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise RuntimeError(f"Expected Q/K tensor [1,...] rank 4; got {tuple(tensor.shape)}")
    value = tensor[0]
    time_axes = [axis for axis, size in enumerate(value.shape) if int(size) == seq_len]
    if len(time_axes) != 1:
        raise RuntimeError(
            f"Cannot identify sequence axis {seq_len} in Q/K shape {tuple(tensor.shape)}"
        )
    time_axis = time_axes[0]
    value = value.movedim(time_axis, 0)
    if value.ndim != 3:
        raise AssertionError("Q/K canonicalization lost rank")
    return value


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _selected_layers(adapter: DecoderAdapter, values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(range(adapter.num_layers)) if not tuple(values) else tuple(
        sorted({int(value) for value in values})
    )
    invalid = [value for value in result if not 0 <= value < adapter.num_layers]
    if invalid:
        raise ValueError(f"Invalid capture layers {invalid}; model has {adapter.num_layers}")
    return result


def _layer_type(attention: nn.Module) -> str:
    direct = getattr(attention, "layer_type", None)
    if direct is None:
        direct = getattr(attention, "attention_type", None)
    if direct is not None:
        return str(direct)
    config = getattr(attention, "config", None)
    layer_idx = getattr(attention, "layer_idx", None)
    layer_types = getattr(config, "layer_types", None)
    if layer_idx is not None and layer_types is not None:
        return str(layer_types[int(layer_idx)])
    if getattr(attention, "sliding_window", None) is not None:
        return "sliding_attention"
    return "unknown"


def _norm_module(attention: nn.Module, kind: str) -> nn.Module | None:
    module = getattr(attention, f"{kind}_norm", None)
    return module if isinstance(module, nn.Module) else None


def kv_source_layers(
    adapter: DecoderAdapter, target_layers: Sequence[int]
) -> dict[int, int]:
    """Resolve Gemma shared-KV layers to the prior producer of the same type."""

    result: dict[int, int] = {}
    for layer in target_layers:
        attention = adapter.attentions[layer]
        if _norm_module(attention, "k") is not None:
            result[layer] = layer
            continue
        layer_type = _layer_type(attention)
        candidates = [
            previous
            for previous in range(layer - 1, -1, -1)
            if _layer_type(adapter.attentions[previous]) == layer_type
            and _norm_module(adapter.attentions[previous], "k") is not None
        ]
        if not candidates:
            raise RuntimeError(
                f"Layer {layer} has no k_norm and no prior {layer_type!r} KV producer"
            )
        result[layer] = candidates[0]
    return result


def capture_positions(
    prompt_len: int,
    continuation_len: int,
    boundaries: TraceBoundaries,
) -> tuple[list[int], list[str]]:
    relative: list[int] = []
    roles: list[str] = []
    for position in range(boundaries.trace_start, boundaries.trace_end):
        relative.append(position)
        roles.append("trace")
    if boundaries.answer_query_start is not None and boundaries.answer_query_end is not None:
        for position in range(boundaries.answer_query_start, boundaries.answer_query_end):
            relative.append(position)
            roles.append("answer_query")
        answer_start = boundaries.answer_query_end
    else:
        answer_start = boundaries.final_start
    for position in range(answer_start, boundaries.final_end):
        if 0 <= position < continuation_len and position not in relative:
            relative.append(position)
            roles.append("answer")
    pairs = sorted(zip(relative, roles), key=lambda item: item[0])
    return [prompt_len + item[0] for item in pairs], [item[1] for item in pairs]


def _position_embeddings_from_call(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    value = kwargs.get("position_embeddings")
    if value is None and len(args) >= 2:
        value = args[1]
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or not all(isinstance(item, torch.Tensor) for item in value)
    ):
        raise RuntimeError("Attention call did not expose (cos, sin) position embeddings")
    return value[0], value[1]


def _position_table(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    value = tensor.detach()
    if value.ndim == 3 and value.shape[0] == 1 and value.shape[1] == seq_len:
        return value[0]
    if value.ndim == 2 and value.shape[0] == seq_len:
        return value
    raise RuntimeError(
        f"Unsupported RoPE table shape {tuple(value.shape)} for seq_len={seq_len}"
    )


def _attention_metadata(
    attention: nn.Module,
    *,
    layer: int,
    kv_source_layer: int,
) -> dict[str, Any]:
    head_dim = int(getattr(attention, "head_dim", 0) or 0)
    scaling = getattr(attention, "scaling", None)
    if scaling is None:
        scaling = head_dim**-0.5 if head_dim else None
    softcap = None
    for name in (
        "attn_logit_softcapping",
        "attention_logit_softcapping",
        "attention_logits_soft_cap",
        "softcap",
    ):
        value = getattr(attention, name, None)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = float(value.detach().cpu().item())
        if value not in (None, 0, 0.0):
            softcap = float(value)
            break
    sliding_window = getattr(attention, "sliding_window", None)
    if sliding_window is None:
        sliding_window = getattr(getattr(attention, "config", None), "sliding_window", None)
    is_sliding = bool(
        getattr(attention, "is_sliding", False)
        or _layer_type(attention) == "sliding_attention"
        or sliding_window is not None
    )
    return {
        "layer": int(layer),
        "layer_type": _layer_type(attention),
        "kv_source_layer": int(kv_source_layer),
        "head_dim": head_dim,
        "scaling": None if scaling is None else float(scaling),
        "softcap": softcap,
        "is_sliding": is_sliding,
        "sliding_window": int(sliding_window) if is_sliding and sliding_window else None,
    }


@torch.inference_mode()
def capture_teacher_forced_trace(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: TracePromptEncoding,
    generation: dict[str, Any],
    *,
    config: V442Config,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "capture_manifest.json"
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    continuation = tuple(int(value) for value in generation["generated_token_ids"])
    boundaries = TraceBoundaries(**generation["boundaries"])
    full_ids = tuple(encoding.input_ids) + continuation
    full_mask = tuple(encoding.attention_mask) + (1,) * len(continuation)
    seq_len = len(full_ids)
    query_positions, query_roles = capture_positions(
        encoding.prompt_token_count, len(continuation), boundaries
    )
    if encoding.mode == "nonthinking":
        if encoding.assistant_prefix_span is None:
            raise RuntimeError("Non-thinking capture requires the Total: prefix span")
        query_positions.insert(0, encoding.assistant_prefix_span[1] - 1)
        query_roles.insert(0, "answer_query")
    if not query_positions:
        raise RuntimeError("No trace/query/answer positions were selected for capture")

    layers = _selected_layers(adapter, config.capture_layers)
    kv_sources = kv_source_layers(adapter, layers)
    source_layers = sorted(set(kv_sources.values()))
    qk_dtype = _torch_dtype(config.qk_save_dtype)
    hidden_dtype = _torch_dtype(config.hidden_save_dtype)
    hidden_cache: dict[int, torch.Tensor] = {}
    q_cache: dict[int, torch.Tensor] = {}
    k_cache: dict[int, torch.Tensor] = {}
    rope_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles: list[Any] = []

    def hidden_hook(layer: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], value: Any) -> None:
            hidden = _tensor_from_output(value)
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != seq_len:
                raise RuntimeError(
                    f"Layer {layer} hidden shape mismatch: {tuple(hidden.shape)}"
                )
            hidden_cache[layer] = (
                hidden[0, query_positions].detach().to(hidden_dtype).cpu().contiguous()
            )

        return hook

    def q_hook(layer: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], value: Any) -> None:
            q = _canonical_thd(_tensor_from_output(value), seq_len)
            q_cache[layer] = q[query_positions].detach().to(qk_dtype).cpu().contiguous()

        return hook

    def k_hook(layer: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], value: Any) -> None:
            k = _canonical_thd(_tensor_from_output(value), seq_len)
            k_cache[layer] = k.detach().to(qk_dtype).cpu().contiguous()

        return hook

    def attention_pre_hook(layer: int):
        def hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            cos, sin = _position_embeddings_from_call(args, kwargs)
            rope_cache[layer] = (
                _position_table(cos, seq_len).to(qk_dtype).cpu().contiguous(),
                _position_table(sin, seq_len).to(qk_dtype).cpu().contiguous(),
            )

        return hook

    try:
        for layer in layers:
            handles.append(adapter.layers[layer].register_forward_hook(hidden_hook(layer)))
            q_norm = _norm_module(adapter.attentions[layer], "q")
            if q_norm is None:
                raise RuntimeError(f"Layer {layer} exposes no q_norm module")
            handles.append(q_norm.register_forward_hook(q_hook(layer)))
            handles.append(
                adapter.attentions[layer].register_forward_pre_hook(
                    attention_pre_hook(layer), with_kwargs=True
                )
            )
        for source in source_layers:
            k_norm = _norm_module(adapter.attentions[source], "k")
            if k_norm is None:
                raise RuntimeError(f"KV source layer {source} exposes no k_norm module")
            handles.append(k_norm.register_forward_hook(k_hook(source)))

        device = model.get_input_embeddings().weight.device
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([full_mask], dtype=torch.long, device=device)
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        try:
            _ = model(**kwargs, logits_to_keep=1)
        except TypeError as error:
            if "logits_to_keep" not in str(error):
                raise
            _ = model(**kwargs)
        elapsed = time.perf_counter() - started
    finally:
        for handle in handles:
            handle.remove()

    missing_hidden = sorted(set(layers) - set(hidden_cache))
    missing_q = sorted(set(layers) - set(q_cache))
    missing_k = sorted(set(source_layers) - set(k_cache))
    missing_rope = sorted(set(layers) - set(rope_cache))
    if missing_hidden or missing_q or missing_k or missing_rope:
        raise RuntimeError(
            "Incomplete trace capture: "
            f"hidden={missing_hidden}, q={missing_q}, k={missing_k}, rope={missing_rope}"
        )

    layer_rows = []
    for source in source_layers:
        _atomic_torch_save(
            k_cache[source], output / f"kv_source_{source:02d}_k_norm.pt"
        )
    for layer in layers:
        source = kv_sources[layer]
        cos, sin = rope_cache[layer]
        _atomic_torch_save(hidden_cache[layer], output / f"layer_{layer:02d}_hidden.pt")
        _atomic_torch_save(q_cache[layer], output / f"layer_{layer:02d}_q_norm.pt")
        _atomic_torch_save(
            {"cos": cos, "sin": sin}, output / f"layer_{layer:02d}_rope.pt"
        )
        metadata = _attention_metadata(
            adapter.attentions[layer], layer=layer, kv_source_layer=source
        )
        metadata.update(
            {
                "q_shape": list(q_cache[layer].shape),
                "k_shape": list(k_cache[source].shape),
                "k_file": f"kv_source_{source:02d}_k_norm.pt",
                "hidden_shape": list(hidden_cache[layer].shape),
                "rope_shape": list(cos.shape),
            }
        )
        layer_rows.append(metadata)

    _atomic_torch_save(torch.tensor(full_ids, dtype=torch.long), output / "input_ids.pt")
    _atomic_torch_save(
        torch.tensor(query_positions, dtype=torch.long), output / "query_positions.pt"
    )
    manifest = {
        "schema_version": "realistic_niah_v4_4_2_trace_capture_v1",
        "stimulus_id": encoding.stimulus_id,
        "model_label": encoding.model_label,
        "model_family": generation["model_family"],
        "mode": encoding.mode,
        "prompt_variant": encoding.prompt_variant,
        "seed": encoding.seed,
        "split": encoding.split,
        "gold_count": encoding.count,
        "prompt_token_count": encoding.prompt_token_count,
        "sequence_length": seq_len,
        "generated_token_count": len(continuation),
        "query_positions": query_positions,
        "query_roles": query_roles,
        "boundaries": generation["boundaries"],
        "needle_spans": [[span.start, span.end] for span in encoding.needle_spans],
        "needle_end_positions": [span.end - 1 for span in encoding.needle_spans],
        "cue_span": encoding.cue_span,
        "passage_span": encoding.passage_span,
        "question_span": encoding.question_span,
        "assistant_prefix_span": encoding.assistant_prefix_span,
        "layers": layer_rows,
        "hidden_save_dtype": config.hidden_save_dtype,
        "qk_save_dtype": config.qk_save_dtype,
        "elapsed_seconds": elapsed,
    }
    _atomic_json(manifest, manifest_path)
    return manifest
