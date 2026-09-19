from __future__ import annotations

import hashlib
import json
import random
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from dataset_generation.dynamic_niah_v2 import DynamicNiahV2Config
from .ablation_analysis import (
    _float_or_none,
    _int_or_none,
    _read_csv_dicts,
    _record_token,
    _score_row,
    _write_csv,
    _write_json,
    _write_jsonl,
    needle_positions_from_segments,
    load_input_metadata,
    select_attention_sink_tokens,
    select_attention_sink_tokens_all,
    select_massive_activation_tokens,
    select_massive_activation_tokens_all,
    select_needle_sensitive_tokens,
    select_needle_sensitive_tokens_all,
    select_needle_span_tokens,
    select_needle_tail_tokens,
)
from .single_example_analysis import (
    SingleExamplePaths,
    _flat_input_ids,
    _read_jsonl,
    render_and_tokenize_messages,
)

DEFAULT_REPRESENTATION_ABLATION_CONFIG_PATH = Path("configs/ablation-representation.json")


@dataclass(frozen=True)
class RepresentationAblationConfig:
    """Configuration for single-example representation-level ablations."""

    num_critical_tokens: int = 10
    randomize_from_top_layer: bool = True
    ablation_random_seed: int = 12345
    critical_token_calc_layer: int = 24
    patterns: tuple[str, ...] = (
        "massive_activation",
        "attention_sink",
        "needle_sensitive",
        "massive_activation_all",
        "attention_sink_all",
        "needle_sensitive_all",
        "needle_span",
        "needle_tail",
    )
    attention_sink_score: str = "received_uniform_ratio"
    max_new_tokens: int | None = None
    temperature: float = 0.0
    stats_dtype: str = "bfloat16"
    stats_accum_dtype: str = "float32"
    stats_split: str = "latter_half"
    profile_allow_single_example: bool = False
    min_std: float = 0.0
    edge_exclusion_tokens: int = 5
    save_unablated_hidden_states: bool = True

    def with_overrides(
        self,
        *,
        num_critical_tokens: int | None = None,
        randomize_from_top_layer: bool | None = None,
        ablation_random_seed: int | None = None,
        critical_token_calc_layer: int | None = None,
    ) -> "RepresentationAblationConfig":
        kwargs: dict[str, Any] = {}
        if num_critical_tokens is not None:
            kwargs["num_critical_tokens"] = int(num_critical_tokens)
        if randomize_from_top_layer is not None:
            kwargs["randomize_from_top_layer"] = bool(randomize_from_top_layer)
        if ablation_random_seed is not None:
            kwargs["ablation_random_seed"] = int(ablation_random_seed)
        if critical_token_calc_layer is not None:
            kwargs["critical_token_calc_layer"] = int(critical_token_calc_layer)
        return replace(self, **kwargs)


def _tuple_patterns(value: Any) -> tuple[str, ...]:
    if value is None:
        return RepresentationAblationConfig.patterns
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_representation_ablation_config(
    path: str | Path = DEFAULT_REPRESENTATION_ABLATION_CONFIG_PATH,
    *,
    num_critical_tokens: int | None = None,
    randomize_from_top_layer: bool | None = None,
    ablation_random_seed: int | None = None,
    critical_token_calc_layer: int | None = None,
) -> RepresentationAblationConfig:
    """Load representation-ablation defaults and apply runtime overrides."""

    config_path = Path(path)
    payload: dict[str, Any] = {}
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "patterns" in payload:
        payload["patterns"] = _tuple_patterns(payload["patterns"])
    cfg = RepresentationAblationConfig(**payload)
    return cfg.with_overrides(
        num_critical_tokens=num_critical_tokens,
        randomize_from_top_layer=randomize_from_top_layer,
        ablation_random_seed=ablation_random_seed,
        critical_token_calc_layer=critical_token_calc_layer,
    )


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower().replace("torch.", "")
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype in representation ablation config: {name}")
    return mapping[normalized]


def get_profile_indices(num_examples: int, cfg: RepresentationAblationConfig) -> list[int]:
    """Return dataset indices used to estimate hidden-state means/stds."""

    n = int(num_examples)
    if n <= 0:
        raise ValueError("Cannot profile hidden states from an empty dataset")
    if n == 1 and not cfg.profile_allow_single_example:
        raise ValueError(
            "Need at least two examples to profile the latter half of the dataset. "
            "Set profile_allow_single_example=true to reuse the only example."
        )
    if n == 1:
        return [0]
    if cfg.stats_split != "latter_half":
        raise ValueError(f"Unsupported stats_split={cfg.stats_split!r}; expected 'latter_half'")
    return list(range(n // 2, n))


def _model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _row_input_ids(tokenizer: Any, row: dict[str, Any], *, thinking_mode: bool = False) -> torch.Tensor:
    messages = row.get("uncontrolled_messages") or row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Dataset row must contain uncontrolled_messages or messages")
    return render_and_tokenize_messages(
        tokenizer, messages, thinking_mode=thinking_mode
    ).input_ids


def _decoder_hidden_states_from_forward_output(output: Any) -> torch.Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("Model forward output does not include hidden_states")
    states = list(hidden_states)
    if len(states) < 2:
        raise ValueError("Expected embedding output plus at least one decoder-layer hidden state")
    return torch.stack(states[1:], dim=0).squeeze(1)


def _pad_stats_to_shape(
    mean: torch.Tensor | None,
    m2: torch.Tensor | None,
    count: torch.Tensor | None,
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layers, seq_len, hidden_dim = shape
    if mean is None or m2 is None or count is None:
        return (
            torch.zeros(shape, dtype=dtype, device=device),
            torch.zeros(shape, dtype=dtype, device=device),
            torch.zeros((layers, seq_len, 1), dtype=dtype, device=device),
        )
    old_layers, old_seq_len, old_hidden_dim = mean.shape
    if old_layers != layers or old_hidden_dim != hidden_dim:
        raise ValueError(
            f"Hidden-state shape changed across profiling examples: "
            f"old={(old_layers, old_seq_len, old_hidden_dim)} new={shape}"
        )
    if old_seq_len >= seq_len:
        return mean, m2, count
    pad_shape = (layers, seq_len - old_seq_len, hidden_dim)
    count_pad_shape = (layers, seq_len - old_seq_len, 1)
    return (
        torch.cat([mean, torch.zeros(pad_shape, dtype=dtype, device=device)], dim=1),
        torch.cat([m2, torch.zeros(pad_shape, dtype=dtype, device=device)], dim=1),
        torch.cat([count, torch.zeros(count_pad_shape, dtype=dtype, device=device)], dim=1),
    )


def profile_hidden_state_distribution(
    *,
    model: Any,
    tokenizer: Any,
    dataset_rows: Sequence[dict[str, Any]],
    cfg: RepresentationAblationConfig,
    out_path: str | Path,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> Path:
    """Estimate per-layer/position/dimension hidden-state mean and std."""

    indices = get_profile_indices(len(dataset_rows), cfg)
    accum_dtype = _torch_dtype(cfg.stats_accum_dtype)
    save_dtype = _torch_dtype(cfg.stats_dtype)
    device = torch.device("cpu")
    mean = m2 = count = None
    input_device = _model_input_device(model)
    thinking_mode = bool(getattr(dynamic_cfg, "thinking_mode", False)) if dynamic_cfg else False

    with torch.no_grad():
        for idx in indices:
            input_ids = _row_input_ids(
                tokenizer, dataset_rows[idx], thinking_mode=thinking_mode
            ).to(input_device)
            output = model(input_ids, output_hidden_states=True, use_cache=False)
            hidden = _decoder_hidden_states_from_forward_output(output).detach().to(
                device=device, dtype=accum_dtype
            )
            mean, m2, count = _pad_stats_to_shape(
                mean,
                m2,
                count,
                tuple(hidden.shape),
                dtype=accum_dtype,
                device=device,
            )
            seq_len = hidden.shape[1]
            current_count = count[:, :seq_len, :]
            delta = hidden - mean[:, :seq_len, :]
            current_count += 1
            mean[:, :seq_len, :] += delta / current_count
            delta2 = hidden - mean[:, :seq_len, :]
            m2[:, :seq_len, :] += delta * delta2

    if mean is None or m2 is None or count is None:
        raise ValueError("No profiling examples were processed")
    variance = torch.zeros_like(mean)
    valid = count > 1
    variance = torch.where(valid, m2 / torch.clamp(count - 1, min=1), variance)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    if cfg.min_std > 0:
        std = torch.clamp(std, min=float(cfg.min_std))
    payload = {
        "schema_version": "representation_hidden_state_stats_v1",
        "mean": mean.to(save_dtype),
        "std": std.to(save_dtype),
        "count": count.to(torch.int32),
        "profile_indices": indices,
        "stats_dtype": cfg.stats_dtype,
        "stats_accum_dtype": cfg.stats_accum_dtype,
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def save_unablated_decoder_hidden_states(
    *,
    model: Any,
    input_ids: torch.Tensor,
    out_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    input_device = _model_input_device(model)
    with torch.no_grad():
        output = model(input_ids.to(input_device), output_hidden_states=True, use_cache=False)
    hidden = _decoder_hidden_states_from_forward_output(output).detach().cpu().to(dtype)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "representation_unablated_hidden_states_v1", "hidden_states": hidden}, path)
    return path


def _rank_all_massive_activation_tokens(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: RepresentationAblationConfig,
    input_ids: Sequence[int],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    return select_massive_activation_tokens_all(
        run_dir=run_dir,
        example_id=example_id,
        cfg=_as_token_ablation_cfg(cfg),
        input_ids=input_ids,
        tokenizer=tokenizer,
    )


def _rank_all_attention_sink_tokens(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: RepresentationAblationConfig,
    input_ids: Sequence[int],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    return select_attention_sink_tokens_all(
        run_dir=run_dir,
        example_id=example_id,
        cfg=_as_token_ablation_cfg(cfg),
        input_ids=input_ids,
        tokenizer=tokenizer,
    )


def _rank_all_needle_sensitive_tokens(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: RepresentationAblationConfig,
    input_ids: Sequence[int],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    return select_needle_sensitive_tokens_all(
        run_dir=run_dir,
        example_id=example_id,
        cfg=_as_token_ablation_cfg(cfg),
        input_ids=input_ids,
        tokenizer=tokenizer,
    )


def _select_needle_span_tokens(
    *,
    needle_segments: Sequence[dict[str, Any]],
    input_ids: Sequence[int],
    tokenizer: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    by_pattern: dict[str, list[dict[str, Any]]] = {}
    for needle_idx, segment in enumerate(sorted(needle_segments, key=lambda item: int(item["start"]))):
        pattern = f"needle_span_{needle_idx}"
        positions = segment.get("positions")
        if not isinstance(positions, list):
            positions = list(range(int(segment["start"]), int(segment["end"])))
        rows = []
        for rank, pos in enumerate([int(p) for p in positions if 0 <= int(p) < len(input_ids)], start=1):
            rows.append(
                _record_token(
                    pattern=pattern,
                    rank=rank,
                    position=pos,
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                    score_name="needle_span_position",
                    score=rank,
                    extra={"needle_id": segment.get("needle_id", needle_idx)},
                )
            )
        by_pattern[pattern] = rows
    return by_pattern


def _as_token_ablation_cfg(cfg: RepresentationAblationConfig) -> Any:
    from .ablation_analysis import AblationConfig

    return AblationConfig(
        num_critical_tokens=cfg.num_critical_tokens,
        critical_token_calc_layer=cfg.critical_token_calc_layer,
        ablation_random_seed=cfg.ablation_random_seed,
        patterns=("massive_activation", "attention_sink", "needle_sensitive"),
        attention_sink_score=cfg.attention_sink_score,
        edge_exclusion_tokens=cfg.edge_exclusion_tokens,
    )


def select_representation_critical_tokens(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: RepresentationAblationConfig,
    input_ids: Sequence[int],
    needle_segments: Sequence[dict[str, Any]],
    tokenizer: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Select existing filtered patterns plus representation-specific all-position patterns."""

    selected: dict[str, list[dict[str, Any]]] = {}
    patterns = set(cfg.patterns)
    token_cfg = _as_token_ablation_cfg(cfg)
    needle_positions = needle_positions_from_segments(needle_segments)
    if "massive_activation" in patterns:
        selected["massive_activation"] = select_massive_activation_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=token_cfg,
            input_ids=input_ids,
            needle_positions=needle_positions,
            tokenizer=tokenizer,
        )
    if "attention_sink" in patterns:
        selected["attention_sink"] = select_attention_sink_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=token_cfg,
            input_ids=input_ids,
            needle_positions=needle_positions,
            tokenizer=tokenizer,
        )
    if "needle_sensitive" in patterns:
        selected["needle_sensitive"] = select_needle_sensitive_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=token_cfg,
            input_ids=input_ids,
            needle_positions=needle_positions,
            tokenizer=tokenizer,
        )
    if "needle_span" in patterns:
        selected.update(
            select_needle_span_tokens(
                needle_segments=needle_segments,
                input_ids=input_ids,
                tokenizer=tokenizer,
            )
        )
    if "needle_tail" in patterns:
        selected.update(
            select_needle_tail_tokens(
                needle_segments=needle_segments,
                cfg=token_cfg,
                input_ids=input_ids,
                needle_positions=needle_positions,
                tokenizer=tokenizer,
            )
        )
    if "massive_activation_all" in patterns:
        selected["massive_activation_all"] = _rank_all_massive_activation_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
    if "attention_sink_all" in patterns:
        selected["attention_sink_all"] = _rank_all_attention_sink_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
    if "needle_sensitive_all" in patterns:
        selected["needle_sensitive_all"] = _rank_all_needle_sensitive_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
    return selected


def save_representation_critical_tokens(
    *,
    run_dir: str | Path,
    selected: dict[str, list[dict[str, Any]]],
    cfg: RepresentationAblationConfig,
) -> dict[str, Path]:
    """Save representation critical tokens without overwriting token-ablation tables."""

    ablation_dir = Path(run_dir) / "tables" / "ablation_representation"
    flat_rows = [row for rows in selected.values() for row in rows]
    fields = [
        "pattern",
        "rank",
        "position",
        "token_id",
        "token",
        "score_name",
        "score",
        "layer",
        "head",
        "needle_id",
    ]
    return {
        "json": _write_json(
            ablation_dir / "critical_tokens.json",
            {
                "config": asdict(cfg),
                "patterns": selected,
            },
        ),
        "csv": _write_csv(ablation_dir / "critical_tokens.csv", flat_rows, fields),
    }


def active_ablation_layers(layer_idx: int, num_layers: int, *, randomize_from_top_layer: bool) -> set[int]:
    """Return decoder-layer indices where hidden states should be randomized."""

    layer = int(layer_idx)
    total = int(num_layers)
    if layer < 0 or layer >= total:
        raise ValueError(f"layer_idx={layer} is out of range for {total} decoder layers")
    if randomize_from_top_layer:
        return set(range(layer, total))
    return set(range(0, layer + 1))


def resolve_decoder_layers(model: Any) -> Sequence[Any]:
    """Resolve the list of decoder blocks for common Hugging Face causal LMs."""

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(
        "Unsupported model architecture for representation ablation: could not find decoder layers"
    )


def _seed_for(*, base_seed: int, pattern: str, layer_idx: int, generation_step: int) -> int:
    payload = f"{base_seed}:{pattern}:{layer_idx}:{generation_step}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31)


def sample_replacement_hidden_states(
    *,
    stats: dict[str, Any],
    layer_idx: int,
    positions: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> dict[int, torch.Tensor]:
    """Sample replacement vectors, warning and randomizing position if stats are too short."""

    mean = stats["mean"]
    std = stats["std"]
    max_pos = int(mean.shape[1])
    py_rng = random.Random(int(generator.initial_seed()) if generator is not None else None)
    replacements: dict[int, torch.Tensor] = {}
    for original_pos in positions:
        stats_pos = int(original_pos)
        if stats_pos < 0 or stats_pos >= max_pos:
            sampled_pos = py_rng.randrange(max_pos)
            message = (
                f"Position {original_pos} exceeds profiled hidden-state statistics length "
                f"{max_pos}; using sampled statistics position {sampled_pos}."
            )
            print(f"WARNING: {message}")
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            stats_pos = sampled_pos
        mu = mean[int(layer_idx), stats_pos, :].to(device=device, dtype=dtype)
        sigma = std[int(layer_idx), stats_pos, :].to(device=device, dtype=dtype)
        noise = torch.randn(mu.shape, device=device, dtype=dtype, generator=generator)
        replacements[int(original_pos)] = mu + sigma * noise
    return replacements


def _replace_layer_output(output: Any, positions: Sequence[int], replacements: dict[int, torch.Tensor]) -> Any:
    if isinstance(output, tuple):
        hidden = output[0].clone()
        rest = output[1:]
    else:
        hidden = output.clone()
        rest = None
    seq_len = hidden.shape[1]
    for pos in positions:
        if 0 <= int(pos) < seq_len and int(pos) in replacements:
            hidden[:, int(pos), :] = replacements[int(pos)].to(device=hidden.device, dtype=hidden.dtype)
    if rest is None:
        return hidden
    return (hidden, *rest)


def _resolve_max_new_tokens(cfg: RepresentationAblationConfig, dynamic_cfg: DynamicNiahV2Config | None) -> int:
    if cfg.max_new_tokens is not None:
        return int(cfg.max_new_tokens)
    if dynamic_cfg is not None and getattr(dynamic_cfg, "max_new_tokens", None) is not None:
        return int(dynamic_cfg.max_new_tokens)
    return 64


def manual_generate_with_representation_ablation(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    cfg: RepresentationAblationConfig,
    stats: dict[str, Any] | None = None,
    pattern: str = "baseline",
    positions: Sequence[int] = (),
    layer_idx: int | None = None,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> str:
    """Greedy decode without model.generate, optionally randomizing layer outputs."""

    decoder_layers = list(resolve_decoder_layers(model))
    input_device = _model_input_device(model)
    generated = input_ids.to(input_device).clone()
    max_new_tokens = _resolve_max_new_tokens(cfg, dynamic_cfg)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    active_layers = set()
    if stats is not None and layer_idx is not None and positions:
        active_layers = active_ablation_layers(
            layer_idx, len(decoder_layers), randomize_from_top_layer=cfg.randomize_from_top_layer
        )
    for generation_step in range(max_new_tokens):
        hooks = []
        try:
            if active_layers:
                for current_layer_idx, layer in enumerate(decoder_layers):
                    if current_layer_idx not in active_layers:
                        continue

                    def hook(_module, _inputs, output, *, current_layer_idx=current_layer_idx):
                        seed = _seed_for(
                            base_seed=cfg.ablation_random_seed,
                            pattern=pattern,
                            layer_idx=int(layer_idx),
                            generation_step=generation_step,
                        )
                        generator = torch.Generator(device=input_device)
                        generator.manual_seed(seed + current_layer_idx)
                        hidden = output[0] if isinstance(output, tuple) else output
                        replacements = sample_replacement_hidden_states(
                            stats=stats,
                            layer_idx=current_layer_idx,
                            positions=positions,
                            device=hidden.device,
                            dtype=hidden.dtype,
                            generator=generator,
                        )
                        return _replace_layer_output(output, positions, replacements)

                    hooks.append(layer.register_forward_hook(hook))
            with torch.no_grad():
                output = model(
                    generated,
                    attention_mask=torch.ones_like(generated, dtype=torch.long, device=generated.device),
                    use_cache=False,
                )
        finally:
            for hook in hooks:
                hook.remove()
        logits = output.logits[:, -1, :]
        if float(cfg.temperature) > 0:
            probs = torch.softmax(logits / float(cfg.temperature), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and int(next_token[0, 0]) == int(eos_token_id):
            break
    prompt_len = int(input_ids.reshape(1, -1).shape[1])
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


def run_representation_ablation_generation(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    example_id: int,
    input_ids: torch.Tensor | Sequence[int],
    selected: dict[str, list[dict[str, Any]]],
    cfg: RepresentationAblationConfig,
    stats: dict[str, Any],
    out_dir: str | Path,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = Path(out_dir)
    generation_dir = out_dir / "generations"
    generation_dir.mkdir(parents=True, exist_ok=True)
    base_input = torch.as_tensor(input_ids, dtype=torch.long)
    if base_input.ndim == 1:
        base_input = base_input.unsqueeze(0)
    decoder_layers = list(resolve_decoder_layers(model))
    prediction_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    baseline_output = manual_generate_with_representation_ablation(
        model=model,
        tokenizer=tokenizer,
        input_ids=base_input,
        cfg=cfg,
        dynamic_cfg=dynamic_cfg,
    )
    baseline_score = _score_row(row, baseline_output)
    (generation_dir / "baseline.txt").write_text(baseline_output, encoding="utf-8")
    baseline_result = {
        "example_id": int(example_id),
        "row_id": row.get("id"),
        "pattern": "baseline",
        "layer_idx": -1,
        "randomize_from_top_layer": bool(cfg.randomize_from_top_layer),
        "num_positions": 0,
        "ablated_positions": json.dumps([]),
        "model_output_text": baseline_output,
        "parse_mode": baseline_score.get("parse_mode"),
        "exact_match": bool(baseline_score.get("exact_match")),
        "accuracy": float(baseline_score["accuracy"]),
        "seed": int(cfg.ablation_random_seed),
    }
    result_rows.append(baseline_result)
    prediction_rows.append({**baseline_score, **baseline_result})

    for pattern, tokens in selected.items():
        positions = [int(item["position"]) for item in tokens]
        if not positions:
            continue
        for layer_idx in range(len(decoder_layers)):
            output = manual_generate_with_representation_ablation(
                model=model,
                tokenizer=tokenizer,
                input_ids=base_input,
                cfg=cfg,
                stats=stats,
                pattern=pattern,
                positions=positions,
                layer_idx=layer_idx,
                dynamic_cfg=dynamic_cfg,
            )
            score = _score_row(row, output)
            generation_path = generation_dir / f"{pattern}_layer{layer_idx}.txt"
            generation_path.write_text(output, encoding="utf-8")
            result = {
                "example_id": int(example_id),
                "row_id": row.get("id"),
                "pattern": pattern,
                "layer_idx": int(layer_idx),
                "randomize_from_top_layer": bool(cfg.randomize_from_top_layer),
                "num_positions": len(positions),
                "ablated_positions": json.dumps(positions),
                "model_output_text": output,
                "parse_mode": score.get("parse_mode"),
                "exact_match": bool(score.get("exact_match")),
                "accuracy": float(score["accuracy"]),
                "seed": int(cfg.ablation_random_seed),
            }
            result_rows.append(result)
            prediction_rows.append({**score, **result})
    return prediction_rows, result_rows


def run_single_example_representation_ablation(
    *,
    paths: SingleExamplePaths,
    row: dict[str, Any],
    dataset_path: str | Path,
    example_id: int,
    model: Any,
    tokenizer: Any,
    uncontrolled_input_ids: torch.Tensor | Sequence[int] | None = None,
    needle_segments: Sequence[dict[str, Any]] | None = None,
    config_path: str | Path = DEFAULT_REPRESENTATION_ABLATION_CONFIG_PATH,
    num_critical_tokens: int | None = None,
    randomize_from_top_layer: bool | None = None,
    ablation_random_seed: int | None = None,
    critical_token_calc_layer: int | None = None,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> dict[str, Any]:
    cfg = load_representation_ablation_config(
        config_path,
        num_critical_tokens=num_critical_tokens,
        randomize_from_top_layer=randomize_from_top_layer,
        ablation_random_seed=ablation_random_seed,
        critical_token_calc_layer=critical_token_calc_layer,
    )
    if uncontrolled_input_ids is None or needle_segments is None:
        metadata = load_input_metadata(paths.run_dir, example_id)
        if uncontrolled_input_ids is None:
            uncontrolled_input_ids = metadata["uncontrolled_input_ids"]
        if needle_segments is None:
            needle_segments = metadata["needle_segments"]
    input_ids_list = _flat_input_ids(uncontrolled_input_ids)
    rep_tensors_dir = paths.tensors_dir / "ablation_representation"
    rep_tables_dir = paths.tables_dir / "ablation_representation"
    stats_path = rep_tensors_dir / "hidden_state_distribution_stats.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu")
    else:
        dataset_rows = _read_jsonl(dataset_path)
        profile_hidden_state_distribution(
            model=model,
            tokenizer=tokenizer,
            dataset_rows=dataset_rows,
            cfg=cfg,
            out_path=stats_path,
            dynamic_cfg=dynamic_cfg,
        )
        stats = torch.load(stats_path, map_location="cpu")
    if cfg.save_unablated_hidden_states:
        save_unablated_decoder_hidden_states(
            model=model,
            input_ids=torch.tensor([input_ids_list], dtype=torch.long),
            out_path=rep_tensors_dir / f"hidden_states_unablated_{int(example_id)}.pt",
            dtype=_torch_dtype(cfg.stats_dtype),
        )
    selected = select_representation_critical_tokens(
        run_dir=paths.run_dir,
        example_id=example_id,
        cfg=cfg,
        input_ids=input_ids_list,
        needle_segments=list(needle_segments),
        tokenizer=tokenizer,
    )
    critical_paths = save_representation_critical_tokens(
        run_dir=paths.run_dir, selected=selected, cfg=cfg
    )
    prediction_rows, result_rows = run_representation_ablation_generation(
        model=model,
        tokenizer=tokenizer,
        row=row,
        example_id=example_id,
        input_ids=torch.tensor([input_ids_list], dtype=torch.long),
        selected=selected,
        cfg=cfg,
        stats=stats,
        out_dir=rep_tables_dir,
        dynamic_cfg=dynamic_cfg,
    )
    predictions_path = _write_jsonl(rep_tables_dir / "ablation_representation_predictions.jsonl", prediction_rows)
    result_fields = [
        "example_id",
        "row_id",
        "pattern",
        "layer_idx",
        "randomize_from_top_layer",
        "num_positions",
        "ablated_positions",
        "model_output_text",
        "parse_mode",
        "exact_match",
        "accuracy",
        "seed",
    ]
    results_path = _write_csv(
        rep_tables_dir / "ablation_representation_results.csv", result_rows, result_fields
    )
    summary = {
        "config": asdict(cfg),
        "example_id": int(example_id),
        "row_id": row.get("id"),
        "baseline": next((r for r in result_rows if r["pattern"] == "baseline"), None),
        "num_patterns": len(selected),
        "pattern_lengths": {pattern: len(rows) for pattern, rows in selected.items()},
        "num_result_rows": len(result_rows),
        "stats_path": str(stats_path),
        "critical_tokens_json": str(critical_paths["json"]),
        "critical_tokens_csv": str(critical_paths["csv"]),
        "predictions_path": str(predictions_path),
        "results_path": str(results_path),
    }
    summary_path = _write_json(rep_tables_dir / "ablation_representation_summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary
