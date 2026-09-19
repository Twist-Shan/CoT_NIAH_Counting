from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from dataset_generation.dynamic_niah_v2 import DynamicNiahV2Config

from .ablation_analysis import (
    _score_row,
    _write_csv,
    _write_json,
    _write_jsonl,
    build_irrelevant_token_pool,
    make_ablated_input_ids,
    needle_positions_from_segments,
    select_attention_sink_tokens,
    select_massive_activation_tokens,
    select_needle_sensitive_tokens,
)
from .ablation_representation_analysis import (
    _as_token_ablation_cfg,
    _rank_all_attention_sink_tokens,
    _rank_all_massive_activation_tokens,
    _rank_all_needle_sensitive_tokens,
    _select_needle_span_tokens,
    _torch_dtype,
    active_ablation_layers,
    resolve_decoder_layers,
    save_unablated_decoder_hidden_states,
)
from .single_example_analysis import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_NIAH_EXAMPLE_ROOT,
    SingleExamplePaths,
    _flat_input_ids,
    _read_jsonl,
    load_jsonl_example,
)

DEFAULT_REPRESENTATION_RESTORE_CONFIG_PATH = Path(
    "configs/ablation-representation-restore.json"
)


@dataclass(frozen=True)
class RepresentationRestoreConfig:
    """Configuration for representation-level hidden-state restoration."""

    num_critical_tokens: int = 10
    randomize_from_top_layer: bool = True
    ablation_random_seed: int = 12345
    critical_token_calc_layer: int = 24
    patterns: tuple[str, ...] = (
        "massive_activation",
        "attention_sink",
        "needle_sensitive",
        "needle_span",
        "needle_tail",
        "massive_activation_all",
        "attention_sink_all",
        "needle_sensitive_all",
    )
    attention_sink_score: str = "received_uniform_ratio"
    max_new_tokens: int | None = None
    temperature: float = 0.0
    haystack_dir: str = "data/haystacks/paul_graham"
    irrelevant_token_pool_size: int = 5000
    edge_exclusion_tokens: int = 5
    save_unablated_hidden_states: bool = True
    hidden_states_dtype: str = "bfloat16"

    def with_overrides(
        self,
        *,
        num_critical_tokens: int | None = None,
        randomize_from_top_layer: bool | None = None,
        ablation_random_seed: int | None = None,
        critical_token_calc_layer: int | None = None,
    ) -> "RepresentationRestoreConfig":
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
        return RepresentationRestoreConfig.patterns
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_representation_restore_config(
    path: str | Path = DEFAULT_REPRESENTATION_RESTORE_CONFIG_PATH,
    *,
    num_critical_tokens: int | None = None,
    randomize_from_top_layer: bool | None = None,
    ablation_random_seed: int | None = None,
    critical_token_calc_layer: int | None = None,
) -> RepresentationRestoreConfig:
    """Load restore-ablation defaults and apply runtime overrides."""

    config_path = Path(path)
    payload: dict[str, Any] = {}
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "patterns" in payload:
        payload["patterns"] = _tuple_patterns(payload["patterns"])
    cfg = RepresentationRestoreConfig(**payload)
    return cfg.with_overrides(
        num_critical_tokens=num_critical_tokens,
        randomize_from_top_layer=randomize_from_top_layer,
        ablation_random_seed=ablation_random_seed,
        critical_token_calc_layer=critical_token_calc_layer,
    )


def resolve_restore_dataset_path(
    run_name: str,
    *,
    base_dir: str | Path = DEFAULT_NIAH_EXAMPLE_ROOT,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
) -> Path:
    """Resolve the restore-mode dataset path and fail loudly if it is missing."""

    selected = str(run_name).strip()
    if not selected:
        raise ValueError("RESTORE_DATASET_RUN_NAME must be a non-empty run folder name")
    dataset_path = Path(base_dir) / selected / dataset_filename
    if not dataset_path.exists() or not dataset_path.is_file():
        raise FileNotFoundError(
            f"Could not find restore Dynamic NIAH v2 dataset: {dataset_path}. "
            "Expected data/niah-example/{RESTORE_DATASET_RUN_NAME}/dynamic_niah_v2.jsonl."
        )
    return dataset_path


def _as_representation_ablation_cfg(cfg: RepresentationRestoreConfig) -> Any:
    from .ablation_representation_analysis import RepresentationAblationConfig

    return RepresentationAblationConfig(
        num_critical_tokens=cfg.num_critical_tokens,
        randomize_from_top_layer=cfg.randomize_from_top_layer,
        ablation_random_seed=cfg.ablation_random_seed,
        critical_token_calc_layer=cfg.critical_token_calc_layer,
        patterns=cfg.patterns,
        attention_sink_score=cfg.attention_sink_score,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        edge_exclusion_tokens=cfg.edge_exclusion_tokens,
        save_unablated_hidden_states=cfg.save_unablated_hidden_states,
    )


def _select_needle_tail_tokens(
    *,
    needle_segments: Sequence[dict[str, Any]],
    input_ids: Sequence[int],
    k: int,
    tokenizer: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    by_pattern: dict[str, list[dict[str, Any]]] = {}
    for needle_idx, segment in enumerate(
        sorted(needle_segments, key=lambda item: int(item["start"]))
    ):
        pattern = f"needle_tail_{needle_idx}"
        positions = segment.get("positions")
        if isinstance(positions, list) and positions:
            tail_start = max(int(pos) for pos in positions) + 1
        else:
            tail_start = int(segment["end"])
        rows = []
        for rank, pos in enumerate(range(tail_start, tail_start + int(k)), start=1):
            if not 0 <= int(pos) < len(input_ids):
                continue
            from .ablation_analysis import _record_token

            rows.append(
                _record_token(
                    pattern=pattern,
                    rank=rank,
                    position=pos,
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                    score_name="needle_tail_position",
                    score=rank,
                    extra={"needle_id": segment.get("needle_id", needle_idx)},
                )
            )
        by_pattern[pattern] = rows
    return by_pattern


def select_representation_restore_critical_tokens(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: RepresentationRestoreConfig,
    input_ids: Sequence[int],
    needle_segments: Sequence[dict[str, Any]],
    tokenizer: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Select filtered/all-position, needle-span, and needle-tail restore patterns."""

    selected: dict[str, list[dict[str, Any]]] = {}
    patterns = set(cfg.patterns)
    rep_cfg = _as_representation_ablation_cfg(cfg)
    token_cfg = _as_token_ablation_cfg(rep_cfg)
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
            _select_needle_span_tokens(
                needle_segments=needle_segments,
                input_ids=input_ids,
                tokenizer=tokenizer,
            )
        )
    if "needle_tail" in patterns:
        selected.update(
            _select_needle_tail_tokens(
                needle_segments=needle_segments,
                input_ids=input_ids,
                k=cfg.num_critical_tokens,
                tokenizer=tokenizer,
            )
        )
    if "massive_activation_all" in patterns:
        selected["massive_activation_all"] = _rank_all_massive_activation_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=rep_cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
    if "attention_sink_all" in patterns:
        selected["attention_sink_all"] = _rank_all_attention_sink_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=rep_cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
    if "needle_sensitive_all" in patterns:
        selected["needle_sensitive_all"] = _rank_all_needle_sensitive_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=rep_cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
    return selected


def save_representation_restore_critical_tokens(
    *,
    run_dir: str | Path,
    selected: dict[str, list[dict[str, Any]]],
    cfg: RepresentationRestoreConfig,
) -> dict[str, Path]:
    ablation_dir = Path(run_dir) / "tables" / "ablation_representation_restore"
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
            {"config": asdict(cfg), "patterns": selected},
        ),
        "csv": _write_csv(ablation_dir / "critical_tokens.csv", flat_rows, fields),
    }


def _seed_for(*, base_seed: int, pattern: str, layer_idx: int, generation_step: int) -> int:
    payload = f"restore:{base_seed}:{pattern}:{layer_idx}:{generation_step}".encode(
        "utf-8"
    )
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31)


def replacement_tokens_for_setting(
    *,
    pool: Sequence[int],
    k: int,
    seed: int,
    pattern: str,
    layer_idx: int,
    sample_with_replacement: bool = False,
) -> list[int]:
    if k <= 0:
        return []
    if not pool:
        raise ValueError("Replacement pool is empty")
    rng = random.Random(f"{int(seed)}:{pattern}:{int(layer_idx)}")
    if sample_with_replacement or k > len(pool):
        return [int(rng.choice(pool)) for _ in range(k)]
    return [int(x) for x in rng.sample(list(pool), k)]


def make_corrupted_needle_input_ids(
    input_ids: torch.Tensor | Sequence[int],
    *,
    needle_segments: Sequence[dict[str, Any]],
    replacement_pool: Sequence[int],
    seed: int,
    pattern: str,
    layer_idx: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Replace all needle-token IDs with setting-specific irrelevant tokens."""

    base_ids = _flat_input_ids(input_ids)
    needle_positions = sorted(needle_positions_from_segments(needle_segments))
    replacement_ids = replacement_tokens_for_setting(
        pool=replacement_pool,
        k=len(needle_positions),
        seed=seed,
        pattern=pattern,
        layer_idx=layer_idx,
        sample_with_replacement=len(needle_positions) > len(replacement_pool),
    )
    corrupted = make_ablated_input_ids(base_ids, needle_positions, replacement_ids)
    records = []
    for pos, replacement_id in zip(needle_positions, replacement_ids, strict=True):
        records.append(
            {
                "position": int(pos),
                "original_token_id": int(base_ids[int(pos)]),
                "replacement_token_id": int(replacement_id),
            }
        )
    return corrupted, records


def _resolve_max_new_tokens(cfg: RepresentationRestoreConfig, dynamic_cfg: DynamicNiahV2Config | None) -> int:
    if cfg.max_new_tokens is not None:
        return int(cfg.max_new_tokens)
    if dynamic_cfg is not None and getattr(dynamic_cfg, "max_new_tokens", None) is not None:
        return int(dynamic_cfg.max_new_tokens)
    if dynamic_cfg is not None and getattr(dynamic_cfg, "thinking_mode", False):
        return 1024
    return 64


def _replace_layer_output_with_clean_hidden(
    output: Any,
    *,
    clean_hidden_states: torch.Tensor,
    layer_idx: int,
    positions: Sequence[int],
) -> Any:
    if isinstance(output, tuple):
        hidden = output[0].clone()
        rest = output[1:]
    else:
        hidden = output.clone()
        rest = None
    seq_len = int(hidden.shape[1])
    clean_seq_len = int(clean_hidden_states.shape[1])
    for pos in positions:
        pos = int(pos)
        if 0 <= pos < seq_len and pos < clean_seq_len:
            hidden[:, pos, :] = clean_hidden_states[int(layer_idx), pos, :].to(
                device=hidden.device, dtype=hidden.dtype
            )
    if rest is None:
        return hidden
    return (hidden, *rest)


def manual_generate_with_representation_restore(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    cfg: RepresentationRestoreConfig,
    clean_hidden_states: torch.Tensor | None = None,
    pattern: str = "baseline",
    positions: Sequence[int] = (),
    layer_idx: int | None = None,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> str:
    """Greedy decode from possibly corrupted tokens, optionally restoring hidden states."""

    decoder_layers = list(resolve_decoder_layers(model))
    input_device = model.get_input_embeddings().weight.device
    generated = input_ids.to(input_device).clone()
    max_new_tokens = _resolve_max_new_tokens(cfg, dynamic_cfg)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    active_layers = set()
    if clean_hidden_states is not None and layer_idx is not None and positions:
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
                        del _module, _inputs
                        return _replace_layer_output_with_clean_hidden(
                            output,
                            clean_hidden_states=clean_hidden_states,
                            layer_idx=current_layer_idx,
                            positions=positions,
                        )

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
            seed = _seed_for(
                base_seed=cfg.ablation_random_seed,
                pattern=pattern,
                layer_idx=-1 if layer_idx is None else int(layer_idx),
                generation_step=generation_step,
            )
            generator = torch.Generator(device=generated.device)
            generator.manual_seed(seed)
            probs = torch.softmax(logits / float(cfg.temperature), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and int(next_token[0, 0]) == int(eos_token_id):
            break
    prompt_len = int(input_ids.reshape(1, -1).shape[1])
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


def _token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.convert_ids_to_tokens(int(token_id)))
    except Exception:
        try:
            return str(tokenizer.decode([int(token_id)]))
        except Exception:
            return f"<id:{int(token_id)}>"


def _annotate_corruption_records(
    records: Sequence[dict[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    annotated = []
    for row in records:
        annotated.append(
            {
                **row,
                "original_token": _token_text(tokenizer, int(row["original_token_id"])),
                "replacement_token": _token_text(tokenizer, int(row["replacement_token_id"])),
            }
        )
    return annotated


def run_representation_restore_generation(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    example_id: int,
    input_ids: torch.Tensor | Sequence[int],
    needle_segments: Sequence[dict[str, Any]],
    selected: dict[str, list[dict[str, Any]]],
    replacement_pool: Sequence[int],
    cfg: RepresentationRestoreConfig,
    clean_hidden_states: torch.Tensor,
    out_dir: str | Path,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = Path(out_dir)
    generation_dir = out_dir / "generations"
    generation_dir.mkdir(parents=True, exist_ok=True)
    base_input = torch.as_tensor(input_ids, dtype=torch.long)
    if base_input.ndim == 1:
        base_input = base_input.unsqueeze(0)
    decoder_layers = list(resolve_decoder_layers(model))
    prediction_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []

    clean_output = manual_generate_with_representation_restore(
        model=model,
        tokenizer=tokenizer,
        input_ids=base_input,
        cfg=cfg,
        dynamic_cfg=dynamic_cfg,
    )
    clean_score = _score_row(row, clean_output)
    (generation_dir / "clean_baseline.txt").write_text(clean_output, encoding="utf-8")
    clean_result = {
        "example_id": int(example_id),
        "row_id": row.get("id"),
        "pattern": "clean_baseline",
        "layer_idx": -1,
        "restore_applied": False,
        "randomize_from_top_layer": bool(cfg.randomize_from_top_layer),
        "num_positions": 0,
        "restored_positions": json.dumps([]),
        "corrupted_needle_positions": json.dumps([]),
        "corrupted_replacement_token_ids": json.dumps([]),
        "model_output_text": clean_output,
        "parse_mode": clean_score.get("parse_mode"),
        "exact_match": bool(clean_score.get("exact_match")),
        "accuracy": float(clean_score["accuracy"]),
        "seed": int(cfg.ablation_random_seed),
    }
    result_rows.append(clean_result)
    prediction_rows.append({**clean_score, **clean_result})

    for pattern, tokens in selected.items():
        positions = [int(item["position"]) for item in tokens]
        if not positions:
            continue
        for layer_idx in range(len(decoder_layers)):
            corrupted_input, raw_records = make_corrupted_needle_input_ids(
                base_input,
                needle_segments=needle_segments,
                replacement_pool=replacement_pool,
                seed=cfg.ablation_random_seed,
                pattern=pattern,
                layer_idx=layer_idx,
            )
            annotated_records = _annotate_corruption_records(raw_records, tokenizer)
            replacement_ids = [int(item["replacement_token_id"]) for item in raw_records]
            corruption_rows.extend(
                {
                    **item,
                    "example_id": int(example_id),
                    "pattern": pattern,
                    "layer_idx": int(layer_idx),
                    "seed": int(cfg.ablation_random_seed),
                }
                for item in annotated_records
            )

            restore_output = manual_generate_with_representation_restore(
                model=model,
                tokenizer=tokenizer,
                input_ids=corrupted_input,
                cfg=cfg,
                clean_hidden_states=clean_hidden_states,
                pattern=pattern,
                positions=positions,
                layer_idx=layer_idx,
                dynamic_cfg=dynamic_cfg,
            )
            restore_score = _score_row(row, restore_output)
            restore_path = generation_dir / f"{pattern}_layer{layer_idx}_restore.txt"
            restore_path.write_text(restore_output, encoding="utf-8")
            restore_result = {
                "example_id": int(example_id),
                "row_id": row.get("id"),
                "pattern": pattern,
                "layer_idx": int(layer_idx),
                "restore_applied": True,
                "randomize_from_top_layer": bool(cfg.randomize_from_top_layer),
                "num_positions": len(positions),
                "restored_positions": json.dumps(positions),
                "corrupted_needle_positions": json.dumps([item["position"] for item in raw_records]),
                "corrupted_replacement_token_ids": json.dumps(replacement_ids),
                "model_output_text": restore_output,
                "parse_mode": restore_score.get("parse_mode"),
                "exact_match": bool(restore_score.get("exact_match")),
                "accuracy": float(restore_score["accuracy"]),
                "seed": int(cfg.ablation_random_seed),
            }
            result_rows.append(restore_result)
            prediction_rows.append({**restore_score, **restore_result})
    return prediction_rows, result_rows, corruption_rows


def run_single_example_representation_restore(
    *,
    paths: SingleExamplePaths,
    row: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    restore_dataset_run_name: str | None = None,
    example_id: int,
    model: Any,
    tokenizer: Any,
    uncontrolled_input_ids: torch.Tensor | Sequence[int] | None = None,
    needle_segments: Sequence[dict[str, Any]] | None = None,
    config_path: str | Path = DEFAULT_REPRESENTATION_RESTORE_CONFIG_PATH,
    num_critical_tokens: int | None = None,
    randomize_from_top_layer: bool | None = None,
    ablation_random_seed: int | None = None,
    critical_token_calc_layer: int | None = None,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> dict[str, Any]:
    cfg = load_representation_restore_config(
        config_path,
        num_critical_tokens=num_critical_tokens,
        randomize_from_top_layer=randomize_from_top_layer,
        ablation_random_seed=ablation_random_seed,
        critical_token_calc_layer=critical_token_calc_layer,
    )
    if restore_dataset_run_name is not None:
        dataset_path = resolve_restore_dataset_path(restore_dataset_run_name)
    elif dataset_path is None:
        raise ValueError("restore_dataset_run_name is required when dataset_path is not provided")
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Restore dataset JSONL does not exist: {dataset_path}")
    if row is None:
        row, _all_rows = load_jsonl_example(dataset_path, example_id)
    else:
        _all_rows = _read_jsonl(dataset_path)

    if uncontrolled_input_ids is None or needle_segments is None:
        from .ablation_analysis import load_input_metadata

        metadata = load_input_metadata(paths.run_dir, example_id)
        if uncontrolled_input_ids is None:
            uncontrolled_input_ids = metadata["uncontrolled_input_ids"]
        if needle_segments is None:
            needle_segments = metadata["needle_segments"]
    input_ids_list = _flat_input_ids(uncontrolled_input_ids)
    rep_tensors_dir = paths.tensors_dir / "ablation_representation_restore"
    rep_tables_dir = paths.tables_dir / "ablation_representation_restore"
    clean_hidden_path = rep_tensors_dir / f"hidden_states_clean_{int(example_id)}.pt"
    if cfg.save_unablated_hidden_states or not clean_hidden_path.exists():
        save_unablated_decoder_hidden_states(
            model=model,
            input_ids=torch.tensor([input_ids_list], dtype=torch.long),
            out_path=clean_hidden_path,
            dtype=_torch_dtype(cfg.hidden_states_dtype),
        )
    clean_payload = torch.load(clean_hidden_path, map_location="cpu")
    clean_hidden_states = clean_payload["hidden_states"]

    selected = select_representation_restore_critical_tokens(
        run_dir=paths.run_dir,
        example_id=example_id,
        cfg=cfg,
        input_ids=input_ids_list,
        needle_segments=list(needle_segments),
        tokenizer=tokenizer,
    )
    critical_paths = save_representation_restore_critical_tokens(
        run_dir=paths.run_dir, selected=selected, cfg=cfg
    )
    replacement_pool = build_irrelevant_token_pool(
        tokenizer=tokenizer,
        haystack_dir=cfg.haystack_dir,
        pool_size=cfg.irrelevant_token_pool_size,
        seed=cfg.ablation_random_seed,
    )
    _write_json(
        rep_tables_dir / "replacement_pool_summary.json",
        {
            "haystack_dir": cfg.haystack_dir,
            "irrelevant_token_pool_size": cfg.irrelevant_token_pool_size,
            "actual_pool_size": len(replacement_pool),
            "seed": cfg.ablation_random_seed,
        },
    )
    prediction_rows, result_rows, corruption_rows = run_representation_restore_generation(
        model=model,
        tokenizer=tokenizer,
        row=row,
        example_id=example_id,
        input_ids=torch.tensor([input_ids_list], dtype=torch.long),
        needle_segments=list(needle_segments),
        selected=selected,
        replacement_pool=replacement_pool,
        cfg=cfg,
        clean_hidden_states=clean_hidden_states,
        out_dir=rep_tables_dir,
        dynamic_cfg=dynamic_cfg,
    )
    predictions_path = _write_jsonl(
        rep_tables_dir / "ablation_representation_restore_predictions.jsonl", prediction_rows
    )
    result_fields = [
        "example_id",
        "row_id",
        "pattern",
        "layer_idx",
        "restore_applied",
        "randomize_from_top_layer",
        "num_positions",
        "restored_positions",
        "corrupted_needle_positions",
        "corrupted_replacement_token_ids",
        "model_output_text",
        "parse_mode",
        "exact_match",
        "accuracy",
        "seed",
    ]
    results_path = _write_csv(
        rep_tables_dir / "ablation_representation_restore_results.csv",
        result_rows,
        result_fields,
    )
    corruptions_path = _write_jsonl(rep_tables_dir / "corrupted_needle_tokens.jsonl", corruption_rows)
    summary = {
        "config": asdict(cfg),
        "example_id": int(example_id),
        "row_id": row.get("id") if isinstance(row, dict) else None,
        "dataset_path": str(dataset_path),
        "clean_baseline": next((r for r in result_rows if r["pattern"] == "clean_baseline"), None),
        "num_patterns": len(selected),
        "pattern_lengths": {pattern: len(rows) for pattern, rows in selected.items()},
        "num_result_rows": len(result_rows),
        "clean_hidden_states_path": str(clean_hidden_path),
        "critical_tokens_json": str(critical_paths["json"]),
        "critical_tokens_csv": str(critical_paths["csv"]),
        "predictions_path": str(predictions_path),
        "results_path": str(results_path),
        "corruptions_path": str(corruptions_path),
    }
    summary_path = _write_json(rep_tables_dir / "ablation_representation_restore_summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary
