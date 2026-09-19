from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


MODEL_DEFAULTS: dict[str, Any] = {
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 256,
    "n_inner": 1024,
    "n_positions": 384,
    "n_ctx": 384,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "attn_pdrop": 0.0,
    "use_cache": False,
}


DEBUG_MODEL_DEFAULTS: dict[str, Any] = {
    **MODEL_DEFAULTS,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 128,
    "n_inner": 512,
}


TRAIN_DEFAULTS: dict[str, Any] = {
    "seq_len": 256,
    "count_min": 1,
    "count_max": 10,
    "train_steps": 10000,
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "grad_clip": 1.0,
    "eval_every": 500,
    "checkpoint_every": 1000,
    "log_every": 50,
    "seed": 1234,
    "thinking_fraction": 0.5,
    "eval_examples_per_count": 1000,
    "probe_examples_per_count": 500,
    "attention_examples_per_count": 100,
}


PRESETS: dict[str, dict[str, Any]] = {
    "debug": {
        "model": deepcopy(DEBUG_MODEL_DEFAULTS),
        "train": {
            **TRAIN_DEFAULTS,
            "seq_len": 64,
            "train_steps": 4,
            "batch_size": 8,
            "warmup_steps": 5,
            "eval_every": 2,
            "checkpoint_every": 2,
            "log_every": 1,
            "eval_examples_per_count": 2,
            "probe_examples_per_count": 1,
            "attention_examples_per_count": 1,
        },
    },
    "main": {
        "model": deepcopy(MODEL_DEFAULTS),
        "train": deepcopy(TRAIN_DEFAULTS),
    },
}


def _maybe_set(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(PRESETS[args.preset])
    train = cfg["train"]
    model = cfg["model"]
    _maybe_set(train, "seq_len", args.seq_len)
    _maybe_set(train, "count_min", args.count_min)
    _maybe_set(train, "count_max", args.count_max)
    _maybe_set(train, "train_steps", args.train_steps)
    _maybe_set(train, "batch_size", args.batch_size)
    _maybe_set(train, "thinking_fraction", args.thinking_fraction)
    _maybe_set(train, "eval_examples_per_count", args.eval_examples_per_count)
    _maybe_set(train, "probe_examples_per_count", args.probe_examples_per_count)
    _maybe_set(train, "attention_examples_per_count", args.attention_examples_per_count)
    _maybe_set(train, "lr", args.lr)
    _maybe_set(train, "seed", args.seed)
    cfg["preset"] = args.preset
    cfg["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg["trace_indices"] = bool(args.trace_indices)
    cfg["format_version"] = "explicit_soft_switch_v2"
    cfg["model"]["vocab_size"] = 100 if cfg["trace_indices"] else 90
    cfg["model"]["bos_token_id"] = 0
    cfg["model"]["eos_token_id"] = 1
    cfg["model"]["pad_token_id"] = 1
    cfg["run_name"] = args.run_name or ""
    cfg["out_root"] = args.out_root
    trace_width = 2 if cfg["trace_indices"] else 1
    max_render_len = int(train["seq_len"]) + trace_width * int(train["count_max"]) + 6
    if max_render_len > int(model["n_positions"]):
        raise ValueError("Rendered sequence can exceed model.n_positions; lower seq_len/count_max or raise n_positions.")
    return cfg


def write_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    lines: list[str] = []
    for section, values in cfg.items():
        if isinstance(values, dict):
            lines.append(f"{section}:")
            for key, value in values.items():
                lines.append(f"  {key}: {json.dumps(value)}")
        else:
            lines.append(f"{section}: {json.dumps(values)}")
    (run_dir / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
