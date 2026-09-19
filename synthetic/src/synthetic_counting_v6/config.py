from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


MODEL_TYPES = ("non_thinking", "thinking_sep_trace")
COUNT_BINS = {"low": {1, 2, 3}, "mid": {4, 5, 6}, "high": {7, 8, 9, 10}}


@dataclass
class ModelConfig:
    architecture: str = "gpt2_lm_head"
    position_embedding: str = "learned_absolute"
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_positions: int = 320
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    tie_word_embeddings: bool = True


@dataclass
class TrainConfig:
    train_steps: int = 20000
    batch_size: int = 128
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_steps: int = 500
    eval_every: int = 500
    log_every: int = 50
    save_every: int = 0


@dataclass
class EvalConfig:
    test_examples_per_count: int = 1000
    val_examples_per_count: int = 200
    probe_train_examples_per_count: int = 500
    probe_test_examples_per_count: int = 500
    attention_examples_per_count: int = 100


@dataclass
class V6Config:
    preset: str = "main"
    seq_len: int = 256
    noise_vocab_size: int = 64
    marker_vocab_size: int = 10
    min_count: int = 1
    max_count: int = 10
    seed: int = 1234
    device: str = "cpu"
    run_probes: bool = True
    run_attention: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_bin(count: int) -> str:
    for name, values in COUNT_BINS.items():
        if int(count) in values:
            return name
    raise ValueError(f"count outside configured bins: {count}")


def debug_config() -> V6Config:
    cfg = V6Config(preset="debug", seq_len=64)
    cfg.model = ModelConfig(n_layer=2, n_head=2, n_embd=128, n_positions=128)
    cfg.train = TrainConfig(
        train_steps=200,
        batch_size=32,
        learning_rate=4e-4,
        weight_decay=0.1,
        warmup_steps=20,
        eval_every=50,
        log_every=10,
        save_every=0,
    )
    cfg.eval = EvalConfig(
        test_examples_per_count=20,
        val_examples_per_count=20,
        probe_train_examples_per_count=20,
        probe_test_examples_per_count=20,
        attention_examples_per_count=10,
    )
    return cfg


def main_config() -> V6Config:
    return V6Config(preset="main")


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"unknown config key: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, preset: str | None = None) -> V6Config:
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = debug_config() if data.get("preset", preset) == "debug" else main_config()
        _merge_dataclass(cfg, data)
    else:
        cfg = debug_config() if preset == "debug" else main_config()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    min_positions = cfg.seq_len + 2 * cfg.max_count + 8
    if cfg.model.n_positions < min_positions:
        cfg.model.n_positions = min_positions
    return cfg


def save_config(cfg: V6Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")


def ensure_run_dirs(run_dir: Path) -> None:
    for subdir in ["data", "metrics", "checkpoints", "plots", "probes", "attention", "report"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

