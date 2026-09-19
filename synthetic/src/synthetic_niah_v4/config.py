from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


MODEL_TYPES = ("non_thinking", "thinking")
STAGES = (
    "train",
    "behavior_eval",
    "cache",
    "probe",
    "directions",
    "patching",
    "steering",
    "plots",
    "report",
    "all",
)


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_positions: int = 320
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0


@dataclass
class TrainConfig:
    steps: int = 10000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_every: int = 500
    checkpoint_every: int = 1000
    seed: int = 1234
    log_every: int = 50


@dataclass
class ProbeConfig:
    examples_per_count_train: int = 1000
    examples_per_count_test: int = 1000
    train_fraction: float = 0.5
    standardize_hidden: bool = True
    ridge_alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    logistic_l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


@dataclass
class SteeringConfig:
    alpha_grid: tuple[float, ...] = (-6, -4, -2, -1, 0, 1, 2, 4, 6)
    examples_per_count: int = 300
    max_direction_configs: int = 24


@dataclass
class V4Config:
    preset: str = "main"
    seq_len: int = 256
    count_min: int = 1
    count_max: int = 10
    out_root: str = "outputs/v4"
    run_name: str = ""
    device: str = "cpu"
    seeds: list[int] = field(default_factory=lambda: [1234])
    eval_examples_per_count: int = 1000
    cache_examples_per_count: int = 250
    direction_pairs_per_count: int = 100
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_root) / self.run_name if self.run_name else Path(self.out_root)


def _debug_config() -> V4Config:
    cfg = V4Config(preset="debug", seq_len=64)
    cfg.model = ModelConfig(n_layer=2, n_head=2, n_embd=128, n_positions=128)
    cfg.train = TrainConfig(
        steps=8,
        batch_size=8,
        lr=4e-4,
        weight_decay=0.01,
        warmup_steps=2,
        eval_every=0,
        checkpoint_every=0,
        seed=1234,
        log_every=4,
    )
    cfg.eval_examples_per_count = 2
    cfg.cache_examples_per_count = 3
    cfg.direction_pairs_per_count = 2
    cfg.probe = ProbeConfig(examples_per_count_train=2, examples_per_count_test=2, train_fraction=0.5)
    cfg.steering = SteeringConfig(alpha_grid=(-2, 0, 2), examples_per_count=1, max_direction_configs=6)
    return cfg


def build_config(args: Any | None = None, **overrides: Any) -> V4Config:
    preset = getattr(args, "preset", None) or overrides.pop("preset", "main")
    cfg = _debug_config() if preset == "debug" else V4Config(preset="main")
    if args is not None:
        for key in [
            "seq_len",
            "count_min",
            "count_max",
            "out_root",
            "run_name",
            "device",
            "eval_examples_per_count",
            "probe_examples_per_count",
            "steering_examples_per_count",
            "train_steps",
        ]:
            value = getattr(args, key, None)
            if value is not None:
                overrides[key] = value
        seeds = getattr(args, "seeds", None)
        if seeds:
            cfg.seeds = [int(part.strip()) for part in str(seeds).split(",") if part.strip()]

    if "seq_len" in overrides:
        cfg.seq_len = int(overrides.pop("seq_len"))
    if "count_min" in overrides:
        cfg.count_min = int(overrides.pop("count_min"))
    if "count_max" in overrides:
        cfg.count_max = int(overrides.pop("count_max"))
    if "out_root" in overrides:
        cfg.out_root = str(overrides.pop("out_root"))
    if "run_name" in overrides and overrides["run_name"]:
        cfg.run_name = str(overrides.pop("run_name"))
    if "device" in overrides and overrides["device"]:
        cfg.device = str(overrides.pop("device"))
    else:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if "train_steps" in overrides:
        cfg.train.steps = int(overrides.pop("train_steps"))
    if "eval_examples_per_count" in overrides:
        cfg.eval_examples_per_count = int(overrides.pop("eval_examples_per_count"))
    if "probe_examples_per_count" in overrides:
        value = int(overrides.pop("probe_examples_per_count"))
        cfg.probe.examples_per_count_train = value
        cfg.probe.examples_per_count_test = value
        cfg.cache_examples_per_count = max(cfg.cache_examples_per_count, value)
    if "steering_examples_per_count" in overrides:
        cfg.steering.examples_per_count = int(overrides.pop("steering_examples_per_count"))

    min_positions = cfg.seq_len + 2 * cfg.count_max + 8
    if cfg.model.n_positions < min_positions:
        cfg.model.n_positions = min_positions
    return cfg


def ensure_output_dirs(run_dir: Path) -> None:
    for subdir in ["checkpoints", "metrics", "tables", "figures", "cache", "artifacts"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
