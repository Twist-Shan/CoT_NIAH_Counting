from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class V10Config:
    preset: str = "debug"
    seed: int = 1234
    seq_len: int = 256
    count_min: int = 1
    count_max: int = 30
    noise_vocab_size: int = 64
    marker_vocab_size: int = 10

    train_steps: int = 10_000
    batch_size: int = 128
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    ar_eval_every: int = 1_000
    checkpoint_every: int = 1_000
    eval_examples_per_count: int = 100
    ar_examples_per_count: int = 20
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4

    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_inner: int = 1024
    n_positions: int = 384

    attention_examples_per_count: int = 20
    attention_causal_examples_per_count: int = 8
    state_train_examples_per_count: int = 40
    state_eval_examples_per_count: int = 15
    state_causal_examples_per_count: int = 6
    fixed_trace_count: int = 15
    analysis_batch_size: int = 64
    patch_offsets: tuple[int, ...] = (-10, -5, -3, -2, -1, 1, 2, 3, 5, 10)
    steering_alphas: tuple[float, ...] = (-5, -3, -2, -1, 0, 1, 2, 3, 5)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def effective_batch_size(self) -> int:
        return int(self.batch_size) * int(self.grad_accum_steps)

    @property
    def max_render_len(self) -> int:
        # thinking: BOS + prompt + Think + 2*count + close + Ans + count + EOS
        return int(self.seq_len) + 2 * int(self.count_max) + 6

    @property
    def modes(self) -> tuple[str, str]:
        return ("nonthinking", "thinking")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["patch_offsets"] = list(self.patch_offsets)
        value["steering_alphas"] = list(self.steering_alphas)
        value["architecture"] = "random-init GPT-2; learned absolute position embeddings"
        value["separate_transformers"] = True
        value["shared_trace_and_answer_numbers"] = True
        return value

    def validate(self) -> None:
        if self.count_min < 1 or self.count_max < self.count_min:
            raise ValueError("count range must satisfy 1 <= count_min <= count_max")
        if self.count_max > self.seq_len:
            raise ValueError("count_max cannot exceed seq_len")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.max_render_len > self.n_positions:
            raise ValueError(
                f"max rendered length {self.max_render_len} exceeds n_positions={self.n_positions}"
            )


def preset_config(preset: str, **overrides: Any) -> V10Config:
    if preset == "debug":
        cfg = V10Config(
            preset="debug",
            seq_len=48,
            count_max=6,
            train_steps=8,
            batch_size=8,
            warmup_steps=2,
            log_every=1,
            eval_every=4,
            ar_eval_every=4,
            checkpoint_every=4,
            eval_examples_per_count=2,
            ar_examples_per_count=1,
            n_layer=2,
            n_head=2,
            n_embd=64,
            n_inner=128,
            n_positions=96,
            attention_examples_per_count=1,
            attention_causal_examples_per_count=1,
            state_train_examples_per_count=2,
            state_eval_examples_per_count=1,
            state_causal_examples_per_count=1,
            fixed_trace_count=3,
            analysis_batch_size=8,
            patch_offsets=(-2, -1, 1, 2),
            steering_alphas=(-1, 0, 1),
        )
    elif preset == "main":
        cfg = V10Config(preset="main")
    else:
        raise ValueError(f"Unknown preset: {preset}")
    known = set(cfg.__dataclass_fields__)
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise TypeError(f"Unknown V10Config overrides: {unknown}")
    cfg = replace(cfg, **overrides)
    cfg.validate()
    return cfg


def config_from_dict(values: dict[str, Any]) -> V10Config:
    data = dict(values)
    data.pop("architecture", None)
    data.pop("separate_transformers", None)
    data.pop("shared_trace_and_answer_numbers", None)
    if "patch_offsets" in data:
        data["patch_offsets"] = tuple(int(value) for value in data["patch_offsets"])
    if "steering_alphas" in data:
        data["steering_alphas"] = tuple(float(value) for value in data["steering_alphas"])
    cfg = V10Config(**data)
    cfg.validate()
    return cfg


def run_name(cfg: V10Config) -> str:
    return (
        f"v10_{cfg.preset}_L{cfg.seq_len}_count{cfg.count_min}-{cfg.count_max}_"
        f"{cfg.n_layer}L{cfg.n_head}H_d{cfg.n_embd}_sharednum_seed{cfg.seed}"
    )


def prepare_run_dir(out_root: str | Path, cfg: V10Config, fixed_name: str | None = None) -> Path:
    root = Path(out_root)
    path = root / (fixed_name or run_name(cfg))
    for subdir in ("tables", "figures", "checkpoints", "analysis", "logs"):
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path
