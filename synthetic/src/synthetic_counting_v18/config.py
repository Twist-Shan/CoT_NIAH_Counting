from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import torch


@dataclass(frozen=True)
class RunSpec:
    name: str
    distribution: str
    context_length: int
    train_max_count: int
    eval_max_count: int
    mode: str
    alpha: float | None = None

    def validate(self) -> None:
        if self.distribution not in {"uniform", "power"}:
            raise ValueError(f"unsupported count distribution: {self.distribution}")
        if self.mode not in {"direct", "cot"}:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.context_length < 1 or self.train_max_count < 1:
            raise ValueError("context length and maximum count must be positive")
        if self.eval_max_count < self.train_max_count:
            raise ValueError("eval_max_count must cover train_max_count")
        if self.eval_max_count > self.context_length:
            raise ValueError("count cannot exceed context length")
        if self.distribution == "power" and (self.alpha is None or self.alpha <= 0):
            raise ValueError("power runs require alpha > 0")


@dataclass(frozen=True)
class ReferenceConfig:
    preset: str = "debug"
    seed: int = 1234
    noise_vocab_size: int = 256
    needle_vocab_size: int = 10
    maximum_count: int = 128
    train_steps: int = 10_000
    batch_size: int = 32
    lr: float = 3e-4
    warmup_steps: int = 200
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "bf16"
    log_every: int = 50
    eval_every: int = 1_000
    checkpoint_every: int = 2_000
    dynamics_examples: int = 96
    final_examples_per_count: int = 96
    eval_batch_size: int = 32
    eval_token_budget: int = 32_768
    attention_examples_per_count: int = 1
    state_train_examples_per_count: int = 2
    state_eval_examples_per_count: int = 1
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_inner: int = 1024
    rope_base: float = 10_000.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def pad_id(self) -> int:
        # N0..N255, M0..M9, ANSWER, START, END, PAD.
        return 269

    @property
    def vocab_size(self) -> int:
        # Add I1..I128 and a disjoint C1..C128 family.
        return 270 + 2 * self.maximum_count

    def validate(self) -> None:
        if self.preset not in {"debug", "main"}:
            raise ValueError(self.preset)
        if (self.noise_vocab_size, self.needle_vocab_size, self.maximum_count) != (256, 10, 128):
            raise ValueError("v18 requires 256 noise tokens, 10 needle tokens, and count 1..128")
        if (self.n_layer, self.n_head, self.n_embd, self.n_inner) != (4, 4, 256, 1024):
            raise ValueError("v18 requires 4 layers, 4 heads, d_model=256, MLP=1024")
        if self.n_embd // self.n_head != 64:
            raise ValueError("v18 requires head_dim=64")
        if self.precision not in {"bf16", "float32"}:
            raise ValueError(self.precision)
        if self.eval_batch_size < 1 or self.eval_token_budget < 1:
            raise ValueError("evaluation batch and token budget must be positive")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "architecture": "pre-norm decoder-only Transformer; 4L/4H/d256/MLP1024; RoPE; tied embedding/unembedding",
                "optimizer": "AdamW beta=(0.9,0.95), weight_decay=0.01; 200-step warmup then cosine decay",
                "prompt_vocabulary": "256 noise tokens plus 10 disjoint marker/needle tokens",
                "direct_objective": "completion-only CE on C_n",
                "cot_objective": "completion-only CE on I_1,M_1,...,I_n,M_n,END,C_n",
                "token_separation": "ordinal index tokens I_k and final count tokens C_n are disjoint",
                "evaluation": "free-running greedy generation without teacher forcing",
                "analysis_count_bands": "1-32, 33-64, 65-96, 97-128",
                "attention_analysis": "direct broad aggregation; CoT k-to-k retrieval, successor preparation, and trace readout",
                "state_analysis": "held-out count/progress probes and PC1-PC6 centroid geometry at embedding and Layers 1-4",
            }
        )
        return result


def _pair(distribution: str, alpha: float | None = None) -> list[RunSpec]:
    suffix = f"_a{alpha:g}" if alpha is not None else ""
    return [
        RunSpec(
            f"{distribution}_L1024_train128_eval128{suffix}_{mode}",
            distribution,
            1024,
            128,
            128,
            mode,
            alpha,
        )
        for mode in ("direct", "cot")
    ]


def canonical_run_specs() -> tuple[RunSpec, ...]:
    """Focused four-model comparison requested for v18."""

    specs = _pair("uniform") + _pair("power", 1.5)
    for spec in specs:
        spec.validate()
    if len(specs) != 4 or len({spec.name for spec in specs}) != 4:
        raise AssertionError("canonical v18 suite must contain four unique runs")
    return tuple(specs)


def select_specs(suite: str, specs: Iterable[RunSpec] | None = None) -> tuple[RunSpec, ...]:
    values = tuple(specs or canonical_run_specs())
    if suite == "all":
        return values
    if suite not in {"power", "uniform"}:
        raise ValueError(f"unknown suite {suite!r}")
    return tuple(spec for spec in values if spec.distribution == suite)


def preset_config(preset: str, **overrides: object) -> ReferenceConfig:
    cfg = ReferenceConfig(preset=preset)
    if preset == "debug":
        cfg = replace(
            cfg,
            train_steps=4,
            batch_size=2,
            warmup_steps=1,
            log_every=1,
            eval_every=2,
            checkpoint_every=2,
            dynamics_examples=8,
            final_examples_per_count=2,
            eval_batch_size=4,
            eval_token_budget=4_096,
            attention_examples_per_count=1,
            state_train_examples_per_count=2,
            state_eval_examples_per_count=1,
            precision="float32",
        )
    elif preset != "main":
        raise ValueError(preset)
    unknown = sorted(set(overrides) - set(cfg.__dataclass_fields__))
    if unknown:
        raise TypeError(f"unknown config overrides: {unknown}")
    cfg = replace(cfg, **overrides)
    cfg.validate()
    return cfg


def debug_run_specs() -> tuple[RunSpec, ...]:
    specs = tuple(
        RunSpec(f"debug_L32_train4_eval6_{mode}", "uniform", 32, 4, 6, mode)
        for mode in ("direct", "cot")
    )
    for spec in specs:
        spec.validate()
    return specs
