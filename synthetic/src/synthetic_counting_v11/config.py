from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch


SUPPORTED_VERSIONS = ("v11", "v12", "v13", "v14", "v15", "v16", "v16_1", "v17")
SUPPORTED_POSITION_ENCODINGS = ("ape", "rope", "rpe")
TARGET_SHAKESPEARE_CHARACTERS = ("S", "H", "A", "K", "E", "R", "s", "h", "a", "k", "e", "r")


@dataclass(frozen=True)
class ExperimentConfig:
    version: str = "v11"
    preset: str = "debug"
    seed: int = 1234
    seq_len: int = 256
    count_min: int = 1
    count_max: int = 30
    noise_vocab_size: int = 64
    marker_vocab_size: int = 10
    noise_source: str = "uniform"
    task_type: str = "inserted_marker"
    target_characters: tuple[str, ...] = TARGET_SHAKESPEARE_CHARACTERS
    training_data_mode: str = "streaming"
    loss_scope: str = "completion"
    count_sampling: str = "uniform"
    power_alpha: float = 1.0
    exponential_beta: float = 0.15
    fixed_train_examples_per_count: int = 512
    corpus_train_fraction: float = 0.80
    corpus_validation_fraction: float = 0.10
    min_candidate_windows: int = 128
    window_sampling: str = "with_replacement"
    position_encodings: tuple[str, ...] = ("ape", "rope", "rpe")

    train_steps: int = 10_000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    warmup_steps: int = 500
    grad_clip: float = 1.0
    precision: str = "float32"
    log_every: int = 50
    eval_every: int = 500
    ar_eval_every: int = 1_000
    checkpoint_every: int = 1_000
    eval_examples_per_count: int = 100
    ar_examples_per_count: int = 10

    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_inner: int = 256
    n_positions: int = 384
    max_relative_distance: int = 256
    rope_base: float = 10_000.0

    attention_examples_per_count: int = 20
    state_train_examples_per_count: int = 40
    state_eval_examples_per_count: int = 15
    analysis_batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def modes(self) -> tuple[str, str]:
        return ("nonthinking", "thinking")

    @property
    def max_render_len(self) -> int:
        task_prefix = 3 if self.task_type == "target_character" else 0
        return int(self.seq_len) + 2 * int(self.count_max) + 6 + task_prefix

    @property
    def model_variants(self) -> tuple[tuple[str, str], ...]:
        return tuple((position, mode) for position in self.position_encodings for mode in self.modes)

    @property
    def count_bins(self) -> tuple[tuple[int, int], ...]:
        bins: list[tuple[int, int]] = []
        start = int(self.count_min)
        while start <= int(self.count_max):
            end = min(start + 9, int(self.count_max))
            bins.append((start, end))
            start = end + 1
        return tuple(bins)

    def count_bin(self, count: int) -> str:
        value = int(count)
        for lo, hi in self.count_bins:
            if lo <= value <= hi:
                return f"{lo}-{hi}"
        raise ValueError(f"count {value} is outside {self.count_min}..{self.count_max}")

    def validate(self) -> None:
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version: {self.version}")
        if self.count_min < 1 or self.count_max < self.count_min:
            raise ValueError("count range must satisfy 1 <= count_min <= count_max")
        if self.count_max > self.seq_len:
            raise ValueError("count_max cannot exceed seq_len")
        expected_width = 64 if self.version in {"v11", "v12", "v13", "v14"} else 256
        expected_inner = 256 if expected_width == 64 else 1024
        if (
            self.n_layer != 4
            or self.n_head != 4
            or self.n_embd != expected_width
            or self.n_inner != expected_inner
        ):
            raise ValueError(
                f"{self.version} requires exactly 4 layers, 4 heads, "
                f"hidden size {expected_width}, and MLP size {expected_inner}"
            )
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.max_render_len > self.n_positions:
            raise ValueError(
                f"max rendered length {self.max_render_len} exceeds n_positions={self.n_positions}"
            )
        invalid = sorted(set(self.position_encodings) - set(SUPPORTED_POSITION_ENCODINGS))
        if invalid:
            raise ValueError(f"Unsupported position encodings: {invalid}")
        if self.noise_source not in {"uniform", "shakespeare_char"}:
            raise ValueError(f"Unsupported noise source: {self.noise_source}")
        if self.task_type not in {"inserted_marker", "target_character"}:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        if self.loss_scope not in {"completion", "all_sequence"}:
            raise ValueError(f"Unsupported loss scope: {self.loss_scope}")
        if self.count_sampling not in {"uniform", "power", "exponential"}:
            raise ValueError(f"Unsupported count sampling: {self.count_sampling}")
        if not (0.0 <= self.adam_beta1 < 1.0 and 0.0 <= self.adam_beta2 < 1.0):
            raise ValueError("Adam betas must be in [0, 1)")
        if self.precision not in {"float32", "bf16"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if not self.target_characters or any(len(char) != 1 for char in self.target_characters):
            raise ValueError("target_characters must contain one-character strings")
        if self.training_data_mode not in {"streaming", "fixed", "split_window_index"}:
            raise ValueError(f"Unsupported training data mode: {self.training_data_mode}")
        if not 0.0 < self.corpus_train_fraction < 1.0:
            raise ValueError("corpus_train_fraction must be in (0, 1)")
        if not 0.0 < self.corpus_validation_fraction < 1.0:
            raise ValueError("corpus_validation_fraction must be in (0, 1)")
        if self.corpus_train_fraction + self.corpus_validation_fraction >= 1.0:
            raise ValueError("train + validation corpus fractions must leave a non-empty test split")
        if self.min_candidate_windows < 1:
            raise ValueError("min_candidate_windows must be positive")
        if self.window_sampling not in {"with_replacement", "without_replacement"}:
            raise ValueError(f"Unsupported window_sampling: {self.window_sampling}")
        if self.version == "v11" and set(self.position_encodings) != set(SUPPORTED_POSITION_ENCODINGS):
            raise ValueError("v11 must compare APE, RoPE, and RPE")
        if self.version in {"v12", "v13", "v14"} and self.position_encodings != ("ape",):
            raise ValueError(f"{self.version} must use APE only")
        if self.version in {"v15", "v16", "v16_1"} and self.position_encodings != ("rope", "rpe"):
            raise ValueError(f"{self.version} must compare RoPE and RPE")
        if self.version == "v17" and self.position_encodings != ("rope",):
            raise ValueError("v17 must use RoPE only")
        if self.version in {"v15", "v16", "v16_1"} and self.loss_scope != "all_sequence":
            raise ValueError(f"{self.version} must use all-sequence autoregressive loss")
        if self.version == "v17" and self.loss_scope != "completion":
            raise ValueError("v17 must use the v10 completion-only autoregressive loss")
        if self.version == "v15" and (
            self.noise_source != "shakespeare_char" or self.task_type != "inserted_marker"
        ):
            raise ValueError("v15 uses Shakespeare haystacks with inserted marker needles")
        if self.version == "v16" and (
            self.noise_source != "shakespeare_char" or self.task_type != "target_character"
        ):
            raise ValueError("v16 counts native target-character occurrences in Shakespeare")
        if self.version == "v16_1":
            if self.noise_source != "shakespeare_char" or self.task_type != "target_character":
                raise ValueError("v16.1 counts native target-character occurrences in Shakespeare")
            if self.training_data_mode != "split_window_index":
                raise ValueError("v16.1 requires split-before-index window data")
            if self.window_sampling != "without_replacement":
                raise ValueError("v16.1 requires without-replacement window sampling")
        if self.version == "v17" and self.count_sampling not in {"power", "exponential"}:
            raise ValueError("v17 requires a decreasing power or exponential count sampler")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["position_encodings"] = list(self.position_encodings)
        result["target_characters"] = list(self.target_characters)
        result["count_bins"] = [list(pair) for pair in self.count_bins]
        result["architecture"] = (
            "v10-style random-init GPT-2/pre-LN causal Transformer core; "
            f"4 layers; 4 heads; d_model={self.n_embd}; MLP={self.n_inner}; "
            f"position_encoding={','.join(self.position_encodings)}; "
            "tied token embedding/unembedding"
        )
        result["architecture_note"] = (
            "v11-v14 are controlled hidden size 64 (d_model=64) experiments; v15-v17, including "
            "v16.1, restore the "
            "v10 d_model=256 and MLP=1024 capacity while keeping 4 layers and 4 heads"
        )
        result["rpe_definition"] = "learned per-layer, per-head causal relative-distance attention bias"
        result["rope_definition"] = (
            f"standard rotary position embedding applied to query/key with base={self.rope_base:g}"
        )
        result["optimizer"] = (
            f"AdamW(beta1={self.adam_beta1:g}, beta2={self.adam_beta2:g}, "
            f"weight_decay={self.weight_decay:g})"
        )
        result["precision_definition"] = (
            "CUDA bfloat16 autocast for forward/loss; float32 otherwise"
            if self.precision == "bf16"
            else "float32"
        )
        result["separate_transformers"] = True
        result["shared_trace_and_answer_numbers"] = True
        result["training_objective"] = (
            "teacher-forced next-token cross-entropy over every non-padding token"
            if self.loss_scope == "all_sequence"
            else "teacher-forced completion-only next-token cross-entropy"
        )
        result["count_sampling_definition"] = {
            "uniform": "p(n) proportional to 1",
            "power": f"p(n) proportional to n^(-{self.power_alpha})",
            "exponential": f"p(n) proportional to exp(-{self.exponential_beta} * (n-count_min))",
        }[self.count_sampling]
        if self.noise_source == "shakespeare_char":
            result["tiny_shakespeare_source"] = (
                "Karpathy char-rnn data/tinyshakespeare/input.txt; "
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            )
        return result


def same_experiment_except_version(
    existing: dict[str, object],
    requested: dict[str, object],
) -> bool:
    experimental_fields = set(ExperimentConfig.__dataclass_fields__) - {"version"}
    return all(existing.get(field) == requested.get(field) for field in experimental_fields)


def _main_config(version: str) -> ExperimentConfig:
    if version == "v11":
        return ExperimentConfig(version="v11", preset="main")
    if version == "v12":
        return ExperimentConfig(
            version="v12",
            preset="main",
            seq_len=512,
            count_max=50,
            n_positions=640,
            max_relative_distance=512,
            position_encodings=("ape",),
        )
    if version == "v13":
        return ExperimentConfig(
            version="v13",
            preset="main",
            training_data_mode="fixed",
            position_encodings=("ape",),
        )
    if version == "v14":
        return ExperimentConfig(
            version="v14",
            preset="main",
            noise_source="shakespeare_char",
            position_encodings=("ape",),
        )
    if version == "v15":
        return ExperimentConfig(
            version="v15",
            preset="main",
            noise_source="shakespeare_char",
            loss_scope="all_sequence",
            position_encodings=("rope", "rpe"),
            n_embd=256,
            n_inner=1024,
        )
    if version == "v16":
        return ExperimentConfig(
            version="v16",
            preset="main",
            noise_source="shakespeare_char",
            task_type="target_character",
            loss_scope="all_sequence",
            position_encodings=("rope", "rpe"),
            n_embd=256,
            n_inner=1024,
        )
    if version == "v16_1":
        return ExperimentConfig(
            version="v16_1",
            preset="main",
            noise_source="shakespeare_char",
            task_type="target_character",
            training_data_mode="split_window_index",
            window_sampling="without_replacement",
            min_candidate_windows=128,
            loss_scope="all_sequence",
            position_encodings=("rope", "rpe"),
            n_embd=256,
            n_inner=1024,
        )
    if version == "v17":
        return ExperimentConfig(
            version="v17",
            preset="main",
            loss_scope="completion",
            count_sampling="power",
            position_encodings=("rope",),
            batch_size=32,
            warmup_steps=200,
            adam_beta1=0.9,
            adam_beta2=0.95,
            precision="bf16",
            rope_base=10_000.0,
            n_embd=256,
            n_inner=1024,
        )
    raise ValueError(f"Unknown experiment version: {version}")


def preset_config(version: str, preset: str, **overrides: Any) -> ExperimentConfig:
    cfg = _main_config(version)
    if preset == "debug":
        debug_count_max = 6 if version == "v12" else 4
        cfg = replace(
            cfg,
            preset="debug",
            seq_len=80 if version == "v12" else 48,
            count_max=debug_count_max,
            n_positions=128 if version == "v12" else 96,
            max_relative_distance=96,
            train_steps=6,
            batch_size=4,
            warmup_steps=2,
            log_every=1,
            eval_every=3,
            ar_eval_every=3,
            checkpoint_every=3,
            eval_examples_per_count=2,
            ar_examples_per_count=1,
            attention_examples_per_count=1,
            state_train_examples_per_count=2,
            state_eval_examples_per_count=1,
            fixed_train_examples_per_count=4,
            analysis_batch_size=8,
            min_candidate_windows=2 if version == "v16_1" else cfg.min_candidate_windows,
        )
    elif preset != "main":
        raise ValueError(f"Unknown preset: {preset}")
    known = set(cfg.__dataclass_fields__)
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise TypeError(f"Unknown ExperimentConfig overrides: {unknown}")
    if "position_encodings" in overrides:
        overrides["position_encodings"] = tuple(overrides["position_encodings"])
    if "target_characters" in overrides:
        overrides["target_characters"] = tuple(overrides["target_characters"])
    cfg = replace(cfg, **overrides)
    cfg.validate()
    return cfg


def config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    data = dict(values)
    for derived in (
        "architecture",
        "architecture_note",
        "rpe_definition",
        "rope_definition",
        "optimizer",
        "precision_definition",
        "separate_transformers",
        "shared_trace_and_answer_numbers",
        "count_bins",
        "tiny_shakespeare_source",
        "training_objective",
        "count_sampling_definition",
    ):
        data.pop(derived, None)
    data["position_encodings"] = tuple(data["position_encodings"])
    data["target_characters"] = tuple(data.get("target_characters", TARGET_SHAKESPEARE_CHARACTERS))
    cfg = ExperimentConfig(**data)
    cfg.validate()
    return cfg


def default_run_name(cfg: ExperimentConfig) -> str:
    positions = "-".join(cfg.position_encodings)
    data_tag = (
        cfg.training_data_mode
        if cfg.training_data_mode in {"fixed", "split_window_index"}
        else cfg.noise_source
    )
    return (
        f"{cfg.version}_{cfg.preset}_L{cfg.seq_len}_count{cfg.count_min}-{cfg.count_max}_"
        f"4L4H_d{cfg.n_embd}_{positions}_{data_tag}_{cfg.loss_scope}_{cfg.count_sampling}_seed{cfg.seed}"
    )


def prepare_run_dir(out_root: str | Path, cfg: ExperimentConfig, run_name: str | None = None) -> Path:
    path = Path(out_root) / (run_name or default_run_name(cfg))
    for subdir in ("tables", "figures", "checkpoints", "analysis", "logs", "data"):
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path
