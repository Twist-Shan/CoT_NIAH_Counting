from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "realistic_niah_v4_4_2_config_v2"
PROTOCOL_VERSION = "realistic_niah_v4_4_2_trace_representation_v1"
PROMPT_VARIANTS = ("cue_present", "cue_absent")
MODES = ("nonthinking", "native_thinking")
MODELS = ("Qwen3-8B", "Gemma4-E4B")


@dataclass(frozen=True)
class V442Config:
    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    design_variant: str = "v4.4"
    model_labels: tuple[str, ...] = MODELS
    modes: tuple[str, ...] = MODES
    prompt_variants: tuple[str, ...] = PROMPT_VARIANTS
    counts: tuple[int, ...] = tuple(range(1, 11))
    seeds: tuple[int, ...] = tuple(range(1234, 1244))
    native_max_new_tokens: int = 4096
    nonthinking_max_new_tokens: int = 16
    qwen_native_temperature: float = 0.6
    qwen_native_top_p: float = 0.95
    qwen_native_top_k: int = 20
    gemma_native_temperature: float = 1.0
    gemma_native_top_p: float = 0.95
    gemma_native_top_k: int = 64
    base_sampling_seed: int = 442_000
    hidden_save_dtype: str = "float16"
    qk_save_dtype: str = "float16"
    attention_compute_dtype: str = "float32"
    capture_layers: tuple[int, ...] = ()
    attention_query_block_size: int = 64
    attention_key_block_size: int = 2048
    attention_trace_bins: int = 128
    attention_topk: int = 32
    legacy_baseline_mode: str = "rerun_all"

    @classmethod
    def from_json(cls, path: str | Path) -> "V442Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown V4.4.2 config keys: {unknown}")
        tuple_fields = {
            "model_labels",
            "modes",
            "prompt_variants",
            "counts",
            "seeds",
            "capture_layers",
        }
        values: dict[str, Any] = dict(payload)
        for key in tuple_fields:
            if key in values:
                values[key] = tuple(values[key])
        result = cls(**values)
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unexpected V4.4.2 config schema")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected V4.4.2 protocol")
        if self.design_variant != "v4.4":
            raise ValueError("V4.4.2 is restricted to the frozen v4.4 panel")
        if self.model_labels != MODELS:
            raise ValueError(f"V4.4.2 requires model order {MODELS}")
        if self.modes != MODES or self.prompt_variants != PROMPT_VARIANTS:
            raise ValueError("V4.4.2 requires the registered 2x2 mode/prompt grid")
        if self.counts != tuple(range(1, 11)):
            raise ValueError("V4.4.2 requires counts 1 through 10")
        if self.seeds != tuple(range(1234, 1244)):
            raise ValueError("V4.4.2 requires the fixed 10-seed panel 1234 through 1243")
        if self.native_max_new_tokens != 4096:
            raise ValueError("Native-thinking generation must match V3 at 4096 tokens")
        if self.nonthinking_max_new_tokens < 2:
            raise ValueError("Non-thinking generation must allow the two-token answer 10")
        if self.hidden_save_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported hidden_save_dtype")
        if self.qk_save_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported qk_save_dtype")
        if self.attention_compute_dtype not in {"float32", "float64"}:
            raise ValueError("Attention reconstruction requires float32 or float64")
        if self.attention_trace_bins <= 0:
            raise ValueError("attention_trace_bins must be positive")
        if self.legacy_baseline_mode != "rerun_all":
            raise ValueError("V4.4.2 requires all eight condition buckets to be rerun")

    def decoding(self, model_label: str, mode: str) -> dict[str, Any]:
        if model_label not in self.model_labels or mode not in self.modes:
            raise ValueError(f"Unregistered condition: {model_label}/{mode}")
        if mode == "nonthinking":
            return {
                "do_sample": False,
                "max_new_tokens": self.nonthinking_max_new_tokens,
            }
        if model_label == "Qwen3-8B":
            return {
                "do_sample": True,
                "max_new_tokens": self.native_max_new_tokens,
                "temperature": self.qwen_native_temperature,
                "top_p": self.qwen_native_top_p,
                "top_k": self.qwen_native_top_k,
            }
        return {
            "do_sample": True,
            "max_new_tokens": self.native_max_new_tokens,
            "temperature": self.gemma_native_temperature,
            "top_p": self.gemma_native_top_p,
            "top_k": self.gemma_native_top_k,
        }

    def sampling_seed(
        self,
        *,
        model_label: str,
        prompt_variant: str,
        seed: int,
        count: int,
    ) -> int:
        model_index = self.model_labels.index(model_label)
        prompt_index = self.prompt_variants.index(prompt_variant)
        return int(
            self.base_sampling_seed
            + model_index * 1_000_000
            + prompt_index * 100_000
            + int(seed) * 100
            + int(count)
        )
