from __future__ import annotations

import argparse

from realistic_niah.runner import EngineConfig


def formal_engine_config(
    model_label: str,
    *,
    tensor_parallel_size: int | None = None,
) -> EngineConfig:
    """Return the single registered engine configuration for a V3.1 model."""

    requested_tp = 2 if model_label == "Gemma4-31B" else 1
    if tensor_parallel_size is not None and tensor_parallel_size != requested_tp:
        raise ValueError(
            f"Formal {model_label} requires tensor_parallel_size={requested_tp}, "
            f"not {tensor_parallel_size}"
        )
    if model_label == "Gemma4-31B":
        return EngineConfig(
            tensor_parallel_size=2,
            request_batch_size=1,
            max_num_seqs=1,
            gpu_memory_utilization=0.92,
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
    if model_label == "Qwen3-32B":
        return EngineConfig(
            request_batch_size=1,
            max_num_seqs=1,
            gpu_memory_utilization=0.92,
        )
    if model_label in {"Gemma4-26B-A4B", "Qwen3-14B"}:
        return EngineConfig(
            request_batch_size=2,
            max_num_seqs=2,
            gpu_memory_utilization=0.92,
        )
    if model_label in {
        "Gemma4-12B",
        "Nemotron-Nano-v2-9B",
        "GLM-4-9B-0414",
        "GLM-Z1-9B-0414",
    }:
        return EngineConfig(
            request_batch_size=4,
            max_num_seqs=4,
            gpu_memory_utilization=0.90,
        )
    if model_label in {
        "Qwen3-8B",
        "Gemma4-E4B",
        "Ministral-3-Instruct-8B",
        "Ministral-3-Reasoning-8B",
    }:
        return EngineConfig(
            request_batch_size=6,
            max_num_seqs=6,
            gpu_memory_utilization=0.90,
        )
    if model_label in {"Qwen3-4B", "Nemotron-3-Nano-4B"}:
        return EngineConfig(
            request_batch_size=8,
            max_num_seqs=8,
            gpu_memory_utilization=0.90,
        )
    raise KeyError(f"No formal V3.1 engine configuration for {model_label}")


def shell_settings(model_label: str, tensor_parallel_size: int) -> str:
    config = formal_engine_config(
        model_label,
        tensor_parallel_size=tensor_parallel_size,
    )
    return " ".join(
        (
            str(config.tensor_parallel_size),
            str(config.request_batch_size),
            str(config.max_num_seqs),
            str(config.gpu_memory_utilization),
            str(int(config.enforce_eager)),
            str(int(config.disable_custom_all_reduce)),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_label")
    parser.add_argument("tensor_parallel_size", type=int)
    args = parser.parse_args()
    print(shell_settings(args.model_label, args.tensor_parallel_size))


if __name__ == "__main__":
    main()
