from __future__ import annotations

import argparse
import json

from realistic_niah.runner import EngineConfig, run_vllm_experiment
from realistic_niah.spec import QUERY_LAYOUT


def _csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value.split(",") if item.strip())


def _csv_strings(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run registered Realistic NIAH V2 requests with offline vLLM."
    )
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--passage-lengths")
    parser.add_argument("--needle-counts")
    parser.add_argument("--seeds")
    parser.add_argument("--prompt-modes")
    parser.add_argument("--query-layout", default=QUERY_LAYOUT)
    parser.add_argument("--cache-dir")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--request-batch-size", type=int, default=32)
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()

    manifest = run_vllm_experiment(
        stimuli_path=args.stimuli,
        output_dir=args.output_dir,
        model=args.model,
        revision=args.revision,
        passage_lengths=_csv_ints(args.passage_lengths),
        needle_counts=_csv_ints(args.needle_counts),
        seeds=_csv_ints(args.seeds),
        prompt_modes=_csv_strings(args.prompt_modes),
        query_layout=args.query_layout,
        engine_config=EngineConfig(
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            request_batch_size=args.request_batch_size,
        ),
        cache_dir=args.cache_dir,
        repo_root=args.repo_root,
        require_clean_git=args.require_clean_git,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
