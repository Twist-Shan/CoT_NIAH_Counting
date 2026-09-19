from __future__ import annotations

import argparse
import json

from realistic_niah.runner import EngineConfig
from realistic_niah.spec import QUERY_LAYOUT
from realistic_niah_v3_1.runner import run_v31_model_bundle


def _csv_ints(value: str | None) -> tuple[int, ...] | None:
    return (
        None if value is None else tuple(int(x) for x in value.split(",") if x.strip())
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all registered V3.1 modes for one model with one load."
    )
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--passage-lengths")
    parser.add_argument("--needle-counts")
    parser.add_argument("--seeds")
    parser.add_argument("--query-layout", default=QUERY_LAYOUT)
    parser.add_argument("--cache-dir")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--request-batch-size", type=int, default=32)
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()
    summary = run_v31_model_bundle(
        stimuli_path=args.stimuli,
        run_root=args.run_root,
        model=args.model,
        revision=args.revision,
        passage_lengths=_csv_ints(args.passage_lengths),
        needle_counts=_csv_ints(args.needle_counts),
        seeds=_csv_ints(args.seeds),
        query_layout=args.query_layout,
        engine_config=EngineConfig(
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            request_batch_size=args.request_batch_size,
        ),
        cache_dir=args.cache_dir,
        repo_root=args.repo_root,
        require_clean_git=args.require_clean_git,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
