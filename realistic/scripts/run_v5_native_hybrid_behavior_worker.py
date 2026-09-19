#!/usr/bin/env python3
"""Run one deterministic modulo-partition of the hybrid behavior ledger."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _complete(output: Path) -> bool:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scheduled = int(manifest.get("scheduled_anchor_condition_trials", -1))
    completed = int(manifest.get("completed_shards", -2))
    return scheduled > 0 and completed == scheduled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--qwen-generations", type=Path, required=True)
    parser.add_argument("--gemma-generations", type=Path, required=True)
    parser.add_argument("--qwen-routing", type=Path, required=True)
    parser.add_argument("--gemma-routing", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")

    jobs = [
        json.loads(line)
        for line in args.jobs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        job for index, job in enumerate(jobs) if index % args.worker_count == args.worker_index
    ]
    paths = {
        "Qwen3-8B": (args.qwen_generations, args.qwen_routing),
        "Gemma4-E4B": (args.gemma_generations, args.gemma_routing),
    }
    env = dict(os.environ)
    env.update({"HF_HOME": str(args.cache_dir), "TOKENIZERS_PARALLELISM": "false"})
    for ordinal, job in enumerate(selected, start=1):
        if job.get("execution_status") == "skipped_empty_split":
            print(
                f"WORKER_SKIP_EMPTY worker={args.worker_index} "
                f"job={job['job_index']} model={job['model_label']} "
                f"grammar={job['grammar']} K={job['bank_size']}",
                flush=True,
            )
            continue
        output = Path(job["output"])
        if _complete(output):
            print(
                f"WORKER_SKIP worker={args.worker_index} ordinal={ordinal}/{len(selected)} "
                f"job={job['job_index']} output={output}",
                flush=True,
            )
            continue
        model = str(job["model_label"])
        generations, routing = paths[model]
        command = [
            args.python,
            "scripts/run_realistic_niah_v5.py",
            "causal-heads-behavior",
            "--config",
            str(args.config),
            "--model",
            model,
            "--cache-dir",
            str(args.cache_dir),
            "--device-map",
            "auto",
            "--torch-dtype",
            "bfloat16",
            "--attention-backend",
            "sdpa",
            "--generations",
            str(generations),
            "--plan",
            str(job["plan"]),
            "--output",
            str(output),
            "--anchor-routing",
            str(routing),
            "--behavior-target-grammar-class",
            str(job["grammar"]),
            "--evaluation-split",
            str(job["evaluation_split"]),
            "--conditions",
            "selected_bank",
            str(job["random_condition"]),
            "--include-secondary",
            "--limit",
            "300",
            "--anchor-sampling",
            "prompt_balanced",
            "--anchor-registry-input",
            str(job["registry"]),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--decode-head-ablation-steps",
            str(job["decode_head_ablation_steps"]),
        ]
        if bool(job["selection_intervention_site_decoupled"]):
            command.append("--allow-selection-intervention-site-decoupling")
        print(
            f"WORKER_START worker={args.worker_index} ordinal={ordinal}/{len(selected)} "
            f"job={job['job_index']} model={model} grammar={job['grammar']} "
            f"K={job['bank_size']} selection={job['selection_anchor_role']} "
            f"intervention={job['intervention_start_anchor_role']}",
            flush=True,
        )
        subprocess.run(command, cwd=args.code_root, env=env, check=True)
        print(
            f"WORKER_COMPLETE worker={args.worker_index} job={job['job_index']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
