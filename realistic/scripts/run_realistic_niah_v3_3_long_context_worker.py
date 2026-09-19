from __future__ import annotations

import argparse
import json
import os

from realistic_niah_v3_3_long_context.runner import run_worker
from realistic_niah_v3_3_long_context.spec import MODEL_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one of the two registered Gemma4-31B TP=2 workers."
    )
    parser.add_argument("--model", required=True, choices=MODEL_LABELS)
    parser.add_argument("--worker-index", type=int, required=True, choices=(0, 1))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=2700.0)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            "Each V3.3 long-context worker must see exactly two CUDA devices"
        )
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("H100" not in name for name in device_names):
        raise RuntimeError(f"Formal execution requires H100 GPUs: {device_names}")
    print(
        "V33_LONG_CONTEXT_TP2_PREFLIGHT "
        + json.dumps(
            {
                "worker_index": args.worker_index,
                "model_label": args.model,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device_count": torch.cuda.device_count(),
                "device_names": device_names,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    summary = run_worker(
        model_label=args.model,
        worker_index=args.worker_index,
        stimuli_path=args.stimuli,
        run_root=args.run_root,
        cache_dir=args.cache_dir,
        repo_root=args.repo_root,
        preflight_timeout_seconds=args.preflight_timeout_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
