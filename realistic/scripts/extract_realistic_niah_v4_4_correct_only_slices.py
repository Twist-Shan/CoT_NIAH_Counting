from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v4.correct_only_slices import build_correct_only_slices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate audited V4.4 overall results from clean-correct slices"
    )
    parser.add_argument("--qwen-run-root", type=Path, required=True)
    parser.add_argument("--gemma-run-root", type=Path, required=True)
    parser.add_argument("--ablation-confirmation-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()
    result = build_correct_only_slices(
        qwen_run_root=args.qwen_run_root,
        gemma_run_root=args.gemma_run_root,
        ablation_confirmation_run_root=args.ablation_confirmation_run_root,
        output_root=args.output_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

