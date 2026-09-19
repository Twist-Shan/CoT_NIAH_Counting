#!/usr/bin/env python3
"""Run one frozen seed through the greedy restoration readout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.bullet_greedy_restore import (  # noqa: E402
    run_bullet_greedy_restore_trials,
)
from scripts.run_realistic_niah_v5_bullet_counterfactual_restore import (  # noqa: E402
    _selected_rows,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_jsonl,
    _model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-layer", type=int, required=True)
    parser.add_argument("--target-occurrences", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.command = "bullet-greedy-restore-smoke"
    rows, _registered, _cohort = _selected_rows(
        generations=args.generations,
        cohort_manifest=args.cohort_manifest,
        model_label=str(args.model),
        phase=str(args.phase),
    )
    selected = [row for row in rows if int(row["seed"]) == int(args.seed)]
    if len(selected) != 1:
        raise ValueError("Smoke seed is not unique in the frozen phase")
    model, tokenizer, adapter = _model(args)
    results = run_bullet_greedy_restore_trials(
        model,
        tokenizer,
        adapter,
        selected[0],
        source_layers=(int(args.source_layer),),
        target_occurrences=tuple(int(value) for value in args.target_occurrences),
        random_seed=20260824 + int(args.seed),
        max_new_tokens=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output, results)
    print(
        json.dumps(
            [
                {
                    "k": int(row["target_occurrence"]),
                    "condition": str(row["condition"]),
                    "prediction": row["greedy_prediction"],
                    "exact": bool(row["greedy_running_exact"]),
                    "completion": str(row["completion_text_raw"]),
                }
                for row in results
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
