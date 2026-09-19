from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_v445_transport_supplement import aligned_decode, load_dataset, rdm_similarity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(4451)
    rows = []
    for model, source, target in (
        ("Qwen3-8B", 28, 29),
        ("Gemma4-E4B", 36, 37),
    ):
        left = load_dataset(
            args.packed_root / "layers" / f"{model}__prompt_running__L{source:02d}.npz"
        )
        right = load_dataset(
            args.packed_root / "layers" / f"{model}__answer_query__L{target:02d}.npz"
        )
        rdm, rdm_low, rdm_high, seeds = rdm_similarity(
            left, right, bootstraps=2_000, rng=rng
        )
        decoded = aligned_decode(
            left, right, rank=3, folds=5, rng=rng, shuffle_repeats=500
        )
        rows.append({
            "model_label": model,
            "source": f"prompt_running_L{source}",
            "target": f"answer_query_L{target}",
            "rdm_spearman": rdm,
            "rdm_ci95_low": rdm_low,
            "rdm_ci95_high": rdm_high,
            "rdm_bootstrap_seed_groups": seeds,
            **decoded,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
