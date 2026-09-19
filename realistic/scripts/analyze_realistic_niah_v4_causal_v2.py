from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from realistic_niah_v4.causal_v2_analysis import write_causal_v2_analysis


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path).resolve(), compression="infer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build V4.4 causal-v2 top-k and all-layer +/-k analysis tables "
            "and Aurora figures from completed formal detail files."
        )
    )
    parser.add_argument("--ablation-detail", required=True)
    parser.add_argument("--prompt-screen-detail", required=True)
    parser.add_argument("--answer-screen-detail", required=True)
    parser.add_argument("--steering-screen-detail", required=True)
    parser.add_argument("--prompt-confirmation-statistics")
    parser.add_argument("--answer-confirmation-statistics")
    parser.add_argument("--steering-confirmation-statistics")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()

    confirmations: list[tuple[str, pd.DataFrame]] = []
    for name, path in (
        ("prompt_patching", args.prompt_confirmation_statistics),
        ("answer_patching", args.answer_confirmation_statistics),
        ("steering", args.steering_confirmation_statistics),
    ):
        if path:
            confirmations.append((name, _read(path)))
    outputs = write_causal_v2_analysis(
        output_dir=Path(args.output_dir).resolve(),
        ablation_detail=_read(args.ablation_detail),
        prompt_screen_detail=_read(args.prompt_screen_detail),
        answer_screen_detail=_read(args.answer_screen_detail),
        steering_screen_detail=_read(args.steering_screen_detail),
        confirmation_tables=confirmations,
        bootstrap_repetitions=int(args.bootstrap_repetitions),
    )
    print(
        json.dumps(
            {name: str(path) for name, path in sorted(outputs.items())},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
