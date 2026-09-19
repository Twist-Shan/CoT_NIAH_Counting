from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v4.prompt_counter_dynamics import (
    analyze_prompt_counter_dynamics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze write-side needle-end attention dispersion and its "
            "association with prompt-counter noise."
        )
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    paths = analyze_prompt_counter_dynamics(
        args.run_root,
        output_dir=args.output_dir,
        top_n=args.top_n,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(
        json.dumps(
            {key: str(path.resolve()) for key, path in paths.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
