from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from realistic_niah_v5.mechanism_dataset import build_mechanism_dataset


def _repository_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the shared geometry panel, registered non-thinking/native-thinking "
            "mode contracts, and causal-extension registry from the exact audited "
            "V4.4 frozen stimulus backbone."
        )
    )
    parser.add_argument(
        "--source-dataset-dir",
        default="exports/run_20260731_v4_numeric_presentation_v3/dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/realistic_niah_count_mechanism_analysis_v2",
    )
    args = parser.parse_args()
    result = build_mechanism_dataset(
        source_dataset_dir=Path(args.source_dataset_dir),
        output_dir=Path(args.output_dir),
        repository_head=_repository_head(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
