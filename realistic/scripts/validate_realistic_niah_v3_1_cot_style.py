from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from realistic_niah_v3_1.cot_style import evaluate_style_annotations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate V3.1 automated CoT style labels against human annotations."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--automated-styles", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unweighted", action="store_true")
    args = parser.parse_args()
    annotations = pd.read_csv(args.annotations)
    automated = pd.read_csv(args.automated_styles)
    summary, per_label, confusion = evaluate_style_annotations(
        annotations,
        automated,
        weight_column=None if args.unweighted else "analysis_weight",
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "style_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    per_label.to_csv(output / "style_validation_per_label.csv", index=False)
    confusion.to_csv(output / "style_validation_confusion.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["confirmatory_automated_reporting_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
