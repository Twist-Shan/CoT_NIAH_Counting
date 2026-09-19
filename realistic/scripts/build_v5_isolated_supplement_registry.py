#!/usr/bin/env python3
"""Compile one isolated generation supplement with the frozen causal-site parser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.build_realistic_niah_v5_causal_site_review import build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    model_config = dict(config["models"][args.model])
    model_config["input_jsonl"] = str(args.generations.resolve())
    model_config["expected_rows"] = int(args.expected_rows)
    model_config["tokenizer_json"] = str(args.tokenizer.resolve())
    config["models"] = {args.model: model_config}
    config["purpose"] = (
        "Compile an isolated adaptive N=10 supplement with the frozen causal-site "
        "parser; never merge raw rows into the registered primary 300."
    )
    config["output_dir"] = str(args.output_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "supplement_causal_site_review_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output, audit = build(config_path)
    print(json.dumps({"output_dir": str(output), **audit}, sort_keys=True))


if __name__ == "__main__":
    main()
