#!/usr/bin/env python3
"""Compile the isolated N=10 supplement with the frozen causal-site parser."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--qwen-generations", type=Path, required=True)
    parser.add_argument("--gemma-generations", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer", type=Path, required=True)
    parser.add_argument("--gemma-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config["purpose"] = (
        "Compile the predeclared N=10 seed supplement with the frozen causal-site parser; "
        "never merge these rows into the registered primary 300."
    )
    config["models"]["Qwen3-8B"]["input_jsonl"] = str(
        args.qwen_generations.resolve()
    )
    config["models"]["Qwen3-8B"]["expected_rows"] = 100
    config["models"]["Qwen3-8B"]["tokenizer_json"] = str(
        args.qwen_tokenizer.resolve()
    )
    config["models"]["Gemma4-E4B"]["input_jsonl"] = str(
        args.gemma_generations.resolve()
    )
    config["models"]["Gemma4-E4B"]["expected_rows"] = 100
    config["models"]["Gemma4-E4B"]["tokenizer_json"] = str(
        args.gemma_tokenizer.resolve()
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
