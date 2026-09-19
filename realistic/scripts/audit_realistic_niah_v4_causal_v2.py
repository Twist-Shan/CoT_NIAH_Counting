from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from realistic_niah_v4.causal_v2 import CausalV2Design
from realistic_niah_v4.causal_v2_audit import (
    audit_causal_v2_run,
    render_audit_markdown,
)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly audit one model's Realistic NIAH V4.4 causal-v2 run."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument(
        "--causal-config", default="configs/realistic_niah_v4_causal_v2.json"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--require-confirmation", action="store_true")
    args = parser.parse_args()

    design = CausalV2Design.from_json(Path(args.causal_config).resolve())
    payload = audit_causal_v2_run(
        run_root=Path(args.run_root).resolve(),
        model_label=args.model,
        stimuli_path=Path(args.stimuli).resolve(),
        design=design,
        require_confirmation=args.require_confirmation,
    )
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(args.run_root).resolve()
        / args.model
        / "numeric"
        / "causal_v2"
        / "audit"
    )
    json_path = output / "audit.json"
    markdown_path = output / "audit.md"
    _write_text_atomic(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(markdown_path, render_audit_markdown(payload))
    print(json.dumps({"audit": str(json_path), **payload}, ensure_ascii=False))
    if payload["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
