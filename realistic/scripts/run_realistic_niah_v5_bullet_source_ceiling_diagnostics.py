#!/usr/bin/env python3
"""Compare Source/Blank greedy ceilings under structural scrub variants."""

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

from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
    build_item_early_stop_with_suffix,
    terminal_suffix_with_optional_newline,
)
from realistic_niah_v5.bullet_greedy_restore import _score_greedy_encoding  # noqa: E402
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_jsonl,
    _model,
)


CONSTRUCTIONS = (
    "current",
    "structural_indices_scrubbed",
    "structural_indices_intact",
    "targeted_explicit_count_scrub",
    "targeted_explicit_count_scrub_masked_index_punctuation",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--constructions", nargs="+", choices=CONSTRUCTIONS, default=CONSTRUCTIONS
    )
    parser.add_argument(
        "--newline-mode", choices=("both", "tight", "prepended"), default="both"
    )
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.command = "bullet-source-ceiling-diagnostics"

    wanted = {int(value) for value in args.seeds}
    rows = [
        json.loads(line)
        for line in args.generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if int(row["seed"]) in wanted]
    if {int(row["seed"]) for row in selected} != wanted:
        raise ValueError("One or more requested diagnostic seeds are absent")
    model, tokenizer, adapter = _model(args)
    results: list[dict[str, object]] = []
    for row in selected:
        for construction in tuple(dict.fromkeys(args.constructions)):
            source_full, blank_full, registry, audit = build_diagnostic_bases(
                row,
                tokenizer,
                random_seed=20260825 + int(row["seed"]),
                construction=construction,
            )
            newline_values = {
                "both": (False, True),
                "tight": (False,),
                "prepended": (True,),
            }[str(args.newline_mode)]
            for prepend_newline in newline_values:
                suffix = terminal_suffix_with_optional_newline(
                    row, tokenizer, prepend_newline=prepend_newline
                )
                for occurrence in range(1, 11):
                    source, early_audit = build_item_early_stop_with_suffix(
                        source_full,
                        registry,
                        target_occurrence=occurrence,
                        terminal_suffix_token_ids=suffix,
                    )
                    blank, blank_early_audit = build_item_early_stop_with_suffix(
                        blank_full,
                        registry,
                        target_occurrence=occurrence,
                        terminal_suffix_token_ids=suffix,
                    )
                    if early_audit != blank_early_audit:
                        raise RuntimeError("Source/Blank early-stop geometry differs")
                    for condition, encoding in (
                        ("source_reference", source),
                        ("blank_reference", blank),
                    ):
                        outcomes = _score_greedy_encoding(
                            model,
                            tokenizer,
                            adapter,
                            encoding,
                            target_k=occurrence,
                            max_new_tokens=int(args.max_new_tokens),
                        )
                        results.append(
                            {
                                "schema_version": "bullet_source_ceiling_diagnostic_v1",
                                "model_label": str(args.model),
                                "seed": int(row["seed"]),
                                "request_id": str(row["request_id"]),
                                "marker_kind": str(audit["marker_kind"]),
                                "construction": construction,
                                "prepend_newline": bool(prepend_newline),
                                "condition": condition,
                                "target_occurrence": occurrence,
                                **early_audit,
                                **outcomes,
                            }
                        )
            print(
                f"[source-ceiling] model={args.model} seed={row['seed']} "
                f"construction={construction}",
                flush=True,
            )
    _atomic_jsonl(args.output, results)
    print(f"[source-ceiling] wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
