#!/usr/bin/env python3
"""CPU-only geometry preflight for the three-entry marker ledger factorial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import load_registered_tokenizer  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.event_ledger import build_marker_event_factorial  # noqa: E402
from scripts.run_realistic_niah_v5_count_stream import _atomic_json  # noqa: E402
from scripts.run_realistic_niah_v5_unified_carrier_transition import (  # noqa: E402
    _read_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--receiver", type=int, default=5)
    parser.add_argument("--source-occurrences", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = tuple(dict.fromkeys(int(value) for value in args.seeds))
    source_occurrences = tuple(int(value) for value in args.source_occurrences)
    rows = _read_rows(args.generations, seeds)
    tokenizer = load_registered_tokenizer(
        resolve_model_spec(args.model), cache_dir=args.cache_dir
    )
    audits: list[dict[str, Any]] = []
    for seed in seeds:
        try:
            source, blank, registry, _scrub = build_diagnostic_bases(
                rows[seed],
                tokenizer,
                random_seed=20260830 + seed,
                construction="targeted_explicit_count_scrub",
            )
            boundaries = {
                occurrence: select_post_item_boundary_position(
                    source, registry, tokenizer, occurrence=occurrence
                )[0]
                for occurrence in range(1, 11)
            }
            variants, geometry = build_marker_event_factorial(
                source,
                blank,
                registry,
                boundaries,
                receiver=int(args.receiver),
                source_occurrences=source_occurrences,
            )
            audits.append(
                {
                    "seed": seed,
                    "eligible": True,
                    "factorial_cell_count": len(variants),
                    "sequence_length": int(variants[0]["encoding"].sequence_length),
                    "insertion_start": int(geometry["insertion_start"]),
                    "event_end": int(geometry["event_end"]),
                    "target_marker_position": int(
                        geometry["target_marker_position"]
                    ),
                    "target_boundary": int(geometry["target_boundary"]),
                    "inserted_slots": geometry["inserted_slots"],
                    "only_marker_token_ids_vary": bool(
                        geometry["only_marker_token_ids_vary"]
                    ),
                }
            )
        except Exception as exc:
            audits.append(
                {
                    "seed": seed,
                    "eligible": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    output = {
        "schema_version": "event_ledger_geometry_preflight_v1",
        "model": args.model,
        "receiver": int(args.receiver),
        "source_occurrences": list(source_occurrences),
        "seeds": list(seeds),
        "eligible_seeds": [
            int(row["seed"]) for row in audits if bool(row["eligible"])
        ],
        "ineligible_seeds": [
            int(row["seed"]) for row in audits if not bool(row["eligible"])
        ],
        "all_eligible": all(bool(row["eligible"]) for row in audits),
        "audits": audits,
        "model_forward_executed": False,
        "effect_outcomes_read": False,
    }
    _atomic_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
