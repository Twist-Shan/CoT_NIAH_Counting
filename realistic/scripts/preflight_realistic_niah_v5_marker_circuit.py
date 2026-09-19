#!/usr/bin/env python3
"""CPU-only deterministic geometry preflight for marker-circuit cohorts."""

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
from realistic_niah_v5.marker_circuit import edge_key_positions  # noqa: E402
from scripts.run_realistic_niah_v5_count_stream import _atomic_json  # noqa: E402
from scripts.run_realistic_niah_v5_event_cache_splice import (  # noqa: E402
    PRIMARY_INVALID_VARIANT,
    VALID_VARIANT,
    build_cache_splice_geometry,
)
from scripts.run_realistic_niah_v5_event_commit_movie import (  # noqa: E402
    build_event_movie_geometry,
)
from scripts.run_realistic_niah_v5_list_event_edit_scan import (  # noqa: E402
    build_list_event_variants,
)
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
    parser.add_argument("--insert-source-occurrence", type=int, default=4)
    parser.add_argument("--delete-occurrence", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = tuple(dict.fromkeys(int(value) for value in args.seeds))
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
            variants = {
                str(variant["event_variant"]): variant
                for variant in build_list_event_variants(
                    source,
                    blank,
                    registry,
                    receiver=int(args.receiver),
                    current_boundary=boundaries[int(args.receiver)],
                    target_boundary=boundaries[int(args.receiver) + 1],
                    insert_source_occurrence=int(args.insert_source_occurrence),
                    delete_occurrence=int(args.delete_occurrence),
                )
            }
            geometry = build_cache_splice_geometry(
                variants[VALID_VARIANT],
                variants[PRIMARY_INVALID_VARIANT],
                registry,
                boundaries,
                receiver=int(args.receiver),
                insert_source_occurrence=int(args.insert_source_occurrence),
            )
            movie = build_event_movie_geometry(
                variants[VALID_VARIANT],
                registry,
                boundaries,
                receiver=int(args.receiver),
                insert_source_occurrence=int(args.insert_source_occurrence),
            )
            audits.append(
                {
                    "seed": seed,
                    "eligible": True,
                    "insertion_start": int(geometry["insertion_start"]),
                    "event_end": int(geometry["event_end"]),
                    "target_marker": int(movie["landmarks"]["target_marker_end"]),
                    "target_boundary": int(movie["landmarks"]["target_boundary"]),
                    "edge_key_positions": edge_key_positions(geometry),
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
        "schema_version": "marker_circuit_geometry_preflight_v1",
        "model": args.model,
        "seeds": list(seeds),
        "all_eligible": all(bool(row["eligible"]) for row in audits),
        "audits": audits,
        "model_forward_executed": False,
        "effect_outcomes_read": False,
    }
    _atomic_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
