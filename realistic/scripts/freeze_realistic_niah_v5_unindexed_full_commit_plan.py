#!/usr/bin/env python3
"""Freeze outcome-blind full-commit pairs for an arbitrary audited seed list."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.native_loop import build_fixed_native_loop_plan  # noqa: E402
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_no_count_enumeration_trace,
)
from scripts.freeze_realistic_niah_v5_full_commit_specificity_plan import (  # noqa: E402
    freeze_plan,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    wanted = set(seeds)
    result: list[dict[str, Any]] = []
    observed: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seed = int(row.get("seed", -1))
        if seed not in wanted:
            continue
        if seed in observed:
            raise ValueError(f"Duplicate selected seed {seed}")
        audit = audit_no_count_enumeration_trace(row)
        if not bool(audit["eligible"]):
            raise ValueError(f"Seed {seed} fails unindexed gate: {audit['reasons']}")
        if int(row.get("gold_count", -1)) != 10:
            raise ValueError(f"Seed {seed} is not N=10")
        result.append(row)
        observed.add(seed)
    if observed != wanted:
        raise ValueError(f"Missing selected seeds {sorted(wanted-observed)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seed-role", choices=("development", "confirmation"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--donor-offsets", type=int, nargs="+", default=(-1, 1))
    parser.add_argument("--random-seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = tuple(int(value) for value in args.seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Selected seeds must be unique")
    rows = _rows(args.generations, seeds)
    base = build_fixed_native_loop_plan(
        rows,
        model_label=str(args.model),
        seeds=seeds,
        seed_role=str(args.seed_role),
        donor_offsets=tuple(int(value) for value in args.donor_offsets),
        candidate_counts=tuple(range(2, 11)),
        sampling_seed=int(args.random_seed),
        require_all_seeds_per_offset=True,
        include_boundaries=True,
    )
    frozen = freeze_plan(
        base,
        donor_offsets=tuple(int(value) for value in args.donor_offsets),
        random_seed=int(args.random_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(args.output, frozen)
    manifest_path = args.output.with_suffix(".manifest.json")
    _atomic_json(
        manifest_path,
        {
            "schema_version": "unindexed_full_commit_pair_plan_v1",
            "model_label": str(args.model),
            "generations": str(args.generations.resolve()),
            "generations_sha256": _sha256(args.generations),
            "seed_role": str(args.seed_role),
            "seeds": list(seeds),
            "donor_offsets": [int(value) for value in args.donor_offsets],
            "random_seed": int(args.random_seed),
            "pair_count": int(len(frozen)),
            "plan_sha256": _sha256(args.output),
            "selection_uses_outcomes": False,
            "selection_rank_used": False,
            "unindexed_gate_required": True,
            "gold_count_required": 10,
            "formal_frozen_prompt_claim_allowed": False,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pair_count": int(len(frozen)),
                "seed_count": len(seeds),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
