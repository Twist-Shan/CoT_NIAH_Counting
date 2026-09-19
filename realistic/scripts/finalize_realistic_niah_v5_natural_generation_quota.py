#!/usr/bin/env python3
"""Seal a preregistered natural-format pool once split quotas are satisfied."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--discovery-pool", type=int, nargs="+", required=True)
    parser.add_argument("--confirmation-pool", type=int, nargs="+", required=True)
    parser.add_argument("--required-discovery", type=int, required=True)
    parser.add_argument("--required-confirmation", type=int, required=True)
    parser.add_argument("--source-counts", type=int, nargs="+", default=(10, 9))
    parser.add_argument("--attempts", type=int, nargs="+", default=(7, 8, 9))
    args = parser.parse_args()

    root = args.generation_root
    plan = json.loads((root / "frozen_generation_plan.json").read_text())
    frozen_pool = tuple(int(value) for value in plan["planned_seeds"])
    supplied_pool = tuple(args.discovery_pool) + tuple(args.confirmation_pool)
    if frozen_pool != supplied_pool:
        raise ValueError("Finalization pool differs from frozen generation plan")
    if plan.get("outcome_blind") is not True or plan.get("selection_rank_used") is not False:
        raise ValueError("Frozen generation plan is not outcome blind")

    selected: list[dict[str, Any]] = []
    seed_audits: list[dict[str, Any]] = []
    for seed in frozen_pool:
        chosen: dict[str, Any] | None = None
        examined = []
        for count in args.source_counts:
            for attempt in args.attempts:
                path = root / "attempts" / f"seed{seed}_N{count}_attempt{attempt}.json"
                if not path.exists():
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
                audit = row["no_count_enumeration_audit"]
                examined.append(
                    {
                        "gold_count": int(count),
                        "attempt": int(attempt),
                        "eligible": bool(audit["eligible"]),
                    }
                )
                if bool(audit["eligible"]):
                    chosen = row
                    break
            if chosen is not None:
                break
        seed_audits.append(
            {"seed": int(seed), "eligible": chosen is not None, "attempts": examined}
        )
        if chosen is not None:
            selected.append(chosen)

    discovery_eligible = sorted(
        (row for row in selected if str(row.get("split")) == "discovery"),
        key=lambda row: int(row["seed"]),
    )
    confirmation_eligible = sorted(
        (row for row in selected if str(row.get("split")) == "confirmation"),
        key=lambda row: int(row["seed"]),
    )
    if len(discovery_eligible) < args.required_discovery:
        raise RuntimeError("Discovery format quota was not met before finalization")
    if len(confirmation_eligible) < args.required_confirmation:
        raise RuntimeError("Confirmation format quota was not met before finalization")
    _atomic_jsonl(root / "selected_generations.jsonl", selected)
    _atomic_json(
        root / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_natural_unnumbered_quota_manifest_v1",
            "status": "PASS_FORMAT_QUOTA_REACHED",
            "model_label": str(plan["model_label"]),
            "planned_seeds": list(frozen_pool),
            "processed_seeds": [row["seed"] for row in seed_audits if row["attempts"]],
            "eligible_discovery_seed_count": len(discovery_eligible),
            "eligible_confirmation_seed_count": len(confirmation_eligible),
            "required_discovery_seed_count": int(args.required_discovery),
            "required_confirmation_seed_count": int(args.required_confirmation),
            "quota_stopping_rule": (
                "stop after the lowest processed seed makes both format-only split "
                "quotas attainable; higher frozen candidate seeds are not generated"
            ),
            "patch_outcomes_available_during_selection": False,
            "teacher_forcing": False,
            "trace_tokens_model_generated": True,
            "outcome_blind": True,
            "selection_rank_used": False,
            "seed_audits": seed_audits,
        },
    )


if __name__ == "__main__":
    main()
