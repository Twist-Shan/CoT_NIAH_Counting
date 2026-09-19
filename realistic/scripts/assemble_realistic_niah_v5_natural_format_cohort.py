#!/usr/bin/env python3
"""Fill format-ineligible base seeds from a preregistered independent seed pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


BASE_DISCOVERY = tuple(range(1234, 1254))
BASE_CONFIRMATION = tuple(range(1254, 1264))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_by_seed(rows: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"Natural format cohort has duplicate seed={seed}")
        result[seed] = dict(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selected", type=Path, required=True)
    parser.add_argument("--supplement-selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-attempt", type=int)
    args = parser.parse_args()

    base_rows = _read_jsonl(args.base_selected)
    supplement_rows = _read_jsonl(args.supplement_selected)
    if args.required_attempt is not None:
        base_rows = [
            row
            for row in base_rows
            if int(row.get("natural_unnumbered_attempt", -1)) == args.required_attempt
        ]
        supplement_rows = [
            row
            for row in supplement_rows
            if int(row.get("natural_unnumbered_attempt", -1)) == args.required_attempt
        ]
    base = _unique_by_seed(base_rows)
    supplement = _unique_by_seed(supplement_rows)
    base_discovery = [base[seed] for seed in BASE_DISCOVERY if seed in base]
    base_confirmation = [base[seed] for seed in BASE_CONFIRMATION if seed in base]
    need_discovery = 20 - len(base_discovery)
    need_confirmation = 10 - len(base_confirmation)
    if need_discovery < 0 or need_confirmation < 0:
        raise ValueError("Base natural format cohort exceeds frozen 20/10 contract")

    supplement_discovery = sorted(
        (row for row in supplement.values() if str(row.get("split")) == "discovery"),
        key=lambda row: int(row["seed"]),
    )
    supplement_confirmation = sorted(
        (row for row in supplement.values() if str(row.get("split")) == "confirmation"),
        key=lambda row: int(row["seed"]),
    )
    if len(supplement_discovery) < need_discovery:
        raise RuntimeError(
            f"Need {need_discovery} eligible discovery supplements, found {len(supplement_discovery)}"
        )
    if len(supplement_confirmation) < need_confirmation:
        raise RuntimeError(
            f"Need {need_confirmation} eligible confirmation supplements, found {len(supplement_confirmation)}"
        )
    chosen_discovery = supplement_discovery[:need_discovery]
    chosen_confirmation = supplement_confirmation[:need_confirmation]
    discovery = sorted(base_discovery + chosen_discovery, key=lambda row: int(row["seed"]))
    confirmation = sorted(
        base_confirmation + chosen_confirmation, key=lambda row: int(row["seed"])
    )
    if len(discovery) != 20 or len(confirmation) != 10:
        raise RuntimeError("Natural format supplement failed to restore the 20/10 contract")
    if {int(row["seed"]) for row in discovery} & {
        int(row["seed"]) for row in confirmation
    }:
        raise RuntimeError("Discovery and confirmation supplement cohorts overlap")

    rows = discovery + confirmation
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    selected = output / "selected_generations.jsonl"
    _atomic_jsonl(selected, rows)
    _atomic_json(
        output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_natural_format_cohort_v1",
            "status": "PASS",
            "selection_rule": (
                "retain eligible original seeds; fill each split with the lowest eligible "
                "seed from its preregistered independent pool; patch outcomes unavailable"
            ),
            "base_discovery_seed_count": len(base_discovery),
            "base_confirmation_seed_count": len(base_confirmation),
            "replacement_discovery_seed_count": len(chosen_discovery),
            "replacement_confirmation_seed_count": len(chosen_confirmation),
            "discovery_seeds": [int(row["seed"]) for row in discovery],
            "confirmation_seeds": [int(row["seed"]) for row in confirmation],
            "discovery_replacements": [int(row["seed"]) for row in chosen_discovery],
            "confirmation_replacements": [int(row["seed"]) for row in chosen_confirmation],
            "outcome_blind": True,
            "selection_rank_used": False,
            "required_attempt": args.required_attempt,
            "prompt_conditioned_a7_auxiliary": args.required_attempt == 7,
            "formal_frozen_prompt_claim_allowed": args.required_attempt is None,
            "selected_generations_sha256": _sha256(selected),
        },
    )


if __name__ == "__main__":
    main()
