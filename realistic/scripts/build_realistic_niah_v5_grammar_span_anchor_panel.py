#!/usr/bin/env python3
"""Freeze a balanced rank-before/after-city terminal-span anchor panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from functools import lru_cache
from typing import Any


DEVELOPMENT_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
BEFORE_CLASSES = (
    "adjacent_rank_before_city",
    "same_unit_rank_before_city",
)
AFTER_CLASSES = ("adjacent_rank_after_city",)


def _balanced_timing_assignment(
    rows: list[dict[str, Any]], phase_seeds: tuple[int, ...]
) -> dict[int, str]:
    """Prefer alternation while deterministically satisfying exact balance."""

    eligible: dict[int, tuple[str, ...]] = {}
    for seed in phase_seeds:
        available: list[str] = []
        for timing, allowed in (
            ("rank_after_city", AFTER_CLASSES),
            ("rank_before_city", BEFORE_CLASSES),
        ):
            if any(
                int(row.get("seed", -1)) == int(seed)
                and str(row.get("target_grammar_class", "")) in allowed
                and int(row.get("to_occurrence", -1))
                == int(row.get("gold_count", -2))
                for row in rows
            ):
                available.append(timing)
        if not available:
            raise ValueError(f"Seed {seed} has no eligible explicit-rank terminal anchor")
        eligible[int(seed)] = tuple(available)
    target = len(phase_seeds) // 2

    @lru_cache(maxsize=None)
    def solve(index: int, after_left: int, before_left: int) -> tuple[str, ...] | None:
        if index == len(phase_seeds):
            return () if after_left == 0 and before_left == 0 else None
        if after_left < 0 or before_left < 0:
            return None
        seed = int(phase_seeds[index])
        preferred = "rank_after_city" if index % 2 == 0 else "rank_before_city"
        choices = (preferred, "rank_before_city" if preferred == "rank_after_city" else "rank_after_city")
        for timing in choices:
            if timing not in eligible[seed]:
                continue
            suffix = solve(
                index + 1,
                after_left - int(timing == "rank_after_city"),
                before_left - int(timing == "rank_before_city"),
            )
            if suffix is not None:
                return (timing, *suffix)
        return None

    assignment = solve(0, target, target)
    if assignment is None:
        raise ValueError("No exact outcome-blind grammar timing balance is feasible")
    return {
        int(seed): timing for seed, timing in zip(phase_seeds, assignment, strict=True)
    }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_panel(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose an alternating, exactly balanced grammar panel without outcomes."""

    if any("selection_rank" in row for row in rows):
        raise ValueError("Grammar-span source registry contains selection_rank")
    expected_seeds = DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS
    selected: list[dict[str, Any]] = []
    for phase_seeds in (DEVELOPMENT_SEEDS, CONFIRMATION_SEEDS):
        assignment = _balanced_timing_assignment(rows, phase_seeds)
        for seed in phase_seeds:
            timing = assignment[int(seed)]
            allowed = AFTER_CLASSES if timing == "rank_after_city" else BEFORE_CLASSES
            candidates = [
                row
                for row in rows
                if int(row.get("seed", -1)) == int(seed)
                and str(row.get("target_grammar_class", "")) in allowed
                and int(row.get("to_occurrence", -1)) == int(row.get("gold_count", -2))
            ]
            if not candidates:
                raise ValueError(
                    f"Seed {seed} has no terminal {timing} anchor in the source registry"
                )
            candidates.sort(
                key=lambda row: (
                    -int(row["gold_count"]),
                    allowed.index(str(row["target_grammar_class"])),
                    str(row["request_id"]),
                )
            )
            value = dict(candidates[0])
            value.update(
                {
                    "grammar_span_outcome_blind": True,
                    "grammar_span_selection_rank_used": False,
                    "grammar_span_timing_stratum": timing,
                    "grammar_span_selection_rule": (
                        "feasibility_balanced_alternation_then_highest_gold_count_"
                        "then_grammar_preference_then_request_id"
                    ),
                }
            )
            selected.append(value)
    if tuple(int(row["seed"]) for row in selected) != expected_seeds:
        raise RuntimeError("Grammar-span panel changed the fixed 20+10 seed order")
    phase_counts: dict[str, dict[str, int]] = {}
    for phase, phase_seeds in (
        ("development", DEVELOPMENT_SEEDS),
        ("confirmation", CONFIRMATION_SEEDS),
    ):
        active = [row for row in selected if int(row["seed"]) in phase_seeds]
        phase_counts[phase] = {
            timing: sum(
                str(row["grammar_span_timing_stratum"]) == timing for row in active
            )
            for timing in ("rank_after_city", "rank_before_city")
        }
    if phase_counts != {
        "development": {"rank_after_city": 10, "rank_before_city": 10},
        "confirmation": {"rank_after_city": 5, "rank_before_city": 5},
    }:
        raise RuntimeError("Grammar-span panel is not exactly timing-balanced")
    core = {
        "schema_version": "realistic_niah_v5_grammar_span_anchor_panel_v1",
        "outcome_blind": True,
        "selection_rank_used": False,
        "development_seed_count": 20,
        "confirmation_seed_count": 10,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "timing_counts_by_phase": phase_counts,
        "selection_rule": (
            "feasibility_balanced_alternation_then_highest_gold_count_then_"
            "grammar_preference_then_request_id"
        ),
        "selected_request_ids": [str(row["request_id"]) for row in selected],
    }
    manifest = {**core, "panel_sha256": _sha256_json(selected), "manifest_sha256": _sha256_json(core)}
    return selected, manifest


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    panel, manifest = build_panel(rows)
    panel_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in panel)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != panel_text:
        raise ValueError("Existing frozen grammar-span panel changed")
    if args.manifest.exists() and args.manifest.read_text(encoding="utf-8") != manifest_text:
        raise ValueError("Existing frozen grammar-span manifest changed")
    _atomic_text(args.output, panel_text)
    _atomic_text(args.manifest, manifest_text)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
