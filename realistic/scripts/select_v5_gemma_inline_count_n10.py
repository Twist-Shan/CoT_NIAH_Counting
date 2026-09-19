#!/usr/bin/env python3
"""Select complete Gemma N=10 traces with one frozen inline-count suffix family."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCHEMA = "realistic_niah_v5_gemma_inline_count_n10_selection_v1"
MODEL = "Gemma4-E4B"
GRAMMAR = "adjacent_rank_after_city"
MARKER = "inline_count"
TARGET_N = 10
SECONDARY_SPLIT_SALT = "gemma_n10_single_episode_count_colon_20_10_v1"
FAMILY_PATTERNS = {
    "count_colon": re.compile(r"\(Count:\s*(\d+)\)\s*$"),
    "count_equals": re.compile(r"\(Count\s*=\s*(\d+)\)\s*$"),
}


def _truth(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(False).astype(str).str.lower().eq("true")


def select(registry: pd.DataFrame, family: str) -> pd.DataFrame:
    pattern = FAMILY_PATTERNS[family]
    eligible = registry.loc[
        registry["model_label"].astype(str).eq(MODEL)
        & _truth(registry, "primary_full_chain_event")
        & _truth(registry, "progress_commit_eligible")
        & _truth(registry, "progress_commit_site_resolved")
        & _truth(registry, "exact_count")
        & registry["trace_category"].astype(str).eq("one_to_one")
        & registry["gold_count"].astype(int).eq(TARGET_N)
        & registry["parsed_count"].astype(int).eq(TARGET_N)
        & registry["grammar_class"].astype(str).eq(GRAMMAR)
        & registry["marker_kind"].astype(str).eq(MARKER)
    ].copy()
    rows: list[dict[str, object]] = []
    for request_id, group in eligible.groupby("request_id", sort=True):
        group = group.sort_values("occurrence", kind="mergesort")
        occurrences = group["occurrence"].astype(int).tolist()
        ranks = group["rank"].astype(int).tolist()
        if len(group) != TARGET_N:
            continue
        if occurrences != list(range(1, TARGET_N + 1)) or ranks != occurrences:
            continue
        suffix_values: list[int] = []
        valid = True
        for event in group.itertuples(index=False):
            match = pattern.search(str(event.item_text).strip())
            if match is None or int(match.group(1)) != int(event.occurrence):
                valid = False
                break
            if str(event.commit_token_text).strip() != ")":
                valid = False
                break
            suffix_values.append(int(match.group(1)))
        if not valid or suffix_values != occurrences:
            continue
        first = group.iloc[0]
        rows.append(
            {
                "model_label": MODEL,
                "request_id": str(request_id),
                "seed": int(first["seed"]),
                "split": str(first["split"]),
                "gold_count": TARGET_N,
                "grammar_class": GRAMMAR,
                "marker_kind": MARKER,
                "surface_family": family,
                "endpoint_family": "closing_parenthesis_after_explicit_count",
                "states": TARGET_N,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"No complete Gemma traces for family {family!r}")
    if result["seed"].duplicated().any():
        raise ValueError("Selected Gemma panel contains duplicate seeds")
    return result.sort_values(["split", "seed"], kind="mergesort").reset_index(
        drop=True
    )


def _single_inline_count_episode(
    generation: dict[str, object], family: str
) -> bool:
    parsed = dict(generation.get("trace_parse", {}))
    parser = dict(parsed.get("parser", {}))
    if not bool(parsed.get("exact_count")):
        return False
    if str(parser.get("trace_category")) != "one_to_one":
        return False
    events = list(dict(parsed.get("episode_parse", {})).get("events", ()))
    if len(events) != TARGET_N:
        return False
    for occurrence, event in enumerate(events, start=1):
        event = dict(event)
        if int(event.get("rank", -1)) != occurrence:
            return False
        if str(event.get("association")) != "rank_after_city":
            return False
        if str(event.get("evidence_family")) != "inline_count":
            return False
        family_pattern = {
            "count_colon": rf"Count:\s*{occurrence}\)",
            "count_equals": rf"Count\s*=\s*{occurrence}\)",
        }[family]
        if re.fullmatch(
            family_pattern, str(event.get("evidence_surface", "")).strip()
        ) is None:
            return False
    return True


def _load_generations(paths: list[Path]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                request_id = str(row["request_id"])
                if request_id in rows:
                    raise ValueError(f"Duplicate generation request_id: {request_id}")
                rows[request_id] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-registry", type=Path, action="append", required=True)
    parser.add_argument("--family", choices=tuple(FAMILY_PATTERNS), required=True)
    parser.add_argument("--generation-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--require-single-episode", action="store_true")
    parser.add_argument(
        "--target-total",
        type=int,
        help=(
            "Parser-only deterministic subsample size before the secondary split; "
            "hidden states and outcomes are never consulted."
        ),
    )
    parser.add_argument("--secondary-20-10", action="store_true")
    parser.add_argument(
        "--secondary-confirmation-count",
        type=int,
        help="Exploratory deterministic split with this many confirmation traces.",
    )
    parser.add_argument("--min-seed", type=int)
    parser.add_argument("--max-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = pd.concat(
        [pd.read_csv(path) for path in args.event_registry], ignore_index=True
    )
    selected = select(registry, args.family)
    if args.require_single_episode:
        if not args.generation_jsonl:
            raise ValueError("--require-single-episode needs --generation-jsonl")
        generations = _load_generations(args.generation_jsonl)
        missing = sorted(
            set(selected["request_id"].astype(str)) - set(generations)
        )
        if missing:
            raise ValueError(f"Missing raw generations for selected traces: {missing}")
        selected = selected.loc[
            selected["request_id"].astype(str).map(
                lambda request_id: _single_inline_count_episode(
                    generations[request_id], args.family
                )
            )
        ].copy()
        if selected.empty:
            raise ValueError("Single-episode audit removed every selected trajectory")
    selection_salt = None
    if args.target_total is not None:
        if args.target_total <= 0:
            raise ValueError("--target-total must be positive")
        if len(selected) < args.target_total:
            raise ValueError(
                f"Target total {args.target_total} exceeds eligible traces {len(selected)}"
            )
        selection_salt = (
            f"gemma_n10_single_episode_{args.family}_target_{args.target_total}_v1"
        )
        selected["selection_hash"] = selected["seed"].astype(int).map(
            lambda seed: hashlib.sha256(
                f"{selection_salt}:{seed}".encode("utf-8")
            ).hexdigest()
        )
        selected = selected.sort_values(
            ["selection_hash", "seed"], kind="mergesort"
        ).iloc[: args.target_total].copy()
    if args.secondary_20_10 and args.secondary_confirmation_count is not None:
        raise ValueError(
            "Use either --secondary-20-10 or --secondary-confirmation-count, not both"
        )
    secondary_confirmation_count = (
        10 if args.secondary_20_10 else args.secondary_confirmation_count
    )
    if secondary_confirmation_count is not None:
        if secondary_confirmation_count <= 0:
            raise ValueError("Secondary confirmation count must be positive")
        if len(selected) <= secondary_confirmation_count:
            raise ValueError(
                "Secondary split needs at least one discovery trace: "
                f"total={len(selected)}, confirmation={secondary_confirmation_count}"
            )
        if args.secondary_20_10 and len(selected) != 30:
            raise ValueError(
                f"Secondary 20/10 split requires exactly 30 traces, got {len(selected)}"
            )
        selected["source_split"] = selected["split"].astype(str)
        secondary_split_salt = (
            f"gemma_n10_single_episode_{args.family}_20_10_v1"
        )
        selected["secondary_split_hash"] = selected["seed"].astype(int).map(
            lambda seed: hashlib.sha256(
                f"{secondary_split_salt}:{seed}".encode("utf-8")
            ).hexdigest()
        )
        ranked = selected.sort_values(
            ["secondary_split_hash", "seed"], kind="mergesort"
        )
        confirmation_ids = set(
            ranked.iloc[:secondary_confirmation_count]["request_id"].astype(str)
        )
        selected["split"] = selected["request_id"].astype(str).map(
            lambda request_id: (
                "confirmation" if request_id in confirmation_ids else "discovery"
            )
        )
        selected["split_role"] = selected["split"].map(
            {
                "discovery": "secondary_discovery",
                "confirmation": "secondary_confirmation",
            }
        )
    if args.min_seed is not None:
        selected = selected.loc[selected["seed"].astype(int).ge(args.min_seed)].copy()
    if args.max_seed is not None:
        selected = selected.loc[selected["seed"].astype(int).le(args.max_seed)].copy()
    if selected.empty:
        raise ValueError("Seed bounds removed every selected trajectory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    summary = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_label": MODEL,
        "grammar_class": GRAMMAR,
        "marker_kind": MARKER,
        "surface_family": args.family,
        "require_single_episode": bool(args.require_single_episode),
        "secondary_20_10": bool(args.secondary_20_10),
        "secondary_confirmation_count": secondary_confirmation_count,
        "secondary_split_salt": (
            f"gemma_n10_single_episode_{args.family}_20_10_v1"
            if secondary_confirmation_count is not None
            else None
        ),
        "target_total": args.target_total,
        "selection_salt": selection_salt,
        "selection_unit": "whole exact one-to-one N=10 trajectory",
        "split_counts": {
            key: int(value)
            for key, value in selected.groupby("split").size().to_dict().items()
        },
        "seeds": selected["seed"].astype(int).tolist(),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
