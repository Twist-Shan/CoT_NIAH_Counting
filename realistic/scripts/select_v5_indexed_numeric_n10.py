#!/usr/bin/env python3
"""Select complete N=10 Qwen traces with the exact numeric list template 1..10."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA = "realistic_niah_v5_indexed_numeric_n10_selection_v1"
STRICT_DASH_SCHEMA = (
    "realistic_niah_v5_indexed_numeric_n10_strict_dash_20_10_selection_v1"
)
MODEL = "Qwen3-8B"
GRAMMAR = "adjacent_rank_before_city"
MARKER = "indexed"
TARGET_N = 10
STRICT_DASH_SPLIT_SALT = "qwen_n10_strict_dash_secondary_holdout_v1"


def _truth(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(False).astype(str).str.lower().eq("true")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select(registry: pd.DataFrame) -> pd.DataFrame:
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
    rows: list[dict[str, Any]] = []
    for request_id, group in eligible.groupby("request_id", sort=True):
        group = group.sort_values("occurrence", kind="mergesort")
        occurrences = group["occurrence"].astype(int).tolist()
        ranks = group["rank"].astype(int).tolist()
        rank_texts = group["rank_text"].astype(str).str.strip().tolist()
        if len(group) != TARGET_N:
            continue
        if occurrences != list(range(1, TARGET_N + 1)):
            continue
        if ranks != occurrences:
            continue
        if rank_texts != [str(value) for value in occurrences]:
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
                "rank_template": "exact_numeric_1_to_10",
                "states": TARGET_N,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No exact numeric-index N=10 trajectories were found")
    if result["seed"].duplicated().any():
        raise ValueError("Selected numeric-index panel contains duplicate seeds")
    return result.sort_values(["split", "seed"], kind="mergesort").reset_index(
        drop=True
    )


def select_strict_dash_20_10(registry: pd.DataFrame) -> pd.DataFrame:
    """Select the compact ``k. city - score`` family and freeze a 20/10 split.

    The surface family was motivated by the Appendix-E cluster diagnosis, so
    this remains an explicitly exploratory secondary panel.  Once the thirty
    text-eligible trajectories are known, however, the 20/10 assignment uses
    only the registered source split and a stable seed hash; hidden states and
    downstream geometry metrics never enter the assignment.
    """

    numeric = select(registry)
    strict_rows: list[dict[str, Any]] = []
    for row in numeric.itertuples(index=False):
        group = registry.loc[
            registry["request_id"].astype(str).eq(str(row.request_id))
        ].copy()
        group = group.sort_values("occurrence", kind="mergesort")
        if len(group) != TARGET_N:
            continue
        valid = True
        for event in group.itertuples(index=False):
            occurrence = int(event.occurrence)
            city = str(event.city).strip()
            item_text = str(event.item_text).strip()
            pattern = re.compile(
                rf"{occurrence}\.\s+{re.escape(city)}\s+-\s+\d+"
            )
            if pattern.fullmatch(item_text) is None:
                valid = False
                break
            if not str(event.commit_token_text).strip().isdigit():
                valid = False
                break
        if not valid:
            continue
        strict_rows.append(
            {
                **row._asdict(),
                "source_split": str(row.split),
                "surface_template": "exact_rank_dot_city_dash_score",
                "endpoint_family": "score_digit",
            }
        )
    strict = pd.DataFrame(strict_rows)
    if strict.empty:
        raise ValueError("No strict k. city - score trajectories were found")
    source_discovery = strict.loc[
        strict["source_split"].astype(str).eq("discovery")
    ].copy()
    source_confirmation = strict.loc[
        strict["source_split"].astype(str).eq("confirmation")
    ].copy()
    if len(source_discovery) != 11 or len(source_confirmation) != 19:
        raise ValueError(
            "Strict-dash source support changed unexpectedly: "
            f"D/C={len(source_discovery)}/{len(source_confirmation)}"
        )
    ranked_confirmation = source_confirmation.assign(
        secondary_split_hash=source_confirmation["seed"].astype(int).map(
            lambda seed: hashlib.sha256(
                f"{STRICT_DASH_SPLIT_SALT}:{seed}".encode("utf-8")
            ).hexdigest()
        )
    ).sort_values(["secondary_split_hash", "seed"], kind="mergesort")
    secondary_confirmation = ranked_confirmation.iloc[:10].copy()
    promoted = ranked_confirmation.iloc[10:].copy()
    source_discovery["split"] = "discovery"
    source_discovery["split_role"] = "original_discovery"
    source_discovery["secondary_split_hash"] = ""
    promoted["split"] = "discovery"
    promoted["split_role"] = "promoted_source_confirmation"
    secondary_confirmation["split"] = "confirmation"
    secondary_confirmation["split_role"] = "secondary_confirmation"
    result = pd.concat(
        [source_discovery, promoted, secondary_confirmation], ignore_index=True
    ).sort_values(["split", "seed"], kind="mergesort")
    counts = result.groupby("split")["request_id"].nunique().to_dict()
    if counts != {"confirmation": 10, "discovery": 20}:
        raise RuntimeError(f"Strict-dash secondary split is not 20/10: {counts}")
    if result["seed"].duplicated().any():
        raise RuntimeError("Strict-dash secondary panel contains duplicate seeds")
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selected = select(pd.read_csv(args.event_registry))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selected_indexed_numeric_trajectories.csv"
    selected.to_csv(selection_path, index=False)
    split_counts = {
        split: int(count)
        for split, count in selected.groupby("split").size().to_dict().items()
    }
    manifest = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_label": MODEL,
        "grammar_class": GRAMMAR,
        "marker_kind": MARKER,
        "rank_template": "rank_text equals the decimal strings 1 through 10",
        "selection_unit": "whole exact one-to-one N=10 trajectory",
        "split_counts": split_counts,
        "event_registry": str(args.event_registry.resolve()),
        "event_registry_sha256": _sha256(args.event_registry),
        "selection": str(selection_path.resolve()),
        "selection_sha256": _sha256(selection_path),
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
