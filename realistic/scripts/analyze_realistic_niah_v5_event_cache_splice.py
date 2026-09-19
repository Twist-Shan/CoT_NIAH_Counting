#!/usr/bin/env python3
"""Analyze the frozen bidirectional marker-ledger KV-cache experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.causal import sign_flip_pvalue


SCHEMA_VERSION = "event_cache_marker_ledger_analysis_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effect_summary(values: Sequence[float]) -> dict[str, Any]:
    active = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if active.ndim != 1 or not len(active) or not np.isfinite(active).all():
        raise ValueError("Effect summary requires finite one-dimensional values")
    return {
        "n_seeds": int(len(active)),
        "mean": float(active.mean()),
        "median": float(np.median(active)),
        "minimum": float(active.min()),
        "maximum": float(active.max()),
        "positive_seed_count": int(np.count_nonzero(active > 0)),
        "two_sided_sign_flip_pvalue": float(sign_flip_pvalue(active)),
        "seed_values": [float(value) for value in active],
    }


def _index_interventions(
    rows: Sequence[Mapping[str, Any]], *, read_layer: int
) -> dict[tuple[int, str, str, str], Mapping[str, Any]]:
    index: dict[tuple[int, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("condition")) != "cache_splice" or int(
            row.get("read_layer", -1)
        ) != int(read_layer):
            continue
        components = "+".join(str(value) for value in row["components"])
        direction = f'{row["donor_variant"]}_to_{row["receiver_variant"]}'
        key = (int(row["seed"]), direction, str(row["region"]), components)
        if key in index:
            raise ValueError(f"Duplicate intervention cell: {key}")
        index[key] = row
    return index


def analyze_batch(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    seeds = tuple(int(value) for value in expected_seeds)
    observed = sorted({int(row["seed"]) for row in rows})
    if observed != sorted(seeds):
        raise ValueError(f"Seed contract mismatch: expected {seeds}, observed {observed}")
    primary = plan["primary_intervention"]
    layer = int(plan["primary_read_layer"])
    directions = tuple(str(value) for value in primary["directions"])
    region = str(primary["region"])
    components = "+".join(str(value) for value in primary["components"])
    index = _index_interventions(rows, read_layer=layer)

    by_region: dict[str, dict[int, float]] = {}
    required_regions = (
        region,
        "closing",
        "payload",
        "event",
        "b5",
        "preceding_event_width",
    )
    for active_region in required_regions:
        by_region[active_region] = {}
        for seed in seeds:
            values = []
            for direction in directions:
                row = index.get((seed, direction, active_region, components))
                if row is None:
                    raise ValueError(
                        f"Missing seed/direction/region cell: {seed, direction, active_region}"
                    )
                value = row.get("donor_axis_progress")
                if value is None:
                    raise ValueError("Primary donor-axis progress is undefined")
                values.append(float(value))
            by_region[active_region][seed] = fmean(values)

    primary_values = [by_region[region][seed] for seed in seeds]
    marker_minus_closing = [
        by_region[region][seed] - by_region["closing"][seed] for seed in seeds
    ]
    marker_minus_payload = [
        by_region[region][seed] - by_region["payload"][seed] for seed in seeds
    ]
    primary_summary = _effect_summary(primary_values)
    closing_summary = _effect_summary(marker_minus_closing)
    payload_summary = _effect_summary(marker_minus_payload)

    alpha = float(primary["alpha"])
    minimum = float(primary["minimum_effect_size"])
    primary_pass = (
        float(primary_summary["mean"]) >= minimum
        and float(primary_summary["two_sided_sign_flip_pvalue"]) < alpha
    )
    specificity_plan = {
        str(value["name"]): value for value in plan["specificity_contrasts"]
    }

    def specificity_pass(name: str, summary: Mapping[str, Any]) -> bool:
        spec = specificity_plan[name]
        return float(summary["mean"]) >= float(spec["minimum_effect_size"]) and float(
            summary["two_sided_sign_flip_pvalue"]
        ) < float(spec["alpha"])

    identity_rows = [
        row
        for row in rows
        if str(row.get("condition")) == "cache_splice"
        and int(row.get("read_layer", -1)) == layer
        and str(row.get("region"))
        in set(plan["implementation_controls"]["identity_regions"])
        and "+".join(str(value) for value in row["components"]) == components
    ]
    identity_pass = len(identity_rows) == len(seeds) * len(directions) * 2 and all(
        int(row["splice_audit"]["changed_elements"]) == 0
        and float(row["donor_axis_progress"]) == 0.0
        for row in identity_rows
    )
    full_event_values = [by_region["event"][seed] for seed in seeds]
    full_event_pass = all(abs(value - 1.0) <= 1e-12 for value in full_event_values)

    return {
        "seeds": list(seeds),
        "primary_marker_axis_progress": primary_summary,
        "marker_minus_closing": closing_summary,
        "marker_minus_payload": payload_summary,
        "region_bidirectional_seed_means": {
            active_region: {
                str(seed): float(value) for seed, value in values.items()
            }
            for active_region, values in by_region.items()
        },
        "criteria": {
            "primary_effect_pass": bool(primary_pass),
            "marker_minus_closing_pass": bool(
                specificity_pass("marker_minus_closing", closing_summary)
            ),
            "marker_minus_payload_pass": bool(
                specificity_pass("marker_minus_payload", payload_summary)
            ),
            "identity_controls_pass": bool(identity_pass),
            "full_event_positive_control_pass": bool(full_event_pass),
        },
        "marker_ledger_supported": bool(
            primary_pass
            and specificity_pass("marker_minus_closing", closing_summary)
            and specificity_pass("marker_minus_payload", payload_summary)
            and identity_pass
            and full_event_pass
        ),
    }


def analyze(
    discovery_path: Path,
    confirmation_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    discovery_rows = _read_jsonl(discovery_path)
    confirmation_rows = _read_jsonl(confirmation_path)
    discovery = analyze_batch(
        discovery_rows, plan, expected_seeds=plan["discovery_seeds"]
    )
    confirmation = analyze_batch(
        confirmation_rows, plan, expected_seeds=plan["confirmation_seeds"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "discovery_path": str(discovery_path.resolve()),
        "discovery_sha256": _sha256(discovery_path),
        "confirmation_path": str(confirmation_path.resolve()),
        "confirmation_sha256": _sha256(confirmation_path),
        "discovery": discovery,
        "confirmation": confirmation,
        "confirmatory_conclusion": (
            "marker_keyed_event_ledger_supported"
            if bool(confirmation["marker_ledger_supported"])
            else "marker_keyed_event_ledger_not_confirmed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.discovery, args.confirmation, args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
