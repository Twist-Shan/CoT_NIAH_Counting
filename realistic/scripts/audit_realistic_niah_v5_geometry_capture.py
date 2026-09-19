#!/usr/bin/env python3
"""Fail closed unless a V5 geometry capture covers the registered 300 panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "realistic_niah_v5_geometry_capture_audit_v1"
COUNTS = tuple(range(1, 11))
DISCOVERY_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
RUNNING_SITES = (
    "pre_city",
    "city_end",
    "city_unit_end",
    "item_end",
    "post_boundary",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def expected_panel() -> set[tuple[str, int, int]]:
    return {
        (
            "discovery" if seed in DISCOVERY_SEEDS else "confirmation",
            seed,
            count,
        )
        for seed in (*DISCOVERY_SEEDS, *CONFIRMATION_SEEDS)
        for count in COUNTS
    }


def audit_capture(index_path: Path) -> dict[str, Any]:
    rows = read_jsonl(index_path)
    observed = {
        (str(row["split"]), int(row["seed"]), int(row["gold_count"]))
        for row in rows
    }
    expected = expected_panel()
    if len(rows) != len(observed):
        raise ValueError(
            f"Capture has duplicate panel keys: rows={len(rows)}, unique={len(observed)}"
        )
    if observed != expected:
        raise ValueError(
            "Capture is not the registered 10-count x 30-seed panel: "
            f"missing={sorted(expected-observed)[:10]}, "
            f"extra={sorted(observed-expected)[:10]}"
        )

    model_labels = {str(row.get("model_label")) for row in rows}
    if len(model_labels) != 1:
        raise ValueError(f"Expected one model label, found {sorted(model_labels)}")
    layer_grids: set[tuple[int, ...]] = set()
    total_items = 0
    trace_categories: Counter[str] = Counter()
    marker_kinds: Counter[str] = Counter()
    split_items: Counter[str] = Counter()
    for row in rows:
        manifest_path = index_path.parent / str(row["manifest_path"])
        states_path = index_path.parent / str(row["states_path"])
        if not manifest_path.is_file() or not states_path.is_file():
            raise FileNotFoundError(
                f"Missing shard files for {row.get('request_id')}: "
                f"{manifest_path}, {states_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        site_rows = list(manifest.get("site_rows", []))
        item_count = int(row["trace_item_count"])
        if item_count < 1:
            raise ValueError(
                f"{row.get('request_id')}: running trace has no parsed items"
            )
        for field in ("request_id", "stimulus_id", "model_label", "split"):
            if str(manifest.get(field)) != str(row.get(field)):
                raise ValueError(
                    f"{row.get('request_id')}: index/manifest {field} mismatch"
                )
        for field in ("seed", "gold_count"):
            if int(manifest.get(field)) != int(row.get(field)):
                raise ValueError(
                    f"{row.get('request_id')}: index/manifest {field} mismatch"
                )
        selected_site_kinds = set(map(str, manifest.get("selected_site_kinds", [])))
        expected_site_kinds = {*RUNNING_SITES, "answer_query_v3"}
        if selected_site_kinds != expected_site_kinds:
            raise ValueError(
                f"{row.get('request_id')}: selected sites are "
                f"{sorted(selected_site_kinds)}; expected {sorted(expected_site_kinds)}"
            )
        observed_site_kinds = Counter(str(site.get("site_kind")) for site in site_rows)
        if set(observed_site_kinds) != expected_site_kinds:
            raise ValueError(
                f"{row.get('request_id')}: materialized sites are "
                f"{sorted(observed_site_kinds)}; expected {sorted(expected_site_kinds)}"
            )
        expected_occurrences = list(range(1, item_count + 1))
        for site_kind in RUNNING_SITES:
            occurrences = sorted(
                int(site["occurrence"])
                for site in site_rows
                if str(site.get("site_kind")) == site_kind
                and site.get("occurrence") is not None
            )
            if occurrences != expected_occurrences:
                raise ValueError(
                    f"{row.get('request_id')}: {site_kind} occurrences are "
                    f"{occurrences}; expected {expected_occurrences}"
                )
        answer_sites = [
            site
            for site in site_rows
            if str(site.get("site_kind")) == "answer_query_v3"
        ]
        if len(answer_sites) != 1:
            raise ValueError(
                f"{row.get('request_id')}: expected one answer_query_v3, "
                f"found {len(answer_sites)}"
            )
        if answer_sites[0].get("occurrence") is not None:
            raise ValueError(
                f"{row.get('request_id')}: answer_query_v3 must not have an occurrence"
            )
        if len(site_rows) != len(RUNNING_SITES) * item_count + 1:
            raise ValueError(
                f"{row.get('request_id')}: expected "
                f"{len(RUNNING_SITES) * item_count + 1} site rows, got {len(site_rows)}"
            )
        for site in site_rows:
            if not bool(site.get("alignment_eligible")):
                raise ValueError(
                    f"{row.get('request_id')}: ineligible aligned site "
                    f"{site.get('site_id')}"
                )
            if site.get("endpoint_token") is None:
                raise ValueError(
                    f"{row.get('request_id')}: site {site.get('site_id')} lacks endpoint"
                )
        layers = tuple(map(int, manifest.get("layers", [])))
        if not layers:
            raise ValueError(f"{row.get('request_id')}: empty layer grid")
        with np.load(states_path, allow_pickle=False) as archive:
            required_arrays = {"layer_indices", "site_states"}
            if not required_arrays.issubset(archive.files):
                raise ValueError(
                    f"{row.get('request_id')}: states archive lacks "
                    f"{sorted(required_arrays - set(archive.files))}"
                )
            stored_layers = tuple(map(int, np.asarray(archive["layer_indices"])))
            site_states_shape = tuple(np.asarray(archive["site_states"]).shape)
        if stored_layers != layers:
            raise ValueError(
                f"{row.get('request_id')}: NPZ/manifest layer grid mismatch"
            )
        expected_shape_prefix = (len(site_rows), len(layers))
        if (
            len(site_states_shape) != 3
            or site_states_shape[:2] != expected_shape_prefix
            or site_states_shape[2] < 1
        ):
            raise ValueError(
                f"{row.get('request_id')}: site_states shape {site_states_shape}; "
                f"expected ({len(site_rows)}, {len(layers)}, hidden_size)"
            )
        if tuple(map(int, manifest.get("site_states_shape", []))) != site_states_shape:
            raise ValueError(
                f"{row.get('request_id')}: NPZ/manifest site_states shape mismatch"
            )
        layer_grids.add(layers)
        total_items += item_count
        split_items[str(row["split"])] += item_count
        trace_categories[str(row.get("trace_category"))] += 1
        marker_kinds[str(row.get("marker_kind"))] += 1
    if len(layer_grids) != 1:
        raise ValueError(f"Capture uses multiple layer grids: {len(layer_grids)}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "capture_index": str(index_path.resolve()),
        "model_label": next(iter(model_labels)),
        "trajectory_rows": len(rows),
        "counts": list(COUNTS),
        "discovery_trajectories": 200,
        "confirmation_trajectories": 100,
        "running_state_rows": total_items,
        "running_state_rows_by_split": dict(sorted(split_items.items())),
        "answer_query_v3_rows": len(rows),
        "layers": list(next(iter(layer_grids))),
        "trace_category_trajectory_counts": dict(sorted(trace_categories.items())),
        "marker_kind_trajectory_counts": dict(sorted(marker_kinds.items())),
        "running_label_rule": (
            "each trajectory contributes exactly its parser-observed 1..M; "
            "gold N and final Total never pad missing states"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_capture(args.capture_index.resolve())
    if args.output is not None:
        atomic_json(args.output.resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
