#!/usr/bin/env python3
"""Strictly audit answer and running states in the transfer captures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MODELS = {"Qwen3-8B": 36, "Gemma4-E4B": 42}
DOMAINS = ("flower", "animal")
SEEDS = tuple(range(1254, 1264))
COUNTS = tuple(range(1, 11))
SCHEMA_VERSION = "realistic_niah_domain_transfer_capture_audit_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_index(
    index_path: str | Path,
    *,
    mode: str,
    model_label: str,
    require_registered_panel: bool = True,
) -> dict[str, Any]:
    if mode not in {"non_thinking", "native_thinking"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if model_label not in MODELS:
        raise ValueError(f"Unsupported model: {model_label}")
    path = Path(index_path).resolve()
    rows = _read_jsonl(path)
    expected_cells = {
        (domain, seed, count)
        for domain in DOMAINS
        for seed in SEEDS
        for count in COUNTS
    }
    observed_cells = {
        (str(row["entity_domain"]), int(row["seed"]), int(row["gold_count"]))
        for row in rows
    }
    if require_registered_panel and (
        len(rows) != len(expected_cells) or observed_cells != expected_cells
    ):
        raise ValueError(
            f"{model_label}/{mode} is not the registered 2 x 10 x 10 panel: "
            f"rows={len(rows)}, cells={len(observed_cells)}, "
            f"missing={sorted(expected_cells-observed_cells)[:5]}"
        )
    expected_running_kind = "running_index" if mode == "non_thinking" else "item_end"
    expected_answer_kind = "answer_query" if mode == "non_thinking" else "answer_query_v3"
    total_running = 0
    total_answers = 0
    state_bytes = 0
    trace_categories: dict[str, int] = {}
    marker_kinds: dict[str, int] = {}
    rescued_rows = 0
    exact_rows = 0
    exact_denominator = 0
    for row in rows:
        if str(row["model_label"]) != model_label:
            raise ValueError(f"Another model appears in {path}")
        manifest_path = path.parent / str(row["manifest_path"])
        states_path = path.parent / str(row["states_path"])
        if not manifest_path.is_file() or not states_path.is_file():
            raise FileNotFoundError(f"Missing shard files for {row['stimulus_id']}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sites = list(manifest.get("site_rows", []))
        running = [site for site in sites if site.get("site_kind") == expected_running_kind]
        answers = [site for site in sites if site.get("site_kind") == expected_answer_kind]
        if len(answers) != 1:
            raise ValueError(
                f"{row['stimulus_id']} has {len(answers)} {expected_answer_kind} sites"
            )
        if int(row.get("running_site_count", -1)) != len(running):
            raise ValueError(f"Stale running_site_count for {row['stimulus_id']}")
        if int(row.get("answer_site_count", -1)) != len(answers):
            raise ValueError(f"Stale answer_site_count for {row['stimulus_id']}")
        if mode == "non_thinking" and len(running) != int(row["gold_count"]):
            raise ValueError(f"Non-thinking running path is not exactly 1..N: {row['stimulus_id']}")
        if mode == "native_thinking" and len(running) != int(row["trace_item_count"]):
            raise ValueError(f"Native running sites/parser item count disagree: {row['stimulus_id']}")
        with np.load(states_path, allow_pickle=False) as archive:
            layers = archive["layer_indices"].astype(int)
            states = np.asarray(archive["site_states"])
            if not np.array_equal(layers, np.arange(MODELS[model_label])):
                raise ValueError(f"Layer registry mismatch in {states_path}")
            if states.ndim != 3 or states.shape[:2] != (
                len(sites),
                MODELS[model_label],
            ):
                raise ValueError(f"State shape/site mismatch in {states_path}: {states.shape}")
            if not np.isfinite(states).all():
                raise ValueError(f"Non-finite running/answer state in {states_path}")
        total_running += len(running)
        total_answers += len(answers)
        state_bytes += states_path.stat().st_size
        if row.get("generation_rescue"):
            rescued_rows += 1
        if row.get("exact_count") is not None:
            exact_denominator += 1
            exact_rows += int(bool(row["exact_count"]))
        for column, accumulator in (
            ("trace_category", trace_categories),
            ("marker_kind", marker_kinds),
        ):
            value = row.get(column)
            if value is not None:
                accumulator[str(value)] = accumulator.get(str(value), 0) + 1
    site_index_path = path.parent / "site_index.jsonl"
    site_manifest_path = path.parent / "site_index_manifest.json"
    if not site_index_path.is_file() or not site_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing reusable site catalog beside {path}: "
            f"{site_index_path.name}, {site_manifest_path.name}"
        )
    flat_sites = _read_jsonl(site_index_path)
    site_summary = json.loads(site_manifest_path.read_text(encoding="utf-8"))
    flat_running = [
        row for row in flat_sites if row.get("site_kind") == expected_running_kind
    ]
    flat_answers = [
        row for row in flat_sites if row.get("site_kind") == expected_answer_kind
    ]
    state_addresses = {
        (str(row["states_path"]), int(row["state_axis"])) for row in flat_sites
    }
    if len(state_addresses) != len(flat_sites):
        raise ValueError(f"Duplicate NPZ state address in {site_index_path}")
    if len(flat_running) != total_running or len(flat_answers) != total_answers:
        raise ValueError(
            f"Flat site catalog count mismatch in {site_index_path}: "
            f"running={len(flat_running)}/{total_running}, "
            f"answer={len(flat_answers)}/{total_answers}"
        )
    if any(
        str(row["model_label"]) != model_label
        or str(row["mode"]).replace("-", "_") != mode
        for row in flat_sites
    ):
        raise ValueError(f"Flat site catalog model/mode mismatch in {site_index_path}")
    if int(site_summary.get("site_rows", -1)) != len(flat_sites):
        raise ValueError(f"Stale site_index_manifest beside {site_index_path}")
    return {
        "model_label": model_label,
        "mode": mode,
        "index_path": str(path),
        "rows": len(rows),
        "unique_cells": len(observed_cells),
        "registered_panel_complete": observed_cells == expected_cells,
        "answer_states": total_answers,
        "running_states": total_running,
        "all_layers": MODELS[model_label],
        "all_states_finite": True,
        "state_archive_bytes": state_bytes,
        "generation_rescue_rows": rescued_rows,
        "exact_count_rows": exact_rows,
        "exact_count_denominator": exact_denominator,
        "trace_category_counts": trace_categories,
        "marker_kind_counts": marker_kinds,
        "site_index_path": str(site_index_path),
        "site_index_rows": len(flat_sites),
        "site_index_audited": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonthinking-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audits = []
    for model in MODELS:
        audits.append(
            audit_index(
                args.nonthinking_root / model / "capture_index.jsonl",
                mode="non_thinking",
                model_label=model,
            )
        )
        audits.append(
            audit_index(
                args.native_root / model / "capture_index.jsonl",
                mode="native_thinking",
                model_label=model,
            )
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
