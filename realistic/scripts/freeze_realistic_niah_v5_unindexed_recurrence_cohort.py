#!/usr/bin/env python3
"""Freeze N=10 count-label-free traces for the recurrence experiment.

Eligibility is outcome-blind.  A row must pass the causal-prefix audit in
``audit_no_count_enumeration_trace``; neither the final answer nor any causal
effect is read during cohort construction.  Seeds are ordered from 1234 and
missing rows are filled by later eligible seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_no_count_enumeration_trace,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _eligible_rows(
    rows: Iterable[dict[str, Any]], *, model: str, min_seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_seed: dict[int, dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        if str(row.get("model_label")) != model:
            status = "wrong_model"
        elif int(row.get("seed", -1)) < min_seed:
            status = "before_min_seed"
        elif int(row.get("gold_count", -1)) != 10:
            status = "not_N10"
        else:
            audit = audit_no_count_enumeration_trace(row)
            if not bool(audit["eligible"]):
                status = "count_label_or_structure_gate_fail"
            else:
                seed = int(row["seed"])
                if seed in by_seed:
                    raise ValueError(f"Duplicate eligible {model} seed {seed}")
                status = "eligible_N10_unindexed"
                by_seed[seed] = {**row, "unindexed_recurrence_audit": audit}
        status_counts[status] = status_counts.get(status, 0) + 1
    return [by_seed[seed] for seed in sorted(by_seed)], status_counts


def _registry_row(
    row: dict[str, Any], *, rank: int, discovery_count: int
) -> dict[str, Any]:
    parser = row["trace_parse"]["parser"]
    raw = str(row.get("raw_output_text", ""))
    return {
        "model_label": str(row["model_label"]),
        "rank": rank,
        "split": "discovery" if rank <= discovery_count else "confirmation",
        "seed": int(row["seed"]),
        "gold_count": int(row["gold_count"]),
        "request_id": str(row["request_id"]),
        "stimulus_id": str(row.get("stimulus_id", "")),
        "marker_kind": str(parser.get("marker_kind", "")),
        "item_count": int(parser.get("item_count", 0)),
        "trace_one_to_one": bool(parser.get("trace_one_to_one", False)),
        "prompt_conditioned_unnumbered": "NATURAL_UNNUMBERED_A"
        in str(row.get("request_id", "")),
        "raw_output_sha256": _sha256_text(raw),
        "selection_used_final_answer": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-input", type=Path, nargs="+", required=True)
    parser.add_argument("--gemma-input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-seed", type=int, default=1234)
    parser.add_argument("--target-count", type=int, default=30)
    parser.add_argument("--discovery-count", type=int, default=20)
    args = parser.parse_args()
    if not 0 < args.discovery_count < args.target_count:
        raise ValueError("Discovery count must be between zero and target count")

    inputs = {
        "Qwen3-8B": args.qwen_input,
        "Gemma4-E4B": args.gemma_input,
    }
    manifests: dict[str, Any] = {}
    for model in MODELS:
        input_paths = tuple(inputs[model])
        all_rows = [row for path in input_paths for row in _read_jsonl(path)]
        eligible, status_counts = _eligible_rows(
            all_rows, model=model, min_seed=int(args.min_seed)
        )
        selected = eligible[: int(args.target_count)]
        registry = [
            _registry_row(
                row, rank=index, discovery_count=int(args.discovery_count)
            )
            for index, row in enumerate(selected, start=1)
        ]
        model_dir = args.output / model
        _atomic_text(model_dir / "selected_rows.jsonl", _jsonl_text(selected))
        _write_csv(model_dir / "seed_registry.csv", registry)
        missing = max(0, int(args.target_count) - len(selected))
        manifest = {
            "schema_version": "realistic_niah_v5_unindexed_recurrence_cohort_v1",
            "status": "PASS" if missing == 0 else "INCOMPLETE",
            "model_label": model,
            "source_paths": [str(path) for path in input_paths],
            "source_sha256": {
                str(path): _sha256_file(path) for path in input_paths
            },
            "eligibility": (
                "N=10, complete one-to-one item trace, and no numbered item label, "
                "ordinal record label, running subtotal, or previously stated total "
                "in any item-state causal prefix"
            ),
            "min_seed": int(args.min_seed),
            "target_count": int(args.target_count),
            "discovery_count": int(args.discovery_count),
            "confirmation_count": int(args.target_count - args.discovery_count),
            "selected_count": len(selected),
            "missing_count": missing,
            "selected_seeds": [int(row["seed"]) for row in selected],
            "discovery_seeds": [
                int(row["seed"])
                for row in selected[: int(args.discovery_count)]
            ],
            "confirmation_seeds": [
                int(row["seed"])
                for row in selected[
                    int(args.discovery_count) : int(args.target_count)
                ]
            ],
            "supplement_search_start_seed": (
                max(int(row["seed"]) for row in all_rows) + 1 if missing else None
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "selection_used_final_answer": False,
            "claim_scope": "format-conditioned unnumbered reasoning",
        }
        _atomic_text(
            model_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        manifests[model] = manifest

    combined = {
        "schema_version": "realistic_niah_v5_unindexed_recurrence_cohort_v1",
        "models": manifests,
        "all_models_complete": all(
            value["status"] == "PASS" for value in manifests.values()
        ),
        "outcome_blind": True,
    }
    _atomic_text(
        args.output / "manifest.json",
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(combined, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
