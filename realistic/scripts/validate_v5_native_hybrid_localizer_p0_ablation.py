#!/usr/bin/env python3
"""Validate and summarize the frozen hybrid-localizer P0 ablation grid.

The validator deliberately reads a supervisor-written job ledger instead of
inferring experimental cells from directory names.  This makes selection site,
intervention onset, control family, split, and expected anchor count explicit
parts of the registered result.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CORRECT_OUTCOME = "correct_next_needle"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_in(output: Path) -> list[dict[str, Any]]:
    shard_dir = output / "shards"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Missing behavior shard directory: {shard_dir}")
    rows: list[dict[str, Any]] = []
    for path in sorted(shard_dir.glob("*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _failures(rows: Iterable[dict[str, Any]]) -> int:
    return sum(row.get("behavior_outcome") != CORRECT_OUTCOME for row in rows)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: observed {actual!r}, expected {expected!r}")


def _split_for_seed(seed: int, config: dict[str, Any]) -> str:
    if seed in set(map(int, config["causal_development_seeds"])):
        return "discovery"
    if seed in set(map(int, config["causal_confirmation_seeds"])):
        return "confirmation"
    raise AssertionError(f"Seed {seed} is outside both frozen causal splits")


def _validate_result_contract(
    rows: list[dict[str, Any]],
    *,
    grammar: str,
    selection_role: str,
    intervention_role: str,
    decode_steps: int,
) -> None:
    expected_decoupled = selection_role != intervention_role
    for row in rows:
        _assert_equal(
            row.get("routed_target_grammar_class"), grammar, "result grammar"
        )
        _assert_equal(
            row.get("head_selection_anchor_role"),
            selection_role,
            "head-selection anchor role",
        )
        _assert_equal(
            row.get("intervention_start_anchor_role"),
            intervention_role,
            "intervention start anchor role",
        )
        _assert_equal(
            int(row.get("head_ablation_decode_steps_requested")),
            decode_steps,
            "decode ablation schedule",
        )
        _assert_equal(
            bool(row.get("selection_intervention_site_decoupled")),
            expected_decoupled,
            "selection/intervention decoupling audit",
        )
        if float(row.get("head_ablation_selected_post_zero_max_abs", 0.0)) != 0.0:
            raise AssertionError("A selected pre-O head slice was not exactly zero")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty summary: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--sidecar-jobs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = _read_json(args.spec)
    config = _read_json(args.config)
    try:
        model_spec = spec["models"][args.model]
    except KeyError as error:
        raise ValueError(f"Model {args.model!r} is absent from the frozen spec") from error
    contract = spec["scientific_contract"]
    random_repeats = int(contract["random_control_repeats"])
    jobs = _read_jsonl(args.jobs)
    if not jobs:
        raise ValueError("Job ledger is empty")

    model_root = args.run_root / args.model
    registry = model_root / "registries" / "all" / "selected_anchor_registry.jsonl"
    registry_rows = _read_jsonl(registry)
    _assert_equal(len(registry_rows), int(model_spec["registry_rows"]), "registry rows")
    _assert_equal(
        _sha256(registry), model_spec["registry_sha256"], "registry SHA256"
    )
    grammar_names = list(model_spec["grammars"])
    registry_by_grammar: dict[str, list[dict[str, Any]]] = {}
    for grammar in grammar_names:
        path = model_root / "registries" / "by_grammar" / f"{grammar}.jsonl"
        rows = _read_jsonl(path)
        if any(row.get("target_grammar_class") != grammar for row in rows):
            raise AssertionError(f"Registry view {grammar} contains another grammar")
        registry_by_grammar[grammar] = rows
    _assert_equal(
        sum(map(len, registry_by_grammar.values())),
        len(registry_rows),
        "sum of mutually exclusive grammar registry views",
    )
    if Counter(row["request_id"] + "::" + row["anchor_equivalence_id"] for row in registry_rows) != Counter(
        row["request_id"] + "::" + row["anchor_equivalence_id"]
        for rows in registry_by_grammar.values()
        for row in rows
    ):
        raise AssertionError("Per-grammar registry views do not exactly partition registry")

    clean_output = model_root / "behaviors" / "clean_full"
    clean_rows = _rows_in(clean_output)
    _assert_equal(len(clean_rows), len(registry_rows), "clean rows")
    _assert_equal({row.get("condition") for row in clean_rows}, {"clean"}, "clean conditions")

    expected_cells = {
        (grammar, int(k))
        for grammar in grammar_names
        for k in model_spec["doses"]
    }
    observed_cells: set[tuple[str, int]] = set()
    summaries: list[dict[str, Any]] = []
    primary_k = int(model_spec["primary_bank_size"])
    for job in jobs:
        if job.get("model_label") != args.model:
            continue
        grammar = str(job["grammar"])
        k = int(job["bank_size"])
        cell = (grammar, k)
        if cell in observed_cells:
            raise AssertionError(f"Duplicate behavior job cell: {cell}")
        observed_cells.add(cell)
        expected_role = str(model_spec["grammars"][grammar])
        _assert_equal(job["selection_anchor_role"], expected_role, "job selection role")
        expected_split = "all" if k == primary_k else "confirmation"
        _assert_equal(job["evaluation_split"], expected_split, "job split")
        expected_anchors = sum(
            1
            for row in registry_by_grammar[grammar]
            if expected_split == "all"
            or _split_for_seed(int(row["seed"]), config) == expected_split
        )
        _assert_equal(
            int(job.get("expected_anchors", expected_anchors)),
            expected_anchors,
            f"ledger anchor count for {cell}",
        )
        if expected_anchors == 0:
            _assert_equal(
                job.get("execution_status"),
                "skipped_empty_split",
                f"empty cell status for {cell}",
            )
            summaries.append(
                {
                    "model_label": args.model,
                    "bank_size": k,
                    "grammar": grammar,
                    "selection_anchor_role": expected_role,
                    "intervention_start_anchor_role": contract[
                        "intervention_start_anchor_role"
                    ],
                    "selection_intervention_site_decoupled": expected_role
                    != contract["intervention_start_anchor_role"],
                    "split": expected_split,
                    "anchors": 0,
                    "exploratory": True,
                    "selected_failures": 0,
                    "selected_failure_rate": None,
                    "random_condition": str(job["random_condition"]),
                    "random_rows": 0,
                    "random_failures": 0,
                    "random_failure_rate": None,
                    "selected_minus_random_failure_rate": None,
                    "output": None,
                }
            )
            continue
        _assert_equal(
            job.get("execution_status"), "registered", f"cell status for {cell}"
        )
        output = Path(job["output"])
        rows = _rows_in(output)
        _validate_result_contract(
            rows,
            grammar=grammar,
            selection_role=expected_role,
            intervention_role=str(contract["intervention_start_anchor_role"]),
            decode_steps=int(contract["decode_head_ablation_steps"]),
        )
        selected = [row for row in rows if row.get("condition") == "selected_bank"]
        random_condition = str(job["random_condition"])
        random_rows = [row for row in rows if row.get("condition") == random_condition]
        unexpected = {
            row.get("condition") for row in rows
        } - {"selected_bank", random_condition}
        if unexpected:
            raise AssertionError(f"Unexpected conditions for {cell}: {unexpected}")
        _assert_equal(len(selected), expected_anchors, f"selected rows for {cell}")
        _assert_equal(
            len(random_rows),
            random_repeats * expected_anchors,
            f"random rows for {cell}",
        )
        selected_by_split = defaultdict(list)
        random_by_split = defaultdict(list)
        for row in selected:
            selected_by_split[str(row["split"])].append(row)
        for row in random_rows:
            random_by_split[str(row["split"])].append(row)
        for split in ("discovery", "confirmation"):
            selected_split = selected_by_split[split]
            random_split = random_by_split[split]
            if expected_split == "confirmation" and split == "discovery":
                continue
            summaries.append(
                {
                    "model_label": args.model,
                    "bank_size": k,
                    "grammar": grammar,
                    "selection_anchor_role": expected_role,
                    "intervention_start_anchor_role": contract[
                        "intervention_start_anchor_role"
                    ],
                    "selection_intervention_site_decoupled": expected_role
                    != contract["intervention_start_anchor_role"],
                    "split": split,
                    "anchors": len(selected_split),
                    "exploratory": split == "confirmation"
                    and len(selected_split) < 10,
                    "selected_failures": _failures(selected_split),
                    "selected_failure_rate": _rate(
                        _failures(selected_split), len(selected_split)
                    ),
                    "random_condition": random_condition,
                    "random_rows": len(random_split),
                    "random_failures": _failures(random_split),
                    "random_failure_rate": _rate(
                        _failures(random_split), len(random_split)
                    ),
                    "selected_minus_random_failure_rate": (
                        _rate(_failures(selected_split), len(selected_split))
                        - _rate(_failures(random_split), len(random_split))
                        if selected_split and random_split
                        else None
                    ),
                    "output": str(output),
                }
            )
    _assert_equal(observed_cells, expected_cells, "registered grammar-dose cells")

    confirmation_rows = [row for row in summaries if row["split"] == "confirmation"]
    dose_rows: list[dict[str, Any]] = []
    for k in map(int, model_spec["doses"]):
        k_rows = [row for row in confirmation_rows if int(row["bank_size"]) == k]
        for scope, scoped in (
            ("all_registered_grammars", k_rows),
            (
                "non_exploratory_grammars",
                [row for row in k_rows if not row["exploratory"]],
            ),
        ):
            selected_n = sum(int(row["anchors"]) for row in scoped)
            selected_failures = sum(int(row["selected_failures"]) for row in scoped)
            random_n = sum(int(row["random_rows"]) for row in scoped)
            random_failures = sum(int(row["random_failures"]) for row in scoped)
            selected_rate = _rate(selected_failures, selected_n)
            random_rate = _rate(random_failures, random_n)
            dose_rows.append(
                {
                    "model_label": args.model,
                    "bank_size": k,
                    "scope": scope,
                    "confirmation_anchors": selected_n,
                    "selected_failures": selected_failures,
                    "selected_failure_rate": selected_rate,
                    "random_rows": random_n,
                    "random_failures": random_failures,
                    "random_failure_rate": random_rate,
                    "selected_minus_random_failure_rate": (
                        selected_rate - random_rate
                        if selected_rate is not None and random_rate is not None
                        else None
                    ),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "grammar_split_results.csv", summaries)
    _write_csv(args.output / "overall_confirmation_dose_response.csv", dose_rows)
    completion = {
        "schema_version": "realistic_niah_v5_native_hybrid_localizer_p0_completion_v1",
        "status": "PASS",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_label": args.model,
        "spec_sha256": _sha256(args.spec),
        "jobs_sha256": _sha256(args.jobs),
        "registry_rows": len(registry_rows),
        "registry_sha256": _sha256(registry),
        "clean_rows": len(clean_rows),
        "clean_failures": _failures(clean_rows),
        "selection_metric": contract["selection_metric"],
        "selection_aggregation": contract["selection_aggregation"],
        "intervention_start_anchor_role": contract[
            "intervention_start_anchor_role"
        ],
        "decode_head_ablation_steps": contract["decode_head_ablation_steps"],
        "doses": list(map(int, model_spec["doses"])),
        "primary_bank_size": primary_k,
        "grammar_split_results": summaries,
        "overall": dose_rows,
    }
    if args.sidecar_jobs is not None:
        sidecar_jobs = _read_jsonl(args.sidecar_jobs)
        _assert_equal(len(sidecar_jobs), 1, "registered sidecar jobs")
        sidecar = sidecar_jobs[0]
        if sidecar["model_label"] == args.model:
            if sidecar.get("execution_status") == "skipped_empty_split":
                completion["registered_sidecar"] = {
                    "diagnostic": sidecar.get("diagnostic"),
                    "status": "not_estimable_empty_confirmation_split",
                    "confirmation_anchors": 0,
                }
                sidecar_rows = []
            else:
                sidecar_rows = _rows_in(Path(sidecar["output"]))
            if sidecar_rows:
                _validate_result_contract(
                    sidecar_rows,
                    grammar=str(sidecar["grammar"]),
                    selection_role=str(sidecar["selection_anchor_role"]),
                    intervention_role=str(sidecar["intervention_start_anchor_role"]),
                    decode_steps=int(sidecar["decode_head_ablation_steps"]),
                )
                selected = [
                    row
                    for row in sidecar_rows
                    if row["condition"] == "selected_bank"
                ]
                random_condition = str(sidecar["random_condition"])
                random_rows = [
                    row
                    for row in sidecar_rows
                    if row["condition"] == random_condition
                ]
                _assert_equal(
                    len(random_rows),
                    random_repeats * len(selected),
                    "sidecar random repeats",
                )
                _assert_equal(
                    len(selected),
                    int(sidecar["expected_anchors"]),
                    "sidecar confirmation anchors",
                )
                completion["registered_sidecar"] = {
                    "diagnostic": sidecar.get("diagnostic"),
                    "grammar": sidecar["grammar"],
                    "bank_size": int(sidecar["bank_size"]),
                    "selection_anchor_role": sidecar["selection_anchor_role"],
                    "intervention_start_anchor_role": sidecar[
                        "intervention_start_anchor_role"
                    ],
                    "confirmation_anchors": len(selected),
                    "selected_failures": _failures(selected),
                    "selected_failure_rate": _rate(
                        _failures(selected), len(selected)
                    ),
                    "random_condition": random_condition,
                    "random_rows": len(random_rows),
                    "random_failures": _failures(random_rows),
                    "random_failure_rate": _rate(
                        _failures(random_rows), len(random_rows)
                    ),
                }
    completion_path = args.output / "hybrid_localizer_p0_ablation_complete.json"
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "model_label": args.model,
        "registry_rows": len(registry_rows),
        "behavior_cells": len(observed_cells),
        "completion": str(completion_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
