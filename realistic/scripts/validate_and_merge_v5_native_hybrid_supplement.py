#!/usr/bin/env python3
"""Validate new P2-bank/P0-onset cells and merge them with frozen P0 results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CORRECT = "correct_next_needle"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(output: Path) -> list[dict[str, Any]]:
    shards = output / "shards"
    if not shards.is_dir():
        raise FileNotFoundError(shards)
    return [row for path in sorted(shards.glob("*.jsonl")) for row in _jsonl(path)]


def _failures(rows: list[dict[str, Any]]) -> int:
    return sum(row.get("behavior_outcome") != CORRECT for row in rows)


def _rate(a: int, b: int) -> float | None:
    return a / b if b else None


def _assert(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: observed={actual!r}, expected={expected!r}")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _split(seed: int, config: dict[str, Any]) -> str:
    if seed in set(map(int, config["causal_development_seeds"])):
        return "discovery"
    if seed in set(map(int, config["causal_confirmation_seeds"])):
        return "confirmation"
    raise AssertionError(f"Unknown causal seed {seed}")


def _validate_contract(
    rows: list[dict[str, Any]], grammar: str, selection_role: str
) -> None:
    for row in rows:
        _assert(row.get("routed_target_grammar_class"), grammar, "grammar")
        _assert(
            row.get("head_selection_anchor_role"), selection_role, "selection role"
        )
        _assert(
            row.get("intervention_start_anchor_role"),
            "p0_item_end",
            "intervention onset",
        )
        _assert(
            int(row.get("head_ablation_decode_steps_requested")),
            -1,
            "persistent decode schedule",
        )
        _assert(
            bool(row.get("selection_intervention_site_decoupled")),
            selection_role != "p0_item_end",
            "selection/intervention decoupling",
        )
        if float(row.get("head_ablation_selected_post_zero_max_abs", 0.0)) != 0.0:
            raise AssertionError("Selected pre-O head slice was not exactly zero")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--sidecar-jobs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = _json(args.spec)
    config = _json(args.config)
    model_spec = spec["models"][args.model]
    supplement_spec = model_spec["supplement_execution"]
    run_grammars = set(supplement_spec["run_grammars"])
    reuse_grammars = set(supplement_spec["reuse_grammars"])
    repeats = int(spec["scientific_contract"]["random_control_repeats"])
    primary_k = int(model_spec["primary_bank_size"])
    doses = list(map(int, model_spec["doses"]))

    prior_full_path = args.code_root / supplement_spec["prior_full_completion"]
    prior_dose_path = args.code_root / supplement_spec["prior_dose_completion"]
    _assert(
        _sha(prior_full_path),
        supplement_spec["prior_full_completion_sha256"],
        "prior full completion SHA",
    )
    _assert(
        _sha(prior_dose_path),
        supplement_spec["prior_dose_completion_sha256"],
        "prior dose completion SHA",
    )
    prior_full = _json(prior_full_path)
    prior_dose = _json(prior_dose_path)
    _assert(prior_full["status"], "PASS", "prior full status")
    _assert(prior_dose["status"], "PASS", "prior dose status")
    _assert(prior_full["registry_sha256"], model_spec["registry_sha256"], "prior registry")

    model_root = args.run_root / args.model
    registry_path = model_root / "registries" / "all" / "selected_anchor_registry.jsonl"
    registry = _jsonl(registry_path)
    _assert(len(registry), int(model_spec["registry_rows"]), "registry rows")
    _assert(_sha(registry_path), model_spec["registry_sha256"], "registry SHA")
    registry_by_grammar = {}
    for grammar in model_spec["grammars"]:
        path = model_root / "registries" / "by_grammar" / f"{grammar}.jsonl"
        registry_by_grammar[grammar] = _jsonl(path)

    jobs = [row for row in _jsonl(args.jobs) if row["model_label"] == args.model]
    expected_cells = {(grammar, k) for grammar in run_grammars for k in doses}
    observed_cells = {(row["grammar"], int(row["bank_size"])) for row in jobs}
    _assert(observed_cells, expected_cells, "supplement job cells")
    cell_summaries: list[dict[str, Any]] = []
    for job in jobs:
        grammar = str(job["grammar"])
        k = int(job["bank_size"])
        selection_role = str(job["selection_anchor_role"])
        _assert(selection_role, "post_marker", "supplement selection role")
        split = "all" if k == primary_k else "confirmation"
        expected_anchors = sum(
            split == "all" or _split(int(row["seed"]), config) == split
            for row in registry_by_grammar[grammar]
        )
        _assert(int(job["expected_anchors"]), expected_anchors, "job anchors")
        if expected_anchors == 0:
            _assert(job["execution_status"], "skipped_empty_split", "empty job")
            cell_summaries.append(
                {
                    "bank_size": k,
                    "grammar": grammar,
                    "split": split,
                    "anchors": 0,
                    "selected_failures": 0,
                    "random_condition": str(job["random_condition"]),
                    "random_rows": 0,
                    "random_failures": 0,
                    "exploratory": True,
                    "selection_anchor_role": selection_role,
                    "provenance": "new_p2_ranked_bank_p0_persistent_ablation",
                }
            )
            continue
        output_rows = _rows(Path(job["output"]))
        _validate_contract(output_rows, grammar, selection_role)
        selected = [row for row in output_rows if row["condition"] == "selected_bank"]
        random_condition = str(job["random_condition"])
        random_rows = [
            row for row in output_rows if row["condition"] == random_condition
        ]
        _assert(len(selected), expected_anchors, "selected rows")
        _assert(len(random_rows), repeats * expected_anchors, "random rows")
        for result_split in ("discovery", "confirmation"):
            selected_split = [row for row in selected if row["split"] == result_split]
            random_split = [row for row in random_rows if row["split"] == result_split]
            if split == "confirmation" and result_split == "discovery":
                continue
            cell_summaries.append(
                {
                    "bank_size": k,
                    "grammar": grammar,
                    "split": result_split,
                    "anchors": len(selected_split),
                    "selected_failures": _failures(selected_split),
                    "random_condition": random_condition,
                    "random_rows": len(random_split),
                    "random_failures": _failures(random_split),
                    "exploratory": result_split == "confirmation"
                    and len(selected_split) < 10,
                    "selection_anchor_role": selection_role,
                    "provenance": "new_p2_ranked_bank_p0_persistent_ablation",
                }
            )

    new_confirmation = {
        (int(row["bank_size"]), row["grammar"]): row
        for row in cell_summaries
        if row["split"] == "confirmation"
    }
    merged_dose_rows = []
    for old in prior_dose["rows"]:
        key = (int(old["bank_size"]), old["grammar"])
        if old["grammar"] in reuse_grammars:
            row = dict(old)
            row.update(
                {
                    "selection_anchor_role": "p0_item_end",
                    "provenance": "reused_frozen_p0_grammar_specific_result",
                }
            )
        else:
            new = new_confirmation[key]
            row = {
                "bank_size": key[0],
                "confirmation_anchors": int(new["anchors"]),
                "exploratory": bool(new["exploratory"]),
                "grammar": key[1],
                "random_condition": new["random_condition"],
                "random_failures": int(new["random_failures"]),
                "selected_failures": int(new["selected_failures"]),
                "selection_anchor_role": "post_marker",
                "provenance": new["provenance"],
            }
        merged_dose_rows.append(row)

    overall = []
    for k in doses:
        rows_k = [row for row in merged_dose_rows if int(row["bank_size"]) == k]
        for scope, scoped in (
            ("all_registered_grammars", rows_k),
            ("non_exploratory_grammars", [row for row in rows_k if not row["exploratory"]]),
        ):
            anchors = sum(int(row["confirmation_anchors"]) for row in scoped)
            selected_failures = sum(int(row["selected_failures"]) for row in scoped)
            random_rows = repeats * anchors
            random_failures = sum(int(row["random_failures"]) for row in scoped)
            selected_rate = _rate(selected_failures, anchors)
            random_rate = _rate(random_failures, random_rows)
            overall.append(
                {
                    "bank_size": k,
                    "scope": scope,
                    "confirmation_anchors": anchors,
                    "selected_failure_rate": selected_rate,
                    "random_failure_rate": random_rate,
                    "selected_minus_random_failure_rate": (
                        selected_rate - random_rate
                        if selected_rate is not None and random_rate is not None
                        else None
                    ),
                }
            )

    prior_full_by_grammar = {row["grammar"]: row for row in prior_full["grammars"]}
    new_primary = defaultdict(dict)
    for row in cell_summaries:
        if int(row["bank_size"]) == primary_k:
            new_primary[row["grammar"]][row["split"]] = row
    merged_full_grammars = []
    for grammar in model_spec["grammars"]:
        if grammar in reuse_grammars:
            row = dict(prior_full_by_grammar[grammar])
            row.update(
                {
                    "selection_anchor_role": "p0_item_end",
                    "provenance": "reused_frozen_p0_grammar_specific_result",
                }
            )
        else:
            by_split = {}
            for split in ("discovery", "confirmation"):
                new = new_primary[grammar][split]
                by_split[split] = {
                    "selected": int(new["anchors"]),
                    "selected_failures": int(new["selected_failures"]),
                    "random": int(new["random_rows"]),
                    "random_condition": new["random_condition"],
                    "random_failures": int(new["random_failures"]),
                }
            row = {
                "grammar": grammar,
                "anchors": sum(value["selected"] for value in by_split.values()),
                "selected_failures": sum(
                    value["selected_failures"] for value in by_split.values()
                ),
                "random_failures": sum(
                    value["random_failures"] for value in by_split.values()
                ),
                "by_split": by_split,
                "selection_anchor_role": "post_marker",
                "provenance": "new_p2_ranked_bank_p0_persistent_ablation",
            }
        merged_full_grammars.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    supplement_payload = {
        "schema_version": "realistic_niah_v5_native_hybrid_supplement_v1",
        "status": "PASS",
        "model_label": args.model,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_grammars": sorted(run_grammars),
        "reuse_grammars": sorted(reuse_grammars),
        "registry_rows": len(registry),
        "registry_sha256": _sha(registry_path),
        "jobs_sha256": _sha(args.jobs),
        "new_cells": cell_summaries,
    }
    if args.sidecar_jobs is not None:
        sidecars = [
            row for row in _jsonl(args.sidecar_jobs) if row["model_label"] == args.model
        ]
        if sidecars:
            _assert(len(sidecars), 1, "sidecar jobs")
            sidecar = sidecars[0]
            sidecar_rows = _rows(Path(sidecar["output"]))
            _validate_contract(sidecar_rows, sidecar["grammar"], "p0_item_end")
            selected = [row for row in sidecar_rows if row["condition"] == "selected_bank"]
            random_condition = sidecar["random_condition"]
            random_rows = [row for row in sidecar_rows if row["condition"] == random_condition]
            _assert(len(random_rows), repeats * len(selected), "sidecar controls")
            supplement_payload["registered_sidecar"] = {
                "grammar": sidecar["grammar"],
                "bank_size": int(sidecar["bank_size"]),
                "confirmation_anchors": len(selected),
                "selected_failures": _failures(selected),
                "random_condition": random_condition,
                "random_rows": len(random_rows),
                "random_failures": _failures(random_rows),
            }

    merged_full = {
        "schema_version": "realistic_niah_v5_native_hybrid_full_v1",
        "status": "PASS",
        "model_label": args.model,
        "bank_size": primary_k,
        "intervention_start_anchor_role": "p0_item_end",
        "persistent_ablation": True,
        "registry_rows": len(registry),
        "registry_sha256": _sha(registry_path),
        "clean_rows": prior_full["clean_rows"],
        "clean_failures": prior_full["clean_failures"],
        "clean_provenance": "reused_frozen_p0_full_panel_completion",
        "prior_full_completion_sha256": _sha(prior_full_path),
        "grammars": merged_full_grammars,
    }
    merged_dose = {
        "schema_version": "realistic_niah_v5_native_hybrid_dose_v1",
        "status": "PASS",
        "model_label": args.model,
        "doses": doses,
        "intervention_start_anchor_role": "p0_item_end",
        "persistent_ablation": True,
        "prior_dose_completion_sha256": _sha(prior_dose_path),
        "rows": merged_dose_rows,
        "overall": overall,
        "reporting_policy": prior_dose["reporting_policy"],
    }
    _write(args.output / "supplement_complete.json", supplement_payload)
    _write(args.output / "hybrid_full_panel_complete.json", merged_full)
    _write(args.output / "hybrid_dose_grid_complete.json", merged_dose)
    _write_csv(args.output / "supplement_cells.csv", cell_summaries)
    _write_csv(args.output / "hybrid_overall_confirmation_dose.csv", overall)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_label": args.model,
                "new_job_cells": len(expected_cells),
                "reused_grammar_cells": len(reuse_grammars) * len(doses),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
