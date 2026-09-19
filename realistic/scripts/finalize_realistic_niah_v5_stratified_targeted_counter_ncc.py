#!/usr/bin/env python3
"""Seal one model's two timing-specific NCC analyses into a consistent summary."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any


TIMINGS = ("rank_after_city", "rank_before_city")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    branches: dict[str, dict[str, Any]] = {}
    branch_summary: dict[str, dict[str, Any]] = {}
    for timing in TIMINGS:
        path = args.output_root / timing / "analysis" / "claim_gates.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("schema_version")
            != "realistic_niah_v5_stratified_targeted_counter_ncc_analysis_v2"
            or result.get("status") != "PASS"
            or str(result.get("model_label")) != args.model
            or str(result.get("timing_branch")) != timing
        ):
            raise ValueError(f"Analysis mismatch or stale schema: {path}")
        primary = result["primary_endpoint_result"]
        validity = primary["readout_validity"]
        branches[timing] = result
        branch_summary[timing] = {
            "primary_endpoint": result["primary_endpoint"],
            "selected_layer": primary["selected_layer"],
            "readout_validity_pass": bool(validity["pass"]),
            "effect_status": primary["ncc_effect_status"],
            "qualified_directional_specific_support": bool(validity["pass"])
            and bool(primary["selected_mask_changes_ncc_directionally"])
            and bool(primary["selected_mask_more_damaging_than_random"]),
        }

    value = {
        "schema_version": (
            "realistic_niah_v5_stratified_targeted_counter_ncc_complete_v2"
        ),
        "status": "PASS",
        "model_label": args.model,
        "branches": branches,
        "branch_summary": branch_summary,
        "raw_margins_pooled": False,
        "outcome_blind": True,
        "selection_rank_used": False,
        "directional_specific_support_requires_valid_clean_readout": True,
        "readout_validity_gate_registered_after_initial_contrast_inspection": True,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    output = args.output or args.output_root / "stratified_ncc_complete.json"
    _atomic_json(output, value)
    print(
        json.dumps(
            {"status": "PASS", "model": args.model, "output": str(output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
