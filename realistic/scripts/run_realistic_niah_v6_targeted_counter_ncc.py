#!/usr/bin/env python3
"""Capture V6 mode-native targeted-query counter-carrier NCC vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _model,
    _prepare_shards,
    _registered_rows,
    _runtime_manifest,
    _safe_stem,
    _spec,
)
from scripts.run_realistic_niah_v5_generated_suffix_state_bridge import (  # noqa: E402
    MODEL_CONTRACTS,
    _load_banks,
)
from realistic_niah_v5.targeted_counter_ncc import (  # noqa: E402
    capture_targeted_counter_ncc,
)
from realistic_niah_v6.kernel import MODE_TIMING_STRATA  # noqa: E402
from realistic_niah_v6.spec import V6Config  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_npz(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-config", type=Path, required=True)
    parser.add_argument("--v5-config", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONTRACTS), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument(
        "--seed-role", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--anchor-registry", type=Path, required=True)
    parser.add_argument("--targeted-registry", type=Path, required=True)
    parser.add_argument("--bank-plan", type=Path, required=True)
    parser.add_argument("--source-layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "targeted-counter-ncc"
    args.cohort = "one_to_one"
    args.row_panel = "trace_patch"
    args.limit = None

    config = V6Config.load(args.v5_config)
    expected_timing = MODE_TIMING_STRATA[config.prompt_mode]
    contract = MODEL_CONTRACTS[str(args.model)]
    if int(args.source_layer) != int(contract["source_layer"]):
        raise ValueError("V6 NCC source-layer contract changed")
    mechanism = _spec(args)
    registered = _registered_rows(args, mechanism)
    anchors = _read_jsonl(args.anchor_registry)
    targeted_rows = _read_jsonl(args.targeted_registry)
    if any("selection_rank" in row for row in anchors + targeted_rows):
        raise ValueError("V6 NCC registries forbid selection_rank")
    anchor_by_id = {str(row["request_id"]): row for row in anchors}
    targeted = {str(row["request_id"]): row for row in targeted_rows}
    if len(anchor_by_id) != len(anchors) or len(targeted) != len(targeted_rows):
        raise ValueError("V6 NCC registries contain duplicate requests")
    if any(
        str(row.get("mode_timing_stratum", expected_timing)) != expected_timing
        for row in anchors + targeted_rows
    ):
        raise ValueError("V6 NCC registry mixes enumeration timing strata")
    rows = [row for row in registered if str(row["request_id"]) in anchor_by_id]
    expected = 20 if args.seed_role == "development" else 10
    if len(rows) != expected or len({int(row["seed"]) for row in rows}) != expected:
        raise ValueError("V6 NCC requires one row per registered seed")
    if {str(row["request_id"]) for row in rows} - set(targeted):
        raise ValueError("V6 NCC targeted registry is incomplete")
    rows.sort(key=lambda row: int(row["seed"]))
    banks = _load_banks(args.bank_plan, model_label=str(args.model))

    plan = {
        "schema_version": "realistic_niah_v6_targeted_counter_ncc_plan_v1",
        "model_label": str(args.model),
        "prompt_mode": config.prompt_mode,
        "mode_timing_stratum": expected_timing,
        "seed_role": str(args.seed_role),
        "seeds": [int(row["seed"]) for row in rows],
        "seed_count": expected,
        "request_ids": [str(row["request_id"]) for row in rows],
        "anchor_registry_sha256": _sha256(args.anchor_registry),
        "targeted_registry_sha256": _sha256(args.targeted_registry),
        "bank_plan_sha256": _sha256(args.bank_plan),
        "source_layer": int(args.source_layer),
        "fit_policy": "discovery_clean_all_occurrences_same_mode_only",
        "confirmation_used_for_fit_or_layer_selection": False,
        "outcome_blind": True,
        "selection_rank_used": False,
    }
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing V6 NCC row plan changed")
    else:
        _atomic_json(plan_path, plan)

    started = time.perf_counter()
    model, tokenizer, adapter = _model(args)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="npz")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'v6_targeted_counter_ncc')}.npz"
        if args.resume and shard.exists():
            skipped += 1
            continue
        arrays, metadata = capture_targeted_counter_ncc(
            model,
            tokenizer,
            adapter,
            row,
            banks=banks,
            targeted_site=targeted[str(row["request_id"])],
            source_layer=int(args.source_layer),
            selected_bank_size=int(contract["bank_size"]),
        )
        metadata.update(
            {
                "v6_prompt_mode": config.prompt_mode,
                "v6_mode_timing_stratum": expected_timing,
                "mechanism_split": str(args.seed_role),
                "row_plan": str(plan_path.resolve()),
            }
        )
        _atomic_npz(shard, arrays, metadata)
        completed += 1
        print(f"[v6 targeted-counter-ncc] {index}/{len(rows)}", flush=True)
    _atomic_json(
        args.output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shards.glob("*.npz"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "seed_count": expected,
                "conditions": 5,
                "prompt_mode": config.prompt_mode,
                "mode_timing_stratum": expected_timing,
                "fit_policy": "discovery_clean_all_occurrences_same_mode_only",
                "outcome_blind": True,
                "selection_rank_used": False,
            },
        ),
    )


if __name__ == "__main__":
    main()
