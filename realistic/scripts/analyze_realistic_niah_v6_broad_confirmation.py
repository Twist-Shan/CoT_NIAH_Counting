#!/usr/bin/env python3
"""Summarize a single discovery-frozen broad-head K on confirmation seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v6.kernel import install_v6_kernel_adapters  # noqa: E402
from realistic_niah_v6.pipeline import sha256_file  # noqa: E402
from realistic_niah_v6.spec import V6Config  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402
from scripts.run_realistic_niah_v6_count_stream import (  # noqa: E402
    _coherent_broad_source_seeds,
)


SCHEMA_VERSION = "realistic_niah_v6_broad_head_confirmation_v1"


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _trial_rows(path: Path) -> list[dict[str, object]]:
    files = sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No broad confirmation shards under {path}")
    rows = []
    for source in files:
        rows.extend(
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--mechanism-config", type=Path, required=True)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument(
        "--source-group", choices=("trace_items", "prompt_records"), required=True
    )
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--discovery-decision", type=Path, required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = V6Config.load(args.v6_config)
    freeze = validate_confirmation_freeze(
        args.confirmation_freeze,
        prompt_mode=config.prompt_mode,
        model_label=str(args.model),
    )
    decision = json.loads(args.discovery_decision.read_text(encoding="utf-8"))
    if decision.get("status") != "frozen_for_confirmation":
        raise ValueError("Broad confirmation requires a discovery-frozen positive K")
    if decision.get("confirmation_outcomes_used") is not False:
        raise ValueError("Broad K decision accessed confirmation outcomes")
    if decision.get("model_label") != args.model:
        raise ValueError("Broad discovery decision has the wrong model")
    if decision.get("source_group") != args.source_group:
        raise ValueError("Broad discovery decision has the wrong source group")
    selected_k = int(decision["selected_bank_size"])

    plan = pd.read_csv(args.frozen_plan)
    plan = plan.loc[
        plan["model_label"].astype(str).eq(str(args.model))
        & pd.to_numeric(plan["bank_size"], errors="raise").eq(selected_k)
    ].copy()
    if plan.empty or set(plan["k_selection_status"].astype(str)) != {
        "frozen_for_confirmation"
    }:
        raise ValueError("Broad plan is not confirmation-locked")
    if not plan["confirmation_locked"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    ).all():
        raise ValueError("Broad plan confirmation lock changed")
    if set(plan["k_selection_decision_sha256"].astype(str)) != {
        str(decision["decision_sha256"])
    }:
        raise ValueError("Broad plan and discovery decision hashes disagree")

    install_v6_kernel_adapters()
    from realistic_niah_v5.count_stream import (
        NativeCountMechanismSpec,
        select_answer_broad_bank_size,
    )

    mechanism = NativeCountMechanismSpec.load(args.mechanism_config)
    if not mechanism.formal_inference_eligible:
        raise ValueError("Broad confirmation mechanism is not frozen")
    trials = pd.DataFrame(_trial_rows(args.trials))
    source_seeds, source_by_slot = _coherent_broad_source_seeds(
        trials,
        mechanism=mechanism,
        model_label=str(args.model),
        source_group=str(args.source_group),
        phase="confirmation",
    )
    curve, seed_effects, _unused_confirmation_decision = (
        select_answer_broad_bank_size(
            trials,
            model_label=str(args.model),
            source_group=str(args.source_group),
            expected_seeds=source_seeds,
            expected_bank_sizes=(selected_k,),
            expected_requests_per_seed=int(mechanism.broad_panel_counts_per_seed),
            expected_random_controls=int(mechanism.random_controls),
            boundary_extension_bank_size=selected_k,
            bootstrap_samples=int(mechanism.bootstrap_samples),
            random_seed=int(args.random_seed),
        )
    )
    if len(curve) != 1 or int(curve.iloc[0]["bank_size"]) != selected_k:
        raise RuntimeError("Broad confirmation analysis changed the frozen K")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "CONFIRMATION_EVALUATED_FROZEN_K",
        "model_label": str(args.model),
        "prompt_mode": config.prompt_mode,
        "source_group": str(args.source_group),
        "selected_k": selected_k,
        "confirmation_curve": curve.to_dict(orient="records")[0],
        "seed_effects": seed_effects.to_dict(orient="records"),
        "analysis_slot_to_true_source_seed": source_by_slot,
        "true_source_seeds": list(source_seeds),
        "discovery_decision": str(args.discovery_decision.resolve()),
        "discovery_decision_sha256": sha256_file(args.discovery_decision),
        "frozen_plan": str(args.frozen_plan.resolve()),
        "frozen_plan_sha256": sha256_file(args.frozen_plan),
        "trials": str(args.trials.resolve()),
        "trials_manifest_sha256": sha256_file(args.trials / "manifest.json"),
        "confirmation_freeze": str(args.confirmation_freeze.resolve()),
        "confirmation_freeze_sha256": sha256_file(args.confirmation_freeze),
        "freeze_contract_sha256": str(freeze["freeze_sha256"]),
        "confirmation_used_for_selection": False,
        "bank_size_reselected": False,
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
