#!/usr/bin/env python3
"""Analyze a coherent V6 native-loop panel with true source seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v6.pipeline import read_jsonl, sha256_file  # noqa: E402
from realistic_niah_v6.spec import MODEL_LABELS, V6Config  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402
from scripts import analyze_realistic_niah_v5_native_loop as legacy  # noqa: E402


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _coherent_contract(
    path: Path,
    *,
    config: V6Config,
    model_label: str,
    phase: str,
) -> dict[str, Any]:
    split = phase
    slots = tuple(
        map(
            int,
            config.discovery_seeds
            if split == "discovery"
            else config.confirmation_seeds,
        )
    )
    required_counts = tuple(int(count) for count in config.counts if int(count) >= 2)
    rows = [
        dict(row)
        for row in read_jsonl(path)
        if str(row.get("model_label")) == model_label
        and str(row.get("split")) == split
        and int(row.get("gold_count", -1)) in set(required_counts)
    ]
    expected_cells = {(slot, count) for slot in slots for count in required_counts}
    observed_cells = [
        (int(row["analysis_slot_seed"]), int(row["gold_count"])) for row in rows
    ]
    if set(observed_cells) != expected_cells or len(observed_cells) != len(
        expected_cells
    ):
        raise ValueError("Native-loop registry changed its fixed slot/count panel")
    source_by_slot: dict[int, int] = {}
    for slot in slots:
        sources = {
            int(row["source_seed"])
            for row in rows
            if int(row["analysis_slot_seed"]) == slot
        }
        if len(sources) != 1:
            raise ValueError(f"Native-loop slot {slot} mixes source seeds")
        source_by_slot[slot] = next(iter(sources))
    sources = tuple(source_by_slot[slot] for slot in slots)
    if len(set(sources)) != len(sources):
        raise ValueError("Two native-loop slots share one source seed")
    return {
        "phase": phase,
        "registry": str(path.resolve()),
        "registry_sha256": sha256_file(path),
        "analysis_slots": list(slots),
        "required_counts": list(required_counts),
        "true_source_seeds": list(sources),
        "analysis_slot_to_true_source_seed": {
            str(slot): source_by_slot[slot] for slot in slots
        },
        "replacement_trajectory_count": sum(
            slot != source_by_slot[slot] for slot in slots
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--cohort-registry", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_LABELS, required=True)
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--confirmation-freeze", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = V6Config.load(args.v6_config)
    if args.phase == "confirmation":
        if args.confirmation_freeze is None:
            raise ValueError("Native-loop confirmation analysis requires a freeze")
        validate_confirmation_freeze(
            args.confirmation_freeze,
            prompt_mode=config.prompt_mode,
            model_label=args.model,
        )
    elif args.confirmation_freeze is not None:
        raise ValueError("Discovery analysis must not open a confirmation freeze")
    contract = _coherent_contract(
        args.cohort_registry,
        config=config,
        model_label=args.model,
        phase=args.phase,
    )
    frame = legacy._read_trials(args.trials)
    observed = set(frame["seed"].astype(int))
    expected = set(map(int, contract["true_source_seeds"]))
    if observed != expected:
        raise ValueError(
            "Native-loop trials changed the true-source panel: "
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )
    if args.phase == "discovery":
        legacy.COUNT_STREAM_DISCOVERY_SEEDS = tuple(contract["true_source_seeds"])
    else:
        legacy.COUNT_STREAM_CONFIRMATION_SEEDS = tuple(
            contract["true_source_seeds"]
        )
    estimands, seed_effects, gates = legacy.analyze(
        frame,
        phase=args.phase,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    legacy._atomic_csv(args.output / "estimands.csv", estimands)
    legacy._atomic_csv(args.output / "seed_effects.csv", seed_effects)
    legacy._atomic_json(args.output / "claim_gates.json", gates)
    manifest = {
        "schema_version": "realistic_niah_v6_native_loop_analysis_adapter_v1",
        "status": "PASS",
        "model_label": args.model,
        "prompt_mode": config.prompt_mode,
        "phase": args.phase,
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
        "membership_adapter_scope": (
            "legacy expected-seed validation constant only; numerical estimands, "
            "pairing, bootstrap, gates, and aggregation are unchanged"
        ),
        "contract": contract,
        "trial_roots": [str(path.resolve()) for path in args.trials],
        "trial_manifest_sha256": {
            str(path.resolve()): sha256_file(path / "manifest.json")
            for path in args.trials
        },
        "claim_gates_sha256": sha256_file(args.output / "claim_gates.json"),
        "confirmation_freeze_sha256": (
            sha256_file(args.confirmation_freeze)
            if args.confirmation_freeze is not None
            else None
        ),
    }
    _atomic_json(args.output / "v6_analysis_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase": args.phase,
                "seed_count": len(expected),
                "native_loop_pass": bool(gates["native_loop_pass"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
