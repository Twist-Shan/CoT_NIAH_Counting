#!/usr/bin/env python3
"""Seal the fresh Bullet-Gemma query-through-carrier replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(import_root))

from realistic_niah_v6.pipeline import sha256_file  # noqa: E402


SCHEMA_VERSION = "realistic_niah_v6_fresh_carrier_replication_complete_v1"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rows(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / "shards").glob("*.jsonl"))
    if len(files) != 10:
        raise ValueError(f"Expected ten fresh carrier shards, observed {len(files)}")
    return [
        json.loads(line)
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    lock = json.loads(args.cohort_lock.read_text(encoding="utf-8"))
    claims_path = args.analysis / "claim_gates.json"
    effects_path = args.analysis / "seed_effects.csv"
    manifest_path = args.trials / "manifest.json"
    adapter_path = args.trials / "v6_adapter_manifest.json"
    for path in (claims_path, effects_path, manifest_path, adapter_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    lock_core = {key: value for key, value in lock.items() if key != "cohort_lock_sha256"}
    if lock.get("cohort_lock_sha256") != _sha256_json(lock_core):
        raise ValueError("Fresh carrier cohort lock hash changed")
    if lock.get("status") != "FROZEN_BEFORE_CAUSAL_INTERVENTION_OUTCOMES":
        raise ValueError("Fresh cohort was not locked before causal outcomes")
    if lock.get("protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("Fresh cohort used a different V2 protocol")
    replication = protocol.get("bullet_gemma_query_through_carrier_replication", {})
    if (
        int(replication.get("fresh_trial_count", -1)) != 10
        or int(replication.get("frozen_selected_k", -1)) != 2
        or str(replication.get("head_ablation_scope")) != "query_through_carrier"
    ):
        raise ValueError("V2 carrier replication contract changed")

    rows = _rows(args.trials)
    if len(rows) != 70:
        raise ValueError(f"Fresh carrier factorial has {len(rows)} rather than 70 rows")
    seeds = sorted({int(row["seed"]) for row in rows})
    if seeds != list(map(int, lock["true_source_seeds"])):
        raise ValueError("Fresh carrier trials changed true source seeds")
    if {
        str(row.get("head_ablation_scope")) for row in rows
    } != {"query_through_carrier"}:
        raise ValueError("Fresh carrier trial scope changed")
    if any(bool(row.get("selection_rank_used", False)) for row in rows):
        raise ValueError("Fresh carrier trials used selection rank")
    if any(not bool(row.get("outcome_blind", False)) for row in rows):
        raise ValueError("Fresh carrier trial lost its outcome-blind row plan")
    per_seed = {}
    for row in rows:
        per_seed.setdefault(int(row["seed"]), []).append(str(row["condition"]))
    expected_conditions = {
        "clean",
        "selected_mask",
        "random_mask_r1",
        "random_mask_r2",
        "random_mask_r3",
        "selected_mask_clean_carrier_restore",
        "selected_mask_matched_position_state_control",
    }
    if any(set(values) != expected_conditions for values in per_seed.values()):
        raise ValueError("Fresh carrier seven-arm factorial changed")
    if (
        int(manifest.get("seed_count", -1)) != 10
        or int(manifest.get("selected_bank_size", -1)) != 2
        or str(manifest.get("head_ablation_scope")) != "query_through_carrier"
    ):
        raise ValueError("Fresh carrier run manifest changed")
    slot_identity = adapter.get("specialized_slot_identity", {})
    if (
        slot_identity.get("status") != "PASS_FIXED_SLOT_TRUE_SOURCE_IDENTITY"
        or int(slot_identity.get("analysis_slot_count", -1)) != 10
    ):
        raise ValueError("Fresh carrier slot/source identity audit failed")
    if (
        claims.get("phase") != "confirmation"
        or int(claims.get("seed_count", -1)) != 10
        or not bool(claims.get("original_query_local_null_retained"))
    ):
        raise ValueError("Fresh carrier analysis contract changed")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "FRESH_CAUSAL_OUTCOME_REPLICATION_COMPLETE",
        "model_label": "Gemma4-E4B",
        "prompt_mode": "enumeration_bullet",
        "replication_kind": (
            "fresh_prospective_causal_outcomes_with_earlier_discovery_frozen_bank"
        ),
        "true_source_seeds": seeds,
        "seed_count": len(seeds),
        "selected_k": 2,
        "source_layer": 16,
        "head_ablation_scope": "query_through_carrier",
        "strong_interval_gate_pass": bool(
            claims.get("targeted_counter_write_strong_gate_pass")
        ),
        "directional_gate_pass": bool(
            claims.get("targeted_counter_write_directional_pass")
        ),
        "original_query_local_null_retained": True,
        "seed_selection_used_intervention_outcomes": False,
        "frozen_bank_changed": False,
        "seed_aliasing": False,
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "qualification": (
            "The causal outcomes and source requests are fresh, while the K=2 "
            "head bank was frozen from the earlier discovery experiment."
        ),
        "artifacts": {
            "protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol),
            },
            "cohort_lock": {
                "path": str(args.cohort_lock.resolve()),
                "sha256": sha256_file(args.cohort_lock),
                "internal_lock_sha256": str(lock["cohort_lock_sha256"]),
            },
            "trials_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "adapter_manifest": {
                "path": str(adapter_path.resolve()),
                "sha256": sha256_file(adapter_path),
            },
            "claim_gates": {
                "path": str(claims_path.resolve()),
                "sha256": sha256_file(claims_path),
            },
            "seed_effects": {
                "path": str(effects_path.resolve()),
                "sha256": sha256_file(effects_path),
            },
        },
    }
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "strong_interval_gate_pass": payload[
                    "strong_interval_gate_pass"
                ],
                "true_source_seeds": seeds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
