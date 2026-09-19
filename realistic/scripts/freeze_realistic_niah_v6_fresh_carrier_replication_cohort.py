#!/usr/bin/env python3
"""Freeze an outcome-blind fresh N=10 cohort for carrier replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(import_root))

from realistic_niah_v6.pipeline import (  # noqa: E402
    sha256_file,
    validate_generation_contracts,
)
from realistic_niah_v6.replacement import (  # noqa: E402
    SELECTED_CELL_SCHEMA_VERSION,
    audit_generation_eligibility,
)
from realistic_niah_v6.spec import V6Config  # noqa: E402


SCHEMA_VERSION = "realistic_niah_v6_fresh_carrier_replication_cohort_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL row {line_number} in {path}")
            rows.append(value)
    return rows


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL row {line_number} in {path}")
            yield value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--base-confirmation-registry", type=Path, required=True)
    parser.add_argument(
        "--exclude-registry", type=Path, action="append", default=[]
    )
    parser.add_argument("--model", default="Gemma4-E4B")
    parser.add_argument("--gold-count", type=int, default=10)
    parser.add_argument("--quota", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.v6_config.resolve()
    protocol_path = args.protocol.resolve()
    generations_path = args.generations.resolve()
    base_path = args.base_confirmation_registry.resolve()
    config = V6Config.load(config_path)
    if config.prompt_mode != "enumeration_bullet":
        raise ValueError("Fresh carrier replication is Bullet-only")
    if str(args.model) != "Gemma4-E4B" or str(args.model) not in config.model_labels:
        raise ValueError("Fresh carrier replication is frozen to Gemma4-E4B")
    if int(args.gold_count) != 10 or int(args.quota) != 10:
        raise ValueError("Fresh carrier replication is frozen to ten N=10 trials")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V2_INTERVENTION_OUTCOMES":
        raise ValueError("V2 follow-up protocol was not frozen before outcomes")
    replication = protocol.get("bullet_gemma_query_through_carrier_replication", {})
    slots = tuple(map(int, replication.get("analysis_slots", ())))
    if slots != tuple(map(int, config.confirmation_seeds)):
        raise ValueError("Protocol analysis slots changed")

    base_rows = _read_jsonl(base_path)
    expected_slots = {
        (int(count), int(seed))
        for count in config.counts
        for seed in config.confirmation_seeds
    }
    observed_slots = {
        (int(row["gold_count"]), int(row["analysis_slot_seed"]))
        for row in base_rows
    }
    if observed_slots != expected_slots or len(base_rows) != len(expected_slots):
        raise ValueError("Base confirmation registry is not a complete V6 panel")
    if any(
        row.get("schema_version") != SELECTED_CELL_SCHEMA_VERSION
        or str(row.get("model_label")) != str(args.model)
        or str(row.get("prompt_mode")) != config.prompt_mode
        or str(row.get("split")) != "confirmation"
        for row in base_rows
    ):
        raise ValueError("Base confirmation registry identity changed")

    exclusion_paths = sorted(
        {path.resolve() for path in [base_path, *args.exclude_registry]}
    )
    excluded_seeds = set(map(int, config.all_seeds))
    excluded_requests: set[str] = set()
    exclusion_audit: list[dict[str, Any]] = []
    for path in exclusion_paths:
        rows = _read_jsonl(path)
        seeds = {int(row["source_seed"]) for row in rows if "source_seed" in row}
        requests = {
            str(row["source_request_id"])
            for row in rows
            if row.get("source_request_id") is not None
        }
        excluded_seeds.update(seeds)
        excluded_requests.update(requests)
        exclusion_audit.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "row_count": len(rows),
                "source_seed_count": len(seeds),
                "source_request_count": len(requests),
            }
        )

    candidates: dict[int, dict[str, Any]] = {}
    duplicate_candidate_seeds: set[int] = set()
    for row in _iter_jsonl(generations_path):
        if str(row.get("model_label", row.get("model", ""))) != str(args.model):
            continue
        if str(row.get("prompt_mode", "")) != config.prompt_mode:
            continue
        if int(row.get("gold_count", -1)) != int(args.gold_count):
            continue
        seed = int(row.get("seed", -1))
        request_id = str(row.get("request_id", ""))
        if seed in excluded_seeds or request_id in excluded_requests:
            continue
        if seed in candidates:
            duplicate_candidate_seeds.add(seed)
            continue
        candidates[seed] = dict(row)
    if duplicate_candidate_seeds:
        raise ValueError(
            f"Fresh candidate seeds are duplicated: {sorted(duplicate_candidate_seeds)}"
        )

    selected: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    selected_source_rows: list[dict[str, Any]] = []
    for candidate_rank, seed in enumerate(sorted(candidates), start=1):
        row = candidates[seed]
        eligibility = audit_generation_eligibility(row)
        attempts.append(
            {
                "candidate_rank": candidate_rank,
                "seed": seed,
                "request_id": str(row["request_id"]),
                "eligible": bool(eligibility["eligible"]),
                "failure_reasons": list(eligibility["failure_reasons"]),
                "intervention_outcomes_read": False,
            }
        )
        if not eligibility["eligible"]:
            continue
        slot = slots[len(selected)]
        selected.append(
            {
                "schema_version": SELECTED_CELL_SCHEMA_VERSION,
                "model_label": str(args.model),
                "prompt_mode": config.prompt_mode,
                "split": "confirmation",
                "gold_count": int(args.gold_count),
                "analysis_slot_seed": int(slot),
                "source_seed": seed,
                "source_request_id": str(row["request_id"]),
                "source_stimulus_id": str(row["stimulus_id"]),
                "replacement_applied": True,
                "original_failure_reasons": [
                    "prospective_fresh_carrier_replication_cohort"
                ],
                "replacement_candidate_rank": candidate_rank,
                "eligibility_rule": (
                    "fresh_v6_parse.strict_causal_eligible_is_true"
                ),
                "intervention_outcomes_read": False,
                "fresh_replication_cohort": True,
            }
        )
        selected_source_rows.append(row)
        if len(selected) == int(args.quota):
            break
    if len(selected) != int(args.quota):
        raise RuntimeError(
            f"Only {len(selected)} unused strict candidates exist for quota {args.quota}"
        )
    selected_seeds = [int(row["source_seed"]) for row in selected]
    if selected_seeds != sorted(selected_seeds) or len(set(selected_seeds)) != len(selected):
        raise RuntimeError("Fresh selected seeds are not unique ascending sources")
    validate_generation_contracts(
        selected_source_rows,
        config,
        model_label=str(args.model),
        config_sha256=sha256_file(config_path),
    )

    retained = [row for row in base_rows if int(row["gold_count"]) != 10]
    registry = sorted(
        [*retained, *selected],
        key=lambda row: (int(row["gold_count"]), int(row["analysis_slot_seed"])),
    )
    registry_slots = [
        (int(row["gold_count"]), int(row["analysis_slot_seed"]))
        for row in registry
    ]
    if set(registry_slots) != expected_slots or len(registry_slots) != len(expected_slots):
        raise RuntimeError("Fresh registry no longer fills every V6 slot once")
    requests = [str(row["source_request_id"]) for row in registry]
    if len(requests) != len(set(requests)):
        raise RuntimeError("Fresh registry reuses one generation request")

    args.output.mkdir(parents=True, exist_ok=True)
    registry_path = args.output / "selected_cells.jsonl"
    attempts_path = args.output / "selection_attempts.jsonl"
    _atomic_jsonl(registry_path, registry)
    _atomic_jsonl(attempts_path, attempts)
    lock_core = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_CAUSAL_INTERVENTION_OUTCOMES",
        "model_label": str(args.model),
        "prompt_mode": config.prompt_mode,
        "gold_count": int(args.gold_count),
        "trial_count": len(selected),
        "analysis_slots": list(slots),
        "true_source_seeds": selected_seeds,
        "slot_to_true_source_seed": {
            str(row["analysis_slot_seed"]): int(row["source_seed"])
            for row in selected
        },
        "candidate_order": "ascending_true_source_seed",
        "eligibility_rule": "fresh_v6_parse.strict_causal_eligible_is_true",
        "selection_inputs": [
            "model_label",
            "prompt_mode",
            "gold_count",
            "true_source_seed",
            "fresh_v6_strict_parser_eligibility",
            "prior_registry_membership"
        ],
        "forbidden_selection_inputs": list(
            replication.get("forbidden_selection_inputs", ())
        ),
        "intervention_outcomes_read": False,
        "hidden_states_read": False,
        "attention_scores_read": False,
        "head_ranks_used_for_seed_selection": False,
        "seed_aliasing": False,
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "v6_config": str(config_path),
        "v6_config_sha256": sha256_file(config_path),
        "generations": str(generations_path),
        "generations_sha256": sha256_file(generations_path),
        "base_confirmation_registry": str(base_path),
        "base_confirmation_registry_sha256": sha256_file(base_path),
        "excluded_registries": exclusion_audit,
        "excluded_source_seed_count": len(excluded_seeds),
        "candidate_attempt_count": len(attempts),
        "selected_cells_sha256": sha256_file(registry_path),
        "selection_attempts_sha256": sha256_file(attempts_path),
    }
    lock = {**lock_core, "cohort_lock_sha256": _sha256_json(lock_core)}
    lock_path = args.output / "cohort_lock.json"
    _atomic_json(lock_path, lock)
    print(
        json.dumps(
            {
                "status": lock["status"],
                "true_source_seeds": selected_seeds,
                "cohort_lock_sha256": lock["cohort_lock_sha256"],
                "selected_cells_sha256": lock["selected_cells_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
