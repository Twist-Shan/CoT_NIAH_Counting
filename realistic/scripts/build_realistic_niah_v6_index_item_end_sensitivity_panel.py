#!/usr/bin/env python3
"""Build the outcome-blind item-end panel for the V6 index sensitivity.

The primary V6 panel builder intentionally binds enumeration-index to the
registered ``post_marker`` anchor.  This wrapper reuses all of its validation
and replacement logic while applying an isolated ``p0_item_end`` contract.
It does not modify the primary builder or any primary artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

try:
    import build_realistic_niah_v6_final_transition_panel as primary
except ModuleNotFoundError:  # package import used by the regression tests
    from scripts import build_realistic_niah_v6_final_transition_panel as primary


CONTRACT_SCHEMA = "realistic_niah_v6_index_item_end_anchor_sensitivity_v1"
CONTRACT_STATUS = "FROZEN_EXPLORATORY_BEFORE_NEW_ARM_OUTCOMES"
AMENDMENT_SCHEMA = (
    "realistic_niah_v6_index_item_end_generation_container_amendment_v1"
)
AMENDMENT_STATUS = (
    "FROZEN_RECOVERY_BEFORE_GEMMA_SENSITIVITY_BEHAVIOR_OUTCOMES"
)


def _validate_contract(path: Path, model: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("Unexpected item-end sensitivity contract schema")
    if value.get("status") != CONTRACT_STATUS:
        raise ValueError("Item-end sensitivity contract is not frozen")
    if value.get("prompt_mode") != "enumeration_index":
        raise ValueError("Item-end sensitivity contract changed prompt mode")
    if value.get("scientific_scope") != "exploratory_discovery_only":
        raise ValueError("Item-end sensitivity must remain discovery-only")
    if value.get("confirmation_authorized") is not False:
        raise ValueError("Item-end sensitivity cannot open confirmation")
    if model not in value.get("models", {}):
        raise ValueError(f"Model {model!r} is absent from the frozen contract")
    cells = set(value.get("fixed_design", {}).get("cells", ()))
    if cells != {
        "p2bank_at_p2",
        "p2bank_at_p0",
        "p0bank_at_p2",
        "p0bank_at_p0",
    }:
        raise ValueError("Frozen 2x2 cell registry changed")
    return value


def _item_end_mode_contract(config) -> tuple[str, str]:
    if config.prompt_mode != "enumeration_index":
        raise ValueError("This exploratory panel is index-only")
    return "p0_item_end", "structural_item_end_sensitivity"


def _canonical_rows_sha256(rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for row in sorted(rows, key=lambda value: str(value["request_id"]))
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_generation_container(
    *,
    generations_path: Path,
    cohort_registry_path: Path,
    model: str,
    model_contract: dict,
    amendment_path: Path | None,
) -> dict:
    expected_hash = str(model_contract["frozen_generations_sha256"])
    observed_hash = primary.sha256_file(generations_path)
    if observed_hash == expected_hash:
        return {
            "status": "PASS_EXACT_FROZEN_CONTAINER",
            "expected_generations_sha256": expected_hash,
            "observed_generations_sha256": observed_hash,
            "amendment": None,
        }
    if amendment_path is None:
        raise ValueError("Generation artifact hash differs from the frozen sensitivity")

    amendment_path = amendment_path.resolve()
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("schema_version") != AMENDMENT_SCHEMA
        or amendment.get("status") != AMENDMENT_STATUS
        or amendment.get("model_label") != model
        or amendment.get("prompt_mode") != "enumeration_index"
        or amendment.get("scientific_scope") != "artifact_container_identity_only"
        or amendment.get("original_frozen_generations_sha256") != expected_hash
        or amendment.get("observed_aggregate_generations_sha256") != observed_hash
        or amendment.get("frozen_cohort_registry_sha256")
        != primary.sha256_file(cohort_registry_path)
    ):
        raise ValueError("Generation-container amendment does not match frozen inputs")
    firewall = amendment.get("outcome_firewall", {})
    if (
        firewall.get("gemma_sensitivity_behavior_outcomes_existed_before_amendment")
        is not False
        or firewall.get("behavior_outcomes_read_for_amendment") is not False
        or firewall.get("head_scores_read_for_amendment") is not False
        or firewall.get("source_write_values_read_for_amendment") is not False
        or any(
            firewall.get(field) is not False
            for field in (
                "fixed_k_changed",
                "analysis_slot_seeds_changed",
                "registered_cells_changed",
                "intervention_scope_changed",
                "scientific_gate_changed",
            )
        )
    ):
        raise ValueError("Generation-container amendment violates the outcome firewall")

    generation_rows = primary.read_jsonl(generations_path)
    if len(generation_rows) != int(amendment["observed_aggregate_row_count"]):
        raise ValueError("Amended generation container row count changed")
    generation_by_id: dict[str, dict] = {}
    for row in generation_rows:
        request_id = str(row["request_id"])
        if request_id in generation_by_id:
            raise ValueError("Amended generation container has duplicate request IDs")
        generation_by_id[request_id] = row

    registry_rows = primary.read_jsonl(cohort_registry_path)
    if len(registry_rows) != int(amendment["frozen_cohort_row_count"]):
        raise ValueError("Frozen discovery cohort row count changed")
    selected_rows: list[dict] = []
    freeze_timestamp = datetime.fromisoformat(
        str(amendment["original_sensitivity_frozen_at_utc"]).replace("Z", "+00:00")
    ).timestamp()
    shard_root = generations_path.parent / "shards"
    for registry_row in registry_rows:
        request_id = str(registry_row["source_request_id"])
        generation_row = generation_by_id.get(request_id)
        if generation_row is None:
            raise ValueError(f"Frozen cohort request is absent: {request_id}")
        selected_rows.append(generation_row)
        pattern = (
            f"seed_{int(registry_row['source_seed'])}__"
            f"count_{int(registry_row['gold_count'])}__*.json"
        )
        pre_freeze_match = False
        for shard_path in shard_root.glob(pattern):
            if shard_path.stat().st_mtime > freeze_timestamp:
                continue
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            if shard == generation_row and str(shard.get("request_id")) == request_id:
                pre_freeze_match = True
                break
        if not pre_freeze_match:
            raise ValueError(
                f"Frozen cohort row lacks an object-equal pre-freeze shard: {request_id}"
            )
    canonical_hash = _canonical_rows_sha256(selected_rows)
    if canonical_hash != amendment.get("canonical_frozen_cohort_rows_sha256"):
        raise ValueError("Canonical frozen-cohort generation digest changed")
    return {
        "status": "PASS_AMENDED_APPENDABLE_CONTAINER_WITH_PRE_FREEZE_ROW_IDENTITY",
        "expected_generations_sha256": expected_hash,
        "observed_generations_sha256": observed_hash,
        "observed_aggregate_row_count": len(generation_rows),
        "frozen_cohort_row_count": len(selected_rows),
        "canonical_frozen_cohort_rows_sha256": canonical_hash,
        "object_equal_pre_freeze_shards": len(selected_rows),
        "amendment": str(amendment_path),
        "amendment_sha256": primary.sha256_file(amendment_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True
    )
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cohort-registry", type=Path, required=True)
    parser.add_argument("--source-writes", type=Path, required=True)
    parser.add_argument("--generation-container-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = _validate_contract(contract_path, args.model)
    model_contract = contract["models"][args.model]
    dependencies = contract["shared_frozen_dependencies"]
    if primary.sha256_file(args.v6_config) != dependencies["v6_index_config_sha256"]:
        raise ValueError("V6 index config hash differs from the frozen sensitivity")
    if primary.sha256_file(args.cohort_registry) != model_contract[
        "frozen_cohort_registry_sha256"
    ]:
        raise ValueError("Cohort registry hash differs from the frozen sensitivity")
    generation_container_audit = _validate_generation_container(
        generations_path=args.generations.resolve(),
        cohort_registry_path=args.cohort_registry.resolve(),
        model=args.model,
        model_contract=model_contract,
        amendment_path=args.generation_container_amendment,
    )

    source_manifest_path = args.source_writes / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("anchor_role") != "p0_item_end":
        raise ValueError("Sensitivity source writes are not p0_item_end")
    if source_manifest.get("model_label") != args.model:
        raise ValueError("Sensitivity source-write model changed")

    original_mode_contract = primary._mode_contract
    primary._mode_contract = _item_end_mode_contract
    try:
        behavior, targeted, panel, manifest = primary.build_panel(
            config_path=args.v6_config.resolve(),
            model_label=args.model,
            generations_path=args.generations.resolve(),
            cohort_registry=args.cohort_registry.resolve(),
            source_writes=args.source_writes.resolve(),
            seed_role="discovery",
        )
    finally:
        primary._mode_contract = original_mode_contract

    if manifest.get("anchor_role") != "p0_item_end":
        raise RuntimeError("Exploratory panel lost its item-end anchor")
    if len(behavior) != len(contract["analysis_slot_seeds"]):
        raise RuntimeError("Exploratory panel changed the frozen seed count")

    args.output.mkdir(parents=True, exist_ok=True)
    primary._atomic_jsonl(args.output / "behavior_anchor_registry.jsonl", behavior)
    primary._atomic_jsonl(args.output / "targeted_registry.jsonl", targeted)
    primary._atomic_jsonl(args.output / "mode_panel.jsonl", panel)
    output_hashes = {
        name: primary.sha256_file(args.output / name)
        for name in (
            "behavior_anchor_registry.jsonl",
            "targeted_registry.jsonl",
            "mode_panel.jsonl",
        )
    }
    sensitivity_manifest = {
        **manifest,
        "schema_version": "realistic_niah_v6_index_item_end_panel_v1",
        "scientific_scope": "exploratory_discovery_only",
        "post_hoc_motivated": True,
        "primary_artifacts_mutated": False,
        "primary_confirmation_mutated": False,
        "contract": str(contract_path),
        "contract_sha256": primary.sha256_file(contract_path),
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": primary.sha256_file(Path(__file__).resolve()),
        "generation_container_audit": generation_container_audit,
        "outputs": output_hashes,
    }
    primary._atomic_json(args.output / "manifest.json", sensitivity_manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": args.model,
                "anchor_role": "p0_item_end",
                "seed_count": len(behavior),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
