from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.analyze_realistic_niah_v6_index_item_end_anchor_sensitivity import (
    analyze_cell,
    bootstrap_summary,
    plan_heads,
)
from scripts.build_realistic_niah_v6_index_item_end_sensitivity_panel import (
    _canonical_rows_sha256,
    _validate_generation_container,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _trial(
    *,
    trial_id: str,
    request_id: str,
    condition: str,
    correct: bool,
    bank_role: str,
    site_role: str,
) -> dict:
    row = {
        "trial_id": trial_id,
        "request_id": request_id,
        "condition": condition,
        "correct_next_needle": correct,
        "status": "ok",
        "trial_complete": True,
    }
    if condition != "clean":
        row.update(
            {
                "head_ablation_decode_steps_requested": -1,
                "head_ablation_selected_post_zero_max_abs": 0.0,
                "intervention_anchor_roles": [site_role],
                "head_selection_anchor_role": bank_role,
            }
        )
    return row


def test_frozen_contract_keeps_primary_and_confirmation_immutable() -> None:
    contract = json.loads(
        (
            ROOT
            / "configs"
            / "realistic_niah_v6_index_item_end_anchor_sensitivity_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["status"] == "FROZEN_EXPLORATORY_BEFORE_NEW_ARM_OUTCOMES"
    assert contract["primary_protocol_unchanged"] is True
    assert contract["primary_confirmation_unchanged"] is True
    assert contract["confirmation_authorized"] is False
    assert contract["k_reselection_allowed"] is False
    assert contract["models"]["Qwen3-8B"]["fixed_k"] == 128
    assert contract["models"]["Gemma4-E4B"]["fixed_k"] == 8
    assert set(contract["fixed_design"]["cells"]) == {
        "p2bank_at_p2",
        "p2bank_at_p0",
        "p0bank_at_p2",
        "p0bank_at_p0",
    }


def test_bootstrap_summary_is_deterministic() -> None:
    first = bootstrap_summary([0.0, 1.0, -1.0], samples=200, seed=19)
    second = bootstrap_summary([0.0, 1.0, -1.0], samples=200, seed=19)
    assert first == second
    assert first["estimate"] == 0.0
    assert first["n_analysis_slot_seeds"] == 3


def test_plan_heads_compares_selected_bank_not_random_controls(tmp_path: Path) -> None:
    plan = tmp_path / "retrieval_anchor_bank_plan.csv"
    plan.write_text(
        "fold,condition,heads\n"
        '0,selected_bank,"[[1, 2], [3, 4]]"\n'
        '0,global_random,"[[9, 9], [8, 8]]"\n'
        '0,global_random,"[[7, 7], [6, 6]]"\n'
        '1,selected_bank,"[[5, 6]]"\n'
        '1,global_random,"[[4, 3]]"\n',
        encoding="utf-8",
    )

    assert plan_heads(plan) == {
        "0": {(1, 2), (3, 4)},
        "1": {(5, 6)},
    }


def test_analyze_item_end_cell_uses_one_selected_and_three_random_per_slot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "behavior" / "p0bank_at_p0"
    rows = []
    for slot, request_id in ((1, "request-a"), (2, "request-b")):
        rows.append(
            _trial(
                trial_id=f"clean-{slot}",
                request_id=request_id,
                condition="clean",
                correct=True,
                bank_role="p0_item_end",
                site_role="p0_item_end",
            )
        )
        rows.append(
            _trial(
                trial_id=f"selected-{slot}",
                request_id=request_id,
                condition="selected_bank",
                correct=slot == 2,
                bank_role="p0_item_end",
                site_role="p0_item_end",
            )
        )
        for repeat in range(3):
            rows.append(
                _trial(
                    trial_id=f"random-{slot}-{repeat}",
                    request_id=request_id,
                    condition="global_random",
                    correct=True,
                    bank_role="p0_item_end",
                    site_role="p0_item_end",
                )
            )
    _write_jsonl(root / "shards" / "trial_rows.jsonl", rows)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")

    summary, slots = analyze_cell(
        name="p0bank_at_p0",
        root=root,
        registry={"request-a": 1, "request-b": 2},
        random_condition="global_random",
        expected_slots={1, 2},
    )

    assert summary["selected_failure_rate"] == 0.5
    assert summary["registered_random_failure_rate"] == 0.0
    assert summary["clean_failure_rate"] == 0.0
    assert slots[1]["selected_minus_random_failure"] == 1.0
    assert slots[2]["selected_minus_random_failure"] == 0.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _container_amendment_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    generation = tmp_path / "generation" / "generations.jsonl"
    registry = tmp_path / "replacement" / "discovery" / "selected_cells.jsonl"
    rows = [
        {"request_id": "request-b", "value": 2},
        {"request_id": "request-a", "value": 1},
    ]
    _write_jsonl(generation, rows)
    _write_jsonl(
        registry,
        [
            {
                "source_request_id": "request-a",
                "source_seed": 11,
                "gold_count": 1,
            },
            {
                "source_request_id": "request-b",
                "source_seed": 12,
                "gold_count": 2,
            },
        ],
    )
    shard_root = generation.parent / "shards"
    shard_root.mkdir(parents=True)
    for seed, count, row in ((11, 1, rows[1]), (12, 2, rows[0])):
        shard = shard_root / f"seed_{seed}__count_{count}__fixture.json"
        shard.write_text(json.dumps(row), encoding="utf-8")
        os.utime(shard, (1_700_000_000, 1_700_000_000))
    amendment = {
        "schema_version": (
            "realistic_niah_v6_index_item_end_generation_container_amendment_v1"
        ),
        "status": "FROZEN_RECOVERY_BEFORE_GEMMA_SENSITIVITY_BEHAVIOR_OUTCOMES",
        "model_label": "Gemma4-E4B",
        "prompt_mode": "enumeration_index",
        "scientific_scope": "artifact_container_identity_only",
        "original_sensitivity_frozen_at_utc": "2026-08-29T18:35:39Z",
        "original_frozen_generations_sha256": "0" * 64,
        "observed_aggregate_generations_sha256": _sha(generation),
        "observed_aggregate_row_count": 2,
        "frozen_cohort_registry_sha256": _sha(registry),
        "frozen_cohort_row_count": 2,
        "canonical_frozen_cohort_rows_sha256": _canonical_rows_sha256(rows),
        "outcome_firewall": {
            "gemma_sensitivity_behavior_outcomes_existed_before_amendment": False,
            "behavior_outcomes_read_for_amendment": False,
            "head_scores_read_for_amendment": False,
            "source_write_values_read_for_amendment": False,
            "fixed_k_changed": False,
            "analysis_slot_seeds_changed": False,
            "registered_cells_changed": False,
            "intervention_scope_changed": False,
            "scientific_gate_changed": False,
        },
    }
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
    return generation, registry, amendment_path, amendment


def test_generation_container_amendment_requires_pre_freeze_row_identity(
    tmp_path: Path,
) -> None:
    generation, registry, amendment_path, amendment = _container_amendment_fixture(
        tmp_path
    )
    audit = _validate_generation_container(
        generations_path=generation,
        cohort_registry_path=registry,
        model="Gemma4-E4B",
        model_contract={
            "frozen_generations_sha256": amendment[
                "original_frozen_generations_sha256"
            ]
        },
        amendment_path=amendment_path,
    )
    assert audit["status"] == (
        "PASS_AMENDED_APPENDABLE_CONTAINER_WITH_PRE_FREEZE_ROW_IDENTITY"
    )
    assert audit["frozen_cohort_row_count"] == 2
    assert audit["object_equal_pre_freeze_shards"] == 2


def test_generation_container_amendment_fails_if_shard_is_post_freeze(
    tmp_path: Path,
) -> None:
    generation, registry, amendment_path, amendment = _container_amendment_fixture(
        tmp_path
    )
    late_shard = next((generation.parent / "shards").glob("seed_11__*.json"))
    os.utime(late_shard, (1_900_000_000, 1_900_000_000))
    with pytest.raises(ValueError, match="pre-freeze shard"):
        _validate_generation_container(
            generations_path=generation,
            cohort_registry_path=registry,
            model="Gemma4-E4B",
            model_contract={
                "frozen_generations_sha256": amendment[
                    "original_frozen_generations_sha256"
                ]
            },
            amendment_path=amendment_path,
        )
