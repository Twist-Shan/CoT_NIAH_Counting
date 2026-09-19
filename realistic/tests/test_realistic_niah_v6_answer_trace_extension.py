from __future__ import annotations

import json
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from realistic_niah_v5 import causal as v5_causal
from realistic_niah_v6.answer_trace_extension import (
    PAIR_SCHEMA,
    POOL_EXHAUSTION_AMENDMENT_SCHEMA,
    RELAY_GEOMETRY_AMENDMENT_SCHEMA,
    coherent_slot_to_source,
    load_contract,
    load_relay_geometry_amendment,
    model_contract,
    select_low_mid_high_edges,
    sha256_file,
    validate_pool_exhaustion_amendment,
    validate_pair_registry,
)
from scripts.analyze_realistic_niah_v5_terminal_relay_mediation import (
    GEOMETRY_REASON,
)
from scripts.analyze_realistic_niah_v6_terminal_relay_mediation import (
    _all_geometry_not_applicable_artifacts,
    _native_estimator_artifacts,
)
from scripts.run_realistic_niah_v5 import causal_patch_conditions_need_basis
from scripts.build_realistic_niah_v6_answer_trace_extension_report import (
    build as build_extension_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "realistic_niah_v6_answer_trace_extension_v1.json"
POOL_AMENDMENT = (
    ROOT
    / "configs"
    / "realistic_niah_v6_answer_trace_pool_exhaustion_amendment1.json"
)
POLICY_AMENDMENT2 = (
    ROOT / "configs" / "realistic_niah_v6_replacement_policy_amendment2.json"
)
RELAY_GEOMETRY_AMENDMENT = (
    ROOT
    / "configs"
    / "realistic_niah_v6_bullet_terminal_relay_suffix4_amendment_v1.json"
)


def test_extension_contract_matches_native_layer_and_relay_grids() -> None:
    contract = load_contract(CONTRACT)
    qwen = model_contract(
        contract, prompt_mode="enumeration_index", model_label="Qwen3-8B"
    )
    gemma = model_contract(
        contract, prompt_mode="enumeration_bullet", model_label="Gemma4-E4B"
    )
    assert qwen["answer_layers"] == [0, 5, 10, 15, 20, 25, 30, 35]
    assert (qwen["relay_source_layer"], qwen["relay_layer"]) == (19, 26)
    assert gemma["answer_layers"] == [0, 6, 12, 18, 23, 29, 35, 41]
    assert (gemma["relay_source_layer"], gemma["relay_layer"]) == (16, 34)
    assert qwen["answer_site_id"] == "answer_query_v3"
    assert gemma["relay_geometry"] == "suffix8"
    assert contract["cohort"]["required_counts_per_slot"] == list(range(1, 11))
    assert contract["structural_amendment"][
        "answer_or_relay_intervention_outputs_existed"
    ] is False


def test_full_residual_answer_patch_does_not_require_a_basis_archive() -> None:
    assert causal_patch_conditions_need_basis(
        ["self_patch", "full_donor_patch"]
    ) is False
    assert causal_patch_conditions_need_basis(
        ["self_patch", "projected_donor_patch"]
    ) is True
    assert causal_patch_conditions_need_basis(
        ["orthogonal_norm_matched"]
    ) is True


def test_bullet_suffix4_amendment_is_uniform_and_keeps_suffix8_auditable() -> None:
    contract = load_contract(CONTRACT)
    amendment = load_relay_geometry_amendment(
        RELAY_GEOMETRY_AMENDMENT,
        extension_contract_path=CONTRACT,
    )
    assert amendment["schema_version"] == RELAY_GEOMETRY_AMENDMENT_SCHEMA
    assert amendment["scope"]["models"] == ["Qwen3-8B", "Gemma4-E4B"]
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        frozen = model_contract(
            contract,
            prompt_mode="enumeration_bullet",
            model_label=model,
            relay_geometry_amendment=amendment,
        )
        assert frozen["relay_original_geometry"] == "suffix8"
        assert frozen["relay_geometry"] == "suffix4"
        assert frozen["relay_scientific_label"] == (
            "post_hoc_task_adapted_bullet_relay_replication"
        )
        assert frozen["relay_original_artifacts_preserved"] is True
    with pytest.raises(ValueError, match="outside Bullet"):
        model_contract(
            contract,
            prompt_mode="enumeration_index",
            model_label="Qwen3-8B",
            relay_geometry_amendment=amendment,
        )


def test_bullet_suffix4_amendment_rejects_outcome_based_mutation(
    tmp_path: Path,
) -> None:
    value = json.loads(RELAY_GEOMETRY_AMENDMENT.read_text(encoding="utf-8"))
    value["selection_firewall"]["suffix4_intervention_outcomes_read"] = True
    mutated = tmp_path / "mutated_suffix4_amendment.json"
    mutated.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="crossed its firewall"):
        load_relay_geometry_amendment(
            mutated,
            extension_contract_path=CONTRACT,
        )


def test_full_residual_kernel_runs_without_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_generate(*_args, **kwargs):
        calls.append(kwargs)
        return {
            "full_answer_text": "Total: 3",
            "completion_text_raw": "3",
            "generated_token_count": 1,
            "generation_truncated": False,
        }

    monkeypatch.setattr(
        v5_causal, "generate_with_residual_interventions", fake_generate
    )
    receiver = SimpleNamespace(
        model_label="Qwen3-8B",
        count=3,
        request_id="receiver",
        seed=1254,
        split="confirmation",
        query_position=7,
    )
    donor = SimpleNamespace(
        model_label="Qwen3-8B",
        count=4,
        request_id="donor",
        seed=1254,
        split="confirmation",
        query_position=8,
    )
    rows = v5_causal.run_projected_patch_trials_from_states(
        object(),
        object(),
        object(),
        receiver,
        torch.tensor([1.0, 2.0]),
        donor,
        torch.tensor([3.0, 4.0]),
        receiver_site_id="answer_query_v3",
        donor_site_id="answer_query_v3",
        layer=5,
        basis=None,
        requested_conditions=("self_patch", "full_donor_patch"),
    )
    assert [row["condition"] for row in rows] == [
        "self_patch",
        "full_donor_patch",
    ]
    assert all(row["rank"] is None for row in rows)
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([1], []),
        ([1, 10], [(1, 10)]),
        ([1, 2, 9], [(1, 2), (2, 9)]),
        (list(range(1, 11)), [(1, 2), (5, 6), (9, 10)]),
        ([1, 3, 4, 8, 10], [(1, 3), (4, 8), (8, 10)]),
    ],
)
def test_low_mid_high_edge_selection_is_outcome_blind_and_deterministic(
    counts: list[int], expected: list[tuple[int, int]]
) -> None:
    assert select_low_mid_high_edges(counts) == expected


def test_coherent_slot_mapping_preserves_true_source_identity() -> None:
    rows = [
        {"seed": 2001, "v6_analysis_slot_seed": 1254, "gold_count": count}
        for count in (1, 2, 3)
    ] + [
        {"seed": 2002, "v6_analysis_slot_seed": 1255, "gold_count": count}
        for count in (1, 2, 3)
    ]
    assert coherent_slot_to_source(rows, expected_slots=[1254, 1255]) == {
        1254: 2001,
        1255: 2002,
    }
    rows[-1]["seed"] = 2001
    with pytest.raises(ValueError, match="source coherent"):
        coherent_slot_to_source(rows, expected_slots=[1254, 1255])


def test_pair_registry_rejects_seed_aliasing() -> None:
    pair = {
        "schema_version": PAIR_SCHEMA,
        "pair_id": "pair",
        "prompt_mode": "enumeration_index",
        "model_label": "Qwen3-8B",
        "seed": 2001,
        "v6_source_seed": 2001,
        "v6_analysis_slot_seed": 1254,
        "receiver_site_id": "answer_query_v3",
        "donor_site_id": "answer_query_v3",
        "receiver_exact_count": True,
        "donor_exact_count": True,
        "pair_selection_uses_patch_outcome": False,
        "layers": [0, 5, 10, 15, 20, 25, 30, 35],
    }
    audit = validate_pair_registry(
        [pair],
        prompt_mode="enumeration_index",
        model_label="Qwen3-8B",
        expected_layers=[0, 5, 10, 15, 20, 25, 30, 35],
        expected_slots=[1254],
    )
    assert audit["registered_pairs"] == 1
    pair["seed"] = 1254
    with pytest.raises(ValueError, match="aliases source seed"):
        validate_pair_registry(
            [pair],
            prompt_mode="enumeration_index",
            model_label="Qwen3-8B",
            expected_layers=[0, 5, 10, 15, 20, 25, 30, 35],
            expected_slots=[1254],
        )


def test_contract_rejects_postfreeze_layer_mutation(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["answer_query_full_state_patching"]["layers"]["Qwen3-8B"][-1] = 34
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="layer grids changed"):
        load_contract(mutated)


def test_answer_trace_pool_amendment_reuses_outcome_blind_frozen_suffix() -> None:
    amendment = json.loads(POOL_AMENDMENT.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_AMENDMENT2.read_text(encoding="utf-8"))
    assert amendment["schema_version"] == POOL_EXHAUSTION_AMENDMENT_SCHEMA
    assert amendment["recovery_rule"]["remaining_analysis_slots"] == [1263]
    assert amendment["recovery_rule"]["required_counts_per_candidate"] == list(
        range(1, 11)
    )
    assert policy["confirmation_pool_exhaustion_amendment"][
        "confirmation_extension_seeds"
    ] == list(range(1514, 1614))
    assert all(
        value is False for value in amendment["selection_firewall"].values()
    )
    assert all(amendment["scientific_scope_unchanged"].values())


def test_answer_trace_pool_amendment_validator_checks_hashes(tmp_path: Path) -> None:
    stimuli = tmp_path / "stimuli.jsonl"
    stimuli.write_text('{"seed":1514,"gold_count":1}\n', encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_bytes(POLICY_AMENDMENT2.read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "PASS_AMENDMENT_RESERVE_POOL",
                "replacement_policy_sha256": sha256_file(policy),
                "stimuli_sha256": sha256_file(stimuli),
            }
        ),
        encoding="utf-8",
    )
    amendment = json.loads(POOL_AMENDMENT.read_text(encoding="utf-8"))
    amendment["immutable_inputs"].update(
        {
            "extension_contract_sha256": sha256_file(CONTRACT),
            "replacement_policy_sha256": sha256_file(policy),
            "pool_manifest_sha256": sha256_file(manifest),
            "replacement_stimuli_sha256": sha256_file(stimuli),
        }
    )
    candidate = tmp_path / "amendment.json"
    candidate.write_text(json.dumps(amendment), encoding="utf-8")
    validate_pool_exhaustion_amendment(
        candidate,
        extension_contract_path=CONTRACT,
        replacement_policy_path=policy,
        pool_manifest_path=manifest,
        replacement_stimuli_path=stimuli,
        prompt_mode="enumeration_bullet",
        model_label="Gemma4-E4B",
    )
    amendment["selection_firewall"]["intervention_outcomes_read"] = True
    candidate.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(ValueError, match="selection firewall"):
        validate_pool_exhaustion_amendment(
            candidate,
            extension_contract_path=CONTRACT,
            replacement_policy_path=policy,
            pool_manifest_path=manifest,
            replacement_stimuli_path=stimuli,
            prompt_mode="enumeration_bullet",
            model_label="Gemma4-E4B",
        )


def test_answer_trace_recovery_is_scoped_to_gemma_bullet() -> None:
    queue = (
        ROOT / "scripts" / "queue_realistic_niah_v6_answer_trace_extension.sh"
    ).read_text(encoding="utf-8")
    supervisor = (
        ROOT / "scripts" / "supervise_realistic_niah_v6_answer_trace_extension.sh"
    ).read_text(encoding="utf-8")
    assert '"$model" == Gemma4-E4B && "$mode" == bullet' in queue
    assert "V6_ANSWER_TRACE_AMENDMENT_POOL" in queue
    assert "V6_ANSWER_TRACE_AMENDMENT_POLICY" in queue
    assert "validate_pool_exhaustion_amendment" in supervisor


def test_suffix4_relay_queue_applies_one_amendment_to_both_bullet_models() -> None:
    queue = (
        ROOT
        / "scripts"
        / "queue_realistic_niah_v6_bullet_terminal_relay_suffix4.sh"
    ).read_text(encoding="utf-8")
    supervisor = (
        ROOT
        / "scripts"
        / "supervise_realistic_niah_v6_bullet_terminal_relay_suffix4.sh"
    ).read_text(encoding="utf-8")
    assert "for model in Qwen3-8B Gemma4-E4B" in queue
    assert "--relay-geometry-amendment \"$AMENDMENT\"" in supervisor
    assert "--geometry suffix4" in supervisor
    assert "terminal_relay_partial_confirmation_suffix4" in supervisor
    assert "original_suffix8_artifacts_preserved" in supervisor


def test_extension_report_merges_bullet_suffix4_and_preserves_suffix8(
    tmp_path: Path,
) -> None:
    contract_sha = sha256_file(CONTRACT)
    amendment_sha = sha256_file(RELAY_GEOMETRY_AMENDMENT)
    run_root = tmp_path / "run"

    def write_relay(
        root: Path,
        *,
        geometry: str,
        estimable: bool,
        adapted: bool,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        eligible = 10 if estimable else 0
        full_na = [] if estimable else list(range(1254, 1264))
        reason = (
            None
            if estimable
            else (
                "not applicable: a trace item is shorter than the requested "
                f"{geometry} geometry"
            )
        )
        audit = {
            "status": "PASS_EXECUTION_COMPLETE",
            "extension_contract_sha256": contract_sha,
            "relay_geometry": geometry,
            "relay_geometry_amendment_sha256": amendment_sha if adapted else None,
            "planned_seed_count": 10,
            "eligible_seed_count": eligible,
            "relay_estimable": estimable,
            "not_estimable_reason": reason,
            "geometry_not_applicable_full_seed_count": len(full_na),
            "geometry_not_applicable_full_seeds": full_na,
            "partial_mediation_primary_pass": estimable,
            "scientific_result": "POSITIVE" if estimable else "NOT_ESTIMABLE_GEOMETRY",
        }
        gate_names = (
            "terminal_state_patch_effect",
            "post_terminal_suffix_specific_mediation",
            "post_terminal_suffix_residual_equivalence",
            "self_reset_is_nondamaging",
            "answer_query_only_mediation",
        )
        gates = {
            "gates": {
                name: {
                    "pass": estimable,
                    "estimate": 0.5 if estimable else None,
                    "ci_low": 0.2 if estimable else None,
                    "ci_high": 0.8 if estimable else None,
                }
                for name in gate_names
            }
        }
        (root / "v6_extension_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        (root / "claim_gates.json").write_text(
            json.dumps(gates), encoding="utf-8"
        )

    for mode in ("enumeration_index", "enumeration_bullet"):
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            extension_root = (
                run_root / mode / model / "causal" / "answer_trace_extension_v1"
            )
            extension_root.mkdir(parents=True, exist_ok=True)
            (extension_root / "extension_complete.json").write_text(
                json.dumps(
                    {
                        "status": "PASS_EXECUTION_COMPLETE",
                        "extension_contract_sha256": contract_sha,
                    }
                ),
                encoding="utf-8",
            )
            answer_root = extension_root / "answer_query_layer_sweep" / "analysis"
            answer_root.mkdir(parents=True, exist_ok=True)
            layers = (
                [0, 5, 10, 15, 20, 25, 30, 35]
                if model == "Qwen3-8B"
                else [0, 6, 12, 18, 23, 29, 35, 41]
            )
            (answer_root / "v6_extension_audit.json").write_text(
                json.dumps(
                    {
                        "pair_registry_audit": {"registered_pairs": 40},
                        "native_analysis_audit": {
                            "descriptive_onset_layer": {model: layers[-2]}
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (answer_root / "layer_effects.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "layer",
                        "seed_clusters",
                        "pairs",
                        "full_donor_adoption",
                        "full_donor_adoption_ci95_low",
                        "full_donor_adoption_ci95_high",
                        "adoption_specificity",
                        "adoption_specificity_ci95_low",
                        "adoption_specificity_ci95_high",
                        "registered_numeric_valid",
                    ),
                )
                writer.writeheader()
                for index, layer in enumerate(layers):
                    estimate = index / (len(layers) - 1)
                    writer.writerow(
                        {
                            "layer": layer,
                            "seed_clusters": 10,
                            "pairs": 40,
                            "full_donor_adoption": estimate,
                            "full_donor_adoption_ci95_low": max(0, estimate - 0.1),
                            "full_donor_adoption_ci95_high": min(1, estimate + 0.1),
                            "adoption_specificity": estimate,
                            "adoption_specificity_ci95_low": max(0, estimate - 0.1),
                            "adoption_specificity_ci95_high": min(1, estimate + 0.1),
                            "registered_numeric_valid": 1,
                        }
                    )
            original_analysis = (
                extension_root
                / "terminal_relay_partial_confirmation"
                / "relay_analysis_confirmation"
            )
            write_relay(
                original_analysis,
                geometry="suffix8",
                estimable=not (mode == "enumeration_bullet" and model == "Gemma4-E4B"),
                adapted=False,
            )
            if mode == "enumeration_bullet":
                adapted_analysis = (
                    extension_root
                    / "terminal_relay_partial_confirmation_suffix4"
                    / "relay_analysis_confirmation"
                )
                write_relay(
                    adapted_analysis,
                    geometry="suffix4",
                    estimable=True,
                    adapted=True,
                )

    summary = build_extension_report(
        run_root,
        tmp_path / "report",
        relay_geometry_amendment_path=RELAY_GEOMETRY_AMENDMENT,
    )
    by_key = {
        (cell["prompt_mode"], cell["model_label"]): cell
        for cell in summary["cells"]
    }
    assert summary["relay_geometry_policy"] == (
        "index_suffix8_bullet_suffix4_task_adapted"
    )
    assert by_key[("enumeration_bullet", "Qwen3-8B")]["relay_geometry"] == (
        "suffix4"
    )
    gemma = by_key[("enumeration_bullet", "Gemma4-E4B")]
    assert gemma["relay_estimable"] is True
    assert gemma["original_suffix8_relay"]["estimable"] is False
    assert gemma["original_suffix8_relay"]["eligible_seed_count"] == 0
    assert gemma["relay_geometry_amendment_sha256"] == amendment_sha


def test_all_geometry_na_relay_is_execution_pass_but_not_estimable() -> None:
    rows = []
    for seed in range(2001, 2011):
        pair_sha256 = f"{seed:064x}"
        for condition in range(6):
            rows.append(
                {
                    "seed": seed,
                    "pair_sha256": pair_sha256,
                    "condition": condition,
                    "status": "not_applicable",
                    "exclusion_reason": GEOMETRY_REASON,
                    "mechanism_split": "confirmation",
                }
            )

    gates, audit = _all_geometry_not_applicable_artifacts(
        rows, phase="confirmation", expected_seed_count=10
    )

    assert gates["estimable"] is False
    assert all(gate["estimable"] is False for gate in gates["gates"].values())
    assert all(gate["pass"] is False for gate in gates["gates"].values())
    assert audit["status"] == "PASS_EXECUTION_NOT_ESTIMABLE_GEOMETRY"
    assert audit["planned_seed_count"] == 10
    assert audit["seed_count"] == 0
    assert audit["geometry_not_applicable_full_seed_count"] == 10


def test_all_geometry_na_relay_still_rejects_wrong_split() -> None:
    rows = [
        {
            "seed": seed,
            "pair_sha256": f"{seed:064x}",
            "status": "not_applicable",
            "exclusion_reason": GEOMETRY_REASON,
            "mechanism_split": "confirmation",
        }
        for seed in range(2001, 2011)
        for _ in range(6)
    ]
    rows[-1]["mechanism_split"] = "development"

    with pytest.raises(ValueError, match="wrong split"):
        _all_geometry_not_applicable_artifacts(
            rows, phase="confirmation", expected_seed_count=10
        )


def test_all_geometry_na_reason_tracks_suffix4() -> None:
    reason = (
        "not applicable: a trace item is shorter than the requested suffix4 geometry"
    )
    rows = [
        {
            "seed": seed,
            "pair_sha256": f"{seed:064x}",
            "status": "not_applicable",
            "exclusion_reason": reason,
            "mechanism_split": "confirmation",
        }
        for seed in range(2001, 2011)
        for _ in range(6)
    ]
    gates, audit = _all_geometry_not_applicable_artifacts(
        rows,
        phase="confirmation",
        expected_seed_count=10,
        geometry="suffix4",
    )
    assert gates["not_estimable_reason"] == reason
    assert audit["geometry"] == "suffix4"


def test_suffix4_wrapper_reuses_native_numeric_estimator(tmp_path: Path) -> None:
    margins = {
        ("self_patch", "natural_relay"): 10.0,
        ("full_donor_patch", "natural_relay"): 6.0,
        ("self_patch", "post_terminal_suffix_clean_reset"): 10.0,
        ("full_donor_patch", "post_terminal_suffix_clean_reset"): 9.0,
        ("self_patch", "answer_query_clean_reset"): 10.0,
        ("full_donor_patch", "answer_query_clean_reset"): 8.0,
    }
    rows = []
    for seed in range(2001, 2011):
        for (source_condition, relay_condition), margin in margins.items():
            rows.append(
                {
                    "status": "ok",
                    "model_label": "Gemma4-E4B",
                    "seed": seed,
                    "request_id": f"request-{seed}",
                    "gold_count": 5,
                    "mechanism_split": "confirmation",
                    "pair_sha256": f"{seed:064x}",
                    "donor_offset": 1,
                    "source_layer": 16,
                    "relay_layer": 34,
                    "source_condition": source_condition,
                    "relay_condition": relay_condition,
                    "correct_count_margin": margin,
                }
            )
    gates, audit = _native_estimator_artifacts(
        rows,
        output=tmp_path,
        phase="confirmation",
        geometry="suffix4",
        source_layer=16,
        relay_layer=34,
        bootstrap_samples=200,
        random_seed=20260821,
    )
    assert audit["geometry"] == "suffix4"
    assert audit["seed_count"] == 10
    assert audit["geometry_not_applicable_pair_count"] == 0
    assert gates["gates"]["terminal_state_patch_effect"]["estimate"] == 4.0
    assert gates["gates"]["post_terminal_suffix_specific_mediation"][
        "estimate"
    ] == 3.0
