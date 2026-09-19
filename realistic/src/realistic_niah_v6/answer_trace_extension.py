"""Frozen contracts for the V6 answer/trace causal extension.

This module contains only outcome-blind registry logic.  The numerical
intervention kernels remain the audited Native-thinking V5 kernels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONTRACT_SCHEMA = "realistic_niah_v6_answer_trace_extension_contract_v1"
POOL_EXHAUSTION_AMENDMENT_SCHEMA = (
    "realistic_niah_v6_answer_trace_pool_exhaustion_amendment_v1"
)
RELAY_GEOMETRY_AMENDMENT_SCHEMA = (
    "realistic_niah_v6_bullet_terminal_relay_geometry_amendment_v1"
)
PAIR_SCHEMA = "realistic_niah_v6_answer_query_layer_pair_v1"
PROMPT_MODES = ("enumeration_index", "enumeration_bullet")
MODEL_LABELS = ("Qwen3-8B", "Gemma4-E4B")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path)
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("Answer/trace extension contract has the wrong schema")
    if value.get("status") != "frozen_before_v6_extension_intervention_outcomes":
        raise ValueError("Answer/trace extension contract is not frozen")
    if tuple(value.get("prompt_modes", ())) != PROMPT_MODES:
        raise ValueError("Answer/trace extension prompt modes changed")
    if tuple(value.get("models", ())) != MODEL_LABELS:
        raise ValueError("Answer/trace extension model registry changed")
    amendment = value.get("structural_amendment", {})
    if amendment.get("answer_or_relay_intervention_outputs_existed") is not False:
        raise ValueError("Answer/trace structural amendment timing is invalid")
    if amendment.get("effect_magnitudes_read") is not False:
        raise ValueError("Answer/trace structural amendment read effect magnitudes")
    cohort = value.get("cohort", {})
    if cohort.get("base_replacement_policy") != (
        "configs/realistic_niah_v6_replacement_policy.json"
    ):
        raise ValueError("Answer/trace extension base replacement policy changed")
    if cohort.get("required_counts_per_slot") != list(range(1, 11)):
        raise ValueError("Answer/trace extension all-count trajectory changed")
    if cohort.get("replacement_unit") != (
        "entire count-1-to-10 source trajectory for the affected analysis slot"
    ):
        raise ValueError("Answer/trace extension replacement unit changed")

    answer = value.get("answer_query_full_state_patching", {})
    if answer.get("site_id") != "answer_query_v3":
        raise ValueError("Answer-query patch site must remain answer_query_v3")
    if answer.get("conditions") != ["self_patch", "full_donor_patch"]:
        raise ValueError("Answer-query patch conditions changed")
    expected_layers = {
        "Qwen3-8B": [0, 5, 10, 15, 20, 25, 30, 35],
        "Gemma4-E4B": [0, 6, 12, 18, 23, 29, 35, 41],
    }
    if answer.get("layers") != expected_layers:
        raise ValueError("Answer-query layer grids changed")

    relay = value.get("terminal_trace_to_answer_partial_mediation", {})
    if relay.get("source_conditions") != ["self_patch", "full_donor_patch"]:
        raise ValueError("Terminal relay source conditions changed")
    if relay.get("relay_conditions") != [
        "natural_relay",
        "answer_query_clean_reset",
        "post_terminal_suffix_clean_reset",
    ]:
        raise ValueError("Terminal relay conditions changed")
    if relay.get("geometry") != "suffix8":
        raise ValueError("Terminal relay geometry changed")
    if relay.get("layers") != {
        "Qwen3-8B": {"source": 19, "relay": 26},
        "Gemma4-E4B": {"source": 16, "relay": 34},
    }:
        raise ValueError("Terminal relay layers changed")
    return value


def validate_pool_exhaustion_amendment(
    path: str | Path,
    *,
    extension_contract_path: str | Path,
    replacement_policy_path: str | Path,
    pool_manifest_path: str | Path,
    replacement_stimuli_path: str | Path,
    prompt_mode: str,
    model_label: str,
) -> dict[str, Any]:
    """Validate the outcome-blind, Gemma-bullet cohort recovery envelope.

    The amendment changes only reserve-pool capacity after the original pool
    failed closed.  It cannot change layers, K, endpoints, decoding, cohort
    slots, strict parsing, or the order in which reserve seeds are considered.
    """

    amendment_path = Path(path)
    contract_path = Path(extension_contract_path)
    policy_path = Path(replacement_policy_path)
    manifest_path = Path(pool_manifest_path)
    stimuli_path = Path(replacement_stimuli_path)
    value = json.loads(amendment_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != POOL_EXHAUSTION_AMENDMENT_SCHEMA:
        raise ValueError("Answer/trace pool amendment has the wrong schema")
    if value.get("status") != (
        "FROZEN_AFTER_POOL_EXHAUSTION_BEFORE_AFFECTED_CELL_INTERVENTIONS"
    ):
        raise ValueError("Answer/trace pool amendment is not frozen")
    if value.get("affected_cell") != {
        "prompt_mode": prompt_mode,
        "model_label": model_label,
        "split": "confirmation",
        "phase": "answer_trace_confirmation",
    }:
        raise ValueError("Answer/trace pool amendment targets a different cell")
    if (prompt_mode, model_label) != ("enumeration_bullet", "Gemma4-E4B"):
        raise ValueError("Answer/trace pool amendment may only recover Gemma bullet")

    evidence = value.get("immutable_inputs", {})
    expected_hashes = {
        "extension_contract_sha256": sha256_file(contract_path),
        "replacement_policy_sha256": sha256_file(policy_path),
        "pool_manifest_sha256": sha256_file(manifest_path),
        "replacement_stimuli_sha256": sha256_file(stimuli_path),
    }
    for key, expected in expected_hashes.items():
        if evidence.get(key) != expected:
            raise ValueError(f"Answer/trace pool amendment {key} changed")

    contract = load_contract(contract_path)
    if contract["cohort"]["required_counts_per_slot"] != list(range(1, 11)):
        raise ValueError("Answer/trace amendment changed the count trajectory")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_audit = policy.get("confirmation_pool_exhaustion_amendment", {})
    if policy_audit.get("confirmation_extension_seeds") != list(range(1514, 1614)):
        raise ValueError("Answer/trace amendment did not reuse the frozen suffix")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_AMENDMENT_RESERVE_POOL":
        raise ValueError("Answer/trace amendment pool did not pass construction")
    if manifest.get("replacement_policy_sha256") != sha256_file(policy_path):
        raise ValueError("Answer/trace amendment pool used another policy")
    if manifest.get("stimuli_sha256") != sha256_file(stimuli_path):
        raise ValueError("Answer/trace amendment stimulus hash changed")

    recovery = value.get("recovery_rule", {})
    if recovery.get("candidate_order") != (
        "continue ascending cumulative confirmation reserve order without reuse"
    ):
        raise ValueError("Answer/trace amendment changed candidate order")
    if recovery.get("required_counts_per_candidate") != list(range(1, 11)):
        raise ValueError("Answer/trace amendment changed strict trajectory scope")
    if recovery.get("pool_exhaustion_policy") != "fail_closed":
        raise ValueError("Answer/trace amendment no longer fails closed")
    if recovery.get("remaining_analysis_slots") != [1263]:
        raise ValueError("Answer/trace amendment changed the unresolved slot")
    firewall = value.get("selection_firewall", {})
    forbidden = (
        "affected_cell_intervention_outputs_existed",
        "intervention_outcomes_read",
        "hidden_states_read",
        "attention_scores_read",
        "head_ranks_read",
        "effect_magnitudes_read",
    )
    if any(firewall.get(key) is not False for key in forbidden):
        raise ValueError("Answer/trace pool amendment crossed the selection firewall")
    unchanged = value.get("scientific_scope_unchanged", {})
    if not unchanged or any(flag is not True for flag in unchanged.values()):
        raise ValueError("Answer/trace pool amendment changed scientific scope")
    return value


def load_relay_geometry_amendment(
    path: str | Path,
    *,
    extension_contract_path: str | Path,
) -> dict[str, Any]:
    """Validate the outcome-blind suffix4 replication envelope.

    The original suffix8 assay is immutable.  This amendment is deliberately
    scoped to both Bullet models so that a model-specific relay outcome cannot
    choose the replacement geometry.
    """

    amendment_path = Path(path)
    contract_path = Path(extension_contract_path)
    value = json.loads(amendment_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != RELAY_GEOMETRY_AMENDMENT_SCHEMA:
        raise ValueError("Terminal relay geometry amendment has the wrong schema")
    if value.get("status") != (
        "FROZEN_AFTER_SUFFIX8_ESTIMABILITY_AUDIT_BEFORE_SUFFIX4_INTERVENTIONS"
    ):
        raise ValueError("Terminal relay geometry amendment is not frozen")
    load_contract(contract_path)
    if value.get("base_extension_contract_sha256") != sha256_file(contract_path):
        raise ValueError("Terminal relay geometry amendment base contract changed")
    if value.get("scientific_label") != (
        "post_hoc_task_adapted_bullet_relay_replication"
    ):
        raise ValueError("Terminal relay geometry amendment lost its evidence label")

    motivation = value.get("motivation", {})
    expected_motivation = {
        "original_geometry": "suffix8",
        "planned_seed_count": 10,
        "planned_pair_count": 190,
        "not_applicable_pair_count": 190,
        "planned_factorial_rows": 1140,
        "eligible_seed_count": 0,
        "suffix8_intervention_outputs_existed": True,
        "suffix8_effect_magnitudes_may_have_been_read": True,
    }
    if any(motivation.get(key) != expected for key, expected in expected_motivation.items()):
        raise ValueError("Terminal relay geometry amendment changed the suffix8 audit")

    scope = value.get("scope", {})
    if scope.get("prompt_modes") != ["enumeration_bullet"]:
        raise ValueError("Terminal relay geometry amendment is not Bullet-only")
    if scope.get("models") != list(MODEL_LABELS):
        raise ValueError("Terminal relay geometry amendment must cover both models")
    if scope.get("split") != "confirmation":
        raise ValueError("Terminal relay geometry amendment changed the split")
    if scope.get("application_unit") != (
        "the complete Bullet grammar across both registered models"
    ):
        raise ValueError("Terminal relay geometry amendment became model-specific")

    adaptation = value.get("adaptation", {})
    if adaptation.get("original_geometry") != "suffix8":
        raise ValueError("Terminal relay geometry amendment changed its baseline")
    if adaptation.get("replication_geometry") != "suffix4":
        raise ValueError("Terminal relay replication geometry must remain suffix4")
    if adaptation.get("geometry_choice_rule") != (
        "use the largest smaller fixed-width geometry already registered in the "
        "audited kernel; do not inspect suffix4 intervention outcomes and do not "
        "adapt width per pair"
    ):
        raise ValueError("Terminal relay geometry amendment changed its choice rule")
    if adaptation.get("pair_ineligibility_rule") != (
        "a pair shorter than suffix4 remains not_applicable in all six factorial arms"
    ):
        raise ValueError("Terminal relay geometry amendment weakened fail-closed N/A")
    if adaptation.get("per_pair_geometry_adaptation") is not False:
        raise ValueError("Terminal relay geometry amendment became pair-adaptive")
    invariant_fields = (
        "same_source_conditions",
        "same_relay_conditions",
        "same_model_specific_layers",
        "same_cohort_and_pair_plan_rule",
        "same_sequence_margin_estimands",
        "same_true_source_seed_bootstrap",
        "same_primary_and_secondary_gates",
        "original_suffix8_artifacts_preserved",
    )
    if any(adaptation.get(field) is not True for field in invariant_fields):
        raise ValueError("Terminal relay geometry amendment changed the estimator")
    if adaptation.get("output_directory_name") != (
        "terminal_relay_partial_confirmation_suffix4"
    ):
        raise ValueError("Terminal relay geometry amendment may overwrite suffix8")

    firewall = value.get("selection_firewall", {})
    forbidden_true = (
        "suffix4_intervention_outputs_existed",
        "suffix4_intervention_outcomes_read",
        "suffix4_effect_magnitudes_read",
        "geometry_chosen_using_suffix4_outcomes",
        "models_selected_using_intervention_outcomes",
        "cohort_changed",
        "pair_selection_rule_changed",
        "source_or_relay_layers_changed",
        "estimands_or_gates_changed",
    )
    if any(firewall.get(field) is not False for field in forbidden_true):
        raise ValueError("Terminal relay geometry amendment crossed its firewall")

    boundary = value.get("claim_boundary", {})
    if boundary.get("may_replace_original_suffix8_assay") is not False:
        raise ValueError("Terminal relay amendment may not replace suffix8")
    if boundary.get("may_be_called_original_frozen_confirmation") is not False:
        raise ValueError("Terminal relay amendment lost its post-hoc label")
    if boundary.get("original_suffix8_not_applicable_result_must_remain_reported") is not True:
        raise ValueError("Terminal relay amendment hid the suffix8 audit")
    return value


def model_contract(
    contract: Mapping[str, Any],
    *,
    prompt_mode: str,
    model_label: str,
    relay_geometry_amendment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported V6 prompt mode: {prompt_mode}")
    if model_label not in MODEL_LABELS:
        raise ValueError(f"Unsupported V6 model: {model_label}")
    answer = contract["answer_query_full_state_patching"]
    relay = contract["terminal_trace_to_answer_partial_mediation"]
    result = {
        "prompt_mode": prompt_mode,
        "model_label": model_label,
        "answer_layers": [int(value) for value in answer["layers"][model_label]],
        "answer_site_id": str(answer["site_id"]),
        "answer_conditions": list(answer["conditions"]),
        "relay_source_layer": int(relay["layers"][model_label]["source"]),
        "relay_layer": int(relay["layers"][model_label]["relay"]),
        "relay_geometry": str(relay["geometry"]),
    }
    if relay_geometry_amendment is not None:
        scope = relay_geometry_amendment.get("scope", {})
        if prompt_mode not in scope.get("prompt_modes", ()):
            raise ValueError("Terminal relay geometry amendment used outside Bullet")
        if model_label not in scope.get("models", ()):
            raise ValueError("Terminal relay geometry amendment omitted this model")
        adaptation = relay_geometry_amendment.get("adaptation", {})
        if adaptation.get("original_geometry") != result["relay_geometry"]:
            raise ValueError("Terminal relay geometry amendment baseline disagrees")
        result.update(
            {
                "relay_original_geometry": result["relay_geometry"],
                "relay_geometry": str(adaptation["replication_geometry"]),
                "relay_scientific_label": str(
                    relay_geometry_amendment["scientific_label"]
                ),
                "relay_original_artifacts_preserved": True,
            }
        )
    else:
        result.update(
            {
                "relay_original_geometry": result["relay_geometry"],
                "relay_scientific_label": "original_registered_suffix8",
                "relay_original_artifacts_preserved": True,
            }
        )
    return result


def select_low_mid_high_edges(
    counts: Sequence[int], *, cap: int = 3
) -> list[tuple[int, int]]:
    """Select adjacent available-count edges nearest frozen low/mid/high anchors."""

    ordered = sorted({int(value) for value in counts})
    edges = list(zip(ordered[:-1], ordered[1:]))
    edge_count = min(int(cap), len(edges))
    if edge_count < 1:
        return []
    targets = {
        1: (5.5,),
        2: (1.5, 9.5),
        3: (1.5, 5.5, 9.5),
    }[edge_count]
    selected: list[tuple[int, int]] = []
    for target in targets:
        remaining = [edge for edge in edges if edge not in selected]
        selected.append(
            min(
                remaining,
                key=lambda edge: (
                    abs(((edge[0] + edge[1]) / 2.0) - target),
                    edge[1] - edge[0],
                    edge[0],
                ),
            )
        )
    return sorted(selected)


def coherent_slot_to_source(
    rows: Iterable[Mapping[str, Any]], *, expected_slots: Sequence[int]
) -> dict[int, int]:
    """Validate one true source trajectory per frozen analysis slot."""

    sources_by_slot: dict[int, set[int]] = {}
    for row in rows:
        source = int(row["seed"])
        slot = int(row.get("v6_analysis_slot_seed", source))
        sources_by_slot.setdefault(slot, set()).add(source)
    expected = {int(value) for value in expected_slots}
    if set(sources_by_slot) != expected:
        raise ValueError(
            "Coherent V6 registry does not cover every frozen slot: "
            f"observed={sorted(sources_by_slot)} expected={sorted(expected)}"
        )
    bad = {
        slot: sorted(sources)
        for slot, sources in sources_by_slot.items()
        if len(sources) != 1
    }
    if bad:
        raise ValueError(f"V6 analysis slots are not source coherent: {bad}")
    mapping = {slot: next(iter(sources)) for slot, sources in sources_by_slot.items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Two V6 analysis slots share one true source seed")
    return dict(sorted(mapping.items()))


def resolve_coherent_answer_trace_panel(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: Any,
    model_label: str,
    policy: Mapping[str, Any],
    base_registry: Iterable[Mapping[str, Any]],
    runtime_failures: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve one strict count-1..10 true-source trajectory per V6 slot."""

    from .replacement import resolve_coherent_broad_panel

    class _AllCountConfirmationPanel:
        confirmation_seeds = tuple(map(int, config.confirmation_seeds))

        @staticmethod
        def broad_counts_for_seed(_slot: int, *, phase: str) -> tuple[int, ...]:
            if phase != "confirmation":
                raise ValueError(
                    f"Unsupported answer/trace resolver phase: {phase}"
                )
            return tuple(map(int, config.counts))

    result = resolve_coherent_broad_panel(
        rows,
        config=config,
        model_label=model_label,
        seed_role="confirmation",
        policy=policy,
        mechanism=_AllCountConfirmationPanel(),
        phase="confirmation",
        base_registry=base_registry,
        runtime_failures=runtime_failures,
    )
    extension_phase = "answer_trace_confirmation"
    for row in result["selected_cells"]:
        row.pop("v6_coherent_broad_phase", None)
        replacement = bool(row.pop("v6_coherent_broad_replacement", False))
        row["v6_coherent_answer_trace_phase"] = extension_phase
        row["v6_coherent_answer_trace_replacement"] = replacement
        if "v6_coherent_source_seed" in row:
            row["v6_coherent_answer_trace_source_seed"] = row.pop(
                "v6_coherent_source_seed"
            )
    for row in result["coherent_mapping"]:
        row["phase"] = extension_phase
        row["selection_basis"] = (
            "lowest_unused_confirmation_reserve_seed_strict_across_entire_"
            "count_1_to_10_trajectory"
        )
    for row in result["attempt_ledger"]:
        row["phase"] = extension_phase
    result.update(
        {
            "phase": extension_phase,
            "resolver_compatibility_kernel": "confirmation",
            "panel_kind": "answer_trace_all_count_true_source_trajectory",
            "required_counts": list(map(int, config.counts)),
        }
    )
    return result


def validate_pair_registry(
    pairs: Sequence[Mapping[str, Any]],
    *,
    prompt_mode: str,
    model_label: str,
    expected_layers: Sequence[int],
    expected_slots: Sequence[int],
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("Answer-query patch pair registry is empty")
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Answer-query patch pair IDs are not unique")
    layers = sorted({tuple(int(v) for v in row["layers"]) for row in pairs})
    expected_layer_tuple = tuple(int(v) for v in expected_layers)
    if layers != [expected_layer_tuple]:
        raise ValueError("Answer-query pair layer grid differs from the contract")
    for row in pairs:
        if row.get("schema_version") != PAIR_SCHEMA:
            raise ValueError("Answer-query pair has the wrong schema")
        if row.get("prompt_mode") != prompt_mode:
            raise ValueError("Answer-query pair prompt mode changed")
        if row.get("model_label") != model_label:
            raise ValueError("Answer-query pair model changed")
        if row.get("receiver_site_id") != "answer_query_v3" or row.get(
            "donor_site_id"
        ) != "answer_query_v3":
            raise ValueError("Answer-query pair site changed")
        if not bool(row.get("receiver_exact_count")) or not bool(
            row.get("donor_exact_count")
        ):
            raise ValueError("Answer-query pair is not clean-correct")
        if bool(row.get("pair_selection_uses_patch_outcome")):
            raise ValueError("Answer-query pair selection used an intervention outcome")
        if int(row["seed"]) != int(row["v6_source_seed"]):
            raise ValueError("Answer-query pair aliases source seed identity")
    sources_by_slot: dict[int, set[int]] = {}
    for row in pairs:
        sources_by_slot.setdefault(int(row["v6_analysis_slot_seed"]), set()).add(
            int(row["v6_source_seed"])
        )
    if any(len(values) != 1 for values in sources_by_slot.values()):
        raise ValueError("Answer-query pairs are not source coherent within slot")
    represented_sources = [
        next(iter(sources_by_slot[slot])) for slot in sorted(sources_by_slot)
    ]
    if len(represented_sources) != len(set(represented_sources)):
        raise ValueError("Two answer-query slots share one true source seed")
    slots = sorted(sources_by_slot)
    if not set(slots).issubset({int(value) for value in expected_slots}):
        raise ValueError("Answer-query pair contains an unregistered confirmation slot")
    return {
        "registered_pairs": len(pairs),
        "represented_slots": slots,
        "represented_true_source_seeds": sorted(represented_sources),
        "layers": list(expected_layer_tuple),
    }
