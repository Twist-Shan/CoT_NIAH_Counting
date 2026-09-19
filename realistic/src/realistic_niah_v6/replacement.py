from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .parsing import parse_trace_record
from .pipeline import read_jsonl, sha256_file
from .spec import V6Config


POLICY_SCHEMA_VERSION = "realistic_niah_v6_replacement_policy_v1"
POLICY_AMENDMENT1_SCHEMA_VERSION = (
    "realistic_niah_v6_replacement_policy_pool_exhaustion_amendment1"
)
POLICY_AMENDMENT2_SCHEMA_VERSION = (
    "realistic_niah_v6_replacement_policy_pool_exhaustion_amendment2"
)
POOL_SCHEMA_VERSION = "realistic_niah_v6_replacement_seed_pool_v1"
SELECTED_CELL_SCHEMA_VERSION = "realistic_niah_v6_resolved_cell_v1"
MAPPING_SCHEMA_VERSION = "realistic_niah_v6_replacement_mapping_v1"
ATTEMPT_SCHEMA_VERSION = "realistic_niah_v6_replacement_attempt_v1"
COHERENT_BROAD_MAPPING_SCHEMA_VERSION = (
    "realistic_niah_v6_coherent_broad_mapping_v1"
)
COHERENT_BROAD_ATTEMPT_SCHEMA_VERSION = (
    "realistic_niah_v6_coherent_broad_attempt_v1"
)
COHERENT_BROAD_POLICY_SCHEMA_VERSION = (
    "realistic_niah_v6_coherent_broad_replacement_policy_v1"
)


def load_replacement_policy(path: str | Path, config: V6Config) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V6 replacement policy must be one JSON object")
    schema_version = value.get("schema_version")
    if schema_version not in {
        POLICY_SCHEMA_VERSION,
        POLICY_AMENDMENT1_SCHEMA_VERSION,
        POLICY_AMENDMENT2_SCHEMA_VERSION,
    }:
        raise ValueError("V6 replacement policy has the wrong schema")
    expected_status = {
        POLICY_SCHEMA_VERSION: "AMENDMENT_FROZEN_BEFORE_REPLACEMENT_MODEL_OUTPUTS",
        POLICY_AMENDMENT1_SCHEMA_VERSION: (
            "POST_EXHAUSTION_AMENDMENT_FROZEN_BEFORE_EXTENSION_MODEL_OUTPUTS"
        ),
        POLICY_AMENDMENT2_SCHEMA_VERSION: (
            "POST_CONFIRMATION_EXHAUSTION_AMENDMENT_FROZEN_BEFORE_EXTENSION_MODEL_OUTPUTS"
        ),
    }[schema_version]
    if value.get("status") != expected_status:
        raise ValueError(
            "V6 replacement amendment was not frozen before reserve outputs"
        )
    if tuple(map(int, value.get("original_discovery_seeds", ()))) != tuple(
        config.discovery_seeds
    ):
        raise ValueError("Replacement policy discovery seeds changed")
    if tuple(map(int, value.get("original_confirmation_seeds", ()))) != tuple(
        config.confirmation_seeds
    ):
        raise ValueError("Replacement policy confirmation seeds changed")
    if tuple(map(int, value.get("counts", ()))) != tuple(config.counts):
        raise ValueError("Replacement policy count grid changed")
    discovery_pool = tuple(
        map(int, value.get("discovery_replacement_seed_pool", ()))
    )
    confirmation_pool = tuple(
        map(int, value.get("confirmation_replacement_seed_pool", ()))
    )
    if not discovery_pool or not confirmation_pool:
        raise ValueError("Both V6 replacement pools must be nonempty")
    pools = discovery_pool + confirmation_pool
    if len(set(pools)) != len(pools):
        raise ValueError("V6 replacement pools overlap or contain duplicates")
    if set(pools) & set(config.all_seeds):
        raise ValueError("A replacement seed overlaps the original V6 panel")
    if discovery_pool != tuple(sorted(discovery_pool)):
        raise ValueError("Discovery replacement pool must be ascending")
    if confirmation_pool != tuple(sorted(confirmation_pool)):
        raise ValueError("Confirmation replacement pool must be ascending")
    if int(value.get("target_successes_per_discovery_count", -1)) != len(
        config.discovery_seeds
    ):
        raise ValueError("Discovery replacement quota changed")
    if int(value.get("target_successes_per_confirmation_count", -1)) != len(
        config.confirmation_seeds
    ):
        raise ValueError("Confirmation replacement quota changed")
    if schema_version == POLICY_AMENDMENT1_SCHEMA_VERSION:
        amendment = value.get("pool_exhaustion_amendment")
        if not isinstance(amendment, Mapping):
            raise ValueError("Replacement pool amendment metadata is missing")
        base_name = Path(str(amendment.get("base_policy", ""))).name
        base_path = source.parent / base_name
        if not base_name or base_path.resolve() == source.resolve():
            raise ValueError("Replacement pool amendment has an invalid base policy")
        if not base_path.is_file():
            raise ValueError("Replacement pool amendment base policy is missing")
        expected_base_hash = str(amendment.get("base_policy_sha256", ""))
        if sha256_file(base_path) != expected_base_hash:
            raise ValueError("Replacement pool amendment base-policy hash changed")
        base_policy = load_replacement_policy(base_path, config)
        if base_policy.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("Replacement pool amendment does not name the V1 policy")
        base_discovery_pool = tuple(
            map(int, base_policy["discovery_replacement_seed_pool"])
        )
        base_confirmation_pool = tuple(
            map(int, base_policy["confirmation_replacement_seed_pool"])
        )
        extension = tuple(
            map(int, amendment.get("discovery_extension_seeds", ()))
        )
        if not extension or extension != tuple(sorted(extension)):
            raise ValueError("Discovery replacement extension must be nonempty and ascending")
        if discovery_pool != base_discovery_pool + extension:
            raise ValueError("Discovery replacement extension is not an exact suffix")
        if confirmation_pool != base_confirmation_pool:
            raise ValueError("Pool-exhaustion amendment changed confirmation seeds")
        if set(extension) & set(base_discovery_pool + base_confirmation_pool):
            raise ValueError("Discovery replacement extension overlaps the frozen pools")
        required_amendment_contract = {
            "trigger": "coherent_native_loop_discovery_pool_exhausted_fail_closed",
            "candidate_order": "ascending_extension_seed_order",
            "capacity_basis": "strict_parser_pass_fail_rate_only",
            "confirmation_pool_unchanged": True,
            "frozen_before_extension_model_outputs": True,
            "intervention_outcomes_read": False,
            "hidden_states_read": False,
            "attention_scores_read": False,
        }
        for name, expected in required_amendment_contract.items():
            if amendment.get(name) != expected:
                raise ValueError(
                    f"Replacement pool amendment field {name} changed"
                )
    elif schema_version == POLICY_AMENDMENT2_SCHEMA_VERSION:
        amendment = value.get("confirmation_pool_exhaustion_amendment")
        if not isinstance(amendment, Mapping):
            raise ValueError("Confirmation replacement pool amendment metadata is missing")
        base_name = Path(str(amendment.get("base_policy", ""))).name
        base_path = source.parent / base_name
        if not base_name or base_path.resolve() == source.resolve():
            raise ValueError(
                "Confirmation replacement pool amendment has an invalid base policy"
            )
        if not base_path.is_file():
            raise ValueError(
                "Confirmation replacement pool amendment base policy is missing"
            )
        expected_base_hash = str(amendment.get("base_policy_sha256", ""))
        if sha256_file(base_path) != expected_base_hash:
            raise ValueError(
                "Confirmation replacement pool amendment base-policy hash changed"
            )
        base_policy = load_replacement_policy(base_path, config)
        if base_policy.get("schema_version") != POLICY_AMENDMENT1_SCHEMA_VERSION:
            raise ValueError(
                "Confirmation pool amendment does not name discovery amendment 1"
            )
        if value.get("pool_exhaustion_amendment") != base_policy.get(
            "pool_exhaustion_amendment"
        ):
            raise ValueError(
                "Confirmation pool amendment changed discovery amendment history"
            )
        base_discovery_pool = tuple(
            map(int, base_policy["discovery_replacement_seed_pool"])
        )
        base_confirmation_pool = tuple(
            map(int, base_policy["confirmation_replacement_seed_pool"])
        )
        extension = tuple(
            map(int, amendment.get("confirmation_extension_seeds", ()))
        )
        if not extension or extension != tuple(sorted(extension)):
            raise ValueError(
                "Confirmation replacement extension must be nonempty and ascending"
            )
        if discovery_pool != base_discovery_pool:
            raise ValueError(
                "Confirmation pool amendment changed the discovery replacement pool"
            )
        if confirmation_pool != base_confirmation_pool + extension:
            raise ValueError(
                "Confirmation replacement extension is not an exact suffix"
            )
        if set(extension) & set(base_discovery_pool + base_confirmation_pool):
            raise ValueError(
                "Confirmation replacement extension overlaps the frozen pools"
            )
        required_amendment_contract = {
            "trigger": "coherent_broad_confirmation_pool_exhausted_fail_closed",
            "trigger_cell": "Gemma4-E4B x enumeration_bullet x confirmation",
            "candidate_order": "ascending_extension_seed_order",
            "capacity_basis": "strict_parser_complete_trajectory_pass_rate_only",
            "discovery_pool_unchanged": True,
            "frozen_before_extension_model_outputs": True,
            "intervention_outcomes_read": False,
            "hidden_states_read": False,
            "attention_scores_read": False,
        }
        for name, expected in required_amendment_contract.items():
            if amendment.get(name) != expected:
                raise ValueError(
                    f"Confirmation replacement pool amendment field {name} changed"
                )
        affected = int(amendment.get("observed_affected_analysis_slots", -1))
        accepted = int(
            amendment.get("accepted_complete_trajectories_before_exhaustion", -1)
        )
        remaining = int(
            amendment.get("remaining_complete_trajectory_shortfall", -1)
        )
        remaining_slots = tuple(
            map(int, amendment.get("remaining_analysis_slots", ()))
        )
        if (affected, accepted, remaining) != (9, 7, 2):
            raise ValueError(
                "Confirmation replacement pool amendment changed the exhaustion audit"
            )
        if remaining_slots != (1262, 1263):
            raise ValueError(
                "Confirmation replacement pool amendment changed the unresolved slots"
            )
    required_contract = {
        "replacement_scope": "model_x_prompt_mode_x_split_x_gold_count",
        "candidate_order": (
            "original_seed_ascending_then_replacement_seed_ascending"
        ),
        "eligibility_rule": "fresh_v6_parse.strict_causal_eligible_is_true",
        "pool_exhaustion_policy": (
            "fail_closed_and_request_an_explicit_protocol_amendment"
        ),
    }
    for name, expected in required_contract.items():
        if value.get(name) != expected:
            raise ValueError(f"Replacement policy field {name} changed")
    forbidden = set(map(str, value.get("forbidden_selection_inputs", ())))
    required_forbidden = {
        "hidden_state",
        "attention_score",
        "source_write_magnitude",
        "head_rank",
        "intervention_outcome",
        "causal_effect",
    }
    if not required_forbidden <= forbidden:
        raise ValueError("Replacement policy lost forbidden outcome fields")
    return value


def load_coherent_broad_policy(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Coherent broad replacement policy must be one object")
    if value.get("schema_version") != COHERENT_BROAD_POLICY_SCHEMA_VERSION:
        raise ValueError("Coherent broad replacement policy has the wrong schema")
    if value.get("status") != (
        "AMENDMENT_FROZEN_BEFORE_COHERENT_BROAD_SELECTION"
    ):
        raise ValueError("Coherent broad replacement policy was not frozen")
    required = {
        "applicable_phases": ["k_selection_discovery", "confirmation"],
        "trigger": (
            "any_registered_original_cell_for_the_broad_seed_panel_fails_"
            "runtime_or_fresh_v6_strict_parse"
        ),
        "replacement_unit": "entire_registered_five_count_broad_seed_panel",
        "candidate_order": "ascending_base_role_specific_reserve_seed_order",
        "candidate_reuse_within_panel": False,
        "eligibility_rule": (
            "one_unused_source_seed_has_fresh_v6_strict_causal_eligible_true_"
            "for_every_registered_count_in_the_panel"
        ),
        "successful_original_cells_in_an_affected_seed_panel": (
            "replace_for_true_source_seed_coherence_and_list_explicitly"
        ),
        "pool_exhaustion_policy": (
            "fail_closed_and_request_an_explicit_protocol_amendment"
        ),
        "statistical_identity_for_k_and_confirmation": "true_source_seed",
        "analysis_slot_identity": "original_seed_slot",
        "seed_aliasing": False,
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise ValueError(f"Coherent broad policy field {name} changed")
    forbidden = set(map(str, value.get("forbidden_selection_inputs", ())))
    if not {
        "hidden_state",
        "attention_score",
        "source_write_magnitude",
        "head_rank",
        "intervention_outcome",
        "causal_effect",
    } <= forbidden:
        raise ValueError("Coherent broad policy lost forbidden outcome fields")
    timing = value.get("timing_context")
    if not isinstance(timing, Mapping) or not bool(
        timing.get("reserve_strict_results_were_not_read_to_choose_this_rule")
    ):
        raise ValueError("Coherent broad policy lacks its timing disclosure")
    return value


def role_contract(
    policy: Mapping[str, Any], config: V6Config, seed_role: str
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if seed_role == "discovery":
        return (
            tuple(map(int, config.discovery_seeds)),
            tuple(map(int, policy["discovery_replacement_seed_pool"])),
            int(policy["target_successes_per_discovery_count"]),
        )
    if seed_role == "confirmation":
        return (
            tuple(map(int, config.confirmation_seeds)),
            tuple(map(int, policy["confirmation_replacement_seed_pool"])),
            int(policy["target_successes_per_confirmation_count"]),
        )
    raise ValueError("Replacement seed role must be discovery or confirmation")


def _failure_reasons(parsed: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not bool(parsed.get("strict_causal_eligible")):
        reasons.append("fresh_v6_strict_parser_failure")
    checks = (
        "enumeration_format_compliant",
        "exact_count",
        "exact_ordered_gold_pairs",
        "all_registered_sites_eligible",
    )
    for name in checks:
        if name in parsed and not bool(parsed.get(name)):
            reasons.append(name)
    parser = parsed.get("parser")
    if isinstance(parser, Mapping):
        status = parser.get("status")
        if status not in {None, "ok", "exact"}:
            reasons.append(f"parser_status:{status}")
    return list(dict.fromkeys(reasons))


def audit_generation_eligibility(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        parsed = parse_trace_record(row)
    except Exception as error:  # Fail closed and preserve the concrete parser error.
        return {
            "eligible": False,
            "failure_reasons": [
                "fresh_v6_parser_exception",
                f"{type(error).__name__}:{error}",
            ],
            "trace_parse": None,
        }
    eligible = bool(parsed.get("strict_causal_eligible"))
    return {
        "eligible": eligible,
        "failure_reasons": [] if eligible else _failure_reasons(parsed),
        "trace_parse": parsed,
    }


def _generation_index(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: V6Config,
    model_label: str,
    original_seeds: tuple[int, ...],
    replacement_pool: tuple[int, ...],
) -> dict[tuple[int, int], dict[str, Any]]:
    allowed_seeds = set(original_seeds) | set(replacement_pool)
    output: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in rows:
        if str(raw.get("model_label", raw.get("model", ""))) != model_label:
            continue
        if str(raw.get("prompt_mode", "")) != config.prompt_mode:
            continue
        seed = int(raw.get("seed", -1))
        count = int(raw.get("gold_count", -1))
        if seed not in allowed_seeds or count not in set(config.counts):
            continue
        key = (seed, count)
        if key in output:
            raise ValueError(f"Duplicate V6 generation cell: {key}")
        output[key] = dict(raw)
    return output


def resolve_replacement_panel(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: V6Config,
    model_label: str,
    seed_role: str,
    policy: Mapping[str, Any],
    runtime_failures: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve the first strict quota without consulting any mechanism output."""

    original_seeds, replacement_pool, quota = role_contract(
        policy, config, seed_role
    )
    indexed = _generation_index(
        rows,
        config=config,
        model_label=model_label,
        original_seeds=original_seeds,
        replacement_pool=replacement_pool,
    )
    runtime_failure_rows = [dict(value) for value in runtime_failures]
    failed_attempt_keys = {
        (int(value["seed"]), int(value["gold_count"]))
        for value in runtime_failure_rows
    }
    selected: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    next_candidates: list[dict[str, int]] = []
    shortfalls: dict[str, int] = {}

    eligibility: dict[tuple[int, int], dict[str, Any]] = {
        key: audit_generation_eligibility(row) for key, row in indexed.items()
    }
    for count in map(int, config.counts):
        eligible_replacements = [
            seed
            for seed in replacement_pool
            if eligibility.get((seed, count), {}).get("eligible") is True
        ]
        replacement_cursor = 0
        for slot_seed in original_seeds:
            original_key = (slot_seed, count)
            original = indexed.get(original_key)
            original_audit = eligibility.get(original_key)
            if original is not None and bool(original_audit and original_audit["eligible"]):
                source = original
                source_seed = slot_seed
                replacement_applied = False
                original_reasons: list[str] = []
                replacement_rank = None
            elif replacement_cursor < len(eligible_replacements):
                source_seed = int(eligible_replacements[replacement_cursor])
                replacement_cursor += 1
                source = indexed[(source_seed, count)]
                replacement_applied = True
                original_reasons = (
                    list(original_audit["failure_reasons"])
                    if original_audit is not None
                    else ["generation_missing_or_runtime_failure"]
                )
                replacement_rank = replacement_pool.index(source_seed) + 1
                mappings.append(
                    {
                        "schema_version": MAPPING_SCHEMA_VERSION,
                        "model_label": model_label,
                        "prompt_mode": config.prompt_mode,
                        "split": seed_role,
                        "gold_count": count,
                        "analysis_slot_seed": int(slot_seed),
                        "original_seed": int(slot_seed),
                        "original_request_id": (
                            str(original.get("request_id")) if original else None
                        ),
                        "original_failure_reasons": original_reasons,
                        "replacement_seed": source_seed,
                        "replacement_request_id": str(source["request_id"]),
                        "replacement_candidate_rank": replacement_rank,
                        "selection_basis": (
                            "lowest_reserved_seed_with_fresh_strict_parser_PASS"
                        ),
                        "intervention_outcomes_read": False,
                    }
                )
            else:
                continue
            selected.append(
                {
                    "schema_version": SELECTED_CELL_SCHEMA_VERSION,
                    "model_label": model_label,
                    "prompt_mode": config.prompt_mode,
                    "split": seed_role,
                    "gold_count": count,
                    "analysis_slot_seed": int(slot_seed),
                    "source_seed": int(source_seed),
                    "source_request_id": str(source["request_id"]),
                    "source_stimulus_id": str(source["stimulus_id"]),
                    "replacement_applied": bool(replacement_applied),
                    "original_failure_reasons": original_reasons,
                    "replacement_candidate_rank": replacement_rank,
                    "eligibility_rule": (
                        "fresh_v6_parse.strict_causal_eligible_is_true"
                    ),
                    "intervention_outcomes_read": False,
                }
            )

        selected_count = sum(
            int(row["gold_count"]) == count for row in selected
        )
        deficit = quota - selected_count
        if deficit < 0:
            raise RuntimeError("Replacement resolution exceeded its fixed quota")
        if deficit:
            shortfalls[str(count)] = deficit
            attempted_seeds = {
                seed for seed in replacement_pool if (seed, count) in indexed
            } | {
                seed for seed in replacement_pool if (seed, count) in failed_attempt_keys
            }
            unattempted = [seed for seed in replacement_pool if seed not in attempted_seeds]
            next_candidates.extend(
                {"seed": int(seed), "gold_count": count}
                for seed in unattempted[:deficit]
            )

        for seed in original_seeds + replacement_pool:
            key = (seed, count)
            row = indexed.get(key)
            audit = eligibility.get(key)
            if row is None and key not in failed_attempt_keys:
                continue
            kind = "original" if seed in set(original_seeds) else "replacement"
            runtime_failure = key in failed_attempt_keys and row is None
            attempts.append(
                {
                    "schema_version": ATTEMPT_SCHEMA_VERSION,
                    "model_label": model_label,
                    "prompt_mode": config.prompt_mode,
                    "split": seed_role,
                    "gold_count": count,
                    "seed": int(seed),
                    "candidate_kind": kind,
                    "candidate_rank": (
                        original_seeds.index(seed) + 1
                        if kind == "original"
                        else replacement_pool.index(seed) + 1
                    ),
                    "generation_present": row is not None,
                    "runtime_failure": runtime_failure,
                    "request_id": str(row.get("request_id")) if row else None,
                    "eligible": bool(audit and audit["eligible"]),
                    "failure_reasons": (
                        list(audit["failure_reasons"])
                        if audit is not None
                        else ["generation_missing_or_runtime_failure"]
                    ),
                    "selected": any(
                        str(value["source_request_id"])
                        == (str(row.get("request_id")) if row else "")
                        for value in selected
                    ),
                    "intervention_outcomes_read": False,
                }
            )

    selected.sort(
        key=lambda row: (int(row["gold_count"]), int(row["analysis_slot_seed"]))
    )
    mappings.sort(
        key=lambda row: (int(row["gold_count"]), int(row["analysis_slot_seed"]))
    )
    attempts.sort(
        key=lambda row: (
            int(row["gold_count"]),
            0 if row["candidate_kind"] == "original" else 1,
            int(row["candidate_rank"]),
        )
    )
    expected_slots = {
        (count, seed) for count in map(int, config.counts) for seed in original_seeds
    }
    observed_slots = {
        (int(row["gold_count"]), int(row["analysis_slot_seed"]))
        for row in selected
    }
    complete = observed_slots == expected_slots
    if complete and shortfalls:
        raise RuntimeError("Complete replacement panel still reports shortfalls")
    if complete and next_candidates:
        raise RuntimeError("Complete replacement panel still requests candidates")
    if len({str(row["source_request_id"]) for row in selected}) != len(selected):
        raise RuntimeError("A generation request was reused within the resolved panel")
    counts = Counter(int(row["gold_count"]) for row in selected)
    return {
        "complete": complete,
        "selected_cells": selected,
        "replacement_mapping": mappings,
        "attempt_ledger": attempts,
        "next_candidates": next_candidates,
        "shortfalls": shortfalls,
        "selected_per_count": {str(count): counts[count] for count in config.counts},
        "replacement_count": len(mappings),
        "original_seed_count": len(original_seeds),
        "quota_per_count": quota,
    }


def resolve_coherent_broad_panel(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: V6Config,
    model_label: str,
    seed_role: str,
    policy: Mapping[str, Any],
    mechanism: Any,
    phase: str,
    base_registry: Iterable[Mapping[str, Any]],
    runtime_failures: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve whole-seed broad panels without splicing source identities.

    The ordinary V6 formal cohort is resolved independently per count.  The
    broad K-selection and confirmation estimands, however, aggregate five
    requests per seed before bootstrapping over seeds.  If one of those five
    original requests fails, mixing an independently selected reserve request
    into the original seed would create a synthetic within-seed trajectory.

    This resolver therefore keeps an unaffected original broad seed intact and
    replaces *all* registered counts of an affected broad seed with one unused
    reserve source seed.  A reserve source is accepted only when every required
    count passes a fresh V6 strict parse.  Selection reads no intervention,
    hidden-state, attention, head-ranking, or causal output.
    """

    original_seeds, replacement_pool, _quota = role_contract(
        policy, config, seed_role
    )
    allowed_phases = {
        "discovery": {"ranking_discovery", "k_selection_discovery"},
        "confirmation": {"confirmation"},
    }
    if phase not in allowed_phases[seed_role]:
        raise ValueError(
            f"Broad phase {phase!r} is incompatible with seed role {seed_role!r}"
        )
    if phase == "ranking_discovery":
        panel_slots = tuple(map(int, mechanism.broad_ranking_seeds))
    elif phase == "k_selection_discovery":
        panel_slots = tuple(map(int, mechanism.broad_k_selection_seeds))
    else:
        panel_slots = tuple(map(int, mechanism.confirmation_seeds))
    if not panel_slots or not set(panel_slots) <= set(original_seeds):
        raise ValueError("Coherent broad slots disagree with the V6 seed role")

    required_by_slot = {
        slot: tuple(
            map(int, mechanism.broad_counts_for_seed(slot, phase=phase))
        )
        for slot in panel_slots
    }
    if any(not values for values in required_by_slot.values()):
        raise ValueError("A coherent broad slot has no registered counts")

    registry = [dict(value) for value in base_registry]
    expected_slots = {
        (int(count), int(seed))
        for count in config.counts
        for seed in original_seeds
    }
    observed_slots = [
        (int(value["gold_count"]), int(value["analysis_slot_seed"]))
        for value in registry
    ]
    if set(observed_slots) != expected_slots or len(observed_slots) != len(
        expected_slots
    ):
        raise ValueError("Base replacement registry does not fill the seed role")
    if {str(value.get("model_label")) for value in registry} != {model_label}:
        raise ValueError("Base replacement registry has the wrong model")
    if {str(value.get("prompt_mode")) for value in registry} != {
        config.prompt_mode
    }:
        raise ValueError("Base replacement registry has the wrong prompt mode")
    if {str(value.get("split")) for value in registry} != {seed_role}:
        raise ValueError("Base replacement registry has the wrong seed role")

    indexed = _generation_index(
        rows,
        config=config,
        model_label=model_label,
        original_seeds=original_seeds,
        replacement_pool=replacement_pool,
    )
    failed_runtime_keys = {
        (int(value["seed"]), int(value["gold_count"]))
        for value in runtime_failures
    }
    eligibility = {
        key: audit_generation_eligibility(value)
        for key, value in indexed.items()
    }

    affected_slots: list[int] = []
    original_failures_by_slot: dict[int, list[dict[str, Any]]] = {}
    for slot in panel_slots:
        failures: list[dict[str, Any]] = []
        for count in required_by_slot[slot]:
            key = (slot, count)
            value = indexed.get(key)
            audit = eligibility.get(key)
            if value is not None and audit and bool(audit["eligible"]):
                continue
            failures.append(
                {
                    "gold_count": count,
                    "request_id": (
                        str(value.get("request_id")) if value is not None else None
                    ),
                    "failure_reasons": (
                        list(audit["failure_reasons"])
                        if audit is not None
                        else ["generation_missing_or_runtime_failure"]
                    ),
                }
            )
        if failures:
            affected_slots.append(slot)
            original_failures_by_slot[slot] = failures

    affected_cells = {
        (slot, count)
        for slot in affected_slots
        for count in required_by_slot[slot]
    }
    protected_source_keys = {
        (int(value["source_seed"]), int(value["gold_count"]))
        for value in registry
        if (
            int(value["analysis_slot_seed"]),
            int(value["gold_count"]),
        )
        not in affected_cells
    }

    attempts: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    accepted_by_slot: dict[int, int] = {}
    consumed_seeds: set[int] = set()
    next_candidates: list[dict[str, int]] = []
    exhausted_slot: int | None = None

    for slot in affected_slots:
        required_counts = required_by_slot[slot]
        accepted: int | None = None
        for candidate_rank, candidate_seed in enumerate(replacement_pool, start=1):
            if candidate_seed in consumed_seeds:
                continue
            source_conflicts = sorted(
                count
                for count in required_counts
                if (candidate_seed, count) in protected_source_keys
            )
            if source_conflicts:
                consumed_seeds.add(candidate_seed)
                attempts.append(
                    {
                        "schema_version": COHERENT_BROAD_ATTEMPT_SCHEMA_VERSION,
                        "model_label": model_label,
                        "prompt_mode": config.prompt_mode,
                        "split": seed_role,
                        "phase": phase,
                        "analysis_slot_seed": slot,
                        "candidate_seed": candidate_seed,
                        "candidate_rank": candidate_rank,
                        "required_counts": list(required_counts),
                        "generated_counts": sorted(
                            count
                            for count in required_counts
                            if (candidate_seed, count) in indexed
                        ),
                        "eligible_counts": [],
                        "selected": False,
                        "failure_reasons": [
                            "source_request_already_used_outside_coherent_panel"
                        ],
                        "conflicting_counts": source_conflicts,
                        "intervention_outcomes_read": False,
                    }
                )
                continue

            missing_counts = [
                count
                for count in required_counts
                if (candidate_seed, count) not in indexed
                and (candidate_seed, count) not in failed_runtime_keys
            ]
            runtime_failed_counts = [
                count
                for count in required_counts
                if (candidate_seed, count) in failed_runtime_keys
                and (candidate_seed, count) not in indexed
            ]
            if missing_counts:
                next_candidates = [
                    {"seed": candidate_seed, "gold_count": count}
                    for count in missing_counts
                ]
                attempts.append(
                    {
                        "schema_version": COHERENT_BROAD_ATTEMPT_SCHEMA_VERSION,
                        "model_label": model_label,
                        "prompt_mode": config.prompt_mode,
                        "split": seed_role,
                        "phase": phase,
                        "analysis_slot_seed": slot,
                        "candidate_seed": candidate_seed,
                        "candidate_rank": candidate_rank,
                        "required_counts": list(required_counts),
                        "generated_counts": sorted(
                            count
                            for count in required_counts
                            if (candidate_seed, count) in indexed
                        ),
                        "eligible_counts": sorted(
                            count
                            for count in required_counts
                            if eligibility.get((candidate_seed, count), {}).get(
                                "eligible"
                            )
                            is True
                        ),
                        "selected": False,
                        "pending_generation": True,
                        "failure_reasons": [],
                        "intervention_outcomes_read": False,
                    }
                )
                break

            consumed_seeds.add(candidate_seed)
            audits = {
                count: eligibility.get((candidate_seed, count))
                for count in required_counts
            }
            eligible_counts = sorted(
                count
                for count, audit in audits.items()
                if audit is not None and bool(audit["eligible"])
            )
            failure_reasons: list[str] = []
            if runtime_failed_counts:
                failure_reasons.append("generation_runtime_failure")
            for count, audit in audits.items():
                if audit is None or bool(audit["eligible"]):
                    continue
                failure_reasons.extend(
                    f"count={count}:{reason}"
                    for reason in audit["failure_reasons"]
                )
            selected = len(eligible_counts) == len(required_counts)
            attempts.append(
                {
                    "schema_version": COHERENT_BROAD_ATTEMPT_SCHEMA_VERSION,
                    "model_label": model_label,
                    "prompt_mode": config.prompt_mode,
                    "split": seed_role,
                    "phase": phase,
                    "analysis_slot_seed": slot,
                    "candidate_seed": candidate_seed,
                    "candidate_rank": candidate_rank,
                    "required_counts": list(required_counts),
                    "generated_counts": sorted(
                        count
                        for count in required_counts
                        if (candidate_seed, count) in indexed
                    ),
                    "eligible_counts": eligible_counts,
                    "selected": selected,
                    "failure_reasons": list(dict.fromkeys(failure_reasons)),
                    "intervention_outcomes_read": False,
                }
            )
            if selected:
                accepted = candidate_seed
                accepted_by_slot[slot] = candidate_seed
                mappings.append(
                    {
                        "schema_version": COHERENT_BROAD_MAPPING_SCHEMA_VERSION,
                        "model_label": model_label,
                        "prompt_mode": config.prompt_mode,
                        "split": seed_role,
                        "phase": phase,
                        "analysis_slot_seed": slot,
                        "original_seed": slot,
                        "replacement_seed": candidate_seed,
                        "replacement_candidate_rank": candidate_rank,
                        "required_counts": list(required_counts),
                        "original_failed_cells": original_failures_by_slot[slot],
                        "selection_basis": (
                            "lowest_unused_reserve_seed_strict_across_entire_"
                            "registered_broad_panel"
                        ),
                        "successful_original_cells_replaced_for_seed_coherence": [
                            count
                            for count in required_counts
                            if count
                            not in {
                                int(value["gold_count"])
                                for value in original_failures_by_slot[slot]
                            }
                        ],
                        "intervention_outcomes_read": False,
                    }
                )
                break
        if next_candidates:
            break
        if accepted is None:
            exhausted_slot = slot
            break

    complete = (
        exhausted_slot is None
        and not next_candidates
        and set(accepted_by_slot) == set(affected_slots)
    )
    selected_cells: list[dict[str, Any]] = []
    if complete:
        registry_by_slot = {
            (int(value["analysis_slot_seed"]), int(value["gold_count"])): value
            for value in registry
        }
        for slot, count in sorted(
            ((seed, count) for count, seed in expected_slots),
            key=lambda value: (value[1], value[0]),
        ):
            base = dict(registry_by_slot[(slot, count)])
            replacement_seed = accepted_by_slot.get(slot)
            if replacement_seed is not None and count in set(
                required_by_slot[slot]
            ):
                source = indexed[(replacement_seed, count)]
                base.update(
                    {
                        "schema_version": SELECTED_CELL_SCHEMA_VERSION,
                        "source_seed": replacement_seed,
                        "source_request_id": str(source["request_id"]),
                        "source_stimulus_id": str(source["stimulus_id"]),
                        "replacement_applied": True,
                        "original_failure_reasons": (
                            [
                                reason
                                for failure in original_failures_by_slot[slot]
                                if int(failure["gold_count"]) == count
                                for reason in failure["failure_reasons"]
                            ]
                        ),
                        "replacement_candidate_rank": (
                            replacement_pool.index(replacement_seed) + 1
                        ),
                        "eligibility_rule": (
                            "fresh_v6_parse.strict_causal_eligible_is_true"
                        ),
                        "intervention_outcomes_read": False,
                        "v6_coherent_broad_phase": phase,
                        "v6_coherent_broad_replacement": True,
                        "v6_coherent_source_seed": replacement_seed,
                    }
                )
            else:
                base["v6_coherent_broad_phase"] = phase
                base["v6_coherent_broad_replacement"] = False
                base["v6_coherent_source_seed"] = int(base["source_seed"])
            selected_cells.append(base)

        request_ids = [str(value["source_request_id"]) for value in selected_cells]
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeError("Coherent broad registry reuses a generation request")
        panel_rows = [
            value
            for value in selected_cells
            if int(value["analysis_slot_seed"]) in set(panel_slots)
            and int(value["gold_count"])
            in set(required_by_slot[int(value["analysis_slot_seed"])])
        ]
        expected_panel_rows = sum(len(value) for value in required_by_slot.values())
        if len(panel_rows) != expected_panel_rows:
            raise RuntimeError("Coherent broad panel has the wrong cell count")
        source_sets = {
            slot: {
                int(value["source_seed"])
                for value in panel_rows
                if int(value["analysis_slot_seed"]) == slot
            }
            for slot in panel_slots
        }
        if any(len(values) != 1 for values in source_sets.values()):
            raise RuntimeError("A broad analysis slot mixes source seeds")
        if len({next(iter(values)) for values in source_sets.values()}) != len(
            source_sets
        ):
            raise RuntimeError("Two broad analysis slots share one source seed")

    return {
        "complete": complete,
        "selected_cells": selected_cells,
        "coherent_mapping": mappings,
        "attempt_ledger": attempts,
        "next_candidates": next_candidates,
        "pool_exhausted": exhausted_slot is not None,
        "exhausted_analysis_slot_seed": exhausted_slot,
        "phase": phase,
        "seed_role": seed_role,
        "panel_slots": list(panel_slots),
        "required_counts_by_slot": {
            str(slot): list(required_by_slot[slot]) for slot in panel_slots
        },
        "affected_slots": affected_slots,
        "accepted_replacement_seed_by_slot": {
            str(slot): seed for slot, seed in accepted_by_slot.items()
        },
        "replacement_seed_count": len(accepted_by_slot),
        "intervention_outcomes_read": False,
    }


def resolve_coherent_native_loop_panel(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: V6Config,
    model_label: str,
    seed_role: str,
    policy: Mapping[str, Any],
    base_registry: Iterable[Mapping[str, Any]],
    runtime_failures: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve a whole-source trajectory for every native-loop analysis slot.

    The frozen native-loop planner draws only from counts 2..10 and treats a
    source seed as the independent trajectory.  Cell-wise replacement would
    therefore manufacture a trajectory that no model run produced.  Reuse the
    already audited coherent-panel resolver with all role slots and all nine
    required counts, then relabel its provenance for this distinct design.
    """

    class _NativeLoopPanel:
        broad_k_selection_seeds = tuple(map(int, config.discovery_seeds))
        confirmation_seeds = tuple(map(int, config.confirmation_seeds))

        @staticmethod
        def broad_counts_for_seed(_slot: int, *, phase: str) -> tuple[int, ...]:
            if phase not in {"k_selection_discovery", "confirmation"}:
                raise ValueError(f"Unsupported native-loop resolver phase: {phase}")
            return tuple(int(count) for count in config.counts if int(count) >= 2)

    resolver_phase = (
        "k_selection_discovery" if seed_role == "discovery" else "confirmation"
    )
    result = resolve_coherent_broad_panel(
        rows,
        config=config,
        model_label=model_label,
        seed_role=seed_role,
        policy=policy,
        mechanism=_NativeLoopPanel(),
        phase=resolver_phase,
        base_registry=base_registry,
        runtime_failures=runtime_failures,
    )
    native_phase = f"native_loop_{seed_role}"
    for row in result["selected_cells"]:
        row.pop("v6_coherent_broad_phase", None)
        broad_replacement = bool(row.pop("v6_coherent_broad_replacement", False))
        row["v6_coherent_native_loop_phase"] = native_phase
        row["v6_coherent_native_loop_replacement"] = broad_replacement
        if "v6_coherent_source_seed" in row:
            row["v6_coherent_native_loop_source_seed"] = row.pop(
                "v6_coherent_source_seed"
            )
    for row in result["coherent_mapping"]:
        row["phase"] = native_phase
        row["selection_basis"] = (
            "lowest_unused_reserve_seed_strict_across_entire_registered_"
            "native_loop_count_2_to_10_panel"
        )
    for row in result["attempt_ledger"]:
        row["phase"] = native_phase
    result.update(
        {
            "phase": native_phase,
            "resolver_compatibility_kernel": resolver_phase,
            "panel_kind": "native_loop_true_source_trajectory",
            "required_counts": [
                int(count) for count in config.counts if int(count) >= 2
            ],
        }
    )
    return result


def resolved_generation_records(
    rows: Iterable[Mapping[str, Any]],
    config: V6Config,
    *,
    registry_path: str | Path,
    model_label: str | None = None,
) -> list[dict[str, Any]]:
    """Materialize the exact outcome-blind formal panel named by a registry."""

    registry = read_jsonl(registry_path)
    if not registry:
        raise ValueError("Resolved V6 cohort registry is empty")
    if any(row.get("schema_version") != SELECTED_CELL_SCHEMA_VERSION for row in registry):
        raise ValueError("Resolved V6 cohort registry has the wrong schema")
    modes = {str(row.get("prompt_mode")) for row in registry}
    if modes != {config.prompt_mode}:
        raise ValueError("Resolved registry prompt mode differs from V6 config")
    models = {str(row.get("model_label")) for row in registry}
    if model_label is not None and models != {model_label}:
        raise ValueError("Resolved registry model differs from requested model")
    if len(models) != 1:
        raise ValueError("Resolved registry must contain exactly one model")
    roles = {str(row.get("split")) for row in registry}
    if len(roles) != 1 or not roles <= {"discovery", "confirmation"}:
        raise ValueError("Resolved registry must contain exactly one seed role")
    role = next(iter(roles))
    original_seeds = (
        tuple(config.discovery_seeds)
        if role == "discovery"
        else tuple(config.confirmation_seeds)
    )
    expected_slots = {
        (int(count), int(seed))
        for count in config.counts
        for seed in original_seeds
    }
    observed_slots = [
        (int(row["gold_count"]), int(row["analysis_slot_seed"]))
        for row in registry
    ]
    if set(observed_slots) != expected_slots or len(observed_slots) != len(expected_slots):
        raise ValueError("Resolved registry does not fill every original analysis slot")
    request_ids = [str(row["source_request_id"]) for row in registry]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Resolved registry reuses one request in multiple slots")
    source_by_id: dict[str, dict[str, Any]] = {}
    for raw in rows:
        request_id = str(raw.get("request_id", ""))
        if request_id in set(request_ids):
            if request_id in source_by_id:
                raise ValueError(f"Duplicate generation request {request_id}")
            source_by_id[request_id] = dict(raw)
    missing = sorted(set(request_ids) - set(source_by_id))
    if missing:
        raise ValueError(f"Resolved generation requests are missing: {missing[:20]}")
    registry_by_id = {str(row["source_request_id"]): row for row in registry}
    output: list[dict[str, Any]] = []
    for registry_row in sorted(
        registry,
        key=lambda row: (int(row["gold_count"]), int(row["analysis_slot_seed"])),
    ):
        request_id = str(registry_row["source_request_id"])
        value = dict(source_by_id[request_id])
        if int(value["seed"]) != int(registry_row["source_seed"]):
            raise ValueError("Resolved registry source seed mismatch")
        if int(value["gold_count"]) != int(registry_row["gold_count"]):
            raise ValueError("Resolved registry count mismatch")
        audit = audit_generation_eligibility(value)
        if not audit["eligible"]:
            raise ValueError(
                f"Resolved registry contains a non-strict row: {request_id}"
            )
        value["trace_parse"] = audit["trace_parse"]
        value["split"] = role
        value["v6_analysis_slot_seed"] = int(registry_row["analysis_slot_seed"])
        value["v6_source_seed"] = int(registry_row["source_seed"])
        value["v6_replacement_applied"] = bool(
            registry_row["replacement_applied"]
        )
        value["v6_resolved_cohort_registry"] = str(Path(registry_path).resolve())
        value["v6_resolved_cohort_registry_sha256"] = sha256_file(registry_path)
        output.append(value)
    if Counter(int(row["gold_count"]) for row in output) != Counter(
        {int(count): len(original_seeds) for count in config.counts}
    ):
        raise RuntimeError("Resolved V6 cohort count balance changed")
    return output
