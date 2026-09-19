#!/usr/bin/env python3
"""Run an allow-listed audited V5 numerical kernel under the V6 protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v6.pipeline import (  # noqa: E402
    read_jsonl,
    registered_records,
    sha256_file,
    validate_generation_contracts,
    write_jsonl,
)
from realistic_niah_v6.spec import V6Config  # noqa: E402
from realistic_niah_v6.kernel import (  # noqa: E402
    SELECTED_BANK_CONTRACT_TARGETS,
    install_v6_selected_bank_contract,
    install_v6_specialized_geometry,
)
from realistic_niah_v6.replacement import (  # noqa: E402
    resolved_generation_records,
)


TARGETS = {
    "analyze-stratified-targeted-counter-ncc": "scripts/analyze_realistic_niah_v5_stratified_targeted_counter_ncc.py",
    "natural-aligned-progress": "scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py",
    "targeted-counter-write": "scripts/run_realistic_niah_v5_targeted_counter_write.py",
    "unified-carrier-transition": "scripts/run_realistic_niah_v5_unified_carrier_transition.py",
    "token-level-ablation": "scripts/run_realistic_niah_v5_token_level_ablation.py",
    "terminal-token-state-bridge": "scripts/run_realistic_niah_v6_terminal_token_state_bridge.py",
    "local-terminal-token-state-bridge": "scripts/run_realistic_niah_v6_local_terminal_token_state_bridge.py",
    "stratified-targeted-counter-ncc": "scripts/run_realistic_niah_v5_stratified_targeted_counter_ncc.py",
    "targeted-counter-logit-margin": "scripts/run_realistic_niah_v5_targeted_counter_logit_margin.py",
    "targeted-counter-ncc": "scripts/run_realistic_niah_v6_targeted_counter_ncc.py",
    "single-seed-walkthrough": "scripts/run_realistic_niah_v5_single_seed_walkthrough.py",
}

CONFIG_FLAGS = {
    "targeted-counter-write": "--v5-config",
    "token-level-ablation": "--config",
    "terminal-token-state-bridge": "--v5-config",
    "local-terminal-token-state-bridge": "--v5-config",
    "stratified-targeted-counter-ncc": "--v5-config",
    "targeted-counter-logit-margin": "--v5-config",
    "targeted-counter-ncc": "--v5-config",
    "single-seed-walkthrough": "--v5-config",
}


SPECIALIZED_SLOT_REGISTRY_FLAGS = {
    "targeted-counter-write": "--anchor-registry",
    "stratified-targeted-counter-ncc": "--panel",
    "targeted-counter-logit-margin": "--panel",
    "targeted-counter-ncc": "--anchor-registry",
    "terminal-token-state-bridge": "--anchor-registry",
    "local-terminal-token-state-bridge": "--anchor-registry",
    "token-level-ablation": "--anchor-registry",
}


def _option(arguments: list[str], name: str) -> tuple[int, str] | None:
    if name not in arguments:
        return None
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise ValueError(f"{name} requires a value")
    return index, arguments[index + 1]


def _registered_adapter(
    rows: Iterable[Mapping[str, Any]],
    config: V6Config,
    *,
    model_label: str | None = None,
) -> list[dict[str, Any]]:
    if _registered_adapter.cohort_registry is not None:
        return resolved_generation_records(
            rows,
            config,
            registry_path=_registered_adapter.cohort_registry,
            model_label=model_label,
        )
    return registered_records(
        rows,
        config,
        model_label=model_label,
        formal_only=not _registered_adapter.include_nonstrict,
    )


_registered_adapter.include_nonstrict = False
_registered_adapter.cohort_registry = None


def _install_count_stream_registered_rows_adapter(
    *,
    config_path: Path,
    include_nonstrict: bool,
    cohort_registry: Path | None,
) -> dict[str, Any]:
    """Install slot-only membership for specialized count-stream consumers."""

    from scripts import run_realistic_niah_v5_count_stream as legacy_count_stream
    from scripts import run_realistic_niah_v6_count_stream as v6_count_stream

    v6_count_stream._registered_adapter.include_nonstrict = include_nonstrict
    v6_count_stream._registered_adapter.config_sha256 = sha256_file(config_path)
    v6_count_stream._registered_adapter.cohort_registry = cohort_registry
    legacy_count_stream.V5Config = V6Config
    legacy_count_stream.registered_records = v6_count_stream._registered_adapter
    legacy_count_stream._registered_rows = lambda rows_args, mechanism: (
        v6_count_stream._v6_registered_rows(
            rows_args,
            mechanism,
            legacy=legacy_count_stream,
        )
    )
    return {
        "status": "INSTALLED",
        "panel_membership_identity": "v6_analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "generation_seed_field": "true_source_seed_unchanged",
        "seed_aliasing": False,
        "applies_to": (
            "specialized runners importing V5 count-stream _registered_rows"
        ),
    }


def _materialize_generation_view(
    arguments: list[str],
    *,
    config: V6Config,
    model_label: str | None,
    include_nonstrict: bool,
    config_path: Path,
    target: str,
    cohort_registry: Path | None,
) -> tuple[list[str], Path | None]:
    located = _option(arguments, "--generations")
    if located is None:
        return arguments, None
    index, raw_path = located
    source = Path(raw_path)
    source_rows = read_jsonl(source)
    validate_generation_contracts(
        source_rows,
        config,
        model_label=model_label,
        config_sha256=sha256_file(config_path),
    )
    if cohort_registry is not None:
        if include_nonstrict:
            raise ValueError("Resolved replacement cohorts are strict-only")
        rows = resolved_generation_records(
            source_rows,
            config,
            registry_path=cohort_registry,
            model_label=model_label,
        )
    else:
        rows = registered_records(
            source_rows,
            config,
            model_label=model_label,
            formal_only=not include_nonstrict,
        )
    if not rows:
        raise ValueError("No V6 rows remain in the requested kernel cohort")
    if target == "natural-aligned-progress":
        adapted = []
        grammar_class = {
            "enumeration_index": "structural_explicit_rank_before_city",
            "enumeration_bullet": "structural_invariant_bullet",
        }[config.prompt_mode]
        for row in rows:
            value = dict(row)
            value["v6_structured_enumeration_format_audit"] = {
                "primary_eligible_indexed_positive_control": True,
                "grammar_class": grammar_class,
                "adapter_reason": (
                    "Use the complete structured-enumeration answer registry; "
                    "no native-thinking early-stop suffix is constructed."
                ),
            }
            # The frozen V5 numerical kernel writes its output label from this
            # historical key.  Keep the eligibility bit only on the V6 audit
            # above so ``_read_rows`` still observes exactly one eligible
            # format audit; this alias carries no selection authority.
            value["indexed_progress_control_format_audit"] = {
                "grammar_class": grammar_class,
                "v6_compatibility_alias": True,
                "selection_authority": False,
            }
            value["v6_structured_enumeration_cohort"] = {
                "selection_population": "indexed_positive_control",
                "prompt_mode": config.prompt_mode,
                "strict_causal_eligible": True,
            }
            adapted.append(value)
        rows = adapted
    identity = (
        f"{sha256_file(source)}|{sha256_file(config_path)}|"
        f"{model_label}|{include_nonstrict}|"
        f"{sha256_file(cohort_registry) if cohort_registry else 'primary'}|"
        f"{target}"
    )
    import hashlib

    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    view = (
        ROOT
        / "work"
        / "realistic_niah_v6_adapter_inputs"
        / f"{config.mode_slug}__{model_label or 'all'}__{digest}.jsonl"
    )
    write_jsonl(view, rows)
    result = list(arguments)
    result[index + 1] = str(view)
    return result, view


def _specialized_seed_role(
    arguments: list[str], *, target: str
) -> tuple[str, str] | None:
    """Return the V6 split and the legacy role spelling for a slot assay."""

    if target == "token-level-ablation":
        located = _option(arguments, "--split")
        if located is None or located[1] == "all":
            return None
        split = str(located[1])
        return split, split
    located = _option(arguments, "--seed-role")
    if located is None:
        return None
    legacy_role = str(located[1])
    split_by_role = {
        "development": "discovery",
        "confirmation": "confirmation",
    }
    if legacy_role not in split_by_role:
        raise ValueError(f"Unknown specialized seed role: {legacy_role}")
    return split_by_role[legacy_role], legacy_role


def _materialize_specialized_seed_role_view(
    arguments: list[str],
    *,
    config: V6Config,
    config_path: Path,
    target: str,
) -> tuple[list[str], Path | None, dict[str, Any]]:
    """Translate only the V6 discovery role spelling for legacy panel kernels.

    V6 freezes panel membership as ``discovery``/``confirmation``.  The two
    inherited timing-stratified kernels call the same folds
    ``development``/``confirmation`` and compare the spelling literally.
    Materialize a process-local panel view that changes only that role label;
    request identity, analysis-slot identity, and true source seed remain
    byte-for-byte equal as parsed values.
    """

    applicable = {
        "stratified-targeted-counter-ncc",
        "targeted-counter-logit-margin",
    }
    base_audit: dict[str, Any] = {
        "status": "NOT_APPLICABLE",
        "target": target,
        "field": "stratified_ncc_seed_role",
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
        "intervention_outcomes_read": False,
        "intervention_values_changed": False,
    }
    if target not in applicable:
        return arguments, None, base_audit
    if {"--help", "-h"} & set(arguments):
        return arguments, None, {**base_audit, "status": "HELP_ONLY"}

    role = _specialized_seed_role(arguments, target=target)
    panel_option = _option(arguments, "--panel")
    if role is None or panel_option is None:
        raise ValueError(
            f"V6 {target} requires a single --seed-role and --panel"
        )
    split, legacy_role = role
    source = Path(panel_option[1]).resolve()
    rows = read_jsonl(source)
    if not rows:
        raise ValueError(f"V6 {target} panel is empty")

    expected_slots = tuple(
        map(
            int,
            config.discovery_seeds
            if split == "discovery"
            else config.confirmation_seeds,
        )
    )
    observed_slots: list[int] = []
    source_seeds: list[int] = []
    for row_index, row in enumerate(rows):
        observed_role = str(row.get("stratified_ncc_seed_role", ""))
        if observed_role != split:
            raise ValueError(
                f"V6 {target} panel row {row_index} has role "
                f"{observed_role!r}, expected frozen V6 role {split!r}"
            )
        if "analysis_slot_seed" not in row:
            raise ValueError(
                f"V6 {target} panel row {row_index} lacks analysis_slot_seed"
            )
        source_seed = int(row.get("source_seed", row.get("seed")))
        if int(row.get("seed", source_seed)) != source_seed:
            raise ValueError(f"V6 {target} panel aliases its true source seed")
        slot_seed = int(row["analysis_slot_seed"])
        if bool(row.get("replacement_applied", source_seed != slot_seed)) != (
            source_seed != slot_seed
        ):
            raise ValueError(f"V6 {target} panel replacement flag disagrees")
        observed_slots.append(slot_seed)
        source_seeds.append(source_seed)

    if tuple(sorted(observed_slots)) != tuple(sorted(expected_slots)):
        raise ValueError(
            f"V6 {target} panel analysis slots changed: "
            f"expected={sorted(expected_slots)}, observed={sorted(observed_slots)}"
        )
    if len(set(observed_slots)) != len(observed_slots):
        raise ValueError(f"V6 {target} panel duplicates an analysis slot")
    if len(set(source_seeds)) != len(source_seeds):
        raise ValueError(f"V6 {target} panel reuses a true source seed")

    # Confirmation already uses the same spelling in V6 and V5.  Keep the
    # original frozen panel path and hash in that case.
    if split == legacy_role:
        return (
            arguments,
            None,
            {
                **base_audit,
                "status": "NOT_NEEDED_ROLE_SPELLING_IDENTICAL",
                "v6_role": split,
                "legacy_role": legacy_role,
                "source_panel": str(source),
                "source_panel_sha256": sha256_file(source),
                "analysis_slot_count": len(observed_slots),
                "true_source_seed_count": len(set(source_seeds)),
            },
        )

    adapted = []
    for row in rows:
        value = dict(row)
        value["stratified_ncc_seed_role"] = legacy_role
        adapted.append(value)
    for original, value in zip(rows, adapted, strict=True):
        if {
            key: item
            for key, item in original.items()
            if key != "stratified_ncc_seed_role"
        } != {
            key: item
            for key, item in value.items()
            if key != "stratified_ncc_seed_role"
        }:
            raise RuntimeError("Specialized role adapter changed a non-role field")

    import hashlib

    identity = (
        f"{sha256_file(source)}|{sha256_file(config_path)}|{target}|"
        f"{split}|{legacy_role}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    view = (
        ROOT
        / "work"
        / "realistic_niah_v6_adapter_inputs"
        / f"specialized_role__{config.mode_slug}__{target}__{digest}.jsonl"
    )
    write_jsonl(view, adapted)
    result = list(arguments)
    result[panel_option[0] + 1] = str(view)
    return (
        result,
        view,
        {
            **base_audit,
            "status": "APPLIED_V6_TO_LEGACY_ROLE_SPELLING",
            "v6_role": split,
            "legacy_role": legacy_role,
            "changed_field_only": "stratified_ncc_seed_role",
            "source_panel": str(source),
            "source_panel_sha256": sha256_file(source),
            "materialized_panel": str(view.resolve()),
            "materialized_panel_sha256": sha256_file(view),
            "analysis_slot_count": len(observed_slots),
            "true_source_seed_count": len(set(source_seeds)),
            "slot_to_true_source_seed": [
                {
                    "analysis_slot_seed": slot_seed,
                    "true_source_seed": source_seed,
                }
                for slot_seed, source_seed in sorted(
                    zip(observed_slots, source_seeds, strict=True)
                )
            ],
        },
    )


def _audit_specialized_slot_identity(
    arguments: list[str],
    *,
    config: V6Config,
    target: str,
) -> dict[str, Any]:
    """Fail closed on slot/source aliasing before a specialized model load.

    The fixed N=10 transition panel is indexed by the original preregistered
    analysis slots.  A replacement request, however, must retain its real
    source seed in every numerical result.  This audit verifies both identities
    against the frozen panel and records the complete mapping in the V6 adapter
    manifest without changing a V5 numerical kernel or intervention output.
    """

    registry_flag = SPECIALIZED_SLOT_REGISTRY_FLAGS.get(target)
    if registry_flag is None:
        return {"status": "NOT_APPLICABLE"}
    if {"--help", "-h"} & set(arguments):
        return {"status": "HELP_ONLY"}
    role = _specialized_seed_role(arguments, target=target)
    if role is None:
        return {
            "status": "NOT_APPLIED_NO_SINGLE_REGISTERED_SPLIT",
            "target": target,
        }
    split, legacy_role = role
    generation_option = _option(arguments, "--generations")
    registry_option = _option(arguments, registry_flag)
    model_option = _option(arguments, "--model")
    if generation_option is None or registry_option is None or model_option is None:
        raise ValueError(
            f"V6 {target} requires --generations, {registry_flag}, and --model "
            "for the slot/source identity audit"
        )
    generations_path = Path(generation_option[1]).resolve()
    registry_path = Path(registry_option[1]).resolve()
    model_label = str(model_option[1])
    generations = read_jsonl(generations_path)
    registry_rows = read_jsonl(registry_path)
    if not registry_rows:
        raise ValueError(f"V6 {target} slot registry is empty")

    registry_by_request: dict[str, list[dict[str, Any]]] = {}
    for row in registry_rows:
        request_id = str(row.get("request_id", ""))
        if not request_id:
            raise ValueError(f"V6 {target} slot registry row lacks request_id")
        registry_by_request.setdefault(request_id, []).append(dict(row))
    duplicated_registry_requests = sorted(
        request_id
        for request_id, rows in registry_by_request.items()
        if len(rows) != 1
    )
    if duplicated_registry_requests:
        raise ValueError(
            f"V6 {target} fixed-transition registry does not contain exactly "
            f"one row per request: {duplicated_registry_requests}"
        )

    matched = [
        dict(row)
        for row in generations
        if str(row.get("request_id", "")) in registry_by_request
        and str(row.get("model_label", model_label)) == model_label
        and str(row.get("split", "")) == split
    ]
    if len(matched) != len(registry_by_request):
        observed = {str(row.get("request_id", "")) for row in matched}
        missing = sorted(set(registry_by_request) - observed)
        raise ValueError(
            f"V6 {target} generations do not reproduce the fixed registry: "
            f"missing={missing}"
        )
    if len({str(row["request_id"]) for row in matched}) != len(matched):
        raise ValueError(f"V6 {target} generations duplicate a fixed request")

    expected_slots = tuple(
        map(
            int,
            config.discovery_seeds
            if split == "discovery"
            else config.confirmation_seeds,
        )
    )
    mappings: list[dict[str, Any]] = []
    for generation in matched:
        request_id = str(generation["request_id"])
        registry = registry_by_request[request_id][0]
        if "v6_analysis_slot_seed" not in generation:
            raise ValueError(
                f"V6 {target} generation {request_id} lacks v6_analysis_slot_seed"
            )
        if "analysis_slot_seed" not in registry:
            raise ValueError(
                f"V6 {target} registry {request_id} lacks analysis_slot_seed"
            )
        source_seed = int(generation["seed"])
        slot_seed = int(generation["v6_analysis_slot_seed"])
        replacement_applied = bool(
            generation.get("v6_replacement_applied", source_seed != slot_seed)
        )
        if int(generation.get("v6_source_seed", source_seed)) != source_seed:
            raise ValueError(f"V6 {target} generation aliases its source seed")
        if int(registry["analysis_slot_seed"]) != slot_seed:
            raise ValueError(
                f"V6 {target} slot disagrees for request {request_id}: "
                f"generation={slot_seed}, registry={registry['analysis_slot_seed']}"
            )
        if int(registry.get("source_seed", registry.get("seed"))) != source_seed:
            raise ValueError(f"V6 {target} registry aliases its source seed")
        if int(registry.get("seed", source_seed)) != source_seed:
            raise ValueError(f"V6 {target} registry row.seed is not the true source")
        if bool(registry.get("replacement_applied", replacement_applied)) != replacement_applied:
            raise ValueError(f"V6 {target} replacement flag disagrees")
        if int(generation.get("gold_count", 0)) != 10 or int(
            registry.get("gold_count", 0)
        ) != 10:
            raise ValueError(f"V6 {target} must use the fixed N=10 transition")
        mappings.append(
            {
                "analysis_slot_seed": slot_seed,
                "true_source_seed": source_seed,
                "request_id": request_id,
                "replacement_applied": replacement_applied,
            }
        )

    mappings.sort(key=lambda row: int(row["analysis_slot_seed"]))
    observed_slots = tuple(int(row["analysis_slot_seed"]) for row in mappings)
    if observed_slots != tuple(sorted(expected_slots)):
        raise ValueError(
            f"V6 {target} analysis-slot contract changed: "
            f"expected={sorted(expected_slots)}, observed={list(observed_slots)}"
        )
    source_seeds = [int(row["true_source_seed"]) for row in mappings]
    if len(set(source_seeds)) != len(source_seeds):
        raise ValueError(
            f"V6 {target} fixed N=10 panel reuses a true source seed"
        )

    bank_audit: dict[str, Any] = {"status": "NOT_APPLICABLE"}
    bank_option = _option(arguments, "--bank-plan")
    if bank_option is not None:
        bank_path = Path(bank_option[1]).resolve()
        with bank_path.open("r", encoding="utf-8", newline="") as handle:
            plan_rows = [dict(row) for row in csv.DictReader(handle)]
        selected = [
            row
            for row in plan_rows
            if str(row.get("model_label")) == model_label
            and str(row.get("condition")) == "selected_bank"
        ]
        bank_size_option = _option(arguments, "--bank-size")
        if bank_size_option is not None:
            selected = [
                row
                for row in selected
                if int(row.get("bank_size", -1)) == int(bank_size_option[1])
            ]
        if len(selected) != 1:
            raise ValueError(
                f"V6 {target} expected one discovery-frozen selected bank, "
                f"found {len(selected)}"
            )
        validation_raw = selected[0].get("validation_seeds")
        if validation_raw in {None, ""}:
            raise ValueError(
                f"V6 {target} selected bank lacks validation_seeds membership"
            )
        validation_seeds = {int(value) for value in json.loads(validation_raw)}
        missing_sources = sorted(set(source_seeds) - validation_seeds)
        if missing_sources:
            raise ValueError(
                f"V6 {target} selected bank omits true source seeds: "
                f"{missing_sources}"
            )
        bank_audit = {
            "status": "PASS_TRUE_SOURCE_MEMBERSHIP",
            "bank_plan": str(bank_path),
            "bank_plan_sha256": sha256_file(bank_path),
            "validation_seed_count": len(validation_seeds),
            "all_true_source_seeds_registered": True,
        }

    return {
        "status": "PASS_FIXED_SLOT_TRUE_SOURCE_IDENTITY",
        "target": target,
        "prompt_mode": config.prompt_mode,
        "model_label": model_label,
        "split": split,
        "legacy_seed_role": legacy_role,
        "fixed_gold_count": 10,
        "analysis_slot_count": len(mappings),
        "true_source_seed_count": len(set(source_seeds)),
        "replacement_count": sum(
            bool(row["replacement_applied"]) for row in mappings
        ),
        "panel_membership_identity": "v6_analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "generation_seed_field": "true_source_seed_unchanged",
        "seed_aliasing": False,
        "slot_to_true_source_mapping": mappings,
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "materialized_generations": str(generations_path),
        "materialized_generations_sha256": sha256_file(generations_path),
        "bank_validation_membership": bank_audit,
        "intervention_outcomes_read": False,
    }


def _materialize_confirmation_bank_membership(
    arguments: list[str],
    *,
    config: V6Config,
    config_path: Path,
    model_label: str | None,
    phase: str,
    target: str,
    cohort_registry: Path | None,
) -> tuple[list[str], Path | None, dict[str, Any]]:
    """Add replacement source seeds to their frozen bank-validation fold.

    Head identities, scores, conditions, and discovery choices stay byte-for-
    byte represented by the original bank plan; only ``validation_seeds`` is
    expanded in a derived process-local CSV after the original plan has passed
    the selected-bank hash contract.
    """

    audit: dict[str, Any] = {
        "status": "NOT_APPLIED",
        "purpose": "specialized_confirmation_true_source_seed_routing",
        "target": target,
        "phase": phase,
        "validation_seed_routing_only": True,
        "selected_heads_or_scores_changed": False,
        "statistical_identity": "true_source_seed",
        "panel_membership_identity": "analysis_slot_seed",
        "seed_aliasing": False,
        "confirmation_intervention_outcomes_read": False,
    }
    if phase != "confirmation" or target not in SELECTED_BANK_CONTRACT_TARGETS:
        return arguments, None, audit
    located = _option(arguments, "--bank-plan")
    if located is None or cohort_registry is None or model_label is None:
        raise ValueError(
            f"V6 confirmation {target} requires --bank-plan, --model, and "
            "--cohort-registry"
        )
    registry_path = cohort_registry.resolve()
    registry_rows = read_jsonl(registry_path)
    roles = {str(row.get("split")) for row in registry_rows}
    if roles != {"confirmation"}:
        raise ValueError("Specialized confirmation registry has the wrong role")
    replacements = sorted(
        {
            (int(row["analysis_slot_seed"]), int(row["source_seed"]))
            for row in registry_rows
            if bool(row["replacement_applied"])
        }
    )
    if not replacements:
        return (
            arguments,
            None,
            {
                **audit,
                "status": "NOT_NEEDED_NO_CONFIRMATION_REPLACEMENTS",
                "cohort_registry": str(registry_path),
                "cohort_registry_sha256": sha256_file(registry_path),
            },
        )

    index, raw_plan = located
    plan = Path(raw_plan).resolve()
    with plan.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not rows or "validation_seeds" not in fieldnames:
        raise ValueError(
            "Discovery-frozen specialized bank has no validation_seeds column"
        )
    original_nonmembership = [
        {key: value for key, value in row.items() if key != "validation_seeds"}
        for row in rows
    ]
    routed: set[tuple[int, int]] = set()
    folds_by_source: dict[int, set[str]] = {
        source_seed: set() for _slot_seed, source_seed in replacements
    }
    for row_index, row in enumerate(rows):
        try:
            validation = {int(value) for value in json.loads(row["validation_seeds"])}
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid validation_seeds in specialized bank row {row_index}"
            ) from error
        additions = {
            source_seed
            for slot_seed, source_seed in replacements
            if slot_seed in validation
        }
        for slot_seed, source_seed in replacements:
            if slot_seed in validation:
                routed.add((slot_seed, source_seed))
        if additions:
            fold = str(row.get("fold", "0"))
            for source_seed in additions:
                folds_by_source[source_seed].add(fold)
            row["validation_seeds"] = json.dumps(sorted(validation | additions))
    if routed != set(replacements):
        raise ValueError(
            "Discovery-frozen specialized bank does not route every "
            f"replacement slot: missing={sorted(set(replacements) - routed)}"
        )
    ambiguous = {
        str(seed): sorted(folds)
        for seed, folds in folds_by_source.items()
        if len(folds) != 1
    }
    if ambiguous:
        raise ValueError(
            "A specialized replacement source maps to multiple folds: "
            f"{ambiguous}"
        )
    observed_nonmembership = [
        {key: value for key, value in row.items() if key != "validation_seeds"}
        for row in rows
    ]
    if observed_nonmembership != original_nonmembership:
        raise RuntimeError("Specialized bank routing changed non-membership fields")

    identity = (
        f"{sha256_file(plan)}|{sha256_file(config_path)}|"
        f"{sha256_file(registry_path)}|{model_label}|{target}"
    )
    import hashlib

    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    view = (
        ROOT
        / "work"
        / "realistic_niah_v6_adapter_inputs"
        / (
            f"specialized_bank_membership__{config.mode_slug}__"
            f"{model_label}__{digest}.csv"
        )
    )
    view.parent.mkdir(parents=True, exist_ok=True)
    temporary = view.with_name(f".{view.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(view)
    result = list(arguments)
    result[index + 1] = str(view)
    return (
        result,
        view,
        {
            **audit,
            "status": "APPLIED_CONFIRMATION_REPLACEMENT_ROUTING",
            "source_plan": str(plan),
            "source_plan_sha256": sha256_file(plan),
            "materialized_plan": str(view.resolve()),
            "materialized_plan_sha256": sha256_file(view),
            "cohort_registry": str(registry_path),
            "cohort_registry_sha256": sha256_file(registry_path),
            "replacement_slot_to_source_seed": [
                {"analysis_slot_seed": slot, "source_seed": source}
                for slot, source in replacements
            ],
            "added_true_source_seeds": sorted(folds_by_source),
        },
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _adapter_manifest_path(arguments: list[str]) -> Path | None:
    located = _option(arguments, "--output")
    if located is None:
        return None
    output = Path(located[1])
    if output.suffix:
        return output.with_suffix(output.suffix + ".v6_adapter.json")
    return output / "v6_adapter_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V6 process adapter for allow-listed report kernels"
    )
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["discovery", "confirmation", "diagnostic"], default="discovery"
    )
    parser.add_argument("--confirmation-freeze", type=Path)
    parser.add_argument(
        "--bank-selection",
        type=Path,
        help=(
            "Discovery-frozen targeted-retrieval selection.json. Required by "
            "targeted report kernels so their process-local K cannot silently "
            "fall back to the Native-thinking report reference."
        ),
    )
    parser.add_argument("--include-nonstrict", action="store_true")
    parser.add_argument("--cohort-registry", type=Path)
    parser.add_argument("kernel_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.v6_config.resolve()
    config = V6Config.load(config_path)
    kernel_args = list(args.kernel_args)
    if kernel_args[:1] == ["--"]:
        kernel_args = kernel_args[1:]
    model_option = _option(kernel_args, "--model")
    model_label = model_option[1] if model_option else None
    if model_label is not None and model_label not in config.model_labels:
        raise ValueError("Kernel model lies outside the V6 registry")
    if args.phase == "confirmation":
        if args.confirmation_freeze is None or model_label is None:
            raise ValueError(
                "Confirmation kernels require --confirmation-freeze and a kernel --model"
            )
        from realistic_niah_v6.suite import validate_confirmation_freeze

        validate_confirmation_freeze(
            args.confirmation_freeze,
            prompt_mode=config.prompt_mode,
            model_label=model_label,
        )

    config_flag = CONFIG_FLAGS.get(args.target)
    if config_flag and config_flag not in kernel_args:
        kernel_args.extend([config_flag, str(config_path)])
    if args.target == "natural-aligned-progress":
        cohort = _option(kernel_args, "--cohort-mode")
        if cohort is None:
            kernel_args.extend(["--cohort-mode", "indexed_positive_control"])
        elif cohort[1] != "indexed_positive_control":
            raise ValueError(
                "V6 natural-aligned progress must use the full enumeration "
                "registry compatibility path"
            )
    kernel_args, materialized = _materialize_generation_view(
        kernel_args,
        config=config,
        model_label=model_label,
        include_nonstrict=args.include_nonstrict,
        config_path=config_path,
        target=args.target,
        cohort_registry=args.cohort_registry,
    )
    import realistic_niah_v5.pipeline as legacy_pipeline
    import realistic_niah_v5.spec as legacy_spec

    legacy_spec.V5Config = V6Config
    legacy_pipeline.registered_records = _registered_adapter
    _registered_adapter.include_nonstrict = args.include_nonstrict
    _registered_adapter.cohort_registry = args.cohort_registry
    # Several specialized runners import V5 count-stream's ``_registered_rows``
    # rather than calling the pipeline function directly.  Install the same V6
    # slot-membership adapter used by the general count-stream wrapper so a
    # reserve source seed is not silently dropped by the legacy 1234..1263
    # membership test.  The adapter keeps row["seed"] as the true source seed.
    count_stream_seed_adapter = _install_count_stream_registered_rows_adapter(
        config_path=config_path,
        include_nonstrict=args.include_nonstrict,
        cohort_registry=args.cohort_registry,
    )
    from realistic_niah_v6.kernel import install_v6_kernel_adapters

    adapter_audit = install_v6_kernel_adapters()
    adapter_audit["specialized_enumeration_geometry"] = (
        install_v6_specialized_geometry(config.prompt_mode)
    )
    help_only = bool({"--help", "-h"} & set(kernel_args))
    selected_bank_contract = None
    if args.target in SELECTED_BANK_CONTRACT_TARGETS and not help_only:
        if args.bank_selection is None:
            raise ValueError(
                f"V6 {args.target} requires --bank-selection from discovery"
            )
        if model_label is None:
            raise ValueError("A selected-bank report kernel requires --model")
        bank_plan = _option(kernel_args, "--bank-plan")
        if bank_plan is None:
            raise ValueError("A selected-bank report kernel requires --bank-plan")
        selected_bank_contract = install_v6_selected_bank_contract(
            args.bank_selection,
            bank_plan[1],
            config=config,
            model_label=model_label,
        )
    elif args.bank_selection is not None and not help_only:
        raise ValueError(
            f"--bank-selection is not applicable to V6 target {args.target}"
        )
    kernel_args, materialized_bank_plan, bank_membership_audit = (
        _materialize_confirmation_bank_membership(
            kernel_args,
            config=config,
            config_path=config_path,
            model_label=model_label,
            phase=args.phase,
            target=args.target,
            cohort_registry=args.cohort_registry,
        )
    )
    kernel_args, materialized_role_panel, seed_role_audit = (
        _materialize_specialized_seed_role_view(
            kernel_args,
            config=config,
            config_path=config_path,
            target=args.target,
        )
    )
    specialized_slot_identity = _audit_specialized_slot_identity(
        kernel_args,
        config=config,
        target=args.target,
    )
    adapter_audit.update(
        {
            "target": args.target,
            "target_path": TARGETS[args.target],
            "v6_config": str(config_path),
            "v6_config_sha256": sha256_file(config_path),
            "prompt_mode": config.prompt_mode,
            "phase": args.phase,
            "formal_cohort": not args.include_nonstrict,
            "materialized_generation_view": (
                str(materialized.resolve()) if materialized else None
            ),
            "cohort_registry": (
                str(args.cohort_registry.resolve())
                if args.cohort_registry is not None
                else None
            ),
            "cohort_registry_sha256": (
                sha256_file(args.cohort_registry)
                if args.cohort_registry is not None
                else None
            ),
            "selected_bank_contract": selected_bank_contract,
            "specialized_bank_membership_adapter": bank_membership_audit,
            "specialized_seed_role_adapter": seed_role_audit,
            "materialized_seed_role_panel": (
                str(materialized_role_panel.resolve())
                if materialized_role_panel is not None
                else None
            ),
            "materialized_bank_membership_view": (
                str(materialized_bank_plan.resolve())
                if materialized_bank_plan is not None
                else None
            ),
            "count_stream_seed_membership_adapter": count_stream_seed_adapter,
            "specialized_slot_identity": specialized_slot_identity,
            "wrapper_argv": list(sys.argv),
            "kernel_argv": list(kernel_args),
        }
    )
    manifest_path = _adapter_manifest_path(kernel_args)
    if manifest_path is not None:
        _atomic_json(manifest_path, {**adapter_audit, "run_status": "DISPATCHED"})
    print(json.dumps(adapter_audit, indent=2, sort_keys=True), flush=True)
    target = ROOT / TARGETS[args.target]
    sys.argv = [str(target), *kernel_args]
    runpy.run_path(str(target), run_name="__main__")
    if manifest_path is not None:
        _atomic_json(manifest_path, {**adapter_audit, "run_status": "COMPLETE"})


if __name__ == "__main__":
    main()
