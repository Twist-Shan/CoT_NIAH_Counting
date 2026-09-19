#!/usr/bin/env python3
"""Expose the audited V5 causal CLI under V6 enumeration contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from realistic_niah_v6.replacement import (  # noqa: E402
    SELECTED_CELL_SCHEMA_VERSION,
    resolved_generation_records,
)
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402


ALLOWED_COMMANDS = {
    "causal-plan",
    "answer-query-causal-plan",
    "pre-city-causal-plan",
    "response-reference-causal-plan",
    "causal-source-writes",
    "causal-heads",
    "causal-source-edge",
    "causal-heads-behavior",
    "causal-pre-city-heads",
    "causal-pre-city-all-sites",
    "causal-marker-needle-patch",
    "causal-answer-query-heads",
    "causal-response-reference-heads",
    "causal-tokens",
    "causal-context",
    "causal-writes",
    "causal-subspace-fit",
    "causal-subspace",
    "causal-patch",
    "causal-analyze",
}

COMMANDS_WITH_CONFIG = {
    "causal-plan",
    "answer-query-causal-plan",
    "pre-city-causal-plan",
    "response-reference-causal-plan",
    "causal-source-writes",
    "causal-heads",
    "causal-source-edge",
    "causal-heads-behavior",
    "causal-pre-city-heads",
    "causal-pre-city-all-sites",
    "causal-tokens",
    "causal-context",
    "causal-writes",
    "causal-subspace-fit",
    "causal-subspace",
    "causal-patch",
    "causal-analyze",
}

PLAN_VALIDATION_COMMANDS = {
    "causal-heads",
    "causal-source-edge",
    "causal-heads-behavior",
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


class _CausalSeedMembershipConfig:
    """Delegate the frozen V6 config while widening legacy seed membership.

    The V5 causal implementation treats ``causal_*_seeds`` both as panel
    membership filters and as the true seed identity used for cross-fitting.
    A resolved V6 replacement cell must therefore add its *source* seed to the
    corresponding membership set.  This proxy never changes ``discovery_seeds``
    or ``confirmation_seeds`` and never aliases a replacement source seed to
    its original analysis slot.
    """

    def __init__(
        self,
        base: V6Config,
        *,
        development_seeds: Iterable[int],
        confirmation_seeds: Iterable[int],
        audit: Mapping[str, Any],
    ) -> None:
        self._base = base
        self.causal_development_seeds = tuple(
            sorted({int(value) for value in development_seeds})
        )
        self.causal_confirmation_seeds = tuple(
            sorted({int(value) for value in confirmation_seeds})
        )
        self._audit = dict(audit)
        if set(self.causal_development_seeds) & set(
            self.causal_confirmation_seeds
        ):
            raise ValueError(
                "Effective V6 causal development and confirmation seeds overlap"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def validate(self) -> None:
        # Validate the immutable on-disk protocol with the real V6 class.  The
        # two effective membership sets above are a process-local compatibility
        # view, not a mutation of the frozen experiment config.
        self._base.validate()

    def to_dict(self) -> dict[str, Any]:
        payload = self._base.to_dict()
        payload["causal_development_seeds"] = list(
            self.causal_development_seeds
        )
        payload["causal_confirmation_seeds"] = list(
            self.causal_confirmation_seeds
        )
        payload["v6_causal_seed_membership_adapter"] = dict(self._audit)
        return payload


def _validated_causal_membership_registry(
    config: V6Config,
    source: Path,
    *,
    model_label: str | None,
    original_development: tuple[int, ...],
    original_confirmation: tuple[int, ...],
) -> dict[str, Any]:
    registry_path = source.resolve()
    rows = read_jsonl(registry_path)
    if not rows:
        raise ValueError("Resolved V6 causal cohort registry is empty")
    if any(
        row.get("schema_version") != SELECTED_CELL_SCHEMA_VERSION for row in rows
    ):
        raise ValueError("Resolved V6 causal cohort registry has the wrong schema")
    modes = {str(row.get("prompt_mode")) for row in rows}
    if modes != {config.prompt_mode}:
        raise ValueError("Resolved V6 causal registry prompt mode changed")
    models = {str(row.get("model_label")) for row in rows}
    if len(models) != 1 or (model_label is not None and models != {model_label}):
        raise ValueError("Resolved V6 causal registry model does not match")
    roles = {str(row.get("split")) for row in rows}
    if len(roles) != 1 or not roles <= {"discovery", "confirmation"}:
        raise ValueError("Resolved V6 causal registry must contain one seed role")
    role = next(iter(roles))
    slot_seeds = (
        original_development if role == "discovery" else original_confirmation
    )
    expected_slots = {
        (int(count), int(seed))
        for count in config.counts
        for seed in slot_seeds
    }
    observed_slots = [
        (int(row["gold_count"]), int(row["analysis_slot_seed"])) for row in rows
    ]
    if (
        len(observed_slots) != len(expected_slots)
        or set(observed_slots) != expected_slots
    ):
        raise ValueError(
            "Resolved V6 causal registry does not fill each analysis slot once"
        )
    source_request_ids = [str(row["source_request_id"]) for row in rows]
    if len(source_request_ids) != len(set(source_request_ids)):
        raise ValueError("Resolved V6 causal registry reuses one source request")
    for row in rows:
        source_seed = int(row["source_seed"])
        slot_seed = int(row["analysis_slot_seed"])
        replacement_applied = bool(row["replacement_applied"])
        if replacement_applied != (source_seed != slot_seed):
            raise ValueError(
                "Resolved V6 replacement flag disagrees with source/slot seeds"
            )
    true_source_seeds = sorted({int(row["source_seed"]) for row in rows})
    return {
        "path": registry_path,
        "sha256": sha256_file(registry_path),
        "role": role,
        "rows": rows,
        "row_count": len(rows),
        "analysis_slot_count": len(expected_slots),
        "true_source_seeds": true_source_seeds,
    }


def _causal_seed_membership_config(
    config: V6Config,
    *,
    cohort_registry: Path | None,
    model_label: str | None,
    additional_cohort_registries: Iterable[Path] = (),
) -> tuple[V6Config | _CausalSeedMembershipConfig, dict[str, Any]]:
    """Build an audited true-source-seed view for legacy causal filters."""

    original_development = tuple(map(int, config.causal_development_seeds))
    original_confirmation = tuple(map(int, config.causal_confirmation_seeds))
    base_audit: dict[str, Any] = {
        "status": "NOT_APPLIED_NO_RESOLVED_COHORT",
        "purpose": "legacy_causal_seed_membership_compatibility",
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "generation_seed_field": "true_source_seed_unchanged",
        "seed_aliasing": False,
        "frozen_config_file_mutated": False,
        "original_causal_development_seeds": list(original_development),
        "original_causal_confirmation_seeds": list(original_confirmation),
        "effective_causal_development_seeds": list(original_development),
        "effective_causal_confirmation_seeds": list(original_confirmation),
        "added_true_source_seeds": [],
        "added_true_source_seeds_by_role": {
            "discovery": [],
            "confirmation": [],
        },
        "registry_role": None,
        "registry_roles": [],
        "cohort_registries": [],
    }
    registry_sources: list[Path] = []
    for raw in (
        ([cohort_registry] if cohort_registry is not None else [])
        + list(additional_cohort_registries)
    ):
        resolved = raw.resolve()
        if resolved not in registry_sources:
            registry_sources.append(resolved)
    if not registry_sources:
        return config, base_audit
    registries = [
        _validated_causal_membership_registry(
            config,
            source,
            model_label=model_label,
            original_development=original_development,
            original_confirmation=original_confirmation,
        )
        for source in registry_sources
    ]
    roles = [str(value["role"]) for value in registries]
    if len(roles) != len(set(roles)):
        raise ValueError(
            "At most one resolved V6 causal membership registry is allowed per role"
        )
    effective_development = set(original_development)
    effective_confirmation = set(original_confirmation)
    added_by_role: dict[str, list[int]] = {
        "discovery": [],
        "confirmation": [],
    }
    for registry in registries:
        role = str(registry["role"])
        true_source_seeds = set(map(int, registry["true_source_seeds"]))
        if role == "discovery":
            effective_development.update(true_source_seeds)
            added_by_role[role] = sorted(
                true_source_seeds - set(original_development)
            )
        else:
            effective_confirmation.update(true_source_seeds)
            added_by_role[role] = sorted(
                true_source_seeds - set(original_confirmation)
            )
    added = sorted(added_by_role["discovery"] + added_by_role["confirmation"])
    registry_audits = [
        {
            "path": str(value["path"]),
            "sha256": str(value["sha256"]),
            "role": str(value["role"]),
            "rows": int(value["row_count"]),
            "analysis_slot_count": int(value["analysis_slot_count"]),
            "true_source_seed_count": len(value["true_source_seeds"]),
        }
        for value in registries
    ]
    audit = {
        **base_audit,
        "status": "APPLIED_TRUE_SOURCE_SEED_MEMBERSHIP",
        "cohort_registry": (
            str(registries[0]["path"]) if len(registries) == 1 else None
        ),
        "cohort_registry_sha256": (
            str(registries[0]["sha256"]) if len(registries) == 1 else None
        ),
        "cohort_registries": registry_audits,
        "registry_role": roles[0] if len(roles) == 1 else "discovery+confirmation",
        "registry_roles": sorted(roles),
        "registry_rows": sum(int(value["row_count"]) for value in registries),
        "analysis_slot_count": sum(
            int(value["analysis_slot_count"]) for value in registries
        ),
        "true_source_seed_count": sum(
            len(value["true_source_seeds"]) for value in registries
        ),
        "added_true_source_seeds": added,
        "added_true_source_seeds_by_role": added_by_role,
        "effective_causal_development_seeds": sorted(effective_development),
        "effective_causal_confirmation_seeds": sorted(effective_confirmation),
        "membership_effect": (
            "legacy filters and seed-based cross-fitting use actual source seeds; "
            "analysis slots remain separately recorded in the resolved registry"
        ),
    }
    proxy = _CausalSeedMembershipConfig(
        config,
        development_seeds=effective_development,
        confirmation_seeds=effective_confirmation,
        audit=audit,
    )
    return proxy, audit


def _route_legacy_causal_seed_role(
    config: V6Config | _CausalSeedMembershipConfig,
    audit: Mapping[str, Any],
    *,
    command: str,
    phase: str,
) -> tuple[V6Config | _CausalSeedMembershipConfig, dict[str, Any]]:
    """Translate V6 phase names for legacy commands with a fixed dev filter.

    ``causal-source-writes`` and ``causal-heads`` predate an explicit
    evaluation-split argument and always filter rows through
    ``causal_development_seeds``. During V6 confirmation their input is already
    the frozen confirmation registry, so the legacy name must point at the
    effective confirmation true-source seeds. This is a process-local
    role-vocabulary swap only: source seed identities, analysis slots, frozen
    banks, and the on-disk V6 config remain unchanged.
    """

    result = dict(audit)
    routing = {
        "status": "NOT_REQUIRED",
        "command": command,
        "phase": phase,
        "legacy_filter_field": None,
        "source_seed_identities_changed": False,
        "analysis_slot_seeds_changed": False,
        "frozen_config_file_mutated": False,
        "confirmation_intervention_outcomes_read": False,
    }
    legacy_development_filter_commands = {
        "causal-source-writes",
        "causal-heads",
    }
    if not (
        phase == "confirmation"
        and command in legacy_development_filter_commands
    ):
        result["legacy_command_seed_role_routing"] = routing
        return config, result

    cohort_development = tuple(map(int, config.causal_development_seeds))
    cohort_confirmation = tuple(map(int, config.causal_confirmation_seeds))
    if not cohort_confirmation:
        raise ValueError(
            "V6 confirmation source writes have no effective confirmation seeds"
        )
    if set(cohort_development) & set(cohort_confirmation):
        raise ValueError(
            "V6 confirmation source-write role routing would overlap seed roles"
        )

    routing.update(
        {
            "status": "APPLIED_CONFIRMATION_TO_LEGACY_DEVELOPMENT_FILTER",
            "legacy_filter_field": "causal_development_seeds",
            "v6_source_role": "confirmation",
            "cohort_effective_development_seeds": list(cohort_development),
            "cohort_effective_confirmation_seeds": list(cohort_confirmation),
            "legacy_effective_development_seeds": list(cohort_confirmation),
            "legacy_effective_confirmation_seeds": list(cohort_development),
            "reason": (
                f"the inherited {command} command has no split flag and "
                "always reads causal_development_seeds"
            ),
        }
    )
    result.update(
        {
            "cohort_effective_causal_development_seeds": list(
                cohort_development
            ),
            "cohort_effective_causal_confirmation_seeds": list(
                cohort_confirmation
            ),
            "effective_causal_development_seeds": list(cohort_confirmation),
            "effective_causal_confirmation_seeds": list(cohort_development),
            "legacy_command_seed_role_routing": routing,
        }
    )
    proxy = _CausalSeedMembershipConfig(
        config,
        development_seeds=cohort_confirmation,
        confirmation_seeds=cohort_development,
        audit=result,
    )
    return proxy, result


def _materialize_generations(
    arguments: list[str],
    *,
    config: V6Config,
    config_path: Path,
    model_label: str | None,
    include_nonstrict: bool,
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
        raise ValueError("No V6 rows remain in the requested causal cohort")
    digest = hashlib.sha256(
        (
            f"{sha256_file(source)}|{sha256_file(config_path)}|"
            f"{model_label}|{include_nonstrict}|"
            f"{sha256_file(cohort_registry) if cohort_registry else 'primary'}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    view = (
        ROOT
        / "work"
        / "realistic_niah_v6_adapter_inputs"
        / f"causal__{config.mode_slug}__{model_label or 'all'}__{digest}.jsonl"
    )
    write_jsonl(view, rows)
    result = list(arguments)
    result[index + 1] = str(view)
    return result, view


def _materialize_confirmation_plan_membership(
    arguments: list[str],
    *,
    config: V6Config,
    config_path: Path,
    model_label: str | None,
    command: str,
    phase: str,
    registry_sources: Iterable[Path],
) -> tuple[list[str], Path | None, dict[str, Any]]:
    """Route replacement source seeds through a frozen confirmation plan.

    Frozen V5 plans store canonical analysis-slot seeds in
    ``validation_seeds``.  If a strict confirmation cell is replaced, the
    selected bank is unchanged; only the actual source seed must be admitted to
    the same validation fold.  This derived CSV view performs exactly that
    membership expansion and fails closed if one source seed would map to
    multiple folds.
    """

    audit: dict[str, Any] = {
        "status": "NOT_APPLIED",
        "purpose": "confirmation_plan_true_source_seed_routing",
        "phase": phase,
        "command": command,
        "validation_seed_routing_only": True,
        "selected_heads_or_scores_changed": False,
        "statistical_identity": "true_source_seed",
        "panel_membership_identity": "analysis_slot_seed",
        "seed_aliasing": False,
        "confirmation_intervention_outcomes_read": False,
    }
    located = _option(arguments, "--plan")
    if phase != "confirmation" or command not in PLAN_VALIDATION_COMMANDS:
        return arguments, None, audit
    if located is None:
        raise ValueError(f"V6 confirmation {command} requires an explicit --plan")

    unique_sources: list[Path] = []
    for raw in registry_sources:
        resolved = raw.resolve()
        if resolved not in unique_sources:
            unique_sources.append(resolved)
    registries = [
        _validated_causal_membership_registry(
            config,
            source,
            model_label=model_label,
            original_development=tuple(map(int, config.causal_development_seeds)),
            original_confirmation=tuple(map(int, config.causal_confirmation_seeds)),
        )
        for source in unique_sources
    ]
    confirmation = [
        value for value in registries if str(value["role"]) == "confirmation"
    ]
    if len(confirmation) != 1:
        raise ValueError(
            "A V6 confirmation plan command requires exactly one confirmation "
            "cohort registry"
        )
    confirmation_registry = confirmation[0]
    replacements = sorted(
        {
            (int(row["analysis_slot_seed"]), int(row["source_seed"]))
            for row in confirmation_registry["rows"]
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
                "confirmation_registry": str(confirmation_registry["path"]),
                "confirmation_registry_sha256": str(
                    confirmation_registry["sha256"]
                ),
            },
        )

    plan_index, raw_plan = located
    plan = Path(raw_plan).resolve()
    with plan.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        plan_rows = list(reader)
    if not plan_rows or "validation_seeds" not in fieldnames:
        raise ValueError(
            "Frozen confirmation plan has no rows or validation_seeds column"
        )
    original_nonmembership = [
        {key: value for key, value in row.items() if key != "validation_seeds"}
        for row in plan_rows
    ]
    folds_by_source: dict[int, set[str]] = {
        source_seed: set() for _slot_seed, source_seed in replacements
    }
    routed_pairs: set[tuple[int, int]] = set()
    for row_index, row in enumerate(plan_rows):
        try:
            validation = {int(value) for value in json.loads(row["validation_seeds"])}
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid validation_seeds in frozen plan row {row_index}"
            ) from error
        additions = {
            source_seed
            for slot_seed, source_seed in replacements
            if slot_seed in validation
        }
        for slot_seed, source_seed in replacements:
            if slot_seed in validation:
                routed_pairs.add((slot_seed, source_seed))
        if additions:
            fold = str(row.get("fold", "0"))
            for source_seed in additions:
                folds_by_source[source_seed].add(fold)
            row["validation_seeds"] = json.dumps(sorted(validation | additions))
    missing_pairs = sorted(set(replacements) - routed_pairs)
    if missing_pairs:
        raise ValueError(
            "Frozen confirmation plan does not route replacement slots: "
            f"{missing_pairs}"
        )
    ambiguous = {
        str(seed): sorted(folds)
        for seed, folds in folds_by_source.items()
        if len(folds) != 1
    }
    if ambiguous:
        raise ValueError(
            "A confirmation replacement source seed maps to multiple folds: "
            f"{ambiguous}"
        )
    observed_nonmembership = [
        {key: value for key, value in row.items() if key != "validation_seeds"}
        for row in plan_rows
    ]
    if observed_nonmembership != original_nonmembership:
        raise RuntimeError("Confirmation plan adapter changed non-membership fields")

    identity = (
        f"{sha256_file(plan)}|{sha256_file(config_path)}|"
        f"{confirmation_registry['sha256']}|{model_label}|{command}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    view = (
        ROOT
        / "work"
        / "realistic_niah_v6_adapter_inputs"
        / (
            f"causal_plan_membership__{config.mode_slug}__"
            f"{model_label or 'all'}__{digest}.csv"
        )
    )
    view.parent.mkdir(parents=True, exist_ok=True)
    temporary = view.with_name(f".{view.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(plan_rows)
    temporary.replace(view)
    result = list(arguments)
    result[plan_index + 1] = str(view)
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
            "confirmation_registry": str(confirmation_registry["path"]),
            "confirmation_registry_sha256": str(confirmation_registry["sha256"]),
            "replacement_slot_to_source_seed": [
                {"analysis_slot_seed": slot, "source_seed": source}
                for slot, source in replacements
            ],
            "added_true_source_seeds": sorted(folds_by_source),
            "source_seed_fold": {
                str(seed): next(iter(folds))
                for seed, folds in sorted(folds_by_source.items())
            },
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


def _install_causal_plan_directory_adapter() -> dict[str, Any]:
    """Route sharded source-write directories to the source-write planner.

    The underlying source-write planner already streams ``shards/*.jsonl``.
    Its public V5 dispatcher, however, probes every input with
    ``pandas.read_csv(..., nrows=0)`` before dispatch and therefore rejects a
    directory.  V6 source writes are intentionally sharded directories, so
    patch only that dispatcher boundary and leave the planner unchanged.
    """
    import realistic_niah_v5.causal as causal

    current = causal.build_causal_plan
    if bool(getattr(current, "_v6_source_write_directory_adapter", False)):
        return {
            "status": "ALREADY_INSTALLED",
            "directory_dispatch": "shards/*.jsonl",
        }
    source_planner = getattr(causal, "_build_source_write_causal_plan", None)
    if source_planner is None:
        raise RuntimeError("V6 requires the audited source-write causal planner")

    def build_causal_plan(
        source_csv: str | Path,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> dict[str, Path]:
        source = Path(source_csv)
        if not source.is_dir():
            return current(source_csv, output_dir, **kwargs)
        shards = sorted((source / "shards").glob("*.jsonl"))
        if not shards:
            raise ValueError(f"Source-write directory has no JSONL shards: {source}")
        first_record: dict[str, Any] | None = None
        with shards[0].open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Source-write shard row is not an object: {shards[0]}")
                    first_record = value
                    break
        if first_record is None:
            raise ValueError(f"First source-write shard is empty: {shards[0]}")
        if "source_specific_ov_write_norm" not in first_record:
            raise ValueError(
                "A sharded causal-plan input must be a source-specific write bank"
            )
        return source_planner(source, output_dir, **kwargs)

    build_causal_plan._v6_source_write_directory_adapter = True  # type: ignore[attr-defined]
    causal.build_causal_plan = build_causal_plan
    return {
        "status": "INSTALLED",
        "directory_dispatch": "shards/*.jsonl",
        "legacy_dispatch_unchanged_for_files": True,
        "scientific_rows_rewritten": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["discovery", "confirmation", "diagnostic"], default="discovery"
    )
    parser.add_argument("--confirmation-freeze", type=Path)
    parser.add_argument(
        "--model-label",
        choices=("Qwen3-8B", "Gemma4-E4B"),
        help=(
            "V6 registry label for commands such as causal-plan whose legacy "
            "subparser infers the model from an input artifact."
        ),
    )
    parser.add_argument("--include-nonstrict", action="store_true")
    parser.add_argument("--cohort-registry", type=Path)
    parser.add_argument(
        "--causal-membership-registry",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional resolved registry used only to expand true-source-seed "
            "membership. This is needed when a confirmation command must retain "
            "the frozen discovery replacement identities as well."
        ),
    )
    parser.add_argument("legacy_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.v6_config.resolve()
    config = V6Config.load(config_path)
    legacy_args = list(args.legacy_args)
    if legacy_args[:1] == ["--"]:
        legacy_args = legacy_args[1:]
    if not legacy_args or legacy_args[0] not in ALLOWED_COMMANDS:
        raise ValueError(
            f"First legacy argument must be one of {sorted(ALLOWED_COMMANDS)}"
        )
    command = legacy_args[0]
    model_option = _option(legacy_args, "--model")
    legacy_model_label = model_option[1] if model_option else None
    if (
        args.model_label is not None
        and legacy_model_label is not None
        and args.model_label != legacy_model_label
    ):
        raise ValueError("Top-level and legacy model labels disagree")
    model_label = args.model_label or legacy_model_label
    if model_label is not None and model_label not in config.model_labels:
        raise ValueError("Causal model lies outside the V6 registry")
    if args.phase == "confirmation":
        if args.confirmation_freeze is None or model_label is None:
            raise ValueError(
                "Confirmation causal runs require --confirmation-freeze and --model"
            )
        validate_confirmation_freeze(
            args.confirmation_freeze,
            prompt_mode=config.prompt_mode,
            model_label=model_label,
        )
    if command == "causal-plan" and not ({"--help", "-h"} & set(legacy_args)):
        if model_label is None:
            raise ValueError(
                "V6 causal-plan requires top-level --model-label so its "
                "model-specific K grid can be audited"
            )
        bank_option = _option(legacy_args, "--bank-size")
        if bank_option is None:
            raise ValueError(
                "V6 causal-plan requires explicit --bank-size from the registered "
                "model-specific grid; no cross-model default is allowed"
            )
        try:
            bank_size = int(bank_option[1])
        except ValueError as error:
            raise ValueError("--bank-size must be an integer") from error
        allowed_grid = config.targeted_bank_grid(model_label)
        if bank_size not in allowed_grid:
            raise ValueError(
                f"V6 {model_label} causal-plan K={bank_size} is outside "
                f"the registered grid {allowed_grid}"
            )
    if command in COMMANDS_WITH_CONFIG:
        existing = _option(legacy_args, "--config")
        if existing is None:
            legacy_args.extend(["--config", str(config_path)])
        else:
            legacy_args[existing[0] + 1] = str(config_path)
    legacy_args, materialized = _materialize_generations(
        legacy_args,
        config=config,
        config_path=config_path,
        model_label=model_label,
        include_nonstrict=args.include_nonstrict,
        cohort_registry=args.cohort_registry,
    )
    membership_registry_sources = (
        ([args.cohort_registry] if args.cohort_registry is not None else [])
        + list(args.causal_membership_registry)
    )
    causal_config, seed_membership_audit = _causal_seed_membership_config(
        config,
        cohort_registry=args.cohort_registry,
        model_label=model_label,
        additional_cohort_registries=args.causal_membership_registry,
    )
    causal_config, seed_membership_audit = _route_legacy_causal_seed_role(
        causal_config,
        seed_membership_audit,
        command=command,
        phase=args.phase,
    )
    legacy_args, materialized_plan, plan_membership_audit = (
        _materialize_confirmation_plan_membership(
            legacy_args,
            config=config,
            config_path=config_path,
            model_label=model_label,
            command=command,
            phase=args.phase,
            registry_sources=membership_registry_sources,
        )
    )

    import realistic_niah_v5.pipeline as legacy_pipeline
    import realistic_niah_v5.spec as legacy_spec

    legacy_spec.V5Config = V6Config
    legacy_pipeline.registered_records = _registered_adapter
    _registered_adapter.include_nonstrict = args.include_nonstrict
    _registered_adapter.cohort_registry = args.cohort_registry
    from realistic_niah_v6.kernel import install_v6_kernel_adapters

    adapter = install_v6_kernel_adapters()
    causal_plan_directory_adapter = (
        _install_causal_plan_directory_adapter()
        if command == "causal-plan"
        else {"status": "NOT_REQUIRED"}
    )
    adapter.update(
        {
            "command": command,
            "prompt_mode": config.prompt_mode,
            "phase": args.phase,
            "formal_cohort": not args.include_nonstrict,
            "v6_config": str(config_path),
            "v6_config_sha256": sha256_file(config_path),
            "model_label": model_label,
            "registered_targeted_bank_grid": (
                list(config.targeted_bank_grid(model_label)) if model_label else None
            ),
            "materialized_generation_view": str(materialized) if materialized else None,
            "materialized_plan_membership_view": (
                str(materialized_plan) if materialized_plan else None
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
            "causal_seed_membership_adapter": seed_membership_audit,
            "causal_plan_membership_adapter": plan_membership_audit,
            "causal_plan_directory_adapter": causal_plan_directory_adapter,
            "additional_causal_membership_registries": [
                str(path.resolve()) for path in args.causal_membership_registry
            ],
            "wrapper_argv": list(sys.argv),
            "legacy_argv": list(legacy_args),
        }
    )
    print(json.dumps(adapter, indent=2, sort_keys=True), flush=True)
    manifest_path = _adapter_manifest_path(legacy_args)
    if manifest_path is not None:
        _atomic_json(manifest_path, {**adapter, "run_status": "DISPATCHED"})
    import run_realistic_niah_v5 as legacy

    legacy.V5Config = V6Config
    legacy.registered_records = _registered_adapter
    legacy._config = lambda _args: causal_config
    sys.argv = [sys.argv[0], *legacy_args]
    legacy.main()
    if manifest_path is not None:
        _atomic_json(manifest_path, {**adapter, "run_status": "COMPLETE"})


if __name__ == "__main__":
    main()
