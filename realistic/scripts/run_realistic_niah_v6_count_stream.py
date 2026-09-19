#!/usr/bin/env python3
"""V6 adapter for the audited V5 count-stream experiment runner."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v6.pipeline import (  # noqa: E402
    read_jsonl,
    registered_records,
    sha256_file,
    validate_generation_contracts,
)
from realistic_niah_v6.spec import V6Config  # noqa: E402
from realistic_niah_v6.replacement import (  # noqa: E402
    resolved_generation_records,
)
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402
from realistic_niah_v6.answer_trace_extension import (  # noqa: E402
    coherent_slot_to_source,
    load_contract as load_answer_trace_extension_contract,
    load_relay_geometry_amendment,
    model_contract as answer_trace_model_contract,
)
from realistic_niah_v5.count_stream import (  # noqa: E402
    build_answer_source_registry,
    trace_patch_geometry_positions,
)


DEFAULT_V6_CONFIG = ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
NO_BASE_CONFIG_COMMANDS = {"plan-broad", "select-broad-k", "fit-basis", "analyze"}


def _extract_option(arguments: list[str], name: str) -> tuple[str | None, list[str]]:
    result = list(arguments)
    if name not in result:
        return None, result
    index = result.index(name)
    if index + 1 >= len(result):
        raise ValueError(f"{name} requires a value")
    value = result[index + 1]
    del result[index : index + 2]
    if name in result:
        raise ValueError(f"{name} may appear only once")
    return value, result


def _extract_flag(arguments: list[str], name: str) -> tuple[bool, list[str]]:
    result = list(arguments)
    present = name in result
    result = [value for value in result if value != name]
    return present, result


def _option_value(arguments: list[str], name: str) -> str | None:
    if name not in arguments:
        return None
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise ValueError(f"{name} requires a value")
    return arguments[index + 1]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _adapter_manifest_path(arguments: list[str]) -> Path | None:
    raw = _option_value(arguments, "--output")
    if raw is None:
        return None
    output = Path(raw)
    if output.suffix:
        return output.with_suffix(output.suffix + ".v6_adapter.json")
    return output / "v6_adapter_manifest.json"


def _validate_mechanism_config(path: Path, config: V6Config) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "realistic_niah_v6_count_stream_v1":
        raise ValueError("V6 count-stream config has the wrong schema")
    experiment_id = str(value.get("experiment_id", ""))
    if config.prompt_mode not in experiment_id:
        raise ValueError(
            "V6 count-stream experiment_id must name its enumeration prompt mode"
        )


def _registered_adapter(
    rows: Iterable[Mapping[str, Any]],
    config: V6Config,
    *,
    model_label: str | None = None,
) -> list[dict[str, Any]]:
    materialized = list(rows)
    validate_generation_contracts(
        materialized,
        config,
        model_label=model_label,
        config_sha256=_registered_adapter.config_sha256,
    )
    if _registered_adapter.cohort_registry is not None:
        return resolved_generation_records(
            materialized,
            config,
            registry_path=_registered_adapter.cohort_registry,
            model_label=model_label,
        )
    return registered_records(
        materialized,
        config,
        model_label=model_label,
        formal_only=not _registered_adapter.include_nonstrict,
    )


_registered_adapter.include_nonstrict = False
_registered_adapter.config_sha256 = None
_registered_adapter.cohort_registry = None


def _v6_registered_rows(
    args: Any,
    mechanism: Any,
    *,
    legacy: Any,
) -> list[dict[str, Any]]:
    """Apply frozen panel membership by analysis slot without aliasing seeds.

    A strict replacement row has a real source seed outside the original
    20/10 registry.  V5 used the seed itself both for panel membership and as
    the statistical identity.  V6 separates those roles: the frozen original
    slot determines discovery/confirmation and broad-panel membership, while
    ``row['seed']`` remains the true source seed in every numerical result.
    """

    config = V6Config.load(args.v5_config)
    legacy._validate_seed_contract(config, mechanism)
    rows = _registered_adapter(
        read_jsonl(args.generations),
        config,
        model_label=args.model,
    )
    selected: list[dict[str, Any]] = []
    exclusion_counts: dict[str, int] = {}
    row_panel = str(getattr(args, "row_panel", "all"))
    broad_phase_by_panel = {
        "broad_ranking": "ranking_discovery",
        "broad_k_selection": "k_selection_discovery",
        "broad_confirmation": "confirmation",
    }
    broad_phase = broad_phase_by_panel.get(row_panel)
    if (
        broad_phase in {"ranking_discovery", "k_selection_discovery"}
        and args.seed_role != "development"
    ):
        raise ValueError(f"{row_panel} requires --seed-role development")
    if broad_phase == "confirmation" and args.seed_role != "confirmation":
        raise ValueError("broad_confirmation requires --seed-role confirmation")
    replacement_rows = 0
    for row in rows:
        source_seed = int(row["seed"])
        slot_seed = int(row.get("v6_analysis_slot_seed", source_seed))
        role = mechanism.seed_role(slot_seed)
        if role != args.seed_role:
            continue
        if broad_phase is not None:
            if mechanism.broad_phase(slot_seed) != broad_phase:
                continue
            allowed_counts = mechanism.broad_counts_for_seed(
                slot_seed, phase=broad_phase
            )
            if int(row.get("gold_count", 0)) not in set(allowed_counts):
                continue
        exclusion = legacy._cohort_exclusion_reason(row, args.cohort)
        if exclusion is not None:
            exclusion_counts[exclusion] = exclusion_counts.get(exclusion, 0) + 1
            continue
        active = dict(row)
        active["mechanism_split"] = role
        active["mechanism_cohort"] = args.cohort
        active["v6_panel_membership_seed"] = slot_seed
        if bool(active.get("v6_replacement_applied")):
            replacement_rows += 1
        selected.append(active)
    if args.seed_role == "confirmation" and not mechanism.formal_inference_eligible:
        raise ValueError(
            "The mechanism config has no frozen fresh-confirmation registry"
        )
    selected.sort(
        key=lambda row: (
            int(row.get("v6_analysis_slot_seed", row["seed"])),
            int(row.get("gold_count", 0)),
            int(row["seed"]),
        )
    )
    if getattr(args, "limit", None) is not None:
        if int(args.limit) < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: int(args.limit)]
    args.cohort_audit = {
        "cohort": args.cohort,
        "parser_source": "active_v6_parser_over_frozen_output",
        "eligible_rows": len(selected),
        "exclusion_counts": exclusion_counts,
        "limit": getattr(args, "limit", None),
        "row_panel": row_panel,
        "broad_phase": broad_phase,
        "replacement_rows": replacement_rows,
        "panel_membership_identity": "v6_analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
    }
    if not selected:
        raise ValueError(f"No {args.seed_role} rows remain for {args.model}")
    return selected


def _registry_identity_by_request() -> dict[str, dict[str, Any]]:
    path = _registered_adapter.cohort_registry
    if path is None:
        return {}
    rows = read_jsonl(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = str(row["source_request_id"])
        if request_id in output:
            raise ValueError(f"Duplicate cohort-registry request: {request_id}")
        output[request_id] = dict(row)
    return output


def _v6_load_count_stream_capture_dataset(
    capture_index: str | Path,
    *,
    site_kinds: Any = None,
    original: Any,
):
    metadata, states = original(capture_index, site_kinds=site_kinds)
    registry = _registry_identity_by_request()
    if not registry:
        raise ValueError(
            "V6 count-basis fitting requires a resolved cohort registry"
        )
    request_ids = metadata["request_id"].astype(str)
    missing = sorted(set(request_ids) - set(registry))
    if missing:
        raise ValueError(
            "Capture metadata contains requests outside the resolved registry: "
            f"{missing[:20]}"
        )
    metadata = metadata.copy()
    metadata["v6_analysis_slot_seed"] = request_ids.map(
        lambda request_id: int(registry[request_id]["analysis_slot_seed"])
    )
    metadata["v6_source_seed"] = request_ids.map(
        lambda request_id: int(registry[request_id]["source_seed"])
    )
    metadata["v6_replacement_applied"] = request_ids.map(
        lambda request_id: bool(registry[request_id]["replacement_applied"])
    )
    if not metadata["seed"].astype(int).eq(
        metadata["v6_source_seed"].astype(int)
    ).all():
        raise ValueError("Capture metadata aliases a replacement source seed")
    return metadata, states


def _v6_capture_answer_source_attention(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    row: Mapping[str, Any],
    *,
    answer_site_id: str,
    layers: Any,
    original: Any,
):
    frame, registry = original(
        model,
        tokenizer,
        adapter,
        row,
        answer_site_id=answer_site_id,
        layers=layers,
    )
    source_seed = int(row["seed"])
    slot_seed = int(row.get("v6_analysis_slot_seed", source_seed))
    frame = frame.copy()
    frame["v6_analysis_slot_seed"] = slot_seed
    frame["v6_source_seed"] = source_seed
    frame["v6_replacement_applied"] = bool(
        row.get("v6_replacement_applied", False)
    )
    registry = dict(registry)
    registry.update(
        {
            "v6_analysis_slot_seed": slot_seed,
            "v6_source_seed": source_seed,
            "v6_replacement_applied": bool(
                row.get("v6_replacement_applied", False)
            ),
            "v6_seed_aliasing": False,
        }
    )
    return frame, registry


def _v6_run_answer_broad_head_trial(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    row: Mapping[str, Any],
    *,
    original: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    result = dict(original(model, tokenizer, adapter, row, **kwargs))
    source_seed = int(row["seed"])
    result.update(
        {
            "v6_analysis_slot_seed": int(
                row.get("v6_analysis_slot_seed", source_seed)
            ),
            "v6_source_seed": source_seed,
            "v6_replacement_applied": bool(
                row.get("v6_replacement_applied", False)
            ),
            "v6_seed_aliasing": False,
        }
    )
    return result


def _v6_command_fit_basis(args: Any, *, legacy: Any) -> None:
    started = time.perf_counter()
    mechanism = legacy._spec(args)
    metadata, states = legacy.load_count_stream_capture_dataset(
        args.capture_index,
        site_kinds=[args.site_kind],
    )
    mask = legacy.count_stream_cohort_mask(metadata, args.cohort)
    membership = metadata["v6_analysis_slot_seed"].astype(int)
    mask &= membership.isin(mechanism.development_seeds).to_numpy()
    if args.capture_split is not None:
        mask &= metadata["split"].astype(str).eq(args.capture_split).to_numpy()
    if args.site_id:
        mask &= metadata["site_id"].astype(str).eq(args.site_id).to_numpy()
    available_layers = sorted(
        int(value) for value in metadata.loc[mask, "layer"].unique()
    )
    layers = available_layers if args.layers is None else sorted(set(args.layers))
    missing = sorted(set(layers) - set(available_layers))
    if missing:
        raise ValueError(f"Basis layers are unavailable: {missing}")
    arrays: dict[str, Any] = {}
    fit_rows = []
    for layer in layers:
        layer_mask = mask & metadata["layer"].astype(int).eq(layer).to_numpy()
        layer_states = legacy.np.asarray(states[layer_mask], dtype=legacy.np.float32)
        labels = metadata.loc[layer_mask, args.label].to_numpy(dtype=int)
        observed_source_seeds = sorted(
            int(value) for value in metadata.loc[layer_mask, "seed"].unique()
        )
        observed_slot_seeds = sorted(
            int(value)
            for value in metadata.loc[
                layer_mask, "v6_analysis_slot_seed"
            ].unique()
        )
        if len(legacy.np.unique(labels)) < 2:
            raise ValueError(f"L{layer} has fewer than two {args.label} classes")
        center, basis, control = legacy.fit_count_stream_basis(
            layer_states,
            labels,
            rank=args.rank,
            seed=args.random_seed + int(layer),
        )
        arrays[f"center_L{layer}"] = center
        arrays[f"basis_L{layer}"] = basis
        arrays[f"control_basis_L{layer}"] = control
        arrays[f"v6_analysis_slot_seeds_L{layer}"] = legacy.np.asarray(
            observed_slot_seeds, dtype=legacy.np.int64
        )
        arrays[f"v6_true_source_seeds_L{layer}"] = legacy.np.asarray(
            observed_source_seeds, dtype=legacy.np.int64
        )
        fit_rows.append(
            {
                "layer": int(layer),
                "observations": int(len(layer_states)),
                "labels": sorted(int(value) for value in legacy.np.unique(labels)),
                "development_seeds": observed_slot_seeds,
                "true_source_seeds": observed_source_seeds,
                "replacement_observations": int(
                    metadata.loc[layer_mask, "v6_replacement_applied"]
                    .astype(bool)
                    .sum()
                ),
                "effective_rank": int(basis.shape[1]),
                "basis_control_max_abs_dot": float(
                    legacy.np.max(legacy.np.abs(basis.T @ control))
                ),
            }
        )
    legacy._atomic_npz(args.output, **arrays)
    audit = legacy._runtime_manifest(
        args,
        mechanism=mechanism,
        started=started,
        completed_shards=0,
        extra={
            "capture_index": str(args.capture_index.resolve()),
            "capture_index_sha256": legacy._sha256(args.capture_index),
            "site_kind": args.site_kind,
            "site_id": args.site_id,
            "cohort": args.cohort,
            "capture_split": args.capture_split,
            "label": args.label,
            "selection_role": "development_only",
            "basis_membership_identity": "v6_analysis_slot_seed",
            "basis_observation_identity": "true_source_request",
            "seed_aliasing": False,
            "confirmation_used_for_fit": bool(
                metadata.loc[mask, "split"].astype(str).eq("confirmation").any()
            ),
            "fits": fit_rows,
            "artifact_sha256": legacy._sha256(args.output),
        },
    )
    legacy._atomic_json(args.output.with_suffix(".json"), audit)


def _v6_command_plan_trace_patch_confirmation(
    args: Any,
    *,
    legacy: Any,
    config: V6Config,
) -> None:
    """Build the frozen sparse trace-patch panel on held-out V6 rows.

    V5 exposes pair-plan construction only for its development split. V6
    registers the same outcome-blind topology on independently frozen
    confirmation slots, so confirmation needs a narrow adapter rather than
    pretending those rows are development data. Panel membership stays on the
    original analysis slot while numerical results keep the true source seed.
    """

    started = time.perf_counter()
    if args.seed_role != "confirmation":
        raise ValueError(
            "The V6 confirmation trace-plan adapter is confirmation-only"
        )
    mechanism = legacy._spec(args)
    rows = legacy._registered_rows(args, mechanism)
    expected_slots = {int(value) for value in config.confirmation_seeds}
    if int(mechanism.trace_patch_seeds_per_cell) != len(expected_slots):
        raise ValueError(
            "Frozen trace-patch cell size differs from the confirmation slot count"
        )

    identity_by_request: dict[str, dict[str, int]] = {}
    for row in rows:
        request_id = str(row["request_id"])
        source_seed = int(row["seed"])
        slot_seed = int(row.get("v6_analysis_slot_seed", source_seed))
        if request_id in identity_by_request:
            raise ValueError(
                f"Duplicate confirmation trace-plan request: {request_id}"
            )
        identity_by_request[request_id] = {
            "source_seed": source_seed,
            "slot_seed": slot_seed,
        }

    plan = legacy.build_sparse_trace_patch_sample_plan(
        rows,
        model_label=args.model,
        donor_offsets=mechanism.trace_patch_donor_offsets,
        seeds_per_cell=mechanism.trace_patch_seeds_per_cell,
        sampling_seed=mechanism.trace_patch_sampling_seed,
        include_count2_terminal_panel=(
            mechanism.trace_patch_include_count2_terminal_panel
        ),
        candidate_counts=mechanism.candidate_counts,
    ).copy()
    plan["v6_analysis_slot_seed"] = plan["request_id"].map(
        lambda request_id: identity_by_request[str(request_id)]["slot_seed"]
    )
    plan["v6_source_seed"] = plan["request_id"].map(
        lambda request_id: identity_by_request[str(request_id)]["source_seed"]
    )
    plan["seed_aliasing"] = False
    plan["panel_membership_identity"] = "v6_analysis_slot_seed"
    plan["statistical_identity"] = "true_source_seed"
    plan["confirmation_outcomes_used_for_selection"] = False
    if not (
        plan["seed"].astype(int) == plan["v6_source_seed"].astype(int)
    ).all():
        raise ValueError("Confirmation trace plan aliases a true source seed")
    for cell_id, frame in plan.groupby("selection_cell_id", sort=False):
        observed_slots = set(frame["v6_analysis_slot_seed"].astype(int))
        if observed_slots != expected_slots or len(frame) != len(expected_slots):
            raise ValueError(
                "Confirmation trace-plan cell lost a frozen analysis slot: "
                f"{cell_id}"
            )

    output = Path(args.output)
    plan_path = output / "trace_patch_pair_plan.csv"
    legacy._atomic_csv(plan_path, plan)
    cell_counts = (
        plan.groupby(
            ["panel_kind", "gold_count", "donor_offset"], as_index=False
        )
        .agg(
            pair_count=("pair_sha256", "nunique"),
            seed_count=("seed", "nunique"),
            analysis_slot_seed_count=("v6_analysis_slot_seed", "nunique"),
            receiver_count=("receiver_occurrence", "nunique"),
        )
        .sort_values(["panel_kind", "gold_count", "donor_offset"])
    )
    legacy._atomic_csv(output / "trace_patch_cell_counts.csv", cell_counts)
    legacy._atomic_json(
        output / "manifest.json",
        legacy._runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "schema_version": (
                    "realistic_niah_v6_confirmation_trace_patch_plan_v1"
                ),
                "status": "PASS_OUTCOME_BLIND_CONFIRMATION_PLAN",
                "selection_policy": "outcome_blind_registry_identity_hash",
                "selection_input_fields": (
                    "sampling_seed,model_label,panel_kind,gold_count,"
                    "donor_offset,true_source_seed,request_id"
                ),
                "confirmation_outcomes_used_for_selection": False,
                "panel_membership_identity": "v6_analysis_slot_seed",
                "statistical_identity": "true_source_seed",
                "registered_analysis_slots": sorted(expected_slots),
                "registered_true_source_seeds": sorted(
                    set(plan["v6_source_seed"].astype(int))
                ),
                "seed_aliasing": False,
                "pair_count": int(len(plan)),
                "local_pair_count": int(plan["panel_kind"].eq("local").sum()),
                "terminal_pair_count": int(
                    plan["panel_kind"].eq("terminal").sum()
                ),
                "cell_count": int(len(cell_counts)),
                "pair_plan": str(plan_path.resolve()),
                "pair_plan_sha256": legacy._sha256(plan_path),
            },
        ),
    )


def _v6_command_plan_broad(args: Any, *, legacy: Any) -> None:
    started = time.perf_counter()
    mechanism = legacy._spec(args)
    captures = legacy._read_capture_frames(args.captures)
    required_identity = {
        "v6_analysis_slot_seed",
        "v6_source_seed",
        "v6_replacement_applied",
    }
    missing_columns = sorted(required_identity - set(captures.columns))
    if missing_columns:
        raise ValueError(
            f"V6 broad capture lacks cohort identity columns {missing_columns}"
        )
    selection_slots = (
        mechanism.development_seeds
        if bool(args.use_all_development_seeds)
        else mechanism.broad_ranking_seeds
    )
    selected = captures.loc[
        captures["model_label"].astype(str).eq(str(args.model))
        & legacy.pd.to_numeric(
            captures["v6_analysis_slot_seed"], errors="coerce"
        ).isin(selection_slots)
    ].copy()
    coverage = selected[
        ["request_id", "v6_analysis_slot_seed", "gold_count"]
    ].drop_duplicates()
    observed_pairs = {
        (int(slot), int(count))
        for slot, count in coverage[
            ["v6_analysis_slot_seed", "gold_count"]
        ].itertuples(index=False, name=None)
    }
    expected_pairs = {
        (int(slot), int(count))
        for slot in selection_slots
        for count in mechanism.candidate_counts
    }
    if observed_pairs != expected_pairs or len(coverage) != len(expected_pairs):
        raise ValueError(
            "V6 broad ranking capture must fill each frozen slot x count cell"
        )
    ranking_input = selected.copy()
    ranking_input["seed"] = ranking_input["v6_analysis_slot_seed"].astype(int)
    ranking = legacy.rank_answer_broad_heads(
        ranking_input,
        source_group=args.source_group,
        development_seeds=selection_slots,
        model_label=args.model,
    )
    ranking["selection_aggregation"] = (
        "request_first_then_analysis_slot_equal_attention_only"
    )
    ranking["v6_selection_identity"] = "analysis_slot_seed"
    ranking["v6_statistical_identity"] = "not_applicable_attention_only_ranking"
    ranking["v6_seed_aliasing"] = False
    sizes = (
        tuple(args.bank_sizes)
        if args.bank_sizes
        else mechanism.development_bank_sizes
    )
    plans = [
        legacy.build_answer_broad_head_plan(
            ranking,
            bank_size=size,
            random_controls=mechanism.random_controls,
            random_seed=args.random_seed,
            allow_selected_random_overlap=(
                mechanism.random_control_overlap_policy
                == "nonthinking_allow_treatment_overlap"
            ),
        )
        for size in sizes
    ]
    plan = legacy.pd.concat(plans, ignore_index=True)
    output = Path(args.output)
    legacy._atomic_csv(output / "answer_broad_head_ranking.csv", ranking)
    legacy._atomic_csv(output / "answer_broad_head_plan.csv", plan)
    replacement_cells = int(
        coverage.merge(
            selected[
                [
                    "request_id",
                    "v6_analysis_slot_seed",
                    "v6_replacement_applied",
                ]
            ].drop_duplicates(),
            on=["request_id", "v6_analysis_slot_seed"],
            how="left",
        )["v6_replacement_applied"]
        .astype(bool)
        .sum()
    )
    legacy._atomic_json(
        output / "manifest.json",
        legacy._runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "captures": str(args.captures.resolve()),
                "source_group": args.source_group,
                "bank_sizes": list(sizes),
                "selection_role": "development_only",
                "ranking_seed_role": (
                    "all_development_discovery"
                    if bool(args.use_all_development_seeds)
                    else "ranking_discovery"
                ),
                "ranking_analysis_slots": list(selection_slots),
                "ranking_true_source_seeds": sorted(
                    int(value) for value in selected["v6_source_seed"].unique()
                ),
                "replacement_cells": replacement_cells,
                "ranking_aggregation": (
                    "request_first_then_analysis_slot_equal_attention_only"
                ),
                "selection_identity": "analysis_slot_seed",
                "statistical_identity": (
                    "not_applicable_attention_only_head_ranking"
                ),
                "seed_aliasing": False,
                "use_all_development_seeds": bool(
                    args.use_all_development_seeds
                ),
                "confirmation_used_for_selection": False,
                "plan_sha256": legacy._sha256(
                    output / "answer_broad_head_plan.csv"
                ),
            },
        ),
    )


def _coherent_broad_source_seeds(
    trials: Any,
    *,
    mechanism: Any,
    model_label: str,
    source_group: str,
    phase: str = "k_selection_discovery",
) -> tuple[tuple[int, ...], dict[str, int]]:
    required = {
        "experiment_id",
        "model_label",
        "source_group",
        "request_id",
        "seed",
        "gold_count",
        "v6_analysis_slot_seed",
        "v6_source_seed",
        "v6_seed_aliasing",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Coherent broad trials lack identity columns {missing}")
    selected = trials.loc[
        trials["experiment_id"].eq("answer_broad_head_ablation")
        & trials["model_label"].eq(str(model_label))
        & trials["source_group"].eq(str(source_group))
    ].copy()
    if "status" in selected.columns:
        selected = selected.loc[selected["status"].fillna("ok").eq("ok")]
    if selected.empty:
        raise ValueError("No coherent broad K-selection trials remain")
    if selected["v6_seed_aliasing"].astype(bool).any():
        raise ValueError("A coherent broad trial reports seed aliasing")
    if not selected["seed"].astype(int).eq(
        selected["v6_source_seed"].astype(int)
    ).all():
        raise ValueError("Broad trial seed differs from its true source seed")
    if phase == "k_selection_discovery":
        slots = tuple(map(int, mechanism.broad_k_selection_seeds))
    elif phase == "confirmation":
        slots = tuple(map(int, mechanism.confirmation_seeds))
    else:
        raise ValueError(f"Unknown coherent broad statistical phase: {phase}")
    source_by_slot: dict[str, int] = {}
    for slot in slots:
        frame = selected.loc[
            selected["v6_analysis_slot_seed"].astype(int).eq(slot)
        ]
        sources = set(frame["seed"].astype(int))
        if len(sources) != 1:
            raise ValueError(
                f"Broad slot {slot} mixes true source seeds: {sorted(sources)}"
            )
        observed_counts = set(frame["gold_count"].astype(int))
        expected_counts = set(
            mechanism.broad_counts_for_seed(
                slot, phase=phase
            )
        )
        if observed_counts != expected_counts:
            raise ValueError(
                f"Broad slot {slot} count coverage changed: "
                f"expected={sorted(expected_counts)} observed={sorted(observed_counts)}"
            )
        if frame[["request_id", "gold_count"]].drop_duplicates()[
            "request_id"
        ].nunique() != len(expected_counts):
            raise ValueError(f"Broad slot {slot} does not have one request/count")
        source_by_slot[str(slot)] = next(iter(sources))
    source_seeds = tuple(source_by_slot[str(slot)] for slot in slots)
    if len(set(source_seeds)) != len(source_seeds):
        raise ValueError("Two broad analysis slots share one true source seed")
    return source_seeds, source_by_slot


def _v6_native_loop_plan_for_rows(
    args: Any,
    mechanism: Any,
    rows: list[dict[str, Any]],
    *,
    legacy: Any,
    config: V6Config,
    adapter_audit: dict[str, Any],
):
    """Build the frozen V5 plan over coherent V6 true-source trajectories."""

    split = "discovery" if args.seed_role == "development" else "confirmation"
    slots = tuple(
        map(
            int,
            config.discovery_seeds
            if split == "discovery"
            else config.confirmation_seeds,
        )
    )
    required_counts = tuple(int(count) for count in config.counts if int(count) >= 2)
    by_slot_count: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        slot = int(row.get("v6_analysis_slot_seed", row["seed"]))
        count = int(row.get("gold_count", 0))
        if slot in set(slots) and count in set(required_counts):
            by_slot_count.setdefault((slot, count), []).append(row)
    source_by_slot: dict[int, int] = {}
    for slot in slots:
        active: list[dict[str, Any]] = []
        for count in required_counts:
            matches = by_slot_count.get((slot, count), [])
            if len(matches) != 1:
                raise ValueError(
                    "Native-loop coherent registry does not contain exactly one "
                    f"row for slot={slot}, count={count}"
                )
            active.extend(matches)
        sources = {int(row["seed"]) for row in active}
        if len(sources) != 1:
            raise ValueError(
                f"Native-loop analysis slot {slot} mixes true source seeds: "
                f"{sorted(sources)}"
            )
        source_by_slot[slot] = next(iter(sources))
    source_seeds = tuple(source_by_slot[slot] for slot in slots)
    if len(set(source_seeds)) != len(source_seeds):
        raise ValueError("Two native-loop slots share one true source seed")

    plan_rows = rows
    if (
        mechanism.experiment_id
        == "realistic_niah_v5_full_commit_specificity_confirmation_v1"
    ):
        plan_rows = [row for row in rows if int(row.get("gold_count", 0)) >= 4]
    adapter_audit["native_loop_seed_identity"] = {
        "status": "PASS_TRUE_SOURCE_COHERENT_TRAJECTORIES",
        "split": split,
        "analysis_slots": list(slots),
        "required_counts": list(required_counts),
        "analysis_slot_to_true_source_seed": {
            str(slot): int(source_by_slot[slot]) for slot in slots
        },
        "true_source_seeds": list(source_seeds),
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
    }
    return legacy.build_fixed_native_loop_plan(
        plan_rows,
        model_label=args.model,
        seeds=source_seeds,
        seed_role=args.seed_role,
        donor_offsets=tuple(int(value) for value in args.donor_offsets),
        candidate_counts=tuple(range(2, 11)),
        sampling_seed=int(args.random_seed),
        require_all_seeds_per_offset=not bool(args.allow_incomplete_offsets),
        include_boundaries=not bool(args.no_boundaries),
    )


def _v6_command_select_broad_k(args: Any, *, legacy: Any) -> None:
    started = time.perf_counter()
    mechanism = legacy._spec(args)
    rows = [
        row
        for file in legacy._trial_files(args.trials)
        for row in legacy.read_jsonl(file)
    ]
    trials = legacy.pd.DataFrame(rows)
    source_seeds, source_by_slot = _coherent_broad_source_seeds(
        trials,
        mechanism=mechanism,
        model_label=args.model,
        source_group=args.source_group,
    )
    curve, seed_effects, decision = legacy.select_answer_broad_bank_size(
        trials,
        model_label=args.model,
        source_group=args.source_group,
        expected_seeds=source_seeds,
        expected_bank_sizes=mechanism.development_bank_sizes,
        expected_requests_per_seed=mechanism.broad_panel_counts_per_seed,
        expected_random_controls=mechanism.random_controls,
        boundary_extension_bank_size=mechanism.boundary_extension_bank_size,
        bootstrap_samples=mechanism.bootstrap_samples,
        random_seed=args.random_seed,
    )
    curve["v6_statistical_identity"] = "true_source_seed"
    curve["v6_seed_aliasing"] = False
    seed_effects["v6_statistical_identity"] = "true_source_seed"
    seed_effects["v6_seed_aliasing"] = False
    decision["v6_analysis_slot_to_true_source_seed"] = source_by_slot
    decision["v6_statistical_identity"] = "true_source_seed"
    decision["v6_seed_aliasing"] = False
    decision["decision_sha256"] = legacy.hashlib.sha256(
        json.dumps(
            {key: value for key, value in decision.items() if key != "decision_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    output = Path(args.output)
    legacy._atomic_csv(output / "k_discovery_curve.csv", curve)
    legacy._atomic_csv(output / "k_discovery_seed_effects.csv", seed_effects)
    legacy._atomic_json(output / "k_selection_decision.json", decision)
    frozen_plan_path: Path | None = None
    selected_k = decision.get("selected_bank_size")
    if decision["status"] == "frozen_for_confirmation" and selected_k is not None:
        plan = legacy._load_head_plan(
            args.plan, model=args.model, bank_size=int(selected_k)
        ).copy()
        if set(plan["source_group"].astype(str)) != {str(args.source_group)}:
            raise ValueError("K decision and head plan use different source groups")
        plan["ranking_split"] = plan.get("selection_split", "ranking_discovery")
        plan["selection_split"] = "k_selection_discovery"
        plan["k_selection_status"] = "frozen_for_confirmation"
        plan["k_selection_decision_sha256"] = str(decision["decision_sha256"])
        plan["confirmation_locked"] = True
        frozen_plan_path = output / "frozen_answer_broad_head_plan.csv"
        legacy._atomic_csv(frozen_plan_path, plan)
    legacy._atomic_json(
        output / "manifest.json",
        legacy._runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "model_label": args.model,
                "source_group": args.source_group,
                "selection_phase": "k_selection_discovery",
                "analysis_slots": list(mechanism.broad_k_selection_seeds),
                "true_source_seeds": list(source_seeds),
                "analysis_slot_to_true_source_seed": source_by_slot,
                "statistical_identity": "true_source_seed",
                "seed_aliasing": False,
                "confirmation_outcomes_used": False,
                "decision": decision,
                "frozen_plan": (
                    str(frozen_plan_path.resolve()) if frozen_plan_path else None
                ),
                "frozen_plan_sha256": (
                    legacy._sha256(frozen_plan_path) if frozen_plan_path else None
                ),
            },
        ),
    )


def _v6_analysis_outcome_for_experiment(
    trials: Any,
    *,
    experiment_id: str,
    requested_outcome: str,
    legacy: Any,
) -> tuple[str, dict[str, Any] | None]:
    """Route the sparse terminal panel to its registered answer endpoint.

    The frozen sparse trace plan contains a local next-city panel and a count-2
    terminal panel.  The latter intentionally has no next-city target and is
    marked ``terminal_panel_answer_only`` by the numerical kernel.  The V5 CLI
    accepts only one ``--outcome`` for every requested experiment, so applying
    the local city outcome to both panels drops every terminal row.  V6 keeps
    the local registered outcome and routes only the topology-marked terminal
    experiment to the Native-thinking final-answer endpoint.
    """

    if not (
        str(experiment_id) == "trace_terminal_state_patching"
        and str(requested_outcome) == "donor_vs_receiver_city_log_odds"
    ):
        return str(requested_outcome), None

    terminal = trials.loc[
        trials["experiment_id"].astype(str).eq(str(experiment_id))
    ].copy()
    if terminal.empty:
        raise ValueError("The registered terminal trace panel is empty")
    required = {
        "terminal_panel_answer_only",
        "local_next_city_outcome_registered",
        "final_answer_outcome_registered",
        "correct_count_margin",
    }
    missing = sorted(required - set(terminal.columns))
    if missing:
        raise ValueError(
            f"Terminal trace rows lack registered endpoint fields {missing}"
        )
    if not terminal["terminal_panel_answer_only"].fillna(False).astype(bool).all():
        raise ValueError("A terminal trace row is not marked answer-only")
    if terminal["local_next_city_outcome_registered"].fillna(False).astype(bool).any():
        raise ValueError("A terminal trace row unexpectedly registers next-city")
    if not terminal["final_answer_outcome_registered"].fillna(False).astype(bool).all():
        raise ValueError("A terminal trace row lacks the final-answer registry flag")
    requested = legacy.pd.to_numeric(
        terminal.get(requested_outcome), errors="coerce"
    )
    if legacy.np.isfinite(requested).any():
        raise ValueError(
            "The answer-only terminal panel unexpectedly contains the local "
            "next-city outcome"
        )
    terminal_outcome = "correct_count_margin"
    answer_values = legacy.pd.to_numeric(
        terminal[terminal_outcome], errors="coerce"
    )
    finite_answer_count = int(legacy.np.isfinite(answer_values).sum())
    if finite_answer_count != len(terminal):
        raise ValueError(
            "The terminal trace panel does not have a finite registered "
            "correct-count margin for every row"
        )
    return terminal_outcome, {
        "schema_version": "realistic_niah_v6_trace_analysis_outcome_adapter_v1",
        "status": "PASS_REGISTERED_TERMINAL_ANSWER_OUTCOME_ROUTING",
        "experiment_id": str(experiment_id),
        "requested_shared_outcome": str(requested_outcome),
        "effective_outcome": terminal_outcome,
        "terminal_row_count": int(len(terminal)),
        "finite_effective_outcome_count": finite_answer_count,
        "selection_basis": (
            "experiment topology and frozen registry flags only; effect "
            "magnitudes are not used"
        ),
        "local_panel_outcome_changed": False,
        "model_trials_recomputed": False,
        "sample_failure": False,
        "seed_replacement_triggered": False,
    }


def _v6_command_analyze(args: Any, *, legacy: Any) -> None:
    """Run frozen V5 contrasts with V6's registered per-panel outcomes."""

    started = time.perf_counter()
    mechanism = legacy._spec(args)
    rows = [
        row
        for file in legacy._trial_files(args.trials)
        for row in legacy.read_jsonl(file)
    ]
    trials = legacy.pd.DataFrame(rows)
    experiments = (
        args.experiment_ids
        if args.experiment_ids
        else sorted(trials["experiment_id"].dropna().astype(str).unique())
    )
    summaries = []
    effects = []
    effective_outcomes: dict[str, str] = {}
    outcome_adapters: list[dict[str, Any]] = []
    for experiment_id in experiments:
        outcome, outcome_adapter = _v6_analysis_outcome_for_experiment(
            trials,
            experiment_id=str(experiment_id),
            requested_outcome=str(args.outcome),
            legacy=legacy,
        )
        effective_outcomes[str(experiment_id)] = outcome
        if outcome_adapter is not None:
            outcome_adapters.append(outcome_adapter)
        strata = list(args.strata)
        if (
            experiment_id == "answer_broad_head_ablation"
            and "source_group" in trials.columns
            and "source_group" not in strata
        ):
            strata.append("source_group")
        if (
            experiment_id == "answer_broad_head_ablation"
            and "bank_size" in trials.columns
            and trials.loc[
                trials["experiment_id"].eq(experiment_id), "bank_size"
            ].nunique(dropna=True)
            > 1
            and "bank_size" not in strata
        ):
            strata.append("bank_size")
        if (
            experiment_id
            in {
                "trace_intermediate_state_patching",
                "trace_terminal_state_patching",
            }
            and "donor_direction" in trials.columns
            and "donor_direction" not in strata
        ):
            strata.append("donor_direction")
        summary, seed_effects = legacy.summarize_linear_contrasts(
            trials,
            experiment_id=experiment_id,
            outcome=outcome,
            bootstrap_samples=mechanism.bootstrap_samples,
            random_seed=args.random_seed,
            stratum_columns=strata,
        )
        summaries.append(summary)
        effects.append(seed_effects)
    combined_summary = legacy.pd.concat(summaries, ignore_index=True)
    combined_effects = legacy.pd.concat(effects, ignore_index=True)
    ledger = legacy.mechanism_decision_ledger(combined_summary)
    ledger["design_status"] = mechanism.status
    ledger["formal_inference_eligible"] = mechanism.formal_inference_eligible
    ledger["claim_scope"] = (
        "fresh_confirmation"
        if mechanism.formal_inference_eligible
        else "development_only_no_confirmatory_claim"
    )
    output = Path(args.output)
    legacy._atomic_csv(output / "estimands.csv", combined_summary)
    legacy._atomic_csv(output / "seed_effects.csv", combined_effects)
    legacy._atomic_csv(output / "mechanism_decision_ledger.csv", ledger)
    trial_files = legacy._trial_files(args.trials)
    legacy._atomic_json(
        output / "manifest.json",
        legacy._runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "trial_files": [str(path.resolve()) for path in trial_files],
                "experiment_ids": list(experiments),
                "requested_outcome": str(args.outcome),
                "effective_outcomes_by_experiment": effective_outcomes,
                "v6_outcome_adapters": outcome_adapters,
                "estimands_sha256": legacy._sha256(output / "estimands.csv"),
                "claim_policy": (
                    "Decision ledger reports registered component gates only; "
                    "it never upgrades them to a unique-circuit or scalar-counter claim."
                ),
            },
        ),
    )


def _v6_command_terminal_relay_mediation(
    args: Any,
    *,
    legacy: Any,
    config: V6Config,
    extension_contract_path: Path,
    relay_geometry_amendment_path: Path | None,
) -> None:
    """Run the frozen V5 relay kernel with explicit V6 seed identities."""

    started = time.perf_counter()
    if args.seed_role != "confirmation":
        raise ValueError("The V6 terminal relay extension is confirmation-only")
    if getattr(args, "limit", None) is not None:
        raise ValueError("Formal V6 relay mediation does not permit row limits")
    contract = load_answer_trace_extension_contract(extension_contract_path)
    relay_geometry_amendment = (
        load_relay_geometry_amendment(
            relay_geometry_amendment_path,
            extension_contract_path=extension_contract_path,
        )
        if relay_geometry_amendment_path is not None
        else None
    )
    frozen = answer_trace_model_contract(
        contract,
        prompt_mode=config.prompt_mode,
        model_label=args.model,
        relay_geometry_amendment=relay_geometry_amendment,
    )
    if (int(args.source_layer), int(args.relay_layer)) != (
        int(frozen["relay_source_layer"]),
        int(frozen["relay_layer"]),
    ):
        raise ValueError("Terminal relay layers differ from the frozen V6 extension")
    if str(args.geometry) != frozen["relay_geometry"]:
        raise ValueError("Terminal relay geometry differs from the frozen V6 extension")

    mechanism = legacy._spec(args)
    rows = legacy._registered_rows(args, mechanism)
    expected_slots = [int(value) for value in config.confirmation_seeds]
    source_by_slot = coherent_slot_to_source(rows, expected_slots=expected_slots)
    request_to_slot = {
        str(row["request_id"]): int(
            row.get("v6_analysis_slot_seed", row["seed"])
        )
        for row in rows
    }
    request_to_source = {
        str(row["request_id"]): int(row["seed"]) for row in rows
    }
    if len(request_to_slot) != len(rows):
        raise ValueError("V6 terminal relay cohort contains duplicate request IDs")

    pair_plan = legacy.build_terminal_serial_pair_plan(
        rows, model_label=args.model
    )
    pair_plan["v6_analysis_slot_seed"] = pair_plan["request_id"].map(
        request_to_slot
    )
    pair_plan["v6_source_seed"] = pair_plan["request_id"].map(request_to_source)
    pair_plan["seed_aliasing"] = False
    if pair_plan["v6_analysis_slot_seed"].isna().any():
        raise ValueError("V6 terminal relay plan lost an analysis-slot identity")
    if set(pair_plan["v6_analysis_slot_seed"].astype(int)) != set(expected_slots):
        raise ValueError("V6 terminal relay plan lost a frozen confirmation slot")
    expected_sources = set(source_by_slot.values())
    if set(pair_plan["seed"].astype(int)) != expected_sources:
        raise ValueError("V6 terminal relay plan lost a true source seed")
    if not (
        pair_plan["seed"].astype(int) == pair_plan["v6_source_seed"].astype(int)
    ).all():
        raise ValueError("V6 terminal relay plan aliases seed identity")

    output = Path(args.output)
    pair_plan_path = output / "terminal_relay_pair_plan.csv"
    legacy._atomic_csv(pair_plan_path, pair_plan)
    cell_counts = (
        pair_plan.groupby(["gold_count", "donor_offset"], as_index=False)
        .agg(
            pair_count=("pair_sha256", "nunique"),
            seed_count=("v6_analysis_slot_seed", "nunique"),
        )
        .sort_values(["gold_count", "donor_offset"])
    )
    legacy._atomic_csv(output / "terminal_relay_cell_counts.csv", cell_counts)
    model, tokenizer, adapter = legacy._model(args)
    row_by_request = {str(row["request_id"]): row for row in rows}
    geometry_reason = (
        "not applicable: a trace item is shorter than the requested "
        f"{args.geometry} geometry"
    )
    geometry_rows: list[dict[str, Any]] = []
    for pair in pair_plan.itertuples(index=False):
        row = row_by_request[str(pair.request_id)]
        _encoding, registry = build_answer_source_registry(
            row, tokenizer, answer_site_id=mechanism.answer_site_id
        )
        receiver_span = registry.trace_items[int(pair.receiver_occurrence) - 1]
        donor_span = registry.trace_items[int(pair.donor_occurrence) - 1]
        receiver_length = int(receiver_span[1]) - int(receiver_span[0])
        donor_length = int(donor_span[1]) - int(donor_span[0])
        try:
            receiver_positions, donor_positions, geometry_audit = (
                trace_patch_geometry_positions(
                    registry,
                    receiver_occurrence=int(pair.receiver_occurrence),
                    donor_occurrence=int(pair.donor_occurrence),
                    geometry=str(args.geometry),
                )
            )
            status = "eligible"
            exclusion_reason = None
            realized_width = len(receiver_positions)
            if len(donor_positions) != realized_width:
                raise RuntimeError("Relay geometry preflight produced unequal widths")
        except ValueError as exc:
            if str(exc) != geometry_reason:
                raise
            status = "not_applicable"
            exclusion_reason = str(exc)
            realized_width = 0
            geometry_audit = {}
        geometry_rows.append(
            {
                "pair_sha256": str(pair.pair_sha256),
                "request_id": str(pair.request_id),
                "v6_analysis_slot_seed": int(pair.v6_analysis_slot_seed),
                "v6_source_seed": int(pair.v6_source_seed),
                "receiver_occurrence": int(pair.receiver_occurrence),
                "donor_occurrence": int(pair.donor_occurrence),
                "receiver_token_length": receiver_length,
                "donor_token_length": donor_length,
                "requested_geometry": str(args.geometry),
                "realized_width": realized_width,
                "status": status,
                "exclusion_reason": exclusion_reason,
                "geometry_capped_by_shorter_span": bool(
                    geometry_audit.get("capped_by_shorter_span", False)
                ),
            }
        )
    geometry_frame = pd.DataFrame(geometry_rows).sort_values(
        ["v6_analysis_slot_seed", "request_id", "receiver_occurrence", "donor_occurrence"]
    )
    geometry_path = output / "terminal_relay_geometry_eligibility.csv"
    legacy._atomic_csv(geometry_path, geometry_frame)
    eligible_geometry = geometry_frame.loc[geometry_frame["status"].eq("eligible")]
    eligible_sources = {
        int(value) for value in eligible_geometry["v6_source_seed"].tolist()
    }
    full_na_sources = sorted(expected_sources - eligible_sources)
    geometry_audit_path = output / "terminal_relay_geometry_eligibility_audit.json"
    legacy._atomic_json(
        geometry_audit_path,
        {
            "schema_version": (
                "realistic_niah_v6_terminal_relay_geometry_eligibility_audit_v1"
            ),
            "status": "PASS_OUTCOME_BLIND_GEOMETRY_AUDIT",
            "prompt_mode": config.prompt_mode,
            "model_label": args.model,
            "patch_geometry": str(args.geometry),
            "planned_pair_count": int(len(geometry_frame)),
            "eligible_pair_count": int(len(eligible_geometry)),
            "not_applicable_pair_count": int(
                geometry_frame["status"].eq("not_applicable").sum()
            ),
            "planned_seed_count": len(expected_sources),
            "eligible_seed_count": len(eligible_sources),
            "geometry_not_applicable_full_seed_count": len(full_na_sources),
            "geometry_not_applicable_full_seeds": full_na_sources,
            "intervention_loop_started_before_audit": False,
            "intervention_outcomes_used_for_geometry_choice": False,
            "per_pair_geometry_adaptation": False,
            "pair_plan_sha256": legacy._sha256(pair_plan_path),
            "extension_contract_sha256": sha256_file(extension_contract_path),
            "relay_geometry_amendment_sha256": (
                sha256_file(relay_geometry_amendment_path)
                if relay_geometry_amendment_path is not None
                else None
            ),
            "eligibility_csv_sha256": legacy._sha256(geometry_path),
        },
    )
    shard_dir = legacy._prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = not_applicable = 0
    for pair_index, pair in enumerate(pair_plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        stem = legacy._safe_stem(
            row["request_id"],
            "terminal_relay",
            pair.receiver_occurrence,
            pair.donor_occurrence,
            args.source_layer,
            args.relay_layer,
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        try:
            results = legacy.run_terminal_state_relay_reset_trials(
                model,
                tokenizer,
                adapter,
                row,
                receiver_occurrence=int(pair.receiver_occurrence),
                donor_occurrence=int(pair.donor_occurrence),
                source_layer=int(args.source_layer),
                relay_layer=int(args.relay_layer),
                geometry=str(args.geometry),
                answer_site_id=mechanism.answer_site_id,
                run_greedy=not args.skip_greedy,
                max_new_tokens=int(args.max_new_tokens),
            )
        except ValueError as exc:
            if "not applicable" not in str(exc).lower():
                raise
            not_applicable += 1
            results = [
                {
                    "schema_version": "realistic_niah_v6_count_stream_trial_v1",
                    "experiment_id": "terminal_state_pre_answer_relay_mediation",
                    "source_condition": source_condition,
                    "relay_condition": relay_condition,
                    "status": "not_applicable",
                    "exclusion_reason": str(exc),
                    "request_id": row["request_id"],
                    "model_label": args.model,
                    "seed": int(row["seed"]),
                    "gold_count": int(pair.gold_count),
                    "source_layer": int(args.source_layer),
                    "relay_layer": int(args.relay_layer),
                    "patch_geometry": str(args.geometry),
                    "receiver_occurrence": int(pair.receiver_occurrence),
                    "donor_occurrence": int(pair.donor_occurrence),
                    "donor_offset": int(pair.donor_offset),
                }
                for source_condition in ("self_patch", "full_donor_patch")
                for relay_condition in legacy.REGISTERED_RELAY_RESET_CONDITIONS
            ]
        for result in results:
            observed_seed = int(result.get("seed", row["seed"]))
            if observed_seed != int(row["seed"]):
                raise ValueError("V6 terminal relay result aliases true source seed")
            result.update(
                {
                    "schema_version": "realistic_niah_v6_count_stream_trial_v1",
                    "prompt_mode": config.prompt_mode,
                    "seed": int(row["seed"]),
                    "v6_source_seed": int(row["seed"]),
                    "v6_analysis_slot_seed": int(pair.v6_analysis_slot_seed),
                    "seed_aliasing": False,
                    "mechanism_split": args.seed_role,
                    "selection_policy": str(pair.selection_policy),
                    "selection_cell_id": str(pair.selection_cell_id),
                    "within_cell_index": int(pair.within_cell_index),
                    "eligible_seed_count": int(pair.eligible_seed_count),
                    "pair_sha256": str(pair.pair_sha256),
                    "pair_plan": str(pair_plan_path.resolve()),
                    "pair_plan_sha256": legacy._sha256(pair_plan_path),
                    "geometry_eligibility_audit_sha256": legacy._sha256(
                        geometry_audit_path
                    ),
                    "extension_contract_sha256": sha256_file(
                        extension_contract_path
                    ),
                    "relay_geometry_amendment_sha256": (
                        sha256_file(relay_geometry_amendment_path)
                        if relay_geometry_amendment_path is not None
                        else None
                    ),
                    "relay_scientific_label": frozen["relay_scientific_label"],
                    "relay_original_geometry": frozen["relay_original_geometry"],
                    "suffix4_intervention_outcomes_used_for_selection": False,
                }
            )
        legacy._atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[v6 count-stream terminal-relay] {pair_index}/{len(pair_plan)}",
            flush=True,
        )

    rows_per_pair = 2 * len(legacy.REGISTERED_RELAY_RESET_CONDITIONS)
    legacy._atomic_json(
        output / "manifest.json",
        legacy._runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "schema_version": "realistic_niah_v6_terminal_relay_run_v1",
                "run_status": "PASS_COMPLETE",
                "prompt_mode": config.prompt_mode,
                "newly_completed": completed,
                "resume_skipped": skipped,
                "not_applicable_pairs_this_run": not_applicable,
                "seed_role": args.seed_role,
                "registered_analysis_slots": expected_slots,
                "registered_true_source_seeds": sorted(expected_sources),
                "analysis_slot_to_true_source_seed": {
                    str(slot): int(source)
                    for slot, source in source_by_slot.items()
                },
                "seed_aliasing": False,
                "selection_policy": "all_eligible_frozen_confirmation_slots",
                "intervention_outcomes_used_for_selection": False,
                "pair_count": int(len(pair_plan)),
                "cell_count": int(len(cell_counts)),
                "rows_per_pair": rows_per_pair,
                "planned_trial_rows": int(len(pair_plan) * rows_per_pair),
                "pair_plan": str(pair_plan_path.resolve()),
                "pair_plan_sha256": legacy._sha256(pair_plan_path),
                "geometry_eligibility_audit": str(geometry_audit_path.resolve()),
                "geometry_eligibility_audit_sha256": legacy._sha256(
                    geometry_audit_path
                ),
                "source_layer": int(args.source_layer),
                "source_patch_layers": list(
                    range(int(args.source_layer), int(args.relay_layer))
                ),
                "relay_layer": int(args.relay_layer),
                "patch_geometry": str(args.geometry),
                "source_conditions": ["self_patch", "full_donor_patch"],
                "relay_conditions": list(
                    legacy.REGISTERED_RELAY_RESET_CONDITIONS
                ),
                "extension_contract": str(extension_contract_path.resolve()),
                "extension_contract_sha256": sha256_file(
                    extension_contract_path
                ),
                "relay_geometry_amendment": (
                    str(relay_geometry_amendment_path.resolve())
                    if relay_geometry_amendment_path is not None
                    else None
                ),
                "relay_geometry_amendment_sha256": (
                    sha256_file(relay_geometry_amendment_path)
                    if relay_geometry_amendment_path is not None
                    else None
                ),
                "relay_original_geometry": frozen["relay_original_geometry"],
                "relay_scientific_label": frozen["relay_scientific_label"],
                "original_suffix8_artifacts_preserved": bool(
                    frozen["relay_original_artifacts_preserved"]
                ),
                "protocol_relation": contract["protocol_relation"],
                "claim_scope": (
                    "terminal_state_to_pre_answer_residual_relay_to_answer_"
                    "partial_mediation"
                ),
            },
        ),
    )
def main() -> None:
    raw = sys.argv[1:]
    v6_path_text, raw = _extract_option(raw, "--v6-config")
    freeze_text, raw = _extract_option(raw, "--confirmation-freeze")
    cohort_text, raw = _extract_option(raw, "--cohort-registry")
    extension_contract_text, raw = _extract_option(raw, "--extension-contract")
    relay_geometry_amendment_text, raw = _extract_option(
        raw, "--relay-geometry-amendment"
    )
    include_nonstrict, raw = _extract_flag(raw, "--include-nonstrict")
    cohort_registry = Path(cohort_text) if cohort_text else None
    if include_nonstrict and cohort_registry is not None:
        raise ValueError("Resolved replacement cohorts are strict-only")
    config_path = Path(v6_path_text) if v6_path_text else DEFAULT_V6_CONFIG
    config = V6Config.load(config_path)
    if raw and raw[0] == "terminal-relay-mediation":
        if extension_contract_text is None:
            raise ValueError(
                "V6 terminal-relay-mediation requires --extension-contract"
            )
        extension_contract_path = Path(extension_contract_text)
        extension_contract = load_answer_trace_extension_contract(
            extension_contract_path
        )
        relay_geometry_amendment_path = (
            Path(relay_geometry_amendment_text)
            if relay_geometry_amendment_text is not None
            else None
        )
        relay_geometry_amendment = (
            load_relay_geometry_amendment(
                relay_geometry_amendment_path,
                extension_contract_path=extension_contract_path,
            )
            if relay_geometry_amendment_path is not None
            else None
        )
        answer_trace_model_contract(
            extension_contract,
            prompt_mode=config.prompt_mode,
            model_label=str(_option_value(raw, "--model")),
            relay_geometry_amendment=relay_geometry_amendment,
        )
    else:
        if extension_contract_text is not None:
            raise ValueError(
                "--extension-contract is only valid for terminal-relay-mediation"
            )
        if relay_geometry_amendment_text is not None:
            raise ValueError(
                "--relay-geometry-amendment is only valid for "
                "terminal-relay-mediation"
            )
        extension_contract_path = None
        relay_geometry_amendment_path = None

    mechanism_text = _option_value(raw, "--mechanism-config")
    mechanism_path = (
        Path(mechanism_text)
        if mechanism_text
        else ROOT
        / "configs"
        / f"realistic_niah_v6_{config.prompt_mode}_count_stream_dev.json"
    )
    _validate_mechanism_config(mechanism_path, config)
    if mechanism_text is None:
        raw.extend(["--mechanism-config", str(mechanism_path)])

    model_label = _option_value(raw, "--model")
    seed_role = _option_value(raw, "--seed-role")
    if seed_role == "confirmation":
        if model_label is None:
            raise ValueError("Confirmation count-stream run requires --model")
        if freeze_text is None:
            raise ValueError(
                "Confirmation is locked; supply --confirmation-freeze after "
                "discovery choices are frozen"
            )
        validate_confirmation_freeze(
            Path(freeze_text),
            prompt_mode=config.prompt_mode,
            model_label=model_label,
        )

    import realistic_niah_v5.pipeline as legacy_pipeline
    import realistic_niah_v5.spec as legacy_spec

    legacy_spec.V5Config = V6Config
    legacy_pipeline.registered_records = _registered_adapter

    from realistic_niah_v6.kernel import install_v6_kernel_adapters
    from realistic_niah_v6.count_stream import (
        install_v6_count_stream_panel_adapter,
    )

    adapter_audit = install_v6_kernel_adapters()
    count_stream_panel_adapter = install_v6_count_stream_panel_adapter()
    adapter_audit.update(
        {
            "command": raw[0] if raw else None,
            "prompt_mode": config.prompt_mode,
            "model_label": model_label,
            "formal_cohort": not include_nonstrict,
            "v6_config": str(config_path.resolve()),
            "v6_config_sha256": sha256_file(config_path),
            "mechanism_config": str(mechanism_path.resolve()),
            "mechanism_config_sha256": sha256_file(mechanism_path),
            "count_stream_panel_adapter": count_stream_panel_adapter,
            "cohort_registry": (
                str(cohort_registry.resolve()) if cohort_registry else None
            ),
            "cohort_registry_sha256": (
                sha256_file(cohort_registry) if cohort_registry else None
            ),
            "wrapper_argv": list(sys.argv),
            "legacy_argv": list(raw),
            "extension_contract": (
                str(extension_contract_path.resolve())
                if extension_contract_path is not None
                else None
            ),
            "extension_contract_sha256": (
                sha256_file(extension_contract_path)
                if extension_contract_path is not None
                else None
            ),
            "relay_geometry_amendment": (
                str(relay_geometry_amendment_path.resolve())
                if relay_geometry_amendment_path is not None
                else None
            ),
            "relay_geometry_amendment_sha256": (
                sha256_file(relay_geometry_amendment_path)
                if relay_geometry_amendment_path is not None
                else None
            ),
        }
    )
    import run_realistic_niah_v5_count_stream as legacy

    legacy.V5Config = V6Config
    legacy.registered_records = _registered_adapter
    legacy._registered_rows = lambda args, mechanism: _v6_registered_rows(
        args, mechanism, legacy=legacy
    )
    legacy.DEFAULT_V5_CONFIG = config_path
    legacy.DEFAULT_MECHANISM_CONFIG = mechanism_path
    original_capture_answer_source_attention = (
        legacy.capture_answer_source_attention
    )
    original_run_answer_broad_head_trial = legacy.run_answer_broad_head_trial
    original_load_count_stream_capture_dataset = (
        legacy.load_count_stream_capture_dataset
    )
    legacy.capture_answer_source_attention = (
        lambda model, tokenizer, adapter, row, **kwargs: (
            _v6_capture_answer_source_attention(
                model,
                tokenizer,
                adapter,
                row,
                original=original_capture_answer_source_attention,
                **kwargs,
            )
        )
    )
    legacy.run_answer_broad_head_trial = (
        lambda model, tokenizer, adapter, row, **kwargs: (
            _v6_run_answer_broad_head_trial(
                model,
                tokenizer,
                adapter,
                row,
                original=original_run_answer_broad_head_trial,
                **kwargs,
            )
        )
    )
    legacy.load_count_stream_capture_dataset = lambda path, **kwargs: (
        _v6_load_count_stream_capture_dataset(
            path,
            original=original_load_count_stream_capture_dataset,
            **kwargs,
        )
    )
    legacy.command_fit_basis = lambda args: _v6_command_fit_basis(
        args, legacy=legacy
    )
    legacy.command_plan_broad = lambda args: _v6_command_plan_broad(
        args, legacy=legacy
    )
    legacy.command_select_broad_k = lambda args: _v6_command_select_broad_k(
        args, legacy=legacy
    )
    legacy.command_analyze = lambda args: _v6_command_analyze(
        args, legacy=legacy
    )
    if raw and raw[0] == "plan-trace-patch" and seed_role == "confirmation":
        if cohort_registry is None:
            raise ValueError(
                "V6 confirmation trace planning requires a frozen cohort registry"
            )
        legacy.command_plan_trace_patch = lambda args: (
            _v6_command_plan_trace_patch_confirmation(
                args,
                legacy=legacy,
                config=config,
            )
        )
    if raw and raw[0] == "terminal-relay-mediation":
        if cohort_registry is None:
            raise ValueError(
                "V6 terminal relay requires a frozen coherent cohort registry"
            )
        assert extension_contract_path is not None
        legacy.command_terminal_relay_mediation = lambda args: (
            _v6_command_terminal_relay_mediation(
                args,
                legacy=legacy,
                config=config,
                extension_contract_path=extension_contract_path,
                relay_geometry_amendment_path=relay_geometry_amendment_path,
            )
        )
    if raw and raw[0] in {
        "plan-native-loop",
        "p0-native-loop",
        "boundary-native-loop",
    }:
        if cohort_registry is None:
            raise ValueError(
                "V6 native-loop commands require a coherent cohort registry"
            )
        legacy._native_loop_plan_for_rows = lambda args, mechanism, rows: (
            _v6_native_loop_plan_for_rows(
                args,
                mechanism,
                rows,
                legacy=legacy,
                config=config,
                adapter_audit=adapter_audit,
            )
        )
    adapter_audit["seed_identity_adapters"] = {
        "basis_fit": {
            "membership": "analysis_slot_seed",
            "observation": "true_source_request",
        },
        "broad_attention_ranking": {
            "selection_unit": "analysis_slot_seed",
            "behavioral_outcomes_used": False,
        },
        "broad_k_selection": {
            "required_registry": "true_source_coherent_five_count_panel",
            "statistical_identity": "true_source_seed",
        },
        "confirmation_trace_patch_plan": {
            "selection": "outcome_blind_frozen_confirmation_registry",
            "panel_membership_identity": "analysis_slot_seed",
            "statistical_identity": "true_source_seed",
            "confirmation_outcomes_used_for_selection": False,
        },
        "native_loop": {
            "required_registry": "true_source_coherent_count_2_to_10_trajectory",
            "panel_membership_identity": "analysis_slot_seed",
            "statistical_identity": "true_source_seed",
        },
        "seed_aliasing": False,
    }
    _registered_adapter.include_nonstrict = include_nonstrict
    _registered_adapter.config_sha256 = sha256_file(config_path)
    _registered_adapter.cohort_registry = cohort_registry

    # Keep the legacy subcommand grammar exactly, translating only the config
    # option name.  All numerical estimands and intervention code stay frozen.
    existing_v5 = _option_value(raw, "--v5-config")
    if existing_v5 is not None:
        index = raw.index("--v5-config")
        raw[index + 1] = str(config_path)
    elif raw and raw[0] not in NO_BASE_CONFIG_COMMANDS:
        raw.extend(["--v5-config", str(config_path)])
    sys.argv = [sys.argv[0], *raw]
    print(json.dumps(adapter_audit, indent=2, sort_keys=True), flush=True)
    manifest_path = _adapter_manifest_path(raw)
    if manifest_path is not None:
        _atomic_json(
            manifest_path, {**adapter_audit, "run_status": "DISPATCHED"}
        )
    legacy.main()
    if manifest_path is not None:
        _atomic_json(manifest_path, {**adapter_audit, "run_status": "COMPLETE"})


if __name__ == "__main__":
    main()
