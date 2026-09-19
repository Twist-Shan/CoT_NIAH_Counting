#!/usr/bin/env python3
"""Run the native-thinking count-stream mechanism experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.count_stream import (  # noqa: E402
    COUNT_STREAM_SCHEMA_VERSION,
    REGISTERED_COMPLEMENTARY_RELAY_CONDITIONS,
    REGISTERED_COMPLEMENTARY_SOURCE_CONDITIONS,
    REGISTERED_HEAD_READOUT_CONDITIONS,
    REGISTERED_HTML_ALIGNED_SPAN_CONDITIONS,
    REGISTERED_HIGHER_IS_BETTER_OUTCOMES,
    REGISTERED_MASK_CONDITIONS,
    REGISTERED_RELAY_RESET_CONDITIONS,
    REGISTERED_RESTORATION_CONDITIONS,
    REGISTERED_STREAM_CONDITIONS,
    REGISTERED_TRACE_FULL_STATE_CONDITIONS,
    REGISTERED_TRACE_PATCH_GEOMETRIES,
    REGISTERED_TRACE_PATCH_LAYER_MODES,
    REGISTERED_TRACE_PATCH_CONDITIONS,
    NativeCountMechanismSpec,
    build_answer_broad_head_plan,
    build_sparse_trace_patch_sample_plan,
    build_terminal_serial_pair_plan,
    build_terminal_last_trace_patch_sample_plan,
    capture_answer_source_attention,
    count_stream_cohort_mask,
    fit_count_stream_basis,
    load_count_stream_capture_dataset,
    mechanism_decision_ledger,
    rank_answer_broad_heads,
    run_answer_broad_head_trial,
    run_answer_source_mask_trial,
    run_count_state_answer_source_factorial_trials,
    run_full_state_patch_answer_source_factorial_trials,
    run_full_state_patch_head_readout_factorial_trials,
    run_html_aligned_terminal_span_trials,
    run_stream_state_trial,
    run_terminal_state_complementary_readout_trials,
    run_terminal_state_relay_reset_trials,
    run_trace_full_state_patch_trials,
    run_trace_intermediate_patch_trials,
    run_trace_terminal_patch_trials,
    run_trace_restoration_trials,
    stream_state_retention_metrics,
    select_answer_broad_bank_size,
    summarize_linear_contrasts,
)
from realistic_niah_v5.integrated_bridge import (  # noqa: E402
    INTEGRATED_BRIDGE_READOUT_CONDITIONS,
    run_integrated_serial_bridge_trials,
)
from realistic_niah_v5.native_loop import (  # noqa: E402
    AVAILABLE_P0_LOOP_CONDITIONS,
    REGISTERED_BOUNDARY_CONDITIONS,
    REGISTERED_FULL_COMMIT_SPECIFICITY_CONDITIONS,
    REGISTERED_P0_LOOP_CONDITIONS,
    REGISTERED_QUERY_MEDIATION_GEOMETRIES,
    build_query_mediation_head_plan,
    build_fixed_native_loop_plan,
    choose_shuffled_commit_donor_occurrence,
    load_frozen_query_mediation_head_plan,
    load_frozen_targeted_bank,
    run_endpoint_boundary_transplant_trials,
    run_html_aligned_local_serial_trials,
    run_p0_native_loop_trials,
    run_p0_query_mediation_trials,
)
from realistic_niah_v5.pipeline import (  # noqa: E402
    read_jsonl,
    registered_records,
)
from realistic_niah_v5.parsing import (  # noqa: E402
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    raw_output_text,
)
from realistic_niah_v5.spec import V5Config  # noqa: E402


DEFAULT_V5_CONFIG = ROOT / "configs" / "realistic_niah_v5.json"
DEFAULT_MECHANISM_CONFIG = (
    ROOT / "configs" / "realistic_niah_v5_native_count_stream_dev.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_stem(*values: Any) -> str:
    text = "__".join(str(value) for value in values)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    clean = "".join(character if character.isalnum() else "_" for character in text)
    return f"{clean[:96].strip('_')}__{digest}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _runtime_manifest(
    args: argparse.Namespace,
    *,
    mechanism: NativeCountMechanismSpec,
    started: float,
    completed_shards: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except Exception:  # pragma: no cover - environment audit only
        torch_version = None
        cuda = False
        devices = []
    return {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "command": str(args.command),
        "argv": sys.argv,
        "mechanism_config": str(args.mechanism_config.resolve()),
        "mechanism_config_sha256": _sha256(args.mechanism_config),
        "v5_config": (
            str(args.v5_config.resolve()) if hasattr(args, "v5_config") else None
        ),
        "v5_config_sha256": (
            _sha256(args.v5_config) if hasattr(args, "v5_config") else None
        ),
        "mechanism_spec": mechanism.to_dict(),
        "formal_inference_eligible": mechanism.formal_inference_eligible,
        "completed_shards": int(completed_shards),
        "elapsed_seconds": float(time.perf_counter() - started),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda_available": cuda,
        "devices": devices,
        "git": _git_state(),
        "cohort": getattr(args, "cohort", None),
        "cohort_audit": getattr(args, "cohort_audit", None),
        **(extra or {}),
    }


def _model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec

    model_spec = resolve_model_spec(args.model)
    return load_registered_model(
        model_spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )


def _spec(args: argparse.Namespace) -> NativeCountMechanismSpec:
    return NativeCountMechanismSpec.load(args.mechanism_config)


def _validate_seed_contract(
    config: V5Config, mechanism: NativeCountMechanismSpec
) -> None:
    """Require the dataset and validated mechanism registries to agree exactly."""

    expected_discovery = tuple(int(value) for value in mechanism.development_seeds)
    expected_confirmation = tuple(int(value) for value in mechanism.confirmation_seeds)
    observed = {
        "discovery_seeds": tuple(int(value) for value in config.discovery_seeds),
        "confirmation_seeds": tuple(int(value) for value in config.confirmation_seeds),
        "causal_development_seeds": tuple(
            int(value) for value in config.causal_development_seeds
        ),
        "causal_confirmation_seeds": tuple(
            int(value) for value in config.causal_confirmation_seeds
        ),
    }
    expected = {
        "discovery_seeds": expected_discovery,
        "confirmation_seeds": expected_confirmation,
        "causal_development_seeds": expected_discovery,
        "causal_confirmation_seeds": expected_confirmation,
    }
    mismatches = {
        key: {"expected": list(expected[key]), "observed": list(value)}
        for key, value in observed.items()
        if value != expected[key]
    }
    if mismatches:
        raise ValueError(
            "Native count-stream dataset and mechanism seed registries disagree: "
            f"{mismatches}"
        )


def _cohort_exclusion_reason(row: dict[str, Any], cohort: str) -> str | None:
    trace_parse = row.get("trace_parse")
    if not isinstance(trace_parse, dict):
        return "missing_trace_parse"
    archived_parser = trace_parse.get("parser")
    if not isinstance(archived_parser, dict):
        return "missing_trace_parse"
    if isinstance(row.get("raw_output_text"), str):
        parser = find_trace_count_sequence(
            raw_output_text(row),
            model_family=infer_model_family(row),
            gold_records=gold_records(row),
        )
        detected = bool(parser.detected)
        trace_one_to_one = bool(parser.trace_one_to_one)
    else:
        # Unit fixtures and pre-V5 legacy rows may retain only the archived
        # parser payload. Production V5 rows always take the active-parser
        # branch so cohort selection and causal site compilation cannot drift.
        detected = bool(archived_parser.get("detected"))
        trace_one_to_one = bool(archived_parser.get("trace_one_to_one"))
    if not detected:
        return "parser_miss"
    if cohort == "parser_hit":
        return None
    if not trace_one_to_one:
        return "not_one_to_one"
    if cohort == "one_to_one":
        return None
    if cohort == "one_to_one_correct":
        return None if bool(trace_parse.get("exact_count")) else "final_count_incorrect"
    raise ValueError(f"Unknown native-thinking cohort: {cohort}")


def _registered_rows(
    args: argparse.Namespace,
    mechanism: NativeCountMechanismSpec,
) -> list[dict[str, Any]]:
    config = V5Config.load(args.v5_config)
    _validate_seed_contract(config, mechanism)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    selected = []
    exclusion_counts: dict[str, int] = {}
    row_panel = str(getattr(args, "row_panel", "all"))
    broad_phase_by_panel = {
        "broad_ranking": "ranking_discovery",
        "broad_k_selection": "k_selection_discovery",
        "broad_confirmation": "confirmation",
    }
    broad_phase = broad_phase_by_panel.get(row_panel)
    if broad_phase in {"ranking_discovery", "k_selection_discovery"} and args.seed_role != "development":
        raise ValueError(f"{row_panel} requires --seed-role development")
    if broad_phase == "confirmation" and args.seed_role != "confirmation":
        raise ValueError("broad_confirmation requires --seed-role confirmation")
    for row in rows:
        role = mechanism.seed_role(int(row["seed"]))
        if role != args.seed_role:
            continue
        if broad_phase is not None:
            if mechanism.broad_phase(int(row["seed"])) != broad_phase:
                continue
            allowed_counts = mechanism.broad_counts_for_seed(
                int(row["seed"]), phase=broad_phase
            )
            if int(row.get("gold_count", 0)) not in set(allowed_counts):
                continue
        exclusion = _cohort_exclusion_reason(row, args.cohort)
        if exclusion is not None:
            exclusion_counts[exclusion] = exclusion_counts.get(exclusion, 0) + 1
            continue
        active = dict(row)
        active["mechanism_split"] = role
        active["mechanism_cohort"] = args.cohort
        selected.append(active)
    if args.seed_role == "confirmation" and not mechanism.formal_inference_eligible:
        raise ValueError(
            "The mechanism config has no frozen fresh-confirmation registry"
        )
    selected.sort(key=lambda row: (int(row["seed"]), int(row.get("gold_count", 0))))
    if getattr(args, "limit", None) is not None:
        if int(args.limit) < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: int(args.limit)]
    args.cohort_audit = {
        "cohort": args.cohort,
        "parser_source": "active_parser_over_frozen_output",
        "eligible_rows": len(selected),
        "exclusion_counts": exclusion_counts,
        "limit": getattr(args, "limit", None),
        "row_panel": row_panel,
        "broad_phase": broad_phase,
    }
    if not selected:
        raise ValueError(f"No {args.seed_role} rows remain for {args.model}")
    return selected


def _prepare_shards(output: Path, *, resume: bool, suffix: str) -> Path:
    shard_dir = output / "shards"
    existing = list(shard_dir.glob(f"*.{suffix}")) if shard_dir.exists() else []
    if existing and not resume:
        raise FileExistsError(
            f"Existing shards found in {shard_dir}; resume or choose a new output"
        )
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir


def command_capture_broad(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    if args.seed_role != "development" or args.row_panel not in {
        "broad_ranking",
        "all",
    }:
        raise ValueError(
            "Broad attention ranking capture requires the frozen "
            "development split and either the broad_ranking panel or the "
            "explicit all-development panel"
        )
    rows = _registered_rows(args, mechanism)
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="csv")
    completed = skipped = 0
    registry_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(row["request_id"], mechanism.answer_site_id)
        shard = shard_dir / f"{stem}.csv"
        registry_path = shard_dir / f"{stem}.registry.json"
        if args.resume and shard.exists() and registry_path.exists():
            skipped += 1
            continue
        frame, registry = capture_answer_source_attention(
            model,
            tokenizer,
            adapter,
            row,
            answer_site_id=mechanism.answer_site_id,
            layers=args.layers,
        )
        frame["mechanism_split"] = args.seed_role
        _atomic_csv(shard, frame)
        _atomic_json(registry_path, registry)
        registry_rows.append(registry)
        completed += 1
        print(f"[count-stream capture-broad] {index}/{len(rows)}", flush=True)
    manifest = _runtime_manifest(
        args,
        mechanism=mechanism,
        started=started,
        completed_shards=len(list(shard_dir.glob("*.csv"))),
        extra={
            "newly_completed": completed,
            "resume_skipped": skipped,
            "selection_prohibited": args.seed_role != "development",
            "capture_role": args.seed_role,
            "registry_rows_written_this_run": len(registry_rows),
        },
    )
    _atomic_json(output / "manifest.json", manifest)


def _read_capture_frames(path: Path) -> pd.DataFrame:
    files = [path] if path.is_file() else sorted((path / "shards").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No broad-capture CSV files found under {path}")
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


def command_plan_broad(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    captures = _read_capture_frames(args.captures)
    selection_seeds = (
        mechanism.development_seeds
        if bool(args.use_all_development_seeds)
        else mechanism.broad_ranking_seeds
    )
    coverage = captures.loc[
        captures["model_label"].astype(str).eq(str(args.model))
        & pd.to_numeric(captures["seed"], errors="coerce").isin(
            selection_seeds
        )
    ][["request_id", "seed", "gold_count"]].drop_duplicates()
    observed_pairs = set(
        coverage[["seed", "gold_count"]].itertuples(index=False, name=None)
    )
    expected_pairs = {
        (int(seed), int(count))
        for seed in selection_seeds
        for count in mechanism.candidate_counts
    }
    if observed_pairs != expected_pairs or len(coverage) != len(expected_pairs):
        raise ValueError(
            "Broad ranking capture must contain exactly one request for every "
            "frozen ranking seed x count cell"
        )
    ranking = rank_answer_broad_heads(
        captures,
        source_group=args.source_group,
        development_seeds=selection_seeds,
        model_label=args.model,
    )
    sizes = (
        tuple(args.bank_sizes) if args.bank_sizes else mechanism.development_bank_sizes
    )
    plans = [
        build_answer_broad_head_plan(
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
    plan = pd.concat(plans, ignore_index=True)
    output = Path(args.output)
    _atomic_csv(output / "answer_broad_head_ranking.csv", ranking)
    _atomic_csv(output / "answer_broad_head_plan.csv", plan)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
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
                "ranking_seeds": list(selection_seeds),
                "use_all_development_seeds": bool(
                    args.use_all_development_seeds
                ),
                "confirmation_used_for_selection": False,
                "plan_sha256": _sha256(output / "answer_broad_head_plan.csv"),
            },
        ),
    )


def command_plan_trace_patch(args: argparse.Namespace) -> None:
    """Freeze the 330-local-plus-20-terminal outcome-blind pair panel."""

    started = time.perf_counter()
    mechanism = _spec(args)
    if args.seed_role != "development":
        raise ValueError("Trace-patch pair sampling is frozen on development rows")
    rows = _registered_rows(args, mechanism)
    plan = build_sparse_trace_patch_sample_plan(
        rows,
        model_label=args.model,
        donor_offsets=mechanism.trace_patch_donor_offsets,
        seeds_per_cell=mechanism.trace_patch_seeds_per_cell,
        sampling_seed=mechanism.trace_patch_sampling_seed,
        include_count2_terminal_panel=(
            mechanism.trace_patch_include_count2_terminal_panel
        ),
        candidate_counts=mechanism.candidate_counts,
    )
    output = Path(args.output)
    plan_path = output / "trace_patch_pair_plan.csv"
    _atomic_csv(plan_path, plan)
    cell_counts = (
        plan.groupby(
            ["panel_kind", "gold_count", "donor_offset"], as_index=False
        )
        .agg(
            pair_count=("pair_sha256", "nunique"),
            seed_count=("seed", "nunique"),
            receiver_count=("receiver_occurrence", "nunique"),
        )
        .sort_values(["panel_kind", "gold_count", "donor_offset"])
    )
    _atomic_csv(output / "trace_patch_cell_counts.csv", cell_counts)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "selection_policy": "outcome_blind_registry_identity_hash",
                "selection_input_fields": (
                    "sampling_seed,model_label,panel_kind,gold_count,"
                    "donor_offset,seed,request_id"
                ),
                "pair_count": int(len(plan)),
                "local_pair_count": int(plan["panel_kind"].eq("local").sum()),
                "terminal_pair_count": int(
                    plan["panel_kind"].eq("terminal").sum()
                ),
                "cell_count": int(len(cell_counts)),
                "pair_plan": str(plan_path.resolve()),
                "pair_plan_sha256": _sha256(plan_path),
            },
        ),
    )


def command_plan_terminal_last_patch(args: argparse.Namespace) -> None:
    """Freeze the 19-cell natural donor-to-final receiver panel."""

    started = time.perf_counter()
    mechanism = _spec(args)
    if args.seed_role != "development":
        raise ValueError("Terminal-last sampling is development-only")
    rows = _registered_rows(args, mechanism)
    per_cell = (
        int(args.seeds_per_cell)
        if args.seeds_per_cell is not None
        else int(mechanism.trace_patch_seeds_per_cell)
    )
    plan = build_terminal_last_trace_patch_sample_plan(
        rows,
        model_label=args.model,
        seeds_per_cell=per_cell,
        sampling_seed=mechanism.trace_patch_sampling_seed,
    )
    output = Path(args.output)
    plan_path = output / "terminal_last_pair_plan.csv"
    _atomic_csv(plan_path, plan)
    cell_counts = (
        plan.groupby(["gold_count", "donor_offset"], as_index=False)
        .agg(
            pair_count=("pair_sha256", "nunique"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["gold_count", "donor_offset"])
    )
    _atomic_csv(output / "terminal_last_cell_counts.csv", cell_counts)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "selection_policy": "outcome_blind_registry_identity_hash",
                "selection_input_fields": (
                    "sampling_seed,model_label,panel_kind,gold_count,"
                    "donor_offset,seed,request_id"
                ),
                "receiver_policy": "final_observed_count_before_answer_query",
                "donor_direction": "past_to_later_receiver",
                "count_ranges": {"-1": [2, 10], "-3": [5, 10], "-5": [7, 10]},
                "seeds_per_cell": per_cell,
                "pair_count": int(len(plan)),
                "cell_count": int(len(cell_counts)),
                "pair_plan": str(plan_path.resolve()),
                "pair_plan_sha256": _sha256(plan_path),
            },
        ),
    )


def command_source_mask(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(row["request_id"], "answer_source_mask")
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        trial_rows = []
        for condition in args.conditions:
            try:
                result = run_answer_source_mask_trial(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    condition=condition,
                    answer_site_id=mechanism.answer_site_id,
                    mask_application=args.mask_application,
                    run_greedy=not args.skip_greedy,
                    max_new_tokens=args.max_new_tokens,
                )
                result["status"] = "ok"
            except ValueError as exc:
                if "not applicable" not in str(exc).lower():
                    raise
                result = {
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                    "experiment_id": "answer_source_mask_factorial",
                    "condition": condition,
                    "status": "not_applicable",
                    "exclusion_reason": str(exc),
                    "request_id": row["request_id"],
                    "model_label": args.model,
                    "seed": int(row["seed"]),
                }
            result["mechanism_split"] = args.seed_role
            trial_rows.append(result)
        _atomic_jsonl(shard, trial_rows)
        completed += 1
        print(f"[count-stream source-mask] {index}/{len(rows)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "mask_application": args.mask_application,
                "head_scope": "all_attention_heads",
                "layer_scope": "all_decoder_layers",
            },
        ),
    )


def _load_head_plan(path: Path, *, model: str, bank_size: int) -> pd.DataFrame:
    plan = pd.read_csv(path)
    selected = plan.loc[
        plan["model_label"].eq(model)
        & pd.to_numeric(plan["bank_size"], errors="coerce").eq(int(bank_size))
    ].copy()
    if selected.empty:
        raise ValueError(f"No {model}/K{bank_size} answer broad-head plan")
    required = {"clean", "selected_bank", "layer_matched_random"}
    if not required <= set(selected["condition"].astype(str)):
        raise ValueError("Head plan is missing registered causal arms")
    return selected


def _load_frozen_serial_head_arms(
    path: Path,
    *,
    model: str,
    expected_random_controls: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one discovery-frozen trace bank and exact matched controls."""

    plan = pd.read_csv(path)
    plan = plan.loc[plan["model_label"].astype(str).eq(str(model))].copy()
    if plan.empty:
        raise ValueError(f"Frozen head plan has no rows for {model}")
    required_columns = {
        "condition",
        "repeat",
        "heads",
        "bank_size",
        "bank_sha256",
        "source_group",
        "k_selection_status",
        "k_selection_decision_sha256",
        "confirmation_locked",
    }
    missing = sorted(required_columns - set(plan.columns))
    if missing:
        raise ValueError(f"Frozen serial head plan is missing {missing}")
    if set(plan["source_group"].astype(str)) != {"trace_items"}:
        raise ValueError("Serial readout requires a frozen trace_items head bank")
    if set(plan["k_selection_status"].astype(str)) != {
        "frozen_for_confirmation"
    }:
        raise ValueError("Serial readout head K was not frozen in discovery")
    locked = plan["confirmation_locked"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    )
    if not locked.all():
        raise ValueError("Serial readout head plan is not confirmation-locked")
    bank_sizes = {
        int(value) for value in pd.to_numeric(plan["bank_size"], errors="raise")
    }
    if len(bank_sizes) != 1:
        raise ValueError("Serial readout plan must contain exactly one frozen K")
    selected_k = int(next(iter(bank_sizes)))
    counts = plan["condition"].astype(str).value_counts().to_dict()
    expected_counts = {
        "clean": 1,
        "selected_bank": 1,
        "layer_matched_random": int(expected_random_controls),
    }
    if counts != expected_counts:
        raise ValueError(
            f"Serial head arms disagree with the frozen design: {counts}"
        )
    arms: list[dict[str, Any]] = []
    for row in plan.sort_values(["condition", "repeat"], kind="stable").itertuples(
        index=False
    ):
        heads = json.loads(str(row.heads))
        if not isinstance(heads, list):
            raise ValueError("Frozen head list must be JSON")
        arms.append(
            {
                "condition": str(row.condition),
                "repeat": int(row.repeat),
                "heads": heads,
                "bank_sha256": str(row.bank_sha256),
            }
        )
    decision_hashes = set(plan["k_selection_decision_sha256"].astype(str))
    if len(decision_hashes) != 1:
        raise ValueError("Frozen head plan mixes K-selection decisions")
    return arms, {
        "source_group": "trace_items",
        "selected_bank_size": selected_k,
        "random_controls": int(expected_random_controls),
        "k_selection_decision_sha256": next(iter(decision_hashes)),
        "head_plan": str(path.resolve()),
        "head_plan_sha256": _sha256(path),
    }


def command_select_broad_k(args: argparse.Namespace) -> None:
    """Freeze one model/source K without opening any confirmation outcome."""

    started = time.perf_counter()
    mechanism = _spec(args)
    rows = [row for file in _trial_files(args.trials) for row in read_jsonl(file)]
    curve, seed_effects, decision = select_answer_broad_bank_size(
        pd.DataFrame(rows),
        model_label=args.model,
        source_group=args.source_group,
        expected_seeds=mechanism.broad_k_selection_seeds,
        expected_bank_sizes=mechanism.development_bank_sizes,
        expected_requests_per_seed=mechanism.broad_panel_counts_per_seed,
        expected_random_controls=mechanism.random_controls,
        boundary_extension_bank_size=mechanism.boundary_extension_bank_size,
        bootstrap_samples=mechanism.bootstrap_samples,
        random_seed=args.random_seed,
    )
    output = Path(args.output)
    _atomic_csv(output / "k_discovery_curve.csv", curve)
    _atomic_csv(output / "k_discovery_seed_effects.csv", seed_effects)
    _atomic_json(output / "k_selection_decision.json", decision)
    frozen_plan_path: Path | None = None
    selected_k = decision.get("selected_bank_size")
    if decision["status"] == "frozen_for_confirmation" and selected_k is not None:
        plan = _load_head_plan(
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
        _atomic_csv(frozen_plan_path, plan)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "model_label": args.model,
                "source_group": args.source_group,
                "selection_phase": "k_selection_discovery",
                "selection_seeds": list(mechanism.broad_k_selection_seeds),
                "confirmation_outcomes_used": False,
                "decision": decision,
                "frozen_plan": (
                    str(frozen_plan_path.resolve()) if frozen_plan_path else None
                ),
                "frozen_plan_sha256": (
                    _sha256(frozen_plan_path) if frozen_plan_path else None
                ),
            },
        ),
    )


def command_broad_heads(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    expected_panel = (
        "broad_confirmation"
        if args.seed_role == "confirmation"
        else "broad_k_selection"
    )
    if args.row_panel != expected_panel:
        raise ValueError(
            f"Broad-head {args.seed_role} runs require --row-panel "
            f"{expected_panel}"
        )
    rows = _registered_rows(args, mechanism)
    bank_sizes = tuple(
        int(value)
        for value in (
            args.bank_sizes
            if args.bank_sizes is not None
            else (args.bank_size,)
        )
    )
    if not bank_sizes or len(set(bank_sizes)) != len(bank_sizes):
        raise ValueError("Broad-head bank sizes must be nonempty and unique")
    if args.seed_role == "confirmation" and len(bank_sizes) != 1:
        raise ValueError("Confirmation opens exactly one discovery-frozen K")
    plan = pd.concat(
        [
            _load_head_plan(args.plan, model=args.model, bank_size=bank_size)
            for bank_size in bank_sizes
        ],
        ignore_index=True,
    )
    if args.seed_role == "confirmation":
        required_columns = {
            "k_selection_status",
            "k_selection_decision_sha256",
            "confirmation_locked",
        }
        missing_columns = sorted(required_columns - set(plan.columns))
        if missing_columns:
            raise ValueError(
                "Confirmation requires a discovery-frozen K plan; missing "
                f"{missing_columns}"
            )
        if set(plan["k_selection_status"].astype(str)) != {
            "frozen_for_confirmation"
        } or not plan["confirmation_locked"].astype(bool).all():
            raise ValueError("The answer broad-head plan is not confirmation-locked")
    source_groups = set(plan["source_group"].astype(str))
    if len(source_groups) != 1:
        raise ValueError("One answer-head run cannot mix source groups")
    source_group = next(iter(source_groups))
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(
            row["request_id"], source_group, "K", *sorted(bank_sizes)
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = []
        for plan_row in plan.itertuples(index=False):
            heads = [
                tuple(int(item) for item in pair) for pair in json.loads(plan_row.heads)
            ]
            result = run_answer_broad_head_trial(
                model,
                tokenizer,
                adapter,
                row,
                heads=heads,
                condition=str(plan_row.condition),
                source_group=source_group,
                answer_site_id=mechanism.answer_site_id,
                run_greedy=not args.skip_greedy,
                max_new_tokens=args.max_new_tokens,
            )
            result.update(
                {
                    "status": "ok",
                    "repeat": int(plan_row.repeat),
                    "bank_size": int(plan_row.bank_size),
                    "realized_ablated_head_count": len(heads),
                    "planned_bank_sha256": str(plan_row.bank_sha256),
                    "selection_metric": str(plan_row.selection_metric),
                    "mechanism_split": args.seed_role,
                    "plan_sha256": _sha256(args.plan),
                }
            )
            results.append(result)
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[count-stream broad-heads] {index}/{len(rows)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "plan": str(args.plan.resolve()),
                "plan_sha256": _sha256(args.plan),
                "source_group": source_group,
                "bank_sizes": list(bank_sizes),
            },
        ),
    )


def command_fit_basis(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    metadata, states = load_count_stream_capture_dataset(
        args.capture_index,
        site_kinds=[args.site_kind],
    )
    mask = count_stream_cohort_mask(metadata, args.cohort)
    mask &= metadata["seed"].astype(int).isin(mechanism.development_seeds).to_numpy()
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
    arrays: dict[str, np.ndarray] = {}
    fit_rows = []
    for layer in layers:
        layer_mask = mask & metadata["layer"].astype(int).eq(layer).to_numpy()
        layer_states = np.asarray(states[layer_mask], dtype=np.float32)
        labels = metadata.loc[layer_mask, args.label].to_numpy(dtype=int)
        observed_seeds = sorted(
            int(value) for value in metadata.loc[layer_mask, "seed"].unique()
        )
        if len(np.unique(labels)) < 2:
            raise ValueError(f"L{layer} has fewer than two {args.label} classes")
        center, basis, control = fit_count_stream_basis(
            layer_states,
            labels,
            rank=args.rank,
            seed=args.random_seed + int(layer),
        )
        arrays[f"center_L{layer}"] = center
        arrays[f"basis_L{layer}"] = basis
        arrays[f"control_basis_L{layer}"] = control
        fit_rows.append(
            {
                "layer": int(layer),
                "observations": int(len(layer_states)),
                "labels": sorted(int(value) for value in np.unique(labels)),
                "development_seeds": observed_seeds,
                "effective_rank": int(basis.shape[1]),
                "basis_control_max_abs_dot": float(np.max(np.abs(basis.T @ control))),
            }
        )
    _atomic_npz(args.output, **arrays)
    audit = _runtime_manifest(
        args,
        mechanism=mechanism,
        started=started,
        completed_shards=0,
        extra={
            "capture_index": str(args.capture_index.resolve()),
            "capture_index_sha256": _sha256(args.capture_index),
            "site_kind": args.site_kind,
            "site_id": args.site_id,
            "cohort": args.cohort,
            "capture_split": args.capture_split,
            "label": args.label,
            "selection_role": "development_only",
            "confirmation_used_for_fit": bool(
                metadata.loc[mask, "split"].astype(str).eq("confirmation").any()
            ),
            "fits": fit_rows,
            "artifact_sha256": _sha256(args.output),
        },
    )
    _atomic_json(args.output.with_suffix(".json"), audit)


def _load_basis(path: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = (
            f"center_L{layer}",
            f"basis_L{layer}",
            f"control_basis_L{layer}",
        )
        missing = [key for key in keys if key not in archive.files]
        if missing:
            raise ValueError(f"Basis artifact is missing {missing}")
        return tuple(np.asarray(archive[key], dtype=np.float32) for key in keys)  # type: ignore[return-value]


def _validate_trace_patch_basis_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path.with_suffix(".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Trace patching requires the fit-basis sidecar manifest: "
            f"{manifest_path}"
        )
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Basis sidecar manifest must contain one JSON object")
    if str(value.get("artifact_sha256")) != _sha256(path):
        raise ValueError("Basis artifact hash disagrees with its sidecar manifest")
    if str(value.get("site_kind")) != "item_end":
        raise ValueError("Trace patch basis must be fitted at item_end sites")
    if str(value.get("label")) != "occurrence":
        raise ValueError("Trace patch basis must use the occurrence label")
    if value.get("confirmation_used_for_fit") is not False:
        raise ValueError("Trace patch basis selection must exclude confirmation data")
    return manifest_path, value


def _load_readout_bases(path: Path, layers: Iterable[int]) -> dict[int, np.ndarray]:
    selected = tuple(sorted({int(value) for value in layers}))
    with np.load(path, allow_pickle=False) as archive:
        missing = [
            int(layer)
            for layer in selected
            if f"basis_L{int(layer)}" not in archive.files
        ]
        if missing:
            raise ValueError(
                "Basis artifact lacks downstream item-end readout bases for "
                f"layers {missing}; refit with every source/readout layer"
            )
        return {
            int(layer): np.asarray(archive[f"basis_L{int(layer)}"], dtype=np.float32)
            for layer in selected
        }


def _validated_trace_patch_plan(
    path: Path,
    *,
    mechanism: NativeCountMechanismSpec,
    rows: list[dict[str, Any]],
    model_label: str,
) -> pd.DataFrame:
    """Reconstruct and verify the frozen sampling plan before GPU work."""

    plan = pd.read_csv(path)
    needed = {
        "schema_version",
        "model_label",
        "panel_kind",
        "request_id",
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "donor_direction",
        "selection_cell_id",
        "selection_rank",
        "sampling_seed",
        "seeds_per_cell",
        "pair_sha256",
    }
    missing = sorted(needed - set(plan.columns))
    if missing:
        raise ValueError(f"Trace-patch pair plan is missing {missing}")
    plan = plan.loc[plan["model_label"].astype(str).eq(str(model_label))].copy()
    if plan.empty:
        raise ValueError(f"Pair plan has no rows for {model_label}")
    for column in (
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "selection_rank",
        "sampling_seed",
        "seeds_per_cell",
    ):
        plan[column] = pd.to_numeric(plan[column], errors="raise").astype(int)
    if set(plan["schema_version"].astype(str)) != {COUNT_STREAM_SCHEMA_VERSION}:
        raise ValueError("Pair plan uses a different count-stream schema")
    if plan["pair_sha256"].astype(str).duplicated().any():
        raise ValueError("Pair plan contains duplicate pair hashes")
    expected = build_sparse_trace_patch_sample_plan(
        rows,
        model_label=model_label,
        donor_offsets=mechanism.trace_patch_donor_offsets,
        seeds_per_cell=mechanism.trace_patch_seeds_per_cell,
        sampling_seed=mechanism.trace_patch_sampling_seed,
        include_count2_terminal_panel=(
            mechanism.trace_patch_include_count2_terminal_panel
        ),
        candidate_counts=mechanism.candidate_counts,
    )
    identity_columns = [
        "panel_kind",
        "selection_cell_id",
        "selection_rank",
        "request_id",
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "donor_direction",
        "sampling_seed",
        "seeds_per_cell",
        "pair_sha256",
    ]
    left = plan[identity_columns].sort_values("pair_sha256").reset_index(drop=True)
    right = expected[identity_columns].sort_values("pair_sha256").reset_index(
        drop=True
    )
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            "Pair plan does not match the outcome-blind plan reconstructed from "
            "the active registry/config"
        ) from exc
    return plan.sort_values(
        ["panel_kind", "gold_count", "donor_offset", "selection_rank"]
    ).reset_index(drop=True)


def _validated_terminal_last_plan(
    path: Path,
    *,
    mechanism: NativeCountMechanismSpec,
    rows: list[dict[str, Any]],
    model_label: str,
) -> pd.DataFrame:
    """Reconstruct the frozen terminal-last plan before loading a model."""

    plan = pd.read_csv(path)
    needed = {
        "schema_version",
        "experiment_id",
        "model_label",
        "panel_kind",
        "request_id",
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "donor_direction",
        "selection_cell_id",
        "selection_rank",
        "sampling_seed",
        "seeds_per_cell",
        "pair_sha256",
    }
    missing = sorted(needed - set(plan.columns))
    if missing:
        raise ValueError(f"Terminal-last pair plan is missing {missing}")
    plan = plan.loc[plan["model_label"].astype(str).eq(str(model_label))].copy()
    if plan.empty:
        raise ValueError(f"Terminal-last plan has no rows for {model_label}")
    for column in (
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "selection_rank",
        "sampling_seed",
        "seeds_per_cell",
    ):
        plan[column] = pd.to_numeric(plan[column], errors="raise").astype(int)
    if set(plan["schema_version"].astype(str)) != {COUNT_STREAM_SCHEMA_VERSION}:
        raise ValueError("Terminal-last plan uses a different schema")
    if set(plan["experiment_id"].astype(str)) != {"trace_terminal_last_pair_plan"}:
        raise ValueError("Terminal-last plan has the wrong experiment id")
    if set(plan["panel_kind"].astype(str)) != {"terminal_last"}:
        raise ValueError("Terminal-last plan contains a different panel kind")
    per_cells = set(plan["seeds_per_cell"].tolist())
    sampling_seeds = set(plan["sampling_seed"].tolist())
    if len(per_cells) != 1 or len(sampling_seeds) != 1:
        raise ValueError("Terminal-last plan mixes sampling designs")
    if sampling_seeds != {int(mechanism.trace_patch_sampling_seed)}:
        raise ValueError("Terminal-last plan uses a different sampling seed")
    expected = build_terminal_last_trace_patch_sample_plan(
        rows,
        model_label=model_label,
        seeds_per_cell=int(next(iter(per_cells))),
        sampling_seed=int(mechanism.trace_patch_sampling_seed),
    )
    identity_columns = [
        "panel_kind",
        "selection_cell_id",
        "selection_rank",
        "request_id",
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "donor_direction",
        "sampling_seed",
        "seeds_per_cell",
        "pair_sha256",
    ]
    left = plan[identity_columns].sort_values("pair_sha256").reset_index(drop=True)
    right = expected[identity_columns].sort_values("pair_sha256").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            "Terminal-last plan does not match the outcome-blind reconstruction"
        ) from exc
    return plan.sort_values(
        ["gold_count", "donor_offset", "selection_rank"]
    ).reset_index(drop=True)


def command_stream_state(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    center, basis, _control = _load_basis(args.basis, args.source_layer)
    active_readout_layers = tuple(
        sorted({int(value) for value in args.readout_layers})
    ) or (int(args.source_layer) + 1,)
    readout_bases = _load_readout_bases(args.basis, active_readout_layers)
    if "clean" not in args.conditions:
        raise ValueError(
            "Stream-state runs require the clean arm for downstream retention metrics"
        )
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Stream-state conditions must be unique")
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    state_dir = output / "states"
    completed = skipped = 0
    for row_index, row in enumerate(rows, start=1):
        occurrences: list[int | None]
        if args.scope == "all":
            occurrences = [None]
        elif args.occurrences:
            occurrences = [int(value) for value in args.occurrences]
        else:
            count = int(row.get("gold_count", len(row.get("gold_records", []))))
            occurrences = list(range(1, count + 1))
        for occurrence in occurrences:
            stem = _safe_stem(
                row["request_id"], "stream", args.scope, occurrence, args.source_layer
            )
            shard = shard_dir / f"{stem}.jsonl"
            if args.resume and shard.exists():
                skipped += 1
                continue
            results = []
            states_by_condition: dict[str, np.ndarray] = {}
            for condition in args.conditions:
                result = run_stream_state_trial(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    condition=condition,
                    source_layer=args.source_layer,
                    center=center,
                    basis=basis,
                    scope=args.scope,
                    occurrence=occurrence,
                    readout_layers=active_readout_layers,
                    answer_site_id=mechanism.answer_site_id,
                    random_seed=args.random_seed
                    + int(row["seed"])
                    + (0 if occurrence is None else int(occurrence)),
                    run_greedy=not args.skip_greedy,
                    max_new_tokens=args.max_new_tokens,
                )
                states = np.asarray(result.pop("readout_states"), dtype=np.float32)
                states_by_condition[str(condition)] = states
                state_path = state_dir / f"{stem}__{condition}.npz"
                _atomic_npz(
                    state_path,
                    states=states,
                    layers=np.asarray(result["readout_layers"], dtype=np.int64),
                    positions=np.asarray(result["readout_positions"], dtype=np.int64),
                )
                result.update(
                    {
                        "status": "ok",
                        "mechanism_split": args.seed_role,
                        "basis": str(args.basis.resolve()),
                        "basis_sha256": _sha256(args.basis),
                        "readout_states_path": str(state_path.relative_to(output)),
                    }
                )
                results.append(result)
            clean_states = states_by_condition["clean"]
            for result in results:
                condition = str(result["condition"])
                result.update(
                    stream_state_retention_metrics(
                        clean_states,
                        states_by_condition[condition],
                        readout_layers=result["readout_layers"],
                        readout_positions=result["readout_positions"],
                        query_position=max(result["readout_positions"]),
                        count_bases=readout_bases,
                    )
                )
            _atomic_jsonl(shard, results)
            completed += 1
        print(f"[count-stream stream-state] {row_index}/{len(rows)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "basis": str(args.basis.resolve()),
                "basis_sha256": _sha256(args.basis),
                "source_layer": int(args.source_layer),
                "source_scope": args.scope,
                "readout_layers": list(active_readout_layers),
                "readout_basis_site_kind": "item_end",
            },
        ),
    )


def command_joint_state_source(args: argparse.Namespace) -> None:
    """Run the count-state x answer-source factorial with cache branching."""

    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    _basis_manifest_path, basis_manifest = _validate_trace_patch_basis_manifest(
        args.basis
    )
    center, basis, _control = _load_basis(args.basis, args.source_layer)
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(
            row["request_id"],
            "joint_state_source",
            args.state_scope,
            args.source_layer,
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        trial_rows = run_count_state_answer_source_factorial_trials(
            model,
            tokenizer,
            adapter,
            row,
            source_layer=args.source_layer,
            center=center,
            basis=basis,
            state_scope=args.state_scope,
            state_occurrence=args.state_occurrence,
            state_conditions=args.state_conditions,
            mask_conditions=args.mask_conditions,
            answer_site_id=mechanism.answer_site_id,
            mask_application=args.mask_application,
            random_seed=args.random_seed + int(row["seed"]),
            run_greedy=not args.skip_greedy,
            max_new_tokens=args.max_new_tokens,
        )
        for result in trial_rows:
            result.update(
                {
                    "status": "ok",
                    "mechanism_split": args.seed_role,
                    "basis": str(args.basis.resolve()),
                    "basis_sha256": _sha256(args.basis),
                    "basis_capture_split": basis_manifest.get("capture_split"),
                }
            )
        _atomic_jsonl(shard, trial_rows)
        completed += 1
        print(f"[count-stream joint-state-source] {index}/{len(rows)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "basis": str(args.basis.resolve()),
                "basis_sha256": _sha256(args.basis),
                "basis_capture_split": basis_manifest.get("capture_split"),
                "source_layer": int(args.source_layer),
                "state_scope": args.state_scope,
                "state_conditions": list(args.state_conditions),
                "mask_conditions": list(args.mask_conditions),
                "mask_application": args.mask_application,
                "long_prefix_forwards_per_request": len(args.state_conditions),
                "query_branches_per_prefix": len(args.mask_conditions),
            },
        ),
    )


def command_trace_patch(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    pair_plan = _validated_trace_patch_plan(
        args.pair_plan,
        mechanism=mechanism,
        rows=rows,
        model_label=args.model,
    )
    planned_request_ids = set(pair_plan["request_id"].astype(str))
    rows = [row for row in rows if str(row["request_id"]) in planned_request_ids]
    row_by_request = {str(row["request_id"]): row for row in rows}
    missing_requests = sorted(planned_request_ids - set(row_by_request))
    if missing_requests:
        raise ValueError(
            f"Pair plan references {len(missing_requests)} absent registry rows"
        )
    basis_manifest_path, basis_manifest = _validate_trace_patch_basis_manifest(
        args.basis
    )
    fitted_mechanism = basis_manifest.get("mechanism_spec")
    if not isinstance(fitted_mechanism, dict):
        raise ValueError("Basis sidecar has no resolved mechanism specification")
    if tuple(int(value) for value in fitted_mechanism.get("development_seeds", ())) != (
        mechanism.development_seeds
    ):
        raise ValueError(
            "Trace patch basis and active mechanism config use different "
            "development seeds"
        )
    _center, basis, _control = _load_basis(args.basis, args.layer)
    active_readout_layers = tuple(
        sorted({int(value) for value in args.readout_layers})
    ) or (int(args.layer) + 1,)
    readout_bases = _load_readout_bases(args.basis, active_readout_layers)
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Trace patch conditions must be unique")
    if not {"clean", "self_patch"} <= set(args.conditions):
        raise ValueError("Trace patch runs require clean and self_patch controls")
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for pair_index, pair in enumerate(pair_plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        receiver = int(pair.receiver_occurrence)
        donor = int(pair.donor_occurrence)
        panel_kind = str(pair.panel_kind)
        stem = _safe_stem(
            row["request_id"], panel_kind, receiver, donor, args.layer
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        common_kwargs = {
            "receiver_occurrence": receiver,
            "donor_occurrence": donor,
            "layer": args.layer,
            "basis": basis,
            "readout_layers": active_readout_layers,
            "readout_bases": readout_bases,
            "conditions": args.conditions,
            "answer_site_id": mechanism.answer_site_id,
            "random_seed": (
                int(args.random_seed)
                + int(row["seed"]) * 101
                + receiver * 17
                + donor
            ),
            "run_greedy": not args.skip_greedy,
            "max_new_tokens": args.max_new_tokens,
        }
        if panel_kind == "local":
            results = run_trace_intermediate_patch_trials(
                model,
                tokenizer,
                adapter,
                row,
                **common_kwargs,
            )
        elif panel_kind == "terminal":
            results = run_trace_terminal_patch_trials(
                model,
                tokenizer,
                adapter,
                row,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown trace-patch panel kind: {panel_kind}")
        for result in results:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "panel_kind": panel_kind,
                    "selection_cell_id": str(pair.selection_cell_id),
                    "selection_rank": int(pair.selection_rank),
                    "pair_sha256": str(pair.pair_sha256),
                    "pair_plan": str(args.pair_plan.resolve()),
                    "pair_plan_sha256": _sha256(args.pair_plan),
                    "basis": str(args.basis.resolve()),
                    "basis_sha256": _sha256(args.basis),
                    "basis_manifest_sha256": _sha256(basis_manifest_path),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[count-stream trace-patch] {pair_index}/{len(pair_plan)}",
            flush=True,
        )
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "planned_pair_count": int(len(pair_plan)),
                "local_pair_count": int(
                    pair_plan["panel_kind"].eq("local").sum()
                ),
                "terminal_pair_count": int(
                    pair_plan["panel_kind"].eq("terminal").sum()
                ),
                "pair_plan": str(args.pair_plan.resolve()),
                "pair_plan_sha256": _sha256(args.pair_plan),
                "basis": str(args.basis.resolve()),
                "basis_sha256": _sha256(args.basis),
                "basis_manifest": str(basis_manifest_path.resolve()),
                "basis_manifest_sha256": _sha256(basis_manifest_path),
                "basis_fit_label": str(basis_manifest["label"]),
                "basis_fit_site_kind": str(basis_manifest["site_kind"]),
                "patch_layer": int(args.layer),
                "readout_layers": list(active_readout_layers),
                "donor_offsets": list(mechanism.trace_patch_donor_offsets),
                "conditions": list(args.conditions),
                "primary_temporal_direction": mechanism.trace_patch_primary_direction,
                "primary_outcome": mechanism.trace_patch_primary_outcome,
                "future_to_past_role": "counterfactual_representational_sensitivity",
            },
        ),
    )


def _native_loop_plan_for_rows(
    args: argparse.Namespace,
    mechanism: NativeCountMechanismSpec,
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    seeds = (
        mechanism.development_seeds
        if args.seed_role == "development"
        else mechanism.confirmation_seeds
    )
    plan_rows = rows
    if (
        mechanism.experiment_id
        == "realistic_niah_v5_full_commit_specificity_confirmation_v1"
    ):
        # The receiver, intended donor, and wrong-ordinal donor must be three
        # distinct natural commits that each own a successor. With occurrences
        # 1..N-1, that structural gate is N >= 4. Apply it before outcome-blind
        # hash sampling, so an ineligible request cannot displace an eligible
        # request from the same seed/offset cell.
        plan_rows = [
            row for row in rows if int(row.get("gold_count", 0)) >= 4
        ]
    return build_fixed_native_loop_plan(
        plan_rows,
        model_label=args.model,
        seeds=seeds,
        seed_role=args.seed_role,
        donor_offsets=tuple(int(value) for value in args.donor_offsets),
        candidate_counts=tuple(range(2, 11)),
        sampling_seed=int(args.random_seed),
        require_all_seeds_per_offset=not bool(args.allow_incomplete_offsets),
        include_boundaries=not bool(args.no_boundaries),
    )


def command_plan_native_loop(args: argparse.Namespace) -> None:
    """Freeze the rank-free 20-discovery or 10-confirmation loop plan."""

    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    plan = _native_loop_plan_for_rows(args, mechanism, rows)
    output = Path(args.output)
    _atomic_csv(output / "native_loop_plan.csv", plan)
    seeds = sorted(int(value) for value in plan["seed"].unique())
    cells = (
        plan.groupby(["panel_kind", "gold_count", "donor_offset"], dropna=False)
        .agg(pair_count=("pair_sha256", "nunique"), seed_count=("seed", "nunique"))
        .reset_index()
    )
    _atomic_csv(output / "native_loop_plan_cells.csv", cells)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "plan": str((output / "native_loop_plan.csv").resolve()),
                "plan_sha256": _sha256(output / "native_loop_plan.csv"),
                "pair_count": int(len(plan)),
                "seed_role": args.seed_role,
                "registered_seeds": seeds,
                "selection_rank_used": False,
                "outcome_blind": True,
                "donor_offsets": list(args.donor_offsets),
                "local_cohort_counts": {
                    str(key): int(value)
                    for key, value in plan["local_cohort_policy"]
                    .value_counts()
                    .to_dict()
                    .items()
                },
            },
        ),
    )


def command_plan_query_mediation_heads(args: argparse.Namespace) -> None:
    """Freeze the active targeted bank and a disjoint matched control."""

    started = time.perf_counter()
    bank = load_frozen_targeted_bank(
        args.targeted_selection,
        args.anchor_routing,
        model_label=args.model,
    )
    ranking = pd.read_csv(args.candidate_ranking)
    required = {"model_label", "layer", "head"}
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise ValueError(f"Query-mediation candidate ranking is missing {missing}")
    candidates = (
        ranking.loc[ranking["model_label"].astype(str).eq(str(args.model))]
        [["layer", "head"]]
        .drop_duplicates()
        .sort_values(["layer", "head"], kind="stable")
    )
    if candidates.empty:
        raise ValueError("No query-mediation candidate heads for this model")
    plan = build_query_mediation_head_plan(
        bank,
        candidates.astype(int).values.tolist(),
        source_layer=int(args.layer),
        random_seed=int(args.random_seed),
        candidate_source_sha256=_sha256(args.candidate_ranking),
    )
    output = Path(args.output)
    plan_path = output / "query_mediation_head_plan.json"
    _atomic_json(plan_path, plan)
    _atomic_json(
        output / "manifest.json",
        {
            "schema_version": COUNT_STREAM_SCHEMA_VERSION,
            "experiment_id": "plan_p0_same_trajectory_query_mediation_heads",
            "model_label": str(args.model),
            "source_layer": int(args.layer),
            "targeted_selection": str(args.targeted_selection.resolve()),
            "targeted_selection_sha256": _sha256(args.targeted_selection),
            "anchor_routing": str(args.anchor_routing.resolve()),
            "anchor_routing_sha256": _sha256(args.anchor_routing),
            "candidate_ranking": str(args.candidate_ranking.resolve()),
            "candidate_ranking_sha256": _sha256(args.candidate_ranking),
            "head_plan": str(plan_path.resolve()),
            "head_plan_sha256": _sha256(plan_path),
            "random_seed": int(args.random_seed),
            "selection_rank_used": False,
            "outcome_blind": True,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def _validated_native_loop_plan(
    args: argparse.Namespace,
    mechanism: NativeCountMechanismSpec,
    rows: list[dict[str, Any]],
    *,
    panel_kind: str,
) -> pd.DataFrame:
    plan = pd.read_csv(args.plan)
    if "selection_rank" in plan.columns:
        raise ValueError("Native-loop formal plans must not contain selection_rank")
    if "selection_rank_used" not in plan.columns:
        raise ValueError("Native-loop plan lacks selection-rank audit")
    used = plan["selection_rank_used"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    )
    if used.any():
        raise ValueError("Native-loop plan reports selection_rank_used=true")
    expected = _native_loop_plan_for_rows(args, mechanism, rows)
    observed = plan.loc[plan["panel_kind"].astype(str).eq(str(panel_kind))].copy()
    expected = expected.loc[
        expected["panel_kind"].astype(str).eq(str(panel_kind))
    ].copy()
    if observed.empty:
        raise ValueError(f"Native-loop plan has no {panel_kind} rows")
    columns = [
        "model_label",
        "seed_role",
        "panel_kind",
        "request_id",
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "observed_item_count",
        "trace_one_to_one",
        "trace_category",
        "local_cohort_policy",
        "pair_sha256",
    ]
    left = observed[columns].sort_values("pair_sha256").reset_index(drop=True)
    right = expected[columns].sort_values("pair_sha256").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            "Native-loop plan disagrees with its outcome-blind reconstruction"
        ) from exc
    return observed.sort_values(
        ["gold_count", "donor_offset", "seed"], kind="stable"
    ).reset_index(drop=True)


def command_p0_native_loop(args: argparse.Namespace) -> None:
    """Run P0 probe -> routed targeted attention -> first-city trials."""

    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    plan = _validated_native_loop_plan(
        args, mechanism, rows, panel_kind="p0_local"
    )
    row_by_request = {str(row["request_id"]): row for row in rows}
    center, basis, _control = _load_basis(args.basis, args.layer)
    bank = load_frozen_targeted_bank(
        args.targeted_selection,
        args.anchor_routing,
        model_label=args.model,
    )
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    requested_conditions = {str(value) for value in args.conditions}
    requires_shuffled = (
        "shuffled_natural_donor_patch" in requested_conditions
    )
    if requested_conditions & set(REGISTERED_FULL_COMMIT_SPECIFICITY_CONDITIONS):
        if not {"clean", "self_patch", "full_donor_patch"} <= requested_conditions:
            raise ValueError(
                "Full-commit specificity runs require clean, self_patch, and "
                "full_donor_patch"
            )
    if requires_shuffled and "shuffled_donor_occurrence" not in plan.columns:
        raise ValueError(
            "The specificity plan lacks frozen shuffled_donor_occurrence"
        )
    for index, pair in enumerate(plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        shuffled_donor_occurrence = None
        if requires_shuffled:
            shuffled_donor_occurrence = int(pair.shuffled_donor_occurrence)
            expected_control = choose_shuffled_commit_donor_occurrence(
                gold_count=int(pair.gold_count),
                receiver_occurrence=int(pair.receiver_occurrence),
                donor_occurrence=int(pair.donor_occurrence),
                random_seed=int(args.random_seed) + int(pair.seed) * 1009,
            )
            if shuffled_donor_occurrence != expected_control:
                raise ValueError(
                    "Frozen shuffled donor disagrees with its outcome-blind "
                    "reconstruction"
                )
        stem = _safe_stem(pair.pair_sha256, "p0_native_loop", args.layer)
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        trials = run_p0_native_loop_trials(
            model,
            tokenizer,
            adapter,
            row,
            receiver_occurrence=int(pair.receiver_occurrence),
            donor_occurrence=int(pair.donor_occurrence),
            shuffled_donor_occurrence=shuffled_donor_occurrence,
            layer=int(args.layer),
            center=center,
            basis=basis,
            targeted_bank=bank,
            conditions=args.conditions,
            random_seed=int(args.random_seed) + int(pair.seed) * 1009,
            run_greedy=not args.skip_greedy,
            max_new_tokens=int(args.max_new_tokens),
        )
        for result in trials:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "pair_sha256": str(pair.pair_sha256),
                    "plan_sha256": _sha256(args.plan),
                    "basis": str(args.basis.resolve()),
                    "basis_sha256": _sha256(args.basis),
                    "targeted_selection": str(args.targeted_selection.resolve()),
                    "anchor_routing": str(args.anchor_routing.resolve()),
                    "plan_local_cohort_policy": str(pair.local_cohort_policy),
                    "plan_observed_item_count": int(pair.observed_item_count),
                    "plan_trace_one_to_one": bool(pair.trace_one_to_one),
                    "plan_trace_category": str(pair.trace_category),
                }
            )
        _atomic_jsonl(shard, trials)
        completed += 1
        print(f"[native-loop p0] {index}/{len(plan)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "seed_role": args.seed_role,
                "pair_count": int(len(plan)),
                "plan": str(args.plan.resolve()),
                "plan_sha256": _sha256(args.plan),
                "basis": str(args.basis.resolve()),
                "basis_sha256": _sha256(args.basis),
                "source_layer": int(args.layer),
                "conditions": list(args.conditions),
                "targeted_bank": bank,
                "selection_rank_used": False,
            },
        ),
    )


def command_p0_query_mediation(args: argparse.Namespace) -> None:
    """Run state x query-local targeted-head mediation on one geometry."""

    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    plan = _validated_native_loop_plan(
        args, mechanism, rows, panel_kind="p0_local"
    )
    row_by_request = {str(row["request_id"]): row for row in rows}
    center, basis, _control = _load_basis(args.basis, args.layer)
    bank = load_frozen_targeted_bank(
        args.targeted_selection,
        args.anchor_routing,
        model_label=args.model,
    )
    head_plan = load_frozen_query_mediation_head_plan(
        args.head_plan,
        bank,
        model_label=args.model,
        source_layer=int(args.layer),
    )
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, pair in enumerate(plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        stem = _safe_stem(
            pair.pair_sha256,
            "p0_query_mediation",
            args.geometry,
            args.layer,
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        trials = run_p0_query_mediation_trials(
            model,
            tokenizer,
            adapter,
            row,
            receiver_occurrence=int(pair.receiver_occurrence),
            donor_occurrence=int(pair.donor_occurrence),
            layer=int(args.layer),
            geometry=str(args.geometry),
            center=center,
            basis=basis,
            targeted_bank=bank,
            head_plan=head_plan,
            random_seed=int(args.random_seed) + int(pair.seed) * 1019,
            run_greedy=not args.skip_greedy,
            max_new_tokens=int(args.max_new_tokens),
        )
        for result in trials:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "pair_sha256": str(pair.pair_sha256),
                    "plan_sha256": _sha256(args.plan),
                    "basis": str(args.basis.resolve()),
                    "basis_sha256": _sha256(args.basis),
                    "targeted_selection": str(args.targeted_selection.resolve()),
                    "anchor_routing": str(args.anchor_routing.resolve()),
                    "head_plan": str(args.head_plan.resolve()),
                    "plan_local_cohort_policy": str(pair.local_cohort_policy),
                    "plan_observed_item_count": int(pair.observed_item_count),
                    "plan_trace_one_to_one": bool(pair.trace_one_to_one),
                    "plan_trace_category": str(pair.trace_category),
                }
            )
        _atomic_jsonl(shard, trials)
        completed += 1
        print(
            f"[native-loop query-mediation {args.geometry}] "
            f"{index}/{len(plan)}",
            flush=True,
        )
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "seed_role": args.seed_role,
                "pair_count": int(len(plan)),
                "plan": str(args.plan.resolve()),
                "plan_sha256": _sha256(args.plan),
                "basis": str(args.basis.resolve()),
                "basis_sha256": _sha256(args.basis),
                "source_layer": int(args.layer),
                "geometry": str(args.geometry),
                "targeted_bank": bank,
                "head_plan": head_plan,
                "selection_rank_used": False,
                "outcome_blind": True,
            },
        ),
    )


def command_boundary_native_loop(args: argparse.Namespace) -> None:
    """Run middle<->terminal endpoint transplants with free continuation."""

    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    plans = [
        _validated_native_loop_plan(
            args, mechanism, rows, panel_kind="terminal_injection"
        ),
        _validated_native_loop_plan(
            args, mechanism, rows, panel_kind="nonterminal_injection"
        ),
    ]
    plan = pd.concat(plans, ignore_index=True)
    row_by_request = {str(row["request_id"]): row for row in rows}
    center, basis, _control = _load_basis(args.basis, args.layer)
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, pair in enumerate(plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        stem = _safe_stem(pair.pair_sha256, "boundary_native_loop", args.layer)
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        trials = run_endpoint_boundary_transplant_trials(
            model,
            tokenizer,
            adapter,
            row,
            receiver_occurrence=int(pair.receiver_occurrence),
            donor_occurrence=int(pair.donor_occurrence),
            layer=int(args.layer),
            center=center,
            basis=basis,
            conditions=args.conditions,
            random_seed=int(args.random_seed) + int(pair.seed) * 1013,
            max_new_tokens=int(args.max_new_tokens),
        )
        for result in trials:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "pair_sha256": str(pair.pair_sha256),
                    "plan_sha256": _sha256(args.plan),
                    "basis": str(args.basis.resolve()),
                    "basis_sha256": _sha256(args.basis),
                    "plan_local_cohort_policy": str(pair.local_cohort_policy),
                    "plan_observed_item_count": int(pair.observed_item_count),
                    "plan_trace_one_to_one": bool(pair.trace_one_to_one),
                    "plan_trace_category": str(pair.trace_category),
                }
            )
        _atomic_jsonl(shard, trials)
        completed += 1
        print(f"[native-loop boundary] {index}/{len(plan)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "seed_role": args.seed_role,
                "pair_count": int(len(plan)),
                "plan": str(args.plan.resolve()),
                "plan_sha256": _sha256(args.plan),
                "basis": str(args.basis.resolve()),
                "basis_sha256": _sha256(args.basis),
                "source_layer": int(args.layer),
                "conditions": list(args.conditions),
                "selection_rank_used": False,
            },
        ),
    )


def command_trace_full_state_patch(args: argparse.Namespace) -> None:
    """Run multi-position one-shot or cumulative full-state transfers."""

    started = time.perf_counter()
    mechanism = _spec(args)
    if getattr(args, "limit", None) is not None:
        raise ValueError(
            "trace-full-state-patch uses --max-selection-rank, not --limit, "
            "so plan reconstruction remains complete"
        )
    rows = _registered_rows(args, mechanism)
    if args.plan_kind == "sparse_local":
        pair_plan = _validated_trace_patch_plan(
            args.pair_plan,
            mechanism=mechanism,
            rows=rows,
            model_label=args.model,
        )
        pair_plan = pair_plan.loc[
            pair_plan["panel_kind"].astype(str).eq("local")
            & pair_plan["donor_direction"].astype(str).eq(
                "past_to_later_receiver"
            )
        ].copy()
    elif args.plan_kind == "terminal_last":
        pair_plan = _validated_terminal_last_plan(
            args.pair_plan,
            mechanism=mechanism,
            rows=rows,
            model_label=args.model,
        )
    else:  # argparse prevents this, but keep the runtime contract explicit.
        raise ValueError(f"Unknown full-state plan kind: {args.plan_kind}")
    if args.max_selection_rank is not None:
        pair_plan = pair_plan.loc[
            pair_plan["selection_rank"].le(int(args.max_selection_rank))
        ].copy()
    if args.selection_cell_ids:
        requested_cells = {str(value) for value in args.selection_cell_ids}
        available_cells = set(pair_plan["selection_cell_id"].astype(str))
        missing_cells = sorted(requested_cells - available_cells)
        if missing_cells:
            raise ValueError(
                f"Full-state plan does not contain requested cells {missing_cells}"
            )
        pair_plan = pair_plan.loc[
            pair_plan["selection_cell_id"].astype(str).isin(requested_cells)
        ].copy()
    if pair_plan.empty:
        raise ValueError("No pairs remain for the full-state patch run")

    planned_request_ids = set(pair_plan["request_id"].astype(str))
    row_by_request = {
        str(row["request_id"]): row
        for row in rows
        if str(row["request_id"]) in planned_request_ids
    }
    missing_requests = sorted(planned_request_ids - set(row_by_request))
    if missing_requests:
        raise ValueError(
            f"Full-state plan references {len(missing_requests)} absent rows"
        )
    basis_manifest_path, basis_manifest = _validate_trace_patch_basis_manifest(
        args.basis
    )
    _center, _basis, _control = _load_basis(args.basis, args.layer)
    active_readout_layers = tuple(
        sorted({int(value) for value in args.readout_layers})
    ) or (int(args.layer) + 1,)
    readout_bases = _load_readout_bases(args.basis, active_readout_layers)
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Full-state patch conditions must be unique")
    if not {"clean", "self_patch"} <= set(args.conditions):
        raise ValueError("Full-state runs require clean and self_patch controls")

    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = not_applicable = 0
    for pair_index, pair in enumerate(pair_plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        receiver = int(pair.receiver_occurrence)
        donor = int(pair.donor_occurrence)
        for geometry in args.geometries:
            for layer_mode in args.layer_modes:
                stem = _safe_stem(
                    row["request_id"],
                    args.plan_kind,
                    receiver,
                    donor,
                    args.layer,
                    geometry,
                    layer_mode,
                )
                shard = shard_dir / f"{stem}.jsonl"
                if args.resume and shard.exists():
                    skipped += 1
                    continue
                try:
                    results = run_trace_full_state_patch_trials(
                        model,
                        tokenizer,
                        adapter,
                        row,
                        receiver_occurrence=receiver,
                        donor_occurrence=donor,
                        layer=args.layer,
                        geometry=geometry,
                        layer_mode=layer_mode,
                        readout_layers=active_readout_layers,
                        readout_bases=readout_bases,
                        conditions=args.conditions,
                        answer_site_id=mechanism.answer_site_id,
                        run_greedy=not args.skip_greedy,
                        max_new_tokens=args.max_new_tokens,
                    )
                except ValueError as exc:
                    if "not applicable" not in str(exc).lower():
                        raise
                    not_applicable += 1
                    results = [
                        {
                            "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                            "experiment_id": "trace_full_state_geometry_patching",
                            "condition": condition,
                            "status": "not_applicable",
                            "exclusion_reason": str(exc),
                            "request_id": row["request_id"],
                            "model_label": args.model,
                            "seed": int(row["seed"]),
                            "gold_count": int(pair.gold_count),
                            "layer": int(args.layer),
                            "patch_geometry": str(geometry),
                            "patch_layer_mode": str(layer_mode),
                            "receiver_occurrence": receiver,
                            "donor_occurrence": donor,
                            "donor_offset": donor - receiver,
                        }
                        for condition in args.conditions
                    ]
                for result in results:
                    result.update(
                        {
                            "mechanism_split": args.seed_role,
                            "panel_kind": str(pair.panel_kind),
                            "plan_kind": args.plan_kind,
                            "selection_cell_id": str(pair.selection_cell_id),
                            "selection_rank": int(pair.selection_rank),
                            "pair_sha256": str(pair.pair_sha256),
                            "pair_plan": str(args.pair_plan.resolve()),
                            "pair_plan_sha256": _sha256(args.pair_plan),
                            "basis": str(args.basis.resolve()),
                            "basis_sha256": _sha256(args.basis),
                            "basis_manifest_sha256": _sha256(basis_manifest_path),
                        }
                    )
                _atomic_jsonl(shard, results)
                completed += 1
        print(
            f"[count-stream trace-full-state-patch] "
            f"{pair_index}/{len(pair_plan)}",
            flush=True,
        )
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "not_applicable_geometry_shards_this_run": not_applicable,
                "planned_pair_count": int(len(pair_plan)),
                "planned_shard_count": int(
                    len(pair_plan) * len(args.geometries) * len(args.layer_modes)
                ),
                "plan_kind": args.plan_kind,
                "pair_plan": str(args.pair_plan.resolve()),
                "pair_plan_sha256": _sha256(args.pair_plan),
                "max_selection_rank": args.max_selection_rank,
                "selection_cell_ids": list(args.selection_cell_ids or ()),
                "basis": str(args.basis.resolve()),
                "basis_sha256": _sha256(args.basis),
                "basis_manifest": str(basis_manifest_path.resolve()),
                "basis_manifest_sha256": _sha256(basis_manifest_path),
                "patch_layer": int(args.layer),
                "readout_layers": list(active_readout_layers),
                "geometries": list(args.geometries),
                "layer_modes": list(args.layer_modes),
                "conditions": list(args.conditions),
                "claim_scope": "development_full_state_sufficiency",
            },
        ),
    )


def command_html_aligned_terminal_span(args: argparse.Namespace) -> None:
    """Run the old-HTML same-position terminal full-span experiment."""

    started = time.perf_counter()
    mechanism = _spec(args)
    if args.cohort != "one_to_one":
        raise ValueError("HTML-aligned terminal spans require cohort=one_to_one")
    eligible_rows = _registered_rows(args, mechanism)
    expected_seeds = (
        tuple(mechanism.development_seeds)
        if args.seed_role == "development"
        else tuple(mechanism.confirmation_seeds)
    )
    rows = []
    for seed in expected_seeds:
        candidates = [
            row for row in eligible_rows if int(row["seed"]) == int(seed)
        ]
        if not candidates:
            raise ValueError(f"HTML-aligned plan has no one-to-one row for seed {seed}")
        candidates.sort(
            key=lambda row: (int(row["gold_count"]), str(row["request_id"])),
            reverse=True,
        )
        rows.append(candidates[0])
    observed_seeds = tuple(int(row["seed"]) for row in rows)
    if observed_seeds != expected_seeds or len(rows) != len(expected_seeds):
        raise ValueError(
            "HTML-aligned highest-one-to-one-count row contract changed"
        )
    registration = {
        "schema_version": "realistic_niah_v5_html_aligned_row_plan_v1",
        "model_label": args.model,
        "phase": args.seed_role,
        "seed_count": len(expected_seeds),
        "seeds": list(expected_seeds),
        "row_selection_rule": "highest_parser_one_to_one_count_per_seed",
        "selected_count_by_seed": {
            str(row["seed"]): int(row["gold_count"]) for row in rows
        },
        "row_count": len(rows),
        "site": "terminal_full_item_span",
        "layer": int(args.layer),
        "layer_mode": "cumulative_clamp",
        "conditions": list(REGISTERED_HTML_ALIGNED_SPAN_CONDITIONS),
        "outcome_blind": True,
        "selection_rank_used": False,
        "request_ids": [str(row["request_id"]) for row in rows],
    }
    registration["plan_sha256"] = hashlib.sha256(
        json.dumps(registration, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != registration:
            raise ValueError("Existing HTML-aligned frozen row plan changed")
    else:
        _atomic_json(plan_path, registration)

    model, tokenizer, adapter = _model(args)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(
            row["request_id"], "html_aligned_terminal_full_span", args.layer
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = run_html_aligned_terminal_span_trials(
            model,
            tokenizer,
            adapter,
            row,
            layer=int(args.layer),
            random_seed=(
                int(args.random_seed)
                + int(row["seed"]) * 1019
                + int(row["gold_count"]) * 9173
            ),
            answer_site_id=mechanism.answer_site_id,
            run_greedy=not args.skip_greedy,
            max_new_tokens=int(args.max_new_tokens),
        )
        for result in results:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "row_plan": str(plan_path.resolve()),
                    "row_plan_sha256": registration["plan_sha256"],
                    "selection_rank_used": False,
                    "outcome_blind": True,
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[count-stream html-aligned-terminal-span] {index}/{len(rows)}",
            flush=True,
        )
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "row_plan": str(plan_path.resolve()),
                "row_plan_sha256": registration["plan_sha256"],
                "planned_rows": len(rows),
                "planned_conditions_per_row": len(
                    REGISTERED_HTML_ALIGNED_SPAN_CONDITIONS
                ),
                "row_selection_rule": "highest_parser_one_to_one_count_per_seed",
                "selected_count_by_seed": registration["selected_count_by_seed"],
                "patch_layer": int(args.layer),
                "layer_mode": "cumulative_clamp",
                "claim_scope": (
                    "same_position_terminal_full_span_final_count_necessity_sufficiency"
                ),
                "outcome_blind": True,
                "selection_rank_used": False,
            },
        ),
    )


def command_html_aligned_local_serial(args: argparse.Namespace) -> None:
    """Run penultimate full-span state -> targeted-head -> city mediation."""

    started = time.perf_counter()
    mechanism = _spec(args)
    if args.cohort != "one_to_one":
        raise ValueError("HTML local serial mediation requires cohort=one_to_one")
    expected_layers = {"Qwen3-8B": 19, "Gemma4-E4B": 16}
    if int(args.layer) != expected_layers[str(args.model)]:
        raise ValueError("HTML local serial source layer changed")
    eligible_rows = _registered_rows(args, mechanism)
    expected_seeds = (
        tuple(mechanism.development_seeds)
        if args.seed_role == "development"
        else tuple(mechanism.confirmation_seeds)
    )
    rows = []
    for seed in expected_seeds:
        candidates = [
            row
            for row in eligible_rows
            if int(row["seed"]) == int(seed) and int(row["gold_count"]) >= 4
        ]
        if not candidates:
            raise ValueError(f"HTML local serial has no eligible row for seed {seed}")
        candidates.sort(
            key=lambda row: (int(row["gold_count"]), str(row["request_id"])),
            reverse=True,
        )
        rows.append(candidates[0])
    bank = load_frozen_targeted_bank(
        args.targeted_selection,
        args.anchor_routing,
        model_label=args.model,
    )
    head_plan = load_frozen_query_mediation_head_plan(
        args.head_plan,
        bank,
        model_label=args.model,
        source_layer=int(args.layer),
    )
    head_token_geometry = str(args.head_token_geometry)
    if head_token_geometry == "query_plus_full_path":
        state_conditions = ["uninformative", "uninformative_target_restore"]
        head_conditions = [
            "intact_full_path",
            "selected_mask_full_path",
            "layer_matched_random_mask_full_path",
            "selected_restore_full_path_from_restored_state",
        ]
        planned_arms_per_row = 7
        schema_version = "realistic_niah_v5_html_local_serial_row_plan_v2"
    else:
        state_conditions = [
            "clean",
            "uninformative",
            "clean_target_ablation",
            "uninformative_target_restore",
        ]
        head_conditions = [
            "intact",
            "selected_mask",
            "layer_matched_random_mask",
            "selected_restore_from_restored_state",
        ]
        planned_arms_per_row = 13
        schema_version = "realistic_niah_v5_html_local_serial_row_plan_v1"
    registration = {
        "schema_version": schema_version,
        "model_label": args.model,
        "phase": args.seed_role,
        "seed_count": len(expected_seeds),
        "seeds": list(expected_seeds),
        "row_count": len(rows),
        "row_selection_rule": "highest_parser_one_to_one_count_per_seed",
        "selected_count_by_seed": {
            str(row["seed"]): int(row["gold_count"]) for row in rows
        },
        "site": "two_before_terminal_full_span_across_one_item_to_terminal_city_query",
        "layer": int(args.layer),
        "layer_mode": "cumulative_clamp",
        "state_conditions": state_conditions,
        "head_conditions": head_conditions,
        "head_token_geometry": head_token_geometry,
        "planned_arms_per_row": planned_arms_per_row,
        "targeted_bank_sha256": str(bank["bank_sha256"]),
        "head_plan_file_sha256": str(head_plan["plan_file_sha256"]),
        "outcome_blind": True,
        "selection_rank_used": False,
        "request_ids": [str(row["request_id"]) for row in rows],
    }
    registration["plan_sha256"] = hashlib.sha256(
        json.dumps(registration, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != registration:
            raise ValueError("Existing HTML local serial row plan changed")
    else:
        _atomic_json(plan_path, registration)

    model, tokenizer, adapter = _model(args)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(
            row["request_id"], "html_aligned_local_serial", args.layer
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = run_html_aligned_local_serial_trials(
            model,
            tokenizer,
            adapter,
            row,
            layer=int(args.layer),
            targeted_bank=bank,
            head_plan=head_plan,
            random_seed=(
                int(args.random_seed)
                + int(row["seed"]) * 1019
                + int(row["gold_count"]) * 9173
            ),
            head_token_geometry=head_token_geometry,
            run_greedy=not args.skip_greedy,
            max_new_tokens=int(args.max_new_tokens),
        )
        for result in results:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "row_plan": str(plan_path.resolve()),
                    "row_plan_sha256": registration["plan_sha256"],
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[count-stream html-aligned-local-serial] {index}/{len(rows)}",
            flush=True,
        )
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "row_plan": str(plan_path.resolve()),
                "row_plan_sha256": registration["plan_sha256"],
                "planned_rows": len(rows),
                "planned_arms_per_row": planned_arms_per_row,
                "head_token_geometry": head_token_geometry,
                "targeted_bank_sha256": str(bank["bank_sha256"]),
                "head_plan_file_sha256": str(head_plan["plan_file_sha256"]),
                "claim_scope": (
                    "one_step_propagated_full_span_to_targeted_heads_to_terminal_city"
                ),
                "outcome_blind": True,
                "selection_rank_used": False,
            },
        ),
    )


def command_full_state_patch_source(args: argparse.Namespace) -> None:
    """Cross terminal full-state donor patches with answer source masks."""

    started = time.perf_counter()
    mechanism = _spec(args)
    if getattr(args, "limit", None) is not None:
        raise ValueError(
            "full-state-patch-source uses --max-selection-rank, not --limit"
        )
    rows = _registered_rows(args, mechanism)
    pair_plan = _validated_terminal_last_plan(
        args.pair_plan,
        mechanism=mechanism,
        rows=rows,
        model_label=args.model,
    )
    if args.min_selection_rank is not None:
        pair_plan = pair_plan.loc[
            pair_plan["selection_rank"].ge(int(args.min_selection_rank))
        ].copy()
    if args.max_selection_rank is not None:
        pair_plan = pair_plan.loc[
            pair_plan["selection_rank"].le(int(args.max_selection_rank))
        ].copy()
    if pair_plan.empty:
        raise ValueError("No pairs remain for the full-state/source factorial")
    planned_request_ids = set(pair_plan["request_id"].astype(str))
    row_by_request = {
        str(row["request_id"]): row
        for row in rows
        if str(row["request_id"]) in planned_request_ids
    }
    missing_requests = sorted(planned_request_ids - set(row_by_request))
    if missing_requests:
        raise ValueError(
            f"Full-state/source plan references {len(missing_requests)} absent rows"
        )
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = not_applicable = 0
    for pair_index, pair in enumerate(pair_plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        receiver = int(pair.receiver_occurrence)
        donor = int(pair.donor_occurrence)
        for geometry in args.geometries:
            stem = _safe_stem(
                row["request_id"],
                "full_state_patch_source",
                receiver,
                donor,
                args.layer,
                geometry,
                args.layer_mode,
            )
            shard = shard_dir / f"{stem}.jsonl"
            if args.resume and shard.exists():
                skipped += 1
                continue
            try:
                results = run_full_state_patch_answer_source_factorial_trials(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    receiver_occurrence=receiver,
                    donor_occurrence=donor,
                    layer=args.layer,
                    geometry=geometry,
                    layer_mode=args.layer_mode,
                    patch_conditions=args.patch_conditions,
                    mask_conditions=args.mask_conditions,
                    answer_site_id=mechanism.answer_site_id,
                    mask_application=args.mask_application,
                    run_greedy=not args.skip_greedy,
                    max_new_tokens=args.max_new_tokens,
                )
                for result in results:
                    result["status"] = "ok"
            except ValueError as exc:
                if "not applicable" not in str(exc).lower():
                    raise
                not_applicable += 1
                results = [
                    {
                        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                        "experiment_id": "full_state_patch_answer_source_factorial",
                        "patch_condition": patch_condition,
                        "mask_condition": mask_condition,
                        "status": "not_applicable",
                        "exclusion_reason": str(exc),
                        "request_id": row["request_id"],
                        "model_label": args.model,
                        "seed": int(row["seed"]),
                        "gold_count": int(pair.gold_count),
                        "layer": int(args.layer),
                        "patch_geometry": str(geometry),
                        "patch_layer_mode": str(args.layer_mode),
                        "receiver_occurrence": receiver,
                        "donor_occurrence": donor,
                        "donor_offset": donor - receiver,
                    }
                    for patch_condition in args.patch_conditions
                    for mask_condition in args.mask_conditions
                ]
            for result in results:
                result.update(
                    {
                        "mechanism_split": args.seed_role,
                        "panel_kind": str(pair.panel_kind),
                        "selection_cell_id": str(pair.selection_cell_id),
                        "selection_rank": int(pair.selection_rank),
                        "pair_sha256": str(pair.pair_sha256),
                        "pair_plan": str(args.pair_plan.resolve()),
                        "pair_plan_sha256": _sha256(args.pair_plan),
                    }
                )
            _atomic_jsonl(shard, results)
            completed += 1
        print(
            f"[count-stream full-state-patch-source] "
            f"{pair_index}/{len(pair_plan)}",
            flush=True,
        )
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "not_applicable_geometry_shards_this_run": not_applicable,
                "planned_pair_count": int(len(pair_plan)),
                "planned_shard_count": int(len(pair_plan) * len(args.geometries)),
                "pair_plan": str(args.pair_plan.resolve()),
                "pair_plan_sha256": _sha256(args.pair_plan),
                "max_selection_rank": args.max_selection_rank,
                "min_selection_rank": args.min_selection_rank,
                "patch_layer": int(args.layer),
                "geometries": list(args.geometries),
                "layer_mode": args.layer_mode,
                "patch_conditions": list(args.patch_conditions),
                "mask_conditions": list(args.mask_conditions),
                "mask_application": args.mask_application,
                "long_forwards_per_pair_geometry": (
                    1 + len(args.patch_conditions)
                ),
                "query_branches_per_patched_prefix": len(args.mask_conditions),
                "claim_scope": "development_full_state_source_mediation",
            },
        ),
    )


def command_serial_patch_heads(args: argparse.Namespace) -> None:
    """Run the terminal-state x frozen answer-head serial-readout factorial."""

    started = time.perf_counter()
    mechanism = _spec(args)
    registered_layers = {"Qwen3-8B": 19, "Gemma4-E4B": 16}
    expected_layer = registered_layers[str(args.model)]
    if int(args.layer) != expected_layer:
        raise ValueError(
            f"{args.model} serial patch layer is frozen to L{expected_layer}"
        )
    if str(args.geometry) != "suffix8":
        raise ValueError("Serial readout geometry is frozen to suffix8")
    if str(args.layer_mode) != "cumulative_clamp":
        raise ValueError("Serial readout layer mode is frozen to cumulative_clamp")
    if getattr(args, "limit", None) is not None:
        raise ValueError("Formal serial readout does not permit row limits")

    rows = _registered_rows(args, mechanism)
    expected_seeds = set(
        mechanism.development_seeds
        if args.seed_role == "development"
        else mechanism.confirmation_seeds
    )
    observed_row_seeds = {int(row["seed"]) for row in rows}
    if observed_row_seeds != expected_seeds:
        raise ValueError(
            "Serial readout requires every canonical seed to contribute at least "
            f"one eligible row: expected={sorted(expected_seeds)} "
            f"observed={sorted(observed_row_seeds)}"
        )
    pair_plan = build_terminal_serial_pair_plan(rows, model_label=args.model)
    if set(pair_plan["seed"].astype(int)) != expected_seeds:
        raise ValueError("Terminal serial plan lost a canonical seed")
    if set(pair_plan["seed_role"].astype(str)) != {str(args.seed_role)}:
        raise ValueError("Terminal serial plan has the wrong seed role")

    head_arms, head_plan_audit = _load_frozen_serial_head_arms(
        args.head_plan,
        model=args.model,
        expected_random_controls=mechanism.random_controls,
    )
    output = Path(args.output)
    pair_plan_path = output / "terminal_serial_pair_plan.csv"
    _atomic_csv(pair_plan_path, pair_plan)
    cell_counts = (
        pair_plan.groupby(["gold_count", "donor_offset"], as_index=False)
        .agg(
            pair_count=("pair_sha256", "nunique"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["gold_count", "donor_offset"])
    )
    _atomic_csv(output / "terminal_serial_cell_counts.csv", cell_counts)

    model, tokenizer, adapter = _model(args)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = not_applicable = 0
    row_by_request = {str(row["request_id"]): row for row in rows}
    for pair_index, pair in enumerate(pair_plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        stem = _safe_stem(
            row["request_id"],
            "serial_patch_heads",
            pair.receiver_occurrence,
            pair.donor_occurrence,
            args.layer,
            args.geometry,
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        try:
            results = run_full_state_patch_head_readout_factorial_trials(
                model,
                tokenizer,
                adapter,
                row,
                receiver_occurrence=int(pair.receiver_occurrence),
                donor_occurrence=int(pair.donor_occurrence),
                layer=int(args.layer),
                geometry=str(args.geometry),
                layer_mode=str(args.layer_mode),
                head_arms=head_arms,
                source_group="trace_items",
                answer_site_id=mechanism.answer_site_id,
                run_greedy=not args.skip_greedy,
                max_new_tokens=int(args.max_new_tokens),
            )
            for result in results:
                result["status"] = "ok"
        except ValueError as exc:
            if "not applicable" not in str(exc).lower():
                raise
            not_applicable += 1
            results = [
                {
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                    "experiment_id": "full_state_patch_head_readout_factorial",
                    "patch_condition": patch_condition,
                    "head_condition": str(arm["condition"]),
                    "head_repeat": int(arm.get("repeat", 0)),
                    "status": "not_applicable",
                    "exclusion_reason": str(exc),
                    "request_id": row["request_id"],
                    "model_label": args.model,
                    "seed": int(row["seed"]),
                    "gold_count": int(pair.gold_count),
                    "layer": int(args.layer),
                    "patch_geometry": str(args.geometry),
                    "patch_layer_mode": str(args.layer_mode),
                    "receiver_occurrence": int(pair.receiver_occurrence),
                    "donor_occurrence": int(pair.donor_occurrence),
                    "donor_offset": int(pair.donor_offset),
                }
                for patch_condition in ("self_patch", "full_donor_patch")
                for arm in head_arms
            ]
        for result in results:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "selection_policy": str(pair.selection_policy),
                    "selection_cell_id": str(pair.selection_cell_id),
                    "within_cell_index": int(pair.within_cell_index),
                    "eligible_seed_count": int(pair.eligible_seed_count),
                    "pair_sha256": str(pair.pair_sha256),
                    "pair_plan": str(pair_plan_path.resolve()),
                    "pair_plan_sha256": _sha256(pair_plan_path),
                    **head_plan_audit,
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[count-stream serial-patch-heads] {pair_index}/{len(pair_plan)}",
            flush=True,
        )
    rows_per_pair = 2 * len(head_arms)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "not_applicable_pairs_this_run": not_applicable,
                "seed_role": args.seed_role,
                "registered_seeds": sorted(expected_seeds),
                "observed_pair_seeds": sorted(
                    int(value) for value in pair_plan["seed"].unique()
                ),
                "selection_policy": "all_eligible_registered_seeds",
                "pair_count": int(len(pair_plan)),
                "cell_count": int(len(cell_counts)),
                "rows_per_pair": rows_per_pair,
                "planned_trial_rows": int(len(pair_plan) * rows_per_pair),
                "pair_plan": str(pair_plan_path.resolve()),
                "pair_plan_sha256": _sha256(pair_plan_path),
                "patch_layer": int(args.layer),
                "patch_geometry": str(args.geometry),
                "patch_layer_mode": str(args.layer_mode),
                "patch_conditions": ["self_patch", "full_donor_patch"],
                "head_conditions": list(REGISTERED_HEAD_READOUT_CONDITIONS),
                **head_plan_audit,
                "claim_scope": "serial_trace_state_to_frozen_heads_to_answer",
            },
        ),
    )


def command_serial_patch_source(args: argparse.Namespace) -> None:
    """Cross terminal state patches with registered all-head source masks."""

    started = time.perf_counter()
    mechanism = _spec(args)
    registered_layers = {"Qwen3-8B": 19, "Gemma4-E4B": 16}
    expected_layer = registered_layers[str(args.model)]
    if int(args.layer) != expected_layer:
        raise ValueError(
            f"{args.model} serial patch layer is frozen to L{expected_layer}"
        )
    if str(args.geometry) != "suffix8":
        raise ValueError("Serial source geometry is frozen to suffix8")
    if str(args.layer_mode) != "cumulative_clamp":
        raise ValueError("Serial source layer mode is frozen to cumulative_clamp")
    if getattr(args, "limit", None) is not None:
        raise ValueError("Formal serial source readout does not permit row limits")
    required_masks = {
        "clean",
        "block_trace_items",
        "block_trace_items_matched_control",
        "block_prompt_records",
        "block_prompt_records_matched_control",
    }
    if set(args.mask_conditions) != required_masks or len(args.mask_conditions) != 5:
        raise ValueError("Serial source readout requires the frozen five-mask panel")

    rows = _registered_rows(args, mechanism)
    expected_seeds = set(
        mechanism.development_seeds
        if args.seed_role == "development"
        else mechanism.confirmation_seeds
    )
    observed_row_seeds = {int(row["seed"]) for row in rows}
    if observed_row_seeds != expected_seeds:
        raise ValueError(
            "Serial source readout requires every canonical seed: "
            f"expected={sorted(expected_seeds)} observed={sorted(observed_row_seeds)}"
        )
    pair_plan = build_terminal_serial_pair_plan(rows, model_label=args.model)
    if set(pair_plan["seed"].astype(int)) != expected_seeds:
        raise ValueError("Terminal serial source plan lost a canonical seed")
    if set(pair_plan["seed_role"].astype(str)) != {str(args.seed_role)}:
        raise ValueError("Terminal serial source plan has the wrong seed role")

    output = Path(args.output)
    pair_plan_path = output / "terminal_serial_pair_plan.csv"
    _atomic_csv(pair_plan_path, pair_plan)
    cell_counts = (
        pair_plan.groupby(["gold_count", "donor_offset"], as_index=False)
        .agg(
            pair_count=("pair_sha256", "nunique"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["gold_count", "donor_offset"])
    )
    _atomic_csv(output / "terminal_serial_cell_counts.csv", cell_counts)

    model, tokenizer, adapter = _model(args)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = not_applicable = 0
    row_by_request = {str(row["request_id"]): row for row in rows}
    for pair_index, pair in enumerate(pair_plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        stem = _safe_stem(
            row["request_id"],
            "serial_patch_source",
            pair.receiver_occurrence,
            pair.donor_occurrence,
            args.layer,
            args.geometry,
            args.mask_application,
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        try:
            results = run_full_state_patch_answer_source_factorial_trials(
                model,
                tokenizer,
                adapter,
                row,
                receiver_occurrence=int(pair.receiver_occurrence),
                donor_occurrence=int(pair.donor_occurrence),
                layer=int(args.layer),
                geometry=str(args.geometry),
                layer_mode=str(args.layer_mode),
                patch_conditions=("self_patch", "full_donor_patch"),
                mask_conditions=tuple(args.mask_conditions),
                answer_site_id=mechanism.answer_site_id,
                mask_application=str(args.mask_application),
                run_greedy=not args.skip_greedy,
                max_new_tokens=int(args.max_new_tokens),
            )
            for result in results:
                result["status"] = "ok"
        except ValueError as exc:
            if "not applicable" not in str(exc).lower():
                raise
            not_applicable += 1
            results = [
                {
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                    "experiment_id": "full_state_patch_answer_source_factorial",
                    "patch_condition": patch_condition,
                    "mask_condition": mask_condition,
                    "status": "not_applicable",
                    "exclusion_reason": str(exc),
                    "request_id": row["request_id"],
                    "model_label": args.model,
                    "seed": int(row["seed"]),
                    "gold_count": int(pair.gold_count),
                    "layer": int(args.layer),
                    "patch_geometry": str(args.geometry),
                    "patch_layer_mode": str(args.layer_mode),
                    "receiver_occurrence": int(pair.receiver_occurrence),
                    "donor_occurrence": int(pair.donor_occurrence),
                    "donor_offset": int(pair.donor_offset),
                    "mask_scope": str(args.mask_application),
                }
                for patch_condition in ("self_patch", "full_donor_patch")
                for mask_condition in args.mask_conditions
            ]
        for result in results:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "selection_policy": str(pair.selection_policy),
                    "selection_cell_id": str(pair.selection_cell_id),
                    "within_cell_index": int(pair.within_cell_index),
                    "eligible_seed_count": int(pair.eligible_seed_count),
                    "pair_sha256": str(pair.pair_sha256),
                    "pair_plan": str(pair_plan_path.resolve()),
                    "pair_plan_sha256": _sha256(pair_plan_path),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[count-stream serial-patch-source] {pair_index}/{len(pair_plan)}",
            flush=True,
        )
    rows_per_pair = 2 * len(args.mask_conditions)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "not_applicable_pairs_this_run": not_applicable,
                "seed_role": args.seed_role,
                "registered_seeds": sorted(expected_seeds),
                "observed_pair_seeds": sorted(
                    int(value) for value in pair_plan["seed"].unique()
                ),
                "selection_policy": "all_eligible_registered_seeds",
                "pair_count": int(len(pair_plan)),
                "cell_count": int(len(cell_counts)),
                "rows_per_pair": rows_per_pair,
                "planned_trial_rows": int(len(pair_plan) * rows_per_pair),
                "pair_plan": str(pair_plan_path.resolve()),
                "pair_plan_sha256": _sha256(pair_plan_path),
                "patch_layer": int(args.layer),
                "patch_geometry": str(args.geometry),
                "patch_layer_mode": str(args.layer_mode),
                "patch_conditions": ["self_patch", "full_donor_patch"],
                "mask_conditions": list(args.mask_conditions),
                "mask_application": str(args.mask_application),
                "claim_scope": "serial_trace_state_to_all_head_source_readout_to_answer",
            },
        ),
    )


def command_terminal_relay_mediation(args: argparse.Namespace) -> None:
    """Cross terminal donor states with clean resets of a later relay."""

    started = time.perf_counter()
    mechanism = _spec(args)
    expected_layers = {"Qwen3-8B": (19, 26), "Gemma4-E4B": (16, 34)}
    expected_source, expected_relay = expected_layers[str(args.model)]
    if (int(args.source_layer), int(args.relay_layer)) != (
        expected_source,
        expected_relay,
    ):
        raise ValueError(
            f"{args.model} relay assay is frozen to source L{expected_source} "
            f"and relay L{expected_relay}"
        )
    if str(args.geometry) != "suffix8":
        raise ValueError("Relay mediation geometry is frozen to suffix8")
    if getattr(args, "limit", None) is not None:
        raise ValueError("Formal relay mediation does not permit row limits")
    rows = _registered_rows(args, mechanism)
    expected_seeds = set(
        mechanism.development_seeds
        if args.seed_role == "development"
        else mechanism.confirmation_seeds
    )
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError("Relay mediation requires every canonical seed")
    pair_plan = build_terminal_serial_pair_plan(rows, model_label=args.model)
    if set(pair_plan["seed"].astype(int)) != expected_seeds:
        raise ValueError("Relay pair plan lost a canonical seed")

    output = Path(args.output)
    pair_plan_path = output / "terminal_relay_pair_plan.csv"
    _atomic_csv(pair_plan_path, pair_plan)
    cell_counts = (
        pair_plan.groupby(["gold_count", "donor_offset"], as_index=False)
        .agg(pair_count=("pair_sha256", "nunique"), seed_count=("seed", "nunique"))
        .sort_values(["gold_count", "donor_offset"])
    )
    _atomic_csv(output / "terminal_relay_cell_counts.csv", cell_counts)
    model, tokenizer, adapter = _model(args)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    row_by_request = {str(row["request_id"]): row for row in rows}
    completed = skipped = not_applicable = 0
    for pair_index, pair in enumerate(pair_plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        stem = _safe_stem(
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
            results = run_terminal_state_relay_reset_trials(
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
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
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
                for relay_condition in REGISTERED_RELAY_RESET_CONDITIONS
            ]
        for result in results:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "selection_policy": str(pair.selection_policy),
                    "selection_cell_id": str(pair.selection_cell_id),
                    "within_cell_index": int(pair.within_cell_index),
                    "eligible_seed_count": int(pair.eligible_seed_count),
                    "pair_sha256": str(pair.pair_sha256),
                    "pair_plan": str(pair_plan_path.resolve()),
                    "pair_plan_sha256": _sha256(pair_plan_path),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[count-stream terminal-relay] {pair_index}/{len(pair_plan)}", flush=True)
    rows_per_pair = 2 * len(REGISTERED_RELAY_RESET_CONDITIONS)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "not_applicable_pairs_this_run": not_applicable,
                "seed_role": args.seed_role,
                "registered_seeds": sorted(expected_seeds),
                "selection_policy": "all_eligible_registered_seeds",
                "pair_count": int(len(pair_plan)),
                "cell_count": int(len(cell_counts)),
                "rows_per_pair": rows_per_pair,
                "planned_trial_rows": int(len(pair_plan) * rows_per_pair),
                "pair_plan": str(pair_plan_path.resolve()),
                "pair_plan_sha256": _sha256(pair_plan_path),
                "source_layer": int(args.source_layer),
                "source_patch_layers": list(
                    range(int(args.source_layer), int(args.relay_layer))
                ),
                "relay_layer": int(args.relay_layer),
                "patch_geometry": str(args.geometry),
                "source_conditions": ["self_patch", "full_donor_patch"],
                "relay_conditions": list(REGISTERED_RELAY_RESET_CONDITIONS),
                "claim_scope": "terminal_state_to_pre_answer_residual_relay_to_answer",
            },
        ),
    )


def command_complementary_readout(args: argparse.Namespace) -> None:
    """Jointly cut Qwen's residual relay and direct trace-source reread."""

    started = time.perf_counter()
    mechanism = _spec(args)
    if str(args.model) != "Qwen3-8B":
        raise ValueError("Complementary readout is discovery-frozen for Qwen3-8B")
    if (int(args.source_layer), int(args.relay_layer)) != (19, 26):
        raise ValueError("Qwen complementary readout is frozen to L19 -> L26")
    if str(args.geometry) != "suffix8":
        raise ValueError("Complementary readout geometry is frozen to suffix8")
    if getattr(args, "limit", None) is not None:
        raise ValueError("Formal complementary readout does not permit row limits")
    rows = _registered_rows(args, mechanism)
    expected_seeds = set(
        mechanism.development_seeds
        if args.seed_role == "development"
        else mechanism.confirmation_seeds
    )
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError("Complementary readout requires every canonical seed")
    pair_plan = build_terminal_serial_pair_plan(rows, model_label=args.model)
    if set(pair_plan["seed"].astype(int)) != expected_seeds:
        raise ValueError("Complementary pair plan lost a canonical seed")

    output = Path(args.output)
    pair_plan_path = output / "complementary_pair_plan.csv"
    _atomic_csv(pair_plan_path, pair_plan)
    cell_counts = (
        pair_plan.groupby(["gold_count", "donor_offset"], as_index=False)
        .agg(pair_count=("pair_sha256", "nunique"), seed_count=("seed", "nunique"))
        .sort_values(["gold_count", "donor_offset"])
    )
    _atomic_csv(output / "complementary_cell_counts.csv", cell_counts)
    model, tokenizer, adapter = _model(args)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    row_by_request = {str(row["request_id"]): row for row in rows}
    completed = skipped = not_applicable = 0
    for pair_index, pair in enumerate(pair_plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        stem = _safe_stem(
            row["request_id"],
            "complementary_readout",
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
            results = run_terminal_state_complementary_readout_trials(
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
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                    "experiment_id": "terminal_state_complementary_readout",
                    "patch_condition": patch_condition,
                    "relay_condition": relay_condition,
                    "mask_condition": mask_condition,
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
                for patch_condition in ("self_patch", "full_donor_patch")
                for relay_condition in REGISTERED_COMPLEMENTARY_RELAY_CONDITIONS
                for mask_condition in REGISTERED_COMPLEMENTARY_SOURCE_CONDITIONS
            ]
        for result in results:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "selection_policy": str(pair.selection_policy),
                    "selection_cell_id": str(pair.selection_cell_id),
                    "within_cell_index": int(pair.within_cell_index),
                    "eligible_seed_count": int(pair.eligible_seed_count),
                    "pair_sha256": str(pair.pair_sha256),
                    "pair_plan": str(pair_plan_path.resolve()),
                    "pair_plan_sha256": _sha256(pair_plan_path),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[count-stream complementary-readout] "
            f"{pair_index}/{len(pair_plan)}",
            flush=True,
        )
    rows_per_pair = (
        2
        * len(REGISTERED_COMPLEMENTARY_RELAY_CONDITIONS)
        * len(REGISTERED_COMPLEMENTARY_SOURCE_CONDITIONS)
    )
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "not_applicable_pairs_this_run": not_applicable,
                "seed_role": args.seed_role,
                "registered_seeds": sorted(expected_seeds),
                "selection_policy": "all_eligible_registered_seeds",
                "pair_count": int(len(pair_plan)),
                "cell_count": int(len(cell_counts)),
                "rows_per_pair": rows_per_pair,
                "planned_trial_rows": int(len(pair_plan) * rows_per_pair),
                "pair_plan": str(pair_plan_path.resolve()),
                "pair_plan_sha256": _sha256(pair_plan_path),
                "source_layer": int(args.source_layer),
                "source_patch_layers": list(
                    range(int(args.source_layer), int(args.relay_layer))
                ),
                "relay_layer": int(args.relay_layer),
                "patch_geometry": str(args.geometry),
                "patch_conditions": ["self_patch", "full_donor_patch"],
                "relay_conditions": list(
                    REGISTERED_COMPLEMENTARY_RELAY_CONDITIONS
                ),
                "mask_conditions": list(
                    REGISTERED_COMPLEMENTARY_SOURCE_CONDITIONS
                ),
                "mask_application": "answer_query_and_answer_tokens",
                "claim_scope": (
                    "terminal_state_to_complementary_residual_and_direct_"
                    "trace_readout_to_answer"
                ),
            },
        ),
    )


def _load_integrated_bridge_banks(path: Path, *, model: str) -> list[dict[str, Any]]:
    plan = pd.read_csv(path)
    if "selection_rank" in plan.columns:
        raise ValueError("Integrated bridge bank plan must not contain selection_rank")
    required = {"model_label", "condition", "repeat", "heads", "bank_sha256"}
    missing = sorted(required - set(plan.columns))
    if missing:
        raise ValueError(f"Integrated bridge bank plan lacks {missing}")
    plan = plan.loc[plan["model_label"].astype(str).eq(str(model))].copy()
    counts = plan["condition"].astype(str).value_counts().to_dict()
    if counts != {"layer_matched_random": 3, "selected_bank": 1}:
        raise ValueError(f"Integrated bridge needs one selected and 3 random banks: {counts}")
    banks: list[dict[str, Any]] = [
        {
            "condition": "clean",
            "repeat": 0,
            "heads": [],
            "bank_sha256": "clean",
        }
    ]
    for row in plan.sort_values(["condition", "repeat"]).itertuples(index=False):
        heads = json.loads(str(row.heads))
        if not heads or len(heads) != len({tuple(value) for value in heads}):
            raise ValueError("Integrated bridge found an empty or duplicate head bank")
        banks.append(
            {
                "condition": str(row.condition),
                "repeat": int(row.repeat),
                "heads": [[int(layer), int(head)] for layer, head in heads],
                "bank_sha256": str(row.bank_sha256),
            }
        )
    return banks


def _load_integrated_anchor_registry(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    required = {
        "request_id",
        "seed",
        "gold_count",
        "from_occurrence",
        "to_occurrence",
        "anchor_equivalence_id",
    }
    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Integrated anchor registry row lacks {missing}")
        request_id = str(row["request_id"])
        if request_id in registry:
            raise ValueError(f"Duplicate integrated routed anchor for {request_id}")
        if int(row["from_occurrence"]) != int(row["gold_count"]) - 1 or int(
            row["to_occurrence"]
        ) != int(row["gold_count"]):
            raise ValueError("Integrated registry contains a non-final transition")
        registry[request_id] = dict(row)
    if not registry:
        raise ValueError("Integrated anchor registry is empty")
    return registry


def command_integrated_serial_bridge(args: argparse.Namespace) -> None:
    """Bridge frozen targeted retrieval into state and confirmed readout routes."""

    started = time.perf_counter()
    mechanism = _spec(args)
    if getattr(args, "limit", None) is not None:
        raise ValueError("Formal integrated bridge does not permit row limits")
    rows = _registered_rows(args, mechanism)
    expected_seeds = set(
        mechanism.development_seeds
        if args.seed_role == "development"
        else mechanism.confirmation_seeds
    )
    anchor_registry = _load_integrated_anchor_registry(args.anchor_registry)
    rows = [row for row in rows if str(row["request_id"]) in anchor_registry]
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError("Integrated bridge requires every canonical seed")
    banks = _load_integrated_bridge_banks(args.bank_plan, model=args.model)
    patch_layers = (
        tuple(range(19, 26))
        if str(args.model) == "Qwen3-8B"
        else tuple(range(16, 42))
    )
    relay_layer = 26 if str(args.model) == "Qwen3-8B" else None
    output = Path(args.output)
    sample_plan = pd.DataFrame(
        [
            {
                "request_id": str(row["request_id"]),
                "model_label": str(args.model),
                "seed": int(row["seed"]),
                "gold_count": int(row["gold_count"]),
                "seed_role": str(args.seed_role),
                "selection_policy": "all_eligible_registered_seeds_and_terminal_counts",
            }
            for row in rows
        ]
    ).sort_values(["seed", "gold_count", "request_id"])
    if sample_plan.duplicated(["seed", "gold_count"]).any():
        raise ValueError("Integrated bridge requires one row per seed/count cell")
    sample_plan_path = output / "integrated_bridge_sample_plan.csv"
    _atomic_csv(sample_plan_path, sample_plan)

    model, tokenizer, adapter = _model(args)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = not_applicable = 0
    rows_per_sample = (
        len(banks)
        * len(INTEGRATED_BRIDGE_READOUT_CONDITIONS)
        * (2 if str(args.bridge_design) == "restoration" else 1)
    )
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(row["request_id"], "integrated_serial_bridge")
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        try:
            results = run_integrated_serial_bridge_trials(
                model,
                tokenizer,
                adapter,
                row,
                banks=banks,
                targeted_site=anchor_registry[str(row["request_id"])],
                patch_layers=patch_layers,
                model_label=str(args.model),
                geometry=str(args.geometry),
                relay_layer=relay_layer,
                write_window=str(args.write_window),
                bridge_design=str(args.bridge_design),
                answer_site_id=mechanism.answer_site_id,
                run_greedy=not args.skip_greedy,
                max_new_tokens=int(args.max_new_tokens),
            )
        except ValueError as exc:
            if "not applicable" not in str(exc).lower():
                raise
            not_applicable += 1
            cells = (
                [
                    {
                        "receiver": None,
                        "mediator": bank,
                        "mediator_condition": "transferred_state",
                    }
                    for bank in banks
                ]
                if str(args.bridge_design) == "transfer"
                else [
                    {
                        "receiver": receiver,
                        "mediator": (
                            receiver
                            if mediator_condition == "self_state"
                            else banks[0]
                        ),
                        "mediator_condition": mediator_condition,
                    }
                    for receiver in banks
                    for mediator_condition in ("self_state", "clean_state_restore")
                ]
            )
            results = [
                {
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                    "experiment_id": (
                        "integrated_targeted_state_readout_bridge"
                        if str(args.bridge_design) == "transfer"
                        else "integrated_targeted_mediator_restoration"
                    ),
                    "bridge_design": str(args.bridge_design),
                    "request_id": str(row["request_id"]),
                    "model_label": str(args.model),
                    "seed": int(row["seed"]),
                    "gold_count": int(row["gold_count"]),
                    "write_condition": str(cell["mediator"]["condition"]),
                    "write_repeat": int(cell["mediator"]["repeat"]),
                    "bank_sha256": str(cell["mediator"]["bank_sha256"]),
                    "receiver_write_condition": (
                        "clean_receiver"
                        if cell["receiver"] is None
                        else str(cell["receiver"]["condition"])
                    ),
                    "receiver_write_repeat": (
                        0
                        if cell["receiver"] is None
                        else int(cell["receiver"]["repeat"])
                    ),
                    "mediator_condition": str(cell["mediator_condition"]),
                    "mediator_state_source": str(cell["mediator"]["condition"]),
                    "greedy_generation_run": False,
                    "readout_condition": readout,
                    "status": "not_applicable",
                    "exclusion_reason": str(exc),
                    "state_patch_geometry": str(args.geometry),
                    "write_window": str(args.write_window),
                }
                for cell in cells
                for readout in INTEGRATED_BRIDGE_READOUT_CONDITIONS
            ]
        for result in results:
            result.update(
                {
                    "mechanism_split": str(args.seed_role),
                    "selection_policy": (
                        "all_eligible_registered_seeds_and_terminal_counts"
                    ),
                    "bank_plan": str(args.bank_plan.resolve()),
                    "bank_plan_sha256": _sha256(args.bank_plan),
                    "anchor_registry": str(args.anchor_registry.resolve()),
                    "anchor_registry_sha256": _sha256(args.anchor_registry),
                    "sample_plan": str(sample_plan_path.resolve()),
                    "sample_plan_sha256": _sha256(sample_plan_path),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[count-stream integrated-serial-bridge] {index}/{len(rows)}",
            flush=True,
        )
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "not_applicable_samples_this_run": not_applicable,
                "seed_role": str(args.seed_role),
                "registered_seeds": sorted(expected_seeds),
                "selection_policy": (
                    "all_eligible_registered_seeds_and_terminal_counts"
                ),
                "selection_rank_used": False,
                "sample_count": int(len(sample_plan)),
                "rows_per_sample": int(rows_per_sample),
                "planned_trial_rows": int(len(sample_plan) * rows_per_sample),
                "bank_plan": str(args.bank_plan.resolve()),
                "bank_plan_sha256": _sha256(args.bank_plan),
                "anchor_registry": str(args.anchor_registry.resolve()),
                "anchor_registry_sha256": _sha256(args.anchor_registry),
                "sample_plan": str(sample_plan_path.resolve()),
                "sample_plan_sha256": _sha256(sample_plan_path),
                "patch_layers": list(patch_layers),
                "relay_layer": relay_layer,
                "patch_geometry": str(args.geometry),
                "readout_conditions": list(INTEGRATED_BRIDGE_READOUT_CONDITIONS),
                "teacher_forced_trace": True,
                "write_window": str(args.write_window),
                "bridge_design": str(args.bridge_design),
                "greedy_generation_policy": (
                    "clean_receiver_self_state_natural_only"
                    if str(args.bridge_design) == "restoration"
                    else "all_cells_when_enabled"
                ),
                "claim_scope": (
                    "targeted_query_to_terminal_state_to_confirmed_readout_bridge"
                ),
            },
        ),
    )


def command_restoration(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(row["request_id"], "restoration", args.layer)
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = run_trace_restoration_trials(
            model,
            tokenizer,
            adapter,
            row,
            layer=args.layer,
            conditions=args.conditions,
            answer_site_id=mechanism.answer_site_id,
            run_greedy=not args.skip_greedy,
            max_new_tokens=args.max_new_tokens,
        )
        for result in results:
            result["mechanism_split"] = args.seed_role
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[count-stream restoration] {index}/{len(rows)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "restoration_layer": int(args.layer),
            },
        ),
    )


def _trial_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif (path / "shards").is_dir():
            files.extend(sorted((path / "shards").glob("*.jsonl")))
        else:
            files.extend(sorted(path.rglob("*.jsonl")))
    if not files:
        raise FileNotFoundError("No JSONL trial shards were found")
    return files


def command_analyze(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = [row for file in _trial_files(args.trials) for row in read_jsonl(file)]
    trials = pd.DataFrame(rows)
    experiments = (
        args.experiment_ids
        if args.experiment_ids
        else sorted(trials["experiment_id"].dropna().astype(str).unique())
    )
    summaries = []
    effects = []
    for experiment_id in experiments:
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
            # Multiple K values are development curves, not one registered
            # confirmation estimand.  Keep them separate so the decision
            # ledger marks an unfrozen source gate as ambiguous.
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
        summary, seed_effects = summarize_linear_contrasts(
            trials,
            experiment_id=experiment_id,
            outcome=args.outcome,
            bootstrap_samples=mechanism.bootstrap_samples,
            random_seed=args.random_seed,
            stratum_columns=strata,
        )
        summaries.append(summary)
        effects.append(seed_effects)
    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_effects = pd.concat(effects, ignore_index=True)
    ledger = mechanism_decision_ledger(combined_summary)
    ledger["design_status"] = mechanism.status
    ledger["formal_inference_eligible"] = mechanism.formal_inference_eligible
    ledger["claim_scope"] = (
        "fresh_confirmation"
        if mechanism.formal_inference_eligible
        else "development_only_no_confirmatory_claim"
    )
    output = Path(args.output)
    _atomic_csv(output / "estimands.csv", combined_summary)
    _atomic_csv(output / "seed_effects.csv", combined_effects)
    _atomic_csv(output / "mechanism_decision_ledger.csv", ledger)
    trial_files = _trial_files(args.trials)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "trial_files": [str(path.resolve()) for path in trial_files],
                "experiment_ids": list(experiments),
                "outcome": args.outcome,
                "estimands_sha256": _sha256(output / "estimands.csv"),
                "claim_policy": (
                    "Decision ledger reports registered component gates only; "
                    "it never upgrades them to a unique-circuit or scalar-counter claim."
                ),
            },
        ),
    )


def _add_configs(parser: argparse.ArgumentParser, *, include_v5: bool = True) -> None:
    parser.add_argument(
        "--mechanism-config", type=Path, default=DEFAULT_MECHANISM_CONFIG
    )
    if include_v5:
        parser.add_argument("--v5-config", type=Path, default=DEFAULT_V5_CONFIG)


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")


def _add_rows(parser: argparse.ArgumentParser) -> None:
    _add_configs(parser)
    _add_model(parser)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument(
        "--seed-role",
        choices=["development", "confirmation"],
        default="development",
    )
    parser.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
        help=(
            "Native trace cohort; one_to_one is the registered default, while "
            "one_to_one_correct is secondary robustness only."
        ),
    )
    parser.add_argument(
        "--row-panel",
        choices=[
            "all",
            "broad_ranking",
            "broad_k_selection",
            "broad_confirmation",
            "trace_patch",
        ],
        default="all",
        help="Frozen seed/count sampling panel; all is for legacy/dev diagnostics.",
    )
    parser.add_argument("--limit", type=int)


def _add_output_resume(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)


def _add_behavior(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Native-thinking count-stream versus final broad-retrieval mechanism"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture-broad",
        help="Capture answer-query attention to prompt/trace source partitions.",
    )
    _add_rows(capture)
    _add_output_resume(capture)
    capture.add_argument("--layers", type=int, nargs="+")
    capture.set_defaults(
        func=command_capture_broad,
        cohort="parser_hit",
        row_panel="broad_ranking",
        seed_role="development",
    )

    plan = subparsers.add_parser(
        "plan-broad",
        help="Freeze development-ranked broad banks and exact matched controls.",
    )
    _add_configs(plan, include_v5=False)
    plan.add_argument("--captures", type=Path, required=True)
    plan.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    plan.add_argument(
        "--source-group",
        required=True,
        choices=[
            "prompt_records",
            "trace_context",
            "trace_items",
            "trace_other",
            "trace_markers",
            "trace_nonmarkers",
            "earlier_trace_items",
            "terminal_trace_item",
        ],
    )
    plan.add_argument("--bank-sizes", type=int, nargs="+")
    plan.add_argument(
        "--use-all-development-seeds",
        action="store_true",
        help=(
            "Rank a fixed-K bank on all 20 development seeds instead of the "
            "legacy 10-seed ranking half. This is appropriate only when K is "
            "fixed prospectively and no separate K-selection fold is used."
        ),
    )
    plan.add_argument("--random-seed", type=int, default=0)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(func=command_plan_broad)

    trace_plan = subparsers.add_parser(
        "plan-trace-patch",
        help="Freeze the outcome-blind sparse directed trace-patch pair panel.",
    )
    _add_rows(trace_plan)
    trace_plan.add_argument("--output", type=Path, required=True)
    trace_plan.set_defaults(
        func=command_plan_trace_patch,
        row_panel="trace_patch",
        seed_role="development",
    )

    terminal_last_plan = subparsers.add_parser(
        "plan-terminal-last-patch",
        help="Freeze natural donors to the final trace-count receiver.",
    )
    _add_rows(terminal_last_plan)
    terminal_last_plan.add_argument("--seeds-per-cell", type=int)
    terminal_last_plan.add_argument("--output", type=Path, required=True)
    terminal_last_plan.set_defaults(
        func=command_plan_terminal_last_patch,
        row_panel="trace_patch",
        seed_role="development",
    )

    select_k = subparsers.add_parser(
        "select-broad-k",
        help="Select and freeze one model/source K from development dose outcomes.",
    )
    _add_configs(select_k, include_v5=False)
    select_k.add_argument("--trials", type=Path, nargs="+", required=True)
    select_k.add_argument("--plan", type=Path, required=True)
    select_k.add_argument(
        "--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"]
    )
    select_k.add_argument(
        "--source-group", required=True, choices=["trace_items", "prompt_records"]
    )
    select_k.add_argument("--random-seed", type=int, default=0)
    select_k.add_argument("--output", type=Path, required=True)
    select_k.set_defaults(func=command_select_broad_k)

    source_mask = subparsers.add_parser(
        "source-mask",
        help="Run the final-query prompt/trace source factorial.",
    )
    _add_rows(source_mask)
    _add_output_resume(source_mask)
    _add_behavior(source_mask)
    source_mask.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_MASK_CONDITIONS),
        default=list(REGISTERED_MASK_CONDITIONS),
    )
    source_mask.add_argument(
        "--mask-application",
        choices=["answer_query_only", "answer_query_and_answer_tokens"],
        default="answer_query_only",
        help=(
            "Query-only matches the top-K answer-query ablation; persistent "
            "also prevents later numeric answer tokens from rereading sources."
        ),
    )
    source_mask.set_defaults(func=command_source_mask)

    broad_heads = subparsers.add_parser(
        "broad-heads",
        help="Ablate a frozen answer-query broad bank and matched random banks.",
    )
    _add_rows(broad_heads)
    _add_output_resume(broad_heads)
    _add_behavior(broad_heads)
    broad_heads.add_argument("--plan", type=Path, required=True)
    broad_size_group = broad_heads.add_mutually_exclusive_group(required=True)
    broad_size_group.add_argument("--bank-size", type=int)
    broad_size_group.add_argument("--bank-sizes", type=int, nargs="+")
    broad_heads.set_defaults(
        func=command_broad_heads,
        cohort="parser_hit",
        row_panel="broad_k_selection",
    )

    fit_basis = subparsers.add_parser(
        "fit-basis",
        help="Fit a development-only progress/count basis plus orthogonal control.",
    )
    _add_configs(fit_basis, include_v5=False)
    fit_basis.add_argument("--capture-index", type=Path, required=True)
    fit_basis.add_argument("--site-kind", required=True)
    fit_basis.add_argument("--site-id")
    fit_basis.add_argument(
        "--capture-split",
        choices=["discovery", "confirmation"],
        help=(
            "Filter by the capture's frozen split. Use discovery for any basis "
            "that will later be evaluated on confirmation rows."
        ),
    )
    fit_basis.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    fit_basis.add_argument(
        "--label", choices=["occurrence", "gold_count"], required=True
    )
    fit_basis.add_argument("--layers", type=int, nargs="+")
    fit_basis.add_argument("--rank", type=int, default=3)
    fit_basis.add_argument("--random-seed", type=int, default=0)
    fit_basis.add_argument("--output", type=Path, required=True)
    fit_basis.set_defaults(func=command_fit_basis)

    stream = subparsers.add_parser(
        "stream-state",
        help="Remove running-index state at item endpoints and follow the suffix.",
    )
    _add_rows(stream)
    _add_output_resume(stream)
    _add_behavior(stream)
    stream.add_argument("--basis", type=Path, required=True)
    stream.add_argument("--source-layer", type=int, required=True)
    stream.add_argument("--readout-layers", type=int, nargs="+", default=[])
    stream.add_argument(
        "--scope", choices=["occurrence", "prefix", "all"], default="occurrence"
    )
    stream.add_argument("--occurrences", type=int, nargs="+")
    stream.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_STREAM_CONDITIONS),
        default=list(REGISTERED_STREAM_CONDITIONS),
    )
    stream.add_argument("--random-seed", type=int, default=0)
    stream.set_defaults(func=command_stream_state)

    joint = subparsers.add_parser(
        "joint-state-source",
        help=(
            "Cross discovery-fitted item-end count-state removal with all-head "
            "answer-query prompt/trace source masks."
        ),
    )
    _add_rows(joint)
    _add_output_resume(joint)
    _add_behavior(joint)
    joint.add_argument("--basis", type=Path, required=True)
    joint.add_argument("--source-layer", type=int, required=True)
    joint.add_argument(
        "--state-scope",
        choices=["terminal", "occurrence", "prefix", "all"],
        default="all",
        help=(
            "all is the stored-state upper bound; terminal targets only the "
            "last registered item endpoint."
        ),
    )
    joint.add_argument("--state-occurrence", type=int)
    joint.add_argument(
        "--state-conditions",
        nargs="+",
        choices=list(REGISTERED_STREAM_CONDITIONS),
        default=list(REGISTERED_STREAM_CONDITIONS),
    )
    joint.add_argument(
        "--mask-conditions",
        nargs="+",
        choices=list(REGISTERED_MASK_CONDITIONS),
        default=[
            "clean",
            "block_trace_items",
            "block_trace_items_matched_control",
            "block_prompt_records",
            "block_prompt_records_matched_control",
        ],
    )
    joint.add_argument(
        "--mask-application",
        choices=["answer_query_only", "answer_query_and_answer_tokens"],
        default="answer_query_only",
    )
    joint.add_argument("--random-seed", type=int, default=20260820)
    joint.set_defaults(func=command_joint_state_source)

    native_plan = subparsers.add_parser(
        "plan-native-loop",
        help=(
            "Freeze the rank-free P0 steering and endpoint boundary pair plan."
        ),
    )
    _add_rows(native_plan)
    native_plan.add_argument(
        "--donor-offsets",
        type=int,
        nargs="+",
        default=[-3, -2, -1, 1, 2, 3],
    )
    native_plan.add_argument("--random-seed", type=int, default=20260821)
    native_plan.add_argument("--allow-incomplete-offsets", action="store_true")
    native_plan.add_argument("--no-boundaries", action="store_true")
    native_plan.add_argument("--output", type=Path, required=True)
    native_plan.set_defaults(func=command_plan_native_loop, row_panel="all")

    query_head_plan = subparsers.add_parser(
        "plan-p0-query-mediation-heads",
        help=(
            "Freeze downstream targeted heads and a disjoint layer-matched "
            "query-local control bank."
        ),
    )
    _add_model(query_head_plan)
    query_head_plan.add_argument("--layer", type=int, required=True)
    query_head_plan.add_argument(
        "--targeted-selection", type=Path, required=True
    )
    query_head_plan.add_argument("--anchor-routing", type=Path, required=True)
    query_head_plan.add_argument("--candidate-ranking", type=Path, required=True)
    query_head_plan.add_argument("--random-seed", type=int, default=20260821)
    query_head_plan.add_argument("--output", type=Path, required=True)
    query_head_plan.set_defaults(func=command_plan_query_mediation_heads)

    p0_loop = subparsers.add_parser(
        "p0-native-loop",
        help=(
            "Patch P0 count state and jointly measure probe, routed attention, "
            "and first-city behavior."
        ),
    )
    _add_rows(p0_loop)
    _add_output_resume(p0_loop)
    _add_behavior(p0_loop)
    p0_loop.add_argument("--plan", type=Path, required=True)
    p0_loop.add_argument("--basis", type=Path, required=True)
    p0_loop.add_argument("--layer", type=int, required=True)
    p0_loop.add_argument("--targeted-selection", type=Path, required=True)
    p0_loop.add_argument("--anchor-routing", type=Path, required=True)
    p0_loop.add_argument(
        "--conditions",
        nargs="+",
        choices=list(AVAILABLE_P0_LOOP_CONDITIONS),
        default=list(REGISTERED_P0_LOOP_CONDITIONS),
    )
    p0_loop.add_argument(
        "--donor-offsets",
        type=int,
        nargs="+",
        default=[-3, -2, -1, 1, 2, 3],
    )
    p0_loop.add_argument("--random-seed", type=int, default=20260821)
    p0_loop.add_argument("--allow-incomplete-offsets", action="store_true")
    p0_loop.add_argument("--no-boundaries", action="store_true")
    p0_loop.set_defaults(
        func=command_p0_native_loop,
        max_new_tokens=48,
        cohort="one_to_one",
        row_panel="all",
    )

    query_mediation = subparsers.add_parser(
        "p0-query-mediation",
        help=(
            "Cross a P0 state patch with query-local targeted-head masks and "
            "selected-head output restoration."
        ),
    )
    _add_rows(query_mediation)
    _add_output_resume(query_mediation)
    _add_behavior(query_mediation)
    query_mediation.add_argument("--plan", type=Path, required=True)
    query_mediation.add_argument("--basis", type=Path, required=True)
    query_mediation.add_argument("--layer", type=int, required=True)
    query_mediation.add_argument(
        "--geometry",
        choices=list(REGISTERED_QUERY_MEDIATION_GEOMETRIES),
        required=True,
    )
    query_mediation.add_argument(
        "--targeted-selection", type=Path, required=True
    )
    query_mediation.add_argument("--anchor-routing", type=Path, required=True)
    query_mediation.add_argument("--head-plan", type=Path, required=True)
    query_mediation.add_argument(
        "--donor-offsets",
        type=int,
        nargs="+",
        default=[-3, -2, -1, 1, 2, 3],
    )
    query_mediation.add_argument("--random-seed", type=int, default=20260821)
    query_mediation.add_argument("--allow-incomplete-offsets", action="store_true")
    query_mediation.add_argument("--no-boundaries", action="store_true")
    query_mediation.set_defaults(
        func=command_p0_query_mediation,
        max_new_tokens=32,
        cohort="one_to_one",
        row_panel="all",
    )

    boundary_loop = subparsers.add_parser(
        "boundary-native-loop",
        help=(
            "Transplant middle/terminal endpoint states and measure "
            "continue-stop behavior."
        ),
    )
    _add_rows(boundary_loop)
    _add_output_resume(boundary_loop)
    _add_behavior(boundary_loop)
    boundary_loop.add_argument("--plan", type=Path, required=True)
    boundary_loop.add_argument("--basis", type=Path, required=True)
    boundary_loop.add_argument("--layer", type=int, required=True)
    boundary_loop.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_BOUNDARY_CONDITIONS),
        default=list(REGISTERED_BOUNDARY_CONDITIONS),
    )
    boundary_loop.add_argument(
        "--donor-offsets",
        type=int,
        nargs="+",
        default=[-3, -2, -1, 1, 2, 3],
    )
    boundary_loop.add_argument("--random-seed", type=int, default=20260821)
    boundary_loop.add_argument("--allow-incomplete-offsets", action="store_true")
    boundary_loop.add_argument("--no-boundaries", action="store_true")
    boundary_loop.set_defaults(
        func=command_boundary_native_loop,
        max_new_tokens=64,
        cohort="one_to_one",
        row_panel="all",
    )

    trace_patch = subparsers.add_parser(
        "trace-patch",
        help=("Patch directed donor states between intermediate trace item endpoints."),
    )
    _add_rows(trace_patch)
    _add_output_resume(trace_patch)
    _add_behavior(trace_patch)
    trace_patch.add_argument("--pair-plan", type=Path, required=True)
    trace_patch.add_argument("--basis", type=Path, required=True)
    trace_patch.add_argument("--layer", type=int, required=True)
    trace_patch.add_argument("--readout-layers", type=int, nargs="+", default=[])
    trace_patch.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_TRACE_PATCH_CONDITIONS),
        default=list(REGISTERED_TRACE_PATCH_CONDITIONS),
    )
    trace_patch.add_argument("--random-seed", type=int, default=0)
    trace_patch.set_defaults(
        func=command_trace_patch,
        max_new_tokens=48,
        cohort="one_to_one",
        row_panel="trace_patch",
    )

    full_state_patch = subparsers.add_parser(
        "trace-full-state-patch",
        help=(
            "Patch endpoint, suffix, or whole-span hidden tensors once or "
            "cumulatively within the native trace."
        ),
    )
    _add_rows(full_state_patch)
    _add_output_resume(full_state_patch)
    _add_behavior(full_state_patch)
    full_state_patch.add_argument("--pair-plan", type=Path, required=True)
    full_state_patch.add_argument(
        "--plan-kind",
        choices=["sparse_local", "terminal_last"],
        required=True,
    )
    full_state_patch.add_argument("--basis", type=Path, required=True)
    full_state_patch.add_argument("--layer", type=int, required=True)
    full_state_patch.add_argument(
        "--readout-layers", type=int, nargs="+", default=[]
    )
    full_state_patch.add_argument(
        "--geometries",
        nargs="+",
        choices=list(REGISTERED_TRACE_PATCH_GEOMETRIES),
        default=list(REGISTERED_TRACE_PATCH_GEOMETRIES),
    )
    full_state_patch.add_argument(
        "--layer-modes",
        nargs="+",
        choices=list(REGISTERED_TRACE_PATCH_LAYER_MODES),
        default=list(REGISTERED_TRACE_PATCH_LAYER_MODES),
    )
    full_state_patch.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_TRACE_FULL_STATE_CONDITIONS),
        default=list(REGISTERED_TRACE_FULL_STATE_CONDITIONS),
    )
    full_state_patch.add_argument("--max-selection-rank", type=int)
    full_state_patch.add_argument("--selection-cell-ids", nargs="+")
    full_state_patch.set_defaults(
        func=command_trace_full_state_patch,
        max_new_tokens=16,
        cohort="one_to_one",
        row_panel="trace_patch",
    )

    html_span = subparsers.add_parser(
        "html-aligned-terminal-span",
        help=(
            "Reproduce the old HTML's same-position, all-trace-control "
            "terminal full-span cumulative ablation/restoration."
        ),
    )
    _add_rows(html_span)
    _add_output_resume(html_span)
    _add_behavior(html_span)
    html_span.add_argument("--layer", type=int, required=True)
    html_span.add_argument("--random-seed", type=int, default=20260821)
    html_span.set_defaults(
        func=command_html_aligned_terminal_span,
        max_new_tokens=16,
        cohort="one_to_one",
        row_panel="all",
    )

    html_local = subparsers.add_parser(
        "html-aligned-local-serial",
        help=(
            "Cross a penultimate full-item clean/control span with frozen "
            "targeted query heads at the next city retrieval."
        ),
    )
    _add_rows(html_local)
    _add_output_resume(html_local)
    _add_behavior(html_local)
    html_local.add_argument("--layer", type=int, required=True)
    html_local.add_argument("--targeted-selection", type=Path, required=True)
    html_local.add_argument("--anchor-routing", type=Path, required=True)
    html_local.add_argument("--head-plan", type=Path, required=True)
    html_local.add_argument(
        "--head-token-geometry",
        choices=["query_only", "query_plus_full_path"],
        default="query_only",
    )
    html_local.add_argument("--random-seed", type=int, default=20260821)
    html_local.set_defaults(
        func=command_html_aligned_local_serial,
        max_new_tokens=32,
        cohort="one_to_one",
        row_panel="all",
    )

    patch_source = subparsers.add_parser(
        "full-state-patch-source",
        help=(
            "Cross terminal full-state donor patches with prompt/trace source "
            "masks at the final answer query."
        ),
    )
    _add_rows(patch_source)
    _add_output_resume(patch_source)
    _add_behavior(patch_source)
    patch_source.add_argument("--pair-plan", type=Path, required=True)
    patch_source.add_argument("--layer", type=int, required=True)
    patch_source.add_argument(
        "--geometries",
        nargs="+",
        choices=list(REGISTERED_TRACE_PATCH_GEOMETRIES),
        default=["suffix8"],
    )
    patch_source.add_argument(
        "--layer-mode",
        choices=list(REGISTERED_TRACE_PATCH_LAYER_MODES),
        default="cumulative_clamp",
    )
    patch_source.add_argument(
        "--patch-conditions",
        nargs="+",
        choices=list(REGISTERED_TRACE_FULL_STATE_CONDITIONS),
        default=["self_patch", "full_donor_patch"],
    )
    patch_source.add_argument(
        "--mask-conditions",
        nargs="+",
        choices=list(REGISTERED_MASK_CONDITIONS),
        default=[
            "clean",
            "block_trace_items",
            "block_trace_items_matched_control",
            "block_prompt_records",
            "block_prompt_records_matched_control",
        ],
    )
    patch_source.add_argument(
        "--mask-application",
        choices=["answer_query_only", "answer_query_and_answer_tokens"],
        default="answer_query_only",
    )
    patch_source.add_argument("--max-selection-rank", type=int, default=5)
    patch_source.add_argument("--min-selection-rank", type=int, default=1)
    patch_source.set_defaults(
        func=command_full_state_patch_source,
        max_new_tokens=16,
        cohort="one_to_one",
        row_panel="trace_patch",
    )

    serial_heads = subparsers.add_parser(
        "serial-patch-heads",
        help=(
            "Cross terminal suffix-8 donor patches with one discovery-frozen "
            "trace-head bank and disjoint layer-matched random banks."
        ),
    )
    _add_rows(serial_heads)
    _add_output_resume(serial_heads)
    _add_behavior(serial_heads)
    serial_heads.add_argument("--head-plan", type=Path, required=True)
    serial_heads.add_argument("--layer", type=int, required=True)
    serial_heads.add_argument(
        "--geometry",
        choices=list(REGISTERED_TRACE_PATCH_GEOMETRIES),
        default="suffix8",
    )
    serial_heads.add_argument(
        "--layer-mode",
        choices=list(REGISTERED_TRACE_PATCH_LAYER_MODES),
        default="cumulative_clamp",
    )
    serial_heads.set_defaults(
        func=command_serial_patch_heads,
        max_new_tokens=16,
        cohort="one_to_one",
        row_panel="trace_patch",
    )

    serial_source = subparsers.add_parser(
        "serial-patch-source",
        help=(
            "Cross all outcome-blind terminal suffix-8 donor patches with the "
            "registered all-head prompt/trace source-mask panel."
        ),
    )
    _add_rows(serial_source)
    _add_output_resume(serial_source)
    _add_behavior(serial_source)
    serial_source.add_argument("--layer", type=int, required=True)
    serial_source.add_argument(
        "--geometry",
        choices=list(REGISTERED_TRACE_PATCH_GEOMETRIES),
        default="suffix8",
    )
    serial_source.add_argument(
        "--layer-mode",
        choices=list(REGISTERED_TRACE_PATCH_LAYER_MODES),
        default="cumulative_clamp",
    )
    serial_source.add_argument(
        "--mask-conditions",
        nargs="+",
        choices=list(REGISTERED_MASK_CONDITIONS),
        default=[
            "clean",
            "block_trace_items",
            "block_trace_items_matched_control",
            "block_prompt_records",
            "block_prompt_records_matched_control",
        ],
    )
    serial_source.add_argument(
        "--mask-application",
        choices=["answer_query_only", "answer_query_and_answer_tokens"],
        default="answer_query_and_answer_tokens",
    )
    serial_source.set_defaults(
        func=command_serial_patch_source,
        max_new_tokens=16,
        cohort="one_to_one",
        row_panel="trace_patch",
    )

    relay = subparsers.add_parser(
        "terminal-relay-mediation",
        help=(
            "Patch a terminal state before a frozen pre-answer layer, then "
            "reset the answer query or the complete post-terminal suffix."
        ),
    )
    _add_rows(relay)
    _add_output_resume(relay)
    _add_behavior(relay)
    relay.add_argument("--source-layer", type=int, required=True)
    relay.add_argument("--relay-layer", type=int, required=True)
    relay.add_argument(
        "--geometry",
        choices=list(REGISTERED_TRACE_PATCH_GEOMETRIES),
        default="suffix8",
    )
    relay.set_defaults(
        func=command_terminal_relay_mediation,
        max_new_tokens=16,
        cohort="one_to_one",
        row_panel="trace_patch",
    )

    complementary = subparsers.add_parser(
        "complementary-readout",
        help=(
            "Cross Qwen terminal state patches with residual-suffix resets "
            "and persistent all-head trace-source masks."
        ),
    )
    _add_rows(complementary)
    _add_output_resume(complementary)
    _add_behavior(complementary)
    complementary.add_argument("--source-layer", type=int, required=True)
    complementary.add_argument("--relay-layer", type=int, required=True)
    complementary.add_argument(
        "--geometry",
        choices=list(REGISTERED_TRACE_PATCH_GEOMETRIES),
        default="suffix8",
    )
    complementary.set_defaults(
        func=command_complementary_readout,
        max_new_tokens=16,
        cohort="one_to_one",
        row_panel="trace_patch",
    )

    integrated = subparsers.add_parser(
        "integrated-serial-bridge",
        help=(
            "Transfer states induced by frozen targeted-write ablations and "
            "cross them with each model's confirmed answer readout cut."
        ),
    )
    _add_rows(integrated)
    _add_output_resume(integrated)
    _add_behavior(integrated)
    integrated.add_argument("--bank-plan", type=Path, required=True)
    integrated.add_argument("--anchor-registry", type=Path, required=True)
    integrated.add_argument(
        "--write-window",
        choices=["exact_query", "query_through_trace"],
        default="exact_query",
    )
    integrated.add_argument(
        "--bridge-design",
        choices=["transfer", "restoration"],
        default="transfer",
    )
    integrated.add_argument(
        "--geometry",
        choices=["suffix8", "full_span"],
        default="suffix8",
    )
    integrated.set_defaults(
        func=command_integrated_serial_bridge,
        max_new_tokens=16,
        cohort="one_to_one",
        row_panel="trace_patch",
    )

    restoration = subparsers.add_parser(
        "restoration",
        help="Corrupt trace tokens and restore clean span/endpoint/marker states.",
    )
    _add_rows(restoration)
    _add_output_resume(restoration)
    _add_behavior(restoration)
    restoration.add_argument("--layer", type=int, required=True)
    restoration.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_RESTORATION_CONDITIONS),
        default=list(REGISTERED_RESTORATION_CONDITIONS),
    )
    restoration.set_defaults(func=command_restoration)

    analyze = subparsers.add_parser(
        "analyze",
        help="Compute request-first, seed-equal registered contrasts and gates.",
    )
    _add_configs(analyze, include_v5=False)
    analyze.add_argument("--trials", type=Path, nargs="+", required=True)
    analyze.add_argument(
        "--experiment-ids",
        nargs="+",
        choices=[
            "answer_source_mask_factorial",
            "answer_broad_head_ablation",
            "trace_intermediate_state_patching",
            "trace_terminal_state_patching",
            "stream_state_retention",
            "trace_source_restoration",
        ],
    )
    analyze.add_argument(
        "--outcome",
        choices=list(REGISTERED_HIGHER_IS_BETTER_OUTCOMES),
        default="correct_count_margin",
        help=(
            "Registered higher-is-better endpoint. Error-valued columns are "
            "excluded to preserve the positive damage/repair orientation."
        ),
    )
    analyze.add_argument(
        "--strata",
        nargs="+",
        default=[],
        choices=[
            "source_group",
            "source_scope",
            "source_occurrences",
            "source_layer",
            "layer",
            "bank_size",
            "donor_direction",
            "receiver_is_visible_marker_token",
        ],
    )
    analyze.add_argument("--random-seed", type=int, default=0)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
