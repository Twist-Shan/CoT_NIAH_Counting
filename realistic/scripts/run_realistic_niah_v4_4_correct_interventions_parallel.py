from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

try:
    import run_realistic_niah_v4_4_correct_interventions as serial
except ModuleNotFoundError:  # imported as scripts.* by tests or notebooks
    from scripts import run_realistic_niah_v4_4_correct_interventions as serial
from realistic_niah_v4.causal_generation import (
    load_generation_labels,
    summarize_generation_head_ablation_v2,
    summarize_generation_residual_patching_v2,
)
from realistic_niah_v4.causal_v2 import CausalV2Design
from realistic_niah_v4.correct_interventions import (
    ABLATION_STIMULUS_KEY_COLUMNS,
    ABLATION_TOP_NS,
    PARALLEL_ASSIGNMENT_METHOD,
    PARALLEL_WORKER_COUNT,
    PATCH_PAIR_KEY_COLUMNS,
    build_parallel_work_plan,
    existing_clean_pair_instances,
    parallel_plan_records,
    select_parallel_work_frame,
    summarize_ablation_n_diagnostics,
    summarize_ablation_population,
    summarize_average_patching_accuracy,
)
from realistic_niah_v4.correct_only_slices import (
    clean_correct_ablation_rows,
    clean_correct_patching_rows,
)
from realistic_niah_v4.spec import V4Config
from realistic_niah_v4.stimuli import load_stimuli


PARALLEL_SCHEMA_VERSION = "realistic_niah_v4_4_correct_intervention_parallel_v1"
WORKER_SCHEMA_VERSION = "realistic_niah_v4_4_correct_intervention_worker_v1"
PARALLEL_IMPLEMENTATION_FILES = (
    "configs/realistic_niah_v4_4_correct_interventions.json",
    "scripts/audit_realistic_niah_v4_4_correct_interventions.py",
    "scripts/launch_realistic_niah_v4_4_correct_interventions_4x4.sh",
    "scripts/run_realistic_niah_v4_4_correct_interventions.py",
    "scripts/run_realistic_niah_v4_4_correct_interventions_parallel.py",
    "src/realistic_niah_v4/causal_generation.py",
    "src/realistic_niah_v4/correct_interventions.py",
)


@dataclass(frozen=True)
class RunContext:
    args: argparse.Namespace
    repo_root: Path
    run_root: Path
    stimuli_path: Path
    definition_path: Path
    base_config_path: Path
    causal_config_path: Path
    prompt_selection: Path
    answer_selection: Path
    prompt_detail: Path
    answer_detail: Path
    ablation_discovery_detail: Path
    ablation_detail: Path
    rankings_path: Path
    config: V4Config
    design: CausalV2Design
    definition: dict[str, Any]
    reserve_seeds: tuple[int, ...]
    top_ns: tuple[int, ...]
    head_bank: str
    stage_design: dict[str, Any]
    design_hash: str
    stage_root: Path


def _implementation_hash(repo_root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for relative in PARALLEL_IMPLEMENTATION_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _runtime_payload(repo_root: Path, *, phase: str) -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "logical_index": int(index),
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "schema_version": PARALLEL_SCHEMA_VERSION,
        "phase": str(phase),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_devices": devices,
        "git": serial._git_state(repo_root),
        "command": sys.argv,
    }


def _build_context(args: argparse.Namespace) -> RunContext:
    if int(args.worker_count) != PARALLEL_WORKER_COUNT:
        raise ValueError(
            f"The formal 4+4 runner requires worker_count={PARALLEL_WORKER_COUNT}"
        )
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    stimuli_path = Path(args.stimuli).resolve()
    definition_path = Path(args.definition).resolve()
    base_config_path = Path(args.base_config).resolve()
    causal_config_path = Path(args.causal_config).resolve()
    prompt_selection = Path(args.prompt_selection).resolve()
    answer_selection = Path(args.answer_selection).resolve()
    prompt_detail = Path(args.prompt_confirmation_detail).resolve()
    answer_detail = Path(args.answer_confirmation_detail).resolve()
    ablation_discovery_detail = Path(args.ablation_discovery_detail).resolve()
    ablation_detail = Path(args.ablation_confirmation_detail).resolve()
    rankings_path = Path(args.head_rankings).resolve()
    required_paths = (
        stimuli_path,
        definition_path,
        base_config_path,
        causal_config_path,
        prompt_selection,
        answer_selection,
        prompt_detail,
        answer_detail,
        ablation_discovery_detail,
        ablation_detail,
        rankings_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    config = V4Config.from_json(base_config_path)
    design = CausalV2Design.from_json(causal_config_path)
    definition = serial._read_definition(definition_path)
    ablation_definition = definition["ablation"]
    if ablation_definition.get("selection_status") != "unfrozen_discovery_only":
        raise ValueError("Ablation top-n must remain unfrozen in discovery")
    top_ns = tuple(int(value) for value in ablation_definition["top_n_candidates"])
    if top_ns != ABLATION_TOP_NS:
        raise ValueError("Ablation discovery must compare top-n=1..32")
    reserve_seeds = tuple(
        range(
            int(definition["reserve_seed_start"]),
            int(definition["reserve_seed_end_inclusive"]) + 1,
        )
    )
    input_hashes = {str(path): serial._sha256(path) for path in required_paths}
    stage_design = {
        "schema_version": serial.SCHEMA_VERSION,
        "model_label": args.model,
        "source_experiment": "V4.4 causal-v2",
        "reserve_seeds": list(reserve_seeds),
        "patch_cluster_target": int(
            definition["patching"]["minimum_seed_clusters_per_model_k_direction"]
        ),
        "correct_ablation_cluster_target": int(
            ablation_definition["minimum_fresh_correct_seed_clusters_per_model"]
        ),
        "ablation_counts": [int(value) for value in ablation_definition["counts"]],
        "ablation_population_pairing": (
            "same fresh-seed prefix and count domain; clean-correct is an exact "
            "row subset of all examples"
        ),
        "ablation_head_bank": str(ablation_definition["head_bank"]),
        "ablation_top_n_candidates": list(top_ns),
        "top_n_selection_status": "unfrozen_discovery_only",
        "random_replicates": int(ablation_definition["random_replicates"]),
        "prompt_full_span_alignment": design.prompt_full_span_alignment,
        "parallel_execution": {
            "layout": "four_independent_single_gpu_workers_per_model",
            "worker_count_per_model": PARALLEL_WORKER_COUNT,
            "assignment_method": PARALLEL_ASSIGNMENT_METHOD,
            "worker_outputs": "isolated_then_strictly_merged",
            "numerical_definition_change": False,
        },
        "input_sha256": input_hashes,
        "implementation_sha256": _implementation_hash(repo_root),
    }
    design_hash = serial._json_hash(stage_design)
    stage_root = (
        run_root
        / args.model
        / "numeric"
        / "correct_interventions"
        / f"confirmation_{design_hash}"
    )
    return RunContext(
        args=args,
        repo_root=repo_root,
        run_root=run_root,
        stimuli_path=stimuli_path,
        definition_path=definition_path,
        base_config_path=base_config_path,
        causal_config_path=causal_config_path,
        prompt_selection=prompt_selection,
        answer_selection=answer_selection,
        prompt_detail=prompt_detail,
        answer_detail=answer_detail,
        ablation_discovery_detail=ablation_discovery_detail,
        ablation_detail=ablation_detail,
        rankings_path=rankings_path,
        config=config,
        design=design,
        definition=definition,
        reserve_seeds=reserve_seeds,
        top_ns=top_ns,
        head_bank=str(ablation_definition["head_bank"]),
        stage_design=stage_design,
        design_hash=design_hash,
        stage_root=stage_root,
    )


def _ensure_design(context: RunContext) -> None:
    context.stage_root.mkdir(parents=True, exist_ok=True)
    design_path = context.stage_root / "design.json"
    if design_path.is_file():
        observed = json.loads(design_path.read_text(encoding="utf-8"))
        if observed != context.stage_design:
            raise RuntimeError(f"Existing stage design differs: {design_path}")
    else:
        serial._write_json(design_path, context.stage_design)


def _discovery_baselines(
    context: RunContext,
    candidate_labels: pd.DataFrame,
    selection: dict[str, Any],
) -> pd.DataFrame:
    seeds = tuple(
        int(value)
        for value in selection["correct_only_ablation"]["shared_discovery_seed_prefix"]
    )
    counts = tuple(int(value) for value in context.stage_design["ablation_counts"])
    if not seeds:
        raise RuntimeError("Shared ablation discovery seed prefix is empty")
    result = candidate_labels[
        pd.to_numeric(candidate_labels["seed"], errors="raise").astype(int).isin(seeds)
        & pd.to_numeric(candidate_labels["gold_count"], errors="raise")
        .astype(int)
        .isin(counts)
    ].copy()
    expected = len(seeds) * len(counts)
    if (
        len(result) != expected
        or result[["stimulus_id", "seed", "gold_count"]].drop_duplicates().shape[0]
        != expected
    ):
        raise RuntimeError("Shared ablation baseline grid is incomplete or duplicated")
    return result.reset_index(drop=True)


def _required_prepare_files(stage_root: Path) -> tuple[Path, ...]:
    return (
        stage_root / "design.json",
        stage_root / "baseline_labels.scanned.csv",
        stage_root / "supplement_selection.json",
        stage_root / "selected_added_pairs.csv",
        stage_root / "eligible_added_correct_ablation_baselines.csv",
        stage_root / "parallel_work_plan.json",
    )


def _verify_prepare(context: RunContext) -> dict[str, Any]:
    marker_path = context.stage_root / "prepare.complete.json"
    if not marker_path.is_file():
        raise RuntimeError(f"Parallel prepare phase is incomplete: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("design_hash") != context.design_hash:
        raise RuntimeError("Prepare marker design hash differs")
    for path in _required_prepare_files(context.stage_root):
        expected = marker.get("file_sha256", {}).get(path.name)
        if not path.is_file() or expected != serial._sha256(path):
            raise RuntimeError(f"Prepared input is missing or changed: {path}")
    return marker


def _prepare(context: RunContext) -> dict[str, Any]:
    started = time.perf_counter()
    _ensure_design(context)
    if (
        context.stage_root / "prepare.complete.json"
    ).is_file() and not context.args.overwrite:
        return _verify_prepare(context)
    serial._write_json(
        context.stage_root / "runtime.prepare.json",
        _runtime_payload(context.repo_root, phase="prepare"),
    )
    existing_answer = pd.read_csv(context.answer_detail, compression="infer")
    existing_pairs = existing_clean_pair_instances(existing_answer)
    existing_ablation = pd.read_csv(context.ablation_detail, compression="infer")
    legacy_ablation_baselines = serial._standard_ablation_baselines(existing_ablation)
    fresh_ablation_target = legacy_ablation_baselines.iloc[0:0].copy()
    stimuli = load_stimuli(context.stimuli_path)
    model, tokenizer, _adapter = serial._load_model(
        model_label=context.args.model,
        config=context.config,
        cache_dir=context.args.cache_dir,
        device_map=context.args.device_map,
    )
    candidate, selection, added_pairs, added_ablation = serial._baseline_prefix(
        model=model,
        tokenizer=tokenizer,
        model_label=context.args.model,
        config=context.config,
        stimuli=stimuli,
        reserve_seeds=context.reserve_seeds,
        stage_root=context.stage_root,
        existing_pairs=existing_pairs,
        existing_ablation_baselines=fresh_ablation_target,
        patch_target=int(context.stage_design["patch_cluster_target"]),
        ablation_target=int(context.stage_design["correct_ablation_cluster_target"]),
        ablation_counts=tuple(context.stage_design["ablation_counts"]),
        max_new_tokens=int(context.args.generation_max_new_tokens),
        overwrite=bool(context.args.overwrite),
    )
    discovery = _discovery_baselines(context, candidate, selection)
    plan = build_parallel_work_plan(
        model_label=context.args.model,
        added_pairs=added_pairs,
        ablation_stimuli=discovery[
            ["model_label", "stimulus_id", "seed", "gold_count"]
        ],
        worker_count=PARALLEL_WORKER_COUNT,
    )
    plan["design_hash"] = context.design_hash
    serial._write_json(context.stage_root / "parallel_work_plan.json", plan)
    files = _required_prepare_files(context.stage_root)
    marker = {
        "status": "complete",
        "schema_version": PARALLEL_SCHEMA_VERSION,
        "phase": "prepare",
        "model_label": context.args.model,
        "design_hash": context.design_hash,
        "scanned_supplement_seeds": selection["scanned_supplement_seeds"],
        "selected_added_pair_instances": int(len(added_pairs)),
        "shared_ablation_stimuli": int(len(discovery)),
        "worker_count": PARALLEL_WORKER_COUNT,
        "file_sha256": {path.name: serial._sha256(path) for path in files},
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    serial._write_json(context.stage_root / "prepare.complete.json", marker)
    return marker


def _worker_root(context: RunContext, worker_index: int) -> Path:
    return (
        context.stage_root
        / "workers"
        / f"worker_{int(worker_index):03d}_of_{PARALLEL_WORKER_COUNT:03d}"
    )


def _assigned_correct_baselines(
    selected_correct: pd.DataFrame,
    assigned_discovery: pd.DataFrame,
) -> pd.DataFrame:
    identities = set(assigned_discovery["stimulus_id"].astype(str))
    return selected_correct[
        selected_correct["stimulus_id"].astype(str).isin(identities)
    ].copy()


def _worker(context: RunContext, worker_index: int) -> dict[str, Any]:
    started = time.perf_counter()
    _ensure_design(context)
    _verify_prepare(context)
    prepare_marker_path = context.stage_root / "prepare.complete.json"
    prepare_hash = serial._sha256(prepare_marker_path)
    plan_path = context.stage_root / "parallel_work_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("design_hash") != context.design_hash:
        raise RuntimeError("Parallel plan design hash differs")
    index = int(worker_index)
    if index < 0 or index >= PARALLEL_WORKER_COUNT:
        raise ValueError(f"worker_index must lie in [0, {PARALLEL_WORKER_COUNT})")
    worker_root = _worker_root(context, index)
    worker_root.mkdir(parents=True, exist_ok=True)
    assigned_pair_records = parallel_plan_records(
        plan, work_kind="patch_pairs", worker_index=index
    )
    assigned_ablation_records = parallel_plan_records(
        plan, work_kind="ablation_stimuli", worker_index=index
    )
    worker_design = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "model_label": context.args.model,
        "design_hash": context.design_hash,
        "parallel_work_plan_sha256": serial._sha256(plan_path),
        "prepare_sha256": prepare_hash,
        "worker_index": index,
        "worker_count": PARALLEL_WORKER_COUNT,
        "assignment_method": PARALLEL_ASSIGNMENT_METHOD,
        "assigned_patch_pairs": assigned_pair_records,
        "assigned_ablation_stimuli": assigned_ablation_records,
    }
    worker_design_path = worker_root / "design.json"
    if worker_design_path.is_file():
        observed = json.loads(worker_design_path.read_text(encoding="utf-8"))
        if observed != worker_design:
            raise RuntimeError(f"Worker design differs: {worker_design_path}")
    else:
        serial._write_json(worker_design_path, worker_design)
    existing_marker_path = worker_root / "complete.json"
    if existing_marker_path.is_file() and not context.args.overwrite:
        marker = json.loads(existing_marker_path.read_text(encoding="utf-8"))
        expected_outputs = {
            "prompt_detail": worker_root
            / "patching"
            / "prompt_patching"
            / "detail.supplement.csv.gz",
            "answer_detail": worker_root
            / "patching"
            / "answer_patching"
            / "detail.supplement.csv.gz",
            "ablation_detail": worker_root
            / "ablation"
            / "detail.all_examples.discovery.csv.gz",
        }
        marker_valid = (
            marker.get("status") == "complete"
            and marker.get("design_hash") == context.design_hash
            and marker.get("parallel_work_plan_sha256") == serial._sha256(plan_path)
            and marker.get("prepare_sha256") == prepare_hash
            and int(marker.get("worker_index", -1)) == index
            and int(marker.get("worker_count", 0)) == PARALLEL_WORKER_COUNT
            and all(
                path.is_file()
                and marker.get("output_sha256", {}).get(name) == serial._sha256(path)
                for name, path in expected_outputs.items()
            )
        )
        if not marker_valid:
            raise RuntimeError(
                f"Existing worker completion is inconsistent: {existing_marker_path}"
            )
        return marker
    serial._write_json(
        worker_root / "runtime.json",
        _runtime_payload(context.repo_root, phase=f"worker-{index}"),
    )
    runtime = json.loads((worker_root / "runtime.json").read_text(encoding="utf-8"))
    if len(runtime.get("cuda_devices", [])) != 1:
        raise RuntimeError("Every 4+4 worker must see exactly one CUDA device")

    candidate = pd.read_csv(context.stage_root / "baseline_labels.scanned.csv")
    selection = json.loads(
        (context.stage_root / "supplement_selection.json").read_text(encoding="utf-8")
    )
    all_pairs = pd.read_csv(context.stage_root / "selected_added_pairs.csv")
    all_correct = pd.read_csv(
        context.stage_root / "eligible_added_correct_ablation_baselines.csv"
    )
    discovery = _discovery_baselines(context, candidate, selection)
    assigned_pairs = select_parallel_work_frame(
        all_pairs,
        plan=plan,
        work_kind="patch_pairs",
        worker_index=index,
    )
    assigned_discovery = select_parallel_work_frame(
        discovery,
        plan=plan,
        work_kind="ablation_stimuli",
        worker_index=index,
    )
    assigned_correct = _assigned_correct_baselines(all_correct, assigned_discovery)
    stimuli = load_stimuli(context.stimuli_path)
    model, tokenizer, adapter = serial._load_model(
        model_label=context.args.model,
        config=context.config,
        cache_dir=context.args.cache_dir,
        device_map=context.args.device_map,
    )
    scanned_seeds = tuple(int(value) for value in selection["scanned_supplement_seeds"])
    rows = serial._select_rows(stimuli, seeds=scanned_seeds, counts=tuple(range(11)))
    encodings = serial._render(
        rows,
        tokenizer=tokenizer,
        model_label=context.args.model,
        config=context.config,
    )
    baseline_labels = load_generation_labels(
        context.stage_root / "baseline_labels.scanned.csv"
    )
    encodings_by_id = {encoding.stimulus_id: encoding for encoding in encodings}
    prompt_outputs = serial._run_patching_family(
        family="prompt_patching",
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        model_label=context.args.model,
        config=context.config,
        design=context.design,
        encodings=encodings,
        baseline_labels=baseline_labels,
        added_pairs=assigned_pairs,
        selection_path=context.prompt_selection,
        existing_detail_path=context.prompt_detail,
        output_root=worker_root,
        max_new_tokens=int(context.args.generation_max_new_tokens),
        overwrite=bool(context.args.overwrite),
        finalize=False,
    )
    answer_outputs = serial._run_patching_family(
        family="answer_patching",
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        model_label=context.args.model,
        config=context.config,
        design=context.design,
        encodings=encodings,
        baseline_labels=baseline_labels,
        added_pairs=assigned_pairs,
        selection_path=context.answer_selection,
        existing_detail_path=context.answer_detail,
        output_root=worker_root,
        max_new_tokens=int(context.args.generation_max_new_tokens),
        overwrite=bool(context.args.overwrite),
        finalize=False,
    )
    ablation_outputs = serial._run_correct_ablation(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        model_label=context.args.model,
        design=context.design,
        encodings_by_id=encodings_by_id,
        baseline_labels=baseline_labels,
        discovery_baselines=assigned_discovery,
        selected_correct_baselines=assigned_correct,
        rankings_path=context.rankings_path,
        legacy_discovery_detail_path=context.ablation_discovery_detail,
        legacy_confirmation_detail_path=context.ablation_detail,
        head_bank=context.head_bank,
        top_ns=context.top_ns,
        random_replicates=int(context.stage_design["random_replicates"]),
        output_root=worker_root,
        max_new_tokens=int(context.args.generation_max_new_tokens),
        overwrite=bool(context.args.overwrite),
        finalize=False,
    )
    output_paths = {
        "prompt_detail": Path(prompt_outputs["new_detail"]),
        "answer_detail": Path(answer_outputs["new_detail"]),
        "ablation_detail": Path(ablation_outputs["new_all_examples_discovery_detail"]),
    }
    marker = {
        "status": "complete",
        "schema_version": WORKER_SCHEMA_VERSION,
        "model_label": context.args.model,
        "design_hash": context.design_hash,
        "parallel_work_plan_sha256": serial._sha256(plan_path),
        "worker_index": index,
        "worker_count": PARALLEL_WORKER_COUNT,
        "assigned_patch_pairs": int(len(assigned_pair_records)),
        "assigned_ablation_stimuli": int(len(assigned_ablation_records)),
        "output_sha256": {
            name: serial._sha256(path) for name, path in output_paths.items()
        },
        "prepare_sha256": prepare_hash,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    serial._write_json(worker_root / "complete.json", marker)
    return marker


def _read_worker_outputs(
    context: RunContext,
) -> tuple[list[dict[str, Any]], dict[str, list[pd.DataFrame]]]:
    plan_path = context.stage_root / "parallel_work_plan.json"
    plan_hash = serial._sha256(plan_path)
    prepare_hash = serial._sha256(context.stage_root / "prepare.complete.json")
    markers: list[dict[str, Any]] = []
    frames: dict[str, list[pd.DataFrame]] = {
        "prompt_patching": [],
        "answer_patching": [],
        "ablation": [],
    }
    for index in range(PARALLEL_WORKER_COUNT):
        root = _worker_root(context, index)
        marker_path = root / "complete.json"
        if not marker_path.is_file():
            raise RuntimeError(f"Worker completion marker is missing: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("status") != "complete"
            or marker.get("design_hash") != context.design_hash
            or marker.get("parallel_work_plan_sha256") != plan_hash
            or marker.get("prepare_sha256") != prepare_hash
            or int(marker.get("worker_index", -1)) != index
            or int(marker.get("worker_count", 0)) != PARALLEL_WORKER_COUNT
        ):
            raise RuntimeError(
                f"Worker completion marker is inconsistent: {marker_path}"
            )
        paths = {
            "prompt_patching": root
            / "patching"
            / "prompt_patching"
            / "detail.supplement.csv.gz",
            "answer_patching": root
            / "patching"
            / "answer_patching"
            / "detail.supplement.csv.gz",
            "ablation": root / "ablation" / "detail.all_examples.discovery.csv.gz",
        }
        marker_keys = {
            "prompt_patching": "prompt_detail",
            "answer_patching": "answer_detail",
            "ablation": "ablation_detail",
        }
        for family, path in paths.items():
            expected = marker["output_sha256"].get(marker_keys[family])
            if not path.is_file() or serial._sha256(path) != expected:
                raise RuntimeError(f"Worker output is missing or changed: {path}")
            frame = pd.read_csv(path, compression="infer")
            frame["parallel_worker_index"] = index
            frames[family].append(frame)
        markers.append(marker)
    return markers, frames


def _validate_pair_partition(
    plan: dict[str, Any], detail: pd.DataFrame, *, family: str
) -> None:
    expected = {
        tuple(str(record[column]) for column in PATCH_PAIR_KEY_COLUMNS)
        for record in parallel_plan_records(plan, work_kind="patch_pairs")
    }
    observed_frame = detail[list(PATCH_PAIR_KEY_COLUMNS)].drop_duplicates().copy()
    observed = {
        tuple(str(value) for value in row)
        for row in observed_frame.itertuples(index=False, name=None)
    }
    if observed != expected:
        raise RuntimeError(f"Merged {family} pair identities differ from work plan")
    ownership = detail.groupby(list(PATCH_PAIR_KEY_COLUMNS))[
        "parallel_worker_index"
    ].nunique()
    if not ownership.eq(1).all():
        raise RuntimeError(f"Merged {family} has a pair written by multiple workers")
    row_keys = [
        *PATCH_PAIR_KEY_COLUMNS,
        "site",
        "patch_protocol",
        "start_layer",
        "condition",
    ]
    if detail.duplicated(row_keys).any():
        raise RuntimeError(f"Merged {family} contains duplicate intervention rows")


def _validate_ablation_partition(plan: dict[str, Any], detail: pd.DataFrame) -> None:
    expected = {
        tuple(str(record[column]) for column in ABLATION_STIMULUS_KEY_COLUMNS)
        for record in parallel_plan_records(plan, work_kind="ablation_stimuli")
    }
    observed_frame = detail[list(ABLATION_STIMULUS_KEY_COLUMNS)].drop_duplicates()
    observed = {
        tuple(str(value) for value in row)
        for row in observed_frame.itertuples(index=False, name=None)
    }
    if observed != expected:
        raise RuntimeError("Merged ablation stimulus identities differ from work plan")
    ownership = detail.groupby(list(ABLATION_STIMULUS_KEY_COLUMNS))[
        "parallel_worker_index"
    ].nunique()
    if not ownership.eq(1).all():
        raise RuntimeError("Merged ablation has a stimulus written by multiple workers")
    row_keys = ["stimulus_id", "head_bank", "top_n", "condition", "random_replicate"]
    if detail.duplicated(row_keys).any():
        raise RuntimeError("Merged ablation contains duplicate intervention rows")


def _merge_patching(
    context: RunContext,
    *,
    family: str,
    worker_frames: Sequence[pd.DataFrame],
    existing_detail_path: Path,
) -> dict[str, Any]:
    root = context.stage_root / "patching" / family
    supplement = pd.concat(worker_frames, ignore_index=True, sort=False)
    sort_columns = [
        "seed",
        "receiver_count",
        "donor_count",
        "site",
        "patch_protocol",
        "start_layer",
        "condition",
    ]
    supplement = supplement.sort_values(sort_columns, kind="stable").reset_index(
        drop=True
    )
    supplement_path = root / "detail.supplement.csv.gz"
    serial._write_csv(supplement, supplement_path, gzip=True)
    serial._write_csv(
        summarize_generation_residual_patching_v2(supplement),
        root / "summary.supplement.csv",
    )
    existing = pd.read_csv(existing_detail_path, compression="infer")
    existing_correct = clean_correct_patching_rows(existing)
    existing_correct["family"] = family
    if "evidence_split" not in existing_correct.columns:
        existing_correct["evidence_split"] = "original_held_out_confirmation"
    combined = pd.concat(
        [existing_correct, supplement], ignore_index=True, sort=False
    ).drop_duplicates(
        [
            "model_label",
            "seed",
            "receiver_count",
            "donor_count",
            "site",
            "patch_protocol",
            "start_layer",
            "condition",
        ],
        keep="first",
    )
    combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    combined_path = root / "detail.clean_correct.combined.csv.gz"
    serial._write_csv(combined, combined_path, gzip=True)
    exact = summarize_average_patching_accuracy(
        combined,
        group_columns=serial.EXACT_PATCH_GROUPS,
        bootstrap_repetitions=context.design.bootstrap_repetitions,
    )
    aggregate = summarize_average_patching_accuracy(
        combined,
        group_columns=serial.AGGREGATE_PATCH_GROUPS,
        bootstrap_repetitions=context.design.bootstrap_repetitions,
    )
    exact_path = root / "average_patching_acc.exact_groups.csv"
    aggregate_path = root / "average_patching_acc.aggregate_groups.csv"
    serial._write_csv(exact, exact_path)
    serial._write_csv(aggregate, aggregate_path)
    capture_rows = []
    for index in range(PARALLEL_WORKER_COUNT):
        worker_detail = (
            _worker_root(context, index)
            / "patching"
            / family
            / "detail.supplement.csv.gz"
        )
        capture_rows.append(
            {
                "worker_index": index,
                "rows": int(len(worker_frames[index])),
                "sha256": serial._sha256(worker_detail),
                "worker_detail": str(
                    worker_detail.relative_to(context.stage_root)
                ).replace("\\", "/"),
            }
        )
    serial._write_json(root / "parallel_capture_index.json", capture_rows)
    if family == "prompt_patching":
        alignments = []
        for index in range(PARALLEL_WORKER_COUNT):
            path = (
                _worker_root(context, index)
                / "patching"
                / family
                / "prompt_full_span_alignment.csv"
            )
            if path.is_file():
                alignments.append(pd.read_csv(path))
        if not alignments:
            raise RuntimeError("No worker prompt-alignment tables were found")
        alignment = pd.concat(
            alignments, ignore_index=True, sort=False
        ).drop_duplicates()
        if (~_boolean_series(alignment["mapping_supported"])).any():
            raise RuntimeError("Merged prompt alignment contains unsupported mappings")
        serial._write_csv(alignment, root / "prompt_full_span_alignment.csv")
    return {
        "new_detail": str(supplement_path),
        "combined_clean_correct_detail": str(combined_path),
        "average_patching_acc_exact": str(exact_path),
        "average_patching_acc_aggregate": str(aggregate_path),
        "new_rows": int(len(supplement)),
        "combined_rows": int(len(combined)),
        "exact_groups": int(len(exact)),
    }


def _merge_ablation(
    context: RunContext,
    *,
    worker_frames: Sequence[pd.DataFrame],
    selected_correct: pd.DataFrame,
) -> dict[str, Any]:
    root = context.stage_root / "ablation"
    all_examples = pd.concat(worker_frames, ignore_index=True, sort=False)
    sort_columns = [
        "seed",
        "gold_count",
        "head_bank",
        "top_n",
        "condition",
        "random_replicate",
    ]
    all_examples = all_examples.sort_values(sort_columns, kind="stable").reset_index(
        drop=True
    )
    all_path = root / "detail.all_examples.discovery.csv.gz"
    serial._write_csv(all_examples, all_path, gzip=True)
    serial._write_csv(
        summarize_generation_head_ablation_v2(all_examples),
        root / "summary.all_examples.discovery.csv",
    )
    correct = clean_correct_ablation_rows(all_examples)
    observed_ids = set(
        correct[["stimulus_id", "seed", "gold_count"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    selected_ids = set(
        selected_correct[["stimulus_id", "seed", "gold_count"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_ids != selected_ids:
        raise RuntimeError("Merged correct-only ablation subset differs from selection")
    correct_path = root / "detail.clean_correct.discovery.csv.gz"
    serial._write_csv(correct, correct_path, gzip=True)
    serial._write_csv(
        summarize_generation_head_ablation_v2(correct),
        root / "summary.clean_correct.discovery.csv",
    )
    legacy_discovery = pd.read_csv(
        context.ablation_discovery_detail, compression="infer"
    )
    legacy_discovery = legacy_discovery[
        legacy_discovery["head_bank"].astype(str).eq(context.head_bank)
    ].copy()
    legacy_discovery_path = (
        root / "detail.legacy_original_all_examples_discovery.csv.gz"
    )
    serial._write_csv(legacy_discovery, legacy_discovery_path, gzip=True)
    legacy = pd.read_csv(context.ablation_detail, compression="infer")
    legacy_path = root / "detail.legacy_fixed_n_confirmation.csv.gz"
    serial._write_csv(legacy, legacy_path, gzip=True)
    overall = summarize_ablation_population(
        all_examples,
        population="all_examples_signed",
        bootstrap_repetitions=context.design.bootstrap_repetitions,
    )
    correct_summary = summarize_ablation_population(
        correct,
        population="clean_correct_only",
        bootstrap_repetitions=context.design.bootstrap_repetitions,
    )
    dual = pd.concat([overall, correct_summary], ignore_index=True, sort=False)
    dual_path = root / "dual_population_ablation_summary.csv"
    serial._write_csv(dual, dual_path)
    overall_diagnostics = summarize_ablation_n_diagnostics(
        all_examples,
        population="all_examples_signed",
        head_bank=context.head_bank,
        bootstrap_repetitions=context.design.bootstrap_repetitions,
    )
    correct_diagnostics = summarize_ablation_n_diagnostics(
        correct,
        population="clean_correct_only",
        head_bank=context.head_bank,
        bootstrap_repetitions=context.design.bootstrap_repetitions,
    )
    diagnostics = pd.concat(
        [overall_diagnostics, correct_diagnostics], ignore_index=True, sort=False
    )
    diagnostics_path = root / "top_n_diagnostics.unfrozen.csv"
    serial._write_csv(diagnostics, diagnostics_path)
    capture_rows = []
    for index in range(PARALLEL_WORKER_COUNT):
        worker_detail = (
            _worker_root(context, index)
            / "ablation"
            / "detail.all_examples.discovery.csv.gz"
        )
        capture_rows.append(
            {
                "worker_index": index,
                "rows": int(len(worker_frames[index])),
                "sha256": serial._sha256(worker_detail),
                "worker_detail": str(
                    worker_detail.relative_to(context.stage_root)
                ).replace("\\", "/"),
            }
        )
    serial._write_json(root / "parallel_capture_index.json", capture_rows)
    return {
        "top_n_selection_status": "unfrozen_discovery_only",
        "new_all_examples_discovery_detail": str(all_path),
        "new_clean_correct_discovery_detail": str(correct_path),
        "legacy_original_all_examples_discovery_detail": str(legacy_discovery_path),
        "legacy_fixed_n_confirmation_detail": str(legacy_path),
        "dual_population_summary": str(dual_path),
        "top_n_diagnostics": str(diagnostics_path),
        "new_all_example_rows": int(len(all_examples)),
        "new_all_example_stimuli": int(all_examples["stimulus_id"].nunique()),
        "new_correct_rows": int(len(correct)),
        "new_correct_stimuli": int(len(selected_ids)),
        "candidate_top_ns": list(context.top_ns),
    }


def _merge(context: RunContext) -> dict[str, Any]:
    started = time.perf_counter()
    _ensure_design(context)
    prepare_marker = _verify_prepare(context)
    serial._write_json(
        context.stage_root / "runtime.merge.json",
        _runtime_payload(context.repo_root, phase="merge"),
    )
    plan = json.loads(
        (context.stage_root / "parallel_work_plan.json").read_text(encoding="utf-8")
    )
    markers, frames = _read_worker_outputs(context)
    prompt = pd.concat(frames["prompt_patching"], ignore_index=True, sort=False)
    answer = pd.concat(frames["answer_patching"], ignore_index=True, sort=False)
    ablation = pd.concat(frames["ablation"], ignore_index=True, sort=False)
    _validate_pair_partition(plan, prompt, family="prompt_patching")
    _validate_pair_partition(plan, answer, family="answer_patching")
    _validate_ablation_partition(plan, ablation)
    prompt_outputs = _merge_patching(
        context,
        family="prompt_patching",
        worker_frames=frames["prompt_patching"],
        existing_detail_path=context.prompt_detail,
    )
    answer_outputs = _merge_patching(
        context,
        family="answer_patching",
        worker_frames=frames["answer_patching"],
        existing_detail_path=context.answer_detail,
    )
    selected_correct = pd.read_csv(
        context.stage_root / "eligible_added_correct_ablation_baselines.csv"
    )
    ablation_outputs = _merge_ablation(
        context,
        worker_frames=frames["ablation"],
        selected_correct=selected_correct,
    )
    selection = json.loads(
        (context.stage_root / "supplement_selection.json").read_text(encoding="utf-8")
    )
    added_pairs = pd.read_csv(context.stage_root / "selected_added_pairs.csv")
    merge_elapsed = float(time.perf_counter() - started)
    worker_elapsed = [float(marker["elapsed_seconds"]) for marker in markers]
    prepare_elapsed = float(prepare_marker["elapsed_seconds"])
    critical_worker_elapsed = max(worker_elapsed)
    estimated_wall_elapsed = prepare_elapsed + critical_worker_elapsed + merge_elapsed
    completion = {
        "status": "complete",
        "schema_version": serial.SCHEMA_VERSION,
        "parallel_schema_version": PARALLEL_SCHEMA_VERSION,
        "design_hash": context.design_hash,
        "model_label": context.args.model,
        "scanned_supplement_seeds": selection["scanned_supplement_seeds"],
        "selected_added_pair_instances": int(len(added_pairs)),
        "added_correct_ablation_stimuli": int(len(selected_correct)),
        "parallel_execution": {
            "worker_count": PARALLEL_WORKER_COUNT,
            "assignment_method": PARALLEL_ASSIGNMENT_METHOD,
            "workers_complete": int(len(markers)),
            "parallel_work_plan_sha256": serial._sha256(
                context.stage_root / "parallel_work_plan.json"
            ),
            "phase_elapsed_seconds": {
                "prepare": prepare_elapsed,
                "workers": worker_elapsed,
                "critical_worker": critical_worker_elapsed,
                "merge": merge_elapsed,
            },
            "estimated_model_wall_seconds": estimated_wall_elapsed,
            "total_gpu_process_seconds": prepare_elapsed + sum(worker_elapsed),
        },
        "prompt_patching": prompt_outputs,
        "answer_patching": answer_outputs,
        "ablation": ablation_outputs,
        "elapsed_seconds": estimated_wall_elapsed,
    }
    serial._write_json(context.stage_root / "complete.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, shard, or merge the formal V4.4 4+4 GPU extension."
    )
    parser.add_argument(
        "--phase", choices=("prepare", "worker", "merge"), required=True
    )
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=PARALLEL_WORKER_COUNT)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-config", default="configs/realistic_niah_v4.json")
    parser.add_argument(
        "--causal-config", default="configs/realistic_niah_v4_causal_v2.json"
    )
    parser.add_argument(
        "--definition",
        default="configs/realistic_niah_v4_4_correct_interventions.json",
    )
    parser.add_argument("--prompt-selection", required=True)
    parser.add_argument("--answer-selection", required=True)
    parser.add_argument("--prompt-confirmation-detail", required=True)
    parser.add_argument("--answer-confirmation-detail", required=True)
    parser.add_argument("--ablation-discovery-detail", required=True)
    parser.add_argument("--ablation-confirmation-detail", required=True)
    parser.add_argument("--head-rankings", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--generation-max-new-tokens", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.phase == "worker" and args.worker_index is None:
        parser.error("--worker-index is required for --phase worker")
    if args.phase != "worker" and args.worker_index is not None:
        parser.error("--worker-index is valid only for --phase worker")
    context = _build_context(args)
    if args.phase == "prepare":
        payload = _prepare(context)
    elif args.phase == "worker":
        payload = _worker(context, int(args.worker_index))
    else:
        payload = _merge(context)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
