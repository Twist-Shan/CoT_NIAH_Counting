from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah_v4.behavior import capture_generation_labels
from realistic_niah_v4.causal_generation import (
    causal_v2_prompt_span_alignment_table,
    load_generation_labels,
    run_generation_head_ablation_v2,
    run_generation_residual_patching_v2,
    summarize_generation_head_ablation_v2,
    summarize_generation_residual_patching_v2,
)
from realistic_niah_v4.causal_v2 import (
    CausalV2Design,
    load_head_phenotype_registry,
)
from realistic_niah_v4.correct_interventions import (
    ABLATION_TOP_NS,
    existing_clean_pair_instances,
    select_sequential_supplement,
    summarize_ablation_n_diagnostics,
    summarize_ablation_population,
    summarize_average_patching_accuracy,
)
from realistic_niah_v4.correct_only_slices import (
    clean_correct_ablation_rows,
    clean_correct_patching_rows,
)
from realistic_niah_v4.modeling import load_registered_model
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli


SCHEMA_VERSION = "realistic_niah_v4_4_correct_intervention_run_v2"
EXACT_PATCH_GROUPS = (
    "model_label",
    "family",
    "site",
    "patch_protocol",
    "start_layer",
    "k",
    "target_direction",
)
AGGREGATE_PATCH_GROUPS = (
    "model_label",
    "family",
    "k",
    "target_direction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if gzip:
        frame.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short"],
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_paths": []}
    return {"commit": commit, "dirty_paths": status}


def _implementation_hash(repo_root: Path) -> str:
    relatives = (
        "configs/realistic_niah_v4_4_correct_interventions.json",
        "scripts/run_realistic_niah_v4_4_correct_interventions.py",
        "src/realistic_niah_v4/causal_generation.py",
        "src/realistic_niah_v4/correct_interventions.py",
    )
    digest = hashlib.sha256()
    for relative in relatives:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _select_rows(
    stimuli: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int],
    counts: Sequence[int],
) -> list[dict[str, Any]]:
    seed_set = {int(value) for value in seeds}
    count_set = {int(value) for value in counts}
    rows = [
        row
        for row in stimuli
        if str(row.get("design_variant")) == "v4.4"
        and int(row["seed"]) in seed_set
        and int(row["gold_count"]) in count_set
    ]
    expected = len(seed_set) * len(count_set)
    if len(rows) != expected:
        raise ValueError(
            f"Stimulus selection has {len(rows)} rows; expected {expected}"
        )
    return rows


def _render(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    model_label: str,
    config: V4Config,
) -> list[PromptEncoding]:
    spec = resolve_model_spec(model_label)
    return [
        render_v4_prompt(
            row,
            tokenizer=tokenizer,
            model_spec=spec,
            config=config,
            answer_format="numeric",
        )
        for row in rows
    ]


def _read_definition(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "realistic_niah_v4_4_correct_interventions_v2"
    ):
        raise ValueError("Unexpected correct-intervention definition schema")
    return payload


def _selection_filter(
    path: Path,
    *,
    family: str,
    model_label: str,
) -> set[tuple[str, str, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("family")) != family:
        raise ValueError(f"Selection family mismatch: {path}")
    if str(payload.get("model_label")) != model_label:
        raise ValueError(f"Selection model mismatch: {path}")
    selected = payload.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"Selection contains no frozen conditions: {path}")
    result = {
        (
            str(item["site"]),
            str(item["patch_protocol"]),
            int(item["start_layer"]),
            int(item["k"]),
        )
        for item in selected
    }
    if len(result) != len(selected):
        raise ValueError(f"Selection contains duplicate conditions: {path}")
    return result


def _standard_ablation_baselines(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "stimulus_id",
        "model_label",
        "seed",
        "gold_count",
        "baseline_outcome",
        "baseline_format_valid",
        "baseline_is_correct",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Ablation detail is missing baseline columns: {missing}")
    selected = detail.copy()
    if "condition" in selected.columns:
        ranked = selected[selected["condition"].astype(str).eq("ranked")]
        if not ranked.empty:
            selected = ranked
    selected = selected[
        [
            "stimulus_id",
            "model_label",
            "seed",
            "gold_count",
            "baseline_outcome",
            "baseline_format_valid",
            "baseline_is_correct",
        ]
    ].drop_duplicates()
    return selected.rename(
        columns={
            "baseline_outcome": "outcome_group",
            "baseline_format_valid": "format_valid",
            "baseline_is_correct": "is_correct",
        }
    ).reset_index(drop=True)


def _load_model(
    *,
    model_label: str,
    config: V4Config,
    cache_dir: str | None,
    device_map: str,
) -> tuple[Any, Any, Any]:
    return load_registered_model(
        resolve_model_spec(model_label),
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=config.model_torch_dtype,
        attention_backend=config.attention_prefix_backend,
    )


def _baseline_prefix(
    *,
    model: Any,
    tokenizer: Any,
    model_label: str,
    config: V4Config,
    stimuli: Sequence[dict[str, Any]],
    reserve_seeds: Sequence[int],
    stage_root: Path,
    existing_pairs: pd.DataFrame,
    existing_ablation_baselines: pd.DataFrame,
    patch_target: int,
    ablation_target: int,
    ablation_counts: Sequence[int],
    max_new_tokens: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    label_frames: list[pd.DataFrame] = []
    selection: dict[str, Any] | None = None
    added_pairs = pd.DataFrame()
    added_ablation = pd.DataFrame()
    for seed in reserve_seeds:
        seed_root = stage_root / "clean_baseline" / f"seed{seed}"
        labels_path = seed_root / "generation_labels.csv"
        if labels_path.is_file() and not overwrite:
            labels = pd.read_csv(labels_path)
        else:
            rows = _select_rows(stimuli, seeds=(seed,), counts=tuple(range(11)))
            encodings = _render(
                rows,
                tokenizer=tokenizer,
                model_label=model_label,
                config=config,
            )
            outputs = capture_generation_labels(
                model,
                tokenizer,
                encodings,
                output_dir=seed_root,
                valid_counts=tuple(range(11)),
                max_new_tokens=max_new_tokens,
                overwrite=overwrite,
            )
            labels = pd.read_csv(outputs["labels"])
        label_frames.append(labels)
        candidate = pd.concat(label_frames, ignore_index=True, sort=False)
        scanned = tuple(int(value) for value in reserve_seeds[: len(label_frames)])
        selection, added_pairs, added_ablation = select_sequential_supplement(
            existing_pairs=existing_pairs,
            existing_ablation_baselines=existing_ablation_baselines,
            candidate_baselines=candidate,
            reserve_seeds=scanned,
            patch_cluster_target=patch_target,
            ablation_cluster_target=ablation_target,
            ablation_counts=ablation_counts,
        )
        _write_json(stage_root / "supplement_selection.provisional.json", selection)
        if selection["selection_status"] == "complete":
            break
    if selection is None:
        raise RuntimeError("Fresh-seed reserve is empty")
    scanned_count = len(selection["scanned_supplement_seeds"])
    selection["reserve_seeds"] = [int(value) for value in reserve_seeds]
    selection["unused_reserve_seeds"] = [
        int(value) for value in reserve_seeds[scanned_count:]
    ]
    candidate = pd.concat(label_frames, ignore_index=True, sort=False)
    _write_csv(candidate, stage_root / "baseline_labels.scanned.csv")
    _write_json(stage_root / "supplement_selection.json", selection)
    _write_csv(added_pairs, stage_root / "selected_added_pairs.csv")
    _write_csv(
        added_ablation,
        stage_root / "eligible_added_correct_ablation_baselines.csv",
    )
    if selection["selection_status"] != "complete":
        raise RuntimeError(
            "Fresh-seed reserve did not satisfy the predeclared correctness quotas"
        )
    return candidate, selection, added_pairs, added_ablation


def _run_patching_family(
    *,
    family: str,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    model_label: str,
    config: V4Config,
    design: CausalV2Design,
    encodings: Sequence[PromptEncoding],
    baseline_labels: dict[str, dict[str, Any]],
    added_pairs: pd.DataFrame,
    selection_path: Path,
    existing_detail_path: Path,
    output_root: Path,
    max_new_tokens: int,
    overwrite: bool,
    finalize: bool = True,
) -> dict[str, Any]:
    condition_filter = _selection_filter(
        selection_path, family=family, model_label=model_label
    )
    family_filter = {
        key for key in condition_filter if int(key[3]) in set(added_pairs["k"])
    }
    sites = tuple(sorted({key[0] for key in family_filter}))
    protocols = tuple(sorted({key[1] for key in family_filter}))
    layers = tuple(sorted({int(key[2]) for key in family_filter}))
    if not sites or not protocols or not layers:
        raise ValueError(f"No selected {family} conditions match added pairs")
    family_root = output_root / "patching" / family
    alignment_path: Path | None = None
    if family == "prompt_patching":
        alignment_pairs = tuple(
            sorted(
                {
                    (int(row.receiver_count), int(row.donor_count))
                    for row in added_pairs.itertuples(index=False)
                }
            )
        )
        alignment = causal_v2_prompt_span_alignment_table(
            encodings,
            count_pairs=alignment_pairs,
            evaluation_seeds=tuple(
                sorted(pd.to_numeric(added_pairs["seed"]).astype(int).unique())
            ),
            alignment_policy=design.prompt_full_span_alignment,
        )
        if (~alignment["mapping_supported"].astype(bool)).any():
            raise RuntimeError("Correct-only prompt patching has unsupported mappings")
        alignment_path = family_root / "prompt_full_span_alignment.csv"
        _write_csv(alignment, alignment_path)
    capture_root = family_root / "capture"
    frames: list[pd.DataFrame] = []
    index_rows: list[dict[str, Any]] = []
    shared_completion_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in added_pairs.sort_values(
        ["seed", "k", "receiver_count", "donor_count"]
    ).itertuples(index=False):
        shard = (
            capture_root
            / "shards"
            / f"seed{int(row.seed)}"
            / f"N{int(row.receiver_count)}_to_N{int(row.donor_count)}.csv.gz"
        )
        if shard.is_file() and not overwrite:
            frame = pd.read_csv(shard, compression="gzip")
        else:
            frame = run_generation_residual_patching_v2(
                model,
                tokenizer,
                adapter,
                encodings,
                baseline_labels=baseline_labels,
                count_pairs=((int(row.receiver_count), int(row.donor_count)),),
                start_layers=layers,
                sites=sites,
                protocols=protocols,
                control_conditions=("donor_transport", "self_patch"),
                condition_filter=family_filter,
                evaluation_seeds=(int(row.seed),),
                shared_completion_cache=shared_completion_cache,
                require_clean_correct_pair=True,
                valid_counts=design.valid_counts,
                max_new_tokens=max_new_tokens,
                full_span_alignment_policy=design.prompt_full_span_alignment,
            )
            frame["family"] = family
            frame["evidence_split"] = "baseline_gated_correct_supplement"
            _write_csv(frame, shard, gzip=True)
        frames.append(frame)
        index_rows.append(
            {
                "seed": int(row.seed),
                "receiver_count": int(row.receiver_count),
                "donor_count": int(row.donor_count),
                "k": int(row.k),
                "rows": int(len(frame)),
                "sha256": _sha256(shard),
                "shard_path": str(shard.relative_to(capture_root)).replace("\\", "/"),
            }
        )
    new_detail = pd.concat(frames, ignore_index=True, sort=False)
    _write_json(capture_root / "capture_index.json", index_rows)
    _write_csv(new_detail, family_root / "detail.supplement.csv.gz", gzip=True)
    _write_csv(
        summarize_generation_residual_patching_v2(new_detail),
        family_root / "summary.supplement.csv",
    )

    if not finalize:
        return {
            "new_detail": str(family_root / "detail.supplement.csv.gz"),
            "new_rows": int(len(new_detail)),
            "alignment": str(alignment_path) if alignment_path else None,
            "finalization_status": "deferred_to_strict_parallel_merge",
        }

    existing = pd.read_csv(existing_detail_path, compression="infer")
    existing_correct = clean_correct_patching_rows(existing)
    existing_correct["family"] = family
    if "evidence_split" not in existing_correct.columns:
        existing_correct["evidence_split"] = "original_held_out_confirmation"
    combined = pd.concat(
        [existing_correct, new_detail], ignore_index=True, sort=False
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
    combined_path = family_root / "detail.clean_correct.combined.csv.gz"
    _write_csv(combined, combined_path, gzip=True)
    exact = summarize_average_patching_accuracy(
        combined,
        group_columns=EXACT_PATCH_GROUPS,
        bootstrap_repetitions=design.bootstrap_repetitions,
    )
    aggregate = summarize_average_patching_accuracy(
        combined,
        group_columns=AGGREGATE_PATCH_GROUPS,
        bootstrap_repetitions=design.bootstrap_repetitions,
    )
    exact_path = family_root / "average_patching_acc.exact_groups.csv"
    aggregate_path = family_root / "average_patching_acc.aggregate_groups.csv"
    _write_csv(exact, exact_path)
    _write_csv(aggregate, aggregate_path)
    return {
        "new_detail": str(family_root / "detail.supplement.csv.gz"),
        "combined_clean_correct_detail": str(combined_path),
        "average_patching_acc_exact": str(exact_path),
        "average_patching_acc_aggregate": str(aggregate_path),
        "new_rows": int(len(new_detail)),
        "combined_rows": int(len(combined)),
        "exact_groups": int(len(exact)),
        "alignment": str(alignment_path) if alignment_path else None,
    }


def _run_correct_ablation(
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    model_label: str,
    design: CausalV2Design,
    encodings_by_id: dict[str, PromptEncoding],
    baseline_labels: dict[str, dict[str, Any]],
    discovery_baselines: pd.DataFrame,
    selected_correct_baselines: pd.DataFrame,
    rankings_path: Path,
    legacy_discovery_detail_path: Path,
    legacy_confirmation_detail_path: Path,
    head_bank: str,
    top_ns: Sequence[int],
    random_replicates: int,
    output_root: Path,
    max_new_tokens: int,
    overwrite: bool,
    finalize: bool = True,
) -> dict[str, Any]:
    rankings = load_head_phenotype_registry(rankings_path)
    if head_bank not in rankings:
        raise KeyError(f"Frozen ranking lacks head bank {head_bank}")
    selected_rankings = {head_bank: rankings[head_bank]}
    ablation_root = output_root / "ablation"
    capture_root = ablation_root / "capture"
    frames: list[pd.DataFrame] = []
    index_rows: list[dict[str, Any]] = []
    identities = discovery_baselines[
        ["stimulus_id", "seed", "gold_count"]
    ].drop_duplicates()
    for row in identities.sort_values(["seed", "gold_count"]).itertuples(index=False):
        stimulus_id = str(row.stimulus_id)
        encoding = encodings_by_id.get(stimulus_id)
        if encoding is None:
            raise KeyError(f"Missing shared-population ablation encoding {stimulus_id}")
        shard = (
            capture_root
            / "shards"
            / f"seed{int(row.seed)}"
            / f"N{int(row.gold_count)}.csv.gz"
        )
        if shard.is_file() and not overwrite:
            frame = pd.read_csv(shard, compression="gzip")
        else:
            frame = run_generation_head_ablation_v2(
                model,
                tokenizer,
                adapter,
                [encoding],
                baseline_labels=baseline_labels,
                rankings=selected_rankings,
                top_ns=tuple(int(value) for value in top_ns),
                random_replicates=int(random_replicates),
                require_full_sweep=False,
                require_correct_baseline=False,
                valid_counts=design.valid_counts,
                max_new_tokens=max_new_tokens,
            )
            frame["evidence_split"] = "fresh_shared_seed_prefix_discovery"
            _write_csv(frame, shard, gzip=True)
        frames.append(frame)
        index_rows.append(
            {
                "stimulus_id": stimulus_id,
                "seed": int(row.seed),
                "gold_count": int(row.gold_count),
                "rows": int(len(frame)),
                "sha256": _sha256(shard),
                "shard_path": str(shard.relative_to(capture_root)).replace("\\", "/"),
            }
        )
    all_examples = pd.concat(frames, ignore_index=True, sort=False)
    _write_json(capture_root / "capture_index.json", index_rows)
    all_path = ablation_root / "detail.all_examples.discovery.csv.gz"
    _write_csv(all_examples, all_path, gzip=True)
    _write_csv(
        summarize_generation_head_ablation_v2(all_examples),
        ablation_root / "summary.all_examples.discovery.csv",
    )
    correct_detail = clean_correct_ablation_rows(all_examples)
    observed_correct_ids = set(
        correct_detail[["stimulus_id", "seed", "gold_count"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    selected_correct_ids = set(
        selected_correct_baselines[["stimulus_id", "seed", "gold_count"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_correct_ids != selected_correct_ids:
        raise RuntimeError(
            "Correct-only detail is not the exact baseline-correct subset of the "
            "registered shared discovery prefix"
        )
    correct_path = ablation_root / "detail.clean_correct.discovery.csv.gz"
    _write_csv(correct_detail, correct_path, gzip=True)
    _write_csv(
        summarize_generation_head_ablation_v2(correct_detail),
        ablation_root / "summary.clean_correct.discovery.csv",
    )

    if not finalize:
        return {
            "top_n_selection_status": "unfrozen_discovery_only",
            "new_all_examples_discovery_detail": str(all_path),
            "new_clean_correct_discovery_detail": str(correct_path),
            "new_all_example_rows": int(len(all_examples)),
            "new_all_example_stimuli": int(len(identities)),
            "new_correct_rows": int(len(correct_detail)),
            "new_correct_stimuli": int(len(selected_correct_ids)),
            "candidate_top_ns": [int(value) for value in top_ns],
            "finalization_status": "deferred_to_strict_parallel_merge",
        }

    legacy_discovery = pd.read_csv(
        legacy_discovery_detail_path, compression="infer"
    )
    legacy_discovery = legacy_discovery[
        legacy_discovery["head_bank"].astype(str).eq(str(head_bank))
    ].copy()
    legacy_discovery_path = (
        ablation_root / "detail.legacy_original_all_examples_discovery.csv.gz"
    )
    _write_csv(legacy_discovery, legacy_discovery_path, gzip=True)
    legacy = pd.read_csv(legacy_confirmation_detail_path, compression="infer")
    legacy_path = ablation_root / "detail.legacy_fixed_n_confirmation.csv.gz"
    _write_csv(legacy, legacy_path, gzip=True)
    all_summary = summarize_ablation_population(
        all_examples,
        population="all_examples_signed",
        bootstrap_repetitions=design.bootstrap_repetitions,
    )
    correct_summary = summarize_ablation_population(
        correct_detail,
        population="clean_correct_only",
        bootstrap_repetitions=design.bootstrap_repetitions,
    )
    dual = pd.concat([all_summary, correct_summary], ignore_index=True, sort=False)
    dual_path = ablation_root / "dual_population_ablation_summary.csv"
    _write_csv(dual, dual_path)
    all_diagnostics = summarize_ablation_n_diagnostics(
        all_examples,
        population="all_examples_signed",
        head_bank=head_bank,
        bootstrap_repetitions=design.bootstrap_repetitions,
    )
    correct_diagnostics = summarize_ablation_n_diagnostics(
        correct_detail,
        population="clean_correct_only",
        head_bank=head_bank,
        bootstrap_repetitions=design.bootstrap_repetitions,
    )
    diagnostics = pd.concat(
        [all_diagnostics, correct_diagnostics], ignore_index=True, sort=False
    )
    diagnostics_path = ablation_root / "top_n_diagnostics.unfrozen.csv"
    _write_csv(diagnostics, diagnostics_path)
    return {
        "top_n_selection_status": "unfrozen_discovery_only",
        "new_all_examples_discovery_detail": str(all_path),
        "new_clean_correct_discovery_detail": str(correct_path),
        "legacy_original_all_examples_discovery_detail": str(
            legacy_discovery_path
        ),
        "legacy_fixed_n_confirmation_detail": str(legacy_path),
        "dual_population_summary": str(dual_path),
        "top_n_diagnostics": str(diagnostics_path),
        "new_all_example_rows": int(len(all_examples)),
        "new_all_example_stimuli": int(len(identities)),
        "new_correct_rows": int(len(correct_detail)),
        "new_correct_stimuli": int(len(selected_correct_ids)),
        "candidate_top_ns": [int(value) for value in top_ns],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline-gated V4.4 patching and dual-population head ablation."
        )
    )
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

    started = time.perf_counter()
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
    definition = _read_definition(definition_path)
    ablation_definition = definition["ablation"]
    if ablation_definition.get("selection_status") != "unfrozen_discovery_only":
        raise ValueError("Ablation top-n must remain unfrozen in this discovery run")
    top_ns = tuple(int(value) for value in ablation_definition["top_n_candidates"])
    if top_ns != ABLATION_TOP_NS:
        raise ValueError("Ablation discovery must compare every top-n from 1 to 32")
    head_bank = str(ablation_definition["head_bank"])
    reserve_seeds = tuple(
        range(
            int(definition["reserve_seed_start"]),
            int(definition["reserve_seed_end_inclusive"]) + 1,
        )
    )
    input_hashes = {str(path): _sha256(path) for path in required_paths}
    stage_design = {
        "schema_version": SCHEMA_VERSION,
        "model_label": args.model,
        "source_experiment": "V4.4 causal-v2",
        "reserve_seeds": list(reserve_seeds),
        "patch_cluster_target": int(
            definition["patching"]["minimum_seed_clusters_per_model_k_direction"]
        ),
        "correct_ablation_cluster_target": int(
            ablation_definition["minimum_fresh_correct_seed_clusters_per_model"]
        ),
        "ablation_counts": [
            int(value) for value in definition["ablation"]["counts"]
        ],
        "ablation_population_pairing": (
            "same fresh-seed prefix and count domain; clean-correct is an exact "
            "row subset of all examples"
        ),
        "ablation_head_bank": head_bank,
        "ablation_top_n_candidates": list(top_ns),
        "top_n_selection_status": "unfrozen_discovery_only",
        "random_replicates": int(ablation_definition["random_replicates"]),
        "prompt_full_span_alignment": design.prompt_full_span_alignment,
        "input_sha256": input_hashes,
        "implementation_sha256": _implementation_hash(repo_root),
    }
    design_hash = _json_hash(stage_design)
    stage_root = (
        run_root
        / args.model
        / "numeric"
        / "correct_interventions"
        / f"confirmation_{design_hash}"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    design_path = stage_root / "design.json"
    if design_path.is_file():
        observed = json.loads(design_path.read_text(encoding="utf-8"))
        if observed != stage_design:
            raise RuntimeError(f"Existing stage design differs: {design_path}")
    else:
        _write_json(design_path, stage_design)
    _write_json(
        stage_root / "runtime.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "git": _git_state(repo_root),
            "command": sys.argv,
        },
    )

    existing_answer = pd.read_csv(answer_detail, compression="infer")
    existing_pairs = existing_clean_pair_instances(existing_answer)
    existing_ablation = pd.read_csv(ablation_detail, compression="infer")
    legacy_ablation_baselines = _standard_ablation_baselines(existing_ablation)
    fresh_ablation_baseline_target = legacy_ablation_baselines.iloc[0:0].copy()
    stimuli = load_stimuli(stimuli_path)
    model, tokenizer, adapter = _load_model(
        model_label=args.model,
        config=config,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
    )
    candidate_labels, selection, added_pairs, added_ablation = _baseline_prefix(
        model=model,
        tokenizer=tokenizer,
        model_label=args.model,
        config=config,
        stimuli=stimuli,
        reserve_seeds=reserve_seeds,
        stage_root=stage_root,
        existing_pairs=existing_pairs,
        existing_ablation_baselines=fresh_ablation_baseline_target,
        patch_target=int(stage_design["patch_cluster_target"]),
        ablation_target=int(stage_design["correct_ablation_cluster_target"]),
        ablation_counts=tuple(stage_design["ablation_counts"]),
        max_new_tokens=int(args.generation_max_new_tokens),
        overwrite=args.overwrite,
    )
    scanned_seeds = tuple(int(value) for value in selection["scanned_supplement_seeds"])
    rows = _select_rows(stimuli, seeds=scanned_seeds, counts=tuple(range(11)))
    encodings = _render(
        rows,
        tokenizer=tokenizer,
        model_label=args.model,
        config=config,
    )
    labels_path = stage_root / "baseline_labels.scanned.csv"
    baseline_labels = load_generation_labels(labels_path)
    encodings_by_id = {encoding.stimulus_id: encoding for encoding in encodings}
    shared_ablation_seeds = tuple(
        int(value)
        for value in selection["correct_only_ablation"][
            "shared_discovery_seed_prefix"
        ]
    )
    ablation_counts = tuple(int(value) for value in stage_design["ablation_counts"])
    if not shared_ablation_seeds:
        raise RuntimeError("Shared ablation discovery seed prefix is empty")
    discovery_ablation_baselines = candidate_labels[
        pd.to_numeric(candidate_labels["seed"], errors="raise")
        .astype(int)
        .isin(shared_ablation_seeds)
        & pd.to_numeric(candidate_labels["gold_count"], errors="raise")
        .astype(int)
        .isin(ablation_counts)
    ].copy()
    expected_ablation_stimuli = len(shared_ablation_seeds) * len(ablation_counts)
    if (
        len(discovery_ablation_baselines) != expected_ablation_stimuli
        or discovery_ablation_baselines[
            ["stimulus_id", "seed", "gold_count"]
        ].drop_duplicates().shape[0]
        != expected_ablation_stimuli
    ):
        raise RuntimeError(
            "Shared all-example ablation baseline grid is incomplete or duplicated"
        )

    prompt_outputs = _run_patching_family(
        family="prompt_patching",
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        model_label=args.model,
        config=config,
        design=design,
        encodings=encodings,
        baseline_labels=baseline_labels,
        added_pairs=added_pairs,
        selection_path=prompt_selection,
        existing_detail_path=prompt_detail,
        output_root=stage_root,
        max_new_tokens=int(args.generation_max_new_tokens),
        overwrite=args.overwrite,
    )
    answer_outputs = _run_patching_family(
        family="answer_patching",
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        model_label=args.model,
        config=config,
        design=design,
        encodings=encodings,
        baseline_labels=baseline_labels,
        added_pairs=added_pairs,
        selection_path=answer_selection,
        existing_detail_path=answer_detail,
        output_root=stage_root,
        max_new_tokens=int(args.generation_max_new_tokens),
        overwrite=args.overwrite,
    )
    ablation_outputs = _run_correct_ablation(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        model_label=args.model,
        design=design,
        encodings_by_id=encodings_by_id,
        baseline_labels=baseline_labels,
        discovery_baselines=discovery_ablation_baselines,
        selected_correct_baselines=added_ablation,
        rankings_path=rankings_path,
        legacy_discovery_detail_path=ablation_discovery_detail,
        legacy_confirmation_detail_path=ablation_detail,
        head_bank=head_bank,
        top_ns=top_ns,
        random_replicates=int(stage_design["random_replicates"]),
        output_root=stage_root,
        max_new_tokens=int(args.generation_max_new_tokens),
        overwrite=args.overwrite,
    )
    completion = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "design_hash": design_hash,
        "model_label": args.model,
        "scanned_supplement_seeds": list(scanned_seeds),
        "selected_added_pair_instances": int(len(added_pairs)),
        "added_correct_ablation_stimuli": int(len(added_ablation)),
        "prompt_patching": prompt_outputs,
        "answer_patching": answer_outputs,
        "ablation": ablation_outputs,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(stage_root / "complete.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
