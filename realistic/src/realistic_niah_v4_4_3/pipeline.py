from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah_v4.modeling import DecoderAdapter, load_registered_model
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import MODEL_SPECS, V4Config
from realistic_niah_v4.stimuli import load_stimuli

from .geometry import (
    fit_and_score_ov_heads,
    load_direction_source_states,
    resolve_value_source_layer,
    select_candidate_and_control_heads,
)
from .interventions import (
    CausalOutput,
    QueryBundle,
    capture_query_bundle,
    directed_intervention_logits,
    intervention_metrics,
    staged_patch_logits,
)
from .io import (
    atomic_csv_gzip,
    atomic_json,
    atomic_jsonl,
    atomic_text,
    atomic_torch_save,
    initialize_isolated_run,
    source_input_manifest,
    stage_root,
    write_stage_status,
)
from .spec import V443Config


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit(repo_root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def initialize_campaign(
    *,
    source_run_root: str | Path,
    output_namespace_root: str | Path,
    run_root: str | Path,
    config: V443Config,
    repo_root: str | Path,
    resume: bool,
) -> Path:
    root = initialize_isolated_run(
        source_run_root=source_run_root,
        output_namespace_root=output_namespace_root,
        run_root=run_root,
        config=config,
        resume=resume,
        repo_commit=_git_commit(repo_root),
    )
    manifest_path = root / "input_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Input manifest already exists: {manifest_path}")
    else:
        atomic_json(manifest_path, source_input_manifest(source_run_root))
    atomic_json(
        root / "campaign.status.json",
        {
            "schema_version": "realistic_niah_v4_4_3_campaign_status_v1",
            "state": "INITIALIZED",
            "updated_unix": time.time(),
            "models": list(config.model_labels),
        },
    )
    return root


def _load_model(
    model_label: str,
    *,
    config: V443Config,
    cache_dir: str | Path,
    device_map: str,
) -> tuple[Any, Any, DecoderAdapter]:
    if model_label not in config.model_labels:
        raise ValueError(f"Model is outside the V4.4.3 registry: {model_label}")
    return load_registered_model(
        MODEL_SPECS[model_label],
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=config.model_torch_dtype,
        attention_backend=config.attention_prefix_backend,
    )


def _load_v4_stimulus_map(
    source_run_root: str | Path,
    *,
    seeds: Sequence[int],
    counts: Sequence[int],
) -> dict[tuple[int, int], dict[str, Any]]:
    wanted_seeds = {int(seed) for seed in seeds}
    wanted_counts = {int(count) for count in counts}
    rows = load_stimuli(Path(source_run_root) / "dataset" / "stimuli.jsonl")
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if str(row["design_variant"]) != "v4.4":
            continue
        seed = int(row["seed"])
        count = int(row["gold_count"])
        if seed not in wanted_seeds or count not in wanted_counts:
            continue
        key = (seed, count)
        if key in selected:
            raise RuntimeError(f"Duplicate V4.4 stimulus for seed/count={key}")
        selected[key] = row
    expected = {(seed, count) for seed in wanted_seeds for count in wanted_counts}
    missing = sorted(expected - set(selected))
    if missing:
        raise RuntimeError(f"V4.4 source grid is incomplete: {missing[:10]}")
    return selected


def _render_encodings(
    stimulus_map: Mapping[tuple[int, int], dict[str, Any]],
    *,
    tokenizer: Any,
    model_label: str,
    v4_config: V4Config,
    seed: int,
    counts: Sequence[int],
) -> dict[int, PromptEncoding]:
    result = {}
    for count in counts:
        result[int(count)] = render_v4_prompt(
            stimulus_map[(int(seed), int(count))],
            tokenizer=tokenizer,
            model_spec=MODEL_SPECS[model_label],
            config=v4_config,
            answer_format="numeric",
        )
    return result


def _direction_path(run_root: str | Path, model_label: str) -> Path:
    return stage_root(run_root, model_label, "mapping") / "directions.pt"


def _load_directions(run_root: str | Path, model_label: str) -> dict[int, dict[str, Any]]:
    path = _direction_path(run_root, model_label)
    if not path.is_file():
        raise FileNotFoundError(f"Missing fitted directions: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "realistic_niah_v4_4_3_directions_v1":
        raise RuntimeError("Unexpected V4.4.3 direction artifact")
    if payload.get("model_label") != model_label:
        raise RuntimeError("Direction artifact model mismatch")
    return {int(layer): values for layer, values in payload["directions"].items()}


def _selection_path(run_root: str | Path, model_label: str) -> Path:
    return stage_root(run_root, model_label, "mapping") / "head_selection.json"


def _load_selection(run_root: str | Path, model_label: str) -> dict[str, Any]:
    path = _selection_path(run_root, model_label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_label") != model_label:
        raise RuntimeError("Head-selection model mismatch")
    return payload


def run_mapping_stage(
    model: Any,
    adapter: DecoderAdapter,
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443Config,
    resume: bool,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    root = stage_root(run_root, model_label, "mapping")
    complete = root / "complete.json"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Mapping stage already complete: {complete}")
        return _load_directions(run_root, model_label), _load_selection(
            run_root, model_label
        )
    write_stage_status(
        run_root,
        model_label=model_label,
        stage="mapping",
        state="RUNNING",
    )
    value_source_layers = {
        int(layer): resolve_value_source_layer(adapter, int(layer))
        for layer in config.target_layers(model_label)
    }
    prompt, answer, input_records = load_direction_source_states(
        source_run_root,
        model_label,
        config,
        prompt_source_layers=value_source_layers,
    )
    scores, directions = fit_and_score_ov_heads(
        model,
        adapter,
        prompt_states=prompt,
        answer_states=answer,
        model_label=model_label,
        config=config,
    )
    selection = select_candidate_and_control_heads(
        scores, model_label=model_label, config=config
    )
    atomic_csv_gzip(scores, root / "head_mapping_scores.csv.gz")
    atomic_json(selection_path := root / "head_selection.json", selection)
    atomic_jsonl(root / "input_shards.jsonl", input_records)
    atomic_torch_save(
        {
            "schema_version": "realistic_niah_v4_4_3_directions_v1",
            "model_label": model_label,
            "fit_counts": list(config.fit_counts),
            "heldout_counts": list(config.heldout_counts),
            "directions": directions,
        },
        root / "directions.pt",
    )
    payload = {
        "schema_version": "realistic_niah_v4_4_3_mapping_complete_v1",
        "model_label": model_label,
        "head_score_rows": int(len(scores)),
        "candidate_head_count": len(selection["candidate_heads"]),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=model_label,
        stage="mapping",
        state="COMPLETE",
        detail=payload,
    )
    return directions, selection


def _head_entries(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    controls = selection["matched_control_heads"]
    for candidate in selection["candidate_heads"]:
        layer = int(candidate["layer"])
        head = int(candidate["head"])
        parent = f"L{layer}H{head}"
        entries.append(
            {
                "role": "candidate",
                "parent_candidate": parent,
                "layer": layer,
                "head": head,
                "sentinel": bool(candidate.get("sentinel", False)),
            }
        )
        control = controls[parent]
        entries.append(
            {
                "role": "matched_control",
                "parent_candidate": parent,
                "layer": int(control["layer"]),
                "head": int(control["head"]),
                "sentinel": False,
            }
        )
    return entries


def _zo_max_candidate_delta(z_output: CausalOutput, o_output: CausalOutput) -> float:
    counts = sorted(z_output.candidate_log_scores)
    if counts != sorted(o_output.candidate_log_scores) or counts != list(range(1, 11)):
        raise RuntimeError("Z/O candidate sequence registries disagree")
    return max(
        abs(
            float(z_output.candidate_log_scores[count])
            - float(o_output.candidate_log_scores[count])
        )
        for count in counts
    )


def _stage_design_hash(
    stage: str, config: V443Config, selection: Mapping[str, Any]
) -> str:
    return _stable_hash(
        {
            "stage": stage,
            "config": config.to_dict(),
            "selection": selection,
        }
    )


def _validate_shard(
    path: Path,
    *,
    expected_interventions: set[str],
    design_hash: str,
) -> bool:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return (
        not frame.empty
        and set(frame["intervention"].astype(str)) == expected_interventions
        and frame["design_hash"].astype(str).eq(design_hash).all()
    )


def run_smoke_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443Config,
    v4_config: V4Config,
    directions: Mapping[int, Mapping[str, Any]],
    selection: Mapping[str, Any],
    resume: bool,
) -> None:
    root = stage_root(run_root, model_label, "smoke")
    complete = root / "complete.json"
    if complete.exists():
        if resume:
            return
        raise FileExistsError(f"Smoke stage already complete: {complete}")
    write_stage_status(
        run_root, model_label=model_label, stage="smoke", state="RUNNING"
    )
    entry = _head_entries(selection)[0]
    layer, head = int(entry["layer"]), int(entry["head"])
    receiver_count, donor_count = config.patch_pairs[0]
    seed = int(config.screen_seeds[0])
    counts = sorted({receiver_count, donor_count, config.injection_counts[0]})
    stimuli = _load_v4_stimulus_map(
        source_run_root, seeds=[seed], counts=counts
    )
    encodings = _render_encodings(
        stimuli,
        tokenizer=tokenizer,
        model_label=model_label,
        v4_config=v4_config,
        seed=seed,
        counts=counts,
    )
    receiver = capture_query_bundle(
        model,
        adapter,
        encodings[receiver_count],
        layers=[layer],
        capture_attention=True,
        capture_values=True,
        cache_logit_tolerance=config.attention_cache_logit_tolerance,
    )
    donor = capture_query_bundle(
        model,
        adapter,
        encodings[donor_count],
        layers=[layer],
        capture_attention=True,
        capture_values=True,
        cache_logit_tolerance=config.attention_cache_logit_tolerance,
    )
    patch = staged_patch_logits(
        model,
        adapter,
        encodings[receiver_count],
        encodings[donor_count],
        receiver=receiver,
        donor=donor,
        layer=layer,
        head=head,
        scramble_fraction=config.position_scramble_fraction,
        orthogonal_label=f"{model_label}:smoke:L{layer}H{head}:norm-control",
    )
    zo_delta = _zo_max_candidate_delta(
        patch["z_donor"][0], patch["o_donor"][0]
    )
    if zo_delta > config.strict_zo_equivalence_tolerance:
        raise RuntimeError(
            f"Z/O equivalence smoke failed: {zo_delta} > "
            f"{config.strict_zo_equivalence_tolerance}"
        )
    injection_encoding = encodings[config.injection_counts[0]]
    injection_bundle = capture_query_bundle(
        model,
        adapter,
        injection_encoding,
        layers=[layer],
        capture_attention=False,
        capture_values=False,
        cache_logit_tolerance=config.attention_cache_logit_tolerance,
    )
    directed = directed_intervention_logits(
        model,
        adapter,
        injection_encoding,
        bundle=injection_bundle,
        layer=layer,
        head=head,
        answer_direction=directions[layer]["u_answer_fit"],
        answer_step_scale=float(directions[layer]["answer_step_scale"]),
        injection_betas=(-1.0, 0.0, 1.0),
        orthogonal_label=f"{model_label}:smoke:L{layer}H{head}:orthogonal",
    )
    payload = {
        "schema_version": "realistic_niah_v4_4_3_smoke_complete_v1",
        "model_label": model_label,
        "seed": seed,
        "head": {"layer": layer, "head": head},
        "z_o_candidate_logit_max_abs_delta": zo_delta,
        "attention_cache_candidate_logit_max_abs_delta": max(
            receiver.attention_cache_candidate_logit_max_abs_delta,
            donor.attention_cache_candidate_logit_max_abs_delta,
        ),
        "attention_cache_candidate_centered_logit_max_abs_delta": max(
            receiver.attention_cache_candidate_centered_logit_max_abs_delta,
            donor.attention_cache_candidate_centered_logit_max_abs_delta,
        ),
        "patch_metrics": {
            name: intervention_metrics(
                baseline_output=receiver,
                intervened_output=values[0],
                encoding=encodings[receiver_count],
                donor_count=donor_count,
            )
            for name, values in patch.items()
        },
        "directed_metrics": {
            name: intervention_metrics(
                baseline_output=injection_bundle,
                intervened_output=values[0],
                encoding=injection_encoding,
            )
            for name, values in directed.items()
        },
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=model_label,
        stage="smoke",
        state="COMPLETE",
        detail={"z_o_candidate_logit_max_abs_delta": zo_delta},
    )


def run_staged_patch_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443Config,
    v4_config: V4Config,
    selection: Mapping[str, Any],
    resume: bool,
) -> pd.DataFrame:
    root = stage_root(run_root, model_label, "staged_patch")
    complete = root / "complete.json"
    detail_path = root / "detail.csv.gz"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Staged patch already complete: {complete}")
        return pd.read_csv(detail_path)
    write_stage_status(
        run_root,
        model_label=model_label,
        stage="staged_patch",
        state="RUNNING",
    )
    entries = _head_entries(selection)
    layers = sorted({int(entry["layer"]) for entry in entries})
    counts = sorted({value for pair in config.patch_pairs for value in pair})
    stimuli = _load_v4_stimulus_map(
        source_run_root, seeds=config.screen_seeds, counts=counts
    )
    design_hash = _stage_design_hash("staged_patch", config, selection)
    atomic_json(
        root / "design.json",
        {
            "schema_version": "realistic_niah_v4_4_3_stage_design_v1",
            "stage": "staged_patch",
            "design_hash": design_hash,
            "screen_seeds": list(config.screen_seeds),
            "directed_pairs": [list(pair) for pair in config.directed_patch_pairs],
            "head_entries": entries,
            "interventions": list(config.patch_interventions),
        },
    )
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    expected = set(config.patch_interventions)
    for seed in config.screen_seeds:
        encodings = _render_encodings(
            stimuli,
            tokenizer=tokenizer,
            model_label=model_label,
            v4_config=v4_config,
            seed=seed,
            counts=counts,
        )
        bundles = {
            count: capture_query_bundle(
                model,
                adapter,
                encodings[count],
                layers=layers,
                capture_attention=True,
                capture_values=True,
                cache_logit_tolerance=config.attention_cache_logit_tolerance,
            )
            for count in counts
        }
        for receiver_count, donor_count in config.directed_patch_pairs:
            for entry in entries:
                layer, head = int(entry["layer"]), int(entry["head"])
                shard = shard_root / (
                    f"seed{seed}_r{receiver_count}_d{donor_count}_"
                    f"{entry['parent_candidate']}_{entry['role']}_L{layer}H{head}.csv.gz"
                )
                if resume and shard.exists() and _validate_shard(
                    shard,
                    expected_interventions=expected,
                    design_hash=design_hash,
                ):
                    continue
                receiver_encoding = encodings[receiver_count]
                results = staged_patch_logits(
                    model,
                    adapter,
                    receiver_encoding,
                    encodings[donor_count],
                    receiver=bundles[receiver_count],
                    donor=bundles[donor_count],
                    layer=layer,
                    head=head,
                    scramble_fraction=config.position_scramble_fraction,
                    orthogonal_label=(
                        f"{model_label}:patch:{entry['parent_candidate']}:"
                        f"{entry['role']}:L{layer}H{head}"
                    ),
                )
                zo_delta = _zo_max_candidate_delta(
                    results["z_donor"][0],
                    results["o_donor"][0],
                )
                if zo_delta > config.strict_zo_equivalence_tolerance:
                    raise RuntimeError(
                        f"Z/O equivalence failed for {model_label} seed={seed} "
                        f"L{layer}H{head}: {zo_delta}"
                    )
                rows = []
                for intervention, (causal_output, delta_norm) in results.items():
                    rows.append(
                        {
                            "schema_version": "realistic_niah_v4_4_3_patch_row_v1",
                            "design_hash": design_hash,
                            "model_label": model_label,
                            "split": "screen",
                            "seed": int(seed),
                            "receiver_count": int(receiver_count),
                            "donor_count": int(donor_count),
                            "parent_candidate": entry["parent_candidate"],
                            "head_role": entry["role"],
                            "layer": layer,
                            "head": head,
                            "sentinel": bool(entry["sentinel"]),
                            "intervention": intervention,
                            "output_delta_norm": float(delta_norm),
                            "z_o_candidate_logit_max_abs_delta": zo_delta,
                            "attention_cache_candidate_logit_max_abs_delta": (
                                bundles[receiver_count].attention_cache_candidate_logit_max_abs_delta
                            ),
                            "attention_cache_candidate_centered_logit_max_abs_delta": (
                                bundles[receiver_count].attention_cache_candidate_centered_logit_max_abs_delta
                            ),
                            **intervention_metrics(
                                baseline_output=bundles[receiver_count],
                                intervened_output=causal_output,
                                encoding=receiver_encoding,
                                donor_count=donor_count,
                            ),
                        }
                    )
                atomic_csv_gzip(pd.DataFrame(rows), shard)
        del bundles, encodings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    shard_paths = sorted(shard_root.glob("*.csv.gz"))
    detail = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
    atomic_csv_gzip(detail, detail_path)
    payload = {
        "schema_version": "realistic_niah_v4_4_3_stage_complete_v1",
        "model_label": model_label,
        "stage": "staged_patch",
        "design_hash": design_hash,
        "shard_count": len(shard_paths),
        "row_count": int(len(detail)),
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=model_label,
        stage="staged_patch",
        state="COMPLETE",
        detail=payload,
    )
    return detail


def run_directed_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443Config,
    v4_config: V4Config,
    directions: Mapping[int, Mapping[str, Any]],
    selection: Mapping[str, Any],
    resume: bool,
) -> pd.DataFrame:
    root = stage_root(run_root, model_label, "directed")
    complete = root / "complete.json"
    detail_path = root / "detail.csv.gz"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Directed stage already complete: {complete}")
        return pd.read_csv(detail_path)
    write_stage_status(
        run_root,
        model_label=model_label,
        stage="directed",
        state="RUNNING",
    )
    entries = _head_entries(selection)
    layers = sorted({int(entry["layer"]) for entry in entries})
    counts = tuple(config.injection_counts)
    stimuli = _load_v4_stimulus_map(
        source_run_root, seeds=config.confirmation_seeds, counts=counts
    )
    design_hash = _stage_design_hash("directed", config, selection)
    expected = {
        "answer_direction_removal",
        "equal_norm_orthogonal_removal",
        *{
            f"signed_answer_direction_injection_beta_{float(beta):+g}"
            for beta in config.injection_betas
        },
    }
    atomic_json(
        root / "design.json",
        {
            "schema_version": "realistic_niah_v4_4_3_stage_design_v1",
            "stage": "directed",
            "design_hash": design_hash,
            "confirmation_seeds": list(config.confirmation_seeds),
            "counts": list(counts),
            "head_entries": entries,
            "interventions": sorted(expected),
            "injection_betas": list(config.injection_betas),
        },
    )
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for seed in config.confirmation_seeds:
        encodings = _render_encodings(
            stimuli,
            tokenizer=tokenizer,
            model_label=model_label,
            v4_config=v4_config,
            seed=seed,
            counts=counts,
        )
        bundles = {
            count: capture_query_bundle(
                model,
                adapter,
                encodings[count],
                layers=layers,
                capture_attention=False,
                capture_values=False,
                cache_logit_tolerance=config.attention_cache_logit_tolerance,
            )
            for count in counts
        }
        for count in counts:
            for entry in entries:
                layer, head = int(entry["layer"]), int(entry["head"])
                shard = shard_root / (
                    f"seed{seed}_n{count}_{entry['parent_candidate']}_"
                    f"{entry['role']}_L{layer}H{head}.csv.gz"
                )
                if resume and shard.exists() and _validate_shard(
                    shard,
                    expected_interventions=expected,
                    design_hash=design_hash,
                ):
                    continue
                result = directed_intervention_logits(
                    model,
                    adapter,
                    encodings[count],
                    bundle=bundles[count],
                    layer=layer,
                    head=head,
                    answer_direction=directions[layer]["u_answer_fit"],
                    answer_step_scale=float(directions[layer]["answer_step_scale"]),
                    injection_betas=config.injection_betas,
                    orthogonal_label=(
                        f"{model_label}:directed:{entry['parent_candidate']}:"
                        f"{entry['role']}:L{layer}H{head}"
                    ),
                )
                rows = []
                for intervention, (causal_output, delta_norm, beta) in result.items():
                    rows.append(
                        {
                            "schema_version": "realistic_niah_v4_4_3_directed_row_v1",
                            "design_hash": design_hash,
                            "model_label": model_label,
                            "split": "confirmation",
                            "seed": int(seed),
                            "gold_count": int(count),
                            "parent_candidate": entry["parent_candidate"],
                            "head_role": entry["role"],
                            "layer": layer,
                            "head": head,
                            "sentinel": bool(entry["sentinel"]),
                            "intervention": intervention,
                            "beta": beta,
                            "output_delta_norm": float(delta_norm),
                            **intervention_metrics(
                                baseline_output=bundles[count],
                                intervened_output=causal_output,
                                encoding=encodings[count],
                            ),
                        }
                    )
                atomic_csv_gzip(pd.DataFrame(rows), shard)
        del bundles, encodings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    shard_paths = sorted(shard_root.glob("*.csv.gz"))
    detail = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
    atomic_csv_gzip(detail, detail_path)
    payload = {
        "schema_version": "realistic_niah_v4_4_3_stage_complete_v1",
        "model_label": model_label,
        "stage": "directed",
        "design_hash": design_hash,
        "shard_count": len(shard_paths),
        "row_count": int(len(detail)),
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=model_label,
        stage="directed",
        state="COMPLETE",
        detail=payload,
    )
    return detail


def run_model_campaign(
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443Config,
    v4_config_path: str | Path,
    cache_dir: str | Path,
    device_map: str,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    model_complete_path = Path(run_root) / "models" / model_label / "complete.json"
    if resume and model_complete_path.is_file():
        return json.loads(model_complete_path.read_text(encoding="utf-8"))
    v4_config = V4Config.from_json(v4_config_path)
    write_stage_status(
        run_root,
        model_label=model_label,
        stage="campaign",
        state="LOADING_MODEL",
    )
    model, tokenizer, adapter = _load_model(
        model_label,
        config=config,
        cache_dir=cache_dir,
        device_map=device_map,
    )
    try:
        directions, selection = run_mapping_stage(
            model,
            adapter,
            source_run_root=source_run_root,
            run_root=run_root,
            model_label=model_label,
            config=config,
            resume=resume,
        )
        run_smoke_stage(
            model,
            tokenizer,
            adapter,
            source_run_root=source_run_root,
            run_root=run_root,
            model_label=model_label,
            config=config,
            v4_config=v4_config,
            directions=directions,
            selection=selection,
            resume=resume,
        )
        patch = run_staged_patch_stage(
            model,
            tokenizer,
            adapter,
            source_run_root=source_run_root,
            run_root=run_root,
            model_label=model_label,
            config=config,
            v4_config=v4_config,
            selection=selection,
            resume=resume,
        )
        directed = run_directed_stage(
            model,
            tokenizer,
            adapter,
            source_run_root=source_run_root,
            run_root=run_root,
            model_label=model_label,
            config=config,
            v4_config=v4_config,
            directions=directions,
            selection=selection,
            resume=resume,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": "realistic_niah_v4_4_3_model_complete_v1",
        "model_label": model_label,
        "patch_rows": int(len(patch)),
        "directed_rows": int(len(directed)),
        "candidate_heads": selection["candidate_heads"],
        "completed_unix": time.time(),
    }
    atomic_json(model_complete_path, payload)
    write_stage_status(
        run_root,
        model_label=model_label,
        stage="campaign",
        state="COMPLETE",
        detail=payload,
    )
    return payload
