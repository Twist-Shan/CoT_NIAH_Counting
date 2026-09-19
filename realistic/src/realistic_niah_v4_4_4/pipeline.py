from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from dataset_generation.dynamic_niah import TokenizerAdapter
from realistic_niah_v4.modeling import DecoderAdapter
from realistic_niah_v4.spec import V4Config
from realistic_niah_v4.stimuli import ControlledFreezeSpec, build_controlled_family
from realistic_niah_v4_4_3.geometry import (
    fit_and_score_ov_heads,
    load_direction_source_states,
    query_to_kv_head,
    resolve_value_source_layer,
)
from realistic_niah_v4_4_3.interventions import (
    candidate_sequence_metrics,
    capture_query_bundle,
    intervention_metrics,
)
from realistic_niah_v4_4_3.io import (
    atomic_csv_gzip,
    atomic_json,
    atomic_torch_save,
    source_input_manifest,
    stage_root,
    write_stage_status,
)
from realistic_niah_v4_4_3.pipeline import (
    _load_model,
    _load_v4_stimulus_map,
    _render_encodings,
)
from realistic_niah_v4_4_3.set_geometry import set_reachable_answer_direction
from realistic_niah_v4_4_3.set_interventions import (
    natural_ov_set_intervention_logits,
)
from realistic_niah_v4_4_3.spec import V443Config

from .interventions import (
    finite_diagnostics,
    max_abs_tensor_delta,
    natural_axis_diagnostics,
    natural_carrier_coefficient,
    natural_ov_mediation_deltas,
    natural_ov_mediation_logits,
    set_output_from_stacked_z,
)
from .spec import V444Config


NAMESPACE_NAME = "v4_4_4_natural_ov"


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
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
    config: V444Config,
    repo_root: str | Path,
    resume: bool,
) -> Path:
    config.validate()
    source = Path(source_run_root).resolve()
    namespace = Path(output_namespace_root).resolve()
    run = Path(run_root).resolve()
    repo = Path(repo_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Direction source run is missing: {source}")
    if namespace.name != NAMESPACE_NAME:
        raise ValueError(f"Output namespace must end in {NAMESPACE_NAME}")
    try:
        run.relative_to(namespace)
    except ValueError as error:
        raise ValueError("V4.4.4 run root must lie below its namespace") from error
    if run == namespace:
        raise ValueError("V4.4.4 run root needs a unique run id")
    if source == namespace or source in namespace.parents or namespace in source.parents:
        raise ValueError("Source and V4.4.4 output namespace overlap")
    config_path = run / "resolved_config.json"
    if run.exists():
        if not resume:
            raise FileExistsError(f"V4.4.4 run already exists: {run}")
        if not config_path.is_file():
            raise RuntimeError("Existing V4.4.4 run has no resolved config")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        normalized = json.loads(json.dumps(config.to_dict()))
        if existing != normalized:
            raise RuntimeError("Refusing to resume with a changed frozen config")
        return run
    namespace.mkdir(parents=True, exist_ok=True)
    run.mkdir()
    atomic_json(config_path, config.to_dict())
    atomic_json(run / "input_manifest.json", source_input_manifest(source))
    atomic_json(
        run / "owner.json",
        {
            "schema_version": "realistic_niah_v4_4_4_owner_v1",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_unix": time.time(),
            "python": platform.python_version(),
            "repo_commit": _git_commit(repo),
            "source_run_root": str(source),
            "output_namespace_root": str(namespace),
            "run_root": str(run),
            "write_contract": {
                "source_run_root": "read_only",
                "run_root": "exclusive_v4_4_4_writer",
                "raw_attention_rows": False,
                "full_hidden_states": False,
                "shards": "temporary_then_os_replace",
            },
        },
    )
    atomic_json(
        run / "campaign.status.json",
        {
            "schema_version": "realistic_niah_v4_4_4_campaign_status_v1",
            "state": "INITIALIZED",
            "updated_unix": time.time(),
        },
    )
    return run


def v444_v4_config(base: V4Config, config: V444Config) -> V4Config:
    result = replace(
        base,
        seeds=config.dataset_seeds,
        discovery_seeds=config.center_seeds,
        confirmation_seeds=config.confirmation_seeds,
    )
    result.validate()
    return result


def freeze_confirmation_dataset(
    *,
    run_root: str | Path,
    repo_root: str | Path,
    v4_config_path: str | Path,
    cache_dir: str | Path,
    config: V444Config,
    resume: bool,
) -> Path:
    root = Path(run_root) / "dataset"
    stimuli_path = root / "stimuli.jsonl"
    complete = root / "complete.json"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"V4.4.4 dataset already complete: {complete}")
        return stimuli_path
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="dataset",
        state="RUNNING",
    )
    repo = Path(repo_root).resolve()
    base = V4Config.from_json(v4_config_path)
    v4_config = v444_v4_config(base, config)
    freeze_spec = ControlledFreezeSpec(
        config=v4_config,
        haystack_dir=str(repo / "data" / "haystacks" / "paul_graham"),
        entities_path=str(repo / "data" / "entities" / "cities.csv"),
        fact_templates_path=str(
            repo / "data" / "templates" / "niah_fact_single_template.txt"
        ),
        tokenizer_cache_dir=str(Path(cache_dir).resolve()),
    )
    freeze_spec.validate()
    tokenizer = TokenizerAdapter(
        v4_config.canonical_tokenizer,
        revision=v4_config.canonical_tokenizer_revision,
        cache_dir=freeze_spec.tokenizer_cache_dir,
    )
    if tokenizer.backend != "huggingface":
        raise RuntimeError(
            f"Canonical tokenizer failed to load: {tokenizer.load_error}"
        )
    rows: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    for seed in config.dataset_seeds:
        family_rows, metadata = build_controlled_family(
            variant="v4.4",
            seed=int(seed),
            tokenizer=tokenizer,
            freeze_spec=freeze_spec,
            fixed_needles=None,
            active_counts=config.counts,
        )
        rows.extend(family_rows)
        families.append(metadata)
        print(
            f"[v4.4.4 dataset] seed={seed} rows={len(rows)}",
            flush=True,
        )
    expected = len(config.dataset_seeds) * len(config.counts)
    if len(rows) != expected:
        raise RuntimeError(f"Generated {len(rows)} rows; expected {expected}")
    keys = {(int(row["seed"]), int(row["gold_count"])) for row in rows}
    expected_keys = {
        (int(seed), int(count))
        for seed in config.dataset_seeds
        for count in config.counts
    }
    if keys != expected_keys:
        raise RuntimeError("Generated V4.4.4 dataset has an incomplete seed/count grid")
    payload = b"".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )
    root.mkdir(parents=True, exist_ok=True)
    temporary = stimuli_path.with_suffix(stimuli_path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(stimuli_path)
    manifest = {
        "schema_version": "realistic_niah_v4_4_4_dataset_manifest_v1",
        "design_variant": "v4.4",
        "seeds": list(config.dataset_seeds),
        "center_seeds": list(config.center_seeds),
        "confirmation_seeds": list(config.confirmation_seeds),
        "counts": list(config.counts),
        "rows": len(rows),
        "stimuli_sha256": _sha256(stimuli_path),
        "families": families,
        "v4_config": v4_config.to_dict(),
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        complete,
        {
            "schema_version": "realistic_niah_v4_4_4_dataset_complete_v1",
            "rows": len(rows),
            "stimuli_sha256": manifest["stimuli_sha256"],
            "completed_unix": time.time(),
        },
    )
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="dataset",
        state="COMPLETE",
        detail={"rows": len(rows)},
    )
    return stimuli_path


def _direction_config(config: V444Config) -> V443Config:
    result = V443Config(
        model_labels=(config.model_label,),
        discovery_seeds=config.direction_discovery_seeds,
        screen_seeds=(1254,),
        confirmation_seeds=(1255,),
        fit_counts=config.fit_counts,
        heldout_counts=config.heldout_counts,
        target_output_layers_qwen=(config.layer,),
        target_output_layers_gemma=(36,),
        model_torch_dtype=config.model_torch_dtype,
        attention_prefix_backend=config.attention_prefix_backend,
        attention_cache_logit_tolerance=config.attention_cache_logit_tolerance,
    )
    result.validate()
    return result


def run_direction_stage(
    model: Any,
    adapter: DecoderAdapter,
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    config: V444Config,
    resume: bool,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    root = stage_root(run_root, config.model_label, "directions")
    complete = root / "complete.json"
    score_path = root / "head_scores.csv.gz"
    direction_path = root / "directions.pt"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Direction stage already complete: {complete}")
        payload = torch.load(direction_path, map_location="cpu", weights_only=False)
        return pd.read_csv(score_path), {
            int(layer): value for layer, value in payload["directions"].items()
        }
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="directions",
        state="RUNNING",
    )
    direction_config = _direction_config(config)
    source_layers = {
        config.layer: resolve_value_source_layer(adapter, config.layer)
    }
    prompt, answer, input_records = load_direction_source_states(
        source_run_root,
        config.model_label,
        direction_config,
        prompt_source_layers=source_layers,
    )
    scores, directions = fit_and_score_ov_heads(
        model,
        adapter,
        prompt_states=prompt,
        answer_states=answer,
        model_label=config.model_label,
        config=direction_config,
    )
    print(
        f"[v4.4.4 directions] model={config.model_label} rows={len(scores)} "
        f"layer=L{config.layer}",
        flush=True,
    )
    layer_directions = directions[config.layer]
    required = {
        "z_count_step_fit_by_head",
        "mapped_fit_by_head",
        "mapped_holdout_by_head",
        "u_answer_fit",
        "u_answer_holdout",
    }
    missing = sorted(required - set(layer_directions))
    if missing:
        raise RuntimeError(f"Direction artifact lacks natural OV fields: {missing}")
    atomic_csv_gzip(scores, score_path)
    atomic_json(root / "input_records.json", {"records": input_records})
    atomic_torch_save(
        {
            "schema_version": "realistic_niah_v4_4_4_directions_v1",
            "model_label": config.model_label,
            "directions": directions,
            "direction_config": direction_config.to_dict(),
        },
        direction_path,
    )
    atomic_json(
        complete,
        {
            "schema_version": "realistic_niah_v4_4_4_direction_complete_v1",
            "rows": len(scores),
            "completed_unix": time.time(),
        },
    )
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="directions",
        state="COMPLETE",
        detail={"rows": len(scores)},
    )
    return scores, directions


def _fit_z_intercept(
    z_values: torch.Tensor, count_values: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    values = z_values.detach().float().cpu()
    counts = torch.as_tensor(count_values, dtype=torch.float32)
    if values.ndim != 3 or values.shape[0] != len(counts):
        raise ValueError("Z-intercept fit received incompatible samples/counts")
    centered = counts - counts.mean()
    denominator = torch.sum(centered.square())
    if float(denominator) <= 0:
        raise ValueError("Z-intercept fit needs multiple counts")
    slope = torch.einsum("n,nhd->hd", centered, values) / denominator
    intercept = values.mean(dim=0) - counts.mean() * slope
    return intercept, slope


def _set_features(
    adapter: DecoderAdapter,
    *,
    config: V444Config,
    heads: Sequence[int],
    z_steps_by_head: torch.Tensor,
    z_samples: torch.Tensor,
    answer_direction: torch.Tensor,
) -> dict[str, float]:
    index = torch.as_tensor(tuple(int(head) for head in heads), dtype=torch.long)
    steps = z_steps_by_head[index]
    axis = natural_axis_diagnostics(
        adapter,
        layer=config.layer,
        heads=heads,
        z_count_steps=steps,
    )
    answer = answer_direction.detach().float().cpu()
    answer = answer / torch.linalg.vector_norm(answer)
    natural_cosine = float(torch.dot(axis["output_unit"], answer))
    _reachable, reachable_cosine = set_reachable_answer_direction(
        adapter,
        layer=config.layer,
        heads=heads,
        answer_direction=answer,
    )
    norms = []
    for sample in z_samples:
        output = set_output_from_stacked_z(
            adapter,
            layer=config.layer,
            heads=heads,
            stacked_z=sample[index],
        )
        norms.append(float(torch.linalg.vector_norm(output)))
    baseline_norm = float(np.mean(norms))
    return {
        "natural_output_step_norm": float(axis["output_step_norm"]),
        "log_natural_output_step_norm": math.log(
            max(float(axis["output_step_norm"]), 1e-12)
        ),
        "natural_output_answer_cosine": natural_cosine,
        "reachable_answer_cosine": float(reachable_cosine),
        "discovery_baseline_output_norm": baseline_norm,
        "log_discovery_baseline_output_norm": math.log(max(baseline_norm, 1e-12)),
    }


def _entry_id(role: str, heads: Sequence[int]) -> str:
    return f"{role}_L28_" + "_".join(f"H{int(head)}" for head in heads)


def run_center_and_control_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    run_root: str | Path,
    config: V444Config,
    v4_config: V4Config,
    head_scores: pd.DataFrame,
    directions: Mapping[int, Mapping[str, Any]],
    resume: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = stage_root(run_root, config.model_label, "center_controls")
    complete = root / "complete.json"
    selection_path = root / "selection.json"
    artifact_path = root / "artifacts.pt"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Center/control stage already complete: {complete}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        return selection, payload["artifacts"]
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="center_controls",
        state="RUNNING",
    )
    stimuli = _load_v4_stimulus_map(
        run_root, seeds=config.center_seeds, counts=config.counts
    )
    z_samples: list[torch.Tensor] = []
    sample_counts: list[int] = []
    cache_raw_max = 0.0
    cache_centered_max = 0.0
    for seed in config.center_seeds:
        encodings = _render_encodings(
            stimuli,
            tokenizer=tokenizer,
            model_label=config.model_label,
            v4_config=v4_config,
            seed=seed,
            counts=config.counts,
        )
        for count in config.counts:
            bundle = capture_query_bundle(
                model,
                adapter,
                encodings[count],
                layers=(config.layer,),
                capture_attention=False,
                capture_values=False,
                cache_logit_tolerance=config.attention_cache_logit_tolerance,
            )
            z_samples.append(
                bundle.z_by_layer[config.layer]
                .detach()
                .float()
                .cpu()
                .reshape(adapter.num_heads[config.layer], adapter.head_dims[config.layer])
            )
            sample_counts.append(int(count))
            cache_raw_max = max(
                cache_raw_max,
                bundle.attention_cache_candidate_logit_max_abs_delta,
            )
            cache_centered_max = max(
                cache_centered_max,
                bundle.attention_cache_candidate_centered_logit_max_abs_delta,
            )
        del encodings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[v4.4.4 center] seed={seed} "
            f"samples={len(z_samples)}/{len(config.center_seeds) * len(config.counts)}",
            flush=True,
        )
    z_tensor = torch.stack(z_samples, dim=0)
    z_intercept, z_observed_slope = _fit_z_intercept(z_tensor, sample_counts)
    layer_directions = directions[config.layer]
    z_steps = layer_directions["z_count_step_fit_by_head"].detach().float().cpu()
    score_layer = head_scores[head_scores["layer"].eq(config.layer)].copy()
    kv_by_head = {
        int(row.head): int(row.kv_head) for row in score_layer.itertuples()
    }
    query_heads = int(adapter.num_heads[config.layer])
    kv_heads = len(set(kv_by_head.values()))
    if query_heads % kv_heads:
        raise RuntimeError("Non-integral GQA grouping in frozen Qwen layer")
    group_size = query_heads // kv_heads
    candidate = config.candidate_heads
    candidate_kv = {kv_by_head[head] for head in candidate}
    if len(candidate_kv) != 1:
        raise RuntimeError("Frozen candidate does not share one KV head as expected")
    relative = tuple(head % group_size for head in candidate)
    if config.control_requires_same_gqa_relative_positions:
        eligible = [
            tuple(kv * group_size + offset for offset in relative)
            for kv in range(kv_heads)
            if tuple(kv * group_size + offset for offset in relative) != candidate
        ]
    else:
        eligible = [
            (left, right)
            for left in range(query_heads)
            for right in range(left + 1, query_heads)
            if {left, right}.isdisjoint(candidate)
            and kv_by_head[left] == kv_by_head[right]
        ]
    if len(eligible) < config.matched_control_count:
        raise RuntimeError("Too few GQA-matched control pairs")
    answer = layer_directions["u_answer_fit"]
    all_sets = [candidate, *eligible]
    feature_rows = []
    for heads in all_sets:
        features = _set_features(
            adapter,
            config=config,
            heads=heads,
            z_steps_by_head=z_steps,
            z_samples=z_tensor,
            answer_direction=answer,
        )
        feature_rows.append(
            {
                "heads": ",".join(str(head) for head in heads),
                "is_candidate": heads == candidate,
                "kv_heads": ",".join(str(kv_by_head[head]) for head in heads),
                "gqa_relative_positions": ",".join(
                    str(head % group_size) for head in heads
                ),
                **features,
            }
        )
    features_frame = pd.DataFrame(feature_rows)
    columns = list(config.control_match_features)
    matrix = features_frame[columns].to_numpy(float)
    scale = matrix.std(axis=0, ddof=0)
    scale[scale <= 1e-12] = 1.0
    target = matrix[0]
    distances = np.sqrt(np.sum(((matrix - target) / scale) ** 2, axis=1))
    features_frame["match_distance"] = distances
    control_frame = features_frame[~features_frame["is_candidate"]].sort_values(
        ["match_distance", "heads"]
    )
    selected_heads = [
        tuple(int(value) for value in text.split(","))
        for text in control_frame.head(config.matched_control_count)["heads"]
    ]
    entries = [
        {
            "set_id": _entry_id("candidate_core", candidate),
            "set_role": "candidate_core",
            "heads": list(candidate),
        },
        *[
            {
                "set_id": _entry_id("matched_control", heads),
                "set_role": "matched_control",
                "heads": list(heads),
            }
            for heads in selected_heads
        ],
    ]
    if config.include_factorial_single_heads:
        entries.extend(
            {
                "set_id": _entry_id("candidate_component", (head,)),
                "set_role": "candidate_component",
                "heads": [head],
            }
            for head in candidate
        )
    entries.extend(
        {
            "set_id": f"registered_nested_L28K{len(heads)}",
            "set_role": "registered_nested_robustness",
            "heads": list(heads),
        }
        for heads in config.secondary_nested_head_sets
    )
    artifacts: dict[str, dict[str, Any]] = {}
    feature_lookup = {
        row["heads"]: row for row in feature_rows
    }
    for entry in entries:
        heads = tuple(int(head) for head in entry["heads"])
        index = torch.as_tensor(heads, dtype=torch.long)
        steps = z_steps[index]
        center = z_intercept[index]
        axis = natural_axis_diagnostics(
            adapter,
            layer=config.layer,
            heads=heads,
            z_count_steps=steps,
        )
        reachable, reachable_cosine = set_reachable_answer_direction(
            adapter,
            layer=config.layer,
            heads=heads,
            answer_direction=answer,
        )
        artifacts[entry["set_id"]] = {
            **entry,
            "layer": config.layer,
            "z_count_steps": steps,
            "z_center": center,
            "observed_z_slope": z_observed_slope[index],
            "natural_output_step": axis["output_step"],
            "natural_output_step_norm": axis["output_step_norm"],
            "reachable_answer_direction": reachable,
            "reachable_answer_cosine": reachable_cosine,
        }
        if len(heads) == 2:
            artifacts[entry["set_id"]]["matching_features"] = feature_lookup[
                ",".join(str(head) for head in heads)
            ]
    selection = {
        "schema_version": "realistic_niah_v4_4_4_control_selection_v1",
        "candidate": entries[0],
        "matched_controls": [
            entry for entry in entries if entry["set_role"] == "matched_control"
        ],
        "factorial_components": [
            entry
            for entry in entries
            if entry["set_role"] == "candidate_component"
        ],
        "registered_nested_sets": [
            entry
            for entry in entries
            if entry["set_role"] == "registered_nested_robustness"
        ],
        "eligible_control_count": len(eligible),
        "match_features": columns,
        "gqa_group_size": group_size,
        "candidate_gqa_relative_positions": list(relative),
        "selection_uses_causal_outcomes": False,
        "center_definition": config.center_definition,
    }
    selection["selection_sha256"] = _stable_hash(selection)
    print(
        "[v4.4.4 controls] candidate={} controls={} nested_k={}".format(
            list(candidate),
            [list(heads) for heads in selected_heads],
            [len(heads) for heads in config.secondary_nested_head_sets],
        ),
        flush=True,
    )
    atomic_csv_gzip(features_frame, root / "control_candidates.csv.gz")
    atomic_json(selection_path, selection)
    atomic_torch_save(
        {
            "schema_version": "realistic_niah_v4_4_4_center_artifacts_v1",
            "artifacts": artifacts,
            "cache_raw_max": cache_raw_max,
            "cache_centered_max": cache_centered_max,
        },
        artifact_path,
    )
    atomic_json(
        complete,
        {
            "schema_version": "realistic_niah_v4_4_4_center_complete_v1",
            "center_sample_count": len(z_samples),
            "selected_control_count": len(selected_heads),
            "cache_raw_max": cache_raw_max,
            "cache_centered_max": cache_centered_max,
            "selection_sha256": selection["selection_sha256"],
            "completed_unix": time.time(),
        },
    )
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="center_controls",
        state="COMPLETE",
        detail={
            "center_sample_count": len(z_samples),
            "selected_control_count": len(selected_heads),
        },
    )
    return selection, artifacts


def _artifact_entries(
    selection: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    definitions = [
        selection["candidate"],
        *selection["matched_controls"],
        *selection.get("factorial_components", []),
        *selection.get("registered_nested_sets", []),
    ]
    output = []
    for item in definitions:
        set_id = str(item["set_id"])
        if set_id not in artifacts:
            raise RuntimeError(f"Missing frozen artifact for {set_id}")
        output.append(dict(artifacts[set_id]))
    return output


def run_smoke_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    run_root: str | Path,
    config: V444Config,
    v4_config: V4Config,
    selection: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    root = stage_root(run_root, config.model_label, "smoke")
    complete = root / "complete.json"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Smoke stage already complete: {complete}")
        return json.loads(complete.read_text(encoding="utf-8"))
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="smoke",
        state="RUNNING",
    )
    entry = artifacts[str(selection["candidate"]["set_id"])]
    seed = config.confirmation_seeds[0]
    counts = tuple(sorted({config.causal_counts[1], *config.mediation_pairs[0]}))
    stimuli = _load_v4_stimulus_map(run_root, seeds=(seed,), counts=counts)
    encodings = _render_encodings(
        stimuli,
        tokenizer=tokenizer,
        model_label=config.model_label,
        v4_config=v4_config,
        seed=seed,
        counts=counts,
    )
    bundles = {
        count: capture_query_bundle(
            model,
            adapter,
            encodings[count],
            layers=(config.layer,),
            capture_attention=False,
            capture_values=False,
            cache_logit_tolerance=config.attention_cache_logit_tolerance,
        )
        for count in counts
    }
    count = config.causal_counts[1]
    directed, diagnostics = natural_ov_set_intervention_logits(
        model,
        adapter,
        encodings[count],
        bundle=bundles[count],
        layer=config.layer,
        heads=entry["heads"],
        z_count_steps=entry["z_count_steps"],
        injection_betas=(0.0, 1.0),
        orthogonal_label="v444:smoke:removal",
        z_center=entry["z_center"],
    )
    beta_zero = directed["natural_ov_z_injection_beta_+0"][0]
    beta_one = directed["natural_ov_z_injection_beta_+1"][0]
    baseline_scores = bundles[count].candidate_log_scores
    zero_score_delta = max(
        abs(beta_zero.candidate_log_scores[key] - baseline_scores[key])
        for key in baseline_scores
    )
    if beta_one.attention_output is None:
        raise RuntimeError("Natural OV smoke captured no post-O output")
    predicted = entry["natural_output_step"].detach().float().cpu()
    realized = (
        beta_one.attention_output
        - bundles[count].attention_output_by_layer[config.layer]
    )
    pre_o_output_delta = max_abs_tensor_delta(realized, predicted)
    low, high = config.mediation_pairs[0]
    _donor_z, _block, _control, mediation_diag = natural_ov_mediation_deltas(
        adapter,
        receiver=bundles[low],
        donor=bundles[high],
        layer=config.layer,
        heads=entry["heads"],
        z_count_steps=entry["z_count_steps"],
        orthogonal_label="v444:smoke:mediation",
    )
    finite_diagnostics(diagnostics)
    finite_diagnostics(mediation_diag)
    if zero_score_delta != 0.0:
        raise RuntimeError(f"Beta=0 does not reproduce baseline: {zero_score_delta}")
    if pre_o_output_delta > config.pre_o_output_equivalence_tolerance:
        raise RuntimeError(
            "Pre-O natural step does not reproduce W_O output step: "
            f"{pre_o_output_delta}"
        )
    if abs(diagnostics["control_output_cosine_with_removed_axis"]) > 1e-4:
        raise RuntimeError("Removal control is not orthogonal to natural axis")
    if abs(mediation_diag["blocked_patch_residual_axis_component"]) > 1e-4:
        raise RuntimeError("Mediation block leaves a natural-axis patch component")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_smoke_complete_v1",
        "seed": seed,
        "beta_zero_candidate_score_max_abs_delta": zero_score_delta,
        "pre_o_output_step_max_abs_delta": pre_o_output_delta,
        "removal_output_norm": diagnostics["removal_output_norm"],
        "removal_control_output_norm": diagnostics["control_output_norm"],
        "removal_control_axis_cosine": diagnostics[
            "control_output_cosine_with_removed_axis"
        ],
        **{f"mediation_{key}": value for key, value in mediation_diag.items()},
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    print(
        "[v4.4.4 smoke] PASS beta0={:.3g} pre_o_max_abs={:.6g} "
        "removal_cos={:.3g} mediation_residual={:.3g}".format(
            zero_score_delta,
            pre_o_output_delta,
            diagnostics["control_output_cosine_with_removed_axis"],
            mediation_diag["blocked_patch_residual_axis_component"],
        ),
        flush=True,
    )
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="smoke",
        state="COMPLETE",
        detail=payload,
    )
    return payload


def _valid_seed_shard(path: Path, *, design_hash: str, rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return (
        len(frame) == int(rows)
        and set(frame["design_hash"].astype(str)) == {str(design_hash)}
        and not frame.isnull().all(axis=1).any()
    )


def run_confirmation_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    run_root: str | Path,
    config: V444Config,
    v4_config: V4Config,
    selection: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = stage_root(run_root, config.model_label, "confirmation")
    complete = root / "complete.json"
    natural_detail_path = root / "natural_activation_detail.csv.gz"
    directed_detail_path = root / "directed_detail.csv.gz"
    mediation_detail_path = root / "mediation_detail.csv.gz"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Confirmation stage already complete: {complete}")
        return (
            pd.read_csv(natural_detail_path),
            pd.read_csv(directed_detail_path),
            pd.read_csv(mediation_detail_path),
        )
    entries = _artifact_entries(selection, artifacts)
    primary_entries = [
        entry
        for entry in entries
        if entry["set_role"] in {"candidate_core", "matched_control"}
    ]
    design = {
        "config": config.to_dict(),
        "selection_sha256": selection["selection_sha256"],
        "entries": [
            {
                "set_id": entry["set_id"],
                "set_role": entry["set_role"],
                "heads": entry["heads"],
            }
            for entry in entries
        ],
    }
    design_hash = _stable_hash(design)
    atomic_json(
        root / "design.json",
        {
            "schema_version": "realistic_niah_v4_4_4_confirmation_design_v1",
            "design_hash": design_hash,
            **design,
        },
    )
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="confirmation",
        state="RUNNING",
        detail={"completed_seeds": 0, "total_seeds": len(config.confirmation_seeds)},
    )
    natural_shards = root / "natural_shards"
    directed_shards = root / "directed_shards"
    mediation_shards = root / "mediation_shards"
    for directory in (natural_shards, directed_shards, mediation_shards):
        directory.mkdir(parents=True, exist_ok=True)
    expected_natural = len(entries) * len(config.counts)
    expected_directed = (
        len(entries)
        * len(config.causal_counts)
        * (2 + len(config.injection_betas))
    )
    expected_mediation = (
        len(primary_entries) * len(config.mediation_pairs) * 3
    )
    stimuli = _load_v4_stimulus_map(
        run_root, seeds=config.confirmation_seeds, counts=config.counts
    )
    confirmation_started = time.monotonic()
    for seed_offset, seed in enumerate(config.confirmation_seeds, start=1):
        seed_started = time.monotonic()
        natural_path = natural_shards / f"seed{seed}.csv.gz"
        directed_path = directed_shards / f"seed{seed}.csv.gz"
        mediation_path = mediation_shards / f"seed{seed}.csv.gz"
        if resume and all(
            (
                _valid_seed_shard(
                    natural_path, design_hash=design_hash, rows=expected_natural
                ),
                _valid_seed_shard(
                    directed_path, design_hash=design_hash, rows=expected_directed
                ),
                _valid_seed_shard(
                    mediation_path, design_hash=design_hash, rows=expected_mediation
                ),
            )
        ):
            print(
                f"[v4.4.4 confirmation] seed={seed} resume-skip "
                f"({seed_offset}/{len(config.confirmation_seeds)})",
                flush=True,
            )
            continue
        encodings = _render_encodings(
            stimuli,
            tokenizer=tokenizer,
            model_label=config.model_label,
            v4_config=v4_config,
            seed=seed,
            counts=config.counts,
        )
        bundles = {
            count: capture_query_bundle(
                model,
                adapter,
                encodings[count],
                layers=(config.layer,),
                capture_attention=False,
                capture_values=False,
                cache_logit_tolerance=config.attention_cache_logit_tolerance,
            )
            for count in config.counts
        }
        natural_rows: list[dict[str, Any]] = []
        directed_rows: list[dict[str, Any]] = []
        mediation_rows: list[dict[str, Any]] = []
        for entry in entries:
            heads_text = ",".join(str(head) for head in entry["heads"])
            for count in config.counts:
                carrier = natural_carrier_coefficient(
                    adapter,
                    bundle=bundles[count],
                    layer=config.layer,
                    heads=entry["heads"],
                    z_count_steps=entry["z_count_steps"],
                    z_center=entry["z_center"],
                )
                baseline = candidate_sequence_metrics(
                    bundles[count].candidate_log_scores, encodings[count]
                )
                natural_rows.append(
                    {
                        "schema_version": "realistic_niah_v4_4_4_natural_row_v1",
                        "design_hash": design_hash,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "set_id": entry["set_id"],
                        "set_role": entry["set_role"],
                        "heads": heads_text,
                        "baseline_expected_count": baseline["expected_count"],
                        "baseline_correct_margin": baseline["correct_count_margin"],
                        "baseline_predicted_count": baseline[
                            "predicted_count_among_candidates"
                        ],
                        **carrier,
                    }
                )
            for count in config.causal_counts:
                results, diagnostics = natural_ov_set_intervention_logits(
                    model,
                    adapter,
                    encodings[count],
                    bundle=bundles[count],
                    layer=config.layer,
                    heads=entry["heads"],
                    z_count_steps=entry["z_count_steps"],
                    injection_betas=config.injection_betas,
                    orthogonal_label=(
                        f"v444:directed:{seed}:{count}:{entry['set_id']}"
                    ),
                    z_center=entry["z_center"],
                )
                finite_diagnostics(diagnostics)
                for intervention, (output, output_norm, beta) in results.items():
                    directed_rows.append(
                        {
                            "schema_version": "realistic_niah_v4_4_4_directed_row_v1",
                            "design_hash": design_hash,
                            "seed": int(seed),
                            "gold_count": int(count),
                            "set_id": entry["set_id"],
                            "set_role": entry["set_role"],
                            "heads": heads_text,
                            "intervention": intervention,
                            "beta": beta,
                            "output_delta_norm": float(output_norm),
                            **diagnostics,
                            **intervention_metrics(
                                baseline_output=bundles[count],
                                intervened_output=output,
                                encoding=encodings[count],
                            ),
                        }
                    )
        for entry in primary_entries:
            heads_text = ",".join(str(head) for head in entry["heads"])
            for receiver_count, donor_count in config.mediation_pairs:
                results, diagnostics = natural_ov_mediation_logits(
                    model,
                    adapter,
                    encodings[receiver_count],
                    receiver=bundles[receiver_count],
                    donor=bundles[donor_count],
                    layer=config.layer,
                    heads=entry["heads"],
                    z_count_steps=entry["z_count_steps"],
                    orthogonal_label=(
                        f"v444:mediation:{seed}:{receiver_count}:{donor_count}:"
                        f"{entry['set_id']}"
                    ),
                )
                finite_diagnostics(diagnostics)
                for intervention, output in results.items():
                    mediation_rows.append(
                        {
                            "schema_version": "realistic_niah_v4_4_4_mediation_row_v1",
                            "design_hash": design_hash,
                            "seed": int(seed),
                            "receiver_count": int(receiver_count),
                            "donor_count": int(donor_count),
                            "set_id": entry["set_id"],
                            "set_role": entry["set_role"],
                            "heads": heads_text,
                            "intervention": intervention,
                            **diagnostics,
                            **intervention_metrics(
                                baseline_output=bundles[receiver_count],
                                intervened_output=output,
                                encoding=encodings[receiver_count],
                                donor_count=donor_count,
                            ),
                        }
                    )
        natural_frame = pd.DataFrame(natural_rows)
        directed_frame = pd.DataFrame(directed_rows)
        mediation_frame = pd.DataFrame(mediation_rows)
        if len(natural_frame) != expected_natural:
            raise RuntimeError("Natural activation shard has the wrong row count")
        if len(directed_frame) != expected_directed:
            raise RuntimeError("Directed shard has the wrong row count")
        if len(mediation_frame) != expected_mediation:
            raise RuntimeError("Mediation shard has the wrong row count")
        atomic_csv_gzip(natural_frame, natural_path)
        atomic_csv_gzip(directed_frame, directed_path)
        atomic_csv_gzip(mediation_frame, mediation_path)
        write_stage_status(
            run_root,
            model_label=config.model_label,
            stage="confirmation",
            state="RUNNING",
            detail={
                "completed_seeds": seed_offset,
                "total_seeds": len(config.confirmation_seeds),
                "last_seed": seed,
            },
        )
        del bundles, encodings, natural_frame, directed_frame, mediation_frame
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - confirmation_started
        mean_seconds = elapsed / seed_offset
        remaining = mean_seconds * (len(config.confirmation_seeds) - seed_offset)
        print(
            f"[v4.4.4 confirmation] seed={seed} "
            f"complete={seed_offset}/{len(config.confirmation_seeds)} "
            f"seed_seconds={time.monotonic() - seed_started:.1f} "
            f"eta_seconds={remaining:.1f}",
            flush=True,
        )
    natural_paths = sorted(natural_shards.glob("seed*.csv.gz"))
    directed_paths = sorted(directed_shards.glob("seed*.csv.gz"))
    mediation_paths = sorted(mediation_shards.glob("seed*.csv.gz"))
    if not (
        len(natural_paths)
        == len(directed_paths)
        == len(mediation_paths)
        == len(config.confirmation_seeds)
    ):
        raise RuntimeError("Confirmation shard count is incomplete")
    natural = pd.concat([pd.read_csv(path) for path in natural_paths], ignore_index=True)
    directed = pd.concat(
        [pd.read_csv(path) for path in directed_paths], ignore_index=True
    )
    mediation = pd.concat(
        [pd.read_csv(path) for path in mediation_paths], ignore_index=True
    )
    atomic_csv_gzip(natural, natural_detail_path)
    atomic_csv_gzip(directed, directed_detail_path)
    atomic_csv_gzip(mediation, mediation_detail_path)
    payload = {
        "schema_version": "realistic_niah_v4_4_4_confirmation_complete_v1",
        "design_hash": design_hash,
        "seed_count": len(config.confirmation_seeds),
        "natural_rows": len(natural),
        "directed_rows": len(directed),
        "mediation_rows": len(mediation),
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="confirmation",
        state="COMPLETE",
        detail=payload,
    )
    return natural, directed, mediation


def run_model_campaign(
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    repo_root: str | Path,
    config: V444Config,
    v4_config_path: str | Path,
    cache_dir: str | Path,
    device_map: str,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    model_root = Path(run_root) / "models" / config.model_label
    model_complete = model_root / "complete.json"
    if resume and model_complete.is_file():
        return json.loads(model_complete.read_text(encoding="utf-8"))
    base_v4_config = V4Config.from_json(v4_config_path)
    current_v4_config = v444_v4_config(base_v4_config, config)
    direction_config = _direction_config(config)
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="campaign",
        state="LOADING_MODEL",
    )
    model, tokenizer, adapter = _load_model(
        config.model_label,
        config=direction_config,
        cache_dir=cache_dir,
        device_map=device_map,
    )
    try:
        scores, directions = run_direction_stage(
            model,
            adapter,
            source_run_root=source_run_root,
            run_root=run_root,
            config=config,
            resume=resume,
        )
        selection, artifacts = run_center_and_control_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            config=config,
            v4_config=current_v4_config,
            head_scores=scores,
            directions=directions,
            resume=resume,
        )
        smoke = run_smoke_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            config=config,
            v4_config=current_v4_config,
            selection=selection,
            artifacts=artifacts,
            resume=resume,
        )
        natural, directed, mediation = run_confirmation_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            config=config,
            v4_config=current_v4_config,
            selection=selection,
            artifacts=artifacts,
            resume=resume,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": "realistic_niah_v4_4_4_model_complete_v1",
        "model_label": config.model_label,
        "selection_sha256": selection["selection_sha256"],
        "smoke": smoke,
        "natural_rows": len(natural),
        "directed_rows": len(directed),
        "mediation_rows": len(mediation),
        "completed_unix": time.time(),
    }
    atomic_json(model_complete, payload)
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage="campaign",
        state="COMPLETE",
        detail=payload,
    )
    return payload
