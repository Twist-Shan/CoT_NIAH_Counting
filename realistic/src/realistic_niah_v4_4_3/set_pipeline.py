from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from realistic_niah_v4.spec import V4Config

from .geometry import fit_and_score_ov_heads, load_direction_source_states, resolve_value_source_layer
from .interventions import capture_query_bundle, intervention_metrics
from .io import (
    atomic_csv_gzip,
    atomic_json,
    atomic_jsonl,
    atomic_torch_save,
    stage_root,
    write_stage_status,
)
from .pipeline import (
    _load_model,
    _load_v4_stimulus_map,
    _render_encodings,
    _stable_hash,
    _validate_shard,
    _zo_max_candidate_delta,
)
from .set_geometry import build_set_direction_artifacts, select_candidate_and_control_sets
from .set_interventions import directed_set_intervention_logits, staged_set_patch_logits
from .set_spec import V443SetConfig


def _mapping_root(run_root: str | Path, model_label: str) -> Path:
    return stage_root(run_root, model_label, "mapping")


def _load_mapping_artifacts(
    run_root: str | Path, model_label: str
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    root = _mapping_root(run_root, model_label)
    base_payload = torch.load(
        root / "directions.pt", map_location="cpu", weights_only=False
    )
    set_payload = torch.load(
        root / "set_directions.pt", map_location="cpu", weights_only=False
    )
    selection = json.loads((root / "set_selection.json").read_text(encoding="utf-8"))
    if base_payload.get("model_label") != model_label:
        raise RuntimeError("Base-direction model mismatch")
    if set_payload.get("model_label") != model_label:
        raise RuntimeError("Set-direction model mismatch")
    if selection.get("model_label") != model_label:
        raise RuntimeError("Set-selection model mismatch")
    base = {int(layer): value for layer, value in base_payload["directions"].items()}
    return base, selection, set_payload["set_directions"]


def run_set_mapping_stage(
    model: Any,
    adapter: Any,
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443SetConfig,
    resume: bool,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    root = _mapping_root(run_root, model_label)
    complete = root / "complete.json"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Set mapping already complete: {complete}")
        return _load_mapping_artifacts(run_root, model_label)
    write_stage_status(run_root, model_label=model_label, stage="mapping", state="RUNNING")
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
    head_scores, base_directions = fit_and_score_ov_heads(
        model,
        adapter,
        prompt_states=prompt,
        answer_states=answer,
        model_label=model_label,
        config=config,
    )
    selection, set_scores = select_candidate_and_control_sets(
        head_scores,
        base_directions,
        model_label=model_label,
        config=config,
    )
    set_directions = build_set_direction_artifacts(
        adapter, base_directions, selection
    )
    root.mkdir(parents=True, exist_ok=True)
    atomic_csv_gzip(head_scores, root / "head_mapping_scores.csv.gz")
    atomic_csv_gzip(set_scores, root / "set_mapping_scores.csv.gz")
    selection_path = root / "set_selection.json"
    atomic_json(selection_path, selection)
    atomic_jsonl(root / "input_shards.jsonl", input_records)
    atomic_torch_save(
        {
            "schema_version": "realistic_niah_v4_4_3_set_base_directions_v1",
            "model_label": model_label,
            "directions": base_directions,
        },
        root / "directions.pt",
    )
    atomic_torch_save(
        {
            "schema_version": "realistic_niah_v4_4_3_set_directions_v1",
            "model_label": model_label,
            "set_directions": set_directions,
        },
        root / "set_directions.pt",
    )
    payload = {
        "schema_version": "realistic_niah_v4_4_3_set_mapping_complete_v1",
        "model_label": model_label,
        "head_score_rows": int(len(head_scores)),
        "set_score_rows": int(len(set_scores)),
        "candidate_set_count": len(selection["candidate_sets"]),
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
    return base_directions, selection, set_directions


def _set_entries(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    controls = selection["matched_control_sets"]
    for candidate in selection["candidate_sets"]:
        set_id = str(candidate["set_id"])
        for role, item in (
            ("candidate_set", candidate),
            ("matched_set", controls[set_id]),
        ):
            entries.append(
                {
                    "set_id": set_id,
                    "set_role": role,
                    "layer": int(item["layer"]),
                    "set_size": int(item["size"]),
                    "heads": tuple(int(head) for head in item["heads"]),
                }
            )
    return entries


def _stage_design_hash(
    stage: str, config: V443SetConfig, selection: Mapping[str, Any]
) -> str:
    return _stable_hash(
        {"stage": stage, "config": config.to_dict(), "selection": selection}
    )


def run_set_smoke_stage(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443SetConfig,
    v4_config: V4Config,
    base_directions: Mapping[int, Mapping[str, Any]],
    selection: Mapping[str, Any],
    set_directions: Mapping[str, Mapping[str, Any]],
    resume: bool,
) -> None:
    root = stage_root(run_root, model_label, "smoke")
    complete = root / "complete.json"
    if complete.exists():
        if resume:
            return
        raise FileExistsError(f"Set smoke already complete: {complete}")
    write_stage_status(run_root, model_label=model_label, stage="smoke", state="RUNNING")
    entry = _set_entries(selection)[0]
    layer = int(entry["layer"])
    seed = int(config.screen_seeds[0])
    receiver_count, donor_count = config.patch_pairs[0]
    counts = sorted({receiver_count, donor_count, config.injection_counts[0]})
    stimuli = _load_v4_stimulus_map(source_run_root, seeds=[seed], counts=counts)
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
    patch = staged_set_patch_logits(
        model,
        adapter,
        encodings[receiver_count],
        encodings[donor_count],
        receiver=receiver,
        donor=donor,
        layer=layer,
        heads=entry["heads"],
        scramble_fraction=config.position_scramble_fraction,
        orthogonal_label=f"{model_label}:set-smoke:{entry['set_id']}:norm",
    )
    zo_delta = _zo_max_candidate_delta(patch["z_donor"][0], patch["o_donor"][0])
    if zo_delta > config.strict_zo_equivalence_tolerance:
        raise RuntimeError(
            f"Set Z/O smoke failed: {zo_delta} > {config.strict_zo_equivalence_tolerance}"
        )
    directed_encoding = encodings[config.injection_counts[0]]
    directed_bundle = capture_query_bundle(
        model,
        adapter,
        directed_encoding,
        layers=[layer],
        capture_attention=False,
        capture_values=False,
        cache_logit_tolerance=config.attention_cache_logit_tolerance,
    )
    artifact = set_directions[f"{entry['set_id']}:{entry['set_role']}"]
    directed = directed_set_intervention_logits(
        model,
        adapter,
        directed_encoding,
        bundle=directed_bundle,
        layer=layer,
        heads=entry["heads"],
        answer_direction=base_directions[layer]["u_answer_fit"],
        reachable_answer_direction=artifact["reachable_answer_direction"],
        answer_step_scale=float(artifact["answer_step_scale"]),
        injection_betas=(-1.0, 0.0, 1.0),
        orthogonal_label=f"{model_label}:set-smoke:{entry['set_id']}:orthogonal",
    )
    payload = {
        "schema_version": "realistic_niah_v4_4_3_set_smoke_complete_v1",
        "model_label": model_label,
        "seed": seed,
        "set": entry,
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
                intervened_output=value[0],
                encoding=encodings[receiver_count],
                donor_count=donor_count,
            )
            for name, value in patch.items()
        },
        "directed_metrics": {
            name: intervention_metrics(
                baseline_output=directed_bundle,
                intervened_output=value[0],
                encoding=directed_encoding,
            )
            for name, value in directed.items()
        },
        "reachable_answer_cosine": float(artifact["reachable_answer_cosine"]),
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


def run_set_staged_patch_stage(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443SetConfig,
    v4_config: V4Config,
    selection: Mapping[str, Any],
    resume: bool,
) -> pd.DataFrame:
    root = stage_root(run_root, model_label, "staged_patch")
    complete = root / "complete.json"
    detail_path = root / "detail.csv.gz"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Set staged patch already complete: {complete}")
        return pd.read_csv(detail_path)
    write_stage_status(run_root, model_label=model_label, stage="staged_patch", state="RUNNING")
    entries = _set_entries(selection)
    layers = sorted({int(entry["layer"]) for entry in entries})
    counts = sorted({value for pair in config.patch_pairs for value in pair})
    stimuli = _load_v4_stimulus_map(
        source_run_root, seeds=config.screen_seeds, counts=counts
    )
    design_hash = _stage_design_hash("set_staged_patch", config, selection)
    expected = set(config.patch_interventions)
    atomic_json(
        root / "design.json",
        {
            "schema_version": "realistic_niah_v4_4_3_set_stage_design_v1",
            "stage": "staged_patch",
            "design_hash": design_hash,
            "screen_seeds": list(config.screen_seeds),
            "directed_pairs": [list(pair) for pair in config.directed_patch_pairs],
            "set_entries": entries,
            "interventions": list(config.patch_interventions),
        },
    )
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
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
                layer = int(entry["layer"])
                shard = shard_root / (
                    f"seed{seed}_r{receiver_count}_d{donor_count}_"
                    f"{entry['set_id']}_{entry['set_role']}.csv.gz"
                )
                if resume and shard.exists() and _validate_shard(
                    shard,
                    expected_interventions=expected,
                    design_hash=design_hash,
                ):
                    continue
                results = staged_set_patch_logits(
                    model,
                    adapter,
                    encodings[receiver_count],
                    encodings[donor_count],
                    receiver=bundles[receiver_count],
                    donor=bundles[donor_count],
                    layer=layer,
                    heads=entry["heads"],
                    scramble_fraction=config.position_scramble_fraction,
                    orthogonal_label=(
                        f"{model_label}:set-patch:{entry['set_id']}:"
                        f"{entry['set_role']}"
                    ),
                )
                zo_delta = _zo_max_candidate_delta(
                    results["z_donor"][0], results["o_donor"][0]
                )
                if zo_delta > config.strict_zo_equivalence_tolerance:
                    raise RuntimeError(
                        f"Set Z/O equivalence failed for {model_label} seed={seed} "
                        f"{entry['set_id']}: {zo_delta}"
                    )
                rows = []
                for intervention, (causal_output, delta_norm) in results.items():
                    rows.append(
                        {
                            "schema_version": "realistic_niah_v4_4_3_set_patch_row_v1",
                            "design_hash": design_hash,
                            "model_label": model_label,
                            "split": "screen",
                            "seed": int(seed),
                            "receiver_count": int(receiver_count),
                            "donor_count": int(donor_count),
                            "set_id": entry["set_id"],
                            "set_role": entry["set_role"],
                            "layer": layer,
                            "set_size": int(entry["set_size"]),
                            "heads": ",".join(str(head) for head in entry["heads"]),
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
                                encoding=encodings[receiver_count],
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
        "schema_version": "realistic_niah_v4_4_3_set_stage_complete_v1",
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


def run_set_directed_stage(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443SetConfig,
    v4_config: V4Config,
    base_directions: Mapping[int, Mapping[str, Any]],
    selection: Mapping[str, Any],
    set_directions: Mapping[str, Mapping[str, Any]],
    resume: bool,
) -> pd.DataFrame:
    root = stage_root(run_root, model_label, "directed")
    complete = root / "complete.json"
    detail_path = root / "detail.csv.gz"
    if complete.exists():
        if not resume:
            raise FileExistsError(f"Set directed stage already complete: {complete}")
        return pd.read_csv(detail_path)
    write_stage_status(run_root, model_label=model_label, stage="directed", state="RUNNING")
    entries = _set_entries(selection)
    layers = sorted({int(entry["layer"]) for entry in entries})
    counts = tuple(config.injection_counts)
    stimuli = _load_v4_stimulus_map(
        source_run_root, seeds=config.confirmation_seeds, counts=counts
    )
    design_hash = _stage_design_hash("set_directed", config, selection)
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
            "schema_version": "realistic_niah_v4_4_3_set_stage_design_v1",
            "stage": "directed",
            "design_hash": design_hash,
            "confirmation_seeds": list(config.confirmation_seeds),
            "counts": list(counts),
            "set_entries": entries,
            "interventions": sorted(expected),
            "injection_betas": list(config.injection_betas),
            "injection_boundary": config.set_injection_boundary,
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
                shard = shard_root / (
                    f"seed{seed}_n{count}_{entry['set_id']}_"
                    f"{entry['set_role']}.csv.gz"
                )
                if resume and shard.exists() and _validate_shard(
                    shard,
                    expected_interventions=expected,
                    design_hash=design_hash,
                ):
                    continue
                layer = int(entry["layer"])
                artifact = set_directions[
                    f"{entry['set_id']}:{entry['set_role']}"
                ]
                result = directed_set_intervention_logits(
                    model,
                    adapter,
                    encodings[count],
                    bundle=bundles[count],
                    layer=layer,
                    heads=entry["heads"],
                    answer_direction=base_directions[layer]["u_answer_fit"],
                    reachable_answer_direction=artifact[
                        "reachable_answer_direction"
                    ],
                    answer_step_scale=float(artifact["answer_step_scale"]),
                    injection_betas=config.injection_betas,
                    orthogonal_label=(
                        f"{model_label}:set-directed:{entry['set_id']}:"
                        f"{entry['set_role']}"
                    ),
                )
                rows = []
                for intervention, (causal_output, delta_norm, beta) in result.items():
                    rows.append(
                        {
                            "schema_version": "realistic_niah_v4_4_3_set_directed_row_v1",
                            "design_hash": design_hash,
                            "model_label": model_label,
                            "split": "confirmation",
                            "seed": int(seed),
                            "gold_count": int(count),
                            "set_id": entry["set_id"],
                            "set_role": entry["set_role"],
                            "layer": layer,
                            "set_size": int(entry["set_size"]),
                            "heads": ",".join(str(head) for head in entry["heads"]),
                            "intervention": intervention,
                            "beta": beta,
                            "output_delta_norm": float(delta_norm),
                            "reachable_answer_cosine": float(
                                artifact["reachable_answer_cosine"]
                            ),
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
        "schema_version": "realistic_niah_v4_4_3_set_stage_complete_v1",
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


def run_set_model_campaign(
    *,
    source_run_root: str | Path,
    run_root: str | Path,
    model_label: str,
    config: V443SetConfig,
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
        base_directions, selection, set_directions = run_set_mapping_stage(
            model,
            adapter,
            source_run_root=source_run_root,
            run_root=run_root,
            model_label=model_label,
            config=config,
            resume=resume,
        )
        run_set_smoke_stage(
            model,
            tokenizer,
            adapter,
            source_run_root=source_run_root,
            run_root=run_root,
            model_label=model_label,
            config=config,
            v4_config=v4_config,
            base_directions=base_directions,
            selection=selection,
            set_directions=set_directions,
            resume=resume,
        )
        patch = run_set_staged_patch_stage(
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
        directed = run_set_directed_stage(
            model,
            tokenizer,
            adapter,
            source_run_root=source_run_root,
            run_root=run_root,
            model_label=model_label,
            config=config,
            v4_config=v4_config,
            base_directions=base_directions,
            selection=selection,
            set_directions=set_directions,
            resume=resume,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": "realistic_niah_v4_4_3_set_model_complete_v1",
        "model_label": model_label,
        "patch_rows": int(len(patch)),
        "directed_rows": int(len(directed)),
        "candidate_sets": selection["candidate_sets"],
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
