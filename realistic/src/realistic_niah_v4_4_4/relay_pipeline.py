from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah_v4.modeling import DecoderAdapter
from realistic_niah_v4.spec import V4Config
from realistic_niah_v4_4_3.interventions import (
    candidate_sequence_metrics,
    capture_query_bundle,
    intervention_metrics,
)
from realistic_niah_v4_4_3.io import (
    atomic_csv_gzip,
    atomic_json,
    atomic_torch_save,
    stage_root,
    write_stage_status,
)
from realistic_niah_v4_4_3.pipeline import (
    _load_model,
    _load_v4_stimulus_map,
    _render_encodings,
)
from realistic_niah_v4_4_3.set_interventions import run_with_set_z_deltas

from .interventions import (
    finite_diagnostics,
    natural_axis_diagnostics,
    set_output_from_stacked_z,
)
from .pipeline import (
    _direction_config,
    _fit_z_intercept,
    _stable_hash,
    v444_v4_config,
)
from .relay import (
    contribution_reconstruction_diagnostics,
    edge_z_from_values,
    global_axis_patch_diagnostics,
    natural_axis_block_for_patch_delta,
    positions_sha256,
    receiver_alpha_donor_v_delta,
    relay_carrier_coefficient,
    relay_removal_deltas,
    resolve_position_set,
    source_contribution_vector,
    stacked_delta_mapping,
)
from .relay_spec import V444RelayConfig
from .spec import V444Config


def _load_base_candidate(
    run_root: str | Path, base_config: V444Config
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = stage_root(run_root, base_config.model_label, "center_controls")
    selection_path = root / "selection.json"
    artifact_path = root / "artifacts.pt"
    if not selection_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError("Completed V4.4.4 center/control artifacts are missing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    candidate = selection["candidate"]
    if tuple(candidate["heads"]) != base_config.candidate_heads:
        raise RuntimeError("Frozen V4.4.4 candidate does not match relay config")
    artifact = payload["artifacts"][candidate["set_id"]]
    return selection, artifact


def _position_attention_mass(
    bundle: Any,
    *,
    layer: int,
    heads: Sequence[int],
    positions: Sequence[int],
) -> float:
    rows = bundle.alpha_by_layer[int(layer)].detach().float().cpu()
    start = int(bundle.alpha_key_start_by_layer[int(layer)])
    indices = torch.as_tensor(
        [int(position) - start for position in positions], dtype=torch.long
    )
    if int(indices.min()) < 0 or int(indices.max()) >= rows.shape[-1]:
        raise ValueError("Relay positions lie outside attention rows")
    return float(rows[list(map(int, heads))][:, indices].sum())


def _seed_slopes(frame: pd.DataFrame, *, value: str) -> np.ndarray:
    slopes = []
    for _seed, group in frame.groupby("seed", sort=True):
        x = group["gold_count"].to_numpy(float)
        y = group[value].to_numpy(float)
        centered = x - x.mean()
        denominator = float(np.sum(centered**2))
        if denominator <= 0:
            raise ValueError("Relay discovery count predictor is constant")
        slopes.append(float(np.sum(centered * (y - y.mean())) / denominator))
    return np.asarray(slopes, dtype=float)


def _select_relay_position_set(
    detail: pd.DataFrame, *, relay_config: V444RelayConfig
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for name in relay_config.all_position_sets:
        group = detail[detail["position_set"].eq(name)]
        slopes = _seed_slopes(group, value="global_axis_contribution_coefficient")
        mean = float(slopes.mean())
        standard_error = float(slopes.std(ddof=1) / math.sqrt(len(slopes)))
        score = mean / max(standard_error, 1e-12)
        positive_fraction = float(np.mean(slopes > 0))
        eligible = name in relay_config.eligible_relay_sets
        qualified = bool(
            eligible
            and mean > 0
            and positive_fraction
            >= relay_config.selection_min_positive_seed_fraction
        )
        rows.append(
            {
                "position_set": name,
                "set_role": "relay_candidate" if eligible else "source_control",
                "mean_count_slope": mean,
                "seed_slope_standard_error": standard_error,
                "selection_t_score": score,
                "positive_seed_fraction": positive_fraction,
                "selection_qualified": qualified,
            }
        )
    summary = pd.DataFrame(rows)
    eligible = summary[summary["set_role"].eq("relay_candidate")].copy()
    qualified = eligible[eligible["selection_qualified"]]
    pool = qualified if not qualified.empty else eligible
    selected = pool.sort_values(
        ["selection_t_score", "mean_count_slope", "position_set"],
        ascending=[False, False, True],
    ).iloc[0]
    payload = {
        "schema_version": "realistic_niah_v4_4_4_relay_selection_v1",
        "selected_position_set": str(selected["position_set"]),
        "selection_qualified": bool(selected["selection_qualified"]),
        "selection_rule": relay_config.selection_rule,
        "selection_min_positive_seed_fraction": (
            relay_config.selection_min_positive_seed_fraction
        ),
        "eligible_relay_sets": list(relay_config.eligible_relay_sets),
        "source_control_sets": list(relay_config.source_control_sets),
        "selection_uses_causal_outcomes": False,
        "selected_discovery_mean_slope": float(selected["mean_count_slope"]),
        "selected_discovery_t_score": float(selected["selection_t_score"]),
        "selected_positive_seed_fraction": float(
            selected["positive_seed_fraction"]
        ),
    }
    return payload, summary


def run_relay_discovery_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    run_root: str | Path,
    base_config: V444Config,
    relay_config: V444RelayConfig,
    v4_config: V4Config,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = stage_root(run_root, relay_config.model_label, "relay_discovery")
    complete = root / "complete.json"
    selection_path = root / "selection.json"
    artifact_path = root / "artifacts.pt"
    detail_path = root / "natural_position_set_detail.csv.gz"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Relay discovery is already complete: {complete}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        artifacts = torch.load(
            artifact_path, map_location="cpu", weights_only=False
        )["position_set_artifacts"]
        return selection, artifacts
    base_selection, base_artifact = _load_base_candidate(run_root, base_config)
    global_step = base_artifact["z_count_steps"].detach().float().cpu()
    global_axis = natural_axis_diagnostics(
        adapter,
        layer=relay_config.layer,
        heads=relay_config.heads,
        z_count_steps=global_step,
    )
    design = {
        "relay_config": relay_config.to_dict(),
        "base_selection_sha256": base_selection["selection_sha256"],
        "global_candidate_set_id": base_selection["candidate"]["set_id"],
    }
    design_hash = _stable_hash(design)
    atomic_json(
        root / "design.json",
        {
            "schema_version": "realistic_niah_v4_4_4_relay_discovery_design_v1",
            "design_hash": design_hash,
            **design,
        },
    )
    write_stage_status(
        run_root,
        model_label=relay_config.model_label,
        stage="relay_discovery",
        state="RUNNING",
    )
    stimuli = _load_v4_stimulus_map(
        run_root,
        seeds=relay_config.discovery_seeds,
        counts=relay_config.counts,
    )
    rows: list[dict[str, Any]] = []
    samples: dict[str, list[torch.Tensor]] = {
        name: [] for name in relay_config.all_position_sets
    }
    sample_counts: list[int] = []
    max_reconstruction = 0.0
    started = time.monotonic()
    for seed_offset, seed in enumerate(relay_config.discovery_seeds, start=1):
        encodings = _render_encodings(
            stimuli,
            tokenizer=tokenizer,
            model_label=relay_config.model_label,
            v4_config=v4_config,
            seed=seed,
            counts=relay_config.counts,
        )
        for count in relay_config.counts:
            encoding = encodings[count]
            bundle = capture_query_bundle(
                model,
                adapter,
                encoding,
                layers=(relay_config.layer,),
                capture_attention=True,
                capture_values=True,
                cache_logit_tolerance=relay_config.attention_cache_logit_tolerance,
            )
            reconstruction = contribution_reconstruction_diagnostics(
                bundle,
                adapter,
                layer=relay_config.layer,
                heads=relay_config.heads,
            )
            max_reconstruction = max(
                max_reconstruction,
                reconstruction["edge_z_reconstruction_relative_l2"],
            )
            contribution = source_contribution_vector(
                bundle,
                adapter,
                layer=relay_config.layer,
                heads=relay_config.heads,
                output_axis=global_axis["output_unit"],
            )
            for name in relay_config.all_position_sets:
                positions = resolve_position_set(
                    encoding, name, contribution=contribution
                )
                edge_z = edge_z_from_values(
                    bundle,
                    bundle,
                    adapter,
                    layer=relay_config.layer,
                    heads=relay_config.heads,
                    positions=positions,
                )
                output = set_output_from_stacked_z(
                    adapter,
                    layer=relay_config.layer,
                    heads=relay_config.heads,
                    stacked_z=edge_z,
                )
                projection = float(torch.dot(output, global_axis["output_unit"]))
                samples[name].append(edge_z)
                rows.append(
                    {
                        "schema_version": (
                            "realistic_niah_v4_4_4_relay_discovery_row_v1"
                        ),
                        "design_hash": design_hash,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "position_set": name,
                        "set_role": (
                            "relay_candidate"
                            if name in relay_config.eligible_relay_sets
                            else "source_control"
                        ),
                        "position_count": len(positions),
                        "positions_sha256": positions_sha256(positions),
                        "attention_mass_across_selected_heads": (
                            _position_attention_mass(
                                bundle,
                                layer=relay_config.layer,
                                heads=relay_config.heads,
                                positions=positions,
                            )
                        ),
                        "global_axis_contribution_projection": projection,
                        "global_axis_contribution_coefficient": projection
                        / float(global_axis["output_step_norm"]),
                        "edge_output_norm": float(torch.linalg.vector_norm(output)),
                        **reconstruction,
                    }
                )
            sample_counts.append(int(count))
            del bundle
        del encodings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - started
        remaining = elapsed / seed_offset * (
            len(relay_config.discovery_seeds) - seed_offset
        )
        print(
            f"[v4.4.4 relay discovery] seed={seed} "
            f"complete={seed_offset}/{len(relay_config.discovery_seeds)} "
            f"eta_seconds={remaining:.1f}",
            flush=True,
        )
    if max_reconstruction > relay_config.contribution_reconstruction_relative_tolerance:
        raise RuntimeError(
            "Receiver-alpha/V reconstruction exceeds tolerance: "
            f"{max_reconstruction}"
        )
    detail = pd.DataFrame(rows)
    expected = (
        len(relay_config.discovery_seeds)
        * len(relay_config.counts)
        * len(relay_config.all_position_sets)
    )
    if len(detail) != expected:
        raise RuntimeError("Relay discovery row count is incomplete")
    selection, selection_summary = _select_relay_position_set(
        detail, relay_config=relay_config
    )
    selection["base_selection_sha256"] = base_selection["selection_sha256"]
    selection["design_hash"] = design_hash
    selection["selection_sha256"] = _stable_hash(selection)
    fit_mask = torch.as_tensor(
        [count in relay_config.fit_counts for count in sample_counts], dtype=torch.bool
    )
    fit_count_values = [
        count
        for count in sample_counts
        if count in relay_config.fit_counts
    ]
    position_set_artifacts: dict[str, Any] = {}
    for name in relay_config.all_position_sets:
        tensor = torch.stack(samples[name], dim=0)
        center, step = _fit_z_intercept(tensor[fit_mask], fit_count_values)
        edge_axis = natural_axis_diagnostics(
            adapter,
            layer=relay_config.layer,
            heads=relay_config.heads,
            z_count_steps=step,
        )
        position_set_artifacts[name] = {
            "position_set": name,
            "set_role": (
                "relay_candidate"
                if name in relay_config.eligible_relay_sets
                else "source_control"
            ),
            "heads": list(relay_config.heads),
            "layer": relay_config.layer,
            "edge_z_center": center,
            "edge_z_count_step": step,
            "edge_output_step": edge_axis["output_step"],
            "edge_output_step_norm": edge_axis["output_step_norm"],
            "edge_global_axis_cosine": float(
                torch.dot(edge_axis["output_unit"], global_axis["output_unit"])
            ),
        }
    atomic_csv_gzip(detail, detail_path)
    atomic_csv_gzip(selection_summary, root / "position_set_selection_summary.csv.gz")
    atomic_json(selection_path, selection)
    atomic_torch_save(
        {
            "schema_version": "realistic_niah_v4_4_4_relay_artifacts_v1",
            "position_set_artifacts": position_set_artifacts,
            "global_z_count_steps": global_step,
            "global_output_step": global_axis["output_step"],
            "global_output_step_norm": global_axis["output_step_norm"],
        },
        artifact_path,
    )
    payload = {
        "schema_version": "realistic_niah_v4_4_4_relay_discovery_complete_v1",
        "design_hash": design_hash,
        "row_count": len(detail),
        "selected_position_set": selection["selected_position_set"],
        "selection_qualified": selection["selection_qualified"],
        "max_edge_z_reconstruction_relative_l2": max_reconstruction,
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=relay_config.model_label,
        stage="relay_discovery",
        state="COMPLETE",
        detail=payload,
    )
    return selection, position_set_artifacts


def _load_relay_artifacts(
    run_root: str | Path, relay_config: V444RelayConfig
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    root = stage_root(run_root, relay_config.model_label, "relay_discovery")
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    payload = torch.load(root / "artifacts.pt", map_location="cpu", weights_only=False)
    artifacts = payload["position_set_artifacts"]
    selected = artifacts[selection["selected_position_set"]]
    return selection, selected, payload["global_z_count_steps"].detach().float().cpu()


def run_relay_smoke_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    run_root: str | Path,
    relay_config: V444RelayConfig,
    v4_config: V4Config,
    resume: bool,
) -> dict[str, Any]:
    root = stage_root(run_root, relay_config.model_label, "relay_smoke")
    complete = root / "complete.json"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Relay smoke is already complete: {complete}")
        return json.loads(complete.read_text(encoding="utf-8"))
    selection, artifact, global_steps = _load_relay_artifacts(
        run_root, relay_config
    )
    seed = relay_config.confirmation_seeds[0]
    receiver_count, donor_count = relay_config.relay_pairs[0]
    stimuli = _load_v4_stimulus_map(
        run_root, seeds=(seed,), counts=(receiver_count, donor_count)
    )
    encodings = _render_encodings(
        stimuli,
        tokenizer=tokenizer,
        model_label=relay_config.model_label,
        v4_config=v4_config,
        seed=seed,
        counts=(receiver_count, donor_count),
    )
    bundles = {
        count: capture_query_bundle(
            model,
            adapter,
            encodings[count],
            layers=(relay_config.layer,),
            capture_attention=True,
            capture_values=True,
            cache_logit_tolerance=relay_config.attention_cache_logit_tolerance,
        )
        for count in (receiver_count, donor_count)
    }
    receiver = bundles[receiver_count]
    contribution = source_contribution_vector(
        receiver,
        adapter,
        layer=relay_config.layer,
        heads=relay_config.heads,
        output_axis=set_output_from_stacked_z(
            adapter,
            layer=relay_config.layer,
            heads=relay_config.heads,
            stacked_z=global_steps,
        ),
    )
    positions = resolve_position_set(
        encodings[receiver_count],
        selection["selected_position_set"],
        contribution=contribution,
    )
    reconstruction = contribution_reconstruction_diagnostics(
        receiver,
        adapter,
        layer=relay_config.layer,
        heads=relay_config.heads,
    )
    patch_delta = receiver_alpha_donor_v_delta(
        receiver,
        bundles[donor_count],
        adapter,
        layer=relay_config.layer,
        heads=relay_config.heads,
        positions=positions,
    )
    block, control, block_diag = natural_axis_block_for_patch_delta(
        adapter,
        layer=relay_config.layer,
        heads=relay_config.heads,
        patch_delta_z=patch_delta,
        global_z_count_steps=global_steps,
        orthogonal_label="v444:relay:smoke:block",
    )
    edge_z = edge_z_from_values(
        receiver,
        receiver,
        adapter,
        layer=relay_config.layer,
        heads=relay_config.heads,
        positions=positions,
    )
    removal, removal_control, removal_diag = relay_removal_deltas(
        adapter,
        layer=relay_config.layer,
        heads=relay_config.heads,
        edge_z=edge_z,
        edge_z_center=artifact["edge_z_center"],
        edge_z_count_step=artifact["edge_z_count_step"],
        orthogonal_label="v444:relay:smoke:removal",
    )
    patch_output = run_with_set_z_deltas(
        model,
        adapter,
        encodings[receiver_count],
        layer=relay_config.layer,
        deltas=stacked_delta_mapping(relay_config.heads, patch_delta),
    )
    predicted = set_output_from_stacked_z(
        adapter,
        layer=relay_config.layer,
        heads=relay_config.heads,
        stacked_z=patch_delta,
    )
    if patch_output.attention_output is None:
        raise RuntimeError("Relay smoke captured no post-O output")
    realized = (
        patch_output.attention_output
        - receiver.attention_output_by_layer[relay_config.layer]
    )
    pre_o_delta = float(torch.max(torch.abs(realized - predicted)))
    finite_diagnostics(block_diag)
    finite_diagnostics(removal_diag)
    if (
        reconstruction["edge_z_reconstruction_relative_l2"]
        > relay_config.contribution_reconstruction_relative_tolerance
    ):
        raise RuntimeError("Relay smoke edge reconstruction exceeds tolerance")
    if pre_o_delta > relay_config.pre_o_output_equivalence_tolerance:
        raise RuntimeError("Relay smoke pre-O delta does not reproduce W_O output")
    if abs(block_diag["relay_axis_block_residual_projection"]) > 1e-4:
        raise RuntimeError("Relay smoke natural-axis block leaves residual signal")
    if abs(block_diag["relay_axis_control_cosine"]) > 1e-4:
        raise RuntimeError("Relay smoke mediation control is not orthogonal")
    if abs(removal_diag["relay_removal_residual_projection"]) > 1e-4:
        raise RuntimeError("Relay smoke removal leaves relay-axis signal")
    if abs(removal_diag["relay_removal_control_axis_cosine"]) > 1e-4:
        raise RuntimeError("Relay smoke removal control is not orthogonal")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_relay_smoke_complete_v1",
        "seed": int(seed),
        "receiver_count": int(receiver_count),
        "donor_count": int(donor_count),
        "selected_position_set": selection["selected_position_set"],
        "position_count": len(positions),
        "pre_o_output_step_max_abs_delta": pre_o_delta,
        **reconstruction,
        **block_diag,
        **removal_diag,
        "block_z_l2": float(torch.linalg.vector_norm(block)),
        "block_control_z_l2": float(torch.linalg.vector_norm(control)),
        "removal_z_l2": float(torch.linalg.vector_norm(removal)),
        "removal_control_z_l2": float(torch.linalg.vector_norm(removal_control)),
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=relay_config.model_label,
        stage="relay_smoke",
        state="COMPLETE",
        detail=payload,
    )
    print(
        "[v4.4.4 relay smoke] PASS set={} positions={} reconstruction={:.3g} "
        "pre_o={:.3g}".format(
            selection["selected_position_set"],
            len(positions),
            reconstruction["edge_z_reconstruction_relative_l2"],
            pre_o_delta,
        ),
        flush=True,
    )
    return payload


def _valid_shard(path: Path, *, design_hash: str, rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return (
        len(frame) == int(rows)
        and "design_hash" in frame
        and set(frame["design_hash"].astype(str)) == {str(design_hash)}
        and not frame.isnull().all(axis=1).any()
    )


def run_relay_confirmation_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    run_root: str | Path,
    base_config: V444Config,
    relay_config: V444RelayConfig,
    v4_config: V4Config,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = stage_root(run_root, relay_config.model_label, "relay_confirmation")
    complete = root / "complete.json"
    natural_path = root / "natural_detail.csv.gz"
    patch_path = root / "edge_patch_detail.csv.gz"
    removal_path = root / "removal_detail.csv.gz"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Relay confirmation is complete: {complete}")
        return (
            pd.read_csv(natural_path),
            pd.read_csv(patch_path),
            pd.read_csv(removal_path),
        )
    base_selection, _base_artifact = _load_base_candidate(run_root, base_config)
    selection, artifact, global_steps = _load_relay_artifacts(
        run_root, relay_config
    )
    design = {
        "relay_config": relay_config.to_dict(),
        "base_selection_sha256": base_selection["selection_sha256"],
        "relay_selection_sha256": selection["selection_sha256"],
        "selected_position_set": selection["selected_position_set"],
        # Capturing an attention row switches the selected attention module to
        # an eager/cache path. Its final logits are a useful numerical audit,
        # but the direct validity gate for this experiment is reconstruction
        # of the original pre-O z from sum(alpha * V) at the selected layer.
        "attention_cache_policy": (
            "diagnostic_only__pre_o_edge_z_reconstruction_is_hard_gate_v2"
        ),
    }
    design_hash = _stable_hash(design)
    atomic_json(
        root / "design.json",
        {
            "schema_version": "realistic_niah_v4_4_4_relay_confirmation_design_v1",
            "design_hash": design_hash,
            **design,
        },
    )
    natural_shards = root / "natural_shards"
    patch_shards = root / "edge_patch_shards"
    removal_shards = root / "removal_shards"
    for directory in (natural_shards, patch_shards, removal_shards):
        directory.mkdir(parents=True, exist_ok=True)
    expected_natural = len(relay_config.counts)
    expected_patch = len(relay_config.relay_pairs) * 3
    expected_removal = len(relay_config.removal_counts) * 2
    stimuli = _load_v4_stimulus_map(
        run_root,
        seeds=relay_config.confirmation_seeds,
        counts=relay_config.counts,
    )
    global_axis = natural_axis_diagnostics(
        adapter,
        layer=relay_config.layer,
        heads=relay_config.heads,
        z_count_steps=global_steps,
    )
    started = time.monotonic()
    write_stage_status(
        run_root,
        model_label=relay_config.model_label,
        stage="relay_confirmation",
        state="RUNNING",
        detail={"completed_seeds": 0, "total_seeds": len(relay_config.confirmation_seeds)},
    )
    for offset, seed in enumerate(relay_config.confirmation_seeds, start=1):
        seed_natural_path = natural_shards / f"seed{seed}.csv.gz"
        seed_patch_path = patch_shards / f"seed{seed}.csv.gz"
        seed_removal_path = removal_shards / f"seed{seed}.csv.gz"
        if resume and all(
            (
                _valid_shard(
                    seed_natural_path, design_hash=design_hash, rows=expected_natural
                ),
                _valid_shard(
                    seed_patch_path, design_hash=design_hash, rows=expected_patch
                ),
                _valid_shard(
                    seed_removal_path, design_hash=design_hash, rows=expected_removal
                ),
            )
        ):
            print(
                f"[v4.4.4 relay confirmation] seed={seed} resume-skip "
                f"({offset}/{len(relay_config.confirmation_seeds)})",
                flush=True,
            )
            continue
        encodings = _render_encodings(
            stimuli,
            tokenizer=tokenizer,
            model_label=relay_config.model_label,
            v4_config=v4_config,
            seed=seed,
            counts=relay_config.counts,
        )
        bundles = {
            count: capture_query_bundle(
                model,
                adapter,
                encodings[count],
                layers=(relay_config.layer,),
                capture_attention=True,
                capture_values=True,
                # Later layers can amplify a small eager/cache difference.
                # Preserve both logit deltas below, but hard-fail on direct
                # selected-layer alpha-V reconstruction instead.
                cache_logit_tolerance=math.inf,
            )
            for count in relay_config.counts
        }
        contributions: dict[int, torch.Tensor] = {}
        positions_by_count: dict[int, tuple[int, ...]] = {}
        edge_z_by_count: dict[int, torch.Tensor] = {}
        reconstruction_by_count: dict[int, dict[str, float]] = {}
        natural_rows: list[dict[str, Any]] = []
        patch_rows: list[dict[str, Any]] = []
        removal_rows: list[dict[str, Any]] = []
        for count in relay_config.counts:
            bundle = bundles[count]
            contribution = source_contribution_vector(
                bundle,
                adapter,
                layer=relay_config.layer,
                heads=relay_config.heads,
                output_axis=global_axis["output_unit"],
            )
            positions = resolve_position_set(
                encodings[count],
                selection["selected_position_set"],
                contribution=contribution,
            )
            edge_z = edge_z_from_values(
                bundle,
                bundle,
                adapter,
                layer=relay_config.layer,
                heads=relay_config.heads,
                positions=positions,
            )
            reconstruction = contribution_reconstruction_diagnostics(
                bundle,
                adapter,
                layer=relay_config.layer,
                heads=relay_config.heads,
            )
            if (
                reconstruction["edge_z_reconstruction_relative_l2"]
                > relay_config.contribution_reconstruction_relative_tolerance
            ):
                raise RuntimeError("Relay confirmation reconstruction exceeds tolerance")
            contributions[count] = contribution
            positions_by_count[count] = positions
            edge_z_by_count[count] = edge_z
            reconstruction_by_count[count] = reconstruction
            carrier = relay_carrier_coefficient(
                adapter,
                layer=relay_config.layer,
                heads=relay_config.heads,
                edge_z=edge_z,
                edge_z_center=artifact["edge_z_center"],
                edge_z_count_step=artifact["edge_z_count_step"],
            )
            baseline = candidate_sequence_metrics(
                bundle.candidate_log_scores, encodings[count]
            )
            global_projection = float(contribution[list(positions)].sum())
            natural_rows.append(
                {
                    "schema_version": "realistic_niah_v4_4_4_relay_natural_row_v1",
                    "design_hash": design_hash,
                    "seed": int(seed),
                    "gold_count": int(count),
                    "position_set": selection["selected_position_set"],
                    "position_count": len(positions),
                    "positions_sha256": positions_sha256(positions),
                    "global_axis_contribution_projection": global_projection,
                    "global_axis_contribution_coefficient": global_projection
                    / float(global_axis["output_step_norm"]),
                    "attention_mass_across_selected_heads": _position_attention_mass(
                        bundle,
                        layer=relay_config.layer,
                        heads=relay_config.heads,
                        positions=positions,
                    ),
                    "attention_cache_candidate_logit_max_abs_delta": float(
                        bundle.attention_cache_candidate_logit_max_abs_delta
                    ),
                    "attention_cache_candidate_centered_logit_max_abs_delta": float(
                        bundle.attention_cache_candidate_centered_logit_max_abs_delta
                    ),
                    "attention_cache_reference_tolerance_exceeded": bool(
                        bundle.attention_cache_candidate_centered_logit_max_abs_delta
                        > relay_config.attention_cache_logit_tolerance
                    ),
                    "baseline_expected_count": baseline["expected_count"],
                    "baseline_correct_margin": baseline["correct_count_margin"],
                    **carrier,
                    **reconstruction,
                }
            )
        for receiver_count, donor_count in relay_config.relay_pairs:
            receiver = bundles[receiver_count]
            positions = positions_by_count[receiver_count]
            patch_delta = receiver_alpha_donor_v_delta(
                receiver,
                bundles[donor_count],
                adapter,
                layer=relay_config.layer,
                heads=relay_config.heads,
                positions=positions,
            )
            block, control, diagnostics = natural_axis_block_for_patch_delta(
                adapter,
                layer=relay_config.layer,
                heads=relay_config.heads,
                patch_delta_z=patch_delta,
                global_z_count_steps=global_steps,
                orthogonal_label=(
                    f"v444:relay:block:{seed}:{receiver_count}:{donor_count}"
                ),
            )
            finite_diagnostics(diagnostics)
            conditions = {
                "receiver_alpha_donor_v_edge_patch": patch_delta,
                "receiver_alpha_donor_v_edge_patch_plus_natural_axis_block": (
                    patch_delta + block
                ),
                "receiver_alpha_donor_v_edge_patch_plus_orthogonal_control": (
                    patch_delta + control
                ),
            }
            for intervention, delta in conditions.items():
                output = run_with_set_z_deltas(
                    model,
                    adapter,
                    encodings[receiver_count],
                    layer=relay_config.layer,
                    deltas=stacked_delta_mapping(relay_config.heads, delta),
                )
                patch_rows.append(
                    {
                        "schema_version": (
                            "realistic_niah_v4_4_4_relay_edge_patch_row_v1"
                        ),
                        "design_hash": design_hash,
                        "seed": int(seed),
                        "receiver_count": int(receiver_count),
                        "donor_count": int(donor_count),
                        "position_set": selection["selected_position_set"],
                        "position_count": len(positions),
                        "positions_sha256": positions_sha256(positions),
                        "intervention": intervention,
                        "applied_z_delta_norm": float(torch.linalg.vector_norm(delta)),
                        **diagnostics,
                        **intervention_metrics(
                            baseline_output=receiver,
                            intervened_output=output,
                            encoding=encodings[receiver_count],
                            donor_count=donor_count,
                        ),
                    }
                )
        for count in relay_config.removal_counts:
            removal, control, diagnostics = relay_removal_deltas(
                adapter,
                layer=relay_config.layer,
                heads=relay_config.heads,
                edge_z=edge_z_by_count[count],
                edge_z_center=artifact["edge_z_center"],
                edge_z_count_step=artifact["edge_z_count_step"],
                orthogonal_label=f"v444:relay:removal:{seed}:{count}",
            )
            finite_diagnostics(diagnostics)
            for intervention, delta in (
                ("relay_axis_removal", removal),
                ("relay_axis_orthogonal_control", control),
            ):
                output = run_with_set_z_deltas(
                    model,
                    adapter,
                    encodings[count],
                    layer=relay_config.layer,
                    deltas=stacked_delta_mapping(relay_config.heads, delta),
                )
                removal_rows.append(
                    {
                        "schema_version": (
                            "realistic_niah_v4_4_4_relay_removal_row_v1"
                        ),
                        "design_hash": design_hash,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "position_set": selection["selected_position_set"],
                        "position_count": len(positions_by_count[count]),
                        "positions_sha256": positions_sha256(
                            positions_by_count[count]
                        ),
                        "intervention": intervention,
                        "applied_z_delta_norm": float(torch.linalg.vector_norm(delta)),
                        **diagnostics,
                        **intervention_metrics(
                            baseline_output=bundles[count],
                            intervened_output=output,
                            encoding=encodings[count],
                        ),
                    }
                )
        natural_frame = pd.DataFrame(natural_rows)
        patch_frame = pd.DataFrame(patch_rows)
        removal_frame = pd.DataFrame(removal_rows)
        if len(natural_frame) != expected_natural:
            raise RuntimeError("Relay natural shard has the wrong row count")
        if len(patch_frame) != expected_patch:
            raise RuntimeError("Relay edge-patch shard has the wrong row count")
        if len(removal_frame) != expected_removal:
            raise RuntimeError("Relay removal shard has the wrong row count")
        atomic_csv_gzip(natural_frame, seed_natural_path)
        atomic_csv_gzip(patch_frame, seed_patch_path)
        atomic_csv_gzip(removal_frame, seed_removal_path)
        write_stage_status(
            run_root,
            model_label=relay_config.model_label,
            stage="relay_confirmation",
            state="RUNNING",
            detail={
                "completed_seeds": offset,
                "total_seeds": len(relay_config.confirmation_seeds),
                "last_seed": seed,
            },
        )
        del bundles, encodings, contributions, edge_z_by_count
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - started
        remaining = elapsed / offset * (
            len(relay_config.confirmation_seeds) - offset
        )
        print(
            f"[v4.4.4 relay confirmation] seed={seed} "
            f"complete={offset}/{len(relay_config.confirmation_seeds)} "
            f"eta_seconds={remaining:.1f}",
            flush=True,
        )
    natural_files = sorted(natural_shards.glob("seed*.csv.gz"))
    patch_files = sorted(patch_shards.glob("seed*.csv.gz"))
    removal_files = sorted(removal_shards.glob("seed*.csv.gz"))
    expected_seeds = len(relay_config.confirmation_seeds)
    if not (
        len(natural_files)
        == len(patch_files)
        == len(removal_files)
        == expected_seeds
    ):
        raise RuntimeError("Relay confirmation shard inventory is incomplete")
    natural = pd.concat([pd.read_csv(path) for path in natural_files], ignore_index=True)
    patch = pd.concat([pd.read_csv(path) for path in patch_files], ignore_index=True)
    removal = pd.concat([pd.read_csv(path) for path in removal_files], ignore_index=True)
    atomic_csv_gzip(natural, natural_path)
    atomic_csv_gzip(patch, patch_path)
    atomic_csv_gzip(removal, removal_path)
    payload = {
        "schema_version": "realistic_niah_v4_4_4_relay_confirmation_complete_v1",
        "design_hash": design_hash,
        "selected_position_set": selection["selected_position_set"],
        "seed_count": expected_seeds,
        "natural_rows": len(natural),
        "edge_patch_rows": len(patch),
        "removal_rows": len(removal),
        "max_attention_cache_candidate_logit_delta": float(
            natural["attention_cache_candidate_logit_max_abs_delta"].max()
        ),
        "max_attention_cache_candidate_centered_logit_delta": float(
            natural[
                "attention_cache_candidate_centered_logit_max_abs_delta"
            ].max()
        ),
        "attention_cache_reference_tolerance_exceedance_rows": int(
            natural["attention_cache_reference_tolerance_exceeded"].astype(bool).sum()
        ),
        "max_edge_z_reconstruction_relative_l2": float(
            natural["edge_z_reconstruction_relative_l2"].max()
        ),
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=relay_config.model_label,
        stage="relay_confirmation",
        state="COMPLETE",
        detail=payload,
    )
    return natural, patch, removal


def run_relay_model_campaign(
    *,
    run_root: str | Path,
    base_config: V444Config,
    relay_config: V444RelayConfig,
    v4_config_path: str | Path,
    cache_dir: str | Path,
    device_map: str,
    resume: bool,
) -> dict[str, Any]:
    relay_config.validate_against_base(base_config)
    model_root = stage_root(run_root, relay_config.model_label, "relay_model")
    complete = model_root / "complete.json"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Relay model campaign is complete: {complete}")
        return json.loads(complete.read_text(encoding="utf-8"))
    base_v4 = V4Config.from_json(v4_config_path)
    current_v4 = v444_v4_config(base_v4, base_config)
    model, tokenizer, adapter = _load_model(
        relay_config.model_label,
        config=_direction_config(base_config),
        cache_dir=cache_dir,
        device_map=device_map,
    )
    try:
        selection, _artifacts = run_relay_discovery_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            base_config=base_config,
            relay_config=relay_config,
            v4_config=current_v4,
            resume=resume,
        )
        smoke = run_relay_smoke_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            relay_config=relay_config,
            v4_config=current_v4,
            resume=resume,
        )
        natural, patch, removal = run_relay_confirmation_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            base_config=base_config,
            relay_config=relay_config,
            v4_config=current_v4,
            resume=resume,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": "realistic_niah_v4_4_4_relay_model_complete_v1",
        "selected_position_set": selection["selected_position_set"],
        "selection_sha256": selection["selection_sha256"],
        "smoke": smoke,
        "natural_rows": len(natural),
        "edge_patch_rows": len(patch),
        "removal_rows": len(removal),
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    return payload
