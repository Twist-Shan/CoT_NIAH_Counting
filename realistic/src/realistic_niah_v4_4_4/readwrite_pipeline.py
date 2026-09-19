from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from realistic_niah_v4.spec import V4Config
from realistic_niah_v4_4_3.interventions import (
    candidate_sequence_metrics,
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
from realistic_niah_v4_4_4.interventions import (
    _orthogonal_equal_output_norm_delta,
    natural_axis_diagnostics,
    set_output_from_stacked_z,
)
from realistic_niah_v4_4_4.pipeline import (
    _direction_config,
    _stable_hash,
    v444_v4_config,
)
from realistic_niah_v4_4_4.relay import natural_axis_block_for_patch_delta
from realistic_niah_v4_4_4.relay_pipeline import _load_base_candidate
from realistic_niah_v4_4_4.spec import V444Config

from .readwrite import (
    READ_COMPONENT_NAMES,
    capture_query_bundle_and_trace,
    fit_count_intercept_and_step,
    full_set_delta_from_bundles,
    read_component_output_diagnostics,
    residual_axis_coefficient,
    run_with_set_z_deltas_and_trace,
    shapley_read_decomposition,
    stable_position_partition,
    stacked_delta_mapping,
)
from .readwrite_spec import V444ReadWriteConfig


DISCOVERY_STAGE = "read_write_discovery"
SMOKE_STAGE = "read_write_smoke"
EVALUATION_STAGE = "read_write_evaluation"


def _attention_cache_diagnostics(
    bundle: Any, *, reference_tolerance: float
) -> dict[str, float | bool]:
    """Record eager/cache logit drift without treating it as a causal hard gate."""

    raw = float(bundle.attention_cache_candidate_logit_max_abs_delta)
    centered = float(bundle.attention_cache_candidate_centered_logit_max_abs_delta)
    if not math.isfinite(raw) or not math.isfinite(centered):
        raise RuntimeError("Attention-cache candidate-logit diagnostics are non-finite")
    return {
        "attention_cache_candidate_logit_max_abs_delta": raw,
        "attention_cache_candidate_centered_logit_max_abs_delta": centered,
        "attention_cache_reference_tolerance_exceeded": bool(
            centered > float(reference_tolerance)
        ),
    }


def _axis_projection(
    delta: torch.Tensor, step: torch.Tensor
) -> dict[str, float]:
    values = delta.detach().float().cpu()
    direction = step.detach().float().cpu()
    if values.shape != direction.shape:
        raise ValueError("Residual delta/count step shapes disagree")
    delta_norm = float(torch.linalg.vector_norm(values))
    step_norm = float(torch.linalg.vector_norm(direction))
    if step_norm <= 0:
        raise ValueError("A downstream count step is zero")
    projection = float(torch.dot(values.flatten(), direction.flatten()) / step_norm)
    return {
        "residual_delta_norm": delta_norm,
        "downstream_count_axis_projection": projection,
        "downstream_count_axis_coefficient": projection / step_norm,
        "downstream_count_axis_cosine": projection / max(delta_norm, 1e-12),
    }


def _attention_mass(
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
        raise ValueError("Position partition lies outside the captured attention row")
    return float(rows[list(map(int, heads))][:, indices].sum())


def _load_discovery_artifacts(
    run_root: str | Path, config: V444ReadWriteConfig
) -> dict[str, Any]:
    path = stage_root(run_root, config.model_label, DISCOVERY_STAGE) / "artifacts.pt"
    if not path.is_file():
        raise FileNotFoundError("V4.4.4 read/write discovery artifacts are missing")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != (
        "realistic_niah_v4_4_4_read_write_artifacts_v1"
    ):
        raise RuntimeError("Unexpected V4.4.4 read/write artifact schema")
    return payload


def run_discovery_stage(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    run_root: str | Path,
    base_config: V444Config,
    config: V444ReadWriteConfig,
    v4_config: V4Config,
    resume: bool,
) -> dict[str, Any]:
    root = stage_root(run_root, config.model_label, DISCOVERY_STAGE)
    complete = root / "complete.json"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Read/write discovery already complete: {complete}")
        return json.loads(complete.read_text(encoding="utf-8"))
    selection, base_artifact = _load_base_candidate(run_root, base_config)
    global_steps = base_artifact["z_count_steps"].detach().float().cpu()
    global_axis = natural_axis_diagnostics(
        adapter,
        layer=config.mediator_layer,
        heads=config.heads,
        z_count_steps=global_steps,
    )
    design = {
        "config": config.to_dict(),
        "base_selection_sha256": selection["selection_sha256"],
        "base_candidate_set_id": selection["candidate"]["set_id"],
    }
    design_hash = _stable_hash(design)
    atomic_json(
        root / "design.json",
        {
            "schema_version": "realistic_niah_v4_4_4_read_write_discovery_design_v1",
            "design_hash": design_hash,
            **design,
        },
    )
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=DISCOVERY_STAGE,
        state="RUNNING",
    )
    stimuli = _load_v4_stimulus_map(
        run_root, seeds=config.discovery_seeds, counts=config.counts
    )
    partition_rows: list[dict[str, Any]] = []
    residual_samples: list[torch.Tensor] = []
    sample_metadata: list[tuple[int, int, bool]] = []
    started = time.monotonic()
    for seed_offset, seed in enumerate(config.discovery_seeds, start=1):
        encodings = _render_encodings(
            stimuli,
            tokenizer=tokenizer,
            model_label=config.model_label,
            v4_config=v4_config,
            seed=seed,
            counts=config.counts,
        )
        for count in config.counts:
            encoding = encodings[count]
            bundle, residuals = capture_query_bundle_and_trace(
                model,
                adapter,
                encoding,
                mediator_layer=config.mediator_layer,
                trace_layers=config.downstream_layers,
                cache_logit_tolerance=base_config.attention_cache_logit_tolerance,
            )
            baseline = candidate_sequence_metrics(bundle.candidate_log_scores, encoding)
            groups = stable_position_partition(
                encoding, tail_width=config.tail_width
            )
            for name, positions in groups.items():
                from realistic_niah_v4_4_4.relay import edge_z_from_values

                edge_z = edge_z_from_values(
                    bundle,
                    bundle,
                    adapter,
                    layer=config.mediator_layer,
                    heads=config.heads,
                    positions=positions,
                )
                output = set_output_from_stacked_z(
                    adapter,
                    layer=config.mediator_layer,
                    heads=config.heads,
                    stacked_z=edge_z,
                )
                projection = float(torch.dot(output, global_axis["output_unit"]))
                partition_rows.append(
                    {
                        "schema_version": (
                            "realistic_niah_v4_4_4_read_write_natural_partition_row_v1"
                        ),
                        "design_hash": design_hash,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "position_group": name,
                        "position_count": len(positions),
                        "attention_mass_across_selected_heads": _attention_mass(
                            bundle,
                            layer=config.mediator_layer,
                            heads=config.heads,
                            positions=positions,
                        ),
                        "global_axis_contribution_projection": projection,
                        "global_axis_contribution_coefficient": projection
                        / float(global_axis["output_step_norm"]),
                        "edge_output_norm": float(torch.linalg.vector_norm(output)),
                        "baseline_predicted_count": int(
                            baseline["predicted_count_among_candidates"]
                        ),
                        "baseline_correct": bool(
                            baseline["predicted_count_among_candidates"] == count
                        ),
                    }
                )
            residual_samples.append(
                torch.stack([residuals[layer] for layer in config.downstream_layers])
            )
            sample_metadata.append(
                (
                    int(seed),
                    int(count),
                    bool(baseline["predicted_count_among_candidates"] == count),
                )
            )
            del bundle, residuals
        del encodings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - started
        remaining = elapsed / seed_offset * (
            len(config.discovery_seeds) - seed_offset
        )
        print(
            f"[v4.4.4 read/write discovery] seed={seed} "
            f"complete={seed_offset}/{len(config.discovery_seeds)} "
            f"eta_seconds={remaining:.1f}",
            flush=True,
        )
    states = torch.stack(residual_samples)
    labels = [item[1] for item in sample_metadata]
    fit_mask = torch.as_tensor(
        [count in config.fit_counts for count in labels], dtype=torch.bool
    )
    fit_labels = [count for count in labels if count in config.fit_counts]
    intercepts: dict[int, torch.Tensor] = {}
    steps: dict[int, torch.Tensor] = {}
    for layer_offset, layer in enumerate(config.downstream_layers):
        intercept, step = fit_count_intercept_and_step(
            states[fit_mask, layer_offset], fit_labels
        )
        intercepts[int(layer)] = intercept
        steps[int(layer)] = step
    axis_rows: list[dict[str, Any]] = []
    for sample_offset, (seed, count, correct) in enumerate(sample_metadata):
        for layer_offset, layer in enumerate(config.downstream_layers):
            coefficient = residual_axis_coefficient(
                states[sample_offset, layer_offset],
                intercept=intercepts[layer],
                step=steps[layer],
            )
            axis_rows.append(
                {
                    "schema_version": "realistic_niah_v4_4_4_read_write_axis_row_v1",
                    "design_hash": design_hash,
                    "seed": seed,
                    "gold_count": count,
                    "count_partition": (
                        "fit" if count in config.fit_counts else "heldout"
                    ),
                    "baseline_correct": correct,
                    "layer": int(layer),
                    "natural_residual_count_coefficient": coefficient,
                    "natural_residual_step_norm": float(
                        torch.linalg.vector_norm(steps[layer])
                    ),
                }
            )
    atomic_csv_gzip(pd.DataFrame(partition_rows), root / "natural_partition_detail.csv.gz")
    atomic_csv_gzip(pd.DataFrame(axis_rows), root / "downstream_axis_detail.csv.gz")
    atomic_torch_save(
        {
            "schema_version": "realistic_niah_v4_4_4_read_write_artifacts_v1",
            "design_hash": design_hash,
            "base_selection_sha256": selection["selection_sha256"],
            "global_z_count_steps": global_steps,
            "global_output_step": global_axis["output_step"],
            "global_output_step_norm": global_axis["output_step_norm"],
            "downstream_intercepts": intercepts,
            "downstream_steps": steps,
        },
        root / "artifacts.pt",
    )
    payload = {
        "schema_version": "realistic_niah_v4_4_4_read_write_discovery_complete_v1",
        "design_hash": design_hash,
        "natural_partition_rows": len(partition_rows),
        "downstream_axis_rows": len(axis_rows),
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=DISCOVERY_STAGE,
        state="COMPLETE",
        detail=payload,
    )
    return payload


def run_smoke_stage(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    run_root: str | Path,
    base_config: V444Config,
    config: V444ReadWriteConfig,
    v4_config: V4Config,
    resume: bool,
) -> dict[str, Any]:
    root = stage_root(run_root, config.model_label, SMOKE_STAGE)
    complete = root / "complete.json"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Read/write smoke already complete: {complete}")
        return json.loads(complete.read_text(encoding="utf-8"))
    artifacts = _load_discovery_artifacts(run_root, config)
    global_steps = artifacts["global_z_count_steps"].detach().float().cpu()
    seed = config.evaluation_seeds[0]
    receiver_count, donor_count = config.donor_pairs[0]
    stimuli = _load_v4_stimulus_map(
        run_root, seeds=(seed,), counts=(receiver_count, donor_count)
    )
    encodings = _render_encodings(
        stimuli,
        tokenizer=tokenizer,
        model_label=config.model_label,
        v4_config=v4_config,
        seed=seed,
        counts=(receiver_count, donor_count),
    )
    receiver, _receiver_trace = capture_query_bundle_and_trace(
        model,
        adapter,
        encodings[receiver_count],
        mediator_layer=config.mediator_layer,
        trace_layers=config.downstream_layers,
        cache_logit_tolerance=base_config.attention_cache_logit_tolerance,
    )
    donor, _donor_trace = capture_query_bundle_and_trace(
        model,
        adapter,
        encodings[donor_count],
        mediator_layer=config.mediator_layer,
        trace_layers=config.downstream_layers,
        cache_logit_tolerance=base_config.attention_cache_logit_tolerance,
    )
    positions = stable_position_partition(
        encodings[receiver_count], tail_width=config.tail_width
    )["all_positions"]
    decomposition = shapley_read_decomposition(
        receiver,
        donor,
        adapter,
        layer=config.mediator_layer,
        heads=config.heads,
        positions=positions,
        closure_relative_tolerance=config.closure_relative_tolerance,
        anchor_to_captured_endpoints=True,
    )
    exact = full_set_delta_from_bundles(
        receiver,
        donor,
        adapter,
        layer=config.mediator_layer,
        heads=config.heads,
    )
    full_delta = decomposition["full"]
    if not isinstance(full_delta, torch.Tensor):
        raise TypeError("Read/write full component is not a tensor")
    full_reconstruction = float(torch.linalg.vector_norm(full_delta - exact))
    reference = float(torch.linalg.vector_norm(exact))
    full_relative = full_reconstruction / max(reference, 1e-12)
    endpoint_relative = max(
        float(decomposition["receiver_endpoint_reconstruction_relative_l2"]),
        float(decomposition["donor_endpoint_reconstruction_relative_l2"]),
    )
    if endpoint_relative > config.edge_reconstruction_relative_tolerance:
        raise RuntimeError("Eager attention-value endpoints miss captured pre-O Z")
    if full_relative > config.closure_relative_tolerance:
        raise RuntimeError("Anchored all-position decomposition misses donor Z")
    traced = run_with_set_z_deltas_and_trace(
        model,
        adapter,
        encodings[receiver_count],
        layer=config.mediator_layer,
        deltas=stacked_delta_mapping(config.heads, full_delta),
        trace_layers=config.downstream_layers,
    )
    diagnostics = read_component_output_diagnostics(
        adapter,
        layer=config.mediator_layer,
        heads=config.heads,
        components=decomposition,
        global_z_count_steps=global_steps,
        count_gap=donor_count - receiver_count,
    )
    if traced.causal_output.attention_output is None:
        raise RuntimeError("Read/write smoke captured no post-O output")
    realized = (
        traced.causal_output.attention_output
        - receiver.attention_output_by_layer[config.mediator_layer]
    )
    predicted = set_output_from_stacked_z(
        adapter,
        layer=config.mediator_layer,
        heads=config.heads,
        stacked_z=full_delta,
    )
    post_o_max_abs = float(torch.max(torch.abs(realized - predicted)))
    if post_o_max_abs > config.pre_o_output_equivalence_tolerance:
        raise RuntimeError("Read/write pre-O intervention does not realize its W_O delta")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_read_write_smoke_complete_v1",
        "seed": int(seed),
        "receiver_count": int(receiver_count),
        "donor_count": int(donor_count),
        "shapley_closure_relative_l2": float(
            decomposition["closure_relative_l2"]
        ),
        "full_z_reconstruction_relative_l2": full_relative,
        "max_eager_endpoint_reconstruction_relative_l2": endpoint_relative,
        "pre_o_to_post_o_max_abs_delta": post_o_max_abs,
        **diagnostics,
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=SMOKE_STAGE,
        state="COMPLETE",
        detail=payload,
    )
    print(
        "[v4.4.4 read/write smoke] PASS closure={:.3g} full_z={:.3g} post_o={:.3g}".format(
            float(decomposition["closure_relative_l2"]),
            full_relative,
            post_o_max_abs,
        ),
        flush=True,
    )
    return payload


def _valid_shard(path: Path, *, design_hash: str, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return bool(
        len(frame) == int(expected_rows)
        and set(frame["design_hash"].astype(str)) == {str(design_hash)}
    )


def run_evaluation_stage(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    run_root: str | Path,
    base_config: V444Config,
    config: V444ReadWriteConfig,
    v4_config: V4Config,
    resume: bool,
) -> dict[str, pd.DataFrame]:
    root = stage_root(run_root, config.model_label, EVALUATION_STAGE)
    complete = root / "complete.json"
    merged_paths = {
        "natural": root / "natural_detail.csv.gz",
        "mechanical": root / "read_mechanical_detail.csv.gz",
        "read": root / "read_causal_detail.csv.gz",
        "read_trace": root / "read_trace_detail.csv.gz",
        "write": root / "write_trace_detail.csv.gz",
    }
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Read/write evaluation already complete: {complete}")
        return {name: pd.read_csv(path) for name, path in merged_paths.items()}
    artifacts = _load_discovery_artifacts(run_root, config)
    global_steps = artifacts["global_z_count_steps"].detach().float().cpu()
    downstream_intercepts = {
        int(layer): tensor.detach().float().cpu()
        for layer, tensor in artifacts["downstream_intercepts"].items()
    }
    downstream_steps = {
        int(layer): tensor.detach().float().cpu()
        for layer, tensor in artifacts["downstream_steps"].items()
    }
    design = {
        "config": config.to_dict(),
        "discovery_design_hash": artifacts["design_hash"],
        "base_selection_sha256": artifacts["base_selection_sha256"],
        "attention_cache_policy": {
            "mode": "diagnostic_only_nonfatal_v1",
            "reference_tolerance": float(base_config.attention_cache_logit_tolerance),
            "hard_gate": "all_key_eager_endpoint_reconstruction_relative_l2",
        },
    }
    design_hash = _stable_hash(design)
    atomic_json(
        root / "design.json",
        {
            "schema_version": "realistic_niah_v4_4_4_read_write_evaluation_design_v1",
            "design_hash": design_hash,
            **design,
        },
    )
    shard_roots = {name: root / f"{name}_shards" for name in merged_paths}
    for directory in shard_roots.values():
        directory.mkdir(parents=True, exist_ok=True)
    expected = {
        "natural": len(config.counts) * len(config.downstream_layers),
        "mechanical": len(config.donor_pairs) * 5,
        "read": len(config.donor_pairs) * 7,
        "read_trace": (
            len(config.donor_pairs) * len(READ_COMPONENT_NAMES) * len(config.downstream_layers)
        ),
        "write": len(config.write_counts) * 4 * len(config.downstream_layers),
    }
    write_axis = natural_axis_diagnostics(
        adapter,
        layer=config.mediator_layer,
        heads=config.heads,
        z_count_steps=global_steps,
    )
    orthogonal_write, orthogonal_write_output = _orthogonal_equal_output_norm_delta(
        adapter,
        layer=config.mediator_layer,
        heads=config.heads,
        z_count_steps=global_steps,
        target_output_norm=float(write_axis["output_step_norm"]),
        label="v445:write:orthogonal-control",
    )
    orthogonal_norm = float(torch.linalg.vector_norm(orthogonal_write_output))
    orthogonal_cosine = float(
        torch.dot(orthogonal_write_output, write_axis["output_unit"])
        / max(orthogonal_norm, 1e-12)
    )
    if abs(orthogonal_cosine) > 1e-4:
        raise RuntimeError("Read/write control is not output-orthogonal")
    stimuli = _load_v4_stimulus_map(
        run_root, seeds=config.evaluation_seeds, counts=config.counts
    )
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=EVALUATION_STAGE,
        state="RUNNING",
        detail={"completed_seeds": 0, "total_seeds": len(config.evaluation_seeds)},
    )
    started = time.monotonic()
    for seed_offset, seed in enumerate(config.evaluation_seeds, start=1):
        paths = {
            name: directory / f"seed{seed}.csv.gz"
            for name, directory in shard_roots.items()
        }
        if resume and all(
            _valid_shard(paths[name], design_hash=design_hash, expected_rows=expected[name])
            for name in paths
        ):
            print(
                f"[v4.4.4 read/write evaluation] seed={seed} resume-skip "
                f"({seed_offset}/{len(config.evaluation_seeds)})",
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
        bundles: dict[int, Any] = {}
        baseline_traces: dict[int, dict[int, torch.Tensor]] = {}
        natural_rows: list[dict[str, Any]] = []
        for count in config.counts:
            bundle, trace = capture_query_bundle_and_trace(
                model,
                adapter,
                encodings[count],
                mediator_layer=config.mediator_layer,
                trace_layers=config.downstream_layers,
                # Later layers can amplify small eager/cache differences.  Keep
                # both logit diagnostics below, while the actual selected-layer
                # alpha-V -> pre-O endpoint reconstruction remains the hard gate.
                cache_logit_tolerance=math.inf,
            )
            bundles[count] = bundle
            baseline_traces[count] = trace
            cache_diagnostics = _attention_cache_diagnostics(
                bundle,
                reference_tolerance=base_config.attention_cache_logit_tolerance,
            )
            baseline = candidate_sequence_metrics(
                bundle.candidate_log_scores, encodings[count]
            )
            for layer in config.downstream_layers:
                natural_rows.append(
                    {
                        "schema_version": "realistic_niah_v4_4_4_read_write_natural_row_v1",
                        "design_hash": design_hash,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "layer": int(layer),
                        "baseline_predicted_count": int(
                            baseline["predicted_count_among_candidates"]
                        ),
                        "baseline_correct": bool(
                            baseline["predicted_count_among_candidates"] == count
                        ),
                        "baseline_expected_count": float(baseline["expected_count"]),
                        "baseline_correct_margin": float(
                            baseline["correct_count_margin"]
                        ),
                        **cache_diagnostics,
                        "natural_residual_count_coefficient": residual_axis_coefficient(
                            trace[layer],
                            intercept=downstream_intercepts[layer],
                            step=downstream_steps[layer],
                        ),
                    }
                )
        mechanical_rows: list[dict[str, Any]] = []
        read_rows: list[dict[str, Any]] = []
        read_trace_rows: list[dict[str, Any]] = []
        for receiver_count, donor_count in config.donor_pairs:
            receiver = bundles[receiver_count]
            donor = bundles[donor_count]
            receiver_encoding = encodings[receiver_count]
            receiver_groups = stable_position_partition(
                receiver_encoding, tail_width=config.tail_width
            )
            donor_groups = stable_position_partition(
                encodings[donor_count], tail_width=config.tail_width
            )
            if receiver_groups != donor_groups:
                raise RuntimeError("Read/write receiver/donor position partitions drift")
            gap = int(donor_count) - int(receiver_count)
            decompositions: dict[str, dict[str, torch.Tensor | float]] = {}
            for group_name, positions in receiver_groups.items():
                decomposition = shapley_read_decomposition(
                    receiver,
                    donor,
                    adapter,
                    layer=config.mediator_layer,
                    heads=config.heads,
                    positions=positions,
                    closure_relative_tolerance=config.closure_relative_tolerance,
                    anchor_to_captured_endpoints=(group_name == "all_positions"),
                )
                decompositions[group_name] = decomposition
                diagnostics = read_component_output_diagnostics(
                    adapter,
                    layer=config.mediator_layer,
                    heads=config.heads,
                    components=decomposition,
                    global_z_count_steps=global_steps,
                    count_gap=gap,
                )
                mechanical_rows.append(
                    {
                        "schema_version": (
                            "realistic_niah_v4_4_4_read_write_mechanical_row_v1"
                        ),
                        "design_hash": design_hash,
                        "seed": int(seed),
                        "receiver_count": int(receiver_count),
                        "donor_count": int(donor_count),
                        "position_group": group_name,
                        "position_count": len(positions),
                        "receiver_attention_mass": _attention_mass(
                            receiver,
                            layer=config.mediator_layer,
                            heads=config.heads,
                            positions=positions,
                        ),
                        "donor_attention_mass": _attention_mass(
                            donor,
                            layer=config.mediator_layer,
                            heads=config.heads,
                            positions=positions,
                        ),
                        **diagnostics,
                    }
                )
            all_components = decompositions["all_positions"]
            exact_full = full_set_delta_from_bundles(
                receiver,
                donor,
                adapter,
                layer=config.mediator_layer,
                heads=config.heads,
            )
            if not isinstance(all_components["full"], torch.Tensor):
                raise TypeError("Read/write full delta is not a tensor")
            exact_error = float(
                torch.linalg.vector_norm(all_components["full"] - exact_full)
            ) / max(float(torch.linalg.vector_norm(exact_full)), 1e-12)
            endpoint_relative = max(
                float(
                    all_components[
                        "receiver_endpoint_reconstruction_relative_l2"
                    ]
                ),
                float(
                    all_components[
                        "donor_endpoint_reconstruction_relative_l2"
                    ]
                ),
            )
            if endpoint_relative > config.edge_reconstruction_relative_tolerance:
                raise RuntimeError(
                    "Read/write eager attention-value endpoint reconstruction drifted"
                )
            if exact_error > config.closure_relative_tolerance:
                raise RuntimeError(
                    "Read/write anchored full patch does not equal donor Z"
                )
            for component_name in READ_COMPONENT_NAMES:
                component_delta = all_components[component_name]
                if not isinstance(component_delta, torch.Tensor):
                    raise TypeError("Read/write component is not a tensor")
                traced = run_with_set_z_deltas_and_trace(
                    model,
                    adapter,
                    receiver_encoding,
                    layer=config.mediator_layer,
                    deltas=stacked_delta_mapping(config.heads, component_delta),
                    trace_layers=config.downstream_layers,
                )
                component_diag = read_component_output_diagnostics(
                    adapter,
                    layer=config.mediator_layer,
                    heads=config.heads,
                    components=all_components,
                    global_z_count_steps=global_steps,
                    count_gap=gap,
                )
                read_rows.append(
                    {
                        "schema_version": "realistic_niah_v4_4_4_read_write_causal_row_v1",
                        "design_hash": design_hash,
                        "seed": int(seed),
                        "receiver_count": int(receiver_count),
                        "donor_count": int(donor_count),
                        "component": component_name,
                        "intervention": "component_patch",
                        **intervention_metrics(
                            baseline_output=receiver,
                            intervened_output=traced.causal_output,
                            encoding=receiver_encoding,
                            donor_count=donor_count,
                        ),
                        "component_output_norm": component_diag[
                            f"{component_name}_output_norm"
                        ],
                        "component_global_axis_projection": component_diag[
                            f"{component_name}_global_axis_projection"
                        ],
                        "component_global_axis_cosine": component_diag[
                            f"{component_name}_global_axis_cosine"
                        ],
                        "component_mechanical_transport": component_diag[
                            f"{component_name}_mechanical_transport"
                        ],
                    }
                )
                for layer in config.downstream_layers:
                    read_trace_rows.append(
                        {
                            "schema_version": (
                                "realistic_niah_v4_4_4_read_write_trace_row_v1"
                            ),
                            "design_hash": design_hash,
                            "seed": int(seed),
                            "receiver_count": int(receiver_count),
                            "donor_count": int(donor_count),
                            "component": component_name,
                            "layer": int(layer),
                            **_axis_projection(
                                traced.residual_by_layer[layer]
                                - baseline_traces[receiver_count][layer],
                                downstream_steps[layer],
                            ),
                        }
                    )
                if component_name == "full":
                    continue
                block, control, _block_diagnostics = natural_axis_block_for_patch_delta(
                    adapter,
                    layer=config.mediator_layer,
                    heads=config.heads,
                    patch_delta_z=component_delta,
                    global_z_count_steps=global_steps,
                    orthogonal_label=(
                        f"v445:read:{component_name}:{seed}:"
                        f"{receiver_count}:{donor_count}"
                    ),
                )
                for intervention, delta in (
                    ("component_patch_plus_natural_axis_block", component_delta + block),
                    ("component_patch_plus_orthogonal_control", component_delta + control),
                ):
                    output = run_with_set_z_deltas(
                        model,
                        adapter,
                        receiver_encoding,
                        layer=config.mediator_layer,
                        deltas=stacked_delta_mapping(config.heads, delta),
                    )
                    read_rows.append(
                        {
                            "schema_version": (
                                "realistic_niah_v4_4_4_read_write_causal_row_v1"
                            ),
                            "design_hash": design_hash,
                            "seed": int(seed),
                            "receiver_count": int(receiver_count),
                            "donor_count": int(donor_count),
                            "component": component_name,
                            "intervention": intervention,
                            **intervention_metrics(
                                baseline_output=receiver,
                                intervened_output=output,
                                encoding=receiver_encoding,
                                donor_count=donor_count,
                            ),
                            "component_output_norm": component_diag[
                                f"{component_name}_output_norm"
                            ],
                            "component_global_axis_projection": component_diag[
                                f"{component_name}_global_axis_projection"
                            ],
                            "component_global_axis_cosine": component_diag[
                                f"{component_name}_global_axis_cosine"
                            ],
                            "component_mechanical_transport": component_diag[
                                f"{component_name}_mechanical_transport"
                            ],
                        }
                    )
        write_rows: list[dict[str, Any]] = []
        for count in config.write_counts:
            encoding = encodings[count]
            baseline = bundles[count]
            baseline_trace = baseline_traces[count]
            for intervention, stacked_delta, signed_beta in (
                ("natural_plus", global_steps * config.write_beta, config.write_beta),
                ("natural_minus", -global_steps * config.write_beta, -config.write_beta),
                (
                    "orthogonal_plus",
                    orthogonal_write * config.write_beta,
                    config.write_beta,
                ),
                (
                    "orthogonal_minus",
                    -orthogonal_write * config.write_beta,
                    -config.write_beta,
                ),
            ):
                traced = run_with_set_z_deltas_and_trace(
                    model,
                    adapter,
                    encoding,
                    layer=config.mediator_layer,
                    deltas=stacked_delta_mapping(config.heads, stacked_delta),
                    trace_layers=config.downstream_layers,
                )
                behavior = intervention_metrics(
                    baseline_output=baseline,
                    intervened_output=traced.causal_output,
                    encoding=encoding,
                )
                for layer in config.downstream_layers:
                    write_rows.append(
                        {
                            "schema_version": "realistic_niah_v4_4_4_read_write_row_v1",
                            "design_hash": design_hash,
                            "seed": int(seed),
                            "gold_count": int(count),
                            "intervention": intervention,
                            "signed_beta": float(signed_beta),
                            "layer": int(layer),
                            "write_output_step_norm": float(
                                write_axis["output_step_norm"]
                            ),
                            "orthogonal_control_output_norm": orthogonal_norm,
                            "orthogonal_control_axis_cosine": orthogonal_cosine,
                            **behavior,
                            **_axis_projection(
                                traced.residual_by_layer[layer]
                                - baseline_trace[layer],
                                downstream_steps[layer],
                            ),
                        }
                    )
        frames = {
            "natural": pd.DataFrame(natural_rows),
            "mechanical": pd.DataFrame(mechanical_rows),
            "read": pd.DataFrame(read_rows),
            "read_trace": pd.DataFrame(read_trace_rows),
            "write": pd.DataFrame(write_rows),
        }
        for name, frame in frames.items():
            if len(frame) != expected[name]:
                raise RuntimeError(
                    f"Read/write {name} seed shard has {len(frame)} rows; "
                    f"expected {expected[name]}"
                )
            atomic_csv_gzip(frame, paths[name])
        del bundles, baseline_traces, encodings, frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - started
        remaining = elapsed / seed_offset * (
            len(config.evaluation_seeds) - seed_offset
        )
        write_stage_status(
            run_root,
            model_label=config.model_label,
            stage=EVALUATION_STAGE,
            state="RUNNING",
            detail={
                "completed_seeds": seed_offset,
                "total_seeds": len(config.evaluation_seeds),
                "eta_seconds": remaining,
            },
        )
        print(
            f"[v4.4.4 read/write evaluation] seed={seed} "
            f"complete={seed_offset}/{len(config.evaluation_seeds)} "
            f"eta_seconds={remaining:.1f}",
            flush=True,
        )
    result: dict[str, pd.DataFrame] = {}
    for name, directory in shard_roots.items():
        files = sorted(directory.glob("seed*.csv.gz"))
        if len(files) != len(config.evaluation_seeds):
            raise RuntimeError(f"Read/write {name} seed shard grid is incomplete")
        frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        expected_total = expected[name] * len(config.evaluation_seeds)
        if len(frame) != expected_total:
            raise RuntimeError(f"Read/write {name} merged row count is incomplete")
        atomic_csv_gzip(frame, merged_paths[name])
        result[name] = frame
    payload = {
        "schema_version": "realistic_niah_v4_4_4_read_write_evaluation_complete_v1",
        "design_hash": design_hash,
        "row_counts": {name: len(frame) for name, frame in result.items()},
        "attention_cache_diagnostics": {
            "policy": "diagnostic_only; eager endpoint reconstruction is the hard gate",
            "reference_tolerance": float(base_config.attention_cache_logit_tolerance),
            "max_raw_candidate_logit_delta": float(
                result["natural"][
                    "attention_cache_candidate_logit_max_abs_delta"
                ].max()
            ),
            "max_centered_candidate_logit_delta": float(
                result["natural"][
                    "attention_cache_candidate_centered_logit_max_abs_delta"
                ].max()
            ),
            "reference_tolerance_exceedance_samples": int(
                result["natural"]
                .drop_duplicates(["seed", "gold_count"])[
                    "attention_cache_reference_tolerance_exceeded"
                ]
                .astype(bool)
                .sum()
            ),
            "sample_count": int(
                len(result["natural"].drop_duplicates(["seed", "gold_count"]))
            ),
        },
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=EVALUATION_STAGE,
        state="COMPLETE",
        detail=payload,
    )
    return result


def run_model_campaign(
    *,
    run_root: str | Path,
    base_config: V444Config,
    config: V444ReadWriteConfig,
    v4_config_path: str | Path,
    cache_dir: str | Path,
    device_map: str,
    resume: bool,
) -> dict[str, Any]:
    config.validate_against_base(base_config)
    base_v4 = V4Config.from_json(v4_config_path)
    v4_config = v444_v4_config(base_v4, base_config)
    model, tokenizer, adapter = _load_model(
        config.model_label,
        config=_direction_config(base_config),
        cache_dir=cache_dir,
        device_map=device_map,
    )
    model.eval()
    started = time.monotonic()
    discovery = run_discovery_stage(
        model,
        tokenizer,
        adapter,
        run_root=run_root,
        base_config=base_config,
        config=config,
        v4_config=v4_config,
        resume=resume,
    )
    smoke = run_smoke_stage(
        model,
        tokenizer,
        adapter,
        run_root=run_root,
        base_config=base_config,
        config=config,
        v4_config=v4_config,
        resume=resume,
    )
    evaluation = run_evaluation_stage(
        model,
        tokenizer,
        adapter,
        run_root=run_root,
        base_config=base_config,
        config=config,
        v4_config=v4_config,
        resume=resume,
    )
    payload = {
        "schema_version": "realistic_niah_v4_4_4_read_write_model_complete_v1",
        "discovery": discovery,
        "smoke": smoke,
        "evaluation_rows": {
            name: len(frame) for name, frame in evaluation.items()
        },
        "runtime_seconds": time.monotonic() - started,
        "completed_unix": time.time(),
    }
    root = stage_root(run_root, config.model_label, "read_write_model")
    atomic_json(root / "complete.json", payload)
    return payload
