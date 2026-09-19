from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from realistic_niah_v4.modeling import DecoderAdapter
from realistic_niah_v4.spec import V4Config
from realistic_niah_v4_4_3.interventions import (
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
from realistic_niah_v4_4_3.spec import V443Config

from .gemma_cross_layer import frozen_selection, unique_sites
from .gemma_cross_layer_spec import FrozenSite
from .gemma_residual import (
    ResidualBundle,
    capture_source_and_residual_bundle,
    equal_norm_orthogonal,
    fit_residual_intercept_and_step,
    residual_component,
    residual_projection_coefficient,
    residual_set_label,
    run_source_patch_with_residual_delta,
    source_replacements,
)
from .gemma_residual_spec import GemmaResidualConfig
from .pipeline import _stable_hash, v444_v4_config


DISCOVERY_STAGE = "residual_path_discovery"
SMOKE_STAGE = "residual_path_smoke"
CONFIRMATION_STAGE = "residual_path_confirmation"


def _direction_config(config: GemmaResidualConfig) -> V443Config:
    result = V443Config(
        model_labels=(config.model_label,),
        discovery_seeds=config.discovery_seeds,
        screen_seeds=(1254,),
        confirmation_seeds=(1255,),
        fit_counts=config.fit_counts,
        heldout_counts=config.heldout_counts,
        target_output_layers_gemma=tuple(
            sorted({site.layer for site in config.candidate_sites})
        ),
        model_torch_dtype=config.model_torch_dtype,
        attention_prefix_backend=config.attention_prefix_backend,
        attention_cache_logit_tolerance=config.attention_cache_logit_tolerance,
    )
    result.validate()
    return result


def _entry_sites(entry: Mapping[str, Any]) -> tuple[FrozenSite, ...]:
    return tuple(FrozenSite.from_value(value) for value in entry["sites"])


def _entries(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(selection["candidate"]), *map(dict, selection["matched_controls"])]


def _valid_shard(path: Path, *, design_hash: str, rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return bool(
        len(frame) == int(rows)
        and "design_hash" in frame
        and set(frame["design_hash"].astype(str)) == {str(design_hash)}
    )


def _capture(
    model: Any,
    adapter: DecoderAdapter,
    encoding: Any,
    *,
    source_layers: Sequence[int],
    residual_layers: Sequence[int],
    config: GemmaResidualConfig,
) -> ResidualBundle:
    return capture_source_and_residual_bundle(
        model,
        adapter,
        encoding,
        source_layers=source_layers,
        residual_layers=residual_layers,
        cache_logit_tolerance=config.attention_cache_logit_tolerance,
    )


def _candidate_log_odds_gain(
    baseline: ResidualBundle,
    output: Any,
    *,
    receiver_count: int,
    donor_count: int,
) -> float:
    before = baseline.query.candidate_log_scores
    after = output.causal_output.candidate_log_scores
    return float(
        (after[int(donor_count)] - after[int(receiver_count)])
        - (before[int(donor_count)] - before[int(receiver_count)])
    )


def _zero_source_replacements(
    adapter: DecoderAdapter, sites: Sequence[FrozenSite]
) -> dict[tuple[int, int], torch.Tensor]:
    """Return a deterministic zero-z ablation for a frozen source bank."""

    return {
        (int(site.layer), int(site.head)): torch.zeros(
            int(adapter.head_dims[int(site.layer)]), dtype=torch.float32
        )
        for site in sites
    }


@torch.inference_mode()
def run_discovery_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    run_root: str | Path,
    config: GemmaResidualConfig,
    v4_config: V4Config,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = stage_root(run_root, config.model_label, DISCOVERY_STAGE)
    complete = root / "complete.json"
    selection_path = root / "selection.json"
    artifact_path = root / "artifacts.pt"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Residual discovery already complete: {complete}")
        return (
            json.loads(selection_path.read_text(encoding="utf-8")),
            torch.load(artifact_path, map_location="cpu", weights_only=False),
        )
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=DISCOVERY_STAGE,
        state="RUNNING",
    )
    selection = frozen_selection(config.candidate_sites, config.matched_control_sets)
    selection["selection_source"] = config.selection_source
    selection["selection_status"] = config.selection_status
    selection["mechanism_variant"] = config.mechanism_variant
    selection["matched_control_sampling_seed"] = config.matched_control_sampling_seed
    selection["residual_layer_selection_uses_confirmation_outcomes"] = False
    selection["selection_sha256"] = _stable_hash(
        {key: value for key, value in selection.items() if key != "selection_sha256"}
    )
    source_layers = tuple(sorted({site.layer for site in config.candidate_sites}))
    residual_layers = (*config.candidate_mediator_layers, config.terminal_trace_layer)
    stimuli = _load_v4_stimulus_map(
        run_root, seeds=config.discovery_seeds, counts=config.counts
    )
    residual_values = {layer: [] for layer in residual_layers}
    residual_counts: list[int] = []
    patch_changes = {layer: [] for layer in config.candidate_mediator_layers}
    discovery_rows: list[dict[str, Any]] = []
    for seed in config.discovery_seeds:
        encodings = _render_encodings(
            stimuli,
            tokenizer=tokenizer,
            model_label=config.model_label,
            v4_config=v4_config,
            seed=seed,
            counts=config.counts,
        )
        bundles = {
            count: _capture(
                model,
                adapter,
                encodings[count],
                source_layers=source_layers,
                residual_layers=residual_layers,
                config=config,
            )
            for count in config.counts
        }
        for count in config.counts:
            residual_counts.append(int(count))
            for layer in residual_layers:
                residual_values[layer].append(bundles[count].residual_by_layer[layer])
        for receiver_count, donor_count in config.donor_pairs:
            patch = run_source_patch_with_residual_delta(
                model,
                adapter,
                encodings[receiver_count],
                replacements=source_replacements(
                    bundles[donor_count].query, adapter, config.candidate_sites
                ),
                mediator_layer=config.candidate_mediator_layers[0],
                terminal_layer=config.terminal_trace_layer,
            )
            # One run only exposes the first mediator.  Additional layers are
            # captured below with a no-edit source-patch trace helper call.
            changes = {
                config.candidate_mediator_layers[0]: (
                    patch.mediator_before
                    - bundles[receiver_count].residual_by_layer[
                        config.candidate_mediator_layers[0]
                    ]
                )
            }
            for layer in config.candidate_mediator_layers[1:]:
                traced = run_source_patch_with_residual_delta(
                    model,
                    adapter,
                    encodings[receiver_count],
                    replacements=source_replacements(
                        bundles[donor_count].query, adapter, config.candidate_sites
                    ),
                    mediator_layer=layer,
                    terminal_layer=config.terminal_trace_layer,
                )
                changes[layer] = (
                    traced.mediator_before
                    - bundles[receiver_count].residual_by_layer[layer]
                )
            for layer, delta in changes.items():
                patch_changes[layer].append(delta)
                discovery_rows.append(
                    {
                        "seed": int(seed),
                        "receiver_count": int(receiver_count),
                        "donor_count": int(donor_count),
                        "layer": int(layer),
                        "induced_residual_norm": float(torch.linalg.vector_norm(delta)),
                    }
                )
        print(f"[residual discovery] seed={seed} complete", flush=True)
        del encodings, bundles
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    axes: dict[str, dict[str, torch.Tensor]] = {}
    layer_scores: list[dict[str, Any]] = []
    for layer in residual_layers:
        intercept, step = fit_residual_intercept_and_step(
            residual_values[layer], residual_counts
        )
        axes[str(layer)] = {"intercept": intercept, "step": step}
        if layer in config.candidate_mediator_layers:
            unit = step / torch.linalg.vector_norm(step)
            values = [float(torch.dot(delta, unit)) for delta in patch_changes[layer]]
            layer_scores.append(
                {
                    "layer": int(layer),
                    "mean_aligned_induced_norm": float(sum(values) / len(values)),
                    "positive_fraction": float(
                        sum(value > 0 for value in values) / len(values)
                    ),
                    "samples": len(values),
                }
            )
    winner = sorted(
        layer_scores,
        key=lambda row: (-float(row["mean_aligned_induced_norm"]), int(row["layer"])),
    )[0]
    selected_layer = int(winner["layer"])
    selection["residual_mediator"] = {
        "layer": selected_layer,
        "rule": "max discovery mean cos(delta,s_count)*norm(delta); layer tie-break",
        "scores": layer_scores,
    }
    artifacts = {
        "schema_version": "realistic_niah_v4_4_4_residual_artifacts_v1",
        "axes": axes,
        "selected_mediator_layer": selected_layer,
        "source_selection_sha256": selection["selection_sha256"],
        "natural_samples": len(residual_counts),
        "patch_samples_per_layer": len(patch_changes[selected_layer]),
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(selection_path, selection)
    atomic_torch_save(artifacts, artifact_path)
    atomic_csv_gzip(pd.DataFrame(discovery_rows), root / "discovery_patch_rows.csv.gz")
    atomic_json(
        complete,
        {
            "schema_version": "realistic_niah_v4_4_4_residual_discovery_complete_v1",
            "selected_mediator_layer": selected_layer,
            "layer_scores": layer_scores,
            "confirmation_outcomes_opened": False,
            "completed_unix": time.time(),
        },
    )
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=DISCOVERY_STAGE,
        state="COMPLETE",
        detail={"selected_mediator_layer": selected_layer},
    )
    return selection, artifacts


def _axis(artifacts: Mapping[str, Any], layer: int, name: str) -> torch.Tensor:
    return artifacts["axes"][str(int(layer))][name].detach().float().cpu()


def _conditions(
    *,
    induced: torch.Tensor,
    count_step: torch.Tensor,
    label: str,
) -> tuple[dict[str, torch.Tensor | None], dict[str, float]]:
    induced = induced.detach().float().cpu()
    exact_norm = float(torch.linalg.vector_norm(induced))
    parallel = residual_component(induced, count_step)
    parallel_norm = float(torch.linalg.vector_norm(parallel))
    exact_control = equal_norm_orthogonal(
        induced, norm=exact_norm, label=f"{label}:exact"
    )
    count_control = equal_norm_orthogonal(
        count_step, norm=parallel_norm, label=f"{label}:count"
    )
    exact_cosine = (
        0.0
        if exact_norm <= 1e-12
        else float(
            torch.dot(exact_control, induced)
            / max(float(torch.linalg.vector_norm(exact_control)) * exact_norm, 1e-12)
        )
    )
    count_cosine = (
        0.0
        if parallel_norm <= 1e-12
        else float(
            torch.dot(count_control, count_step)
            / max(
                float(torch.linalg.vector_norm(count_control))
                * float(torch.linalg.vector_norm(count_step)),
                1e-12,
            )
        )
    )
    return (
        {
            "source_patch": None,
            "source_patch_plus_exact_block": -induced,
            "source_patch_plus_exact_orthogonal": exact_control,
            "source_patch_plus_count_axis_block": -parallel,
            "source_patch_plus_count_axis_orthogonal": count_control,
        },
        {
            "induced_residual_norm": exact_norm,
            "count_component_norm": parallel_norm,
            "count_component_fraction": parallel_norm / max(exact_norm, 1e-12),
            "exact_control_cosine": exact_cosine,
            "count_control_cosine": count_cosine,
        },
    )


@torch.inference_mode()
def run_smoke_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    run_root: str | Path,
    config: GemmaResidualConfig,
    v4_config: V4Config,
    selection: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    root = stage_root(run_root, config.model_label, SMOKE_STAGE)
    complete = root / "complete.json"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Residual smoke already complete: {complete}")
        return json.loads(complete.read_text(encoding="utf-8"))
    seed = config.confirmation_seeds[0]
    receiver_count, donor_count = config.donor_pairs[0]
    layer = int(artifacts["selected_mediator_layer"])
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
    source_layers = tuple(sorted({site.layer for site in config.candidate_sites}))
    residual_layers = (layer, config.terminal_trace_layer)
    receiver = _capture(
        model,
        adapter,
        encodings[receiver_count],
        source_layers=source_layers,
        residual_layers=residual_layers,
        config=config,
    )
    donor = _capture(
        model,
        adapter,
        encodings[donor_count],
        source_layers=source_layers,
        residual_layers=residual_layers,
        config=config,
    )
    replacements = source_replacements(donor.query, adapter, config.candidate_sites)
    patch = run_source_patch_with_residual_delta(
        model,
        adapter,
        encodings[receiver_count],
        replacements=replacements,
        mediator_layer=layer,
        terminal_layer=config.terminal_trace_layer,
    )
    induced = patch.mediator_before - receiver.residual_by_layer[layer]
    conditions, diagnostics = _conditions(
        induced=induced,
        count_step=_axis(artifacts, layer, "step"),
        label=f"residual-smoke:{seed}:{receiver_count}:{donor_count}",
    )
    exact = run_source_patch_with_residual_delta(
        model,
        adapter,
        encodings[receiver_count],
        replacements=replacements,
        mediator_layer=layer,
        terminal_layer=config.terminal_trace_layer,
        residual_delta=conditions["source_patch_plus_exact_block"],
    )
    closure = float(
        torch.linalg.vector_norm(
            exact.mediator_after - receiver.residual_by_layer[layer]
        )
    ) / max(float(torch.linalg.vector_norm(induced)), 1e-12)
    if closure > config.residual_closure_relative_tolerance:
        raise RuntimeError(f"Residual exact-block closure failed: {closure}")
    if (
        abs(diagnostics["exact_control_cosine"])
        > config.control_orthogonality_tolerance
    ):
        raise RuntimeError("Residual exact control is not orthogonal")
    if (
        abs(diagnostics["count_control_cosine"])
        > config.control_orthogonality_tolerance
    ):
        raise RuntimeError("Residual count-axis control is not orthogonal")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_residual_smoke_complete_v1",
        "seed": int(seed),
        "selected_mediator_layer": layer,
        "exact_block_closure_relative_l2": closure,
        **diagnostics,
        "completed_unix": time.time(),
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(complete, payload)
    print(f"[residual smoke] PASS layer={layer} closure={closure:.4g}", flush=True)
    return payload


@torch.inference_mode()
def run_confirmation_stage(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    *,
    run_root: str | Path,
    config: GemmaResidualConfig,
    v4_config: V4Config,
    selection: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    resume: bool,
) -> pd.DataFrame:
    root = stage_root(run_root, config.model_label, CONFIRMATION_STAGE)
    complete = root / "complete.json"
    detail_path = root / "residual_detail.csv.gz"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Residual confirmation complete: {complete}")
        return pd.read_csv(detail_path)
    entries = _entries(selection)
    all_sites = unique_sites([_entry_sites(entry) for entry in entries])
    source_layers = tuple(sorted({site.layer for site in all_sites}))
    layer = int(artifacts["selected_mediator_layer"])
    residual_layers = (layer, config.terminal_trace_layer)
    design = {
        "config": config.to_dict(),
        "source_selection_sha256": selection["selection_sha256"],
        "selected_mediator_layer": layer,
        "residual_layer_scores": selection["residual_mediator"]["scores"],
        "entries": entries,
    }
    design_hash = _stable_hash(design)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        root / "design.json",
        {
            "schema_version": "realistic_niah_v4_4_4_residual_design_v1",
            "design_hash": design_hash,
            **design,
        },
    )
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    clean_shard_root = root / "clean_ablation_shards"
    clean_detail_path = root / "clean_ablation_detail.csv.gz"
    if config.require_clean_necessity:
        clean_shard_root.mkdir(parents=True, exist_ok=True)
    condition_count = 5
    expected_rows = len(entries) * len(config.donor_pairs) * condition_count
    expected_clean_rows = len(entries) * len(config.counts)
    stimuli = _load_v4_stimulus_map(
        run_root, seeds=config.confirmation_seeds, counts=config.counts
    )
    count_step = _axis(artifacts, layer, "step")
    terminal_step = _axis(artifacts, config.terminal_trace_layer, "step")
    started = time.monotonic()
    for seed_offset, seed in enumerate(config.confirmation_seeds, start=1):
        shard = shard_root / f"seed{seed}.csv.gz"
        clean_shard = clean_shard_root / f"seed{seed}.csv.gz"
        residual_valid = resume and _valid_shard(
            shard, design_hash=design_hash, rows=expected_rows
        )
        clean_valid = not config.require_clean_necessity or (
            resume
            and _valid_shard(
                clean_shard,
                design_hash=design_hash,
                rows=expected_clean_rows,
            )
        )
        if residual_valid and clean_valid:
            print(
                f"[residual confirmation] seed={seed} resume-skip "
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
            count: _capture(
                model,
                adapter,
                encodings[count],
                source_layers=source_layers,
                residual_layers=residual_layers,
                config=config,
            )
            for count in config.counts
        }
        if not residual_valid:
            rows: list[dict[str, Any]] = []
            for entry in entries:
                sites = _entry_sites(entry)
                set_label = residual_set_label(sites)
                for receiver_count, donor_count in config.donor_pairs:
                    receiver = bundles[receiver_count]
                    donor = bundles[donor_count]
                    replacements = source_replacements(donor.query, adapter, sites)
                    patch = run_source_patch_with_residual_delta(
                        model,
                        adapter,
                        encodings[receiver_count],
                        replacements=replacements,
                        mediator_layer=layer,
                        terminal_layer=config.terminal_trace_layer,
                    )
                    induced = patch.mediator_before - receiver.residual_by_layer[layer]
                    conditions, diagnostics = _conditions(
                        induced=induced,
                        count_step=count_step,
                        label=(
                            f"residual:{seed}:{entry['set_id']}:"
                            f"{receiver_count}:{donor_count}"
                        ),
                    )
                    outputs = {"source_patch": patch}
                    for name, delta in conditions.items():
                        if name == "source_patch":
                            continue
                        outputs[name] = run_source_patch_with_residual_delta(
                            model,
                            adapter,
                            encodings[receiver_count],
                            replacements=replacements,
                            mediator_layer=layer,
                            terminal_layer=config.terminal_trace_layer,
                            residual_delta=delta,
                        )
                    baseline_terminal = receiver.residual_by_layer[
                        config.terminal_trace_layer
                    ]
                    for name, output in outputs.items():
                        metrics = intervention_metrics(
                            baseline_output=receiver.query,
                            intervened_output=output.causal_output,
                            encoding=encodings[receiver_count],
                            donor_count=donor_count,
                        )
                        terminal_delta = output.terminal_state - baseline_terminal
                        closure = float(
                            torch.linalg.vector_norm(
                                output.mediator_after
                                - receiver.residual_by_layer[layer]
                            )
                        )
                        rows.append(
                            {
                                "schema_version": "realistic_niah_v4_4_4_residual_row_v1",
                                "design_hash": design_hash,
                                "seed": int(seed),
                                "receiver_count": int(receiver_count),
                                "donor_count": int(donor_count),
                                "semantic_count_shift": int(
                                    donor_count - receiver_count
                                ),
                                "set_id": str(entry["set_id"]),
                                "set_role": str(entry["set_role"]),
                                "sites": set_label,
                                "mediator_layer": layer,
                                "terminal_layer": int(config.terminal_trace_layer),
                                "condition": name,
                                "donor_log_odds_gain": _candidate_log_odds_gain(
                                    receiver,
                                    output,
                                    receiver_count=receiver_count,
                                    donor_count=donor_count,
                                ),
                                "terminal_count_adoption": (
                                    residual_projection_coefficient(
                                        terminal_delta, terminal_step
                                    )
                                    / float(donor_count - receiver_count)
                                ),
                                "mediator_distance_from_clean": closure,
                                **diagnostics,
                                **{
                                    key: value
                                    for key, value in metrics.items()
                                    if isinstance(value, (bool, int, float, str))
                                },
                            }
                        )
            frame = pd.DataFrame(rows)
            if len(frame) != expected_rows:
                raise RuntimeError(
                    f"Residual confirmation shard has {len(frame)} rows; "
                    f"expected {expected_rows}"
                )
            atomic_csv_gzip(frame, shard)

        if config.require_clean_necessity and not clean_valid:
            clean_rows: list[dict[str, Any]] = []
            for entry in entries:
                sites = _entry_sites(entry)
                replacements = _zero_source_replacements(adapter, sites)
                for count in config.counts:
                    baseline = bundles[count]
                    ablated = run_source_patch_with_residual_delta(
                        model,
                        adapter,
                        encodings[count],
                        replacements=replacements,
                        mediator_layer=layer,
                        terminal_layer=config.terminal_trace_layer,
                    )
                    metrics = intervention_metrics(
                        baseline_output=baseline.query,
                        intervened_output=ablated.causal_output,
                        encoding=encodings[count],
                    )
                    baseline_correct = int(metrics["baseline_predicted_count"] == count)
                    intervened_correct = int(
                        metrics["intervened_predicted_count"] == count
                    )
                    clean_rows.append(
                        {
                            "schema_version": (
                                "realistic_niah_v4_4_4_clean_bank_ablation_row_v1"
                            ),
                            "design_hash": design_hash,
                            "seed": int(seed),
                            "gold_count": int(count),
                            "set_id": str(entry["set_id"]),
                            "set_role": str(entry["set_role"]),
                            "sites": residual_set_label(sites),
                            "condition": "clean_zero_z_bank_ablation",
                            "baseline_correct": baseline_correct,
                            "intervened_correct": intervened_correct,
                            "clean_correct_failure": int(
                                baseline_correct and not intervened_correct
                            ),
                            **{
                                key: value
                                for key, value in metrics.items()
                                if isinstance(value, (bool, int, float, str))
                            },
                        }
                    )
            clean_frame = pd.DataFrame(clean_rows)
            if len(clean_frame) != expected_clean_rows:
                raise RuntimeError(
                    f"Clean ablation shard has {len(clean_frame)} rows; "
                    f"expected {expected_clean_rows}"
                )
            atomic_csv_gzip(clean_frame, clean_shard)
        elapsed = time.monotonic() - started
        remaining = (
            elapsed / seed_offset * (len(config.confirmation_seeds) - seed_offset)
        )
        print(
            f"[residual confirmation] seed={seed} "
            f"complete={seed_offset}/{len(config.confirmation_seeds)} "
            f"eta_seconds={remaining:.1f}",
            flush=True,
        )
        del encodings, bundles
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    shards = sorted(shard_root.glob("seed*.csv.gz"))
    if len(shards) != len(config.confirmation_seeds):
        raise RuntimeError("Residual confirmation shard grid is incomplete")
    detail = pd.concat([pd.read_csv(path) for path in shards], ignore_index=True)
    if len(detail) != expected_rows * len(config.confirmation_seeds):
        raise RuntimeError("Residual confirmation merged row count is wrong")
    atomic_csv_gzip(detail, detail_path)
    clean_rows_total = 0
    if config.require_clean_necessity:
        clean_shards = sorted(clean_shard_root.glob("seed*.csv.gz"))
        if len(clean_shards) != len(config.confirmation_seeds):
            raise RuntimeError("Clean ablation confirmation shard grid is incomplete")
        clean_detail = pd.concat(
            [pd.read_csv(path) for path in clean_shards], ignore_index=True
        )
        clean_rows_total = expected_clean_rows * len(config.confirmation_seeds)
        if len(clean_detail) != clean_rows_total:
            raise RuntimeError("Clean ablation merged row count is wrong")
        atomic_csv_gzip(clean_detail, clean_detail_path)
    atomic_json(
        complete,
        {
            "schema_version": "realistic_niah_v4_4_4_residual_confirmation_complete_v1",
            "rows": len(detail),
            "clean_ablation_rows": clean_rows_total,
            "design_hash": design_hash,
            "completed_unix": time.time(),
        },
    )
    return detail


def run_model_campaign(
    *,
    run_root: str | Path,
    config: GemmaResidualConfig,
    v4_config_path: str | Path,
    cache_dir: str | Path,
    device_map: str,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    model_root = Path(run_root) / "models" / config.model_label
    complete = model_root / "residual_path_complete.json"
    if resume and complete.is_file():
        return json.loads(complete.read_text(encoding="utf-8"))
    base = V4Config.from_json(v4_config_path)
    current = v444_v4_config(base, config)  # type: ignore[arg-type]
    model, tokenizer, adapter = _load_model(
        config.model_label,
        config=_direction_config(config),
        cache_dir=cache_dir,
        device_map=device_map,
    )
    try:
        selection, artifacts = run_discovery_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            config=config,
            v4_config=current,
            resume=resume,
        )
        smoke = run_smoke_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            config=config,
            v4_config=current,
            selection=selection,
            artifacts=artifacts,
            resume=resume,
        )
        detail = run_confirmation_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            config=config,
            v4_config=current,
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
        "schema_version": "realistic_niah_v4_4_4_residual_model_complete_v1",
        "source_selection_sha256": selection["selection_sha256"],
        "selected_mediator_layer": int(artifacts["selected_mediator_layer"]),
        "smoke": smoke,
        "detail_rows": len(detail),
        "completed_unix": time.time(),
    }
    model_root.mkdir(parents=True, exist_ok=True)
    atomic_json(complete, payload)
    return payload
