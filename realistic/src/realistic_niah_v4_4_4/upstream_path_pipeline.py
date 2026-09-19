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
    stage_root,
    write_stage_status,
)
from realistic_niah_v4_4_3.pipeline import (
    _load_model,
    _load_v4_stimulus_map,
    _render_encodings,
)
from realistic_niah_v4_4_4.interventions import set_output_from_stacked_z
from realistic_niah_v4_4_4.pipeline import _direction_config, _stable_hash, v444_v4_config
from realistic_niah_v4_4_4.spec import V444Config

from .upstream_path import (
    MultiSiteBundle,
    answer_query_full_replacements,
    assert_aligned_pair,
    broad_retrieval_score,
    capture_multisite_bundle,
    late_block_and_control,
    run_path_intervention,
    slot_edge_qk_deltas,
    slot_state_replacements,
    stacked_delta_mapping,
    stacked_late_z,
)
from .upstream_path_spec import FrozenBroadHead, V444UpstreamPathConfig


BASE_STAGE = "upstream_path_base"
EXPANDED_STAGE = "upstream_path_expanded"
SMOKE_STAGE = "upstream_path_smoke"


def _stage_name(expanded: bool) -> str:
    return EXPANDED_STAGE if expanded else BASE_STAGE


def _late_names(config: V444UpstreamPathConfig, *, expanded: bool) -> tuple[str, ...]:
    if not expanded:
        return (config.primary_late_set,)
    return tuple(name for name, _heads in config.late_head_sets if name != config.primary_late_set)


def _candidate_keys(candidates: Sequence[FrozenBroadHead]) -> tuple[tuple[int, int], ...]:
    return tuple((item.layer, item.head) for item in candidates)


def _metric_prefix(payload: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (bool, int, float, str)):
            result[f"{prefix}_{key}"] = value
    return result


def _route_kwargs(
    route: str,
    receiver: MultiSiteBundle,
    donor: MultiSiteBundle,
    receiver_encoding: Any,
    donor_encoding: Any,
    adapter: Any,
    *,
    heads: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    if route == "slot_edge_qk":
        return {
            "query_deltas": slot_edge_qk_deltas(
                receiver,
                donor,
                receiver_encoding,
                donor_encoding,
                adapter,
                heads=heads,
            )
        }
    if route == "answer_query_full":
        assert_aligned_pair(receiver_encoding, donor_encoding)
        return {
            "query_replacements": answer_query_full_replacements(
                donor, adapter, heads=heads
            )
        }
    if route == "slot_state":
        assert_aligned_pair(receiver_encoding, donor_encoding)
        return {"slot_replacements": slot_state_replacements(donor, heads=heads)}
    raise KeyError(route)


def _valid_shard(path: Path, *, design_hash: str, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return (
        len(frame) == int(expected_rows)
        and "design_hash" in frame
        and set(frame["design_hash"].astype(str)) == {str(design_hash)}
    )


def _natural_rows(
    *,
    seed: int,
    count: int,
    bundle: MultiSiteBundle,
    encoding: Any,
    candidates: Sequence[FrozenBroadHead],
    design_hash: str,
) -> list[dict[str, Any]]:
    baseline = candidate_sequence_metrics(bundle.query.candidate_log_scores, encoding)
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "schema_version": "realistic_niah_v4_4_4_upstream_natural_row_v1",
                "design_hash": design_hash,
                "seed": int(seed),
                "gold_count": int(count),
                "candidate_rank": int(rank),
                "layer": int(candidate.layer),
                "head": int(candidate.head),
                "v442_cue_present_score": float(candidate.cue_present_score),
                "v442_cue_absent_score": float(candidate.cue_absent_score),
                "v442_stable_score": float(candidate.stable_score),
                "baseline_expected_count": float(baseline["expected_count"]),
                "baseline_predicted_count": int(baseline["predicted_count_among_candidates"]),
                "baseline_correct": bool(baseline["predicted_count_among_candidates"] == count),
                "attention_cache_raw_logit_delta": float(
                    bundle.query.attention_cache_candidate_logit_max_abs_delta
                ),
                "attention_cache_centered_logit_delta": float(
                    bundle.query.attention_cache_candidate_centered_logit_max_abs_delta
                ),
                **broad_retrieval_score(
                    bundle.query,
                    encoding,
                    layer=candidate.layer,
                    head=candidate.head,
                ),
            }
        )
    return rows


def _effect_row(
    *,
    seed: int,
    receiver_count: int,
    donor_count: int,
    early_size: int,
    early_heads: Sequence[tuple[int, int]],
    route: str,
    late_name: str,
    late_heads: Sequence[int],
    receiver: MultiSiteBundle,
    receiver_encoding: Any,
    adapter: Any,
    early_run: Any,
    block_run: Any,
    control_run: Any,
    induced: torch.Tensor,
    block: torch.Tensor,
    diagnostics: Mapping[str, float],
    design_hash: str,
) -> dict[str, Any]:
    early_metrics = intervention_metrics(
        baseline_output=receiver.query,
        intervened_output=early_run.causal_output,
        encoding=receiver_encoding,
        donor_count=donor_count,
    )
    block_metrics = intervention_metrics(
        baseline_output=receiver.query,
        intervened_output=block_run.causal_output,
        encoding=receiver_encoding,
        donor_count=donor_count,
    )
    control_metrics = intervention_metrics(
        baseline_output=receiver.query,
        intervened_output=control_run.causal_output,
        encoding=receiver_encoding,
        donor_count=donor_count,
    )
    baseline_late = stacked_late_z(
        receiver.query.z_by_layer[28], adapter, layer=28, heads=late_heads
    )
    block_after = stacked_late_z(
        block_run.late_z_after, adapter, layer=28, heads=late_heads
    )
    closure_l2 = float(torch.linalg.vector_norm(block_after - baseline_late))
    closure_relative = closure_l2 / max(float(torch.linalg.vector_norm(induced)), 1e-12)
    block_before = stacked_late_z(
        block_run.late_z_before, adapter, layer=28, heads=late_heads
    )
    control_before = stacked_late_z(
        control_run.late_z_before, adapter, layer=28, heads=late_heads
    )
    early_before = stacked_late_z(
        early_run.late_z_before, adapter, layer=28, heads=late_heads
    )
    reproducibility = max(
        float(torch.linalg.vector_norm(block_before - early_before)),
        float(torch.linalg.vector_norm(control_before - early_before)),
    ) / max(float(torch.linalg.vector_norm(early_before)), 1e-12)
    semantic_gap = int(donor_count) - int(receiver_count)
    early_transport = float(early_metrics["continuous_normalized_transport"])
    block_transport = float(block_metrics["continuous_normalized_transport"])
    control_transport = float(control_metrics["continuous_normalized_transport"])
    baseline_scores = receiver.query.candidate_log_scores

    def donor_log_odds_gain(output: Any) -> float:
        scores = output.causal_output.candidate_log_scores
        baseline_log_odds = float(baseline_scores[donor_count]) - float(
            baseline_scores[receiver_count]
        )
        intervened_log_odds = float(scores[donor_count]) - float(
            scores[receiver_count]
        )
        return intervened_log_odds - baseline_log_odds

    early_log_odds_gain = donor_log_odds_gain(early_run)
    block_log_odds_gain = donor_log_odds_gain(block_run)
    control_log_odds_gain = donor_log_odds_gain(control_run)
    block_output = set_output_from_stacked_z(
        adapter, layer=28, heads=late_heads, stacked_z=block
    )
    return {
        "schema_version": "realistic_niah_v4_4_4_upstream_effect_row_v1",
        "design_hash": design_hash,
        "seed": int(seed),
        "receiver_count": int(receiver_count),
        "donor_count": int(donor_count),
        "semantic_count_shift": int(semantic_gap),
        "early_set": f"top{int(early_size)}",
        "early_set_size": int(early_size),
        "early_heads": ",".join(f"L{layer}H{head}" for layer, head in early_heads),
        "route": str(route),
        "late_set": str(late_name),
        "late_heads": ",".join(f"H{head}" for head in late_heads),
        "early_transport": early_transport,
        "late_block_transport": block_transport,
        "late_control_transport": control_transport,
        "block_suppression": early_transport - block_transport,
        "control_suppression": early_transport - control_transport,
        "mediation_specificity": control_transport - block_transport,
        "early_donor_log_odds_gain": early_log_odds_gain,
        "late_block_donor_log_odds_gain": block_log_odds_gain,
        "late_control_donor_log_odds_gain": control_log_odds_gain,
        "donor_log_odds_block_suppression": (
            early_log_odds_gain - block_log_odds_gain
        ),
        "donor_log_odds_control_suppression": (
            early_log_odds_gain - control_log_odds_gain
        ),
        "donor_log_odds_mediation_specificity": (
            control_log_odds_gain - block_log_odds_gain
        ),
        "late_block_output_norm": float(torch.linalg.vector_norm(block_output)),
        "late_block_closure_l2": closure_l2,
        "late_block_closure_relative_l2": closure_relative,
        "late_prefill_reproducibility_relative_l2": reproducibility,
        **dict(diagnostics),
        **_metric_prefix(early_metrics, "early"),
        **_metric_prefix(block_metrics, "block"),
        **_metric_prefix(control_metrics, "control"),
    }


@torch.inference_mode()
def run_smoke_stage(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    run_root: str | Path,
    base_config: V444Config,
    config: V444UpstreamPathConfig,
    v4_config: V4Config,
    resume: bool,
) -> dict[str, Any]:
    """Exercise every route and both late controls before committing pilot shards."""

    root = stage_root(run_root, config.model_label, SMOKE_STAGE)
    complete = root / "complete.json"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Upstream-path smoke already complete: {complete}")
        return json.loads(complete.read_text(encoding="utf-8"))
    seed = config.evaluation_seeds[0]
    receiver_count, donor_count = config.donor_pairs[0]
    design = {
        "schema_version": "realistic_niah_v4_4_4_upstream_smoke_design_v1",
        "seed": seed,
        "receiver_count": receiver_count,
        "donor_count": donor_count,
        "early_set": "top2",
        "routes": list(config.routes),
        "late_set": config.primary_late_set,
    }
    design_hash = _stable_hash(design)
    atomic_json(root / "design.json", {**design, "design_hash": design_hash})
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=SMOKE_STAGE,
        state="RUNNING",
        detail={"design_hash": design_hash},
    )
    stimuli = _load_v4_stimulus_map(
        run_root,
        seeds=(seed,),
        counts=(receiver_count, donor_count),
    )
    encodings = _render_encodings(
        stimuli,
        tokenizer=tokenizer,
        model_label=config.model_label,
        v4_config=v4_config,
        seed=seed,
        counts=(receiver_count, donor_count),
    )
    all_early = _candidate_keys(config.early_candidates)
    receiver = capture_multisite_bundle(
        model,
        adapter,
        encodings[receiver_count],
        early_heads=all_early,
        mediator_layer=config.mediator_layer,
        cache_logit_tolerance=math.inf,
    )
    donor = capture_multisite_bundle(
        model,
        adapter,
        encodings[donor_count],
        early_heads=all_early,
        mediator_layer=config.mediator_layer,
        cache_logit_tolerance=math.inf,
    )
    assert_aligned_pair(encodings[receiver_count], encodings[donor_count])
    early_heads = _candidate_keys(config.early_set(2))
    late_name = config.primary_late_set
    late_heads = config.late_sets[late_name]
    rows: list[dict[str, Any]] = []
    for route in config.routes:
        route_kwargs = _route_kwargs(
            route,
            receiver,
            donor,
            encodings[receiver_count],
            encodings[donor_count],
            adapter,
            heads=early_heads,
        )
        early_run = run_path_intervention(
            model,
            adapter,
            encodings[receiver_count],
            mediator_layer=config.mediator_layer,
            **route_kwargs,
        )
        baseline_late = stacked_late_z(
            receiver.query.z_by_layer[config.mediator_layer],
            adapter,
            layer=config.mediator_layer,
            heads=late_heads,
        )
        induced = stacked_late_z(
            early_run.late_z_before,
            adapter,
            layer=config.mediator_layer,
            heads=late_heads,
        ) - baseline_late
        block, control, diagnostics = late_block_and_control(
            adapter,
            layer=config.mediator_layer,
            heads=late_heads,
            induced_delta=induced,
            label=f"v444-upstream-smoke:{seed}:{route}",
        )
        block_run = run_path_intervention(
            model,
            adapter,
            encodings[receiver_count],
            mediator_layer=config.mediator_layer,
            late_replacements=stacked_delta_mapping(
                layer=config.mediator_layer,
                heads=late_heads,
                stacked=baseline_late,
            ),
            **route_kwargs,
        )
        control_run = run_path_intervention(
            model,
            adapter,
            encodings[receiver_count],
            mediator_layer=config.mediator_layer,
            late_deltas=stacked_delta_mapping(
                layer=config.mediator_layer,
                heads=late_heads,
                stacked=control,
            ),
            **route_kwargs,
        )
        rows.append(
            _effect_row(
                seed=seed,
                receiver_count=receiver_count,
                donor_count=donor_count,
                early_size=2,
                early_heads=early_heads,
                route=route,
                late_name=late_name,
                late_heads=late_heads,
                receiver=receiver,
                receiver_encoding=encodings[receiver_count],
                adapter=adapter,
                early_run=early_run,
                block_run=block_run,
                control_run=control_run,
                induced=induced,
                block=block,
                diagnostics=diagnostics,
                design_hash=design_hash,
            )
        )
    closure_max = max(abs(float(row["late_block_closure_relative_l2"])) for row in rows)
    orthogonality_max = max(
        abs(float(row["late_control_output_cosine_to_induced"])) for row in rows
    )
    reproducibility_max = max(
        abs(float(row["late_prefill_reproducibility_relative_l2"])) for row in rows
    )
    if closure_max > config.block_closure_relative_tolerance:
        raise RuntimeError(f"Upstream smoke L28 closure failed: {closure_max}")
    if orthogonality_max > config.control_orthogonality_tolerance:
        raise RuntimeError(f"Upstream smoke late control is not orthogonal: {orthogonality_max}")
    if reproducibility_max > 1e-5:
        raise RuntimeError(f"Upstream smoke prefill is not reproducible: {reproducibility_max}")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_upstream_smoke_complete_v1",
        "design_hash": design_hash,
        "row_count": len(rows),
        "closure_max": closure_max,
        "orthogonality_max": orthogonality_max,
        "reproducibility_max": reproducibility_max,
        "rows": rows,
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=SMOKE_STAGE,
        state="COMPLETE",
        detail={key: value for key, value in payload.items() if key != "rows"},
    )
    return payload


@torch.inference_mode()
def run_stage(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    run_root: str | Path,
    base_config: V444Config,
    config: V444UpstreamPathConfig,
    v4_config: V4Config,
    expanded: bool,
    resume: bool,
) -> dict[str, Any]:
    config.validate_against_base(base_config)
    stage = _stage_name(expanded)
    late_names = _late_names(config, expanded=expanded)
    root = stage_root(run_root, config.model_label, stage)
    complete = root / "complete.json"
    if complete.is_file():
        if not resume:
            raise FileExistsError(f"Upstream-path stage already complete: {complete}")
        return json.loads(complete.read_text(encoding="utf-8"))
    design = {
        "schema_version": "realistic_niah_v4_4_4_upstream_design_v1",
        "stage": stage,
        "expanded": bool(expanded),
        "late_sets": [[name, list(config.late_sets[name])] for name in late_names],
        "config": config.to_dict(),
        "frozen_ranking_rule": (
            "descending min(V4.4.2 cue-present broad score, "
            "V4.4.2 cue-absent broad score), frozen before this causal stage"
        ),
    }
    design_hash = _stable_hash(design)
    atomic_json(root / "design.json", {**design, "design_hash": design_hash})
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=stage,
        state="RUNNING",
        detail={"design_hash": design_hash},
    )
    stimuli = _load_v4_stimulus_map(
        run_root, seeds=config.evaluation_seeds, counts=config.counts
    )
    all_early = _candidate_keys(config.early_candidates)
    expected_effect_rows = (
        len(config.donor_pairs)
        * len(config.early_set_sizes)
        * len(config.routes)
        * len(late_names)
    )
    expected_natural_rows = len(config.counts) * len(config.early_candidates)
    started = time.monotonic()
    for seed_offset, seed in enumerate(config.evaluation_seeds, start=1):
        effect_path = root / "effects" / f"seed_{seed}.csv.gz"
        natural_path = root / "natural" / f"seed_{seed}.csv.gz"
        natural_ok = expanded or _valid_shard(
            natural_path, design_hash=design_hash, expected_rows=expected_natural_rows
        )
        if resume and natural_ok and _valid_shard(
            effect_path, design_hash=design_hash, expected_rows=expected_effect_rows
        ):
            print(
                f"[v4.4.4 upstream {stage}] seed={seed} resume-skip "
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
        bundles: dict[int, MultiSiteBundle] = {}
        natural_rows: list[dict[str, Any]] = []
        for count in config.counts:
            bundle = capture_multisite_bundle(
                model,
                adapter,
                encodings[count],
                early_heads=all_early,
                mediator_layer=config.mediator_layer,
                cache_logit_tolerance=math.inf,
            )
            bundles[count] = bundle
            if not expanded:
                natural_rows.extend(
                    _natural_rows(
                        seed=seed,
                        count=count,
                        bundle=bundle,
                        encoding=encodings[count],
                        candidates=config.early_candidates,
                        design_hash=design_hash,
                    )
                )
        effect_rows: list[dict[str, Any]] = []
        for receiver_count, donor_count in config.donor_pairs:
            receiver = bundles[receiver_count]
            donor = bundles[donor_count]
            receiver_encoding = encodings[receiver_count]
            donor_encoding = encodings[donor_count]
            assert_aligned_pair(receiver_encoding, donor_encoding)
            for early_size in config.early_set_sizes:
                early_heads = _candidate_keys(config.early_set(early_size))
                for route in config.routes:
                    route_kwargs = _route_kwargs(
                        route,
                        receiver,
                        donor,
                        receiver_encoding,
                        donor_encoding,
                        adapter,
                        heads=early_heads,
                    )
                    early_run = run_path_intervention(
                        model,
                        adapter,
                        receiver_encoding,
                        mediator_layer=config.mediator_layer,
                        **route_kwargs,
                    )
                    for late_name in late_names:
                        late_heads = config.late_sets[late_name]
                        baseline_late = stacked_late_z(
                            receiver.query.z_by_layer[config.mediator_layer],
                            adapter,
                            layer=config.mediator_layer,
                            heads=late_heads,
                        )
                        early_late = stacked_late_z(
                            early_run.late_z_before,
                            adapter,
                            layer=config.mediator_layer,
                            heads=late_heads,
                        )
                        induced = early_late - baseline_late
                        block, control, diagnostics = late_block_and_control(
                            adapter,
                            layer=config.mediator_layer,
                            heads=late_heads,
                            induced_delta=induced,
                            label=(
                                f"v444-upstream:{stage}:{seed}:{receiver_count}:"
                                f"{donor_count}:{early_size}:{route}:{late_name}"
                            ),
                        )
                        block_run = run_path_intervention(
                            model,
                            adapter,
                            receiver_encoding,
                            mediator_layer=config.mediator_layer,
                            late_replacements=stacked_delta_mapping(
                                layer=config.mediator_layer,
                                heads=late_heads,
                                stacked=baseline_late,
                            ),
                            **route_kwargs,
                        )
                        control_run = run_path_intervention(
                            model,
                            adapter,
                            receiver_encoding,
                            mediator_layer=config.mediator_layer,
                            late_deltas=stacked_delta_mapping(
                                layer=config.mediator_layer,
                                heads=late_heads,
                                stacked=control,
                            ),
                            **route_kwargs,
                        )
                        effect_rows.append(
                            _effect_row(
                                seed=seed,
                                receiver_count=receiver_count,
                                donor_count=donor_count,
                                early_size=early_size,
                                early_heads=early_heads,
                                route=route,
                                late_name=late_name,
                                late_heads=late_heads,
                                receiver=receiver,
                                receiver_encoding=receiver_encoding,
                                adapter=adapter,
                                early_run=early_run,
                                block_run=block_run,
                                control_run=control_run,
                                induced=induced,
                                block=block,
                                diagnostics=diagnostics,
                                design_hash=design_hash,
                            )
                        )
        if len(effect_rows) != expected_effect_rows:
            raise RuntimeError("Upstream effect row count mismatch before shard commit")
        atomic_csv_gzip(pd.DataFrame(effect_rows), effect_path)
        if not expanded:
            if len(natural_rows) != expected_natural_rows:
                raise RuntimeError("Upstream natural row count mismatch before shard commit")
            atomic_csv_gzip(pd.DataFrame(natural_rows), natural_path)
        del bundles
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[v4.4.4 upstream {stage}] seed={seed} complete "
            f"({seed_offset}/{len(config.evaluation_seeds)}), rows={len(effect_rows)}",
            flush=True,
        )
    effect_files = sorted((root / "effects").glob("seed_*.csv.gz"))
    if len(effect_files) != len(config.evaluation_seeds):
        raise RuntimeError("Upstream stage has missing effect shards")
    effects = pd.concat([pd.read_csv(path) for path in effect_files], ignore_index=True)
    merged_effect = root / "effects.csv.gz"
    atomic_csv_gzip(effects, merged_effect)
    natural_count = 0
    if not expanded:
        natural_files = sorted((root / "natural").glob("seed_*.csv.gz"))
        if len(natural_files) != len(config.evaluation_seeds):
            raise RuntimeError("Upstream base stage has missing natural shards")
        natural = pd.concat([pd.read_csv(path) for path in natural_files], ignore_index=True)
        natural_count = len(natural)
        atomic_csv_gzip(natural, root / "natural.csv.gz")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_upstream_stage_complete_v1",
        "stage": stage,
        "expanded": bool(expanded),
        "design_hash": design_hash,
        "late_sets": list(late_names),
        "effect_rows": int(len(effects)),
        "natural_rows": int(natural_count),
        "seed_count": len(config.evaluation_seeds),
        "runtime_seconds": time.monotonic() - started,
        "completed_unix": time.time(),
    }
    atomic_json(complete, payload)
    write_stage_status(
        run_root,
        model_label=config.model_label,
        stage=stage,
        state="COMPLETE",
        detail=payload,
    )
    return payload


def run_model_stage(
    *,
    run_root: str | Path,
    base_config: V444Config,
    config: V444UpstreamPathConfig,
    v4_config_path: str | Path,
    cache_dir: str | Path,
    device_map: str,
    expanded: bool,
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
    if not expanded:
        run_smoke_stage(
            model,
            tokenizer,
            adapter,
            run_root=run_root,
            base_config=base_config,
            config=config,
            v4_config=v4_config,
            resume=resume,
        )
    return run_stage(
        model,
        tokenizer,
        adapter,
        run_root=run_root,
        base_config=base_config,
        config=config,
        v4_config=v4_config,
        expanded=expanded,
        resume=resume,
    )
