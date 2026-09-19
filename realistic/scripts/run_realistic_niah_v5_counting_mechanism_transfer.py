#!/usr/bin/env python3
"""Run paper-aligned counting-mechanism experiments on native thinking traces.

This is an exploratory transfer of CountScope, continued counting, mean
position-difference steering, separator collapse, and maximum-latent-count
interventions.  It deliberately reports argmax/greedy adoption in addition to
the paper's probability-based causal-influence score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# The V5 implementation owns the stricter sequence-length-audited capture.
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from realistic_niah_v5.count_stream import (  # noqa: E402
    build_answer_source_registry,
)
from realistic_niah_v5.counting_mechanism_transfer import (  # noqa: E402
    SCHEMA_VERSION,
    align_occurrence_regions,
    align_region_groups,
    build_first_pass_tstar_answer_source_registry,
    build_immediate_count_query_encoding,
    continued_count_expected,
    countscope_blank_encoding,
    gather_aligned_states,
    item_candidate_tokens,
    layerwise_centroid_deltas,
    maximum_latent_count_expected,
    norm_matched_orthogonal_control,
    occurrence_region_groups,
    occurrence_region_positions,
    paper_causal_influence,
    prefill_with_block_input_intervention,
    prefix_through_boundary,
    prompt_scrubbed_encoding,
    score_native_item_candidates,
    score_prefill,
)
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_first_occurrence_prefix_clean,
    audit_no_count_enumeration_trace,
)
from realistic_niah_v4_4_5.restoration import (  # noqa: E402
    generate_answer_completion_from_prefill,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
EXPERIMENTS = (
    "countscope",
    "continued_counting",
    "linear_additivity",
    "separator_collapse",
    "maximum_count",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Counting-mechanism config must be one JSON object")
    if str(value.get("schema_version")) not in {
        "counting_mechanism_transfer_plan_v1",
        "counting_mechanism_transfer_plan_v2",
    }:
        raise ValueError("Unsupported counting-mechanism transfer config")
    experiments = value.get("experiments")
    if not isinstance(experiments, dict):
        raise ValueError("Config has no experiments mapping")
    unknown = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Config contains unknown experiments: {unknown}")
    candidates = tuple(int(item) for item in value.get("candidate_counts", ()))
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Candidate count registry is empty or duplicated")
    if tuple(sorted(candidates)) != candidates:
        raise ValueError("Candidate count registry must be sorted")
    return value


def _read_rows(
    path: Path,
    *,
    selection_mode: str = "unique_seed",
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        lines = list(source)
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        seed = int(row["seed"])
        if str(selection_mode) == "first_pass_noindex":
            audit = audit_first_occurrence_prefix_clean(row)
            frozen_audits = [
                (str(key), value)
                for key, value in row.items()
                if str(key).endswith("_format_audit")
                and isinstance(value, Mapping)
                and "primary_eligible_prefix_clean" in value
            ]
            cohorts = [
                (str(key), value)
                for key, value in row.items()
                if str(key).endswith("_cohort")
                and isinstance(value, Mapping)
                and str(value.get("selection_population", ""))
                == "first_pass_noindex_enumeration"
            ]
            if len(frozen_audits) != 1 or len(cohorts) != 1:
                raise ValueError(
                    f"Seed {seed} lacks one frozen first-pass audit/cohort"
                )
            frozen_key, frozen_audit = frozen_audits[0]
            cohort_key, cohort = cohorts[0]
            if not bool(audit["eligible"]):
                raise ValueError(
                    f"Seed {seed} fails recomputed prefix-clean audit: "
                    f"{audit['reasons']}"
                )
            if not bool(frozen_audit.get("primary_eligible_prefix_clean")):
                raise ValueError(f"Seed {seed} fails frozen prefix-clean audit")
            if int(audit["t_star_char"]) != int(frozen_audit["t_star_char"]):
                raise ValueError(f"Seed {seed} t_star changed since cohort freeze")
            split = str(cohort.get("split", ""))
            if split not in {"discovery", "confirmation"}:
                raise ValueError(f"Seed {seed} has invalid frozen cohort split")
            row = dict(row)
            row["source_generation_split"] = str(row.get("split", ""))
            row["split"] = split
            row["active_first_pass_audit_key"] = frozen_key
            row["active_first_pass_cohort_key"] = cohort_key
        else:
            audit = audit_no_count_enumeration_trace(row)
            if not bool(audit["eligible"]):
                raise ValueError(
                    f"Seed {seed} fails unindexed audit: {audit['reasons']}"
                )
        if int(row.get("gold_count", -1)) < 1:
            raise ValueError(f"Seed {seed} has no positive gold count")
        grouped.setdefault(seed, []).append(row)
    result: dict[int, dict[str, Any]] = {}
    mode = str(selection_mode)
    for seed, rows in sorted(grouped.items()):
        if mode in {"unique_seed", "first_pass_noindex"}:
            if len(rows) != 1:
                raise ValueError(f"Duplicate generation seed {seed}")
            selected = rows[0]
        elif mode == "maximum_gold_count_per_seed":
            maximum = max(int(row["gold_count"]) for row in rows)
            candidates = [
                row for row in rows if int(row["gold_count"]) == maximum
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Maximum-count row selection has an unresolved seed tie: "
                    f"seed={seed} count={maximum} rows={len(candidates)}"
                )
            selected = candidates[0]
        else:
            raise ValueError(f"Unknown generation row selection mode: {mode}")
        result[seed] = selected
    if not result:
        raise ValueError("Generation file is empty")
    return result


def _read_existing_trials(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as source:
        return [json.loads(line) for line in source if line.strip()]


def _experiment_seeds(
    config: Mapping[str, Any],
    experiment: str,
    override: Sequence[int] | None,
) -> tuple[int, ...]:
    if override:
        values = tuple(int(value) for value in override)
    else:
        active = config["experiments"][experiment]
        key = "eval_seeds" if experiment == "linear_additivity" else "seeds"
        values = tuple(int(value) for value in active.get(key, config.get("seeds", ())))
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{experiment} has no unique evaluation seeds")
    return values


def _layers(adapter: Any) -> tuple[int, ...]:
    return tuple(range(int(adapter.num_layers)))


def _all_trace_capture(
    model: Any,
    adapter: Any,
    encoding: Any,
    registry: Any,
) -> tuple[tuple[int, ...], dict[int, torch.Tensor]]:
    item_positions = tuple(
        int(value) for value in registry.positions("trace_items")
    )
    # Include the first position after every item.  The pre-block state at an
    # item's final token has not processed that token yet; ``post_item`` is the
    # earliest state that has causally incorporated the entire item.
    post_item_positions = tuple(
        int(end) for _start, end in registry.trace_items
    )
    positions = tuple(sorted(set(item_positions + post_item_positions)))
    captures = capture_decoder_block_input_states(
        model,
        adapter,
        encoding,
        positions,
        layers=_layers(adapter),
    )
    return positions, captures


def _clean_prefill(
    model: Any,
    adapter: Any,
    encoding: Any,
    *,
    audit_position: int,
) -> Any:
    prefill, _captures, applications, realized = prefill_with_block_input_intervention(
        model,
        adapter,
        encoding,
        positions=(int(audit_position),),
        layer_values=None,
        intervention_kind="replace",
    )
    if applications or realized:
        raise RuntimeError("Clean prefill unexpectedly applied an intervention")
    return prefill


def _hook_audit(
    applications: Mapping[int, int], realized: Mapping[int, float]
) -> dict[str, Any]:
    return {
        "patch_hook_applications": {
            str(layer): int(value) for layer, value in sorted(applications.items())
        },
        "patch_realized_fro_norm_by_layer": {
            str(layer): float(value) for layer, value in sorted(realized.items())
        },
    }


def occurrence_region_positions_or_none(
    registry: Any, occurrence: int, region: str
) -> tuple[int, ...]:
    try:
        return occurrence_region_positions(registry, occurrence, region)
    except ValueError:
        return ()


def _common(
    *,
    experiment: str,
    condition: str,
    seed: int,
    encoding: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": str(experiment),
        "condition": str(condition),
        "phase": str(config.get("phase", "discovery")),
        "model_label": str(encoding.model_label),
        "request_id": str(encoding.request_id),
        "seed": int(seed),
        "dataset_split": str(encoding.split),
        "gold_count": int(encoding.count),
        "explicit_item_index_present": False,
        "explicit_item_index_present_before_tstar": False,
        "future_post_tstar_text_present_in_model_input": False,
        "no_count_enumeration_audit_required": True,
        "selection_uses_current_experiment_outcomes": False,
    }


def _decode_countscope_state(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    receiver: Any,
    receiver_position: int,
    state_by_layer: Mapping[int, torch.Tensor],
    baseline_prefill: Any,
    *,
    expected_count: int,
    original_count: int,
    run_greedy: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    replacements = {
        int(layer): torch.as_tensor(states).reshape(1, -1)
        for layer, states in state_by_layer.items()
    }
    prefill, _capture, applications, realized = prefill_with_block_input_intervention(
        model,
        adapter,
        receiver,
        positions=(int(receiver_position),),
        layer_values=replacements,
        intervention_kind="replace",
        norm_rescale_replacement=True,
    )
    baseline = score_prefill(
        model,
        tokenizer,
        receiver,
        baseline_prefill,
        target_count=int(expected_count),
        original_count=int(original_count),
        run_greedy=False,
        max_new_tokens=int(max_new_tokens),
    )
    patched = score_prefill(
        model,
        tokenizer,
        receiver,
        prefill,
        target_count=int(expected_count),
        original_count=int(original_count),
        run_greedy=bool(run_greedy),
        max_new_tokens=int(max_new_tokens),
    )
    return {
        **patched,
        "paper_ci": paper_causal_influence(
            baseline,
            patched,
            expected_count=int(expected_count),
            original_count=int(original_count),
        ),
        "blank_target_probability": float(baseline["target_probability"]),
        "blank_original_probability": float(
            baseline["candidate_probabilities_by_count"][str(int(original_count))]
        ),
        **_hook_audit(applications, realized),
    }


def _greedy_successor(
    model: Any,
    tokenizer: Any,
    prefix: Any,
    prefill: Any,
    candidates: Mapping[int, Sequence[int]],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    maximum_candidate = max(len(tuple(value)) for value in candidates.values())
    generated = generate_answer_completion_from_prefill(
        model,
        tokenizer,
        prefix,
        prefill,
        max_new_tokens=max(int(max_new_tokens), int(maximum_candidate)),
    )
    token_ids = tuple(int(value) for value in generated["generated_token_ids"])
    matches = [
        int(occurrence)
        for occurrence, candidate in candidates.items()
        if token_ids[: len(tuple(candidate))] == tuple(int(value) for value in candidate)
    ]
    return {
        "greedy_successor_occurrence": matches[0] if len(matches) == 1 else None,
        "greedy_successor_unique_match": len(matches) == 1,
        "greedy_successor_matches": matches,
        "greedy_successor_token_ids": list(token_ids),
        "greedy_successor_text": str(generated["completion_text"]),
    }


def _run_countscope(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encoding: Any,
    registry: Any,
    capture_positions: Sequence[int],
    captures: Mapping[int, torch.Tensor],
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    spec = config["experiments"]["countscope"]
    receiver_occurrence = int(spec.get("receiver_occurrence", 1))
    receiver_modes = tuple(
        str(value) for value in spec.get("receiver_modes", ("scrubbed_blank",))
    )
    norm_rescale_replacement = bool(
        spec.get("norm_rescale_replacement", True)
    )
    if not receiver_modes or len(set(receiver_modes)) != len(receiver_modes):
        raise ValueError("CountScope receiver modes must be unique and nonempty")
    for receiver_mode in receiver_modes:
        if receiver_mode == "scrubbed_blank":
            receiver, receiver_audit = countscope_blank_encoding(
                encoding,
                registry,
                tokenizer,
                receiver_occurrence=receiver_occurrence,
                random_seed=int(config["random_seed"]) + int(encoding.seed) * 1009,
            )
        elif receiver_mode == "natural_prefix":
            receiver, early_audit = build_immediate_count_query_encoding(
                encoding,
                registry,
                tokenizer,
                target_occurrence=receiver_occurrence,
            )
            receiver_audit = {
                **early_audit,
                "target_context": "natural_single_prefix_receiver",
                "prompt_records_scrubbed": False,
                "item_alphanumeric_semantics_scrubbed": False,
                "target_input_ids_sha256": hashlib.sha256(
                    json.dumps(list(receiver.input_ids)).encode("utf-8")
                ).hexdigest(),
            }
        elif receiver_mode == "scrubbed_prompt_natural_item":
            prompt_scrubbed, prompt_audit = prompt_scrubbed_encoding(
                encoding,
                registry,
                tokenizer,
                random_seed=int(config["random_seed"]) + int(encoding.seed) * 1009,
            )
            receiver, early_audit = build_immediate_count_query_encoding(
                prompt_scrubbed,
                registry,
                tokenizer,
                target_occurrence=receiver_occurrence,
            )
            receiver_audit = {
                **prompt_audit,
                **early_audit,
                "target_context": "scrubbed_prompt_natural_single_item_receiver",
                "prompt_records_scrubbed": True,
                "item_alphanumeric_semantics_scrubbed": False,
                "target_input_ids_sha256": hashlib.sha256(
                    json.dumps(list(receiver.input_ids)).encode("utf-8")
                ).hexdigest(),
            }
        else:
            raise ValueError(f"Unknown CountScope receiver mode: {receiver_mode}")
        receiver_prefill = _clean_prefill(
            model,
            adapter,
            receiver,
            audit_position=int(registry.trace_items[receiver_occurrence - 1][1]) - 1,
        )
        for donor in (int(value) for value in spec["donor_occurrences"]):
            for region in (str(value) for value in spec["regions"]):
                started = time.perf_counter()
                alignment = align_occurrence_regions(
                    registry,
                    source_occurrences=(donor,),
                    receiver_occurrences=(receiver_occurrence,),
                    region=region,
                )
                replacements = gather_aligned_states(
                    captures, capture_positions, alignment.donor_positions
                )
                prefill, _readout, applications, realized = (
                    prefill_with_block_input_intervention(
                        model,
                        adapter,
                        receiver,
                        positions=alignment.receiver_positions,
                        layer_values=replacements,
                        intervention_kind="replace",
                        norm_rescale_replacement=norm_rescale_replacement,
                    )
                )
                baseline = score_prefill(
                    model,
                    tokenizer,
                    receiver,
                    receiver_prefill,
                    target_count=donor,
                    original_count=receiver_occurrence,
                    run_greedy=bool(config["run_greedy"]),
                    max_new_tokens=int(config["max_new_tokens"]),
                )
                patched = score_prefill(
                    model,
                    tokenizer,
                    receiver,
                    prefill,
                    target_count=donor,
                    original_count=receiver_occurrence,
                    run_greedy=bool(config["run_greedy"]),
                    max_new_tokens=int(config["max_new_tokens"]),
                )
                rows.append(
                    {
                        **_common(
                            experiment="countscope",
                            condition=(
                                "all_layer_donor_to_blank_receiver"
                                if receiver_mode == "scrubbed_blank"
                                else f"all_layer_donor_to_{receiver_mode}_receiver"
                            ),
                            seed=int(encoding.seed),
                            encoding=encoding,
                            config=config,
                        ),
                        "donor_occurrence": donor,
                        "receiver_occurrence": receiver_occurrence,
                        "receiver_mode": receiver_mode,
                        "region": region,
                        "patch_layer_mode": "all_decoder_block_inputs",
                        "replacement_norm_rescaled_to_receiver": (
                            norm_rescale_replacement
                        ),
                        "paper_ci": paper_causal_influence(
                            baseline,
                            patched,
                            expected_count=donor,
                            original_count=receiver_occurrence,
                        ),
                        "blank_target_probability": float(
                            baseline["target_probability"]
                        ),
                        "blank_predicted_count": int(
                            baseline["predicted_count_among_candidates"]
                        ),
                        "blank_target_is_candidate_argmax": bool(
                            baseline["target_is_candidate_argmax"]
                        ),
                        "blank_greedy_prediction": baseline.get("greedy_prediction"),
                        "blank_greedy_target_adoption": bool(
                            baseline.get("greedy_target_adoption", False)
                        ),
                        "candidate_new_adoption": bool(
                            patched["target_is_candidate_argmax"]
                            and not baseline["target_is_candidate_argmax"]
                        ),
                        "greedy_new_adoption": bool(
                            patched.get("greedy_target_adoption", False)
                            and not baseline.get("greedy_target_adoption", False)
                        ),
                        "elapsed_seconds": float(time.perf_counter() - started),
                        **alignment.to_dict(),
                        **receiver_audit,
                        **_hook_audit(applications, realized),
                        **patched,
                    }
                )


def _run_continued(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encoding: Any,
    registry: Any,
    target: Any,
    target_audit: Mapping[str, Any],
    capture_positions: Sequence[int],
    captures: Mapping[int, torch.Tensor],
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    spec = config["experiments"]["continued_counting"]
    norm_rescale_replacement = bool(
        spec.get("norm_rescale_replacement", True)
    )
    target_start = int(spec.get("target_start_occurrence", 1))
    if target_start != 1:
        raise ValueError("V1 continued-counting transfer requires target start 1")
    all_layers = _layers(adapter)
    target_item_count = len(registry.trace_items)
    final_target, final_target_audit = build_immediate_count_query_encoding(
        target,
        registry,
        tokenizer,
        target_occurrence=target_item_count,
    )
    target_baseline_prefill = _clean_prefill(
        model,
        adapter,
        final_target,
        audit_position=int(registry.trace_items[0][1]) - 1,
    )
    countscope_receiver, countscope_audit = countscope_blank_encoding(
        encoding,
        registry,
        tokenizer,
        receiver_occurrence=1,
        random_seed=int(config["random_seed"]) + int(encoding.seed) * 2017,
    )
    countscope_position = int(registry.trace_items[0][1]) - 1
    countscope_baseline = _clean_prefill(
        model,
        adapter,
        countscope_receiver,
        audit_position=countscope_position,
    )
    candidate_items = item_candidate_tokens(encoding, registry)
    prefix_baselines: dict[int, tuple[Any, dict[str, Any]]] = {}
    for k in (int(value) for value in spec["k_values"]):
        prefix = prefix_through_boundary(
            target, int(registry.trace_items[k - 1][1]) - 1
        )
        baseline_prefill = _clean_prefill(
            model,
            adapter,
            prefix,
            audit_position=int(registry.trace_items[k - 1][1]) - 1,
        )
        prefix_baselines[k] = (
            prefix,
            score_native_item_candidates(
                model,
                prefix,
                baseline_prefill,
                candidate_items,
                target=k + 1,
            ),
        )

    for source_end in (int(value) for value in spec["source_end_occurrences"]):
        for k in (int(value) for value in spec["k_values"]):
            if source_end < k or k >= target_item_count:
                continue
            source_occurrences = tuple(range(source_end - k + 1, source_end + 1))
            receiver_occurrences = tuple(range(1, k + 1))
            read_occurrences = tuple(
                k + int(hop)
                for hop in spec.get("read_hops", (1, 2))
                if k + int(hop) <= target_item_count
            )
            read_positions = tuple(
                int(registry.trace_items[value - 1][1]) - 1
                for value in read_occurrences
            )
            final_expected = continued_count_expected(
                source_end, target_item_count, k
            )
            for region in (str(value) for value in spec["regions"]):
                started = time.perf_counter()
                alignment = align_occurrence_regions(
                    registry,
                    source_occurrences=source_occurrences,
                    receiver_occurrences=receiver_occurrences,
                    region=region,
                )
                replacements = gather_aligned_states(
                    captures, capture_positions, alignment.donor_positions
                )
                prefill, readout, applications, realized = (
                    prefill_with_block_input_intervention(
                        model,
                        adapter,
                        final_target,
                        positions=alignment.receiver_positions,
                        layer_values=replacements,
                        intervention_kind="replace",
                        readout_layers=all_layers,
                        readout_positions=read_positions,
                        norm_rescale_replacement=norm_rescale_replacement,
                    )
                )
                baseline_final = score_prefill(
                    model,
                    tokenizer,
                    final_target,
                    target_baseline_prefill,
                    target_count=final_expected,
                    original_count=target_item_count,
                    run_greedy=False,
                    max_new_tokens=int(config["max_new_tokens"]),
                )
                patched_final = score_prefill(
                    model,
                    tokenizer,
                    final_target,
                    prefill,
                    target_count=final_expected,
                    original_count=target_item_count,
                    run_greedy=bool(config["run_greedy"]),
                    max_new_tokens=int(config["max_new_tokens"]),
                )

                early_stop_rows: list[dict[str, Any]] = []
                for hop in (
                    int(value) for value in spec.get("early_stop_hops", ())
                ):
                    target_occurrence = k + hop
                    if hop < 1 or target_occurrence > target_item_count:
                        continue
                    early_target, early_stop_audit = (
                        build_immediate_count_query_encoding(
                            target,
                            registry,
                            tokenizer,
                            target_occurrence=target_occurrence,
                        )
                    )
                    early_baseline_prefill = _clean_prefill(
                        model,
                        adapter,
                        early_target,
                        audit_position=int(
                            registry.trace_items[target_occurrence - 1][1]
                        )
                        - 1,
                    )
                    early_prefill, _early_capture, early_apps, early_norms = (
                        prefill_with_block_input_intervention(
                            model,
                            adapter,
                            early_target,
                            positions=alignment.receiver_positions,
                            layer_values=replacements,
                            intervention_kind="replace",
                            norm_rescale_replacement=(
                                norm_rescale_replacement
                            ),
                        )
                    )
                    expected = source_end + hop
                    early_baseline = score_prefill(
                        model,
                        tokenizer,
                        early_target,
                        early_baseline_prefill,
                        target_count=expected,
                        original_count=target_occurrence,
                        run_greedy=False,
                        max_new_tokens=int(config["max_new_tokens"]),
                    )
                    early_patched = score_prefill(
                        model,
                        tokenizer,
                        early_target,
                        early_prefill,
                        target_count=expected,
                        original_count=target_occurrence,
                        run_greedy=bool(config["run_greedy"]),
                        max_new_tokens=int(config["max_new_tokens"]),
                    )
                    early_stop_rows.append(
                        {
                            "hop_after_patch": hop,
                            "target_occurrence": target_occurrence,
                            "continued_expected_count": expected,
                            "original_count": target_occurrence,
                            "paper_ci": paper_causal_influence(
                                early_baseline,
                                early_patched,
                                expected_count=expected,
                                original_count=target_occurrence,
                            ),
                            "baseline_target_probability": float(
                                early_baseline["target_probability"]
                            ),
                            "baseline_original_probability": float(
                                early_baseline["candidate_probabilities_by_count"][
                                    str(target_occurrence)
                                ]
                            ),
                            **_hook_audit(early_apps, early_norms),
                            **early_stop_audit,
                            **early_patched,
                        }
                    )

                hop_rows: list[dict[str, Any]] = []
                for hop_index, occurrence in enumerate(read_occurrences):
                    expected = source_end + (occurrence - k)
                    states = {
                        layer: readout[layer][hop_index]
                        for layer in all_layers
                    }
                    decoded = _decode_countscope_state(
                        model,
                        tokenizer,
                        adapter,
                        countscope_receiver,
                        countscope_position,
                        states,
                        countscope_baseline,
                        expected_count=expected,
                        original_count=1,
                        run_greedy=bool(config["run_greedy"]),
                        max_new_tokens=int(config["max_new_tokens"]),
                    )
                    hop_rows.append(
                        {
                            "target_occurrence": int(occurrence),
                            "hop_after_patch": int(occurrence - k),
                            "continued_expected_count": int(expected),
                            **decoded,
                        }
                    )

                donor_successor = source_end + 1
                if region == "post_item":
                    # This state is the first block input after the item, so it
                    # lies one position beyond the shorter successor-scoring
                    # prefix.  Keep the count readouts and omit only that
                    # incompatible content-continuation assay.
                    successor_scores = {}
                    greedy_successor = {}
                    greedy_donor_adoption = False
                    successor_available = False
                    successor_unavailable_reason = (
                        "post_item_state_requires_next_token_context"
                    )
                    prefix_apps: dict[int, int] = {}
                    prefix_norms: dict[int, float] = {}
                else:
                    prefix, baseline_successor = prefix_baselines[k]
                    prefix_prefill, _unused, prefix_apps, prefix_norms = (
                        prefill_with_block_input_intervention(
                            model,
                            adapter,
                            prefix,
                            positions=alignment.receiver_positions,
                            layer_values=replacements,
                            intervention_kind="replace",
                            norm_rescale_replacement=(
                                norm_rescale_replacement
                            ),
                        )
                    )
                if region != "post_item" and donor_successor in candidate_items:
                    successor_scores = score_native_item_candidates(
                        model,
                        prefix,
                        prefix_prefill,
                        candidate_items,
                        target=donor_successor,
                        baseline=baseline_successor,
                    )
                    greedy_successor = _greedy_successor(
                        model,
                        tokenizer,
                        prefix,
                        prefix_prefill,
                        candidate_items,
                        max_new_tokens=int(
                            spec.get("successor_max_new_tokens", 24)
                        ),
                    )
                    greedy_donor_adoption = bool(
                        greedy_successor["greedy_successor_occurrence"]
                        == donor_successor
                    )
                    successor_available = True
                    successor_unavailable_reason = None
                elif region != "post_item":
                    successor_scores = {}
                    greedy_successor = {}
                    greedy_donor_adoption = False
                    successor_available = False
                    successor_unavailable_reason = (
                        "donor_successor_outside_observed_trace"
                    )
                rows.append(
                    {
                        **_common(
                            experiment="continued_counting",
                            condition="last_k_source_to_first_k_target",
                            seed=int(encoding.seed),
                            encoding=encoding,
                            config=config,
                        ),
                        "source_end_occurrence": source_end,
                        "source_occurrences": list(source_occurrences),
                        "target_occurrences": list(receiver_occurrences),
                        "k": k,
                        "region": region,
                        "target_item_count": target_item_count,
                        "continued_final_expected_count": final_expected,
                        "continued_next_expected_count": source_end + 1,
                        "patch_layer_mode": "all_decoder_block_inputs",
                        "replacement_norm_rescaled_to_receiver": (
                            norm_rescale_replacement
                        ),
                        "paper_ci": paper_causal_influence(
                            baseline_final,
                            patched_final,
                            expected_count=final_expected,
                            original_count=target_item_count,
                        ),
                        "baseline_expected_probability": float(
                            baseline_final["target_probability"]
                        ),
                        "baseline_original_probability": float(
                            baseline_final["candidate_probabilities_by_count"][
                                str(target_item_count)
                            ]
                        ),
                        "boundary_countscope_readouts": hop_rows,
                        "early_stop_count_readouts": early_stop_rows,
                        "successor_readout": {
                            **successor_scores,
                            **greedy_successor,
                            "available": successor_available,
                            "unavailable_reason": (
                                None
                                if successor_available
                                else successor_unavailable_reason
                            ),
                            "donor_successor_occurrence": donor_successor,
                            "receiver_natural_successor_occurrence": k + 1,
                            "greedy_donor_successor_adoption": (
                                greedy_donor_adoption
                            ),
                            "prefix_patch_audit": _hook_audit(
                                prefix_apps, prefix_norms
                            ),
                        },
                        "elapsed_seconds": float(time.perf_counter() - started),
                        **alignment.to_dict(),
                        **target_audit,
                        **final_target_audit,
                        "countscope_receiver_audit": countscope_audit,
                        **_hook_audit(applications, realized),
                        **patched_final,
                    }
                )


def _fit_position_centroids(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    rows_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    fit_seeds: Sequence[int],
    layers: Sequence[int],
    candidate_counts: Sequence[int],
    registry_builder: Any = build_answer_source_registry,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    sums: dict[int, dict[int, torch.Tensor]] = {
        int(layer): {} for layer in layers
    }
    counts: dict[int, dict[int, int]] = {int(layer): {} for layer in layers}
    for index, seed in enumerate(fit_seeds, start=1):
        row = rows_by_seed[int(seed)]
        encoding, registry = registry_builder(
            row, tokenizer, candidate_counts=candidate_counts
        )
        endpoints = tuple(int(end) - 1 for _start, end in registry.trace_items)
        captured = capture_decoder_block_input_states(
            model, adapter, encoding, endpoints, layers=layers
        )
        for layer in layers:
            for occurrence, state in enumerate(captured[int(layer)], start=1):
                active = state.detach().float().cpu()
                sums[int(layer)][occurrence] = (
                    sums[int(layer)].get(occurrence, torch.zeros_like(active)) + active
                )
                counts[int(layer)][occurrence] = (
                    counts[int(layer)].get(occurrence, 0) + 1
                )
        print(
            f"[linear-fit] {index}/{len(fit_seeds)} seed={seed}", flush=True
        )
    centroids = {
        layer: {
            occurrence: sums[layer][occurrence] / counts[layer][occurrence]
            for occurrence in sorted(sums[layer])
        }
        for layer in sorted(sums)
    }
    audit = {
        "fit_seeds": [int(value) for value in fit_seeds],
        "fit_seed_count": len(tuple(fit_seeds)),
        "layers": [int(value) for value in layers],
        "occurrences_by_layer": {
            str(layer): sorted(int(value) for value in centroids[layer])
            for layer in sorted(centroids)
        },
        "fit_rows_by_occurrence": {
            str(occurrence): int(count)
            for occurrence, count in sorted(next(iter(counts.values())).items())
        },
        "fit_uses_evaluation_outcomes": False,
        "geometry": "mean_block_input_endpoint_by_occurrence",
    }
    return centroids, audit


def _run_linear(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encoding: Any,
    registry: Any,
    target: Any,
    target_audit: Mapping[str, Any],
    centroids: Mapping[int, Mapping[int, torch.Tensor]],
    centroid_audit: Mapping[str, Any],
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    spec = config["experiments"]["linear_additivity"]
    if "layer_bands" in spec:
        layer_bands = tuple(
            (str(name), tuple(int(value) for value in values))
            for name, values in spec["layer_bands"].items()
        )
    else:
        active = tuple(int(value) for value in spec["layers"])
        layer_bands = (("registered", active),)
    if not layer_bands or any(not values for _name, values in layer_bands):
        raise ValueError("Linear-additivity layer bands must be nonempty")
    conditions = tuple(str(value) for value in spec["conditions"])
    alpha = float(spec.get("alpha", 1.0))
    for receiver in (int(value) for value in spec["receiver_occurrences"]):
        early, early_audit = build_immediate_count_query_encoding(
            target, registry, tokenizer, target_occurrence=receiver
        )
        endpoint = int(registry.trace_items[receiver - 1][1]) - 1
        baseline_prefill = _clean_prefill(
            model, adapter, early, audit_position=endpoint
        )
        for shift in (int(value) for value in spec["shifts"]):
            intended = receiver + shift
            if not 1 <= intended <= len(registry.trace_items):
                continue
            baseline = score_prefill(
                model,
                tokenizer,
                early,
                baseline_prefill,
                target_count=intended,
                original_count=receiver,
                run_greedy=False,
                max_new_tokens=int(config["max_new_tokens"]),
            )
            for band_name, active_layers in layer_bands:
                delta = layerwise_centroid_deltas(
                    {layer: centroids[layer] for layer in active_layers},
                    receiver_occurrence=receiver,
                    target_occurrence=intended,
                )
                for condition in conditions:
                    started = time.perf_counter()
                    if condition == "position_difference":
                        values = delta
                    elif condition == "opposite_position_difference":
                        values = {layer: -value for layer, value in delta.items()}
                    elif condition == "norm_matched_orthogonal":
                        values = {
                            layer: norm_matched_orthogonal_control(
                                value,
                                seed=(
                                    int(config["random_seed"])
                                    + int(encoding.seed) * 1009
                                    + int(layer) * 37
                                    + int(receiver) * 13
                                    + int(shift)
                                ),
                            )
                            for layer, value in delta.items()
                        }
                    else:
                        raise ValueError(
                            f"Unknown linear-additivity condition {condition}"
                        )
                    values = {
                        int(layer): float(alpha) * value
                        for layer, value in values.items()
                    }
                    prefill, _capture, applications, realized = (
                        prefill_with_block_input_intervention(
                            model,
                            adapter,
                            early,
                            positions=(endpoint,),
                            layer_values=values,
                            intervention_kind="add",
                        )
                    )
                    patched = score_prefill(
                        model,
                        tokenizer,
                        early,
                        prefill,
                        target_count=intended,
                        original_count=receiver,
                        run_greedy=bool(config["run_greedy"]),
                        max_new_tokens=int(config["max_new_tokens"]),
                    )
                    rows.append(
                        {
                            **_common(
                                experiment="linear_additivity",
                                condition=condition,
                                seed=int(encoding.seed),
                                encoding=encoding,
                                config=config,
                            ),
                            "receiver_occurrence": receiver,
                            "intended_occurrence": intended,
                            "position_difference": shift,
                            "steering_band": band_name,
                            "steering_layers": list(active_layers),
                            "steering_alpha": alpha,
                            "steering_token_region": "item_closing_endpoint",
                            "paper_ci": paper_causal_influence(
                                baseline,
                                patched,
                                expected_count=intended,
                                original_count=receiver,
                            ),
                            "baseline_target_probability": float(
                                baseline["target_probability"]
                            ),
                            "baseline_original_probability": float(
                                baseline["candidate_probabilities_by_count"][
                                    str(receiver)
                                ]
                            ),
                            "expected_count_shift_from_baseline": float(
                                patched["expected_count"]
                                - baseline["expected_count"]
                            ),
                            "donor_aligned_expected_shift": float(
                                (1 if shift > 0 else -1)
                                * (
                                    patched["expected_count"]
                                    - baseline["expected_count"]
                                )
                            ),
                            "elapsed_seconds": float(
                                time.perf_counter() - started
                            ),
                            **target_audit,
                            **early_audit,
                            "geometry_audit": centroid_audit,
                            **_hook_audit(applications, realized),
                            **patched,
                        }
                    )


def _run_separator(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encoding: Any,
    registry: Any,
    target: Any,
    target_audit: Mapping[str, Any],
    capture_positions: Sequence[int],
    captures: Mapping[int, torch.Tensor],
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> None:
    spec = config["experiments"]["separator_collapse"]
    source_setting = spec.get("source_occurrence", 1)
    target_setting = spec["target_occurrences"]
    target_group_mode = str(spec.get("target_group_mode", "all_later"))
    if str(source_setting) == "first_available_after_1":
        available_markers = tuple(
            occurrence
            for occurrence in range(2, len(registry.trace_items) + 1)
            if occurrence_region_positions_or_none(
                registry, occurrence, "marker"
            )
        )
        if len(available_markers) < 2:
            skipped_rows.append(
                {
                    **_common(
                        experiment="separator_collapse",
                        condition="collapse_registered_separator_markers",
                        seed=int(encoding.seed),
                        encoding=encoding,
                        config=config,
                    ),
                    "skip_reason": "fewer_than_two_registered_separator_markers",
                    "available_marker_occurrences": list(available_markers),
                    "selection_uses_current_experiment_outcomes": False,
                }
            )
            source = 2
            targets = tuple(range(3, len(registry.trace_items) + 1))
            separator_selection = (
                "pure_separator_unavailable_fixed_occurrence_controls_only"
            )
            pure_separator_available = False
        else:
            source = int(available_markers[0])
            if str(target_setting) != "all_later_available_markers":
                raise ValueError(
                    "Adaptive separator source requires all later available markers"
                )
            targets = tuple(int(value) for value in available_markers[1:])
            separator_selection = "format_only_first_and_later_registered_markers"
            pure_separator_available = True
    else:
        source = int(source_setting)
        targets = tuple(int(value) for value in target_setting)
        separator_selection = "fixed_occurrence_registry"
        pure_separator_available = True
    if target_group_mode == "all_later":
        target_groups = (targets,)
    elif target_group_mode == "singleton_each_later":
        target_groups = tuple((value,) for value in targets)
    elif target_group_mode == "prefix_doses":
        target_groups = tuple(targets[:index] for index in range(1, len(targets) + 1))
    else:
        raise ValueError(f"Unknown separator target-group mode: {target_group_mode}")
    if not target_groups or any(not values for values in target_groups):
        raise ValueError("Separator target groups must be nonempty")
    immediate, immediate_audit = build_immediate_count_query_encoding(
        target,
        registry,
        tokenizer,
        target_occurrence=len(registry.trace_items),
    )
    baseline_prefill = _clean_prefill(
        model,
        adapter,
        immediate,
        audit_position=int(registry.trace_items[0][1]) - 1,
    )
    baseline = score_prefill(
        model,
        tokenizer,
        immediate,
        baseline_prefill,
        target_count=len(registry.trace_items),
        original_count=len(registry.trace_items),
        run_greedy=bool(config["run_greedy"]),
        max_new_tokens=int(config["max_new_tokens"]),
    )
    for region in (str(value) for value in spec["regions"]):
        if region == "marker" and not pure_separator_available:
            continue
        for active_targets in target_groups:
            started = time.perf_counter()
            condition = f"collapse_{region}_to_occurrence_{source}"
            if target_group_mode != "all_later":
                condition += f"_dose_{len(active_targets)}"
            try:
                source_group = occurrence_region_positions(registry, source, region)
                source_groups = tuple(source_group for _ in active_targets)
                receiver_groups = occurrence_region_groups(
                    registry, active_targets, region
                )
            except ValueError as error:
                skipped_rows.append(
                    {
                        **_common(
                            experiment="separator_collapse",
                            condition=condition,
                            seed=int(encoding.seed),
                            encoding=encoding,
                            config=config,
                        ),
                        "source_occurrence": source,
                        "target_occurrences": list(active_targets),
                        "separator_target_group_mode": target_group_mode,
                        "separator_target_group_size": len(active_targets),
                        "region": region,
                        "skip_reason": "registered_separator_region_unavailable",
                        "skip_detail": str(error),
                        "selection_uses_current_experiment_outcomes": False,
                    }
                )
                continue
            alignment = align_region_groups(source_groups, receiver_groups)
            replacements = gather_aligned_states(
                captures, capture_positions, alignment.donor_positions
            )
            prefill, _capture, applications, realized = (
                prefill_with_block_input_intervention(
                    model,
                    adapter,
                    immediate,
                    positions=alignment.receiver_positions,
                    layer_values=replacements,
                    intervention_kind="replace",
                    norm_rescale_replacement=True,
                )
            )
            patched = score_prefill(
                model,
                tokenizer,
                immediate,
                prefill,
                target_count=len(registry.trace_items),
                original_count=len(registry.trace_items),
                run_greedy=bool(config["run_greedy"]),
                max_new_tokens=int(config["max_new_tokens"]),
            )
            rows.append(
                {
                    **_common(
                        experiment="separator_collapse",
                        condition=condition,
                        seed=int(encoding.seed),
                        encoding=encoding,
                        config=config,
                    ),
                    "source_occurrence": source,
                    "target_occurrences": list(active_targets),
                    "separator_occurrence_selection": separator_selection,
                    "separator_target_group_mode": target_group_mode,
                    "separator_target_group_size": len(active_targets),
                    "region": region,
                    "correct_probability_drop": float(
                        baseline["target_probability"]
                        - patched["target_probability"]
                    ),
                    "expected_count_shift": float(
                        patched["expected_count"] - baseline["expected_count"]
                    ),
                    "baseline_predicted_count": int(
                        baseline["predicted_count_among_candidates"]
                    ),
                    "baseline_greedy_prediction": baseline.get("greedy_prediction"),
                    "elapsed_seconds": float(time.perf_counter() - started),
                    **alignment.to_dict(),
                    **target_audit,
                    **immediate_audit,
                    **_hook_audit(applications, realized),
                    **patched,
                }
            )


def _run_maximum(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encoding: Any,
    registry: Any,
    target: Any,
    target_audit: Mapping[str, Any],
    capture_positions: Sequence[int],
    captures: Mapping[int, torch.Tensor],
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    spec = config["experiments"]["maximum_count"]
    early_cache: dict[int, tuple[Any, Any, Mapping[str, Any]]] = {}
    for pair in spec["source_target_pairs"]:
        source_end, target_end = (int(pair[0]), int(pair[1]))
        if target_end not in early_cache:
            early, early_audit = build_immediate_count_query_encoding(
                target, registry, tokenizer, target_occurrence=target_end
            )
            endpoint = int(registry.trace_items[target_end - 1][1]) - 1
            baseline_prefill = _clean_prefill(
                model, adapter, early, audit_position=endpoint
            )
            early_cache[target_end] = (early, baseline_prefill, early_audit)
        early, baseline_prefill, early_audit = early_cache[target_end]
        for k in (int(value) for value in spec["k_values"]):
            if source_end < k or target_end < k:
                continue
            source_occurrences = tuple(range(source_end - k + 1, source_end + 1))
            receiver_occurrences = tuple(range(target_end - k + 1, target_end + 1))
            expected = maximum_latent_count_expected(source_end, target_end, k)
            baseline = score_prefill(
                model,
                tokenizer,
                early,
                baseline_prefill,
                target_count=expected,
                original_count=target_end,
                run_greedy=False,
                max_new_tokens=int(config["max_new_tokens"]),
            )
            for region in (str(value) for value in spec["regions"]):
                started = time.perf_counter()
                alignment = align_occurrence_regions(
                    registry,
                    source_occurrences=source_occurrences,
                    receiver_occurrences=receiver_occurrences,
                    region=region,
                )
                replacements = gather_aligned_states(
                    captures, capture_positions, alignment.donor_positions
                )
                prefill, _capture, applications, realized = (
                    prefill_with_block_input_intervention(
                        model,
                        adapter,
                        early,
                        positions=alignment.receiver_positions,
                        layer_values=replacements,
                        intervention_kind="replace",
                        norm_rescale_replacement=True,
                    )
                )
                patched = score_prefill(
                    model,
                    tokenizer,
                    early,
                    prefill,
                    target_count=expected,
                    original_count=target_end,
                    run_greedy=bool(config["run_greedy"]),
                    max_new_tokens=int(config["max_new_tokens"]),
                )
                rows.append(
                    {
                        **_common(
                            experiment="maximum_count",
                            condition="last_k_source_to_last_k_target",
                            seed=int(encoding.seed),
                            encoding=encoding,
                            config=config,
                        ),
                        "source_end_occurrence": source_end,
                        "target_end_occurrence": target_end,
                        "source_occurrences": list(source_occurrences),
                        "target_occurrences": list(receiver_occurrences),
                        "k": k,
                        "region": region,
                        "maximum_latent_count_expected": expected,
                        "copy_source_hypothesis": source_end,
                        "target_minus_k_hypothesis": target_end - k,
                        "paper_ci": paper_causal_influence(
                            baseline,
                            patched,
                            expected_count=expected,
                            original_count=target_end,
                        ),
                        "baseline_target_probability": float(
                            baseline["target_probability"]
                        ),
                        "baseline_original_probability": float(
                            baseline["candidate_probabilities_by_count"][str(target_end)]
                        ),
                        "elapsed_seconds": float(time.perf_counter() - started),
                        **alignment.to_dict(),
                        **target_audit,
                        **early_audit,
                        **_hook_audit(applications, realized),
                        **patched,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.config)
    selection_mode = str(config.get("row_selection", "unique_seed"))
    registry_builder = (
        build_first_pass_tstar_answer_source_registry
        if selection_mode == "first_pass_noindex"
        else build_answer_source_registry
    )
    rows_by_seed = _read_rows(
        args.generations,
        selection_mode=selection_mode,
    )
    active_experiments = tuple(
        args.experiments
        or [
            name
            for name in EXPERIMENTS
            if bool(config["experiments"].get(name, {}).get("enabled", False))
        ]
    )
    if not active_experiments:
        raise ValueError("No counting-mechanism experiment is enabled")
    candidate_counts = tuple(int(value) for value in config["candidate_counts"])

    required_seeds: set[int] = set()
    experiment_seeds: dict[str, tuple[int, ...]] = {}
    for experiment in active_experiments:
        values = _experiment_seeds(config, experiment, args.seeds)
        experiment_seeds[experiment] = values
        required_seeds.update(values)
    if "linear_additivity" in active_experiments:
        fit = tuple(
            int(value)
            for value in config["experiments"]["linear_additivity"]["fit_seeds"]
        )
        required_seeds.update(fit)
    missing = sorted(required_seeds - set(rows_by_seed))
    if missing:
        raise ValueError(f"Generation file lacks required seeds: {missing}")

    model, tokenizer, adapter = _model(args)
    linear_centroids: dict[int, dict[int, torch.Tensor]] | None = None
    linear_audit: dict[str, Any] | None = None
    if "linear_additivity" in active_experiments:
        linear_spec = config["experiments"]["linear_additivity"]
        if "layer_bands" in linear_spec:
            geometry_layers = tuple(
                sorted(
                    {
                        int(layer)
                        for values in linear_spec["layer_bands"].values()
                        for layer in values
                    }
                )
            )
        else:
            geometry_layers = tuple(int(value) for value in linear_spec["layers"])
        if not geometry_layers:
            raise ValueError("Linear-additivity config selects no layers")
        linear_centroids, linear_audit = _fit_position_centroids(
            model,
            tokenizer,
            adapter,
            rows_by_seed,
            fit_seeds=tuple(int(value) for value in linear_spec["fit_seeds"]),
            layers=geometry_layers,
            candidate_counts=candidate_counts,
            registry_builder=registry_builder,
        )

    all_evaluation_seeds = sorted(
        {seed for values in experiment_seeds.values() for seed in values}
    )
    output_rows = (
        _read_existing_trials(args.output / "trials.jsonl") if args.resume else []
    )
    skipped_rows = (
        _read_existing_trials(args.output / "skipped_trials.jsonl")
        if args.resume
        else []
    )
    completed_seeds = {int(row["seed"]) for row in output_rows}
    if not completed_seeds <= set(all_evaluation_seeds):
        raise ValueError("Resume checkpoint contains seeds outside this run")
    expected_phase = str(config.get("phase", "discovery"))
    if any(str(row.get("phase")) != expected_phase for row in output_rows):
        raise ValueError("Resume checkpoint phase differs from the active config")
    evaluation_seeds = [
        seed for seed in all_evaluation_seeds if seed not in completed_seeds
    ]
    if args.resume:
        print(
            f"[counting-transfer] resume completed={len(completed_seeds)} "
            f"remaining={len(evaluation_seeds)}",
            flush=True,
        )
    for seed_index, seed in enumerate(evaluation_seeds, start=1):
        row = rows_by_seed[int(seed)]
        encoding, registry = registry_builder(
            row,
            tokenizer,
            candidate_counts=candidate_counts,
        )
        target, target_audit = prompt_scrubbed_encoding(
            encoding,
            registry,
            tokenizer,
            random_seed=int(config["random_seed"]) + int(seed) * 101,
        )
        needs_full_capture = any(
            experiment in active_experiments
            and seed in experiment_seeds[experiment]
            for experiment in (
                "countscope",
                "continued_counting",
                "separator_collapse",
                "maximum_count",
            )
        )
        capture_positions: tuple[int, ...] = ()
        captures: dict[int, torch.Tensor] = {}
        if needs_full_capture:
            capture_positions, captures = _all_trace_capture(
                model, adapter, encoding, registry
            )
        if "countscope" in active_experiments and seed in experiment_seeds["countscope"]:
            _run_countscope(
                model,
                tokenizer,
                adapter,
                encoding,
                registry,
                capture_positions,
                captures,
                config,
                output_rows,
            )
        if (
            "continued_counting" in active_experiments
            and seed in experiment_seeds["continued_counting"]
        ):
            _run_continued(
                model,
                tokenizer,
                adapter,
                encoding,
                registry,
                target,
                target_audit,
                capture_positions,
                captures,
                config,
                output_rows,
            )
        if (
            "linear_additivity" in active_experiments
            and seed in experiment_seeds["linear_additivity"]
        ):
            assert linear_centroids is not None and linear_audit is not None
            _run_linear(
                model,
                tokenizer,
                adapter,
                encoding,
                registry,
                target,
                target_audit,
                linear_centroids,
                linear_audit,
                config,
                output_rows,
            )
        if (
            "separator_collapse" in active_experiments
            and seed in experiment_seeds["separator_collapse"]
        ):
            _run_separator(
                model,
                tokenizer,
                adapter,
                encoding,
                registry,
                target,
                target_audit,
                capture_positions,
                captures,
                config,
                output_rows,
                skipped_rows,
            )
        if "maximum_count" in active_experiments and seed in experiment_seeds["maximum_count"]:
            _run_maximum(
                model,
                tokenizer,
                adapter,
                encoding,
                registry,
                target,
                target_audit,
                capture_positions,
                captures,
                config,
                output_rows,
            )
        _atomic_jsonl(args.output / "trials.jsonl", output_rows)
        _atomic_jsonl(args.output / "skipped_trials.jsonl", skipped_rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[counting-transfer] seed {seed_index}/{len(evaluation_seeds)} "
            f"seed={seed} rows={len(output_rows)}",
            flush=True,
        )

    _atomic_jsonl(args.output / "trials.jsonl", output_rows)
    _atomic_jsonl(args.output / "skipped_trials.jsonl", skipped_rows)
    experiment_counts = {
        experiment: sum(row["experiment"] == experiment for row in output_rows)
        for experiment in active_experiments
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "phase": str(config.get("phase", "discovery")),
        "claim_scope": "exploratory paper-method transfer to unindexed native thinking",
        "model": str(args.model),
        "active_experiments": list(active_experiments),
        "experiment_seeds": {
            name: list(values) for name, values in experiment_seeds.items()
        },
        "experiment_row_counts": experiment_counts,
        "trial_count": len(output_rows),
        "resumed_from_seed_count": len(completed_seeds),
        "skipped_trial_count": len(skipped_rows),
        "skipped_trials_path": "skipped_trials.jsonl",
        "candidate_counts": list(candidate_counts),
        "generation_row_selection": selection_mode,
        "selected_gold_counts_by_seed": {
            str(seed): int(row["gold_count"])
            for seed, row in sorted(rows_by_seed.items())
        },
        "run_greedy": bool(config["run_greedy"]),
        "elapsed_seconds": float(time.perf_counter() - started),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "generations_path": str(args.generations.resolve()),
        "generations_sha256": _sha256(args.generations),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "linear_geometry_audit": linear_audit,
        "scientific_guards": {
            "explicit_item_indices_absent_from_model_input": True,
            "source_generation_may_have_post_tstar_indices": (
                True if selection_mode == "first_pass_noindex" else None
            ),
            "prompt_records_scrubbed_for_behavioral_readouts": True,
            "countscope_receiver_semantics_scrubbed": True,
            "paper_ci_includes_one_half_factor": True,
            "argmax_and_greedy_reported": bool(config["run_greedy"]),
            "k1_reported_separately_from_k2_k3": True,
            "multi_token_success_does_not_imply_single_state_recurrence": True,
            "first_pass_tstar_only": selection_mode == "first_pass_noindex",
            "future_recap_available_to_context": (
                False if selection_mode == "first_pass_noindex" else None
            ),
            "internal_k_is_within_seed_and_not_outcome_selected": True,
        },
    }
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
