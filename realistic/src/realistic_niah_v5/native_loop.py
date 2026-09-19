"""Minimal causal loop experiments for native-thinking counting.

This module intentionally implements only the two missing experiment families
in the registered mechanism chain:

``P0 count state -> routed targeted retrieval`` and
``item endpoint state -> next retrieval / continue-stop``.

The targeted head bank is never selected here.  It must be supplied as a
previously frozen model-specific JSON artifact.  Pair plans contain every
registered seed in each eligible count/offset cell and deliberately contain no
``selection_rank`` field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _attention_tensor,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _extract_attentions,
    _extract_shared_kv_states,
    _replace_output_tensor,
    _temporary_attention_backend,
    _tensor_from_output,
    capture_post_block_states,
)
from realistic_niah_v4_4_3.interventions import (
    clone_prefill_output_for_scoring,
)
from realistic_niah_v4_4_5.restoration import (
    generate_answer_completion_from_prefill,
)

from .causal import (
    _first_generated_city_record,
    completion_metrics,
    mechanism_continuations,
)
from .count_stream import (
    COUNT_STREAM_SCHEMA_VERSION,
    _fixed_state_transform,
    _full_state_patch_layers,
    _prefill_with_state_replacements,
    _prefix_with_layerwise_state_replacements,
    _prefix_forward,
    _query_forward_from_prefix,
    _query_from_prefix_with_head_ablation,
    _score_trace_continuation,
    _sha256_json,
    _validate_head_bank,
    build_html_aligned_uninformative_trace_encoding,
    build_answer_source_registry,
    trace_patch_geometry_positions,
    valid_trace_patch_receivers,
)
from .encoding import NativeTraceEncoding, build_native_causal_encoding
from .parsing import (
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    output_token_ids,
    raw_output_text,
)


NATIVE_LOOP_SCHEMA_VERSION = "realistic_niah_v5_native_loop_v1"

REGISTERED_P0_LOOP_CONDITIONS = (
    "clean",
    "self_patch",
    "full_donor_patch",
    "count_subspace_transplant",
    "norm_matched_orthogonal_patch",
    "count_component_removed",
    "count_component_restored",
)

# Optional full-state controls for the commit -> next-query specificity
# extension.  They are deliberately kept out of REGISTERED_P0_LOOP_CONDITIONS
# so the sealed historical experiment and its default CLI remain unchanged.
REGISTERED_FULL_COMMIT_SPECIFICITY_CONDITIONS = (
    "full_delta_norm_matched_orthogonal_r0",
    "full_delta_norm_matched_orthogonal_r1",
    "full_delta_norm_matched_orthogonal_r2",
    "opposite_full_delta_patch",
    "shuffled_natural_donor_patch",
)

AVAILABLE_P0_LOOP_CONDITIONS = (
    *REGISTERED_P0_LOOP_CONDITIONS,
    *REGISTERED_FULL_COMMIT_SPECIFICITY_CONDITIONS,
)

REGISTERED_BOUNDARY_CONDITIONS = (
    "clean",
    "self_patch",
    "full_donor_patch",
    "count_subspace_transplant",
    "norm_matched_orthogonal_patch",
)

REGISTERED_QUERY_MEDIATION_HEAD_CONDITIONS = (
    "intact",
    "selected_mask",
    "layer_matched_random_mask",
)

REGISTERED_QUERY_MEDIATION_GEOMETRIES = (
    "endpoint",
    "suffix4",
    "suffix8",
    "suffix_cap4",
    "suffix_cap8",
)

REGISTERED_QUERY_MEDIATION_ENDPOINT_STATES = (
    "self_patch",
    "full_donor_patch",
    "count_subspace_transplant",
    "norm_matched_orthogonal_patch",
)

REGISTERED_QUERY_MEDIATION_SPAN_STATES = (
    "self_patch",
    "full_donor_patch",
)

REGISTERED_HTML_LOCAL_SERIAL_STATES = (
    "clean",
    "uninformative",
    "clean_target_ablation",
    "uninformative_target_restore",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registered_bank_membership_hashes(
    heads: Sequence[Sequence[int]],
) -> dict[str, str]:
    """Return both hash encodings used by frozen V5 head-bank artifacts.

    Historical targeted-selection files hashed ``json.dumps(heads)`` with its
    default separators.  Newer count-stream artifacts use compact canonical
    JSON.  The underlying ordered membership is identical, so loaders must
    audit against the artifact's declared encoding rather than silently
    treating whitespace as a membership change.
    """

    normalized = [[int(value[0]), int(value[1])] for value in heads]
    legacy_payload = json.dumps(normalized).encode("utf-8")
    return {
        "legacy_json_default": hashlib.sha256(legacy_payload).hexdigest(),
        "canonical_compact_json": _sha256_json(normalized),
    }


def load_frozen_targeted_bank(
    selection_path: str | Path,
    routing_path: str | Path,
    *,
    model_label: str,
) -> dict[str, Any]:
    """Load and cross-audit a frozen targeted bank and grammar routing."""

    selection_file = Path(selection_path)
    routing_file = Path(routing_path)
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    routing = json.loads(routing_file.read_text(encoding="utf-8"))
    if str(selection.get("model_label")) != str(model_label):
        raise ValueError("Targeted selection belongs to another model")
    development = selection.get("development_selection")
    if not isinstance(development, Mapping):
        raise ValueError("Targeted selection has no development_selection")
    heads = tuple(
        (int(value[0]), int(value[1]))
        for value in development.get("primary_bank_heads", ())
    )
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("Frozen targeted bank must contain unique heads")
    if int(development.get("primary_bank_size", -1)) != len(heads):
        raise ValueError("Frozen targeted bank size disagrees with membership")
    expected_sha = str(development.get("primary_bank_sha256", ""))
    membership_hashes = _registered_bank_membership_hashes(heads)
    matching_encodings = sorted(
        name for name, value in membership_hashes.items() if value == expected_sha
    )
    if not matching_encodings:
        raise ValueError("Frozen targeted bank membership hash mismatch")
    routed_head_bank = routing.get("head_bank")
    if not isinstance(routed_head_bank, Mapping):
        raise ValueError("Targeted routing has no head_bank")
    if str(routed_head_bank.get("selected_bank_sha256")) != expected_sha:
        raise ValueError("Selection and routing bank hashes disagree")
    routes = routing.get("routes")
    if not isinstance(routes, Mapping) or not routes:
        raise ValueError("Targeted routing has no grammar routes")
    for grammar, route in routes.items():
        if not isinstance(route, Mapping):
            raise ValueError(f"Invalid route for grammar {grammar}")
        required = tuple(str(value) for value in route.get("required", ()))
        if len(required) != 1:
            raise ValueError(
                f"Native-loop routing requires one frozen role for {grammar}"
            )
    return {
        "model_label": str(model_label),
        "heads": heads,
        "bank_size": len(heads),
        "bank_sha256": expected_sha,
        "bank_hash_encoding": matching_encodings[0],
        "bank_membership_hashes": membership_hashes,
        "routes": {
            str(grammar): tuple(str(value) for value in route["required"])
            for grammar, route in routes.items()
        },
        "routing_policy_id": str(routing.get("policy_id")),
        "selection_path": str(selection_file.resolve()),
        "selection_file_sha256": _sha256_file(selection_file),
        "routing_path": str(routing_file.resolve()),
        "routing_file_sha256": _sha256_file(routing_file),
        "selection_rank_used": False,
    }


def build_query_mediation_head_plan(
    targeted_bank: Mapping[str, Any],
    candidate_heads: Sequence[Sequence[int]],
    *,
    source_layer: int,
    random_seed: int,
    candidate_source_sha256: str,
) -> dict[str, Any]:
    """Freeze one disjoint layer-matched control for query-local mediation."""

    source = int(source_layer)
    model_label = str(targeted_bank["model_label"])
    selected = tuple(
        (int(layer), int(head))
        for layer, head in targeted_bank["heads"]
        if int(layer) > source
    )
    if not selected:
        raise ValueError("No targeted head lies downstream of the source patch")
    if len(set(selected)) != len(selected):
        raise ValueError("Active targeted mediation bank contains duplicates")

    candidates_by_layer: dict[int, set[int]] = {}
    for raw_layer, raw_head in candidate_heads:
        layer = int(raw_layer)
        head = int(raw_head)
        candidates_by_layer.setdefault(layer, set()).add(head)
    selected_by_layer: dict[int, set[int]] = {}
    for layer, head in selected:
        selected_by_layer.setdefault(layer, set()).add(head)

    control: list[tuple[int, int]] = []
    for layer, selected_heads in sorted(selected_by_layer.items()):
        available = sorted(
            candidates_by_layer.get(layer, set()) - selected_heads
        )
        count = len(selected_heads)
        if len(available) < count:
            raise ValueError(
                f"L{layer} has {count} selected heads but only "
                f"{len(available)} disjoint controls"
            )
        layer_seed = int.from_bytes(
            hashlib.sha256(
                f"{model_label}:{source}:{layer}:{int(random_seed)}".encode(
                    "utf-8"
                )
            ).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(layer_seed)
        chosen = sorted(
            int(value)
            for value in rng.choice(
                np.asarray(available, dtype=np.int64),
                size=count,
                replace=False,
            )
        )
        control.extend((layer, head) for head in chosen)

    control_tuple = tuple(control)
    if set(control_tuple) & set(selected):
        raise RuntimeError("Layer-matched mediation control overlaps treatment")
    layer_counts = {
        str(layer): len(heads)
        for layer, heads in sorted(selected_by_layer.items())
    }
    selected_json = [[layer, head] for layer, head in selected]
    control_json = [[layer, head] for layer, head in control_tuple]
    return {
        "schema_version": "realistic_niah_v5_query_mediation_head_plan_v1",
        "model_label": model_label,
        "source_layer": source,
        "targeted_bank_size": int(targeted_bank["bank_size"]),
        "targeted_bank_sha256": str(targeted_bank["bank_sha256"]),
        "active_selected_heads": selected_json,
        "active_selected_size": len(selected_json),
        "active_selected_sha256": _sha256_json(selected_json),
        "layer_composition": layer_counts,
        "layer_matched_random_heads": control_json,
        "layer_matched_random_size": len(control_json),
        "layer_matched_random_sha256": _sha256_json(control_json),
        "random_control_overlap_count": 0,
        "random_seed": int(random_seed),
        "candidate_source_sha256": str(candidate_source_sha256),
        "selection_rank_used": False,
        "outcome_blind": True,
    }


def load_frozen_query_mediation_head_plan(
    path: str | Path,
    targeted_bank: Mapping[str, Any],
    *,
    model_label: str,
    source_layer: int,
) -> dict[str, Any]:
    """Cross-audit a frozen selected/random query-local head plan."""

    plan_file = Path(path)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if str(plan.get("model_label")) != str(model_label):
        raise ValueError("Query-mediation head plan belongs to another model")
    if int(plan.get("source_layer", -1)) != int(source_layer):
        raise ValueError("Query-mediation head plan uses another source layer")
    if bool(plan.get("selection_rank_used", True)):
        raise ValueError("Query-mediation head plan used selection_rank")
    if not bool(plan.get("outcome_blind", False)):
        raise ValueError("Query-mediation head plan is not outcome-blind")
    if str(plan.get("targeted_bank_sha256")) != str(
        targeted_bank["bank_sha256"]
    ):
        raise ValueError("Query-mediation plan targets another frozen bank")

    expected_selected = tuple(
        (int(layer), int(head))
        for layer, head in targeted_bank["heads"]
        if int(layer) > int(source_layer)
    )
    selected = tuple(
        (int(value[0]), int(value[1]))
        for value in plan.get("active_selected_heads", ())
    )
    random = tuple(
        (int(value[0]), int(value[1]))
        for value in plan.get("layer_matched_random_heads", ())
    )
    if selected != expected_selected:
        raise ValueError("Query-mediation selected heads changed membership/order")
    if not random or len(set(random)) != len(random):
        raise ValueError("Query-mediation random bank is empty or duplicated")
    if set(selected) & set(random):
        raise ValueError("Query-mediation random bank overlaps selected heads")
    selected_json = [[layer, head] for layer, head in selected]
    random_json = [[layer, head] for layer, head in random]
    if str(plan.get("active_selected_sha256")) != _sha256_json(selected_json):
        raise ValueError("Query-mediation selected-bank hash mismatch")
    if str(plan.get("layer_matched_random_sha256")) != _sha256_json(
        random_json
    ):
        raise ValueError("Query-mediation random-bank hash mismatch")
    selected_counts: dict[int, int] = {}
    random_counts: dict[int, int] = {}
    for layer, _head in selected:
        selected_counts[layer] = selected_counts.get(layer, 0) + 1
    for layer, _head in random:
        random_counts[layer] = random_counts.get(layer, 0) + 1
    if selected_counts != random_counts:
        raise ValueError("Query-mediation random bank is not layer matched")
    return {
        **plan,
        "active_selected_heads": selected,
        "layer_matched_random_heads": random,
        "plan_path": str(plan_file.resolve()),
        "plan_file_sha256": _sha256_file(plan_file),
    }


def build_fixed_native_loop_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
    seeds: Sequence[int],
    seed_role: str,
    donor_offsets: Sequence[int] = (-3, -2, -1, 1, 2, 3),
    candidate_counts: Sequence[int] = tuple(range(2, 11)),
    sampling_seed: int = 20260821,
    require_all_seeds_per_offset: bool = True,
    include_boundaries: bool = True,
) -> pd.DataFrame:
    """Build an outcome-blind, rank-free plan for loop and boundary trials.

    Each seed contributes one identity-hash-selected row for every signed
    offset.  The eligible count ranges are the previously agreed panels:
    ``abs(offset)=1`` uses counts 2..10, ``abs(offset)=2`` uses 4..10, and
    ``abs(offset)=3`` uses 5..10 (subject to a valid local donor/receiver).
    Boundary panels add one middle-to-terminal and one terminal-to-nonterminal
    transplant per seed.  This preserves 20/10 seed-level inference without
    pretending every parser-eligible seed owns every count cell.
    """

    label = str(model_label)
    role = str(seed_role)
    expected_seeds = tuple(sorted({int(value) for value in seeds}))
    offsets = tuple(int(value) for value in donor_offsets)
    counts = tuple(sorted({int(value) for value in candidate_counts}))
    if role not in {"development", "confirmation"}:
        raise ValueError("seed_role must be development or confirmation")
    if not expected_seeds:
        raise ValueError("Native-loop plan needs at least one seed")
    if not offsets or 0 in offsets or len(offsets) != len(set(offsets)):
        raise ValueError("donor_offsets must be unique and nonzero")

    registry: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in rows:
        seed = int(raw["seed"])
        count = int(raw.get("gold_count", len(gold_records(raw))))
        if seed not in set(expected_seeds) or count not in set(counts):
            continue
        row_model = raw.get("model_label")
        if row_model not in {None, label}:
            continue
        key = (seed, count)
        if key in registry:
            raise ValueError(f"Duplicate native-loop seed/count row {key}")
        observed_item_count = count
        trace_one_to_one = True
        trace_category = "unit_fixture_assumed_one_to_one"
        if isinstance(raw.get("raw_output_text"), str):
            parsed = find_trace_count_sequence(
                raw_output_text(raw),
                model_family=infer_model_family(raw),
                gold_records=gold_records(raw),
            )
            if not bool(parsed.detected):
                continue
            observed_item_count = int(parsed.item_count)
            trace_one_to_one = bool(parsed.trace_one_to_one)
            trace_category = str(parsed.trace_category)
        registry[key] = {
            "request_id": str(raw["request_id"]),
            "seed": seed,
            "gold_count": count,
            "alignment_pair_id": raw.get("alignment_pair_id"),
            "cross_model_exact_sample_alignment": bool(
                raw.get("cross_model_exact_sample_alignment", False)
            ),
            "observed_item_count": observed_item_count,
            "trace_one_to_one": trace_one_to_one,
            "trace_category": trace_category,
        }
    missing_seeds = sorted(
        set(expected_seeds) - {seed for seed, _count in registry}
    )
    if missing_seeds:
        raise ValueError(f"Native-loop registry lost seeds {missing_seeds}")

    plan_rows: list[dict[str, Any]] = []

    def priority(
        row: Mapping[str, Any], panel: str, offset: int, receiver: int
    ) -> str:
        if bool(row.get("cross_model_exact_sample_alignment")):
            if not row.get("alignment_pair_id"):
                raise ValueError("Aligned native-loop row lacks alignment_pair_id")
            payload = {
                "sampling_seed": int(sampling_seed),
                "seed_role": role,
                "panel_kind": str(panel),
                "seed": int(row["seed"]),
                "gold_count": int(row["gold_count"]),
                "donor_offset": int(offset),
                "receiver_occurrence": int(receiver),
                "alignment_pair_id": str(row["alignment_pair_id"]),
            }
        else:
            payload = {
                "sampling_seed": int(sampling_seed),
                "model_label": label,
                "seed_role": role,
                "panel_kind": str(panel),
                "seed": int(row["seed"]),
                "gold_count": int(row["gold_count"]),
                "donor_offset": int(offset),
                "receiver_occurrence": int(receiver),
                "request_id": str(row["request_id"]),
            }
        return _sha256_json(payload)

    def append_pair(
        *,
        row: Mapping[str, Any],
        panel: str,
        receiver: int,
        donor: int,
        digest: str,
    ) -> None:
        pair_identity = {
            "model_label": label,
            "seed_role": role,
            "panel_kind": str(panel),
            "request_id": str(row["request_id"]),
            "seed": int(row["seed"]),
            "gold_count": int(row["gold_count"]),
            "receiver_occurrence": int(receiver),
            "donor_occurrence": int(donor),
            "donor_offset": int(donor - receiver),
        }
        plan_rows.append(
            {
                "schema_version": NATIVE_LOOP_SCHEMA_VERSION,
                "experiment_id": "native_loop_pair_plan",
                **pair_identity,
                "donor_direction": (
                    "past_to_later_receiver"
                    if int(donor) < int(receiver)
                    else "future_to_earlier_receiver"
                ),
                "receiver_is_terminal": bool(
                    int(receiver) == int(row["gold_count"])
                ),
                "donor_is_terminal": bool(
                    int(donor) == int(row["gold_count"])
                ),
                "outcome_blind_priority_sha256": str(digest),
                "selection_input_fields": (
                    "sampling_seed,seed_role,panel_kind,seed,gold_count,"
                    "donor_offset,alignment_pair_id"
                    if bool(row.get("cross_model_exact_sample_alignment"))
                    else "sampling_seed,model_label,seed_role,panel_kind,seed,"
                    "gold_count,donor_offset,request_id"
                ),
                "alignment_pair_id": row.get("alignment_pair_id"),
                "cross_model_exact_sample_alignment": bool(
                    row.get("cross_model_exact_sample_alignment", False)
                ),
                "sampling_seed": int(sampling_seed),
                "selection_rank_used": False,
                "observed_item_count": int(row["observed_item_count"]),
                "trace_one_to_one": bool(row["trace_one_to_one"]),
                "trace_category": str(row["trace_category"]),
                "local_cohort_policy": (
                    "one_to_one_full_trace"
                    if bool(row["trace_one_to_one"])
                    and int(row["observed_item_count"])
                    == int(row["gold_count"])
                    else "partial_unique_local_transition_fallback"
                ),
                "pair_sha256": _sha256_json(pair_identity),
            }
        )

    minimum_count_by_distance = {1: 2, 2: 4, 3: 5}
    unknown_distances = sorted(
        {abs(offset) for offset in offsets} - set(minimum_count_by_distance)
    )
    if unknown_distances:
        raise ValueError(f"Unregistered native-loop distances {unknown_distances}")
    for seed in expected_seeds:
        for offset in offsets:
            candidates: list[tuple[str, dict[str, Any], int]] = []
            full_trace_candidates: list[tuple[str, dict[str, Any], int]] = []
            minimum_count = minimum_count_by_distance[abs(offset)]
            for (row_seed, count), row in registry.items():
                if row_seed != seed or count < minimum_count:
                    continue
                if str(row["trace_category"]) not in {
                    "one_to_one",
                    "partial_unique",
                    "unit_fixture_assumed_one_to_one",
                }:
                    continue
                for receiver in valid_trace_patch_receivers(
                    int(row["observed_item_count"]), offset
                ):
                    candidate = (
                        priority(row, "p0_local", offset, receiver),
                        row,
                        int(receiver),
                    )
                    candidates.append(candidate)
                    if bool(row["trace_one_to_one"]) and int(
                        row["observed_item_count"]
                    ) == int(row["gold_count"]):
                        full_trace_candidates.append(candidate)
            if full_trace_candidates:
                candidates = full_trace_candidates
            if not candidates and require_all_seeds_per_offset:
                raise ValueError(
                    f"Seed {seed} has no local candidate for offset {offset:+d}"
                )
            if not candidates:
                continue
            digest, row, receiver = min(candidates, key=lambda value: value[0])
            append_pair(
                row=row,
                panel="p0_local",
                receiver=receiver,
                donor=receiver + offset,
                digest=digest,
            )

        if not include_boundaries:
            continue
        terminal_candidates: list[tuple[str, dict[str, Any], int]] = []
        nonterminal_candidates: list[tuple[str, dict[str, Any], int]] = []
        for (row_seed, count), row in registry.items():
            if row_seed != seed:
                continue
            if not bool(row["trace_one_to_one"]) or int(
                row["observed_item_count"]
            ) != int(count):
                continue
            for receiver in range(2, count):
                terminal_candidates.append(
                    (
                        priority(row, "terminal_injection", count - receiver, receiver),
                        row,
                        int(receiver),
                    )
                )
            if count >= 2:
                nonterminal_candidates.append(
                    (
                        priority(row, "nonterminal_injection", -1, count),
                        row,
                        int(count),
                    )
                )
        if not terminal_candidates or not nonterminal_candidates:
            raise ValueError(f"Seed {seed} has no eligible boundary transplant")
        digest, row, receiver = min(
            terminal_candidates, key=lambda value: value[0]
        )
        append_pair(
            row=row,
            panel="terminal_injection",
            receiver=receiver,
            donor=int(row["gold_count"]),
            digest=digest,
        )
        digest, row, receiver = min(
            nonterminal_candidates, key=lambda value: value[0]
        )
        append_pair(
            row=row,
            panel="nonterminal_injection",
            receiver=receiver,
            donor=receiver - 1,
            digest=digest,
        )

    plan = pd.DataFrame(plan_rows)
    if plan.empty:
        raise ValueError("Native-loop plan is empty")
    if "selection_rank" in plan.columns:
        raise RuntimeError("Native-loop plan must never contain selection_rank")
    if plan["pair_sha256"].duplicated().any():
        raise RuntimeError("Native-loop plan emitted duplicate pairs")
    local = plan.loc[plan["panel_kind"].eq("p0_local")]
    if require_all_seeds_per_offset:
        for _offset, cell in local.groupby("donor_offset", dropna=False):
            if set(cell["seed"].astype(int)) != set(expected_seeds):
                raise RuntimeError(
                    "A native-loop signed offset lost a registered seed"
                )
    boundary = plan.loc[plan["panel_kind"].ne("p0_local")]
    for _panel, cell in boundary.groupby("panel_kind", dropna=False):
        if set(cell["seed"].astype(int)) != set(expected_seeds):
            raise RuntimeError("A native-loop boundary panel lost a registered seed")
    return plan.sort_values(
        ["panel_kind", "donor_offset", "seed", "gold_count"], kind="stable"
    ).reset_index(drop=True)


def native_loop_condition_states(
    receiver_state: np.ndarray | torch.Tensor,
    donor_state: np.ndarray | torch.Tensor,
    center: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
    *,
    random_seed: int,
) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
    """Construct count transplant, removal/restoration, and matched controls."""

    receiver = torch.as_tensor(receiver_state).detach().float().cpu().reshape(-1)
    donor = torch.as_tensor(donor_state).detach().float().cpu().reshape(-1)
    origin = torch.as_tensor(center).detach().float().cpu().reshape(-1)
    axes = torch.as_tensor(basis).detach().float().cpu()
    if receiver.shape != donor.shape or receiver.shape != origin.shape:
        raise ValueError("Receiver, donor, and center widths disagree")
    if axes.ndim != 2 or axes.shape[0] != receiver.numel():
        raise ValueError("Count basis must have shape [hidden, rank]")
    gram = axes.T @ axes
    if not torch.allclose(gram, torch.eye(axes.shape[1]), atol=2e-4, rtol=2e-4):
        raise ValueError("Count basis must be orthonormal")

    full_delta = donor - receiver
    count_delta = (full_delta @ axes) @ axes.T
    count_component = ((receiver - origin) @ axes) @ axes.T
    removed = receiver - count_component
    restored = removed + count_component

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    random = torch.randn(receiver.shape, generator=generator)
    random = random - (random @ axes) @ axes.T
    random_norm = float(torch.linalg.vector_norm(random))
    target_norm = float(torch.linalg.vector_norm(count_delta))
    if target_norm <= 1e-12:
        orthogonal = torch.zeros_like(random)
    elif random_norm <= 1e-12:
        raise RuntimeError("Could not construct an orthogonal control direction")
    else:
        orthogonal = random * (target_norm / random_norm)
    overlap = float(torch.max(torch.abs(orthogonal @ axes)))
    if overlap > max(1e-5, target_norm * 2e-5):
        raise RuntimeError("Native-loop control is not orthogonal")

    states: dict[str, torch.Tensor | None] = {
        "clean": None,
        "self_patch": receiver.clone(),
        "full_donor_patch": donor.clone(),
        "count_subspace_transplant": receiver + count_delta,
        "norm_matched_orthogonal_patch": receiver + orthogonal,
        "count_component_removed": removed,
        "count_component_restored": restored,
    }
    receiver_coordinate = (receiver - origin) @ axes
    donor_coordinate = (donor - origin) @ axes
    coordinate_delta = donor_coordinate - receiver_coordinate
    denominator = float(torch.dot(coordinate_delta, coordinate_delta))
    condition_audit: dict[str, dict[str, Any]] = {}
    for name, state in states.items():
        active = receiver if state is None else state
        coordinate = (active - origin) @ axes
        fraction = (
            float(torch.dot(coordinate - receiver_coordinate, coordinate_delta))
            / denominator
            if denominator > 1e-12
            else 0.0
        )
        condition_audit[name] = {
            "condition_patch_delta_norm": float(
                torch.linalg.vector_norm(active - receiver)
            ),
            "condition_count_coordinate": coordinate.tolist(),
            "condition_target_count_fraction": fraction,
            "condition_distance_to_receiver_count_coordinate": float(
                torch.linalg.vector_norm(coordinate - receiver_coordinate)
            ),
            "condition_distance_to_donor_count_coordinate": float(
                torch.linalg.vector_norm(coordinate - donor_coordinate)
            ),
        }
    return states, {
        "basis_rank": int(axes.shape[1]),
        "receiver_count_coordinate": receiver_coordinate.tolist(),
        "donor_count_coordinate": donor_coordinate.tolist(),
        "full_donor_delta_norm": float(torch.linalg.vector_norm(full_delta)),
        "count_subspace_delta_norm": target_norm,
        "orthogonal_control_delta_norm": float(
            torch.linalg.vector_norm(orthogonal)
        ),
        "orthogonal_control_count_max_abs_coordinate": overlap,
        "removed_count_component_norm": float(
            torch.linalg.vector_norm(count_component)
        ),
        "restoration_identity_max_abs_error": float(
            torch.max(torch.abs(restored - receiver))
        ),
        "condition_audit": condition_audit,
    }


def choose_shuffled_commit_donor_occurrence(
    *,
    gold_count: int,
    receiver_occurrence: int,
    donor_occurrence: int,
    random_seed: int,
) -> int:
    """Choose an outcome-blind natural commit with the wrong ordinal.

    A control donor must own a successor transition, so occurrence ``N`` is
    excluded. We first prefer the donor mirrored across the receiver because
    it matches absolute donor distance. If that state has no successor, a
    deterministic random choice is made among the closest-distance states.
    """

    count = int(gold_count)
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    if count < 4:
        raise ValueError("Full-commit specificity needs at least four items")
    if not 1 <= receiver < count or not 1 <= donor < count or donor == receiver:
        raise ValueError(
            "Receiver and donor must own distinct successor transitions"
        )
    candidates = [
        occurrence
        for occurrence in range(1, count)
        if occurrence not in {receiver, donor}
    ]
    if not candidates:
        raise ValueError("No natural shuffled commit donor is available")
    offset = donor - receiver
    mirrored = receiver - offset
    if mirrored in candidates:
        return int(mirrored)
    target_distance = abs(offset)
    best_mismatch = min(
        abs(abs(occurrence - receiver) - target_distance)
        for occurrence in candidates
    )
    finalists = sorted(
        occurrence
        for occurrence in candidates
        if abs(abs(occurrence - receiver) - target_distance) == best_mismatch
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    selected = int(torch.randint(len(finalists), (1,), generator=generator).item())
    return int(finalists[selected])


def full_commit_specificity_condition_states(
    receiver_state: np.ndarray | torch.Tensor,
    donor_state: np.ndarray | torch.Tensor,
    *,
    shuffled_donor_state: np.ndarray | torch.Tensor | None,
    random_seed: int,
    random_replicates: int = 3,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build complete-delta-matched and natural-donor controls.

    This construction makes no claim about a linear count subspace. Random
    controls match the complete donor-minus-receiver norm and are orthogonal
    to that realized transition. The opposite control is antipodal with the
    same norm. The shuffled control, when supplied, is an unmodified commit
    state from a different ordinal in the same natural trace.
    """

    receiver = torch.as_tensor(receiver_state).detach().float().cpu().reshape(-1)
    donor = torch.as_tensor(donor_state).detach().float().cpu().reshape(-1)
    if receiver.shape != donor.shape:
        raise ValueError("Receiver and donor widths disagree")
    full_delta = donor - receiver
    full_norm = float(torch.linalg.vector_norm(full_delta))
    if full_norm <= 1e-12:
        raise ValueError("Full donor delta is zero")
    unit_delta = full_delta / full_norm
    states: dict[str, torch.Tensor] = {
        "opposite_full_delta_patch": receiver - full_delta,
    }
    audits: dict[str, dict[str, Any]] = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    for replicate in range(int(random_replicates)):
        random = torch.randn(receiver.shape, generator=generator)
        random = random - torch.dot(random, unit_delta) * unit_delta
        random_norm = float(torch.linalg.vector_norm(random))
        if random_norm <= 1e-12:
            raise RuntimeError("Could not sample a full-delta orthogonal control")
        delta = random * (full_norm / random_norm)
        name = f"full_delta_norm_matched_orthogonal_r{replicate}"
        states[name] = receiver + delta
    shuffled = None
    if shuffled_donor_state is not None:
        shuffled = (
            torch.as_tensor(shuffled_donor_state)
            .detach()
            .float()
            .cpu()
            .reshape(-1)
        )
        if shuffled.shape != receiver.shape:
            raise ValueError("Shuffled donor width disagrees")
        states["shuffled_natural_donor_patch"] = shuffled

    for name, state in states.items():
        delta = state - receiver
        norm = float(torch.linalg.vector_norm(delta))
        cosine = float(
            torch.dot(delta, full_delta) / max(norm * full_norm, 1e-12)
        )
        audits[name] = {
            "condition_patch_delta_norm": norm,
            "condition_full_donor_delta_norm_ratio": norm / full_norm,
            "condition_full_donor_delta_cosine": cosine,
            "condition_distance_to_full_donor": float(
                torch.linalg.vector_norm(state - donor)
            ),
            "condition_is_natural_commit_state": bool(
                name == "shuffled_natural_donor_patch"
            ),
        }
    random_names = [
        name
        for name in states
        if name.startswith("full_delta_norm_matched_orthogonal_r")
    ]
    max_random_cosine = max(
        abs(audits[name]["condition_full_donor_delta_cosine"])
        for name in random_names
    )
    max_random_norm_error = max(
        abs(audits[name]["condition_full_donor_delta_norm_ratio"] - 1.0)
        for name in random_names
    )
    if max_random_cosine > 2e-5:
        raise RuntimeError("Full-delta random control is not orthogonal")
    if max_random_norm_error > 2e-5:
        raise RuntimeError("Full-delta random control is not norm matched")
    return states, {
        "full_donor_delta_norm": full_norm,
        "full_delta_random_replicates": int(random_replicates),
        "full_delta_random_max_abs_cosine": max_random_cosine,
        "full_delta_random_max_relative_norm_error": max_random_norm_error,
        "shuffled_donor_delta_norm": (
            None
            if shuffled is None
            else float(torch.linalg.vector_norm(shuffled - receiver))
        ),
        "condition_audit": audits,
    }


def _query_from_prefix_with_attentions(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    prefix_output: Any,
    *,
    query_position: int,
    replacement_state: torch.Tensor | None,
    replacement_layer: int,
) -> tuple[list[torch.Tensor], list[int], int]:
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    query = int(query_position)
    past = getattr(prefix_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Routed attention prefix returned no KV cache")
    applications = 0
    handle = None
    if replacement_state is not None:
        fixed = replacement_state.detach().float().reshape(1, 1, -1)

        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            nonlocal applications
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or int(hidden.shape[1]) != 1:
                return output
            if int(hidden.shape[-1]) != int(fixed.shape[-1]):
                raise RuntimeError("Routed query patch width disagrees with model")
            applications += 1
            return _replace_output_tensor(
                output,
                fixed.to(device=hidden.device, dtype=hidden.dtype).expand_as(hidden),
            )

        handle = adapter.layers[int(replacement_layer)].register_forward_hook(hook)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": attention_mask[:, : query + 1],
        "past_key_values": past,
        "use_cache": False,
        "output_attentions": True,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.tensor(
            [[query]], dtype=torch.long, device=input_ids.device
        )
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.tensor(
            [query], dtype=torch.long, device=input_ids.device
        )
    shared = _extract_shared_kv_states(prefix_output)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = shared
    try:
        with _temporary_attention_backend(model, "eager"):
            output = model(**kwargs)
    finally:
        if handle is not None:
            handle.remove()
    if replacement_state is not None and applications != 1:
        raise RuntimeError(
            f"Routed query patch must apply once, observed {applications}"
        )
    attentions = _extract_attentions(output)
    if len(attentions) != int(adapter.num_layers):
        raise RuntimeError("Routed query returned the wrong attention layer count")
    rows: list[torch.Tensor] = []
    key_starts: list[int] = []
    for layer, value in enumerate(attentions):
        tensor = _attention_tensor(value)
        if int(tensor.shape[0]) != 1 or int(tensor.shape[2]) != 1:
            raise RuntimeError(f"L{layer} did not return one routed query row")
        row = tensor[0, :, 0].detach().float().cpu()
        rows.append(row)
        key_starts.append(query + 1 - int(row.shape[-1]))
    return rows, key_starts, applications


def _routed_targeted_attention_metrics(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_specification: Mapping[str, Any],
    receiver_position: int,
    source_layer: int,
    replacement_state: torch.Tensor | None,
    targeted_bank: Mapping[str, Any],
    donor_successor_city: str | None,
) -> dict[str, Any]:
    """Measure frozen-bank prompt-record ordinal after a P0 intervention."""

    grammar_pair = str(receiver_specification.get("grammar_pair", ""))
    grammar = grammar_pair.split(" -> ")[-1]
    routes = targeted_bank["routes"]
    if grammar not in routes:
        raise ValueError(f"No frozen targeted route for grammar {grammar}")
    required_role = str(tuple(routes[grammar])[0])
    route_specs, _excluded = mechanism_continuations(
        row, tokenizer, mechanism="retrieval_anchor_localization"
    )
    candidates = []
    receiver_occurrence = int(receiver_specification["from_occurrence"])
    for value in route_specs:
        roles = {str(role) for role in value.get("anchor_roles", ())}
        if (
            int(value["from_occurrence"]) == receiver_occurrence
            and required_role in roles
        ):
            candidates.append(dict(value))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one routed {required_role} anchor for occurrence "
            f"{receiver_occurrence}, found {len(candidates)}"
        )
    route_spec = candidates[0]
    route_query_output = int(route_spec["query_output_token_index"])
    route_encoding = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=route_query_output,
        sequence_output_token_end=route_query_output + 1,
        selected_site=route_spec,
    )
    query = int(route_encoding.query_position)
    source = int(receiver_position)
    if source > query:
        raise RuntimeError("Frozen retrieval query occurs before its P0 state")

    input_ids, attention_mask = _encoding_tensors(model, route_encoding)
    if source < query:
        prefix_encoding = replace(
            route_encoding,
            input_ids=tuple(int(value) for value in route_encoding.input_ids[:query]),
            attention_mask=tuple(
                int(value) for value in route_encoding.attention_mask[:query]
            ),
            query_position=query - 1,
        )
        prefix_output, source_applications, _realized = (
            _prefill_with_state_replacements(
                model,
                adapter,
                prefix_encoding,
                layer=int(source_layer),
                positions=(source,),
                states=(
                    None
                    if replacement_state is None
                    else replacement_state.reshape(1, -1)
                ),
            )
        )
        rows, key_starts, query_applications = _query_from_prefix_with_attentions(
            model,
            adapter,
            route_encoding,
            prefix_output,
            query_position=query,
            replacement_state=None,
            replacement_layer=int(source_layer),
        )
    else:
        prefix_output = _prefix_forward(
            model,
            adapter,
            input_ids[:, :query],
            attention_mask[:, :query],
        )
        source_applications = 0
        rows, key_starts, query_applications = _query_from_prefix_with_attentions(
            model,
            adapter,
            route_encoding,
            prefix_output,
            query_position=query,
            replacement_state=replacement_state,
            replacement_layer=int(source_layer),
        )

    ordered_gold = tuple(gold_records(row))
    ordinal_by_city = {
        str(record["city"]).casefold(): index
        for index, record in enumerate(ordered_gold, start=1)
    }
    span_by_city = {
        str(span.city).casefold(): span for span in route_encoding.prompt_record_spans
    }
    if set(ordinal_by_city) != set(span_by_city):
        raise RuntimeError("Prompt spans and gold city ordinals disagree")
    active_heads = tuple(
        (int(layer), int(head))
        for layer, head in targeted_bank["heads"]
        if int(layer) > int(source_layer)
    )
    if not active_heads:
        raise ValueError("No frozen targeted head lies downstream of P0 patch")
    mass_by_ordinal = {ordinal: 0.0 for ordinal in ordinal_by_city.values()}
    for layer, head in active_heads:
        if not 0 <= layer < len(rows) or not 0 <= head < int(rows[layer].shape[0]):
            raise ValueError(f"Invalid frozen targeted head L{layer}H{head}")
        attention = rows[layer][head]
        key_start = int(key_starts[layer])
        key_end = key_start + int(attention.shape[-1])
        for city, ordinal in ordinal_by_city.items():
            span = span_by_city[city]
            left = max(int(span.start), key_start)
            right = min(int(span.end), key_end)
            if right > left:
                mass_by_ordinal[ordinal] += float(
                    attention[left - key_start : right - key_start].sum().item()
                )
    total_mass = float(sum(mass_by_ordinal.values()))
    expected_ordinal = (
        float(
            sum(ordinal * mass for ordinal, mass in mass_by_ordinal.items())
            / total_mass
        )
        if total_mass > 0.0
        else np.nan
    )
    top_ordinal = int(
        max(mass_by_ordinal, key=lambda value: (mass_by_ordinal[value], -value))
    )
    receiver_city = str(receiver_specification["target_city"])
    receiver_ordinal = ordinal_by_city[receiver_city.casefold()]
    donor_ordinal = (
        None
        if donor_successor_city is None
        else ordinal_by_city[str(donor_successor_city).casefold()]
    )
    return {
        "targeted_route_grammar": grammar,
        "targeted_route_anchor_role": required_role,
        "targeted_route_query_output_token_index": route_query_output,
        "targeted_route_query_full_token_index": query,
        "targeted_route_source_before_query": bool(source < query),
        "targeted_bank_size": int(targeted_bank["bank_size"]),
        "targeted_bank_sha256": str(targeted_bank["bank_sha256"]),
        "targeted_bank_downstream_head_count": len(active_heads),
        "targeted_bank_prompt_record_mass": total_mass,
        "targeted_bank_expected_source_ordinal": expected_ordinal,
        "targeted_bank_top_source_ordinal": top_ordinal,
        "targeted_bank_source_masses": {
            str(key): float(value) for key, value in mass_by_ordinal.items()
        },
        "receiver_successor_source_ordinal": receiver_ordinal,
        "receiver_successor_attention_mass": float(
            mass_by_ordinal[receiver_ordinal]
        ),
        "donor_successor_source_ordinal": donor_ordinal,
        "donor_successor_attention_mass": (
            None if donor_ordinal is None else float(mass_by_ordinal[donor_ordinal])
        ),
        "donor_minus_receiver_successor_attention_mass": (
            None
            if donor_ordinal is None
            else float(
                mass_by_ordinal[donor_ordinal]
                - mass_by_ordinal[receiver_ordinal]
            )
        ),
        "targeted_attention_source_patch_applications": int(source_applications),
        "targeted_attention_query_patch_applications": int(query_applications),
    }


def _query_from_prefix_with_mediation_hooks(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    prefix_output: Any,
    query_attention_mask: torch.Tensor,
    source_layer: int,
    query_replacement_state: torch.Tensor | None,
    zero_heads: Sequence[tuple[int, int]] = (),
    capture_heads: Sequence[tuple[int, int]] = (),
    head_replacements: Mapping[int, torch.Tensor] | None = None,
) -> tuple[Any, dict[int, torch.Tensor], dict[str, Any]]:
    """Run one query with source-state and exact pre-O head interventions."""

    zero_grouped = _validate_head_bank(adapter, zero_heads)
    capture_grouped = _validate_head_bank(adapter, capture_heads)
    replacement_values = {
        int(layer): torch.as_tensor(value).detach().float().cpu()
        for layer, value in (head_replacements or {}).items()
    }
    replacement_grouped = (
        capture_grouped if replacement_values else {}
    )
    if replacement_values and set(replacement_values) != set(replacement_grouped):
        raise ValueError("Head replacements do not cover the captured bank layers")
    if set(zero_grouped) & set(replacement_grouped):
        raise ValueError("Cannot zero and restore the same mediation layer")

    source_applications = 0
    source_handle = None
    if query_replacement_state is not None:
        fixed = query_replacement_state.detach().float().reshape(1, 1, -1)

        def source_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            nonlocal source_applications
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or int(hidden.shape[1]) != 1:
                return output
            if int(hidden.shape[-1]) != int(fixed.shape[-1]):
                raise RuntimeError("Query-state replacement width disagrees")
            source_applications += 1
            return _replace_output_tensor(
                output,
                fixed.to(device=hidden.device, dtype=hidden.dtype).expand_as(
                    hidden
                ),
            )

        source_handle = adapter.layers[int(source_layer)].register_forward_hook(
            source_hook
        )

    captures: dict[int, torch.Tensor] = {}
    head_applications: dict[int, int] = {}
    handles = []
    intervention_layers = sorted(
        set(zero_grouped) | set(capture_grouped) | set(replacement_grouped)
    )
    for layer in intervention_layers:
        zero_layer_heads = zero_grouped.get(layer, ())
        capture_layer_heads = capture_grouped.get(layer, ())
        replacement_layer_heads = replacement_grouped.get(layer, ())
        width = int(adapter.head_dims[layer])
        expected = int(adapter.num_heads[layer]) * width
        head_applications[layer] = 0

        def head_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            zero_layer_heads: tuple[int, ...] = zero_layer_heads,
            capture_layer_heads: tuple[int, ...] = capture_layer_heads,
            replacement_layer_heads: tuple[int, ...] = replacement_layer_heads,
            width: int = width,
            expected: int = expected,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Query mediation o_proj received no tensor")
            value = args[0]
            if value.ndim != 3 or value.shape[1] != 1 or value.shape[-1] != expected:
                raise RuntimeError("Query mediation head hook saw an invalid shape")
            if capture_layer_heads:
                captures[layer] = value.detach().float().cpu().clone()
            patched = value.clone()
            for head in zero_layer_heads:
                left = int(head) * width
                patched[:, 0, left : left + width] = 0
            if replacement_layer_heads:
                replacement = replacement_values[layer].to(
                    device=value.device, dtype=value.dtype
                )
                if replacement.shape != value.shape:
                    raise RuntimeError("Query head replacement width disagrees")
                for head in replacement_layer_heads:
                    left = int(head) * width
                    patched[:, 0, left : left + width] = replacement[
                        :, 0, left : left + width
                    ]
            head_applications[layer] += 1
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(head_hook)
        )
    try:
        query_output = _query_forward_from_prefix(
            model,
            adapter,
            encoding,
            prefix_output=prefix_output,
            query_attention_mask=query_attention_mask,
        )
    finally:
        if source_handle is not None:
            source_handle.remove()
        for handle in handles:
            handle.remove()
    if query_replacement_state is not None and source_applications != 1:
        raise RuntimeError(
            "Query-state replacement must apply once, observed "
            f"{source_applications}"
        )
    bad_layers = sorted(
        layer for layer, count in head_applications.items() if count != 1
    )
    if bad_layers:
        raise RuntimeError(
            f"Query mediation head hooks did not apply once: {bad_layers}"
        )
    if capture_grouped and set(captures) != set(capture_grouped):
        raise RuntimeError("Query mediation did not capture every selected layer")
    return query_output, captures, {
        "query_state_patch_applications": int(source_applications),
        "query_head_hook_applications": {
            str(layer): int(count)
            for layer, count in sorted(head_applications.items())
        },
    }


def _query_city_outcomes(
    model: Any,
    tokenizer: Any,
    encoding: NativeTraceEncoding,
    query_output: Any,
    *,
    receiver_path: Sequence[int],
    donor_path: Sequence[int],
    city_token_offset: int,
    known_cities: Sequence[str],
    receiver_city: str,
    donor_city: str,
    run_greedy: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    receiver_score = _score_trace_continuation(
        model,
        encoding,
        query_output,
        receiver_path,
        city_token_offset=int(city_token_offset),
    )
    donor_score = _score_trace_continuation(
        model,
        encoding,
        query_output,
        donor_path,
        city_token_offset=int(city_token_offset),
    )
    outcomes: dict[str, Any] = {
        "receiver_query_path_log_probability": receiver_score[
            "sequence_log_probability"
        ],
        "donor_query_path_log_probability": donor_score[
            "sequence_log_probability"
        ],
        "donor_vs_receiver_query_path_log_odds": float(
            donor_score["sequence_log_probability"]
            - receiver_score["sequence_log_probability"]
        ),
        "receiver_query_city_log_probability": receiver_score[
            "city_log_probability"
        ],
        "donor_query_city_log_probability": donor_score["city_log_probability"],
        "donor_vs_receiver_query_city_log_odds": float(
            donor_score["city_log_probability"]
            - receiver_score["city_log_probability"]
        ),
    }
    if run_greedy:
        completion = generate_answer_completion_from_prefill(
            model,
            tokenizer,
            encoding,
            clone_prefill_output_for_scoring(query_output),
            max_new_tokens=int(max_new_tokens),
        )
        generated_city, city_start, city_evidence = _first_generated_city_record(
            str(completion["completion_text"]), known_cities
        )
        outcomes.update(
            {
                "query_completion_text": str(completion["completion_text"]),
                "query_generation_truncated": bool(
                    completion["generation_truncated"]
                ),
                "query_first_generated_city_record": generated_city,
                "query_first_generated_city_record_char_start": city_start,
                "query_first_generated_city_record_evidence": city_evidence,
                "query_donor_city_adoption": bool(
                    generated_city is not None
                    and generated_city.casefold() == donor_city.casefold()
                ),
                "query_receiver_city_retention": bool(
                    generated_city is not None
                    and generated_city.casefold() == receiver_city.casefold()
                ),
            }
        )
    return outcomes


def validate_query_mediation_positions(
    receiver_positions: Sequence[int],
    donor_positions: Sequence[int],
    *,
    query_position: int,
) -> dict[str, Any]:
    """Audit causal receiver positions while allowing offline future donors."""

    receivers = tuple(int(value) for value in receiver_positions)
    donors = tuple(int(value) for value in donor_positions)
    query = int(query_position)
    if not receivers or not donors or len(receivers) != len(donors):
        raise ValueError("Query mediation needs aligned donor/receiver positions")
    if max(receivers) > query:
        raise ValueError("A query-mediation receiver position lies after the query")
    return {
        "receiver_source_before_or_at_query": True,
        "donor_capture_is_offline_counterfactual": bool(max(donors) > query),
        "donor_positions_after_receiver_query_count": sum(
            position > query for position in donors
        ),
    }


@torch.inference_mode()
def run_p0_query_mediation_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    layer: int,
    geometry: str,
    center: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
    targeted_bank: Mapping[str, Any],
    head_plan: Mapping[str, Any],
    random_seed: int,
    run_greedy: bool = False,
    max_new_tokens: int = 32,
) -> list[dict[str, Any]]:
    """Cross one written P0 state with query-local targeted-head mediation."""

    name = str(geometry)
    if name not in REGISTERED_QUERY_MEDIATION_GEOMETRIES:
        raise ValueError(f"Unknown query-mediation geometry: {name}")
    source_layer = int(layer)
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    progress_specs, _excluded = mechanism_continuations(
        row, tokenizer, mechanism="progress_transition"
    )
    progress_by_occurrence = {
        int(value["from_occurrence"]): dict(value) for value in progress_specs
    }
    if receiver not in progress_by_occurrence or donor not in progress_by_occurrence:
        raise ValueError("Query mediation pair lacks a progress transition")
    receiver_progress = progress_by_occurrence[receiver]
    donor_progress = progress_by_occurrence[donor]

    grammar_pair = str(receiver_progress.get("grammar_pair", ""))
    grammar = grammar_pair.split(" -> ")[-1]
    routes = targeted_bank["routes"]
    if grammar not in routes:
        raise ValueError(f"No frozen targeted route for grammar {grammar}")
    required_role = str(tuple(routes[grammar])[0])
    route_specs, _route_excluded = mechanism_continuations(
        row, tokenizer, mechanism="retrieval_anchor_localization"
    )
    route_candidates = [
        dict(value)
        for value in route_specs
        if int(value["from_occurrence"]) == receiver
        and required_role in {str(role) for role in value.get("anchor_roles", ())}
    ]
    if len(route_candidates) != 1:
        raise ValueError(
            f"Expected one routed {required_role} query, found "
            f"{len(route_candidates)}"
        )
    route_spec = route_candidates[0]
    route_query_output = int(route_spec["query_output_token_index"])
    route_encoding = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=route_query_output,
        sequence_output_token_end=route_query_output + 1,
        selected_site=route_spec,
    )
    query = int(route_encoding.query_position)
    answer_encoding, registry = build_answer_source_registry(row, tokenizer)
    receiver_positions, donor_positions, geometry_audit = (
        trace_patch_geometry_positions(
            registry,
            receiver_occurrence=receiver,
            donor_occurrence=donor,
            geometry=name,
        )
    )
    position_audit = validate_query_mediation_positions(
        receiver_positions,
        donor_positions,
        query_position=query,
    )
    capture_positions = receiver_positions + donor_positions
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        answer_encoding,
        capture_positions,
        layers=[source_layer],
    )
    width = len(receiver_positions)
    receiver_states = captured[source_layer][:width].clone()
    donor_states = captured[source_layer][width:].clone()
    if name == "endpoint":
        endpoint_states, state_audit = native_loop_condition_states(
            receiver_states[0],
            donor_states[0],
            center,
            basis,
            random_seed=int(random_seed),
        )
        state_conditions = REGISTERED_QUERY_MEDIATION_ENDPOINT_STATES
        replacements = {
            condition: endpoint_states[condition].reshape(1, -1)
            for condition in state_conditions
        }
    else:
        state_conditions = REGISTERED_QUERY_MEDIATION_SPAN_STATES
        replacements = {
            "self_patch": receiver_states,
            "full_donor_patch": donor_states,
        }
        state_audit = {
            "basis_rank": int(torch.as_tensor(basis).shape[1]),
            "condition_audit": {
                "self_patch": {
                    "condition_patch_delta_norm": 0.0,
                    "condition_target_count_fraction": 0.0,
                },
                "full_donor_patch": {
                    "condition_patch_delta_norm": float(
                        torch.linalg.vector_norm(
                            donor_states.float() - receiver_states.float()
                        )
                    ),
                    "condition_target_count_fraction": None,
                },
            },
        }

    query_output_index = route_query_output
    target_start = int(route_spec["target_output_token_start"])
    target_end = int(route_spec["target_output_token_end"])
    baseline_ids = output_token_ids(row)
    receiver_path = tuple(
        int(value) for value in baseline_ids[query_output_index + 1 : target_end]
    )
    city_offset = target_start - query_output_index - 1
    receiver_city_ids = tuple(int(value) for value in route_spec["target_token_ids"])
    if receiver_path[city_offset:] != receiver_city_ids:
        raise RuntimeError("Routed receiver path disagrees with its target city")
    donor_city_ids = tuple(
        int(value) for value in donor_progress["target_token_ids"]
    )
    donor_path = receiver_path[:city_offset] + donor_city_ids
    receiver_city = str(route_spec["target_city"])
    donor_city = str(donor_progress["target_city"])
    known_cities = tuple(str(value["city"]) for value in gold_records(row))

    active_selected = tuple(
        (int(value[0]), int(value[1]))
        for value in head_plan["active_selected_heads"]
    )
    random_heads = tuple(
        (int(value[0]), int(value[1]))
        for value in head_plan["layer_matched_random_heads"]
    )
    if active_selected != tuple(
        (int(layer_index), int(head))
        for layer_index, head in targeted_bank["heads"]
        if int(layer_index) > source_layer
    ):
        raise ValueError("Active query-mediation heads disagree with targeted bank")

    prefix_encoding = replace(
        route_encoding,
        input_ids=tuple(int(value) for value in route_encoding.input_ids[:query]),
        attention_mask=tuple(
            int(value) for value in route_encoding.attention_mask[:query]
        ),
        query_position=query - 1,
    )
    _input_ids, query_attention_mask = _encoding_tensors(model, route_encoding)
    prefixes: dict[str, Any] = {}
    query_states: dict[str, torch.Tensor | None] = {}
    prefix_audits: dict[str, dict[str, Any]] = {}
    for condition in state_conditions:
        replacement = replacements[condition]
        prefix_indices = [
            index for index, position in enumerate(receiver_positions) if position < query
        ]
        query_indices = [
            index for index, position in enumerate(receiver_positions) if position == query
        ]
        if len(query_indices) > 1:
            raise RuntimeError("Query mediation selected the query position twice")
        prefix_positions = tuple(receiver_positions[index] for index in prefix_indices)
        prefix_states = (
            replacement[prefix_indices]
            if prefix_indices
            else None
        )
        prefix, prefix_applications, prefix_norm = _prefill_with_state_replacements(
            model,
            adapter,
            prefix_encoding,
            layer=source_layer,
            positions=prefix_positions,
            states=prefix_states,
        )
        prefixes[condition] = prefix
        query_states[condition] = (
            replacement[query_indices[0]] if query_indices else None
        )
        prefix_audits[condition] = {
            "prefix_state_patch_applications": int(prefix_applications),
            "prefix_state_patch_realized_fro_norm": float(prefix_norm),
            "query_state_position_patched": bool(query_indices),
        }

    common = {
        "schema_version": NATIVE_LOOP_SCHEMA_VERSION,
        "experiment_id": "p0_same_trajectory_query_mediation",
        "request_id": str(row["request_id"]),
        "model_label": str(answer_encoding.model_label),
        "seed": int(row["seed"]),
        "gold_count": len(known_cities),
        "layer": source_layer,
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "donor_offset": donor - receiver,
        "patch_geometry": name,
        "targeted_route_grammar": grammar,
        "targeted_route_anchor_role": required_role,
        "targeted_route_query_output_token_index": route_query_output,
        "targeted_route_query_full_token_index": query,
        "receiver_expected_next_city": receiver_city,
        "donor_expected_next_city": donor_city,
        "teacher_forced_tokens_p0_to_query": query - max(receiver_positions),
        "query_to_city_token_count": len(receiver_path),
        "query_to_city_interstitial_token_count": city_offset,
        "targeted_bank_sha256": str(targeted_bank["bank_sha256"]),
        "targeted_bank_size": int(targeted_bank["bank_size"]),
        "active_targeted_head_count": len(active_selected),
        "active_targeted_head_sha256": _sha256_json(active_selected),
        "random_head_sha256": _sha256_json(random_heads),
        "head_plan_file_sha256": str(head_plan["plan_file_sha256"]),
        "selection_rank_used": False,
        **geometry_audit,
        **position_audit,
        **{key: value for key, value in state_audit.items() if key != "condition_audit"},
    }
    rows: list[dict[str, Any]] = []
    captures_by_state: dict[str, dict[int, torch.Tensor]] = {}
    for condition in state_conditions:
        for head_condition, zero_heads in (
            ("intact", ()),
            ("selected_mask", active_selected),
            ("layer_matched_random_mask", random_heads),
        ):
            query_output, captures, hook_audit = (
                _query_from_prefix_with_mediation_hooks(
                    model,
                    adapter,
                    route_encoding,
                    prefix_output=clone_prefill_output_for_scoring(
                        prefixes[condition]
                    ),
                    query_attention_mask=query_attention_mask,
                    source_layer=source_layer,
                    query_replacement_state=query_states[condition],
                    zero_heads=zero_heads,
                    capture_heads=(active_selected if head_condition == "intact" else ()),
                )
            )
            if head_condition == "intact":
                captures_by_state[condition] = captures
            outcomes = _query_city_outcomes(
                model,
                tokenizer,
                route_encoding,
                query_output,
                receiver_path=receiver_path,
                donor_path=donor_path,
                city_token_offset=city_offset,
                known_cities=known_cities,
                receiver_city=receiver_city,
                donor_city=donor_city,
                run_greedy=run_greedy,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    **common,
                    "state_condition": condition,
                    "carrier_state_condition": condition,
                    "mediator_source_condition": condition,
                    "head_condition": head_condition,
                    "head_intervention_scope": "registered_retrieval_query_only",
                    "head_restoration_mode": "none",
                    **prefix_audits[condition],
                    **state_audit["condition_audit"][condition],
                    **hook_audit,
                    **outcomes,
                }
            )

    restoration_pairs = [("full_donor_patch", "self_patch")]
    if name == "endpoint":
        restoration_pairs.append(
            ("count_subspace_transplant", "norm_matched_orthogonal_patch")
        )
    for mediator_source, carrier in restoration_pairs:
        query_output, _captures, hook_audit = (
            _query_from_prefix_with_mediation_hooks(
                model,
                adapter,
                route_encoding,
                prefix_output=clone_prefill_output_for_scoring(prefixes[carrier]),
                query_attention_mask=query_attention_mask,
                source_layer=source_layer,
                query_replacement_state=query_states[carrier],
                capture_heads=active_selected,
                head_replacements=captures_by_state[mediator_source],
            )
        )
        outcomes = _query_city_outcomes(
            model,
            tokenizer,
            route_encoding,
            query_output,
            receiver_path=receiver_path,
            donor_path=donor_path,
            city_token_offset=city_offset,
            known_cities=known_cities,
            receiver_city=receiver_city,
            donor_city=donor_city,
            run_greedy=run_greedy,
            max_new_tokens=max_new_tokens,
        )
        rows.append(
            {
                **common,
                "state_condition": f"{mediator_source}_heads_into_{carrier}",
                "carrier_state_condition": carrier,
                "mediator_source_condition": mediator_source,
                "head_condition": "selected_restore",
                "head_intervention_scope": "registered_retrieval_query_only",
                "head_restoration_mode": "cumulative_selected_pre_o_slice_restore",
                **prefix_audits[carrier],
                **state_audit["condition_audit"][carrier],
                **hook_audit,
                **outcomes,
            }
        )
    expected = len(state_conditions) * len(
        REGISTERED_QUERY_MEDIATION_HEAD_CONDITIONS
    ) + len(restoration_pairs)
    if len(rows) != expected:
        raise RuntimeError(
            f"Query mediation emitted {len(rows)} rows, expected {expected}"
        )
    return rows


@torch.inference_mode()
def run_p0_native_loop_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    shuffled_donor_occurrence: int | None = None,
    layer: int,
    center: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
    targeted_bank: Mapping[str, Any],
    conditions: Sequence[str] = REGISTERED_P0_LOOP_CONDITIONS,
    random_seed: int = 0,
    run_greedy: bool = True,
    max_new_tokens: int = 48,
) -> list[dict[str, Any]]:
    """Steer one P0 count state and measure probe, attention, and next city."""

    requested = tuple(str(value) for value in conditions)
    if len(set(requested)) != len(requested):
        raise ValueError("P0 loop conditions must be unique")
    unknown = sorted(set(requested) - set(AVAILABLE_P0_LOOP_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown P0 loop conditions: {unknown}")
    if not {"clean", "self_patch"} <= set(requested):
        raise ValueError("P0 loop trials require clean and self_patch")
    specificity_requested = set(requested) & set(
        REGISTERED_FULL_COMMIT_SPECIFICITY_CONDITIONS
    )
    if specificity_requested and "full_donor_patch" not in requested:
        raise ValueError("Full-commit specificity requires full_donor_patch")
    if (
        "shuffled_natural_donor_patch" in specificity_requested
        and shuffled_donor_occurrence is None
    ):
        raise ValueError("Shuffled natural-donor control lacks an occurrence")
    count = len(gold_records(row))
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    shuffled_donor = (
        None
        if shuffled_donor_occurrence is None
        else int(shuffled_donor_occurrence)
    )
    if not 1 < receiver < count:
        raise ValueError("P0 loop receiver must be strictly intermediate")
    if donor == receiver or not 1 <= donor < count:
        raise ValueError("P0 loop donor must own a next-item transition")
    if shuffled_donor is not None and (
        shuffled_donor in {receiver, donor}
        or not 1 <= shuffled_donor < count
    ):
        raise ValueError(
            "Shuffled donor must own a distinct next-item transition"
        )
    source_layer = int(layer)
    if not 0 <= source_layer < int(adapter.num_layers) - 1:
        raise ValueError("P0 loop layer must leave a downstream decoder layer")

    progress_specs, _excluded = mechanism_continuations(
        row, tokenizer, mechanism="progress_transition"
    )
    by_occurrence = {
        int(value["from_occurrence"]): dict(value) for value in progress_specs
    }
    if receiver not in by_occurrence or donor not in by_occurrence:
        raise ValueError("P0 loop pair lacks a progress transition")
    receiver_spec = by_occurrence[receiver]
    donor_spec = by_occurrence[donor]
    shuffled_spec = (
        None
        if shuffled_donor is None
        else by_occurrence.get(shuffled_donor)
    )
    if shuffled_donor is not None and shuffled_spec is None:
        raise ValueError("Shuffled donor lacks a progress transition")
    answer_encoding, registry = build_answer_source_registry(row, tokenizer)
    endpoints = tuple(end - 1 for _start, end in registry.trace_items)
    receiver_position = int(endpoints[receiver - 1])
    donor_position = int(endpoints[donor - 1])
    shuffled_position = (
        None
        if shuffled_donor is None
        else int(endpoints[shuffled_donor - 1])
    )
    capture_positions = [receiver_position, donor_position]
    if shuffled_position is not None:
        capture_positions.append(shuffled_position)
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        answer_encoding,
        capture_positions,
        layers=[source_layer],
    )
    receiver_state, donor_state = captured[source_layer][:2]
    shuffled_state = (
        None
        if shuffled_position is None
        else captured[source_layer][2]
    )
    states, state_audit = native_loop_condition_states(
        receiver_state,
        donor_state,
        center,
        basis,
        random_seed=int(random_seed),
    )
    if specificity_requested:
        specificity_states, specificity_audit = (
            full_commit_specificity_condition_states(
                receiver_state,
                donor_state,
                shuffled_donor_state=shuffled_state,
                random_seed=int(random_seed) + 1_000_003,
                random_replicates=3,
            )
        )
        states.update(specificity_states)
        state_audit["condition_audit"].update(
            specificity_audit["condition_audit"]
        )
        state_audit.update(
            {
                f"full_specificity_{key}": value
                for key, value in specificity_audit.items()
                if key not in {"condition_audit", "full_donor_delta_norm"}
            }
        )

    query_output_index = int(receiver_spec["query_output_token_index"])
    target_start = int(receiver_spec["target_output_token_start"])
    target_end = int(receiver_spec["target_output_token_end"])
    baseline_ids = output_token_ids(row)
    receiver_path = tuple(baseline_ids[query_output_index + 1 : target_end])
    city_offset = target_start - query_output_index - 1
    receiver_city_ids = tuple(int(value) for value in receiver_spec["target_token_ids"])
    if receiver_path[city_offset:] != receiver_city_ids:
        raise RuntimeError("Receiver path disagrees with registered target city")
    donor_city_ids = tuple(int(value) for value in donor_spec["target_token_ids"])
    donor_path = receiver_path[:city_offset] + donor_city_ids
    shuffled_city_ids = (
        None
        if shuffled_spec is None
        else tuple(int(value) for value in shuffled_spec["target_token_ids"])
    )
    shuffled_path = (
        None
        if shuffled_city_ids is None
        else receiver_path[:city_offset] + shuffled_city_ids
    )
    local_encoding = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=query_output_index,
        sequence_output_token_end=query_output_index + 1,
        selected_site=receiver_spec,
    )
    if int(local_encoding.query_position) != receiver_position:
        raise RuntimeError("P0 local encoding moved the receiver endpoint")

    known_cities = tuple(str(value["city"]) for value in gold_records(row))
    receiver_next_city = str(receiver_spec["target_city"])
    donor_next_city = str(donor_spec["target_city"])
    shuffled_next_city = (
        None if shuffled_spec is None else str(shuffled_spec["target_city"])
    )
    ordinal_by_city = {
        str(record["city"]).casefold(): index
        for index, record in enumerate(gold_records(row), start=1)
    }
    shuffled_successor_ordinal = (
        None
        if shuffled_next_city is None
        else int(ordinal_by_city[shuffled_next_city.casefold()])
    )
    common = {
        "schema_version": NATIVE_LOOP_SCHEMA_VERSION,
        "experiment_id": "p0_count_state_to_targeted_retrieval",
        **(
            {
                "experiment_variant": "full_commit_specificity_v1",
                "full_commit_specificity_uses_count_subspace": False,
            }
            if specificity_requested
            else {}
        ),
        "request_id": str(row["request_id"]),
        "model_label": str(answer_encoding.model_label),
        "seed": int(row["seed"]),
        "gold_count": count,
        "layer": source_layer,
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "shuffled_donor_occurrence": shuffled_donor,
        "donor_offset": donor - receiver,
        "shuffled_donor_offset": (
            None if shuffled_donor is None else shuffled_donor - receiver
        ),
        "donor_direction": (
            "past_to_later_receiver"
            if donor < receiver
            else "future_to_earlier_receiver"
        ),
        "future_donor_is_counterfactual_not_natural_stream": bool(donor > receiver),
        "receiver_position": receiver_position,
        "donor_position": donor_position,
        "shuffled_donor_position": shuffled_position,
        "receiver_expected_next_city": receiver_next_city,
        "donor_expected_next_city": donor_next_city,
        "shuffled_donor_expected_next_city": shuffled_next_city,
        "shuffled_donor_successor_source_ordinal": shuffled_successor_ordinal,
        "teacher_forced_interstitial_token_count": city_offset,
        "probe_site": "patched_p0_post_block_state",
        "attention_site": "frozen_grammar_routed_targeted_query",
        "generation_endpoint": "first_identifiable_gold_city",
        "selection_rank_used": False,
        **{key: value for key, value in state_audit.items() if key != "condition_audit"},
    }
    results: list[dict[str, Any]] = []
    for condition in requested:
        replacement_state = states[condition]
        prefill, applications, realized_norm = _prefill_with_state_replacements(
            model,
            adapter,
            local_encoding,
            layer=source_layer,
            positions=(receiver_position,),
            states=(
                None
                if replacement_state is None
                else replacement_state.reshape(1, -1)
            ),
        )
        receiver_score = _score_trace_continuation(
            model,
            local_encoding,
            prefill,
            receiver_path,
            city_token_offset=city_offset,
        )
        donor_score = _score_trace_continuation(
            model,
            local_encoding,
            prefill,
            donor_path,
            city_token_offset=city_offset,
        )
        shuffled_score = (
            None
            if shuffled_path is None
            else _score_trace_continuation(
                model,
                local_encoding,
                prefill,
                shuffled_path,
                city_token_offset=city_offset,
            )
        )
        attention = _routed_targeted_attention_metrics(
            model,
            tokenizer,
            adapter,
            row,
            receiver_specification=receiver_spec,
            receiver_position=receiver_position,
            source_layer=source_layer,
            replacement_state=replacement_state,
            targeted_bank=targeted_bank,
            donor_successor_city=donor_next_city,
        )
        local_outcomes: dict[str, Any] = {
            "receiver_path_log_probability": receiver_score[
                "sequence_log_probability"
            ],
            "donor_path_log_probability": donor_score["sequence_log_probability"],
            "donor_vs_receiver_path_log_odds": float(
                donor_score["sequence_log_probability"]
                - receiver_score["sequence_log_probability"]
            ),
            "receiver_city_log_probability": receiver_score["city_log_probability"],
            "donor_city_log_probability": donor_score["city_log_probability"],
            "donor_vs_receiver_city_log_odds": float(
                donor_score["city_log_probability"]
                - receiver_score["city_log_probability"]
            ),
        }
        if shuffled_score is not None:
            local_outcomes.update(
                {
                    "shuffled_donor_path_log_probability": shuffled_score[
                        "sequence_log_probability"
                    ],
                    "shuffled_donor_city_log_probability": shuffled_score[
                        "city_log_probability"
                    ],
                    "shuffled_donor_vs_receiver_city_log_odds": float(
                        shuffled_score["city_log_probability"]
                        - receiver_score["city_log_probability"]
                    ),
                    "donor_vs_shuffled_donor_city_log_odds": float(
                        donor_score["city_log_probability"]
                        - shuffled_score["city_log_probability"]
                    ),
                }
            )
        if shuffled_successor_ordinal is not None:
            source_masses = attention["targeted_bank_source_masses"]
            shuffled_mass = float(source_masses[str(shuffled_successor_ordinal)])
            donor_mass = float(attention["donor_successor_attention_mass"])
            receiver_mass = float(attention["receiver_successor_attention_mass"])
            local_outcomes.update(
                {
                    "shuffled_donor_successor_attention_mass": shuffled_mass,
                    "shuffled_minus_receiver_successor_attention_mass": float(
                        shuffled_mass - receiver_mass
                    ),
                    "donor_minus_shuffled_successor_attention_mass": float(
                        donor_mass - shuffled_mass
                    ),
                }
            )
        if run_greedy:
            completion = generate_answer_completion_from_prefill(
                model,
                tokenizer,
                local_encoding,
                prefill,
                max_new_tokens=int(max_new_tokens),
            )
            generated_city, city_start, city_evidence = _first_generated_city_record(
                str(completion["completion_text"]), known_cities
            )
            local_outcomes.update(
                {
                    "local_completion_text": str(completion["completion_text"]),
                    "local_generation_truncated": bool(
                        completion["generation_truncated"]
                    ),
                    "first_generated_city_record": generated_city,
                    "first_generated_city_record_char_start": city_start,
                    "first_generated_city_record_evidence": city_evidence,
                    "donor_city_adoption": bool(
                        generated_city is not None
                        and generated_city.casefold() == donor_next_city.casefold()
                    ),
                    "receiver_city_retention": bool(
                        generated_city is not None
                        and generated_city.casefold() == receiver_next_city.casefold()
                    ),
                    "shuffled_donor_city_adoption": bool(
                        generated_city is not None
                        and shuffled_next_city is not None
                        and generated_city.casefold()
                        == shuffled_next_city.casefold()
                    ),
                }
            )
        results.append(
            {
                **common,
                "condition": condition,
                "status": "ok",
                "local_patch_hook_applications": int(applications),
                "local_patch_realized_fro_norm": float(realized_norm),
                **state_audit["condition_audit"][condition],
                **attention,
                **local_outcomes,
            }
        )
    return results


@torch.inference_mode()
def _html_local_target_outcomes(
    model: Any,
    tokenizer: Any,
    encoding: NativeTraceEncoding,
    query_output: Any,
    *,
    target_path: Sequence[int],
    city_token_offset: int,
    target_city: str,
    alternative_city_paths: Sequence[tuple[str, Sequence[int]]],
    known_cities: Sequence[str],
    run_greedy: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    target = _score_trace_continuation(
        model,
        encoding,
        query_output,
        target_path,
        city_token_offset=int(city_token_offset),
    )
    alternatives = []
    for city, path in alternative_city_paths:
        score = _score_trace_continuation(
            model,
            encoding,
            query_output,
            path,
            city_token_offset=int(city_token_offset),
        )
        alternatives.append((str(city), score))
    best_alternative = max(
        alternatives,
        key=lambda value: float(value[1]["city_log_probability"]),
    )
    outcomes: dict[str, Any] = {
        "target_query_path_log_probability": float(
            target["sequence_log_probability"]
        ),
        "target_query_city_log_probability": float(target["city_log_probability"]),
        "best_alternative_city": best_alternative[0],
        "best_alternative_query_city_log_probability": float(
            best_alternative[1]["city_log_probability"]
        ),
        "target_vs_best_other_city_log_odds": float(
            target["city_log_probability"]
            - best_alternative[1]["city_log_probability"]
        ),
    }
    if run_greedy:
        completion = generate_answer_completion_from_prefill(
            model,
            tokenizer,
            encoding,
            clone_prefill_output_for_scoring(query_output),
            max_new_tokens=int(max_new_tokens),
        )
        generated_city, city_start, city_evidence = _first_generated_city_record(
            str(completion["completion_text"]), known_cities
        )
        outcomes.update(
            {
                "query_completion_text": str(completion["completion_text"]),
                "query_generation_truncated": bool(
                    completion["generation_truncated"]
                ),
                "query_first_generated_city_record": generated_city,
                "query_first_generated_city_record_char_start": city_start,
                "query_first_generated_city_record_evidence": city_evidence,
                "query_target_city_adoption": bool(
                    generated_city is not None
                    and generated_city.casefold() == str(target_city).casefold()
                ),
            }
        )
    return outcomes


@torch.inference_mode()
def _score_trace_continuation_with_mediation_hooks(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    prefill_output: Any,
    token_ids: Sequence[int],
    *,
    city_token_offset: int,
    zero_heads: Sequence[tuple[int, int]] = (),
    capture_heads: Sequence[tuple[int, int]] = (),
    head_replacements: Mapping[int, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[int, torch.Tensor], dict[str, Any]]:
    """Score a path while intervening at every teacher-forced input token.

    The ordinary query-mediation experiment touches only the registered query
    token.  This helper keeps that query intervention separate and extends the
    same exact pre-O head-slice mask/capture/restore operation over the forced
    inputs between the query and the scored city.  Captures and replacements
    are path matched, so unequal candidate-city token lengths are never
    aligned or truncated.
    """

    tokens = tuple(int(value) for value in token_ids)
    city_offset = int(city_token_offset)
    if not tokens or not 0 <= city_offset < len(tokens):
        raise ValueError("Trace continuation has invalid city-token bounds")
    zero_grouped = _validate_head_bank(adapter, zero_heads)
    capture_grouped = _validate_head_bank(adapter, capture_heads)
    replacement_values = {
        int(layer): torch.as_tensor(value).detach().float().cpu()
        for layer, value in (head_replacements or {}).items()
    }
    replacement_grouped = capture_grouped if replacement_values else {}
    if replacement_values and set(replacement_values) != set(replacement_grouped):
        raise ValueError("Continuation replacements do not cover captured layers")
    if set(zero_grouped) & set(replacement_grouped):
        raise ValueError("Cannot zero and restore one continuation layer")

    scoring_output = clone_prefill_output_for_scoring(prefill_output)
    logits = getattr(scoring_output, "logits", None)
    past = getattr(scoring_output, "past_key_values", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or past is None:
        raise RuntimeError("Trace continuation prefill exposes no logits/cache")
    first_log_probabilities = torch.log_softmax(logits[0, -1].float(), dim=-1)
    token_log_probabilities = [first_log_probabilities[tokens[0]]]
    captures: dict[int, torch.Tensor] = {}
    applications: dict[int, int] = {}
    forced_input_count = max(0, len(tokens) - 1)
    if forced_input_count:
        intervention_layers = sorted(
            set(zero_grouped) | set(capture_grouped) | set(replacement_grouped)
        )
        handles = []
        for layer in intervention_layers:
            zero_layer_heads = zero_grouped.get(layer, ())
            capture_layer_heads = capture_grouped.get(layer, ())
            replacement_layer_heads = replacement_grouped.get(layer, ())
            width = int(adapter.head_dims[layer])
            expected = int(adapter.num_heads[layer]) * width
            applications[layer] = 0

            def head_hook(
                _module: Any,
                args: tuple[Any, ...],
                *,
                layer: int = layer,
                zero_layer_heads: tuple[int, ...] = zero_layer_heads,
                capture_layer_heads: tuple[int, ...] = capture_layer_heads,
                replacement_layer_heads: tuple[int, ...] = replacement_layer_heads,
                width: int = width,
                expected: int = expected,
            ) -> tuple[Any, ...]:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError("Continuation o_proj received no tensor")
                value = args[0]
                if (
                    value.ndim != 3
                    or int(value.shape[0]) != 1
                    or int(value.shape[1]) != forced_input_count
                    or int(value.shape[-1]) != expected
                ):
                    raise RuntimeError(
                        "Continuation head hook saw an invalid tensor shape"
                    )
                if capture_layer_heads:
                    captures[layer] = value.detach().float().cpu().clone()
                patched = value.clone()
                for head in zero_layer_heads:
                    left = int(head) * width
                    patched[..., left : left + width] = 0
                if replacement_layer_heads:
                    replacement = replacement_values[layer].to(
                        device=value.device, dtype=value.dtype
                    )
                    if replacement.shape != value.shape:
                        raise RuntimeError(
                            "Path-matched continuation replacement shape disagrees"
                        )
                    for head in replacement_layer_heads:
                        left = int(head) * width
                        patched[..., left : left + width] = replacement[
                            ..., left : left + width
                        ]
                applications[layer] += 1
                return (patched, *args[1:])

            handles.append(
                adapter.output_projections[layer].register_forward_pre_hook(
                    head_hook
                )
            )
        try:
            device = logits.device
            continuation_inputs = torch.tensor(
                [tokens[:-1]], dtype=torch.long, device=device
            )
            continuation_mask = torch.ones_like(continuation_inputs)
            base_mask = torch.tensor(
                [encoding.attention_mask], dtype=torch.long, device=device
            )
            kwargs: dict[str, Any] = {
                "input_ids": continuation_inputs,
                "attention_mask": torch.cat((base_mask, continuation_mask), dim=1),
                "past_key_values": past,
                "use_cache": False,
            }
            positions = torch.arange(
                encoding.sequence_length,
                encoding.sequence_length + forced_input_count,
                dtype=torch.long,
                device=device,
            )
            if _accepts_keyword(model, "position_ids"):
                kwargs["position_ids"] = positions.unsqueeze(0)
            if _accepts_keyword(model, "cache_position"):
                kwargs["cache_position"] = positions
            shared = _extract_shared_kv_states(scoring_output)
            if shared is not None and _accepts_keyword(model, "shared_kv_states"):
                kwargs["shared_kv_states"] = shared
            continuation_output = model(**kwargs)
        finally:
            for handle in handles:
                handle.remove()
        continuation_logits = getattr(continuation_output, "logits", None)
        if (
            not isinstance(continuation_logits, torch.Tensor)
            or continuation_logits.ndim != 3
            or int(continuation_logits.shape[1]) != forced_input_count
        ):
            raise RuntimeError("Trace continuation returned unexpected logits")
        continuation_log_probabilities = torch.log_softmax(
            continuation_logits[0].float(), dim=-1
        )
        token_log_probabilities.extend(
            continuation_log_probabilities[index, token]
            for index, token in enumerate(tokens[1:])
        )
        bad_layers = sorted(
            layer for layer, count in applications.items() if count != 1
        )
        if bad_layers:
            raise RuntimeError(
                "Continuation mediation hooks did not apply once: "
                f"{bad_layers}"
            )
        if capture_grouped and set(captures) != set(capture_grouped):
            raise RuntimeError("Continuation did not capture every selected layer")
    elif replacement_values:
        raise RuntimeError(
            "A one-token candidate path has no continuation head state to restore"
        )

    values = torch.stack(token_log_probabilities).detach().float().cpu()
    city_values = values[city_offset:]
    return (
        {
            "sequence_log_probability": float(values.sum()),
            "mean_token_log_probability": float(values.mean()),
            "city_log_probability": float(city_values.sum()),
            "mean_city_token_log_probability": float(city_values.mean()),
            "token_count": len(tokens),
            "city_token_count": len(tokens) - city_offset,
        },
        captures,
        {
            "continuation_forced_input_token_count": forced_input_count,
            "continuation_head_hook_applications": {
                str(layer): int(count)
                for layer, count in sorted(applications.items())
            },
        },
    )


def _html_local_hooked_candidate_panel(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    query_output: Any,
    *,
    target_path: Sequence[int],
    city_token_offset: int,
    target_city: str,
    alternative_city_paths: Sequence[tuple[str, Sequence[int]]],
    zero_heads: Sequence[tuple[int, int]] = (),
    capture_heads: Sequence[tuple[int, int]] = (),
    continuation_replacements: Mapping[str, Mapping[int, torch.Tensor]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    """Score the target-city panel with path-matched multi-token head hooks."""

    candidates = [("target", str(target_city), tuple(target_path))]
    candidates.extend(
        (f"alternative_{index}", str(city), tuple(path))
        for index, (city, path) in enumerate(alternative_city_paths)
    )
    scored: dict[str, tuple[str, dict[str, Any]]] = {}
    captures: dict[str, dict[int, torch.Tensor]] = {}
    audits: dict[str, Any] = {}
    replacements = continuation_replacements or {}
    for key, city, path in candidates:
        score, path_captures, path_audit = (
            _score_trace_continuation_with_mediation_hooks(
                model,
                adapter,
                encoding,
                query_output,
                path,
                city_token_offset=int(city_token_offset),
                zero_heads=zero_heads,
                capture_heads=capture_heads,
                head_replacements=replacements.get(key),
            )
        )
        scored[key] = (city, score)
        captures[key] = path_captures
        audits[key] = path_audit
    alternatives = [scored[key] for key, _city, _path in candidates[1:]]
    best_alternative = max(
        alternatives,
        key=lambda value: float(value[1]["city_log_probability"]),
    )
    target = scored["target"][1]
    outcomes = {
        "target_query_path_log_probability": float(
            target["sequence_log_probability"]
        ),
        "target_query_city_log_probability": float(target["city_log_probability"]),
        "best_alternative_city": best_alternative[0],
        "best_alternative_query_city_log_probability": float(
            best_alternative[1]["city_log_probability"]
        ),
        "target_vs_best_other_city_log_odds": float(
            target["city_log_probability"]
            - best_alternative[1]["city_log_probability"]
        ),
    }
    return outcomes, captures, audits


@torch.inference_mode()
def run_html_aligned_local_serial_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    layer: int,
    targeted_bank: Mapping[str, Any],
    head_plan: Mapping[str, Any],
    random_seed: int,
    head_token_geometry: str = "query_only",
    run_greedy: bool = False,
    max_new_tokens: int = 32,
) -> list[dict[str, Any]]:
    """Propagate a full item span across one item, then read it via targeted heads."""

    source_layer = int(layer)
    head_geometry = str(head_token_geometry)
    if head_geometry not in {"query_only", "query_plus_full_path"}:
        raise ValueError(f"Unknown HTML local head-token geometry: {head_geometry}")
    if head_geometry == "query_plus_full_path" and run_greedy:
        raise ValueError("Full-path head mediation is teacher-forced only")
    clean_answer, registry = build_answer_source_registry(row, tokenizer)
    count = len(registry.trace_items)
    if count < 4:
        raise ValueError("HTML local serial mediation requires at least four items")
    target_occurrence = count - 2
    query_from_occurrence = count - 1
    span_start, span_end = registry.trace_items[target_occurrence - 1]
    target_positions = tuple(range(int(span_start), int(span_end)))
    all_control_answer, control_audit = build_html_aligned_uninformative_trace_encoding(
        clean_answer,
        registry,
        tokenizer,
        random_seed=int(random_seed),
    )
    target_only_ids = list(clean_answer.input_ids)
    target_only_ids[int(span_start) : int(span_end)] = all_control_answer.input_ids[
        int(span_start) : int(span_end)
    ]
    control_answer = replace(clean_answer, input_ids=tuple(target_only_ids))
    target_control_start = int(
        control_audit["control_prompt_window_starts"][target_occurrence - 1]
    )
    changed_target_tokens = sum(
        clean_answer.input_ids[position] != control_answer.input_ids[position]
        for position in target_positions
    )
    control_audit.update(
        {
            "control_construction": "same_length_target_item_prompt_background_window",
            "all_trace_items_replaced": False,
            "target_item_only_control": True,
            "controlled_occurrences": [target_occurrence],
            "control_prompt_window_starts": [target_control_start],
            "control_prompt_window_starts_sha256": _sha256_json(
                [target_control_start]
            ),
            "changed_trace_item_token_count": int(changed_target_tokens),
            "changed_trace_item_token_fraction": float(
                changed_target_tokens / len(target_positions)
            ),
            "control_input_ids_sha256": _sha256_json(control_answer.input_ids),
        }
    )
    patch_layers = _full_state_patch_layers(
        source_layer=source_layer,
        num_layers=int(adapter.num_layers),
        layer_mode="cumulative_clamp",
    )
    _clean_logits, clean_capture = capture_post_block_states(
        model,
        adapter,
        clean_answer,
        target_positions,
        layers=patch_layers,
    )
    _control_logits, control_capture = capture_post_block_states(
        model,
        adapter,
        control_answer,
        target_positions,
        layers=patch_layers,
    )
    clean_states = {
        patch_layer: clean_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    control_states = {
        patch_layer: control_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }

    progress_specs, _excluded = mechanism_continuations(
        row, tokenizer, mechanism="progress_transition"
    )
    progress_by_occurrence = {
        int(value["from_occurrence"]): dict(value) for value in progress_specs
    }
    if query_from_occurrence not in progress_by_occurrence:
        raise ValueError("Intervening item has no registered successor transition")
    progress = progress_by_occurrence[query_from_occurrence]
    grammar_pair = str(progress.get("grammar_pair", ""))
    grammar = grammar_pair.split(" -> ")[-1]
    routes = targeted_bank["routes"]
    if grammar not in routes:
        raise ValueError(f"No frozen targeted route for grammar {grammar}")
    required_role = str(tuple(routes[grammar])[0])
    route_specs, _route_excluded = mechanism_continuations(
        row, tokenizer, mechanism="retrieval_anchor_localization"
    )
    route_candidates = [
        dict(value)
        for value in route_specs
        if int(value["from_occurrence"]) == query_from_occurrence
        and required_role in {str(role) for role in value.get("anchor_roles", ())}
    ]
    if len(route_candidates) != 1:
        raise ValueError(
            f"Expected one post-intervening {required_role} query, found "
            f"{len(route_candidates)}"
        )
    route_spec = route_candidates[0]
    query_output_index = int(route_spec["query_output_token_index"])
    clean_route = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=query_output_index,
        sequence_output_token_end=query_output_index + 1,
        selected_site=route_spec,
    )
    route_length = clean_route.sequence_length
    if tuple(clean_route.input_ids) != tuple(clean_answer.input_ids[:route_length]):
        raise RuntimeError("Local route prefix is not aligned to the answer trace")
    control_route = replace(
        clean_route,
        input_ids=tuple(control_answer.input_ids[:route_length]),
    )
    query = int(clean_route.query_position)
    if max(target_positions) >= query:
        raise RuntimeError("Propagated full span must precede its retrieval query")
    state_conditions: dict[
        str, tuple[NativeTraceEncoding, Mapping[int, torch.Tensor] | None]
    ] = {
        "clean": (clean_route, None),
        "uninformative": (control_route, None),
        "clean_target_ablation": (clean_route, control_states),
        "uninformative_target_restore": (control_route, clean_states),
    }
    prefixes: dict[str, Any] = {}
    prefix_audits: dict[str, dict[str, Any]] = {}
    for condition in REGISTERED_HTML_LOCAL_SERIAL_STATES:
        active_encoding, replacements = state_conditions[condition]
        prefix, applications, norms = _prefix_with_layerwise_state_replacements(
            model,
            adapter,
            active_encoding,
            positions=target_positions,
            replacements=replacements,
        )
        prefixes[condition] = prefix
        prefix_audits[condition] = {
            "patch_hook_applications": {
                str(key): value for key, value in sorted(applications.items())
            },
            "patch_realized_fro_norm_by_layer": {
                str(key): value for key, value in sorted(norms.items())
            },
        }

    active_selected = tuple(
        (int(value[0]), int(value[1]))
        for value in head_plan["active_selected_heads"]
    )
    random_heads = tuple(
        (int(value[0]), int(value[1]))
        for value in head_plan["layer_matched_random_heads"]
    )
    expected_selected = tuple(
        (int(layer_index), int(head))
        for layer_index, head in targeted_bank["heads"]
        if int(layer_index) > source_layer
    )
    if active_selected != expected_selected or set(active_selected) & set(random_heads):
        raise ValueError("HTML local serial head plan changed")
    _input_ids, query_attention_mask = _encoding_tensors(model, clean_route)

    baseline_ids = output_token_ids(row)
    target_start = int(route_spec["target_output_token_start"])
    target_end = int(route_spec["target_output_token_end"])
    target_path = tuple(
        int(value) for value in baseline_ids[query_output_index + 1 : target_end]
    )
    city_offset = target_start - query_output_index - 1
    target_city_ids = tuple(int(value) for value in route_spec["target_token_ids"])
    if target_path[city_offset:] != target_city_ids:
        raise RuntimeError("Local route path disagrees with the terminal city")
    fixed_prefix = target_path[:city_offset]
    target_city = str(route_spec["target_city"])
    alternatives: list[tuple[str, tuple[int, ...]]] = []
    seen_cities = {target_city.casefold()}
    for value in progress_specs:
        city = str(value["target_city"])
        if city.casefold() in seen_cities:
            continue
        seen_cities.add(city.casefold())
        alternatives.append(
            (
                city,
                fixed_prefix
                + tuple(int(token) for token in value["target_token_ids"]),
            )
        )
    if not alternatives:
        raise ValueError("HTML local serial query has no alternative city")
    known_cities = tuple(str(value["city"]) for value in gold_records(row))

    common = {
        "schema_version": NATIVE_LOOP_SCHEMA_VERSION,
        "experiment_id": "html_aligned_one_step_full_span_serial_mediation",
        "request_id": clean_answer.request_id,
        "model_label": clean_answer.model_label,
        "seed": clean_answer.seed,
        "dataset_split": clean_answer.split,
        "gold_count": clean_answer.count,
        "layer": source_layer,
        "target_occurrence": target_occurrence,
        "target_is_two_before_terminal": True,
        "query_from_occurrence": query_from_occurrence,
        "intervening_item_count": 1,
        "target_span": [int(span_start), int(span_end)],
        "patch_geometry": "full_item_span_same_position",
        "patch_layer_mode": "cumulative_clamp",
        "patch_layers": list(patch_layers),
        "patch_token_count": len(target_positions),
        "targeted_route_grammar": grammar,
        "targeted_route_anchor_role": required_role,
        "targeted_route_query_output_token_index": query_output_index,
        "targeted_route_query_full_token_index": query,
        "target_city": target_city,
        "active_targeted_head_count": len(active_selected),
        "active_targeted_head_sha256": _sha256_json(active_selected),
        "random_head_sha256": _sha256_json(random_heads),
        "targeted_bank_sha256": str(targeted_bank["bank_sha256"]),
        "head_plan_file_sha256": str(head_plan["plan_file_sha256"]),
        "head_token_geometry": head_geometry,
        "selection_rank_used": False,
        "outcome_blind": True,
        **control_audit,
    }
    if head_geometry == "query_plus_full_path":
        source_state = "uninformative_target_restore"
        carrier_state = "uninformative"
        panel_rows: list[dict[str, Any]] = []

        def run_panel(
            state_condition: str,
            head_condition: str,
            *,
            zero_heads: Sequence[tuple[int, int]] = (),
            capture_heads: Sequence[tuple[int, int]] = (),
            query_replacements: Mapping[int, torch.Tensor] | None = None,
            continuation_replacements: Mapping[
                str, Mapping[int, torch.Tensor]
            ]
            | None = None,
            restoration_source: str | None = None,
        ) -> tuple[
            dict[str, Any],
            dict[int, torch.Tensor],
            dict[str, dict[int, torch.Tensor]],
        ]:
            active_encoding = state_conditions[state_condition][0]
            query_output, query_captures, query_audit = (
                _query_from_prefix_with_mediation_hooks(
                    model,
                    adapter,
                    active_encoding,
                    prefix_output=clone_prefill_output_for_scoring(
                        prefixes[state_condition]
                    ),
                    query_attention_mask=query_attention_mask,
                    source_layer=source_layer,
                    query_replacement_state=None,
                    zero_heads=zero_heads,
                    capture_heads=capture_heads,
                    head_replacements=query_replacements,
                )
            )
            outcomes, path_captures, continuation_audit = (
                _html_local_hooked_candidate_panel(
                    model,
                    adapter,
                    active_encoding,
                    query_output,
                    target_path=target_path,
                    city_token_offset=city_offset,
                    target_city=target_city,
                    alternative_city_paths=alternatives,
                    zero_heads=zero_heads,
                    capture_heads=capture_heads,
                    continuation_replacements=continuation_replacements,
                )
            )
            panel_rows.append(
                {
                    **common,
                    "state_condition": state_condition,
                    "head_condition": head_condition,
                    "head_restoration_source_state": restoration_source,
                    "head_restoration_mode": (
                        "selected_pre_o_path_matched_query_plus_full_path_restore"
                        if restoration_source is not None
                        else "none"
                    ),
                    "head_intervention_scope": (
                        "registered_query_plus_every_teacher_forced_path_input"
                    ),
                    "target_head_intervention_token_count": len(target_path),
                    "candidate_path_count": 1 + len(alternatives),
                    **prefix_audits[state_condition],
                    **query_audit,
                    "continuation_hook_audit": continuation_audit,
                    **outcomes,
                }
            )
            return outcomes, query_captures, path_captures

        _source_outcomes, source_query_captures, source_path_captures = run_panel(
            source_state,
            "intact_full_path",
            capture_heads=active_selected,
        )
        run_panel(carrier_state, "intact_full_path")
        for condition, heads in (
            ("selected_mask_full_path", active_selected),
            ("layer_matched_random_mask_full_path", random_heads),
        ):
            run_panel(source_state, condition, zero_heads=heads)
            run_panel(carrier_state, condition, zero_heads=heads)
        run_panel(
            carrier_state,
            "selected_restore_full_path_from_restored_state",
            capture_heads=active_selected,
            query_replacements=source_query_captures,
            continuation_replacements=source_path_captures,
            restoration_source=source_state,
        )
        if len(panel_rows) != 7:
            raise RuntimeError(
                f"Full-path local serial emitted {len(panel_rows)} rows, expected 7"
            )
        return panel_rows

    rows: list[dict[str, Any]] = []
    captures_by_state: dict[str, dict[int, torch.Tensor]] = {}
    for state_condition in REGISTERED_HTML_LOCAL_SERIAL_STATES:
        active_encoding = state_conditions[state_condition][0]
        for head_condition, zero_heads in (
            ("intact", ()),
            ("selected_mask", active_selected),
            ("layer_matched_random_mask", random_heads),
        ):
            query_output, captures, hook_audit = (
                _query_from_prefix_with_mediation_hooks(
                    model,
                    adapter,
                    active_encoding,
                    prefix_output=clone_prefill_output_for_scoring(
                        prefixes[state_condition]
                    ),
                    query_attention_mask=query_attention_mask,
                    source_layer=source_layer,
                    query_replacement_state=None,
                    zero_heads=zero_heads,
                    capture_heads=(active_selected if head_condition == "intact" else ()),
                )
            )
            if head_condition == "intact":
                captures_by_state[state_condition] = captures
            outcomes = _html_local_target_outcomes(
                model,
                tokenizer,
                active_encoding,
                query_output,
                target_path=target_path,
                city_token_offset=city_offset,
                target_city=target_city,
                alternative_city_paths=alternatives,
                known_cities=known_cities,
                run_greedy=run_greedy,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    **common,
                    "state_condition": state_condition,
                    "head_condition": head_condition,
                    "head_restoration_source_state": None,
                    **prefix_audits[state_condition],
                    **hook_audit,
                    **outcomes,
                }
            )

    mediator_source = "uninformative_target_restore"
    receiver_state = "uninformative"
    restored_output, _captures, restored_audit = (
        _query_from_prefix_with_mediation_hooks(
            model,
            adapter,
            state_conditions[receiver_state][0],
            prefix_output=clone_prefill_output_for_scoring(prefixes[receiver_state]),
            query_attention_mask=query_attention_mask,
            source_layer=source_layer,
            query_replacement_state=None,
            capture_heads=active_selected,
            head_replacements=captures_by_state[mediator_source],
        )
    )
    restored_outcomes = _html_local_target_outcomes(
        model,
        tokenizer,
        state_conditions[receiver_state][0],
        restored_output,
        target_path=target_path,
        city_token_offset=city_offset,
        target_city=target_city,
        alternative_city_paths=alternatives,
        known_cities=known_cities,
        run_greedy=run_greedy,
        max_new_tokens=max_new_tokens,
    )
    rows.append(
        {
            **common,
            "state_condition": receiver_state,
            "head_condition": "selected_restore_from_restored_state",
            "head_restoration_source_state": mediator_source,
            "head_restoration_mode": "selected_pre_o_slice_restore",
            **prefix_audits[receiver_state],
            **restored_audit,
            **restored_outcomes,
        }
    )
    return rows


@torch.inference_mode()
def run_endpoint_boundary_transplant_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    receiver_occurrence: int,
    donor_occurrence: int,
    layer: int,
    center: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
    conditions: Sequence[str] = REGISTERED_BOUNDARY_CONDITIONS,
    random_seed: int = 0,
    max_new_tokens: int = 64,
) -> list[dict[str, Any]]:
    """Transplant terminal/nonterminal endpoint states and freely continue."""

    requested = tuple(str(value) for value in conditions)
    if len(set(requested)) != len(requested):
        raise ValueError("Boundary conditions must be unique")
    unknown = sorted(set(requested) - set(REGISTERED_BOUNDARY_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown boundary conditions: {unknown}")
    if not {"clean", "self_patch"} <= set(requested):
        raise ValueError("Boundary trials require clean and self_patch")
    count = len(gold_records(row))
    receiver = int(receiver_occurrence)
    donor = int(donor_occurrence)
    if not 1 <= receiver <= count or not 1 <= donor <= count or donor == receiver:
        raise ValueError("Boundary donor/receiver occurrences are invalid")
    terminal_injection = donor == count and receiver < count
    nonterminal_injection = receiver == count and donor < count
    if not (terminal_injection or nonterminal_injection):
        raise ValueError(
            "Boundary trial must be middle->terminal or terminal->nonterminal"
        )

    answer_encoding, registry = build_answer_source_registry(row, tokenizer)
    endpoints = tuple(end - 1 for _start, end in registry.trace_items)
    receiver_position = int(endpoints[receiver - 1])
    donor_position = int(endpoints[donor - 1])
    source_layer = int(layer)
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        answer_encoding,
        [receiver_position, donor_position],
        layers=[source_layer],
    )
    receiver_state, donor_state = captured[source_layer]
    states, state_audit = native_loop_condition_states(
        receiver_state,
        donor_state,
        center,
        basis,
        random_seed=int(random_seed),
    )
    output_query = receiver_position - int(answer_encoding.prompt_token_count)
    local_encoding = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=output_query,
        sequence_output_token_end=output_query + 1,
        selected_site={
            "site_id": f"item_end:{receiver}",
            "site_kind": "item_end",
            "occurrence": receiver,
            "anchor_role": "p0_item_end",
        },
    )
    if int(local_encoding.query_position) != receiver_position:
        raise RuntimeError("Boundary local encoding moved the receiver endpoint")
    records = tuple(gold_records(row))
    known_cities = tuple(str(value["city"]) for value in records)
    receiver_successor = (
        str(records[receiver]["city"]) if receiver < count else None
    )
    donor_successor = str(records[donor]["city"]) if donor < count else None
    panel_kind = (
        "terminal_injection" if terminal_injection else "nonterminal_injection"
    )
    common = {
        "schema_version": NATIVE_LOOP_SCHEMA_VERSION,
        "experiment_id": "endpoint_state_update_stop_transplant",
        "panel_kind": panel_kind,
        "request_id": str(row["request_id"]),
        "model_label": str(answer_encoding.model_label),
        "seed": int(row["seed"]),
        "gold_count": count,
        "layer": source_layer,
        "receiver_occurrence": receiver,
        "donor_occurrence": donor,
        "donor_offset": donor - receiver,
        "receiver_is_terminal": receiver == count,
        "donor_is_terminal": donor == count,
        "receiver_position": receiver_position,
        "donor_position": donor_position,
        "receiver_expected_successor_city": receiver_successor,
        "donor_expected_successor_city": donor_successor,
        "selection_rank_used": False,
        **{key: value for key, value in state_audit.items() if key != "condition_audit"},
    }
    results: list[dict[str, Any]] = []
    for condition in requested:
        replacement_state = states[condition]
        prefill, applications, realized_norm = _prefill_with_state_replacements(
            model,
            adapter,
            local_encoding,
            layer=source_layer,
            positions=(receiver_position,),
            states=(
                None
                if replacement_state is None
                else replacement_state.reshape(1, -1)
            ),
        )
        completion = generate_answer_completion_from_prefill(
            model,
            tokenizer,
            local_encoding,
            prefill,
            max_new_tokens=int(max_new_tokens),
        )
        completion_text = str(completion["completion_text"])
        generated_city, city_start, city_evidence = _first_generated_city_record(
            completion_text, known_cities
        )
        numeric = completion_metrics(completion, gold_count=count)
        continued = generated_city is not None
        boundary_target_adoption = (
            (not continued)
            if terminal_injection
            else bool(
                donor_successor is not None
                and generated_city is not None
                and generated_city.casefold() == donor_successor.casefold()
            )
        )
        results.append(
            {
                **common,
                "condition": condition,
                "status": "ok",
                "local_patch_hook_applications": int(applications),
                "local_patch_realized_fro_norm": float(realized_norm),
                **state_audit["condition_audit"][condition],
                "local_completion_text": completion_text,
                "local_generation_truncated": bool(
                    completion["generation_truncated"]
                ),
                "first_generated_city_record": generated_city,
                "first_generated_city_record_char_start": city_start,
                "first_generated_city_record_evidence": city_evidence,
                "continued_with_known_city": bool(continued),
                "stopped_before_known_city": bool(not continued),
                "receiver_successor_retention": bool(
                    receiver_successor is not None
                    and generated_city is not None
                    and generated_city.casefold() == receiver_successor.casefold()
                ),
                "donor_successor_adoption": bool(
                    donor_successor is not None
                    and generated_city is not None
                    and generated_city.casefold() == donor_successor.casefold()
                ),
                "boundary_target_adoption": bool(boundary_target_adoption),
                **numeric,
            }
        )
    return results
