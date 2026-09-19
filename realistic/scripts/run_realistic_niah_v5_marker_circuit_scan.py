#!/usr/bin/env python3
"""Localize the marker-ledger mechanism by cache components, layers, and edges.

The assay has three intervention families over the same equal-length valid-item
and markerless-valid-payload histories:

1. transplant only the inserted marker's K, V, or K+V in every decoder layer;
2. repeat those marker transplants in disjoint four-layer bands; and
3. leave every cached state unchanged but delete exactly one suffix-query to
   inserted-event-key attention edge with an explicit 4D causal mask.

All token ids, absolute positions, and suffixes are held fixed within a causal
comparison.  The markerless history receives the same absolute-position edge
masks as a position-matched control.  Custom-4D clean baselines are recorded
separately so mask-format/kernel differences cannot be mistaken for an edge
effect.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import (  # noqa: E402
    _accepts_keyword,
    _bounded_logits_kwargs,
    _encoding_tensors,
)
from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.causal import sign_flip_pvalue  # noqa: E402
from realistic_niah_v5.event_cache_splice import (  # noqa: E402
    clone_cache,
    splice_cache_positions,
)
from realistic_niah_v5.marker_circuit import (  # noqa: E402
    build_cached_suffix_causal_mask,
    edge_key_positions,
    marker_layer_bands,
    mask_single_attention_edge,
)
from scripts.run_realistic_niah_v5_event_cache_splice import (  # noqa: E402
    PRIMARY_INVALID_VARIANT,
    PRIMARY_READ_LAYER,
    VALID_VARIANT,
    advance_event_cache,
    append_condition_rows,
    build_cache_splice_geometry,
    capture_target_from_cache,
    prefill_common_prefix,
    state_equivalence,
)
from scripts.run_realistic_niah_v5_boundary_equivariance import (  # noqa: E402
    decode_count_probe,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_event_commit_movie import (  # noqa: E402
    build_event_movie_geometry,
)
from scripts.run_realistic_niah_v5_list_event_edit_scan import (  # noqa: E402
    build_list_event_variants,
)
from scripts.run_realistic_niah_v5_overwrite_mechanism_scan import (  # noqa: E402
    _load_probes,
)
from scripts.run_realistic_niah_v5_unified_carrier_transition import (  # noqa: E402
    _read_rows,
)


SCHEMA_VERSION = "marker_circuit_scan_v1"
COMPONENTS = {
    "K": ("key",),
    "V": ("value",),
    "KV": ("key", "value"),
}
EDGE_QUERY_ROLES = ("target_marker", "target_boundary")
EDGE_KEY_ROLES = (
    "inserted_marker",
    "inserted_payload_first",
    "inserted_payload_mid",
    "inserted_closing",
    "pre_insertion_b5",
)


def _mask_value_is_allowed(value: torch.Tensor) -> bool:
    """Interpret the boolean or additive mask value received by attention."""

    scalar = value.detach().reshape(()).item()
    if value.dtype == torch.bool:
        return bool(scalar)
    # Transformers additive masks use 0 for allowed and a very negative value
    # for blocked edges.  This also accepts finite fp16/bf16 minimum values.
    return float(scalar) > -1.0e4


@torch.inference_mode()
def capture_target_with_4d_mask(
    model: Any,
    adapter: Any,
    encoding: Any,
    past: Any,
    attention_mask: torch.Tensor,
    *,
    prefix_length: int,
    target_position: int,
    read_layers: Sequence[int],
    audited_query_position: int,
    audited_key_position: int,
    expected_edge_allowed: bool,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Capture target states and audit one exact mask edge in every layer."""

    start = int(prefix_length)
    target = int(target_position)
    query = int(audited_query_position)
    key = int(audited_key_position)
    layers = tuple(sorted({int(value) for value in read_layers}))
    suffix_width = target + 1 - start
    relative_target = target - start
    relative_query = query - start
    expected_shape = (1, 1, suffix_width, target + 1)
    if tuple(attention_mask.shape) != expected_shape:
        raise ValueError(
            f"Explicit attention mask has shape {tuple(attention_mask.shape)}, "
            f"expected {expected_shape}"
        )
    if not 0 <= relative_query < suffix_width or not 0 <= key <= query:
        raise ValueError("Audited query/key edge lies outside the cached suffix graph")

    input_ids, _standard_mask = _encoding_tensors(model, encoding)
    device = input_ids.device
    positions = torch.arange(start, target + 1, dtype=torch.long, device=device)
    explicit_mask = attention_mask.to(device=device)
    captured: dict[int, torch.Tensor] = {}
    runtime_rows: dict[int, dict[str, Any]] = {}
    handles = []

    for layer in layers:

        def block_hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Decoder block input is not a positional tensor")
            hidden = args[0]
            if hidden.ndim == 3 and int(hidden.shape[1]) == suffix_width:
                captured[layer] = hidden[0, relative_target].detach().float().cpu()

        handles.append(adapter.layers[layer].register_forward_pre_hook(block_hook))

    for layer, attention in enumerate(adapter.attentions):

        def attention_hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            *,
            layer: int = layer,
        ) -> None:
            runtime_mask = kwargs.get("attention_mask")
            if not isinstance(runtime_mask, torch.Tensor):
                raise RuntimeError(
                    f"Layer {layer} attention did not receive a tensor mask"
                )
            if runtime_mask.ndim != 4:
                raise RuntimeError(
                    f"Layer {layer} received a non-4D attention mask: "
                    f"{tuple(runtime_mask.shape)}"
                )
            if int(runtime_mask.shape[-2]) != suffix_width or int(
                runtime_mask.shape[-1]
            ) != target + 1:
                raise RuntimeError(
                    f"Layer {layer} changed explicit mask geometry to "
                    f"{tuple(runtime_mask.shape)}"
                )
            value = runtime_mask[0, 0, relative_query, key]
            allowed = _mask_value_is_allowed(value)
            runtime_rows[layer] = {
                "layer": int(layer),
                "mask_dtype": str(runtime_mask.dtype),
                "mask_shape": [int(size) for size in runtime_mask.shape],
                "audited_edge_allowed": bool(allowed),
                "audited_edge_value": float(value.detach().float().item()),
            }
            if allowed != bool(expected_edge_allowed):
                raise RuntimeError(
                    f"Layer {layer} audited edge allowed={allowed}, expected "
                    f"{bool(expected_edge_allowed)}"
                )

        handles.append(
            attention.register_forward_pre_hook(attention_hook, with_kwargs=True)
        )

    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, start : target + 1],
        "attention_mask": explicit_mask,
        "past_key_values": past,
        "use_cache": False,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = positions.unsqueeze(0)
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = positions
    try:
        model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()

    missing_states = sorted(set(layers) - set(captured))
    missing_masks = sorted(set(range(int(adapter.num_layers))) - set(runtime_rows))
    if missing_states:
        raise RuntimeError(f"Explicit-mask target capture missed layers {missing_states}")
    if missing_masks:
        raise RuntimeError(f"Explicit-mask runtime audit missed layers {missing_masks}")
    observed = {bool(row["audited_edge_allowed"]) for row in runtime_rows.values()}
    if observed != {bool(expected_edge_allowed)}:
        raise RuntimeError("Attention layers disagree about the audited edge")
    return captured, {
        "audited_query_position": query,
        "audited_query_relative_position": relative_query,
        "audited_key_position": key,
        "expected_edge_allowed": bool(expected_edge_allowed),
        "all_layers_observed": True,
        "all_layers_expected_edge_state": True,
        "attention_layer_count": len(runtime_rows),
        "runtime_mask_dtypes": sorted(
            {str(row["mask_dtype"]) for row in runtime_rows.values()}
        ),
        "runtime_mask_shapes": sorted(
            {tuple(row["mask_shape"]) for row in runtime_rows.values()}
        ),
        "per_layer": [runtime_rows[layer] for layer in sorted(runtime_rows)],
    }


def _append_rows(
    rows: list[dict[str, Any]],
    *,
    metadata: Mapping[str, Any],
    **kwargs: Any,
) -> None:
    """Reuse the frozen probe/state metrics and attach circuit metadata."""

    start = len(rows)
    append_condition_rows(rows, **kwargs)
    for row in rows[start:]:
        row["schema_version"] = SCHEMA_VERSION
        row.update(metadata)


def _mean_cell(
    active: Sequence[Mapping[str, Any]], *, fields: Sequence[str]
) -> dict[str, Any]:
    progress = [
        float(row["donor_axis_progress"])
        for row in active
        if row["donor_axis_progress"] is not None
    ]
    return {
        **{field: active[0].get(field) for field in fields},
        "n_trials": len(active),
        "n_seeds": len({int(row["seed"]) for row in active}),
        "mean_donor_axis_progress": fmean(progress) if progress else None,
        "median_donor_axis_progress": (
            float(np.median(progress)) if progress else None
        ),
        "positive_axis_progress_rate": (
            float(np.mean(np.asarray(progress) > 0)) if progress else None
        ),
        "axis_progress_sign_flip_pvalue": (
            sign_flip_pvalue(progress) if progress else None
        ),
        "mean_off_axis_norm_over_clean_contrast": fmean(
            float(row["off_axis_norm_over_clean_contrast"])
            for row in active
            if row["off_axis_norm_over_clean_contrast"] is not None
        ),
        "prediction_counts": {
            str(label): int(count)
            for label, count in sorted(
                Counter(int(row["probe_prediction"]) for row in active).items()
            )
        },
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize prespecified cells and apply deterministic discovery selectors."""

    splice_fields = (
        "condition",
        "component_label",
        "layer_band",
        "causally_precedes_read",
        "donor_variant",
        "receiver_variant",
        "read_layer",
    )
    splice_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["condition"]) not in {"marker_component", "marker_layer_band"}:
            continue
        key = tuple(row.get(field) for field in splice_fields)
        splice_groups[key].append(row)
    splice_cells = [
        _mean_cell(active, fields=splice_fields)
        for _key, active in sorted(splice_groups.items(), key=lambda pair: str(pair[0]))
    ]

    edge_fields = (
        "condition",
        "receiver_variant",
        "donor_variant",
        "edge_query_role",
        "edge_key_role",
        "read_layer",
    )
    edge_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["condition"]) != "exact_attention_edge_mask":
            continue
        key = tuple(row.get(field) for field in edge_fields)
        edge_groups[key].append(row)
    edge_cells = [
        _mean_cell(active, fields=edge_fields)
        for _key, active in sorted(edge_groups.items(), key=lambda pair: str(pair[0]))
    ]

    def bidirectional_splice_scores(condition: str) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for cell in splice_cells:
            if (
                str(cell["condition"]) == condition
                and int(cell["read_layer"]) == PRIMARY_READ_LAYER
                and cell["mean_donor_axis_progress"] is not None
            ):
                label = (
                    str(cell["component_label"]),
                    str(cell["layer_band"]),
                )
                grouped[label].append(float(cell["mean_donor_axis_progress"]))
        return [
            {
                "component_label": component,
                "layer_band": band,
                "direction_count": len(values),
                "bidirectional_mean_donor_axis_progress": fmean(values),
            }
            for (component, band), values in sorted(grouped.items())
        ]

    component_scores = bidirectional_splice_scores("marker_component")
    band_scores = bidirectional_splice_scores("marker_layer_band")
    component_candidates = [
        cell for cell in component_scores if cell["component_label"] in {"K", "V"}
    ]
    band_candidates = [
        cell
        for cell in band_scores
        if cell["component_label"] == "KV"
        and not str(cell["layer_band"]).endswith("postread_control")
    ]
    valid_edge_candidates = [
        cell
        for cell in edge_cells
        if str(cell["receiver_variant"]) == VALID_VARIANT
        and str(cell["edge_key_role"]) == "inserted_marker"
        and int(cell["read_layer"]) == PRIMARY_READ_LAYER
    ]
    winning_component = (
        max(
            component_candidates,
            key=lambda cell: float(cell["bidirectional_mean_donor_axis_progress"]),
        )
        if component_candidates
        else None
    )
    winning_band = (
        max(
            band_candidates,
            key=lambda cell: float(cell["bidirectional_mean_donor_axis_progress"]),
        )
        if band_candidates
        else None
    )
    winning_edge = (
        max(valid_edge_candidates, key=lambda cell: float(cell["mean_donor_axis_progress"]))
        if valid_edge_candidates
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "splice_cells": splice_cells,
        "edge_cells": edge_cells,
        "component_bidirectional_scores_l24": component_scores,
        "band_bidirectional_scores_l24": band_scores,
        "deterministic_discovery_selection": {
            "component_rule": (
                "argmax bidirectional L24 donor-axis progress: all-layer marker K vs V"
            ),
            "winning_component": winning_component,
            "band_rule": (
                "argmax bidirectional L24 donor-axis progress: pre-read marker K+V bands"
            ),
            "winning_band": winning_band,
            "edge_rule": (
                "argmax valid-branch L24 progress toward markerless clean state after "
                "masking target-marker or target-boundary query to inserted-marker key"
            ),
            "winning_edge_query": winning_edge,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--receiver", type=int, default=5)
    parser.add_argument("--read-layers", type=int, nargs="+", default=[15, 16, 24])
    parser.add_argument("--insert-source-occurrence", type=int, default=4)
    parser.add_argument("--delete-occurrence", type=int, default=3)
    parser.add_argument("--band-width", type=int, default=4)
    parser.add_argument(
        "--families",
        choices=("components", "bands", "edges"),
        nargs="+",
        default=["components", "bands", "edges"],
    )
    parser.add_argument(
        "--component-labels", choices=tuple(COMPONENTS), nargs="+", default=list(COMPONENTS)
    )
    parser.add_argument(
        "--edge-query-roles",
        choices=EDGE_QUERY_ROLES,
        nargs="+",
        default=list(EDGE_QUERY_ROLES),
    )
    parser.add_argument(
        "--edge-key-roles",
        choices=EDGE_KEY_ROLES,
        nargs="+",
        default=list(EDGE_KEY_ROLES),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "marker-circuit-scan"

    seeds = tuple(dict.fromkeys(int(value) for value in args.evaluation_seeds))
    read_layers = tuple(sorted({int(value) for value in args.read_layers}))
    families = tuple(dict.fromkeys(str(value) for value in args.families))
    component_labels = tuple(
        dict.fromkeys(str(value) for value in args.component_labels)
    )
    edge_query_roles = tuple(
        dict.fromkeys(str(value) for value in args.edge_query_roles)
    )
    edge_key_roles = tuple(dict.fromkeys(str(value) for value in args.edge_key_roles))
    if not seeds or PRIMARY_READ_LAYER not in read_layers:
        raise ValueError("Seeds and the preregistered L24 read are required")

    probes = _load_probes(args.frozen_probes, read_layers)
    source_rows = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    if max(read_layers) >= int(adapter.num_layers):
        raise ValueError("A read layer is outside the decoder")
    all_layers = tuple(range(int(adapter.num_layers)))
    bands = marker_layer_bands(
        num_layers=int(adapter.num_layers),
        read_layer=PRIMARY_READ_LAYER,
        band_width=int(args.band_width),
    )

    trials: list[dict[str, Any]] = []
    geometry_audits: list[dict[str, Any]] = []
    custom_mask_equivalence: list[dict[str, Any]] = []
    edge_mask_audits: list[dict[str, Any]] = []

    for seed in seeds:
        row = source_rows[seed]
        source, blank, registry, _scrub_audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        boundaries = {
            occurrence: select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=occurrence
            )[0]
            for occurrence in range(1, 11)
        }
        variants = {
            str(variant["event_variant"]): variant
            for variant in build_list_event_variants(
                source,
                blank,
                registry,
                receiver=int(args.receiver),
                current_boundary=boundaries[int(args.receiver)],
                target_boundary=boundaries[int(args.receiver) + 1],
                insert_source_occurrence=int(args.insert_source_occurrence),
                delete_occurrence=int(args.delete_occurrence),
            )
        }
        valid = variants[VALID_VARIANT]
        invalid = variants[PRIMARY_INVALID_VARIANT]
        geometry = build_cache_splice_geometry(
            valid,
            invalid,
            registry,
            boundaries,
            receiver=int(args.receiver),
            insert_source_occurrence=int(args.insert_source_occurrence),
        )
        movie = build_event_movie_geometry(
            valid,
            registry,
            boundaries,
            receiver=int(args.receiver),
            insert_source_occurrence=int(args.insert_source_occurrence),
        )
        insertion_start = int(geometry["insertion_start"])
        event_end = int(geometry["event_end"])
        target_position = int(geometry["target_position"])
        query_positions = {
            "target_marker": int(movie["landmarks"]["target_marker_end"]),
            "target_boundary": int(movie["landmarks"]["target_boundary"]),
        }
        key_positions = edge_key_positions(geometry)
        if any(position < event_end for position in query_positions.values()):
            raise RuntimeError("An exact-edge query lies before the cached suffix")
        geometry_audits.append(
            {
                "seed": int(seed),
                "insertion_start": insertion_start,
                "event_end": event_end,
                "target_position": target_position,
                "query_positions": query_positions,
                "key_positions": key_positions,
                "regions": {
                    key: list(value) for key, value in geometry["regions"].items()
                },
                "changed_event_token_positions": geometry[
                    "changed_event_token_positions"
                ],
            }
        )

        common = prefill_common_prefix(model, valid["encoding"], end=insertion_start)
        caches = {
            VALID_VARIANT: advance_event_cache(
                model,
                valid["encoding"],
                common,
                start=insertion_start,
                end=event_end,
            ),
            PRIMARY_INVALID_VARIANT: advance_event_cache(
                model,
                invalid["encoding"],
                common,
                start=insertion_start,
                end=event_end,
            ),
        }
        del common
        encodings = {
            VALID_VARIANT: valid["encoding"],
            PRIMARY_INVALID_VARIANT: invalid["encoding"],
        }
        clean_states = {
            label: capture_target_from_cache(
                model,
                adapter,
                encodings[label],
                clone_cache(caches[label]),
                prefix_length=event_end,
                target_position=target_position,
                read_layers=read_layers,
            )
            for label in (VALID_VARIANT, PRIMARY_INVALID_VARIANT)
        }

        for label in (VALID_VARIANT, PRIMARY_INVALID_VARIANT):
            _append_rows(
                trials,
                metadata={
                    "family": "clean_cache",
                    "component_label": None,
                    "layer_band": None,
                    "causally_precedes_read": None,
                    "edge_query_role": None,
                    "edge_key_role": None,
                },
                seed=seed,
                request_id=str(row["request_id"]),
                condition="clean_cache",
                region=None,
                components=(),
                donor_variant=None,
                receiver_variant=label,
                captured=clean_states[label],
                clean_states=clean_states,
                probes=probes,
                splice_audit=None,
            )

        directions = (
            (VALID_VARIANT, PRIMARY_INVALID_VARIANT),
            (PRIMARY_INVALID_VARIANT, VALID_VARIANT),
        )
        if "components" in families:
            for donor_variant, receiver_variant in directions:
                for component_label in component_labels:
                    components = COMPONENTS[component_label]
                    hybrid, audit = splice_cache_positions(
                        caches[receiver_variant],
                        caches[donor_variant],
                        positions=geometry["regions"]["marker"],
                        layers=all_layers,
                        components=components,
                    )
                    captured = capture_target_from_cache(
                        model,
                        adapter,
                        encodings[receiver_variant],
                        hybrid,
                        prefix_length=event_end,
                        target_position=target_position,
                        read_layers=read_layers,
                    )
                    _append_rows(
                        trials,
                        metadata={
                            "family": "marker_component",
                            "component_label": component_label,
                            "layer_band": "all_layers",
                            "causally_precedes_read": None,
                            "edge_query_role": None,
                            "edge_key_role": None,
                        },
                        seed=seed,
                        request_id=str(row["request_id"]),
                        condition="marker_component",
                        region="marker",
                        components=components,
                        donor_variant=donor_variant,
                        receiver_variant=receiver_variant,
                        captured=captured,
                        clean_states=clean_states,
                        probes=probes,
                        splice_audit=audit,
                    )
                    del hybrid

        if "bands" in families:
            for donor_variant, receiver_variant in directions:
                for band in bands:
                    for component_label in component_labels:
                        components = COMPONENTS[component_label]
                        hybrid, audit = splice_cache_positions(
                            caches[receiver_variant],
                            caches[donor_variant],
                            positions=geometry["regions"]["marker"],
                            layers=band["layers"],
                            components=components,
                        )
                        captured = capture_target_from_cache(
                            model,
                            adapter,
                            encodings[receiver_variant],
                            hybrid,
                            prefix_length=event_end,
                            target_position=target_position,
                            read_layers=read_layers,
                        )
                        _append_rows(
                            trials,
                            metadata={
                                "family": "marker_layer_band",
                                "component_label": component_label,
                                "layer_band": str(band["label"]),
                                "causally_precedes_read": bool(
                                    band["causally_precedes_read"]
                                ),
                                "edge_query_role": None,
                                "edge_key_role": None,
                            },
                            seed=seed,
                            request_id=str(row["request_id"]),
                            condition="marker_layer_band",
                            region="marker",
                            components=components,
                            donor_variant=donor_variant,
                            receiver_variant=receiver_variant,
                            captured=captured,
                            clean_states=clean_states,
                            probes=probes,
                            splice_audit=audit,
                        )
                        del hybrid

        if "edges" in families:
            custom_clean: dict[str, dict[int, torch.Tensor]] = {}
            clean_masks: dict[str, torch.Tensor] = {}
            audit_query = query_positions[edge_query_roles[0]]
            audit_key = key_positions[edge_key_roles[0]]
            for label in (VALID_VARIANT, PRIMARY_INVALID_VARIANT):
                clean_mask = build_cached_suffix_causal_mask(
                    encodings[label].attention_mask,
                    prefix_length=event_end,
                    end=target_position + 1,
                    device=_encoding_tensors(model, encodings[label])[0].device,
                )
                clean_masks[label] = clean_mask
                captured, runtime_audit = capture_target_with_4d_mask(
                    model,
                    adapter,
                    encodings[label],
                    clone_cache(caches[label]),
                    clean_mask,
                    prefix_length=event_end,
                    target_position=target_position,
                    read_layers=read_layers,
                    audited_query_position=audit_query,
                    audited_key_position=audit_key,
                    expected_edge_allowed=True,
                )
                custom_clean[label] = captured
                for layer in read_layers:
                    equivalence = state_equivalence(
                        clean_states[label][layer], captured[layer]
                    )
                    standard_decoded = decode_count_probe(
                        probes[layer], clean_states[label][layer].numpy()
                    )
                    custom_decoded = decode_count_probe(
                        probes[layer], captured[layer].numpy()
                    )
                    custom_mask_equivalence.append(
                        {
                            "seed": int(seed),
                            "event_variant": label,
                            "read_layer": int(layer),
                            **equivalence,
                            "standard_probe_prediction": int(
                                standard_decoded["probe_prediction"]
                            ),
                            "custom_probe_prediction": int(
                                custom_decoded["probe_prediction"]
                            ),
                            "probe_prediction_match": int(
                                standard_decoded["probe_prediction"]
                            )
                            == int(custom_decoded["probe_prediction"]),
                            "runtime_mask_audit": {
                                key: value
                                for key, value in runtime_audit.items()
                                if key != "per_layer"
                            },
                        }
                    )
                    if (
                        equivalence["cosine_similarity"] < 0.999
                        or int(standard_decoded["probe_prediction"])
                        != int(custom_decoded["probe_prediction"])
                    ):
                        raise RuntimeError(
                            "Explicit 4D clean mask failed standard-cache equivalence"
                        )
                _append_rows(
                    trials,
                    metadata={
                        "family": "custom_4d_clean",
                        "component_label": None,
                        "layer_band": None,
                        "causally_precedes_read": None,
                        "edge_query_role": None,
                        "edge_key_role": None,
                    },
                    seed=seed,
                    request_id=str(row["request_id"]),
                    condition="custom_4d_clean",
                    region=None,
                    components=(),
                    donor_variant=None,
                    receiver_variant=label,
                    captured=captured,
                    clean_states=custom_clean,
                    probes=probes,
                    splice_audit=None,
                )

            for receiver_variant in (VALID_VARIANT, PRIMARY_INVALID_VARIANT):
                donor_variant = (
                    PRIMARY_INVALID_VARIANT
                    if receiver_variant == VALID_VARIANT
                    else VALID_VARIANT
                )
                for query_role in edge_query_roles:
                    query_position = query_positions[query_role]
                    for key_role in edge_key_roles:
                        key_position = key_positions[key_role]
                        masked, construction_audit = mask_single_attention_edge(
                            clean_masks[receiver_variant],
                            prefix_length=event_end,
                            query_position=query_position,
                            key_position=key_position,
                        )
                        captured, runtime_audit = capture_target_with_4d_mask(
                            model,
                            adapter,
                            encodings[receiver_variant],
                            clone_cache(caches[receiver_variant]),
                            masked,
                            prefix_length=event_end,
                            target_position=target_position,
                            read_layers=read_layers,
                            audited_query_position=query_position,
                            audited_key_position=key_position,
                            expected_edge_allowed=False,
                        )
                        edge_mask_audits.append(
                            {
                                "seed": int(seed),
                                "receiver_variant": receiver_variant,
                                "edge_query_role": query_role,
                                "edge_key_role": key_role,
                                "construction_audit": construction_audit,
                                "runtime_audit": {
                                    key: value
                                    for key, value in runtime_audit.items()
                                    if key != "per_layer"
                                },
                            }
                        )
                        _append_rows(
                            trials,
                            metadata={
                                "family": "exact_attention_edge_mask",
                                "component_label": None,
                                "layer_band": "all_layers_all_heads_one_edge",
                                "causally_precedes_read": None,
                                "edge_query_role": query_role,
                                "edge_key_role": key_role,
                                "edge_query_position": query_position,
                                "edge_key_position": key_position,
                                "edge_masked_count": 1,
                                "edge_runtime_all_layers_verified": bool(
                                    runtime_audit[
                                        "all_layers_expected_edge_state"
                                    ]
                                ),
                            },
                            seed=seed,
                            request_id=str(row["request_id"]),
                            condition="exact_attention_edge_mask",
                            region=None,
                            components=(),
                            donor_variant=donor_variant,
                            receiver_variant=receiver_variant,
                            captured=captured,
                            clean_states=custom_clean,
                            probes=probes,
                            splice_audit=None,
                        )

        del caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[marker-circuit-scan] seed={seed} complete", flush=True)

    summary = {
        **summarize(trials),
        "model_label": str(args.model),
        "evaluation_seeds": list(seeds),
        "receiver": int(args.receiver),
        "read_layers": list(read_layers),
        "families": list(families),
        "component_labels": list(component_labels),
        "edge_query_roles": list(edge_query_roles),
        "edge_key_roles": list(edge_key_roles),
        "layer_bands": [
            {
                "label": str(band["label"]),
                "layers": list(band["layers"]),
                "causally_precedes_read": bool(band["causally_precedes_read"]),
            }
            for band in bands
        ],
        "geometry_audits": geometry_audits,
        "custom_4d_clean_equivalence": custom_mask_equivalence,
        "custom_4d_clean_all_probe_predictions_match": all(
            bool(row["probe_prediction_match"]) for row in custom_mask_equivalence
        ),
        "custom_4d_clean_min_cosine_similarity": (
            min(float(row["cosine_similarity"]) for row in custom_mask_equivalence)
            if custom_mask_equivalence
            else None
        ),
        "edge_mask_audits": edge_mask_audits,
        "all_edge_masks_delete_exactly_one_allowed_edge": all(
            int(row["construction_audit"]["masked_edge_count"]) == 1
            and bool(row["runtime_audit"]["all_layers_expected_edge_state"])
            for row in edge_mask_audits
        ),
        "trial_count": len(trials),
        "estimand_note": (
            "Cache splice rows measure movement from the clean receiver toward the "
            "clean donor. Edge rows use custom-4D clean endpoints and measure "
            "movement toward the opposite event branch; the markerless branch is "
            "an absolute-position matched control, not a donor-cache transplant."
        ),
    }
    _atomic_jsonl(args.output, trials)
    _atomic_json(args.summary, summary)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "evaluation_seeds": list(seeds),
                "output": str(args.output),
                "summary": str(args.summary),
                "trial_count": len(trials),
                "custom_4d_clean_all_probe_predictions_match": summary[
                    "custom_4d_clean_all_probe_predictions_match"
                ],
                "custom_4d_clean_min_cosine_similarity": summary[
                    "custom_4d_clean_min_cosine_similarity"
                ],
                "all_edge_masks_delete_exactly_one_allowed_edge": summary[
                    "all_edge_masks_delete_exactly_one_allowed_edge"
                ],
                "deterministic_discovery_selection": summary[
                    "deterministic_discovery_selection"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
