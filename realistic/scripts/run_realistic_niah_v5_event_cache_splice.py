#!/usr/bin/env python3
"""Test whether a valid list event leaves a localized K/V commit record.

For each seed, the script prefills one common history up to the insertion
point, clones that cache, and advances it through either a valid copied list
item or an equal-length markerless control.  It then transplants same-position
K/V fields between the two caches and feeds the identical original next item.

The full-event K+V swap is a positive implementation control: because the two
histories are identical before the inserted event, it reconstructs the donor
cache by construction.  The mechanistic tests are the strict subspans,
especially the shared-surface closing token.  A large closing-only effect would
support a compact transaction/commit record; effects distributed over marker
and payload positions instead support a distributed event ledger.
"""

from __future__ import annotations

import argparse
import json
import math
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
    cache_layer_views,
    cache_sequence_length,
    clone_cache,
    splice_cache_positions,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
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


SCHEMA_VERSION = "event_cache_splice_v1"
VALID_VARIANT = "insert_valid_item"
PRIMARY_INVALID_VARIANT = "insert_markerless_valid_payload"
PRIMARY_READ_LAYER = 24
REGION_COMPONENTS = (
    ("event", ("key", "value")),
    ("event", ("key",)),
    ("event", ("value",)),
    ("marker", ("key", "value")),
    ("payload", ("key", "value")),
    ("closing", ("key", "value")),
    ("preclosing", ("key", "value")),
    ("b5", ("key", "value")),
    ("preceding_event_width", ("key", "value")),
)


def build_cache_splice_geometry(
    valid_variant: Mapping[str, Any],
    invalid_variant: Mapping[str, Any],
    registry: Any,
    boundaries: Mapping[int, int],
    *,
    receiver: int,
    insert_source_occurrence: int,
) -> dict[str, Any]:
    """Freeze aligned event, subspan, cache-prefix, and target geometry."""

    if str(valid_variant["event_variant"]) != VALID_VARIANT:
        raise ValueError("The valid cache branch has the wrong event variant")
    if str(invalid_variant["event_variant"]) != PRIMARY_INVALID_VARIANT:
        raise ValueError("The invalid cache branch has the wrong event variant")
    delta = int(valid_variant["token_delta"])
    if delta <= 0 or int(invalid_variant["token_delta"]) != delta:
        raise ValueError("Valid and invalid event insertions are not equal length")
    valid_encoding = valid_variant["encoding"]
    invalid_encoding = invalid_variant["encoding"]
    if valid_encoding.sequence_length != invalid_encoding.sequence_length:
        raise ValueError("Event branches changed total sequence length")
    if tuple(valid_encoding.attention_mask) != tuple(invalid_encoding.attention_mask):
        raise ValueError("Event branches changed the attention mask")

    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    insertion_start = int(items[int(receiver)][0])
    event_end = insertion_start + delta
    target_position = int(valid_variant["target_boundary"])
    if target_position != int(invalid_variant["target_boundary"]):
        raise ValueError("Event branches do not share a target boundary")
    if event_end > target_position:
        raise ValueError("Inserted event overlaps the target boundary")
    valid_ids = tuple(int(value) for value in valid_encoding.input_ids)
    invalid_ids = tuple(int(value) for value in invalid_encoding.input_ids)
    if valid_ids[:insertion_start] != invalid_ids[:insertion_start]:
        raise ValueError("Event branches differ before the insertion point")
    if valid_ids[event_end : target_position + 1] != invalid_ids[
        event_end : target_position + 1
    ]:
        raise ValueError("Event branches do not have an identical target suffix")

    movie = build_event_movie_geometry(
        valid_variant,
        registry,
        boundaries,
        receiver=int(receiver),
        insert_source_occurrence=int(insert_source_occurrence),
    )
    event = tuple(range(insertion_start, event_end))
    event_set = set(event)
    marker = tuple(
        position
        for position in event
        if str(movie["roles"].get(position)) == "inserted_marker"
    )
    closing = (int(movie["landmarks"]["inserted_event_boundary"]),)
    payload = tuple(
        position
        for position in event
        if str(movie["roles"].get(position)) == "inserted_payload"
        and position not in closing
    )
    if not marker or not payload or closing[0] not in event_set:
        raise ValueError("Inserted marker/payload/closing geometry is incomplete")
    preclosing = tuple(position for position in event if position not in closing)
    preceding_start = insertion_start - delta
    if preceding_start < int(valid_encoding.prompt_token_count):
        raise ValueError("Equal-width preceding control crosses into the prompt")
    regions = {
        "event": event,
        "marker": marker,
        "payload": payload,
        "closing": closing,
        "preclosing": preclosing,
        "b5": (int(valid_variant["current_boundary"]),),
        "preceding_event_width": tuple(range(preceding_start, insertion_start)),
    }
    if set(marker) & set(payload) or set(marker) & set(closing):
        raise RuntimeError("Semantic cache regions overlap unexpectedly")
    return {
        "insertion_start": insertion_start,
        "event_end": event_end,
        "prefix_length": event_end,
        "target_position": target_position,
        "event_token_count": delta,
        "regions": regions,
        "valid_event_token_ids": list(valid_ids[insertion_start:event_end]),
        "invalid_event_token_ids": list(invalid_ids[insertion_start:event_end]),
        "changed_event_token_positions": [
            position
            for position in event
            if valid_ids[position] != invalid_ids[position]
        ],
        "identical_suffix_token_count": target_position + 1 - event_end,
        "closing_surface_token_identical": valid_ids[closing[0]]
        == invalid_ids[closing[0]],
    }


def _forward_from_cache(
    model: Any,
    encoding: Any,
    past: Any,
    *,
    start: int,
    end: int,
    use_cache: bool,
) -> Any:
    """Run encoding[start:end] at its unchanged absolute positions."""

    left, right = int(start), int(end)
    if not 0 <= left < right <= int(encoding.sequence_length):
        raise ValueError("Cached-forward token interval is invalid")
    if cache_sequence_length(past) != left:
        raise ValueError("Cached-forward start does not match cache length")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    device = input_ids.device
    positions = torch.arange(left, right, dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, left:right],
        "attention_mask": attention_mask[:, :right],
        "past_key_values": past,
        "use_cache": bool(use_cache),
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = positions.unsqueeze(0)
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = positions
    return model(**kwargs)


@torch.inference_mode()
def prefill_common_prefix(model: Any, encoding: Any, *, end: int) -> Any:
    """Materialize one reusable cache for the history before the event."""

    right = int(end)
    if not 1 <= right <= int(encoding.sequence_length):
        raise ValueError("Common-prefix endpoint is invalid")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    output = model(
        input_ids=input_ids[:, :right],
        attention_mask=attention_mask[:, :right],
        use_cache=True,
        output_attentions=False,
        **_bounded_logits_kwargs(model),
    )
    past = getattr(output, "past_key_values", None)
    if past is None or cache_sequence_length(past) != right:
        raise RuntimeError("Common prefix did not return the expected KV cache")
    if getattr(output, "shared_kv_states", None) is not None:
        raise RuntimeError("This experiment does not support shared-KV architectures")
    return past


@torch.inference_mode()
def advance_event_cache(
    model: Any,
    encoding: Any,
    common_cache: Any,
    *,
    start: int,
    end: int,
) -> Any:
    """Clone a common cache and append one event branch."""

    branch = clone_cache(common_cache)
    output = _forward_from_cache(
        model,
        encoding,
        branch,
        start=int(start),
        end=int(end),
        use_cache=True,
    )
    past = getattr(output, "past_key_values", None)
    if past is None or cache_sequence_length(past) != int(end):
        raise RuntimeError("Event branch did not append the expected KV fields")
    return past


@torch.inference_mode()
def capture_target_from_cache(
    model: Any,
    adapter: Any,
    encoding: Any,
    past: Any,
    *,
    prefix_length: int,
    target_position: int,
    read_layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Capture target block inputs while continuing from a frozen cache."""

    start = int(prefix_length)
    target = int(target_position)
    layers = tuple(sorted({int(value) for value in read_layers}))
    if not layers or any(not 0 <= layer < int(adapter.num_layers) for layer in layers):
        raise ValueError("Cached target read layers are invalid")
    if not start <= target < int(encoding.sequence_length):
        raise ValueError("Cached target position is outside the suffix")
    suffix_width = target + 1 - start
    relative_target = target - start
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Decoder block input is not a positional tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != suffix_width:
                return
            captured[layer] = hidden[0, relative_target].detach().float().cpu()

        handles.append(adapter.layers[layer].register_forward_pre_hook(hook))
    try:
        _forward_from_cache(
            model,
            encoding,
            past,
            start=start,
            end=target + 1,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(layers) - set(captured))
    if missing:
        raise RuntimeError(f"Cached target capture missed layers {missing}")
    return captured


def cache_difference(left: Any, right: Any) -> dict[str, Any]:
    """Return exact and maximum differences between two materialized caches."""

    left_views = cache_layer_views(left)
    right_views = cache_layer_views(right)
    if len(left_views) != len(right_views):
        raise ValueError("Compared caches have different layer counts")
    changed = 0
    maximum = 0.0
    for left_view, right_view in zip(left_views, right_views):
        for component in ("key", "value"):
            a = getattr(left_view, component)
            b = getattr(right_view, component)
            if a.shape != b.shape:
                raise ValueError("Compared cache tensors have different shapes")
            delta = a.float() - b.float()
            changed += int(torch.count_nonzero(delta).item())
            maximum = max(maximum, float(delta.abs().max().item()))
    return {
        "changed_elements": int(changed),
        "maximum_absolute_delta": float(maximum),
        "exactly_equal": changed == 0,
    }


def state_axis_metrics(
    state: torch.Tensor, receiver: torch.Tensor, donor: torch.Tensor
) -> dict[str, float | None]:
    """Measure movement along and outside the clean donor-receiver contrast."""

    active = state.detach().float().reshape(-1)
    base = receiver.detach().float().reshape(-1)
    target = donor.detach().float().reshape(-1)
    axis = target - base
    displacement = active - base
    denominator = float(torch.dot(axis, axis).item())
    if denominator <= 1e-12:
        progress = None
        off_axis = None
    else:
        progress = float(torch.dot(displacement, axis).item() / denominator)
        residual = displacement - progress * axis
        off_axis = float(
            torch.linalg.vector_norm(residual).item()
            / max(torch.linalg.vector_norm(axis).item(), 1e-12)
        )
    return {
        "donor_axis_progress": progress,
        "off_axis_norm_over_clean_contrast": off_axis,
        "l2_to_receiver": float(torch.linalg.vector_norm(active - base).item()),
        "l2_to_donor": float(torch.linalg.vector_norm(active - target).item()),
        "clean_contrast_l2": float(torch.linalg.vector_norm(axis).item()),
    }


def state_equivalence(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    a = left.detach().float().reshape(-1)
    b = right.detach().float().reshape(-1)
    delta = a - b
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return {
        "maximum_absolute_delta": float(delta.abs().max().item()),
        "mean_absolute_delta": float(delta.abs().mean().item()),
        "cosine_similarity": float(torch.dot(a, b).item() / denominator),
    }


def _compact_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in audit.items()
        if key != "per_layer"
    }


def append_condition_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    request_id: str,
    condition: str,
    region: str | None,
    components: Sequence[str],
    donor_variant: str | None,
    receiver_variant: str,
    captured: Mapping[int, torch.Tensor],
    clean_states: Mapping[str, Mapping[int, torch.Tensor]],
    probes: Mapping[int, Mapping[str, Any]],
    splice_audit: Mapping[str, Any] | None,
) -> None:
    for layer, state in sorted(captured.items()):
        decoded = decode_count_probe(probes[int(layer)], state.numpy())
        scores = tuple(float(value) for value in decoded["probe_scores"])
        receiver_state = clean_states[receiver_variant][int(layer)]
        donor_state = (
            clean_states[donor_variant][int(layer)]
            if donor_variant is not None
            else receiver_state
        )
        receiver_decoded = decode_count_probe(
            probes[int(layer)], receiver_state.numpy()
        )
        donor_decoded = decode_count_probe(probes[int(layer)], donor_state.numpy())
        receiver_scores = tuple(
            float(value) for value in receiver_decoded["probe_scores"]
        )
        donor_scores = tuple(float(value) for value in donor_decoded["probe_scores"])
        axis = state_axis_metrics(state, receiver_state, donor_state)
        receiver_margin = receiver_scores[6] - receiver_scores[5]
        donor_margin = donor_scores[6] - donor_scores[5]
        active_margin = scores[6] - scores[5]
        margin_denominator = donor_margin - receiver_margin
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "seed": int(seed),
                "request_id": str(request_id),
                "condition": str(condition),
                "region": region,
                "components": list(components),
                "donor_variant": donor_variant,
                "receiver_variant": str(receiver_variant),
                "read_layer": int(layer),
                "probe_prediction": int(decoded["probe_prediction"]),
                "probe_softmax_expected_count": float(
                    decoded["probe_softmax_expected_count"]
                ),
                "probe_margin_7_minus_6": float(active_margin),
                "receiver_clean_prediction": int(
                    receiver_decoded["probe_prediction"]
                ),
                "donor_clean_prediction": int(donor_decoded["probe_prediction"]),
                "receiver_clean_margin_7_minus_6": float(receiver_margin),
                "donor_clean_margin_7_minus_6": float(donor_margin),
                "probe_margin_progress": (
                    float((active_margin - receiver_margin) / margin_denominator)
                    if abs(margin_denominator) > 1e-12
                    else None
                ),
                "donor_prediction_match": int(decoded["probe_prediction"])
                == int(donor_decoded["probe_prediction"]),
                "receiver_prediction_match": int(decoded["probe_prediction"])
                == int(receiver_decoded["probe_prediction"]),
                **axis,
                "splice_audit": (
                    _compact_audit(splice_audit)
                    if splice_audit is not None
                    else None
                ),
                "tokens_changed": False,
                "attention_mask_changed": False,
                "positions_changed": False,
            }
        )


def summarize_trials(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["condition"]) != "cache_splice":
            continue
        grouped[
            (
                str(row["donor_variant"]),
                str(row["receiver_variant"]),
                str(row["region"]),
                "+".join(str(value) for value in row["components"]),
                int(row["read_layer"]),
            )
        ].append(row)
    cells = []
    for key, active in sorted(grouped.items()):
        donor, receiver, region, components, layer = key
        progress = [
            float(row["donor_axis_progress"])
            for row in active
            if row["donor_axis_progress"] is not None
        ]
        margin_progress = [
            float(row["probe_margin_progress"])
            for row in active
            if row["probe_margin_progress"] is not None
        ]
        changed = [int(row["splice_audit"]["changed_elements"]) for row in active]
        cells.append(
            {
                "donor_variant": donor,
                "receiver_variant": receiver,
                "direction": f"{donor}_to_{receiver}",
                "region": region,
                "components": components,
                "read_layer": layer,
                "n_seeds": len(active),
                "mean_donor_axis_progress": fmean(progress),
                "median_donor_axis_progress": float(np.median(progress)),
                "positive_axis_progress_rate": float(np.mean(np.asarray(progress) > 0)),
                "axis_progress_sign_flip_pvalue": sign_flip_pvalue(progress),
                "mean_probe_margin_progress": (
                    fmean(margin_progress) if margin_progress else None
                ),
                "donor_prediction_match_rate": float(
                    np.mean([bool(row["donor_prediction_match"]) for row in active])
                ),
                "receiver_prediction_match_rate": float(
                    np.mean([bool(row["receiver_prediction_match"]) for row in active])
                ),
                "prediction_counts": {
                    str(label): int(count)
                    for label, count in sorted(
                        Counter(int(row["probe_prediction"]) for row in active).items()
                    )
                },
                "mean_changed_cache_elements": fmean(changed),
                "all_splices_exact_identity": all(value == 0 for value in changed),
            }
        )

    primary = [
        cell
        for cell in cells
        if cell["region"] == "closing"
        and cell["components"] == "key+value"
        and int(cell["read_layer"]) == PRIMARY_READ_LAYER
    ]
    positive_control = [
        cell
        for cell in cells
        if cell["region"] == "event"
        and cell["components"] == "key+value"
        and int(cell["read_layer"]) == PRIMARY_READ_LAYER
    ]
    identity_controls = [
        cell
        for cell in cells
        if cell["region"] in {"b5", "preceding_event_width"}
        and cell["components"] == "key+value"
        and int(cell["read_layer"]) == PRIMARY_READ_LAYER
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "cells": cells,
        "primary_estimand": {
            "description": (
                "bidirectional mean donor-axis progress at L24 after swapping "
                "only all-layer closing-token K/V"
            ),
            "cells": primary,
            "bidirectional_mean_donor_axis_progress": (
                fmean(float(cell["mean_donor_axis_progress"]) for cell in primary)
                if primary
                else None
            ),
        },
        "full_event_positive_control": positive_control,
        "pre_event_identity_controls": identity_controls,
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
    parser.add_argument(
        "--dense-equivalence-seeds",
        type=int,
        nargs="*",
        default=[],
        help="Optional seeds on which cached clean reads are compared with dense forward.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "event-cache-splice"

    seeds = tuple(dict.fromkeys(int(value) for value in args.evaluation_seeds))
    receiver = int(args.receiver)
    read_layers = tuple(sorted({int(value) for value in args.read_layers}))
    dense_seeds = {int(value) for value in args.dense_equivalence_seeds}
    if not seeds or not 2 <= receiver <= 8:
        raise ValueError("Event-cache seed/receiver geometry is invalid")
    if not read_layers or PRIMARY_READ_LAYER not in read_layers:
        raise ValueError("The preregistered L24 primary read must be present")
    if not dense_seeds.issubset(set(seeds)):
        raise ValueError("Dense-equivalence seeds must be evaluation seeds")

    probes = _load_probes(args.frozen_probes, read_layers)
    source_rows = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    if max(read_layers) >= int(adapter.num_layers):
        raise ValueError("A read layer is outside the decoder")
    trials: list[dict[str, Any]] = []
    geometry_audits: list[dict[str, Any]] = []
    equivalence_audits: list[dict[str, Any]] = []
    positive_control_audits: list[dict[str, Any]] = []

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
                receiver=receiver,
                current_boundary=boundaries[receiver],
                target_boundary=boundaries[receiver + 1],
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
            receiver=receiver,
            insert_source_occurrence=int(args.insert_source_occurrence),
        )
        if not bool(geometry["closing_surface_token_identical"]):
            raise RuntimeError("Closing-token surface identity is required")
        geometry_audits.append(
            {
                "seed": int(seed),
                **{
                    key: value
                    for key, value in geometry.items()
                    if key != "regions"
                },
                "regions": {
                    key: list(value) for key, value in geometry["regions"].items()
                },
            }
        )

        insertion_start = int(geometry["insertion_start"])
        event_end = int(geometry["event_end"])
        target_position = int(geometry["target_position"])
        common = prefill_common_prefix(
            model, valid["encoding"], end=insertion_start
        )
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
            append_condition_rows(
                trials,
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

        if seed in dense_seeds:
            for label in (VALID_VARIANT, PRIMARY_INVALID_VARIANT):
                dense = capture_decoder_block_input_states(
                    model,
                    adapter,
                    encodings[label],
                    (target_position,),
                    layers=read_layers,
                )
                for layer in read_layers:
                    dense_state = dense[layer][0]
                    cached_state = clean_states[label][layer]
                    dense_decoded = decode_count_probe(
                        probes[layer], dense_state.numpy()
                    )
                    cached_decoded = decode_count_probe(
                        probes[layer], cached_state.numpy()
                    )
                    equivalence_audits.append(
                        {
                            "seed": int(seed),
                            "event_variant": label,
                            "read_layer": int(layer),
                            **state_equivalence(dense_state, cached_state),
                            "dense_probe_prediction": int(
                                dense_decoded["probe_prediction"]
                            ),
                            "cached_probe_prediction": int(
                                cached_decoded["probe_prediction"]
                            ),
                            "probe_prediction_match": int(
                                dense_decoded["probe_prediction"]
                            )
                            == int(cached_decoded["probe_prediction"]),
                        }
                    )

        for donor_variant, receiver_variant in (
            (VALID_VARIANT, PRIMARY_INVALID_VARIANT),
            (PRIMARY_INVALID_VARIANT, VALID_VARIANT),
        ):
            for region, components in REGION_COMPONENTS:
                hybrid, audit = splice_cache_positions(
                    caches[receiver_variant],
                    caches[donor_variant],
                    positions=geometry["regions"][region],
                    components=components,
                )
                if region in {"b5", "preceding_event_width"} and not bool(
                    audit["exact_identity_splice"]
                ):
                    raise RuntimeError(
                        f"Pre-event identity control changed cache fields: {region}"
                    )
                if region == "event" and tuple(components) == ("key", "value"):
                    comparison = cache_difference(hybrid, caches[donor_variant])
                    positive_control_audits.append(
                        {
                            "seed": int(seed),
                            "donor_variant": donor_variant,
                            "receiver_variant": receiver_variant,
                            **comparison,
                        }
                    )
                    if not bool(comparison["exactly_equal"]):
                        raise RuntimeError(
                            "Full-event K/V swap failed to reconstruct donor cache"
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
                append_condition_rows(
                    trials,
                    seed=seed,
                    request_id=str(row["request_id"]),
                    condition="cache_splice",
                    region=region,
                    components=components,
                    donor_variant=donor_variant,
                    receiver_variant=receiver_variant,
                    captured=captured,
                    clean_states=clean_states,
                    probes=probes,
                    splice_audit=audit,
                )
                del hybrid
        del caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[event-cache-splice] seed={seed} complete", flush=True)

    summary = {
        **summarize_trials(trials),
        "model_label": str(args.model),
        "evaluation_seeds": list(seeds),
        "receiver": receiver,
        "read_layers": list(read_layers),
        "primary_invalid_variant": PRIMARY_INVALID_VARIANT,
        "region_components": [
            {"region": region, "components": list(components)}
            for region, components in REGION_COMPONENTS
        ],
        "full_event_kv_is_positive_control_not_mechanistic_evidence": True,
        "geometry_audits": geometry_audits,
        "dense_cache_equivalence": equivalence_audits,
        "dense_cache_equivalence_all_probe_predictions_match": all(
            bool(row["probe_prediction_match"]) for row in equivalence_audits
        ),
        "full_event_positive_control_cache_audits": positive_control_audits,
        "full_event_positive_control_all_exact": all(
            bool(row["exactly_equal"]) for row in positive_control_audits
        ),
        "trial_count": len(trials),
    }
    _atomic_jsonl(args.output, trials)
    _atomic_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
