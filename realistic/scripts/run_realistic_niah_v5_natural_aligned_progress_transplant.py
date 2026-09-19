#!/usr/bin/env python3
"""Transplant a natural step-k state into a position-aligned natural step-j."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.counting_mechanism_transfer import (  # noqa: E402
    build_first_pass_tstar_answer_source_registry,
)
from realistic_niah_v5.count_stream import (  # noqa: E402
    build_answer_source_registry,
)
from realistic_niah_v5.natural_aligned_progress import (  # noqa: E402
    PATCH_SCOPES,
    SITE_POLICIES,
    align_natural_donor_prompt,
    matched_post_item_sites,
    post_item_sites_at_tail_offset,
    resolve_natural_patch_span,
)
from realistic_niah_v5.native_loop import load_frozen_targeted_bank  # noqa: E402
from realistic_niah_v5.parsing import gold_records  # noqa: E402
from realistic_niah_v4_4_5.restoration import (  # noqa: E402
    generate_answer_completion_from_prefill,
)
from scripts.run_realistic_niah_v5_cross_seed_counter_recurrence import (  # noqa: E402
    prefix_through_boundary,
)
from scripts.run_realistic_niah_v5_same_site_progress_transplant import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _chunk_forward,
    _chunked_prefill,
    _encoding_tensors,
    _experiment_model,
    _read_rows,
    _replace_output_tensor,
    _score_candidates,
    _tensor_from_output,
    _targeted_attention_metrics,
)
from realistic_niah_v5.same_site_progress_transplant import (  # noqa: E402
    generated_bullet_city_ordinals,
)


SCHEMA_VERSION = "natural_aligned_progress_transplant_v2"
CONDITIONS = ("receiver_self", "native_donor", "donor_to_receiver")
COHORT_MODES = (
    "natural_noindex",
    "prompt_conditioned_noindex",
    "indexed_positive_control",
)


@torch.inference_mode()
def _chunked_capture_span_states(
    model: Any,
    adapter: Any,
    encoding: Any,
    *,
    site: int,
    width: int,
    layers: tuple[int, ...],
    chunk_size: int,
) -> dict[int, torch.Tensor]:
    """Capture post-block states for a contiguous span ending at ``site``."""

    active_width = int(width)
    start_site = int(site) - active_width + 1
    if active_width <= 0 or start_site < 0:
        raise ValueError("Patch span must be a positive in-bounds width")
    if int(site) != int(encoding.sequence_length) - 1:
        raise ValueError("The capture encoding must end at the shared site")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    previous = None
    for start in range(0, start_site, int(chunk_size)):
        previous = _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=start,
            end=min(start_site, start + int(chunk_size)),
            previous=previous,
        )
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or int(hidden.shape[1]) != active_width:
                raise RuntimeError("Shared-span capture observed an unexpected chunk")
            captured[layer] = hidden[0].detach().float().cpu()

        return hook

    for layer in layers:
        handles.append(adapter.layers[int(layer)].register_forward_hook(make_hook(int(layer))))
    try:
        _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=start_site,
            end=int(site) + 1,
            previous=previous,
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = set(int(value) for value in layers) - set(captured)
    if missing:
        raise RuntimeError(f"Failed to capture shared-span layers: {sorted(missing)}")
    return captured


@torch.inference_mode()
def _chunked_prefill_with_span_replacement(
    model: Any,
    adapter: Any,
    encoding: Any,
    *,
    layer: int,
    site: int,
    states: torch.Tensor,
    chunk_size: int,
) -> tuple[Any, int, float]:
    """Patch one contiguous post-block span and finish a causal prefill."""

    fixed = torch.as_tensor(states).detach().float().cpu()
    if fixed.ndim != 2 or int(fixed.shape[0]) <= 0:
        raise ValueError("Span replacement states must have shape [width, hidden]")
    width = int(fixed.shape[0])
    start_site = int(site) - width + 1
    if start_site < 0 or int(site) >= int(encoding.sequence_length):
        raise ValueError("The shared patch span lies outside the encoding")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    previous = None
    for start in range(0, start_site, int(chunk_size)):
        previous = _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=start,
            end=min(start_site, start + int(chunk_size)),
            previous=previous,
        )
    applications = 0
    realized_norm = 0.0

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        nonlocal applications, realized_norm
        hidden = _tensor_from_output(output)
        if hidden.ndim != 3 or int(hidden.shape[1]) != width:
            raise RuntimeError("Shared-span patch observed an unexpected chunk")
        replacement = fixed.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
        patched = hidden.clone()
        realized_norm = float(
            torch.linalg.vector_norm(hidden.float() - replacement.float()).detach().cpu()
        )
        patched[:, :, :] = replacement
        applications += 1
        return _replace_output_tensor(output, patched)

    handle = adapter.layers[int(layer)].register_forward_hook(hook)
    try:
        previous = _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=start_site,
            end=int(site) + 1,
            previous=previous,
        )
    finally:
        handle.remove()
    if applications != 1:
        raise RuntimeError(f"Shared-span patch applied {applications} times")
    for start in range(int(site) + 1, int(encoding.sequence_length), int(chunk_size)):
        previous = _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=start,
            end=min(int(encoding.sequence_length), start + int(chunk_size)),
            previous=previous,
        )
    assert previous is not None
    return previous, applications, realized_norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument(
        "--cohort-mode", choices=COHORT_MODES, default="natural_noindex"
    )
    parser.add_argument("--gold-count", type=int, required=True)
    parser.add_argument("--receiver-occurrence", type=int, required=True)
    parser.add_argument("--donor-occurrence", type=int, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--generation-conditions", nargs="+", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--run-attention", action="store_true")
    parser.add_argument("--targeted-selection", type=Path)
    parser.add_argument("--targeted-routing", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--tail-window", type=int, default=4)
    parser.add_argument("--site-policy", choices=SITE_POLICIES, default="latest_structural")
    parser.add_argument("--tail-offset", type=int)
    parser.add_argument("--patch-scope", choices=PATCH_SCOPES, default="fixed_suffix")
    parser.add_argument("--patch-width", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conditions = tuple(str(value) for value in args.conditions)
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    generation_conditions = tuple(str(value) for value in args.generation_conditions)
    if not set(generation_conditions) <= set(conditions):
        raise ValueError("Generation conditions must be evaluated conditions")
    layers = tuple(sorted({int(value) for value in args.layers}))
    requested_patch_width = int(args.patch_width)
    if requested_patch_width <= 0:
        raise ValueError("patch-width must be positive")
    indexed_positive_control = str(args.cohort_mode) == "indexed_positive_control"
    prompt_conditioned_noindex = (
        str(args.cohort_mode) == "prompt_conditioned_noindex"
    )
    if indexed_positive_control:
        selection_population = "indexed_positive_control"
        eligibility_field = "primary_eligible_indexed_positive_control"
    elif prompt_conditioned_noindex:
        selection_population = "gemma_prompt_conditioned_noindex_found_v3"
        eligibility_field = "primary_eligible_prompt_conditioned_noindex"
    else:
        selection_population = "first_pass_noindex_enumeration"
        eligibility_field = "primary_eligible_prefix_clean"
    rows = _read_rows(
        args.generations,
        gold_count=int(args.gold_count),
        seeds=args.seeds,
        max_seeds=args.max_seeds,
        selection_population=selection_population,
        eligibility_field=eligibility_field,
    )
    model, tokenizer, adapter = _experiment_model(args)
    targeted_bank = None
    if args.run_attention:
        if args.targeted_selection is None or args.targeted_routing is None:
            raise ValueError("Attention readout requires frozen configs")
        targeted_bank = load_frozen_targeted_bank(
            args.targeted_selection,
            args.targeted_routing,
            model_label=str(args.model),
        )

    trials: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    receiver_successor = int(args.receiver_occurrence) + 1
    donor_successor = int(args.donor_occurrence) + 1
    for row_index, row in enumerate(rows, start=1):
        if indexed_positive_control:
            encoding, registry = build_answer_source_registry(
                row,
                tokenizer,
                candidate_counts=tuple(range(1, int(args.gold_count) + 1)),
            )
        else:
            encoding, registry = build_first_pass_tstar_answer_source_registry(
                row,
                tokenizer,
                candidate_counts=tuple(range(1, int(args.gold_count) + 1)),
                selection_population=selection_population,
                eligibility_field=eligibility_field,
            )
        if args.tail_offset is None:
            receiver_site, donor_site, site_audit = matched_post_item_sites(
                encoding,
                registry.trace_items,
                receiver_occurrence=int(args.receiver_occurrence),
                donor_occurrence=int(args.donor_occurrence),
                tokenizer=tokenizer,
                tail_window=int(args.tail_window),
                site_policy=str(args.site_policy),
            )
        else:
            receiver_site, donor_site, site_audit = post_item_sites_at_tail_offset(
                encoding,
                registry.trace_items,
                receiver_occurrence=int(args.receiver_occurrence),
                donor_occurrence=int(args.donor_occurrence),
                tokenizer=tokenizer,
                tail_offset=int(args.tail_offset),
            )
        if donor_site > receiver_site:
            receiver_encoding = encoding
            aligned_donor, alignment_audit = align_natural_donor_prompt(
                encoding,
                registry,
                receiver_site=receiver_site,
                donor_site=donor_site,
                tokenizer=tokenizer,
                require_surface_match=bool(site_audit["surface_token_matched"]),
            )
            donor_encoding = aligned_donor
            shared_site = receiver_site
            aligned_prompt_role = "donor"
        else:
            aligned_receiver, alignment_audit = align_natural_donor_prompt(
                encoding,
                registry,
                receiver_site=donor_site,
                donor_site=receiver_site,
                tokenizer=tokenizer,
                require_surface_match=bool(site_audit["surface_token_matched"]),
            )
            receiver_encoding = aligned_receiver
            donor_encoding = encoding
            shared_site = donor_site
            aligned_prompt_role = "receiver"
        receiver_prefix = prefix_through_boundary(receiver_encoding, shared_site)
        donor_prefix = prefix_through_boundary(donor_encoding, shared_site)
        if receiver_prefix.sequence_length != donor_prefix.sequence_length:
            raise RuntimeError("Aligned natural prefixes have different lengths")
        if bool(site_audit["surface_token_matched"]) and (
            int(receiver_prefix.input_ids[-1]) != int(donor_prefix.input_ids[-1])
        ):
            raise RuntimeError("Aligned natural prefixes have different commit tokens")
        receiver_transition = tuple(
            int(value)
            for value in encoding.input_ids[
                receiver_site + 1 : int(registry.trace_items[receiver_successor - 1][1])
            ]
        )
        donor_transition = tuple(
            int(value)
            for value in encoding.input_ids[
                donor_site + 1 : int(registry.trace_items[donor_successor - 1][1])
            ]
        )
        if not receiver_transition or not donor_transition:
            raise RuntimeError("A natural successor transition is empty")
        # Binary labels are local assay labels: 1=the receiver's complete
        # natural transition to N[j+1], 2=the donor's complete natural
        # transition to N[k+1].
        candidates = {1: receiver_transition, 2: donor_transition}
        cities = tuple(str(value["city"]) for value in gold_records(row))
        receiver_item_start = int(
            registry.trace_items[int(args.receiver_occurrence) - 1][0]
        )
        donor_item_start = int(registry.trace_items[int(args.donor_occurrence) - 1][0])
        patch_geometry = resolve_natural_patch_span(
            registry.trace_items,
            receiver_occurrence=int(args.receiver_occurrence),
            donor_occurrence=int(args.donor_occurrence),
            receiver_site=receiver_site,
            donor_site=donor_site,
            patch_scope=str(args.patch_scope),
            patch_width=requested_patch_width,
        )
        patch_width = int(patch_geometry["effective_patch_width"])
        receiver_span_token_ids = tuple(
            int(value)
            for value in encoding.input_ids[
                receiver_site - patch_width + 1 : receiver_site + 1
            ]
        )
        donor_span_token_ids = tuple(
            int(value)
            for value in encoding.input_ids[donor_site - patch_width + 1 : donor_site + 1]
        )
        geometry_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": str(row["request_id"]),
                "seed": int(row["seed"]),
                "receiver_occurrence_j": int(args.receiver_occurrence),
                "donor_occurrence_k": int(args.donor_occurrence),
                "aligned_absolute_site": shared_site,
                "aligned_prompt_role": aligned_prompt_role,
                **patch_geometry,
                "patch_width": patch_width,
                "receiver_span_token_ids": list(receiver_span_token_ids),
                "donor_span_token_ids": list(donor_span_token_ids),
                "patch_span_surface_matched": receiver_span_token_ids
                == donor_span_token_ids,
                "receiver_span_text": tokenizer.decode(
                    list(receiver_span_token_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "donor_span_text": tokenizer.decode(
                    list(donor_span_token_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                **site_audit,
                **alignment_audit,
            }
        )

        receiver_states = _chunked_capture_span_states(
            model,
            adapter,
            receiver_prefix,
            site=shared_site,
            width=patch_width,
            layers=layers,
            chunk_size=int(args.prefill_chunk_size),
        )
        donor_states = _chunked_capture_span_states(
            model,
            adapter,
            donor_prefix,
            site=shared_site,
            width=patch_width,
            layers=layers,
            chunk_size=int(args.prefill_chunk_size),
        )
        for layer in layers:
            specifications = {
                "receiver_self": (receiver_prefix, receiver_states[layer]),
                "native_donor": (donor_prefix, donor_states[layer]),
                "donor_to_receiver": (receiver_prefix, donor_states[layer]),
            }
            for condition in conditions:
                prefix, state = specifications[condition]
                prefill, applications, realized_norm = (
                    _chunked_prefill_with_span_replacement(
                        model,
                        adapter,
                        prefix,
                        layer=layer,
                        site=shared_site,
                        states=state,
                        chunk_size=int(args.prefill_chunk_size),
                    )
                )
                generation_outcome: dict[str, Any] = {}
                if condition in generation_conditions:
                    generation_prefill, _generation_apps, _generation_norm = (
                        _chunked_prefill_with_span_replacement(
                            model,
                            adapter,
                            prefix,
                            layer=layer,
                            site=shared_site,
                            states=state,
                            chunk_size=int(args.prefill_chunk_size),
                        )
                    )
                    completion = generate_answer_completion_from_prefill(
                        model,
                        tokenizer,
                        prefix,
                        generation_prefill,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    generated = generated_bullet_city_ordinals(
                        str(completion["completion_text"]), cities
                    )
                    first = generated["first_generated_known_city_ordinal"]
                    close_char = generated["reasoning_close_char_position"]
                    close_token_count = (
                        None
                        if close_char is None
                        else len(
                            tokenizer.encode(
                                str(completion["completion_text"])[: int(close_char)],
                                add_special_tokens=False,
                            )
                        )
                    )
                    generation_outcome = {
                        "completion_text": str(completion["completion_text"]),
                        "generated_token_count": int(completion["generated_token_count"]),
                        "generation_truncated": bool(completion["generation_truncated"]),
                        "stopped_on_eos": bool(completion["stopped_on_eos"]),
                        "reasoning_close_token_count": close_token_count,
                        "greedy_donor_successor_adoption": bool(
                            first == donor_successor
                        ),
                        "greedy_receiver_successor_retention": bool(
                            first == receiver_successor
                        ),
                        **generated,
                    }
                scored = _score_candidates(model, prefix, prefill, candidates)
                outcome: dict[str, Any] = {
                    **scored,
                    "donor_vs_receiver_sum_logodds": float(
                        scored["sum_logprob_scores"][1]
                        - scored["sum_logprob_scores"][0]
                    ),
                    "receiver_successor_argmax": bool(
                        int(scored["predicted_occurrence_sum_logprob"]) == 1
                    ),
                    "donor_successor_argmax": bool(
                        int(scored["predicted_occurrence_sum_logprob"]) == 2
                    ),
                    "receiver_transition_token_count": len(receiver_transition),
                    "donor_transition_token_count": len(donor_transition),
                }
                if args.run_attention:
                    prequery = prefix_through_boundary(prefix, shared_site - 1)
                    if patch_width == 1:
                        prefix_output = _chunked_prefill(
                            model,
                            adapter,
                            prequery,
                            chunk_size=int(args.prefill_chunk_size),
                        )
                    else:
                        prefix_output, _prior_apps, _prior_norm = (
                            _chunked_prefill_with_span_replacement(
                                model,
                                adapter,
                                prequery,
                                layer=layer,
                                site=shared_site - 1,
                                states=state[:-1],
                                chunk_size=int(args.prefill_chunk_size),
                            )
                        )
                    outcome.update(
                        _targeted_attention_metrics(
                            model,
                            adapter,
                            prefix,
                            prefix_output,
                            query_position=shared_site,
                            patch_layer=layer,
                            replacement_state=state[-1],
                            targeted_bank=targeted_bank,
                            ordered_cities=cities,
                            receiver_successor=receiver_successor,
                            donor_successor=donor_successor,
                        )
                    )
                trials.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "request_id": str(row["request_id"]),
                        "seed": int(row["seed"]),
                        "gold_count": int(args.gold_count),
                        "cohort_mode": str(args.cohort_mode),
                        "grammar_class": str(
                            row.get("indexed_progress_control_format_audit", {}).get(
                                "grammar_class",
                                row.get(
                                    "gemma_prompt_conditioned_noindex_format_audit",
                                    {},
                                ).get("grammar_class", "natural_noindex"),
                            )
                        ),
                        "receiver_occurrence_j": int(args.receiver_occurrence),
                        "donor_occurrence_k": int(args.donor_occurrence),
                        "receiver_successor": receiver_successor,
                        "donor_successor": donor_successor,
                        "layer": layer,
                        "condition": condition,
                        "shared_commit_position": shared_site,
                        "shared_commit_token_id": int(prefix.input_ids[-1]),
                        "patch_width": patch_width,
                        "patch_scope": str(args.patch_scope),
                        "requested_patch_width": requested_patch_width,
                        "equal_length_complete_item": bool(
                            patch_geometry["equal_length_complete_item"]
                        ),
                        "receiver_item_coverage": float(
                            patch_geometry["receiver_item_coverage"]
                        ),
                        "donor_item_coverage": float(
                            patch_geometry["donor_item_coverage"]
                        ),
                        "patch_applications": int(applications),
                        "realized_patch_delta_norm": float(realized_norm),
                        **generation_outcome,
                        **outcome,
                    }
                )
        print(
            f"[natural-aligned] {row_index}/{len(rows)} seed={row['seed']} "
            f"j={args.receiver_occurrence} k={args.donor_occurrence}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output / "trials.jsonl", trials)
    _atomic_jsonl(args.output / "geometry_audit.jsonl", geometry_rows)
    summary: list[dict[str, Any]] = []
    for layer in layers:
        for condition in conditions:
            group = [
                row
                for row in trials
                if int(row["layer"]) == layer and row["condition"] == condition
            ]
            summary.append(
                {
                    "layer": layer,
                    "condition": condition,
                    "seed_count": len(group),
                    "patch_scope": str(args.patch_scope),
                    "mean_effective_patch_width": float(
                        np.mean([row["patch_width"] for row in group])
                    ),
                    "equal_length_complete_item_rate": float(
                        np.mean(
                            [row["equal_length_complete_item"] for row in group]
                        )
                    ),
                    "mean_donor_vs_receiver_sum_logodds": float(
                        np.mean([row["donor_vs_receiver_sum_logodds"] for row in group])
                    ),
                    "receiver_successor_argmax_rate": float(
                        np.mean([row["receiver_successor_argmax"] for row in group])
                    ),
                    "donor_successor_argmax_rate": float(
                        np.mean([row["donor_successor_argmax"] for row in group])
                    ),
                    "mean_attention_log_ratio": (
                        None
                        if not args.run_attention
                        else float(
                            np.mean(
                                [row["donor_vs_receiver_attention_log_ratio"] for row in group]
                            )
                        )
                    ),
                    "greedy_donor_adoption_rate": (
                        None
                        if condition not in generation_conditions
                        else float(
                            np.mean(
                                [row["greedy_donor_successor_adoption"] for row in group]
                            )
                        )
                    ),
                    "mean_generated_token_count": (
                        None
                        if condition not in generation_conditions
                        else float(np.mean([row["generated_token_count"] for row in group]))
                    ),
                    "mean_generated_known_city_count": (
                        None
                        if condition not in generation_conditions
                        else float(
                            np.mean(
                                [row["generated_known_city_count_any_surface"] for row in group]
                            )
                        )
                    ),
                }
            )
    _atomic_jsonl(args.output / "summary.jsonl", summary)
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "model": str(args.model),
            "cohort_mode": str(args.cohort_mode),
            "gold_count": int(args.gold_count),
            "receiver_occurrence_j": int(args.receiver_occurrence),
            "donor_occurrence_k": int(args.donor_occurrence),
            "layers": list(layers),
            "conditions": list(conditions),
            "generation_conditions": list(generation_conditions),
            "tail_window": int(args.tail_window),
            "site_policy": str(args.site_policy),
            "tail_offset": args.tail_offset,
            "patch_scope": str(args.patch_scope),
            "requested_patch_width": requested_patch_width,
            "effective_patch_widths": sorted(
                {int(row["effective_patch_width"]) for row in geometry_rows}
            ),
            "equal_length_complete_item_rate": float(
                np.mean(
                    [row["equal_length_complete_item"] for row in geometry_rows]
                )
            ),
            "seed_count": len(rows),
            "run_attention": bool(args.run_attention),
            "surface_token_matched": all(
                bool(row["surface_token_matched"]) for row in geometry_rows
            ),
            "absolute_position_matched": True,
            "donor_is_natural_noindex_step": not (
                indexed_positive_control or prompt_conditioned_noindex
            ),
            "prompt_conditioned_noindex": prompt_conditioned_noindex,
            "prompt_modified": prompt_conditioned_noindex,
            "indexed_positive_control": indexed_positive_control,
            "visible_progress_confound_allowed": indexed_positive_control,
            "spontaneous_noindex_internal_counter_claim_allowed": not (
                indexed_positive_control or prompt_conditioned_noindex
            ),
            "prompt_conditioned_auxiliary_claim_allowed": prompt_conditioned_noindex,
            "alignment_deletes_only_nonrecord_prompt_filler": True,
        },
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
