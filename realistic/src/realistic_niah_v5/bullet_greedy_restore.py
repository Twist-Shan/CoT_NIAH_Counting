"""Greedy-integer readout for marker-scrubbed list-state restoration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import DecoderAdapter, _encoding_tensors
from realistic_niah_v4_4_5.restoration import generate_answer_completion_from_prefill

from .bullet_counterfactual_restore import (
    audit_complete_marker_scrubbable_list,
    build_marker_scrubbed_list_registry,
    build_scrubbed_source_and_blank,
)
from .causal import completion_metrics
from .count_stream import _prefix_forward, _sha256_json
from .indexed_counter_patch import (
    build_minimal_item_early_stop_encoding,
    capture_decoder_block_input_states,
    minimal_terminal_suffix_token_ids,
    prefill_with_single_decoder_block_input_replacement,
)
from .encoding import NativeTraceEncoding


def _greedy_integer_outcomes(
    model: Any,
    tokenizer: Any,
    encoding: NativeTraceEncoding,
    prefill: Any,
    *,
    target_k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    completion = generate_answer_completion_from_prefill(
        model,
        tokenizer,
        encoding,
        prefill,
        max_new_tokens=int(max_new_tokens),
    )
    metrics = completion_metrics(completion, gold_count=int(target_k))
    prediction = metrics["prediction"]
    return {
        **completion,
        "greedy_prediction": int(prediction) if prediction is not None else None,
        "greedy_running_exact": bool(metrics["exact_count"]),
        "greedy_running_signed_error": metrics["signed_error"],
        "greedy_running_absolute_error": metrics["absolute_error"],
        "greedy_output_is_integer_1_to_10": bool(
            prediction is not None and 1 <= int(prediction) <= 10
        ),
        "greedy_readout_target_k": int(target_k),
        "greedy_readout_mode": "free_full_vocabulary_argmax_decode_then_parse_integer",
    }


@torch.inference_mode()
def _score_greedy_encoding(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    target_k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    return _greedy_integer_outcomes(
        model,
        tokenizer,
        encoding,
        prefill,
        target_k=int(target_k),
        max_new_tokens=int(max_new_tokens),
    )


@torch.inference_mode()
def run_bullet_greedy_restore_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layers: Sequence[int],
    target_occurrences: Sequence[int] = tuple(range(1, 11)),
    random_seed: int,
    max_new_tokens: int = 2,
    answer_site_id: str = "answer_query_v3",
) -> list[dict[str, Any]]:
    """Run Source, Blank, and item-k restoration with free greedy decoding."""

    trace_audit = audit_complete_marker_scrubbable_list(row)
    if not trace_audit["eligible"]:
        raise ValueError(f"Marker-scrubbable list audit failed: {trace_audit['reasons']}")
    clean_full, registry, registry_audit = build_marker_scrubbed_list_registry(
        row,
        tokenizer,
        answer_site_id=answer_site_id,
        trace_audit=trace_audit,
    )
    source_full, blank_full, scrub_audit = build_scrubbed_source_and_blank(
        clean_full,
        registry,
        tokenizer,
        random_seed=int(random_seed),
    )
    suffix_ids, suffix_audit = minimal_terminal_suffix_token_ids(row, tokenizer)
    layers = tuple(sorted({int(value) for value in source_layers}))
    targets = tuple(sorted({int(value) for value in target_occurrences}))
    if not layers or layers[0] < 0 or layers[-1] >= int(adapter.num_layers):
        raise ValueError("Greedy restoration layer registry is invalid")
    if not targets or min(targets) < 1 or max(targets) > 10:
        raise ValueError("Greedy restoration targets must lie in 1..10")
    if int(max_new_tokens) < 2:
        raise ValueError("Greedy integer readout needs at least two output tokens")

    architecture = (
        "residual_only_with_unpatched_per_layer_inputs_and_shared_kv"
        if any(
            bool(getattr(attention, "is_kv_shared_layer", False))
            for attention in adapter.attentions
        )
        else "standard_residual_stream"
    )
    results: list[dict[str, Any]] = []
    for occurrence in targets:
        source, source_early_audit = build_minimal_item_early_stop_encoding(
            source_full,
            registry,
            target_occurrence=occurrence,
            terminal_suffix_token_ids=suffix_ids,
        )
        blank, blank_early_audit = build_minimal_item_early_stop_encoding(
            blank_full,
            registry,
            target_occurrence=occurrence,
            terminal_suffix_token_ids=suffix_ids,
        )
        if source_early_audit != blank_early_audit:
            raise RuntimeError("Source/Blank greedy early-stop geometry differs")
        start, end = registry.trace_items[occurrence - 1]
        positions = tuple(range(int(start), int(end)))
        source_capture = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            positions,
            layers=layers,
        )
        source_outcomes = _score_greedy_encoding(
            model,
            tokenizer,
            adapter,
            source,
            target_k=occurrence,
            max_new_tokens=int(max_new_tokens),
        )
        blank_outcomes = _score_greedy_encoding(
            model,
            tokenizer,
            adapter,
            blank,
            target_k=occurrence,
            max_new_tokens=int(max_new_tokens),
        )
        common = {
            "schema_version": "realistic_niah_v5_marker_scrubbed_greedy_restore_v1",
            "experiment_id": "marker_scrubbed_list_greedy_integer_restoration",
            "request_id": str(source.request_id),
            "model_label": str(source.model_label),
            "seed": int(source.seed),
            "dataset_split": str(source.split),
            "source_gold_count": int(source.count),
            "target_occurrence": int(occurrence),
            "answer_site_id": answer_site_id,
            "patch_site": "decoder_block_input_residual_tensor",
            "patch_layer_mode": "single_decoder_block_input_once",
            "upper_layers_recomputed_after_patch": True,
            "source_base_scrubbed_before_state_capture": True,
            "blank_visible_list_items_replaced": int(occurrence),
            "blank_earlier_list_items_remain_blank_during_restoration": True,
            "future_list_items_removed": 10 - int(occurrence),
            "readout_mode": "immediate_minimal_native_Total_free_greedy_integer",
            "diagnostic_suffix_used": False,
            "candidate_scoring_used": False,
            "greedy_max_new_tokens": int(max_new_tokens),
            "selection_uses_final_answer": False,
            "outcome_blind": True,
            "architecture_patch_scope": architecture,
            "gemma_complete_model_state_patch": architecture
            == "standard_residual_stream",
            "registry_sha256": registry.to_dict()["registry_sha256"],
            **trace_audit,
            **registry_audit,
            **scrub_audit,
            **suffix_audit,
            **source_early_audit,
        }
        results.extend(
            [
                {
                    **common,
                    "condition": "source_reference",
                    "source_layer": -1,
                    "patch_token_count": 0,
                    **source_outcomes,
                },
                {
                    **common,
                    "condition": "blank_reference",
                    "source_layer": -1,
                    "patch_token_count": 0,
                    **blank_outcomes,
                },
            ]
        )
        for layer in layers:
            prefill, applications, realized_norm = (
                prefill_with_single_decoder_block_input_replacement(
                    model,
                    adapter,
                    blank,
                    positions=positions,
                    layer=int(layer),
                    replacement_states=source_capture[int(layer)],
                )
            )
            outcomes = _greedy_integer_outcomes(
                model,
                tokenizer,
                blank,
                prefill,
                target_k=occurrence,
                max_new_tokens=int(max_new_tokens),
            )
            results.append(
                {
                    **common,
                    "condition": "source_list_item_k_to_blank_restoration",
                    "source_layer": int(layer),
                    "patch_layers": [int(layer)],
                    "patch_layer_count": 1,
                    "patch_token_count": len(positions),
                    "patch_positions_sha256": _sha256_json(positions),
                    "donor_receiver_positions_identical": True,
                    "patch_hook_applications": {str(layer): int(applications)},
                    "patch_realized_fro_norm_by_layer": {
                        str(layer): float(realized_norm)
                    },
                    **outcomes,
                }
            )
    return results
