"""Small-sample diagnostics for marker-scrubbed list-state restoration.

These interventions are implementation and localization diagnostics.  They do
not replace the frozen single-layer, single-item confirmation estimand.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.modeling import DecoderAdapter, _encoding_tensors

from .bullet_counterfactual_restore import (
    audit_complete_marker_scrubbable_list,
    build_marker_scrubbed_list_registry,
    build_scrubbed_source_and_blank,
    running_target_metrics,
)
from .count_stream import _prefix_forward, _score_and_generate_prefill
from .indexed_counter_patch import (
    build_minimal_item_early_stop_encoding,
    capture_decoder_block_input_states,
    minimal_terminal_suffix_token_ids,
    prefill_with_single_decoder_block_input_replacement,
)
from .encoding import NativeTraceEncoding


@torch.inference_mode()
def capture_decoder_block_input_states_prefill(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    positions: Sequence[int],
    *,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Capture block inputs through the same cached prefill used for scoring."""

    selected_positions = tuple(int(value) for value in positions)
    selected_layers = tuple(sorted({int(value) for value in layers}))
    if not selected_positions or len(set(selected_positions)) != len(selected_positions):
        raise ValueError("Prefill capture positions must be unique and nonempty")
    if not selected_layers or any(
        not 0 <= layer < int(adapter.num_layers) for layer in selected_layers
    ):
        raise ValueError("Prefill capture layers are invalid")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in selected_layers:

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Decoder block input is not a positional tensor")
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return
            captured[layer] = (
                hidden[0, list(selected_positions)].detach().float().cpu()
            )

        handles.append(adapter.layers[layer].register_forward_pre_hook(hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(selected_layers) - set(captured))
    if missing:
        raise RuntimeError(f"Cached prefill capture missed layers {missing}")
    return captured


@torch.inference_mode()
def prefill_with_decoder_block_input_replacements(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    positions: Sequence[int],
    replacements: Mapping[int, torch.Tensor],
) -> tuple[Any, dict[int, int], dict[int, float]]:
    """Replace the same positions at one or more decoder-block inputs."""

    selected_positions = tuple(int(value) for value in positions)
    if not selected_positions or len(set(selected_positions)) != len(selected_positions):
        raise ValueError("Diagnostic patch positions must be unique and nonempty")
    replacement_map = {
        int(layer): torch.as_tensor(states).detach().float().cpu()
        for layer, states in replacements.items()
    }
    if not replacement_map:
        raise ValueError("Diagnostic patch needs at least one replacement layer")
    for layer, states in replacement_map.items():
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Diagnostic patch layer L{layer} is invalid")
        if states.ndim != 2 or int(states.shape[0]) != len(selected_positions):
            raise ValueError("Diagnostic states must have shape [positions, hidden]")

    applications = {layer: 0 for layer in replacement_map}
    realized_norms = {layer: 0.0 for layer in replacement_map}
    handles = []
    for layer, states in sorted(replacement_map.items()):

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            states: torch.Tensor = states,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Decoder block input is not a positional tensor")
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return None
            before = hidden[:, list(selected_positions), :]
            replacement = states.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
            if replacement.shape != before.shape:
                raise RuntimeError("Diagnostic replacement width disagrees with model")
            patched = hidden.clone()
            patched[:, list(selected_positions), :] = replacement
            realized_norms[layer] = float(
                torch.linalg.vector_norm(before.float() - replacement.float())
                .detach()
                .cpu()
            )
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    violations = sorted(layer for layer, count in applications.items() if count != 1)
    if violations:
        raise RuntimeError(
            "Every diagnostic block-input patch must apply once; bad layers "
            f"{violations}"
        )
    return prefill, applications, realized_norms


def _score_prefill(
    model: Any,
    tokenizer: Any,
    encoding: NativeTraceEncoding,
    prefill: Any,
    *,
    target_k: int,
) -> dict[str, Any]:
    outcomes = _score_and_generate_prefill(
        model,
        tokenizer,
        encoding,
        prefill,
        run_greedy=False,
        max_new_tokens=1,
    )
    return {
        **outcomes,
        **running_target_metrics(outcomes, target_k=int(target_k)),
        **_next_token_count_metrics(prefill, encoding, target_k=int(target_k)),
    }


def _next_token_count_metrics(
    prefill: Any,
    encoding: NativeTraceEncoding,
    *,
    target_k: int,
) -> dict[str, Any]:
    """Score single-token digits 1--9 directly after native ``Total:``.

    Qwen encodes ``10`` as the two tokens ``1`` and ``0``; therefore a direct
    next-token diagnostic cannot distinguish 1 from 10.  The formal joint
    sequence metric remains present separately.
    """

    logits = getattr(prefill, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Diagnostic prefill exposes no rank-3 logits")
    all_candidates = sorted(
        (int(count), tuple(int(token) for token in token_ids))
        for count, token_ids in encoding.count_candidate_answer_token_ids
    )
    if [count for count, _tokens in all_candidates] != list(range(1, 11)):
        raise RuntimeError("Next-token diagnostic requires counts 1 through 10")
    candidates = all_candidates[:9]
    if any(len(tokens) != 1 for _count, tokens in candidates):
        raise RuntimeError(
            "A candidate digit 1--9 is not a single native answer token: "
            f"{[(count, len(tokens), tokens) for count, tokens in candidates]}"
        )
    token_ids = [tokens[0] for _count, tokens in candidates]
    if len(set(token_ids)) != 9:
        raise RuntimeError("Candidate digits 1--9 do not have distinct next tokens")
    scores = logits[0, -1, token_ids].detach().float().cpu().numpy()
    probabilities = np.exp(scores - float(np.max(scores)))
    probabilities /= float(np.sum(probabilities))
    target = int(target_k)
    if not 1 <= target <= 9:
        raise ValueError("Direct next-token diagnostic target must lie in 1..9")
    target_index = target - 1
    predicted = int(np.argmax(scores)) + 1
    return {
        "next_token_candidate_logits": ",".join(str(float(value)) for value in scores),
        "next_token_candidate_probabilities": ",".join(
            str(float(value)) for value in probabilities
        ),
        "next_token_target_logit": float(scores[target_index]),
        "next_token_target_margin": float(
            scores[target_index] - np.max(np.delete(scores, target_index))
        ),
        "next_token_target_probability": float(probabilities[target_index]),
        "next_token_predicted_running_count": predicted,
        "next_token_running_target_exact": bool(predicted == target),
        "next_token_candidates_are_single_distinct_tokens": True,
        "next_token_candidate_set": "1,2,3,4,5,6,7,8,9",
        "next_token_count10_excluded_because_multitoken": True,
    }


def _candidate_log_scores(outcomes: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in str(outcomes["candidate_log_scores"]).split(",")],
        dtype=float,
    )


@torch.inference_mode()
def run_bullet_restoration_diagnostics(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layer: int,
    target_occurrences: Sequence[int],
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
) -> list[dict[str, Any]]:
    """Run identity, distributed-state, and single-item persistence checks."""

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
    layer = int(source_layer)
    if not 0 <= layer < int(adapter.num_layers):
        raise ValueError("Diagnostic source layer is invalid")
    targets = tuple(sorted({int(value) for value in target_occurrences}))
    if not targets or min(targets) < 1 or max(targets) > 10:
        raise ValueError("Diagnostic target occurrences must lie in 1..10")

    result: list[dict[str, Any]] = []
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
            raise RuntimeError("Source/Blank diagnostic geometry differs")
        if source.sequence_length != blank.sequence_length:
            raise RuntimeError("Source/Blank diagnostic sequence lengths differ")

        all_positions = tuple(range(int(source.sequence_length)))
        item_k_start, item_k_end = registry.trace_items[occurrence - 1]
        item_k_positions = tuple(range(int(item_k_start), int(item_k_end)))
        all_item_positions = tuple(
            position
            for start, end in registry.trace_items[:occurrence]
            for position in range(int(start), int(end))
        )
        source_at_layer_legacy = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            item_k_positions,
            layers=(layer,),
        )[layer]
        source_at_layer = capture_decoder_block_input_states_prefill(
            model,
            adapter,
            source,
            all_positions,
            layers=(layer,),
        )[layer]
        higher_layers = tuple(range(layer + 1, int(adapter.num_layers)))
        source_k_higher = (
            capture_decoder_block_input_states_prefill(
                model,
                adapter,
                source,
                item_k_positions,
                layers=higher_layers,
            )
            if higher_layers
            else {}
        )

        source_ids, source_mask = _encoding_tensors(model, source)
        source_prefill = _prefix_forward(model, adapter, source_ids, source_mask)
        source_outcomes = _score_prefill(
            model, tokenizer, source, source_prefill, target_k=occurrence
        )
        blank_ids, blank_mask = _encoding_tensors(model, blank)
        blank_prefill = _prefix_forward(model, adapter, blank_ids, blank_mask)
        blank_outcomes = _score_prefill(
            model, tokenizer, blank, blank_prefill, target_k=occurrence
        )
        common = {
            "schema_version": "realistic_niah_v5_bullet_restore_diagnostic_v1",
            "experiment_id": "marker_scrubbed_list_restoration_diagnostics",
            "request_id": str(source.request_id),
            "model_label": str(source.model_label),
            "seed": int(source.seed),
            "target_occurrence": int(occurrence),
            "source_layer": layer,
            "diagnostic_only": True,
            "formal_confirmation_unchanged": True,
            "patch_site": "decoder_block_input",
            "source_base_scrubbed_before_state_capture": True,
            "future_list_items_removed": 10 - int(occurrence),
            "diagnostic_suffix_used": False,
            **trace_audit,
            **registry_audit,
            **scrub_audit,
            **suffix_audit,
            **source_early_audit,
        }
        result.extend(
            [
                {**common, "condition": "source_reference", **source_outcomes},
                {**common, "condition": "blank_reference", **blank_outcomes},
            ]
        )

        single_prefill, single_apps, single_norm = (
            prefill_with_single_decoder_block_input_replacement(
                model,
                adapter,
                blank,
                positions=item_k_positions,
                layer=layer,
                replacement_states=source_at_layer_legacy,
            )
        )
        result.append(
            {
                **common,
                "condition": "single_item_once",
                "donor_capture_path": "legacy_use_cache_false",
                "patch_positions": len(item_k_positions),
                "patch_layers": [layer],
                "patch_hook_applications": {str(layer): int(single_apps)},
                "patch_realized_fro_norm_by_layer": {str(layer): float(single_norm)},
                **_score_prefill(
                    model, tokenizer, blank, single_prefill, target_k=occurrence
                ),
            }
        )

        aligned_prefill, aligned_apps, aligned_norm = (
            prefill_with_single_decoder_block_input_replacement(
                model,
                adapter,
                blank,
                positions=item_k_positions,
                layer=layer,
                replacement_states=source_at_layer[list(item_k_positions)],
            )
        )
        result.append(
            {
                **common,
                "condition": "single_item_once_cache_aligned",
                "donor_capture_path": "same_cached_prefill_as_scoring",
                "patch_positions": len(item_k_positions),
                "patch_layers": [layer],
                "patch_hook_applications": {str(layer): int(aligned_apps)},
                "patch_realized_fro_norm_by_layer": {str(layer): float(aligned_norm)},
                **_score_prefill(
                    model, tokenizer, blank, aligned_prefill, target_k=occurrence
                ),
            }
        )

        all_items_prefill, all_items_apps, all_items_norms = (
            prefill_with_decoder_block_input_replacements(
                model,
                adapter,
                blank,
                positions=all_item_positions,
                replacements={layer: source_at_layer[list(all_item_positions)]},
            )
        )
        result.append(
            {
                **common,
                "condition": "all_items_once",
                "donor_capture_path": "same_cached_prefill_as_scoring",
                "patch_positions": len(all_item_positions),
                "patch_layers": [layer],
                "patch_hook_applications": all_items_apps,
                "patch_realized_fro_norm_by_layer": all_items_norms,
                **_score_prefill(
                    model, tokenizer, blank, all_items_prefill, target_k=occurrence
                ),
            }
        )

        cumulative_replacements = {
            layer: source_at_layer[list(item_k_positions)],
            **source_k_higher,
        }
        cumulative_prefill, cumulative_apps, cumulative_norms = (
            prefill_with_decoder_block_input_replacements(
                model,
                adapter,
                blank,
                positions=item_k_positions,
                replacements=cumulative_replacements,
            )
        )
        result.append(
            {
                **common,
                "condition": "single_item_cumulative_clamp",
                "donor_capture_path": "same_cached_prefill_as_scoring",
                "patch_positions": len(item_k_positions),
                "patch_layers": list(range(layer, int(adapter.num_layers))),
                "patch_hook_applications": cumulative_apps,
                "patch_realized_fro_norm_by_layer": cumulative_norms,
                **_score_prefill(
                    model, tokenizer, blank, cumulative_prefill, target_k=occurrence
                ),
            }
        )

        identity_prefill, identity_apps, identity_norms = (
            prefill_with_decoder_block_input_replacements(
                model,
                adapter,
                blank,
                positions=all_positions,
                replacements={layer: source_at_layer},
            )
        )
        identity_outcomes = _score_prefill(
            model, tokenizer, blank, identity_prefill, target_k=occurrence
        )
        max_abs_delta = float(
            np.max(
                np.abs(
                    _candidate_log_scores(identity_outcomes)
                    - _candidate_log_scores(source_outcomes)
                )
            )
        )
        next_token_max_abs_delta = float(
            np.max(
                np.abs(
                    np.asarray(
                        [
                            float(value)
                            for value in str(
                                identity_outcomes["next_token_candidate_logits"]
                            ).split(",")
                        ]
                    )
                    - np.asarray(
                        [
                            float(value)
                            for value in str(
                                source_outcomes["next_token_candidate_logits"]
                            ).split(",")
                        ]
                    )
                )
            )
        )
        source_last_logits = source_prefill.logits[0, -1].detach().float().cpu()
        identity_last_logits = identity_prefill.logits[0, -1].detach().float().cpu()
        prefill_last_logit_max_abs_delta = float(
            torch.max(torch.abs(source_last_logits - identity_last_logits)).item()
        )
        result.append(
            {
                **common,
                "condition": "full_prefix_identity",
                "donor_capture_path": "same_cached_prefill_as_scoring",
                "patch_positions": len(all_positions),
                "patch_layers": [layer],
                "patch_hook_applications": identity_apps,
                "patch_realized_fro_norm_by_layer": identity_norms,
                "source_candidate_log_score_max_abs_delta": max_abs_delta,
                "source_next_token_candidate_logit_max_abs_delta": (
                    next_token_max_abs_delta
                ),
                "source_prefill_last_logit_max_abs_delta": (
                    prefill_last_logit_max_abs_delta
                ),
                **identity_outcomes,
            }
        )
    return result
