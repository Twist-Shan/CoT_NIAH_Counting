#!/usr/bin/env python3
"""Cross-seed behavioral assay of an unindexed boundary counter and its +1 update.

The readout is the model's native next bullet, not a diagnostic ``Total:``
suffix.  A donor boundary state from trace A is placed at a fixed, blanked
carrier boundary in trace B.  After one teacher-forced, unnumbered B item, the
propagated state is transplanted into a blank trace C.  A count-like recurrence
predicts B[k+1] before the transition and C[k+2] after it.

This runner is intended for the explicitly labeled format-conditioned
unnumbered auxiliary cohort.  It refuses rows that fail the causal-prefix
no-enumeration audit.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
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
    _encoding_tensors,
)
from realistic_niah_v4_4_3.interventions import (  # noqa: E402
    _output_logits,
    _repeat_batch_tree,
)
from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.bullet_counterfactual_restore import (  # noqa: E402
    _replace_positions_from_pool,
    _safe_ordinary_prompt_token_pool,
)
from realistic_niah_v5.boundary_counter_probe import (  # noqa: E402
    transplant_boundary_and_capture_later_state,
)
from realistic_niah_v5.count_stream import (  # noqa: E402
    _prefix_forward,
    build_answer_source_registry,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
    prefill_with_single_decoder_block_input_replacement,
)
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_no_count_enumeration_trace,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)


def _read_rows(path: Path, seeds: Sequence[int]) -> dict[int, dict[str, Any]]:
    wanted = {int(value) for value in seeds}
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row.get("seed", -1))
        if seed not in wanted:
            continue
        if seed in selected:
            raise ValueError(f"Duplicate requested seed {seed}")
        audit = audit_no_count_enumeration_trace(row)
        if not audit["eligible"]:
            raise ValueError(f"Seed {seed} fails unindexed audit: {audit['reasons']}")
        if int(row.get("gold_count", -1)) != 10:
            raise ValueError(f"Seed {seed} is not N=10")
        selected[seed] = row
    if set(selected) != wanted:
        raise ValueError(f"Missing requested seeds: {sorted(wanted-set(selected))}")
    return selected


def _triples(values: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    flat = tuple(int(value) for value in values)
    if not flat or len(flat) % 3:
        raise ValueError("Seed triples must be supplied as A B C groups")
    result = tuple(zip(flat[::3], flat[1::3], flat[2::3]))
    if any(len(set(group)) != 3 for group in result):
        raise ValueError("Each donor/transition/assay triple must use distinct seeds")
    return result


def _pairs(values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    flat = tuple(int(value) for value in values)
    if not flat or len(flat) % 2:
        raise ValueError("Layer pairs must be supplied as L_in L_out groups")
    pairs = tuple(zip(flat[::2], flat[1::2]))
    if any(left >= right for left, right in pairs):
        raise ValueError("Every transition read layer must exceed its patch layer")
    return pairs


def _contains_alphanumeric(tokenizer: Any, token_id: int) -> bool:
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return any(character.isalnum() for character in text)


def build_trace_blank(
    encoding: Any,
    registry: Any,
    tokenizer: Any,
    *,
    salt: str,
) -> tuple[Any, dict[str, Any]]:
    """Blank item semantics while preserving prompt evidence and separators."""

    original = tuple(int(value) for value in encoding.input_ids)
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    positions = tuple(
        position
        for start, end in registry.trace_items
        for position in range(int(start), int(end))
        if original[position] not in special
        and _contains_alphanumeric(tokenizer, original[position])
    )
    pool = _safe_ordinary_prompt_token_pool(encoding, registry, tokenizer)
    blank_ids, changed = _replace_positions_from_pool(
        original,
        positions,
        pool,
        salt=str(salt),
        tokenizer=tokenizer,
    )
    blank = replace(encoding, input_ids=tuple(blank_ids))
    return blank, {
        "trace_item_semantic_positions": len(positions),
        "changed_token_count": int(changed),
        "prompt_record_tokens_preserved": True,
        "trace_separators_preserved": True,
    }


def prefix_through_boundary(encoding: Any, boundary: int) -> Any:
    end = int(boundary) + 1
    if not 0 < end <= int(encoding.sequence_length):
        raise ValueError("Carrier boundary is outside the encoding")
    return replace(
        encoding,
        input_ids=tuple(encoding.input_ids[:end]),
        attention_mask=tuple(encoding.attention_mask[:end]),
        query_position=end - 1,
    )


def append_transition_segment(
    prefix: Any,
    source: Any,
    registry: Any,
    *,
    occurrence: int,
) -> tuple[Any, dict[str, int]]:
    k = int(occurrence)
    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    if not 1 <= k < len(items):
        raise ValueError("Transition item must have a following item separator")
    start = items[k - 1][0]
    next_start = items[k][0]
    segment = tuple(int(value) for value in source.input_ids[start:next_start])
    if not segment:
        raise ValueError("Transition item segment is empty")
    ids = tuple(prefix.input_ids) + segment
    mask = tuple(prefix.attention_mask) + (1,) * len(segment)
    result = replace(prefix, input_ids=ids, attention_mask=mask, query_position=len(ids) - 1)
    return result, {
        "transition_occurrence": k,
        "transition_token_count": len(segment),
        "transition_start": len(prefix.input_ids),
        "transition_end": len(ids),
        "transition_boundary": len(ids) - 1,
    }


def item_candidate_tokens(encoding: Any, registry: Any) -> dict[int, tuple[int, ...]]:
    result = {
        occurrence: tuple(int(value) for value in encoding.input_ids[int(start) : int(end)])
        for occurrence, (start, end) in enumerate(registry.trace_items, start=1)
    }
    if set(result) != set(range(1, 11)) or any(len(tokens) < 2 for tokens in result.values()):
        raise ValueError("Native next-item candidates must cover ten nontrivial items")
    if len(set(result.values())) != 10:
        raise ValueError("Native next-item candidate token sequences are not unique")
    return result


def _clone_prefill(prefill: Any) -> Any:
    past = getattr(prefill, "past_key_values", None)
    if past is None:
        raise RuntimeError("Patched candidate prefill returned no KV cache")
    values: dict[str, Any] = {
        "logits": _output_logits(prefill),
        "past_key_values": copy.deepcopy(past),
    }
    shared = getattr(prefill, "shared_kv_states", None)
    if shared is not None:
        values["shared_kv_states"] = copy.deepcopy(shared)
    return type("CandidatePrefill", (), values)()


@torch.inference_mode()
def score_native_item_candidates(
    model: Any,
    prefix: Any,
    prefill: Any,
    candidates: Mapping[int, Sequence[int]],
    *,
    target: int,
    baseline: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Score ten receiver-native item strings from one patched prefix cache."""

    branch = _clone_prefill(prefill)
    prefill_logits = _output_logits(branch)[0, -1].detach().float()
    past = branch.past_key_values
    ordered = [(k, tuple(int(value) for value in candidates[k])) for k in range(1, 11)]
    max_inputs = max(len(tokens) - 1 for _k, tokens in ordered)
    device = prefill_logits.device
    continuation_ids = torch.zeros((10, max_inputs), dtype=torch.long, device=device)
    continuation_mask = torch.zeros_like(continuation_ids)
    for row, (_k, tokens) in enumerate(ordered):
        values = tokens[:-1]
        continuation_ids[row, : len(values)] = torch.tensor(values, device=device)
        continuation_mask[row, : len(values)] = 1
    repeater = getattr(past, "batch_repeat_interleave", None)
    if not callable(repeater):
        raise RuntimeError("Transformers cache cannot branch native item candidates")
    repeater(10)
    base_mask = torch.tensor(
        [prefix.attention_mask], dtype=torch.long, device=device
    ).repeat(10, 1)
    attention_mask = torch.cat((base_mask, continuation_mask), dim=1)
    kwargs: dict[str, Any] = {
        "input_ids": continuation_ids,
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": False,
    }
    positions = torch.arange(
        int(prefix.sequence_length),
        int(prefix.sequence_length) + max_inputs,
        dtype=torch.long,
        device=device,
    )
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = positions.unsqueeze(0).expand(10, -1)
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = positions
    shared = getattr(branch, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = _repeat_batch_tree(shared, 10)
    continuation = model(**kwargs)
    logits = _output_logits(continuation).detach().float()
    first_log_probs = torch.log_softmax(prefill_logits, dim=-1)
    later_log_probs = torch.log_softmax(logits, dim=-1)
    sum_scores: list[float] = []
    mean_scores: list[float] = []
    for row, (_k, tokens) in enumerate(ordered):
        score = first_log_probs[tokens[0]]
        for offset, token in enumerate(tokens[1:]):
            score = score + later_log_probs[row, offset, token]
        value = float(score.detach().cpu())
        sum_scores.append(value)
        mean_scores.append(value / len(tokens))
    target_index = int(target) - 1
    other_mean = [value for index, value in enumerate(mean_scores) if index != target_index]
    other_sum = [value for index, value in enumerate(sum_scores) if index != target_index]
    mean_shifted = np.asarray(mean_scores) - float(max(mean_scores))
    probabilities = np.exp(mean_shifted)
    probabilities /= probabilities.sum()
    result = {
        "target_occurrence": int(target),
        "predicted_occurrence_mean_logprob": int(np.argmax(mean_scores)) + 1,
        "predicted_occurrence_sum_logprob": int(np.argmax(sum_scores)) + 1,
        "target_exact_mean_logprob": bool(int(np.argmax(mean_scores)) == target_index),
        "target_exact_sum_logprob": bool(int(np.argmax(sum_scores)) == target_index),
        "target_mean_logprob_margin": float(mean_scores[target_index] - max(other_mean)),
        "target_sum_logprob_margin": float(sum_scores[target_index] - max(other_sum)),
        "target_probability_mean_logprob": float(probabilities[target_index]),
        "mean_logprob_scores": [float(value) for value in mean_scores],
        "sum_logprob_scores": [float(value) for value in sum_scores],
        "candidate_token_counts": [len(tokens) for _k, tokens in ordered],
    }
    if baseline is not None:
        baseline_mean = np.asarray(baseline["mean_logprob_scores"], dtype=float)
        baseline_sum = np.asarray(baseline["sum_logprob_scores"], dtype=float)
        if baseline_mean.shape != (10,) or baseline_sum.shape != (10,):
            raise ValueError("Native-item baseline must contain ten candidate scores")
        delta_mean = np.asarray(mean_scores, dtype=float) - baseline_mean
        delta_sum = np.asarray(sum_scores, dtype=float) - baseline_sum
        other_delta_mean = np.delete(delta_mean, target_index)
        other_delta_sum = np.delete(delta_sum, target_index)
        delta_shifted = delta_mean - float(np.max(delta_mean))
        delta_probabilities = np.exp(delta_shifted)
        delta_probabilities /= delta_probabilities.sum()
        result.update(
            {
                "baseline_corrected": True,
                "delta_mean_logprob_scores": delta_mean.tolist(),
                "delta_sum_logprob_scores": delta_sum.tolist(),
                "predicted_occurrence_delta_mean_logprob": int(np.argmax(delta_mean))
                + 1,
                "predicted_occurrence_delta_sum_logprob": int(np.argmax(delta_sum)) + 1,
                "target_exact_delta_mean_logprob": bool(
                    int(np.argmax(delta_mean)) == target_index
                ),
                "target_exact_delta_sum_logprob": bool(
                    int(np.argmax(delta_sum)) == target_index
                ),
                "target_delta_mean_logprob_margin": float(
                    delta_mean[target_index] - np.max(other_delta_mean)
                ),
                "target_delta_sum_logprob_margin": float(
                    delta_sum[target_index] - np.max(other_delta_sum)
                ),
                "target_probability_delta_mean_logprob": float(
                    delta_probabilities[target_index]
                ),
            }
        )
    return result


@torch.inference_mode()
def unpatched_candidate_baseline(
    model: Any,
    adapter: Any,
    prefix: Any,
    candidates: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """Score receiver-native items before a donor state is introduced."""

    input_ids, attention_mask = _encoding_tensors(model, prefix)
    prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    return score_native_item_candidates(
        model,
        prefix,
        prefill,
        candidates,
        target=1,
    )


def patched_candidate_readout(
    model: Any,
    adapter: Any,
    prefix: Any,
    *,
    layer: int,
    state: torch.Tensor,
    candidates: Mapping[int, Sequence[int]],
    target: int,
    baseline: Mapping[str, Sequence[float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    position = int(prefix.sequence_length) - 1
    prefill, applications, norm = prefill_with_single_decoder_block_input_replacement(
        model,
        adapter,
        prefix,
        positions=(position,),
        layer=int(layer),
        replacement_states=torch.as_tensor(state).reshape(1, -1),
    )
    scored = score_native_item_candidates(
        model,
        prefix,
        prefill,
        candidates,
        target=int(target),
        baseline=baseline,
    )
    return scored, {
        "patch_position": position,
        "patch_layer": int(layer),
        "patch_applications": int(applications),
        "patch_realized_l2_norm": float(norm),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["patch_layer"]), int(row["read_layer"]), str(row["condition"]))
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (patch_layer, read_layer, condition), values in sorted(grouped.items()):
        result.append(
            {
                "patch_layer": patch_layer,
                "read_layer": read_layer,
                "condition": condition,
                "trial_count": len(values),
                "exact_accuracy_mean_logprob": float(
                    np.mean([bool(value["target_exact_mean_logprob"]) for value in values])
                ),
                "exact_accuracy_sum_logprob": float(
                    np.mean([bool(value["target_exact_sum_logprob"]) for value in values])
                ),
                "exact_accuracy_delta_mean_logprob": float(
                    np.mean(
                        [bool(value["target_exact_delta_mean_logprob"]) for value in values]
                    )
                ),
                "exact_accuracy_delta_sum_logprob": float(
                    np.mean(
                        [bool(value["target_exact_delta_sum_logprob"]) for value in values]
                    )
                ),
                "mean_target_margin": float(
                    np.mean([float(value["target_mean_logprob_margin"]) for value in values])
                ),
                "mean_target_probability": float(
                    np.mean([float(value["target_probability_mean_logprob"]) for value in values])
                ),
                "mean_target_delta_margin": float(
                    np.mean(
                        [float(value["target_delta_mean_logprob_margin"]) for value in values]
                    )
                ),
                "mean_target_delta_probability": float(
                    np.mean(
                        [float(value["target_probability_delta_mean_logprob"]) for value in values]
                    )
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seed-triples", type=int, nargs="+", required=True)
    parser.add_argument("--occurrences", type=int, nargs="+", default=(2, 5, 8))
    parser.add_argument("--carrier-occurrence", type=int, default=5)
    parser.add_argument("--layer-pairs", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    triples = _triples(args.seed_triples)
    pairs = _pairs(args.layer_pairs)
    occurrences = tuple(int(value) for value in args.occurrences)
    if not occurrences or min(occurrences) < 1 or max(occurrences) > 8:
        raise ValueError("Recurrence occurrences must lie in 1..8")
    carrier = int(args.carrier_occurrence)
    if not 1 <= carrier < 10:
        raise ValueError("Carrier occurrence must lie in 1..9")
    seeds = tuple(sorted({seed for group in triples for seed in group}))
    rows = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    if any(left < 0 or right >= int(adapter.num_layers) for left, right in pairs):
        raise ValueError("One or more layer pairs are outside the decoder")

    compiled: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        row = rows[seed]
        encoding, registry = build_answer_source_registry(row, tokenizer)
        blank, blank_audit = build_trace_blank(
            encoding,
            registry,
            tokenizer,
            salt=f"cross-seed-counter|{args.model}|{seed}",
        )
        boundary, boundary_audit = select_post_item_boundary_position(
            blank, registry, tokenizer, occurrence=carrier
        )
        compiled[seed] = {
            "row": row,
            "source": encoding,
            "blank": blank,
            "registry": registry,
            "carrier_boundary": boundary,
            "carrier_prefix": prefix_through_boundary(blank, boundary),
            "candidates": item_candidate_tokens(encoding, registry),
            "blank_audit": blank_audit,
            "boundary_audit": boundary_audit,
        }

    for seed in seeds:
        value = compiled[seed]
        value["candidate_baseline"] = unpatched_candidate_baseline(
            model,
            adapter,
            value["carrier_prefix"],
            value["candidates"],
        )

    results: list[dict[str, Any]] = []
    all_layers = tuple(sorted({value for pair in pairs for value in pair}))
    for donor_seed, transition_seed, assay_seed in triples:
        donor = compiled[donor_seed]
        transition = compiled[transition_seed]
        assay = compiled[assay_seed]
        donor_boundaries = {
            k: select_post_item_boundary_position(
                donor["source"], donor["registry"], tokenizer, occurrence=k
            )[0]
            for k in sorted(set(occurrences) | {k + 1 for k in occurrences})
        }
        ordered_occurrences = tuple(sorted(donor_boundaries))
        captured = capture_decoder_block_input_states(
            model,
            adapter,
            donor["source"],
            tuple(donor_boundaries[k] for k in ordered_occurrences),
            layers=all_layers,
        )
        state_index = {k: index for index, k in enumerate(ordered_occurrences)}

        for patch_layer, read_layer in pairs:
            for k in occurrences:
                common = {
                    "schema_version": "cross_seed_unindexed_counter_recurrence_v1",
                    "model_label": str(args.model),
                    "donor_seed": donor_seed,
                    "transition_seed": transition_seed,
                    "assay_seed": assay_seed,
                    "donor_occurrence": k,
                    "carrier_occurrence": carrier,
                    "patch_layer": patch_layer,
                    "read_layer": read_layer,
                    "readout": "native_next_bullet_10_way",
                    "primary_scoring": "blank_baseline_corrected_mean_token_logprob",
                    "diagnostic_total_suffix_used": False,
                    "format_conditioned_auxiliary": True,
                }
                state_in = captured[patch_layer][state_index[k]]
                state_same_read = captured[read_layer][state_index[k]]
                natural_successor = captured[read_layer][state_index[k + 1]]

                direct, direct_patch = patched_candidate_readout(
                    model,
                    adapter,
                    transition["carrier_prefix"],
                    layer=patch_layer,
                    state=state_in,
                    candidates=transition["candidates"],
                    target=k + 1,
                    baseline=transition["candidate_baseline"],
                )
                results.append(
                    {**common, "condition": "cross_seed_direct", **direct_patch, **direct}
                )

                transition_encoding, transition_audit = append_transition_segment(
                    transition["carrier_prefix"],
                    transition["source"],
                    transition["registry"],
                    occurrence=k + 1,
                )
                propagated, patch_apps, read_apps, transition_norm = (
                    transplant_boundary_and_capture_later_state(
                        model,
                        adapter,
                        transition_encoding,
                        patch_position=int(transition["carrier_prefix"].sequence_length) - 1,
                        patch_layer=patch_layer,
                        replacement_state=state_in,
                        read_position=int(transition_encoding.sequence_length) - 1,
                        read_layer=read_layer,
                    )
                )
                propagated_score, propagated_patch = patched_candidate_readout(
                    model,
                    adapter,
                    assay["carrier_prefix"],
                    layer=read_layer,
                    state=propagated,
                    candidates=assay["candidates"],
                    target=k + 2,
                    baseline=assay["candidate_baseline"],
                )
                results.append(
                    {
                        **common,
                        "condition": "propagated_plus_one",
                        "transition_patch_applications": int(patch_apps),
                        "transition_read_applications": int(read_apps),
                        "transition_patch_realized_l2_norm": float(transition_norm),
                        **transition_audit,
                        **propagated_patch,
                        **propagated_score,
                    }
                )

                same_score, same_patch = patched_candidate_readout(
                    model,
                    adapter,
                    assay["carrier_prefix"],
                    layer=read_layer,
                    state=state_same_read,
                    candidates=assay["candidates"],
                    target=k + 1,
                    baseline=assay["candidate_baseline"],
                )
                results.append(
                    {**common, "condition": "no_transition_state", **same_patch, **same_score}
                )

                successor_score, successor_patch = patched_candidate_readout(
                    model,
                    adapter,
                    assay["carrier_prefix"],
                    layer=read_layer,
                    state=natural_successor,
                    candidates=assay["candidates"],
                    target=k + 2,
                    baseline=assay["candidate_baseline"],
                )
                results.append(
                    {
                        **common,
                        "condition": "natural_successor_ceiling",
                        **successor_patch,
                        **successor_score,
                    }
                )
                print(
                    f"[cross-seed-recurrence] model={args.model} A/B/C="
                    f"{donor_seed}/{transition_seed}/{assay_seed} k={k} "
                    f"L{patch_layer}->L{read_layer} direct="
                    f"{direct['predicted_occurrence_delta_mean_logprob']} propagated="
                    f"{propagated_score['predicted_occurrence_delta_mean_logprob']} ceiling="
                    f"{successor_score['predicted_occurrence_delta_mean_logprob']}",
                    flush=True,
                )

    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output / "trials.jsonl", results)
    summary_rows = _summary(results)
    _atomic_jsonl(args.output / "summary.csv.jsonl", summary_rows)
    manifest = {
        "schema_version": "cross_seed_unindexed_counter_recurrence_v1",
        "status": "PASS",
        "model_label": str(args.model),
        "seed_triples": [list(value) for value in triples],
        "occurrences": list(occurrences),
        "carrier_occurrence": carrier,
        "layer_pairs": [list(value) for value in pairs],
        "trial_count": len(results),
        "conditions": sorted({str(value["condition"]) for value in results}),
        "summary": summary_rows,
        "selection_uses_final_answer": False,
        "diagnostic_total_suffix_used": False,
        "formal_frozen_prompt_claim_allowed": False,
        "claim_scope": "format-conditioned unnumbered reasoning auxiliary",
    }
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
