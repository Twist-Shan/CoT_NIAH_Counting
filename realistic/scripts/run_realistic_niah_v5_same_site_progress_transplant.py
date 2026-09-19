#!/usr/bin/env python3
"""Patch logical progress states captured at one physical boundary.

Equal-length event-factorial branches share the same prompt, separators,
absolute commit position, and surface commit token.  They differ only in the
semantic validity of copied event payloads.  The complete state from
logical count ``k`` is transplanted into the count-``j`` branch at that shared
site.  The assay then measures native-item scores, frozen-bank attention to
``N[j+1]`` versus ``N[k+1]``, greedy first-item adoption, and remaining bullet
events before the reasoning close.
"""

from __future__ import annotations

import argparse
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
    _bounded_logits_kwargs,
    _encoding_tensors,
    _replace_output_tensor,
    _tensor_from_output,
)
from realistic_niah_v4_4_5.restoration import (  # noqa: E402
    generate_answer_completion_from_prefill,
)
from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.bullet_counterfactual_restore import (  # noqa: E402
    _replace_positions_from_pool,
    _safe_ordinary_prompt_token_pool,
)
from realistic_niah_v5.count_stream import (  # noqa: E402
    _uses_shared_kv,
)
from realistic_niah_v5.counting_mechanism_transfer import (  # noqa: E402
    build_first_pass_tstar_answer_source_registry,
)
from realistic_niah_v5.event_ledger import (  # noqa: E402
    build_semantic_event_factorial,
)
from realistic_niah_v5.native_loop import (  # noqa: E402
    _query_from_prefix_with_attentions,
    full_commit_specificity_condition_states,
    load_frozen_targeted_bank,
)
from realistic_niah_v5.parsing import gold_records  # noqa: E402
from realistic_niah_v5.same_site_progress_transplant import (  # noqa: E402
    canonical_marker_bits,
    donor_receiver_logodds,
    generated_bullet_city_ordinals,
    native_item_candidates,
    query_prefix_before_city,
    select_count_cells,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_cross_seed_counter_recurrence import (  # noqa: E402
    _output_logits,
    _repeat_batch_tree,
    prefix_through_boundary,
)


SCHEMA_VERSION = "same_site_progress_transplant_v1"
ALL_CONDITIONS = (
    "self_patch",
    "native_donor_branch",
    "full_donor_patch",
    "same_count_alternative_patch",
    "full_delta_norm_matched_orthogonal_r0",
    "opposite_full_delta_patch",
    "wrong_count_natural_patch",
)


def _experiment_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Load the model, optionally forcing an actual two-GPU layer split.

    Accelerate's ``auto`` and ``balanced`` maps keep Qwen3-8B entirely on one
    80 GiB H100 because its weights fit.  The roughly 10k-token SDPA prefill
    then exhausts that device.  ``balanced-split`` explicitly places the
    first and second halves of decoder blocks on separate GPUs; this reduces
    resident weights per attention-activation device without changing any
    model computation.
    """

    if str(args.device_map) != "balanced-split":
        return _model(args)
    if torch.cuda.device_count() < 2:
        raise RuntimeError("balanced-split requires at least two CUDA devices")

    import transformers

    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec

    model_spec = resolve_model_spec(args.model)
    config = transformers.AutoConfig.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        trust_remote_code=False,
    )
    num_layers = int(config.num_hidden_layers)
    split = (num_layers + 1) // 2
    device_map: dict[str, int] = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": 1,
        "lm_head": 1,
    }
    for layer in range(num_layers):
        device_map[f"model.layers.{layer}"] = 0 if layer < split else 1
    return load_registered_model(
        model_spec,
        cache_dir=args.cache_dir,
        device_map=device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )


def _chunk_forward(
    model: Any,
    adapter: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    start: int,
    end: int,
    previous: Any | None,
) -> Any:
    """Advance an exact causal prefill over ``[start, end)``."""

    if not 0 <= int(start) < int(end) <= int(input_ids.shape[1]):
        raise ValueError("Invalid chunk interval")
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, int(start) : int(end)],
        "attention_mask": attention_mask[:, : int(end)],
        "use_cache": True,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if previous is not None:
        past = getattr(previous, "past_key_values", None)
        if past is None:
            raise RuntimeError("A chunked prefill step returned no KV cache")
        kwargs["past_key_values"] = past
        shared = getattr(previous, "shared_kv_states", None)
        if shared is not None and _accepts_keyword(model, "shared_kv_states"):
            kwargs["shared_kv_states"] = shared
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.arange(
            int(start), int(end), dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.arange(
            int(start), int(end), dtype=torch.long, device=input_ids.device
        )
    if _uses_shared_kv(adapter):
        kwargs["return_shared_kv_states"] = True
    return model(**kwargs)


@torch.inference_mode()
def _chunked_capture_site_states(
    model: Any,
    adapter: Any,
    encoding: Any,
    *,
    site: int,
    layers: Sequence[int],
    chunk_size: int,
) -> dict[int, torch.Tensor]:
    """Capture post-block states at one site without a quadratic full prefill."""

    if int(site) != int(encoding.sequence_length) - 1:
        raise ValueError("The capture encoding must end at the shared site")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    previous = None
    for start in range(0, int(site), int(chunk_size)):
        previous = _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=start,
            end=min(int(site), start + int(chunk_size)),
            previous=previous,
        )
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or int(hidden.shape[1]) != 1:
                raise RuntimeError("Shared-site capture expected a one-token chunk")
            captured[layer] = hidden[0, 0].detach().float().cpu()

        return hook

    for layer in layers:
        handles.append(adapter.layers[int(layer)].register_forward_hook(make_hook(int(layer))))
    try:
        _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=int(site),
            end=int(site) + 1,
            previous=previous,
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = set(int(value) for value in layers) - set(captured)
    if missing:
        raise RuntimeError(f"Failed to capture shared-site layers: {sorted(missing)}")
    return captured


@torch.inference_mode()
def _chunked_prefill(
    model: Any,
    adapter: Any,
    encoding: Any,
    *,
    chunk_size: int,
) -> Any:
    """Prefill an unchanged causal prefix in bounded-size exact chunks."""

    input_ids, attention_mask = _encoding_tensors(model, encoding)
    previous = None
    for start in range(0, int(encoding.sequence_length), int(chunk_size)):
        previous = _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=start,
            end=min(int(encoding.sequence_length), start + int(chunk_size)),
            previous=previous,
        )
    if previous is None:
        raise ValueError("Cannot prefill an empty encoding")
    return previous


@torch.inference_mode()
def _chunked_prefill_with_site_replacement(
    model: Any,
    adapter: Any,
    encoding: Any,
    *,
    layer: int,
    site: int,
    state: torch.Tensor,
    chunk_size: int,
) -> tuple[Any, int, float]:
    """Patch one causal site and finish the prefill in bounded-size chunks."""

    if not 0 <= int(site) < int(encoding.sequence_length):
        raise ValueError("The shared site lies outside the encoding")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    previous = None
    for start in range(0, int(site), int(chunk_size)):
        previous = _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=start,
            end=min(int(site), start + int(chunk_size)),
            previous=previous,
        )

    applications = 0
    realized_norm = 0.0

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        nonlocal applications, realized_norm
        hidden = _tensor_from_output(output)
        if hidden.ndim != 3 or int(hidden.shape[1]) != 1:
            raise RuntimeError("Shared-site patch expected a one-token chunk")
        replacement = state.to(device=hidden.device, dtype=hidden.dtype).reshape(1, -1)
        before = hidden[:, 0, :]
        patched = hidden.clone()
        patched[:, 0, :] = replacement
        realized_norm = float(
            torch.linalg.vector_norm(before.float() - replacement.float())
            .detach()
            .cpu()
        )
        applications += 1
        return _replace_output_tensor(output, patched)

    handle = adapter.layers[int(layer)].register_forward_hook(hook)
    try:
        previous = _chunk_forward(
            model,
            adapter,
            input_ids,
            attention_mask,
            start=int(site),
            end=int(site) + 1,
            previous=previous,
        )
    finally:
        handle.remove()
    if applications != 1:
        raise RuntimeError(f"Shared-site patch applied {applications} times")
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


def _read_rows(
    path: Path,
    *,
    gold_count: int,
    seeds: Sequence[int] | None,
    max_seeds: int | None,
    selection_population: str = "first_pass_noindex_enumeration",
    eligibility_field: str = "primary_eligible_prefix_clean",
) -> list[dict[str, Any]]:
    requested = None if seeds is None else {int(value) for value in seeds}
    rows: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    # Do not use ``str.splitlines`` here: native passages contain literal
    # U+2028/U+2029 separators inside otherwise valid JSON strings, and Python
    # treats those codepoints as line boundaries.  File iteration splits only
    # the JSONL record delimiter.
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("gold_count", -1)) != int(gold_count):
                continue
            if requested is not None and int(row["seed"]) not in requested:
                continue
            request_id = str(row["request_id"])
            if request_id in request_ids:
                raise ValueError(f"Duplicate request_id {request_id}")
            audit_rows = [
                value
                for key, value in row.items()
                if str(key).endswith("_format_audit")
                and isinstance(value, Mapping)
                and str(eligibility_field) in value
            ]
            cohort_rows = [
                value
                for key, value in row.items()
                if str(key).endswith("_cohort")
                and isinstance(value, Mapping)
                and str(value.get("selection_population", ""))
                == str(selection_population)
            ]
            if len(audit_rows) != 1 or len(cohort_rows) != 1:
                raise ValueError(
                    f"Seed {row.get('seed')} lacks one frozen "
                    f"{selection_population} audit/cohort"
                )
            if not bool(audit_rows[0].get(str(eligibility_field))):
                raise ValueError(
                    f"Seed {row.get('seed')} fails the frozen "
                    f"{eligibility_field} gate"
                )
            request_ids.add(request_id)
            rows.append(row)
    rows.sort(key=lambda value: int(value["seed"]))
    if requested is not None:
        missing = sorted(requested - {int(row["seed"]) for row in rows})
        if missing:
            raise ValueError(f"Requested seeds are absent: {missing}")
    if max_seeds is not None:
        rows = rows[: int(max_seeds)]
    if not rows:
        raise ValueError("No eligible generation rows were selected")
    return rows


def _semantic_neutral_encoding(
    encoding: Any,
    registry: Any,
    tokenizer: Any,
    *,
    random_seed: int,
    protected_positions: Sequence[int] = (),
) -> tuple[Any, dict[str, Any]]:
    protected = {int(value) for value in protected_positions}
    item_positions = tuple(
        position
        for start, end in registry.trace_items
        for position in range(int(start), int(end))
    )
    semantic_positions = tuple(
        position
        for position in item_positions
        if position not in protected
        if any(
            character.isalnum()
            for character in tokenizer.decode(
                [int(encoding.input_ids[position])],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    )
    if not semantic_positions:
        raise ValueError("The first-pass trace has no semantic event tokens")
    original = tuple(int(value) for value in encoding.input_ids)
    pool = _safe_ordinary_prompt_token_pool(encoding, registry, tokenizer)
    neutral_ids, changed = _replace_positions_from_pool(
        original,
        semantic_positions,
        pool,
        salt=f"{encoding.request_id}|same-site-semantic-neutral|{int(random_seed)}",
        tokenizer=tokenizer,
    )
    changed_positions = tuple(
        index
        for index, (left, right) in enumerate(zip(original, neutral_ids))
        if int(left) != int(right)
    )
    if changed_positions != tuple(sorted(semantic_positions)):
        raise RuntimeError("Semantic-neutral branch changed a structural token")
    neutral = replace(encoding, input_ids=neutral_ids)
    return neutral, {
        "neutral_semantic_token_count": len(semantic_positions),
        "neutral_semantic_changed_token_count": int(changed),
        "neutral_semantic_positions": list(semantic_positions),
        "neutral_protected_commit_positions": sorted(protected),
        "neutral_semantic_only_edit": True,
        "structural_punctuation_and_whitespace_preserved": True,
    }


def _query_tokens_before_city(
    tokenizer: Any,
    candidate_tokens: Sequence[int],
    city: str,
) -> tuple[int, ...]:
    errors: list[str] = []
    for surface in (str(city), " " + str(city), "\n" + str(city), ", " + str(city)):
        city_ids = tuple(
            int(value)
            for value in tokenizer.encode(surface, add_special_tokens=False)
        )
        try:
            return query_prefix_before_city(candidate_tokens, city_ids)
        except ValueError as error:
            errors.append(str(error))
    raise ValueError(
        f"The successor city {city!r} has no token-boundary match: {errors}"
    )


@torch.inference_mode()
def _score_candidates(
    model: Any,
    prefix: Any,
    prefill: Any,
    candidates: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """Generic-N version of the native candidate sequence scorer."""

    ordered_keys = tuple(sorted(int(value) for value in candidates))
    if ordered_keys != tuple(range(1, len(ordered_keys) + 1)):
        raise ValueError("Candidate ordinals must be contiguous from one")
    ordered = [
        (key, tuple(int(value) for value in candidates[key]))
        for key in ordered_keys
    ]
    if any(not tokens for _key, tokens in ordered):
        raise ValueError("A candidate sequence is empty")
    # Candidate expansion mutates the Transformers cache in place.  This
    # scorer therefore owns ``prefill``; callers must run greedy generation
    # first (or build a separate prefill) rather than relying on a cache
    # deepcopy, which is unsupported by current non-leaf DynamicCache tensors.
    branch = prefill
    prefill_logits = _output_logits(branch)[0, -1].detach().float()
    past = branch.past_key_values
    count = len(ordered)
    max_inputs = max(len(tokens) - 1 for _key, tokens in ordered)
    device = prefill_logits.device
    continuation_ids = torch.zeros(
        (count, max_inputs), dtype=torch.long, device=device
    )
    continuation_mask = torch.zeros_like(continuation_ids)
    for row_index, (_key, tokens) in enumerate(ordered):
        values = tokens[:-1]
        if values:
            continuation_ids[row_index, : len(values)] = torch.tensor(
                values, dtype=torch.long, device=device
            )
            continuation_mask[row_index, : len(values)] = 1
    repeater = getattr(past, "batch_repeat_interleave", None)
    if not callable(repeater):
        raise RuntimeError("Transformers cache cannot branch item candidates")
    repeater(count)
    base_mask = torch.tensor(
        [prefix.attention_mask], dtype=torch.long, device=device
    ).repeat(count, 1)
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
        kwargs["position_ids"] = positions.unsqueeze(0).expand(count, -1)
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = positions
    shared = getattr(branch, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = _repeat_batch_tree(shared, count)
    continuation = model(**kwargs)
    first_log_probs = torch.log_softmax(prefill_logits, dim=-1)
    later_log_probs = torch.log_softmax(
        _output_logits(continuation).detach().float(), dim=-1
    )
    sum_scores: list[float] = []
    mean_scores: list[float] = []
    for row_index, (_key, tokens) in enumerate(ordered):
        score = first_log_probs[tokens[0]]
        for offset, token in enumerate(tokens[1:]):
            score = score + later_log_probs[row_index, offset, token]
        value = float(score.detach().cpu())
        sum_scores.append(value)
        mean_scores.append(value / len(tokens))
    return {
        "predicted_occurrence_sum_logprob": int(np.argmax(sum_scores)) + 1,
        "predicted_occurrence_mean_logprob": int(np.argmax(mean_scores)) + 1,
        "sum_logprob_scores": sum_scores,
        "mean_logprob_scores": mean_scores,
        "candidate_token_counts": [len(tokens) for _key, tokens in ordered],
    }


def _extended_query_encoding(prefix: Any, query_tokens: Sequence[int]) -> Any:
    tokens = tuple(int(value) for value in query_tokens)
    if not tokens:
        raise ValueError("The attention query path is empty")
    ids = tuple(int(value) for value in prefix.input_ids) + tokens
    mask = tuple(int(value) for value in prefix.attention_mask) + (1,) * len(tokens)
    return replace(
        prefix,
        input_ids=ids,
        attention_mask=mask,
        query_position=len(ids) - 1,
    )


def _targeted_attention_metrics(
    model: Any,
    adapter: Any,
    full_encoding: Any,
    prequery_prefill: Any,
    *,
    query_position: int,
    patch_layer: int,
    replacement_state: torch.Tensor,
    targeted_bank: Mapping[str, Any],
    ordered_cities: Sequence[str],
    receiver_successor: int,
    donor_successor: int,
) -> dict[str, Any]:
    rows, key_starts, query_applications = _query_from_prefix_with_attentions(
        model,
        adapter,
        full_encoding,
        prequery_prefill,
        query_position=int(query_position),
        replacement_state=replacement_state,
        replacement_layer=int(patch_layer),
    )
    spans = {
        str(span.city).casefold(): span
        for span in full_encoding.prompt_record_spans
    }
    cities = tuple(str(value) for value in ordered_cities)
    if set(spans) != {value.casefold() for value in cities}:
        raise RuntimeError("Prompt record spans disagree with ordered gold cities")
    active_heads = tuple(
        (int(layer), int(head))
        for layer, head in targeted_bank["heads"]
        if int(layer) > int(patch_layer)
    )
    mass_by_ordinal = {index: 0.0 for index in range(1, len(cities) + 1)}
    for layer, head in active_heads:
        attention = rows[layer][head]
        key_start = int(key_starts[layer])
        key_end = key_start + int(attention.shape[-1])
        for ordinal, city in enumerate(cities, start=1):
            span = spans[city.casefold()]
            left = max(int(span.start), key_start)
            right = min(int(span.end), key_end)
            if right > left:
                mass_by_ordinal[ordinal] += float(
                    attention[left - key_start : right - key_start].sum().item()
                )
    receiver_mass = float(mass_by_ordinal[int(receiver_successor)])
    donor_mass = float(mass_by_ordinal[int(donor_successor)])
    top = int(max(mass_by_ordinal, key=lambda key: mass_by_ordinal[key]))
    total = float(sum(mass_by_ordinal.values()))
    expected = (
        None
        if total <= 0.0
        else float(
            sum(key * value for key, value in mass_by_ordinal.items()) / total
        )
    )
    receiver_share = receiver_mass / max(total, 1e-12)
    donor_share = donor_mass / max(total, 1e-12)
    return {
        "targeted_bank_size": int(targeted_bank["bank_size"]),
        "targeted_bank_sha256": str(targeted_bank["bank_sha256"]),
        "targeted_bank_downstream_head_count": len(active_heads),
        "targeted_bank_prompt_record_mass": total,
        "targeted_bank_top_source_ordinal": top,
        "targeted_bank_expected_source_ordinal": expected,
        "targeted_bank_source_masses": {
            str(key): float(value) for key, value in mass_by_ordinal.items()
        },
        "receiver_successor_attention_mass": receiver_mass,
        "donor_successor_attention_mass": donor_mass,
        "receiver_successor_attention_share": receiver_share,
        "donor_successor_attention_share": donor_share,
        "donor_minus_receiver_successor_attention_share": float(
            donor_share - receiver_share
        ),
        "donor_vs_receiver_attention_log_ratio": float(
            np.log(max(donor_mass, 1e-12))
            - np.log(max(receiver_mass, 1e-12))
        ),
        "donor_minus_receiver_successor_attention_mass": float(
            donor_mass - receiver_mass
        ),
        "attention_query_forward_applications": int(query_applications),
    }


def _condition_state_bank(
    receiver_state: torch.Tensor,
    donor_state: torch.Tensor,
    *,
    alternative_state: torch.Tensor | None,
    wrong_state: torch.Tensor,
    random_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    controls, audit = full_commit_specificity_condition_states(
        receiver_state,
        donor_state,
        shuffled_donor_state=wrong_state,
        random_seed=int(random_seed),
        random_replicates=1,
    )
    states = {
        "self_patch": torch.as_tensor(receiver_state).reshape(-1),
        "native_donor_branch": torch.as_tensor(donor_state).reshape(-1),
        "full_donor_patch": torch.as_tensor(donor_state).reshape(-1),
        "wrong_count_natural_patch": controls.pop(
            "shuffled_natural_donor_patch"
        ),
        **controls,
    }
    if alternative_state is not None:
        states["same_count_alternative_patch"] = torch.as_tensor(
            alternative_state
        ).reshape(-1)
    condition_audit = dict(audit["condition_audit"])
    condition_audit["wrong_count_natural_patch"] = condition_audit.pop(
        "shuffled_natural_donor_patch"
    )
    full_delta = torch.as_tensor(donor_state).float().reshape(-1) - torch.as_tensor(
        receiver_state
    ).float().reshape(-1)
    full_norm = float(torch.linalg.vector_norm(full_delta))
    condition_audit["self_patch"] = {
        "condition_patch_delta_norm": 0.0,
        "condition_full_donor_delta_norm_ratio": 0.0,
        "condition_full_donor_delta_cosine": 0.0,
        "condition_is_natural_commit_state": True,
    }
    condition_audit["full_donor_patch"] = {
        "condition_patch_delta_norm": full_norm,
        "condition_full_donor_delta_norm_ratio": 1.0,
        "condition_full_donor_delta_cosine": 1.0,
        "condition_is_natural_commit_state": True,
    }
    condition_audit["native_donor_branch"] = {
        **condition_audit["full_donor_patch"],
        "condition_is_donor_context_gate": True,
    }
    if alternative_state is not None:
        alternative_delta = (
            torch.as_tensor(alternative_state).float().reshape(-1)
            - torch.as_tensor(receiver_state).float().reshape(-1)
        )
        alternative_norm = float(torch.linalg.vector_norm(alternative_delta))
        condition_audit["same_count_alternative_patch"] = {
            "condition_patch_delta_norm": alternative_norm,
            "condition_full_donor_delta_norm_ratio": alternative_norm
            / max(full_norm, 1e-12),
            "condition_full_donor_delta_cosine": float(
                torch.dot(alternative_delta, full_delta)
                / max(alternative_norm * full_norm, 1e-12)
            ),
            "condition_is_natural_commit_state": True,
        }
    audit = {**audit, "condition_audit": condition_audit}
    return states, audit


def _wrong_valid_count(active: int, factor_count: int) -> int:
    if int(active) != int(factor_count):
        return int(factor_count)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--gold-count", type=int, required=True)
    parser.add_argument("--base-count", type=int, required=True)
    parser.add_argument("--source-occurrences", type=int, nargs="+", required=True)
    parser.add_argument("--donor-valid-counts", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS))
    parser.add_argument("--generation-conditions", nargs="+", default=[])
    parser.add_argument("--run-attention", action="store_true")
    parser.add_argument("--targeted-selection", type=Path)
    parser.add_argument("--targeted-routing", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--random-seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if int(args.prefill_chunk_size) <= 0:
        raise ValueError("prefill chunk size must be positive")

    conditions = tuple(str(value) for value in args.conditions)
    unknown = sorted(set(conditions) - set(ALL_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    generation_conditions = tuple(str(value) for value in args.generation_conditions)
    if not set(generation_conditions) <= set(conditions):
        raise ValueError("Generation conditions must be evaluated conditions")
    layers = tuple(sorted({int(value) for value in args.layers}))
    factor_count = len(args.source_occurrences)
    if not 1 <= factor_count <= 4:
        raise ValueError("The ledger factorial requires one to four source events")
    donors = tuple(sorted({int(value) for value in args.donor_valid_counts}))
    if any(not 1 <= value <= factor_count for value in donors):
        raise ValueError("Donor valid counts lie outside the factorial")
    if any(int(args.base_count) + value >= int(args.gold_count) for value in donors):
        raise ValueError("Every logical donor count must own a successor")

    rows = _read_rows(
        args.generations,
        gold_count=int(args.gold_count),
        seeds=args.seeds,
        max_seeds=args.max_seeds,
    )
    model, tokenizer, adapter = _experiment_model(args)
    if any(not 0 <= layer < int(adapter.num_layers) - 1 for layer in layers):
        raise ValueError("Every patch layer must leave a downstream block")
    targeted_bank = None
    if args.run_attention:
        if args.targeted_selection is None or args.targeted_routing is None:
            raise ValueError("Attention readout requires frozen selection and routing")
        targeted_bank = load_frozen_targeted_bank(
            args.targeted_selection,
            args.targeted_routing,
            model_label=str(args.model),
        )

    trials: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        encoding, registry = build_first_pass_tstar_answer_source_registry(
            row,
            tokenizer,
            candidate_counts=tuple(range(1, int(args.gold_count) + 1)),
        )
        if len(registry.trace_items) != int(args.gold_count):
            raise ValueError(
                f"Seed {row['seed']} has {len(registry.trace_items)} trace items"
            )
        boundary_map = {
            occurrence: select_post_item_boundary_position(
                encoding,
                registry,
                tokenizer,
                occurrence=occurrence,
            )[0]
            for occurrence in range(1, int(args.gold_count) + 1)
        }
        neutral, neutral_audit = _semantic_neutral_encoding(
            encoding,
            registry,
            tokenizer,
            random_seed=int(args.random_seed) + int(row["seed"]),
            protected_positions=tuple(boundary_map.values()),
        )
        variants, geometry = build_semantic_event_factorial(
            encoding,
            neutral,
            registry,
            boundary_map,
            receiver=int(args.base_count),
            source_occurrences=tuple(int(value) for value in args.source_occurrences),
        )
        selected = select_count_cells(
            variants,
            factor_count=factor_count,
            donor_valid_counts=donors,
        )
        by_bits = {
            tuple(int(value) for value in variant["marker_bits"]): variant
            for variant in variants
        }
        receiver_variant = by_bits[canonical_marker_bits(factor_count, 0)]
        site = int(geometry["inserted_slots"][-1]["event_boundary"])
        receiver_prefix = prefix_through_boundary(
            receiver_variant["encoding"], site
        )
        site_tokens = {
            int(variant["encoding"].input_ids[site]) for variant in variants
        }
        if len(site_tokens) != 1:
            raise RuntimeError("Factorial branches changed the commit surface token")
        prefix_lengths = {
            int(prefix_through_boundary(variant["encoding"], site).sequence_length)
            for variant in variants
        }
        if prefix_lengths != {site + 1}:
            raise RuntimeError("Factorial branches changed the commit position")

        needed_bits = {canonical_marker_bits(factor_count, 0)}
        for active in donors:
            primary = selected[active]["primary"]
            alternative = selected[active]["alternative"]
            needed_bits.add(tuple(int(value) for value in primary["marker_bits"]))
            if alternative is not None:
                needed_bits.add(
                    tuple(int(value) for value in alternative["marker_bits"])
                )
            wrong = _wrong_valid_count(active, factor_count)
            needed_bits.add(canonical_marker_bits(factor_count, wrong))
        state_bank: dict[tuple[int, ...], dict[int, torch.Tensor]] = {}
        for bits in sorted(needed_bits):
            active_prefix = prefix_through_boundary(by_bits[bits]["encoding"], site)
            captured = _chunked_capture_site_states(
                model,
                adapter,
                active_prefix,
                site=site,
                layers=layers,
                chunk_size=int(args.prefill_chunk_size),
            )
            state_bank[bits] = {
                layer: captured[layer] for layer in layers
            }

        candidates = native_item_candidates(encoding, registry.trace_items)
        ordered_cities = tuple(str(value["city"]) for value in gold_records(row))
        if len(ordered_cities) != int(args.gold_count):
            raise RuntimeError("Gold city count disagrees with N")
        receiver_successor = int(args.base_count) + 1
        query_tokens: tuple[int, ...] = ()
        full_query_encoding = None
        prequery_encoding = None
        query_position = None
        if args.run_attention:
            # The first downstream attention after transplant occurs at the
            # shared commit token itself.  Query that exact site instead of a
            # teacher-forced successor token, so surface and absolute position
            # remain fully matched across receiver and donor conditions.
            full_query_encoding = receiver_prefix
            query_position = site
            prequery_encoding = prefix_through_boundary(
                full_query_encoding, query_position - 1
            )

        geometry_rows.append(
            {
                "request_id": str(row["request_id"]),
                "seed": int(row["seed"]),
                "base_count": int(args.base_count),
                "factor_count": factor_count,
                "source_occurrences": [int(value) for value in args.source_occurrences],
                "shared_commit_position": site,
                "shared_commit_token_id": next(iter(site_tokens)),
                "shared_commit_token_text": tokenizer.decode(
                    [next(iter(site_tokens))],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "query_position": query_position,
                "query_teacher_forced_token_count": len(query_tokens),
                "query_teacher_forced_text": tokenizer.decode(
                    list(query_tokens),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "all_cells_equal_length": bool(geometry["all_cells_equal_length"]),
                "all_cells_equal_attention_mask": bool(
                    geometry["all_cells_equal_attention_mask"]
                ),
                "only_event_semantic_token_ids_vary": bool(
                    geometry["only_event_semantic_token_ids_vary"]
                ),
                **neutral_audit,
            }
        )

        for active in donors:
            donor_count = int(args.base_count) + active
            donor_successor = donor_count + 1
            primary_variant = selected[active]["primary"]
            alternative_variant = selected[active]["alternative"]
            primary_bits = tuple(
                int(value) for value in primary_variant["marker_bits"]
            )
            donor_prefix = prefix_through_boundary(
                primary_variant["encoding"], site
            )
            alternative_bits = (
                None
                if alternative_variant is None
                else tuple(
                    int(value) for value in alternative_variant["marker_bits"]
                )
            )
            wrong_active = _wrong_valid_count(active, factor_count)
            wrong_bits = canonical_marker_bits(factor_count, wrong_active)
            wrong_count = int(args.base_count) + wrong_active
            for layer in layers:
                states, state_audit = _condition_state_bank(
                    state_bank[canonical_marker_bits(factor_count, 0)][layer],
                    state_bank[primary_bits][layer],
                    alternative_state=(
                        None
                        if alternative_bits is None
                        else state_bank[alternative_bits][layer]
                    ),
                    wrong_state=state_bank[wrong_bits][layer],
                    random_seed=(
                        int(args.random_seed)
                        + int(row["seed"]) * 1009
                        + active * 53
                        + layer
                    ),
                )
                for condition in conditions:
                    if condition not in states:
                        continue
                    state = states[condition]
                    condition_prefix = (
                        donor_prefix
                        if condition == "native_donor_branch"
                        else receiver_prefix
                    )
                    prefill, applications, realized_norm = (
                        _chunked_prefill_with_site_replacement(
                            model,
                            adapter,
                            condition_prefix,
                            layer=layer,
                            site=site,
                            state=state,
                            chunk_size=int(args.prefill_chunk_size),
                        )
                    )
                    generation_outcome: dict[str, Any] = {}
                    if condition in generation_conditions:
                        # Greedy decoding mutates DynamicCache in place, so it
                        # receives an independent prefill.  ``prefill`` remains
                        # pristine for the native-candidate likelihood assay.
                        generation_prefill, _generation_apps, _generation_norm = (
                            _chunked_prefill_with_site_replacement(
                                model,
                                adapter,
                                condition_prefix,
                                layer=layer,
                                site=site,
                                state=state,
                                chunk_size=int(args.prefill_chunk_size),
                            )
                        )
                        completion = generate_answer_completion_from_prefill(
                            model,
                            tokenizer,
                            condition_prefix,
                            generation_prefill,
                            max_new_tokens=int(args.max_new_tokens),
                        )
                        generated = generated_bullet_city_ordinals(
                            str(completion["completion_text"]), ordered_cities
                        )
                        first = generated["first_generated_known_city_ordinal"]
                        generation_outcome = {
                            "local_completion_text": str(
                                completion["completion_text"]
                            ),
                            "local_generation_truncated": bool(
                                completion["generation_truncated"]
                            ),
                            "local_generated_token_count": int(
                                completion["generated_token_count"]
                            ),
                            "local_stopped_on_eos": bool(
                                completion["stopped_on_eos"]
                            ),
                            **generated,
                            "greedy_donor_successor_adoption": bool(
                                first == donor_successor
                            ),
                            "greedy_receiver_successor_retention": bool(
                                first == receiver_successor
                            ),
                            "expected_receiver_remaining_events": int(
                                args.gold_count - args.base_count
                            ),
                            "expected_donor_remaining_events": int(
                                args.gold_count - donor_count
                            ),
                            "expected_remaining_event_shift": -active,
                        }
                    scored = _score_candidates(
                        model, condition_prefix, prefill, candidates
                    )
                    sum_scores = scored["sum_logprob_scores"]
                    mean_scores = scored["mean_logprob_scores"]
                    outcome: dict[str, Any] = {
                        **scored,
                        "donor_vs_receiver_sum_logodds": donor_receiver_logodds(
                            sum_scores,
                            donor_successor=donor_successor,
                            receiver_successor=receiver_successor,
                        ),
                        "donor_vs_receiver_mean_logodds": donor_receiver_logodds(
                            mean_scores,
                            donor_successor=donor_successor,
                            receiver_successor=receiver_successor,
                        ),
                        "donor_successor_sum_argmax": bool(
                            int(scored["predicted_occurrence_sum_logprob"])
                            == donor_successor
                        ),
                        "donor_successor_mean_argmax": bool(
                            int(scored["predicted_occurrence_mean_logprob"])
                            == donor_successor
                        ),
                        "receiver_successor_sum_argmax": bool(
                            int(scored["predicted_occurrence_sum_logprob"])
                            == receiver_successor
                        ),
                    }
                    if args.run_attention:
                        assert query_position is not None
                        condition_query_encoding = condition_prefix
                        condition_prequery_encoding = prefix_through_boundary(
                            condition_query_encoding, query_position - 1
                        )
                        query_prefill = _chunked_prefill(
                            model,
                            adapter,
                            condition_prequery_encoding,
                            chunk_size=int(args.prefill_chunk_size),
                        )
                        outcome.update(
                            _targeted_attention_metrics(
                                model,
                                adapter,
                                condition_query_encoding,
                                query_prefill,
                                query_position=query_position,
                                patch_layer=layer,
                                replacement_state=state,
                                targeted_bank=targeted_bank,
                                ordered_cities=ordered_cities,
                                receiver_successor=receiver_successor,
                                donor_successor=donor_successor,
                            )
                        )
                        outcome["attention_prefix_patch_applications"] = 0
                    outcome.update(generation_outcome)
                    trials.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "model_label": str(args.model),
                            "request_id": str(row["request_id"]),
                            "seed": int(row["seed"]),
                            "gold_count": int(args.gold_count),
                            "condition": condition,
                            "condition_uses_native_donor_context": bool(
                                condition == "native_donor_branch"
                            ),
                            "layer": layer,
                            "base_count_j": int(args.base_count),
                            "donor_count_k": donor_count,
                            "donor_valid_event_count": active,
                            "wrong_count": wrong_count,
                            "receiver_successor_occurrence": receiver_successor,
                            "donor_successor_occurrence": donor_successor,
                            "receiver_event_bits": list(
                                canonical_marker_bits(factor_count, 0)
                            ),
                            "donor_event_bits": list(primary_bits),
                            "same_count_alternative_event_bits": (
                                None
                                if alternative_bits is None
                                else list(alternative_bits)
                            ),
                            "wrong_count_event_bits": list(wrong_bits),
                            "shared_commit_position": site,
                            "shared_commit_surface_token_identical": True,
                            "shared_absolute_position_identical": True,
                            "only_prior_event_semantic_tokens_differ_between_branches": True,
                            "diagnostic_total_suffix_used": False,
                            "visible_item_indices_used": False,
                            "count_subspace_used": False,
                            "full_commit_vector_used": True,
                            "patch_applications": int(applications),
                            "patch_realized_l2_norm": float(realized_norm),
                            **state_audit["condition_audit"][condition],
                            **outcome,
                        }
                    )
        print(
            f"[same-site] {row_index}/{len(rows)} seed={row['seed']} "
            f"N={args.gold_count} j={args.base_count} layers={list(layers)}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output / "trials.jsonl", trials)
    _atomic_jsonl(args.output / "geometry_audit.jsonl", geometry_rows)
    summary_rows: list[dict[str, Any]] = []
    keys = sorted(
        {
            (int(row["layer"]), int(row["donor_valid_event_count"]), str(row["condition"]))
            for row in trials
        }
    )
    for layer, active, condition in keys:
        group = [
            row
            for row in trials
            if int(row["layer"]) == layer
            and int(row["donor_valid_event_count"]) == active
            and str(row["condition"]) == condition
        ]
        summary_rows.append(
            {
                "layer": layer,
                "donor_valid_event_count": active,
                "condition": condition,
                "seed_count": len({int(row["seed"]) for row in group}),
                "mean_donor_vs_receiver_sum_logodds": float(
                    np.mean([row["donor_vs_receiver_sum_logodds"] for row in group])
                ),
                "donor_successor_sum_argmax_rate": float(
                    np.mean([bool(row["donor_successor_sum_argmax"]) for row in group])
                ),
                "mean_donor_minus_receiver_attention": (
                    None
                    if not args.run_attention
                    else float(
                        np.mean(
                            [
                                row["donor_minus_receiver_successor_attention_mass"]
                                for row in group
                            ]
                        )
                    )
                ),
                "mean_donor_vs_receiver_attention_log_ratio": (
                    None
                    if not args.run_attention
                    else float(
                        np.mean(
                            [
                                row["donor_vs_receiver_attention_log_ratio"]
                                for row in group
                            ]
                        )
                    )
                ),
                "mean_targeted_expected_source_ordinal": (
                    None
                    if not args.run_attention
                    else float(
                        np.mean(
                            [
                                row["targeted_bank_expected_source_ordinal"]
                                for row in group
                            ]
                        )
                    )
                ),
                "greedy_donor_adoption_rate": (
                    None
                    if condition not in generation_conditions
                    else float(
                        np.mean(
                            [bool(row["greedy_donor_successor_adoption"]) for row in group]
                        )
                    )
                ),
                "mean_generated_known_city_bullets": (
                    None
                    if condition not in generation_conditions
                    else float(
                        np.mean(
                            [row["generated_known_city_bullet_count"] for row in group]
                        )
                    )
                ),
                "mean_generated_known_city_mentions": (
                    None
                    if condition not in generation_conditions
                    else float(
                        np.mean(
                            [row["generated_known_city_count_any_surface"] for row in group]
                        )
                    )
                ),
                "mean_generated_token_count": (
                    None
                    if condition not in generation_conditions
                    else float(
                        np.mean([row["local_generated_token_count"] for row in group])
                    )
                ),
            }
        )
    _atomic_jsonl(args.output / "summary.jsonl", summary_rows)
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "model_label": str(args.model),
            "gold_count": int(args.gold_count),
            "base_count_j": int(args.base_count),
            "factor_count": factor_count,
            "source_occurrences": [int(value) for value in args.source_occurrences],
            "donor_valid_counts": list(donors),
            "layers": list(layers),
            "conditions": list(conditions),
            "generation_conditions": list(generation_conditions),
            "seed_count": len(rows),
            "trial_count": len(trials),
            "run_attention": bool(args.run_attention),
            "targeted_bank": targeted_bank,
            "same_surface_commit_token": True,
            "same_absolute_commit_position": True,
            "equal_length_factorial": True,
            "only_prior_event_semantic_tokens_vary": True,
            "selection_uses_outcomes": False,
            "claim_scope_if_positive": (
                "same-site distributed progress state causally controls successor routing"
            ),
        },
    )
    print(json.dumps(summary_rows, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
