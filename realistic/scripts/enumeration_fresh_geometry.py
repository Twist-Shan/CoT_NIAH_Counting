"""Outcome-independent geometry for the fresh cross-mode alignment supplement.

Imports of model packages are deferred so the mask and cohort contracts can be
tested without CUDA or Transformers. Each worker handles exactly one mode;
the legacy V6 adapters must never leak into a Thinking worker.
"""
from __future__ import annotations

import hashlib
import json


def json_sha(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def enumeration_format_eligible(parsed):
    """Require the registered item grammar/order, independently of final Total."""
    return all(bool(parsed.get(key)) for key in (
        "enumeration_format_compliant", "exact_ordered_gold_pairs",
        "marker_kind_compliant", "parser_forward_one_to_one",
    ))


def common_eligible_seeds(ledgers, candidates, *, field, quota):
    """Return a fixed-order intersection; never look at an intervention score."""
    if not ledgers or quota <= 0 or len(candidates) != len(set(candidates)):
        raise ValueError("Expected nonempty ledgers, positive quota, unique candidates")
    by_mode = []
    for ledger in ledgers:
        n10 = [row for row in ledger if int(row["gold_count"]) == 10]
        seeds = [int(row["seed"]) for row in n10]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate N=10 source seed")
        lookup = {int(row["seed"]): row for row in n10}
        missing = set(candidates) - lookup.keys()
        if missing:
            raise ValueError(f"Unaccounted candidate seeds: {sorted(missing)}")
        by_mode.append(lookup)
    return [seed for seed in candidates
            if all(bool(table[seed][field]) for table in by_mode)][:quota]


def answer_blank_positions(*, prompt_count, query, sequence_length, record_spans):
    """Five Read masks, including traces with no parsed items.

The old combined Read/Retrieve registry requires a parsed item. Read's five
source regions need only a prompt boundary and a legal answer query. Excluding
item-free answers would add an unintended format filter to the unfiltered pool.
"""
    if not 1 < prompt_count <= query == sequence_length - 1:
        raise ValueError("Invalid prompt/query boundaries")
    records = set()
    for start, end in record_spans:
        if not 1 <= start < end <= prompt_count:
            raise ValueError("A prompt record crosses BOS or prompt boundary")
        records.update(range(start, end))
    if not records:
        raise ValueError("No prompt records")
    prompt = set(range(1, prompt_count))
    trace = set(range(prompt_count, query))
    masks = {"clean": (), "prompt_all_blank": tuple(sorted(prompt)),
             "prompt_records_blank": tuple(sorted(records)),
             "trace_all_blank": tuple(sorted(trace)),
             "prompt_and_trace_blank": tuple(sorted(prompt | trace))}
    assert all(all(0 < p < query for p in values) for values in masks.values())
    return masks


def build_read_geometry(row, tokenizer, *, mode):
    if mode == "thinking":
        from realistic_niah_v5.encoding import build_native_trace_encoding as build
    else:
        from realistic_niah_v6.encoding import build_structured_trace_encoding as build
    encoding = build(row, tokenizer, site_id="answer_query_v3", candidate_counts=tuple(range(1, 11)))
    spans = [(int(span.start), int(span.end)) for span in encoding.prompt_record_spans]
    masks = answer_blank_positions(prompt_count=encoding.prompt_token_count,
                                   query=encoding.query_position,
                                   sequence_length=encoding.sequence_length,
                                   record_spans=spans)
    audit = {"query_position": encoding.query_position,
             "sequence_length": encoding.sequence_length,
             "prompt_token_count": encoding.prompt_token_count,
             "prompt_record_spans": spans,
             "input_ids_sha256": json_sha(list(encoding.input_ids)),
             "attention_mask_sha256": json_sha(list(encoding.attention_mask)),
             "selected_site": dict(encoding.selected_site),
             "conditions": {name: {"positions_sha256": json_sha(list(positions)),
                                    "token_count": len(positions)}
                            for name, positions in masks.items()},
             "requires_parsed_trace_item": False}
    return encoding, masks, audit


def adapt_update_row(row, *, model, mode):
    """Attach the legacy numerical kernel's metadata without changing outputs."""
    value = dict(row)
    if mode != "thinking":
        from realistic_niah_v6.parsing import parse_trace_record
        parsed = parse_trace_record(row)
        eligible = enumeration_format_eligible(parsed)
        grammar = {"enumeration_index": "structural_explicit_rank_before_city",
                   "enumeration_bullet": "structural_invariant_bullet"}[mode]
        population = "indexed_positive_control"
        field = "primary_eligible_indexed_positive_control"
        audit = {field: eligible, "grammar_class": grammar,
                 "selection_used_final_answer": False,
                 "selection_basis": "registered_item_grammar_and_exact_passage_order"}
        value["v6_structured_enumeration_format_audit"] = audit
        value["indexed_progress_control_format_audit"] = {
            "grammar_class": grammar, "v6_compatibility_alias": True,
            "selection_authority": False}
        value["v6_structured_enumeration_cohort"] = {
            "selection_population": population, "prompt_mode": mode,
            "selection_used_final_answer": False}
        cohort_mode = "indexed_positive_control"
    elif model == "Qwen3-8B":
        from scripts.scan_realistic_niah_v5_frozen_prompt_noindex_n3 import format_audit
        audit = format_audit(row)
        field = "primary_eligible_prefix_clean"
        population = "first_pass_noindex_enumeration"
        eligible = bool(audit[field])
        value["fresh_native_format_audit"] = audit
        value["fresh_native_cohort"] = {
            "selection_population": population, "fixed_count": int(row["gold_count"]),
            "split": row["split"], "selection_used_final_answer": False}
        cohort_mode = "natural_noindex"
    else:
        from scripts.run_realistic_niah_v5_gemma_prompt_conditioned_noindex import (
            audit_prompt_conditioned_noindex_row, AUDIT_KEY, COHORT_KEY,
        )
        audit = audit_prompt_conditioned_noindex_row(row)
        field = "primary_eligible_prompt_conditioned_noindex"
        population = "gemma_prompt_conditioned_noindex_found_v3"
        eligible = bool(audit[field])
        value[AUDIT_KEY] = audit
        value[COHORT_KEY] = {
            "selection_population": population, "fixed_count": int(row["gold_count"]),
            "split": row["split"], "selection_used_final_answer": False}
        cohort_mode = "prompt_conditioned_noindex"
    if mode == "thinking":
        observed = [str(item["city"]).casefold() for item in audit["first_occurrences"]]
        expected = [str(item["city"]).casefold() for item in row["gold_records"]]
        eligible = eligible and observed == expected
    metadata = {"format_eligible": eligible, "cohort_mode": cohort_mode,
                "selection_population": population, "eligibility_field": field,
                "selection_used_final_answer": False, "intervention_outcomes_accessed": False}
    return value, metadata


def build_update_geometry(row, tokenizer, *, metadata, cfg):
    from realistic_niah_v5.count_stream import build_answer_source_registry
    from realistic_niah_v5.counting_mechanism_transfer import build_first_pass_tstar_answer_source_registry
    from realistic_niah_v5.natural_aligned_progress import (
        post_item_sites_at_tail_offset, resolve_natural_patch_span,
        align_natural_donor_prompt,
    )
    from scripts.run_realistic_niah_v5_cross_seed_counter_recurrence import prefix_through_boundary
    if not metadata["format_eligible"]:
        raise ValueError("Baseline does not meet the frozen item-format/order rule")
    if metadata["cohort_mode"] == "indexed_positive_control":
        encoding, registry = build_answer_source_registry(row, tokenizer)
    else:
        encoding, registry = build_first_pass_tstar_answer_source_registry(
            row, tokenizer, selection_population=metadata["selection_population"],
            eligibility_field=metadata["eligibility_field"])
    if len(registry.trace_items) != 10:
        raise ValueError("N=10 update needs ten observed items")
    results = []
    for k in cfg["donor_k"]:
        for direction in cfg["directions"]:
            j = k - 1 if direction == "forward" else k + 1
            rsite, dsite, site = post_item_sites_at_tail_offset(
                encoding, registry.trace_items, receiver_occurrence=j,
                donor_occurrence=k, tokenizer=tokenizer, tail_offset=0)
            aligned, align = align_natural_donor_prompt(
                encoding, registry, receiver_site=min(rsite, dsite),
                donor_site=max(rsite, dsite), tokenizer=tokenizer,
                require_surface_match=bool(site["surface_token_matched"]))
            shared = min(rsite, dsite)
            left = prefix_through_boundary(encoding, shared)
            right = prefix_through_boundary(aligned, shared)
            assert left.sequence_length == right.sequence_length
            for scope in cfg["scopes"]:
                width = 4 if scope == "four_token_tail" else 1
                patch = resolve_natural_patch_span(
                    registry.trace_items, receiver_occurrence=j, donor_occurrence=k,
                    receiver_site=rsite, donor_site=dsite,
                    patch_scope="item_span" if scope == "item_span" else "fixed_suffix",
                    patch_width=width)
                results.append({"donor_occurrence": k, "receiver_occurrence": j,
                                "direction": direction, "scope": scope,
                                "shared_absolute_endpoint": shared,
                                "aligned_prompt_role": "donor" if dsite > rsite else "receiver",
                                "unaltered_prefix_sha256": json_sha(list(left.input_ids)),
                                "aligned_prefix_sha256": json_sha(list(right.input_ids)),
                                **site, **align, **patch})
    return {"answer_registry": registry.to_dict(), "trials": results}
