#!/usr/bin/env python3
"""Discovery-only post-t-star delay scan followed by a frozen confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_tstar_forced_total import (  # noqa: E402
    FORCED_SUFFIX,
    _eos_ids,
    _indexed,
    _validate_sources,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    extract_immediate_integer,
    extract_total,
    read_json,
    read_jsonl,
    sha256_file,
)
from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.tstar_prefix import sha256_json  # noqa: E402
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    explicit_count_cues,
)


SCHEMA = "realistic_niah_v5_tstar_post_evidence_delay_scan_v1"
DELAYS = (1, 2, 4, 8)
PLAN_FILENAME = "frozen_tstar_delay_scan_plan_v1.json"
DISCOVERY_ROWS_FILENAME = "discovery_tstar_delay_scan_rows_v1.jsonl"
DISCOVERY_CSV_FILENAME = "discovery_tstar_delay_scan_summary_v1.csv"
DISCOVERY_SUMMARY_FILENAME = "discovery_tstar_delay_scan_summary_v1.json"
BOUNDARY_FILENAME = "frozen_tstar_boundary_choice_v1.json"
CONFIRMATION_ROWS_FILENAME = "confirmation_frozen_tstar_boundary_rows_v1.jsonl"
CONFIRMATION_CSV_FILENAME = "confirmation_frozen_tstar_boundary_summary_v1.csv"
MANIFEST_FILENAME = "tstar_delay_scan_manifest_v1.json"
BOUNDARY_V2_FILENAME = "frozen_tstar_boundary_choice_v2.json"
CONFIRMATION_V2_ROWS_FILENAME = "confirmation_frozen_tstar_boundary_rows_v2.jsonl"
CONFIRMATION_V2_CSV_FILENAME = "confirmation_frozen_tstar_boundary_summary_v2.csv"
MANIFEST_V2_FILENAME = "tstar_delay_scan_manifest_v2.json"
MANIFEST_V3_FILENAME = "tstar_delay_scan_manifest_v3.json"


def _plan(
    args: argparse.Namespace,
    *,
    source_manifest: Mapping[str, Any],
    context_manifest: Mapping[str, Any],
    fixed_count: int,
    seeds: list[int],
) -> dict[str, Any]:
    spec = resolve_model_spec(str(args.model))
    return {
        "schema_version": SCHEMA,
        "status": "FROZEN_BEFORE_DISCOVERY_DELAY_OUTPUTS",
        "model_label": str(args.model),
        "model_id": str(spec.model_id),
        "model_revision": spec.revision,
        "fixed_count": int(fixed_count),
        "discovery_seeds": [int(value) for value in source_manifest["discovery_seeds"]],
        "confirmation_seeds": [
            int(value) for value in source_manifest["confirmation_seeds"]
        ],
        "all_seeds": [int(value) for value in seeds],
        "candidate_post_tstar_delays_tokens": list(DELAYS),
        "forced_suffix": FORCED_SUFFIX,
        "decoding": {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        },
        "boundary_eligibility": (
            "all 20 discovery extended prefixes contain no explicit count/index cue "
            "under the prefix-clean grammar and no repeated gold-city mention"
        ),
        "boundary_selection_rule": (
            "among eligible delays, maximize discovery forced-total gold accuracy; "
            "break ties by choosing the smallest token delay"
        ),
        "confirmation_policy": (
            "generate only the single frozen delay; confirmation delay outcomes are "
            "inaccessible to boundary selection"
        ),
        "source_selected_rows": str(args.selected_rows.resolve()),
        "source_selected_rows_sha256": sha256_file(args.selected_rows),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "tstar_contexts": str(args.contexts.resolve()),
        "tstar_contexts_sha256": sha256_file(args.contexts),
        "tstar_context_manifest": str(args.context_manifest.resolve()),
        "tstar_context_manifest_sha256": sha256_file(args.context_manifest),
        "device_map": str(args.device_map),
        "torch_dtype": str(args.torch_dtype),
        "attention_backend": str(args.attention_backend),
    }


def _gold_city_hits(text: str, source: Mapping[str, Any]) -> list[str]:
    hits: list[str] = []
    for record in source.get("gold_records", ()):
        city = str(record.get("city", ""))
        if city and re.search(rf"(?<!\w){re.escape(city)}(?!\w)", text, re.IGNORECASE):
            hits.append(city)
    return hits


def _condition_input(
    source: Mapping[str, Any],
    context: Mapping[str, Any],
    tokenizer: Any,
    *,
    delay: int,
    closure_ids: tuple[int, ...],
) -> dict[str, Any]:
    output_ids = tuple(int(value) for value in source["output_token_ids"])
    base_end = int(context["output_token_end"])
    delayed_end = base_end + int(delay)
    if not 0 < base_end < delayed_end < len(output_ids):
        raise ValueError("Post-t-star delay exceeds the archived nonterminal output")
    eos_ids = {int(value) for value in source.get("generation_eos_token_ids", ())}
    if any(value in eos_ids for value in output_ids[base_end:delayed_end]):
        raise ValueError("Post-t-star delay crossed an archived EOS token")
    extended_output_ids = output_ids[:delayed_end]
    extended_prefix = tokenizer.decode(
        list(extended_output_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    raw = str(source["raw_output_text"])
    if not raw.startswith(extended_prefix):
        raise ValueError("Delayed output tokens are not an exact archived text prefix")
    base_prefix = str(context["raw_prefix_text"])
    if not extended_prefix.startswith(base_prefix):
        raise ValueError("Delayed prefix does not extend the frozen t-star prefix")
    delay_text = extended_prefix[len(base_prefix) :]
    audit_key = f"noindex_n{int(source['gold_count'])}_format_audit"
    audit = source[audit_key]
    reasoning_start = int(audit["reasoning_start_char"])
    cities = [str(record["city"]) for record in source.get("gold_records", ())]
    cues = explicit_count_cues(
        extended_prefix[reasoning_start:],
        cities=cities,
        max_index=int(source["gold_count"]),
        offset=reasoning_start,
    )
    gold_hits = _gold_city_hits(delay_text, source)
    prompt_ids = tuple(int(value) for value in source["input_ids"])
    forced_input_ids = prompt_ids + extended_output_ids + closure_ids
    prompt_mask = tuple(int(value) for value in source.get("attention_mask", ()))
    if len(prompt_mask) != len(prompt_ids):
        raise ValueError("Source prompt attention mask length changed")
    forced_mask = prompt_mask + (1,) * (len(extended_output_ids) + len(closure_ids))
    return {
        "delay_tokens": int(delay),
        "delayed_output_token_end": delayed_end,
        "delay_output_token_ids": list(output_ids[base_end:delayed_end]),
        "delay_text": delay_text,
        "extended_prefix_text": extended_prefix,
        "explicit_count_cues": cues,
        "post_tstar_gold_city_mentions": gold_hits,
        "site_clean": not cues and not gold_hits,
        "forced_input_ids": forced_input_ids,
        "forced_attention_mask": forced_mask,
    }


def _generate_condition(
    *,
    source: Mapping[str, Any],
    context: Mapping[str, Any],
    condition: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    plan_hash: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    device = model.get_input_embeddings().weight.device
    eos_ids = _eos_ids(model, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = eos_ids[0]
    forced_input_ids = tuple(int(value) for value in condition["forced_input_ids"])
    forced_mask = tuple(int(value) for value in condition["forced_attention_mask"])
    seed = int(source["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    kwargs: dict[str, Any] = {
        "input_ids": torch.tensor([forced_input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([forced_mask], dtype=torch.long, device=device),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)
    with torch.inference_mode():
        generated = model.generate(**kwargs)
    sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
    continuation_ids = tuple(
        int(value)
        for value in sequences[0, len(forced_input_ids) :].detach().cpu().tolist()
    )
    continuation_raw = tokenizer.decode(
        list(continuation_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    continuation_clean = tokenizer.decode(
        list(continuation_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    forced_total = extract_immediate_integer(continuation_clean)
    gold_total = int(source["gold_count"])
    source_total = extract_total(str(source.get("clean_output_text", "")))
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "plan_sha256": plan_hash,
        "seed": seed,
        "split": str(context["split"]),
        "fixed_count": gold_total,
        "request_id": str(source.get("request_id", "")),
        "source_row_sha256": str(context["source_row_sha256"]),
        "tstar_context_row_sha256": sha256_json(context),
        "t_star_char": int(context["t_star_char"]),
        "tstar_output_token_end": int(context["output_token_end"]),
        "delay_tokens": int(condition["delay_tokens"]),
        "delayed_output_token_end": int(condition["delayed_output_token_end"]),
        "delay_output_token_ids": list(condition["delay_output_token_ids"]),
        "delay_text": str(condition["delay_text"]),
        "extended_prefix_text": str(condition["extended_prefix_text"]),
        "explicit_count_cues": list(condition["explicit_count_cues"]),
        "post_tstar_gold_city_mentions": list(
            condition["post_tstar_gold_city_mentions"]
        ),
        "site_clean": bool(condition["site_clean"]),
        "forced_suffix": FORCED_SUFFIX,
        "forced_input_ids": list(forced_input_ids),
        "forced_attention_mask": list(forced_mask),
        "forced_input_ids_sha256": sha256_json(forced_input_ids),
        "forced_continuation_token_ids": list(continuation_ids),
        "forced_continuation_text_raw": continuation_raw,
        "forced_continuation_text_clean": continuation_clean,
        "forced_total": forced_total,
        "source_total": source_total,
        "gold_total": gold_total,
        "forced_total_gold_correct": forced_total == gold_total,
        "forced_total_matches_source": (
            forced_total is not None and forced_total == source_total
        ),
        "immediate_integer_only": forced_total is not None,
        "stopped_on_eos": bool(
            continuation_ids and continuation_ids[-1] in set(eos_ids)
        ),
        "interpretation": (
            "standardized forced-stop readout after a frozen post-evidence token delay"
        ),
    }


def _summaries(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for delay in DELAYS:
        subset = [row for row in rows if int(row["delay_tokens"]) == delay]
        output.append(
            {
                "delay_tokens": delay,
                "row_count": len(subset),
                "site_clean_n": sum(bool(row["site_clean"]) for row in subset),
                "forced_total_parsed_n": sum(
                    row.get("forced_total") is not None for row in subset
                ),
                "forced_total_gold_correct_n": sum(
                    bool(row["forced_total_gold_correct"]) for row in subset
                ),
                "forced_total_matches_source_n": sum(
                    bool(row["forced_total_matches_source"]) for row in subset
                ),
                "immediate_integer_only_n": sum(
                    bool(row["immediate_integer_only"]) for row in subset
                ),
                "stopped_on_eos_n": sum(
                    bool(row["stopped_on_eos"]) for row in subset
                ),
                "boundary_eligible": bool(subset) and all(
                    bool(row["site_clean"]) for row in subset
                ),
            }
        )
    return output


def choose_boundary_summary(
    summaries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [dict(row) for row in summaries if bool(row["boundary_eligible"])]
    if not eligible:
        raise RuntimeError("No delay is cue-free for the full discovery cohort")
    return min(
        eligible,
        key=lambda row: (
            -int(row["forced_total_gold_correct_n"]),
            int(row["delay_tokens"]),
        ),
    )


def _explicit_cue_only_summaries(
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Version-2 eligibility: repeated known evidence is not an index leak."""

    output: list[dict[str, Any]] = []
    for delay in DELAYS:
        subset = [row for row in rows if int(row["delay_tokens"]) == delay]
        cue_free_n = sum(not row.get("explicit_count_cues") for row in subset)
        output.append(
            {
                "delay_tokens": delay,
                "row_count": len(subset),
                "explicit_cue_free_n": cue_free_n,
                "repeated_gold_city_n": sum(
                    bool(row.get("post_tstar_gold_city_mentions")) for row in subset
                ),
                "forced_total_gold_correct_n": sum(
                    bool(row["forced_total_gold_correct"]) for row in subset
                ),
                "forced_total_matches_source_n": sum(
                    bool(row["forced_total_matches_source"]) for row in subset
                ),
                "boundary_eligible": bool(subset)
                and cue_free_n == len(subset),
            }
        )
    return output


def _csv_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(row["seed"]),
            "split": str(row["split"]),
            "delay_tokens": int(row["delay_tokens"]),
            "delay_text": str(row["delay_text"]),
            "site_clean": bool(row["site_clean"]),
            "gold_total": int(row["gold_total"]),
            "source_total": row["source_total"],
            "forced_total": row["forced_total"],
            "forced_total_gold_correct": bool(row["forced_total_gold_correct"]),
            "forced_total_matches_source": bool(row["forced_total_matches_source"]),
            "immediate_integer_only": bool(row["immediate_integer_only"]),
            "stopped_on_eos": bool(row["stopped_on_eos"]),
            "forced_continuation_text_clean": str(
                row["forced_continuation_text_clean"]
            ),
        }
        for row in rows
    ]


def _load_common(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = read_json(args.source_manifest)
    context_manifest = read_json(args.context_manifest)
    seeds, fixed_count = _validate_sources(
        selected_path=args.selected_rows,
        selected_manifest=source_manifest,
        context_path=args.contexts,
        context_manifest=context_manifest,
    )
    source_by_seed = _indexed(read_jsonl(args.selected_rows), label="source")
    context_by_seed = _indexed(read_jsonl(args.contexts), label="context")
    if set(source_by_seed) != set(seeds) or set(context_by_seed) != set(seeds):
        raise ValueError("Delay-scan inputs do not contain the frozen seed set")
    plan = _plan(
        args,
        source_manifest=source_manifest,
        context_manifest=context_manifest,
        fixed_count=fixed_count,
        seeds=seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / PLAN_FILENAME
    atomic_json(plan_path, plan)
    return {
        "source_manifest": source_manifest,
        "context_manifest": context_manifest,
        "seeds": seeds,
        "fixed_count": fixed_count,
        "source_by_seed": source_by_seed,
        "context_by_seed": context_by_seed,
        "plan_path": plan_path,
        "plan_hash": sha256_file(plan_path),
    }


def _load_model_and_closure(args: argparse.Namespace) -> tuple[Any, Any, tuple[int, ...]]:
    model, tokenizer, _adapter = load_registered_model(
        resolve_model_spec(str(args.model)),
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    closure_ids = tuple(
        int(value) for value in tokenizer.encode(FORCED_SUFFIX, add_special_tokens=False)
    )
    closure_text = tokenizer.decode(
        list(closure_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if closure_text != FORCED_SUFFIX:
        raise ValueError("Forced suffix does not round-trip through the tokenizer")
    return model, tokenizer, closure_ids


def run_discovery(args: argparse.Namespace, common: Mapping[str, Any]) -> None:
    discovery_seeds = [
        int(value) for value in common["source_manifest"]["discovery_seeds"]
    ]
    shard_dir = args.output_dir / "discovery_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, closure_ids = _load_model_and_closure(args)
    rows: list[dict[str, Any]] = []
    total = len(discovery_seeds) * len(DELAYS)
    ordinal = 0
    for seed in discovery_seeds:
        source = common["source_by_seed"][seed]
        context = common["context_by_seed"][seed]
        if sha256_json(source) != context.get("source_row_sha256"):
            raise ValueError(f"Source row hash mismatch for seed={seed}")
        for delay in DELAYS:
            ordinal += 1
            shard = shard_dir / f"seed{seed}_delay{delay}.json"
            if args.resume and shard.is_file():
                row = read_json(shard)
                if row.get("plan_sha256") != common["plan_hash"]:
                    raise ValueError("Resumed discovery shard uses another plan")
            else:
                condition = _condition_input(
                    source,
                    context,
                    tokenizer,
                    delay=delay,
                    closure_ids=closure_ids,
                )
                row = _generate_condition(
                    source=source,
                    context=context,
                    condition=condition,
                    tokenizer=tokenizer,
                    model=model,
                    plan_hash=str(common["plan_hash"]),
                    max_new_tokens=int(args.max_new_tokens),
                )
                atomic_json(shard, row)
            rows.append(row)
            print(
                f"[delay-discovery] {ordinal}/{total} seed={seed} d={delay} "
                f"clean={int(row['site_clean'])} forced={row['forced_total']} "
                f"gold={row['gold_total']}",
                flush=True,
            )
    rows_path = args.output_dir / DISCOVERY_ROWS_FILENAME
    atomic_jsonl(rows_path, rows)
    atomic_csv(args.output_dir / DISCOVERY_CSV_FILENAME, _csv_rows(rows))
    summary = {
        "schema_version": SCHEMA,
        "status": "DISCOVERY_COMPLETE",
        "plan_sha256": common["plan_hash"],
        "delay_summaries": _summaries(rows),
        "files": {
            DISCOVERY_ROWS_FILENAME: sha256_file(rows_path),
            DISCOVERY_CSV_FILENAME: sha256_file(
                args.output_dir / DISCOVERY_CSV_FILENAME
            ),
        },
        "confirmation_outcomes_accessed": False,
    }
    atomic_json(args.output_dir / DISCOVERY_SUMMARY_FILENAME, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def freeze_boundary(args: argparse.Namespace, common: Mapping[str, Any]) -> None:
    rows_path = args.output_dir / DISCOVERY_ROWS_FILENAME
    summary_path = args.output_dir / DISCOVERY_SUMMARY_FILENAME
    rows = read_jsonl(rows_path)
    summary = read_json(summary_path)
    if summary.get("plan_sha256") != common["plan_hash"]:
        raise ValueError("Discovery summary uses another frozen plan")
    expected = len(common["source_manifest"]["discovery_seeds"]) * len(DELAYS)
    if len(rows) != expected:
        raise ValueError("Discovery delay grid is incomplete")
    summaries = _summaries(rows)
    chosen = choose_boundary_summary(summaries)
    confirmation_dir = args.output_dir / "confirmation_shards"
    if confirmation_dir.exists() and any(confirmation_dir.glob("*.json")):
        raise ValueError("Confirmation shards already exist before boundary freeze")
    boundary = {
        "schema_version": SCHEMA,
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "plan_sha256": common["plan_hash"],
        "discovery_rows_sha256": sha256_file(rows_path),
        "discovery_summary_sha256": sha256_file(summary_path),
        "candidate_summaries": summaries,
        "chosen_delay_tokens": int(chosen["delay_tokens"]),
        "selection_rule": (
            "eligible iff 20/20 discovery sites are explicit-cue-free and add no "
            "gold-city mention; maximize correct_n; ties choose smallest delay"
        ),
        "confirmation_outcomes_accessed": False,
    }
    atomic_json(args.output_dir / BOUNDARY_FILENAME, boundary)
    print(json.dumps(boundary, ensure_ascii=False, indent=2, sort_keys=True))


def freeze_boundary_v2(args: argparse.Namespace, common: Mapping[str, Any]) -> None:
    """Freeze a transparent eligibility amendment before confirmation.

    V1 rejected every delay because one seed immediately repeated the already
    observed K-th city.  That is not an explicit count/index cue and adds no new
    unique evidence.  V2 therefore keeps the original outcome-based selection
    rule but limits leakage eligibility to the preregistered explicit-cue
    grammar.  The amendment is deliberately recorded after discovery and
    before any delayed confirmation generation.
    """

    rows_path = args.output_dir / DISCOVERY_ROWS_FILENAME
    summary_path = args.output_dir / DISCOVERY_SUMMARY_FILENAME
    rows = read_jsonl(rows_path)
    summary = read_json(summary_path)
    if summary.get("plan_sha256") != common["plan_hash"]:
        raise ValueError("Discovery summary uses another frozen plan")
    expected = len(common["source_manifest"]["discovery_seeds"]) * len(DELAYS)
    if len(rows) != expected:
        raise ValueError("Discovery delay grid is incomplete")
    v1_summaries = _summaries(rows)
    if any(bool(value["boundary_eligible"]) for value in v1_summaries):
        raise ValueError("V2 amendment is only valid after the V1 eligibility impasse")
    v2_summaries = _explicit_cue_only_summaries(rows)
    chosen = choose_boundary_summary(v2_summaries)
    for directory_name in ("confirmation_shards", "confirmation_shards_v2"):
        directory = args.output_dir / directory_name
        if directory.exists() and any(directory.glob("*.json")):
            raise ValueError("Confirmation shards already exist before V2 freeze")
    boundary = {
        "schema_version": SCHEMA,
        "boundary_policy_version": "explicit_count_cue_only_v2",
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "plan_sha256": common["plan_hash"],
        "discovery_rows_sha256": sha256_file(rows_path),
        "discovery_summary_sha256": sha256_file(summary_path),
        "v1_candidate_summaries": v1_summaries,
        "candidate_summaries": v2_summaries,
        "chosen_delay_tokens": int(chosen["delay_tokens"]),
        "eligibility_amendment": (
            "V1 additionally excluded any repeated gold-city mention. Discovery "
            "showed that this rejects all delays solely because seed 1978 begins "
            "by restating its already observed K-th city, without an explicit "
            "count/index cue or new unique evidence. V2 excludes only explicit "
            "count/index/Total cues under the frozen prefix-clean grammar."
        ),
        "selection_rule": (
            "eligible iff 20/20 discovery extended prefixes are explicit-cue-free; "
            "maximize correct_n; ties choose the smallest delay"
        ),
        "discovery_outcomes_accessed_before_amendment": True,
        "confirmation_outcomes_accessed": False,
    }
    atomic_json(args.output_dir / BOUNDARY_V2_FILENAME, boundary)
    print(json.dumps(boundary, ensure_ascii=False, indent=2, sort_keys=True))


def run_confirmation(
    args: argparse.Namespace,
    common: Mapping[str, Any],
    *,
    boundary_filename: str = BOUNDARY_FILENAME,
    shard_directory: str = "confirmation_shards",
    rows_filename: str = CONFIRMATION_ROWS_FILENAME,
    csv_filename: str = CONFIRMATION_CSV_FILENAME,
    manifest_filename: str = MANIFEST_FILENAME,
    supersedes_manifest_filename: str | None = None,
) -> None:
    boundary_path = args.output_dir / boundary_filename
    boundary = read_json(boundary_path)
    if boundary.get("status") != "FROZEN_BEFORE_CONFIRMATION":
        raise ValueError("Boundary was not frozen before confirmation")
    if boundary.get("plan_sha256") != common["plan_hash"]:
        raise ValueError("Frozen boundary uses another plan")
    discovery_rows_path = args.output_dir / DISCOVERY_ROWS_FILENAME
    if boundary.get("discovery_rows_sha256") != sha256_file(discovery_rows_path):
        raise ValueError("Discovery rows changed after boundary freeze")
    chosen_delay = int(boundary["chosen_delay_tokens"])
    if chosen_delay not in DELAYS:
        raise ValueError("Frozen delay is outside the preregistered grid")
    confirmation_seeds = [
        int(value) for value in common["source_manifest"]["confirmation_seeds"]
    ]
    shard_dir = args.output_dir / shard_directory
    shard_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, closure_ids = _load_model_and_closure(args)
    rows: list[dict[str, Any]] = []
    for ordinal, seed in enumerate(confirmation_seeds, start=1):
        shard = shard_dir / f"seed{seed}_delay{chosen_delay}.json"
        if args.resume and shard.is_file():
            row = read_json(shard)
            if row.get("plan_sha256") != common["plan_hash"]:
                raise ValueError("Resumed confirmation shard uses another plan")
        else:
            source = common["source_by_seed"][seed]
            context = common["context_by_seed"][seed]
            condition = _condition_input(
                source,
                context,
                tokenizer,
                delay=chosen_delay,
                closure_ids=closure_ids,
            )
            row = _generate_condition(
                source=source,
                context=context,
                condition=condition,
                tokenizer=tokenizer,
                model=model,
                plan_hash=str(common["plan_hash"]),
                max_new_tokens=int(args.max_new_tokens),
            )
            row["frozen_boundary_choice_sha256"] = sha256_file(boundary_path)
            atomic_json(shard, row)
        rows.append(row)
        print(
            f"[delay-confirmation] {ordinal}/10 seed={seed} d={chosen_delay} "
            f"clean={int(row['site_clean'])} forced={row['forced_total']} "
            f"gold={row['gold_total']}",
            flush=True,
        )
    rows_path = args.output_dir / rows_filename
    csv_path = args.output_dir / csv_filename
    atomic_jsonl(rows_path, rows)
    atomic_csv(csv_path, _csv_rows(rows))
    boundary_policy_version = str(
        boundary.get("boundary_policy_version", "site_clean_v1")
    )
    explicit_cue_free_n = sum(
        not bool(row.get("explicit_count_cues")) for row in rows
    )
    repeated_gold_city_n = sum(
        bool(row.get("post_tstar_gold_city_mentions")) for row in rows
    )
    legacy_v1_site_clean_n = sum(bool(row["site_clean"]) for row in rows)
    if boundary_policy_version == "explicit_count_cue_only_v2":
        eligibility_metric = "explicit_count_cue_free"
        eligible_under_frozen_policy_n = explicit_cue_free_n
    else:
        eligibility_metric = "legacy_v1_site_clean"
        eligible_under_frozen_policy_n = legacy_v1_site_clean_n
    result = {
        "row_count": len(rows),
        "chosen_delay_tokens": chosen_delay,
        "boundary_policy_version": boundary_policy_version,
        "eligibility_metric": eligibility_metric,
        "eligible_under_frozen_policy_n": eligible_under_frozen_policy_n,
        "explicit_cue_free_n": explicit_cue_free_n,
        "repeated_gold_city_n": repeated_gold_city_n,
        "legacy_v1_site_clean_n": legacy_v1_site_clean_n,
        "forced_total_parsed_n": sum(row.get("forced_total") is not None for row in rows),
        "forced_total_gold_correct_n": sum(
            bool(row["forced_total_gold_correct"]) for row in rows
        ),
        "forced_total_matches_source_n": sum(
            bool(row["forced_total_matches_source"]) for row in rows
        ),
        "immediate_integer_only_n": sum(
            bool(row["immediate_integer_only"]) for row in rows
        ),
        "stopped_on_eos_n": sum(bool(row["stopped_on_eos"]) for row in rows),
    }
    legacy_v1_discovery_summaries = read_json(
        args.output_dir / DISCOVERY_SUMMARY_FILENAME
    )["delay_summaries"]
    discovery_summaries = boundary.get(
        "candidate_summaries", legacy_v1_discovery_summaries
    )
    manifest = {
        "schema_version": SCHEMA,
        "status": "COMPLETE",
        "model_label": str(args.model),
        "fixed_count": int(common["fixed_count"]),
        "plan_sha256": common["plan_hash"],
        "frozen_boundary_choice_sha256": sha256_file(boundary_path),
        "boundary_policy_version": boundary_policy_version,
        "chosen_delay_tokens": chosen_delay,
        "discovery_delay_summaries": discovery_summaries,
        "legacy_v1_discovery_delay_summaries": legacy_v1_discovery_summaries,
        "confirmation": result,
        "files": {
            PLAN_FILENAME: sha256_file(common["plan_path"]),
            DISCOVERY_ROWS_FILENAME: sha256_file(discovery_rows_path),
            DISCOVERY_CSV_FILENAME: sha256_file(
                args.output_dir / DISCOVERY_CSV_FILENAME
            ),
            DISCOVERY_SUMMARY_FILENAME: sha256_file(
                args.output_dir / DISCOVERY_SUMMARY_FILENAME
            ),
            boundary_filename: sha256_file(boundary_path),
            rows_filename: sha256_file(rows_path),
            csv_filename: sha256_file(csv_path),
        },
        "confirmation_used_for_boundary_selection": False,
    }
    if supersedes_manifest_filename is not None:
        superseded_path = args.output_dir / supersedes_manifest_filename
        if superseded_path.is_file():
            manifest["supersedes_manifest"] = {
                "filename": supersedes_manifest_filename,
                "sha256": sha256_file(superseded_path),
                "reason": (
                    "Reporting-only revision: expose V2 explicit-cue eligibility as "
                    "the primary discovery summary and retain V1 site-clean counts "
                    "as a labeled legacy audit. Trial rows, frozen boundary, and "
                    "confirmation outcomes are unchanged."
                ),
            }
    atomic_json(args.output_dir / manifest_filename, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "discovery",
            "freeze",
            "confirmation",
            "freeze-v2",
            "confirmation-v2",
        ),
        required=True,
    )
    parser.add_argument("--selected-rows", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common = _load_common(args)
    if args.stage == "discovery":
        run_discovery(args, common)
    elif args.stage == "freeze":
        freeze_boundary(args, common)
    elif args.stage == "confirmation":
        run_confirmation(args, common)
    elif args.stage == "freeze-v2":
        freeze_boundary_v2(args, common)
    else:
        run_confirmation(
            args,
            common,
            boundary_filename=BOUNDARY_V2_FILENAME,
            shard_directory="confirmation_shards_v2",
            rows_filename=CONFIRMATION_V2_ROWS_FILENAME,
            csv_filename=CONFIRMATION_V2_CSV_FILENAME,
            manifest_filename=MANIFEST_V3_FILENAME,
            supersedes_manifest_filename=MANIFEST_V2_FILENAME,
        )


if __name__ == "__main__":
    main()
