"""Compile a common terminal-relay panel using baseline geometry only, on CPU."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def candidate_pairs(read_seeds, ranges):
    if len(read_seeds) != len(set(read_seeds)):
        raise ValueError("Duplicate source seeds")
    result = []
    for seed in sorted(read_seeds):
        for offset, counts in sorted(ranges.items(), key=lambda item: int(item[0])):
            offset = int(offset)
            for count in counts:
                if not 1 <= count + offset < count:
                    raise ValueError("Relay donor must be a prior observed item")
                result.append({"seed": int(seed), "gold_count": count,
                               "receiver_occurrence": count, "donor_occurrence": count + offset,
                               "donor_offset": offset, "pair_id": f"seed{seed}_N{count}_offset{offset}"})
    if len({row["pair_id"] for row in result}) != len(result):
        raise ValueError("Duplicate count/offset cells")
    return result


def common_width_pairs(ledgers, *, widths):
    if not ledgers:
        raise ValueError("No mode ledgers")
    lookups = [{row["pair_id"]: row for row in ledger} for ledger in ledgers]
    for ledger, lookup in zip(ledgers, lookups):
        if len(ledger) != len(lookup):
            raise ValueError("Duplicate pairs in mode ledger")
    keys = set(lookups[0])
    if any(set(table) != keys for table in lookups):
        raise ValueError("Modes do not account for the same candidate pairs")
    common = {str(width): sorted(key for key in keys if all(table[key]["widths"][str(width)]["eligible"] for table in lookups))
              for width in widths}
    common["joint_widths"] = sorted(set.intersection(*(set(common[str(width)]) for width in widths)))
    return common


def relay_baseline_exclusion(source_entry, parsed, *, model, mode, cohort_policy):
    if not source_entry["read_eligible"]:
        return source_entry.get("read_exclusion", "No legal answer query")
    if cohort_policy == "native_assay_specific_per_mode":
        # Native terminal-relay uses --cohort one_to_one. The no-index
        # first-pass gate belongs to its Update assay, not to this assay.
        if not parsed.get("detected"):
            return "Native relay cohort: parser_miss"
        if not parsed.get("trace_one_to_one"):
            return "Native relay cohort: not_one_to_one"
        return None
    if not source_entry["format_eligible"]:
        return "Baseline does not meet mode-specific item-format/order gate"
    if not (model == "Gemma4-E4B" and mode == "thinking"):
        if not parsed.get("trace_one_to_one") or parsed.get("trace_order_class") != "forward":
            return "Actual full baseline trace is not one-to-one and forward"
    return None


def rebase_actual_answer_geometry(first_encoding, first_registry, actual_encoding):
    """Keep audited FOUND item spans but retain the actual full answer prefix."""
    terminal_end = int(first_registry.trace_items[-1][1])
    prompt = int(actual_encoding.prompt_token_count)
    query = int(actual_encoding.query_position)
    if first_registry.prompt_token_count != prompt or terminal_end > query:
        raise ValueError("FOUND registry crosses the actual answer query")
    if tuple(first_encoding.input_ids[:terminal_end]) != tuple(actual_encoding.input_ids[:terminal_end]):
        raise ValueError("FOUND semantic registry changes literal source tokens")
    item_positions = {position for start, end in first_registry.trace_items for position in range(start, end)}
    other = sorted(set(range(prompt, query)) - item_positions)
    other_spans = []
    for position in other:
        if other_spans and other_spans[-1][1] == position:
            other_spans[-1] = (other_spans[-1][0], position + 1)
        else:
            other_spans.append((position, position + 1))
    registry = replace(first_registry, answer_site_id="answer_query_v3",
                       sequence_length=actual_encoding.sequence_length, query_position=query,
                       trace_context=((prompt, query),), trace_other=tuple(other_spans))
    registry.validate()
    encoding = replace(actual_encoding, trace_item_spans=first_encoding.trace_item_spans,
                       slot_spans=first_encoding.slot_spans, needle_spans=first_encoding.needle_spans)
    assert tuple(encoding.input_ids) == tuple(actual_encoding.input_ids)
    assert tuple(encoding.attention_mask) == tuple(actual_encoding.attention_mask)
    return encoding, registry


def literal_decoded_offsets(tokenizer, raw, token_ids):
    """Align noncanonical generated IDs only when every decoded prefix is exact.

    Generation can produce a valid token sequence that differs from encoding
    its decoded text (including at the prefilled FOUND boundary). Never replace
    those IDs with a canonical re-tokenization. Reject unstable Unicode/decoder
    boundaries instead of guessing their character offsets.
    """
    offsets = []
    previous = ""
    for end in range(1, len(token_ids) + 1):
        prefix = tokenizer.decode(list(token_ids[:end]), skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        if not prefix.startswith(previous) or not raw.startswith(prefix):
            raise ValueError("Stored output has a non-monotone decoded token boundary")
        offsets.append((len(previous), len(prefix)))
        previous = prefix
    if previous != raw:
        raise ValueError("Stored output IDs do not decode to the frozen raw text")
    return tuple(offsets)


def build_found_actual_geometry(row, tokenizer):
    """Reuse frozen FOUND semantics, with literal stored-token alignment."""
    from realistic_niah_v5 import tstar_prefix, counting_mechanism_transfer as transfer
    from realistic_niah_v5.causal_sites import CausalSiteError, OutputTokenMap, build_output_token_map
    from realistic_niah_v5.parsing import output_token_ids, raw_output_text
    from scripts.enumeration_fresh_geometry import build_read_geometry
    strategy = "canonical_roundtrip_exact"
    try:
        token_map = build_output_token_map(row, tokenizer)
    except CausalSiteError as error:
        # Only a noncanonical segmentation is eligible for this fallback.
        # Missing offsets and other tokenizer failures remain fatal/explicit.
        if not str(error).startswith("Re-tokenized output differs from frozen output IDs"):
            raise
        ids, raw = tuple(output_token_ids(row)), raw_output_text(row)
        token_map = OutputTokenMap(raw_text=raw, token_ids=ids,
            offsets=literal_decoded_offsets(tokenizer, raw, ids), tokenizer=tokenizer)
        strategy = "literal_stored_ids_exact_monotone_decoded_prefixes"
    old_prefix_map, old_transfer_map = tstar_prefix.build_output_token_map, transfer.build_output_token_map
    def frozen_map(source_row, source_tokenizer):
        if source_row is not row or source_tokenizer is not tokenizer:
            raise RuntimeError("FOUND token map was requested for a different input")
        return token_map
    tstar_prefix.build_output_token_map = transfer.build_output_token_map = frozen_map
    try:
        first_encoding, first_registry = transfer.build_first_pass_tstar_answer_source_registry(
            row, tokenizer, selection_population="gemma_prompt_conditioned_noindex_found_v3",
            eligibility_field="primary_eligible_prompt_conditioned_noindex")
    finally:
        tstar_prefix.build_output_token_map, transfer.build_output_token_map = old_prefix_map, old_transfer_map
    actual_encoding, _masks, _audit = build_read_geometry(row, tokenizer, mode="thinking")
    encoding, registry = rebase_actual_answer_geometry(first_encoding, first_registry, actual_encoding)
    return encoding, registry, strategy


def compile_cell(args, cfg, baseline):
    tick = time.monotonic()
    folder = args.output / args.model / args.mode
    folder.mkdir(parents=True, exist_ok=False)
    code = args.causal_stage / "code"
    code_manifest = json.loads((args.causal_stage / "code_manifest.json").read_text())
    for name, expected in {**code_manifest["original_code_sha256"], **code_manifest["additive_code_sha256"]}.items():
        assert sha(code / name) == expected, name
    sys.path[:0] = [str(code / "src"), str(code)]
    if args.mode != "thinking":
        from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
        install_v6_kernel_adapters()
        install_v6_specialized_geometry(args.mode)
    from realistic_niah_v4.modeling import load_registered_tokenizer
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v5.count_stream import build_answer_source_registry, trace_patch_geometry_positions
    from realistic_niah_v5.causal_sites import CausalSiteError
    source = args.causal_stage / "registries" / args.model / args.mode
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["baseline_manifest_sha256"] == sha(args.bundle / "manifest.json")
    assert sha(source / "ledger.json") == manifest["ledger_sha256"]
    assert sha(source / "adapted_generations.jsonl") == manifest["adapted_generations_sha256"]
    ledger = json.loads((source / "ledger.json").read_text())
    read_entries = {(row["seed"], row["gold_count"]): row for row in ledger if "unfiltered_read" in row["roles"]}
    assert len(read_entries) == 100
    rows = {}
    with (source / "adapted_generations.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if (int(row["seed"]), int(row["gold_count"])) in read_entries:
                rows[int(row["seed"]), int(row["gold_count"])] = row
    assert set(rows) == set(read_entries)
    tokenizer = load_registered_tokenizer(resolve_model_spec(args.model), cache_dir=args.cache_dir)
    candidates = candidate_pairs(baseline["read_seeds"], cfg["count_ranges_by_donor_offset"])
    result, compiled = [], {}
    for pair in candidates:
        key = pair["seed"], pair["gold_count"]
        row, source_entry = rows[key], read_entries[key]
        record = {**pair, "model": args.model, "mode": args.mode,
                  "request_id": row["request_id"], "widths": {}}
        error = relay_baseline_exclusion(source_entry, row["trace_parse"]["parser"],
            model=args.model, mode=args.mode, cohort_policy=cfg.get("cohort_policy", "legacy_common_update_gate"))
        if error is None:
            try:
                if key not in compiled:
                    if args.model == "Gemma4-E4B" and args.mode == "thinking":
                        # The frozen FOUND audit already verifies exactly one
                        # line per gold record in order. The legacy generic
                        # parser does not recognize this literal marker.
                        compiled[key] = build_found_actual_geometry(row, tokenizer)
                    else:
                        encoding, registry = build_answer_source_registry(row, tokenizer, answer_site_id=cfg["answer_site_id"])
                        compiled[key] = encoding, registry, "original_full_baseline_parser"
                encoding, registry, mapping_strategy = compiled[key]
                if len(registry.trace_items) != pair["gold_count"]:
                    raise ValueError("Actual full answer registry does not contain exactly N items")
                record["answer_registry"] = registry.to_dict()
                record["registry_item_source"] = ("frozen_FOUND_semantic_lines" if args.model == "Gemma4-E4B" and args.mode == "thinking" else "full_baseline_trace_parser")
                record["query_source"] = "actual_full_baseline_answer_query_v3"
                record["semantic_token_mapping"] = mapping_strategy
                record["input_ids_sha256"] = hashlib.sha256(json.dumps(list(encoding.input_ids)).encode()).hexdigest()
                end, query = registry.trace_items[-1][1], registry.query_position
                if end > query:
                    raise ValueError("Terminal item crosses the answer query")
                record["query_reset_positions"] = [query]
                record["post_terminal_reset_positions"] = list(range(end, query + 1))
                record["query_token_id"] = int(encoding.input_ids[query])
            except (ValueError, CausalSiteError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        for width in cfg["widths"]:
            geometry = {"eligible": False}
            if error is not None:
                geometry["exclusion"] = error
            else:
                try:
                    receiver, donor, audit = trace_patch_geometry_positions(registry,
                        receiver_occurrence=pair["receiver_occurrence"], donor_occurrence=pair["donor_occurrence"],
                        geometry=f"suffix{width}")
                    assert len(receiver) == len(donor) == width
                    geometry.update(eligible=True, **audit)
                except ValueError as exc:
                    geometry["exclusion"] = str(exc)
            record["widths"][str(width)] = geometry
        result.append(record)
    write(folder / "pairs.json", result)
    summary = {"status": "FROZEN_BASELINE_GEOMETRY", "model": args.model, "mode": args.mode,
               "candidate_pairs": len(result), "candidate_source_seeds": len(baseline["read_seeds"]),
               "eligible_by_width": {str(width): sum(row["widths"][str(width)]["eligible"] for row in result) for width in cfg["widths"]},
               "exclusions_by_width": {str(width): dict(Counter(row["widths"][str(width)].get("exclusion") for row in result if not row["widths"][str(width)]["eligible"])) for width in cfg["widths"]},
               "source_registry_sha256": sha(source / "manifest.json"), "pairs_sha256": sha(folder / "pairs.json"),
               "baseline_manifest_sha256": sha(args.bundle / "manifest.json"), "code_manifest_sha256": sha(args.causal_stage / "code_manifest.json"),
               "protocol_sha256": sha(args.protocol), "compiler_sha256": sha(__file__),
               "final_correctness_used_for_selection": False, "relay_intervention_outcomes_read": False,
               "seconds": time.monotonic() - tick, "command": sys.argv}
    write(folder / "manifest.json", summary)
    print(json.dumps(summary), flush=True)


def finalize(args, cfg):
    tick = time.monotonic()
    target = args.output / "common_cohorts.json"
    if target.exists():
        raise FileExistsError(target)
    result = {"status": "FROZEN_BEFORE_RELAY_INTERVENTIONS", "models": {}, "manifests": {},
              "cohort_policy": cfg.get("cohort_policy", "legacy_common_update_gate"),
              "protocol_sha256": sha(args.protocol), "compiler_sha256": sha(__file__),
              "baseline_manifest_sha256": sha(args.bundle / "manifest.json"),
              "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv}
    for model in cfg["models"]:
        ledgers = []
        for mode in cfg["modes"]:
            folder = args.output / model / mode
            manifest = json.loads((folder / "manifest.json").read_text())
            assert manifest["protocol_sha256"] == result["protocol_sha256"]
            assert manifest["compiler_sha256"] == result["compiler_sha256"]
            assert manifest["baseline_manifest_sha256"] == result["baseline_manifest_sha256"]
            assert sha(folder / "pairs.json") == manifest["pairs_sha256"]
            ledgers.append(json.loads((folder / "pairs.json").read_text()))
            result["manifests"][f"{model}/{mode}"] = sha(folder / "manifest.json")
        common = common_width_pairs(ledgers, widths=cfg["widths"])
        lookup = {row["pair_id"]: row for row in ledgers[0]}
        result["models"][model] = {"pairs": common,
            "coverage": {width: {"pairs": len(keys), "source_seeds": len({lookup[key]["seed"] for key in keys}),
                                 "count_offset_cells": dict(Counter(f"N{lookup[key]['gold_count']}/offset{lookup[key]['donor_offset']}" for key in keys))}
                         for width, keys in common.items()},
            "candidate_pairs_per_mode": len(ledgers[0])}
        if cfg.get("cohort_policy") == "native_assay_specific_per_mode":
            per_mode = {}
            for mode, ledger in zip(cfg["modes"], ledgers):
                own = common_width_pairs([ledger], widths=cfg["widths"])
                per_mode[mode] = {"pairs": own,
                    "coverage": {width: {"pairs": len(keys), "source_seeds": len({lookup[key]["seed"] for key in keys})}
                                 for width, keys in own.items()}}
            result["models"][model]["per_mode"] = per_mode
            result["models"][model]["common_intersection_role"] = "supplementary_only_not_primary_selection"
    result["seconds"] = time.monotonic() - tick
    write(target, result)
    print(json.dumps({"status": result["status"], "coverage": {model: value["coverage"] for model, value in result["models"].items()},
        "per_mode_coverage": {model: {mode: data["coverage"] for mode, data in value.get("per_mode", {}).items()} for model, value in result["models"].items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "causal-stage", "protocol", "cache-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--mode", choices=("enumeration_index", "enumeration_bullet", "thinking"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--finalize", action="store_true")
    group.add_argument("--all-cells", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text(encoding="utf-8"))
    baseline = json.loads((args.bundle / "manifest.json").read_text())
    if args.all_cells:
        args.output.mkdir(parents=True, exist_ok=False)
        tick = time.monotonic()
        write(args.output / "protocol.json", cfg)
        common = ["--bundle", str(args.bundle), "--causal-stage", str(args.causal_stage),
                  "--protocol", str(args.protocol), "--cache-dir", str(args.cache_dir), "--output", str(args.output)]
        jobs = [[sys.executable, str(Path(__file__).resolve()), *common, "--model", model, "--mode", mode]
                for model in cfg["models"] for mode in cfg["modes"]]
        write(args.output / "commands.json", {"jobs": jobs, "protocol_sha256": sha(args.protocol), "compiler_sha256": sha(__file__)})
        state = {"status": "RUNNING", "completed_cells": 0, "total_cells": len(jobs), "pid": os.getpid()}
        def save():
            state["seconds"] = time.monotonic() - tick
            path = args.output / "pipeline_status.json"
            temp = path.with_suffix(".tmp")
            write(temp, state)
            temp.replace(path)
        save()
        try:
            for command in jobs:
                state["command"] = command
                save()
                result = subprocess.run(command, check=False)
                if result.returncode:
                    raise RuntimeError(f"CPU cell failed with exit code {result.returncode}: {command[-4:]}")
                state["completed_cells"] += 1
                save()
            finalize(args, cfg)
            state.update(status="COMPLETE", phase="RELAY_REGISTRY_FROZEN_GPU_RUNNER_PENDING")
        except BaseException as error:
            state.update(status="FAILED", error=repr(error))
            raise
        finally:
            save()
    elif args.finalize:
        finalize(args, cfg)
    else:
        if args.model is None or args.mode is None:
            parser.error("Each CPU cell needs --model and --mode")
        compile_cell(args, cfg, baseline)


if __name__ == "__main__":
    main()
