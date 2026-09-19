"""CPU-only integrity and source-seed summaries for the frozen fresh Read assay."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = ("clean", "prompt_all_blank", "prompt_records_blank",
              "trace_all_blank", "prompt_and_trace_blank")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def key(row):
    return int(row["seed"]), int(row["gold_count"])


def unique(rows):
    result = {}
    for row in rows:
        require(key(row) not in result, f"Duplicate input: {key(row)}")
        result[key(row)] = row
    return result


def matched_trials(trials, eligible):
    result = {}
    for row in trials:
        k = (*key(row), row["condition"])
        require(k not in result, f"Duplicate trial: {k}")
        result[k] = row
    expected = {(*k, condition) for k in eligible for condition in CONDITIONS}
    require(set(result) == expected,
            f"Unmatched five-arm trial grid: missing={len(expected - result.keys())}, "
            f"unexpected={len(result.keys() - expected)}")
    return result


def verify_masks(geometry):
    prompt, query, length = (int(geometry[k]) for k in
                             ("prompt_token_count", "query_position", "sequence_length"))
    require(1 < prompt <= query == length - 1, "Invalid Read boundaries")
    records = set()
    for start, end in geometry["prompt_record_spans"]:
        require(1 <= start < end <= prompt, "Prompt record crosses boundary")
        records.update(range(start, end))
    require(bool(records), "No prompt records")
    masks = {"clean": [], "prompt_all_blank": list(range(1, prompt)),
             "prompt_records_blank": sorted(records), "trace_all_blank": list(range(prompt, query)),
             "prompt_and_trace_blank": list(range(1, query))}
    require(set(geometry["conditions"]) == set(CONDITIONS), "Unexpected mask condition")
    for condition, positions in masks.items():
        record = geometry["conditions"][condition]
        require(record == {"positions_sha256": json_sha(positions), "token_count": len(positions)},
                f"Mask mismatch: {condition}")
    return masks


def verify_trial(trial, entry, geometry, *, layers, parse_total, max_new_tokens):
    condition = trial["condition"]
    mask = geometry["conditions"][condition]
    require(trial["status"] == "ok", "Failed trial")
    for name in ("seed", "gold_count", "request_id", "geometry_sha256"):
        require(trial[name] == entry[name], f"Trial input mismatch: {name}")
    require(trial["blank_token_count"] == mask["token_count"], "Blank token count mismatch")
    require(trial["blank_positions_sha256"] == mask["positions_sha256"], "Blank mask hash mismatch")
    hooks = trial["blank_hook_audit"]
    layer_hooks = hooks["blank_layer_hook_applications"]
    require(set(layer_hooks) == {str(i) for i in range(layers)}, "Missing layer hook evidence")
    count = hooks["blank_embedding_hook_applications"]
    probes = trial["embedding_zero_probe"]
    if mask["token_count"]:
        require(count >= 1 and all(n == count for n in layer_hooks.values()), "Incomplete blank hooks")
        require(len(probes) == count and all(math.isfinite(x) and x == 0 for x in probes),
                "Blank embedding was not exactly zero")
        require(hooks["blank_prefill_sequence_lengths"] == [geometry["sequence_length"]] * count,
                "Wrong prefill length")
    else:
        require(count == 0 and not probes and not hooks["blank_prefill_sequence_lengths"]
                and all(n == 0 for n in layer_hooks.values()), "Unexpected clean intervention")
    generated = trial["generated"]
    tokens = generated["generated_token_ids"]
    require(0 < len(tokens) <= max_new_tokens, "Invalid continuation length")
    stopped = tokens[-1] in generated["generation_eos_token_ids"]
    truncated = len(tokens) >= max_new_tokens and not stopped
    require(generated["stopped_on_eos"] == stopped, "EOS mismatch")
    require(generated["generation_truncated"] == truncated, "Truncation mismatch")
    require(generated["generated_token_count"] == len(tokens), "Raw token count mismatch")
    require(generated["full_answer_text"] == "Total:" + generated["completion_text"],
            "Answer text reconstruction mismatch")
    prediction = parse_total(generated["full_answer_text"])
    delta = None if prediction is None else prediction - int(entry["gold_count"])
    metrics = {"prediction": prediction, "exact_count": prediction == int(entry["gold_count"]),
               "signed_error": delta, "absolute_error": None if delta is None else abs(delta),
               "completion_text_raw": generated["completion_text_raw"],
               "generated_token_count": len(tokens), "generation_truncated": truncated}
    for name, value in metrics.items():
        require(trial[name] == value, f"Saved metric differs from raw output: {name}")


def cluster_mean(observations, *, repetitions=10000, seed=20260915):
    """Pooled input mean; resample source seeds, retaining all within-seed inputs."""
    import numpy as np
    groups = defaultdict(list)
    for source_seed, value in observations:
        require(math.isfinite(value), "Non-finite observation")
        groups[int(source_seed)].append(float(value))
    require(bool(groups), "Empty observed denominator")
    ordered = sorted(groups)
    numerators = np.array([sum(groups[k]) for k in ordered])
    denominators = np.array([len(groups[k]) for k in ordered])
    interval = None
    if len(ordered) >= 2:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(ordered), size=(repetitions, len(ordered)))
        values = numerators[draws].sum(axis=1) / denominators[draws].sum(axis=1)
        interval = np.quantile(values, [0.025, 0.975]).tolist()
    return {"numerator": float(numerators.sum()), "denominator": int(denominators.sum()),
            "value": float(numerators.sum() / denominators.sum()), "ci95": interval,
            "source_seeds": ordered, "independent_seed_count": len(ordered),
            "by_seed": [{"seed": k, "numerator": float(numerators[i]),
                         "denominator": int(denominators[i])} for i, k in enumerate(ordered)]}


def audit_cell(root, model, mode, parse_total, cfg, *, result_folder=None):
    causal = root / "fresh_causal_v1"
    registry = causal / "registries" / model / mode
    folder = result_folder or causal / "gpu_jobs/formal/read" / model / mode
    manifest, status, runtime = read(registry / "manifest.json"), read(folder / "status.json"), read(folder / "runtime.json")
    require(manifest["status"] == "FROZEN_BASELINE_GEOMETRY", "Unfrozen registry")
    require(manifest["selection_used_final_correctness"] is False and
            manifest["intervention_outcomes_accessed"] is False, "Outcome-selected registry")
    for name, field in (("ledger.json", "ledger_sha256"), ("adapted_generations.jsonl", "adapted_generations_sha256")):
        require(sha(registry / name) == manifest[field], f"Registry hash mismatch: {name}")
    require(sha(root / "fresh_v1/manifest.json") == manifest["baseline_manifest_sha256"], "Baseline manifest mismatch")
    original_path = root / "fresh_v1/baseline" / model / mode / "generations.jsonl"
    require(sha(original_path) == manifest["baseline_generations_sha256"], "Baseline file mismatch")
    original, adapted = unique(jsonl(original_path)), unique(jsonl(registry / "adapted_generations.jsonl"))
    ledger = read(registry / "ledger.json")
    population = unique([row for row in ledger if "unfiltered_read" in row["roles"]])
    first = cfg["first_seed"] + cfg["discovery_seed_count"]
    expected = {(s, n) for s in range(first, first + cfg["read_seed_count"]) for n in cfg["read_counts"]}
    require(set(population) == expected and len(expected) == 100, "Read population differs from planned unfiltered pool")
    require(read(folder / "coverage.json") == [row for row in ledger if "unfiltered_read" in row["roles"]],
            "Coverage differs from original registry")
    eligible = {k: v for k, v in population.items() if v["read_eligible"]}
    require(status["status"] == "COMPLETE" and status["smoke"] is False, "Incomplete or smoke-only result")
    require(status["total"] == status["completed"] == len(eligible) * 5, "Wrong trial total")
    require(status["population"] == 100 and status["eligible_inputs"] == len(eligible), "Wrong population status")
    require(sha(folder / "trials.jsonl") == status["trials_sha256"], "Trials file hash mismatch")
    require(runtime["registry_sha256"] == sha(registry / "manifest.json"), "Runtime registry mismatch")
    require(runtime["runner_sha256"] == sha(causal / "code/scripts/run_enumeration_fresh_read.py"), "Runner mismatch")
    require(runtime["backend"] == cfg["attention_backend"] and runtime["max_new_tokens"] == cfg["read"]["max_new_tokens"],
            "Runtime does not match frozen protocol")
    trials = matched_trials(jsonl(folder / "trials.jsonl"), eligible)
    alignment_counts = Counter()
    for k, entry in population.items():
        frozen_path = registry / entry["geometry_file"]
        require(sha(frozen_path) == entry["geometry_sha256"], "Geometry file mismatch")
        frozen = read(frozen_path)
        require(json_sha(original[k]) == frozen["source_row_sha256"], "Original row mismatch")
        require(json_sha(adapted[k]) == frozen["adapted_row_sha256"], "Adapted row mismatch")
        parsed = original[k]["trace_parse"]
        require(entry["baseline_exact_count"] == parsed["exact_count"] and
                entry["baseline_parsed_count"] == parsed["parsed_count"], "Baseline metrics mismatch")
        if k not in eligible:
            require(bool(entry.get("read_exclusion")) and "read" not in frozen, "Unexplained Read exclusion")
            continue
        geometry = frozen["read"]
        verify_masks(geometry)
        require(geometry["requires_parsed_trace_item"] is False, "Unexpected item requirement")
        selected = geometry["selected_site"]
        alignment_counts[selected["alignment_strategy"]] += 1
        # Literal prefixes permit a tokenizer-free reconstruction from saved token IDs.
        if selected["alignment_strategy"] == "literal_baseline_token_prefix":
            source = original[k]
            prefix = source["output_token_ids"][:selected["prefix_token_count"]]
            ids = list(source["input_ids"]) + prefix
            mask = list(source["attention_mask"]) + [1] * len(prefix)
            require(json_sha(ids) == geometry["input_ids_sha256"], "Literal input IDs mismatch")
            require(json_sha(mask) == geometry["attention_mask_sha256"], "Attention mask mismatch")
        for condition in CONDITIONS:
            trial = trials[(*k, condition)]
            require(trial["model"] == model and trial["mode"] == mode, "Mixed cell")
            verify_trial(trial, entry, geometry, layers=runtime["layers"], parse_total=parse_total,
                         max_new_tokens=runtime["max_new_tokens"])
    bootstrap = {"repetitions": cfg["bootstrap"]["repetitions"], "seed": cfg["bootstrap"]["seed"]}
    conditions = {}
    clean_disagreements = []
    for condition in CONDITIONS:
        rows = [trials[(*k, condition)] for k in eligible]
        rate = cluster_mean([(r["seed"], int(r["exact_count"])) for r in rows], **bootstrap)
        delta = cluster_mean([(r["seed"], int(r["exact_count"]) - int(trials[(*key(r), "clean")]["exact_count"]))
                              for r in rows], **bootstrap)
        conditions[condition] = {"accuracy": rate, "paired_change_from_clean": delta,
            "parsed": sum(r["prediction"] is not None for r in rows),
            "truncated": sum(r["generation_truncated"] for r in rows),
            "correct_but_truncated": sum(r["exact_count"] and r["generation_truncated"] for r in rows),
            "population_accuracy_bounds_unexecuted_unknown":
                [rate["numerator"] / 100, (rate["numerator"] + 100 - len(eligible)) / 100],
            "by_count": {str(n): {"correct": sum(r["exact_count"] for r in rows if r["gold_count"] == n),
                                  "total": sum(r["gold_count"] == n for r in rows)}
                         for n in cfg["read_counts"]}}
    for k, entry in eligible.items():
        clean = trials[(*k, "clean")]
        if clean["prediction"] != entry["baseline_parsed_count"]:
            clean_disagreements.append({"seed": k[0], "gold_count": k[1],
                                        "baseline_prediction": entry["baseline_parsed_count"],
                                        "replayed_prediction": clean["prediction"]})
    return {"model": model, "mode": mode, "audit": "PASS", "population": 100,
            "eligible_inputs": len(eligible), "formal_rows": len(trials),
            "baseline_correct_population": sum(bool(e["baseline_exact_count"]) for e in population.values()),
            "exclusions": [e for k, e in population.items() if k not in eligible],
            "alignment_strategies": dict(alignment_counts),
            "clean_replay_disagreements": clean_disagreements,
            "comparison_role": ("prompted_FOUND_auxiliary_not_natural_Native" if model == "Gemma4-E4B" and mode == "thinking"
                                else "fresh_native_reference" if mode == "thinking" else "primary_enumeration"),
            "runtime": runtime, "conditions": conditions}
