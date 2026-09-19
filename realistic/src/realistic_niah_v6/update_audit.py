"""CPU checks and paired statistics for the frozen fresh Enumeration Update."""
from __future__ import annotations
from collections import Counter
import math
from pathlib import Path, PurePosixPath

from .aligned_reporting import continuation_summary
from .read_audit import cluster_mean, json_sha, read, require, sha
from .update_n10 import validate_selection

CONDITIONS = ("receiver_self", "native_donor", "donor_to_receiver")
SCOPES = ("endpoint", "four_token_tail", "item_span")


def jsonl(path):
    # JSON may contain literal U+2028/U+2029; those are not JSONL delimiters.
    import json
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def matched_trials(rows, seeds):
    keys = [(r["seed"], r["condition"]) for r in rows]
    expected = {(seed, condition) for seed in seeds for condition in CONDITIONS}
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            "Incomplete or duplicate three-condition Update grid")
    return dict(zip(keys, rows))


def verify_trial(trial, raw, hooks, *, cities, parser, layer, k, direction, scope, max_tokens):
    j = k - 1 if direction == "forward" else k + 1
    expected = {"layer": layer, "gold_count": 10, "donor_occurrence_k": k,
                "receiver_occurrence_j": j, "donor_successor": k + 1, "receiver_successor": j + 1,
                "patch_scope": "item_span" if scope == "item_span" else "fixed_suffix",
                "patch_applications": 1}
    for field, value in expected.items():
        require(trial[field] == value, f"Update geometry mismatch: {field}")
    width = trial["patch_width"]
    require(type(width) is int and width > 0, "Invalid patch width")
    if scope != "item_span":
        require(width == trial["requested_patch_width"] == (4 if scope == "four_token_tail" else 1),
                "Wrong fixed patch width")
    for field in ("donor_item_coverage", "receiver_item_coverage"):
        require(0 < trial[field] <= 1, "Invalid item coverage")
    for field in ("seed", "condition", "completion_text", "generated_token_count", "stopped_on_eos", "generation_truncated"):
        require(trial[field] == raw[field], f"Raw-generation mismatch: {field}")
    tokens = raw["generated_token_ids"]
    require(all(type(x) is int and x >= 0 for x in tokens), "Invalid generated token ID")
    require(0 < len(tokens) == raw["generated_token_count"] <= max_tokens, "Invalid token count")
    eos = tokens[-1] in raw["generation_eos_token_ids"]
    require(type(raw["stopped_on_eos"]) is bool and raw["stopped_on_eos"] == eos, "EOS flag mismatch")
    require(type(raw["generation_truncated"]) is bool and
            raw["generation_truncated"] == (len(tokens) == max_tokens and not eos), "Truncation flag mismatch")
    require(raw["query_position"] == trial["shared_commit_position"], "Query position mismatch")
    require(len(hooks) == 2, "Missing scoring/generation hook evidence")
    for hook in hooks:
        require(hook["seed"] == trial["seed"] and hook["layer"] == layer and
                hook["site"] == raw["query_position"] and hook["width"] == width and hook["applications"] == 1,
                "Hook geometry mismatch")
        norm = hook["delta_norm"]
        require(math.isfinite(norm) and (norm > 0 if trial["condition"] == "donor_to_receiver" else norm == 0),
                "Invalid actual patch norm")
    require(hooks[0]["input_ids_sha256"] == hooks[1]["input_ids_sha256"], "Scoring/generation input mismatch")
    require(trial["realized_patch_delta_norm"] == hooks[0]["delta_norm"], "Saved patch norm mismatch")
    parsed = parser(trial["completion_text"], cities)
    for field, value in parsed.items():
        require(trial[field] == value, f"Frozen city parser mismatch: {field}")
    first = parsed["first_generated_known_city_ordinal"]
    require(trial["greedy_donor_successor_adoption"] == (first == k + 1) and
            trial["greedy_receiver_successor_retention"] == (first == j + 1), "Successor flag mismatch")
    scores = trial["sum_logprob_scores"]
    require(len(scores) == 2 and all(math.isfinite(v) for v in scores + trial["mean_logprob_scores"]),
            "Nonfinite likelihood")
    require(math.isclose(scores[1] - scores[0], trial["donor_vs_receiver_sum_logodds"], rel_tol=1e-7, abs_tol=1e-7),
            "Likelihood contrast mismatch")


def summarize(rows, *, bootstrap):
    """Keep every trial; paired differences resample whole source-seed clusters."""
    bootstrap = {key: bootstrap[key] for key in ("repetitions", "seed")}
    results = []
    for scope in SCOPES:
        for direction in ("forward", "backward"):
            for k in (4, 6, 8, "all"):
                selected = [r for r in rows if r["audit_scope"] == scope and r["audit_direction"] == direction
                            and (k == "all" or r["donor_occurrence_k"] == k)]
                by_condition = {c: [r for r in selected if r["condition"] == c] for c in CONDITIONS}
                self_rows = {(r["seed"], r["donor_occurrence_k"]): r for r in by_condition["receiver_self"]}
                conditions = {}
                for condition, group in by_condition.items():
                    conditions[condition] = {
                        "trials": len(group), "truncated": sum(r["generation_truncated"] for r in group),
                        "no_known_city": sum(not r["generated_known_city_ordinals_any_surface"] for r in group),
                        "receiver_successor_retention": cluster_mean(
                            [(r["seed"], int(r["greedy_receiver_successor_retention"])) for r in group], **bootstrap),
                        "continuation": {str(h): continuation_summary(group, h, draws=bootstrap["repetitions"],
                                                                       random_seed=bootstrap["seed"])
                                         for h in (1, 2, 3, 4)}}
                target = by_condition["donor_to_receiver"]
                paired_adoption, paired_logodds = [], []
                for row in target:
                    control = self_rows[row["seed"], row["donor_occurrence_k"]]
                    paired_adoption.append((row["seed"], int(row["greedy_donor_successor_adoption"]) -
                                            int(control["greedy_donor_successor_adoption"])))
                    paired_logodds.append((row["seed"], row["donor_vs_receiver_sum_logodds"] -
                                            control["donor_vs_receiver_sum_logodds"]))
                results.append({"scope": scope, "direction": direction, "donor_k": k, "conditions": conditions,
                                "paired_target_minus_self_adoption": cluster_mean(paired_adoption, **bootstrap),
                                "paired_target_minus_self_logodds": cluster_mean(paired_logodds, **bootstrap)})
    return results


def verify_n10_runtime(runtime, previous, *, selection_sha256, runner_sha256, seeds, layer):
    require(runtime["n10_selection_manifest_sha256"] == selection_sha256,
            "Runtime used a different N10 selection")
    require(runtime["runner_sha256"] == runner_sha256 and runtime["patch_layer_zero_based"] == layer,
            "Runtime used a different N10 wrapper or layer")
    require(runtime["seeds"] == seeds, "N10 runtime seed order differs")
    command = runtime["command"]
    require(command.count("--selection-manifest") == 1, "Missing or duplicate selection argument")
    index = command.index("--selection-manifest")
    require(index + 1 < len(command) and
            PurePosixPath(command[index + 1]).parts[-2:] == ("fresh_n10_update_v1", "selection_manifest.json"),
            "Wrong N10 selection command path")
    for field in ("model_revision", "model_source_sha256", "dtype", "backend", "torch", "transformers",
                  "layers", "jobs", "attention_readout", "all_three_conditions_generate"):
        require(runtime[field] == previous[field], f"N10 numerical environment changed: {field}")


def audit_cell(root, cell, manifest, cfg, parser, source_root, *, inputs_root=None,
               selection=None, phase="formal", statistics=True):
    require(phase in ("formal", "smoke") and (selection is not None or phase == "formal"),
            "Unsupported Update audit phase")
    inputs_root = inputs_root or root
    update = root / ("fresh_n10_update_v1" if selection is not None else "fresh_native_update_v1")
    causal = inputs_root / "fresh_causal_v1"

    def relocated(path):
        relative = PurePosixPath(path).relative_to(PurePosixPath(source_root))
        local = root / relative
        return local if local.exists() else inputs_root / relative

    model, mode = cell["model"], cell["mode"]
    registry, cohort_path = relocated(cell["registry"]), relocated(cell["cohort"])
    registered, ledger = read(registry / "manifest.json"), read(registry / "ledger.json")
    require(sha(registry / "manifest.json") == cell["registry_sha256"] and sha(cohort_path) == cell["cohort_sha256"],
            "Cell provenance mismatch")
    for name, key in (("ledger.json", "ledger_sha256"), ("adapted_generations.jsonl", "adapted_generations_sha256")):
        require(sha(registry / name) == registered[key], f"Frozen registry mismatch: {name}")
    cohort = read(cohort_path)
    require(cohort["selection_used_final_correctness"] is False and cohort["intervention_outcomes_accessed"] is False,
            "Outcome-screened primary cohort")
    baseline = read(inputs_root / "fresh_v1/manifest.json")
    candidates = sorted(baseline["confirmation_candidates"])
    n10 = {r["seed"]: r for r in ledger if r["gold_count"] == 10}
    require(len(n10) == sum(r["gold_count"] == 10 for r in ledger), "Duplicate N10 registry")
    protocol = read(update / "protocol.json")
    quota = protocol["confirmation_quota"] if selection is not None else protocol["update"]["confirmation_quota_per_model_mode"]
    chosen = [s for s in candidates if n10[s]["update_eligible"]][:quota]
    require(chosen == cell["selected_seeds"] and len(chosen) == quota, "Primary sampling differs from frozen order")
    if selection is None:
        require(not set(cell["reused_seeds"]) & set(cell["missing_seeds"]) and
                sorted(cell["reused_seeds"] + cell["missing_seeds"]) == chosen, "Bad reuse/new partition")
    else:
        require(cell["confirmation_seeds"] == chosen == cohort["models"][model]["confirmation_seeds"],
                "N10 confirmation differs from frozen input cohort")
    inputs = {r["seed"]: r for r in jsonl(registry / "adapted_generations.jsonl") if r["gold_count"] == 10}
    for seed in chosen:
        entry = n10[seed]
        geometry = relocated(PurePosixPath(cell["registry"]) / entry["geometry_file"])
        require(sha(geometry) == entry["geometry_sha256"] and
                json_sha(inputs[seed]) == read(geometry)["adapted_row_sha256"], "Adapted model input mismatch")
        require(len(inputs[seed]["gold_records"]) == 10, "Wrong city registry length")
    runtimes, sources = [], []
    layer = (validate_selection(selection, model, mode, chosen) if selection is not None else
             cfg["update"]["layers_one_based"][model][mode] - 1)
    if phase == "smoke":
        chosen, quota = chosen[:1], 1
    source_groups = [(chosen, update / "jobs" / phase, cohort_path)] if selection is not None else (
            (cell["reused_seeds"], causal / "gpu_jobs/formal/update", causal / "cohorts.json"),
            (cell["missing_seeds"], update / "jobs/formal", cohort_path))
    for seeds, parent, expected_cohort in source_groups:
        if not seeds:
            continue
        source = parent / model / mode
        runtime, status = read(source / "runtime.json"), read(source / "status.json")
        require(status["status"] == "COMPLETE" and status["smoke"] == (phase == "smoke"), "Incomplete Update source")
        require(runtime["model"] == model and runtime["mode"] == mode and
                runtime["registry_sha256"] == cell["registry_sha256"] and
                runtime["cohorts_sha256"] == sha(expected_cohort), "Wrong source population")
        runner = (update if selection is not None else causal) / "code/scripts/run_enumeration_fresh_update.py"
        require(runtime["runner_sha256"] == sha(runner) and
                runtime["patch_layer_zero_based"] == layer and runtime["backend"] == cfg["attention_backend"] and
                runtime["attention_readout"] is False and runtime["all_three_conditions_generate"] is True,
                "Wrong numerical kernel or settings")
        require(set(seeds) <= set(runtime["seeds"]), "Selected seed not executed")
        if selection is not None:
            verify_n10_runtime(runtime, read(relocated(cell["source_runtime"])),
                               selection_sha256=sha(update / "selection_manifest.json"), runner_sha256=sha(runner),
                               seeds=chosen, layer=layer)
            require(status["completed_rows"] == quota * 54 and status["completed_jobs"] == 18,
                    "Incomplete N10 worker grid")
        runtimes.append(runtime)
        sources.append((seeds, source))
    for other in runtimes[1:]:
        for field in ("model_revision", "model_source_sha256", "dtype", "backend", "torch", "transformers",
                      "layers", "patch_layer_zero_based", "jobs", "attention_readout", "all_three_conditions_generate"):
            require(other[field] == runtimes[0][field], f"Reused/new runtime mismatch: {field}")
    rows, widths, geometry_counts = [], Counter(), Counter()
    for k in (4, 6, 8):
        for direction in ("forward", "backward"):
            for scope in SCOPES:
                name = f"k{k}_{direction}_{scope}"
                folder = update / ("primary" if phase == "formal" else "jobs/smoke") / model / mode / name
                merged = {n: [] for n in ("trials.jsonl", "raw_generations.jsonl", "prefill_hooks.jsonl")}
                geometries = {}
                for seeds, source in sources:
                    audit = read(source / name / "technical_audit.json")
                    require(audit["status"] == "PASS" and audit["backend_after"] == "sdpa", "Failed source audit")
                    for geometry in jsonl(source / name / "geometry_audit.jsonl"):
                        if geometry["seed"] not in seeds:
                            continue
                        require(geometry["seed"] not in geometries, "Duplicate source geometry")
                        geometries[geometry["seed"]] = geometry
                    for file, key in (("trials.jsonl", "trials_sha256"), ("raw_generations.jsonl", "raw_sha256"),
                                      ("prefill_hooks.jsonl", "hooks_sha256")):
                        require(sha(source / name / file) == audit[key], "Source output changed")
                        merged[file].extend(r for r in jsonl(source / name / file) if r["seed"] in seeds)
                for file, original in merged.items():
                    require(jsonl(folder / file) == original, f"Primary differs from chosen source rows: {name}/{file}")
                trials, raw, hooks = (merged[n] for n in ("trials.jsonl", "raw_generations.jsonl", "prefill_hooks.jsonl"))
                matched = matched_trials(trials, chosen)
                require(set(geometries) == set(chosen), "Missing source geometry")
                require(len(raw) == len(trials) and len(hooks) == 2 * len(trials), "Missing raw/hook rows")
                receiver_hash = {}
                for i, (trial, generation) in enumerate(zip(trials, raw)):
                    cities = [str(v["city"]) for v in inputs[trial["seed"]]["gold_records"]]
                    require(trial["request_id"] == inputs[trial["seed"]]["request_id"], "Wrong source request")
                    geometry = geometries[trial["seed"]]
                    require(geometry["aligned_absolute_site"] == trial["shared_commit_position"] and
                            geometry["effective_patch_width"] == trial["patch_width"] and
                            geometry["endpoint_aligned"] is True and geometry["hidden_state_resampling"] is False,
                            "Saved patch geometry mismatch")
                    role = "donor" if trial["condition"] == "native_donor" else "receiver"
                    require(trial["shared_commit_token_id"] == geometry[f"{role}_commit_token_id"],
                            "Commit token differs from actual condition prefix")
                    for field in ("receiver_item_coverage", "donor_item_coverage", "equal_length_complete_item"):
                        require(trial[field] == geometry[field], "Item coverage mismatch")
                    pair = hooks[2*i:2*i+2]
                    verify_trial(trial, generation, pair, cities=cities, parser=parser, layer=layer,
                                 k=k, direction=direction, scope=scope, max_tokens=cfg["update"]["max_new_tokens"])
                    receiver_hash[trial["seed"], trial["condition"]] = pair[0]["input_ids_sha256"]
                    widths[scope, trial["patch_width"]] += 1
                    rows.append({**trial, "audit_scope": scope, "audit_direction": direction})
                for seed in chosen:
                    require(receiver_hash[seed, "receiver_self"] == receiver_hash[seed, "donor_to_receiver"],
                            "Self/Target input not paired")
                    # Equal absolute sites need not have the same surface token.
                    # Target uses the receiver input; native donor has its own prefix.
                    for field in ("shared_commit_position", "patch_width"):
                        require(len({matched[seed, c][field] for c in CONDITIONS}) == 1, "Three conditions not geometry matched")
                    require(matched[seed, "receiver_self"]["shared_commit_token_id"] ==
                            matched[seed, "donor_to_receiver"]["shared_commit_token_id"], "Self/Target commit mismatch")
                    geometry = geometries[seed]
                    require(geometry["deletion_avoids_prompt_records"] and geometry["deletion_avoids_special_tokens"] and
                            geometry["deletion_before_answer"], "Alignment deleted protected input")
                    surface = geometry["receiver_commit_token_id"] == geometry["donor_commit_token_id"]
                    require(geometry["surface_token_matched"] == surface, "Incorrect surface matching label")
                    geometry_counts[scope, "same_endpoint_token" if surface else "different_endpoint_token"] += 1
    require(len(rows) == quota * 54, "Incomplete primary cell")
    result = {"model": model, "mode": mode, "audit": "PASS", "phase": phase,
            "formal_rows": len(rows) if phase == "formal" else 0, "smoke_rows": len(rows) if phase == "smoke" else 0,
            "layer_one_based": layer + 1, "selected_seeds": chosen, "eligible_candidates": sum(n10[s]["update_eligible"] for s in candidates),
            "confirmation_inputs_reused": selection is not None,
            "patch_width_rows": {f"{scope}/{width}": number for (scope, width), number in sorted(widths.items())},
            "geometry_counts": {f"{scope}/{label}": number for (scope, label), number in sorted(geometry_counts.items())},
            "groups": summarize(rows, bootstrap=cfg["bootstrap"]) if statistics else []}
    if selection is None:
        result.update(reused_seeds=cell["reused_seeds"], new_seeds=cell["missing_seeds"])
    else:
        result.update(reused_intervention_seeds=[], newly_executed_intervention_seeds=chosen)
    return result
