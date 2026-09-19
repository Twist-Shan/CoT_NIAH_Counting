"""Audit frozen Retrieve queries, persistent head hooks and seed-level effects."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v6.read_audit import read, require, sha, json_sha, jsonl
from realistic_niah_v6.fresh_behavior_audit import unique_grid, verify_generation, seed_mean, load_frozen_city_scorer
from analyze_enumeration_fresh_read import frozen_parser
from analyze_enumeration_fresh_relay import verify_inventory
from analyze_enumeration_fresh_answer_patch import load_module


def trial_key(row):
    return row["query_id"], row["dose_k"], row["condition"], row["repeat"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    inv, dependency = verify_inventory(a.source), verify_inventory(a.inputs)
    require(sha(a.inputs / "backup_manifest.json") == inv["dependency"]["manifest_sha256"], "Input backup mismatch")
    stage, registry = a.source / "fresh_retrieve_gpu_v2", a.source / "fresh_retrieve_registry_v1"
    manifest, reg, cfg = read(stage / "manifest.json"), read(registry / "manifest.json"), read(stage / "protocol.json")
    for file, expected in ((stage / "protocol.json", manifest["protocol_sha256"]), (stage / "jobs.json", manifest["jobs_sha256"]),
        (registry / "manifest.json", manifest["registry_manifest_sha256"]), (a.inputs / "fresh_causal_v1/code_manifest.json", reg["causal_code_manifest_sha256"])):
        require(sha(file) == expected, "Frozen Retrieve contract mismatch")
    for name, expected in manifest["entrypoints_sha256"].items():
        require(sha(stage / "code" / name) == expected, "Frozen Retrieve code mismatch")
    compiler = load_module(stage / "code/scripts/prepare_enumeration_fresh_retrieve.py", "_retrieve_compiler")
    worker = load_module(stage / "code/scripts/run_enumeration_fresh_retrieve.py", "_retrieve_worker")
    parse_total = frozen_parser(a.inputs)
    scorer_path = a.inputs / "fresh_causal_v1/code/src/realistic_niah_v5/causal.py"
    score_city = load_frozen_city_scorer(scorer_path)
    require(not reg["uses_final_correctness_for_selection"] and not reg["uses_intervention_outcomes"], "Outcome-selected Retrieve population")
    require(not set(reg["discovery_seeds"]) & set(reg["confirmation_seeds"]), "Discovery/confirmation overlap")
    cells, all_derived = [], []
    for mode in cfg["modes"]:
        queries_by_model = {model: read(registry / model / mode / "queries.json") for model in cfg["models"]}
        selected, missing = compiler.choose_confirmation(queries_by_model, reg["confirmation_seeds"])
        require(not missing, "Missing confirmation source seeds")
        for model in cfg["models"]:
            require(selected[model] == read(registry / model / mode / "confirmation.json"), "Highest-common-count selector mismatch")
    for cell in reg["cells"]:
        model, mode = cell["model"], cell["mode"]
        folder = registry / model / mode
        source = a.inputs / "fresh_causal_v1/registries" / model / mode
        for file, expected in ((source / "manifest.json", cell["source_registry_sha256"]),
            (source / "adapted_generations.jsonl", cell["source_generations_sha256"]),
            (folder / "queries.json", cell["queries_sha256"]), (folder / "confirmation.json", cell["confirmation_sha256"])):
            require(sha(file) == expected, "Query/input hash mismatch")
        queries = read(folder / "queries.json")
        by_id = {q["query_id"]: q for q in queries}
        require(len(by_id) == len(queries), "Duplicate query ID")
        inputs = {r["request_id"]: r for r in jsonl(source / "adapted_generations.jsonl")}
        for q in queries:
            row = inputs[q["request_id"]]
            require(json_sha(row) == q["row_sha256"], "Query source row mismatch")
            require(q["frozen_anchor_role"] == cfg["anchor_roles"][model][mode], "Anchor role mismatch")
            count = q["query_output_token_index"] + 1
            ids = row["input_ids"] + row["output_token_ids"][:count]
            mask = row["attention_mask"] + [1] * count
            require(q["query_position"] == len(ids) - 1 and json_sha(ids) == q["input_ids_sha256"] and json_sha(mask) == q["attention_mask_sha256"], "Literal query input reconstruction mismatch")
            span = next(s for s in row["prompt_record_spans"] if s["city"] == q["target_city"])
            require([span["start"], span["end"]] == q["source_record_span"], "Retrieval target span mismatch")
        local = stage / "jobs/localize" / model / mode
        observations, ranking = jsonl(local / "observations.jsonl"), read(local / "ranking.json")
        loc_grid = unique_grid(observations, ["query_id"], {(q["query_id"],) for q in queries if q["split"] == "discovery"})
        for observation in loc_grid.values():
            query = by_id[observation["query_id"]]
            require(observation["query_position"] == query["query_position"] and observation["source_record_span"] == query["source_record_span"], "Localization position mismatch")
        recomputed = compiler.rank_heads(observations)
        require([(r["layer"], r["head"]) for r in recomputed] == [(r["layer"], r["head"]) for r in ranking], "Discovery head order mismatch")
        max_error = max(abs(x["score"] - y["score"]) for x, y in zip(recomputed, ranking))
        require(max_error <= 1e-12, "Discovery score mismatch")
        bank_folder = stage / "banks" / model / mode
        bank, plan = read(bank_folder / "banks.json"), read(bank_folder / "plan.json")
        require(sha(bank_folder / "banks.json") == plan["banks_sha256"] and sha(local / "observations.jsonl") == plan["discovery_observations_sha256"], "Bank evidence mismatch")
        require(plan["stage_manifest_sha256"] == sha(stage / "manifest.json") and not plan["uses_confirmation_outcomes"], "Bank provenance mismatch")
        require(compiler.freeze_banks(ranking, cfg["dose_grid"][model], reg["confirmation_seeds"], salt=f'{cfg["random_seed"]}/{model}/{mode}', repeats=cfg["random_repeats"]) == bank, "Frozen random bank not reproducible")
        confirmation = read(folder / "confirmation.json")
        expected_tasks = []
        for q in confirmation:
            base = {"query_id": q["query_id"], "request_id": q["request_id"], "seed": q["seed"]}
            expected_tasks.append({**base, "k": 0, "condition": "clean", "repeat": 0, "heads": []})
            for k in bank["ks"]:
                expected_tasks.append({**base, "k": k, "condition": "selected_bank", "repeat": 0, "heads": bank["selected_heads"][:k]})
                expected_tasks.extend({**base, "k": k, "condition": bank["random_control_by_k"][str(k)], "repeat": repeat, "heads": heads}
                    for repeat, heads in enumerate(bank["random_banks"][str(q["seed"])][str(k)]))
        require(plan["tasks"] == expected_tasks, "Behavior task grid not generated from frozen banks")
        coverage = {}
        for phase in ("behavior_smoke", "formal"):
            job = stage / "jobs" / phase / model / mode
            status, runtime = read(job / "status.json"), read(job / "runtime.json")
            require(status["status"] == "COMPLETE" and sha(job / "trials.jsonl") == status["trials_sha256"], "Behavior job incomplete or changed")
            require(runtime["stage_manifest_sha256"] == sha(stage / "manifest.json") and runtime["backend"] == cfg["attention_backend"], "Behavior runtime mismatch")
            tasks = plan["smoke_tasks"] if phase == "behavior_smoke" else expected_tasks
            trials = jsonl(job / "trials.jsonl")
            grid = unique_grid(trials, ["query_id", "dose_k", "condition", "repeat"],
                {(t["query_id"], t["k"], t["condition"], t["repeat"]) for t in tasks})
            require(len(grid) == status["completed_rows"] == status["expected_rows"], "Behavior row count mismatch")
            for task in tasks:
                row = grid[task["query_id"], task["k"], task["condition"], task["repeat"]]
                query = by_id[task["query_id"]]
                worker.validate_trial(row, task, query, row["prefill_zeroed_head_l2_by_layer"])
                require(row["status"] == "ok" and row["trial_complete"] and row["model_label"] == model and row["mode"] == mode, "Behavior row identity/status mismatch")
                require(row["plan_sha256"] == sha(bank_folder / "plan.json") and row["request_id"] == query["request_id"], "Behavior query/plan hash mismatch")
                require(row["free_generation_max_new_tokens"] == cfg["max_new_tokens"] and row["head_ablation_tensor_site"] == "attention_output_projection_input_pre_o", "Intervention budget/site mismatch")
                verify_generation(row, cfg["max_new_tokens"])
                ids, target = row["generated_token_ids"], query["target_token_ids"]
                offset = query["target_output_token_start"] - query["query_output_token_index"] - 1
                require(offset >= 0, "Target city precedes query")
                at_offset = ids[offset:offset + len(target)] == target
                prefix = offset == 0 and at_offset
                require(row["generated_target_city_exact_at_registered_path_offset"] == at_offset and row["generated_exact_target_city_token_prefix"] == prefix, "Target token-prefix audit mismatch")
                scored = score_city(row["completion_text"], expected_city=query["target_city"],
                    gold_cities=[r["city"] for r in inputs[query["request_id"]]["prompt_record_spans"]], exact_target_prefix=prefix)
                require(all(row[k] == v for k, v in scored.items()), "Frozen semantic-city parser mismatch")
                if phase == "formal":
                    prediction = parse_total(row["completion_text"])
                    all_derived.append({"model": model, "mode": mode, "query_id": query["query_id"], "seed": query["seed"],
                        "gold_count": query["gold_count"], "dose_k": task["k"], "condition": task["condition"], "repeat": task["repeat"],
                        "next_city_failure": int(not scored["correct_next_needle"]), "final_exact_count_failure": int(prediction != query["gold_count"]),
                        "prediction": prediction, "invalid_count": prediction not in range(1, 11), "truncated": row["generation_truncated"]})
            coverage[phase] = {"rows": len(trials), "truncated": sum(r["generation_truncated"] for r in trials)}
        derived = [r for r in all_derived if r["model"] == model and r["mode"] == mode]
        clean = {r["seed"]: r for r in derived if r["condition"] == "clean"}
        require(set(clean) == set(reg["confirmation_seeds"]), "Clean source-seed coverage")
        stats = []
        for k in bank["ks"]:
            selected = {r["seed"]: r for r in derived if r["dose_k"] == k and r["condition"] == "selected_bank"}
            control = bank["random_control_by_k"][str(k)]
            random = {s: [r for r in derived if r["seed"] == s and r["dose_k"] == k and r["condition"] == control] for s in clean}
            require(set(selected) == set(clean) and all(len(v) == cfg["random_repeats"] for v in random.values()), "Paired effect coverage")
            metrics = {}
            for metric in cfg["primary_metrics"]:
                terms = {name: [] for name in ["clean", "selected", "random", *cfg["contrasts"]]}
                for seed, baseline in clean.items():
                    c, s, r = baseline[metric], selected[seed][metric], float(np.mean([t[metric] for t in random[seed]]))
                    for name, value in {"clean": c, "selected": s, "random": r, "selected_minus_clean": s-c,
                                        "random_minus_clean": r-c, "selected_minus_random": s-r}.items():
                        terms[name].append((seed, value))
                metrics[metric] = {name: seed_mean(values) for name, values in terms.items()}
            stats.append({"dose_k": k, "random_control": control, "metrics": metrics})
        cells.append({"model": model, "mode": mode, "audit": "PASS", "coverage": coverage, "discovery_queries": len(observations),
            "confirmation_counts": [q["gold_count"] for q in confirmation], "source_seed_count": len(clean),
            "ranking_max_absolute_score_error": max_error, "random_capacity_fallbacks": bank["layer_matching_infeasible"], "dose_response": stats})
        print(json.dumps({"model": model, "mode": mode, "audit": "PASS", "coverage": coverage,
            "largest_dose": {metric: stats[-1]["metrics"][metric]["selected_minus_clean"]["estimate"] for metric in cfg["primary_metrics"]}}), flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "enumeration_fresh_retrieve_audit_v1", "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(), "cells": cells,
        "source_files_verified": len(inv["files"]), "input_files_verified": len(dependency["files"]),
        "script_sha256": sha(__file__), "helper_sha256": sha(ROOT / "src/realistic_niah_v6/fresh_behavior_audit.py"),
        "frozen_city_parser_source_sha256": sha(scorer_path), "frozen_total_parser_sha256": sha(a.inputs / "fresh_causal_v1/code/src/realistic_niah/parsing.py"),
        "source_inventory_sha256": sha(a.source / "backup_manifest.json"), "input_inventory_sha256": sha(a.inputs / "backup_manifest.json"),
        "estimand": "Within-seed random repeats averaged before paired clean-corrected failure effects; source seeds equally weighted.",
        "bootstrap": {"repetitions": 10000, "seed": 20260915, "unit": "true_source_seed"},
        "limitations": ["Saved hooks and literal input IDs audited on CPU; no GPU rerun or tokenizer decode.",
            "Next-city scoring uses the frozen Native semantic-record parser, including its fixed target-token fallback.",
            "Invalid outputs and truncations stay in all denominators; a valid answer before truncation follows the frozen Total parser.",
            "Ten seed clusters, three random repeats per seed. Pointwise intervals across fixed doses are not simultaneous confidence bands.",
            "QI uses post-marker queries; the other cells use item-end queries. Cross-mode populations and grammar remain distinct."],
        "command": sys.argv, "seconds": time.monotonic() - start}
    (a.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (a.output / "parsed_outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_derived), encoding="utf-8")
    lines = ["# Enumeration Retrieve audit", "", "Frozen query selection, discovery head ranking, deterministic banks, persistent pre-O hooks, raw-token flags and semantic-city parsing all passed.",
             "Effects are changes in failure probability. Random repeats are averaged within seed; intervals resample the ten source seeds.", "",
             "| Model / mode | K | Endpoint | Clean failure | Selected - clean | Random - clean | Selected - random (95% CI) |", "| --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for cell in cells:
        for dose in cell["dose_response"]:
            for metric, terms in dose["metrics"].items():
                effect = terms["selected_minus_random"]
                lo, hi = effect["ci95"]
                lines.append(f'| {cell["model"]} / {cell["mode"]} | {dose["dose_k"]} | {metric} | {100*terms["clean"]["estimate"]:.1f}% | {100*terms["selected_minus_clean"]["estimate"]:.1f} pp | {100*terms["random_minus_clean"]["estimate"]:.1f} pp | {100*effect["estimate"]:.1f} [{100*lo:.1f}, {100*hi:.1f}] pp |')
    lines.extend(["", *[f"- {x}" for x in report["limitations"]]])
    (a.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
