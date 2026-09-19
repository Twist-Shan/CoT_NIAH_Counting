"""Audit saved relay hooks/generations and compute Native-aligned score-margin damage."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v6.read_audit import read, require, sha
from realistic_niah_v6.aligned_reporting import relay_point, relay_summary
from analyze_enumeration_fresh_read import frozen_parser


def verify_inventory(source):
    inventory = read(source / "backup_manifest.json")
    for name, record in inventory["files"].items():
        path = source / name
        require(path.resolve().is_relative_to(source.resolve()), "Unsafe inventory path")
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                f"Archived member changed: {name}")
    return inventory


def outcome_audit(row, parse_total, budget):
    raw = row["raw_generation"]
    ids = raw["generated_token_ids"]
    require(len(ids) == raw["generated_token_count"] == row["generated_token_count"] <= budget,
            "Generation length mismatch")
    eos = bool(ids and ids[-1] in raw["generation_eos_token_ids"])
    require(raw["stopped_on_eos"] == eos, "EOS mismatch")
    require(raw["generation_truncated"] == row["generation_truncated"] == (len(ids) >= budget and not eos),
            "Truncation mismatch")
    prediction = parse_total(raw["full_answer_text"])
    gold = row["gold_count"]
    require(prediction == row["prediction"] and (prediction == gold) == row["exact_count"], "Frozen parser mismatch")
    require((prediction is None) == row["invalid_count_output"], "Invalid answer flag mismatch")
    counts = np.asarray([int(x) for x in row["candidate_counts"].split(",")])
    scores = np.asarray([float(x) for x in row["candidate_log_scores"].split(",")])
    probs = np.asarray([float(x) for x in row["candidate_probabilities"].split(",")])
    require(counts.tolist() == list(range(1, 11)) and np.isfinite(scores).all(), "Candidate score coverage")
    computed_probs = np.exp(scores - scores.max())
    computed_probs /= computed_probs.sum()
    require(np.allclose(probs, computed_probs, atol=2e-6, rtol=2e-6), "Candidate normalization mismatch")
    margin = scores[gold - 1] - np.max(scores[counts != gold])
    require(abs(margin - row["correct_count_margin"]) < 2e-5, "Margin definition mismatch")
    require(abs(scores[gold - 1] - row["correct_count_log_score"]) < 2e-5, "Correct log score mismatch")
    expected = float(counts @ computed_probs)
    require(abs(expected - row["expected_count"]) < 2e-5, "Expected count mismatch")
    require(abs(-abs(expected - gold) - row["expected_count_utility"]) < 2e-5, "Expected count utility mismatch")


def summarize(rows, cfg):
    seeds = sorted({r["seed"] for r in rows})
    if len(seeds) < 2:
        stat = relay_point(np.mean([r["natural_damage"] for r in rows]),
                           np.mean([r["remaining_signed_damage"] for r in rows]))
        stat.update(source_seed_count=len(seeds), directed_pair_count=len(rows), ci95=None,
                    ratio_ci_status="fewer_than_two_independent_seeds", interpretation="descriptive_only")
    else:
        stat = relay_summary(rows, draws=cfg["bootstrap"]["repetitions"], random_seed=cfg["bootstrap"]["seed"])
        interval = stat["ci95"]["natural_damage"]
        if interval[0] <= 0 <= interval[1]:
            stat["ratio_ci_status"] = "natural_damage_interval_crosses_zero"
            for key in ("signed_reduction", "absolute_reduction"):
                stat["ci95"].pop(key, None)
            stat["interpretation"] = "Ratio point is descriptive; denominator sign is uncertain. Use signed damage and interaction."
    stat["source_seeds"] = seeds
    stat["sign_reversal_after_reset"] = stat["natural_damage"] * stat["remaining_signed_damage"] < 0
    return stat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Verified relay archive extraction")
    p.add_argument("--inputs", type=Path, required=True, help="Verified Update archive extraction")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    inventory = verify_inventory(a.source)
    dependency = verify_inventory(a.inputs)
    require(sha(a.inputs / "backup_manifest.json") == inventory["dependency"]["manifest_sha256"], "Input dependency mismatch")
    stage = a.source / "fresh_relay_native_gpu_v1"
    registry = a.source / "fresh_relay_native_registry_v1"
    manifest, cfg = read(stage / "manifest.json"), read(stage / "protocol.json")
    require(sha(stage / "protocol.json") == manifest["protocol_copy_sha256"], "Protocol hash")
    require(sha(stage / "job_manifest.json") == manifest["job_manifest_sha256"], "Job contract hash")
    require(sha(registry / "common_cohorts.json") == manifest["cohorts_sha256"], "Cohort hash")
    for name, expected in manifest["entrypoints_sha256"].items():
        require(sha(stage / "code" / name) == expected, "Frozen relay entrypoint changed")
    require(sha(a.inputs / "fresh_causal_v1/code_manifest.json") == manifest["causal_code_manifest_sha256"], "Causal dependency hash")
    worker_path = stage / "code/scripts/run_enumeration_fresh_relay.py"
    spec = importlib.util.spec_from_file_location("_frozen_relay_worker", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    parse_total = frozen_parser(a.inputs)
    cohorts = read(registry / "common_cohorts.json")
    cells, all_effects = [], []
    for model in cfg["models"]:
        for mode in cfg["modes"]:
            folder = registry / model / mode
            pm = read(folder / "manifest.json")
            require(sha(folder / "manifest.json") == cohorts["manifests"][f"{model}/{mode}"], "Pair manifest hash")
            require(sha(folder / "pairs.json") == pm["pairs_sha256"], "Pairs hash")
            require(not pm["final_correctness_used_for_selection"] and not pm["relay_intervention_outcomes_read"], "Selection gate changed")
            require(sha(a.inputs / "fresh_causal_v1/registries" / model / mode / "manifest.json") == pm["source_registry_sha256"], "Source input registry mismatch")
            pairs = {r["pair_id"]: r for r in read(folder / "pairs.json")}
            layers = cfg["layers_zero_based"][model]
            source_layers = list(range(layers["source"], layers["relay"]))
            totals = {}
            for phase in ("smoke", "formal"):
                job = stage / "jobs" / phase / model / mode
                status, runtime = read(job / "status.json"), read(job / "runtime.json")
                require(status["status"] == "COMPLETE" and sha(job / "trials.jsonl") == status["trials_sha256"], "Job completion/hash mismatch")
                require(runtime["stage_manifest_sha256"] == sha(stage / "manifest.json") and runtime["backend"] == cfg["attention_backend"], "Runtime mismatch")
                tasks = worker.selected_relay_tasks(cohorts, model, cfg["widths"], smoke=phase == "smoke", mode=mode)
                rows = [json.loads(line) for line in (job / "trials.jsonl").read_text().splitlines()]
                require(len(rows) == status["completed_rows"] == len(tasks) * 6, "Grid size mismatch")
                grouped = defaultdict(list)
                for row in rows:
                    grouped[(row["width"], row["pair_id"])].append(row)
                require(set(grouped) == {(t["width"], t["pair_id"]) for t in tasks}, "Trial pair set mismatch")
                for (width, pair_id), panel in grouped.items():
                    pair = pairs[pair_id]
                    worker.validate_relay_panel(panel, [r["source_hook_audit"] for r in panel],
                                                [r["raw_generation"] for r in panel], pair, width, source_layers)
                    for row in panel:
                        for key in ("request_id", "seed", "gold_count", "receiver_occurrence", "donor_occurrence", "donor_offset"):
                            require(row[key] == pair[key], f"Pair metadata mismatch: {key}")
                        require(row["model_label"] == model and row["mode"] == mode and row["status"] == "ok", "Row identity/status mismatch")
                        require(row["source_patch_layers"] == source_layers and row["relay_layer"] == layers["relay"], "Layer contract mismatch")
                        require(row["pair_registry_sha256"] == pm["pairs_sha256"] and row["registry_sha256"] == pair["answer_registry"]["registry_sha256"], "Row registry mismatch")
                        outcome_audit(row, parse_total, cfg["max_new_tokens"])
                    indexed = {(r["source_condition"], r["relay_condition"]): r for r in panel}
                    self_natural = indexed[("self_patch", "natural_relay")]
                    for reset in cfg["relay_conditions"]:
                        self_reset = indexed[("self_patch", reset)]
                        require(self_reset["candidate_log_scores"] == self_natural["candidate_log_scores"] and
                                self_reset["raw_generation"] == self_natural["raw_generation"], "Self reset changed clean output")
                    if phase == "formal":
                        natural = self_natural["correct_count_margin"] - indexed[("full_donor_patch", "natural_relay")]["correct_count_margin"]
                        for reset in cfg["relay_conditions"][1:]:
                            remaining = indexed[("self_patch", reset)]["correct_count_margin"] - indexed[("full_donor_patch", reset)]["correct_count_margin"]
                            all_effects.append({"model": model, "mode": mode, "width": width, "pair_id": pair_id,
                                               "seed": pair["seed"], "gold_count": pair["gold_count"], "donor_offset": pair["donor_offset"],
                                               "reset": reset, "natural_damage": natural, "remaining_signed_damage": remaining})
                totals[phase] = {"rows": len(rows), "invalid_answers": sum(r["invalid_count_output"] for r in rows),
                                 "truncated": sum(r["generation_truncated"] for r in rows)}
            effects = [r for r in all_effects if r["model"] == model and r["mode"] == mode]
            width_pairs = {w: {r["pair_id"] for r in effects if r["width"] == w} for w in cfg["widths"]}
            common = set.intersection(*width_pairs.values())
            estimates = []
            for scope in ("all_eligible", "width_common_pairs"):
                for width in cfg["widths"]:
                    for reset in cfg["relay_conditions"][1:]:
                        subset = [r for r in effects if r["width"] == width and r["reset"] == reset and (scope == "all_eligible" or r["pair_id"] in common)]
                        if subset:
                            estimates.append({"scope": scope, "width": width, "reset": reset, **summarize(subset, cfg)})
            cells.append({"model": model, "mode": mode, "audit": "PASS", "coverage": totals, "estimates": estimates})
            print(json.dumps({"model": model, "mode": mode, "audit": "PASS", "coverage": totals}), flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "enumeration_fresh_relay_audit_v1", "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
              "metric": "correct_count_margin = S(N) - max_{c != N} S(c), Native Appendix definition",
              "estimand": "Equal-weight source-seed means of within-pair self-minus-donor margin damage; paired seed bootstrap, ratio of means.",
              "bootstrap": cfg["bootstrap"], "primary_width": cfg["primary_width"], "cells": cells,
              "relay_files_verified": len(inventory["files"]), "input_files_verified": len(dependency["files"]),
              "relay_inventory_sha256": sha(a.source / "backup_manifest.json"), "input_inventory_sha256": sha(a.inputs / "backup_manifest.json"),
              "script_sha256": sha(__file__), "statistics_module_sha256": sha(ROOT / "src/realistic_niah_v6/aligned_reporting.py"),
              "limitations": ["Saved-token and hook audit, no GPU rerun or tokenizer decoding.",
                "Eight-token coverage is limited by literal item length; one-seed estimates have no bootstrap interval.",
                "If the natural-damage interval crosses zero, ratio intervals are suppressed and points remain descriptive.",
                "Four- versus eight-token effects must use width_common_pairs; cross-mode/model cohorts differ."],
              "command": sys.argv, "seconds": time.monotonic() - start}
    (a.output / "audit.json").write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (a.output / "paired_effects.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_effects), encoding="utf-8")
    lines = ["# Enumeration terminal-relay audit", "", "Score-margin damage follows the current Thinking appendix. All saved trial grids, source/reset hooks, raw-answer parsing and score arithmetic passed.",
             "Intervals resample independent source seeds (10,000 draws). Four-token and eight-token rows below have different eligible populations; matched-width analyses are in audit.json.", "",
             "| Model / mode | Width | Pairs / seeds | Natural damage | Remaining damage after suffix reset | Signed reduction (95% CI) |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for cell in cells:
        for stat in cell["estimates"]:
            if stat["scope"] != "all_eligible" or stat["reset"] != "post_terminal_suffix_clean_reset":
                continue
            point = stat["signed_reduction"]
            interval = (stat["ci95"] or {}).get("signed_reduction")
            ratio = "not estimable" if point is None else f"{100*point:.1f}%"
            ratio += f" ({100*interval[0]:.1f}, {100*interval[1]:.1f})" if interval else " (descriptive)"
            lines.append(f'| {cell["model"]} / {cell["mode"]} | {stat["width"]} | {stat["directed_pair_count"]} / {stat["source_seed_count"]} | {stat["natural_damage"]:.4f} | {stat["remaining_signed_damage"]:.4f} | {ratio} |')
    lines.extend(["", *[f"- {x}" for x in report["limitations"]]])
    (a.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(a.output), "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
