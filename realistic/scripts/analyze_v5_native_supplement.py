#!/usr/bin/env python3
"""Audit downloaded Native supplement results and build a reproducible report.

No GPU inference is performed. Incomplete or mismatched runs are rejected.
Historical data and the frozen execution bundle are never modified.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v5.native_supplement import (
    ANSWER_SEEDS, LAYERS, MODELS, PROGRESS_SEEDS, compare_replay, digest,
    read_rows, summarize_backward, validate_answer, validate_pairs,
    validate_progress, verify_bundle, write_json,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_out(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_csv_values(expected: Path, actual: Path) -> None:
    left, right = pd.read_csv(expected), pd.read_csv(actual)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, atol=1e-12, rtol=1e-12)


def analyze(bundle: Path, runs: Path, out: Path) -> dict:
    plan = verify_bundle(bundle)
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def record(path: Path):
        sources[str(path.resolve())] = digest(path)
        return path

    record(bundle / "plan.json")
    record(Path(__file__))
    record(ROOT / "src/realistic_niah_v5/native_supplement.py")
    spec = importlib.util.spec_from_file_location(
        "frozen_native_answer_analysis", bundle / "code/scripts/analyze_v5_answer_query_layer_sweep.py"
    )
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    dense_summary, layer_export, audits, common = {}, [], {}, []
    for model, name in zip(MODELS, ("qwen_dense", "gemma_dense")):
        run, inputs = runs / name, bundle / "inputs" / model
        completion = load_json(record(run / "completion_audit.json"))
        manifest = load_json(record(run / "run_manifest.json"))
        assert completion["status"] == "PASS" and completion["model"] == model
        assert completion["task"] == "answer_dense" and completion["fresh_confirmation"] is False
        assert completion["plan_sha256"] == manifest["plan_sha256"] == plan["plan_sha256"]
        pairs = read_rows(record(inputs / "pairs.jsonl"))
        common.append(sorted(validate_pairs(pairs, model)))
        trials = run / "dense/trials.jsonl"
        record(trials)
        assert completion["trials_sha256"] == digest(trials)
        rows = read_rows(trials)
        grid = validate_answer(rows, pairs, list(range(LAYERS[model])))
        assert grid["trials"] == completion["trials"] == 80 * LAYERS[model]
        replay = compare_replay(
            read_rows(record(inputs / "historical_answer_trials.jsonl")),
            read_rows(record(run / "replay/trials.jsonl")),
        )
        assert replay == load_json(record(run / "historical_replay_audit.json"))
        assert replay["status"] == "PASS" and replay["compared_trials"] == 640
        for phase in ("replay", "missing"):
            loaded = load_json(record(run / phase / "loaded_model.json"))
            assert loaded["layers"] == LAYERS[model] and loaded["training"] is False
            assert loaded["effective_attention_backend"] == plan["attention_backend"]
            assert loaded["dtype"] == "torch.bfloat16"
            assert loaded["model_spec"] == plan["models"][model]["model_spec"]
            assert loaded["observed_revision"] in (None, loaded["model_spec"]["revision"])
            assert load_json(record(run / phase / "process_status.json"))["exit_code"] == 0
        recomputed = out / "recomputed" / name
        result = frozen.analyze(trials, inputs / "pairs.jsonl", recomputed,
                                expected_layers=list(range(LAYERS[model])))
        remote = load_json(record(run / "dense/analysis/audit.json"))
        assert remote["status"] == "passed" and remote["layers"] == list(range(LAYERS[model]))
        assert remote["trials_sha256"] == digest(trials)
        assert remote["pairs_sha256"] == digest(inputs / "pairs.jsonl")
        assert remote["bootstrap_repetitions"] == 10000
        for filename in ("layer_effects.csv", "seed_effects.csv", "pair_effects.csv", "detail.csv"):
            check_csv_values(record(run / "dense/analysis" / filename), recomputed / filename)
        layers = pd.read_csv(recomputed / "layer_effects.csv").to_dict("records")
        for layer in layers:
            measured = [r for r in rows if r["layer"] == layer["layer"] and r["condition"] == "full_donor_patch"]
            successes = sum(r["prediction"] == r["donor_count"] for r in measured)
            assert len(measured) == 40 and abs(successes / 40 - layer["full_donor_adoption"]) < 1e-12
            layer_export.append({
                "model": model, "layer_index_zero_based": int(layer["layer"]),
                "layer_one_based": int(layer["layer"]) + 1,
                "target_adoptions": successes, "pairs": 40, "seed_clusters": 10,
                "adoption_rate": layer["full_donor_adoption"],
                "ci95_low": layer["full_donor_adoption_ci95_low"],
                "ci95_high": layer["full_donor_adoption_ci95_high"],
                "invalid_outputs": sum(r["prediction"] not in range(1, 11) for r in measured),
                "receiver_retained": sum(r["prediction"] == r["gold_count"] for r in measured),
            })
        endpoint = [r for r in layer_export if r["model"] == model][-1]
        onset = result["descriptive_onset_layer"][model]
        dense_summary[model] = {
            "layers": LAYERS[model], "records": len(rows), "replay_records": 640,
            "new_layer_records": len(rows) - 640, "self_patch_correct": len(rows) // 2,
            "endpoint": endpoint, "descriptive_first_half_adoption_layer_one_based": None if onset is None else onset + 1,
            "elapsed_seconds": completion["elapsed_seconds"],
        }
        audits[name] = {"grid": grid, "replay": replay, "all_four_csv_exports_recomputed": True}
    assert common[0] == common[1]
    csv_out(out / "answer_all_layers.csv", layer_export)

    run, inputs = runs / "gemma_backward", bundle / "inputs/Gemma4-E4B"
    completion = load_json(record(run / "completion_audit.json"))
    manifest = load_json(record(run / "run_manifest.json"))
    assert completion["status"] == "PASS" and completion["trials"] == 90
    assert completion["task"] == "gemma_backward" and completion["model"] == "Gemma4-E4B"
    assert completion["plan_sha256"] == manifest["plan_sha256"] == plan["plan_sha256"]
    trials = run / "backward/trials.jsonl"
    assert completion["trials_sha256"] == digest(record(trials))
    rows = read_rows(trials)
    validate_progress(rows, [4, 6, 8], direction="backward")
    replay_rows = read_rows(record(run / "replay_forward_k6/trials.jsonl"))
    validate_progress(replay_rows, [6], direction="forward")
    replay = compare_replay(read_rows(record(inputs / "historical_forward_k6.jsonl")), replay_rows, progress=True)
    assert replay == load_json(record(run / "historical_replay_audit.json"))
    assert replay["status"] == "PASS" and replay["compared_trials"] == 30
    assert len({r["targeted_bank_sha256"] for r in replay_rows}) == 1
    assert {r["targeted_bank_sha256"] for r in rows} == {r["targeted_bank_sha256"] for r in replay_rows}
    for phase in ("replay_forward_k6", "backward_k4", "backward_k6", "backward_k8"):
        geometry = read_rows(record(run / phase / "geometry_audit.jsonl"))
        assert sorted(r["seed"] for r in geometry) == PROGRESS_SEEDS
        for row in geometry:
            assert row["endpoint_aligned"] and row["aligned_absolute_site"] == row["aligned_donor_site"]
            assert row["deletion_avoids_prompt_records"] and row["deletion_avoids_special_tokens"]
            assert not row["hidden_state_resampling"]
        loaded = load_json(record(run / phase / "loaded_model.json"))
        assert loaded["effective_attention_backend"] == "sdpa" and loaded["dtype"] == "torch.bfloat16"
        assert loaded["layers"] == 42 and loaded["training"] is False
        assert loaded["model_spec"] == plan["models"]["Gemma4-E4B"]["model_spec"]
        assert loaded["observed_revision"] in (None, loaded["model_spec"]["revision"])
        assert load_json(record(run / phase / "process_status.json"))["exit_code"] == 0
    backward = summarize_backward(rows)
    remote_backward = load_json(record(run / "backward/analysis.json"))
    numeric_keys = {"adoption_gain_patch_minus_self", "adoption_gain_ci95", "per_seed"}
    assert {k: v for k, v in backward.items() if k not in numeric_keys} == {
        k: v for k, v in remote_backward.items() if k not in numeric_keys}
    for key in ("adoption_gain_patch_minus_self", "adoption_gain_ci95"):
        np.testing.assert_allclose(backward[key], remote_backward[key], atol=1e-12, rtol=1e-12)
    assert [r["seed"] for r in backward["per_seed"]] == [r["seed"] for r in remote_backward["per_seed"]]
    np.testing.assert_allclose([r["adoption_gain"] for r in backward["per_seed"]],
                               [r["adoption_gain"] for r in remote_backward["per_seed"]], atol=1e-12, rtol=1e-12)
    csv_out(out / "gemma_backward_stepwise_trials.csv", backward["hop_rows"])
    csv_out(out / "gemma_backward_stepwise.csv", backward["stepwise"])
    csv_out(out / "gemma_backward_seed_effects.csv", backward["per_seed"])
    conditions = []
    for k in (4, 6, 8, "all"):
        for condition, label in (("receiver_self", "Receiver self-patch"), ("donor_to_receiver", "Target-to-Receiver patch")):
            selected = [r for r in rows if r["condition"] == condition and (k == "all" or r["donor_occurrence_k"] == k)]
            for row in selected:
                cities = row["generated_known_city_ordinals_any_surface"]
                assert row["first_generated_known_city_ordinal"] == (cities[0] if cities else None)
                assert row["greedy_receiver_successor_retention"] == (row["first_generated_known_city_ordinal"] == row["receiver_successor"])
            conditions.append({
                "target_k": k, "condition": label, "trials": len(selected),
                "target_successor_adoptions": sum(r["greedy_donor_successor_adoption"] for r in selected),
                "receiver_successor_retained": sum(r["greedy_receiver_successor_retention"] for r in selected),
                "truncated_generations": sum(r["generation_truncated"] for r in selected),
            })
    csv_out(out / "gemma_backward_conditions.csv", conditions)
    backward["conditions"] = conditions
    backward["elapsed_seconds"] = completion["elapsed_seconds"]
    audits["gemma_backward"] = {"replay": replay, "geometry_all_phases_passed": True,
                                 "backward_summary_recomputed": True, "records": 90}
    summary = {
        "status": "PASS", "plan_sha256": plan["plan_sha256"],
        "fresh_confirmation": False, "dense": dense_summary, "gemma_backward": backward,
        "audits": audits, "source_sha256": sources,
    }
    write_json(out / "summary.json", summary)
    return summary




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.bundle.resolve(), args.runs.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "dense_records": sum(d["records"] for d in result["dense"].values()),
                      "backward_records": result["gemma_backward"]["trials"], "output": str(args.output)}, indent=2))
