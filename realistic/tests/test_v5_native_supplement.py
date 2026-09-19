from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from realistic_niah_v5 import native_supplement as n

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("test_native_supplement_runner", ROOT / "scripts/run_v5_native_supplement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pairs(model="Qwen3-8B"):
    return [
        {"pair_id": f"{model}_{s}_{r}_{t}", "model_label": model, "split": "confirmation", "seed": s,
         "receiver_count": r, "donor_count": t, "alignment_key": [s, r, t],
         "receiver_request_id": f"{model}_{s}_{r}", "donor_request_id": f"{model}_{s}_{t}",
         "receiver_site_id": "answer_query_v3", "donor_site_id": "answer_query_v3",
         "receiver_exact_count": True, "donor_exact_count": True, "pair_selection_uses_patch_outcome": False}
        for s in n.ANSWER_SEEDS for r, t in [(1, 2), (2, 1), (9, 10), (10, 9)]
    ]


def answer_rows(registry, layers):
    return [
        {"pair_id": p["pair_id"], "layer": l, "condition": a, "model_label": p["model_label"],
         "seed": p["seed"], "request_id": p["receiver_request_id"], "donor_request_id": p["donor_request_id"],
         "gold_count": p["receiver_count"], "donor_count": p["donor_count"],
         "receiver_site_id": "answer_query_v3", "donor_site_id": "answer_query_v3", "source_positions": [10000],
         "prediction": p["receiver_count"] if a == "self_patch" else p["donor_count"],
         "completion_text_raw": str(p["receiver_count"] if a == "self_patch" else p["donor_count"]), "generated_token_count": 1}
        for p in registry for l in layers for a in n.ANSWER_ARMS
    ]


def test_directed_cross_model_registry_and_layer_budgets():
    assert n.validate_pairs(pairs(), n.MODELS[0]) == n.validate_pairs(pairs(n.MODELS[1]), n.MODELS[1])
    assert sum(80 * (n.LAYERS[m] - len(n.OLD_LAYERS[m])) for m in n.MODELS) == 4960
    assert sum(80 * n.LAYERS[m] for m in n.MODELS) == 6240


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_seed", "outcome_selection", "wrong_site"])
def test_pair_freeze_rejects_invalid_registry(mutation):
    p = pairs()
    if mutation == "duplicate": p[1] = deepcopy(p[0])
    if mutation == "wrong_seed": p[0]["seed"] = 1234
    if mutation == "outcome_selection": p[0]["pair_selection_uses_patch_outcome"] = True
    if mutation == "wrong_site": p[0]["receiver_site_id"] = "answer_query"
    with pytest.raises(ValueError): n.validate_pairs(p, n.MODELS[0])


@pytest.mark.parametrize("mutation", ["missing_layer", "duplicate", "wrong_request", "self_failure", "length"])
def test_answer_grid_validation(mutation):
    p = pairs()
    rows = answer_rows(p, [0, 1])
    n.validate_answer(rows, p, [0, 1])
    if mutation == "missing_layer": rows = [r for r in rows if r["layer"] == 0]
    if mutation == "duplicate": rows.append(deepcopy(rows[0]))
    if mutation == "wrong_request": rows[0]["request_id"] = "wrong"
    if mutation == "self_failure": rows[0]["prediction"] = 9
    if mutation == "length": rows[0]["generated_token_count"] = 17
    with pytest.raises(ValueError): n.validate_answer(rows, p, [0, 1])


def test_invalid_target_output_is_retained_as_failure():
    p = pairs()
    rows = answer_rows(p, [0])
    rows[1]["prediction"] = None
    rows[1]["completion_text_raw"] = "unparsable"
    assert n.validate_answer(rows, p, [0])["trials"] == 80


def test_resume_validates_shard_pair_index_and_both_conditions(tmp_path):
    p = pairs()
    path = tmp_path / "L001/00001.jsonl"
    n.write_rows(path, answer_rows([p[0]], [1]))
    n.validate_answer_shards(tmp_path, p, [1])
    n.write_rows(path, answer_rows([p[1]], [1]))
    with pytest.raises(ValueError, match="Invalid resume shard"):
        n.validate_answer_shards(tmp_path, p, [1])


def test_replay_checks_literal_output_and_position():
    old = answer_rows(pairs(), [0])
    new = deepcopy(old)
    assert n.compare_replay(old, new)["status"] == "PASS"
    new[1]["source_positions"] = [10001]
    assert n.compare_replay(old, new)["changed_trials"] == 1


def progress_rows():
    rows = []
    for seed in n.PROGRESS_SEEDS:
        for k in (4, 6, 8):
            for arm in n.PROGRESS_ARMS:
                cities = list(range(k + (2 if arm == "receiver_self" else 1), 11))
                rows.append({
                    "seed": seed, "receiver_occurrence_j": k + 1, "donor_occurrence_k": k, "condition": arm,
                    "layer": 16, "gold_count": 10, "patch_scope": "item_span", "cohort_mode": "prompt_conditioned_noindex",
                    "patch_applications": 1, "donor_successor": k + 1, "receiver_successor": k + 2,
                    "donor_vs_receiver_sum_logodds": 1.0, "donor_vs_receiver_attention_log_ratio": 0.1,
                    "generated_known_city_ordinals_any_surface": cities, "first_generated_known_city_ordinal": cities[0],
                    "greedy_donor_successor_adoption": cities[0] == k + 1, "completion_text": "example", "generated_token_count": len(cities),
                })
    return rows


def test_backward_stats_cluster_by_seed_and_stepwise_risk_sets():
    rows = progress_rows()
    result = n.summarize_backward(rows)
    assert result["seed_clusters"] == 10
    assert result["adoption_gain_ci95"] == [1.0, 1.0]
    assert [(r["successes"], r["eligible"]) for r in result["stepwise"]] == [(30, 30), (30, 30), (20, 20), (20, 20)]
    # A truncated continuation succeeds twice, fails at step 3, then leaves
    # the step-4 risk set. It remains in the available-but-failed step 3 set.
    patch = next(r for r in rows if r["seed"] == n.PROGRESS_SEEDS[0] and r["donor_occurrence_k"] == 4 and r["condition"] == "donor_to_receiver")
    patch["generated_known_city_ordinals_any_surface"] = [5, 6]
    patch["generation_truncated"] = True
    result = n.summarize_backward(rows)
    assert [(r["successes"], r["eligible"]) for r in result["stepwise"]][2:] == [(19, 20), (19, 19)]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "wrong_direction", "nonfinite", "bad_flag"])
def test_backward_conditions_are_complete_and_typed(mutation):
    rows = progress_rows()
    if mutation == "duplicate": rows.append(deepcopy(rows[0]))
    if mutation == "missing": rows.pop()
    if mutation == "wrong_direction": rows[0]["receiver_occurrence_j"] = 3
    if mutation == "nonfinite": rows[0]["donor_vs_receiver_sum_logodds"] = float("nan")
    if mutation == "bad_flag": rows[0]["greedy_donor_successor_adoption"] = True
    with pytest.raises(ValueError): n.validate_progress(rows, [4, 6, 8], direction="backward")


def test_jsonl_preserves_unicode_separator_inside_passage(tmp_path):
    p = tmp_path / "trace.jsonl"
    n.write_rows(p, [{"user_text": "a\u2028b\u2029c"}, {"seed": 1254}])
    assert len(n.read_rows(p)) == 2
    assert n.read_rows(p)[0]["user_text"] == "a\u2028b\u2029c"


def test_bundle_integrity_rejects_modified_input(tmp_path):
    p = tmp_path / "input.json"
    p.write_text("{}")
    core = {"files_sha256": {"input.json": n.digest(p)}}
    n.write_json(tmp_path / "plan.json", {**core, "plan_sha256": n.canonical_digest(core)})
    n.verify_bundle(tmp_path)
    p.write_text("[]")
    with pytest.raises(ValueError, match="Frozen bundle mismatch"): n.verify_bundle(tmp_path)


def test_registered_commands_use_fixed_layers_and_gemma_direction(tmp_path, monkeypatch):
    runner = load_runner()
    plan = {"models": {m: {"old_layers": n.OLD_LAYERS[m], "missing_layers": sorted(set(range(n.LAYERS[m])) - set(n.OLD_LAYERS[m]))} for m in n.MODELS},
            "torch_dtype": "bfloat16", "attention_backend": "sdpa"}
    monkeypatch.setattr(runner, "verify_bundle", lambda _: plan)
    jobs = runner.commands(tmp_path / "bundle", tmp_path / "run", n.MODELS[0], "answer_dense", tmp_path / "cache", "auto")
    assert [len(j["layers"]) for j in jobs] == [8, 28]
    assert all("--restartable" in j["args"] for j in jobs)
    jobs = runner.commands(tmp_path / "bundle", tmp_path / "run", n.MODELS[1], "gemma_backward", tmp_path / "cache", "auto")
    assert [j["phase"] for j in jobs] == ["replay_forward_k6", "backward_k4", "backward_k6", "backward_k8"]
    assert [j["args"][j["args"].index("--receiver-occurrence") + 1] for j in jobs] == ["5", "5", "7", "9"]


def test_replay_failure_stops_before_new_gpu_layers(tmp_path, monkeypatch):
    runner = load_runner()
    bundle = tmp_path / "bundle"
    out = tmp_path / "run"
    model = n.MODELS[0]
    p = pairs()
    historical = answer_rows(p, n.OLD_LAYERS[model])
    n.write_rows(bundle / "inputs" / model / "pairs.jsonl", p)
    n.write_rows(bundle / "inputs" / model / "historical_answer_trials.jsonl", historical)
    plan = {"plan_sha256": "test", "models": {model: {"old_layers": n.OLD_LAYERS[model], "missing_layers": [1], "all_layers": list(range(36))}},
            "torch_dtype": "bfloat16", "attention_backend": "sdpa"}
    monkeypatch.setattr(runner, "verify_bundle", lambda _: plan)
    monkeypatch.setattr(runner, "runtime_snapshot", lambda: {"mock": True})
    phases = []
    def fake_phase(bundle, output, model, job, plan):
        phases.append(job["phase"])
        rows = deepcopy(historical)
        rows[1]["completion_text_raw"] = "different output"
        n.write_rows(Path(job["output"]), rows)
    monkeypatch.setattr(runner, "run_phase", fake_phase)
    args = argparse.Namespace(bundle=bundle, output=out, model=model, task="answer_dense", cache_dir=tmp_path / "cache", device_map="auto", execute=True)
    with pytest.raises(ValueError, match="Historical replay differs"):
        runner.run(args)
    assert phases == ["replay"]
    assert json.loads((out / "failure.json").read_text())["status"] == "FAILED"
