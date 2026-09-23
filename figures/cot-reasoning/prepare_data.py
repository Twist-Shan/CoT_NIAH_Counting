"""Build the CoT-reasoning main figure from audited local experiment outputs.

Run: python figures/cot-reasoning/build_figure.py
No model inference is performed. See README.md for estimands and scope limits.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import PercentFormatter
import numpy as np

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / "realistic"
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)
FINAL = ROOT / "output/pdf/cot_reasoning_main.pdf"
FINAL.parent.mkdir(parents=True, exist_ok=True)
MODELS = ["Qwen3-8B", "Gemma4-E4B"]
COLORS = ["#168DCA", "#E87824"]
INK, AXIS, GRID = "#30312E", "#8190A5", "#E7E8EE"
BROWN, WARM, EDGE = "#B87652", "#E7E3D7", "#B1B0A7"
ALIGNED = REPO / "work/v5_native_sample_aligned_20260829"
SCOPE = REPO / "work/same_site_progress_transplant_20260827/n10_patch_scope_frozen_v2"
SOURCES: dict[str, str] = {}
TIMINGS = {}


def record(p: Path) -> Path:
    SOURCES[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def read_json(p: Path):
    return json.loads(record(p).read_text(encoding="utf-8"))


def read_csv(p: Path):
    with record(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(p: Path):
    return [json.loads(x) for x in record(p).read_text(encoding="utf-8").split('\n') if x.strip()]


def write_csv(name: str, rows: list[dict]):
    assert rows, name
    with (DATA / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flag(x):
    return str(x).lower() in {"true", "1", "1.0"}


def seed_ci(values, draws):
    a = np.asarray(values, dtype=float)
    assert a.shape == (10,) and np.isfinite(a).all()
    low, high = np.quantile(a[draws].mean(axis=1), [.025, .975])
    return float(a.mean()), float(low), float(high)


# Validate and hash the measurement inputs directly.
for name in ["alignment_audit.json", "output_alignment_audit.json"]:
    assert read_json(ALIGNED / name)["status"] == "PASS"
record(ROOT / "figures/non-thinking/build_nonthinking_section.py")
record(ROOT / "runs/paper_figures/figures/nonthinking_form_retrieve_consolidate.pdf")

# B. Exact-aligned targeted bank necessity; preserve registered intervals.
ablation, ablation_seeds, ablation_arms = [], [], []
gate_map = {
    "final_count_failure": "targeted_bank_changes_final_count",
    "joint_retrieval_and_count_failure": "retrieval_failure_propagates_to_count",
}
ablation_support = {}
for model, bank in zip(MODELS, [128, 6]):
    p = ALIGNED / "runs" / model / "targeted_retrieval/analysis_confirmation"
    audit = read_json(p / "audit.json")
    gates = read_json(p / "claim_gates.json")
    seeds = read_csv(p / "seed_effects.csv")
    arms = read_csv(p / "anchor_arms.csv")
    assert audit["status"] == "PASS" and audit["seed_count"] == 10
    assert audit["seeds"] == list(range(1254, 1264))
    assert audit["decode_head_ablation_steps"] == -1
    assert len(arms) == 50 and Counter(r["condition"] for r in arms) == {
        "clean": 10, "selected_bank": 10, "layer_matched_random": 30,
    }
    ablation_support[model] = sorted(
        (int(r["seed"]), int(r["gold_count"]), int(r["from_occurrence"]),
         int(r["to_occurrence"]), r["condition"], int(r["repeat"])) for r in arms
    )
    for r in arms:
        ablation_arms.append(dict(model=model, **{k: r[k] for k in [
            "seed", "gold_count", "from_occurrence", "to_occurrence", "condition",
            "repeat", "bank_size", "final_count_failure", "next_city_failure",
            "joint_retrieval_and_count_failure", "generation_truncated",
        ]}))
    for outcome, gate_id in gate_map.items():
        rr = sorted([r for r in seeds if r["outcome"] == outcome], key=lambda r: int(r["seed"]))
        assert len(rr) == 10
        estimate = float(np.mean([float(r["selected_minus_random"]) for r in rr]))
        gate = gates["gates"][gate_id]
        assert abs(estimate - gate["estimate"]) < 1e-12
        for s in rr:
            seed = int(s["seed"])
            a = [r for r in arms if int(r["seed"]) == seed]
            selected = float(next(r for r in a if r["condition"] == "selected_bank")[outcome])
            random = np.mean([float(r[outcome]) for r in a if r["condition"] == "layer_matched_random"])
            assert abs(selected - random - float(s["selected_minus_random"])) < 1e-12
            ablation_seeds.append(dict(model=model, **s))
        ablation.append(dict(model=model, bank_size=bank, outcome=outcome, mean=estimate,
                             ci95_low=gate["ci_low"], ci95_high=gate["ci_high"], seed_clusters=10))
assert ablation_support[MODELS[0]] == ablation_support[MODELS[1]]
write_csv("ablation_plot.csv", ablation)
write_csv("ablation_seed_effects.csv", ablation_seeds)
write_csv("ablation_arms.csv", ablation_arms)

# C. Natural Qwen no-index traces. The three frozen geometries share all 60 keys.
scope_source = read_json(SCOPE / "frozen_scope_analysis.json")
manual = read_json(SCOPE / "item_span_generation_manual_audit.json")
assert manual["summary"]["donor_adoption_count"] == 43
assert manual["manual_review"]["recap_only_false_positive_count"] == 0
scope_order = ["item_end_w1", "event_tail_w4", "item_span"]
scope_counts = [10, 15, 43]
scope_rows, scope_seeds, scope_plot, hops = [], [], [], []
draws = np.random.default_rng(20260911).integers(0, 10, (10000, 10))
scope_support = {}
for scope, layer, expected in zip(scope_order, [26, 0, 0], scope_counts):
    cells = [r for r in scope_source["cells"] if r["scope"] == scope]
    assert len(cells) == 60 and {r["layer"] for r in cells} == {layer}
    seed_list = sorted({r["seed"] for r in cells})
    assert len(seed_list) == 10
    assert all(sum(r["seed"] == s for r in cells) == 6 for s in seed_list)
    assert sum(r["patched_greedy_donor_adoption"] for r in cells) == expected
    scope_support[scope] = sorted((r["seed"], r["donor_occurrence_k"], r["direction"]) for r in cells)
    raw_cells = {}
    for k in [4, 6, 8]:
        for direction in ["forward_skip", "backward_rewind"]:
            raw = read_jsonl(SCOPE / f"confirmation10/{scope}/{direction}_k{k}/trials.jsonl")
            assert len(raw) == 30
            for trial in raw:
                raw_cells[(int(trial["seed"]), k, direction, trial["condition"])] = trial
                if scope == "item_span" and trial["condition"] == "donor_to_receiver":
                    ordinals = trial["generated_known_city_ordinals_any_surface"]
                    for hop in range(1, 11 - k):
                        expected_prefix = list(range(k + 1, k + hop + 1))
                        hops.append(dict(seed=int(trial["seed"]), direction=direction,
                                         donor_occurrence_k=k, hop=hop,
                                         adopted_first=bool(ordinals and ordinals[0] == k + 1),
                                         exact_prefix=ordinals[:hop] == expected_prefix,
                                         truncated=trial["generation_truncated"]))
    for c in cells:
        key = (c["seed"], c["donor_occurrence_k"], c["direction"])
        patched, original = [raw_cells[(*key, arm)] for arm in ["donor_to_receiver", "receiver_self"]]
        assert patched["patch_applications"] == 1
        assert patched["first_generated_known_city_ordinal"] == c["patched_first_known_city_ordinal"]
        target = c["donor_occurrence_k"] + 1
        assert original["first_generated_known_city_ordinal"] != target
        scope_rows.append(dict(scope=scope, layer=layer, seed=c["seed"], direction=c["direction"],
                               donor_occurrence_k=c["donor_occurrence_k"], receiver_occurrence_j=c["receiver_occurrence_j"],
                               patch_width=c["patch_width"],
                               patched_adoption=int(patched["first_generated_known_city_ordinal"] == target),
                               self_adoption=int(original["first_generated_known_city_ordinal"] == target)))
    values = []
    for s in seed_list:
        value = np.mean([r["patched_greedy_donor_adoption"] for r in cells if r["seed"] == s])
        values.append(value)
        scope_seeds.append(dict(scope=scope, layer=layer, seed=s, adoption=float(value), self_adoption=0.0, cells=6))
    mean, low, high = seed_ci(values, draws)
    scope_plot.append(dict(scope=scope, layer=layer, successes=expected, cells=60, seed_clusters=10,
                           mean=mean, ci95_low=low, ci95_high=high, self_adoption=0.0))
assert scope_support[scope_order[0]] == scope_support[scope_order[1]] == scope_support[scope_order[2]]
hop2 = [h for h in hops if h["hop"] == 2 and h["adopted_first"]]
assert len(hop2) == 43 and sum(h["exact_prefix"] for h in hop2) == 43
write_csv("progress_state_plot.csv", scope_plot)
write_csv("progress_state_seed_effects.csv", scope_seeds)
write_csv("progress_state_cells.csv", scope_rows)
write_csv("continuation_hops.csv", hops)

# D. Actual greedy Target-count adoption from an explicitly selected audit.
# Without the source descriptor, retain the reviewed historical eight-layer plot.
answer_source_path = OUT / "answer_state_source.json"
answer_source = read_json(answer_source_path) if answer_source_path.exists() else None
ANSWER_SCOPE = "40 common directed pairs/model;10 seeds;8 registered layers; actual greedy Target-count adoption"
if answer_source is not None:
    assert answer_source["schema"] == "native_dense_answer_source_v1"
    answer_report = read_json(ROOT / answer_source["local_audit"])
    assert answer_report["status"] == "PASS" and answer_report["fresh_confirmation"] is False
    assert answer_report["plan_sha256"] == answer_source["plan_sha256"]
    ANSWER_SCOPE = "40 common directed pairs/model;10 seeds;all 36 Qwen and 42 Gemma decoder layers; historical replay exact; existing-cohort extension"

    def verified_answer_source(path):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == answer_report["source_sha256"][str(path.resolve())], path
        return path

answer, answer_seeds, answer_details = [], [], []
pair_support = {}
for model, historical_layers, layer_count, run_name in zip(
        MODELS, [[0, 5, 10, 15, 20, 25, 30, 35], [0, 6, 12, 18, 23, 29, 35, 41]],
        [36, 42], ["qwen_dense", "gemma_dense"]):
    if answer_source is None:
        layers = historical_layers
        p = ALIGNED / "runs" / model / "answer_query_layer_sweep/analysis"
        pair_path = ALIGNED / f"answer_query_layer_sweep_plan/{model}_pairs.jsonl"
    else:
        layers = list(range(layer_count))
        run = ROOT / answer_source["runs"] / run_name
        p = run / "dense/analysis"
        pair_path = ROOT / answer_source["bundle"] / "inputs" / model / "pairs.jsonl"
        completion = read_json(verified_answer_source(run / "completion_audit.json"))
        replay = read_json(verified_answer_source(run / "historical_replay_audit.json"))
        assert completion["status"] == "PASS" and completion["trials"] == 80 * layer_count
        assert completion["plan_sha256"] == answer_source["plan_sha256"]
        assert replay["status"] == "PASS" and replay["compared_trials"] == 640 and replay["changed_trials"] == 0
        raw = record(verified_answer_source(run / "dense/trials.jsonl"))
        assert completion["trials_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
        for source_file in [pair_path, *[p / name for name in ["audit.json", "layer_effects.csv", "seed_effects.csv", "detail.csv", "pair_effects.csv"]]]:
            verified_answer_source(source_file)
    audit = read_json(p / "audit.json")
    assert audit["status"] == "passed" and audit["registered_pairs"] == 40
    assert audit["layers"] == layers and audit["bootstrap_repetitions"] == 10000
    rows, seeds = read_csv(p / "layer_effects.csv"), read_csv(p / "seed_effects.csv")
    details = read_csv(p / "detail.csv")
    record(p / "pair_effects.csv")
    assert len(details) == 80 * len(layers) and len(rows) == len(layers) and len(seeds) == 10 * len(layers)
    pairs = read_jsonl(pair_path)
    pair_support[model] = sorted((int(r["seed"]), int(r["receiver_count"]), int(r["donor_count"])) for r in pairs)
    assert len(pair_support[model]) == 40
    assert all(not r["pair_selection_uses_patch_outcome"] for r in pairs)
    assert all(r["donor_exact_count"] and r["receiver_exact_count"] for r in pairs)
    assert Counter(r["condition"] for r in details) == {"self_patch": 40 * len(layers), "full_donor_patch": 40 * len(layers)}
    assert all(flag(r["receiver_retention"]) for r in details if r["condition"] == "self_patch")
    for r in details:
        answer_details.append({k: v for k, v in r.items() if k != "completion_text_raw"})
    for r in rows:
        layer = int(r["layer"])
        ss = [s for s in seeds if int(s["layer"]) == layer]
        assert len(ss) == 10 and {int(s["pairs"]) for s in ss} == {4}
        for s in ss:
            raw = [r for r in details if r["condition"] == "full_donor_patch"
                   and int(r["layer"]) == layer and int(r["seed"]) == int(s["seed"])]
            assert len(raw) == 4
            assert abs(np.mean([flag(r["donor_adoption"]) for r in raw]) - float(s["full_donor_adoption"])) < 1e-12
        assert abs(np.mean([float(s["full_donor_adoption"]) for s in ss]) - float(r["full_donor_adoption"])) < 1e-12
        answer.append(dict(model=model, layer=layer, mean=float(r["full_donor_adoption"]),
                           ci95_low=float(r["full_donor_adoption_ci95_low"]),
                           ci95_high=float(r["full_donor_adoption_ci95_high"]), pairs=40, seed_clusters=10))
    answer_seeds.extend(seeds)
assert pair_support[MODELS[0]] == pair_support[MODELS[1]]
write_csv("answer_state_plot.csv", answer)
write_csv("answer_state_seed_effects.csv", answer_seeds)
write_csv("answer_state_detail.csv", answer_details)
TIMINGS["load_and_verify_data_seconds"] = time.perf_counter() - START
