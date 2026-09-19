"""Recheck frozen inputs, raw accuracy scores and both ablation assays on CPU.

Writes a new report audit; never modifies captures, frozen plans or GPU results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict

import numpy as np

B = Path(__file__).resolve().parents[1]
REPO = B.parent
RUN = B / "runs/topk_completion_20260907_v1"
OUT = B / "runs/report_refresh_20260908_v1"
MODELS = ["Qwen3-8B", "Gemma4-E4B"]
TASKS = ["kth", "category"]
MODES = ["nonthinking", "native_thinking"]
BOOTSTRAP_SEED = 20260907


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def local(remote):
    run, rest = remote.split("/additional_experiments/", 1)[1].split("/", 1)
    path = (B / "runs" / run / "downloaded" / rest).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" else path


def write_csv(name, rows):
    assert rows
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimate(rows, metric):
    groups = defaultdict(list)
    for r in rows:
        groups[r["seed"]].append(float(r[metric]))
    v = np.array([np.mean(groups[s]) for s in sorted(groups)])
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(v), (20000, len(v)))
    lo, hi = np.quantile(v[idx].mean(axis=1), [.025, .975])
    return dict(n=len(rows), seeds=len(v), mean=float(v.mean()), lower=float(lo), upper=float(hi))


def background(case):
    cursor = 0
    parts = []
    for record in case["records"]:
        parts.append(case["passage"][cursor:record["char_start"]])
        cursor = record["char_end"]
    parts.append(case["passage"][cursor:])
    return "".join(parts)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    checks = Counter()
    timings = {}
    cfg = read(RUN / "downloaded/protocol.json")
    # Verify the scoring implementation before importing it.
    for key, path in [("helpers/run.py", B / "run.py"),
                      ("helpers/protocol.py", B / "protocol.py"),
                      ("helpers/diagnostics.py", B / "diagnostics.py"),
                      ("src/realistic_niah/parsing.py", REPO / "src/realistic_niah/parsing.py"),
                      ("src/realistic_niah_v5/causal.py", REPO / "src/realistic_niah_v5/causal.py")]:
        assert sha(path.read_bytes()) == cfg["files"][key], key
        checks["frozen_scoring_files"] += 1
    sys.path[:0] = [str(B), str(REPO / "src")]
    from run import summarize_generation
    from protocol import user_prompt
    from category_trial import prompt as category_prompt
    from realistic_niah_v5.causal import _first_generated_city_record

    def targeted_score(g, p):
        raw = g["completion_text_raw"]
        pred, start, _ = _first_generated_city_record(raw, [r["city"] for r in p["case"]["records"]])
        hits = [] if pred is None else [(start, pred)]
        for m in re.finditer(r"(?:city|flower) score audit,\s+([^\n,.]+?) received a score", raw, re.I):
            hits.append((m.start(1), m[1]))
        prediction = min(hits)[1] if hits else None
        correct = prediction is not None and prediction.casefold() == p["target"]["target_city"].casefold()
        return prediction, correct

    plans = {}
    natural = []
    identities, users, bases, source_hashes, kth_cases = {}, {}, {}, {}, {}
    target_roles = Counter()
    for model in MODELS:
        for task in TASKS:
            panel = read(RUN / "downloaded/plans" / f"{task}_{model}.json")
            assert len(panel) == 600
            seen = set()
            for p in panel:
                c = p["case"]
                key = (model, task, p["mode"], p["case_id"])
                assert key not in plans
                plans[key] = p
                assert p["seed"] == c["seed"]
                assert p["split"] == ("discovery" if p["seed"] < 1254 else "confirmation")
                assert len(c["records"]) == c["total_records"] == 10
                assert [r["ordinal"] for r in c["records"]] == list(range(1, 11))
                assert sha(c["passage"].encode()) == c["passage_sha256"]
                for r in c["records"]:
                    assert c["passage"][r["char_start"]:r["char_end"]] == r["text"]
                if task == "kth":
                    r = c["records"][c["level"] - 1]
                    assert c["gold"] == f"{r['city']}|{r['score']}"
                    kth_cases.setdefault(c["seed"], c)
                else:
                    assert c["gold"] == str(sum(r["category"] == c["target_category"] for r in c["records"]))
                identity = (task, c["case_id"])
                packed = json.dumps(c, sort_keys=True)
                assert identities.setdefault(identity, packed) == packed
                bg = background(c)
                assert bases.setdefault(c["seed"], bg) == bg
                assert source_hashes.setdefault(c["seed"], c["source_passage_sha256"]) == c["source_passage_sha256"]
                for name, h in p["source_hashes"].items():
                    assert sha(local(p["source"] + "/" + name).read_bytes()) == h
                    checks["natural_source_hashes"] += 1
                prompt = read(local(p["source"] + "/prompt.json"))
                generation = read(local(p["source"] + "/generation.json"))
                assert prompt["user_text"] == (user_prompt(c, p["mode"]) if task == "kth" else category_prompt(c, p["mode"]))
                assert sha(prompt["user_text"].encode()) == prompt["user_text_sha256"]
                if task == "kth":
                    assert sha(prompt["rendered_prompt"].encode()) == prompt["rendered_prompt_sha256"]
                    assert prompt["attention_mask"] == [1] * len(prompt["input_ids"])
                    assert prompt["chat_template_kwargs"]["enable_thinking"] == (p["mode"] == "native_thinking")
                    assert prompt["assistant_prefill_text"] == (c["answer_prefix"] if p["mode"] == "nonthinking" else "")
                else:
                    # Category captures use a compact schema. The entire prompt.json,
                    # including rendered text and IDs, was hash-verified above.
                    assert prompt["case_id"] == p["case_id"] and prompt["mode"] == p["mode"]
                    checks["compact_category_prompt_schema"] += 1
                assert prompt["rendered_prompt"].count(prompt["user_text"]) == 1
                assert prompt["rendered_prompt"].endswith(c["answer_prefix"]) == (p["mode"] == "nonthinking")
                userkey = (task, p["mode"], c["case_id"])
                assert users.setdefault(userkey, prompt["user_text"]) == prompt["user_text"]
                score = summarize_generation(generation, c, mode=p["mode"], prefixed=p["mode"] == "nonthinking")
                assert score["correct"] == generation["correct"]
                assert score["prediction"] == generation["prediction"]
                checks["natural_outputs_rescored"] += 1
                row = dict(model=model, task=task, mode=p["mode"], case_id=p["case_id"], seed=p["seed"],
                           split=p["split"], level=c["level"], target_category=c.get("target_category", ""),
                           correct=int(score["correct"]), prediction=score["prediction"],
                           unparsed=int(score["prediction"] is None), truncated=int(generation["generation_truncated"]),
                           prompt_tokens=len(prompt["input_ids"]), generated_tokens=generation["generated_token_count"])
                natural.append(row)
                seen.add((p["mode"], p["seed"], p["case_id"]))
                if p["target"] and p["split"] == "confirmation":
                    target_roles[model, task, p["target"]["role"]] += 1
            assert len(seen) == 600
    # Check record identities, scores and source background between the two tasks.
    for task, cid in identities:
        if task != "category":
            continue
        c = json.loads(identities[task, cid])
        source = kth_cases[c["seed"]]
        for r, s in zip(c["records"], source["records"]):
            assert (r["source_city"], r["score"], r["ordinal"]) == (s["city"], s["score"], s["ordinal"])
        checks["cross_task_record_sets"] += 1
    checks.update(unique_task_inputs=len(identities), aligned_seed_backgrounds=len(bases), matching_model_user_prompts=len(users))
    timings["inputs_and_natural_seconds"] = time.perf_counter() - started
    print(json.dumps(dict(stage="inputs_and_natural", checks=dict(checks))), flush=True)

    # Membership transfer is a different claim from reusing a selection algorithm.
    memberships = []
    for model in MODELS:
        for task in TASKS:
            banks = read(RUN / "downloaded/banks" / f"{task}_{model}.json")
            name = "qwen_shared_k128" if model.startswith("Qwen") else "gemma_shared_k6"
            mainfile = REPO / "configs" / f"realistic_niah_v5_{name}_targeted_selection_frozen.json"
            frozen = read(mainfile)["development_selection"]["primary_bank_heads"]
            assert banks["native_thinking_targeted"]["heads"][:len(frozen)] == frozen
            memberships.append(dict(model=model, task=task, main_k=len(frozen), transferred_prefix=True,
                                    source=str(mainfile.relative_to(REPO)), sha256=sha(mainfile.read_bytes())))
            if model.startswith("Gemma"):
                assert banks["native_thinking_targeted"]["heads"] == read(RUN / "downloaded/gemma_k8_source.json")["development_selection"]["primary_bank_heads"]
            for mode in MODES:
                bank = banks[mode + "_broad"]
                groups = defaultdict(lambda: defaultdict(list))
                for p in plans.values():
                    if p["mode"] != mode or p["split"] != "discovery" or p["source"].split("/")[-4] != model or p["case"]["task"] != ("kth_needle" if task == "kth" else "category_count"):
                        continue
                    remote = cfg["sources"][model][task] + f"/full/{model}/{task}/attention/{mode}_{p['case_id']}.json"
                    path = local(remote)
                    assert sha(path.read_bytes()) == cfg["source_hashes"][remote]
                    for layer, head, value in read(path)["heads"]:
                        groups[layer, head][p["seed"]].append(value)
                scores = {h: np.mean([np.mean(v) for v in seeds.values()]) for h, seeds in groups.items()}
                ranked = sorted(scores, key=lambda h: (-scores[h], h))
                assert [list(h) for h in ranked] == [r[:2] for r in bank["ranking"]]
                assert np.allclose([scores[h] for h in ranked], [r[2] for r in bank["ranking"]], atol=1e-12, rtol=0)
                widths = read(RUN / "downloaded/full" / model / "environment.json")["widths"]
                heads, counts = [], Counter()
                for layer, head in ranked:
                    if counts[layer] < widths[layer] // 2:
                        heads.append([layer, head]); counts[layer] += 1
                    if len(heads) == len(bank["heads"]):
                        break
                assert heads == bank["heads"]
                checks["broad_rankings_recomputed"] += 1
    print(json.dumps(dict(stage="selection", checks=dict(checks))), flush=True)

    audit = read(RUN / "analysis/audit.json")
    details, errors = [], []
    hist = Counter()
    with tarfile.open(RUN / "results.tgz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in audit["file_hashes"]:
                continue
            data = archive.extractfile(member).read()
            assert sha(data) == audit["file_hashes"][name]
            x = json.loads(data)
            _, model, task, key, cid, _ = name.split("/")
            mode, assay = key.rsplit("_", 1)
            p = plans[model, task, mode, cid]
            assert p["split"] == "confirmation"
            assert x["prefix_length"] == (p["broad_prefix"] if assay == "broad" else p["target"]["prefix_length"])
            assert [a["name"] for a in x["arms"]] == ["clean", "selected", "random_0", "random_1", "random_2"]
            vals = []
            for a in x["arms"]:
                if assay == "broad":
                    rescored = summarize_generation(a["generation"], p["case"], mode=mode, prefixed=True)
                    pred, correct = rescored["prediction"], rescored["correct"]
                else:
                    pred, correct = targeted_score(a["generation"], p)
                assert bool(correct) == bool(a["score"]["correct"])
                assert pred == a["score"]["prediction"]
                vals.append(int(correct))
                hist[model, task, mode, assay, x["k"], a["name"], "unparsed"] += int(pred is None)
                hist[model, task, mode, assay, x["k"], a["name"], "truncated"] += int(a["generation"]["generation_truncated"])
                checks["ablation_arms_rescored"] += 1
            rand = sum(vals[2:]) / 3
            details.append(dict(model=model, task=task, mode=mode, assay=assay, k=x["k"],
                                case_id=cid, seed=x["seed"], clean=vals[0], selected=vals[1], random=rand,
                                delta=rand - vals[1], clean_drop=vals[0] - vals[1]))
            checks["ablation_points"] += 1
    assert checks["ablation_points"] == 6739 and checks["ablation_arms_rescored"] == 33695
    group = defaultdict(list)
    for r in details:
        group[r["model"], r["task"], r["mode"], r["assay"], r["k"]].append(r)
    summary = []
    for key, rows in sorted(group.items()):
        for metric in ["clean", "selected", "random", "delta", "clean_drop"]:
            summary.append(dict(zip(["model", "task", "mode", "assay", "k"], key), metric=metric, **estimate(rows, metric)))
    with (RUN / "analysis/summary.csv").open(encoding="utf-8") as f:
        for old in csv.DictReader(f):
            row = next(r for r in summary if all(str(r[k]) == old[k] for k in ["model", "task", "mode", "assay", "k", "metric"]))
            assert np.allclose([row[k] for k in ["mean", "lower", "upper"]], [float(old[k]) for k in ["mean", "lower", "upper"]], atol=1e-12, rtol=0)
            assert row["n"] == int(old["n"]) and row["seeds"] == int(old["seeds"])
            checks["existing_summary_rows_reproduced"] += 1
    assert checks["existing_summary_rows_reproduced"] == 296

    nat_summary, paired = [], []
    for model in MODELS:
        for task in TASKS:
            for split in ["all", "discovery", "confirmation"]:
                panel = [r for r in natural if r["model"] == model and r["task"] == task and (split == "all" or r["split"] == split)]
                for mode in MODES:
                    sub = [r for r in panel if r["mode"] == mode]
                    nat_summary.append(dict(model=model, task=task, mode=mode, split=split,
                                            correct=sum(r["correct"] for r in sub),
                                            unparsed=sum(r["unparsed"] for r in sub),
                                            truncated=sum(r["truncated"] for r in sub), **estimate(sub, "correct")))
                bycase = defaultdict(dict)
                for r in panel:
                    bycase[r["case_id"]][r["mode"]] = r
                pairs = [dict(seed=v["nonthinking"]["seed"], delta=v["native_thinking"]["correct"] - v["nonthinking"]["correct"]) for v in bycase.values()]
                paired.append(dict(model=model, task=task, split=split, **estimate(pairs, "delta")))

    support = []
    for task in TASKS:
        for mode, assay in [("nonthinking", "broad"), ("native_thinking", "broad"), ("native_thinking", "targeted")]:
            sets = {}
            for model in MODELS:
                sets[model] = {p["case_id"] for (m, t, mo, _), p in plans.items() if (m, t, mo) == (model, task, mode) and p["split"] == "confirmation" and bool(p["broad_prefix"] if assay == "broad" else p["target"])}
            support.append(dict(task=task, mode=mode, assay=assay, qwen_n=len(sets[MODELS[0]]), gemma_n=len(sets[MODELS[1]]), intersection_n=len(sets[MODELS[0]] & sets[MODELS[1]]), identical=sets[MODELS[0]] == sets[MODELS[1]]))
    write_csv("natural_per_case.csv", natural)
    write_csv("natural_summary.csv", nat_summary)
    write_csv("natural_mode_difference.csv", paired)
    write_csv("ablation_summary.csv", summary)
    write_csv("ablation_per_case.csv", details)
    write_csv("cross_model_support.csv", support)
    write_csv("generation_diagnostics.csv", [dict(zip(["model", "task", "mode", "assay", "k", "condition", "metric"], k), count=v) for k, v in sorted(hist.items())])
    timings["total_seconds"] = time.perf_counter() - started
    result = dict(status="PASS", date="2026-09-08", checks=dict(checks), timings=timings,
                  inference=dict(unit="seed", replicates=20000, bootstrap_seed=BOOTSTRAP_SEED, interval="pointwise percentile 95%; unadjusted"),
                  targeted_membership=memberships, cross_model_support=support,
                  target_anchor_roles=[dict(model=k[0], task=k[1], role=k[2], n=v) for k, v in sorted(target_roles.items())],
                  scope="Exact paired inputs within task; identical seeds and source background across tasks; model-specific native endpoints. Broad rankings refit on new-task discovery; Targeted old shared bank transferred.",
                  results_hashes={p.name: sha(p.read_bytes()) for p in OUT.glob("*.csv")})
    (OUT / "alignment_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "audit_timing.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="PASS", checks=dict(checks), timings=timings, common_support=support)), flush=True)


if __name__ == "__main__":
    main()
