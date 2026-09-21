"""Source-count matching after single-layer answer state patching.

Only pairs with correct source and target baseline answers are eligible.
Every layer uses the same pairs; unparseable or invalid outputs are failures.
The estimator pools eligible pairs. Confidence intervals resample seed clusters
and recompute the pooled numerator/denominator (10,000 percentile draws).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

MODELS = {"Qwen3-8B": 36, "Gemma4-E4B": 42}
CONDITIONS = ("donor_transport", "same_count_seed", "self_patch")


def flag(value):
    if value in (True, "True", "true", "1"):
        return True
    if value in (False, "False", "false", "0", ""):
        return False
    raise ValueError(f"Unexpected flag: {value!r}")


def pair_key(row):
    return (int(row["seed"]), row["receiver_stimulus_id"], row["donor_stimulus_id"])


def compute(exports: Path, repetitions=10000, random_seed=20260921):
    started = time.perf_counter()
    curves, details, audit = [], [], {}
    inputs = {}
    for model, n_layers in MODELS.items():
        path = exports / "v4_4_causal_v2_overall_clean_correct/clean_correct" / model / "patching/answer_patching/screen/detail.clean_correct.csv.gz"
        data = path.read_bytes()
        inputs[str(path.relative_to(exports))] = hashlib.sha256(data).hexdigest()
        raw = list(csv.DictReader(gzip.decompress(data).decode("utf-8-sig").splitlines()))
        rows = [r for r in raw if r["patch_protocol"] == "single_layer" and r["site"] == "answer_query"]
        assert set(r["condition"] for r in rows) == set(CONDITIONS)
        assert all(r["status"] == "ok" and flag(r["baseline_is_correct"]) and r["donor_baseline_outcome"] == "correct" for r in rows)
        assert all(int(r["donor_count"]) != int(r["receiver_count"]) for r in rows)
        model_summary = {}
        for condition in CONDITIONS:
            selected = [r for r in rows if r["condition"] == condition]
            by_layer = defaultdict(list)
            for r in selected:
                by_layer[int(r["start_layer"])].append(r)
            assert set(by_layer) == set(range(n_layers))
            expected_pairs = {pair_key(r) for r in by_layer[0]}
            seeds = sorted({p[0] for p in expected_pairs})
            assert seeds == list(range(1254, 1259))
            rng = np.random.default_rng(random_seed)
            draws = rng.integers(0, len(seeds), size=(repetitions, len(seeds)))
            for layer, layer_rows in sorted(by_layer.items()):
                assert len(layer_rows) == len(expected_pairs)
                assert {pair_key(r) for r in layer_rows} == expected_pairs
                seed_hits, seed_n = Counter(), Counter()
                for r in layer_rows:
                    source = int(r["donor_count"])
                    valid = flag(r["patched_format_valid"]) and flag(r["transport_numeric_valid"])
                    predicted = float(r["patched_predicted_count"]) if r["patched_predicted_count"] else None
                    hit = valid and predicted is not None and np.isfinite(predicted) and predicted == source
                    assert bool(hit) == flag(r["strict_target_hit"])
                    if condition == "donor_transport":
                        assert int(r["state_donor_count"]) == source
                    else:
                        assert int(r["state_donor_count"]) == int(r["receiver_count"])
                    seed = int(r["seed"])
                    seed_hits[seed] += int(hit)
                    seed_n[seed] += 1
                    details.append(dict(model=model, layer=layer+1, condition=condition, seed=seed,
                                        source_count=source, target_count=int(r["receiver_count"]),
                                        patched_count=predicted, valid=valid, source_match=int(hit)))
                hits = np.array([seed_hits[s] for s in seeds])
                totals = np.array([seed_n[s] for s in seeds])
                samples = hits[draws].sum(axis=1) / totals[draws].sum(axis=1)
                low, high = np.quantile(samples, [.025, .975])
                curves.append(dict(model=model, layer=layer+1, condition=condition,
                                   successes=int(hits.sum()), pairs=int(totals.sum()),
                                   mean=float(hits.sum()/totals.sum()), ci95_low=float(low), ci95_high=float(high),
                                   seeds=len(seeds)))
            final = curves[-1]
            model_summary[condition] = {k: final[k] for k in ("successes", "pairs", "mean", "ci95_low", "ci95_high")}
        audit[model] = dict(layers=n_layers, final_layer=model_summary,
                            invalid_different_count=sum(not r["valid"] for r in details if r["model"] == model and r["condition"] == "donor_transport"))
    return curves, details, dict(status="PASS", models=audit, input_sha256=inputs,
                                 metric="Proportion of predictions matching the source count after answer state patching.",
                                 eligibility="Both source and target baseline answers correct; fixed pairs across all layers.",
                                 controls="Reference count remains that of the corresponding different-count source; no subtraction.",
                                 estimator="Pooled eligible-pair ratio; invalid outputs count as failures.",
                                 confidence_interval="Pointwise percentile seed-cluster bootstrap; pooled ratio recomputed in each draw.",
                                 repetitions=repetitions, random_seed=random_seed, elapsed_seconds=time.perf_counter()-started)


def save(exports: Path, output: Path):
    curves, details, audit = compute(exports)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("answer_source_count_curves.csv", curves), ("answer_source_count_details.csv", details)]:
        with (output/name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/"analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(save(args.exports, args.output), indent=2))
