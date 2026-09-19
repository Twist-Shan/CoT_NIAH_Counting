"""Independently verify the exploratory noise audit and its input controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "work/nonthinking_v44_geometry_300_150_136_166_78"
EXPORT = REPO / "exports/run_20260731_v4_numeric_presentation_v3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    out = args.output.resolve()
    predictions = pd.read_csv(out / "confirmation_predictions.csv")
    curves = pd.read_csv(out / "dispersion_by_count.csv")
    summary = json.loads((out / "summary.json").read_text())
    hashes, checks, behavior = {}, [], []
    for (model, layer), group in predictions.groupby(["model", "layer_one_based"]):
        assert len(group) == 100 and not group.duplicated(["seed", "N"]).any()
        stats = group.groupby("N")["probe_prediction"].agg(["mean", "std"])
        expected = curves[(curves.model == model) & (curves.layer_one_based == layer)].sort_values("N")
        np.testing.assert_allclose(stats["std"], expected.probe_sd, rtol=1e-12)
        np.testing.assert_allclose(stats["mean"], expected.probe_mean, rtol=1e-12)
        checks.append(f"{model} L{layer}: independently grouped prediction means and SDs agree")
    for model in ["Qwen3-8B", "Gemma4-E4B"]:
        base = SOURCE / model / "numeric/representation/answer_query_all_layers_v1"
        layer = summary[model]["primary"]["layer_one_based"] - 1
        raw = np.empty((10, 10), dtype=object)
        with np.load(out / f"{model}_frozen_probe.npz", allow_pickle=False) as probe:
            for seed in range(1254, 1264):
                for n in range(1, 11):
                    file = base / f"shards/v4.4/V4_4_T10000_N{n}_seed{seed}.npz"
                    with np.load(file, allow_pickle=False) as z:
                        axis = list(z["layer_indices"]).index(layer)
                        state = z["query_states"][axis].astype(np.float64)
                    raw[seed - 1254, n - 1] = state
                    standardized = (state - probe["feature_mean"]) / probe["feature_scale"]
                    coordinates = (standardized - probe["pca_mean"]) @ probe["pca_components"].T
                    actual = coordinates @ probe["ridge_coef"] + probe["ridge_intercept"]
                    expected = predictions[(predictions.model == model) & (predictions.layer_one_based == layer + 1) & (predictions.seed == seed) & (predictions.N == n)].probe_prediction.item()
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        states = np.stack([np.stack(list(row)) for row in raw])
        direct_rms = np.sqrt(states.var(axis=0, ddof=1).mean(axis=-1))
        expected = curves[(curves.model == model) & curves.primary_layer].sort_values("N").raw_rms
        np.testing.assert_allclose(direct_rms, expected, rtol=1e-12)
        checks.append(f"{model}: 100 saved-probe predictions reconstructed from original shards; original-space dispersion reproduced")
        b = EXPORT / model / "numeric/behavior/capture/generation_labels.csv"
        hashes[str(b.relative_to(REPO))] = hashlib.sha256(b.read_bytes()).hexdigest()
        df = pd.read_csv(b)
        df = df[(df.design_variant == "v4.4") & df.seed.between(1254, 1263)].copy()
        assert len(df) == 100 and not df.duplicated(["seed", "gold_count"]).any()
        assert df.parsed_count.notna().all(), "Missing outputs must not silently disappear"
        stat = df.groupby("gold_count").agg(n=("seed", "size"), parsed=("parsed_count", "count"), mean_generated=("parsed_count", "mean"), sd_generated=("parsed_count", "std"), accuracy=("is_correct", "mean"))
        stat.insert(0, "model", model)
        stat.to_csv(out / f"{model}_observed_generation_by_count.csv")
        behavior.extend(stat.reset_index().to_dict("records"))
    p = EXPORT / "dataset/stimuli.jsonl"
    hashes[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding="utf-8") as stream:
        stimuli = [r for line in stream if (r := json.loads(line)).get("design_variant") == "v4.4" and 1234 <= int(r["seed"]) <= 1263]
    assert len(stimuli) == 300
    assert len({(r["seed"], r["gold_count"]) for r in stimuli}) == 300
    for seed in range(1234, 1264):
        group = sorted((r for r in stimuli if r["seed"] == seed), key=lambda r: r["gold_count"])
        assert [r["gold_count"] for r in group] == list(range(1, 11))
        assert {r["canonical_passage_tokens"] for r in group} == {10000}
        schedules = [[(s["canonical_span_start"], s["canonical_span_end"]) for s in r["slots"]] for r in group]
        assert all(schedule == schedules[0] for schedule in schedules)
        for r in group:
            active = [s["slot_index"] for s in r["slots"] if s["active"]]
            assert active == list(range(1, r["gold_count"] + 1))
    checks.append("All 300 stimuli have exactly 10000 canonical passage tokens, seed-paired fixed slot positions, and nested active slots 1..N")
    result = {"status": "PASS", "checks": checks, "behavior": behavior, "supplementary_source_sha256": hashes,
              "elapsed_seconds": time.perf_counter() - started,
              "position_limit": "The answer-query boundary is fixed, but later N activates later slots; needle count and occupied spatial extent remain coupled.",
              "interpretation": "This verifies calculations and input controls, not the biological or causal interpretation of dispersion as noise."}
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks, "seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
