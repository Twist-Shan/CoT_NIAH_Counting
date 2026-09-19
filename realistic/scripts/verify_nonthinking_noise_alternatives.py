"""Independent arithmetic checks and joint discovery/confirmation seed bootstrap.

This extension records direction-fitting uncertainty without choosing new layers
or favorable subsets. Projection kernels avoid copying hidden vectors 10000 times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "work/nonthinking_v44_geometry_300_150_136_166_78"


def weighted_projection_bootstrap(train, test, discovery_weights, confirmation_weights):
    """Draw-specific unit directions and pooled within-pair variance.

    Shapes: train=[20,10,d], test=[10,10,d]; weights resample whole seed profiles.
    """
    delta = np.diff(train, axis=1)
    pairs = np.stack([test[:, :-1], test[:, 1:]], axis=-2)
    pairs = pairs - pairs.mean(axis=(0, 2), keepdims=True)
    kernel = np.einsum("tpd,spcd->tspc", delta, pairs, optimize=True)
    gram = np.einsum("tpd,upd->ptu", delta, delta, optimize=True)
    wd = discovery_weights / train.shape[0]
    norm2 = np.einsum("bt,ptu,bu->bp", wd, gram, wd, optimize=True)
    assert (norm2 > 0).all()
    projections = np.einsum("bt,tspc->bspc", wd, kernel, optimize=True) / np.sqrt(norm2)[:, None, :, None]
    n = test.shape[0]
    means = np.einsum("bs,bspc->bpc", confirmation_weights, projections) / n
    second = np.einsum("bs,bspc->bpc", confirmation_weights, projections ** 2)
    var = (second - n * means ** 2) / (n - 1)
    assert var.min() > -1e-8
    noise = np.sqrt(np.maximum(var, 0).mean(-1))
    gap = means[:, :, 1] - means[:, :, 0]
    assert np.all(np.abs(gap) > 1e-10)
    return noise, gap, noise / np.abs(gap), projections


def ratios(a):
    return np.sqrt(np.mean(a[:, 7:] ** 2, axis=1) / np.mean(a[:, :2] ** 2, axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started = time.perf_counter()
    out = args.output.resolve()
    protocol = json.loads((out / "protocol.json").read_text())
    pair_table = pd.read_csv(out / "adjacent_pair_metrics.csv")
    count_table = pd.read_csv(out / "count_metrics.csv")
    rng = np.random.default_rng(1234)
    d = rng.integers(0, 20, (10000, 20))
    c = rng.integers(0, 10, (10000, 10))
    wd = np.stack([(d == s).sum(1) for s in range(20)], axis=1)
    wc = np.stack([(c == s).sum(1) for s in range(10)], axis=1)
    summaries, rows, checks = {}, [], []
    for model in ["Qwen3-8B", "Gemma4-E4B"]:
        layers = [protocol["primary_layers"][model], protocol["secondary_layers"][model]]
        base = SOURCE / model / "numeric/representation/answer_query_all_layers_v1"
        states = []
        for seed in range(1234, 1264):
            for n in range(1, 11):
                with np.load(base / f"shards/v4.4/V4_4_T10000_N{n}_seed{seed}.npz", allow_pickle=False) as z:
                    axes = [z["layer_indices"].tolist().index(layer - 1) for layer in layers]
                    states.append(z["query_states"][axes].astype(np.float64))
        states = np.stack(states)
        summaries[model] = {}
        for ai, layer in enumerate(layers):
            train = states[:200, ai].reshape(20, 10, -1)
            test = states[200:, ai].reshape(10, 10, -1)
            direction = np.diff(train.mean(0), axis=0)
            direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
            q = np.stack([np.stack([test[:, k] @ direction[k], test[:, k+1] @ direction[k]], axis=-1) for k in range(9)], axis=1)
            noise = np.sqrt(q.var(axis=0, ddof=1).mean(axis=-1))
            gap = np.diff(q.mean(axis=0), axis=-1)[:, 0]
            relative = noise / np.abs(gap)
            expected = pair_table[(pair_table.model == model) & (pair_table.layer == layer) & (pair_table.method == "local_centroid_direction")].sort_values("N_lower")
            for name, values in [("pooled_sd", noise), ("mean_gap", gap), ("relative_noise", relative)]:
                np.testing.assert_allclose(values, expected[name], rtol=1e-10, atol=1e-10)
            # Independently evaluate the stored centroid basis on source states.
            with np.load(out / f"{model}_L{layer}_directions.npz", allow_pickle=False) as z:
                np.testing.assert_allclose(direction, z["adjacent_unit_directions"], atol=1e-12)
                for rank in [1, 3, 9]:
                    x = test @ z["centroid_basis"][:rank].T
                    actual = np.sqrt(x.var(0, ddof=1).sum(-1))
                    expected_count = count_table[(count_table.model == model) & (count_table.layer == layer) & (count_table.metric == f"centroid_subspace_rank{rank}")].sort_values("N")
                    np.testing.assert_allclose(actual, expected_count.sd, rtol=1e-10)
            bn, bg, br, bp = weighted_projection_bootstrap(train, test, wd, wc)
            # First 20 bootstrap draws checked by literal resampling and dot products.
            for b in range(20):
                tr = train[d[b]]
                te = test[c[b]]
                v = np.diff(tr.mean(0), axis=0)
                v /= np.linalg.norm(v, axis=-1, keepdims=True)
                p = np.stack([np.stack([te[:, k] @ v[k], te[:, k+1] @ v[k]], axis=-1) for k in range(9)], axis=1)
                pn = np.sqrt(p.var(0, ddof=1).mean(-1))
                pg = np.diff(p.mean(0), axis=-1)[:, 0]
                np.testing.assert_allclose(pn, bn[b], atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(pg, bg[b], atol=1e-9, rtol=1e-9)
            ns = {"noise_ratio": {"point": float(ratios(noise[None])[0]), "joint_ci95": np.quantile(ratios(bn), [.025, .975]).tolist()},
                  "gap_ratio": {"point": float(ratios(np.abs(gap)[None])[0]), "joint_ci95": np.quantile(ratios(np.abs(bg)), [.025, .975]).tolist()},
                  "relative_noise_ratio": {"point": float(ratios(relative[None])[0]), "joint_ci95": np.quantile(ratios(br), [.025, .975]).tolist()},
                  "max_nonpositive_gap_draw_fraction": float((bg <= 0).mean(0).max())}
            summaries[model][f"L{layer}"] = ns
            ci = np.quantile(br, [.025, .975], axis=0)
            for k in range(9):
                rows.append({"model": model, "layer": layer, "primary_layer": layer == layers[0], "N_lower": k + 1,
                             "relative_noise": relative[k], "joint_ci_low": ci[0, k], "joint_ci_high": ci[1, k]})
            checks.append(f"{model} L{layer}: all metrics independently reconstructed; 20 joint-bootstrap draws match literal resampling")
            print(model, layer, json.dumps(ns), flush=True)
    pd.DataFrame(rows).to_csv(out / "joint_bootstrap_relative_noise.csv", index=False)
    (out / "joint_bootstrap_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    verification = {"status": "PASS", "checks": checks,
                    "joint_bootstrap": "Resample discovery seed profiles and confirmation seed profiles independently, 10000 times; refit local directions for every draw. Existing layer choices stay frozen.",
                    "limits": "Historical samples are reused. Layers are not reselected within draws. Intervals are exploratory and not multiplicity-adjusted.",
                    "elapsed_seconds": time.perf_counter() - started,
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
