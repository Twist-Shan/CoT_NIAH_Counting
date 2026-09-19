"""Compare explicit noise/precision definitions on the existing cached cohort.

Exploratory: all definitions, ranks and both frozen layers are retained. No
inference, layer reselection or paper edits. See protocol.json for estimands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from analyze_nonthinking_noise_by_count import bootstrap_dispersion, summarize_variance, self_check

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "work/nonthinking_v44_geometry_300_150_136_166_78"
PREVIOUS = REPO / "outputs/nonthinking_noise_by_count_20260919"
SELECTION = REPO.parent / "Figures/nonthinking_ncc_selection_20260913/selection.json"


def quantiles(x):
    return np.quantile(x, [.025, .975], axis=0)


def ratio_summary(point, bootstrap, low, high):
    """RMS high/low ratio across specified count or adjacent-pair conditions."""
    ratio = np.sqrt(np.mean(np.square(point[high])) / np.mean(np.square(point[low])))
    draws = np.sqrt(np.mean(np.square(bootstrap[:, high]), axis=1) / np.mean(np.square(bootstrap[:, low]), axis=1))
    return {"high_over_low": float(ratio), "ci95": quantiles(draws).tolist()}


def adjacent_statistics(projected, weights):
    """x[seed,pair,side] holds projections on a discovery-frozen direction.

    No covariance inversion and no claim of optimal Fisher information.
    Estimate the actual confirmation separation and SD separately. Ratios
    whose bootstrap separation crosses zero are flagged, not clipped.
    """
    x = np.asarray(projected, dtype=np.float64)
    ns = len(x)
    means = x.mean(0)
    var = x.var(0, ddof=1)
    noise = np.sqrt(var.mean(-1))
    gap = means[:, 1] - means[:, 0]
    bm = np.einsum("bs,spt->bpt", weights, x) / ns
    centered = x - x.mean(0)
    bc = np.einsum("bs,spt->bpt", weights, centered) / ns
    bv = (np.einsum("bs,spt->bpt", weights, centered ** 2) - ns * bc ** 2) / (ns - 1)
    if bv.min() < -1e-9:
        raise ValueError("Negative bootstrap variance")
    bn = np.sqrt(np.maximum(bv, 0).mean(-1))
    bg = bm[:, :, 1] - bm[:, :, 0]
    if np.any(np.abs(gap) < 1e-12):
        raise ValueError("Zero estimated separation: relative-noise ratio undefined")
    valid_boot = np.abs(bg) > 1e-12
    if not valid_boot.all():
        raise ValueError("A bootstrap separation is zero: report d-prime instead of a finite ratio")
    relative = noise / np.abs(gap)
    br = bn / np.abs(bg)
    # Equal-prior binary AUC for the same frozen direction, with ties worth 1/2.
    diff = x[:, None, :, 1] - x[None, :, :, 0]
    auc = ((diff > 0) + 0.5 * (diff == 0)).mean((0, 1))
    rows = []
    nci, gci, rci = quantiles(bn), quantiles(bg), quantiles(br)
    for j in range(x.shape[1]):
        rows.append({"N_lower": j + 1, "N_upper": j + 2,
                     "pooled_sd": float(noise[j]), "pooled_sd_lo": nci[0, j], "pooled_sd_hi": nci[1, j],
                     "mean_gap": float(gap[j]), "mean_gap_lo": gci[0, j], "mean_gap_hi": gci[1, j],
                     "relative_noise": float(relative[j]), "relative_noise_lo": rci[0, j], "relative_noise_hi": rci[1, j],
                     "signed_dprime": float(gap[j] / noise[j]), "auc": float(auc[j]),
                     "nonpositive_gap_bootstrap_fraction": float(np.mean(bg[:, j] <= 0))})
    result = {"pooled_sd": ratio_summary(noise, bn, slice(0, 2), slice(7, 9)),
              "gap_magnitude": ratio_summary(np.abs(gap), np.abs(bg), slice(0, 2), slice(7, 9)),
              "relative_noise": ratio_summary(relative, br, slice(0, 2), slice(7, 9)),
              "all_gaps_positive": bool((gap > 0).all()),
              "max_nonpositive_gap_bootstrap_fraction": float(np.mean(bg <= 0, axis=0).max()),
              "auc_low": float(auc[:2].mean()), "auc_high": float(auc[7:].mean())}
    return rows, result


def check_adjacent():
    rng = np.random.default_rng(4321)
    z = rng.normal(size=(10, 9, 2))
    z[:, :, 1] += 4
    w = np.ones((2, 10), dtype=int)
    rows, _ = adjacent_statistics(z, w)
    direct = np.sqrt(z.var(0, ddof=1).mean(-1)) / np.abs(np.diff(z.mean(0), axis=1)[:, 0])
    np.testing.assert_allclose([r["relative_noise"] for r in rows], direct)
    rescaled, _ = adjacent_statistics(z * 7 + 100, w)
    np.testing.assert_allclose([r["relative_noise"] for r in rescaled], direct, atol=1e-12)
    for row in rows:
        np.testing.assert_allclose(row["relative_noise_lo"], row["relative_noise_hi"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    args = ap.parse_args()
    started = time.perf_counter()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError("Use a new output directory")
    self_check()
    check_adjacent()
    sel = json.loads(SELECTION.read_text())
    models = ["Qwen3-8B", "Gemma4-E4B"]
    protocol = {
        "status": "exploratory extension after seeing the first audit; not confirmatory preregistration",
        "primary_layers": {m: sel["selected"][m]["answer_query_scan"]["layer"] + 1 for m in models},
        "secondary_layers": {"Qwen3-8B": 36, "Gemma4-E4B": 42},
        "discovery": list(range(1234, 1254)), "confirmation": list(range(1254, 1264)),
        "counts": list(range(1, 11)), "context": "10000 canonical passage tokens",
        "definitions": {
            "centroid_subspace_rms": "Within-N RMS distance in the raw-space orthonormal span of discovery count centroids; ranks 1,3,9, with 3 as summary rank.",
            "adjacent_tangent_sd": "Pooled confirmation SD along each discovery centroid-difference unit vector in raw hidden space.",
            "adjacent_relative_noise": "Pooled SD divided by absolute confirmation mean separation, both on the frozen local direction; inverse absolute projected d-prime.",
            "ridge_relative_noise": "Same adjacent-pair ratio after the frozen global count readout; PCA16/32/64 with Ridge(alpha=1).",
            "orthogonal_rms": "Remaining original-space within-N variance after removing the three-dimensional discovery centroid subspace.",
            "angle_dispersion": "RMS within-N Euclidean dispersion after normalizing each state to unit norm; a scale-invariant angular-spread diagnostic.",
        },
        "bootstrap": "10000 paired confirmation-seed resamples, frozen discovery mappings; unadjusted pointwise percentile intervals",
        "comparison": "N=8..10 versus N=1..3 for per-count SD; pairs 8/9 and 9/10 versus 1/2 and 2/3 for local measures",
        "output_probabilities": "Matching historical numeric attention captures mark candidate probabilities as deferred; entropy unavailable from these files. Greedy counts cannot reconstruct a per-prompt probability distribution.",
        "limits": ["No model reruns or selection on correctness.", "All predefined ranks and both layers retained.",
                   "Higher relative noise may reflect smaller count separation rather than larger absolute variance.",
                   "Cached states vary across input content/positions; stochastic inference noise is not measured.",
                   "The same historical confirmation seeds are reused; new results are exploratory.",
                   "The directions are external measurements, not validated model-used causal readouts.",
                   "CIs condition on discovery directions; denominator uncertainty is resampled, direction-learning uncertainty is not.",
                   "No multiple-comparison correction; no single selected positive metric is treated as a mechanism proof."],
        "method_sources": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC4451760/", "https://www.nature.com/articles/s41467-021-26793-9"],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rng = np.random.default_rng(1234)
    draws = rng.integers(0, 10, (args.bootstrap_draws, 10))
    weights = np.stack([(draws == i).sum(1) for i in range(10)], axis=1)
    summaries, count_rows, pair_rows, prediction_rows, hashes = {}, [], [], [], {}
    hashes[str(SELECTION)] = hashlib.sha256(SELECTION.read_bytes()).hexdigest()
    previous = pd.read_csv(PREVIOUS / "confirmation_predictions.csv")
    for model in models:
        tick = time.perf_counter()
        primary = protocol["primary_layers"][model]
        layers = [primary, protocol["secondary_layers"][model]]
        base = SOURCE / model / "numeric/representation/answer_query_all_layers_v1"
        idx = base / "capture_index.jsonl"
        records = sorted([json.loads(s) for s in idx.read_text().splitlines() if s], key=lambda r: (r["seed"], r["count"]))
        assert [(r["seed"], r["count"]) for r in records] == [(s, n) for s in range(1234, 1264) for n in range(1, 11)]
        hashes[str(idx)] = hashlib.sha256(idx.read_bytes()).hexdigest()
        states = []
        for r in records:
            assert r["design_variant"] == "v4.4" and r["answer_format"] == "numeric"
            assert r["split"] == ("discovery" if r["seed"] < 1254 else "confirmation")
            p = base / r["shard_path"]
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(p, allow_pickle=False) as z:
                axes = [z["layer_indices"].tolist().index(layer - 1) for layer in layers]
                x = z["query_states"][axes].astype(np.float32)
                assert np.isfinite(x).all()
                states.append(x)
        states = np.stack(states)
        summaries[model] = {}
        for axis, layer in enumerate(layers):
            train = states[:200, axis].reshape(20, 10, -1).astype(np.float64)
            test = states[200:, axis].reshape(10, 10, -1).astype(np.float64)
            result = {}
            means = train.mean(0)
            centered_means = means - means.mean(0)
            _, svals, basis = np.linalg.svd(centered_means, full_matrices=False)
            assert np.linalg.matrix_rank(centered_means) == 9
            total_var, total_boot = bootstrap_dispersion(test, weights)
            def add_count(metric, values, variance=None, bootstrap=None):
                if variance is None:
                    variance, bootstrap = bootstrap_dispersion(values, weights)
                result[metric] = summarize_variance(variance, bootstrap)
                ci = quantiles(np.sqrt(bootstrap))
                for j in range(10):
                    count_rows.append({"model": model, "layer": layer, "primary_layer": layer == primary,
                                       "metric": metric, "N": j + 1, "sd": float(np.sqrt(variance[j])),
                                       "ci_low": ci[0, j], "ci_high": ci[1, j]})
            add_count("raw_total_rms", None, total_var / test.shape[-1], total_boot / test.shape[-1])
            for rank in [1, 3, 9]:
                projected = test @ basis[:rank].T
                v, b = bootstrap_dispersion(projected, weights)
                add_count(f"centroid_subspace_rank{rank}", None, v, b)
                if rank == 3:
                    # Orthogonal decomposition is in raw Euclidean space.
                    assert np.all(total_var >= v - 1e-8)
                    assert np.all(total_boot >= b - 1e-7)
                    add_count("orthogonal_to_centroid_rank3", None, np.maximum(total_var - v, 0), np.maximum(total_boot - b, 0))
                    result["centroid_rank3_mean_signal_fraction"] = float((svals[:3] ** 2).sum() / (svals ** 2).sum())
            norm = np.linalg.norm(test, axis=-1, keepdims=True)
            assert (norm > 0).all()
            add_count("unit_norm_state_dispersion", test / norm)
            directions = np.diff(means, axis=0)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            adjacent = np.stack([np.stack([test[:, k] @ directions[k], test[:, k+1] @ directions[k]], axis=-1) for k in range(9)], axis=1)
            rows, metrics = adjacent_statistics(adjacent, weights)
            result["local_centroid_direction"] = metrics
            for row in rows:
                pair_rows.append({"model": model, "layer": layer, "primary_layer": layer == primary, "method": "local_centroid_direction", **row})
            np.savez_compressed(out / f"{model}_L{layer}_directions.npz", centroid_basis=basis[:9], adjacent_unit_directions=directions,
                                discovery_centroids=means, confirmation_pair_projections=adjacent,
                                confirmation_centroid_projections=test @ basis[:9].T)
            for rank in [16, 32, 64]:
                # Match the previous script's float32 preprocessing exactly.
                xtr = states[:200, axis]
                xte = states[200:, axis]
                scaler = StandardScaler().fit(xtr)
                pca = PCA(n_components=rank, svd_solver="randomized", random_state=1234)
                ztr = pca.fit_transform(scaler.transform(xtr))
                zte = pca.transform(scaler.transform(xte))
                ridge = Ridge(alpha=1).fit(ztr, np.tile(np.arange(1, 11), 20))
                pred = ridge.predict(zte).astype(np.float64).reshape(10, 10)
                if rank == 32:
                    old = previous[(previous.model == model) & (previous.layer_one_based == layer)].sort_values(["seed", "N"])
                    np.testing.assert_allclose(pred.ravel(), old.probe_prediction, atol=2e-5, rtol=2e-5)
                add_count(f"ridge_pca{rank}_sd", pred[..., None])
                pair = np.stack([pred[:, :-1], pred[:, 1:]], axis=-1)
                rows, metrics = adjacent_statistics(pair, weights)
                result[f"ridge_pca{rank}_adjacent"] = metrics
                for row in rows:
                    pair_rows.append({"model": model, "layer": layer, "primary_layer": layer == primary, "method": f"ridge_pca{rank}", **row})
                for i, values in enumerate(pred):
                    for j, value in enumerate(values):
                        prediction_rows.append({"model": model, "layer": layer, "pca_rank": rank, "seed": i + 1254, "N": j + 1, "prediction": value})
            summaries[model][f"L{layer}"] = result
            print(json.dumps({"model": model, "layer": layer, "rank3": result["centroid_subspace_rank3"],
                              "local": result["local_centroid_direction"], "ridge32": result["ridge_pca32_adjacent"]}), flush=True)
        print(f"{model}: {time.perf_counter() - tick:.2f}s", flush=True)
    counts = pd.DataFrame(count_rows)
    pairs = pd.DataFrame(pair_rows)
    counts.to_csv(out / "count_metrics.csv", index=False)
    pairs.to_csv(out / "adjacent_pair_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "readout_sensitivity_predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    # Summarize all metrics in a compact table, including contrary outcomes.
    comparison = []
    for model, levels in summaries.items():
        for level, metrics in levels.items():
            for metric, val in metrics.items():
                if isinstance(val, dict) and "sd_high_8_10_over_low_1_3" in val:
                    comparison.append({"model": model, "layer": level, "metric": metric,
                                       "ratio": val["sd_high_8_10_over_low_1_3"], "ci_low": val["ratio_ci95"][0], "ci_high": val["ratio_ci95"][1]})
                elif isinstance(val, dict) and "relative_noise" in val:
                    for part in ["pooled_sd", "gap_magnitude", "relative_noise"]:
                        v = val[part]
                        comparison.append({"model": model, "layer": level, "metric": f"{metric}/{part}",
                                           "ratio": v["high_over_low"], "ci_low": v["ci95"][0], "ci_high": v["ci95"][1]})
    pd.DataFrame(comparison).to_csv(out / "all_comparisons.csv", index=False)
    plot_results(out, counts, pairs, protocol)
    audit = {"status": "PASS", "checks": ["bootstrap explicit resampling and mean-shift invariance", "relative-noise scale/offset invariance", "complete 300-prompt/model panels", "PCA32 predictions reproduce previous audit", "orthogonal variance decomposition nonnegative"],
             "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
             "source_sha256": hashes, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "command": sys.argv, "elapsed_seconds": time.perf_counter() - started, "inference": False, "paper_edits": False}
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def plot_results(out, counts, pairs, protocol):
    fig, axes = plt.subplots(2, 4, figsize=(13.1, 6.2), layout="constrained")
    for i, model in enumerate(["Qwen3-8B", "Gemma4-E4B"]):
        color = ["#267DA0", "#C76C39"][i]
        group = counts[(counts.model == model) & counts.primary_layer & (counts.metric == "centroid_subspace_rank3")].sort_values("N")
        axes[i, 0].plot(group.N, group.sd, "o-", color=color, ms=4)
        axes[i, 0].fill_between(group.N, group.ci_low, group.ci_high, color=color, alpha=.18)
        pairs_model = pairs[(pairs.model == model) & pairs.primary_layer & (pairs.method == "local_centroid_direction")].sort_values("N_lower")
        for col, key in enumerate(["pooled_sd", "mean_gap", "relative_noise"], 1):
            x = pairs_model.N_lower + .5
            axes[i, col].plot(x, pairs_model[key], "o-", color=color, ms=4)
            axes[i, col].fill_between(x, pairs_model[key + ("_lo" if key != "pooled_sd" else "_lo")], pairs_model[key + "_hi"], color=color, alpha=.18)
            axes[i, col].set_xlabel("Adjacent counts (pair midpoint)")
        axes[i, 0].set_xlabel("Total count N")
        axes[i, 0].set_ylabel(f"{model} / L{protocol['primary_layers'][model]}\nWithin-N RMS")
        axes[i, 1].set_ylabel("Pooled projected SD")
        axes[i, 2].set_ylabel("Projected mean separation")
        axes[i, 3].set_ylabel("SD / mean separation")
    for ax, title in zip(axes[0], ["Count-centroid subspace (rank 3)", "Local counting direction: noise", "Local counting direction: signal", "Local noise relative to signal"]):
        ax.set_title(title, fontsize=10)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.suptitle("Alternative noise definitions on frozen discovery directions\n10 confirmation seeds; shaded pointwise 95% seed-bootstrap intervals", fontsize=12)
    fig.savefig(out / "alternative_noise_definitions.png", dpi=175)
    fig.savefig(out / "alternative_noise_definitions.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
