"""Exploratory CPU audit of count-conditional dispersion in cached answer states.

Run from any directory with --output <new-run-directory>. No inference or paper edits.
The existing discovery NCC layers are frozen before inspecting dispersion outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
SOURCE = REPO / "work/nonthinking_v44_geometry_300_150_136_166_78"
SELECTION = WORKSPACE / "Figures/nonthinking_ncc_selection_20260913/selection.json"
MODELS = ("Qwen3-8B", "Gemma4-E4B")


def bootstrap_dispersion(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased total within-class variance; x is [seed, count, feature].

    Each bootstrap weight vector resamples entire seed profiles across all counts.
    Variance is centered at each resample's own count-specific mean, not at the
    discovery centroid: otherwise mean bias would be mistaken for dispersion.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2 or weights.shape[1] != n or not np.all(weights.sum(1) == n):
        raise ValueError("Invalid seed sample or bootstrap weights")
    centered = x - x.mean(axis=0, keepdims=True)
    gram = np.einsum("scd,tcd->cst", centered, centered, optimize=True)
    diagonal = np.diagonal(gram, axis1=1, axis2=2)
    total = np.einsum("bs,cs->bc", weights, diagonal)
    correction = np.einsum("bs,cst,bt->bc", weights, gram, weights, optimize=True) / n
    variance = (total - correction) / (n - 1)
    if variance.min() < -1e-6:
        raise ValueError("Numerically negative variance")
    point = x.var(axis=0, ddof=1).sum(axis=-1)
    return point, np.maximum(variance, 0)


def summarize_variance(variance: np.ndarray, bootstrap: np.ndarray) -> dict:
    if np.any(variance <= 0):
        raise ValueError("Zero class variance cannot support log-scale trend")
    low, high = slice(0, 3), slice(7, 10)
    ratio = np.sqrt(variance[high].mean() / variance[low].mean())
    boot_ratio = np.sqrt(bootstrap[:, high].mean(1) / bootstrap[:, low].mean(1))
    logn = np.log(np.arange(1, 11, dtype=float))
    slope_weights = (logn - logn.mean()) / np.square(logn - logn.mean()).sum()
    exponent = np.log(np.sqrt(variance)) @ slope_weights
    boot_exponent = np.log(np.sqrt(np.maximum(bootstrap, 1e-30))) @ slope_weights
    return {
        "sd_high_8_10_over_low_1_3": float(ratio),
        "ratio_ci95": np.quantile(boot_ratio, [.025, .975]).tolist(),
        "log_sd_log_N_slope": float(exponent),
        "slope_ci95": np.quantile(boot_exponent, [.025, .975]).tolist(),
    }


def self_check() -> None:
    x = np.arange(60, dtype=float).reshape(5, 4, 3) ** 1.1
    weights = np.array([[1, 1, 1, 1, 1], [2, 0, 1, 1, 1], [5, 0, 0, 0, 0]])
    point, boot = bootstrap_dispersion(x, weights)
    np.testing.assert_allclose(point, x.var(0, ddof=1).sum(-1))
    for i, w in enumerate(weights):
        resampled = x[np.repeat(np.arange(5), w)]
        np.testing.assert_allclose(boot[i], resampled.var(0, ddof=1).sum(-1), atol=1e-9)
    _, shifted = bootstrap_dispersion(x + np.arange(4)[None, :, None] * 1000, weights)
    np.testing.assert_allclose(boot, shifted, atol=1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", default=10000, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()
    started = time.perf_counter()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError("Use a new run directory to preserve prior results")
    self_check()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    protocol = {
        "analysis": "exploratory count-conditional dispersion at the pre-answer query",
        "source": str(SOURCE), "selection_source": str(SELECTION),
        "selected_layer_one_based": {m: selection["selected"][m]["answer_query_scan"]["layer"] + 1 for m in MODELS},
        "sensitivity_layers": "also evaluate the last transformer block; raw dispersion at all layers",
        "discovery_seeds": list(range(1234, 1254)), "confirmation_seeds": list(range(1254, 1264)),
        "counts": list(range(1, 11)), "nominal_length": "10000 canonical Qwen-token passage",
        "conditioning": "all prompts, no correctness or format filtering",
        "probe": "discovery StandardScaler -> unwhitened PCA32 -> Ridge(alpha=1); no clipping or hyperparameter search",
        "raw_dispersion": "sqrt(trace of unbiased within-N covariance / hidden dimension)",
        "readout_dispersion": "unbiased within-N SD of frozen continuous count predictions",
        "bootstrap": "paired seed resampling on confirmation; frozen discovery mapping; pointwise percentile intervals",
        "bootstrap_draws": args.bootstrap_draws, "random_seed": args.seed,
        "comparison": "pooled variance at N=8..10 versus N=1..3; exploratory log SD versus log N slope",
        "limitations": [
            "Reused historical confirmation samples; no new independent replication.",
            "Dispersion across prompts includes content and position differences; it is not measured stochastic neural noise.",
            "An external linear readout is not proof of the model's own causal readout.",
            "Fixed 10k mechanism protocol differs from the behavioral benchmark, including supplied Total: prefix.",
            "Only N=1..10 and Qwen3-8B/Gemma4-E4B are covered; no claim about all counts, lengths or families.",
            "Intervals condition on the fitted discovery basis; they do not include layer selection or probe fitting uncertainty.",
            "Raw high-dimensional distances and projected count noise measure different quantities.",
        ],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    hashes = {str(SELECTION.relative_to(WORKSPACE)): hashlib.sha256(SELECTION.read_bytes()).hexdigest()}
    rng = np.random.default_rng(args.seed)
    sampled = rng.integers(0, 10, size=(args.bootstrap_draws, 10))
    weights = np.stack([(sampled == k).sum(1) for k in range(10)], axis=1)
    all_curves, all_rows, layer_rows, summaries, inventories = [], [], [], {}, []
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), layout="constrained")
    for mi, model in enumerate(MODELS):
        tick = time.perf_counter()
        base = SOURCE / model / "numeric/representation/answer_query_all_layers_v1"
        index = base / "capture_index.jsonl"
        hashes[str(index.relative_to(WORKSPACE))] = hashlib.sha256(index.read_bytes()).hexdigest()
        records = sorted([json.loads(s) for s in index.read_text().splitlines() if s.strip()], key=lambda r: (int(r["seed"]), int(r["count"])))
        expected = [(s, n) for s in range(1234, 1264) for n in range(1, 11)]
        if [(r["seed"], r["count"]) for r in records] != expected:
            raise ValueError("Incomplete or duplicated seed/count panel")
        xs, layer_indices, lengths = [], None, []
        for r in records:
            if r["design_variant"] != "v4.4" or r["answer_format"] != "numeric":
                raise ValueError("Unexpected prompt protocol")
            if r["split"] != ("discovery" if r["seed"] < 1254 else "confirmation"):
                raise ValueError("Incorrect discovery/confirmation split")
            p = base / r["shard_path"]
            hashes[str(p.relative_to(WORKSPACE))] = hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(p, allow_pickle=False) as z:
                li = z["layer_indices"]
                if layer_indices is None:
                    layer_indices = li.copy()
                np.testing.assert_array_equal(li, layer_indices)
                a = z["query_states"].astype(np.float32)
                if not np.isfinite(a).all() or list(a.shape) != r["array_shape"]:
                    raise ValueError("Invalid cached state")
                if int(z["query_position"][0]) != r["query_position"]:
                    raise ValueError("Query position does not match index")
                xs.append(a)
                lengths.append(r["sequence_length"])
        x = np.stack(xs)
        del xs
        lengths = np.array(lengths).reshape(30, 10)
        inventories.append({"model": model, "prompts": len(records), "states_shape": list(x.shape),
                            "mean_native_tokens_by_N": lengths.mean(0).tolist(),
                            "within_seed_length_range": [int(np.ptp(lengths, axis=1).min()), int(np.ptp(lengths, axis=1).max())]})
        y = np.tile(np.arange(1, 11), 20)
        yt = np.tile(np.arange(1, 11), 10)
        selected = protocol["selected_layer_one_based"][model] - 1
        summaries[model] = {}
        for axis, layer in enumerate(layer_indices):
            test = x[200:, axis].reshape(10, 10, -1).astype(np.float64)
            rawvar = test.var(0, ddof=1).mean(-1)
            for n, v in enumerate(rawvar, 1):
                layer_rows.append({"model": model, "layer_one_based": int(layer) + 1, "N": n, "raw_rms": float(np.sqrt(v)), "primary_layer": int(layer) == selected})
            if int(layer) not in {selected, int(layer_indices[-1])}:
                continue
            # Complete held-out seed profiles are retained at every count.
            variance, bootstrap = bootstrap_dispersion(test, weights)
            variance /= test.shape[-1]
            bootstrap /= test.shape[-1]
            scaler = StandardScaler().fit(x[:200, axis])
            train = scaler.transform(x[:200, axis])
            pca = PCA(n_components=32, svd_solver="randomized", random_state=args.seed)
            ztrain = pca.fit_transform(train)
            ztest = pca.transform(scaler.transform(x[200:, axis]))
            probe = Ridge(alpha=1).fit(ztrain, y)
            pred = probe.predict(ztest).astype(np.float64).reshape(10, 10)
            pvar, pboot = bootstrap_dispersion(pred[..., None], weights)
            rawsummary, psummary = summarize_variance(variance, bootstrap), summarize_variance(pvar, pboot)
            record = {"layer_one_based": int(layer) + 1, "raw_dispersion": rawsummary,
                      "readout_dispersion": psummary, "probe_confirmation_r2": float(r2_score(yt, pred.ravel())),
                      "probe_mean_by_N": pred.mean(0).tolist(), "probe_sd_by_N": np.sqrt(pvar).tolist()}
            summaries[model]["primary" if int(layer) == selected else "last_layer"] = record
            low_raw, high_raw = np.quantile(np.sqrt(bootstrap), [.025, .975], axis=0)
            low_p, high_p = np.quantile(np.sqrt(pboot), [.025, .975], axis=0)
            for ni in range(10):
                all_curves.append({"model": model, "layer_one_based": int(layer) + 1, "primary_layer": int(layer) == selected, "N": ni + 1,
                                   "raw_rms": float(np.sqrt(variance[ni])), "raw_rms_ci_low": low_raw[ni], "raw_rms_ci_high": high_raw[ni],
                                   "probe_sd": float(np.sqrt(pvar[ni])), "probe_sd_ci_low": low_p[ni], "probe_sd_ci_high": high_p[ni],
                                   "probe_mean": float(pred[:, ni].mean()), "probe_bias": float(pred[:, ni].mean() - ni - 1),
                                   "confirmation_seeds": 10})
                for si in range(10):
                    all_rows.append({"model": model, "layer_one_based": int(layer) + 1, "seed": si + 1254,
                                     "N": ni + 1, "probe_prediction": pred[si, ni], "native_tokens": lengths[20 + si, ni]})
            if int(layer) == selected:
                color = ["#267DA0", "#C76C39"][mi]
                n = np.arange(1, 11)
                norm = np.sqrt(variance[:3].mean())
                for col, point, lo, hi in [(0, np.sqrt(variance) / norm, low_raw / norm, high_raw / norm), (1, np.sqrt(pvar), low_p, high_p)]:
                    axes[mi, col].plot(n, point, "o-", color=color, lw=1.6, ms=4)
                    axes[mi, col].fill_between(n, lo, hi, color=color, alpha=.18)
                axes[mi, 2].plot(n, pred.mean(0), "o-", color=color, lw=1.6, label="Frozen probe")
                axes[mi, 2].plot(n, n, "--", color="gray", lw=1, label="Correct count")
                axes[mi, 0].set_ylabel(f"{model} / L{int(layer)+1}\nRelative within-N RMS")
                axes[mi, 1].set_ylabel("Within-N SD (count units)")
                axes[mi, 2].set_ylabel("Mean decoded count")
                np.savez_compressed(out / f"{model}_frozen_probe.npz", feature_mean=scaler.mean_, feature_scale=scaler.scale_,
                                    pca_mean=pca.mean_, pca_components=pca.components_, ridge_coef=probe.coef_, ridge_intercept=probe.intercept_)
        print(json.dumps({"model": model, "seconds": time.perf_counter() - tick, "results": summaries[model]}), flush=True)
        del x
    for ax in axes.flat:
        ax.set_xlabel("Total needle count N")
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    axes[0, 0].set_title("All representation directions")
    axes[0, 1].set_title("Frozen count readout direction")
    axes[0, 2].set_title("Readout mean: bias kept separate")
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.suptitle("Non-thinking at 10k: exploratory cached-state analysis\n20 discovery seeds for the probe; 10 confirmation seeds per N", fontsize=12)
    fig.savefig(out / "noise_by_count.png", dpi=180)
    fig.savefig(out / "noise_by_count.pdf")
    plt.close(fig)
    pd.DataFrame(all_curves).to_csv(out / "dispersion_by_count.csv", index=False)
    pd.DataFrame(all_rows).to_csv(out / "confirmation_predictions.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(out / "raw_dispersion_all_layers.csv", index=False)
    (out / "inventory.json").write_text(json.dumps(inventories, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    audit = {"status": "PASS", "self_checks": "explicit resampling matches Gram calculation; per-N mean-shift invariance; complete panels; finite values and matching layers",
             "source_sha256": hashes, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
             "elapsed_seconds": time.perf_counter() - started, "model_inference": False, "paper_edits": False}
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
