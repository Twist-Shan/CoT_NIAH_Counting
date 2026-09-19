"""Plot the full N=1..10 adjacent-count relative-noise curve at NCC layers."""
from pathlib import Path
import hashlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / "realistic/outputs/nonthinking_noise_alternatives_20260919"
SELECTION = ROOT / "figures/nonthinking_ncc_selection_20260913/selection.json"


def main():
    started = time.perf_counter()
    selection = json.loads(SELECTION.read_text())
    joint = pd.read_csv(DATA / "joint_bootstrap_relative_noise.csv")
    detail = pd.read_csv(DATA / "adjacent_pair_metrics.csv")
    selected_rows = []
    styles = [
        ("Qwen3-8B", "Qwen3-8B", "#168DCA", "o", "-"),
        ("Gemma4-E4B", "Gemma-4-E4B", "#E87824", "s", "--"),
    ]
    plt.rcParams.update({
        "font.family": "Times New Roman", "mathtext.fontset": "stix",
        "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 12,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 10, "axes.linewidth": .7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#161923", "axes.labelcolor": "#161923",
        "xtick.color": "#161923", "ytick.color": "#161923",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(6.5, 2.85))
    fig.subplots_adjust(left=.10, right=.985, bottom=.22, top=.97)
    upper = 0
    for model, display, color, marker, linestyle in styles:
        choice = selection["selected"][model]["answer_query_scan"]
        layer = int(choice["layer"]) + 1
        rows = joint[(joint.model == model) & (joint.layer == layer)].sort_values("N_lower").copy()
        assert rows.N_lower.tolist() == list(range(1, 10)) and rows.primary_layer.all()
        original = detail[(detail.model == model) & (detail.layer == layer) & (detail.method == "local_centroid_direction")].sort_values("N_lower")
        np.testing.assert_allclose(rows.relative_noise, original.relative_noise, rtol=1e-12)
        np.testing.assert_allclose(rows.relative_noise, original.pooled_sd / original.mean_gap.abs(), rtol=1e-12)
        assert np.isfinite(rows[["relative_noise", "joint_ci_low", "joint_ci_high"]]).all().all()
        assert (rows.joint_ci_low >= 0).all() and (rows.joint_ci_high >= rows.joint_ci_low).all()
        x = rows.N_lower.to_numpy() + .5
        y = rows.relative_noise.to_numpy()
        low, high = rows.joint_ci_low.to_numpy(), rows.joint_ci_high.to_numpy()
        upper = max(upper, high.max())
        ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0, zorder=1)
        line, = ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.65,
                        marker=marker, markersize=4.8, markeredgecolor="white",
                        markeredgewidth=.6, zorder=3, label=display)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        for (_, r), (_, d) in zip(rows.iterrows(), original.iterrows()):
            selected_rows.append({"model": model, "layer_one_based": layer,
                                  "N_lower": int(r.N_lower), "N_upper": int(r.N_lower) + 1,
                                  "pair_midpoint": r.N_lower + .5,
                                  "relative_noise": r.relative_noise,
                                  "joint_ci95_low": r.joint_ci_low, "joint_ci95_high": r.joint_ci_high,
                                  "pooled_sd": d.pooled_sd, "mean_separation": d.mean_gap,
                                  "discovery_seeds": 20, "confirmation_seeds_per_count": 10})
    ax.set_xlim(1.25, 9.75)
    ax.set_xticks(np.arange(1, 10) + .5, [f"{n}\N{EN DASH}{n+1}" for n in range(1, 10)])
    ymax = np.ceil((upper + .05) * 2) / 2
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + .25, .5))
    ax.set_xlabel("Adjacent counts", labelpad=7)
    ax.set_ylabel("Relative noise", labelpad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#727986")
    ax.tick_params(axis="both", width=.6, length=3.5)
    ax.grid(axis="y", color="#E7E8EE", linewidth=.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, handlelength=2.1, labelspacing=.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boundary = fig.bbox
    outside = []
    for item in [*fig.texts, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        box = item.get_window_extent(renderer)
        if box.x0 < boundary.x0 - 1 or box.y0 < boundary.y0 - 1 or box.x1 > boundary.x1 + 1 or box.y1 > boundary.y1 + 1:
            outside.append(item.get_text())
    assert not outside, outside
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"relative_noise_n1_to_10.{ext}", dpi=300)
    plt.close(fig)
    pd.DataFrame(selected_rows).to_csv(OUT / "plot_data.csv", index=False)
    sources = [SELECTION, DATA / "joint_bootstrap_relative_noise.csv", DATA / "adjacent_pair_metrics.csv", DATA / "verification.json"]
    manifest = {
        "status": "PASS", "layers": {m: int(selection["selected"][m]["answer_query_scan"]["layer"]) + 1 for m, *_ in styles},
        "layer_indexing": "one-based", "selection_target": "final count at answer query, discovery NCC scan",
        "definition": "For each adjacent pair, pooled within-count SD divided by absolute confirmation mean separation on its discovery-frozen centroid-difference direction.",
        "points": "9 adjacent-count pairs per model, spanning all counts N=1..10; plotted at pair midpoints",
        "intervals": "10000 independent discovery-seed and confirmation-seed profile bootstrap draws, refitting directions; original layer selection stays frozen",
        "curves": "measured points joined by straight lines; no smoothing, interpolation or trend constraint",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "canvas_inches": [6.5, 2.85], "text_outside_canvas": outside,
        "annotation_policy": "Only axes and model legend in the graphic; layer, setting, metric definition and interval description are in the manuscript caption.",
        "elapsed_seconds": time.perf_counter() - started, "model_inference": False, "paper_edits": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["status", "layers", "points", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
