"""Build the complete head-score CSV and compact representation diagnostics.

Uses archived analysis tables only; no inference or refitting is performed.
The manuscript installs these generated assets after visual/coverage checks.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

START = time.perf_counter()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / "realistic/reports/v4_non-thinking_causal"
MODELS = ["Qwen3-8B", "Gemma4-E4B"]
COLORS = {"Qwen3-8B": "#168DCA", "Gemma4-E4B": "#E87824"}
INPUTS = {}


def read_csv(path):
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return list(csv.DictReader(data.decode("utf-8-sig").splitlines()))


atlas = [r for r in read_csv(REPORT / "v4_4/realistic_niah_v4_head_atlas.csv")
         if r["variant"] == "v4.4" and r["pooling"] == "span_sum"]
membership = read_csv(REPORT / "v4_4_causal_v2/full_span_topk/full_span_topk_membership.csv")
selected = {
    m: {(int(r["layer"]), int(r["head"])) for r in membership
        if r["model_label"] == m and int(r["top_n"]) == 32 and int(r["rank"]) <= k}
    for m, k in zip(MODELS, [32, 6])
}
rows = []
lookup = {}
for r in atlas:
    m, layer, head = r["model"], int(r["layer"]), int(r["head"])
    score, mass, coverage = (float(r[name]) for name in
                             ["pool_primary", "pool_sum", "pool_coverage"])
    assert all(np.isfinite(v) and 0 <= v <= 1 for v in [score, mass, coverage])
    assert int(r["seeds"]) == 20 and int(r["examples"]) == 180
    key = (m, layer, head)
    assert key not in lookup
    row = dict(model=m, layer=layer, head=head, broad_retrieval_score=score,
               needle_attention_mass=mass, effective_coverage=coverage,
               selected=(layer, head) in selected[m], seeds=20, prompts=180)
    rows.append(row)
    lookup[key] = row
expected = {
    "Qwen3-8B": {(l, h) for l in range(36) for h in range(32)},
    "Gemma4-E4B": {(l, h) for l in [5, 11, 17, 23, 29, 35, 41] for h in range(8)},
}
for m in MODELS:
    assert {(r["layer"], r["head"]) for r in rows if r["model"] == m} == expected[m]
assert [len(selected[m]) for m in MODELS] == [32, 6]
rows.sort(key=lambda r: (MODELS.index(r["model"]), r["layer"], r["head"]))
with (HERE / "nonthinking_head_scores.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)



# Retain needle-end diagnostics and the requested answer-query classifier sweep.
ridge = read_csv(REPORT / "v4_4_extension/geometry/count_regression_summary.csv")
ranks = read_csv(REPORT / "v4_4_extension/geometry/rank_and_compression_by_layer.csv")
classifiers = {
    m: read_csv(REPORT / f"v4_4_extension/classification/classification_all_{short}/answer_classifier_metrics.csv")
    for m, short in zip(MODELS, ["qwen", "gemma"])
}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": .6, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.35))
fig.subplots_adjust(left=.083, right=.99, bottom=.205, top=.76, wspace=.47)
fig.legend([Line2D([0], [0], color=c, lw=2) for c in COLORS.values()],
           ["Qwen", "Gemma"], loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(.5, 1.025))
for ax, title, ylabel in zip(
    axs, ["A  Running-index readout", "B  Needle-end geometry", "C  Final-count readout"],
    [r"Held-out $R^2$", "Variance in top 3 PCs", "Classification accuracy"],
):
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Layer")
    ax.set_xlim(-1, 42)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#929CA8")
    ax.tick_params(width=.6, length=3, pad=3, color="#929CA8")
    ax.grid(axis="y", color="#E2E8EE", linewidth=.6)
    ax.set_axisbelow(True)

for model, color in COLORS.items():
    rr = sorted([r for r in ridge if r["model_label"] == model
                 and r["role"] == "prompt_running" and r["algorithm"] == "ridge"],
                key=lambda r: int(r["layer"]))
    gg = sorted([r for r in ranks if r["model_label"] == model
                 and r["role"] == "prompt_running"], key=lambda r: int(r["layer"]))
    assert len(rr) == len(gg) == {"Qwen3-8B": 36, "Gemma4-E4B": 42}[model]
    axs[0].plot([int(r["layer"]) for r in rr], [float(r["r2_mean"]) for r in rr],
                color=color)
    for metric, style in [("total_variance_capture_k3", "-"),
                          ("centroid_curve_capture_k3", "--")]:
        axs[1].plot([int(r["layer"]) for r in gg], [float(r[metric]) for r in gg],
                    color=color, ls=style)
    for algorithm, style in [("nearest_centroid", "-"), ("logistic_l2", "--")]:
        cc = sorted([r for r in classifiers[model] if r["algorithm"] == algorithm],
                    key=lambda r: int(r["layer"]))
        assert len(cc) == len(rr)
        axs[2].plot([int(r["layer"]) for r in cc], [float(r["accuracy"]) for r in cc],
                    color=color, ls=style)
axs[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
axs[1].legend([Line2D([0], [0], color="#46505B", ls=s) for s in ["-", "--"]],
              ["Individual states", "Centroids"], loc="center",
              bbox_to_anchor=(.535, .875), bbox_transform=fig.transFigure,
              ncol=2, columnspacing=.8, frameon=False, handlelength=1.6,
              borderaxespad=0)
axs[2].set_ylim(0, .88)
axs[2].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
axs[2].axhline(.1, color="#87919C", ls=":", lw=.8)
axs[2].legend([Line2D([0], [0], color="#46505B", ls=s) for s in ["-", "--"]],
              ["Nearest centroid", "Logistic"], loc="upper left",
              frameon=False, handlelength=1.5, borderaxespad=.1)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for text in fig.findobj(matplotlib.text.Text):
    if text.get_visible() and text.get_text():
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, text.get_text()
for ext in ["pdf", "svg", "png"]:
    fig.savefig(HERE / f"nonthinking_representation_diagnostics.{ext}", dpi=300)
plt.close(fig)
manifest = {
    "status": "PASS", "inputs": INPUTS, "elapsed_seconds": time.perf_counter()-START,
    "score_rows_by_model": dict(Counter(r["model"] for r in rows)),
    "selected_heads_by_model": {m: len(selected[m]) for m in MODELS},
    "coverage": "All Qwen heads; all measured Gemma global-attention heads",
    "scoring": "Mean per-prompt full-span B_h; 20 discovery seeds x counts 2--10",
    "csv_precision": "full floating-point precision",
    "head_map_builder": "build_head_maps.py",
    "geometry": "Needle-end ridge/PCA and answer-query classification; unchanged archived values",
    "palette": COLORS,
}
(HERE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
