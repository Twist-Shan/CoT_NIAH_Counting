"""Render complete layer-by-head score maps from the audited score inventory.

Run build_assets.py first. Frozen-bank ranks are read from the original
membership file, never recomputed from rounded display values.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCORES = HERE / "nonthinking_head_scores.csv"
MEMBERS = (ROOT / "realistic/reports/v4_non-thinking_causal/"
           "v4_4_causal_v2/full_span_topk/full_span_topk_membership.csv")
with SCORES.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
with MEMBERS.open(encoding="utf-8-sig", newline="") as f:
    members = list(csv.DictReader(f))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 10,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "text.color": "#252B31", "axes.labelcolor": "#252B31",
    "xtick.color": "#252B31", "ytick.color": "#252B31",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
fig = plt.figure(figsize=(6.5, 6.8))
specs = [
    ("Qwen3-8B", "#168DCA", list(range(36)), 32, 32,
     [.085, .36, .80, .585], "A  Qwen: full head scores"),
    ("Gemma4-E4B", "#E87824", [5, 11, 17, 23, 29, 35, 41], 8, 6,
     [.085, .075, .80, .18], "B  Gemma: global-attention head scores"),
]
norm = Normalize(0, .6)
annotations = []
inventory = {}
for model, color, layers, nheads, k, rect, title in specs:
    selected = [r for r in members if r["model_label"] == model
                and int(r["top_n"]) == 32 and int(r["rank"]) <= k]
    assert {int(r["rank"]) for r in selected} == set(range(1, k + 1))
    data = {(int(r["layer"]), int(r["head"])): float(r["broad_retrieval_score"])
            for r in rows if r["model"] == model}
    assert set(data) == {(l, h) for l in layers for h in range(nheads)}
    matrix = np.array([[data[l, h] for h in range(nheads)] for l in layers])
    assert np.isfinite(matrix).all() and np.all((matrix >= 0) & (matrix <= .6))
    ax = fig.add_axes(rect)
    cmap = LinearSegmentedColormap.from_list(model, ["#F7FAFC", color], N=256)
    mesh = ax.pcolormesh(np.arange(nheads + 1) - .5,
                        np.arange(len(layers) + 1) - .5, matrix,
                        cmap=cmap, norm=norm, shading="flat",
                        edgecolors="white", linewidth=.18, rasterized=False)
    ax.set_xlim(-.5, nheads - .5)
    ax.set_ylim(len(layers) - .5, -.5)
    xlabels = list(range(0, nheads, 4)) + [31] if nheads == 32 else list(range(nheads))
    ylabels = list(range(0, 36, 3)) + [35] if nheads == 32 else list(range(len(layers)))
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels, [layers[i] for i in ylabels])
    ax.set_xlabel("Head index", labelpad=4)
    ax.set_ylabel("Layer" if nheads == 32 else "Global layer", labelpad=5)
    ax.set_title(title, loc="left", fontweight="bold", pad=9)
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for r in selected:
        layer, head, rank = int(r["layer"]), int(r["head"]), int(r["rank"])
        row = layers.index(layer)
        ax.add_patch(Rectangle((head - .5, row - .5), 1, 1,
                               fill=False, edgecolor="#273743", linewidth=.65))
        text = ax.text(head, row, str(rank), ha="center", va="center",
                       fontsize=7, fontweight="bold", color="#111111")
        annotations.append((text, ax, head, row))
    cax = fig.add_axes([.91, rect[1], .022, rect[3]])
    cb = fig.colorbar(mesh, cax=cax, ticks=[0, .2, .4, .6])
    cb.set_label(r"Broad retrieval score $B_h$", fontsize=8.5, labelpad=5)
    cb.ax.tick_params(labelsize=8, length=2, width=.5, pad=2)
    cb.outline.set_linewidth(.45)
    cb.outline.set_edgecolor("#929CA8")
    inventory[model] = {"shape": list(matrix.shape), "heads": len(data),
                        "selected": k, "layers": layers}

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
visible_text = [(t.get_text(), t.get_window_extent(renderer))
                for t in fig.findobj(matplotlib.text.Text)
                if t.get_visible() and t.get_text()]
outside = [s for s, b in visible_text if b.x0 < 0 or b.y0 < 0
           or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height]
assert not outside, outside
overlaps = []
for i, (s, a) in enumerate(visible_text):
    for t, b in visible_text[i + 1:]:
        if (min(a.x1, b.x1) - max(a.x0, b.x0) > .5
                and min(a.y1, b.y1) - max(a.y0, b.y0) > .5):
            overlaps.append((s, t))
assert not overlaps, overlaps
for text, ax, head, row in annotations:
    box = text.get_window_extent(renderer)
    corners = ax.transData.transform([[head - .5, row - .5], [head + .5, row + .5]])
    assert box.x0 > corners[:, 0].min() and box.x1 < corners[:, 0].max()
    assert box.y0 > corners[:, 1].min() and box.y1 < corners[:, 1].max()

for ext in ["pdf", "svg", "png"]:
    fig.savefig(HERE / f"nonthinking_full_head_scores.{ext}", dpi=300)
plt.close(fig)
audit = {
    "status": "PASS", "inventory": inventory, "scale": [0, .6],
    "palette": {s[0]: s[1] for s in specs},
    "annotations": "Original frozen-bank ranks, inside outlined cells",
    "dimensions_inches": [6.5, 6.8],
    "font_points": {"axis": 8.5, "tick": 8, "title": 10, "rank": 7},
    "text_outside_figure": outside, "text_overlaps": overlaps,
    "all_rank_labels_inside_cells": True,
    "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [SCORES, MEMBERS]},
}
(HERE / "head_maps_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
