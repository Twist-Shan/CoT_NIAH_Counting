"""Render the Non-thinking head map in the current Thinking appendix style.

Run from the workspace with python -s <this file>.
Reads the existing full-score table and frozen head membership without refitting.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SCORES = ROOT / "figures/nonthinking_appendix_20260911/nonthinking_head_scores.csv"
MEMBERS = ROOT / ("realistic/reports/v4_non-thinking_causal/"
                  "v4_4_causal_v2/full_span_topk/full_span_topk_membership.csv")
STYLE = ROOT / "figures/thinking_appendix_style_20260913/build_figures.py"
REFERENCE = ROOT / "figures/enumeration_thinking_style_20260917/build_figures.py"
MODELS = [("Qwen3-8B", "Qwen", "#168DCA", list(range(36)), 32, 32),
          ("Gemma4-E4B", "Gemma", "#E87824", [5, 11, 17, 23, 29, 35, 41], 8, 6)]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    spec = importlib.util.spec_from_file_location("thinking_appendix_style", STYLE)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.OUT = OUT
    style.setup()
    scores, members = read_csv(SCORES), read_csv(MEMBERS)
    assert len(scores) == 1208
    fig = plt.figure(figsize=(6.5, 3.5))
    plotted, inventories, axes_bounds, colorbar_bounds = [], {}, [], []

    for j, (model, short, color, layers, nheads, k) in enumerate(MODELS):
        rows = [r for r in scores if r["model"] == model]
        data = {(int(r["layer"]), int(r["head"])): r for r in rows}
        assert len(data) == len(rows) == len(layers) * nheads
        assert set(data) == {(layer, head) for layer in layers for head in range(nheads)}
        selected = [r for r in members if r["model_label"] == model
                    and int(r["top_n"]) == 32 and int(r["rank"]) <= k]
        assert {int(r["rank"]) for r in selected} == set(range(1, k + 1))
        ranks = {(int(r["layer"]), int(r["head"])): int(r["rank"]) for r in selected}
        assert len(ranks) == k and set(ranks) <= set(data)
        assert {key for key, r in data.items() if r["selected"].lower() == "true"} == set(ranks)
        matrix = np.array([[float(data[layer, head]["broad_retrieval_score"])
                            for head in range(nheads)] for layer in layers])
        assert np.isfinite(matrix).all() and 0 <= matrix.min() <= matrix.max() <= 1

        rect = [[.095, .59][j], .18, .335, .66]
        ax = fig.add_axes(rect)
        mesh = ax.pcolormesh(np.arange(nheads + 1) + .5,
                            np.arange(len(layers) + 1) + .5, matrix,
                            cmap=LinearSegmentedColormap.from_list(short, ["#F7FAFC", color]),
                            norm=Normalize(0, 1), edgecolors="white", linewidth=.18,
                            rasterized=False)
        ax.set(xlim=(.5, nheads + .5), ylim=(len(layers) + .5, .5))
        xticks = [1, 9, 17, 25, 32] if j == 0 else list(range(1, 9))
        yrows = [0, 5, 11, 17, 23, 29, 35] if j == 0 else list(range(7))
        ax.set_xticks(xticks)
        ax.set_yticks([i + 1 for i in yrows], [layers[i] + 1 for i in yrows])
        ax.set_xlabel("Head index", labelpad=4)
        ax.set_ylabel("Layer" if j == 0 else "Global layer", labelpad=4)
        ax.set_title(f'{"AB"[j]}. {short}', loc="left", fontsize=11,
                     fontweight="normal", pad=8)
        ax.tick_params(length=0, pad=4)
        ax.spines[:].set_visible(False)
        for layer, head in ranks:
            ax.add_patch(Rectangle((head + .5, layers.index(layer) + .5), 1, 1,
                                   fill=False, edgecolor="#273743", linewidth=.5))

        cb_rect = [[.443, .938][j], .18, .015, .66]
        cb = fig.colorbar(mesh, cax=fig.add_axes(cb_rect), ticks=[0, .5, 1])
        cb.ax.tick_params(labelsize=9, length=2, width=.5, pad=2)
        cb.outline.set_linewidth(.5)
        axes_bounds.append(rect)
        colorbar_bounds.append(cb_rect)
        inventories[model] = {
            "shape": list(matrix.shape), "heads": len(rows), "selected": k,
            "layers_display_one_based": [layer + 1 for layer in layers],
            "minimum_score": float(matrix.min()), "maximum_score": float(matrix.max()),
            "frozen_bank": [dict(layer=int(r["layer"]) + 1, head=int(r["head"]) + 1,
                                 rank=int(r["rank"])) for r in selected],
        }
        for (layer, head), row in data.items():
            plotted.append(dict(row, layer_display_one_based=layer + 1,
                                head_display_one_based=head + 1,
                                frozen_bank_rank=ranks.get((layer, head))))

    fig.text(.5, .975, r"Broad retrieval score $B_h$ (shared 0--1 scale)",
             ha="center", va="top", fontsize=9)
    name = "nonthinking_full_head_scores"
    style.save(fig, name, plotted, {
        "models": inventories, "axis_order": "x=head; y=layer, shallow to deep downward",
        "color_range": [0, 1], "in_cell_rank_labels": False,
        "axes_bounds": axes_bounds, "colorbar_bounds": colorbar_bounds,
    })
    (OUT / "plot_data.json").write_text(json.dumps(plotted, indent=2) + "\n", encoding="utf-8")
    sources = [SCORES, MEMBERS, STYLE, REFERENCE, Path(__file__)]
    manifest = {
        "status": "PASS", "style_reference": REFERENCE.relative_to(ROOT).as_posix(),
        "figure": style.FIGURES[name], "statistics_recomputed": False,
        "indexing": {"source": "zero-based", "display": "one-based layers and heads"},
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
