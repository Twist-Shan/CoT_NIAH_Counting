"""Render archived CoT representation data at the appendix's printed size.

Run extract_data.py first, then this file with scientific Python.
No inference, model selection, PCA fitting, or point subsampling occurs here.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = OUT / "data"
REFERENCE = ROOT / "figures/nonthinking_cue_domain_pca_20260911/build_figure.py"
MODELS = [("Qwen3-8B", "Qwen", "#168DCA", 36),
          ("Gemma4-E4B", "Gemma", "#E87824", 42)]
INK, GRAY, GRID = "#161923", "#8190A5", "#E7E8EE"
INPUTS, AUDITS = {}, {}


def read(path):
    raw = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8-sig")


def rows(name):
    result = list(csv.DictReader(read(DATA / name).split("\n")))
    for row in result:
        for key in ["seed", "count_or_index", "layer_display_one_based", "layer_zero_based"]:
            if key in row and row[key] != "":
                row[key] = int(row[key])
        for key in ["pc1", "pc2", "pc3"]:
            if key in row:
                row[key] = float(row[key])
    return result


def renderer():
    read(REFERENCE)
    spec = importlib.util.spec_from_file_location("cot_reference_pca", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ["STYLE_SOURCE", "GRID_SOURCE", "COLORBAR_SOURCE"]:
        read(getattr(module, name))
    return module


def style():
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.frameon": False, "axes.linewidth": .7,
        "lines.linewidth": 1.5, "lines.markersize": 4, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK,
        "ytick.color": INK, "figure.facecolor": "white", "savefig.facecolor": "white"})


def layout_check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t, b in texts if b.x0 < -.5 or b.y0 < -.5
               or b.x1 > fig.bbox.width + .5 or b.y1 > fig.bbox.height + .5]
    overlaps = []
    for i, (a, ab) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if min(ab.x1, bb.x1) - max(ab.x0, bb.x0) > .8 and min(ab.y1, bb.y1) - max(ab.y0, bb.y0) > .8:
                overlaps.append([a.get_text(), b.get_text()])
    point_overlaps = []
    for ax in fig.axes:
        if not hasattr(ax, "_pca_markers"):
            continue
        for text in ax.texts:
            if text.get_text() not in ["PC1", "PC2", "PC3"]:
                continue
            box = text.get_window_extent(renderer)
            for point, size in ax._pca_markers:
                x, y = ax.transData.transform(point)
                radius = math.sqrt(size) * fig.dpi / 144
                if box.x0 - radius < x < box.x1 + radius and box.y0 - radius < y < box.y1 + radius:
                    point_overlaps.append(text.get_text())
                    break
    return {"text_outside": outside, "text_overlaps": overlaps,
            "axis_label_marker_overlaps": point_overlaps,
            "font_sizes": sorted({t.get_fontsize() for t, _ in texts})}


def save(fig, name, panels):
    checks = layout_check(fig)
    files = {}
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=300)
        files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    AUDITS[name] = {"canvas_inches": list(fig.get_size_inches()), "checks": checks,
                    "panels": panels, "files_sha256": files}
    plt.close(fig)
    if any(checks[k] for k in ["text_outside", "text_overlaps", "axis_label_marker_overlaps"]):
        raise ValueError(f"{name}: {checks}")


def count_key(fig, ref, label, y=.042):
    cmap = ListedColormap([ref.COUNT_COLORS[k] for k in range(1, 11)])
    boundaries = np.arange(.5, 11, 1)
    ax = fig.add_axes([.31, y, .38, .017])
    cb = fig.colorbar(ScalarMappable(norm=BoundaryNorm(boundaries, cmap.N), cmap=cmap),
                     cax=ax, orientation="horizontal", boundaries=boundaries,
                     ticks=[1, 10], spacing="uniform", drawedges=True)
    cb.dividers.set_color("white")
    cb.dividers.set_linewidth(.65)
    cb.outline.set_linewidth(.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.tick_params(labelsize=9, length=2, pad=2, width=.6)
    fig.text(.5, .005, label, fontsize=9, ha="center", va="bottom")


def panel(ax, title, ylabel, xlabel):
    ax.set_title(title, loc="left", fontsize=11, fontweight="normal", pad=7)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7, pad=3)


def canonical_pca(ref, metadata):
    data = [r for r in rows("canonical_selected_pca.csv") if r["split"] == "confirmation"]
    fig = plt.figure(figsize=(6.5, 5.3))
    panels = []
    for i, (endpoint, heading, y, hy) in enumerate([
            ("running_index", "Running index: item-end states", .565, .982),
            ("final_count", "Final count: answer-query states", .125, .542)]):
        fig.text(.5, hy, heading, ha="center", va="top", fontsize=11)
        for j, (model, short, _, _) in enumerate(MODELS):
            meta = metadata["canonical"]["models"][model][endpoint]
            group = [r for r in data if r["model"] == model and r["endpoint"] == endpoint]
            assert len(group) == meta["states_by_split"]["confirmation"]
            assert set(r["seed"] for r in group) == set(range(1254, 1264))
            ax = fig.add_axes([[.035, .535][j], y, .430, .300])
            title = f"{'ABCD'[2*i+j]}. {short} L{meta['selected_layer_display_one_based']}"
            display = ref.draw_panel(ax, [(group, "o", True, "-")], meta["evr"], title, ref.CAMERA)
            panels.append({"model": model, "endpoint": endpoint, "points": len(group),
                           "layer": meta["selected_layer_display_one_based"], "display": display})
    count_key(fig, ref, "Running index / final count")
    ref.place_axis_labels(fig)
    save(fig, "cot_count_pca", panels)


def readouts(metadata):
    data = rows("canonical_layerwise_readouts.csv")
    fig = plt.figure(figsize=(6.5, 4.8))
    fig.legend([Line2D([], [], color=c) for _, _, c, _ in MODELS], [m[1] for m in MODELS],
               loc="upper center", bbox_to_anchor=(.54, 1.004), ncol=2, frameon=False)
    panels = []
    for i, (split, title_split) in enumerate([("discovery_cv", "discovery"), ("confirmation", "confirmation")]):
        for j, (endpoint, name) in enumerate([("running_index", "Running index"), ("final_count", "Final count")]):
            ax = fig.add_axes([[.10, .61][j], [.595, .16][i], .365, .275])
            panel(ax, f"{'ABCD'[2*i+j]}. {name}: {title_split}", "Balanced accuracy", "Layer")
            for model, short, color, n in MODELS:
                rr = sorted([r for r in data if r["model"] == model and r["endpoint"] == endpoint],
                            key=lambda r: r["layer_display_one_based"])
                assert len(rr) == n
                selected = [r for r in rr if r["is_selected_layer"] == "1"]
                assert len(selected) == 1
                for method, ls in [("ncc", "-"), ("logistic", "--")]:
                    values = np.array([float(r[f"{split}_{method}"]) for r in rr])
                    assert np.isfinite(values).all() and np.all((0 <= values) & (values <= 1))
                    ax.plot([r["layer_display_one_based"] for r in rr], values, color=color, ls=ls)
                    s = selected[0]
                    ax.plot(s["layer_display_one_based"], float(s[f"{split}_{method}"]),
                            "o", ms=3.3, color=color, mec="white", mew=.5, zorder=5)
                panels.append({"model": model, "endpoint": endpoint, "split": split,
                               "layers": n, "selected_layer": selected[0]["layer_display_one_based"]})
            ax.set(xlim=(.2, 42.8), xticks=[1, 11, 21, 31, 41], ylim=(0, 1.05), yticks=[0, .25, .5, .75, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.axhline(.1, color=GRAY, ls=":", lw=.8)
    fig.legend([Line2D([], [], color=INK, ls=s) for s in ["-", "--"]],
               ["Nearest centroid", "Logistic"], loc="lower center",
               bbox_to_anchor=(.54, .008), ncol=2, frameon=False, handlelength=1.7)
    save(fig, "cot_count_readouts", panels)


def domain_pca(ref, metadata):
    domain = rows("domain_selected_pca.csv")
    fig = plt.figure(figsize=(6.5, 5.7))
    panels = []
    specs = [("domain", "running_index", "Needle domain: running-index states", ["city", "flower"],
              ["City", "Flower"], .585, .985, .953),
             ("domain", "answer_token", "Needle domain: final-count states", ["city", "flower"],
              ["City", "Flower"], .120, .520, .488)]
    for i, (kind, endpoint, heading, conditions, labels, y, hy, ly) in enumerate(specs):
        fig.text(.5, hy, heading, ha="center", va="top", fontsize=11)
        fig.legend([Line2D([], [], color=GRAY, marker=s, ms=4, lw=.8, ls=ls)
                    for s, ls in [("o", "-"), ("^", "--")]], labels,
                   loc="upper center", bbox_to_anchor=(.5, ly), ncol=2,
                   frameon=False, handlelength=1.6, handletextpad=.4, columnspacing=2, borderaxespad=0)
        for j, (model, short, _, _) in enumerate(MODELS):
            meta = metadata[kind]["models"][model][endpoint]
            groups = []
            for condition in conditions:
                rr = [r for r in domain
                      if r["model"] == model and r["endpoint"] == endpoint
                      and r["domain"] == condition]
                assert rr
                groups.append(rr)
            evr = meta["pca3_explained_variance_ratio"]
            ax = fig.add_axes([[.035, .535][j], y, .430, .280])
            title = f"{'ABCD'[2*i+j]}. {short} L{meta['selected_layer_display_one_based']}"
            display = ref.draw_panel(ax, [(groups[0], "o", True, "-"), (groups[1], "^", False, "--")],
                                     evr, title, ref.DISPLAY_CAMERAS[kind])
            panels.append({"model": model, "endpoint": endpoint, "control": kind,
                           "conditions": conditions, "points_per_condition": list(map(len, groups)),
                           "layer": meta["selected_layer_display_one_based"], "display": display})
    count_key(fig, ref, "Running index / final count", y=.030)
    ref.place_axis_labels(fig)
    save(fig, "cot_domain_pca", panels)


def main():
    start = time.perf_counter()
    style()
    read(Path(__file__).resolve())
    ref = renderer()
    meta = json.loads(read(DATA / "metadata.json"))
    canonical_pca(ref, meta)
    readouts(meta)
    domain_pca(ref, meta)
    for rel, digest in INPUTS.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    output = {"status": "PASS", "model_inference": False, "pca_refit": False,
              "selection_changed": False, "points_subsampled": False,
              "axis_signs_changed": False, "indexing": "one-based layers",
              "source_sha256": INPUTS, "figures": AUDITS,
              "font": "Times New Roman / STIX", "font_points": {"title": 11, "axis": 10, "tick_legend": 9},
              "reference_renderer": REFERENCE.relative_to(ROOT).as_posix(),
              "elapsed_seconds": time.perf_counter() - start}
    (OUT / "figure_manifest.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "figures": list(AUDITS), "seconds": output["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
