"""Render archived enumeration PCA3 coordinates in the existing appendix style.

No inference, PCA fitting, layer selection, axis-sign selection, or subsampling
occurs here. Coordinates are the archived confirmation projections fitted on
discovery states. Run with Python/NumPy/Matplotlib from any working directory.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import numpy as np

from appendix_style import apply_style, font_profile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / "realistic"
PAPER = ROOT / "runs/paper_figures/figures"
DATA = OUT / "data"
REPORT = REPO / "reports/NiaH_Enumeration_report.html"
REFERENCE = ROOT / "figures/nonthinking_cue_domain_pca_20260911/build_figure.py"
METRICS = REPO / "work/v6_report_remote/native_aligned_representation"
INPUT_HASHES: dict[str, str] = {}


def read(path: Path) -> str:
    raw = path.read_bytes()
    INPUT_HASHES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8-sig")


def reference_renderer():
    """Reuse the checked Nonthinking renderer, including count palette/grid."""
    read(REFERENCE)
    spec = importlib.util.spec_from_file_location("enumeration_reference_pca", REFERENCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load checked PCA renderer: {REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ["STYLE_SOURCE", "GRID_SOURCE", "COLORBAR_SOURCE"]:
        read(getattr(module, name))
    return module


def source_data():
    source = read(REPORT)
    marker = "const ENUM_GEOMETRY="
    if source.count(marker) != 1:
        raise ValueError("Expected exactly one archived PCA coordinate payload")
    payload, _ = json.JSONDecoder().raw_decode(source.split(marker, 1)[1])
    match = re.search(r'<script id="report-manifest" type="application/json">(.*?)</script>', source, re.S)
    if match is None:
        raise ValueError("Missing report provenance manifest")
    manifest = json.loads(match.group(1))
    if (payload["status"] != "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
            or payload["fit_split"] != "discovery" or payload["display_split"] != "confirmation"):
        raise ValueError("PCA fit/display split contract changed")
    selected = {}
    for endpoint, name in [("running", "running_index_selected.csv"), ("final", "final_count_selected.csv")]:
        for row in csv.DictReader(read(METRICS / name).splitlines()):
            key = row["prompt_mode"] + "|" + row["model_label"]
            selected[(endpoint, key)] = row
    alignment = json.loads(read(METRICS / "alignment_audit.json"))
    if alignment["analysis_population"] != "original_registered_all_sample_panel":
        raise ValueError("Representation population changed")
    read(METRICS / "analysis_manifest.json")
    return payload, selected, manifest, alignment


def panel_data(payload, selected, endpoint, key):
    entry = payload[endpoint][key]
    layer = int(entry["default_layer"])
    selection = selected[(endpoint, key)]
    if layer != int(selection["layer"]):
        raise ValueError("Archived layer differs from discovery-selected layer")
    layer_data = entry["layers"][str(layer)]
    expected_rows = 518 if endpoint == "running" else 100
    expected_discovery = 1056 if endpoint == "running" else 200
    if (len(layer_data["rows"]) != expected_rows
            or int(entry["confirmation_rows"]) != expected_rows
            or int(entry["discovery_rows"]) != expected_discovery):
        raise ValueError("Unexpected frozen PCA population size")
    mode, model = key.split("|")
    rows = []
    for index, row in enumerate(layer_data["rows"]):
        if len(row) != 5 or not np.isfinite(row).all():
            raise ValueError("Invalid archived PCA coordinate row")
        seed, label, x, y, z = row
        if not 1 <= int(label) <= 10:
            raise ValueError("PCA count label outside the registered support")
        rows.append({"format": mode, "model": model, "endpoint": endpoint,
                     "token_site": entry["token_site"], "layer_source_zero_based": layer,
                     "layer_display_one_based": layer + 1, "archived_row_index": index,
                     "seed": int(seed), "count_or_index": int(label),
                     "pc1": float(x), "pc2": float(y), "pc3": float(z)})
    if set(r["seed"] for r in rows) != set(range(1254, 1264)):
        raise ValueError("Archived confirmation seed panel changed")
    if endpoint == "final" and {(r["seed"], r["count_or_index"]) for r in rows} != {
            (seed, n) for seed in range(1254, 1264) for n in range(1, 11)}:
        raise ValueError("Final-count PCA lost a registered sample")
    meta = {"endpoint": endpoint, "cell": key, "token_site": entry["token_site"],
            "label": entry["label"], "layer_source_zero_based": layer,
            "layer_display_one_based": layer + 1, "discovery_fit_rows": expected_discovery,
            "confirmation_display_rows": len(rows), "confirmation_seeds": 10,
            "label_support": {str(n): sum(r["count_or_index"] == n for r in rows) for n in range(1, 11)},
            "discovery_explained_variance_ratio": layer_data["evr"],
            "discovery_axis_signs_already_applied": layer_data["axis_signs"],
            "archived_coordinates_sha256": hashlib.sha256(json.dumps(layer_data["rows"], separators=(",", ":")).encode()).hexdigest()}
    return rows, meta


def layout_check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
                  if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t, b in text_boxes if b.x0 < -.5 or b.y0 < -.5
               or b.x1 > fig.bbox.width + .5 or b.y1 > fig.bbox.height + .5]
    overlaps = []
    for i, (a, ab) in enumerate(text_boxes):
        for b, bb in text_boxes[i + 1:]:
            if min(ab.x1, bb.x1) - max(ab.x0, bb.x0) > .8 and min(ab.y1, bb.y1) - max(ab.y0, bb.y0) > .8:
                overlaps.append([a.get_text(), b.get_text()])
    label_point_overlaps = []
    for ax in fig.axes:
        if not hasattr(ax, "_pca_markers"):
            continue
        for label in ax.texts:
            if label.get_text() not in ["PC1", "PC2", "PC3"]:
                continue
            box = label.get_window_extent(renderer)
            for point, size in ax._pca_markers:
                x, y = ax.transData.transform(point)
                radius = math.sqrt(size) * fig.dpi / 144
                if box.x0 - radius < x < box.x1 + radius and box.y0 - radius < y < box.y1 + radius:
                    label_point_overlaps.append(label.get_text())
                    break
    return {"text_outside": outside, "text_overlaps": overlaps,
            "axis_label_marker_overlaps": label_point_overlaps}


def main():
    start = time.perf_counter()
    DATA.mkdir(exist_ok=True)
    PAPER.mkdir(exist_ok=True)
    read(Path(__file__).resolve())
    read(OUT / "appendix_style.py")
    renderer = reference_renderer()
    payload, selected, source_manifest, alignment = source_data()
    sizes = apply_style(canvas_width=6.5)
    figures, all_rows, all_panels = [], [], []
    # Reuse the reference's default orthographic camera; no per-cell visual selection.
    camera = {key: renderer.CAMERA[key] for key in ["yaw_deg", "pitch_deg", "camera_source"]}
    for mode in ["enumeration_index", "enumeration_bullet"]:
        fig = plt.figure(figsize=(6.5, 5.3), facecolor="white")
        fig.text(.25, .979, "Running index", ha="center", va="top", fontsize=sizes["title"])
        fig.text(.75, .979, "Final count", ha="center", va="top", fontsize=sizes["title"])
        panel_metadata = []
        for row_index, (model, short) in enumerate([("Qwen3-8B", "Qwen"), ("Gemma4-E4B", "Gemma")]):
            for column, endpoint in enumerate(["running", "final"]):
                key = mode + "|" + model
                rows, meta = panel_data(payload, selected, endpoint, key)
                all_rows.extend(rows)
                letter = "ABCD"[row_index * 2 + column]
                ax = fig.add_axes([[.035, .535][column], [.565, .125][row_index], .430, .300])
                display = renderer.draw_panel(ax, [(rows, "o", True, "-")],
                                              meta["discovery_explained_variance_ratio"],
                                              f"{letter}. {short} L{meta['layer_display_one_based']}", camera)
                meta.update({"panel": letter, "display": display})
                panel_metadata.append(meta)
                all_panels.append(meta)
        colors = ListedColormap([renderer.COUNT_COLORS[k] for k in range(1, 11)])
        boundaries = np.arange(.5, 11, 1)
        norm = BoundaryNorm(boundaries, colors.N)
        key_ax = fig.add_axes([.31, .04, .38, .017])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=colors), cax=key_ax,
                               orientation="horizontal", boundaries=boundaries,
                               ticks=[1, 10], spacing="uniform", drawedges=True)
        colorbar.dividers.set_color("white")
        colorbar.dividers.set_linewidth(.65)
        colorbar.outline.set_linewidth(.6)
        colorbar.outline.set_edgecolor(renderer.INK)
        colorbar.ax.tick_params(labelsize=sizes["tick"], length=2, pad=2, width=.6)
        fig.text(.5, .005, "Running index / final count", fontsize=sizes["annotation"], ha="center", va="bottom")
        renderer.place_axis_labels(fig)
        checks = layout_check(fig)
        if any(checks.values()):
            # Save a diagnostic image even if a layout check needs correction.
            fig.savefig(OUT / f"enumeration_pca_{mode.removeprefix('enumeration_')}_diagnostic.png", dpi=180)
            raise ValueError(f"PCA layout verification failed: {checks}")
        stem = "enumeration_pca_" + mode.removeprefix("enumeration_")
        files = {}
        for ext in ["pdf", "svg", "png"]:
            path = OUT / (stem + "." + ext)
            fig.savefig(path, dpi=300, metadata={"Title": f"{mode.removeprefix('enumeration_').title()} enumeration count PCA"} if ext == "pdf" else None)
            files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(OUT / (stem + ".pdf"), PAPER / (stem + ".pdf"))
        figures.append({"name": stem, "canvas_inches": [6.5, 5.3], "panels": panel_metadata,
                        "files_sha256": files, "checks": checks})
        plt.close(fig)
    csv_path = DATA / "enumeration_pca_coordinates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    metadata = {"status": "PASS_ARCHIVED_COORDINATES_RENDERED", "source_sha256": INPUT_HASHES,
                "coordinate_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "archived_payload_sha256": source_manifest["representation_manifold"]["payload_sha256"],
                "original_transform": payload["transform"], "selection_firewall": payload["selection_firewall"],
                "coordinate_fit": "Discovery-only StandardScaler + unwhitened PCA3; archived coordinates",
                "model_inference": False, "pca_refit": False, "layer_reselected": False,
                "points_subsampled": False, "axis_signs_modified": False,
                "display_indexing": "one-based; source layer indices remain zero-based",
                "analysis_population": alignment["analysis_population"],
                "individual_states_displayed": len(all_rows),
                "style": font_profile(), "reference_renderer": REFERENCE.relative_to(ROOT).as_posix(),
                "camera": camera, "count_colors": renderer.COUNT_COLORS,
                "centroid_colors": renderer.CENTROID_COLORS,
                "centroid_description": "Confirmation means per label connected in numeric order; descriptive only",
                "coordinates_precision": "Archived HTML coordinates rounded to six decimal places; no raw hidden state access required",
                "figures": figures, "elapsed_seconds": time.perf_counter() - start}
    (OUT / "pca_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"status": metadata["status"], "figures": [f["name"] for f in figures],
                      "individual_states": len(all_rows), "elapsed_seconds": metadata["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
