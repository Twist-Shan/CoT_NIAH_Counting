"""Reproduce supporting enumeration plots from archived estimates, without refits.

All intervals and effects are copied from the two embedded report payloads.
The downstream count-margin snapshot preserves rounded scalar strings; these
are plotted at their stored precision, with no reconstructed uncertainty.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from appendix_style import apply_style, font_profile, MODELS, COLORS, INK, GRID

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / "realistic"
PAPER = ROOT / "runs/paper_figures/figures"
DATA = OUT / "data"
SOURCES, ROWS, CHECKS = {}, [], {}
MODES = ["enumeration_index", "enumeration_bullet"]
MODE_NAMES = ["Index", "Bullet"]
SIZE = apply_style(6.5)


def read(path):
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8")


REPORT = REPO / "reports/NiaH_Enumeration_report.html"
match = re.search(r'<script id="report-manifest" type="application/json">(.*?)</script>', read(REPORT), re.S)
assert match, "Missing sealed report-manifest"
PAYLOAD = json.loads(match.group(1))
assert PAYLOAD["status"] == "PASS_EVIDENCE_COLLECTED"
SUMMARY_PATH = REPO / "work_remote_snapshots/v6_enumeration_final_20frame/NiaH_V6_Index_Bullet_Replication_report.summary.json"
SUMMARY = json.loads(read(SUMMARY_PATH))
read(OUT / "appendix_style.py")
read(ROOT / font_profile()["reference"])


def effect(figure, panel, mode, model, estimand, record, pointer, **metadata):
    mean, low, high = (float(record[k]) for k in ["mean_effect", "ci_low", "ci_high"])
    assert np.isfinite([mean, low, high]).all()
    assert low <= mean + 1e-12 <= high + 1e-12, (estimand, record)
    seeds = int(record.get("n_seeds", record.get("seed_count", metadata.get("seed_count", 0))))
    assert seeds > 0
    row = dict(figure=figure, panel=panel, mode=mode, model=model,
               estimand=estimand, mean=mean, ci95_low=low, ci95_high=high,
               seed_count=seeds, source_pointer=pointer, **metadata)
    ROWS.append(row)
    return row


def find(records, name, field="estimand"):
    matches = [(i, r) for i, r in enumerate(records) if r[field] == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def point(ax, x, row, color, marker="o", fill=True):
    ax.errorbar(x, row["mean"],
                yerr=[[max(0, row["mean"]-row["ci95_low"])],
                      [max(0, row["ci95_high"]-row["mean"])]],
                color=color, marker=marker, markersize=4.4,
                markerfacecolor=color if fill else "white", markeredgewidth=1,
                linestyle="none", capsize=2, elinewidth=.9, capthick=.9,
                zorder=5)


def canvas(titles, notes=None):
    fig = plt.figure(figsize=(6.5, 5.55), facecolor="white")
    rects = [[.115, .59, .335, .245], [.615, .59, .335, .245],
             [.115, .16, .335, .245], [.615, .16, .335, .245]]
    axes = [fig.add_axes(rect) for rect in rects]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=.65)
        ax.axhline(0, color="#A9AFB9", linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.7, pad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        left = .025 if i % 2 == 0 else .525
        top = .935 if i < 2 else .505
        fig.text(left, top, "ABCD"[i] + ".", fontsize=SIZE["letter"], va="center")
        fig.text(left+.037, top, title, fontsize=SIZE["title"], va="center")
        if notes:
            fig.text(left+.037, top-.043, notes[i], fontsize=SIZE["annotation"], va="center")
    return fig, axes


def models_legend(fig, extra=None, y=.055):
    handles = [Line2D([], [], color=c, marker="o", linestyle="none", label=m, markersize=4.5)
               for m, c in zip(["Qwen", "Gemma"], COLORS)]
    handles += extra or []
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, y), ncol=len(handles),
               handlelength=1.3, columnspacing=1.15, handletextpad=.4)


def save(fig, name):
    for panel, ax in zip("ABCDEF", fig.axes):
        low, high = ax.get_ylim()
        rows = [r for r in ROWS if r["figure"] == name and r["panel"] == panel]
        assert rows
        assert all(low <= r["ci95_low"] <= r["ci95_high"] <= high for r in rows), (name, panel, "Effect or interval clipped")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    clipped = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0-1 or box.y0 < bounds.y0-1 or box.x1 > bounds.x1+1 or box.y1 > bounds.y1+1:
            clipped.append(text.get_text())
    assert not clipped, (name, clipped)
    CHECKS[name] = {"canvas_inches": list(fig.get_size_inches()),
                    "visible_text_outside_canvas": clipped,
                    "paper_font_profile": font_profile()}
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=220, facecolor="white")
    shutil.copy2(OUT / f"{name}.pdf", PAPER / f"{name}.pdf")
    assert (OUT / f"{name}.pdf").read_bytes() == (PAPER / f"{name}.pdf").read_bytes()
    plt.close(fig)


def endpoint_controls():
    name = "enumeration_endpoint_controls"
    fig = plt.figure(figsize=(6.5, 6.0), facecolor="white")
    axes = []
    for row_index, label in enumerate(["successor attention", "city log-odds", "greedy city transfer"]):
        bottom = .735 - row_index*.295
        for mode_index, grammar in enumerate(MODE_NAMES):
            ax = fig.add_axes([.115+mode_index*.5, bottom, .335, .175])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=GRID, linewidth=.65)
            ax.axhline(0, color="#A9AFB9", linewidth=.65)
            ax.set_axisbelow(True)
            ax.tick_params(length=3, width=.7, pad=3)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            left = .025 + mode_index*.5
            fig.text(left, bottom+.222, "ABCDEF"[len(axes)]+".", fontsize=SIZE["letter"], va="center")
            fig.text(left+.037, bottom+.222, f"{grammar}: {label}", fontsize=SIZE["title"], va="center")
            axes.append(ax)
    for outcome_index, outcome in enumerate(["targeted_attention", "city_log_odds", "greedy_city_adoption"]):
        for mode_index, mode in enumerate(MODES):
            p = 2*outcome_index + mode_index
            ax = axes[p]
            for m, (model, color) in enumerate(zip(MODELS, COLORS)):
                key = f"{mode}|{model}"
                records = PAYLOAD["cells"][key]["full_commit_to_query"]["confirmation"]["estimands"]
                for c, control in enumerate(["self", "orthogonal"]):
                    xs, ys = [], []
                    for distance in [1, 2, 3]:
                        estimand = f"full_commit_{outcome}_vs_{control}_distance_{distance}"
                        index, source = find(records, estimand)
                        row = effect(name, "ABCDEF"[p], mode, model, estimand, source,
                                     f"report-manifest/cells/{key}/full_commit_to_query/confirmation/estimands/{index}",
                                     distance=distance, pair_count=int(source["pair_count"]),
                                     control=control, analysis_status="saved-arm reanalysis")
                        assert row["seed_count"] == 10 and row["pair_count"] == 20
                        x = distance + (m-.5)*.11 + (c-.5)*.045
                        point(ax, x, row, color, "o" if c == 0 else "s", c == 0)
                        xs.append(x)
                        ys.append(row["mean"])
                    ax.plot(xs, ys, color=color, linestyle="-" if c == 0 else "--", linewidth=.9, alpha=.8)
            ax.set_xlim(.7, 3.3)
            ax.set_xticks([1, 2, 3])
            if outcome_index == 2:
                ax.set_xlabel("Absolute progress offset")
            ax.set_ylabel(["Summed attention\neffect", "City log-odds effect", "Donor-city\nadoption effect"][outcome_index])
            if outcome_index == 2:
                ax.set_ylim(-.04, .36)
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            elif outcome_index == 1:
                ax.set_ylim(-.5, 20)
            else:
                # Keep the archived sum over downstream selected heads. Bank
                # sizes and active-head counts differ: no K normalization.
                ax.set_ylim((-.4, 9) if mode_index == 0 else (-1.4, 30))
    models_legend(fig, [Line2D([], [], color=INK, marker="o", label="vs self", linewidth=.9),
                        Line2D([], [], color=INK, marker="s", markerfacecolor="white", linestyle="--", label="vs orthogonal", linewidth=.9)], y=.045)
    save(fig, name)


def source_record(key, contrast):
    sources = SUMMARY["snapshots"]["format_specific_source_scrub_restore"][key]["sources"]
    assert len(sources) == 1 and sources[0]["kind"] == "csv"
    source = sources[0]
    rows = source["summary"]["preview"]
    assert len(rows) == source["summary"]["row_count"], "Refuse a truncated CSV preview"
    index, result = find(rows, contrast, "contrast")
    return result, f"snapshots/format_specific_source_scrub_restore/{key}/sources/0/summary/preview/{index}", source


def source_restoration():
    name = "enumeration_source_restoration"
    fig, axes = canvas(["Index: source restoration", "Bullet: source restoration",
                        "Full span versus endpoint", "Ordinary-position control"],
                       ["Qwen L20; Gemma L17"]*2 + ["Paired restoration difference", "Full span minus ordinary repair"])
    for mode_index, mode in enumerate(MODES):
        ax = axes[mode_index]
        contrasts = ["full_span_repair", "endpoint_repair"] + (["marker_repair"] if mode_index == 0 else [])
        for m, (model, color) in enumerate(zip(MODELS, COLORS)):
            key = f"{mode}|{model}"
            for x, contrast in enumerate(contrasts):
                result, pointer, source = source_record(key, contrast)
                row = effect(name, "AB"[mode_index], mode, model, contrast, result, pointer,
                             layer_one_based=int(result["layer"])+1, original_source=source["path"],
                             original_sha256=source["sha256"])
                point(ax, x+(m-.5)*.14, row, color)
            for p, contrast in [(2, "full_span_vs_endpoint_restore"), (3, "full_span_vs_ordinary_repair_specificity")]:
                result, pointer, source = source_record(key, contrast)
                row = effect(name, "ABCD"[p], mode, model, contrast, result, pointer,
                             layer_one_based=int(result["layer"])+1, original_source=source["path"],
                             original_sha256=source["sha256"])
                point(axes[p], mode_index+(m-.5)*.14, row, color)
        ax.set_xticks(range(len(contrasts)), ["Full span", "Endpoint"] + (["Marker"] if mode_index == 0 else []))
        ax.set_xlim(-.4, len(contrasts)-.6)
        ax.set_ylabel("Count-margin improvement")
        ax.set_ylim(-1.5, 23)
    for ax in axes[2:]:
        ax.set_xticks([0, 1], MODE_NAMES)
        ax.set_xlim(-.4, 1.4)
        ax.set_ylabel("Count-margin difference")
        ax.set_ylim(-1.5, 23)
    models_legend(fig)
    save(fig, name)


def margin_controls():
    name = "enumeration_margin_controls"
    fig, axes = canvas(["Index: count geometry", "Bullet: count geometry",
                        "Index: final-count margin", "Bullet: final-count margin"],
                       ["Nearest-centroid margin", "Nearest-centroid margin",
                        "Teacher-forced sequence scores", "Teacher-forced sequence scores"])
    for mode_index, mode in enumerate(MODES):
        for m, (model, color) in enumerate(zip(MODELS, COLORS)):
            key = f"{mode}|{model}"
            ncc = PAYLOAD["cells"][key]["ncc"]
            for x, estimand in enumerate(["selected_correct_centroid_margin_loss", "random_mean_correct_centroid_margin_loss", "selected_vs_random_margin_loss_specificity"]):
                index, result = find(ncc["all_estimands"], estimand)
                row = effect(name, "AB"[mode_index], mode, model, estimand, result,
                             f"report-manifest/cells/{key}/ncc/all_estimands/{index}",
                             layer_one_based=int(ncc["selected_layer"])+1)
                assert row["seed_count"] == 10
                point(axes[mode_index], x+(m-.5)*.14, row, color)
            sources = SUMMARY["snapshots"]["direct_count_output_margin"][key]["sources"]
            assert len(sources) == 1
            source = sources[0]
            scalars = dict(source["interesting_scalars"])
            prefix = "endpoint_results.final_answer_sequence_margin.confirmation."
            seeds = int(scalars[prefix + "seed_count"])
            for x, estimand in enumerate(["selected_margin_loss", "selected_vs_random_specificity"]):
                result = {field: scalars[prefix+estimand+"."+field] for field in ["mean_effect", "ci_low", "ci_high"]}
                result["seed_count"] = seeds
                row = effect(name, "CD"[mode_index], mode, model, estimand, result,
                             f"snapshots/direct_count_output_margin/{key}/sources/0/interesting_scalars/{prefix}{estimand}",
                             stored_precision="rounded archive scalar strings; typically five decimals",
                             original_source=source["path"], original_sha256=source["sha256"])
                assert row["seed_count"] == 10
                point(axes[mode_index+2], x+(m-.5)*.14, row, color)
        axes[mode_index].set_xticks([0, 1, 2], ["Selected", "Random", "Selected\n− random"])
        axes[mode_index].set_xlim(-.4, 2.4)
        axes[mode_index].set_ylim(-440, 330)
        axes[mode_index].set_ylabel("Centroid-margin loss")
        axes[mode_index+2].set_xticks([0, 1], ["Selected", "Selected − random"])
        axes[mode_index+2].set_xlim(-.4, 1.4)
        axes[mode_index+2].set_ylim(-1.05, 2.8)
        axes[mode_index+2].set_ylabel("Count-sequence margin loss")
    models_legend(fig)
    save(fig, name)


def terminal_controls():
    name = "enumeration_terminal_controls"
    estimands = ["terminal_token_necessity", "terminal_token_sufficiency",
                 "token_written_state_sufficiency", "token_effect_requires_terminal_state"]
    fig, axes = canvas(["Terminal-token necessity", "Terminal-token restoration",
                        "Written-state restoration", "Terminal-state occlusion"],
                       ["Damage in the intact trace", "All item spans replaced",
                        "All item spans replaced", "Restored minus occluded state"])
    for p, estimand in enumerate(estimands):
        ax = axes[p]
        for mode_index, mode in enumerate(MODES):
            for m, (model, color) in enumerate(zip(MODELS, COLORS)):
                key = f"{mode}|{model}"
                records = PAYLOAD["cells"][key]["terminal"]["baseline"]["all_estimands"]
                matches = [(i,r) for i,r in enumerate(records) if r["estimand"] == estimand and r["outcome"] == "expected_count_utility"]
                assert len(matches) == 1
                index, result = matches[0]
                row = effect(name, "ABCD"[p], mode, model, estimand, result,
                             f"report-manifest/cells/{key}/terminal/baseline/all_estimands/{index}",
                             outcome="expected_count_utility", coefficients=result["coefficients"])
                assert row["seed_count"] == 10
                point(ax, mode_index+(m-.5)*.14, row, color)
        ax.set_xticks([0, 1], MODE_NAMES)
        ax.set_xlim(-.4, 1.4)
        ax.set_ylabel("Expected-count utility effect")
        ax.set_ylim((-.15, 1.8) if p == 0 else (-1.4, 3.2))
    models_legend(fig)
    save(fig, name)


def main():
    DATA.mkdir(exist_ok=True)
    PAPER.mkdir(exist_ok=True)
    endpoint_controls()
    source_restoration()
    margin_controls()
    terminal_controls()
    assert len(ROWS) == 72 + 18 + 20 + 16, len(ROWS)
    (DATA / "supporting_plotted_estimates.json").write_text(json.dumps(ROWS, indent=2)+"\n", encoding="utf-8")
    fields = sorted({key for row in ROWS for key in row})
    with (DATA / "supporting_plotted_estimates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ROWS:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k,v in row.items()})
    manifest = {"status": "ARCHIVED_ESTIMATES_REPRODUCED", "source_sha256": SOURCES,
                "row_count": len(ROWS), "plot_checks": CHECKS,
                "uncertainty": "Copied pointwise 95% seed-bootstrap intervals; no refit or bootstrap.",
                "layer_convention": "One-based paper labels converted from zero-based source layers.",
                "precision_qualification": "Only final-count margin uses rounded strings from the archive's interesting_scalars; other effects retain embedded numeric precision.",
                "endpoint_qualification": "Historical saved-arm contrasts. Gemma sensitivity to corrected temporary attention backend has not been established by paired replay.",
                "coverage": "See supporting_coverage.md for counterparts and unavailable local raw sources."}
    (OUT / "supporting_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"figures": list(CHECKS), "estimates": len(ROWS), "font_profile": font_profile()}, indent=2))


if __name__ == "__main__":
    main()
