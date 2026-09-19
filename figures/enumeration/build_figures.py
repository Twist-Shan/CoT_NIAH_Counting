"""Publication enumeration figures, copied from frozen estimates without inference.

Run from any directory with Python, NumPy, and Matplotlib. All uncertainty is
copied from the registered estimate, except the explicitly documented monotone
transformation of the absolute residual ratio in the terminal-relay panel.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
from appendix_style import apply_style, font_profile, COLORS, INK, AXIS, GRID

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / "realistic"
DATA = OUT / "data"
PAPER = ROOT / "runs/paper_figures/figures"
DATA.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)
SOURCES = {}
ROWS = []
CHECKS = {}
MODELS = ["Qwen3-8B", "Gemma4-E4B"]
MODES = ["enumeration_index", "enumeration_bullet"]
CANVAS_WIDTH, CANVAS_HEIGHT = 11.7, 8.37


def read(path):
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8")


REPORT = REPO / "reports/NiaH_Enumeration_report.html"
match = re.search(r'<script id="report-manifest" type="application/json">(.*?)</script>', read(REPORT), re.S)
assert match, "Sealed report-manifest payload missing"
PAYLOAD = json.loads(match.group(1))
assert PAYLOAD["status"] == "PASS_EVIDENCE_COLLECTED"
FONT = font_profile()
read(OUT / "appendix_style.py")
read(ROOT / FONT["reference"])
SUMMARY_PATH = REPO / "work_remote_snapshots/v6_enumeration_final_20frame/NiaH_V6_Index_Bullet_Replication_report.summary.json"
SUMMARY = json.loads(read(SUMMARY_PATH))
read(ROOT / "figures/cot-reasoning/prepare_data.py")
read(ROOT / "figures/cot-reasoning/build_figure.py")
read(ROOT / "figures/non-thinking/build_nonthinking_section_v8.py")
SCALE = CANVAS_WIDTH / FONT["paper_width_inches"]
SIZE = apply_style(CANVAS_WIDTH)


def record(figure, panel, mode, model, condition, mean, low=None, high=None, **extra):
    mean = float(mean)
    assert np.isfinite(mean)
    if low is not None:
        low, high = float(low), float(high)
        assert np.isfinite([low, high]).all() and low <= mean + 1e-12 <= high + 1e-12
    row = dict(figure=figure, panel=panel, mode=mode, model=model,
               condition=condition, mean=mean, ci95_low=low, ci95_high=high, **extra)
    ROWS.append(row)
    return row


def point(ax, x, row, color, marker="o", fill=True, size=6):
    kw = dict(color=color, marker=marker, markersize=size, markerfacecolor=color if fill else "white",
              markeredgewidth=1.15, linestyle="none", zorder=4)
    if row["ci95_low"] is None:
        ax.plot(x, row["mean"], **kw)
    else:
        ax.errorbar(x, row["mean"], yerr=[[max(0, row["mean"] - row["ci95_low"])],
                                         [max(0, row["ci95_high"] - row["mean"])]],
                    capsize=3, elinewidth=1.1, capthick=1.1, **kw)


def canvas(titles):
    fig = plt.figure(figsize=(CANVAS_WIDTH, CANVAS_HEIGHT), facecolor="white")
    rectangles = [[.09, .595, .365, .25], [.595, .595, .365, .25],
                  [.09, .15, .365, .25], [.595, .15, .365, .25]]
    axes = [fig.add_axes(r) for r in rectangles]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=.6 * SCALE)
        ax.set_axisbelow(True)
        ax.tick_params(length=3 * SCALE, width=.7 * SCALE, pad=3 * SCALE)
    for i, title in enumerate(titles):
        x, y = (.02 if i % 2 == 0 else .525), (.916 if i < 2 else .47)
        fig.text(x, y, "ABCD"[i] + ".", fontsize=SIZE["letter"], weight="normal", va="center")
        fig.text(x + .037, y, title, fontsize=SIZE["title"], weight="normal", va="center")
    fig.legend([Line2D([], [], color=c, marker="o", ms=5, lw=1.8) for c in COLORS],
               ["Qwen", "Gemma"], ncol=2, loc="center", bbox_to_anchor=(.5, .976),
               handlelength=1.3, columnspacing=1.8)
    return fig, axes


def percent(ax, difference=False):
    if difference:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}"))
    else:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def grammar_x(ax):
    ax.set(xlim=(-.52, 1.52), xticks=[0, 1], xticklabels=["Index", "Bullet"])


def legend_shapes(fig, labels, markers, anchor, filled=None, ncol=2):
    filled = filled or [True] * len(labels)
    fig.legend([Line2D([], [], color=INK, marker=m, ls="", ms=5.5,
                      mfc=INK if f else "white") for m, f in zip(markers, filled)],
               labels, loc="center", bbox_to_anchor=anchor, ncol=ncol,
               handlelength=.7, handletextpad=.45, columnspacing=1.0,
               fontsize=SIZE["annotation"])


def save(fig, stem, title):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [(t.get_text(), t.get_window_extent(renderer))
              for t in fig.findobj(matplotlib.text.Text)
              if t.get_visible() and t.get_text()]
    outside = [s for s, b in labels if b.x0 < -1 or b.y0 < -1 or
               b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1]
    overlaps = []
    for i, (s, a) in enumerate(labels):
        for t, b in labels[i + 1:]:
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1.5:
                overlaps.append([s, t])
    assert not outside, (stem, "Outside text", outside)
    assert not overlaps, (stem, "Text collision", overlaps)
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"Title": title, "Author": "Anonymous Authors", "CreationDate": None})
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    shutil.copyfile(pdf, PAPER / pdf.name)
    CHECKS[stem] = {"outside_text": outside, "text_overlaps": overlaps,
                    "canvas_inches": list(fig.get_size_inches()),
                    "paper_width_inches": 6.5, "paper_height_inches": CANVAS_HEIGHT * 6.5 / CANVAS_WIDTH,
                    "smallest_font_at_paper_width_pt": SIZE["annotation"] / SCALE,
                    "vector_pdf": str(pdf.relative_to(ROOT).as_posix()),
                    "png_pixels": [int(CANVAS_WIDTH * 220), int(CANVAS_HEIGHT * 220)],
                    "manuscript_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    plt.close(fig)


# Figure 1: the original confirmation floor is retained beside the subsequent
# prospectively frozen discovery sensitivity; it is not relabeled confirmation.
stem = "enumeration_retrieval_write"
fig, axes = canvas(["Targeted retrieval", "Index anchor sensitivity",
                    "Carrier deformation", "Carrier restoration"])
for j, mode in enumerate(MODES):
    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        cell = PAYLOAD["cells"][f"{mode}|{model}"]
        value = cell["targeted_retrieval"]["binary"]
        r = record(stem, "A", mode, model, "selected_minus_registered_random",
                   value["seed_equal_selected_minus_random_failure"],
                   value["seed_bootstrap_95_lo"], value["seed_bootstrap_95_hi"],
                   seed_count=value["seed_count"], k=cell["selected_k"],
                   random_condition=value["random_condition"],
                   source_pointer=f"/cells/{mode}|{model}/targeted_retrieval/binary")
        x = j + [-.15, .15][i]
        point(axes[0], x, r, color)
        axes[0].text(x, -.28, f"K={cell['selected_k']}", ha="center", va="top", color=color,
                     fontsize=SIZE["annotation"], transform=axes[0].get_xaxis_transform())
        for panel_index, metric in [(2, "selected_carrier_deformation"), (3, "clean_carrier_restoration")]:
            effect = next(v for v in cell["carrier"]["baseline"]["primary_estimands"] if v["estimand"] == metric)
            r = record(stem, "ABCD"[panel_index], mode, model, "query_local",
                       effect["mean_effect"], effect["ci_low"], effect["ci_high"],
                       seed_count=effect["n_seeds"], metric=metric,
                       source_pointer=f"/cells/{mode}|{model}/carrier/baseline/primary_estimands/{metric}")
            point(axes[panel_index], x, r, color)
            if mode == "enumeration_bullet" and model == "Gemma4-E4B":
                fresh = cell["carrier"]["fresh_query_through_carrier_replication"]["analysis"]
                effect = next(v for v in fresh["primary_estimands"] if v["estimand"] == metric)
                r = record(stem, "ABCD"[panel_index], mode, model, "query_through_carrier_fresh_seeds",
                           effect["mean_effect"], effect["ci_low"], effect["ci_high"],
                           seed_count=effect["n_seeds"], metric=metric,
                           source_pointer=f"/cells/{mode}|{model}/carrier/fresh_query_through_carrier_replication/analysis/primary_estimands/{metric}")
                point(axes[panel_index], 1.32, r, color, marker="D", fill=False)
for i, (model, color) in enumerate(zip(MODELS, COLORS)):
    analysis = PAYLOAD["index_item_end_anchor_sensitivity"]["analyses"][model]
    assert analysis["decision_is_exploratory"] and not analysis["may_replace_primary_result"]
    for bank, marker, offset in [("p2", "o", -.06), ("p0", "s", .06)]:
        for j, site in enumerate(["p0", "p2"]):
            key = f"{bank}bank_at_{site}"
            value = analysis["cells"][key]["selected_minus_random_failure"]
            r = record(stem, "B", "enumeration_index", model, key, value["estimate"],
                       *value["ci95"], analysis_slot_seeds=value["n_analysis_slot_seeds"],
                       evidence="exploratory_discovery_sensitivity", k=analysis["fixed_k"],
                       source_pointer=f"/index_item_end_anchor_sensitivity/analyses/{model}/cells/{key}/selected_minus_random_failure")
            point(axes[1], j + [-.16, .16][i] + offset, r, color, marker, fill=bank == "p2")
for ax in axes[:2]:
    ax.set(ylim=(-.05, 1.10), yticks=[0, .5, 1], ylabel="Failure difference (pp)")
    percent(ax, difference=True)
grammar_x(axes[0])
axes[1].set(xlim=(-.48, 1.48), xticks=[0, 1], xticklabels=["Item end", "Post-marker"])
axes[1].set_xlabel("Ablation starts at")
legend_shapes(fig, ["Post-marker bank", "Item-end bank"], ["o", "s"], (.777, .871), [True, False])
for ax in axes[2:]:
    grammar_x(ax)
    ax.axhline(0, color=AXIS, lw=.8)
axes[2].set(ylim=(-.035, .75), yticks=[0, .2, .4, .6], ylabel="Carrier RMS change")
axes[3].set(ylim=(-.05, 1.12), yticks=[0, .25, .5, .75, 1], ylabel="Commit RMS reduction")
legend_shapes(fig, ["Query only", "Query through carrier (fresh seeds)"], ["o", "D"], (.5, .042), [True, False])
save(fig, stem, "Enumeration: retrieval support and carrier-state restoration")


# Figure 2: success means the entire target path prefix is correct at the
# stated depth. The denominator always retains all registered paired trials.
stem = "enumeration_continuation"
fig, axes = canvas(["Index continuation", "Bullet continuation",
                    "Index direction dependence", "Bullet direction dependence"])
depths = [1, 2, 4]
multihop = PAYLOAD["followup_v3"]["full_item_multihop"]
for j, mode in enumerate(MODES):
    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        cell = next(c for c in multihop["cell_summaries"] if c["prompt_mode"] == mode and c["model_label"] == model)
        values = []
        for depth in depths:
            value = cell[f"depth_{depth}"]
            assert value["receiver_self_donor_rate"] == 0
            effect = value["paired_effect"]
            assert abs(effect["mean_effect"] - value["patched_rate"]) < 1e-12
            r = record(stem, "AB"[j], mode, model, "patched_target_prefix", value["patched_rate"],
                       effect["ci_low"], effect["ci_high"], depth=depth, seed_count=effect["n_seeds"],
                       directed_pairs=20, native_target_rate=value["native_donor_rate"], self_target_rate=0,
                       source_pointer=f"/followup_v3/full_item_multihop/cell_summaries/{mode}|{model}/depth_{depth}")
            values.append(r["mean"])
            point(axes[j], depth + [-.04, .04][i], r, color)
        axes[j].plot(np.asarray(depths) + [-.04, .04][i], values, color=color, lw=1.8)
        # Native target continuation is a reference condition, shown without
        # fabricated uncertainty because only the frozen rates are available.
        axes[j].plot(depths, [cell[f"depth_{d}"]["native_donor_rate"] for d in depths],
                     color=color, ls=(0, (3, 2)), lw=1.1, alpha=.65)
        for direction_index, direction in enumerate(["forward_skip", "backward_rewind"]):
            dcell = next(c for c in multihop["direction_summaries"] if c["prompt_mode"] == mode and
                         c["model_label"] == model and c["direction"] == direction)
            value = dcell["depth_4"]
            effect = value["paired_effect"]
            assert value["receiver_self_donor_rate"] == 0
            r = record(stem, "CD"[j], mode, model, direction, value["patched_rate"],
                       effect["ci_low"], effect["ci_high"], depth=4, seed_count=effect["n_seeds"],
                       directed_pairs=10, source_pointer=f"/followup_v3/full_item_multihop/direction_summaries/{mode}|{model}|{direction}/depth_4")
            point(axes[j + 2], direction_index + [-.12, .12][i], r, color)
    axes[j].set(xticks=depths, xlim=(.6, 4.4), xlabel="Continuation depth (items)", ylabel="Donor-path success")
    axes[j + 2].set(xticks=[0, 1], xticklabels=["Forward", "Backward"], xlim=(-.5, 1.5),
                    ylabel="Four-item path success")
for ax in axes:
    ax.set(ylim=(-.05, 1.1), yticks=[0, .5, 1])
    ax.axhline(0, color=AXIS, lw=.8)
    percent(ax)
fig.legend([Line2D([], [], color=INK, lw=1.8), Line2D([], [], color=INK, lw=1.1, ls=(0, (3, 2)))],
           ["Patched receiver", "Unmodified donor"], loc="center", bbox_to_anchor=(.5, .042),
           ncol=2, fontsize=SIZE["annotation"])
save(fig, stem, "Enumeration: cumulative target continuation and directional asymmetry")


# Figure 3: eight frozen sampled layers only, displayed one-based. The relay
# fraction is an absolute-damage reduction; it is NOT interaction / total.
stem = "enumeration_readout"
fig, axes = canvas(["Index answer-state patching", "Bullet answer-state patching",
                    "Answer-source blanking", "Terminal-to-suffix relay"])
for j, mode in enumerate(MODES):
    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        cell = next(c for c in PAYLOAD["answer_trace_extension"]["cells"] if c["prompt_mode"] == mode and c["model_label"] == model)
        x, y = [], []
        for value in cell["answer_layer_effects"]:
            layer = value["layer"] + 1
            r = record(stem, "AB"[j], mode, model, "full_target_answer_query", value["full_donor_adoption"],
                       value["full_donor_adoption_ci95_low"], value["full_donor_adoption_ci95_high"],
                       layer=layer, source_layer_zero_based=value["layer"], directed_pairs=value["pairs"],
                       seed_count=value["seed_clusters"], numeric_valid=value["registered_numeric_valid"],
                       source_pointer=f"/answer_trace_extension/cells/{mode}|{model}/answer_layer_effects/layer={value['layer']}")
            point(axes[j], layer, r, color, size=4.4)
            x.append(layer)
            y.append(r["mean"])
        assert len(x) == 8 and max(x) == [36, 42][i]
        axes[j].plot(x, y, color=color, lw=1.8)
        source = next(s for s in SUMMARY["snapshots"]["answer_token_source_ablation"][f"{mode}|{model}"]["sources"]
                      if "token_ablation_answer" in s.get("path", ""))
        for condition, marker, offset, fill in [("prompt_records_blank", "o", -.04, False),
                                                ("trace_all_blank", "D", .04, True)]:
            value = next(r for r in source["summary"]["preview"] if r["condition"] == condition)
            r = record(stem, "C", mode, model, condition, float(value["mean_delta_exact_count"]),
                       seed_count=int(value["seed_count"]), uncertainty="not supplied in summary; not inferred",
                       source_file=SUMMARY_PATH.relative_to(ROOT).as_posix(), source_pointer=source["path"])
            point(axes[2], j + [-.15, .15][i] + offset, r, color, marker=marker, fill=fill)
        ratio = cell["suffix_residual_ratio"]
        r = record(stem, "D", mode, model, "one_minus_absolute_residual_ratio", 1 - ratio["estimate"],
                   1 - ratio["high"], 1 - ratio["low"], geometry=cell["relay_geometry"],
                   seed_count=cell["relay_eligible_seed_count"], registered_seed_count=cell["relay_planned_seed_count"],
                   terminal_damage=cell["terminal_patch"]["estimate"],
                   suffix_specific_mediation=cell["suffix_mediation"]["estimate"],
                   absolute_residual_ratio=ratio["estimate"],
                   source_pointer=f"/answer_trace_extension/cells/{mode}|{model}/suffix_residual_ratio")
        natural = cell["terminal_patch"]["estimate"]
        residual = natural - cell["suffix_mediation"]["estimate"]
        assert abs(abs(residual / natural) - ratio["estimate"]) < 1e-7
        point(axes[3], j + [-.15, .15][i], r, color)
        axes[3].text(j + [-.20, .20][i], -.26, f"{cell['relay_eligible_seed_count']}/10", ha="center", va="top",
                     fontsize=SIZE["annotation"], color=color, transform=axes[3].get_xaxis_transform())
    axes[j].set(xlim=(-.5, 43.5), xticks=[1, 10, 20, 30, 42], xlabel="Layer", ylabel="Donor-count adoption",
                ylim=(-.05, 1.08), yticks=[0, .5, 1])
    percent(axes[j])
grammar_x(axes[2])
axes[2].set(ylim=(-1.12, .12), yticks=[-1, -.5, 0], ylabel="Accuracy change (pp)")
percent(axes[2], difference=True)
axes[2].axhline(0, color=AXIS, lw=.8)
legend_shapes(fig, ["Prompt records", "Full trace"], ["o", "D"], (.27, .432), [False, True])
axes[3].set(xticks=[0, 1], xticklabels=["Index (suffix 8)", "Bullet (suffix 4)"], xlim=(-.55, 1.55),
            ylim=(-.05, 1.1), yticks=[0, .5, 1], ylabel="Absolute-damage reduction")
percent(axes[3])
fig.text(.5, .032, "Bullet suffix 4: task-adapted replication; Index suffix 8: original registered geometry.",
         ha="center", fontsize=SIZE["annotation"])
save(fig, stem, "Enumeration: answer-state execution, source dependence, and partial terminal relay")


# Figure 4: archived held-out probes on the exactly aligned original panel.
# Marker layers are read from the frozen selected tables, never reselected here.
stem = "enumeration_representations"
representation_dir = REPO / "work/v6_report_remote/native_aligned_representation"
representation_manifest = json.loads(read(representation_dir / "analysis_manifest.json"))
assert representation_manifest["confirmation_used_for_selection"] is False
assert representation_manifest["evaluation_split"] == "confirmation"
representation_audit = json.loads(read(representation_dir / "alignment_audit.json"))
assert representation_audit["replacement_rows_allowed"] is False
fig, axes = canvas(["Index running-count state", "Bullet running-count state",
                    "Index final-count state", "Bullet final-count state"])
for row_index, (endpoint, prefix, manifest_prefix, expected_rows) in enumerate([
        ("running_index", "running_index", "running", 518),
        ("final_count", "final_count", "final", 100)]):
    paths = [representation_dir / f"{prefix}_candidate_metrics.csv",
             representation_dir / f"{prefix}_selected.csv"]
    candidates, selected_rows = [list(csv.DictReader(read(p).splitlines())) for p in paths]
    for p, key in zip(paths, [f"{manifest_prefix}_candidates", f"{manifest_prefix}_selected"]):
        assert SOURCES[p.relative_to(ROOT).as_posix()] == representation_manifest["outputs"][key]["sha256"]
    for column_index, mode in enumerate(MODES):
        panel_index = 2 * row_index + column_index
        ax = axes[panel_index]
        for model_index, (model, color) in enumerate(zip(MODELS, COLORS)):
            cell_rows = sorted([r for r in candidates if r["prompt_mode"] == mode and r["model_label"] == model],
                               key=lambda r: int(r["layer"]))
            selected = next(r for r in selected_rows if r["prompt_mode"] == mode and r["model_label"] == model)
            selected_layer = int(selected["layer"]) + 1
            assert [int(r["layer"]) + 1 for r in cell_rows] == list(range(1, [36, 42][model_index] + 1))
            for metric, linestyle in [("logistic", "-"), ("ncc", (0, (3.5, 2.2)))]:
                metric_key = f"confirmation_{metric}_balanced_accuracy"
                x, y = [], []
                for value in cell_rows:
                    assert int(value["confirmation_rows"]) == expected_rows
                    assert value["exact_four_cell_sample_alignment"] == "True"
                    assert int(value["retained_class_count"]) == 10
                    assert float(value["chance_balanced_accuracy"]) == .1
                    layer = int(value["layer"]) + 1
                    r = record(stem, "ABCD"[panel_index], mode, model, metric,
                               float(value[metric_key]), layer=layer, source_layer_zero_based=int(value["layer"]),
                               endpoint=endpoint, token_site=value["token_site"],
                               seed_count=int(value["confirmation_seed_count"]), confirmation_rows=expected_rows,
                               discovery_rows=int(value["discovery_oof_rows"]),
                               selected_layer=selected_layer, selected_layer_marker=layer == selected_layer and metric == "logistic",
                               uncertainty="not supplied for layer curves; not inferred",
                               source_file=paths[0].relative_to(ROOT).as_posix(),
                               source_pointer=f"{mode}|{model}|layer={int(value['layer'])}|{metric_key}")
                    x.append(layer)
                    y.append(r["mean"])
                ax.plot(x, y, color=color, ls=linestyle, lw=1.8 if metric == "logistic" else 1.45,
                        alpha=1 if metric == "logistic" else .9)
                expected_selected_value = float(selected[metric_key])
                assert abs(y[x.index(selected_layer)] - expected_selected_value) < 1e-12
                if metric == "logistic":
                    ax.plot(selected_layer, expected_selected_value, marker="o", ms=5,
                            color=color, lw=0, zorder=5)
        ax.axhline(.1, color=AXIS, ls=(0, (2, 2)), lw=.9)
        ax.set(xlim=(-.5, 43.5), xticks=[1, 10, 20, 30, 42], xlabel="Layer",
               ylim=(0, 1.08), yticks=[0, .5, 1], ylabel="Balanced accuracy")
        percent(ax)
fig.legend([Line2D([], [], color=INK, lw=1.8),
            Line2D([], [], color=INK, lw=1.45, ls=(0, (3.5, 2.2))),
            Line2D([], [], color=INK, marker="o", ms=5, lw=0),
            Line2D([], [], color=AXIS, lw=.9, ls=(0, (2, 2)))],
           ["Logistic", "NCC", "Discovery-selected layer", "Chance (10%)"],
           loc="center", bbox_to_anchor=(.5, .042), ncol=4, fontsize=SIZE["annotation"],
           columnspacing=1.2)
save(fig, stem, "Enumeration: held-out running and final count representations across layers")


# Preserve exact plotted values, source paths, statistical scope, and all remote
# hashes asserted by the sealed report. The latter are not claimed reverified.
(DATA / "plot_data.json").write_text(json.dumps(ROWS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
columns = sorted(set().union(*(r.keys() for r in ROWS)))
with (DATA / "plot_data.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(ROWS)
manifest = {
    "schema": "enumeration_publication_figures_v1", "source_status": PAYLOAD["status"],
    "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "verified_local_sha256": SOURCES,
    "sealed_report_asserted_source_sha256": PAYLOAD["source_sha256"],
    "answer_extension_artifact_hashes": {f"{c['prompt_mode']}|{c['model_label']}": c["artifact_hashes"]
                                         for c in PAYLOAD["answer_trace_extension"]["cells"]},
    "plot_rows": len(ROWS), "font_reference": FONT, "colors": dict(zip(MODELS, COLORS)),
    "statistical_policy": "Copy frozen point estimates and intervals. No re-fitting, no new model forward, no resampling. All 8 sampled answer layers retained. One-based display only.",
    "source_scope_note": "Hashes under sealed_report_asserted_source_sha256 preserve upstream provenance; only verified_local_sha256 were recomputed locally by this builder.",
    "relay_fraction_definition": "1 - abs(D_suffix / D_natural); CI = [1 - ratio_CI_high, 1 - ratio_CI_low]. Bullet-Gemma residual is negative and is retained via absolute value; do not equate this endpoint with specific mediation divided by natural damage.",
    "continuation_interval_note": "Paired-adoption intervals equal patched-rate intervals because self-target adoption is exactly zero in every registered trial; asserted per cell.",
    "representation_scope": {"cohort": "original_registered_all_sample_panel; no reserve replacements",
                             "running_common_discovery_rows": 1056, "running_common_confirmation_rows": 518,
                             "final_discovery_rows": 200, "final_confirmation_rows": 100,
                             "selected_layers": "copied from frozen selected CSVs, not reselected",
                             "upstream_candidate_and_selected_hashes_reverified": True,
                             "classifiers": "logistic and nearest-class-centroid (NCC)",
                             "preprocessing": "original 16-component whitened PCA; no refitting in this builder"},
    "render_checks": CHECKS, "elapsed_seconds": time.perf_counter() - START,
    "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"figures": list(CHECKS), "plot_rows": len(ROWS), "elapsed_seconds": manifest["elapsed_seconds"]}))
